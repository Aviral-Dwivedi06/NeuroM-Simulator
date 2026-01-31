import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# CONFIGURATION
# ============================================================

class NeuroConfig:
    def __init__(self):
        self.NUM_NEURONS = 32
        self.FANOUT_MODE = 1          # 0 = full, 1 = half
        self.VAULTS = 4

        # Neuron model
        self.THRESHOLD = 10.0
        self.LEAK_RATE = 0.05
        self.GATE_VDIST = 2.0

        # Weight precision
        self.RAW_WEIGHT_BITS = 4
        self.EFF_WEIGHT_BITS = 3

        # STDP
        self.STDP_SHIFT = 2
        self.STDP_T1 = 5

        # Homeostasis
        self.HOMEOSTASIS_RATE = 0.001
        self.TARGET_RATE = 0.10


# ============================================================
# NEUROSEQUENCE FSM (Memory-Centric Control)
# ============================================================

class NeuroSeqState:
    IDLE = 0
    READ = 1
    ACCUM = 2
    UPDATE = 3
    LEARN = 4


# ============================================================
# SYNAPTIC MEMORY BANK
# ============================================================

class SynapticMemoryBank:
    def __init__(self, rows, cols, cfg):
        self.cfg = cfg

        self.even_bank = np.random.randint(
            0, 2**cfg.RAW_WEIGHT_BITS, (rows, cols // 2)
        )
        self.odd_bank = np.random.randint(
            0, 2**cfg.RAW_WEIGHT_BITS, (rows, cols // 2)
        )

        self.active_bitmap = np.random.choice([0, 1], rows, p=[0.2, 0.8])
        self.delta_accumulator = np.zeros((rows, cols))

        self.idle_cycles = 0
        self.zero_spike_skips = 0
        self.zero_weight_skips = 0

    def read_weight(self, row, col):
        if self.active_bitmap[row] == 0:
            return 0

        bank = self.odd_bank if (col & 1) else self.even_bank
        raw = bank[row, col >> 1]
        mask = (1 << self.cfg.EFF_WEIGHT_BITS) - 1
        return raw & mask


# ============================================================
# NEURON UNIT
# ============================================================

class NeuronUnit:
    def __init__(self, neuron_id, cfg):
        self.id = neuron_id
        self.cfg = cfg

        self.v_mem = 0.0
        self.last_spike_time = -100
        self.is_gated = False

        # Weight cache (hardware-inspired)
        self.cache = {}
        self.cache_order = deque(maxlen=4)
        self.cache_hits = 0
        self.cache_misses = 0

        # Probes
        self.v_mem_probe = []
        self.syn_current_probe = []
        self.spike_train = []

    def get_weight_cached(self, mem, row, col):
        key = (row, col)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        self.cache_misses += 1
        w = mem.read_weight(row, col)
        self.cache[key] = w
        self.cache_order.append(key)
        if len(self.cache_order) == self.cache_order.maxlen:
            self.cache.pop(self.cache_order[0], None)
        return w

    def update(self, syn_sum, t):
        self.syn_current_probe.append(syn_sum)

        # Clock-gating
        if syn_sum == 0 and abs(self.v_mem - self.cfg.THRESHOLD) > self.cfg.GATE_VDIST:
            self.is_gated = True
            self.v_mem_probe.append(self.v_mem)
            return False

        self.is_gated = False
        self.v_mem = self.v_mem * (1 - self.cfg.LEAK_RATE) + syn_sum
        self.v_mem_probe.append(self.v_mem)

        if self.v_mem >= self.cfg.THRESHOLD:
            self.v_mem = 0.0
            self.last_spike_time = t
            self.spike_train.append(t)
            return True

        return False


# ============================================================
# NEUROMORPHIC VAULT
# ============================================================

class NeuromorphicVault:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mem = SynapticMemoryBank(cfg.NUM_NEURONS, cfg.NUM_NEURONS, cfg)
        self.neurons = [NeuronUnit(i, cfg) for i in range(cfg.NUM_NEURONS)]

        self.seq_state = NeuroSeqState.IDLE
        self.seq_trace = []

        self.metrics = {
            "cycles": 0,
            "learning_cycles": 0,
            "inference_cycles": 0,
            "gated_neurons": 0
        }

    def stdp_kernel(self, dt):
        if dt < self.cfg.STDP_T1:
            return 0.2 * dt + 0.1
        return 0.0

    def process_timestep(self, t, spike_vec):
        if not spike_vec.any():
            self.seq_state = NeuroSeqState.IDLE
            self.mem.idle_cycles += 1
            self.seq_trace.append(self.seq_state)
            self.metrics["cycles"] += 1
            return

        self.seq_state = NeuroSeqState.READ
        self.seq_trace.append(self.seq_state)

        fanout = spike_vec if self.cfg.FANOUT_MODE == 0 else spike_vec[:self.cfg.NUM_NEURONS // 2]
        spike_idx = np.where(fanout > 0)[0]

        fired = []

        for n in self.neurons:
            syn_sum = 0
            self.seq_state = NeuroSeqState.ACCUM

            for s in spike_idx:
                if spike_vec[s] == 0:
                    self.mem.zero_spike_skips += 1
                    continue

                w = n.get_weight_cached(self.mem, n.id, s)
                if w == 0:
                    self.mem.zero_weight_skips += 1
                    continue

                syn_sum += w

            self.seq_state = NeuroSeqState.UPDATE
            if n.update(syn_sum, t):
                fired.append(n.id)

            if n.is_gated:
                self.metrics["gated_neurons"] += 1

        if fired:
            self.seq_state = NeuroSeqState.LEARN
            self.metrics["learning_cycles"] += 1

            for n_id in fired:
                for s in spike_idx:
                    dt = (t >> self.cfg.STDP_SHIFT)
                    self.mem.delta_accumulator[n_id, s] += self.stdp_kernel(dt)

            # Homeostatic plasticity
            firing_rate = len(fired) / self.cfg.NUM_NEURONS
            for n in self.neurons:
                self.mem.delta_accumulator[n.id, :] += (
                    self.cfg.HOMEOSTASIS_RATE *
                    (self.cfg.TARGET_RATE - firing_rate)
                )

        else:
            self.metrics["inference_cycles"] += 1

        self.seq_trace.append(self.seq_state)
        self.metrics["cycles"] += 1


# ============================================================
# REPORTING
# ============================================================

def print_hardware_report(vault, cfg):
    neurons = vault.neurons
    total_hits = sum(n.cache_hits for n in neurons)
    total_misses = sum(n.cache_misses for n in neurons)
    total_accesses = total_hits + total_misses
    hit_rate = 100 * total_hits / total_accesses if total_accesses else 0

    total_ops = vault.metrics["cycles"] * cfg.NUM_NEURONS
    effective_ops = (
        total_ops
        - vault.mem.zero_spike_skips
        - vault.mem.zero_weight_skips
    )

    print("\n================ Hardware Performance Report ================")
    print(f"Total Cycles               : {vault.metrics['cycles']}")
    print(f"Inference Cycles           : {vault.metrics['inference_cycles']}")
    print(f"Learning Cycles            : {vault.metrics['learning_cycles']}")
    print(f"Clock-Gated Neurons        : {vault.metrics['gated_neurons']}")
    print(f"SRAM Idle Cycles           : {vault.mem.idle_cycles}")
    print(f"Zero-Spike Skips           : {vault.mem.zero_spike_skips}")
    print(f"Zero-Weight Skips          : {vault.mem.zero_weight_skips}")
    print(f"Weight Cache Hit Rate      : {hit_rate:.2f} %")
    print(f"Effective Utilization      : {100*effective_ops/total_ops:.2f} %")
    print(f"Pending Weight Updates     : {np.sum(vault.mem.delta_accumulator):.2f}")
    print("=============================================================")


# ============================================================
# EXECUTION
# ============================================================

def run_simulation():
    cfg = NeuroConfig()
    vault = NeuromorphicVault(cfg)

    T = 60
    spike_stream = (np.random.rand(T, cfg.NUM_NEURONS) > 0.8).astype(int)

    print(f"\nStarting Advanced NeuroX-Sim ({cfg.VAULTS} Vault Architecture)")

    for t in range(T):
        vault.process_timestep(t, spike_stream[t])

    # -------- Plot 1: Membrane Potential --------
    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.plot(vault.neurons[i].v_mem_probe, label=f"Neuron {i}")
    plt.axhline(cfg.THRESHOLD, linestyle="--", label="Threshold")
    plt.title("Membrane Potential Dynamics")
    plt.xlabel("Cycle")
    plt.ylabel("V_mem")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # -------- Plot 2: Synaptic Current --------
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.plot(vault.neurons[i].syn_current_probe, label=f"Neuron {i}")
    plt.title("Synaptic Current")
    plt.xlabel("Cycle")
    plt.ylabel("I_syn")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # -------- Plot 3: NeuroSequence FSM --------
    plt.figure(figsize=(12, 3))
    plt.plot(vault.seq_trace, drawstyle="steps-post")
    plt.yticks(
        [0, 1, 2, 3, 4],
        ["IDLE", "READ", "ACCUM", "UPDATE", "LEARN"]
    )
    plt.title("Memory-Centric NeuroSequence FSM")
    plt.xlabel("Cycle")
    plt.grid(alpha=0.3)
    plt.show()

    # -------- Plot 4: Spike Raster --------
    plt.figure(figsize=(12, 5))
    for i, n in enumerate(vault.neurons[:16]):
        plt.scatter(n.spike_train, [i] * len(n.spike_train), s=6)
    plt.title("Spike Raster (Reservoir Dynamics)")
    plt.xlabel("Time")
    plt.ylabel("Neuron ID")
    plt.grid(alpha=0.3)
    plt.show()

    # -------- Plot 5: Weight Update Distribution --------
    plt.figure(figsize=(6, 4))
    plt.hist(vault.mem.delta_accumulator.flatten(), bins=40)
    plt.title("Synaptic Update Distribution")
    plt.xlabel("ΔWeight")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.show()

    print_hardware_report(vault, cfg)


if __name__ == "__main__":
    run_simulation()
