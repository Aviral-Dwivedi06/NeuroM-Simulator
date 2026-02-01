
# NeuroM-Sim 
## (Neuromorphic Memory Centric Simulator)

**Developed by Team Morpheus for the Micron Mimory Competition**

## Project Overview

**NeuroM-Sim** is a cycle-accurate hardware simulation framework designed to evaluate the efficiency of memory-centric neuromorphic architectures. Unlike standard software neural networks, this framework simulates the low-level hardware interactions between **Synaptic Memory Banks** and **Neuron Units**, specifically focusing on how memory bandwidth and organization (Vaults) impact spiking neural network (SNN) performance.

This project addresses the "Memory Wall" by implementing hardware-inspired features like weight caching, zero-spike skipping, and clock-gating, aligning with Micron's focus on high-performance, high-density memory solutions.

This framework serves as a demonstration of **Memory-Centric Neuromorphic Computing**. By embedding state-machines within vault controllers (as seen in architectures like Neurocube), we demonstrate how to minimize data movement and maximize the efficiency of the memory tier for AI workloads.

---

## Key Hardware Features

* **Vault-Based Architecture:** Simulates a multi-vault memory organization where each vault manages its own synaptic memory and neuron clusters.
* **NeuroSequence FSM:** A custom Finite State Machine (IDLE -> READ -> ACCUM -> UPDATE -> LEARN) that governs the memory-centric control flow.
* **SRAM Optimization:** * **Weight Caching:** Hardware-inspired cache to reduce redundant memory reads.
* **Zero-Skipping:** Logic to bypass memory fetches for zero-value spikes or weights, saving dynamic power.
* **Clock-Gating:** Disables neuron membrane updates when input activity is below a threshold.


* **Synaptic Plasticity:** On-chip learning via Spike-Timing-Dependent Plasticity (STDP) and Homeostatic rate control.

---

## Project Structure

```text
Neuromorphic_Framework/
├── main.py                 # Core simulation engine, FSM, and logic
├── README.md               # Project documentation
└── requirements.txt        # numpy, matplotlib

```

---

## Technical Specifications

* **Neuron Model:** Leaky Integrate-and-Fire (LIF) with gated execution.
* **Memory Structure:** Split-bank (Even/Odd) synaptic storage to simulate interleaved memory access.
* **Precision:** Configurable bit-width (default 4-bit raw, 3-bit effective) for energy-efficient weight storage.
* **Learning:** Dual-mode plasticity (STDP for correlation, Homeostasis for stability).

---

## How to Run

1. **Environment Setup**:
```bash
pip install numpy matplotlib

```


2. **Execute Simulation**:
```bash
python main.py

```


3. **Analyze Hardware Reports**:
Upon completion, the framework generates a detailed **Hardware Performance Report** including:
* Weight Cache Hit Rate
* Effective Hardware Utilization %
* Clock-Gating Efficiency
* Zero-Spike/Weight Skip counts



---

## Visualization Suite

The framework automatically generates five diagnostic plots to analyze the system:

1. **Membrane Potential Dynamics**: Tracks the V_mem of individual hardware neurons.
2. **Synaptic Current**: Visualizes the input load on the processing tier.
3. **NeuroSequence FSM**: Displays the real-time state transitions of the memory controller.
4. **Spike Raster**: Maps the spiking activity (Reservoir Dynamics) over time.
5. **Synaptic Update Distribution**: Shows the histogram of weight changes (ΔWeight) pending for the next memory write-back.

---

To observe higher hardware utilization, modify the `NeuroConfig` class in `main.py` to adjust the `LEAK_RATE` or increase the input spike density in the `run_simulation` function.

## Future Plans & Scalability

HBM3/4 Integration: Future iterations will include a specialized simulation layer for High Bandwidth Memory (HBM) stacks, modeling the through-silicon via (TSV) latency and thermal constraints of 3D-stacked synaptic banks.

Asynchronous Processing Elements (PE): Moving from a centralized FSM to a distributed, "Near-Data" processing model where logic is embedded directly within the logic base of a 3D-DRAM stack.

Crossbar Memristive Arrays: Integration of RRAM/PCM models to simulate non-volatile synaptic weights, allowing for "Instant-On" neuromorphic states and near-zero leakage power.

Large-Scale Topology (Mesh-of-Vaults): Scalability testing for Network-on-Chip (NoC) architectures where thousands of vaults communicate via an asynchronous packet-switched mesh.

## Conclusion
The NeuroM-Sim framework by Team Morpheus addresses a significant bottleneck in modern architectures: the Von Neumann Paradox. By shifting the focus from "Processing-Centric" to "Memory-Centric" design, we have demonstrated a system where the memory bank is no longer a passive storage unit but an active participant in the neural computation.

For the Micron Mimory Competition, this simulator provides a quantitative look at how specialized hardware features—such as weight caching, vault-local FSMs, and zero-skipping—can drastically reduce the energy-per-synaptic-operation. Our results indicate that through aggressive memory-tier optimization, neuromorphic hardware can achieve the high-density, low-power targets required for the next generation of ubiquitous edge intelligence.
