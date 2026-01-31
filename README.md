## README.md

```markdown
# NeuroX-Sim: Memory-Centric Neuromorphic Architecture Simulator

NeuroX-Sim is a **cycle-aware, memory-centric digital neuromorphic architecture simulator** designed to study event-driven spiking neural networks (SNNs) from a **hardware-first perspective**.

The simulator is inspired by modern neuromorphic processors such as **Neurocube**, **LSMCore**, and other memory-centric neuromorphic systems, focusing on **co-located memory and computation**, **sparsity exploitation**, and **hardware-observable metrics**.

---

## Key Features

### Memory-Centric Execution
- Explicit **NeuroSequence FSM** (IDLE, READ, ACCUMULATE, UPDATE, LEARN)
- Event-driven control flow instead of clock-driven neuron updates
- Models **memory-driven state machines** used in real neuromorphic hardware

### Event-Driven & Sparse Computation
- Zero-spike skipping
- Zero-weight skipping
- Clock-gated neuron updates
- Dynamic sparsity exploitation for energy-efficient execution

### Digital Neuron Model
- Leaky Integrate-and-Fire (LIF) neurons
- Threshold-based spiking
- Leakage-dominated clock gating

### Learning & Plasticity
- Spike-Timing-Dependent Plasticity (STDP)
- Multi-timescale learning
- Slow **homeostatic plasticity** to stabilize firing rates
- Accumulated synaptic updates (write-buffer style)

### Hardware-Inspired Memory System
- Even/Odd synaptic banks
- Effective weight masking (precision reduction)
- Active bitmap for synapse enable/disable
- Small per-neuron weight cache with hit/miss statistics

### Reservoir / LSM-Style Dynamics
- Recurrent spiking activity
- Spike raster visualization
- Suitable for studying liquid-state behavior

---

## Observability & Outputs

The simulator provides **hardware-meaningful observability**, including:

### Plots
- Membrane potential evolution
- Synaptic current traces
- NeuroSequence FSM timeline
- Spike raster (reservoir dynamics)
- Synaptic update distribution

### Console Metrics
- Total cycles
- Inference vs learning cycles
- Clock-gated neuron count
- SRAM idle cycles
- Zero-spike and zero-weight skips
- Cache hit rate
- Effective hardware utilization
- Pending synaptic updates

These metrics mirror those reported in **neuromorphic ASIC papers**, enabling architectural comparison and analysis.

---

## 🏗 Project Structure

```

Neuromorphic_Framework/
│
├── neurosim.py          # Main neuromorphic simulator
├── README.md            # Project documentation
├── requirements.txt    # Python dependencies
└── .gitignore           # Git ignore rules

````

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Run the simulator

```bash
python neurosim.py
```

The simulator will:

* Execute the neuromorphic model
* Display plots for neural and architectural behavior
* Print a detailed hardware performance report

---

## Intended Use Cases

* Neuromorphic architecture exploration
* Memory-centric computing research
* Studying sparsity and event-driven execution
* Educational tool for digital neuromorphic systems
* Pre-RTL architectural validation

---

##  Conceptual Inspiration

This project is inspired by:

* Memory-centric neuromorphic processors
* Digital spiking neural network accelerators
* Liquid State Machines (LSM)
* STDP-based on-chip learning architectures

The simulator does **not** aim to be a biological simulator, but rather a **hardware-faithful architectural model**.

---

## Future Extensions

Planned or possible extensions include:

* ADC/DAC quantization effects
* Memristor non-linearity models
* Vault-to-vault interconnect contention
* Energy estimation models
* RTL correlation hooks
* Dataset-driven spike injection

---


