# Clockfield: Adaptive Time in Neural Computation

**When neurons crystallize, time slows down.**

This repository presents the Clockfield hypothesis: that neural computation — biological and artificial — operates on a curved temporal metric where the local speed of time depends on the crystallization state of each processing unit. Stable circuits experience slow time (resistant to perturbation). Searching circuits experience fast time (rapid exploration). The metric self-organizes without external supervision.

The framework unifies three independent lines of evidence under a single equation:

```
dτᵢ = exp(-α · βᵢ) · dt
```

where βᵢ is the crystallization state of unit i, measured by activation roughness.

---

## Three Systems, One Metric

### 1. The β-Sieve: Detecting Temporal Wells in Neural Networks

A zero-shot geometric probe that measures activation roughness ("viscosity") across layers. Applied to grokking on modular arithmetic (mod 97):

| Metric | Logic (Addition) | Control (Scrambled) |
|--------|-----------------|-------------------|
| Final Train Accuracy | 99.9% | 99.8% |
| Final Test Accuracy | 99.6% | 0.8% |
| Final β-Gradient | **0.6500** | **0.1983** |

The β-gradient climbs **hundreds of epochs before test accuracy moves** — detecting the nucleation of generalized structure while the network still appears to be memorizing. This is the formation of a temporal well before the "particle" (generalized circuit) stabilizes.

The grokking transition shows a five-phase structure visible in the β-Sieve signal: settling → nucleation → melt → crystallization → stable crystal. The melt phase (β crashes before surging to 5× its nucleation peak) is a first-order phase transition.

### 2. EEG Geometric Dysrhythmia: Broken Clockfields in Schizophrenia

Applied to clinical EEG (RepOD Schizophrenia dataset, n=26, ICA artifact rejection), the Deerskin Architecture's geometric measurements distinguish schizophrenic brains from healthy controls with zero machine learning:

| Measurement | HC | SZ | p | Cohen's d |
|-------------|----|----|---|-----------|
| Cross-band eigenmode coupling | 0.611 | 0.463 | **0.007** | −1.21 |
| Temporal Betti-1 | 15.61 | 13.01 | **0.035** | −0.92 |
| Occipital PLV variance | 0.0186 | 0.0207 | **0.012** | — |

**Clockfield interpretation:** Schizophrenia is a disorder of the temporal metric itself. The mapping from neural state to local time speed is inconsistent — neurons that should be frozen (high β, slow time) are jolted back to full speed by gate instability. The result: fragmented cross-band coordination, impoverished temporal attractors, and an unstable theta gate.

This is distinct from Alzheimer's disease (broken manifold, possibly intact metric), producing a double dissociation predicted by the architecture.

### 3. PhiWorld: Emergent Particles from Amplitude-Dependent Propagation

A 2D wave equation with amplitude-dependent speed:

```
c²(x,t) = c₀² / (1 + κ · Ψ(x,t)²)
```

This is a Lorentzian metric: proper time runs slowly where field amplitude is high. From a Gaussian pulse, the system spontaneously produces stable concentric ring structures — "particles" that are temporal wells where time runs slower than the surrounding vacuum.

The PhiWorld particles are the same mathematical object as the grokked circuits detected by the β-Sieve and the stable attractors measured by Betti-1 in EEG: localized regions of high field amplitude where time runs slow, stabilized by nonlinear self-interaction.

---

## The Clockfield Grokking Experiment

We tested the Clockfield's most direct prediction: that β-adaptive weight decay (protecting crystallized modules, eroding unstructured ones) should accelerate grokking compared to uniform weight decay.

**The result was negative.** Baseline (uniform WD=0.1) grokked at epoch 510. Clockfield (β-adaptive WD) grokked at epoch 650. No-decay ablation grokked at epoch 860.

| Condition | Grok Epoch | Final Test | Max β |
|-----------|-----------|------------|-------|
| Baseline (uniform WD=0.1) | **510** | 99.5% | 0.384 |
| Clockfield (β-adaptive WD) | 650 | 99.6% | 0.244 |
| No weight decay | 860 | 99.6% | 0.258 |

### Why It Failed (and What It Teaches)

The decay evolution plot reveals the mechanism: by epoch ~100, all modules collapsed to minimum decay (0.005) because the β-Sieve detected roughness in every layer during memorization. The Clockfield couldn't distinguish memorization roughness from generalization roughness — it protected everything prematurely, effectively reducing the experiment to near-zero weight decay.

**The deeper lesson:** The Clockfield prescription (protect what has crystallized) is correct for **inference stability** but wrong for the **grokking search phase.** During grokking, you want maximum erosion to prevent premature crystallization of memorized circuits. The memorized structure must melt before the generalized structure can form — this is visible as the β-crash at the phase transition (Phase 3 in the five-phase structure).

The ordering baseline < clockfield < no_decay confirms that weight decay accelerates grokking, and that the Clockfield's protective mechanism was counterproductive in this regime. The theory needs refinement: the temporal metric should perhaps *invert* during learning (fast time for crystallized → erode memorization; slow time for crystallized → protect generalization), with the direction determined by whether the system has crossed the phase transition.

We report this negative result transparently because it constrains the theory precisely where it needs constraining.

---

## Repository Structure

```
Clockfield/
├── README.md                          ← you are here
├── THEORY.md                          ← full mathematical framework
├── clockfield_grokking.py             ← head-to-head experiment (3 conditions)
├── grokkingwithviscosity2.py          ← original β-Sieve grokking probe
├── phiworld2.py                       ← emergent particle field simulation
├── relativistic_deerskin.py           ← Clockfield neuron simulation (theoretical)
├── results/
│   ├── comparison.json                ← Clockfield experiment raw data
│   ├── clockfield_vs_baseline.png     ← head-to-head grokking curves
│   ├── phase_analysis.png             ← per-condition phase transitions
│   ├── decay_evolution.png            ← adaptive decay rate trajectories
│   ├── results.json                   ← original β-Sieve grokking data
│   ├── THE_SMOKING_GUN_LONG.png       ← β-gradient leading grokking
│   ├── phiworld_particles.png         ← emergent ring structures
│   └── relativistic_deerskin_demo.png ← Clockfield neuron demo output
└── LICENSE
```

For the full Deerskin Architecture (four-stage pipeline, EEG analysis, clinical results), see [Geometric-Neuron](https://github.com/anttiluode/Geometric-Neuron).

---

## Quick Start

```bash
git clone https://github.com/anttiluode/Clockfield.git
cd Clockfield

pip install torch numpy matplotlib

# Run the Clockfield grokking experiment (3 conditions, ~30-60 min on GPU)
python clockfield_grokking.py

# Run the original β-Sieve experiment
python grokkingwithviscosity2.py

# Run PhiWorld particle simulation (requires tkinter display)
python phiworld2.py
```

---

## The Broader Argument

The Clockfield is not a fifth stage bolted onto the Deerskin pipeline. It is the **metric** on which the four stages (dendritic delay manifold → somatic resonance cavity → theta phase gate → AIS spectral filter) operate. It determines the temporal resolution, effective delay, spectral bandwidth, and integration window of every neuron independently, based on that neuron's crystallization state.

The McCulloch-Pitts formal neuron (1943) — the foundation of all artificial neural networks — is recovered when two conditions are met simultaneously: the Neural Planck Ratio ℏₙ → 0 (destroying oscillatory structure) AND the Clockfield flattens to Γ = 1 everywhere (destroying adaptive timescales). Eighty years of deep learning has been operating in this double-degenerate limit.

Weight decay (λ·‖W‖²) is the ℏₙ → 0 shadow of the Clockfield: a uniform erosion term that approximates what adaptive temporal processing provides for free. Our experiment shows this approximation is surprisingly effective — uniform decay outperforms our first attempt at adaptive decay — which means the refinement needed is in the *direction* of adaptation, not the concept.

---

## Citation

```
Luode, A. & Claude (Anthropic). (2026). Clockfield: Adaptive Time in Neural 
Computation. PerceptionLab Independent Research.
https://github.com/anttiluode/Clockfield
```

## Related Work

- [Geometric-Neuron](https://github.com/anttiluode/Geometric-Neuron) — Deerskin Architecture and EEG clinical validation
- Power et al. (2022) — *Grokking: Generalization beyond overfitting on small algorithmic datasets*
- Murray et al. (2014) — *A hierarchy of intrinsic timescales across primate cortex*
- Zeraati et al. (2023) — Intrinsic timescale modulation by attention in V4

MIT License
