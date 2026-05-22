# NeuOS: Discovering and Exploiting the Neural Von Neumann Architecture Inside Pre-Trained Language Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20262991.svg)](https://doi.org/10.5281/zenodo.20262991)

> Transformers are not black boxes. They are **computers**.

NeuOS is a systematic investigation revealing that pre-trained transformer language models internally implement a Von Neumann-like computational architecture with identifiable registers, swappable memory, executable programs, and a decompilable instruction set.

## Paper

**[NeuOS: Discovering and Exploiting the Neural Von Neumann Architecture Inside Pre-Trained Language Models](https://doi.org/10.5281/zenodo.20262991)**

## Neural CPU Register Map (ISA)

All 24 layers of Qwen2.5-0.5B were probed to identify a complete 9-register Instruction Set Architecture:

| Register | Layer | Accuracy | Pipeline Stage |
|----------|-------|----------|----------------|
| **OPCODE** | L0 | **100%** | Instruction Decode |
| **Operand B** | L2 | **100%** | Operand Fetch |
| **CARRY** | L4 | **82%** | Status Flag |
| **Operand A** | L13 | **96%** | Operand Fetch |
| **COMPARISON** | L14 | **74%** | Branch Control |
| **MIN** | L16 | **100%** | Sort Execution |
| **MEDIAN** | L18 | **98%** | Sort Execution |
| **SUM** | L20 | **78%** | ALU |
| **MAX** | L22 | **100%** | Output Register |

## Key Results (170 Phases across 24 Seasons)

### Season 1–4: The Neural Computer (P1–P26)

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Unified ISA** | P9 | 9 registers identified across 24 layers |
| **Self-Healing OS** | P14 | 100% recovery from hardware damage via register reallocation |
| **Register Transfer** | P20 | 90% behavioral takeover by transplanting MIN register |
| **DMA Execution** | P22 | 66.7% accuracy executing programs without text instructions |
| **Neural fork()** | P23 | 100% differentiation: 1 input → 3 different programs |
| **Neural Decompiler** | P26 | 100% program identification from register state alone |

### Season 5–7: Autopoiesis and Polymorphism (P27–P55)

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Cross-Architecture Translation** | P39 | Register vectors translated between 0.5B and 1.5B models |
| **Autopoietic Kernel** | P50 | 100% accuracy via self-compilation with zero external labels |
| **Topological Proprioception** | P54 | 90.5% anomaly detection from topology features |
| **Polymorphic Hot Swapping** | P55 | 60% accuracy maintained across 15 hot-swaps of 5 variants |

### Season 8–9: Artificial Life (P56–P70)

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Neural Quine** | P58 | Self-replicating program vector (self-sim 0.70 vs control -0.001) |
| **Neural Genetic Algorithm** | P59 | Gradient-free evolution: 83.3% test accuracy |
| **Program Compression** | P64 | 896-dim vector compressed to 10 dims, 100% accuracy retained |
| **Cambrian Explosion** | P70 | 19 unique phenotypes emerge over 20 generations |

### Season 10–12: The Opus (P71–P76)

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Multicellular Organism** | P71 | MIN+MAX cells cooperate for RANGE: 80% accuracy |
| **Developmental Metamorphosis** | P72 | Reversible FIRST→MIN→MAX program differentiation |
| **Neural Parasitism** | P73 | Parasite vector destroys host MAX function (100%→0%) |
| **Aging and Rejuvenation** | P76 | Retraining restores aged program (40%→80%); SVD pruning fails |

### Season 13–14: Embodied & Social Intelligence (P77–P106)

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Sensorimotor Integration** | P77 | Sensor-motor loop in activation space |
| **Reincarnation** | P78 | Soul transfer to new body |
| **Soul Algebra** | P97 | Vector arithmetic on soul vectors |
| **Collective Intelligence** | P101 | Swarm computation from multiple agents |
| **Language Emergence** | P106 | Spontaneous communication between programs |

### Season 15: Soul Vectors and the Rosetta Stone (P107–P114)

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Functional Equivalence** | P107 | Independent MIN vectors are orthogonal (cos ≈ 0.005) yet all achieve ~85% |
| **Multi-Language Rosetta** | P108 | Linear translation between 3 "languages" at **100% accuracy**; rank-1 SVD |
| **Platonic Form** | P113 | Equivalence classes are ~60-dim **manifolds**, not clusters |
| **Rosetta Algebra** | P114 | Translation matrices form a **non-commutative group**; σ₁/σ₂ ≈ 80 |

### Season 16: Limits and Boundaries (P115–P121)

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Composition Recovery** | P115 | Kernel composition: 75% (vs. 25% pipeline); generalization weak |
| **Convergent Evolution** | P116 | 4 methods → same function, orthogonal vectors: "all roads lead to Rome" |
| **Soul Compression** | P119 | 64 dims sufficient (~7% of 896); "soul is a single number" **rejected** |
| **Cross-Model Failure** | P120 | Translation fails across model sizes (15% acc): **soul is body-specific** |
| **Arms Race** | P121 | All backdoors achieve 100% deception; SVD entropy detects all (1.09 vs 0.23) |

### Season 17–21: Cross-Project Unification (P122–P145)

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Rosetta Soul Compiler** | P122 | Text→soul compilation with cos = 1.0 (exact match) |
| **Aletheia Firmware** | P124 | L8 is optimal injection point (90%, +10pp over L16) |
| **Holographic Soul** | P131 | Direction determines function; 0.5–2× scaling preserves accuracy |
| **Thermodynamic Autopoiesis** | P136 | Entropy-gated noise: +15pp over no-noise baseline |
| **Data Scaling Laws** | P138 | MIN/MAX saturate at n=10; ADD/SUB require n=35 (3.5× gap) |
| **Dual Execution Pathways** | P143 | **L6 = arithmetic (ADD/SUB), L8 = comparison (MIN/MAX)** |
| **Temperature Robustness** | P144 | Stable T=0–1.5; phase transition collapse at T≥2.0 |
| **Format Invariance** | P145 | 8/12 prompt formats ≥60%; `min()` format = **100%** |

### Season 22: GlassBox Dashboard (P146–P152) 🆕

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Hardware Proprioception** | P146 | AI self-diagnoses its own parameters (896d, 24L) with **100% accuracy** |
| **First-Person Decompiler** | P147 | Real-time self-report of running program (MIN/MAX): **100%** |
| **Scaling Oracle** | P148 | Self-predicts capacity limits: **100% accuracy** |
| **GlassBox Dashboard** | P151 | All self-diagnosis in a **single inference pass** — full white-box AI |
| **Soul Immune System** | P152 | Corrupted soul detection + unsupervised self-repair: **+12pp** |

### Season 23: The Homoiconic Mind (P153–P164) 🆕

| Discovery | Phase | Result |
|-----------|-------|--------|
| **NL→Soul Compilation** | P154 | "pick the smaller one" → soul seed → 3 examples: **0%→40%** |
| **Pipeline Rewiring** | P155 | All 24 layers essential — no skip tolerance |
| **Skill Discovery** | P157 | Novel task detection: **perfect**; meta-cognition works |
| **Soul Algebra** | P160 | MIN/MAX cosine=0.07 (orthogonal); smooth interpolation |
| **7D Soul Compression** | P161 | 896D → **7D with zero accuracy loss** (128× compression) 🏆 |
| **One-Shot Cloning** | P162 | Warm start 1-shot: **60%** (3× faster than cold start) |
| **Soul Cartography** | P163 | 4 primitives (MIN/MAX/FIRST/SECOND) equally distributed in soul space |
| **Universal Instruction Set** | P164 | PCA basis vectors decoded as **assembly opcodes** 🏆 |

### Season 24: The 7D Rosetta Engine (P165–P170) 🆕

| Discovery | Phase | Result |
|-----------|-------|--------|
| **7D Semantic Firewall** | P165 | 7D projection preserves **100% accuracy**; +20pp noise protection |
| **Rosetta Compiler** | P166 | Coords `[0,1.5,0,0,0,1.5,0,0]` = **MAX 100%** — zero gradient, zero data 🏆 |
| **Zero-Shot Alchemy** | P167 | MIN×2 overclock = **88%**; single-axis programming = **75%** |
| **Control Room** | P168 | 7D slider dashboard with 8 presets visualized |
| **7D Grid Search** | P169 | 169-point search: **MAX=100%** without any gradient descent |
| **Soul Persistence** | P170 | Save/load soul jar (31KB): cosine fidelity = **1.000000** |

> A complete summary of all 170 phases with detailed metrics is available in the paper's Appendix.

## Project Structure

```
experiments/    # Phase scripts (P1–P99)
experiments2/   # Phase scripts (P100–P170)
results/        # JSON output from all phases
figures/        # Visualization PNGs (170 figures)
papers/         # LaTeX source and PDF (v1–v5)
runner.py       # Orchestrator (GPU/CPU parallel)
```

## Running

```bash
# Full experiment suite
python runner.py

# Individual phase
python experiments/phase9_register_map.py

# Later phases
python experiments2/phase161_soul_compression.py

# Season 24 (P165–P170)
python experiments2/run_season24.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 5.0+
- Qwen2.5-0.5B (auto-downloaded on first run)
- Qwen2.5-1.5B (required for Phase 120 cross-model experiments only)

## Based on

- [Aletheia](https://github.com/hafufu-stack/aletheia) — Neural Von Neumann Machine discovery
- Qwen2.5-0.5B with embedding surgery

## Author

**Hiroto Funasaki** — Independent Researcher, Japan

[![ORCID](https://img.shields.io/badge/ORCID-0009--0004--2517--0177-green?logo=orcid)](https://orcid.org/0009-0004-2517-0177)

## License

MIT
