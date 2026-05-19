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

## Key Results (76 Phases across 12 Seasons)

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

> A complete summary of all 76 phases with detailed metrics is available in the paper's Appendix (Table 2).

## Project Structure

```
experiments/    # Phase scripts (P1-P76)
results/        # JSON output from all phases
figures/        # Visualization PNGs
papers/         # LaTeX source and PDF (v1, v2)
runner.py       # Orchestrator (GPU/CPU parallel)
```

## Running

```bash
# Full experiment suite
python runner.py

# Individual phase
python experiments/phase9_register_map.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 5.0+
- Qwen2.5-0.5B (auto-downloaded on first run)

## Based on

- [Aletheia](https://github.com/hafufu-stack/aletheia) — Neural Von Neumann Machine discovery
- Qwen2.5-0.5B with embedding surgery

## Author

**Hiroto Funasaki** — Independent Researcher, Japan

[![ORCID](https://img.shields.io/badge/ORCID-0009--0004--2517--0177-green?logo=orcid)](https://orcid.org/0009-0004-2517-0177)

## License

MIT
