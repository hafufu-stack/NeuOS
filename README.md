# NeuOS: Discovering and Exploiting the Neural Von Neumann Architecture Inside Pre-Trained Language Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20262991.svg)](https://doi.org/10.5281/zenodo.20262991)

> Transformers are not black boxes. They are **computers**.

NeuOS is a systematic investigation revealing that pre-trained transformer language models internally implement a Von Neumann-like computational architecture with identifiable registers, swappable memory, executable programs, and a decompilable instruction set.

## Paper

**[NeuOS: Discovering and Exploiting the Neural Von Neumann Architecture Inside Pre-Trained Language Models](https://doi.org/10.5281/zenodo.20262991)**

## Key Results (26 Phases)

### Neural CPU Register Map (ISA)

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

### Breakthrough Results

| Discovery | Phase | Result |
|-----------|-------|--------|
| **Self-Healing OS** | P14 | 100% recovery from hardware damage via dynamic register reallocation |
| **ISA Universality** | P15 | Register positions stable across prompt formats (+/-2 layers) |
| **Register Transfer** | P20 | 90% behavioral takeover by transplanting L16 (MIN) register |
| **DMA Execution** | P22 | 66.7% accuracy executing programs without any text instructions |
| **Neural fork()** | P23 | 100% differentiation: 1 input -> 3 different programs |
| **Neural Decompiler** | P26 | 100% program identification from register state alone |

### All Phases

| Phase | Name | Key Finding |
|-------|------|-------------|
| P1 | Sorting Register | MIN=L16 (100%), MAX=L18 (99%) |
| P2 | Conditional Branch | Comparison=L15 (99%) |
| P3 | Write Optimization | Best single layer: L22 (50%) |
| P4 | Graceful Degradation | Catastrophic: 10% dropout destroys computation |
| P5 | Thermodynamic Scheduling | Thermodynamic scheduling beats round-robin |
| P6 | Blackbox Device Probing | Linear R2=0.98 |
| P7 | Instruction Taxonomy | 8-class OPCODE classification |
| P8 | Multi-Program Execution | 0% (register collision) |
| **P9** | **Register Map** | **Unified ISA: 9 registers L0-L22** |
| P10 | Chained Computation | Intermediate sum exists (57%) but chain fails |
| P11 | Neural Clock | 0% (embedding layer overwrites) |
| P12 | Context Switch | 0% (hidden state restoration fails) |
| P13 | Wetware Hypervisor | 0% (Hill muscle equilibrium) |
| **P14** | **Self-Healing** | **100% recovery via register reallocation** |
| **P15** | **ISA Universality** | **Format-invariant register positions** |
| P16 | Neural Executable | 0% (tokenization boundary mismatch) |
| **P17** | **KV-Cache Paging** | **Baseline-matching multitasking** |
| P18 | Cache Clock | KV continuation works (limited by base accuracy) |
| P19 | Symbiotic Polymorphism | 0% (0.5B model too small for control) |
| **P20** | **Register Transfer** | **90% takeover; L0 cos_sim=0.99 (universal parser)** |
| **P22** | **DMA Execution** | **66.7% without text instructions** |
| **P23** | **Neural fork()** | **75% MIN accuracy, 100% differentiation** |
| **P24** | **Execution Port** | **Optimal injection at L5-L8, not native layer** |
| P25 | Register Algebra | MIN-MAX cos_sim=0.981; linear synthesis fails |
| **P26** | **Neural Decompiler** | **100% program identification (5 operations)** |

## Project Structure

```
experiments/    # Phase scripts (P1-P26)
results/        # JSON output from all phases
figures/        # Visualization PNGs
papers/         # LaTeX source and PDF
runner.py       # Orchestrator v5 (GPU/CPU parallel)
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

- [Aletheia](https://github.com/hafufu-stack/aletheia) - Neural Von Neumann Machine discovery
- Qwen2.5-0.5B with embedding surgery

## Author

**Hiroto Funasaki** — Independent Researcher, Japan

[![ORCID](https://img.shields.io/badge/ORCID-0009--0004--2517--0177-green?logo=orcid)](https://orcid.org/0009-0004-2517-0177)

## License

MIT
