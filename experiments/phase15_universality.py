# -*- coding: utf-8 -*-
"""
Phase 15: ISA Universality Test (Opus Original)
Is the register layout universal across different architectures?

Compare the normalized register positions (layer/total_layers)
between Qwen-0.5B (24L) and the P9 results.

If OPCODE is always at ~L0 (0%), Operands at ~8-54%, and
Output at ~83-92%, this proves the pipeline is an emergent
property of transformer architecture, not specific to Qwen.

This phase re-probes Qwen-0.5B with a different prompt format
("# compute X" vs "def f(): return X") to test format invariance.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def probe_register(model, tok, prompts, labels, n_layers):
    """Probe each layer."""
    all_hidden = {l: [] for l in range(n_layers)}
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for l in range(n_layers):
            h = out.hidden_states[l+1][0, -1, :].float().cpu().numpy()
            all_hidden[l].append(h)

    y = np.array(labels)
    layer_accs = {}
    for l in range(n_layers):
        X = np.array(all_hidden[l])
        try:
            n_unique = len(np.unique(y))
            if n_unique < 2:
                acc = 0.0
            else:
                clf = LogisticRegression(max_iter=500, random_state=42)
                cv = min(5, n_unique)
                scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
                acc = scores.mean()
        except Exception:
            acc = 0.0
        layer_accs[l] = round(float(acc), 4)
    return layer_accs


def main():
    print("[P15] ISA Universality Test")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # P9 results (reference - "def f()" format)
    p9_peaks = {
        'OPCODE': 0, 'Operand_B': 2, 'CARRY': 4, 'Operand_A': 13,
        'COMPARISON': 14, 'MIN': 16, 'MEDIAN': 18, 'SUM': 20, 'MAX': 22,
    }

    # === Test with DIFFERENT prompt format ===
    # Format B: "# compute X + Y = "  (comment style, no "def")
    print("  Probing with alternate prompt format...")
    import random
    random.seed(42)

    # Operand registers (arithmetic)
    arith_prompts_b = []
    labels_a = []
    labels_b = []
    labels_op = []
    for a in range(1, 8):
        for b in range(1, 8):
            arith_prompts_b.append(f"# {a} + {b} =")
            labels_a.append(a)
            labels_b.append(b)
            labels_op.append(0)
            arith_prompts_b.append(f"# {a} - {b} =")
            labels_a.append(a)
            labels_b.append(b)
            labels_op.append(1)

    # Limit
    if len(arith_prompts_b) > 120:
        indices = random.sample(range(len(arith_prompts_b)), 120)
        arith_prompts_b = [arith_prompts_b[i] for i in indices]
        labels_a = [labels_a[i] for i in indices]
        labels_b = [labels_b[i] for i in indices]
        labels_op = [labels_op[i] for i in indices]

    format_b = {}
    format_b['OPCODE'] = probe_register(model, tok, arith_prompts_b, labels_op, n_layers)
    format_b['Operand_A'] = probe_register(model, tok, arith_prompts_b, labels_a, n_layers)
    format_b['Operand_B'] = probe_register(model, tok, arith_prompts_b, labels_b, n_layers)

    # Sorting
    sort_prompts_b = []
    labels_min = []
    labels_max = []
    import itertools
    digits = list(range(1, 7))
    combos = list(itertools.combinations(digits, 3))
    random.shuffle(combos)
    for combo in combos[:30]:
        for perm in itertools.permutations(combo):
            sort_prompts_b.append(f"# min({perm[0]},{perm[1]},{perm[2]}) =")
            labels_min.append(min(perm))
            labels_max.append(max(perm))

    format_b['MIN'] = probe_register(model, tok, sort_prompts_b, labels_min, n_layers)
    format_b['MAX'] = probe_register(model, tok, sort_prompts_b, labels_max, n_layers)

    # Print comparison
    print("\n  === P9 vs Format B peaks ===")
    comparison = {}
    for reg in ['OPCODE', 'Operand_A', 'Operand_B', 'MIN', 'MAX']:
        p9_layer = p9_peaks[reg]
        p9_norm = p9_layer / (n_layers - 1)
        fb_best = max(format_b[reg], key=format_b[reg].get)
        fb_norm = fb_best / (n_layers - 1)
        fb_acc = format_b[reg][fb_best]
        shift = fb_best - p9_layer
        print(f"    {reg:12s}: P9=L{p9_layer:2d} ({p9_norm:.0%})  "
              f"FormatB=L{fb_best:2d} ({fb_norm:.0%})  "
              f"shift={shift:+d}  acc={fb_acc:.1%}")
        comparison[reg] = {
            'p9_layer': p9_layer, 'format_b_layer': fb_best,
            'format_b_acc': fb_acc, 'shift': shift,
            'p9_normalized': round(p9_norm, 3),
            'fb_normalized': round(fb_norm, 3),
        }

    # Save
    format_b_json = {reg: {str(l): v for l, v in ld.items()} for reg, ld in format_b.items()}
    output = {
        'phase': 15, 'name': 'isa_universality',
        'n_layers': n_layers, 'comparison': comparison,
        'format_b_profiles': format_b_json,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase15_universality.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Overlay P9 and Format B profiles
    colors = {'OPCODE': '#e91e63', 'Operand_A': '#e74c3c', 'Operand_B': '#e67e22',
              'MIN': '#3498db', 'MAX': '#9b59b6'}
    layers = list(range(n_layers))

    for reg in ['OPCODE', 'Operand_A', 'Operand_B', 'MIN', 'MAX']:
        accs = [format_b[reg][l] for l in layers]
        axes[0].plot(layers, accs, '-', linewidth=2, color=colors[reg],
                     label=f'{reg} (format B)', alpha=0.8)
        # Mark P9 peak
        axes[0].axvline(x=p9_peaks[reg], color=colors[reg], linestyle='--', alpha=0.3)

    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Probe Accuracy', fontsize=12)
    axes[0].set_title('Format B: "# X + Y =" vs P9 peaks (dashed)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Shift diagram
    regs = list(comparison.keys())
    p9_positions = [comparison[r]['p9_normalized'] * 100 for r in regs]
    fb_positions = [comparison[r]['fb_normalized'] * 100 for r in regs]
    y_pos = range(len(regs))

    axes[1].barh(y_pos, p9_positions, 0.35, label='P9 ("def f()")', color='tab:blue', alpha=0.7)
    axes[1].barh([y + 0.35 for y in y_pos], fb_positions, 0.35,
                 label='Format B ("# X+Y")', color='tab:red', alpha=0.7)
    axes[1].set_yticks([y + 0.175 for y in y_pos])
    axes[1].set_yticklabels(regs, fontsize=11)
    axes[1].set_xlabel('Normalized Position (% of layers)', fontsize=12)
    axes[1].set_title('Register Position: Format Invariance', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('Phase 15: ISA Universality\nIs the register layout invariant to prompt format?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase15_universality.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
