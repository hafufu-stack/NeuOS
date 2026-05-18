# -*- coding: utf-8 -*-
"""
Phase 8: Multi-Program Execution
Can a transformer execute multiple independent computations
in a single forward pass? (Neural "multiprocessing")

Test: "3+4= and 8-2=" in one prompt.
Probe whether A,B registers for BOTH problems exist independently.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import torch.nn.functional as F
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

# Single-task prompts (control)
SINGLE_TASKS = [
    ("def f(): return 3 + 4 =", 3, 4, 7),
    ("def f(): return 5 + 2 =", 5, 2, 7),
    ("def f(): return 8 + 1 =", 8, 1, 9),
    ("def f(): return 6 + 3 =", 6, 3, 9),
    ("def f(): return 2 + 7 =", 2, 7, 9),
    ("def f(): return 4 + 5 =", 4, 5, 9),
    ("def f(): return 1 + 6 =", 1, 6, 7),
    ("def f(): return 9 + 0 =", 9, 0, 9),
]

# Multi-task prompts: two independent computations
MULTI_TASKS = [
    ("def f(): return 3+4\ndef g(): return 8-2\nf() =", 3, 4, 7, 8, 2, 6),
    ("def f(): return 5+2\ndef g(): return 9-1\nf() =", 5, 2, 7, 9, 1, 8),
    ("def f(): return 8+1\ndef g(): return 7-3\nf() =", 8, 1, 9, 7, 3, 4),
    ("def f(): return 6+3\ndef g(): return 5-2\nf() =", 6, 3, 9, 5, 2, 3),
    ("def f(): return 2+7\ndef g(): return 6-4\nf() =", 2, 7, 9, 6, 4, 2),
    ("def f(): return 4+5\ndef g(): return 8-6\nf() =", 4, 5, 9, 8, 6, 2),
    ("def f(): return 1+6\ndef g(): return 9-5\nf() =", 1, 6, 7, 9, 5, 4),
    ("def f(): return 9+0\ndef g(): return 7-1\nf() =", 9, 0, 9, 7, 1, 6),
]

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9"]


def main():
    print("[P8] Multi-Program Execution")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # === Part A: Does multi-task prompt preserve single-task accuracy? ===
    print("  Part A: Accuracy comparison...")
    single_correct = 0
    for prompt, a, b, ans in SINGLE_TASKS:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        expected = str(ans)
        if pred == expected:
            single_correct += 1

    multi_correct_f = 0
    for prompt, a1, b1, ans1, a2, b2, ans2 in MULTI_TASKS:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(ans1):
            multi_correct_f += 1

    single_acc = single_correct / len(SINGLE_TASKS)
    multi_acc = multi_correct_f / len(MULTI_TASKS)
    print(f"    Single-task accuracy: {single_acc:.1%}")
    print(f"    Multi-task accuracy (f): {multi_acc:.1%}")

    # === Part B: Probe for BOTH computations' operands ===
    print("  Part B: Operand register probing...")
    # Collect hidden states and labels for multi-task prompts
    hidden_multi = {l: [] for l in range(n_layers)}
    labels_a1 = []  # First computation's operand A
    labels_a2 = []  # Second computation's operand A

    for prompt, a1, b1, ans1, a2, b2, ans2 in MULTI_TASKS:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for l in range(n_layers):
            h = out.hidden_states[l+1][0, -1, :].float().cpu().numpy()
            hidden_multi[l].append(h)
        labels_a1.append(a1)
        labels_a2.append(a2)

    # Probe for first and second operands
    results = {'accuracy': {'single': round(single_acc, 4), 'multi': round(multi_acc, 4)}}
    for target_name, labels in [('operand_A1', labels_a1), ('operand_A2', labels_a2)]:
        y = np.array(labels)
        layer_probe = {}
        for l in range(n_layers):
            X = np.array(hidden_multi[l])
            try:
                if len(np.unique(y)) < 2:
                    acc = 0.0
                else:
                    clf = LogisticRegression(max_iter=500, random_state=42)
                    scores = cross_val_score(clf, X, y, cv=min(3, len(np.unique(y))),
                                             scoring='accuracy')
                    acc = scores.mean()
            except Exception:
                acc = 0.0
            layer_probe[str(l)] = round(float(acc), 4)
        results[target_name] = layer_probe
        best_l = max(layer_probe, key=layer_probe.get)
        print(f"    {target_name}: best=L{best_l} ({layer_probe[best_l]:.1%})")

    # Save
    output = {
        'phase': 8, 'name': 'multi_program_execution',
        'n_single': len(SINGLE_TASKS), 'n_multi': len(MULTI_TASKS),
        'n_layers': n_layers, 'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase8_multiprogram.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Accuracy comparison
    axes[0].bar(['Single\nTask', 'Multi\nTask'], [single_acc, multi_acc],
                color=['tab:blue', 'tab:orange'], edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Output Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([single_acc, multi_acc]):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Operand A1 probe
    layers = list(range(n_layers))
    a1_accs = [results['operand_A1'][str(l)] for l in layers]
    a2_accs = [results['operand_A2'][str(l)] for l in layers]
    axes[1].plot(layers, a1_accs, 'o-', linewidth=2, label='Operand A1 (f)', color='tab:blue')
    axes[1].plot(layers, a2_accs, 's-', linewidth=2, label='Operand A2 (g)', color='tab:red')
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Probe Accuracy', fontsize=12)
    axes[1].set_title('Register Independence', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    # Summary
    axes[2].axis('off')
    summary = (
        f"Multi-Program Execution\n\n"
        f"Single-task: {single_acc:.0%}\n"
        f"Multi-task: {multi_acc:.0%}\n\n"
        f"A1 best: L{max(results['operand_A1'], key=results['operand_A1'].get)}\n"
        f"A2 best: L{max(results['operand_A2'], key=results['operand_A2'].get)}\n\n"
        f"Independent registers = multiprocessing"
    )
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 8: Multi-Program Execution\nCan the Neural CPU run two programs at once?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase8_multiprogram.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
