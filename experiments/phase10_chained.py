# -*- coding: utf-8 -*-
"""
Phase 10: Chained Computation - Register Reuse
Can the transformer do multi-step arithmetic? (3 + 4 + 2 = ?)
If so, does it REUSE the SUM register from step 1 as the
operand for step 2?

This tests whether the Neural CPU has a "program counter" that
advances through multi-step programs.

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


def main():
    print("[P10] Chained Computation - Register Reuse")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # === Part A: Can the model do chained addition? ===
    print("  Part A: Chained addition accuracy...")

    # Single-step (control)
    single_problems = []
    for a in range(1, 7):
        for b in range(1, 7):
            if a + b < 10:
                single_problems.append((f"def f(): return {a} + {b} =", a + b))

    # Two-step chains
    chain_problems = []
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(1, 5):
                s = a + b + c
                if s < 10 and a + b < 10:
                    chain_problems.append((f"def f(): return {a} + {b} + {c} =",
                                          a, b, c, a + b, s))

    import random
    random.seed(42)
    if len(chain_problems) > 40:
        chain_problems = random.sample(chain_problems, 40)

    # Test single-step accuracy
    single_correct = 0
    for prompt, ans in single_problems[:20]:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(ans):
            single_correct += 1
    single_acc = single_correct / min(20, len(single_problems))
    print(f"    Single-step accuracy: {single_acc:.1%}")

    # Test chain accuracy
    chain_correct = 0
    for prompt, a, b, c, intermediate, final in chain_problems:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(final):
            chain_correct += 1
    chain_acc = chain_correct / len(chain_problems) if chain_problems else 0
    print(f"    Chain accuracy: {chain_acc:.1%}")

    # === Part B: Probe for intermediate result ===
    print("  Part B: Intermediate result probing...")
    # In "3 + 4 + 2 =", can we find the intermediate sum (7) at some layer?
    all_hidden = {l: [] for l in range(n_layers)}
    labels_a = []
    labels_b = []
    labels_c = []
    labels_intermediate = []
    labels_final = []

    for prompt, a, b, c, intermediate, final in chain_problems:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for l in range(n_layers):
            h = out.hidden_states[l+1][0, -1, :].float().cpu().numpy()
            all_hidden[l].append(h)
        labels_a.append(a)
        labels_b.append(b)
        labels_c.append(c)
        labels_intermediate.append(intermediate)
        labels_final.append(final)

    results = {
        'accuracy': {'single': round(single_acc, 4), 'chain': round(chain_acc, 4)}
    }
    for target_name, y_arr in [('operand_A', labels_a), ('operand_B', labels_b),
                                ('operand_C', labels_c),
                                ('intermediate_sum', labels_intermediate),
                                ('final_sum', labels_final)]:
        y = np.array(y_arr)
        layer_probe = {}
        for l in range(n_layers):
            X = np.array(all_hidden[l])
            try:
                n_unique = len(np.unique(y))
                if n_unique < 2:
                    acc = 0.0
                else:
                    clf = LogisticRegression(max_iter=500, random_state=42)
                    cv = min(5, n_unique, len(y) // n_unique)
                    if cv < 2:
                        cv = 2
                    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
                    acc = scores.mean()
            except Exception:
                acc = 0.0
            layer_probe[str(l)] = round(float(acc), 4)
        results[target_name] = layer_probe
        best_l = max(layer_probe, key=layer_probe.get)
        print(f"    {target_name}: best=L{best_l} ({layer_probe[best_l]:.1%})")

    # Save
    output = {
        'phase': 10, 'name': 'chained_computation',
        'n_single': len(single_problems), 'n_chain': len(chain_problems),
        'n_layers': n_layers, 'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase10_chained.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy comparison
    axes[0].bar(['Single\n(a+b)', 'Chain\n(a+b+c)'], [single_acc, chain_acc],
                color=['tab:blue', 'tab:red'], edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Computation Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([single_acc, chain_acc]):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Register profiles
    layers = list(range(n_layers))
    for target, color, marker in [
        ('operand_A', 'tab:red', 'o'),
        ('operand_B', 'tab:orange', 's'),
        ('operand_C', 'tab:green', '^'),
        ('intermediate_sum', 'tab:purple', 'D'),
        ('final_sum', 'tab:blue', 'v'),
    ]:
        accs = [results[target][str(l)] for l in layers]
        axes[1].plot(layers, accs, f'{marker}-', linewidth=2, markersize=4,
                     label=target, color=color, alpha=0.8)

    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Probe Accuracy', fontsize=12)
    axes[1].set_title('Register Profiles for a+b+c', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    # Key question visualization
    axes[2].axis('off')
    inter_best_l = max(results['intermediate_sum'], key=results['intermediate_sum'].get)
    inter_best_acc = results['intermediate_sum'][inter_best_l]
    final_best_l = max(results['final_sum'], key=results['final_sum'].get)
    final_best_acc = results['final_sum'][final_best_l]

    verdict = "YES - Register Reuse!" if inter_best_acc > 0.5 else "NO - No intermediate storage"
    summary = (
        f"Chained Computation\n"
        f"a + b + c = ?\n\n"
        f"Intermediate (a+b):\n"
        f"  L{inter_best_l} ({inter_best_acc:.0%})\n\n"
        f"Final (a+b+c):\n"
        f"  L{final_best_l} ({final_best_acc:.0%})\n\n"
        f"Verdict: {verdict}"
    )
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 10: Chained Computation\n'
                 'Does the Neural CPU reuse registers across steps?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase10_chained.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
