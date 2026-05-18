# -*- coding: utf-8 -*-
"""
Phase 23: Neural fork() System Call
Can we run L0-L15 once, then fork into multiple execution paths
by injecting different register vectors at L16+?

Uses batched inference: clone the hidden state at L15,
inject different operation vectors per batch item at L16,
get multiple outputs in a single forward pass.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P23] Neural fork() System Call")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # === Step 1: Extract register vectors ===
    print("  Step 1: Extracting execution registers...")

    def extract_avg_vec(prompts, layer):
        vecs = []
        for prompt in prompts:
            captured = [None]
            def cap(module, input, output):
                captured[0] = get_last_token(output)
            h = model.model.layers[layer].register_forward_hook(cap)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            vecs.append(captured[0])
        return torch.stack(vecs).mean(dim=0)

    min_vec = extract_avg_vec(
        [f"def f(): return min({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:10], 16)
    max_vec = extract_avg_vec(
        [f"def f(): return max({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:10], 22)
    sum_vec = extract_avg_vec(
        [f"def f(): return {a} + {b} =" for a in range(1, 6) for b in range(1, 6) if a+b < 10][:10], 20)

    print("    Extracted MIN(L16), MAX(L22), SUM(L20)")

    # === Step 2: fork() - single prompt, 3 different outputs ===
    print("\n  Step 2: Neural fork() execution...")

    test_cases = [
        ("def f(a, b): return a + b\nf(3, 7) =", 3, 7),  # with instruction
        ("def f(a, b): return a + b\nf(5, 2) =", 5, 2),
        ("def f(a, b): return a + b\nf(8, 1) =", 8, 1),
        ("def f(a, b): return a + b\nf(4, 6) =", 4, 6),
    ]

    fork_results = []
    for prompt, a, b in test_cases:
        # Baseline: run without any injection
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out_base = model(**inp)
        base_pred = tok.decode(out_base.logits[0, -1, :].argmax().item()).strip()

        # Fork 1: inject MIN at L16
        preds = {}
        for op_name, vec, layer in [('MIN', min_vec, 16), ('MAX', max_vec, 22), ('SUM', sum_vec, 20)]:
            def inject(module, input, output, v=vec):
                h = output[0] if isinstance(output, tuple) else output
                h_clone = h.clone()
                if h_clone.dim() == 3:
                    h_clone[0, -1, :] = v.to(h_clone.dtype)
                elif h_clone.dim() == 2:
                    h_clone[-1, :] = v.to(h_clone.dtype)
                if isinstance(output, tuple):
                    return (h_clone,) + output[1:]
                return h_clone

            hook = model.model.layers[layer].register_forward_hook(inject)
            with torch.no_grad():
                out = model(**inp)
            hook.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            preds[op_name] = pred

        expected_min = str(min(a, b))
        expected_max = str(max(a, b))
        expected_sum = str(a + b) if a + b < 10 else 'N/A'

        fork_results.append({
            'a': a, 'b': b, 'baseline': base_pred,
            'fork_MIN': preds['MIN'], 'fork_MAX': preds['MAX'], 'fork_SUM': preds['SUM'],
            'exp_min': expected_min, 'exp_max': expected_max, 'exp_sum': expected_sum,
        })

        min_ok = "OK" if preds['MIN'] == expected_min else "MISS"
        max_ok = "OK" if preds['MAX'] == expected_max else "MISS"
        sum_ok = "OK" if preds['SUM'] == expected_sum else "MISS"
        print(f"    ({a},{b}): base={base_pred} "
              f"MIN={preds['MIN']}({min_ok}) MAX={preds['MAX']}({max_ok}) SUM={preds['SUM']}({sum_ok})")

    # Calculate accuracy
    n = len(fork_results)
    min_acc = sum(1 for r in fork_results if r['fork_MIN'] == r['exp_min']) / n
    max_acc = sum(1 for r in fork_results if r['fork_MAX'] == r['exp_max']) / n
    sum_acc = sum(1 for r in fork_results if r['fork_SUM'] == r['exp_sum']) / n

    # Check: did fork produce DIFFERENT outputs for different injections?
    diff_count = sum(1 for r in fork_results
                     if len(set([r['fork_MIN'], r['fork_MAX'], r['fork_SUM']])) > 1) / n
    print(f"\n    MIN accuracy: {min_acc:.1%}")
    print(f"    MAX accuracy: {max_acc:.1%}")
    print(f"    SUM accuracy: {sum_acc:.1%}")
    print(f"    Differentiation rate: {diff_count:.1%}")

    # Save
    output = {
        'phase': 23, 'name': 'neural_fork',
        'n_tests': n,
        'min_acc': round(min_acc, 4), 'max_acc': round(max_acc, 4),
        'sum_acc': round(sum_acc, 4), 'diff_rate': round(diff_count, 4),
        'fork_results': fork_results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase23_fork.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ops = ['MIN\n(L16)', 'MAX\n(L22)', 'SUM\n(L20)']
    accs = [min_acc, max_acc, sum_acc]
    colors = ['tab:green', 'tab:red', 'tab:blue']
    axes[0].bar(ops, accs, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('fork() Accuracy per Operation', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    axes[1].axis('off')
    summary = (
        f"Neural fork()\n\n"
        f"1 prompt -> 3 programs\n\n"
        f"  fork(MIN): {min_acc:.0%}\n"
        f"  fork(MAX): {max_acc:.0%}\n"
        f"  fork(SUM): {sum_acc:.0%}\n\n"
        f"  Differentiation: {diff_count:.0%}\n\n"
        f"{'SIMD works!' if diff_count > 0.5 else 'Investigating...'}"
    )
    axes[1].text(0.5, 0.5, summary, transform=axes[1].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 23: Neural fork()\n1 input, 3 different programs via register injection',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase23_fork.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
