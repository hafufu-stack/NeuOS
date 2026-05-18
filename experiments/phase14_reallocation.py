# -*- coding: utf-8 -*-
"""
Phase 14: Dynamic Register Reallocation
P4 showed catastrophic degradation at 10% layer dropout.
Can we survive hardware failure by dynamically remapping
registers to undamaged layers?

Method:
  1. Destroy a critical register (e.g. L16 = MIN)
  2. Measure degraded performance
  3. Apply dynamic remapping: route through alternative layers
  4. Measure recovery

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P14] Dynamic Register Reallocation")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # Test: arithmetic (uses L13=Operand_A, L20=SUM)
    problems = []
    for a in range(1, 7):
        for b in range(1, 7):
            if a + b < 10:
                problems.append((f"def f(): return {a} + {b} =", str(a + b)))

    import random
    random.seed(42)
    if len(problems) > 25:
        problems = random.sample(problems, 25)

    # === Baseline: no damage ===
    print("  Baseline: no damage...")
    baseline = 0
    for prompt, ans in problems:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == ans:
            baseline += 1
    baseline_acc = baseline / len(problems)
    print(f"    Baseline: {baseline_acc:.1%}")

    # === Damage scenarios ===
    damage_layers = [4, 13, 16, 20]  # CARRY, Operand_A, MIN, SUM
    results = {}

    for damage_l in damage_layers:
        # Step 1: Measure damage
        print(f"  Damaging L{damage_l}...")

        def damage_hook(module, input, output):
            """Zero out hidden states at damaged layer (hardware failure)."""
            if isinstance(output, tuple):
                h = output[0].clone()
                if h.dim() == 3:
                    h[0, -1, :] = 0.0
                elif h.dim() == 2:
                    h[-1, :] = 0.0
                return (h,) + output[1:]
            else:
                h = output.clone()
                if h.dim() == 3:
                    h[0, -1, :] = 0.0
                elif h.dim() == 2:
                    h[-1, :] = 0.0
                return h

        damaged_correct = 0
        handle = model.model.layers[damage_l].register_forward_hook(damage_hook)
        for prompt, ans in problems:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            if pred == ans:
                damaged_correct += 1
        handle.remove()
        damaged_acc = damaged_correct / len(problems)
        print(f"    Damaged: {damaged_acc:.1%}")

        # Step 2: Try remapping - use adjacent layer's state instead
        # Strategy: copy the hidden state from layer damage_l-1 to damage_l
        best_remap_acc = 0
        best_remap_src = None

        for remap_src in [damage_l - 2, damage_l - 1, damage_l + 1, damage_l + 2]:
            if remap_src < 0 or remap_src >= n_layers or remap_src == damage_l:
                continue

            remap_correct = 0
            for prompt, ans in problems:
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                # Capture the source layer's state
                captured = [None]
                def cap_hook(module, input, output):
                    captured[0] = get_last_token(output)

                h_cap = model.model.layers[remap_src].register_forward_hook(cap_hook)

                # Also damage the target layer, but inject source layer's state
                def remap_hook(module, input, output):
                    if captured[0] is not None:
                        return replace_last_token(output, captured[0])
                    return output

                h_dam = model.model.layers[damage_l].register_forward_hook(remap_hook)

                with torch.no_grad():
                    out = model(**inp)
                h_cap.remove()
                h_dam.remove()

                pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
                if pred == ans:
                    remap_correct += 1

            remap_acc = remap_correct / len(problems)
            if remap_acc > best_remap_acc:
                best_remap_acc = remap_acc
                best_remap_src = remap_src

        print(f"    Best remap: L{best_remap_src}->L{damage_l} ({best_remap_acc:.1%})")

        # Step 3: Noise injection (add Gaussian noise instead of zeroing)
        noise_levels = [0.1, 0.5, 1.0, 2.0]
        noise_results = {}
        for noise_std in noise_levels:
            noise_correct = 0
            def noise_hook(module, input, output, std=noise_std):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    noise = torch.randn_like(h) * std
                    if h.dim() == 3:
                        h[0, -1, :] += noise[0, -1, :]
                    elif h.dim() == 2:
                        h[-1, :] += noise[-1, :]
                    return (h,) + output[1:]
                return output

            handle = model.model.layers[damage_l].register_forward_hook(noise_hook)
            for prompt, ans in problems:
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
                if pred == ans:
                    noise_correct += 1
            handle.remove()
            noise_results[str(noise_std)] = round(noise_correct / len(problems), 4)

        results[f'L{damage_l}'] = {
            'damaged_acc': round(damaged_acc, 4),
            'best_remap_acc': round(best_remap_acc, 4),
            'best_remap_src': best_remap_src,
            'noise_resilience': noise_results,
            'recovery': round(best_remap_acc - damaged_acc, 4),
        }

    # Save
    output = {
        'phase': 14, 'name': 'dynamic_register_reallocation',
        'n_problems': len(problems), 'n_layers': n_layers,
        'baseline_acc': round(baseline_acc, 4),
        'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase14_reallocation.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Damage vs remap comparison
    dl_labels = [f'L{l}' for l in damage_layers]
    damaged_accs = [results[f'L{l}']['damaged_acc'] for l in damage_layers]
    remap_accs = [results[f'L{l}']['best_remap_acc'] for l in damage_layers]
    x = np.arange(len(damage_layers))
    w = 0.3
    axes[0].bar(x - w/2, damaged_accs, w, label='Damaged', color='tab:red', edgecolor='black')
    axes[0].bar(x + w/2, remap_accs, w, label='Remapped', color='tab:green', edgecolor='black')
    axes[0].axhline(y=baseline_acc, color='blue', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_acc:.0%})')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dl_labels)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Damage vs Recovery', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize=9)

    # Noise resilience
    for l in damage_layers:
        nr = results[f'L{l}']['noise_resilience']
        stds = [float(k) for k in nr.keys()]
        accs = [nr[k] for k in nr.keys()]
        axes[1].plot(stds, accs, 'o-', linewidth=2, label=f'L{l}', markersize=5)
    axes[1].set_xlabel('Noise Std', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Noise Resilience by Layer', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=baseline_acc, color='blue', linestyle='--', alpha=0.3)

    # Summary
    axes[2].axis('off')
    recoveries = [results[f'L{l}']['recovery'] for l in damage_layers]
    avg_recovery = np.mean(recoveries)
    summary = (
        f"Dynamic Register Reallocation\n\n"
        f"Baseline: {baseline_acc:.0%}\n\n"
    )
    for l in damage_layers:
        r = results[f'L{l}']
        summary += f"L{l}: {r['damaged_acc']:.0%} -> {r['best_remap_acc']:.0%}\n"
    summary += f"\nAvg recovery: +{avg_recovery:.0%}"
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=12, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 14: Dynamic Register Reallocation\nCan we survive hardware failure via remapping?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase14_reallocation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
