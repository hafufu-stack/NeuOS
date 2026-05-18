# -*- coding: utf-8 -*-
"""
Phase 3: Write Optimization - Beyond 83%
Can we beat P209's 83% write success rate?

Uses embedding surgery (Aletheia standard).
Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ARITH_PROBLEMS = [
    ("def f(): return 3 + 4 =", " 7", " 5"),
    ("def f(): return 5 + 2 =", " 7", " 3"),
    ("def f(): return 8 + 1 =", " 9", " 6"),
    ("def f(): return 6 + 3 =", " 9", " 4"),
    ("def f(): return 4 + 4 =", " 8", " 2"),
    ("def f(): return 2 + 7 =", " 9", " 5"),
    ("def f(): return 1 + 6 =", " 7", " 3"),
    ("def f(): return 5 + 3 =", " 8", " 4"),
    ("def f(): return 7 + 2 =", " 9", " 1"),
    ("def f(): return 3 + 3 =", " 6", " 8"),
    ("def f(): return 9 + 0 =", " 9", " 3"),
    ("def f(): return 2 + 5 =", " 7", " 1"),
]


def main():
    print("[P3] Write Optimization - Beyond 83%")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # Verify baseline first
    baseline_ok = 0
    for prompt, correct, inject in ARITH_PROBLEMS:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred_id = out.logits[0, -1, :].argmax().item()
        correct_id = tok.encode(correct)[-1]
        if pred_id == correct_id:
            baseline_ok += 1
    print(f"  Baseline accuracy: {baseline_ok}/{len(ARITH_PROBLEMS)}")

    results = {}

    # === Method A: Multi-layer Full Replacement ===
    print("  Method A: Multi-layer Full Replacement...")
    success_a = 0
    for prompt, correct, inject in ARITH_PROBLEMS:
        inject_val = inject.strip()
        donor = f"def f(): return {inject_val}\nf() ="
        inp_donor = tok(donor, return_tensors='pt').to(DEVICE)
        inp_target = tok(prompt, return_tensors='pt').to(DEVICE)

        # Collect donor hidden states via output_hidden_states
        with torch.no_grad():
            out_d = model(**inp_donor, output_hidden_states=True)
        donor_states = {}
        for l in range(n_layers):
            hs = out_d.hidden_states[l+1]
            donor_states[l] = get_last_token(hs)

        replace_layers = list(range(max(0, n_layers-7), n_layers-1))

        def make_hook(layer_idx):
            dvec = donor_states[layer_idx]
            def hook_fn(module, input, output):
                return replace_last_token(output, dvec)
            return hook_fn

        handles = []
        for l in replace_layers:
            handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))

        with torch.no_grad():
            out = model(**inp_target)
            pred_id = out.logits[0, -1, :].argmax().item()

        for h in handles:
            h.remove()

        inject_id = tok.encode(inject)[-1]
        if pred_id == inject_id:
            success_a += 1

    acc_a = success_a / len(ARITH_PROBLEMS)
    results['full_replacement'] = round(acc_a, 4)
    print(f"    Multi-layer replacement: {acc_a:.1%}")

    # === Method B: Single-layer replacement sweep ===
    print("  Method B: Single-layer replacement sweep...")
    layer_accs = {}
    for target_l in range(n_layers):
        success_c = 0
        for prompt, correct, inject in ARITH_PROBLEMS:
            inject_val = inject.strip()
            donor = f"def f(): return {inject_val}\nf() ="
            inp_donor = tok(donor, return_tensors='pt').to(DEVICE)
            inp_target = tok(prompt, return_tensors='pt').to(DEVICE)

            with torch.no_grad():
                out_d = model(**inp_donor, output_hidden_states=True)
            donor_h = get_last_token(out_d.hidden_states[target_l+1])

            def single_hook(module, input, output):
                return replace_last_token(output, donor_h)

            handle = model.model.layers[target_l].register_forward_hook(single_hook)
            with torch.no_grad():
                out = model(**inp_target)
                pred_id = out.logits[0, -1, :].argmax().item()
            handle.remove()

            inject_id = tok.encode(inject)[-1]
            if pred_id == inject_id:
                success_c += 1

        layer_accs[str(target_l)] = round(success_c / len(ARITH_PROBLEMS), 4)

    results['single_layer_sweep'] = layer_accs
    best_l = max(layer_accs, key=layer_accs.get)
    print(f"    Single-layer best: L{best_l} ({layer_accs[best_l]:.1%})")

    # Save
    output = {
        'phase': 3, 'name': 'write_optimization',
        'n_problems': len(ARITH_PROBLEMS), 'n_layers': n_layers,
        'baseline_accuracy': round(baseline_ok / len(ARITH_PROBLEMS), 4),
        'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase3_write_opt.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    layers = list(range(n_layers))
    sweep_accs = [layer_accs[str(l)] for l in layers]
    axes[0].plot(layers, sweep_accs, 'o-', linewidth=2, markersize=5, color='tab:purple')
    axes[0].set_xlabel('Replacement Layer', fontsize=12)
    axes[0].set_ylabel('Write Success Rate', fontsize=12)
    axes[0].set_title('Single-Layer Replacement Sweep', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.83, color='red', linestyle='--', alpha=0.5, label='P209 baseline (83%)')
    axes[0].legend()

    axes[1].axis('off')
    summary = (
        f"Write Optimization Results\n\n"
        f"Baseline: {baseline_ok}/{len(ARITH_PROBLEMS)}\n"
        f"Multi-layer Replace: {acc_a:.0%}\n"
        f"Best Single Layer: L{best_l} ({layer_accs[best_l]:.0%})\n\n"
        f"P209 Baseline: 83%"
    )
    axes[1].text(0.5, 0.5, summary, transform=axes[1].transAxes,
                 fontsize=14, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 3: Write Optimization\nCan we beat P209\'s 83% write success?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase3_write_opt.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
