# -*- coding: utf-8 -*-
"""
Phase 131: The Holographic Soul
Does the soul's power lie in its DIRECTION or MAGNITUDE?

"If the soul is a hologram, only its angle matters."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8


def train_soul(model, tok, data, device, layer=LAYER, seed=42, epochs=150):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for p, t in data:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(device)
            def inj(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def evaluate(model, tok, vec, data, device, layer=LAYER):
    c = 0
    for p, e in data:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0, -1, :].argmax().item()).strip() == e:
            c += 1
    return c / len(data)


def main():
    print("[P131] The Holographic Soul")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    task_data = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")],
    }
    test_data = {
        'MIN': [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")],
        'MAX': [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")],
    }

    # Train original souls
    souls = {}
    for task in ['MIN', 'MAX']:
        souls[task] = train_soul(model, tok, task_data[task], DEVICE)
        orig_norm = souls[task].norm().item()
        print("  %s soul trained. L2 norm = %.4f" % (task, orig_norm))

    # Test at various magnitudes
    # Scales: 0.01x, 0.1x, 0.5x, 1.0x (original), 2.0x, 5.0x, 10.0x
    # Also: normalized (unit vector)
    scales = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    results = {}
    for task in ['MIN', 'MAX']:
        all_data = task_data[task] + test_data[task]
        orig_norm = souls[task].norm().item()
        unit_vec = souls[task] / souls[task].norm()

        results[task] = {}
        for scale in scales:
            scaled_vec = unit_vec * (orig_norm * scale)
            acc = evaluate(model, tok, scaled_vec, all_data, DEVICE)
            results[task][scale] = acc
            print("    %s | scale=%.2f (norm=%.2f): acc=%.0f%%" % (
                task, scale, orig_norm * scale, acc * 100))

        # Also test pure unit vector (norm=1.0 exactly)
        acc_unit = evaluate(model, tok, unit_vec, all_data, DEVICE)
        results[task]['unit'] = acc_unit
        print("    %s | UNIT vector (norm=1.0): acc=%.0f%%" % (task, acc_unit * 100))

    # Find the critical magnitude threshold
    for task in ['MIN', 'MAX']:
        orig_acc = results[task][1.0]
        threshold = None
        for s in scales:
            if results[task][s] >= orig_acc * 0.9:
                threshold = s
                break
        print("  %s: Original acc=%.0f%%, threshold scale=%.2f" % (
            task, orig_acc * 100, threshold or -1))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'MIN': '#2196F3', 'MAX': '#FF5722'}

    # Panel 1: Accuracy vs Scale (log scale x-axis)
    ax = axes[0]
    for task in ['MIN', 'MAX']:
        accs = [results[task][s] for s in scales]
        orig_norm = souls[task].norm().item()
        norms = [orig_norm * s for s in scales]
        ax.plot(scales, accs, 'o-', color=colors[task], label=task,
                markersize=8, linewidth=2)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Original')
    ax.set_xscale('log')
    ax.set_xlabel('Scale Factor (relative to original)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title('Accuracy vs Magnitude', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Accuracy vs actual L2 norm
    ax = axes[1]
    for task in ['MIN', 'MAX']:
        accs = [results[task][s] for s in scales]
        orig_norm = souls[task].norm().item()
        norms = [orig_norm * s for s in scales]
        ax.plot(norms, accs, 's-', color=colors[task], label=task,
                markersize=8, linewidth=2)
        # Mark unit vector
        ax.plot(1.0, results[task]['unit'], '*', color=colors[task],
                markersize=15, markeredgecolor='black',
                label='%s unit vec' % task)
    ax.set_xlabel('L2 Norm of Soul Vector', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title('Accuracy vs L2 Norm', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Bar chart - Direction vs Magnitude decomposition
    ax = axes[2]
    tasks = ['MIN', 'MAX']
    x = np.arange(len(tasks))
    w = 0.2
    # Original, half-magnitude, double-magnitude, unit-vector
    configs = [
        ('0.1x', 0.1, '#BBDEFB'),
        ('0.5x', 0.5, '#64B5F6'),
        ('1.0x (orig)', 1.0, '#1976D2'),
        ('2.0x', 2.0, '#0D47A1'),
        ('Unit (norm=1)', 'unit', '#FF9800'),
    ]
    for ci, (label, key, color) in enumerate(configs):
        vals = [results[t][key] for t in tasks]
        bars = ax.bar(x + ci*w - 2*w, vals, w, label=label, color=color,
                      edgecolor='black', linewidth=0.5)
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.02,
                        "%.0f%%" % (bar.get_height() * 100),
                        ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.3)
    ax.set_title('Direction vs Magnitude', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('Phase 131: The Holographic Soul\n'
                 '"If the soul is a hologram, only its angle matters"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase131_holographic.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save
    output = {
        'phase': 131, 'name': 'holographic_soul',
        'layer': LAYER,
        'original_norms': {t: round(souls[t].norm().item(), 4) for t in tasks},
        'scales': scales,
        'results': {t: {str(k): round(v, 4) for k, v in results[t].items()} for t in tasks},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase131_holographic.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
