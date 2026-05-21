# -*- coding: utf-8 -*-
"""
Phase 132: The Stem Cell Soul
Can the max-chaos vector from P129's phase diagram differentiate into any function?

"At the edge of chaos, all possibilities exist."
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


def get_output_entropy(model, tok, vec, prompt, device, layer=LAYER):
    """Get output distribution entropy for a single prompt."""
    def inj(m, i, o, v=vec): return replace_last_token(o, v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    probs = torch.softmax(out.logits[0, -1, :], dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
    top5 = torch.topk(probs, 5)
    return entropy, [(tok.decode(idx.item()).strip(), prob.item())
                     for idx, prob in zip(top5.indices, top5.values)]


def main():
    print("[P132] The Stem Cell Soul")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    task_data = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")],
        'ADD': [("3, 2) =","5"),("1, 4) =","5"),("2, 6) =","8"),
                 ("3, 3) =","6"),("4, 1) =","5")],
        'SUB': [("7, 3) =","4"),("5, 2) =","3"),("9, 1) =","8"),
                 ("6, 4) =","2"),("8, 3) =","5")],
    }

    # Train corner souls
    print("  Training corner souls...")
    souls = {}
    for task in task_data:
        souls[task] = train_soul(model, tok, task_data[task], DEVICE)

    # Create stem cell: the max-chaos interpolation from P129
    # Max entropy was at t=0.15, s=0.0 (near MIN corner but slightly toward MAX)
    # Also test the center (t=0.5, s=0.5) and other chaos candidates
    stem_configs = {
        'P129 max-chaos (t=0.15,s=0.0)': lambda: (
            0.85 * 1.0 * souls['MIN'] + 0.15 * 1.0 * souls['MAX'] +
            0.85 * 0.0 * souls['ADD'] + 0.15 * 0.0 * souls['SUB']),
        'Center (t=0.5,s=0.5)': lambda: (
            0.5 * 0.5 * souls['MIN'] + 0.5 * 0.5 * souls['MAX'] +
            0.5 * 0.5 * souls['ADD'] + 0.5 * 0.5 * souls['SUB']),
        'Equal mix (mean)': lambda: (
            souls['MIN'] + souls['MAX'] + souls['ADD'] + souls['SUB']) / 4,
        'Zero vector': lambda: torch.zeros_like(souls['MIN']),
        'Random vector': lambda: torch.randn_like(souls['MIN']) * souls['MIN'].norm() * 0.01,
    }

    # Test each stem cell on all tasks
    print("  Testing stem cells on all tasks...")
    stem_results = {}
    stem_entropies = {}
    for stem_name, make_stem in stem_configs.items():
        stem_vec = make_stem()
        stem_results[stem_name] = {}
        entropies = []
        for task in task_data:
            acc = evaluate(model, tok, stem_vec, task_data[task], DEVICE)
            stem_results[stem_name][task] = acc

            # Get entropy on first prompt
            ent, top5 = get_output_entropy(model, tok, stem_vec,
                                           task_data[task][0][0], DEVICE)
            entropies.append(ent)
        stem_entropies[stem_name] = float(np.mean(entropies))
        print("    %s:" % stem_name[:40])
        for task in task_data:
            print("      %s: %.0f%%" % (task, stem_results[stem_name][task] * 100))
        print("      Mean entropy: %.2f bits" % stem_entropies[stem_name])

    # Differentiation test: Can a tiny nudge toward a task make stem cell specialize?
    print("  Differentiation test: stem + epsilon * task_soul...")
    stem_vec = stem_configs['P129 max-chaos (t=0.15,s=0.0)']()
    epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    diff_results = {}
    for task in ['MIN', 'MAX']:
        diff_results[task] = {}
        task_direction = souls[task] / souls[task].norm()
        for eps in epsilons:
            nudged = stem_vec + eps * souls[task].norm() * task_direction
            acc = evaluate(model, tok, nudged, task_data[task], DEVICE)
            diff_results[task][eps] = acc
        print("    %s differentiation: %s" % (
            task, ["%s=%.0f%%" % (e, diff_results[task][e]*100) for e in epsilons]))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Stem cell performance heatmap
    ax = axes[0]
    stem_names_short = ['Max-chaos', 'Center', 'Equal mix', 'Zero', 'Random']
    task_names = list(task_data.keys())
    heat_data = np.array([[stem_results[sn][t] for t in task_names]
                          for sn in stem_configs.keys()])
    im = ax.imshow(heat_data, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    for i in range(len(stem_names_short)):
        for j in range(len(task_names)):
            ax.text(j, i, "%.0f%%" % (heat_data[i, j] * 100),
                    ha='center', va='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names)
    ax.set_yticks(range(len(stem_names_short)))
    ax.set_yticklabels(stem_names_short, fontsize=8)
    ax.set_title('Stem Cell Performance', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: Differentiation curves
    ax = axes[1]
    colors = {'MIN': '#2196F3', 'MAX': '#FF5722'}
    for task in ['MIN', 'MAX']:
        accs = [diff_results[task][e] for e in epsilons]
        ax.plot(epsilons, accs, 'o-', color=colors[task], label=task,
                markersize=8, linewidth=2)
    ax.set_xlabel('Epsilon (nudge strength)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Stem Cell Differentiation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # Panel 3: Entropy comparison
    ax = axes[2]
    x = np.arange(len(stem_names_short))
    ent_vals = [stem_entropies[sn] for sn in stem_configs.keys()]
    bars = ax.bar(x, ent_vals, color=['#E91E63', '#9C27B0', '#3F51B5',
                                       '#607D8B', '#795548'],
                  edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(stem_names_short, fontsize=8, rotation=15)
    ax.set_ylabel('Mean Output Entropy (bits)')
    ax.set_title('Entropy by Stem Type', fontweight='bold')
    for bar, v in zip(bars, ent_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                "%.2f" % v, ha='center', fontsize=9)

    plt.suptitle('Phase 132: The Stem Cell Soul\n'
                 '"At the edge of chaos, all possibilities exist"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase132_stem_cell.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 132, 'name': 'stem_cell_soul',
        'layer': LAYER,
        'stem_results': {k: {t: round(v, 4) for t, v in sv.items()}
                         for k, sv in stem_results.items()},
        'stem_entropies': {k: round(v, 4) for k, v in stem_entropies.items()},
        'differentiation': {t: {str(e): round(v, 4) for e, v in ev.items()}
                           for t, ev in diff_results.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase132_stem_cell.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
