# -*- coding: utf-8 -*-
"""
Phase 169: 7D Gradient-Free Optimization
Since souls live in 7D, we can find optimal coordinates via
systematic grid search -- no gradient descent needed!

"Brute-force the mind. 7 dimensions is small enough."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import product
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8


def train_soul(model, tok, data, device, layer=LAYER, epochs=100, seed=42):
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


def coords_to_soul(pca, coords, device, n_components=8):
    full = np.zeros(n_components)
    for i, c in enumerate(coords):
        if i < n_components:
            full[i] = c
    v = pca.inverse_transform(full.reshape(1, -1))[0]
    return torch.tensor(v, dtype=torch.float32, device=device)


def evaluate_fast(model, tok, soul_vec, test_data, device, layer=LAYER):
    correct = 0
    for prompt, expected in test_data:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data) if test_data else 0


def main():
    print("[P169] 7D Gradient-Free Optimization")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Build PCA basis from trained souls
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("1, 5) =","1"),("8, 4) =","4")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("1, 5) =","5"),("8, 4) =","8")]

    souls = {}
    for seed in [42, 100, 200, 300]:
        souls['MIN_s%d' % seed] = train_soul(model, tok, min_data, DEVICE, seed=seed)
        souls['MAX_s%d' % seed] = train_soul(model, tok, max_data, DEVICE, seed=seed)

    matrix = np.array([v.cpu().numpy() for v in souls.values()])
    pca = PCA(n_components=8)
    pca.fit(matrix)

    # Get trained soul coordinates as reference
    min_trained_coords = pca.transform(souls['MIN_s42'].cpu().numpy().reshape(1, -1))[0]
    max_trained_coords = pca.transform(souls['MAX_s42'].cpu().numpy().reshape(1, -1))[0]

    # Baseline
    min_baseline = evaluate_fast(model, tok, souls['MIN_s42'], min_test, DEVICE)
    max_baseline = evaluate_fast(model, tok, souls['MAX_s42'], max_test, DEVICE)
    print("  Baselines: MIN=%.0f%%, MAX=%.0f%%" % (min_baseline*100, max_baseline*100))

    results = {}

    # Strategy 1: 2D Grid search on the two most important axes (PC1, PC5)
    # PC1 = MAX/MIN axis, PC5 = FIRST/SECOND axis (which is also MIN/MAX discriminator)
    print("\n  === Strategy 1: 2D Grid Search (PC1 x PC5) ===")
    grid_values = np.linspace(-3, 3, 13)  # 13 points per axis = 169 combos
    grid_results_min = np.zeros((len(grid_values), len(grid_values)))
    grid_results_max = np.zeros((len(grid_values), len(grid_values)))

    best_min = {'acc': 0, 'coords': None}
    best_max = {'acc': 0, 'coords': None}

    for i, pc1_val in enumerate(grid_values):
        for j, pc5_val in enumerate(grid_values):
            coords = [0, pc1_val, 0, 0, 0, pc5_val, 0, 0]
            soul = coords_to_soul(pca, coords, DEVICE)

            min_acc = evaluate_fast(model, tok, soul, min_test, DEVICE)
            max_acc = evaluate_fast(model, tok, soul, max_test, DEVICE)

            grid_results_min[i, j] = min_acc
            grid_results_max[i, j] = max_acc

            if min_acc > best_min['acc']:
                best_min = {'acc': min_acc, 'coords': coords[:], 'pc1': pc1_val, 'pc5': pc5_val}
            if max_acc > best_max['acc']:
                best_max = {'acc': max_acc, 'coords': coords[:], 'pc1': pc1_val, 'pc5': pc5_val}

        if (i + 1) % 4 == 0:
            print("  Row %d/%d done (best MIN=%.0f%%, MAX=%.0f%%)" % (
                i+1, len(grid_values), best_min['acc']*100, best_max['acc']*100))

    print("  Grid search best MIN: %.0f%% at PC1=%.1f, PC5=%.1f" % (
        best_min['acc']*100, best_min['pc1'], best_min['pc5']))
    print("  Grid search best MAX: %.0f%% at PC1=%.1f, PC5=%.1f" % (
        best_max['acc']*100, best_max['pc1'], best_max['pc5']))

    results['grid_2d'] = {
        'best_min': {'acc': round(best_min['acc'], 4),
                     'pc1': float(best_min['pc1']), 'pc5': float(best_min['pc5'])},
        'best_max': {'acc': round(best_max['acc'], 4),
                     'pc1': float(best_max['pc1']), 'pc5': float(best_max['pc5'])},
    }

    # Strategy 2: Refine around the best point found
    print("\n  === Strategy 2: Local Refinement ===")
    refine_range = 0.5
    refine_steps = 5
    for target, best, test_data, name in [
        ('MIN', best_min, min_test, 'min'),
        ('MAX', best_max, max_test, 'max'),
    ]:
        best_refined = best.copy()
        for pc_idx, pc_name in [(1, 'PC1'), (5, 'PC5')]:
            center = best['coords'][pc_idx]
            for delta in np.linspace(-refine_range, refine_range, refine_steps * 2 + 1):
                coords = best['coords'][:]
                coords[pc_idx] = center + delta
                soul = coords_to_soul(pca, coords, DEVICE)
                acc = evaluate_fast(model, tok, soul, test_data, DEVICE)
                if acc > best_refined['acc']:
                    best_refined = {'acc': acc, 'coords': coords[:],
                                   'pc1': coords[1], 'pc5': coords[5]}

        print("  Refined %s: %.0f%% at PC1=%.2f, PC5=%.2f" % (
            target, best_refined['acc']*100, best_refined['pc1'], best_refined['pc5']))
        results['refined_%s' % name] = {
            'acc': round(best_refined['acc'], 4),
            'pc1': float(best_refined['pc1']),
            'pc5': float(best_refined['pc5']),
        }

    # Strategy 3: Compare gradient vs gradient-free
    print("\n  === Strategy 3: Gradient vs Gradient-Free ===")
    comparison = {
        'gradient_trained': {
            'min_acc': round(min_baseline, 4), 'max_acc': round(max_baseline, 4),
            'method': '100 epochs Adam',
        },
        'grid_search': {
            'min_acc': round(best_min['acc'], 4), 'max_acc': round(best_max['acc'], 4),
            'method': '169 grid points (0 gradient)',
        },
        'coordinate_direct': {
            'min_acc': None, 'max_acc': None,
            'method': 'P166 combined coords',
        },
    }
    # Test P166's combined coords
    soul_combined_min = coords_to_soul(pca, [0, -1.5, 0, 0, 0, -1.5, 0, 0], DEVICE)
    soul_combined_max = coords_to_soul(pca, [0, 1.5, 0, 0, 0, 1.5, 0, 0], DEVICE)
    comparison['coordinate_direct']['min_acc'] = round(
        evaluate_fast(model, tok, soul_combined_min, min_test, DEVICE), 4)
    comparison['coordinate_direct']['max_acc'] = round(
        evaluate_fast(model, tok, soul_combined_max, max_test, DEVICE), 4)

    for method, r in comparison.items():
        print("  %-20s: MIN=%.0f%% MAX=%.0f%% (%s)" % (
            method, (r['min_acc'] or 0)*100, (r['max_acc'] or 0)*100, r['method']))
    results['comparison'] = comparison

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: MIN accuracy heatmap
    ax = axes[0]
    im = ax.imshow(grid_results_min, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1, origin='lower',
                   extent=[grid_values[0], grid_values[-1],
                           grid_values[0], grid_values[-1]])
    ax.set_xlabel('PC5 (FIRST <-> SECOND)')
    ax.set_ylabel('PC1 (MIN <-> MAX)')
    ax.set_title('MIN Accuracy Heatmap\n(2D grid search)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Mark best point
    ax.plot(best_min['pc5'], best_min['pc1'], 'r*', markersize=15, markeredgecolor='black')
    # Mark trained soul location
    ax.plot(min_trained_coords[5], min_trained_coords[1], 'ws', markersize=10,
            markeredgecolor='black', label='Trained MIN')
    ax.legend(fontsize=8)

    # Panel 2: MAX accuracy heatmap
    ax = axes[1]
    im = ax.imshow(grid_results_max, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1, origin='lower',
                   extent=[grid_values[0], grid_values[-1],
                           grid_values[0], grid_values[-1]])
    ax.set_xlabel('PC5 (FIRST <-> SECOND)')
    ax.set_ylabel('PC1 (MIN <-> MAX)')
    ax.set_title('MAX Accuracy Heatmap\n(2D grid search)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.plot(best_max['pc5'], best_max['pc1'], 'b*', markersize=15, markeredgecolor='black')
    ax.plot(max_trained_coords[5], max_trained_coords[1], 'ws', markersize=10,
            markeredgecolor='black', label='Trained MAX')
    ax.legend(fontsize=8)

    # Panel 3: Method comparison
    ax = axes[2]
    methods = ['Gradient\n(100 epochs)', 'Grid Search\n(169 pts)', 'Direct\nCoords']
    min_vals = [comparison['gradient_trained']['min_acc'],
                comparison['grid_search']['min_acc'],
                comparison['coordinate_direct']['min_acc']]
    max_vals = [comparison['gradient_trained']['max_acc'],
                comparison['grid_search']['max_acc'],
                comparison['coordinate_direct']['max_acc']]
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w/2, min_vals, w, label='MIN', color='#E91E63', edgecolor='black')
    ax.bar(x + w/2, max_vals, w, label='MAX', color='#2196F3', edgecolor='black')
    for i in range(len(methods)):
        ax.text(i - w/2, min_vals[i] + 0.02, '%.0f%%' % (min_vals[i]*100),
                ha='center', fontsize=10, fontweight='bold')
        ax.text(i + w/2, max_vals[i] + 0.02, '%.0f%%' % (max_vals[i]*100),
                ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_title('Gradient vs Gradient-Free\n(training data vs coordinates)', fontweight='bold')

    plt.suptitle('Phase 169: 7D Gradient-Free Optimization\n'
                 '"Brute-force the mind. 7 dimensions is small enough."',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase169_grid_search.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 169, 'name': '7d_gradient_free_optimization',
        'grid_shape': [len(grid_values), len(grid_values)],
        'total_evaluations': len(grid_values) ** 2,
        'results': results,
        'trained_coords': {
            'min': [round(c, 4) for c in min_trained_coords.tolist()],
            'max': [round(c, 4) for c in max_trained_coords.tolist()],
        },
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase169_grid_search.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P169 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
