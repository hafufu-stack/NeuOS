# -*- coding: utf-8 -*-
"""
Phase 163: Soul Space Cartography
Systematic exploration of the 7-dimensional soul subspace.

Since P161 proved that souls live in a tiny 7D manifold within 896D space,
we can EXHAUSTIVELY MAP this space to discover unknown programs.

"Draw the complete map of all possible minds."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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


def probe_soul(model, tok, soul_vec, prompts, device, layer=LAYER):
    """Run a soul on multiple prompts, return predictions and entropies."""
    results = []
    for prompt in prompts:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        pred = tok.decode(logits.argmax().item()).strip()
        conf = probs.max().item()
        results.append({'pred': pred, 'entropy': entropy, 'conf': conf})
    return results


def classify_behavior(predictions, prompts_info):
    """Classify what operation a soul appears to compute."""
    # Check against known operations
    ops = {
        'MIN': [info['min'] for info in prompts_info],
        'MAX': [info['max'] for info in prompts_info],
        'FIRST': [info['first'] for info in prompts_info],
        'SECOND': [info['second'] for info in prompts_info],
    }
    preds = [p['pred'] for p in predictions]

    scores = {}
    for op_name, expected in ops.items():
        match = sum(1 for p, e in zip(preds, expected) if p == e)
        scores[op_name] = match / len(preds)

    best_op = max(scores, key=scores.get)
    best_score = scores[best_op]

    if best_score < 0.4:
        return 'UNKNOWN', scores, best_score
    return best_op, scores, best_score


def main():
    print("[P163] Soul Space Cartography")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train diverse souls for PCA basis
    print("  Training souls for PCA basis...")
    datasets = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")],
    }

    souls = {}
    for name, data in datasets.items():
        for seed in [42, 100, 200, 300]:
            key = '%s_s%d' % (name, seed)
            souls[key] = train_soul(model, tok, data, DEVICE, seed=seed)

    # Fit PCA
    soul_matrix = np.array([v.cpu().numpy() for v in souls.values()])
    n_pca = min(len(souls), 8)
    pca = PCA(n_components=n_pca)
    pca.fit(soul_matrix)
    mean_vec = pca.mean_

    # Project known souls to PCA space
    known_coords = {}
    for name, vec in souls.items():
        coords = pca.transform(vec.cpu().numpy().reshape(1, -1))[0]
        known_coords[name] = coords.tolist()

    # Probe prompts with ground truth for all operations
    probe_prompts = ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) =", "9, 3) =",
                     "7, 4) =", "2, 8) =", "6, 1) ="]
    prompts_info = [
        {'first': '3', 'second': '7', 'min': '3', 'max': '7'},
        {'first': '5', 'second': '2', 'min': '2', 'max': '5'},
        {'first': '8', 'second': '1', 'min': '1', 'max': '8'},
        {'first': '4', 'second': '6', 'min': '4', 'max': '6'},
        {'first': '9', 'second': '3', 'min': '3', 'max': '9'},
        {'first': '7', 'second': '4', 'min': '4', 'max': '7'},
        {'first': '2', 'second': '8', 'min': '2', 'max': '8'},
        {'first': '6', 'second': '1', 'min': '1', 'max': '6'},
    ]

    # Exploration: sample points in PCA space
    print("\n  Exploring soul space grid...")
    exploration_results = []

    # Strategy 1: Walk along each PCA axis
    for axis in range(min(4, n_pca)):
        for scale in [-3.0, -2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0, 3.0]:
            coords = np.zeros(n_pca)
            coords[axis] = scale
            vec_896 = pca.inverse_transform(coords.reshape(1, -1))[0]
            soul_vec = torch.tensor(vec_896, dtype=torch.float32, device=DEVICE)

            preds = probe_soul(model, tok, soul_vec, probe_prompts, DEVICE)
            behavior, scores, best_score = classify_behavior(preds, prompts_info)
            avg_entropy = np.mean([p['entropy'] for p in preds])

            exploration_results.append({
                'method': 'axis_%d' % axis,
                'scale': scale,
                'coords': coords.tolist(),
                'behavior': behavior,
                'best_score': round(best_score, 4),
                'avg_entropy': round(avg_entropy, 4),
                'predictions': [p['pred'] for p in preds],
                'scores': {k: round(v, 4) for k, v in scores.items()},
            })

        print("  Axis %d: %s" % (axis, 
              ', '.join(['%.1f->%s' % (r['scale'], r['behavior']) 
                        for r in exploration_results if r['method'] == 'axis_%d' % axis])))

    # Strategy 2: Interpolate between known souls
    print("\n  Interpolating between MIN and MAX...")
    min_coord = np.array(known_coords['MIN_s42'])
    max_coord = np.array(known_coords['MAX_s42'])
    interp_results = []

    for alpha in np.linspace(0, 1, 11):
        coords = alpha * min_coord + (1 - alpha) * max_coord
        vec_896 = pca.inverse_transform(coords.reshape(1, -1))[0]
        soul_vec = torch.tensor(vec_896, dtype=torch.float32, device=DEVICE)

        preds = probe_soul(model, tok, soul_vec, probe_prompts, DEVICE)
        behavior, scores, best_score = classify_behavior(preds, prompts_info)
        avg_entropy = np.mean([p['entropy'] for p in preds])

        interp_results.append({
            'alpha': round(alpha, 2),
            'behavior': behavior,
            'best_score': round(best_score, 4),
            'avg_entropy': round(avg_entropy, 4),
            'min_score': round(scores['MIN'], 4),
            'max_score': round(scores['MAX'], 4),
            'predictions': [p['pred'] for p in preds],
        })
        print("  alpha=%.2f: %s (MIN=%.0f%% MAX=%.0f%% H=%.2f)" % (
            alpha, behavior, scores['MIN']*100, scores['MAX']*100, avg_entropy))

    # Strategy 3: Random exploration in PCA space
    print("\n  Random exploration (50 points)...")
    np.random.seed(42)
    random_results = []
    behavior_counts = {}
    for i in range(50):
        coords = np.random.randn(n_pca) * 1.5
        vec_896 = pca.inverse_transform(coords.reshape(1, -1))[0]
        soul_vec = torch.tensor(vec_896, dtype=torch.float32, device=DEVICE)

        preds = probe_soul(model, tok, soul_vec, probe_prompts, DEVICE)
        behavior, scores, best_score = classify_behavior(preds, prompts_info)
        avg_entropy = np.mean([p['entropy'] for p in preds])

        random_results.append({
            'idx': i, 'behavior': behavior,
            'best_score': round(best_score, 4),
            'avg_entropy': round(avg_entropy, 4),
            'coords': coords[:4].tolist(),
        })
        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

    print("  Behavior distribution: %s" % behavior_counts)

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Interpolation MIN <-> MAX in PCA space
    ax = axes[0]
    alphas = [r['alpha'] for r in interp_results]
    min_scores = [r['min_score'] for r in interp_results]
    max_scores = [r['max_score'] for r in interp_results]
    entropies = [r['avg_entropy'] for r in interp_results]
    ax.plot(alphas, min_scores, 'ro-', linewidth=2, markersize=6, label='MIN match')
    ax.plot(alphas, max_scores, 'bs-', linewidth=2, markersize=6, label='MAX match')
    ax2 = ax.twinx()
    ax2.plot(alphas, entropies, 'g^--', linewidth=1.5, markersize=5, label='Entropy', alpha=0.7)
    ax2.set_ylabel('Entropy', color='green')
    ax.set_xlabel('alpha (0=MAX, 1=MIN)')
    ax.set_ylabel('Match Score')
    ax.set_title('PCA-Space Interpolation\n(MIN <-> MAX)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Random exploration behavior distribution
    ax = axes[1]
    labels = sorted(behavior_counts.keys())
    sizes = [behavior_counts[k] for k in labels]
    colors = {'MIN': '#E91E63', 'MAX': '#2196F3', 'FIRST': '#4CAF50',
              'SECOND': '#FF9800', 'UNKNOWN': '#9E9E9E'}
    pie_colors = [colors.get(l, '#9E9E9E') for l in labels]
    ax.pie(sizes, labels=labels, autopct='%1.0f%%', colors=pie_colors,
           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Random Soul Space Exploration\n(50 random points in 7D)',
                 fontweight='bold')

    # Panel 3: 2D PCA projection of known souls + exploration
    ax = axes[2]
    # Known souls
    for name, coords in known_coords.items():
        color = '#E91E63' if 'MIN' in name else '#2196F3'
        marker = 'o' if 's42' in name else 'x'
        size = 150 if 's42' in name else 80
        ax.scatter(coords[0], coords[1], c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=1, zorder=10)
        if 's42' in name:
            ax.annotate(name.split('_')[0], (coords[0], coords[1]),
                       textcoords="offset points", xytext=(10, 5),
                       fontweight='bold', fontsize=11)

    # Random exploration points colored by behavior
    for r in random_results:
        c = r['coords']
        color = colors.get(r['behavior'], '#9E9E9E')
        ax.scatter(c[0], c[1], c=color, s=30, alpha=0.5, edgecolors='none')

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Soul Space Map (2D projection)\nRed=MIN, Blue=MAX, Gray=Unknown',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 163: Soul Space Cartography\n'
                 '"The complete map of all possible minds"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase163_cartography.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 163, 'name': 'soul_space_cartography',
        'n_pca_components': n_pca,
        'known_coords': {k: v[:4] for k, v in known_coords.items()},
        'axis_exploration': exploration_results,
        'interpolation': interp_results,
        'random_behavior_distribution': behavior_counts,
        'n_random_samples': len(random_results),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase163_cartography.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
