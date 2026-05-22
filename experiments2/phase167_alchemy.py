# -*- coding: utf-8 -*-
"""
Phase 167: Zero-Shot Skill Alchemy
Combine PCA axes to synthesize NOVEL operations that were never trained.
No data. No gradient. Just algebra on the 7D coordinates.

"Mix the elements. Create new matter from nothing."
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


def evaluate_multi(model, tok, soul_vec, prompts, device, layer=LAYER):
    """Evaluate on prompts and return predictions."""
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
        results.append({'pred': pred, 'entropy': round(entropy, 4)})
    return results


def coords_to_soul(pca, coords, device, n_components=8):
    full = np.zeros(n_components)
    for i, c in enumerate(coords):
        if i < n_components:
            full[i] = c
    v = pca.inverse_transform(full.reshape(1, -1))[0]
    return torch.tensor(v, dtype=torch.float32, device=device)


def main():
    print("[P167] Zero-Shot Skill Alchemy")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train base souls for PCA
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]

    souls = {}
    for seed in [42, 100, 200, 300]:
        souls['MIN_s%d' % seed] = train_soul(model, tok, min_data, DEVICE, seed=seed)
        souls['MAX_s%d' % seed] = train_soul(model, tok, max_data, DEVICE, seed=seed)

    # Build PCA
    matrix = np.array([v.cpu().numpy() for v in souls.values()])
    pca = PCA(n_components=8)
    pca.fit(matrix)

    # Get known coordinates
    min_coords = pca.transform(souls['MIN_s42'].cpu().numpy().reshape(1, -1))[0]
    max_coords = pca.transform(souls['MAX_s42'].cpu().numpy().reshape(1, -1))[0]

    # Test prompts with ground truth for multiple operations
    prompts = ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) =", "9, 3) =",
               "7, 4) =", "2, 8) =", "6, 1) ="]
    ground_truth = {
        'MIN':    ['3', '2', '1', '4', '3', '4', '2', '1'],
        'MAX':    ['7', '5', '8', '6', '9', '7', '8', '6'],
        'FIRST':  ['3', '5', '8', '4', '9', '7', '2', '6'],
        'SECOND': ['7', '2', '1', '6', '3', '4', '8', '1'],
        'DIFF':   ['4', '3', '7', '2', '6', '3', '6', '5'],  # |a-b|
        'MEAN':   ['5', '4', '5', '5', '6', '6', '5', '4'],  # round((a+b)/2)
    }

    # Alchemy recipes: combine coordinates to try to create novel operations
    recipes = {
        # 1. Overclock existing operations
        'MIN_2x': {'coords': (min_coords * 2.0).tolist(),
                   'description': 'MIN overclocked 2x'},
        'MAX_2x': {'coords': (max_coords * 2.0).tolist(),
                   'description': 'MAX overclocked 2x'},

        # 2. Midpoint alchemy
        'midpoint': {'coords': ((min_coords + max_coords) / 2.0).tolist(),
                     'description': '(MIN + MAX) / 2 = ???'},

        # 3. Difference direction
        'diff_direction': {'coords': (max_coords - min_coords).tolist(),
                          'description': 'MAX - MIN direction'},
        'anti_diff': {'coords': (min_coords - max_coords).tolist(),
                     'description': 'MIN - MAX direction'},

        # 4. Pure axis experiments
        'only_PC0_pos': {'coords': [3.0, 0, 0, 0, 0, 0, 0, 0],
                         'description': 'PC0=+3 only'},
        'only_PC1_pos': {'coords': [0, 3.0, 0, 0, 0, 0, 0, 0],
                         'description': 'PC1=+3 only'},
        'only_PC5_neg': {'coords': [0, 0, 0, 0, 0, -3.0, 0, 0],
                         'description': 'PC5=-3 only (strong MIN)'},
        'only_PC5_pos': {'coords': [0, 0, 0, 0, 0, 3.0, 0, 0],
                         'description': 'PC5=+3 only (strong MAX)'},

        # 5. Novel combinations: attempt MEAN, DIFF, etc.
        'attempt_mean': {'coords': [0, 0, 1.5, 1.5, 0, 0, 0, 0],
                        'description': 'PC2+PC3 = mean attempt'},
        'attempt_diff': {'coords': [0, 2.0, 0, 0, 2.0, 0, 0, 0],
                        'description': 'PC1+PC4 = diff attempt'},
        'all_positive': {'coords': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0],
                        'description': 'All axes at +1.0'},
        'all_negative': {'coords': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0],
                        'description': 'All axes at -1.0'},

        # 6. Orthogonal to MIN-MAX plane
        'orthogonal': {'coords': [0, 0, 2.0, 0, 2.0, 0, 0, 0],
                      'description': 'Orthogonal to MIN-MAX plane'},
    }

    # Run all recipes
    print("\n  === ALCHEMY RESULTS ===")
    recipe_results = {}
    for name, recipe in recipes.items():
        soul = coords_to_soul(pca, recipe['coords'], DEVICE)
        preds_raw = evaluate_multi(model, tok, soul, prompts, DEVICE)
        preds = [p['pred'] for p in preds_raw]
        avg_entropy = np.mean([p['entropy'] for p in preds_raw])

        # Score against all known operations
        scores = {}
        for op_name, expected in ground_truth.items():
            match = sum(1 for p, e in zip(preds, expected) if p == e)
            scores[op_name] = round(match / len(preds), 4)

        best_op = max(scores, key=scores.get)
        best_score = scores[best_op]

        recipe_results[name] = {
            'description': recipe['description'],
            'predictions': preds,
            'scores': scores,
            'best_op': best_op,
            'best_score': best_score,
            'avg_entropy': round(avg_entropy, 4),
        }
        print("  %-20s: %s(%.0f%%) H=%.2f preds=%s" % (
            name[:20], best_op, best_score*100, avg_entropy,
            ','.join(preds[:5])))

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Best match per recipe
    ax = axes[0]
    rnames = list(recipe_results.keys())
    r_scores = [recipe_results[r]['best_score'] for r in rnames]
    r_ops = [recipe_results[r]['best_op'] for r in rnames]
    op_colors = {'MIN': '#E91E63', 'MAX': '#2196F3', 'FIRST': '#4CAF50',
                 'SECOND': '#FF9800', 'DIFF': '#9C27B0', 'MEAN': '#009688'}
    bar_colors = [op_colors.get(op, '#9E9E9E') for op in r_ops]
    bars = ax.barh(range(len(rnames)), r_scores, color=bar_colors, edgecolor='black')
    ax.set_yticks(range(len(rnames)))
    ax.set_yticklabels([r[:18] for r in rnames], fontsize=7)
    ax.set_xlabel('Best Match Score')
    ax.set_title('Alchemy Results\n(what did we create?)', fontweight='bold')
    ax.set_xlim(0, 1.1)
    for i, (bar, op) in enumerate(zip(bars, r_ops)):
        ax.text(bar.get_width() + 0.02, i, op, va='center', fontsize=8, fontweight='bold')

    # Panel 2: Score breakdown for top recipes
    ax = axes[1]
    top_recipes = ['MIN_2x', 'MAX_2x', 'midpoint', 'diff_direction', 'only_PC5_neg']
    ops = list(ground_truth.keys())
    x = np.arange(len(top_recipes))
    width = 0.12
    for oi, op in enumerate(ops[:4]):
        vals = [recipe_results[r]['scores'].get(op, 0) for r in top_recipes]
        color = op_colors.get(op, '#9E9E9E')
        ax.bar(x + oi * width, vals, width, label=op, color=color, edgecolor='black')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([r[:12] for r in top_recipes], fontsize=8)
    ax.set_ylabel('Match Score')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Score Breakdown\n(top 5 recipes)', fontweight='bold')
    ax.set_ylim(0, 1.1)

    # Panel 3: Entropy landscape
    ax = axes[2]
    entropies = [recipe_results[r]['avg_entropy'] for r in rnames]
    colors_e = [op_colors.get(recipe_results[r]['best_op'], '#9E9E9E') for r in rnames]
    ax.scatter(range(len(rnames)), entropies, c=colors_e, s=100,
               edgecolors='black', linewidths=1)
    for i, (name, ent) in enumerate(zip(rnames, entropies)):
        ax.annotate(name[:10], (i, ent), textcoords="offset points",
                   xytext=(0, 8), ha='center', fontsize=6, rotation=45)
    ax.set_ylabel('Average Entropy')
    ax.set_title('Entropy Landscape\n(lower = more confident)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 167: Zero-Shot Skill Alchemy\n'
                 '"Mix the elements. Create new matter from nothing."',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase167_alchemy.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 167, 'name': 'zero_shot_alchemy',
        'n_recipes': len(recipes),
        'recipe_results': recipe_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase167_alchemy.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P167 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
