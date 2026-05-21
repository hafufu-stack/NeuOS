# -*- coding: utf-8 -*-
"""
Phase 130: Soul Phylogenetic Tree
Train many souls with different seeds and build a phylogenetic tree.
Which souls are related? Do functional clusters emerge?

"Evolution left its fingerprints in the geometry of soul space."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
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
    print("[P130] Soul Phylogenetic Tree")
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

    test_data = {
        'MIN': [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")],
        'MAX': [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")],
        'ADD': [("1, 2) =","3"),("3, 4) =","7"),("2, 5) =","7")],
        'SUB': [("9, 5) =","4"),("6, 1) =","5"),("4, 3) =","1")],
    }

    # Train 10 souls per task with different seeds
    n_seeds = 10
    seeds = list(range(42, 42 + n_seeds))
    task_names = list(task_data.keys())

    all_vecs = []
    all_labels = []
    all_accs = []
    all_seed_labels = []

    print("  Training %d souls x %d tasks = %d total..." % (
        n_seeds, len(task_names), n_seeds * len(task_names)))

    for task in task_names:
        for seed in seeds:
            vec = train_soul(model, tok, task_data[task], DEVICE, seed=seed)
            acc = evaluate(model, tok, vec, test_data[task], DEVICE)
            all_vecs.append(vec.cpu().numpy())
            all_labels.append(task)
            all_accs.append(acc)
            all_seed_labels.append("%s_s%d" % (task, seed))
        print("    %s: %d souls trained, mean test acc = %.0f%%" % (
            task, n_seeds, np.mean(all_accs[-n_seeds:]) * 100))

    X = np.array(all_vecs)

    # Compute pairwise distances
    dists = pdist(X, metric='cosine')

    # Hierarchical clustering
    Z = linkage(dists, method='ward')

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Within-task and cross-task cosine similarities
    cosine_sim = 1 - pdist(X, metric='cosine')
    n = len(all_labels)
    within_cos = {t: [] for t in task_names}
    cross_cos = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if all_labels[i] == all_labels[j]:
                within_cos[all_labels[i]].append(cosine_sim[idx])
            else:
                cross_cos.append(cosine_sim[idx])
            idx += 1

    within_means = {t: float(np.mean(v)) for t, v in within_cos.items()}
    cross_mean = float(np.mean(cross_cos))

    print("\n  Within-task cosine similarities:")
    for t in task_names:
        print("    %s: %.4f" % (t, within_means[t]))
    print("  Cross-task mean: %.4f" % cross_mean)

    # Plot: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {'MIN': '#2196F3', 'MAX': '#FF5722', 'ADD': '#4CAF50', 'SUB': '#9C27B0'}

    # Panel 1: Dendrogram
    ax = axes[0]
    # Color leaves by task
    leaf_colors = {}
    for i, label in enumerate(all_labels):
        leaf_colors[i] = colors[label]

    def color_func(k):
        if k < n:
            return colors[all_labels[k]]
        return 'gray'

    dendro = dendrogram(Z, ax=ax, labels=all_seed_labels,
                       leaf_rotation=90, leaf_font_size=6,
                       color_threshold=0)  # no auto-coloring
    ax.set_title('Soul Phylogenetic Tree', fontweight='bold')
    ax.set_ylabel('Distance (Ward)')
    # Color tick labels
    xlbls = ax.get_xticklabels()
    for lbl in xlbls:
        task = lbl.get_text().split('_')[0]
        lbl.set_color(colors.get(task, 'black'))

    # Panel 2: PCA scatter
    ax = axes[1]
    for task in task_names:
        idx = [i for i, l in enumerate(all_labels) if l == task]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                  c=colors[task], s=80, alpha=0.8,
                  edgecolors='black', linewidths=0.5,
                  label='%s (cos=%.3f)' % (task, within_means[task]))
    ax.set_xlabel('PC1 (%.1f%%)' % (pca.explained_variance_ratio_[0]*100))
    ax.set_ylabel('PC2 (%.1f%%)' % (pca.explained_variance_ratio_[1]*100))
    ax.set_title('Soul Space (PCA)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Panel 3: Within vs Cross cosine bar chart
    ax = axes[2]
    x = np.arange(len(task_names) + 1)
    vals = [within_means[t] for t in task_names] + [cross_mean]
    bar_colors = [colors[t] for t in task_names] + ['gray']
    bars = ax.bar(x, vals, color=bar_colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names + ['Cross'], fontsize=9)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Within vs Cross-Task Similarity', fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                "%.3f" % v, ha='center', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.suptitle('Phase 130: Soul Phylogenetic Tree\n'
                 '"Evolution left its fingerprints in the geometry of soul space"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase130_phylogenetic.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    output = {
        'phase': 130, 'name': 'soul_phylogenetic',
        'n_seeds': n_seeds, 'layer': LAYER,
        'tasks': task_names,
        'within_cosine': within_means,
        'cross_cosine': round(cross_mean, 4),
        'mean_accuracies': {t: round(float(np.mean(
            [all_accs[i] for i, l in enumerate(all_labels) if l == t])), 4)
            for t in task_names},
        'pca_explained_var': [round(float(v), 4)
                              for v in pca.explained_variance_ratio_[:5]],
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase130_phylogenetic.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
