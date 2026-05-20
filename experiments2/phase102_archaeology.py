# -*- coding: utf-8 -*-
"""
Phase 102: Soul Archaeology (Reverse Engineering)
Given a trained program vector, can we determine what it does WITHOUT running it?
Analyze vector properties (norm, PCA coordinates, cluster membership) to
predict function. A form of static analysis for neural programs.

"We can read the soul without executing it."

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compile_prog(model, tok, train, layer, device, seed=42, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device)*0.01; vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(device)
            def inj(m,i,o,v=vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

def main():
    print("[P102] Soul Archaeology (Reverse Engineering)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    tl = 8
    for p in model.parameters(): p.requires_grad = False

    tasks = {
        'MIN': {'train': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                          ("4, 6) =","4"),("9, 3) =","3")]},
        'MAX': {'train': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                          ("4, 6) =","6"),("9, 3) =","9")]},
        'FIRST': {'train': [("3, 7) =","3"),("5, 2) =","5"),("8, 1) =","8"),
                            ("4, 6) =","4"),("9, 3) =","9")]},
        'LAST': {'train': [("3, 7) =","7"),("5, 2) =","2"),("8, 1) =","1"),
                           ("4, 6) =","6"),("9, 3) =","3")]},
    }

    # Step 1: Generate corpus of labeled souls
    print("  Step 1: Generating labeled soul corpus...")
    souls = []
    labels = []
    for task_name, task_data in tasks.items():
        for seed in range(8):
            v = compile_prog(model, tok, task_data['train'], tl, DEVICE,
                           seed=seed*100+hash(task_name)%1000, epochs=80)
            souls.append(v.cpu().numpy().flatten())
            labels.append(task_name)
            print(f"    {task_name} seed={seed}: norm={v.norm().item():.2f}")

    X = np.array(souls)  # (32, 896)
    label_ids = [list(tasks.keys()).index(l) for l in labels]

    # Step 2: PCA analysis
    print("\n  Step 2: PCA projection...")
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    print(f"    Top 3 variance explained: {explained[:3]}")
    print(f"    Cumulative (5 PC): {sum(explained[:5]):.2%}")

    # Step 3: K-Means clustering
    print("\n  Step 3: K-Means clustering (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca[:, :5])

    # Evaluate clustering quality
    from collections import Counter
    cluster_purity = 0
    for c in range(4):
        mask = clusters == c
        if mask.sum() == 0: continue
        counts = Counter(np.array(labels)[mask])
        cluster_purity += counts.most_common(1)[0][1]
    cluster_purity /= len(labels)
    print(f"    Cluster purity: {cluster_purity:.0%}")

    # Step 4: Can we predict task from vector properties?
    print("\n  Step 4: Feature-based classification...")
    norms = np.linalg.norm(X, axis=1)
    # Use simple features: norm + top-5 PCA
    features = np.column_stack([norms, X_pca[:, :5]])

    # Leave-one-out classification using nearest neighbor
    correct = 0
    for i in range(len(features)):
        test_f = features[i]
        test_l = label_ids[i]
        dists = np.linalg.norm(features - test_f, axis=1)
        dists[i] = np.inf
        nearest = np.argmin(dists)
        if label_ids[nearest] == test_l: correct += 1
    nn_acc = correct / len(features)
    print(f"    Nearest-neighbor accuracy: {nn_acc:.0%}")

    # 3-NN
    correct_3nn = 0
    for i in range(len(features)):
        test_f = features[i]
        test_l = label_ids[i]
        dists = np.linalg.norm(features - test_f, axis=1)
        dists[i] = np.inf
        nearest_3 = np.argsort(dists)[:3]
        votes = Counter([label_ids[j] for j in nearest_3])
        if votes.most_common(1)[0][0] == test_l: correct_3nn += 1
    nn3_acc = correct_3nn / len(features)
    print(f"    3-NN accuracy: {nn3_acc:.0%}")

    # Step 5: Norm statistics per task
    print("\n  Step 5: Norm statistics by task...")
    norm_stats = {}
    for task_name in tasks:
        task_norms = norms[[i for i, l in enumerate(labels) if l == task_name]]
        norm_stats[task_name] = {
            'mean': round(float(task_norms.mean()), 4),
            'std': round(float(task_norms.std()), 4),
        }
        print(f"    {task_name}: mean={task_norms.mean():.3f}, std={task_norms.std():.3f}")

    # Save
    output = {
        'phase': 102, 'name': 'soul_archaeology',
        'pca_variance': [round(float(v), 4) for v in explained[:10]],
        'cluster_purity': round(float(cluster_purity), 4),
        'nn_accuracy': round(float(nn_acc), 4),
        'nn3_accuracy': round(float(nn3_acc), 4),
        'norm_stats': norm_stats,
        'num_souls': len(souls),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase102_archaeology.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PCA scatter
    task_names = list(tasks.keys())
    colors_map = {'MIN': 'tab:blue', 'MAX': 'tab:red',
                  'FIRST': 'tab:green', 'LAST': 'tab:orange'}
    for tn in task_names:
        mask = [i for i, l in enumerate(labels) if l == tn]
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_map[tn],
                       label=tn, s=60, edgecolors='black', alpha=0.8)
    axes[0].set_xlabel(f'PC1 ({explained[0]:.0%})')
    axes[0].set_ylabel(f'PC2 ({explained[1]:.0%})')
    axes[0].set_title('Soul Space (PCA)', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Classification accuracy
    methods = ['1-NN', '3-NN', 'Cluster\nPurity', 'Random\nBaseline']
    accs = [nn_acc, nn3_acc, cluster_purity, 0.25]
    colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:gray']
    axes[1].bar(methods, accs, color=colors, edgecolor='black')
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.2)
    axes[1].set_title('Static Analysis Accuracy', fontweight='bold')
    for i, v in enumerate(accs):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Norm distributions
    for tn in task_names:
        task_norms = norms[[i for i, l in enumerate(labels) if l == tn]]
        axes[2].hist(task_norms, bins=8, alpha=0.5, label=tn,
                    color=colors_map[tn], edgecolor='black')
    axes[2].set_xlabel('Vector Norm'); axes[2].set_ylabel('Count')
    axes[2].set_title('Norm Distribution by Task', fontweight='bold')
    axes[2].legend()

    plt.suptitle('Phase 102: Soul Archaeology\n'
                 '"We can read the soul without executing it"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase102_archaeology.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
