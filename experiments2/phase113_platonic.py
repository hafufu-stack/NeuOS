# -*- coding: utf-8 -*-
"""
Phase 113: The Platonic Form (Functional Equivalence Classes)
Train many vectors (different seeds) for the same task.
Do they form a manifold? What is its intrinsic dimensionality?

"Many bodies, one soul -- the form is invariant."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compile_prog(model, tok, train, layer, device, seed=42, epochs=80):
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

def evaluate_vec(model, tok, vec, data, layer, device):
    c = 0
    for p, e in data:
        def inj(m,i,o,v=vec): return replace_last_token(o,v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
    return c / len(data)

def mle_intrinsic_dim(X, k=5):
    """MLE intrinsic dimensionality (Levina & Bickel 2005)."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, _ = nn.kneighbors(X)
    dists = dists[:, 1:]
    dims = []
    for i in range(len(X)):
        d = dists[i]; d = d[d > 0]
        if len(d) < 2: continue
        T_k = d[-1]
        log_ratios = np.log(T_k / d[:-1])
        s = np.sum(log_ratios)
        if s > 0: dims.append(len(log_ratios) / s)
    return np.mean(dims) if dims else 0.0

def main():
    print("[P113] The Platonic Form (Functional Equivalence Classes)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    tl = 8
    for p in model.parameters(): p.requires_grad = False

    min_train = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")]
    max_train = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")]
    all_min = min_train + [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")]
    all_max = max_train + [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")]

    N_SEEDS = 30
    print(f"  Training {N_SEEDS} MIN vectors...")
    min_vecs, min_accs = [], []
    for s in range(N_SEEDS):
        v = compile_prog(model, tok, min_train, tl, DEVICE, seed=s*37, epochs=80)
        acc = evaluate_vec(model, tok, v, all_min, tl, DEVICE)
        min_vecs.append(v.cpu().numpy()); min_accs.append(acc)
        if (s+1) % 10 == 0:
            print(f"    {s+1}/{N_SEEDS}, avg acc={np.mean(min_accs[-10:]):.2f}")

    print(f"  Training {N_SEEDS} MAX vectors...")
    max_vecs, max_accs = [], []
    for s in range(N_SEEDS):
        v = compile_prog(model, tok, max_train, tl, DEVICE, seed=s*37+1000, epochs=80)
        acc = evaluate_vec(model, tok, v, all_max, tl, DEVICE)
        max_vecs.append(v.cpu().numpy()); max_accs.append(acc)
        if (s+1) % 10 == 0:
            print(f"    {s+1}/{N_SEEDS}, avg acc={np.mean(max_accs[-10:]):.2f}")

    X_min = np.array(min_vecs); X_max = np.array(max_vecs)
    X_all = np.concatenate([X_min, X_max])

    # Cosine similarity analysis
    def cosine_matrix(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / (norms + 1e-8)
        return Xn @ Xn.T

    cos_min = cosine_matrix(X_min)
    cos_max = cosine_matrix(X_max)
    mn = X_min / (np.linalg.norm(X_min, axis=1, keepdims=True)+1e-8)
    mx = X_max / (np.linalg.norm(X_max, axis=1, keepdims=True)+1e-8)
    cos_cross = mn @ mx.T
    w_min = cos_min[np.triu_indices(N_SEEDS, k=1)]
    w_max = cos_max[np.triu_indices(N_SEEDS, k=1)]
    w_cross = cos_cross.flatten()

    print(f"  Within-MIN cos: {w_min.mean():.4f} +/- {w_min.std():.4f}")
    print(f"  Within-MAX cos: {w_max.mean():.4f} +/- {w_max.std():.4f}")
    print(f"  Cross cos: {w_cross.mean():.4f} +/- {w_cross.std():.4f}")

    # PCA
    pca = PCA(n_components=min(20, N_SEEDS))
    pca.fit(X_all)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.9)) + 1
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1

    # MLE intrinsic dim
    id_min = mle_intrinsic_dim(X_min, k=min(5, N_SEEDS-2))
    id_max = mle_intrinsic_dim(X_max, k=min(5, N_SEEDS-2))
    id_all = mle_intrinsic_dim(X_all, k=min(5, N_SEEDS*2-2))
    print(f"  PCA 90%={n_90}d, 95%={n_95}d; MLE: MIN={id_min:.1f}, MAX={id_max:.1f}")

    # t-SNE
    X_2d = TSNE(n_components=2, perplexity=min(15, N_SEEDS-1),
                random_state=42).fit_transform(X_all)

    output = {
        'phase': 113, 'name': 'platonic_form', 'n_seeds': N_SEEDS,
        'min_acc_mean': round(float(np.mean(min_accs)), 4),
        'max_acc_mean': round(float(np.mean(max_accs)), 4),
        'within_min_cos': round(float(w_min.mean()), 4),
        'within_max_cos': round(float(w_max.mean()), 4),
        'cross_cos': round(float(w_cross.mean()), 4),
        'pca_90_dims': int(n_90), 'pca_95_dims': int(n_95),
        'mle_dim_min': round(float(id_min), 2),
        'mle_dim_max': round(float(id_max), 2),
        'mle_dim_all': round(float(id_all), 2),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase113_platonic.json'), 'w') as f:
        json.dump(output, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (lab, c, m) in enumerate([('MIN','tab:blue','o'),('MAX','tab:red','s')]):
        idx = slice(i*N_SEEDS, (i+1)*N_SEEDS)
        axes[0].scatter(X_2d[idx,0], X_2d[idx,1], c=c, marker=m, s=40,
                       label=lab, edgecolors='black', alpha=0.7)
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
    axes[0].set_title('Equivalence Classes (each point = different seed)', fontweight='bold')
    axes[0].legend()

    axes[1].hist(w_min, bins=25, alpha=0.6, color='tab:blue', label='Within MIN', density=True)
    axes[1].hist(w_max, bins=25, alpha=0.6, color='tab:red', label='Within MAX', density=True)
    axes[1].hist(w_cross, bins=25, alpha=0.6, color='gray', label='Cross', density=True)
    axes[1].set_xlabel('Cosine Similarity'); axes[1].set_ylabel('Density')
    axes[1].set_title('Pairwise Cosine Distributions', fontweight='bold')
    axes[1].legend(fontsize=8)

    axes[2].plot(range(1, len(cumvar)+1), cumvar, 'k-o', lw=2, ms=4)
    axes[2].axhline(y=0.9, color='red', ls='--', label='90%')
    axes[2].axhline(y=0.95, color='orange', ls='--', label='95%')
    axes[2].set_xlabel('PCs'); axes[2].set_ylabel('Cumulative Variance')
    axes[2].set_title(f'Dimensionality (MLE: MIN={id_min:.1f}, MAX={id_max:.1f})',
                      fontweight='bold')
    axes[2].legend()

    plt.suptitle('Phase 113: The Platonic Form\n"Many bodies, one soul"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase113_platonic.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
