# -*- coding: utf-8 -*-
"""
Phase 161: Soul Compression
How many dimensions of the 896-dim soul vector are actually needed?
PCA-compress souls and test if they still work.

"How small can a soul be and still be a soul?"
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


def evaluate(model, tok, soul_vec, test_data, device, layer=LAYER):
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


def compress_soul(soul_vec, pca_model, n_components):
    """Compress a soul vector via PCA: project to n_components, then reconstruct."""
    v = soul_vec.cpu().numpy().reshape(1, -1)
    # Project down
    projected = pca_model.transform(v)[:, :n_components]
    # Zero out remaining components
    full_projected = np.zeros((1, pca_model.n_components_))
    full_projected[0, :n_components] = projected[0]
    # Reconstruct
    reconstructed = pca_model.inverse_transform(full_projected)
    return torch.tensor(reconstructed[0], dtype=soul_vec.dtype, device=soul_vec.device)


def main():
    print("[P161] Soul Compression")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train 4 souls
    print("  Training 4 base souls...")
    tasks = {
        'MIN': {
            'train': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                      ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                      ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                      ("1, 3) =","1")],
            'test':  [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                      ("1, 5) =","1"),("8, 4) =","4")],
        },
        'MAX': {
            'train': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                      ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                      ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                      ("1, 3) =","3")],
            'test':  [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                      ("1, 5) =","5"),("8, 4) =","8")],
        },
    }

    souls = {}
    for name, task in tasks.items():
        souls[name] = train_soul(model, tok, task['train'], DEVICE,
                                  seed=42 if name == 'MIN' else 43)

    # Also create diverse souls by varying seed
    for seed in [100, 200, 300]:
        v = train_soul(model, tok, tasks['MIN']['train'], DEVICE, seed=seed)
        souls['MIN_s%d' % seed] = v
        v = train_soul(model, tok, tasks['MAX']['train'], DEVICE, seed=seed)
        souls['MAX_s%d' % seed] = v

    print("  Collected %d soul vectors" % len(souls))

    # Fit PCA on all souls
    soul_matrix = np.array([v.cpu().numpy() for v in souls.values()])
    n_pca = min(len(souls), soul_matrix.shape[1])
    pca = PCA(n_components=n_pca)
    pca.fit(soul_matrix)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    print("  PCA explained variance (first 5): %s" % ', '.join(
        ['%.4f' % v for v in explained_var[:5]]))
    print("  Cumulative 90%%: %d components" % (
        np.searchsorted(cumulative_var, 0.9) + 1))

    # Test compression at different levels
    test_dims = [1, 2, 3, 4, 5, 6, 7, 8]
    test_dims = [d for d in test_dims if d <= n_pca]

    compression_results = {}
    for task_name in ['MIN', 'MAX']:
        print("\n  === %s ===" % task_name)
        original_soul = souls[task_name]
        original_acc = evaluate(model, tok, original_soul,
                               tasks[task_name]['test'], DEVICE)
        print("  Original (896d): %.0f%%" % (original_acc * 100))

        task_results = {'original_acc': round(original_acc, 4), 'compressed': {}}

        for n_dim in test_dims:
            compressed = compress_soul(original_soul, pca, n_dim)
            comp_acc = evaluate(model, tok, compressed,
                               tasks[task_name]['test'], DEVICE)

            # Reconstruction error
            recon_err = (original_soul - compressed).norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                original_soul.unsqueeze(0), compressed.unsqueeze(0)).item()

            task_results['compressed'][n_dim] = {
                'accuracy': round(comp_acc, 4),
                'recon_error': round(recon_err, 4),
                'cosine': round(cos_sim, 4),
            }
            print("  %dd: acc=%.0f%%, cos=%.4f, err=%.2f" % (
                n_dim, comp_acc * 100, cos_sim, recon_err))

        compression_results[task_name] = task_results

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Compression vs accuracy
    ax = axes[0]
    for task_name, color in [('MIN', '#E91E63'), ('MAX', '#2196F3')]:
        r = compression_results[task_name]
        dims = sorted([int(d) for d in r['compressed'].keys()])
        accs = [r['compressed'][d]['accuracy'] for d in dims]
        ax.plot(dims, accs, 'o-', color=color, linewidth=2, markersize=8,
                label='%s' % task_name)
        ax.axhline(y=r['original_acc'], color=color, linestyle='--', alpha=0.5)
    ax.set_xlabel('PCA Dimensions')
    ax.set_ylabel('Accuracy')
    ax.set_title('Compression vs Accuracy\n(dashed = original 896d)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # Panel 2: PCA explained variance
    ax = axes[1]
    n_show = min(10, len(explained_var))
    ax.bar(range(1, n_show + 1), explained_var[:n_show],
           color='#9C27B0', edgecolor='black', alpha=0.7, label='Individual')
    ax.plot(range(1, n_show + 1), cumulative_var[:n_show],
            'ro-', linewidth=2, label='Cumulative')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax.set_xlabel('PCA Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Soul Space Dimensionality\n(how many dims matter?)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Cosine similarity vs dimensions
    ax = axes[2]
    for task_name, color in [('MIN', '#E91E63'), ('MAX', '#2196F3')]:
        r = compression_results[task_name]
        dims = sorted([int(d) for d in r['compressed'].keys()])
        cos_vals = [r['compressed'][d]['cosine'] for d in dims]
        ax.plot(dims, cos_vals, 's-', color=color, linewidth=2, markersize=8,
                label='%s' % task_name)
    ax.set_xlabel('PCA Dimensions')
    ax.set_ylabel('Cosine Similarity to Original')
    ax.set_title('Reconstruction Fidelity', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.suptitle('Phase 161: Soul Compression\n'
                 '"How small can a soul be and still be a soul?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase161_soul_compression.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 161, 'name': 'soul_compression',
        'n_souls': len(souls),
        'pca_explained_variance': [round(v, 6) for v in explained_var[:10].tolist()],
        'pca_cumulative_90pct': int(np.searchsorted(cumulative_var, 0.9) + 1),
        'compression_results': compression_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase161_soul_compression.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
