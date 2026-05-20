# -*- coding: utf-8 -*-
"""
Phase 119: Soul Vector Compression
Can we compress 896-dim soul vectors to 1 dimension and still compute MIN?

"If translation is rank-1, the soul's essence might be a single number."
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

def train_vec(model, tok, data, layer, device, seed, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in data:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(device)
            def inj(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

def eval_vec(model, tok, vec, data, layer, device):
    c = 0
    for p, e in data:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0, -1, :].argmax().item()).strip() == e: c += 1
    return c / len(data)

def main():
    print("[P119] Soul Vector Compression")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    tl = 8
    for p in model.parameters(): p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 2) =","2"),
                ("6, 3) =","3"),("2, 9) =","2")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 2) =","7"),
                ("6, 3) =","6"),("2, 9) =","9")]

    # Step 1: Train 30 MIN vectors + 30 MAX vectors
    N = 30
    print("  Training %d MIN + %d MAX vectors..." % (N, N))
    min_vecs = []
    max_vecs = []
    for i in range(N):
        v = train_vec(model, tok, min_data[:5], tl, DEVICE, seed=i*41)
        min_vecs.append(v)
        v = train_vec(model, tok, max_data[:5], tl, DEVICE, seed=i*41+1000)
        max_vecs.append(v)
        if (i+1) % 10 == 0:
            print("    %d/%d done" % (i+1, N))

    M_min = torch.stack(min_vecs)  # (30, 896)
    M_max = torch.stack(max_vecs)

    # Step 2: PCA of combined space
    M_all = torch.cat([M_min, M_max], dim=0)  # (60, 896)
    M_np = M_all.cpu().numpy()
    mean = M_np.mean(axis=0)
    M_centered = M_np - mean
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)

    # Step 3: Compress to k dimensions and test
    ks = [1, 2, 4, 8, 16, 32, 64, 128]
    compression_results = {}
    for k in ks:
        print("  Testing k=%d compression..." % k)
        basis = Vt[:k]  # (k, 896) - top-k principal components

        # Project and reconstruct MIN vectors
        min_accs = []
        max_accs = []
        for i in range(N):
            # Compress: project onto k-dim subspace, then reconstruct
            v_min = min_vecs[i].cpu().numpy() - mean
            coords_min = v_min @ basis.T  # (k,)
            v_min_recon = coords_min @ basis + mean
            v_min_t = torch.tensor(v_min_recon, dtype=torch.float32, device=DEVICE)
            acc = eval_vec(model, tok, v_min_t, min_data, tl, DEVICE)
            min_accs.append(acc)

            v_max = max_vecs[i].cpu().numpy() - mean
            coords_max = v_max @ basis.T
            v_max_recon = coords_max @ basis + mean
            v_max_t = torch.tensor(v_max_recon, dtype=torch.float32, device=DEVICE)
            acc = eval_vec(model, tok, v_max_t, max_data, tl, DEVICE)
            max_accs.append(acc)

        compression_results[k] = {
            'min_acc': round(float(np.mean(min_accs)), 4),
            'max_acc': round(float(np.mean(max_accs)), 4),
            'avg_acc': round(float(np.mean(min_accs + max_accs)), 4),
        }
        print("    k=%d: MIN=%.0f%%, MAX=%.0f%%" % (
            k, np.mean(min_accs)*100, np.mean(max_accs)*100))

    # Step 4: Morphing along PC1
    print("  Testing PC1 morphing...")
    pc1 = Vt[0]  # first principal component
    # Project MIN and MAX centroids onto PC1
    min_centroid = M_min.cpu().numpy().mean(axis=0) - mean
    max_centroid = M_max.cpu().numpy().mean(axis=0) - mean
    min_coord = float(min_centroid @ pc1)
    max_coord = float(max_centroid @ pc1)
    print("    MIN centroid PC1 coord: %.3f" % min_coord)
    print("    MAX centroid PC1 coord: %.3f" % max_coord)

    # Sweep from MIN to MAX along PC1
    n_sweep = 11
    sweep_coords = np.linspace(min_coord * 1.5, max_coord * 1.5, n_sweep)
    sweep_results = []
    for coord in sweep_coords:
        v_recon = coord * pc1 + mean
        v_t = torch.tensor(v_recon, dtype=torch.float32, device=DEVICE)
        min_acc = eval_vec(model, tok, v_t, min_data, tl, DEVICE)
        max_acc = eval_vec(model, tok, v_t, max_data, tl, DEVICE)
        sweep_results.append({
            'coord': round(float(coord), 4),
            'min_acc': round(float(min_acc), 4),
            'max_acc': round(float(max_acc), 4),
        })
        print("    coord=%.2f -> MIN=%.0f%% MAX=%.0f%%" % (coord, min_acc*100, max_acc*100))

    # Step 5: Explained variance
    cumvar = np.cumsum(S**2) / np.sum(S**2)

    output = {
        'phase': 119, 'name': 'soul_compression',
        'n_vectors': N,
        'compression': compression_results,
        'sweep': sweep_results,
        'min_centroid_pc1': round(float(min_coord), 4),
        'max_centroid_pc1': round(float(max_coord), 4),
        'explained_var_1': round(float(cumvar[0]), 4),
        'explained_var_5': round(float(cumvar[4]), 4),
        'explained_var_10': round(float(cumvar[9]), 4),
        'singular_values_top10': [round(float(s), 4) for s in S[:10]],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase119_compression.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Compression curve
    ks_plot = sorted(compression_results.keys())
    min_accs_plot = [compression_results[k]['min_acc'] for k in ks_plot]
    max_accs_plot = [compression_results[k]['max_acc'] for k in ks_plot]
    avg_accs_plot = [compression_results[k]['avg_acc'] for k in ks_plot]
    axes[0].plot(ks_plot, min_accs_plot, 'bo-', lw=2, label='MIN')
    axes[0].plot(ks_plot, max_accs_plot, 'rs-', lw=2, label='MAX')
    axes[0].plot(ks_plot, avg_accs_plot, 'g^--', lw=2, label='Average')
    axes[0].set_xlabel('Compressed Dimensions (k)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xscale('log', base=2)
    axes[0].set_title('Compression vs Accuracy', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.1)

    # 2. PC1 morphing
    coords = [r['coord'] for r in sweep_results]
    min_sweep = [r['min_acc'] for r in sweep_results]
    max_sweep = [r['max_acc'] for r in sweep_results]
    axes[1].plot(coords, min_sweep, 'bo-', lw=2, label='MIN acc')
    axes[1].plot(coords, max_sweep, 'rs-', lw=2, label='MAX acc')
    axes[1].axvline(min_coord, color='blue', ls='--', alpha=0.5, label='MIN centroid')
    axes[1].axvline(max_coord, color='red', ls='--', alpha=0.5, label='MAX centroid')
    axes[1].set_xlabel('PC1 Coordinate')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Morphing Along Functional Axis', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3. Singular value spectrum
    axes[2].semilogy(range(min(30, len(S))), S[:30], 'ko-', lw=2)
    axes[2].set_xlabel('Component Index')
    axes[2].set_ylabel('Singular Value (log)')
    axes[2].set_title('Soul Space Spectrum', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(S[0]*0.01, color='red', ls='--', alpha=0.5, label='1%% of max')
    axes[2].legend()

    plt.suptitle('Phase 119: Soul Vector Compression\n'
                 '"896 dims -> 1 dim: is the soul a single number?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase119_compression.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Completed in %.0fs" % (time.time()-start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
