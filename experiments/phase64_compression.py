# -*- coding: utf-8 -*-
"""
Phase 64: Program Compression
P51 showed 896-dim vectors implement simple functions.
How many dimensions do you ACTUALLY need?
SVD/PCA to find the minimal subspace preserving function.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_prog(model, tok, train, layer, device, seed=42):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(100):
        for prompt, target_str in train:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def inject(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def evaluate_vec(model, tok, vec, test_data, layer, device):
    correct = 0
    for prompt, expected in test_data:
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data)


def main():
    print("[P64] Program Compression")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile multiple variants of MIN (for PCA basis)
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    test_data = [("7, 2) =", "2"), ("6, 3) =", "3"), ("2, 9) =", "2"),
                 ("5, 4) =", "4"), ("3, 8) =", "3")]

    print("  Compiling 10 MIN variants for PCA...")
    variants = []
    for i in range(10):
        v = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=i*100)
        variants.append(v.cpu().numpy().flatten())

    variants_matrix = np.array(variants)

    # Full accuracy baseline
    best_vec = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=0)
    full_acc = evaluate_vec(model, tok, best_vec, min_data + test_data, target_layer, DEVICE)
    print(f"  Full 896-dim accuracy: {full_acc:.0%}")

    # SVD compression test
    print("\n  Compression test (SVD truncation)...")
    U, S, Vt = np.linalg.svd(variants_matrix, full_matrices=False)

    # Energy in each component
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy

    # Test at different compression levels
    test_dims = [1, 2, 3, 5, 10, 20, 50, 100, 200, 448, 896]
    compression_results = {}

    vec_np = best_vec.cpu().numpy().flatten()

    for k in test_dims:
        if k > min(variants_matrix.shape):
            k = min(variants_matrix.shape)
        # Project into top-k SVD subspace
        Vk = Vt[:k, :]  # top-k right singular vectors
        projected = vec_np @ Vk.T  # project to k dims
        reconstructed = projected @ Vk  # reconstruct to 896 dims

        # Evaluate reconstructed vector
        recon_vec = torch.tensor(reconstructed, device=DEVICE, dtype=torch.float32)
        acc = evaluate_vec(model, tok, recon_vec, min_data + test_data, target_layer, DEVICE)
        compression_results[k] = round(acc, 4)

        # Also test random subspace of same dim for comparison
        rand_basis = np.random.randn(k, hidden_size)
        rand_basis, _ = np.linalg.qr(rand_basis.T)
        rand_basis = rand_basis[:, :k].T
        rand_proj = vec_np @ rand_basis.T
        rand_recon = rand_proj @ rand_basis
        rand_vec = torch.tensor(rand_recon, device=DEVICE, dtype=torch.float32)
        rand_acc = evaluate_vec(model, tok, rand_vec, min_data + test_data, target_layer, DEVICE)

        if k <= 50 or k in [100, 200, 448, 896]:
            print(f"    k={k}: SVD={acc:.0%}, Random={rand_acc:.0%}, "
                  f"energy={cumulative_energy[min(k-1, len(cumulative_energy)-1)]:.2%}")

    # Find minimum dimensions for 100% accuracy
    min_dims_100 = None
    for k in sorted(compression_results.keys()):
        if compression_results[k] >= 1.0:
            min_dims_100 = k
            break

    print(f"\n  Minimum dims for 100%: {min_dims_100 if min_dims_100 else '>896'}")
    print(f"  Compression ratio: {hidden_size}/{min_dims_100}x = "
          f"{hidden_size/min_dims_100:.1f}x" if min_dims_100 else "  Not achieved")

    # Save
    output = {
        'phase': 64, 'name': 'program_compression',
        'full_dim': hidden_size,
        'full_accuracy': round(full_acc, 4),
        'compression_results': compression_results,
        'min_dims_100': min_dims_100,
        'compression_ratio': round(hidden_size / min_dims_100, 1) if min_dims_100 else None,
        'sv_energy_top5': [round(float(e), 4) for e in cumulative_energy[:5]],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase64_compression.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(range(min(10, len(S))), S[:10], color='tab:blue', edgecolor='black')
    axes[0].set_xlabel('Component'); axes[0].set_ylabel('Singular Value')
    axes[0].set_title('SVD Spectrum (top 10)', fontweight='bold')

    dims = sorted(compression_results.keys())
    accs = [compression_results[k] for k in dims]
    axes[1].plot(dims, accs, 'go-', linewidth=2, markersize=6)
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='100%')
    if min_dims_100:
        axes[1].axvline(x=min_dims_100, color='red', linestyle='--',
                       label=f'Min dims={min_dims_100}')
    axes[1].set_xlabel('Dimensions'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs Compression', fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(range(1, len(cumulative_energy)+1), cumulative_energy, 'b-', linewidth=2)
    axes[2].set_xlabel('Components'); axes[2].set_ylabel('Cumulative Energy')
    axes[2].set_title('Energy Distribution', fontweight='bold')
    axes[2].axhline(y=0.99, color='red', linestyle='--', label='99% energy')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Phase 64: Program Compression\n896-dim -> {min_dims_100 if min_dims_100 else "?"}-dim '
                f'({hidden_size/min_dims_100:.1f}x compression)' if min_dims_100 else
                'Phase 64: Program Compression',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase64_compression.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
