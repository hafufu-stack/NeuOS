# -*- coding: utf-8 -*-
"""
Phase 75: Language Emergence
Programs communicate by exchanging PCA coordinates.
A 'speaker' encodes its function as a 2D coordinate,
and a 'listener' decodes the coordinate to identify the function.
Primitive language emerges from shared representation space.

Uses P65's PCA genome map.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_prog(model, tok, train, layer, device, seed=42, epochs=80):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(epochs):
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


def main():
    print("[P75] Language Emergence")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile vocabulary of programs
    programs = {
        'MIN': [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")],
        'MAX': [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")],
        'FIRST': [("3, 7) =", "3"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                  ("4, 6) =", "4"), ("9, 3) =", "9")],
        'SECOND': [("3, 7) =", "7"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                   ("4, 6) =", "6"), ("9, 3) =", "3")],
    }

    print("  Building program vocabulary...")
    all_vecs = []
    all_labels = []
    for name, data in programs.items():
        for seed in range(8):
            v = compile_prog(model, tok, data, target_layer, DEVICE,
                           seed=hash(name) % 10000 + seed * 100, epochs=80)
            all_vecs.append(v.cpu().numpy().flatten())
            all_labels.append(name)
        print(f"    {name}: 8 variants compiled")

    # Step 1: Learn shared PCA language
    print("\n  Step 1: Learning shared coordinate system (PCA)...")
    pca = PCA(n_components=5)
    coords = pca.fit_transform(np.array(all_vecs))
    explained = pca.explained_variance_ratio_
    print(f"    Variance explained: {explained[:3]}")

    # Step 2: Can a listener identify function from coordinates alone?
    print("\n  Step 2: Communication test (speak -> listen)...")
    # Leave-one-out: speaker sends coordinates, listener classifies
    knn = KNeighborsClassifier(n_neighbors=3)
    correct = 0
    for i in range(len(coords)):
        X_train = np.delete(coords, i, axis=0)
        y_train = [l for j, l in enumerate(all_labels) if j != i]
        X_test = coords[i:i+1]
        y_test = all_labels[i]
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)[0]
        if pred == y_test:
            correct += 1
    comm_accuracy = correct / len(coords)
    print(f"    Communication accuracy (LOO): {comm_accuracy:.0%}")

    # Step 3: Dimensionality of language
    print("\n  Step 3: How many dimensions needed for perfect communication?")
    dim_results = {}
    for n_dim in [1, 2, 3, 5]:
        coords_nd = coords[:, :n_dim]
        correct = 0
        for i in range(len(coords_nd)):
            X_train = np.delete(coords_nd, i, axis=0)
            y_train = [l for j, l in enumerate(all_labels) if j != i]
            X_test = coords_nd[i:i+1]
            y_test = all_labels[i]
            knn.fit(X_train, y_train)
            pred = knn.predict(X_test)[0]
            if pred == y_test:
                correct += 1
        acc = correct / len(coords_nd)
        dim_results[n_dim] = round(acc, 4)
        print(f"    {n_dim}D: {acc:.0%}")

    # Step 4: Can programs "name" each other?
    print("\n  Step 4: Naming test (reconstruct from coordinates)...")
    # Take PCA coordinates, reconstruct program, test accuracy
    test_p = ["3, 7) =", "5, 2) =", "8, 1) ="]
    min_exp = ["3", "2", "1"]
    max_exp = ["7", "5", "8"]

    # Reconstruct MIN from its PCA centroid
    min_indices = [i for i, l in enumerate(all_labels) if l == 'MIN']
    min_centroid = np.mean(coords[min_indices], axis=0)
    reconstructed = pca.inverse_transform(min_centroid.reshape(1, -1)).flatten()
    recon_vec = torch.tensor(reconstructed, dtype=torch.float32, device=DEVICE)

    recon_correct = 0
    for prompt, exp in zip(test_p, min_exp):
        def inject(module, input, output, v=recon_vec):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == exp:
            recon_correct += 1
    recon_acc = recon_correct / len(test_p)
    print(f"    Reconstructed MIN accuracy: {recon_acc:.0%}")

    # Save
    output = {
        'phase': 75, 'name': 'language_emergence',
        'n_programs': len(all_vecs),
        'pca_variance': [round(float(v), 4) for v in explained],
        'communication_accuracy': round(comm_accuracy, 4),
        'dim_results': dim_results,
        'reconstruction_accuracy': round(recon_acc, 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase75_language.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PCA map with labels
    colors_map = {'MIN': 'tab:blue', 'MAX': 'tab:red',
                  'FIRST': 'tab:green', 'SECOND': 'tab:purple'}
    for name in programs:
        idxs = [i for i, l in enumerate(all_labels) if l == name]
        x = [coords[i, 0] for i in idxs]
        y = [coords[i, 1] for i in idxs]
        axes[0].scatter(x, y, c=colors_map[name], label=name, s=80, edgecolor='black')
    axes[0].set_xlabel(f'PC1 ({explained[0]:.0%})')
    axes[0].set_ylabel(f'PC2 ({explained[1]:.0%})')
    axes[0].set_title('Language Space (PCA)', fontweight='bold')
    axes[0].legend()

    dims = sorted(dim_results.keys())
    accs = [dim_results[d] for d in dims]
    axes[1].plot(dims, accs, 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Language Dimensions')
    axes[1].set_ylabel('Communication Accuracy')
    axes[1].set_title('Language Complexity vs Accuracy', fontweight='bold')
    axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1.1)

    axes[2].axis('off')
    summary = (f"LANGUAGE EMERGENCE\n{'='*30}\n\n"
               f"Programs: {len(all_vecs)} (4 functions x 8)\n"
               f"PCA dims: {explained[0]:.0%}+{explained[1]:.0%}\n\n"
               f"Communication: {comm_accuracy:.0%}\n"
               f"1D language: {dim_results.get(1, 0):.0%}\n"
               f"2D language: {dim_results.get(2, 0):.0%}\n"
               f"5D language: {dim_results.get(5, 0):.0%}\n\n"
               f"Reconstruction: {recon_acc:.0%}")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                fontsize=10, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 75: Language Emergence\nPrograms communicate via shared coordinate system',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase75_language.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
