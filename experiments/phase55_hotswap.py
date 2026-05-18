# -*- coding: utf-8 -*-
"""
Phase 55: Polymorphic Hot-Swapping
Rotate between P51's polymorphic variants every few steps.
Function stays 100% while binary constantly changes.
Moving Target Defense for neural security.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_variant(model, tok, train_data, layer, device, seed):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(80):
        for prompt, target_str in train_data:
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
    print("[P55] Polymorphic Hot-Swapping")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile 5 polymorphic variants of MIN
    print("  Compiling 5 MIN variants...")
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    variants = []
    for i in range(5):
        v = compile_variant(model, tok, min_data, target_layer, DEVICE, seed=i*100)
        variants.append(v)
        print(f"    Variant {i} compiled")

    # Measure pairwise similarities
    vecs_np = [v.cpu().numpy().flatten() for v in variants]
    sim_matrix = cosine_similarity(np.array(vecs_np))
    triu = sim_matrix[np.triu_indices(5, k=1)]
    avg_sim = float(np.mean(triu))
    print(f"    Avg pairwise similarity: {avg_sim:.4f}")

    # Test: run 20 queries, swapping variant every step
    print("\n  Running with hot-swapping...")
    test_queries = [
        ("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
        ("4, 6) =", "4"), ("9, 3) =", "3"), ("7, 2) =", "2"),
        ("6, 3) =", "3"), ("2, 9) =", "2"), ("5, 4) =", "4"),
        ("3, 8) =", "3"), ("7, 1) =", "1"), ("6, 2) =", "2"),
        ("4, 9) =", "4"), ("8, 3) =", "3"), ("2, 5) =", "2"),
        ("9, 1) =", "1"), ("3, 6) =", "3"), ("7, 4) =", "4"),
        ("5, 8) =", "5"), ("2, 3) =", "2"),
    ]

    results = []
    variant_used = []
    for step, (prompt, expected) in enumerate(test_queries):
        # Random variant selection (hot-swap)
        idx = np.random.randint(0, len(variants))
        variant_used.append(idx)
        vec = variants[idx]

        def inject_hs(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_hs)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        correct = pred == expected
        results.append(correct)

    accuracy = sum(results) / len(results)
    n_unique = len(set(variant_used))
    swap_count = sum(1 for i in range(1, len(variant_used))
                     if variant_used[i] != variant_used[i-1])

    print(f"\n  Hot-swap accuracy: {accuracy:.0%}")
    print(f"  Variants used: {n_unique}/5")
    print(f"  Swaps: {swap_count}/{len(test_queries)-1}")
    print(f"  Avg binary similarity: {avg_sim:.4f}")

    # Save
    output = {
        'phase': 55, 'name': 'polymorphic_hot_swapping',
        'accuracy': round(accuracy, 4),
        'n_variants': len(variants),
        'n_swaps': swap_count,
        'avg_binary_sim': round(avg_sim, 4),
        'variant_sequence': variant_used,
        'results': results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase55_hotswap.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(['Hot-Swap\nAccuracy'], [accuracy],
                color='tab:green', edgecolor='black')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Polymorphic Execution\n{swap_count} swaps, {accuracy:.0%} correct',
                      fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].text(0, accuracy+0.03, f'{accuracy:.0%}', ha='center', fontweight='bold', fontsize=16)

    axes[1].plot(range(len(variant_used)), variant_used, 'o-', markersize=6)
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Variant ID')
    axes[1].set_title('Variant Switching Pattern', fontweight='bold')
    axes[1].set_yticks(range(5))
    axes[1].grid(True, alpha=0.3)

    im = axes[2].imshow(sim_matrix, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    axes[2].set_title(f'Binary Similarity\navg={avg_sim:.4f}', fontweight='bold')
    axes[2].set_xlabel('Variant'); axes[2].set_ylabel('Variant')
    plt.colorbar(im, ax=axes[2])
    for i in range(5):
        for j in range(5):
            axes[2].text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)

    plt.suptitle('Phase 55: Polymorphic Hot-Swapping\nMoving Target Defense - constant binary mutation',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase55_hotswap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
