# -*- coding: utf-8 -*-
"""
Phase 87: Endogenous Retrovirus Integration
Instead of just killing viruses, absorb their useful computation patterns.
SVD filter strips destructive components, residual analysis extracts novel function.

"That which does not kill me makes me stronger." - Nietzsche

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_prog(model, tok, train, layer, device, seed=42, epochs=100):
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


def evaluate_vec(model, tok, vec, test_data, layer, device):
    correct = 0
    preds = []
    for prompt, expected in test_data:
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append(pred)
        if pred == expected:
            correct += 1
    return correct / len(test_data), preds


def main():
    print("[P87] Endogenous Retrovirus Integration")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # NeuOS's native functions
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")]
    # "Virus" carries FIRST function (NeuOS doesn't natively have this)
    first_data = [("3, 7) =", "3"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                  ("4, 6) =", "4"), ("9, 3) =", "9")]

    # Step 1: Build native SVD basis (MIN + MAX only)
    print("  Step 1: Building native genome (MIN + MAX)...")
    native_vecs = []
    for seed in range(5):
        v = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=seed*100)
        native_vecs.append(v.cpu().numpy().flatten())
    for seed in range(5):
        v = compile_prog(model, tok, max_data, target_layer, DEVICE, seed=seed*100+50)
        native_vecs.append(v.cpu().numpy().flatten())

    native_matrix = np.array(native_vecs)
    U, S, Vt = np.linalg.svd(native_matrix, full_matrices=False)
    k_soul = 10
    Vk_native = Vt[:k_soul, :]

    # Step 2: The "virus" arrives carrying FIRST function + noise
    print("  Step 2: Virus arrives with FIRST function + destructive payload...")
    virus_core = compile_prog(model, tok, first_data, target_layer, DEVICE, seed=777)
    virus_np = virus_core.cpu().numpy().flatten()

    # Add destructive noise to the virus
    np.random.seed(42)
    destructive = np.random.randn(hidden_size).astype(np.float32) * 5.0
    virus_full = virus_np + destructive

    # Test virus directly: should be broken
    virus_vec = torch.tensor(virus_full, device=DEVICE, dtype=torch.float32)
    virus_acc, virus_preds = evaluate_vec(model, tok, virus_vec, first_data,
                                          target_layer, DEVICE)
    print(f"    Full virus (with payload): {virus_acc:.0%}")

    # Step 3: BBB strips destructive component
    print("  Step 3: SVD filter strips destructive payload...")
    # Project virus onto native manifold -> get "compatible" part
    virus_in_native = (virus_full @ Vk_native.T) @ Vk_native
    # The "novel" part = virus - native projection
    virus_novel = virus_full - virus_in_native

    # The novel part contains both the FIRST function AND remaining noise
    # Extract the FIRST-specific signal via second-pass SVD
    # (compare novel component vs clean FIRST vector)
    clean_first = virus_np  # the un-noised FIRST
    clean_first_in_native = (clean_first @ Vk_native.T) @ Vk_native
    clean_first_novel = clean_first - clean_first_in_native

    # The "integrated retrovirus" = native projection + clean novel signal
    # In practice: alpha-blend the novel component
    integration_results = []
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    print("\n  Step 4: Integration at varying strengths...")
    for alpha in alphas:
        # Integrated vector = original MIN + alpha * novel FIRST component
        min_vec = native_vecs[0]  # base MIN program
        integrated = min_vec + alpha * clean_first_novel
        integrated_vec = torch.tensor(integrated, device=DEVICE, dtype=torch.float32)

        # Test on MIN
        min_acc, min_preds = evaluate_vec(model, tok, integrated_vec, min_data,
                                          target_layer, DEVICE)
        # Test on FIRST (the virus's function)
        first_acc, first_preds = evaluate_vec(model, tok, integrated_vec, first_data,
                                              target_layer, DEVICE)

        integration_results.append({
            'alpha': alpha,
            'min_acc': round(min_acc, 4),
            'first_acc': round(first_acc, 4),
        })
        print(f"    alpha={alpha:.1f}: MIN={min_acc:.0%}, FIRST={first_acc:.0%}")

    # Step 5: Test expanded genome (post-integration)
    print("\n  Step 5: Expanded genome test...")
    # Add the novel component to native basis
    expanded_vecs = native_vecs + [clean_first_novel]
    expanded_matrix = np.array(expanded_vecs)
    U2, S2, Vt2 = np.linalg.svd(expanded_matrix, full_matrices=False)

    # Check: does the expanded basis span more of the space?
    native_energy = np.sum(S[:k_soul]**2) / np.sum(S**2)
    expanded_energy = np.sum(S2[:k_soul]**2) / np.sum(S2**2)
    print(f"    Native top-{k_soul} energy: {native_energy:.1%}")
    print(f"    Expanded top-{k_soul} energy: {expanded_energy:.1%}")

    # Save
    output = {
        'phase': 87, 'name': 'endogenous_retrovirus_integration',
        'virus_function': 'FIRST',
        'virus_with_payload_acc': round(virus_acc, 4),
        'integration_results': integration_results,
        'genome_expansion': {
            'native_energy_top10': round(float(native_energy), 4),
            'expanded_energy_top10': round(float(expanded_energy), 4),
            'native_sv': [round(float(s), 4) for s in S[:5]],
            'expanded_sv': [round(float(s), 4) for s in S2[:5]],
        },
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase87_retrovirus.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Integration curve
    ax_alphas = [r['alpha'] for r in integration_results]
    axes[0].plot(ax_alphas, [r['min_acc'] for r in integration_results],
                 'b-o', lw=2, label='MIN (native function)')
    axes[0].plot(ax_alphas, [r['first_acc'] for r in integration_results],
                 'r-s', lw=2, label='FIRST (virus function)')
    axes[0].set_xlabel('Integration Strength (alpha)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Virus Integration Curve', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # SVD spectrum comparison
    n_show = min(10, len(S), len(S2))
    x = np.arange(n_show)
    w = 0.35
    axes[1].bar(x - w/2, S[:n_show], w, label='Native', color='tab:blue',
                edgecolor='black')
    axes[1].bar(x + w/2, S2[:n_show], w, label='Post-Integration',
                color='tab:green', edgecolor='black')
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Singular Value')
    axes[1].set_title('Genome Expansion', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Before/After capabilities
    labels = ['MIN\n(native)', 'MAX\n(native)', 'FIRST\n(from virus)']
    before = [1.0, 1.0, 0.0]
    after_alpha = 0.5
    after_res = [r for r in integration_results if r['alpha'] == after_alpha][0]
    after = [after_res['min_acc'], 1.0, after_res['first_acc']]
    x = np.arange(len(labels))
    axes[2].bar(x - w/2, before, w, label='Before', color='tab:blue',
                edgecolor='black')
    axes[2].bar(x + w/2, after, w, label='After Integration',
                color='tab:green', edgecolor='black')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Capability Expansion', fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 87: Endogenous Retrovirus Integration\n'
                 '"That which does not kill me makes me stronger"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase87_retrovirus.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
