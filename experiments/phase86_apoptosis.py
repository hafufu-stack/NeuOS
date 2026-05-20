# -*- coding: utf-8 -*-
"""
Phase 86: Viral Apoptosis & Reincarnation
When infection is detected, the program vector self-destructs (apoptosis)
and reincarnates as a polymorphic variant (P51-style).

"The body dies, but the soul lives on in a new form."

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


def detect_infection(vec, svd_basis, threshold=0.3):
    """Detect infection by checking how much of the vector is outside the soul manifold."""
    vec_np = vec.cpu().numpy().flatten()
    projected = (vec_np @ svd_basis.T) @ svd_basis
    residual = vec_np - projected
    residual_ratio = np.linalg.norm(residual) / (np.linalg.norm(vec_np) + 1e-8)
    return residual_ratio > threshold, residual_ratio


def main():
    print("[P86] Viral Apoptosis & Reincarnation")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    test_data = [("7, 2) =", "2"), ("6, 3) =", "3"), ("2, 9) =", "2"),
                 ("5, 4) =", "4"), ("3, 8) =", "3")]
    all_data = min_data + test_data

    # Step 1: Build SVD basis (soul manifold)
    print("  Step 1: Building soul manifold...")
    variants = []
    for seed in range(10):
        v = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=seed*100)
        variants.append(v.cpu().numpy().flatten())
    variants_matrix = np.array(variants)
    U, S, Vt = np.linalg.svd(variants_matrix, full_matrices=False)
    k_soul = 10
    Vk = Vt[:k_soul, :]

    # Step 2: Compile the "original body"
    print("  Step 2: Compiling original body...")
    original = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42)
    original_acc = evaluate_vec(model, tok, original, all_data, target_layer, DEVICE)
    print(f"    Original accuracy: {original_acc:.0%}")

    # Step 3: Infect with escalating malware
    print("\n  Step 3: Infection -> Detection -> Apoptosis -> Reincarnation cycle")
    results = []
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    reincarnation_seeds = iter(range(1000, 2000))

    for eps in epsilons:
        print(f"\n    --- Malware eps={eps} ---")
        original_np = original.cpu().numpy().flatten()

        # Infect
        np.random.seed(int(eps * 100))
        noise = np.random.randn(hidden_size).astype(np.float32)
        noise = noise / np.linalg.norm(noise) * eps
        infected_np = original_np + noise
        infected_vec = torch.tensor(infected_np, device=DEVICE, dtype=torch.float32)
        infected_acc = evaluate_vec(model, tok, infected_vec, all_data, target_layer, DEVICE)

        # Detect
        is_infected, residual = detect_infection(infected_vec, Vk)
        print(f"    Infected acc: {infected_acc:.0%}, "
              f"detected: {is_infected}, residual: {residual:.3f}")

        # Apoptosis: destroy the infected vector
        # (In real NeuOS: zero out L16 register space)
        apoptosis_vec = torch.zeros(hidden_size, device=DEVICE)

        # Reincarnation: compile a NEW body from a different seed
        # The "soul" (10-dim core) is preserved but the "body" is entirely new
        new_seed = next(reincarnation_seeds)
        reincarnated = compile_prog(model, tok, min_data, target_layer, DEVICE,
                                    seed=new_seed)
        reincarnated_acc = evaluate_vec(model, tok, reincarnated, all_data,
                                        target_layer, DEVICE)

        # Check: is the reincarnated body different from the original?
        cos_sim = float(torch.nn.functional.cosine_similarity(
            original.unsqueeze(0), reincarnated.unsqueeze(0)).item())

        # Check: is the reincarnated body immune to the SAME malware?
        reincarnated_np = reincarnated.cpu().numpy().flatten()
        re_infected_np = reincarnated_np + noise
        re_infected_vec = torch.tensor(re_infected_np, device=DEVICE, dtype=torch.float32)
        re_infected_acc = evaluate_vec(model, tok, re_infected_vec, all_data,
                                        target_layer, DEVICE)

        print(f"    Reincarnated acc: {reincarnated_acc:.0%}, "
              f"cos_sim to original: {cos_sim:.3f}")
        print(f"    Same malware on new body: {re_infected_acc:.0%}")

        results.append({
            'epsilon': eps,
            'infected_acc': round(float(infected_acc), 4),
            'detected': bool(is_infected),
            'residual': round(float(residual), 4),
            'reincarnated_acc': round(float(reincarnated_acc), 4),
            'cos_sim_to_original': round(float(cos_sim), 4),
            're_infected_acc': round(float(re_infected_acc), 4),
            'reincarnation_seed': new_seed,
        })

    # Summary
    avg_reincarnated = np.mean([r['reincarnated_acc'] for r in results])
    avg_infected = np.mean([r['infected_acc'] for r in results])
    detection_rate = np.mean([1 if r['detected'] else 0 for r in results])

    output = {
        'phase': 86, 'name': 'viral_apoptosis_reincarnation',
        'original_accuracy': round(original_acc, 4),
        'avg_infected_accuracy': round(float(avg_infected), 4),
        'avg_reincarnated_accuracy': round(float(avg_reincarnated), 4),
        'detection_rate': round(float(detection_rate), 4),
        'soul_dims': k_soul,
        'results': results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase86_apoptosis.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    eps_list = [r['epsilon'] for r in results]
    axes[0].plot(eps_list, [r['infected_acc'] for r in results],
                 'r-o', lw=2, label='Infected (dying)')
    axes[0].plot(eps_list, [r['reincarnated_acc'] for r in results],
                 'g-s', lw=2, label='Reincarnated (new body)')
    axes[0].axhline(y=original_acc, color='blue', ls='--', alpha=0.5,
                     label='Original')
    axes[0].set_xlabel('Malware Strength')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Death & Rebirth Cycle', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps_list, [r['cos_sim_to_original'] for r in results],
                 'purple', lw=2, marker='D')
    axes[1].set_xlabel('Malware Strength')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Body Dissimilarity\n(low = different body, same soul)',
                       fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(eps_list, [r['re_infected_acc'] for r in results],
                 'orange', lw=2, marker='^', label='Same virus on new body')
    axes[2].plot(eps_list, [r['infected_acc'] for r in results],
                 'r--', lw=1.5, label='Same virus on old body')
    axes[2].axhline(y=original_acc, color='blue', ls='--', alpha=0.5,
                     label='Clean baseline')
    axes[2].set_xlabel('Malware Strength')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Virus Resistance After Reincarnation', fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 86: Viral Apoptosis & Reincarnation\n'
                 '"The body dies, but the soul lives on in a new form"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase86_apoptosis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Detection rate: {detection_rate:.0%}")
    print(f"  Avg infected acc: {avg_infected:.0%}")
    print(f"  Avg reincarnated acc: {avg_reincarnated:.0%}")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
