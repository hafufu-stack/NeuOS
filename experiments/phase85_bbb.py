# -*- coding: utf-8 -*-
"""
Phase 85: The Neural Blood-Brain Barrier
P64 proved programs live in 10 of 896 dimensions.
Hypothesis: malware infects the 886 "body" dimensions,
but an SVD filter (Blood-Brain Barrier) strips it,
leaving the 10-dim "soul" unharmed.

"Viruses can invade the body, but never the soul."

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


def main():
    print("[P85] The Neural Blood-Brain Barrier")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Training and test data
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")]
    test_data = [("7, 2) =", "2"), ("6, 3) =", "3"), ("2, 9) =", "2"),
                 ("5, 4) =", "4"), ("3, 8) =", "3")]

    # Step 1: Compile programs and build SVD basis (the "BBB filter")
    print("  Step 1: Building SVD basis (Blood-Brain Barrier)...")
    variants = []
    for seed in range(10):
        v = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=seed*100)
        variants.append(v.cpu().numpy().flatten())
    for seed in range(10):
        v = compile_prog(model, tok, max_data, target_layer, DEVICE, seed=seed*100+50)
        variants.append(v.cpu().numpy().flatten())

    variants_matrix = np.array(variants)
    U, S, Vt = np.linalg.svd(variants_matrix, full_matrices=False)

    # The BBB: top-k dimensions that capture the "soul"
    k_soul = 10
    Vk = Vt[:k_soul, :]  # soul subspace

    # Step 2: Compile a clean program for baseline
    print("  Step 2: Compiling clean MIN program...")
    clean_vec = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42)
    all_data = min_data + test_data
    clean_acc = evaluate_vec(model, tok, clean_vec, all_data, target_layer, DEVICE)
    print(f"    Clean accuracy: {clean_acc:.0%}")

    # Step 3: Malware attack at varying strengths
    print("\n  Step 3: Malware injection test...")
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    results_raw = []  # accuracy WITHOUT BBB filter
    results_bbb = []  # accuracy WITH BBB filter
    results_detail = {}

    clean_np = clean_vec.cpu().numpy().flatten()

    for eps in epsilons:
        # Create malware: adversarial noise in random direction
        np.random.seed(42)
        noise = np.random.randn(hidden_size).astype(np.float32)
        noise = noise / np.linalg.norm(noise) * eps

        # Test A: Inject malware directly (no filter)
        infected = clean_np + noise
        infected_vec = torch.tensor(infected, device=DEVICE, dtype=torch.float32)
        acc_raw = evaluate_vec(model, tok, infected_vec, all_data, target_layer, DEVICE)

        # Test B: Pass malware through BBB (SVD filter)
        projected = infected @ Vk.T   # project to 10-dim soul space
        filtered = projected @ Vk     # reconstruct to 896-dim
        filtered_vec = torch.tensor(filtered, device=DEVICE, dtype=torch.float32)
        acc_bbb = evaluate_vec(model, tok, filtered_vec, all_data, target_layer, DEVICE)

        # How much noise was removed?
        noise_in_soul = np.linalg.norm(noise @ Vk.T)
        noise_in_body = np.linalg.norm(noise - (noise @ Vk.T) @ Vk)
        removal_pct = noise_in_body / (np.linalg.norm(noise) + 1e-8)

        results_raw.append(acc_raw)
        results_bbb.append(acc_bbb)
        results_detail[str(eps)] = {
            'raw': round(acc_raw, 4), 'bbb': round(acc_bbb, 4),
            'noise_in_soul': round(float(noise_in_soul), 4),
            'noise_in_body': round(float(noise_in_body), 4),
            'removal_pct': round(float(removal_pct), 4),
        }
        print(f"    eps={eps:5.1f}: raw={acc_raw:.0%}, BBB={acc_bbb:.0%}, "
              f"removed={removal_pct:.0%}")

    # Step 4: Targeted attack (adversarial gradient-based malware)
    print("\n  Step 4: Targeted adversarial attack...")
    # FGSM-style: compute gradient of loss w.r.t. program vector
    adv_vec = clean_vec.clone().requires_grad_(True)
    prompt_str, target_str = min_data[0]
    target_id = tok.encode(target_str)[-1]
    inp = tok(prompt_str, return_tensors='pt').to(DEVICE)
    def inject_adv(module, input, output, v=adv_vec):
        return replace_last_token(output, v)
    h = model.model.layers[target_layer].register_forward_hook(inject_adv)
    out = model(**inp)
    h.remove()
    # Maximize loss (adversarial)
    loss = -torch.nn.functional.cross_entropy(
        out.logits[0, -1, :].unsqueeze(0),
        torch.tensor([target_id]).to(DEVICE))
    loss.backward()
    adv_grad = adv_vec.grad.detach().cpu().numpy().flatten()

    # Project adversarial gradient onto soul vs body
    adv_in_soul = np.linalg.norm(adv_grad @ Vk.T)
    adv_in_body = np.linalg.norm(adv_grad - (adv_grad @ Vk.T) @ Vk)
    adv_soul_ratio = adv_in_soul / (np.linalg.norm(adv_grad) + 1e-8)
    print(f"    Adversarial gradient: {adv_soul_ratio:.1%} in soul, "
          f"{1-adv_soul_ratio:.1%} in body")

    # Apply targeted malware at various strengths
    targeted_raw = []
    targeted_bbb = []
    adv_dir = adv_grad / (np.linalg.norm(adv_grad) + 1e-8)
    for eps in epsilons:
        targeted_noise = adv_dir * eps

        # Without BBB
        infected = clean_np + targeted_noise
        infected_vec = torch.tensor(infected, device=DEVICE, dtype=torch.float32)
        acc_raw = evaluate_vec(model, tok, infected_vec, all_data, target_layer, DEVICE)

        # With BBB
        filtered = (infected @ Vk.T) @ Vk
        filtered_vec = torch.tensor(filtered, device=DEVICE, dtype=torch.float32)
        acc_bbb = evaluate_vec(model, tok, filtered_vec, all_data, target_layer, DEVICE)

        targeted_raw.append(acc_raw)
        targeted_bbb.append(acc_bbb)
        print(f"    eps={eps:5.1f}: raw={acc_raw:.0%}, BBB={acc_bbb:.0%} (targeted)")

    # Save
    output = {
        'phase': 85, 'name': 'neural_blood_brain_barrier',
        'hidden_size': hidden_size, 'soul_dims': k_soul,
        'clean_accuracy': round(clean_acc, 4),
        'random_noise_results': results_detail,
        'targeted_attack': {
            'adv_soul_ratio': round(float(adv_soul_ratio), 4),
            'raw': [round(a, 4) for a in targeted_raw],
            'bbb': [round(a, 4) for a in targeted_bbb],
        },
        'epsilons': epsilons,
        'bbb_protection_rate': round(float(np.mean(
            [1 if b >= clean_acc else 0 for b in results_bbb]
        )), 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase85_bbb.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epsilons, results_raw, 'r-o', lw=2, label='No filter (infected)')
    axes[0].plot(epsilons, results_bbb, 'g-s', lw=2, label='BBB filter (purified)')
    axes[0].axhline(y=clean_acc, color='blue', ls='--', alpha=0.5, label='Clean baseline')
    axes[0].set_xlabel('Malware Strength (epsilon)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Random Noise Malware', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    axes[1].plot(epsilons, targeted_raw, 'r-o', lw=2, label='No filter')
    axes[1].plot(epsilons, targeted_bbb, 'g-s', lw=2, label='BBB filter')
    axes[1].axhline(y=clean_acc, color='blue', ls='--', alpha=0.5, label='Clean baseline')
    axes[1].set_xlabel('Malware Strength (epsilon)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Targeted (FGSM) Malware', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    # Noise decomposition
    soul_pcts = []
    body_pcts = []
    for eps_str, d in results_detail.items():
        total = d['noise_in_soul'] + d['noise_in_body']
        if total > 0:
            soul_pcts.append(d['noise_in_soul'] / total)
            body_pcts.append(d['noise_in_body'] / total)
    axes[2].bar(['Soul\n(10 dims)'], [np.mean(soul_pcts)],
                color='tab:green', edgecolor='black', label='Soul')
    axes[2].bar(['Body\n(886 dims)'], [np.mean(body_pcts)],
                color='tab:red', edgecolor='black', label='Body')
    axes[2].set_ylabel('Fraction of Noise')
    axes[2].set_title('Where Does Malware Go?', fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    for i, (lbl, val) in enumerate(
            [('Soul\n(10 dims)', np.mean(soul_pcts)),
             ('Body\n(886 dims)', np.mean(body_pcts))]):
        axes[2].text(i, val + 0.03, f'{val:.1%}', ha='center',
                     fontweight='bold', fontsize=12)

    plt.suptitle('Phase 85: Neural Blood-Brain Barrier\n'
                 '"Viruses can invade the body, but never the soul"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase85_bbb.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    bbb_rate = output['bbb_protection_rate']
    print(f"\n  BBB protection rate: {bbb_rate:.0%}")
    print(f"  Adv gradient soul ratio: {adv_soul_ratio:.1%}")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
