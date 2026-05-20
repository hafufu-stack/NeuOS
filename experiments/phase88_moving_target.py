# -*- coding: utf-8 -*-
"""
Phase 88: Epigenetic Moving Target Defense
Rotate the program vector in 896-dim space every step.
Only NeuOS knows the rotation key. Attackers aim at a moving target.

"You cannot hit what you cannot see."

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


def random_rotation_matrix(dim, seed):
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    np.random.seed(seed)
    A = np.random.randn(dim, dim).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det=+1)
    d = np.diag(R)
    signs = np.sign(d)
    Q = Q * signs[np.newaxis, :]
    return Q


def evaluate_with_rotation(model, tok, vec, test_data, layer, device, rot_key=None):
    """Evaluate a vector, optionally applying rotation defense."""
    correct = 0
    for prompt, expected in test_data:
        exec_vec = vec
        if rot_key is not None:
            # Rotate the vector before injection (defense)
            Q = torch.tensor(rot_key, device=device, dtype=torch.float32)
            exec_vec = Q @ vec
        def inject(module, input, output, v=exec_vec):
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
    print("[P88] Epigenetic Moving Target Defense")
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

    # Step 1: Compile program
    print("  Step 1: Compiling MIN program...")
    clean_vec = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42)
    clean_acc = evaluate_with_rotation(model, tok, clean_vec, all_data,
                                        target_layer, DEVICE)
    print(f"    Clean accuracy: {clean_acc:.0%}")

    # Step 2: Train a rotation-adapted program
    # Key insight: we train the program in "rotated space" so
    # rotation + injection = correct computation
    print("  Step 2: Training rotation-adapted programs...")
    n_rotations = 5
    rotation_results = []

    for rot_idx in range(n_rotations):
        Q = random_rotation_matrix(hidden_size, seed=rot_idx * 1000)
        Q_inv = Q.T  # orthogonal -> inverse = transpose

        # Train a program that works when Q is applied before injection
        # vec_rotated = Q @ vec_original
        # So we train in rotated space: inject Q @ vec, optimize vec
        torch.manual_seed(rot_idx * 100)
        vec = torch.randn(hidden_size, device=DEVICE) * 0.01
        vec.requires_grad_(True)
        Q_t = torch.tensor(Q, device=DEVICE, dtype=torch.float32)
        opt = torch.optim.Adam([vec], lr=0.01)

        for epoch in range(100):
            for prompt, target_str in min_data:
                target_id = tok.encode(target_str)[-1]
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                rotated_vec = Q_t @ vec
                def inject(module, input, output, v=rotated_vec):
                    return replace_last_token(output, v)
                h = model.model.layers[target_layer].register_forward_hook(inject)
                out = model(**inp)
                h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([target_id]).to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()

        vec_trained = vec.detach()

        # Test: does Q @ vec work?
        acc_with_key = evaluate_with_rotation(model, tok, vec_trained, all_data,
                                               target_layer, DEVICE, rot_key=Q)
        # Test: does vec alone work? (attacker has the "encrypted" vector)
        acc_no_key = evaluate_with_rotation(model, tok, vec_trained, all_data,
                                             target_layer, DEVICE, rot_key=None)

        rotation_results.append({
            'rotation': rot_idx,
            'acc_with_key': round(acc_with_key, 4),
            'acc_without_key': round(acc_no_key, 4),
        })
        print(f"    Rotation {rot_idx}: with_key={acc_with_key:.0%}, "
              f"without_key={acc_no_key:.0%}")

    # Step 3: Attack simulation - attacker targets the un-rotated vector
    print("\n  Step 3: Attack simulation (targeting wrong coordinates)...")
    # Attacker knows the FUNCTION but not the rotation key
    # They craft malware for the un-rotated space
    attacker_vec = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42)

    # FGSM attack on the clean (un-rotated) vector
    adv_vec = attacker_vec.clone().requires_grad_(True)
    prompt_str, target_str = min_data[0]
    target_id = tok.encode(target_str)[-1]
    inp = tok(prompt_str, return_tensors='pt').to(DEVICE)
    def inject_adv(module, input, output, v=adv_vec):
        return replace_last_token(output, v)
    h = model.model.layers[target_layer].register_forward_hook(inject_adv)
    out = model(**inp)
    h.remove()
    loss = -torch.nn.functional.cross_entropy(
        out.logits[0, -1, :].unsqueeze(0),
        torch.tensor([target_id]).to(DEVICE))
    loss.backward()
    adv_direction = adv_vec.grad.detach()
    adv_direction = adv_direction / (adv_direction.norm() + 1e-8)

    attack_results = []
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]

    for eps in epsilons:
        malware = adv_direction * eps

        # Attack on un-rotated target (attacker's intended target)
        infected_clean = attacker_vec + malware
        acc_static = evaluate_with_rotation(model, tok, infected_clean, all_data,
                                             target_layer, DEVICE)

        # Attack on rotation-defended target
        # Attacker applies malware to the encrypted vector
        # But malware was crafted for un-rotated coordinates -> misses
        Q = random_rotation_matrix(hidden_size, seed=0)
        Q_t = torch.tensor(Q, device=DEVICE, dtype=torch.float32)

        # Use the rotation-trained vector from step 2
        rot_vec = rotation_results[0]  # first rotation variant
        # Re-train quickly for this specific test
        torch.manual_seed(0)
        vec_r = torch.randn(hidden_size, device=DEVICE) * 0.01
        vec_r.requires_grad_(True)
        opt = torch.optim.Adam([vec_r], lr=0.01)
        for epoch in range(80):
            for prompt, tgt in min_data:
                tid = tok.encode(tgt)[-1]
                i = tok(prompt, return_tensors='pt').to(DEVICE)
                rv = Q_t @ vec_r
                def inj(m, inp, o, v=rv): return replace_last_token(o, v)
                hk = model.model.layers[target_layer].register_forward_hook(inj)
                o = model(**i)
                hk.remove()
                l = torch.nn.functional.cross_entropy(
                    o.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
                opt.zero_grad(); l.backward(); opt.step()

        # Apply malware to the "encrypted" vector (wrong coordinates)
        infected_rotated = vec_r.detach() + malware  # malware in wrong space!
        acc_moving = evaluate_with_rotation(model, tok, infected_rotated, all_data,
                                             target_layer, DEVICE, rot_key=Q)

        attack_results.append({
            'epsilon': eps,
            'static_target': round(acc_static, 4),
            'moving_target': round(acc_moving, 4),
        })
        print(f"    eps={eps:.1f}: static={acc_static:.0%}, "
              f"moving={acc_moving:.0%}")

    # Summary
    avg_with_key = np.mean([r['acc_with_key'] for r in rotation_results])
    avg_without_key = np.mean([r['acc_without_key'] for r in rotation_results])

    output = {
        'phase': 88, 'name': 'epigenetic_moving_target',
        'clean_accuracy': round(clean_acc, 4),
        'rotation_defense': rotation_results,
        'avg_with_key': round(float(avg_with_key), 4),
        'avg_without_key': round(float(avg_without_key), 4),
        'attack_results': attack_results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase88_moving_target.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Rotation defense
    r_ids = [r['rotation'] for r in rotation_results]
    axes[0].bar([x - 0.15 for x in r_ids],
                [r['acc_with_key'] for r in rotation_results],
                0.3, label='With key', color='tab:green', edgecolor='black')
    axes[0].bar([x + 0.15 for x in r_ids],
                [r['acc_without_key'] for r in rotation_results],
                0.3, label='Without key', color='tab:red', edgecolor='black')
    axes[0].set_xlabel('Rotation ID')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Rotation Defense\n(only key holder computes correctly)',
                       fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # Attack resistance
    eps_list = [r['epsilon'] for r in attack_results]
    axes[1].plot(eps_list, [r['static_target'] for r in attack_results],
                 'r-o', lw=2, label='Static target')
    axes[1].plot(eps_list, [r['moving_target'] for r in attack_results],
                 'g-s', lw=2, label='Moving target')
    axes[1].axhline(y=clean_acc, color='blue', ls='--', alpha=0.5,
                     label='Clean baseline')
    axes[1].set_xlabel('Attack Strength (epsilon)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Attack Resistance', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Summary bar
    labels = ['Static\nDefense', 'Moving\nTarget', 'Clean\nBaseline']
    vals = [
        np.mean([r['static_target'] for r in attack_results]),
        np.mean([r['moving_target'] for r in attack_results]),
        clean_acc
    ]
    colors = ['tab:red', 'tab:green', 'tab:blue']
    axes[2].bar(labels, vals, color=colors, edgecolor='black')
    axes[2].set_ylabel('Mean Accuracy Under Attack')
    axes[2].set_title('Moving Target Advantage', fontweight='bold')
    for i, v in enumerate(vals):
        axes[2].text(i, v + 0.02, f'{v:.0%}', ha='center',
                     fontweight='bold', fontsize=12)
    axes[2].set_ylim(0, 1.2)

    plt.suptitle('Phase 88: Epigenetic Moving Target Defense\n'
                 '"You cannot hit what you cannot see"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase88_moving_target.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Avg with key: {avg_with_key:.0%}")
    print(f"  Avg without key: {avg_without_key:.0%}")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
