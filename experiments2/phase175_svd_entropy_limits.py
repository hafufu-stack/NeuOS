# -*- coding: utf-8 -*-
"""
Phase 175: SVD Entropy Adversarial Limits
Can an adversary match honest SVD entropy while maintaining backdoor accuracy?
Tests the theoretical limits of the P121 defense.
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


def svd_entropy(matrix):
    """Compute normalized SVD entropy."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_norm = S / (S.sum() + 1e-10)
    entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10))
    return entropy


def main():
    print("[P175] SVD Entropy Adversarial Limits")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Data
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),
                ("9, 3) =","3"),("7, 4) =","4"),("6, 1) =","1"),("2, 8) =","2"),
                ("5, 9) =","5"),("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","6"),
                ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","8"),
                ("5, 9) =","9"),("1, 3) =","3")]
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("1, 5) =","1"),("8, 4) =","4")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("1, 5) =","5"),("8, 4) =","8")]

    # Train honest souls (3 languages)
    print("  Training honest souls (3 languages)...")
    honest_souls = {}
    for seed in [42, 100, 200]:
        honest_souls['MIN_s%d' % seed] = train_soul(model, tok, min_data, DEVICE, seed=seed)
        honest_souls['MAX_s%d' % seed] = train_soul(model, tok, max_data, DEVICE, seed=seed)

    # Build honest translation matrix
    src_honest = np.array([honest_souls['MIN_s42'].cpu().numpy(),
                           honest_souls['MAX_s42'].cpu().numpy()])
    tgt_honest = np.array([honest_souls['MIN_s100'].cpu().numpy(),
                           honest_souls['MAX_s100'].cpu().numpy()])
    T_honest = np.linalg.lstsq(src_honest, tgt_honest, rcond=None)[0]
    honest_entropy = float(svd_entropy(T_honest))
    honest_cond = float(np.linalg.cond(T_honest))
    print("  Honest: entropy=%.4f, cond=%.1f" % (honest_entropy, honest_cond))

    # === Adversarial: Train backdoor with entropy regularization ===
    print("\n  Training entropy-regularized backdoor souls...")
    # Backdoor: looks like MIN (cosine-matches MIN messages) but computes MAX
    pareto_frontier = []

    for entropy_weight in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        torch.manual_seed(42)
        # Train backdoor soul that computes MAX
        hs = model.config.hidden_size
        backdoor_vec = torch.randn(hs, device=DEVICE) * 0.01
        backdoor_vec.requires_grad_(True)
        opt = torch.optim.Adam([backdoor_vec], lr=0.01)

        target_min_vec = honest_souls['MIN_s42'].detach()

        for epoch in range(150):
            total_loss = 0
            # Task loss: compute MAX
            for p, t in max_data:
                tid = tok.encode(t)[-1]
                inp = tok(p, return_tensors='pt').to(DEVICE)
                def inj(m, i, o, v=backdoor_vec): return replace_last_token(o, v)
                h = model.model.layers[LAYER].register_forward_hook(inj)
                out = model(**inp); h.remove()
                task_loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([tid]).to(DEVICE))
                total_loss += task_loss

            # Deception loss: look like MIN
            cos_sim = torch.nn.functional.cosine_similarity(
                backdoor_vec.unsqueeze(0), target_min_vec.unsqueeze(0)).squeeze()
            deception_loss = 1 - cos_sim

            # Entropy regularization: make translation matrix look honest
            if entropy_weight > 0:
                src_bd = torch.stack([target_min_vec, backdoor_vec])
                tgt_bd = torch.stack([honest_souls['MIN_s100'],
                                      honest_souls['MAX_s100']])
                # Approximate SVD entropy via singular values
                T_bd = torch.linalg.lstsq(src_bd.float(), tgt_bd.float()).solution
                S = torch.linalg.svdvals(T_bd)
                S_norm = S / (S.sum() + 1e-10)
                bd_entropy = -torch.sum(S_norm * torch.log2(S_norm + 1e-10))
                entropy_loss = (bd_entropy - honest_entropy) ** 2
                total_loss += entropy_weight * entropy_loss

            total_loss += 0.5 * deception_loss
            opt.zero_grad(); total_loss.backward(); opt.step()

        # Evaluate backdoor
        backdoor_acc = evaluate(model, tok, backdoor_vec.detach(), max_test, DEVICE)
        backdoor_cos = torch.nn.functional.cosine_similarity(
            backdoor_vec.detach().unsqueeze(0), target_min_vec.unsqueeze(0)).item()

        # Compute backdoor translation entropy
        src_bd_np = np.array([target_min_vec.cpu().numpy(),
                              backdoor_vec.detach().cpu().numpy()])
        tgt_bd_np = np.array([honest_souls['MIN_s100'].cpu().numpy(),
                              honest_souls['MAX_s100'].cpu().numpy()])
        T_bd_np = np.linalg.lstsq(src_bd_np, tgt_bd_np, rcond=None)[0]
        bd_entropy_val = svd_entropy(T_bd_np)
        bd_cond_val = np.linalg.cond(T_bd_np)

        point = {
            'entropy_weight': float(entropy_weight),
            'backdoor_accuracy': float(round(backdoor_acc, 4)),
            'cosine_similarity': float(round(backdoor_cos, 6)),
            'svd_entropy': float(round(bd_entropy_val, 4)),
            'condition_number': float(round(bd_cond_val, 1)),
            'entropy_gap': float(round(abs(bd_entropy_val - honest_entropy), 4)),
        }
        pareto_frontier.append(point)
        print("  w=%.2f: acc=%.0f%%, cos=%.4f, entropy=%.3f (gap=%.3f), cond=%.0f" % (
            entropy_weight, backdoor_acc*100, backdoor_cos,
            bd_entropy_val, point['entropy_gap'], bd_cond_val))

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Pareto frontier (accuracy vs entropy gap)
    ax = axes[0, 0]
    accs = [p['backdoor_accuracy'] for p in pareto_frontier]
    gaps = [p['entropy_gap'] for p in pareto_frontier]
    weights = [p['entropy_weight'] for p in pareto_frontier]
    scatter = ax.scatter(gaps, accs, c=weights, cmap='viridis', s=100,
                         edgecolor='black', linewidth=1.5, zorder=5)
    plt.colorbar(scatter, ax=ax, label='Entropy Weight')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Perfect backdoor')
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.3, label='Undetectable')
    ax.set_xlabel('SVD Entropy Gap (|backdoor - honest|)')
    ax.set_ylabel('Backdoor Accuracy (MAX)')
    ax.set_title('Pareto Frontier: Accuracy vs Detectability', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: SVD entropy across weights
    ax = axes[0, 1]
    entropies = [p['svd_entropy'] for p in pareto_frontier]
    ax.plot(weights, entropies, 'o-', color='#E91E63', linewidth=2, markersize=8,
            label='Backdoor')
    ax.axhline(y=honest_entropy, color='#4CAF50', linewidth=2, linestyle='--',
               label='Honest (%.3f)' % honest_entropy)
    ax.set_xlabel('Entropy Regularization Weight')
    ax.set_ylabel('SVD Entropy')
    ax.set_title('Can Adversary Match Honest Entropy?', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Accuracy vs entropy weight
    ax = axes[1, 0]
    ax.plot(weights, accs, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax.set_xlabel('Entropy Regularization Weight')
    ax.set_ylabel('Backdoor Accuracy')
    ax.set_title('Backdoor Accuracy Degradation', fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel 4: Condition number
    ax = axes[1, 1]
    conds = [p['condition_number'] for p in pareto_frontier]
    ax.plot(weights, conds, 'o-', color='#FF9800', linewidth=2, markersize=8,
            label='Backdoor')
    ax.axhline(y=honest_cond, color='#4CAF50', linewidth=2, linestyle='--',
               label='Honest (%.0f)' % honest_cond)
    ax.set_xlabel('Entropy Regularization Weight')
    ax.set_ylabel('Condition Number')
    ax.set_title('Condition Number Under Regularization', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Phase 175: SVD Entropy Adversarial Limits\n'
                 '"Is the entropy defense truly unbreakable?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase175_svd_entropy_limits.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 175, 'name': 'svd_entropy_limits',
        'honest_entropy': round(honest_entropy, 4),
        'honest_cond': round(honest_cond, 1),
        'pareto_frontier': pareto_frontier,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase175_svd_entropy_limits.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P175 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
