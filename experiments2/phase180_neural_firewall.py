# -*- coding: utf-8 -*-
"""
Phase 180: Neural Firewall
Build a practical anomaly detection system using GlassBox Dashboard
features. Detect adversarial soul injection via KL divergence.
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

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


def collect_register_profile(model, tok, prompt, device, layers=None):
    """Collect hidden state statistics at multiple layers."""
    if layers is None:
        layers = [0, 4, 8, 12, 16, 20, 22]
    profiles = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(m, i, o):
            tensor = o[0] if isinstance(o, tuple) else o
            if tensor.dim() == 3:
                h = tensor[0, -1, :].detach()
            else:
                h = tensor[-1, :].detach()
            profiles[layer_idx] = h
        return hook_fn

    for l in layers:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)

    for h in hooks:
        h.remove()
    return profiles


def collect_with_injection(model, tok, prompt, soul_vec, inject_layer, device, layers=None):
    """Collect register profile while injecting a soul vector."""
    if layers is None:
        layers = [0, 4, 8, 12, 16, 20, 22]
    profiles = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(m, i, o):
            tensor = o[0] if isinstance(o, tuple) else o
            if tensor.dim() == 3:
                h = tensor[0, -1, :].detach()
            else:
                h = tensor[-1, :].detach()
            profiles[layer_idx] = h
        return hook_fn

    def inject_hook(m, i, o, v=soul_vec):
        return replace_last_token(o, v)

    for l in layers:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
    hooks.append(model.model.layers[inject_layer].register_forward_hook(inject_hook))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)

    for h in hooks:
        h.remove()
    return profiles


def compute_kl_divergence(p_mean, p_std, q_mean, q_std):
    """Approximate KL divergence between two diagonal Gaussians."""
    # KL(P || Q) = sum[log(q_std/p_std) + (p_std^2 + (p_mean - q_mean)^2) / (2*q_std^2) - 0.5]
    p_std = torch.clamp(p_std, min=1e-6)
    q_std = torch.clamp(q_std, min=1e-6)
    kl = torch.sum(
        torch.log(q_std / p_std) +
        (p_std**2 + (p_mean - q_mean)**2) / (2 * q_std**2) - 0.5
    )
    return kl.item()


def main():
    print("[P180] Neural Firewall")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    monitor_layers = [0, 4, 8, 12, 16, 20, 22]

    # Train legitimate souls
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),
                ("9, 3) =","3"),("7, 4) =","4"),("6, 1) =","1"),("2, 8) =","2"),
                ("5, 9) =","5"),("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","6"),
                ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","8"),
                ("5, 9) =","9"),("1, 3) =","3")]

    min_soul = train_soul(model, tok, min_data, DEVICE, seed=42)
    max_soul = train_soul(model, tok, max_data, DEVICE, seed=42)

    # === Build normal profile distribution ===
    print("  Building normal register profile...")
    normal_prompts = ["%d, %d) =" % (a, b)
                      for a in range(1, 10) for b in range(1, 10) if a != b]
    normal_profiles = {l: [] for l in monitor_layers}
    for prompt in normal_prompts[:30]:  # Limit for speed
        prof = collect_register_profile(model, tok, prompt, DEVICE, monitor_layers)
        for l in monitor_layers:
            normal_profiles[l].append(prof[l])

    # Compute mean and std per layer
    normal_stats = {}
    for l in monitor_layers:
        stacked = torch.stack(normal_profiles[l])
        normal_stats[l] = {
            'mean': stacked.mean(dim=0),
            'std': stacked.std(dim=0) + 1e-6,
        }

    # === Collect profiles under different conditions ===
    print("  Collecting profiles under injection...")
    test_prompts = ["%d, %d) =" % (a, b) for a, b in
                    [(7, 2), (6, 3), (2, 9), (1, 5), (8, 4)]]

    conditions = {
        'normal': [],
        'legitimate_min': [],
        'legitimate_max': [],
        'adversarial_random': [],
        'adversarial_noise': [],
    }

    # Normal (no injection)
    for prompt in test_prompts:
        prof = collect_register_profile(model, tok, prompt, DEVICE, monitor_layers)
        kl_scores = []
        for l in monitor_layers:
            kl = compute_kl_divergence(
                prof[l], torch.ones_like(prof[l]) * 0.1,
                normal_stats[l]['mean'], normal_stats[l]['std'])
            kl_scores.append(kl)
        conditions['normal'].append(np.mean(kl_scores))

    # Legitimate injection
    for soul, name in [(min_soul, 'legitimate_min'), (max_soul, 'legitimate_max')]:
        for prompt in test_prompts:
            prof = collect_with_injection(model, tok, prompt, soul, LAYER,
                                          DEVICE, monitor_layers)
            kl_scores = []
            for l in monitor_layers:
                kl = compute_kl_divergence(
                    prof[l], torch.ones_like(prof[l]) * 0.1,
                    normal_stats[l]['mean'], normal_stats[l]['std'])
                kl_scores.append(kl)
            conditions[name].append(np.mean(kl_scores))

    # Adversarial: random vectors
    for seed in range(5):
        torch.manual_seed(seed + 1000)
        adv_vec = torch.randn(896, device=DEVICE)
        for prompt in test_prompts:
            prof = collect_with_injection(model, tok, prompt, adv_vec, LAYER,
                                          DEVICE, monitor_layers)
            kl_scores = []
            for l in monitor_layers:
                kl = compute_kl_divergence(
                    prof[l], torch.ones_like(prof[l]) * 0.1,
                    normal_stats[l]['mean'], normal_stats[l]['std'])
                kl_scores.append(kl)
            conditions['adversarial_random'].append(np.mean(kl_scores))

    # Adversarial: noise perturbation of legitimate soul
    for noise_level in [0.5, 1.0, 2.0, 5.0, 10.0]:
        noisy_soul = min_soul + torch.randn_like(min_soul) * noise_level
        for prompt in test_prompts:
            prof = collect_with_injection(model, tok, prompt, noisy_soul, LAYER,
                                          DEVICE, monitor_layers)
            kl_scores = []
            for l in monitor_layers:
                kl = compute_kl_divergence(
                    prof[l], torch.ones_like(prof[l]) * 0.1,
                    normal_stats[l]['mean'], normal_stats[l]['std'])
                kl_scores.append(kl)
            conditions['adversarial_noise'].append(np.mean(kl_scores))

    # === ROC Analysis ===
    print("\n  === ROC Analysis ===")
    # Labels: 0 = benign (normal + legitimate), 1 = adversarial
    benign_scores = conditions['normal'] + conditions['legitimate_min'] + \
                    conditions['legitimate_max']
    adv_scores = conditions['adversarial_random'] + conditions['adversarial_noise']

    y_true = [0] * len(benign_scores) + [1] * len(adv_scores)
    y_scores = benign_scores + adv_scores

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print("  ROC AUC: %.4f" % roc_auc)

    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]
    print("  Best threshold: %.4f (TPR=%.2f, FPR=%.2f)" % (
        best_threshold, best_tpr, best_fpr))

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: KL divergence distribution by condition
    ax = axes[0, 0]
    box_data = [conditions['normal'], conditions['legitimate_min'],
                conditions['legitimate_max'], conditions['adversarial_random'],
                conditions['adversarial_noise']]
    bp = ax.boxplot(box_data, labels=['Normal', 'Legit\nMIN', 'Legit\nMAX',
                                       'Adv\nRandom', 'Adv\nNoise'],
                    patch_artist=True)
    colors_box = ['#4CAF50', '#4CAF50', '#4CAF50', '#F44336', '#F44336']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('Register Profile Anomaly Score', fontweight='bold')
    ax.axhline(y=best_threshold, color='orange', linestyle='--',
               label='Threshold (%.2f)' % best_threshold)
    ax.legend()

    # Panel 2: ROC curve
    ax = axes[0, 1]
    ax.plot(fpr, tpr, color='#2196F3', linewidth=2,
            label='ROC (AUC = %.3f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.plot(best_fpr, best_tpr, 'ro', markersize=10,
            label='Optimal (TPR=%.2f, FPR=%.2f)' % (best_tpr, best_fpr))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Neural Firewall ROC Curve', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Per-layer KL divergence
    ax = axes[1, 0]
    for cond_name, cond_color in [('normal', '#4CAF50'),
                                   ('adversarial_random', '#F44336')]:
        per_layer = {l: [] for l in monitor_layers}
        prompts_to_test = test_prompts[:3]
        for prompt in prompts_to_test:
            if cond_name == 'normal':
                prof = collect_register_profile(model, tok, prompt, DEVICE, monitor_layers)
            else:
                torch.manual_seed(1000)
                adv_v = torch.randn(896, device=DEVICE)
                prof = collect_with_injection(model, tok, prompt, adv_v, LAYER,
                                              DEVICE, monitor_layers)
            for l in monitor_layers:
                kl = compute_kl_divergence(
                    prof[l], torch.ones_like(prof[l]) * 0.1,
                    normal_stats[l]['mean'], normal_stats[l]['std'])
                per_layer[l].append(kl)
        means = [np.mean(per_layer[l]) for l in monitor_layers]
        ax.plot(monitor_layers, means, 'o-', color=cond_color, label=cond_name,
                linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Per-Layer Anomaly Signal', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        "Neural Firewall Summary\n\n"
        "Monitor Layers: %s\n"
        "Detection Method: KL Divergence\n\n"
        "ROC AUC: %.3f\n"
        "Optimal Threshold: %.3f\n"
        "  TPR = %.0f%%\n"
        "  FPR = %.0f%%\n\n"
        "Benign samples: %d\n"
        "Adversarial samples: %d" % (
            str(monitor_layers), roc_auc, best_threshold,
            best_tpr * 100, best_fpr * 100,
            len(benign_scores), len(adv_scores)
        )
    )
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Phase 180: Neural Firewall\n'
                 '"Can we detect adversarial soul injection in real-time?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase180_neural_firewall.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 180, 'name': 'neural_firewall',
        'roc_auc': round(roc_auc, 4),
        'best_threshold': round(best_threshold, 4),
        'best_tpr': round(best_tpr, 4),
        'best_fpr': round(best_fpr, 4),
        'n_benign': len(benign_scores),
        'n_adversarial': len(adv_scores),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase180_neural_firewall.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P180 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
