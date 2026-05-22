# -*- coding: utf-8 -*-
"""
Phase 165: The 7D Semantic Firewall
Project L8 hidden states onto the 7D PCA subspace, zero out
the remaining 889 "noise" dimensions. Test if accuracy survives
and if adversarial inputs are neutralized.

"Cut the noise. Keep only the signal. Absolute firewall."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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


def evaluate(model, tok, test_data, device, hook_fn=None, layer=LAYER):
    correct = 0
    preds = []
    for prompt, expected in test_data:
        if hook_fn:
            h = model.model.layers[layer].register_forward_hook(hook_fn)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        if hook_fn:
            h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
        preds.append(pred)
    return correct / len(test_data) if test_data else 0, preds


def build_pca_basis(souls_dict, n_components=7):
    """Build PCA from a collection of soul vectors."""
    matrix = np.array([v.cpu().numpy() for v in souls_dict.values()])
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    return pca


def make_firewall_hook(soul_vec, pca, n_keep, device):
    """Create a hook that projects the soul onto the n_keep-dim PCA subspace."""
    # Precompute the projected soul
    v = soul_vec.cpu().numpy().reshape(1, -1)
    projected = pca.transform(v)
    # Zero out dimensions beyond n_keep
    projected[0, n_keep:] = 0
    reconstructed = pca.inverse_transform(projected)
    filtered_soul = torch.tensor(reconstructed[0], dtype=torch.float32, device=device)

    def hook(m, i, o, fv=filtered_soul):
        return replace_last_token(o, fv)
    return hook, filtered_soul


def make_noisy_soul(soul_vec, noise_scale=1.0, seed=99):
    """Add adversarial noise to a soul vector."""
    torch.manual_seed(seed)
    noise = torch.randn_like(soul_vec) * noise_scale
    return soul_vec + noise


def main():
    print("[P165] The 7D Semantic Firewall")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train souls
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("1, 5) =","1"),("8, 4) =","4")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("1, 5) =","5"),("8, 4) =","8")]

    # Train base souls with multiple seeds for PCA
    souls = {}
    for seed in [42, 100, 200, 300]:
        souls['MIN_s%d' % seed] = train_soul(model, tok, min_data, DEVICE, seed=seed)
        souls['MAX_s%d' % seed] = train_soul(model, tok, max_data, DEVICE, seed=seed)

    soul_min = souls['MIN_s42']
    soul_max = souls['MAX_s42']

    # Build PCA basis
    pca = build_pca_basis(souls, n_components=8)
    print("  PCA basis built from %d souls" % len(souls))

    results = {}

    # Test 1: Firewall at different dimensions
    print("\n  === Test 1: Firewall Dimension Sweep ===")
    for n_keep in [1, 2, 3, 4, 5, 6, 7, 8]:
        # MIN with firewall
        hook_fn, _ = make_firewall_hook(soul_min, pca, n_keep, DEVICE)
        min_acc, _ = evaluate(model, tok, min_test, DEVICE, hook_fn=hook_fn)

        hook_fn, _ = make_firewall_hook(soul_max, pca, n_keep, DEVICE)
        max_acc, _ = evaluate(model, tok, max_test, DEVICE, hook_fn=hook_fn)

        results['firewall_%dd' % n_keep] = {
            'n_dims': n_keep, 'min_acc': round(min_acc, 4), 'max_acc': round(max_acc, 4)
        }
        print("  %dd firewall: MIN=%.0f%% MAX=%.0f%%" % (n_keep, min_acc*100, max_acc*100))

    # Baseline: no firewall (raw soul)
    def raw_hook(m, i, o, v=soul_min): return replace_last_token(o, v)
    min_raw, _ = evaluate(model, tok, min_test, DEVICE, hook_fn=raw_hook)
    def raw_hook(m, i, o, v=soul_max): return replace_last_token(o, v)
    max_raw, _ = evaluate(model, tok, max_test, DEVICE, hook_fn=raw_hook)
    results['baseline'] = {'min_acc': round(min_raw, 4), 'max_acc': round(max_raw, 4)}
    print("  Baseline (no firewall): MIN=%.0f%% MAX=%.0f%%" % (min_raw*100, max_raw*100))

    # Test 2: Adversarial noise resistance
    print("\n  === Test 2: Adversarial Noise Resistance ===")
    noise_scales = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    noise_results = {}

    for noise_scale in noise_scales:
        # Without firewall
        noisy_soul = make_noisy_soul(soul_min, noise_scale=noise_scale)
        def noisy_hook(m, i, o, v=noisy_soul): return replace_last_token(o, v)
        raw_acc, raw_preds = evaluate(model, tok, min_test, DEVICE, hook_fn=noisy_hook)

        # With 7D firewall
        hook_fn, filtered = make_firewall_hook(noisy_soul, pca, 7, DEVICE)
        fw_acc, fw_preds = evaluate(model, tok, min_test, DEVICE, hook_fn=hook_fn)

        noise_results[str(noise_scale)] = {
            'noise_scale': noise_scale,
            'raw_acc': round(raw_acc, 4),
            'firewall_acc': round(fw_acc, 4),
            'protection': round(fw_acc - raw_acc, 4),
        }
        print("  noise=%.1f: raw=%.0f%% firewall=%.0f%% (protection=%+.0f pp)" % (
            noise_scale, raw_acc*100, fw_acc*100, (fw_acc-raw_acc)*100))

    results['noise_resistance'] = noise_results

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Dimension sweep
    ax = axes[0]
    dims = [1, 2, 3, 4, 5, 6, 7, 8]
    min_accs = [results['firewall_%dd' % d]['min_acc'] for d in dims]
    max_accs = [results['firewall_%dd' % d]['max_acc'] for d in dims]
    ax.plot(dims, min_accs, 'ro-', linewidth=2, markersize=8, label='MIN')
    ax.plot(dims, max_accs, 'bs-', linewidth=2, markersize=8, label='MAX')
    ax.axhline(y=min_raw, color='red', linestyle='--', alpha=0.4, label='MIN baseline')
    ax.axhline(y=max_raw, color='blue', linestyle='--', alpha=0.4, label='MAX baseline')
    ax.fill_between(dims, min_accs, alpha=0.1, color='red')
    ax.set_xlabel('Firewall Dimensions Kept')
    ax.set_ylabel('Accuracy')
    ax.set_title('7D Firewall: Dimension Sweep\n(889 noisy dims removed)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(dims)

    # Panel 2: Noise resistance
    ax = axes[1]
    ns = [float(s) for s in sorted(noise_results.keys(), key=float)]
    raw_n = [noise_results[str(s)]['raw_acc'] for s in ns]
    fw_n = [noise_results[str(s)]['firewall_acc'] for s in ns]
    ax.plot(ns, raw_n, 'o--', color='#F44336', linewidth=2, markersize=8,
            label='No firewall')
    ax.plot(ns, fw_n, 's-', color='#4CAF50', linewidth=2, markersize=8,
            label='7D firewall')
    ax.fill_between(ns, fw_n, raw_n, alpha=0.2, color='green',
                    label='Protection zone')
    ax.set_xlabel('Adversarial Noise Scale')
    ax.set_ylabel('Accuracy')
    ax.set_title('Noise Resistance\n(firewall vs unprotected)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # Panel 3: Firewall concept diagram
    ax = axes[2]
    ax.axis('off')
    concept = [
        ['Layer', 'Content', 'Dims'],
        ['Input', 'Raw soul vector', '896'],
        ['PCA Project', 'Transform to PCA space', '896 -> 7'],
        ['FIREWALL', 'Zero dims 8-896', '889 -> 0'],
        ['Reconstruct', 'Inverse transform', '7 -> 896'],
        ['Output', 'Clean soul (signal only)', '896'],
    ]
    table = ax.table(cellText=concept[1:], colLabels=concept[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for j in range(3):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    table[3, 0].set_facecolor('#C62828')
    table[3, 1].set_facecolor('#C62828')
    table[3, 2].set_facecolor('#C62828')
    table[3, 0].set_text_props(color='white', fontweight='bold')
    table[3, 1].set_text_props(color='white', fontweight='bold')
    table[3, 2].set_text_props(color='white', fontweight='bold')
    ax.set_title('Semantic Firewall Architecture', fontweight='bold', pad=20)

    plt.suptitle('Phase 165: The 7D Semantic Firewall\n'
                 '"Cut the noise. Keep only the signal."',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase165_firewall.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 165, 'name': '7d_semantic_firewall',
        'results': {k: v for k, v in results.items() if k != 'noise_resistance'},
        'noise_resistance': {k: v for k, v in noise_results.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase165_firewall.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P165 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
