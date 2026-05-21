# -*- coding: utf-8 -*-
"""
Phase 135: Latent Soul Verifier
Can the model predict if a soul vector will succeed BEFORE producing output?

"The soul knows its own confidence."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8


def train_soul(model, tok, data, device, layer=LAYER, seed=42, epochs=150):
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


def get_all_layer_outputs(model, tok, vec, prompt, device, layer=LAYER):
    """Run model with soul injection and capture all layer outputs."""
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            layer_outputs[layer_idx] = t[0, -1, :].detach().cpu().numpy()
        return hook_fn

    hooks = []
    for l_idx in range(len(model.model.layers)):
        h = model.model.layers[l_idx].register_forward_hook(make_hook(l_idx))
        hooks.append(h)

    # Injection hook
    def inj(m, i, o, v=vec): return replace_last_token(o, v)
    inj_h = model.model.layers[layer].register_forward_hook(inj)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)

    inj_h.remove()
    for h in hooks:
        h.remove()

    pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
    return layer_outputs, pred


def main():
    print("[P135] Latent Soul Verifier")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    task_data = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")],
    }
    test_data = {
        'MIN': [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")],
        'MAX': [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")],
    }

    # Train souls with multiple seeds for diversity
    print("  Training diverse soul vectors (10 seeds x 2 tasks)...")
    all_data_points = []  # (layer_outputs_dict, correct_bool)

    for task in ['MIN', 'MAX']:
        all_prompts = task_data[task] + test_data[task]
        for seed in range(42, 52):
            vec = train_soul(model, tok, task_data[task], DEVICE, seed=seed)
            for prompt, expected in all_prompts:
                lo, pred = get_all_layer_outputs(model, tok, vec, prompt, DEVICE)
                correct = (pred == expected)
                all_data_points.append((lo, correct, task, prompt, seed))

        # Also test with WRONG soul (MIN soul on MAX data, etc.)
        wrong_task = 'MAX' if task == 'MIN' else 'MIN'
        for seed in [42, 43, 44]:
            vec = train_soul(model, tok, task_data[task], DEVICE, seed=seed)
            for prompt, expected in task_data[wrong_task] + test_data[wrong_task]:
                lo, pred = get_all_layer_outputs(model, tok, vec, prompt, DEVICE)
                correct = (pred == expected)
                all_data_points.append((lo, correct, 'WRONG_%s_on_%s' % (task, wrong_task),
                                        prompt, seed))

    n_correct = sum(1 for _, c, _, _, _ in all_data_points if c)
    n_total = len(all_data_points)
    print("  Collected %d data points (%d correct, %d wrong)" % (
        n_total, n_correct, n_total - n_correct))

    # Train linear probes at each layer to predict success
    n_layers = len(model.model.layers)
    probe_layers = list(range(0, n_layers, 2)) + [n_layers - 1]  # every other layer + last
    probe_layers = sorted(set(probe_layers))

    print("  Training linear probes at %d layers..." % len(probe_layers))
    probe_results = {}
    for probe_layer in probe_layers:
        X = np.array([dp[0][probe_layer] for dp in all_data_points])
        y = np.array([1 if dp[1] else 0 for dp in all_data_points])

        # Simple train/test split (first 70% train, rest test)
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            probe_results[probe_layer] = {'accuracy': 0.5, 'auroc': 0.5}
            continue

        clf = LogisticRegression(max_iter=1000, C=0.1)
        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        try:
            auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        except Exception:
            auroc = 0.5

        probe_results[probe_layer] = {
            'accuracy': round(float(test_acc), 4),
            'auroc': round(float(auroc), 4),
        }
        print("    L%d: acc=%.0f%%, AUROC=%.3f" % (probe_layer, test_acc*100, auroc))

    # Find best probe layer
    best_layer = max(probe_results, key=lambda l: probe_results[l]['auroc'])
    print("  Best verifier layer: L%d (AUROC=%.3f)" % (
        best_layer, probe_results[best_layer]['auroc']))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Probe accuracy across layers
    ax = axes[0]
    layers_plot = sorted(probe_results.keys())
    accs_plot = [probe_results[l]['accuracy'] for l in layers_plot]
    aurocs_plot = [probe_results[l]['auroc'] for l in layers_plot]
    ax.plot(layers_plot, accs_plot, 'o-', color='#2196F3', label='Accuracy',
            markersize=6, linewidth=2)
    ax.plot(layers_plot, aurocs_plot, 's-', color='#FF5722', label='AUROC',
            markersize=6, linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.axvline(x=LAYER, color='green', linestyle=':', alpha=0.5, label='Injection (L8)')
    ax.axvline(x=best_layer, color='red', linestyle=':', alpha=0.5,
               label='Best probe (L%d)' % best_layer)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Score')
    ax.set_title('Verifier Performance by Layer', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Best layer confusion-like bar chart
    ax = axes[1]
    best_acc = probe_results[best_layer]['accuracy']
    best_auroc = probe_results[best_layer]['auroc']
    baseline_acc = n_correct / n_total
    bars = ax.bar(['Majority\nbaseline', 'Best probe\n(L%d)' % best_layer],
                  [baseline_acc, best_acc],
                  color=['#9E9E9E', '#4CAF50'], edgecolor='black')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                "%.1f%%" % (bar.get_height() * 100), ha='center', fontsize=11)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.2)
    ax.set_title('Verifier vs Baseline\n(AUROC=%.3f)' % best_auroc, fontweight='bold')

    # Panel 3: AUROC curve shape across layers
    ax = axes[2]
    ax.fill_between(layers_plot, 0.5, aurocs_plot, alpha=0.3, color='#FF5722')
    ax.plot(layers_plot, aurocs_plot, 'o-', color='#FF5722', linewidth=2, markersize=6)
    ax.axhline(y=0.5, color='gray', linestyle='--')
    ax.set_xlabel('Layer')
    ax.set_ylabel('AUROC')
    ax.set_title('Where Does the Model "Know" It Will Succeed?', fontweight='bold')
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    # Annotate injection and best
    ax.annotate('Injection\n(L%d)' % LAYER, xy=(LAYER, aurocs_plot[layers_plot.index(LAYER)]
                if LAYER in layers_plot else 0.5),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.suptitle('Phase 135: Latent Soul Verifier\n'
                 '"The soul knows its own confidence"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase135_verifier.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 135, 'name': 'latent_soul_verifier',
        'injection_layer': LAYER,
        'n_data_points': n_total,
        'n_correct': n_correct,
        'baseline_accuracy': round(n_correct / n_total, 4),
        'probe_results': {str(k): v for k, v in probe_results.items()},
        'best_layer': best_layer,
        'best_auroc': probe_results[best_layer]['auroc'],
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase135_verifier.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
