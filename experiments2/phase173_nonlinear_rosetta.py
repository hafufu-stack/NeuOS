# -*- coding: utf-8 -*-
"""
Phase 173: Nonlinear Cross-Model Rosetta
Can MLPs or kernel methods translate souls between 0.5B and 1.5B?
Phase 120 showed linear fails (15%). We try nonlinear.
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER_05B = 8
LAYER_15B = 10  # Scaled layer for 1.5B (28 layers)


def train_soul(model, tok, data, device, layer, epochs=100, seed=42):
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


def evaluate(model, tok, soul_vec, test_data, device, layer):
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


class MLPTranslator(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("[P173] Nonlinear Cross-Model Rosetta")
    start = time.time()

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

    # --- Train souls on 0.5B ---
    print("  Loading 0.5B model...")
    model_05, tok_05 = load_model('Qwen/Qwen2.5-0.5B', device=DEVICE, surgery=True)
    for p in model_05.parameters():
        p.requires_grad = False

    print("  Training souls on 0.5B...")
    souls_05 = {}
    for seed in [42, 100, 200, 300]:
        souls_05['MIN_s%d' % seed] = train_soul(model_05, tok_05, min_data, DEVICE,
                                                 LAYER_05B, seed=seed)
        souls_05['MAX_s%d' % seed] = train_soul(model_05, tok_05, max_data, DEVICE,
                                                 LAYER_05B, seed=seed)

    # Native accuracy on 0.5B
    min_acc_05 = evaluate(model_05, tok_05, souls_05['MIN_s42'], min_test, DEVICE, LAYER_05B)
    max_acc_05 = evaluate(model_05, tok_05, souls_05['MAX_s42'], max_test, DEVICE, LAYER_05B)
    print("  0.5B native: MIN=%.0f%%, MAX=%.0f%%" % (min_acc_05*100, max_acc_05*100))

    del model_05; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Train souls on 1.5B ---
    print("  Loading 1.5B model...")
    try:
        model_15, tok_15 = load_model('Qwen/Qwen2.5-1.5B', device=DEVICE, surgery=True)
    except Exception as e:
        print("  WARNING: Could not load 1.5B model: %s" % e)
        print("  Skipping cross-model experiment.")
        output = {
            'phase': 173, 'name': 'nonlinear_rosetta',
            'error': 'Could not load Qwen2.5-1.5B: %s' % str(e),
            'elapsed': round(time.time() - start, 1),
        }
        with open(os.path.join(RESULTS_DIR, 'phase173_nonlinear_rosetta.json'), 'w') as f:
            json.dump(output, f, indent=2)
        return

    for p in model_15.parameters():
        p.requires_grad = False

    print("  Training souls on 1.5B...")
    souls_15 = {}
    for seed in [42, 100, 200, 300]:
        souls_15['MIN_s%d' % seed] = train_soul(model_15, tok_15, min_data, DEVICE,
                                                 LAYER_15B, seed=seed)
        souls_15['MAX_s%d' % seed] = train_soul(model_15, tok_15, max_data, DEVICE,
                                                 LAYER_15B, seed=seed)

    min_acc_15 = evaluate(model_15, tok_15, souls_15['MIN_s42'], min_test, DEVICE, LAYER_15B)
    max_acc_15 = evaluate(model_15, tok_15, souls_15['MAX_s42'], max_test, DEVICE, LAYER_15B)
    print("  1.5B native: MIN=%.0f%%, MAX=%.0f%%" % (min_acc_15*100, max_acc_15*100))

    # --- Translation Methods ---
    dim_05 = 896
    dim_15 = 1536

    # Build training pairs: 0.5B soul -> 1.5B soul (matching seeds)
    X_train = np.array([souls_05['MIN_s%d' % s].cpu().numpy() for s in [42, 100, 200]] +
                       [souls_05['MAX_s%d' % s].cpu().numpy() for s in [42, 100, 200]])
    Y_train = np.array([souls_15['MIN_s%d' % s].cpu().numpy() for s in [42, 100, 200]] +
                       [souls_15['MAX_s%d' % s].cpu().numpy() for s in [42, 100, 200]])
    # Hold out seed=300 for test
    X_test = np.array([souls_05['MIN_s300'].cpu().numpy(),
                       souls_05['MAX_s300'].cpu().numpy()])
    Y_test = np.array([souls_15['MIN_s300'].cpu().numpy(),
                       souls_15['MAX_s300'].cpu().numpy()])

    translation_results = {}

    # Method 1: Linear (baseline, reproduce P120)
    print("\n  Method 1: Linear Translation...")
    from sklearn.linear_model import Ridge
    linear = Ridge(alpha=1.0)
    linear.fit(X_train, Y_train)
    Y_pred_linear = linear.predict(X_test)
    for i, (name, test_d) in enumerate([('MIN', min_test), ('MAX', max_test)]):
        translated = torch.tensor(Y_pred_linear[i], dtype=torch.float32, device=DEVICE)
        acc = evaluate(model_15, tok_15, translated, test_d, DEVICE, LAYER_15B)
        translation_results['linear_%s' % name] = round(acc, 4)
        print("    Linear %s: %.0f%%" % (name, acc*100))

    # Method 2: Kernel Ridge Regression
    print("  Method 2: Kernel Ridge Regression...")
    for kernel in ['rbf', 'poly']:
        kr = KernelRidge(alpha=1.0, kernel=kernel)
        kr.fit(X_train, Y_train)
        Y_pred_kr = kr.predict(X_test)
        for i, (name, test_d) in enumerate([('MIN', min_test), ('MAX', max_test)]):
            translated = torch.tensor(Y_pred_kr[i], dtype=torch.float32, device=DEVICE)
            acc = evaluate(model_15, tok_15, translated, test_d, DEVICE, LAYER_15B)
            translation_results['kernel_%s_%s' % (kernel, name)] = round(acc, 4)
            print("    Kernel(%s) %s: %.0f%%" % (kernel, name, acc*100))

    # Method 3: MLP
    print("  Method 3: MLP Translation...")
    mlp = MLPTranslator(dim_05, dim_15, hidden=512).to(DEVICE)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    Y_t = torch.tensor(Y_train, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
    for epoch in range(2000):
        pred = mlp(X_t)
        loss = torch.nn.functional.mse_loss(pred, Y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 500 == 0:
            print("    Epoch %d: loss=%.6f" % (epoch, loss.item()))

    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        Y_pred_mlp = mlp(X_test_t).cpu().numpy()
    for i, (name, test_d) in enumerate([('MIN', min_test), ('MAX', max_test)]):
        translated = torch.tensor(Y_pred_mlp[i], dtype=torch.float32, device=DEVICE)
        acc = evaluate(model_15, tok_15, translated, test_d, DEVICE, LAYER_15B)
        translation_results['mlp_%s' % name] = round(acc, 4)
        print("    MLP %s: %.0f%%" % (name, acc*100))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Translation accuracy comparison
    ax = axes[0]
    methods = ['Linear', 'Kernel(RBF)', 'Kernel(Poly)', 'MLP']
    min_accs = [translation_results.get('linear_MIN', 0),
                translation_results.get('kernel_rbf_MIN', 0),
                translation_results.get('kernel_poly_MIN', 0),
                translation_results.get('mlp_MIN', 0)]
    max_accs = [translation_results.get('linear_MAX', 0),
                translation_results.get('kernel_rbf_MAX', 0),
                translation_results.get('kernel_poly_MAX', 0),
                translation_results.get('mlp_MAX', 0)]
    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax.bar(x - w/2, min_accs, w, label='MIN', color='#E91E63',
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + w/2, max_accs, w, label='MAX', color='#2196F3',
                   edgecolor='black', linewidth=1.5)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    '%.0f%%' % (h*100), ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Accuracy on 1.5B')
    ax.set_title('Cross-Model Translation\n(0.5B -> 1.5B)', fontweight='bold')
    ax.legend()
    ax.axhline(y=min_acc_15, color='#E91E63', linestyle=':', alpha=0.5, label='1.5B native MIN')
    ax.axhline(y=max_acc_15, color='#2196F3', linestyle=':', alpha=0.5, label='1.5B native MAX')

    # Panel 2: Native vs translated
    ax = axes[1]
    categories = ['0.5B Native', '1.5B Native', 'Best Translation']
    best_min = max(min_accs)
    best_max = max(max_accs)
    min_vals = [min_acc_05, min_acc_15, best_min]
    max_vals = [max_acc_05, max_acc_15, best_max]
    x = np.arange(len(categories))
    ax.bar(x - w/2, min_vals, w, label='MIN', color='#E91E63',
           edgecolor='black', linewidth=1.5)
    ax.bar(x + w/2, max_vals, w, label='MAX', color='#2196F3',
           edgecolor='black', linewidth=1.5)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title('Native vs Translated Performance', fontweight='bold')
    ax.legend()

    # Panel 3: Summary text
    ax = axes[2]
    ax.axis('off')
    best_method = methods[np.argmax([sum(p) for p in zip(min_accs, max_accs)])]
    summary = (
        "Cross-Model Soul Translation\n"
        "0.5B (896D) -> 1.5B (1536D)\n\n"
        "Linear baseline (P120): %.0f%% / %.0f%%\n"
        "Best nonlinear (%s):\n  MIN=%.0f%%, MAX=%.0f%%\n\n"
        "1.5B Native:\n  MIN=%.0f%%, MAX=%.0f%%\n\n"
        "Conclusion:\n"
        "%s" % (
            translation_results.get('linear_MIN', 0)*100,
            translation_results.get('linear_MAX', 0)*100,
            best_method, best_min*100, best_max*100,
            min_acc_15*100, max_acc_15*100,
            "Nonlinear helps!" if (best_min + best_max) > (
                translation_results.get('linear_MIN', 0) +
                translation_results.get('linear_MAX', 0)
            ) else "Nonlinear does NOT help."
        )
    )
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Phase 173: Nonlinear Cross-Model Rosetta\n'
                 '"Can the soul transcend its body with nonlinear translation?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase173_nonlinear_rosetta.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 173, 'name': 'nonlinear_rosetta',
        'native_05b': {'MIN': round(min_acc_05, 4), 'MAX': round(max_acc_05, 4)},
        'native_15b': {'MIN': round(min_acc_15, 4), 'MAX': round(max_acc_15, 4)},
        'translation_results': translation_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase173_nonlinear_rosetta.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P173 completed in %.0fs" % (time.time() - start))
    del model_15; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
