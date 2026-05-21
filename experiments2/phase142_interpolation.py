# -*- coding: utf-8 -*-
"""
Phase 142: Soul Interpolation & Arithmetic
Linear interpolation between MIN and MAX souls.
At what mixing ratio does behavior switch?
Also: soul_A + soul_B, soul_A - soul_B, negation.

"Between MIN and MAX lies a continuous spectrum of mathematical identity."
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


def evaluate_with_vec(model, tok, vec, data, device, layer=LAYER):
    c = 0
    for p, e in data:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0, -1, :].argmax().item()).strip() == e:
            c += 1
    return c / len(data)


def get_predictions(model, tok, vec, prompts, device, layer=LAYER):
    preds = []
    for p in prompts:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append(pred)
    return preds


def main():
    print("[P142] Soul Interpolation & Arithmetic")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")]
    test_prompts_data = [("7, 2) =",""),("6, 3) =",""),("2, 9) =",""),
                          ("1, 5) =",""),("8, 4) =","")]
    test_prompts = [p for p, _ in test_prompts_data]

    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("1, 5) =","1"),("8, 4) =","4")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                 ("1, 5) =","5"),("8, 4) =","8")]

    # Train base souls
    print("  Training MIN and MAX souls...")
    soul_min = train_soul(model, tok, min_data, DEVICE, seed=42)
    soul_max = train_soul(model, tok, max_data, DEVICE, seed=42)

    cos_min_max = torch.nn.functional.cosine_similarity(
        soul_min.unsqueeze(0), soul_max.unsqueeze(0)).item()
    print("  Cosine(MIN, MAX) = %.4f" % cos_min_max)

    # 1. Linear interpolation: alpha * MIN + (1-alpha) * MAX
    alphas = np.arange(0, 1.01, 0.05)
    interp_results = []

    print("  Interpolating (%d steps)..." % len(alphas))
    for alpha in alphas:
        vec = alpha * soul_min + (1 - alpha) * soul_max
        min_acc = evaluate_with_vec(model, tok, vec, min_test, DEVICE)
        max_acc = evaluate_with_vec(model, tok, vec, max_test, DEVICE)
        preds = get_predictions(model, tok, vec, test_prompts, DEVICE)
        interp_results.append({
            'alpha': round(float(alpha), 2),
            'min_acc': round(min_acc, 4),
            'max_acc': round(max_acc, 4),
            'preds': preds
        })

    # 2. Soul arithmetic operations
    print("  Testing soul arithmetic...")
    arith_ops = {
        'MIN': soul_min,
        'MAX': soul_max,
        '-MIN (negation)': -soul_min,
        '-MAX (negation)': -soul_max,
        'MIN + MAX': soul_min + soul_max,
        'MIN - MAX': soul_min - soul_max,
        'MAX - MIN': soul_max - soul_min,
        '2*MIN - MAX': 2 * soul_min - soul_max,
        '2*MAX - MIN': 2 * soul_max - soul_min,
        'AVG (0.5+0.5)': 0.5 * soul_min + 0.5 * soul_max,
    }

    arith_results = {}
    for name, vec in arith_ops.items():
        min_acc = evaluate_with_vec(model, tok, vec, min_test, DEVICE)
        max_acc = evaluate_with_vec(model, tok, vec, max_test, DEVICE)
        preds = get_predictions(model, tok, vec, test_prompts, DEVICE)
        arith_results[name] = {
            'min_acc': round(min_acc, 4),
            'max_acc': round(max_acc, 4),
            'preds': preds
        }
        print("    %s: MIN=%.0f%% MAX=%.0f%% preds=%s" % (
            name, min_acc*100, max_acc*100, preds))

    # 3. Find exact crossover point
    crossover_alpha = None
    for i in range(len(interp_results) - 1):
        r1 = interp_results[i]
        r2 = interp_results[i + 1]
        if r1['max_acc'] > r1['min_acc'] and r2['min_acc'] >= r2['max_acc']:
            crossover_alpha = (r1['alpha'] + r2['alpha']) / 2
            break
    print("  Crossover alpha: %s" % crossover_alpha)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Interpolation curve
    ax = axes[0]
    ax.plot([r['alpha'] for r in interp_results],
            [r['min_acc'] for r in interp_results],
            'o-', color='#2196F3', label='MIN accuracy',
            markersize=4, linewidth=2)
    ax.plot([r['alpha'] for r in interp_results],
            [r['max_acc'] for r in interp_results],
            's-', color='#FF5722', label='MAX accuracy',
            markersize=4, linewidth=2)
    if crossover_alpha:
        ax.axvline(x=crossover_alpha, color='green', linestyle='--',
                   label='Crossover (%.2f)' % crossover_alpha)
    ax.fill_between([r['alpha'] for r in interp_results],
                    [r['min_acc'] for r in interp_results],
                    [r['max_acc'] for r in interp_results],
                    alpha=0.1, color='gray')
    ax.set_xlabel('alpha (0=pure MAX, 1=pure MIN)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Soul Interpolation: MIN <-> MAX', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Soul arithmetic bar chart
    ax = axes[1]
    ops = list(arith_results.keys())
    min_accs = [arith_results[o]['min_acc'] for o in ops]
    max_accs = [arith_results[o]['max_acc'] for o in ops]
    x = np.arange(len(ops))
    w = 0.35
    ax.bar(x - w/2, min_accs, w, label='MIN eval', color='#2196F3',
           edgecolor='black')
    ax.bar(x + w/2, max_accs, w, label='MAX eval', color='#FF5722',
           edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Accuracy')
    ax.set_title('Soul Arithmetic Operations', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)

    # Panel 3: Prediction table for interpolation
    ax = axes[2]
    ax.axis('off')
    # Show predictions at key alpha values
    key_alphas = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    table_data = []
    for alpha in key_alphas:
        idx = int(alpha / 0.05)
        if idx >= len(interp_results):
            idx = len(interp_results) - 1
        r = interp_results[idx]
        row = ["%.1f" % alpha] + r['preds']
        table_data.append(row)
    # Add reference rows
    table_data.append(['[MIN]', '2', '3', '2', '1', '4'])
    table_data.append(['[MAX]', '7', '6', '9', '5', '8'])
    prompts_short = ['7,2', '6,3', '2,9', '1,5', '8,4']
    table = ax.table(cellText=table_data,
                     colLabels=['alpha'] + prompts_short,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(6):
        table[0, j].set_facecolor('#1976D2')
        table[0, j].set_text_props(color='white', fontweight='bold')
    n_rows = len(table_data)
    table[n_rows - 1, 0].set_facecolor('#E3F2FD')
    table[n_rows, 0].set_facecolor('#FBE9E7')
    for j in range(6):
        table[n_rows - 1, j].set_facecolor('#E3F2FD')
        table[n_rows, j].set_facecolor('#FBE9E7')
    ax.set_title('Predictions at Key Alphas', fontweight='bold',
                 fontsize=12, pad=20)

    plt.suptitle('Phase 142: Soul Interpolation & Arithmetic\n'
                 '"Between MIN and MAX lies a continuous spectrum"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase142_interpolation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 142, 'name': 'soul_interpolation',
        'layer': LAYER,
        'cosine_min_max': round(cos_min_max, 4),
        'crossover_alpha': crossover_alpha,
        'interp_results': interp_results,
        'arith_results': {k: v for k, v in arith_results.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase142_interpolation.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
