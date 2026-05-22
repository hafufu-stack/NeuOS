# -*- coding: utf-8 -*-
"""
Phase 160: Soul Algebra
Can we create new operations by vector arithmetic on existing souls?

Tests:
- MAX - MIN = ???  (direction of "prefer larger")
- MIN + MAX = ???  (sum of opposites)
- (MIN + MAX) / 2 = ???  (average = identity?)
- MIN * -1 = ???  (negation = MAX?)
- Interpolation: alpha*MIN + (1-alpha)*MAX for alpha in [0,1]

"If souls are vectors, do they obey the laws of linear algebra?"
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


def evaluate_detailed(model, tok, soul_vec, test_data, device, layer=LAYER):
    preds = []
    for prompt, expected in test_data:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append({'prompt': prompt, 'expected': expected, 'pred': pred,
                      'correct': pred == expected})
    acc = sum(1 for p in preds if p['correct']) / len(preds) if preds else 0
    return acc, preds


def main():
    print("[P160] Soul Algebra")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train base souls
    print("  Training MIN, MAX, ADD, SUB souls...")
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]

    soul_min = train_soul(model, tok, min_data, DEVICE, seed=42)
    soul_max = train_soul(model, tok, max_data, DEVICE, seed=43)

    # Test data for evaluation
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("1, 5) =","1"),("8, 4) =","4")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("1, 5) =","5"),("8, 4) =","8")]

    # Baselines
    min_acc, _ = evaluate_detailed(model, tok, soul_min, min_test, DEVICE)
    max_acc, _ = evaluate_detailed(model, tok, soul_max, max_test, DEVICE)
    print("  MIN baseline: %.0f%%" % (min_acc * 100))
    print("  MAX baseline: %.0f%%" % (max_acc * 100))

    # Soul algebra operations
    algebra_ops = {
        'MIN': soul_min,
        'MAX': soul_max,
        '-MIN (negation)': -soul_min,
        '-MAX (negation)': -soul_max,
        'MAX - MIN': soul_max - soul_min,
        'MIN - MAX': soul_min - soul_max,
        '(MIN+MAX)/2': (soul_min + soul_max) / 2,
        'MIN * 0.5': soul_min * 0.5,
        'MIN * 2.0': soul_min * 2.0,
        'MAX * 0.5': soul_max * 0.5,
        'MAX * 2.0': soul_max * 2.0,
    }

    # Add interpolation
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for alpha in alphas:
        name = 'interp_%.1f' % alpha
        algebra_ops[name] = alpha * soul_min + (1 - alpha) * soul_max

    results = {}
    print("\n  Evaluating algebraic souls on MIN and MAX tests...")
    for op_name, vec in algebra_ops.items():
        min_a, min_p = evaluate_detailed(model, tok, vec, min_test, DEVICE)
        max_a, max_p = evaluate_detailed(model, tok, vec, max_test, DEVICE)
        results[op_name] = {
            'min_acc': round(min_a, 4), 'max_acc': round(max_a, 4),
            'min_preds': [p['pred'] for p in min_p],
            'max_preds': [p['pred'] for p in max_p],
            'norm': round(vec.norm().item(), 4),
        }
        print("  %-20s | MIN=%.0f%% MAX=%.0f%% | norm=%.1f" % (
            op_name, min_a*100, max_a*100, vec.norm().item()))

    # Cosine similarity matrix between base souls
    cos_matrix = {}
    base_souls = {'MIN': soul_min, 'MAX': soul_max}
    for n1, v1 in base_souls.items():
        for n2, v2 in base_souls.items():
            cos = torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)).item()
            cos_matrix['%s_vs_%s' % (n1, n2)] = round(cos, 4)
    print("\n  Cosine: MIN-MAX = %.4f" % cos_matrix['MIN_vs_MAX'])

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Interpolation curve
    ax = axes[0]
    interp_min = [results['interp_%.1f' % a]['min_acc'] for a in alphas]
    interp_max = [results['interp_%.1f' % a]['max_acc'] for a in alphas]
    ax.plot(alphas, interp_min, 'ro-', linewidth=2, markersize=8, label='MIN accuracy')
    ax.plot(alphas, interp_max, 'bo-', linewidth=2, markersize=8, label='MAX accuracy')
    ax.fill_between(alphas, interp_min, alpha=0.1, color='red')
    ax.fill_between(alphas, interp_max, alpha=0.1, color='blue')
    ax.set_xlabel('alpha (0=MAX, 1=MIN)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Soul Interpolation\nalpha*MIN + (1-alpha)*MAX', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    # Panel 2: Algebra operation results
    ax = axes[1]
    ops_to_show = ['MIN', 'MAX', '-MIN (negation)', '-MAX (negation)',
                   'MAX - MIN', '(MIN+MAX)/2', 'MIN * 0.5', 'MIN * 2.0']
    x = np.arange(len(ops_to_show))
    w = 0.35
    min_vals = [results[op]['min_acc'] for op in ops_to_show]
    max_vals = [results[op]['max_acc'] for op in ops_to_show]
    ax.barh(x - w/2, min_vals, w, label='on MIN test', color='#E91E63', edgecolor='black')
    ax.barh(x + w/2, max_vals, w, label='on MAX test', color='#2196F3', edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels([op[:18] for op in ops_to_show], fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.legend(fontsize=8)
    ax.set_title('Soul Algebra Operations', fontweight='bold')
    ax.set_xlim(0, 1.1)

    # Panel 3: Norm vs accuracy
    ax = axes[2]
    all_norms = [results[op]['norm'] for op in results]
    all_min_accs = [results[op]['min_acc'] for op in results]
    all_max_accs = [results[op]['max_acc'] for op in results]
    ax.scatter(all_norms, all_min_accs, c='red', s=60, label='MIN acc', alpha=0.7)
    ax.scatter(all_norms, all_max_accs, c='blue', s=60, label='MAX acc', alpha=0.7)
    ax.set_xlabel('Soul Vector Norm')
    ax.set_ylabel('Accuracy')
    ax.set_title('Norm vs Accuracy\n(is magnitude important?)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 160: Soul Algebra\n'
                 '"If souls are vectors, do they obey linear algebra?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase160_soul_algebra.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 160, 'name': 'soul_algebra',
        'baselines': {'min': round(min_acc, 4), 'max': round(max_acc, 4)},
        'algebra_results': results,
        'cosine_matrix': cos_matrix,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase160_soul_algebra.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
