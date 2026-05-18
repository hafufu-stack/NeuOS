# -*- coding: utf-8 -*-
"""
Phase 25: Register Algebra (Opus Original)
Can we do vector arithmetic on execution registers to
CREATE NEW PROGRAMS that never existed?

Method:
  1. Extract MIN_vec and MAX_vec from L16
  2. Compute RANGE_vec = MAX_vec - MIN_vec (should compute max-min = range)
  3. Compute AVG_vec = (MIN_vec + MAX_vec) / 2 (should compute average?)
  4. Compute NEGATION_vec = 2*SUM_vec - MIN_vec (algebraic program synthesis)
  5. Inject synthesized vectors and check outputs

If this works: we can COMPILE programs in latent space!

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


def extract_register(model, tok, prompts, layer):
    vecs = []
    for prompt in prompts:
        captured = [None]
        def cap(module, input, output):
            captured[0] = get_last_token(output)
        h = model.model.layers[layer].register_forward_hook(cap)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        vecs.append(captured[0])
    return torch.stack(vecs).mean(dim=0)


def test_injection(model, tok, vec, layer, test_data, expected_fn):
    """Inject vec at layer, return accuracy and predictions."""
    correct = 0
    total = 0
    preds = []
    for data_str, a, b in test_data:
        expected = expected_fn(a, b)
        if expected is None or (isinstance(expected, (int, float)) and expected >= 10):
            preds.append('N/A')
            continue
        total += 1

        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)

        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(data_str, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()

        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append(pred)
        if pred == str(expected):
            correct += 1

    acc = correct / total if total > 0 else 0
    return acc, preds


def main():
    print("[P25] Register Algebra (Opus Original)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    LAYER = 16  # The universal execution port from P24

    # === Extract base registers at L16 ===
    print("  Extracting base registers at L16...")
    min_vec = extract_register(model, tok,
        [f"def f(): return min({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:12], LAYER)
    max_vec = extract_register(model, tok,
        [f"def f(): return max({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:12], LAYER)
    sum_vec = extract_register(model, tok,
        [f"def f(): return {a} + {b} =" for a in range(1, 6) for b in range(1, 6) if a+b < 10][:12], LAYER)
    sub_vec = extract_register(model, tok,
        [f"def f(): return {a} - {b} =" for a in range(3, 9) for b in range(1, 4) if a > b][:12], LAYER)

    # Cosine similarity between registers
    from sklearn.metrics.pairwise import cosine_similarity
    vecs_np = {
        'MIN': min_vec.float().cpu().numpy().flatten(),
        'MAX': max_vec.float().cpu().numpy().flatten(),
        'SUM': sum_vec.float().cpu().numpy().flatten(),
        'SUB': sub_vec.float().cpu().numpy().flatten(),
    }
    print("  L16 Register similarity:")
    for n1 in vecs_np:
        sims = []
        for n2 in vecs_np:
            sim = cosine_similarity(vecs_np[n1].reshape(1,-1), vecs_np[n2].reshape(1,-1))[0,0]
            sims.append(f"{sim:.3f}")
        print(f"    {n1}: [{', '.join(sims)}]")

    # === Synthesize new programs via vector algebra ===
    print("\n  Synthesizing programs via register algebra...")

    # 1. MIDPOINT = (MIN + MAX) / 2 — should produce... what?
    mid_vec = (min_vec + max_vec) / 2

    # 2. DIFF = MAX - MIN — range/difference direction
    diff_vec = max_vec - min_vec

    # 3. ANTI-MIN = 2*MAX - MIN — push away from MIN toward MAX
    anti_min_vec = 2 * max_vec - min_vec

    # 4. BLEND(0.7) = 0.3*MIN + 0.7*MAX — weighted blend
    blend_vec = 0.3 * min_vec + 0.7 * max_vec

    # 5. SUM-MIN = SUM - MIN + MAX — algebraic combination
    sum_min_max_vec = sum_vec - min_vec + max_vec

    test_data = [
        ("3, 7) =", 3, 7),
        ("2, 8) =", 2, 8),
        ("5, 1) =", 5, 1),
        ("4, 6) =", 4, 6),
        ("7, 2) =", 7, 2),
        ("3, 5) =", 3, 5),
    ]

    # Test base operations first
    results = {}
    print("\n  Base operations:")
    for name, vec, fn in [
        ('MIN', min_vec, min),
        ('MAX', max_vec, max),
        ('SUM', sum_vec, lambda a, b: a + b),
        ('SUB', sub_vec, lambda a, b: a - b),
    ]:
        acc, preds = test_injection(model, tok, vec, LAYER, test_data, fn)
        results[name] = {'accuracy': round(acc, 4), 'predictions': preds}
        print(f"    {name}: {acc:.1%} preds={preds}")

    # Test synthesized operations
    print("\n  Synthesized programs:")
    synth_ops = [
        ('MID=(MIN+MAX)/2', mid_vec, lambda a, b: None),  # unknown target
        ('DIFF=MAX-MIN', diff_vec, lambda a, b: abs(a-b)),
        ('ANTI-MIN=2MAX-MIN', anti_min_vec, lambda a, b: max(a,b)),
        ('BLEND=0.3MIN+0.7MAX', blend_vec, lambda a, b: max(a,b)),
        ('SUM-MIN+MAX', sum_min_max_vec, lambda a, b: a + b),
    ]

    for name, vec, fn in synth_ops:
        # Collect predictions regardless of expected
        preds = []
        for data_str, a, b in test_data:
            def inject(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[LAYER].register_forward_hook(inject)
            inp = tok(data_str, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            preds.append(pred)

        # Analyze: what function does this synthesized vector compute?
        pattern = []
        for pred, (_, a, b) in zip(preds, test_data):
            try:
                p = int(pred)
                pattern.append({
                    'pred': p, 'a': a, 'b': b,
                    'is_min': p == min(a,b), 'is_max': p == max(a,b),
                    'is_sum': p == a+b, 'is_diff': p == abs(a-b),
                    'is_a': p == a, 'is_b': p == b,
                })
            except ValueError:
                pattern.append({'pred': pred, 'a': a, 'b': b})

        # Count matches
        n_valid = sum(1 for p in pattern if isinstance(p.get('pred'), int))
        if n_valid > 0:
            min_match = sum(p.get('is_min', False) for p in pattern) / n_valid
            max_match = sum(p.get('is_max', False) for p in pattern) / n_valid
            sum_match = sum(p.get('is_sum', False) for p in pattern) / n_valid
            diff_match = sum(p.get('is_diff', False) for p in pattern) / n_valid
        else:
            min_match = max_match = sum_match = diff_match = 0

        results[name] = {
            'predictions': preds,
            'min_match': round(min_match, 4),
            'max_match': round(max_match, 4),
            'sum_match': round(sum_match, 4),
            'diff_match': round(diff_match, 4),
        }
        best_fn = max([('MIN', min_match), ('MAX', max_match), ('SUM', sum_match), ('DIFF', diff_match)],
                      key=lambda x: x[1])
        print(f"    {name}: preds={preds} -> closest to {best_fn[0]}({best_fn[1]:.0%})")

    # Save
    output = {
        'phase': 25, 'name': 'register_algebra',
        'injection_layer': LAYER,
        'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase25_algebra.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Base operations
    base_ops = ['MIN', 'MAX', 'SUM', 'SUB']
    base_accs = [results[op]['accuracy'] for op in base_ops]
    colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:purple']
    axes[0].bar(base_ops, base_accs, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Base Operations @L16', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(base_accs):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Synthesized: function similarity
    synth_names = [name for name, _, _ in synth_ops]
    min_scores = [results[n].get('min_match', 0) for n in synth_names]
    max_scores = [results[n].get('max_match', 0) for n in synth_names]
    sum_scores = [results[n].get('sum_match', 0) for n in synth_names]

    x = np.arange(len(synth_names))
    w = 0.25
    axes[1].bar(x - w, min_scores, w, label='=MIN', color='tab:green')
    axes[1].bar(x, max_scores, w, label='=MAX', color='tab:red')
    axes[1].bar(x + w, sum_scores, w, label='=SUM', color='tab:blue')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.split('=')[0] for n in synth_names], fontsize=8, rotation=20)
    axes[1].set_ylabel('Match Rate', fontsize=11)
    axes[1].set_title('Synthesized Programs\nWhat function do they compute?', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 1.1)

    axes[2].axis('off')
    summary = "Register Algebra\n\n"
    summary += "Base @L16:\n"
    for op in base_ops:
        summary += f"  {op}: {results[op]['accuracy']:.0%}\n"
    summary += "\nSynthesized:\n"
    for name, _, _ in synth_ops:
        r = results[name]
        best = max([('MIN', r['min_match']), ('MAX', r['max_match']),
                    ('SUM', r['sum_match']), ('DIFF', r['diff_match'])], key=lambda x: x[1])
        summary += f"  {name.split('=')[0]}: -> {best[0]}({best[1]:.0%})\n"
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=11, va='center', ha='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 25: Register Algebra\nCan we synthesize new programs via vector arithmetic?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase25_algebra.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
