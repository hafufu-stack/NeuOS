# -*- coding: utf-8 -*-
"""
Phase 150: Adaptive Hypervisor
Combines P148 (entropy-based failure detection) with P149 (double-pass).
When the model detects "I can't solve this" via high entropy,
it automatically triggers a second pass to boost capacity.

"A wise OS knows when to think twice."
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


def infer_with_entropy(model, tok, prompt, device, soul_vec=None, layer=LAYER):
    """Single pass returning (prediction, entropy, top1_prob, logits)."""
    hooks = []
    if soul_vec is not None:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        hooks.append(model.model.layers[layer].register_forward_hook(inj))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    for h in hooks:
        h.remove()

    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    top1_prob = probs.max().item()
    pred = tok.decode(logits.argmax().item()).strip()
    return pred, entropy, top1_prob


def adaptive_infer(model, tok, prompt, device, soul_vec, entropy_threshold,
                   layer=LAYER, scale_factor=1.5):
    """Adaptive inference: if entropy is high, boost soul magnitude and retry."""
    pred, entropy, conf = infer_with_entropy(model, tok, prompt, device, soul_vec, layer)

    strategy = 'single'
    attempts = 1

    if entropy > entropy_threshold:
        # Strategy 1: Boost soul magnitude
        boosted = soul_vec * scale_factor
        pred2, ent2, conf2 = infer_with_entropy(model, tok, prompt, device, boosted, layer)
        attempts = 2

        if ent2 < entropy:
            pred, entropy, conf = pred2, ent2, conf2
            strategy = 'boosted_%.1fx' % scale_factor

        # Strategy 2: Try different layer (L6 for arithmetic)
        pred3, ent3, conf3 = infer_with_entropy(model, tok, prompt, device, soul_vec, layer=6)
        attempts = 3

        if ent3 < entropy:
            pred, entropy, conf = pred3, ent3, conf3
            strategy = 'layer_swap_L6'

    return pred, entropy, conf, strategy, attempts


def main():
    print("[P150] Adaptive Hypervisor")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train MIN and MAX souls
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]

    print("  Training MIN and MAX souls...")
    soul_min = train_soul(model, tok, min_data, DEVICE, seed=42)
    soul_max = train_soul(model, tok, max_data, DEVICE, seed=43)

    # Calibration: find entropy threshold from easy tasks
    print("  Calibrating entropy threshold...")
    easy_tests = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                  ("1, 5) =","1"),("8, 4) =","4")]
    easy_entropies = []
    for prompt, expected in easy_tests:
        _, ent, _ = infer_with_entropy(model, tok, prompt, DEVICE, soul_min)
        easy_entropies.append(ent)
    threshold = np.mean(easy_entropies) + 2 * np.std(easy_entropies)
    print("  Threshold: %.4f (mean=%.4f, std=%.4f)" % (
        threshold, np.mean(easy_entropies), np.std(easy_entropies)))

    # Test cases: mix of easy and challenging
    test_cases = [
        # Easy (should pass single)
        ("7, 2) =", "2", soul_min, "MIN(7,2)"),
        ("6, 3) =", "3", soul_min, "MIN(6,3)"),
        ("2, 9) =", "2", soul_min, "MIN(2,9)"),
        ("8, 4) =", "4", soul_min, "MIN(8,4)"),
        ("3, 7) =", "7", soul_max, "MAX(3,7)"),
        ("5, 2) =", "5", soul_max, "MAX(5,2)"),
        # Harder: numbers close together
        ("4, 5) =", "4", soul_min, "MIN(4,5)"),
        ("6, 7) =", "6", soul_min, "MIN(6,7)"),
        ("8, 9) =", "8", soul_min, "MIN(8,9)"),
        ("3, 4) =", "4", soul_max, "MAX(3,4)"),
        # Edge: same number
        ("5, 5) =", "5", soul_min, "MIN(5,5)"),
        ("3, 3) =", "3", soul_max, "MAX(3,3)"),
        # Tricky: reversed order
        ("1, 9) =", "1", soul_min, "MIN(1,9)"),
        ("9, 1) =", "1", soul_min, "MIN(9,1)"),
        ("1, 9) =", "9", soul_max, "MAX(1,9)"),
        ("9, 1) =", "9", soul_max, "MAX(9,1)"),
    ]

    print("\n  --- Standard Inference ---")
    standard_results = []
    for prompt, expected, soul, desc in test_cases:
        pred, ent, conf = infer_with_entropy(model, tok, prompt, DEVICE, soul)
        correct = (pred == expected)
        standard_results.append({
            'desc': desc, 'pred': pred, 'expected': expected,
            'correct': correct, 'entropy': round(ent, 4), 'confidence': round(conf, 6)
        })

    std_acc = sum(1 for r in standard_results if r['correct']) / len(standard_results)
    print("  Standard accuracy: %.0f%%" % (std_acc * 100))

    print("\n  --- Adaptive Inference ---")
    adaptive_results = []
    for prompt, expected, soul, desc in test_cases:
        pred, ent, conf, strategy, attempts = adaptive_infer(
            model, tok, prompt, DEVICE, soul, threshold)
        correct = (pred == expected)
        adaptive_results.append({
            'desc': desc, 'pred': pred, 'expected': expected,
            'correct': correct, 'entropy': round(ent, 4),
            'confidence': round(conf, 6), 'strategy': strategy, 'attempts': attempts
        })
        if strategy != 'single':
            print("  %s: %s -> %s (strategy=%s, attempts=%d)" % (
                desc, expected, pred, strategy, attempts))

    adp_acc = sum(1 for r in adaptive_results if r['correct']) / len(adaptive_results)
    print("  Adaptive accuracy: %.0f%%" % (adp_acc * 100))

    # Strategy breakdown
    strategies = {}
    for r in adaptive_results:
        s = r['strategy']
        if s not in strategies:
            strategies[s] = {'total': 0, 'correct': 0}
        strategies[s]['total'] += 1
        if r['correct']:
            strategies[s]['correct'] += 1

    print("\n  Strategy breakdown:")
    for s, v in strategies.items():
        print("    %s: %d/%d = %.0f%%" % (s, v['correct'], v['total'],
              v['correct']/v['total']*100))

    # Scale factor sweep
    print("\n  Scale factor sweep...")
    scale_results = {}
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        correct = 0
        for prompt, expected, soul, desc in test_cases:
            scaled = soul * scale
            pred, _, _ = infer_with_entropy(model, tok, prompt, DEVICE, scaled)
            if pred == expected:
                correct += 1
        acc = correct / len(test_cases)
        scale_results[scale] = round(acc, 4)
        print("    scale=%.2f: acc=%.0f%%" % (scale, acc * 100))

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Standard vs Adaptive accuracy
    ax = axes[0]
    bars = ax.bar(['Standard\nInference', 'Adaptive\nHypervisor'],
                  [std_acc, adp_acc],
                  color=['#2196F3', '#4CAF50'], edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, [std_acc, adp_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy')
    ax.set_title('Standard vs Adaptive Inference', fontweight='bold')
    # Add strategy counts
    strat_text = '\n'.join(['%s: %d/%d' % (s, v['correct'], v['total'])
                           for s, v in strategies.items()])
    ax.text(0.95, 0.05, strat_text, transform=ax.transAxes, fontsize=8,
            va='bottom', ha='right', bbox=dict(boxstyle='round', facecolor='wheat'))

    # Panel 2: Entropy distribution
    ax = axes[1]
    std_ent = [r['entropy'] for r in standard_results]
    correct_ent = [r['entropy'] for r in standard_results if r['correct']]
    wrong_ent = [r['entropy'] for r in standard_results if not r['correct']]
    if correct_ent:
        ax.hist(correct_ent, bins=15, alpha=0.7, color='#4CAF50', label='Correct', edgecolor='black')
    if wrong_ent:
        ax.hist(wrong_ent, bins=15, alpha=0.7, color='#F44336', label='Wrong', edgecolor='black')
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
               label='Threshold (%.2f)' % threshold)
    ax.set_xlabel('Output Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Entropy Distribution\n(Correct vs Wrong)', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel 3: Scale factor sweep
    ax = axes[2]
    scales = sorted(scale_results.keys())
    accs = [scale_results[s] for s in scales]
    ax.plot(scales, accs, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(scales, accs, alpha=0.1, color='blue')
    ax.axhline(y=std_acc, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Soul Scale Factor')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Soul Magnitude', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 150: Adaptive Hypervisor\n'
                 '"A wise OS knows when to think twice"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase150_adaptive_hypervisor.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 150, 'name': 'adaptive_hypervisor',
        'standard_accuracy': round(std_acc, 4),
        'adaptive_accuracy': round(adp_acc, 4),
        'entropy_threshold': round(threshold, 4),
        'strategy_breakdown': {s: {'acc': v['correct']/v['total'], 'n': v['total']}
                               for s, v in strategies.items()},
        'scale_sweep': scale_results,
        'standard_results': standard_results,
        'adaptive_results': adaptive_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase150_adaptive_hypervisor.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
