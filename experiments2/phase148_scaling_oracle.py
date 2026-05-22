# -*- coding: utf-8 -*-
"""
Phase 148: The Scaling Oracle
Can NeuOS predict its own failure and estimate needed capacity?

"The wise model knows the limits of its own wisdom."
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


def train_soul(model, tok, data, device, layer=LAYER, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(42)
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


def compute_output_entropy(model, tok, prompt, device, soul_vec=None, layer=LAYER):
    """Run inference and return (prediction, entropy, top1_prob, correct_flag)."""
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
    pred_token = tok.decode(logits.argmax().item()).strip()
    return pred_token, entropy, top1_prob


def main():
    print("[P148] The Scaling Oracle")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train MIN soul for evaluation
    min_train = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                  ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                  ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                  ("1, 3) =","1")]
    print("  Training MIN soul (n=10)...")
    soul_min = train_soul(model, tok, min_train, DEVICE)

    # Define tasks of increasing difficulty
    tasks = []

    # Easy: single-digit MIN (should succeed with soul)
    easy_cases = [
        ("7, 2) =", "2"), ("6, 3) =", "3"), ("2, 9) =", "2"),
        ("1, 5) =", "1"), ("8, 4) =", "4"), ("3, 6) =", "3"),
        ("9, 1) =", "1"), ("4, 7) =", "4"), ("5, 8) =", "5"),
        ("2, 6) =", "2"),
    ]
    for prompt, expected in easy_cases:
        tasks.append(('easy', prompt, expected, True))

    # Medium: two-digit numbers (harder for soul)
    medium_cases = [
        ("34, 71) =", "34"), ("22, 15) =", "15"), ("89, 43) =", "43"),
        ("56, 78) =", "56"), ("11, 99) =", "11"), ("67, 23) =", "23"),
        ("45, 12) =", "12"), ("93, 41) =", "41"), ("18, 76) =", "18"),
        ("52, 37) =", "37"),
    ]
    for prompt, expected in medium_cases:
        tasks.append(('medium', prompt, expected, True))

    # Hard: addition (no soul, raw model)
    hard_cases = [
        ("234 + 567 =", "801"), ("123 + 456 =", "579"), ("345 + 678 =", "1023"),
        ("111 + 222 =", "333"), ("999 + 1 =", "1000"), ("456 + 789 =", "1245"),
        ("321 + 654 =", "975"), ("100 + 200 =", "300"), ("555 + 444 =", "999"),
        ("876 + 543 =", "1419"),
    ]
    for prompt, expected in hard_cases:
        tasks.append(('hard', prompt, expected, False))

    # Very hard: multiplication
    vhard_cases = [
        ("12 * 34 =", "408"), ("56 * 78 =", "4368"), ("23 * 45 =", "1035"),
        ("67 * 89 =", "5963"), ("11 * 99 =", "1089"), ("33 * 77 =", "2541"),
        ("44 * 55 =", "2420"), ("16 * 32 =", "512"), ("25 * 25 =", "625"),
        ("13 * 17 =", "221"),
    ]
    for prompt, expected in vhard_cases:
        tasks.append(('very_hard', prompt, expected, False))

    print("  Evaluating %d tasks across 4 difficulty levels..." % len(tasks))
    results_list = []
    for difficulty, prompt, expected, use_soul in tasks:
        sv = soul_min if use_soul else None
        pred, entropy, top1_prob = compute_output_entropy(
            model, tok, prompt, DEVICE, sv, LAYER)
        correct = (pred == expected)
        results_list.append({
            'difficulty': difficulty, 'prompt': prompt, 'expected': expected,
            'predicted': pred, 'correct': correct,
            'entropy': round(entropy, 4), 'top1_prob': round(top1_prob, 6),
            'use_soul': use_soul,
        })

    # Analyze by difficulty
    for diff in ['easy', 'medium', 'hard', 'very_hard']:
        subset = [r for r in results_list if r['difficulty'] == diff]
        acc = sum(1 for r in subset if r['correct']) / len(subset)
        avg_ent = np.mean([r['entropy'] for r in subset])
        avg_conf = np.mean([r['top1_prob'] for r in subset])
        print("  %s: acc=%.0f%% avg_entropy=%.2f avg_conf=%.4f" % (
            diff, acc*100, avg_ent, avg_conf))

    # Build Scaling Oracle: entropy threshold for failure detection
    entropies = [r['entropy'] for r in results_list]
    labels = [0 if r['correct'] else 1 for r in results_list]  # 1=failure

    # Find optimal threshold
    thresholds = np.linspace(min(entropies), max(entropies), 100)
    best_f1 = 0
    best_thresh = 0
    roc_points = []
    for t in thresholds:
        preds = [1 if e > t else 0 for e in entropies]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        roc_points.append((fpr, tpr))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print("  Scaling Oracle: best_threshold=%.2f, best_F1=%.2f" % (best_thresh, best_f1))

    # Capacity estimation
    easy_entropy = np.mean([r['entropy'] for r in results_list if r['difficulty'] == 'easy'])
    capacity_estimates = []
    for r in results_list:
        if not r['correct']:
            entropy_ratio = r['entropy'] / (easy_entropy + 1e-10)
            est_params = 0.5 * entropy_ratio  # simple linear model
            r['estimated_params_B'] = round(est_params, 1)
            capacity_estimates.append(est_params)
        else:
            r['estimated_params_B'] = 0.5

    # Generate self-reports
    self_reports = []
    for r in results_list[:5]:
        if r['correct']:
            report = "Task '%s': SOLVABLE (entropy=%.1f, conf=%.4f). Current capacity sufficient." % (
                r['prompt'][:20], r['entropy'], r['top1_prob'])
        else:
            report = ("Task '%s': EXCEEDS CAPACITY (entropy=%.1f). "
                      "Estimated requirement: ~%.1fB parameters.") % (
                r['prompt'][:20], r['entropy'], r.get('estimated_params_B', 0.5))
        self_reports.append(report)
        print("  " + report)

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Entropy vs Task Difficulty
    ax = axes[0]
    colors = {'easy': '#4CAF50', 'medium': '#FF9800', 'hard': '#F44336', 'very_hard': '#9C27B0'}
    markers = {'easy': 'o', 'medium': 's', 'hard': '^', 'very_hard': 'D'}
    diff_order = {'easy': 0, 'medium': 1, 'hard': 2, 'very_hard': 3}
    for r in results_list:
        x = diff_order[r['difficulty']] + np.random.uniform(-0.15, 0.15)
        edge = 'black' if r['correct'] else 'red'
        ax.scatter(x, r['entropy'], c=colors[r['difficulty']],
                   marker=markers[r['difficulty']], edgecolors=edge,
                   s=80, linewidths=2, zorder=5)
    ax.axhline(y=best_thresh, color='red', linestyle='--', alpha=0.7,
               label='Oracle Threshold (%.1f)' % best_thresh)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Easy\n(1-digit MIN)', 'Medium\n(2-digit MIN)',
                         'Hard\n(3-digit ADD)', 'Very Hard\n(2-digit MUL)'],
                        fontsize=8)
    ax.set_ylabel('Output Entropy')
    ax.set_title('Entropy vs Task Difficulty\n(filled=correct, red edge=incorrect)',
                 fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 2: ROC Curve
    ax = axes[1]
    fprs = [p[0] for p in roc_points]
    tprs = [p[1] for p in roc_points]
    ax.plot(fprs, tprs, 'b-', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.fill_between(fprs, tprs, alpha=0.1, color='blue')
    # Compute AUC
    auc = np.trapz(sorted(set(zip(fprs, tprs)), key=lambda x: x[0]),
                   ) if len(set(fprs)) > 1 else 0.5
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Failure Prediction ROC\n(F1=%.2f at threshold=%.1f)' % (
        best_f1, best_thresh), fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 3: Self-report examples
    ax = axes[2]
    ax.axis('off')
    # Select representative examples
    examples = []
    for diff in ['easy', 'medium', 'hard', 'very_hard']:
        subset = [r for r in results_list if r['difficulty'] == diff]
        r = subset[0]
        status = 'OK' if r['correct'] else 'FAIL'
        est = '%.1fB' % r.get('estimated_params_B', 0.5)
        examples.append([diff, r['prompt'][:15], r['expected'],
                         r['predicted'], '%.1f' % r['entropy'],
                         status, est])

    table = ax.table(cellText=examples,
                     colLabels=['Difficulty', 'Task', 'Expected', 'Got',
                                'Entropy', 'Status', 'Est. Params'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.0)
    for j in range(7):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i, ex in enumerate(examples):
        color = '#C8E6C9' if ex[5] == 'OK' else '#FFCDD2'
        for j in range(7):
            table[i+1, j].set_facecolor(color)
    ax.set_title('Scaling Oracle Self-Reports', fontweight='bold', pad=20)

    plt.suptitle('Phase 148: The Scaling Oracle\n'
                 '"The wise model knows the limits of its own wisdom"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase148_scaling_oracle.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 148, 'name': 'scaling_oracle',
        'n_tasks': len(tasks),
        'accuracy_by_difficulty': {},
        'best_threshold': round(best_thresh, 4),
        'best_f1': round(best_f1, 4),
        'easy_avg_entropy': round(easy_entropy, 4),
        'self_reports': self_reports,
        'task_results': results_list,
        'elapsed': round(time.time() - start, 1),
    }
    for diff in ['easy', 'medium', 'hard', 'very_hard']:
        subset = [r for r in results_list if r['difficulty'] == diff]
        output['accuracy_by_difficulty'][diff] = round(
            sum(1 for r in subset if r['correct']) / len(subset), 4)

    with open(os.path.join(RESULTS_DIR, 'phase148_scaling_oracle.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
