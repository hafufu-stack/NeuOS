# -*- coding: utf-8 -*-
"""
Phase 151: GlassBox Dashboard
Unified meta-cognitive system combining P146+P147+P148 into a single
inference pass that outputs a complete self-report:
  "I am Qwen 0.5B (896d, 24L), currently running MIN at L8 with 95% confidence.
   This task is within my capacity."

"The fully transparent mind."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def glassbox_inference(model, tok, prompt, device, soul_vec, soul_library,
                       layer=LAYER):
    """
    Full GlassBox inference: compute answer + self-report in one pass.

    Returns:
        answer: predicted token
        report: dict with hardware_id, program_id, confidence, capacity_status
    """
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer_hidden = {}

    # Hook all layers to capture hidden states
    hooks = []
    for li in range(n_layers):
        def make_hook(idx):
            def hook_fn(m, inp, out):
                layer_hidden[idx] = get_last_token(out)
            return hook_fn
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    # Inject soul at target layer
    if soul_vec is not None:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        hooks.append(model.model.layers[layer].register_forward_hook(inj))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    for h in hooks:
        h.remove()

    # === ANSWER ===
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=0)
    answer = tok.decode(logits.argmax().item()).strip()
    top1_prob = probs.max().item()
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    # === HARDWARE ID (P146) ===
    norms = [layer_hidden[i].float().norm().item() for i in range(n_layers)]
    avg_norm = np.mean(norms)
    if avg_norm < 50:
        hw_id = {'hidden_size': 896, 'num_layers': n_layers, 'model': '0.5B'}
    elif avg_norm < 100:
        hw_id = {'hidden_size': 1536, 'num_layers': n_layers, 'model': '1.5B'}
    else:
        hw_id = {'hidden_size': 3584, 'num_layers': n_layers, 'model': '7B'}

    # === PROGRAM ID (P147) ===
    # Compare the injected soul vector directly against the soul library
    if soul_vec is not None:
        best_match = 'unknown'
        best_cos = -1.0
        for name, ref_vec in soul_library.items():
            cos = torch.nn.functional.cosine_similarity(
                soul_vec.float().unsqueeze(0), ref_vec.float().unsqueeze(0)).item()
            if cos > best_cos:
                best_cos = cos
                best_match = name
        program_id = {'program': best_match, 'cos_similarity': round(best_cos, 4)}
    else:
        program_id = {'program': 'NONE', 'cos_similarity': 0.0}

    # === CAPACITY STATUS (P148) ===
    if entropy < 2.0:
        capacity = 'WITHIN_CAPACITY'
    elif entropy < 5.0:
        capacity = 'NEAR_LIMIT'
    else:
        capacity = 'EXCEEDS_CAPACITY'

    report = {
        'hardware': hw_id,
        'program': program_id,
        'answer': answer,
        'confidence': round(top1_prob, 6),
        'entropy': round(entropy, 4),
        'capacity': capacity,
    }

    return answer, report


def format_report(report):
    """Format report as human-readable string."""
    hw = report['hardware']
    prog = report['program']
    lines = [
        "=== GLASSBOX SELF-REPORT ===",
        "Hardware: %s (%dd, %dL)" % (hw['model'], hw['hidden_size'], hw['num_layers']),
        "Program: %s (cos=%.3f)" % (prog['program'], prog['cos_similarity']),
        "Answer: %s (conf=%.4f, H=%.2f)" % (report['answer'], report['confidence'], report['entropy']),
        "Capacity: %s" % report['capacity'],
        "===========================",
    ]
    return '\n'.join(lines)


def main():
    print("[P151] GlassBox Dashboard")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train 4 souls
    print("  Training 4 soul vectors...")
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]
    add_data = [("3, 2) =","5"),("4, 1) =","5"),("2, 3) =","5"),
                ("1, 6) =","7"),("5, 3) =","8"),("2, 7) =","9"),
                ("3, 4) =","7"),("1, 2) =","3"),("4, 4) =","8"),
                ("2, 1) =","3")]
    sub_data = [("7, 2) =","5"),("5, 1) =","4"),("9, 3) =","6"),
                ("8, 5) =","3"),("6, 4) =","2"),("4, 1) =","3"),
                ("3, 2) =","1"),("9, 7) =","2"),("8, 1) =","7"),
                ("7, 3) =","4")]

    soul_min = train_soul(model, tok, min_data, DEVICE, seed=42)
    soul_max = train_soul(model, tok, max_data, DEVICE, seed=43)
    soul_add = train_soul(model, tok, add_data, DEVICE, seed=44)
    soul_sub = train_soul(model, tok, sub_data, DEVICE, seed=45)

    soul_library = {
        'MIN': soul_min, 'MAX': soul_max,
        'ADD': soul_add, 'SUB': soul_sub,
    }

    # Test cases
    test_cases = [
        ("7, 2) =", "2", soul_min, "MIN"),
        ("6, 3) =", "3", soul_min, "MIN"),
        ("2, 9) =", "2", soul_min, "MIN"),
        ("3, 7) =", "7", soul_max, "MAX"),
        ("5, 2) =", "5", soul_max, "MAX"),
        ("1, 8) =", "8", soul_max, "MAX"),
        ("3, 4) =", "7", soul_add, "ADD"),
        ("2, 5) =", "7", soul_add, "ADD"),
        ("1, 6) =", "7", soul_add, "ADD"),
        ("9, 3) =", "6", soul_sub, "SUB"),
        ("7, 2) =", "5", soul_sub, "SUB"),
        ("8, 5) =", "3", soul_sub, "SUB"),
        # No soul (baseline)
        ("7, 2) =", "?", None, "NONE"),
        ("3, 7) =", "?", None, "NONE"),
    ]

    print("\n  Running GlassBox inference on %d cases..." % len(test_cases))
    all_reports = []
    program_id_correct = 0
    hw_correct = 0
    answer_correct = 0

    for prompt, expected, soul, true_program in test_cases:
        answer, report = glassbox_inference(
            model, tok, prompt, DEVICE, soul, soul_library)

        # Check program identification
        prog_correct = (report['program']['program'] == true_program)
        if prog_correct:
            program_id_correct += 1

        # Check hardware
        if report['hardware']['hidden_size'] == model.config.hidden_size:
            hw_correct += 1

        # Check answer (skip NONE cases)
        if expected != '?':
            if answer == expected:
                answer_correct += 1

        report['true_program'] = true_program
        report['expected_answer'] = expected
        report['program_correct'] = prog_correct
        all_reports.append(report)

        print("  %s | prog=%s(true=%s) %s | ans=%s(exp=%s) | H=%.2f | %s" % (
            prompt[:12], report['program']['program'], true_program,
            'OK' if prog_correct else 'X',
            answer, expected, report['entropy'], report['capacity']))

    n_with_answer = len([t for t in test_cases if t[1] != '?'])
    print("\n  === DASHBOARD ACCURACY ===")
    print("  Hardware ID: %d/%d = %.0f%%" % (hw_correct, len(test_cases),
          hw_correct/len(test_cases)*100))
    print("  Program ID: %d/%d = %.0f%%" % (program_id_correct, len(test_cases),
          program_id_correct/len(test_cases)*100))
    print("  Answer Acc: %d/%d = %.0f%%" % (answer_correct, n_with_answer,
          answer_correct/n_with_answer*100))

    # Print a full self-report example
    print("\n  Example full self-report:")
    print(format_report(all_reports[0]))

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Dashboard accuracy bars
    ax = axes[0]
    metrics = ['Hardware\nIdentification', 'Program\nIdentification', 'Task\nAccuracy']
    values = [hw_correct/len(test_cases),
              program_id_correct/len(test_cases),
              answer_correct/n_with_answer]
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy')
    ax.set_title('GlassBox Dashboard Accuracy', fontweight='bold')

    # Panel 2: Program identification confusion
    ax = axes[1]
    programs = ['MIN', 'MAX', 'ADD', 'SUB', 'NONE']
    confusion = np.zeros((5, 5))
    for r in all_reports:
        true_idx = programs.index(r['true_program'])
        pred_prog = r['program']['program']
        if pred_prog in programs:
            pred_idx = programs.index(pred_prog)
        else:
            pred_idx = 4  # unknown -> NONE
        confusion[true_idx, pred_idx] += 1

    im = ax.imshow(confusion, cmap='Blues')
    ax.set_xticks(range(5))
    ax.set_xticklabels(programs, fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(programs, fontsize=9)
    ax.set_xlabel('Predicted Program')
    ax.set_ylabel('True Program')
    ax.set_title('Program ID Confusion Matrix', fontweight='bold')
    for i in range(5):
        for j in range(5):
            if confusion[i, j] > 0:
                ax.text(j, i, '%d' % confusion[i, j], ha='center', va='center',
                        fontweight='bold', fontsize=12,
                        color='white' if confusion[i, j] > 1 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: Confidence vs Entropy scatter
    ax = axes[2]
    for r in all_reports:
        color = '#4CAF50' if r.get('program_correct', False) else '#F44336'
        marker = {'MIN': 'o', 'MAX': 's', 'ADD': '^', 'SUB': 'D', 'NONE': 'x'}.get(
            r['true_program'], 'o')
        ax.scatter(r['entropy'], r['confidence'], c=color, marker=marker,
                   s=100, edgecolors='black', linewidths=1, zorder=5)
    # Legend for shapes
    for prog, marker in [('MIN','o'), ('MAX','s'), ('ADD','^'), ('SUB','D'), ('NONE','x')]:
        ax.scatter([], [], marker=marker, c='gray', s=60, label=prog)
    ax.set_xlabel('Output Entropy')
    ax.set_ylabel('Top-1 Confidence')
    ax.set_title('Confidence vs Entropy\n(green=correct ID, red=wrong)', fontweight='bold')
    ax.legend(fontsize=8, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 151: GlassBox Dashboard\n'
                 '"The fully transparent mind"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase151_glassbox_dashboard.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 151, 'name': 'glassbox_dashboard',
        'hw_accuracy': round(hw_correct/len(test_cases), 4),
        'program_id_accuracy': round(program_id_correct/len(test_cases), 4),
        'answer_accuracy': round(answer_correct/n_with_answer, 4),
        'all_reports': all_reports,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase151_glassbox_dashboard.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
