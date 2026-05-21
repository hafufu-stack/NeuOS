# -*- coding: utf-8 -*-
"""
Phase 134: Cross-Layer Dual Firmware
Train different task souls at different layers and test dynamic routing
for compound multi-step tasks.

"A brain with two firmware slots routes each thought to its proper layer."
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


def train_soul(model, tok, data, device, layer=8, seed=42, epochs=150):
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


def evaluate(model, tok, vec, data, device, layer=8):
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


def infer_single(model, tok, vec, prompt, device, layer=8):
    """Run inference with soul and return predicted token string."""
    def inj(m, i, o, v=vec): return replace_last_token(o, v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad(): out = model(**inp)
    h.remove()
    pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
    return pred


def main():
    print("[P134] Cross-Layer Dual Firmware")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    task_data = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")],
        'ADD': [("3, 2) =","5"),("1, 4) =","5"),("2, 6) =","8"),
                 ("3, 3) =","6"),("4, 1) =","5")],
        'SUB': [("7, 3) =","4"),("5, 2) =","3"),("9, 1) =","8"),
                 ("6, 4) =","2"),("8, 3) =","5")],
    }
    test_data = {
        'MIN': [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")],
        'MAX': [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")],
        'ADD': [("1, 2) =","3"),("3, 4) =","7"),("2, 5) =","7")],
        'SUB': [("9, 5) =","4"),("6, 1) =","5"),("4, 3) =","1")],
    }

    L8 = 8
    L16 = 16

    # --- Step 1: Train MIN soul at L8 ---
    print("  Training MIN soul at L8...")
    min_soul_L8 = train_soul(model, tok, task_data['MIN'], DEVICE, layer=L8, seed=42)

    # --- Step 2: Train ADD soul at L8 AND L16 ---
    print("  Training ADD soul at L8...")
    add_soul_L8 = train_soul(model, tok, task_data['ADD'], DEVICE, layer=L8, seed=42)
    print("  Training ADD soul at L16...")
    add_soul_L16 = train_soul(model, tok, task_data['ADD'], DEVICE, layer=L16, seed=42)

    # Train MAX soul at L8 and SUB soul at L8 and L16 for compound tasks
    print("  Training MAX soul at L8...")
    max_soul_L8 = train_soul(model, tok, task_data['MAX'], DEVICE, layer=L8, seed=42)
    print("  Training SUB soul at L8...")
    sub_soul_L8 = train_soul(model, tok, task_data['SUB'], DEVICE, layer=L8, seed=42)
    print("  Training SUB soul at L16...")
    sub_soul_L16 = train_soul(model, tok, task_data['SUB'], DEVICE, layer=L16, seed=42)

    # --- Step 3: Single-task accuracy ---
    print("\n  Single-task accuracy:")
    single_accs = {}
    for name, vec, layer, dkey in [
        ('MIN@L8', min_soul_L8, L8, 'MIN'),
        ('ADD@L8', add_soul_L8, L8, 'ADD'),
        ('ADD@L16', add_soul_L16, L16, 'ADD'),
        ('MAX@L8', max_soul_L8, L8, 'MAX'),
        ('SUB@L8', sub_soul_L8, L8, 'SUB'),
        ('SUB@L16', sub_soul_L16, L16, 'SUB'),
    ]:
        train_acc = evaluate(model, tok, vec, task_data[dkey], DEVICE, layer=layer)
        test_acc = evaluate(model, tok, vec, test_data[dkey], DEVICE, layer=layer)
        single_accs[name] = {'train': round(train_acc, 4), 'test': round(test_acc, 4)}
        print("    %s: train=%.0f%% test=%.0f%%" % (name, train_acc*100, test_acc*100))

    # --- Step 4 & 5: Compound tasks ---
    # Define compound test cases: (description, step1_task, step1_layer, step1_soul,
    #                               a, b, step2_task, step2_layer, step2_soul, c, expected)
    # Format: step1(a,b) -> result, then step2(result, c) -> final
    compound_examples = [
        # MIN then ADD: MIN(3,7)=3, ADD(3,2)=5
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 3, 7, 'ADD', L16, add_soul_L16, 2, 5),
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 5, 2, 'ADD', L16, add_soul_L16, 4, 6),
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 8, 1, 'ADD', L16, add_soul_L16, 3, 4),
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 9, 3, 'ADD', L16, add_soul_L16, 1, 4),
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 4, 6, 'ADD', L16, add_soul_L16, 5, 9),
        # MAX then SUB: MAX(3,7)=7, SUB(7,2)=5
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 3, 7, 'SUB', L16, sub_soul_L16, 2, 5),
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 5, 2, 'SUB', L16, sub_soul_L16, 1, 4),
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 8, 1, 'SUB', L16, sub_soul_L16, 3, 5),
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 4, 6, 'SUB', L16, sub_soul_L16, 2, 4),
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 9, 3, 'SUB', L16, sub_soul_L16, 4, 5),
    ]

    # Also build fixed-layer versions (both steps at L8)
    compound_fixed = [
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 3, 7, 'ADD', L8, add_soul_L8, 2, 5),
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 5, 2, 'ADD', L8, add_soul_L8, 4, 6),
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 8, 1, 'ADD', L8, add_soul_L8, 3, 4),
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 9, 3, 'ADD', L8, add_soul_L8, 1, 4),
        ('MIN->ADD', 'MIN', L8, min_soul_L8, 4, 6, 'ADD', L8, add_soul_L8, 5, 9),
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 3, 7, 'SUB', L8, sub_soul_L8, 2, 5),
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 5, 2, 'SUB', L8, sub_soul_L8, 1, 4),
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 8, 1, 'SUB', L8, sub_soul_L8, 3, 5),
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 4, 6, 'SUB', L8, sub_soul_L8, 2, 4),
        ('MAX->SUB', 'MAX', L8, max_soul_L8, 9, 3, 'SUB', L8, sub_soul_L8, 4, 5),
    ]

    def run_compound(examples, label):
        """Run compound task examples, return per-type accuracy."""
        results_by_type = {}
        for desc, t1, l1, s1, a, b, t2, l2, s2, c, expected in examples:
            # Step 1: compute step1(a, b)
            prompt1 = "%d, %d) =" % (a, b)
            pred1 = infer_single(model, tok, s1, prompt1, DEVICE, layer=l1)
            # Step 2: use pred1 with c
            try:
                pred1_int = int(pred1)
            except ValueError:
                pred1_int = -999  # will fail
            prompt2 = "%d, %d) =" % (pred1_int, c) if pred1_int != -999 else "0, 0) ="
            pred2 = infer_single(model, tok, s2, prompt2, DEVICE, layer=l2)

            correct = (pred2.strip() == str(expected))
            step1_correct = (pred1.strip() == str(min(a, b) if t1 == 'MIN' else max(a, b)))

            if desc not in results_by_type:
                results_by_type[desc] = {'total': 0, 'correct': 0,
                                          'step1_correct': 0, 'details': []}
            results_by_type[desc]['total'] += 1
            results_by_type[desc]['correct'] += int(correct)
            results_by_type[desc]['step1_correct'] += int(step1_correct)
            results_by_type[desc]['details'].append({
                'input': '(%d,%d)->%s->(%s,%d)->%s' % (a, b, t1, pred1, c, t2),
                'pred1': pred1, 'pred2': pred2,
                'expected': str(expected), 'correct': correct
            })

        for desc in results_by_type:
            r = results_by_type[desc]
            acc = r['correct'] / r['total'] * 100
            s1acc = r['step1_correct'] / r['total'] * 100
            print("    %s [%s]: step1_acc=%.0f%%, compound_acc=%.0f%% (%d/%d)" % (
                label, desc, s1acc, acc, r['correct'], r['total']))
        return results_by_type

    print("\n  Compound tasks (dynamic routing: comparison@L8, arithmetic@L16):")
    dynamic_results = run_compound(compound_examples, "dynamic")

    print("\n  Compound tasks (fixed layer: all@L8):")
    fixed_results = run_compound(compound_fixed, "fixed")

    # --- Step 6: Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Single-task accuracy bar chart
    ax = axes[0]
    labels = list(single_accs.keys())
    train_vals = [single_accs[k]['train'] * 100 for k in labels]
    test_vals = [single_accs[k]['test'] * 100 for k in labels]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, train_vals, w, label='Train', color='#2196F3',
                   edgecolor='black')
    bars2 = ax.bar(x + w/2, test_vals, w, label='Test', color='#FF5722',
                   edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Single-Task Accuracy per Layer', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 110)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                "%.0f" % bar.get_height(), ha='center', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                "%.0f" % bar.get_height(), ha='center', fontsize=8)

    # Panel 2: Compound task accuracy (fixed vs dynamic)
    ax = axes[1]
    compound_types = sorted(set(k for k in dynamic_results.keys()))
    dyn_accs = []
    fix_accs = []
    for ct in compound_types:
        dr = dynamic_results[ct]
        dyn_accs.append(dr['correct'] / dr['total'] * 100)
        fr = fixed_results[ct]
        fix_accs.append(fr['correct'] / fr['total'] * 100)
    # Add overall
    compound_types.append('Overall')
    total_dyn = sum(dynamic_results[ct]['correct'] for ct in dynamic_results)
    total_n_dyn = sum(dynamic_results[ct]['total'] for ct in dynamic_results)
    total_fix = sum(fixed_results[ct]['correct'] for ct in fixed_results)
    total_n_fix = sum(fixed_results[ct]['total'] for ct in fixed_results)
    dyn_accs.append(total_dyn / total_n_dyn * 100 if total_n_dyn > 0 else 0)
    fix_accs.append(total_fix / total_n_fix * 100 if total_n_fix > 0 else 0)

    x = np.arange(len(compound_types))
    w = 0.35
    bars1 = ax.bar(x - w/2, fix_accs, w, label='Fixed (all@L8)',
                   color='#9E9E9E', edgecolor='black')
    bars2 = ax.bar(x + w/2, dyn_accs, w, label='Dynamic (L8+L16)',
                   color='#4CAF50', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(compound_types, fontsize=10)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Compound Task: Fixed vs Dynamic Routing', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 110)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                "%.0f" % bar.get_height(), ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                "%.0f" % bar.get_height(), ha='center', fontsize=9)

    plt.suptitle('Phase 134: Cross-Layer Dual Firmware\n'
                 '"A brain with two firmware slots routes each thought to its proper layer"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase134_dual_firmware.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    # Convert details for JSON serialization
    def serialize_results(res):
        out = {}
        for k, v in res.items():
            out[k] = {
                'total': v['total'],
                'correct': v['correct'],
                'step1_correct': v['step1_correct'],
                'accuracy': round(v['correct'] / v['total'], 4) if v['total'] > 0 else 0,
                'details': v['details'],
            }
        return out

    output = {
        'phase': 134, 'name': 'cross_layer_dual_firmware',
        'layers': {'comparison': L8, 'arithmetic': L16},
        'single_task_accuracy': single_accs,
        'compound_dynamic': serialize_results(dynamic_results),
        'compound_fixed': serialize_results(fixed_results),
        'dynamic_overall_acc': round(total_dyn / total_n_dyn, 4) if total_n_dyn > 0 else 0,
        'fixed_overall_acc': round(total_fix / total_n_fix, 4) if total_n_fix > 0 else 0,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase134_dual_firmware.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Dynamic routing overall: %.0f%%" % (
        total_dyn / total_n_dyn * 100 if total_n_dyn > 0 else 0))
    print("  Fixed routing overall:   %.0f%%" % (
        total_fix / total_n_fix * 100 if total_n_fix > 0 else 0))
    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
