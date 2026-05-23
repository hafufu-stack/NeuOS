# -*- coding: utf-8 -*-
"""
Phase 171: 7D Soul Algebra - Complete Algebraic Structure
Test group/ring operations in 7D soul space: identity, inverse,
addition, scalar multiplication, and composition table.
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


def evaluate_behavior(model, tok, soul_vec, test_data, device, layer=LAYER):
    """Evaluate what the model outputs for each test prompt."""
    preds = []
    for prompt, _ in test_data:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append(pred)
    return preds


def classify_behavior(preds, test_data):
    """Classify which task the predictions match."""
    # Build expected outputs for each task
    tasks = {}
    for prompt, _ in test_data:
        nums = [int(c) for c in prompt.replace(') =', '').split(',')]
        a, b = nums[0], nums[1]
        tasks['MIN'] = tasks.get('MIN', []) + [str(min(a, b))]
        tasks['MAX'] = tasks.get('MAX', []) + [str(max(a, b))]
        tasks['FIRST'] = tasks.get('FIRST', []) + [str(a)]
        tasks['SECOND'] = tasks.get('SECOND', []) + [str(b)]
        tasks['SUM'] = tasks.get('SUM', []) + [str(a + b)]
        tasks['DIFF'] = tasks.get('DIFF', []) + [str(abs(a - b))]

    best_task = 'UNKNOWN'
    best_acc = 0
    for task_name, expected in tasks.items():
        matches = sum(1 for p, e in zip(preds, expected) if p == e)
        acc = matches / len(preds)
        if acc > best_acc:
            best_acc = acc
            best_task = task_name
    return best_task, best_acc


def main():
    print("[P171] 7D Soul Algebra - Complete Structure")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Data
    train_data = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),
                 ("9, 3) =","3"),("7, 4) =","4"),("6, 1) =","1"),("2, 8) =","2"),
                 ("5, 9) =","5"),("1, 3) =","1")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","6"),
                 ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","8"),
                 ("5, 9) =","9"),("1, 3) =","3")],
        'FIRST': [("3, 7) =","3"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","4"),
                   ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","2"),
                   ("5, 9) =","5"),("1, 3) =","1")],
        'SECOND': [("3, 7) =","7"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","6"),
                    ("9, 3) =","3"),("7, 4) =","4"),("6, 1) =","1"),("2, 8) =","8"),
                    ("5, 9) =","9"),("1, 3) =","3")],
    }
    test_data = [("7, 2) =","?"),("6, 3) =","?"),("2, 9) =","?"),
                 ("1, 5) =","?"),("8, 4) =","?")]

    # Train base souls
    print("  Training 4 base souls...")
    souls_896 = {}
    for task_name, data in train_data.items():
        souls_896[task_name] = train_soul(model, tok, data, DEVICE, seed=42)
        print("    %s trained" % task_name)

    # Build PCA (use min of n_samples and 7)
    matrix = np.array([v.cpu().numpy() for v in souls_896.values()])
    N_COMP = min(matrix.shape[0], 7)
    pca = PCA(n_components=N_COMP)
    pca.fit(matrix)

    # Project to reduced space
    coords_7d = {}
    for name, vec in souls_896.items():
        coords_7d[name] = pca.transform(vec.cpu().numpy().reshape(1, -1))[0]
        print("    %dD coords: %s" % (N_COMP, np.round(coords_7d[name], 3)))

    # === ALGEBRAIC OPERATIONS ===
    results = {}

    # 1. Identity: zero vector
    print("\n  === Identity (zero vector) ===")
    zero_7d = np.zeros(N_COMP)
    zero_896 = pca.inverse_transform(zero_7d.reshape(1, -1))[0]
    zero_vec = torch.tensor(zero_896, dtype=torch.float32, device=DEVICE)
    preds = evaluate_behavior(model, tok, zero_vec, test_data, DEVICE)
    task, acc = classify_behavior(preds, test_data)
    results['identity'] = {'behavior': task, 'accuracy': round(acc, 4), 'preds': preds}
    print("    Zero vector -> %s (%.0f%%)" % (task, acc*100))

    # 2. Negation: -MIN, -MAX
    print("\n  === Negation ===")
    negation_results = {}
    for name in ['MIN', 'MAX', 'FIRST', 'SECOND']:
        neg_7d = -coords_7d[name]
        neg_896 = pca.inverse_transform(neg_7d.reshape(1, -1))[0]
        neg_vec = torch.tensor(neg_896, dtype=torch.float32, device=DEVICE)
        preds = evaluate_behavior(model, tok, neg_vec, test_data, DEVICE)
        task, acc = classify_behavior(preds, test_data)
        negation_results[name] = {'maps_to': task, 'accuracy': round(acc, 4)}
        print("    -%s -> %s (%.0f%%)" % (name, task, acc*100))
    results['negation'] = negation_results

    # 3. Addition: pairwise sums
    print("\n  === Addition (pairwise) ===")
    addition_results = {}
    task_names = list(coords_7d.keys())
    for i, name_a in enumerate(task_names):
        for name_b in task_names[i:]:
            sum_7d = (coords_7d[name_a] + coords_7d[name_b]) / 2  # normalized mean
            sum_896 = pca.inverse_transform(sum_7d.reshape(1, -1))[0]
            sum_vec = torch.tensor(sum_896, dtype=torch.float32, device=DEVICE)
            preds = evaluate_behavior(model, tok, sum_vec, test_data, DEVICE)
            task, acc = classify_behavior(preds, test_data)
            key = "%s+%s" % (name_a, name_b)
            addition_results[key] = {'maps_to': task, 'accuracy': round(acc, 4)}
            print("    (%s+%s)/2 -> %s (%.0f%%)" % (name_a, name_b, task, acc*100))
    results['addition'] = addition_results

    # 4. Scalar multiplication
    print("\n  === Scalar Multiplication ===")
    scalar_results = {}
    for scale in [0.5, 1.0, 1.5, 2.0, 3.0, -1.0, -2.0]:
        for name in ['MIN', 'MAX']:
            scaled_7d = coords_7d[name] * scale
            scaled_896 = pca.inverse_transform(scaled_7d.reshape(1, -1))[0]
            scaled_vec = torch.tensor(scaled_896, dtype=torch.float32, device=DEVICE)
            preds = evaluate_behavior(model, tok, scaled_vec, test_data, DEVICE)
            task, acc = classify_behavior(preds, test_data)
            key = "%.1f*%s" % (scale, name)
            scalar_results[key] = {'maps_to': task, 'accuracy': round(acc, 4)}
            print("    %s -> %s (%.0f%%)" % (key, task, acc*100))
    results['scalar'] = scalar_results

    # 5. Subtraction: MAX - MIN (attempting RANGE = max-min)
    print("\n  === Subtraction (RANGE attempt) ===")
    diff_7d = coords_7d['MAX'] - coords_7d['MIN']
    diff_896 = pca.inverse_transform(diff_7d.reshape(1, -1))[0]
    diff_vec = torch.tensor(diff_896, dtype=torch.float32, device=DEVICE)
    preds = evaluate_behavior(model, tok, diff_vec, test_data, DEVICE)
    task, acc = classify_behavior(preds, test_data)
    results['subtraction_range'] = {'maps_to': task, 'accuracy': round(acc, 4), 'preds': preds}
    print("    MAX - MIN -> %s (%.0f%%)" % (task, acc*100))

    # 6. Composition table: full 4x4 addition table
    print("\n  === Composition Table ===")
    comp_table = np.zeros((4, 4), dtype=object)
    comp_acc = np.zeros((4, 4))
    for i, na in enumerate(task_names):
        for j, nb in enumerate(task_names):
            mid_7d = (coords_7d[na] + coords_7d[nb]) / 2
            mid_896 = pca.inverse_transform(mid_7d.reshape(1, -1))[0]
            mid_vec = torch.tensor(mid_896, dtype=torch.float32, device=DEVICE)
            preds = evaluate_behavior(model, tok, mid_vec, test_data, DEVICE)
            task, acc = classify_behavior(preds, test_data)
            comp_table[i, j] = task
            comp_acc[i, j] = acc
    results['composition_table'] = {
        'tasks': task_names,
        'table': comp_table.tolist(),
        'accuracies': comp_acc.tolist()
    }

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Negation mapping
    ax = axes[0, 0]
    neg_names = list(negation_results.keys())
    neg_targets = [negation_results[n]['maps_to'] for n in neg_names]
    neg_accs = [negation_results[n]['accuracy'] for n in neg_names]
    colors = ['#E91E63', '#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(neg_names, neg_accs, color=colors, edgecolor='black', linewidth=1.5)
    for bar, target, val in zip(bars, neg_targets, neg_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '-%s -> %s\n%.0f%%' % (neg_names[bars.index(bar)], target, val*100),
                ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.3)
    ax.set_ylabel('Accuracy')
    ax.set_title('Negation: -X -> ?', fontweight='bold')

    # Panel 2: Scalar multiplication
    ax = axes[0, 1]
    scales_min = [(k, v) for k, v in scalar_results.items() if 'MIN' in k]
    scales_max = [(k, v) for k, v in scalar_results.items() if 'MAX' in k]
    x = [float(k.split('*')[0]) for k, v in scales_min]
    y_min = [v['accuracy'] for k, v in scales_min]
    y_max = [v['accuracy'] for k, v in scales_max]
    ax.plot(x, y_min, 'o-', color='#E91E63', label='MIN', linewidth=2, markersize=8)
    ax.plot(x, y_max, 's-', color='#2196F3', label='MAX', linewidth=2, markersize=8)
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Accuracy')
    ax.set_title('Scalar Multiplication: k * soul', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Composition table heatmap
    ax = axes[1, 0]
    im = ax.imshow(comp_acc, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(task_names, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(task_names, fontsize=9)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, '%s\n%.0f%%' % (comp_table[i, j], comp_acc[i, j]*100),
                    ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_title('Addition Table: (A+B)/2 -> ?', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Accuracy')

    # Panel 4: 7D coordinate comparison
    ax = axes[1, 1]
    x_pos = np.arange(N_COMP)
    width = 0.2
    for i, name in enumerate(task_names):
        ax.bar(x_pos + i*width, coords_7d[name], width, label=name,
               color=colors[i], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('PC Index')
    ax.set_ylabel('Coordinate Value')
    ax.set_title('%dD Coordinates of Base Souls' % N_COMP, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xticks(x_pos + 1.5*width)
    ax.set_xticklabels(['PC%d' % i for i in range(N_COMP)])
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Phase 171: 7D Soul Algebra - Complete Algebraic Structure',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase171_soul_algebra.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 171, 'name': 'soul_algebra_complete',
        'results': {k: v for k, v in results.items() if k != 'composition_table'},
        'composition_table': results['composition_table'],
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase171_soul_algebra.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n  P171 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
