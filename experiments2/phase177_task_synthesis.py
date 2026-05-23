# -*- coding: utf-8 -*-
"""
Phase 177: 7D Task Synthesis
Can we synthesize new tasks (MEAN, RANGE, CLAMP, ABS_DIFF)
by coordinate arithmetic in 7D soul space?
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


def evaluate_outputs(model, tok, soul_vec, prompts, device, layer=LAYER):
    """Get model outputs for a list of prompts."""
    preds = []
    for prompt in prompts:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append(pred)
    return preds


def classify_behavior(preds, test_pairs):
    """Classify what task the predictions match, including new tasks."""
    tasks = {}
    for (a, b), pred in zip(test_pairs, preds):
        for name, func in [
            ('MIN', lambda a, b: min(a, b)),
            ('MAX', lambda a, b: max(a, b)),
            ('FIRST', lambda a, b: a),
            ('SECOND', lambda a, b: b),
            ('SUM', lambda a, b: a + b),
            ('DIFF', lambda a, b: abs(a - b)),
            ('MEAN_FLOOR', lambda a, b: (a + b) // 2),
            ('RANGE', lambda a, b: max(a, b) - min(a, b)),
        ]:
            expected = str(func(a, b))
            if name not in tasks:
                tasks[name] = {'correct': 0, 'total': 0}
            tasks[name]['total'] += 1
            if pred == expected:
                tasks[name]['correct'] += 1

    best_task = 'UNKNOWN'
    best_acc = 0
    for name, data in tasks.items():
        acc = data['correct'] / data['total']
        if acc > best_acc:
            best_acc = acc
            best_task = name
    all_accs = {name: round(data['correct'] / data['total'], 4) for name, data in tasks.items()}
    return best_task, best_acc, all_accs


def main():
    print("[P177] 7D Task Synthesis")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train base souls
    train_sets = {
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

    print("  Training base souls...")
    souls = {}
    for name, data in train_sets.items():
        souls[name] = train_soul(model, tok, data, DEVICE, seed=42)

    # PCA
    matrix = np.array([v.cpu().numpy() for v in souls.values()])
    N_COMP = min(matrix.shape[0], 7)
    pca = PCA(n_components=N_COMP)
    pca.fit(matrix)

    coords = {}
    for name, vec in souls.items():
        coords[name] = pca.transform(vec.cpu().numpy().reshape(1, -1))[0]

    # Test pairs
    test_pairs = [(7, 2), (6, 3), (2, 9), (1, 5), (8, 4),
                  (3, 8), (4, 1), (9, 6), (5, 7), (2, 3)]
    test_prompts = ["%d, %d) =" % (a, b) for a, b in test_pairs]

    # === SYNTHESIS EXPERIMENTS ===
    synthesis_results = {}

    # 1. Coordinate arithmetic
    print("\n  === Coordinate Arithmetic Synthesis ===")
    syntheses = {
        'MEAN=(MIN+MAX)/2': (coords['MIN'] + coords['MAX']) / 2,
        'RANGE=MAX-MIN': coords['MAX'] - coords['MIN'],
        'SORT_ATTEMPT=MIN+MAX': coords['MIN'] + coords['MAX'],
        '(FIRST+SECOND)/2': (coords['FIRST'] + coords['SECOND']) / 2,
        'FIRST-SECOND': coords['FIRST'] - coords['SECOND'],
        '2*MIN-MAX': 2 * coords['MIN'] - coords['MAX'],
        '2*MAX-MIN': 2 * coords['MAX'] - coords['MIN'],
        '(MIN+FIRST)/2': (coords['MIN'] + coords['FIRST']) / 2,
        '(MAX+SECOND)/2': (coords['MAX'] + coords['SECOND']) / 2,
    }

    for name, synth_coords in syntheses.items():
        vec_896 = pca.inverse_transform(synth_coords.reshape(1, -1))[0]
        vec = torch.tensor(vec_896, dtype=torch.float32, device=DEVICE)
        preds = evaluate_outputs(model, tok, vec, test_prompts, DEVICE)
        best_task, best_acc, all_accs = classify_behavior(preds, test_pairs)
        synthesis_results[name] = {
            'best_match': best_task, 'best_accuracy': round(best_acc, 4),
            'all_accuracies': all_accs, 'preds': preds
        }
        print("    %s -> %s (%.0f%%)" % (name, best_task, best_acc*100))

    # 2. Random coordinate exploration
    print("\n  === Random Coordinate Exploration ===")
    rng = np.random.RandomState(42)
    random_results = []
    for i in range(30):
        rand_coords = rng.uniform(-3, 3, size=N_COMP)
        vec_896 = pca.inverse_transform(rand_coords.reshape(1, -1))[0]
        vec = torch.tensor(vec_896, dtype=torch.float32, device=DEVICE)
        preds = evaluate_outputs(model, tok, vec, test_prompts, DEVICE)
        best_task, best_acc, all_accs = classify_behavior(preds, test_pairs)
        random_results.append({
            'coords': rand_coords.tolist(),
            'best_match': best_task,
            'best_accuracy': round(best_acc, 4),
        })
    # Count discovered behaviors
    behavior_counts = {}
    for r in random_results:
        b = r['best_match']
        behavior_counts[b] = behavior_counts.get(b, 0) + 1
    print("    Discovered behaviors: %s" % behavior_counts)

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Synthesis results
    ax = axes[0, 0]
    synth_names = list(synthesis_results.keys())
    synth_tasks = [synthesis_results[n]['best_match'] for n in synth_names]
    synth_accs = [synthesis_results[n]['best_accuracy'] for n in synth_names]
    task_colors = {'MIN': '#E91E63', 'MAX': '#2196F3', 'FIRST': '#FF9800',
                   'SECOND': '#4CAF50', 'SUM': '#9C27B0', 'DIFF': '#795548',
                   'MEAN_FLOOR': '#00BCD4', 'RANGE': '#FF5722', 'UNKNOWN': '#9E9E9E'}
    bar_colors = [task_colors.get(t, '#9E9E9E') for t in synth_tasks]
    bars = ax.barh(range(len(synth_names)), synth_accs, color=bar_colors,
                   edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(synth_names)))
    ax.set_yticklabels(synth_names, fontsize=8)
    for i, (task, acc) in enumerate(zip(synth_tasks, synth_accs)):
        ax.text(acc + 0.02, i, '%s %.0f%%' % (task, acc*100), va='center',
                fontsize=9, fontweight='bold')
    ax.set_xlabel('Accuracy')
    ax.set_title('Coordinate Arithmetic Synthesis', fontweight='bold')
    ax.set_xlim(0, 1.3)

    # Panel 2: Random exploration distribution
    ax = axes[0, 1]
    behaviors = list(behavior_counts.keys())
    counts = [behavior_counts[b] for b in behaviors]
    bar_colors = [task_colors.get(b, '#9E9E9E') for b in behaviors]
    bars = ax.bar(behaviors, counts, color=bar_colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', fontweight='bold', fontsize=11)
    ax.set_ylabel('Count (out of 30)')
    ax.set_title('Random Exploration: Discovered Behaviors', fontweight='bold')

    # Panel 3: 7D coordinates of syntheses
    ax = axes[1, 0]
    for i, name in enumerate(list(syntheses.keys())[:5]):  # Top 5
        ax.plot(range(N_COMP), syntheses[name], 'o-', label=name, linewidth=1.5, markersize=5)
    ax.set_xlabel('PC Index')
    ax.set_ylabel('Coordinate Value')
    ax.set_title('%dD Coordinates of Synthesized Souls' % N_COMP, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    rows = [['Synthesis', 'Result', 'Acc']]
    for name in synth_names:
        r = synthesis_results[name]
        rows.append([name[:20], r['best_match'], '%.0f%%' % (r['best_accuracy']*100)])
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for j in range(3):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Synthesis Summary', fontweight='bold', pad=20)

    plt.suptitle('Phase 177: 7D Task Synthesis\n'
                 '"Can coordinate arithmetic create new computations?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase177_task_synthesis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 177, 'name': 'task_synthesis',
        'synthesis_results': {k: {kk: vv for kk, vv in v.items() if kk != 'preds'}
                              for k, v in synthesis_results.items()},
        'random_exploration': {'behavior_counts': behavior_counts,
                               'n_explored': len(random_results)},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase177_task_synthesis.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P177 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
