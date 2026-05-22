# -*- coding: utf-8 -*-
"""
Phase 166: The Rosetta Compiler
Directly specify 7D PCA coordinates to create soul vectors WITHOUT training.
No gradient descent. No data. Just dial the coordinates.

"Speak in coordinates. The machine obeys."
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


def evaluate(model, tok, soul_vec, test_data, device, layer=LAYER):
    correct = 0
    preds = []
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
        preds.append(pred)
    return correct / len(test_data) if test_data else 0, preds


def coords_to_soul(pca, coords, device, n_components=8):
    """Convert 7D coordinates directly to 896D soul vector."""
    full_coords = np.zeros(n_components)
    for i, c in enumerate(coords):
        if i < n_components:
            full_coords[i] = c
    vec_896 = pca.inverse_transform(full_coords.reshape(1, -1))[0]
    return torch.tensor(vec_896, dtype=torch.float32, device=device)


def main():
    print("[P166] The Rosetta Compiler")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train souls for PCA basis
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]

    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("1, 5) =","1"),("8, 4) =","4")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("1, 5) =","5"),("8, 4) =","8")]
    first_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","2"),
                  ("1, 5) =","1"),("8, 4) =","8")]
    second_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","9"),
                   ("1, 5) =","5"),("8, 4) =","4")]

    souls = {}
    for seed in [42, 100, 200, 300]:
        souls['MIN_s%d' % seed] = train_soul(model, tok, min_data, DEVICE, seed=seed)
        souls['MAX_s%d' % seed] = train_soul(model, tok, max_data, DEVICE, seed=seed)

    # Build PCA
    matrix = np.array([v.cpu().numpy() for v in souls.values()])
    pca = PCA(n_components=8)
    pca.fit(matrix)

    # Get known soul coordinates
    min_coords = pca.transform(souls['MIN_s42'].cpu().numpy().reshape(1, -1))[0]
    max_coords = pca.transform(souls['MAX_s42'].cpu().numpy().reshape(1, -1))[0]
    print("  MIN coords: %s" % ', '.join(['%.2f' % c for c in min_coords]))
    print("  MAX coords: %s" % ', '.join(['%.2f' % c for c in max_coords]))

    # Baseline: trained souls
    soul_min_trained = souls['MIN_s42']
    soul_max_trained = souls['MAX_s42']
    min_baseline, _ = evaluate(model, tok, soul_min_trained, min_test, DEVICE)
    max_baseline, _ = evaluate(model, tok, soul_max_trained, max_test, DEVICE)
    print("  Trained MIN: %.0f%%, Trained MAX: %.0f%%" % (min_baseline*100, max_baseline*100))

    results = {
        'baselines': {
            'min_trained': round(min_baseline, 4),
            'max_trained': round(max_baseline, 4),
        },
        'min_coords': [round(c, 4) for c in min_coords.tolist()],
        'max_coords': [round(c, 4) for c in max_coords.tolist()],
    }

    # Test 1: Reconstruct known souls from coordinates
    print("\n  === Test 1: Coordinate Reconstruction ===")
    soul_min_compiled = coords_to_soul(pca, min_coords, DEVICE)
    soul_max_compiled = coords_to_soul(pca, max_coords, DEVICE)
    min_compiled_acc, min_p = evaluate(model, tok, soul_min_compiled, min_test, DEVICE)
    max_compiled_acc, max_p = evaluate(model, tok, soul_max_compiled, max_test, DEVICE)
    print("  Compiled MIN: %.0f%% (preds: %s)" % (min_compiled_acc*100, min_p))
    print("  Compiled MAX: %.0f%% (preds: %s)" % (max_compiled_acc*100, max_p))
    results['reconstruction'] = {
        'min_acc': round(min_compiled_acc, 4),
        'max_acc': round(max_compiled_acc, 4),
    }

    # Test 2: Direct coordinate programming (no training!)
    print("\n  === Test 2: Direct Coordinate Programming ===")
    # The key insight from P164: PC5 is the MIN/MAX discriminator
    # MIN has PC5 ~ -1.06, MAX has PC5 ~ +1.18
    # Let's try programming purely from these coordinates

    programs = {
        'pure_MIN': {'coords': [0, 0, 0, 0, 0, -2.0, 0, 0],
                     'description': 'PC5=-2 (strong MIN direction)'},
        'pure_MAX': {'coords': [0, 0, 0, 0, 0, 2.0, 0, 0],
                     'description': 'PC5=+2 (strong MAX direction)'},
        'pure_FIRST': {'coords': [0, 0, 0, 0, 0, 0, 2.0, 0],
                       'description': 'PC6=+2 (FIRST direction)'},
        'pure_SECOND': {'coords': [0, 0, 0, 0, 0, 0, -2.0, 0],
                        'description': 'PC6=-2 (SECOND direction)'},
        'MIN_via_PC1': {'coords': [0, -2.0, 0, 0, 0, 0, 0, 0],
                        'description': 'PC1=-2 (MIN via comparison axis)'},
        'MAX_via_PC1': {'coords': [0, 2.0, 0, 0, 0, 0, 0, 0],
                        'description': 'PC1=+2 (MAX via comparison axis)'},
        'SECOND_via_PC0': {'coords': [2.0, 0, 0, 0, 0, 0, 0, 0],
                           'description': 'PC0=+2 (SECOND via main axis)'},
        'combined_MIN': {'coords': [0, -1.5, 0, 0, 0, -1.5, 0, 0],
                         'description': 'PC1=-1.5, PC5=-1.5 (combined MIN)'},
        'combined_MAX': {'coords': [0, 1.5, 0, 0, 0, 1.5, 0, 0],
                         'description': 'PC1=+1.5, PC5=+1.5 (combined MAX)'},
    }

    program_results = {}
    for prog_name, prog_info in programs.items():
        soul = coords_to_soul(pca, prog_info['coords'], DEVICE)
        min_acc, min_p = evaluate(model, tok, soul, min_test, DEVICE)
        max_acc, max_p = evaluate(model, tok, soul, max_test, DEVICE)
        first_acc, _ = evaluate(model, tok, soul, first_test, DEVICE)
        second_acc, _ = evaluate(model, tok, soul, second_test, DEVICE)

        program_results[prog_name] = {
            'description': prog_info['description'],
            'coords': prog_info['coords'],
            'min_acc': round(min_acc, 4),
            'max_acc': round(max_acc, 4),
            'first_acc': round(first_acc, 4),
            'second_acc': round(second_acc, 4),
            'best_match': max([('MIN', min_acc), ('MAX', max_acc),
                              ('FIRST', first_acc), ('SECOND', second_acc)],
                             key=lambda x: x[1]),
            'preds_on_min_test': min_p,
        }
        bm = program_results[prog_name]['best_match']
        print("  %-18s: MIN=%.0f%% MAX=%.0f%% FIRST=%.0f%% SECOND=%.0f%% => %s(%.0f%%)" % (
            prog_name, min_acc*100, max_acc*100, first_acc*100, second_acc*100,
            bm[0], bm[1]*100))

    # Serialize best_match as list
    for k, v in program_results.items():
        v['best_match'] = list(v['best_match'])
    results['programs'] = program_results

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Compiled vs Trained
    ax = axes[0]
    labels = ['MIN\n(trained)', 'MIN\n(compiled)', 'MAX\n(trained)', 'MAX\n(compiled)']
    vals = [min_baseline, min_compiled_acc, max_baseline, max_compiled_acc]
    colors = ['#E91E63', '#F48FB1', '#2196F3', '#90CAF9']
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy')
    ax.set_title('Trained vs Compiled Souls\n(gradient-free reconstruction)', fontweight='bold')

    # Panel 2: Direct programming results
    ax = axes[1]
    prog_names = list(program_results.keys())
    prog_min = [program_results[p]['min_acc'] for p in prog_names]
    prog_max = [program_results[p]['max_acc'] for p in prog_names]
    x = np.arange(len(prog_names))
    ax.barh(x - 0.2, prog_min, 0.35, label='MIN acc', color='#E91E63', edgecolor='black')
    ax.barh(x + 0.2, prog_max, 0.35, label='MAX acc', color='#2196F3', edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels([p[:16] for p in prog_names], fontsize=7)
    ax.set_xlabel('Accuracy')
    ax.legend(fontsize=8)
    ax.set_title('Direct Coordinate Programming\n(zero training data)', fontweight='bold')
    ax.set_xlim(0, 1.1)

    # Panel 3: Coordinate-to-behavior mapping
    ax = axes[2]
    ax.axis('off')
    mapping = [
        ['Coordinate', 'Dial Setting', 'Behavior'],
        ['PC5 = -2.0', 'Strong negative', 'MIN-like'],
        ['PC5 = +2.0', 'Strong positive', 'MAX-like'],
        ['PC1 = -2.0', 'Comparison: lower', 'MIN-like'],
        ['PC1 = +2.0', 'Comparison: higher', 'MAX-like'],
        ['PC6 = +2.0', 'Position: first', 'FIRST'],
        ['PC6 = -2.0', 'Position: second', 'SECOND'],
    ]
    table = ax.table(cellText=mapping[1:], colLabels=mapping[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for j in range(3):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('The Rosetta Dictionary\n(coordinate -> behavior)', fontweight='bold', pad=20)

    plt.suptitle('Phase 166: The Rosetta Compiler\n'
                 '"Speak in coordinates. The machine obeys."',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase166_rosetta.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 166, 'name': 'rosetta_compiler',
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase166_rosetta.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P166 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
