# -*- coding: utf-8 -*-
"""
Phase 164: Universal Instruction Set
Decode what each PCA basis vector means as a "program instruction."

The 7 PCA components are the "assembly opcodes" of NeuOS.
What does each one DO when activated alone?

"Reverse-engineer the instruction set of the soul."
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


def probe_basis(model, tok, soul_vec, device, layer=LAYER):
    """Probe a soul vector on diverse prompts to understand its behavior."""
    prompts = [
        ("3, 7) =", {'min': '3', 'max': '7', 'first': '3', 'second': '7', 'add': '10', 'sub': '-4'}),
        ("5, 2) =", {'min': '2', 'max': '5', 'first': '5', 'second': '2', 'add': '7', 'sub': '3'}),
        ("8, 1) =", {'min': '1', 'max': '8', 'first': '8', 'second': '1', 'add': '9', 'sub': '7'}),
        ("4, 6) =", {'min': '4', 'max': '6', 'first': '4', 'second': '6', 'add': '10', 'sub': '-2'}),
        ("9, 3) =", {'min': '3', 'max': '9', 'first': '9', 'second': '3', 'add': '12', 'sub': '6'}),
        ("7, 4) =", {'min': '4', 'max': '7', 'first': '7', 'second': '4', 'add': '11', 'sub': '3'}),
        ("2, 8) =", {'min': '2', 'max': '8', 'first': '2', 'second': '8', 'add': '10', 'sub': '-6'}),
        ("6, 1) =", {'min': '1', 'max': '6', 'first': '6', 'second': '1', 'add': '7', 'sub': '5'}),
        ("1, 9) =", {'min': '1', 'max': '9', 'first': '1', 'second': '9', 'add': '10', 'sub': '-8'}),
        ("5, 5) =", {'min': '5', 'max': '5', 'first': '5', 'second': '5', 'add': '10', 'sub': '0'}),
    ]

    preds = []
    entropies = []
    for prompt, expected in prompts:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        pred = tok.decode(logits.argmax().item()).strip()
        preds.append(pred)
        entropies.append(entropy)

    # Score against each known operation
    op_scores = {}
    for op in ['min', 'max', 'first', 'second']:
        expected_list = [p[1][op] for p in prompts]
        match = sum(1 for p, e in zip(preds, expected_list) if p == e)
        op_scores[op] = match / len(preds)

    # Check for constant output
    unique_preds = set(preds)
    is_constant = len(unique_preds) == 1
    constant_val = list(unique_preds)[0] if is_constant else None

    avg_entropy = np.mean(entropies)

    return {
        'predictions': preds,
        'entropies': [round(e, 4) for e in entropies],
        'avg_entropy': round(avg_entropy, 4),
        'op_scores': {k: round(v, 4) for k, v in op_scores.items()},
        'is_constant': is_constant,
        'constant_val': constant_val,
        'best_op': max(op_scores, key=op_scores.get),
        'best_op_score': round(max(op_scores.values()), 4),
    }


def main():
    print("[P164] Universal Instruction Set")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train diverse souls
    print("  Training diverse soul vectors...")
    datasets = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")],
    }

    souls = {}
    for name, data in datasets.items():
        for seed in [42, 100, 200, 300]:
            key = '%s_s%d' % (name, seed)
            souls[key] = train_soul(model, tok, data, DEVICE, seed=seed)

    # Fit PCA
    soul_matrix = np.array([v.cpu().numpy() for v in souls.values()])
    n_pca = min(len(souls), 8)
    pca = PCA(n_components=n_pca)
    pca.fit(soul_matrix)

    explained_var = pca.explained_variance_ratio_
    print("  PCA explained variance: %s" % ', '.join(
        ['%.4f' % v for v in explained_var]))

    # Decode each basis vector
    print("\n  === DECODING PCA BASIS VECTORS ===")
    basis_results = {}

    for comp_idx in range(n_pca):
        print("\n  --- Component %d (%.1f%% variance) ---" % (
            comp_idx, explained_var[comp_idx] * 100))

        comp_results = {
            'variance_ratio': round(explained_var[comp_idx], 6),
            'scales': {},
        }

        for scale in [-3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0]:
            coords = np.zeros(n_pca)
            coords[comp_idx] = scale
            vec_896 = pca.inverse_transform(coords.reshape(1, -1))[0]
            soul_vec = torch.tensor(vec_896, dtype=torch.float32, device=DEVICE)

            result = probe_basis(model, tok, soul_vec, DEVICE)
            comp_results['scales'][str(scale)] = result

            if result['is_constant']:
                desc = "CONST(%s)" % result['constant_val']
            else:
                desc = "%s(%.0f%%)" % (result['best_op'].upper(), result['best_op_score'] * 100)
            print("    scale=%+.1f: %s  H=%.2f  preds=%s" % (
                scale, desc, result['avg_entropy'],
                ','.join(result['predictions'][:5])))

        # Determine what this component "does"
        pos_result = comp_results['scales']['2.0']
        neg_result = comp_results['scales']['-2.0']
        zero_result = comp_results['scales']['0']

        interpretation = []
        if pos_result['best_op'] != neg_result['best_op']:
            interpretation.append('+%s / -%s axis' % (
                pos_result['best_op'].upper(), neg_result['best_op'].upper()))
        if zero_result['is_constant']:
            interpretation.append('zero=const(%s)' % zero_result['constant_val'])

        comp_results['interpretation'] = ' | '.join(interpretation) if interpretation else 'unclear'
        basis_results[comp_idx] = comp_results
        print("  Interpretation: %s" % comp_results['interpretation'])

    # Project known souls onto PCA axes to understand their "programs"
    print("\n  === KNOWN SOUL DECOMPOSITIONS ===")
    soul_decompositions = {}
    for name, vec in souls.items():
        coords = pca.transform(vec.cpu().numpy().reshape(1, -1))[0]
        soul_decompositions[name] = {
            'coords': [round(c, 4) for c in coords.tolist()],
            'dominant_component': int(np.argmax(np.abs(coords))),
            'dominant_value': round(coords[np.argmax(np.abs(coords))], 4),
        }
        if 's42' in name:
            print("  %s: dominant=PC%d (%.2f), coords=%s" % (
                name, soul_decompositions[name]['dominant_component'],
                soul_decompositions[name]['dominant_value'],
                ', '.join(['%.2f' % c for c in coords[:5]])))

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: What each component does
    ax = axes[0]
    comp_labels = []
    pos_ops = []
    neg_ops = []
    for ci in range(n_pca):
        r = basis_results[ci]
        pos_r = r['scales']['2.0']
        neg_r = r['scales']['-2.0']
        comp_labels.append('PC%d\n(%.0f%%)' % (ci, r['variance_ratio'] * 100))
        pos_ops.append(pos_r['best_op_score'])
        neg_ops.append(neg_r['best_op_score'])

    x = np.arange(len(comp_labels))
    ax.bar(x - 0.2, pos_ops, 0.35, label='+scale', color='#4CAF50', edgecolor='black')
    ax.bar(x + 0.2, neg_ops, 0.35, label='-scale', color='#F44336', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=8)
    ax.set_ylabel('Best Operation Match')
    ax.set_title('What Each PCA Component Does', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add operation labels
    for ci in range(n_pca):
        r = basis_results[ci]
        pos_r = r['scales']['2.0']
        neg_r = r['scales']['-2.0']
        ax.text(ci - 0.2, pos_ops[ci] + 0.02, pos_r['best_op'][:3].upper(),
                ha='center', fontsize=7, fontweight='bold')
        ax.text(ci + 0.2, neg_ops[ci] + 0.02, neg_r['best_op'][:3].upper(),
                ha='center', fontsize=7, fontweight='bold')

    # Panel 2: Soul decomposition (bar chart of PCA coords for MIN and MAX)
    ax = axes[1]
    min_coords = soul_decompositions['MIN_s42']['coords']
    max_coords = soul_decompositions['MAX_s42']['coords']
    x = np.arange(n_pca)
    ax.bar(x - 0.2, min_coords, 0.35, label='MIN', color='#E91E63', edgecolor='black')
    ax.bar(x + 0.2, max_coords, 0.35, label='MAX', color='#2196F3', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['PC%d' % i for i in range(n_pca)])
    ax.set_ylabel('Coordinate Value')
    ax.set_title('Soul Decomposition\n(how MIN & MAX differ)', fontweight='bold')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 3: Explained variance
    ax = axes[2]
    cumvar = np.cumsum(explained_var)
    ax.bar(range(1, n_pca + 1), explained_var, color='#9C27B0',
           edgecolor='black', alpha=0.7, label='Individual')
    ax.plot(range(1, n_pca + 1), cumvar, 'ro-', linewidth=2, label='Cumulative')
    ax.set_xlabel('PCA Component')
    ax.set_ylabel('Variance Ratio')
    ax.set_title('Instruction Set Size\n(how many opcodes exist?)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    plt.suptitle('Phase 164: Universal Instruction Set\n'
                 '"The assembly language of the soul"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase164_instruction_set.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Build instruction set summary
    instruction_set = []
    for ci in range(n_pca):
        r = basis_results[ci]
        instruction_set.append({
            'opcode': 'PC%d' % ci,
            'variance': float(round(r['variance_ratio'], 4)),
            'interpretation': r['interpretation'],
            'positive_behavior': r['scales']['2.0']['best_op'],
            'negative_behavior': r['scales']['-2.0']['best_op'],
        })

    output = {
        'phase': 164, 'name': 'universal_instruction_set',
        'n_components': n_pca,
        'instruction_set': instruction_set,
        'basis_results': {str(k): {
            'variance_ratio': float(v['variance_ratio']),
            'interpretation': v['interpretation'],
        } for k, v in basis_results.items()},
        'soul_decompositions': {k: {
            'coords': [float(c) for c in v['coords']],
            'dominant_component': int(v['dominant_component']),
            'dominant_value': float(v['dominant_value']),
        } for k, v in soul_decompositions.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase164_instruction_set.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
