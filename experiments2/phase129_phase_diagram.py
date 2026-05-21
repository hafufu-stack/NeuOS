# -*- coding: utf-8 -*-
"""
Phase 129: Soul Phase Diagram
2D interpolation map between MIN, MAX, ADD, SUB souls.
Maps the topology of computation in soul vector space.

"What lies between minimum and maximum? Between addition and subtraction?"
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8  # Use L8 (P128 finding: optimal layer)


def train_soul(model, tok, data, device, layer=LAYER, seed=42, epochs=150):
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


def classify_output(model, tok, vec, prompts, device, layer=LAYER):
    """Run soul on prompts and return predicted outputs."""
    preds = []
    for p, _ in prompts:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        logits = out.logits[0, -1, :]
        pred = tok.decode(logits.argmax().item()).strip()
        preds.append(pred)
    return preds


def score_behavior(preds, task_answers):
    """Score how well predictions match each task's expected answers."""
    scores = {}
    for task_name, expected in task_answers.items():
        correct = sum(1 for p, e in zip(preds, expected) if p == e)
        scores[task_name] = correct / len(expected)
    return scores


def main():
    print("[P129] Soul Phase Diagram")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Training data
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

    # Test prompts (shared across all)
    test_prompts = [("3, 7) =",""), ("5, 2) =",""), ("8, 1) =",""),
                    ("4, 6) =",""), ("9, 3) =","")]
    task_answers = {
        'MIN': ["3", "2", "1", "4", "3"],
        'MAX': ["7", "5", "8", "6", "9"],
    }

    # Train 4 corner souls
    print("  Training 4 corner souls (L8)...")
    souls = {}
    for task_name, data in task_data.items():
        souls[task_name] = train_soul(model, tok, data, DEVICE, seed=42)
        preds = classify_output(model, tok, souls[task_name], test_prompts, DEVICE)
        print("    %s trained. Outputs: %s" % (task_name, preds))

    # Build 2D phase diagram
    # Axis 1: MIN <-> MAX (horizontal, t)
    # Axis 2: ADD <-> SUB (vertical, s)
    # Interpolated soul = (1-t)(1-s)*MIN + t(1-s)*MAX + (1-t)s*ADD + ts*SUB
    resolution = 21  # 21x21 grid
    ts = np.linspace(0, 1, resolution)
    ss = np.linspace(0, 1, resolution)

    # For each grid point, determine dominant behavior
    print("  Scanning %dx%d phase diagram..." % (resolution, resolution))
    phase_map = np.zeros((resolution, resolution))  # dominant task index
    min_score_map = np.zeros((resolution, resolution))
    max_score_map = np.zeros((resolution, resolution))
    entropy_map = np.zeros((resolution, resolution))
    output_map = [[None]*resolution for _ in range(resolution)]

    for si, s in enumerate(ss):
        for ti, t in enumerate(ts):
            # Bilinear interpolation
            interp = ((1-t)*(1-s) * souls['MIN'] +
                      t*(1-s) * souls['MAX'] +
                      (1-t)*s * souls['ADD'] +
                      t*s * souls['SUB'])

            preds = classify_output(model, tok, interp, test_prompts, DEVICE)
            scores = score_behavior(preds, task_answers)

            min_score_map[si, ti] = scores['MIN']
            max_score_map[si, ti] = scores['MAX']

            # Determine dominant behavior
            if scores['MIN'] > scores['MAX']:
                phase_map[si, ti] = 0  # MIN dominant
            elif scores['MAX'] > scores['MIN']:
                phase_map[si, ti] = 1  # MAX dominant
            else:
                phase_map[si, ti] = 0.5  # Tied/uncertain

            # Output entropy
            from collections import Counter
            counts = Counter(preds)
            total = len(preds)
            probs = np.array([counts[k]/total for k in counts])
            entropy_map[si, ti] = float(-np.sum(probs * np.log2(probs + 1e-10)))

            output_map[si][ti] = preds

        if si % 5 == 0:
            print("    Row %d/%d done" % (si+1, resolution))

    print("  Phase diagram complete!")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: MIN/MAX dominance map
    ax = axes[0]
    cmap_phase = LinearSegmentedColormap.from_list('minmax', ['#2196F3', '#FFFFFF', '#FF5722'])
    im = ax.imshow(phase_map, origin='lower', extent=[0,1,0,1],
                   cmap=cmap_phase, vmin=0, vmax=1, aspect='equal',
                   interpolation='bilinear')
    ax.set_xlabel('MIN <--------> MAX', fontsize=11)
    ax.set_ylabel('Pure <--------> ADD/SUB', fontsize=11)
    ax.set_title('Phase Diagram: Dominant Behavior', fontweight='bold')
    ax.text(0.05, 0.05, 'MIN', fontsize=14, fontweight='bold', color='#1565C0',
            transform=ax.transAxes)
    ax.text(0.85, 0.05, 'MAX', fontsize=14, fontweight='bold', color='#D84315',
            transform=ax.transAxes)
    ax.text(0.05, 0.90, 'ADD', fontsize=14, fontweight='bold', color='#2E7D32',
            transform=ax.transAxes)
    ax.text(0.85, 0.90, 'SUB', fontsize=14, fontweight='bold', color='#6A1B9A',
            transform=ax.transAxes)
    plt.colorbar(im, ax=ax, shrink=0.8, label='MIN(0) <-> MAX(1)')

    # Panel 2: MIN accuracy landscape
    ax = axes[1]
    im = ax.imshow(min_score_map, origin='lower', extent=[0,1,0,1],
                   cmap='Blues', vmin=0, vmax=1, aspect='equal',
                   interpolation='bilinear')
    ax.set_xlabel('MIN <--------> MAX', fontsize=11)
    ax.set_ylabel('Pure <--------> ADD/SUB', fontsize=11)
    ax.set_title('MIN Accuracy Landscape', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Accuracy')
    # Add contour
    ax.contour(ts, ss, min_score_map, levels=[0.5], colors='white',
               linewidths=2, linestyles='dashed')

    # Panel 3: Entropy map (uncertainty/chaos regions)
    ax = axes[2]
    im = ax.imshow(entropy_map, origin='lower', extent=[0,1,0,1],
                   cmap='inferno', aspect='equal', interpolation='bilinear')
    ax.set_xlabel('MIN <--------> MAX', fontsize=11)
    ax.set_ylabel('Pure <--------> ADD/SUB', fontsize=11)
    ax.set_title('Output Entropy (Chaos Map)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Entropy (bits)')
    # Mark maximum entropy point
    max_ent_idx = np.unravel_index(entropy_map.argmax(), entropy_map.shape)
    ax.plot(ts[max_ent_idx[1]], ss[max_ent_idx[0]], 'w*', markersize=15,
            markeredgecolor='black', markeredgewidth=1)
    ax.annotate('Max chaos', xy=(ts[max_ent_idx[1]], ss[max_ent_idx[0]]),
                xytext=(ts[max_ent_idx[1]]+0.1, ss[max_ent_idx[0]]+0.1),
                color='white', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white'))

    plt.suptitle('Phase 129: Soul Phase Diagram\n'
                 '"The geography of computation in soul space"',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase129_phase_diagram.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    output = {
        'phase': 129,
        'name': 'soul_phase_diagram',
        'resolution': resolution,
        'layer': LAYER,
        'corners': list(task_data.keys()),
        'min_score_map': min_score_map.tolist(),
        'max_score_map': max_score_map.tolist(),
        'entropy_map': entropy_map.tolist(),
        'phase_map': phase_map.tolist(),
        'max_entropy': float(entropy_map.max()),
        'max_entropy_location': [float(ss[max_ent_idx[0]]), float(ts[max_ent_idx[1]])],
        'phase_boundary_fraction': float(np.mean(np.abs(phase_map - 0.5) < 0.3)),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase129_phase_diagram.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Max entropy: %.3f at (t=%.2f, s=%.2f)" % (
        entropy_map.max(), ts[max_ent_idx[1]], ss[max_ent_idx[0]]))
    print("  Phase boundary fraction: %.1f%%" % (
        np.mean(np.abs(phase_map - 0.5) < 0.3) * 100))
    print("  Completed in %.0fs" % (time.time() - start))

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
