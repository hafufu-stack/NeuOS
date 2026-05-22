# -*- coding: utf-8 -*-
"""
Phase 168: The Control Room UI
A terminal-based interactive soul synthesizer dashboard.
Display the 7 PCA sliders and show real-time predictions.

Uses matplotlib interactive mode for a visual dashboard.

"The control panel of the mind."
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


def coords_to_soul(pca, coords, device, n_components=8):
    full = np.zeros(n_components)
    for i, c in enumerate(coords):
        if i < n_components:
            full[i] = c
    v = pca.inverse_transform(full.reshape(1, -1))[0]
    return torch.tensor(v, dtype=torch.float32, device=device)


def run_inference(model, tok, soul_vec, prompt, device, layer=LAYER):
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
    conf = probs.max().item()
    # Top 3 predictions
    top3_vals, top3_idx = probs.topk(3)
    top3 = [(tok.decode(idx.item()).strip(), round(val.item(), 4))
            for val, idx in zip(top3_vals, top3_idx)]
    return pred, round(entropy, 4), round(conf, 4), top3


def main():
    print("[P168] The Control Room UI")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train base souls for PCA
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]

    souls = {}
    for seed in [42, 100, 200, 300]:
        souls['MIN_s%d' % seed] = train_soul(model, tok, min_data, DEVICE, seed=seed)
        souls['MAX_s%d' % seed] = train_soul(model, tok, max_data, DEVICE, seed=seed)

    # Build PCA
    matrix = np.array([v.cpu().numpy() for v in souls.values()])
    pca = PCA(n_components=8)
    pca.fit(matrix)

    # Get known soul coordinates for reference
    min_coords = pca.transform(souls['MIN_s42'].cpu().numpy().reshape(1, -1))[0]
    max_coords = pca.transform(souls['MAX_s42'].cpu().numpy().reshape(1, -1))[0]

    # Component meanings from P164
    pc_meanings = [
        'SECOND <-> MAX',   # PC0
        'MIN <-> MAX',      # PC1
        'unclear',          # PC2
        'MIN <-> FIRST',    # PC3
        'FIRST (both)',     # PC4
        'SECOND <-> FIRST', # PC5 (KEY: MIN/MAX discriminator)
        'SECOND <-> FIRST', # PC6
        'FIRST <-> SECOND', # PC7
    ]

    # Generate a comprehensive dashboard image
    test_prompts = [
        ("3, 7) =", {'min': '3', 'max': '7', 'first': '3', 'second': '7'}),
        ("5, 2) =", {'min': '2', 'max': '5', 'first': '5', 'second': '2'}),
        ("8, 1) =", {'min': '1', 'max': '8', 'first': '8', 'second': '1'}),
        ("4, 6) =", {'min': '4', 'max': '6', 'first': '4', 'second': '6'}),
        ("9, 3) =", {'min': '3', 'max': '9', 'first': '9', 'second': '3'}),
    ]

    # Generate dashboard for several preset configurations
    presets = {
        'MIN (trained)': min_coords.tolist(),
        'MAX (trained)': max_coords.tolist(),
        'Zero (origin)': [0]*8,
        'Pure PC5-': [0, 0, 0, 0, 0, -2.5, 0, 0],
        'Pure PC5+': [0, 0, 0, 0, 0, 2.5, 0, 0],
        'Pure PC1+': [0, 2.5, 0, 0, 0, 0, 0, 0],
        'Combined MIN': [0, -1.5, 0, 0, 0, -1.5, 0, 0],
        'Combined MAX': [0, 1.5, 0, 0, 0, 1.5, 0, 0],
    }

    dashboard_data = {}
    print("\n  === CONTROL ROOM DASHBOARD ===")
    for preset_name, coords in presets.items():
        soul = coords_to_soul(pca, coords, DEVICE)
        preset_results = []
        for prompt, truth in test_prompts:
            pred, entropy, conf, top3 = run_inference(model, tok, soul, prompt, DEVICE)
            preset_results.append({
                'prompt': prompt, 'pred': pred,
                'entropy': entropy, 'conf': conf, 'top3': top3,
            })

        # Classify behavior
        preds = [r['pred'] for r in preset_results]
        min_match = sum(1 for r, (_, t) in zip(preset_results, test_prompts)
                       if r['pred'] == t['min']) / len(test_prompts)
        max_match = sum(1 for r, (_, t) in zip(preset_results, test_prompts)
                       if r['pred'] == t['max']) / len(test_prompts)
        first_match = sum(1 for r, (_, t) in zip(preset_results, test_prompts)
                        if r['pred'] == t['first']) / len(test_prompts)
        second_match = sum(1 for r, (_, t) in zip(preset_results, test_prompts)
                         if r['pred'] == t['second']) / len(test_prompts)

        avg_entropy = np.mean([r['entropy'] for r in preset_results])
        avg_conf = np.mean([r['conf'] for r in preset_results])

        dashboard_data[preset_name] = {
            'coords': [round(c, 3) for c in coords],
            'results': preset_results,
            'scores': {
                'MIN': round(min_match, 4), 'MAX': round(max_match, 4),
                'FIRST': round(first_match, 4), 'SECOND': round(second_match, 4),
            },
            'avg_entropy': round(avg_entropy, 4),
            'avg_conf': round(avg_conf, 4),
        }

        best_op = max(dashboard_data[preset_name]['scores'].items(), key=lambda x: x[1])
        print("  %-18s: %s(%.0f%%) H=%.2f conf=%.2f preds=%s" % (
            preset_name, best_op[0], best_op[1]*100, avg_entropy, avg_conf,
            ','.join(preds)))

    # ---- MEGA DASHBOARD PLOT ----
    fig = plt.figure(figsize=(20, 14))

    # Top: Slider visualization for all presets
    ax_sliders = fig.add_axes([0.05, 0.55, 0.9, 0.35])
    preset_names = list(presets.keys())
    n_presets = len(preset_names)
    n_pcs = 7  # Show 7 components

    im_data = np.zeros((n_presets, n_pcs))
    for pi, pname in enumerate(preset_names):
        coords = presets[pname]
        for ci in range(n_pcs):
            im_data[pi, ci] = coords[ci]

    im = ax_sliders.imshow(im_data, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    ax_sliders.set_xticks(range(n_pcs))
    ax_sliders.set_xticklabels(['PC%d\n%s' % (i, pc_meanings[i][:12]) for i in range(n_pcs)],
                                fontsize=8)
    ax_sliders.set_yticks(range(n_presets))
    ax_sliders.set_yticklabels(preset_names, fontsize=9)
    ax_sliders.set_title('Control Room: 7D Soul Synthesizer\n'
                         'Blue = negative, Red = positive',
                         fontweight='bold', fontsize=14)
    plt.colorbar(im, ax=ax_sliders, shrink=0.8, label='Coordinate Value')

    # Add text annotations
    for pi in range(n_presets):
        for ci in range(n_pcs):
            val = im_data[pi, ci]
            if abs(val) > 0.1:
                ax_sliders.text(ci, pi, '%.1f' % val, ha='center', va='center',
                               fontsize=8, fontweight='bold',
                               color='white' if abs(val) > 1.5 else 'black')

    # Bottom left: Behavior radar for each preset
    ax_results = fig.add_axes([0.05, 0.05, 0.45, 0.42])
    ops = ['MIN', 'MAX', 'FIRST', 'SECOND']
    x = np.arange(len(preset_names))
    width = 0.18
    op_colors = {'MIN': '#E91E63', 'MAX': '#2196F3', 'FIRST': '#4CAF50', 'SECOND': '#FF9800'}
    for oi, op in enumerate(ops):
        vals = [dashboard_data[pn]['scores'][op] for pn in preset_names]
        ax_results.bar(x + oi * width, vals, width, label=op,
                      color=op_colors[op], edgecolor='black')
    ax_results.set_xticks(x + width * 1.5)
    ax_results.set_xticklabels([n[:12] for n in preset_names], fontsize=7, rotation=30)
    ax_results.set_ylabel('Match Score')
    ax_results.legend(fontsize=8, ncol=4)
    ax_results.set_title('Operation Match per Preset', fontweight='bold')
    ax_results.set_ylim(0, 1.1)

    # Bottom right: Entropy & confidence
    ax_meta = fig.add_axes([0.55, 0.05, 0.4, 0.42])
    entropies = [dashboard_data[pn]['avg_entropy'] for pn in preset_names]
    confs = [dashboard_data[pn]['avg_conf'] for pn in preset_names]
    x = np.arange(len(preset_names))
    ax_meta.bar(x - 0.2, entropies, 0.35, label='Avg Entropy', color='#9C27B0',
               edgecolor='black', alpha=0.8)
    ax_meta2 = ax_meta.twinx()
    ax_meta2.bar(x + 0.2, confs, 0.35, label='Avg Confidence', color='#FF9800',
                edgecolor='black', alpha=0.8)
    ax_meta.set_xticks(x)
    ax_meta.set_xticklabels([n[:12] for n in preset_names], fontsize=7, rotation=30)
    ax_meta.set_ylabel('Entropy', color='#9C27B0')
    ax_meta2.set_ylabel('Confidence', color='#FF9800')
    ax_meta.legend(loc='upper left', fontsize=8)
    ax_meta2.legend(loc='upper right', fontsize=8)
    ax_meta.set_title('Entropy & Confidence per Preset', fontweight='bold')

    plt.savefig(os.path.join(FIGURES_DIR, 'phase168_control_room.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 168, 'name': 'control_room_ui',
        'pc_meanings': pc_meanings,
        'presets': {k: {'coords': [round(c, 3) for c in v]} for k, v in presets.items()},
        'dashboard_data': {k: {
            'scores': v['scores'],
            'avg_entropy': v['avg_entropy'],
            'avg_conf': v['avg_conf'],
        } for k, v in dashboard_data.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase168_control_room.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P168 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
