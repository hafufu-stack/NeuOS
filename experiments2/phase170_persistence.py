# -*- coding: utf-8 -*-
"""
Phase 170: Soul Persistence & Portability
Save the PCA basis + soul coordinates to a compact file.
Reload them without the original training data or model history.
Verify they still produce identical results.

"A soul in a jar. Open it anywhere, anytime."
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
SOUL_JAR_PATH = os.path.join(RESULTS_DIR, 'soul_jar.npz')


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


def save_soul_jar(pca, soul_coords, metadata, path):
    """Save PCA basis + soul coordinates to a compact .npz file."""
    np.savez_compressed(path,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_variance=pca.explained_variance_ratio_,
        **{('soul_%s' % name): np.array(coords) for name, coords in soul_coords.items()},
    )
    # Save metadata separately as JSON
    meta_path = path.replace('.npz', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return os.path.getsize(path), os.path.getsize(meta_path)


def load_soul_jar(path, device):
    """Load PCA basis + soul coordinates from a .npz file."""
    data = np.load(path)
    pca = PCA(n_components=data['pca_components'].shape[0])
    pca.components_ = data['pca_components']
    pca.mean_ = data['pca_mean']
    pca.explained_variance_ratio_ = data['pca_variance']
    # Needed for inverse_transform
    pca.n_features_in_ = data['pca_mean'].shape[0]

    souls = {}
    for key in data.files:
        if key.startswith('soul_'):
            name = key[5:]
            coords = data[key]
            # Reconstruct 896d vector
            full_coords = np.zeros(pca.components_.shape[0])
            full_coords[:len(coords)] = coords
            vec_896 = pca.inverse_transform(full_coords.reshape(1, -1))[0]
            souls[name] = torch.tensor(vec_896, dtype=torch.float32, device=device)

    meta_path = path.replace('.npz', '_meta.json')
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    return pca, souls, metadata


def main():
    print("[P170] Soul Persistence & Portability")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train base souls
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

    souls = {}
    for seed in [42, 100, 200, 300]:
        souls['MIN_s%d' % seed] = train_soul(model, tok, min_data, DEVICE, seed=seed)
        souls['MAX_s%d' % seed] = train_soul(model, tok, max_data, DEVICE, seed=seed)

    # Build PCA
    matrix = np.array([v.cpu().numpy() for v in souls.values()])
    pca = PCA(n_components=8)
    pca.fit(matrix)

    # Baselines with original trained souls
    min_orig_acc, min_orig_preds = evaluate(model, tok, souls['MIN_s42'], min_test, DEVICE)
    max_orig_acc, max_orig_preds = evaluate(model, tok, souls['MAX_s42'], max_test, DEVICE)
    print("  Original MIN: %.0f%% (preds: %s)" % (min_orig_acc*100, min_orig_preds))
    print("  Original MAX: %.0f%% (preds: %s)" % (max_orig_acc*100, max_orig_preds))

    # Save to soul jar
    print("\n  === SAVING SOUL JAR ===")
    min_coords = pca.transform(souls['MIN_s42'].cpu().numpy().reshape(1, -1))[0]
    max_coords = pca.transform(souls['MAX_s42'].cpu().numpy().reshape(1, -1))[0]

    # Also save the "compiled" souls from P166
    compiled_souls = {
        'MIN_trained': min_coords.tolist(),
        'MAX_trained': max_coords.tolist(),
        'MIN_compiled': [0, -1.5, 0, 0, 0, -1.5, 0, 0],
        'MAX_compiled': [0, 1.5, 0, 0, 0, 1.5, 0, 0],
    }

    metadata = {
        'model': 'Qwen2.5-0.5B-Instruct',
        'hidden_size': 896,
        'injection_layer': LAYER,
        'n_pca_components': 8,
        'effective_dimensions': 7,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'soul_names': list(compiled_souls.keys()),
    }

    npz_size, meta_size = save_soul_jar(pca, compiled_souls, metadata, SOUL_JAR_PATH)
    total_size = npz_size + meta_size
    print("  Soul jar saved: %s" % SOUL_JAR_PATH)
    print("  Size: %.1f KB (npz=%.1f KB, meta=%.1f KB)" % (
        total_size/1024, npz_size/1024, meta_size/1024))
    print("  Contains %d souls" % len(compiled_souls))

    # Original 896d soul vectors for comparison
    orig_size = 896 * 4 * len(compiled_souls)  # float32
    compression_ratio = orig_size / total_size
    print("  Compression: %.0fx (%.1f KB -> %.1f KB)" % (
        compression_ratio, orig_size/1024, total_size/1024))

    # Reload and verify
    print("\n  === LOADING SOUL JAR ===")
    pca_loaded, loaded_souls, loaded_meta = load_soul_jar(SOUL_JAR_PATH, DEVICE)
    print("  Loaded %d souls from jar" % len(loaded_souls))
    print("  Metadata: %s" % loaded_meta.get('model', 'unknown'))

    # Verify loaded souls produce identical results
    print("\n  === VERIFICATION ===")
    verification = {}
    for soul_name in ['MIN_trained', 'MAX_trained', 'MIN_compiled', 'MAX_compiled']:
        if soul_name not in loaded_souls:
            continue
        test_data = min_test if 'MIN' in soul_name else max_test
        acc, preds = evaluate(model, tok, loaded_souls[soul_name], test_data, DEVICE)
        verification[soul_name] = {
            'accuracy': round(acc, 4),
            'predictions': preds,
        }
        print("  %s: %.0f%% (preds: %s)" % (soul_name, acc*100, preds))

    # Check fidelity: cosine similarity between original and loaded
    print("\n  === FIDELITY CHECK ===")
    min_orig_vec = souls['MIN_s42']
    min_loaded_vec = loaded_souls.get('MIN_trained')
    if min_loaded_vec is not None:
        cos = torch.nn.functional.cosine_similarity(
            min_orig_vec.unsqueeze(0), min_loaded_vec.unsqueeze(0)).item()
        l2 = (min_orig_vec - min_loaded_vec).norm().item()
        print("  MIN: cosine=%.6f, L2=%.6f" % (cos, l2))
    else:
        cos = 0; l2 = float('inf')

    max_orig_vec = souls['MAX_s42']
    max_loaded_vec = loaded_souls.get('MAX_trained')
    if max_loaded_vec is not None:
        cos_max = torch.nn.functional.cosine_similarity(
            max_orig_vec.unsqueeze(0), max_loaded_vec.unsqueeze(0)).item()
        l2_max = (max_orig_vec - max_loaded_vec).norm().item()
        print("  MAX: cosine=%.6f, L2=%.6f" % (cos_max, l2_max))
    else:
        cos_max = 0; l2_max = float('inf')

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original vs Loaded accuracy
    ax = axes[0]
    names = list(verification.keys())
    accs = [verification[n]['accuracy'] for n in names]
    colors = ['#E91E63' if 'MIN' in n else '#2196F3' for n in names]
    hatches = ['' if 'trained' in n else '///' for n in names]
    bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=1.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy')
    ax.set_title('Soul Jar Verification\n(loaded from disk)', fontweight='bold')
    ax.set_xticklabels(names, fontsize=8, rotation=15)

    # Panel 2: Size comparison
    ax = axes[1]
    categories = ['Raw Vectors\n(896d x float32)', 'Soul Jar\n(7D + PCA)']
    sizes = [orig_size / 1024, total_size / 1024]
    colors = ['#F44336', '#4CAF50']
    bars = ax.bar(categories, sizes, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                '%.1f KB' % val, ha='center', fontweight='bold', fontsize=12)
    ax.set_ylabel('Size (KB)')
    ax.set_title('Storage Compression\n(%.0fx smaller!)' % compression_ratio,
                 fontweight='bold')

    # Panel 3: Soul jar contents
    ax = axes[2]
    ax.axis('off')
    contents = [
        ['Component', 'Size', 'Content'],
        ['PCA Basis', '%.1f KB' % (npz_size/1024), '8 x 896 matrix'],
        ['PCA Mean', 'included', '896-dim vector'],
        ['Soul Coords', '< 1 KB', '4 souls x 8 floats'],
        ['Metadata', '%.1f KB' % (meta_size/1024), 'model, layer, date'],
        ['TOTAL', '%.1f KB' % (total_size/1024), '%d souls portable' % len(compiled_souls)],
    ]
    table = ax.table(cellText=contents[1:], colLabels=contents[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for j in range(3):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    table[5, 0].set_facecolor('#E8F5E9')
    table[5, 1].set_facecolor('#E8F5E9')
    table[5, 2].set_facecolor('#E8F5E9')
    ax.set_title('Soul Jar Contents', fontweight='bold', pad=20)

    plt.suptitle('Phase 170: Soul Persistence & Portability\n'
                 '"A soul in a jar. Open it anywhere, anytime."',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase170_persistence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 170, 'name': 'soul_persistence',
        'jar_path': SOUL_JAR_PATH,
        'jar_size_bytes': total_size,
        'raw_size_bytes': orig_size,
        'compression_ratio': round(compression_ratio, 1),
        'n_souls': len(compiled_souls),
        'verification': verification,
        'fidelity': {
            'min_cosine': round(cos, 6),
            'max_cosine': round(cos_max, 6),
        },
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase170_persistence.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P170 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
