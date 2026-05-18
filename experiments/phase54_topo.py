# -*- coding: utf-8 -*-
"""
Phase 54: Topological Proprioception
Fix P49's overfit (Train 100%, Test 0%) by using topology-invariant
features: L2-norm ratios, layer-to-layer divergence, entropy.
Not raw vectors (which overfit to specific noise).

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, get_last_token, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_topo_features(model, tok, prompt, layers, device, inject_vec=None, inject_layer=None):
    """Extract topology-invariant features (norms, ratios, entropy)."""
    vecs = {}
    for layer in layers:
        cap = [None]
        def capture(module, input, output):
            cap[0] = get_last_token(output)
        hooks = []
        if inject_vec is not None and inject_layer is not None and layer >= inject_layer:
            def inject_fn(module, input, output, v=inject_vec):
                return replace_last_token(output, v)
            hooks.append(model.model.layers[inject_layer].register_forward_hook(inject_fn))
        hooks.append(model.model.layers[layer].register_forward_hook(capture))
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inp)
        for h in hooks:
            h.remove()
        vecs[layer] = cap[0].float().cpu().numpy().flatten()

    # Topological features
    features = []
    norms = {l: np.linalg.norm(v) for l, v in vecs.items()}
    for l in layers:
        features.append(norms[l])

    # Layer-to-layer norm ratios
    sorted_layers = sorted(layers)
    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i+1]
        ratio = norms[l2] / (norms[l1] + 1e-8)
        features.append(ratio)

    # Cosine similarity between consecutive layers
    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i+1]
        cos = np.dot(vecs[l1], vecs[l2]) / (norms[l1] * norms[l2] + 1e-8)
        features.append(cos)

    # Entropy of top-k elements per layer
    for l in layers:
        v = np.abs(vecs[l])
        p = v / (v.sum() + 1e-8)
        ent = -np.sum(p * np.log(p + 1e-10))
        features.append(ent)

    return np.array(features)


def main():
    print("[P54] Topological Proprioception")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    LAYERS = [2, 8, 16, 22]

    # Generate normal and anomalous data
    print("  Generating training data...")
    prompts = [f"def f(): return min({a}, {b}) ="
               for a in range(2, 8) for b in range(2, 8) if a != b]

    normal_feats = []
    for p in prompts[:20]:
        feat = extract_topo_features(model, tok, p, LAYERS, DEVICE)
        normal_feats.append(feat)

    # Generate anomalies with DIFFERENT noise vectors
    anomaly_feats = []
    for i, p in enumerate(prompts[:20]):
        noise = torch.randn(hidden_size, device=DEVICE) * (2.0 + i * 0.5)
        feat = extract_topo_features(model, tok, p, LAYERS, DEVICE,
                                     inject_vec=noise, inject_layer=16)
        anomaly_feats.append(feat)

    X = np.array(normal_feats + anomaly_feats)
    y = [0] * len(normal_feats) + [1] * len(anomaly_feats)

    # Cross-validated accuracy
    print("  Training topological anomaly detector...")
    clf = LogisticRegression(max_iter=500, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    cv_acc = scores.mean()
    print(f"    CV accuracy: {cv_acc:.0%} (+/- {scores.std():.0%})")

    # Train final model
    clf.fit(X, y)
    train_acc = clf.score(X, y)

    # Test on UNSEEN noise vectors (the P49 failure case)
    print("\n  Testing on unseen noise vectors...")
    test_normal = []
    test_anomaly = []
    for p in prompts[20:30]:
        feat = extract_topo_features(model, tok, p, LAYERS, DEVICE)
        test_normal.append(feat)
    for i, p in enumerate(prompts[20:30]):
        noise = torch.randn(hidden_size, device=DEVICE) * (3.0 + i * 0.3)
        feat = extract_topo_features(model, tok, p, LAYERS, DEVICE,
                                     inject_vec=noise, inject_layer=16)
        test_anomaly.append(feat)

    X_test = np.array(test_normal + test_anomaly)
    y_test = [0] * len(test_normal) + [1] * len(test_anomaly)
    test_acc = clf.score(X_test, y_test)
    preds = clf.predict(X_test)
    tp = sum(1 for p, t in zip(preds, y_test) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, y_test) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, y_test) if p == 0 and t == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"    Test accuracy: {test_acc:.0%}")
    print(f"    Precision: {precision:.0%}, Recall: {recall:.0%}")
    print(f"    (P49 baseline: Test 0%)")

    # Save
    output = {
        'phase': 54, 'name': 'topological_proprioception',
        'cv_accuracy': round(float(cv_acc), 4),
        'train_acc': round(train_acc, 4),
        'test_acc': round(test_acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'n_features': len(normal_feats[0]),
        'p49_baseline_test': 0.0,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase54_topo.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(['P49\n(Raw Vec)', 'P54\n(Topological)'],
                [0.0, test_acc],
                color=['tab:red', 'tab:green'], edgecolor='black')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Proprioception: P49 vs P54', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([0.0, test_acc]):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=14)

    axes[1].bar(['Precision', 'Recall', 'CV Acc'],
                [precision, recall, cv_acc],
                color=['tab:blue', 'tab:orange', 'tab:purple'], edgecolor='black')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Detection Quality', fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate([precision, recall, cv_acc]):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    plt.suptitle('Phase 54: Topological Proprioception\nNoise-invariant anomaly detection via topology features',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase54_topo.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
