# -*- coding: utf-8 -*-
"""
Phase 66: Neural Immune System
Combine P52 (attack detection), P54 (anomaly detection via topology),
and P63 (self-recognition) into an adaptive immune system.
Memory cells remember past attacks. Self/non-self distinction.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_prog(model, tok, train, layer, device, seed=42):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(80):
        for prompt, target_str in train:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def inject(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def extract_topo_features(model, tok, vec, prompt, layers, device):
    """Extract topological features (P54-style) for immune system."""
    caps = {}
    hooks = []
    for l in layers:
        def make_cap(layer_idx):
            def cap_fn(module, input, output):
                caps[layer_idx] = get_last_token(output).float().cpu()
            return cap_fn
        h = model.model.layers[l].register_forward_hook(make_cap(l))
        hooks.append(h)

    def inject(module, input, output, v=vec):
        return replace_last_token(output, v)
    h_inj = model.model.layers[layers[0]].register_forward_hook(inject)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)
    h_inj.remove()
    for h in hooks:
        h.remove()

    # Topological features: norm ratios, entropy changes
    features = []
    prev_norm = None
    for l in layers:
        v = caps[l].flatten()
        norm = v.norm().item()
        entropy = -(torch.softmax(v, 0) * torch.log_softmax(v, 0)).sum().item()
        features.append(norm)
        features.append(entropy)
        if prev_norm is not None:
            features.append(norm / (prev_norm + 1e-8))
        prev_norm = norm
    return np.array(features)


def main():
    print("[P66] Neural Immune System")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8
    immune_layers = [8, 12, 16, 20, 22]

    for p in model.parameters():
        p.requires_grad = False

    # Compile "self" programs
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")]

    print("  Building immune memory (self programs)...")
    self_vecs = []
    for seed in range(8):
        v = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=seed*100)
        self_vecs.append(v)
    for seed in range(8):
        v = compile_prog(model, tok, max_data, target_layer, DEVICE, seed=seed*100+50)
        self_vecs.append(v)

    # Generate "pathogen" programs (random noise, adversarial)
    print("  Generating pathogens (non-self)...")
    pathogen_vecs = []
    for i in range(16):
        # Random noise pathogens
        p = torch.randn(model.config.hidden_size, device=DEVICE) * np.random.uniform(0.5, 5.0)
        pathogen_vecs.append(p)

    # Extract immune features for all
    prompt = "3, 7) ="
    print("  Extracting immune signatures...")
    self_features = []
    for v in self_vecs:
        feat = extract_topo_features(model, tok, v, prompt, immune_layers, DEVICE)
        self_features.append(feat)

    pathogen_features = []
    for v in pathogen_vecs:
        feat = extract_topo_features(model, tok, v, prompt, immune_layers, DEVICE)
        pathogen_features.append(feat)

    # Train immune classifier (SVM = adaptive immunity)
    X = np.array(self_features + pathogen_features)
    y = np.array([1]*len(self_features) + [0]*len(pathogen_features))

    # Leave-one-out cross-validation
    print("\n  Training adaptive immune system (SVM)...")
    correct = 0
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i]
        clf = SVC(kernel='rbf', gamma='scale')
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]
        if pred == y_test:
            correct += 1
    loo_accuracy = correct / len(X)
    print(f"    LOO accuracy: {loo_accuracy:.0%}")

    # Test on novel threats
    print("\n  Testing on novel threats...")
    clf_full = SVC(kernel='rbf', gamma='scale')
    clf_full.fit(X, y)

    novel_results = {}
    # Novel self (new seed)
    novel_self = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=9999)
    feat = extract_topo_features(model, tok, novel_self, prompt, immune_layers, DEVICE)
    pred = clf_full.predict(feat.reshape(1, -1))[0]
    novel_results['novel_self'] = int(pred)
    print(f"    Novel self program: {'ACCEPTED' if pred == 1 else 'REJECTED'}")

    # Novel pathogen (scaled noise)
    novel_pathogen = torch.randn(model.config.hidden_size, device=DEVICE) * 10
    feat = extract_topo_features(model, tok, novel_pathogen, prompt, immune_layers, DEVICE)
    pred = clf_full.predict(feat.reshape(1, -1))[0]
    novel_results['novel_pathogen'] = int(pred)
    print(f"    Novel pathogen: {'ACCEPTED' if pred == 1 else 'REJECTED'}")

    # Autoimmune test: slightly corrupted self
    corrupted = self_vecs[0] + torch.randn_like(self_vecs[0]) * 0.1
    feat = extract_topo_features(model, tok, corrupted, prompt, immune_layers, DEVICE)
    pred = clf_full.predict(feat.reshape(1, -1))[0]
    novel_results['corrupted_self'] = int(pred)
    print(f"    Corrupted self (slight): {'ACCEPTED' if pred == 1 else 'REJECTED'}")

    heavily_corrupted = self_vecs[0] + torch.randn_like(self_vecs[0]) * 2.0
    feat = extract_topo_features(model, tok, heavily_corrupted, prompt, immune_layers, DEVICE)
    pred = clf_full.predict(feat.reshape(1, -1))[0]
    novel_results['heavily_corrupted'] = int(pred)
    print(f"    Heavily corrupted self: {'ACCEPTED' if pred == 1 else 'REJECTED'}")

    # Save
    output = {
        'phase': 66, 'name': 'neural_immune_system',
        'n_self': len(self_vecs), 'n_pathogen': len(pathogen_vecs),
        'loo_accuracy': round(loo_accuracy, 4),
        'novel_results': novel_results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase66_immune.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(['Self\n(16 programs)', 'Pathogen\n(16 noise)'],
                [len(self_vecs), len(pathogen_vecs)],
                color=['tab:green', 'tab:red'], edgecolor='black')
    axes[0].set_title('Immune Training Data', fontweight='bold')
    axes[0].set_ylabel('Count')

    axes[1].bar(['LOO\nAccuracy'], [loo_accuracy],
                color='tab:blue', edgecolor='black')
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Adaptive Immunity (SVM)', fontweight='bold')
    axes[1].text(0, loo_accuracy+0.03, f'{loo_accuracy:.0%}', ha='center', fontweight='bold', fontsize=14)

    # Novel threats
    labels = list(novel_results.keys())
    values = list(novel_results.values())
    colors = ['tab:green' if v == 1 else 'tab:red' for v in values]
    axes[2].bar(range(len(labels)), values, color=colors, edgecolor='black')
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=8)
    axes[2].set_ylabel('Prediction (1=self, 0=pathogen)')
    axes[2].set_title('Novel Threat Response', fontweight='bold')

    plt.suptitle('Phase 66: Neural Immune System\nSelf/non-self discrimination with adaptive immunity',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase66_immune.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
