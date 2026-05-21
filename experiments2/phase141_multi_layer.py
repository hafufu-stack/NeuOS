# -*- coding: utf-8 -*-
"""
Phase 141: Multi-Layer Soul Injection
What if we inject the soul at MULTIPLE layers simultaneously?
P137 showed MIN peaks at L6, L8, L14 - combine them!

"A soul that resonates across multiple frequencies becomes unbreakable."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_multi_layer_soul(model, tok, data, device, layers, seed=42, epochs=150):
    """Train a SINGLE soul vector injected at MULTIPLE layers."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for p, t in data:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(device)
            hooks = []
            for layer in layers:
                def inj(m, i, o, v=vec): return replace_last_token(o, v)
                h = model.model.layers[layer].register_forward_hook(inj)
                hooks.append(h)
            out = model(**inp)
            for h in hooks:
                h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def train_independent_souls(model, tok, data, device, layers, seed=42, epochs=150):
    """Train INDEPENDENT soul vectors for each layer."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vecs = {l: torch.randn(hs, device=device) * 0.01 for l in layers}
    for v in vecs.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(list(vecs.values()), lr=0.01)
    for _ in range(epochs):
        for p, t in data:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(device)
            hooks = []
            for layer in layers:
                def inj(m, i, o, v=vecs[layer]): return replace_last_token(o, v)
                h = model.model.layers[layer].register_forward_hook(inj)
                hooks.append(h)
            out = model(**inp)
            for h in hooks:
                h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return {l: v.detach() for l, v in vecs.items()}


def evaluate_multi(model, tok, vecs_or_vec, data, device, layers, shared=True):
    c = 0
    for p, e in data:
        hooks = []
        for layer in layers:
            if shared:
                def inj(m, i, o, v=vecs_or_vec): return replace_last_token(o, v)
            else:
                def inj(m, i, o, v=vecs_or_vec[layer]): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inj)
            hooks.append(h)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        for h in hooks:
            h.remove()
        if tok.decode(out.logits[0, -1, :].argmax().item()).strip() == e:
            c += 1
    return c / len(data)


def evaluate_single(model, tok, vec, data, device, layer):
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


def main():
    print("[P141] Multi-Layer Soul Injection")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    train_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                   ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                  ("1, 5) =","1"),("8, 4) =","4")]
    all_data = train_data + test_data

    # Layer configs to test (based on P137: MIN peaks at L6, L8, L14)
    configs = {
        'L8_only': [8],
        'L6_only': [6],
        'L6+L8': [6, 8],
        'L6+L14': [6, 14],
        'L8+L14': [8, 14],
        'L6+L8+L14': [6, 8, 14],
        'L4+L8+L12': [4, 8, 12],
        'L6+L8+L14+L18': [6, 8, 14, 18],
    }

    results_shared = {}
    results_independent = {}

    for name, layers in configs.items():
        print("  Config %s (layers=%s)..." % (name, layers))

        # Strategy 1: Shared soul vector
        vec = train_multi_layer_soul(model, tok, train_data, DEVICE, layers)
        train_acc = evaluate_multi(model, tok, vec, train_data, DEVICE, layers, shared=True)
        test_acc = evaluate_multi(model, tok, vec, test_data, DEVICE, layers, shared=True)
        all_acc = evaluate_multi(model, tok, vec, all_data, DEVICE, layers, shared=True)
        results_shared[name] = {
            'train': round(train_acc, 4), 'test': round(test_acc, 4),
            'all': round(all_acc, 4), 'n_layers': len(layers)}
        print("    Shared:  train=%.0f%% test=%.0f%%" % (train_acc*100, test_acc*100))

        # Strategy 2: Independent souls per layer (only for multi-layer)
        if len(layers) > 1:
            vecs = train_independent_souls(model, tok, train_data, DEVICE, layers)
            train_acc = evaluate_multi(model, tok, vecs, train_data, DEVICE, layers, shared=False)
            test_acc = evaluate_multi(model, tok, vecs, test_data, DEVICE, layers, shared=False)
            all_acc = evaluate_multi(model, tok, vecs, all_data, DEVICE, layers, shared=False)
            results_independent[name] = {
                'train': round(train_acc, 4), 'test': round(test_acc, 4),
                'all': round(all_acc, 4), 'n_layers': len(layers)}
            # Cosine between layer-specific souls
            cos_pairs = {}
            for l1, l2 in combinations(layers, 2):
                cos = torch.nn.functional.cosine_similarity(
                    vecs[l1].unsqueeze(0), vecs[l2].unsqueeze(0)).item()
                cos_pairs["L%d-L%d" % (l1, l2)] = round(cos, 4)
            results_independent[name]['cosine_pairs'] = cos_pairs
            print("    Indep:   train=%.0f%% test=%.0f%% cos=%s" % (
                train_acc*100, test_acc*100,
                {k: "%.3f" % v for k, v in cos_pairs.items()}))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Shared soul accuracy by config
    ax = axes[0]
    names = list(results_shared.keys())
    test_accs = [results_shared[n]['test'] for n in names]
    n_layers = [results_shared[n]['n_layers'] for n in names]
    colors_bar = ['#2196F3' if nl == 1 else '#4CAF50' if nl == 2
                  else '#FF5722' if nl == 3 else '#9C27B0' for nl in n_layers]
    bars = ax.bar(range(len(names)), test_accs, color=colors_bar, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Shared Soul: Multi-Layer Injection', fontweight='bold')
    ax.set_ylim(0, 1.15)
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                "%.0f%%" % (acc*100), ha='center', fontsize=9)

    # Panel 2: Shared vs Independent comparison
    ax = axes[1]
    multi_names = [n for n in names if n in results_independent]
    x = np.arange(len(multi_names))
    w = 0.35
    shared_vals = [results_shared[n]['test'] for n in multi_names]
    indep_vals = [results_independent[n]['test'] for n in multi_names]
    ax.bar(x - w/2, shared_vals, w, label='Shared soul', color='#2196F3',
           edgecolor='black')
    ax.bar(x + w/2, indep_vals, w, label='Independent souls', color='#FF5722',
           edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(multi_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Shared vs Independent Souls', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)

    # Panel 3: Accuracy vs number of injection layers
    ax = axes[2]
    for name in names:
        nl = results_shared[name]['n_layers']
        ta = results_shared[name]['test']
        ax.scatter(nl, ta, s=100, color=colors_bar[names.index(name)],
                   edgecolors='black', zorder=5)
        ax.annotate(name, (nl, ta), fontsize=7, ha='center',
                    xytext=(0, 8), textcoords='offset points')
    ax.set_xlabel('Number of Injection Layers')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Does More Layers = Better?', fontweight='bold')
    ax.set_xlim(0.5, 5)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 141: Multi-Layer Soul Injection\n'
                 '"A soul that resonates across multiple frequencies"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase141_multi_layer.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 141, 'name': 'multi_layer_soul',
        'results_shared': results_shared,
        'results_independent': results_independent,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase141_multi_layer.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
