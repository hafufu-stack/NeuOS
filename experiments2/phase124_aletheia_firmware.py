# -*- coding: utf-8 -*-
"""
Phase 124: Aletheia Firmware (NeuOS x Aletheia)
Test multi-layer simultaneous soul injection vs single-layer injection.

"One layer whispers. Five layers roar."
"""
import torch, json, os, gc, numpy as np, time, sys, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TARGET_LAYERS = [4, 8, 12, 16, 20]
MULTI_CONFIGS = {
    'L4+L16': [4, 16],
    'L8+L16': [8, 16],
    'L4+L8+L16': [4, 8, 16],
    'All 5 layers': [4, 8, 12, 16, 20],
}

MIN_TRAIN = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
             ("4, 6) =","4"),("9, 3) =","3")]
MIN_TEST  = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
             ("8, 5) =","5"),("1, 4) =","1")]


def gradient_train(model, tok, train, layer, device, seed=42, epochs=150):
    """Standard gradient soul vector training at a specific layer."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in train:
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


def evaluate_single_layer(model, tok, vec, data, layer, device):
    """Evaluate a soul vector at one layer."""
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


def evaluate_multi_layer(model, tok, layer_vecs, data, device):
    """
    Evaluate with multiple soul vectors injected at their respective layers.
    layer_vecs: dict of {layer_idx: soul_vector}
    """
    c = 0
    for p, e in data:
        hooks = []
        for layer_idx, vec in layer_vecs.items():
            def inj(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer_idx].register_forward_hook(inj)
            hooks.append(h)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        for h in hooks:
            h.remove()
        if tok.decode(out.logits[0, -1, :].argmax().item()).strip() == e:
            c += 1
    return c / len(data)


def main():
    print("[P124] Aletheia Firmware")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    all_data = MIN_TRAIN + MIN_TEST

    # ========== Step 1: Train souls at each layer ==========
    print("  Step 1: Training soul vectors at each target layer...")
    layer_souls = {}
    single_accs = {}
    for layer in TARGET_LAYERS:
        print(f"    Training at L{layer} (150 epochs, seed=42)...")
        vec = gradient_train(model, tok, MIN_TRAIN, layer, DEVICE,
                             seed=42, epochs=150)
        layer_souls[layer] = vec

        # Evaluate single layer
        acc_train = evaluate_single_layer(model, tok, vec, MIN_TRAIN, layer, DEVICE)
        acc_test = evaluate_single_layer(model, tok, vec, MIN_TEST, layer, DEVICE)
        acc_all = evaluate_single_layer(model, tok, vec, all_data, layer, DEVICE)
        single_accs[layer] = {
            'train': round(acc_train, 4),
            'test': round(acc_test, 4),
            'all': round(acc_all, 4),
        }
        print(f"      L{layer}: train={acc_train:.2%}, "
              f"test={acc_test:.2%}, all={acc_all:.2%}")

    # ========== Step 2: Test multi-layer combinations ==========
    print("\n  Step 2: Testing multi-layer injection combinations...")
    multi_accs = {}
    for config_name, layers in MULTI_CONFIGS.items():
        vecs = {l: layer_souls[l] for l in layers}
        acc_train = evaluate_multi_layer(model, tok, vecs, MIN_TRAIN, DEVICE)
        acc_test = evaluate_multi_layer(model, tok, vecs, MIN_TEST, DEVICE)
        acc_all = evaluate_multi_layer(model, tok, vecs, all_data, DEVICE)
        multi_accs[config_name] = {
            'layers': layers,
            'train': round(acc_train, 4),
            'test': round(acc_test, 4),
            'all': round(acc_all, 4),
        }
        print(f"    {config_name}: train={acc_train:.2%}, "
              f"test={acc_test:.2%}, all={acc_all:.2%}")

    # ========== Save results ==========
    output = {
        'phase': 124, 'name': 'aletheia_firmware',
        'target_layers': TARGET_LAYERS,
        'single_layer_accs': {f'L{k}': v for k, v in single_accs.items()},
        'multi_layer_accs': multi_accs,
        'multi_configs': {k: v for k, v in MULTI_CONFIGS.items()},
        'train_size': len(MIN_TRAIN),
        'test_size': len(MIN_TEST),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase124_aletheia_firmware.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Results saved.")

    # ========== Plot ==========
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Collect all configurations for bar chart
    all_configs = []
    all_accs_all = []
    all_accs_train = []
    all_accs_test = []
    bar_colors = []

    # Single layer configs
    for layer in TARGET_LAYERS:
        all_configs.append(f'L{layer}')
        all_accs_all.append(single_accs[layer]['all'])
        all_accs_train.append(single_accs[layer]['train'])
        all_accs_test.append(single_accs[layer]['test'])
        bar_colors.append('steelblue')

    # Multi layer configs
    multi_colors = ['coral', 'salmon', 'tomato', 'firebrick']
    for idx, (config_name, info) in enumerate(multi_accs.items()):
        all_configs.append(config_name)
        all_accs_all.append(info['all'])
        all_accs_train.append(info['train'])
        all_accs_test.append(info['test'])
        bar_colors.append(multi_colors[idx % len(multi_colors)])

    # Panel 1: Bar chart
    ax = axes[0]
    x = np.arange(len(all_configs))
    w = 0.25
    bars1 = ax.bar(x - w, all_accs_train, w, label='Train', color='lightblue',
                   edgecolor='black')
    bars2 = ax.bar(x, all_accs_test, w, label='Test', color='lightsalmon',
                   edgecolor='black')
    bars3 = ax.bar(x + w, all_accs_all, w, label='All', color=bar_colors,
                   edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(all_configs, fontsize=8, rotation=35, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.3)
    ax.set_title('Single vs Multi-Layer Injection', fontweight='bold')
    ax.legend(fontsize=9)
    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.0%}', ha='center', fontsize=7)
    # Add separator line between single and multi
    ax.axvline(x=len(TARGET_LAYERS) - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(TARGET_LAYERS) / 2 - 0.5, 1.2, 'Single', ha='center',
            fontsize=9, color='steelblue', fontweight='bold')
    ax.text(len(TARGET_LAYERS) + len(MULTI_CONFIGS) / 2 - 0.5, 1.2, 'Multi',
            ha='center', fontsize=9, color='firebrick', fontweight='bold')

    # Panel 2: Heatmap of layer combinations
    ax = axes[1]
    # Create a matrix: rows = configs (single + multi), cols = layers
    n_configs = len(all_configs)
    n_layers = len(TARGET_LAYERS)
    heat = np.zeros((n_configs, n_layers))

    # Fill in which layers are active for each config
    for i, layer in enumerate(TARGET_LAYERS):
        # Single layer: only that layer is active
        heat[i, TARGET_LAYERS.index(layer)] = single_accs[layer]['all']

    for j, (config_name, info) in enumerate(multi_accs.items()):
        row_idx = len(TARGET_LAYERS) + j
        for layer in info['layers']:
            heat[row_idx, TARGET_LAYERS.index(layer)] = info['all']

    im = ax.imshow(heat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'L{l}' for l in TARGET_LAYERS])
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(all_configs, fontsize=8)
    ax.set_xlabel('Injection Layer')
    ax.set_ylabel('Configuration')
    for i in range(n_configs):
        for j in range(n_layers):
            if heat[i, j] > 0:
                ax.text(j, i, f'{heat[i, j]:.0%}',
                        ha='center', va='center', fontsize=7,
                        color='white' if heat[i, j] > 0.5 else 'black')
            else:
                ax.text(j, i, '-', ha='center', va='center',
                        fontsize=7, color='gray')
    plt.colorbar(im, ax=ax)
    ax.set_title('Layer Activation Heatmap', fontweight='bold')
    # Separator
    ax.axhline(y=len(TARGET_LAYERS) - 0.5, color='white', linewidth=2)

    plt.suptitle('Phase 124: Aletheia Firmware\n'
                 '"One layer whispers. Five layers roar."',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase124_aletheia_firmware.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  === Summary ===")
    print("  Single-layer injection:")
    for layer in TARGET_LAYERS:
        print(f"    L{layer}: {single_accs[layer]['all']:.2%}")
    print("  Multi-layer injection:")
    for config_name, info in multi_accs.items():
        print(f"    {config_name}: {info['all']:.2%}")
    best_single = max(TARGET_LAYERS, key=lambda l: single_accs[l]['all'])
    best_multi = max(multi_accs.keys(), key=lambda k: multi_accs[k]['all'])
    print(f"  Best single: L{best_single} ({single_accs[best_single]['all']:.2%})")
    print(f"  Best multi: {best_multi} ({multi_accs[best_multi]['all']:.2%})")
    print(f"  Completed in {time.time()-start:.0f}s")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
