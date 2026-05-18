# -*- coding: utf-8 -*-
"""
Phase 24: Execution Port Discovery (Opus Original)
P22 showed MIN@L16=67% but MAX@L22=0%. Why?

Hypothesis: L16 is the UNIVERSAL EXECUTION PORT.
Any operation injected at L16 will execute.
Operations injected at other layers fail because they lack
downstream propagation distance.

Method: For each operation (MIN, MAX, SUM), try injection at
EVERY layer (0-23) and measure accuracy.
Creates a complete "executability heatmap."

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_register(model, tok, prompts, layer):
    """Extract average hidden state at a specific layer."""
    vecs = []
    for prompt in prompts:
        captured = [None]
        def cap(module, input, output):
            captured[0] = get_last_token(output)
        h = model.model.layers[layer].register_forward_hook(cap)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        vecs.append(captured[0])
    return torch.stack(vecs).mean(dim=0)


def main():
    print("[P24] Execution Port Discovery (Opus Original)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # === Extract operation vectors from their NATIVE layers ===
    print("  Extracting operation vectors from native layers...")
    operations = {
        'MIN': {
            'prompts': [f"def f(): return min({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:12],
            'native_layer': 16,
            'fn': min,
        },
        'MAX': {
            'prompts': [f"def f(): return max({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:12],
            'native_layer': 22,
            'fn': max,
        },
        'SUM': {
            'prompts': [f"def f(): return {a} + {b} =" for a in range(1, 6) for b in range(1, 6) if a+b < 10][:12],
            'native_layer': 20,
            'fn': lambda a, b: a + b,
        },
    }

    # Extract from native layer AND from ALL layers
    op_vecs = {}  # {op_name: {layer: vec}}
    for op_name, op_info in operations.items():
        op_vecs[op_name] = {}
        # Extract from native layer
        native_vec = extract_register(model, tok, op_info['prompts'], op_info['native_layer'])
        op_vecs[op_name]['native'] = native_vec
        # Also extract from key candidate layers
        for layer in range(0, n_layers, 2):  # every 2 layers for speed
            vec = extract_register(model, tok, op_info['prompts'], layer)
            op_vecs[op_name][layer] = vec
        print(f"    {op_name}: extracted from {n_layers//2} layers + native(L{op_info['native_layer']})")

    # === Test data ===
    test_data = [
        ("3, 7) =", 3, 7),
        ("2, 8) =", 2, 8),
        ("5, 1) =", 5, 1),
        ("9, 4) =", 9, 4),
        ("6, 3) =", 6, 3),
        ("4, 2) =", 4, 2),
    ]

    # === Sweep: inject each operation's native vector at every layer ===
    print("\n  Layer sweep: inject native vec at each layer...")
    heatmap = {}  # {op_name: {inject_layer: accuracy}}

    for op_name, op_info in operations.items():
        native_vec = op_vecs[op_name]['native']
        heatmap[op_name] = {}

        for inject_layer in range(n_layers):
            correct = 0
            total = 0
            for data_str, a, b in test_data:
                expected = op_info['fn'](a, b)
                if expected >= 10:
                    continue
                total += 1

                def inject(module, input, output, v=native_vec):
                    return replace_last_token(output, v)

                h = model.model.layers[inject_layer].register_forward_hook(inject)
                inp = tok(data_str, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                h.remove()

                pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
                if pred == str(expected):
                    correct += 1

            acc = correct / total if total > 0 else 0
            heatmap[op_name][inject_layer] = round(acc, 4)

        # Print best layers
        best_layers = sorted(heatmap[op_name].items(), key=lambda x: -x[1])[:3]
        print(f"    {op_name} best: {[(f'L{l}', f'{a:.0%}') for l, a in best_layers]}")

    # === Also test: extract at L16 for ALL operations (is L16 universal?) ===
    print("\n  Universal port test: extract ALL ops at L16, inject at L16...")
    l16_results = {}
    for op_name, op_info in operations.items():
        vec_at_l16 = extract_register(model, tok, op_info['prompts'], 16)
        correct = 0
        total = 0
        for data_str, a, b in test_data:
            expected = op_info['fn'](a, b)
            if expected >= 10:
                continue
            total += 1

            def inject(module, input, output, v=vec_at_l16):
                return replace_last_token(output, v)

            h = model.model.layers[16].register_forward_hook(inject)
            inp = tok(data_str, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()

            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            if pred == str(expected):
                correct += 1

        acc = correct / total if total > 0 else 0
        l16_results[op_name] = round(acc, 4)
        print(f"    {op_name}@L16: {acc:.1%}")

    # Save
    output = {
        'phase': 24, 'name': 'execution_port',
        'heatmap': {op: {str(k): v for k, v in layers.items()} for op, layers in heatmap.items()},
        'l16_universal': l16_results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase24_port.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot: executability heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Heatmap
    ops = list(heatmap.keys())
    layers = list(range(n_layers))
    mat = np.zeros((len(ops), len(layers)))
    for i, op in enumerate(ops):
        for j, layer in enumerate(layers):
            mat[i, j] = heatmap[op].get(layer, 0)

    im = axes[0].imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[0].set_yticks(range(len(ops)))
    axes[0].set_yticklabels(ops)
    axes[0].set_xlabel('Injection Layer', fontsize=11)
    axes[0].set_title('Executability Heatmap\n(native vec injected at each layer)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='Accuracy')
    # Mark native layers
    for i, op in enumerate(ops):
        native_l = operations[op]['native_layer']
        axes[0].plot(native_l, i, 'w*', markersize=15, markeredgecolor='black')

    # Per-operation line plot
    for op in ops:
        accs = [heatmap[op].get(l, 0) for l in layers]
        axes[1].plot(layers, accs, 'o-', linewidth=2, markersize=4, label=op)
    axes[1].set_xlabel('Injection Layer', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Accuracy vs Injection Layer', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # L16 universal test
    l16_ops = list(l16_results.keys())
    l16_accs = [l16_results[op] for op in l16_ops]
    colors = ['tab:green', 'tab:red', 'tab:blue']
    axes[2].bar(l16_ops, l16_accs, color=colors, edgecolor='black')
    axes[2].set_ylabel('Accuracy', fontsize=11)
    axes[2].set_title('L16 as Universal Port\n(all ops extracted & injected at L16)', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate(l16_accs):
        axes[2].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    plt.suptitle('Phase 24: Execution Port Discovery\nWhich layers can execute which programs?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase24_port.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
