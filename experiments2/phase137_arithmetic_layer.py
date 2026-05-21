# -*- coding: utf-8 -*-
"""
Phase 137: Arithmetic Layer Scan
Scan all 24 layers for ADD/SUB/MIN/MAX to find the optimal layer per task.
P130 showed ADD/SUB get 0% at L8 -- where do they actually work?

"Each operation has a natural home in the architecture."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_soul(model, tok, data, device, layer=8, seed=42, epochs=150):
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


def evaluate(model, tok, vec, data, device, layer=8):
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
    print("[P137] Arithmetic Layer Scan (all 24 layers)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    n_layers = len(model.model.layers)
    print("  Model has %d layers" % n_layers)

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
    test_data = {
        'MIN': [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")],
        'MAX': [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")],
        'ADD': [("1, 2) =","3"),("3, 4) =","7"),("2, 5) =","7")],
        'SUB': [("9, 5) =","4"),("6, 1) =","5"),("4, 3) =","1")],
    }

    task_names = ['MIN', 'MAX', 'ADD', 'SUB']
    # results[task][layer] = {'train': acc, 'test': acc}
    results = {t: {} for t in task_names}

    for layer_idx in range(n_layers):
        layer_start = time.time()
        for task in task_names:
            vec = train_soul(model, tok, task_data[task], DEVICE,
                           layer=layer_idx, seed=42, epochs=150)
            train_acc = evaluate(model, tok, vec, task_data[task], DEVICE,
                               layer=layer_idx)
            test_acc = evaluate(model, tok, vec, test_data[task], DEVICE,
                              layer=layer_idx)
            results[task][layer_idx] = {
                'train': round(train_acc, 4),
                'test': round(test_acc, 4),
            }
        layer_time = time.time() - layer_start
        accs_str = "  ".join(["%s: %3.0f%%/%3.0f%%" % (
            t, results[t][layer_idx]['train']*100,
            results[t][layer_idx]['test']*100) for t in task_names])
        print("  L%02d: %s  (%.1fs)" % (layer_idx, accs_str, layer_time))

    # Find optimal layers
    print("\n  Optimal layers (by test accuracy):")
    optimal = {}
    for task in task_names:
        best_layer = max(range(n_layers),
                        key=lambda l: results[task][l]['test'])
        best_acc = results[task][best_layer]['test']
        optimal[task] = {'layer': best_layer, 'test_acc': round(best_acc, 4)}
        print("    %s: L%d (test=%.0f%%)" % (task, best_layer, best_acc*100))

    # Layer specialization index
    comparison_layers = [optimal['MIN']['layer'], optimal['MAX']['layer']]
    arithmetic_layers = [optimal['ADD']['layer'], optimal['SUB']['layer']]
    comp_mean = np.mean(comparison_layers)
    arith_mean = np.mean(arithmetic_layers)
    specialization_index = abs(comp_mean - arith_mean)
    print("\n  Layer specialization index: %.1f" % specialization_index)
    print("    Comparison tasks (MIN,MAX) mean optimal layer: %.1f" % comp_mean)
    print("    Arithmetic tasks (ADD,SUB) mean optimal layer: %.1f" % arith_mean)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {'MIN': '#2196F3', 'MAX': '#FF5722', 'ADD': '#4CAF50', 'SUB': '#9C27B0'}

    # Panel 1: Line plot of accuracy vs layer
    ax = axes[0]
    layers_x = list(range(n_layers))
    for task in task_names:
        test_accs = [results[task][l]['test'] * 100 for l in layers_x]
        ax.plot(layers_x, test_accs, 'o-', color=colors[task],
                label=task, linewidth=2, markersize=4)
        # Mark optimal
        opt_l = optimal[task]['layer']
        opt_a = optimal[task]['test_acc'] * 100
        ax.plot(opt_l, opt_a, '*', color=colors[task], markersize=15,
                markeredgecolor='black', markeredgewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Task Accuracy vs Layer', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-5, 105)
    ax.set_xticks(layers_x)
    ax.set_xticklabels([str(l) for l in layers_x], fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Heatmap of layer x task
    ax = axes[1]
    heatmap_data = np.zeros((len(task_names), n_layers))
    for i, task in enumerate(task_names):
        for l in range(n_layers):
            heatmap_data[i, l] = results[task][l]['test'] * 100

    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=100, interpolation='nearest')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(l) for l in range(n_layers)], fontsize=7)
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names, fontsize=11, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_title('Layer x Task Accuracy Heatmap', fontweight='bold')

    # Add text annotations for high values
    for i in range(len(task_names)):
        for j in range(n_layers):
            val = heatmap_data[i, j]
            if val > 0:
                text_color = 'white' if val < 50 else 'black'
                ax.text(j, i, "%.0f" % val, ha='center', va='center',
                        fontsize=6, color=text_color)

    # Mark optimal with star
    for i, task in enumerate(task_names):
        opt_l = optimal[task]['layer']
        ax.plot(opt_l, i, '*', color='gold', markersize=12,
                markeredgecolor='black', markeredgewidth=1)

    plt.colorbar(im, ax=ax, label='Accuracy (%)', shrink=0.8)

    plt.suptitle('Phase 137: Arithmetic Layer Scan\n'
                 '"Each operation has a natural home in the architecture" '
                 '(specialization=%.1f layers)' % specialization_index,
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase137_arithmetic_layer.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_json = {}
    for task in task_names:
        results_json[task] = {}
        for l in range(n_layers):
            results_json[task][str(l)] = results[task][l]

    output = {
        'phase': 137, 'name': 'arithmetic_layer_scan',
        'n_layers': n_layers, 'tasks': task_names,
        'layer_results': results_json,
        'optimal_layers': optimal,
        'specialization_index': round(specialization_index, 2),
        'comparison_mean_layer': round(comp_mean, 1),
        'arithmetic_mean_layer': round(arith_mean, 1),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase137_arithmetic_layer.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
