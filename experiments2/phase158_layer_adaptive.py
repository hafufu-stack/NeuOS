# -*- coding: utf-8 -*-
"""
Phase 158: Layer-Adaptive Autopoiesis
Fix P154/P157's limitation: ADD/SUB failed at L8 because L6 is optimal
for arithmetic. NeuOS now auto-discovers the best layer per operation.

"Don't just learn WHAT to think -- learn WHERE to think it."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TASK_DATA = {
    'MIN': {
        'train': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                  ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                  ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                  ("1, 3) =","1")],
        'test':  [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                  ("1, 5) =","1"),("8, 4) =","4")],
    },
    'MAX': {
        'train': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                  ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                  ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                  ("1, 3) =","3")],
        'test':  [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                  ("1, 5) =","5"),("8, 4) =","8")],
    },
    'ADD': {
        'train': [("3, 2) =","5"),("4, 1) =","5"),("2, 3) =","5"),
                  ("1, 6) =","7"),("5, 3) =","8"),("2, 7) =","9"),
                  ("3, 4) =","7"),("1, 2) =","3"),("4, 4) =","8"),
                  ("2, 1) =","3")],
        'test':  [("1, 3) =","4"),("2, 5) =","7"),("4, 3) =","7"),
                  ("3, 6) =","9"),("1, 8) =","9")],
    },
    'SUB': {
        'train': [("7, 2) =","5"),("5, 1) =","4"),("9, 3) =","6"),
                  ("8, 5) =","3"),("6, 4) =","2"),("4, 1) =","3"),
                  ("3, 2) =","1"),("9, 7) =","2"),("8, 1) =","7"),
                  ("7, 3) =","4")],
        'test':  [("6, 2) =","4"),("9, 5) =","4"),("8, 3) =","5"),
                  ("5, 4) =","1"),("7, 1) =","6")],
    },
}

CANDIDATE_LAYERS = [4, 6, 8, 10, 12, 16]


def train_soul(model, tok, data, device, layer, epochs=100, seed=42):
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


def evaluate(model, tok, soul_vec, test_data, device, layer):
    correct = 0
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
    return correct / len(test_data) if test_data else 0


def auto_discover_layer(model, tok, train_data, val_data, device,
                        candidate_layers=CANDIDATE_LAYERS, epochs=80):
    """Train a soul at each candidate layer, return the best one."""
    best_layer = None
    best_acc = -1
    best_soul = None
    layer_results = {}

    for layer in candidate_layers:
        soul = train_soul(model, tok, train_data, device, layer,
                         epochs=epochs, seed=42)
        # Validate on the last 3 training examples (held-out)
        acc = evaluate(model, tok, soul, val_data, device, layer)
        layer_results[layer] = {'accuracy': round(acc, 4)}

        if acc > best_acc:
            best_acc = acc
            best_layer = layer
            best_soul = soul

    return best_layer, best_soul, best_acc, layer_results


def main():
    print("[P158] Layer-Adaptive Autopoiesis")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    all_results = {}
    task_best_layers = {}

    for task_name, task in TASK_DATA.items():
        print("\n  === %s ===" % task_name)
        # Split train into train/val
        train_split = task['train'][:7]
        val_split = task['train'][7:]

        # Auto-discover best layer
        print("  Scanning %d candidate layers..." % len(CANDIDATE_LAYERS))
        best_layer, best_soul, val_acc, layer_results = auto_discover_layer(
            model, tok, train_split, val_split, DEVICE)

        # Test on held-out test set
        test_acc = evaluate(model, tok, best_soul, task['test'], DEVICE, best_layer)

        # Also compare with fixed L8
        soul_L8 = train_soul(model, tok, train_split, DEVICE, layer=8, epochs=80)
        test_acc_L8 = evaluate(model, tok, soul_L8, task['test'], DEVICE, layer=8)

        task_best_layers[task_name] = best_layer
        all_results[task_name] = {
            'best_layer': best_layer,
            'val_accuracy': round(val_acc, 4),
            'test_accuracy': round(test_acc, 4),
            'test_accuracy_L8': round(test_acc_L8, 4),
            'improvement': round(test_acc - test_acc_L8, 4),
            'layer_scan': {str(k): v for k, v in layer_results.items()},
        }

        print("  Best layer: L%d (val=%.0f%%, test=%.0f%%)" % (
            best_layer, val_acc * 100, test_acc * 100))
        print("  Fixed L8: test=%.0f%%" % (test_acc_L8 * 100))
        print("  Improvement: %+.0f pp" % ((test_acc - test_acc_L8) * 100))
        for layer, r in sorted(layer_results.items()):
            marker = " <-- BEST" if layer == best_layer else ""
            print("    L%d: val=%.0f%%%s" % (layer, r['accuracy'] * 100, marker))

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    task_names = list(all_results.keys())

    # Panel 1: Fixed L8 vs Adaptive
    ax = axes[0]
    x = np.arange(len(task_names))
    w = 0.35
    fixed_accs = [all_results[t]['test_accuracy_L8'] for t in task_names]
    adaptive_accs = [all_results[t]['test_accuracy'] for t in task_names]
    ax.bar(x - w/2, fixed_accs, w, label='Fixed L8', color='#F44336', edgecolor='black')
    ax.bar(x + w/2, adaptive_accs, w, label='Adaptive Layer', color='#4CAF50', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.set_title('Fixed L8 vs Layer-Adaptive', fontweight='bold')
    for i in range(len(task_names)):
        ax.text(i - w/2, fixed_accs[i] + 0.02, '%.0f%%' % (fixed_accs[i]*100),
                ha='center', fontsize=9)
        ax.text(i + w/2, adaptive_accs[i] + 0.02, '%.0f%%' % (adaptive_accs[i]*100),
                ha='center', fontsize=9)
        # Show best layer
        ax.text(i + w/2, adaptive_accs[i] - 0.08,
                'L%d' % all_results[task_names[i]]['best_layer'],
                ha='center', fontsize=8, color='white', fontweight='bold')

    # Panel 2: Layer scan heatmap
    ax = axes[1]
    heatmap = np.zeros((len(task_names), len(CANDIDATE_LAYERS)))
    for ti, task in enumerate(task_names):
        for li, layer in enumerate(CANDIDATE_LAYERS):
            heatmap[ti, li] = all_results[task]['layer_scan'].get(
                str(layer), {}).get('accuracy', 0)
    im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(CANDIDATE_LAYERS)))
    ax.set_xticklabels(['L%d' % l for l in CANDIDATE_LAYERS])
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_xlabel('Injection Layer')
    ax.set_title('Layer-Task Compatibility\n(accuracy heatmap)', fontweight='bold')
    for ti in range(len(task_names)):
        for li in range(len(CANDIDATE_LAYERS)):
            val = heatmap[ti, li]
            ax.text(li, ti, '%.0f%%' % (val*100), ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if val < 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Accuracy')

    # Panel 3: Best layer per task
    ax = axes[2]
    best_layers = [all_results[t]['best_layer'] for t in task_names]
    colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800']
    bars = ax.bar(task_names, best_layers, color=colors, edgecolor='black', linewidth=1.5)
    for bar, layer, acc in zip(bars, best_layers, adaptive_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                'L%d\n(%.0f%%)' % (layer, acc*100),
                ha='center', fontweight='bold', fontsize=10)
    ax.set_ylabel('Best Layer Index')
    ax.set_title('Auto-Discovered Optimal Layer\nper Operation', fontweight='bold')
    ax.set_ylim(0, max(best_layers) + 4)

    plt.suptitle('Phase 158: Layer-Adaptive Autopoiesis\n'
                 '"Don\'t just learn WHAT -- learn WHERE to think"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase158_layer_adaptive.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 158, 'name': 'layer_adaptive_autopoiesis',
        'task_best_layers': task_best_layers,
        'results': all_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase158_layer_adaptive.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
