# -*- coding: utf-8 -*-
"""
Phase 138: Arithmetic Data Scaling
Is ADD/SUB's failure due to insufficient training data?
Test with 5, 10, 20, 40 examples.

"Maybe the soul just needs more memories to learn arithmetic."
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
LAYER = 8


def generate_data(task, n, seed=42):
    """Generate n training examples for a task."""
    rng = random.Random(seed)
    data = []
    seen = set()
    while len(data) < n:
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if task == 'MIN':
            ans = str(min(a, b))
        elif task == 'MAX':
            ans = str(max(a, b))
        elif task == 'ADD':
            s = a + b
            if s > 9:  # keep single digit answers
                continue
            ans = str(s)
        elif task == 'SUB':
            if a <= b:
                continue
            ans = str(a - b)
        data.append(("%d, %d) =" % (a, b), ans))
    return data


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


def evaluate(model, tok, vec, data, device, layer=LAYER):
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
    print("[P138] Arithmetic Data Scaling")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    tasks = ['MIN', 'MAX', 'ADD', 'SUB']
    n_trains = [5, 10, 20, 35]
    # Fixed test set (generated with different seed)
    test_data = {task: generate_data(task, 10, seed=999) for task in tasks}

    results = {task: {} for task in tasks}

    for task in tasks:
        for n in n_trains:
            train_data = generate_data(task, n, seed=42)
            vec = train_soul(model, tok, train_data, DEVICE, epochs=200)
            train_acc = evaluate(model, tok, vec, train_data, DEVICE)
            test_acc = evaluate(model, tok, vec, test_data[task], DEVICE)
            results[task][n] = {'train': round(train_acc, 4), 'test': round(test_acc, 4)}
            print("  %s n=%d: train=%.0f%% test=%.0f%%" % (
                task, n, train_acc * 100, test_acc * 100))

    # Also test at best layers found by P137
    print("\n  Testing ADD at L5, SUB at L0 (P137 optimal layers)...")
    for task, best_layer in [('ADD', 5), ('SUB', 0)]:
        for n in n_trains:
            train_data = generate_data(task, n, seed=42)
            vec = train_soul(model, tok, train_data, DEVICE, layer=best_layer, epochs=200)
            train_acc = evaluate(model, tok, vec, train_data, DEVICE, layer=best_layer)
            test_acc = evaluate(model, tok, vec, test_data[task], DEVICE, layer=best_layer)
            results[task]["%d_bestL" % n] = {
                'train': round(train_acc, 4), 'test': round(test_acc, 4),
                'layer': best_layer}
            print("  %s n=%d @L%d: train=%.0f%% test=%.0f%%" % (
                task, n, best_layer, train_acc * 100, test_acc * 100))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'MIN': '#2196F3', 'MAX': '#FF5722', 'ADD': '#4CAF50', 'SUB': '#9C27B0'}

    # Panel 1: Test accuracy vs data size (L8)
    ax = axes[0]
    for task in tasks:
        accs = [results[task][n]['test'] for n in n_trains]
        ax.plot(n_trains, accs, 'o-', color=colors[task], label=task,
                markersize=8, linewidth=2)
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy vs Data Size (L8)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # Panel 2: Train accuracy vs data size
    ax = axes[1]
    for task in tasks:
        accs = [results[task][n]['train'] for n in n_trains]
        ax.plot(n_trains, accs, 's--', color=colors[task], label=task,
                markersize=8, linewidth=2)
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Train Accuracy')
    ax.set_title('Train Accuracy vs Data Size (L8)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # Panel 3: ADD/SUB at best layer vs L8
    ax = axes[2]
    for task in ['ADD', 'SUB']:
        l8_accs = [results[task][n]['test'] for n in n_trains]
        best_key = ["%d_bestL" % n for n in n_trains]
        best_accs = [results[task][k]['test'] for k in best_key]
        ax.plot(n_trains, l8_accs, 'o--', color=colors[task], alpha=0.5,
                label='%s @L8' % task)
        ax.plot(n_trains, best_accs, 's-', color=colors[task],
                label='%s @best L' % task, markersize=8, linewidth=2)
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('ADD/SUB: L8 vs Best Layer', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    plt.suptitle('Phase 138: Arithmetic Data Scaling\n'
                 '"Maybe the soul just needs more memories"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase138_data_scaling.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 138, 'name': 'arithmetic_data_scaling',
        'default_layer': LAYER,
        'n_trains': n_trains,
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase138_data_scaling.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
