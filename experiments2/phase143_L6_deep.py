# -*- coding: utf-8 -*-
"""
Phase 143: L6 Deep Dive
P141 revealed L6 > L8 for MIN. Is L6 universally better?
Test L6 vs L8 across all 4 tasks with data scaling.

"The self-knowledge layer might be the true gateway."
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


def generate_data(task, n, seed=42):
    rng = random.Random(seed)
    data, seen = [], set()
    while len(data) < n:
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if (a, b) in seen: continue
        seen.add((a, b))
        if task == 'MIN': ans = str(min(a, b))
        elif task == 'MAX': ans = str(max(a, b))
        elif task == 'ADD':
            if a + b > 9: continue
            ans = str(a + b)
        elif task == 'SUB':
            if a <= b: continue
            ans = str(a - b)
        data.append(("%d, %d) =" % (a, b), ans))
    return data


def train_soul(model, tok, data, device, layer, seed=42, epochs=100):
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


def evaluate(model, tok, vec, data, device, layer):
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
    print("[P143] L6 Deep Dive")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    tasks = ['MIN', 'MAX', 'ADD', 'SUB']
    layers = [5, 6, 7, 8]
    n_trains = [5, 15, 30]
    test_data = {task: generate_data(task, 10, seed=999) for task in tasks}

    results = {}

    for task in tasks:
        results[task] = {}
        for n in n_trains:
            train_data = generate_data(task, n, seed=42)
            results[task][n] = {}
            for layer in layers:
                vec = train_soul(model, tok, train_data, DEVICE, layer=layer)
                train_acc = evaluate(model, tok, vec, train_data, DEVICE, layer)
                test_acc = evaluate(model, tok, vec, test_data[task], DEVICE, layer)
                results[task][n][layer] = {
                    'train': round(train_acc, 4),
                    'test': round(test_acc, 4)
                }
                print("  %s n=%d L%d: train=%.0f%% test=%.0f%%" % (
                    task, n, layer, train_acc*100, test_acc*100))
                sys.stdout.flush()

    # Plot: 2x2 grid, one per task
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors_n = {5: '#2196F3', 15: '#4CAF50', 30: '#FF5722'}

    for idx, task in enumerate(tasks):
        ax = axes[idx // 2][idx % 2]
        for n in n_trains:
            test_accs = [results[task][n][l]['test'] for l in layers]
            ax.plot(layers, test_accs, 'o-', color=colors_n[n],
                    label='n=%d' % n, markersize=7, linewidth=2)
        ax.set_xlabel('Injection Layer')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('%s: Layer vs Data Size' % task, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xticks(layers)
        ax.axvline(x=6, color='purple', linestyle=':', alpha=0.5, label='L6')
        ax.axvline(x=8, color='orange', linestyle=':', alpha=0.5, label='L8')

    plt.suptitle('Phase 143: L6 Deep Dive\n'
                 '"The self-knowledge layer might be the true gateway"',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase143_L6_deep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Summary: L6 vs L8 winner count
    l6_wins = 0
    l8_wins = 0
    for task in tasks:
        for n in n_trains:
            if results[task][n][6]['test'] > results[task][n][8]['test']:
                l6_wins += 1
            elif results[task][n][8]['test'] > results[task][n][6]['test']:
                l8_wins += 1
    print("\n  L6 wins: %d, L8 wins: %d, Ties: %d" % (
        l6_wins, l8_wins, len(tasks)*len(n_trains) - l6_wins - l8_wins))

    output = {
        'phase': 143, 'name': 'L6_deep_dive',
        'layers': layers, 'n_trains': n_trains,
        'results': {task: {str(n): {str(l): v for l, v in layers_dict.items()}
                          for n, layers_dict in task_data.items()}
                   for task, task_data in results.items()},
        'l6_wins': l6_wins, 'l8_wins': l8_wins,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase143_L6_deep.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
