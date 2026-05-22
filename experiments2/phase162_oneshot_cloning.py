# -*- coding: utf-8 -*-
"""
Phase 162: One-Shot Soul Cloning
Use an existing soul as a warm start and fine-tune with just 1 example.

"I already know MIN. Show me ONE example of MAX, and I'll figure it out."
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
LAYER = 8

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
}


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


def finetune_soul(model, tok, init_vec, data, device, layer=LAYER,
                  epochs=50, lr=0.01):
    """Fine-tune an existing soul vector on new data."""
    vec = init_vec.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=lr)
    history = []
    for ep in range(epochs):
        total_loss = 0
        for p, t in data:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(device)
            def inj(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            total_loss += loss.item()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            history.append({'epoch': ep, 'loss': round(total_loss / len(data), 4)})
    return vec.detach(), history


def evaluate(model, tok, soul_vec, test_data, device, layer=LAYER):
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


def main():
    print("[P162] One-Shot Soul Cloning")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train base MIN soul
    print("  Training base MIN soul...")
    soul_min = train_soul(model, tok, TASK_DATA['MIN']['train'], DEVICE, seed=42)
    min_acc = evaluate(model, tok, soul_min, TASK_DATA['MIN']['test'], DEVICE)
    print("  MIN baseline: %.0f%%" % (min_acc * 100))

    # Experiment: learn MAX from different starting points
    n_examples_list = [1, 2, 3, 5, 10]
    results = {}

    for n_ex in n_examples_list:
        print("\n  === %d-shot learning ===" % n_ex)
        train_subset = TASK_DATA['MAX']['train'][:n_ex]

        # Method 1: Cold start (random init)
        print("  Cold start (random init)...")
        hs = model.config.hidden_size
        torch.manual_seed(99)
        cold_init = torch.randn(hs, device=DEVICE) * 0.01
        cold_soul, cold_hist = finetune_soul(model, tok, cold_init, train_subset,
                                              DEVICE, epochs=50, lr=0.01)
        cold_acc = evaluate(model, tok, cold_soul, TASK_DATA['MAX']['test'], DEVICE)

        # Method 2: Warm start from MIN
        print("  Warm start from MIN soul...")
        warm_soul, warm_hist = finetune_soul(model, tok, soul_min, train_subset,
                                              DEVICE, epochs=50, lr=0.01)
        warm_acc = evaluate(model, tok, warm_soul, TASK_DATA['MAX']['test'], DEVICE)

        # Method 3: Warm start from MIN, low LR
        print("  Warm start, low LR...")
        gentle_soul, gentle_hist = finetune_soul(model, tok, soul_min, train_subset,
                                                  DEVICE, epochs=50, lr=0.003)
        gentle_acc = evaluate(model, tok, gentle_soul, TASK_DATA['MAX']['test'], DEVICE)

        # Full gradient baseline
        full_soul = train_soul(model, tok, train_subset, DEVICE, epochs=100, seed=43)
        full_acc = evaluate(model, tok, full_soul, TASK_DATA['MAX']['test'], DEVICE)

        results[n_ex] = {
            'cold_acc': round(cold_acc, 4),
            'warm_acc': round(warm_acc, 4),
            'gentle_acc': round(gentle_acc, 4),
            'full_acc': round(full_acc, 4),
        }
        print("  Cold=%.0f%%, Warm=%.0f%%, Gentle=%.0f%%, Full=%.0f%%" % (
            cold_acc*100, warm_acc*100, gentle_acc*100, full_acc*100))

    # Also test: MIN -> MAX transfer retains MIN?
    print("\n  === Transfer retention test ===")
    warm_1shot, _ = finetune_soul(model, tok, soul_min,
                                   TASK_DATA['MAX']['train'][:1], DEVICE, epochs=50)
    retained_min = evaluate(model, tok, warm_1shot, TASK_DATA['MIN']['test'], DEVICE)
    learned_max = evaluate(model, tok, warm_1shot, TASK_DATA['MAX']['test'], DEVICE)
    print("  After 1-shot MAX transfer:")
    print("    MIN retention: %.0f%%" % (retained_min * 100))
    print("    MAX learned: %.0f%%" % (learned_max * 100))
    retention = {
        'min_retention': round(retained_min, 4),
        'max_learned': round(learned_max, 4),
    }

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: N-shot learning curves
    ax = axes[0]
    n_exs = sorted(results.keys())
    cold_accs = [results[n]['cold_acc'] for n in n_exs]
    warm_accs = [results[n]['warm_acc'] for n in n_exs]
    gentle_accs = [results[n]['gentle_acc'] for n in n_exs]
    full_accs = [results[n]['full_acc'] for n in n_exs]

    ax.plot(n_exs, cold_accs, 'o--', color='gray', linewidth=2, markersize=8,
            label='Cold start')
    ax.plot(n_exs, warm_accs, 's-', color='#4CAF50', linewidth=2, markersize=8,
            label='Warm (from MIN)')
    ax.plot(n_exs, gentle_accs, 'D-', color='#FF9800', linewidth=2, markersize=8,
            label='Gentle (low LR)')
    ax.plot(n_exs, full_accs, '^-', color='#2196F3', linewidth=2, markersize=8,
            label='Full gradient')
    ax.set_xlabel('Number of MAX examples')
    ax.set_ylabel('MAX Test Accuracy')
    ax.set_title('N-Shot Soul Cloning\n(learning MAX from MIN)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(n_exs)

    # Panel 2: 1-shot comparison
    ax = axes[1]
    one_shot = results[1]
    methods = ['Cold\nStart', 'Warm\n(MIN)', 'Gentle\n(low LR)', 'Full\nGradient']
    accs = [one_shot['cold_acc'], one_shot['warm_acc'],
            one_shot['gentle_acc'], one_shot['full_acc']]
    colors = ['#9E9E9E', '#4CAF50', '#FF9800', '#2196F3']
    bars = ax.bar(methods, accs, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('MAX Test Accuracy')
    ax.set_title('1-Shot Learning Comparison', fontweight='bold')

    # Panel 3: Transfer retention
    ax = axes[2]
    labels = ['MIN\n(original task)', 'MAX\n(new task)']
    vals = [retention['min_retention'], retention['max_learned']]
    colors = ['#E91E63', '#2196F3']
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy')
    ax.set_title('Catastrophic Forgetting Test\n(does learning MAX erase MIN?)',
                 fontweight='bold')

    plt.suptitle('Phase 162: One-Shot Soul Cloning\n'
                 '"Show me ONE example and I\'ll figure it out"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase162_oneshot_cloning.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 162, 'name': 'one_shot_soul_cloning',
        'min_baseline': round(min_acc, 4),
        'nshot_results': {str(k): v for k, v in results.items()},
        'transfer_retention': retention,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase162_oneshot_cloning.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
