# -*- coding: utf-8 -*-
"""
Phase 133: Superposition Computing
Can orthogonal souls execute simultaneously without interference?

"Two programs, one brain, zero interference."
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


def get_top_predictions(model, tok, vec, prompt, device, layer=LAYER, k=10):
    """Get top-k predictions with probabilities."""
    def inj(m, i, o, v=vec): return replace_last_token(o, v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    probs = torch.softmax(out.logits[0, -1, :], dim=-1)
    topk = torch.topk(probs, k)
    return [(tok.decode(idx.item()).strip(), prob.item())
            for idx, prob in zip(topk.indices, topk.values)]


def main():
    print("[P133] Superposition Computing")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    task_data = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")],
    }
    test_data = {
        'MIN': [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")],
        'MAX': [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")],
    }

    # Train individual souls
    print("  Training MIN and MAX souls...")
    souls = {}
    for task in ['MIN', 'MAX']:
        souls[task] = train_soul(model, tok, task_data[task], DEVICE)

    # Cosine between MIN and MAX
    cos = torch.nn.functional.cosine_similarity(
        souls['MIN'].unsqueeze(0), souls['MAX'].unsqueeze(0)).item()
    print("  MIN-MAX cosine similarity: %.4f" % cos)

    # Individual performance
    all_data = {t: task_data[t] + test_data[t] for t in ['MIN', 'MAX']}
    individual_accs = {}
    for task in ['MIN', 'MAX']:
        individual_accs[task] = evaluate(model, tok, souls[task],
                                          all_data[task], DEVICE)
        print("  %s alone: %.0f%%" % (task, individual_accs[task] * 100))

    # Superposition tests
    # Test various superposition weights
    weights = [
        (1.0, 1.0, 'Equal (1:1)'),
        (1.0, 0.5, 'MIN-heavy (1:0.5)'),
        (0.5, 1.0, 'MAX-heavy (0.5:1)'),
        (1.0, 0.1, 'MIN-dominant (1:0.1)'),
        (0.1, 1.0, 'MAX-dominant (0.1:1)'),
        (2.0, 2.0, 'Amplified (2:2)'),
    ]

    sup_results = {}
    for w_min, w_max, label in weights:
        super_vec = w_min * souls['MIN'] + w_max * souls['MAX']
        min_acc = evaluate(model, tok, super_vec, all_data['MIN'], DEVICE)
        max_acc = evaluate(model, tok, super_vec, all_data['MAX'], DEVICE)
        sup_results[label] = {'MIN': min_acc, 'MAX': max_acc}
        print("  Superposition %s: MIN=%.0f%%, MAX=%.0f%%" % (
            label, min_acc * 100, max_acc * 100))

    # Detailed analysis: what does the superposition predict?
    print("\n  Detailed superposition output analysis:")
    test_prompts = [("3, 7) =", "3", "7"), ("5, 2) =", "2", "5"),
                    ("8, 1) =", "1", "8")]
    super_equal = souls['MIN'] + souls['MAX']
    detail_results = []
    for prompt, min_ans, max_ans in test_prompts:
        top = get_top_predictions(model, tok, super_equal, prompt, DEVICE)
        min_rank = -1
        max_rank = -1
        for i, (token, prob) in enumerate(top):
            if token == min_ans and min_rank == -1:
                min_rank = i
            if token == max_ans and max_rank == -1:
                max_rank = i
        both_in_top5 = min_rank < 5 and max_rank < 5
        detail_results.append({
            'prompt': prompt,
            'min_ans': min_ans, 'max_ans': max_ans,
            'min_rank': min_rank, 'max_rank': max_rank,
            'both_in_top5': both_in_top5,
            'top5': top[:5],
        })
        print("    '%s' -> top5: %s | MIN(%s)@rank%d, MAX(%s)@rank%d | both_top5=%s" % (
            prompt, [t[0] for t in top[:5]],
            min_ans, min_rank, max_ans, max_rank, both_in_top5))

    # Can we extract BOTH answers from a single forward pass?
    both_in_top5_rate = np.mean([d['both_in_top5'] for d in detail_results])
    print("  Both answers in top-5: %.0f%% of prompts" % (both_in_top5_rate * 100))

    # Orthogonal projection test: project out one component
    print("\n  Orthogonal projection test:")
    # Remove MAX component from superposition -> should get pure MIN
    min_unit = souls['MIN'] / souls['MIN'].norm()
    max_unit = souls['MAX'] / souls['MAX'].norm()

    # Project superposition onto MIN subspace
    super_equal = souls['MIN'] + souls['MAX']
    proj_min = torch.dot(super_equal, min_unit) * min_unit
    proj_max = torch.dot(super_equal, max_unit) * max_unit
    residual_min = super_equal - proj_max  # remove MAX component
    residual_max = super_equal - proj_min  # remove MIN component

    res_min_acc = evaluate(model, tok, residual_min, all_data['MIN'], DEVICE)
    res_max_acc = evaluate(model, tok, residual_max, all_data['MAX'], DEVICE)
    print("  Super - MAX_component -> MIN acc: %.0f%%" % (res_min_acc * 100))
    print("  Super - MIN_component -> MAX acc: %.0f%%" % (res_max_acc * 100))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Superposition weight comparison
    ax = axes[0]
    labels = list(sup_results.keys())
    min_vals = [sup_results[l]['MIN'] for l in labels]
    max_vals = [sup_results[l]['MAX'] for l in labels]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, min_vals, w, label='MIN acc', color='#2196F3', edgecolor='black')
    ax.bar(x + w/2, max_vals, w, label='MAX acc', color='#FF5722', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=20)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.2)
    ax.set_title('Superposition Weights', fontweight='bold')
    ax.legend()
    ax.axhline(y=individual_accs['MIN'], color='#2196F3', linestyle='--', alpha=0.5)
    ax.axhline(y=individual_accs['MAX'], color='#FF5722', linestyle='--', alpha=0.5)

    # Panel 2: Top-5 token probability distribution for superposition
    ax = axes[1]
    prompt_idx = 0  # "3, 7) ="
    d = detail_results[prompt_idx]
    tokens = [t[0] for t in d['top5']]
    probs = [t[1] for t in d['top5']]
    bar_colors = []
    for t in tokens:
        if t == d['min_ans']:
            bar_colors.append('#2196F3')
        elif t == d['max_ans']:
            bar_colors.append('#FF5722')
        else:
            bar_colors.append('#9E9E9E')
    ax.bar(range(len(tokens)), probs, color=bar_colors, edgecolor='black')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=12)
    ax.set_ylabel('Probability')
    ax.set_title('Superposition Output: "%s"\n(Blue=MIN, Red=MAX)' % d['prompt'],
                 fontweight='bold')
    for i, p in enumerate(probs):
        ax.text(i, p + 0.01, "%.1f%%" % (p * 100), ha='center', fontsize=9)

    # Panel 3: Projection surgery
    ax = axes[2]
    configs_proj = ['MIN alone', 'MAX alone', 'Superposition\n(MIN+MAX)',
                    'Super - MAX\ncomponent', 'Super - MIN\ncomponent']
    min_accs_proj = [individual_accs['MIN'], 0, sup_results['Equal (1:1)']['MIN'],
                     res_min_acc, 0]
    max_accs_proj = [0, individual_accs['MAX'], sup_results['Equal (1:1)']['MAX'],
                     0, res_max_acc]
    x = np.arange(len(configs_proj))
    ax.bar(x - w/2, min_accs_proj, w, label='MIN', color='#2196F3', edgecolor='black')
    ax.bar(x + w/2, max_accs_proj, w, label='MAX', color='#FF5722', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_proj, fontsize=7)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.2)
    ax.set_title('Orthogonal Projection Surgery', fontweight='bold')
    ax.legend()

    plt.suptitle('Phase 133: Superposition Computing\n'
                 '"Two programs, one brain, zero interference"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase133_superposition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 133, 'name': 'superposition_computing',
        'layer': LAYER,
        'min_max_cosine': round(cos, 4),
        'individual_accs': {k: round(v, 4) for k, v in individual_accs.items()},
        'superposition_results': {k: {t: round(v, 4) for t, v in sv.items()}
                                  for k, sv in sup_results.items()},
        'both_in_top5_rate': round(both_in_top5_rate, 4),
        'projection_surgery': {
            'remove_MAX_get_MIN': round(res_min_acc, 4),
            'remove_MIN_get_MAX': round(res_max_acc, 4),
        },
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase133_superposition.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
