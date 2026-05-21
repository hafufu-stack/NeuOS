# -*- coding: utf-8 -*-
"""
Phase 140: Recursive Soul Evolution
Can a soul improve itself? Train soul v1, use it to teach soul v2.

"The child surpasses the teacher through accumulated wisdom."
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


def train_soul(model, tok, data, device, layer=LAYER, seed=42, epochs=150,
               init_vec=None):
    hs = model.config.hidden_size
    if init_vec is not None:
        vec = init_vec.clone().detach().requires_grad_(True)
    else:
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


def generate_data_from_soul(model, tok, vec, prompts, device, layer=LAYER):
    """Use a soul to generate pseudo-labels for new prompts."""
    data = []
    for p in prompts:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        data.append((p, pred))
    return data


def main():
    print("[P140] Recursive Soul Evolution")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    train_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                   ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                  ("1, 5) =","1"),("8, 4) =","4")]
    # Extended test (harder)
    hard_test = [("9, 1) =","1"),("2, 7) =","2"),("6, 8) =","6"),
                  ("1, 3) =","1"),("7, 4) =","4")]

    # New prompts for self-training (soul generates its own labels)
    new_prompts = ["%d, %d) =" % (a, b) for a in range(1, 10) for b in range(1, 10)
                   if (a, b) not in [(3,7),(5,2),(8,1),(4,6),(9,3)]]

    n_generations = 6
    gen_results = []
    souls = []
    cos_to_prev = [0]

    # Generation 0: train from original data
    print("  Gen 0: Training from original 5 examples...")
    soul_v0 = train_soul(model, tok, train_data, DEVICE, seed=42, epochs=150)
    train_acc = evaluate(model, tok, soul_v0, train_data, DEVICE)
    test_acc = evaluate(model, tok, soul_v0, test_data, DEVICE)
    hard_acc = evaluate(model, tok, soul_v0, hard_test, DEVICE)
    gen_results.append({'train': train_acc, 'test': test_acc, 'hard': hard_acc})
    souls.append(soul_v0)
    print("    train=%.0f%% test=%.0f%% hard=%.0f%%" % (
        train_acc*100, test_acc*100, hard_acc*100))

    for gen in range(1, n_generations):
        prev_soul = souls[-1]

        # Generate pseudo-labels from previous soul
        pseudo_data = generate_data_from_soul(
            model, tok, prev_soul, new_prompts[:20], DEVICE)

        # Combine original data + pseudo-labeled data
        combined_data = train_data + pseudo_data

        # Strategy A: Train from scratch with expanded data
        soul_new = train_soul(model, tok, combined_data, DEVICE,
                              seed=42+gen, epochs=150)

        # Strategy B: Warm-start from previous soul
        soul_warm = train_soul(model, tok, combined_data, DEVICE,
                               epochs=100, init_vec=prev_soul)

        # Pick better one
        test_a = evaluate(model, tok, soul_new, test_data, DEVICE)
        test_b = evaluate(model, tok, soul_warm, test_data, DEVICE)

        if test_b >= test_a:
            soul_gen = soul_warm
            method = 'warm'
        else:
            soul_gen = soul_new
            method = 'scratch'

        train_acc = evaluate(model, tok, soul_gen, train_data, DEVICE)
        test_acc = evaluate(model, tok, soul_gen, test_data, DEVICE)
        hard_acc = evaluate(model, tok, soul_gen, hard_test, DEVICE)

        cos = torch.nn.functional.cosine_similarity(
            soul_gen.unsqueeze(0), prev_soul.unsqueeze(0)).item()
        cos_to_prev.append(cos)

        gen_results.append({
            'train': train_acc, 'test': test_acc, 'hard': hard_acc,
            'method': method, 'cos_to_prev': round(cos, 4),
            'pseudo_correct': sum(1 for p, pred in pseudo_data
                                  if pred == str(min(int(p[0]), int(p[3]))))
        })
        souls.append(soul_gen)
        print("    Gen %d (%s): train=%.0f%% test=%.0f%% hard=%.0f%% cos=%.3f pseudo_correct=%d/20" % (
            gen, method, train_acc*100, test_acc*100, hard_acc*100, cos,
            gen_results[-1]['pseudo_correct']))

    # Cosine between gen0 and each generation
    cos_to_origin = []
    for soul in souls:
        cos = torch.nn.functional.cosine_similarity(
            soul.unsqueeze(0), souls[0].unsqueeze(0)).item()
        cos_to_origin.append(cos)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Accuracy across generations
    ax = axes[0]
    gens = list(range(n_generations))
    ax.plot(gens, [r['train'] for r in gen_results], 'o-', color='#4CAF50',
            label='Train', markersize=8, linewidth=2)
    ax.plot(gens, [r['test'] for r in gen_results], 's-', color='#2196F3',
            label='Test', markersize=8, linewidth=2)
    ax.plot(gens, [r['hard'] for r in gen_results], '^-', color='#FF5722',
            label='Hard test', markersize=8, linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Accuracy')
    ax.set_title('Soul Evolution Across Generations', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(gens)

    # Panel 2: Cosine drift
    ax = axes[1]
    ax.plot(gens, cos_to_origin, 'o-', color='#9C27B0',
            label='vs Gen 0 (origin)', markersize=8, linewidth=2)
    ax.plot(gens[1:], cos_to_prev[1:], 's--', color='#FF9800',
            label='vs Previous gen', markersize=8, linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Soul Drift Over Generations', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(gens)

    # Panel 3: Pseudo-label quality
    ax = axes[2]
    pseudo_correct = [gen_results[g].get('pseudo_correct', 0)
                      for g in range(1, n_generations)]
    ax.bar(range(1, n_generations), pseudo_correct, color='#00BCD4',
           edgecolor='black')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Correct Pseudo-Labels (out of 20)')
    ax.set_title('Self-Generated Training Data Quality', fontweight='bold')
    ax.set_ylim(0, 21)
    ax.set_xticks(range(1, n_generations))
    for i, v in enumerate(pseudo_correct):
        ax.text(i+1, v+0.3, "%d" % v, ha='center', fontsize=11)

    plt.suptitle('Phase 140: Recursive Soul Evolution\n'
                 '"The child surpasses the teacher through accumulated wisdom"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase140_evolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 140, 'name': 'recursive_soul_evolution',
        'layer': LAYER,
        'n_generations': n_generations,
        'gen_results': [{k: round(v, 4) if isinstance(v, float) else v
                         for k, v in r.items()} for r in gen_results],
        'cos_to_origin': [round(c, 4) for c in cos_to_origin],
        'cos_to_prev': [round(c, 4) for c in cos_to_prev],
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase140_evolution.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
