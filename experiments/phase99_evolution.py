# -*- coding: utf-8 -*-
"""
Phase 99: Darwinian Soul Evolution
Can evolution discover program vectors WITHOUT gradient descent?
Use a genetic algorithm: random population -> fitness evaluation ->
selection -> crossover -> mutation -> repeat.

"Natural selection is the blind watchmaker."

Model: Qwen2.5-0.5B (GPU)
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

def evaluate_vec(model, tok, vec, data, layer, device):
    c = 0
    for p, e in data:
        def inj(m,i,o,v=vec): return replace_last_token(o,v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
    return c / len(data)

def compile_prog(model, tok, train, layer, device, seed=42, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device)*0.01; vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(device)
            def inj(m,i,o,v=vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

def crossover(parent1, parent2):
    """Uniform crossover in vector space."""
    mask = torch.rand_like(parent1) > 0.5
    child = torch.where(mask, parent1, parent2)
    return child

def mutate(vec, rate=0.1, scale=0.5):
    """Gaussian mutation."""
    noise = torch.randn_like(vec) * scale
    mask = torch.rand_like(vec) < rate
    return vec + noise * mask.float()

def main():
    print("[P99] Darwinian Soul Evolution")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    train_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                  ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("5, 4) =","4"),("3, 8) =","3")]
    all_data = train_data + test_data

    # Gradient-trained baseline
    print("  Gradient-trained baseline...")
    grad_vec = compile_prog(model, tok, train_data, tl, DEVICE, seed=42, epochs=100)
    grad_acc = evaluate_vec(model, tok, grad_vec, all_data, tl, DEVICE)
    print(f"    Gradient baseline: {grad_acc:.0%}")

    # Genetic Algorithm
    POP_SIZE = 30
    N_GENERATIONS = 50
    ELITE_SIZE = 5
    MUTATION_RATE = 0.15
    MUTATION_SCALE = 0.3

    print(f"\n  Evolution: pop={POP_SIZE}, gen={N_GENERATIONS}")

    # Initialize random population
    torch.manual_seed(42)
    population = [torch.randn(hs, device=DEVICE) * 0.5 for _ in range(POP_SIZE)]

    history = []
    best_ever_acc = 0.0
    best_ever_vec = None

    for gen in range(N_GENERATIONS):
        # Evaluate fitness
        fitnesses = []
        for ind in population:
            acc = evaluate_vec(model, tok, ind, train_data, tl, DEVICE)
            fitnesses.append(float(acc))

        # Track stats
        best_idx = np.argmax(fitnesses)
        best_acc = fitnesses[best_idx]
        avg_acc = float(np.mean(fitnesses))
        test_acc = evaluate_vec(model, tok, population[best_idx], test_data, tl, DEVICE)

        if best_acc > best_ever_acc:
            best_ever_acc = best_acc
            best_ever_vec = population[best_idx].clone()

        history.append({
            'gen': gen,
            'best_train': round(float(best_acc), 4),
            'avg_train': round(float(avg_acc), 4),
            'best_test': round(float(test_acc), 4),
        })

        if (gen+1) % 10 == 0:
            print(f"    Gen {gen+1}: best_train={best_acc:.0%}, "
                  f"avg={avg_acc:.0%}, test={test_acc:.0%}")

        # Selection: tournament
        sorted_idx = np.argsort(fitnesses)[::-1]
        elites = [population[i].clone() for i in sorted_idx[:ELITE_SIZE]]

        # Generate next generation
        new_pop = list(elites)  # Keep elites
        while len(new_pop) < POP_SIZE:
            # Tournament selection
            idx1, idx2 = np.random.choice(POP_SIZE, 2, replace=False)
            p1 = population[idx1] if fitnesses[idx1] > fitnesses[idx2] else population[idx2]
            idx3, idx4 = np.random.choice(POP_SIZE, 2, replace=False)
            p2 = population[idx3] if fitnesses[idx3] > fitnesses[idx4] else population[idx4]
            child = crossover(p1, p2)
            child = mutate(child, MUTATION_RATE, MUTATION_SCALE)
            new_pop.append(child)

        population = new_pop

    # Final evaluation of best evolved vector
    if best_ever_vec is not None:
        final_train = evaluate_vec(model, tok, best_ever_vec, train_data, tl, DEVICE)
        final_test = evaluate_vec(model, tok, best_ever_vec, test_data, tl, DEVICE)
        final_all = evaluate_vec(model, tok, best_ever_vec, all_data, tl, DEVICE)
    else:
        final_train = final_test = final_all = 0

    print(f"\n  Final evolved: train={final_train:.0%}, test={final_test:.0%}, "
          f"all={final_all:.0%}")
    print(f"  Gradient baseline: {grad_acc:.0%}")

    # Cosine similarity: evolved vs gradient
    if best_ever_vec is not None:
        cos_sim = float(torch.nn.functional.cosine_similarity(
            best_ever_vec.unsqueeze(0), grad_vec.unsqueeze(0)).item())
        print(f"  Cosine(evolved, gradient): {cos_sim:.4f}")
    else:
        cos_sim = 0

    # Save
    output = {
        'phase': 99, 'name': 'darwinian_evolution',
        'gradient_baseline': round(float(grad_acc), 4),
        'evolved_train': round(float(final_train), 4),
        'evolved_test': round(float(final_test), 4),
        'evolved_all': round(float(final_all), 4),
        'cosine_evolved_gradient': round(cos_sim, 4),
        'pop_size': POP_SIZE, 'generations': N_GENERATIONS,
        'history': history,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase99_evolution.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gens = [h['gen'] for h in history]
    axes[0].plot(gens, [h['best_train'] for h in history], 'g-', lw=2, label='Best (train)')
    axes[0].plot(gens, [h['avg_train'] for h in history], 'b-', lw=1.5, alpha=0.7,
                 label='Avg (train)')
    axes[0].plot(gens, [h['best_test'] for h in history], 'r--', lw=2, label='Best (test)')
    axes[0].axhline(y=grad_acc, color='gold', ls='--', lw=2, label='Gradient baseline')
    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Evolutionary Learning Curve', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.1)

    # Comparison bar
    labels = ['Gradient\n(100 ep)', 'Evolved\n(50 gen)', 'Random\nInit']
    vals = [grad_acc, final_all, history[0]['avg_train'] if history else 0]
    colors = ['tab:blue', 'tab:green', 'tab:gray']
    axes[1].bar(labels, vals, color=colors, edgecolor='black')
    axes[1].set_ylabel('Accuracy (all data)')
    axes[1].set_title('Gradient vs Evolution', fontweight='bold')
    axes[1].set_ylim(0, 1.2)
    for i, v in enumerate(vals):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=12)

    # Population diversity over time
    axes[2].fill_between(gens,
                         [h['avg_train'] for h in history],
                         [h['best_train'] for h in history],
                         alpha=0.3, color='tab:green', label='Diversity band')
    axes[2].plot(gens, [h['best_train'] for h in history], 'g-', lw=2)
    axes[2].plot(gens, [h['avg_train'] for h in history], 'b-', lw=1.5)
    axes[2].set_xlabel('Generation'); axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Population Diversity', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 99: Darwinian Soul Evolution\n'
                 '"Natural selection is the blind watchmaker"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase99_evolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
