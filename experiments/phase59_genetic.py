# -*- coding: utf-8 -*-
"""
Phase 59: Neural Genetic Algorithm
Use P51's polymorphic variants as a gene pool.
Apply selection, crossover, mutation to evolve programs
WITHOUT gradient descent. Pure evolution in activation space.

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
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_fitness(model, tok, vec, test_data, layer, device):
    """Evaluate how well vec implements the target function."""
    correct = 0
    for prompt, expected in test_data:
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data)


def crossover(parent_a, parent_b):
    """Uniform crossover: randomly pick dimensions from each parent."""
    mask = torch.rand_like(parent_a) > 0.5
    child = torch.where(mask, parent_a, parent_b)
    return child


def mutate(vec, rate=0.1, scale=0.5):
    """Gaussian mutation on random dimensions."""
    mask = torch.rand_like(vec) < rate
    noise = torch.randn_like(vec) * scale
    return vec + mask * noise


def main():
    print("[P59] Neural Genetic Algorithm")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Target: MIN function
    train_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                  ("4, 6) =", "4"), ("9, 3) =", "3")]
    test_data = [("7, 2) =", "2"), ("6, 3) =", "3"), ("2, 9) =", "2"),
                 ("5, 4) =", "4"), ("3, 8) =", "3"), ("7, 1) =", "1")]

    # Initialize random population
    POP_SIZE = 20
    N_GENERATIONS = 30
    population = [torch.randn(hidden_size, device=DEVICE) * 0.1 for _ in range(POP_SIZE)]

    best_history = []
    avg_history = []

    print(f"  Population: {POP_SIZE}, Generations: {N_GENERATIONS}")
    for gen in range(N_GENERATIONS):
        # Evaluate fitness
        fitnesses = [evaluate_fitness(model, tok, v, train_data, target_layer, DEVICE)
                      for v in population]
        best_fit = max(fitnesses)
        avg_fit = np.mean(fitnesses)
        best_history.append(best_fit)
        avg_history.append(avg_fit)

        if gen % 5 == 0:
            print(f"    Gen {gen}: best={best_fit:.0%}, avg={avg_fit:.0%}")

        # Selection: tournament
        new_pop = []
        sorted_indices = sorted(range(POP_SIZE), key=lambda i: fitnesses[i], reverse=True)
        # Elitism: keep top 2
        new_pop.append(population[sorted_indices[0]].clone())
        new_pop.append(population[sorted_indices[1]].clone())

        while len(new_pop) < POP_SIZE:
            # Tournament selection (k=3)
            t1, t2, t3 = np.random.choice(POP_SIZE, 3, replace=False)
            parent_a = population[max([t1, t2, t3], key=lambda i: fitnesses[i])]
            t1, t2, t3 = np.random.choice(POP_SIZE, 3, replace=False)
            parent_b = population[max([t1, t2, t3], key=lambda i: fitnesses[i])]
            child = crossover(parent_a, parent_b)
            child = mutate(child, rate=0.1, scale=0.3)
            new_pop.append(child)

        population = new_pop

    # Final evaluation
    final_fitnesses = [evaluate_fitness(model, tok, v, train_data, target_layer, DEVICE)
                        for v in population]
    best_idx = max(range(POP_SIZE), key=lambda i: final_fitnesses[i])
    best_vec = population[best_idx]
    best_train = final_fitnesses[best_idx]

    # Test on unseen data
    best_test = evaluate_fitness(model, tok, best_vec, test_data, target_layer, DEVICE)
    print(f"\n  Best evolved program:")
    print(f"    Train: {best_train:.0%}")
    print(f"    Test:  {best_test:.0%}")

    # Compare with gradient descent (P35 baseline)
    print("  Compiling gradient baseline for comparison...")
    grad_vec = torch.randn(hidden_size, device=DEVICE) * 0.01
    grad_vec.requires_grad_(True)
    opt = torch.optim.Adam([grad_vec], lr=0.01)
    for epoch in range(100):
        for prompt, target_str in train_data:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject_g(module, input, output, v=grad_vec):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_g)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    grad_eval = grad_vec.detach()
    grad_train = evaluate_fitness(model, tok, grad_eval, train_data, target_layer, DEVICE)
    grad_test = evaluate_fitness(model, tok, grad_eval, test_data, target_layer, DEVICE)
    print(f"  Gradient descent: Train={grad_train:.0%}, Test={grad_test:.0%}")

    # Save
    output = {
        'phase': 59, 'name': 'neural_genetic_algorithm',
        'pop_size': POP_SIZE, 'n_generations': N_GENERATIONS,
        'evolved_train': round(best_train, 4),
        'evolved_test': round(best_test, 4),
        'gradient_train': round(grad_train, 4),
        'gradient_test': round(grad_test, 4),
        'best_history': [round(b, 4) for b in best_history],
        'avg_history': [round(a, 4) for a in avg_history],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase59_genetic.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(best_history, 'g-', linewidth=2, label='Best')
    axes[0].plot(avg_history, 'b--', linewidth=1, label='Average')
    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Fitness')
    axes[0].set_title('Evolutionary Progress', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(-0.05, 1.05)

    axes[1].bar(['Evolved\n(Train)', 'Evolved\n(Test)', 'Gradient\n(Train)', 'Gradient\n(Test)'],
                [best_train, best_test, grad_train, grad_test],
                color=['tab:green', 'limegreen', 'tab:blue', 'cornflowerblue'],
                edgecolor='black')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Evolution vs Gradient Descent', fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate([best_train, best_test, grad_train, grad_test]):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    plt.suptitle('Phase 59: Neural Genetic Algorithm\nEvolving programs without gradients',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase59_genetic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
