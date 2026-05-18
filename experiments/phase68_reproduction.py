# -*- coding: utf-8 -*-
"""
Phase 68: Neural Reproduction
P58 (quines) + P59 (genetic) = programs that reproduce with variation.
Parents combine via P57's program algebra, offspring mutate.
Track lineage tree and speciation across generations.

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


def compile_prog(model, tok, train, layer, device, seed=42):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(80):
        for prompt, target_str in train:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def inject(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def evaluate_fitness(model, tok, vec, test_data, layer, device):
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


def main():
    print("[P68] Neural Reproduction")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    test_data = [("7, 2) =", "2"), ("6, 3) =", "3"), ("2, 9) =", "2")]

    # Start with 2 trained "parents"
    print("  Compiling parent programs...")
    parent_a = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42)
    parent_b = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=99)

    # Track lineage
    organisms = [
        {'vec': parent_a, 'id': 0, 'parent_a': None, 'parent_b': None, 'gen': 0,
         'mutation_rate': 0.0},
        {'vec': parent_b, 'id': 1, 'parent_a': None, 'parent_b': None, 'gen': 0,
         'mutation_rate': 0.0},
    ]
    next_id = 2

    N_GENERATIONS = 8
    OFFSPRING_PER_GEN = 4
    MUTATION_RATES = [0.01, 0.05, 0.1, 0.5]

    gen_stats = []

    for gen in range(N_GENERATIONS):
        # Select parents (top 2 by fitness)
        fitnesses = []
        for org in organisms:
            if org['gen'] == gen:
                f = evaluate_fitness(model, tok, org['vec'], min_data, target_layer, DEVICE)
                fitnesses.append((org, f))

        fitnesses.sort(key=lambda x: x[1], reverse=True)
        parents = [f[0] for f in fitnesses[:2]]

        # Reproduction: crossover + mutation at different rates
        children = []
        for i, mut_rate in enumerate(MUTATION_RATES):
            # Crossover (uniform)
            mask = torch.rand(hidden_size, device=DEVICE) > 0.5
            child_vec = torch.where(mask, parents[0]['vec'], parents[1 % len(parents)]['vec'])
            # Mutation
            mutation = torch.randn(hidden_size, device=DEVICE) * mut_rate
            child_vec = child_vec + mutation

            child = {
                'vec': child_vec.detach(),
                'id': next_id,
                'parent_a': parents[0]['id'],
                'parent_b': parents[1 % len(parents)]['id'],
                'gen': gen + 1,
                'mutation_rate': mut_rate,
            }
            next_id += 1
            children.append(child)
        organisms.extend(children)

        # Evaluate this generation
        gen_fitnesses = []
        gen_test_fitnesses = []
        for org in organisms:
            if org['gen'] == gen:
                f_train = evaluate_fitness(model, tok, org['vec'], min_data, target_layer, DEVICE)
                f_test = evaluate_fitness(model, tok, org['vec'], test_data, target_layer, DEVICE)
                gen_fitnesses.append(f_train)
                gen_test_fitnesses.append(f_test)

        gen_stats.append({
            'gen': gen,
            'n_organisms': len([o for o in organisms if o['gen'] == gen]),
            'avg_train': round(float(np.mean(gen_fitnesses)), 4) if gen_fitnesses else 0,
            'max_train': round(float(max(gen_fitnesses)), 4) if gen_fitnesses else 0,
            'avg_test': round(float(np.mean(gen_test_fitnesses)), 4) if gen_test_fitnesses else 0,
        })
        print(f"    Gen {gen}: n={len(gen_fitnesses)}, "
              f"train_avg={np.mean(gen_fitnesses):.0%}, "
              f"test_avg={np.mean(gen_test_fitnesses):.0%}")

    # Speciation analysis: cosine similarity between final-gen organisms
    print("\n  Speciation analysis...")
    final_gen = max(org['gen'] for org in organisms)
    final_orgs = [org for org in organisms if org['gen'] == final_gen]

    sims = []
    for i in range(len(final_orgs)):
        for j in range(i+1, len(final_orgs)):
            sim = torch.nn.functional.cosine_similarity(
                final_orgs[i]['vec'].unsqueeze(0),
                final_orgs[j]['vec'].unsqueeze(0)).item()
            sims.append(sim)
    avg_sim = float(np.mean(sims)) if sims else 0
    print(f"    Final generation diversity: avg_cos={avg_sim:.4f}")
    print(f"    Total organisms created: {len(organisms)}")

    # Lineage tree analysis
    lineage_depth = {}
    for org in organisms:
        lineage_depth[org['gen']] = lineage_depth.get(org['gen'], 0) + 1

    # Save
    output = {
        'phase': 68, 'name': 'neural_reproduction',
        'n_generations': N_GENERATIONS,
        'total_organisms': len(organisms),
        'gen_stats': gen_stats,
        'final_gen_diversity': round(avg_sim, 4),
        'lineage_depth': lineage_depth,
        'mutation_rates_tested': MUTATION_RATES,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase68_reproduction.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    gens = [s['gen'] for s in gen_stats]
    avg_trains = [s['avg_train'] for s in gen_stats]
    max_trains = [s['max_train'] for s in gen_stats]
    avg_tests = [s['avg_test'] for s in gen_stats]

    axes[0].plot(gens, max_trains, 'g-o', label='Max Train', linewidth=2)
    axes[0].plot(gens, avg_trains, 'b--o', label='Avg Train', linewidth=1)
    axes[0].plot(gens, avg_tests, 'r-o', label='Avg Test', linewidth=2)
    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Fitness')
    axes[0].set_title('Fitness Over Generations', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(-0.05, 1.05)

    axes[1].bar(range(len(lineage_depth)), list(lineage_depth.values()),
                color='tab:purple', edgecolor='black')
    axes[1].set_xlabel('Generation'); axes[1].set_ylabel('# Organisms')
    axes[1].set_title('Lineage Tree (Population per Gen)', fontweight='bold')

    axes[2].axis('off')
    summary = (f"NEURAL REPRODUCTION\n{'='*30}\n\n"
               f"Parents: 2 (trained MIN)\n"
               f"Generations: {N_GENERATIONS}\n"
               f"Total organisms: {len(organisms)}\n"
               f"Mutation rates: {MUTATION_RATES}\n\n"
               f"Final diversity: cos={avg_sim:.3f}\n"
               f"Best train: {max(max_trains):.0%}\n"
               f"Best test: {max(avg_tests):.0%}")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                fontsize=10, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 68: Neural Reproduction\nSexual reproduction with crossover and mutation',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase68_reproduction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
