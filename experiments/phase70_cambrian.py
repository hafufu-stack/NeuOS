# -*- coding: utf-8 -*-
"""
Phase 70: Cambrian Explosion
Start with a single random seed program.
Apply evolution, reproduction, competition, and ecological pressure.
Watch as new functions spontaneously emerge from nothing.
How many distinct functions can evolve from pure noise?

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_behavior_signature(model, tok, vec, layer, device):
    """Get a program's behavioral signature across standard inputs."""
    prompts = ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) =", "9, 3) ="]
    preds = []
    for prompt in prompts:
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        # Sanitize to ASCII for cp932 safety
        pred = pred.encode('ascii', errors='replace').decode('ascii')
        preds.append(pred)
    return tuple(preds)


# Known function signatures for classification
KNOWN_FUNCTIONS = {
    ('3', '2', '1', '4', '3'): 'MIN',
    ('7', '5', '8', '6', '9'): 'MAX',
    ('3', '5', '8', '4', '9'): 'FIRST',
    ('7', '2', '1', '6', '3'): 'SECOND',
    ('1', '7', '9', '6', '8'): 'SUM_MOD10',
}


def classify_function(sig):
    """Classify a behavioral signature."""
    if sig in KNOWN_FUNCTIONS:
        return KNOWN_FUNCTIONS[sig]
    # Check partial matches
    min_sig = ('3', '2', '1', '4', '3')
    max_sig = ('7', '5', '8', '6', '9')
    min_match = sum(a == b for a, b in zip(sig, min_sig))
    max_match = sum(a == b for a, b in zip(sig, max_sig))
    if min_match >= 4:
        return 'NEAR_MIN'
    if max_match >= 4:
        return 'NEAR_MAX'
    # Check if outputs are all numbers
    try:
        nums = [int(s) for s in sig]
        return f'NUM_{"-".join(sig)}'
    except ValueError:
        return f'OTHER_{"-".join(str(s)[:3] for s in sig)}'


def main():
    print("[P70] Cambrian Explosion")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Start from a SINGLE random seed
    print("  Seeding primordial soup with random noise...")
    POP_SIZE = 30
    N_GENERATIONS = 20
    MUTATION_RATE = 0.3
    ELITE_KEEP = 5

    population = [torch.randn(hidden_size, device=DEVICE) * 0.1 for _ in range(POP_SIZE)]

    # Track species diversity over time
    species_history = []
    unique_functions_history = []

    for gen in range(N_GENERATIONS):
        # Get behavioral signatures
        signatures = []
        for vec in population:
            sig = get_behavior_signature(model, tok, vec, target_layer, DEVICE)
            signatures.append(sig)

        # Classify
        species = [classify_function(sig) for sig in signatures]
        species_counts = Counter(species)
        unique_functions = len(set(signatures))
        species_history.append(dict(species_counts))
        unique_functions_history.append(unique_functions)

        if gen % 4 == 0 or gen == N_GENERATIONS - 1:
            print(f"    Gen {gen}: {len(species_counts)} species, "
                  f"{unique_functions} unique functions")
            for sp, cnt in species_counts.most_common(3):
                safe_sp = sp.encode('ascii', errors='replace').decode('ascii')
                print(f"      {safe_sp}: {cnt}")

        # Fitness: reward consistency and known function matching
        fitnesses = []
        for sig in signatures:
            func = classify_function(sig)
            if func in ['MIN', 'MAX', 'FIRST', 'SECOND']:
                fitness = 1.0  # Known function = max fitness
            elif func.startswith('NEAR_'):
                fitness = 0.7  # Close to known function
            elif func.startswith('NUM_'):
                fitness = 0.4  # At least produces numbers
            else:
                fitness = 0.1  # Random non-numeric output
            fitnesses.append(fitness)

        # Selection + Reproduction
        sorted_indices = sorted(range(len(population)),
                              key=lambda i: fitnesses[i], reverse=True)
        new_pop = [population[i].clone() for i in sorted_indices[:ELITE_KEEP]]

        while len(new_pop) < POP_SIZE:
            # Tournament selection
            t = np.random.choice(len(population), 3, replace=False)
            parent = population[max(t, key=lambda i: fitnesses[i])]
            # Mutation
            child = parent + torch.randn_like(parent) * MUTATION_RATE
            new_pop.append(child)

        population = new_pop

    # Final census
    print("\n  Final Cambrian Census:")
    final_sigs = []
    final_species = []
    for vec in population:
        sig = get_behavior_signature(model, tok, vec, target_layer, DEVICE)
        final_sigs.append(sig)
        final_species.append(classify_function(sig))

    final_counts = Counter(final_species)
    for sp, cnt in final_counts.most_common():
        safe_sp = sp.encode('ascii', errors='replace').decode('ascii')
        print(f"    {safe_sp}: {cnt} ({cnt/len(population):.0%})")

    unique_final = len(set(final_sigs))
    print(f"\n  Unique behavioral phenotypes: {unique_final}")
    print(f"  Named species: {len(final_counts)}")

    # Did any known function emerge from PURE NOISE?
    emerged_known = [sp for sp in final_counts if sp in ['MIN', 'MAX', 'FIRST', 'SECOND']]
    print(f"  Known functions emerged: {emerged_known if emerged_known else 'None'}")

    # Save
    output = {
        'phase': 70, 'name': 'cambrian_explosion',
        'pop_size': POP_SIZE, 'n_generations': N_GENERATIONS,
        'final_species': dict(final_counts),
        'unique_phenotypes': unique_final,
        'emerged_known': emerged_known,
        'species_history': species_history,
        'unique_functions_history': unique_functions_history,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase70_cambrian.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(unique_functions_history, 'g-o', linewidth=2)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Unique Functions')
    axes[0].set_title('Functional Diversity Over Time', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    sp_names = [s for s, _ in final_counts.most_common()]
    sp_vals = [c for _, c in final_counts.most_common()]
    colors_sp = []
    for s in sp_names:
        if s == 'MIN': colors_sp.append('tab:blue')
        elif s == 'MAX': colors_sp.append('tab:red')
        elif s.startswith('NEAR'): colors_sp.append('tab:orange')
        elif s.startswith('NUM'): colors_sp.append('tab:green')
        else: colors_sp.append('tab:gray')
    axes[1].barh(range(len(sp_names)), sp_vals, color=colors_sp, edgecolor='black')
    axes[1].set_yticks(range(len(sp_names)))
    axes[1].set_yticklabels(sp_names, fontsize=8)
    axes[1].set_xlabel('Count')
    axes[1].set_title('Final Species Distribution', fontweight='bold')

    # Diversity over time (number of species)
    n_species = [len(sh) for sh in species_history]
    axes[2].plot(n_species, 'r-o', linewidth=2)
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('# Species')
    axes[2].set_title('Species Count Over Time', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 70: Cambrian Explosion\nSpontaneous function emergence from random noise',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase70_cambrian.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
