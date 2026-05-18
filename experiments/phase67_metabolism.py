# -*- coding: utf-8 -*-
"""
Phase 67: Neural Metabolism
Programs consume 'energy' (forward passes). Each has a budget.
Programs that produce useful output earn more energy.
Natural selection emerges from resource competition.

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


def evaluate_usefulness(model, tok, vec, prompts, expected, layer, device):
    """Evaluate how useful a program is (reward = correct answers)."""
    correct = 0
    for prompt, exp in zip(prompts, expected):
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == exp:
            correct += 1
    return correct / len(prompts)


def main():
    print("[P67] Neural Metabolism")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Environment: tasks that provide "food"
    tasks = {
        'MIN': {'prompts': ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) ="],
                'expected': ["3", "2", "1", "4"]},
        'MAX': {'prompts': ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) ="],
                'expected': ["7", "5", "8", "6"]},
    }

    # Initialize population with random programs
    POP_SIZE = 20
    N_ROUNDS = 15
    INITIAL_ENERGY = 5.0
    ENERGY_PER_CORRECT = 2.0
    ENERGY_COST = 1.0  # Cost per round of existence
    REPRODUCE_THRESHOLD = 8.0

    population = []
    for i in range(POP_SIZE):
        vec = torch.randn(hidden_size, device=DEVICE) * 0.1
        population.append({
            'vec': vec,
            'energy': INITIAL_ENERGY,
            'age': 0,
            'births': 0,
            'id': i,
        })

    pop_history = []
    energy_history = []
    alive_history = []

    print(f"  Population: {POP_SIZE}, Rounds: {N_ROUNDS}")
    next_id = POP_SIZE

    for rnd in range(N_ROUNDS):
        # Choose random task this round
        task_name = np.random.choice(list(tasks.keys()))
        task = tasks[task_name]

        # Each organism tries the task
        for org in population:
            reward = evaluate_usefulness(
                model, tok, org['vec'],
                task['prompts'], task['expected'],
                target_layer, DEVICE)
            org['energy'] += reward * ENERGY_PER_CORRECT
            org['energy'] -= ENERGY_COST
            org['age'] += 1

        # Death: remove organisms with no energy
        population = [org for org in population if org['energy'] > 0]

        # Reproduction: organisms with enough energy split
        new_orgs = []
        for org in population:
            if org['energy'] >= REPRODUCE_THRESHOLD:
                # Split energy, mutate vector
                child_vec = org['vec'] + torch.randn_like(org['vec']) * 0.3
                child = {
                    'vec': child_vec,
                    'energy': org['energy'] / 2,
                    'age': 0,
                    'births': 0,
                    'id': next_id,
                }
                next_id += 1
                org['energy'] /= 2
                org['births'] += 1
                new_orgs.append(child)
        population.extend(new_orgs)

        # Record
        avg_energy = np.mean([o['energy'] for o in population]) if population else 0
        pop_history.append(len(population))
        energy_history.append(avg_energy)
        alive_history.append(len(population))

        if rnd % 3 == 0 or rnd == N_ROUNDS - 1:
            # Measure specialization
            min_scores = []
            max_scores = []
            for org in population[:min(5, len(population))]:
                ms = evaluate_usefulness(model, tok, org['vec'],
                    tasks['MIN']['prompts'], tasks['MIN']['expected'], target_layer, DEVICE)
                xs = evaluate_usefulness(model, tok, org['vec'],
                    tasks['MAX']['prompts'], tasks['MAX']['expected'], target_layer, DEVICE)
                min_scores.append(ms)
                max_scores.append(xs)
            print(f"    Round {rnd}: pop={len(population)}, avg_energy={avg_energy:.1f}, "
                  f"task={task_name}, "
                  f"MIN_avg={np.mean(min_scores):.0%}, MAX_avg={np.mean(max_scores):.0%}")

    # Final analysis
    print(f"\n  Final population: {len(population)}")
    survivors = sorted(population, key=lambda x: x['energy'], reverse=True)
    specialization = {}
    for org in survivors[:min(10, len(survivors))]:
        ms = evaluate_usefulness(model, tok, org['vec'],
            tasks['MIN']['prompts'], tasks['MIN']['expected'], target_layer, DEVICE)
        xs = evaluate_usefulness(model, tok, org['vec'],
            tasks['MAX']['prompts'], tasks['MAX']['expected'], target_layer, DEVICE)
        spec = 'MIN' if ms > xs else ('MAX' if xs > ms else 'GENERALIST')
        specialization[org['id']] = {
            'min_score': round(ms, 4), 'max_score': round(xs, 4),
            'energy': round(org['energy'], 2), 'age': org['age'],
            'specialization': spec,
        }

    spec_counts = {}
    for s in specialization.values():
        spec_counts[s['specialization']] = spec_counts.get(s['specialization'], 0) + 1
    print(f"  Specialization: {spec_counts}")

    # Save
    output = {
        'phase': 67, 'name': 'neural_metabolism',
        'initial_pop': POP_SIZE, 'final_pop': len(population),
        'n_rounds': N_ROUNDS,
        'pop_history': pop_history,
        'energy_history': [round(e, 2) for e in energy_history],
        'specialization_counts': spec_counts,
        'top_survivors': specialization,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase67_metabolism.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(pop_history, 'g-', linewidth=2)
    axes[0].set_xlabel('Round'); axes[0].set_ylabel('Population Size')
    axes[0].set_title('Population Dynamics', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(energy_history, 'b-', linewidth=2)
    axes[1].set_xlabel('Round'); axes[1].set_ylabel('Avg Energy')
    axes[1].set_title('Energy Evolution', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    if specialization:
        sp_names = list(spec_counts.keys())
        sp_vals = [spec_counts[n] for n in sp_names]
        axes[2].bar(sp_names, sp_vals, color=['tab:blue', 'tab:red', 'tab:gray'][:len(sp_names)],
                    edgecolor='black')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Survivor Specialization', fontweight='bold')

    plt.suptitle('Phase 67: Neural Metabolism\nNatural selection through energy competition',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase67_metabolism.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
