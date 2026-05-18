# -*- coding: utf-8 -*-
"""
Phase 61: Neural Ecosystem
Run MIN, MAX, SUM as co-existing "species" in shared activation space.
Each species competes for execution. Does a stable ecosystem emerge?

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cos_sim_np
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_species(model, tok, train_data, layer, device, seed):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(80):
        for prompt, target_str in train_data:
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


def evaluate_vec(model, tok, vec, test_data, layer, device):
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
    print("[P61] Neural Ecosystem")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Define species
    species_data = {
        'MIN': [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")],
        'MAX': [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")],
        'SUM': [("3, 7) =", "1"), ("5, 2) =", "7"), ("1, 8) =", "9"),
                ("4, 2) =", "6"), ("3, 5) =", "8")],
    }

    # Compile each species (3 individuals per species)
    print("  Compiling species...")
    ecosystem = {}
    for name, data in species_data.items():
        individuals = []
        for i in range(3):
            v = compile_species(model, tok, data, target_layer, DEVICE, seed=hash(name) % 1000 + i * 100)
            acc = evaluate_vec(model, tok, v, data, target_layer, DEVICE)
            individuals.append({'vec': v, 'fitness': acc})
        ecosystem[name] = individuals
        print(f"    {name}: fitnesses = {[ind['fitness'] for ind in individuals]}")

    # Competition: mix all individuals, test cross-species interference
    print("\n  Cross-species interaction test...")
    cross_results = {}
    for sp_a in species_data:
        for sp_b in species_data:
            if sp_a >= sp_b:
                continue
            # Mix: average the two species' best vectors
            vec_a = ecosystem[sp_a][0]['vec']
            vec_b = ecosystem[sp_b][0]['vec']
            mixed = 0.5 * vec_a + 0.5 * vec_b

            acc_a = evaluate_vec(model, tok, mixed, species_data[sp_a], target_layer, DEVICE)
            acc_b = evaluate_vec(model, tok, mixed, species_data[sp_b], target_layer, DEVICE)
            cross_results[f'{sp_a}+{sp_b}'] = {
                f'{sp_a}_acc': round(acc_a, 4),
                f'{sp_b}_acc': round(acc_b, 4),
            }
            print(f"    {sp_a}+{sp_b}: {sp_a}={acc_a:.0%}, {sp_b}={acc_b:.0%}")

    # Niche separation test: pairwise cosine between species
    print("\n  Niche separation (cosine similarity)...")
    species_vecs = {}
    for name in ecosystem:
        species_vecs[name] = ecosystem[name][0]['vec'].cpu().numpy().flatten()

    niche_sims = {}
    names = list(species_vecs.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            sim = cos_sim_np(species_vecs[names[i]].reshape(1,-1),
                           species_vecs[names[j]].reshape(1,-1))[0,0]
            key = f'{names[i]}-{names[j]}'
            niche_sims[key] = round(float(sim), 4)
            print(f"    {key}: cos={sim:.4f}")

    # Ecosystem stability: weighted sum with varying dominance
    print("\n  Ecosystem stability test...")
    stability = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # MIN vs MAX at different mixing ratios
        mixed = alpha * ecosystem['MIN'][0]['vec'] + (1-alpha) * ecosystem['MAX'][0]['vec']
        min_acc = evaluate_vec(model, tok, mixed, species_data['MIN'], target_layer, DEVICE)
        max_acc = evaluate_vec(model, tok, mixed, species_data['MAX'], target_layer, DEVICE)
        stability[str(alpha)] = {'MIN': round(min_acc, 4), 'MAX': round(max_acc, 4)}

    # Save
    output = {
        'phase': 61, 'name': 'neural_ecosystem',
        'species_fitness': {name: [ind['fitness'] for ind in inds]
                           for name, inds in ecosystem.items()},
        'cross_results': cross_results,
        'niche_sims': niche_sims,
        'stability': stability,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase61_ecosystem.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    species_names = list(ecosystem.keys())
    best_fits = [max(ind['fitness'] for ind in ecosystem[n]) for n in species_names]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    axes[0].bar(species_names, best_fits, color=colors, edgecolor='black')
    axes[0].set_ylabel('Best Fitness')
    axes[0].set_title('Species Fitness', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(best_fits):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Niche separation heatmap
    n_sp = len(species_names)
    sim_mat = np.eye(n_sp)
    for i in range(n_sp):
        for j in range(i+1, n_sp):
            key = f'{species_names[i]}-{species_names[j]}'
            if key in niche_sims:
                sim_mat[i, j] = niche_sims[key]
                sim_mat[j, i] = niche_sims[key]
    im = axes[1].imshow(sim_mat, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    axes[1].set_xticks(range(n_sp)); axes[1].set_xticklabels(species_names)
    axes[1].set_yticks(range(n_sp)); axes[1].set_yticklabels(species_names)
    axes[1].set_title('Niche Separation', fontweight='bold')
    plt.colorbar(im, ax=axes[1])
    for i in range(n_sp):
        for j in range(n_sp):
            axes[1].text(j, i, f'{sim_mat[i,j]:.2f}', ha='center', va='center', fontsize=10)

    # Stability
    alphas = sorted(stability.keys(), key=float)
    min_accs = [stability[a]['MIN'] for a in alphas]
    max_accs = [stability[a]['MAX'] for a in alphas]
    axes[2].plot([float(a) for a in alphas], min_accs, 'b-o', label='MIN', linewidth=2)
    axes[2].plot([float(a) for a in alphas], max_accs, 'r-o', label='MAX', linewidth=2)
    axes[2].set_xlabel('alpha (MIN weight)')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Ecosystem Stability\nalpha*MIN + (1-alpha)*MAX', fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 61: Neural Ecosystem\n3 species competing in shared activation space',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase61_ecosystem.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
