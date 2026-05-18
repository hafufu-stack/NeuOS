# -*- coding: utf-8 -*-
"""
Phase 62: Dream Compilation
Run model on random text, capture activation patterns,
then try to "crystallize" them into executable programs.
Can the model dream new programs from noise?

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, get_last_token, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P62] Dream Compilation")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    DREAM_L = 8

    for p in model.parameters():
        p.requires_grad = False

    # Step 1: Generate "dreams" from random text
    print("  Step 1: Dreaming (capturing activations from random text)...")
    dream_prompts = [
        "The sun rises over", "Calculate the result of",
        "In mathematics, we know", "The largest number is",
        "Minimum of all values", "Return the maximum",
        "Sum equals the total", "Subtract and find",
        "Compare two numbers", "The answer is exactly",
        "First operand plus", "Second value minus",
        "Greater than or equal", "Less than the given",
        "Output the smaller", "Pick the bigger one",
        "Add these together", "Multiply by factor",
        "Divide equally among", "Result of computation",
    ]

    dream_vecs = []
    for p in dream_prompts:
        cap = [None]
        def capture(module, input, output):
            cap[0] = get_last_token(output)
        h = model.model.layers[DREAM_L].register_forward_hook(capture)
        inp = tok(p, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        dream_vecs.append(cap[0].float().cpu().numpy().flatten())

    # Step 2: Cluster dreams
    print("  Step 2: Clustering dreams...")
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(np.array(dream_vecs))
    centroids = kmeans.cluster_centers_

    for c in range(n_clusters):
        members = [dream_prompts[i] for i in range(len(dream_prompts)) if labels[i] == c]
        print(f"    Cluster {c}: {members[:3]}")

    # Step 3: Inject dream centroids as programs
    print("\n  Step 3: Testing dream programs...")
    test_prompts = ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) ="]
    dream_results = {}

    for c in range(n_clusters):
        centroid_vec = torch.tensor(centroids[c], device=DEVICE, dtype=torch.float32)
        preds = []
        for tp in test_prompts:
            def inject_dream(module, input, output, v=centroid_vec):
                return replace_last_token(output, v)
            h = model.model.layers[DREAM_L].register_forward_hook(inject_dream)
            inp = tok(tp, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            preds.append(pred)
        dream_results[f'cluster_{c}'] = preds
        print(f"    Cluster {c} program: {preds}")

    # Step 4: Compare dream outputs to known functions
    known_functions = {
        'MIN': ["3", "2", "1", "4"],
        'MAX': ["7", "5", "8", "6"],
        'FIRST': ["3", "5", "8", "4"],
        'SECOND': ["7", "2", "1", "6"],
    }

    function_matches = {}
    for c_name, c_preds in dream_results.items():
        best_func = None
        best_match = 0
        for fn_name, fn_expected in known_functions.items():
            match = sum(p == e for p, e in zip(c_preds, fn_expected))
            if match > best_match:
                best_match = match
                best_func = fn_name
        function_matches[c_name] = {
            'best_function': best_func if best_match > 0 else 'UNKNOWN',
            'match_rate': best_match / len(test_prompts),
            'predictions': c_preds,
        }
        if best_match > 0:
            print(f"    {c_name} -> {best_func} ({best_match}/{len(test_prompts)})")

    # Dream diversity
    sim_matrix = cosine_similarity(centroids)

    # Save
    output = {
        'phase': 62, 'name': 'dream_compilation',
        'n_dreams': len(dream_prompts),
        'n_clusters': n_clusters,
        'dream_results': dream_results,
        'function_matches': function_matches,
        'centroid_sim_matrix': sim_matrix.tolist(),
        'cluster_sizes': [int(sum(labels == c)) for c in range(n_clusters)],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase62_dream.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Cluster sizes
    sizes = [sum(labels == c) for c in range(n_clusters)]
    axes[0].bar(range(n_clusters), sizes, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
                edgecolor='black')
    axes[0].set_xlabel('Cluster'); axes[0].set_ylabel('# Dreams')
    axes[0].set_title('Dream Clusters', fontweight='bold')

    # Function identification
    func_names = [function_matches[f'cluster_{c}']['best_function'] for c in range(n_clusters)]
    match_rates = [function_matches[f'cluster_{c}']['match_rate'] for c in range(n_clusters)]
    colors_bar = ['tab:green' if r > 0.5 else 'tab:red' for r in match_rates]
    axes[1].bar(range(n_clusters), match_rates, color=colors_bar, edgecolor='black')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Match Rate')
    axes[1].set_title('Dream -> Function Match', fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    for i, (fn, r) in enumerate(zip(func_names, match_rates)):
        axes[1].text(i, r+0.03, f'{fn}\n{r:.0%}', ha='center', fontweight='bold', fontsize=9)

    # Centroid similarity
    im = axes[2].imshow(sim_matrix, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    axes[2].set_title('Dream Centroid Similarity', fontweight='bold')
    plt.colorbar(im, ax=axes[2])
    for i in range(n_clusters):
        for j in range(n_clusters):
            axes[2].text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)

    plt.suptitle('Phase 62: Dream Compilation\nCrystallizing random activations into executable programs',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase62_dream.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
