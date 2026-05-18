# -*- coding: utf-8 -*-
"""
Phase 1: Beyond Arithmetic - Sorting Register Discovery
Can the Neural Von Neumann Machine sort? Do Fetch-Decode-Execute-Store
registers exist for SORTING, not just arithmetic?

Method: Linear probes at every layer for sorting tasks:
  "Sort: 3,1,4,2 ->" should produce "1,2,3,4"
  Probe each layer for: min value, max value, sorted order position.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sorting problems: (prompt, sorted_first_element)
# We probe whether the model knows the minimum at each layer
SORT_PROBLEMS = []
import itertools
digits = list(range(1, 7))  # 1-6
for combo in itertools.combinations(digits, 3):
    for perm in itertools.permutations(combo):
        a, b, c = perm
        prompt = f"def sort({a},{b},{c}): return "
        minimum = min(perm)
        maximum = max(perm)
        median_val = sorted(perm)[1]
        SORT_PROBLEMS.append({
            'prompt': prompt,
            'values': list(perm),
            'min': minimum,
            'max': maximum,
            'median': median_val,
            'sorted': sorted(perm),
        })

# Limit to manageable size
import random
random.seed(42)
if len(SORT_PROBLEMS) > 120:
    SORT_PROBLEMS = random.sample(SORT_PROBLEMS, 120)

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9"]


def main():
    print("[P1] Sorting Register Discovery")
    print(f"  Device: {DEVICE}")
    print(f"  Problems: {len(SORT_PROBLEMS)}")
    start_time = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=DEVICE,
        local_files_only=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Layers: {n_layers}")

    # Collect hidden states for all problems
    all_hidden = {l: [] for l in range(n_layers)}
    all_labels = {'min': [], 'max': [], 'median': []}

    for prob in SORT_PROBLEMS:
        inp = tok(prob['prompt'], return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for l in range(n_layers):
            h = out.hidden_states[l+1][0, -1, :].float().cpu().numpy()
            all_hidden[l].append(h)

        all_labels['min'].append(prob['min'])
        all_labels['max'].append(prob['max'])
        all_labels['median'].append(prob['median'])

    # Train linear probes at each layer
    results = {}
    for target_name in ['min', 'max', 'median']:
        results[target_name] = {}
        y = np.array(all_labels[target_name])
        for l in range(n_layers):
            X = np.array(all_hidden[l])
            try:
                clf = LogisticRegression(max_iter=500, random_state=42)
                scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                acc = scores.mean()
            except Exception:
                acc = 0.0
            results[target_name][str(l)] = round(float(acc), 4)

        best_layer = max(results[target_name], key=results[target_name].get)
        best_acc = results[target_name][best_layer]
        print(f"  {target_name}: best=L{best_layer} ({best_acc:.1%})")

    # Save results
    output = {
        'phase': 1,
        'name': 'sorting_register_discovery',
        'n_problems': len(SORT_PROBLEMS),
        'n_layers': n_layers,
        'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase1_sorting.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, target_name in enumerate(['min', 'max', 'median']):
        layers = list(range(n_layers))
        accs = [results[target_name][str(l)] for l in layers]
        axes[i].plot(layers, accs, 'o-', linewidth=2, markersize=4)
        axes[i].set_xlabel('Layer', fontsize=12)
        axes[i].set_ylabel('Probe Accuracy', fontsize=12)
        axes[i].set_title(f'{target_name.upper()} Register', fontsize=14, fontweight='bold')
        axes[i].set_ylim(0, 1.05)
        axes[i].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='chance')
        best_l = int(max(results[target_name], key=results[target_name].get))
        axes[i].axvline(x=best_l, color='green', linestyle=':', alpha=0.7)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Phase 1: Sorting Register Discovery\n'
                 'Can the Neural Von Neumann Machine sort?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase1_sorting.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.0f}s")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
