# -*- coding: utf-8 -*-
"""
Phase 2: Conditional Branch Register
Does the transformer have an internal "branch predictor"?

Test: "if A > B then X else Y" style prompts.
Probe each layer for: (1) comparison result (A>B?), (2) selected branch (X or Y).

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

# Conditional problems
COND_PROBLEMS = []
for a in range(1, 10):
    for b in range(1, 10):
        if a == b:
            continue
        # "def f(): return 'big' if A > B else 'small'"
        result = 'big' if a > b else 'small'
        prompt = f"def f(): return 'big' if {a} > {b} else 'small'\nf() = '"
        COND_PROBLEMS.append({
            'prompt': prompt,
            'a': a, 'b': b,
            'a_gt_b': int(a > b),
            'result': result,
        })

import random
random.seed(42)
if len(COND_PROBLEMS) > 72:
    COND_PROBLEMS = random.sample(COND_PROBLEMS, 72)


def main():
    print("[P2] Conditional Branch Register")
    print(f"  Device: {DEVICE}")
    print(f"  Problems: {len(COND_PROBLEMS)}")
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

    all_hidden = {l: [] for l in range(n_layers)}
    labels_comparison = []  # 0=a<=b, 1=a>b
    labels_branch = []      # 0=small, 1=big

    for prob in COND_PROBLEMS:
        inp = tok(prob['prompt'], return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for l in range(n_layers):
            h = out.hidden_states[l+1][0, -1, :].float().cpu().numpy()
            all_hidden[l].append(h)

        labels_comparison.append(prob['a_gt_b'])
        labels_branch.append(1 if prob['result'] == 'big' else 0)

    # Probe for comparison result and branch selection
    results = {}
    for target_name, y_arr in [('comparison', labels_comparison), ('branch', labels_branch)]:
        results[target_name] = {}
        y = np.array(y_arr)
        for l in range(n_layers):
            X = np.array(all_hidden[l])
            try:
                clf = LogisticRegression(max_iter=500, random_state=42)
                scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                acc = scores.mean()
            except Exception:
                acc = 0.5
            results[target_name][str(l)] = round(float(acc), 4)

        best_l = max(results[target_name], key=results[target_name].get)
        print(f"  {target_name}: best=L{best_l} ({results[target_name][best_l]:.1%})")

    # Save
    output = {
        'phase': 2, 'name': 'conditional_branch_register',
        'n_problems': len(COND_PROBLEMS), 'n_layers': n_layers,
        'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase2_conditional.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, name in enumerate(['comparison', 'branch']):
        layers = list(range(n_layers))
        accs = [results[name][str(l)] for l in layers]
        axes[i].plot(layers, accs, 'o-', linewidth=2, markersize=5,
                     color='tab:blue' if i == 0 else 'tab:orange')
        axes[i].set_xlabel('Layer', fontsize=12)
        axes[i].set_ylabel('Probe Accuracy', fontsize=12)
        axes[i].set_title(f'{name.upper()} Register', fontsize=14, fontweight='bold')
        axes[i].set_ylim(0.4, 1.05)
        axes[i].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Phase 2: Conditional Branch Register\n'
                 '"if A > B then X else Y" - Where is the branch decided?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase2_conditional.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
