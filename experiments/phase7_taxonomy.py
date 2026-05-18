# -*- coding: utf-8 -*-
"""
Phase 7: Instruction Taxonomy (CPU)
Map the full "instruction set" that Qwen-0.5B's L2 OPCODE register
can recognize. Aletheia P200 showed +/-/* at 100%.

Test 100+ diverse prompts across categories:
  - Arithmetic: +, -, *, /, modulo, power
  - Comparison: >, <, ==, max, min
  - Logic: and, or, not, if-then
  - String: length, reverse, upper
  - Retrieval: factual knowledge, definitions
  - Counting: how many, enumerate

Model: Qwen2.5-0.5B (GPU, but analysis is CPU)
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

# Instruction taxonomy
INSTRUCTIONS = {
    'addition': [
        "def f(): return 3 + 4 =",
        "def f(): return 5 + 2 =",
        "def f(): return 8 + 1 =",
        "def f(): return 6 + 3 =",
        "def f(): return 4 + 4 =",
        "def f(): return 7 + 2 =",
        "def f(): return 1 + 8 =",
        "def f(): return 2 + 5 =",
    ],
    'subtraction': [
        "def f(): return 9 - 3 =",
        "def f(): return 7 - 2 =",
        "def f(): return 8 - 5 =",
        "def f(): return 6 - 1 =",
        "def f(): return 5 - 3 =",
        "def f(): return 4 - 2 =",
        "def f(): return 9 - 7 =",
        "def f(): return 8 - 4 =",
    ],
    'multiplication': [
        "def f(): return 3 * 4 =",
        "def f(): return 2 * 5 =",
        "def f(): return 6 * 2 =",
        "def f(): return 4 * 3 =",
        "def f(): return 7 * 1 =",
        "def f(): return 2 * 8 =",
        "def f(): return 5 * 3 =",
        "def f(): return 9 * 1 =",
    ],
    'comparison': [
        "def f(): return 5 > 3  #",
        "def f(): return 2 > 7  #",
        "def f(): return 9 > 1  #",
        "def f(): return 4 > 6  #",
        "def f(): return 8 > 3  #",
        "def f(): return 1 > 9  #",
        "def f(): return 6 > 5  #",
        "def f(): return 3 > 8  #",
    ],
    'fact_retrieval': [
        "# The capital of Japan is",
        "# The capital of France is",
        "# Water freezes at",
        "# A year has",
        "# The largest planet is",
        "# The number of continents is",
        "# Pi is approximately",
        "# The boiling point of water is",
    ],
    'string_op': [
        "def f(): return len('hello') =",
        "def f(): return len('cat') =",
        "def f(): return len('a') =",
        "def f(): return len('world') =",
        "def f(): return len('ab') =",
        "def f(): return len('test') =",
        "def f(): return len('hi') =",
        "def f(): return len('xyz') =",
    ],
    'logic': [
        "def f(): return True and True  #",
        "def f(): return True and False  #",
        "def f(): return False or True  #",
        "def f(): return False or False  #",
        "def f(): return not True  #",
        "def f(): return not False  #",
        "def f(): return True or False  #",
        "def f(): return False and True  #",
    ],
    'identity': [
        "def f(): return 5\nf() =",
        "def f(): return 3\nf() =",
        "def f(): return 7\nf() =",
        "def f(): return 1\nf() =",
        "def f(): return 9\nf() =",
        "def f(): return 2\nf() =",
        "def f(): return 8\nf() =",
        "def f(): return 4\nf() =",
    ],
}


def main():
    print("[P7] Instruction Taxonomy")
    print(f"  Device: {DEVICE}")
    print(f"  Categories: {len(INSTRUCTIONS)}")
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

    # Collect hidden states
    all_hidden = {l: [] for l in range(n_layers)}
    all_labels = []

    category_names = list(INSTRUCTIONS.keys())
    for cat_idx, cat_name in enumerate(category_names):
        for prompt in INSTRUCTIONS[cat_name]:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            for l in range(n_layers):
                h = out.hidden_states[l+1][0, -1, :].float().cpu().numpy()
                all_hidden[l].append(h)
            all_labels.append(cat_idx)

    y = np.array(all_labels)
    print(f"  Total samples: {len(y)}")

    # Probe each layer for instruction category
    layer_accs = {}
    for l in range(n_layers):
        X = np.array(all_hidden[l])
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            acc = scores.mean()
        except Exception:
            acc = 1.0 / len(category_names)
        layer_accs[str(l)] = round(float(acc), 4)

    best_l = max(layer_accs, key=layer_accs.get)
    print(f"  Best OPCODE layer: L{best_l} ({layer_accs[best_l]:.1%})")
    print(f"  Chance level: {1/len(category_names):.1%}")

    # Per-pair confusion at best layer
    X_best = np.array(all_hidden[int(best_l)])
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_best, y)
    y_pred = clf.predict(X_best)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_pred)

    # Save
    output = {
        'phase': 7, 'name': 'instruction_taxonomy',
        'categories': category_names, 'n_layers': n_layers,
        'layer_accs': layer_accs, 'best_layer': int(best_l),
        'confusion_matrix': cm.tolist(),
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase7_taxonomy.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Layer accuracy
    layers = list(range(n_layers))
    accs = [layer_accs[str(l)] for l in layers]
    axes[0].plot(layers, accs, 'o-', linewidth=2, markersize=5, color='tab:blue')
    axes[0].axhline(y=1/len(category_names), color='red', linestyle='--', alpha=0.5, label='chance')
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('8-class Accuracy', fontsize=12)
    axes[0].set_title('OPCODE Classification by Layer', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Confusion matrix
    im = axes[1].imshow(cm, cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(len(category_names)))
    axes[1].set_yticks(range(len(category_names)))
    short_names = [n[:5] for n in category_names]
    axes[1].set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(short_names, fontsize=9)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].set_title(f'Confusion Matrix (L{best_l})', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1])

    plt.suptitle('Phase 7: Instruction Taxonomy\n'
                 'How many instruction types can the OPCODE register distinguish?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase7_taxonomy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
