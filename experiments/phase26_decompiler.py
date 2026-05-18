# -*- coding: utf-8 -*-
"""
Phase 26: The Neural Decompiler (Opus Original)
Can we identify what program is running by observing its registers?

Method:
  1. Build a "register library": known vectors for MIN, MAX, SUM, SUB at L16
  2. Run unknown prompts (with task instructions), capture L16 state
  3. Compare L16 state to library via cosine similarity
  4. Check if we can correctly identify the operation

This is runtime program identification -- the AI equivalent of
a debugger's "what function is currently executing?" query.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def capture_at_layer(model, tok, prompt, layer):
    captured = [None]
    def cap(module, input, output):
        captured[0] = get_last_token(output)
    h = model.model.layers[layer].register_forward_hook(cap)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model(**inp)
    h.remove()
    return captured[0].float().cpu().numpy().flatten()


def main():
    print("[P26] The Neural Decompiler (Opus Original)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # === Step 1: Build register library ===
    print("  Step 1: Building register library...")
    LAYERS_TO_CHECK = [0, 2, 4, 8, 12, 14, 16, 18, 20, 22]

    library_prompts = {
        'MIN': [f"def f(): return min({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:8],
        'MAX': [f"def f(): return max({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:8],
        'SUM': [f"def f(): return {a} + {b} =" for a in range(1, 6) for b in range(1, 6) if a+b < 10][:8],
        'SUB': [f"def f(): return {a} - {b} =" for a in range(3, 9) for b in range(1, 4) if a > b][:8],
        'SORT': [f"def f(): return sorted([{a}, {b}]) =" for a in range(2, 8) for b in range(2, 8) if a != b][:8],
    }

    # Build library: average vector per operation per layer
    library = {}  # {layer: {op: vec}}
    for layer in LAYERS_TO_CHECK:
        library[layer] = {}
        for op_name, prompts in library_prompts.items():
            vecs = [capture_at_layer(model, tok, p, layer) for p in prompts]
            library[layer][op_name] = np.mean(vecs, axis=0)
        print(f"    L{layer}: {len(library_prompts)} operations indexed")

    # === Step 2: Test prompts (new, unseen examples) ===
    print("\n  Step 2: Decompiling unknown programs...")

    test_prompts = [
        ("def f(): return min(9, 3) =", "MIN"),
        ("def f(): return min(4, 7) =", "MIN"),
        ("def f(): return max(1, 6) =", "MAX"),
        ("def f(): return max(5, 9) =", "MAX"),
        ("def f(): return 2 + 6 =", "SUM"),
        ("def f(): return 4 + 3 =", "SUM"),
        ("def f(): return 8 - 2 =", "SUB"),
        ("def f(): return 7 - 5 =", "SUB"),
        ("def f(): return sorted([3, 1]) =", "SORT"),
        ("def f(): return sorted([7, 2]) =", "SORT"),
    ]

    # For each test, capture vector and find nearest library entry
    results_per_layer = {}
    for layer in LAYERS_TO_CHECK:
        correct = 0
        predictions = []
        for prompt, true_op in test_prompts:
            vec = capture_at_layer(model, tok, prompt, layer)

            # Compare to library
            best_op = None
            best_sim = -1
            for op_name, lib_vec in library[layer].items():
                sim = cosine_similarity(vec.reshape(1, -1), lib_vec.reshape(1, -1))[0, 0]
                if sim > best_sim:
                    best_sim = sim
                    best_op = op_name

            predictions.append({
                'prompt': prompt[:30], 'true': true_op,
                'predicted': best_op, 'confidence': round(float(best_sim), 4),
                'correct': best_op == true_op,
            })
            if best_op == true_op:
                correct += 1

        acc = correct / len(test_prompts)
        results_per_layer[layer] = {
            'accuracy': round(acc, 4),
            'predictions': predictions,
        }
        print(f"    L{layer}: {acc:.1%} ({correct}/{len(test_prompts)})")

    # === Step 3: Confusion matrix at best layer ===
    best_layer = max(results_per_layer, key=lambda l: results_per_layer[l]['accuracy'])
    print(f"\n  Best decompiler layer: L{best_layer} ({results_per_layer[best_layer]['accuracy']:.0%})")

    # Build confusion matrix
    ops = list(library_prompts.keys())
    conf_mat = np.zeros((len(ops), len(ops)))
    for pred_info in results_per_layer[best_layer]['predictions']:
        true_idx = ops.index(pred_info['true'])
        pred_idx = ops.index(pred_info['predicted'])
        conf_mat[true_idx, pred_idx] += 1

    # Normalize
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    conf_mat_norm = np.divide(conf_mat, row_sums, where=row_sums > 0, out=np.zeros_like(conf_mat))

    # === Step 4: Multi-layer decompiler (majority vote) ===
    print("\n  Step 4: Multi-layer decompiler (majority vote)...")
    multi_correct = 0
    for prompt, true_op in test_prompts:
        votes = {}
        for layer in LAYERS_TO_CHECK:
            vec = capture_at_layer(model, tok, prompt, layer)
            best_op = max(library[layer], key=lambda op:
                         cosine_similarity(vec.reshape(1,-1), library[layer][op].reshape(1,-1))[0,0])
            votes[best_op] = votes.get(best_op, 0) + 1
        predicted = max(votes, key=votes.get)
        if predicted == true_op:
            multi_correct += 1
    multi_acc = multi_correct / len(test_prompts)
    print(f"    Multi-layer accuracy: {multi_acc:.1%}")

    # Save
    output = {
        'phase': 26, 'name': 'neural_decompiler',
        'layers_checked': LAYERS_TO_CHECK,
        'best_layer': best_layer,
        'accuracy_per_layer': {str(l): results_per_layer[l]['accuracy'] for l in LAYERS_TO_CHECK},
        'multi_layer_accuracy': round(multi_acc, 4),
        'confusion_matrix': conf_mat.tolist(),
        'ops': ops,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase26_decompiler.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy per layer
    layer_labels = [f'L{l}' for l in LAYERS_TO_CHECK]
    layer_accs = [results_per_layer[l]['accuracy'] for l in LAYERS_TO_CHECK]
    axes[0].bar(layer_labels, layer_accs, color='tab:blue', edgecolor='black')
    axes[0].set_ylabel('Classification Accuracy', fontsize=11)
    axes[0].set_title('Decompiler Accuracy per Layer', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=1.0/len(ops), color='red', linestyle='--', label=f'Chance ({1/len(ops):.0%})')
    axes[0].legend()
    for i, v in enumerate(layer_accs):
        if v > 0.3:
            axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=9)

    # Confusion matrix
    im = axes[1].imshow(conf_mat_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xticks(range(len(ops)))
    axes[1].set_xticklabels(ops, fontsize=9)
    axes[1].set_yticks(range(len(ops)))
    axes[1].set_yticklabels(ops, fontsize=9)
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('True', fontsize=11)
    axes[1].set_title(f'Confusion Matrix (L{best_layer})', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[1])
    for i in range(len(ops)):
        for j in range(len(ops)):
            axes[1].text(j, i, f'{conf_mat_norm[i,j]:.0%}', ha='center', va='center',
                        fontsize=9, color='white' if conf_mat_norm[i,j] > 0.5 else 'black')

    axes[2].axis('off')
    summary = (
        f"Neural Decompiler\n\n"
        f"Best single layer: L{best_layer} ({results_per_layer[best_layer]['accuracy']:.0%})\n"
        f"Multi-layer vote: {multi_acc:.0%}\n"
        f"Chance: {1/len(ops):.0%}\n\n"
        f"5 operations identified:\n"
        f"  MIN, MAX, SUM, SUB, SORT\n\n"
    )
    if multi_acc > 0.6:
        summary += "We can READ running programs!"
    elif multi_acc > 0.3:
        summary += "Partial program identification"
    else:
        summary += "Programs are encrypted"
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=12, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 26: The Neural Decompiler\nCan we identify running programs from register state?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase26_decompiler.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
