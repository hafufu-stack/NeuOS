# -*- coding: utf-8 -*-
"""
Phase 79: Neural Compiler Bootstrapping
Can NeuOS learn UNKNOWN functions from I/O examples alone?
Given only 5 input-output pairs, self-compile a program vector
that generalizes to unseen inputs.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Unknown functions to discover
FUNCTIONS = {
    'ADD': lambda a, b: a + b,
    'SUBTRACT': lambda a, b: a - b,
    'MULTIPLY': lambda a, b: a * b,
    'ABS_DIFF': lambda a, b: abs(a - b),
}


def compile_from_examples(model, tok, examples, layer, device, epochs=100):
    """Self-compile a program from I/O examples only (no function name given)."""
    hs = model.config.hidden_size
    torch.manual_seed(42)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    history = []
    for epoch in range(epochs):
        total_loss = 0
        for a, b, result in examples:
            prompt = f"{a}, {b}) ="
            target_str = f" {result}"
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
            total_loss += loss.item()
        if epoch % 20 == 0 or epoch == epochs - 1:
            history.append({'epoch': epoch, 'loss': round(total_loss/len(examples), 4)})
    return vec.detach(), history


def eval_function(model, tok, vec, test_pairs, layer, device):
    """Evaluate compiled vector on unseen test pairs."""
    correct = 0
    details = []
    for a, b, expected in test_pairs:
        prompt = f"{a}, {b}) ="
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        is_correct = pred == str(expected)
        if is_correct: correct += 1
        details.append({'input': f"({a},{b})", 'expected': str(expected),
                       'predicted': pred, 'correct': is_correct})
    return correct / len(test_pairs), details


def main():
    print("[P79] Neural Compiler Bootstrapping")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False
    layer = 8  # Program injection layer

    all_results = {}
    for fn_name, fn in FUNCTIONS.items():
        print(f"\n  === {fn_name} ===")

        # Generate train examples (5 only!)
        np.random.seed(42)
        pairs = [(int(a), int(b)) for a, b in
                 np.random.randint(1, 9, size=(15, 2))]
        train_pairs = [(a, b, fn(a, b)) for a, b in pairs[:5]]
        test_pairs = [(a, b, fn(a, b)) for a, b in pairs[5:]
                      if 0 <= fn(a, b) <= 9]  # single digit results only

        if len(test_pairs) < 3:
            # Fallback: generate more test pairs with single-digit results
            extra = [(a, b, fn(a, b)) for a in range(1, 10) for b in range(1, 10)
                     if 0 <= fn(a, b) <= 9 and (a, b) not in pairs[:5]]
            test_pairs = extra[:10]

        print(f"    Train: {[(a,b,r) for a,b,r in train_pairs]}")
        print(f"    Test pairs: {len(test_pairs)}")

        # Compile
        vec, history = compile_from_examples(
            model, tok, train_pairs, layer, DEVICE, epochs=120)

        # Evaluate
        acc, details = eval_function(model, tok, vec, test_pairs, layer, DEVICE)
        print(f"    Accuracy: {acc:.0%} ({sum(d['correct'] for d in details)}/{len(details)})")

        # Show some predictions
        for d in details[:5]:
            mark = "OK" if d['correct'] else "XX"
            print(f"      {d['input']} -> {d['predicted']} "
                  f"(expected {d['expected']}) [{mark}]")

        all_results[fn_name] = {
            'train_examples': [(a, b, r) for a, b, r in train_pairs],
            'test_accuracy': round(acc, 4),
            'n_test': len(test_pairs),
            'training_history': history,
            'details': details,
            'vec_norm': round(vec.norm().item(), 4),
        }

    # Cross-function similarity
    print("\n  Cross-function vector similarity:")
    vecs = {}
    for fn_name in FUNCTIONS:
        # Re-compile with consistent seed for comparison
        train = [(a, b, FUNCTIONS[fn_name](a, b))
                 for a, b in [(3,5),(2,7),(4,1),(8,3),(6,2)]
                 if 0 <= FUNCTIONS[fn_name](a, b) <= 9]
        if len(train) >= 3:
            v, _ = compile_from_examples(model, tok, train[:5], layer, DEVICE, epochs=80)
            vecs[fn_name] = v

    sim_matrix = {}
    for n1 in vecs:
        for n2 in vecs:
            if n1 < n2:
                cs = torch.nn.functional.cosine_similarity(
                    vecs[n1].unsqueeze(0), vecs[n2].unsqueeze(0)).item()
                sim_matrix[f"{n1}_vs_{n2}"] = round(cs, 4)
                print(f"    {n1} vs {n2}: cos_sim={cs:.4f}")

    output = {
        'phase': 79, 'name': 'compiler_bootstrapping',
        'functions': all_results,
        'cross_similarity': sim_matrix,
        'best_function': max(all_results, key=lambda k: all_results[k]['test_accuracy']),
        'best_accuracy': max(r['test_accuracy'] for r in all_results.values()),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase79_bootstrap.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy per function
    names = list(all_results.keys())
    accs = [all_results[n]['test_accuracy'] for n in names]
    colors = ['tab:blue','tab:orange','tab:green','tab:red']
    axes[0].bar(names, accs, color=colors[:len(names)], edgecolor='black')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Unknown Function Inference', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, a in enumerate(accs):
        axes[0].text(i, a+0.03, f'{a:.0%}', ha='center', fontweight='bold')

    # Training loss curves
    for i, n in enumerate(names):
        h = all_results[n]['training_history']
        epochs = [x['epoch'] for x in h]
        losses = [x['loss'] for x in h]
        axes[1].plot(epochs, losses, '-o', label=n, color=colors[i], linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Cross-similarity heatmap
    fn_list = list(vecs.keys())
    n = len(fn_list)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            key = f"{fn_list[i]}_vs_{fn_list[j]}" if fn_list[i] < fn_list[j] else f"{fn_list[j]}_vs_{fn_list[i]}"
            if key in sim_matrix:
                mat[i,j] = mat[j,i] = sim_matrix[key]
    im = axes[2].imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_xticks(range(n)); axes[2].set_yticks(range(n))
    axes[2].set_xticklabels(fn_list, rotation=45, ha='right', fontsize=8)
    axes[2].set_yticklabels(fn_list, fontsize=8)
    for i in range(n):
        for j in range(n):
            axes[2].text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    axes[2].set_title('Function Vector Similarity', fontweight='bold')

    plt.suptitle('Phase 79: Neural Compiler Bootstrapping\n'
                'Learning unknown functions from 5 I/O examples',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase79_bootstrap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Best: {output['best_function']} at {output['best_accuracy']:.0%}")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
