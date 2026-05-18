# -*- coding: utf-8 -*-
"""
Phase 43: Neural Debugger
Build a real-time state inspector using P41's probes.
Read (operation, operand_a, operand_b, result) from hidden states
during execution, like gdb for neural networks.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P43] Neural Debugger")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)

    # Optimal layers from P41
    OP_LAYER = 2      # operation: 100%
    OPB_LAYER = 8     # operand_b: 88%
    OPA_LAYER = 16    # operand_a: 90%
    RESULT_LAYER = 22 # result: 61%

    # Step 1: Train probe classifiers
    print("  Step 1: Training probe classifiers...")
    ops = {'MIN': 'min({a}, {b})', 'MAX': 'max({a}, {b})',
           'SUM': '{a} + {b}', 'SUB': '{a} - {b}'}
    train_data = []
    for op_name, template in ops.items():
        for a in range(2, 8):
            for b in range(2, 8):
                if a == b: continue
                fn = {'MIN': min, 'MAX': max, 'SUM': lambda x,y: x+y, 'SUB': lambda x,y: x-y}[op_name]
                r = fn(a, b)
                if r < 0 or r >= 10: continue
                expr = template.format(a=a, b=b)
                train_data.append((f"def f(): return {expr} =", op_name, a, b, r))

    # Extract vectors at all probe layers
    probe_vecs = {l: [] for l in [OP_LAYER, OPB_LAYER, OPA_LAYER, RESULT_LAYER]}
    labels = {'op': [], 'a': [], 'b': [], 'result': []}

    for prompt, op, a, b, r in train_data:
        labels['op'].append(op)
        labels['a'].append(a)
        labels['b'].append(b)
        labels['result'].append(r)
        for layer in probe_vecs:
            cap = [None]
            def capture(module, input, output):
                cap[0] = get_last_token(output)
            h = model.model.layers[layer].register_forward_hook(capture)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            probe_vecs[layer].append(cap[0].float().cpu().numpy().flatten())

    # Train probes
    probes = {}
    for name, layer, y in [('op', OP_LAYER, labels['op']),
                            ('operand_a', OPA_LAYER, labels['a']),
                            ('operand_b', OPB_LAYER, labels['b']),
                            ('result', RESULT_LAYER, labels['result'])]:
        X = np.array(probe_vecs[layer])
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X, y)
        train_acc = clf.score(X, y)
        probes[name] = {'clf': clf, 'layer': layer, 'train_acc': round(train_acc, 4)}
        print(f"    {name} probe (L{layer}): train_acc={train_acc:.1%}")

    # Step 2: Run debugger on unseen programs
    print("\n  Step 2: Debugging unseen programs...")
    test_programs = [
        ("def f(): return min(9, 3) =", "MIN", 9, 3, 3),
        ("def f(): return max(1, 8) =", "MAX", 1, 8, 8),
        ("def f(): return 7 + 2 =", "SUM", 7, 2, 9),
        ("def f(): return 6 - 4 =", "SUB", 6, 4, 2),
        ("def f(): return min(2, 7) =", "MIN", 2, 7, 2),
        ("def f(): return max(5, 3) =", "MAX", 5, 3, 5),
        ("def f(): return 3 + 4 =", "SUM", 3, 4, 7),
        ("def f(): return 8 - 3 =", "SUB", 8, 3, 5),
    ]

    debug_results = []
    for prompt, true_op, true_a, true_b, true_r in test_programs:
        # Capture states at all probe layers
        states = {}
        for layer in [OP_LAYER, OPB_LAYER, OPA_LAYER, RESULT_LAYER]:
            cap = [None]
            def capture(module, input, output):
                cap[0] = get_last_token(output)
            h = model.model.layers[layer].register_forward_hook(capture)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            states[layer] = cap[0].float().cpu().numpy().flatten()

        # Read state via probes
        pred_op = probes['op']['clf'].predict(states[OP_LAYER].reshape(1,-1))[0]
        pred_a = probes['operand_a']['clf'].predict(states[OPA_LAYER].reshape(1,-1))[0]
        pred_b = probes['operand_b']['clf'].predict(states[OPB_LAYER].reshape(1,-1))[0]
        pred_r = probes['result']['clf'].predict(states[RESULT_LAYER].reshape(1,-1))[0]

        all_correct = (pred_op == true_op and pred_a == true_a and
                       pred_b == true_b and pred_r == true_r)

        debug_results.append({
            'prompt': prompt[:35], 'true': f'{true_op}({true_a},{true_b})={true_r}',
            'debug': f'{pred_op}({pred_a},{pred_b})={pred_r}',
            'op_ok': pred_op == true_op, 'a_ok': pred_a == true_a,
            'b_ok': pred_b == true_b, 'r_ok': pred_r == true_r,
            'all_ok': all_correct,
        })

        status = 'PERFECT' if all_correct else 'PARTIAL'
        print(f"    {prompt[:35]}")
        print(f"      TRUE:  {true_op}({true_a}, {true_b}) = {true_r}")
        print(f"      DEBUG: {pred_op}({pred_a}, {pred_b}) = {pred_r}  [{status}]")

    # Compute accuracies
    op_acc = sum(r['op_ok'] for r in debug_results) / len(debug_results)
    a_acc = sum(r['a_ok'] for r in debug_results) / len(debug_results)
    b_acc = sum(r['b_ok'] for r in debug_results) / len(debug_results)
    r_acc = sum(r['r_ok'] for r in debug_results) / len(debug_results)
    full_acc = sum(r['all_ok'] for r in debug_results) / len(debug_results)

    print(f"\n  Debugger accuracy: op={op_acc:.0%}, a={a_acc:.0%}, "
          f"b={b_acc:.0%}, r={r_acc:.0%}, FULL={full_acc:.0%}")

    # Save
    output = {
        'phase': 43, 'name': 'neural_debugger',
        'probe_layers': {'op': OP_LAYER, 'a': OPA_LAYER, 'b': OPB_LAYER, 'r': RESULT_LAYER},
        'probe_train_accs': {n: p['train_acc'] for n, p in probes.items()},
        'debug_results': debug_results,
        'test_accuracies': {'op': round(op_acc, 4), 'a': round(a_acc, 4),
                           'b': round(b_acc, 4), 'r': round(r_acc, 4), 'full': round(full_acc, 4)},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase43_debugger.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cats = ['Operation\n(L2)', 'Operand A\n(L16)', 'Operand B\n(L8)', 'Result\n(L22)', 'ALL\nCorrect']
    accs = [op_acc, a_acc, b_acc, r_acc, full_acc]
    colors = ['tab:green' if a >= 0.75 else 'tab:orange' if a >= 0.5 else 'tab:red' for a in accs]
    axes[0].bar(cats, accs, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Neural Debugger: Test Accuracy', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Debug output visualization
    axes[1].axis('off')
    header = "Neural Debugger Output\n" + "="*40 + "\n\n"
    for r in debug_results[:6]:
        status = 'OK' if r['all_ok'] else 'XX'
        header += f"[{status}] TRUE: {r['true']}\n     READ: {r['debug']}\n\n"
    axes[1].text(0.05, 0.95, header, transform=axes[1].transAxes, fontsize=9,
                va='top', family='monospace', bbox=dict(boxstyle='round', facecolor='black', alpha=0.9),
                color='lime')

    plt.suptitle('Phase 43: Neural Debugger (gdb for LLMs)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase43_debugger.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
