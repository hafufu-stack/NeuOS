# -*- coding: utf-8 -*-
"""
Phase 41: Register Fingerprinting (Opus Original)
Can we read not just WHICH operation, but WHICH SPECIFIC NUMBERS
are being computed from the register state?
Memory content forensics: decode (a, b, op) from hidden states.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P41] Register Fingerprinting")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)

    # Generate diverse dataset
    prompts = []
    labels_a = []
    labels_b = []
    labels_op = []
    labels_result = []

    ops = {
        'MIN': (lambda a, b: min(a, b), "min({a}, {b})"),
        'MAX': (lambda a, b: max(a, b), "max({a}, {b})"),
        'SUM': (lambda a, b: a + b, "{a} + {b}"),
        'SUB': (lambda a, b: a - b, "{a} - {b}"),
    }

    for op_name, (fn, template) in ops.items():
        for a in range(2, 8):
            for b in range(2, 8):
                if a == b:
                    continue
                result = fn(a, b)
                if result < 0 or result >= 10:
                    continue
                expr = template.format(a=a, b=b)
                prompt = f"def f(): return {expr} ="
                prompts.append(prompt)
                labels_a.append(a)
                labels_b.append(b)
                labels_op.append(op_name)
                labels_result.append(result)

    print(f"  Dataset: {len(prompts)} samples")

    # Extract register vectors at key layers
    PROBE_LAYERS = [0, 2, 4, 8, 13, 16, 20, 22]
    vecs_by_layer = {l: [] for l in PROBE_LAYERS}

    print("  Extracting register vectors...")
    for prompt in prompts:
        for layer in PROBE_LAYERS:
            cap = [None]
            def capture(module, input, output):
                cap[0] = get_last_token(output)
            h = model.model.layers[layer].register_forward_hook(capture)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            vecs_by_layer[layer].append(cap[0].float().cpu().numpy().flatten())

    # Probe: can we decode a, b, op, result from each layer?
    print("\n  Probing register contents...")
    probe_results = {}

    for target_name, target_labels in [('operand_a', labels_a), ('operand_b', labels_b),
                                        ('operation', labels_op), ('result', labels_result)]:
        layer_accs = {}
        for layer in PROBE_LAYERS:
            X = np.array(vecs_by_layer[layer])
            y = target_labels
            try:
                clf = LogisticRegression(max_iter=500, random_state=42)
                scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
                acc = round(float(scores.mean()), 4)
            except Exception:
                acc = 0.0
            layer_accs[layer] = acc

        best_layer = max(layer_accs, key=layer_accs.get)
        probe_results[target_name] = {
            'layer_accs': {str(l): a for l, a in layer_accs.items()},
            'best_layer': best_layer,
            'best_acc': layer_accs[best_layer],
        }
        print(f"    {target_name}: best L{best_layer} = {layer_accs[best_layer]:.1%}")
        for l in PROBE_LAYERS:
            print(f"      L{l}: {layer_accs[l]:.1%}")

    # Save
    output = {
        'phase': 41, 'name': 'register_fingerprinting',
        'n_samples': len(prompts),
        'probe_results': probe_results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase41_fingerprint.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (target_name, ax) in enumerate(zip(
            ['operand_a', 'operand_b', 'operation', 'result'], axes.flat)):
        pr = probe_results[target_name]
        layers = [int(l) for l in pr['layer_accs'].keys()]
        accs = list(pr['layer_accs'].values())
        best_l = pr['best_layer']
        colors = ['tab:green' if l == best_l else 'tab:blue' for l in layers]
        ax.bar([f'L{l}' for l in layers], accs, color=colors, edgecolor='black')
        ax.set_ylabel('Probe Accuracy')
        ax.set_title(f'{target_name} (best: L{best_l}={pr["best_acc"]:.0%})',
                    fontweight='bold')
        ax.set_ylim(0, 1.1)
        for i, v in enumerate(accs):
            ax.text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=7)

    plt.suptitle('Phase 41: Register Fingerprinting\nCan we read memory contents (a, b, op, result) from register state?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase41_fingerprint.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
