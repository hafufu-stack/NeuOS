# -*- coding: utf-8 -*-
"""
Phase 27: Multi-Port DMA (Deep Think P27)
Fix MAX/SUM DMA failure by simultaneous multi-layer injection.

Hypothesis: MAX/SUM fail because they depend on flag registers (CARRY@L4,
COMPARISON@L14) that aren't set when injecting at a single layer.
Solution: Inject MULTIPLE registers simultaneously.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, time, sys
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


def extract_register(model, tok, prompts, layer):
    vecs = []
    for prompt in prompts:
        captured = [None]
        def cap(module, input, output):
            captured[0] = get_last_token(output)
        h = model.model.layers[layer].register_forward_hook(cap)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        vecs.append(captured[0])
    return torch.stack(vecs).mean(dim=0)


def main():
    print("[P27] Multi-Port DMA")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)

    # Extract register vectors for each operation at MULTIPLE layers
    print("  Extracting multi-layer register vectors...")
    ops = {
        'MIN': {'prompts': [f"def f(): return min({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:10],
                'fn': min, 'key_layers': [16]},
        'MAX': {'prompts': [f"def f(): return max({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:10],
                'fn': max, 'key_layers': [4, 14, 22]},
        'SUM': {'prompts': [f"def f(): return {a} + {b} =" for a in range(1,6) for b in range(1,6) if a+b<10][:10],
                'fn': lambda a,b: a+b, 'key_layers': [2, 4, 13, 20]},
    }

    # For each op, extract vectors at all key layers
    reg_vecs = {}
    for op_name, info in ops.items():
        reg_vecs[op_name] = {}
        for layer in info['key_layers']:
            reg_vecs[op_name][layer] = extract_register(model, tok, info['prompts'], layer)
        # Also extract at ALL layers for the full pipeline injection
        for layer in [0, 2, 4, 8, 13, 14, 16, 18, 20, 22]:
            if layer not in reg_vecs[op_name]:
                reg_vecs[op_name][layer] = extract_register(model, tok, info['prompts'], layer)
        print(f"    {op_name}: {len(reg_vecs[op_name])} layers extracted")

    test_data = [
        ("3, 7) =", 3, 7), ("2, 8) =", 2, 8), ("5, 1) =", 5, 1),
        ("9, 4) =", 9, 4), ("6, 3) =", 6, 3), ("4, 2) =", 4, 2),
    ]

    results = {}

    # Test configurations: single-port vs multi-port
    configs = {
        'MIN_L16': {'op': 'MIN', 'layers': [16]},
        'MAX_L22_only': {'op': 'MAX', 'layers': [22]},
        'MAX_L14_L22': {'op': 'MAX', 'layers': [14, 22]},
        'MAX_L4_L14_L22': {'op': 'MAX', 'layers': [4, 14, 22]},
        'MAX_full_pipe': {'op': 'MAX', 'layers': [0, 2, 4, 8, 13, 14, 16, 18, 20, 22]},
        'SUM_L20_only': {'op': 'SUM', 'layers': [20]},
        'SUM_L4_L13_L20': {'op': 'SUM', 'layers': [4, 13, 20]},
        'SUM_L2_L4_L13_L20': {'op': 'SUM', 'layers': [2, 4, 13, 20]},
        'SUM_full_pipe': {'op': 'SUM', 'layers': [0, 2, 4, 8, 13, 14, 16, 18, 20, 22]},
    }

    print("\n  Testing configurations...")
    for cfg_name, cfg in configs.items():
        op_name = cfg['op']
        inject_layers = cfg['layers']
        fn = ops[op_name]['fn']
        correct = 0
        total = 0
        preds = []

        for data_str, a, b in test_data:
            expected = fn(a, b)
            if expected >= 10:
                continue
            total += 1
            hooks = []
            for layer in inject_layers:
                vec = reg_vecs[op_name][layer]
                def make_hook(v):
                    def hook(module, input, output):
                        return replace_last_token(output, v)
                    return hook
                h = model.model.layers[layer].register_forward_hook(make_hook(vec))
                hooks.append(h)

            inp = tok(data_str, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            for h in hooks:
                h.remove()

            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            preds.append(pred)
            if pred == str(expected):
                correct += 1

        acc = correct / total if total > 0 else 0
        results[cfg_name] = {'accuracy': round(acc, 4), 'predictions': preds,
                             'layers': inject_layers, 'op': op_name}
        print(f"    {cfg_name}: {acc:.1%} ({len(inject_layers)} ports)")

    # Save
    output = {'phase': 27, 'name': 'multi_port_dma', 'results': results,
              'elapsed': round(time.time()-start, 1)}
    with open(os.path.join(RESULTS_DIR, 'phase27_multiport.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # MAX progression
    max_cfgs = ['MAX_L22_only', 'MAX_L14_L22', 'MAX_L4_L14_L22', 'MAX_full_pipe']
    max_accs = [results[c]['accuracy'] for c in max_cfgs]
    max_ports = [len(results[c]['layers']) for c in max_cfgs]
    axes[0].bar(range(len(max_cfgs)), max_accs, color=['tab:red','tab:orange','tab:green','tab:blue'],
                edgecolor='black')
    axes[0].set_xticks(range(len(max_cfgs)))
    axes[0].set_xticklabels([f"{p} port(s)" for p in max_ports], fontsize=9)
    axes[0].set_ylabel('DMA Accuracy')
    axes[0].set_title('MAX: Single vs Multi-Port DMA', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(max_accs):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # SUM progression
    sum_cfgs = ['SUM_L20_only', 'SUM_L4_L13_L20', 'SUM_L2_L4_L13_L20', 'SUM_full_pipe']
    sum_accs = [results[c]['accuracy'] for c in sum_cfgs]
    sum_ports = [len(results[c]['layers']) for c in sum_cfgs]
    axes[1].bar(range(len(sum_cfgs)), sum_accs, color=['tab:red','tab:orange','tab:green','tab:blue'],
                edgecolor='black')
    axes[1].set_xticks(range(len(sum_cfgs)))
    axes[1].set_xticklabels([f"{p} port(s)" for p in sum_ports], fontsize=9)
    axes[1].set_ylabel('DMA Accuracy')
    axes[1].set_title('SUM: Single vs Multi-Port DMA', fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate(sum_accs):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    plt.suptitle('Phase 27: Multi-Port DMA\nDoes injecting multiple registers fix MAX/SUM?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase27_multiport.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
