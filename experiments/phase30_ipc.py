# -*- coding: utf-8 -*-
"""
Phase 30: Neural IPC (Deep Think P28)
Inter-Process Communication: pipe output of process A (a+b)
directly to input of process B (+c) via latent vectors.
No text decoding in between.

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


def main():
    print("[P30] Neural IPC (Inter-Process Communication)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)

    # Test cases: a + b + c where a+b < 10 and (a+b)+c < 10
    test_cases = [
        (1, 2, 3, 6), (2, 3, 1, 6), (1, 1, 1, 3), (3, 2, 4, 9),
        (2, 1, 3, 6), (1, 3, 2, 6), (4, 1, 2, 7), (2, 2, 1, 5),
    ]

    results = {'direct': [], 'ipc': [], 'ipc_multi': []}

    # Method 1: Direct text (baseline) - "def f(): return a + b + c ="
    print("\n  Method 1: Direct text (a+b+c)...")
    for a, b, c, expected in test_cases:
        prompt = f"def f(): return {a} + {b} + {c} ="
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        results['direct'].append({'a': a, 'b': b, 'c': c, 'expected': expected,
                                  'pred': pred, 'correct': pred == str(expected)})

    direct_acc = sum(r['correct'] for r in results['direct']) / len(results['direct'])
    print(f"    Direct accuracy: {direct_acc:.1%}")

    # Method 2: IPC via latent pipe
    # Step A: Run "def f(): return a + b =" -> capture L20 (SUM register)
    # Step B: Inject captured L20 vec into L2 (data register) of "+ c ="
    print("\n  Method 2: Neural IPC (L20 -> L2 pipe)...")
    for a, b, c, expected in test_cases:
        # Process A: compute a + b
        prompt_a = f"def f(): return {a} + {b} ="
        captured_sum = [None]
        def cap_sum(module, input, output):
            captured_sum[0] = get_last_token(output)
        h = model.model.layers[20].register_forward_hook(cap_sum)
        inp_a = tok(prompt_a, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp_a)
        h.remove()
        sum_vec = captured_sum[0]

        # Process B: inject sum_vec at L2 (data) for "+ c ="
        prompt_b = f"def f(): return x + {c} ="
        def inject_data(module, input, output, v=sum_vec):
            return replace_last_token(output, v)
        h = model.model.layers[2].register_forward_hook(inject_data)
        inp_b = tok(prompt_b, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out_b = model(**inp_b)
        h.remove()
        pred = tok.decode(out_b.logits[0, -1, :].argmax().item()).strip()
        results['ipc'].append({'a': a, 'b': b, 'c': c, 'expected': expected,
                               'pred': pred, 'correct': pred == str(expected)})

    ipc_acc = sum(r['correct'] for r in results['ipc']) / len(results['ipc'])
    print(f"    IPC accuracy (L20->L2): {ipc_acc:.1%}")

    # Method 3: IPC with multi-layer injection (L20 -> L2 AND L13)
    print("\n  Method 3: IPC multi-pipe (L20 -> L2+L13)...")
    for a, b, c, expected in test_cases:
        prompt_a = f"def f(): return {a} + {b} ="
        captured_vecs = {}
        for capture_layer in [13, 20]:
            cap = [None]
            def make_cap(c):
                def fn(module, input, output):
                    c[0] = get_last_token(output)
                return fn
            h = model.model.layers[capture_layer].register_forward_hook(make_cap(cap))
            inp_a = tok(prompt_a, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp_a)
            h.remove()
            captured_vecs[capture_layer] = cap[0]

        prompt_b = f"def f(): return x + {c} ="
        hooks = []
        for inject_layer, vec in [(2, captured_vecs[20]), (13, captured_vecs[13])]:
            def make_hook(v):
                def hook(module, input, output):
                    return replace_last_token(output, v)
                return hook
            h = model.model.layers[inject_layer].register_forward_hook(make_hook(vec))
            hooks.append(h)

        inp_b = tok(prompt_b, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out_b = model(**inp_b)
        for h in hooks:
            h.remove()
        pred = tok.decode(out_b.logits[0, -1, :].argmax().item()).strip()
        results['ipc_multi'].append({'a': a, 'b': b, 'c': c, 'expected': expected,
                                     'pred': pred, 'correct': pred == str(expected)})

    ipc_multi_acc = sum(r['correct'] for r in results['ipc_multi']) / len(results['ipc_multi'])
    print(f"    IPC multi-pipe accuracy: {ipc_multi_acc:.1%}")

    # Save
    output = {
        'phase': 30, 'name': 'neural_ipc',
        'direct_acc': round(direct_acc, 4),
        'ipc_acc': round(ipc_acc, 4),
        'ipc_multi_acc': round(ipc_multi_acc, 4),
        'results': {k: v for k, v in results.items()},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase30_ipc.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    methods = ['Direct\n(a+b+c)', 'IPC Single\n(L20->L2)', 'IPC Multi\n(L20->L2+L13)']
    accs = [direct_acc, ipc_acc, ipc_multi_acc]
    colors = ['tab:gray', 'tab:blue', 'tab:green']
    ax.bar(methods, accs, color=colors, edgecolor='black')
    ax.set_ylabel('Accuracy')
    ax.set_title('Phase 30: Neural IPC\nCan latent pipe replace text for chained computation?',
                 fontweight='bold')
    ax.set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        ax.text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase30_ipc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
