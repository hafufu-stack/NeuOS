# -*- coding: utf-8 -*-
"""
Phase 36: 1.5B ISA with Surgery
P32 showed 1.5B gets 0% without embedding surgery.
Apply surgery to 1.5B and recheck ISA + DMA.

Model: Qwen2.5-1.5B (GPU)
"""
import torch, json, os, gc, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import apply_surgery, get_last_token, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P36] 1.5B ISA with Surgery")
    print(f"  Device: {DEVICE}")
    start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-1.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).to(DEVICE)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"  Model: Qwen2.5-1.5B ({n_layers} layers)")

    # === Step 1: Without surgery ===
    print("\n  Step 1: WITHOUT surgery...")
    no_surg = {}
    for op, prompts, fn in [
        ('MIN', [f"def f(): return min({a}, {b}) =" for a in [3,5,7] for b in [2,4,6] if a!=b], min),
        ('MAX', [f"def f(): return max({a}, {b}) =" for a in [3,5,7] for b in [2,4,6] if a!=b], max),
        ('SUM', [f"def f(): return {a} + {b} =" for a in [1,2,3] for b in [1,2,3] if a+b<10],
         lambda a,b: a+b),
    ]:
        correct = 0
        total = len(prompts)
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            expr = prompt.split('return ')[-1].split(' =')[0]
            try:
                expected = str(eval(expr))
            except Exception:
                expected = ""
            if pred == expected:
                correct += 1
        no_surg[op] = round(correct/total, 4)
        print(f"    {op}: {correct}/{total} = {correct/total:.0%}")

    # === Step 2: WITH surgery ===
    print("\n  Step 2: WITH surgery...")
    apply_surgery(model, tok, strength=2.0)

    with_surg = {}
    for op, prompts, fn in [
        ('MIN', [f"def f(): return min({a}, {b}) =" for a in [3,5,7] for b in [2,4,6] if a!=b], min),
        ('MAX', [f"def f(): return max({a}, {b}) =" for a in [3,5,7] for b in [2,4,6] if a!=b], max),
        ('SUM', [f"def f(): return {a} + {b} =" for a in [1,2,3] for b in [1,2,3] if a+b<10],
         lambda a,b: a+b),
    ]:
        correct = 0
        total = len(prompts)
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            expr = prompt.split('return ')[-1].split(' =')[0]
            try:
                expected = str(eval(expr))
            except Exception:
                expected = ""
            if pred == expected:
                correct += 1
        with_surg[op] = round(correct/total, 4)
        print(f"    {op}: {correct}/{total} = {correct/total:.0%}")

    # === Step 3: Register probe at key layers ===
    print("\n  Step 3: Register probe (1.5B with surgery)...")
    from sklearn.linear_model import LogisticRegression

    probe_layers = list(range(0, n_layers, 4))  # sample every 4 layers
    probe_prompts = [f"def f(): return min({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:20]

    min_best_layer = -1
    min_best_acc = 0
    probe_results = {}

    for layer in probe_layers:
        vecs = []
        labels = []
        for prompt in probe_prompts:
            a = int(prompt.split('min(')[1].split(',')[0])
            b = int(prompt.split(', ')[1].split(')')[0])
            expected_min = min(a, b)

            cap = [None]
            def capture(module, input, output):
                cap[0] = get_last_token(output)
            h = model.model.layers[layer].register_forward_hook(capture)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            vecs.append(cap[0].float().cpu().numpy().flatten())
            labels.append(expected_min)

        if len(set(labels)) >= 2:
            from sklearn.model_selection import cross_val_score
            X = [v for v in vecs]
            y = labels
            try:
                clf = LogisticRegression(max_iter=500)
                scores = cross_val_score(clf, X, y, cv=min(3, len(set(y))), scoring='accuracy')
                acc = round(float(scores.mean()), 4)
            except Exception:
                acc = 0.0
        else:
            acc = 0.0

        probe_results[layer] = acc
        if acc > min_best_acc:
            min_best_acc = acc
            min_best_layer = layer

    print(f"    MIN register best: L{min_best_layer} ({min_best_acc:.0%})")

    # === Step 4: DMA on 1.5B with surgery ===
    print("\n  Step 4: DMA test (1.5B with surgery)...")
    mid_layer = n_layers // 2
    min_vecs_1_5b = []
    for p in probe_prompts[:8]:
        cap = [None]
        def capture2(module, input, output):
            cap[0] = get_last_token(output)
        h = model.model.layers[mid_layer].register_forward_hook(capture2)
        inp = tok(p, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        min_vecs_1_5b.append(cap[0])
    min_vec = torch.stack(min_vecs_1_5b).mean(dim=0)

    dma_data = [("3, 7) =", 3, 7), ("5, 2) =", 5, 2), ("8, 1) =", 8, 1),
                ("4, 6) =", 4, 6), ("9, 3) =", 9, 3), ("7, 2) =", 7, 2)]
    dma_correct = 0
    dma_total = 0
    for data_str, a, b in dma_data:
        expected = min(a, b)
        dma_total += 1
        def inject(module, input, output, v=min_vec):
            return replace_last_token(output, v)
        h = model.model.layers[mid_layer].register_forward_hook(inject)
        inp = tok(data_str, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(expected):
            dma_correct += 1
    dma_acc = dma_correct / dma_total
    print(f"    1.5B DMA (MIN@L{mid_layer}): {dma_acc:.1%}")

    # Save
    output = {
        'phase': 36, 'name': '1_5b_isa_surgery',
        'model': 'Qwen2.5-1.5B', 'n_layers': n_layers,
        'without_surgery': no_surg,
        'with_surgery': with_surg,
        'probe_results': {str(k): v for k, v in probe_results.items()},
        'min_best_layer': min_best_layer, 'min_best_acc': min_best_acc,
        'dma_acc': round(dma_acc, 4), 'dma_layer': mid_layer,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase36_1_5b_surgery.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Surgery comparison
    ops = list(no_surg.keys())
    x = range(len(ops))
    axes[0].bar([i-0.15 for i in x], [no_surg[op] for op in ops], 0.3,
                label='No Surgery', color='tab:red', edgecolor='black')
    axes[0].bar([i+0.15 for i in x], [with_surg[op] for op in ops], 0.3,
                label='With Surgery', color='tab:green', edgecolor='black')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(ops)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('1.5B: Surgery Effect', fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)

    # Probe
    layers_p = sorted(probe_results.keys())
    accs_p = [probe_results[l] for l in layers_p]
    axes[1].plot([f'L{l}' for l in layers_p], accs_p, 'bo-', linewidth=2)
    axes[1].set_ylabel('Probe Accuracy')
    axes[1].set_title(f'1.5B MIN Register (best: L{min_best_layer}={min_best_acc:.0%})',
                      fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # DMA comparison
    axes[2].bar(['0.5B P22\n(no surgery)', '1.5B P32\n(no surgery)', '1.5B P36\n(surgery)'],
                [0.667, 0.0, dma_acc],
                color=['tab:orange', 'tab:red', 'tab:blue'], edgecolor='black')
    axes[2].set_ylabel('DMA Accuracy')
    axes[2].set_title('DMA Across Models', fontweight='bold')
    axes[2].set_ylim(0, 1.1)

    plt.suptitle('Phase 36: 1.5B ISA with Surgery\nDoes the register architecture scale?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase36_1_5b_surgery.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
