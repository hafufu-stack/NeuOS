# -*- coding: utf-8 -*-
"""
Phase 22: Direct Memory Access (DMA) Execution
Can we execute a program by injecting ONLY an L16 register vector,
without ANY textual instruction? Proves: data = program.

Method:
  1. Extract "MIN execution vector" from L16 in a sorting context
  2. Feed raw data "3, 7" with NO task instruction
  3. Inject the MIN vector at L16 during inference
  4. Check if output = min(3, 7) = 3

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


def extract_register(model, tok, prompts, layer):
    """Extract average hidden state at a specific layer from multiple prompts."""
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
    print("[P22] Direct Memory Access (DMA) Execution")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)

    # === Step 1: Extract execution vectors for different operations ===
    print("  Step 1: Extracting execution vectors...")
    
    # MIN at L16
    min_prompts = [f"def f(): return min({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:15]
    min_vec = extract_register(model, tok, min_prompts, layer=16)
    
    # MAX at L22
    max_prompts = [f"def f(): return max({a}, {b}) =" for a in range(2, 8) for b in range(2, 8) if a != b][:15]
    max_vec = extract_register(model, tok, max_prompts, layer=22)
    
    # SUM at L20
    sum_prompts = [f"def f(): return {a} + {b} =" for a in range(1, 6) for b in range(1, 6) if a+b < 10][:15]
    sum_vec = extract_register(model, tok, sum_prompts, layer=20)

    print(f"    Extracted MIN(L16), MAX(L22), SUM(L20) vectors")

    # === Step 2: DMA test - inject execution vector into raw data ===
    print("\n  Step 2: DMA Execution on raw data...")
    
    # Test data: "X, Y) =" with no function context
    test_cases = [
        ("3, 7) =", 3, 7),
        ("2, 8) =", 2, 8),
        ("5, 1) =", 5, 1),
        ("9, 4) =", 9, 4),
        ("6, 3) =", 6, 3),
        ("4, 4) =", 4, 4),
    ]

    # Control: no injection
    print("    Control (no injection):")
    control_preds = []
    for data_str, a, b in test_cases:
        inp = tok(data_str, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        control_preds.append(pred)
    print(f"      Predictions: {control_preds}")

    # DMA: inject MIN vector at L16
    dma_results = {}
    for op_name, vec, layer, expected_fn in [
        ('MIN', min_vec, 16, min),
        ('MAX', max_vec, 22, max),
        ('SUM', sum_vec, 20, lambda a, b: a + b),
    ]:
        correct = 0
        preds = []
        for data_str, a, b in test_cases:
            expected = expected_fn(a, b)
            if expected >= 10:  # skip multi-digit
                continue

            def inject(module, input, output, v=vec):
                return replace_last_token(output, v)

            h = model.model.layers[layer].register_forward_hook(inject)
            inp = tok(data_str, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()

            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            preds.append(pred)
            if pred == str(expected):
                correct += 1

        n_valid = sum(1 for _, a, b in test_cases if expected_fn(a, b) < 10)
        acc = correct / n_valid if n_valid > 0 else 0
        dma_results[op_name] = {
            'accuracy': round(acc, 4),
            'predictions': preds,
            'layer': layer,
        }
        print(f"    DMA {op_name}(L{layer}): {acc:.1%} predictions={preds}")

    # === Step 3: Cross-injection (wrong layer) ===
    print("\n  Step 3: Cross-injection (MIN vec at L22 instead of L16)...")
    cross_correct = 0
    for data_str, a, b in test_cases:
        expected_min = min(a, b)
        def inject(module, input, output, v=min_vec):
            return replace_last_token(output, v)
        h = model.model.layers[22].register_forward_hook(inject)  # Wrong layer!
        inp = tok(data_str, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(expected_min):
            cross_correct += 1
    cross_acc = cross_correct / len(test_cases)
    print(f"    Cross-injection (MIN at L22): {cross_acc:.1%}")

    # Save
    output = {
        'phase': 22, 'name': 'dma_execution',
        'dma_results': {k: v['accuracy'] for k, v in dma_results.items()},
        'cross_injection': round(cross_acc, 4),
        'control_predictions': control_preds,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase22_dma.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ops = list(dma_results.keys())
    accs = [dma_results[op]['accuracy'] for op in ops]
    colors = ['tab:green', 'tab:red', 'tab:blue']
    bars = axes[0].bar(ops, accs, color=colors, edgecolor='black')
    axes[0].bar(['Cross\n(wrong layer)'], [cross_acc], color='tab:gray', edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('DMA Execution: No Instructions Needed', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(accs + [cross_acc]):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    axes[1].axis('off')
    best_op = max(dma_results, key=lambda k: dma_results[k]['accuracy'])
    summary = (
        f"Direct Memory Access\n\n"
        f"No text instruction given!\n"
        f"Only raw data: '3, 7) ='\n\n"
    )
    for op in ops:
        summary += f"  DMA {op}(L{dma_results[op]['layer']}): {dma_results[op]['accuracy']:.0%}\n"
    summary += f"\n  Cross (wrong layer): {cross_acc:.0%}\n\n"
    summary += "Data IS Program!" if any(v['accuracy'] > 0.3 for v in dma_results.values()) else "Layer-specific"
    axes[1].text(0.5, 0.5, summary, transform=axes[1].transAxes,
                 fontsize=12, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 22: DMA Execution\nCan we run programs without ANY text instructions?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase22_dma.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
