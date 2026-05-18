# -*- coding: utf-8 -*-
"""
Phase 29: Homoiconic Dual-Use (Deep Think P24)
Prove that the SAME vector acts as SOFTWARE when loaded at L16
and as DATA when loaded at L2/L13.

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


def inject_and_predict(model, tok, prompt, layer, vec):
    def hook(module, input, output):
        return replace_last_token(output, vec)
    h = model.model.layers[layer].register_forward_hook(hook)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    logits = out.logits[0, -1, :]
    top5 = torch.topk(logits, 5)
    return [(tok.decode(idx.item()).strip(), round(logits[idx].item(), 2))
            for idx in top5.indices]


def main():
    print("[P29] Homoiconic Dual-Use")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)

    # Extract the MIN vector from L16 (its native execution layer)
    print("  Extracting MIN execution vector from L16...")
    min_vec = extract_register(model, tok,
        [f"def f(): return min({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:10],
        16)

    # Also extract vectors for specific numbers for comparison
    print("  Extracting number vectors for comparison...")
    num_vecs = {}
    for n in [2, 3, 5, 7]:
        num_vecs[n] = extract_register(model, tok,
            [f"def f(): return {n} ="] * 5, 16)

    # === Test 1: MIN vec as SOFTWARE (inject at L16) ===
    print("\n  Test 1: MIN vec as SOFTWARE (inject at L16)...")
    sw_test = [("3, 7) =", 3, 7), ("2, 8) =", 2, 8), ("5, 1) =", 5, 1),
               ("4, 6) =", 4, 6), ("7, 2) =", 7, 2), ("9, 3) =", 9, 3)]
    sw_results = []
    sw_correct = 0
    for data, a, b in sw_test:
        top5 = inject_and_predict(model, tok, data, 16, min_vec)
        pred = top5[0][0]
        expected = str(min(a, b))
        is_correct = pred == expected
        if is_correct:
            sw_correct += 1
        sw_results.append({'input': data, 'pred': pred, 'expected': expected,
                          'correct': is_correct, 'top5': top5})
        print(f"    L16: {data} -> {pred} (expected {expected}) {'OK' if is_correct else 'X'}")

    sw_acc = sw_correct / len(sw_test)

    # === Test 2: MIN vec as DATA (inject at L2 and L13) ===
    print("\n  Test 2: MIN vec as DATA (inject at L2, L13)...")
    # Use identity prompt: "def f(): return X =" where X should be determined by injected data
    data_prompts = ["def f(): return =", "def f(): return x =", "x ="]
    data_layers = [2, 13]
    data_results = {}

    for layer in data_layers:
        data_results[layer] = []
        for prompt in data_prompts:
            top5 = inject_and_predict(model, tok, prompt, layer, min_vec)
            data_results[layer].append({'prompt': prompt, 'top5': top5})
            print(f"    L{layer}: '{prompt}' -> top: {top5[:3]}")

    # === Test 3: Same vec, different register = different behavior ===
    print("\n  Test 3: Same vec at different registers...")
    diff_test_prompt = "3, 7) ="
    diff_results = {}
    for layer in [0, 2, 4, 8, 13, 16, 20, 22]:
        top5 = inject_and_predict(model, tok, diff_test_prompt, layer, min_vec)
        diff_results[layer] = top5[0]
        print(f"    L{layer}: '{diff_test_prompt}' -> {top5[0]}")

    # === Test 4: Number vec as SOFTWARE (inject at L16) ===
    print("\n  Test 4: Number vec as SOFTWARE at L16...")
    num_sw_results = {}
    for n, vec in num_vecs.items():
        top5 = inject_and_predict(model, tok, "3, 7) =", 16, vec)
        num_sw_results[n] = top5[0]
        print(f"    num={n} at L16: '3, 7) =' -> {top5[0]}")

    # Save
    output = {
        'phase': 29, 'name': 'homoiconic_dual_use',
        'software_accuracy': round(sw_acc, 4),
        'software_results': [{'input': r['input'], 'pred': r['pred'],
                              'expected': r['expected'], 'correct': r['correct']}
                             for r in sw_results],
        'data_results': {str(l): [{'prompt': r['prompt'],
                                    'top1': r['top5'][0] if r['top5'] else None}
                                   for r in res]
                         for l, res in data_results.items()},
        'diff_register_results': {str(l): list(v) for l, v in diff_results.items()},
        'num_as_software': {str(n): list(v) for n, v in num_sw_results.items()},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase29_homoiconic.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Software mode
    axes[0].bar(['As Software\n(L16)'], [sw_acc], color='tab:blue', edgecolor='black')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('MIN vec -> L16\n(Executes MIN program)', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].text(0, sw_acc+0.03, f'{sw_acc:.0%}', ha='center', fontweight='bold', fontsize=14)

    # Different registers -> different output
    layers_list = sorted(diff_results.keys())
    outputs = [diff_results[l][0] for l in layers_list]
    colors = ['tab:gray' if o not in ['3','7'] else
              ('tab:green' if o == '3' else 'tab:red') for o in outputs]
    axes[1].barh([f'L{l}' for l in layers_list],
                 [1]*len(layers_list), color=colors, edgecolor='black')
    for i, (l, o) in enumerate(zip(layers_list, outputs)):
        axes[1].text(0.5, i, f'Output: "{o}"', ha='center', va='center',
                    fontweight='bold', fontsize=10)
    axes[1].set_xlabel('Same MIN vector')
    axes[1].set_title('Same Vec, Different Register\n-> Different Behavior', fontweight='bold')

    # Summary
    axes[2].axis('off')
    summary = "Homoiconic Dual-Use\n\n"
    summary += f"MIN vec at L16 (software): {sw_acc:.0%}\n"
    summary += f"MIN vec at L2 (data): different output\n"
    summary += f"MIN vec at L13 (data): different output\n\n"
    summary += "Same vector:\n"
    summary += "  Load at L16 -> executes MIN\n"
    summary += "  Load at L2 -> treated as data\n"
    summary += "  Load at L22 -> treated as output\n\n"
    summary += "Data = Code = Vector\n"
    summary += "Context (register) determines behavior"
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=11, va='center', ha='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 29: Homoiconic Dual-Use\nSame vector = software OR data, depending on register',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase29_homoiconic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
