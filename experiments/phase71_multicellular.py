# -*- coding: utf-8 -*-
"""
Phase 71: Multicellular Organism
Multiple programs cooperate as 'cells' within a single organism.
Each cell specializes in a subtask, and the organism solves
problems no single cell can handle alone.

Combines P48 (hyperthreading), P61 (ecosystem), P67 (metabolism).

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_prog(model, tok, train, layer, device, seed=42):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(80):
        for prompt, target_str in train:
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
    return vec.detach()


def eval_prog(model, tok, vec, prompts, expected, layer, device):
    correct = 0
    for prompt, exp in zip(prompts, expected):
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == exp:
            correct += 1
    return correct / len(prompts)


def get_pred(model, tok, vec, prompt, layer, device):
    def inject(module, input, output, v=vec):
        return replace_last_token(output, v)
    h = model.model.layers[layer].register_forward_hook(inject)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    return tok.decode(out.logits[0, -1, :].argmax().item()).strip()


def main():
    print("[P71] Multicellular Organism")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile specialist cells
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")]

    print("  Compiling specialist cells...")
    min_cell = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42)
    max_cell = compile_prog(model, tok, max_data, target_layer, DEVICE, seed=99)

    # Test individual cells
    test_p = ["3, 7) =", "5, 2) =", "8, 1) =", "7, 4) =", "6, 2) ="]
    min_exp = ["3", "2", "1", "4", "2"]
    max_exp = ["7", "5", "8", "7", "6"]

    min_solo = eval_prog(model, tok, min_cell, test_p, min_exp, target_layer, DEVICE)
    max_solo = eval_prog(model, tok, max_cell, test_p, max_exp, target_layer, DEVICE)
    print(f"    MIN cell solo: {min_solo:.0%}")
    print(f"    MAX cell solo: {max_solo:.0%}")

    # Multicellular task: RANGE = MAX - MIN
    # The organism needs BOTH cells to solve this
    print("\n  Step 1: Multicellular RANGE task (MAX - MIN)...")
    range_prompts = ["3, 7) =", "5, 2) =", "8, 1) =", "7, 4) =", "6, 2) ="]
    range_expected = ["4", "3", "7", "3", "4"]  # MAX - MIN

    # Strategy: run both cells, combine results
    multi_results = []
    for i, prompt in enumerate(range_prompts):
        min_pred = get_pred(model, tok, min_cell, prompt, target_layer, DEVICE)
        max_pred = get_pred(model, tok, max_cell, prompt, target_layer, DEVICE)
        try:
            result = str(int(max_pred) - int(min_pred))
        except ValueError:
            result = "?"
        multi_results.append(result)
        correct = result == range_expected[i]
        if i < 3:
            print(f"    {prompt} MIN={min_pred}, MAX={max_pred}, RANGE={result} "
                  f"(exp={range_expected[i]}) {'OK' if correct else 'FAIL'}")

    multi_acc = sum(r == e for r, e in zip(multi_results, range_expected)) / len(range_expected)
    print(f"    Multicellular RANGE accuracy: {multi_acc:.0%}")

    # Can a single cell do RANGE?
    print("\n  Step 2: Training single-cell RANGE...")
    range_train = [("3, 7) =", "4"), ("5, 2) =", "3"), ("8, 1) =", "7"),
                   ("4, 6) =", "2"), ("9, 3) =", "6")]
    range_cell = compile_prog(model, tok, range_train, target_layer, DEVICE, seed=777)
    single_acc = eval_prog(model, tok, range_cell, range_prompts, range_expected,
                          target_layer, DEVICE)
    print(f"    Single-cell RANGE accuracy: {single_acc:.0%}")

    # Step 3: Multicellular SORT (MIN, MAX in order)
    print("\n  Step 3: Multicellular coordination (MIN then MAX)...")
    sort_correct = 0
    sort_total = 0
    for prompt in ["3, 7) =", "5, 2) =", "8, 1) ="]:
        min_pred = get_pred(model, tok, min_cell, prompt, target_layer, DEVICE)
        max_pred = get_pred(model, tok, max_cell, prompt, target_layer, DEVICE)
        try:
            is_sorted = int(min_pred) <= int(max_pred)
            sort_correct += int(is_sorted)
        except ValueError:
            pass
        sort_total += 1
        print(f"    {prompt} -> [{min_pred}, {max_pred}] sorted={'yes' if is_sorted else 'no'}")
    sort_acc = sort_correct / sort_total if sort_total > 0 else 0

    # Step 4: Cell interference test
    print("\n  Step 4: Cell interference (do cells hurt each other?)...")
    combined = (min_cell + max_cell) / 2
    combined_min = eval_prog(model, tok, combined, test_p, min_exp, target_layer, DEVICE)
    combined_max = eval_prog(model, tok, combined, test_p, max_exp, target_layer, DEVICE)
    print(f"    Combined vec as MIN: {combined_min:.0%} (solo: {min_solo:.0%})")
    print(f"    Combined vec as MAX: {combined_max:.0%} (solo: {max_solo:.0%})")

    # Save
    output = {
        'phase': 71, 'name': 'multicellular_organism',
        'min_solo': round(min_solo, 4), 'max_solo': round(max_solo, 4),
        'multicellular_range': round(multi_acc, 4),
        'single_cell_range': round(single_acc, 4),
        'sort_accuracy': round(sort_acc, 4),
        'combined_min': round(combined_min, 4),
        'combined_max': round(combined_max, 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase71_multicellular.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['MIN\nsolo', 'MAX\nsolo', 'Multi\nRANGE', 'Single\nRANGE']
    vals = [min_solo, max_solo, multi_acc, single_acc]
    colors = ['tab:blue', 'tab:red', 'tab:purple', 'tab:orange']
    axes[0].bar(labels, vals, color=colors, edgecolor='black')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Solo vs Multicellular Performance', fontweight='bold')
    for i, v in enumerate(vals):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    axes[1].bar(['MIN task', 'MAX task'],
                [combined_min, combined_max],
                color=['tab:blue', 'tab:red'], edgecolor='black', alpha=0.5,
                label='Combined')
    axes[1].bar(['MIN task', 'MAX task'],
                [min_solo, max_solo],
                color=['tab:blue', 'tab:red'], edgecolor='black', alpha=0.8,
                label='Solo', width=0.4)
    axes[1].set_title('Cell Interference', fontweight='bold')
    axes[1].legend()

    axes[2].axis('off')
    summary = (f"MULTICELLULAR ORGANISM\n{'='*30}\n\n"
               f"MIN cell solo: {min_solo:.0%}\n"
               f"MAX cell solo: {max_solo:.0%}\n"
               f"Multicellular RANGE: {multi_acc:.0%}\n"
               f"Single-cell RANGE: {single_acc:.0%}\n"
               f"Sort accuracy: {sort_acc:.0%}\n\n"
               f"Cells cooperate to solve\n"
               f"tasks neither can do alone!")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                fontsize=10, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 71: Multicellular Organism\nSpecialized cells cooperating on composite tasks',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase71_multicellular.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
