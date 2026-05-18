# -*- coding: utf-8 -*-
"""
Phase 57: Program Arithmetic (Opus Original)
Can we ADD/SUBTRACT program vectors?
What does MIN_vec + MAX_vec produce? Program algebra.

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
    for epoch in range(100):
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


def test_vec(model, tok, vec, prompts, layer, device):
    """Run vec on prompts, return list of predictions."""
    preds = []
    for prompt in prompts:
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append(pred)
    return preds


def main():
    print("[P57] Program Arithmetic")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile base programs
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")]

    print("  Compiling MIN and MAX...")
    min_vec = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42)
    max_vec = compile_prog(model, tok, max_data, target_layer, DEVICE, seed=99)

    test_prompts = ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) =", "9, 3) =",
                    "7, 2) =", "6, 3) =", "2, 9) ="]
    min_expected = ["3", "2", "1", "4", "3", "2", "3", "2"]
    max_expected = ["7", "5", "8", "6", "9", "7", "6", "9"]

    # Test base programs
    min_preds = test_vec(model, tok, min_vec, test_prompts, target_layer, DEVICE)
    max_preds = test_vec(model, tok, max_vec, test_prompts, target_layer, DEVICE)
    print(f"  MIN accuracy: {sum(p==e for p,e in zip(min_preds, min_expected))/len(min_expected):.0%}")
    print(f"  MAX accuracy: {sum(p==e for p,e in zip(max_preds, max_expected))/len(max_expected):.0%}")

    # Arithmetic operations
    print("\n  Testing program arithmetic...")
    arithmetic_results = {}

    operations = {
        'MIN + MAX': min_vec + max_vec,
        'MIN - MAX': min_vec - max_vec,
        'MAX - MIN': max_vec - min_vec,
        '0.5*MIN + 0.5*MAX': 0.5 * min_vec + 0.5 * max_vec,
        '2*MIN': 2.0 * min_vec,
        '2*MAX': 2.0 * max_vec,
        'MIN * -1 (negation)': -1.0 * min_vec,
    }

    for op_name, op_vec in operations.items():
        preds = test_vec(model, tok, op_vec, test_prompts, target_layer, DEVICE)
        min_match = sum(p == e for p, e in zip(preds, min_expected))
        max_match = sum(p == e for p, e in zip(preds, max_expected))

        arithmetic_results[op_name] = {
            'predictions': preds,
            'min_match': min_match,
            'max_match': max_match,
            'total': len(preds),
        }
        print(f"    {op_name}: preds={preds[:5]}... "
              f"(MIN match: {min_match}/{len(preds)}, MAX match: {max_match}/{len(preds)})")

    # Save
    output = {
        'phase': 57, 'name': 'program_arithmetic',
        'test_prompts': test_prompts,
        'min_expected': min_expected,
        'max_expected': max_expected,
        'arithmetic_results': arithmetic_results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase57_arithmetic.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    names = list(arithmetic_results.keys())
    min_matches = [arithmetic_results[n]['min_match'] / arithmetic_results[n]['total'] for n in names]
    max_matches = [arithmetic_results[n]['max_match'] / arithmetic_results[n]['total'] for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, min_matches, w, label='MIN match', color='tab:blue', edgecolor='black')
    ax.bar(x + w/2, max_matches, w, label='MAX match', color='tab:orange', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Match Rate')
    ax.set_title('Program Arithmetic: Vector Operations on Compiled Programs', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('Phase 57: Program Arithmetic\nWhat happens when you add/subtract program vectors?',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase57_arithmetic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
