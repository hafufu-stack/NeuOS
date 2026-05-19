# -*- coding: utf-8 -*-
"""
Phase 73: Neural Symbiosis
MIN and MAX form a symbiotic pair that cooperatively solves
problems neither can solve alone. Tests mutualism vs parasitism.

Combines P66 (immune) + P61 (ecosystem).

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


def compile_prog(model, tok, train, layer, device, seed=42, epochs=80):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(epochs):
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


def main():
    print("[P73] Neural Symbiosis")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")]

    min_vec = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42)
    max_vec = compile_prog(model, tok, max_data, target_layer, DEVICE, seed=99)

    test_p = ["3, 7) =", "5, 2) =", "8, 1) =", "7, 4) ="]
    min_exp = ["3", "2", "1", "4"]
    max_exp = ["7", "5", "8", "7"]

    min_solo = eval_prog(model, tok, min_vec, test_p, min_exp, target_layer, DEVICE)
    max_solo = eval_prog(model, tok, max_vec, test_p, max_exp, target_layer, DEVICE)
    print(f"  Solo: MIN={min_solo:.0%}, MAX={max_solo:.0%}")

    # Step 1: Mutualism - co-optimize MIN+MAX jointly
    print("\n  Step 1: Mutualistic co-optimization...")
    # Train a 'symbiotic adapter' that helps both
    sym_vec = torch.randn(model.config.hidden_size, device=DEVICE) * 0.01
    sym_vec.requires_grad_(True)
    opt = torch.optim.Adam([sym_vec], lr=0.005)

    for epoch in range(60):
        # Alternate: help MIN then help MAX
        for prompt, target_str in min_data[:3]:
            combined = min_vec + sym_vec * 0.3
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject(module, input, output, v=combined):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

        for prompt, target_str in max_data[:3]:
            combined = max_vec + sym_vec * 0.3
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject(module, input, output, v=combined):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

    sym_vec = sym_vec.detach()

    # Test symbiotic benefit
    min_sym = eval_prog(model, tok, min_vec + sym_vec * 0.3, test_p, min_exp, target_layer, DEVICE)
    max_sym = eval_prog(model, tok, max_vec + sym_vec * 0.3, test_p, max_exp, target_layer, DEVICE)
    print(f"    MIN+symbiont: {min_sym:.0%} (solo: {min_solo:.0%})")
    print(f"    MAX+symbiont: {max_sym:.0%} (solo: {max_solo:.0%})")
    mutualism = (min_sym >= min_solo) and (max_sym >= max_solo)
    print(f"    Mutualism: {'YES' if mutualism else 'NO'}")

    # Step 2: Parasitism test - does symbiont help one at other's expense?
    print("\n  Step 2: Parasitism test...")
    # Train parasitic adapter that ONLY helps MIN at MAX's expense
    para_vec = torch.randn(model.config.hidden_size, device=DEVICE) * 0.01
    para_vec.requires_grad_(True)
    opt = torch.optim.Adam([para_vec], lr=0.01)
    for epoch in range(60):
        for prompt, target_str in min_data:
            combined = max_vec + para_vec * 0.5  # Inject into MAX host
            target_id = tok.encode(target_str)[-1]  # But train for MIN output
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject(module, input, output, v=combined):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    para_vec = para_vec.detach()

    max_parasitized = eval_prog(model, tok, max_vec + para_vec * 0.5,
                                test_p, max_exp, target_layer, DEVICE)
    min_via_parasite = eval_prog(model, tok, max_vec + para_vec * 0.5,
                                test_p, min_exp, target_layer, DEVICE)
    print(f"    MAX (parasitized): {max_parasitized:.0%} (healthy: {max_solo:.0%})")
    print(f"    MIN output from parasitized MAX: {min_via_parasite:.0%}")
    parasitism = (max_parasitized < max_solo)
    print(f"    Parasitism detected: {'YES' if parasitism else 'NO'}")

    # Step 3: Commensalism - one benefits, other unaffected
    print("\n  Step 3: Relationship classification...")
    relationships = []
    for scale in [0.1, 0.3, 0.5, 1.0]:
        m = eval_prog(model, tok, min_vec + sym_vec * scale, test_p, min_exp, target_layer, DEVICE)
        x = eval_prog(model, tok, max_vec + sym_vec * scale, test_p, max_exp, target_layer, DEVICE)
        relationships.append({
            'scale': scale,
            'min_benefit': round(m - min_solo, 4),
            'max_benefit': round(x - max_solo, 4),
        })
        print(f"    scale={scale}: MIN delta={m-min_solo:+.0%}, MAX delta={x-max_solo:+.0%}")

    # Save
    output = {
        'phase': 73, 'name': 'neural_symbiosis',
        'min_solo': round(min_solo, 4), 'max_solo': round(max_solo, 4),
        'min_symbiotic': round(min_sym, 4), 'max_symbiotic': round(max_sym, 4),
        'mutualism': mutualism,
        'max_parasitized': round(max_parasitized, 4),
        'parasitism': parasitism,
        'relationships': relationships,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase73_symbiosis.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_pos = np.arange(2)
    axes[0].bar(x_pos - 0.15, [min_solo, max_solo], 0.3, label='Solo',
                color='lightblue', edgecolor='black')
    axes[0].bar(x_pos + 0.15, [min_sym, max_sym], 0.3, label='Symbiotic',
                color='tab:purple', edgecolor='black')
    axes[0].set_xticks(x_pos); axes[0].set_xticklabels(['MIN', 'MAX'])
    axes[0].set_title('Mutualism Test', fontweight='bold')
    axes[0].legend(); axes[0].set_ylim(0, 1.1)

    axes[1].bar(['MAX\nhealthy', 'MAX\nparasitized', 'MIN via\nparasite'],
                [max_solo, max_parasitized, min_via_parasite],
                color=['tab:red', 'tab:orange', 'tab:blue'], edgecolor='black')
    axes[1].set_title('Parasitism Test', fontweight='bold')
    axes[1].set_ylim(0, 1.1)

    scales = [r['scale'] for r in relationships]
    axes[2].plot(scales, [r['min_benefit'] for r in relationships], 'b-o',
                label='MIN benefit', linewidth=2)
    axes[2].plot(scales, [r['max_benefit'] for r in relationships], 'r-o',
                label='MAX benefit', linewidth=2)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Symbiont Scale')
    axes[2].set_ylabel('Performance Delta')
    axes[2].set_title('Dose-Response', fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 73: Neural Symbiosis\nMutualism, parasitism, and commensalism',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase73_symbiosis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
