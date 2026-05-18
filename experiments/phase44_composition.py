# -*- coding: utf-8 -*-
"""
Phase 44: Program Composition (Neural Pipeline)
Can we chain two operations: first MIN(a,b), then +1?
Load two programs at different pipeline stages.

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


def main():
    print("[P44] Program Composition (Neural Pipeline)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size

    for p in model.parameters():
        p.requires_grad = False

    # Goal: compose MIN and +1 into a single pipeline
    # Step 1: Compile individual program vectors
    print("  Step 1: Compiling individual programs...")

    # Compile MIN program via gradient descent (P35 method)
    programs = {}
    for prog_name, train, layer in [
        ('MIN', [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                 ("4, 6) =", "4"), ("9, 3) =", "3"), ("2, 5) =", "2")], 8),
        ('PLUS1', [("3) =", "4"), ("5) =", "6"), ("1) =", "2"),
                   ("7) =", "8"), ("4) =", "5"), ("2) =", "3")], 20),
    ]:
        vec = torch.randn(hidden_size, device=DEVICE) * 0.01
        vec.requires_grad_(True)
        opt = torch.optim.Adam([vec], lr=0.01)

        for epoch in range(100):
            for prompt, target_str in train:
                target_id = tok.encode(target_str)[-1]
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                def inject(module, input, output, v=vec):
                    return replace_last_token(output, v)
                h = model.model.layers[layer].register_forward_hook(inject)
                out = model(**inp)
                h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([target_id]).to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Test individual
        correct = 0
        for prompt, target_str in train:
            vec_eval = vec.detach()
            def inject_eval(module, input, output, v=vec_eval):
                return replace_last_token(output, v)
            h = model.model.layers[layer].register_forward_hook(inject_eval)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            if pred == target_str:
                correct += 1
        acc = correct / len(train)
        programs[prog_name] = {'vec': vec.detach(), 'layer': layer, 'acc': round(acc, 4)}
        print(f"    {prog_name} @ L{layer}: {acc:.0%}")

    # Step 2: Compose: inject BOTH programs simultaneously
    print("\n  Step 2: Composing MIN + PLUS1...")
    compose_tests = [
        ("3, 7) =", 3, 7, min(3,7)+1),   # min(3,7)+1 = 4
        ("5, 2) =", 5, 2, min(5,2)+1),   # min(5,2)+1 = 3
        ("8, 1) =", 8, 1, min(8,1)+1),   # min(8,1)+1 = 2
        ("4, 6) =", 4, 6, min(4,6)+1),   # min(4,6)+1 = 5
        ("9, 3) =", 9, 3, min(9,3)+1),   # min(9,3)+1 = 4
        ("7, 2) =", 7, 2, min(7,2)+1),   # min(7,2)+1 = 3
    ]

    results = {'min_only': [], 'plus1_only': [], 'composed': []}

    for data_str, a, b, expected_composed in compose_tests:
        expected_min = min(a, b)
        expected_plus1 = str(expected_composed)

        # MIN only
        def inject_min(module, input, output, v=programs['MIN']['vec']):
            return replace_last_token(output, v)
        h = model.model.layers[programs['MIN']['layer']].register_forward_hook(inject_min)
        inp = tok(data_str, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred_min = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        results['min_only'].append(pred_min == str(expected_min))

        # PLUS1 only
        def inject_p1(module, input, output, v=programs['PLUS1']['vec']):
            return replace_last_token(output, v)
        h = model.model.layers[programs['PLUS1']['layer']].register_forward_hook(inject_p1)
        inp = tok(data_str, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred_p1 = tok.decode(out.logits[0, -1, :].argmax().item()).strip()

        # COMPOSED: inject BOTH (MIN at L8 + PLUS1 at L20)
        def inject_min2(module, input, output, v=programs['MIN']['vec']):
            return replace_last_token(output, v)
        def inject_p1_2(module, input, output, v=programs['PLUS1']['vec']):
            return replace_last_token(output, v)
        h1 = model.model.layers[programs['MIN']['layer']].register_forward_hook(inject_min2)
        h2 = model.model.layers[programs['PLUS1']['layer']].register_forward_hook(inject_p1_2)
        inp = tok(data_str, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h1.remove(); h2.remove()
        pred_composed = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        results['composed'].append(pred_composed == expected_plus1)

        print(f"    ({a},{b}): MIN={pred_min}(exp {expected_min}), "
              f"COMPOSED={pred_composed}(exp {expected_plus1})")

    min_acc = sum(results['min_only']) / len(results['min_only'])
    comp_acc = sum(results['composed']) / len(results['composed'])

    # Save
    output = {
        'phase': 44, 'name': 'program_composition',
        'programs': {n: {'layer': p['layer'], 'acc': p['acc']} for n, p in programs.items()},
        'min_only_acc': round(min_acc, 4),
        'composed_acc': round(comp_acc, 4),
        'test_results': {
            'min_only': results['min_only'],
            'composed': results['composed'],
        },
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase44_composition.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(['MIN only\n(L8)', 'PLUS1 only\n(L20)', 'MIN+PLUS1\n(L8+L20)'],
                [min_acc, programs['PLUS1']['acc'], comp_acc],
                color=['tab:blue', 'tab:orange', 'tab:green'], edgecolor='black')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Program Composition', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([min_acc, programs['PLUS1']['acc'], comp_acc]):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=13)

    axes[1].axis('off')
    summary = ("Neural Pipeline Architecture\n"
               "="*40 + "\n\n"
               "Input: (a, b)\n"
               "  |  L8:  MIN program injected\n"
               "  |  L9-L19: propagation\n"
               "  |  L20: PLUS1 program injected\n"
               "  v\n"
               f"Output: min(a,b) + 1\n\n"
               f"Composition accuracy: {comp_acc:.0%}\n"
               f"Individual MIN: {min_acc:.0%}\n"
               f"Individual +1: {programs['PLUS1']['acc']:.0%}")
    axes[1].text(0.5, 0.5, summary, transform=axes[1].transAxes,
                fontsize=11, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    plt.suptitle('Phase 44: Program Composition\nChaining MIN + PLUS1 in the neural pipeline',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase44_composition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
