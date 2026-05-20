# -*- coding: utf-8 -*-
"""
Phase 109: Dynamic Division of Labor
Two souls on a mixed MIN+MAX task. Does specialization emerge spontaneously?
No one tells them who does what -- they must self-organize.

"Order from chaos: the invisible hand of gradient descent."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_vec(model, tok, vec, data, layer, device):
    c = 0
    for p, e in data:
        def inj(m,i,o,v=vec): return replace_last_token(o,v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
    return c / len(data)

def main():
    print("[P109] Dynamic Division of Labor")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size
    for p in model.parameters(): p.requires_grad = False

    # Mixed task: min() and max() prompts interleaved
    min_data = [("min(3,7)=","3"),("min(5,2)=","2"),("min(8,1)=","1"),
                ("min(4,6)=","4"),("min(9,3)=","3")]
    max_data = [("max(3,7)=","7"),("max(5,2)=","5"),("max(8,1)=","8"),
                ("max(4,6)=","6"),("max(9,3)=","9")]
    mixed = min_data + max_data

    min_test = [("min(7,2)=","2"),("min(6,3)=","3"),("min(2,9)=","2")]
    max_test = [("max(7,2)=","7"),("max(6,3)=","6"),("max(2,9)=","9")]
    mixed_test = min_test + max_test

    # Step 1: Single soul baseline on mixed task
    print("  Step 1: Single soul on mixed task...")
    torch.manual_seed(42)
    solo = torch.randn(hs, device=DEVICE)*0.01; solo.requires_grad_(True)
    opt = torch.optim.Adam([solo], lr=0.01)
    for ep in range(150):
        for p, t in mixed:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(DEVICE)
            def inj(m,i,o,v=solo): return replace_last_token(o,v)
            h = model.model.layers[8].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    solo_final = solo.detach()
    solo_min = evaluate_vec(model, tok, solo_final, min_data+min_test, 8, DEVICE)
    solo_max = evaluate_vec(model, tok, solo_final, max_data+max_test, 8, DEVICE)
    solo_mixed = evaluate_vec(model, tok, solo_final, mixed+mixed_test, 8, DEVICE)
    print(f"    Solo on MIN: {solo_min:.0%}, MAX: {solo_max:.0%}, "
          f"mixed: {solo_mixed:.0%}")

    # Step 2: Two souls jointly trained (L4 + L16)
    print("\n  Step 2: Two souls jointly on mixed task...")
    torch.manual_seed(42)
    soul_A = torch.randn(hs, device=DEVICE)*0.01; soul_A.requires_grad_(True)
    soul_B = torch.randn(hs, device=DEVICE)*0.01; soul_B.requires_grad_(True)
    opt2 = torch.optim.Adam([soul_A, soul_B], lr=0.01)

    for ep in range(150):
        for p, t in mixed:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(DEVICE)
            def injA(m,i,o,v=soul_A): return replace_last_token(o,v)
            def injB(m,i,o,v=soul_B): return replace_last_token(o,v)
            hA = model.model.layers[4].register_forward_hook(injA)
            hB = model.model.layers[16].register_forward_hook(injB)
            out = model(**inp); hA.remove(); hB.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt2.zero_grad(); loss.backward(); opt2.step()

    sA = soul_A.detach(); sB = soul_B.detach()

    # Step 3: Ablation study - which soul does what?
    print("\n  Step 3: Ablation (testing each soul alone)...")
    # Both together
    def eval_dual(model, tok, vA, vB, data, device):
        c = 0
        for p, e in data:
            def injA(m,i,o,v=vA): return replace_last_token(o,v)
            def injB(m,i,o,v=vB): return replace_last_token(o,v)
            hA = model.model.layers[4].register_forward_hook(injA)
            hB = model.model.layers[16].register_forward_hook(injB)
            inp = tok(p, return_tensors='pt').to(device)
            with torch.no_grad(): out = model(**inp)
            hA.remove(); hB.remove()
            if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
        return c / len(data)

    both_min = eval_dual(model, tok, sA, sB, min_data+min_test, DEVICE)
    both_max = eval_dual(model, tok, sA, sB, max_data+max_test, DEVICE)
    both_mixed = eval_dual(model, tok, sA, sB, mixed+mixed_test, DEVICE)

    # Soul A alone (L4)
    a_min = evaluate_vec(model, tok, sA, min_data+min_test, 4, DEVICE)
    a_max = evaluate_vec(model, tok, sA, max_data+max_test, 4, DEVICE)
    # Soul B alone (L16)
    b_min = evaluate_vec(model, tok, sB, min_data+min_test, 16, DEVICE)
    b_max = evaluate_vec(model, tok, sB, max_data+max_test, 16, DEVICE)

    print(f"    Both: MIN={both_min:.0%}, MAX={both_max:.0%}, mixed={both_mixed:.0%}")
    print(f"    Soul A alone (L4): MIN={a_min:.0%}, MAX={a_max:.0%}")
    print(f"    Soul B alone (L16): MIN={b_min:.0%}, MAX={b_max:.0%}")

    # Specialization index: how much does each soul prefer one task?
    spec_A = abs(a_min - a_max)
    spec_B = abs(b_min - b_max)
    print(f"    Specialization index: A={spec_A:.2f}, B={spec_B:.2f}")

    output = {
        'phase': 109, 'name': 'division_of_labor',
        'solo': {'min': round(float(solo_min), 4),
                 'max': round(float(solo_max), 4),
                 'mixed': round(float(solo_mixed), 4)},
        'dual': {'min': round(float(both_min), 4),
                 'max': round(float(both_max), 4),
                 'mixed': round(float(both_mixed), 4)},
        'soul_A_alone': {'min': round(float(a_min), 4),
                         'max': round(float(a_max), 4)},
        'soul_B_alone': {'min': round(float(b_min), 4),
                         'max': round(float(b_max), 4)},
        'specialization': {'A': round(float(spec_A), 4),
                          'B': round(float(spec_B), 4)},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase109_division.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Comparison: solo vs dual
    labels = ['MIN', 'MAX', 'Mixed']
    solo_vals = [solo_min, solo_max, solo_mixed]
    dual_vals = [both_min, both_max, both_mixed]
    x = np.arange(3); w = 0.35
    axes[0].bar(x-w/2, solo_vals, w, label='Single Soul', color='tab:gray',
                edgecolor='black')
    axes[0].bar(x+w/2, dual_vals, w, label='Dual Souls', color='tab:purple',
                edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Accuracy'); axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Solo vs Dual System', fontweight='bold')
    axes[0].legend()

    # Ablation: specialization
    tasks = ['MIN', 'MAX']
    a_vals = [a_min, a_max]
    b_vals = [b_min, b_max]
    x = np.arange(2); w = 0.35
    axes[1].bar(x-w/2, a_vals, w, label='Soul A (L4)', color='tab:blue',
                edgecolor='black')
    axes[1].bar(x+w/2, b_vals, w, label='Soul B (L16)', color='tab:red',
                edgecolor='black')
    axes[1].set_xticks(x); axes[1].set_xticklabels(tasks)
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.2)
    axes[1].set_title(f'Specialization\n(A={spec_A:.2f}, B={spec_B:.2f})',
                      fontweight='bold')
    axes[1].legend()
    for i, v in enumerate(a_vals):
        axes[1].text(i-w/2, v+0.03, f'{v:.0%}', ha='center', fontsize=9)
    for i, v in enumerate(b_vals):
        axes[1].text(i+w/2, v+0.03, f'{v:.0%}', ha='center', fontsize=9)

    # Summary
    all_labels = ['Solo\nMixed', 'Dual\nMixed', 'A alone\nMIN', 'A alone\nMAX',
                  'B alone\nMIN', 'B alone\nMAX']
    all_vals = [solo_mixed, both_mixed, a_min, a_max, b_min, b_max]
    all_colors = ['gray', 'purple', 'blue', 'blue', 'red', 'red']
    axes[2].bar(all_labels, all_vals, color=[f'tab:{c}' for c in all_colors],
                edgecolor='black', alpha=0.8)
    axes[2].set_ylabel('Accuracy'); axes[2].set_ylim(0, 1.2)
    axes[2].set_title('Full Ablation Summary', fontweight='bold')
    for i, v in enumerate(all_vals):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold',
                    fontsize=8)

    plt.suptitle('Phase 109: Dynamic Division of Labor\n'
                 '"Order from chaos: the invisible hand of gradient descent"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase109_division.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
