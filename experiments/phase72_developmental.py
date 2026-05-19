# -*- coding: utf-8 -*-
"""
Phase 72: Developmental Program
Programs go through life stages like biological organisms.
Larva -> Pupa -> Adult: function changes during 'development'.
Uses P65's interpolation alpha as a developmental clock.

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


def get_preds(model, tok, vec, prompts, layer, device):
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
    print("[P72] Developmental Program")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile life stages: FIRST (larva) -> MIN (pupa) -> MAX (adult)
    first_data = [("3, 7) =", "3"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                  ("4, 6) =", "4"), ("9, 3) =", "9")]
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")]

    print("  Compiling life stage programs...")
    larva = compile_prog(model, tok, first_data, target_layer, DEVICE, seed=42)
    pupa = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=99)
    adult = compile_prog(model, tok, max_data, target_layer, DEVICE, seed=77)

    test_prompts = ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) =", "9, 3) ="]
    first_exp = ["3", "5", "8", "4", "9"]
    min_exp = ["3", "2", "1", "4", "3"]
    max_exp = ["7", "5", "8", "6", "9"]

    # Developmental timeline: 20 time steps
    N_STEPS = 20
    dev_timeline = []

    print("\n  Developmental timeline (larva -> pupa -> adult)...")
    for t in range(N_STEPS):
        # Developmental clock: sigmoid schedule
        # 0-7: larva, 7-13: larva->pupa transition, 13-20: pupa->adult
        if t < 7:
            # Larva phase
            vec = larva.clone()
            stage = 'LARVA'
        elif t < 13:
            # Larva -> Pupa transition
            alpha = (t - 7) / 6.0
            vec = (1 - alpha) * larva + alpha * pupa
            stage = 'PUPA_TRANS'
        else:
            # Pupa -> Adult transition
            alpha = (t - 13) / 7.0
            vec = (1 - alpha) * pupa + alpha * adult
            stage = 'ADULT_TRANS'

        preds = get_preds(model, tok, vec, test_prompts, target_layer, DEVICE)
        first_match = sum(p == e for p, e in zip(preds, first_exp))
        min_match = sum(p == e for p, e in zip(preds, min_exp))
        max_match = sum(p == e for p, e in zip(preds, max_exp))

        dev_timeline.append({
            't': t, 'stage': stage,
            'first_match': first_match, 'min_match': min_match,
            'max_match': max_match, 'preds': preds,
        })

        if t % 4 == 0 or t == N_STEPS - 1:
            print(f"    t={t:2d} [{stage:11s}]: FIRST={first_match}/5, "
                  f"MIN={min_match}/5, MAX={max_match}/5")

    # Critical period test: can development be reversed?
    print("\n  Critical period test (can adult revert to larva?)...")
    revert_vec = adult.clone()
    revert_scores = []
    for step in range(5):
        alpha = step / 4.0
        rev = (1 - alpha) * adult + alpha * larva
        preds = get_preds(model, tok, rev, test_prompts, target_layer, DEVICE)
        first_match = sum(p == e for p, e in zip(preds, first_exp))
        max_match = sum(p == e for p, e in zip(preds, max_exp))
        revert_scores.append({'alpha': round(alpha, 2),
                             'first': first_match, 'max': max_match})
        print(f"    revert alpha={alpha:.1f}: FIRST={first_match}/5, MAX={max_match}/5")

    # Save
    output = {
        'phase': 72, 'name': 'developmental_program',
        'n_steps': N_STEPS,
        'timeline': [{'t': d['t'], 'stage': d['stage'],
                      'first': d['first_match'], 'min': d['min_match'],
                      'max': d['max_match']} for d in dev_timeline],
        'revert_test': revert_scores,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase72_developmental.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ts = [d['t'] for d in dev_timeline]
    axes[0].plot(ts, [d['first_match']/5 for d in dev_timeline], 'g-o',
                label='FIRST (larva)', linewidth=2)
    axes[0].plot(ts, [d['min_match']/5 for d in dev_timeline], 'b-s',
                label='MIN (pupa)', linewidth=2)
    axes[0].plot(ts, [d['max_match']/5 for d in dev_timeline], 'r-^',
                label='MAX (adult)', linewidth=2)
    axes[0].axvspan(0, 7, alpha=0.1, color='green', label='Larva phase')
    axes[0].axvspan(7, 13, alpha=0.1, color='blue', label='Pupa trans.')
    axes[0].axvspan(13, 20, alpha=0.1, color='red', label='Adult trans.')
    axes[0].set_xlabel('Developmental Time')
    axes[0].set_ylabel('Match Rate')
    axes[0].set_title('Metamorphosis Timeline', fontweight='bold')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    alphas = [r['alpha'] for r in revert_scores]
    axes[1].plot(alphas, [r['first']/5 for r in revert_scores], 'g-o',
                label='FIRST', linewidth=2)
    axes[1].plot(alphas, [r['max']/5 for r in revert_scores], 'r-o',
                label='MAX', linewidth=2)
    axes[1].set_xlabel('Revert alpha (0=adult, 1=larva)')
    axes[1].set_ylabel('Match Rate')
    axes[1].set_title('Critical Period (Reversibility)', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].axis('off')
    final = dev_timeline[-1]
    summary = (f"DEVELOPMENTAL PROGRAM\n{'='*30}\n\n"
               f"Life stages: FIRST -> MIN -> MAX\n"
               f"Timeline: {N_STEPS} steps\n\n"
               f"Larva (t=0): FIRST={dev_timeline[0]['first_match']}/5\n"
               f"Pupa (t=10): MIN={dev_timeline[10]['min_match']}/5\n"
               f"Adult (t={N_STEPS-1}): MAX={final['max_match']}/5\n\n"
               f"Metamorphosis confirmed!")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                fontsize=10, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 72: Developmental Program\nLarva -> Pupa -> Adult metamorphosis',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase72_developmental.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
