# -*- coding: utf-8 -*-
"""
Phase 11: The Neural Clock - Multi-cycle Execution via Register Forwarding
P10 showed a+b+c fails (2.5%) because the ALU is single-cycle.
Solution: manually forward the intermediate result back to the operand register.

Method:
  1. Run "a + b =" -> extract SUM register (L20) hidden state
  2. Inject that vector into Operand A register (L13) for a NEW forward pass
  3. Run "X + c =" where X is the forwarded intermediate

This implements a "system clock" for the Neural CPU.

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
    print("[P11] The Neural Clock - Multi-cycle Execution")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # Test problems: a + b + c where a+b < 10 and a+b+c < 10
    problems = []
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(1, 5):
                if a + b < 10 and a + b + c < 10:
                    problems.append((a, b, c, a + b, a + b + c))
    import random
    random.seed(42)
    if len(problems) > 30:
        problems = random.sample(problems, 30)

    # === Baseline: single-pass (expected ~2.5%) ===
    print("  Baseline: single-pass a+b+c...")
    baseline_correct = 0
    for a, b, c, inter, final in problems:
        prompt = f"def f(): return {a} + {b} + {c} ="
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(final):
            baseline_correct += 1
    baseline_acc = baseline_correct / len(problems)
    print(f"    Baseline: {baseline_acc:.1%}")

    # === Method A: Two-pass with Register Forwarding ===
    print("  Method A: Register Forwarding (SUM->Operand)...")
    # For each register pair (source_layer, target_layer), try forwarding
    forward_results = {}
    source_layers = [18, 19, 20, 21, 22]  # SUM / output region
    target_layers = [2, 12, 13, 14]        # Operand region

    best_acc = 0
    best_pair = None

    for src_l in source_layers:
        for tgt_l in target_layers:
            correct = 0
            for a, b, c, inter, final in problems:
                # Pass 1: compute a + b
                prompt1 = f"def f(): return {a} + {b} ="
                inp1 = tok(prompt1, return_tensors='pt').to(DEVICE)

                # Capture hidden state at source layer
                captured = [None]
                def cap_hook(module, input, output):
                    captured[0] = get_last_token(output)
                h1 = model.model.layers[src_l].register_forward_hook(cap_hook)
                with torch.no_grad():
                    model(**inp1)
                h1.remove()
                intermediate_vec = captured[0]

                # Pass 2: inject intermediate as operand, compute + c
                # Use a "dummy" prompt where the first operand will be overwritten
                prompt2 = f"def f(): return 0 + {c} ="
                inp2 = tok(prompt2, return_tensors='pt').to(DEVICE)

                def inject_hook(module, input, output):
                    return replace_last_token(output, intermediate_vec)

                h2 = model.model.layers[tgt_l].register_forward_hook(inject_hook)
                with torch.no_grad():
                    out2 = model(**inp2)
                h2.remove()

                pred = tok.decode(out2.logits[0, -1, :].argmax().item()).strip()
                if pred == str(final):
                    correct += 1

            acc = correct / len(problems)
            forward_results[f"L{src_l}->L{tgt_l}"] = round(acc, 4)
            if acc > best_acc:
                best_acc = acc
                best_pair = (src_l, tgt_l)

    if best_pair:
        print(f"    Best forwarding: L{best_pair[0]}->L{best_pair[1]} ({best_acc:.1%})")
    else:
        print(f"    No forwarding succeeded (all 0%)")

    # === Method B: Multi-cycle loop (3+4+2+1 = 10) ===
    print("  Method B: Multi-cycle loop (4 operands)...")
    loop_problems = []
    for a in range(1, 4):
        for b in range(1, 4):
            for c in range(1, 4):
                for d in range(1, 3):
                    total = a + b + c + d
                    if a+b < 10 and a+b+c < 10 and total < 10:
                        loop_problems.append((a, b, c, d, total))
    random.seed(42)
    if len(loop_problems) > 20:
        loop_problems = random.sample(loop_problems, 20)

    if best_pair:
        src_l, tgt_l = best_pair
        loop_correct = 0
        for a, b, c, d, total in loop_problems:
            operands = [a, b, c, d]
            # Cycle 1: a + b
            prompt = f"def f(): return {operands[0]} + {operands[1]} ="
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            captured = [None]
            def cap(module, input, output):
                captured[0] = get_last_token(output)
            h = model.model.layers[src_l].register_forward_hook(cap)
            with torch.no_grad():
                model(**inp)
            h.remove()
            current_vec = captured[0]

            # Cycles 2+: inject and add next operand
            for op in operands[2:]:
                prompt = f"def f(): return 0 + {op} ="
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                def inj(module, input, output):
                    return replace_last_token(output, current_vec)
                h = model.model.layers[tgt_l].register_forward_hook(inj)
                captured2 = [None]
                def cap2(module, input, output):
                    captured2[0] = get_last_token(output)
                h2 = model.model.layers[src_l].register_forward_hook(cap2)
                with torch.no_grad():
                    model(**inp)
                h.remove()
                h2.remove()
                current_vec = captured2[0]

            # Final readout
            prompt_final = f"def f(): return 0 + 0 ="
            inp_final = tok(prompt_final, return_tensors='pt').to(DEVICE)
            def final_inj(module, input, output):
                return replace_last_token(output, current_vec)
            h = model.model.layers[tgt_l].register_forward_hook(final_inj)
            with torch.no_grad():
                out = model(**inp_final)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            if pred == str(total):
                loop_correct += 1

        loop_acc = loop_correct / len(loop_problems) if loop_problems else 0
        print(f"    4-operand loop: {loop_acc:.1%}")
    else:
        loop_acc = 0

    # Save
    output = {
        'phase': 11, 'name': 'neural_clock',
        'n_problems': len(problems), 'n_loop': len(loop_problems),
        'baseline_acc': round(baseline_acc, 4),
        'best_forwarding': {
            'source': best_pair[0] if best_pair else None,
            'target': best_pair[1] if best_pair else None,
            'accuracy': round(best_acc, 4),
        },
        'all_forwarding': forward_results,
        'loop_4op_acc': round(loop_acc, 4),
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase11_clock.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy comparison
    labels = ['Single-pass\n(baseline)', '2-cycle\n(forwarding)', '4-cycle\n(loop)']
    accs = [baseline_acc, best_acc, loop_acc]
    colors = ['tab:gray', 'tab:blue', 'tab:red']
    axes[0].bar(labels, accs, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Neural Clock: Multi-Cycle vs Single-Pass', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Forwarding heatmap
    fwd_matrix = np.zeros((len(source_layers), len(target_layers)))
    for i, sl in enumerate(source_layers):
        for j, tl in enumerate(target_layers):
            key = f"L{sl}->L{tl}"
            fwd_matrix[i, j] = forward_results.get(key, 0)
    im = axes[1].imshow(fwd_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(target_layers)))
    axes[1].set_xticklabels([f'L{l}' for l in target_layers])
    axes[1].set_yticks(range(len(source_layers)))
    axes[1].set_yticklabels([f'L{l}' for l in source_layers])
    axes[1].set_xlabel('Target (Operand)', fontsize=12)
    axes[1].set_ylabel('Source (SUM)', fontsize=12)
    axes[1].set_title('Forwarding Success Rate', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1])

    # Summary
    axes[2].axis('off')
    summary = (
        f"The Neural Clock\n\n"
        f"Baseline (1 pass): {baseline_acc:.0%}\n"
        f"2-cycle forward: {best_acc:.0%}\n"
        f"  (L{best_pair[0] if best_pair else '?'} -> L{best_pair[1] if best_pair else '?'})\n"
        f"4-cycle loop: {loop_acc:.0%}\n\n"
        f"{'Clock WORKS!' if best_acc > baseline_acc else 'Clock needs tuning'}"
    )
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 11: The Neural Clock\nCan we chain multi-cycle computation via register forwarding?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase11_clock.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
