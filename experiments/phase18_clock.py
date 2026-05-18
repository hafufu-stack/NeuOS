# -*- coding: utf-8 -*-
"""
Phase 18: The Cache Clock (Multi-cycle via KV Cache Write)
P11 failed because embedding layer overwrites injected vectors.
Solution: write intermediate results to KV cache (RAM), not registers.

Method for a+b+c:
  1. Run "def f(): return a + b =" -> get output token (intermediate)
  2. APPEND intermediate token to input: "def f(): return a + b = 7 + c ="
  3. The KV cache from step 1 is reused via past_key_values

This is autoregressive chain-of-thought, but controlled programmatically.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P18] The Cache Clock (Multi-cycle via KV Cache)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)

    # Test problems
    import random
    random.seed(42)
    problems_2step = []
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                if a + b < 10 and a + b + c < 10:
                    problems_2step.append((a, b, c, a + b, a + b + c))
    if len(problems_2step) > 30:
        problems_2step = random.sample(problems_2step, 30)

    problems_3step = []
    for a in range(1, 4):
        for b in range(1, 4):
            for c in range(1, 4):
                for d in range(1, 3):
                    total = a + b + c + d
                    if a+b < 10 and a+b+c < 10 and total < 10:
                        problems_3step.append((a, b, c, d, total))
    if len(problems_3step) > 20:
        problems_3step = random.sample(problems_3step, 20)

    # === Baseline: single pass ===
    print("  Baseline: single-pass a+b+c...")
    baseline_correct = 0
    for a, b, c, inter, final in problems_2step:
        prompt = f"def f(): return {a} + {b} + {c} ="
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(final):
            baseline_correct += 1
    baseline_acc = baseline_correct / len(problems_2step)
    print(f"    Baseline: {baseline_acc:.1%}")

    # === Method A: 2-cycle clock (a+b -> intermediate -> +c) ===
    print("  Method A: 2-cycle cache clock...")
    clock2_correct = 0
    for a, b, c, inter, final in problems_2step:
        # Cycle 1: compute a + b
        prompt1 = f"def f(): return {a} + {b} ="
        inp1 = tok(prompt1, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out1 = model(**inp1, use_cache=True)

        # Read intermediate result
        pred_inter_id = out1.logits[0, -1, :].argmax().item()
        pred_inter = tok.decode(pred_inter_id).strip()

        # Cycle 2: continue with " {inter} + c ="
        # Append to KV cache: the intermediate token + " + c ="
        continuation = f" {pred_inter} + {c} ="
        inp2 = tok(continuation, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out2 = model(input_ids=inp2.input_ids, past_key_values=out1.past_key_values)

        pred_final = tok.decode(out2.logits[0, -1, :].argmax().item()).strip()
        if pred_final == str(final):
            clock2_correct += 1

    clock2_acc = clock2_correct / len(problems_2step)
    print(f"    2-cycle clock: {clock2_acc:.1%}")

    # === Method B: 3-cycle clock (a+b+c+d) ===
    print("  Method B: 3-cycle cache clock...")
    clock3_correct = 0
    for a, b, c, d, total in problems_3step:
        # Cycle 1
        prompt = f"def f(): return {a} + {b} ="
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, use_cache=True)
        kv = out.past_key_values
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()

        # Cycle 2
        cont = f" {pred} + {c} ="
        inp2 = tok(cont, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out2 = model(input_ids=inp2.input_ids, past_key_values=kv, use_cache=True)
        kv = out2.past_key_values
        pred = tok.decode(out2.logits[0, -1, :].argmax().item()).strip()

        # Cycle 3
        cont = f" {pred} + {d} ="
        inp3 = tok(cont, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out3 = model(input_ids=inp3.input_ids, past_key_values=kv)
        pred_final = tok.decode(out3.logits[0, -1, :].argmax().item()).strip()

        if pred_final == str(total):
            clock3_correct += 1

    clock3_acc = clock3_correct / len(problems_3step) if problems_3step else 0
    print(f"    3-cycle clock: {clock3_acc:.1%}")

    # === Method C: N-cycle arbitrary loop ===
    print("  Method C: N-cycle loop (sum 1 to N)...")
    loop_results = {}
    for N in [3, 5, 7, 10]:
        expected = N * (N + 1) // 2
        if expected >= 10:
            # Multi-digit, skip for now
            loop_results[str(N)] = -1
            continue

        prompt = f"def f(): return 1 + 2 ="
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, use_cache=True)
        kv = out.past_key_values
        running = tok.decode(out.logits[0, -1, :].argmax().item()).strip()

        for i in range(3, N + 1):
            cont = f" {running} + {i} ="
            inp_c = tok(cont, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out_c = model(input_ids=inp_c.input_ids, past_key_values=kv, use_cache=True)
            kv = out_c.past_key_values
            running = tok.decode(out_c.logits[0, -1, :].argmax().item()).strip()

        correct = 1 if running == str(expected) else 0
        loop_results[str(N)] = correct
        print(f"    sum(1..{N}): expected={expected}, got={running}, "
              f"{'OK' if correct else 'MISS'}")

    # Save
    output = {
        'phase': 18, 'name': 'cache_clock',
        'baseline': round(baseline_acc, 4),
        'clock_2cycle': round(clock2_acc, 4),
        'clock_3cycle': round(clock3_acc, 4),
        'loop_results': loop_results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase18_clock.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    labels = ['1-pass\nbaseline', '2-cycle\nclock', '3-cycle\nclock']
    accs = [baseline_acc, clock2_acc, clock3_acc]
    colors = ['tab:gray', 'tab:blue', 'tab:red']
    axes[0].bar(labels, accs, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Cache Clock vs Single-Pass', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Loop results
    ns = [int(k) for k in loop_results.keys() if loop_results[k] >= 0]
    loop_accs = [loop_results[str(n)] for n in ns]
    axes[1].bar([str(n) for n in ns], loop_accs, color='tab:purple', edgecolor='black')
    axes[1].set_xlabel('N (sum 1..N)', fontsize=12)
    axes[1].set_ylabel('Correct', fontsize=12)
    axes[1].set_title('N-cycle Loop Test', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1.2)

    axes[2].axis('off')
    summary = (
        f"The Cache Clock\n\n"
        f"Baseline (1-pass): {baseline_acc:.0%}\n"
        f"2-cycle (a+b+c): {clock2_acc:.0%}\n"
        f"3-cycle (a+b+c+d): {clock3_acc:.0%}\n\n"
        f"P11 (register fwd): 0%\n"
        f"P18 (KV cache): {clock2_acc:.0%}\n\n"
        f"{'CLOCK WORKS!' if clock2_acc > baseline_acc else 'Investigating...'}"
    )
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 18: The Cache Clock\nMulti-cycle computation via KV cache write/read',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase18_clock.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
