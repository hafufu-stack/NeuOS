# -*- coding: utf-8 -*-
"""
Phase 17: KV-Cache Paging (True Multitasking OS)
P12 failed because hidden state restoration doesn't work.
Solution: swap entire KV caches (the TRUE RAM of transformers).

FIXED: Use input_ids only (no attention_mask) with past_key_values
to let the model auto-infer position_ids from cache length.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model
from kv_utils import swap_out, swap_in, run_prefix, continue_from_saved

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P17] KV-Cache Paging (True Multitasking) [FIXED]")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)

    tasks = [
        ("def f(): return 3 + 4", " =", "7",
         "def f(): return max(2, 8)", " =", "8"),
        ("def f(): return 5 + 2", " =", "7",
         "def f(): return max(9, 1)", " =", "9"),
        ("def f(): return 8 + 1", " =", "9",
         "def f(): return max(3, 7)", " =", "7"),
        ("def f(): return 6 + 3", " =", "9",
         "def f(): return max(5, 4)", " =", "5"),
        ("def f(): return 2 + 7", " =", "9",
         "def f(): return max(6, 2)", " =", "6"),
        ("def f(): return 4 + 4", " =", "8",
         "def f(): return max(1, 9)", " =", "9"),
        ("def f(): return 1 + 6", " =", "7",
         "def f(): return max(4, 8)", " =", "8"),
        ("def f(): return 3 + 3", " =", "6",
         "def f(): return max(7, 3)", " =", "7"),
    ]

    # === Baseline ===
    print("  Baseline: independent execution...")
    baseline_a = 0
    baseline_b = 0
    for pa_pre, pa_cont, a_ans, pb_pre, pb_cont, b_ans in tasks:
        full_a = pa_pre + pa_cont
        inp = tok(full_a, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == a_ans.strip():
            baseline_a += 1

        full_b = pb_pre + pb_cont
        inp = tok(full_b, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == b_ans.strip():
            baseline_b += 1

    ba = baseline_a / len(tasks)
    bb = baseline_b / len(tasks)
    print(f"    Baseline A: {ba:.1%}")
    print(f"    Baseline B: {bb:.1%}")

    # === KV-Cache Paging ===
    print("  KV-Cache Paging: swap out/in...")
    paging_a = 0
    paging_b = 0

    for pa_pre, pa_cont, a_ans, pb_pre, pb_cont, b_ans in tasks:
        # Step 1: Run A prefix -> swap out
        _, kv_a = run_prefix(model, tok, pa_pre, DEVICE)
        kv_a_cpu = swap_out(kv_a)

        # Step 2: Run B prefix -> swap out
        _, kv_b = run_prefix(model, tok, pb_pre, DEVICE)
        kv_b_cpu = swap_out(kv_b)

        # Step 3: Swap in A, continue
        logits_a, _ = continue_from_saved(model, tok, pa_cont, kv_a_cpu, DEVICE)
        pred_a = tok.decode(logits_a[0, -1, :].argmax().item()).strip()
        if pred_a == a_ans.strip():
            paging_a += 1

        # Step 4: Swap in B, continue
        logits_b, _ = continue_from_saved(model, tok, pb_cont, kv_b_cpu, DEVICE)
        pred_b = tok.decode(logits_b[0, -1, :].argmax().item()).strip()
        if pred_b == b_ans.strip():
            paging_b += 1

    pa = paging_a / len(tasks)
    pb = paging_b / len(tasks)
    print(f"    Paged A: {pa:.1%}")
    print(f"    Paged B: {pb:.1%}")

    # === Stress test: 4 programs ===
    print("  Stress test: 4-way interleave...")
    stress_programs = [
        ("def f(): return 3 + 4", " =", "7"),
        ("def f(): return max(2, 8)", " =", "8"),
        ("def f(): return 9 - 3", " =", "6"),
        ("def f(): return min(5, 1)", " =", "1"),
    ]

    saved_kvs = []
    for prefix, cont, ans in stress_programs:
        _, kv = run_prefix(model, tok, prefix, DEVICE)
        saved_kvs.append(swap_out(kv))

    stress_correct = 0
    for i in reversed(range(len(stress_programs))):
        prefix, cont, ans = stress_programs[i]
        logits, _ = continue_from_saved(model, tok, cont, saved_kvs[i], DEVICE)
        pred = tok.decode(logits[0, -1, :].argmax().item()).strip()
        if pred == ans.strip():
            stress_correct += 1

    stress_acc = stress_correct / len(stress_programs)
    print(f"    4-way stress: {stress_acc:.1%}")

    # Save
    output = {
        'phase': 17, 'name': 'kv_cache_paging',
        'n_pairs': len(tasks),
        'results': {
            'baseline_A': round(ba, 4), 'baseline_B': round(bb, 4),
            'paged_A': round(pa, 4), 'paged_B': round(pb, 4),
            'stress_4way': round(stress_acc, 4),
        },
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase17_paging.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(2)
    w = 0.3
    axes[0].bar(x - w/2, [ba, bb], w, label='Baseline', color='tab:blue', edgecolor='black')
    axes[0].bar(x + w/2, [pa, pb], w, label='KV Paged', color='tab:green', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Task A\n(arithmetic)', 'Task B\n(max)'])
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('KV-Cache Paging vs Baseline', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=11)

    axes[1].axis('off')
    verdict = "TRUE MULTITASKING!" if pa > 0.5 and pb > 0.5 else "Partial"
    summary = (
        f"KV-Cache Paging\n\n"
        f"Baseline A: {ba:.0%} / B: {bb:.0%}\n"
        f"Paged A: {pa:.0%} / B: {pb:.0%}\n"
        f"4-way stress: {stress_acc:.0%}\n\n"
        f"P12 (hidden state): 0%\n"
        f"P17 (KV cache): {(pa+pb)/2:.0%}\n\n"
        f"{verdict}"
    )
    axes[1].text(0.5, 0.5, summary, transform=axes[1].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 17: KV-Cache Paging\nTrue multitasking via RAM (KV cache) swapping',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase17_paging.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
