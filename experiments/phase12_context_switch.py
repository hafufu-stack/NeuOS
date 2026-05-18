# -*- coding: utf-8 -*-
"""
Phase 12: Neural Context Switching
P8's multi-task failure was due to register collision.
Solution: save/restore hidden states between tasks, like a real OS.

Method:
  1. Run Program A (arithmetic) for 1 forward pass -> save all hidden states to RAM
  2. Run Program B (comparison) for 1 forward pass -> save all hidden states to RAM
  3. Restore Program A states and read the output
  4. Restore Program B states and read the output
  Both should produce correct results despite sharing the same "hardware"

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

# Program pairs: arithmetic + comparison
TASK_PAIRS = [
    ("def f(): return 3 + 4 =", " 7", "def f(): return max(2,8) =", " 8"),
    ("def f(): return 5 + 2 =", " 7", "def f(): return max(9,1) =", " 9"),
    ("def f(): return 8 + 1 =", " 9", "def f(): return max(3,7) =", " 7"),
    ("def f(): return 6 + 3 =", " 9", "def f(): return max(5,4) =", " 5"),
    ("def f(): return 2 + 7 =", " 9", "def f(): return max(6,2) =", " 6"),
    ("def f(): return 4 + 4 =", " 8", "def f(): return max(1,9) =", " 9"),
    ("def f(): return 1 + 6 =", " 7", "def f(): return max(4,8) =", " 8"),
    ("def f(): return 3 + 3 =", " 6", "def f(): return max(7,3) =", " 7"),
]


def main():
    print("[P12] Neural Context Switching")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # === Baseline: run each task independently ===
    print("  Baseline: independent execution...")
    baseline_a = 0
    baseline_b = 0
    for prompt_a, ans_a, prompt_b, ans_b in TASK_PAIRS:
        # Task A
        inp = tok(prompt_a, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == ans_a.strip():
            baseline_a += 1

        # Task B
        inp = tok(prompt_b, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == ans_b.strip():
            baseline_b += 1

    ba = baseline_a / len(TASK_PAIRS)
    bb = baseline_b / len(TASK_PAIRS)
    print(f"    Task A baseline: {ba:.1%}")
    print(f"    Task B baseline: {bb:.1%}")

    # === Context Switching: save/restore full hidden states ===
    print("  Context Switching: save/restore hidden states...")
    cs_a = 0
    cs_b = 0

    for prompt_a, ans_a, prompt_b, ans_b in TASK_PAIRS:
        # Run Task A, save full state
        inp_a = tok(prompt_a, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out_a = model(**inp_a, output_hidden_states=True)
        # Save: all hidden states (context)
        context_a = {}
        for l in range(n_layers):
            context_a[l] = get_last_token(out_a.hidden_states[l+1]).cpu()
        logits_a = out_a.logits[0, -1, :].clone()

        # Run Task B, save full state
        inp_b = tok(prompt_b, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out_b = model(**inp_b, output_hidden_states=True)
        context_b = {}
        for l in range(n_layers):
            context_b[l] = get_last_token(out_b.hidden_states[l+1]).cpu()
        logits_b = out_b.logits[0, -1, :].clone()

        # Restore Task A: inject saved states at ALL layers
        def make_restore_hook(layer_idx, context):
            vec = context[layer_idx].to(DEVICE)
            def hook_fn(module, input, output):
                return replace_last_token(output, vec)
            return hook_fn

        handles = []
        for l in range(n_layers):
            h = model.model.layers[l].register_forward_hook(
                make_restore_hook(l, context_a))
            handles.append(h)

        # Run dummy forward pass - the hooks will override with saved context
        dummy_inp = tok(prompt_b, return_tensors='pt').to(DEVICE)  # wrong prompt!
        with torch.no_grad():
            restored_a = model(**dummy_inp)
        for h in handles:
            h.remove()

        pred_a = tok.decode(restored_a.logits[0, -1, :].argmax().item()).strip()
        if pred_a == ans_a.strip():
            cs_a += 1

        # Restore Task B
        handles = []
        for l in range(n_layers):
            h = model.model.layers[l].register_forward_hook(
                make_restore_hook(l, context_b))
            handles.append(h)

        dummy_inp = tok(prompt_a, return_tensors='pt').to(DEVICE)  # wrong prompt!
        with torch.no_grad():
            restored_b = model(**dummy_inp)
        for h in handles:
            h.remove()

        pred_b = tok.decode(restored_b.logits[0, -1, :].argmax().item()).strip()
        if pred_b == ans_b.strip():
            cs_b += 1

    csa = cs_a / len(TASK_PAIRS)
    csb = cs_b / len(TASK_PAIRS)
    print(f"    Context-restored A: {csa:.1%}")
    print(f"    Context-restored B: {csb:.1%}")

    # === Time-sliced execution: interleave layers ===
    print("  Time-sliced: alternate layers between tasks...")
    ts_a = 0
    ts_b = 0

    for prompt_a, ans_a, prompt_b, ans_b in TASK_PAIRS:
        # Get both contexts
        inp_a = tok(prompt_a, return_tensors='pt').to(DEVICE)
        inp_b = tok(prompt_b, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out_a = model(**inp_a, output_hidden_states=True)
            out_b = model(**inp_b, output_hidden_states=True)

        ctx_a = {l: get_last_token(out_a.hidden_states[l+1]).cpu() for l in range(n_layers)}
        ctx_b = {l: get_last_token(out_b.hidden_states[l+1]).cpu() for l in range(n_layers)}

        # Run with Task A at even layers, Task B at odd layers
        # Then read output - which task "wins"?
        handles = []
        for l in range(n_layers):
            ctx = ctx_a if l % 2 == 0 else ctx_b
            h = model.model.layers[l].register_forward_hook(make_restore_hook(l, ctx))
            handles.append(h)

        dummy = tok("x", return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out_interleave = model(**dummy)
        for h in handles:
            h.remove()

        pred = tok.decode(out_interleave.logits[0, -1, :].argmax().item()).strip()
        if pred == ans_a.strip():
            ts_a += 1
        if pred == ans_b.strip():
            ts_b += 1

    tsa = ts_a / len(TASK_PAIRS)
    tsb = ts_b / len(TASK_PAIRS)
    print(f"    Time-sliced output matches A: {tsa:.1%}")
    print(f"    Time-sliced output matches B: {tsb:.1%}")

    # Save
    output = {
        'phase': 12, 'name': 'context_switching',
        'n_pairs': len(TASK_PAIRS), 'n_layers': n_layers,
        'results': {
            'baseline_A': round(ba, 4), 'baseline_B': round(bb, 4),
            'restored_A': round(csa, 4), 'restored_B': round(csb, 4),
            'timeslice_A': round(tsa, 4), 'timeslice_B': round(tsb, 4),
        },
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase12_context_switch.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(2)
    w = 0.25
    axes[0].bar(x - w, [ba, bb], w, label='Baseline', color='tab:blue', edgecolor='black')
    axes[0].bar(x, [csa, csb], w, label='Context Restored', color='tab:green', edgecolor='black')
    axes[0].bar(x + w, [tsa, tsb], w, label='Time-sliced', color='tab:orange', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Task A\n(arithmetic)', 'Task B\n(max)'])
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Context Switch Performance', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=10)

    axes[1].axis('off')
    summary = (
        f"Neural Context Switching\n\n"
        f"Baseline A: {ba:.0%} / B: {bb:.0%}\n"
        f"Restored A: {csa:.0%} / B: {csb:.0%}\n"
        f"Timeslice A: {tsa:.0%} / B: {tsb:.0%}\n\n"
        f"{'Context switch WORKS!' if csa > 0.5 and csb > 0.5 else 'Partial success'}"
    )
    axes[1].text(0.5, 0.5, summary, transform=axes[1].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 12: Neural Context Switching\nCan we run multiple programs via state save/restore?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase12_context_switch.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
