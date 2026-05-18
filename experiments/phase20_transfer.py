# -*- coding: utf-8 -*-
"""
Phase 20: Register Transfer Protocol (Opus Original)
Can we transfer a register vector learned in one task
to zero-shot enable computation in a different context?

If the OPCODE register at L0 truly encodes "what to compute"
independent of format, then extracting it from task A and
injecting it into task B should enable B to compute A's function.

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
    print("[P20] Register Transfer Protocol (Opus Original)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # === Experiment A: OPCODE Transfer ===
    # Extract OPCODE from "addition" context, inject into "subtraction" prompt
    print("  Exp A: Transfer OPCODE from addition to subtraction context...")

    # Source: addition OPCODE (L0)
    add_prompts = [f"def f(): return {a} + {b} =" for a in range(1, 6) for b in range(1, 6) if a+b < 10]
    sub_prompts = [f"def f(): return {a} - {b} =" for a in range(3, 8) for b in range(1, 4) if a-b >= 0]

    # Capture OPCODE at L0 for addition
    add_opcodes = []
    for prompt in add_prompts[:10]:
        captured = [None]
        def cap(module, input, output):
            captured[0] = get_last_token(output)
        h = model.model.layers[0].register_forward_hook(cap)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        add_opcodes.append(captured[0])

    # Average OPCODE vector
    avg_add_opcode = torch.stack(add_opcodes).mean(dim=0)

    # Baseline: subtraction accuracy
    sub_baseline = 0
    for prompt in sub_prompts[:10]:
        a, b = int(prompt.split('return ')[1].split(' - ')[0]), int(prompt.split(' - ')[1].split(' =')[0])
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(a - b):
            sub_baseline += 1
    sub_base_acc = sub_baseline / min(len(sub_prompts), 10)
    print(f"    Subtraction baseline: {sub_base_acc:.1%}")

    # Inject addition OPCODE into subtraction prompts
    transfer_add_correct = 0
    transfer_add_to_sum = 0
    for prompt in sub_prompts[:10]:
        a, b = int(prompt.split('return ')[1].split(' - ')[0]), int(prompt.split(' - ')[1].split(' =')[0])

        def inject(module, input, output):
            return replace_last_token(output, avg_add_opcode)

        h = model.model.layers[0].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()

        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(a - b):
            transfer_add_correct += 1  # still does subtraction
        if pred == str(a + b) and a + b < 10:
            transfer_add_to_sum += 1   # switched to addition!

    transfer_still_sub = transfer_add_correct / min(len(sub_prompts), 10)
    transfer_to_add = transfer_add_to_sum / min(len(sub_prompts), 10)
    print(f"    After injection: still subtraction={transfer_still_sub:.1%}, "
          f"switched to addition={transfer_to_add:.1%}")

    # === Experiment B: Sort register transfer ===
    print("\n  Exp B: Transfer MIN register from sort to arithmetic...")

    # Get MIN register (L16) from sorting context
    sort_prompts = [f"def f(): return min({a}, {b}) =" for a in range(1, 7) for b in range(1, 7) if a != b]
    min_vecs = []
    for prompt in sort_prompts[:10]:
        captured = [None]
        def cap(module, input, output):
            captured[0] = get_last_token(output)
        h = model.model.layers[16].register_forward_hook(cap)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        min_vecs.append(captured[0])
    avg_min_vec = torch.stack(min_vecs).mean(dim=0)

    # Inject MIN register into arithmetic context
    arith_prompts = [(f"def f(): return {a} + {b} =", a, b) for a in range(2, 7) for b in range(2, 7) if a+b < 10]
    min_inject_count = 0
    for prompt, a, b in arith_prompts[:10]:
        def inject_min(module, input, output):
            return replace_last_token(output, avg_min_vec)

        h = model.model.layers[16].register_forward_hook(inject_min)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()

        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        expected_min = str(min(a, b))
        expected_sum = str(a + b)
        if pred == expected_min:
            min_inject_count += 1

    min_takeover = min_inject_count / min(len(arith_prompts), 10)
    print(f"    MIN register takeover rate: {min_takeover:.1%}")

    # === Experiment C: Layer fingerprinting ===
    print("\n  Exp C: Register similarity across tasks...")
    from sklearn.metrics.pairwise import cosine_similarity

    task_vecs = {}
    task_prompts = {
        'add': "def f(): return 3 + 4 =",
        'sub': "def f(): return 7 - 2 =",
        'max': "def f(): return max(3, 7) =",
        'min': "def f(): return min(3, 7) =",
    }

    for task_name, prompt in task_prompts.items():
        task_vecs[task_name] = {}
        for layer in [0, 2, 13, 16, 20]:
            captured = [None]
            def cap(module, input, output):
                captured[0] = get_last_token(output)
            h = model.model.layers[layer].register_forward_hook(cap)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            task_vecs[task_name][layer] = captured[0].float().cpu().numpy().flatten()

    # Compute cosine similarity at L0 (OPCODE) between tasks
    tasks_list = list(task_vecs.keys())
    sim_matrix = {}
    for layer in [0, 2, 13, 16, 20]:
        mat = np.zeros((len(tasks_list), len(tasks_list)))
        for i, t1 in enumerate(tasks_list):
            for j, t2 in enumerate(tasks_list):
                v1 = task_vecs[t1][layer].reshape(1, -1)
                v2 = task_vecs[t2][layer].reshape(1, -1)
                mat[i, j] = cosine_similarity(v1, v2)[0, 0]
        sim_matrix[layer] = mat.tolist()
        if layer == 0:
            print(f"    L0 (OPCODE) similarity:")
            for i, t1 in enumerate(tasks_list):
                sims = [f"{mat[i,j]:.2f}" for j in range(len(tasks_list))]
                print(f"      {t1}: [{', '.join(sims)}]")

    # Save
    output = {
        'phase': 20, 'name': 'register_transfer',
        'opcode_transfer': {
            'sub_baseline': round(sub_base_acc, 4),
            'still_sub_after_inject': round(transfer_still_sub, 4),
            'switched_to_add': round(transfer_to_add, 4),
        },
        'min_takeover': round(min_takeover, 4),
        'similarity_L0': sim_matrix[0],
        'tasks': tasks_list,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase20_transfer.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # OPCODE transfer
    labels = ['Baseline\n(subtraction)', 'Still sub\n(after inject)', 'Switched\nto addition']
    vals = [sub_base_acc, transfer_still_sub, transfer_to_add]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    axes[0].bar(labels, vals, color=colors, edgecolor='black')
    axes[0].set_ylabel('Rate', fontsize=12)
    axes[0].set_title('OPCODE Transfer: Add->Sub', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Similarity heatmap at L0
    mat0 = np.array(sim_matrix[0])
    im = axes[1].imshow(mat0, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    axes[1].set_xticks(range(len(tasks_list)))
    axes[1].set_xticklabels(tasks_list, fontsize=10)
    axes[1].set_yticks(range(len(tasks_list)))
    axes[1].set_yticklabels(tasks_list, fontsize=10)
    axes[1].set_title('L0 (OPCODE) Cosine Similarity', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1])
    for i in range(len(tasks_list)):
        for j in range(len(tasks_list)):
            axes[1].text(j, i, f'{mat0[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, color='black')

    axes[2].axis('off')
    summary = (
        f"Register Transfer Protocol\n\n"
        f"OPCODE (L0) Transfer:\n"
        f"  Sub baseline: {sub_base_acc:.0%}\n"
        f"  Switched to add: {transfer_to_add:.0%}\n\n"
        f"MIN (L16) Takeover:\n"
        f"  Rate: {min_takeover:.0%}\n\n"
        f"{'Transfer works!' if transfer_to_add > 0.1 else 'Registers are task-bound'}"
    )
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 20: Register Transfer Protocol (Opus Original)\nCan register vectors be transplanted across tasks?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase20_transfer.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
