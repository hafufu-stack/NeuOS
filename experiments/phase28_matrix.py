# -*- coding: utf-8 -*-
"""
Phase 28: Execution Matrix (Deep Think P22 - full 24x24 transplant)
Create a complete donor(i) -> target(j) transplant heatmap.
Extract from task A at layer i, inject into task B at layer j.

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

# Use every 2 layers for speed (12x12 instead of 24x24)
LAYERS = list(range(0, 24, 2))


def main():
    print("[P28] Execution Matrix (24x24 transplant)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)

    # Source: MIN task, Target: ADD task
    min_prompts = [f"def f(): return min({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:8]
    add_prompts = [f"def f(): return {a} + {b} =" for a in range(1,6) for b in range(1,6) if a+b<10][:8]

    # Extract MIN vectors at all layers
    print("  Extracting MIN vectors at all layers...")
    min_vecs = {}
    for layer in LAYERS:
        vecs = []
        for prompt in min_prompts:
            captured = [None]
            def cap(module, input, output):
                captured[0] = get_last_token(output)
            h = model.model.layers[layer].register_forward_hook(cap)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            vecs.append(captured[0])
        min_vecs[layer] = torch.stack(vecs).mean(dim=0)

    # Test: inject MIN vec from layer i at layer j during ADD
    test_data = [
        (f"def f(): return {a} + {b} =", a, b) for a in range(2,7) for b in range(2,7) if a+b<10
    ][:6]

    print(f"  Building {len(LAYERS)}x{len(LAYERS)} execution matrix...")
    matrix = np.zeros((len(LAYERS), len(LAYERS)))

    for i_idx, donor_layer in enumerate(LAYERS):
        donor_vec = min_vecs[donor_layer]
        for j_idx, target_layer in enumerate(LAYERS):
            correct = 0
            total = 0
            for prompt, a, b in test_data:
                expected_min = min(a, b)
                total += 1

                def inject(module, input, output, v=donor_vec):
                    return replace_last_token(output, v)

                h = model.model.layers[target_layer].register_forward_hook(inject)
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                h.remove()

                pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
                if pred == str(expected_min):
                    correct += 1

            matrix[i_idx, j_idx] = correct / total if total > 0 else 0

        print(f"    Donor L{donor_layer}: best target L{LAYERS[np.argmax(matrix[i_idx])]}"
              f" ({matrix[i_idx].max():.0%})")

    # Save
    output = {
        'phase': 28, 'name': 'execution_matrix',
        'layers': LAYERS,
        'matrix': matrix.tolist(),
        'task_donor': 'MIN', 'task_target': 'ADD',
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase28_matrix.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f'L{l}' for l in LAYERS], fontsize=8)
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels([f'L{l}' for l in LAYERS], fontsize=8)
    ax.set_xlabel('Target Layer (injected into ADD)', fontsize=11)
    ax.set_ylabel('Donor Layer (extracted from MIN)', fontsize=11)
    plt.colorbar(im, label='Takeover Rate (outputs MIN instead of SUM)')
    ax.set_title('Phase 28: Execution Matrix\nMIN donor -> ADD target transplant', fontsize=13, fontweight='bold')

    # Annotate cells
    for i in range(len(LAYERS)):
        for j in range(len(LAYERS)):
            if matrix[i, j] > 0:
                ax.text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center',
                       fontsize=7, color='white' if matrix[i,j]>0.5 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase28_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
