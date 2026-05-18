# -*- coding: utf-8 -*-
"""
Phase 32: 1.5B Wetware Hypervisor (Deep Think P23/P29)
Scale up to Qwen2.5-1.5B and retry bio-device control.
Use Hill-type muscle model with DMA-based real-time control loop.

Model: Qwen2.5-1.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import get_last_token, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def hill_muscle(activation, length=1.0, v_max=10.0, f_max=100.0):
    """Hill-type muscle: nonlinear force-velocity-activation model."""
    # Force-length: bell curve centered at optimal length
    f_l = np.exp(-((length - 1.0) ** 2) / 0.2)
    # Force-velocity: simplified
    force = f_max * activation * f_l
    # Velocity depends on force and load
    velocity = v_max * (1 - force / (f_max + 1e-8)) * activation
    return force, velocity


def load_1_5b():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-1.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    return model, tok


def main():
    print("[P32] 1.5B Wetware Hypervisor")
    print(f"  Device: {DEVICE}")
    start = time.time()

    model, tok = load_1_5b()
    n_layers = model.config.num_hidden_layers
    print(f"  Model: Qwen2.5-1.5B ({n_layers} layers)")

    # === Step 1: ISA mapping on 1.5B (quick check) ===
    print("\n  Step 1: Quick ISA check on 1.5B...")
    # Check if the register architecture scales
    isa_prompts = {
        'MIN': [f"def f(): return min({a}, {b}) =" for a in [3,5,7] for b in [2,4,6] if a!=b],
        'MAX': [f"def f(): return max({a}, {b}) =" for a in [3,5,7] for b in [2,4,6] if a!=b],
        'SUM': [f"def f(): return {a} + {b} =" for a in [1,2,3] for b in [1,2,3] if a+b<10],
    }

    isa_results = {}
    for op_name, prompts in isa_prompts.items():
        correct = 0
        total = len(prompts)
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            expected = str(eval(prompt.split('return ')[-1].split(' =')[0]))
            if pred == expected:
                correct += 1
        isa_results[op_name] = round(correct/total, 4)
        print(f"    {op_name}: {correct}/{total} = {correct/total:.0%}")

    # === Step 2: Muscle control via prompt ===
    print("\n  Step 2: Muscle control via prompt-based ICL...")
    target_force = 50.0
    trajectory = []
    activation = 0.5
    length = 1.0

    # Give model history and ask for next activation
    history = []
    n_steps = 15
    for step in range(n_steps):
        force, velocity = hill_muscle(activation, length)
        error = target_force - force
        history.append({'step': step, 'activation': round(activation, 3),
                        'force': round(force, 2), 'error': round(error, 2)})
        trajectory.append({'step': step, 'activation': activation, 'force': force,
                          'error': error, 'length': length})

        # Build prompt with history
        hist_str = "; ".join([f"a={h['activation']},f={h['force']},e={h['error']}"
                              for h in history[-5:]])
        prompt = (f"Muscle controller. Target force={target_force}. "
                  f"History: {hist_str}. "
                  f"Next activation (0-1): ")

        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=5, do_sample=False)
        response = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        try:
            new_act = float(response.split()[0].rstrip('.,;'))
            new_act = max(0.0, min(1.0, new_act))
        except (ValueError, IndexError):
            new_act = activation + 0.05 * np.sign(error)
            new_act = max(0.0, min(1.0, new_act))

        activation = new_act
        if step < 5 or step == n_steps-1:
            print(f"    Step {step}: act={activation:.3f}, force={force:.1f}, "
                  f"error={error:.1f}, response='{response[:20]}'")

    final_error = abs(trajectory[-1]['error'])
    converged = final_error < 10.0

    # === Step 3: DMA-style register control (1.5B) ===
    print("\n  Step 3: DMA register injection test on 1.5B...")
    # Extract MIN vector at a middle layer
    mid_layer = n_layers // 2  # ~14 for 28-layer model
    min_prompts_1_5b = [f"def f(): return min({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:8]
    min_vecs = []
    for p in min_prompts_1_5b:
        cap = [None]
        def capture(module, input, output):
            cap[0] = get_last_token(output)
        h = model.model.layers[mid_layer].register_forward_hook(capture)
        inp = tok(p, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        min_vecs.append(cap[0])
    min_vec = torch.stack(min_vecs).mean(dim=0)

    # DMA test
    dma_data = [("3, 7) =", 3, 7), ("5, 2) =", 5, 2), ("8, 1) =", 8, 1),
                ("4, 6) =", 4, 6), ("9, 3) =", 9, 3), ("7, 2) =", 7, 2)]
    dma_correct = 0
    dma_total = 0
    for data_str, a, b in dma_data:
        expected = min(a, b)
        dma_total += 1
        def inject(module, input, output, v=min_vec):
            return replace_last_token(output, v)
        h = model.model.layers[mid_layer].register_forward_hook(inject)
        inp = tok(data_str, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == str(expected):
            dma_correct += 1
    dma_acc = dma_correct / dma_total
    print(f"    1.5B DMA (MIN@L{mid_layer}): {dma_acc:.1%}")

    # Save
    output = {
        'phase': 32, 'name': 'wetware_1_5b',
        'model': 'Qwen2.5-1.5B', 'n_layers': n_layers,
        'isa_results': isa_results,
        'muscle_control': {
            'target_force': target_force,
            'final_error': round(final_error, 2),
            'converged': converged,
            'trajectory': [{'step': t['step'], 'activation': round(t['activation'], 3),
                           'force': round(t['force'], 2), 'error': round(t['error'], 2)}
                          for t in trajectory],
        },
        'dma_1_5b': {'layer': mid_layer, 'accuracy': round(dma_acc, 4)},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase32_wetware.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # ISA check
    ops = list(isa_results.keys())
    accs = [isa_results[op] for op in ops]
    axes[0].bar(ops, accs, color=['tab:green','tab:red','tab:blue'], edgecolor='black')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('1.5B ISA Check', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Muscle trajectory
    steps = [t['step'] for t in trajectory]
    forces = [t['force'] for t in trajectory]
    axes[1].plot(steps, forces, 'bo-', linewidth=2, markersize=5, label='Force')
    axes[1].axhline(y=target_force, color='red', linestyle='--', label=f'Target ({target_force})')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Force')
    axes[1].set_title(f'Muscle Control (1.5B)\nFinal error: {final_error:.1f}', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # DMA comparison
    axes[2].bar(['0.5B DMA\n(P22)', '1.5B DMA'], [0.667, dma_acc],
                color=['tab:orange', 'tab:blue'], edgecolor='black')
    axes[2].set_ylabel('DMA Accuracy')
    axes[2].set_title('DMA: 0.5B vs 1.5B', fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    axes[2].text(0, 0.667+0.03, '66.7%', ha='center', fontweight='bold')
    axes[2].text(1, dma_acc+0.03, f'{dma_acc:.0%}', ha='center', fontweight='bold')

    plt.suptitle('Phase 32: 1.5B Wetware Hypervisor\nScaling NeuOS to larger models',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase32_wetware.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
