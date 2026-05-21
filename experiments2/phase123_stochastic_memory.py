# -*- coding: utf-8 -*-
"""
Phase 123: Stochastic Memory Palace (NeuOS x SNN-Genesis)
Test if adding optimal Gaussian noise to hidden states during multi-step
soul execution improves working memory (breaking P117's 1-2 step limit).

"Sometimes noise is the signal."
"""
import torch, json, os, gc, numpy as np, time, sys, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INJECT_LAYER = 16
SIGMAS = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
N_TEST_CASES = 20


def gradient_train(model, tok, train, layer, device, seed=42, epochs=150):
    """Standard gradient soul vector training."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in train:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(device)
            def inj(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def evaluate_single(model, tok, vec, prompt, expected, layer, device, sigma=0.0):
    """Run one forward pass with soul injection + optional noise."""
    def inj(m, i, o, v=vec, s=sigma):
        h = o[0].clone() if isinstance(o, tuple) else o.clone()
        # Replace last token with soul
        if h.dim() == 3:
            h[0, -1, :] = v.to(h.dtype)
            if s > 0:
                noise_scale = s * h.std().item()
                noise = torch.randn_like(h) * noise_scale
                h = h + noise
                # Re-set the soul vector after noise (keep it clean)
                h[0, -1, :] = v.to(h.dtype)
        elif h.dim() == 2:
            h[-1, :] = v.to(h.dtype)
            if s > 0:
                noise_scale = s * h.std().item()
                noise = torch.randn_like(h) * noise_scale
                h = h + noise
                h[-1, :] = v.to(h.dtype)
        if isinstance(o, tuple):
            return (h,) + o[1:]
        return h

    hook = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    hook.remove()
    pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
    return pred == expected, pred


def generate_test_sequences(n_cases, rng):
    """Generate random test sequences of varying lengths (2-5 digits)."""
    test_cases = []
    for _ in range(n_cases):
        length = rng.randint(2, 6)  # 2 to 5 inclusive
        seq = [rng.randint(1, 10) for _ in range(length)]  # 1 to 9
        test_cases.append(seq)
    return test_cases


def run_multistep_min(model, tok, vec, sequence, layer, device, sigma=0.0):
    """
    Run multi-step MIN on a sequence.
    For [a, b, c, d]:
      Step 1: min(a, b) -> r1 via prompt 'a, b) ='
      Step 2: min(r1, c) -> r2 via prompt 'r1, c) ='
      Step 3: min(r2, d) -> r3 via prompt 'r2, d) ='
    Returns (list of step results, list of step correctness).
    """
    n_steps = len(sequence) - 1
    current_val = sequence[0]
    step_results = []
    step_correct = []

    for step in range(n_steps):
        next_val = sequence[step + 1]
        expected_min = min(current_val, next_val)
        prompt = f"{current_val}, {next_val}) ="
        expected_str = str(expected_min)

        # First step: no noise. Subsequent steps: apply sigma
        step_sigma = 0.0 if step == 0 else sigma

        correct, pred = evaluate_single(
            model, tok, vec, prompt, expected_str, layer, device, step_sigma)
        step_results.append(pred)
        step_correct.append(correct)

        # Use predicted value for next step (even if wrong, to test cascading)
        try:
            current_val = int(pred)
        except ValueError:
            current_val = -1  # Will cause subsequent steps to fail

    return step_results, step_correct


def main():
    print("[P123] Stochastic Memory Palace")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    # Step 1: Train MIN soul vector
    print("  Training MIN soul vector (150 epochs)...")
    min_train = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")]
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")]
    min_vec = gradient_train(model, tok, min_train, INJECT_LAYER, DEVICE,
                             seed=42, epochs=150)

    # Quick sanity check
    base_acc = 0
    for p_str, e in min_train + min_test:
        ok, _ = evaluate_single(model, tok, min_vec, p_str, e,
                                INJECT_LAYER, DEVICE, sigma=0.0)
        base_acc += ok
    base_acc /= len(min_train + min_test)
    print(f"    Baseline single-step accuracy: {base_acc:.2%}")

    # Step 2: Generate test sequences
    rng = np.random.RandomState(42)
    test_sequences = generate_test_sequences(N_TEST_CASES, rng)

    # Group by step count
    by_steps = {}
    for seq in test_sequences:
        n = len(seq) - 1  # number of steps
        if n not in by_steps:
            by_steps[n] = []
        by_steps[n].append(seq)

    step_counts = sorted(by_steps.keys())
    print(f"  Test sequences: {len(test_sequences)} total")
    for sc in step_counts:
        print(f"    {sc} steps: {len(by_steps[sc])} sequences")

    # Step 3: Sweep sigma x steps
    print("  Running sigma sweep...")
    # results[sigma][n_steps] = accuracy
    results = {}
    detailed_results = []

    for sigma in SIGMAS:
        results[sigma] = {}
        print(f"    sigma={sigma:.2f}:")
        for n_steps in step_counts:
            correct_total = 0
            total = 0
            for seq in by_steps[n_steps]:
                # Run multiple times for stochastic evaluation (if sigma > 0)
                n_repeats = 5 if sigma > 0 else 1
                seq_correct = 0
                for rep in range(n_repeats):
                    _, step_ok = run_multistep_min(
                        model, tok, min_vec, seq, INJECT_LAYER, DEVICE, sigma)
                    # Final answer correct?
                    if step_ok[-1]:
                        seq_correct += 1
                correct_total += seq_correct
                total += n_repeats

            acc = correct_total / total if total > 0 else 0
            results[sigma][n_steps] = acc
            detailed_results.append({
                'sigma': sigma, 'steps': n_steps,
                'accuracy': round(acc, 4), 'n_trials': total
            })
            print(f"      {n_steps} steps: acc={acc:.2%} ({correct_total}/{total})")

    # Step 4: Save results
    output = {
        'phase': 123, 'name': 'stochastic_memory_palace',
        'base_accuracy': round(base_acc, 4),
        'inject_layer': INJECT_LAYER,
        'sigmas': SIGMAS,
        'step_counts': step_counts,
        'n_test_cases': N_TEST_CASES,
        'results_by_sigma_steps': detailed_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase123_stochastic_memory.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print("  Results saved.")

    # Step 5: Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Heatmap (steps x sigma)
    ax = axes[0]
    heat_data = np.zeros((len(step_counts), len(SIGMAS)))
    for i, ns in enumerate(step_counts):
        for j, sig in enumerate(SIGMAS):
            heat_data[i, j] = results[sig].get(ns, 0)

    im = ax.imshow(heat_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(SIGMAS)))
    ax.set_xticklabels([f'{s:.2f}' for s in SIGMAS], fontsize=8, rotation=45)
    ax.set_yticks(range(len(step_counts)))
    ax.set_yticklabels([f'{s} steps' for s in step_counts])
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('Number of steps')
    for i in range(len(step_counts)):
        for j in range(len(SIGMAS)):
            ax.text(j, i, f'{heat_data[i, j]:.0%}',
                    ha='center', va='center', fontsize=8,
                    color='white' if heat_data[i, j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax)
    ax.set_title('Accuracy: Steps x Noise', fontweight='bold')

    # Panel 2: Line plot (accuracy vs sigma for each step count)
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(step_counts)))
    for i, ns in enumerate(step_counts):
        accs = [results[sig].get(ns, 0) for sig in SIGMAS]
        ax.plot(SIGMAS, accs, 'o-', color=colors[i], label=f'{ns} steps',
                linewidth=2, markersize=6)
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('Final-step Accuracy')
    ax.set_title('Stochastic Resonance Curve', fontweight='bold')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=base_acc, color='gray', linestyle='--', alpha=0.5,
               label='Baseline (1-step)')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 123: Stochastic Memory Palace\n'
                 '"Sometimes noise is the signal"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase123_stochastic_memory.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  === Summary ===")
    print(f"  Baseline single-step acc: {base_acc:.2%}")
    for ns in step_counts:
        best_sig = max(SIGMAS, key=lambda s: results[s].get(ns, 0))
        best_acc = results[best_sig].get(ns, 0)
        no_noise_acc = results[0.0].get(ns, 0)
        print(f"  {ns} steps: no-noise={no_noise_acc:.2%}, "
              f"best sigma={best_sig:.2f} -> {best_acc:.2%}")
    print(f"  Completed in {time.time()-start:.0f}s")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
