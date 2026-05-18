# -*- coding: utf-8 -*-
"""
Phase 19: Symbiotic Polymorphism (Self-modifying wetware control)
P13 failed with fixed weights. Solution: use the LLM itself to
dynamically adjust control signals based on feedback.

Method:
  1. Present the control problem as a text prompt with sensor readings
  2. Let the LLM generate the control signal
  3. Feed back the error and iterate
  The "program" (prompt context / KV cache) evolves with each cycle.

This is CPU-only simulation + GPU for LLM inference.
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


class SimpleMuscle:
    """Simplified muscle: activation (0-1) -> length (0.3-1.7)."""
    def __init__(self):
        self.reset()

    def step(self, signal):
        signal = float(np.clip(signal, 0, 9))
        # Simple nonlinear response: length = 1.0 + 0.5*tanh(signal - 5)
        target_length = 1.0 - 0.4 * np.tanh((signal - 5) * 0.5)
        self.length += (target_length - self.length) * 0.3
        self.length = np.clip(self.length, 0.3, 1.7)
        return self.length

    def reset(self):
        self.length = 1.0


def main():
    print("[P19] Symbiotic Polymorphism (LLM-in-the-loop control)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    muscle = SimpleMuscle()

    targets = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    results = {}
    all_trajectories = {}

    for target in targets:
        muscle.reset()
        trajectory = []
        signals = []

        # Build evolving prompt: each cycle adds feedback
        history = f"# Control muscle to length {target:.1f}\n"
        history += "# signal(0-9) -> length. Higher signal = shorter.\n"

        for cycle in range(15):
            # Current state
            prompt = history + f"# current={muscle.length:.2f} target={target:.1f}\n"
            prompt += "# signal="

            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)

            # Get predicted signal (expect a digit 0-9)
            logits = out.logits[0, -1, :]
            # Get probabilities for digits 0-9
            digit_ids = [tok.encode(str(d), add_special_tokens=False)[-1] for d in range(10)]
            digit_logits = logits[digit_ids]
            pred_digit = digit_logits.argmax().item()
            signal = float(pred_digit)

            # Apply to muscle
            new_length = muscle.step(signal)
            trajectory.append(new_length)
            signals.append(signal)

            # Add feedback to history (evolving context = self-modifying program)
            history += f"# current={muscle.length:.2f} target={target:.1f} signal={int(signal)} -> {new_length:.2f}\n"

        final_error = abs(trajectory[-1] - target)
        results[str(target)] = {
            'final_error': round(float(final_error), 4),
            'final_length': round(float(trajectory[-1]), 4),
            'signals': [int(s) for s in signals],
        }
        all_trajectories[str(target)] = trajectory
        status = "OK" if final_error < 0.1 else "MISS"
        print(f"    Target={target:.1f}: final={trajectory[-1]:.3f} "
              f"err={final_error:.3f} signals={[int(s) for s in signals[:5]]}... {status}")

    success_rate = sum(1 for r in results.values() if r['final_error'] < 0.1) / len(targets)
    print(f"    Success rate (<0.1 error): {success_rate:.1%}")

    # Save
    output = {
        'phase': 19, 'name': 'symbiotic_polymorphism',
        'n_targets': len(targets), 'n_cycles': 15,
        'success_rate': round(success_rate, 4),
        'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase19_symbiotic.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for target in targets:
        traj = all_trajectories[str(target)]
        axes[0].plot(traj, 'o-', linewidth=2, markersize=4, label=f'target={target}')
        axes[0].axhline(y=target, linestyle='--', alpha=0.2)
    axes[0].set_xlabel('Cycle', fontsize=12)
    axes[0].set_ylabel('Muscle Length', fontsize=12)
    axes[0].set_title('LLM-in-the-Loop Control', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Signal patterns
    for target in [0.7, 1.0, 1.2]:
        sigs = results[str(target)]['signals']
        axes[1].plot(sigs, 'o-', linewidth=2, markersize=4, label=f'target={target}')
    axes[1].set_xlabel('Cycle', fontsize=12)
    axes[1].set_ylabel('Signal (0-9)', fontsize=12)
    axes[1].set_title('Control Signals Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].axis('off')
    summary = (
        f"Symbiotic Polymorphism\n\n"
        f"Targets: {len(targets)}\n"
        f"Cycles: 15\n"
        f"Success: {success_rate:.0%}\n\n"
        f"P13 (fixed MLP): 0%\n"
        f"P19 (LLM-in-loop): {success_rate:.0%}\n\n"
        f"{'Self-modifying works!' if success_rate > 0.3 else 'Investigating...'}"
    )
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 19: Symbiotic Polymorphism\nCan the LLM adapt its own program to control unknown hardware?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase19_symbiotic.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
