# -*- coding: utf-8 -*-
"""
Phase 13: The Wetware Hypervisor
Can NeuOS learn to control an unknown nonlinear device?

Uses DAgger (Dataset Aggregation): iteratively collect on-policy data
with learned controller, label with PID expert, retrain.

This is pure CPU simulation (no GPU needed).
"""
import json, os, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42)


class HillMuscle:
    """Hill-type muscle model: nonlinear contraction with fatigue."""
    def __init__(self):
        self.reset()
        self.dt = 0.01

    def step(self, neural_signal):
        neural_signal = float(np.clip(neural_signal, 0, 1))
        self.activation += (neural_signal - self.activation) * 0.3
        fl = np.exp(-((self.length - 1.0) ** 2) / 0.1)
        if self.velocity <= 0:
            fv = (1.0 + self.velocity * 0.5) / (1.0 - self.velocity * 0.3)
        else:
            fv = 1.3 - 0.3 / (1.0 + self.velocity * 0.5)
        fv = np.clip(fv, 0.0, 1.8)
        fatigue_factor = 1.0 - self.fatigue * 0.7
        self.fatigue = np.clip(self.fatigue + self.activation * 0.01 - 0.005, 0, 1)
        force = self.activation * fl * fv * fatigue_factor
        accel = force - 0.5 * (self.length - 1.0) - 0.1 * self.velocity
        self.velocity += accel * self.dt
        self.length = np.clip(self.length + self.velocity * self.dt, 0.3, 1.7)
        return self.length

    def reset(self):
        self.length = 1.0
        self.velocity = 0.0
        self.fatigue = 0.0
        self.activation = 0.0


def pid_expert(target, muscle):
    """PID expert that knows nothing about the muscle model."""
    error = target - muscle.length
    # Simple proportional control scaled for the muscle
    return float(np.clip(0.5 + 5.0 * error, 0, 1))


def main():
    print("[P13] The Wetware Hypervisor (DAgger)")
    start_time = time.time()

    muscle = HillMuscle()

    # === DAgger iterations ===
    all_X = []
    all_Y = []
    controller = None
    scaler = StandardScaler()
    n_dagger = 5
    dagger_history = []

    for dagger_iter in range(n_dagger):
        iter_X = []
        iter_Y = []
        beta = max(0.0, 1.0 - dagger_iter * 0.3)  # mix ratio (expert vs learned)

        for episode in range(200):
            muscle.reset()
            target = np.random.uniform(0.5, 1.4)

            for step in range(200):
                error = target - muscle.length
                features = [target, muscle.length, muscle.velocity, muscle.fatigue, error]

                # Get expert action (always)
                expert_action = pid_expert(target, muscle)

                # Get learned action
                if controller is not None and np.random.random() > beta:
                    state_scaled = scaler.transform([features])
                    learned_action = float(np.clip(controller.predict(state_scaled)[0], 0, 1))
                    action = learned_action  # use learned policy
                else:
                    action = expert_action  # use expert

                # Add noise for exploration
                action = float(np.clip(action + np.random.normal(0, 0.05), 0, 1))

                iter_X.append(features)
                iter_Y.append(expert_action)  # always label with expert

                muscle.step(action)

        all_X.extend(iter_X)
        all_Y.extend(iter_Y)

        X = np.array(all_X)
        Y = np.array(all_Y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        controller = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu', max_iter=300,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, learning_rate_init=0.001,
        )
        controller.fit(X_scaled, Y)

        # Evaluate
        eval_errors = []
        for target in [0.5, 0.7, 1.0, 1.3]:
            muscle.reset()
            for step in range(300):
                error = target - muscle.length
                state = scaler.transform([[target, muscle.length, muscle.velocity,
                                           muscle.fatigue, error]])
                signal = float(np.clip(controller.predict(state)[0], 0, 1))
                muscle.step(signal)
            eval_errors.append(abs(muscle.length - target))

        mean_err = np.mean(eval_errors)
        dagger_history.append({'iter': dagger_iter, 'mean_error': round(float(mean_err), 4),
                               'n_samples': len(X), 'beta': round(beta, 2)})
        print(f"    DAgger iter {dagger_iter}: mean_error={mean_err:.3f} "
              f"beta={beta:.1f} samples={len(X)}")

    # === Final evaluation ===
    print("  Final evaluation...")
    targets = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    results = {}

    for target in targets:
        muscle.reset()
        trajectory = []
        for step in range(300):
            error = target - muscle.length
            state = scaler.transform([[target, muscle.length, muscle.velocity,
                                       muscle.fatigue, error]])
            signal = float(np.clip(controller.predict(state)[0], 0, 1))
            muscle.step(signal)
            trajectory.append(muscle.length)

        final_error = abs(trajectory[-1] - target)
        results[str(target)] = {
            'final_error': round(float(final_error), 4),
            'final_length': round(float(trajectory[-1]), 4),
        }
        status = "OK" if final_error < 0.1 else "MISS"
        print(f"    Target={target:.1f}: final={trajectory[-1]:.3f} "
              f"err={final_error:.3f} {status}")

    success_rate = sum(1 for r in results.values() if r['final_error'] < 0.1) / len(targets)
    print(f"    Success rate (<0.1 error): {success_rate:.1%}")

    # Save
    output = {
        'phase': 13, 'name': 'wetware_hypervisor',
        'method': 'DAgger',
        'n_dagger_iters': n_dagger,
        'dagger_history': dagger_history,
        'success_rate': round(success_rate, 4),
        'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase13_wetware.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Trajectories
    for target in [0.5, 0.7, 1.0, 1.3]:
        muscle.reset()
        traj = []
        for step in range(300):
            error = target - muscle.length
            state = scaler.transform([[target, muscle.length, muscle.velocity,
                                       muscle.fatigue, error]])
            signal = float(np.clip(controller.predict(state)[0], 0, 1))
            muscle.step(signal)
            traj.append(muscle.length)
        axes[0].plot(traj, label=f'target={target}', linewidth=2)
        axes[0].axhline(y=target, linestyle='--', alpha=0.3)

    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Muscle Length', fontsize=12)
    axes[0].set_title('Muscle Control Trajectories', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # DAgger convergence
    iters = [d['iter'] for d in dagger_history]
    errs = [d['mean_error'] for d in dagger_history]
    axes[1].plot(iters, errs, 'o-', linewidth=2, markersize=8, color='tab:blue')
    axes[1].set_xlabel('DAgger Iteration', fontsize=12)
    axes[1].set_ylabel('Mean Error', fontsize=12)
    axes[1].set_title('DAgger Convergence', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Summary
    axes[2].axis('off')
    summary = (
        f"Wetware Hypervisor (DAgger)\n\n"
        f"DAgger iters: {n_dagger}\n"
        f"Final samples: {len(all_X)}\n"
        f"Success: {success_rate:.0%}\n\n"
        f"Hill Muscle Model:\n"
        f" Force-length + velocity\n"
        f" + fatigue dynamics\n\n"
        f"{'Wetware adapted!' if success_rate > 0.5 else 'Partial adaptation'}"
    )
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=12, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 13: Wetware Hypervisor\nCan NeuOS learn to control a nonlinear biological device?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase13_wetware.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")


if __name__ == '__main__':
    main()
