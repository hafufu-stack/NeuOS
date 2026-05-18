# -*- coding: utf-8 -*-
"""
Phase 6: Blackbox Device Probing (CPU)
Can a neural network learn the behavior of an unknown "device"
from random input-output probing alone?

Simulate a "mystery device" with unknown transfer function.
Agent sends random inputs, observes outputs, builds a model.
Measure: how many probes needed to predict device behavior at 95%?

Model: None (pure simulation + small MLP, CPU only)
"""
import json, os, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42)


class MysteryDevice:
    """A device with an unknown transfer function."""
    def __init__(self, device_type, input_dim=4, output_dim=2):
        self.type = device_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        np.random.seed(hash(device_type) % 2**31)
        self.W = np.random.randn(input_dim, output_dim) * 0.5
        self.b = np.random.randn(output_dim) * 0.3
        self.nonlinearity = device_type  # 'linear', 'relu', 'sigmoid', 'quadratic'

    def forward(self, x):
        z = x @ self.W + self.b
        if self.nonlinearity == 'linear':
            return z
        elif self.nonlinearity == 'relu':
            return np.maximum(0, z)
        elif self.nonlinearity == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.nonlinearity == 'quadratic':
            return z ** 2
        elif self.nonlinearity == 'periodic':
            return np.sin(z * 2)
        return z


class NeuralProber:
    """Learns device behavior from random probing."""
    def __init__(self, input_dim, output_dim, hidden=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        # Simple 2-layer MLP weights
        self.W1 = np.random.randn(input_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        self.lr = 0.01

    def predict(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def train_step(self, x, y_true):
        # Forward
        z1 = x @ self.W1 + self.b1
        h = np.maximum(0, z1)
        y_pred = h @ self.W2 + self.b2
        # Loss
        loss = np.mean((y_pred - y_true) ** 2)
        # Backward (manual grad)
        dy = 2 * (y_pred - y_true) / y_true.shape[0]
        dW2 = h.T @ dy
        db2 = dy.sum(axis=0)
        dh = dy @ self.W2.T
        dz1 = dh * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return loss


def probe_device(device, max_probes=500, batch_size=16):
    """Motor babbling: send random inputs, learn from outputs."""
    prober = NeuralProber(device.input_dim, device.output_dim)
    # Generate test set
    X_test = np.random.randn(100, device.input_dim)
    Y_test = device.forward(X_test)

    history = []
    for probe_round in range(max_probes // batch_size):
        # Random probe
        X_probe = np.random.randn(batch_size, device.input_dim)
        Y_probe = device.forward(X_probe)

        # Train on observation
        for _ in range(5):
            loss = prober.train_step(X_probe, Y_probe)

        # Evaluate
        Y_pred = prober.predict(X_test)
        mse = np.mean((Y_pred - Y_test) ** 2)
        # R^2
        ss_res = np.sum((Y_test - Y_pred) ** 2)
        ss_tot = np.sum((Y_test - Y_test.mean()) ** 2)
        r2 = 1 - ss_res / max(1e-10, ss_tot)

        n_probes = (probe_round + 1) * batch_size
        history.append({'probes': n_probes, 'mse': float(mse), 'r2': float(r2)})

    return history


def main():
    print("[P6] Blackbox Device Probing")
    start_time = time.time()

    device_types = ['linear', 'relu', 'sigmoid', 'quadratic', 'periodic']
    all_results = {}

    for dtype in device_types:
        device = MysteryDevice(dtype)
        history = probe_device(device)
        all_results[dtype] = history

        # Find probe count for R^2 > 0.95
        probe_95 = None
        for h in history:
            if h['r2'] >= 0.95:
                probe_95 = h['probes']
                break

        final_r2 = history[-1]['r2']
        print(f"  {dtype}: R2={final_r2:.3f} "
              f"(95% at {probe_95 if probe_95 else '>500'} probes)")

    # Save
    output = {
        'phase': 6, 'name': 'blackbox_device_probing',
        'device_types': device_types,
        'results': all_results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase6_blackbox.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for i, dtype in enumerate(device_types):
        probes = [h['probes'] for h in all_results[dtype]]
        r2s = [h['r2'] for h in all_results[dtype]]
        mses = [h['mse'] for h in all_results[dtype]]
        axes[0].plot(probes, r2s, '-', linewidth=2, label=dtype, color=colors[i])
        axes[1].plot(probes, mses, '-', linewidth=2, label=dtype, color=colors[i])

    axes[0].set_xlabel('Number of Probes', fontsize=12)
    axes[0].set_ylabel('R^2 Score', fontsize=12)
    axes[0].set_title('Device Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    axes[1].set_xlabel('Number of Probes', fontsize=12)
    axes[1].set_ylabel('MSE (log scale)', fontsize=12)
    axes[1].set_title('Prediction Error', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Phase 6: Blackbox Device Probing\nMotor Babbling -> Device Model',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase6_blackbox.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")


if __name__ == '__main__':
    main()
