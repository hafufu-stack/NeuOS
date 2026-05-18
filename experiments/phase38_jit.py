# -*- coding: utf-8 -*-
"""
Phase 38: Cybernetic JIT Compilation
Use P35's gradient synthesis to compile a muscle control driver.
No text prompts - pure latent-space driver synthesis.

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


def hill_muscle(activation, length=1.0, f_max=100.0):
    """Hill-type muscle model."""
    f_l = np.exp(-((length - 1.0) ** 2) / 0.2)
    force = f_max * activation * f_l
    return force


def main():
    print("[P38] Cybernetic JIT Compilation")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size

    for p in model.parameters():
        p.requires_grad = False

    # Target: compile a control driver that outputs the correct activation
    # to achieve target forces via the Hill muscle model
    target_layer = 8  # Injection point (P35 showed L8 works)

    # Training data: for each target force, what activation is needed?
    # force = f_max * activation * f_l(1.0) = 100 * activation * 1.0
    # So activation = force / 100
    # We map: prompt with target force info -> output correct digit (activation * 10)
    train_data = []
    for target_force in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        activation = target_force / 100.0
        output_digit = int(activation * 10)  # 1-9
        if output_digit > 9:
            output_digit = 9
        prompt = f"force={target_force}) ="
        train_data.append((prompt, str(output_digit), target_force, activation))

    print(f"  Training data: {len(train_data)} cases")

    # Compile the control driver vector
    driver_vec = torch.randn(hidden_size, device=DEVICE) * 0.01
    driver_vec.requires_grad_(True)
    optimizer = torch.optim.Adam([driver_vec], lr=0.01)

    loss_history = []
    acc_history = []

    n_epochs = 150
    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        for prompt, target_str, target_force, activation in train_data:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)

            def inject(module, input, output, v=driver_vec):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()

            logits = out.logits[0, -1, :]
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([target_id]).to(DEVICE))
            total_loss += loss.item()
            if logits.argmax().item() == target_id:
                correct += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_data)
        train_acc = correct / len(train_data)
        loss_history.append(avg_loss)
        acc_history.append(train_acc)
        if epoch % 30 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch}: loss={avg_loss:.3f}, acc={train_acc:.1%}")

    # Test: closed-loop simulation
    print("\n  Closed-loop muscle control simulation...")
    driver_eval = driver_vec.detach()
    trajectory = []
    test_forces = [20, 50, 80, 30, 70, 10, 90, 40, 60]

    for target_force in test_forces:
        prompt = f"force={target_force}) ="
        def inject_eval(module, input, output, v=driver_eval):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_eval)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()

        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        try:
            pred_activation = float(pred) / 10.0
        except ValueError:
            pred_activation = 0.0

        actual_force = hill_muscle(pred_activation)
        error = abs(target_force - actual_force)
        trajectory.append({
            'target_force': target_force,
            'pred_digit': pred,
            'pred_activation': round(pred_activation, 2),
            'actual_force': round(actual_force, 1),
            'error': round(error, 1),
        })
        print(f"    Target={target_force}, pred_act={pred_activation:.1f}, "
              f"actual_force={actual_force:.0f}, error={error:.0f}")

    avg_error = np.mean([t['error'] for t in trajectory])
    perfect = sum(1 for t in trajectory if t['error'] < 5.0)
    ctrl_acc = perfect / len(trajectory)

    # Save
    output = {
        'phase': 38, 'name': 'cybernetic_jit',
        'final_train_acc': round(train_acc, 4),
        'avg_control_error': round(float(avg_error), 2),
        'control_accuracy': round(ctrl_acc, 4),
        'trajectory': trajectory,
        'loss_history': [round(l, 4) for l in loss_history],
        'acc_history': [round(a, 4) for a in acc_history],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase38_jit.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Driver Compilation', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(acc_history, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy', fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)

    targets = [t['target_force'] for t in trajectory]
    actuals = [t['actual_force'] for t in trajectory]
    axes[2].plot(targets, targets, 'r--', linewidth=1, label='Perfect')
    axes[2].scatter(targets, actuals, c='tab:blue', s=80, zorder=5, edgecolors='black')
    axes[2].set_xlabel('Target Force')
    axes[2].set_ylabel('Actual Force')
    axes[2].set_title(f'Control: avg error={avg_error:.0f}, acc={ctrl_acc:.0%}', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 38: Cybernetic JIT Compilation\nGradient-compiled muscle control driver',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase38_jit.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
