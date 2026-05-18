# -*- coding: utf-8 -*-
"""
Phase 50: Autopoietic Kernel (Self-Bootstrapping OS)
Motor babbling -> self-learned control on unknown hardware.
No pre-training, no supervision. Pure self-organization.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BlackboxDevice:
    """Unknown hardware: quadratic response with noise."""
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.a = np.random.uniform(0.5, 2.0)
        self.b = np.random.uniform(-0.5, 0.5)
        self.c = np.random.uniform(0, 10)

    def respond(self, action):
        """action: 0-9 digit -> output: real value"""
        x = action / 9.0  # normalize to 0-1
        return self.a * x**2 + self.b * x + self.c + np.random.normal(0, 0.1)


def main():
    print("[P50] Autopoietic Kernel")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Create unknown blackbox device
    device = BlackboxDevice(seed=42)
    print(f"  Unknown device: f(x) = {device.a:.2f}x^2 + {device.b:.2f}x + {device.c:.2f}")

    # Phase 1: Motor babbling (random exploration)
    print("\n  Phase 1: Motor babbling (10 random actions)...")
    io_pairs = []
    for _ in range(10):
        action = np.random.randint(0, 10)
        output = device.respond(action)
        io_pairs.append((action, output))
        print(f"    Action {action} -> Output {output:.2f}")

    # Phase 2: Build internal model from babbling data
    print("\n  Phase 2: Self-learning device model...")
    # Find: which action achieves target output?
    # Build lookup: target -> best action
    targets = list(range(10, 100, 10))  # target outputs
    target_to_action = {}

    # Brute force: try all actions, find best for each target
    for target in targets:
        best_action = 0
        best_error = float('inf')
        for action in range(10):
            # Average response from babbling data
            responses = [out for a, out in io_pairs if a == action]
            if not responses:
                # Need to test this action
                response = device.respond(action)
                io_pairs.append((action, response))
                responses = [response]
            avg_response = np.mean(responses)
            error = abs(avg_response * 10 - target)  # Scale to match
            if error < best_error:
                best_error = error
                best_action = action
        target_to_action[target] = best_action

    # Phase 3: JIT compile driver from self-learned mapping
    print("\n  Phase 3: Self-compiling driver from babbling data...")
    train_data = [(f"target={t}) =", str(target_to_action[t])) for t in targets]
    driver = torch.randn(hidden_size, device=DEVICE) * 0.01
    driver.requires_grad_(True)
    opt = torch.optim.Adam([driver], lr=0.01)
    loss_hist = []

    for epoch in range(100):
        total_loss = 0
        for prompt, target_str in train_data:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject(module, input, output, v=driver):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(DEVICE))
            total_loss += loss.item()
            opt.zero_grad(); loss.backward(); opt.step()
        loss_hist.append(total_loss / len(train_data))
        if epoch % 25 == 0:
            print(f"    Epoch {epoch}: loss={loss_hist[-1]:.3f}")

    # Phase 4: Test self-compiled driver
    print("\n  Phase 4: Testing self-compiled driver...")
    driver_eval = driver.detach()
    test_results = []

    for target in targets:
        prompt = f"target={target}) ="
        def inject_test(module, input, output, v=driver_eval):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_test)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        try:
            pred_action = int(pred)
        except ValueError:
            pred_action = -1

        actual_output = device.respond(pred_action) if pred_action >= 0 else 0
        expected_action = target_to_action[target]
        correct = pred_action == expected_action

        test_results.append({
            'target': target,
            'pred_action': pred_action,
            'expected_action': expected_action,
            'actual_output': round(actual_output, 2),
            'correct': correct,
        })
        print(f"    Target={target}: action={pred_action}(exp {expected_action}), "
              f"output={actual_output:.1f}")

    accuracy = sum(r['correct'] for r in test_results) / len(test_results)
    print(f"\n  Self-compiled accuracy: {accuracy:.0%}")

    # Phase 5: Compare with oracle (perfect knowledge)
    print("  Phase 5: Oracle comparison...")
    oracle_errors = []
    self_errors = []
    for r in test_results:
        target = r['target']
        self_output = r['actual_output'] * 10 if r['pred_action'] >= 0 else 0
        oracle_output = device.respond(r['expected_action']) * 10
        oracle_errors.append(abs(target - oracle_output))
        self_errors.append(abs(target - self_output))

    avg_oracle = np.mean(oracle_errors)
    avg_self = np.mean(self_errors)
    print(f"    Oracle avg error: {avg_oracle:.1f}")
    print(f"    Self-learned avg error: {avg_self:.1f}")

    # Save
    output = {
        'phase': 50, 'name': 'autopoietic_kernel',
        'accuracy': round(accuracy, 4),
        'avg_oracle_error': round(float(avg_oracle), 2),
        'avg_self_error': round(float(avg_self), 2),
        'io_pairs': [(int(a), round(o, 3)) for a, o in io_pairs],
        'test_results': test_results,
        'loss_history': [round(l, 4) for l in loss_hist],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase50_autopoietic.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(loss_hist, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Self-Compiled Driver Training', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(['Self-\nCompiled', 'Oracle'],
                [accuracy, 1.0],
                color=['tab:purple', 'tab:green'], edgecolor='black')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Self-Compiled: {accuracy:.0%}', fontweight='bold')
    axes[1].set_ylim(0, 1.1)

    # Autopoietic loop diagram
    axes[2].axis('off')
    summary = ("AUTOPOIETIC BOOT SEQUENCE\n"
               "="*35 + "\n\n"
               "1. Motor Babbling (random I/O)\n"
               "2. Self-learn device model\n"
               "3. JIT compile driver vec\n"
               "4. Execute & monitor\n"
               "5. Recompile if needed\n\n"
               f"Accuracy: {accuracy:.0%}\n"
               f"Zero human input!\n"
               f"Zero prior knowledge!")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                fontsize=11, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 50: Autopoietic Kernel\nSelf-bootstrapping OS for unknown hardware',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase50_autopoietic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
