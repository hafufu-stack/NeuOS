# -*- coding: utf-8 -*-
"""
Phase 56: Soul Migration
Transfer autopoietic kernel from 0.5B to 1.5B.
Same boot sequence, different body. Ghost in the Shell.

Model: Qwen2.5-0.5B -> Qwen2.5-1.5B
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
    """Same device as P50 for fair comparison."""
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.a = np.random.uniform(0.5, 2.0)
        self.b = np.random.uniform(-0.5, 0.5)
        self.c = np.random.uniform(0, 10)

    def respond(self, action):
        x = action / 9.0
        return self.a * x**2 + self.b * x + self.c + np.random.normal(0, 0.1)


def autopoietic_boot(model, tok, device_hw, target_layer, device_torch, body_name):
    """The autopoietic kernel: same code, any body."""
    hidden_size = model.config.hidden_size

    # Phase 1: Motor babbling
    print(f"    [{body_name}] Motor babbling...")
    io_pairs = []
    for _ in range(10):
        action = np.random.randint(0, 10)
        output = device_hw.respond(action)
        io_pairs.append((action, output))

    # Phase 2: Self-learn mapping
    targets = list(range(10, 100, 10))
    target_to_action = {}
    for target in targets:
        best_action, best_error = 0, float('inf')
        for action in range(10):
            responses = [out for a, out in io_pairs if a == action]
            if not responses:
                response = device_hw.respond(action)
                io_pairs.append((action, response))
                responses = [response]
            avg = np.mean(responses)
            error = abs(avg * 10 - target)
            if error < best_error:
                best_error = error
                best_action = action
        target_to_action[target] = best_action

    # Phase 3: JIT compile
    print(f"    [{body_name}] JIT compiling...")
    train_data = [(f"target={t}) =", str(target_to_action[t])) for t in targets]
    driver = torch.randn(hidden_size, device=device_torch) * 0.01
    driver.requires_grad_(True)
    opt = torch.optim.Adam([driver], lr=0.01)
    for epoch in range(100):
        for prompt, target_str in train_data:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device_torch)
            def inject(module, input, output, v=driver):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(device_torch))
            opt.zero_grad(); loss.backward(); opt.step()

    # Phase 4: Test
    print(f"    [{body_name}] Testing...")
    driver_eval = driver.detach()
    correct = 0
    for target in targets:
        prompt = f"target={target}) ="
        def inject_test(module, input, output, v=driver_eval):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_test)
        inp = tok(prompt, return_tensors='pt').to(device_torch)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        try:
            pred_action = int(pred)
        except ValueError:
            pred_action = -1
        if pred_action == target_to_action[target]:
            correct += 1

    accuracy = correct / len(targets)
    print(f"    [{body_name}] Accuracy: {accuracy:.0%}")
    return accuracy


def main():
    print("[P56] Soul Migration")
    print(f"  Device: {DEVICE}")
    start = time.time()

    # Same hardware for fair comparison
    np.random.seed(42)
    device_hw = BlackboxDevice(seed=42)

    results = {}

    # Boot in 0.5B body
    print("\n  Body 1: Qwen2.5-0.5B")
    model_05, tok_05 = load_model(device=DEVICE, surgery=True)
    for p in model_05.parameters():
        p.requires_grad = False
    np.random.seed(123)
    acc_05 = autopoietic_boot(model_05, tok_05, device_hw, 8, DEVICE, '0.5B')
    results['0.5B'] = round(acc_05, 4)
    del model_05; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Boot in 1.5B body (different architecture!)
    print("\n  Body 2: Qwen2.5-1.5B")
    model_15, tok_15 = load_model(model_id='Qwen/Qwen2.5-1.5B', device=DEVICE, surgery=True)
    for p in model_15.parameters():
        p.requires_grad = False
    np.random.seed(123)  # Same random seed for fair comparison
    acc_15 = autopoietic_boot(model_15, tok_15, device_hw, 4, DEVICE, '1.5B')
    results['1.5B'] = round(acc_15, 4)
    del model_15; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    migration_success = acc_15 > 0
    print(f"\n  Soul Migration Results:")
    print(f"    0.5B body: {acc_05:.0%}")
    print(f"    1.5B body: {acc_15:.0%}")
    print(f"    Migration: {'SUCCESS' if migration_success else 'FAILED'}")

    # Save
    output = {
        'phase': 56, 'name': 'soul_migration',
        'results': results,
        'migration_success': migration_success,
        'same_kernel_code': True,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase56_migration.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(['0.5B\n(Original)', '1.5B\n(Migrated)'],
                [acc_05, acc_15],
                color=['tab:blue', 'tab:purple'], edgecolor='black')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Soul Migration: Same Kernel, Different Body',
                      fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([acc_05, acc_15]):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=14)

    axes[1].axis('off')
    summary = ("SOUL MIGRATION PROTOCOL\n"
               "="*35 + "\n\n"
               "Same autopoietic kernel code:\n"
               "  1. Motor Babbling\n"
               "  2. Self-learn device\n"
               "  3. JIT compile driver\n"
               "  4. Execute\n\n"
               f"0.5B (896-dim): {acc_05:.0%}\n"
               f"1.5B (1536-dim): {acc_15:.0%}\n\n"
               f"{'Ghost in the Shell: CONFIRMED' if migration_success else 'Migration failed'}")
    axes[1].text(0.5, 0.5, summary, transform=axes[1].transAxes,
                fontsize=11, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 56: Soul Migration\nSame kernel boots in different neural architectures',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase56_migration.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")

if __name__ == '__main__':
    main()
