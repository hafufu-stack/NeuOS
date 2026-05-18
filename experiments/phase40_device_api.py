# -*- coding: utf-8 -*-
"""
Phase 40: Universal Device API (Zero-Shot Driver Prediction)
Train a hypernetwork that predicts optimal L16 driver vectors
from a few I/O examples, without gradient descent at inference.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, torch.nn as nn, json, os, gc, numpy as np, time, sys
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


class DriverHypernetwork(nn.Module):
    """Predicts optimal driver vector from I/O examples."""
    def __init__(self, hidden_size, n_io_pairs=4):
        super().__init__()
        # Each I/O pair: (input_val, output_val) -> 2 floats
        self.io_encoder = nn.Sequential(
            nn.Linear(n_io_pairs * 2, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.driver_head = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, hidden_size),
        )

    def forward(self, io_pairs):
        """io_pairs: (batch, n_io_pairs * 2)"""
        encoded = self.io_encoder(io_pairs)
        driver = self.driver_head(encoded)
        return driver


def main():
    print("[P40] Universal Device API (Zero-Shot Driver)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Step 1: Generate ground-truth drivers for different "devices"
    # Each device is a simple function: y = f(x) for x in 1-9
    # Device types: identity, double, half, complement, square
    print("  Step 1: Compiling ground-truth drivers...")
    devices = {
        'identity': lambda x: x,
        'double': lambda x: min(x * 2, 9),
        'complement': lambda x: 9 - x,
        'clamp5': lambda x: min(x, 5),
    }

    gt_drivers = {}
    for dev_name, dev_fn in devices.items():
        train_data = [(f"{x}) =", str(dev_fn(x))) for x in range(1, 8) if dev_fn(x) < 10]
        driver = torch.randn(hidden_size, device=DEVICE) * 0.01
        driver.requires_grad_(True)
        opt = torch.optim.Adam([driver], lr=0.01)

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
                opt.zero_grad()
                loss.backward()
                opt.step()

        gt_drivers[dev_name] = driver.detach().clone()

        # Test accuracy
        correct = 0
        for prompt, target_str in train_data:
            def inject_eval(module, input, output, v=gt_drivers[dev_name]):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_eval)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            if pred == target_str:
                correct += 1
        acc = correct / len(train_data)
        print(f"    {dev_name}: compiled ({acc:.0%})")

    # Step 2: Train hypernetwork
    print("\n  Step 2: Training hypernetwork...")
    n_io = 4
    hyper = DriverHypernetwork(hidden_size, n_io).to(DEVICE)
    hyper_opt = torch.optim.Adam(hyper.parameters(), lr=0.001)

    hyper_losses = []
    for epoch in range(500):
        total_loss = 0
        for dev_name, dev_fn in devices.items():
            # Sample random I/O pairs
            xs = np.random.choice(range(1, 8), n_io, replace=False)
            io_flat = []
            for x in xs:
                io_flat.extend([float(x), float(dev_fn(x))])
            io_tensor = torch.tensor([io_flat]).float().to(DEVICE)

            pred_driver = hyper(io_tensor).squeeze()
            gt = gt_drivers[dev_name]
            loss = torch.nn.functional.mse_loss(pred_driver, gt)
            total_loss += loss.item()
            hyper_opt.zero_grad()
            loss.backward()
            hyper_opt.step()

        hyper_losses.append(total_loss / len(devices))
        if epoch % 100 == 0:
            print(f"    Epoch {epoch}: loss={total_loss/len(devices):.6f}")

    # Step 3: Zero-shot driver generation
    print("\n  Step 3: Zero-shot driver generation...")
    hyper.eval()
    zs_results = {}
    for dev_name, dev_fn in devices.items():
        # Give hypernetwork 4 I/O examples
        xs = [1, 3, 5, 7]
        io_flat = []
        for x in xs:
            io_flat.extend([float(x), float(dev_fn(x))])
        io_tensor = torch.tensor([io_flat]).float().to(DEVICE)

        with torch.no_grad():
            pred_driver = hyper(io_tensor).squeeze()

        # Test with predicted driver
        test_data = [(f"{x}) =", str(dev_fn(x))) for x in range(1, 10) if dev_fn(x) < 10]
        correct = 0
        preds = []
        for prompt, target_str in test_data:
            def inject_zs(module, input, output, v=pred_driver):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_zs)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            preds.append(pred)
            if pred == target_str:
                correct += 1

        acc = correct / len(test_data)
        zs_results[dev_name] = {'accuracy': round(acc, 4), 'predictions': preds}
        print(f"    {dev_name}: {acc:.0%} (preds: {preds})")

    avg_zs = np.mean([r['accuracy'] for r in zs_results.values()])

    # Save
    output = {
        'phase': 40, 'name': 'universal_device_api',
        'devices': list(devices.keys()),
        'zs_results': zs_results,
        'avg_zero_shot_acc': round(float(avg_zs), 4),
        'hyper_losses': [round(l, 6) for l in hyper_losses[::20]],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase40_device_api.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(hyper_losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Hypernetwork Training', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    devs = list(zs_results.keys())
    accs = [zs_results[d]['accuracy'] for d in devs]
    colors = ['tab:green' if a > 0.5 else 'tab:red' for a in accs]
    axes[1].bar(devs, accs, color=colors, edgecolor='black')
    axes[1].set_ylabel('Zero-Shot Accuracy')
    axes[1].set_title(f'Zero-Shot Driver Generation\nAvg: {avg_zs:.0%}', fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    axes[2].axis('off')
    summary = (f"Universal Device API\n\n"
               f"Hypernetwork: {n_io} I/O pairs -> driver vec\n\n")
    for d in devs:
        summary += f"  {d}: {zs_results[d]['accuracy']:.0%}\n"
    summary += f"\nAvg zero-shot: {avg_zs:.0%}\n"
    summary += f"\nNo gradient descent at inference!\n"
    summary += f"Just show examples -> get driver"
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=11, va='center', ha='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 40: Universal Device API\nZero-shot driver generation from I/O examples',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase40_device_api.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model, hyper; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
