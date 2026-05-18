# -*- coding: utf-8 -*-
"""
Phase 53: Liquid Homeostasis
Fix P47's over-adaptation (-82%) by using smooth, continuous
adaptation with exponential moving average (EMA) instead of
aggressive recompilation. LNN-inspired time constant.

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


def hill_muscle(activation, f_max=100.0, fatigue=1.0):
    return f_max * activation * fatigue


def main():
    print("[P53] Liquid Homeostasis")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile initial driver
    print("  Compiling initial driver...")
    train_data = [(f"force={f}) =", str(f // 10)) for f in range(10, 100, 10)]
    driver = torch.randn(hidden_size, device=DEVICE) * 0.01
    driver.requires_grad_(True)
    opt = torch.optim.Adam([driver], lr=0.01)
    for epoch in range(100):
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
            opt.zero_grad(); loss.backward(); opt.step()
    base_driver = driver.detach().clone()

    # Fatigue simulation
    fatigue_schedule = [1.0]*5 + [0.7]*5 + [0.5]*5 + [0.3]*5 + [1.0]*5
    target_forces = [50, 30, 70, 20, 80] * 5

    strategies = {}

    # Strategy A: No adaptation (P47 baseline)
    errors_none = []
    for fatigue, target in zip(fatigue_schedule, target_forces):
        prompt = f"force={target}) ="
        def inject_n(module, input, output, v=base_driver):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_n)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        try: act = float(pred) / 10.0
        except ValueError: act = 0.0
        actual = hill_muscle(act, fatigue=fatigue)
        errors_none.append(abs(target - actual))
    strategies['no_adapt'] = errors_none

    # Strategy B: Liquid adaptation (EMA-based smooth correction)
    print("\n  Testing liquid homeostasis (EMA)...")
    liquid_driver = base_driver.clone()
    liquid_driver.requires_grad_(True)
    liquid_opt = torch.optim.SGD([liquid_driver], lr=0.001)  # Very small LR
    ema_tau = 0.95  # Time constant (high = slow adaptation)
    errors_liquid = []

    for step, (fatigue, target) in enumerate(zip(fatigue_schedule, target_forces)):
        prompt = f"force={target}) ="
        # Forward pass
        def inject_l(module, input, output, v=liquid_driver):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_l)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        try: act = float(pred) / 10.0
        except ValueError: act = 0.0
        actual = hill_muscle(act, fatigue=fatigue)
        error = abs(target - actual)
        errors_liquid.append(error)

        # Smooth correction: only if error > 5, single gradient step
        if error > 5:
            adjusted = min(int(target / fatigue), 90) if fatigue > 0.1 else target
            adjusted_digit = str(adjusted // 10)
            target_id = tok.encode(adjusted_digit)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject_train(module, input, output, v=liquid_driver):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_train)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(DEVICE))
            liquid_opt.zero_grad(); loss.backward(); liquid_opt.step()
            # EMA smoothing: blend with base driver
            with torch.no_grad():
                liquid_driver.data = ema_tau * liquid_driver.data + (1-ema_tau) * base_driver
    strategies['liquid'] = errors_liquid

    # Strategy C: P47's aggressive recompilation (for comparison)
    aggressive_driver = base_driver.clone()
    aggressive_driver.requires_grad_(True)
    agg_opt = torch.optim.Adam([aggressive_driver], lr=0.005)
    errors_agg = []
    for step, (fatigue, target) in enumerate(zip(fatigue_schedule, target_forces)):
        prompt = f"force={target}) ="
        def inject_a(module, input, output, v=aggressive_driver):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_a)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        try: act = float(pred) / 10.0
        except ValueError: act = 0.0
        actual = hill_muscle(act, fatigue=fatigue)
        error = abs(target - actual)
        errors_agg.append(error)
        if error > 10:
            adjusted = min(int(target / fatigue), 90) if fatigue > 0.1 else target
            target_id = tok.encode(str(adjusted // 10))[-1]
            for _ in range(10):
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                def inject_t(module, input, output, v=aggressive_driver):
                    return replace_last_token(output, v)
                h = model.model.layers[target_layer].register_forward_hook(inject_t)
                out = model(**inp)
                h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([target_id]).to(DEVICE))
                agg_opt.zero_grad(); loss.backward(); agg_opt.step()
    strategies['aggressive'] = errors_agg

    avgs = {k: round(float(np.mean(v)), 2) for k, v in strategies.items()}
    print(f"\n  Results:")
    for k, v in avgs.items():
        print(f"    {k}: avg error = {v}")

    improvement_vs_none = (avgs['no_adapt'] - avgs['liquid']) / avgs['no_adapt'] * 100 if avgs['no_adapt'] > 0 else 0
    improvement_vs_agg = (avgs['aggressive'] - avgs['liquid']) / avgs['aggressive'] * 100 if avgs['aggressive'] > 0 else 0

    # Save
    output = {
        'phase': 53, 'name': 'liquid_homeostasis',
        'avg_errors': avgs,
        'improvement_vs_none': round(float(improvement_vs_none), 1),
        'improvement_vs_aggressive': round(float(improvement_vs_agg), 1),
        'ema_tau': ema_tau,
        'strategies': {k: [round(e, 1) for e in v] for k, v in strategies.items()},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase53_liquid.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps = range(len(fatigue_schedule))
    axes[0].fill_between(steps, 0, [1-f for f in fatigue_schedule], alpha=0.3, color='red', label='Fatigue')
    axes[0].plot(steps, fatigue_schedule, 'r-', linewidth=2)
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Muscle Health')
    axes[0].set_title('Hardware Degradation', fontweight='bold')
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(steps, errors_none, 'r-', linewidth=1, alpha=0.5, label=f'None ({avgs["no_adapt"]:.0f})')
    axes[1].plot(steps, errors_agg, 'orange', linewidth=1, alpha=0.5, label=f'P47 Aggressive ({avgs["aggressive"]:.0f})')
    axes[1].plot(steps, errors_liquid, 'g-', linewidth=2, label=f'Liquid ({avgs["liquid"]:.0f})')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Error')
    axes[1].set_title('Control Error', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].bar(['No\nAdapt', 'P47\nAggressive', 'Liquid\n(EMA)'],
                [avgs['no_adapt'], avgs['aggressive'], avgs['liquid']],
                color=['tab:red', 'tab:orange', 'tab:green'], edgecolor='black')
    axes[2].set_ylabel('Avg Error')
    axes[2].set_title('Homeostasis Comparison', fontweight='bold')
    for i, v in enumerate([avgs['no_adapt'], avgs['aggressive'], avgs['liquid']]):
        axes[2].text(i, v+1, f'{v:.1f}', ha='center', fontweight='bold')

    plt.suptitle('Phase 53: Liquid Homeostasis\nSmooth EMA adaptation vs aggressive recompilation',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase53_liquid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
