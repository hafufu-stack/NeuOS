# -*- coding: utf-8 -*-
"""
Phase 47: Hardware Fatigue & Neural Homeostasis
When muscle hardware degrades mid-execution, can NeuOS
recompile its driver on-the-fly to maintain control?

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
    f_l = np.exp(-((1.0 - 1.0) ** 2) / 0.2)
    return f_max * activation * f_l * fatigue


def main():
    print("[P47] Hardware Fatigue & Neural Homeostasis")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Phase 1: Compile initial driver (healthy muscle, fatigue=1.0)
    print("  Phase 1: Compiling initial driver (healthy)...")
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

    # Test initial driver
    initial_driver = driver.detach().clone()
    print("    Initial driver compiled")

    # Phase 2: Simulate fatigue scenario
    print("\n  Phase 2: Fatigue simulation with runtime recompilation...")
    fatigue_schedule = [1.0]*5 + [0.7]*5 + [0.5]*5 + [0.3]*5 + [1.0]*5  # Recovery
    target_forces = [50, 30, 70, 20, 80] * 5

    # Strategy A: No adaptation (use initial driver)
    no_adapt_errors = []
    # Strategy B: Runtime recompilation every time error > threshold
    adapt_errors = []
    adapt_driver = initial_driver.clone()
    adapt_driver.requires_grad_(True)
    adapt_opt = torch.optim.Adam([adapt_driver], lr=0.005)
    recompile_count = 0

    for step, (fatigue, target_force) in enumerate(zip(fatigue_schedule, target_forces)):
        prompt = f"force={target_force}) ="
        target_digit = target_force // 10

        # No adaptation
        def inject_no(module, input, output, v=initial_driver):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_no)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        try:
            pred_act = float(pred) / 10.0
        except ValueError:
            pred_act = 0.0
        actual_force = hill_muscle(pred_act, fatigue=fatigue)
        no_adapt_errors.append(abs(target_force - actual_force))

        # With adaptation
        def inject_ad(module, input, output, v=adapt_driver):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_ad)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred_ad = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        try:
            pred_act_ad = float(pred_ad) / 10.0
        except ValueError:
            pred_act_ad = 0.0
        actual_force_ad = hill_muscle(pred_act_ad, fatigue=fatigue)
        error_ad = abs(target_force - actual_force_ad)
        adapt_errors.append(error_ad)

        # Runtime recompilation if error > 10
        if error_ad > 10:
            recompile_count += 1
            # Adjust target for fatigued muscle
            adjusted_target = min(int(target_force / fatigue), 90) if fatigue > 0.1 else target_force
            adjusted_digit = str(adjusted_target // 10)
            target_id = tok.encode(adjusted_digit)[-1]
            for _ in range(10):  # Quick 10-step recompilation
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                def inject_train(module, input, output, v=adapt_driver):
                    return replace_last_token(output, v)
                h = model.model.layers[target_layer].register_forward_hook(inject_train)
                out = model(**inp)
                h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([target_id]).to(DEVICE))
                adapt_opt.zero_grad(); loss.backward(); adapt_opt.step()

        if step % 5 == 0:
            print(f"    Step {step}: fatigue={fatigue:.1f}, "
                  f"no_adapt_err={no_adapt_errors[-1]:.0f}, "
                  f"adapt_err={adapt_errors[-1]:.0f}")

    avg_no_adapt = np.mean(no_adapt_errors)
    avg_adapt = np.mean(adapt_errors)
    improvement = (avg_no_adapt - avg_adapt) / avg_no_adapt * 100 if avg_no_adapt > 0 else 0

    print(f"\n  Results:")
    print(f"    No adaptation: avg error = {avg_no_adapt:.1f}")
    print(f"    With homeostasis: avg error = {avg_adapt:.1f}")
    print(f"    Improvement: {improvement:.0f}%")
    print(f"    Recompilations: {recompile_count}")

    # Save
    output = {
        'phase': 47, 'name': 'hardware_fatigue_homeostasis',
        'avg_error_no_adapt': round(float(avg_no_adapt), 2),
        'avg_error_adapted': round(float(avg_adapt), 2),
        'improvement_pct': round(float(improvement), 1),
        'recompile_count': recompile_count,
        'no_adapt_errors': [round(e, 1) for e in no_adapt_errors],
        'adapt_errors': [round(e, 1) for e in adapt_errors],
        'fatigue_schedule': fatigue_schedule,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase47_homeostasis.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps = range(len(fatigue_schedule))
    axes[0].plot(steps, fatigue_schedule, 'r-', linewidth=2, label='Fatigue')
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('Muscle Health')
    axes[0].set_title('Hardware Degradation', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, no_adapt_errors, 'r-', linewidth=2, alpha=0.7, label='No adapt')
    axes[1].plot(steps, adapt_errors, 'g-', linewidth=2, label='Homeostasis')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Control Error')
    axes[1].set_title('Error Over Time', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].bar(['No\nAdaptation', 'With\nHomeostasis'],
                [avg_no_adapt, avg_adapt],
                color=['tab:red', 'tab:green'], edgecolor='black')
    axes[2].set_ylabel('Avg Error')
    axes[2].set_title(f'Improvement: {improvement:.0f}%\n({recompile_count} recompilations)',
                      fontweight='bold')
    for i, v in enumerate([avg_no_adapt, avg_adapt]):
        axes[2].text(i, v+1, f'{v:.1f}', ha='center', fontweight='bold')

    plt.suptitle('Phase 47: Hardware Fatigue & Neural Homeostasis\nRuntime recompilation when muscle degrades',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase47_homeostasis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
