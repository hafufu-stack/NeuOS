# -*- coding: utf-8 -*-
"""
Phase 176: Temperature Phase Transition - Fine-Grained Measurement
Precisely map the T=1.5-2.0 phase transition discovered in Phase 144.
Determine if first-order (discontinuous) or second-order (continuous).
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8
N_SAMPLES = 50  # samples per temperature


def train_soul(model, tok, data, device, layer=LAYER, epochs=100, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for p, t in data:
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


def sample_at_temperature(model, tok, soul_vec, prompt, expected, device,
                          layer, temperature, n_samples):
    """Sample n_samples outputs at given temperature, return stats."""
    correct = 0
    probs_correct = []
    entropies = []

    inp = tok(prompt, return_tensors='pt').to(device)

    for _ in range(n_samples):
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        with torch.no_grad():
            out = model(**inp)
        h.remove()

        logits = out.logits[0, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        # Sample
        if temperature < 1e-6:
            sampled_id = logits.argmax().item()
        else:
            sampled_id = torch.multinomial(probs, 1).item()

        pred = tok.decode(sampled_id).strip()
        if pred == expected:
            correct += 1

        # P(correct token)
        target_id = tok.encode(expected)[-1]
        p_correct = probs[target_id].item()
        probs_correct.append(p_correct)

        # Entropy
        log_probs = torch.log2(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        entropies.append(entropy)

    return {
        'accuracy': correct / n_samples,
        'p_correct_mean': float(np.mean(probs_correct)),
        'p_correct_std': float(np.std(probs_correct)),
        'entropy_mean': float(np.mean(entropies)),
        'entropy_std': float(np.std(entropies)),
    }


def main():
    print("[P176] Temperature Phase Transition")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train souls
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),
                ("9, 3) =","3"),("7, 4) =","4"),("6, 1) =","1"),("2, 8) =","2"),
                ("5, 9) =","5"),("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","6"),
                ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","8"),
                ("5, 9) =","9"),("1, 3) =","3")]

    print("  Training MIN and MAX souls...")
    min_soul = train_soul(model, tok, min_data, DEVICE, seed=42)
    max_soul = train_soul(model, tok, max_data, DEVICE, seed=42)

    # Temperature sweep: fine-grained around the transition
    temperatures = list(np.arange(0.0, 1.0, 0.25)) + \
                   list(np.arange(1.0, 1.5, 0.1)) + \
                   list(np.arange(1.5, 2.05, 0.05)) + \
                   list(np.arange(2.1, 3.1, 0.2))

    test_prompts = [("7, 2) =", "2", "7"), ("6, 3) =", "3", "6"),
                    ("2, 9) =", "2", "9"), ("1, 5) =", "1", "5"),
                    ("8, 4) =", "4", "8")]

    print("  Sweeping %d temperatures..." % len(temperatures))
    sweep_data = {'MIN': [], 'MAX': []}

    for T in temperatures:
        for task, soul, idx in [('MIN', min_soul, 1), ('MAX', max_soul, 2)]:
            agg = {'accuracy': [], 'p_correct': [], 'entropy': []}
            for prompt, min_exp, max_exp in test_prompts:
                expected = min_exp if task == 'MIN' else max_exp
                stats = sample_at_temperature(model, tok, soul, prompt, expected,
                                              DEVICE, LAYER, T, N_SAMPLES)
                agg['accuracy'].append(stats['accuracy'])
                agg['p_correct'].append(stats['p_correct_mean'])
                agg['entropy'].append(stats['entropy_mean'])

            point = {
                'temperature': round(T, 3),
                'accuracy': round(float(np.mean(agg['accuracy'])), 4),
                'p_correct': round(float(np.mean(agg['p_correct'])), 4),
                'entropy': round(float(np.mean(agg['entropy'])), 4),
            }
            sweep_data[task].append(point)

        print("    T=%.2f: MIN_acc=%.0f%% MAX_acc=%.0f%%" % (
            T, sweep_data['MIN'][-1]['accuracy']*100,
            sweep_data['MAX'][-1]['accuracy']*100))

    # Compute derivative (dp/dT) to detect transition
    for task in ['MIN', 'MAX']:
        temps = [p['temperature'] for p in sweep_data[task]]
        p_vals = [p['p_correct'] for p in sweep_data[task]]
        if len(temps) > 2:
            dp_dt = np.gradient(p_vals, temps)
            for i, point in enumerate(sweep_data[task]):
                point['dp_dT'] = round(float(dp_dt[i]), 6)

    # Find critical temperature
    for task in ['MIN', 'MAX']:
        dp_vals = [abs(p.get('dp_dT', 0)) for p in sweep_data[task]]
        if dp_vals:
            crit_idx = np.argmax(dp_vals)
            crit_T = sweep_data[task][crit_idx]['temperature']
            print("  %s critical T = %.2f (max |dp/dT| = %.4f)" % (
                task, crit_T, dp_vals[crit_idx]))

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'MIN': '#E91E63', 'MAX': '#2196F3'}

    # Panel 1: Accuracy vs Temperature
    ax = axes[0, 0]
    for task in ['MIN', 'MAX']:
        temps = [p['temperature'] for p in sweep_data[task]]
        accs = [p['accuracy'] for p in sweep_data[task]]
        ax.plot(temps, accs, 'o-', color=colors[task], label=task,
                linewidth=2, markersize=4)
    ax.axvspan(1.5, 2.0, alpha=0.1, color='red', label='Transition zone')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Accuracy (sampled)')
    ax.set_title('Accuracy vs Temperature', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: P(correct token) vs Temperature
    ax = axes[0, 1]
    for task in ['MIN', 'MAX']:
        temps = [p['temperature'] for p in sweep_data[task]]
        probs = [p['p_correct'] for p in sweep_data[task]]
        ax.plot(temps, probs, 'o-', color=colors[task], label=task,
                linewidth=2, markersize=4)
    ax.axvspan(1.5, 2.0, alpha=0.1, color='red')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('P(correct token)')
    ax.set_title('Token Probability vs Temperature', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: dp/dT (derivative) to find critical point
    ax = axes[1, 0]
    for task in ['MIN', 'MAX']:
        temps = [p['temperature'] for p in sweep_data[task]]
        dp = [p.get('dp_dT', 0) for p in sweep_data[task]]
        ax.plot(temps, dp, 'o-', color=colors[task], label=task,
                linewidth=2, markersize=4)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('dP/dT')
    ax.set_title('Rate of Change (Phase Transition Signature)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Panel 4: Output Entropy
    ax = axes[1, 1]
    for task in ['MIN', 'MAX']:
        temps = [p['temperature'] for p in sweep_data[task]]
        ents = [p['entropy'] for p in sweep_data[task]]
        ax.plot(temps, ents, 'o-', color=colors[task], label=task,
                linewidth=2, markersize=4)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Output Entropy (bits)')
    ax.set_title('Output Distribution Entropy', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Phase 176: Temperature Phase Transition\n'
                 '"Where exactly does soul control collapse?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase176_temperature_transition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 176, 'name': 'temperature_transition',
        'sweep_data': sweep_data,
        'n_temperatures': len(temperatures),
        'n_samples_per_point': N_SAMPLES,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase176_temperature_transition.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P176 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
