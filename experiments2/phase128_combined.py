# -*- coding: utf-8 -*-
"""
Phase 128: Combined Firmware + Stochastic Resonance
Best of P123 (noise) + P124 (L8 injection) together.

Question: Does L8 injection + optimal noise break the memory wall further?
"""
import torch, json, os, gc, numpy as np, time, sys, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_soul(model, tok, data, device, layer=16, seed=42, epochs=150):
    """Train soul vector at specified layer."""
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


def run_multistep_min(model, tok, sequence, soul_vec, layer, sigma, device, n_reps=5):
    """Run multi-step MIN on a sequence with noise injection."""
    if len(sequence) < 2:
        return False

    current = sequence[0]
    for step_idx in range(1, len(sequence)):
        a = current
        b = sequence[step_idx]
        prompt = "%d, %d) =" % (a, b)
        expected = str(min(a, b))

        # Multiple reps for noisy runs
        reps = 1 if sigma == 0 else n_reps
        votes = []
        for _ in range(reps):
            def inj(m, i, o, v=soul_vec, s=sigma):
                r = replace_last_token(o, v)
                if s > 0 and step_idx > 0:
                    t = r[0] if isinstance(r, tuple) else r
                    noise = torch.randn_like(t) * s * t.std()
                    noisy = t + noise
                    if isinstance(r, tuple):
                        r = (noisy,) + r[1:]
                    else:
                        r = noisy
                return r
            h = model.model.layers[layer].register_forward_hook(inj)
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            votes.append(pred)

        # Majority vote
        from collections import Counter
        majority = Counter(votes).most_common(1)[0][0]

        try:
            current = int(majority)
        except ValueError:
            return False

        if str(current) != expected:
            return False

    return True


def main():
    print("[P128] Combined Firmware + Stochastic Resonance")
    print("  Question: L8 + noise > L8 alone? > L16 + noise?")
    start = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3")]

    # Train souls at L8 and L16
    print("  Training MIN soul at L8...")
    soul_L8 = train_soul(model, tok, min_data, DEVICE, layer=8, seed=42)
    print("  Training MIN soul at L16...")
    soul_L16 = train_soul(model, tok, min_data, DEVICE, layer=16, seed=42)

    # Generate test sequences
    random.seed(42)
    test_seqs = {
        2: [random.sample(range(1, 10), 2) for _ in range(20)],
        3: [random.sample(range(1, 10), 3) for _ in range(20)],
        4: [random.sample(range(1, 10), 4) for _ in range(20)],
        5: [[random.randint(1, 9) for _ in range(5)] for _ in range(20)],
    }

    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5]
    configs = [
        ('L16', 16, soul_L16),
        ('L8', 8, soul_L8),
    ]

    results = {}
    for config_name, layer, soul in configs:
        for sigma in sigmas:
            for n_steps in [2, 3, 4, 5]:
                key = "%s_s%.2f_%dstep" % (config_name, sigma, n_steps)
                correct = 0
                total = len(test_seqs[n_steps])
                for seq in test_seqs[n_steps]:
                    if run_multistep_min(model, tok, seq, soul, layer, sigma, DEVICE, n_reps=3):
                        correct += 1
                acc = correct / total
                results[key] = acc
                print("    %s | sigma=%.2f | %d steps: %.0f%%" % (config_name, sigma, n_steps, acc * 100))

    # Organize for plotting
    step_counts = [2, 3, 4, 5]

    # Build comparison matrices
    l16_matrix = np.zeros((len(step_counts), len(sigmas)))
    l8_matrix = np.zeros((len(step_counts), len(sigmas)))
    for si, s in enumerate(sigmas):
        for ni, n in enumerate(step_counts):
            l16_matrix[ni, si] = results.get("L16_s%.2f_%dstep" % (s, n), 0)
            l8_matrix[ni, si] = results.get("L8_s%.2f_%dstep" % (s, n), 0)

    improvement = l8_matrix - l16_matrix

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: L16 baseline
    ax = axes[0]
    im = ax.imshow(l16_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    for i in range(len(step_counts)):
        for j in range(len(sigmas)):
            ax.text(j, i, "%.0f%%" % (l16_matrix[i, j] * 100),
                    ha='center', va='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(["%.2f" % s for s in sigmas])
    ax.set_yticks(range(len(step_counts)))
    ax.set_yticklabels(["%d steps" % n for n in step_counts])
    ax.set_xlabel('Noise sigma')
    ax.set_title('L16 (NeuOS Default)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: L8 (Aletheia optimal)
    ax = axes[1]
    im = ax.imshow(l8_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    for i in range(len(step_counts)):
        for j in range(len(sigmas)):
            ax.text(j, i, "%.0f%%" % (l8_matrix[i, j] * 100),
                    ha='center', va='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(["%.2f" % s for s in sigmas])
    ax.set_yticks(range(len(step_counts)))
    ax.set_yticklabels(["%d steps" % n for n in step_counts])
    ax.set_xlabel('Noise sigma')
    ax.set_title('L8 (Aletheia Optimal)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: Improvement (L8 - L16)
    ax = axes[2]
    im = ax.imshow(improvement, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    for i in range(len(step_counts)):
        for j in range(len(sigmas)):
            v = improvement[i, j]
            color = 'white' if abs(v) > 0.2 else 'black'
            sign = '+' if v > 0 else ''
            ax.text(j, i, "%s%.0f%%" % (sign, v * 100),
                    ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(["%.2f" % s for s in sigmas])
    ax.set_yticks(range(len(step_counts)))
    ax.set_yticklabels(["%d steps" % n for n in step_counts])
    ax.set_xlabel('Noise sigma')
    ax.set_title('Improvement: L8 - L16', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Phase 128: Combined Firmware + Stochastic Resonance\n'
                 '"The right layer + the right noise = breakthrough?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase128_combined.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    output = {
        'phase': 128,
        'name': 'combined_firmware_stochastic',
        'configs': ['L16', 'L8'],
        'sigmas': sigmas,
        'step_counts': step_counts,
        'results': {k: round(v, 4) for k, v in results.items()},
        'l8_best': max(results.items(), key=lambda x: x[1] if x[0].startswith('L8') else 0),
        'l16_best': max(results.items(), key=lambda x: x[1] if x[0].startswith('L16') else 0),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase128_combined.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  === Best configs ===")
    for cfg in ['L16', 'L8']:
        best_key = max([k for k in results if k.startswith(cfg)], key=lambda k: results[k])
        print("    %s best: %s = %.0f%%" % (cfg, best_key, results[best_key] * 100))
    print("  Completed in %.0fs" % (time.time() - start))

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
