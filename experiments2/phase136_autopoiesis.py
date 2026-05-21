# -*- coding: utf-8 -*-
"""
Phase 136: Thermodynamic Autopoiesis
Inject noise ONLY when output entropy exceeds a threshold.
Constant noise degrades multi-step; entropy-gated noise may preserve stability.

"The cell only repairs itself when it senses damage."
"""
import torch, json, os, gc, numpy as np, time, sys, random
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


def train_soul(model, tok, data, device, layer=LAYER, seed=42, epochs=150):
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


def infer_with_entropy(model, tok, vec, prompt, device, layer=LAYER):
    """Run inference, return (predicted_token, entropy)."""
    def inj(m, i, o, v=vec): return replace_last_token(o, v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-12)
    entropy = -(probs * log_probs).sum().item()
    pred = tok.decode(logits.argmax().item()).strip()
    return pred, entropy


def infer_with_noise(model, tok, vec, prompt, device, sigma=0.3, layer=LAYER):
    """Run inference with noise added to hidden states at the soul layer."""
    noise = torch.randn_like(vec) * sigma
    noisy_vec = vec + noise
    def inj(m, i, o, v=noisy_vec): return replace_last_token(o, v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    logits = out.logits[0, -1, :]
    pred = tok.decode(logits.argmax().item()).strip()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-12)
    entropy = -(probs * log_probs).sum().item()
    return pred, entropy


def generate_multistep_min(n_steps, rng):
    """Generate a multi-step MIN sequence.
    Returns list of (a, b) pairs where each step takes min of previous result and new number.
    Also returns expected intermediate results.
    """
    numbers = [rng.randint(1, 9) for _ in range(n_steps + 1)]
    steps = []
    expected = []
    current = numbers[0]
    for i in range(n_steps):
        a = current
        b = numbers[i + 1]
        steps.append((a, b))
        current = min(a, b)
        expected.append(current)
    return steps, expected


def main():
    print("[P136] Thermodynamic Autopoiesis")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    task_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")]

    print("  Training MIN soul at L8...")
    vec = train_soul(model, tok, task_data, DEVICE, layer=LAYER, seed=42)

    # Generate multi-step test sequences
    rng = random.Random(42)
    step_counts = [2, 3, 4, 5]
    n_per_length = 20

    test_seqs = {}
    for n_steps in step_counts:
        seqs = []
        for _ in range(n_per_length):
            steps, expected = generate_multistep_min(n_steps, rng)
            seqs.append((steps, expected))
        test_seqs[n_steps] = seqs

    # Thresholds to test
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, float('inf')]
    sigma = 0.3

    # Run autopoiesis experiment
    print("  Testing entropy-gated noise injection...")
    # results[threshold][n_steps] = accuracy
    auto_results = {}
    entropy_log = {}

    for thresh in thresholds:
        auto_results[thresh] = {}
        entropy_log[thresh] = {}
        thresh_label = "inf" if thresh == float('inf') else "%.1f" % thresh
        for n_steps in step_counts:
            correct = 0
            total = len(test_seqs[n_steps])
            step_entropies = []
            noise_applied_count = 0
            total_steps = 0

            for steps, expected in test_seqs[n_steps]:
                current_correct = True
                for step_idx, ((a, b), exp) in enumerate(zip(steps, expected)):
                    prompt = "%d, %d) =" % (a, b)
                    pred, ent = infer_with_entropy(model, tok, vec, prompt, DEVICE)
                    step_entropies.append(ent)
                    total_steps += 1

                    if ent > thresh:
                        # Re-run with noise
                        pred, ent2 = infer_with_noise(model, tok, vec, prompt, DEVICE,
                                                       sigma=sigma)
                        noise_applied_count += 1

                    if pred.strip() != str(exp):
                        current_correct = False
                        break  # chain broken
                    # Update a for next step if needed (already handled by expected)

                if current_correct:
                    correct += 1

            acc = correct / total
            auto_results[thresh][n_steps] = acc
            entropy_log[thresh][n_steps] = {
                'mean_entropy': float(np.mean(step_entropies)) if step_entropies else 0,
                'noise_applied': noise_applied_count,
                'total_steps': total_steps,
            }

        noise_rate = sum(entropy_log[thresh][ns]['noise_applied']
                        for ns in step_counts)
        total_rate = sum(entropy_log[thresh][ns]['total_steps']
                        for ns in step_counts)
        print("    threshold=%s: accs=%s, noise_rate=%d/%d" % (
            thresh_label,
            ", ".join(["%.0f%%" % (auto_results[thresh][ns]*100) for ns in step_counts]),
            noise_rate, total_rate))

    # Also run: no noise (sigma=0) and constant noise (sigma=0.3)
    print("  Running baselines...")
    baseline_results = {'no_noise': {}, 'constant_noise': {}}
    for n_steps in step_counts:
        # No noise
        correct_nn = 0
        correct_cn = 0
        for steps, expected in test_seqs[n_steps]:
            # No noise
            ok_nn = True
            for (a, b), exp in zip(steps, expected):
                prompt = "%d, %d) =" % (a, b)
                pred, _ = infer_with_entropy(model, tok, vec, prompt, DEVICE)
                if pred.strip() != str(exp):
                    ok_nn = False
                    break
            if ok_nn:
                correct_nn += 1

            # Constant noise
            ok_cn = True
            for (a, b), exp in zip(steps, expected):
                prompt = "%d, %d) =" % (a, b)
                pred, _ = infer_with_noise(model, tok, vec, prompt, DEVICE, sigma=sigma)
                if pred.strip() != str(exp):
                    ok_cn = False
                    break
            if ok_cn:
                correct_cn += 1

        baseline_results['no_noise'][n_steps] = correct_nn / len(test_seqs[n_steps])
        baseline_results['constant_noise'][n_steps] = correct_cn / len(test_seqs[n_steps])

    print("    no_noise: %s" % ", ".join(
        ["%.0f%%" % (baseline_results['no_noise'][ns]*100) for ns in step_counts]))
    print("    constant_noise (sigma=%.1f): %s" % (sigma, ", ".join(
        ["%.0f%%" % (baseline_results['constant_noise'][ns]*100) for ns in step_counts])))

    # Find best autopoiesis threshold
    best_thresh = None
    best_avg = -1
    for thresh in thresholds:
        avg = np.mean([auto_results[thresh][ns] for ns in step_counts])
        if avg > best_avg:
            best_avg = avg
            best_thresh = thresh
    best_label = "inf" if best_thresh == float('inf') else "%.1f" % best_thresh
    print("  Best threshold: %s (avg acc=%.1f%%)" % (best_label, best_avg*100))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_steps = {2: '#2196F3', 3: '#FF5722', 4: '#4CAF50', 5: '#9C27B0'}

    # Panel 1: Line plot of accuracy vs threshold for each step count
    ax = axes[0]
    thresh_labels = ["%.1f" % t if t != float('inf') else "inf" for t in thresholds]
    for ns in step_counts:
        accs = [auto_results[t][ns] * 100 for t in thresholds]
        ax.plot(range(len(thresholds)), accs, 'o-', color=colors_steps[ns],
                label='%d steps' % ns, linewidth=2, markersize=6)
        # Add baseline lines
        ax.axhline(y=baseline_results['no_noise'][ns] * 100,
                   color=colors_steps[ns], linestyle='--', alpha=0.3)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels(thresh_labels)
    ax.set_xlabel('Entropy Threshold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Autopoiesis: Accuracy vs Entropy Threshold', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # Panel 2: Bar chart comparing no-noise vs constant-noise vs best-autopoiesis
    ax = axes[1]
    methods = ['No Noise', 'Constant\n(sigma=0.3)', 'Autopoiesis\n(best=%s)' % best_label]
    x = np.arange(len(step_counts))
    w = 0.25
    nn_vals = [baseline_results['no_noise'][ns] * 100 for ns in step_counts]
    cn_vals = [baseline_results['constant_noise'][ns] * 100 for ns in step_counts]
    ap_vals = [auto_results[best_thresh][ns] * 100 for ns in step_counts]

    bars1 = ax.bar(x - w, nn_vals, w, label='No Noise', color='#2196F3',
                   edgecolor='black')
    bars2 = ax.bar(x, cn_vals, w, label='Constant Noise', color='#FF5722',
                   edgecolor='black')
    bars3 = ax.bar(x + w, ap_vals, w, label='Autopoiesis (best)', color='#4CAF50',
                   edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(["%d steps" % ns for ns in step_counts])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Noise Strategy Comparison', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        "%.0f" % h, ha='center', fontsize=8)

    plt.suptitle('Phase 136: Thermodynamic Autopoiesis\n'
                 '"The cell only repairs itself when it senses damage"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase136_autopoiesis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    # Convert float('inf') keys to strings for JSON
    auto_results_json = {}
    for t in thresholds:
        key = "inf" if t == float('inf') else str(t)
        auto_results_json[key] = {str(ns): round(auto_results[t][ns], 4)
                                   for ns in step_counts}
    entropy_log_json = {}
    for t in thresholds:
        key = "inf" if t == float('inf') else str(t)
        entropy_log_json[key] = {str(ns): entropy_log[t][ns] for ns in step_counts}

    output = {
        'phase': 136, 'name': 'thermodynamic_autopoiesis',
        'layer': LAYER, 'sigma': sigma,
        'thresholds': [t if t != float('inf') else 'inf' for t in thresholds],
        'step_counts': step_counts,
        'n_per_length': n_per_length,
        'autopoiesis_results': auto_results_json,
        'entropy_log': entropy_log_json,
        'baseline_no_noise': {str(ns): round(baseline_results['no_noise'][ns], 4)
                               for ns in step_counts},
        'baseline_constant_noise': {str(ns): round(baseline_results['constant_noise'][ns], 4)
                                     for ns in step_counts},
        'best_threshold': best_label,
        'best_avg_accuracy': round(best_avg, 4),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase136_autopoiesis.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
