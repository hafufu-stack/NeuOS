# -*- coding: utf-8 -*-
"""
Phase 152: Soul Immune System
When the model detects incorrect output via entropy monitoring,
can it self-correct by adjusting the soul vector automatically?

Like a biological immune system: detect infection -> mount response -> heal.

"The self-healing OS."
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


def infer(model, tok, prompt, device, soul_vec, layer=LAYER):
    def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    pred = tok.decode(logits.argmax().item()).strip()
    return pred, entropy


def corrupt_soul(soul_vec, noise_level, seed=None):
    """Add noise to a soul vector to simulate damage/corruption."""
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(soul_vec) * noise_level * soul_vec.norm()
    return soul_vec + noise


def self_correct(model, tok, prompt, device, corrupted_soul, target_token,
                 layer=LAYER, lr=0.05, max_steps=20):
    """
    Immune response: given a corrupted soul, perform gradient descent
    to minimize entropy (maximize confidence) without knowing the right answer.
    Uses only entropy as the signal -- no ground truth labels.
    """
    vec = corrupted_soul.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=lr)

    history = []
    for step in range(max_steps):
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        out = model(**inp)
        h.remove()

        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=0)
        # Minimize entropy (maximize confidence in whatever it predicts)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        opt.zero_grad()
        entropy.backward()
        opt.step()

        pred = tok.decode(logits.argmax().item()).strip()
        history.append({
            'step': step, 'entropy': entropy.item(),
            'prediction': pred, 'correct': (pred == target_token)
        })

    final_pred = history[-1]['prediction']
    final_entropy = history[-1]['entropy']
    return vec.detach(), final_pred, final_entropy, history


def main():
    print("[P152] Soul Immune System")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    # Keep gradients enabled for self-correction
    for p in model.parameters():
        p.requires_grad = False

    # Train healthy souls
    print("  Training healthy MIN soul...")
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1"),("8, 5) =","5"),("6, 2) =","2"),
                ("9, 7) =","7"),("4, 1) =","1"),("3, 8) =","3")]
    soul_min = train_soul(model, tok, min_data, DEVICE, epochs=150)

    # Verify healthy soul
    test_cases = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                  ("1, 5) =","1"),("8, 4) =","4"),("4, 7) =","4"),
                  ("9, 6) =","6"),("3, 8) =","3")]

    print("\n  Healthy soul baseline:")
    healthy_acc = 0
    for prompt, expected in test_cases:
        pred, ent = infer(model, tok, prompt, DEVICE, soul_min)
        correct = (pred == expected)
        if correct: healthy_acc += 1
    healthy_acc /= len(test_cases)
    print("  Healthy accuracy: %.0f%%" % (healthy_acc * 100))

    # Corruption levels to test
    noise_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = {}

    for noise in noise_levels:
        print("\n  --- Noise level: %.1f ---" % noise)
        corrupted = corrupt_soul(soul_min, noise, seed=42)

        # Evaluate corrupted soul
        corrupt_correct = 0
        for prompt, expected in test_cases:
            pred, _ = infer(model, tok, prompt, DEVICE, corrupted)
            if pred == expected: corrupt_correct += 1
        corrupt_acc = corrupt_correct / len(test_cases)

        # Self-correct each test case
        healed_correct = 0
        correction_histories = []
        for prompt, expected in test_cases:
            # Start from corrupted soul
            corrupted_copy = corrupt_soul(soul_min, noise, seed=42)
            healed_vec, final_pred, final_ent, history = self_correct(
                model, tok, prompt, DEVICE, corrupted_copy, expected,
                max_steps=15, lr=0.03)
            if final_pred == expected:
                healed_correct += 1
            correction_histories.append({
                'prompt': prompt, 'expected': expected,
                'corrupted_pred': history[0]['prediction'] if history else '?',
                'healed_pred': final_pred,
                'steps': len(history),
                'initial_entropy': history[0]['entropy'] if history else 0,
                'final_entropy': final_ent,
            })

        healed_acc = healed_correct / len(test_cases)
        results[noise] = {
            'corrupted_accuracy': round(corrupt_acc, 4),
            'healed_accuracy': round(healed_acc, 4),
            'recovery': round(healed_acc - corrupt_acc, 4),
            'details': correction_histories,
        }
        print("  Corrupted: %.0f%% -> Healed: %.0f%% (recovery: %+.0f pp)" % (
            corrupt_acc*100, healed_acc*100, (healed_acc-corrupt_acc)*100))

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Accuracy vs noise level
    ax = axes[0]
    noise_vals = sorted(results.keys())
    corrupt_accs = [results[n]['corrupted_accuracy'] for n in noise_vals]
    healed_accs = [results[n]['healed_accuracy'] for n in noise_vals]
    ax.plot(noise_vals, corrupt_accs, 'ro-', linewidth=2, markersize=8, label='Corrupted')
    ax.plot(noise_vals, healed_accs, 'go-', linewidth=2, markersize=8, label='Self-Healed')
    ax.axhline(y=healthy_acc, color='blue', linestyle='--', linewidth=2,
               label='Healthy Baseline (%.0f%%)' % (healthy_acc*100))
    ax.fill_between(noise_vals, corrupt_accs, healed_accs, alpha=0.15, color='green')
    ax.set_xlabel('Noise Level (fraction of soul norm)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Soul Immune Response\n(green area = recovery)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    # Panel 2: Recovery amount by noise level
    ax = axes[1]
    recoveries = [results[n]['recovery'] for n in noise_vals]
    colors = ['#4CAF50' if r > 0 else '#F44336' for r in recoveries]
    bars = ax.bar([str(n) for n in noise_vals], recoveries, color=colors,
                  edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, recoveries):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.02 if val >= 0 else -0.05),
                '%+.0f pp' % (val*100), ha='center', fontweight='bold', fontsize=10)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Recovery (healed - corrupted)')
    ax.set_title('Immune Response Strength\n(positive = successful healing)', fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Panel 3: Entropy during healing (one example)
    ax = axes[2]
    # Pick noise=0.3 if available
    example_noise = 0.3 if 0.3 in results else noise_vals[len(noise_vals)//2]
    example = results[example_noise]['details'][0]
    # Re-run to get full history
    corrupted_example = corrupt_soul(soul_min, example_noise, seed=42)
    _, _, _, hist = self_correct(
        model, tok, test_cases[0][0], DEVICE, corrupted_example,
        test_cases[0][1], max_steps=15, lr=0.03)
    steps = [h['step'] for h in hist]
    ents = [h['entropy'] for h in hist]
    correct_flags = [h['correct'] for h in hist]
    ax.plot(steps, ents, 'b-o', linewidth=2, markersize=6)
    # Color correct steps green
    for s, e, c in zip(steps, ents, correct_flags):
        ax.scatter(s, e, c='green' if c else 'red', s=60, zorder=5, edgecolors='black')
    ax.set_xlabel('Healing Step')
    ax.set_ylabel('Output Entropy')
    ax.set_title('Healing Trajectory (noise=%.1f)\n"%s"' % (
        example_noise, test_cases[0][0][:15]), fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 152: Soul Immune System\n'
                 '"The self-healing OS"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase152_soul_immune.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 152, 'name': 'soul_immune_system',
        'healthy_accuracy': round(healthy_acc, 4),
        'results_by_noise': {str(k): v for k, v in results.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase152_soul_immune.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
