# -*- coding: utf-8 -*-
"""
Phase 144: Soul Temperature Sensitivity
How does generation temperature interact with soul injection?
At high temp, does randomness overpower the soul's control?

"Heat reveals the strength of conviction."
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


def evaluate_with_temperature(model, tok, vec, data, device, temp,
                               layer=LAYER, n_samples=20):
    """Evaluate with sampling at given temperature."""
    correct = 0
    total = 0
    entropy_sum = 0

    for p, e in data:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()

        logits = out.logits[0, -1, :]

        # Greedy (temperature = 0)
        if temp == 0:
            pred = tok.decode(logits.argmax().item()).strip()
            if pred == e:
                correct += 1
            total += 1
            # Compute entropy of softmax
            probs = torch.softmax(logits, dim=0)
            ent = -(probs * (probs + 1e-10).log()).sum().item() / np.log(2)
            entropy_sum += ent
        else:
            # Sample n_samples times and count majority
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=0)
            ent = -(probs * (probs + 1e-10).log()).sum().item() / np.log(2)
            entropy_sum += ent

            sample_correct = 0
            for _ in range(n_samples):
                sampled = torch.multinomial(probs, 1).item()
                pred = tok.decode(sampled).strip()
                if pred == e:
                    sample_correct += 1
            # Majority voting accuracy
            if sample_correct > n_samples / 2:
                correct += 1
            total += 1

    return correct / total, entropy_sum / len(data)


def get_prob_of_correct(model, tok, vec, data, device, temp, layer=LAYER):
    """Get average probability of the correct token."""
    prob_sum = 0
    for p, e in data:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        logits = out.logits[0, -1, :]
        if temp > 0:
            probs = torch.softmax(logits / temp, dim=0)
        else:
            probs = torch.softmax(logits, dim=0)
        tid = tok.encode(e)[-1]
        prob_sum += probs[tid].item()
    return prob_sum / len(data)


def main():
    print("[P144] Soul Temperature Sensitivity")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")]
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("1, 5) =","1"),("8, 4) =","4")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                 ("1, 5) =","5"),("8, 4) =","8")]

    print("  Training souls...")
    soul_min = train_soul(model, tok, min_data, DEVICE)
    soul_max = train_soul(model, tok, max_data, DEVICE)

    temperatures = [0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    results = {'MIN': [], 'MAX': [], 'no_soul': []}

    print("  Sweeping temperatures...")
    for temp in temperatures:
        # MIN soul
        acc, ent = evaluate_with_temperature(
            model, tok, soul_min, min_test, DEVICE, temp)
        prob = get_prob_of_correct(model, tok, soul_min, min_test, DEVICE, temp)
        results['MIN'].append({
            'temp': temp, 'acc': round(acc, 4),
            'entropy': round(ent, 4), 'prob_correct': round(prob, 4)
        })

        # MAX soul
        acc, ent = evaluate_with_temperature(
            model, tok, soul_max, max_test, DEVICE, temp)
        prob = get_prob_of_correct(model, tok, soul_max, max_test, DEVICE, temp)
        results['MAX'].append({
            'temp': temp, 'acc': round(acc, 4),
            'entropy': round(ent, 4), 'prob_correct': round(prob, 4)
        })

        # No soul baseline
        zero_vec = torch.zeros(model.config.hidden_size, device=DEVICE)
        acc_ns, ent_ns = evaluate_with_temperature(
            model, tok, zero_vec, min_test, DEVICE, temp)
        results['no_soul'].append({
            'temp': temp, 'acc': round(acc_ns, 4),
            'entropy': round(ent_ns, 4)
        })

        print("    T=%.1f: MIN=%.0f%% (p=%.3f, H=%.1f) MAX=%.0f%% (p=%.3f, H=%.1f)" % (
            temp, results['MIN'][-1]['acc']*100, prob,
            results['MIN'][-1]['entropy'],
            results['MAX'][-1]['acc']*100,
            results['MAX'][-1]['prob_correct'],
            results['MAX'][-1]['entropy']))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Accuracy vs Temperature
    ax = axes[0]
    ax.plot(temperatures, [r['acc'] for r in results['MIN']],
            'o-', color='#2196F3', label='MIN soul', markersize=7, linewidth=2)
    ax.plot(temperatures, [r['acc'] for r in results['MAX']],
            's-', color='#FF5722', label='MAX soul', markersize=7, linewidth=2)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Accuracy (majority vote)')
    ax.set_title('Soul Control vs Temperature', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xscale('symlog', linthresh=0.1)

    # Panel 2: P(correct) vs Temperature
    ax = axes[1]
    ax.plot(temperatures, [r['prob_correct'] for r in results['MIN']],
            'o-', color='#2196F3', label='MIN P(correct)', markersize=7, linewidth=2)
    ax.plot(temperatures, [r['prob_correct'] for r in results['MAX']],
            's-', color='#FF5722', label='MAX P(correct)', markersize=7, linewidth=2)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('P(correct token)')
    ax.set_title('Correct Token Probability', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('symlog', linthresh=0.1)

    # Panel 3: Entropy vs Temperature
    ax = axes[2]
    ax.plot(temperatures, [r['entropy'] for r in results['MIN']],
            'o-', color='#2196F3', label='MIN entropy', markersize=7, linewidth=2)
    ax.plot(temperatures, [r['entropy'] for r in results['MAX']],
            's-', color='#FF5722', label='MAX entropy', markersize=7, linewidth=2)
    ax.plot(temperatures, [r['entropy'] for r in results['no_soul']],
            '^--', color='gray', label='No soul', markersize=7, linewidth=2)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Output Entropy (bits)')
    ax.set_title('Output Entropy vs Temperature', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('symlog', linthresh=0.1)

    plt.suptitle('Phase 144: Soul Temperature Sensitivity\n'
                 '"Heat reveals the strength of conviction"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase144_temperature.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 144, 'name': 'temperature_sensitivity',
        'layer': LAYER,
        'temperatures': temperatures,
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase144_temperature.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
