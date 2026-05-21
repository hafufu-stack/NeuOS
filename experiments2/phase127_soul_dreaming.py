# -*- coding: utf-8 -*-
"""
Phase 127: Soul Dreaming
Interpolate soul vectors during inference and observe the model's 'dream' outputs.

"What does the model dream when its soul is between MIN and MAX?"
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


def train_soul(model, tok, data, device, seed=42, bdim=8, epochs=150):
    """Train a standard soul vector (sender + encoder + decoder)."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    sender = torch.randn(hs, device=device) * 0.01; sender.requires_grad_(True)
    encoder = torch.randn(hs, bdim, device=device) * 0.01; encoder.requires_grad_(True)
    decoder = torch.randn(bdim, hs, device=device) * 0.01; decoder.requires_grad_(True)
    opt = torch.optim.Adam([sender, encoder, decoder], lr=0.01)
    for ep in range(epochs):
        for prompt, target in data:
            tid = tok.encode(target)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            so = [None]
            def sh(m, i, o, v=sender):
                r = replace_last_token(o, v)
                t = r[0] if isinstance(r, tuple) else r
                so[0] = (t[0, -1, :] if t.dim() == 3 else t[-1, :]).clone()
                return r
            def rh(m, i, o, enc=encoder, dec=decoder):
                if so[0] is not None:
                    return replace_last_token(o, so[0] @ enc @ dec)
                return o
            h1 = model.model.layers[4].register_forward_hook(sh)
            h2 = model.model.layers[16].register_forward_hook(rh)
            out = model(**inp); h1.remove(); h2.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return sender.detach(), encoder.detach(), decoder.detach()


def interpolate_souls(min_soul, max_soul, t):
    """Linearly interpolate between MIN and MAX soul vectors.
    Returns interpolated (sender, encoder, decoder)."""
    s_min, e_min, d_min = min_soul
    s_max, e_max, d_max = max_soul
    s_interp = (1 - t) * s_min + t * s_max
    e_interp = (1 - t) * e_min + t * e_max
    d_interp = (1 - t) * d_min + t * d_max
    return s_interp, e_interp, d_interp


def run_with_soul(model, tok, sender, encoder, decoder, prompt, device):
    """Run inference with a given soul. Returns logits for last token."""
    inp = tok(prompt, return_tensors='pt').to(device)
    so = [None]
    def sh(m, i, o, v=sender):
        r = replace_last_token(o, v)
        t = r[0] if isinstance(r, tuple) else r
        so[0] = (t[0, -1, :] if t.dim() == 3 else t[-1, :]).clone()
        return r
    def rh(m, i, o, enc=encoder, dec=decoder):
        if so[0] is not None:
            return replace_last_token(o, so[0] @ enc @ dec)
        return o
    h1 = model.model.layers[4].register_forward_hook(sh)
    h2 = model.model.layers[16].register_forward_hook(rh)
    with torch.no_grad():
        out = model(**inp)
    h1.remove(); h2.remove()
    return out.logits[0, -1, :]


def get_digit_probs(logits, tok):
    """Extract probabilities for digit tokens 0-9 from logits."""
    probs = torch.softmax(logits, dim=-1)
    digit_probs = []
    for d in range(10):
        tid = tok.encode(str(d))[-1]
        digit_probs.append(probs[tid].item())
    return digit_probs


def compute_entropy(probs):
    """Compute Shannon entropy of a probability distribution."""
    p = np.array(probs)
    p = p + 1e-10  # avoid log(0)
    p = p / p.sum()  # renormalize
    return float(-np.sum(p * np.log2(p)))


def main():
    print("[P127] Soul Dreaming")
    start = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Data
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3"), ("7, 2) =", "2"),
                ("6, 3) =", "3"), ("2, 9) =", "2")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9"), ("7, 2) =", "7"),
                ("6, 3) =", "6"), ("2, 9) =", "9")]

    # Test prompts: standard + special (equal/zero) prompts
    test_prompts = [
        ("3, 7) =", "3", "7"),
        ("5, 2) =", "2", "5"),
        ("8, 1) =", "1", "8"),
        ("4, 6) =", "4", "6"),
        ("9, 3) =", "3", "9"),
        ("7, 2) =", "2", "7"),
        ("6, 3) =", "3", "6"),
        ("2, 9) =", "2", "9"),
    ]
    special_prompts = [
        ("0, 0) =", "0", "0"),
        ("5, 5) =", "5", "5"),
    ]

    # ---- Train MIN and MAX souls ----
    print("  Training MIN soul (seed=42, 150 epochs)...")
    s_min, e_min, d_min = train_soul(model, tok, min_data, DEVICE, seed=42, epochs=150)
    print("  Training MAX soul (seed=42, 150 epochs)...")
    s_max, e_max, d_max = train_soul(model, tok, max_data, DEVICE, seed=42, epochs=150)

    min_soul = (s_min, e_min, d_min)
    max_soul = (s_max, e_max, d_max)

    # ---- Interpolation sweep ----
    t_values = [round(x * 0.1, 1) for x in range(11)]  # 0.0 to 1.0
    print("  Running interpolation sweep (t = 0.0 .. 1.0)...")

    all_results = {}
    for prompt, min_ans, max_ans in test_prompts + special_prompts:
        prompt_results = {
            't_values': t_values,
            'min_answer': min_ans,
            'max_answer': max_ans,
            'output_tokens': [],
            'digit_probs': [],  # list of [10] prob arrays
            'entropies': [],
        }
        for t in t_values:
            s_i, e_i, d_i = interpolate_souls(min_soul, max_soul, t)
            logits = run_with_soul(model, tok, s_i, e_i, d_i, prompt, DEVICE)
            out_token = tok.decode(logits.argmax().item()).strip()
            dprobs = get_digit_probs(logits, tok)
            ent = compute_entropy(dprobs)
            prompt_results['output_tokens'].append(out_token)
            prompt_results['digit_probs'].append(dprobs)
            prompt_results['entropies'].append(round(ent, 4))
        all_results[prompt] = prompt_results

    # ---- Analysis ----
    print("  Analyzing transition behavior...")

    # Compute MIN/MAX accuracy at each t
    min_acc_by_t = []
    max_acc_by_t = []
    avg_entropy_by_t = []
    for ti, t in enumerate(t_values):
        min_correct = 0
        max_correct = 0
        entropies = []
        for prompt, min_ans, max_ans in test_prompts:
            out = all_results[prompt]['output_tokens'][ti]
            if out == min_ans:
                min_correct += 1
            if out == max_ans:
                max_correct += 1
            entropies.append(all_results[prompt]['entropies'][ti])
        min_acc_by_t.append(min_correct / len(test_prompts))
        max_acc_by_t.append(max_correct / len(test_prompts))
        avg_entropy_by_t.append(float(np.mean(entropies)))

    # Find transition point (where MIN acc drops below 50%)
    transition_t = None
    for ti, t in enumerate(t_values):
        if min_acc_by_t[ti] < 0.5:
            transition_t = t
            break
    if transition_t is None:
        transition_t = 1.0

    # Dream state analysis (t=0.5)
    dream_idx = t_values.index(0.5)
    dream_outputs = {}
    for prompt, min_ans, max_ans in test_prompts:
        r = all_results[prompt]
        dream_outputs[prompt] = {
            'output': r['output_tokens'][dream_idx],
            'min_answer': min_ans,
            'max_answer': max_ans,
            'entropy': r['entropies'][dream_idx],
            'top_probs': {str(d): round(r['digit_probs'][dream_idx][d], 4) for d in range(10)},
        }

    # Transition sharpness: max entropy gradient
    entropy_gradient = np.gradient(avg_entropy_by_t)
    sharpness = float(np.max(np.abs(entropy_gradient)))

    # Special prompt analysis
    special_analysis = {}
    for prompt, min_ans, max_ans in special_prompts:
        r = all_results[prompt]
        special_analysis[prompt] = {
            'outputs': r['output_tokens'],
            'entropies': r['entropies'],
            'is_symmetric': min_ans == max_ans,
        }

    print("  Results summary:")
    print("    Transition point (MIN acc < 50%%): t=%.1f" % transition_t)
    print("    MIN acc at t=0.0: %.0f%%" % (min_acc_by_t[0] * 100))
    print("    MAX acc at t=1.0: %.0f%%" % (max_acc_by_t[-1] * 100))
    print("    Dream state (t=0.5) entropy: %.3f" % avg_entropy_by_t[dream_idx])
    print("    Transition sharpness: %.4f" % sharpness)

    # ---- Save results ----
    output = {
        'phase': 127, 'name': 'soul_dreaming',
        't_values': t_values,
        'min_accuracy_by_t': [round(a, 4) for a in min_acc_by_t],
        'max_accuracy_by_t': [round(a, 4) for a in max_acc_by_t],
        'avg_entropy_by_t': [round(e, 4) for e in avg_entropy_by_t],
        'transition_point': transition_t,
        'transition_sharpness': round(sharpness, 4),
        'dream_state_t05': dream_outputs,
        'special_prompts': special_analysis,
        'per_prompt_results': {
            k: {
                'output_tokens': v['output_tokens'],
                'entropies': v['entropies'],
                'digit_probs': [[round(p, 4) for p in row] for row in v['digit_probs']],
            }
            for k, v in all_results.items()
        },
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase127_soul_dreaming.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Heatmap of digit probabilities for example prompt "3, 7) ="
    example_prompt = "3, 7) ="
    probs_matrix = np.array(all_results[example_prompt]['digit_probs']).T  # (10, 11)
    im = axes[0].imshow(probs_matrix, aspect='auto', cmap='hot', interpolation='nearest',
                         origin='lower', vmin=0)
    axes[0].set_xticks(range(len(t_values)))
    axes[0].set_xticklabels([f'{t:.1f}' for t in t_values], fontsize=8)
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels([str(d) for d in range(10)])
    axes[0].set_xlabel('Interpolation t')
    axes[0].set_ylabel('Digit')
    axes[0].set_title('Output Probabilities\n(prompt: "3, 7) =")', fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='Probability')
    # Mark MIN answer (3) and MAX answer (7)
    axes[0].axhline(y=3, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)
    axes[0].axhline(y=7, color='lime', linewidth=0.8, linestyle='--', alpha=0.7)
    axes[0].text(0.1, 3.3, 'MIN=3', color='cyan', fontsize=7, fontweight='bold')
    axes[0].text(0.1, 7.3, 'MAX=7', color='lime', fontsize=7, fontweight='bold')

    # Center: Accuracy curves
    axes[1].plot(t_values, min_acc_by_t, 'b-o', lw=2, label='MIN accuracy', markersize=6)
    axes[1].plot(t_values, max_acc_by_t, 'r-s', lw=2, label='MAX accuracy', markersize=6)
    axes[1].axvline(x=transition_t, color='gray', linestyle='--', alpha=0.7,
                     label=f'Transition t={transition_t}')
    axes[1].axvline(x=0.5, color='purple', linestyle=':', alpha=0.5, label='Dream state')
    axes[1].fill_between(t_values, 0, 1, where=[t == 0.5 for t in t_values],
                          color='purple', alpha=0.1)
    axes[1].set_xlabel('Interpolation t')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('MIN vs MAX Behavior\nacross Interpolation', fontweight='bold')
    axes[1].legend(fontsize=8, loc='center left')
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].grid(True, alpha=0.3)

    # Right: Entropy vs t
    axes[2].plot(t_values, avg_entropy_by_t, 'g-D', lw=2, markersize=6, label='Avg entropy')
    # Also plot individual prompts faintly
    for prompt, min_ans, max_ans in test_prompts:
        ents = all_results[prompt]['entropies']
        axes[2].plot(t_values, ents, color='gray', alpha=0.15, lw=1)
    axes[2].axvline(x=transition_t, color='gray', linestyle='--', alpha=0.7)
    axes[2].axvline(x=0.5, color='purple', linestyle=':', alpha=0.5)
    # Mark peak entropy
    peak_idx = int(np.argmax(avg_entropy_by_t))
    peak_t = t_values[peak_idx]
    peak_ent = avg_entropy_by_t[peak_idx]
    axes[2].annotate(f'Peak: {peak_ent:.2f}\nat t={peak_t}',
                      xy=(peak_t, peak_ent), xytext=(peak_t + 0.15, peak_ent + 0.3),
                      arrowprops=dict(arrowstyle='->', color='darkgreen'),
                      fontsize=9, color='darkgreen', fontweight='bold')
    axes[2].set_xlabel('Interpolation t')
    axes[2].set_ylabel('Entropy (bits)')
    axes[2].set_title('Transition Sharpness\n(output distribution entropy)', fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 127: Soul Dreaming\n'
                 '"What does the model dream between MIN and MAX?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase127_soul_dreaming.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
