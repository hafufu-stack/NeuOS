# -*- coding: utf-8 -*-
"""
Phase 178: Multi-Token Soul
Inject soul vectors at multiple token positions, not just last token.
Test if position-specific injection enables richer control.
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


def replace_token_at_pos(output, donor_vec, pos):
    """Replace hidden state at specific position in layer output."""
    if isinstance(output, tuple):
        h = output[0].clone()
        if h.dim() == 3:
            h[0, pos, :] = donor_vec.to(h.dtype)
        elif h.dim() == 2:
            h[pos, :] = donor_vec.to(h.dtype)
        return (h,) + output[1:]
    else:
        h = output.clone()
        if h.dim() == 3:
            h[0, pos, :] = donor_vec.to(h.dtype)
        elif h.dim() == 2:
            h[pos, :] = donor_vec.to(h.dtype)
        return h


def replace_multi_tokens(output, vecs_and_positions):
    """Replace hidden states at multiple positions."""
    if isinstance(output, tuple):
        h = output[0].clone()
        for vec, pos in vecs_and_positions:
            if h.dim() == 3:
                h[0, pos, :] = vec.to(h.dtype)
            elif h.dim() == 2:
                h[pos, :] = vec.to(h.dtype)
        return (h,) + output[1:]
    else:
        h = output.clone()
        for vec, pos in vecs_and_positions:
            if h.dim() == 3:
                h[0, pos, :] = vec.to(h.dtype)
            elif h.dim() == 2:
                h[pos, :] = vec.to(h.dtype)
        return h


def train_soul_at_pos(model, tok, data, device, layer, pos, epochs=100, seed=42):
    """Train soul vector injected at a specific token position."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for p, t in data:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(device)
            seq_len = inp['input_ids'].shape[1]
            actual_pos = pos if pos >= 0 else seq_len + pos
            if actual_pos < 0 or actual_pos >= seq_len:
                actual_pos = seq_len - 1
            def inj(m, i, o, v=vec, p=actual_pos):
                return replace_token_at_pos(o, v, p)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def evaluate_with_pos(model, tok, soul_vec, test_data, device, layer, pos):
    """Evaluate with soul injected at specific position."""
    correct = 0
    for prompt, expected in test_data:
        inp = tok(prompt, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]
        actual_pos = pos if pos >= 0 else seq_len + pos
        if actual_pos < 0 or actual_pos >= seq_len:
            actual_pos = seq_len - 1
        def inj(m, i, o, v=soul_vec, p=actual_pos):
            return replace_token_at_pos(o, v, p)
        h = model.model.layers[layer].register_forward_hook(inj)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data) if test_data else 0


def evaluate_with_multi(model, tok, vecs_and_positions, test_data, device, layer):
    """Evaluate with multiple soul vectors at different positions."""
    correct = 0
    for prompt, expected in test_data:
        inp = tok(prompt, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]
        resolved = []
        for vec, pos in vecs_and_positions:
            actual_pos = pos if pos >= 0 else seq_len + pos
            if actual_pos < 0 or actual_pos >= seq_len:
                actual_pos = seq_len - 1
            resolved.append((vec, actual_pos))
        def inj(m, i, o, vps=resolved):
            return replace_multi_tokens(o, vps)
        h = model.model.layers[layer].register_forward_hook(inj)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data) if test_data else 0


def main():
    print("[P178] Multi-Token Soul")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),
                ("9, 3) =","3"),("7, 4) =","4"),("6, 1) =","1"),("2, 8) =","2"),
                ("5, 9) =","5"),("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","6"),
                ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","8"),
                ("5, 9) =","9"),("1, 3) =","3")]
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("1, 5) =","1"),("8, 4) =","4")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("1, 5) =","5"),("8, 4) =","8")]

    results = {}

    # === Experiment 1: Position sweep for single-token injection ===
    print("  === Single-Token Position Sweep ===")
    # Prompt "3, 7) =" tokenizes to several tokens; test injection at each
    sample_inp = tok("3, 7) =", return_tensors='pt')
    seq_len = sample_inp['input_ids'].shape[1]
    print("  Sample prompt has %d tokens" % seq_len)

    pos_results = {}
    for pos in range(seq_len):
        min_soul = train_soul_at_pos(model, tok, min_data, DEVICE, LAYER, pos, seed=42)
        min_acc = evaluate_with_pos(model, tok, min_soul, min_test, DEVICE, LAYER, pos)
        max_soul = train_soul_at_pos(model, tok, max_data, DEVICE, LAYER, pos, seed=42)
        max_acc = evaluate_with_pos(model, tok, max_soul, max_test, DEVICE, LAYER, pos)
        pos_results[pos] = {'MIN': round(min_acc, 4), 'MAX': round(max_acc, 4)}
        print("    pos=%d: MIN=%.0f%%, MAX=%.0f%%" % (pos, min_acc*100, max_acc*100))
    results['position_sweep'] = pos_results

    # === Experiment 2: Multi-position injection ===
    print("\n  === Multi-Position Injection ===")
    # Train separate souls for positions 0 and -1
    min_soul_p0 = train_soul_at_pos(model, tok, min_data, DEVICE, LAYER, 0, seed=42)
    min_soul_last = train_soul_at_pos(model, tok, min_data, DEVICE, LAYER, -1, seed=42)
    max_soul_p0 = train_soul_at_pos(model, tok, max_data, DEVICE, LAYER, 0, seed=42)
    max_soul_last = train_soul_at_pos(model, tok, max_data, DEVICE, LAYER, -1, seed=42)

    multi_configs = {
        'MIN@last_only': ([(min_soul_last, -1)], min_test),
        'MIN@first_only': ([(min_soul_p0, 0)], min_test),
        'MIN@both': ([(min_soul_p0, 0), (min_soul_last, -1)], min_test),
        'MAX@last_only': ([(max_soul_last, -1)], max_test),
        'MAX@first_only': ([(max_soul_p0, 0)], max_test),
        'MAX@both': ([(max_soul_p0, 0), (max_soul_last, -1)], max_test),
        # Cross: MIN at first, MAX at last -> what happens?
        'MIN@first+MAX@last': ([(min_soul_p0, 0), (max_soul_last, -1)], min_test),
        'MAX@first+MIN@last': ([(max_soul_p0, 0), (min_soul_last, -1)], max_test),
    }

    multi_results = {}
    for config_name, (vps, test_data) in multi_configs.items():
        acc = evaluate_with_multi(model, tok, vps, test_data, DEVICE, LAYER)
        multi_results[config_name] = round(acc, 4)
        print("    %s: %.0f%%" % (config_name, acc*100))
    results['multi_position'] = multi_results

    # === Experiment 3: Cross-task position mixing ===
    print("\n  === Cross-Task Position Mixing ===")
    # Does injecting MIN at pos 0 and MAX at pos -1 create SORT-like behavior?
    sort_test = [("7, 2) =", "?"), ("6, 3) =", "?"), ("2, 9) =", "?"),
                 ("1, 5) =", "?"), ("8, 4) =", "?")]
    for config_name, vps in [
        ('MIN_first+MAX_last', [(min_soul_p0, 0), (max_soul_last, -1)]),
        ('MAX_first+MIN_last', [(max_soul_p0, 0), (min_soul_last, -1)]),
    ]:
        preds = []
        for prompt, _ in sort_test:
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            sl = inp['input_ids'].shape[1]
            resolved = [(v, p if p >= 0 else sl + p) for v, p in vps]
            def inj(m, i, o, vps=resolved):
                return replace_multi_tokens(o, vps)
            h = model.model.layers[LAYER].register_forward_hook(inj)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            preds.append(pred)
        results['cross_task_%s' % config_name] = preds
        print("    %s -> preds: %s" % (config_name, preds))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Position sweep
    ax = axes[0]
    positions = sorted(pos_results.keys())
    min_accs = [pos_results[p]['MIN'] for p in positions]
    max_accs = [pos_results[p]['MAX'] for p in positions]
    ax.bar(np.array(positions) - 0.15, min_accs, 0.3, label='MIN', color='#E91E63',
           edgecolor='black')
    ax.bar(np.array(positions) + 0.15, max_accs, 0.3, label='MAX', color='#2196F3',
           edgecolor='black')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Accuracy')
    ax.set_title('Single-Token Position Sweep', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: Multi-position comparison
    ax = axes[1]
    config_names = list(multi_results.keys())
    accs = list(multi_results.values())
    colors = ['#E91E63' if 'MIN' in n.split('+')[0] else '#2196F3' for n in config_names]
    bars = ax.barh(range(len(config_names)), accs, color=colors, edgecolor='black')
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names, fontsize=8)
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc + 0.02, i, '%.0f%%' % (acc*100), va='center',
                fontweight='bold', fontsize=9)
    ax.set_xlabel('Accuracy')
    ax.set_title('Multi-Position Injection', fontweight='bold')
    ax.set_xlim(0, 1.3)

    # Panel 3: Summary
    ax = axes[2]
    ax.axis('off')
    best_single = max(max(min_accs), max(max_accs))
    best_multi = max(multi_results.values())
    summary = (
        "Multi-Token Soul Results\n\n"
        "Best single-position: %.0f%%\n"
        "  (position %d for %s)\n\n"
        "Best multi-position: %.0f%%\n"
        "  (%s)\n\n"
        "Improvement: %+.0f pp\n\n"
        "Cross-task mixing:\n"
        "  Creates hybrid behavior?" % (
            best_single * 100,
            positions[np.argmax(min_accs)] if max(min_accs) > max(max_accs)
            else positions[np.argmax(max_accs)],
            'MIN' if max(min_accs) > max(max_accs) else 'MAX',
            best_multi * 100,
            max(multi_results, key=multi_results.get),
            (best_multi - best_single) * 100,
        )
    )
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Phase 178: Multi-Token Soul\n'
                 '"What happens when we inject at multiple positions?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase178_multitoken_soul.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 178, 'name': 'multitoken_soul',
        'seq_len': seq_len,
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase178_multitoken_soul.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n  P178 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
