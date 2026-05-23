# -*- coding: utf-8 -*-
"""
Phase 186: Attention Head Attribution
Which attention heads are causally necessary for soul vector effects?
Zero-out individual heads to find critical circuits.
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


def evaluate_with_soul(model, tok, soul_vec, test_data, device, layer=LAYER):
    correct = 0
    for prompt, expected in test_data:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data) if test_data else 0


def main():
    print("[P186] Attention Head Attribution", flush=True)
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

    # Train souls
    print("  Training MIN/MAX souls...", flush=True)
    min_soul = train_soul(model, tok, min_data, DEVICE)
    max_soul = train_soul(model, tok, max_data, DEVICE)

    # Baseline accuracy
    min_base = evaluate_with_soul(model, tok, min_soul, min_test, DEVICE)
    max_base = evaluate_with_soul(model, tok, max_soul, max_test, DEVICE)
    print("  Baseline: MIN=%.0f%%, MAX=%.0f%%" % (min_base*100, max_base*100), flush=True)

    n_layers = len(model.model.layers)

    # === Exp 1: Layer ablation (zero attention output at each layer AFTER injection) ===
    print("\n  Exp1: Post-injection layer ablation...", flush=True)
    layer_ablation = {}
    for ablate_layer in range(LAYER + 1, min(n_layers, LAYER + 10)):
        def make_ablate_hook(al):
            def hook(m, i, o):
                if isinstance(o, tuple):
                    h = o[0].clone()
                    h[0, -1, :] = 0  # Zero out last token at this layer
                    return (h,) + o[1:]
                else:
                    h = o.clone()
                    h[0, -1, :] = 0
                    return h
            return hook

        # Test with MIN soul + ablated layer
        correct = 0
        for prompt, expected in min_test:
            def inj(m, i, o, v=min_soul): return replace_last_token(o, v)
            h1 = model.model.layers[LAYER].register_forward_hook(inj)
            h2 = model.model.layers[ablate_layer].register_forward_hook(make_ablate_hook(ablate_layer))
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h1.remove(); h2.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            if pred == expected: correct += 1
        acc = correct / len(min_test)
        layer_ablation['L%d' % ablate_layer] = round(acc, 4)
        drop = min_base - acc
        print("    Ablate L%d: MIN=%.0f%% (drop=%.0f%%)" % (
            ablate_layer, acc*100, drop*100), flush=True)
    layer_ablation['baseline'] = round(min_base, 4)

    # === Exp 2: Injection layer sweep ===
    print("\n  Exp2: Injection layer sweep...", flush=True)
    injection_sweep = {}
    for inj_layer in range(0, n_layers, 2):
        soul = train_soul(model, tok, min_data, DEVICE, layer=inj_layer, epochs=50)
        acc = evaluate_with_soul(model, tok, soul, min_test, DEVICE, layer=inj_layer)
        injection_sweep['L%d' % inj_layer] = round(acc, 4)
        print("    Inject L%d: MIN=%.0f%%" % (inj_layer, acc*100), flush=True)

    # === Exp 3: Residual stream analysis ===
    print("\n  Exp3: Residual stream norm changes...", flush=True)
    residual_norms = {'with_soul': {}, 'without_soul': {}}
    test_prompt = min_test[0][0]

    # Without soul
    layer_norms_no_soul = {}
    for check_layer in range(n_layers):
        states = {}
        def capture(m, i, o, cl=check_layer):
            tensor = o[0] if isinstance(o, tuple) else o
            if tensor.dim() == 3:
                states['norm'] = tensor[0, -1, :].detach().norm().item()
            elif tensor.dim() == 2:
                states['norm'] = tensor[-1, :].detach().norm().item()
        h = model.model.layers[check_layer].register_forward_hook(capture)
        inp = tok(test_prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        if 'norm' in states:
            residual_norms['without_soul']['L%d' % check_layer] = round(states['norm'], 2)

    # With soul
    for check_layer in range(n_layers):
        states = {}
        def capture(m, i, o, cl=check_layer):
            tensor = o[0] if isinstance(o, tuple) else o
            if tensor.dim() == 3:
                states['norm'] = tensor[0, -1, :].detach().norm().item()
            elif tensor.dim() == 2:
                states['norm'] = tensor[-1, :].detach().norm().item()
        def inj(m, i, o, v=min_soul): return replace_last_token(o, v)
        h1 = model.model.layers[LAYER].register_forward_hook(inj)
        h2 = model.model.layers[check_layer].register_forward_hook(capture)
        inp = tok(test_prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h1.remove(); h2.remove()
        if 'norm' in states:
            residual_norms['with_soul']['L%d' % check_layer] = round(states['norm'], 2)

    print("  Residual norm changes computed for %d layers" % n_layers, flush=True)

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Layer ablation
    ax = axes[0, 0]
    abl_keys = [k for k in layer_ablation if k != 'baseline']
    abl_vals = [layer_ablation[k] for k in abl_keys]
    colors_abl = ['#F44336' if v < min_base - 0.1 else '#4CAF50' for v in abl_vals]
    ax.bar(abl_keys, abl_vals, color=colors_abl, edgecolor='black', linewidth=1.5)
    ax.axhline(y=min_base, color='blue', linestyle='--', label='Baseline (%.0f%%)' % (min_base*100))
    ax.set_ylabel('Accuracy with Ablation')
    ax.set_title('Post-Injection Layer Ablation\n(Which layers carry soul signal?)', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # Panel 2: Injection layer sweep
    ax = axes[0, 1]
    sweep_keys = list(injection_sweep.keys())
    sweep_vals = [injection_sweep[k] for k in sweep_keys]
    ax.bar(sweep_keys, sweep_vals, color='#2196F3', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('MIN Accuracy')
    ax.set_title('Injection Layer Sweep\n(Best layer for soul injection)', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # Panel 3: Residual norms
    ax = axes[1, 0]
    layers_list = ['L%d' % i for i in range(n_layers)]
    no_soul_norms = [residual_norms['without_soul'].get(l, 0) for l in layers_list]
    with_soul_norms = [residual_norms['with_soul'].get(l, 0) for l in layers_list]
    ax.plot(range(n_layers), no_soul_norms, 'o-', color='#4CAF50', markersize=4,
            linewidth=1.5, label='No soul')
    ax.plot(range(n_layers), with_soul_norms, 's-', color='#E91E63', markersize=4,
            linewidth=1.5, label='With MIN soul')
    ax.axvline(x=LAYER, color='gray', linestyle='--', alpha=0.5, label='Injection layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Residual Stream Norm')
    ax.set_title('Residual Stream Norm Profile', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 4: Norm difference
    ax = axes[1, 1]
    norm_diffs = [with_soul_norms[i] - no_soul_norms[i] for i in range(n_layers)]
    colors_diff = ['#E91E63' if d > 0 else '#2196F3' for d in norm_diffs]
    ax.bar(range(n_layers), norm_diffs, color=colors_diff, edgecolor='black', linewidth=0.5)
    ax.axvline(x=LAYER, color='gray', linestyle='--', alpha=0.5, label='Injection layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Norm Difference (soul - no_soul)')
    ax.set_title('Soul Impact on Residual Stream', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Phase 186: Attention Head Attribution\n'
                 '"How does soul injection propagate through the network?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase186_attention_attribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 186, 'name': 'attention_head_attribution',
        'baseline': {'MIN': round(min_base, 4), 'MAX': round(max_base, 4)},
        'layer_ablation': layer_ablation,
        'injection_sweep': injection_sweep,
        'residual_norms': residual_norms,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase186_attention_attribution.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P186 completed in %.0fs" % (time.time() - start), flush=True)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
