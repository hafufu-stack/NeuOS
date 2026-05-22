# -*- coding: utf-8 -*-
"""
Phase 155: Dynamic Pipeline Rewiring
The model self-diagnoses the task type and dynamically reroutes its own
forward pass by skipping or bypassing layers at inference time.

"I know which parts of my brain to use for this task."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

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


def infer_normal(model, tok, prompt, device, soul_vec, layer=LAYER):
    """Standard inference with soul injection."""
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


def infer_with_skip(model, tok, prompt, device, soul_vec,
                    inject_layer=LAYER, skip_layers=None):
    """Inference with layer skipping: bypass certain layers entirely."""
    hooks = []

    # Install skip hooks (pass through input unchanged)
    if skip_layers:
        for li in skip_layers:
            def make_skip(idx):
                def hook_fn(m, inp, out):
                    # Return input unchanged (skip this layer's computation)
                    if isinstance(inp, tuple):
                        return inp[0]
                    return inp
                return hook_fn
            hooks.append(model.model.layers[li].register_forward_hook(make_skip(li)))

    # Soul injection
    def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
    hooks.append(model.model.layers[inject_layer].register_forward_hook(inj))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    for h in hooks:
        h.remove()

    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    pred = tok.decode(logits.argmax().item()).strip()
    return pred, entropy


def infer_with_warp(model, tok, prompt, device, soul_vec,
                    inject_layer=LAYER, warp_from=11, warp_to=16):
    """
    Inference with layer warp: copy hidden state from warp_from to warp_to,
    skipping intermediate layers (the 'grammar police' bypass).
    """
    captured = {}
    hooks = []

    # Capture at warp_from
    def capture_hook(m, inp, out):
        captured['state'] = get_last_token(out)
    hooks.append(model.model.layers[warp_from].register_forward_hook(capture_hook))

    # Inject captured state at warp_to
    def warp_hook(m, inp, out):
        if 'state' in captured:
            return replace_last_token(out, captured['state'])
    hooks.append(model.model.layers[warp_to].register_forward_hook(warp_hook))

    # Soul injection
    def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
    hooks.append(model.model.layers[inject_layer].register_forward_hook(inj))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    for h in hooks:
        h.remove()

    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    pred = tok.decode(logits.argmax().item()).strip()
    return pred, entropy


def main():
    print("[P155] Dynamic Pipeline Rewiring")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train souls
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]

    print("  Training MIN and MAX souls...")
    soul_min = train_soul(model, tok, min_data, DEVICE, seed=42)
    soul_max = train_soul(model, tok, max_data, DEVICE, seed=43)

    test_cases = [
        ("7, 2) =", "2", soul_min, "MIN"),
        ("6, 3) =", "3", soul_min, "MIN"),
        ("2, 9) =", "2", soul_min, "MIN"),
        ("1, 5) =", "1", soul_min, "MIN"),
        ("8, 4) =", "4", soul_min, "MIN"),
        ("3, 7) =", "7", soul_max, "MAX"),
        ("5, 2) =", "5", soul_max, "MAX"),
        ("1, 8) =", "8", soul_max, "MAX"),
        ("4, 6) =", "6", soul_max, "MAX"),
        ("9, 1) =", "9", soul_max, "MAX"),
    ]

    # Routing configurations to test
    configs = {
        'normal': {'desc': 'Standard (all 24 layers)', 'skip': None, 'warp': None},
        'skip_12_15': {'desc': 'Skip L12-L15 (grammar bypass)', 'skip': [12,13,14,15], 'warp': None},
        'skip_18_21': {'desc': 'Skip L18-L21 (late bypass)', 'skip': [18,19,20,21], 'warp': None},
        'skip_0_3': {'desc': 'Skip L0-L3 (early bypass)', 'skip': [0,1,2,3], 'warp': None},
        'warp_11_16': {'desc': 'Warp L11->L16 (GSF bypass)', 'skip': None, 'warp': (11, 16)},
        'warp_8_16': {'desc': 'Warp L8->L16 (max bypass)', 'skip': None, 'warp': (8, 16)},
    }

    all_results = {}

    for config_name, config in configs.items():
        print("\n  --- %s ---" % config['desc'])
        correct = 0
        entropies = []

        for prompt, expected, soul, task in test_cases:
            if config['warp']:
                pred, ent = infer_with_warp(model, tok, prompt, DEVICE, soul,
                                             warp_from=config['warp'][0],
                                             warp_to=config['warp'][1])
            elif config['skip']:
                pred, ent = infer_with_skip(model, tok, prompt, DEVICE, soul,
                                             skip_layers=config['skip'])
            else:
                pred, ent = infer_normal(model, tok, prompt, DEVICE, soul)

            if pred == expected:
                correct += 1
            entropies.append(ent)

        acc = correct / len(test_cases)
        avg_ent = np.mean(entropies)
        all_results[config_name] = {
            'desc': config['desc'],
            'accuracy': round(acc, 4),
            'avg_entropy': round(avg_ent, 4),
        }
        print("  Accuracy: %.0f%%, Avg entropy: %.4f" % (acc * 100, avg_ent))

    # Adaptive routing: pick best config per-sample based on entropy
    print("\n  --- Adaptive Routing (entropy-minimizing) ---")
    adaptive_correct = 0
    adaptive_details = []
    for prompt, expected, soul, task in test_cases:
        best_pred = None
        best_ent = float('inf')
        best_config = None

        for config_name, config in configs.items():
            if config['warp']:
                pred, ent = infer_with_warp(model, tok, prompt, DEVICE, soul,
                                             warp_from=config['warp'][0],
                                             warp_to=config['warp'][1])
            elif config['skip']:
                pred, ent = infer_with_skip(model, tok, prompt, DEVICE, soul,
                                             skip_layers=config['skip'])
            else:
                pred, ent = infer_normal(model, tok, prompt, DEVICE, soul)

            if ent < best_ent:
                best_ent = ent
                best_pred = pred
                best_config = config_name

        correct = (best_pred == expected)
        if correct:
            adaptive_correct += 1
        adaptive_details.append({
            'task': task, 'prompt': prompt[:12],
            'chosen_config': best_config, 'pred': best_pred,
            'expected': expected, 'entropy': round(best_ent, 4),
            'correct': correct,
        })

    adaptive_acc = adaptive_correct / len(test_cases)
    print("  Adaptive accuracy: %.0f%%" % (adaptive_acc * 100))
    all_results['adaptive'] = {
        'desc': 'Adaptive (entropy-minimizing)',
        'accuracy': round(adaptive_acc, 4),
        'details': adaptive_details,
    }

    # Config usage in adaptive mode
    config_usage = {}
    for d in adaptive_details:
        c = d['chosen_config']
        config_usage[c] = config_usage.get(c, 0) + 1
    print("  Config usage:", config_usage)

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Accuracy per config
    ax = axes[0]
    config_names = [k for k in all_results.keys() if k != 'adaptive']
    accs = [all_results[k]['accuracy'] for k in config_names]
    colors_list = ['#2196F3', '#FF9800', '#9C27B0', '#E91E63', '#00BCD4', '#607D8B']
    bars = ax.barh(range(len(config_names)), accs,
                   color=colors_list[:len(config_names)], edgecolor='black')
    # Add adaptive as special
    ax.barh(len(config_names), adaptive_acc, color='#4CAF50', edgecolor='black',
            hatch='//')
    all_names = config_names + ['adaptive']
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels([all_results[k]['desc'][:30] for k in all_names], fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_xlim(0, 1.1)
    for i, v in enumerate(accs + [adaptive_acc]):
        ax.text(v + 0.02, i, '%.0f%%' % (v*100), va='center', fontweight='bold')
    ax.set_title('Accuracy by Routing Config', fontweight='bold')

    # Panel 2: Entropy per config
    ax = axes[1]
    ents = [all_results[k].get('avg_entropy', 0) for k in config_names]
    ax.bar(range(len(config_names)), ents,
           color=colors_list[:len(config_names)], edgecolor='black')
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels([k[:10] for k in config_names], fontsize=8, rotation=30)
    ax.set_ylabel('Average Entropy')
    ax.set_title('Entropy by Routing Config\n(lower = more confident)', fontweight='bold')

    # Panel 3: Adaptive routing choices
    ax = axes[2]
    if config_usage:
        labels = list(config_usage.keys())
        sizes = list(config_usage.values())
        ax.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90,
               colors=colors_list[:len(labels)])
    ax.set_title('Adaptive Router:\nWhich config was chosen?', fontweight='bold')

    plt.suptitle('Phase 155: Dynamic Pipeline Rewiring\n'
                 '"I know which parts of my brain to use"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase155_pipeline_rewiring.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 155, 'name': 'dynamic_pipeline_rewiring',
        'results': all_results,
        'adaptive_accuracy': round(adaptive_acc, 4),
        'config_usage': config_usage,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase155_pipeline_rewiring.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
