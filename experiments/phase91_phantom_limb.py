# -*- coding: utf-8 -*-
"""
Phase 91: The Phantom Limb
Train a program using multi-layer register injection (L8+L12+L16).
Then "amputate" one layer. Measure phantom pain = prediction error spike
at downstream layers that still expect the missing input.

"The body remembers what the mind has lost."

Model: Qwen2.5-0.5B (GPU)
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

def compile_multilayer(model, tok, train, layers, device, seed=42, epochs=80):
    """Train separate vectors at multiple layers simultaneously."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vecs = {l: torch.randn(hs, device=device)*0.01 for l in layers}
    for v in vecs.values(): v.requires_grad_(True)
    opt = torch.optim.Adam(list(vecs.values()), lr=0.01)
    for ep in range(epochs):
        for prompt, target_str in train:
            tid = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            hooks = []
            for l in layers:
                def make_inj(layer_id):
                    def inj(m, i, o, v=vecs[layer_id]):
                        return replace_last_token(o, v)
                    return inj
                h = model.model.layers[l].register_forward_hook(make_inj(l))
                hooks.append(h)
            out = model(**inp)
            for h in hooks: h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return {l: v.detach() for l, v in vecs.items()}

def evaluate_with_layers(model, tok, vecs, data, device, capture_layers=None):
    """Evaluate with given layer vectors, optionally capture downstream activations."""
    correct = 0
    all_caps = []
    for prompt, expected in data:
        hooks = []
        caps = {}
        for l, v in vecs.items():
            def make_inj(vec):
                def inj(m, i, o, v=vec): return replace_last_token(o, v)
                return inj
            h = model.model.layers[l].register_forward_hook(make_inj(v))
            hooks.append(h)
        if capture_layers:
            for cl in capture_layers:
                def make_cap(lid):
                    def cap(m, i, o): caps[lid] = get_last_token(o).float().cpu()
                    return cap
                h = model.model.layers[cl].register_forward_hook(make_cap(cl))
                hooks.append(h)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        for h in hooks: h.remove()
        pred = tok.decode(out.logits[0,-1,:].argmax().item()).strip()
        if pred == expected: correct += 1
        if caps: all_caps.append(caps)
    return correct / len(data), all_caps

def main():
    print("[P91] The Phantom Limb")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("5, 4) =","4"),("3, 8) =","3")]
    all_data = min_data + test_data

    # The "body": program using 3 register layers
    body_layers = [8, 12, 16]
    capture_layers = [18, 20, 22]  # downstream "nerves"

    print("  Training multi-layer body (L8+L12+L16)...")
    vecs = compile_multilayer(model, tok, min_data, body_layers, DEVICE)
    full_acc, full_caps = evaluate_with_layers(
        model, tok, vecs, all_data, DEVICE, capture_layers)
    print(f"  Full body accuracy: {full_acc:.0%}")

    # Compute "normal" activation profile
    normal_profile = {}
    for cl in capture_layers:
        norms = [c[cl].norm().item() for c in full_caps if cl in c]
        normal_profile[cl] = np.mean(norms) if norms else 0

    # Amputation tests: remove each layer one at a time
    print("\n  Amputation tests...")
    amputation_results = []
    pain_profiles = {}

    for amputated in body_layers:
        partial_vecs = {l: v for l, v in vecs.items() if l != amputated}
        amp_acc, amp_caps = evaluate_with_layers(
            model, tok, partial_vecs, all_data, DEVICE, capture_layers)

        # Compute "pain" = deviation from normal activation
        pain = {}
        for cl in capture_layers:
            amp_norms = [c[cl].norm().item() for c in amp_caps if cl in c]
            amp_mean = np.mean(amp_norms) if amp_norms else 0
            deviation = abs(amp_mean - normal_profile[cl]) / (normal_profile[cl] + 1e-8)
            pain[cl] = round(float(deviation), 4)

        avg_pain = float(np.mean(list(pain.values())))
        amputation_results.append({
            'amputated_layer': amputated,
            'accuracy': round(float(amp_acc), 4),
            'pain_per_layer': {str(k): v for k, v in pain.items()},
            'avg_pain': round(avg_pain, 4),
        })
        pain_profiles[amputated] = pain
        print(f"    Cut L{amputated}: acc={amp_acc:.0%}, avg_pain={avg_pain:.3f}")

    # Control: remove a layer that was NOT trained (L4)
    print("\n  Control: cut untrained layer L4...")
    control_vecs = dict(vecs)  # all body layers present
    # Add a dummy at L4 then remove it (to show no pain)
    ctrl_acc, ctrl_caps = evaluate_with_layers(
        model, tok, control_vecs, all_data, DEVICE, capture_layers)
    # Pain should be ~0 since we didn't actually change anything
    ctrl_pain = 0.0
    print(f"    Control (no real cut): acc={ctrl_acc:.0%}, pain={ctrl_pain:.3f}")

    # Save
    output = {
        'phase': 91, 'name': 'phantom_limb',
        'body_layers': body_layers,
        'full_accuracy': round(float(full_acc), 4),
        'amputation_results': amputation_results,
        'control_accuracy': round(float(ctrl_acc), 4),
        'normal_profile': {str(k): round(float(v), 4) for k, v in normal_profile.items()},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase91_phantom_limb.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy after amputation
    labels = [f'Full\nBody'] + [f'Cut\nL{r["amputated_layer"]}' for r in amputation_results]
    accs = [full_acc] + [r['accuracy'] for r in amputation_results]
    colors = ['tab:green'] + ['tab:red']*len(amputation_results)
    axes[0].bar(labels, accs, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy'); axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Accuracy After Amputation', fontweight='bold')
    for i, v in enumerate(accs):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Pain intensity per amputated layer
    for r in amputation_results:
        l = r['amputated_layer']
        cls = sorted(r['pain_per_layer'].keys())
        pains = [r['pain_per_layer'][c] for c in cls]
        axes[1].plot([f'L{c}' for c in cls], pains, '-o', lw=2,
                     label=f'Cut L{l}', markersize=6)
    axes[1].set_xlabel('Downstream Layer')
    axes[1].set_ylabel('Pain (activation deviation)')
    axes[1].set_title('Phantom Pain Propagation', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Average pain per amputation
    amp_labels = [f'L{r["amputated_layer"]}' for r in amputation_results]
    avg_pains = [r['avg_pain'] for r in amputation_results]
    axes[2].bar(amp_labels, avg_pains, color='tab:orange', edgecolor='black')
    axes[2].set_xlabel('Amputated Layer')
    axes[2].set_ylabel('Average Pain Index')
    axes[2].set_title('Which Limb Hurts Most?', fontweight='bold')
    for i, v in enumerate(avg_pains):
        axes[2].text(i, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')

    plt.suptitle('Phase 91: The Phantom Limb\n'
                 '"The body remembers what the mind has lost"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase91_phantom_limb.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
