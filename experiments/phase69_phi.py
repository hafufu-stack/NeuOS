# -*- coding: utf-8 -*-
"""
Phase 69: Integrated Information (Phi)
Measure Phi (integrated information) for neural programs.
How much information does the system generate "above and beyond"
its independent parts? IIT-inspired consciousness measure.

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
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_prog(model, tok, train, layer, device, seed=42):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(80):
        for prompt, target_str in train:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def inject(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def compute_phi(model, tok, vec, prompt, inject_layer, read_layers, device):
    """
    Compute a simplified Phi (integrated information) measure.
    Phi = MI(whole) - sum(MI(parts))
    Using mutual information between layer activations as proxy.
    """
    # Get activations at multiple layers
    caps = {}
    hooks = []
    for l in read_layers:
        def make_cap(layer_idx):
            def fn(module, input, output):
                caps[layer_idx] = get_last_token(output).float().cpu().flatten()
            return fn
        h = model.model.layers[l].register_forward_hook(make_cap(l))
        hooks.append(h)

    def inject_fn(module, input, output, v=vec):
        return replace_last_token(output, v)
    h_inj = model.model.layers[inject_layer].register_forward_hook(inject_fn)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)
    h_inj.remove()
    for h in hooks:
        h.remove()

    # Compute "integration" as: how much does knowing layer L_i tell you about L_j?
    # Use cosine similarity as proxy for mutual information
    layers = sorted(caps.keys())
    vecs_list = [caps[l].numpy() for l in layers]

    # Whole-system information: concatenated correlation
    whole = np.concatenate(vecs_list)
    whole_entropy = -np.sum(np.abs(whole) * np.log(np.abs(whole) + 1e-10)) / len(whole)

    # Part information: entropy of each layer independently
    part_entropies = []
    for v in vecs_list:
        e = -np.sum(np.abs(v) * np.log(np.abs(v) + 1e-10)) / len(v)
        part_entropies.append(e)

    # Inter-layer correlations (integration measure)
    correlations = []
    for i in range(len(vecs_list)):
        for j in range(i+1, len(vecs_list)):
            # Truncate to same length
            min_len = min(len(vecs_list[i]), len(vecs_list[j]))
            corr = np.corrcoef(vecs_list[i][:min_len], vecs_list[j][:min_len])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlations.append(abs(corr))

    # Phi = average inter-layer correlation (integration beyond parts)
    phi = np.mean(correlations) if correlations else 0.0

    return {
        'phi': float(phi),
        'whole_entropy': float(whole_entropy),
        'part_entropies': [float(e) for e in part_entropies],
        'correlations': [float(c) for c in correlations],
    }


def main():
    print("[P69] Integrated Information (Phi)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8
    read_layers = [10, 14, 18, 22]

    for p in model.parameters():
        p.requires_grad = False

    # Test programs
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")]

    min_vec = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42)
    max_vec = compile_prog(model, tok, max_data, target_layer, DEVICE, seed=99)

    # Measure Phi for different program types
    prompts = ["3, 7) =", "5, 2) =", "8, 1) ="]
    results = {}

    print("\n  Measuring Phi for trained programs...")
    for name, vec in [('MIN', min_vec), ('MAX', max_vec)]:
        phis = []
        for prompt in prompts:
            phi_data = compute_phi(model, tok, vec, prompt, target_layer, read_layers, DEVICE)
            phis.append(phi_data['phi'])
        avg_phi = float(np.mean(phis))
        results[name] = {
            'avg_phi': round(avg_phi, 4),
            'detail': phi_data,
        }
        print(f"    {name}: Phi = {avg_phi:.4f}")

    print("\n  Measuring Phi for random noise...")
    noise_phis = []
    for i in range(5):
        noise_vec = torch.randn(model.config.hidden_size, device=DEVICE) * 0.1
        phis = []
        for prompt in prompts:
            phi_data = compute_phi(model, tok, noise_vec, prompt, target_layer, read_layers, DEVICE)
            phis.append(phi_data['phi'])
        avg_phi = float(np.mean(phis))
        noise_phis.append(avg_phi)
    results['NOISE'] = {
        'avg_phi': round(float(np.mean(noise_phis)), 4),
        'std': round(float(np.std(noise_phis)), 4),
    }
    print(f"    NOISE: Phi = {np.mean(noise_phis):.4f} +/- {np.std(noise_phis):.4f}")

    print("\n  Measuring Phi for zero vector (no program)...")
    zero_vec = torch.zeros(model.config.hidden_size, device=DEVICE)
    phis = []
    for prompt in prompts:
        phi_data = compute_phi(model, tok, zero_vec, prompt, target_layer, read_layers, DEVICE)
        phis.append(phi_data['phi'])
    results['ZERO'] = {'avg_phi': round(float(np.mean(phis)), 4)}
    print(f"    ZERO: Phi = {np.mean(phis):.4f}")

    # Phi vs program complexity
    print("\n  Phi vs complexity...")
    complexity_phis = {}
    for n_train in [1, 2, 3, 5]:
        v = compile_prog(model, tok, min_data[:n_train], target_layer, DEVICE, seed=42)
        phis = []
        for prompt in prompts:
            phi_data = compute_phi(model, tok, v, prompt, target_layer, read_layers, DEVICE)
            phis.append(phi_data['phi'])
        avg_phi = float(np.mean(phis))
        complexity_phis[n_train] = round(avg_phi, 4)
        print(f"    n_train={n_train}: Phi={avg_phi:.4f}")

    # Save
    output = {
        'phase': 69, 'name': 'integrated_information',
        'results': {k: v for k, v in results.items()},
        'complexity_phis': complexity_phis,
        'read_layers': read_layers,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase69_phi.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = ['MIN', 'MAX', 'NOISE', 'ZERO']
    phis_plot = [results[n]['avg_phi'] for n in names]
    colors = ['tab:blue', 'tab:red', 'tab:gray', 'lightgray']
    axes[0].bar(names, phis_plot, color=colors, edgecolor='black')
    axes[0].set_ylabel('Phi (Integration)')
    axes[0].set_title('Integrated Information by Program Type', fontweight='bold')
    for i, v in enumerate(phis_plot):
        axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')

    # Complexity vs Phi
    ns = sorted(complexity_phis.keys())
    phi_vals = [complexity_phis[n] for n in ns]
    axes[1].plot(ns, phi_vals, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Training Examples')
    axes[1].set_ylabel('Phi')
    axes[1].set_title('Phi vs Program Complexity', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Layer correlations for MIN
    if 'MIN' in results and 'detail' in results['MIN']:
        corrs = results['MIN']['detail']['correlations']
        pairs = []
        for i in range(len(read_layers)):
            for j in range(i+1, len(read_layers)):
                pairs.append(f'L{read_layers[i]}-L{read_layers[j]}')
        if len(pairs) == len(corrs):
            axes[2].bar(range(len(corrs)), corrs, color='tab:purple', edgecolor='black')
            axes[2].set_xticks(range(len(pairs)))
            axes[2].set_xticklabels(pairs, rotation=45, fontsize=8)
            axes[2].set_ylabel('|Correlation|')
            axes[2].set_title('Inter-Layer Integration (MIN)', fontweight='bold')

    plt.suptitle('Phase 69: Integrated Information (Phi)\nMeasuring consciousness in neural programs',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase69_phi.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
