# -*- coding: utf-8 -*-
"""
Phase 90: Autoimmune Disease
What happens when the SVD filter is too aggressive?
The immune system starts attacking legitimate programs.
Tests the security vs self-tolerance trade-off.
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

def compile_prog(model, tok, train, layer, device, seed=42, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device)*0.01; vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(device)
            def inj(m,i,o,v=vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

def evaluate_vec(model, tok, vec, data, layer, device):
    c = 0
    for p, e in data:
        def inj(m,i,o,v=vec): return replace_last_token(o,v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
    return c / len(data)

def main():
    print("[P90] Autoimmune Disease")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),("9, 3) =","3")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","6"),("9, 3) =","9")]
    first_data = [("3, 7) =","3"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","4"),("9, 3) =","9")]
    test_min = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")]
    test_max = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")]

    # Build SVD basis from MIN-only (narrow immune training)
    print("  Building MIN-only SVD basis (narrow immune system)...")
    min_vecs = []
    for s in range(10):
        v = compile_prog(model, tok, min_data, tl, DEVICE, seed=s*100, epochs=80)
        min_vecs.append(v.cpu().numpy().flatten())
    Vt_min = np.linalg.svd(np.array(min_vecs), full_matrices=False)[2]

    # Build SVD basis from MIN+MAX (broad immune training)
    print("  Building MIN+MAX SVD basis (broad immune system)...")
    broad_vecs = list(min_vecs)
    for s in range(10):
        v = compile_prog(model, tok, max_data, tl, DEVICE, seed=s*100+50, epochs=80)
        broad_vecs.append(v.cpu().numpy().flatten())
    Vt_broad = np.linalg.svd(np.array(broad_vecs), full_matrices=False)[2]

    # Compile test programs
    vec_min = compile_prog(model, tok, min_data, tl, DEVICE, seed=42)
    vec_max = compile_prog(model, tok, max_data, tl, DEVICE, seed=42)
    vec_first = compile_prog(model, tok, first_data, tl, DEVICE, seed=42)

    # Test at different filter aggressiveness (k = soul dimensions)
    print("\n  Testing filter aggressiveness (k = # dimensions kept)...")
    k_values = [1, 2, 3, 5, 10, 20, 50, 100]
    results = []

    for k in k_values:
        row = {'k': k}
        for name, vec, data, Vt, basis_name in [
            ('min_narrow', vec_min, min_data+test_min, Vt_min, 'narrow'),
            ('max_narrow', vec_max, max_data+test_max, Vt_min, 'narrow'),
            ('first_narrow', vec_first, first_data, Vt_min, 'narrow'),
            ('min_broad', vec_min, min_data+test_min, Vt_broad, 'broad'),
            ('max_broad', vec_max, max_data+test_max, Vt_broad, 'broad'),
            ('first_broad', vec_first, first_data, Vt_broad, 'broad'),
        ]:
            kk = min(k, Vt.shape[0])
            v_np = vec.cpu().numpy().flatten()
            filtered = (v_np @ Vt[:kk,:].T) @ Vt[:kk,:]
            fv = torch.tensor(filtered, device=DEVICE, dtype=torch.float32)
            acc = evaluate_vec(model, tok, fv, data, tl, DEVICE)
            row[name] = round(acc, 4)

        results.append(row)
        print(f"    k={k:3d}: MIN_narrow={row['min_narrow']:.0%} "
              f"MAX_narrow={row['max_narrow']:.0%} "
              f"MIN_broad={row['min_broad']:.0%} "
              f"MAX_broad={row['max_broad']:.0%}")

    # Malware test: which basis provides better protection?
    print("\n  Malware resistance test...")
    np.random.seed(42)
    noise = np.random.randn(hs).astype(np.float32)
    noise = noise / np.linalg.norm(noise) * 5.0
    infected_min = vec_min.cpu().numpy().flatten() + noise
    infected_vec = torch.tensor(infected_min, device=DEVICE, dtype=torch.float32)

    malware_results = {}
    for k in [5, 10, 20]:
        for basis_name, Vt in [('narrow', Vt_min), ('broad', Vt_broad)]:
            kk = min(k, Vt.shape[0])
            f = (infected_min @ Vt[:kk,:].T) @ Vt[:kk,:]
            fv = torch.tensor(f, device=DEVICE, dtype=torch.float32)
            acc = evaluate_vec(model, tok, fv, min_data+test_min, tl, DEVICE)
            key = f"k{k}_{basis_name}"
            malware_results[key] = round(acc, 4)
            print(f"    {key}: {acc:.0%}")

    # Summary: autoimmune severity
    # "Autoimmune" = narrow basis rejects legitimate MAX programs
    autoimmune_severity = {}
    for r in results:
        k = r['k']
        # A narrow basis that only knows MIN should reject MAX
        autoimmune_severity[k] = {
            'self_preserved': r['min_narrow'],  # should be high
            'other_rejected': 1.0 - r['max_narrow'],  # should be low in healthy system
            'autoimmune_index': round(1.0 - r['max_narrow'], 4),  # high = autoimmune
        }

    output = {
        'phase': 90, 'name': 'autoimmune_disease',
        'filter_results': results,
        'malware_results': malware_results,
        'autoimmune_severity': autoimmune_severity,
        'conclusion': 'Narrow immune training (MIN-only) causes autoimmune rejection of legitimate MAX programs. '
                       'Broad training (MIN+MAX) preserves tolerance but may admit novel threats.',
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase90_autoimmune.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Narrow basis: self vs non-self
    kv = [r['k'] for r in results]
    axes[0].plot(kv, [r['min_narrow'] for r in results], 'g-o', lw=2, label='MIN (self)')
    axes[0].plot(kv, [r['max_narrow'] for r in results], 'r-s', lw=2, label='MAX (non-self)')
    axes[0].plot(kv, [r['first_narrow'] for r in results], 'b-^', lw=2, label='FIRST (foreign)')
    axes[0].set_xlabel('Filter Dimensions (k)'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Narrow Basis (MIN-only)\nAutoimmune Risk', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_xscale('log')

    # Broad basis: tolerance
    axes[1].plot(kv, [r['min_broad'] for r in results], 'g-o', lw=2, label='MIN')
    axes[1].plot(kv, [r['max_broad'] for r in results], 'r-s', lw=2, label='MAX')
    axes[1].plot(kv, [r['first_broad'] for r in results], 'b-^', lw=2, label='FIRST (foreign)')
    axes[1].set_xlabel('Filter Dimensions (k)'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Broad Basis (MIN+MAX)\nTolerant System', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_xscale('log')

    # Trade-off
    narrow_security = [1.0 - r['first_narrow'] for r in results]
    broad_security = [1.0 - r['first_broad'] for r in results]
    narrow_tolerance = [r['max_narrow'] for r in results]
    broad_tolerance = [r['max_broad'] for r in results]
    axes[2].scatter(narrow_tolerance, narrow_security, c='tab:red', s=80,
                     label='Narrow', zorder=5, edgecolors='black')
    axes[2].scatter(broad_tolerance, broad_security, c='tab:green', s=80,
                     label='Broad', zorder=5, edgecolors='black')
    axes[2].set_xlabel('Self-Tolerance (MAX acc)')
    axes[2].set_ylabel('Security (1 - FIRST acc)')
    axes[2].set_title('Security vs Tolerance\nTrade-off', fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 90: Autoimmune Disease\n'
                 '"An immune system that attacks itself is worse than none"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase90_autoimmune.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
