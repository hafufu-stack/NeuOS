# -*- coding: utf-8 -*-
"""
Phase 97: Soul Algebra
Test if soul vectors have algebraic structure:
  - Interpolation: slerp(MIN, MAX) = ?
  - Arithmetic: MIN + MAX = RANGE?
  - Analogy: MIN - MAX + FIRST = ?

"If souls are vectors, they must obey the laws of vector spaces."

Model: Qwen2.5-0.5B (GPU)
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
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

def evaluate_multi(model, tok, vec, datasets, layer, device):
    """Evaluate vec on multiple datasets, return {name: accuracy}."""
    results = {}
    for name, data in datasets.items():
        c = 0
        for p, e in data:
            def inj(m,i,o,v=vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            inp = tok(p, return_tensors='pt').to(device)
            with torch.no_grad(): out = model(**inp)
            h.remove()
            if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
        results[name] = round(float(c / len(data)), 4)
    return results

def slerp(v0, v1, t):
    """Spherical linear interpolation."""
    v0n = v0 / (v0.norm() + 1e-8)
    v1n = v1 / (v1.norm() + 1e-8)
    omega = torch.acos(torch.clamp(torch.dot(v0n, v1n), -1.0, 1.0))
    if omega.abs() < 1e-6:
        return (1-t)*v0 + t*v1
    return (torch.sin((1-t)*omega)/torch.sin(omega))*v0 + \
           (torch.sin(t*omega)/torch.sin(omega))*v1

def main():
    print("[P97] Soul Algebra")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    tl = 8
    for p in model.parameters(): p.requires_grad = False

    datasets = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),
                ("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("5, 4) =","4"),("3, 8) =","3")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),
                ("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("5, 4) =","5"),("3, 8) =","8")],
        'FIRST': [("3, 7) =","3"),("5, 2) =","5"),("8, 1) =","8"),
                  ("4, 6) =","4"),("9, 3) =","9"),
                  ("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","2"),
                  ("5, 4) =","5"),("3, 8) =","3")],
    }

    # Train base souls
    print("  Training base souls...")
    train_min = datasets['MIN'][:5]
    train_max = datasets['MAX'][:5]
    train_first = datasets['FIRST'][:5]

    soul_min = compile_prog(model, tok, train_min, tl, DEVICE, seed=42)
    soul_max = compile_prog(model, tok, train_max, tl, DEVICE, seed=42)
    soul_first = compile_prog(model, tok, train_first, tl, DEVICE, seed=42)

    base_accs = {
        'MIN_soul': evaluate_multi(model, tok, soul_min, datasets, tl, DEVICE),
        'MAX_soul': evaluate_multi(model, tok, soul_max, datasets, tl, DEVICE),
        'FIRST_soul': evaluate_multi(model, tok, soul_first, datasets, tl, DEVICE),
    }
    print(f"  MIN soul: {base_accs['MIN_soul']}")
    print(f"  MAX soul: {base_accs['MAX_soul']}")
    print(f"  FIRST soul: {base_accs['FIRST_soul']}")

    # Experiment 1: Interpolation (slerp between MIN and MAX)
    print("\n  Exp 1: Slerp interpolation MIN <-> MAX")
    interp_results = []
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for alpha in alphas:
        blended = slerp(soul_min, soul_max, alpha)
        accs = evaluate_multi(model, tok, blended, datasets, tl, DEVICE)
        interp_results.append({'alpha': alpha, **accs})
        if alpha in [0.0, 0.5, 1.0]:
            print(f"    alpha={alpha:.1f}: MIN={accs['MIN']:.0%}, "
                  f"MAX={accs['MAX']:.0%}, FIRST={accs['FIRST']:.0%}")

    # Experiment 2: Arithmetic (addition, subtraction)
    print("\n  Exp 2: Soul arithmetic")
    arith_results = {}

    # MIN + MAX (does it become something new?)
    sum_vec = soul_min + soul_max
    sum_vec = sum_vec / (sum_vec.norm() + 1e-8) * soul_min.norm()
    arith_results['MIN+MAX'] = evaluate_multi(model, tok, sum_vec, datasets, tl, DEVICE)
    print(f"    MIN+MAX: {arith_results['MIN+MAX']}")

    # MIN - MAX + FIRST (analogy: MIN is to MAX as ? is to FIRST)
    analogy = soul_min - soul_max + soul_first
    analogy = analogy / (analogy.norm() + 1e-8) * soul_min.norm()
    arith_results['MIN-MAX+FIRST'] = evaluate_multi(model, tok, analogy, datasets, tl, DEVICE)
    print(f"    MIN-MAX+FIRST: {arith_results['MIN-MAX+FIRST']}")

    # Average of all three
    avg_vec = (soul_min + soul_max + soul_first) / 3
    avg_vec = avg_vec / (avg_vec.norm() + 1e-8) * soul_min.norm()
    arith_results['AVG(all)'] = evaluate_multi(model, tok, avg_vec, datasets, tl, DEVICE)
    print(f"    AVG(all): {arith_results['AVG(all)']}")

    # Experiment 3: Orthogonal projection (remove MAX from MIN)
    print("\n  Exp 3: Soul subtraction (project out)")
    max_dir = soul_max / (soul_max.norm() + 1e-8)
    min_minus_max_proj = soul_min - (soul_min @ max_dir) * max_dir
    min_minus_max_proj = min_minus_max_proj / (min_minus_max_proj.norm() + 1e-8) * soul_min.norm()
    arith_results['MIN_orthog_MAX'] = evaluate_multi(
        model, tok, min_minus_max_proj, datasets, tl, DEVICE)
    print(f"    MIN orthog MAX: {arith_results['MIN_orthog_MAX']}")

    # Cosine similarity matrix
    souls = {'MIN': soul_min, 'MAX': soul_max, 'FIRST': soul_first}
    cos_matrix = {}
    for n1, v1 in souls.items():
        for n2, v2 in souls.items():
            sim = float(torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)).item())
            cos_matrix[f'{n1}-{n2}'] = round(sim, 4)
    print(f"\n  Cosine similarities: {cos_matrix}")

    # Save
    output = {
        'phase': 97, 'name': 'soul_algebra',
        'base_accuracies': base_accs,
        'interpolation': interp_results,
        'arithmetic': arith_results,
        'cosine_matrix': cos_matrix,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase97_soul_algebra.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Interpolation curves
    for task in ['MIN', 'MAX', 'FIRST']:
        vals = [r[task] for r in interp_results]
        axes[0].plot(alphas, vals, '-o', lw=2, ms=4, label=task)
    axes[0].set_xlabel('alpha (0=MIN, 1=MAX)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Slerp Interpolation\nMIN <-> MAX', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0.5, color='gray', ls='--', alpha=0.3)

    # Arithmetic results
    ops = list(arith_results.keys())
    x = np.arange(len(ops))
    w = 0.25
    for i, task in enumerate(['MIN', 'MAX', 'FIRST']):
        vals = [arith_results[op][task] for op in ops]
        axes[1].bar(x + i*w, vals, w, label=task, edgecolor='black')
    axes[1].set_xticks(x + w); axes[1].set_xticklabels(ops, fontsize=7, rotation=15)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Soul Arithmetic', fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].set_ylim(0, 1.2)

    # Cosine similarity heatmap
    names = ['MIN', 'MAX', 'FIRST']
    mat = np.array([[cos_matrix[f'{n1}-{n2}'] for n2 in names] for n1 in names])
    im = axes[2].imshow(mat, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(names)
    axes[2].set_yticks(range(3)); axes[2].set_yticklabels(names)
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center',
                        fontweight='bold', fontsize=11)
    axes[2].set_title('Soul Cosine Similarity', fontweight='bold')
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    plt.suptitle('Phase 97: Soul Algebra\n'
                 '"If souls are vectors, they must obey the laws of vector spaces"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase97_soul_algebra.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
