# -*- coding: utf-8 -*-
"""
Phase 116: Convergent Evolution
4 methods (gradient, GA, iterated learning, random search) -> same function?

"All roads lead to Rome, but they take different routes."
"""
import torch, json, os, gc, numpy as np, time, sys, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def gradient_train(model, tok, train, layer, device, seed=42, epochs=80):
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

def ga_evolve(model, tok, data, layer, device, pop_size=20, n_gen=20, seed=0):
    hs = model.config.hidden_size
    rng = np.random.RandomState(seed)
    pop = [torch.randn(hs, device=device)*0.1 for _ in range(pop_size)]
    for gen in range(n_gen):
        fits = [evaluate_vec(model, tok, v, data, layer, device) for v in pop]
        si = np.argsort(fits)[::-1]
        new_pop = [pop[si[0]].clone(), pop[si[1]].clone()]
        while len(new_pop) < pop_size:
            i, j = rng.choice(pop_size, 2, replace=False)
            parent = pop[i] if fits[i] >= fits[j] else pop[j]
            child = parent.clone() + torch.randn(hs, device=device)*0.05
            new_pop.append(child)
        pop = new_pop
    fits = [evaluate_vec(model, tok, v, data, layer, device) for v in pop]
    return pop[int(np.argmax(fits))].detach()

def iterated_learn(model, tok, train, layer, device, n_gen=5, seed=0):
    hs = model.config.hidden_size
    # Train initial teacher
    teacher = gradient_train(model, tok, train, layer, device, seed=seed, epochs=60)
    for gen in range(n_gen):
        # Student imitates teacher's outputs
        student_data = []
        for p, _ in train:
            def inj(m,i,o,v=teacher): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            inp = tok(p, return_tensors='pt').to(device)
            with torch.no_grad(): out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0,-1,:].argmax().item()).strip()
            student_data.append((p, pred))
        teacher = gradient_train(model, tok, student_data, layer, device,
                                 seed=seed+(gen+1)*100, epochs=60)
    return teacher.detach()

def random_search(model, tok, data, layer, device, n_tries=500, seed=0):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    best_v, best_a = None, 0
    for _ in range(n_tries):
        v = torch.randn(hs, device=device) * 0.5
        a = evaluate_vec(model, tok, v, data, layer, device)
        if a > best_a: best_a = a; best_v = v.clone()
    return best_v.detach() if best_v is not None else torch.zeros(hs, device=device)

def main():
    print("[P116] Convergent Evolution")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    tl = 8
    for p in model.parameters(): p.requires_grad = False

    train = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
             ("4, 6) =","4"),("9, 3) =","3")]
    test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")]
    all_data = train + test

    N_PER = 15
    methods = {'Gradient': [], 'GA': [], 'Iterated': [], 'Random': []}
    accs = {'Gradient': [], 'GA': [], 'Iterated': [], 'Random': []}

    print(f"  Gradient descent ({N_PER} seeds)...")
    for s in range(N_PER):
        v = gradient_train(model, tok, train, tl, DEVICE, seed=s*41)
        methods['Gradient'].append(v.cpu().numpy())
        accs['Gradient'].append(evaluate_vec(model, tok, v, all_data, tl, DEVICE))
    print(f"    mean acc={np.mean(accs['Gradient']):.2f}")

    print(f"  GA evolution ({N_PER} runs)...")
    for s in range(N_PER):
        v = ga_evolve(model, tok, all_data, tl, DEVICE, pop_size=15, n_gen=15, seed=s*53)
        methods['GA'].append(v.cpu().numpy())
        accs['GA'].append(evaluate_vec(model, tok, v, all_data, tl, DEVICE))
    print(f"    mean acc={np.mean(accs['GA']):.2f}")

    print(f"  Iterated learning ({N_PER} chains)...")
    for s in range(N_PER):
        v = iterated_learn(model, tok, train, tl, DEVICE, n_gen=4, seed=s*67)
        methods['Iterated'].append(v.cpu().numpy())
        accs['Iterated'].append(evaluate_vec(model, tok, v, all_data, tl, DEVICE))
    print(f"    mean acc={np.mean(accs['Iterated']):.2f}")

    print(f"  Random search ({N_PER} batches)...")
    for s in range(N_PER):
        v = random_search(model, tok, all_data, tl, DEVICE, n_tries=300, seed=s*79)
        methods['Random'].append(v.cpu().numpy())
        accs['Random'].append(evaluate_vec(model, tok, v, all_data, tl, DEVICE))
    print(f"    mean acc={np.mean(accs['Random']):.2f}")

    # Cross-method cosine analysis
    all_vecs = []
    all_labels = []
    for name, vs in methods.items():
        all_vecs.extend(vs); all_labels.extend([name]*len(vs))
    X = np.array(all_vecs)

    # Pairwise cosine between methods
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + 1e-8)
    cos_matrix = Xn @ Xn.T

    method_names = list(methods.keys())
    cross_cos = {}
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            idx1 = [k for k, l in enumerate(all_labels) if l == m1]
            idx2 = [k for k, l in enumerate(all_labels) if l == m2]
            vals = cos_matrix[np.ix_(idx1, idx2)]
            if i == j: vals = vals[np.triu_indices(len(idx1), k=1)]
            else: vals = vals.flatten()
            cross_cos[f'{m1}_vs_{m2}'] = round(float(np.mean(vals)), 4) if len(vals) > 0 else 0

    # t-SNE
    perp = min(10, N_PER*4 - 1)
    X_2d = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X)

    # PCA
    pca = PCA(n_components=min(10, N_PER*4))
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    output = {
        'phase': 116, 'name': 'convergent_evolution',
        'n_per_method': N_PER,
        'mean_accs': {k: round(float(np.mean(v)), 4) for k, v in accs.items()},
        'cross_method_cos': cross_cos,
        'pca_cumvar_5': round(float(cumvar[min(4, len(cumvar)-1)]), 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase116_convergent.json'), 'w') as f:
        json.dump(output, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_map = {'Gradient':'tab:blue','GA':'tab:orange','Iterated':'tab:green','Random':'tab:red'}
    markers_map = {'Gradient':'o','GA':'s','Iterated':'^','Random':'D'}
    for name in method_names:
        idx = [i for i, l in enumerate(all_labels) if l == name]
        axes[0].scatter(X_2d[idx,0], X_2d[idx,1], c=colors_map[name],
                       marker=markers_map[name], s=50, label=name,
                       edgecolors='black', alpha=0.7)
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
    axes[0].set_title('4 Methods in Soul Space', fontweight='bold')
    axes[0].legend(fontsize=9)

    means = [np.mean(accs[m]) for m in method_names]
    stds = [np.std(accs[m]) for m in method_names]
    bars = axes[1].bar(method_names, means, yerr=stds, capsize=5,
                       color=[colors_map[m] for m in method_names], edgecolor='black')
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.2)
    axes[1].set_title('Mean Accuracy by Method', fontweight='bold')
    for i, v in enumerate(means):
        axes[1].text(i, v+stds[i]+0.03, f'{v:.0%}', ha='center', fontsize=9)

    # Cross-method cosine heatmap
    cos_mat = np.zeros((4, 4))
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            k = f'{m1}_vs_{m2}'
            cos_mat[i,j] = cross_cos.get(k, 0)
    im = axes[2].imshow(cos_mat, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels(method_names, fontsize=8, rotation=30)
    axes[2].set_yticks(range(4)); axes[2].set_yticklabels(method_names, fontsize=8)
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f'{cos_mat[i,j]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=axes[2])
    axes[2].set_title('Cross-Method Cosine Similarity', fontweight='bold')

    plt.suptitle('Phase 116: Convergent Evolution\n"All roads lead to Rome"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase116_convergent.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
