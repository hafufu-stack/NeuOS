# -*- coding: utf-8 -*-
"""
Phase 105: Soul Speciation (Adaptive Radiation)
Start from ONE ancestor soul. Place copies in different "environments"
(different tasks). After independent evolution, measure how much the
descendants have diverged. Does specialization destroy shared structure?

"From one ancestor, many species."

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

def specialize(model, tok, init_vec, train, layer, device, epochs=100):
    """Continue training from an existing vector (fine-tune)."""
    vec = init_vec.clone().detach().requires_grad_(True)
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

def main():
    print("[P105] Soul Speciation (Adaptive Radiation)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    environments = {
        'MIN': {'train': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                          ("4, 6) =","4"),("9, 3) =","3")],
                'test': [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                         ("5, 4) =","4"),("3, 8) =","3")]},
        'MAX': {'train': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                          ("4, 6) =","6"),("9, 3) =","9")],
                'test': [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                         ("5, 4) =","5"),("3, 8) =","8")]},
        'FIRST': {'train': [("3, 7) =","3"),("5, 2) =","5"),("8, 1) =","8"),
                            ("4, 6) =","4"),("9, 3) =","9")],
                  'test': [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","2"),
                           ("5, 4) =","5"),("3, 8) =","3")]},
        'LAST': {'train': [("3, 7) =","7"),("5, 2) =","2"),("8, 1) =","1"),
                           ("4, 6) =","6"),("9, 3) =","3")],
                 'test': [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","9"),
                          ("5, 4) =","4"),("3, 8) =","8")]},
    }

    # Step 1: Create common ancestor
    print("  Step 1: Creating common ancestor...")
    ancestor = compile_prog(model, tok, environments['MIN']['train'],
                           tl, DEVICE, seed=42, epochs=50)
    print(f"    Ancestor norm: {ancestor.norm():.3f}")

    # Step 2: Speciation (place ancestor in each environment)
    print("\n  Step 2: Speciation (adapting to environments)...")
    species = {}
    epoch_snapshots = {env: [] for env in environments}

    for env_name, env_data in environments.items():
        # Take snapshots during specialization
        vec = ancestor.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([vec], lr=0.01)
        for ep in range(100):
            for p, t in env_data['train']:
                tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(DEVICE)
                def inj(m,i,o,v=vec): return replace_last_token(o,v)
                h = model.model.layers[tl].register_forward_hook(inj)
                out = model(**inp); h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()

            if (ep+1) % 10 == 0:
                v_snap = vec.detach().clone()
                cos_to_ancestor = float(torch.nn.functional.cosine_similarity(
                    v_snap.unsqueeze(0), ancestor.unsqueeze(0)).item())
                acc = evaluate_vec(model, tok, v_snap,
                                 env_data['train'] + env_data['test'], tl, DEVICE)
                epoch_snapshots[env_name].append({
                    'epoch': ep+1,
                    'cos_to_ancestor': round(cos_to_ancestor, 4),
                    'accuracy': round(float(acc), 4),
                })

        species[env_name] = vec.detach()
        final_acc = evaluate_vec(model, tok, species[env_name],
                                env_data['train'] + env_data['test'], tl, DEVICE)
        cos_anc = float(torch.nn.functional.cosine_similarity(
            species[env_name].unsqueeze(0), ancestor.unsqueeze(0)).item())
        print(f"    {env_name}: acc={final_acc:.0%}, cos(ancestor)={cos_anc:.3f}")

    # Step 3: Phylogenetic distance matrix
    print("\n  Step 3: Phylogenetic distances...")
    env_names = list(environments.keys())
    dist_matrix = np.zeros((4, 4))
    cos_matrix = {}
    for i, n1 in enumerate(env_names):
        for j, n2 in enumerate(env_names):
            cos_sim = float(torch.nn.functional.cosine_similarity(
                species[n1].unsqueeze(0), species[n2].unsqueeze(0)).item())
            dist_matrix[i, j] = 1 - cos_sim
            cos_matrix[f'{n1}-{n2}'] = round(cos_sim, 4)

    print(f"    Distance matrix (1 - cosine):")
    for i, n in enumerate(env_names):
        dists = [f"{dist_matrix[i,j]:.3f}" for j in range(4)]
        print(f"      {n}: {', '.join(dists)}")

    # Step 4: Cross-environment performance
    print("\n  Step 4: Cross-environment generalization...")
    cross_perf = {}
    for sp_name in env_names:
        cross_perf[sp_name] = {}
        for env_name, env_data in environments.items():
            acc = evaluate_vec(model, tok, species[sp_name],
                             env_data['train'] + env_data['test'], tl, DEVICE)
            cross_perf[sp_name][env_name] = round(float(acc), 4)
    print("    Cross-performance matrix:")
    header = "         " + "  ".join(f"{n:>6}" for n in env_names)
    print(f"    {header}")
    for sp in env_names:
        vals = "  ".join(f"{cross_perf[sp][e]:>6.0%}" for e in env_names)
        print(f"    {sp:>6}:  {vals}")

    # Save
    output = {
        'phase': 105, 'name': 'soul_speciation',
        'cosine_matrix': cos_matrix,
        'cross_performance': cross_perf,
        'divergence_history': {k: v for k, v in epoch_snapshots.items()},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase105_speciation.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Divergence over time
    for env_name in env_names:
        snaps = epoch_snapshots[env_name]
        eps = [s['epoch'] for s in snaps]
        cos_vals = [s['cos_to_ancestor'] for s in snaps]
        axes[0].plot(eps, cos_vals, '-o', lw=2, ms=4, label=env_name)
    axes[0].set_xlabel('Specialization Epochs')
    axes[0].set_ylabel('Cosine to Ancestor')
    axes[0].set_title('Divergence from Ancestor', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Phylogenetic distance heatmap
    im = axes[1].imshow(dist_matrix, cmap='YlOrRd', vmin=0, vmax=1.5)
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels(env_names, fontsize=9)
    axes[1].set_yticks(range(4)); axes[1].set_yticklabels(env_names, fontsize=9)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f'{dist_matrix[i,j]:.2f}', ha='center', va='center',
                        fontweight='bold', fontsize=10)
    axes[1].set_title('Phylogenetic Distance\n(1 - cosine)', fontweight='bold')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # Cross-environment performance
    x = np.arange(4); w = 0.2
    for i, sp in enumerate(env_names):
        vals = [cross_perf[sp][e] for e in env_names]
        axes[2].bar(x + i*w, vals, w, label=f'{sp} species', edgecolor='black')
    axes[2].set_xticks(x + 1.5*w)
    axes[2].set_xticklabels(env_names, fontsize=9)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Cross-Environment\nGeneralization', fontweight='bold')
    axes[2].legend(fontsize=7, ncol=2); axes[2].set_ylim(0, 1.2)

    plt.suptitle('Phase 105: Soul Speciation\n"From one ancestor, many species"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase105_speciation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
