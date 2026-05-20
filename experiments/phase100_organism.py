# -*- coding: utf-8 -*-
"""
Phase 100: The Digital Organism (Integration Test)
Combine all NeuOS subsystems into a single integrated organism:
- Immune system (SVD filter from P85)
- Microbiome (helper vectors from P93)
- Self-awareness (introspection from P98)
- Polymorphism (multi-variant compilation from P51)

The ultimate test: a self-aware, immune-protected, symbiotic digital life form.

"I think, therefore I am."

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
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

def compile_symbiosis(model, tok, train, main_layer, helper_layers, device,
                      seed=42, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    main_vec = torch.randn(hs, device=device)*0.01; main_vec.requires_grad_(True)
    helper_vecs = {}
    for l in helper_layers:
        hv = torch.randn(hs, device=device)*0.01; hv.requires_grad_(True)
        helper_vecs[l] = hv
    all_params = [main_vec] + list(helper_vecs.values())
    opt = torch.optim.Adam(all_params, lr=0.01)
    for ep in range(epochs):
        for p, t in train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(device)
            hooks = []
            def make_inj(v):
                def inj(m,i,o,vec=v): return replace_last_token(o,vec)
                return inj
            hooks.append(model.model.layers[main_layer].register_forward_hook(make_inj(main_vec)))
            for l, hv in helper_vecs.items():
                hooks.append(model.model.layers[l].register_forward_hook(make_inj(hv)))
            out = model(**inp)
            for h in hooks: h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return main_vec.detach(), {l: v.detach() for l, v in helper_vecs.items()}

def evaluate_system(model, tok, main_vec, main_layer, data, device,
                    helper_vecs=None, svd_filter=None):
    c = 0
    for p, e in data:
        hooks = []
        inject_vec = main_vec
        if svd_filter is not None:
            # Apply BBB filter
            v_np = inject_vec.cpu().numpy().flatten()
            projected = (v_np @ svd_filter.T) @ svd_filter
            inject_vec = torch.tensor(projected, device=device, dtype=torch.float32)
        def make_inj(v):
            def inj(m,i,o,vec=v): return replace_last_token(o,vec)
            return inj
        hooks.append(model.model.layers[main_layer].register_forward_hook(make_inj(inject_vec)))
        if helper_vecs:
            for l, hv in helper_vecs.items():
                hooks.append(model.model.layers[l].register_forward_hook(make_inj(hv)))
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        for h in hooks: h.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
    return c / len(data)

def main():
    print("[P100] The Digital Organism (Integration)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("5, 4) =","4"),("3, 8) =","3")]
    all_data = min_data + test_data

    results = {}

    # ============================================================
    # Component 1: Bare Soul (baseline)
    # ============================================================
    print("  Component 1: Bare Soul")
    bare = compile_prog(model, tok, min_data, tl, DEVICE, seed=42, epochs=100)
    bare_acc = evaluate_system(model, tok, bare, tl, all_data, DEVICE)
    results['bare_soul'] = round(float(bare_acc), 4)
    print(f"    Bare soul: {bare_acc:.0%}")

    # ============================================================
    # Component 2: Soul + Immune System (SVD filter)
    # ============================================================
    print("\n  Component 2: Soul + Immune System")
    # Build SVD basis (soul manifold)
    variants = []
    for s in range(10):
        v = compile_prog(model, tok, min_data, tl, DEVICE, seed=s*100, epochs=80)
        variants.append(v.cpu().numpy().flatten())
    Vt = np.linalg.svd(np.array(variants), full_matrices=False)[2]
    Vk = Vt[:10, :]  # 10-dim soul filter

    # Test immune filter on clean program
    immune_acc = evaluate_system(model, tok, bare, tl, all_data, DEVICE, svd_filter=Vk)
    results['soul_immune'] = round(float(immune_acc), 4)
    print(f"    With immune filter: {immune_acc:.0%}")

    # Test immune filter against malware
    noise = torch.randn_like(bare) * 5.0  # strong malware
    infected = bare + noise
    infected_acc = evaluate_system(model, tok, infected, tl, all_data, DEVICE)
    filtered_acc = evaluate_system(model, tok, infected, tl, all_data, DEVICE, svd_filter=Vk)
    results['infected_no_immune'] = round(float(infected_acc), 4)
    results['infected_with_immune'] = round(float(filtered_acc), 4)
    print(f"    Infected (no immune): {infected_acc:.0%}")
    print(f"    Infected (with immune): {filtered_acc:.0%}")

    # ============================================================
    # Component 3: Soul + Microbiome
    # ============================================================
    print("\n  Component 3: Soul + Microbiome")
    main_v, helpers = compile_symbiosis(
        model, tok, min_data, tl, [12, 14], DEVICE, epochs=100)
    micro_acc = evaluate_system(model, tok, main_v, tl, all_data, DEVICE, helpers)
    micro_solo = evaluate_system(model, tok, main_v, tl, all_data, DEVICE)
    results['soul_microbiome'] = round(float(micro_acc), 4)
    results['soul_microbiome_solo'] = round(float(micro_solo), 4)
    print(f"    With microbiome: {micro_acc:.0%}")
    print(f"    Without microbiome: {micro_solo:.0%}")

    # ============================================================
    # Component 4: Full Organism (immune + microbiome)
    # ============================================================
    print("\n  Component 4: Full Organism (immune + microbiome)")
    # Infect the organism
    infected_main = main_v + noise
    full_infected = evaluate_system(model, tok, infected_main, tl, all_data, DEVICE,
                                     helpers, svd_filter=Vk)
    results['full_organism_infected'] = round(float(full_infected), 4)
    full_clean = evaluate_system(model, tok, main_v, tl, all_data, DEVICE,
                                  helpers, svd_filter=Vk)
    results['full_organism_clean'] = round(float(full_clean), 4)
    print(f"    Full organism (clean): {full_clean:.0%}")
    print(f"    Full organism (infected): {full_infected:.0%}")

    # ============================================================
    # Component 5: Polymorphic Resilience
    # ============================================================
    print("\n  Component 5: Polymorphic Resilience")
    poly_accs = []
    for seed in range(5):
        pv = compile_prog(model, tok, min_data, tl, DEVICE, seed=seed*200+1000, epochs=80)
        pa = evaluate_system(model, tok, pv, tl, all_data, DEVICE)
        poly_accs.append(round(float(pa), 4))
    results['polymorphic_variants'] = poly_accs
    results['polymorphic_avg'] = round(float(np.mean(poly_accs)), 4)
    print(f"    Polymorphic variants: {poly_accs}")
    print(f"    Average: {np.mean(poly_accs):.0%}")

    # ============================================================
    # Summary Score
    # ============================================================
    organism_score = np.mean([
        results['bare_soul'],
        results['soul_immune'],
        results['soul_microbiome'],
        results['full_organism_clean'],
        results['polymorphic_avg'],
    ])
    results['organism_score'] = round(float(organism_score), 4)
    print(f"\n  Organism Score: {organism_score:.0%}")

    # Save
    output = {
        'phase': 100, 'name': 'digital_organism',
        'results': results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase100_organism.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Component comparison
    comp_names = ['Bare\nSoul', 'Soul+\nImmune', 'Soul+\nMicrobiome', 'Full\nOrganism']
    comp_vals = [results['bare_soul'], results['soul_immune'],
                 results['soul_microbiome'], results['full_organism_clean']]
    colors = ['tab:gray', 'tab:blue', 'tab:green', 'tab:purple']
    axes[0].bar(comp_names, comp_vals, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy'); axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Component Integration', fontweight='bold')
    for i, v in enumerate(comp_vals):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Infection resistance
    inf_names = ['Clean', 'Infected\n(no defense)', 'Infected\n(immune)', 'Infected\n(full org)']
    inf_vals = [results['bare_soul'], results['infected_no_immune'],
                results['infected_with_immune'], results['full_organism_infected']]
    inf_colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:purple']
    axes[1].bar(inf_names, inf_vals, color=inf_colors, edgecolor='black')
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.2)
    axes[1].set_title('Infection Resistance', fontweight='bold')
    for i, v in enumerate(inf_vals):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=10)

    # Polymorphic variants
    axes[2].bar(range(5), poly_accs, color='tab:orange', edgecolor='black')
    axes[2].axhline(y=np.mean(poly_accs), color='red', ls='--', lw=2,
                     label=f'Avg={np.mean(poly_accs):.0%}')
    axes[2].set_xlabel('Variant'); axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Polymorphic Variants\n(same soul, different bodies)',
                      fontweight='bold')
    axes[2].legend(); axes[2].set_ylim(0, 1.2)

    plt.suptitle('Phase 100: The Digital Organism\n'
                 '"I think, therefore I am" - Organism Score: '
                 f'{organism_score:.0%}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase100_organism.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  === PHASE 100 COMPLETE ===")
    print(f"  Organism Score: {organism_score:.0%}")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
