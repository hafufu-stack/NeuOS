# -*- coding: utf-8 -*-
"""
Phase 93: The Neural Microbiome
Place "helper" vectors at intermediate layers (L12, L14) that act as
symbiotic microbiome. Test if main program + microbiome outperforms
the main program alone on complex tasks.

"No organism survives alone."

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

def compile_symbiosis(model, tok, train, main_layer, helper_layers, device,
                      seed=42, epochs=100):
    """Co-train main program + helper microbiome vectors."""
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
                    helper_vecs=None):
    c = 0
    for p, e in data:
        hooks = []
        def make_inj(v):
            def inj(m,i,o,vec=v): return replace_last_token(o,vec)
            return inj
        hooks.append(model.model.layers[main_layer].register_forward_hook(make_inj(main_vec)))
        if helper_vecs:
            for l, hv in helper_vecs.items():
                hooks.append(model.model.layers[l].register_forward_hook(make_inj(hv)))
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        for h in hooks: h.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
    return c / len(data)

def main():
    print("[P93] The Neural Microbiome")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    # Complex task: requires understanding both MIN and comparison
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("5, 4) =","4"),("3, 8) =","3")]
    all_data = min_data + test_data

    main_layer = 8
    microbiome_configs = [
        {'name': 'solo', 'helpers': []},
        {'name': 'L12', 'helpers': [12]},
        {'name': 'L12+L14', 'helpers': [12, 14]},
        {'name': 'L12+L14+L16', 'helpers': [12, 14, 16]},
        {'name': 'L10+L12+L14', 'helpers': [10, 12, 14]},
    ]

    results = []
    for config in microbiome_configs:
        name = config['name']
        helpers = config['helpers']
        print(f"\n  Config: {name} (main=L{main_layer}, helpers={helpers})")

        if helpers:
            main_v, helper_vs = compile_symbiosis(
                model, tok, min_data, main_layer, helpers, DEVICE, epochs=80)
            acc_full = evaluate_system(model, tok, main_v, main_layer, all_data,
                                       DEVICE, helper_vs)
            # Test: main alone (remove microbiome)
            acc_solo = evaluate_system(model, tok, main_v, main_layer, all_data,
                                       DEVICE, None)
            # Test: microbiome without main (remove host)
            if len(helpers) >= 1:
                first_helper = helpers[0]
                acc_microbe = evaluate_system(
                    model, tok, helper_vs[first_helper], first_helper,
                    all_data, DEVICE, None)
            else:
                acc_microbe = 0
        else:
            main_v = compile_prog(model, tok, min_data, main_layer, DEVICE, epochs=80)
            acc_full = evaluate_system(model, tok, main_v, main_layer, all_data,
                                       DEVICE, None)
            acc_solo = acc_full
            acc_microbe = 0

        r = {
            'config': name,
            'accuracy_full': round(float(acc_full), 4),
            'accuracy_solo': round(float(acc_solo), 4),
            'accuracy_microbe_only': round(float(acc_microbe), 4),
            'symbiosis_bonus': round(float(acc_full - acc_solo), 4),
        }
        results.append(r)
        print(f"    Full={acc_full:.0%}, Solo={acc_solo:.0%}, "
              f"Microbe={acc_microbe:.0%}, Bonus={acc_full-acc_solo:+.0%}")

    # Save
    output = {
        'phase': 93, 'name': 'neural_microbiome',
        'main_layer': main_layer,
        'results': results,
        'best_config': max(results, key=lambda x: x['accuracy_full'])['config'],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase93_microbiome.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = [r['config'] for r in results]
    x = np.arange(len(names))

    # Full system accuracy
    axes[0].bar(x, [r['accuracy_full'] for r in results],
                color='tab:green', edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylabel('Accuracy'); axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Full System Accuracy', fontweight='bold')
    for i, r in enumerate(results):
        axes[0].text(i, r['accuracy_full']+0.03, f'{r["accuracy_full"]:.0%}',
                     ha='center', fontweight='bold', fontsize=9)

    # Symbiosis bonus
    bonuses = [r['symbiosis_bonus'] for r in results]
    colors = ['tab:green' if b > 0 else 'tab:red' for b in bonuses]
    axes[1].bar(x, bonuses, color=colors, edgecolor='black')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8)
    axes[1].set_ylabel('Accuracy Bonus')
    axes[1].set_title('Symbiosis Bonus\n(Full - Solo)', fontweight='bold')
    axes[1].axhline(y=0, color='black', lw=1)

    # Decomposition: host vs microbiome vs synergy
    w = 0.25
    axes[2].bar(x-w, [r['accuracy_solo'] for r in results], w,
                label='Host alone', color='tab:blue', edgecolor='black')
    axes[2].bar(x, [r['accuracy_microbe_only'] for r in results], w,
                label='Microbiome alone', color='tab:orange', edgecolor='black')
    axes[2].bar(x+w, [r['accuracy_full'] for r in results], w,
                label='Symbiosis', color='tab:green', edgecolor='black')
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, fontsize=8)
    axes[2].set_ylabel('Accuracy'); axes[2].set_ylim(0, 1.2)
    axes[2].set_title('Host vs Microbiome vs Symbiosis', fontweight='bold')
    axes[2].legend(fontsize=8)

    plt.suptitle('Phase 93: The Neural Microbiome\n'
                 '"No organism survives alone"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase93_microbiome.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
