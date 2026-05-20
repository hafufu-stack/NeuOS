# -*- coding: utf-8 -*-
"""
Phase 92: Cross-Embodiment Soul Migration
Train a soul on task A (MIN), SVD compress to 10-dim, then use as
seed for task B (MAX). Compare convergence speed vs tabula rasa.

"A soul that has known gravity learns to fly faster."

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

def compile_with_curve(model, tok, train, test, layer, device, init_vec=None,
                       seed=42, epochs=100, eval_every=10):
    """Compile a program and track learning curve."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    if init_vec is not None:
        vec = init_vec.clone().to(device)
    else:
        vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    curve = []
    for ep in range(epochs):
        for prompt, target_str in train:
            tid = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def inj(m,i,o,v=vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % eval_every == 0:
            c = 0
            for p, e in test:
                def inj2(m,i,o,v=vec): return replace_last_token(o,v)
                hk = model.model.layers[layer].register_forward_hook(inj2)
                inp2 = tok(p, return_tensors='pt').to(device)
                with torch.no_grad(): o2 = model(**inp2)
                hk.remove()
                if tok.decode(o2.logits[0,-1,:].argmax().item()).strip() == e: c += 1
            curve.append((ep+1, round(c/len(test), 4)))
    return vec.detach(), curve

def main():
    print("[P92] Cross-Embodiment Soul Migration")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    # Tasks as "embodiments"
    tasks = {
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
    }

    # Step 1: Build SVD soul from MIN variants
    print("  Step 1: Building MIN soul basis...")
    min_vecs = []
    for s in range(10):
        v, _ = compile_with_curve(model, tok, tasks['MIN']['train'],
                                   tasks['MIN']['test'], tl, DEVICE,
                                   seed=s*100, epochs=80, eval_every=80)
        min_vecs.append(v.cpu().numpy().flatten())
    Vt = np.linalg.svd(np.array(min_vecs), full_matrices=False)[2]
    k_soul = 10
    Vk = Vt[:k_soul, :]

    # Extract the "soul" of a trained MIN program
    trained_min, _ = compile_with_curve(model, tok, tasks['MIN']['train'],
                                         tasks['MIN']['test'], tl, DEVICE,
                                         seed=42, epochs=100, eval_every=100)
    min_np = trained_min.cpu().numpy().flatten()
    soul = (min_np @ Vk.T) @ Vk  # 10-dim soul reconstructed to 896-dim
    soul_vec = torch.tensor(soul, device=DEVICE, dtype=torch.float32)

    # Step 2: Migration test - use MIN soul as seed for other tasks
    print("\n  Step 2: Soul migration tests...")
    results = {}

    for target_name in ['MAX', 'FIRST']:
        print(f"\n    --- Migrating MIN soul -> {target_name} ---")
        td = tasks[target_name]

        # Tabula rasa (random init)
        print(f"    Tabula rasa...")
        _, curve_blank = compile_with_curve(
            model, tok, td['train'], td['test'], tl, DEVICE,
            seed=42, epochs=100, eval_every=5)

        # Soul-seeded (MIN soul as init)
        print(f"    Soul-seeded...")
        _, curve_soul = compile_with_curve(
            model, tok, td['train'], td['test'], tl, DEVICE,
            init_vec=soul_vec, epochs=100, eval_every=5)

        results[target_name] = {
            'tabula_rasa': curve_blank,
            'soul_seeded': curve_soul,
            'final_blank': curve_blank[-1][1] if curve_blank else 0,
            'final_soul': curve_soul[-1][1] if curve_soul else 0,
        }

        # Find epoch where each reaches 60%
        def find_threshold(curve, t=0.6):
            for ep, acc in curve:
                if acc >= t: return ep
            return None
        ep_blank = find_threshold(curve_blank)
        ep_soul = find_threshold(curve_soul)
        speedup = (ep_blank / ep_soul) if (ep_blank and ep_soul) else None
        results[target_name]['epoch_60_blank'] = ep_blank
        results[target_name]['epoch_60_soul'] = ep_soul
        results[target_name]['speedup'] = round(float(speedup), 2) if speedup else None

        print(f"    Final: blank={results[target_name]['final_blank']:.0%}, "
              f"soul={results[target_name]['final_soul']:.0%}")
        if speedup:
            print(f"    60% threshold: blank=ep{ep_blank}, soul=ep{ep_soul} "
                  f"-> {speedup:.1f}x speedup")

    # Save
    output = {
        'phase': 92, 'name': 'cross_embodiment_soul_migration',
        'soul_dims': k_soul, 'source_task': 'MIN',
        'results': {k: {kk: vv for kk, vv in v.items()
                        if kk not in ('tabula_rasa', 'soul_seeded')}
                    for k, v in results.items()},
        'learning_curves': {k: {'tabula_rasa': v['tabula_rasa'],
                                'soul_seeded': v['soul_seeded']}
                            for k, v in results.items()},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase92_soul_migration.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, target in enumerate(['MAX', 'FIRST']):
        r = results[target]
        if r['tabula_rasa']:
            eps_b, accs_b = zip(*r['tabula_rasa'])
            axes[idx].plot(eps_b, accs_b, 'r-o', lw=2, ms=4, label='Tabula Rasa')
        if r['soul_seeded']:
            eps_s, accs_s = zip(*r['soul_seeded'])
            axes[idx].plot(eps_s, accs_s, 'g-s', lw=2, ms=4, label='MIN Soul Seed')
        axes[idx].set_xlabel('Epoch'); axes[idx].set_ylabel('Test Accuracy')
        axes[idx].set_title(f'MIN -> {target}\nSoul Migration', fontweight='bold')
        axes[idx].legend(); axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-0.05, 1.1)

    # Summary bar
    labels = [f'{t}\nBlank' for t in ['MAX','FIRST']] + \
             [f'{t}\nSoul' for t in ['MAX','FIRST']]
    vals = ([results[t]['final_blank'] for t in ['MAX','FIRST']] +
            [results[t]['final_soul'] for t in ['MAX','FIRST']])
    colors = ['tab:red','tab:red','tab:green','tab:green']
    axes[2].bar(range(len(labels)), vals, color=colors, edgecolor='black')
    axes[2].set_xticks(range(len(labels))); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('Final Accuracy'); axes[2].set_ylim(0, 1.2)
    axes[2].set_title('Soul Transfer Advantage', fontweight='bold')
    for i, v in enumerate(vals):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=10)

    plt.suptitle('Phase 92: Cross-Embodiment Soul Migration\n'
                 '"A soul that has known gravity learns to fly faster"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase92_soul_migration.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
