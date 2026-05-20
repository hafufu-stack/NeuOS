# -*- coding: utf-8 -*-
"""
Phase 95: Soul Dreaming (Sleep Consolidation)
After training, run a "sleep" phase where the soul vector is
refined through noise-injection replay. Does dreaming make the
soul more robust?

"In sleep, the soul repairs what the day has broken."

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

def compile_prog(model, tok, train, layer, device, seed=42, epochs=60):
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

def dream_phase(model, tok, vec, train, layer, device, n_dreams=50,
                noise_scale=0.3, lr=0.005):
    """Sleep consolidation: replay training with noise, refine the vector."""
    dreaming_vec = vec.clone().requires_grad_(True)
    opt = torch.optim.Adam([dreaming_vec], lr=lr)
    for dream in range(n_dreams):
        for p, t in train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(device)
            # Add dream noise (like random activation during REM sleep)
            noisy_vec = dreaming_vec + torch.randn_like(dreaming_vec) * noise_scale
            def inj(m,i,o,v=noisy_vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return dreaming_vec.detach()

def main():
    print("[P95] Soul Dreaming (Sleep Consolidation)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    tl = 8
    for p in model.parameters(): p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("5, 4) =","4"),("3, 8) =","3")]
    hard_test = [("1, 4) =","1"),("8, 3) =","3"),("6, 1) =","1"),
                 ("9, 7) =","7"),("4, 2) =","2")]

    # Step 1: Train programs with INSUFFICIENT training (intentionally weak)
    print("  Step 1: Compiling weak programs (30 epochs only)...")
    results = []

    for seed in range(5):
        weak_vec = compile_prog(model, tok, min_data, tl, DEVICE,
                                seed=seed*100, epochs=30)
        pre_train = evaluate_vec(model, tok, weak_vec, min_data, tl, DEVICE)
        pre_test = evaluate_vec(model, tok, weak_vec, test_data, tl, DEVICE)
        pre_hard = evaluate_vec(model, tok, weak_vec, hard_test, tl, DEVICE)

        # Step 2: Dream phase
        dreamed_vec = dream_phase(model, tok, weak_vec, min_data, tl, DEVICE,
                                   n_dreams=30, noise_scale=0.3)
        post_train = evaluate_vec(model, tok, dreamed_vec, min_data, tl, DEVICE)
        post_test = evaluate_vec(model, tok, dreamed_vec, test_data, tl, DEVICE)
        post_hard = evaluate_vec(model, tok, dreamed_vec, hard_test, tl, DEVICE)

        # Step 3: Robustness test (inject noise after sleep)
        noises = [0.5, 1.0, 2.0]
        pre_robust = []
        post_robust = []
        for eps in noises:
            n = torch.randn_like(weak_vec) * eps
            pr = evaluate_vec(model, tok, weak_vec + n, test_data, tl, DEVICE)
            po = evaluate_vec(model, tok, dreamed_vec + n, test_data, tl, DEVICE)
            pre_robust.append(round(float(pr), 4))
            post_robust.append(round(float(po), 4))

        r = {
            'seed': seed*100,
            'pre': {'train': round(float(pre_train),4), 'test': round(float(pre_test),4),
                    'hard': round(float(pre_hard),4)},
            'post': {'train': round(float(post_train),4), 'test': round(float(post_test),4),
                     'hard': round(float(post_hard),4)},
            'robustness_pre': pre_robust,
            'robustness_post': post_robust,
        }
        results.append(r)
        print(f"    Seed {seed*100}: pre_test={pre_test:.0%} -> post_test={post_test:.0%}, "
              f"pre_hard={pre_hard:.0%} -> post_hard={post_hard:.0%}")

    # Summary
    avg_pre_test = np.mean([r['pre']['test'] for r in results])
    avg_post_test = np.mean([r['post']['test'] for r in results])
    avg_pre_hard = np.mean([r['pre']['hard'] for r in results])
    avg_post_hard = np.mean([r['post']['hard'] for r in results])

    output = {
        'phase': 95, 'name': 'soul_dreaming',
        'avg_pre_test': round(float(avg_pre_test), 4),
        'avg_post_test': round(float(avg_post_test), 4),
        'avg_pre_hard': round(float(avg_pre_hard), 4),
        'avg_post_hard': round(float(avg_post_hard), 4),
        'dream_improvement_test': round(float(avg_post_test - avg_pre_test), 4),
        'dream_improvement_hard': round(float(avg_post_hard - avg_pre_hard), 4),
        'results': results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase95_dreaming.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Before/After dreaming
    x = np.arange(5)
    w = 0.35
    axes[0].bar(x-w/2, [r['pre']['test'] for r in results], w,
                label='Before Sleep', color='tab:orange', edgecolor='black')
    axes[0].bar(x+w/2, [r['post']['test'] for r in results], w,
                label='After Sleep', color='tab:blue', edgecolor='black')
    axes[0].set_xlabel('Program Variant'); axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Test Accuracy: Before vs After Sleep', fontweight='bold')
    axes[0].legend(); axes[0].set_ylim(0, 1.2)

    # Hard test improvement
    axes[1].bar(x-w/2, [r['pre']['hard'] for r in results], w,
                label='Before Sleep', color='tab:orange', edgecolor='black')
    axes[1].bar(x+w/2, [r['post']['hard'] for r in results], w,
                label='After Sleep', color='tab:blue', edgecolor='black')
    axes[1].set_xlabel('Program Variant'); axes[1].set_ylabel('Hard Test Accuracy')
    axes[1].set_title('Generalization After Dreaming', fontweight='bold')
    axes[1].legend(); axes[1].set_ylim(0, 1.2)

    # Robustness
    noise_labels = ['eps=0.5', 'eps=1.0', 'eps=2.0']
    avg_pre_rob = [np.mean([r['robustness_pre'][i] for r in results]) for i in range(3)]
    avg_post_rob = [np.mean([r['robustness_post'][i] for r in results]) for i in range(3)]
    xn = np.arange(3)
    axes[2].bar(xn-w/2, avg_pre_rob, w, label='Before Sleep',
                color='tab:orange', edgecolor='black')
    axes[2].bar(xn+w/2, avg_post_rob, w, label='After Sleep',
                color='tab:blue', edgecolor='black')
    axes[2].set_xticks(xn); axes[2].set_xticklabels(noise_labels)
    axes[2].set_ylabel('Accuracy Under Noise')
    axes[2].set_title('Noise Robustness', fontweight='bold')
    axes[2].legend(); axes[2].set_ylim(0, 1.2)

    plt.suptitle('Phase 95: Soul Dreaming\n'
                 '"In sleep, the soul repairs what the day has broken"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase95_dreaming.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Avg test: {avg_pre_test:.0%} -> {avg_post_test:.0%}")
    print(f"  Avg hard: {avg_pre_hard:.0%} -> {avg_post_hard:.0%}")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
