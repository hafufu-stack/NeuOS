# -*- coding: utf-8 -*-
"""
Phase 51: Program Polymorphism (Opus Original)
P46 showed programs are many-to-one (different vectors, same function).
Intentionally generate N diverse vectors for the same function.
Measure the "program genome" diversity.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_program(model, tok, train_data, target_layer, device, seed):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(80):
        for prompt, target_str in train_data:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def inject(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    # Test
    correct = 0
    vec_eval = vec.detach()
    for prompt, target_str in train_data:
        def inject_t(module, input, output, v=vec_eval):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_t)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == target_str:
            correct += 1
    return vec_eval, correct / len(train_data)


def main():
    print("[P51] Program Polymorphism")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    N_VARIANTS = 10
    programs = {
        'MIN': [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")],
        'MAX': [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")],
    }

    all_results = {}
    for prog_name, train_data in programs.items():
        print(f"\n  Compiling {N_VARIANTS} variants of {prog_name}...")
        variants = []
        accs = []
        for i in range(N_VARIANTS):
            vec, acc = compile_program(model, tok, train_data, target_layer, DEVICE, seed=i*100+42)
            variants.append(vec.cpu().numpy().flatten())
            accs.append(acc)
            if i < 3:
                print(f"    Variant {i}: acc={acc:.0%}")

        # Compute pairwise cosine similarities
        sim_matrix = cosine_similarity(np.array(variants))
        # Upper triangle only (exclude diagonal)
        triu = sim_matrix[np.triu_indices(N_VARIANTS, k=1)]
        avg_sim = float(np.mean(triu))
        std_sim = float(np.std(triu))
        min_sim = float(np.min(triu))
        max_sim = float(np.max(triu))
        avg_acc = float(np.mean(accs))

        all_results[prog_name] = {
            'avg_acc': round(avg_acc, 4),
            'avg_pairwise_sim': round(avg_sim, 4),
            'std_sim': round(std_sim, 4),
            'min_sim': round(min_sim, 4),
            'max_sim': round(max_sim, 4),
            'sim_matrix': sim_matrix.tolist(),
        }
        print(f"    {prog_name}: avg_acc={avg_acc:.0%}, "
              f"avg_sim={avg_sim:.3f} +/- {std_sim:.3f}, "
              f"range=[{min_sim:.3f}, {max_sim:.3f}]")

    # Save
    output = {
        'phase': 51, 'name': 'program_polymorphism',
        'n_variants': N_VARIANTS,
        'results': all_results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase51_polymorphism.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (prog_name, ax) in enumerate(zip(all_results, [axes[0], axes[1]])):
        r = all_results[prog_name]
        im = ax.imshow(np.array(r['sim_matrix']), cmap='RdYlGn', vmin=-0.5, vmax=1.0)
        ax.set_title(f'{prog_name} Variants\navg sim={r["avg_pairwise_sim"]:.3f}',
                    fontweight='bold')
        ax.set_xlabel('Variant'); ax.set_ylabel('Variant')
        plt.colorbar(im, ax=ax)

    # Summary
    names = list(all_results.keys())
    sims = [all_results[n]['avg_pairwise_sim'] for n in names]
    axes[2].bar(names, sims, color=['tab:blue', 'tab:orange'], edgecolor='black')
    axes[2].set_ylabel('Avg Pairwise Cosine Similarity')
    axes[2].set_title('Program Diversity\n(lower = more polymorphic)', fontweight='bold')
    axes[2].set_ylim(-0.5, 1.0)
    for i, v in enumerate(sims):
        axes[2].text(i, v+0.03, f'{v:.3f}', ha='center', fontweight='bold')

    plt.suptitle(f'Phase 51: Program Polymorphism\n{N_VARIANTS} different vectors for the same function',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase51_polymorphism.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
