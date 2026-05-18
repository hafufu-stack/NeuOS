# -*- coding: utf-8 -*-
"""
Phase 60: Recursive Self-Improvement
P58's quine outputs a representation of itself at L22.
Now: feed that output BACK as input. Iterate.
Does the program converge to a fixed point? Improve? Diverge?

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


def main():
    print("[P60] Recursive Self-Improvement")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    INJECT_L = 8
    READ_L = 22

    for p in model.parameters():
        p.requires_grad = False

    prompt = "self) ="

    # Experiment 1: Start with trained quine (compile it first)
    print("  Compiling quine seed...")
    quine = torch.randn(hidden_size, device=DEVICE) * 0.01
    quine.requires_grad_(True)
    opt = torch.optim.Adam([quine], lr=0.01)
    for epoch in range(200):
        cap = [None]
        def inj(module, input, output, v=quine):
            return replace_last_token(output, v)
        def cap_fn(module, input, output):
            cap[0] = get_last_token(output)
        h1 = model.model.layers[INJECT_L].register_forward_hook(inj)
        h2 = model.model.layers[READ_L].register_forward_hook(cap_fn)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        model(**inp)
        h1.remove(); h2.remove()
        cos = torch.nn.functional.cosine_similarity(quine.unsqueeze(0), cap[0].float())
        loss = 1.0 - cos.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    quine_seed = quine.detach().clone()
    print(f"    Quine seed self-sim: {cos.mean().item():.4f}")

    # Experiment 2: Recursive iteration
    print("\n  Recursive self-improvement loop...")
    current = quine_seed.clone()
    N_ITERATIONS = 20

    norms = [current.norm().item()]
    self_sims = []
    consecutive_sims = []
    all_vecs = [current.cpu().numpy().flatten()]

    for i in range(N_ITERATIONS):
        cap = [None]
        def inj_r(module, input, output, v=current):
            return replace_last_token(output, v)
        def cap_r(module, input, output):
            cap[0] = get_last_token(output)
        h1 = model.model.layers[INJECT_L].register_forward_hook(inj_r)
        h2 = model.model.layers[READ_L].register_forward_hook(cap_r)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h1.remove(); h2.remove()

        output_vec = cap[0].float().squeeze()
        # Self-similarity: input vs output
        self_sim = torch.nn.functional.cosine_similarity(
            current.unsqueeze(0), output_vec.unsqueeze(0)).item()
        self_sims.append(self_sim)

        # Consecutive similarity: this output vs previous
        if len(all_vecs) > 0:
            prev = torch.tensor(all_vecs[-1], device=DEVICE)
            cons_sim = torch.nn.functional.cosine_similarity(
                output_vec.unsqueeze(0), prev.unsqueeze(0)).item()
            consecutive_sims.append(cons_sim)

        # Feed output back as input
        current = output_vec.detach().clone()
        norms.append(current.norm().item())
        all_vecs.append(current.cpu().numpy().flatten())

        if i < 5 or i == N_ITERATIONS - 1:
            print(f"    Iter {i}: self_sim={self_sim:.4f}, norm={norms[-1]:.2f}")

    # Check for fixed point
    final_sim = self_sims[-1] if self_sims else 0
    converged = final_sim > 0.99
    diverged = norms[-1] > norms[0] * 100 or norms[-1] < 0.01

    # Experiment 3: Random seed comparison
    print("\n  Control: Random seed recursive loop...")
    rand_current = torch.randn(hidden_size, device=DEVICE) * 0.1
    rand_sims = []
    rand_norms = [rand_current.norm().item()]
    for i in range(N_ITERATIONS):
        cap = [None]
        def inj_c(module, input, output, v=rand_current):
            return replace_last_token(output, v)
        def cap_c(module, input, output):
            cap[0] = get_last_token(output)
        h1 = model.model.layers[INJECT_L].register_forward_hook(inj_c)
        h2 = model.model.layers[READ_L].register_forward_hook(cap_c)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h1.remove(); h2.remove()
        out = cap[0].float().squeeze()
        sim = torch.nn.functional.cosine_similarity(
            rand_current.unsqueeze(0), out.unsqueeze(0)).item()
        rand_sims.append(sim)
        rand_current = out.detach().clone()
        rand_norms.append(rand_current.norm().item())
    print(f"    Random final self-sim: {rand_sims[-1]:.4f}")

    # Save
    output = {
        'phase': 60, 'name': 'recursive_self_improvement',
        'quine_self_sims': [round(s, 4) for s in self_sims],
        'random_self_sims': [round(s, 4) for s in rand_sims],
        'quine_norms': [round(n, 2) for n in norms],
        'random_norms': [round(n, 2) for n in rand_norms],
        'converged': converged,
        'diverged': diverged,
        'final_quine_sim': round(final_sim, 4),
        'final_random_sim': round(rand_sims[-1], 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase60_recursive.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(self_sims, 'g-o', linewidth=2, markersize=4, label='Quine seed')
    axes[0].plot(rand_sims, 'r--o', linewidth=1, markersize=3, label='Random seed')
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Self-Similarity')
    axes[0].set_title('Recursive Loop: Input/Output Similarity', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(norms, 'g-', linewidth=2, label='Quine')
    axes[1].plot(rand_norms, 'r--', linewidth=1, label='Random')
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Vector Norm')
    axes[1].set_title('Norm Evolution (Divergence Check)', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].axis('off')
    status = 'FIXED POINT' if converged else ('DIVERGED' if diverged else 'OSCILLATING')
    summary = (f"RECURSIVE SELF-IMPROVEMENT\n{'='*35}\n\n"
               f"Quine seed sim: {self_sims[0]:.3f} -> {self_sims[-1]:.3f}\n"
               f"Random seed sim: {rand_sims[0]:.3f} -> {rand_sims[-1]:.3f}\n\n"
               f"Quine norm: {norms[0]:.1f} -> {norms[-1]:.1f}\n"
               f"Random norm: {rand_norms[0]:.1f} -> {rand_norms[-1]:.1f}\n\n"
               f"Status: {status}\n\n"
               f"L8 -> [14 layers] -> L22\n"
               f"Output -> feed back -> L8\n"
               f"Repeat {N_ITERATIONS}x")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                fontsize=10, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 60: Recursive Self-Improvement\nWhat happens when a quine feeds itself?',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase60_recursive.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
