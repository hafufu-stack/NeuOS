# -*- coding: utf-8 -*-
"""
Phase 58: Neural Quine (Opus Original)
Compile a program that outputs a representation of itself.
Self-replication in activation space.

The quine test: inject vec at L8, read state at L22.
Does the output state encode information about the injected vec?
Can the model "reflect" on its own program?

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P58] Neural Quine")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    INJECT_L = 8
    READ_L = 22

    for p in model.parameters():
        p.requires_grad = False

    # Step 1: Compile a self-reflective program
    # Train: inject vec at L8, maximize cosine similarity between vec and L22 output
    print("  Step 1: Compiling quine (self-reflective program)...")
    prompt = "self) ="

    quine_vec = torch.randn(hidden_size, device=DEVICE) * 0.01
    quine_vec.requires_grad_(True)
    optimizer = torch.optim.Adam([quine_vec], lr=0.01)

    loss_history = []
    sim_history = []

    for epoch in range(200):
        output_state = [None]
        def inject_q(module, input, output, v=quine_vec):
            return replace_last_token(output, v)
        def capture_q(module, input, output):
            output_state[0] = get_last_token(output)

        h_inj = model.model.layers[INJECT_L].register_forward_hook(inject_q)
        h_cap = model.model.layers[READ_L].register_forward_hook(capture_q)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        out = model(**inp)
        h_inj.remove(); h_cap.remove()

        # Quine loss: maximize cosine similarity between input and output
        cos_sim = torch.nn.functional.cosine_similarity(
            quine_vec.unsqueeze(0), output_state[0].float())
        loss = 1.0 - cos_sim.mean()  # Minimize (1 - similarity)
        loss_history.append(loss.item())
        sim_history.append(cos_sim.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            print(f"    Epoch {epoch}: loss={loss.item():.4f}, "
                  f"self_sim={cos_sim.mean().item():.4f}")

    final_sim = sim_history[-1]
    print(f"    Final self-similarity: {final_sim:.4f}")

    # Step 2: Control - random vectors (no training)
    print("\n  Step 2: Control (random vectors)...")
    control_sims = []
    for i in range(10):
        rand_vec = torch.randn(hidden_size, device=DEVICE) * 0.01
        output_state = [None]
        def inject_r(module, input, output, v=rand_vec):
            return replace_last_token(output, v)
        def capture_r(module, input, output):
            output_state[0] = get_last_token(output)
        h1 = model.model.layers[INJECT_L].register_forward_hook(inject_r)
        h2 = model.model.layers[READ_L].register_forward_hook(capture_r)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h1.remove(); h2.remove()
        sim = torch.nn.functional.cosine_similarity(
            rand_vec.unsqueeze(0), output_state[0].float()).item()
        control_sims.append(sim)

    avg_control = np.mean(control_sims)
    print(f"    Avg control similarity: {avg_control:.4f}")

    # Step 3: Test quine on different prompts
    print("\n  Step 3: Testing quine portability...")
    quine_eval = quine_vec.detach()
    port_sims = []
    test_prompts = ["hello) =", "data) =", "run) =", "calc) =", "0) ="]
    for p in test_prompts:
        output_state = [None]
        def inject_p(module, input, output, v=quine_eval):
            return replace_last_token(output, v)
        def capture_p(module, input, output):
            output_state[0] = get_last_token(output)
        h1 = model.model.layers[INJECT_L].register_forward_hook(inject_p)
        h2 = model.model.layers[READ_L].register_forward_hook(capture_p)
        inp = tok(p, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h1.remove(); h2.remove()
        sim = torch.nn.functional.cosine_similarity(
            quine_eval.unsqueeze(0), output_state[0].float()).item()
        port_sims.append(sim)
        print(f"    '{p}': self_sim={sim:.4f}")

    avg_portable = np.mean(port_sims)

    # Save
    output = {
        'phase': 58, 'name': 'neural_quine',
        'final_self_sim': round(final_sim, 4),
        'avg_control_sim': round(float(avg_control), 4),
        'avg_portable_sim': round(float(avg_portable), 4),
        'quine_amplification': round(final_sim / (abs(avg_control) + 1e-8), 2),
        'loss_history': [round(l, 4) for l in loss_history[::10]],
        'sim_history': [round(s, 4) for s in sim_history[::10]],
        'portable_sims': [round(s, 4) for s in port_sims],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase58_quine.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(sim_history, 'g-', linewidth=2, label='Quine training')
    axes[0].axhline(y=avg_control, color='red', linestyle='--', label=f'Random baseline ({avg_control:.3f})')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Self-Similarity')
    axes[0].set_title('Quine Convergence', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(['Quine\n(trained)', 'Random\n(control)', 'Portable\n(new prompts)'],
                [final_sim, avg_control, avg_portable],
                color=['tab:green', 'tab:red', 'tab:purple'], edgecolor='black')
    axes[1].set_ylabel('Self-Similarity')
    axes[1].set_title('Quine vs Control', fontweight='bold')
    for i, v in enumerate([final_sim, avg_control, avg_portable]):
        axes[1].text(i, v+0.02 if v >= 0 else v-0.05, f'{v:.3f}',
                    ha='center', fontweight='bold')

    axes[2].bar(range(len(port_sims)), port_sims, color='tab:purple', edgecolor='black')
    axes[2].set_xticks(range(len(test_prompts)))
    axes[2].set_xticklabels([p[:6] for p in test_prompts], rotation=30)
    axes[2].set_ylabel('Self-Similarity')
    axes[2].set_title('Quine Portability', fontweight='bold')
    axes[2].axhline(y=avg_control, color='red', linestyle='--')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 58: Neural Quine\nA program that outputs a representation of itself',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase58_quine.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
