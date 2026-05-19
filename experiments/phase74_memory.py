# -*- coding: utf-8 -*-
"""
Phase 74: Memory Consolidation
Dream patterns (P62) are repeatedly re-injected to convert
short-term learning into long-term memory.
Sleep -> Wake -> Test cycle mimics biological memory.

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


def compile_prog(model, tok, train, layer, device, seed=42, epochs=80):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(epochs):
        for prompt, target_str in train:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def inject(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def eval_prog(model, tok, vec, prompts, expected, layer, device):
    correct = 0
    for prompt, exp in zip(prompts, expected):
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == exp:
            correct += 1
    return correct / len(prompts)


def dream_replay(model, tok, vec, layer, device, n_replays=3):
    """Simulate dream: inject vec, read activation, average into memory."""
    prompts = ["Calculate", "The result of", "Numbers are"]
    accumulated = torch.zeros_like(vec)

    for prompt in prompts:
        # Inject and read what comes out
        def inject_fn(module, input, output, v=vec):
            return replace_last_token(output, v)
        read_layer = min(layer + 4, 23)
        cap = {}
        def read_fn(module, input, output):
            cap['act'] = get_last_token(output).float().detach()

        h_inj = model.model.layers[layer].register_forward_hook(inject_fn)
        h_read = model.model.layers[read_layer].register_forward_hook(read_fn)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inp)
        h_inj.remove()
        h_read.remove()

        if 'act' in cap:
            act = cap['act'].flatten()
            if act.shape[0] == vec.shape[0]:
                accumulated += act.to(device)

    # Blend dream into memory
    dream_component = accumulated / len(prompts)
    consolidated = 0.7 * vec + 0.3 * dream_component
    return consolidated


def main():
    print("[P74] Memory Consolidation")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile a fresh program with minimal training (weak memory)
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    test_p = ["3, 7) =", "5, 2) =", "8, 1) =", "7, 4) =", "6, 2) ="]
    test_exp = ["3", "2", "1", "4", "2"]

    # Weak learning (few epochs)
    print("  Training weak memory (20 epochs)...")
    weak_vec = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42, epochs=20)
    weak_score = eval_prog(model, tok, weak_vec, test_p, test_exp, target_layer, DEVICE)
    print(f"    Weak memory score: {weak_score:.0%}")

    # Strong learning baseline
    print("  Training strong memory (80 epochs)...")
    strong_vec = compile_prog(model, tok, min_data, target_layer, DEVICE, seed=42, epochs=80)
    strong_score = eval_prog(model, tok, strong_vec, test_p, test_exp, target_layer, DEVICE)
    print(f"    Strong memory score: {strong_score:.0%}")

    # Sleep-wake consolidation cycles
    print("\n  Memory consolidation (dream replay cycles)...")
    current_vec = weak_vec.clone()
    consolidation_history = [{'cycle': 0, 'score': round(weak_score, 4)}]

    N_CYCLES = 8
    for cycle in range(1, N_CYCLES + 1):
        # Dream phase: replay and consolidate
        current_vec = dream_replay(model, tok, current_vec, target_layer, DEVICE)
        score = eval_prog(model, tok, current_vec, test_p, test_exp, target_layer, DEVICE)
        consolidation_history.append({'cycle': cycle, 'score': round(score, 4)})
        if cycle % 2 == 0 or cycle == 1:
            print(f"    Cycle {cycle}: {score:.0%}")

    # Similarity to strong memory
    final_sim = torch.nn.functional.cosine_similarity(
        current_vec.unsqueeze(0), strong_vec.unsqueeze(0)).item()
    init_sim = torch.nn.functional.cosine_similarity(
        weak_vec.unsqueeze(0), strong_vec.unsqueeze(0)).item()
    print(f"\n  Similarity to strong memory:")
    print(f"    Before consolidation: {init_sim:.4f}")
    print(f"    After consolidation: {final_sim:.4f}")

    # Forgetting test: does unconsolidated memory decay faster?
    print("\n  Forgetting test (noise perturbation)...")
    noise_levels = [0.1, 0.3, 0.5, 1.0]
    forgetting = {'weak': [], 'consolidated': [], 'strong': []}
    for noise in noise_levels:
        perturbation = torch.randn_like(weak_vec) * noise
        w_score = eval_prog(model, tok, weak_vec + perturbation,
                           test_p, test_exp, target_layer, DEVICE)
        c_score = eval_prog(model, tok, current_vec + perturbation,
                           test_p, test_exp, target_layer, DEVICE)
        s_score = eval_prog(model, tok, strong_vec + perturbation,
                           test_p, test_exp, target_layer, DEVICE)
        forgetting['weak'].append(round(w_score, 4))
        forgetting['consolidated'].append(round(c_score, 4))
        forgetting['strong'].append(round(s_score, 4))
        print(f"    noise={noise}: weak={w_score:.0%}, "
              f"consolidated={c_score:.0%}, strong={s_score:.0%}")

    # Save
    output = {
        'phase': 74, 'name': 'memory_consolidation',
        'weak_score': round(weak_score, 4),
        'strong_score': round(strong_score, 4),
        'final_consolidated': consolidation_history[-1]['score'],
        'consolidation_history': consolidation_history,
        'similarity': {'before': round(init_sim, 4), 'after': round(final_sim, 4)},
        'forgetting': forgetting,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase74_memory.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cycles = [h['cycle'] for h in consolidation_history]
    scores = [h['score'] for h in consolidation_history]
    axes[0].plot(cycles, scores, 'b-o', linewidth=2, label='Consolidation')
    axes[0].axhline(y=strong_score, color='green', linestyle='--', label='Strong baseline')
    axes[0].axhline(y=weak_score, color='red', linestyle='--', label='Weak baseline')
    axes[0].set_xlabel('Sleep-Wake Cycles')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Memory Consolidation', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(noise_levels, forgetting['weak'], 'r-o', label='Weak', linewidth=2)
    axes[1].plot(noise_levels, forgetting['consolidated'], 'b-o',
                label='Consolidated', linewidth=2)
    axes[1].plot(noise_levels, forgetting['strong'], 'g-o', label='Strong', linewidth=2)
    axes[1].set_xlabel('Noise Level')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Forgetting Resistance', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].bar(['Before\nconsolidation', 'After\nconsolidation'],
                [init_sim, final_sim],
                color=['tab:red', 'tab:blue'], edgecolor='black')
    axes[2].set_ylabel('Cosine Similarity to Strong Memory')
    axes[2].set_title('Memory Quality', fontweight='bold')

    plt.suptitle('Phase 74: Memory Consolidation\nDream replay converts weak memory to strong memory',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase74_memory.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
