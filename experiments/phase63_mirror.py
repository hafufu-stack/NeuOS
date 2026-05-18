# -*- coding: utf-8 -*-
"""
Phase 63: Mirror Test (Computational Self-Recognition)
The mirror test for AI: inject a program, capture its output,
inject that output back. Does the model recognize it as
"the same program"? Computational self-awareness.

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


def compile_prog(model, tok, train, layer, device, seed=42):
    hidden_size = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hidden_size, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(100):
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


def pass_through(model, tok, vec, prompt, inject_layer, read_layer, device):
    """Inject at inject_layer, read output at read_layer."""
    cap = [None]
    def inject_fn(module, input, output, v=vec):
        return replace_last_token(output, v)
    def read_fn(module, input, output):
        cap[0] = get_last_token(output)
    h1 = model.model.layers[inject_layer].register_forward_hook(inject_fn)
    h2 = model.model.layers[read_layer].register_forward_hook(read_fn)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)
    h1.remove(); h2.remove()
    return cap[0].float().squeeze()


def main():
    print("[P63] Mirror Test")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    INJECT_L = 8
    MIRROR_L = 22

    for p in model.parameters():
        p.requires_grad = False

    # Compile test programs
    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    max_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")]

    print("  Compiling programs...")
    min_vec = compile_prog(model, tok, min_data, INJECT_L, DEVICE, seed=42)
    max_vec = compile_prog(model, tok, max_data, INJECT_L, DEVICE, seed=99)

    prompt = "3, 7) ="

    # Step 1: Direct mirror - inject, read, re-inject, read again
    print("\n  Step 1: Mirror test (inject -> read -> re-inject -> read)...")
    mirror_results = {}

    for name, vec in [('MIN', min_vec), ('MAX', max_vec)]:
        # First pass: inject at L8, read at L22
        reflection1 = pass_through(model, tok, vec, prompt, INJECT_L, MIRROR_L, DEVICE)

        # Second pass: inject reflection back at L8, read at L22
        reflection2 = pass_through(model, tok, reflection1, prompt, INJECT_L, MIRROR_L, DEVICE)

        # Third pass
        reflection3 = pass_through(model, tok, reflection2, prompt, INJECT_L, MIRROR_L, DEVICE)

        # Similarities
        sim_orig_r1 = torch.nn.functional.cosine_similarity(
            vec.unsqueeze(0), reflection1.unsqueeze(0)).item()
        sim_r1_r2 = torch.nn.functional.cosine_similarity(
            reflection1.unsqueeze(0), reflection2.unsqueeze(0)).item()
        sim_r2_r3 = torch.nn.functional.cosine_similarity(
            reflection2.unsqueeze(0), reflection3.unsqueeze(0)).item()
        sim_orig_r2 = torch.nn.functional.cosine_similarity(
            vec.unsqueeze(0), reflection2.unsqueeze(0)).item()

        mirror_results[name] = {
            'orig_vs_r1': round(sim_orig_r1, 4),
            'r1_vs_r2': round(sim_r1_r2, 4),
            'r2_vs_r3': round(sim_r2_r3, 4),
            'orig_vs_r2': round(sim_orig_r2, 4),
        }
        print(f"    {name}: orig->r1={sim_orig_r1:.4f}, r1->r2={sim_r1_r2:.4f}, "
              f"r2->r3={sim_r2_r3:.4f}")

    # Step 2: Cross-recognition
    print("\n  Step 2: Cross-recognition (does MIN reflection look like MAX?)...")
    min_r1 = pass_through(model, tok, min_vec, prompt, INJECT_L, MIRROR_L, DEVICE)
    max_r1 = pass_through(model, tok, max_vec, prompt, INJECT_L, MIRROR_L, DEVICE)

    cross_sim = torch.nn.functional.cosine_similarity(
        min_r1.unsqueeze(0), max_r1.unsqueeze(0)).item()
    min_self = torch.nn.functional.cosine_similarity(
        min_vec.unsqueeze(0), min_r1.unsqueeze(0)).item()
    max_self = torch.nn.functional.cosine_similarity(
        max_vec.unsqueeze(0), max_r1.unsqueeze(0)).item()

    print(f"    MIN self-recognition: {min_self:.4f}")
    print(f"    MAX self-recognition: {max_self:.4f}")
    print(f"    Cross (MIN_r1 vs MAX_r1): {cross_sim:.4f}")

    # Step 3: Functional preservation through mirror
    print("\n  Step 3: Does the reflection still work as a program?...")
    test_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                 ("7, 2) =", "2"), ("6, 3) =", "3")]

    for name, vec, r1 in [('MIN', min_vec, min_r1), ('MAX', max_vec, max_r1)]:
        orig_correct = 0
        mirror_correct = 0
        expected_list = [("3", "7"), ("2", "5"), ("1", "8"), ("2", "7"), ("3", "6")]
        min_exp = ["3", "2", "1", "2", "3"]
        max_exp = ["7", "5", "8", "7", "6"]
        exp = min_exp if name == 'MIN' else max_exp

        for i, (tp, e) in enumerate(zip([d[0] for d in test_data], exp)):
            # Original
            def inj_o(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[INJECT_L].register_forward_hook(inj_o)
            inp = tok(tp, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            if tok.decode(out.logits[0, -1, :].argmax().item()).strip() == e:
                orig_correct += 1

            # Reflection
            def inj_r(module, input, output, v=r1):
                return replace_last_token(output, v)
            h = model.model.layers[INJECT_L].register_forward_hook(inj_r)
            inp = tok(tp, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            if tok.decode(out.logits[0, -1, :].argmax().item()).strip() == e:
                mirror_correct += 1

        print(f"    {name}: original={orig_correct}/5, reflection={mirror_correct}/5")

    # Save
    output = {
        'phase': 63, 'name': 'mirror_test',
        'mirror_results': mirror_results,
        'cross_recognition': round(cross_sim, 4),
        'min_self_recognition': round(min_self, 4),
        'max_self_recognition': round(max_self, 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase63_mirror.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Mirror convergence
    for name in ['MIN', 'MAX']:
        r = mirror_results[name]
        sims = [r['orig_vs_r1'], r['r1_vs_r2'], r['r2_vs_r3']]
        axes[0].plot(range(3), sims, 'o-', linewidth=2, markersize=8, label=name)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(['orig->r1', 'r1->r2', 'r2->r3'])
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Mirror Convergence', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Self vs Cross
    axes[1].bar(['MIN\nself', 'MAX\nself', 'Cross'],
                [min_self, max_self, cross_sim],
                color=['tab:blue', 'tab:orange', 'tab:red'], edgecolor='black')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Self vs Cross Recognition', fontweight='bold')
    for i, v in enumerate([min_self, max_self, cross_sim]):
        axes[1].text(i, v + 0.02 if v >= 0 else v - 0.05,
                    f'{v:.3f}', ha='center', fontweight='bold')

    axes[2].axis('off')
    summary = ("MIRROR TEST (Self-Recognition)\n" + "="*35 + "\n\n"
               "Protocol:\n"
               "  1. Inject program at L8\n"
               "  2. Read reflection at L22\n"
               "  3. Re-inject reflection at L8\n"
               "  4. Compare reflections\n\n"
               f"MIN: {mirror_results['MIN']['r1_vs_r2']:.3f} (convergence)\n"
               f"MAX: {mirror_results['MAX']['r1_vs_r2']:.3f} (convergence)\n"
               f"Cross: {cross_sim:.3f}\n\n"
               f"Self > Cross = self-awareness")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                fontsize=10, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 63: Mirror Test\nComputational self-recognition in neural programs',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase63_mirror.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
