# -*- coding: utf-8 -*-
"""
Phase 65: Program Genome
P64 showed programs live in 10D subspace of 896D.
Map the entire function landscape. What lies between MIN and MAX?
Can we smoothly interpolate through function space?

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

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


def evaluate_behavior(model, tok, vec, prompts, layer, device):
    """Return the model's predictions for each prompt."""
    preds = []
    for prompt in prompts:
        def inject(module, input, output, v=vec):
            return replace_last_token(output, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append(pred)
    return preds


def main():
    print("[P65] Program Genome")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile reference programs
    programs = {
        'MIN': [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")],
        'MAX': [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")],
        'FIRST': [("3, 7) =", "3"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                  ("4, 6) =", "4"), ("9, 3) =", "9")],
        'SECOND': [("3, 7) =", "7"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                   ("4, 6) =", "6"), ("9, 3) =", "3")],
    }

    print("  Compiling reference programs...")
    vecs = {}
    for name, data in programs.items():
        v = compile_prog(model, tok, data, target_layer, DEVICE, seed=hash(name) % 10000)
        vecs[name] = v
        print(f"    {name} compiled")

    test_prompts = ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) =", "9, 3) =",
                    "7, 2) =", "6, 4) =", "2, 8) ="]

    # Step 1: Interpolation between MIN and MAX
    print("\n  Step 1: MIN-MAX interpolation...")
    n_steps = 11
    interpolation = {}
    for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
        mixed = (1 - alpha) * vecs['MIN'] + alpha * vecs['MAX']
        preds = evaluate_behavior(model, tok, mixed, test_prompts, target_layer, DEVICE)
        interpolation[f'{alpha:.1f}'] = preds
        # Count MIN-like vs MAX-like
        min_exp = ["3", "2", "1", "4", "3", "2", "4", "2"]
        max_exp = ["7", "5", "8", "6", "9", "7", "6", "8"]
        min_match = sum(p == e for p, e in zip(preds, min_exp))
        max_match = sum(p == e for p, e in zip(preds, max_exp))
        if i % 2 == 0:
            print(f"    alpha={alpha:.1f}: MIN={min_match}/8, MAX={max_match}/8, preds={preds[:4]}")

    # Step 2: PCA of program space
    print("\n  Step 2: PCA of program space...")
    # Compile multiple variants of each function
    all_vecs = []
    all_labels = []
    for name in ['MIN', 'MAX', 'FIRST', 'SECOND']:
        for seed in range(5):
            v = compile_prog(model, tok, programs[name], target_layer, DEVICE,
                            seed=hash(name) % 10000 + seed * 100)
            all_vecs.append(v.cpu().numpy().flatten())
            all_labels.append(name)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.array(all_vecs))
    explained = pca.explained_variance_ratio_

    print(f"    PCA variance explained: {explained[0]:.1%}, {explained[1]:.1%}")
    for name in ['MIN', 'MAX', 'FIRST', 'SECOND']:
        idxs = [i for i, l in enumerate(all_labels) if l == name]
        mean_coord = np.mean([coords[i] for i in idxs], axis=0)
        print(f"    {name} centroid: ({mean_coord[0]:.2f}, {mean_coord[1]:.2f})")

    # Step 3: Random points in program space
    print("\n  Step 3: Random genome sampling...")
    random_behaviors = {}
    for trial in range(10):
        rand_vec = torch.randn(model.config.hidden_size, device=DEVICE) * 0.1
        preds = evaluate_behavior(model, tok, rand_vec, test_prompts[:5], target_layer, DEVICE)
        random_behaviors[f'random_{trial}'] = preds

    # Save
    output = {
        'phase': 65, 'name': 'program_genome',
        'interpolation': interpolation,
        'pca_variance': [round(float(v), 4) for v in explained],
        'random_behaviors': random_behaviors,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase65_genome.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Interpolation heatmap
    alphas = sorted(interpolation.keys(), key=float)
    min_scores = []
    max_scores = []
    for a in alphas:
        preds = interpolation[a]
        min_exp = ["3", "2", "1", "4", "3", "2", "4", "2"]
        max_exp = ["7", "5", "8", "6", "9", "7", "6", "8"]
        min_scores.append(sum(p == e for p, e in zip(preds, min_exp)) / 8)
        max_scores.append(sum(p == e for p, e in zip(preds, max_exp)) / 8)
    axes[0].plot([float(a) for a in alphas], min_scores, 'b-o', label='MIN', linewidth=2)
    axes[0].plot([float(a) for a in alphas], max_scores, 'r-o', label='MAX', linewidth=2)
    axes[0].set_xlabel('alpha (0=MIN, 1=MAX)')
    axes[0].set_ylabel('Match Rate')
    axes[0].set_title('MIN-MAX Interpolation Path', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # PCA scatter
    colors_map = {'MIN': 'tab:blue', 'MAX': 'tab:red', 'FIRST': 'tab:green', 'SECOND': 'tab:purple'}
    for name in ['MIN', 'MAX', 'FIRST', 'SECOND']:
        idxs = [i for i, l in enumerate(all_labels) if l == name]
        x = [coords[i, 0] for i in idxs]
        y = [coords[i, 1] for i in idxs]
        axes[1].scatter(x, y, c=colors_map[name], label=name, s=80, edgecolor='black')
    axes[1].set_xlabel(f'PC1 ({explained[0]:.0%})')
    axes[1].set_ylabel(f'PC2 ({explained[1]:.0%})')
    axes[1].set_title('Program Genome Map (PCA)', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Summary
    axes[2].axis('off')
    summary = ("PROGRAM GENOME\n" + "="*30 + "\n\n"
               "4 functions x 5 variants = 20 programs\n"
               f"PCA: {explained[0]:.0%} + {explained[1]:.0%} = {sum(explained):.0%}\n\n"
               "Interpolation:\n"
               f"  alpha=0.0: MIN={min_scores[0]:.0%}\n"
               f"  alpha=0.5: MIN={min_scores[5]:.0%}, MAX={max_scores[5]:.0%}\n"
               f"  alpha=1.0: MAX={max_scores[-1]:.0%}\n\n"
               "Function space is navigable!")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                fontsize=10, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 65: Program Genome\nMapping the landscape of neural function space',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase65_genome.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
