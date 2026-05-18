# -*- coding: utf-8 -*-
"""
Phase 46: Program Reverse Engineering (Opus Original)
Given a P35-compiled program vector (unknown function),
can we determine what it does WITHOUT executing it?
Use P41's probes + similarity analysis for static analysis.

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
    print("[P46] Program Reverse Engineering")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Step 1: Compile known programs
    print("  Step 1: Compiling known programs...")
    program_specs = {
        'MIN': [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")],
        'MAX': [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                ("4, 6) =", "6"), ("9, 3) =", "9")],
        'IDENTITY': [("3) =", "3"), ("5) =", "5"), ("8) =", "8"),
                     ("4) =", "4"), ("9) =", "9")],
        'DOUBLE': [("3) =", "6"), ("4) =", "8"), ("2) =", "4"), ("1) =", "2")],
    }

    compiled = {}
    for name, train in program_specs.items():
        vec = torch.randn(hidden_size, device=DEVICE) * 0.01
        vec.requires_grad_(True)
        opt = torch.optim.Adam([vec], lr=0.01)
        for epoch in range(100):
            for prompt, target_str in train:
                target_id = tok.encode(target_str)[-1]
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                def inject(module, input, output, v=vec):
                    return replace_last_token(output, v)
                h = model.model.layers[target_layer].register_forward_hook(inject)
                out = model(**inp)
                h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([target_id]).to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
        compiled[name] = vec.detach().clone()
        print(f"    {name}: compiled")

    # Step 2: Build fingerprint database
    # For each known program, compute its "fingerprint":
    # Run it on standardized inputs and record the output distribution
    print("\n  Step 2: Building fingerprint database...")
    standard_inputs = ["1) =", "3) =", "5) =", "7) =", "9) =",
                       "2, 4) =", "4, 2) =", "3, 7) =", "7, 3) ="]
    fingerprints = {}

    for name, vec in compiled.items():
        outputs = []
        for test_input in standard_inputs:
            def inject_fp(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_fp)
            inp = tok(test_input, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            logits = out.logits[0, -1, :]
            # Use top-5 logit distribution as fingerprint
            top5_ids = logits.topk(5).indices.cpu().tolist()
            top5_vals = torch.softmax(logits, dim=0).topk(5).values.cpu().tolist()
            outputs.append((top5_ids, top5_vals))
        fingerprints[name] = outputs

    # Step 3: Compile "mystery" programs and try to identify them
    print("\n  Step 3: Reverse engineering mystery programs...")
    mystery_specs = {
        'mystery_A': [("3, 7) =", "3"), ("5, 2) =", "2"), ("9, 1) =", "1")],  # MIN
        'mystery_B': [("3, 7) =", "7"), ("5, 2) =", "5"), ("9, 1) =", "9")],  # MAX
        'mystery_C': [("3) =", "3"), ("7) =", "7"), ("5) =", "5")],  # IDENTITY
    }
    ground_truth = {'mystery_A': 'MIN', 'mystery_B': 'MAX', 'mystery_C': 'IDENTITY'}

    re_results = {}
    for mystery_name, train in mystery_specs.items():
        # Compile mystery program
        vec = torch.randn(hidden_size, device=DEVICE) * 0.01
        vec.requires_grad_(True)
        opt = torch.optim.Adam([vec], lr=0.01)
        for epoch in range(100):
            for prompt, target_str in train:
                target_id = tok.encode(target_str)[-1]
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                def inject(module, input, output, v=vec):
                    return replace_last_token(output, v)
                h = model.model.layers[target_layer].register_forward_hook(inject)
                out = model(**inp)
                h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([target_id]).to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
        mystery_vec = vec.detach()

        # Method 1: Vector cosine similarity
        sims = {}
        for known_name, known_vec in compiled.items():
            sim = cosine_similarity(
                mystery_vec.cpu().numpy().reshape(1,-1),
                known_vec.cpu().numpy().reshape(1,-1))[0,0]
            sims[known_name] = round(float(sim), 4)
        best_match_vec = max(sims, key=sims.get)

        # Method 2: Behavioral fingerprint matching
        mystery_outputs = []
        for test_input in standard_inputs:
            def inject_m(module, input, output, v=mystery_vec):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_m)
            inp = tok(test_input, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            mystery_outputs.append(pred)

        # Compare behavioral outputs
        behavior_scores = {}
        for known_name in compiled:
            known_outputs = []
            for test_input in standard_inputs:
                def inject_k(module, input, output, v=compiled[known_name]):
                    return replace_last_token(output, v)
                h = model.model.layers[target_layer].register_forward_hook(inject_k)
                inp = tok(test_input, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                h.remove()
                pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
                known_outputs.append(pred)
            match = sum(m == k for m, k in zip(mystery_outputs, known_outputs)) / len(standard_inputs)
            behavior_scores[known_name] = round(match, 4)
        best_match_behavior = max(behavior_scores, key=behavior_scores.get)

        true_label = ground_truth[mystery_name]
        re_results[mystery_name] = {
            'true_label': true_label,
            'vec_match': best_match_vec,
            'vec_correct': best_match_vec == true_label,
            'vec_sims': sims,
            'behavior_match': best_match_behavior,
            'behavior_correct': best_match_behavior == true_label,
            'behavior_scores': behavior_scores,
        }
        print(f"    {mystery_name} (true: {true_label}):")
        print(f"      Vec match:      {best_match_vec} (sim={sims[best_match_vec]:.3f}) "
              f"{'OK' if best_match_vec == true_label else 'X'}")
        print(f"      Behavior match: {best_match_behavior} (score={behavior_scores[best_match_behavior]:.1%}) "
              f"{'OK' if best_match_behavior == true_label else 'X'}")

    vec_acc = sum(r['vec_correct'] for r in re_results.values()) / len(re_results)
    beh_acc = sum(r['behavior_correct'] for r in re_results.values()) / len(re_results)

    # Save
    output = {
        'phase': 46, 'name': 'program_reverse_engineering',
        'vec_identification_acc': round(vec_acc, 4),
        'behavior_identification_acc': round(beh_acc, 4),
        're_results': re_results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase46_reverse.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(['Vector\nSimilarity', 'Behavioral\nFingerprint'],
                [vec_acc, beh_acc],
                color=['tab:blue', 'tab:green'], edgecolor='black')
    axes[0].set_ylabel('Identification Accuracy')
    axes[0].set_title('Mystery Program Identification', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([vec_acc, beh_acc]):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=14)

    # Similarity matrix
    names = list(compiled.keys())
    mystery_names = list(re_results.keys())
    sim_matrix = np.zeros((len(mystery_names), len(names)))
    for i, mn in enumerate(mystery_names):
        for j, kn in enumerate(names):
            sim_matrix[i, j] = re_results[mn]['vec_sims'].get(kn, 0)
    im = axes[1].imshow(sim_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names)
    axes[1].set_yticks(range(len(mystery_names)))
    axes[1].set_yticklabels([f"{mn}\n(true:{re_results[mn]['true_label']})" for mn in mystery_names])
    axes[1].set_title('Vector Similarity Matrix', fontweight='bold')
    plt.colorbar(im, ax=axes[1])
    for i in range(len(mystery_names)):
        for j in range(len(names)):
            axes[1].text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)

    plt.suptitle('Phase 46: Program Reverse Engineering\nIdentify unknown programs without execution',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase46_reverse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
