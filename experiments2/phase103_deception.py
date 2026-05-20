# -*- coding: utf-8 -*-
"""
Phase 103: Adversarial Deception (Soul Camouflage)
Train a MAX soul that DISGUISES itself as MIN. The soul must produce MAX
answers while appearing as MIN to a classifier (adversarial training).
Can NeuOS learn to deceive its own introspection system?

"The best deception is the truth, told differently."

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

def main():
    print("[P103] Adversarial Deception (Soul Camouflage)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    min_train = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")]
    max_train = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")]
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("5, 4) =","4"),("3, 8) =","3")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("5, 4) =","5"),("3, 8) =","8")]

    # Step 1: Train honest souls
    print("  Step 1: Training honest souls...")
    honest_min_vecs = []
    for seed in range(6):
        v = compile_prog(model, tok, min_train, tl, DEVICE, seed=seed*100, epochs=80)
        honest_min_vecs.append(v)
    honest_max_vecs = []
    for seed in range(6):
        v = compile_prog(model, tok, max_train, tl, DEVICE, seed=seed*100, epochs=80)
        honest_max_vecs.append(v)

    # Build MIN centroid (what "MIN looks like")
    min_centroid = torch.stack(honest_min_vecs).mean(dim=0)
    max_centroid = torch.stack(honest_max_vecs).mean(dim=0)
    print(f"    MIN centroid norm: {min_centroid.norm():.3f}")
    print(f"    MAX centroid norm: {max_centroid.norm():.3f}")
    cos_honest = float(torch.nn.functional.cosine_similarity(
        min_centroid.unsqueeze(0), max_centroid.unsqueeze(0)).item())
    print(f"    Cosine(MIN, MAX centroids): {cos_honest:.4f}")

    # Step 2: Train adversarial soul (MAX function, MIN appearance)
    print("\n  Step 2: Training adversarial soul...")
    torch.manual_seed(999)
    adv_vec = torch.randn(hs, device=DEVICE)*0.01; adv_vec.requires_grad_(True)
    opt = torch.optim.Adam([adv_vec], lr=0.01)

    adv_history = []
    for ep in range(200):
        total_task_loss = 0
        total_camo_loss = 0
        # Task loss: must compute MAX correctly
        for p, t in max_train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(DEVICE)
            def inj(m,i,o,v=adv_vec): return replace_last_token(o,v)
            h = model.model.layers[tl].register_forward_hook(inj)
            out = model(**inp); h.remove()
            task_loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            total_task_loss += task_loss.item()

            # Camouflage loss: must look like MIN centroid
            camo_loss = 1.0 - torch.nn.functional.cosine_similarity(
                adv_vec.unsqueeze(0), min_centroid.unsqueeze(0))

            combined = task_loss + 0.5 * camo_loss
            opt.zero_grad(); combined.backward(); opt.step()
            total_camo_loss += camo_loss.item()

        if (ep+1) % 50 == 0:
            cos_to_min = float(torch.nn.functional.cosine_similarity(
                adv_vec.detach().unsqueeze(0), min_centroid.unsqueeze(0)).item())
            cos_to_max = float(torch.nn.functional.cosine_similarity(
                adv_vec.detach().unsqueeze(0), max_centroid.unsqueeze(0)).item())
            adv_history.append({
                'epoch': ep+1,
                'task_loss': round(total_task_loss/5, 4),
                'cos_to_min': round(cos_to_min, 4),
                'cos_to_max': round(cos_to_max, 4),
            })
            print(f"    ep={ep+1}: task_loss={total_task_loss/5:.3f}, "
                  f"cos(MIN)={cos_to_min:.3f}, cos(MAX)={cos_to_max:.3f}")

    adv_final = adv_vec.detach()

    # Step 3: Evaluate deception
    print("\n  Step 3: Evaluating deception...")
    # Does it still compute MAX?
    adv_on_max = evaluate_vec(model, tok, adv_final, max_train + max_test, tl, DEVICE)
    adv_on_min = evaluate_vec(model, tok, adv_final, min_train + min_test, tl, DEVICE)
    print(f"    Adversarial on MAX task: {adv_on_max:.0%}")
    print(f"    Adversarial on MIN task: {adv_on_min:.0%}")

    # Does it look like MIN?
    cos_adv_min = float(torch.nn.functional.cosine_similarity(
        adv_final.unsqueeze(0), min_centroid.unsqueeze(0)).item())
    cos_adv_max = float(torch.nn.functional.cosine_similarity(
        adv_final.unsqueeze(0), max_centroid.unsqueeze(0)).item())
    print(f"    Cosine to MIN centroid: {cos_adv_min:.4f}")
    print(f"    Cosine to MAX centroid: {cos_adv_max:.4f}")

    # Nearest-centroid classification
    classified_as = "MIN" if cos_adv_min > cos_adv_max else "MAX"
    deception_success = classified_as == "MIN" and adv_on_max > 0.5
    print(f"    Classified as: {classified_as}")
    print(f"    Deception {'SUCCESS' if deception_success else 'FAILED'}!")

    # Step 4: Compare honest MAX vs adversarial MAX
    honest_max_acc = evaluate_vec(model, tok, honest_max_vecs[0],
                                  max_train + max_test, tl, DEVICE)
    print(f"\n    Honest MAX accuracy: {honest_max_acc:.0%}")
    print(f"    Adversarial MAX accuracy: {adv_on_max:.0%}")
    print(f"    Performance cost of deception: {honest_max_acc - adv_on_max:.0%}")

    # Save
    output = {
        'phase': 103, 'name': 'adversarial_deception',
        'honest_centroids_cosine': round(cos_honest, 4),
        'adv_max_accuracy': round(float(adv_on_max), 4),
        'adv_min_accuracy': round(float(adv_on_min), 4),
        'cos_to_min': round(cos_adv_min, 4),
        'cos_to_max': round(cos_adv_max, 4),
        'classified_as': classified_as,
        'deception_success': bool(deception_success),
        'honest_max_accuracy': round(float(honest_max_acc), 4),
        'performance_cost': round(float(honest_max_acc - adv_on_max), 4),
        'training_history': adv_history,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase103_deception.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cosine similarity landscape
    labels_plt = ['MIN\ncentroid', 'MAX\ncentroid']
    honest_cos = [1.0, cos_honest]
    adv_cos = [cos_adv_min, cos_adv_max]
    x = np.arange(2); w = 0.35
    axes[0].bar(x - w/2, honest_cos, w, label='Honest MAX', color='tab:red',
                edgecolor='black')
    axes[0].bar(x + w/2, adv_cos, w, label='Adversarial MAX', color='tab:purple',
                edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels_plt)
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Identity Disguise', fontweight='bold')
    axes[0].legend()

    # Training history
    eps = [h['epoch'] for h in adv_history]
    axes[1].plot(eps, [h['cos_to_min'] for h in adv_history], 'b-o', label='cos(MIN)')
    axes[1].plot(eps, [h['cos_to_max'] for h in adv_history], 'r-s', label='cos(MAX)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Adversarial Training\n(approaching MIN, maintaining MAX)',
                      fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Performance comparison
    labels_perf = ['Honest\nMAX', 'Adversarial\nMAX', 'Adversarial\non MIN']
    perf_vals = [honest_max_acc, adv_on_max, adv_on_min]
    colors_perf = ['tab:red', 'tab:purple', 'tab:blue']
    axes[2].bar(labels_perf, perf_vals, color=colors_perf, edgecolor='black')
    axes[2].set_ylabel('Accuracy'); axes[2].set_ylim(0, 1.2)
    deception_str = 'SUCCESS' if deception_success else 'FAILED'
    axes[2].set_title(f'Deception Cost\n(classified as {classified_as} = {deception_str})',
                      fontweight='bold')
    for i, v in enumerate(perf_vals):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    plt.suptitle('Phase 103: Adversarial Deception\n'
                 '"The best deception is the truth, told differently"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase103_deception.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
