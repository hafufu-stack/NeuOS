# -*- coding: utf-8 -*-
"""
Phase 76: Aging & Death
Programs age: SVD energy ratio changes over repeated use.
Compare young vs old programs. Do they degrade? Can they rejuvenate?

Uses P64's compression ratio as a vitality metric.

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
    history = []
    for epoch in range(epochs):
        total_loss = 0
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
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == epochs - 1:
            history.append({
                'epoch': epoch,
                'vec_snapshot': vec.detach().clone(),
                'loss': round(total_loss / len(train), 4),
                'norm': round(vec.detach().norm().item(), 4),
            })
    return vec.detach(), history


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


def measure_vitality(vec):
    """Measure program vitality using SVD energy concentration."""
    v = vec.cpu().numpy().flatten()
    # SVD energy in top-k components
    # Higher concentration = healthier program (P64 showed 10 dims suffice)
    u_vals = np.sort(np.abs(v))[::-1]
    total = np.sum(u_vals**2)
    top10_energy = np.sum(u_vals[:10]**2) / (total + 1e-10)
    top50_energy = np.sum(u_vals[:50]**2) / (total + 1e-10)
    return {
        'top10_energy': round(float(top10_energy), 4),
        'top50_energy': round(float(top50_energy), 4),
        'norm': round(float(np.linalg.norm(v)), 4),
        'entropy': round(float(-np.sum((u_vals/total) * np.log(u_vals/total + 1e-10))), 4),
    }


def main():
    print("[P76] Aging & Death")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    min_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                ("4, 6) =", "4"), ("9, 3) =", "3")]
    test_p = ["3, 7) =", "5, 2) =", "8, 1) =", "7, 4) =", "6, 2) ="]
    test_exp = ["3", "2", "1", "4", "2"]

    # Step 1: Track aging during training
    print("  Step 1: Tracking program aging during development...")
    vec, history = compile_prog(model, tok, min_data, target_layer, DEVICE,
                               seed=42, epochs=100)

    aging_data = []
    for snap in history:
        vitality = measure_vitality(snap['vec_snapshot'])
        acc = eval_prog(model, tok, snap['vec_snapshot'], test_p, test_exp,
                       target_layer, DEVICE)
        aging_data.append({
            'epoch': snap['epoch'],
            'accuracy': round(acc, 4),
            'loss': snap['loss'],
            'norm': snap['norm'],
            **vitality,
        })
        print(f"    epoch={snap['epoch']:3d}: acc={acc:.0%}, "
              f"norm={snap['norm']:.2f}, "
              f"top10_E={vitality['top10_energy']:.3f}")

    # Step 2: Accelerated aging (noise accumulation)
    print("\n  Step 2: Accelerated aging (noise injection)...")
    young = vec.clone()
    aged_results = []
    for age in range(10):
        noise = torch.randn_like(young) * 0.05
        young = young + noise
        acc = eval_prog(model, tok, young, test_p, test_exp, target_layer, DEVICE)
        vitality = measure_vitality(young)
        aged_results.append({
            'age': age + 1,
            'accuracy': round(acc, 4),
            **vitality,
        })
        if age % 2 == 0:
            print(f"    age={age+1}: acc={acc:.0%}, "
                  f"top10_E={vitality['top10_energy']:.3f}")

    # Step 3: Rejuvenation (can we restore an aged program?)
    print("\n  Step 3: Rejuvenation attempts...")
    aged_vec = young.clone()  # The most aged version
    aged_acc = eval_prog(model, tok, aged_vec, test_p, test_exp, target_layer, DEVICE)

    # Method 1: Re-training (stem cell therapy)
    print("  Method 1: Re-training (stem cell therapy)...")
    rejuv1 = aged_vec.clone()
    rejuv1.requires_grad_(True)
    opt = torch.optim.Adam([rejuv1], lr=0.005)
    for epoch in range(30):
        for prompt, target_str in min_data[:3]:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject(module, input, output, v=rejuv1):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    rejuv1_acc = eval_prog(model, tok, rejuv1.detach(), test_p, test_exp,
                          target_layer, DEVICE)
    print(f"    Aged: {aged_acc:.0%} -> Rejuvenated: {rejuv1_acc:.0%}")

    # Method 2: SVD pruning (remove noise dimensions)
    print("  Method 2: SVD pruning (telomere repair)...")
    v_np = aged_vec.cpu().numpy().flatten()
    # Keep only top-k components by magnitude
    indices = np.argsort(np.abs(v_np))[::-1]
    pruned = np.zeros_like(v_np)
    pruned[indices[:50]] = v_np[indices[:50]]  # Keep top 50
    rejuv2_vec = torch.tensor(pruned, dtype=torch.float32, device=DEVICE)
    rejuv2_acc = eval_prog(model, tok, rejuv2_vec, test_p, test_exp,
                          target_layer, DEVICE)
    print(f"    Aged: {aged_acc:.0%} -> SVD-pruned: {rejuv2_acc:.0%}")

    # Original (young) baseline
    young_acc = eval_prog(model, tok, vec, test_p, test_exp, target_layer, DEVICE)

    # Save
    output = {
        'phase': 76, 'name': 'aging_and_death',
        'aging_during_training': aging_data,
        'accelerated_aging': aged_results,
        'rejuvenation': {
            'aged': round(aged_acc, 4),
            'retrained': round(rejuv1_acc, 4),
            'svd_pruned': round(rejuv2_acc, 4),
            'young_baseline': round(young_acc, 4),
        },
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase76_aging.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = [d['epoch'] for d in aging_data]
    accs = [d['accuracy'] for d in aging_data]
    energies = [d['top10_energy'] for d in aging_data]
    ax2 = axes[0].twinx()
    axes[0].plot(epochs, accs, 'b-o', label='Accuracy', linewidth=2)
    ax2.plot(epochs, energies, 'r--s', label='Top10 Energy', linewidth=2)
    axes[0].set_xlabel('Training Epoch')
    axes[0].set_ylabel('Accuracy', color='blue')
    ax2.set_ylabel('Top10 Energy', color='red')
    axes[0].set_title('Development & Vitality', fontweight='bold')

    ages = [d['age'] for d in aged_results]
    age_accs = [d['accuracy'] for d in aged_results]
    axes[1].plot(ages, age_accs, 'r-o', linewidth=2)
    axes[1].axhline(y=young_acc, color='green', linestyle='--', label='Young baseline')
    axes[1].set_xlabel('Age (noise cycles)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accelerated Aging', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    labels = ['Young', 'Aged', 'Retrained', 'SVD\nPruned']
    vals = [young_acc, aged_acc, rejuv1_acc, rejuv2_acc]
    colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:purple']
    axes[2].bar(labels, vals, color=colors, edgecolor='black')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Rejuvenation Results', fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate(vals):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    plt.suptitle('Phase 76: Aging & Death\nProgram vitality, senescence, and rejuvenation',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase76_aging.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
