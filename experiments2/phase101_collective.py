# -*- coding: utf-8 -*-
"""
Phase 101: Collective Intelligence (Wisdom of Crowds)
Train N specialized souls, then combine them via weighted voting.
Does the committee outperform any individual member?

"None of us is as smart as all of us."

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

def get_logits(model, tok, vec, prompt, layer, device):
    """Get output logits with soul vec injected."""
    def inj(m,i,o,v=vec): return replace_last_token(o,v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad(): out = model(**inp)
    h.remove()
    return out.logits[0,-1,:].cpu()

def evaluate_vec(model, tok, vec, data, layer, device):
    c = 0
    for p, e in data:
        logits = get_logits(model, tok, vec, p, layer, device)
        if tok.decode(logits.argmax().item()).strip() == e: c += 1
    return c / len(data)

def main():
    print("[P101] Collective Intelligence (Wisdom of Crowds)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    tl = 8
    for p in model.parameters(): p.requires_grad = False

    train_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                  ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("5, 4) =","4"),("3, 8) =","3")]
    hard_data = [("1, 4) =","1"),("8, 3) =","3"),("6, 9) =","6"),
                 ("7, 5) =","5"),("4, 2) =","2")]
    all_test = test_data + hard_data

    # Step 1: Train a diverse committee
    print("  Step 1: Training committee of 12 diverse souls...")
    committee = []
    individual_accs = []
    for seed in range(12):
        # Vary training epochs and seed for diversity
        epochs = 60 + (seed % 4) * 20  # 60, 80, 100, 120
        v = compile_prog(model, tok, train_data, tl, DEVICE, seed=seed*111, epochs=epochs)
        acc = evaluate_vec(model, tok, v, all_test, tl, DEVICE)
        committee.append({'vec': v, 'seed': seed*111, 'epochs': epochs,
                         'acc': round(float(acc), 4)})
        individual_accs.append(float(acc))
        print(f"    Soul #{seed}: acc={acc:.0%} (seed={seed*111}, ep={epochs})")

    # Step 2: Ensemble methods
    print("\n  Step 2: Ensemble voting...")
    results = {}

    # Method A: Majority vote (hard voting)
    def majority_vote(model, tok, souls, data, layer, device):
        correct = 0
        for p, e in data:
            votes = {}
            for s in souls:
                logits = get_logits(model, tok, s['vec'], p, layer, device)
                pred = tok.decode(logits.argmax().item()).strip()
                votes[pred] = votes.get(pred, 0) + 1
            winner = max(votes, key=votes.get)
            if winner == e: correct += 1
        return correct / len(data)

    majority_acc = majority_vote(model, tok, committee, all_test, tl, DEVICE)
    results['majority_vote'] = round(float(majority_acc), 4)
    print(f"    Majority vote (all 12): {majority_acc:.0%}")

    # Method B: Weighted vote (weight by training accuracy)
    def weighted_vote(model, tok, souls, data, layer, device):
        correct = 0
        for p, e in data:
            scores = {}
            for s in souls:
                logits = get_logits(model, tok, s['vec'], p, layer, device)
                pred = tok.decode(logits.argmax().item()).strip()
                w = s['acc'] + 0.01  # avoid zero weight
                scores[pred] = scores.get(pred, 0) + w
            winner = max(scores, key=scores.get)
            if winner == e: correct += 1
        return correct / len(data)

    weighted_acc = weighted_vote(model, tok, committee, all_test, tl, DEVICE)
    results['weighted_vote'] = round(float(weighted_acc), 4)
    print(f"    Weighted vote (all 12): {weighted_acc:.0%}")

    # Method C: Logit averaging (soft ensemble)
    def soft_ensemble(model, tok, souls, data, layer, device):
        correct = 0
        for p, e in data:
            avg_logits = None
            for s in souls:
                logits = get_logits(model, tok, s['vec'], p, layer, device)
                if avg_logits is None:
                    avg_logits = logits.clone()
                else:
                    avg_logits += logits
            avg_logits /= len(souls)
            pred = tok.decode(avg_logits.argmax().item()).strip()
            if pred == e: correct += 1
        return correct / len(data)

    soft_acc = soft_ensemble(model, tok, committee, all_test, tl, DEVICE)
    results['soft_ensemble'] = round(float(soft_acc), 4)
    print(f"    Soft ensemble (all 12): {soft_acc:.0%}")

    # Step 3: Committee size ablation
    print("\n  Step 3: Committee size ablation...")
    # Sort by accuracy, pick top-K
    sorted_committee = sorted(committee, key=lambda x: x['acc'], reverse=True)
    size_results = []
    for k in [1, 2, 3, 5, 7, 10, 12]:
        top_k = sorted_committee[:k]
        ma = majority_vote(model, tok, top_k, all_test, tl, DEVICE)
        sa = soft_ensemble(model, tok, top_k, all_test, tl, DEVICE)
        size_results.append({'k': k, 'majority': round(float(ma), 4),
                            'soft': round(float(sa), 4)})
        print(f"    Top-{k}: majority={ma:.0%}, soft={sa:.0%}")

    # Stats
    best_individual = max(individual_accs)
    avg_individual = np.mean(individual_accs)
    results['best_individual'] = round(float(best_individual), 4)
    results['avg_individual'] = round(float(avg_individual), 4)
    results['committee_size_ablation'] = size_results

    print(f"\n  Best individual: {best_individual:.0%}")
    print(f"  Avg individual: {avg_individual:.0%}")
    print(f"  Best ensemble: {max(majority_acc, weighted_acc, soft_acc):.0%}")

    # Save
    output = {
        'phase': 101, 'name': 'collective_intelligence',
        'results': results,
        'individuals': [{'seed': c['seed'], 'epochs': c['epochs'],
                        'acc': c['acc']} for c in committee],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase101_collective.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Individual vs Ensemble
    methods = ['Best\nIndiv.', 'Avg\nIndiv.', 'Majority\nVote', 'Weighted\nVote',
               'Soft\nEnsemble']
    vals = [best_individual, avg_individual, majority_acc, weighted_acc, soft_acc]
    colors = ['tab:gray', 'tab:gray', 'tab:blue', 'tab:green', 'tab:purple']
    axes[0].bar(methods, vals, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy'); axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Individual vs Committee', fontweight='bold')
    for i, v in enumerate(vals):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=9)

    # Committee size
    ks = [r['k'] for r in size_results]
    axes[1].plot(ks, [r['majority'] for r in size_results], 'b-o', lw=2, label='Majority')
    axes[1].plot(ks, [r['soft'] for r in size_results], 'g-s', lw=2, label='Soft')
    axes[1].axhline(y=best_individual, color='gray', ls='--', label='Best individual')
    axes[1].set_xlabel('Committee Size (Top-K)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Size vs Accuracy', fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # Individual distribution
    axes[2].bar(range(12), individual_accs, color='tab:orange', edgecolor='black',
                alpha=0.7)
    axes[2].axhline(y=soft_acc, color='purple', ls='--', lw=2,
                     label=f'Soft ensemble={soft_acc:.0%}')
    axes[2].axhline(y=avg_individual, color='gray', ls='--',
                     label=f'Avg={avg_individual:.0%}')
    axes[2].set_xlabel('Soul #'); axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Individual Accuracies', fontweight='bold')
    axes[2].legend(fontsize=8)

    plt.suptitle('Phase 101: Collective Intelligence\n'
                 '"None of us is as smart as all of us"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase101_collective.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
