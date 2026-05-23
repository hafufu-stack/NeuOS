# -*- coding: utf-8 -*-
"""
Phase 182: LoRA-Soul Correspondence
Is LoRA (rank-1 weight update) equivalent to soul vector injection?
Compare the two approaches in the 7D soul space.
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8


def train_soul(model, tok, data, device, layer=LAYER, epochs=100, seed=42):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(epochs):
        for p, t in data:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(device)
            def inj(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def evaluate(model, tok, test_data, device, inject_fn=None, layer=LAYER):
    correct = 0
    for prompt, expected in test_data:
        hooks = []
        if inject_fn is not None:
            hooks.append(model.model.layers[layer].register_forward_hook(inject_fn))
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        for h in hooks:
            h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data) if test_data else 0


class ManualLoRA(torch.nn.Module):
    """Manual LoRA implementation for a specific layer's output projection."""
    def __init__(self, in_dim, out_dim, rank=1):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = torch.nn.Parameter(torch.randn(rank, out_dim) * 0.01)

    def forward(self, x):
        return x + x @ self.A @ self.B


def main():
    print("[P182] LoRA-Soul Correspondence")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),
                ("9, 3) =","3"),("7, 4) =","4"),("6, 1) =","1"),("2, 8) =","2"),
                ("5, 9) =","5"),("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","6"),
                ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","8"),
                ("5, 9) =","9"),("1, 3) =","3")]
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("1, 5) =","1"),("8, 4) =","4")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("1, 5) =","5"),("8, 4) =","8")]

    results = {}

    # === Train soul vectors (baseline) ===
    print("  Training soul vectors...")
    min_soul = train_soul(model, tok, min_data, DEVICE, seed=42)
    max_soul = train_soul(model, tok, max_data, DEVICE, seed=42)

    def make_soul_inject(soul):
        def inj(m, i, o, v=soul): return replace_last_token(o, v)
        return inj

    soul_min_acc = evaluate(model, tok, min_test, DEVICE, make_soul_inject(min_soul))
    soul_max_acc = evaluate(model, tok, max_test, DEVICE, make_soul_inject(max_soul))
    results['soul'] = {'MIN': round(soul_min_acc, 4), 'MAX': round(soul_max_acc, 4)}
    print("  Soul: MIN=%.0f%%, MAX=%.0f%%" % (soul_min_acc*100, soul_max_acc*100))

    # === Train Manual LoRA ===
    print("\n  Training Manual LoRA (rank 1, 2, 4)...")
    lora_results = {}
    lora_deltas = {}  # Store weight deltas for comparison

    for rank in [1, 2, 4]:
        for task_name, train_d, test_d in [('MIN', min_data, min_test),
                                            ('MAX', max_data, max_test)]:
            torch.manual_seed(42)
            lora = ManualLoRA(896, 896, rank=rank).to(DEVICE)
            opt = torch.optim.Adam(lora.parameters(), lr=0.005)

            for epoch in range(100):
                for p, t in train_d:
                    tid = tok.encode(t)[-1]
                    inp = tok(p, return_tensors='pt').to(DEVICE)

                    def lora_hook(m, i, o, lora_mod=lora):
                        if isinstance(o, tuple):
                            h = o[0]
                        else:
                            h = o
                        if h.dim() == 3:
                            modified = lora_mod(h[0, -1:, :])  # (1, 896)
                            new_h = torch.cat([h[0, :-1, :], modified], dim=0).unsqueeze(0)
                        else:
                            new_h = h
                        if isinstance(o, tuple):
                            return (new_h,) + o[1:]
                        return new_h

                    hook = model.model.layers[LAYER].register_forward_hook(lora_hook)
                    out = model(**inp); hook.remove()
                    loss = torch.nn.functional.cross_entropy(
                        out.logits[0, -1, :].unsqueeze(0),
                        torch.tensor([tid]).to(DEVICE))
                    opt.zero_grad(); loss.backward(); opt.step()

            # Evaluate
            def lora_eval_hook(m, i, o, lora_mod=lora):
                if isinstance(o, tuple):
                    h = o[0].clone()
                    if h.dim() == 3:
                        h[0, -1, :] = lora_mod(h[0, -1, :].unsqueeze(0)).squeeze(0)
                    return (h,) + o[1:]
                else:
                    h = o.clone()
                    if h.dim() == 3:
                        h[0, -1, :] = lora_mod(h[0, -1, :].unsqueeze(0)).squeeze(0)
                    return h

            acc = evaluate(model, tok, test_d, DEVICE, lora_eval_hook)
            key = 'rank%d_%s' % (rank, task_name)
            lora_results[key] = round(acc, 4)
            print("    LoRA rank=%d %s: %.0f%%" % (rank, task_name, acc*100))

            # Compute effective weight delta (A @ B)
            with torch.no_grad():
                delta = (lora.A @ lora.B).detach()
                lora_deltas[key] = delta

    results['lora'] = lora_results

    # === Compare LoRA delta with soul vector ===
    print("\n  === LoRA-Soul Comparison ===")
    comparison = {}

    # For rank-1 LoRA: delta is a matrix, soul is a vector
    # The LoRA effect on the last token is: h_new = h + h @ A @ B
    # The soul effect is: h_new = soul_vector (replacement)
    # These are fundamentally different: LoRA is input-dependent, soul is fixed

    # But we can compute the "effective soul" of LoRA: apply LoRA to the mean hidden state
    print("  Computing mean hidden states...")
    h_mean_states = []
    for prompt, _ in min_data + max_data:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        states = {}
        def capture(m, i, o):
            tensor = o[0] if isinstance(o, tuple) else o
            if tensor.dim() == 3:
                states['h'] = tensor[0, -1, :].detach()
            elif tensor.dim() == 2:
                states['h'] = tensor[-1, :].detach()
        hook = model.model.layers[LAYER].register_forward_hook(capture)
        with torch.no_grad():
            model(**inp)
        hook.remove()
        if 'h' in states:
            h_mean_states.append(states['h'])

    h_mean = torch.stack(h_mean_states).mean(dim=0)

    # Effective soul of LoRA = h_mean @ A @ B (the additive perturbation)
    for task_name in ['MIN', 'MAX']:
        key = 'rank1_%s' % task_name
        if key in lora_deltas:
            delta = lora_deltas[key]
            effective_soul = (h_mean.unsqueeze(0) @ delta).squeeze(0)
            soul_vec = min_soul if task_name == 'MIN' else max_soul

            cos = torch.nn.functional.cosine_similarity(
                effective_soul.unsqueeze(0), soul_vec.unsqueeze(0)).item()
            comparison['%s_cosine' % task_name] = round(cos, 6)
            comparison['%s_soul_norm' % task_name] = round(soul_vec.norm().item(), 4)
            comparison['%s_lora_effect_norm' % task_name] = round(effective_soul.norm().item(), 4)
            print("    %s: cos(LoRA_effect, soul) = %.4f" % (task_name, cos))
    results['comparison'] = comparison

    # Build PCA with souls + LoRA effective vectors
    print("\n  Projecting to 7D soul space...")
    all_vecs = [min_soul.cpu().numpy(), max_soul.cpu().numpy()]
    vec_labels = ['soul_MIN', 'soul_MAX']
    for task_name in ['MIN', 'MAX']:
        key = 'rank1_%s' % task_name
        if key in lora_deltas:
            eff = (h_mean.unsqueeze(0) @ lora_deltas[key]).squeeze(0)
            all_vecs.append(eff.cpu().numpy())
            vec_labels.append('lora_%s' % task_name)

    pca = PCA(n_components=min(7, len(all_vecs)))
    pca.fit(np.array(all_vecs))
    coords_7d = {}
    for label, vec in zip(vec_labels, all_vecs):
        coords_7d[label] = pca.transform(vec.reshape(1, -1))[0]

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Accuracy comparison
    ax = axes[0, 0]
    methods = ['Soul', 'LoRA r=1', 'LoRA r=2', 'LoRA r=4']
    min_accs = [soul_min_acc,
                lora_results.get('rank1_MIN', 0),
                lora_results.get('rank2_MIN', 0),
                lora_results.get('rank4_MIN', 0)]
    max_accs = [soul_max_acc,
                lora_results.get('rank1_MAX', 0),
                lora_results.get('rank2_MAX', 0),
                lora_results.get('rank4_MAX', 0)]
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w/2, min_accs, w, label='MIN', color='#E91E63',
           edgecolor='black', linewidth=1.5)
    ax.bar(x + w/2, max_accs, w, label='MAX', color='#2196F3',
           edgecolor='black', linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel('Accuracy')
    ax.set_title('Soul vs LoRA: Task Accuracy', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.2)

    # Panel 2: Cosine similarity
    ax = axes[0, 1]
    cos_vals = [comparison.get('MIN_cosine', 0), comparison.get('MAX_cosine', 0)]
    bars = ax.bar(['MIN', 'MAX'], cos_vals, color=['#E91E63', '#2196F3'],
                  edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, cos_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.4f' % val, ha='center', fontweight='bold', fontsize=11)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('cos(LoRA_effect, Soul Vector)', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Panel 3: 7D projection
    ax = axes[1, 0]
    markers = {'soul_MIN': 'o', 'soul_MAX': 's', 'lora_MIN': '^', 'lora_MAX': 'v'}
    colors_map = {'soul_MIN': '#E91E63', 'soul_MAX': '#2196F3',
                  'lora_MIN': '#FF9800', 'lora_MAX': '#4CAF50'}
    for label in vec_labels:
        if label in coords_7d:
            c = coords_7d[label]
            ax.scatter(c[0], c[1] if len(c) > 1 else 0,
                      marker=markers.get(label, 'o'),
                      color=colors_map.get(label, 'gray'),
                      s=150, label=label, edgecolor='black', linewidth=1.5, zorder=5)
    ax.set_xlabel('PC0')
    ax.set_ylabel('PC1')
    ax.set_title('Soul vs LoRA in PCA Space', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        "LoRA-Soul Correspondence\n\n"
        "Key Finding:\n"
        "  Soul: direct replacement of h[-1]\n"
        "  LoRA: additive h + h@A@B\n\n"
        "Cosine Similarity:\n"
        "  MIN: %.4f\n"
        "  MAX: %.4f\n\n"
        "Interpretation:\n"
        "  %s" % (
            comparison.get('MIN_cosine', 0),
            comparison.get('MAX_cosine', 0),
            "High cosine -> same subspace!" if abs(comparison.get('MIN_cosine', 0)) > 0.5
            else "Low cosine -> different mechanisms"
        )
    )
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Phase 182: LoRA-Soul Correspondence\n'
                 '"Is rank-1 LoRA the weight-space twin of soul injection?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase182_lora_soul.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 182, 'name': 'lora_soul',
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase182_lora_soul.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P182 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
