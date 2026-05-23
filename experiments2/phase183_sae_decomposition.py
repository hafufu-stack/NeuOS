# -*- coding: utf-8 -*-
"""
Phase 183: SAE Feature Decomposition
Decompose soul vectors into Sparse Autoencoder features.
Which monosemantic features are responsible for MIN vs MAX?
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8


class SparseAutoencoder(torch.nn.Module):
    """Simple sparse autoencoder with L1 regularization."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, input_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x):
        return self.relu(self.encoder(x))


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


def evaluate(model, tok, soul_vec, test_data, device, layer=LAYER):
    correct = 0
    for prompt, expected in test_data:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data) if test_data else 0


def main():
    print("[P183] SAE Feature Decomposition")
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

    # === Step 1: Collect hidden states for SAE training ===
    print("  Collecting hidden states at L%d..." % LAYER)
    hidden_states = []
    prompts = ["%d, %d) =" % (a, b) for a in range(1, 10) for b in range(1, 10) if a != b]

    for prompt in prompts[:200]:
        states = {}
        def capture(m, i, o):
            tensor = o[0] if isinstance(o, tuple) else o
            if tensor.dim() == 3:
                states['h'] = tensor[0, -1, :].detach()
            elif tensor.dim() == 2:
                states['h'] = tensor[-1, :].detach()
        hook = model.model.layers[LAYER].register_forward_hook(capture)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        hook.remove()
        if 'h' in states:
            hidden_states.append(states['h'])

    H = torch.stack(hidden_states)
    print("  Collected %d hidden states, shape=%s" % (H.shape[0], H.shape))

    # === Step 2: Train SAE ===
    print("  Training Sparse Autoencoder...")
    LATENT_DIM = 2048
    L1_COEFF = 0.01

    sae = SparseAutoencoder(896, LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=0.001)

    sae_losses = []
    for epoch in range(500):
        # Mini-batch
        perm = torch.randperm(H.shape[0])[:64]
        batch = H[perm]
        decoded, encoded = sae(batch)
        recon_loss = torch.nn.functional.mse_loss(decoded, batch)
        sparsity_loss = L1_COEFF * encoded.abs().mean()
        loss = recon_loss + sparsity_loss
        opt.zero_grad(); loss.backward(); opt.step()
        sae_losses.append(loss.item())
        if epoch % 100 == 0:
            n_active = (encoded > 0.01).float().sum(dim=1).mean().item()
            print("    Epoch %d: recon=%.6f, sparsity=%.6f, active=%.0f/%d" % (
                epoch, recon_loss.item(), sparsity_loss.item(), n_active, LATENT_DIM))

    # === Step 3: Train soul vectors ===
    print("\n  Training soul vectors...")
    min_soul = train_soul(model, tok, min_data, DEVICE, seed=42)
    max_soul = train_soul(model, tok, max_data, DEVICE, seed=42)

    min_acc = evaluate(model, tok, min_soul, min_test, DEVICE)
    max_acc = evaluate(model, tok, max_soul, max_test, DEVICE)
    print("  MIN accuracy: %.0f%%, MAX accuracy: %.0f%%" % (min_acc*100, max_acc*100))

    # === Step 4: Decompose souls into SAE features ===
    print("\n  Decomposing soul vectors into SAE features...")
    with torch.no_grad():
        min_features = sae.encode(min_soul.unsqueeze(0)).squeeze(0)
        max_features = sae.encode(max_soul.unsqueeze(0)).squeeze(0)

    # Active features
    min_active = (min_features > 0.01).nonzero(as_tuple=True)[0]
    max_active = (max_features > 0.01).nonzero(as_tuple=True)[0]
    print("  MIN: %d active features" % len(min_active))
    print("  MAX: %d active features" % len(max_active))

    # Shared and unique features
    min_set = set(min_active.cpu().numpy())
    max_set = set(max_active.cpu().numpy())
    shared = min_set & max_set
    min_unique = min_set - max_set
    max_unique = max_set - min_set
    print("  Shared: %d, MIN-unique: %d, MAX-unique: %d" % (
        len(shared), len(min_unique), len(max_unique)))

    # Top features by activation magnitude
    min_top_k = min_features.topk(10)
    max_top_k = max_features.topk(10)

    results = {
        'min_n_active': len(min_active),
        'max_n_active': len(max_active),
        'n_shared': len(shared),
        'n_min_unique': len(min_unique),
        'n_max_unique': len(max_unique),
        'min_top10_indices': min_top_k.indices.cpu().tolist(),
        'min_top10_values': [round(v, 4) for v in min_top_k.values.cpu().tolist()],
        'max_top10_indices': max_top_k.indices.cpu().tolist(),
        'max_top10_values': [round(v, 4) for v in max_top_k.values.cpu().tolist()],
    }

    # === Step 5: Feature ablation study ===
    print("\n  Feature ablation study...")
    ablation_results = {}

    # Test: reconstruct soul from SAE, inject, and evaluate
    with torch.no_grad():
        min_reconstructed = sae.decoder(min_features.unsqueeze(0)).squeeze(0)
        max_reconstructed = sae.decoder(max_features.unsqueeze(0)).squeeze(0)

    recon_min_acc = evaluate(model, tok, min_reconstructed, min_test, DEVICE)
    recon_max_acc = evaluate(model, tok, max_reconstructed, max_test, DEVICE)
    ablation_results['reconstructed'] = {
        'MIN': round(recon_min_acc, 4), 'MAX': round(recon_max_acc, 4)
    }
    print("  Reconstructed: MIN=%.0f%%, MAX=%.0f%%" % (recon_min_acc*100, recon_max_acc*100))

    # Ablate unique features (set to zero) and test
    print("  Ablating unique features...")
    for soul_name, soul_feat, unique_feat, test_d, soul_label in [
        ('MIN', min_features, min_unique, min_test, 'min'),
        ('MAX', max_features, max_unique, max_test, 'max'),
    ]:
        # Original accuracy from reconstructed
        orig_acc = recon_min_acc if soul_label == 'min' else recon_max_acc

        # Ablate unique features
        ablated = soul_feat.clone()
        for idx in unique_feat:
            ablated[idx] = 0
        with torch.no_grad():
            ablated_soul = sae.decoder(ablated.unsqueeze(0)).squeeze(0)
        ablated_acc = evaluate(model, tok, ablated_soul, test_d, DEVICE)
        ablation_results['%s_ablate_unique' % soul_label] = round(ablated_acc, 4)
        print("  %s ablate unique: %.0f%% (was %.0f%%)" % (
            soul_name, ablated_acc*100, orig_acc*100))

        # Ablate shared features
        shared_ablated = soul_feat.clone()
        for idx in shared:
            shared_ablated[idx] = 0
        with torch.no_grad():
            shared_ablated_soul = sae.decoder(shared_ablated.unsqueeze(0)).squeeze(0)
        shared_acc = evaluate(model, tok, shared_ablated_soul, test_d, DEVICE)
        ablation_results['%s_ablate_shared' % soul_label] = round(shared_acc, 4)
        print("  %s ablate shared: %.0f%%" % (soul_name, shared_acc*100))

    # Toggle: swap unique features between MIN and MAX
    print("\n  Feature swapping...")
    swapped_min = min_features.clone()
    swapped_max = max_features.clone()
    for idx in min_unique:
        swapped_min[idx] = 0
    for idx in max_unique:
        swapped_min[idx] = max_features[idx]
    for idx in max_unique:
        swapped_max[idx] = 0
    for idx in min_unique:
        swapped_max[idx] = min_features[idx]

    with torch.no_grad():
        swapped_min_soul = sae.decoder(swapped_min.unsqueeze(0)).squeeze(0)
        swapped_max_soul = sae.decoder(swapped_max.unsqueeze(0)).squeeze(0)

    swap_min_acc = evaluate(model, tok, swapped_min_soul, max_test, DEVICE)  # Should it now do MAX?
    swap_max_acc = evaluate(model, tok, swapped_max_soul, min_test, DEVICE)  # Should it now do MIN?
    ablation_results['swap_min_as_max'] = round(swap_min_acc, 4)
    ablation_results['swap_max_as_min'] = round(swap_max_acc, 4)
    print("  Swapped MIN (should do MAX): %.0f%%" % (swap_min_acc*100))
    print("  Swapped MAX (should do MIN): %.0f%%" % (swap_max_acc*100))

    results['ablation'] = ablation_results

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Feature activation comparison
    ax = axes[0, 0]
    n_show = 30  # Show top features
    combined_top = set(min_top_k.indices.cpu().numpy()) | set(max_top_k.indices.cpu().numpy())
    combined_top = sorted(combined_top)[:n_show]
    x = np.arange(len(combined_top))
    min_vals = [min_features[i].item() for i in combined_top]
    max_vals = [max_features[i].item() for i in combined_top]
    w = 0.35
    ax.bar(x - w/2, min_vals, w, label='MIN', color='#E91E63', alpha=0.8)
    ax.bar(x + w/2, max_vals, w, label='MAX', color='#2196F3', alpha=0.8)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Activation')
    ax.set_title('Top SAE Feature Activations', fontweight='bold')
    ax.legend()
    ax.set_xticks(x[::3])
    ax.set_xticklabels([str(combined_top[i]) for i in range(0, len(combined_top), 3)],
                       fontsize=7, rotation=45)

    # Panel 2: Venn diagram (as bars)
    ax = axes[0, 1]
    categories = ['MIN\nUnique', 'Shared', 'MAX\nUnique']
    counts = [len(min_unique), len(shared), len(max_unique)]
    colors = ['#E91E63', '#9C27B0', '#2196F3']
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', fontweight='bold', fontsize=14)
    ax.set_ylabel('Number of Active Features')
    ax.set_title('Feature Overlap: MIN vs MAX', fontweight='bold')

    # Panel 3: Ablation results
    ax = axes[1, 0]
    abl_names = ['Original\nMIN', 'Recon\nMIN', 'Ablate\nUnique', 'Ablate\nShared',
                 'Original\nMAX', 'Recon\nMAX', 'Ablate\nUnique', 'Ablate\nShared']
    abl_vals = [min_acc,
                ablation_results['reconstructed']['MIN'],
                ablation_results.get('min_ablate_unique', 0),
                ablation_results.get('min_ablate_shared', 0),
                max_acc,
                ablation_results['reconstructed']['MAX'],
                ablation_results.get('max_ablate_unique', 0),
                ablation_results.get('max_ablate_shared', 0)]
    abl_colors = ['#E91E63']*4 + ['#2196F3']*4
    bars = ax.bar(range(len(abl_names)), abl_vals, color=abl_colors,
                  edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(abl_names)))
    ax.set_xticklabels(abl_names, fontsize=7, rotation=45)
    ax.set_ylabel('Accuracy')
    ax.set_title('Feature Ablation Study', fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5)

    # Panel 4: Training curve + summary
    ax = axes[1, 1]
    ax.plot(sae_losses, color='#4CAF50', linewidth=1, alpha=0.7)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('SAE Loss')
    ax.set_title('SAE Training Loss\n(latent=%d, L1=%.3f)' % (LATENT_DIM, L1_COEFF),
                 fontweight='bold')
    ax.grid(alpha=0.3)

    # Add swap results as text
    ax2 = ax.twinx()
    ax2.set_ylabel('')
    ax2.set_yticks([])
    swap_text = ("Feature Swap Test:\n"
                 "MIN+MAX_unique -> MAX? %.0f%%\n"
                 "MAX+MIN_unique -> MIN? %.0f%%" % (
                     swap_min_acc*100, swap_max_acc*100))
    ax.text(0.02, 0.98, swap_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.suptitle('Phase 183: SAE Feature Decomposition\n'
                 '"What are the monosemantic building blocks of soul vectors?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase183_sae_decomposition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 183, 'name': 'sae_decomposition',
        'sae_config': {'latent_dim': LATENT_DIM, 'l1_coeff': L1_COEFF,
                       'n_training_states': len(hidden_states)},
        'soul_accuracy': {'MIN': round(min_acc, 4), 'MAX': round(max_acc, 4)},
        'results': results,
        'final_sae_loss': round(sae_losses[-1], 6),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase183_sae_decomposition.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P183 completed in %.0fs" % (time.time() - start))
    del model, sae; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
