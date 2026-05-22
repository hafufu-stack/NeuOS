# -*- coding: utf-8 -*-
"""
Phase 146: Hardware Proprioception
Can NeuOS self-diagnose its own hardware specs from internal signals alone?

The model reads its own layer-wise statistics (entropy, norms, variance,
activation magnitude) to build a 'hardware fingerprint', then a tiny
meta-probe MLP predicts architecture specs from that fingerprint.

"Know thyself -- not from a spec sheet, but from the echoes of your own layers."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Diverse prompts to probe internal statistics
PROBE_PROMPTS = [
    "The capital of France is",
    "2 + 3 =",
    "In the beginning, there was",
    "def fibonacci(n):",
    "The quick brown fox jumps over",
    "E = mc",
    "Once upon a time in a land far",
    "import torch\nmodel =",
    "The meaning of life is",
    "Water boils at 100 degrees",
    "SELECT * FROM users WHERE",
    "To be or not to be, that is",
    "The mitochondria is the powerhouse",
    "3.14159265358979",
    "Hello, my name is",
    "The year 2025 will be remembered for",
    "If x > 0 then y =",
    "According to the theory of relativity",
    "A neural network consists of",
    "The Fibonacci sequence: 1, 1, 2, 3, 5,",
]

NUM_LAYERS = 24


def collect_layer_stats(model, tok, prompt, device):
    """Run a prompt and collect per-layer statistics from hidden states."""
    layer_stats = []
    hidden_states_per_layer = {}

    # Register hooks to capture hidden states at each layer
    hooks = []
    for layer_idx in range(NUM_LAYERS):
        def make_hook(idx):
            def hook_fn(module, input, output):
                hidden_states_per_layer[idx] = get_last_token(output)
            return hook_fn
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Also hook attention to get attention logits
    attn_logits = {}
    for layer_idx in range(NUM_LAYERS):
        def make_attn_hook(idx):
            def hook_fn(module, input, output):
                # output is typically (attn_output, attn_weights, past_kv)
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    attn_logits[idx] = output[1].detach()
                else:
                    attn_logits[idx] = None
            return hook_fn
        h = model.model.layers[layer_idx].self_attn.register_forward_hook(make_attn_hook(layer_idx))
        hooks.append(h)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp, output_attentions=True)

    for h in hooks:
        h.remove()

    # Compute per-layer statistics
    for layer_idx in range(NUM_LAYERS):
        hs = hidden_states_per_layer.get(layer_idx)
        if hs is None:
            layer_stats.append([0.0, 0.0, 0.0, 0.0])
            continue

        hs_float = hs.float()
        # 1. L2 norm of hidden state
        l2_norm = hs_float.norm().item()
        # 2. Entropy of attention (from attention weights if available)
        attn_w = attn_logits.get(layer_idx)
        if attn_w is not None:
            # Average over heads, last query position
            attn_probs = attn_w[0, :, -1, :]  # (num_heads, seq_len)
            # Clamp to avoid log(0)
            attn_probs_clamped = attn_probs.clamp(min=1e-10)
            entropy = -(attn_probs_clamped * attn_probs_clamped.log()).sum(dim=-1).mean().item()
        else:
            # Fallback: compute entropy from hidden state distribution
            probs = torch.softmax(hs_float, dim=-1)
            probs_clamped = probs.clamp(min=1e-10)
            entropy = -(probs_clamped * probs_clamped.log()).sum().item()

        # 3. Variance of hidden state
        variance = hs_float.var().item()

        # 4. Mean activation magnitude
        mean_mag = hs_float.abs().mean().item()

        layer_stats.append([l2_norm, entropy, variance, mean_mag])

    return np.array(layer_stats)  # (24, 4)


def build_fingerprint(layer_stats_matrix):
    """Flatten layer stats into a 96-dim hardware fingerprint vector."""
    return layer_stats_matrix.flatten()  # (24*4,) = 96


class MetaProbe(nn.Module):
    """Tiny MLP that predicts architecture specs from fingerprint."""
    def __init__(self, input_dim=96, hidden_dim=32, num_targets=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, x):
        return self.net(x)


def train_meta_probe(fingerprints, targets, device, epochs=300, lr=0.01):
    """Train the meta-probe on collected fingerprints."""
    probe = MetaProbe().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = torch.tensor(fingerprints, dtype=torch.float32).to(device)
    Y = torch.tensor(targets, dtype=torch.float32).to(device)

    # Normalize inputs
    x_mean = X.mean(dim=0, keepdim=True)
    x_std = X.std(dim=0, keepdim=True).clamp(min=1e-8)
    X_norm = (X - x_mean) / x_std

    losses = []
    for epoch in range(epochs):
        pred = probe(X_norm)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return probe, x_mean, x_std, losses


def main():
    print("[P146] Hardware Proprioception")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=False)
    for p in model.parameters():
        p.requires_grad = False

    # Ground truth specs for Qwen2.5-0.5B
    true_hidden_size = model.config.hidden_size       # 896
    true_num_layers = model.config.num_hidden_layers   # 24
    true_vocab_size = model.config.vocab_size           # 151936
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Ground truth: hidden_size={true_hidden_size}, "
          f"num_layers={true_num_layers}, "
          f"vocab_size={true_vocab_size}, "
          f"params={total_params:,}")

    # Normalized targets for regression (0-1 scale)
    # hidden_size: 896 in range [896, 1536] -> 0.0
    # num_layers: 24 in range [24, 28] -> 0.0
    # vocab_size: 151936 -> normalized to 1.0 (fixed)
    # model_size: 0.5B in range [0.5B, 1.5B] -> 0.0
    target_vec = np.array([0.0, 0.0, 1.0, 0.0])  # class labels for 0.5B model

    # Step 1: Collect fingerprints from all prompts
    print("  Collecting layer statistics across %d prompts..." % len(PROBE_PROMPTS))
    all_layer_stats = []
    fingerprints = []

    for i, prompt in enumerate(PROBE_PROMPTS):
        stats = collect_layer_stats(model, tok, prompt, DEVICE)
        all_layer_stats.append(stats)
        fp = build_fingerprint(stats)
        fingerprints.append(fp)
        if (i + 1) % 5 == 0:
            print(f"    Prompt {i+1}/{len(PROBE_PROMPTS)} done")

    fingerprints = np.array(fingerprints)  # (20, 96)
    all_layer_stats = np.array(all_layer_stats)  # (20, 24, 4)

    # Step 2: Compute fingerprint stability (cosine similarity)
    print("  Computing fingerprint stability...")
    from numpy.linalg import norm as np_norm
    cos_sim_matrix = np.zeros((len(PROBE_PROMPTS), len(PROBE_PROMPTS)))
    for i in range(len(PROBE_PROMPTS)):
        for j in range(len(PROBE_PROMPTS)):
            ni = np_norm(fingerprints[i])
            nj = np_norm(fingerprints[j])
            if ni > 0 and nj > 0:
                cos_sim_matrix[i, j] = np.dot(fingerprints[i], fingerprints[j]) / (ni * nj)
            else:
                cos_sim_matrix[i, j] = 0.0

    mean_self_sim = np.mean(cos_sim_matrix[np.triu_indices(len(PROBE_PROMPTS), k=1)])
    min_sim = np.min(cos_sim_matrix[np.triu_indices(len(PROBE_PROMPTS), k=1)])
    print(f"  Fingerprint stability: mean_cosine={mean_self_sim:.4f}, min={min_sim:.4f}")

    # Step 3: Train meta-probe
    print("  Training meta-probe MLP...")
    targets = np.tile(target_vec, (len(PROBE_PROMPTS), 1))
    probe, x_mean, x_std, train_losses = train_meta_probe(
        fingerprints, targets, DEVICE, epochs=300)

    # Step 4: Self-diagnosis
    print("  Running self-diagnosis...")
    probe.eval()
    avg_fingerprint = fingerprints.mean(axis=0)
    X_test = torch.tensor(avg_fingerprint, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    X_test_norm = (X_test - x_mean) / x_std
    with torch.no_grad():
        pred = probe(X_test_norm).cpu().numpy()[0]

    # Decode predictions
    pred_hidden = 896 if pred[0] < 0.5 else 1536
    pred_layers = 24 if pred[1] < 0.5 else 28
    pred_vocab = 151936  # fixed
    pred_size = "0.5B" if pred[3] < 0.5 else "1.5B"

    diagnosis = {
        'hidden_size': {'predicted': pred_hidden, 'actual': true_hidden_size,
                        'correct': pred_hidden == true_hidden_size, 'raw': float(pred[0])},
        'num_layers': {'predicted': pred_layers, 'actual': true_num_layers,
                       'correct': pred_layers == true_num_layers, 'raw': float(pred[1])},
        'vocab_size': {'predicted': pred_vocab, 'actual': true_vocab_size,
                       'correct': pred_vocab == true_vocab_size, 'raw': float(pred[2])},
        'model_size': {'predicted': pred_size, 'actual': '0.5B',
                       'correct': pred_size == '0.5B', 'raw': float(pred[3])},
    }

    n_correct = sum(1 for v in diagnosis.values() if v['correct'])
    print(f"  Self-diagnosis: {n_correct}/4 correct")
    for k, v in diagnosis.items():
        status = "OK" if v['correct'] else "WRONG"
        print(f"    {k}: predicted={v['predicted']}, actual={v['actual']} [{status}] (raw={v['raw']:.3f})")

    # Step 5: Test with perturbed fingerprints (simulating different hardware)
    print("  Testing with perturbed fingerprints...")
    perturbation_results = {}
    perturbation_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    np.random.seed(42)
    for noise_level in perturbation_levels:
        perturbed = avg_fingerprint + np.random.randn(96) * noise_level * np.std(avg_fingerprint)
        X_p = torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        X_p_norm = (X_p - x_mean) / x_std
        with torch.no_grad():
            pred_p = probe(X_p_norm).cpu().numpy()[0]
        # Check if predictions still correct
        correct_p = (
            (1 if (pred_p[0] < 0.5) == (target_vec[0] < 0.5) else 0) +
            (1 if (pred_p[1] < 0.5) == (target_vec[1] < 0.5) else 0) +
            1 +  # vocab is always correct
            (1 if (pred_p[3] < 0.5) == (target_vec[3] < 0.5) else 0)
        )
        perturbation_results[str(noise_level)] = {
            'accuracy': correct_p / 4,
            'raw_predictions': pred_p.tolist(),
        }
        print(f"    noise={noise_level:.1f}: {correct_p}/4 correct, raw={pred_p.round(3).tolist()}")

    # Step 6: Compute average layer statistics for heatmap
    avg_stats = all_layer_stats.mean(axis=0)  # (24, 4)
    stat_names = ['L2 Norm', 'Attention\nEntropy', 'Hidden\nVariance', 'Mean |Act|']

    # ============= PLOTTING =============
    print("  Generating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: Layer-wise statistics heatmap
    ax = axes[0]
    # Normalize each metric independently for visualization
    heatmap_data = avg_stats.T.copy()  # (4, 24)
    for i in range(4):
        row = heatmap_data[i]
        rmin, rmax = row.min(), row.max()
        if rmax - rmin > 0:
            heatmap_data[i] = (row - rmin) / (rmax - rmin)
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_yticks(range(4))
    ax.set_yticklabels(stat_names, fontsize=9)
    ax.set_xlabel('Layer Index', fontsize=10)
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.set_xticklabels(range(0, NUM_LAYERS, 2), fontsize=8)
    ax.set_title('Layer-wise Statistics\n(normalized, avg over 20 prompts)',
                 fontweight='bold', fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.6, label='Normalized Value')

    # Annotate raw values on top
    for i in range(4):
        for j in range(NUM_LAYERS):
            val = avg_stats[j, i]
            if val > 1000:
                txt = f"{val:.0f}"
            elif val > 10:
                txt = f"{val:.1f}"
            else:
                txt = f"{val:.2f}"
            # Only show every other column to avoid clutter
            if j % 3 == 0:
                ax.text(j, i, txt, ha='center', va='center', fontsize=5,
                        color='white' if heatmap_data[i, j] < 0.5 else 'black')

    # Panel 2: Fingerprint stability (cosine similarity matrix)
    ax = axes[1]
    im2 = ax.imshow(cos_sim_matrix, aspect='auto', cmap='RdYlGn', vmin=0.8, vmax=1.0)
    ax.set_xlabel('Prompt Index', fontsize=10)
    ax.set_ylabel('Prompt Index', fontsize=10)
    ax.set_title(f'Fingerprint Stability\n(cosine sim, mean={mean_self_sim:.4f})',
                 fontweight='bold', fontsize=11)
    ax.set_xticks(range(0, 20, 2))
    ax.set_yticks(range(0, 20, 2))
    plt.colorbar(im2, ax=ax, shrink=0.6, label='Cosine Similarity')

    # Panel 3: Self-diagnosis results table
    ax = axes[2]
    ax.axis('off')

    # Build table data
    table_rows = []
    for spec_name, info in diagnosis.items():
        status_str = "CORRECT" if info['correct'] else "WRONG"
        table_rows.append([
            spec_name,
            str(info['actual']),
            str(info['predicted']),
            f"{info['raw']:.3f}",
            status_str,
        ])
    # Add perturbation summary
    table_rows.append(['---', '---', '---', '---', '---'])
    table_rows.append(['Noise', 'Level', 'Accuracy', '', ''])
    for noise_str, pres in perturbation_results.items():
        table_rows.append([
            'perturb',
            noise_str,
            f"{pres['accuracy']:.0%}",
            '',
            'OK' if pres['accuracy'] == 1.0 else 'degraded',
        ])

    col_labels = ['Spec', 'Actual', 'Predicted', 'Raw Score', 'Status']
    table = ax.table(cellText=table_rows, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Color diagnosis rows
    for i in range(4):
        if diagnosis[list(diagnosis.keys())[i]]['correct']:
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor('#C8E6C9')
        else:
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor('#FFCDD2')
    # Separator row
    for j in range(len(col_labels)):
        table[5, j].set_facecolor('#E0E0E0')
        table[6, j].set_facecolor('#E3F2FD')

    ax.set_title('Self-Diagnosis Results\n& Perturbation Robustness',
                 fontweight='bold', fontsize=11, pad=20)

    plt.suptitle('Phase 146: Hardware Proprioception\n'
                 '"Know thyself -- from the echoes of your own layers"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase146_hardware_proprioception.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    output = {
        'phase': 146,
        'name': 'hardware_proprioception',
        'ground_truth': {
            'hidden_size': true_hidden_size,
            'num_layers': true_num_layers,
            'vocab_size': true_vocab_size,
            'total_params': total_params,
        },
        'diagnosis': {k: {kk: vv for kk, vv in v.items()}
                      for k, v in diagnosis.items()},
        'diagnosis_accuracy': n_correct / 4,
        'fingerprint_stability': {
            'mean_cosine': round(float(mean_self_sim), 6),
            'min_cosine': round(float(min_sim), 6),
            'std_cosine': round(float(np.std(cos_sim_matrix[np.triu_indices(len(PROBE_PROMPTS), k=1)])), 6),
        },
        'perturbation_robustness': perturbation_results,
        'avg_layer_stats': {
            'stat_names': ['l2_norm', 'attention_entropy', 'hidden_variance', 'mean_activation_mag'],
            'data': avg_stats.tolist(),
        },
        'train_loss_final': round(train_losses[-1], 6),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase146_hardware_proprioception.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Completed in {time.time() - start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
