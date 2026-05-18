# -*- coding: utf-8 -*-
"""
Phase 37: Rosetta HAL (Hardware Abstraction Layer)
Map raw binary/numeric sensor data into register space via contrastive learning.
Prove that raw numbers can be "grounded" into the model's latent space.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, torch.nn as nn, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SensorEncoder(nn.Module):
    """Lightweight encoder: raw sensor value -> register space."""
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, hidden_size),
        )
    def forward(self, x):
        return self.net(x)


def main():
    print("[P37] Rosetta HAL (Hardware Abstraction Layer)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size

    # Step 1: Extract ground-truth register vectors for numbers 0-9
    print("  Step 1: Extracting register vectors for numbers 0-9...")
    target_layer = 13  # Operand A register
    number_vecs = {}
    for n in range(10):
        prompts = [f"def f(): return {n} ="] * 3
        vecs = []
        for p in prompts:
            cap = [None]
            def capture(module, input, output):
                cap[0] = get_last_token(output)
            h = model.model.layers[target_layer].register_forward_hook(capture)
            inp = tok(p, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            vecs.append(cap[0])
        number_vecs[n] = torch.stack(vecs).mean(dim=0).detach()

    # Step 2: Train sensor encoder via contrastive learning
    print("  Step 2: Training sensor encoder...")
    encoder = SensorEncoder(hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    loss_fn = nn.CosineEmbeddingLoss()

    loss_history = []
    for epoch in range(200):
        total_loss = 0
        # Positive pairs: sensor value n -> register vec for n
        for n in range(10):
            sensor_val = torch.tensor([[float(n)]]).to(DEVICE)
            target_vec = number_vecs[n].unsqueeze(0)
            pred_vec = encoder(sensor_val)
            # Positive: should match
            loss_pos = loss_fn(pred_vec, target_vec, torch.ones(1).to(DEVICE))
            # Negative: should NOT match random other number
            neg_n = (n + np.random.randint(1, 10)) % 10
            neg_vec = number_vecs[neg_n].unsqueeze(0)
            loss_neg = loss_fn(pred_vec, neg_vec, -torch.ones(1).to(DEVICE))
            loss = loss_pos + loss_neg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(total_loss / 10)
        if epoch % 50 == 0:
            print(f"    Epoch {epoch}: loss={total_loss/10:.4f}")

    # Step 3: Test - can encoded sensor values be used as register values?
    print("\n  Step 3: Testing sensor-to-register alignment...")
    encoder.eval()
    align_results = {}
    for n in range(10):
        sensor_val = torch.tensor([[float(n)]]).to(DEVICE)
        with torch.no_grad():
            encoded = encoder(sensor_val)
        # Cosine similarity to all register vecs
        sims = {}
        for m in range(10):
            cos = torch.nn.functional.cosine_similarity(
                encoded, number_vecs[m].unsqueeze(0)
            ).item()
            sims[m] = round(cos, 4)
        best = max(sims, key=sims.get)
        align_results[n] = {'best_match': best, 'correct': best == n, 'sim': sims[best]}
        if n < 5:
            print(f"    Sensor {n} -> best match: {best} (sim={sims[best]:.3f}) "
                  f"{'OK' if best==n else 'X'}")
    align_acc = sum(1 for r in align_results.values() if r['correct']) / len(align_results)
    print(f"    Alignment accuracy: {align_acc:.0%}")

    # Step 4: DMA with encoded sensor values
    print("\n  Step 4: DMA with sensor-encoded vectors...")
    from utils import replace_last_token
    dma_tests = [(3, 7), (5, 2), (8, 1), (4, 6)]
    dma_correct = 0
    dma_total = 0
    for a, b in dma_tests:
        # Encode sensor values
        with torch.no_grad():
            vec_a = encoder(torch.tensor([[float(a)]]).to(DEVICE)).squeeze()
        # Inject at L13 (Operand A register) during min computation
        prompt = f"def f(): return min(x, {b}) ="
        def inject(module, input, output, v=vec_a):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        expected = str(min(a, b))
        dma_total += 1
        if pred == expected:
            dma_correct += 1
        print(f"    min({a}, {b}): pred={pred}, expected={expected}")
    dma_acc = dma_correct / dma_total

    # Save
    output = {
        'phase': 37, 'name': 'rosetta_hal',
        'alignment_accuracy': round(align_acc, 4),
        'dma_with_sensor': round(dma_acc, 4),
        'loss_history': [round(l, 4) for l in loss_history[::10]],
        'align_results': {str(k): v for k, v in align_results.items()},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase37_hal.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Sensor Encoder Training', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Alignment matrix
    sim_matrix = np.zeros((10, 10))
    encoder.eval()
    for n in range(10):
        with torch.no_grad():
            enc = encoder(torch.tensor([[float(n)]]).to(DEVICE))
        for m in range(10):
            sim_matrix[n, m] = torch.nn.functional.cosine_similarity(
                enc, number_vecs[m].unsqueeze(0)).item()
    im = axes[1].imshow(sim_matrix, cmap='YlOrRd', vmin=-1, vmax=1)
    axes[1].set_xlabel('Register Vec (number)'); axes[1].set_ylabel('Sensor Value')
    axes[1].set_title('Sensor-Register Alignment', fontweight='bold')
    plt.colorbar(im, ax=axes[1])

    axes[2].bar(['Alignment\nAccuracy', 'DMA with\nSensor'], [align_acc, dma_acc],
                color=['tab:green', 'tab:blue'], edgecolor='black')
    axes[2].set_ylim(0, 1.1); axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Rosetta HAL Results', fontweight='bold')
    for i, v in enumerate([align_acc, dma_acc]):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=14)

    plt.suptitle('Phase 37: Rosetta HAL\nRaw sensor values -> register space via contrastive learning',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase37_hal.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model, encoder; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
