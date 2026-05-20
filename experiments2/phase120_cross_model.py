# -*- coding: utf-8 -*-
"""
Phase 120: Cross-Model Rosetta
Can soul vectors be translated between different-sized models?

"If the soul transcends its body, it should work in any body."
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

def train_vec(model, tok, data, layer, device, seed, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in data:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(device)
            def inj(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

def eval_vec(model, tok, vec, data, layer, device):
    c = 0
    for p, e in data:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0, -1, :].argmax().item()).strip() == e: c += 1
    return c / len(data)

def learn_cross_T(vecs_src, vecs_tgt, device, epochs=1000):
    """Learn cross-model translation: src_dim -> tgt_dim."""
    src_dim = vecs_src.shape[1]
    tgt_dim = vecs_tgt.shape[1]
    T = torch.randn(src_dim, tgt_dim, device=device) * 0.01
    T.requires_grad_(True)
    opt = torch.optim.Adam([T], lr=0.005)
    for ep in range(epochs):
        pred = vecs_src @ T
        loss = torch.nn.functional.mse_loss(pred, vecs_tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 200 == 0:
            print("      epoch %d: loss=%.6f" % (ep, loss.item()))
    return T.detach()

def main():
    print("[P120] Cross-Model Rosetta")
    start = time.time()

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 2) =","2"),
                ("6, 3) =","3"),("2, 9) =","2")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 2) =","7"),
                ("6, 3) =","6"),("2, 9) =","9")]
    train_data = min_data[:5]
    test_data = min_data[5:]

    # ===== Model A: Qwen2.5-0.5B (hidden=896, layers=24) =====
    print("  Loading Model A (0.5B)...")
    model_a, tok_a = load_model('Qwen/Qwen2.5-0.5B', device=DEVICE, surgery=True)
    for p in model_a.parameters(): p.requires_grad = False
    layer_a = 8  # Proven injection layer for 0.5B

    N = 15  # vectors per function
    print("  Training %d MIN + %d MAX vectors on 0.5B..." % (N, N))
    min_vecs_a = []
    max_vecs_a = []
    for i in range(N):
        v = train_vec(model_a, tok_a, train_data, layer_a, DEVICE, seed=i*41)
        min_vecs_a.append(v)
        v = train_vec(model_a, tok_a, max_data[:5], layer_a, DEVICE, seed=i*41+500)
        max_vecs_a.append(v)

    # Verify on model A
    a_min_accs = [eval_vec(model_a, tok_a, v, min_data, layer_a, DEVICE) for v in min_vecs_a]
    a_max_accs = [eval_vec(model_a, tok_a, v, max_data, layer_a, DEVICE) for v in max_vecs_a]
    print("    0.5B MIN: %.0f%%, MAX: %.0f%%" % (
        np.mean(a_min_accs)*100, np.mean(a_max_accs)*100))

    # Free model A from GPU
    del model_a; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ===== Model B: Qwen2.5-1.5B (hidden=1536, layers=28) =====
    print("  Loading Model B (1.5B)...")
    model_b, tok_b = load_model('Qwen/Qwen2.5-1.5B', device=DEVICE, surgery=True)
    for p in model_b.parameters(): p.requires_grad = False
    layer_b = 10  # Scale layer proportionally: 8/24 * 28 ~ 9.3 -> 10

    print("  Training %d MIN + %d MAX vectors on 1.5B..." % (N, N))
    min_vecs_b = []
    max_vecs_b = []
    for i in range(N):
        v = train_vec(model_b, tok_b, train_data, layer_b, DEVICE, seed=i*41)
        min_vecs_b.append(v)
        v = train_vec(model_b, tok_b, max_data[:5], layer_b, DEVICE, seed=i*41+500)
        max_vecs_b.append(v)

    b_min_accs = [eval_vec(model_b, tok_b, v, min_data, layer_b, DEVICE) for v in min_vecs_b]
    b_max_accs = [eval_vec(model_b, tok_b, v, max_data, layer_b, DEVICE) for v in max_vecs_b]
    print("    1.5B MIN: %.0f%%, MAX: %.0f%%" % (
        np.mean(b_min_accs)*100, np.mean(b_max_accs)*100))

    # ===== Cross-Model Translation: 0.5B -> 1.5B =====
    print("  Learning cross-model translation (0.5B -> 1.5B)...")
    M_src = torch.stack(min_vecs_a[:10]).to(DEVICE)  # 10 for training
    M_tgt = torch.stack(min_vecs_b[:10]).to(DEVICE)
    T_cross = learn_cross_T(M_src, M_tgt, DEVICE, epochs=1000)

    # Test: translate remaining 5 MIN vectors from 0.5B -> 1.5B
    print("  Testing translated vectors on 1.5B...")
    translated_accs_min = []
    for i in range(10, N):
        v_src = min_vecs_a[i].to(DEVICE)
        v_translated = v_src @ T_cross
        acc = eval_vec(model_b, tok_b, v_translated, min_data, layer_b, DEVICE)
        translated_accs_min.append(acc)
    print("    Translated MIN on 1.5B: %.0f%%" % (np.mean(translated_accs_min)*100))

    # Also translate MAX (zero-shot: T learned only from MIN pairs)
    print("  Zero-shot MAX translation (T learned from MIN only)...")
    translated_accs_max = []
    M_max_src = torch.stack(max_vecs_a).to(DEVICE)
    for i in range(N):
        v_src = max_vecs_a[i].to(DEVICE)
        v_translated = v_src @ T_cross
        acc = eval_vec(model_b, tok_b, v_translated, max_data, layer_b, DEVICE)
        translated_accs_max.append(acc)
    print("    Translated MAX on 1.5B: %.0f%%" % (np.mean(translated_accs_max)*100))

    # SVD analysis of cross-model T
    _, S_cross, _ = torch.linalg.svd(T_cross)
    S_cross_np = S_cross.cpu().numpy()

    # Also learn reverse translation: 1.5B -> 0.5B
    print("  Learning reverse translation (1.5B -> 0.5B)...")
    T_reverse = learn_cross_T(M_tgt, M_src, DEVICE, epochs=1000)
    _, S_reverse, _ = torch.linalg.svd(T_reverse)

    # Round-trip: 0.5B -> 1.5B -> 0.5B
    T_roundtrip = T_cross @ T_reverse
    _, S_rt, _ = torch.linalg.svd(T_roundtrip)
    # Check if roundtrip ~ identity
    I_src = torch.eye(min(T_roundtrip.shape), device=DEVICE)
    rt_err = float(torch.norm(T_roundtrip[:I_src.shape[0], :I_src.shape[1]] - I_src).item())

    del model_b; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    output = {
        'phase': 120, 'name': 'cross_model_rosetta',
        'model_a': '0.5B (896d, 24L)', 'model_b': '1.5B (1536d, 28L)',
        'n_vectors': N,
        'native_0.5B_min': round(float(np.mean(a_min_accs)), 4),
        'native_0.5B_max': round(float(np.mean(a_max_accs)), 4),
        'native_1.5B_min': round(float(np.mean(b_min_accs)), 4),
        'native_1.5B_max': round(float(np.mean(b_max_accs)), 4),
        'translated_min_on_1.5B': round(float(np.mean(translated_accs_min)), 4),
        'translated_max_on_1.5B': round(float(np.mean(translated_accs_max)), 4),
        'cross_T_shape': list(T_cross.shape),
        'cross_T_svd_top10': [round(float(s), 4) for s in S_cross_np[:10]],
        'cross_T_cond': round(float(S_cross_np[0] / (S_cross_np[-1]+1e-8)), 2),
        'roundtrip_error': round(rt_err, 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase120_cross_model.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Accuracy comparison
    labels = ['0.5B\nMIN', '0.5B\nMAX', '1.5B\nMIN', '1.5B\nMAX',
              'Translated\nMIN', 'Translated\nMAX']
    vals = [np.mean(a_min_accs), np.mean(a_max_accs),
            np.mean(b_min_accs), np.mean(b_max_accs),
            np.mean(translated_accs_min), np.mean(translated_accs_max)]
    colors = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green',
              'tab:orange', 'tab:orange']
    bars = axes[0].bar(labels, vals, color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Native vs Translated Accuracy', fontweight='bold')
    for i, v in enumerate(vals):
        axes[0].text(i, v+0.03, '%.0f%%' % (v*100), ha='center',
                    fontweight='bold', fontsize=9)

    # 2. Cross-model SVD spectrum
    axes[1].semilogy(range(min(20, len(S_cross_np))), S_cross_np[:20],
                     'bo-', lw=2, label='0.5B -> 1.5B')
    S_rev_np = S_reverse.cpu().numpy()
    axes[1].semilogy(range(min(20, len(S_rev_np))), S_rev_np[:20],
                     'rs-', lw=2, label='1.5B -> 0.5B')
    axes[1].set_xlabel('SV Index')
    axes[1].set_ylabel('Magnitude (log)')
    axes[1].set_title('Cross-Model Translation SVD', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Translation matrix heatmap (top-left corner)
    T_np = T_cross.cpu().numpy()
    show_size = min(32, T_np.shape[0], T_np.shape[1])
    im = axes[2].imshow(T_np[:show_size, :show_size], cmap='RdBu_r',
                         aspect='auto', vmin=-0.1, vmax=0.1)
    axes[2].set_xlabel('Target dim (1.5B)')
    axes[2].set_ylabel('Source dim (0.5B)')
    axes[2].set_title('T_cross (top-left %dx%d)' % (show_size, show_size),
                      fontweight='bold')
    plt.colorbar(im, ax=axes[2])

    plt.suptitle('Phase 120: Cross-Model Rosetta\n'
                 '"If the soul transcends its body, it should work in any body"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase120_cross_model.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Completed in %.0fs" % (time.time()-start))
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
