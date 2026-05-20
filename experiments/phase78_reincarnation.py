# -*- coding: utf-8 -*-
"""
Phase 78: Neural Reincarnation
Transfer program vectors from CartPole to MountainCar.
Compress P77's controller via SVD (P64 method), inject as past life seed.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compress_vec(vec, k=10):
    v = vec.cpu().numpy().flatten()
    idx = np.argsort(np.abs(v))[::-1]
    out = np.zeros_like(v)
    out[idx[:k]] = v[idx[:k]]
    return torch.tensor(out, dtype=torch.float32)


def train_mountaincar(model, tok, device, init_prog=None, n_ep=60, lr=0.005):
    hs = model.config.hidden_size
    prompt = "compute:"
    proj_in = torch.randn(hs, 2, device=device) * 0.01
    proj_in.requires_grad_(True)
    proj_out = torch.randn(3, hs, device=device) * 0.01
    proj_out.requires_grad_(True)
    if init_prog is not None:
        pv = init_prog.clone().to(device); pv.requires_grad_(True)
    else:
        pv = torch.randn(hs, device=device) * 0.01; pv.requires_grad_(True)
    opt = torch.optim.Adam([proj_in, proj_out, pv], lr=lr)
    env = gym.make('MountainCar-v0')
    scores = []
    for ep in range(n_ep):
        obs, _ = env.reset(); total = 0; done = False; prev = None
        while not done:
            st = torch.tensor(obs, dtype=torch.float32, device=device)
            sv = proj_in @ st
            def inj_d(m, i, o, v=sv): return replace_last_token(o, v)
            def inj_p(m, i, o, v=pv): return replace_last_token(o, v)
            col = {}
            def rd(m, i, o): col['out'] = get_last_token(o)
            h1 = model.model.layers[2].register_forward_hook(inj_d)
            h2 = model.model.layers[8].register_forward_hook(inj_p)
            h3 = model.model.layers[22].register_forward_hook(rd)
            inp = tok(prompt, return_tensors='pt').to(device)
            out = model(**inp)
            h1.remove(); h2.remove(); h3.remove()
            al = proj_out @ col['out']
            ap = torch.softmax(al, dim=0)
            action = int(al.argmax().item())
            if prev is not None:
                pc = col['out'][:2]
                ac = sv - prev
                cl = -torch.nn.functional.cosine_similarity(
                    pc.unsqueeze(0), ac[:2].unsqueeze(0).detach()).mean()
                ent = -(ap * torch.log(ap + 1e-8)).sum()
                loss = cl - 0.01 * ent
                opt.zero_grad(); loss.backward(); opt.step()
            prev = sv.detach()
            obs, r, term, trunc, _ = env.step(action)
            total += r; done = term or trunc
        scores.append(total)
        if ep % 10 == 0: print(f"    ep={ep:3d}: score={total:.0f}")
    env.close()
    return scores


def main():
    print("[P78] Neural Reincarnation")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False
    hs = model.config.hidden_size

    # Load P77 past life
    p77_path = os.path.join(RESULTS_DIR, 'phase77_vectors.pt')
    if os.path.exists(p77_path):
        past = torch.load(p77_path, map_location='cpu', weights_only=True)['prog_vec']
        print(f"  Loaded P77 vector: norm={past.norm():.4f}")
    else:
        print("  WARNING: P77 not found, using synthetic"); past = torch.randn(hs) * 0.1

    # Compress
    compressed = compress_vec(past, k=10)
    csim = torch.nn.functional.cosine_similarity(
        past.unsqueeze(0), compressed.unsqueeze(0)).item()
    print(f"  Compression cos_sim: {csim:.4f}")

    # Train with past life
    print("\n  Training WITH past life...")
    scores_w = train_mountaincar(model, tok, DEVICE, init_prog=compressed, n_ep=60)

    # Train without
    print("\n  Training WITHOUT past life...")
    scores_wo = train_mountaincar(model, tok, DEVICE, init_prog=None, n_ep=60)

    # Analysis
    w = 10
    rm = lambda s: [np.mean(s[max(0,i-w):i+1]) for i in range(len(s))]
    rm_w, rm_wo = rm(scores_w), rm(scores_wo)

    output = {
        'phase': 78, 'name': 'neural_reincarnation',
        'compression_cos_sim': round(csim, 4),
        'with_past_life': {'scores': [float(s) for s in scores_w],
                          'mean_last10': round(float(np.mean(scores_w[-10:])), 2)},
        'tabula_rasa': {'scores': [float(s) for s in scores_wo],
                       'mean_last10': round(float(np.mean(scores_wo[-10:])), 2)},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase78_reincarnation.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(rm_w, 'b-', lw=2, label='With past life')
    axes[0].plot(rm_wo, 'r-', lw=2, label='Tabula rasa')
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Score (rolling mean)')
    axes[0].set_title('Learning Speed', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    labels = ['With\nPast Life', 'Tabula\nRasa']
    means = [np.mean(scores_w[-10:]), np.mean(scores_wo[-10:])]
    axes[1].bar(labels, means, color=['tab:blue','tab:red'], edgecolor='black')
    axes[1].set_ylabel('Mean Score (last 10)'); axes[1].set_title('Final', fontweight='bold')
    for i, m in enumerate(means):
        axes[1].text(i, m+2, f'{m:.1f}', ha='center', fontweight='bold')

    axes[2].bar(['Original','10D'], [past.norm().item(), compressed.norm().item()],
               color=['tab:green','tab:orange'], edgecolor='black')
    axes[2].set_ylabel('Norm')
    axes[2].set_title(f'Compression (cos={csim:.3f})', fontweight='bold')

    plt.suptitle('Phase 78: Neural Reincarnation\nPast life transfer across environments',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase78_reincarnation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  With past life: {np.mean(scores_w[-10:]):.1f}")
    print(f"  Tabula rasa: {np.mean(scores_wo[-10:]):.1f}")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
