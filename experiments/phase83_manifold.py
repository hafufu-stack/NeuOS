# -*- coding: utf-8 -*-
"""
Phase 83: Hardware-Software Equivalence (Environment Manifold)
Prove that NeuOS treats "hardware bugs" and "game rule changes"
identically. Adaptation vectors for both land on the same manifold.

Test: CartPole with (A) action delay (hardware bug) vs
(B) modified gravity (physics change). Are adaptation vectors
in the same space?

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


class NoisyCartPole:
    """CartPole wrapper: action is randomly flipped with probability p."""
    def __init__(self, noise_prob=0.2):
        self.env = gym.make('CartPole-v1')
        self.noise_prob = noise_prob
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.random() < self.noise_prob:
            action = 1 - action  # Flip action (hardware bug)
        return self.env.step(action)

    def close(self):
        self.env.close()


class HeavyCartPole:
    """CartPole wrapper: gravity is multiplied (game rule change)."""
    def __init__(self, gravity_mult=1.5):
        self.env = gym.make('CartPole-v1')
        self.env.unwrapped.gravity *= gravity_mult
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def train_adapter(model, tok, env, device, n_episodes=60, lr=0.005):
    """Train an adaptation vector for a modified environment."""
    hs = model.config.hidden_size
    prompt = "compute:"
    proj_in = torch.randn(hs, 4, device=device) * 0.01; proj_in.requires_grad_(True)
    proj_out = torch.randn(2, hs, device=device) * 0.01; proj_out.requires_grad_(True)
    prog_vec = torch.randn(hs, device=device) * 0.01; prog_vec.requires_grad_(True)
    opt = torch.optim.Adam([proj_in, proj_out, prog_vec], lr=lr)
    scores = []

    for ep in range(n_episodes):
        obs, _ = env.reset(); total = 0; done = False; prev = None
        while not done:
            st = torch.tensor(obs, dtype=torch.float32, device=device)
            sv = proj_in @ st
            col = {}
            def inj_d(m, i, o, v=sv): return replace_last_token(o, v)
            def inj_p(m, i, o, v=prog_vec): return replace_last_token(o, v)
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
                cl = -torch.nn.functional.cosine_similarity(
                    col['out'][:4].unsqueeze(0), (sv-prev)[:4].unsqueeze(0).detach()).mean()
                ent = -(ap * torch.log(ap + 1e-8)).sum()
                loss = cl - 0.02 * ent
                opt.zero_grad(); loss.backward(); opt.step()
            prev = sv.detach()
            obs, r, term, trunc, _ = env.step(action)
            total += r; done = term or trunc
        scores.append(total)
        if ep % 15 == 0:
            print(f"      ep={ep:3d}: score={total:.0f}")

    return proj_in.detach(), proj_out.detach(), prog_vec.detach(), scores


def main():
    print("[P83] Hardware-Software Equivalence")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    # Step 1: Normal CartPole (baseline)
    print("  Step 1: Normal CartPole...")
    env_normal = gym.make('CartPole-v1')
    pi_n, po_n, pv_n, scores_n = train_adapter(
        model, tok, env_normal, DEVICE, n_episodes=60)
    env_normal.close()

    # Step 2: Noisy CartPole (hardware bug: 20% action flip)
    print("\n  Step 2: Noisy CartPole (hardware bug: 20% action flip)...")
    env_noisy = NoisyCartPole(noise_prob=0.2)
    pi_h, po_h, pv_h, scores_h = train_adapter(
        model, tok, env_noisy, DEVICE, n_episodes=60)
    env_noisy.close()

    # Step 3: Heavy CartPole (game rule: 1.5x gravity)
    print("\n  Step 3: Heavy CartPole (rule change: 1.5x gravity)...")
    env_heavy = HeavyCartPole(gravity_mult=1.5)
    pi_g, po_g, pv_g, scores_g = train_adapter(
        model, tok, env_heavy, DEVICE, n_episodes=60)
    env_heavy.close()

    # Step 4: Compare adaptation vectors
    print("\n  Step 4: Comparing adaptation vectors...")
    # Program vectors are the most interesting (they encode the strategy)
    cs_nh = torch.nn.functional.cosine_similarity(
        pv_n.unsqueeze(0), pv_h.unsqueeze(0)).item()
    cs_ng = torch.nn.functional.cosine_similarity(
        pv_n.unsqueeze(0), pv_g.unsqueeze(0)).item()
    cs_hg = torch.nn.functional.cosine_similarity(
        pv_h.unsqueeze(0), pv_g.unsqueeze(0)).item()
    print(f"    Normal vs HW-bug: cos_sim={cs_nh:.4f}")
    print(f"    Normal vs Rule-change: cos_sim={cs_ng:.4f}")
    print(f"    HW-bug vs Rule-change: cos_sim={cs_hg:.4f}")

    # Adaptation delta vectors (difference from normal)
    delta_hw = pv_h - pv_n  # Hardware adaptation
    delta_game = pv_g - pv_n  # Game rule adaptation
    cs_deltas = torch.nn.functional.cosine_similarity(
        delta_hw.unsqueeze(0), delta_game.unsqueeze(0)).item()
    print(f"    Delta(HW) vs Delta(Game): cos_sim={cs_deltas:.4f}")

    # PCA of all three vectors
    all_vecs = torch.stack([pv_n, pv_h, pv_g]).cpu().numpy()
    mean_vec = all_vecs.mean(axis=0)
    centered = all_vecs - mean_vec
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:2].T
    var_explained = (S[:2]**2).sum() / (S**2).sum()
    print(f"    PCA variance explained (2D): {var_explained:.2%}")

    output = {
        'phase': 83, 'name': 'hw_sw_equivalence',
        'similarities': {
            'normal_vs_hw_bug': round(cs_nh, 4),
            'normal_vs_rule_change': round(cs_ng, 4),
            'hw_bug_vs_rule_change': round(cs_hg, 4),
            'delta_hw_vs_delta_game': round(cs_deltas, 4),
        },
        'pca_variance_explained': round(float(var_explained), 4),
        'scores': {
            'normal': [float(s) for s in scores_n],
            'hw_bug': [float(s) for s in scores_h],
            'rule_change': [float(s) for s in scores_g],
        },
        'interpretation': (
            'Positive delta cos_sim = HW and SW adaptation share the same manifold. '
            'NeuOS does not distinguish between hardware bugs and game rule changes.'
        ),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase83_manifold.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Learning curves
    w = 10
    rm = lambda s: [np.mean(s[max(0,i-w):i+1]) for i in range(len(s))]
    axes[0].plot(rm(scores_n), 'b-', lw=2, label='Normal')
    axes[0].plot(rm(scores_h), 'r-', lw=2, label='HW bug (20% flip)')
    axes[0].plot(rm(scores_g), 'g-', lw=2, label='Rule (1.5x grav)')
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Score')
    axes[0].set_title('Adaptation in Modified Environments', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # PCA scatter
    labels_pca = ['Normal', 'HW Bug', 'Rule Change']
    colors_pca = ['blue', 'red', 'green']
    for i, (label, color) in enumerate(zip(labels_pca, colors_pca)):
        axes[1].scatter(projected[i, 0], projected[i, 1], c=color,
                       s=200, label=label, zorder=5, edgecolors='black')
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
    axes[1].set_title(f'Adaptation Manifold (PCA)\n'
                     f'{var_explained:.0%} variance explained', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Similarity matrix
    sim_mat = np.array([
        [1.0, cs_nh, cs_ng],
        [cs_nh, 1.0, cs_hg],
        [cs_ng, cs_hg, 1.0]
    ])
    im = axes[2].imshow(sim_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_xticks(range(3)); axes[2].set_yticks(range(3))
    axes[2].set_xticklabels(['Normal', 'HW', 'Rule'], fontsize=9)
    axes[2].set_yticklabels(['Normal', 'HW', 'Rule'], fontsize=9)
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f'{sim_mat[i,j]:.3f}', ha='center',
                        va='center', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    axes[2].set_title('Program Vector Similarity', fontweight='bold')

    plt.suptitle('Phase 83: HW-SW Equivalence\n'
                '"Hardware bugs and game rules are the same to NeuOS"',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase83_manifold.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
