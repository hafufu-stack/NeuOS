# -*- coding: utf-8 -*-
"""
Phase 82: Intrinsic Curiosity Vector
Solve MountainCar using curiosity (prediction error as reward).
A prediction submodule learns to predict next-state. When prediction
fails (surprise), NeuOS gets "pleasure" = positive gradient.
This drives exploration into unseen states.

Inspired by: Random Network Distillation (Burda et al., 2018)

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


def get_last_token_live(output):
    """Like get_last_token but WITHOUT detach - keeps gradient graph alive."""
    tensor = output[0] if isinstance(output, tuple) else output
    if tensor.dim() == 3:
        return tensor[0, -1, :]
    elif tensor.dim() == 2:
        return tensor[-1, :]
    return tensor


def train_curious(model, tok, device, n_episodes=80, lr=0.005):
    """Train MountainCar controller with intrinsic curiosity.

    v2 fixes:
    1. Stochastic action selection (sample from softmax) for exploration
    2. State visitation count for novelty bonus
    3. REINFORCE with curiosity reward per episode
    4. Live gradients through model hooks
    """
    hs = model.config.hidden_size
    prompt = "compute:"

    # Controller parameters
    proj_in = torch.randn(hs, 2, device=device) * 0.01
    proj_in.requires_grad_(True)
    proj_out = torch.randn(3, hs, device=device) * 0.01
    proj_out.requires_grad_(True)
    prog_vec = torch.randn(hs, device=device) * 0.01
    prog_vec.requires_grad_(True)

    # Prediction submodule: predict next state from current output
    pred_weight = torch.randn(2, hs, device=device) * 0.01
    pred_weight.requires_grad_(True)

    opt_ctrl = torch.optim.Adam([proj_in, proj_out, prog_vec], lr=lr)
    opt_pred = torch.optim.Adam([pred_weight], lr=lr * 2)

    env = gym.make('MountainCar-v0')
    scores = []; curiosity_hist = []; pred_error_hist = []

    # State visitation count (discretized position x velocity grid)
    visit_counts = {}

    def discretize_state(obs):
        pos_bin = int((obs[0] + 1.2) / 0.1)  # position: -1.2 to 0.6
        vel_bin = int((obs[1] + 0.07) / 0.007)  # velocity: -0.07 to 0.07
        return (pos_bin, vel_bin)

    for ep in range(n_episodes):
        obs, _ = env.reset(); total = 0; done = False
        ep_curiosity = []; ep_pred_err = []; steps = 0
        log_probs = []; curiosity_rewards = []
        prev_out_vec = None; prev_obs = None
        max_pos = obs[0]  # track rightmost position reached

        while not done:
            st = torch.tensor(obs, dtype=torch.float32, device=device)
            sv = proj_in @ st  # LIVE gradient!
            col = {}
            def inj_d(m, i, o, v=sv): return replace_last_token(o, v)
            def inj_p(m, i, o, v=prog_vec): return replace_last_token(o, v)
            def rd(m, i, o): col['out'] = get_last_token_live(o)  # LIVE!
            h1 = model.model.layers[2].register_forward_hook(inj_d)
            h2 = model.model.layers[8].register_forward_hook(inj_p)
            h3 = model.model.layers[22].register_forward_hook(rd)
            inp = tok(prompt, return_tensors='pt').to(device)
            out = model(**inp)
            h1.remove(); h2.remove(); h3.remove()

            out_vec = col['out']  # LIVE gradient!
            al = proj_out @ out_vec  # differentiable!
            ap = torch.softmax(al / 0.5, dim=0)  # temperature=0.5 for exploration

            # SAMPLE action (stochastic exploration!)
            dist = torch.distributions.Categorical(ap)
            action_t = dist.sample()
            action = action_t.item()
            log_prob = dist.log_prob(action_t)
            log_probs.append(log_prob)

            # Curiosity components:
            curiosity_r = 0.0

            # 1. State visitation novelty
            state_key = discretize_state(obs)
            visit_counts[state_key] = visit_counts.get(state_key, 0) + 1
            novelty = 1.0 / np.sqrt(visit_counts[state_key])
            curiosity_r += novelty

            # 2. Prediction error (only after first step)
            if prev_out_vec is not None and prev_obs is not None:
                predicted_state = pred_weight @ prev_out_vec
                actual_state = torch.tensor(prev_obs, dtype=torch.float32,
                                           device=device)
                pred_error = (predicted_state - actual_state).pow(2).sum()

                # Update predictor to minimize prediction error
                opt_pred.zero_grad()
                pred_error.backward()
                opt_pred.step()

                curiosity_r += min(pred_error.item(), 5.0)
                ep_pred_err.append(pred_error.item())

            # 3. Position progress bonus (reached new rightmost position)
            if obs[0] > max_pos:
                curiosity_r += 2.0 * (obs[0] - max_pos)
                max_pos = obs[0]

            curiosity_rewards.append(curiosity_r)
            ep_curiosity.append(curiosity_r)

            prev_out_vec = out_vec.detach()
            prev_obs = obs.copy()
            obs, r, term, trunc, _ = env.step(action)
            total += r; done = term or trunc; steps += 1

        # REINFORCE update: use curiosity as reward
        if len(log_probs) > 0:
            # Discount curiosity rewards
            gamma = 0.99
            returns = []
            G = 0
            for cr in reversed(curiosity_rewards):
                G = cr + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, device=device)
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Policy gradient
            policy_loss = 0
            for lp, ret in zip(log_probs, returns):
                policy_loss -= lp * ret  # maximize curiosity
            policy_loss = policy_loss / len(log_probs)

            opt_ctrl.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_([proj_in, proj_out, prog_vec], 1.0)
            opt_ctrl.step()

        scores.append(total)
        mc = np.mean(ep_curiosity) if ep_curiosity else 0
        mp = np.mean(ep_pred_err) if ep_pred_err else 0
        curiosity_hist.append(mc)
        pred_error_hist.append(mp)

        if ep % 10 == 0 or ep == n_episodes - 1:
            print(f"    ep={ep:3d}: score={total:.0f}, "
                  f"curiosity={mc:.4f}, maxpos={max_pos:.3f}")

    env.close()
    return (proj_in.detach(), proj_out.detach(), prog_vec.detach(),
            pred_weight.detach(), scores, curiosity_hist, pred_error_hist)


def train_no_curiosity(model, tok, device, n_episodes=80, lr=0.005):
    """Baseline: train MountainCar without curiosity (P78-style)."""
    hs = model.config.hidden_size
    prompt = "compute:"
    proj_in = torch.randn(hs, 2, device=device) * 0.01; proj_in.requires_grad_(True)
    proj_out = torch.randn(3, hs, device=device) * 0.01; proj_out.requires_grad_(True)
    pv = torch.randn(hs, device=device) * 0.01; pv.requires_grad_(True)
    opt = torch.optim.Adam([proj_in, proj_out, pv], lr=lr)
    env = gym.make('MountainCar-v0')
    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(); total = 0; done = False; prev = None
        while not done:
            st = torch.tensor(obs, dtype=torch.float32, device=device)
            sv = proj_in @ st
            col = {}
            def inj_d(m, i, o, v=sv): return replace_last_token(o, v)
            def inj_p(m, i, o, v=pv): return replace_last_token(o, v)
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
                    col['out'][:2].unsqueeze(0), (sv - prev)[:2].unsqueeze(0).detach()).mean()
                ent = -(ap * torch.log(ap + 1e-8)).sum()
                loss = cl - 0.01 * ent
                opt.zero_grad(); loss.backward(); opt.step()
            prev = sv.detach()
            obs, r, term, trunc, _ = env.step(action)
            total += r; done = term or trunc
        scores.append(total)
        if ep % 20 == 0:
            print(f"    [no-curiosity] ep={ep:3d}: score={total:.0f}")
    env.close()
    return scores


def main():
    print("[P82] Intrinsic Curiosity Vector")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    # Step 1: Train WITH curiosity
    print("  Step 1: Training WITH curiosity drive...")
    (pi, po, pv, pw, scores_c, cur_hist,
     pred_hist) = train_curious(model, tok, DEVICE, n_episodes=80)

    # Step 2: Train WITHOUT curiosity (baseline)
    print("\n  Step 2: Training WITHOUT curiosity (baseline)...")
    scores_nc = train_no_curiosity(model, tok, DEVICE, n_episodes=80)

    # Analysis
    w = 10
    rm = lambda s: [np.mean(s[max(0,i-w):i+1]) for i in range(len(s))]
    rm_c = rm(scores_c); rm_nc = rm(scores_nc)
    best_c = max(scores_c); best_nc = max(scores_nc)

    output = {
        'phase': 82, 'name': 'intrinsic_curiosity_vector',
        'with_curiosity': {
            'scores': [float(s) for s in scores_c],
            'mean_last10': round(float(np.mean(scores_c[-10:])), 2),
            'best': float(best_c),
            'curiosity_trajectory': [round(c, 4) for c in cur_hist],
        },
        'without_curiosity': {
            'scores': [float(s) for s in scores_nc],
            'mean_last10': round(float(np.mean(scores_nc[-10:])), 2),
            'best': float(best_nc),
        },
        'curiosity_advantage': round(float(np.mean(scores_c[-10:]) - np.mean(scores_nc[-10:])), 2),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase82_curiosity.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(rm_c, 'g-', lw=2, label='With curiosity')
    axes[0].plot(rm_nc, 'r-', lw=2, label='Without curiosity')
    axes[0].axhline(y=-200, color='gray', ls=':', label='Default (-200)')
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Score (rolling mean)')
    axes[0].set_title('Curiosity vs No Curiosity', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    ax2 = axes[1].twinx()
    axes[1].plot(cur_hist, 'g-', lw=2, label='Curiosity reward')
    ax2.plot(pred_hist, 'b--', lw=1.5, label='Prediction error')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Curiosity (surprise)', color='green')
    ax2.set_ylabel('Prediction error', color='blue')
    axes[1].set_title('Curiosity Dynamics', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    labels = ['With\nCuriosity', 'Without\nCuriosity']
    means = [np.mean(scores_c[-10:]), np.mean(scores_nc[-10:])]
    axes[2].bar(labels, means, color=['tab:green', 'tab:red'], edgecolor='black')
    axes[2].set_ylabel('Mean Score (last 10)')
    axes[2].set_title('Final Performance', fontweight='bold')
    for i, m in enumerate(means):
        axes[2].text(i, m+2, f'{m:.1f}', ha='center', fontweight='bold')

    plt.suptitle('Phase 82: Intrinsic Curiosity Vector\n'
                '"I explore not for reward, but because surprise is pleasure"',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase82_curiosity.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Curiosity advantage: {output['curiosity_advantage']}")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
