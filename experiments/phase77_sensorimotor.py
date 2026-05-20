# -*- coding: utf-8 -*-
"""
Phase 77: Sensorimotor Loop
Connect NeuOS registers to Gymnasium CartPole environment.
Can NeuOS control an environment WITHOUT reinforcement learning,
using only its autopoietic self-compilation?

L2 (data register) <- environment state (4D -> 896D projection)
L22 (output register) -> action (896D -> 2D projection)

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


def evaluate_random(env_name='CartPole-v1', n_episodes=20):
    """Baseline: random actions."""
    env = gym.make(env_name)
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    return scores


def evaluate_neuos(model, tok, proj_in, proj_out, prog_vec,
                   input_layer, output_layer, device,
                   env_name='CartPole-v1', n_episodes=20):
    """Run NeuOS with learned projections in CartPole."""
    env = gym.make(env_name)
    scores = []
    prompt = "compute:"
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        done = False
        while not done:
            # Project state into hidden space
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            state_vec = proj_in @ state_tensor  # (896,)

            # Inject state into L2 and program into L16
            collected = {}
            def inject_data(module, input, output, v=state_vec):
                return replace_last_token(output, v)
            def inject_prog(module, input, output, v=prog_vec):
                return replace_last_token(output, v)
            def read_output(module, input, output):
                collected['out'] = get_last_token(output)

            h1 = model.model.layers[input_layer].register_forward_hook(inject_data)
            h2 = model.model.layers[8].register_forward_hook(inject_prog)
            h3 = model.model.layers[output_layer].register_forward_hook(read_output)

            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                model(**inp)
            h1.remove(); h2.remove(); h3.remove()

            # Project output to action space
            out_vec = collected['out']
            action_logits = proj_out @ out_vec  # (2,)
            action = int(action_logits.argmax().item())

            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    return scores


def train_sensorimotor(model, tok, device, input_layer=2, output_layer=22,
                       n_episodes=50, lr=0.01):
    """Train projection matrices using self-supervised sensorimotor loop.
    Loss: internal consistency - the output should correlate with
    state changes (temporal difference of state representations)."""
    hidden_size = model.config.hidden_size
    prompt = "compute:"

    # Learnable projections
    proj_in = torch.randn(hidden_size, 4, device=device) * 0.01
    proj_in.requires_grad_(True)
    proj_out = torch.randn(2, hidden_size, device=device) * 0.01
    proj_out.requires_grad_(True)

    # Program vector (sensorimotor controller)
    prog_vec = torch.randn(hidden_size, device=device) * 0.01
    prog_vec.requires_grad_(True)

    opt = torch.optim.Adam([proj_in, proj_out, prog_vec], lr=lr)
    env = gym.make('CartPole-v1')

    training_scores = []
    training_losses = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        ep_loss = 0
        steps = 0
        prev_state_vec = None

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            state_vec = proj_in @ state_tensor

            def inject_data(module, input, output, v=state_vec):
                return replace_last_token(output, v)
            def inject_prog(module, input, output, v=prog_vec):
                return replace_last_token(output, v)

            collected = {}
            def read_output(module, input, output):
                collected['out'] = get_last_token(output)

            h1 = model.model.layers[input_layer].register_forward_hook(inject_data)
            h2 = model.model.layers[8].register_forward_hook(inject_prog)
            h3 = model.model.layers[output_layer].register_forward_hook(read_output)

            inp = tok(prompt, return_tensors='pt').to(device)
            out = model(**inp)
            h1.remove(); h2.remove(); h3.remove()

            out_vec = collected['out']
            action_logits = proj_out @ out_vec
            action_probs = torch.softmax(action_logits, dim=0)
            action = int(action_logits.argmax().item())

            # Self-supervised loss: temporal coherence
            # The output should be predictive of state transitions
            if prev_state_vec is not None:
                state_change = state_vec - prev_state_vec
                # Loss: output vector should align with state change direction
                pred_change = out_vec[:4]  # First 4 dims as prediction
                actual_change_proj = proj_in @ torch.tensor(
                    obs, dtype=torch.float32, device=device) - prev_state_vec
                coherence_loss = -torch.nn.functional.cosine_similarity(
                    pred_change.unsqueeze(0),
                    actual_change_proj[:4].unsqueeze(0).detach()
                ).mean()

                # Entropy bonus for exploration
                entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()
                loss = coherence_loss - 0.01 * entropy

                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()

            prev_state_vec = state_vec.detach()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        training_scores.append(total_reward)
        avg_loss = ep_loss / max(steps - 1, 1)
        training_losses.append(avg_loss)

        if ep % 10 == 0 or ep == n_episodes - 1:
            print(f"    ep={ep:3d}: score={total_reward:.0f}, loss={avg_loss:.4f}")

    env.close()
    return (proj_in.detach(), proj_out.detach(), prog_vec.detach(),
            training_scores, training_losses)


def main():
    print("[P77] Sensorimotor Loop")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)

    for p in model.parameters():
        p.requires_grad = False

    INPUT_LAYER = 2   # L2: Data register
    OUTPUT_LAYER = 22  # L22: Output register

    # Step 1: Random baseline
    print("  Step 1: Random baseline...")
    random_scores = evaluate_random(n_episodes=30)
    print(f"    Random: mean={np.mean(random_scores):.1f}, "
          f"max={np.max(random_scores):.0f}")

    # Step 2: Train sensorimotor loop
    print("\n  Step 2: Training sensorimotor controller...")
    (proj_in, proj_out, prog_vec,
     train_scores, train_losses) = train_sensorimotor(
        model, tok, DEVICE,
        input_layer=INPUT_LAYER, output_layer=OUTPUT_LAYER,
        n_episodes=80, lr=0.005)

    # Step 3: Evaluate trained controller
    print("\n  Step 3: Evaluating trained controller...")
    neuos_scores = evaluate_neuos(
        model, tok, proj_in, proj_out, prog_vec,
        INPUT_LAYER, OUTPUT_LAYER, DEVICE, n_episodes=30)
    print(f"    NeuOS: mean={np.mean(neuos_scores):.1f}, "
          f"max={np.max(neuos_scores):.0f}")

    # Step 4: Ablation - random projections (no training)
    print("\n  Step 4: Ablation - random projections...")
    hidden_size = model.config.hidden_size
    rand_proj_in = torch.randn(hidden_size, 4, device=DEVICE) * 0.01
    rand_proj_out = torch.randn(2, hidden_size, device=DEVICE) * 0.01
    rand_prog = torch.randn(hidden_size, device=DEVICE) * 0.01
    ablation_scores = evaluate_neuos(
        model, tok, rand_proj_in, rand_proj_out, rand_prog,
        INPUT_LAYER, OUTPUT_LAYER, DEVICE, n_episodes=30)
    print(f"    Random proj: mean={np.mean(ablation_scores):.1f}")

    # Save results
    output = {
        'phase': 77, 'name': 'sensorimotor_loop',
        'random_baseline': {
            'mean': round(float(np.mean(random_scores)), 2),
            'std': round(float(np.std(random_scores)), 2),
            'max': float(np.max(random_scores)),
            'scores': [float(s) for s in random_scores],
        },
        'neuos_controller': {
            'mean': round(float(np.mean(neuos_scores)), 2),
            'std': round(float(np.std(neuos_scores)), 2),
            'max': float(np.max(neuos_scores)),
            'scores': [float(s) for s in neuos_scores],
        },
        'random_projections': {
            'mean': round(float(np.mean(ablation_scores)), 2),
            'scores': [float(s) for s in ablation_scores],
        },
        'training': {
            'scores': [float(s) for s in train_scores],
            'losses': [round(l, 4) for l in train_losses],
        },
        'improvement_over_random': round(
            float(np.mean(neuos_scores) / np.mean(random_scores)), 2),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase77_sensorimotor.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Save vectors for P78
    torch.save({
        'proj_in': proj_in.cpu(), 'proj_out': proj_out.cpu(),
        'prog_vec': prog_vec.cpu(),
    }, os.path.join(RESULTS_DIR, 'phase77_vectors.pt'))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training curve
    window = 5
    if len(train_scores) >= window:
        smoothed = np.convolve(train_scores,
                              np.ones(window)/window, mode='valid')
        axes[0].plot(range(len(smoothed)), smoothed, 'b-', linewidth=2,
                    label='Smoothed (5-ep)')
    axes[0].plot(train_scores, 'b-', alpha=0.3, label='Raw')
    axes[0].axhline(y=np.mean(random_scores), color='red', linestyle='--',
                   label=f'Random ({np.mean(random_scores):.0f})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score (steps survived)')
    axes[0].set_title('Training Progress', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Comparison
    labels = ['Random', 'NeuOS', 'Rand Proj\n(ablation)']
    means = [np.mean(random_scores), np.mean(neuos_scores),
             np.mean(ablation_scores)]
    stds = [np.std(random_scores), np.std(neuos_scores),
            np.std(ablation_scores)]
    colors = ['tab:red', 'tab:blue', 'tab:gray']
    axes[1].bar(labels, means, yerr=stds, color=colors, edgecolor='black',
               capsize=5)
    axes[1].set_ylabel('Mean Score')
    axes[1].set_title('Controller Comparison', fontweight='bold')
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1].text(i, m + s + 2, f'{m:.1f}', ha='center', fontweight='bold')

    # Score distributions
    axes[2].boxplot([random_scores, neuos_scores, ablation_scores],
                   labels=['Random', 'NeuOS', 'Rand Proj'])
    axes[2].set_ylabel('Score')
    axes[2].set_title('Score Distribution', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Phase 77: Sensorimotor Loop\n'
                'NeuOS registers connected to CartPole environment',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase77_sensorimotor.png'),
               dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Improvement over random: {output['improvement_over_random']}x")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
