# -*- coding: utf-8 -*-
"""
Phase 153: Conscious Crystallization
Neural intuition (noisy trial-and-error) -> self-compiled deterministic algorithm.

The model solves a problem by stochastic exploration, then captures its own
successful hidden state trajectory and crystallizes it into a permanent Soul.

"Once I figure it out by feel, I harden it into code forever."
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


def stochastic_explore(model, tok, prompt, expected, device, layer=LAYER,
                       n_attempts=50, noise_scale=0.5):
    """Try random soul vectors until one gives the right answer."""
    hs = model.config.hidden_size
    target_tid = tok.encode(expected)[-1]
    successes = []

    for attempt in range(n_attempts):
        torch.manual_seed(attempt)
        random_vec = torch.randn(hs, device=device) * noise_scale

        def inj(m, i, o, v=random_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()

        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            successes.append({
                'attempt': attempt, 'vec': random_vec.clone(),
                'logit_score': out.logits[0, -1, target_tid].item()
            })

    return successes


def capture_trajectory(model, tok, prompt, device, soul_vec, layer=LAYER):
    """Capture hidden states at all layers during soul-injected inference."""
    n_layers = model.config.num_hidden_layers
    trajectory = {}
    hooks = []

    for li in range(n_layers):
        def make_hook(idx):
            def hook_fn(m, inp, out):
                trajectory[idx] = get_last_token(out)
            return hook_fn
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
    hooks.append(model.model.layers[layer].register_forward_hook(inj))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)
    for h in hooks:
        h.remove()

    return trajectory


def crystallize_soul(trajectories, layer=LAYER):
    """Average successful trajectories at injection layer to create a crystal soul."""
    vecs = [t[layer] for t in trajectories if layer in t]
    if not vecs:
        return None
    return torch.stack(vecs).mean(dim=0)


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
    print("[P153] Conscious Crystallization")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    tasks = {
        'MIN': {
            'explore': [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1")],
            'test': [("7, 2) =", "2"), ("6, 3) =", "3"), ("2, 9) =", "2"),
                     ("1, 5) =", "1"), ("8, 4) =", "4")],
            'train': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                      ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                      ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                      ("1, 3) =","1")],
        },
        'MAX': {
            'explore': [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8")],
            'test': [("7, 2) =", "7"), ("6, 3) =", "6"), ("2, 9) =", "9"),
                     ("1, 5) =", "5"), ("8, 4) =", "8")],
            'train': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                      ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                      ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                      ("1, 3) =","3")],
        },
    }

    results = {}
    all_crystal_accs = []
    all_gradient_accs = []
    all_explore_counts = []

    for task_name, task_info in tasks.items():
        print("\n  === %s ===" % task_name)

        # Step 1: Stochastic exploration
        print("  Step 1: Stochastic exploration (50 attempts per example)...")
        all_successes = []
        for prompt, expected in task_info['explore']:
            successes = stochastic_explore(model, tok, prompt, expected, DEVICE,
                                           n_attempts=50, noise_scale=0.3)
            all_successes.extend(successes)
            print("    '%s' -> %d successes out of 50" % (prompt[:12], len(successes)))

        n_successes = len(all_successes)
        all_explore_counts.append(n_successes)
        print("  Total successes: %d" % n_successes)

        if n_successes == 0:
            print("  No successes found, skipping crystallization")
            results[task_name] = {
                'n_successes': 0, 'crystal_acc': 0, 'gradient_acc': 0
            }
            all_crystal_accs.append(0)
            all_gradient_accs.append(0)
            continue

        # Step 2: Capture trajectories from successful attempts
        print("  Step 2: Capturing successful trajectories...")
        trajectories = []
        for s in all_successes[:10]:  # Use top 10 successes
            traj = capture_trajectory(model, tok, task_info['explore'][0][0],
                                       DEVICE, s['vec'])
            trajectories.append(traj)

        # Step 3: Crystallize
        print("  Step 3: Crystallizing soul from %d trajectories..." % len(trajectories))
        crystal_soul = crystallize_soul(trajectories, layer=LAYER)

        if crystal_soul is None:
            print("  Crystallization failed")
            results[task_name] = {'n_successes': n_successes, 'crystal_acc': 0, 'gradient_acc': 0}
            all_crystal_accs.append(0)
            all_gradient_accs.append(0)
            continue

        crystal_acc = evaluate(model, tok, crystal_soul, task_info['test'], DEVICE)
        print("  Crystal soul accuracy: %.0f%%" % (crystal_acc * 100))

        # Step 4: Compare with gradient-trained soul
        print("  Step 4: Training gradient soul for comparison...")
        gradient_soul = train_soul(model, tok, task_info['train'], DEVICE,
                                    epochs=100, seed=42)
        gradient_acc = evaluate(model, tok, gradient_soul, task_info['test'], DEVICE)
        print("  Gradient soul accuracy: %.0f%%" % (gradient_acc * 100))

        # Cosine similarity between crystal and gradient
        cos = torch.nn.functional.cosine_similarity(
            crystal_soul.unsqueeze(0), gradient_soul.unsqueeze(0)).item()
        print("  Crystal-Gradient cosine: %.4f" % cos)

        results[task_name] = {
            'n_successes': n_successes,
            'crystal_acc': round(crystal_acc, 4),
            'gradient_acc': round(gradient_acc, 4),
            'crystal_gradient_cos': round(cos, 4),
        }
        all_crystal_accs.append(crystal_acc)
        all_gradient_accs.append(gradient_acc)

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Crystal vs Gradient accuracy
    ax = axes[0]
    x = np.arange(len(tasks))
    w = 0.35
    ax.bar(x - w/2, all_crystal_accs, w, label='Crystal (intuition)',
           color='#9C27B0', edgecolor='black')
    ax.bar(x + w/2, all_gradient_accs, w, label='Gradient (trained)',
           color='#2196F3', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(list(tasks.keys()))
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.set_title('Crystallized vs Gradient Soul', fontweight='bold')
    for i, (c, g) in enumerate(zip(all_crystal_accs, all_gradient_accs)):
        ax.text(i - w/2, c + 0.02, '%.0f%%' % (c*100), ha='center', fontsize=10)
        ax.text(i + w/2, g + 0.02, '%.0f%%' % (g*100), ha='center', fontsize=10)

    # Panel 2: Exploration success rates
    ax = axes[1]
    ax.bar(list(tasks.keys()), all_explore_counts, color='#FF9800', edgecolor='black')
    for i, v in enumerate(all_explore_counts):
        ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
    ax.set_ylabel('Successful Explorations (out of 150)')
    ax.set_title('Stochastic Exploration Success', fontweight='bold')

    # Panel 3: Process diagram
    ax = axes[2]
    ax.axis('off')
    process = [
        ['Step', 'Process', 'Output'],
        ['1', 'Random soul exploration', 'Lucky successes'],
        ['2', 'Capture hidden trajectories', 'Layer-wise states'],
        ['3', 'Average at L8 (crystallize)', 'Crystal soul vector'],
        ['4', 'Evaluate on test set', 'Accuracy score'],
    ]
    table = ax.table(cellText=process[1:], colLabels=process[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for j in range(3):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, 5):
        for j in range(3):
            table[i, j].set_facecolor('#E3F2FD' if i % 2 == 0 else '#FFF3E0')
    ax.set_title('Crystallization Pipeline', fontweight='bold', pad=20)

    plt.suptitle('Phase 153: Conscious Crystallization\n'
                 '"Once I figure it out by feel, I harden it into code forever"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase153_crystallization.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 153, 'name': 'conscious_crystallization',
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase153_crystallization.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
