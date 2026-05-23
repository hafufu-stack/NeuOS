# -*- coding: utf-8 -*-
"""
Phase 179: Soul Vector Dynamics
Model soul vector evolution as a continuous dynamical system.
Train Neural ODE to learn trajectories between program states.
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

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


def evaluate_behavior(model, tok, soul_vec, test_data, device, layer=LAYER):
    preds = []
    for prompt, _ in test_data:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        preds.append(tok.decode(out.logits[0, -1, :].argmax().item()).strip())
    return preds


def task_accuracy(preds, test_data, task):
    """Check accuracy against a specific task."""
    correct = 0
    for pred, (prompt, _) in zip(preds, test_data):
        nums = [int(c) for c in prompt.replace(') =', '').split(',')]
        a, b = nums[0], nums[1]
        if task == 'MIN':
            expected = str(min(a, b))
        elif task == 'MAX':
            expected = str(max(a, b))
        elif task == 'FIRST':
            expected = str(a)
        else:
            expected = str(b)
        if pred == expected:
            correct += 1
    return correct / len(preds) if preds else 0


class VectorFieldMLP(torch.nn.Module):
    """MLP that predicts dv/dt given current state v."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, dim),
        )

    def forward(self, v):
        return self.net(v)


def euler_integrate(vector_field, v0, n_steps, dt=0.1):
    """Simple Euler integration of dv/dt = f(v)."""
    trajectory = [v0.clone()]
    v = v0.clone()
    for _ in range(n_steps):
        dvdt = vector_field(v)
        v = v + dt * dvdt
        trajectory.append(v.clone())
    return trajectory


def main():
    print("[P179] Soul Vector Dynamics")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train base souls
    train_sets = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),
                 ("9, 3) =","3"),("7, 4) =","4"),("6, 1) =","1"),("2, 8) =","2"),
                 ("5, 9) =","5"),("1, 3) =","1")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","6"),
                 ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","8"),
                 ("5, 9) =","9"),("1, 3) =","3")],
        'FIRST': [("3, 7) =","3"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","4"),
                   ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","2"),
                   ("5, 9) =","5"),("1, 3) =","1")],
    }
    test_data = [("7, 2) =","?"),("6, 3) =","?"),("2, 9) =","?"),
                 ("1, 5) =","?"),("8, 4) =","?")]

    print("  Training base souls...")
    souls = {}
    for name, data in train_sets.items():
        souls[name] = train_soul(model, tok, data, DEVICE, seed=42)

    # Build PCA
    matrix = np.array([v.cpu().numpy() for v in souls.values()])
    pca = PCA(n_components=min(matrix.shape[0], 7))
    pca.fit(matrix)

    # === Experiment 1: Linear interpolation trajectory ===
    print("\n  === Linear Interpolation Trajectory ===")
    n_points = 20
    interp_results = []
    for i in range(n_points + 1):
        alpha = i / n_points
        interp_vec = (1 - alpha) * souls['MIN'] + alpha * souls['MAX']
        preds = evaluate_behavior(model, tok, interp_vec, test_data, DEVICE)
        min_acc = task_accuracy(preds, test_data, 'MIN')
        max_acc = task_accuracy(preds, test_data, 'MAX')
        first_acc = task_accuracy(preds, test_data, 'FIRST')
        interp_results.append({
            'alpha': round(alpha, 2),
            'MIN_acc': round(min_acc, 4),
            'MAX_acc': round(max_acc, 4),
            'FIRST_acc': round(first_acc, 4),
        })
    print("  Linear interpolation: %d points evaluated" % len(interp_results))

    # === Experiment 2: Train vector field (Neural ODE-lite) ===
    print("\n  === Training Vector Field ===")
    # Create training trajectory: FIRST -> MIN -> MAX
    waypoints = [souls['FIRST'], souls['MIN'], souls['MAX']]
    # Generate trajectory samples: pairs of (v_t, dv/dt)
    # dv/dt = (v_{t+1} - v_t) / dt
    train_v = []
    train_dvdt = []
    dt = 1.0
    for i in range(len(waypoints) - 1):
        # Intermediate points
        for j in range(10):
            alpha = j / 10
            v_t = (1 - alpha) * waypoints[i] + alpha * waypoints[i + 1]
            dvdt = (waypoints[i + 1] - waypoints[i]) / dt
            train_v.append(v_t)
            train_dvdt.append(dvdt)
    train_v = torch.stack(train_v)
    train_dvdt = torch.stack(train_dvdt)

    # Train vector field MLP
    vf = VectorFieldMLP(896, hidden=256).to(DEVICE)
    opt = torch.optim.Adam(vf.parameters(), lr=0.001)
    for epoch in range(1000):
        pred_dvdt = vf(train_v)
        loss = torch.nn.functional.mse_loss(pred_dvdt, train_dvdt)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 200 == 0:
            print("    Epoch %d: loss=%.6f" % (epoch, loss.item()))

    # === Experiment 3: Integrate from different starting points ===
    print("\n  === Integration from Starting Points ===")
    integration_results = {}

    # Start from FIRST soul, integrate forward
    with torch.no_grad():
        trajectory = euler_integrate(vf, souls['FIRST'].clone(), n_steps=30, dt=0.1)

    traj_behaviors = []
    for i, v in enumerate(trajectory[::3]):  # Sample every 3 steps
        preds = evaluate_behavior(model, tok, v, test_data, DEVICE)
        min_acc = task_accuracy(preds, test_data, 'MIN')
        max_acc = task_accuracy(preds, test_data, 'MAX')
        first_acc = task_accuracy(preds, test_data, 'FIRST')
        traj_behaviors.append({
            'step': i * 3, 'MIN': round(min_acc, 4),
            'MAX': round(max_acc, 4), 'FIRST': round(first_acc, 4)
        })
    integration_results['from_FIRST'] = traj_behaviors
    print("  FIRST trajectory: %d snapshots" % len(traj_behaviors))

    # Start from random point
    torch.manual_seed(99)
    random_start = torch.randn(896, device=DEVICE) * 0.01
    with torch.no_grad():
        rand_traj = euler_integrate(vf, random_start, n_steps=30, dt=0.1)

    rand_behaviors = []
    for i, v in enumerate(rand_traj[::3]):
        preds = evaluate_behavior(model, tok, v, test_data, DEVICE)
        min_acc = task_accuracy(preds, test_data, 'MIN')
        max_acc = task_accuracy(preds, test_data, 'MAX')
        rand_behaviors.append({
            'step': i * 3, 'MIN': round(min_acc, 4), 'MAX': round(max_acc, 4)
        })
    integration_results['from_random'] = rand_behaviors

    # Project trajectories to 7D for visualization
    traj_7d = np.array([pca.transform(v.cpu().numpy().reshape(1, -1))[0]
                        for v in trajectory])
    rand_traj_7d = np.array([pca.transform(v.cpu().numpy().reshape(1, -1))[0]
                             for v in rand_traj])
    waypoints_7d = np.array([pca.transform(v.cpu().numpy().reshape(1, -1))[0]
                             for v in waypoints])

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Linear interpolation
    ax = axes[0, 0]
    alphas = [r['alpha'] for r in interp_results]
    ax.plot(alphas, [r['MIN_acc'] for r in interp_results], 'o-',
            color='#E91E63', label='MIN', linewidth=2, markersize=4)
    ax.plot(alphas, [r['MAX_acc'] for r in interp_results], 's-',
            color='#2196F3', label='MAX', linewidth=2, markersize=4)
    ax.plot(alphas, [r['FIRST_acc'] for r in interp_results], '^-',
            color='#FF9800', label='FIRST', linewidth=2, markersize=4)
    ax.set_xlabel('Interpolation alpha (0=MIN, 1=MAX)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Linear Interpolation MIN -> MAX', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Trajectory in PC0-PC1 space
    ax = axes[0, 1]
    ax.plot(traj_7d[:, 0], traj_7d[:, 1], '-', color='#E91E63',
            linewidth=1.5, alpha=0.7, label='From FIRST')
    ax.plot(traj_7d[0, 0], traj_7d[0, 1], 'o', color='#E91E63', markersize=10)
    ax.plot(traj_7d[-1, 0], traj_7d[-1, 1], 's', color='#E91E63', markersize=10)
    ax.plot(rand_traj_7d[:, 0], rand_traj_7d[:, 1], '-', color='#9E9E9E',
            linewidth=1.5, alpha=0.7, label='From random')
    for i, name in enumerate(['FIRST', 'MIN', 'MAX']):
        ax.plot(waypoints_7d[i, 0], waypoints_7d[i, 1], '*',
                markersize=15, zorder=10, label=name)
    ax.set_xlabel('PC0')
    ax.set_ylabel('PC1')
    ax.set_title('Trajectories in 7D Soul Space (PC0-PC1)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: Behavior evolution along FIRST trajectory
    ax = axes[1, 0]
    steps = [r['step'] for r in traj_behaviors]
    ax.plot(steps, [r['MIN'] for r in traj_behaviors], 'o-',
            color='#E91E63', label='MIN', linewidth=2)
    ax.plot(steps, [r['MAX'] for r in traj_behaviors], 's-',
            color='#2196F3', label='MAX', linewidth=2)
    ax.plot(steps, [r['FIRST'] for r in traj_behaviors], '^-',
            color='#FF9800', label='FIRST', linewidth=2)
    ax.set_xlabel('Integration Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Behavior Along Learned Trajectory\n(Starting from FIRST)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Phase portrait (vector field in PC0-PC1)
    ax = axes[1, 1]
    # Sample grid in PC0-PC1 and compute vector field
    grid_n = 8
    pc0_range = np.linspace(traj_7d[:, 0].min() - 1, traj_7d[:, 0].max() + 1, grid_n)
    pc1_range = np.linspace(traj_7d[:, 1].min() - 1, traj_7d[:, 1].max() + 1, grid_n)
    U = np.zeros((grid_n, grid_n))
    V = np.zeros((grid_n, grid_n))
    n_comp = pca.n_components_
    for i, pc0 in enumerate(pc0_range):
        for j, pc1 in enumerate(pc1_range):
            coords_pca = np.zeros(n_comp)
            coords_pca[0] = pc0
            coords_pca[1] = pc1
            v_896 = pca.inverse_transform(coords_pca.reshape(1, -1))[0]
            v_t = torch.tensor(v_896, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                dv = vf(v_t).cpu().numpy()
            dv_pca = pca.transform(dv.reshape(1, -1))[0]
            U[j, i] = dv_pca[0]
            V[j, i] = dv_pca[1]
    PC0, PC1 = np.meshgrid(pc0_range, pc1_range)
    ax.quiver(PC0, PC1, U, V, alpha=0.6, color='gray')
    for i, name in enumerate(['FIRST', 'MIN', 'MAX']):
        ax.plot(waypoints_7d[i, 0], waypoints_7d[i, 1], '*',
                markersize=15, zorder=10, label=name)
    ax.set_xlabel('PC0')
    ax.set_ylabel('PC1')
    ax.set_title('Vector Field (Phase Portrait)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle('Phase 179: Soul Vector Dynamics\n'
                 '"Can programs evolve along learned trajectories?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase179_soul_dynamics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 179, 'name': 'soul_dynamics',
        'interpolation': interp_results,
        'integration': integration_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase179_soul_dynamics.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P179 completed in %.0fs" % (time.time() - start))
    del model, vf; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
