# -*- coding: utf-8 -*-
"""
Phase 172: Data Scaling Theory
Information-theoretic analysis of why comparison tasks (MIN/MAX)
need n=10 while arithmetic tasks (ADD/SUB) need n=35.
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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


def generate_pairs(n, seed=0):
    """Generate n unique (a, b) pairs with a != b, single digit."""
    rng = np.random.RandomState(seed)
    pairs = set()
    while len(pairs) < n:
        a, b = rng.randint(1, 10, size=2)
        if a != b:
            pairs.add((int(a), int(b)))
    return list(pairs)[:n]


def make_task_data(pairs, task):
    """Create (prompt, target) pairs for a given task."""
    data = []
    for a, b in pairs:
        prompt = "%d, %d) =" % (a, b)
        if task == 'MIN':
            target = str(min(a, b))
        elif task == 'MAX':
            target = str(max(a, b))
        elif task == 'ADD':
            target = str(a + b)
        elif task == 'SUB':
            target = str(abs(a - b))
        else:
            target = str(a)
        data.append((prompt, target))
    return data


def exp_saturation(n, a, tau):
    """Exponential saturation: acc = a * (1 - exp(-n/tau))"""
    return a * (1 - np.exp(-np.array(n) / tau))


def compute_task_entropy(pairs, task):
    """Compute output entropy H(Y) and conditional entropy H(Y|X) for a task."""
    outputs = []
    for a, b in pairs:
        if task == 'MIN':
            outputs.append(min(a, b))
        elif task == 'MAX':
            outputs.append(max(a, b))
        elif task == 'ADD':
            outputs.append(a + b)
        elif task == 'SUB':
            outputs.append(abs(a - b))
    # H(Y) - entropy of output distribution
    vals, counts = np.unique(outputs, return_counts=True)
    probs = counts / counts.sum()
    h_y = -np.sum(probs * np.log2(probs + 1e-10))
    # Number of unique output values
    n_unique = len(vals)
    return h_y, n_unique


def main():
    print("[P172] Data Scaling Theory")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    tasks = ['MIN', 'MAX', 'ADD', 'SUB']
    n_values = [5, 8, 10, 15, 20, 25, 30, 35]
    test_pairs = generate_pairs(10, seed=999)

    # Scaling experiment
    print("  Sweeping training set sizes...")
    scaling_data = {task: [] for task in tasks}
    for task in tasks:
        test_data = make_task_data(test_pairs, task)
        for n in n_values:
            train_pairs = generate_pairs(n, seed=42)
            train_data = make_task_data(train_pairs, task)
            soul = train_soul(model, tok, train_data, DEVICE, seed=42)
            acc = evaluate(model, tok, soul, test_data, DEVICE)
            scaling_data[task].append(round(acc, 4))
            print("    %s n=%d -> %.0f%%" % (task, n, acc*100))

    # Fit saturation curves
    print("\n  Fitting saturation curves...")
    fit_params = {}
    for task in tasks:
        accs = np.array(scaling_data[task])
        try:
            popt, _ = curve_fit(exp_saturation, n_values, accs,
                                p0=[1.0, 10.0], bounds=([0, 1], [1.5, 100]),
                                maxfev=5000)
            fit_params[task] = {'a': round(popt[0], 4), 'tau': round(popt[1], 4)}
            print("    %s: a=%.3f, tau=%.1f" % (task, popt[0], popt[1]))
        except Exception as e:
            fit_params[task] = {'a': 0, 'tau': 0, 'error': str(e)}
            print("    %s: fit failed (%s)" % (task, e))

    # Information-theoretic analysis
    print("\n  Information-theoretic analysis...")
    all_pairs = generate_pairs(45, seed=0)  # Full set for entropy
    info_theory = {}
    for task in tasks:
        h_y, n_unique = compute_task_entropy(all_pairs, task)
        info_theory[task] = {
            'output_entropy_bits': round(h_y, 4),
            'n_unique_outputs': n_unique,
        }
        print("    %s: H(Y)=%.2f bits, %d unique outputs" % (task, h_y, n_unique))

    # Ratio analysis
    comparison_tau = np.mean([fit_params[t].get('tau', 0) for t in ['MIN', 'MAX']])
    arithmetic_tau = np.mean([fit_params[t].get('tau', 0) for t in ['ADD', 'SUB']])
    tau_ratio = arithmetic_tau / max(comparison_tau, 1e-6)
    print("\n  Tau ratio (arithmetic/comparison): %.2f" % tau_ratio)

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'MIN': '#E91E63', 'MAX': '#2196F3', 'ADD': '#FF9800', 'SUB': '#4CAF50'}

    # Panel 1: Raw scaling curves
    ax = axes[0, 0]
    for task in tasks:
        ax.plot(n_values, scaling_data[task], 'o-', color=colors[task],
                label=task, linewidth=2, markersize=8)
    ax.set_xlabel('Training Set Size (n)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Data Scaling Laws', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Fitted curves
    ax = axes[0, 1]
    n_smooth = np.linspace(1, 40, 100)
    for task in tasks:
        ax.scatter(n_values, scaling_data[task], color=colors[task], s=60, zorder=5)
        if fit_params[task].get('tau', 0) > 0:
            y_fit = exp_saturation(n_smooth, fit_params[task]['a'], fit_params[task]['tau'])
            ax.plot(n_smooth, y_fit, '--', color=colors[task],
                    label='%s (tau=%.1f)' % (task, fit_params[task]['tau']), linewidth=2)
    ax.set_xlabel('Training Set Size (n)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Saturation Fit: acc = a*(1-exp(-n/tau))', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Tau comparison
    ax = axes[1, 0]
    tau_vals = [fit_params[t].get('tau', 0) for t in tasks]
    bars = ax.bar(tasks, tau_vals, color=[colors[t] for t in tasks],
                  edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, tau_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                'tau=%.1f' % val, ha='center', fontweight='bold', fontsize=11)
    ax.set_ylabel('Characteristic Scale (tau)')
    ax.set_title('Learning Time Constants\n(Arithmetic/Comparison = %.1fx)' % tau_ratio,
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Panel 4: Information-theoretic
    ax = axes[1, 1]
    h_vals = [info_theory[t]['output_entropy_bits'] for t in tasks]
    n_uniq = [info_theory[t]['n_unique_outputs'] for t in tasks]
    x = np.arange(len(tasks))
    w = 0.35
    bars1 = ax.bar(x - w/2, h_vals, w, color=[colors[t] for t in tasks],
                   edgecolor='black', label='H(Y) bits', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, n_uniq, w, color=[colors[t] for t in tasks],
                    edgecolor='black', label='# unique outputs', alpha=0.4, hatch='///')
    ax.set_ylabel('Output Entropy H(Y) [bits]')
    ax2.set_ylabel('# Unique Output Values')
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_title('Information-Theoretic Complexity', fontweight='bold')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')

    plt.suptitle('Phase 172: Data Scaling Theory\n'
                 '"Why does comparison need 10 examples but arithmetic needs 35?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase172_scaling_theory.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 172, 'name': 'data_scaling_theory',
        'scaling_data': scaling_data,
        'fit_params': fit_params,
        'info_theory': info_theory,
        'tau_ratio': round(tau_ratio, 4),
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase172_scaling_theory.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P172 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
