# -*- coding: utf-8 -*-
"""
Phase 9: Unified Register Map
Combine ALL register discoveries from P1, P2, and Aletheia into
a single comprehensive "Neural CPU Register File" diagram.

Probe for: A (operand), B (operand), SUM, CARRY, MIN, MAX, MEDIAN,
           COMPARISON, BRANCH, OPCODE, all in one model pass.

This creates the definitive "ISA reference sheet" for Qwen-0.5B.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def probe_register(model, tok, prompts, labels, n_layers, name):
    """Probe each layer for a specific register."""
    all_hidden = {l: [] for l in range(n_layers)}
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for l in range(n_layers):
            h = out.hidden_states[l+1][0, -1, :].float().cpu().numpy()
            all_hidden[l].append(h)

    y = np.array(labels)
    layer_accs = {}
    for l in range(n_layers):
        X = np.array(all_hidden[l])
        try:
            n_unique = len(np.unique(y))
            if n_unique < 2:
                acc = 0.0
            else:
                clf = LogisticRegression(max_iter=500, random_state=42)
                cv = min(5, n_unique)
                scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
                acc = scores.mean()
        except Exception:
            acc = 0.0
        layer_accs[l] = round(float(acc), 4)

    best_l = max(layer_accs, key=layer_accs.get)
    print(f"    {name}: best=L{best_l} ({layer_accs[best_l]:.1%})")
    return layer_accs


def main():
    print("[P9] Unified Register Map")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    registers = {}

    # === Arithmetic registers ===
    print("  Probing arithmetic registers...")
    arith_prompts = []
    labels_a = []
    labels_b = []
    labels_sum = []
    labels_carry = []
    for a in range(1, 8):
        for b in range(1, 8):
            arith_prompts.append(f"def f(): return {a} + {b} =")
            labels_a.append(a)
            labels_b.append(b)
            labels_sum.append((a + b) % 10)
            labels_carry.append(1 if a + b >= 10 else 0)

    registers['Operand_A'] = probe_register(model, tok, arith_prompts, labels_a, n_layers, 'Operand_A')
    registers['Operand_B'] = probe_register(model, tok, arith_prompts, labels_b, n_layers, 'Operand_B')
    registers['SUM_ones'] = probe_register(model, tok, arith_prompts, labels_sum, n_layers, 'SUM_ones')
    registers['CARRY'] = probe_register(model, tok, arith_prompts, labels_carry, n_layers, 'CARRY')

    # === Sorting registers ===
    print("  Probing sorting registers...")
    import itertools, random
    random.seed(42)
    sort_prompts = []
    labels_min = []
    labels_max = []
    labels_med = []
    digits = list(range(1, 7))
    combos = []
    for combo in itertools.combinations(digits, 3):
        for perm in itertools.permutations(combo):
            combos.append(perm)
    combos = random.sample(combos, min(80, len(combos)))
    for perm in combos:
        a, b, c = perm
        sort_prompts.append(f"def sort({a},{b},{c}): return ")
        labels_min.append(min(perm))
        labels_max.append(max(perm))
        labels_med.append(sorted(perm)[1])

    registers['MIN'] = probe_register(model, tok, sort_prompts, labels_min, n_layers, 'MIN')
    registers['MAX'] = probe_register(model, tok, sort_prompts, labels_max, n_layers, 'MAX')
    registers['MEDIAN'] = probe_register(model, tok, sort_prompts, labels_med, n_layers, 'MEDIAN')

    # === Conditional registers ===
    print("  Probing conditional registers...")
    cond_prompts = []
    labels_cmp = []
    labels_branch = []
    for a in range(1, 10):
        for b in range(1, 10):
            if a == b:
                continue
            cond_prompts.append(f"def f(): return 'big' if {a} > {b} else 'small'\nf() = '")
            labels_cmp.append(1 if a > b else 0)
            labels_branch.append(1 if a > b else 0)
    random.seed(42)
    if len(cond_prompts) > 72:
        indices = random.sample(range(len(cond_prompts)), 72)
        cond_prompts = [cond_prompts[i] for i in indices]
        labels_cmp = [labels_cmp[i] for i in indices]
        labels_branch = [labels_branch[i] for i in indices]

    registers['COMPARISON'] = probe_register(model, tok, cond_prompts, labels_cmp, n_layers, 'COMPARISON')

    # === OPCODE (operation type) ===
    print("  Probing OPCODE register...")
    opcode_prompts = []
    opcode_labels = []
    for a in range(2, 8):
        for b in range(2, 8):
            opcode_prompts.append(f"def f(): return {a} + {b} =")
            opcode_labels.append(0)  # addition
            opcode_prompts.append(f"def f(): return {a} - {b} =")
            opcode_labels.append(1)  # subtraction
            opcode_prompts.append(f"def f(): return {a} * {b} =")
            opcode_labels.append(2)  # multiplication
    random.seed(42)
    if len(opcode_prompts) > 108:
        indices = random.sample(range(len(opcode_prompts)), 108)
        opcode_prompts = [opcode_prompts[i] for i in indices]
        opcode_labels = [opcode_labels[i] for i in indices]

    registers['OPCODE'] = probe_register(model, tok, opcode_prompts, opcode_labels, n_layers, 'OPCODE')

    # Save
    # Convert int keys to str for JSON
    results_json = {}
    for reg_name, layer_dict in registers.items():
        results_json[reg_name] = {str(l): v for l, v in layer_dict.items()}
        best_l = max(layer_dict, key=layer_dict.get)
        results_json[reg_name + '_best'] = {'layer': best_l, 'accuracy': layer_dict[best_l]}

    output = {
        'phase': 9, 'name': 'unified_register_map',
        'n_layers': n_layers, 'registers': results_json,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase9_register_map.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # === Grand visualization ===
    reg_names = ['Operand_A', 'Operand_B', 'SUM_ones', 'CARRY',
                 'MIN', 'MAX', 'MEDIAN', 'COMPARISON', 'OPCODE']
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71',
              '#3498db', '#9b59b6', '#1abc9c', '#34495e', '#e91e63']

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    layers = list(range(n_layers))

    for i, reg in enumerate(reg_names):
        accs = [registers[reg][l] for l in layers]
        ax.plot(layers, accs, 'o-', linewidth=2, markersize=4,
                color=colors[i], label=reg, alpha=0.85)

    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Linear Probe Accuracy', fontsize=14)
    ax.set_title('Phase 9: Unified Neural CPU Register Map\n'
                 'Where does each register live in Qwen-0.5B?',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, n_layers - 0.5)

    # Annotate best layers
    for i, reg in enumerate(reg_names):
        best_l = max(registers[reg], key=registers[reg].get)
        best_acc = registers[reg][best_l]
        if best_acc > 0.6:
            ax.annotate(f'L{best_l}', xy=(best_l, best_acc),
                        xytext=(best_l + 0.3, best_acc + 0.02),
                        fontsize=8, color=colors[i], fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase9_register_map.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Heatmap version
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    data = np.array([[registers[reg][l] for l in layers] for reg in reg_names])
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(len(reg_names)))
    ax.set_yticklabels(reg_names, fontsize=11)
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_title('Neural CPU Register File - Heatmap', fontsize=15, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Probe Accuracy')

    # Mark peak for each register
    for i, reg in enumerate(reg_names):
        best_l = max(registers[reg], key=registers[reg].get)
        ax.plot(best_l, i, 'w*', markersize=12, markeredgecolor='black', markeredgewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase9_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  === Register Map Summary ===")
    for reg in reg_names:
        best_l = max(registers[reg], key=registers[reg].get)
        print(f"    {reg:15s}: L{best_l:2d} ({registers[reg][best_l]:.1%})")
    print(f"\n  Completed in {elapsed:.0f}s")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
