# -*- coding: utf-8 -*-
"""
Phase 181: Multimodal ISA Probe
Does a Vision-Language Model also have a register-like ISA?
Probe Qwen2-VL (or fallback to text-only with visual-like tasks).
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


def probe_layer_register(model, tok, prompts_and_targets, device, n_layers):
    """Probe each layer to find which layer best encodes which target value."""
    from sklearn.linear_model import Ridge

    # Collect hidden states
    layer_states = {l: [] for l in range(n_layers)}
    targets = []

    for prompt, target_val in prompts_and_targets:
        hooks = []
        states = {}

        def make_hook(layer_idx):
            def hook_fn(m, i, o):
                tensor = o[0] if isinstance(o, tuple) else o
                if tensor.dim() == 3:
                    states[layer_idx] = tensor[0, -1, :].detach().cpu().numpy()
                elif tensor.dim() == 2:
                    states[layer_idx] = tensor[-1, :].detach().cpu().numpy()
            return hook_fn

        for l in range(n_layers):
            hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inp)

        for h in hooks:
            h.remove()

        for l in range(n_layers):
            if l in states:
                layer_states[l].append(states[l])
        targets.append(target_val)

    # Probe each layer
    targets = np.array(targets)
    layer_accuracies = {}
    n_train = len(targets) * 3 // 4

    for l in range(n_layers):
        if len(layer_states[l]) < len(targets):
            layer_accuracies[l] = 0
            continue
        X = np.array(layer_states[l])
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = targets[:n_train], targets[n_train:]

        if len(X_test) == 0:
            layer_accuracies[l] = 0
            continue

        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_test)
        # Round predictions and check accuracy
        correct = sum(1 for p, t in zip(np.round(preds), y_test) if abs(p - t) < 0.5)
        layer_accuracies[l] = round(correct / len(y_test), 4)

    return layer_accuracies


def main():
    print("[P181] Multimodal ISA Probe")
    start = time.time()

    # Try to load a VL model
    vl_model_loaded = False
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", local_files_only=True,
            torch_dtype=torch.float32
        ).to(DEVICE)
        vl_model.eval()
        vl_model_loaded = True
        print("  Loaded Qwen2-VL-2B-Instruct")
    except Exception as e:
        print("  Could not load VL model: %s" % str(e)[:100])
        print("  Falling back to comparative ISA probing on text models")

    results = {}

    if not vl_model_loaded:
        # Fallback: Compare ISA structure across model sizes
        print("\n  === Comparative ISA Probing (0.5B vs 1.5B) ===")

        # Probe 0.5B
        print("  Probing 0.5B...")
        model_05, tok_05 = load_model('Qwen/Qwen2.5-0.5B', device=DEVICE, surgery=True)
        for p in model_05.parameters():
            p.requires_grad = False

        tasks_05 = {}
        # MIN probing
        min_probes = [("%d, %d) =" % (a, b), min(a, b))
                      for a in range(1, 10) for b in range(1, 10) if a != b]
        tasks_05['MIN'] = probe_layer_register(model_05, tok_05, min_probes[:40],
                                               DEVICE, 24)
        # MAX probing
        max_probes = [("%d, %d) =" % (a, b), max(a, b))
                      for a in range(1, 10) for b in range(1, 10) if a != b]
        tasks_05['MAX'] = probe_layer_register(model_05, tok_05, max_probes[:40],
                                               DEVICE, 24)
        # SUM probing
        sum_probes = [("%d, %d) =" % (a, b), a + b)
                      for a in range(1, 8) for b in range(1, 8) if a != b]
        tasks_05['SUM'] = probe_layer_register(model_05, tok_05, sum_probes[:40],
                                               DEVICE, 24)
        # OPCODE probing (classify task type)
        opcode_probes = []
        for a in range(1, 6):
            for b in range(a+1, 7):
                opcode_probes.append(("def f(): return min(%d, %d) =" % (a, b), 0))
                opcode_probes.append(("def f(): return max(%d, %d) =" % (a, b), 1))
        tasks_05['OPCODE'] = probe_layer_register(model_05, tok_05, opcode_probes[:40],
                                                   DEVICE, 24)

        results['model_05b'] = {task: dict(accs) for task, accs in tasks_05.items()}

        # Find peak layers
        for task_name, accs in tasks_05.items():
            peak_layer = max(accs, key=accs.get)
            peak_acc = accs[peak_layer]
            print("    0.5B %s: peak at L%d (%.0f%%)" % (task_name, peak_layer, peak_acc*100))

        del model_05; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Probe 1.5B
        print("\n  Probing 1.5B...")
        try:
            model_15, tok_15 = load_model('Qwen/Qwen2.5-1.5B', device=DEVICE, surgery=True)
            for p in model_15.parameters():
                p.requires_grad = False

            tasks_15 = {}
            tasks_15['MIN'] = probe_layer_register(model_15, tok_15, min_probes[:40],
                                                   DEVICE, 28)
            tasks_15['MAX'] = probe_layer_register(model_15, tok_15, max_probes[:40],
                                                   DEVICE, 28)
            tasks_15['SUM'] = probe_layer_register(model_15, tok_15, sum_probes[:40],
                                                   DEVICE, 28)
            tasks_15['OPCODE'] = probe_layer_register(model_15, tok_15, opcode_probes[:40],
                                                       DEVICE, 28)

            results['model_15b'] = {task: {str(k): v for k, v in accs.items()}
                                    for task, accs in tasks_15.items()}

            for task_name, accs in tasks_15.items():
                peak_layer = max(accs, key=accs.get)
                peak_acc = accs[peak_layer]
                print("    1.5B %s: peak at L%d (%.0f%%)" % (
                    task_name, peak_layer, peak_acc*100))

            del model_15; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            has_15b = True
        except Exception as e:
            print("  Could not load 1.5B: %s" % str(e)[:100])
            has_15b = False

        # === Compare ISA structure ===
        print("\n  === ISA Structure Comparison ===")
        # Normalize layer positions to [0, 1] for comparison
        isa_comparison = {}
        for task_name in ['MIN', 'MAX', 'SUM', 'OPCODE']:
            accs_05 = tasks_05[task_name]
            peak_05 = max(accs_05, key=accs_05.get)
            norm_05 = peak_05 / 23  # 24 layers, 0-indexed

            if has_15b:
                accs_15 = tasks_15[task_name]
                peak_15 = max(accs_15, key=accs_15.get)
                norm_15 = peak_15 / 27  # 28 layers
            else:
                peak_15 = -1
                norm_15 = -1

            isa_comparison[task_name] = {
                '0.5B_layer': peak_05, '0.5B_normalized': round(norm_05, 3),
                '1.5B_layer': peak_15, '1.5B_normalized': round(norm_15, 3),
            }
        results['isa_comparison'] = isa_comparison

        # === PLOT ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        task_colors = {'MIN': '#E91E63', 'MAX': '#2196F3',
                       'SUM': '#FF9800', 'OPCODE': '#4CAF50'}

        # Panel 1: 0.5B register map
        ax = axes[0, 0]
        for task_name, accs in tasks_05.items():
            layers = sorted(accs.keys())
            vals = [accs[l] for l in layers]
            ax.plot(layers, vals, 'o-', color=task_colors[task_name],
                    label=task_name, linewidth=2, markersize=4)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probe Accuracy')
        ax.set_title('0.5B (24 layers) Register Map', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 2: 1.5B register map (if available)
        ax = axes[0, 1]
        if has_15b:
            for task_name, accs in tasks_15.items():
                layers = sorted(accs.keys())
                vals = [accs[l] for l in layers]
                ax.plot(layers, vals, 'o-', color=task_colors[task_name],
                        label=task_name, linewidth=2, markersize=4)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Probe Accuracy')
            ax.set_title('1.5B (28 layers) Register Map', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, '1.5B model not available', ha='center',
                    va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('1.5B Register Map (N/A)', fontweight='bold')

        # Panel 3: Normalized ISA comparison
        ax = axes[1, 0]
        task_names = list(isa_comparison.keys())
        x = np.arange(len(task_names))
        w = 0.35
        vals_05 = [isa_comparison[t]['0.5B_normalized'] for t in task_names]
        ax.bar(x - w/2, vals_05, w, label='0.5B', color='#2196F3',
               edgecolor='black', linewidth=1.5)
        if has_15b:
            vals_15 = [isa_comparison[t]['1.5B_normalized'] for t in task_names]
            ax.bar(x + w/2, vals_15, w, label='1.5B', color='#FF9800',
                   edgecolor='black', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(task_names)
        ax.set_ylabel('Normalized Peak Layer Position')
        ax.set_title('ISA Position Scaling', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Panel 4: Summary
        ax = axes[1, 1]
        ax.axis('off')
        rows = [['Register', '0.5B Layer', '1.5B Layer', 'Scale Factor']]
        for t in task_names:
            ic = isa_comparison[t]
            sf = ic['1.5B_layer'] / max(ic['0.5B_layer'], 1) if has_15b and ic['1.5B_layer'] > 0 else 'N/A'
            rows.append([t, 'L%d' % ic['0.5B_layer'],
                        'L%d' % ic['1.5B_layer'] if has_15b else 'N/A',
                        '%.2f' % sf if isinstance(sf, float) else sf])
        table = ax.table(cellText=rows[1:], colLabels=rows[0],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.0)
        for j in range(4):
            table[0, j].set_facecolor('#1565C0')
            table[0, j].set_text_props(color='white', fontweight='bold')
        ax.set_title('Cross-Model ISA Comparison', fontweight='bold', pad=20)

        plt.suptitle('Phase 181: Multimodal ISA Probe\n'
                     '"Does the register architecture scale across model sizes?"',
                     fontsize=13, fontweight='bold')
    else:
        # VL model path (if loaded)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'VL model probe results here',
                ha='center', va='center', transform=ax.transAxes)
        plt.suptitle('Phase 181: Multimodal ISA Probe', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase181_multimodal_isa.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Convert integer keys to strings for JSON
    json_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            json_results[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    json_results[k][k2] = {str(kk): vv for kk, vv in v2.items()}
                else:
                    json_results[k][k2] = v2
        else:
            json_results[k] = v

    output = {
        'phase': 181, 'name': 'multimodal_isa',
        'vl_model_loaded': vl_model_loaded,
        'results': json_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase181_multimodal_isa.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n  P181 completed in %.0fs" % (time.time() - start))
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
