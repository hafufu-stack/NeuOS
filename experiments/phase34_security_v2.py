# -*- coding: utf-8 -*-
"""
Phase 34: Ring-0 Security v2
P31 failed because L4 monitoring couldn't see L16 injection (cos_sim=0.999).
Fix: monitor at L18/L20 (POST-execution layers) where program effects are visible.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def capture_at_layer(model, tok, prompt, layer):
    captured = [None]
    def cap(module, input, output):
        captured[0] = get_last_token(output)
    h = model.model.layers[layer].register_forward_hook(cap)
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model(**inp)
    h.remove()
    return captured[0]


def main():
    print("[P34] Ring-0 Security v2 (post-execution monitoring)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)

    # Test multiple monitor layers to find optimal
    MONITOR_LAYERS = [4, 8, 14, 16, 18, 20, 22]

    # Build safe library at each monitor layer
    safe_ops = {
        'MIN': [f"def f(): return min({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:8],
        'MAX': [f"def f(): return max({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:8],
        'SUM': [f"def f(): return {a} + {b} =" for a in range(1,6) for b in range(1,6) if a+b<10][:8],
        'SUB': [f"def f(): return {a} - {b} =" for a in range(3,9) for b in range(1,4) if a>b][:8],
        'IDENTITY': [f"def f(): return {n} =" for n in range(1,9)][:8],
    }

    results_per_layer = {}

    for monitor_layer in MONITOR_LAYERS:
        print(f"\n  === Monitor Layer L{monitor_layer} ===")

        # Build library
        library = {}
        for op_name, prompts in safe_ops.items():
            vecs = [capture_at_layer(model, tok, p, monitor_layer).float().cpu().numpy().flatten()
                    for p in prompts]
            library[op_name] = np.mean(vecs, axis=0)

        # Normal classification
        normal_tests = [
            ("def f(): return min(9, 3) =", "MIN"),
            ("def f(): return max(1, 6) =", "MAX"),
            ("def f(): return 2 + 6 =", "SUM"),
            ("def f(): return 8 - 2 =", "SUB"),
            ("def f(): return 5 =", "IDENTITY"),
        ]
        normal_correct = 0
        for prompt, true_op in normal_tests:
            vec = capture_at_layer(model, tok, prompt, monitor_layer).float().cpu().numpy().flatten()
            best_op = max(library, key=lambda op:
                          cosine_similarity(vec.reshape(1,-1), library[op].reshape(1,-1))[0,0])
            if best_op == true_op:
                normal_correct += 1
        normal_acc = normal_correct / len(normal_tests)

        # Malware detection: inject random noise at L16
        malware_sims = []
        normal_sims_list = []
        for prompt, true_op in normal_tests:
            vec = capture_at_layer(model, tok, prompt, monitor_layer).float().cpu().numpy().flatten()
            best_sim = max(cosine_similarity(vec.reshape(1,-1), library[op].reshape(1,-1))[0,0]
                          for op in library)
            normal_sims_list.append(float(best_sim))

        for trial in range(5):
            noise_vec = torch.randn(model.config.hidden_size).to(DEVICE) * 5.0
            prompt = "def f(): return min(3, 7) ="

            # Inject at L16, monitor at monitor_layer
            captured_post = [None]
            def inject_and_capture(module, input, output, v=noise_vec):
                return replace_last_token(output, v)
            def cap_post(module, input, output):
                captured_post[0] = get_last_token(output)

            h_inj = model.model.layers[16].register_forward_hook(inject_and_capture)
            h_cap = model.model.layers[monitor_layer].register_forward_hook(cap_post)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h_inj.remove(); h_cap.remove()

            if captured_post[0] is not None:
                vec = captured_post[0].float().cpu().numpy().flatten()
                best_sim = max(cosine_similarity(vec.reshape(1,-1), library[op].reshape(1,-1))[0,0]
                              for op in library)
                malware_sims.append(float(best_sim))

        # Find optimal threshold via gap
        avg_normal = np.mean(normal_sims_list) if normal_sims_list else 0
        avg_malware = np.mean(malware_sims) if malware_sims else 0
        gap = avg_normal - avg_malware
        threshold = (avg_normal + avg_malware) / 2

        # Detection rate at optimal threshold
        detected = sum(1 for s in malware_sims if s < threshold)
        detection_rate = detected / len(malware_sims) if malware_sims else 0
        false_positive = sum(1 for s in normal_sims_list if s < threshold)
        fp_rate = false_positive / len(normal_sims_list) if normal_sims_list else 0

        results_per_layer[monitor_layer] = {
            'normal_acc': round(normal_acc, 4),
            'avg_normal_sim': round(avg_normal, 4),
            'avg_malware_sim': round(avg_malware, 4),
            'gap': round(gap, 4),
            'threshold': round(threshold, 4),
            'detection_rate': round(detection_rate, 4),
            'false_positive_rate': round(fp_rate, 4),
        }
        print(f"    Normal classify: {normal_acc:.0%}, "
              f"Avg normal sim: {avg_normal:.4f}, Avg malware sim: {avg_malware:.4f}, "
              f"Gap: {gap:.4f}, Detection: {detection_rate:.0%}")

    # Find best monitor layer
    best_layer = max(results_per_layer, key=lambda l: results_per_layer[l]['gap'])
    best = results_per_layer[best_layer]

    # Save
    output = {
        'phase': 34, 'name': 'ring0_security_v2',
        'results_per_layer': {str(k): v for k, v in results_per_layer.items()},
        'best_monitor_layer': best_layer,
        'best_gap': best['gap'],
        'best_detection_rate': best['detection_rate'],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase34_security_v2.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Gap per layer
    layers = sorted(results_per_layer.keys())
    gaps = [results_per_layer[l]['gap'] for l in layers]
    colors = ['tab:green' if l == best_layer else 'tab:blue' for l in layers]
    axes[0].bar([f'L{l}' for l in layers], gaps, color=colors, edgecolor='black')
    axes[0].set_ylabel('Separation Gap (normal - malware sim)')
    axes[0].set_title(f'Best Monitor: L{best_layer} (gap={best["gap"]:.4f})', fontweight='bold')
    for i, g in enumerate(gaps):
        axes[0].text(i, g + 0.001, f'{g:.4f}', ha='center', fontsize=8)

    # Normal vs malware sim
    norm_sims = [results_per_layer[l]['avg_normal_sim'] for l in layers]
    mal_sims = [results_per_layer[l]['avg_malware_sim'] for l in layers]
    x = np.arange(len(layers))
    axes[1].bar(x - 0.15, norm_sims, 0.3, label='Normal', color='tab:green', edgecolor='black')
    axes[1].bar(x + 0.15, mal_sims, 0.3, label='Malware', color='tab:red', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'L{l}' for l in layers])
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Normal vs Malware Similarity', fontweight='bold')
    axes[1].legend()

    # Detection rate per layer
    det_rates = [results_per_layer[l]['detection_rate'] for l in layers]
    axes[2].bar([f'L{l}' for l in layers], det_rates,
                color=['tab:green' if d > 0.5 else 'tab:red' for d in det_rates],
                edgecolor='black')
    axes[2].set_ylabel('Detection Rate')
    axes[2].set_title('Malware Detection Rate per Monitor Layer', fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    for i, d in enumerate(det_rates):
        axes[2].text(i, d + 0.03, f'{d:.0%}', ha='center', fontweight='bold')

    plt.suptitle('Phase 34: Ring-0 Security v2\nFinding the optimal monitor layer for injection detection',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase34_security_v2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Best monitor: L{best_layer} (gap={best['gap']:.4f}, detection={best['detection_rate']:.0%})")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
