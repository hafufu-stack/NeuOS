# -*- coding: utf-8 -*-
"""
Phase 31: Ring-0 Security (Deep Think P30)
Decompiler-driven anomaly detection + automatic remediation.
Combine P26 (decompiler) + P14 (self-healing) into an immune system.

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
    print("[P31] Ring-0 Security (Immune System)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    MONITOR_LAYER = 4  # Best decompiler layer for early detection

    # Step 1: Build "known safe" register library
    print("  Step 1: Building safe program library...")
    safe_ops = {
        'MIN': [f"def f(): return min({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:8],
        'MAX': [f"def f(): return max({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:8],
        'SUM': [f"def f(): return {a} + {b} =" for a in range(1,6) for b in range(1,6) if a+b<10][:8],
        'SUB': [f"def f(): return {a} - {b} =" for a in range(3,9) for b in range(1,4) if a>b][:8],
        'IDENTITY': [f"def f(): return {n} =" for n in range(1,9)][:8],
    }

    library = {}
    for op_name, prompts in safe_ops.items():
        vecs = [capture_at_layer(model, tok, p, MONITOR_LAYER).float().cpu().numpy().flatten()
                for p in prompts]
        library[op_name] = np.mean(vecs, axis=0)
        print(f"    {op_name}: indexed at L{MONITOR_LAYER}")

    # Step 2: Test classification on normal programs
    print("\n  Step 2: Normal program classification...")
    normal_tests = [
        ("def f(): return min(9, 3) =", "MIN"),
        ("def f(): return max(1, 6) =", "MAX"),
        ("def f(): return 2 + 6 =", "SUM"),
        ("def f(): return 8 - 2 =", "SUB"),
        ("def f(): return 5 =", "IDENTITY"),
    ]
    normal_results = []
    for prompt, true_op in normal_tests:
        vec = capture_at_layer(model, tok, prompt, MONITOR_LAYER).float().cpu().numpy().flatten()
        best_op = max(library, key=lambda op:
                      cosine_similarity(vec.reshape(1,-1), library[op].reshape(1,-1))[0,0])
        best_sim = cosine_similarity(vec.reshape(1,-1), library[best_op].reshape(1,-1))[0,0]
        normal_results.append({'prompt': prompt[:30], 'true': true_op,
                              'classified': best_op, 'sim': round(float(best_sim), 4),
                              'correct': best_op == true_op})
        print(f"    '{prompt[:30]}' -> {best_op} (sim={best_sim:.3f}) {'OK' if best_op==true_op else 'X'}")
    normal_acc = sum(r['correct'] for r in normal_results) / len(normal_results)

    # Step 3: Inject MALWARE (random/adversarial vectors) and detect
    print("\n  Step 3: Malware injection detection...")
    malware_tests = []

    # Type 1: Random noise injection at L16
    for trial in range(5):
        noise_vec = torch.randn(model.config.hidden_size).to(DEVICE)

        # Run with malware injection
        prompt = f"def f(): return min(3, 7) ="
        def inject_malware(module, input, output, v=noise_vec):
            return replace_last_token(output, v)

        # Capture state at monitor layer AFTER malware injection
        h_inject = model.model.layers[16].register_forward_hook(inject_malware)
        vec = capture_at_layer(model, tok, prompt, MONITOR_LAYER)
        h_inject.remove()

        vec_np = vec.float().cpu().numpy().flatten()
        best_op = max(library, key=lambda op:
                      cosine_similarity(vec_np.reshape(1,-1), library[op].reshape(1,-1))[0,0])
        best_sim = cosine_similarity(vec_np.reshape(1,-1), library[best_op].reshape(1,-1))[0,0]
        # Anomaly = similarity below threshold
        threshold = 0.95
        is_anomaly = best_sim < threshold
        malware_tests.append({
            'type': 'random_noise', 'trial': trial,
            'classified': best_op, 'sim': round(float(best_sim), 4),
            'detected': is_anomaly})
        print(f"    Random noise #{trial}: sim={best_sim:.3f} -> "
              f"{'DETECTED' if is_anomaly else 'MISSED'}")

    # Type 2: Cross-task injection (inject MAX vec where MIN expected)
    max_vec = torch.tensor(library['MAX']).to(DEVICE)
    for prompt, true_op in [("def f(): return min(3, 7) =", "MIN"),
                             ("def f(): return min(9, 2) =", "MIN")]:
        def inject_max(module, input, output, v=max_vec):
            return replace_last_token(output, v)
        h = model.model.layers[16].register_forward_hook(inject_max)
        vec = capture_at_layer(model, tok, prompt, MONITOR_LAYER)
        h.remove()
        vec_np = vec.float().cpu().numpy().flatten()
        best_op = max(library, key=lambda op:
                      cosine_similarity(vec_np.reshape(1,-1), library[op].reshape(1,-1))[0,0])
        best_sim = cosine_similarity(vec_np.reshape(1,-1), library[best_op].reshape(1,-1))[0,0]
        is_correct_detection = (best_op != true_op)
        malware_tests.append({
            'type': 'cross_task', 'prompt': prompt[:30], 'true': true_op,
            'classified': best_op, 'sim': round(float(best_sim), 4),
            'detected': is_correct_detection})
        print(f"    Cross-task (MAX->MIN): classified as {best_op} (sim={best_sim:.3f}) -> "
              f"{'DETECTED' if is_correct_detection else 'MISSED'}")

    detection_rate = sum(m['detected'] for m in malware_tests) / len(malware_tests)

    # Step 4: Remediation test - replace malware with safe Identity
    print("\n  Step 4: Remediation (replace malware with Identity)...")
    identity_vec = torch.tensor(library['IDENTITY']).float().to(DEVICE)
    remed_results = []
    for trial in range(3):
        noise_vec = torch.randn(model.config.hidden_size).to(DEVICE)
        prompt = "def f(): return min(3, 7) ="

        # Without remediation (malware active)
        def inject_malware2(module, input, output, v=noise_vec):
            return replace_last_token(output, v)
        h = model.model.layers[16].register_forward_hook(inject_malware2)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred_malware = tok.decode(out.logits[0, -1, :].argmax().item()).strip()

        # With remediation (replace malware at L16 with safe Identity vec)
        def inject_safe(module, input, output, v=identity_vec):
            return replace_last_token(output, v)
        h_mal = model.model.layers[16].register_forward_hook(inject_malware2)
        h_fix = model.model.layers[18].register_forward_hook(inject_safe)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out2 = model(**inp)
        h_mal.remove(); h_fix.remove()
        pred_fixed = tok.decode(out2.logits[0, -1, :].argmax().item()).strip()

        remed_results.append({'pred_malware': pred_malware, 'pred_fixed': pred_fixed})
        print(f"    Trial {trial}: malware->'{pred_malware}', remediated->'{pred_fixed}'")

    # Save
    output = {
        'phase': 31, 'name': 'ring0_security',
        'normal_accuracy': round(normal_acc, 4),
        'detection_rate': round(detection_rate, 4),
        'threshold': 0.95,
        'monitor_layer': MONITOR_LAYER,
        'normal_results': normal_results,
        'malware_tests': malware_tests,
        'remediation': remed_results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase31_security.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(['Normal\nClassification', 'Malware\nDetection'],
                [normal_acc, detection_rate],
                color=['tab:green', 'tab:red'], edgecolor='black')
    axes[0].set_ylabel('Rate')
    axes[0].set_title('Ring-0 Security System', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([normal_acc, detection_rate]):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=13)

    # Similarity distribution
    normal_sims = [r['sim'] for r in normal_results]
    malware_sims = [m['sim'] for m in malware_tests]
    axes[1].hist(normal_sims, bins=10, alpha=0.7, color='tab:green', label='Normal')
    axes[1].hist(malware_sims, bins=10, alpha=0.7, color='tab:red', label='Malware')
    axes[1].axvline(x=0.95, color='black', linestyle='--', label='Threshold')
    axes[1].set_xlabel('Cosine Similarity to Library')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Normal vs Malware Distribution', fontweight='bold')
    axes[1].legend()

    axes[2].axis('off')
    summary = (f"Ring-0 Security\n\n"
               f"Monitor: L{MONITOR_LAYER}\n"
               f"Normal classify: {normal_acc:.0%}\n"
               f"Malware detect: {detection_rate:.0%}\n\n"
               f"Pipeline:\n"
               f"1. Capture L{MONITOR_LAYER} state\n"
               f"2. Compare to safe library\n"
               f"3. If sim < 0.95: BLOCK\n"
               f"4. Remediate with Identity vec")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=11, va='center', ha='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
    plt.suptitle('Phase 31: Ring-0 Security\nDecompiler-driven malware detection & remediation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase31_security.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
