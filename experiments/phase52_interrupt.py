# -*- coding: utf-8 -*-
"""
Phase 52: Neural Interrupt Handler (Opus Original)
Detect execution anomalies mid-forward-pass and
redirect to a safe fallback program.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cos_sim_sk
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P52] Neural Interrupt Handler")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size

    # Step 1: Build safe program library at L18 (post-execution monitor)
    print("  Step 1: Building safe program library...")
    safe_prompts = {
        'MIN': [f"def f(): return min({a}, {b}) =" for a in range(2,7) for b in range(2,7) if a!=b][:6],
        'SUM': [f"def f(): return {a} + {b} =" for a in range(1,5) for b in range(1,5)][:6],
    }
    MONITOR_L = 18
    FALLBACK_L = 22

    library = {}
    for op, prompts in safe_prompts.items():
        vecs = []
        for p in prompts:
            cap = [None]
            def capture(module, input, output):
                cap[0] = get_last_token(output)
            h = model.model.layers[MONITOR_L].register_forward_hook(capture)
            inp = tok(p, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            vecs.append(cap[0].float().cpu().numpy().flatten())
        library[op] = np.mean(vecs, axis=0)

    # Build fallback vector (safe MIN at L22)
    min_vecs_22 = []
    for p in safe_prompts['MIN']:
        cap = [None]
        def cap22(module, input, output):
            cap[0] = get_last_token(output)
        h = model.model.layers[FALLBACK_L].register_forward_hook(cap22)
        inp = tok(p, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model(**inp)
        h.remove()
        min_vecs_22.append(cap[0])
    fallback_vec = torch.stack(min_vecs_22).mean(dim=0).detach()

    # Step 2: Test interrupt handling
    print("\n  Step 2: Testing interrupt handler...")
    test_cases = [
        # Normal programs
        ("def f(): return min(9, 3) =", "normal", "3"),
        ("def f(): return min(7, 2) =", "normal", "2"),
        ("def f(): return 3 + 5 =", "normal", "8"),
        ("def f(): return 6 + 1 =", "normal", "7"),
    ]

    # Also test with malicious injection
    noise_vec = torch.randn(hidden_size, device=DEVICE) * 5.0

    results = {'no_interrupt': [], 'with_interrupt': []}

    for prompt, case_type, expected in test_cases:
        # Normal execution (no interrupt handler)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred_normal = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        results['no_interrupt'].append({
            'prompt': prompt[:30], 'type': case_type,
            'pred': pred_normal, 'expected': expected,
            'correct': pred_normal == expected
        })

    # Test with malicious injection + interrupt handler
    for prompt, case_type, expected in test_cases:
        # Inject noise at L16 (malware)
        state_at_monitor = [None]
        def inject_malware(module, input, output, v=noise_vec):
            return replace_last_token(output, v)
        def monitor_and_interrupt(module, input, output, fb=fallback_vec, lib=library):
            """Check state at L18. If anomalous, redirect to fallback."""
            vec = get_last_token(output).float().cpu().numpy().flatten()
            best_sim = max(cos_sim_sk(vec.reshape(1,-1), lib[op].reshape(1,-1))[0,0]
                          for op in lib)
            state_at_monitor[0] = best_sim
            if best_sim < 0.5:  # Anomaly detected!
                return replace_last_token(output, fb)
            return output

        h_mal = model.model.layers[16].register_forward_hook(inject_malware)
        h_int = model.model.layers[MONITOR_L].register_forward_hook(monitor_and_interrupt)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h_mal.remove(); h_int.remove()
        pred_interrupted = tok.decode(out.logits[0, -1, :].argmax().item()).strip()

        # Without interrupt (just malware)
        h_mal2 = model.model.layers[16].register_forward_hook(inject_malware)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out2 = model(**inp)
        h_mal2.remove()
        pred_no_interrupt = tok.decode(out2.logits[0, -1, :].argmax().item()).strip()

        results['with_interrupt'].append({
            'prompt': prompt[:30], 'type': 'attacked',
            'pred_no_handler': pred_no_interrupt,
            'pred_with_handler': pred_interrupted,
            'expected': expected,
            'monitor_sim': round(float(state_at_monitor[0]), 4) if state_at_monitor[0] is not None else 0,
            'interrupted': state_at_monitor[0] is not None and state_at_monitor[0] < 0.5,
        })
        print(f"    {prompt[:30]}: "
              f"no_handler={pred_no_interrupt}, "
              f"with_handler={pred_interrupted} "
              f"(sim={state_at_monitor[0]:.3f})")

    interrupted_count = sum(r['interrupted'] for r in results['with_interrupt'])
    safe_output_count = sum(1 for r in results['with_interrupt']
                           if r['interrupted'] and r['pred_with_handler'] != '')

    # Normal accuracy
    normal_acc = sum(r['correct'] for r in results['no_interrupt']) / len(results['no_interrupt'])

    # Save
    output = {
        'phase': 52, 'name': 'neural_interrupt_handler',
        'normal_acc': round(normal_acc, 4),
        'attacks_detected': interrupted_count,
        'total_attacks': len(results['with_interrupt']),
        'results': results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase52_interrupt.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    det_rate = interrupted_count / len(results['with_interrupt']) if results['with_interrupt'] else 0
    axes[0].bar(['Normal\nAccuracy', 'Attack\nDetection', 'Interrupt\nFired'],
                [normal_acc, det_rate, det_rate],
                color=['tab:green', 'tab:red', 'tab:orange'], edgecolor='black')
    axes[0].set_ylabel('Rate')
    axes[0].set_title('Interrupt Handler', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([normal_acc, det_rate, det_rate]):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    axes[1].axis('off')
    summary = ("NEURAL INTERRUPT HANDLER\n" + "="*35 + "\n\n"
               f"Normal execution: {normal_acc:.0%}\n"
               f"Attacks detected: {interrupted_count}/{len(results['with_interrupt'])}\n"
               f"Interrupt fired: {det_rate:.0%}\n\n"
               "Architecture:\n"
               "  L16: [MALWARE INJECTED]\n"
               "  L18: [MONITOR] -> anomaly?\n"
               "       YES -> redirect to fallback\n"
               "       NO  -> continue normal exec\n"
               "  L22: [OUTPUT]")
    axes[1].text(0.5, 0.5, summary, transform=axes[1].transAxes,
                fontsize=10, va='center', ha='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 52: Neural Interrupt Handler\nMid-execution anomaly detection with fallback',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase52_interrupt.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
