# -*- coding: utf-8 -*-
"""
Phase 49: Neural Proprioception (Self-Monitoring)
OS monitors its own register state autonomously.
Detect anomalies BEFORE they cause output errors.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, get_last_token, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P49] Neural Proprioception")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size

    # Step 1: Build proprioceptive model
    # Monitor L2, L8, L16, L22 simultaneously -> predict if output will be correct
    print("  Step 1: Building proprioceptive baseline...")

    MONITOR_LAYERS = [2, 8, 16, 22]
    prompts_correct = []
    prompts_wrong = []

    # Generate correct executions
    for a in range(2, 8):
        for b in range(2, 8):
            if a == b: continue
            prompt = f"def f(): return min({a}, {b}) ="
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            expected = str(min(a, b))
            if pred == expected:
                prompts_correct.append(prompt)
            else:
                prompts_wrong.append(prompt)

    print(f"    Correct: {len(prompts_correct)}, Wrong: {len(prompts_wrong)}")

    # Generate anomalous executions (injected noise)
    anomaly_prompts = []
    for a in range(2, 8):
        for b in range(2, 8):
            if a == b: continue
            anomaly_prompts.append(f"def f(): return min({a}, {b}) =")

    # Collect states
    normal_states = []
    anomaly_states = []

    for prompt in prompts_correct[:20]:
        state = []
        for layer in MONITOR_LAYERS:
            cap = [None]
            def capture(module, input, output):
                cap[0] = get_last_token(output)
            h = model.model.layers[layer].register_forward_hook(capture)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            state.append(cap[0].float().cpu().numpy().flatten())
        normal_states.append(np.concatenate(state))

    # Anomaly: inject random noise at L16
    noise_vec = torch.randn(hidden_size, device=DEVICE) * 3.0
    for prompt in anomaly_prompts[:20]:
        state = []
        for layer in MONITOR_LAYERS:
            cap = [None]
            def capture_a(module, input, output):
                cap[0] = get_last_token(output)
            def inject_noise(module, input, output, v=noise_vec):
                return replace_last_token(output, v)

            hooks = []
            if layer >= 16:
                h_inj = model.model.layers[16].register_forward_hook(inject_noise)
                hooks.append(h_inj)
            h_cap = model.model.layers[layer].register_forward_hook(capture_a)
            hooks.append(h_cap)

            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            for hk in hooks:
                hk.remove()
            state.append(cap[0].float().cpu().numpy().flatten())
        anomaly_states.append(np.concatenate(state))

    # Step 2: Train anomaly detector
    print("\n  Step 2: Training anomaly detector...")
    X = np.array(normal_states + anomaly_states)
    y = [0] * len(normal_states) + [1] * len(anomaly_states)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"    Train accuracy: {train_acc:.0%}")

    # Step 3: Test on unseen data
    print("\n  Step 3: Testing proprioception on unseen data...")
    test_normal = []
    test_anomaly = []

    for prompt in prompts_correct[20:30]:
        state = []
        for layer in MONITOR_LAYERS:
            cap = [None]
            def capture_t(module, input, output):
                cap[0] = get_last_token(output)
            h = model.model.layers[layer].register_forward_hook(capture_t)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            h.remove()
            state.append(cap[0].float().cpu().numpy().flatten())
        test_normal.append(np.concatenate(state))

    noise_vec2 = torch.randn(hidden_size, device=DEVICE) * 3.0
    for prompt in anomaly_prompts[20:30]:
        state = []
        for layer in MONITOR_LAYERS:
            cap = [None]
            def capture_ta(module, input, output):
                cap[0] = get_last_token(output)
            def inject_noise2(module, input, output, v=noise_vec2):
                return replace_last_token(output, v)
            hooks = []
            if layer >= 16:
                h_inj = model.model.layers[16].register_forward_hook(inject_noise2)
                hooks.append(h_inj)
            h_cap = model.model.layers[layer].register_forward_hook(capture_ta)
            hooks.append(h_cap)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model(**inp)
            for hk in hooks:
                hk.remove()
            state.append(cap[0].float().cpu().numpy().flatten())
        test_anomaly.append(np.concatenate(state))

    X_test = np.array(test_normal + test_anomaly)
    y_test = [0] * len(test_normal) + [1] * len(test_anomaly)
    test_acc = clf.score(X_test, y_test)
    preds = clf.predict(X_test)
    tp = sum(1 for p, t in zip(preds, y_test) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, y_test) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, y_test) if p == 0 and t == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"    Test accuracy: {test_acc:.0%}")
    print(f"    Precision: {precision:.0%}, Recall: {recall:.0%}")

    # Save
    output = {
        'phase': 49, 'name': 'neural_proprioception',
        'train_acc': round(train_acc, 4),
        'test_acc': round(test_acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'n_normal': len(normal_states),
        'n_anomaly': len(anomaly_states),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase49_proprioception.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(['Train', 'Test'], [train_acc, test_acc],
                color=['tab:blue', 'tab:green'], edgecolor='black')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Proprioceptive Anomaly Detection', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate([train_acc, test_acc]):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=14)

    axes[1].bar(['Precision', 'Recall'], [precision, recall],
                color=['tab:orange', 'tab:purple'], edgecolor='black')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Detection Quality', fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate([precision, recall]):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=14)

    plt.suptitle('Phase 49: Neural Proprioception\nSelf-monitoring for pre-emptive anomaly detection',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase49_proprioception.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
