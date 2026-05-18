# -*- coding: utf-8 -*-
"""
Phase 4: Graceful Degradation - Layer Dropout at Inference
Does computation survive partial hardware failure?

Uses embedding surgery (Aletheia standard).
Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ARITH_TEST = [
    ("def f(): return 1 + 1 =", " 2"), ("def f(): return 3 + 4 =", " 7"),
    ("def f(): return 5 + 5 =", " 10"), ("def f(): return 8 + 1 =", " 9"),
    ("def f(): return 6 + 3 =", " 9"), ("def f(): return 4 + 4 =", " 8"),
    ("def f(): return 2 + 7 =", " 9"), ("def f(): return 7 + 2 =", " 9"),
]
FACT_TEST = [
    ("# The capital of Japan is", " Tokyo"),
    ("# The capital of France is", " Paris"),
    ("# The largest planet is", " Jupiter"),
    ("# Water freezes at", " 0"),
    ("# A year has", " 365"),
    ("# The number of continents is", " 7"),
    ("# Pi is approximately", " 3"),
    ("# The boiling point of water is", " 100"),
]


def main():
    print("[P4] Graceful Degradation - Layer Dropout")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_trials = 5
    results = {'arithmetic': {}, 'factual': {}}

    for rate in dropout_rates:
        arith_accs = []
        fact_accs = []

        for trial in range(n_trials):
            np.random.seed(trial * 100 + int(rate * 100))
            n_skip = int(n_layers * rate)
            skippable = list(range(2, n_layers - 2))
            skip_set = set(np.random.choice(skippable, min(n_skip, len(skippable)), replace=False))

            def make_skip_hook(layer_idx):
                def hook_fn(module, input, output):
                    if layer_idx in skip_set:
                        inp_h = input[0] if isinstance(input, tuple) else input
                        if isinstance(output, tuple):
                            return (inp_h,) + output[1:]
                        return inp_h
                    return output
                return hook_fn

            handles = []
            for l in range(n_layers):
                h = model.model.layers[l].register_forward_hook(make_skip_hook(l))
                handles.append(h)

            correct = 0
            for prompt, answer in ARITH_TEST:
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                pred = tok.decode(out.logits[0, -1, :].argmax().item())
                ans_tok = tok.decode(tok.encode(answer)[-1])
                if pred.strip() == ans_tok.strip():
                    correct += 1
            arith_accs.append(correct / len(ARITH_TEST))

            correct = 0
            for prompt, answer in FACT_TEST:
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    out = model(**inp)
                pred = tok.decode(out.logits[0, -1, :].argmax().item())
                ans_tok = tok.decode(tok.encode(answer)[-1])
                if pred.strip() == ans_tok.strip():
                    correct += 1
            fact_accs.append(correct / len(FACT_TEST))

            for h in handles:
                h.remove()

        results['arithmetic'][str(rate)] = round(float(np.mean(arith_accs)), 4)
        results['factual'][str(rate)] = round(float(np.mean(fact_accs)), 4)
        print(f"  rate={rate:.0%}: arith={np.mean(arith_accs):.1%} fact={np.mean(fact_accs):.1%}")

    output = {
        'phase': 4, 'name': 'graceful_degradation',
        'n_layers': n_layers, 'n_trials': n_trials,
        'dropout_rates': dropout_rates, 'results': results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase4_degradation.json'), 'w') as f:
        json.dump(output, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    rates_pct = [r * 100 for r in dropout_rates]
    arith_vals = [results['arithmetic'][str(r)] for r in dropout_rates]
    fact_vals = [results['factual'][str(r)] for r in dropout_rates]

    ax.plot(rates_pct, arith_vals, 'o-', linewidth=2.5, markersize=8,
            label='Arithmetic', color='tab:blue')
    ax.plot(rates_pct, fact_vals, 's-', linewidth=2.5, markersize=8,
            label='Factual', color='tab:orange')
    ax.set_xlabel('Layer Dropout Rate (%)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Phase 4: Graceful Degradation\nCan computation survive partial hardware failure?',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase4_degradation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
