# -*- coding: utf-8 -*-
"""
Phase 185: Soul Capacity Bottleneck
Why does ADD cap at 20%? Is it a fundamental information capacity limit?
Tests: multi-vector souls, higher-rank injections, dimension analysis.
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8


def generate_add_data(n, seed=42):
    rng = np.random.RandomState(seed)
    pairs = set()
    while len(pairs) < n:
        a, b = rng.randint(1, 10, size=2)
        if a != b:
            pairs.add((int(a), int(b)))
    return [("%d, %d) =" % (a, b), str(a + b)) for a, b in list(pairs)[:n]]


def generate_min_data(n, seed=42):
    rng = np.random.RandomState(seed)
    pairs = set()
    while len(pairs) < n:
        a, b = rng.randint(1, 10, size=2)
        if a != b:
            pairs.add((int(a), int(b)))
    return [("%d, %d) =" % (a, b), str(min(a, b))) for a, b in list(pairs)[:n]]


def evaluate(model, tok, inject_fn, test_data, device, layer=LAYER):
    correct = 0
    for prompt, expected in test_data:
        h = model.model.layers[layer].register_forward_hook(inject_fn)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data) if test_data else 0


def main():
    print("[P185] Soul Capacity Bottleneck", flush=True)
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    hs = model.config.hidden_size
    add_train = generate_add_data(35, seed=42)
    add_test = generate_add_data(10, seed=999)
    min_train = generate_min_data(10, seed=42)
    min_test = generate_min_data(5, seed=999)

    results = {}

    # === Experiment 1: More training epochs ===
    print("  Exp1: More epochs (100, 200, 300)...", flush=True)
    epoch_results = {}
    for n_epochs in [100, 200, 300]:
        torch.manual_seed(42)
        vec = torch.randn(hs, device=DEVICE) * 0.01
        vec.requires_grad_(True)
        opt = torch.optim.Adam([vec], lr=0.01)
        for _ in range(n_epochs):
            for p, t in add_train:
                tid = tok.encode(t)[-1]
                inp = tok(p, return_tensors='pt').to(DEVICE)
                def inj(m, i, o, v=vec): return replace_last_token(o, v)
                h = model.model.layers[LAYER].register_forward_hook(inj)
                out = model(**inp); h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([tid]).to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
        acc = evaluate(model, tok,
                       lambda m, i, o, v=vec.detach(): replace_last_token(o, v),
                       add_test, DEVICE)
        epoch_results[str(n_epochs)] = round(acc, 4)
        print("    %d epochs -> %.0f%%" % (n_epochs, acc*100), flush=True)
    results['more_epochs'] = epoch_results

    # === Experiment 2: Multi-layer injection ===
    print("\n  Exp2: Multi-layer injection for ADD...", flush=True)
    layer_configs = [(6,), (8,), (6, 8), (4, 6, 8), (6, 7, 8, 9)]
    layer_results = {}
    for layers in layer_configs:
        torch.manual_seed(42)
        vecs = {l: torch.randn(hs, device=DEVICE) * 0.01 for l in layers}
        for v in vecs.values():
            v.requires_grad_(True)
        opt = torch.optim.Adam(list(vecs.values()), lr=0.01)
        for _ in range(100):
            for p, t in add_train:
                tid = tok.encode(t)[-1]
                inp = tok(p, return_tensors='pt').to(DEVICE)
                hooks = []
                for l in layers:
                    def inj(m, i, o, v=vecs[l]): return replace_last_token(o, v)
                    hooks.append(model.model.layers[l].register_forward_hook(inj))
                out = model(**inp)
                for h in hooks: h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([tid]).to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()

        # Evaluate
        def multi_eval(m, i, o):
            pass  # placeholder

        hooks_eval = []
        def make_eval_hook(layers_dict):
            def hook_fn(test_d):
                correct = 0
                for prompt, expected in test_d:
                    hs_list = []
                    for l in layers_dict:
                        def inj(m, i, o, v=layers_dict[l].detach()):
                            return replace_last_token(o, v)
                        hs_list.append(model.model.layers[l].register_forward_hook(inj))
                    inp = tok(prompt, return_tensors='pt').to(DEVICE)
                    with torch.no_grad():
                        out = model(**inp)
                    for h in hs_list: h.remove()
                    pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
                    if pred == expected: correct += 1
                return correct / len(test_d) if test_d else 0
            return hook_fn

        eval_fn = make_eval_hook(vecs)
        acc = eval_fn(add_test)
        key = '+'.join('L%d' % l for l in layers)
        layer_results[key] = round(acc, 4)
        print("    %s -> %.0f%%" % (key, acc*100), flush=True)
    results['multi_layer'] = layer_results

    # === Experiment 3: Higher learning rate / different optimizer ===
    print("\n  Exp3: Different learning rates...", flush=True)
    lr_results = {}
    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        torch.manual_seed(42)
        vec = torch.randn(hs, device=DEVICE) * 0.01
        vec.requires_grad_(True)
        opt = torch.optim.Adam([vec], lr=lr)
        for _ in range(100):
            for p, t in add_train:
                tid = tok.encode(t)[-1]
                inp = tok(p, return_tensors='pt').to(DEVICE)
                def inj(m, i, o, v=vec): return replace_last_token(o, v)
                h = model.model.layers[LAYER].register_forward_hook(inj)
                out = model(**inp); h.remove()
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0, -1, :].unsqueeze(0),
                    torch.tensor([tid]).to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
        acc = evaluate(model, tok,
                       lambda m, i, o, v=vec.detach(): replace_last_token(o, v),
                       add_test, DEVICE)
        lr_results['lr_%.3f' % lr] = round(acc, 4)
        print("    lr=%.3f -> %.0f%%" % (lr, acc*100), flush=True)
    results['learning_rates'] = lr_results

    # === Experiment 4: Additive injection (not replacement) ===
    print("\n  Exp4: Additive injection (h + alpha*v) for ADD...", flush=True)
    additive_results = {}
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        try:
            torch.manual_seed(42)
            vec = torch.randn(hs, device=DEVICE) * 0.01
            vec.requires_grad_(True)
            opt = torch.optim.Adam([vec], lr=0.01)
            for _ in range(100):
                for p, t in add_train:
                    tid = tok.encode(t)[-1]
                    inp = tok(p, return_tensors='pt').to(DEVICE)
                    def add_inj(m, i, o, v=vec, a=alpha):
                        if isinstance(o, tuple):
                            h = o[0].clone()
                            h = h.detach()
                            h[0, -1, :] = h[0, -1, :] + a * v
                            return (h,) + o[1:]
                        return o
                    h = model.model.layers[LAYER].register_forward_hook(add_inj)
                    out = model(**inp); h.remove()
                    loss = torch.nn.functional.cross_entropy(
                        out.logits[0, -1, :].unsqueeze(0),
                        torch.tensor([tid]).to(DEVICE))
                    opt.zero_grad(); loss.backward(); opt.step()

            # Evaluate additive
            def make_add_eval(v_det, a):
                def fn(m, i, o):
                    if isinstance(o, tuple):
                        h = o[0].clone()
                        h[0, -1, :] = h[0, -1, :] + a * v_det
                        return (h,) + o[1:]
                    return o
                return fn
            acc = evaluate(model, tok, make_add_eval(vec.detach(), alpha),
                           add_test, DEVICE)
            additive_results['alpha_%.1f' % alpha] = round(acc, 4)
            print("    alpha=%.1f -> %.0f%%" % (alpha, acc*100), flush=True)
        except Exception as e:
            additive_results['alpha_%.1f' % alpha] = 'error: %s' % str(e)[:80]
            print("    alpha=%.1f -> ERROR: %s" % (alpha, str(e)[:80]), flush=True)
    results['additive_injection'] = additive_results

    # === Experiment 5: Comparison - MIN with same setup (control) ===
    print("\n  Exp5: MIN control (same setup)...", flush=True)
    torch.manual_seed(42)
    vec = torch.randn(hs, device=DEVICE) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for _ in range(100):
        for p, t in min_train:
            tid = tok.encode(t)[-1]
            inp = tok(p, return_tensors='pt').to(DEVICE)
            def inj(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[LAYER].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    min_acc = evaluate(model, tok,
                       lambda m, i, o, v=vec.detach(): replace_last_token(o, v),
                       min_test, DEVICE)
    results['min_control'] = round(min_acc, 4)
    print("    MIN control -> %.0f%%" % (min_acc*100), flush=True)

    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Epochs
    ax = axes[0, 0]
    ep_keys = sorted(epoch_results.keys(), key=int)
    ax.bar(ep_keys, [epoch_results[k] for k in ep_keys],
           color='#E91E63', edgecolor='black', linewidth=1.5)
    ax.axhline(y=min_acc, color='#4CAF50', linestyle='--', linewidth=2,
               label='MIN control (%.0f%%)' % (min_acc*100))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('ADD Test Accuracy')
    ax.set_title('Effect of Training Duration', fontweight='bold')
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Panel 2: Multi-layer
    ax = axes[0, 1]
    lk = list(layer_results.keys())
    lv = [layer_results[k] for k in lk]
    ax.barh(lk, lv, color='#2196F3', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('ADD Test Accuracy')
    ax.set_title('Multi-Layer Injection', fontweight='bold')
    ax.set_xlim(0, 1.1); ax.grid(alpha=0.3, axis='x')

    # Panel 3: Learning rates
    ax = axes[1, 0]
    lrs = sorted(lr_results.keys())
    ax.plot([float(k.split('_')[1]) for k in lrs],
            [lr_results[k] for k in lrs],
            'o-', color='#FF9800', linewidth=2, markersize=8)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('ADD Test Accuracy')
    ax.set_title('Learning Rate Sensitivity', fontweight='bold')
    ax.set_xscale('log'); ax.grid(alpha=0.3)

    # Panel 4: Additive injection
    ax = axes[1, 1]
    ak = sorted(additive_results.keys())
    alphas = [float(k.split('_')[1]) for k in ak]
    ax.plot(alphas, [additive_results[k] for k in ak],
            'o-', color='#9C27B0', linewidth=2, markersize=8, label='Additive')
    ax.axhline(y=epoch_results['100'], color='#E91E63', linestyle='--',
               label='Replace (baseline)')
    ax.set_xlabel('Alpha (injection strength)')
    ax.set_ylabel('ADD Test Accuracy')
    ax.set_title('Additive vs Replacement Injection', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle('Phase 185: Soul Capacity Bottleneck\n'
                 '"Why does ADD cap at 20%? Is it a fundamental limit?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase185_capacity_bottleneck.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 185, 'name': 'soul_capacity_bottleneck',
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase185_capacity_bottleneck.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P185 completed in %.0fs" % (time.time() - start), flush=True)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
