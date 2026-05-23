# -*- coding: utf-8 -*-
"""
Phase 174: Multi-step Composition
Solve the composition problem (P115: 33% test acc) using
KV-cache paging, recurrent injection, and scratchpad approaches.
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


def evaluate_single(model, tok, soul_vec, prompt, device, layer=LAYER):
    """Evaluate a single prompt and return predicted token string."""
    def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    return tok.decode(out.logits[0, -1, :].argmax().item()).strip()


def main():
    print("[P174] Multi-step Composition")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train MIN and MAX souls
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),("4, 6) =","4"),
                ("9, 3) =","3"),("7, 4) =","4"),("6, 1) =","1"),("2, 8) =","2"),
                ("5, 9) =","5"),("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),("4, 6) =","6"),
                ("9, 3) =","9"),("7, 4) =","7"),("6, 1) =","6"),("2, 8) =","8"),
                ("5, 9) =","9"),("1, 3) =","3")]

    print("  Training MIN and MAX souls...")
    min_soul = train_soul(model, tok, min_data, DEVICE, seed=42)
    max_soul = train_soul(model, tok, max_data, DEVICE, seed=42)

    # Multi-step test cases: compute SORT(a, b) = [min(a,b), max(a,b)]
    # And RANGE(a, b) = max(a,b) - min(a,b)
    test_cases = [
        (7, 2), (6, 3), (2, 9), (1, 5), (8, 4),
        (3, 8), (4, 1), (9, 6), (5, 7), (2, 3),
    ]

    results = {}

    # === Method 1: Sequential Two-Pass ===
    # Step 1: Use MIN soul to get min(a,b)
    # Step 2: Use MAX soul to get max(a,b)
    # Result: [min_result, max_result]
    print("\n  === Method 1: Sequential Two-Pass ===")
    sort_correct = 0
    range_correct = 0
    method1_details = []
    for a, b in test_cases:
        prompt = "%d, %d) =" % (a, b)
        min_pred = evaluate_single(model, tok, min_soul, prompt, DEVICE)
        max_pred = evaluate_single(model, tok, max_soul, prompt, DEVICE)
        expected_min = str(min(a, b))
        expected_max = str(max(a, b))
        sort_ok = (min_pred == expected_min and max_pred == expected_max)
        if sort_ok:
            sort_correct += 1
        # Try computing range
        try:
            range_val = int(max_pred) - int(min_pred)
            range_ok = (range_val == max(a, b) - min(a, b))
        except ValueError:
            range_ok = False
            range_val = None
        if range_ok:
            range_correct += 1
        method1_details.append({
            'input': (a, b), 'min_pred': min_pred, 'max_pred': max_pred,
            'sort_correct': sort_ok, 'range_val': range_val
        })
    results['sequential'] = {
        'sort_accuracy': round(sort_correct / len(test_cases), 4),
        'range_accuracy': round(range_correct / len(test_cases), 4),
    }
    print("    SORT accuracy: %.0f%%" % (sort_correct / len(test_cases) * 100))
    print("    RANGE accuracy: %.0f%%" % (range_correct / len(test_cases) * 100))

    # === Method 2: KV-Cache Chaining ===
    # Run MIN first, save KV cache, then continue with MAX context
    print("\n  === Method 2: KV-Cache Chaining ===")
    kv_sort_correct = 0
    method2_details = []
    for a, b in test_cases:
        prompt = "%d, %d) =" % (a, b)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)

        # Step 1: Run with MIN soul, capture KV cache
        def inj_min(m, i, o, v=min_soul): return replace_last_token(o, v)
        h = model.model.layers[LAYER].register_forward_hook(inj_min)
        with torch.no_grad():
            out1 = model(**inp, use_cache=True)
        h.remove()
        min_pred = tok.decode(out1.logits[0, -1, :].argmax().item()).strip()
        kv_cache = out1.past_key_values

        # Step 2: Continue from KV cache with MAX soul
        next_token = out1.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        def inj_max(m, i, o, v=max_soul): return replace_last_token(o, v)
        h = model.model.layers[LAYER].register_forward_hook(inj_max)
        with torch.no_grad():
            out2 = model(input_ids=next_token, past_key_values=kv_cache, use_cache=True)
        h.remove()
        max_pred = tok.decode(out2.logits[0, -1, :].argmax().item()).strip()

        expected_min = str(min(a, b))
        expected_max = str(max(a, b))
        sort_ok = (min_pred == expected_min and max_pred == expected_max)
        if sort_ok:
            kv_sort_correct += 1
        method2_details.append({
            'input': (a, b), 'min_pred': min_pred, 'max_pred': max_pred,
            'sort_correct': sort_ok
        })
    results['kv_chain'] = {
        'sort_accuracy': round(kv_sort_correct / len(test_cases), 4),
    }
    print("    KV-Chain SORT accuracy: %.0f%%" % (kv_sort_correct / len(test_cases) * 100))

    # === Method 3: Recurrent Multi-Layer Injection ===
    # Inject MIN at L6 and MAX at L8 simultaneously
    print("\n  === Method 3: Dual-Layer Injection ===")
    dual_correct = 0
    method3_details = []
    for a, b in test_cases:
        prompt = "%d, %d) =" % (a, b)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        def inj_l6(m, i, o, v=min_soul): return replace_last_token(o, v)
        def inj_l8(m, i, o, v=max_soul): return replace_last_token(o, v)
        h6 = model.model.layers[6].register_forward_hook(inj_l6)
        h8 = model.model.layers[8].register_forward_hook(inj_l8)
        with torch.no_grad():
            out = model(**inp)
        h6.remove(); h8.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        # What does dual injection produce?
        expected_min = str(min(a, b))
        expected_max = str(max(a, b))
        is_min = pred == expected_min
        is_max = pred == expected_max
        method3_details.append({
            'input': (a, b), 'pred': pred,
            'is_min': is_min, 'is_max': is_max
        })
        if is_min or is_max:
            dual_correct += 1
    results['dual_layer'] = {
        'any_correct': round(dual_correct / len(test_cases), 4),
        'dominant_behavior': 'check details'
    }
    print("    Dual-layer any-correct: %.0f%%" % (dual_correct / len(test_cases) * 100))

    # === Method 4: Multi-step with intermediate prompt ===
    # Step 1: Compute min with soul, then build new prompt with result
    print("\n  === Method 4: Intermediate Prompt Chaining ===")
    chain_sort_correct = 0
    chain_range_correct = 0
    method4_details = []
    for a, b in test_cases:
        # Step 1: MIN
        prompt1 = "%d, %d) =" % (a, b)
        min_pred = evaluate_single(model, tok, min_soul, prompt1, DEVICE)
        # Step 2: MAX with min result in context
        prompt2 = "%s, %d, %d) =" % (min_pred, a, b)
        max_pred = evaluate_single(model, tok, max_soul, prompt2, DEVICE)
        expected_min = str(min(a, b))
        expected_max = str(max(a, b))
        sort_ok = (min_pred == expected_min and max_pred == expected_max)
        if sort_ok:
            chain_sort_correct += 1
        try:
            range_val = int(max_pred) - int(min_pred)
            range_ok = (range_val == max(a, b) - min(a, b))
        except ValueError:
            range_ok = False
            range_val = None
        if range_ok:
            chain_range_correct += 1
        method4_details.append({
            'input': (a, b), 'min_pred': min_pred, 'max_pred': max_pred,
            'sort_correct': sort_ok
        })
    results['chain_prompt'] = {
        'sort_accuracy': round(chain_sort_correct / len(test_cases), 4),
        'range_accuracy': round(chain_range_correct / len(test_cases), 4),
    }
    print("    Chain SORT accuracy: %.0f%%" % (chain_sort_correct / len(test_cases) * 100))
    print("    Chain RANGE accuracy: %.0f%%" % (chain_range_correct / len(test_cases) * 100))

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Method comparison for SORT
    ax = axes[0]
    methods = ['Sequential\nTwo-Pass', 'KV-Cache\nChaining', 'Dual-Layer\nInjection',
               'Prompt\nChaining']
    sort_accs = [
        results['sequential']['sort_accuracy'],
        results['kv_chain']['sort_accuracy'],
        results['dual_layer']['any_correct'],
        results['chain_prompt']['sort_accuracy'],
    ]
    colors = ['#E91E63', '#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(methods, sort_accs, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, sort_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Accuracy')
    ax.set_title('SORT Composition: min(a,b), max(a,b)', fontweight='bold')
    ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, label='P115 baseline (33%)')
    ax.legend()

    # Panel 2: RANGE computation
    ax = axes[1]
    range_methods = ['Sequential', 'Prompt Chain']
    range_accs = [
        results['sequential']['range_accuracy'],
        results['chain_prompt']['range_accuracy'],
    ]
    bars = ax.bar(range_methods, range_accs, color=['#E91E63', '#4CAF50'],
                  edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, range_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Accuracy')
    ax.set_title('RANGE = max - min\n(2-step computation)', fontweight='bold')

    # Panel 3: Detail table
    ax = axes[2]
    ax.axis('off')
    rows = [['Method', 'SORT Acc', 'Approach']]
    rows.append(['Sequential', '%.0f%%' % (sort_accs[0]*100), 'MIN then MAX separately'])
    rows.append(['KV-Cache', '%.0f%%' % (sort_accs[1]*100), 'Continue from saved state'])
    rows.append(['Dual-Layer', '%.0f%%' % (sort_accs[2]*100), 'MIN@L6 + MAX@L8 simultaneous'])
    rows.append(['Prompt Chain', '%.0f%%' % (sort_accs[3]*100), 'Result of step 1 in prompt 2'])
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for j in range(3):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Multi-step Composition Methods', fontweight='bold', pad=20)

    plt.suptitle('Phase 174: Multi-step Composition\n'
                 '"Can we chain soul operations to build complex programs?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase174_multistep_composition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 174, 'name': 'multistep_composition',
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase174_multistep_composition.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  P174 completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
