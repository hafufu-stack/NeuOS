# -*- coding: utf-8 -*-
"""
Phase 149: Time-Space Hypervisor
Can looping through the model twice emulate a deeper network?

"If the brain is too small, think twice."
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


def single_pass(model, tok, prompt, device, soul_vec=None, layer=LAYER):
    """Single forward pass with optional soul injection. Returns predicted token."""
    hooks = []
    if soul_vec is not None:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        hooks.append(model.model.layers[layer].register_forward_hook(inj))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    for h in hooks:
        h.remove()
    pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
    return pred


def double_pass_text(model, tok, prompt1, prompt2_template, device,
                     soul1=None, soul2=None, layer=LAYER):
    """Two-pass via text: Pass 1 generates intermediate, Pass 2 uses it."""
    # Pass 1: get intermediate answer
    intermediate = single_pass(model, tok, prompt1, device, soul1, layer)
    # Pass 2: build new prompt with intermediate result
    prompt2 = prompt2_template.replace('{INT}', intermediate)
    final = single_pass(model, tok, prompt2, device, soul2, layer)
    return intermediate, final


def double_pass_latent(model, tok, prompt, device, soul_vec=None, layer=LAYER):
    """Two-pass via latent transfer: inject L23 output from pass 1 into L0 of pass 2."""
    captured_final = {}

    # Pass 1: capture final layer output
    def capture_hook(m, i, o):
        tensor = o[0] if isinstance(o, tuple) else o
        if tensor.dim() == 3:
            captured_final['vec'] = tensor[0, -1, :].detach().clone()
        else:
            captured_final['vec'] = tensor[-1, :].detach().clone()

    hooks = []
    n_layers = model.config.num_hidden_layers
    hooks.append(model.model.layers[n_layers - 1].register_forward_hook(capture_hook))
    if soul_vec is not None:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        hooks.append(model.model.layers[layer].register_forward_hook(inj))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)
    for h in hooks:
        h.remove()

    if 'vec' not in captured_final:
        return single_pass(model, tok, prompt, device, soul_vec, layer)

    # Pass 2: inject captured vector at L0
    latent_vec = captured_final['vec']
    hooks2 = []
    def inject_latent(m, i, o, v=latent_vec): return replace_last_token(o, v)
    hooks2.append(model.model.layers[0].register_forward_hook(inject_latent))
    if soul_vec is not None:
        def inj2(m, i, o, v=soul_vec): return replace_last_token(o, v)
        hooks2.append(model.model.layers[layer].register_forward_hook(inj2))

    with torch.no_grad():
        out = model(**inp)
    for h in hooks2:
        h.remove()
    pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
    return pred


def main():
    print("[P149] Time-Space Hypervisor")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train souls
    print("  Training MIN, MAX, ADD souls...")
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1"),("8, 5) =","5"),("6, 2) =","2"),
                ("9, 7) =","7"),("4, 1) =","1"),("3, 8) =","3")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3"),("8, 5) =","8"),("6, 2) =","6"),
                ("9, 7) =","9"),("4, 1) =","4"),("3, 8) =","8")]
    add_data = [("3, 2) =","5"),("4, 1) =","5"),("2, 3) =","5"),
                ("1, 6) =","7"),("5, 3) =","8"),("2, 7) =","9"),
                ("3, 4) =","7"),("1, 2) =","3"),("4, 4) =","8"),
                ("2, 1) =","3"),("3, 3) =","6"),("1, 1) =","2"),
                ("5, 4) =","9"),("2, 6) =","8"),("1, 4) =","5")]

    soul_min = train_soul(model, tok, min_data, DEVICE, seed=42)
    soul_max = train_soul(model, tok, max_data, DEVICE, seed=43)
    soul_add = train_soul(model, tok, add_data, DEVICE, seed=44)

    # === Test Cases ===
    # Type 1: Single-step (baseline)
    single_tests = [
        ("7, 2) =", "2", soul_min, "MIN(7,2)"),
        ("6, 3) =", "3", soul_min, "MIN(6,3)"),
        ("2, 9) =", "2", soul_min, "MIN(2,9)"),
        ("1, 5) =", "1", soul_min, "MIN(1,5)"),
        ("8, 4) =", "4", soul_min, "MIN(8,4)"),
        ("7, 2) =", "7", soul_max, "MAX(7,2)"),
        ("6, 3) =", "6", soul_max, "MAX(6,3)"),
        ("2, 9) =", "9", soul_max, "MAX(2,9)"),
        ("1, 5) =", "5", soul_max, "MAX(1,5)"),
        ("8, 4) =", "8", soul_max, "MAX(8,4)"),
    ]

    # Type 2: Two-step via text (MAX then MIN)
    two_step_tests = [
        # max(a,b), then min(result, c)
        {"a": 3, "b": 7, "c": 5, "expected_int": "7", "expected_final": "5",
         "desc": "min(max(3,7),5)"},
        {"a": 2, "b": 8, "c": 6, "expected_int": "8", "expected_final": "6",
         "desc": "min(max(2,8),6)"},
        {"a": 4, "b": 1, "c": 3, "expected_int": "4", "expected_final": "3",
         "desc": "min(max(4,1),3)"},
        {"a": 6, "b": 9, "c": 7, "expected_int": "9", "expected_final": "7",
         "desc": "min(max(6,9),7)"},
        {"a": 5, "b": 3, "c": 4, "expected_int": "5", "expected_final": "4",
         "desc": "min(max(5,3),4)"},
        {"a": 1, "b": 8, "c": 2, "expected_int": "8", "expected_final": "2",
         "desc": "min(max(1,8),2)"},
        {"a": 7, "b": 2, "c": 9, "expected_int": "7", "expected_final": "7",
         "desc": "min(max(7,2),9)"},
        {"a": 3, "b": 5, "c": 1, "expected_int": "5", "expected_final": "1",
         "desc": "min(max(3,5),1)"},
        {"a": 9, "b": 4, "c": 6, "expected_int": "9", "expected_final": "6",
         "desc": "min(max(9,4),6)"},
        {"a": 2, "b": 6, "c": 3, "expected_int": "6", "expected_final": "3",
         "desc": "min(max(2,6),3)"},
    ]

    # === Run Tests ===
    print("\n  --- Single-Pass Baseline ---")
    single_results = []
    for prompt, expected, soul, desc in single_tests:
        pred = single_pass(model, tok, prompt, DEVICE, soul)
        correct = (pred == expected)
        single_results.append({'desc': desc, 'pred': pred, 'expected': expected,
                               'correct': correct})
        print("  %s: pred=%s expected=%s %s" % (desc, pred, expected,
              'OK' if correct else 'WRONG'))
    single_acc = sum(1 for r in single_results if r['correct']) / len(single_results)
    print("  Single-pass accuracy: %.0f%%" % (single_acc * 100))

    print("\n  --- Double-Pass (Text Transfer) ---")
    double_text_results = []
    for t in two_step_tests:
        prompt1 = "%d, %d) =" % (t['a'], t['b'])
        prompt2_template = "{INT}, %d) =" % t['c']
        intermediate, final = double_pass_text(
            model, tok, prompt1, prompt2_template, DEVICE, soul_max, soul_min)
        int_correct = (intermediate == t['expected_int'])
        final_correct = (final == t['expected_final'])
        double_text_results.append({
            'desc': t['desc'], 'intermediate': intermediate, 'final': final,
            'expected_int': t['expected_int'], 'expected_final': t['expected_final'],
            'int_correct': int_correct, 'final_correct': final_correct,
        })
        print("  %s: int=%s(%s) final=%s(%s) %s" % (
            t['desc'], intermediate, t['expected_int'], final, t['expected_final'],
            'OK' if final_correct else 'WRONG'))
    double_text_int_acc = sum(1 for r in double_text_results if r['int_correct']) / len(double_text_results)
    double_text_final_acc = sum(1 for r in double_text_results if r['final_correct']) / len(double_text_results)
    print("  Double-pass text: int_acc=%.0f%% final_acc=%.0f%%" % (
        double_text_int_acc * 100, double_text_final_acc * 100))

    print("\n  --- Double-Pass (Latent Transfer) ---")
    latent_results = []
    for prompt, expected, soul, desc in single_tests:
        pred = double_pass_latent(model, tok, prompt, DEVICE, soul)
        correct = (pred == expected)
        latent_results.append({'desc': desc, 'pred': pred, 'expected': expected,
                               'correct': correct})
        print("  %s: pred=%s expected=%s %s" % (desc, pred, expected,
              'OK' if correct else 'WRONG'))
    latent_acc = sum(1 for r in latent_results if r['correct']) / len(latent_results)
    print("  Latent transfer accuracy: %.0f%%" % (latent_acc * 100))

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Single vs Double-pass bar chart
    ax = axes[0]
    categories = ['Single-Step\n(1 pass)', 'Two-Step\nIntermediate', 'Two-Step\nFinal']
    values = [single_acc, double_text_int_acc, double_text_final_acc]
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy')
    ax.set_title('Single-Pass vs Double-Pass\n(Text Transfer)', fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    # Panel 2: Pipeline diagram (as text table)
    ax = axes[1]
    ax.axis('off')
    pipeline_data = [
        ['Step', 'Operation', 'Soul', 'Input', 'Output'],
        ['Pass 1', 'MAX(a,b)', 'soul_max @ L8', 'a, b) =', 'intermediate'],
        ['Pass 2', 'MIN(int,c)', 'soul_min @ L8', 'int, c) =', 'final answer'],
    ]
    table = ax.table(cellText=pipeline_data[1:], colLabels=pipeline_data[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    for j in range(5):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
        table[1, j].set_facecolor('#E3F2FD')
        table[2, j].set_facecolor('#E8F5E9')
    ax.set_title('Time-Space Hypervisor Pipeline\n"Think Twice"', fontweight='bold', pad=20)

    # Panel 3: Latent vs Text Transfer
    ax = axes[2]
    methods = ['Single Pass', 'Double Pass\n(Text)', 'Double Pass\n(Latent)']
    accs = [single_acc, double_text_final_acc, latent_acc]
    bar_colors = ['#2196F3', '#4CAF50', '#FF5722']
    bars = ax.bar(methods, accs, color=bar_colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy')
    ax.set_title('Transfer Method Comparison', fontweight='bold')

    plt.suptitle('Phase 149: Time-Space Hypervisor\n'
                 '"If the brain is too small, think twice"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase149_time_space_hypervisor.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 149, 'name': 'time_space_hypervisor',
        'single_pass_accuracy': round(single_acc, 4),
        'double_pass_text_intermediate_accuracy': round(double_text_int_acc, 4),
        'double_pass_text_final_accuracy': round(double_text_final_acc, 4),
        'latent_transfer_accuracy': round(latent_acc, 4),
        'single_results': single_results,
        'double_text_results': double_text_results,
        'latent_results': latent_results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase149_time_space_hypervisor.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
