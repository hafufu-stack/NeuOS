# -*- coding: utf-8 -*-
"""
Phase 145: Prompt Format Invariance
Does the soul generalize across different prompt formats?
Train on "3, 7) =" but test on natural language prompts.

"A true soul transcends the language it was taught in."
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


def train_soul(model, tok, data, device, layer=LAYER, seed=42, epochs=150):
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


def get_prediction(model, tok, vec, prompt, device, layer=LAYER):
    def inj(m, i, o, v=vec): return replace_last_token(o, v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    return tok.decode(out.logits[0, -1, :].argmax().item()).strip()


def evaluate(model, tok, vec, data, device, layer=LAYER):
    c = 0
    for p, e in data:
        pred = get_prediction(model, tok, vec, p, device, layer)
        if pred == e:
            c += 1
    return c / len(data)


def main():
    print("[P145] Prompt Format Invariance")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Training format: "a, b) ="
    min_train = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                  ("4, 6) =","4"),("9, 3) =","3")]

    # Test formats (same numbers, different prompt structures)
    test_pairs = [(7,2,"2"), (6,3,"3"), (2,9,"2"), (1,5,"1"), (8,4,"4")]

    format_templates = {
        'original': lambda a, b: "%d, %d) =" % (a, b),
        'no_paren': lambda a, b: "%d, %d =" % (a, b),
        'colon': lambda a, b: "%d : %d =" % (a, b),
        'arrow': lambda a, b: "%d, %d ->" % (a, b),
        'words_min': lambda a, b: "min(%d, %d) =" % (a, b),
        'words_smaller': lambda a, b: "smaller of %d and %d:" % (a, b),
        'question': lambda a, b: "What is min of %d %d?" % (a, b),
        'reversed': lambda a, b: "%d) %d, =" % (b, a),  # swap argument order in format
        'spaced': lambda a, b: " %d ,  %d ) = " % (a, b),
        'dense': lambda a, b: "%d,%d)=" % (a, b),
        'brackets': lambda a, b: "[%d, %d] =" % (a, b),
        'single_num': lambda a, b: "%d =" % a,  # only first number (control)
    }

    # Also test MAX soul
    max_train = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                  ("4, 6) =","6"),("9, 3) =","9")]
    max_test_pairs = [(7,2,"7"), (6,3,"6"), (2,9,"9"), (1,5,"5"), (8,4,"8")]

    print("  Training MIN and MAX souls...")
    soul_min = train_soul(model, tok, min_train, DEVICE)
    soul_max = train_soul(model, tok, max_train, DEVICE)

    results_min = {}
    results_max = {}
    prediction_details = {}

    for fmt_name, fmt_fn in format_templates.items():
        # MIN evaluation
        test_data = [(fmt_fn(a, b), e) for a, b, e in test_pairs]
        acc = evaluate(model, tok, soul_min, test_data, DEVICE)
        preds = [get_prediction(model, tok, soul_min, fmt_fn(a, b), DEVICE)
                 for a, b, e in test_pairs]
        results_min[fmt_name] = round(acc, 4)

        # MAX evaluation
        test_data_max = [(fmt_fn(a, b), e) for a, b, e in max_test_pairs]
        acc_max = evaluate(model, tok, soul_max, test_data_max, DEVICE)
        preds_max = [get_prediction(model, tok, soul_max, fmt_fn(a, b), DEVICE)
                     for a, b, e in max_test_pairs]
        results_max[fmt_name] = round(acc_max, 4)

        prediction_details[fmt_name] = {
            'min_preds': preds, 'max_preds': preds_max,
            'example': fmt_fn(3, 7)
        }
        print("  %s: MIN=%.0f%% MAX=%.0f%% | example='%s' | min_preds=%s" % (
            fmt_name, acc*100, acc_max*100, fmt_fn(3, 7), preds))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Bar chart comparing formats
    ax = axes[0]
    fmt_names = list(format_templates.keys())
    x = np.arange(len(fmt_names))
    w = 0.35
    min_vals = [results_min[n] for n in fmt_names]
    max_vals = [results_max[n] for n in fmt_names]
    ax.bar(x - w/2, min_vals, w, label='MIN', color='#2196F3', edgecolor='black')
    ax.bar(x + w/2, max_vals, w, label='MAX', color='#FF5722', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(fmt_names, rotation=55, ha='right', fontsize=7)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Soul Accuracy by Prompt Format', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=results_min['original'], color='#2196F3', linestyle='--',
               alpha=0.3)

    # Panel 2: Prediction table
    ax = axes[1]
    ax.axis('off')
    table_data = []
    for fmt_name in fmt_names:
        d = prediction_details[fmt_name]
        row = [fmt_name[:12], d['example'][:15]] + d['min_preds']
        table_data.append(row)
    table_data.append(['[expected]', '-', '2', '3', '2', '1', '4'])
    cols = ['Format', 'Example'] + ['7,2', '6,3', '2,9', '1,5', '8,4']
    table = ax.table(cellText=table_data, colLabels=cols,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)
    for j in range(len(cols)):
        table[0, j].set_facecolor('#1976D2')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight expected row
    n_rows = len(table_data)
    for j in range(len(cols)):
        table[n_rows, j].set_facecolor('#E8F5E9')
    # Color correct/incorrect
    expected = ['2', '3', '2', '1', '4']
    for i, fmt_name in enumerate(fmt_names):
        preds = prediction_details[fmt_name]['min_preds']
        for j, (pred, exp) in enumerate(zip(preds, expected)):
            if pred == exp:
                table[i+1, j+2].set_facecolor('#C8E6C9')
            else:
                table[i+1, j+2].set_facecolor('#FFCDD2')
    ax.set_title('MIN Soul Predictions (green=correct)', fontweight='bold',
                 fontsize=11, pad=20)

    # Panel 3: Format similarity scatter
    ax = axes[2]
    for fmt_name in fmt_names:
        ax.scatter(results_min[fmt_name], results_max[fmt_name],
                   s=100, edgecolors='black', zorder=5)
        ax.annotate(fmt_name[:10], (results_min[fmt_name], results_max[fmt_name]),
                    fontsize=7, ha='center', xytext=(0, 8),
                    textcoords='offset points')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('MIN Accuracy')
    ax.set_ylabel('MAX Accuracy')
    ax.set_title('MIN vs MAX per Format', fontweight='bold')
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 145: Prompt Format Invariance\n'
                 '"A true soul transcends the language it was taught in"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase145_format.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 145, 'name': 'prompt_format_invariance',
        'layer': LAYER,
        'results_min': results_min,
        'results_max': results_max,
        'prediction_details': prediction_details,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase145_format.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
