# -*- coding: utf-8 -*-
"""
Phase 139: Soul Cross-Task Confusion Matrix
What happens when you inject the WRONG soul?
Full NxN confusion matrix: train soul for task A, test on task B.

"Every mistake reveals the structure of the mind."
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


def evaluate(model, tok, vec, data, device, layer=LAYER):
    c = 0
    for p, e in data:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0, -1, :].argmax().item()).strip() == e:
            c += 1
    return c / len(data)


def get_predictions(model, tok, vec, data, device, layer=LAYER):
    """Get all predictions for analysis."""
    preds = []
    for p, e in data:
        def inj(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append(pred)
    return preds


def main():
    print("[P139] Soul Cross-Task Confusion Matrix")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    task_data = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")],
        'ADD': [("3, 2) =","5"),("1, 4) =","5"),("2, 6) =","8"),
                 ("3, 3) =","6"),("4, 1) =","5")],
        'SUB': [("7, 3) =","4"),("5, 2) =","3"),("9, 1) =","8"),
                 ("6, 4) =","2"),("8, 3) =","5")],
    }
    # Shared prompts for cross-evaluation
    eval_prompts = [("3, 7) =",), ("5, 2) =",), ("8, 1) =",),
                    ("4, 6) =",), ("9, 3) =",)]
    # Expected answers per task
    expected = {
        'MIN': ["3", "2", "1", "4", "3"],
        'MAX': ["7", "5", "8", "6", "9"],
        'ADD': ["10", "7", "9", "10", "12"],  # multi-digit, will fail
        'SUB': ["-4", "3", "7", "-2", "6"],  # negatives possible
    }
    # For accuracy, only use single-digit expected
    eval_data = {
        'MIN': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")],
        'MAX': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")],
    }

    tasks = list(task_data.keys())

    # Train one soul per task
    print("  Training 4 souls...")
    souls = {}
    for task in tasks:
        souls[task] = train_soul(model, tok, task_data[task], DEVICE)

    # No-soul baseline
    print("  Running no-soul baseline...")
    baseline_preds = {}
    for prompt in ["3, 7) =", "5, 2) =", "8, 1) =", "4, 6) =", "9, 3) ="]:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        baseline_preds[prompt] = pred
    print("  Baseline predictions: %s" % list(baseline_preds.values()))

    # Full confusion matrix: inject soul_A, evaluate on task_B criteria
    confusion = np.zeros((len(tasks), len(tasks)))
    prediction_matrix = {}

    for i, soul_task in enumerate(tasks):
        for j, eval_task in enumerate(tasks):
            if eval_task in eval_data:
                acc = evaluate(model, tok, souls[soul_task],
                              eval_data[eval_task], DEVICE)
            else:
                acc = 0.0  # ADD/SUB eval needs special handling
            confusion[i, j] = acc

        # Get actual predictions
        prompts_list = [("3, 7) =",""),("5, 2) =",""),("8, 1) =",""),
                        ("4, 6) =",""),("9, 3) =","")]
        preds = get_predictions(model, tok, souls[soul_task], prompts_list, DEVICE)
        prediction_matrix[soul_task] = preds
        print("  Soul=%s -> outputs: %s" % (soul_task, preds))

    # Analyze: what patterns emerge?
    # For each soul, what is the dominant "behavior"?
    print("\n  Prediction analysis:")
    min_answers = set(["3", "2", "1", "4", "3"])
    max_answers = set(["7", "5", "8", "6", "9"])
    for soul_task in tasks:
        preds = prediction_matrix[soul_task]
        min_match = sum(1 for p, e in zip(preds, ["3","2","1","4","3"]) if p == e)
        max_match = sum(1 for p, e in zip(preds, ["7","5","8","6","9"]) if p == e)
        print("    Soul_%s: MIN-match=%d/5, MAX-match=%d/5, unique=%d" % (
            soul_task, min_match, max_match, len(set(preds))))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Confusion heatmap
    ax = axes[0]
    im = ax.imshow(confusion, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            color = 'white' if confusion[i, j] > 0.5 else 'black'
            ax.text(j, i, "%.0f%%" % (confusion[i, j] * 100),
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color=color)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=11)
    ax.set_xlabel('Evaluated as', fontsize=11)
    ax.set_ylabel('Soul trained for', fontsize=11)
    ax.set_title('Cross-Task Confusion Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: Prediction table
    ax = axes[1]
    ax.axis('off')
    prompts = ["3,7", "5,2", "8,1", "4,6", "9,3"]
    table_data = []
    for soul in tasks:
        row = [soul] + prediction_matrix[soul]
        table_data.append(row)
    # Add expected rows
    table_data.append(['[MIN]', '3', '2', '1', '4', '3'])
    table_data.append(['[MAX]', '7', '5', '8', '6', '9'])
    table = ax.table(cellText=table_data,
                     colLabels=['Soul'] + prompts,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(6):
        table[0, j].set_facecolor('#1976D2')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Color expected rows
    for j in range(6):
        table[5, j].set_facecolor('#E3F2FD')
        table[6, j].set_facecolor('#FBE9E7')
    # Highlight correct predictions
    for i, soul in enumerate(tasks):
        for j, pred in enumerate(prediction_matrix[soul]):
            # Check if matches MIN or MAX
            min_exp = ['3', '2', '1', '4', '3'][j]
            max_exp = ['7', '5', '8', '6', '9'][j]
            if pred == min_exp and soul == 'MIN':
                table[i+1, j+1].set_facecolor('#C8E6C9')
            elif pred == max_exp and soul == 'MAX':
                table[i+1, j+1].set_facecolor('#C8E6C9')
    ax.set_title('Prediction Table', fontweight='bold', fontsize=12, pad=20)

    # Panel 3: Soul similarity vs behavioral similarity
    ax = axes[2]
    # Cosine similarity between souls
    cos_matrix = np.zeros((4, 4))
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            cos_matrix[i, j] = torch.nn.functional.cosine_similarity(
                souls[t1].unsqueeze(0), souls[t2].unsqueeze(0)).item()
    im = ax.imshow(cos_matrix, cmap='RdBu', vmin=-0.2, vmax=0.2, aspect='equal')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "%.3f" % cos_matrix[i, j],
                    ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(tasks)
    ax.set_yticks(range(4))
    ax.set_yticklabels(tasks)
    ax.set_title('Soul Cosine Similarity', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Phase 139: Soul Cross-Task Confusion\n'
                 '"Every mistake reveals the structure of the mind"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase139_confusion.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 139, 'name': 'cross_task_confusion',
        'layer': LAYER,
        'confusion_matrix': confusion.tolist(),
        'cosine_matrix': cos_matrix.tolist(),
        'predictions': prediction_matrix,
        'baseline_preds': baseline_preds,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase139_confusion.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
