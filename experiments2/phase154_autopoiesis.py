# -*- coding: utf-8 -*-
"""
Phase 154: Rough Soul Autopoiesis
Natural language "rough instructions" -> auto-refined into precise soul.

Human gives vague description: "sort of pick the smaller one"
NeuOS converts to a rough soul vector, then self-refines via entropy minimization.

"From a whisper of intention, forge an algorithm."
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

# Rough natural language descriptions (varying quality)
ROUGH_DESCRIPTIONS = {
    'MIN': [
        "pick the smaller one",
        "get the little number",
        "return whichever is less",
        "find the minimum",
        "the tinier of the two values",
    ],
    'MAX': [
        "pick the bigger one",
        "get the large number",
        "return whichever is more",
        "find the maximum",
        "the greater of the two values",
    ],
    'ADD': [
        "add them up",
        "combine both numbers",
        "total of the two",
        "sum them together",
        "put both numbers together into one",
    ],
    'SUB': [
        "take the second away from the first",
        "difference between the numbers",
        "first minus second",
        "subtract them",
        "remove the second from the first number",
    ],
}

TASK_DATA = {
    'MIN': {
        'train': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                  ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                  ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                  ("1, 3) =","1")],
        'test':  [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                  ("1, 5) =","1"),("8, 4) =","4")],
    },
    'MAX': {
        'train': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                  ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                  ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                  ("1, 3) =","3")],
        'test':  [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                  ("1, 5) =","5"),("8, 4) =","8")],
    },
    'ADD': {
        'train': [("3, 2) =","5"),("4, 1) =","5"),("2, 3) =","5"),
                  ("1, 6) =","7"),("5, 3) =","8"),("2, 7) =","9"),
                  ("3, 4) =","7"),("1, 2) =","3"),("4, 4) =","8"),
                  ("2, 1) =","3")],
        'test':  [("1, 3) =","4"),("2, 5) =","7"),("4, 3) =","7"),
                  ("3, 6) =","9"),("1, 8) =","9")],
    },
    'SUB': {
        'train': [("7, 2) =","5"),("5, 1) =","4"),("9, 3) =","6"),
                  ("8, 5) =","3"),("6, 4) =","2"),("4, 1) =","3"),
                  ("3, 2) =","1"),("9, 7) =","2"),("8, 1) =","7"),
                  ("7, 3) =","4")],
        'test':  [("6, 2) =","4"),("9, 5) =","4"),("8, 3) =","5"),
                  ("5, 4) =","1"),("7, 1) =","6")],
    },
}


def text_to_rough_soul(model, tok, description, device, layer=LAYER):
    """Convert text description to a rough soul vector via mean-pooled embedding."""
    inp = tok(description, return_tensors='pt').to(device)
    hidden_states = {}

    def make_hook(idx):
        def hook_fn(m, i, o):
            if isinstance(o, tuple):
                hidden_states[idx] = o[0].detach()
            else:
                hidden_states[idx] = o.detach()
        return hook_fn

    hooks = []
    for li in range(model.config.num_hidden_layers):
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**inp)
    for h in hooks:
        h.remove()

    # Use hidden state at the target injection layer, mean over tokens
    hs = hidden_states[layer]  # (1, seq_len, hidden)
    rough_vec = hs[0].mean(dim=0)  # (hidden,)
    return rough_vec


def refine_soul(model, tok, rough_vec, calibration_data, device, layer=LAYER,
                lr=0.02, max_steps=30):
    """
    Self-refine a rough soul by minimizing cross-entropy loss on calibration examples.
    This is 'supervised autopoiesis' - uses a few examples to sharpen the rough intent.
    """
    vec = rough_vec.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=lr)

    history = []
    for step in range(max_steps):
        total_loss = 0
        for prompt, expected in calibration_data:
            tid = tok.encode(expected)[-1]
            def inj(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inj)
            inp = tok(prompt, return_tensors='pt').to(device)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            total_loss += loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        # Evaluate accuracy at this step
        acc = evaluate(model, tok, vec.detach(), calibration_data, device, layer)
        history.append({'step': step, 'loss': total_loss.item(), 'acc': acc})

    return vec.detach(), history


def evaluate(model, tok, soul_vec, test_data, device, layer=LAYER):
    correct = 0
    for prompt, expected in test_data:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == expected:
            correct += 1
    return correct / len(test_data) if test_data else 0


def train_soul_full(model, tok, data, device, layer=LAYER, epochs=100, seed=42):
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


def main():
    print("[P154] Rough Soul Autopoiesis")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    results = {}

    for task_name, descriptions in ROUGH_DESCRIPTIONS.items():
        print("\n  === %s ===" % task_name)
        task = TASK_DATA[task_name]

        # Step 1: Convert rough descriptions to soul vectors
        print("  Step 1: Compiling %d rough descriptions..." % len(descriptions))
        rough_souls = []
        rough_accs_test = []
        for desc in descriptions:
            rough = text_to_rough_soul(model, tok, desc, DEVICE)
            acc = evaluate(model, tok, rough, task['test'], DEVICE)
            rough_souls.append(rough)
            rough_accs_test.append(acc)
            print("    '%s' -> test acc = %.0f%%" % (desc[:35], acc * 100))

        # Average the rough souls
        avg_rough = torch.stack(rough_souls).mean(dim=0)
        avg_rough_acc = evaluate(model, tok, avg_rough, task['test'], DEVICE)
        print("  Averaged rough soul acc: %.0f%%" % (avg_rough_acc * 100))

        # Step 2: Self-refine using only 3 calibration examples
        print("  Step 2: Auto-refining with 3 calibration examples (30 steps)...")
        calibration = task['train'][:3]  # Only 3 examples!
        refined_soul, refine_history = refine_soul(
            model, tok, avg_rough, calibration, DEVICE, max_steps=30)
        refined_acc = evaluate(model, tok, refined_soul, task['test'], DEVICE)
        print("  Refined soul acc: %.0f%%" % (refined_acc * 100))

        # Step 3: Full gradient baseline (100 epochs, 10 examples)
        print("  Step 3: Full gradient baseline...")
        gradient_soul = train_soul_full(model, tok, task['train'], DEVICE)
        gradient_acc = evaluate(model, tok, gradient_soul, task['test'], DEVICE)
        print("  Gradient baseline acc: %.0f%%" % (gradient_acc * 100))

        # Cosine similarity: rough -> refined -> gradient
        cos_rough_refined = torch.nn.functional.cosine_similarity(
            avg_rough.unsqueeze(0), refined_soul.unsqueeze(0)).item()
        cos_refined_gradient = torch.nn.functional.cosine_similarity(
            refined_soul.unsqueeze(0), gradient_soul.unsqueeze(0)).item()
        print("  cos(rough, refined)=%.4f, cos(refined, gradient)=%.4f" % (
            cos_rough_refined, cos_refined_gradient))

        results[task_name] = {
            'rough_accs': [round(a, 4) for a in rough_accs_test],
            'avg_rough_acc': round(avg_rough_acc, 4),
            'refined_acc': round(refined_acc, 4),
            'gradient_acc': round(gradient_acc, 4),
            'cos_rough_refined': round(cos_rough_refined, 4),
            'cos_refined_gradient': round(cos_refined_gradient, 4),
            'refine_history': [{'step': h['step'], 'loss': round(h['loss'], 4),
                                'acc': round(h['acc'], 4)} for h in refine_history],
        }

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    task_names = list(results.keys())

    # Panel 1: 3-stage accuracy comparison
    ax = axes[0]
    x = np.arange(len(task_names))
    w = 0.25
    rough_vals = [results[t]['avg_rough_acc'] for t in task_names]
    refined_vals = [results[t]['refined_acc'] for t in task_names]
    gradient_vals = [results[t]['gradient_acc'] for t in task_names]
    ax.bar(x - w, rough_vals, w, label='Rough (NL only)', color='#FF9800', edgecolor='black')
    ax.bar(x, refined_vals, w, label='Refined (3 examples)', color='#4CAF50', edgecolor='black')
    ax.bar(x + w, gradient_vals, w, label='Gradient (full)', color='#2196F3', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.set_title('Autopoiesis: Rough -> Refined -> Gradient', fontweight='bold')
    for i in range(len(task_names)):
        for j, vals in enumerate([rough_vals, refined_vals, gradient_vals]):
            offset = (j - 1) * w
            ax.text(i + offset, vals[i] + 0.02, '%.0f%%' % (vals[i]*100),
                    ha='center', fontsize=7)

    # Panel 2: Refinement learning curves
    ax = axes[1]
    colors = {'MIN': '#E91E63', 'MAX': '#2196F3', 'ADD': '#4CAF50', 'SUB': '#FF9800'}
    for task in task_names:
        hist = results[task]['refine_history']
        steps = [h['step'] for h in hist]
        accs = [h['acc'] for h in hist]
        ax.plot(steps, accs, '-o', color=colors.get(task, 'gray'),
                label=task, markersize=3, linewidth=2)
    ax.set_xlabel('Refinement Step')
    ax.set_ylabel('Calibration Accuracy')
    ax.set_title('Self-Refinement Learning Curves\n(3 examples only)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # Panel 3: Cosine similarity trajectory
    ax = axes[2]
    cos_rr = [results[t]['cos_rough_refined'] for t in task_names]
    cos_rg = [results[t]['cos_refined_gradient'] for t in task_names]
    x = np.arange(len(task_names))
    ax.bar(x - 0.2, cos_rr, 0.35, label='Rough->Refined', color='#9C27B0', edgecolor='black')
    ax.bar(x + 0.2, cos_rg, 0.35, label='Refined->Gradient', color='#00BCD4', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Soul Vector Alignment\n(does refinement approach gradient?)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 154: Rough Soul Autopoiesis\n'
                 '"From a whisper of intention, forge an algorithm"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase154_autopoiesis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 154, 'name': 'rough_soul_autopoiesis',
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase154_autopoiesis.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
