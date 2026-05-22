# -*- coding: utf-8 -*-
"""
Phase 156: Multiverse State Forking
Fork the brain state at a decision point, try multiple souls in parallel,
evaluate each fork's entropy, and merge the best future.

"Before I speak, I simulate all possible minds and choose the wisest."
"""
import torch, json, os, gc, numpy as np, time, sys, copy
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


def single_infer(model, tok, prompt, device, soul_vec, layer=LAYER):
    """Single inference with soul injection, returns pred, entropy, logits."""
    def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    pred = tok.decode(logits.argmax().item()).strip()
    top1_prob = probs.max().item()
    return pred, entropy, top1_prob


def multiverse_infer(model, tok, prompt, device, soul_library, layer=LAYER):
    """
    Fork: try all souls in the library, pick the one with lowest entropy.
    This simulates "trying multiple minds" before committing to an answer.
    """
    best_pred = None
    best_entropy = float('inf')
    best_soul_name = None
    best_conf = 0
    fork_results = []

    for soul_name, soul_vec in soul_library.items():
        pred, entropy, conf = single_infer(model, tok, prompt, device, soul_vec, layer)
        fork_results.append({
            'soul': soul_name, 'pred': pred,
            'entropy': round(entropy, 4), 'conf': round(conf, 6)
        })
        if entropy < best_entropy:
            best_entropy = entropy
            best_pred = pred
            best_soul_name = soul_name
            best_conf = conf

    return best_pred, best_entropy, best_soul_name, best_conf, fork_results


def main():
    print("[P156] Multiverse State Forking")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train 4 souls
    print("  Training 4 soul vectors...")
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]
    add_data = [("3, 2) =","5"),("4, 1) =","5"),("2, 3) =","5"),
                ("1, 6) =","7"),("5, 3) =","8"),("2, 7) =","9"),
                ("3, 4) =","7"),("1, 2) =","3"),("4, 4) =","8"),
                ("2, 1) =","3")]
    sub_data = [("7, 2) =","5"),("5, 1) =","4"),("9, 3) =","6"),
                ("8, 5) =","3"),("6, 4) =","2"),("4, 1) =","3"),
                ("3, 2) =","1"),("9, 7) =","2"),("8, 1) =","7"),
                ("7, 3) =","4")]

    soul_min = train_soul(model, tok, min_data, DEVICE, seed=42)
    soul_max = train_soul(model, tok, max_data, DEVICE, seed=43)
    soul_add = train_soul(model, tok, add_data, DEVICE, seed=44)
    soul_sub = train_soul(model, tok, sub_data, DEVICE, seed=45)

    soul_library = {
        'MIN': soul_min, 'MAX': soul_max,
        'ADD': soul_add, 'SUB': soul_sub,
    }

    # Test: model doesn't know which operation to use
    # The "multiverse" tries all 4 and picks the lowest-entropy answer
    test_cases = [
        # The correct soul is unknown to the system!
        ("7, 2) =", "2", "MIN", "Take the smaller"),
        ("6, 3) =", "3", "MIN", "Take the smaller"),
        ("2, 9) =", "2", "MIN", "Take the smaller"),
        ("3, 7) =", "7", "MAX", "Take the larger"),
        ("5, 2) =", "5", "MAX", "Take the larger"),
        ("1, 8) =", "8", "MAX", "Take the larger"),
        ("3, 4) =", "7", "ADD", "Add them"),
        ("2, 5) =", "7", "ADD", "Add them"),
        ("1, 6) =", "7", "ADD", "Add them"),
        ("9, 3) =", "6", "SUB", "Subtract"),
        ("7, 2) =", "5", "SUB", "Subtract"),
        ("8, 5) =", "3", "SUB", "Subtract"),
    ]

    print("\n  --- Single Soul (oracle, knows correct soul) ---")
    oracle_correct = 0
    for prompt, expected, true_soul, desc in test_cases:
        pred, _, _ = single_infer(model, tok, prompt, DEVICE,
                                   soul_library[true_soul])
        if pred == expected:
            oracle_correct += 1
    oracle_acc = oracle_correct / len(test_cases)
    print("  Oracle accuracy: %.0f%%" % (oracle_acc * 100))

    print("\n  --- Multiverse Forking (tries all souls) ---")
    multi_correct = 0
    soul_selection_correct = 0
    all_fork_details = []

    for prompt, expected, true_soul, desc in test_cases:
        pred, ent, chosen, conf, forks = multiverse_infer(
            model, tok, prompt, DEVICE, soul_library)
        correct = (pred == expected)
        soul_correct = (chosen == true_soul)
        if correct:
            multi_correct += 1
        if soul_correct:
            soul_selection_correct += 1

        all_fork_details.append({
            'prompt': prompt[:12], 'desc': desc,
            'expected': expected, 'true_soul': true_soul,
            'chosen_soul': chosen, 'pred': pred,
            'entropy': round(ent, 4), 'correct': correct,
            'soul_correct': soul_correct, 'forks': forks,
        })
        status = "OK" if correct else "WRONG"
        soul_status = "OK" if soul_correct else "X"
        print("  %s | chose=%s(true=%s) %s | pred=%s(exp=%s) %s | H=%.3f" % (
            prompt[:12], chosen, true_soul, soul_status,
            pred, expected, status, ent))

    multi_acc = multi_correct / len(test_cases)
    soul_acc = soul_selection_correct / len(test_cases)
    print("\n  Multiverse accuracy: %.0f%%" % (multi_acc * 100))
    print("  Soul selection accuracy: %.0f%%" % (soul_acc * 100))
    print("  Oracle accuracy: %.0f%%" % (oracle_acc * 100))

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Oracle vs Multiverse
    ax = axes[0]
    bars = ax.bar(['Oracle\n(knows soul)', 'Multiverse\n(entropy fork)',
                   'Soul Selection\nAccuracy'],
                  [oracle_acc, multi_acc, soul_acc],
                  color=['#2196F3', '#4CAF50', '#FF9800'],
                  edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, [oracle_acc, multi_acc, soul_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Accuracy')
    ax.set_title('Oracle vs Multiverse Forking', fontweight='bold')

    # Panel 2: Entropy landscape per fork
    ax = axes[1]
    # Show entropy for each soul across all test cases
    soul_names = list(soul_library.keys())
    for si, soul_name in enumerate(soul_names):
        ents = []
        for detail in all_fork_details:
            for f in detail['forks']:
                if f['soul'] == soul_name:
                    ents.append(f['entropy'])
        ax.boxplot(ents, positions=[si], widths=0.6,
                   patch_artist=True,
                   boxprops=dict(facecolor=['#E91E63','#2196F3','#4CAF50','#FF9800'][si],
                                 alpha=0.5))
    ax.set_xticks(range(len(soul_names)))
    ax.set_xticklabels(soul_names)
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Distribution by Soul Fork', fontweight='bold')

    # Panel 3: Soul selection confusion matrix
    ax = axes[2]
    conf_matrix = np.zeros((4, 4))
    for detail in all_fork_details:
        true_idx = soul_names.index(detail['true_soul'])
        chosen_idx = soul_names.index(detail['chosen_soul'])
        conf_matrix[true_idx, chosen_idx] += 1
    im = ax.imshow(conf_matrix, cmap='Blues')
    ax.set_xticks(range(4))
    ax.set_xticklabels(soul_names, fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(soul_names, fontsize=10)
    ax.set_xlabel('Chosen Soul (by entropy)')
    ax.set_ylabel('True Soul')
    ax.set_title('Soul Selection Confusion', fontweight='bold')
    for i in range(4):
        for j in range(4):
            if conf_matrix[i, j] > 0:
                ax.text(j, i, '%d' % conf_matrix[i, j], ha='center', va='center',
                        fontweight='bold', fontsize=14,
                        color='white' if conf_matrix[i, j] > 1 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Phase 156: Multiverse State Forking\n'
                 '"Before I speak, I simulate all possible minds"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase156_multiverse.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 156, 'name': 'multiverse_state_forking',
        'oracle_accuracy': round(oracle_acc, 4),
        'multiverse_accuracy': round(multi_acc, 4),
        'soul_selection_accuracy': round(soul_acc, 4),
        'fork_details': all_fork_details,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase156_multiverse.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
