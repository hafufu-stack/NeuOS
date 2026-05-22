# -*- coding: utf-8 -*-
"""
Phase 157: Emergent Skill Discovery
NeuOS detects unknown operations and autonomously trains new souls.

Given input-output pairs for an UNKNOWN operation, the system:
1. Checks if any existing soul matches (cosine similarity)
2. If no match, declares "This is a NEW operation"
3. Auto-trains a new soul vector and adds it to the library
4. Verifies the new skill works

"I don't know what this is, but give me a minute and I'll learn it."
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


def evaluate(model, tok, soul_vec, test_data, device, layer=LAYER):
    correct = 0
    preds = []
    for prompt, expected in test_data:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        preds.append(pred)
        if pred == expected:
            correct += 1
    return correct / len(test_data) if test_data else 0, preds


def check_existing_souls(model, tok, new_data, soul_library, device, layer=LAYER):
    """Try each existing soul on the new data, return best match."""
    best_soul = None
    best_acc = 0
    results = {}
    for name, vec in soul_library.items():
        acc, _ = evaluate(model, tok, vec, new_data, device, layer)
        results[name] = round(acc, 4)
        if acc > best_acc:
            best_acc = acc
            best_soul = name
    return best_soul, best_acc, results


def autonomous_learn(model, tok, train_data, test_data, soul_library, device,
                     layer=LAYER, novelty_threshold=0.6, seed=42):
    """
    Full autonomous skill discovery pipeline:
    1. Check if existing souls can handle this
    2. If not, train a new soul
    3. Verify the new soul
    4. Compare with existing souls (novelty check)
    """
    # Step 1: Can existing souls handle this?
    best_existing, best_acc, match_results = check_existing_souls(
        model, tok, test_data, soul_library, device, layer)

    is_novel = best_acc < novelty_threshold
    diagnosis = {
        'best_existing_soul': best_existing,
        'best_existing_acc': round(best_acc, 4),
        'all_match_results': match_results,
        'is_novel': is_novel,
    }

    if not is_novel:
        diagnosis['action'] = 'REUSE_EXISTING'
        diagnosis['new_soul_acc'] = None
        return diagnosis, None

    # Step 2: This is novel! Train a new soul
    new_soul = train_soul(model, tok, train_data, device, layer, epochs=100, seed=seed)

    # Step 3: Verify
    new_acc, new_preds = evaluate(model, tok, new_soul, test_data, device, layer)
    diagnosis['new_soul_acc'] = round(new_acc, 4)
    diagnosis['new_preds'] = new_preds
    diagnosis['action'] = 'LEARNED_NEW' if new_acc > best_acc else 'LEARN_FAILED'

    # Step 4: Cosine distance to existing souls (novelty confirmation)
    cos_distances = {}
    for name, vec in soul_library.items():
        cos = torch.nn.functional.cosine_similarity(
            new_soul.unsqueeze(0), vec.unsqueeze(0)).item()
        cos_distances[name] = round(cos, 4)
    diagnosis['cos_to_existing'] = cos_distances

    return diagnosis, new_soul


def main():
    print("[P157] Emergent Skill Discovery")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Build initial soul library (MIN, MAX only)
    print("  Building initial soul library (MIN, MAX)...")
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 4) =","4"),
                ("6, 1) =","1"),("2, 8) =","2"),("5, 9) =","5"),
                ("1, 3) =","1")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 4) =","7"),
                ("6, 1) =","6"),("2, 8) =","8"),("5, 9) =","9"),
                ("1, 3) =","3")]

    soul_library = {
        'MIN': train_soul(model, tok, min_data, DEVICE, seed=42),
        'MAX': train_soul(model, tok, max_data, DEVICE, seed=43),
    }
    print("  Library: %s" % list(soul_library.keys()))

    # Present unknown operations
    unknown_ops = {
        'ADD': {
            'name': 'Unknown Operation Alpha',
            'train': [("3, 2) =","5"),("4, 1) =","5"),("2, 3) =","5"),
                      ("1, 6) =","7"),("5, 3) =","8"),("2, 7) =","9"),
                      ("3, 4) =","7"),("1, 2) =","3"),("4, 4) =","8"),
                      ("2, 1) =","3")],
            'test':  [("1, 3) =","4"),("2, 5) =","7"),("4, 3) =","7"),
                      ("3, 6) =","9"),("1, 8) =","9")],
        },
        'SUB': {
            'name': 'Unknown Operation Beta',
            'train': [("7, 2) =","5"),("5, 1) =","4"),("9, 3) =","6"),
                      ("8, 5) =","3"),("6, 4) =","2"),("4, 1) =","3"),
                      ("3, 2) =","1"),("9, 7) =","2"),("8, 1) =","7"),
                      ("7, 3) =","4")],
            'test':  [("6, 2) =","4"),("9, 5) =","4"),("8, 3) =","5"),
                      ("5, 4) =","1"),("7, 1) =","6")],
        },
        'MIN_KNOWN': {
            'name': 'Redundant Operation Gamma',
            'train': [("4, 8) =","4"),("7, 1) =","1"),("6, 3) =","3"),
                      ("9, 2) =","2"),("5, 8) =","5")],
            'test':  [("3, 7) =","3"),("8, 2) =","2"),("6, 1) =","1"),
                      ("9, 4) =","4"),("5, 7) =","5")],
        },
    }

    all_results = {}
    for op_key, op_info in unknown_ops.items():
        print("\n  === Encountering: %s ===" % op_info['name'])
        print("  (True operation: %s)" % op_key)

        diagnosis, new_soul = autonomous_learn(
            model, tok, op_info['train'], op_info['test'],
            soul_library, DEVICE, seed=hash(op_key) % 1000)

        if diagnosis['is_novel']:
            if diagnosis['action'] == 'LEARNED_NEW':
                new_name = op_info['name'].split()[-1]  # "Alpha", "Beta"
                soul_library[new_name] = new_soul
                print("  NOVEL! Learned new skill '%s' (acc=%.0f%%)" % (
                    new_name, diagnosis['new_soul_acc'] * 100))
                print("  Library now: %s" % list(soul_library.keys()))
            else:
                print("  NOVEL but learning failed (acc=%.0f%%)" % (
                    diagnosis['new_soul_acc'] * 100 if diagnosis['new_soul_acc'] else 0))
        else:
            print("  KNOWN! Matches existing soul '%s' (acc=%.0f%%)" % (
                diagnosis['best_existing_soul'], diagnosis['best_existing_acc'] * 100))

        all_results[op_key] = {
            'display_name': op_info['name'],
            'diagnosis': diagnosis,
        }

    # Summary
    print("\n  === SKILL DISCOVERY SUMMARY ===")
    print("  Final library: %s" % list(soul_library.keys()))
    novel_count = sum(1 for r in all_results.values() if r['diagnosis']['is_novel'])
    reuse_count = sum(1 for r in all_results.values() if not r['diagnosis']['is_novel'])
    print("  Novel skills discovered: %d" % novel_count)
    print("  Existing skills reused: %d" % reuse_count)

    # ---- PLOT ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Discovery timeline
    ax = axes[0]
    op_names = list(all_results.keys())
    colors = []
    labels = []
    accs = []
    for op in op_names:
        r = all_results[op]['diagnosis']
        if r['is_novel'] and r['action'] == 'LEARNED_NEW':
            colors.append('#4CAF50')
            labels.append('NEW: %s' % op)
            accs.append(r['new_soul_acc'])
        elif r['is_novel']:
            colors.append('#F44336')
            labels.append('FAILED: %s' % op)
            accs.append(r.get('new_soul_acc', 0) or 0)
        else:
            colors.append('#2196F3')
            labels.append('KNOWN: %s' % op)
            accs.append(r['best_existing_acc'])

    bars = ax.bar(range(len(op_names)), accs, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(op_names)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                '%.0f%%' % (val*100), ha='center', fontweight='bold')
    ax.set_title('Skill Discovery Timeline', fontweight='bold')

    # Panel 2: Library growth
    ax = axes[1]
    lib_sizes = [2]  # Start with MIN, MAX
    lib_labels = ['Initial\n(MIN, MAX)']
    current_size = 2
    for op in op_names:
        r = all_results[op]['diagnosis']
        if r['is_novel'] and r['action'] == 'LEARNED_NEW':
            current_size += 1
        lib_sizes.append(current_size)
        lib_labels.append('After\n%s' % op)

    ax.plot(range(len(lib_sizes)), lib_sizes, 'go-', linewidth=3, markersize=10)
    ax.fill_between(range(len(lib_sizes)), lib_sizes, alpha=0.1, color='green')
    ax.set_xticks(range(len(lib_sizes)))
    ax.set_xticklabels(lib_labels, fontsize=8)
    ax.set_ylabel('Soul Library Size')
    ax.set_title('Autonomous Library Growth', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 3: Cosine similarity of new souls to existing
    ax = axes[2]
    cos_data = {}
    for op in op_names:
        r = all_results[op]['diagnosis']
        if 'cos_to_existing' in r:
            cos_data[op] = r['cos_to_existing']

    if cos_data:
        existing_names = list(next(iter(cos_data.values())).keys())
        x = np.arange(len(cos_data))
        w = 0.8 / len(existing_names)
        for ei, ex_name in enumerate(existing_names):
            vals = [cos_data[op].get(ex_name, 0) for op in cos_data]
            ax.bar(x + ei * w, vals, w, label='vs %s' % ex_name, edgecolor='black')
        ax.set_xticks(x + w * len(existing_names) / 2)
        ax.set_xticklabels(list(cos_data.keys()), fontsize=10)
        ax.set_ylabel('Cosine Similarity')
        ax.legend(fontsize=8)
    ax.set_title('New Soul Novelty\n(low cos = truly new)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 157: Emergent Skill Discovery\n'
                 '"I don\'t know what this is, but give me a minute..."',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase157_skill_discovery.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 157, 'name': 'emergent_skill_discovery',
        'initial_library': ['MIN', 'MAX'],
        'final_library': list(soul_library.keys()),
        'novel_discovered': novel_count,
        'existing_reused': reuse_count,
        'results': {k: v for k, v in all_results.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase157_skill_discovery.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
