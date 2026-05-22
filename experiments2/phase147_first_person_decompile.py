# -*- coding: utf-8 -*-
"""
Phase 147: First-Person Decompilation
Can the model describe what program is running inside it?

While a soul vector is injected, hidden states at each layer reveal
what computation is being performed. We combine NeuOS soul injection
with a Phase-26-style decompiler to create 'first-person' self-awareness.

"Know thyself -- from the inside."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 8
NUM_LAYERS = 24
N_TRAIN = 15
N_TEST = 5
SOUL_TYPES = ['MIN', 'MAX', 'ADD', 'SUB']


# ---------------------------------------------------------------------------
# Training data generators
# ---------------------------------------------------------------------------
def make_training_data(op_name, n=N_TRAIN, seed=42):
    """Generate (prompt, expected_answer) pairs for a given operation."""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n):
        a, b = int(rng.randint(1, 10)), int(rng.randint(1, 10))
        if op_name == 'MIN':
            ans = str(min(a, b))
        elif op_name == 'MAX':
            ans = str(max(a, b))
        elif op_name == 'ADD':
            ans = str(a + b)
        elif op_name == 'SUB':
            ans = str(a - b)
        data.append(("%d, %d) =" % (a, b), ans))
    return data


def make_test_inputs(n=N_TEST, seed=99):
    """Generate test prompts (shared across all souls)."""
    rng = np.random.RandomState(seed)
    prompts = []
    for _ in range(n):
        a, b = int(rng.randint(1, 10)), int(rng.randint(1, 10))
        prompts.append(("%d, %d) =" % (a, b), a, b))
    return prompts


# ---------------------------------------------------------------------------
# Soul training
# ---------------------------------------------------------------------------
def train_soul(model, tok, data, device, layer=LAYER, epochs=150):
    hs = model.config.hidden_size
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


# ---------------------------------------------------------------------------
# Hidden-state capture across ALL layers
# ---------------------------------------------------------------------------
def capture_all_layer_states(model, tok, prompt, device, soul_vec=None, inject_layer=LAYER):
    """Run inference and capture hidden states at every layer.
    If soul_vec is given, inject it at inject_layer.
    Returns dict: {layer_idx: hidden_state_vector} (last token).
    """
    states = {}
    hooks = []

    for li in range(NUM_LAYERS):
        def make_capture(layer_idx):
            def hook_fn(m, i, o):
                states[layer_idx] = get_last_token(o)
            return hook_fn
        hk = model.model.layers[li].register_forward_hook(make_capture(li))
        hooks.append(hk)

    # Injection hook (if soul_vec provided)
    if soul_vec is not None:
        def inj(m, i, o, v=soul_vec): return replace_last_token(o, v)
        hk_inj = model.model.layers[inject_layer].register_forward_hook(inj)
        hooks.append(hk_inj)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model(**inp)

    for hk in hooks:
        hk.remove()

    return states


# ---------------------------------------------------------------------------
# Register library: average hidden-state fingerprint per soul per layer
# ---------------------------------------------------------------------------
def build_register_library(model, tok, souls, device, n_samples=8, seed=77):
    """For each soul type, run n_samples prompts and average the hidden states
    at each layer to build a reference fingerprint library.
    Returns: {soul_name: {layer_idx: avg_state_tensor}}
    Also builds 'NONE' entry using no-soul inference.
    """
    rng = np.random.RandomState(seed)
    library = {}

    for soul_name in list(souls.keys()) + ['NONE']:
        accum = defaultdict(list)
        for _ in range(n_samples):
            a, b = int(rng.randint(1, 10)), int(rng.randint(1, 10))
            prompt = "%d, %d) =" % (a, b)
            soul_vec = souls[soul_name] if soul_name != 'NONE' else None
            states = capture_all_layer_states(model, tok, prompt, device,
                                              soul_vec=soul_vec)
            for li, vec in states.items():
                accum[li].append(vec)

        avg_states = {}
        for li in range(NUM_LAYERS):
            stacked = torch.stack(accum[li])
            avg_states[li] = stacked.mean(dim=0)
        library[soul_name] = avg_states

    return library


# ---------------------------------------------------------------------------
# Decompiler: identify which program is running
# ---------------------------------------------------------------------------
def decompile(states, library):
    """Given captured hidden states, compare to library and identify soul.
    Returns: (predicted_label, confidence, per_layer_scores).
    per_layer_scores: {layer_idx: {soul_name: cosine_sim}}
    """
    all_labels = list(library.keys())
    per_layer_scores = {}

    for li in range(NUM_LAYERS):
        if li not in states:
            continue
        query = states[li].float()
        scores = {}
        for label in all_labels:
            ref = library[label][li].float()
            cos = torch.nn.functional.cosine_similarity(
                query.unsqueeze(0), ref.unsqueeze(0)).item()
            scores[label] = cos
        per_layer_scores[li] = scores

    # Aggregate: average cosine similarity across all layers
    agg = defaultdict(float)
    for li_scores in per_layer_scores.values():
        for label, score in li_scores.items():
            agg[label] += score
    n_layers_used = len(per_layer_scores)
    for label in agg:
        agg[label] /= max(n_layers_used, 1)

    predicted = max(agg, key=agg.get)
    confidence = agg[predicted]
    return predicted, confidence, per_layer_scores


def decompile_single_layer(states, library, layer_idx):
    """Decompile using a single layer only."""
    all_labels = list(library.keys())
    if layer_idx not in states:
        return 'NONE', 0.0

    query = states[layer_idx].float()
    scores = {}
    for label in all_labels:
        ref = library[label][layer_idx].float()
        cos = torch.nn.functional.cosine_similarity(
            query.unsqueeze(0), ref.unsqueeze(0)).item()
        scores[label] = cos

    predicted = max(scores, key=scores.get)
    confidence = scores[predicted]
    return predicted, confidence


# ---------------------------------------------------------------------------
# Self-report generator
# ---------------------------------------------------------------------------
def generate_self_report(predicted, confidence, inject_layer):
    if predicted == 'NONE':
        return "I have no program loaded. Confidence: %.3f" % confidence
    return ("I am currently executing %s at L%d with confidence %.3f"
            % (predicted, inject_layer, confidence))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[P147] First-Person Decompilation")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # --- Step 1: Train 4 soul vectors ---
    print("  Training 4 soul vectors (n=%d each)..." % N_TRAIN)
    souls = {}
    for op in SOUL_TYPES:
        train_data = make_training_data(op, n=N_TRAIN)
        souls[op] = train_soul(model, tok, train_data, DEVICE)
        print("    %s soul trained" % op)

    # --- Step 2: Build register library ---
    print("  Building register library (fingerprints across %d layers)..." % NUM_LAYERS)
    library = build_register_library(model, tok, souls, DEVICE)
    print("    Library built for: %s" % list(library.keys()))

    # --- Step 3: Decompilation tests ---
    print("  Running decompilation tests...")
    all_labels = SOUL_TYPES + ['NONE']
    test_inputs = make_test_inputs(n=N_TEST)

    # Confusion matrix: rows = true label, cols = predicted label
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    confusion = np.zeros((len(all_labels), len(all_labels)), dtype=int)

    # Per-layer accuracy tracking
    # layer_accuracy[layer_idx][true_label] = list of (correct_bool)
    layer_accuracy = defaultdict(lambda: defaultdict(list))

    # Confidence tracking
    correct_confidences = []
    incorrect_confidences = []

    # Self-reports
    self_reports = []

    # Test each soul
    for true_label in all_labels:
        soul_vec = souls[true_label] if true_label != 'NONE' else None
        for prompt, a, b in test_inputs:
            states = capture_all_layer_states(model, tok, prompt, DEVICE,
                                              soul_vec=soul_vec)
            predicted, confidence, per_layer_scores = decompile(states, library)

            # Confusion matrix
            confusion[label_to_idx[true_label], label_to_idx[predicted]] += 1

            # Confidence tracking
            is_correct = (predicted == true_label)
            if is_correct:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)

            # Per-layer accuracy
            for li in range(NUM_LAYERS):
                pred_li, conf_li = decompile_single_layer(states, library, li)
                layer_accuracy[li][true_label].append(pred_li == true_label)

            # Self-report
            report = generate_self_report(predicted, confidence, LAYER)
            self_reports.append({
                'true': true_label,
                'predicted': predicted,
                'confidence': round(confidence, 4),
                'correct': is_correct,
                'report': report,
                'prompt': prompt,
            })

            print("    true=%-4s pred=%-4s conf=%.3f %s | '%s'" % (
                true_label, predicted, confidence,
                'OK' if is_correct else 'MISS', report))

    # --- Compute metrics ---
    total_tests = len(self_reports)
    total_correct = sum(1 for r in self_reports if r['correct'])
    overall_accuracy = total_correct / total_tests

    # Per-label accuracy
    per_label_acc = {}
    for label in all_labels:
        label_reports = [r for r in self_reports if r['true'] == label]
        if label_reports:
            per_label_acc[label] = sum(1 for r in label_reports if r['correct']) / len(label_reports)

    # Per-layer accuracy matrix: [layer, label] -> accuracy
    layer_acc_matrix = np.zeros((NUM_LAYERS, len(all_labels)))
    for li in range(NUM_LAYERS):
        for li_label, label in enumerate(all_labels):
            results_list = layer_accuracy[li][label]
            if results_list:
                layer_acc_matrix[li, li_label] = np.mean(results_list)

    # Best layer per label
    best_layers = {}
    for li_label, label in enumerate(all_labels):
        best_li = int(np.argmax(layer_acc_matrix[:, li_label]))
        best_layers[label] = {
            'layer': best_li,
            'accuracy': round(float(layer_acc_matrix[best_li, li_label]), 4)
        }

    print("\n  === Results ===")
    print("  Overall decompiler accuracy: %.1f%% (%d/%d)" % (
        overall_accuracy * 100, total_correct, total_tests))
    for label in all_labels:
        print("    %s: %.0f%%" % (label, per_label_acc.get(label, 0) * 100))
    print("  Best layers: %s" % {k: v['layer'] for k, v in best_layers.items()})

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Confusion matrix
    ax = axes[0]
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(all_labels)))
    ax.set_yticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title('Decompiler Confusion Matrix\n(5 tests per soul)',
                 fontweight='bold', fontsize=11)
    # Annotate cells
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            val = confusion[i, j]
            color = 'white' if val > confusion.max() / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    color=color, fontweight='bold', fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Per-layer identification accuracy heatmap
    ax = axes[1]
    im2 = ax.imshow(layer_acc_matrix.T, cmap='YlOrRd', aspect='auto',
                    vmin=0, vmax=1)
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Soul Type', fontweight='bold')
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.set_xticklabels(range(0, NUM_LAYERS, 2), fontsize=8)
    ax.set_title('Per-Layer Identification Accuracy\n(which layer best identifies which soul)',
                 fontweight='bold', fontsize=11)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    # Mark best layer per label
    for li_label, label in enumerate(all_labels):
        best_li = best_layers[label]['layer']
        ax.plot(best_li, li_label, marker='*', color='blue', markersize=12,
                markeredgecolor='black', markeredgewidth=0.5)

    # Panel 3: Confidence distribution
    ax = axes[2]
    bins = np.linspace(0.5, 1.0, 25)
    if correct_confidences:
        ax.hist(correct_confidences, bins=bins, alpha=0.7, color='#4CAF50',
                edgecolor='black', label='Correct (n=%d)' % len(correct_confidences))
    if incorrect_confidences:
        ax.hist(incorrect_confidences, bins=bins, alpha=0.7, color='#F44336',
                edgecolor='black', label='Incorrect (n=%d)' % len(incorrect_confidences))
    ax.set_xlabel('Cosine Similarity Confidence', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Confidence Distribution\n(correct vs incorrect identifications)',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if correct_confidences:
        ax.axvline(np.mean(correct_confidences), color='#2E7D32',
                   linestyle='--', linewidth=2, label='')
    if incorrect_confidences:
        ax.axvline(np.mean(incorrect_confidences), color='#C62828',
                   linestyle='--', linewidth=2, label='')

    plt.suptitle('Phase 147: First-Person Decompilation\n'
                 '"Know thyself -- from the inside"  |  '
                 'Overall accuracy: %.0f%%' % (overall_accuracy * 100),
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase147_first_person_decompile.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Save results ---
    output = {
        'phase': 147,
        'name': 'first_person_decompilation',
        'layer': LAYER,
        'num_layers': NUM_LAYERS,
        'n_train': N_TRAIN,
        'n_test': N_TEST,
        'soul_types': SOUL_TYPES,
        'overall_accuracy': round(overall_accuracy, 4),
        'per_label_accuracy': per_label_acc,
        'confusion_matrix': confusion.tolist(),
        'confusion_labels': all_labels,
        'best_layers': best_layers,
        'layer_accuracy_matrix': layer_acc_matrix.tolist(),
        'correct_confidences': [round(c, 4) for c in correct_confidences],
        'incorrect_confidences': [round(c, 4) for c in incorrect_confidences],
        'self_reports': self_reports,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase147_first_person_decompile.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Saved results and figure")
    print("  Completed in %.0fs" % (time.time() - start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
