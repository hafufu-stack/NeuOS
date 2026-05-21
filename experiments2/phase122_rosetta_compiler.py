# -*- coding: utf-8 -*-
"""
Phase 122: Rosetta Soul Compiler (NeuOS x Project Rosetta)
Generate soul vectors from TEXT DESCRIPTIONS instead of gradient training.

"If the soul is a language, can we compile it from words?"
"""
import torch, json, os, gc, numpy as np, time, sys, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Task descriptions (5+ paraphrases each) ---
TASK_DESCRIPTIONS = {
    'MIN': [
        'calculate the minimum value',
        'find the smallest number',
        'return the lesser of two numbers',
        'pick the lower value from the pair',
        'output the minimum of the inputs',
        'select whichever number is smaller',
    ],
    'MAX': [
        'find the maximum number',
        'calculate the largest value',
        'return the greater of two numbers',
        'pick the higher value from the pair',
        'output the maximum of the inputs',
        'select whichever number is bigger',
    ],
    'ADD': [
        'add the numbers together',
        'compute the sum of the inputs',
        'return the total of both values',
        'calculate the addition result',
        'find what the two numbers add up to',
        'sum the pair of numbers',
    ],
    'SUB': [
        'subtract the second from the first',
        'compute the difference of two numbers',
        'take away the second number from the first',
        'find first minus second',
        'calculate the subtraction result',
        'remove the second value from the first',
    ],
}

# Task data for training soul vectors
TASK_DATA = {
    'MIN': {
        'train': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                  ("4, 6) =","4"),("9, 3) =","3")],
        'test':  [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")],
    },
    'MAX': {
        'train': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                  ("4, 6) =","6"),("9, 3) =","9")],
        'test':  [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")],
    },
    'ADD': {
        'train': [("3, 2) =","5"),("1, 4) =","5"),("2, 6) =","8"),
                  ("3, 3) =","6"),("4, 1) =","5")],
        'test':  [("1, 2) =","3"),("3, 4) =","7"),("2, 5) =","7")],
    },
    'SUB': {
        'train': [("7, 3) =","4"),("5, 2) =","3"),("9, 1) =","8"),
                  ("6, 4) =","2"),("8, 3) =","5")],
        'test':  [("9, 5) =","4"),("6, 1) =","5"),("4, 3) =","1")],
    },
}

EMBED_DIM = 64
INJECT_LAYER = 16


def evaluate_vec(model, tok, vec, data, layer, device):
    """Evaluate a soul vector on data, return accuracy."""
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


def gradient_train(model, tok, train, layer, device, seed=42, epochs=150):
    """Standard gradient soul vector training."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in train:
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


def get_text_embedding(model, tok, text, device):
    """Get mean-pooled hidden state from model for a text description."""
    inp = tok(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inp, output_hidden_states=True)
    # Use last hidden layer, mean over all tokens
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden)
    return last_hidden[0].mean(dim=0).cpu().numpy()  # (hidden,)


def main():
    print("[P122] Rosetta Soul Compiler")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    task_names = list(TASK_DESCRIPTIONS.keys())
    all_data = {k: TASK_DATA[k]['train'] + TASK_DATA[k]['test'] for k in task_names}

    # ========== STEP 1: Collect text embeddings ==========
    print("  Step 1: Collecting text embeddings...")
    text_embeddings = {}  # task -> list of (description, embedding)
    for task, descs in TASK_DESCRIPTIONS.items():
        text_embeddings[task] = []
        for desc in descs:
            emb = get_text_embedding(model, tok, desc, DEVICE)
            text_embeddings[task].append((desc, emb))
        print(f"    {task}: {len(descs)} descriptions embedded")

    # Fit PCA on all embeddings -> 64d
    all_embs = []
    for task in task_names:
        for _, emb in text_embeddings[task]:
            all_embs.append(emb)
    all_embs = np.array(all_embs)
    n_pca = min(EMBED_DIM, len(all_embs))
    pca_text = PCA(n_components=n_pca)
    pca_text.fit(all_embs)
    print("    PCA fitted: %dd (requested %dd), explained var = %.4f" % (
          n_pca, EMBED_DIM, pca_text.explained_variance_ratio_.sum()))

    # Transform all embeddings to 64d
    text_emb_pca = {}  # task -> list of 64d vectors
    for task in task_names:
        text_emb_pca[task] = []
        for desc, emb in text_embeddings[task]:
            text_emb_pca[task].append(pca_text.transform(emb.reshape(1, -1))[0])

    # ========== STEP 2: Train gradient souls ==========
    print("  Step 2: Training gradient soul vectors (150 epochs)...")
    gradient_souls = {}
    gradient_accs = {}
    for task in task_names:
        v = gradient_train(model, tok, TASK_DATA[task]['train'],
                           INJECT_LAYER, DEVICE, seed=42, epochs=150)
        gradient_souls[task] = v
        acc = evaluate_vec(model, tok, v, all_data[task], INJECT_LAYER, DEVICE)
        gradient_accs[task] = acc
        print(f"    {task}: gradient acc = {acc:.2%}")

    # ========== STEP 3: Train linear mapping W ==========
    print("  Step 3: Training linear mapping (Ridge regression)...")
    # Build training data for ridge: text_pca -> soul_vector
    X_train_ridge = []
    Y_train_ridge = []
    for task in task_names:
        soul_np = gradient_souls[task].cpu().numpy()
        for emb_pca in text_emb_pca[task]:
            X_train_ridge.append(emb_pca)
            Y_train_ridge.append(soul_np)
    X_train_ridge = np.array(X_train_ridge)
    Y_train_ridge = np.array(Y_train_ridge)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_ridge, Y_train_ridge)
    print(f"    Ridge trained: {X_train_ridge.shape} -> {Y_train_ridge.shape}")

    # ========== STEP 4: Compile souls from text ==========
    print("  Step 4: Compiling souls from text descriptions...")
    compiled_souls = {}  # task -> list of (desc, compiled_vec)
    for task in task_names:
        compiled_souls[task] = []
        for idx, (desc, _) in enumerate(text_embeddings[task]):
            emb_pca = text_emb_pca[task][idx].reshape(1, -1)
            compiled_np = ridge.predict(emb_pca)[0]
            compiled_vec = torch.tensor(compiled_np, dtype=torch.float32, device=DEVICE)
            compiled_souls[task].append((desc, compiled_vec))

    # ========== STEP 5: Evaluate compiled souls ==========
    print("  Step 5: Evaluating compiled souls...")
    compiled_accs = {}
    compiled_per_desc = {}
    for task in task_names:
        accs_for_task = []
        for desc, cvec in compiled_souls[task]:
            acc = evaluate_vec(model, tok, cvec, all_data[task], INJECT_LAYER, DEVICE)
            accs_for_task.append(acc)
            print(f"    {task} | '{desc[:40]}...' -> acc={acc:.2%}")
        compiled_accs[task] = float(np.mean(accs_for_task))
        compiled_per_desc[task] = accs_for_task

    # Cosine similarity between compiled souls (within and across tasks)
    print("  Computing cosine similarities...")
    all_compiled_vecs = []
    all_compiled_labels = []
    for task in task_names:
        for desc, cvec in compiled_souls[task]:
            all_compiled_vecs.append(cvec.cpu().numpy())
            all_compiled_labels.append(task)
    compiled_matrix = np.array(all_compiled_vecs)
    norms = np.linalg.norm(compiled_matrix, axis=1, keepdims=True)
    compiled_normed = compiled_matrix / (norms + 1e-8)
    cos_sim = compiled_normed @ compiled_normed.T

    # Within-task vs cross-task cosine
    within_cos = {}
    cross_cos = {}
    for t1 in task_names:
        idx1 = [i for i, l in enumerate(all_compiled_labels) if l == t1]
        vals = cos_sim[np.ix_(idx1, idx1)]
        triu = vals[np.triu_indices(len(idx1), k=1)]
        within_cos[t1] = float(np.mean(triu)) if len(triu) > 0 else 0.0
        for t2 in task_names:
            if t1 >= t2:
                continue
            idx2 = [i for i, l in enumerate(all_compiled_labels) if l == t2]
            cross_vals = cos_sim[np.ix_(idx1, idx2)].flatten()
            cross_cos[f'{t1}_vs_{t2}'] = float(np.mean(cross_vals))

    # Cosine between compiled and gradient souls
    comp_vs_grad_cos = {}
    for task in task_names:
        grad_np = gradient_souls[task].cpu().numpy()
        grad_norm = grad_np / (np.linalg.norm(grad_np) + 1e-8)
        sims = []
        for _, cvec in compiled_souls[task]:
            c_np = cvec.cpu().numpy()
            c_norm = c_np / (np.linalg.norm(c_np) + 1e-8)
            sims.append(float(np.dot(grad_norm, c_norm)))
        comp_vs_grad_cos[task] = float(np.mean(sims))
        print(f"    {task}: compiled-vs-gradient cosine = {np.mean(sims):.4f}")

    # ========== PCA for visualization ==========
    # Include gradient souls in the PCA
    all_viz_vecs = []
    all_viz_labels = []
    all_viz_types = []  # 'gradient' or 'compiled'
    for task in task_names:
        all_viz_vecs.append(gradient_souls[task].cpu().numpy())
        all_viz_labels.append(task)
        all_viz_types.append('gradient')
        for _, cvec in compiled_souls[task]:
            all_viz_vecs.append(cvec.cpu().numpy())
            all_viz_labels.append(task)
            all_viz_types.append('compiled')
    viz_X = np.array(all_viz_vecs)
    pca_viz = PCA(n_components=2)
    viz_2d = pca_viz.fit_transform(viz_X)

    # ========== Save results ==========
    output = {
        'phase': 122, 'name': 'rosetta_soul_compiler',
        'gradient_accs': {k: round(v, 4) for k, v in gradient_accs.items()},
        'compiled_accs': {k: round(v, 4) for k, v in compiled_accs.items()},
        'compiled_vs_gradient_cosine': {k: round(v, 4) for k, v in comp_vs_grad_cos.items()},
        'within_task_cosine': {k: round(v, 4) for k, v in within_cos.items()},
        'cross_task_cosine': {k: round(v, 4) for k, v in cross_cos.items()},
        'pca_text_explained_var': round(float(pca_text.explained_variance_ratio_.sum()), 4),
        'embed_dim': EMBED_DIM,
        'inject_layer': INJECT_LAYER,
        'n_descriptions_per_task': {k: len(v) for k, v in TASK_DESCRIPTIONS.items()},
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase122_rosetta_compiler.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved.")

    # ========== Plot ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_map = {'MIN': 'tab:blue', 'MAX': 'tab:orange',
                  'ADD': 'tab:green', 'SUB': 'tab:red'}

    # Panel 1: Accuracy comparison (gradient vs compiled)
    ax = axes[0]
    x_pos = np.arange(len(task_names))
    w = 0.35
    grad_vals = [gradient_accs[t] for t in task_names]
    comp_vals = [compiled_accs[t] for t in task_names]
    bars1 = ax.bar(x_pos - w/2, grad_vals, w, label='Gradient', color='steelblue',
                   edgecolor='black')
    bars2 = ax.bar(x_pos + w/2, comp_vals, w, label='Compiled', color='coral',
                   edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_names)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.3)
    ax.set_title('Gradient vs Compiled Accuracy', fontweight='bold')
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.0%}', ha='center', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.0%}', ha='center', fontsize=8)

    # Panel 2: Cosine similarity matrix
    ax = axes[1]
    n_total = len(all_compiled_labels)
    im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1)
    # Add task boundaries
    boundaries = []
    cur = 0
    for task in task_names:
        cnt = sum(1 for l in all_compiled_labels if l == task)
        boundaries.append((cur, cur + cnt, task))
        cur += cnt
    tick_pos = [(s + e) / 2 for s, e, _ in boundaries]
    tick_lab = [t for _, _, t in boundaries]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, fontsize=9)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab, fontsize=9)
    plt.colorbar(im, ax=ax)
    ax.set_title('Compiled Soul Cosine Similarity', fontweight='bold')

    # Panel 3: PCA projection
    ax = axes[2]
    for task in task_names:
        # Compiled
        idx_c = [i for i in range(len(all_viz_labels))
                 if all_viz_labels[i] == task and all_viz_types[i] == 'compiled']
        ax.scatter(viz_2d[idx_c, 0], viz_2d[idx_c, 1],
                   c=colors_map[task], marker='o', s=50, alpha=0.7,
                   edgecolors='black', label=f'{task} (compiled)')
        # Gradient
        idx_g = [i for i in range(len(all_viz_labels))
                 if all_viz_labels[i] == task and all_viz_types[i] == 'gradient']
        ax.scatter(viz_2d[idx_g, 0], viz_2d[idx_g, 1],
                   c=colors_map[task], marker='*', s=200, edgecolors='black',
                   label=f'{task} (gradient)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('PCA: All Souls by Task', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)

    plt.suptitle('Phase 122: Rosetta Soul Compiler\n'
                 '"If the soul is a language, can we compile it from words?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase122_rosetta_compiler.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  === Summary ===")
    for t in task_names:
        print(f"  {t}: gradient={gradient_accs[t]:.2%}, "
              f"compiled={compiled_accs[t]:.2%}, "
              f"cos={comp_vs_grad_cos[t]:.4f}")
    print(f"  Completed in {time.time()-start:.0f}s")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
