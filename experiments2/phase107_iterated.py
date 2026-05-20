# -*- coding: utf-8 -*-
"""
Phase 107: Iterated Learning (Telephone Game / Memetic Drift)
Chain 20 generations of teacher->student imitation (P104 approach).
Does knowledge drift, converge to a Platonic ideal, or maintain fidelity?

"What survives the telephone game is the essence."
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compile_prog(model, tok, train, layer, device, seed=42, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device)*0.01; vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(device)
            def inj(m,i,o,v=vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

def get_teacher_logits(model, tok, vec, prompt, layer, device):
    def inj(m,i,o,v=vec): return replace_last_token(o,v)
    h = model.model.layers[layer].register_forward_hook(inj)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad(): out = model(**inp)
    h.remove()
    return out.logits[0,-1,:].detach()

def imitate(model, tok, teacher_vec, layer, device, prompts, epochs=100, seed=0):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    student = torch.randn(hs, device=device)*0.01; student.requires_grad_(True)
    opt = torch.optim.Adam([student], lr=0.01)
    # Collect teacher demonstrations
    traces = []
    for p in prompts:
        logits = get_teacher_logits(model, tok, teacher_vec, p, layer, device)
        traces.append((p, logits))
    for ep in range(epochs):
        for p, t_logits in traces:
            inp = tok(p, return_tensors='pt').to(device)
            def inj(m,i,o,v=student): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            s_logp = torch.nn.functional.log_softmax(out.logits[0,-1,:], dim=-1)
            t_p = torch.nn.functional.softmax(t_logits, dim=-1)
            loss = torch.nn.functional.kl_div(s_logp, t_p, reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
    return student.detach()

def evaluate_vec(model, tok, vec, data, layer, device):
    c = 0
    for p, e in data:
        def inj(m,i,o,v=vec): return replace_last_token(o,v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
    return c / len(data)

def main():
    print("[P107] Iterated Learning (Telephone Game)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    tl = 8
    for p in model.parameters(): p.requires_grad = False

    train_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                  ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("5, 4) =","4"),("3, 8) =","3")]
    all_data = train_data + test_data
    prompts = [p for p, _ in all_data]

    N_GEN = 20
    print(f"  Running {N_GEN}-generation telephone game...")

    # Generation 0: Train original teacher from data
    teacher = compile_prog(model, tok, train_data, tl, DEVICE, seed=42, epochs=100)
    gen0_acc = evaluate_vec(model, tok, teacher, all_data, tl, DEVICE)

    lineage = [teacher.cpu().numpy().flatten()]
    accuracies = [float(gen0_acc)]
    cosines_to_gen0 = [1.0]
    norms = [float(teacher.norm().item())]
    print(f"    Gen 0 (original): acc={gen0_acc:.0%}, norm={teacher.norm():.3f}")

    current = teacher
    for gen in range(1, N_GEN + 1):
        student = imitate(model, tok, current, tl, DEVICE, prompts,
                         epochs=80, seed=gen*111)
        acc = evaluate_vec(model, tok, student, all_data, tl, DEVICE)
        cos_to_0 = float(torch.nn.functional.cosine_similarity(
            student.unsqueeze(0), teacher.unsqueeze(0)).item())
        lineage.append(student.cpu().numpy().flatten())
        accuracies.append(float(acc))
        cosines_to_gen0.append(cos_to_0)
        norms.append(float(student.norm().item()))
        if gen % 5 == 0 or gen == 1:
            print(f"    Gen {gen}: acc={acc:.0%}, cos(gen0)={cos_to_0:.3f}, "
                  f"norm={student.norm():.3f}")
        current = student

    # PCA trajectory
    X = np.array(lineage)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Consecutive cosines
    consecutive_cos = []
    for i in range(1, len(lineage)):
        v1 = torch.tensor(lineage[i-1]); v2 = torch.tensor(lineage[i])
        c = float(torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)).item())
        consecutive_cos.append(c)

    output = {
        'phase': 107, 'name': 'iterated_learning',
        'n_generations': N_GEN,
        'accuracies': [round(a, 4) for a in accuracies],
        'cosines_to_gen0': [round(c, 4) for c in cosines_to_gen0],
        'consecutive_cosines': [round(c, 4) for c in consecutive_cos],
        'norms': [round(n, 4) for n in norms],
        'final_accuracy': round(accuracies[-1], 4),
        'final_cos_to_gen0': round(cosines_to_gen0[-1], 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase107_iterated.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    gens = list(range(N_GEN + 1))

    axes[0].plot(gens, accuracies, 'b-o', lw=2, ms=4)
    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Knowledge Fidelity Across Generations', fontweight='bold')
    axes[0].set_ylim(0, 1.1); axes[0].grid(True, alpha=0.3)

    axes[1].plot(gens, cosines_to_gen0, 'r-o', lw=2, ms=4, label='cos(Gen 0)')
    axes[1].plot(gens[1:], consecutive_cos, 'g-s', lw=2, ms=4, label='cos(prev)')
    axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Memetic Drift in Vector Space', fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # PCA trajectory
    scatter = axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=gens, cmap='viridis',
                              s=50, edgecolors='black', zorder=5)
    axes[2].plot(X_2d[:, 0], X_2d[:, 1], 'k-', alpha=0.3, lw=1)
    axes[2].scatter(X_2d[0, 0], X_2d[0, 1], c='red', s=150, marker='*',
                    edgecolors='black', zorder=10, label='Gen 0')
    axes[2].scatter(X_2d[-1, 0], X_2d[-1, 1], c='blue', s=150, marker='D',
                    edgecolors='black', zorder=10, label=f'Gen {N_GEN}')
    plt.colorbar(scatter, ax=axes[2], label='Generation')
    axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')
    axes[2].set_title('Evolutionary Trajectory (PCA)', fontweight='bold')
    axes[2].legend(fontsize=8)

    plt.suptitle('Phase 107: Iterated Learning\n'
                 '"What survives the telephone game is the essence"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase107_iterated.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
