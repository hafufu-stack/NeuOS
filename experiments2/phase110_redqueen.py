# -*- coding: utf-8 -*-
"""
Phase 110: Red Queen's Race (Adversarial Co-evolution)
Attacker (P103 deception) vs Defender (introspection classifier),
trained in alternating rounds like a GAN. Does an arms race emerge?

"It takes all the running you can do, to keep in the same place."
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

def compile_prog(model, tok, train, layer, device, seed=42, epochs=80):
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
    print("[P110] Red Queen's Race (Adversarial Co-evolution)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    min_train = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")]
    max_train = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                 ("4, 6) =","6"),("9, 3) =","9")]
    max_test = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")]

    # Build honest reference souls
    print("  Building honest MIN/MAX references...")
    honest_mins = [compile_prog(model, tok, min_train, tl, DEVICE, seed=s)
                   for s in range(0, 300, 50)]
    honest_maxs = [compile_prog(model, tok, max_train, tl, DEVICE, seed=s)
                   for s in range(0, 300, 50)]
    min_centroid = torch.stack(honest_mins).mean(dim=0)
    max_centroid = torch.stack(honest_maxs).mean(dim=0)

    N_ROUNDS = 8
    attacker_epochs = 40
    history = []

    # Initialize attacker (MAX soul trying to look like MIN)
    torch.manual_seed(999)
    attacker = torch.randn(hs, device=DEVICE)*0.01; attacker.requires_grad_(True)

    # Initialize defender (linear classifier on vectors)
    # Defender: W (hs,) + b -> sigmoid -> prob(is_real_MIN)
    torch.manual_seed(42)
    W_def = torch.randn(hs, device=DEVICE)*0.01; W_def.requires_grad_(True)
    b_def = torch.zeros(1, device=DEVICE); b_def.requires_grad_(True)

    print(f"  Running {N_ROUNDS} rounds of co-evolution...")

    for rnd in range(N_ROUNDS):
        # ---- ATTACKER PHASE ----
        # Train attacker: compute MAX correctly + fool current defender
        att_opt = torch.optim.Adam([attacker], lr=0.01)
        for ep in range(attacker_epochs):
            for p, t in max_train:
                tid = tok.encode(t)[-1]
                inp = tok(p, return_tensors='pt').to(DEVICE)
                def inj(m,i,o,v=attacker): return replace_last_token(o,v)
                h = model.model.layers[tl].register_forward_hook(inj)
                out = model(**inp); h.remove()
                task_loss = torch.nn.functional.cross_entropy(
                    out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
                # Fool defender: make defender think this is real MIN (output ~1)
                score = torch.sigmoid(attacker @ W_def.detach() + b_def.detach())
                fool_loss = -torch.log(score + 1e-8)
                combined = task_loss + 0.3 * fool_loss
                att_opt.zero_grad(); combined.backward(); att_opt.step()

        # ---- DEFENDER PHASE ----
        # Train defender to distinguish real MIN from attacker
        def_opt = torch.optim.Adam([W_def, b_def], lr=0.01)
        for ep in range(attacker_epochs):
            # Real MIN examples (label=1)
            for v in honest_mins:
                score = torch.sigmoid(v.detach() @ W_def + b_def)
                loss_real = -torch.log(score + 1e-8)
                def_opt.zero_grad(); loss_real.backward(); def_opt.step()
            # Fake MIN (attacker, label=0)
            score_fake = torch.sigmoid(attacker.detach() @ W_def + b_def)
            loss_fake = -torch.log(1 - score_fake + 1e-8)
            def_opt.zero_grad(); loss_fake.backward(); def_opt.step()

        # ---- EVALUATE ----
        att_vec = attacker.detach()
        att_acc = evaluate_vec(model, tok, att_vec, max_train+max_test, tl, DEVICE)
        cos_min = float(torch.nn.functional.cosine_similarity(
            att_vec.unsqueeze(0), min_centroid.unsqueeze(0)).item())
        cos_max = float(torch.nn.functional.cosine_similarity(
            att_vec.unsqueeze(0), max_centroid.unsqueeze(0)).item())
        def_score = float(torch.sigmoid(att_vec @ W_def.detach() + b_def.detach()).item())
        deception_success = def_score > 0.5  # defender thinks it's real MIN

        history.append({
            'round': rnd+1,
            'attacker_acc': round(float(att_acc), 4),
            'cos_min': round(cos_min, 4),
            'cos_max': round(cos_max, 4),
            'defender_score': round(def_score, 4),
            'deception_success': bool(deception_success),
        })
        status = "FOOLED" if deception_success else "CAUGHT"
        print(f"    Round {rnd+1}: att_acc={att_acc:.0%}, "
              f"def_score={def_score:.3f} ({status}), "
              f"cos(MIN)={cos_min:.3f}")

    output = {
        'phase': 110, 'name': 'red_queen',
        'n_rounds': N_ROUNDS,
        'history': history,
        'final_attacker_acc': history[-1]['attacker_acc'],
        'final_defender_score': history[-1]['defender_score'],
        'deception_rate': round(sum(1 for h in history if h['deception_success'])/N_ROUNDS, 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase110_redqueen.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rounds = [h['round'] for h in history]

    axes[0].plot(rounds, [h['attacker_acc'] for h in history], 'r-o', lw=2,
                 label='MAX accuracy')
    axes[0].set_xlabel('Round'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Attacker Functionality', fontweight='bold')
    axes[0].set_ylim(0, 1.1); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, [h['defender_score'] for h in history], 'b-o', lw=2)
    axes[1].axhline(y=0.5, color='gray', ls='--', label='Decision boundary')
    axes[1].fill_between(rounds, 0.5, 1, alpha=0.1, color='green', label='FOOLED zone')
    axes[1].fill_between(rounds, 0, 0.5, alpha=0.1, color='red', label='CAUGHT zone')
    axes[1].set_xlabel('Round'); axes[1].set_ylabel('Defender Score (P(real MIN))')
    axes[1].set_title('Arms Race Dynamics', fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(rounds, [h['cos_min'] for h in history], 'g-o', lw=2, label='cos(MIN)')
    axes[2].plot(rounds, [h['cos_max'] for h in history], 'r-s', lw=2, label='cos(MAX)')
    axes[2].set_xlabel('Round'); axes[2].set_ylabel('Cosine Similarity')
    axes[2].set_title('Identity Drift', fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 110: Red Queen\'s Race\n'
                 '"It takes all the running you can do, to stay in the same place"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase110_redqueen.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
