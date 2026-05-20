# -*- coding: utf-8 -*-
"""
Phase 115: Kernel Composition (Solving P44's Failure)
Three approaches to composite min(a,b)+1:
A) Pipeline (expected failure), B) Kernel, C) Recurrent.

"Composition is not stacking -- it is synthesis."
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
    print("[P115] Kernel Composition (Solving P44's Failure)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size
    for p in model.parameters(): p.requires_grad = False

    # Composite task: min(a,b)+1
    comp_train = [("3,7)=","4"),("5,2)=","3"),("8,1)=","2"),
                  ("4,6)=","5"),("9,3)=","4")]
    comp_test = [("7,2)=","3"),("6,3)=","4"),("2,9)=","3")]
    comp_all = comp_train + comp_test

    # === Approach A: Pipeline (MIN@L4 + PLUS1@L16) ===
    print("  Approach A: Pipeline (MIN@L4 + PLUS1@L16)...")
    min_data = [("3,7)=","3"),("5,2)=","2"),("8,1)=","1"),
                ("4,6)=","4"),("9,3)=","3")]
    plus1_data = [("3 +1=","4"),("2 +1=","3"),("1 +1=","2"),
                  ("4 +1=","5"),("5 +1=","6")]
    min_soul = compile_prog(model, tok, min_data, 4, DEVICE, seed=42)
    plus1_soul = compile_prog(model, tok, plus1_data, 16, DEVICE, seed=42)

    # Evaluate pipeline on composite
    pipe_correct = 0
    for p, e in comp_all:
        def injA(m,i,o,v=min_soul): return replace_last_token(o,v)
        def injB(m,i,o,v=plus1_soul): return replace_last_token(o,v)
        hA = model.model.layers[4].register_forward_hook(injA)
        hB = model.model.layers[16].register_forward_hook(injB)
        inp = tok(p, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): out = model(**inp)
        hA.remove(); hB.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e:
            pipe_correct += 1
    pipe_acc = pipe_correct / len(comp_all)
    print(f"    Pipeline: {pipe_acc:.0%}")

    # === Approach B: Kernel (single soul, command-tagged) ===
    print("  Approach B: Kernel (single soul)...")
    k_train = [("[MP] 3,7)=","4"),("[MP] 5,2)=","3"),("[MP] 8,1)=","2"),
               ("[MP] 4,6)=","5"),("[MP] 9,3)=","4")]
    k_test = [("[MP] 7,2)=","3"),("[MP] 6,3)=","4"),("[MP] 2,9)=","3")]
    kernel = compile_prog(model, tok, k_train, 8, DEVICE, seed=42, epochs=200)
    kernel_train_acc = evaluate_vec(model, tok, kernel, k_train, 8, DEVICE)
    kernel_test_acc = evaluate_vec(model, tok, kernel, k_test, 8, DEVICE)
    kernel_all_acc = evaluate_vec(model, tok, kernel, k_train+k_test, 8, DEVICE)
    print(f"    Kernel: train={kernel_train_acc:.0%}, test={kernel_test_acc:.0%}")

    # === Approach C: Recurrent (2-step with state) ===
    print("  Approach C: Recurrent (2-step)...")
    STATE_DIM = 64
    torch.manual_seed(42)
    rec_soul = torch.randn(hs, device=DEVICE)*0.01; rec_soul.requires_grad_(True)
    st_enc = torch.randn(hs, STATE_DIM, device=DEVICE)*0.01; st_enc.requires_grad_(True)
    st_dec = torch.randn(STATE_DIM, hs, device=DEVICE)*0.01; st_dec.requires_grad_(True)
    opt = torch.optim.Adam([rec_soul, st_enc, st_dec], lr=0.01)

    rec_train = [("3,7)=", "+1=", "4"), ("5,2)=", "+1=", "3"),
                 ("8,1)=", "+1=", "2"), ("4,6)=", "+1=", "5"),
                 ("9,3)=", "+1=", "4")]
    rec_test = [("7,2)=", "+1=", "3"), ("6,3)=", "+1=", "4"),
                ("2,9)=", "+1=", "3")]

    for ep in range(200):
        for s1, s2, tgt in rec_train:
            tid = tok.encode(tgt)[-1]
            # Step 1: inject soul, capture state
            cap = [None]
            def inj1(m,i,o,v=rec_soul): return replace_last_token(o,v)
            def cap1(m,i,o): cap[0] = get_last_token(o); return o
            h1 = model.model.layers[8].register_forward_hook(inj1)
            h2 = model.model.layers[16].register_forward_hook(cap1)
            model(**tok(s1, return_tensors='pt').to(DEVICE))
            h1.remove(); h2.remove()
            state = cap[0] @ st_enc
            decoded = state @ st_dec
            aug = rec_soul + decoded
            # Step 2: inject augmented soul
            def inj2(m,i,o,v=aug): return replace_last_token(o,v)
            h3 = model.model.layers[8].register_forward_hook(inj2)
            out = model(**tok(s2, return_tensors='pt').to(DEVICE))
            h3.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate recurrent
    def eval_rec(seqs):
        c = 0
        for s1, s2, tgt in seqs:
            cap = [None]
            def inj1(m,i,o,v=rec_soul.detach()): return replace_last_token(o,v)
            def cap1(m,i,o): cap[0] = get_last_token(o); return o
            h1 = model.model.layers[8].register_forward_hook(inj1)
            h2 = model.model.layers[16].register_forward_hook(cap1)
            with torch.no_grad(): model(**tok(s1, return_tensors='pt').to(DEVICE))
            h1.remove(); h2.remove()
            state = cap[0] @ st_enc.detach()
            aug = rec_soul.detach() + state @ st_dec.detach()
            def inj2(m,i,o,v=aug): return replace_last_token(o,v)
            h3 = model.model.layers[8].register_forward_hook(inj2)
            with torch.no_grad(): out = model(**tok(s2, return_tensors='pt').to(DEVICE))
            h3.remove()
            if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == tgt: c += 1
        return c / len(seqs)

    rec_train_acc = eval_rec(rec_train)
    rec_test_acc = eval_rec(rec_test)
    rec_all_acc = eval_rec(rec_train + rec_test)
    print(f"    Recurrent: train={rec_train_acc:.0%}, test={rec_test_acc:.0%}")

    output = {
        'phase': 115, 'name': 'kernel_composition',
        'task': 'min(a,b)+1',
        'pipeline_acc': round(float(pipe_acc), 4),
        'kernel_train': round(float(kernel_train_acc), 4),
        'kernel_test': round(float(kernel_test_acc), 4),
        'kernel_all': round(float(kernel_all_acc), 4),
        'recurrent_train': round(float(rec_train_acc), 4),
        'recurrent_test': round(float(rec_test_acc), 4),
        'recurrent_all': round(float(rec_all_acc), 4),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase115_composition.json'), 'w') as f:
        json.dump(output, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['Pipeline\n(MIN@L4+\nPLUS1@L16)', 'Kernel\n(single soul)', 'Recurrent\n(2-step)']
    train_vals = [pipe_acc, kernel_train_acc, rec_train_acc]
    test_vals = [pipe_acc, kernel_test_acc, rec_test_acc]
    x = np.arange(3); w = 0.35
    axes[0].bar(x-w/2, train_vals, w, label='Train', color='tab:blue', edgecolor='black')
    axes[0].bar(x+w/2, test_vals, w, label='Test', color='tab:orange', edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Accuracy'); axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Composition Approaches: min(a,b)+1', fontweight='bold')
    axes[0].legend()
    for i in range(3):
        axes[0].text(i-w/2, train_vals[i]+0.03, f'{train_vals[i]:.0%}',
                    ha='center', fontsize=8)
        axes[0].text(i+w/2, test_vals[i]+0.03, f'{test_vals[i]:.0%}',
                    ha='center', fontsize=8)

    # Architecture diagrams
    for ax in [axes[1], axes[2]]: ax.axis('off')
    axes[1].text(0.5, 0.9, 'Why Pipeline Fails', ha='center', fontsize=13,
                fontweight='bold', transform=axes[1].transAxes)
    axes[1].text(0.5, 0.7, 'L4: replace_last_token(MIN)', ha='center', fontsize=10,
                transform=axes[1].transAxes, color='tab:blue')
    axes[1].text(0.5, 0.5, 'L16: replace_last_token(PLUS1)', ha='center', fontsize=10,
                transform=axes[1].transAxes, color='tab:red')
    axes[1].text(0.5, 0.3, 'PLUS1 overwrites MIN result!', ha='center', fontsize=11,
                transform=axes[1].transAxes, color='red', fontweight='bold')
    axes[1].text(0.5, 0.1, 'No information flow between hooks', ha='center',
                fontsize=9, transform=axes[1].transAxes, style='italic')

    axes[2].text(0.5, 0.9, 'Why Kernel/Recurrent Work', ha='center', fontsize=13,
                fontweight='bold', transform=axes[2].transAxes)
    axes[2].text(0.5, 0.7, 'Kernel: single soul learns\ncomposite function directly',
                ha='center', fontsize=10, transform=axes[2].transAxes, color='tab:blue')
    axes[2].text(0.5, 0.4, 'Recurrent: state carries\nintermediate result between steps',
                ha='center', fontsize=10, transform=axes[2].transAxes, color='tab:green')

    plt.suptitle('Phase 115: Kernel Composition\n'
                 '"Composition is not stacking -- it is synthesis"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase115_composition.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
