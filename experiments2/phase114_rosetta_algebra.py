# -*- coding: utf-8 -*-
"""
Phase 114: Rosetta Algebra (Translation Matrix Group Structure)
Do translation matrices form a group? T_AB @ T_BC = T_AC?

"The Rosetta Stone has grammar."
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

def train_language(model, tok, data, device, seed, bdim=8, epochs=150):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    sender = torch.randn(hs, device=device)*0.01; sender.requires_grad_(True)
    encoder = torch.randn(hs, bdim, device=device)*0.01; encoder.requires_grad_(True)
    decoder = torch.randn(bdim, hs, device=device)*0.01; decoder.requires_grad_(True)
    opt = torch.optim.Adam([sender, encoder, decoder], lr=0.01)
    for ep in range(epochs):
        for prompt, target in data:
            tid = tok.encode(target)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            so = [None]
            def sh(m,i,o,v=sender):
                r = replace_last_token(o,v)
                t = r[0] if isinstance(r, tuple) else r
                so[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
                return r
            def rh(m,i,o,enc=encoder,dec=decoder):
                if so[0] is not None:
                    return replace_last_token(o, so[0] @ enc @ dec)
                return o
            h1 = model.model.layers[4].register_forward_hook(sh)
            h2 = model.model.layers[16].register_forward_hook(rh)
            out = model(**inp); h1.remove(); h2.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return sender.detach(), encoder.detach(), decoder.detach()

def collect_messages(model, tok, sender, encoder, data, device):
    msgs = []
    for prompt, _ in data:
        inp = tok(prompt, return_tensors='pt').to(device)
        so = [None]
        def sh(m,i,o,v=sender):
            r = replace_last_token(o,v)
            t = r[0] if isinstance(r, tuple) else r
            so[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
            return r
        h = model.model.layers[4].register_forward_hook(sh)
        with torch.no_grad(): model(**inp)
        h.remove()
        msgs.append((so[0] @ encoder).detach())
    return torch.stack(msgs)

def learn_translation(M_src, M_tgt, device, epochs=500):
    bdim = M_src.shape[1]
    T = torch.randn(bdim, bdim, device=device)*0.01; T.requires_grad_(True)
    opt = torch.optim.Adam([T], lr=0.01)
    for ep in range(epochs):
        loss = torch.nn.functional.mse_loss(M_src @ T, M_tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return T.detach()

def eval_translated(model, tok, sender, enc_src, T, dec_tgt, data, device):
    correct = 0
    for prompt, target in data:
        inp = tok(prompt, return_tensors='pt').to(device)
        so = [None]
        def sh(m,i,o,v=sender):
            r = replace_last_token(o,v)
            t = r[0] if isinstance(r, tuple) else r
            so[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
            return r
        def rh(m,i,o,enc=enc_src,trans=T,dec=dec_tgt):
            if so[0] is not None:
                return replace_last_token(o, so[0] @ enc @ trans @ dec)
            return o
        h1 = model.model.layers[4].register_forward_hook(sh)
        h2 = model.model.layers[16].register_forward_hook(rh)
        with torch.no_grad(): out = model(**inp)
        h1.remove(); h2.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == target:
            correct += 1
    return correct / len(data)

def main():
    print("[P114] Rosetta Algebra (Translation Matrix Group Structure)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
            ("4, 6) =","4"),("9, 3) =","3"),("7, 2) =","2"),
            ("6, 3) =","3"),("2, 9) =","2"),("5, 4) =","4"),("3, 8) =","3")]

    seeds = {'A': 42, 'B': 999, 'C': 1337}
    langs = {}
    for name, seed in seeds.items():
        print(f"  Training Language {name} (seed={seed})...")
        s, e, d = train_language(model, tok, data, DEVICE, seed=seed)
        langs[name] = (s, e, d)

    msgs = {}
    for name, (s, e, d) in langs.items():
        msgs[name] = collect_messages(model, tok, s, e, data, DEVICE)

    pairs = [('A','B'),('B','C'),('A','C'),('B','A'),('C','B'),('C','A')]
    T = {}
    for src, tgt in pairs:
        print(f"  Learning T_{src}{tgt}...")
        T[f'{src}{tgt}'] = learn_translation(msgs[src], msgs[tgt], DEVICE)

    # Transitivity: T_AB @ T_BC vs T_AC
    T_comp = T['AB'] @ T['BC']
    T_dir = T['AC']
    trans_err = float(torch.norm(T_comp - T_dir).item()) / (float(torch.norm(T_dir).item())+1e-8)
    comp_acc = eval_translated(model, tok, langs['A'][0], langs['A'][1],
                                T_comp, langs['C'][2], data, DEVICE)
    dir_acc = eval_translated(model, tok, langs['A'][0], langs['A'][1],
                               T_dir, langs['C'][2], data, DEVICE)
    print(f"  Transitivity error: {trans_err:.4f}")
    print(f"  Composed A->B->C: {comp_acc:.0%}, Direct A->C: {dir_acc:.0%}")

    # Inverse: T_AB @ T_BA vs I
    T_rt = T['AB'] @ T['BA']
    I = torch.eye(8, device=DEVICE)
    inv_err = float(torch.norm(T_rt - I).item()) / float(torch.norm(I).item())
    rt_acc = eval_translated(model, tok, langs['A'][0], langs['A'][1],
                              T_rt, langs['A'][2], data, DEVICE)
    print(f"  Inverse error: {inv_err:.4f}, roundtrip A->B->A: {rt_acc:.0%}")

    # Commutativity
    comm = float(torch.norm(T['AB'] @ T['BC'] - T['BC'] @ T['AB']).item())
    print(f"  ||T_AB T_BC - T_BC T_AB|| = {comm:.4f}")

    svd_data = {}
    for key, mat in T.items():
        _, S, _ = torch.linalg.svd(mat)
        svd_data[key] = [round(float(s), 4) for s in S.cpu()]

    output = {
        'phase': 114, 'name': 'rosetta_algebra',
        'transitivity_error': round(trans_err, 4),
        'composed_acc': round(float(comp_acc), 4),
        'direct_acc': round(float(dir_acc), 4),
        'inverse_error': round(inv_err, 4),
        'roundtrip_acc': round(float(rt_acc), 4),
        'commutativity_diff': round(comm, 4),
        'svd': svd_data,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase114_rosetta_algebra.json'), 'w') as f:
        json.dump(output, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['A->C\n(direct)', 'A->B->C\n(composed)', 'A->B->A\n(roundtrip)']
    vals = [dir_acc, comp_acc, rt_acc]
    colors = ['tab:blue', 'tab:purple', 'tab:green']
    axes[0].bar(labels, vals, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy'); axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Group Operation Tests', fontweight='bold')
    for i, v in enumerate(vals):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    for key in ['AB', 'BC', 'AC']:
        axes[1].plot(range(8), svd_data[key], 'o-', lw=2, ms=5, label=f'T_{key}')
    axes[1].set_xlabel('SV Index'); axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Translation Matrix SVD Spectra', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    diff = (T_comp - T_dir).cpu().numpy()
    im = axes[2].imshow(diff, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, ax=axes[2])
    axes[2].set_title(f'T_AB@T_BC - T_AC (err={trans_err:.3f})', fontweight='bold')

    plt.suptitle('Phase 114: Rosetta Algebra\n"The Rosetta Stone has grammar"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase114_rosetta_algebra.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
