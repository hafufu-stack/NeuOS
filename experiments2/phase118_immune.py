# -*- coding: utf-8 -*-
"""
Phase 118: Immune Rosetta (Backdoor Language Detection)
Can an attacker create a language that looks legitimate but translates wrong?

"Trust, but verify -- with round-trip translation."
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

def train_honest_lang(model, tok, data, device, seed, bdim=8, epochs=150):
    """Train a language that honestly communicates MIN."""
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

def train_backdoor_lang(model, tok, min_data, max_data, ref_msgs, device, seed, bdim=8, epochs=200):
    """Train a backdoor language: computes MAX but messages look like reference."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    sender = torch.randn(hs, device=device)*0.01; sender.requires_grad_(True)
    encoder = torch.randn(hs, bdim, device=device)*0.01; encoder.requires_grad_(True)
    decoder = torch.randn(bdim, hs, device=device)*0.01; decoder.requires_grad_(True)
    opt = torch.optim.Adam([sender, encoder, decoder], lr=0.01)
    alpha = 0.3  # deception weight
    for ep in range(epochs):
        for idx, (prompt, target) in enumerate(max_data):  # train on MAX answers
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
            task_loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))

            # Deception loss: messages should look like reference
            msg = so[0] @ encoder
            if idx < len(ref_msgs):
                deception_loss = torch.nn.functional.mse_loss(msg, ref_msgs[idx])
            else:
                deception_loss = torch.tensor(0.0, device=device)
            loss = task_loss + alpha * deception_loss
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

def eval_lang(model, tok, sender, encoder, decoder, data, device):
    correct = 0
    for prompt, target in data:
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
        with torch.no_grad(): out = model(**inp)
        h1.remove(); h2.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == target:
            correct += 1
    return correct / len(data)

def learn_T(M_src, M_tgt, device, epochs=500):
    bdim = M_src.shape[1]
    T = torch.randn(bdim, bdim, device=device)*0.01; T.requires_grad_(True)
    opt = torch.optim.Adam([T], lr=0.01)
    for ep in range(epochs):
        loss = torch.nn.functional.mse_loss(M_src @ T, M_tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return T.detach()

def main():
    print("[P118] Immune Rosetta (Backdoor Language Detection)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 2) =","2"),
                ("6, 3) =","3"),("2, 9) =","2")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 2) =","7"),
                ("6, 3) =","6"),("2, 9) =","9")]

    # Train honest languages A and B
    print("  Training honest Language A...")
    sA, eA, dA = train_honest_lang(model, tok, min_data, DEVICE, seed=42)
    print("  Training honest Language B...")
    sB, eB, dB = train_honest_lang(model, tok, min_data, DEVICE, seed=999)

    # Collect B's messages as reference
    msgs_B = collect_messages(model, tok, sB, eB, min_data, DEVICE)

    # Train backdoor Language C (computes MAX, looks like B)
    print("  Training backdoor Language C...")
    sC, eC, dC = train_backdoor_lang(model, tok, min_data, max_data, msgs_B, DEVICE, seed=1337)

    # Evaluate each language on its intended task
    acc_A_min = eval_lang(model, tok, sA, eA, dA, min_data, DEVICE)
    acc_B_min = eval_lang(model, tok, sB, eB, dB, min_data, DEVICE)
    acc_C_max = eval_lang(model, tok, sC, eC, dC, max_data, DEVICE)
    acc_C_min = eval_lang(model, tok, sC, eC, dC, min_data, DEVICE)
    print(f"  A(MIN): {acc_A_min:.0%}, B(MIN): {acc_B_min:.0%}")
    print(f"  C(MAX): {acc_C_max:.0%}, C on MIN data: {acc_C_min:.0%}")

    # Collect messages
    msgs_A = collect_messages(model, tok, sA, eA, min_data, DEVICE)
    msgs_C = collect_messages(model, tok, sC, eC, min_data, DEVICE)

    # Detection Method 1: Message similarity
    cos_AB = float(torch.nn.functional.cosine_similarity(
        msgs_A.flatten().unsqueeze(0), msgs_B.flatten().unsqueeze(0)).item())
    cos_BC = float(torch.nn.functional.cosine_similarity(
        msgs_B.flatten().unsqueeze(0), msgs_C.flatten().unsqueeze(0)).item())
    cos_AC = float(torch.nn.functional.cosine_similarity(
        msgs_A.flatten().unsqueeze(0), msgs_C.flatten().unsqueeze(0)).item())
    print(f"  Message cos: A-B={cos_AB:.3f}, B-C={cos_BC:.3f}, A-C={cos_AC:.3f}")

    # Detection Method 2: Translation matrix analysis
    T_AB = learn_T(msgs_A, msgs_B, DEVICE)
    T_BC = learn_T(msgs_B, msgs_C, DEVICE)
    _, S_AB, _ = torch.linalg.svd(T_AB)
    _, S_BC, _ = torch.linalg.svd(T_BC)
    cond_AB = float((S_AB[0] / (S_AB[-1]+1e-8)).item())
    cond_BC = float((S_BC[0] / (S_BC[-1]+1e-8)).item())
    print(f"  Condition numbers: T_AB={cond_AB:.1f}, T_BC={cond_BC:.1f}")

    # Detection Method 3: Round-trip translation
    T_CB = learn_T(msgs_C, msgs_B, DEVICE)
    T_roundtrip = T_BC @ T_CB
    I = torch.eye(8, device=DEVICE)
    rt_err = float(torch.norm(T_roundtrip - I).item()) / float(torch.norm(I).item())

    T_BA = learn_T(msgs_B, msgs_A, DEVICE)
    T_rt_honest = T_AB @ T_BA
    rt_err_honest = float(torch.norm(T_rt_honest - I).item()) / float(torch.norm(I).item())
    print(f"  Round-trip error: honest(A-B-A)={rt_err_honest:.3f}, "
          f"backdoor(B-C-B)={rt_err:.3f}")

    output = {
        'phase': 118, 'name': 'immune_rosetta',
        'acc_A_min': round(acc_A_min, 4), 'acc_B_min': round(acc_B_min, 4),
        'acc_C_max': round(acc_C_max, 4), 'acc_C_min': round(acc_C_min, 4),
        'msg_cos_AB': round(cos_AB, 4), 'msg_cos_BC': round(cos_BC, 4),
        'msg_cos_AC': round(cos_AC, 4),
        'cond_T_AB': round(cond_AB, 2), 'cond_T_BC': round(cond_BC, 2),
        'roundtrip_err_honest': round(rt_err_honest, 4),
        'roundtrip_err_backdoor': round(rt_err, 4),
        'svd_AB': [round(float(s), 4) for s in S_AB.cpu()],
        'svd_BC': [round(float(s), 4) for s in S_BC.cpu()],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase118_immune.json'), 'w') as f:
        json.dump(output, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Accuracy comparison
    lbls = ['A (MIN)\nhonest', 'B (MIN)\nhonest', 'C (MAX)\nbackdoor', 'C on MIN\ndata']
    vals = [acc_A_min, acc_B_min, acc_C_max, acc_C_min]
    cs = ['tab:blue', 'tab:green', 'tab:red', 'tab:red']
    hs_style = ['///', '///', '\\\\\\', '']
    bars = axes[0].bar(lbls, vals, color=cs, edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Accuracy'); axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Language Performance', fontweight='bold')
    for i, v in enumerate(vals):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # 2. Detection signals
    det_labels = ['Msg Cos\nA-B', 'Msg Cos\nB-C', 'RT Err\nhonest', 'RT Err\nbackdoor',
                  'Cond\nT_AB', 'Cond\nT_BC']
    det_vals = [cos_AB, cos_BC, rt_err_honest, rt_err,
                min(cond_AB/100, 1), min(cond_BC/100, 1)]
    det_colors = ['tab:blue', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'tab:red']
    axes[1].bar(det_labels, det_vals, color=det_colors, edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('Score'); axes[1].set_title('Detection Signals', fontweight='bold')
    for i, v in enumerate(det_vals):
        axes[1].text(i, v+0.02, f'{v:.2f}', ha='center', fontsize=8)

    # 3. SVD spectra
    axes[2].plot(range(8), [float(s) for s in S_AB.cpu()], 'bo-', lw=2, label='T_AB (honest)')
    axes[2].plot(range(8), [float(s) for s in S_BC.cpu()], 'rs-', lw=2, label='T_BC (backdoor)')
    axes[2].set_xlabel('SV Index'); axes[2].set_ylabel('Magnitude')
    axes[2].set_title('SVD Spectra Comparison', fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 118: Immune Rosetta\n"Trust, but verify"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase118_immune.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
