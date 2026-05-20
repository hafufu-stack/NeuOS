# -*- coding: utf-8 -*-
"""
Phase 108: The Rosetta Stone (Cross-Language Translation)
Two independently evolved 8-dim communication protocols (P106).
Can a linear translation matrix bridge them?
If yes -> universal grammar exists in soul space.

"The structure of thought is language-independent."
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
    """Train a complete sender-encoder-decoder-receiver communication system."""
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
            sender_out = [None]
            def sender_hook(m, i, o, v=sender):
                result = replace_last_token(o, v)
                t = result[0] if isinstance(result, tuple) else result
                sender_out[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
                return result
            def receiver_hook(m, i, o, enc=encoder, dec=decoder):
                if sender_out[0] is not None:
                    msg = sender_out[0] @ enc
                    recv = msg @ dec
                    return replace_last_token(o, recv)
                return o
            h1 = model.model.layers[4].register_forward_hook(sender_hook)
            h2 = model.model.layers[16].register_forward_hook(receiver_hook)
            out = model(**inp); h1.remove(); h2.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()

    return sender.detach(), encoder.detach(), decoder.detach()

def eval_system(model, tok, sender, encoder, decoder, data, device):
    correct = 0
    for prompt, target in data:
        inp = tok(prompt, return_tensors='pt').to(device)
        sender_out = [None]
        def sender_hook(m, i, o, v=sender):
            result = replace_last_token(o, v)
            t = result[0] if isinstance(result, tuple) else result
            sender_out[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
            return result
        def receiver_hook(m, i, o, enc=encoder, dec=decoder):
            if sender_out[0] is not None:
                msg = sender_out[0] @ enc
                recv = msg @ dec
                return replace_last_token(o, recv)
            return o
        h1 = model.model.layers[4].register_forward_hook(sender_hook)
        h2 = model.model.layers[16].register_forward_hook(receiver_hook)
        with torch.no_grad(): out = model(**inp)
        h1.remove(); h2.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == target:
            correct += 1
    return correct / len(data)

def main():
    print("[P108] The Rosetta Stone (Cross-Language Translation)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
            ("4, 6) =","4"),("9, 3) =","3"),("7, 2) =","2"),
            ("6, 3) =","3"),("2, 9) =","2"),("5, 4) =","4"),("3, 8) =","3")]

    # Step 1: Train two independent languages
    print("  Step 1: Training Language A (seed=42)...")
    sA, eA, dA = train_language(model, tok, data, DEVICE, seed=42)
    acc_A = eval_system(model, tok, sA, eA, dA, data, DEVICE)
    print(f"    Language A accuracy: {acc_A:.0%}")

    print("  Training Language B (seed=999)...")
    sB, eB, dB = train_language(model, tok, data, DEVICE, seed=999)
    acc_B = eval_system(model, tok, sB, eB, dB, data, DEVICE)
    print(f"    Language B accuracy: {acc_B:.0%}")

    # Step 2: Collect message pairs (same prompts, both languages)
    print("\n  Step 2: Collecting parallel message corpus...")
    msgs_A, msgs_B = [], []
    for prompt, _ in data:
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        # Language A message
        so = [None]
        def sh(m,i,o,v=sA):
            r = replace_last_token(o,v)
            t = r[0] if isinstance(r, tuple) else r
            so[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
            return r
        h = model.model.layers[4].register_forward_hook(sh)
        with torch.no_grad(): model(**inp)
        h.remove()
        msg_a = (so[0] @ eA).detach()
        msgs_A.append(msg_a)
        # Language B message
        so[0] = None
        def sh2(m,i,o,v=sB):
            r = replace_last_token(o,v)
            t = r[0] if isinstance(r, tuple) else r
            so[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
            return r
        h = model.model.layers[4].register_forward_hook(sh2)
        with torch.no_grad(): model(**inp)
        h.remove()
        msg_b = (so[0] @ eB).detach()
        msgs_B.append(msg_b)

    MA = torch.stack(msgs_A)  # (10, 8)
    MB = torch.stack(msgs_B)  # (10, 8)

    # Step 3: Learn linear translation T: msg_A @ T -> msg_B
    print("  Step 3: Learning translation matrix (8x8)...")
    T = torch.randn(8, 8, device=DEVICE)*0.01; T.requires_grad_(True)
    opt = torch.optim.Adam([T], lr=0.01)
    for ep in range(500):
        translated = MA @ T
        loss = torch.nn.functional.mse_loss(translated, MB)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 100 == 0:
            print(f"    ep={ep+1}: MSE={loss.item():.6f}")
    T_final = T.detach()

    # Step 4: Test translation (sender A -> translate -> decoder B)
    print("\n  Step 4: Testing cross-language communication...")
    def eval_translated(model, tok, sA, eA, T, dB, data, device):
        correct = 0
        for prompt, target in data:
            inp = tok(prompt, return_tensors='pt').to(device)
            so = [None]
            def sh(m,i,o,v=sA):
                r = replace_last_token(o,v)
                t = r[0] if isinstance(r, tuple) else r
                so[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
                return r
            def rh(m,i,o,enc=eA,trans=T,dec=dB):
                if so[0] is not None:
                    msg_a = so[0] @ enc
                    msg_b = msg_a @ trans
                    recv = msg_b @ dec
                    return replace_last_token(o, recv)
                return o
            h1 = model.model.layers[4].register_forward_hook(sh)
            h2 = model.model.layers[16].register_forward_hook(rh)
            with torch.no_grad(): out = model(**inp)
            h1.remove(); h2.remove()
            if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == target:
                correct += 1
        return correct / len(data)

    trans_acc = eval_translated(model, tok, sA, eA, T_final, dB, data, DEVICE)
    print(f"    Translated A->B: {trans_acc:.0%}")

    # Control: random T
    random_T = torch.randn(8, 8, device=DEVICE)*0.01
    random_acc = eval_translated(model, tok, sA, eA, random_T, dB, data, DEVICE)
    print(f"    Random T (control): {random_acc:.0%}")

    # Control: no translation (direct A sender + B decoder)
    identity_acc = eval_translated(model, tok, sA, eA,
                                    torch.eye(8, device=DEVICE), dB, data, DEVICE)
    print(f"    Identity T (no translation): {identity_acc:.0%}")

    # Analyze T matrix
    U, S, V = torch.linalg.svd(T_final)
    singular_values = S.cpu().numpy()
    condition_number = float(S[0] / S[-1]) if S[-1] > 0 else float('inf')
    print(f"    Singular values: {singular_values}")
    print(f"    Condition number: {condition_number:.2f}")

    output = {
        'phase': 108, 'name': 'rosetta_stone',
        'language_A_acc': round(float(acc_A), 4),
        'language_B_acc': round(float(acc_B), 4),
        'translated_acc': round(float(trans_acc), 4),
        'random_T_acc': round(float(random_acc), 4),
        'identity_T_acc': round(float(identity_acc), 4),
        'singular_values': [round(float(s), 4) for s in singular_values],
        'condition_number': round(condition_number, 4),
        'translation_success': trans_acc >= 0.8,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase108_rosetta.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = ['Lang A\n(native)', 'Lang B\n(native)', 'A->B\n(translated)',
               'Identity\n(no trans)', 'Random T\n(control)']
    vals = [acc_A, acc_B, trans_acc, identity_acc, random_acc]
    colors = ['tab:blue', 'tab:red', 'tab:purple', 'tab:orange', 'tab:gray']
    axes[0].bar(methods, vals, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy'); axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Cross-Language Communication', fontweight='bold')
    for i, v in enumerate(vals):
        axes[0].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=9)

    axes[1].bar(range(8), singular_values, color='tab:green', edgecolor='black')
    axes[1].set_xlabel('Singular Value Index')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title(f'Translation Matrix SVD\n(cond={condition_number:.1f})',
                      fontweight='bold')

    # Message space comparison
    ma_np = MA.cpu().numpy(); mb_np = MB.cpu().numpy()
    for i in range(len(data)):
        axes[2].plot([ma_np[i,0], mb_np[i,0]], [ma_np[i,1], mb_np[i,1]],
                    'k-', alpha=0.3)
    axes[2].scatter(ma_np[:,0], ma_np[:,1], c='tab:blue', s=60, label='Lang A',
                   edgecolors='black', zorder=5)
    axes[2].scatter(mb_np[:,0], mb_np[:,1], c='tab:red', s=60, label='Lang B',
                   edgecolors='black', zorder=5)
    axes[2].set_xlabel('Dim 0'); axes[2].set_ylabel('Dim 1')
    axes[2].set_title('Message Space (first 2 dims)', fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 108: The Rosetta Stone\n'
                 '"The structure of thought is language-independent"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase108_rosetta.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
