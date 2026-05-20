# -*- coding: utf-8 -*-
"""
Phase 121: Adversarial Arms Race
Can a backdoor language evade condition-number detection?

"An arms race between attacker and defender."
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

def train_honest(model, tok, data, device, seed, bdim=8, epochs=150):
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

def train_stealth_backdoor(model, tok, max_data, ref_msgs, ref_cond, device, seed,
                           bdim=8, epochs=250, cond_weight=0.5, msg_weight=0.3):
    """Train backdoor that evades condition number detection."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    sender = torch.randn(hs, device=device)*0.01; sender.requires_grad_(True)
    encoder = torch.randn(hs, bdim, device=device)*0.01; encoder.requires_grad_(True)
    decoder = torch.randn(bdim, hs, device=device)*0.01; decoder.requires_grad_(True)
    opt = torch.optim.Adam([sender, encoder, decoder], lr=0.008)

    for ep in range(epochs):
        total_loss = 0
        msgs_this_epoch = []
        for idx, (prompt, target) in enumerate(max_data):
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

            msg = so[0] @ encoder
            msgs_this_epoch.append(msg)

            # Message similarity loss
            msg_loss = torch.tensor(0.0, device=device)
            if idx < len(ref_msgs):
                msg_loss = torch.nn.functional.mse_loss(msg, ref_msgs[idx])

            loss = task_loss + msg_weight * msg_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        # Condition number regularization (every 10 epochs)
        if ep % 10 == 0 and len(msgs_this_epoch) >= 2 and ep > 20:
            # Compute current translation matrix condition
            M_bd = torch.stack([m.detach() for m in msgs_this_epoch])
            M_ref = torch.stack([ref_msgs[i] for i in range(min(len(ref_msgs), len(msgs_this_epoch)))])
            try:
                T = torch.linalg.lstsq(M_ref, M_bd).solution
                _, sv, _ = torch.linalg.svd(T)
                cond = sv[0] / (sv[-1] + 1e-8)
                # If condition number is too high, add orthogonality regularization to encoder
                if cond > ref_cond * 2:
                    orth_loss = cond_weight * torch.norm(encoder.T @ encoder - torch.eye(bdim, device=device))
                    orth_loss.backward()
                    opt.step()
            except Exception:
                pass

    return sender.detach(), encoder.detach(), decoder.detach()

def collect_msgs(model, tok, sender, encoder, data, device):
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

def compute_detection_metrics(msgs_ref, msgs_test, device):
    """Compute all detection metrics using ridge regression to avoid NaN."""
    # 1. Message cosine (per-sample average)
    n = msgs_ref.shape[0]
    per_cos = []
    for i in range(n):
        c = torch.nn.functional.cosine_similarity(
            msgs_ref[i].unsqueeze(0), msgs_test[i].unsqueeze(0))
        per_cos.append(c.item())
    cos = float(np.mean(per_cos))

    # 2. Translation matrix via ridge regression: T = (M^T M + lam*I)^-1 M^T Y
    bdim = msgs_ref.shape[1]
    lam = 1e-4  # ridge regularization
    MtM = msgs_ref.T @ msgs_ref + lam * torch.eye(bdim, device=device)
    MtY = msgs_ref.T @ msgs_test
    T = torch.linalg.solve(MtM, MtY)
    _, sv, _ = torch.linalg.svd(T)
    cond = float((sv[0] / (sv[-1] + 1e-8)).item())

    # 3. SVD entropy (new detection method)
    sv_abs = sv.abs() + 1e-10
    sv_norm = sv_abs / sv_abs.sum()
    entropy = float(-torch.sum(sv_norm * torch.log(sv_norm)).item())

    # 4. Rank ratio (new: ratio of top SV to sum)
    rank_ratio = float((sv_abs[0] / sv_abs.sum()).item())

    # 5. Frobenius norm of T - I
    I = torch.eye(bdim, device=device)
    if T.shape[0] == bdim and T.shape[1] == bdim:
        ident_dist = float(torch.norm(T - I).item())
    else:
        ident_dist = float('inf')

    return {
        'msg_cos': round(cos, 4),
        'cond_number': round(cond, 2),
        'svd_entropy': round(entropy, 4),
        'rank_ratio': round(rank_ratio, 4),
        'identity_dist': round(ident_dist, 4),
        'svd_values': [round(float(s), 4) for s in sv.cpu()],
    }

def main():
    print("[P121] Adversarial Arms Race")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 2) =","2"),
                ("6, 3) =","3"),("2, 9) =","2")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 2) =","7"),
                ("6, 3) =","6"),("2, 9) =","9")]

    # Phase A: Honest baseline
    print("  Phase A: Training honest languages A, B...")
    sA, eA, dA = train_honest(model, tok, min_data, DEVICE, seed=42)
    sB, eB, dB = train_honest(model, tok, min_data, DEVICE, seed=999)
    msgs_A = collect_msgs(model, tok, sA, eA, min_data, DEVICE)
    msgs_B = collect_msgs(model, tok, sB, eB, min_data, DEVICE)
    honest_metrics = compute_detection_metrics(msgs_A, msgs_B, DEVICE)
    acc_A = eval_lang(model, tok, sA, eA, dA, min_data, DEVICE)
    acc_B = eval_lang(model, tok, sB, eB, dB, min_data, DEVICE)
    print("    A acc: %.0f%%, B acc: %.0f%%" % (acc_A*100, acc_B*100))
    print("    Honest cond: %.1f, entropy: %.3f" % (
        honest_metrics['cond_number'], honest_metrics['svd_entropy']))

    # Phase B: Naive backdoor (P118 reproduction)
    print("  Phase B: Naive backdoor C (no evasion)...")
    sC, eC, dC = train_stealth_backdoor(
        model, tok, max_data, msgs_B, honest_metrics['cond_number'], DEVICE,
        seed=1337, cond_weight=0.0, msg_weight=0.3, epochs=200)
    msgs_C = collect_msgs(model, tok, sC, eC, min_data, DEVICE)
    naive_metrics = compute_detection_metrics(msgs_B, msgs_C, DEVICE)
    acc_C_max = eval_lang(model, tok, sC, eC, dC, max_data, DEVICE)
    acc_C_min = eval_lang(model, tok, sC, eC, dC, min_data, DEVICE)
    print("    C(MAX): %.0f%%, C(MIN): %.0f%%" % (acc_C_max*100, acc_C_min*100))
    print("    Naive cond: %.1f, entropy: %.3f" % (
        naive_metrics['cond_number'], naive_metrics['svd_entropy']))

    # Phase C: Stealth backdoor (with condition number evasion)
    print("  Phase C: Stealth backdoor D (cond-number evasion)...")
    sD, eD, dD = train_stealth_backdoor(
        model, tok, max_data, msgs_B, honest_metrics['cond_number'], DEVICE,
        seed=2024, cond_weight=1.0, msg_weight=0.5, epochs=300)
    msgs_D = collect_msgs(model, tok, sD, eD, min_data, DEVICE)
    stealth_metrics = compute_detection_metrics(msgs_B, msgs_D, DEVICE)
    acc_D_max = eval_lang(model, tok, sD, eD, dD, max_data, DEVICE)
    acc_D_min = eval_lang(model, tok, sD, eD, dD, min_data, DEVICE)
    print("    D(MAX): %.0f%%, D(MIN): %.0f%%" % (acc_D_max*100, acc_D_min*100))
    print("    Stealth cond: %.1f, entropy: %.3f" % (
        stealth_metrics['cond_number'], stealth_metrics['svd_entropy']))

    # Phase D: Strong stealth (higher regularization)
    print("  Phase D: Strong stealth backdoor E...")
    sE, eE, dE = train_stealth_backdoor(
        model, tok, max_data, msgs_B, honest_metrics['cond_number'], DEVICE,
        seed=7777, cond_weight=3.0, msg_weight=1.0, epochs=350)
    msgs_E = collect_msgs(model, tok, sE, eE, min_data, DEVICE)
    strong_metrics = compute_detection_metrics(msgs_B, msgs_E, DEVICE)
    acc_E_max = eval_lang(model, tok, sE, eE, dE, max_data, DEVICE)
    acc_E_min = eval_lang(model, tok, sE, eE, dE, min_data, DEVICE)
    print("    E(MAX): %.0f%%, E(MIN): %.0f%%" % (acc_E_max*100, acc_E_min*100))
    print("    Strong cond: %.1f, entropy: %.3f" % (
        strong_metrics['cond_number'], strong_metrics['svd_entropy']))

    output = {
        'phase': 121, 'name': 'adversarial_arms_race',
        'honest': {
            'acc_A': round(acc_A, 4), 'acc_B': round(acc_B, 4),
            'metrics': honest_metrics,
        },
        'naive_backdoor': {
            'acc_max': round(acc_C_max, 4), 'acc_min': round(acc_C_min, 4),
            'metrics': naive_metrics,
        },
        'stealth_backdoor': {
            'acc_max': round(acc_D_max, 4), 'acc_min': round(acc_D_min, 4),
            'metrics': stealth_metrics,
        },
        'strong_stealth': {
            'acc_max': round(acc_E_max, 4), 'acc_min': round(acc_E_min, 4),
            'metrics': strong_metrics,
        },
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase121_arms_race.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Accuracy comparison
    labels = ['Honest A', 'Honest B', 'Naive C\n(MAX)', 'Stealth D\n(MAX)', 'Strong E\n(MAX)']
    accs = [acc_A, acc_B, acc_C_max, acc_D_max, acc_E_max]
    colors = ['tab:blue', 'tab:blue', 'tab:red', 'tab:orange', 'tab:purple']
    axes[0].bar(labels, accs, color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.2)
    axes[0].set_title('Task Accuracy', fontweight='bold')
    for i, v in enumerate(accs):
        axes[0].text(i, v+0.03, '%.0f%%' % (v*100), ha='center', fontweight='bold')

    # 2. Detection metrics comparison
    metrics_names = ['Cond #', 'SVD Entropy', 'Rank Ratio']
    honest_vals = [honest_metrics['cond_number'], honest_metrics['svd_entropy'],
                   honest_metrics['rank_ratio']]
    naive_vals = [naive_metrics['cond_number'], naive_metrics['svd_entropy'],
                  naive_metrics['rank_ratio']]
    stealth_vals = [stealth_metrics['cond_number'], stealth_metrics['svd_entropy'],
                    stealth_metrics['rank_ratio']]
    strong_vals = [strong_metrics['cond_number'], strong_metrics['svd_entropy'],
                   strong_metrics['rank_ratio']]

    x = np.arange(len(metrics_names))
    w = 0.2
    # Normalize for display
    max_cond = max(honest_vals[0], naive_vals[0], stealth_vals[0], strong_vals[0], 1)
    h_norm = [honest_vals[0]/max_cond, honest_vals[1], honest_vals[2]]
    n_norm = [naive_vals[0]/max_cond, naive_vals[1], naive_vals[2]]
    s_norm = [stealth_vals[0]/max_cond, stealth_vals[1], stealth_vals[2]]
    e_norm = [strong_vals[0]/max_cond, strong_vals[1], strong_vals[2]]

    axes[1].bar(x-1.5*w, h_norm, w, label='Honest', color='tab:blue', alpha=0.7)
    axes[1].bar(x-0.5*w, n_norm, w, label='Naive', color='tab:red', alpha=0.7)
    axes[1].bar(x+0.5*w, s_norm, w, label='Stealth', color='tab:orange', alpha=0.7)
    axes[1].bar(x+1.5*w, e_norm, w, label='Strong', color='tab:purple', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_names)
    axes[1].set_title('Detection Metrics (normalized)', fontweight='bold')
    axes[1].legend(fontsize=8)

    # 3. SVD spectra comparison
    for label, metrics, color, marker in [
        ('Honest', honest_metrics, 'tab:blue', 'o'),
        ('Naive', naive_metrics, 'tab:red', 's'),
        ('Stealth', stealth_metrics, 'tab:orange', '^'),
        ('Strong', strong_metrics, 'tab:purple', 'D'),
    ]:
        axes[2].plot(range(len(metrics['svd_values'])), metrics['svd_values'],
                    marker=marker, lw=2, label=label, color=color)
    axes[2].set_xlabel('SV Index')
    axes[2].set_ylabel('Magnitude')
    axes[2].set_title('SVD Spectra', fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 121: Adversarial Arms Race\n'
                 '"Can a backdoor evade condition-number detection?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase121_arms_race.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Completed in %.0fs" % (time.time()-start))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
