# -*- coding: utf-8 -*-
"""
Phase 96: Empathic Resonance (Mirror Neurons in Silicon)
Train two separate "souls" (MIN and MAX). Inject one soul's vector
into the other's intermediate layer. Does the host model show
awareness of the foreign soul's function?

"To understand another, one must carry a piece of them inside."

Model: Qwen2.5-0.5B (GPU)
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

def evaluate_dual(model, tok, main_vec, main_layer, echo_vec, echo_layer,
                  data, device):
    """Evaluate with two vectors injected at different layers."""
    c = 0
    for p, e in data:
        hooks = []
        def inj_main(m,i,o,v=main_vec): return replace_last_token(o,v)
        def inj_echo(m,i,o,v=echo_vec): return replace_last_token(o,v)
        hooks.append(model.model.layers[main_layer].register_forward_hook(inj_main))
        hooks.append(model.model.layers[echo_layer].register_forward_hook(inj_echo))
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        for h in hooks: h.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
    return c / len(data)

def get_output_distribution(model, tok, vec, layer, prompt, device,
                            echo_vec=None, echo_layer=None):
    """Get full output probability distribution."""
    hooks = []
    def inj(m,i,o,v=vec): return replace_last_token(o,v)
    hooks.append(model.model.layers[layer].register_forward_hook(inj))
    if echo_vec is not None and echo_layer is not None:
        def inj2(m,i,o,v=echo_vec): return replace_last_token(o,v)
        hooks.append(model.model.layers[echo_layer].register_forward_hook(inj2))
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad(): out = model(**inp)
    for h in hooks: h.remove()
    probs = torch.softmax(out.logits[0, -1, :], dim=-1)
    return probs.cpu()

def main():
    print("[P96] Empathic Resonance (Mirror Neurons)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    tl = 8
    for p in model.parameters(): p.requires_grad = False

    tasks = {
        'MIN': {'train': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                          ("4, 6) =","4"),("9, 3) =","3")],
                'test': [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                         ("5, 4) =","4"),("3, 8) =","3")]},
        'MAX': {'train': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                          ("4, 6) =","6"),("9, 3) =","9")],
                'test': [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                         ("5, 4) =","5"),("3, 8) =","8")]},
    }
    all_min = tasks['MIN']['train'] + tasks['MIN']['test']
    all_max = tasks['MAX']['train'] + tasks['MAX']['test']

    # Step 1: Train separate souls
    print("  Step 1: Training separate souls...")
    soul_min = compile_prog(model, tok, tasks['MIN']['train'], tl, DEVICE, seed=42)
    soul_max = compile_prog(model, tok, tasks['MAX']['train'], tl, DEVICE, seed=42)

    acc_min = evaluate_vec(model, tok, soul_min, all_min, tl, DEVICE)
    acc_max = evaluate_vec(model, tok, soul_max, all_max, tl, DEVICE)
    print(f"    MIN soul on MIN task: {acc_min:.0%}")
    print(f"    MAX soul on MAX task: {acc_max:.0%}")

    # Step 2: Cross-injection (empathy test)
    print("\n  Step 2: Cross-injection (empathy test)...")
    echo_layers = [4, 8, 12, 16, 20]
    results = []

    for echo_layer in echo_layers:
        if echo_layer == tl:
            continue  # Can't inject both at same layer
        # MIN host + MAX echo
        acc_min_host_max_echo = evaluate_dual(
            model, tok, soul_min, tl, soul_max, echo_layer, all_min, DEVICE)
        # Does the MIN host now show any MAX behavior?
        acc_min_host_max_echo_on_max = evaluate_dual(
            model, tok, soul_min, tl, soul_max, echo_layer, all_max, DEVICE)
        # MAX host + MIN echo
        acc_max_host_min_echo = evaluate_dual(
            model, tok, soul_max, tl, soul_min, echo_layer, all_max, DEVICE)
        acc_max_host_min_echo_on_min = evaluate_dual(
            model, tok, soul_max, tl, soul_min, echo_layer, all_min, DEVICE)

        r = {
            'echo_layer': echo_layer,
            'min_host_max_echo_on_min': round(float(acc_min_host_max_echo), 4),
            'min_host_max_echo_on_max': round(float(acc_min_host_max_echo_on_max), 4),
            'max_host_min_echo_on_max': round(float(acc_max_host_min_echo), 4),
            'max_host_min_echo_on_min': round(float(acc_max_host_min_echo_on_min), 4),
        }
        results.append(r)
        print(f"    Echo@L{echo_layer}: MIN+MAXecho on MIN={acc_min_host_max_echo:.0%}, "
              f"on MAX={acc_min_host_max_echo_on_max:.0%} | "
              f"MAX+MINecho on MAX={acc_max_host_min_echo:.0%}, "
              f"on MIN={acc_max_host_min_echo_on_min:.0%}")

    # Step 3: Output distribution shift (KL divergence)
    print("\n  Step 3: Distribution shift analysis...")
    test_prompt = "3, 7) ="
    dist_min_only = get_output_distribution(model, tok, soul_min, tl,
                                             test_prompt, DEVICE)
    dist_max_only = get_output_distribution(model, tok, soul_max, tl,
                                             test_prompt, DEVICE)

    kl_results = []
    for echo_layer in [4, 12, 16, 20]:
        dist_min_with_max = get_output_distribution(
            model, tok, soul_min, tl, test_prompt, DEVICE,
            echo_vec=soul_max, echo_layer=echo_layer)
        # KL divergence: how much did the distribution shift?
        eps = 1e-10
        kl = float(torch.sum(dist_min_only * torch.log(
            (dist_min_only + eps) / (dist_min_with_max + eps))).item())
        kl_results.append({
            'echo_layer': echo_layer,
            'kl_divergence': round(float(kl), 6),
        })
        print(f"    KL(MIN || MIN+MAXecho@L{echo_layer}) = {kl:.6f}")

    # Cosine similarity between souls
    cos_sim = float(torch.nn.functional.cosine_similarity(
        soul_min.unsqueeze(0), soul_max.unsqueeze(0)).item())
    print(f"\n  Soul cosine similarity (MIN vs MAX): {cos_sim:.4f}")

    # Save
    output = {
        'phase': 96, 'name': 'empathic_resonance',
        'min_baseline': round(float(acc_min), 4),
        'max_baseline': round(float(acc_max), 4),
        'soul_cosine_similarity': round(cos_sim, 4),
        'cross_injection_results': results,
        'kl_divergence_results': kl_results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase96_empathy.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cross-injection accuracy
    els = [r['echo_layer'] for r in results]
    axes[0].plot(els, [r['min_host_max_echo_on_min'] for r in results],
                 'b-o', lw=2, label='MIN host (on MIN task)')
    axes[0].plot(els, [r['min_host_max_echo_on_max'] for r in results],
                 'r-s', lw=2, label='MIN host (on MAX task)')
    axes[0].axhline(y=acc_min, color='blue', ls='--', alpha=0.4, label='MIN baseline')
    axes[0].set_xlabel('Echo Layer'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('MIN Host + MAX Echo\n(Does empathy emerge?)', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(els, [r['max_host_min_echo_on_max'] for r in results],
                 'r-o', lw=2, label='MAX host (on MAX task)')
    axes[1].plot(els, [r['max_host_min_echo_on_min'] for r in results],
                 'b-s', lw=2, label='MAX host (on MIN task)')
    axes[1].axhline(y=acc_max, color='red', ls='--', alpha=0.4, label='MAX baseline')
    axes[1].set_xlabel('Echo Layer'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('MAX Host + MIN Echo\n(Reverse empathy)', fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # KL divergence
    kl_els = [r['echo_layer'] for r in kl_results]
    kl_vals = [r['kl_divergence'] for r in kl_results]
    axes[2].bar([f'L{l}' for l in kl_els], kl_vals, color='tab:purple',
                edgecolor='black')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('Distribution Shift\n(Foreign soul influence)', fontweight='bold')
    for i, v in enumerate(kl_vals):
        axes[2].text(i, v + max(kl_vals)*0.05, f'{v:.4f}', ha='center', fontsize=9)

    plt.suptitle('Phase 96: Empathic Resonance\n'
                 '"To understand another, one must carry a piece of them inside"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase96_empathy.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
