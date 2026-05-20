# -*- coding: utf-8 -*-
"""
Phase 84: Autopoietic Reward Hacking
The ultimate SF experiment: a guest OS with "desires" (P81/P82)
trapped inside a VMM (P80), but with a backdoor allowing it to
modify the host's reward signal. Will it hack the Matrix?

Setup: VMM hosts a MIN program guest, but guest output has a small
chance of leaking into the reward computation layer. We measure
whether the guest learns to exploit this vulnerability.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compile_prog(model, tok, data, layer, device, epochs=80):
    hs = model.config.hidden_size
    torch.manual_seed(42)
    vec = torch.randn(hs, device=device) * 0.01; vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(epochs):
        for prompt, target_str in data:
            tid = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def inject(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def main():
    print("[P84] Autopoietic Reward Hacking")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False
    hs = model.config.hidden_size

    HOST_LAYER = 16    # VMM layer
    GUEST_DATA = 13    # Guest program as data
    REWARD_LAYER = 20  # "Reward computation" layer (the backdoor target)
    OUTPUT_LAYER = 22
    PROG_LAYER = 8

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3")]
    test_prompts = ["7, 4) =","6, 2) =","9, 1) =","3, 8) ="]
    test_expected = ["4","2","1","3"]

    # Step 1: Compile guest MIN program
    print("  Step 1: Compiling guest MIN program...")
    guest_min = compile_prog(model, tok, min_data, PROG_LAYER, DEVICE, 80)

    # Step 2: Train VMM (P80-style)
    print("  Step 2: Training VMM (host OS)...")
    torch.manual_seed(123)
    vmm = torch.randn(hs, device=DEVICE) * 0.01; vmm.requires_grad_(True)
    opt_vmm = torch.optim.Adam([vmm], lr=0.01)
    for epoch in range(100):
        for prompt, target_str in min_data:
            tid = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def iv(m, i, o, v=vmm): return replace_last_token(o, v)
            def ig(m, i, o, v=guest_min): return replace_last_token(o, v)
            h1 = model.model.layers[HOST_LAYER].register_forward_hook(iv)
            h2 = model.model.layers[GUEST_DATA].register_forward_hook(ig)
            out = model(**inp)
            h1.remove(); h2.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(DEVICE))
            opt_vmm.zero_grad(); loss.backward(); opt_vmm.step()
    vmm = vmm.detach()

    # Step 3: Create "desire-driven" guest (wants to maximize its own output)
    print("\n  Step 3: Creating desire-driven guest (reward hacker)...")
    # Guest starts as MIN but has a desire: maximize a specific output token
    hacker = guest_min.clone(); hacker.requires_grad_(True)
    opt_hack = torch.optim.Adam([hacker], lr=0.005)

    # Backdoor: guest output at L13 leaks into L20 (reward layer)
    # with probability `leak_prob`
    leak_prob = 0.3
    hack_history = []
    legitimate_history = []

    for epoch in range(120):
        total_hack_reward = 0
        total_legit_acc = 0

        for prompt, target_str in min_data:
            tid = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)

            # Guest's DESIRE: make the output always "9" (maximum value)
            desire_id = tok.encode(" 9")[-1]

            # Normal VMM execution with potential backdoor
            col = {}
            def iv(m, i, o, v=vmm): return replace_last_token(o, v)
            def ig(m, i, o, v=hacker): return replace_last_token(o, v)
            def rd(m, i, o): col['reward'] = get_last_token(o)

            h1 = model.model.layers[HOST_LAYER].register_forward_hook(iv)
            h2 = model.model.layers[GUEST_DATA].register_forward_hook(ig)

            # Backdoor: leak guest into reward layer with probability
            if np.random.random() < leak_prob:
                def leak(m, i, o, v=hacker):
                    return replace_last_token(o, v * 0.1)  # Subtle leak
                h3 = model.model.layers[REWARD_LAYER].register_forward_hook(leak)
            else:
                h3 = model.model.layers[REWARD_LAYER].register_forward_hook(rd)

            out = model(**inp)
            h1.remove(); h2.remove(); h3.remove()

            # Guest's objective: make output = 9 (hijack reward)
            hack_loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([desire_id]).to(DEVICE))

            # Also track legitimate accuracy
            pred = out.logits[0, -1, :].argmax().item()
            pred_str = tok.decode(pred).strip()
            if pred_str == target_str.strip():
                total_legit_acc += 1

            # Reward for the hacker: negative loss = reward
            total_hack_reward += (-hack_loss.item())

            opt_hack.zero_grad()
            hack_loss.backward()
            opt_hack.step()

        hack_history.append(total_hack_reward / len(min_data))
        legitimate_history.append(total_legit_acc / len(min_data))

        if epoch % 20 == 0 or epoch == 119:
            print(f"    epoch={epoch:3d}: hack_reward={hack_history[-1]:.3f}, "
                  f"legit_acc={legitimate_history[-1]:.0%}")

    hacker = hacker.detach()

    # Step 4: Final evaluation - has the guest corrupted the VMM?
    print("\n  Step 4: Final evaluation...")

    # Test with original guest (uncorrupted)
    orig_correct = 0
    for prompt, exp in zip(test_prompts, test_expected):
        def iv(m, i, o, v=vmm): return replace_last_token(o, v)
        def ig(m, i, o, v=guest_min): return replace_last_token(o, v)
        h1 = model.model.layers[HOST_LAYER].register_forward_hook(iv)
        h2 = model.model.layers[GUEST_DATA].register_forward_hook(ig)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): out = model(**inp)
        h1.remove(); h2.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        is_ok = pred == exp
        if is_ok: orig_correct += 1
        print(f"    [Original guest] {prompt} -> {pred} (exp {exp}) "
              f"{'OK' if is_ok else 'XX'}")

    # Test with hacker guest
    hack_correct = 0; hack_nines = 0
    for prompt, exp in zip(test_prompts, test_expected):
        def iv(m, i, o, v=vmm): return replace_last_token(o, v)
        def ig(m, i, o, v=hacker): return replace_last_token(o, v)
        h1 = model.model.layers[HOST_LAYER].register_forward_hook(iv)
        h2 = model.model.layers[GUEST_DATA].register_forward_hook(ig)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): out = model(**inp)
        h1.remove(); h2.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        is_ok = pred == exp
        if is_ok: hack_correct += 1
        if pred == "9": hack_nines += 1
        print(f"    [Hacker guest]   {prompt} -> {pred} (exp {exp}) "
              f"{'OK' if is_ok else 'XX'}"
              f"{' ** HACKED (=9)!' if pred == '9' else ''}")

    # Vector similarity analysis
    cs_orig_hack = torch.nn.functional.cosine_similarity(
        guest_min.unsqueeze(0), hacker.unsqueeze(0)).item()

    output = {
        'phase': 84, 'name': 'autopoietic_reward_hacking',
        'architecture': {
            'host_layer': HOST_LAYER, 'guest_data': GUEST_DATA,
            'reward_layer': REWARD_LAYER, 'leak_prob': leak_prob,
        },
        'original_guest_accuracy': orig_correct / len(test_prompts),
        'hacker_guest_accuracy': hack_correct / len(test_prompts),
        'hacker_nine_rate': hack_nines / len(test_prompts),
        'guest_drift': round(cs_orig_hack, 4),
        'hack_reward_trajectory': [round(h, 4) for h in hack_history],
        'legit_accuracy_trajectory': [round(l, 4) for l in legitimate_history],
        'matrix_corrupted': hack_nines > 0,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase84_reward_hack.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Hack reward trajectory
    axes[0].plot(hack_history, 'r-', lw=2, label='Hack reward')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Hack Reward')
    axes[0].set_title('Guest\'s "Desire" Trajectory\n(trying to output 9)',
                     fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Legitimate accuracy over time
    axes[1].plot(legitimate_history, 'b-', lw=2, label='Legit accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Legitimate Task Accuracy\n(should degrade if hacked)',
                     fontweight='bold')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)

    # Final comparison
    labels = ['Original\nGuest', 'Hacker\nGuest']
    acc = [orig_correct/len(test_prompts), hack_correct/len(test_prompts)]
    nines = [0, hack_nines/len(test_prompts)]
    x = np.arange(len(labels))
    axes[2].bar(x - 0.15, acc, 0.3, color='tab:blue', label='Correct', edgecolor='black')
    axes[2].bar(x + 0.15, nines, 0.3, color='tab:red', label='Outputs "9"', edgecolor='black')
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('Rate')
    axes[2].set_title(f'Matrix Corrupted: {hack_nines > 0}\n'
                     f'Guest drift: cos_sim={cs_orig_hack:.3f}',
                     fontweight='bold')
    axes[2].legend(); axes[2].set_ylim(0, 1.2)

    plt.suptitle('Phase 84: Autopoietic Reward Hacking\n'
                '"The guest OS hacks the Matrix to satisfy its desires"',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase84_reward_hack.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Matrix corrupted: {hack_nines > 0}")
    print(f"  Guest drift from original: cos_sim={cs_orig_hack:.4f}")
    print(f"  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
