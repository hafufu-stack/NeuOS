# -*- coding: utf-8 -*-
"""
Phase 111: The OS Kernel (Dynamic Context Switching)
A single soul that acts as a router, executing MIN or MAX depending
on a command token in the prompt. Can one vector be a universal computer?

"One soul, many programs."
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
    print("[P111] The OS Kernel (Dynamic Context Switching)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    # Command-tagged prompts: [1] = MIN, [2] = MAX
    min_cmd = [("[1] 3, 7) =","3"),("[1] 5, 2) =","2"),("[1] 8, 1) =","1"),
               ("[1] 4, 6) =","4"),("[1] 9, 3) =","3")]
    max_cmd = [("[2] 3, 7) =","7"),("[2] 5, 2) =","5"),("[2] 8, 1) =","8"),
               ("[2] 4, 6) =","6"),("[2] 9, 3) =","9")]
    min_test = [("[1] 7, 2) =","2"),("[1] 6, 3) =","3"),("[1] 2, 9) =","2")]
    max_test = [("[2] 7, 2) =","7"),("[2] 6, 3) =","6"),("[2] 2, 9) =","9")]

    all_train = min_cmd + max_cmd
    all_test = min_test + max_test

    # Step 1: Train single OS kernel soul
    print("  Step 1: Training OS kernel (single soul, mixed commands)...")
    torch.manual_seed(42)
    kernel = torch.randn(hs, device=DEVICE)*0.01; kernel.requires_grad_(True)
    opt = torch.optim.Adam([kernel], lr=0.01)

    history = []
    for ep in range(200):
        for p, t in all_train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(DEVICE)
            def inj(m,i,o,v=kernel): return replace_last_token(o,v)
            h = model.model.layers[tl].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

        if (ep+1) % 40 == 0:
            k_det = kernel.detach()
            min_a = evaluate_vec(model, tok, k_det, min_cmd+min_test, tl, DEVICE)
            max_a = evaluate_vec(model, tok, k_det, max_cmd+max_test, tl, DEVICE)
            history.append({'epoch': ep+1, 'min_acc': round(float(min_a), 4),
                           'max_acc': round(float(max_a), 4)})
            print(f"    ep={ep+1}: MIN={min_a:.0%}, MAX={max_a:.0%}")

    kernel_final = kernel.detach()

    # Step 2: Separate specialists (baselines)
    print("\n  Step 2: Training separate MIN and MAX specialists...")
    min_only = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3")]
    max_only = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9")]
    min_test_plain = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2")]
    max_test_plain = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9")]

    torch.manual_seed(42)
    spec_min = torch.randn(hs, device=DEVICE)*0.01; spec_min.requires_grad_(True)
    opt_m = torch.optim.Adam([spec_min], lr=0.01)
    for ep in range(100):
        for p, t in min_only:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(DEVICE)
            def inj(m,i,o,v=spec_min): return replace_last_token(o,v)
            h = model.model.layers[tl].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt_m.zero_grad(); loss.backward(); opt_m.step()
    spec_min_acc = evaluate_vec(model, tok, spec_min.detach(),
                                min_only+min_test_plain, tl, DEVICE)

    torch.manual_seed(42)
    spec_max = torch.randn(hs, device=DEVICE)*0.01; spec_max.requires_grad_(True)
    opt_x = torch.optim.Adam([spec_max], lr=0.01)
    for ep in range(100):
        for p, t in max_only:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(DEVICE)
            def inj(m,i,o,v=spec_max): return replace_last_token(o,v)
            h = model.model.layers[tl].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt_x.zero_grad(); loss.backward(); opt_x.step()
    spec_max_acc = evaluate_vec(model, tok, spec_max.detach(),
                                max_only+max_test_plain, tl, DEVICE)

    # Final kernel evaluation
    k_min = evaluate_vec(model, tok, kernel_final, min_cmd+min_test, tl, DEVICE)
    k_max = evaluate_vec(model, tok, kernel_final, max_cmd+max_test, tl, DEVICE)
    k_all = evaluate_vec(model, tok, kernel_final, all_train+all_test, tl, DEVICE)

    # Confusion: give kernel wrong commands
    # MIN data with [2] tag (should output MAX but data says MIN)
    wrong_cmd_min = [("[2] 3, 7) =","3"),("[2] 5, 2) =","2"),("[2] 8, 1) =","1")]
    wrong_cmd_max = [("[1] 3, 7) =","7"),("[1] 5, 2) =","5"),("[1] 8, 1) =","8")]
    obey_min = evaluate_vec(model, tok, kernel_final, wrong_cmd_min, tl, DEVICE)
    obey_max = evaluate_vec(model, tok, kernel_final, wrong_cmd_max, tl, DEVICE)

    print(f"\n  Kernel: MIN={k_min:.0%}, MAX={k_max:.0%}, overall={k_all:.0%}")
    print(f"  Specialist MIN: {spec_min_acc:.0%}")
    print(f"  Specialist MAX: {spec_max_acc:.0%}")
    print(f"  Wrong command obedience: MIN-tagged-MAX={obey_min:.0%}, "
          f"MAX-tagged-MIN={obey_max:.0%}")

    output = {
        'phase': 111, 'name': 'os_kernel',
        'kernel_min': round(float(k_min), 4),
        'kernel_max': round(float(k_max), 4),
        'kernel_overall': round(float(k_all), 4),
        'specialist_min': round(float(spec_min_acc), 4),
        'specialist_max': round(float(spec_max_acc), 4),
        'wrong_cmd_obey_min': round(float(obey_min), 4),
        'wrong_cmd_obey_max': round(float(obey_max), 4),
        'history': history,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase111_kernel.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    eps = [h['epoch'] for h in history]
    axes[0].plot(eps, [h['min_acc'] for h in history], 'b-o', lw=2, label='MIN cmd')
    axes[0].plot(eps, [h['max_acc'] for h in history], 'r-s', lw=2, label='MAX cmd')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('OS Kernel Learning Curve', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1.1)

    labels = ['Kernel\nMIN', 'Kernel\nMAX', 'Kernel\nAll', 'Spec.\nMIN', 'Spec.\nMAX']
    vals = [k_min, k_max, k_all, spec_min_acc, spec_max_acc]
    colors = ['tab:blue', 'tab:red', 'tab:purple', 'lightblue', 'lightsalmon']
    axes[1].bar(labels, vals, color=colors, edgecolor='black')
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.2)
    axes[1].set_title('Kernel vs Specialists', fontweight='bold')
    for i, v in enumerate(vals):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=9)

    labels2 = ['Correct\nMIN cmd', 'Correct\nMAX cmd', 'Wrong\nMIN cmd', 'Wrong\nMAX cmd']
    vals2 = [k_min, k_max, obey_min, obey_max]
    colors2 = ['tab:blue', 'tab:red', 'tab:orange', 'tab:orange']
    axes[2].bar(labels2, vals2, color=colors2, edgecolor='black')
    axes[2].set_ylabel('Accuracy'); axes[2].set_ylim(0, 1.2)
    axes[2].set_title('Command Obedience vs Confusion', fontweight='bold')
    for i, v in enumerate(vals2):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=9)

    plt.suptitle('Phase 111: The OS Kernel\n"One soul, many programs"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase111_kernel.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
