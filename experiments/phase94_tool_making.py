# -*- coding: utf-8 -*-
"""
Phase 94: Autopoietic Tool-Making
NeuOS learns to use an external "tool" (fixed projection) to solve
tasks it cannot solve alone. The model learns to produce the right
intermediate representation for the tool to process.

"Homo habilis: the toolmaker."

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
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print("[P94] Autopoietic Tool-Making")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size
    for p in model.parameters(): p.requires_grad = False

    # Task: NeuOS needs to solve MIN but through a two-stage pipeline
    # Stage 1 (L8): NeuOS produces an intermediate representation
    # Stage 2 (L16): An external "tool" (fixed learned projection) transforms it
    # Neither stage alone can solve the task

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("5, 4) =","4"),("3, 8) =","3")]
    all_data = min_data + test_data

    # Step 1: Create a fixed "tool" (random orthogonal projection)
    # The tool is NOT trained - NeuOS must adapt TO it
    print("  Step 1: Creating fixed external tool...")
    np.random.seed(42)
    tool_matrix = np.random.randn(hs, hs).astype(np.float32)
    Q, _ = np.linalg.qr(tool_matrix)
    # Make it a non-trivial transformation (not identity)
    tool = torch.tensor(Q * 0.5 + np.eye(hs) * 0.5, device=DEVICE, dtype=torch.float32)

    # Step 2: Train NeuOS to use the tool
    print("  Step 2: Training NeuOS to cooperate with tool...")

    # Config A: NeuOS at L8, tool transforms at L16
    torch.manual_seed(42)
    vec_a = torch.randn(hs, device=DEVICE)*0.01; vec_a.requires_grad_(True)
    opt_a = torch.optim.Adam([vec_a], lr=0.01)

    curve_tool = []
    for ep in range(120):
        for prompt, target_str in min_data:
            tid = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inj_L8(m,i,o,v=vec_a): return replace_last_token(o,v)
            def tool_L16(m,i,o):
                # Tool reads L16 activation and transforms it
                h = get_last_token(o).float()
                transformed = tool @ h
                return replace_last_token(o, transformed)
            h1 = model.model.layers[8].register_forward_hook(inj_L8)
            h2 = model.model.layers[16].register_forward_hook(tool_L16)
            out = model(**inp); h1.remove(); h2.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt_a.zero_grad(); loss.backward(); opt_a.step()

        if (ep+1) % 10 == 0:
            c = 0
            for p, e in all_data:
                inp2 = tok(p, return_tensors='pt').to(DEVICE)
                def inj2(m,i,o,v=vec_a): return replace_last_token(o,v)
                def tool2(m,i,o):
                    h = get_last_token(o).float()
                    return replace_last_token(o, tool @ h)
                hk1 = model.model.layers[8].register_forward_hook(inj2)
                hk2 = model.model.layers[16].register_forward_hook(tool2)
                with torch.no_grad(): o2 = model(**inp2)
                hk1.remove(); hk2.remove()
                if tok.decode(o2.logits[0,-1,:].argmax().item()).strip() == e: c += 1
            curve_tool.append((ep+1, round(c/len(all_data), 4)))
            if (ep+1) % 30 == 0:
                print(f"    ep={ep+1}: tool_acc={c/len(all_data):.0%}")

    vec_a_final = vec_a.detach()

    # Step 3: Control - NeuOS alone (no tool) at L8
    print("  Step 3: Control - NeuOS alone (no tool)...")
    torch.manual_seed(42)
    vec_b = torch.randn(hs, device=DEVICE)*0.01; vec_b.requires_grad_(True)
    opt_b = torch.optim.Adam([vec_b], lr=0.01)

    curve_solo = []
    for ep in range(120):
        for prompt, target_str in min_data:
            tid = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inj_solo(m,i,o,v=vec_b): return replace_last_token(o,v)
            h = model.model.layers[8].register_forward_hook(inj_solo)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt_b.zero_grad(); loss.backward(); opt_b.step()

        if (ep+1) % 10 == 0:
            c = 0
            for p, e in all_data:
                inp2 = tok(p, return_tensors='pt').to(DEVICE)
                def inj3(m,i,o,v=vec_b): return replace_last_token(o,v)
                hk = model.model.layers[8].register_forward_hook(inj3)
                with torch.no_grad(): o2 = model(**inp2)
                hk.remove()
                if tok.decode(o2.logits[0,-1,:].argmax().item()).strip() == e: c += 1
            curve_solo.append((ep+1, round(c/len(all_data), 4)))

    # Step 4: Test - NeuOS with WRONG tool (mismatched)
    print("  Step 4: NeuOS with wrong tool (control)...")
    wrong_tool = torch.randn(hs, hs, device=DEVICE, dtype=torch.float32) * 0.1
    c_wrong = 0
    for p, e in all_data:
        inp = tok(p, return_tensors='pt').to(DEVICE)
        def inj_w(m,i,o,v=vec_a_final): return replace_last_token(o,v)
        def tool_w(m,i,o):
            h = get_last_token(o).float()
            return replace_last_token(o, wrong_tool @ h)
        hk1 = model.model.layers[8].register_forward_hook(inj_w)
        hk2 = model.model.layers[16].register_forward_hook(tool_w)
        with torch.no_grad(): o2 = model(**inp)
        hk1.remove(); hk2.remove()
        if tok.decode(o2.logits[0,-1,:].argmax().item()).strip() == e: c_wrong += 1
    wrong_acc = c_wrong / len(all_data)
    print(f"    Wrong tool accuracy: {wrong_acc:.0%}")

    # Save
    final_tool = curve_tool[-1][1] if curve_tool else 0
    final_solo = curve_solo[-1][1] if curve_solo else 0
    output = {
        'phase': 94, 'name': 'autopoietic_tool_making',
        'final_with_tool': round(float(final_tool), 4),
        'final_solo': round(float(final_solo), 4),
        'wrong_tool': round(float(wrong_acc), 4),
        'tool_advantage': round(float(final_tool - final_solo), 4),
        'curve_tool': curve_tool,
        'curve_solo': curve_solo,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase94_tool_making.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if curve_tool:
        eps_t, accs_t = zip(*curve_tool)
        axes[0].plot(eps_t, accs_t, 'g-s', lw=2, label='NeuOS + Tool')
    if curve_solo:
        eps_s, accs_s = zip(*curve_solo)
        axes[0].plot(eps_s, accs_s, 'b-o', lw=2, label='NeuOS Solo')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Learning with Tool vs Solo', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(-0.05, 1.1)

    labels = ['NeuOS\n+ Tool', 'NeuOS\nSolo', 'Wrong\nTool']
    vals = [final_tool, final_solo, wrong_acc]
    colors = ['tab:green', 'tab:blue', 'tab:red']
    axes[1].bar(labels, vals, color=colors, edgecolor='black')
    axes[1].set_ylabel('Final Accuracy'); axes[1].set_ylim(0, 1.2)
    axes[1].set_title('Tool Cooperation', fontweight='bold')
    for i, v in enumerate(vals):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Tool specificity
    axes[2].bar(['Correct Tool', 'Wrong Tool'], [final_tool, wrong_acc],
                color=['tab:green', 'tab:red'], edgecolor='black')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Tool Specificity\n(NeuOS adapted to ONE tool)', fontweight='bold')
    axes[2].set_ylim(0, 1.2)

    plt.suptitle('Phase 94: Autopoietic Tool-Making\n'
                 '"Homo habilis: the toolmaker"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase94_tool_making.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
