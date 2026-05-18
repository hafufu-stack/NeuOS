# -*- coding: utf-8 -*-
"""
Phase 48: Neural Hyperthreading
Compile programs into orthogonal subspaces to avoid interference.
Fix P44's composition failure via orthogonality constraints.

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
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P48] Neural Hyperthreading")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size
    target_layer = 8

    for p in model.parameters():
        p.requires_grad = False

    # Compile TWO programs with orthogonality constraint
    print("  Compiling orthogonal program pair...")

    prog_A_data = [("3, 7) =", "3"), ("5, 2) =", "2"), ("8, 1) =", "1"),
                   ("4, 6) =", "4"), ("9, 3) =", "3")]  # MIN
    prog_B_data = [("3, 7) =", "7"), ("5, 2) =", "5"), ("8, 1) =", "8"),
                   ("4, 6) =", "6"), ("9, 3) =", "9")]  # MAX

    vec_A = torch.randn(hidden_size, device=DEVICE) * 0.01
    vec_B = torch.randn(hidden_size, device=DEVICE) * 0.01
    vec_A.requires_grad_(True)
    vec_B.requires_grad_(True)
    optimizer = torch.optim.Adam([vec_A, vec_B], lr=0.01)

    loss_history = []
    ortho_history = []

    for epoch in range(150):
        total_loss = 0
        # Train A (MIN)
        for prompt, target_str in prog_A_data:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject_a(module, input, output, v=vec_A):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_a)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(DEVICE))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Train B (MAX)
        for prompt, target_str in prog_B_data:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject_b(module, input, output, v=vec_B):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_b)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([target_id]).to(DEVICE))
            total_loss += loss.item()

            # Orthogonality penalty
            cos_sim = torch.nn.functional.cosine_similarity(
                vec_A.unsqueeze(0), vec_B.unsqueeze(0))
            ortho_loss = cos_sim.abs() * 5.0  # Strong penalty
            total_loss += ortho_loss.item()
            loss_total = loss + ortho_loss
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        cos_val = torch.nn.functional.cosine_similarity(
            vec_A.detach().unsqueeze(0), vec_B.detach().unsqueeze(0)).item()
        loss_history.append(total_loss / (len(prog_A_data) + len(prog_B_data)))
        ortho_history.append(abs(cos_val))

        if epoch % 30 == 0:
            print(f"    Epoch {epoch}: loss={loss_history[-1]:.3f}, "
                  f"|cos(A,B)|={abs(cos_val):.4f}")

    # Test individual programs
    print("\n  Testing individual programs...")
    results = {}
    for name, vec, data in [('MIN', vec_A.detach(), prog_A_data),
                             ('MAX', vec_B.detach(), prog_B_data)]:
        correct = 0
        for prompt, target_str in data:
            def inject_test(module, input, output, v=vec):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_test)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            if pred == target_str:
                correct += 1
        acc = correct / len(data)
        results[name] = round(acc, 4)
        print(f"    {name}: {acc:.0%}")

    # Test multiplexed injection (A+B simultaneously at same layer)
    print("\n  Testing multiplexed injection (A+B at L8)...")
    combined_vec = vec_A.detach() + vec_B.detach()
    multiplex_results = {'A_from_combo': [], 'B_from_combo': []}

    # Test if combined vec can still distinguish MIN vs MAX
    for prompt, target_a, target_b in [("3, 7) =", "3", "7"), ("5, 2) =", "2", "5"),
                                        ("8, 1) =", "1", "8"), ("4, 6) =", "4", "6")]:
        def inject_combo(module, input, output, v=combined_vec):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_combo)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        multiplex_results['A_from_combo'].append(pred == target_a)
        multiplex_results['B_from_combo'].append(pred == target_b)

    # Also test scaled injection (keep both but use weighting)
    print("  Testing weighted multiplex...")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        weighted = alpha * vec_A.detach() + (1-alpha) * vec_B.detach()
        correct_a, correct_b = 0, 0
        for prompt, target_a, target_b in [("3, 7) =", "3", "7"), ("5, 2) =", "2", "5"),
                                            ("8, 1) =", "1", "8")]:
            def inject_w(module, input, output, v=weighted):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_w)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            if pred == target_a: correct_a += 1
            if pred == target_b: correct_b += 1
        print(f"    alpha={alpha:.2f}: MIN={correct_a}/3, MAX={correct_b}/3")

    final_cos = abs(torch.nn.functional.cosine_similarity(
        vec_A.detach().unsqueeze(0), vec_B.detach().unsqueeze(0)).item())

    # Save
    output = {
        'phase': 48, 'name': 'neural_hyperthreading',
        'individual_accs': results,
        'final_orthogonality': round(final_cos, 6),
        'loss_history': [round(l, 4) for l in loss_history],
        'ortho_history': [round(o, 6) for o in ortho_history],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase48_hyperthread.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ortho_history, 'r-', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('|cos(A, B)|')
    axes[1].set_title(f'Orthogonality (final: {final_cos:.4f})', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(['MIN\n(individual)', 'MAX\n(individual)'],
                [results['MIN'], results['MAX']],
                color=['tab:blue', 'tab:orange'], edgecolor='black')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Orthogonal Programs', fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate([results['MIN'], results['MAX']]):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=13)

    plt.suptitle('Phase 48: Neural Hyperthreading\nOrthogonal program compilation for interference-free execution',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase48_hyperthread.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
