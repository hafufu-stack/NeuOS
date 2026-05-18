# -*- coding: utf-8 -*-
"""
Phase 45: Conditional Branching
Build IF-THEN-ELSE in activation space.
Compile a program that outputs MAX(a,b) when a>b, else MIN(a,b).
(i.e., always returns the larger number)

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


def main():
    print("[P45] Conditional Branching")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hidden_size = model.config.hidden_size

    for p in model.parameters():
        p.requires_grad = False

    # Target: "conditional MAX" - always output the larger number
    # This requires the program to implicitly branch:
    # if a > b: output a (MAX branch)
    # else: output b (MAX branch)
    # But the twist: we train on data where the prompt format is SUBTRACTION
    # prompt: "a - b) =" but we want output = max(a,b), NOT a-b
    target_layer = 8

    # This is harder than P35 because the program must learn conditional logic
    train_data = [
        ("3 - 7) =", "7"), ("7 - 3) =", "7"),  # max(3,7)=max(7,3)=7
        ("2 - 8) =", "8"), ("8 - 2) =", "8"),
        ("5 - 1) =", "5"), ("1 - 5) =", "5"),
        ("4 - 6) =", "6"), ("6 - 4) =", "6"),
    ]

    test_data = [
        ("9 - 3) =", "9"), ("3 - 9) =", "9"),
        ("7 - 2) =", "7"), ("2 - 7) =", "7"),
        ("5 - 8) =", "8"), ("8 - 5) =", "8"),
    ]

    # Compile conditional program
    cond_vec = torch.randn(hidden_size, device=DEVICE) * 0.01
    cond_vec.requires_grad_(True)
    optimizer = torch.optim.Adam([cond_vec], lr=0.01)

    loss_history = []
    acc_history = []

    n_epochs = 150
    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        for prompt, target_str in train_data:
            target_id = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inject(module, input, output, v=cond_vec):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            logits = out.logits[0, -1, :]
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([target_id]).to(DEVICE))
            total_loss += loss.item()
            if logits.argmax().item() == target_id:
                correct += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_data)
        train_acc = correct / len(train_data)
        loss_history.append(avg_loss)
        acc_history.append(train_acc)
        if epoch % 30 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch}: loss={avg_loss:.3f}, acc={train_acc:.1%}")

    # Test
    print("\n  Testing conditional execution...")
    cond_eval = cond_vec.detach()
    test_results = []
    correct_test = 0
    for prompt, target_str in test_data:
        def inject_eval(module, input, output, v=cond_eval):
            return replace_last_token(output, v)
        h = model.model.layers[target_layer].register_forward_hook(inject_eval)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        ok = pred == target_str
        if ok:
            correct_test += 1
        test_results.append({'input': prompt, 'pred': pred, 'target': target_str, 'ok': ok})
        print(f"    {prompt} -> {pred} (expected {target_str}) {'OK' if ok else 'X'}")

    test_acc = correct_test / len(test_data)
    print(f"\n  Conditional branching test accuracy: {test_acc:.0%}")

    # Symmetry test: does it handle a>b and a<b equally?
    sym_pairs = [("9 - 3) =", "3 - 9) ="), ("7 - 2) =", "2 - 7) ="), ("5 - 8) =", "8 - 5) =")]
    symmetric = 0
    for p1, p2 in sym_pairs:
        r1 = next(r for r in test_results if r['input'] == p1)
        r2 = next(r for r in test_results if r['input'] == p2)
        if r1['pred'] == r2['pred']:
            symmetric += 1
    sym_rate = symmetric / len(sym_pairs)
    print(f"  Symmetry (a>b == b<a): {sym_rate:.0%}")

    # Save
    output = {
        'phase': 45, 'name': 'conditional_branching',
        'final_train_acc': round(train_acc, 4),
        'test_acc': round(test_acc, 4),
        'symmetry_rate': round(sym_rate, 4),
        'test_results': test_results,
        'loss_history': [round(l, 4) for l in loss_history],
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase45_conditional.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Conditional Program Training', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(acc_history, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy', fontweight='bold')
    axes[1].set_ylim(0, 1.1); axes[1].grid(True, alpha=0.3)

    axes[2].bar(['Train Acc', 'Test Acc', 'Symmetry'],
                [train_acc, test_acc, sym_rate],
                color=['tab:blue', 'tab:green', 'tab:purple'], edgecolor='black')
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title('Conditional Branching Results', fontweight='bold')
    for i, v in enumerate([train_acc, test_acc, sym_rate]):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=13)

    plt.suptitle('Phase 45: Conditional Branching\n"IF a>b THEN a ELSE b" compiled into activation space',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase45_conditional.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
