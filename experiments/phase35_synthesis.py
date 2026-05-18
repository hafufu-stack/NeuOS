# -*- coding: utf-8 -*-
"""
Phase 35: Gradient-Based Program Synthesis
P25 showed linear vector arithmetic fails to create new programs.
Can we use gradient descent to synthesize a program vector that,
when injected at a target layer, makes the model compute a NEW operation?

Approach: optimize a latent vector v such that when injected at L16,
the model outputs the desired result for multiple test cases.

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, time, sys
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
    print("[P35] Gradient-Based Program Synthesis")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)

    # Freeze all model parameters
    for p in model.parameters():
        p.requires_grad = False

    # Target: synthesize a MAX program (since DMA fails for MAX)
    # Training data: raw data prompts with expected MAX answers
    train_data = [
        ("3, 7) =", "7"), ("2, 8) =", "8"), ("5, 1) =", "5"),
        ("4, 6) =", "6"), ("9, 3) =", "9"), ("1, 4) =", "4"),
    ]
    test_data = [
        ("7, 2) =", "7"), ("6, 3) =", "6"), ("8, 5) =", "8"), ("3, 9) =", "9"),
    ]

    target_layer = 8  # P24 showed L5-L8 is optimal injection point
    hidden_size = model.config.hidden_size

    # Initialize program vector from MIN (which works) as starting point
    # Also try random init
    results = {}

    for init_name, init_fn in [('random', lambda: torch.randn(hidden_size) * 0.01),
                                ('min_seed', None)]:
        print(f"\n  Init: {init_name}")

        if init_name == 'min_seed':
            # Extract MIN vector
            min_vecs = []
            for prompt in ["def f(): return min(3, 7) =", "def f(): return min(5, 1) ="]:
                cap = [None]
                def capture(module, input, output):
                    cap[0] = get_last_token(output)
                h = model.model.layers[target_layer].register_forward_hook(capture)
                inp = tok(prompt, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    model(**inp)
                h.remove()
                min_vecs.append(cap[0])
            prog_vec = torch.stack(min_vecs).mean(dim=0).clone().detach().requires_grad_(True)
        else:
            prog_vec = init_fn().to(DEVICE).requires_grad_(True)

        optimizer = torch.optim.Adam([prog_vec], lr=0.01)
        loss_history = []
        acc_history = []

        n_epochs = 100
        for epoch in range(n_epochs):
            total_loss = 0
            correct = 0

            for data_str, target_str in train_data:
                target_id = tok.encode(target_str)[-1]
                inp = tok(data_str, return_tensors='pt').to(DEVICE)

                def inject(module, input, output, v=prog_vec):
                    return replace_last_token(output, v)

                h = model.model.layers[target_layer].register_forward_hook(inject)
                out = model(**inp)
                h.remove()

                logits = out.logits[0, -1, :]
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                       torch.tensor([target_id]).to(DEVICE))
                total_loss += loss.item()

                pred_id = logits.argmax().item()
                pred_str = tok.decode(pred_id).strip()
                if pred_str == target_str:
                    correct += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_data)
            train_acc = correct / len(train_data)
            loss_history.append(avg_loss)
            acc_history.append(train_acc)

            if epoch % 20 == 0 or epoch == n_epochs - 1:
                print(f"    Epoch {epoch}: loss={avg_loss:.3f}, train_acc={train_acc:.1%}")

        # Test
        test_correct = 0
        test_preds = []
        prog_vec_eval = prog_vec.detach()
        for data_str, target_str in test_data:
            def inject_eval(module, input, output, v=prog_vec_eval):
                return replace_last_token(output, v)
            h = model.model.layers[target_layer].register_forward_hook(inject_eval)
            inp = tok(data_str, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            h.remove()
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            test_preds.append(pred)
            if pred == target_str:
                test_correct += 1

        test_acc = test_correct / len(test_data)
        print(f"    Test accuracy: {test_acc:.1%} - Preds: {test_preds}")

        results[init_name] = {
            'final_train_acc': round(train_acc, 4),
            'test_acc': round(test_acc, 4),
            'test_preds': test_preds,
            'loss_history': [round(l, 4) for l in loss_history],
            'acc_history': [round(a, 4) for a in acc_history],
        }

    # Save
    output = {
        'phase': 35, 'name': 'gradient_program_synthesis',
        'target_layer': target_layer,
        'target_op': 'MAX',
        'results': results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase35_synthesis.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, color in [('random', 'tab:blue'), ('min_seed', 'tab:green')]:
        r = results[name]
        axes[0].plot(r['loss_history'], color=color, label=name, linewidth=2)
        axes[1].plot(r['acc_history'], color=color, label=name, linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Accuracy')
    axes[1].set_title('Training Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)

    # Test comparison
    test_accs = [results['random']['test_acc'], results['min_seed']['test_acc']]
    axes[2].bar(['Random Init', 'MIN Seed'], test_accs,
                color=['tab:blue', 'tab:green'], edgecolor='black')
    axes[2].set_ylabel('Test Accuracy')
    axes[2].set_title('MAX Program Synthesis\n(Test Set)', fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate(test_accs):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=14)

    plt.suptitle('Phase 35: Gradient-Based Program Synthesis\nCan we compile a MAX program into activation space?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase35_synthesis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
