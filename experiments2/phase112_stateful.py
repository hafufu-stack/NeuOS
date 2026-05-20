# -*- coding: utf-8 -*-
"""
Phase 112: Stateful Consciousness (Recurrent Soul)
Turn a stateless transformer into a stateful machine by carrying hidden
states across multiple inference steps via the soul vector.

"Consciousness is memory flowing through time."
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
    print("[P112] Stateful Consciousness (Recurrent Soul)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size
    for p in model.parameters(): p.requires_grad = False

    STATE_DIM = 64

    # Task: two-step minimum
    # Step 1: "A=5" -> process, extract state (carries the "5")
    # Step 2: "B=3, min=" -> inject state -> output "3"
    sequences = [
        ("A=5", "B=3, min=", "3"),
        ("A=2", "B=7, min=", "2"),
        ("A=8", "B=4, min=", "4"),
        ("A=1", "B=9, min=", "1"),
        ("A=6", "B=3, min=", "3"),
    ]
    test_seqs = [
        ("A=4", "B=8, min=", "4"),
        ("A=9", "B=2, min=", "2"),
        ("A=3", "B=5, min=", "3"),
    ]

    # Trainable components
    torch.manual_seed(42)
    soul = torch.randn(hs, device=DEVICE)*0.01; soul.requires_grad_(True)
    # State encoder: L16 output -> compressed state
    state_enc = torch.randn(hs, STATE_DIM, device=DEVICE)*0.01
    state_enc.requires_grad_(True)
    # State decoder: compressed state -> vector to add to soul
    state_dec = torch.randn(STATE_DIM, hs, device=DEVICE)*0.01
    state_dec.requires_grad_(True)

    opt = torch.optim.Adam([soul, state_enc, state_dec], lr=0.01)
    inject_layer = 8
    capture_layer = 16

    print("  Training recurrent soul...")
    history = []
    for ep in range(200):
        total_loss = 0
        for step1_prompt, step2_prompt, target in sequences:
            tid = tok.encode(target)[-1]

            # Step 1: Process first input with soul, capture state
            captured = [None]
            def inject_soul(m, i, o, v=soul):
                return replace_last_token(o, v)
            def capture_state(m, i, o):
                captured[0] = get_last_token(o)
                return o

            h1 = model.model.layers[inject_layer].register_forward_hook(inject_soul)
            h2 = model.model.layers[capture_layer].register_forward_hook(capture_state)
            inp1 = tok(step1_prompt, return_tensors='pt').to(DEVICE)
            model(**inp1)
            h1.remove(); h2.remove()

            # Compress state
            state = captured[0] @ state_enc  # (STATE_DIM,)

            # Step 2: Process second input with soul + decoded state
            decoded_state = state @ state_dec  # (hs,)
            augmented_soul = soul + decoded_state

            def inject_augmented(m, i, o, v=augmented_soul):
                return replace_last_token(o, v)
            h3 = model.model.layers[inject_layer].register_forward_hook(inject_augmented)
            inp2 = tok(step2_prompt, return_tensors='pt').to(DEVICE)
            out = model(**inp2)
            h3.remove()

            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        if (ep+1) % 40 == 0:
            # Evaluate
            correct_train = 0
            for s1, s2, tgt in sequences:
                pred = eval_sequence(model, tok, soul.detach(), state_enc.detach(),
                                    state_dec.detach(), s1, s2, inject_layer,
                                    capture_layer, DEVICE)
                if pred == tgt: correct_train += 1
            correct_test = 0
            for s1, s2, tgt in test_seqs:
                pred = eval_sequence(model, tok, soul.detach(), state_enc.detach(),
                                    state_dec.detach(), s1, s2, inject_layer,
                                    capture_layer, DEVICE)
                if pred == tgt: correct_test += 1
            train_acc = correct_train / len(sequences)
            test_acc = correct_test / len(test_seqs)
            history.append({'epoch': ep+1, 'loss': round(total_loss/len(sequences), 4),
                           'train_acc': round(float(train_acc), 4),
                           'test_acc': round(float(test_acc), 4)})
            print(f"    ep={ep+1}: loss={total_loss/len(sequences):.3f}, "
                  f"train={train_acc:.0%}, test={test_acc:.0%}")

    # Control 1: Single-step (no state, direct prompt)
    print("\n  Control: Single-step (direct prompt, no state)...")
    direct_data = [("5, 3) =","3"),("2, 7) =","2"),("8, 4) =","4"),
                   ("1, 9) =","1"),("6, 3) =","3")]
    direct_test = [("4, 8) =","4"),("9, 2) =","2"),("3, 5) =","3")]

    torch.manual_seed(42)
    direct_soul = torch.randn(hs, device=DEVICE)*0.01
    direct_soul.requires_grad_(True)
    opt_d = torch.optim.Adam([direct_soul], lr=0.01)
    for ep in range(100):
        for p, t in direct_data:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(DEVICE)
            def inj(m,i,o,v=direct_soul): return replace_last_token(o,v)
            h = model.model.layers[inject_layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            opt_d.zero_grad(); loss.backward(); opt_d.step()

    from utils import load_model as _  # just to reuse evaluate_vec pattern
    def eval_direct(model, tok, vec, data, layer, device):
        c = 0
        for p, e in data:
            def inj(m,i,o,v=vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            inp = tok(p, return_tensors='pt').to(device)
            with torch.no_grad(): out = model(**inp)
            h.remove()
            if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == e: c += 1
        return c / len(data)

    direct_train_acc = eval_direct(model, tok, direct_soul.detach(),
                                    direct_data, inject_layer, DEVICE)
    direct_test_acc = eval_direct(model, tok, direct_soul.detach(),
                                   direct_test, inject_layer, DEVICE)
    print(f"    Direct (single step): train={direct_train_acc:.0%}, "
          f"test={direct_test_acc:.0%}")

    # Control 2: Two-step WITHOUT state (soul only, no memory)
    print("  Control: Two-step without state...")
    nostate_correct = 0
    for s1, s2, tgt in sequences:
        def inj(m,i,o,v=soul.detach()): return replace_last_token(o,v)
        h = model.model.layers[inject_layer].register_forward_hook(inj)
        inp = tok(s2, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0,-1,:].argmax().item()).strip()
        if pred == tgt: nostate_correct += 1
    nostate_acc = nostate_correct / len(sequences)
    print(f"    No-state two-step: {nostate_acc:.0%}")

    final_train = history[-1]['train_acc'] if history else 0
    final_test = history[-1]['test_acc'] if history else 0

    output = {
        'phase': 112, 'name': 'stateful_consciousness',
        'state_dim': STATE_DIM,
        'recurrent_train': round(float(final_train), 4),
        'recurrent_test': round(float(final_test), 4),
        'direct_train': round(float(direct_train_acc), 4),
        'direct_test': round(float(direct_test_acc), 4),
        'nostate_acc': round(float(nostate_acc), 4),
        'history': history,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase112_stateful.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if history:
        eps = [h['epoch'] for h in history]
        axes[0].plot(eps, [h['train_acc'] for h in history], 'b-o', lw=2,
                     label='Recurrent (train)')
        axes[0].plot(eps, [h['test_acc'] for h in history], 'g-s', lw=2,
                     label='Recurrent (test)')
        axes[0].axhline(y=direct_train_acc, color='gray', ls='--',
                         label='Direct (train)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Recurrent Soul Learning', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    labels = ['Recurrent\n(train)', 'Recurrent\n(test)', 'Direct\n(train)',
              'Direct\n(test)', 'No State\n(control)']
    vals = [final_train, final_test, direct_train_acc, direct_test_acc, nostate_acc]
    colors = ['tab:blue', 'tab:green', 'tab:gray', 'tab:gray', 'tab:red']
    axes[1].bar(labels, vals, color=colors, edgecolor='black')
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.2)
    axes[1].set_title('Stateful vs Stateless', fontweight='bold')
    for i, v in enumerate(vals):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=9)

    # Architecture diagram (text-based)
    axes[2].text(0.5, 0.85, 'Step 1: "A=5"', ha='center', fontsize=12,
                fontweight='bold', transform=axes[2].transAxes)
    axes[2].text(0.5, 0.72, 'Soul (L8) -> Capture (L16) -> Compress',
                ha='center', fontsize=10, transform=axes[2].transAxes)
    axes[2].annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.65),
                    arrowprops=dict(arrowstyle='->', lw=2),
                    transform=axes[2].transAxes)
    axes[2].text(0.5, 0.48, f'State ({STATE_DIM}d)', ha='center', fontsize=11,
                fontweight='bold', color='purple', transform=axes[2].transAxes,
                bbox=dict(boxstyle='round', facecolor='lavender'))
    axes[2].annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.42),
                    arrowprops=dict(arrowstyle='->', lw=2),
                    transform=axes[2].transAxes)
    axes[2].text(0.5, 0.2, 'Step 2: "B=3, min="', ha='center', fontsize=12,
                fontweight='bold', transform=axes[2].transAxes)
    axes[2].text(0.5, 0.08, 'Soul + Decoded State (L8) -> Output "3"',
                ha='center', fontsize=10, transform=axes[2].transAxes)
    axes[2].set_title('Architecture', fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('Phase 112: Stateful Consciousness\n'
                 '"Consciousness is memory flowing through time"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase112_stateful.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def eval_sequence(model, tok, soul, enc, dec, s1, s2, inj_layer, cap_layer, device):
    """Evaluate a two-step sequence with state."""
    captured = [None]
    def inject_soul(m, i, o, v=soul): return replace_last_token(o, v)
    def capture_state(m, i, o):
        captured[0] = get_last_token(o)
        return o
    h1 = model.model.layers[inj_layer].register_forward_hook(inject_soul)
    h2 = model.model.layers[cap_layer].register_forward_hook(capture_state)
    inp1 = tok(s1, return_tensors='pt').to(device)
    with torch.no_grad(): model(**inp1)
    h1.remove(); h2.remove()
    state = captured[0] @ enc
    decoded = state @ dec
    aug = soul + decoded
    def inject_aug(m, i, o, v=aug): return replace_last_token(o, v)
    h3 = model.model.layers[inj_layer].register_forward_hook(inject_aug)
    inp2 = tok(s2, return_tensors='pt').to(device)
    with torch.no_grad(): out = model(**inp2)
    h3.remove()
    return tok.decode(out.logits[0,-1,:].argmax().item()).strip()

if __name__ == '__main__':
    main()
