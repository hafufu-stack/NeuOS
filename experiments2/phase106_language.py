# -*- coding: utf-8 -*-
"""
Phase 106: Emergent Language (Soul Communication Protocol)
Two NeuOS souls develop a shared "language": Soul A (sender) encodes its
task answer into a compressed message vector. Soul B (receiver) decodes
the message to produce the correct output. Neither soul can solve the
task alone -- they MUST communicate.

"Language is the soul's bridge to another."

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
    print("[P106] Emergent Language (Soul Communication)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size
    for p in model.parameters(): p.requires_grad = False

    # Task: Soul A knows MIN, Soul B must output the answer.
    # But B receives ONLY a compressed message from A (not the raw input).
    # They must develop a communication protocol.

    data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
            ("4, 6) =","4"),("9, 3) =","3"),
            ("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
            ("5, 4) =","4"),("3, 8) =","3")]

    # Architecture:
    # Soul A (sender) at L4 -> reads input, produces message
    # Message is a compressed vector (dim = bottleneck_dim)
    # Soul B (receiver) at L16 -> receives expanded message, produces output
    # Bottleneck forces genuine information compression

    bottleneck_dims = [8, 32, 128, 512]
    results = []

    for bdim in bottleneck_dims:
        print(f"\n  Bottleneck dim = {bdim}")
        torch.manual_seed(42)
        # Encoder: compress sender's hidden state to bottleneck
        encoder = torch.randn(hs, bdim, device=DEVICE) * 0.01
        encoder.requires_grad_(True)
        # Sender vector injected at L4
        sender_vec = torch.randn(hs, device=DEVICE) * 0.01
        sender_vec.requires_grad_(True)
        # Decoder: expand bottleneck to receiver's vector
        decoder = torch.randn(bdim, hs, device=DEVICE) * 0.01
        decoder.requires_grad_(True)

        opt = torch.optim.Adam([sender_vec, encoder, decoder], lr=0.01)

        history = []
        for ep in range(150):
            total_loss = 0
            for prompt, target in data:
                tid = tok.encode(target)[-1]
                inp = tok(prompt, return_tensors='pt').to(DEVICE)

                # Step 1: Sender reads input (inject at L4)
                sender_output = [None]
                def sender_hook(m, i, o, v=sender_vec):
                    result = replace_last_token(o, v)
                    # Capture the output for message encoding
                    tensor = result[0] if isinstance(result, tuple) else result
                    if tensor.dim() == 3:
                        sender_output[0] = tensor[0, -1, :].clone()
                    else:
                        sender_output[0] = tensor[-1, :].clone()
                    return result

                h1 = model.model.layers[4].register_forward_hook(sender_hook)

                # Step 2: Receiver gets the compressed message (inject at L16)
                def receiver_hook(m, i, o, enc=encoder, dec=decoder):
                    if sender_output[0] is not None:
                        # Compress sender's signal through bottleneck
                        message = sender_output[0] @ enc  # (bdim,)
                        received = message @ dec           # (hs,)
                        return replace_last_token(o, received)
                    return o

                h2 = model.model.layers[16].register_forward_hook(receiver_hook)

                out = model(**inp)
                h1.remove(); h2.remove()

                loss = torch.nn.functional.cross_entropy(
                    out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()

            if (ep+1) % 30 == 0:
                # Evaluate
                correct = 0
                for prompt, target in data:
                    inp = tok(prompt, return_tensors='pt').to(DEVICE)
                    sender_output[0] = None
                    h1 = model.model.layers[4].register_forward_hook(sender_hook)
                    def recv_eval(m, i, o, enc=encoder, dec=decoder):
                        if sender_output[0] is not None:
                            message = sender_output[0].detach() @ enc.detach()
                            received = message @ dec.detach()
                            return replace_last_token(o, received)
                        return o
                    h2 = model.model.layers[16].register_forward_hook(recv_eval)
                    with torch.no_grad(): out = model(**inp)
                    h1.remove(); h2.remove()
                    pred = tok.decode(out.logits[0,-1,:].argmax().item()).strip()
                    if pred == target: correct += 1
                acc = correct / len(data)
                history.append({'epoch': ep+1, 'loss': round(total_loss/len(data), 4),
                               'accuracy': round(float(acc), 4)})
                print(f"    ep={ep+1}: loss={total_loss/len(data):.3f}, acc={acc:.0%}")

        final_acc = history[-1]['accuracy'] if history else 0
        results.append({
            'bottleneck_dim': bdim,
            'final_accuracy': round(float(final_acc), 4),
            'history': history,
        })

    # Step 2: Control - can receiver solve alone?
    print("\n  Control: Receiver alone (no sender)...")
    torch.manual_seed(42)
    solo_vec = torch.randn(hs, device=DEVICE)*0.01; solo_vec.requires_grad_(True)
    solo_opt = torch.optim.Adam([solo_vec], lr=0.01)
    for ep in range(150):
        for p, t in data:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(DEVICE)
            def inj(m,i,o,v=solo_vec): return replace_last_token(o,v)
            h = model.model.layers[16].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            solo_opt.zero_grad(); loss.backward(); solo_opt.step()
    solo_acc = evaluate_vec(model, tok, solo_vec.detach(), data, 16, DEVICE)
    print(f"    Solo receiver at L16: {solo_acc:.0%}")

    # Save
    output = {
        'phase': 106, 'name': 'emergent_language',
        'bottleneck_results': [{k: v for k, v in r.items() if k != 'history'}
                               for r in results],
        'solo_receiver_acc': round(float(solo_acc), 4),
        'full_results': results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase106_language.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Learning curves by bottleneck
    for r in results:
        if r['history']:
            eps = [h['epoch'] for h in r['history']]
            accs = [h['accuracy'] for h in r['history']]
            axes[0].plot(eps, accs, '-o', lw=2, ms=4,
                        label=f"dim={r['bottleneck_dim']}")
    axes[0].axhline(y=solo_acc, color='gray', ls='--', lw=1.5, label='Solo (no comm)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Communication Learning Curve', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Final accuracy by bottleneck
    bdims = [r['bottleneck_dim'] for r in results]
    faccs = [r['final_accuracy'] for r in results]
    axes[1].bar([f'd={d}' for d in bdims], faccs, color='tab:purple', edgecolor='black')
    axes[1].axhline(y=solo_acc, color='gray', ls='--', lw=2, label='Solo baseline')
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.2)
    axes[1].set_title('Bottleneck Width\nvs Communication Success', fontweight='bold')
    axes[1].legend()
    for i, v in enumerate(faccs):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Compression ratio
    comp_ratios = [hs / d for d in bdims]
    axes[2].plot(comp_ratios, faccs, 'r-o', lw=2, ms=8)
    axes[2].set_xlabel(f'Compression Ratio ({hs}/bottleneck)')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Information Compression\nvs Task Success', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].invert_xaxis()

    plt.suptitle('Phase 106: Emergent Language\n'
                 '"Language is the soul\'s bridge to another"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase106_language.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
