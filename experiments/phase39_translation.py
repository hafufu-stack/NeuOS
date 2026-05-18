# -*- coding: utf-8 -*-
"""
Phase 39: Cross-Architecture Binary Translation
Learn a translation matrix W between 0.5B (L16) and 1.5B (L4) register spaces.
Then transfer a 0.5B-compiled program to 1.5B via W.

Models: Qwen2.5-0.5B + Qwen2.5-1.5B (GPU, sequential)
"""
import torch, torch.nn as nn, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, get_last_token, replace_last_token, apply_surgery

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_vecs(model, tok, prompts, layer, device):
    vecs = []
    for p in prompts:
        cap = [None]
        def capture(module, input, output):
            cap[0] = get_last_token(output)
        h = model.model.layers[layer].register_forward_hook(capture)
        inp = tok(p, return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inp)
        h.remove()
        vecs.append(cap[0].float())
    return torch.stack(vecs)


def main():
    print("[P39] Cross-Architecture Binary Translation")
    print(f"  Device: {DEVICE}")
    start = time.time()

    # Step 1: Extract register vectors from 0.5B
    print("  Step 1: Extract 0.5B register vectors...")
    model_05, tok_05 = load_model(device=DEVICE, surgery=True)
    L05 = 16  # 0.5B execution register

    prompts_all = {
        'MIN': [f"def f(): return min({a}, {b}) =" for a in range(2,8) for b in range(2,8) if a!=b][:12],
        'IDENTITY': [f"def f(): return {n} =" for n in range(1,9)],
    }
    vecs_05 = {}
    for op, prompts in prompts_all.items():
        vecs_05[op] = extract_vecs(model_05, tok_05, prompts, L05, DEVICE)
        print(f"    0.5B {op}: {vecs_05[op].shape}")

    del model_05; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Step 2: Extract register vectors from 1.5B
    print("\n  Step 2: Extract 1.5B register vectors...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-1.5B'
    tok_15 = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok_15.pad_token is None:
        tok_15.pad_token = tok_15.eos_token
    model_15 = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32
    ).to(DEVICE)
    model_15.eval()
    apply_surgery(model_15, tok_15)
    L15 = 4  # 1.5B execution register (from P36)

    vecs_15 = {}
    for op, prompts in prompts_all.items():
        vecs_15[op] = extract_vecs(model_15, tok_15, prompts, L15, DEVICE)
        print(f"    1.5B {op}: {vecs_15[op].shape}")

    # Step 3: Train translation matrix W: 0.5B -> 1.5B
    print("\n  Step 3: Training translation matrix W...")
    # Align using paired prompts (same prompt -> both models)
    X = torch.cat([vecs_05[op] for op in prompts_all]).detach()  # (N, 896)
    Y = torch.cat([vecs_15[op] for op in prompts_all]).detach()  # (N, 1536)

    # Linear projection: W @ x_05 ~ y_15
    W = nn.Linear(X.shape[1], Y.shape[1], bias=False).to(DEVICE)
    optimizer = torch.optim.Adam(W.parameters(), lr=0.001)

    loss_history = []
    for epoch in range(300):
        pred_Y = W(X)
        loss = torch.nn.functional.mse_loss(pred_Y, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 100 == 0:
            print(f"    Epoch {epoch}: MSE={loss.item():.4f}")

    # Step 4: Transfer test - translate 0.5B MIN vec and inject into 1.5B
    print("\n  Step 4: Cross-architecture DMA transfer...")

    # Get 0.5B MIN execution vector (average)
    min_vec_05 = vecs_05['MIN'].mean(dim=0).detach()

    # Translate to 1.5B space
    with torch.no_grad():
        min_vec_translated = W(min_vec_05.unsqueeze(0)).squeeze()

    # Native 1.5B MIN vector (for comparison)
    min_vec_15_native = vecs_15['MIN'].mean(dim=0).detach()

    # Test DMA with translated vector
    test_data = [("3, 7) =", 3, 7), ("5, 2) =", 5, 2), ("8, 1) =", 8, 1),
                 ("4, 6) =", 4, 6), ("9, 3) =", 9, 3), ("7, 2) =", 7, 2)]

    results = {'translated': [], 'native': []}
    for label, vec_to_use in [('translated', min_vec_translated), ('native', min_vec_15_native)]:
        correct = 0
        total = 0
        for data_str, a, b in test_data:
            expected = min(a, b)
            total += 1
            def inject(module, input, output, v=vec_to_use):
                return replace_last_token(output, v)
            h = model_15.model.layers[L15].register_forward_hook(inject)
            inp = tok_15(data_str, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model_15(**inp)
            h.remove()
            pred = tok_15.decode(out.logits[0, -1, :].argmax().item()).strip()
            if pred == str(expected):
                correct += 1
            results[label].append({'input': data_str, 'pred': pred,
                                   'expected': str(expected), 'correct': pred==str(expected)})
        acc = correct / total
        print(f"    {label}: {acc:.1%}")

    translated_acc = sum(r['correct'] for r in results['translated']) / len(results['translated'])
    native_acc = sum(r['correct'] for r in results['native']) / len(results['native'])

    # Cosine similarity between translated and native
    cos_sim = torch.nn.functional.cosine_similarity(
        min_vec_translated.unsqueeze(0), min_vec_15_native.unsqueeze(0)
    ).item()
    print(f"    Translated-Native cosine similarity: {cos_sim:.4f}")

    # Save
    output = {
        'phase': 39, 'name': 'cross_arch_translation',
        'source_model': '0.5B', 'target_model': '1.5B',
        'source_layer': L05, 'target_layer': L15,
        'translation_cos_sim': round(cos_sim, 4),
        'translated_acc': round(translated_acc, 4),
        'native_acc': round(native_acc, 4),
        'loss_history': [round(l, 6) for l in loss_history[::10]],
        'results': results,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase39_translation.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Translation Matrix Training', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(['0.5B->1.5B\n(Translated)', '1.5B Native'],
                [translated_acc, native_acc],
                color=['tab:purple', 'tab:blue'], edgecolor='black')
    axes[1].set_ylabel('DMA Accuracy')
    axes[1].set_title(f'Cross-Arch DMA\ncos_sim={cos_sim:.3f}', fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate([translated_acc, native_acc]):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=14)

    axes[2].axis('off')
    summary = (f"Cross-Architecture Translation\n\n"
               f"0.5B L{L05} -> W -> 1.5B L{L15}\n\n"
               f"Translated acc: {translated_acc:.0%}\n"
               f"Native acc: {native_acc:.0%}\n"
               f"Cosine sim: {cos_sim:.4f}\n\n"
               f"'Write once, run anywhere'\n"
               f"Neural Rosetta 2")
    axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                 fontsize=12, va='center', ha='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
    plt.suptitle('Phase 39: Cross-Architecture Binary Translation\n0.5B program -> translation matrix -> 1.5B execution',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase39_translation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model_15; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
