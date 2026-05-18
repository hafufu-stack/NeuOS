# -*- coding: utf-8 -*-
"""
Phase 33: Attention Archaeology (Opus Original)
Analyze attention patterns during different operations.
Can we read the "call stack" from attention heads?

Which tokens does the model attend to at each register layer?
Does MIN computation attend to specific operand positions?

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print("[P33] Attention Archaeology (Opus Original)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='eager'
    ).to(DEVICE)
    model.eval()
    from utils import apply_surgery
    apply_surgery(model, tok)
    n_layers = model.config.num_hidden_layers

    # Run with attention output
    tasks = {
        'MIN': "def f(): return min(3, 7) =",
        'MAX': "def f(): return max(3, 7) =",
        'SUM': "def f(): return 3 + 7 =",
        'SUB': "def f(): return 7 - 3 =",
    }

    all_attentions = {}
    token_labels = {}

    for task_name, prompt in tasks.items():
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        tokens = [tok.decode(t) for t in inp.input_ids[0]]
        token_labels[task_name] = tokens

        with torch.no_grad():
            out = model(**inp, output_attentions=True)

        # out.attentions: tuple of (batch, n_heads, seq, seq) per layer
        # Extract last-token attention for each layer
        attn_per_layer = []
        for layer_idx, attn in enumerate(out.attentions):
            # Average over heads, get last token's attention
            last_token_attn = attn[0, :, -1, :].mean(dim=0).cpu().numpy()
            attn_per_layer.append(last_token_attn.tolist())
        all_attentions[task_name] = attn_per_layer

        # Find which tokens get most attention at key layers
        for reg_name, reg_layer in [('OPCODE', 0), ('OperandB', 2), ('MIN', 16), ('SUM', 20), ('MAX', 22)]:
            if reg_layer < len(out.attentions):
                attn_vals = out.attentions[reg_layer][0, :, -1, :].mean(dim=0).cpu().numpy()
                top3_idx = np.argsort(attn_vals)[-3:][::-1]
                top3 = [(tokens[i], round(float(attn_vals[i]), 3)) for i in top3_idx]
                if task_name == list(tasks.keys())[0]:
                    print(f"    {task_name} L{reg_layer}({reg_name}): "
                          f"attends to {top3}")

    # Compute attention divergence between tasks
    print("\n  Computing attention divergence...")
    key_layers = [0, 2, 4, 13, 16, 18, 20, 22]
    divergence = {}
    task_pairs = [('MIN', 'MAX'), ('MIN', 'SUM'), ('MAX', 'SUM'), ('SUM', 'SUB')]

    for t1, t2 in task_pairs:
        pair_div = []
        for layer in key_layers:
            a1 = np.array(all_attentions[t1][layer])
            a2 = np.array(all_attentions[t2][layer])
            # Pad to same length
            max_len = max(len(a1), len(a2))
            a1_pad = np.pad(a1, (0, max_len - len(a1)))
            a2_pad = np.pad(a2, (0, max_len - len(a2)))
            # Jensen-Shannon divergence (simplified as L1 distance)
            div = np.abs(a1_pad - a2_pad).sum()
            pair_div.append(round(float(div), 4))
        divergence[f'{t1}_vs_{t2}'] = dict(zip([f'L{l}' for l in key_layers], pair_div))
        print(f"    {t1} vs {t2}: max divergence at "
              f"L{key_layers[np.argmax(pair_div)]} ({max(pair_div):.3f})")

    # Save
    output = {
        'phase': 33, 'name': 'attention_archaeology',
        'tasks': list(tasks.keys()),
        'divergence': divergence,
        'token_labels': {k: v for k, v in token_labels.items()},
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase33_attention.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot: attention heatmaps for MIN task
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    min_tokens = token_labels['MIN']
    min_attns = all_attentions['MIN']

    for idx, (layer, ax) in enumerate(zip([0, 2, 13, 16, 20, 22], axes.flat)):
        attn = np.array(min_attns[layer])
        ax.barh(range(len(min_tokens)), attn, color='tab:blue', edgecolor='none')
        ax.set_yticks(range(len(min_tokens)))
        ax.set_yticklabels(min_tokens, fontsize=7)
        ax.set_xlabel('Attention Weight')
        ax.set_title(f'L{layer} (last token attends to...)', fontsize=10, fontweight='bold')
        ax.invert_yaxis()

    plt.suptitle('Phase 33: Attention Archaeology\nWhat does the last token "look at" during MIN(3,7)?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase33_attention.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Divergence
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    for pair_name, pair_vals in divergence.items():
        layers_plot = list(pair_vals.keys())
        vals = list(pair_vals.values())
        ax2.plot(layers_plot, vals, 'o-', linewidth=2, markersize=5, label=pair_name)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Attention Divergence (L1)')
    ax2.set_title('Phase 33: When do attention patterns diverge between tasks?',
                  fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase33_divergence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
