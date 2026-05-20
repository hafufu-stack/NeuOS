# -*- coding: utf-8 -*-
"""
Phase 117: Memory Palace (Multi-Step Sequential Memory)
How many sequential observations can a stateful soul remember?

"Working memory has a capacity -- what is the soul's?"
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

def gen_sequences(n_steps, n_train=8, n_test=4, seed=42):
    """Generate N-step MIN sequences."""
    rng = np.random.RandomState(seed)
    letters = 'ABCDEFGHIJ'
    seqs = []
    seen = set()
    for _ in range(n_train + n_test + 50):  # oversample to get unique
        nums = rng.randint(1, 10, size=n_steps).tolist()
        key = tuple(nums)
        if key in seen: continue
        seen.add(key)
        prompts = []
        for i in range(n_steps - 1):
            prompts.append(f"{letters[i]}={nums[i]}")
        prompts.append(f"{letters[n_steps-1]}={nums[n_steps-1]},min=")
        seqs.append((prompts, str(min(nums))))
        if len(seqs) >= n_train + n_test: break
    return seqs[:n_train], seqs[n_train:n_train+n_test]

def train_memory(model, tok, train_seqs, n_steps, state_dim, device, epochs=200):
    hs = model.config.hidden_size
    torch.manual_seed(42)
    soul = torch.randn(hs, device=device)*0.01; soul.requires_grad_(True)
    st_enc = torch.randn(hs, state_dim, device=device)*0.01; st_enc.requires_grad_(True)
    st_dec = torch.randn(state_dim, hs, device=device)*0.01; st_dec.requires_grad_(True)
    # GRU-like update gate
    W_z = torch.randn(state_dim*2, state_dim, device=device)*0.01; W_z.requires_grad_(True)
    W_h = torch.randn(state_dim*2, state_dim, device=device)*0.01; W_h.requires_grad_(True)

    opt = torch.optim.Adam([soul, st_enc, st_dec, W_z, W_h], lr=0.01)

    for ep in range(epochs):
        total_loss = 0
        for prompts, target in train_seqs:
            tid = tok.encode(target)[-1]
            state = torch.zeros(state_dim, device=device)

            # Process intermediate steps
            for step_prompt in prompts[:-1]:
                cap = [None]
                def inj(m,i,o,v=soul): return replace_last_token(o,v)
                def cap_fn(m,i,o): cap[0] = get_last_token(o); return o
                h1 = model.model.layers[8].register_forward_hook(inj)
                h2 = model.model.layers[16].register_forward_hook(cap_fn)
                model(**tok(step_prompt, return_tensors='pt').to(device))
                h1.remove(); h2.remove()

                obs = cap[0] @ st_enc
                combined = torch.cat([state, obs])
                z = torch.sigmoid(combined @ W_z)
                h_new = torch.tanh(combined @ W_h)
                state = (1 - z) * state + z * h_new

            # Final step: use state to produce answer
            decoded = state @ st_dec
            aug = soul + decoded
            def inj_final(m,i,o,v=aug): return replace_last_token(o,v)
            h3 = model.model.layers[8].register_forward_hook(inj_final)
            out = model(**tok(prompts[-1], return_tensors='pt').to(device))
            h3.remove()

            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

    return soul.detach(), st_enc.detach(), st_dec.detach(), W_z.detach(), W_h.detach()

def eval_memory(model, tok, seqs, soul, st_enc, st_dec, W_z, W_h, state_dim, device):
    correct = 0
    for prompts, target in seqs:
        state = torch.zeros(state_dim, device=device)
        for step_prompt in prompts[:-1]:
            cap = [None]
            def inj(m,i,o,v=soul): return replace_last_token(o,v)
            def cap_fn(m,i,o): cap[0] = get_last_token(o); return o
            h1 = model.model.layers[8].register_forward_hook(inj)
            h2 = model.model.layers[16].register_forward_hook(cap_fn)
            with torch.no_grad(): model(**tok(step_prompt, return_tensors='pt').to(device))
            h1.remove(); h2.remove()
            obs = cap[0] @ st_enc
            combined = torch.cat([state, obs])
            z = torch.sigmoid(combined @ W_z)
            h_new = torch.tanh(combined @ W_h)
            state = (1 - z) * state + z * h_new

        decoded = state @ st_dec
        aug = soul + decoded
        def inj_f(m,i,o,v=aug): return replace_last_token(o,v)
        h3 = model.model.layers[8].register_forward_hook(inj_f)
        with torch.no_grad(): out = model(**tok(prompts[-1], return_tensors='pt').to(device))
        h3.remove()
        if tok.decode(out.logits[0,-1,:].argmax().item()).strip() == target:
            correct += 1
    return correct / len(seqs)

def main():
    print("[P117] Memory Palace (Multi-Step Sequential Memory)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    N_STEPS = [2, 3, 4, 5]
    STATE_DIMS = [16, 32, 64, 128]
    results_grid = {}

    for ns in N_STEPS:
        for sd in STATE_DIMS:
            print(f"  N={ns}, state_dim={sd}...", end=' ')
            train_s, test_s = gen_sequences(ns, n_train=8, n_test=4, seed=ns*100+sd)
            soul, se, sd_m, wz, wh = train_memory(
                model, tok, train_s, ns, sd, DEVICE, epochs=150)
            tr_acc = eval_memory(model, tok, train_s, soul, se, sd_m, wz, wh, sd, DEVICE)
            te_acc = eval_memory(model, tok, test_s, soul, se, sd_m, wz, wh, sd, DEVICE)
            results_grid[f'{ns}_{sd}'] = {
                'train': round(float(tr_acc), 4), 'test': round(float(te_acc), 4)}
            print(f"train={tr_acc:.0%}, test={te_acc:.0%}")

    output = {
        'phase': 117, 'name': 'memory_palace',
        'n_steps': N_STEPS, 'state_dims': STATE_DIMS,
        'results': results_grid,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase117_memory.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (metric, title) in enumerate([('train','Train Accuracy'),('test','Test Accuracy')]):
        mat = np.zeros((len(N_STEPS), len(STATE_DIMS)))
        for i, ns in enumerate(N_STEPS):
            for j, sd in enumerate(STATE_DIMS):
                mat[i, j] = results_grid[f'{ns}_{sd}'][metric]
        im = axes[ax_idx].imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[ax_idx].set_xticks(range(len(STATE_DIMS)))
        axes[ax_idx].set_xticklabels(STATE_DIMS)
        axes[ax_idx].set_yticks(range(len(N_STEPS)))
        axes[ax_idx].set_yticklabels(N_STEPS)
        axes[ax_idx].set_xlabel('State Dimension')
        axes[ax_idx].set_ylabel('Memory Steps (N)')
        axes[ax_idx].set_title(title, fontweight='bold')
        for i in range(len(N_STEPS)):
            for j in range(len(STATE_DIMS)):
                axes[ax_idx].text(j, i, f'{mat[i,j]:.0%}', ha='center', va='center',
                                 fontsize=11, fontweight='bold',
                                 color='white' if mat[i,j] < 0.5 else 'black')
        plt.colorbar(im, ax=axes[ax_idx])

    plt.suptitle('Phase 117: Memory Palace\n'
                 '"Working memory has a capacity -- what is the soul\'s?"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase117_memory.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
