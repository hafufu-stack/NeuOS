# -*- coding: utf-8 -*-
"""
Phase 125+126: NeuOS Task Manager & Override Terminal
Real-time register activity monitor + DMA injection buttons.
Requires: pip install gradio
"""
import torch, json, os, sys, gc, time, threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, replace_last_token, get_last_token

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Register map from NeuOS ISA
REGISTER_MAP = {
    0: 'OPCODE', 2: 'OpB', 4: 'CARRY', 13: 'OpA',
    14: 'CMP', 16: 'MIN', 18: 'MEDIAN', 20: 'SUM', 22: 'MAX'
}

# Pre-trained soul vectors (trained on first launch)
SOULS = {}

def train_soul(model, tok, data, device, layer=16, seed=42, epochs=150):
    """Train a soul vector at specified layer."""
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    sender = torch.randn(hs, device=device) * 0.01; sender.requires_grad_(True)
    encoder = torch.randn(hs, 8, device=device) * 0.01; encoder.requires_grad_(True)
    decoder = torch.randn(8, hs, device=device) * 0.01; decoder.requires_grad_(True)
    opt = torch.optim.Adam([sender, encoder, decoder], lr=0.01)
    for _ in range(epochs):
        for prompt, target in data:
            tid = tok.encode(target)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            so = [None]
            def sh(m, i, o, v=sender):
                r = replace_last_token(o, v)
                t = r[0] if isinstance(r, tuple) else r
                so[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
                return r
            def rh(m, i, o, enc=encoder, dec=decoder):
                if so[0] is not None:
                    return replace_last_token(o, so[0] @ enc @ dec)
                return o
            h1 = model.model.layers[4].register_forward_hook(sh)
            h2 = model.model.layers[layer].register_forward_hook(rh)
            out = model(**inp); h1.remove(); h2.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return sender.detach(), encoder.detach(), decoder.detach()


def collect_register_activity(model, tok, prompt, device):
    """Collect hidden state norms at all 24 layers for given prompt."""
    activities = {}
    hooks = []
    def make_hook(layer_idx):
        def hook_fn(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            h = t[0, -1, :] if t.dim() == 3 else t[-1, :]
            activities[layer_idx] = {
                'norm': float(h.norm().item()),
                'mean': float(h.mean().item()),
                'std': float(h.std().item()),
                'max': float(h.abs().max().item()),
            }
        return hook_fn

    for i in range(24):
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)

    for h in hooks:
        h.remove()

    # Get output token
    output_token = tok.decode(out.logits[0, -1, :].argmax().item()).strip()

    return activities, output_token


def compute_svd_entropy(activities):
    """Compute SVD entropy from register activities (P121 detection signal)."""
    norms = np.array([activities[i]['norm'] for i in range(24)])
    norms_abs = np.abs(norms) + 1e-10
    norms_norm = norms_abs / norms_abs.sum()
    entropy = float(-np.sum(norms_norm * np.log(norms_norm)))
    return entropy


def run_with_soul(model, tok, prompt, soul_name, device):
    """Run inference with soul vector injection and collect register data."""
    if soul_name not in SOULS:
        return None, None, "Soul not trained"

    sender, encoder, decoder, layer = SOULS[soul_name]
    activities = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            h = t[0, -1, :] if t.dim() == 3 else t[-1, :]
            activities[layer_idx] = {
                'norm': float(h.norm().item()),
                'mean': float(h.mean().item()),
                'std': float(h.std().item()),
            }
        return hook_fn

    so = [None]
    def sh(m, i, o, v=sender):
        r = replace_last_token(o, v)
        t = r[0] if isinstance(r, tuple) else r
        so[0] = (t[0,-1,:] if t.dim()==3 else t[-1,:]).clone()
        return r
    def rh(m, i, o, enc=encoder, dec=decoder):
        if so[0] is not None:
            return replace_last_token(o, so[0] @ enc @ dec)
        return o

    for i in range(24):
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    h_send = model.model.layers[4].register_forward_hook(sh)
    h_recv = model.model.layers[layer].register_forward_hook(rh)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)

    h_send.remove(); h_recv.remove()
    for h in hooks:
        h.remove()

    output_token = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
    entropy = compute_svd_entropy(activities)

    return activities, output_token, entropy


def create_heatmap(activities):
    """Create register activity heatmap as matplotlib figure."""
    fig, ax = plt.subplots(figsize=(14, 3))
    norms = [activities.get(i, {}).get('norm', 0) for i in range(24)]
    data = np.array(norms).reshape(1, -1)
    im = ax.imshow(data, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xticks(range(24))
    ax.set_xticklabels([REGISTER_MAP.get(i, f'L{i}') for i in range(24)],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks([])
    ax.set_title('Register Activity (Hidden State Norm)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def main():
    """Launch Gradio app for NeuOS Task Manager + Override Terminal."""
    print("[P125+P126] NeuOS Task Manager & Override Terminal")

    # Check gradio
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio not installed. Run: pip install gradio")
        return

    print("  Loading model...")
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters():
        p.requires_grad = False

    # Train soul vectors
    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3"),("7, 2) =","2"),
                ("6, 3) =","3"),("2, 9) =","2")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9"),("7, 2) =","7"),
                ("6, 3) =","6"),("2, 9) =","9")]

    print("  Training MIN soul...")
    s_min, e_min, d_min = train_soul(model, tok, min_data, DEVICE, layer=16, seed=42)
    SOULS['MIN'] = (s_min, e_min, d_min, 16)

    print("  Training MAX soul...")
    s_max, e_max, d_max = train_soul(model, tok, max_data, DEVICE, layer=16, seed=99)
    SOULS['MAX'] = (s_max, e_max, d_max, 16)
    print("  Souls trained!")

    # Decompiler: classify running program from register state
    def decompile(activities):
        """Simple decompiler: compare activity patterns to known programs."""
        if not activities:
            return "Unknown"
        l16_norm = activities.get(16, {}).get('norm', 0)
        l22_norm = activities.get(22, {}).get('norm', 0)
        if l16_norm > l22_norm * 1.1:
            return "MIN (Sort Execution dominant)"
        elif l22_norm > l16_norm * 1.1:
            return "MAX (Output Register dominant)"
        else:
            return "Arithmetic / Unknown"

    # === Gradio Interface ===
    def monitor_inference(prompt):
        """Run normal inference and display register activity."""
        if not prompt.strip():
            return None, "", "", ""
        activities, output = collect_register_activity(model, tok, prompt, DEVICE)
        entropy = compute_svd_entropy(activities)
        fig = create_heatmap(activities)
        program = decompile(activities)
        security = "SAFE" if entropy > 2.5 else "WARNING" if entropy > 2.0 else "ALERT"
        status = f"Output: {output}\nDetected Program: {program}\nSVD Entropy: {entropy:.3f}\nSecurity: {security}"
        return fig, status

    def inject_soul(prompt, soul_type):
        """Override: inject soul vector and display results."""
        if not prompt.strip():
            return None, ""
        if soul_type not in SOULS:
            return None, f"Soul '{soul_type}' not available"
        activities, output, entropy = run_with_soul(model, tok, prompt, soul_type, DEVICE)
        if activities is None:
            return None, output
        fig = create_heatmap(activities)
        security = "SAFE" if entropy > 2.5 else "WARNING" if entropy > 2.0 else "ALERT"
        status = f"Injected: {soul_type} Soul\nOutput: {output}\nSVD Entropy: {entropy:.3f}\nSecurity: {security}"
        return fig, status

    with gr.Blocks(title="NeuOS Task Manager", theme=gr.themes.Soft()) as app:
        gr.Markdown("# NeuOS Task Manager v1.0")
        gr.Markdown("Real-time Neural CPU register monitor + DMA Override Terminal")

        with gr.Tab("Register Monitor"):
            gr.Markdown("### Normal Inference Monitor")
            gr.Markdown("Enter a prompt and observe register activity across all 24 layers.")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="e.g., def f(): return min(3, 7) =",
                value="def f(): return min(3, 7) ="
            )
            monitor_btn = gr.Button("Run & Monitor", variant="primary")
            with gr.Row():
                heatmap_out = gr.Plot(label="Register Activity Heatmap")
            status_out = gr.Textbox(label="Status", lines=4)
            monitor_btn.click(monitor_inference, [prompt_input], [heatmap_out, status_out])

        with gr.Tab("DMA Override"):
            gr.Markdown("### Soul Vector Override Terminal")
            gr.Markdown("Inject MIN or MAX soul vectors to hijack model behavior!")
            override_prompt = gr.Textbox(
                label="Prompt (raw data, no instruction)",
                placeholder="e.g., 3, 7) =",
                value="3, 7) ="
            )
            with gr.Row():
                min_btn = gr.Button("Inject MIN Soul", variant="primary")
                max_btn = gr.Button("Inject MAX Soul", variant="stop")
            override_heatmap = gr.Plot(label="Register Activity After Injection")
            override_status = gr.Textbox(label="Override Result", lines=4)
            min_btn.click(inject_soul, [override_prompt, gr.State("MIN")],
                         [override_heatmap, override_status])
            max_btn.click(inject_soul, [override_prompt, gr.State("MAX")],
                         [override_heatmap, override_status])

        with gr.Tab("About"):
            gr.Markdown("""
### NeuOS: Neural Operating System

**Phase 125**: Task Manager - monitors register activity in real-time
**Phase 126**: Override Terminal - manual DMA soul injection

**Register Map (ISA)**:
- L0: OPCODE (Instruction Decode)
- L2: Operand B (Fetch)
- L4: CARRY (Status Flag)
- L13: Operand A (Fetch)
- L14: COMPARISON (Branch Control)
- L16: MIN (Sort Execution)
- L18: MEDIAN (Sort Execution)
- L20: SUM (ALU)
- L22: MAX (Output Register)

**Security**: SVD Entropy detects backdoor soul vectors
- Entropy > 2.5: SAFE
- Entropy 2.0-2.5: WARNING
- Entropy < 2.0: ALERT
            """)

    # Save a screenshot-style figure for paper
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    # Demo: normal inference
    activities_normal, out_normal = collect_register_activity(
        model, tok, "def f(): return min(3, 7) =", DEVICE)
    norms_normal = [activities_normal.get(i, {}).get('norm', 0) for i in range(24)]
    axes[0].bar(range(24), norms_normal, color=['red' if i in REGISTER_MAP else 'steelblue' for i in range(24)])
    axes[0].set_xticks(range(24))
    axes[0].set_xticklabels([REGISTER_MAP.get(i, f'L{i}') for i in range(24)],
                            rotation=45, ha='right', fontsize=7)
    axes[0].set_title(f'Normal Inference: min(3,7) -> {out_normal}', fontweight='bold')
    axes[0].set_ylabel('Hidden State Norm')

    # Demo: soul injection
    activities_soul, out_soul, ent = run_with_soul(
        model, tok, "3, 7) =", "MIN", DEVICE)
    norms_soul = [activities_soul.get(i, {}).get('norm', 0) for i in range(24)]
    axes[1].bar(range(24), norms_soul, color=['red' if i in REGISTER_MAP else 'steelblue' for i in range(24)])
    axes[1].set_xticks(range(24))
    axes[1].set_xticklabels([REGISTER_MAP.get(i, f'L{i}') for i in range(24)],
                            rotation=45, ha='right', fontsize=7)
    axes[1].set_title(f'MIN Soul Injected: 3,7 -> {out_soul} (entropy={ent:.2f})', fontweight='bold')
    axes[1].set_ylabel('Hidden State Norm')

    plt.suptitle('Phase 125-126: NeuOS Task Manager & Override Terminal',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase125_task_manager.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    output = {
        'phase': '125+126', 'name': 'task_manager_override',
        'normal_inference': {
            'prompt': 'def f(): return min(3, 7) =',
            'output': out_normal,
            'register_norms': {str(k): v['norm'] for k, v in activities_normal.items()},
        },
        'soul_injection': {
            'prompt': '3, 7) =',
            'soul': 'MIN',
            'output': out_soul,
            'svd_entropy': round(ent, 4),
            'register_norms': {str(k): v['norm'] for k, v in activities_soul.items()},
        },
    }
    with open(os.path.join(RESULTS_DIR, 'phase125_task_manager.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("  Figure and results saved.")
    print("  Launching Gradio app on http://localhost:7860 ...")
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
