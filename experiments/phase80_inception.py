# -*- coding: utf-8 -*-
"""
Phase 80: Autopoietic Inception (OS-in-OS)
Can NeuOS host another NeuOS inside its own registers?
Host NeuOS-A runs a VMM vector at L16.
Guest NeuOS-B's MIN program is loaded as DATA at L13.
Verify: host L22 output matches guest's direct execution.

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


def compile_prog(model, tok, data, layer, device, epochs=80):
    hs = model.config.hidden_size
    torch.manual_seed(42)
    vec = torch.randn(hs, device=device) * 0.01
    vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for epoch in range(epochs):
        for prompt, target_str in data:
            tid = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            def inject(m, i, o, v=vec): return replace_last_token(o, v)
            h = model.model.layers[layer].register_forward_hook(inject)
            out = model(**inp)
            h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()


def train_vmm(model, tok, guest_prog, host_layer, guest_data_layer,
              output_layer, data, device, epochs=100):
    """Train a VMM vector that executes whatever program is in guest_data_layer."""
    hs = model.config.hidden_size
    torch.manual_seed(123)
    vmm = torch.randn(hs, device=device) * 0.01
    vmm.requires_grad_(True)
    opt = torch.optim.Adam([vmm], lr=0.01)
    history = []
    for epoch in range(epochs):
        total_loss = 0
        for prompt, target_str in data:
            tid = tok.encode(target_str)[-1]
            inp = tok(prompt, return_tensors='pt').to(device)
            # Inject VMM at host_layer, guest program at guest_data_layer
            def inj_vmm(m, i, o, v=vmm): return replace_last_token(o, v)
            def inj_guest(m, i, o, v=guest_prog): return replace_last_token(o, v)
            h1 = model.model.layers[host_layer].register_forward_hook(inj_vmm)
            h2 = model.model.layers[guest_data_layer].register_forward_hook(inj_guest)
            out = model(**inp)
            h1.remove(); h2.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0, -1, :].unsqueeze(0),
                torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            history.append({'epoch': epoch, 'loss': round(total_loss/len(data), 4)})
    return vmm.detach(), history


def eval_prog(model, tok, vec, prompts, expected, layer, device):
    correct = 0
    for prompt, exp in zip(prompts, expected):
        def inject(m, i, o, v=vec): return replace_last_token(o, v)
        h = model.model.layers[layer].register_forward_hook(inject)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        if pred == exp: correct += 1
    return correct / len(prompts)


def get_output_vec(model, tok, prompt, inject_layer, inject_vec, read_layer, device):
    """Get the hidden state at read_layer when inject_vec is at inject_layer."""
    col = {}
    def inject(m, i, o, v=inject_vec): return replace_last_token(o, v)
    def read(m, i, o): col['out'] = get_last_token(o)
    h1 = model.model.layers[inject_layer].register_forward_hook(inject)
    h2 = model.model.layers[read_layer].register_forward_hook(read)
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad(): model(**inp)
    h1.remove(); h2.remove()
    return col['out']


def main():
    print("[P80] Autopoietic Inception (OS-in-OS)")
    print(f"  Device: {DEVICE}")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    for p in model.parameters(): p.requires_grad = False

    HOST_LAYER = 16    # VMM lives here
    GUEST_DATA = 13    # Guest program loaded as data here
    OUTPUT_LAYER = 22  # Read output here
    PROG_LAYER = 8     # Standard program injection layer

    # Programs to test as guests
    programs = {
        'MIN': {'data': [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                         ("4, 6) =","4"),("9, 3) =","3")],
                'test_p': ["7, 4) =","6, 2) =","9, 1) ="],
                'test_e': ["4","2","1"]},
        'MAX': {'data': [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                         ("4, 6) =","6"),("9, 3) =","9")],
                'test_p': ["7, 4) =","6, 2) =","9, 1) ="],
                'test_e': ["7","6","9"]},
    }

    all_results = {}
    guest_vecs = {}

    # Step 1: Compile guest programs
    print("\n  Step 1: Compiling guest programs...")
    for name, info in programs.items():
        vec = compile_prog(model, tok, info['data'], PROG_LAYER, DEVICE, epochs=80)
        acc = eval_prog(model, tok, vec, info['test_p'], info['test_e'], PROG_LAYER, DEVICE)
        guest_vecs[name] = vec
        print(f"    {name} direct accuracy: {acc:.0%}")
        all_results[name] = {'direct_accuracy': round(acc, 4)}

    # Step 2: Train VMM with MIN as guest
    print("\n  Step 2: Training VMM (host=L16, guest_data=L13)...")
    vmm_vec, vmm_history = train_vmm(
        model, tok, guest_vecs['MIN'], HOST_LAYER, GUEST_DATA,
        OUTPUT_LAYER, programs['MIN']['data'], DEVICE, epochs=120)

    # Step 3: Test VMM with MIN guest
    print("\n  Step 3: Testing VMM + MIN guest...")
    for prompt, exp in zip(programs['MIN']['test_p'], programs['MIN']['test_e']):
        def inj_vmm(m, i, o, v=vmm_vec): return replace_last_token(o, v)
        def inj_guest(m, i, o, v=guest_vecs['MIN']): return replace_last_token(o, v)
        h1 = model.model.layers[HOST_LAYER].register_forward_hook(inj_vmm)
        h2 = model.model.layers[GUEST_DATA].register_forward_hook(inj_guest)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): out = model(**inp)
        h1.remove(); h2.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        print(f"    VMM+MIN: {prompt} -> {pred} (expected {exp})")

    # Step 4: Hot-swap guest to MAX (without retraining VMM!)
    print("\n  Step 4: Hot-swap guest to MAX (VMM unchanged)...")
    for prompt, exp in zip(programs['MAX']['test_p'], programs['MAX']['test_e']):
        def inj_vmm(m, i, o, v=vmm_vec): return replace_last_token(o, v)
        def inj_guest(m, i, o, v=guest_vecs['MAX']): return replace_last_token(o, v)
        h1 = model.model.layers[HOST_LAYER].register_forward_hook(inj_vmm)
        h2 = model.model.layers[GUEST_DATA].register_forward_hook(inj_guest)
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad(): out = model(**inp)
        h1.remove(); h2.remove()
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        print(f"    VMM+MAX: {prompt} -> {pred} (expected {exp})")

    # Step 5: Compare output vectors (host-mediated vs direct)
    print("\n  Step 5: Output vector similarity analysis...")
    similarities = {}
    for name in programs:
        cos_sims = []
        for prompt in programs[name]['test_p']:
            # Direct execution
            direct = get_output_vec(model, tok, prompt, PROG_LAYER,
                                   guest_vecs[name], OUTPUT_LAYER, DEVICE)
            # VMM-mediated execution
            col = {}
            def inj_vmm(m, i, o, v=vmm_vec): return replace_last_token(o, v)
            def inj_guest(m, i, o, v=guest_vecs[name]): return replace_last_token(o, v)
            def read(m, i, o): col['out'] = get_last_token(o)
            h1 = model.model.layers[HOST_LAYER].register_forward_hook(inj_vmm)
            h2 = model.model.layers[GUEST_DATA].register_forward_hook(inj_guest)
            h3 = model.model.layers[OUTPUT_LAYER].register_forward_hook(read)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            with torch.no_grad(): model(**inp)
            h1.remove(); h2.remove(); h3.remove()
            mediated = col['out']
            cs = torch.nn.functional.cosine_similarity(
                direct.unsqueeze(0), mediated.unsqueeze(0)).item()
            cos_sims.append(cs)
        mean_cs = np.mean(cos_sims)
        similarities[name] = round(mean_cs, 4)
        print(f"    {name}: mean cos_sim(direct, VMM-mediated) = {mean_cs:.4f}")
        all_results[name]['vmm_cos_sim'] = round(mean_cs, 4)

    output = {
        'phase': 80, 'name': 'autopoietic_inception',
        'architecture': {
            'host_layer': HOST_LAYER, 'guest_data_layer': GUEST_DATA,
            'output_layer': OUTPUT_LAYER, 'prog_layer': PROG_LAYER,
        },
        'programs': all_results,
        'vmm_training': vmm_history,
        'guest_swap_test': 'MAX guest with MIN-trained VMM',
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase80_inception.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # VMM training loss
    epochs = [h['epoch'] for h in vmm_history]
    losses = [h['loss'] for h in vmm_history]
    axes[0].plot(epochs, losses, 'b-o', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('VMM Training Loss', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Similarity comparison
    names = list(similarities.keys())
    sims = [similarities[n] for n in names]
    axes[1].bar(names, sims, color=['tab:blue','tab:orange'], edgecolor='black')
    axes[1].set_ylabel('cos_sim(direct, VMM-mediated)')
    axes[1].set_title('Output Similarity\n(Direct vs VMM Execution)', fontweight='bold')
    axes[1].set_ylim(-0.1, 1.1)
    for i, s in enumerate(sims):
        axes[1].text(i, s+0.03, f'{s:.3f}', ha='center', fontweight='bold')

    # Architecture diagram (text)
    axes[2].axis('off')
    arch_text = (
        "NeuOS Inception Architecture\n\n"
        "Host NeuOS-A:\n"
        f"  L{HOST_LAYER}: VMM vector (trained)\n"
        f"  L{GUEST_DATA}: Guest program (data)\n"
        f"  L{OUTPUT_LAYER}: Output (read)\n\n"
        "Guest NeuOS-B:\n"
        f"  MIN/MAX program vector\n"
        f"  Loaded as DATA at L{GUEST_DATA}\n\n"
        "Test: Hot-swap guest\n"
        "  MIN -> MAX without\n"
        "  retraining VMM"
    )
    axes[2].text(0.5, 0.5, arch_text, transform=axes[2].transAxes,
                fontsize=11, ha='center', va='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Phase 80: Autopoietic Inception\nOS-in-OS: VMM hosts guest programs',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase80_inception.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
