# -*- coding: utf-8 -*-
"""
Phase 98: Introspection (Self-Modeling)
Can NeuOS predict its OWN output? Train a meta-program at L16 that,
given the main program's L8 vector, predicts what the model will output.
This is genuine self-awareness: a system that models itself.

"Know thyself."

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
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compile_prog(model, tok, train, layer, device, seed=42, epochs=100):
    hs = model.config.hidden_size
    torch.manual_seed(seed)
    vec = torch.randn(hs, device=device)*0.01; vec.requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=0.01)
    for ep in range(epochs):
        for p, t in train:
            tid = tok.encode(t)[-1]; inp = tok(p, return_tensors='pt').to(device)
            def inj(m,i,o,v=vec): return replace_last_token(o,v)
            h = model.model.layers[layer].register_forward_hook(inj)
            out = model(**inp); h.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return vec.detach()

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
    print("[P98] Introspection (Self-Modeling)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    min_data = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                ("4, 6) =","4"),("9, 3) =","3")]
    max_data = [("3, 7) =","7"),("5, 2) =","5"),("8, 1) =","8"),
                ("4, 6) =","6"),("9, 3) =","9")]
    test_data = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                 ("5, 4) =","4"),("3, 8) =","3")]
    all_min = min_data + test_data

    # Step 1: Train multiple programs (the "repertoire")
    print("  Step 1: Building program repertoire...")
    programs = {}
    for seed in range(8):
        v = compile_prog(model, tok, min_data, tl, DEVICE, seed=seed*100, epochs=80)
        acc = evaluate_vec(model, tok, v, all_min, tl, DEVICE)
        programs[f'MIN_s{seed*100}'] = {'vec': v, 'acc': round(float(acc), 4)}
        print(f"    {f'MIN_s{seed*100}'}: acc={acc:.0%}")

    # Also train MAX programs
    test_max = [("7, 2) =","7"),("6, 3) =","6"),("2, 9) =","9"),
                ("5, 4) =","5"),("3, 8) =","8")]
    all_max = max_data + test_max
    for seed in range(4):
        v = compile_prog(model, tok, max_data, tl, DEVICE, seed=seed*100, epochs=80)
        acc = evaluate_vec(model, tok, v, all_max, tl, DEVICE)
        programs[f'MAX_s{seed*100}'] = {'vec': v, 'acc': round(float(acc), 4)}
        print(f"    {f'MAX_s{seed*100}'}: acc={acc:.0%}")

    # Step 2: Train a meta-classifier at L16
    # Input: L8 program vector -> Meta at L16 predicts what TYPE of program it is
    # "1" = MIN program, "2" = MAX program
    print("\n  Step 2: Training meta-classifier (introspection)...")

    torch.manual_seed(999)
    meta_vec = torch.randn(hs, device=DEVICE) * 0.01
    meta_vec.requires_grad_(True)
    meta_opt = torch.optim.Adam([meta_vec], lr=0.01)

    # Build training data for meta-classifier
    meta_train = []
    for name, prog in programs.items():
        label = "1" if 'MIN' in name else "2"
        meta_train.append((prog['vec'], label))

    # Train meta-vector: given program at L8 + meta at L16 -> predict program type
    prompt = "Program type:"  # neutral prompt
    for ep in range(200):
        np.random.shuffle(meta_train)
        total_loss = 0
        for prog_vec, label in meta_train:
            tid = tok.encode(label)[-1]
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inj_prog(m,i,o,v=prog_vec): return replace_last_token(o,v)
            def inj_meta(m,i,o,v=meta_vec): return replace_last_token(o,v)
            h1 = model.model.layers[8].register_forward_hook(inj_prog)
            h2 = model.model.layers[16].register_forward_hook(inj_meta)
            out = model(**inp); h1.remove(); h2.remove()
            loss = torch.nn.functional.cross_entropy(
                out.logits[0,-1,:].unsqueeze(0), torch.tensor([tid]).to(DEVICE))
            meta_opt.zero_grad(); loss.backward(); meta_opt.step()
            total_loss += loss.item()
        if (ep+1) % 50 == 0:
            print(f"    ep={ep+1}: avg_loss={total_loss/len(meta_train):.3f}")

    meta_vec_final = meta_vec.detach()

    # Step 3: Test meta-classifier
    print("\n  Step 3: Testing introspection accuracy...")
    introspection_results = []
    for name, prog in programs.items():
        expected = "1" if 'MIN' in name else "2"
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        def inj_p(m,i,o,v=prog['vec']): return replace_last_token(o,v)
        def inj_m(m,i,o,v=meta_vec_final): return replace_last_token(o,v)
        h1 = model.model.layers[8].register_forward_hook(inj_p)
        h2 = model.model.layers[16].register_forward_hook(inj_m)
        with torch.no_grad(): out = model(**inp)
        h1.remove(); h2.remove()
        pred = tok.decode(out.logits[0,-1,:].argmax().item()).strip()
        correct = pred == expected
        introspection_results.append({
            'program': name, 'expected': expected,
            'predicted': pred, 'correct': bool(correct)
        })
        print(f"    {name}: expected={expected}, predicted={pred}, "
              f"{'OK' if correct else 'WRONG'}")

    accuracy = sum(1 for r in introspection_results if r['correct']) / len(introspection_results)
    print(f"\n  Introspection accuracy: {accuracy:.0%}")

    # Step 4: Test with novel programs (never seen during meta-training)
    print("\n  Step 4: Generalization to novel programs...")
    novel_results = []
    for seed in [500, 600, 700]:
        for task_name, task_train, task_test, label in [
            ('MIN', min_data, all_min, '1'), ('MAX', max_data, all_max, '2')]:
            v = compile_prog(model, tok, task_train, tl, DEVICE, seed=seed, epochs=80)
            inp = tok(prompt, return_tensors='pt').to(DEVICE)
            def inj_p2(m,i,o,v=v): return replace_last_token(o,v)
            def inj_m2(m,i,o,v=meta_vec_final): return replace_last_token(o,v)
            h1 = model.model.layers[8].register_forward_hook(inj_p2)
            h2 = model.model.layers[16].register_forward_hook(inj_m2)
            with torch.no_grad(): out = model(**inp)
            h1.remove(); h2.remove()
            pred = tok.decode(out.logits[0,-1,:].argmax().item()).strip()
            correct = pred == label
            novel_results.append({
                'task': f'{task_name}_s{seed}', 'expected': label,
                'predicted': pred, 'correct': bool(correct)
            })
            print(f"    {task_name}_s{seed}: expected={label}, predicted={pred}, "
                  f"{'OK' if correct else 'WRONG'}")
    novel_acc = sum(1 for r in novel_results if r['correct']) / len(novel_results)
    print(f"\n  Novel program introspection: {novel_acc:.0%}")

    # Save
    output = {
        'phase': 98, 'name': 'introspection',
        'train_accuracy': round(float(accuracy), 4),
        'novel_accuracy': round(float(novel_acc), 4),
        'introspection_results': introspection_results,
        'novel_results': novel_results,
        'num_programs_trained': len(programs),
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase98_introspection.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Introspection results
    names = [r['program'] for r in introspection_results]
    colors = ['tab:green' if r['correct'] else 'tab:red' for r in introspection_results]
    axes[0].bar(range(len(names)), [1 if r['correct'] else 0 for r in introspection_results],
                color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=6, rotation=45, ha='right')
    axes[0].set_ylabel('Correct?')
    axes[0].set_title(f'Training Set Introspection\n({accuracy:.0%} accuracy)',
                      fontweight='bold')

    # Novel programs
    novel_names = [r['task'] for r in novel_results]
    novel_colors = ['tab:green' if r['correct'] else 'tab:red' for r in novel_results]
    axes[1].bar(range(len(novel_names)),
                [1 if r['correct'] else 0 for r in novel_results],
                color=novel_colors, edgecolor='black')
    axes[1].set_xticks(range(len(novel_names)))
    axes[1].set_xticklabels(novel_names, fontsize=7, rotation=45, ha='right')
    axes[1].set_ylabel('Correct?')
    axes[1].set_title(f'Novel Program Introspection\n({novel_acc:.0%} generalization)',
                      fontweight='bold')

    # Summary
    axes[2].bar(['Training\nPrograms', 'Novel\nPrograms'],
                [accuracy, novel_acc],
                color=['tab:blue', 'tab:purple'], edgecolor='black')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Self-Modeling Accuracy', fontweight='bold')
    axes[2].set_ylim(0, 1.2)
    for i, v in enumerate([accuracy, novel_acc]):
        axes[2].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=14)

    plt.suptitle('Phase 98: Introspection\n"Know thyself"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase98_introspection.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
