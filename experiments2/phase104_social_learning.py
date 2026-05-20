# -*- coding: utf-8 -*-
"""
Phase 104: Social Learning (Imitation)
Instead of training from data, can NeuOS learn by "watching" another soul?
Capture the activation trace of a skilled soul, then use that trace to
bootstrap a new soul (distillation without access to training data).

"I learned by watching you."

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

def get_teacher_traces(model, tok, teacher_vec, prompts, layer, device):
    """Record teacher's output logits for each prompt (the 'demonstration')."""
    traces = []
    for prompt in prompts:
        def inj(m,i,o,v=teacher_vec): return replace_last_token(o,v)
        h = model.model.layers[layer].register_forward_hook(inj)
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**inp)
        h.remove()
        traces.append({
            'prompt': prompt,
            'logits': out.logits[0,-1,:].detach(),  # teacher's output distribution
        })
    return traces

def main():
    print("[P104] Social Learning (Imitation)")
    start = time.time()
    model, tok = load_model(device=DEVICE, surgery=True)
    hs = model.config.hidden_size; tl = 8
    for p in model.parameters(): p.requires_grad = False

    min_train = [("3, 7) =","3"),("5, 2) =","2"),("8, 1) =","1"),
                 ("4, 6) =","4"),("9, 3) =","3")]
    min_test = [("7, 2) =","2"),("6, 3) =","3"),("2, 9) =","2"),
                ("5, 4) =","4"),("3, 8) =","3")]
    all_min = min_train + min_test

    # Step 1: Train a skilled teacher
    print("  Step 1: Training teacher soul...")
    teacher = compile_prog(model, tok, min_train, tl, DEVICE, seed=42, epochs=100)
    teacher_acc = evaluate_vec(model, tok, teacher, all_min, tl, DEVICE)
    print(f"    Teacher accuracy: {teacher_acc:.0%}")

    # Step 2: Collect teacher demonstrations (NO access to labels!)
    print("\n  Step 2: Collecting teacher demonstrations...")
    demo_prompts = [p for p, _ in min_train] + [p for p, _ in min_test]
    # Also add novel prompts the student hasn't seen
    novel_prompts = ["1, 5) =", "4, 8) =", "6, 2) =", "9, 7) =", "3, 3) ="]
    all_prompts = demo_prompts + novel_prompts
    traces = get_teacher_traces(model, tok, teacher, all_prompts, tl, DEVICE)
    print(f"    Collected {len(traces)} demonstrations")

    # Step 3: Train student by imitating teacher's OUTPUT DISTRIBUTION
    print("\n  Step 3: Training student via imitation...")
    torch.manual_seed(777)
    student = torch.randn(hs, device=DEVICE)*0.01; student.requires_grad_(True)
    opt = torch.optim.Adam([student], lr=0.01)

    imitation_history = []
    for ep in range(150):
        total_loss = 0
        for trace in traces:
            inp = tok(trace['prompt'], return_tensors='pt').to(DEVICE)
            def inj(m,i,o,v=student): return replace_last_token(o,v)
            h = model.model.layers[tl].register_forward_hook(inj)
            out = model(**inp); h.remove()
            # KL divergence between student and teacher distributions
            student_logp = torch.nn.functional.log_softmax(out.logits[0,-1,:], dim=-1)
            teacher_p = torch.nn.functional.softmax(trace['logits'], dim=-1)
            loss = torch.nn.functional.kl_div(student_logp, teacher_p,
                                              reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        if (ep+1) % 30 == 0:
            acc = evaluate_vec(model, tok, student.detach(), all_min, tl, DEVICE)
            imitation_history.append({
                'epoch': ep+1,
                'loss': round(total_loss/len(traces), 4),
                'accuracy': round(float(acc), 4),
            })
            print(f"    ep={ep+1}: loss={total_loss/len(traces):.4f}, acc={acc:.0%}")

    student_final = student.detach()

    # Step 4: Compare with direct training
    print("\n  Step 4: Comparison baselines...")
    # Baseline: train from scratch with same seed
    baseline = compile_prog(model, tok, min_train, tl, DEVICE, seed=777, epochs=100)
    baseline_acc = evaluate_vec(model, tok, baseline, all_min, tl, DEVICE)

    # Student with fewer epochs
    student_acc = evaluate_vec(model, tok, student_final, all_min, tl, DEVICE)

    # Random baseline
    random_vec = torch.randn(hs, device=DEVICE)*0.01
    random_acc = evaluate_vec(model, tok, random_vec, all_min, tl, DEVICE)

    print(f"    Teacher: {teacher_acc:.0%}")
    print(f"    Student (imitation): {student_acc:.0%}")
    print(f"    Baseline (direct train): {baseline_acc:.0%}")
    print(f"    Random: {random_acc:.0%}")

    # Cosine similarity
    cos_st = float(torch.nn.functional.cosine_similarity(
        student_final.unsqueeze(0), teacher.unsqueeze(0)).item())
    cos_bt = float(torch.nn.functional.cosine_similarity(
        baseline.unsqueeze(0), teacher.unsqueeze(0)).item())
    print(f"    Cosine(student, teacher): {cos_st:.4f}")
    print(f"    Cosine(baseline, teacher): {cos_bt:.4f}")

    # Step 5: Few-shot imitation (how many demos needed?)
    print("\n  Step 5: Demo count ablation...")
    demo_ablation = []
    for n_demos in [1, 3, 5, 10, 15]:
        torch.manual_seed(777)
        sv = torch.randn(hs, device=DEVICE)*0.01; sv.requires_grad_(True)
        opt = torch.optim.Adam([sv], lr=0.01)
        subset = traces[:n_demos]
        for ep in range(150):
            for trace in subset:
                inp = tok(trace['prompt'], return_tensors='pt').to(DEVICE)
                def inj(m,i,o,v=sv): return replace_last_token(o,v)
                hk = model.model.layers[tl].register_forward_hook(inj)
                out = model(**inp); hk.remove()
                student_logp = torch.nn.functional.log_softmax(
                    out.logits[0,-1,:], dim=-1)
                teacher_p = torch.nn.functional.softmax(trace['logits'], dim=-1)
                loss = torch.nn.functional.kl_div(student_logp, teacher_p,
                                                  reduction='batchmean')
                opt.zero_grad(); loss.backward(); opt.step()
        acc = evaluate_vec(model, tok, sv.detach(), all_min, tl, DEVICE)
        demo_ablation.append({'n_demos': n_demos, 'accuracy': round(float(acc), 4)})
        print(f"    {n_demos} demos: {acc:.0%}")

    # Save
    output = {
        'phase': 104, 'name': 'social_learning',
        'teacher_accuracy': round(float(teacher_acc), 4),
        'student_accuracy': round(float(student_acc), 4),
        'baseline_accuracy': round(float(baseline_acc), 4),
        'random_accuracy': round(float(random_acc), 4),
        'cos_student_teacher': round(cos_st, 4),
        'cos_baseline_teacher': round(cos_bt, 4),
        'demo_ablation': demo_ablation,
        'imitation_history': imitation_history,
        'elapsed': round(time.time()-start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase104_social_learning.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Learning curve
    eps = [h['epoch'] for h in imitation_history]
    axes[0].plot(eps, [h['accuracy'] for h in imitation_history], 'g-o', lw=2,
                 label='Student (imitation)')
    axes[0].axhline(y=teacher_acc, color='blue', ls='--', lw=2, label='Teacher')
    axes[0].axhline(y=baseline_acc, color='gray', ls='--', lw=1.5,
                     label='Direct training')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Imitation Learning Curve', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Comparison
    methods = ['Teacher', 'Student\n(imit.)', 'Direct\nTrain', 'Random']
    vals = [teacher_acc, student_acc, baseline_acc, random_acc]
    colors = ['tab:blue', 'tab:green', 'tab:gray', 'tab:red']
    axes[1].bar(methods, vals, color=colors, edgecolor='black')
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1.2)
    axes[1].set_title('Knowledge Transfer', fontweight='bold')
    for i, v in enumerate(vals):
        axes[1].text(i, v+0.03, f'{v:.0%}', ha='center', fontweight='bold')

    # Demo ablation
    ns = [d['n_demos'] for d in demo_ablation]
    accs = [d['accuracy'] for d in demo_ablation]
    axes[2].plot(ns, accs, 'b-o', lw=2, ms=8)
    axes[2].axhline(y=teacher_acc, color='blue', ls='--', alpha=0.5,
                     label='Teacher')
    axes[2].set_xlabel('Number of Demonstrations')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Few-Shot Imitation', fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 104: Social Learning\n"I learned by watching you"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase104_social_learning.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Completed in {time.time()-start:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
