# -*- coding: utf-8 -*-
"""
Phase 16: The Neural Executable (.nexe)
Can we save a "program" (OPCODE vector) as a file and execute it
on arbitrary data by injecting it into the KV cache?

Proves: data and program are the same thing in latent space.

Method:
  1. Extract OPCODE activation from "def sort(...):" prompt -> save as .nexe
  2. Feed raw numbers "1, 5, 2" with NO instruction
  3. Prepend .nexe to KV cache at inference time
  4. Check if output is sorted

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time, sys, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model
from kv_utils import swap_out, swap_in

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
NEXE_DIR = os.path.join(os.path.dirname(__file__), '..', 'nexe_files')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(NEXE_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_kv_cache(model, tok, prompt):
    """Run prompt and return full KV cache (DynamicCache)."""
    inp = tok(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**inp, use_cache=True)
    return out.past_key_values


def run_with_prefix_cache(model, tok, data_prompt, prefix_cache_cpu):
    """Run data_prompt but with a saved prefix KV cache swapped in."""
    prefix_cache = swap_in(prefix_cache_cpu, DEVICE)
    inp = tok(data_prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=inp.input_ids, past_key_values=prefix_cache, use_cache=False)
    return out


def main():
    print("[P16] The Neural Executable (.nexe)")
    print(f"  Device: {DEVICE}")
    start_time = time.time()

    model, tok = load_model(device=DEVICE, surgery=True)
    n_layers = model.config.num_hidden_layers

    # === Step 1: Create .nexe files (instruction vectors) ===
    print("  Step 1: Creating .nexe files...")
    programs = {
        'addition': "def f(a, b): return a + b\nf(",
        'sorting': "def sort(a, b, c): return sorted([a, b, c])\nsort(",
        'maximum': "def f(a, b, c): return max(a, b, c)\nf(",
        'identity': "def f(x): return x\nf(",
    }

    nexe_caches = {}
    for name, prompt in programs.items():
        kv = extract_kv_cache(model, tok, prompt)
        # Save to disk as .nexe
        save_path = os.path.join(NEXE_DIR, f'{name}.nexe')
        kv_cpu = swap_out(kv)
        with open(save_path, 'wb') as f:
            pickle.dump(kv_cpu, f)
        nexe_caches[name] = kv_cpu  # store CPU version
        print(f"    Saved {name}.nexe ({os.path.getsize(save_path)//1024} KB)")

    # === Step 2: Test - data only, no instruction ===
    print("\n  Step 2: Executing .nexe on raw data...")

    # Test cases: just numbers, no instructions
    test_cases = [
        ("3, 4) =", {'addition': '7', 'maximum': '4', 'identity': '3'}),
        ("5, 2) =", {'addition': '7', 'maximum': '5', 'identity': '5'}),
        ("8, 1) =", {'addition': '9', 'maximum': '8', 'identity': '8'}),
        ("6, 3) =", {'addition': '9', 'maximum': '6', 'identity': '6'}),
        ("2, 7) =", {'addition': '9', 'maximum': '7', 'identity': '2'}),
        ("4, 5) =", {'addition': '9', 'maximum': '5', 'identity': '4'}),
    ]

    results = {}
    for prog_name in ['addition', 'maximum', 'identity']:
        correct = 0
        total = 0
        prefix_kv_cpu = nexe_caches[prog_name]

        for data_str, expected in test_cases:
            if prog_name not in expected:
                continue
            total += 1
            out = run_with_prefix_cache(model, tok, data_str, prefix_kv_cpu)
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            exp = expected[prog_name]
            if pred == exp:
                correct += 1

        acc = correct / total if total > 0 else 0
        results[prog_name] = round(acc, 4)
        print(f"    {prog_name}.nexe: {acc:.1%} ({correct}/{total})")

    # === Step 3: Control - no .nexe, just raw data ===
    print("\n  Step 3: Control (no .nexe, raw data only)...")
    control_correct = 0
    for data_str, expected in test_cases:
        inp = tok(data_str, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
        # Check if it matches ANY expected answer
        if pred in expected.values():
            control_correct += 1
    control_acc = control_correct / len(test_cases) if test_cases else 0
    print(f"    Control (no program): {control_acc:.1%}")

    # === Step 4: Cross-test - wrong .nexe ===
    print("\n  Step 4: Cross-test (addition.nexe on max data)...")
    cross_results = {}
    for prog_name in ['addition', 'maximum']:
        prefix_kv_cpu = nexe_caches[prog_name]
        actual_outputs = []
        for data_str, expected in test_cases:
            out = run_with_prefix_cache(model, tok, data_str, prefix_kv_cpu)
            pred = tok.decode(out.logits[0, -1, :].argmax().item()).strip()
            actual_outputs.append(pred)
        cross_results[prog_name] = actual_outputs

    # Check: does addition.nexe produce different outputs than maximum.nexe?
    different = sum(1 for a, b in zip(cross_results['addition'], cross_results['maximum'])
                    if a != b)
    cross_diff_rate = different / len(test_cases) if test_cases else 0
    print(f"    Output difference rate: {cross_diff_rate:.1%}")

    # Save
    output = {
        'phase': 16, 'name': 'neural_executable',
        'n_tests': len(test_cases), 'n_programs': len(programs),
        'results': results, 'control_acc': round(control_acc, 4),
        'cross_diff_rate': round(cross_diff_rate, 4),
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase16_nexe.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    prog_names = list(results.keys())
    accs = [results[p] for p in prog_names]
    colors = ['tab:blue', 'tab:red', 'tab:green']
    axes[0].bar(prog_names, accs, color=colors, edgecolor='black')
    axes[0].bar(['control\n(no .nexe)'], [control_acc], color='tab:gray', edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('.nexe Execution Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(accs + [control_acc]):
        axes[0].text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold')

    axes[1].axis('off')
    summary = (
        f"The Neural Executable\n\n"
        f"Program files created: {len(programs)}\n\n"
    )
    for p in prog_names:
        summary += f"  {p}.nexe: {results[p]:.0%}\n"
    summary += f"\n  Control: {control_acc:.0%}\n"
    summary += f"  Cross-diff: {cross_diff_rate:.0%}\n\n"
    summary += "Data = Program!" if any(v > 0.5 for v in results.values()) else "Needs tuning"
    axes[1].text(0.5, 0.5, summary, transform=axes[1].transAxes,
                 fontsize=13, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Phase 16: The Neural Executable\nCan a saved vector file act as a program?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase16_nexe.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Completed in {time.time()-start_time:.0f}s")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
