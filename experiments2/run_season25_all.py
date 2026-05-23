# -*- coding: utf-8 -*-
"""
Season 25 Runner: Execute all Phase 171-183 experiments sequentially.
Beeps on completion (or crash via try/finally).
"""
import gc, time, os, sys

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_DIR)

PHASES = [
    ('phase171_soul_algebra', 'P171: 7D Soul Algebra'),
    ('phase172_scaling_theory', 'P172: Data Scaling Theory'),
    ('phase173_nonlinear_rosetta', 'P173: Nonlinear Cross-Model Rosetta'),
    ('phase174_multistep_composition', 'P174: Multi-step Composition'),
    ('phase175_svd_entropy_limits', 'P175: SVD Entropy Adversarial Limits'),
    ('phase176_temperature_transition', 'P176: Temperature Phase Transition'),
    ('phase177_task_synthesis', 'P177: 7D Task Synthesis'),
    ('phase178_multitoken_soul', 'P178: Multi-Token Soul'),
    ('phase179_soul_dynamics', 'P179: Soul Vector Dynamics'),
    ('phase180_neural_firewall', 'P180: Neural Firewall'),
    ('phase181_multimodal_isa', 'P181: Multimodal ISA Probe'),
    ('phase182_lora_soul', 'P182: LoRA-Soul Correspondence'),
    ('phase183_sae_decomposition', 'P183: SAE Feature Decomposition'),
]


def beep_notification(n=5):
    """Beep n times using winsound."""
    try:
        import winsound
        for _ in range(n):
            winsound.Beep(1000, 500)
            time.sleep(0.3)
    except Exception:
        pass


def main():
    print("=" * 60)
    print("  NeuOS Season 25: Full Experiment Suite (P171-P183)")
    print("  %d phases scheduled" % len(PHASES))
    print("=" * 60)
    total_start = time.time()
    completed = []
    failed = []

    try:
        for module_name, description in PHASES:
            print("\n" + "=" * 60)
            print("  Starting: %s" % description)
            print("=" * 60)
            phase_start = time.time()

            try:
                mod = __import__(module_name)
                mod.main()
                elapsed = time.time() - phase_start
                completed.append((description, elapsed))
                print("  [OK] %s completed in %.0fs" % (description, elapsed))
            except Exception as e:
                elapsed = time.time() - phase_start
                failed.append((description, str(e)))
                print("  [FAIL] %s failed after %.0fs: %s" % (description, elapsed, e))
                import traceback
                traceback.print_exc()

            # Cleanup between phases
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Summary
        total_elapsed = time.time() - total_start
        print("\n" + "=" * 60)
        print("  SEASON 25 COMPLETE")
        print("  Total time: %.0fs (%.1f min)" % (total_elapsed, total_elapsed / 60))
        print("  Completed: %d / %d" % (len(completed), len(PHASES)))
        if failed:
            print("  Failed:")
            for name, err in failed:
                print("    - %s: %s" % (name, err[:80]))
        print("=" * 60)

    finally:
        # Always beep, even on crash
        print("\n  Beeping notification...")
        beep_notification(5)


if __name__ == '__main__':
    main()
