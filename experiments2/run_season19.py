# -*- coding: utf-8 -*-
"""
Season 19-21 Runner: Sequential execution of P131-P137
Reads C:\\tmp\\experiment_control.csv for completion action (beep or hibernate)
"""
import sys, os, time, gc, csv

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_DIR)

PHASES = [
    ('P131', 'Holographic Soul', 'phase131_holographic'),
    ('P132', 'Stem Cell Soul', 'phase132_stem_cell'),
    ('P133', 'Superposition Computing', 'phase133_superposition'),
    ('P134', 'Cross-Layer Dual Firmware', 'phase134_dual_firmware'),
    ('P135', 'Latent Soul Verifier', 'phase135_verifier'),
    ('P136', 'Thermodynamic Autopoiesis', 'phase136_autopoiesis'),
    ('P137', 'Arithmetic Layer Scan', 'phase137_arithmetic_layer'),
]

CONTROL_CSV = r'C:\tmp\experiment_control.csv'


def read_completion_action():
    """Read completion action from CSV. Returns 'beep' or 'hibernate'."""
    try:
        with open(CONTROL_CSV, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            action = row[0].strip().lower()
            if action in ('beep', 'hibernate'):
                return action
    except Exception as e:
        print("  Warning: Could not read control CSV: %s" % e)
    return 'beep'  # default


def do_beep():
    """Play completion beep sound."""
    try:
        import winsound
        for _ in range(5):
            winsound.Beep(1000, 500)
            time.sleep(0.3)
        print("Beep notification sent.")
    except Exception:
        print("Beep failed (not on Windows?)")


def do_hibernate():
    """Hibernate the system."""
    print("Hibernating... (disabled for safety)")
    # os.system("shutdown /h")  # Removed for GitHub


def main():
    print("=" * 60)
    print("NeuOS Seasons 19-21 Runner")
    print("Phases: %s" % ', '.join(p[0] for p in PHASES))
    print("=" * 60)
    total_start = time.time()

    results = {}
    for phase_id, phase_name, module_name in PHASES:
        print("\n  Starting %s: %s..." % (phase_id, phase_name))
        phase_start = time.time()
        try:
            mod = __import__(module_name)
            mod.main()
            elapsed = time.time() - phase_start
            results[phase_id] = 'OK (%.0fs)' % elapsed
            print("  %s: %s completed in %.0fs" % (phase_id, phase_name, elapsed))
        except Exception as e:
            elapsed = time.time() - phase_start
            results[phase_id] = 'ERROR: %s' % str(e)[:100]
            print("  ERROR in %s: %s: %s" % (phase_id, phase_name, str(e)[:200]))

        # Cleanup between phases
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("All experiments completed in %.0fs (%.1f min)" % (
        total_elapsed, total_elapsed / 60))
    for phase_id, status in results.items():
        print("  %s: %s" % (phase_id, status))
    print("=" * 60)

    # Completion action
    action = read_completion_action()
    print("Completion action: %s" % action)
    if action == 'hibernate':
        do_hibernate()
    else:
        do_beep()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("FATAL ERROR: %s" % str(e))
        # Still try to do completion action
        action = read_completion_action()
        if action == 'hibernate':
            do_hibernate()
        else:
            do_beep()
