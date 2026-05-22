# -*- coding: utf-8 -*-
"""
Season 24 Runner: The 7D Rosetta Engine
Runs P165-P168 sequentially, then checks experiment_control.csv
for completion action (beep or hibernate).
"""
import os, sys, gc, time, csv

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_CSV = r'C:\tmp\experiment_control.csv'


def get_completion_action():
    """Read experiment_control.csv for completion action."""
    if not os.path.exists(CONTROL_CSV):
        return 'beep'
    with open(CONTROL_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('key') == 'completion_action':
                return row.get('value', 'beep')
    return 'beep'


def main():
    print("=" * 60)
    print("  Season 24: The 7D Rosetta Engine")
    print("  P165-P168 | The complete mind control system")
    print("=" * 60)
    start = time.time()

    phases = [
        ('phase165_firewall', 'P165: 7D Semantic Firewall'),
        ('phase166_rosetta', 'P166: The Rosetta Compiler'),
        ('phase167_alchemy', 'P167: Zero-Shot Skill Alchemy'),
        ('phase168_control_room', 'P168: The Control Room UI'),
    ]

    for module_name, desc in phases:
        print("\n%s" % ("=" * 50))
        print("  %s" % desc)
        print("=" * 50)
        phase_start = time.time()
        try:
            mod = __import__(module_name)
            mod.main()
            elapsed = time.time() - phase_start
            print("  %s completed in %.0fs" % (desc.split(':')[0], elapsed))
        except Exception as e:
            print("  ERROR in %s: %s" % (module_name, str(e)))
            import traceback
            traceback.print_exc()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    total = time.time() - start
    print("\n" + "=" * 60)
    print("  All Season 24 experiments completed in %.0fs (%.1f min)" % (total, total/60))

    # Completion action
    action = get_completion_action()
    print("  Completion action: %s" % action)

    if action == 'hibernate':
        print("  Hibernate requested. (disabled in public repo)")
    else:
        # Beep
        try:
            import winsound
            for _ in range(8):
                winsound.Beep(1200, 400)
                time.sleep(0.2)
        except:
            pass
        print("  Done! Beep notification sent.")


if __name__ == '__main__':
    main()
