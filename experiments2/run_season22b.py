# -*- coding: utf-8 -*-
"""
Season 22 Part 2 Runner: P150-P152
Reads C:\\tmp\\experiment_control.csv for completion action.
"""
import gc, torch, time, os, sys, csv

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_DIR)
CONTROL_CSV = r'C:\tmp\experiment_control.csv'


def read_control():
    try:
        with open(CONTROL_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row.get('action_on_complete', 'beep').strip().lower()
    except Exception:
        return 'beep'


def run_phase(module_name, phase_name):
    print('\n' + '=' * 60)
    print('  Starting %s...' % phase_name)
    print('=' * 60)
    start = time.time()
    try:
        mod = __import__(module_name)
        mod.main()
    except Exception as e:
        print('  ERROR in %s: %s' % (phase_name, e))
        import traceback
        traceback.print_exc()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('  %s completed in %.0fs' % (phase_name, time.time() - start))


def do_completion():
    action = read_control()
    print('\nCompletion action: %s' % action)
    if action == 'hibernate':
        print('Hibernating...')
        os.system('shutdown /h')
    else:
        try:
            import winsound
            for _ in range(5):
                winsound.Beep(1000, 500)
                time.sleep(0.3)
        except Exception:
            pass
        print('Done! Beep notification sent.')


def main():
    print('NeuOS Season 22 Part 2: Self-Healing White-Box OS')
    print('Start: %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    overall_start = time.time()

    phases = [
        ('phase150_adaptive_hypervisor', 'P150: Adaptive Hypervisor'),
        ('phase151_glassbox_dashboard', 'P151: GlassBox Dashboard'),
        ('phase152_soul_immune', 'P152: Soul Immune System'),
    ]

    try:
        for module, name in phases:
            run_phase(module, name)

        elapsed = time.time() - overall_start
        print('\nAll S22-Part2 experiments completed in %.0fs (%.1f min)' % (
            elapsed, elapsed / 60))
    finally:
        do_completion()


if __name__ == '__main__':
    main()
