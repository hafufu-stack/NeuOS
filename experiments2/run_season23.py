# -*- coding: utf-8 -*-
"""
Season 23 Runner: The Homoiconic Mind (P153-P157)
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
    print('NeuOS Season 23: The Homoiconic Mind')
    print('Start: %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    overall_start = time.time()

    phases = [
        ('phase153_crystallization', 'P153: Conscious Crystallization'),
        ('phase154_autopoiesis', 'P154: Rough Soul Autopoiesis'),
        ('phase155_pipeline_rewiring', 'P155: Dynamic Pipeline Rewiring'),
        ('phase156_multiverse', 'P156: Multiverse State Forking'),
        ('phase157_skill_discovery', 'P157: Emergent Skill Discovery'),
    ]

    try:
        for module, name in phases:
            run_phase(module, name)

        elapsed = time.time() - overall_start
        print('\nAll Season 23 experiments completed in %.0fs (%.1f min)' % (
            elapsed, elapsed / 60))
    finally:
        do_completion()


if __name__ == '__main__':
    main()
