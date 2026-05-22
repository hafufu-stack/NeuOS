# -*- coding: utf-8 -*-
"""
Season 22 Runner: The White-Box OS (P146-P149)
Sequential execution with beep/hibernate control via C:\\tmp\\experiment_control.csv

CSV format:
  action_on_complete
  beep          <-- 'beep' (default) or 'hibernate'
"""
import gc, torch, time, os, sys, csv

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_DIR)
CONTROL_CSV = r'C:\tmp\experiment_control.csv'


def read_control():
    """Read action_on_complete from CSV. Default: beep"""
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
    """Execute completion action (beep or hibernate)."""
    action = read_control()
    print('\nCompletion action: %s' % action)
    if action == 'hibernate':
        print('Hibernating in 10 seconds...')
        time.sleep(10)
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
    print('NeuOS Season 22: The White-Box OS')
    print('Start: %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    overall_start = time.time()

    phases = [
        ('phase146_hardware_proprioception', 'P146: Hardware Proprioception'),
        ('phase147_first_person_decompile', 'P147: First-Person Decompilation'),
        ('phase148_scaling_oracle', 'P148: The Scaling Oracle'),
        ('phase149_time_space_hypervisor', 'P149: Time-Space Hypervisor'),
    ]

    try:
        for module, name in phases:
            run_phase(module, name)

        elapsed = time.time() - overall_start
        print('\nAll Season 22 experiments completed in %.0fs (%.1f min)' % (
            elapsed, elapsed / 60))
    finally:
        do_completion()


if __name__ == '__main__':
    main()
