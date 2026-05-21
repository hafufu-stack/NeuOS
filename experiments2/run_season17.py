# -*- coding: utf-8 -*-
"""
NeuOS Season 17 Runner: Hafufu Universe Crossovers
Sequential execution of P122, P123, P124, P127.
"""
import gc, torch, time, os, sys, csv

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('  %s completed in %.0fs' % (phase_name, time.time() - start))


def main():
    print('NeuOS Season 17: Hafufu Universe Crossovers')
    print('Start: %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    overall_start = time.time()

    phases = [
        ('phase122_rosetta_compiler', 'P122: Rosetta Soul Compiler'),
        ('phase123_stochastic_memory', 'P123: Stochastic Memory Palace'),
        ('phase124_aletheia_firmware', 'P124: Aletheia Firmware'),
        ('phase127_soul_dreaming', 'P127: Soul Dreaming'),
    ]

    action = read_control()

    try:
        for module, name in phases:
            run_phase(module, name)

        elapsed = time.time() - overall_start
        print('\nAll experiments completed in %.0fs (%.1f min)' % (elapsed, elapsed / 60))
    finally:
        if action == 'hibernate':
            print('Hibernating in 10 seconds...')
            print('Hibernating... (disabled for safety)')
            # os.system('shutdown /h')  # Removed for GitHub
        else:
            # Default: beep
            try:
                import winsound
                for _ in range(5):
                    winsound.Beep(1000, 500)
                    time.sleep(0.3)
            except Exception:
                pass
            print('Done! Beep notification sent.')


if __name__ == '__main__':
    main()
