# -*- coding: utf-8 -*-
"""
Parallel runner for Phases 113-118.
Batch 1: P113 + P114 + P115 (3 parallel)
Batch 2: P116 + P117 + P118 (3 parallel)
"""
import subprocess, sys, os, time

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), 'experiments2')

BATCH1 = [
    os.path.join(EXPERIMENTS_DIR, 'phase113_platonic.py'),
    os.path.join(EXPERIMENTS_DIR, 'phase114_rosetta_algebra.py'),
    os.path.join(EXPERIMENTS_DIR, 'phase115_composition.py'),
]
BATCH2 = [
    os.path.join(EXPERIMENTS_DIR, 'phase116_convergent.py'),
    os.path.join(EXPERIMENTS_DIR, 'phase117_memory.py'),
    os.path.join(EXPERIMENTS_DIR, 'phase118_immune.py'),
]

def run_batch(batch, batch_name):
    print(f"\n{'='*60}")
    print(f"  {batch_name}: {len(batch)} processes")
    print(f"{'='*60}")
    start = time.time()
    procs = []
    for script in batch:
        print(f"  Starting {os.path.basename(script)}...")
        p = subprocess.Popen(
            [sys.executable, script],
            cwd=os.path.dirname(script),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace'
        )
        procs.append((os.path.basename(script), p))

    # Wait for all to complete
    for name, p in procs:
        stdout, _ = p.communicate()
        status = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
        print(f"\n  [{status}] {name}")
        # Print last 5 lines
        lines = [l for l in stdout.strip().split('\n') if l.strip()]
        for line in lines[-5:]:
            print(f"    {line}")

    elapsed = time.time() - start
    print(f"\n  {batch_name} completed in {elapsed:.0f}s")
    return elapsed

def main():
    total_start = time.time()
    print("=" * 60)
    print("  NeuOS Phase 113-118 Parallel Runner")
    print("=" * 60)

    t1 = run_batch(BATCH1, "Batch 1 (P113+P114+P115)")
    t2 = run_batch(BATCH2, "Batch 2 (P116+P117+P118)")

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE! Total: {total:.0f}s ({total/60:.1f}min)")
    print(f"  Batch 1: {t1:.0f}s, Batch 2: {t2:.0f}s")
    print(f"{'='*60}")

    # Beep notification
    try:
        import winsound
        for _ in range(5):
            winsound.Beep(1000, 500)
            time.sleep(0.3)
    except Exception:
        pass

if __name__ == '__main__':
    main()
