# -*- coding: utf-8 -*-
"""
Phase 5: Thermodynamic Task Scheduling Simulation (CPU)
Can physics-based dynamics schedule tasks better than round-robin?

Simulate N tasks competing for shared compute resources using
SNN-inspired nutrient field dynamics:
  - Each task has a priority and resource demand
  - "Nutrient" = available compute capacity
  - Tasks "grow" (get allocated) when nutrient > demand
  - Tasks "sleep" when nutrient is depleted
  - Compare vs round-robin and priority queue baselines

Model: None (pure simulation, CPU only)
"""
import json, os, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42)


class Task:
    def __init__(self, tid, priority, demand, total_work):
        self.tid = tid
        self.priority = priority  # higher = more important
        self.demand = demand      # resource units needed per step
        self.total_work = total_work
        self.work_done = 0
        self.completion_time = None
        self.active = True

    def is_complete(self):
        return self.work_done >= self.total_work


def generate_tasks(n=20):
    tasks = []
    for i in range(n):
        priority = np.random.randint(1, 11)
        demand = np.random.uniform(0.5, 3.0)
        total_work = np.random.uniform(5, 50)
        tasks.append(Task(i, priority, demand, total_work))
    return tasks


def round_robin_scheduler(tasks_orig, total_resource, max_steps=500):
    """Classic round-robin: equal time slices."""
    tasks = [Task(t.tid, t.priority, t.demand, t.total_work) for t in tasks_orig]
    completions = {}
    for step in range(max_steps):
        active = [t for t in tasks if not t.is_complete()]
        if not active:
            break
        per_task = total_resource / max(1, len(active))
        for t in active:
            work = min(per_task, t.demand)
            t.work_done += work
            if t.is_complete() and t.tid not in completions:
                completions[t.tid] = step
    return completions


def priority_queue_scheduler(tasks_orig, total_resource, max_steps=500):
    """Priority queue: highest priority gets resources first."""
    tasks = [Task(t.tid, t.priority, t.demand, t.total_work) for t in tasks_orig]
    completions = {}
    for step in range(max_steps):
        active = sorted([t for t in tasks if not t.is_complete()],
                        key=lambda t: -t.priority)
        if not active:
            break
        remaining = total_resource
        for t in active:
            alloc = min(remaining, t.demand)
            t.work_done += alloc
            remaining -= alloc
            if t.is_complete() and t.tid not in completions:
                completions[t.tid] = step
            if remaining <= 0:
                break
    return completions


def thermodynamic_scheduler(tasks_orig, total_resource, max_steps=500):
    """Nutrient field scheduler: tasks compete for resources physically."""
    tasks = [Task(t.tid, t.priority, t.demand, t.total_work) for t in tasks_orig]
    completions = {}
    # Each task has an "energy" that determines resource absorption
    energies = np.array([t.priority * 1.0 for t in tasks])

    for step in range(max_steps):
        active_mask = np.array([not t.is_complete() for t in tasks])
        if not active_mask.any():
            break

        # Nutrient competition: Boltzmann distribution
        active_energies = energies * active_mask
        if active_energies.sum() == 0:
            break

        # Temperature decreases over time (simulated annealing)
        temp = max(0.1, 5.0 * (1 - step / max_steps))
        probs = np.exp(active_energies / temp)
        probs = probs * active_mask
        probs = probs / max(1e-10, probs.sum())

        # Allocate resources proportionally
        allocations = probs * total_resource
        for i, t in enumerate(tasks):
            if not t.is_complete():
                work = min(allocations[i], t.demand)
                t.work_done += work
                # Energy increases with remaining work (hungry tasks grow)
                energies[i] = t.priority * (1 + (t.total_work - t.work_done) / t.total_work)
                if t.is_complete() and t.tid not in completions:
                    completions[t.tid] = step
    return completions


def main():
    print("[P5] Thermodynamic Task Scheduling")
    start_time = time.time()

    n_scenarios = 10
    total_resource = 10.0
    all_results = {'round_robin': [], 'priority_queue': [], 'thermodynamic': []}
    metrics = {'weighted_completion': {}, 'high_priority_latency': {}, 'throughput': {}}

    for scenario in range(n_scenarios):
        np.random.seed(scenario * 7 + 42)
        tasks = generate_tasks(n=20)

        rr = round_robin_scheduler(tasks, total_resource)
        pq = priority_queue_scheduler(tasks, total_resource)
        td = thermodynamic_scheduler(tasks, total_resource)

        for name, completions in [('round_robin', rr), ('priority_queue', pq), ('thermodynamic', td)]:
            # Weighted completion time (priority * completion_time)
            wct = sum(tasks[tid].priority * t for tid, t in completions.items()) / max(1, sum(tasks[tid].priority for tid in completions))
            all_results[name].append({
                'wct': round(wct, 2),
                'completed': len(completions),
                'total': len(tasks),
            })

    # Aggregate
    for name in all_results:
        wcts = [r['wct'] for r in all_results[name]]
        completed = [r['completed'] for r in all_results[name]]
        metrics['weighted_completion'][name] = round(np.mean(wcts), 2)
        metrics['throughput'][name] = round(np.mean(completed), 1)
        print(f"  {name}: WCT={np.mean(wcts):.1f} Throughput={np.mean(completed):.0f}/20")

    # Save
    output = {
        'phase': 5, 'name': 'thermodynamic_scheduling',
        'n_scenarios': n_scenarios, 'metrics': metrics,
        'all_results': all_results,
        'elapsed': round(time.time() - start_time, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase5_scheduling.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    names = ['Round\nRobin', 'Priority\nQueue', 'Thermo-\ndynamic']
    keys = ['round_robin', 'priority_queue', 'thermodynamic']
    colors = ['tab:gray', 'tab:blue', 'tab:red']

    # WCT (lower is better)
    wcts = [metrics['weighted_completion'][k] for k in keys]
    axes[0].bar(names, wcts, color=colors, edgecolor='black')
    axes[0].set_ylabel('Weighted Completion Time (lower=better)', fontsize=11)
    axes[0].set_title('Priority-Weighted Latency', fontsize=14, fontweight='bold')
    for i, v in enumerate(wcts):
        axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

    # Throughput
    tps = [metrics['throughput'][k] for k in keys]
    axes[1].bar(names, tps, color=colors, edgecolor='black')
    axes[1].set_ylabel('Tasks Completed (out of 20)', fontsize=11)
    axes[1].set_title('Throughput', fontsize=14, fontweight='bold')
    for i, v in enumerate(tps):
        axes[1].text(i, v + 0.3, f'{v:.0f}', ha='center', fontweight='bold')

    plt.suptitle('Phase 5: Thermodynamic Task Scheduling\nPhysics-based vs Traditional Schedulers',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase5_scheduling.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Completed in {time.time()-start_time:.0f}s")


if __name__ == '__main__':
    main()
