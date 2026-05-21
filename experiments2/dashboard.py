# -*- coding: utf-8 -*-
"""
NeuOS Master Dashboard: Aggregate analysis of ALL experimental results.
CPU-only: reads JSON results and generates summary figures.
"""
import json, os, glob, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')


def load_all_results():
    """Load all phase JSON results."""
    results = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, 'phase*.json'))):
        fname = os.path.basename(path)
        try:
            with open(path) as f:
                data = json.load(f)
            phase_num = data.get('phase', 0)
            results[phase_num] = data
        except Exception as e:
            print("  Warning: Could not load %s: %s" % (fname, str(e)[:60]))
    return results


def main():
    print("[Dashboard] NeuOS Master Analysis (CPU-only)")
    start = time.time()

    results = load_all_results()
    phases = sorted(results.keys())
    print("  Loaded %d phase results: P%d - P%d" % (len(phases), min(phases), max(phases)))

    # === Analysis 1: Timeline of key accuracy metrics ===
    # Extract accuracy from each phase where available
    timeline_data = {}
    for p in phases:
        r = results[p]
        name = r.get('name', 'unknown')

        # Try various accuracy fields
        acc = None
        if 'gradient_accs' in r:
            acc = np.mean(list(r['gradient_accs'].values()))
        elif 'single_layer_accs' in r:
            accs = [v.get('all', v.get('test', 0)) for v in r['single_layer_accs'].values()]
            acc = max(accs) if accs else None
        elif 'base_accuracy' in r:
            acc = r['base_accuracy']
        elif 'individual_accs' in r:
            acc = np.mean(list(r['individual_accs'].values()))
        elif 'mean_accuracies' in r:
            vals = [v for v in r['mean_accuracies'].values() if v > 0]
            acc = np.mean(vals) if vals else None

        elapsed = r.get('elapsed', 0)
        timeline_data[p] = {
            'name': name, 'accuracy': acc, 'elapsed': elapsed
        }

    # === Analysis 2: Season-level aggregation ===
    season_map = {
        'S1-12': list(range(1, 77)),
        'S13': list(range(77, 91)),
        'S14': list(range(91, 107)),
        'S15': list(range(107, 115)),
        'S16': list(range(115, 122)),
        'S17': list(range(122, 129)),
        'S18': list(range(129, 131)),
    }

    season_stats = {}
    for season, phase_range in season_map.items():
        phase_results = [timeline_data[p] for p in phase_range if p in timeline_data]
        accs = [pr['accuracy'] for pr in phase_results if pr['accuracy'] is not None]
        times = [pr['elapsed'] for pr in phase_results if pr['elapsed'] > 0]
        season_stats[season] = {
            'n_phases': len(phase_results),
            'mean_acc': float(np.mean(accs)) if accs else None,
            'total_time': sum(times),
            'phases_with_data': len(accs),
        }

    # === Analysis 3: Key findings compilation ===
    key_findings = []

    # P113: Platonic Form
    if 113 in results:
        key_findings.append(('P113', 'Platonic Form', 'Within-class cos ~0.005'))

    # P114: Rosetta Algebra
    if 114 in results:
        key_findings.append(('P114', 'Rosetta Algebra', 'Rank-1 translation'))

    # P119: Soul Compression
    if 119 in results:
        key_findings.append(('P119', 'Compression', '64/896 dims sufficient'))

    # P120: Cross-Model
    if 120 in results:
        key_findings.append(('P120', 'Cross-Model', 'Translation FAILS (15%)'))

    # P121: Arms Race
    if 121 in results:
        key_findings.append(('P121', 'Arms Race', 'All backdoors 100% deception'))

    # P122: Rosetta Compiler
    if 122 in results:
        key_findings.append(('P122', 'Soul Compiler', 'Text->Soul cos=1.0'))

    # P123: Stochastic Memory
    if 123 in results:
        key_findings.append(('P123', 'Stochastic Memory', '4-step: 40%->68% w/ noise'))

    # P124: Aletheia Firmware
    if 124 in results:
        key_findings.append(('P124', 'Firmware', 'L8 optimal (90% > L16 80%)'))

    # P128: Combined
    if 128 in results:
        key_findings.append(('P128', 'Combined', 'L8 +20pp across all conditions'))

    # P129: Phase Diagram
    if 129 in results:
        r = results[129]
        key_findings.append(('P129', 'Phase Diagram',
                            'Fractal boundary %.1f%% of space' % (
                                r.get('phase_boundary_fraction', 0) * 100)))

    # P130: Phylogenetic
    if 130 in results:
        key_findings.append(('P130', 'Phylogenetic', '40 souls: within-cos=0.003-0.032'))

    # === PLOT: 3 panels ===
    fig = plt.figure(figsize=(20, 14))

    # Panel 1 (top-left): Timeline of accuracy
    ax1 = fig.add_subplot(2, 2, 1)
    acc_phases = [p for p in phases if timeline_data.get(p, {}).get('accuracy') is not None]
    acc_vals = [timeline_data[p]['accuracy'] for p in acc_phases]
    colors_timeline = []
    for p in acc_phases:
        if p < 77:
            colors_timeline.append('#9E9E9E')  # S1-12
        elif p < 107:
            colors_timeline.append('#2196F3')  # S13-14
        elif p < 122:
            colors_timeline.append('#4CAF50')  # S15-16
        elif p < 131:
            colors_timeline.append('#FF5722')  # S17-18
        else:
            colors_timeline.append('#9C27B0')  # S19+
    ax1.scatter(acc_phases, acc_vals, c=colors_timeline, s=40, alpha=0.7, edgecolors='black',
                linewidths=0.5)
    ax1.set_xlabel('Phase Number')
    ax1.set_ylabel('Peak Accuracy')
    ax1.set_title('NeuOS Accuracy Timeline (%d phases)' % len(acc_phases), fontweight='bold')
    ax1.set_ylim(-0.05, 1.15)
    ax1.grid(True, alpha=0.3)
    # Season boundaries
    for boundary, label in [(77, 'S13'), (107, 'S15'), (122, 'S17'), (131, 'S19')]:
        if boundary <= max(phases):
            ax1.axvline(x=boundary, color='red', linestyle=':', alpha=0.3)
            ax1.text(boundary + 0.5, 1.1, label, fontsize=8, color='red')

    # Panel 2 (top-right): Compute time by season
    ax2 = fig.add_subplot(2, 2, 2)
    seasons = list(season_stats.keys())
    times = [season_stats[s]['total_time'] / 60 for s in seasons]  # minutes
    n_phases_s = [season_stats[s]['n_phases'] for s in seasons]
    bars = ax2.bar(range(len(seasons)), times,
                   color=['#9E9E9E', '#2196F3', '#2196F3', '#4CAF50',
                          '#4CAF50', '#FF5722', '#FF5722'],
                   edgecolor='black')
    ax2.set_xticks(range(len(seasons)))
    ax2.set_xticklabels(seasons, fontsize=9)
    ax2.set_ylabel('Total Compute (minutes)')
    ax2.set_title('Compute Time by Season', fontweight='bold')
    for bar, t, n in zip(bars, times, n_phases_s):
        if t > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     "%.0fm\n(%d)" % (t, n), ha='center', fontsize=8)

    # Panel 3 (bottom-left): Key findings table
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    if key_findings:
        table_data = [[p, n, f] for p, n, f in key_findings]
        table = ax3.table(cellText=table_data,
                         colLabels=['Phase', 'Name', 'Key Finding'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        # Color header
        for j in range(3):
            table[0, j].set_facecolor('#1976D2')
            table[0, j].set_text_props(color='white', fontweight='bold')
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            color = '#E3F2FD' if i % 2 == 0 else 'white'
            for j in range(3):
                table[i, j].set_facecolor(color)
    ax3.set_title('Key Discoveries', fontweight='bold', fontsize=12, pad=20)

    # Panel 4 (bottom-right): Layer optimization summary
    ax4 = fig.add_subplot(2, 2, 4)
    # Data from P124 and P128
    if 124 in results:
        r124 = results[124]
        layers = ['L4', 'L8', 'L12', 'L16', 'L20']
        accs_124 = [r124['single_layer_accs'][l]['all'] for l in layers]
        ax4.plot(range(len(layers)), accs_124, 'o-', color='#2196F3',
                label='P124 (MIN single-layer)', markersize=8, linewidth=2)
        ax4.set_xticks(range(len(layers)))
        ax4.set_xticklabels(layers)
    ax4.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='L16 baseline')
    ax4.set_xlabel('Injection Layer')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Layer Optimization (P124)', fontweight='bold')
    ax4.legend()
    ax4.set_ylim(0.5, 1.05)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('NeuOS Research Dashboard\n'
                 '%d Phases | Seasons 1-18 | Hafufu Research Universe' % len(phases),
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'master_dashboard.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Dashboard saved to figures/master_dashboard.png")

    # === Summary stats ===
    total_compute = sum(r.get('elapsed', 0) for r in results.values())
    print("\n  === NeuOS Research Stats ===")
    print("  Total phases with data: %d" % len(phases))
    print("  Total compute time: %.1f min (%.1f hours)" % (
        total_compute / 60, total_compute / 3600))
    print("  Key findings: %d" % len(key_findings))
    for p, n, f in key_findings:
        print("    %s (%s): %s" % (p, n, f))
    print("  Completed in %.0fs" % (time.time() - start))


if __name__ == '__main__':
    main()
