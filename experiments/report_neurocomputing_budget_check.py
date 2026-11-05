"""Report the frozen new-split/budget diagnostic without selecting outcomes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/neuro_redundancy_budget_check_v1'
METRICS = ['free_sw', 'original11_sw', 'rf_auc', 'linear_auc']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    status = json.loads((OUT/'STATUS.json').read_text())
    audit = json.loads((OUT/'audit.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert status['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    frame = pd.read_csv(OUT/'metrics.csv')
    assert len(frame) == 36 and set(frame.seed) == {102, 103, 104}
    assert set(frame.split) == {3102} and set(frame.steps) == {1600, 6400}
    assert set(frame.dependent_dimensions) == {3, 16, 32} and set(frame.reference) == {'val'}
    summary = frame.groupby(['steps', 'dependent_dimensions', 'method'])[METRICS].agg(['mean', 'std', 'count'])
    summary.to_csv(OUT/'mean_sd.csv')
    pairs, comparison = [], []
    for steps in [1600, 6400]:
        for dv in [3, 16, 32]:
            q = frame[(frame.steps == steps) & (frame.dependent_dimensions == dv)]
            for seed in [102, 103, 104]:
                s = q[q.seed == seed].set_index('method')
                a, d = s.loc['osscf'], s.loc['direct_repair']
                pairs.append({'steps': steps, 'dependent_dimensions': dv, 'seed': seed,
                              'free_sw_difference_os_minus_direct': a.free_sw-d.free_sw,
                              'relative_free_sw_gain': 1-a.free_sw/d.free_sw,
                              'original11_sw_difference_os_minus_direct': a.original11_sw-d.original11_sw,
                              'rf_auc_difference_os_minus_direct': a.rf_auc-d.rf_auc})
            a, d = summary.loc[(steps, dv, 'osscf')], summary.loc[(steps, dv, 'direct_repair')]
            p = pd.DataFrame([r for r in pairs if r['steps'] == steps and r['dependent_dimensions'] == dv])
            row = {'steps': steps, 'dependent_dimensions': dv, 'n_seeds': 3}
            for metric in METRICS:
                row.update({f'os_{metric}_mean': a[(metric, 'mean')], f'direct_{metric}_mean': d[(metric, 'mean')],
                            f'os_{metric}_sd': a[(metric, 'std')], f'direct_{metric}_sd': d[(metric, 'std')]})
            row.update(relative_free_sw_gain_of_means=1-a[('free_sw', 'mean')]/d[('free_sw', 'mean')],
                       free_sw_seed_wins=int((p.relative_free_sw_gain > 0).sum()),
                       original11_sw_seed_wins=int((p.original11_sw_difference_os_minus_direct < 0).sum()))
            comparison.append(row)
    pairs = pd.DataFrame(pairs)
    comp = pd.DataFrame(comparison)
    pairs.to_csv(OUT/'paired_differences.csv', index=False)
    comp.to_csv(OUT/'comparison.csv', index=False)

    cross = []
    for dv in [3, 16, 32]:
        a = frame[(frame.steps == 1600) & (frame.method == 'osscf') & (frame.dependent_dimensions == dv)].set_index('seed')
        d = frame[(frame.steps == 6400) & (frame.method == 'direct_repair') & (frame.dependent_dimensions == dv)].set_index('seed')
        cross.append({'dependent_dimensions': dv, 'os1600_free_sw_mean': a.free_sw.mean(),
                      'direct6400_free_sw_mean': d.free_sw.mean(),
                      'direct6400_free_sw_no_worse_seeds': int((d.free_sw <= a.free_sw).sum()),
                      'os1600_original11_sw_mean': a.original11_sw.mean(),
                      'direct6400_original11_sw_mean': d.original11_sw.mean(),
                      'direct6400_original11_sw_no_worse_seeds': int((d.original11_sw <= a.original11_sw).sum())})
    cross = pd.DataFrame(cross)
    cross.to_csv(OUT/'cross_budget.csv', index=False)
    changes = []
    for method in ['osscf', 'direct_repair']:
        for dv in [3, 16, 32]:
            a = frame[(frame.steps == 1600) & (frame.method == method) & (frame.dependent_dimensions == dv)].set_index('seed')
            b = frame[(frame.steps == 6400) & (frame.method == method) & (frame.dependent_dimensions == dv)].set_index('seed')
            changes.append({'method': method, 'dependent_dimensions': dv,
                            'free_sw_gain_of_means_with_more_steps': 1-b.free_sw.mean()/a.free_sw.mean(),
                            'free_sw_improved_seeds': int((b.free_sw < a.free_sw).sum()),
                            'original11_sw_gain_of_means_with_more_steps': 1-b.original11_sw.mean()/a.original11_sw.mean()})
    pd.DataFrame(changes).to_csv(OUT/'budget_changes.csv', index=False)

    gates = {}
    for steps in [1600, 6400]:
        q = comp[(comp.steps == steps) & comp.dependent_dimensions.isin([16, 32])]
        gates[str(steps)] = {
            'both_high_levels_mean_free_sw_gain_ge_5_percent': bool((q.relative_free_sw_gain_of_means >= .05).all()),
            'both_high_levels_win_at_least_two_of_three_seeds': bool((q.free_sw_seed_wins >= 2).all()),
            'both_high_levels_original11_mean_not_worse': bool((q.os_original11_sw_mean <= q.direct_original11_sw_mean).all())}
    short_ok, long_ok = all(gates['1600'].values()), all(gates['6400'].values())
    if short_ok and long_ok:
        decision = 'FITTING_SIGNAL_SURVIVES_THIS_DIAGNOSTIC'
    elif short_ok:
        decision = 'BUDGET_SENSITIVE_SIGNAL_NOT_BROAD_SUPERIORITY'
    elif long_ok:
        decision = 'PARTIAL_SIGNAL_ONLY_AT_LONG_BUDGET'
    else:
        decision = 'STOP_EXPANDING_BROAD_REDUNDANCY_SUPERIORITY_CLAIM'
    verdict = {'status': decision, 'checks': gates, 'scope': protocol['decision_rule_per_budget']['scope'],
               'rf_auc_deficits': comp[['steps', 'dependent_dimensions', 'os_rf_auc_mean', 'direct_rf_auc_mean']].to_dict('records')}
    (OUT/'decision.json').write_text(json.dumps(verdict, indent=2))

    old = pd.read_csv(ROOT/'outputs/neuro_redundancy_replication_v1/replication_metrics.csv')
    old = old[(old.alpha == 1) & old.seed.isin([102, 103, 104]) & old.dependent_dimensions.isin([3, 16, 32])].copy()
    old['steps'] = 1600
    old['phase'] = 'earlier_split3101_matched_seeds'
    new = frame.copy()
    new['phase'] = 'new_split3102'
    together = pd.concat([old, new], ignore_index=True)
    together.to_csv(OUT/'split_comparison_per_run.csv', index=False)
    together.groupby(['split', 'steps', 'dependent_dimensions', 'method'])[METRICS].agg(['mean', 'std', 'count']).to_csv(OUT/'split_comparison_mean_sd.csv')

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for i, steps in enumerate([1600, 6400]):
        for j, (metric, label) in enumerate([('free_sw', 'Free SW (8 fixed fields)'), ('original11_sw', 'SW on original 11 fields')]):
            ax = axes[i, j]
            for method, color, display in [('osscf', '#5754c6', 'OS-SCF'), ('direct_repair', '#d66a20', 'Direct + reconstruction')]:
                q = summary.xs((steps, method), level=('steps', 'method')).sort_index()
                ax.errorbar(q.index, q[(metric, 'mean')], yerr=q[(metric, 'std')], marker='o', capsize=4, color=color, label=display)
            ax.set_xticks([3, 16, 32])
            ax.set_xlabel('Dependent fields (8 free fields fixed)')
            ax.set_ylabel(label+'; lower is better')
            ax.set_title(f'{steps:,} steps for both methods')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('New split 3102: mean and descriptive SD over seeds 102-104\n'
                 'alpha=1; width=128; validation only; OS fits reused across field counts', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(OUT/'new_split_budget_summary.png', dpi=180)
    fig.savefig(OUT/'new_split_budget_summary.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for ax, steps in zip(axes, [1600, 6400]):
        for seed in [102, 103, 104]:
            q = pairs[(pairs.steps == steps) & (pairs.seed == seed)].sort_values('dependent_dimensions')
            ax.plot(q.dependent_dimensions, q.relative_free_sw_gain*100, marker='o', label=f'Seed {seed}')
        ax.axhline(0, color='#444', linewidth=.8)
        ax.set_xticks([3, 16, 32])
        ax.set_xlabel('Dependent fields')
        ax.set_ylabel('Free SW improvement over Direct (%)')
        ax.set_title(f'{steps:,} steps; positive favors OS-SCF')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'paired_budget_gains.png', dpi=180)
    fig.savefig(OUT/'paired_budget_gains.pdf')
    plt.close(fig)

    display = comp[['steps', 'dependent_dimensions', 'os_free_sw_mean', 'direct_free_sw_mean',
                    'relative_free_sw_gain_of_means', 'free_sw_seed_wins',
                    'os_original11_sw_mean', 'direct_original11_sw_mean', 'os_rf_auc_mean', 'direct_rf_auc_mean']]
    lines = ['Material Passport', 'Origin Skill: academic-research-suite / experiment-agent',
             'Origin Mode: run + validate', 'Verification Status: ANALYZED (checkpoint and metric replay passed)',
             'Version: redundancy-budget-check-v1', '', 'Prespecified decision: '+decision, '',
             'New split3102, three seeds, 24 model fits. Validation only; not an external-data confirmation.', '',
             display.to_string(index=False), '', 'Direct6400 compared with OS1600:', cross.to_string(index=False),
             '', 'Decision checks:', json.dumps(gates, indent=2), '', 'Limits:']+protocol['notes']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n', encoding='utf8')
    input_names = ['protocol.json', 'provenance.json', 'STATUS.json', 'audit.json', 'metrics.csv', 'utility_per_fit.csv']
    generated = ['mean_sd.csv', 'paired_differences.csv', 'comparison.csv', 'cross_budget.csv', 'budget_changes.csv',
                 'decision.json', 'split_comparison_per_run.csv', 'split_comparison_mean_sd.csv',
                 'new_split_budget_summary.png', 'new_split_budget_summary.pdf',
                 'paired_budget_gains.png', 'paired_budget_gains.pdf', 'RESULTS.txt']
    manifest = {'script_sha256': sha(Path(__file__)), 'inputs': {n: sha(OUT/n) for n in input_names},
                'earlier_split_metrics_sha256': sha(ROOT/'outputs/neuro_redundancy_replication_v1/replication_metrics.csv'),
                'outputs': {n: sha(OUT/n) for n in generated}}
    (OUT/'report_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(display.to_string(index=False))
    print('Decision:', decision)
    print(cross.to_string(index=False))


if __name__ == '__main__':
    main()
