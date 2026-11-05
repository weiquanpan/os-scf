"""Describe all prespecified follow-up seeds, separately from screening seed 101."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/neuro_redundancy_replication_v1'
PILOT = ROOT/'outputs/neuro_redundancy_pilot_s101_v1'
SEEDS = [102, 103, 104, 105]
LEVELS = [3, 8, 16, 32]
METRICS = ['free_sw', 'original11_sw', 'rf_auc', 'linear_auc', 'expanded_full_sw']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs, rows, pairs = {}, [], []
    for seed in SEEDS:
        folder = OUT/f'seed{seed}'
        for name in ['STATUS.json', 'audit.json', 'protocol.json', 'provenance.json',
                     'metrics.csv', 'paired_differences.csv']:
            inputs[f'seed{seed}/{name}'] = sha(folder/name)
        assert json.loads((folder/'STATUS.json').read_text())['status'] == 'COMPLETE'
        audit = json.loads((folder/'audit.json').read_text())
        assert audit['status'] == 'PASS' and audit['max_metric_replay_error'] == 0
        f = pd.read_csv(folder/'metrics.csv')
        assert len(f) == 16 and set(f.seed) == {seed} and set(f.reference) == {'val'}
        rows.append(f)
        p = pd.read_csv(folder/'paired_differences.csv')
        p['seed'] = seed
        pairs.append(p)
    frame = pd.concat(rows, ignore_index=True)
    paired = pd.concat(pairs, ignore_index=True)
    frame.to_csv(OUT/'replication_metrics.csv', index=False)
    paired.to_csv(OUT/'replication_pairs.csv', index=False)
    summary = frame.groupby(['alpha', 'dependent_dimensions', 'method'])[METRICS].agg(['mean', 'std', 'count'])
    summary.to_csv(OUT/'replication_mean_sd.csv')
    comparison = []
    for alpha in [1.0, 0.5]:
        for dv in LEVELS:
            osrow = summary.loc[(alpha, dv, 'osscf')]
            direct = summary.loc[(alpha, dv, 'direct_repair')]
            p = paired[(paired.alpha == alpha) & (paired.dependent_dimensions == dv)]
            comparison.append({
                'alpha': alpha, 'dependent_dimensions': dv, 'n_new_seeds': 4,
                'os_free_sw_mean': osrow[('free_sw', 'mean')],
                'os_free_sw_sd': osrow[('free_sw', 'std')],
                'direct_free_sw_mean': direct[('free_sw', 'mean')],
                'direct_free_sw_sd': direct[('free_sw', 'std')],
                'free_sw_relative_gain_of_means': 1-osrow[('free_sw', 'mean')]/direct[('free_sw', 'mean')],
                'free_sw_seed_wins': int((p.relative_free_sw_gain > 0).sum()),
                'os_original11_sw_mean': osrow[('original11_sw', 'mean')],
                'direct_original11_sw_mean': direct[('original11_sw', 'mean')],
                'original11_sw_seed_wins': int((p.original11_sw_difference_os_minus_direct < 0).sum()),
                'os_rf_auc_mean': osrow[('rf_auc', 'mean')],
                'direct_rf_auc_mean': direct[('rf_auc', 'mean')],
                'os_linear_auc_mean': osrow[('linear_auc', 'mean')],
                'direct_linear_auc_mean': direct[('linear_auc', 'mean')],
            })
    comp = pd.DataFrame(comparison)
    comp.to_csv(OUT/'comparison.csv', index=False)
    # Keep screening rows accessible, but separate from the four-seed main summary.
    pilot = pd.read_csv(PILOT/'metrics.csv')
    inputs['pilot/metrics.csv'] = sha(PILOT/'metrics.csv')
    both = pd.concat([pilot.assign(stage='screening'), frame.assign(stage='replication')], ignore_index=True)
    both.to_csv(OUT/'all_five_seeds_exploratory.csv', index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), squeeze=False)
    for i, alpha in enumerate([1., .5]):
        for j, (metric, label) in enumerate([('free_sw', 'Free SW (8 fixed fields)'),
                                           ('original11_sw', 'SW on the original 11 fields')]):
            ax = axes[i, j]
            for method, color, display in [('osscf', '#5754c6', 'OS-SCF'),
                                            ('direct_repair', '#d66a20', 'Direct + reconstruction')]:
                a = summary.xs((alpha, method), level=('alpha', 'method')).sort_index()
                ax.errorbar(a.index, a[(metric, 'mean')], yerr=a[(metric, 'std')],
                            marker='o', capsize=4, color=color, label=display)
            ax.set_xticks(LEVELS)
            ax.set_xlabel('Dependent fields (8 free fields fixed)')
            ax.set_ylabel(label+'; lower is better')
            ax.set_title(f'alpha={alpha:g}: '+('primary' if alpha == 1 else 'sensitivity'))
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Follow-up seeds 102-105: mean and descriptive seed SD\n'
                 'LendingClub split 3101, validation only; screening seed 101 excluded', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(OUT/'replication_summary.png', dpi=180)
    fig.savefig(OUT/'replication_summary.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))
    for ax, alpha in zip(axes, [1., .5]):
        for seed in SEEDS:
            p = paired[(paired.alpha == alpha) & (paired.seed == seed)].sort_values('dependent_dimensions')
            ax.plot(p.dependent_dimensions, 100*p.relative_free_sw_gain, marker='o', label=f'Seed {seed}')
        ax.axhline(0, color='#444444', linewidth=.8)
        ax.set_xticks(LEVELS)
        ax.set_xlabel('Dependent fields (8 free fields fixed)')
        ax.set_ylabel('Free SW improvement over Direct (%)')
        ax.set_title(f'alpha={alpha:g}; positive favors OS-SCF')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'replication_paired_gains.png', dpi=180)
    fig.savefig(OUT/'replication_paired_gains.pdf')
    plt.close(fig)

    fallacies = {
        'Simpson aggregation': 'Checked: alpha and redundancy levels reported separately; individual paired seed effects retained.',
        'Ecological inference': 'Not applicable: no claim about individual borrowers from aggregate model metrics.',
        'Selection/Berkson': 'Caution: LendingClub was formula-filtered and historically used; semi-synthetic extensions are not new observed data.',
        'Collider conditioning': 'No adjustment-based causal claim about people; this is a controlled computational comparison.',
        'Base-rate neglect': 'Paired methods share categorical draws per seed; AUC is descriptive and not a deployment decision threshold.',
        'Regression to mean': 'Caution: expansion was triggered by pilot seed101; four follow-up seeds are summarized separately, with no seed substitution.',
        'Survivorship': 'Checked: all four prespecified follow-up seeds and all levels are included; no failed or unfavorable run omitted.',
        'Look-elsewhere': 'Checked: both predefined alphas and every level are reported; no p-value or best-seed claim.',
        'Forking paths': 'Caution: sequential exploratory design; protocols and new seed list frozen before follow-up runs, not publicly preregistered.',
        'Causal overreach': 'No attribution solely to gradient dilution or ZCA; dimensions, loss allocation and parameter count co-vary.',
        'Reverse causality': 'Not applicable to controlled added-field intervention; no real-world causal interpretation.'
    }
    report = {'status': 'ANALYZED', 'seeds': SEEDS, 'screening_seed_excluded_from_main_summary': 101,
              'audit': '40 saved model sample replays and 64 metric rows passed; training and utility not independently refitted.',
              'statistical_tests': 0, 'uncertainty': 'Descriptive SD across four training seeds on one fixed validation split.',
              'fallacy_scan_coverage': '11/11', 'fallacy_scan': fallacies,
              'limits': ['One historically used split; no new-split or held-out-test confirmation.',
                         'No TabSyn or Free-TabSyn comparison in this phase.',
                         'Fixed width/steps; not a matched-time or matched-parameter efficiency study.',
                         'Repeated OS points share a checkpoint within each seed/alpha.',
                         'No manuscript claims updated from this exploratory study.']}
    (OUT/'interpretation.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    lines = ['Material Passport', 'Origin Skill: academic-research-suite / experiment-agent',
             'Origin Mode: validate', 'Verification Status: ANALYZED', '',
             'Four follow-up seeds (102--105), separate from screening seed101.',
             'All results use the same split3101 validation set. Not a confirmatory test.', '',
             comp.to_string(index=False), '', 'Limits:']+report['limits']+['', 'Statistical fallacy scan: 11/11 checked.']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n', encoding='utf8')
    manifest = {'script_sha256': sha(Path(__file__)), 'inputs': inputs,
                'outputs': {p.name: sha(p) for p in OUT.iterdir()
                            if p.is_file() and p.name not in ['report_manifest.json', 'replication_plan.json']}}
    (OUT/'report_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(comp.to_string(index=False))


if __name__ == '__main__':
    main()
