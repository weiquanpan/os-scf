"""Fixed 24-fit follow-up: another split and fourfold training budgets.

Uses the frozen pilot's formula, fitting, sampling and evaluation functions.
Only the in-memory seed and iteration count vary according to protocol.json.
"""
from __future__ import annotations

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import copy
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

from experiments import neurocomputing_redundancy_pilot as pilot

nc, torch, np, pd = pilot.nc, pilot.torch, pilot.np, pilot.pd
ROOT = pilot.ROOT
OUT = ROOT/'outputs/neuro_redundancy_budget_check_v1'
PILOT = ROOT/'outputs/neuro_redundancy_pilot_s101_v1'
REPLICATION = ROOT/'outputs/neuro_redundancy_replication_v1'
PROTOCOL = {
    'version': 'redundancy-budget-check-v1', 'dataset': 'lendingclub', 'split': 3102,
    'training_seeds': [102, 103, 104], 'training_steps': [1600, 6400],
    'dependent_dimensions': [3, 16, 32], 'free_dimensions': 8, 'alpha': 1.0,
    'width': 128, 'lr': .001, 'batch_size': 512, 'sample_steps': 80,
    'training_rows': 20000, 'evaluation_rows': 5000, 'reference': 'val',
    'methods': ['osscf', 'direct_repair'], 'expected_fits': 24,
    'expected_metric_rows': 36, 'extra_columns': pilot.EXTRAS,
    'primary_metric': 'free_sw', 'fixed_secondary_metric': 'original11_sw',
    'decision_rule_per_budget': {
        'high_redundancy_levels': [16, 32], 'min_relative_mean_free_sw_gain': .05,
        'min_paired_free_sw_wins_out_of_three': 2,
        'require_original11_sw_mean_not_worse': True,
        'scope': 'Descriptive go/no-go heuristic, not a significance test.'
    },
    'notes': [
        'Fixed new split relative to the redundancy pilot, but from the same historically used LendingClub pool.',
        'This remains validation-set exploration; no held-out-test metric evaluation.',
        'Training functions and schedules are reused unchanged from the frozen pilot.',
        'Both budgets restart from the same seed and RNG streams; no validation checkpoint selection.',
        'OS-SCF is fitted once per seed/budget and reconstructed at three levels; reused points are not independent fits.',
        'Report both equal-budget comparisons and Direct-6400 versus OS-SCF-1600.',
        'All 24 fits and every outcome retained; no early stopping after favorable or unfavorable results.',
        'Match width/steps, not parameter count or elapsed time. Timings on a shared desktop GPU are descriptive.',
        'Original eight free fields and eleven-field metrics have fixed dimensions; expanded SW is secondary.',
        'No significance claims from three seeds; do not treat overlapping splits as independent datasets.',
        'Do not modify frozen main-study sources, original results or manuscripts.'
    ]
}


def provenance():
    main = json.loads((pilot.EVIDENCE/'provenance.json').read_text())
    for name, digest in main['sources'].items():
        assert nc.sha(ROOT/name) == digest, name
    old = json.loads((PILOT/'provenance.json').read_text())
    pilot_path = Path(pilot.__file__).resolve()
    assert nc.sha(pilot_path) == old['sources'][str(pilot_path.relative_to(ROOT))]
    sources = dict(main['sources'])
    for path in [pilot_path, Path(__file__).resolve(), ROOT/'experiments/report_neurocomputing_budget_check.py']:
        sources[str(path.relative_to(ROOT))] = nc.sha(path)
    folder = pilot.EVIDENCE/'split3102/data/lendingclub'
    paths = [folder/'train.csv', folder/'val.csv', folder/'manifest.json',
             PILOT/'protocol.json', PILOT/'provenance.json', REPLICATION/'comparison.csv',
             REPLICATION/'replication_metrics.csv']
    manuscript = json.loads((ROOT/'outputs/manuscript_revision_neurocomputing_20261103/delivery_manifest.json').read_text())
    for name in ['paper.tex', 'appendix.tex']:
        path = ROOT/'paper-v6'/name
        assert nc.sha(path) == manuscript['source_files'][name]
        paths.append(path)
    return {'parent_main_identity': main['identity'], 'pilot_identity': old['identity'],
            'sources': sources, 'inputs': {str(p.relative_to(ROOT)): nc.sha(p) for p in paths},
            'protocol': PROTOCOL, 'helper_config_before_runtime_overrides': copy.deepcopy(pilot.CONFIG)}


def prepare(out, b):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output already exists: {out}')
    identity = provenance()
    identity['identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    nc.dump(out/'protocol.json', PROTOCOL)
    nc.dump(out/'provenance.json', identity)
    nc.dump(out/'algebra_checks.json', pilot.check_algebra(b))
    for name in identity['sources']:
        destination = out/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    nc.dump(out/'STATUS.json', {'status': 'PREPARED', 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
    print('Prepared 24-fit protocol:', out, flush=True)


def run(out, b):
    original_helper_config = copy.deepcopy(pilot.CONFIG)
    frozen = json.loads((out/'provenance.json').read_text())
    assert {k: v for k, v in frozen.items() if k != 'identity'} == provenance()
    assert json.loads((out/'protocol.json').read_text()) == PROTOCOL
    assert json.loads((out/'STATUS.json').read_text())['status'] == 'PREPARED'
    assert len(b.train_x) == 20000 and len(b.ref_x) == 5000
    started = time.time()
    nc.dump(out/'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'start': started, 'completed_fits': 0})
    nc.dump(out/'environment.json', {'python': sys.version, 'torch': torch.__version__,
            'numpy': np.__version__, 'platform': platform.platform(), 'gpu_start': pilot.gpu_snapshot()})
    records, replays, prefix_checks, all_utilities = [], [], [], []
    histories = {}
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if nc.DEVICE.type == 'cuda':
            warmup = torch.randn((256, 256), device=nc.DEVICE)
            warmup = warmup @ warmup
            nc.sync()
            del warmup
        augmented = {dv: (pilot.augment(b, b.train_x, b.train_k, dv),
                           pilot.augment(b, b.ref_x, b.ref_k, dv))
                     for dv in PROTOCOL['dependent_dimensions']}
        (out/'checkpoints').mkdir()
        (out/'samples').mkdir()
        for seed in PROTOCOL['training_seeds']:
            pilot.CONFIG['training_seed'] = seed
            k = nc.empirical_k(b, len(b.ref_x), seed+500)
            for steps in PROTOCOL['training_steps']:
                pilot.CONFIG['steps'] = steps
                jobs = [('osscf', 3)] + [('direct_repair', dv) for dv in PROTOCOL['dependent_dimensions']]
                for method, training_dv in jobs:
                    tag = f'seed{seed}_steps{steps}_{method}_dv{training_dv}'
                    nc.event(out, stage='fit_start', tag=tag, completed_fits=len(replays))
                    target = b.train_x[:, :8] if method == 'osscf' else augmented[training_dv][0]
                    model, norm, metadata, history = pilot.fit(target, b, 1.0)
                    checkpoint = out/'checkpoints'/f'{tag}.pt'
                    torch.save({'model': model.state_dict(), 'normalizer': norm, 'metadata': metadata,
                                'seed': seed, 'steps': steps, 'alpha': 1.0}, checkpoint)
                    nc.dump(out/'checkpoints'/f'{tag}.history.json', history)
                    key = (seed, method, training_dv)
                    histories[(key, steps)] = {r['step']: r['loss'] for r in history}
                    if steps == 6400:
                        short = histories[(key, 1600)]
                        long = histories[(key, 6400)]
                        common = sorted(set(short) & set(long))
                        max_error = max(abs(short[t]-long[t]) for t in common)
                        assert max_error == 0, (tag, max_error)
                        prefix_checks.append({'seed': seed, 'method': method, 'dependent_dimensions': training_dv,
                                              'common_logged_steps': common, 'max_loss_difference': max_error})
                    nc.sync()
                    sample_started = time.perf_counter()
                    x = pilot.sample(model, norm, b, k)
                    nc.sync()
                    metadata['original11_sampling_seconds'] = time.perf_counter()-sample_started
                    saved = torch.load(checkpoint, map_location=nc.DEVICE, weights_only=False)
                    replay_model = pilot.ConditionalVelocity(model.dim, b.spec.onehot_dim, 128).to(nc.DEVICE)
                    replay_model.load_state_dict(saved['model'])
                    replay = pilot.sample(replay_model, saved['normalizer'], b, k)
                    error = float((x-replay).abs().max())
                    assert error == 0, (tag, error)
                    replays.append({'tag': tag, 'max_sample_replay_error': error})
                    del saved, replay_model, replay
                    utilities = pilot.utility(b, nc.frame(b, x, k), method, seed)
                    nc.dump(out/'samples'/f'{tag}.utility.json', utilities)
                    scores = {r['utility_model']: r for r in utilities}
                    for row in utilities:
                        all_utilities.append({**row, 'steps': steps, 'training_dependent_dimensions': training_dv,
                                              'fit_tag': tag, 'os_fit_reused': method == 'osscf'})
                    for dv in (PROTOCOL['dependent_dimensions'] if method == 'osscf' else [training_dv]):
                        xx = pilot.augment(b, x, k, dv)
                        name = f'seed{seed}_steps{steps}_{method}_dv{dv}'
                        sample_file = out/'samples'/f'{name}.npz'
                        np.savez_compressed(sample_file, x=xx.cpu().numpy(), k=k.cpu().numpy())
                        metrics = pilot.evaluate(b, xx, k, *augmented[dv], dv)
                        record = {'seed': seed, 'split': 3102, 'steps': steps, 'alpha': 1.0, 'reference': 'val',
                                  'method': method, 'dependent_dimensions': dv, 'free_dimensions': 8,
                                  'checkpoint': str(checkpoint.relative_to(out)),
                                  'sample_file': str(sample_file.relative_to(out)), 'sample_sha256': nc.sha(sample_file),
                                  'os_fit_reused': method == 'osscf', **metadata, **metrics,
                                  'rf_auc': scores['rf']['roc_auc'], 'linear_auc': scores['linear']['roc_auc']}
                        nc.dump(out/'samples'/f'{name}.metrics.json', record)
                        records.append(record)
                    nc.event(out, stage='fit_complete', tag=tag, fit_seconds=metadata['fit_wall_seconds'],
                             free_sw=records[-1]['free_sw'], original11_sw=records[-1]['original11_sw'],
                             completed_fits=len(replays), total_fits=24)
                    nc.dump(out/'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'start': started,
                                               'completed_fits': len(replays), 'total_fits': 24})
                    del model, norm
        assert len(replays) == 24 and len(records) == 36 and len(prefix_checks) == 12
        pd.DataFrame(records).to_csv(out/'metrics.csv', index=False)
        pd.DataFrame(all_utilities).to_csv(out/'utility_per_fit.csv', index=False)
        errors = []
        for r in records:
            path = out/r['sample_file']
            assert nc.sha(path) == r['sample_sha256']
            saved = np.load(path)
            xx, kk = torch.from_numpy(saved['x']).to(nc.DEVICE), torch.from_numpy(saved['k']).to(nc.DEVICE)
            measured = pilot.evaluate(b, xx, kk, *augmented[r['dependent_dimensions']], r['dependent_dimensions'])
            errors.extend(abs(measured[key]-r[key]) for key in measured)
        assert max(errors) == 0
        for seed in PROTOCOL['training_seeds']:
            for steps in PROTOCOL['training_steps']:
                arrays = [np.load(out/r['sample_file'])['x'][:, :11] for r in records
                          if r['seed'] == seed and r['steps'] == steps and r['method'] == 'osscf']
                assert len(arrays) == 3 and all(np.array_equal(arrays[0], a) for a in arrays[1:])
        pilot.CONFIG.clear()
        pilot.CONFIG.update(original_helper_config)
        assert {k: v for k, v in frozen.items() if k != 'identity'} == provenance()
        nc.dump(out/'audit.json', {'status': 'PASS', 'checkpoint_replays': replays,
                'metric_rows_replayed': len(records), 'max_metric_replay_error': max(errors),
                'paired_budget_logged_loss_checks': prefix_checks,
                'os_original_fields_identical_across_levels': True,
                'frozen_sources_inputs_and_manuscripts_unchanged': True,
                'scope': 'All checkpoint sample replays and numerical metric replays. Utility not independently refitted; no independent full-training reproduction.'})
        nc.dump(out/'STATUS.json', {'status': 'COMPLETE', 'duration_seconds': time.time()-started,
                'trained_models': 24, 'metric_rows': 36, 'utility_rows': len(all_utilities),
                'gpu_end': pilot.gpu_snapshot()})
        nc.event(out, stage='complete', trained_models=24, metric_rows=36)
    except BaseException as error:
        nc.dump(out/'STATUS.json', {'status': 'FAILED', 'error': repr(error),
                'completed_fits': len(replays), 'duration_seconds': time.time()-started})
        raise
    finally:
        pilot.CONFIG.clear()
        pilot.CONFIG.update(original_helper_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    b = nc.bundle('lendingclub', 3102, pilot.EVIDENCE, reference='val')
    if args.prepare:
        prepare(args.out.resolve(), b)
    else:
        run(args.out.resolve(), b)


if __name__ == '__main__':
    main()
