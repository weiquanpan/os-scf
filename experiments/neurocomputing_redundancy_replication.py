"""Prespecified validation replication after a successful one-seed pilot.

Run --check-only first, then --prepare, then --run with the same output directory.
All design choices and the descriptive expansion gate are frozen before training.
"""
from __future__ import annotations

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.neurocomputing import common as nc
from experiments.neurocomputing.evaluate import utility
from constrained_tabular_fm.continuous_original_space import make_free_flow_normalizer
from constrained_tabular_fm.mechanism_ablations import ConditionalVelocity
from constrained_tabular_fm.metrics import sliced_wasserstein

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/neurocomputing_v2'
MONTHS = (1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36)
EXTRAS = ['scheduled_total_interest'] + [f'{name}_month{m}' for m in MONTHS
          for name in ['scheduled_balance', 'scheduled_cumulative_interest']]
CONFIG = {
    'version': 'redundancy-replication-v1', 'dataset': 'lendingclub', 'split': 3101,
    'training_seed': 102, 'reference': 'val', 'free_dimensions': 8,
    'dependent_dimensions': [3, 8, 16, 32], 'alphas': [1.0, 0.5],
    'primary_alpha': 1.0, 'steps': 1600, 'width': 128, 'lr': 0.001,
    'batch_size': 512, 'sample_steps': 80, 'sample_rows': 5000,
    'methods': ['osscf', 'direct_repair'], 'extra_columns': EXTRAS,
    'primary_metric': 'free_sw', 'fixed_block_metric': 'original11_sw',
    'selection': 'None: alpha/steps/seed/levels fixed before this pilot.',
    'gate': {'min_relative_free_sw_gain_at_dv16_and32': 0.05,
             'min_gain_increase_dv32_minus_dv3': 0.05,
             'require_original11_sw_no_worse_at_dv16_and32': True,
             'max_rf_auc_deficit_at_dv16_and32': 0.005,
             'scope': 'Resource-allocation heuristic only, not a significance test.'},
    'notes': [
        'One of four prespecified replication seeds 102--105 on the same split, after screening seed 101; still exploratory.',
        'No test-set metric evaluation. All levels and both alphas are retained.',
        'Observed original 11 fields are retained. Extra fields are deterministic schedules, not observed repayment records.',
        'OS-SCF is trained once per alpha and reused at all redundancy levels; repeated rows are not independent fits.',
        'Same width/steps, not matched parameter count or wall-clock budget.',
        'Independent RNG streams pair minibatch indices, times and first eight base-noise coordinates across dimensions.',
        'Alpha 0.5 is a prespecified sensitivity analysis and cannot rescue a failed primary-alpha gate.',
        'Extra-field Full SW changes dimension; do not compare its absolute values across levels as a common metric.',
        'Timing is descriptive on a shared desktop GPU, including preprocessing in fit_wall_seconds.',
        'No training or manuscript files from the frozen main study are modified.'
    ]
}


def gpu_snapshot():
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def schedule(u, periods):
    """Stable float64 schedules from free fields; independent of observed payment.

    B_t = P * (1 - expm1(t*log1p(r))/expm1(N*log1p(r))).
    Cumulative interest = t*A - (P-B_t). Fields use the paper's rate domain.
    """
    v = u.double()
    principal = v[:, 0].clamp_min(1.0)
    rate = v[:, 1].clamp(0.0, 40.0) / 1200.0
    n = periods.double().clamp_min(1.0)
    zero = rate.abs() < 1e-8
    log_rate = torch.log1p(rate)
    denom = torch.expm1(n * log_rate)
    safe = torch.where(zero, torch.ones_like(denom), denom)
    payment = torch.where(zero, principal / n, principal * rate * (1.0 + denom) / safe)
    fields = [payment * n - principal]
    for month in MONTHS:
        t = torch.minimum(torch.full_like(n, month), n)
        balance = torch.where(zero, principal * (1.0 - t / n),
                              principal * (1.0 - torch.expm1(t * log_rate) / safe))
        fields += [balance, payment * t - (principal - balance)]
    return torch.stack(fields, 1)


def augment(b, original, k, dv):
    assert dv in CONFIG['dependent_dimensions']
    if dv == 3:
        return original.clone()
    extra = schedule(original[:, :8], b.spec.term_months(k)).to(original.dtype)
    return torch.cat([original, extra[:, :dv - 3]], 1)


def check_algebra(b):
    # Independent monthly recurrence, covering zero rate and both allowed terms.
    values = torch.zeros((4, 8), dtype=torch.float64, device=nc.DEVICE)
    values[:, 0] = 10000.0
    values[:, 1] = torch.tensor([0.0, 0.0, 12.0, 24.0], device=nc.DEVICE)
    periods = torch.tensor([36.0, 60.0, 36.0, 60.0], device=nc.DEVICE)
    actual = schedule(values, periods)
    errors = []
    for i in range(4):
        p, r, n = float(values[i, 0]), float(values[i, 1]) / 1200, int(periods[i])
        payment = p / n if r == 0 else p * r / (1 - (1+r)**(-n))
        balance, interest = p, 0.0
        expected = [payment * n - p]
        saved = {}
        for m in range(1, n + 1):
            charge = balance * r
            interest += charge
            balance += charge - payment
            saved[m] = (balance, interest)
        for m in MONTHS:
            expected.extend(saved[min(m, n)])
        errors.append(float(np.max(np.abs(actual[i].cpu().numpy() - expected))))
    assert max(errors) < 1e-7, errors
    x, k = b.train_x[:20], b.train_k[:20]
    full = augment(b, x, k, 32)
    for dv in CONFIG['dependent_dimensions']:
        a = augment(b, x, k, dv)
        assert torch.equal(a[:, :11], x)
        assert torch.equal(a, full[:, :8+dv])
        assert torch.isfinite(a).all()
    return {'status': 'PASS', 'independent_schedule_max_abs_error': max(errors),
            'nested_prefixes_preserve_original_fields': True}


def provenance():
    original = json.loads((EVIDENCE / 'provenance.json').read_text())
    for name, digest in original['sources'].items():
        assert nc.sha(ROOT / name) == digest, f'Frozen source changed: {name}'
    folder = EVIDENCE / 'split3101/data/lendingclub'
    sources = dict(original['sources'])
    sources[str(Path(__file__).relative_to(ROOT))] = nc.sha(__file__)
    data = {str(p.relative_to(ROOT)): nc.sha(p) for p in
            [folder/'train.csv', folder/'val.csv', folder/'manifest.json']}
    pilot = ROOT/'outputs/neuro_redundancy_pilot_s101_v1'
    assert json.loads((pilot/'decision.json').read_text())['status'] == 'EXPAND_CANDIDATE'
    assert json.loads((pilot/'audit.json').read_text())['status'] == 'PASS'
    for name in ['provenance.json', 'protocol.json', 'decision.json', 'audit.json']:
        data[str((pilot/name).relative_to(ROOT))] = nc.sha(pilot/name)
    return {'parent_identity': original['identity'], 'sources': sources, 'inputs': data,
            'config': CONFIG}


def prepare(out, b):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing pilot: {out}')
    out.mkdir(parents=True, exist_ok=True)
    identity = provenance()
    identity['identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    nc.dump(out / 'protocol.json', CONFIG)
    nc.dump(out / 'provenance.json', identity)
    nc.dump(out / 'checks.json', check_algebra(b))
    for name in identity['sources']:
        destination = out/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    nc.dump(out/'STATUS.json', {'status': 'PREPARED', 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
    print(f'Prepared immutable protocol: {out}', flush=True)


def generator(offset):
    return torch.Generator(device=nc.DEVICE).manual_seed(CONFIG['training_seed'] + offset)


def fit(target, b, alpha):
    nc.sync()
    started = time.perf_counter()
    nc.seed(CONFIG['training_seed'])
    if nc.DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    else:
        initial_memory = 0
    norm = make_free_flow_normalizer(target, whiten_shrinkage=alpha)
    z = norm.transform(target)
    kh = b.spec.one_hot(b.train_k)
    model = ConditionalVelocity(z.shape[1], kh.shape[1], CONFIG['width']).to(nc.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    streams = [generator(offset) for offset in (1000, 2000, 3000, 4000)]
    nc.sync()
    train_started = time.perf_counter()
    history = []
    for step in range(CONFIG['steps']):
        idx = torch.randint(len(z), (CONFIG['batch_size'],), device=nc.DEVICE, generator=streams[0])
        t = torch.rand((len(idx), 1), device=nc.DEVICE, generator=streams[1])
        free_noise = torch.randn((len(idx), 8), device=nc.DEVICE, generator=streams[2])
        dep_noise = torch.randn((len(idx), z.shape[1]-8), device=nc.DEVICE, generator=streams[3])
        noise = torch.cat([free_noise, dep_noise], 1)
        y = z[idx]
        loss = (model((1-t)*noise+t*y, kh[idx], t) - (y-noise)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == CONFIG['steps'] - 1:
            history.append({'step': step + 1, 'loss': float(loss.detach())})
    nc.sync()
    ended = time.perf_counter()
    peak = torch.cuda.max_memory_allocated() if nc.DEVICE.type == 'cuda' else 0
    meta = {'fit_wall_seconds': ended-started, 'optimizer_seconds': ended-train_started,
            'peak_torch_allocated_mib': peak/2**20, 'peak_increment_mib': (peak-initial_memory)/2**20,
            'parameters': sum(p.numel() for p in model.parameters()), 'flow_dimension': target.shape[1]}
    return model, norm, meta, history


@torch.no_grad()
def sample(model, norm, b, k):
    model.eval()
    free = torch.randn((len(k), 8), device=nc.DEVICE, generator=generator(600))
    dep = torch.randn((len(k), model.dim-8), device=nc.DEVICE, generator=generator(601))
    z = torch.cat([free, dep], 1)
    kh = b.spec.one_hot(k)
    for i in range(CONFIG['sample_steps']):
        t = torch.full((len(k), 1), i / CONFIG['sample_steps'], device=nc.DEVICE)
        z = (z + model(z, kh, t) / CONFIG['sample_steps']).clamp(-6, 6)
    return nc.decode(b, norm.inverse(z)[:, :8], k)


def evaluate(b, x, k, train_aug, ref_aug, dv):
    base = nc.metrics(b, x[:, :11], k)
    a = x.cpu().numpy().astype(float)
    tr = train_aug.cpu().numpy().astype(float)
    ref = ref_aug.cpu().numpy().astype(float)
    mu, sd = tr.mean(0), tr.std(0).clip(1e-6)
    aa, rr = (a-mu)/sd, (ref-mu)/sd
    extra_residual = 0.0
    if dv > 3:
        expected = schedule(x[:, :8], b.spec.term_months(k))[:, :dv-3].cpu().numpy()
        extra_residual = float(np.max(np.abs(a[:, 11:] - expected)))
        assert extra_residual < .02, extra_residual
    assert base['violation_rate'] == 0
    return {'free_sw': base['free_sw'], 'original11_sw': base['sw'],
            'original3_dependent_sw': base['dep_sw'],
            'expanded_full_sw': sliced_wasserstein(aa, rr, seed=17),
            'expanded_dependent_sw': sliced_wasserstein(aa[:, 8:], rr[:, 8:], seed=23),
            'violation_rate': base['violation_rate'], 'extra_max_abs_residual_dollars': extra_residual}


def write_report(out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(records)
    frame.to_csv(out/'metrics.csv', index=False)
    primary = frame[frame.alpha == CONFIG['primary_alpha']]
    pairs = []
    for alpha in CONFIG['alphas']:
        for dv in CONFIG['dependent_dimensions']:
            q = frame[(frame.alpha == alpha) & (frame.dependent_dimensions == dv)].set_index('method')
            osrow, direct = q.loc['osscf'], q.loc['direct_repair']
            pairs.append({'alpha': alpha, 'dependent_dimensions': dv,
                          'relative_free_sw_gain': 1-osrow.free_sw/direct.free_sw,
                          'original11_sw_difference_os_minus_direct': osrow.original11_sw-direct.original11_sw,
                          'rf_auc_difference_os_minus_direct': osrow.rf_auc-direct.rf_auc})
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(out/'paired_differences.csv', index=False)
    q = pair_frame[pair_frame.alpha == 1.0].set_index('dependent_dimensions')
    gates = {
        'free_sw_gain_ge_5_percent_at_both_high_levels': bool((q.loc[[16, 32], 'relative_free_sw_gain'] >= .05).all()),
        'gain_increases_ge_5_percentage_points': bool(q.loc[32, 'relative_free_sw_gain']-q.loc[3, 'relative_free_sw_gain'] >= .05),
        'original11_sw_no_worse_at_both_high_levels': bool((q.loc[[16, 32], 'original11_sw_difference_os_minus_direct'] <= 0).all()),
        'rf_auc_deficit_le_0_005_at_both_high_levels': bool((q.loc[[16, 32], 'rf_auc_difference_os_minus_direct'] >= -.005).all()),
        'all_equation_checks_pass': bool((frame.violation_rate == 0).all()),
    }
    decision = {'status': 'EXPAND_CANDIDATE' if all(gates.values()) else 'DO_NOT_EXPAND_AS_SUPERIORITY_STUDY',
                'checks': gates, 'interpretation': 'One-seed validation screening only; no significance or generalization claim.'}
    nc.dump(out/'decision.json', decision)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), squeeze=False)
    for row_index, alpha in enumerate(CONFIG['alphas']):
        for column, (metric, ylabel) in enumerate([('free_sw', 'Free SW (8 fields; lower is better)'),
                                                  ('original11_sw', 'Original 11-field SW (lower is better)'),
                                                  ('rf_auc', 'RF TSTR AUC (higher is better)')]):
            ax = axes[row_index, column]
            for method, color, label in [('osscf', '#5754c6', 'OS-SCF (one fit reused)'),
                                          ('direct_repair', '#d66a20', 'Direct + reconstruction')]:
                a = frame[(frame.alpha == alpha) & (frame.method == method)].sort_values('dependent_dimensions')
                ax.plot(a.dependent_dimensions, a[metric], marker='o', color=color, label=label)
            ax.set_xticks(CONFIG['dependent_dimensions'])
            ax.set_xlabel('Number of dependent fields (free fields fixed at 8)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'alpha = {alpha:g}' + (' (primary)' if alpha == 1 else ' (sensitivity)'))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Exploratory redundancy replication: LendingClub split 3101, seed {CONFIG["training_seed"]}, validation only\n'
                 '1,600 steps; width 128; no seed uncertainty estimate', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(out/'redundancy_pilot.png', dpi=180)
    fig.savefig(out/'redundancy_pilot.pdf')
    plt.close(fig)
    lines = ['Material Passport', 'Origin Skill: academic-research-suite / experiment-agent',
             'Origin Mode: run + validate', 'Verification Status: ANALYZED (sample replay checked; training not independently rerun)',
             'Version: redundancy-replication-v1', '', 'Single-seed validation pilot; NOT confirmatory evidence.',
             'Decision: '+decision['status'], '', 'Primary alpha=1 results:',
             primary[['dependent_dimensions', 'method', 'free_sw', 'original11_sw', 'rf_auc', 'linear_auc']].to_string(index=False),
             '', 'All paired comparisons:', pair_frame.to_string(index=False), '',
             'Prespecified screening gate:', json.dumps(gates, indent=2), '', 'Interpretation limits:'] + CONFIG['notes']
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n', encoding='utf8')
    return decision


def run(out, b):
    frozen = json.loads((out/'provenance.json').read_text())
    current = provenance()
    assert {k: v for k, v in frozen.items() if k != 'identity'} == current
    assert json.loads((out/'protocol.json').read_text()) == CONFIG
    assert json.loads((out/'STATUS.json').read_text())['status'] == 'PREPARED', 'Run directory already used; no overwrite/retry.'
    nc.dump(out/'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'start': time.time()})
    started = time.time()
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        nc.dump(out/'environment.json', {'python': sys.version, 'torch': torch.__version__,
                'numpy': np.__version__, 'platform': platform.platform(), 'gpu_start': gpu_snapshot()})
        # Warm-up CUDA outside all individual fit measurements.
        if nc.DEVICE.type == 'cuda':
            dummy = torch.randn((256, 256), device=nc.DEVICE)
            dummy = dummy @ dummy
            nc.sync()
            del dummy
        assert len(b.train_x) == 20000 and len(b.ref_x) == CONFIG['sample_rows']
        k = nc.empirical_k(b, len(b.ref_x), CONFIG['training_seed']+500)
        augmented = {dv: (augment(b, b.train_x, b.train_k, dv), augment(b, b.ref_x, b.ref_k, dv))
                     for dv in CONFIG['dependent_dimensions']}
        records, replays = [], []
        (out/'checkpoints').mkdir()
        (out/'samples').mkdir()
        for alpha in CONFIG['alphas']:
            os_bundle = None
            jobs = [('osscf', 3)] + [('direct_repair', dv) for dv in CONFIG['dependent_dimensions']]
            for method, training_dv in jobs:
                tag = f'a{alpha:g}_{method}_dv{training_dv}'
                nc.event(out, stage='fit_start', tag=tag)
                target = b.train_x[:, :8] if method == 'osscf' else augmented[training_dv][0]
                model, norm, meta, history = fit(target, b, alpha)
                checkpoint = out/'checkpoints'/f'{tag}.pt'
                torch.save({'model': model.state_dict(), 'normalizer': norm, 'meta': meta}, checkpoint)
                nc.dump(out/'checkpoints'/f'{tag}.history.json', history)
                nc.sync()
                sample_started = time.perf_counter()
                x = sample(model, norm, b, k)
                nc.sync()
                meta['original11_sampling_seconds'] = time.perf_counter()-sample_started
                # Load the saved checkpoint into a fresh network and reproduce samples.
                saved = torch.load(checkpoint, map_location=nc.DEVICE, weights_only=False)
                replay_model = ConditionalVelocity(model.dim, b.spec.onehot_dim, CONFIG['width']).to(nc.DEVICE)
                replay_model.load_state_dict(saved['model'])
                replay = sample(replay_model, saved['normalizer'], b, k)
                error = float((x-replay).abs().max())
                assert error == 0, (tag, error)
                replays.append({'tag': tag, 'max_sample_replay_error': error})
                del replay_model, saved, replay
                util = utility(b, nc.frame(b, x, k), method, CONFIG['training_seed'])
                nc.dump(out/'samples'/f'{tag}.utility.json', util)
                scores = {r['utility_model']: r for r in util}
                for dv in (CONFIG['dependent_dimensions'] if method == 'osscf' else [training_dv]):
                    xx = augment(b, x, k, dv)
                    name = f'a{alpha:g}_{method}_dv{dv}'
                    sample_path = out/'samples'/f'{name}.npz'
                    np.savez_compressed(sample_path, x=xx.cpu().numpy(), k=k.cpu().numpy())
                    measured = evaluate(b, xx, k, *augmented[dv], dv)
                    record = {'alpha': alpha, 'method': method, 'dependent_dimensions': dv,
                              'free_dimensions': 8, 'seed': CONFIG['training_seed'], 'split': 3101,
                              'reference': 'val', 'checkpoint': str(checkpoint.relative_to(out)),
                              'sample_file': str(sample_path.relative_to(out)), 'sample_sha256': nc.sha(sample_path),
                              'os_fit_reused_across_levels': method == 'osscf',
                              **meta, **measured, 'rf_auc': scores['rf']['roc_auc'], 'linear_auc': scores['linear']['roc_auc']}
                    nc.dump(out/'samples'/f'{name}.metrics.json', record)
                    records.append(record)
                nc.event(out, stage='fit_complete', tag=tag, seconds=meta['fit_wall_seconds'],
                         free_sw=records[-1]['free_sw'], rf_auc=records[-1]['rf_auc'])
                del model, norm
        assert len(records) == 16 and len(replays) == 10
        # Replay every saved array/metric and check invariance of reused OS free samples.
        metric_errors = []
        for r in records:
            file = out/r['sample_file']; assert nc.sha(file) == r['sample_sha256']
            saved = np.load(file)
            xx = torch.from_numpy(saved['x']).to(nc.DEVICE)
            kk = torch.from_numpy(saved['k']).to(nc.DEVICE)
            measured = evaluate(b, xx, kk, *augmented[r['dependent_dimensions']], r['dependent_dimensions'])
            metric_errors.extend(abs(measured[key]-r[key]) for key in measured)
        assert max(metric_errors) == 0
        for alpha in CONFIG['alphas']:
            samples = [np.load(out/r['sample_file'])['x'][:, :11] for r in records
                       if r['alpha'] == alpha and r['method'] == 'osscf']
            assert all(np.array_equal(samples[0], x) for x in samples[1:])
        assert provenance() == current
        nc.dump(out/'audit.json', {'status': 'PASS', 'checkpoint_replays': replays,
                'metric_rows_replayed': len(records), 'max_metric_replay_error': max(metric_errors),
                'os_original_fields_identical_across_levels': True, 'frozen_sources_and_inputs_unchanged': True,
                'scope': 'All sample checkpoints and numerical metrics; not an independent training or utility refit.'})
        decision = write_report(out, records)
        nc.dump(out/'STATUS.json', {'status': 'COMPLETE', 'duration_seconds': time.time()-started,
                'trained_models': len(replays), 'metric_rows': len(records), 'decision': decision['status'],
                'gpu_end': gpu_snapshot()})
        nc.event(out, stage='complete', decision=decision['status'])
    except BaseException as error:
        nc.dump(out/'STATUS.json', {'status': 'FAILED', 'error': repr(error), 'duration_seconds': time.time()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, choices=[102, 103, 104, 105], required=True)
    args = parser.parse_args()
    CONFIG['training_seed'] = args.seed
    if args.out is None:
        args.out = ROOT/f'outputs/neuro_redundancy_replication_v1/seed{args.seed}'
    b = nc.bundle('lendingclub', 3101, EVIDENCE, reference='val')
    if args.check_only:
        print(json.dumps(check_algebra(b), indent=2))
    elif args.prepare:
        prepare(args.out.resolve(), b)
    else:
        run(args.out.resolve(), b)


if __name__ == '__main__':
    main()
