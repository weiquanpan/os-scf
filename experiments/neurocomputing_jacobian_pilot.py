"""One-seed validation pilot of fixed, train-derived Jacobian loss weights.

Four fits only. Frozen main-study implementations, data and manuscripts are read-only.
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
from experiments import neurocomputing_c1_diagnostics as diag

nc, np, pd, torch = pilot.nc, pilot.np, pilot.pd, pilot.torch
ROOT = pilot.ROOT
MAIN = pilot.EVIDENCE
OUT = ROOT/'outputs/neuro_jacobian_pilot_s106_v1'
METHODS = ['osscf', 'os_jacobian', 'os_permuted', 'direct_repair']
PROTOCOL = {
    'version': 'jacobian-weighted-pilot-v1', 'dataset': 'lendingclub', 'split': 3101,
    'training_seed': 106, 'reference': 'val', 'train_rows': 20000, 'evaluation_rows': 5000,
    'methods': METHODS, 'expected_fits': 4, 'free_dimensions': 8, 'dependent_dimensions': 3,
    'alpha': 1.0, 'steps': 1600, 'width': 128, 'batch_size': 512, 'lr': .001, 'sample_steps': 80,
    'weight_definition': {
        'metric_units': 'train population standard deviations, floor 1e-6; original 11 numerical fields',
        'G_u': 'diag(1/s_u^2) + mean_train[J_phi.T diag(1/s_v^2) J_phi]',
        'coordinate_pullback': 'A = diag(normalizer.std) @ normalizer.W_inv; G_z = A.T @ G_u @ A',
        'approximation': 'constant diagonal of G_z, normalized to mean 1',
        'shrinkage': .25, 'max_weight_ratio': 4.0,
        'processing_order': 'normalize diagonal; shrink toward ones; upper-cap at 4*minimum weight; renormalize mean to 1; cast float32',
        'permutation': [5, 6, 7, 0, 1, 2, 3, 4],
        'permutation_reason': 'fixed cyclic shift by three, same weight multiset; not chosen from outcomes',
        'sample_dependence': 'none: fixed global weights, not endpoint-dependent per-row weighting',
    },
    'training': 'Same conditional MLP and paired minibatch/time/free-noise RNG streams as redundancy pilot; weighted squared velocity residual averaged over batch and dimensions.',
    'selection': 'No alpha/strength/seed scan, validation checkpoint selection, early stopping or rerun after results.',
    'primary_metric': 'original11_sw', 'secondary_metric': 'original3_dependent_sw',
    'guardrails': ['free_sw', 'rf_auc', 'formula_validity'],
    'tail_quantiles': [.8, .9], 'tail_definition': 'training quantiles of standardized reconstruction Jacobian operator norm; same rule as C1, evaluated on validation only',
    'pilot_benefit_screen': {
        'full_sw_relative_improvement_over_os_at_least': .02,
        'dependent_sw_no_worse_than_os': True,
        'free_sw_max_relative_worsening_vs_os': .05,
        'rf_auc_max_absolute_drop_vs_os': .005,
        'all_formula_checks_pass': True,
        'scope': 'Descriptive resource-allocation heuristic, not statistical significance or a new theoretical guarantee.',
    },
    'mechanism_diagnostic': 'Separately check whether geometry weights beat the fixed permutation on full SW; one permutation and seed cannot establish specificity.',
    'direct_comparison': 'Report full/free/dependent SW and utility against same-width same-steps Direct; not a validation-selected strongest Direct.',
    'notes': ['Fresh training seed relative to the preceding 101--105 pilots; same historically used split3101.',
              'No held-out test metrics in this pilot. Earlier C1 test results motivated the hypothesis, so the research remains exploratory.',
              'Only original eleven fields; no newly appended amortization columns.',
              'Three OS variants share architecture, initialization, transforms, sampling and reconstruction; only fixed loss weights differ.',
              'Same width and optimization steps are not parameter-count or wall-time matching against Direct.',
              'No integer/FICO lattice projection; structural equation validity is narrower than complete business validity.',
              'All four outcomes retained. No multi-seed expansion or B noise experiment in this invocation.',
              'Training losses with different weights are not directly comparable performance measures.'],
    'timeout_seconds': 600,
}


def tensor_hash(*values):
    h = hashlib.sha256()
    for value in values:
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def derive_weights(b):
    x = b.train_x.cpu().numpy().astype(float)
    sd = x.std(0).clip(1e-6)
    _, j = diag.lc_phi_and_jac(b.train_df)
    jv = j/sd[8:][None, :, None]
    gu = np.diag(1/sd[:8]**2) + np.einsum('nvi,nvj->ij', jv, jv)/len(jv)
    norm = pilot.make_free_flow_normalizer(b.train_x[:, :8], whiten_shrinkage=1.)
    a = np.diag(norm.std.detach().cpu().numpy().reshape(-1).astype(float)) @ norm.W_inv.detach().cpu().numpy().astype(float)
    gz = a.T @ gu @ a
    diagonal = np.diag(gz); normalized = diagonal/diagonal.mean()
    shrinkage = PROTOCOL['weight_definition']['shrinkage']
    shrunk = (1-shrinkage)+shrinkage*normalized
    cap = PROTOCOL['weight_definition']['max_weight_ratio']*shrunk.min()
    capped = np.minimum(shrunk, cap)
    weights = (capped/capped.mean()).astype(np.float32)
    permuted = weights[PROTOCOL['weight_definition']['permutation']]
    assert np.all(weights > 0) and abs(float(weights.mean())-1) < 1e-6
    assert weights.max()/weights.min() <= 4+1e-6
    assert np.array_equal(np.sort(weights), np.sort(permuted))
    assert not np.array_equal(weights, permuted)
    jstd = j*sd[:8][None, None, :]/sd[8:][None, :, None]
    energy = (jstd*jstd).sum((1, 2)); top = int(np.ceil(.01*len(energy)))
    # Independent quadratic-form identity checks the coordinate pullback.
    e = np.random.default_rng(7106).normal(size=(12, 8))
    left = np.einsum('ni,ij,nj->n', e, gz, e)
    du = e @ a.T
    right = np.einsum('ni,ij,nj->n', du, gu, du)
    pullback_error = float(np.max(np.abs(left-right)/(1+np.abs(right))))
    assert pullback_error < 1e-12
    st = diag.sensitivity(b.train_df, sd)
    return {'free_columns': b.cols[:8], 'metric_train_std': sd.tolist(), 'raw_diagonal_Gz': diagonal.tolist(),
            'normalized_diagonal': normalized.tolist(), 'geometry_weights': weights.tolist(),
            'permuted_weights': permuted.tolist(), 'permutation': PROTOCOL['weight_definition']['permutation'],
            'weight_ratio': float(weights.max()/weights.min()), 'cap_active': bool(np.any(shrunk > cap)),
            'Gz_eigenvalues': np.linalg.eigvalsh(gz).tolist(),
            'top_one_percent_jacobian_energy_share': float(np.sort(energy)[-top:].sum()/energy.sum()),
            'coordinate_pullback_relative_error': pullback_error,
            'tail_thresholds': {str(q): float(np.quantile(st, q)) for q in PROTOCOL['tail_quantiles']}}


def arithmetic_checks():
    result = diag.checks()
    g = torch.Generator(device='cpu').manual_seed(7106)
    err = torch.randn(16, 8, generator=g, dtype=torch.float64, requires_grad=True)
    plain = err.square().mean()
    weighted = (err.square()*torch.ones(8, dtype=torch.float64)).mean()
    grad_a = torch.autograd.grad(plain, err, retain_graph=True)[0]
    grad_b = torch.autograd.grad(weighted, err)[0]
    assert plain.item() == weighted.item() and torch.equal(grad_a, grad_b)
    result.update(uniform_weights_loss_and_gradient_exact=True)
    return result


def provenance():
    main = json.loads((MAIN/'provenance.json').read_text())
    sources = dict(main['sources'])
    for name, digest in sources.items():
        assert nc.sha(ROOT/name) == digest, name
    old_pilot = json.loads((ROOT/'outputs/neuro_redundancy_pilot_s101_v1/provenance.json').read_text())
    for name, digest in old_pilot['sources'].items():
        if Path(name).name == 'neurocomputing_redundancy_pilot.py': assert nc.sha(ROOT/name) == digest
    c1 = json.loads((ROOT/'outputs/neurocomputing_advantage_c1_v1/provenance.json').read_text())
    for name, digest in c1['analysis_source'].items(): assert nc.sha(ROOT/name) == digest
    for path in [Path(pilot.__file__), Path(diag.__file__), Path(__file__)]:
        sources[str(path.relative_to(ROOT))] = nc.sha(path)
    folder = MAIN/'split3101/data/lendingclub'
    manifest = json.loads((folder/'manifest.json').read_text())
    for name in ['train.csv', 'val.csv']:
        assert nc.sha(folder/name) == manifest['files'][name]
    paths = [folder/'train.csv', folder/'val.csv', folder/'manifest.json',
             ROOT/'outputs/neuro_redundancy_pilot_s101_v1/provenance.json',
             ROOT/'outputs/neurocomputing_advantage_c1_v1/provenance.json']
    paper = json.loads((ROOT/'outputs/manuscript_revision_neurocomputing_20261103/delivery_manifest.json').read_text())
    for name in ['paper.tex', 'appendix.tex']:
        path = ROOT/'paper-v6'/name; assert nc.sha(path) == paper['source_files'][name]; paths.append(path)
    return {'parent_main_identity': main['identity'], 'sources': sources,
            'inputs': {str(p.relative_to(ROOT)): nc.sha(p) for p in paths}, 'protocol': PROTOCOL}


def prepare(out, b):
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    checks = arithmetic_checks(); weights = derive_weights(b); identity = provenance()
    out.mkdir(parents=True)
    nc.dump(out/'protocol.json', PROTOCOL); nc.dump(out/'provenance.json', identity)
    nc.dump(out/'weights.json', weights); nc.dump(out/'numerical_checks.json', checks)
    for name in identity['sources']:
        dest = out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    nc.dump(out/'STATUS.json', {'status': 'PREPARED', 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
    print('Prepared one-seed four-arm pilot:', out, flush=True)
    print('Geometry weights:', weights['geometry_weights'], flush=True)


def fit(target, b, weights):
    # Same normalization, network and stochastic streams as frozen pilot.fit.
    nc.sync(); started = time.perf_counter(); nc.seed(PROTOCOL['training_seed'])
    norm = pilot.make_free_flow_normalizer(target, whiten_shrinkage=1.)
    z = norm.transform(target); kh = b.spec.one_hot(b.train_k)
    model = pilot.ConditionalVelocity(z.shape[1], kh.shape[1], PROTOCOL['width']).to(nc.DEVICE)
    initial_hash = tensor_hash(*model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['lr'])
    streams = [pilot.generator(offset) for offset in (1000, 2000, 3000, 4000)]
    w = torch.tensor(weights, device=nc.DEVICE, dtype=z.dtype)
    assert len(w) == z.shape[1]
    history, probes = [], []
    for step in range(PROTOCOL['steps']):
        idx = torch.randint(len(z), (PROTOCOL['batch_size'],), device=nc.DEVICE, generator=streams[0])
        t = torch.rand((len(idx), 1), device=nc.DEVICE, generator=streams[1])
        free_noise = torch.randn((len(idx), 8), device=nc.DEVICE, generator=streams[2])
        dep_noise = torch.randn((len(idx), z.shape[1]-8), device=nc.DEVICE, generator=streams[3])
        noise = torch.cat([free_noise, dep_noise], 1); y = z[idx]
        squared = (model((1-t)*noise+t*y, kh[idx], t)-(y-noise)).square()
        loss = (squared*w).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite weighted loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 100 == 0 or step == PROTOCOL['steps']-1:
            history.append({'step': step+1, 'weighted_loss': float(loss.detach()), 'plain_mse': float(squared.detach().mean())})
        if step in [0, 800, PROTOCOL['steps']-1]:
            probes.append({'step': step+1, 'indices_time_free_noise_sha256': tensor_hash(idx, t, free_noise)})
    nc.sync()
    metadata = {'fit_wall_seconds': time.perf_counter()-started,
                'parameters': sum(p.numel() for p in model.parameters()), 'flow_dimension': z.shape[1],
                'initial_parameters_sha256': initial_hash, 'paired_training_probes': probes}
    return model, norm, metadata, history


def tail_metrics(b, x, k, weights, method):
    train_sd = np.array(weights['metric_train_std'])
    real, gen = b.ref_df, nc.frame(b, x, k)
    sr, sg = diag.sensitivity(real, train_sd), diag.sensitivity(gen, train_sd)
    tr = b.train_x.cpu().numpy().astype(float)
    r = b.ref_x.cpu().numpy().astype(float); a = x.cpu().numpy().astype(float)
    mu = tr.mean(0); ar, aa = (r-mu)/train_sd, (a-mu)/train_sd
    projectors = {'full': diag.directions(11, 17), 'free': diag.directions(8, 19)}
    result = []
    for quantile in PROTOCOL['tail_quantiles']:
        threshold = weights['tail_thresholds'][str(quantile)]
        ir, ig = sr >= threshold, sg >= threshold
        row = {'method': method, 'tail_fraction': 1-quantile, 'threshold': threshold,
               'n_reference': int(ir.sum()), 'n_generated': int(ig.sum()),
               'reference_mass': float(ir.mean()), 'generated_mass': float(ig.mean()),
               'mass_difference': float(ig.mean()-ir.mean()), 'absolute_mass_error': abs(float(ig.mean()-ir.mean()))}
        for block, dirs in projectors.items():
            d = dirs.shape[0]
            row[f'{block}_sw'] = diag.projected_sw(ar[ir, :d]@dirs, aa[ig, :d]@dirs)
        row['quantile_mae_std'] = float(np.abs(np.quantile(ar[ir], [.1, .5, .9], axis=0)-
                                             np.quantile(aa[ig], [.1, .5, .9], axis=0)).mean()) if ig.any() else None
        result.append(row)
    return result


def run(out, b):
    assert json.loads((out/'STATUS.json').read_text())['status'] == 'PREPARED'
    frozen = json.loads((out/'provenance.json').read_text()); assert frozen == provenance()
    assert json.loads((out/'protocol.json').read_text()) == PROTOCOL
    weights = json.loads((out/'weights.json').read_text()); assert weights == derive_weights(b)
    helper_config = copy.deepcopy(pilot.CONFIG)
    started = time.time(); records, tails, replays = [], [], []
    nc.dump(out/'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'completed_fits': 0, 'total': 4})
    try:
        pilot.CONFIG.update(training_seed=106, steps=1600, sample_steps=80)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        nc.dump(out/'environment.json', {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                                       'platform': platform.platform(), 'gpu_start': pilot.gpu_snapshot()})
        assert len(b.train_x) == 20000 and len(b.ref_x) == 5000
        k = nc.empirical_k(b, 5000, 106+500)
        for folder in ['samples', 'checkpoints']: (out/folder).mkdir()
        for method in METHODS:
            if time.time()-started > PROTOCOL['timeout_seconds']: raise TimeoutError('Pilot timeout')
            nc.event(out, stage='fit_start', method=method, completed_fits=len(records))
            is_direct = method == 'direct_repair'
            target = b.train_x if is_direct else b.train_x[:, :8]
            w = weights['geometry_weights'] if method == 'os_jacobian' else weights['permuted_weights'] if method == 'os_permuted' else [1.]*target.shape[1]
            model, norm, meta, history = fit(target, b, w)
            checkpoint = out/'checkpoints'/f'{method}.pt'
            torch.save({'model': model.state_dict(), 'normalizer': norm, 'weights': w, 'metadata': meta,
                        'protocol': PROTOCOL}, checkpoint)
            nc.dump(out/'checkpoints'/f'{method}.history.json', history)
            x = pilot.sample(model, norm, b, k)
            saved = torch.load(checkpoint, map_location=nc.DEVICE, weights_only=False)
            copy_model = pilot.ConditionalVelocity(model.dim, b.spec.onehot_dim, 128).to(nc.DEVICE)
            copy_model.load_state_dict(saved['model'])
            replay = pilot.sample(copy_model, saved['normalizer'], b, k)
            error = float((x-replay).abs().max()); assert error == 0
            replays.append({'method': method, 'max_checkpoint_sampling_replay_error': error})
            del copy_model, saved, replay
            file = out/'samples'/f'{method}.npz'
            np.savez_compressed(file, x=x.cpu().numpy(), k=k.cpu().numpy())
            nc.frame(b, x, k).to_csv(out/'samples'/f'{method}.csv', index=False)
            measured = nc.metrics(b, x, k); assert measured['violation_rate'] == 0
            util = pilot.utility(b, nc.frame(b, x, k), method, 106)
            nc.dump(out/'samples'/f'{method}.utility.json', util)
            by_predictor = {r['utility_model']: r for r in util}
            record = {'method': method, 'split': 3101, 'seed': 106, 'reference': 'val',
                      'weights': w, 'checkpoint': str(checkpoint.relative_to(out)),
                      'sample_file': str(file.relative_to(out)), 'sample_sha256': nc.sha(file),
                      **meta, **measured, 'original11_sw': measured['sw'], 'original3_dependent_sw': measured['dep_sw'],
                      'rf_auc': by_predictor['rf']['roc_auc'], 'linear_auc': by_predictor['linear']['roc_auc']}
            nc.dump(out/'samples'/f'{method}.metrics.json', record)
            records.append(record); tails += tail_metrics(b, x, k, weights, method)
            nc.dump(out/'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'completed_fits': len(records),
                                       'total': 4, 'elapsed_seconds': time.time()-started})
            nc.event(out, stage='fit_complete', method=method, full_sw=record['sw'], free_sw=record['free_sw'], rf_auc=record['rf_auc'])
            del model, norm
        assert len(records) == 4
        # Check all stored numerical metrics and identical paired training inputs.
        replay_errors = []
        for r in records:
            file = out/r['sample_file']; assert nc.sha(file) == r['sample_sha256']
            saved = np.load(file)
            x, kk = torch.from_numpy(saved['x']).to(nc.DEVICE), torch.from_numpy(saved['k']).to(nc.DEVICE)
            metric = nc.metrics(b, x, kk)
            replay_errors += [abs(metric[key]-r[key]) for key in metric]
            assert torch.equal(kk, k)
        assert max(replay_errors) == 0
        assert len({r['initial_parameters_sha256'] for r in records[:3]}) == 1
        assert all(r['paired_training_probes'] == records[0]['paired_training_probes'] for r in records)
        flat = [{k: v for k, v in r.items() if k not in ['weights', 'paired_training_probes']} for r in records]
        pd.DataFrame(flat).to_csv(out/'metrics.csv', index=False)
        pd.DataFrame(tails).to_csv(out/'tail_metrics.csv', index=False)
        pilot.CONFIG.clear(); pilot.CONFIG.update(helper_config)
        assert frozen == provenance()
        nc.dump(out/'audit.json', {'status': 'PASS', 'checkpoint_replays': replays, 'metric_rows_replayed': 4,
                                  'max_metric_replay_error': max(replay_errors),
                                  'three_os_initializations_identical': True, 'four_arms_paired_training_probes_identical': True,
                                  'category_tuples_identical': True, 'frozen_sources_inputs_manuscripts_unchanged': True,
                                  'scope': 'Checkpoint sampling and numerical metrics replayed; no independent complete retraining or utility refit.'})
        report(out)
        nc.dump(out/'STATUS.json', {'status': 'COMPLETE', 'trained_models': 4, 'metric_rows': 4,
                                   'duration_seconds': time.time()-started, 'gpu_end': pilot.gpu_snapshot()})
        nc.dump(out/'report_manifest.json', {'files': {p.name: nc.sha(p) for p in out.iterdir()
                                                       if p.is_file() and p.name != 'report_manifest.json'}})
        print('Jacobian pilot complete:', out, flush=True)
    except BaseException as e:
        nc.dump(out/'STATUS.json', {'status': 'FAILED', 'error': repr(e), 'completed_fits': len(records),
                                   'duration_seconds': time.time()-started})
        raise
    finally:
        pilot.CONFIG.clear(); pilot.CONFIG.update(helper_config)


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = pd.read_csv(out/'metrics.csv').set_index('method')
    base, geo, perm, direct = (f.loc[m] for m in METHODS)
    comparisons = []
    for comparator in ['osscf', 'os_permuted', 'direct_repair']:
        other = f.loc[comparator]
        comparisons.append({'comparison': 'os_jacobian versus '+comparator,
                            **{key+'_relative_gain': 1-geo[key]/other[key] for key in ['original11_sw', 'original3_dependent_sw', 'free_sw']},
                            'rf_auc_difference': geo.rf_auc-other.rf_auc, 'linear_auc_difference': geo.linear_auc-other.linear_auc})
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv', index=False)
    gates = {'full_sw_improvement_ge_2_percent_vs_os': bool(geo.original11_sw <= .98*base.original11_sw),
             'dependent_sw_no_worse_vs_os': bool(geo.original3_dependent_sw <= base.original3_dependent_sw),
             'free_sw_worsening_le_5_percent_vs_os': bool(geo.free_sw <= 1.05*base.free_sw),
             'rf_auc_drop_le_0_005_vs_os': bool(geo.rf_auc >= base.rf_auc-.005),
             'all_formula_checks_pass': bool((f.violation_rate == 0).all())}
    geometry_better = bool(geo.original11_sw < perm.original11_sw)
    status = ('MULTISEED_CANDIDATE_ONLY' if geometry_better else 'WEIGHTING_SIGNAL_NOT_GEOMETRY_SPECIFIC') if all(gates.values()) else 'NO_CLEAR_GAIN_IN_THIS_PILOT'
    nc.dump(out/'decision.json', {'status': status, 'benefit_screen': gates,
                                  'geometry_full_sw_better_than_fixed_permutation': geometry_better,
                                  'geometry_full_sw_better_than_direct': bool(geo.original11_sw < direct.original11_sw),
                                  'scope': 'One-seed descriptive screening, no significance or generalization claim.'})
    display = ['OS original', 'OS geometry', 'OS permuted', 'Direct + reconstruction']
    colors = ['#6657ba', '#2a9d8f', '#d7a344', '#4d83ab']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, metric, label in zip(axes, ['original11_sw', 'original3_dependent_sw', 'free_sw', 'rf_auc'],
                                ['Original-11 SW (lower)', 'Dependent SW (lower)', 'Free SW (lower)', 'RF AUC (higher)']):
        vals = f.reindex(METHODS)[metric]
        ax.bar(range(4), vals, color=colors)
        ax.set_xticks(range(4), display, rotation=30, ha='right', fontsize=8); ax.set_ylabel(label)
        for i, value in enumerate(vals): ax.text(i, value, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, vals.max()*1.12); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('LendingClub split3101, seed106; validation only; 1600 steps; alpha=1', fontsize=11)
    fig.tight_layout(); fig.savefig(out/'pilot_comparison.png', dpi=170); fig.savefig(out/'pilot_comparison.pdf'); plt.close(fig)
    tail = pd.read_csv(out/'tail_metrics.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, metric, ylabel in zip(axes, ['generated_mass', 'full_sw', 'free_sw'],
                                 ['Region probability mass', 'Within-region full SW', 'Within-region free SW']):
        for method, label, color in zip(METHODS, display, colors):
            q = tail[tail.method == method].sort_values('tail_fraction')
            ax.plot(q.tail_fraction*100, q[metric], 'o-', label=label, color=color)
        if metric == 'generated_mass':
            ax.plot(q.tail_fraction*100, q.reference_mass, 'k--', label='Real validation mass')
        ax.set_xlabel('Training-defined top sensitivity region (%)'); ax.set_xticks([10, 20]); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('One-seed secondary tail diagnostics; nested regions, no significance inference')
    fig.tight_layout(); fig.savefig(out/'pilot_tail_diagnostics.png', dpi=170); fig.savefig(out/'pilot_tail_diagnostics.pdf'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--run', action='store_true')
    group.add_argument('--check-only', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(arithmetic_checks(), indent=2)); return
    b = nc.bundle('lendingclub', 3101, MAIN, reference='val')
    if args.prepare: prepare(args.out.resolve(), b)
    else: run(args.out.resolve(), b)


if __name__ == '__main__': main()
