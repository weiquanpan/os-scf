"""Frozen, post-hoc C1 diagnostics on saved samples; no model training.

Prepare before run. Inputs and the main study remain read-only.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'outputs/neurocomputing_v2'
DEFAULT_OUT = ROOT / 'outputs/neurocomputing_advantage_c1_v1'
METHODS = ['osscf', 'direct_repair', 'free_tabsyn', 'tabsyn_repair']
ALL_METHODS = METHODS + ['train_bootstrap']
SPECS = {
    'lendingclub': {
        'num': ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low', 'open_acc',
                'revol_util', 'total_acc', 'installment', 'loan_income_ratio', 'fico_range_high'],
        'cat': ['term_months', 'grade', 'home_ownership', 'verification_status', 'purpose',
                'application_type', 'loan_status'], 'free': 8, 'target': 'loan_status',
        'predictor_num': ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low', 'open_acc',
                          'revol_util', 'total_acc', 'installment', 'loan_income_ratio', 'fico_range_high'],
    },
    'bikesharing': {
        'num': ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt'],
        'cat': ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit'],
        'free': 6, 'target': 'cnt', 'predictor_num': ['temp', 'atemp', 'hum', 'windspeed'],
    },
    'taxi': {
        'num': ['trip_distance', 'duration_minutes', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 'airport_fee', 'total_amount'],
        'cat': ['VendorID', 'RatecodeID', 'payment_type', 'passenger_count', 'pickup_hour',
                'pickup_weekday', 'PULocationID', 'DOLocationID'],
        'free': 10, 'target': 'total_amount', 'predictor_num': ['trip_distance', 'duration_minutes'],
    },
}
PROTOCOL = {
    'version': 'c1-posthoc-v1', 'analysis_status': 'post-hoc exploratory analysis of already evaluated test samples',
    'datasets': list(SPECS), 'splits': [3101, 3102, 3103], 'seeds': [101, 102, 103, 104, 105],
    'methods': METHODS, 'reference_control': 'train_bootstrap',
    'expected_model_sample_files': 180, 'expected_control_sample_files': 45,
    'groups': {
        'lendingclub': 'grade in {A,B,C,D,E-G} crossed with term_months; other raw labels retained',
        'bikesharing': 'season crossed with floor(hr/6) crossed with workingday; four fixed six-hour bins',
        'taxi': 'VendorID crossed with payment_type; raw labels retained',
    },
    'primary': 'reference-mass-weighted mean within-group full numerical SW (not a W2 theoretical bound)',
    'conditional_distance': '128 fixed projections; exact empirical 1D W2 via merged quantile intervals for unequal counts; root mean squared over projections',
    'weights': 'real test group proportions; no renormalization over missing generated groups',
    'missing_group_policy': 'primary conditional distance is undefined if any reference group is absent; report missing mass and TV',
    'small_group_policy': 'retain all nonempty groups in primary, flag counts below 30; also report shared subset with reference and ALL FOUR methods each >=30',
    'shared_subset_weights': 'renormalize reference weights on exactly the same eligible groups, report retained reference mass',
    'scaling': 'float32 round-trip of numeric CSV columns, then float64; real TRAIN mean/population std floor 1e-6; no within-group scaling',
    'projection_seeds': {'full': 17, 'free': 19, 'predictor_target_joint': 29}, 'projections': 128,
    'finite_sample_reference': '20 paired independent bootstrap draws from each real TRAIN group, sized to real-test and generated group counts; descriptive benchmark, not a debiased estimator or confidence interval',
    'bootstrap_replicates': 20,
    'relations': 'only actual TSTR predictors; Pearson/Spearman target correlation MAE; per-categorical-feature reference-weighted target-mean error divided by real train target std; report missing-level coverage',
    'joint_sw': 'numeric TSTR predictors plus target (charged_off=1 for LC), train-standardized; unconditional and group-conditional',
    'tail_diagnostic': 'LC only; train 80th/90th percentiles of operator norm of diag(1/std_v) J_phi diag(std_u), guards included, quantile box projection excluded; apply same thresholds to real and synthetic; report region mass, free/full SW and marginal quantile error',
    'tail_quantiles': [0.8, 0.9], 'marginal_quantiles': [0.1, 0.5, 0.9],
    'utility': 'reuse saved RF and linear TSTR scores; do not refit predictors',
    'aggregation': 'mean over 5 seeds within split, then equally average 3 overlapping splits; split SD descriptive only; paired win counts not independent Bernoulli trials',
    'limits': ['No generator training, new sampling, checkpoint selection, or B noise experiment.',
               'Coarse groups are not full categorical conditioning. Numerical SW is not full W2.',
               'Associations and subgroup metrics do not establish the cause of utility differences.',
               'No significance tests, causal claims, or selection of favorable groups or seeds.',
               'No learned conditional mutual information estimator.',
               'Frozen main-study and manuscript files remain unchanged.'],
    'hard_timeout_seconds': 1800,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def labels(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.map(lambda x: str(int(x)) if pd.notna(x) and float(x).is_integer() else str(x)).to_numpy(str)
    return series.fillna('__MISSING__').astype(str).str.strip().to_numpy(str)


def group_labels(frame, dataset):
    if dataset == 'lendingclub':
        grade = labels(frame.grade)
        grade = np.where(np.isin(grade, ['E', 'F', 'G']), 'E-G', grade)
        cols = [grade, labels(frame.term_months)]
    elif dataset == 'bikesharing':
        cols = [labels(frame.season), (frame.hr.to_numpy(float)//6).astype(int).astype(str), labels(frame.workingday)]
    else:
        cols = [labels(frame.VendorID), labels(frame.payment_type)]
    return np.array(['|'.join(row) for row in zip(*cols)])


def numeric(frame, cols):
    a = frame[cols].to_numpy(np.float32).astype(np.float64)
    assert np.isfinite(a).all(), cols
    return a


def target(frame, spec):
    if spec['target'] == 'loan_status':
        values = labels(frame.loan_status)
        assert set(values) <= {'charged_off', 'fully_paid'}, set(values)
        return (values == 'charged_off').astype(float)
    return numeric(frame, [spec['target']])[:, 0]


def directions(d, seed):
    a = np.random.default_rng(seed).normal(size=(d, 128))
    return a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-12)


def projected_sw(a, b):
    """Exact empirical quantile integration for each fixed projection."""
    n, m = len(a), len(b)
    if not n or not m:
        return float('nan')
    aa, bb = np.sort(a, axis=0), np.sort(b, axis=0)
    if n == m:
        return float(np.sqrt(np.mean((aa-bb)**2)))
    edges = np.unique(np.r_[np.arange(n+1)/n, np.arange(m+1)/m])
    mid = (edges[:-1]+edges[1:])/2
    ia = np.minimum((mid*n).astype(int), n-1)
    ib = np.minimum((mid*m).astype(int), m-1)
    return float(np.sqrt(np.dot(np.diff(edges), np.mean((aa[ia]-bb[ib])**2, axis=1))))


def correlation(a, b):
    x, y = a-a.mean(), b-b.mean()
    den = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den > 0 else float('nan')


def lc_phi_and_jac(frame):
    a = numeric(frame, SPECS['lendingclub']['num'][:8])
    raw_p, raw_r, raw_y = a[:, 0], a[:, 1], a[:, 2]
    p, r, y = np.maximum(raw_p, 1), np.clip(raw_r, 0, 40)/1200, np.maximum(raw_y, 1)
    n = np.maximum(frame.term_months.to_numpy(float), 1)
    zero = np.abs(r) < 1e-8
    q = np.exp(-n*np.log1p(r)); den = -np.expm1(-n*np.log1p(r))
    safe = np.where(zero, 1, den)
    factor = np.where(zero, 1/n, r/safe)
    derivative = np.where(zero, 0, (safe-r*n*q/(1+r))/(safe**2))
    phi = np.column_stack([p*factor, p/y, a[:, 4]+4])
    jac = np.zeros((len(a), 3, 8))
    jac[:, 0, 0] = factor*(raw_p > 1)
    jac[:, 0, 1] = p*derivative/1200*((raw_r > 0)&(raw_r < 40))
    jac[:, 1, 0] = (raw_p > 1)/y
    jac[:, 1, 2] = -p/y**2*(raw_y > 1)
    jac[:, 2, 4] = 1
    return phi, jac


def sensitivity(frame, sd):
    _, j = lc_phi_and_jac(frame)
    scaled = j*sd[:8][None, None, :]/sd[8:][None, :, None]
    return np.linalg.svd(scaled, compute_uv=False)[:, 0]


def checks():
    rng = np.random.default_rng(77)
    errors = []
    for n, m in [(1, 3), (2, 3), (3, 5), (7, 7)]:
        a, b = rng.normal(size=(n, 4)), rng.normal(size=(m, 4))
        lcm = int(np.lcm(n, m))
        # Independent discrete equal-mass expansion, not merged interval indexing.
        ax = np.repeat(np.sort(a, axis=0), lcm//n, axis=0)
        bx = np.repeat(np.sort(b, axis=0), lcm//m, axis=0)
        expected = float(np.sqrt(np.mean((ax-bx)**2)))
        errors.append(abs(projected_sw(a, b)-expected))
        assert abs(projected_sw(a, b)-projected_sw(b, a)) < 1e-12
        assert projected_sw(a, a) == 0
        assert abs(projected_sw(a+3, b+3)-expected) < 1e-12
    assert max(errors) < 1e-12
    f = pd.DataFrame([[10000., 12., 50000., 20., 700., 8., 30., 15.],
                      [20000., 24., 75000., 10., 650., 4., 60., 10.]],
                     columns=SPECS['lendingclub']['num'][:8])
    f['term_months'] = [36, 60]
    _, jac = lc_phi_and_jac(f)
    max_relative = 0.
    # Float32 input conversion is retained; use representable finite-difference steps.
    for j, col in enumerate(SPECS['lendingclub']['num'][:8]):
        step = 0.125 if col == 'int_rate' else 1.
        lo, hi = f.copy(), f.copy(); lo[col] -= step; hi[col] += step
        numerical = (lc_phi_and_jac(hi)[0]-lc_phi_and_jac(lo)[0])/(2*step)
        err = np.max(np.abs(numerical-jac[:, :, j])/(1+np.abs(jac[:, :, j])))
        max_relative = max(max_relative, float(err))
    assert max_relative < 2e-5, max_relative
    return {'status': 'PASS', 'unequal_sw_max_error_against_mass_expansion': max(errors),
            'jacobian_max_scaled_finite_difference_error': max_relative}


def input_paths():
    paths = [MAIN/'provenance.json', MAIN/'summary/metrics_per_run.csv', MAIN/'summary/utility_per_run.csv',
             ROOT/'paper-v6/paper.tex', ROOT/'paper-v6/appendix.tex']
    for ds in SPECS:
        for split in PROTOCOL['splits']:
            folder = MAIN/f'split{split}/data/{ds}'
            paths += [folder/'train.csv', folder/'test.csv', folder/'manifest.json']
            manifest = json.loads((folder/'manifest.json').read_text())
            for name in ['train.csv', 'test.csv']:
                assert sha(folder/name) == manifest['files'][name]
            for seed in PROTOCOL['seeds']:
                folder = MAIN/f'split{split}/final/{ds}/seed{seed}'
                for method in ALL_METHODS:
                    p, meta = folder/f'{method}.csv', folder/f'{method}.metrics.json'
                    record = json.loads(meta.read_text())
                    assert record['reference'] == 'test'
                    assert sha(p) == record['sample_sha256'], p
                    paths += [p, meta]
                    if method in METHODS:
                        paths.append(folder/f'{method}.utility.json')
    return paths


def prepare(out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    check = checks()
    main = json.loads((MAIN/'provenance.json').read_text())
    for name, digest in main['sources'].items():
        assert sha(ROOT/name) == digest, name
    paths = input_paths()
    record = {'parent_identity': main['identity'], 'main_sources': main['sources'],
              'analysis_source': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
              'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths}}
    out.mkdir(parents=True)
    dump(out/'protocol.json', PROTOCOL); dump(out/'provenance.json', record)
    dump(out/'numerical_checks.json', check)
    shutil.copy2(__file__, out/Path(__file__).name)
    dump(out/'STATUS.json', {'status': 'PREPARED', 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
    print('Prepared C1 diagnostic; inputs:', len(paths), flush=True)


def verify_inputs(out):
    p = json.loads((out/'provenance.json').read_text())
    for section in ['main_sources', 'analysis_source', 'inputs']:
        for name, digest in p[section].items():
            assert sha(ROOT/name) == digest, (section, name)
    assert json.loads((out/'protocol.json').read_text()) == PROTOCOL
    return {key: len(p[key]) for key in ['main_sources', 'analysis_source', 'inputs']}


def association_metrics(train, real, gen, spec, identity):
    yr, yg, yt = target(real, spec), target(gen, spec), target(train, spec)
    sy = max(float(yt.std()), 1e-6)
    details, cat_details = [], []
    for col in spec['predictor_num']:
        xr, xg = numeric(real, [col])[:, 0], numeric(gen, [col])[:, 0]
        for kind in ['pearson', 'spearman']:
            a, b = (correlation(xr, yr), correlation(xg, yg)) if kind == 'pearson' else (
                correlation(rankdata(xr), rankdata(yr)), correlation(rankdata(xg), rankdata(yg)))
            details.append({**identity, 'feature': col, 'association': kind, 'real': a, 'generated': b,
                            'absolute_error': abs(a-b)})
    categorical_errors, coverages = [], []
    for col in spec['cat']:
        if col == spec['target']:
            continue
        ar, ag = labels(real[col]), labels(gen[col])
        coverage, error = 0., 0.
        for level in sorted(set(ar)):
            ir, ig = ar == level, ag == level
            mass = float(ir.mean()); nr, ng = int(ir.sum()), int(ig.sum())
            mr = float(yr[ir].mean()); mg = float(yg[ig].mean()) if ng else float('nan')
            e = abs(mr-mg)/sy
            if ng:
                coverage += mass; error += mass*e
            cat_details.append({**identity, 'feature': col, 'level': level, 'n_real': nr, 'n_generated': ng,
                                'reference_mass': mass, 'real_target_mean': mr, 'generated_target_mean': mg,
                                'absolute_standardized_mean_error': e})
        coverages.append(coverage)
        categorical_errors.append(error/coverage if coverage else float('nan'))
    d = pd.DataFrame(details)
    values = {f'target_{kind}_mae': float(d[d.association == kind].absolute_error.mean())
              for kind in ['pearson', 'spearman']}
    values.update(target_mean_error_std=abs(float(yg.mean()-yr.mean()))/sy,
                  target_categorical_mean_error_covered=float(np.mean(categorical_errors)),
                  target_categorical_mean_coverage=float(np.mean(coverages)),
                  target_categorical_min_coverage=float(np.min(coverages)),
                  undefined_numeric_associations=int(d.absolute_error.isna().sum()))
    # Strict total metric stays undefined when any feature has missing reference levels.
    values['target_categorical_mean_error_strict'] = (float(np.mean(categorical_errors))
                                                      if min(coverages) >= 1-1e-12 else float('nan'))
    return values, details, cat_details


def finite_sample_floor(train_proj, nr, ng, seed):
    vals = []
    for rep in range(PROTOCOL['bootstrap_replicates']):
        a = np.random.default_rng(stable_seed(seed, rep, 'a')).integers(len(train_proj), size=nr)
        b = np.random.default_rng(stable_seed(seed, rep, 'b')).integers(len(train_proj), size=ng)
        vals.append(projected_sw(train_proj[a], train_proj[b]))
    return np.array(vals)


def run(out):
    assert json.loads((out/'STATUS.json').read_text())['status'] == 'PREPARED'
    verify_inputs(out)
    started = time.time()
    dump(out/'environment.json', {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                                 'platform': platform.platform(), 'mode': 'CPU diagnostics only, no torch model fit'})
    dump(out/'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'completed_samples': 0, 'total': 225})
    rows, group_rows, numeric_relations, cat_relations, tails, threshold_rows = [], [], [], [], [], []
    replays, floor_cache = [], {}
    try:
        for ds, spec in SPECS.items():
            for split in PROTOCOL['splits']:
                data = MAIN/f'split{split}/data/{ds}'
                train, real = pd.read_csv(data/'train.csv'), pd.read_csv(data/'test.csv')
                tx, rx = numeric(train, spec['num']), numeric(real, spec['num'])
                mu, sd = tx.mean(0), tx.std(0).clip(1e-6)
                tj = np.column_stack([numeric(train, spec['predictor_num']), target(train, spec)])
                rj = np.column_stack([numeric(real, spec['predictor_num']), target(real, spec)])
                jm, js = tj.mean(0), tj.std(0).clip(1e-6)
                projectors = {'full': directions(len(spec['num']), 17), 'free': directions(spec['free'], 19),
                              'joint': directions(tj.shape[1], 29)}
                trproj = {'full': ((tx-mu)/sd)@projectors['full'],
                          'free': ((tx-mu)/sd)[:, :spec['free']]@projectors['free'],
                          'joint': ((tj-jm)/js)@projectors['joint']}
                rproj = {'full': ((rx-mu)/sd)@projectors['full'],
                         'free': ((rx-mu)/sd)[:, :spec['free']]@projectors['free'],
                         'joint': ((rj-jm)/js)@projectors['joint']}
                rt, tt = group_labels(real, ds), group_labels(train, ds)
                groups = sorted(set(rt)); real_counts = {g: int((rt == g).sum()) for g in groups}
                if ds == 'lendingclub':
                    sr = sensitivity(real, sd); st = sensitivity(train, sd)
                    thresholds = {q: float(np.quantile(st, q)) for q in PROTOCOL['tail_quantiles']}
                    for q, value in thresholds.items():
                        threshold_rows.append({'split': split, 'train_quantile': q, 'threshold': value,
                                               'training_region_mass': float((st >= value).mean()),
                                               'test_region_mass': float((sr >= value).mean())})
                for seed in PROTOCOL['seeds']:
                    folder = MAIN/f'split{split}/final/{ds}/seed{seed}'
                    frames = {m: pd.read_csv(folder/f'{m}.csv') for m in ALL_METHODS}
                    gl = {m: group_labels(f, ds) for m, f in frames.items()}
                    eligible = {g for g in groups if real_counts[g] >= 30 and
                                all(np.sum(gl[m] == g) >= 30 for m in METHODS)}
                    common_mass = sum(real_counts[g] for g in eligible)/len(real)
                    for method, gen in frames.items():
                        if time.time()-started > PROTOCOL['hard_timeout_seconds']:
                            raise TimeoutError('C1 fixed 1800-second timeout')
                        ident = {'dataset': ds, 'split': split, 'seed': seed, 'method': method}
                        gx = numeric(gen, spec['num'])
                        gj = np.column_stack([numeric(gen, spec['predictor_num']), target(gen, spec)])
                        gproj = {'full': ((gx-mu)/sd)@projectors['full'],
                                 'free': ((gx-mu)/sd)[:, :spec['free']]@projectors['free'],
                                 'joint': ((gj-jm)/js)@projectors['joint']}
                        record = {**ident, 'n_real': len(real), 'n_generated': len(gen)}
                        for block in projectors:
                            record[f'global_{block}_sw'] = projected_sw(rproj[block], gproj[block])
                        old = json.loads((folder/f'{method}.metrics.json').read_text())
                        err = max(abs(record['global_full_sw']-old['sw']), abs(record['global_free_sw']-old['free_sw']))
                        assert err < 2e-6, (ident, err)
                        replays.append({**ident, 'max_global_metric_difference': err})
                        union = set(rt) | set(gl[method])
                        record['group_tv'] = .5*sum(abs(np.mean(rt == g)-np.mean(gl[method] == g)) for g in union)
                        record['generated_unseen_group_mass'] = float(np.mean(~np.isin(gl[method], groups)))
                        totals = {b: 0. for b in projectors}; common = dict(totals)
                        missing = 0.; small = 0.; boot_total = np.zeros(PROTOCOL['bootstrap_replicates'])
                        for g in groups:
                            ri, gi, ti = rt == g, gl[method] == g, tt == g
                            nr, ng, nt = int(ri.sum()), int(gi.sum()), int(ti.sum())
                            weight = nr/len(real)
                            if ng == 0: missing += weight
                            if min(nr, ng) < 30: small += weight
                            gr = {**ident, 'group': g, 'n_train': nt, 'n_real': nr, 'n_generated': ng,
                                  'reference_mass': weight, 'shared_eligible': g in eligible}
                            for block in projectors:
                                dist = projected_sw(rproj[block][ri], gproj[block][gi])
                                gr[f'{block}_sw'] = dist
                                totals[block] += weight*dist
                                if g in eligible: common[block] += weight*dist
                            key = (ds, split, seed, g, nr, ng)
                            if nt and ng:
                                if key not in floor_cache:
                                    floor_cache[key] = finite_sample_floor(trproj['full'][ti], nr, ng,
                                                                           stable_seed(ds, split, seed, g))
                                vals = floor_cache[key]
                                gr['bootstrap_full_sw_mean'] = float(vals.mean())
                                boot_total += weight*vals
                            else:
                                gr['bootstrap_full_sw_mean'] = float('nan'); boot_total[:] = np.nan
                            group_rows.append(gr)
                        record.update({f'conditional_{b}_sw': value for b, value in totals.items()})
                        record.update({f'conditional_{b}_sw_common': value/common_mass if common_mass else float('nan')
                                       for b, value in common.items()})
                        record.update(missing_reference_group_mass=missing, sparse_reference_mass=small,
                                      common_reference_mass=common_mass,
                                      conditional_full_sw_bootstrap_mean=float(boot_total.mean()),
                                      conditional_full_sw_bootstrap_sd=float(boot_total.std(ddof=1)))
                        rel, nums, cats = association_metrics(train, real, gen, spec, ident)
                        record.update(rel); numeric_relations += nums; cat_relations += cats
                        if method in METHODS:
                            utilities = json.loads((folder/f'{method}.utility.json').read_text())
                            metric = 'roc_auc' if ds == 'lendingclub' else 'r2'
                            for u in utilities:
                                record[f'{u["utility_model"]}_score'] = u[metric]
                            record['utility_metric'] = metric
                        rows.append(record)
                        if ds == 'lendingclub':
                            sg = sensitivity(gen, sd)
                            for q, threshold in thresholds.items():
                                ir, ig = sr >= threshold, sg >= threshold
                                ar, ag = (rx-mu)/sd, (gx-mu)/sd
                                qt = float(np.abs(np.quantile(ar[ir], PROTOCOL['marginal_quantiles'], axis=0)-
                                                 np.quantile(ag[ig], PROTOCOL['marginal_quantiles'], axis=0)).mean()) if ig.any() else float('nan')
                                tails.append({**ident, 'tail_fraction': 1-q, 'threshold': threshold,
                                              'n_real': int(ir.sum()), 'n_generated': int(ig.sum()),
                                              'real_mass': float(ir.mean()), 'generated_mass': float(ig.mean()),
                                              'mass_difference': float(ig.mean()-ir.mean()),
                                              'absolute_mass_error': abs(float(ig.mean()-ir.mean())),
                                              'full_sw': projected_sw(rproj['full'][ir], gproj['full'][ig]),
                                              'free_sw': projected_sw(rproj['free'][ir], gproj['free'][ig]),
                                              'quantile_mae_std': qt})
                        dump(out/'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(),
                                                'completed_samples': len(rows), 'total': 225,
                                                'elapsed_seconds': time.time()-started, 'last': ident})
                    print(f'C1 {ds} split{split} seed{seed}: {len(rows)}/225 samples', flush=True)
        assert len(rows) == 225 and sum(r['method'] in METHODS for r in rows) == 180
        tables = {'per_sample.csv': rows, 'per_group.csv': group_rows, 'numeric_target_associations.csv': numeric_relations,
                  'categorical_target_means.csv': cat_relations, 'lendingclub_tail_metrics.csv': tails,
                  'lendingclub_tail_thresholds.csv': threshold_rows, 'global_metric_replay.csv': replays}
        for name, records in tables.items():
            pd.DataFrame(records).to_csv(out/name, index=False)
        # Reconstruct every conditional aggregate from the saved group table.
        p, g = pd.read_csv(out/'per_sample.csv'), pd.read_csv(out/'per_group.csv')
        agg_error = 0.
        for row in p.itertuples():
            group = g[(g.dataset == row.dataset)&(g.split == row.split)&(g.seed == row.seed)&(g.method == row.method)]
            for block in ['full', 'free', 'joint']:
                values = group[f'{block}_sw']
                total = float(np.dot(group.reference_mass, values))
                saved = getattr(row, f'conditional_{block}_sw')
                if np.isfinite(total): agg_error = max(agg_error, abs(total-saved))
                else: assert np.isnan(saved)
        assert agg_error < 1e-12
        counts = verify_inputs(out)
        dump(out/'audit.json', {'status': 'PASS', 'verified_files': counts, 'global_metric_rows_replayed': len(replays),
                               'max_global_replay_difference': max(r['max_global_metric_difference'] for r in replays),
                               'saved_group_aggregation_max_difference': agg_error, 'source_sample_hashes_verified': 225,
                               'new_training_runs': 0, 'new_predictor_fits': 0, 'main_sources_inputs_manuscripts_unchanged': True,
                               'scope': 'Independent unequal-count arithmetic checks; analytic Jacobian finite differences; original full/free scores replayed from CSV; no new full-training reproduction.'})
        summarize(out)
        dump(out/'STATUS.json', {'status': 'COMPLETE', 'samples_analyzed': len(rows), 'model_samples': 180,
                                'control_samples': 45, 'duration_seconds': time.time()-started})
        hashes = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'report_manifest.json'}
        dump(out/'report_manifest.json', {'files': hashes, 'interpretation': 'post-hoc diagnostics only'})
        print('C1 complete:', out, flush=True)
    except BaseException as e:
        dump(out/'STATUS.json', {'status': 'FAILED', 'error': repr(e), 'samples_completed': len(rows),
                                'elapsed_seconds': time.time()-started})
        raise


def summarize(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = pd.read_csv(out/'per_sample.csv')
    metric_cols = ['global_full_sw', 'global_free_sw', 'global_joint_sw', 'conditional_full_sw',
                   'conditional_free_sw', 'conditional_joint_sw', 'conditional_full_sw_common',
                   'group_tv', 'missing_reference_group_mass', 'sparse_reference_mass', 'common_reference_mass',
                   'conditional_full_sw_bootstrap_mean', 'target_pearson_mae', 'target_spearman_mae',
                   'target_categorical_mean_error_covered', 'target_categorical_mean_error_strict',
                   'target_categorical_mean_coverage', 'target_categorical_min_coverage',
                   'target_mean_error_std', 'rf_score', 'linear_score']
    within = p.groupby(['dataset', 'split', 'method'])[metric_cols].agg(['mean', 'std', 'count'])
    within.to_csv(out/'within_split.csv')
    split_means = p.groupby(['dataset', 'split', 'method'])[metric_cols].mean()
    split_means.to_csv(out/'split_means.csv')
    summary = split_means.groupby(['dataset', 'method']).agg(['mean', 'std', 'count'])
    summary.to_csv(out/'summary.csv')
    paired = []
    compare_metrics = ['global_full_sw', 'conditional_full_sw', 'conditional_free_sw', 'global_joint_sw',
                       'conditional_joint_sw', 'target_pearson_mae', 'target_spearman_mae',
                       'target_categorical_mean_error_covered', 'rf_score', 'linear_score']
    for ds in SPECS:
        a = p[(p.dataset == ds)&(p.method == 'osscf')].set_index(['split', 'seed'])
        for method in METHODS[1:]:
            b = p[(p.dataset == ds)&(p.method == method)].set_index(['split', 'seed'])
            for index in a.index:
                row = {'dataset': ds, 'split': index[0], 'seed': index[1], 'comparator': method}
                for metric in compare_metrics:
                    row[f'{metric}_os_minus_comparator'] = a.loc[index, metric]-b.loc[index, metric]
                row['both_conditional_and_rf_favor_os'] = (a.loc[index, 'conditional_full_sw'] < b.loc[index, 'conditional_full_sw'] and a.loc[index, 'rf_score'] > b.loc[index, 'rf_score'])
                paired.append(row)
    paired = pd.DataFrame(paired); paired.to_csv(out/'paired_differences.csv', index=False)
    comparisons = []
    for (ds, method), part in paired.groupby(['dataset', 'comparator']):
        row = {'dataset': ds, 'comparator': method, 'n_descriptive_pairs': len(part),
               'both_conditional_and_rf_favor_os_pairs': int(part.both_conditional_and_rf_favor_os.sum())}
        for metric in compare_metrics:
            vals = part[f'{metric}_os_minus_comparator']
            row[f'{metric}_mean_difference'] = float(vals.mean())
            row[f'{metric}_os_wins'] = int((vals > 0).sum() if metric.endswith('_score') else (vals < 0).sum())
        comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(out/'comparison.csv', index=False)
    tail = pd.read_csv(out/'lendingclub_tail_metrics.csv')
    tm = ['real_mass', 'generated_mass', 'mass_difference', 'absolute_mass_error', 'full_sw', 'free_sw', 'quantile_mae_std']
    tail.groupby(['split', 'tail_fraction', 'method'])[tm].mean().groupby(['tail_fraction', 'method']).agg(['mean', 'std']).to_csv(out/'tail_summary.csv')
    display = {'osscf': 'OS-SCF', 'direct_repair': 'Direct + reconstruction', 'free_tabsyn': 'Free-TabSyn',
               'tabsyn_repair': 'TabSyn + reconstruction', 'train_bootstrap': 'Train bootstrap'}
    colors = ['#6657ba', '#3b86b4', '#d99a34', '#52a28e', '#969696']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, ds in zip(axes, SPECS):
        q = summary.loc[ds].reindex(ALL_METHODS)
        y = q[('conditional_full_sw', 'mean')]; e = q[('conditional_full_sw', 'std')]
        ax.bar(np.arange(5), y, yerr=e, color=colors, capsize=3)
        ax.plot(np.arange(5), q[('conditional_full_sw_bootstrap_mean', 'mean')], 'kx', label='Train-pool finite-sample benchmark')
        ax.set_xticks(np.arange(5), [display[m] for m in ALL_METHODS], rotation=30, ha='right', fontsize=8)
        ax.set_title(ds); ax.set_ylabel('Weighted conditional full SW (lower is better)')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(fontsize=7)
    fig.suptitle('Post-hoc C1: coarse conditional fidelity; bars show SD across overlapping split means', fontsize=10)
    fig.tight_layout(); fig.savefig(out/'conditional_diagnostics.png', dpi=170); fig.savefig(out/'conditional_diagnostics.pdf'); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for col, ds in enumerate(SPECS):
        q = summary.loc[ds].reindex(ALL_METHODS)
        for row, metric in enumerate(['global_joint_sw', 'target_spearman_mae', 'target_categorical_mean_error_covered']):
            ax = axes[row, col]
            ax.bar(np.arange(5), q[(metric, 'mean')], yerr=q[(metric, 'std')], color=colors, capsize=3)
            ax.set_xticks(np.arange(5), [display[m] for m in ALL_METHODS], rotation=30, ha='right', fontsize=7)
            ax.set_title(ds if row == 0 else '')
            ax.set_ylabel({'global_joint_sw': 'Predictor-target joint SW', 'target_spearman_mae': 'Target Spearman MAE',
                           'target_categorical_mean_error_covered': 'Categorical target-mean error (covered levels)'}[metric], fontsize=8)
            ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Actual TSTR predictors only; association diagnostics do not establish causation', fontsize=11)
    fig.tight_layout(); fig.savefig(out/'utility_relation_diagnostics.png', dpi=170); fig.savefig(out/'utility_relation_diagnostics.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for j, metric in enumerate(['generated_mass', 'free_sw', 'quantile_mae_std']):
        ax = axes[j]
        for mi, method in enumerate(ALL_METHODS):
            q = tail[tail.method == method].groupby('tail_fraction')[metric].mean().sort_index()
            ax.plot(q.index*100, q.values, 'o-', color=colors[mi], label=display[method])
        if metric == 'generated_mass':
            q = tail.groupby('tail_fraction').real_mass.mean().sort_index()
            ax.plot(q.index*100, q.values, 'k--', label='Real test mass')
        ax.set_xlabel('Training-defined top sensitivity region (%)'); ax.set_xticks([10, 20])
        ax.set_ylabel({'generated_mass': 'Generated probability mass', 'free_sw': 'Within-region free SW',
                       'quantile_mae_std': 'Within-region marginal quantile MAE'}[metric]); ax.grid(alpha=.2)
    axes[2].legend(fontsize=7)
    fig.suptitle('LendingClub only: standardized reconstruction Jacobian; nested regions are not independent')
    fig.tight_layout(); fig.savefig(out/'lendingclub_sensitivity.png', dpi=170); fig.savefig(out/'lendingclub_sensitivity.pdf'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument('--prepare', action='store_true'); flags.add_argument('--run', action='store_true')
    flags.add_argument('--check-only', action='store_true')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.check_only: print(json.dumps(checks(), indent=2)); return
    if args.prepare: prepare(args.out.resolve())
    else: run(args.out.resolve())


if __name__ == '__main__':
    main()
