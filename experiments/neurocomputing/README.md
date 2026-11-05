# Neurocomputing revision experiments

This package leaves the historical P1/P2 experiments and paper-v6 untouched.
Run from `D:/Flux/flux2`, with `MKL_THREADING_LAYER=SEQUENTIAL` set before Python
starts. The launcher also sets it before importing numerical libraries.

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='4'
python -m experiments.neurocomputing.test_protocol
python -m experiments.neurocomputing.run --out outputs/neurocomputing_smoke --smoke --stage all
python -m experiments.neurocomputing_layout_audit --out outputs/neurocomputing_smoke
python -m experiments.neurocomputing.run --out outputs/neurocomputing_v2 --stage all
python -m experiments.neurocomputing_layout_audit --out outputs/neurocomputing_v2
```

For the complete sequential job with an archived v6 manuscript, stage logs,
30-second process monitoring and a 24-hour overall timeout:

```powershell
python -m experiments.neurocomputing_complete --out outputs/neurocomputing_v2
```

`RUN_STATUS.json` reports the active stage and worker PID. Do not start a second
driver for the same output directory. The driver stops on the first failure;
fixing code requires a new output root unless only reporting code is separately
run against the immutable evidence.

To finalize an already-trained run, add --finalize-only to that command.
The new coordinator supersedes the archived neurocomputing.execute driver.
The frozen training files remain unchanged. CSV audit inputs are explicitly
made row-contiguous, matching the generated tensors. This fixes layout-dependent
FP32 reductions at residual thresholds without editing metrics or loosening
tolerances. The initial audit is retained in audit_initial; all saved
samples/metric hashes are checked unchanged. audit/precision_sensitivity.csv
additionally reports FP64 residual sensitivity for every main result.

Stages: `prepare`, `pilot`, `flow`, `ablation`, `tabsyn`, `tabddpm`, `references`,
`summarize`, `all`. Dataset, split and training-seed filters select a subset of
the frozen protocol, e.g. `--datasets bikesharing --splits 3101 --seeds 101`.
Run sequentially on one GPU. Each successful run has a completion marker;
TabSyn saves optimizer, scheduler and RNG states every validation checkpoint.
Failed attempts remain in the output directory. TabDDPM retries restart an
unfinished fit because upstream does not expose optimizer checkpointing.

The code/config identity is frozen at first use of an output directory.
Changed training code requires a new output directory; stale results are never
silently reused. `source_snapshot` includes actual modified workspace files,
not merely a Git commit. A whole invocation completing is distinct from the
audit-created `COMPLETE.json` for the complete protocol.

## Protocol

Three group-disjoint train/validation/test splits per dataset, five training
seeds per split. Identical records stay together; repeated random splits still
overlap across repetitions and are NOT 15 independent datasets. Existing
LendingClub/BikeSharing data are historically used development pools.

Public categorical schema domains are fixed before splits. Other vocabularies
are training-only with an explicit unknown token. Test OOV rates are reported.
Numerical transformations and clipping bounds are training-only.
Full/free/dependent SW uses the respective numerical coordinates, standardized
with training-set means and standard deviations. It does not include categorical
one-hot vectors. Categorical TV and downstream utility must be read alongside SW.

Flow controls use the same category samples and a shared validation-selected
alpha/step pair minimizing average full SW of hard and post-hoc reconstructed
flows. Direct flow is a diagnostic from that same checkpoint, not a separately
optimized unconstrained competitor. The alpha ablation holds step counts fixed.

TabSyn is a **capped upstream-component adapter**, not a claim of exact official
reproduction: upstream Transformer VAE, 1024-wide denoiser, training-only
quantile normalization, beta/LR scheduling, validation checkpoint selection,
and explicit maximum epochs/early stopping. VAE validation uses MSE+CE instead
of upstream CE-only scheduling; beta decay can repeat every 10 stale epochs.
Caps are fixed before final test evaluation, and cap-limited runs are flagged.
Diffusion validation sampling preserves training RNG state. Full and reduced
variants have identical budget/selection rules. TabDDPM uses the upstream
training/sampling code, with integer-subsample sklearn compatibility patch;
its internal test placeholder contains validation rows only.
TabDDPM uses a 128-by-128 MLP, 100 diffusion time steps, and validation selection
between 3000 and 10000 training steps. This is a declared-budget upstream
configuration, not an exhaustive per-dataset reproduction of strong TabDDPM
performance. Poor fidelity under this configuration does not invalidate the
underlying method.

NYC January 2024 yellow-taxi benchmark: fixed quality domain (valid January
pickup, 1–180 minutes, .1–100 miles, cash/card, 1–6 passengers, nonnegative fee
fields, total <=1000). Vendor 1's `extra` includes congestion/airport charges;
its reconstruction subtracts these duplicated components. This is an empirical
source-specific accounting convention, not a universal taxi law. Residual-
violating records are retained and measured. 5000 development rows are set
aside; a deterministic independent sample of up to 100000 rows forms the pool.
Data source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

RF and linear TSTR use fixed hyperparameters and matched-size real references.
Bike target operands and taxi fee operands are excluded from predictors.
Rounding sensitivity applies lattice projection before dependent reconstruction.
Synchronized wall times may include other desktop GPU activity; they do not
support an exclusive-hardware speed claim.

## Evidence

`summary/metrics_per_run.csv`, `within_split.csv`, `between_split.csv`,
`paired_sw_by_split.csv`, `utility_per_run.csv`, `ablation_per_run.csv`.
`audit/report.json` verifies partition separation, hashes, metric replay,
structural guarantees and expected results. All errors and incomplete runs
remain visible. The audit does not establish model superiority or acceptance.

After the complete protocol passes the audit, run the independent post-hoc
raw-label categorical diagnostics and submission-oriented report:

```powershell
python -m experiments.neurocomputing_raw_category_audit --out outputs/neurocomputing_v2
python -m experiments.neurocomputing_replay_audit --out outputs/neurocomputing_v2
python -m experiments.neurocomputing_report --out outputs/neurocomputing_v2
```

`raw_category_audit` retains original nominal labels rather than collapsing test
OOV labels together. It reports marginal, pairwise and full joint TV; full joint
TV is sample-size dependent and should be read beside the bootstrap reference.
These diagnostics never change model selection. `report/RESULTS.txt` summarizes
the actual results and limitations; its CSV, LaTeX and PDF/PNG companions use
equal-weight split means. Error bars are descriptive SDs, not confidence
intervals based on independent splits. The report refuses incomplete evidence.
The additional replay audit recalculates all ablation metrics and regenerates
samples from four fixed saved checkpoints per dataset (the first declared
split/seed). It does not overwrite samples or retune models.

