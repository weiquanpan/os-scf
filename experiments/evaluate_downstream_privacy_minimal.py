"""Minimal downstream utility and privacy checks for real/semi-real tables.

This script is intentionally standalone. It reads existing generated CSV files
and writes additional evaluation tables without changing the training/generation
pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    train_csv: Path
    reference_csv: Path
    target: str
    positive_label: str | None
    continuous: tuple[str, ...]
    categorical: tuple[str, ...]
    utility_exclude: tuple[str, ...]
    default_run_dirs: tuple[Path, ...]


METHODS = {
    "original_space_constrained_flow": "ours_structural_coordinate",
    "ctgan": "ctgan",
    "tabddpm_official": "tabddpm_official",
    "tabsyn_adapter": "tabsyn_adapter",
    "tabsyn_adapter_posthoc_repair": "tabsyn_adapter_posthoc_repair",
}

DEFAULT_METHODS = (
    "original_space_constrained_flow",
    "tabsyn_adapter",
    "tabsyn_adapter_posthoc_repair",
    "ctgan",
)

METHOD_DISPLAY_ORDER = {
    "real_trtr_matched": 0,
    "real_holdout_control": 0,
    "ours_structural_coordinate": 1,
    "tabsyn_adapter": 2,
    "tabsyn_adapter_posthoc_repair": 3,
    "ctgan": 4,
    "tabddpm_official": 90,
    "train_bootstrap_sanity": 95,
}


DATASETS = {
    "lendingclub": DatasetSpec(
        name="lendingclub",
        task="classification",
        train_csv=Path("data/lendingclub_preprocessed/train.csv"),
        reference_csv=Path("data/lendingclub_preprocessed/reference.csv"),
        target="loan_status",
        positive_label="charged_off",
        continuous=(
            "loan_amnt",
            "int_rate",
            "annual_inc",
            "dti",
            "fico_range_low",
            "open_acc",
            "revol_util",
            "total_acc",
            "installment",
            "loan_income_ratio",
            "fico_range_high",
        ),
        categorical=(
            "term_months",
            "grade",
            "home_ownership",
            "verification_status",
            "purpose",
            "application_type",
            "loan_status",
        ),
        utility_exclude=("loan_status",),
        default_run_dirs=(
            Path("outputs/lendingclub_full_budget_methods"),
            Path("outputs/lendingclub_full_budget_methods_seed14"),
            Path("outputs/lendingclub_full_budget_methods_seed15"),
        ),
    ),
    "bikesharing": DatasetSpec(
        name="bikesharing",
        task="regression",
        train_csv=Path("data/bike_sharing_preprocessed/train.csv"),
        reference_csv=Path("data/bike_sharing_preprocessed/reference.csv"),
        target="cnt",
        positive_label=None,
        continuous=(
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "casual",
            "registered",
            "cnt",
        ),
        categorical=(
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
        ),
        utility_exclude=("casual", "registered", "cnt"),
        default_run_dirs=(
            Path("outputs/bikesharing_methods_seed13"),
            Path("outputs/bikesharing_methods_seed14"),
            Path("outputs/bikesharing_methods_seed15"),
        ),
    ),
}


def make_one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    out = df.copy()
    for column in spec.continuous:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in spec.categorical:
        if column in out:
            out[column] = out[column].astype("string").fillna("__missing__").astype(str)
    return out


def sample_rows(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def required_columns(spec: DatasetSpec) -> list[str]:
    return list(spec.continuous + spec.categorical)


def check_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def utility_features(spec: DatasetSpec) -> list[str]:
    excluded = set(spec.utility_exclude)
    return [column for column in required_columns(spec) if column not in excluded]


def build_preprocessor(
    spec: DatasetSpec,
    features: list[str],
    *,
    dense: bool = False,
) -> ColumnTransformer:
    numeric_features = [column for column in features if column in spec.continuous]
    categorical_features = [column for column in features if column in spec.categorical]
    sparse_threshold = 0.0 if dense else 0.3
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
    )


def build_estimator(
    spec: DatasetSpec,
    features: list[str],
    args: argparse.Namespace,
    seed: int,
) -> Pipeline:
    if args.utility_model == "linear":
        if spec.task == "classification":
            model = LogisticRegression(
                max_iter=args.linear_max_iter,
                class_weight="balanced",
                random_state=seed,
            )
        else:
            model = Ridge(alpha=1.0, random_state=seed)
        dense = False
    else:
        if spec.task == "classification":
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=args.n_jobs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                random_state=seed,
                n_jobs=args.n_jobs,
            )
        dense = False
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(spec, features, dense=dense)),
            ("model", model),
        ]
    )


def clean_utility_frame(
    df: pd.DataFrame,
    spec: DatasetSpec,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    columns = features + [spec.target]
    check_columns(df, columns, "utility frame")
    frame = df[columns].copy()
    if spec.task == "classification":
        frame[spec.target] = frame[spec.target].astype("string").fillna("__missing__").astype(str)
        frame = frame[frame[spec.target] != "__missing__"]
    else:
        frame[spec.target] = pd.to_numeric(frame[spec.target], errors="coerce")
        frame = frame[np.isfinite(frame[spec.target].to_numpy(dtype=float))]
    frame = normalize_columns(frame, spec)
    return frame[features], frame[spec.target]


def classification_metrics(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    positive_label: str | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    labels = list(getattr(model.named_steps["model"], "classes_", []))
    if positive_label is not None and positive_label in labels:
        y_true_bin = (y_true.to_numpy() == positive_label).astype(int)
        y_pred_bin = (np.asarray(y_pred) == positive_label).astype(int)
        metrics["f1_positive"] = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        proba = model.predict_proba(x_test)
        positive_index = labels.index(positive_label)
        score = proba[:, positive_index]
        if len(np.unique(y_true_bin)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true_bin, score))
            metrics["average_precision"] = float(average_precision_score(y_true_bin, score))
    return metrics


def evaluate_utility(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spec: DatasetSpec,
    train_source: str,
    method: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    features = utility_features(spec)
    x_train, y_train = clean_utility_frame(train_df, spec, features)
    x_test, y_test = clean_utility_frame(test_df, spec, features)
    row: dict[str, Any] = {
        "dataset": spec.name,
        "seed": seed,
        "method": method,
        "train_source": train_source,
        "task": spec.task,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "utility_model": args.utility_model,
    }
    if len(x_train) == 0 or len(x_test) == 0:
        row["status"] = "skipped_empty"
        return row
    if spec.task == "classification" and y_train.nunique() < 2:
        row["status"] = "skipped_single_train_class"
        return row

    estimator = build_estimator(spec, features, args, seed)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    if spec.task == "classification":
        row.update(classification_metrics(estimator, x_test, y_test, y_pred, spec.positive_label))
    else:
        y_true = y_test.to_numpy(dtype=float)
        y_hat = np.asarray(y_pred, dtype=float)
        row.update(
            {
                "r2": float(r2_score(y_true, y_hat)),
                "rmse": float(math.sqrt(mean_squared_error(y_true, y_hat))),
                "mae": float(mean_absolute_error(y_true, y_hat)),
            }
        )
    row["status"] = "ok"
    return row


def encoded_matrix(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    columns: list[str],
) -> Any:
    return preprocessor.transform(df[columns])


def nearest_distances(reference_matrix: Any, query_matrix: Any) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(reference_matrix)
    distances, _ = nn.kneighbors(query_matrix, return_distance=True)
    return distances[:, 0]


def quantile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, q))


def evaluate_privacy(
    synthetic_df: pd.DataFrame,
    real_train_df: pd.DataFrame,
    real_reference_df: pd.DataFrame,
    spec: DatasetSpec,
    method: str,
    seed: int,
    args: argparse.Namespace,
    *,
    membership_auc: bool = True,
) -> dict[str, Any]:
    columns = required_columns(spec)
    check_columns(synthetic_df, columns, f"{method} synthetic frame")
    check_columns(real_train_df, columns, "real train frame")
    check_columns(real_reference_df, columns, "real reference frame")

    train = normalize_columns(sample_rows(real_train_df[columns], args.privacy_real_rows, seed), spec)
    reference = normalize_columns(
        sample_rows(real_reference_df[columns], args.privacy_real_rows, seed + 1),
        spec,
    )
    synthetic = normalize_columns(
        sample_rows(synthetic_df[columns], args.privacy_synthetic_rows, seed + 2),
        spec,
    )

    preprocessor = build_preprocessor(spec, columns, dense=False)
    train_matrix = preprocessor.fit_transform(train[columns])
    reference_matrix = encoded_matrix(preprocessor, reference, columns)
    synthetic_matrix = encoded_matrix(preprocessor, synthetic, columns)

    dcr_train = nearest_distances(train_matrix, synthetic_matrix)
    dcr_reference = nearest_distances(reference_matrix, synthetic_matrix)
    ratio = dcr_train / np.maximum(dcr_reference, args.distance_epsilon)
    near_duplicates = dcr_train <= args.duplicate_epsilon

    row: dict[str, Any] = {
        "dataset": spec.name,
        "seed": seed,
        "method": method,
        "n_synthetic": int(len(synthetic)),
        "n_real_train": int(len(train)),
        "n_real_reference": int(len(reference)),
        "distance_space": "train-standardized numeric + train-one-hot categorical",
        "dcr_train_mean": float(np.mean(dcr_train)),
        "dcr_train_median": float(np.median(dcr_train)),
        "dcr_train_p05": quantile(dcr_train, 0.05),
        "dcr_reference_mean": float(np.mean(dcr_reference)),
        "dcr_reference_median": float(np.median(dcr_reference)),
        "dcr_reference_p05": quantile(dcr_reference, 0.05),
        "dcr_train_over_reference_median": float(np.median(ratio)),
        "near_duplicate_rate_train": float(np.mean(near_duplicates)),
        "status": "ok",
    }

    if membership_auc:
        try:
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(synthetic_matrix)
            member_dist = nn.kneighbors(train_matrix, return_distance=True)[0][:, 0]
            nonmember_dist = nn.kneighbors(reference_matrix, return_distance=True)[0][:, 0]
            labels = np.concatenate(
                [np.ones(len(member_dist), dtype=int), np.zeros(len(nonmember_dist), dtype=int)]
            )
            scores = -np.concatenate([member_dist, nonmember_dist])
            row["membership_auc"] = float(roc_auc_score(labels, scores))
        except ValueError:
            row["membership_auc"] = float("nan")
    return row


def load_seed(run_dir: Path, fallback_index: int) -> int:
    metrics_json = run_dir / "metrics.json"
    if metrics_json.exists():
        try:
            payload = json.loads(metrics_json.read_text(encoding="utf-8"))
            return int(payload.get("config", {}).get("seed", fallback_index))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    for part in run_dir.name.split("_"):
        if part.startswith("seed"):
            try:
                return int(part.replace("seed", ""))
            except ValueError:
                pass
    return fallback_index


def load_synthetic_frames(
    run_dir: Path,
    methods: list[str],
    spec: DatasetSpec,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for method in methods:
        path = run_dir / f"{method}.csv"
        frame = read_csv(path, required=False)
        if frame is None:
            print(f"[warn] missing {path}; skipping {method}")
            continue
        check_columns(frame, required_columns(spec), str(path))
        frames[method] = normalize_columns(frame, spec)
    return frames


def evaluate_dataset(
    spec: DatasetSpec,
    run_dirs: list[Path],
    methods: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_train = normalize_columns(read_csv(spec.train_csv), spec)
    real_reference = normalize_columns(read_csv(spec.reference_csv), spec)
    utility_rows: list[dict[str, Any]] = []
    privacy_rows: list[dict[str, Any]] = []

    for fallback_index, run_dir in enumerate(run_dirs, start=13):
        if not run_dir.exists():
            print(f"[warn] missing run dir {run_dir}; skipping")
            continue
        seed = load_seed(run_dir, fallback_index)
        synthetic_frames = load_synthetic_frames(run_dir, methods, spec)
        if not synthetic_frames:
            continue

        synthetic_n = min(len(frame) for frame in synthetic_frames.values())
        max_train_rows = args.max_train_rows if args.max_train_rows is not None else synthetic_n
        real_train_eval = sample_rows(real_train, max_train_rows, seed)
        real_reference_eval = sample_rows(real_reference, args.max_test_rows, seed + 1)

        utility_rows.append(
            evaluate_utility(
                real_train_eval,
                real_reference_eval,
                spec,
                train_source="real",
                method="real_trtr_matched",
                seed=seed,
                args=args,
            )
        )

        holdout_control = evaluate_privacy(
            sample_rows(real_reference, synthetic_n, seed + 3),
            real_train,
            real_reference,
            spec,
            "real_holdout_control",
            seed,
            args,
            membership_auc=False,
        )
        for key in (
            "dcr_reference_mean",
            "dcr_reference_median",
            "dcr_reference_p05",
            "dcr_train_over_reference_median",
        ):
            holdout_control[key] = float("nan")
        privacy_rows.append(holdout_control)

        if args.include_bootstrap_control and (run_dir / "train_bootstrap_sanity.csv").exists():
            bootstrap = normalize_columns(pd.read_csv(run_dir / "train_bootstrap_sanity.csv"), spec)
            privacy_rows.append(
                evaluate_privacy(
                    bootstrap,
                    real_train,
                    real_reference,
                    spec,
                    "train_bootstrap_sanity",
                    seed,
                    args,
                    membership_auc=True,
                )
            )

        for source_method, frame in synthetic_frames.items():
            paper_method = METHODS.get(source_method, source_method)
            train_frame = sample_rows(frame, args.max_train_rows, seed)
            utility_rows.append(
                evaluate_utility(
                    train_frame,
                    real_reference_eval,
                    spec,
                    train_source="synthetic",
                    method=paper_method,
                    seed=seed,
                    args=args,
                )
            )
            privacy_rows.append(
                evaluate_privacy(
                    frame,
                    real_train,
                    real_reference,
                    spec,
                    paper_method,
                    seed,
                    args,
                    membership_auc=True,
                )
            )

    return pd.DataFrame(utility_rows), pd.DataFrame(privacy_rows)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [
        column
        for column in df.columns
        if column not in group_cols
        and pd.api.types.is_numeric_dtype(df[column])
        and column not in {"seed"}
    ]
    summary = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([str(part) for part in column if part != ""]).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary = sort_for_display(summary)
    return summary


def sort_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [column for column in ["dataset", "train_source"] if column in df.columns]
    work = df.copy()
    if "method" in work.columns:
        work["_method_order"] = work["method"].map(lambda name: METHOD_DISPLAY_ORDER.get(str(name), 50))
        sort_cols.append("_method_order")
    if sort_cols:
        work = work.sort_values(sort_cols, kind="stable")
    return work.drop(columns=["_method_order"], errors="ignore").reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    small = df.loc[:, [column for column in columns if column in df.columns]].head(max_rows).copy()
    for column in small.columns:
        if pd.api.types.is_float_dtype(small[column]):
            small[column] = small[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    headers = list(small.columns)
    rows = small.astype(str).values.tolist()
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rows)) if rows else len(str(header))
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_outputs(
    utility: pd.DataFrame,
    privacy: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    utility.to_csv(output_dir / "utility_per_seed.csv", index=False)
    privacy.to_csv(output_dir / "privacy_per_seed.csv", index=False)

    utility_ok = utility
    if not utility.empty and "status" in utility:
        utility_ok = utility[utility["status"] == "ok"]
    privacy_ok = privacy
    if not privacy.empty and "status" in privacy:
        privacy_ok = privacy[privacy["status"] == "ok"]

    utility_summary = summarize(utility_ok, ["dataset", "method", "train_source"])
    privacy_summary = summarize(privacy_ok, ["dataset", "method"])
    utility_summary.to_csv(output_dir / "utility_summary.csv", index=False)
    privacy_summary.to_csv(output_dir / "privacy_summary.csv", index=False)

    lines = [
        "# Minimal downstream utility and privacy evaluation",
        "",
        "Utility uses TSTR for synthetic methods and matched-size TRTR for the real baseline.",
        "Privacy uses nearest-neighbor DCR in a train-standardized mixed feature space.",
        "",
        "## Utility summary",
        "",
    ]
    utility_columns = [
        "dataset",
        "method",
        "train_source",
        "roc_auc_mean",
        "f1_macro_mean",
        "f1_positive_mean",
        "r2_mean",
        "rmse_mean",
        "mae_mean",
        "n_train_mean",
        "n_test_mean",
    ]
    lines.append(markdown_table(utility_summary, utility_columns))
    lines.extend(["", "## Privacy summary", ""])
    privacy_columns = [
        "dataset",
        "method",
        "dcr_train_median_mean",
        "dcr_reference_median_mean",
        "dcr_train_over_reference_median_mean",
        "near_duplicate_rate_train_mean",
        "membership_auc_mean",
    ]
    lines.append(markdown_table(privacy_summary, privacy_columns))
    lines.append("")
    (output_dir / "main_table.md").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "methods": METHODS,
        "args": json_safe(vars(args)),
        "notes": {
            "utility": "TSTR trains downstream models on synthetic rows and tests on real reference rows. real_trtr_matched trains on a matched-size real-train subset.",
            "privacy": "DCR is distance to closest record. Smaller train DCR and high membership AUC can indicate memorization risk. This is not a differential privacy guarantee.",
            "default_methods": "Default synthetic comparisons are structural-coordinate flow, TabSyn adapter, TabSyn adapter plus post-hoc repair, and CTGAN. The real_holdout_control row is generated automatically.",
            "tabddpm_policy": "tabddpm_official is excluded from the default utility/privacy comparison because its real/semi-real distributional fidelity is diagnostic rather than a credible key comparator; pass --include-diagnostic-tabddpm to reproduce its rows.",
            "bikesharing_features": "Utility excludes casual and registered when predicting cnt to avoid the cnt = casual + registered leakage shortcut.",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate minimal TSTR/TRTR utility and nearest-neighbor privacy checks."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help=(
            "Generated CSV stems to evaluate from each run directory. "
            "Defaults are original_space_constrained_flow, tabsyn_adapter, "
            "tabsyn_adapter_posthoc_repair, and ctgan. tabddpm_official is "
            "retained as an opt-in diagnostic row rather than a key comparator."
        ),
    )
    parser.add_argument(
        "--include-diagnostic-tabddpm",
        action="store_true",
        help="Also evaluate tabddpm_official as a diagnostic row.",
    )
    parser.add_argument("--lendingclub-run-dirs", nargs="*", type=Path)
    parser.add_argument("--bikesharing-run-dirs", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/downstream_privacy_minimal"))
    parser.add_argument("--utility-model", choices=["rf", "linear"], default="rf")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--linear-max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-test-rows", type=int, default=5000)
    parser.add_argument("--privacy-real-rows", type=int, default=10000)
    parser.add_argument("--privacy-synthetic-rows", type=int, default=3000)
    parser.add_argument("--duplicate-epsilon", type=float, default=1e-8)
    parser.add_argument("--distance-epsilon", type=float, default=1e-12)
    parser.add_argument("--include-bootstrap-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.include_diagnostic_tabddpm and "tabddpm_official" not in args.methods:
        args.methods.append("tabddpm_official")
    all_utility: list[pd.DataFrame] = []
    all_privacy: list[pd.DataFrame] = []

    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        custom_dirs = getattr(args, f"{dataset_name}_run_dirs")
        run_dirs = list(custom_dirs) if custom_dirs else list(spec.default_run_dirs)
        print(f"[eval] {dataset_name}: {len(run_dirs)} run dir(s)")
        utility, privacy = evaluate_dataset(spec, run_dirs, args.methods, args)
        dataset_dir = args.output_dir / dataset_name
        write_outputs(utility, privacy, dataset_dir, args)
        print(f"[write] {dataset_dir}")
        all_utility.append(utility)
        all_privacy.append(privacy)

    combined_utility = pd.concat(all_utility, ignore_index=True) if all_utility else pd.DataFrame()
    combined_privacy = pd.concat(all_privacy, ignore_index=True) if all_privacy else pd.DataFrame()
    write_outputs(combined_utility, combined_privacy, args.output_dir, args)
    print(f"[write] {args.output_dir}")


if __name__ == "__main__":
    main()
