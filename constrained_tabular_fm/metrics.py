from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from constrained_tabular_fm.constraints import ConstraintSet


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    n_proj: int = 128,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(x.shape[1], n_proj))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True) + 1e-12
    x_proj = np.sort(x @ dirs, axis=0)
    y_proj = np.sort(y @ dirs, axis=0)
    return float(np.sqrt(np.mean((x_proj - y_proj) ** 2)))


def evaluate_samples(
    name: str,
    samples: np.ndarray,
    reference: np.ndarray,
    constraints: ConstraintSet,
    seed: int = 0,
) -> dict[str, float | str]:
    samples = np.asarray(samples, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if samples.shape[1] != reference.shape[1]:
        raise ValueError(
            f"{name}: generated dim {samples.shape[1]} != reference dim {reference.shape[1]}"
        )

    out: dict[str, float | str] = {
        "model": name,
        "n_samples": int(samples.shape[0]),
        "sliced_wasserstein": sliced_wasserstein(samples, reference, seed=seed),
        "mean_l2": float(np.linalg.norm(samples.mean(axis=0) - reference.mean(axis=0))),
        "cov_frobenius": float(
            np.linalg.norm(np.cov(samples.T) - np.cov(reference.T), ord="fro")
        ),
    }
    out.update(constraints.metrics_np(samples))

    ref_constraint_metrics = constraints.metrics_np(reference)
    for key, value in ref_constraint_metrics.items():
        if key.startswith("constraint_") or key == "violation_rate":
            out[f"reference_{key}"] = value
    return out


def write_metrics_csv(path: str | Path, rows: list[dict[str, float | str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
