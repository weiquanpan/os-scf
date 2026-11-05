from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from constrained_tabular_fm.constraints import ConstraintSet


def save_constraint_histogram(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    constraints: ConstraintSet,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(constraints.constraints) == 0:
        return
    n_constraints = len(constraints.constraints)
    fig, axes = plt.subplots(
        n_constraints,
        1,
        figsize=(8, 3.6 * n_constraints),
        squeeze=False,
    )
    for ax, constraint in zip(axes[:, 0], constraints.constraints):
        all_residuals = [constraint.residual_np(values) for values in arrays.values()]
        max_abs = max(float(np.percentile(np.abs(r), 99.0)) for r in all_residuals)
        max_abs = max(max_abs, constraint.tolerance * 2.0)
        bins = np.linspace(-max_abs, max_abs, 90)
        for name, values in arrays.items():
            ax.hist(
                constraint.residual_np(values),
                bins=bins,
                density=True,
                alpha=0.45,
                label=name,
            )
        ax.axvline(constraint.tolerance, color="black", linewidth=1, linestyle="--")
        ax.axvline(-constraint.tolerance, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(f"constraint residual: {constraint.name}")
        ax.set_ylabel("density")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_pair_scatter(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    x_col: int,
    y_col: int,
    x_label: str,
    y_label: str,
    seed: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    plt.figure(figsize=(8, 4.8))
    for name, values in arrays.items():
        n = min(values.shape[0], 900)
        idx = rng.choice(values.shape[0], n, replace=False)
        plt.scatter(values[idx, x_col], values[idx, y_col], s=6, alpha=0.35, label=name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
