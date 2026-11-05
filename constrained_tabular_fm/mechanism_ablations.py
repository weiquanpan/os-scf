from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from constrained_tabular_fm import bikesharing_methods as bike
from constrained_tabular_fm import lendingclub_methods as lc
from constrained_tabular_fm.mixed_methods import ContinuousNormalizer, set_seed


@dataclass
class LendingClubMechanismAblationConfig(lc.LendingClubConfig):
    output_dir: str = "outputs/lendingclub_mechanism_ablations"
    train_rows: int = 20_000
    eval_rows: int = 5_000
    flow_steps: int = 800
    sample_steps: int = 80
    trajectory_samples: int = 1024
    threshold_multipliers: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    repair_trajectory_steps: int = 24
    repair_blend_rate: float = 0.25


@dataclass
class BikeSharingMechanismAblationConfig(bike.BikeSharingConfig):
    output_dir: str = "outputs/bikesharing_mechanism_ablations"
    train_rows: int = 12_000
    eval_rows: int = 3_000
    flow_steps: int = 600
    sample_steps: int = 80
    trajectory_samples: int = 1024
    threshold_multipliers: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    repair_trajectory_steps: int = 24
    repair_blend_rate: float = 0.25


class ConditionalVelocity(nn.Module):
    def __init__(self, dim: int, onehot_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(dim + onehot_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x_norm: torch.Tensor, k_onehot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_norm, k_onehot, t], dim=1))


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
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
        writer.writerows(rows)


def _write_markdown(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if "rate" in column or "tv" in column:
                    values.append(f"{100.0 * value:.2f}%")
                else:
                    values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _train_conditional_flow(
    z_train: torch.Tensor,
    k_train: torch.Tensor,
    one_hot: Callable[[torch.Tensor], torch.Tensor],
    cfg: lc.LendingClubConfig | bike.BikeSharingConfig,
) -> tuple[ConditionalVelocity, ContinuousNormalizer, dict[str, float]]:
    normalizer = ContinuousNormalizer(z_train)
    target = normalizer.transform(z_train)
    k_onehot = one_hot(k_train)
    model = ConditionalVelocity(target.shape[1], k_onehot.shape[1], cfg.hidden_dim).to(target.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    last: dict[str, float] = {}
    for _step in range(cfg.flow_steps):
        idx = torch.randint(0, target.shape[0], (cfg.batch_size,), device=target.device)
        z1 = target[idx]
        z0 = torch.randn_like(z1)
        t = torch.rand(z1.shape[0], 1, device=target.device)
        z_t = (1.0 - t) * z0 + t * z1
        pred = model(z_t, k_onehot[idx], t)
        loss = F.mse_loss(pred, z1 - z0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = {"flow_loss": float(loss.detach().cpu())}
    return model, normalizer, last


@torch.no_grad()
def _sample_conditional_flow(
    model: ConditionalVelocity,
    normalizer: ContinuousNormalizer,
    k_train: torch.Tensor,
    one_hot: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    sample_steps: int,
    trajectory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, float, torch.Tensor]]]:
    idx = torch.randint(0, k_train.shape[0], (n,), device=k_train.device)
    k = k_train[idx].clone()
    k_onehot = one_hot(k)
    z_norm = torch.randn(n, model.dim, device=k_train.device)
    dt = 1.0 / max(1, sample_steps)
    record_steps = set(np.linspace(0, sample_steps, min(sample_steps + 1, 25), dtype=int).tolist())
    records: list[tuple[int, float, torch.Tensor]] = []
    for step in range(sample_steps + 1):
        if trajectory and step in record_steps:
            records.append((step, step / max(1, sample_steps), normalizer.inverse(z_norm).detach().clone()))
        if step == sample_steps:
            break
        t = torch.full((n, 1), step / max(1, sample_steps - 1), device=k_train.device)
        z_norm = (z_norm + dt * model(z_norm, k_onehot, t)).clamp(-6.0, 6.0)
    return normalizer.inverse(z_norm), k, records


def _save_trajectory_plot(path: str | Path, rows: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return repr(exc)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model, model_rows in grouped.items():
        ordered = sorted(model_rows, key=lambda item: float(item["t"]))
        ax.plot(
            [float(item["t"]) for item in ordered],
            [float(item["constraint_norm_mean"]) for item in ordered],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=model,
        )
    ax.set_xlabel("trajectory time / repair progress")
    ax.set_ylabel("mean normalized ||c(x)||")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return None


def _lc_decode_hard(spec: lc.LendingClubSpec, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    u = z[:, : spec.free_dim]
    lower = spec.lower[: spec.free_dim].to(z.device, z.dtype)
    upper = spec.upper[: spec.free_dim].to(z.device, z.dtype)
    u = torch.minimum(torch.maximum(u, lower), upper)
    return spec.decode_free(u, k)


def _lc_decode_residual(spec: lc.LendingClubSpec, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    x = _lc_decode_hard(spec, z[:, : spec.free_dim], k)
    x = x.clone()
    x[:, spec.free_dim :] = x[:, spec.free_dim :] + z[:, spec.free_dim :]
    return spec.clamp_continuous(x)


def _lc_decode_direct(spec: lc.LendingClubSpec, z: torch.Tensor, _k: torch.Tensor) -> torch.Tensor:
    return spec.clamp_continuous(z)


def _lc_constraint_rows(
    spec: lc.LendingClubSpec,
    model: str,
    x: torch.Tensor,
    k: torch.Tensor,
    threshold_multipliers: tuple[float, ...],
) -> list[dict[str, Any]]:
    residual = spec.residuals(x, k).detach().cpu().numpy().astype(np.float64)
    base = np.asarray(spec.thresholds, dtype=np.float64)
    rows = []
    for multiplier in threshold_multipliers:
        thresholds = base * float(multiplier)
        scaled = np.abs(residual) / thresholds[None, :]
        violation = scaled > 1.0
        rows.append(
            {
                "model": model,
                "threshold_multiplier": float(multiplier),
                "installment_tolerance": float(thresholds[0]),
                "ratio_tolerance": float(thresholds[1]),
                "fico_tolerance": float(thresholds[2]),
                "constraint_nmae": float(np.mean(scaled)),
                "violation_rate": float(np.mean(np.any(violation, axis=1))),
                "installment_violation_rate": float(np.mean(violation[:, 0])),
                "loan_income_ratio_violation_rate": float(np.mean(violation[:, 1])),
                "fico_range_high_violation_rate": float(np.mean(violation[:, 2])),
            }
        )
    return rows


def _lc_single_constraint_rows(
    spec: lc.LendingClubSpec,
    model: str,
    x: torch.Tensor,
    k: torch.Tensor,
) -> list[dict[str, Any]]:
    residual = spec.residuals(x, k).detach().cpu().numpy().astype(np.float64)
    thresholds = np.asarray(spec.thresholds, dtype=np.float64)
    names = ("installment", "loan_income_ratio", "fico_range_high")
    groups = {"all_constraints": (0, 1, 2)}
    groups.update({f"{name}_only": (idx,) for idx, name in enumerate(names)})
    rows = []
    for group_name, indices in groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        scaled = np.abs(residual[:, idx]) / thresholds[idx][None, :]
        violation = scaled > 1.0
        row: dict[str, Any] = {
            "model": model,
            "constraint_set": group_name,
            "n_constraints": int(len(indices)),
            "constraint_nmae": float(np.mean(scaled)),
            "violation_rate": float(np.mean(np.any(violation, axis=1))),
        }
        for local, original in enumerate(indices):
            row[f"{names[original]}_mae"] = float(np.mean(np.abs(residual[:, original])))
            row[f"{names[original]}_violation_rate"] = float(np.mean(violation[:, local]))
        rows.append(row)
    return rows


def _lc_trajectory_row(
    spec: lc.LendingClubSpec,
    model: str,
    step: int,
    t: float,
    x: torch.Tensor,
    k: torch.Tensor,
) -> dict[str, Any]:
    residual = spec.residuals(x, k).detach().cpu().numpy().astype(np.float64)
    scaled = np.abs(residual) / np.asarray(spec.thresholds, dtype=np.float64)[None, :]
    norm = np.linalg.norm(scaled, axis=1)
    return {
        "model": model,
        "step": int(step),
        "t": float(t),
        "constraint_norm_mean": float(np.mean(norm)),
        "constraint_norm_p50": float(np.quantile(norm, 0.50)),
        "constraint_norm_p90": float(np.quantile(norm, 0.90)),
        "constraint_nmae": float(np.mean(scaled)),
        "installment_abs_scaled_mean": float(np.mean(scaled[:, 0])),
        "ratio_abs_scaled_mean": float(np.mean(scaled[:, 1])),
        "fico_abs_scaled_mean": float(np.mean(scaled[:, 2])),
    }


def run_lendingclub_mechanism_ablations(cfg: LendingClubMechanismAblationConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = lc.choose_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, ref_df, codec = lc.load_lendingclub_data(cfg)
    train_x, train_k, ref_x, ref_k = lc.tensors_from_frames(train_df, ref_df, codec, device)
    spec = lc.build_spec(train_x, codec, cfg)
    n = ref_x.shape[0]

    free_model, free_norm, free_history = _train_conditional_flow(
        train_x[:, : spec.free_dim],
        train_k,
        spec.one_hot,
        cfg,
    )
    free_z, free_k, free_records = _sample_conditional_flow(
        free_model,
        free_norm,
        train_k,
        spec.one_hot,
        n,
        cfg.sample_steps,
        trajectory=True,
    )
    free_x = _lc_decode_hard(spec, free_z, free_k)

    residual_train = spec.residuals(train_x, train_k)
    residual_target = torch.cat([train_x[:, : spec.free_dim], residual_train], dim=1)
    residual_model, residual_norm, residual_history = _train_conditional_flow(
        residual_target,
        train_k,
        spec.one_hot,
        cfg,
    )
    residual_z, residual_k, residual_records = _sample_conditional_flow(
        residual_model,
        residual_norm,
        train_k,
        spec.one_hot,
        n,
        cfg.sample_steps,
        trajectory=True,
    )
    residual_x = _lc_decode_residual(spec, residual_z, residual_k)

    direct_model, direct_norm, direct_history = _train_conditional_flow(train_x, train_k, spec.one_hot, cfg)
    direct_z, direct_k, direct_records = _sample_conditional_flow(
        direct_model,
        direct_norm,
        train_k,
        spec.one_hot,
        n,
        cfg.sample_steps,
        trajectory=True,
    )
    direct_x = _lc_decode_direct(spec, direct_z, direct_k)

    bootstrap_x, bootstrap_k = lc.sample_train_bootstrap(train_x, train_k, n)
    methods = {
        "free_only_hard_reconstruction": (free_x, free_k),
        "free_plus_learned_residual": (residual_x, residual_k),
        "dependent_columns_direct_flow": (direct_x, direct_k),
        "train_bootstrap_sanity": (bootstrap_x, bootstrap_k),
    }
    for name, (x, k) in methods.items():
        lc.save_lendingclub_csv(out_dir / f"{name}.csv", spec, codec, x, k)

    rows = [
        lc.evaluate_lendingclub_samples(spec, name, x, k, ref_x, ref_k)
        for name, (x, k) in methods.items()
    ]
    lc.write_metric_csv(out_dir / "metrics.csv", rows)
    lc.write_main_table(out_dir / "main_table.md", rows)

    threshold_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    for name, (x, k) in methods.items():
        threshold_rows.extend(_lc_constraint_rows(spec, name, x, k, cfg.threshold_multipliers))
        single_rows.extend(_lc_single_constraint_rows(spec, name, x, k))
    _write_csv(out_dir / "threshold_sensitivity.csv", threshold_rows)
    _write_csv(out_dir / "single_constraint_ablation.csv", single_rows)
    _write_markdown(
        out_dir / "single_constraint_ablation.md",
        single_rows,
        ["model", "constraint_set", "constraint_nmae", "violation_rate"],
    )

    trajectory_rows: list[dict[str, Any]] = []
    traj_n = min(cfg.trajectory_samples, n)
    traj_k = free_k[:traj_n]
    for step, t, z in free_records:
        trajectory_rows.append(
            _lc_trajectory_row(spec, "free_only_hard_reconstruction", step, t, _lc_decode_hard(spec, z[:traj_n], traj_k), traj_k)
        )
    traj_k = residual_k[:traj_n]
    for step, t, z in residual_records:
        trajectory_rows.append(
            _lc_trajectory_row(spec, "free_plus_learned_residual", step, t, _lc_decode_residual(spec, z[:traj_n], traj_k), traj_k)
        )
    traj_k = direct_k[:traj_n]
    for step, t, z in direct_records:
        trajectory_rows.append(
            _lc_trajectory_row(spec, "dependent_columns_direct_flow", step, t, _lc_decode_direct(spec, z[:traj_n], traj_k), traj_k)
        )
    repair_x = direct_x[:traj_n].clone()
    repair_k = direct_k[:traj_n]
    for step in range(cfg.repair_trajectory_steps + 1):
        trajectory_rows.append(
            _lc_trajectory_row(
                spec,
                "direct_flow_relaxed_repair",
                step,
                step / max(1, cfg.repair_trajectory_steps),
                repair_x,
                repair_k,
            )
        )
        if step < cfg.repair_trajectory_steps:
            target = spec.targets(repair_x, repair_k)
            repair_x[:, spec.free_dim :] = repair_x[:, spec.free_dim :] + cfg.repair_blend_rate * (
                target - repair_x[:, spec.free_dim :]
            )
    _write_csv(out_dir / "trajectory_constraint_norm.csv", trajectory_rows)
    plot_error = _save_trajectory_plot(out_dir / "trajectory_constraint_norm.png", trajectory_rows)

    result = {
        "config": asdict(cfg),
        "device": str(device),
        "columns": {
            "continuous": list(lc.CONTINUOUS_COLUMNS),
            "free_continuous": list(lc.FREE_CONTINUOUS_COLUMNS),
            "dependent_continuous": list(lc.DEPENDENT_CONTINUOUS_COLUMNS),
            "categorical": list(lc.CATEGORICAL_COLUMNS),
        },
        "thresholds": {
            "installment": cfg.installment_tolerance,
            "loan_income_ratio": cfg.ratio_tolerance,
            "fico_range_high": cfg.fico_tolerance,
        },
        "histories": {
            "free_only_hard_reconstruction": free_history,
            "free_plus_learned_residual": residual_history,
            "dependent_columns_direct_flow": direct_history,
        },
        "metrics": rows,
        "outputs": {
            "main_table": str(out_dir / "main_table.md"),
            "threshold_sensitivity": str(out_dir / "threshold_sensitivity.csv"),
            "single_constraint_ablation": str(out_dir / "single_constraint_ablation.csv"),
            "trajectory_constraint_norm": str(out_dir / "trajectory_constraint_norm.csv"),
            "trajectory_plot": str(out_dir / "trajectory_constraint_norm.png"),
        },
        "plot_error": plot_error,
        "notes": {
            "free_only_hard_reconstruction": "Learns only structural free coordinates and reconstructs installment, ratio, and FICO by equations.",
            "free_plus_learned_residual": "Learns free coordinates plus epsilon and reconstructs x = Phi(u, epsilon).",
            "dependent_columns_direct_flow": "Uses the same conditional flow family but directly generates all continuous columns, including dependent formula columns.",
            "direct_flow_relaxed_repair": "Starts from direct-flow samples and repeatedly moves dependent columns toward Phi(u) to expose repair residual decay.",
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _bike_decode_hard(spec: bike.BikeSharingSpec, z: torch.Tensor, _k: torch.Tensor) -> torch.Tensor:
    u = z[:, : spec.free_dim]
    lower = spec.lower[: spec.free_dim].to(z.device, z.dtype)
    upper = spec.upper[: spec.free_dim].to(z.device, z.dtype)
    u = torch.minimum(torch.maximum(u, lower), upper)
    return spec.decode_free(u)


def _bike_decode_residual(spec: bike.BikeSharingSpec, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    x = _bike_decode_hard(spec, z[:, : spec.free_dim], k)
    x = x.clone()
    x[:, spec.free_dim :] = x[:, spec.free_dim :] + z[:, spec.free_dim :]
    return spec.clamp_continuous(x)


def _bike_decode_direct(spec: bike.BikeSharingSpec, z: torch.Tensor, _k: torch.Tensor) -> torch.Tensor:
    return spec.clamp_continuous(z)


def _bike_constraint_rows(
    spec: bike.BikeSharingSpec,
    model: str,
    x: torch.Tensor,
    threshold_multipliers: tuple[float, ...],
) -> list[dict[str, Any]]:
    residual = spec.residuals(x).detach().cpu().numpy().astype(np.float64)
    rows = []
    for multiplier in threshold_multipliers:
        threshold = spec.cnt_tolerance * float(multiplier)
        scaled = np.abs(residual[:, 0]) / threshold
        rows.append(
            {
                "model": model,
                "threshold_multiplier": float(multiplier),
                "cnt_tolerance": float(threshold),
                "constraint_nmae": float(np.mean(scaled)),
                "violation_rate": float(np.mean(scaled > 1.0)),
            }
        )
    return rows


def _bike_trajectory_row(
    spec: bike.BikeSharingSpec,
    model: str,
    step: int,
    t: float,
    x: torch.Tensor,
) -> dict[str, Any]:
    residual = spec.residuals(x).detach().cpu().numpy().astype(np.float64)
    scaled = np.abs(residual[:, 0]) / spec.cnt_tolerance
    return {
        "model": model,
        "step": int(step),
        "t": float(t),
        "constraint_norm_mean": float(np.mean(scaled)),
        "constraint_norm_p50": float(np.quantile(scaled, 0.50)),
        "constraint_norm_p90": float(np.quantile(scaled, 0.90)),
        "constraint_nmae": float(np.mean(scaled)),
        "cnt_abs_scaled_mean": float(np.mean(scaled)),
    }


def run_bikesharing_mechanism_ablations(cfg: BikeSharingMechanismAblationConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = bike.choose_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, ref_df, codec = bike.load_bikesharing_data(cfg)
    train_x, train_k, ref_x, ref_k = bike.tensors_from_frames(train_df, ref_df, codec, device)
    spec = bike.build_spec(train_x, codec, cfg)
    n = ref_x.shape[0]

    free_model, free_norm, free_history = _train_conditional_flow(
        train_x[:, : spec.free_dim],
        train_k,
        spec.one_hot,
        cfg,
    )
    free_z, free_k, free_records = _sample_conditional_flow(
        free_model,
        free_norm,
        train_k,
        spec.one_hot,
        n,
        cfg.sample_steps,
        trajectory=True,
    )
    free_x = _bike_decode_hard(spec, free_z, free_k)

    residual_train = spec.residuals(train_x)
    residual_target = torch.cat([train_x[:, : spec.free_dim], residual_train], dim=1)
    residual_model, residual_norm, residual_history = _train_conditional_flow(
        residual_target,
        train_k,
        spec.one_hot,
        cfg,
    )
    residual_z, residual_k, residual_records = _sample_conditional_flow(
        residual_model,
        residual_norm,
        train_k,
        spec.one_hot,
        n,
        cfg.sample_steps,
        trajectory=True,
    )
    residual_x = _bike_decode_residual(spec, residual_z, residual_k)

    direct_model, direct_norm, direct_history = _train_conditional_flow(train_x, train_k, spec.one_hot, cfg)
    direct_z, direct_k, direct_records = _sample_conditional_flow(
        direct_model,
        direct_norm,
        train_k,
        spec.one_hot,
        n,
        cfg.sample_steps,
        trajectory=True,
    )
    direct_x = _bike_decode_direct(spec, direct_z, direct_k)

    bootstrap_x, bootstrap_k = bike.sample_train_bootstrap(train_x, train_k, n)
    methods = {
        "free_only_hard_reconstruction": (free_x, free_k),
        "free_plus_learned_residual": (residual_x, residual_k),
        "dependent_columns_direct_flow": (direct_x, direct_k),
        "train_bootstrap_sanity": (bootstrap_x, bootstrap_k),
    }
    for name, (x, k) in methods.items():
        bike.save_bikesharing_csv(out_dir / f"{name}.csv", spec, codec, x, k)

    rows = [
        bike.evaluate_bikesharing_samples(spec, name, x, k, ref_x, ref_k)
        for name, (x, k) in methods.items()
    ]
    bike.write_metric_csv(out_dir / "metrics.csv", rows)
    bike.write_main_table(out_dir / "main_table.md", rows)

    threshold_rows: list[dict[str, Any]] = []
    for name, (x, _k) in methods.items():
        threshold_rows.extend(_bike_constraint_rows(spec, name, x, cfg.threshold_multipliers))
    _write_csv(out_dir / "threshold_sensitivity.csv", threshold_rows)

    trajectory_rows: list[dict[str, Any]] = []
    traj_n = min(cfg.trajectory_samples, n)
    traj_k = free_k[:traj_n]
    for step, t, z in free_records:
        trajectory_rows.append(_bike_trajectory_row(spec, "free_only_hard_reconstruction", step, t, _bike_decode_hard(spec, z[:traj_n], traj_k)))
    traj_k = residual_k[:traj_n]
    for step, t, z in residual_records:
        trajectory_rows.append(_bike_trajectory_row(spec, "free_plus_learned_residual", step, t, _bike_decode_residual(spec, z[:traj_n], traj_k)))
    traj_k = direct_k[:traj_n]
    for step, t, z in direct_records:
        trajectory_rows.append(_bike_trajectory_row(spec, "dependent_columns_direct_flow", step, t, _bike_decode_direct(spec, z[:traj_n], traj_k)))
    repair_x = direct_x[:traj_n].clone()
    for step in range(cfg.repair_trajectory_steps + 1):
        trajectory_rows.append(
            _bike_trajectory_row(
                spec,
                "direct_flow_relaxed_repair",
                step,
                step / max(1, cfg.repair_trajectory_steps),
                repair_x,
            )
        )
        if step < cfg.repair_trajectory_steps:
            target = spec.targets(repair_x)
            repair_x[:, spec.free_dim :] = repair_x[:, spec.free_dim :] + cfg.repair_blend_rate * (
                target - repair_x[:, spec.free_dim :]
            )
    _write_csv(out_dir / "trajectory_constraint_norm.csv", trajectory_rows)
    plot_error = _save_trajectory_plot(out_dir / "trajectory_constraint_norm.png", trajectory_rows)

    result = {
        "config": asdict(cfg),
        "device": str(device),
        "columns": {
            "continuous": list(bike.CONTINUOUS_COLUMNS),
            "free_continuous": list(bike.FREE_CONTINUOUS_COLUMNS),
            "dependent_continuous": list(bike.DEPENDENT_CONTINUOUS_COLUMNS),
            "categorical": list(bike.CATEGORICAL_COLUMNS),
        },
        "thresholds": {"cnt": cfg.cnt_tolerance},
        "histories": {
            "free_only_hard_reconstruction": free_history,
            "free_plus_learned_residual": residual_history,
            "dependent_columns_direct_flow": direct_history,
        },
        "metrics": rows,
        "outputs": {
            "main_table": str(out_dir / "main_table.md"),
            "threshold_sensitivity": str(out_dir / "threshold_sensitivity.csv"),
            "trajectory_constraint_norm": str(out_dir / "trajectory_constraint_norm.csv"),
            "trajectory_plot": str(out_dir / "trajectory_constraint_norm.png"),
        },
        "plot_error": plot_error,
        "notes": {
            "free_only_hard_reconstruction": "Learns free continuous coordinates and reconstructs cnt = casual + registered.",
            "free_plus_learned_residual": "Learns free coordinates plus epsilon and reconstructs x = Phi(u, epsilon).",
            "dependent_columns_direct_flow": "Uses the same conditional flow family but directly generates cnt together with free columns.",
            "direct_flow_relaxed_repair": "Starts from direct-flow samples and repeatedly moves cnt toward casual + registered.",
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
