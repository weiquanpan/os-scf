from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from constrained_tabular_fm.config import ExperimentConfig
from constrained_tabular_fm.data import make_table_spec, save_table_csv
from constrained_tabular_fm.io import save_array_bundle
from constrained_tabular_fm.metrics import evaluate_samples, write_metrics_csv
from constrained_tabular_fm.plotting import save_constraint_histogram, save_pair_scatter
from constrained_tabular_fm.sampling import langevin_sample
from constrained_tabular_fm.train import choose_device, set_seed, train_flux_model


def model_summary(model: torch.nn.Module) -> dict[str, object]:
    summary: dict[str, object] = {
        "class": model.__class__.__name__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    if hasattr(model, "weight"):
        summary["weight"] = getattr(model, "weight").detach().cpu().tolist()
    if hasattr(model, "bias"):
        summary["bias"] = getattr(model, "bias").detach().cpu().tolist()
    return summary


def run_tabular_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = make_table_spec(
        cfg.dataset,
        residual_std=cfg.residual_std,
        violation_threshold=cfg.violation_threshold,
    )
    train_data = spec.sample(cfg.n_train_rows, device)
    eval_reference = spec.sample(cfg.n_eval_rows, device)

    unconstrained_model, unconstrained_history = train_flux_model(
        cfg,
        spec,
        train_data,
        constrained=False,
    )
    constrained_model, constrained_history = train_flux_model(
        cfg,
        spec,
        train_data,
        constrained=True,
    )

    unconstrained_samples = langevin_sample(
        unconstrained_model,
        cfg.n_eval_rows,
        cfg.sample_steps,
        cfg.sample_dt,
        cfg.init_scale,
        device,
    )
    constrained_samples = langevin_sample(
        constrained_model,
        cfg.n_eval_rows,
        cfg.sample_steps,
        cfg.sample_dt,
        cfg.init_scale,
        device,
        constraints=spec.constraints,
        project_noise=True,
        repair_steps=1,
    )

    train_np = train_data.detach().cpu().numpy()
    reference_np = eval_reference.detach().cpu().numpy()
    unconstrained_np = unconstrained_samples.detach().cpu().numpy()
    constrained_np = constrained_samples.detach().cpu().numpy()

    save_array_bundle(
        out_dir / "tables.npz",
        train=train_np,
        reference=reference_np,
        unconstrained_flux=unconstrained_np,
        constrained_flux=constrained_np,
    )
    save_table_csv(out_dir / "train.csv", train_np, spec.columns)
    save_table_csv(out_dir / "reference.csv", reference_np, spec.columns)
    save_table_csv(out_dir / "unconstrained_flux.csv", unconstrained_np, spec.columns)
    save_table_csv(out_dir / "constrained_flux.csv", constrained_np, spec.columns)

    metrics_rows = [
        evaluate_samples(
            "unconstrained_flux",
            unconstrained_np,
            reference_np,
            spec.constraints,
            seed=cfg.seed,
        ),
        evaluate_samples(
            "constrained_flux",
            constrained_np,
            reference_np,
            spec.constraints,
            seed=cfg.seed,
        ),
    ]

    results: dict[str, object] = {
        "config": asdict(cfg),
        "dataset": cfg.dataset,
        "columns": spec.columns,
        "histories": {
            "unconstrained_flux": unconstrained_history,
            "constrained_flux": constrained_history,
        },
        "metrics": metrics_rows,
        "learned_fields": {
            "unconstrained_flux": model_summary(unconstrained_model),
            "constrained_flux": model_summary(constrained_model),
        },
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_metrics_csv(out_dir / "metrics.csv", metrics_rows)

    save_constraint_histogram(
        out_dir / "constraint_residual_hist.png",
        {
            "reference": reference_np,
            "unconstrained_flux": unconstrained_np,
            "constrained_flux": constrained_np,
        },
        spec.constraints,
    )
    y_col = len(spec.columns) - 1
    save_pair_scatter(
        out_dir / f"{spec.columns[0]}_vs_{spec.columns[y_col]}_scatter.png",
        {
            "reference": reference_np,
            "unconstrained_flux": unconstrained_np,
            "constrained_flux": constrained_np,
        },
        x_col=0,
        y_col=y_col,
        x_label=spec.columns[0],
        y_label=spec.columns[y_col],
        seed=cfg.seed,
    )

    return results


def run_accounting_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    cfg.dataset = "accounting"
    return run_tabular_experiment(cfg)


def run_complex_ledger_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    cfg.dataset = "complex"
    return run_tabular_experiment(cfg)
