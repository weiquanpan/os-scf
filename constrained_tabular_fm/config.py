from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    seed: int = 7
    dataset: str = "accounting"
    n_train_rows: int = 4000
    n_eval_rows: int = 4000
    n_train_steps: int = 3000
    batch_size: int = 1024
    lr: float = 3e-3
    model_type: str = "linear"
    hidden_dim: int = 128
    depth: int = 3
    residual_std: float = 0.08
    constraint_weight: float = 0.15
    repair_rate: float = 450.0
    constraint_batch_scale: float = 2.0
    sample_steps: int = 700
    sample_dt: float = 7e-4
    init_scale: float = 2.0
    violation_threshold: float = 0.20
    output_dir: str = "outputs/constrained_tabular_flux_modular"
    device: str = "auto"
