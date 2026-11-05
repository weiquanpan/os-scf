from __future__ import annotations

import numpy as np
import torch

from constrained_tabular_fm.config import ExperimentConfig
from constrained_tabular_fm.losses import constraint_repair_loss, direct_flux_loss
from constrained_tabular_fm.models import make_drift_field


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_batch(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    idx = torch.randint(0, data.shape[0], (batch_size,), device=data.device)
    return data[idx]


def train_flux_model(
    cfg: ExperimentConfig,
    spec,
    train_data: torch.Tensor,
    constrained: bool,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    model = make_drift_field(
        cfg.model_type,
        dim=spec.dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        constraints=spec.constraints if constrained else None,
        repair_rate=cfg.repair_rate if constrained else 0.0,
    ).to(train_data.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict[str, float]] = []

    for step in range(1, cfg.n_train_steps + 1):
        x = sample_batch(train_data, cfg.batch_size)
        flux = direct_flux_loss(model, x, spec)
        repair = torch.zeros((), device=train_data.device)
        if constrained:
            probe = spec.sample_constraint_probe(
                cfg.batch_size,
                cfg.constraint_batch_scale,
                train_data.device,
            )
            repair = constraint_repair_loss(
                model,
                probe,
                spec.constraints,
                cfg.repair_rate,
            )
        loss = flux + cfg.constraint_weight * repair

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % max(1, cfg.n_train_steps // 10) == 0:
            history.append(
                {
                    "step": step,
                    "flux_loss": float(flux.detach().cpu()),
                    "constraint_loss": float(repair.detach().cpu()),
                    "total_loss": float(loss.detach().cpu()),
                }
            )
    return model, history
