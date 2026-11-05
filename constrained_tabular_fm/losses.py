from __future__ import annotations

import torch

from constrained_tabular_fm.constraints import ConstraintSet
from constrained_tabular_fm.models import LinearDriftField


def exact_divergence(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    parts = []
    for idx in range(y.shape[1]):
        grad = torch.autograd.grad(
            y[:, idx].sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0][:, idx]
        parts.append(grad)
    return torch.stack(parts, dim=1).sum(dim=1)


def direct_flux_loss(
    model: LinearDriftField,
    x: torch.Tensor,
    spec,
) -> torch.Tensor:
    """Direct Fokker-Planck flux residual for the analytic toy distribution.

    This equals

        div(f - score) + (f - score) . score

    up to algebraic rearrangement. The official paper implementation uses a
    projected estimator with KDE scores; this direct form is intended only for
    low-dimensional distributions where the score and divergence are known.
    """
    x_grad = x.detach().requires_grad_(True)
    score = spec.score(x_grad)
    div_score = spec.div_score(x_grad)
    drift = model(x_grad)
    div_drift = exact_divergence(drift, x_grad)
    rhs = (div_score + (score * score).sum(dim=1)).detach()
    lhs = div_drift + (drift * score.detach()).sum(dim=1)
    scale = rhs.abs().mean().clamp_min(1.0)
    return ((lhs - rhs) / scale).pow(2).mean()


def constraint_repair_loss(
    model: LinearDriftField,
    x: torch.Tensor,
    constraints: ConstraintSet,
    repair_rate: float,
) -> torch.Tensor:
    residuals = constraints.residuals_torch(x)
    velocities = constraints.residual_velocities_from_state_torch(x, model(x))
    if residuals.numel() == 0:
        return x.new_zeros(())
    denom = repair_rate * residuals.detach().std(dim=0).clamp_min(1.0)
    return ((velocities + repair_rate * residuals) / denom).pow(2).mean()
