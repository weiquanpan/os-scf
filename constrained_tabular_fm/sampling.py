from __future__ import annotations

import math

import torch

from constrained_tabular_fm.constraints import ConstraintSet


@torch.no_grad()
def langevin_sample(
    model: torch.nn.Module,
    n: int,
    steps: int,
    dt: float,
    init_scale: float,
    device: torch.device,
    constraints: ConstraintSet | None = None,
    project_noise: bool = False,
    repair_steps: int = 0,
) -> torch.Tensor:
    dim = int(getattr(model, "dim"))
    x = init_scale * torch.randn(n, dim, device=device)
    noise_scale = math.sqrt(2.0 * dt)
    repair_rate = float(getattr(model, "repair_rate", 0.0))
    repair_dt = dt if repair_rate == 0.0 else min(dt, 0.45 / repair_rate)
    for _ in range(steps):
        noise = torch.randn_like(x)
        if constraints is not None and project_noise:
            noise = constraints.tangent_projection_torch(x, noise)
        x = x + dt * model(x) + noise_scale * noise
        if constraints is not None and repair_rate > 0.0 and repair_steps > 0:
            for _ in range(repair_steps):
                x = x + repair_dt * constraints.repair_drift_torch(x, repair_rate)
        x = x.clamp(min=-20.0, max=20.0)
    return x
