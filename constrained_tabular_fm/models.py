from __future__ import annotations

import torch
from torch import nn

from constrained_tabular_fm.constraints import ConstraintSet


class LinearDriftField(nn.Module):
    """Linear drift plus an optional fixed constraint-repair component."""

    def __init__(
        self,
        dim: int,
        constraints: ConstraintSet | None = None,
        repair_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(-0.05 * torch.eye(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.constraints = constraints
        self.repair_rate = float(repair_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drift = x @ self.weight.T + self.bias
        if self.constraints is not None and self.repair_rate != 0.0:
            drift = drift + self.constraints.repair_drift_torch(x, self.repair_rate)
        return drift

    def divergence(self) -> torch.Tensor:
        div = torch.trace(self.weight)
        if self.constraints is not None and self.repair_rate != 0.0:
            div = div + self.weight.new_tensor(
                self.constraints.repair_divergence(self.repair_rate)
            )
        return div


class MLPDriftField(nn.Module):
    """MLP drift plus an optional fixed constraint-repair component."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        constraints: ConstraintSet | None = None,
        repair_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.constraints = constraints
        self.repair_rate = float(repair_rate)

        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self._init_small_output()

    def _init_small_output(self) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drift = self.net(x)
        if self.constraints is not None and self.repair_rate != 0.0:
            drift = drift + self.constraints.repair_drift_torch(x, self.repair_rate)
        return drift


def make_drift_field(
    model_type: str,
    dim: int,
    hidden_dim: int,
    depth: int,
    constraints: ConstraintSet | None = None,
    repair_rate: float = 0.0,
) -> nn.Module:
    if model_type == "linear":
        return LinearDriftField(
            dim=dim,
            constraints=constraints,
            repair_rate=repair_rate,
        )
    if model_type == "mlp":
        return MLPDriftField(
            dim=dim,
            hidden_dim=hidden_dim,
            depth=depth,
            constraints=constraints,
            repair_rate=repair_rate,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
