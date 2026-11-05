from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class LinearEqualityConstraint:
    """A linear equality constraint c^T x = offset."""

    name: str
    coefficients: tuple[float, ...]
    offset: float = 0.0
    tolerance: float = 0.20

    def coefficients_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.coefficients, device=device, dtype=dtype)

    def residual_torch(self, x: torch.Tensor) -> torch.Tensor:
        c = self.coefficients_tensor(x.device, x.dtype)
        return x @ c - x.new_tensor(self.offset)

    def residual_np(self, x: np.ndarray) -> np.ndarray:
        c = np.asarray(self.coefficients, dtype=x.dtype)
        return x @ c - self.offset

    def gradient_torch(self, x: torch.Tensor) -> torch.Tensor:
        c = self.coefficients_tensor(x.device, x.dtype)
        return c.unsqueeze(0).expand(x.shape[0], -1)

    def repair_drift_torch(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        """Return drift g such that d(c^T x - offset)/dt = -rate * residual."""
        c = self.coefficients_tensor(x.device, x.dtype)
        denom = torch.dot(c, c).clamp_min(1e-12)
        residual = self.residual_torch(x)
        return -rate * residual[:, None] * c[None, :] / denom

    def repair_divergence(self, rate: float) -> float:
        return -float(rate)

    def residual_velocity_torch(self, x: torch.Tensor, drift: torch.Tensor) -> torch.Tensor:
        c = self.coefficients_tensor(drift.device, drift.dtype)
        return drift @ c


@dataclass(frozen=True)
class NonlinearEqualityConstraint:
    """A differentiable equality constraint r(x) = 0."""

    name: str
    residual_torch_fn: Callable[[torch.Tensor], torch.Tensor]
    residual_np_fn: Callable[[np.ndarray], np.ndarray]
    gradient_torch_fn: Callable[[torch.Tensor], torch.Tensor]
    tolerance: float = 0.20

    def residual_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_torch_fn(x)

    def residual_np(self, x: np.ndarray) -> np.ndarray:
        return self.residual_np_fn(x)

    def gradient_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.gradient_torch_fn(x)

    def repair_drift_torch(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        grad = self.gradient_torch(x)
        denom = grad.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
        residual = self.residual_torch(x)
        return -rate * residual[:, None] * grad / denom

    def repair_divergence(self, rate: float) -> float:
        raise NotImplementedError("Use exact autograd divergence for nonlinear constraints.")

    def residual_velocity_torch(self, x: torch.Tensor, drift: torch.Tensor) -> torch.Tensor:
        grad = self.gradient_torch_fn(x)
        return (grad * drift).sum(dim=1)


class ConstraintSet:
    def __init__(self, constraints: Iterable[LinearEqualityConstraint | NonlinearEqualityConstraint]):
        self.constraints = list(constraints)

    def __len__(self) -> int:
        return len(self.constraints)

    def repair_drift_torch(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        if not self.constraints or rate == 0.0:
            return torch.zeros_like(x)
        if len(self.constraints) == 1:
            return self.constraints[0].repair_drift_torch(x, rate)

        residuals = self.residuals_torch(x)
        jacobian = torch.stack([c.gradient_torch(x) for c in self.constraints], dim=1)
        gram = torch.bmm(jacobian, jacobian.transpose(1, 2))
        eye = torch.eye(len(self.constraints), device=x.device, dtype=x.dtype).unsqueeze(0)
        coeff = torch.linalg.solve(
            gram + 1e-6 * eye,
            residuals.unsqueeze(-1),
        ).squeeze(-1)
        return -rate * torch.einsum("bkd,bk->bd", jacobian, coeff)

    def tangent_projection_torch(self, x: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """Project vectors onto the local tangent space of all constraints."""
        if not self.constraints:
            return vectors
        jacobian = torch.stack([c.gradient_torch(x) for c in self.constraints], dim=1)
        gram = torch.bmm(jacobian, jacobian.transpose(1, 2))
        eye = torch.eye(len(self.constraints), device=x.device, dtype=x.dtype).unsqueeze(0)
        normal_velocity = torch.einsum("bkd,bd->bk", jacobian, vectors)
        coeff = torch.linalg.solve(
            gram + 1e-6 * eye,
            normal_velocity.unsqueeze(-1),
        ).squeeze(-1)
        correction = torch.einsum("bkd,bk->bd", jacobian, coeff)
        return vectors - correction

    def repair_divergence(self, rate: float) -> float:
        return sum(c.repair_divergence(rate) for c in self.constraints)

    def residuals_torch(self, x: torch.Tensor) -> torch.Tensor:
        if not self.constraints:
            return x.new_zeros((x.shape[0], 0))
        return torch.stack([c.residual_torch(x) for c in self.constraints], dim=1)

    def residual_velocities_from_state_torch(
        self,
        x: torch.Tensor,
        drift: torch.Tensor,
    ) -> torch.Tensor:
        if not self.constraints:
            return drift.new_zeros((drift.shape[0], 0))
        return torch.stack(
            [c.residual_velocity_torch(x, drift) for c in self.constraints],
            dim=1,
        )

    def metrics_np(self, x: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        residuals = []
        violations = []
        for constraint in self.constraints:
            residual = constraint.residual_np(x)
            residuals.append(residual)
            violations.append(np.abs(residual) > constraint.tolerance)
            prefix = f"constraint_{constraint.name}"
            out[f"{prefix}_mae"] = float(np.mean(np.abs(residual)))
            out[f"{prefix}_std"] = float(np.std(residual))
            out[f"{prefix}_violation_rate"] = float(
                np.mean(np.abs(residual) > constraint.tolerance)
            )
        if residuals:
            residual_matrix = np.stack(residuals, axis=1)
            violation_matrix = np.stack(violations, axis=1)
            out["constraint_mae"] = float(np.mean(np.abs(residual_matrix)))
            out["constraint_std"] = float(np.std(residual_matrix))
            out["violation_rate"] = float(np.mean(np.any(violation_matrix, axis=1)))
        return out


def accounting_constraints(tolerance: float = 0.20) -> ConstraintSet:
    return ConstraintSet(
        [
            LinearEqualityConstraint(
                name="total_eq_a_plus_b",
                coefficients=(-1.0, -1.0, 1.0),
                offset=0.0,
                tolerance=tolerance,
            )
        ]
    )


def _complex_g1_torch(x: torch.Tensor) -> torch.Tensor:
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    return torch.sin(a) + 0.35 * b.square() - 0.25 * a * c + 0.20 * torch.tanh(b * c)


def _complex_g2_torch(x: torch.Tensor) -> torch.Tensor:
    a, b, c, y = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return torch.tanh(a + b) + 0.50 * y + 0.20 * c.square() - 0.15 * a * c + 0.10 * y * b


def _complex_r1_torch(x: torch.Tensor) -> torch.Tensor:
    return x[:, 3] - _complex_g1_torch(x)


def _complex_r2_torch(x: torch.Tensor) -> torch.Tensor:
    return x[:, 4] - _complex_g2_torch(x)


def _complex_r1_np(x: np.ndarray) -> np.ndarray:
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    return x[:, 3] - (np.sin(a) + 0.35 * b**2 - 0.25 * a * c + 0.20 * np.tanh(b * c))


def _complex_r2_np(x: np.ndarray) -> np.ndarray:
    a, b, c, y = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return x[:, 4] - (np.tanh(a + b) + 0.50 * y + 0.20 * c**2 - 0.15 * a * c + 0.10 * y * b)


def _complex_r1_grad_torch(x: torch.Tensor) -> torch.Tensor:
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    sech2_bc = 1.0 - torch.tanh(b * c).square()
    return torch.stack(
        [
            -torch.cos(a) + 0.25 * c,
            -0.70 * b - 0.20 * c * sech2_bc,
            0.25 * a - 0.20 * b * sech2_bc,
            ones,
            zeros,
        ],
        dim=1,
    )


def _complex_r2_grad_torch(x: torch.Tensor) -> torch.Tensor:
    a, b, c, y = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    ones = torch.ones_like(a)
    sech2_ab = 1.0 - torch.tanh(a + b).square()
    return torch.stack(
        [
            -sech2_ab + 0.15 * c,
            -sech2_ab - 0.10 * y,
            -0.40 * c + 0.15 * a,
            -0.50 - 0.10 * b,
            ones,
        ],
        dim=1,
    )


def complex_ledger_constraints(tolerance: float = 0.15) -> ConstraintSet:
    return ConstraintSet(
        [
            NonlinearEqualityConstraint(
                name="y_eq_stable_nonlinear_chain",
                residual_torch_fn=_complex_r1_torch,
                residual_np_fn=_complex_r1_np,
                gradient_torch_fn=_complex_r1_grad_torch,
                tolerance=tolerance,
            ),
            NonlinearEqualityConstraint(
                name="z_eq_hierarchical_nonlinear_chain",
                residual_torch_fn=_complex_r2_torch,
                residual_np_fn=_complex_r2_np,
                gradient_torch_fn=_complex_r2_grad_torch,
                tolerance=tolerance,
            ),
        ]
    )


def _causal_m1_torch(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    return 0.80 * x1 - 0.45 * x2 + 0.30 * torch.tanh(x1 * x2)


def _causal_m2_torch(x: torch.Tensor) -> torch.Tensor:
    x1, x2, m1 = x[:, 0], x[:, 1], x[:, 2]
    return torch.sin(x1) + 0.55 * m1 + 0.25 * x2.square() - 0.20 * x1 * m1


def _causal_treat_torch(x: torch.Tensor) -> torch.Tensor:
    x1, x2, m1, m2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    score = 0.70 * x1 - 0.50 * x2 + 0.35 * m1 + 0.25 * m2
    return torch.tanh(score) + 0.15 * x1 * x2


def _causal_outcome_torch(x: torch.Tensor) -> torch.Tensor:
    x1, x2, m1, m2, treat = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    return (
        0.80 * treat
        + 0.55 * m2
        + 0.35 * m1 * treat
        - 0.20 * x1.square()
        + 0.15 * x2 * m2
    )


def _causal_r_m1_torch(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2] - _causal_m1_torch(x)


def _causal_r_m2_torch(x: torch.Tensor) -> torch.Tensor:
    return x[:, 3] - _causal_m2_torch(x)


def _causal_r_treat_torch(x: torch.Tensor) -> torch.Tensor:
    return x[:, 4] - _causal_treat_torch(x)


def _causal_r_outcome_torch(x: torch.Tensor) -> torch.Tensor:
    return x[:, 5] - _causal_outcome_torch(x)


def _causal_r_m1_np(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[:, 0], x[:, 1]
    return x[:, 2] - (0.80 * x1 - 0.45 * x2 + 0.30 * np.tanh(x1 * x2))


def _causal_r_m2_np(x: np.ndarray) -> np.ndarray:
    x1, x2, m1 = x[:, 0], x[:, 1], x[:, 2]
    return x[:, 3] - (np.sin(x1) + 0.55 * m1 + 0.25 * x2**2 - 0.20 * x1 * m1)


def _causal_r_treat_np(x: np.ndarray) -> np.ndarray:
    x1, x2, m1, m2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    score = 0.70 * x1 - 0.50 * x2 + 0.35 * m1 + 0.25 * m2
    return x[:, 4] - (np.tanh(score) + 0.15 * x1 * x2)


def _causal_r_outcome_np(x: np.ndarray) -> np.ndarray:
    x1, x2, m1, m2, treat = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    pred = 0.80 * treat + 0.55 * m2 + 0.35 * m1 * treat - 0.20 * x1**2 + 0.15 * x2 * m2
    return x[:, 5] - pred


def _causal_r_m1_grad_torch(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    zeros = torch.zeros_like(x1)
    ones = torch.ones_like(x1)
    sech2 = 1.0 - torch.tanh(x1 * x2).square()
    return torch.stack(
        [
            -0.80 - 0.30 * x2 * sech2,
            0.45 - 0.30 * x1 * sech2,
            ones,
            zeros,
            zeros,
            zeros,
        ],
        dim=1,
    )


def _causal_r_m2_grad_torch(x: torch.Tensor) -> torch.Tensor:
    x1, x2, m1 = x[:, 0], x[:, 1], x[:, 2]
    zeros = torch.zeros_like(x1)
    ones = torch.ones_like(x1)
    return torch.stack(
        [
            -torch.cos(x1) + 0.20 * m1,
            -0.50 * x2,
            -0.55 + 0.20 * x1,
            ones,
            zeros,
            zeros,
        ],
        dim=1,
    )


def _causal_r_treat_grad_torch(x: torch.Tensor) -> torch.Tensor:
    x1, x2, m1, m2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    zeros = torch.zeros_like(x1)
    ones = torch.ones_like(x1)
    score = 0.70 * x1 - 0.50 * x2 + 0.35 * m1 + 0.25 * m2
    sech2 = 1.0 - torch.tanh(score).square()
    return torch.stack(
        [
            -0.70 * sech2 - 0.15 * x2,
            0.50 * sech2 - 0.15 * x1,
            -0.35 * sech2,
            -0.25 * sech2,
            ones,
            zeros,
        ],
        dim=1,
    )


def _causal_r_outcome_grad_torch(x: torch.Tensor) -> torch.Tensor:
    x1, x2, m1, m2, treat = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    ones = torch.ones_like(x1)
    return torch.stack(
        [
            0.40 * x1,
            -0.15 * m2,
            -0.35 * treat,
            -0.55 - 0.15 * x2,
            -0.80 - 0.35 * m1,
            ones,
        ],
        dim=1,
    )


def causal_dag_constraints(tolerance: float = 0.12) -> ConstraintSet:
    return ConstraintSet(
        [
            NonlinearEqualityConstraint(
                name="m1_eq_causes_x1_x2",
                residual_torch_fn=_causal_r_m1_torch,
                residual_np_fn=_causal_r_m1_np,
                gradient_torch_fn=_causal_r_m1_grad_torch,
                tolerance=tolerance,
            ),
            NonlinearEqualityConstraint(
                name="m2_eq_causes_x1_x2_m1",
                residual_torch_fn=_causal_r_m2_torch,
                residual_np_fn=_causal_r_m2_np,
                gradient_torch_fn=_causal_r_m2_grad_torch,
                tolerance=tolerance,
            ),
            NonlinearEqualityConstraint(
                name="treat_eq_propensity_parents",
                residual_torch_fn=_causal_r_treat_torch,
                residual_np_fn=_causal_r_treat_np,
                gradient_torch_fn=_causal_r_treat_grad_torch,
                tolerance=tolerance,
            ),
            NonlinearEqualityConstraint(
                name="outcome_eq_structural_response",
                residual_torch_fn=_causal_r_outcome_torch,
                residual_np_fn=_causal_r_outcome_np,
                gradient_torch_fn=_causal_r_outcome_grad_torch,
                tolerance=tolerance,
            ),
        ]
    )


def panel_financial_constraints(
    tolerance: float = 0.08,
    n_assets: int = 2,
    horizon: int = 3,
) -> ConstraintSet:
    dim = n_assets * horizon + n_assets * (horizon + 1) + n_assets * horizon

    def r_idx(asset: int, t: int) -> int:
        return asset * horizon + (t - 1)

    def logp_idx(asset: int, t: int) -> int:
        return n_assets * horizon + asset * (horizon + 1) + t

    def logvol_idx(asset: int, t: int) -> int:
        return n_assets * horizon + n_assets * (horizon + 1) + asset * horizon + (t - 1)

    constraints: list[NonlinearEqualityConstraint] = []

    for asset in range(n_assets):
        label = chr(ord("a") + asset)
        for t in range(1, horizon + 1):
            ri = r_idx(asset, t)
            lp_prev = logp_idx(asset, t - 1)
            lp_now = logp_idx(asset, t)

            def residual_torch(x, ri=ri, lp_prev=lp_prev, lp_now=lp_now):
                return x[:, lp_now] - x[:, lp_prev] - x[:, ri]

            def residual_np(x, ri=ri, lp_prev=lp_prev, lp_now=lp_now):
                return x[:, lp_now] - x[:, lp_prev] - x[:, ri]

            def grad_torch(x, ri=ri, lp_prev=lp_prev, lp_now=lp_now):
                grad = torch.zeros((x.shape[0], dim), device=x.device, dtype=x.dtype)
                grad[:, lp_now] = 1.0
                grad[:, lp_prev] = -1.0
                grad[:, ri] = -1.0
                return grad

            constraints.append(
                NonlinearEqualityConstraint(
                    name=f"logprice_{label}_{t}_recursion",
                    residual_torch_fn=residual_torch,
                    residual_np_fn=residual_np,
                    gradient_torch_fn=grad_torch,
                    tolerance=tolerance,
                )
            )

    for asset in range(n_assets):
        other = 1 - asset
        label = chr(ord("a") + asset)
        for t in range(2, horizon + 1):
            hv_prev = logvol_idx(asset, t - 1)
            hv_now = logvol_idx(asset, t)
            r_self_prev = r_idx(asset, t - 1)
            r_other_prev = r_idx(other, t - 1)

            def residual_torch(
                x,
                hv_prev=hv_prev,
                hv_now=hv_now,
                r_self_prev=r_self_prev,
                r_other_prev=r_other_prev,
            ):
                pred = (
                    0.65 * x[:, hv_prev]
                    + 0.20 * torch.tanh(3.0 * x[:, r_self_prev])
                    + 0.10 * x[:, r_other_prev]
                )
                return x[:, hv_now] - pred

            def residual_np(
                x,
                hv_prev=hv_prev,
                hv_now=hv_now,
                r_self_prev=r_self_prev,
                r_other_prev=r_other_prev,
            ):
                pred = (
                    0.65 * x[:, hv_prev]
                    + 0.20 * np.tanh(3.0 * x[:, r_self_prev])
                    + 0.10 * x[:, r_other_prev]
                )
                return x[:, hv_now] - pred

            def grad_torch(
                x,
                hv_prev=hv_prev,
                hv_now=hv_now,
                r_self_prev=r_self_prev,
                r_other_prev=r_other_prev,
            ):
                grad = torch.zeros((x.shape[0], dim), device=x.device, dtype=x.dtype)
                sech2 = 1.0 - torch.tanh(3.0 * x[:, r_self_prev]).square()
                grad[:, hv_now] = 1.0
                grad[:, hv_prev] = -0.65
                grad[:, r_self_prev] = -0.60 * sech2
                grad[:, r_other_prev] = -0.10
                return grad

            constraints.append(
                NonlinearEqualityConstraint(
                    name=f"logvol_{label}_{t}_lag_spillover",
                    residual_torch_fn=residual_torch,
                    residual_np_fn=residual_np,
                    gradient_torch_fn=grad_torch,
                    tolerance=tolerance,
                )
            )

    return ConstraintSet(constraints)
