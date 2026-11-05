from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from constrained_tabular_fm.constraints import (
    ConstraintSet,
    accounting_constraints,
    complex_ledger_constraints,
    causal_dag_constraints,
    panel_financial_constraints,
)


@dataclass
class AccountingTableSpec:
    """Three-column continuous table with total = a + b + noise."""

    residual_std: float = 0.08
    violation_threshold: float = 0.20

    @property
    def columns(self) -> tuple[str, ...]:
        return ("a", "b", "total")

    @property
    def dim(self) -> int:
        return 3

    @property
    def constraints(self) -> ConstraintSet:
        return accounting_constraints(self.violation_threshold)

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        a = torch.randn(n, device=device)
        b = torch.randn(n, device=device)
        residual = self.residual_std * torch.randn(n, device=device)
        total = a + b + residual
        return torch.stack([a, b, total], dim=1)

    def sample_constraint_probe(
        self,
        n: int,
        residual_scale: float,
        device: torch.device,
    ) -> torch.Tensor:
        a = 1.5 * torch.randn(n, device=device)
        b = 1.5 * torch.randn(n, device=device)
        standardized_residual = residual_scale * torch.randn(n, device=device)
        total = a + b + self.residual_std * standardized_residual
        return torch.stack([a, b, total], dim=1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, 0]
        b = x[:, 1]
        residual = x[:, 2] - x[:, 0] - x[:, 1]
        inv_var = 1.0 / (self.residual_std * self.residual_std)
        return torch.stack(
            [
                -a + residual * inv_var,
                -b + residual * inv_var,
                -residual * inv_var,
            ],
            dim=1,
        )

    def div_score(self, x: torch.Tensor) -> torch.Tensor:
        inv_var = 1.0 / (self.residual_std * self.residual_std)
        return torch.full((x.shape[0],), -2.0 - 3.0 * inv_var, device=x.device)


@dataclass
class ComplexLedgerSpec:
    """Five-column table with two nonlinear hierarchical constraints.

    Columns:

    - a, b, c: independent source variables
    - y: nonlinear derived column
    - z: nonlinear second-level derived column

    The near-hard relationships are:

        y = sin(a) + 0.35 b^2 - 0.25 a c + 0.20 tanh(b c) + eps_1
        z = tanh(a + b) + 0.50 y + 0.20 c^2 - 0.15 a c + 0.10 y b + eps_2

    This remains analytically tractable because the transformation from
    (a, b, c, eps_1, eps_2) to (a, b, c, y, z) is triangular with determinant 1.
    """

    residual_std: float = 0.05
    violation_threshold: float = 0.15

    @property
    def columns(self) -> tuple[str, ...]:
        return ("a", "b", "c", "y", "z")

    @property
    def dim(self) -> int:
        return 5

    @property
    def constraints(self) -> ConstraintSet:
        return complex_ledger_constraints(self.violation_threshold)

    def _g1(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.sin(a) + 0.35 * b.square() - 0.25 * a * c + 0.20 * torch.tanh(b * c)

    def _g2(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(a + b) + 0.50 * y + 0.20 * c.square() - 0.15 * a * c + 0.10 * y * b

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        a = torch.randn(n, device=device)
        b = torch.randn(n, device=device)
        c = torch.randn(n, device=device)
        eps1 = self.residual_std * torch.randn(n, device=device)
        eps2 = self.residual_std * torch.randn(n, device=device)
        y = self._g1(a, b, c) + eps1
        z = self._g2(a, b, c, y) + eps2
        return torch.stack([a, b, c, y, z], dim=1)

    def sample_constraint_probe(
        self,
        n: int,
        residual_scale: float,
        device: torch.device,
    ) -> torch.Tensor:
        a = 1.4 * torch.randn(n, device=device)
        b = 1.4 * torch.randn(n, device=device)
        c = 1.4 * torch.randn(n, device=device)
        eps1 = self.residual_std * residual_scale * torch.randn(n, device=device)
        eps2 = self.residual_std * residual_scale * torch.randn(n, device=device)
        y = self._g1(a, b, c) + eps1
        z = self._g2(a, b, c, y) + eps2
        return torch.stack([a, b, c, y, z], dim=1)

    def residuals(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b, c, y, z = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        r1 = y - self._g1(a, b, c)
        r2 = z - self._g2(a, b, c, y)
        return r1, r2

    def log_prob_unnormalized(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = x[:, 0], x[:, 1], x[:, 2]
        r1, r2 = self.residuals(x)
        inv_var = 1.0 / (self.residual_std * self.residual_std)
        return -0.5 * (a.square() + b.square() + c.square()) - 0.5 * inv_var * (
            r1.square() + r2.square()
        )

    def score(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logp = self.log_prob_unnormalized(x).sum()
        return torch.autograd.grad(logp, x, create_graph=True, retain_graph=True)[0]

    def div_score(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        score = self.score(x)
        parts = []
        for idx in range(self.dim):
            grad = torch.autograd.grad(
                score[:, idx].sum(),
                x,
                create_graph=True,
                retain_graph=True,
            )[0][:, idx]
            parts.append(grad)
        return torch.stack(parts, dim=1).sum(dim=1)


@dataclass
class CausalDAGSpec:
    """Six-column table generated by nonlinear structural causal equations.

    DAG:

        x1, x2 -> m1
        x1, x2, m1 -> m2
        x1, x2, m1, m2 -> treat
        x1, x2, m1, m2, treat -> outcome

    The structural equations are triangular with additive small noise. That
    makes this a compact benchmark for causal-structure-aware tabular synthesis:
    baseline models see a normal continuous table, while constrained Flux also
    sees the DAG equations as equality constraints.
    """

    residual_std: float = 0.04
    violation_threshold: float = 0.12

    @property
    def columns(self) -> tuple[str, ...]:
        return ("x1", "x2", "m1", "m2", "treat", "outcome")

    @property
    def dim(self) -> int:
        return 6

    @property
    def constraints(self) -> ConstraintSet:
        return causal_dag_constraints(self.violation_threshold)

    def _g_m1(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 0.80 * x1 - 0.45 * x2 + 0.30 * torch.tanh(x1 * x2)

    def _g_m2(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        m1: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sin(x1) + 0.55 * m1 + 0.25 * x2.square() - 0.20 * x1 * m1

    def _g_treat(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        m1: torch.Tensor,
        m2: torch.Tensor,
    ) -> torch.Tensor:
        score = 0.70 * x1 - 0.50 * x2 + 0.35 * m1 + 0.25 * m2
        return torch.tanh(score) + 0.15 * x1 * x2

    def _g_outcome(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        m1: torch.Tensor,
        m2: torch.Tensor,
        treat: torch.Tensor,
    ) -> torch.Tensor:
        return (
            0.80 * treat
            + 0.55 * m2
            + 0.35 * m1 * treat
            - 0.20 * x1.square()
            + 0.15 * x2 * m2
        )

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        x1 = torch.randn(n, device=device)
        x2 = torch.randn(n, device=device)
        eps = self.residual_std * torch.randn(n, 4, device=device)
        m1 = self._g_m1(x1, x2) + eps[:, 0]
        m2 = self._g_m2(x1, x2, m1) + eps[:, 1]
        treat = self._g_treat(x1, x2, m1, m2) + eps[:, 2]
        outcome = self._g_outcome(x1, x2, m1, m2, treat) + eps[:, 3]
        return torch.stack([x1, x2, m1, m2, treat, outcome], dim=1)

    def sample_constraint_probe(
        self,
        n: int,
        residual_scale: float,
        device: torch.device,
    ) -> torch.Tensor:
        x1 = 1.35 * torch.randn(n, device=device)
        x2 = 1.35 * torch.randn(n, device=device)
        eps = self.residual_std * residual_scale * torch.randn(n, 4, device=device)
        m1 = self._g_m1(x1, x2) + eps[:, 0]
        m2 = self._g_m2(x1, x2, m1) + eps[:, 1]
        treat = self._g_treat(x1, x2, m1, m2) + eps[:, 2]
        outcome = self._g_outcome(x1, x2, m1, m2, treat) + eps[:, 3]
        return torch.stack([x1, x2, m1, m2, treat, outcome], dim=1)

    def residuals(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, x2, m1, m2, treat, outcome = (
            x[:, 0],
            x[:, 1],
            x[:, 2],
            x[:, 3],
            x[:, 4],
            x[:, 5],
        )
        r_m1 = m1 - self._g_m1(x1, x2)
        r_m2 = m2 - self._g_m2(x1, x2, m1)
        r_treat = treat - self._g_treat(x1, x2, m1, m2)
        r_outcome = outcome - self._g_outcome(x1, x2, m1, m2, treat)
        return r_m1, r_m2, r_treat, r_outcome

    def log_prob_unnormalized(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        residuals = self.residuals(x)
        residual_energy = sum(r.square() for r in residuals)
        inv_var = 1.0 / (self.residual_std * self.residual_std)
        return -0.5 * (x1.square() + x2.square()) - 0.5 * inv_var * residual_energy

    def score(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        logp = self.log_prob_unnormalized(x).sum()
        return torch.autograd.grad(logp, x, create_graph=True, retain_graph=True)[0]

    def div_score(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        score = self.score(x)
        parts = []
        for idx in range(self.dim):
            grad = torch.autograd.grad(
                score[:, idx].sum(),
                x,
                create_graph=True,
                retain_graph=True,
            )[0][:, idx]
            parts.append(grad)
        return torch.stack(parts, dim=1).sum(dim=1)


@dataclass
class PanelFinancialSpec:
    """Flattened two-asset, three-period financial panel.

    Each row is a short panel window. It contains returns, log prices, and
    log-volatility states for two assets. The near-hard constraints encode:

        logp_i,t = logp_i,t-1 + r_i,t + eps_price
        h_i,t = 0.65 h_i,t-1 + 0.20 tanh(3 r_i,t-1)
                + 0.10 r_j,t-1 + eps_vol

    The second relation gives the synthetic data a genuine time-series flavor:
    current volatility depends on lagged own return and lagged cross-asset return.
    """

    residual_std: float = 0.03
    violation_threshold: float = 0.08
    n_assets: int = 2
    horizon: int = 3
    return_std: float = 0.45
    temporal_corr: float = 0.35
    cross_asset_corr: float = 0.55
    logp0_std: float = 0.35
    logvol1_mean: float = -0.70
    logvol1_std: float = 0.25

    @property
    def columns(self) -> tuple[str, ...]:
        labels = tuple(chr(ord("a") + idx) for idx in range(self.n_assets))
        returns = tuple(
            f"r_{label}_{t}"
            for label in labels
            for t in range(1, self.horizon + 1)
        )
        log_prices = tuple(
            f"logp_{label}_{t}"
            for label in labels
            for t in range(0, self.horizon + 1)
        )
        log_vols = tuple(
            f"logvol_{label}_{t}"
            for label in labels
            for t in range(1, self.horizon + 1)
        )
        return returns + log_prices + log_vols

    @property
    def dim(self) -> int:
        return (
            self.n_assets * self.horizon
            + self.n_assets * (self.horizon + 1)
            + self.n_assets * self.horizon
        )

    @property
    def constraints(self) -> ConstraintSet:
        return panel_financial_constraints(
            tolerance=self.violation_threshold,
            n_assets=self.n_assets,
            horizon=self.horizon,
        )

    def r_idx(self, asset: int, t: int) -> int:
        return asset * self.horizon + (t - 1)

    def logp_idx(self, asset: int, t: int) -> int:
        return self.n_assets * self.horizon + asset * (self.horizon + 1) + t

    def logvol_idx(self, asset: int, t: int) -> int:
        return (
            self.n_assets * self.horizon
            + self.n_assets * (self.horizon + 1)
            + asset * self.horizon
            + (t - 1)
        )

    def return_covariance_np(self) -> np.ndarray:
        time = np.fromfunction(
            lambda i, j: self.temporal_corr ** np.abs(i - j),
            (self.horizon, self.horizon),
            dtype=float,
        )
        asset = np.full((self.n_assets, self.n_assets), self.cross_asset_corr)
        np.fill_diagonal(asset, 1.0)
        return (self.return_std**2) * np.kron(asset, time)

    def _return_covariance_torch(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(self.return_covariance_np(), device=device, dtype=dtype)

    def _return_precision_torch(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cov = self._return_covariance_torch(device, dtype)
        return torch.linalg.inv(cov)

    def _sample(
        self,
        n: int,
        device: torch.device,
        residual_scale: float,
        prior_scale: float,
    ) -> torch.Tensor:
        dtype = torch.float32
        cov = self._return_covariance_torch(device, dtype)
        chol = torch.linalg.cholesky(cov)
        returns = prior_scale * (torch.randn(n, self.n_assets * self.horizon, device=device) @ chol.T)

        x = torch.zeros(n, self.dim, device=device)
        x[:, : self.n_assets * self.horizon] = returns

        for asset in range(self.n_assets):
            lp0 = prior_scale * self.logp0_std * torch.randn(n, device=device)
            h1 = self.logvol1_mean + prior_scale * self.logvol1_std * torch.randn(n, device=device)
            x[:, self.logp_idx(asset, 0)] = lp0
            x[:, self.logvol_idx(asset, 1)] = h1

        noise_std = self.residual_std * residual_scale
        for asset in range(self.n_assets):
            for t in range(1, self.horizon + 1):
                price_noise = noise_std * torch.randn(n, device=device)
                x[:, self.logp_idx(asset, t)] = (
                    x[:, self.logp_idx(asset, t - 1)]
                    + x[:, self.r_idx(asset, t)]
                    + price_noise
                )

        for asset in range(self.n_assets):
            other = 1 - asset
            for t in range(2, self.horizon + 1):
                vol_noise = noise_std * torch.randn(n, device=device)
                x[:, self.logvol_idx(asset, t)] = (
                    0.65 * x[:, self.logvol_idx(asset, t - 1)]
                    + 0.20 * torch.tanh(3.0 * x[:, self.r_idx(asset, t - 1)])
                    + 0.10 * x[:, self.r_idx(other, t - 1)]
                    + vol_noise
                )
        return x

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return self._sample(n, device, residual_scale=1.0, prior_scale=1.0)

    def sample_constraint_probe(
        self,
        n: int,
        residual_scale: float,
        device: torch.device,
    ) -> torch.Tensor:
        return self._sample(n, device, residual_scale=residual_scale, prior_scale=1.35)

    def _price_residual(self, x: torch.Tensor, asset: int, t: int) -> torch.Tensor:
        return (
            x[:, self.logp_idx(asset, t)]
            - x[:, self.logp_idx(asset, t - 1)]
            - x[:, self.r_idx(asset, t)]
        )

    def _vol_residual(self, x: torch.Tensor, asset: int, t: int) -> torch.Tensor:
        other = 1 - asset
        pred = (
            0.65 * x[:, self.logvol_idx(asset, t - 1)]
            + 0.20 * torch.tanh(3.0 * x[:, self.r_idx(asset, t - 1)])
            + 0.10 * x[:, self.r_idx(other, t - 1)]
        )
        return x[:, self.logvol_idx(asset, t)] - pred

    def residuals(self, x: torch.Tensor) -> torch.Tensor:
        values = []
        for asset in range(self.n_assets):
            for t in range(1, self.horizon + 1):
                values.append(self._price_residual(x, asset, t))
        for asset in range(self.n_assets):
            for t in range(2, self.horizon + 1):
                values.append(self._vol_residual(x, asset, t))
        return torch.stack(values, dim=1)

    def log_prob_unnormalized(self, x: torch.Tensor) -> torch.Tensor:
        returns = x[:, : self.n_assets * self.horizon]
        precision = self._return_precision_torch(x.device, x.dtype)
        ret_quad = torch.einsum("bi,ij,bj->b", returns, precision, returns)

        lp0 = torch.stack(
            [x[:, self.logp_idx(asset, 0)] for asset in range(self.n_assets)],
            dim=1,
        )
        h1 = torch.stack(
            [x[:, self.logvol_idx(asset, 1)] for asset in range(self.n_assets)],
            dim=1,
        )
        residuals = self.residuals(x)
        inv_residual_var = 1.0 / (self.residual_std * self.residual_std)
        return (
            -0.5 * ret_quad
            -0.5 * (lp0 / self.logp0_std).square().sum(dim=1)
            -0.5 * ((h1 - self.logvol1_mean) / self.logvol1_std).square().sum(dim=1)
            -0.5 * inv_residual_var * residuals.square().sum(dim=1)
        )

    def score(self, x: torch.Tensor) -> torch.Tensor:
        precision = self._return_precision_torch(x.device, x.dtype)
        score = torch.zeros_like(x)
        n_return_cols = self.n_assets * self.horizon
        score[:, :n_return_cols] = -(x[:, :n_return_cols] @ precision)

        for asset in range(self.n_assets):
            score[:, self.logp_idx(asset, 0)] += -x[:, self.logp_idx(asset, 0)] / (
                self.logp0_std * self.logp0_std
            )
            score[:, self.logvol_idx(asset, 1)] += -(
                x[:, self.logvol_idx(asset, 1)] - self.logvol1_mean
            ) / (self.logvol1_std * self.logvol1_std)

        inv_residual_var = 1.0 / (self.residual_std * self.residual_std)
        for asset in range(self.n_assets):
            for t in range(1, self.horizon + 1):
                residual = self._price_residual(x, asset, t)
                scaled = inv_residual_var * residual
                score[:, self.logp_idx(asset, t)] += -scaled
                score[:, self.logp_idx(asset, t - 1)] += scaled
                score[:, self.r_idx(asset, t)] += scaled

        for asset in range(self.n_assets):
            other = 1 - asset
            for t in range(2, self.horizon + 1):
                residual = self._vol_residual(x, asset, t)
                scaled = inv_residual_var * residual
                r_prev = x[:, self.r_idx(asset, t - 1)]
                sech2 = 1.0 - torch.tanh(3.0 * r_prev).square()
                score[:, self.logvol_idx(asset, t)] += -scaled
                score[:, self.logvol_idx(asset, t - 1)] += 0.65 * scaled
                score[:, self.r_idx(asset, t - 1)] += 0.60 * sech2 * scaled
                score[:, self.r_idx(other, t - 1)] += 0.10 * scaled
        return score

    def div_score(self, x: torch.Tensor) -> torch.Tensor:
        precision = self._return_precision_torch(x.device, x.dtype)
        inv_residual_var = 1.0 / (self.residual_std * self.residual_std)
        base = (
            -torch.trace(precision)
            - self.n_assets / (self.logp0_std * self.logp0_std)
            - self.n_assets / (self.logvol1_std * self.logvol1_std)
        )
        div = x.new_full((x.shape[0],), 0.0)
        div = div + base

        for _asset in range(self.n_assets):
            for _t in range(1, self.horizon + 1):
                div += -inv_residual_var * 3.0

        for asset in range(self.n_assets):
            for t in range(2, self.horizon + 1):
                residual = self._vol_residual(x, asset, t)
                r_prev = x[:, self.r_idx(asset, t - 1)]
                tanh = torch.tanh(3.0 * r_prev)
                sech2 = 1.0 - tanh.square()
                grad_norm_sq = 1.0 + 0.65**2 + 0.10**2 + (0.60 * sech2).square()
                laplacian_residual = 3.60 * sech2 * tanh
                div += -inv_residual_var * (grad_norm_sq + residual * laplacian_residual)
        return div


def make_table_spec(
    dataset: str,
    residual_std: float,
    violation_threshold: float,
) -> AccountingTableSpec | ComplexLedgerSpec | CausalDAGSpec | PanelFinancialSpec:
    if dataset == "accounting":
        return AccountingTableSpec(
            residual_std=residual_std,
            violation_threshold=violation_threshold,
        )
    if dataset == "complex":
        return ComplexLedgerSpec(
            residual_std=residual_std,
            violation_threshold=violation_threshold,
        )
    if dataset == "causal":
        return CausalDAGSpec(
            residual_std=residual_std,
            violation_threshold=violation_threshold,
        )
    if dataset == "panel":
        return PanelFinancialSpec(
            residual_std=residual_std,
            violation_threshold=violation_threshold,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def save_table_csv(path: str | Path, values: np.ndarray, columns: tuple[str, ...]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(columns)
    np.savetxt(path, values, delimiter=",", header=header, comments="")
