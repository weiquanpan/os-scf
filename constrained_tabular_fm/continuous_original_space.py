from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from constrained_tabular_fm.data import make_table_spec, save_table_csv
from constrained_tabular_fm.metrics import evaluate_samples, write_metrics_csv
from constrained_tabular_fm.train import choose_device, set_seed


@dataclass
class ContinuousOriginalSpaceConfig:
    dataset: str
    seed: int = 13
    n_train_steps: int = 1200
    batch_size: int = 256
    sample_steps: int = 200
    lr: float = 1e-3
    hidden_dim: int = 128
    depth: int = 3
    residual_std: float = 0.05
    violation_threshold: float = 0.15
    sample_rows: int = 4000
    device: str = "auto"
    method_name: str = "original_space_structural_coordinate_flow"
    # Preconditioning / training upgrades for the free-coordinate flow.
    # Default is standardize + (shrinkage) ZCA whitening. Per-column rank
    # Gaussianization is available but OFF by default: it only ever matched
    # whitening on the already-Gaussian synthetic free spaces, while badly
    # distorting real columns with ties / zero-spikes (e.g. count features).
    normalizer: str = "gaussianize"  # "standard" | "gaussianize"
    whiten: bool = True
    gaussianize: bool = False
    whiten_shrinkage: float = 0.0
    integrator: str = "heun"  # "euler" | "heun"
    ema_decay: float = 0.999
    time_sampling: str = "logitnormal"  # "uniform" | "logitnormal"


class StandardNormalizer:
    def __init__(self, x: torch.Tensor) -> None:
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True).clamp_min(1e-5)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.detach().cpu().reshape(-1).tolist(),
            "std": self.std.detach().cpu().reshape(-1).tolist(),
        }


_SQRT2 = float(np.sqrt(2.0))


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _normal_icdf(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1.0 - 1e-6)
    return _SQRT2 * torch.erfinv(2.0 * p - 1.0)


class GaussianizingNormalizer:
    """Precondition structural free coordinates for a small rectified flow.

    The marginal stage is either per-column standardization (default in the
    factory below) or optional empirical-CDF rank Gaussianization. The second
    stage is optional shrinkage ZCA whitening. Rank Gaussianization is kept as
    an ablation because it can distort tied or zero-inflated real columns.
    """

    def __init__(
        self,
        x: torch.Tensor,
        whiten: bool = True,
        gaussianize: bool = True,
        whiten_shrinkage: float = 0.0,
    ) -> None:
        self.device = x.device
        self.dtype = x.dtype
        self.gaussianize = gaussianize
        self.whiten_shrinkage = float(min(max(whiten_shrinkage, 0.0), 1.0))
        # Sorted reference quantiles for the empirical CDF / inverse CDF.
        sorted_x, _ = torch.sort(x, dim=0)
        self.sorted = sorted_x  # (n, d)
        n = x.shape[0]
        # Plotting positions in (0, 1), avoiding the singular tails.
        self.grid = (torch.arange(n, device=x.device, dtype=x.dtype) + 0.5) / n
        if not gaussianize:
            # Fall back to per-column standardization for the marginal stage so
            # that discrete / heavy-tailed real columns are not distorted by an
            # empirical-CDF round trip; whitening (below) can still apply.
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True).clamp_min(1e-5)
        z = self._marginal(x)
        self.whiten = whiten
        if whiten:
            cov = torch.atleast_2d(torch.cov(z.T))
            d = cov.shape[0]
            eye = torch.eye(d, device=x.device, dtype=x.dtype)
            # Shrinkage toward identity: alpha=0 -> full ZCA, alpha=1 -> no whitening.
            alpha = self.whiten_shrinkage
            cov = (1.0 - alpha) * cov + alpha * eye + 1e-4 * eye
            evals, evecs = torch.linalg.eigh(cov)
            evals = evals.clamp_min(1e-6)
            self.W = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
            self.W_inv = evecs @ torch.diag(evals.sqrt()) @ evecs.T
        else:
            self.W = None
            self.W_inv = None

    def _marginal(self, x: torch.Tensor) -> torch.Tensor:
        if self.gaussianize:
            return self._gaussianize(x)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def _inverse_marginal(self, z: torch.Tensor) -> torch.Tensor:
        if self.gaussianize:
            return self._inverse_gaussianize(z)
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def _gaussianize(self, x: torch.Tensor) -> torch.Tensor:
        cols = []
        for j in range(x.shape[1]):
            ranks = torch.searchsorted(self.sorted[:, j].contiguous(), x[:, j].contiguous())
            ranks = ranks.clamp(0, self.grid.shape[0] - 1)
            cols.append(_normal_icdf(self.grid[ranks]))
        return torch.stack(cols, dim=1)

    def _inverse_gaussianize(self, z: torch.Tensor) -> torch.Tensor:
        p = _normal_cdf(z)
        n = self.grid.shape[0]
        pos = (p * n - 0.5).clamp(0, n - 1)
        lo = pos.floor().long()
        hi = (lo + 1).clamp_max(n - 1)
        frac = (pos - lo.to(pos.dtype)).clamp(0.0, 1.0)
        cols = []
        for j in range(z.shape[1]):
            col = self.sorted[:, j]
            cols.append(col[lo[:, j]] * (1 - frac[:, j]) + col[hi[:, j]] * frac[:, j])
        return torch.stack(cols, dim=1)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        z = self._marginal(x.to(self.device))
        if self.whiten:
            z = z @ self.W.T
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        if self.whiten:
            z = z @ self.W_inv.T
        return self._inverse_marginal(z)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "rank_gaussianize" if self.gaussianize else "standardize",
            "whiten": bool(self.whiten),
            "whiten_shrinkage": self.whiten_shrinkage,
            "n_ref": int(self.grid.shape[0]),
        }


def make_free_flow_normalizer(
    x: torch.Tensor,
    whiten_shrinkage: float = 0.5,
) -> "GaussianizingNormalizer":
    """Standardize + shrinkage ZCA-whitening preconditioner for free-coordinate
    flows. Whitening decorrelates the free coordinates so the small MLP only has
    to transport an isotropic Gaussian; shrinkage toward identity keeps the
    transform well-behaved on heterogeneous real columns (default 0.5). Rank
    Gaussianization is intentionally disabled (see ContinuousOriginalSpaceConfig).
    """
    return GaussianizingNormalizer(
        x, whiten=True, gaussianize=False, whiten_shrinkage=whiten_shrinkage
    )


def make_normalizer(x: torch.Tensor, cfg: ContinuousOriginalSpaceConfig):
    if cfg.normalizer == "gaussianize":
        return GaussianizingNormalizer(
            x,
            whiten=cfg.whiten,
            gaussianize=cfg.gaussianize,
            whiten_shrinkage=cfg.whiten_shrinkage,
        )
    return StandardNormalizer(x)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # Warmup-adjusted decay so short training runs are not dominated by the
        # random initialization: d_t = min(decay, (1+t)/(10+t)).
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


class TimeVelocity(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.dim = dim
        layers: list[nn.Module] = []
        in_dim = dim + 1
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


def load_table_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)


def encode_free_coordinates(dataset: str, spec, x: torch.Tensor) -> torch.Tensor:
    if dataset == "accounting":
        return x[:, :2]
    if dataset == "complex":
        return x[:, :3]
    if dataset == "causal":
        return x[:, :2]
    if dataset == "panel":
        parts = [x[:, : spec.n_assets * spec.horizon]]
        parts.append(
            torch.stack(
                [x[:, spec.logp_idx(asset, 0)] for asset in range(spec.n_assets)],
                dim=1,
            )
        )
        parts.append(
            torch.stack(
                [x[:, spec.logvol_idx(asset, 1)] for asset in range(spec.n_assets)],
                dim=1,
            )
        )
        return torch.cat(parts, dim=1)
    raise ValueError(f"Unsupported dataset: {dataset}")


def decode_free_coordinates(dataset: str, spec, u: torch.Tensor) -> torch.Tensor:
    if dataset == "accounting":
        a, b = u[:, 0], u[:, 1]
        return torch.stack([a, b, a + b], dim=1)

    if dataset == "complex":
        a, b, c = u[:, 0], u[:, 1], u[:, 2]
        y = spec._g1(a, b, c)
        z = spec._g2(a, b, c, y)
        return torch.stack([a, b, c, y, z], dim=1)

    if dataset == "causal":
        x1, x2 = u[:, 0], u[:, 1]
        m1 = spec._g_m1(x1, x2)
        m2 = spec._g_m2(x1, x2, m1)
        treat = spec._g_treat(x1, x2, m1, m2)
        outcome = spec._g_outcome(x1, x2, m1, m2, treat)
        return torch.stack([x1, x2, m1, m2, treat, outcome], dim=1)

    if dataset == "panel":
        n = u.shape[0]
        x = torch.zeros(n, spec.dim, device=u.device, dtype=u.dtype)
        n_returns = spec.n_assets * spec.horizon
        x[:, :n_returns] = u[:, :n_returns]
        offset = n_returns
        for asset in range(spec.n_assets):
            x[:, spec.logp_idx(asset, 0)] = u[:, offset + asset]
        offset += spec.n_assets
        for asset in range(spec.n_assets):
            x[:, spec.logvol_idx(asset, 1)] = u[:, offset + asset]

        for asset in range(spec.n_assets):
            for t in range(1, spec.horizon + 1):
                x[:, spec.logp_idx(asset, t)] = (
                    x[:, spec.logp_idx(asset, t - 1)] + x[:, spec.r_idx(asset, t)]
                )

        for asset in range(spec.n_assets):
            other = 1 - asset
            for t in range(2, spec.horizon + 1):
                x[:, spec.logvol_idx(asset, t)] = (
                    0.65 * x[:, spec.logvol_idx(asset, t - 1)]
                    + 0.20 * torch.tanh(3.0 * x[:, spec.r_idx(asset, t - 1)])
                    + 0.10 * x[:, spec.r_idx(other, t - 1)]
                )
        return x

    raise ValueError(f"Unsupported dataset: {dataset}")


def _sample_time(batch: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "logitnormal":
        # SD3/Flux-style logit-normal schedule: emphasizes mid-trajectory times
        # where the velocity field is hardest to fit.
        return torch.sigmoid(torch.randn(batch, 1, device=device))
    return torch.rand(batch, 1, device=device)


def train_free_flow(
    u_train: torch.Tensor,
    cfg: ContinuousOriginalSpaceConfig,
) -> tuple[TimeVelocity, object, list[dict[str, float]]]:
    normalizer = make_normalizer(u_train, cfg)
    target = normalizer.transform(u_train)
    model = TimeVelocity(target.shape[1], cfg.hidden_dim, cfg.depth).to(target.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_train_steps)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay and cfg.ema_decay > 0 else None
    history: list[dict[str, float]] = []

    for step in range(1, cfg.n_train_steps + 1):
        idx = torch.randint(0, target.shape[0], (cfg.batch_size,), device=target.device)
        u1 = target[idx]
        u0 = torch.randn_like(u1)
        t = _sample_time(u1.shape[0], target.device, cfg.time_sampling)
        u_t = (1.0 - t) * u0 + t * u1
        pred = model(u_t, t)
        # Engineering objective: rectified-flow velocity regression on structural
        # free coordinates. This is Flux/v-Flux motivated, but it is not the full
        # KDE/projected-Fisher Flux Matching loss from the source paper.
        loss = F.mse_loss(pred, u1 - u0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        if step == 1 or step % max(1, cfg.n_train_steps // 10) == 0:
            history.append({"step": step, "flow_loss": float(loss.detach().cpu())})

    if ema is not None:
        ema.copy_to(model)
    return model, normalizer, history


@torch.no_grad()
def sample_free_flow(
    model: TimeVelocity,
    normalizer: object,
    n: int,
    sample_steps: int,
    device: torch.device,
    integrator: str = "heun",
) -> torch.Tensor:
    u = torch.randn(n, model.dim, device=device)
    steps = max(1, sample_steps)
    dt = 1.0 / steps
    for step in range(steps):
        # Left-endpoint time grid consistent with dt: t in {0, 1/S, ..., 1-1/S}.
        t0 = step * dt
        t_a = torch.full((n, 1), t0, device=device)
        v_a = model(u, t_a)
        if integrator == "heun" and step < steps - 1:
            t_b = torch.full((n, 1), t0 + dt, device=device)
            u_pred = u + dt * v_a
            v_b = model(u_pred, t_b)
            u = u + 0.5 * dt * (v_a + v_b)
        else:
            u = u + dt * v_a
    return normalizer.inverse(u)


def run_continuous_original_space(
    data_dir: Path,
    output_dir: Path,
    cfg: ContinuousOriginalSpaceConfig,
) -> dict[str, object]:
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    spec = make_table_spec(
        cfg.dataset,
        residual_std=cfg.residual_std,
        violation_threshold=cfg.violation_threshold,
    )

    train_np = load_table_csv(data_dir / "train.csv")
    reference_np = load_table_csv(data_dir / "reference.csv")
    n = min(cfg.sample_rows, reference_np.shape[0])
    train_x = torch.tensor(train_np, device=device, dtype=torch.float32)
    reference_x = torch.tensor(reference_np[:n], device=device, dtype=torch.float32)

    u_train = encode_free_coordinates(cfg.dataset, spec, train_x)
    model, normalizer, history = train_free_flow(u_train, cfg)
    u_sample = sample_free_flow(
        model, normalizer, n, cfg.sample_steps, device, integrator=cfg.integrator
    )
    samples = decode_free_coordinates(cfg.dataset, spec, u_sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_np = samples.detach().cpu().numpy()
    reference_eval_np = reference_x.detach().cpu().numpy()
    save_table_csv(
        output_dir / "continuous_original_space_constrained_flux.csv",
        samples_np,
        spec.columns,
    )
    row = evaluate_samples(
        "continuous_original_space_constrained_flux",
        samples_np,
        reference_eval_np,
        spec.constraints,
        seed=cfg.seed,
    )
    write_metrics_csv(output_dir / "metrics.csv", [row])

    result: dict[str, object] = {
        "config": asdict(cfg),
        "model": "continuous_original_space_constrained_flux",
        "source_data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "free_dim": int(u_train.shape[1]),
        "table_dim": int(train_x.shape[1]),
        "implementation_name": cfg.method_name,
        "paper_positioning": (
            "Original-space structural-coordinate constrained flow: a "
            "rectified-flow objective over known free coordinates, followed by "
            "hard reconstruction of dependent columns through structural "
            "equations. Motivated by v-Flux/Flux Matching, but not the full "
            "KDE/projected-Fisher Algorithm 1 Flux loss."
        ),
        "normalizer": normalizer.to_dict(),
        "history": history,
        "metrics": [row],
        "notes": {
            "method": (
                "Learns a rectified-flow model over original structural free coordinates "
                "and reconstructs dependent columns through the known structural equations. "
                "This hard original-space parameterization removes off-manifold residuals."
            )
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
