from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from constrained_tabular_fm.metrics import sliced_wasserstein


@dataclass
class MixedMethodConfig:
    seed: int = 13
    n_train_rows: int = 3000
    n_eval_rows: int = 3000
    batch_size: int = 256
    hidden_dim: int = 128
    latent_dim: int = 10
    ae_epochs: int = 120
    flow_steps: int = 1200
    sample_steps: int = 80
    lr: float = 1e-3
    residual_std: float = 0.01
    violation_threshold: float = 0.05
    latent_constraint_steps: int = 80
    latent_constraint_lr: float = 0.04
    latent_constraint_weight: float = 25.0
    latent_prior_weight: float = 0.05
    direct_repair_rate: float = 6.0
    direct_struct_weight: float = 0.05
    output_dir: str = "outputs/mixed_loan_methods"
    device: str = "auto"


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ContinuousNormalizer:
    def __init__(self, x: torch.Tensor) -> None:
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True).clamp_min(1e-5)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.std.to(x_norm.device) + self.mean.to(x_norm.device)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.detach().cpu().reshape(-1).tolist(),
            "std": self.std.detach().cpu().reshape(-1).tolist(),
        }


class LoanMixedSpec:
    """Mixed-type loan table with category-dependent hard constraints.

    Continuous columns are scaled business quantities:
    income and loan_amount are in 10k-dollar units; monthly_payment is also in
    10k-dollar units per month. Categorical term values depend on loan_type.
    """

    continuous_columns = (
        "income",
        "loan_amount",
        "interest_rate",
        "monthly_payment",
        "credit_limit",
    )
    categorical_columns = (
        "loan_type",
        "employment_status",
        "approved",
        "term_bucket",
    )
    category_names = (
        ("consumer", "mortgage", "business"),
        ("salaried", "self_employed", "unemployed"),
        ("denied", "approved"),
        ("short", "medium", "long"),
    )
    category_sizes = (3, 3, 2, 3)

    def __init__(self, residual_std: float = 0.01, violation_threshold: float = 0.05) -> None:
        self.residual_std = float(residual_std)
        self.violation_threshold = float(violation_threshold)

    @property
    def continuous_dim(self) -> int:
        return len(self.continuous_columns)

    @property
    def categorical_dim(self) -> int:
        return len(self.categorical_columns)

    @property
    def onehot_dim(self) -> int:
        return int(sum(self.category_sizes))

    def term_years(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(
            [
                [3.0, 5.0, 7.0],
                [15.0, 20.0, 30.0],
                [5.0, 10.0, 15.0],
            ],
            device=device,
            dtype=dtype,
        )

    def payment_type_factor(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor([1.04, 1.00, 1.07], device=device, dtype=dtype)

    def credit_type_base(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor([0.22, 0.06, 0.34], device=device, dtype=dtype)

    def credit_amount_factor(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor([0.01, 0.00, 0.04], device=device, dtype=dtype)

    def credit_employment_adjustment(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.tensor([0.03, -0.01, -0.12], device=device, dtype=dtype)

    def one_hot(self, k: torch.Tensor) -> torch.Tensor:
        parts = [
            F.one_hot(k[:, idx], num_classes=size).to(dtype=torch.float32)
            for idx, size in enumerate(self.category_sizes)
        ]
        return torch.cat(parts, dim=1)

    def split_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = []
        start = 0
        for size in self.category_sizes:
            out.append(logits[:, start : start + size])
            start += size
        return tuple(out)

    def probs_from_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(F.softmax(part, dim=1) for part in self.split_logits(logits))

    def sample(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        loan_type = torch.multinomial(
            torch.tensor([0.45, 0.33, 0.22], device=device),
            n,
            replacement=True,
        )
        employment = torch.multinomial(
            torch.tensor([0.62, 0.25, 0.13], device=device),
            n,
            replacement=True,
        )
        term_bucket = torch.multinomial(
            torch.tensor([0.30, 0.42, 0.28], device=device),
            n,
            replacement=True,
        )

        income_loc = torch.tensor([8.5, 7.0, 2.8], device=device)[employment]
        income_scale = torch.tensor([2.0, 2.8, 1.0], device=device)[employment]
        income = (income_loc + income_scale * torch.randn(n, device=device)).clamp(0.8, 24.0)

        amount_loc = torch.tensor([2.6, 24.0, 10.5], device=device)[loan_type]
        amount_scale = torch.tensor([1.2, 9.0, 5.0], device=device)[loan_type]
        income_load = torch.tensor([0.14, 1.10, 0.55], device=device)[loan_type] * income
        loan_amount = (amount_loc + amount_scale * torch.randn(n, device=device) + income_load).clamp(
            0.3,
            90.0,
        )

        base_rate = torch.tensor([0.125, 0.055, 0.088], device=device)[loan_type]
        employment_risk = torch.tensor([0.000, 0.014, 0.046], device=device)[employment]
        interest_rate = (base_rate + employment_risk + 0.012 * torch.randn(n, device=device)).clamp(
            0.012,
            0.240,
        )

        k_without_approval = torch.stack(
            [
                loan_type,
                employment,
                torch.zeros_like(loan_type),
                term_bucket,
            ],
            dim=1,
        )
        raw_payment = self.hard_monthly_payment(loan_amount, interest_rate, loan_type, term_bucket)
        annual_payment = 12.0 * raw_payment
        dti = annual_payment / income.clamp_min(0.2)
        score = (
            2.15
            - 4.5 * dti
            + torch.tensor([0.40, 0.08, -0.80], device=device)[employment]
            + torch.tensor([0.05, 0.28, -0.08], device=device)[loan_type]
            + 0.25 * torch.randn(n, device=device)
        )
        approved = torch.bernoulli(torch.sigmoid(score)).to(torch.long)
        k = k_without_approval.clone()
        k[:, 2] = approved

        target_payment, target_credit = self.targets(
            torch.stack(
                [
                    income,
                    loan_amount,
                    interest_rate,
                    raw_payment,
                    torch.zeros_like(raw_payment),
                ],
                dim=1,
            ),
            k,
        )
        payment_noise = self.residual_std * torch.randn(n, device=device)
        credit_noise = self.residual_std * torch.randn(n, device=device)
        monthly_payment = target_payment + payment_noise
        credit_limit = target_credit + credit_noise
        x = torch.stack(
            [
                income,
                loan_amount,
                interest_rate,
                monthly_payment,
                credit_limit,
            ],
            dim=1,
        )
        return self.clamp_continuous(x), k

    def hard_monthly_payment(
        self,
        amount: torch.Tensor,
        rate: torch.Tensor,
        loan_type: torch.Tensor,
        term_bucket: torch.Tensor,
    ) -> torch.Tensor:
        years = self.term_years(amount.device, amount.dtype)[loan_type, term_bucket]
        type_factor = self.payment_type_factor(amount.device, amount.dtype)[loan_type]
        return self._payment_formula(amount, rate, years, type_factor)

    def _payment_formula(
        self,
        amount: torch.Tensor,
        rate: torch.Tensor,
        years: torch.Tensor,
        type_factor: torch.Tensor,
    ) -> torch.Tensor:
        monthly_rate = rate.clamp(0.001, 0.300) / 12.0
        periods = (years * 12.0).clamp_min(1.0)
        growth = torch.pow(1.0 + monthly_rate, periods)
        amortized = amount * monthly_rate * growth / (growth - 1.0).clamp_min(1e-5)
        return type_factor * amortized

    def hard_credit_limit(
        self,
        income: torch.Tensor,
        amount: torch.Tensor,
        loan_type: torch.Tensor,
        employment: torch.Tensor,
    ) -> torch.Tensor:
        base = self.credit_type_base(income.device, income.dtype)[loan_type]
        amount_factor = self.credit_amount_factor(income.device, income.dtype)[loan_type]
        emp_adj = self.credit_employment_adjustment(income.device, income.dtype)[employment]
        return (income * (base + emp_adj) + amount_factor * amount).clamp(0.0, 12.0)

    def targets(self, x: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        approved = k[:, 2].to(dtype=x.dtype)
        payment = self.hard_monthly_payment(x[:, 1], x[:, 2], k[:, 0], k[:, 3])
        credit = self.hard_credit_limit(x[:, 0], x[:, 1], k[:, 0], k[:, 1])
        return approved * payment, approved * credit

    def soft_targets(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_type, p_emp, p_approved, p_term = self.probs_from_logits(logits)
        approved_prob = p_approved[:, 1]

        years = self.term_years(x.device, x.dtype)
        type_factor = self.payment_type_factor(x.device, x.dtype)
        payment_grid = self._payment_formula(
            x[:, 1, None, None],
            x[:, 2, None, None],
            years[None, :, :],
            type_factor[None, :, None],
        )
        payment_weight = p_type[:, :, None] * p_term[:, None, :]
        expected_payment = approved_prob * (payment_grid * payment_weight).sum(dim=(1, 2))

        base = self.credit_type_base(x.device, x.dtype)
        amount_factor = self.credit_amount_factor(x.device, x.dtype)
        emp_adj = self.credit_employment_adjustment(x.device, x.dtype)
        credit_grid = (
            x[:, 0, None, None] * (base[None, :, None] + emp_adj[None, None, :])
            + x[:, 1, None, None] * amount_factor[None, :, None]
        ).clamp(0.0, 12.0)
        credit_weight = p_type[:, :, None] * p_emp[:, None, :]
        expected_credit = approved_prob * (credit_grid * credit_weight).sum(dim=(1, 2))
        return expected_payment, expected_credit

    def residuals(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        payment_target, credit_target = self.targets(x, k)
        return torch.stack(
            [
                x[:, 3] - payment_target,
                x[:, 4] - credit_target,
            ],
            dim=1,
        )

    def soft_residuals(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        payment_target, credit_target = self.soft_targets(x, logits)
        return torch.stack(
            [
                x[:, 3] - payment_target,
                x[:, 4] - credit_target,
            ],
            dim=1,
        )

    def repair(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x_fixed = self.clamp_continuous(x)
        payment_target, credit_target = self.targets(x_fixed, k)
        return torch.stack(
            [
                x_fixed[:, 0],
                x_fixed[:, 1],
                x_fixed[:, 2],
                payment_target,
                credit_target,
            ],
            dim=1,
        )

    def repair_velocity(self, x: torch.Tensor, k: torch.Tensor, rate: float) -> torch.Tensor:
        target = self.repair(x, k)
        velocity = torch.zeros_like(x)
        velocity[:, 3] = rate * (target[:, 3] - x[:, 3])
        velocity[:, 4] = rate * (target[:, 4] - x[:, 4])
        return velocity

    def clamp_continuous(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                x[:, 0].clamp(0.2, 26.0),
                x[:, 1].clamp(0.2, 95.0),
                x[:, 2].clamp(0.005, 0.280),
                x[:, 3].clamp(-0.5, 12.0),
                x[:, 4].clamp(-0.5, 14.0),
            ],
            dim=1,
        )

    def to_numpy_table(self, x: torch.Tensor, k: torch.Tensor) -> np.ndarray:
        return torch.cat([k.to(dtype=x.dtype), x], dim=1).detach().cpu().numpy()

    def table_columns(self) -> tuple[str, ...]:
        return self.categorical_columns + self.continuous_columns


class MixedVAE(nn.Module):
    def __init__(
        self,
        cont_dim: int,
        onehot_dim: int,
        category_sizes: tuple[int, ...],
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.category_sizes = category_sizes
        input_dim = cont_dim + onehot_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.dec_cont = nn.Linear(hidden_dim, cont_dim)
        self.dec_cat = nn.Linear(hidden_dim, onehot_dim)

    def encode(self, x_norm: torch.Tensor, k_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x_norm, k_onehot], dim=1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 6.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(z)
        return self.dec_cont(h), self.dec_cat(h)

    def forward(
        self,
        x_norm: torch.Tensor,
        k_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_norm, k_onehot)
        z = self.reparameterize(mu, logvar)
        x_hat, cat_logits = self.decode(z)
        return x_hat, cat_logits, mu, logvar


class ConditionalVelocity(nn.Module):
    def __init__(
        self,
        cont_dim: int,
        onehot_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cont_dim + onehot_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cont_dim),
        )

    def forward(self, x_norm: torch.Tensor, k_onehot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_norm, k_onehot, t], dim=1))


def _sample_batch(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, n, (batch_size,), device=device)


def _categorical_loss(spec: LoanMixedSpec, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    parts = spec.split_logits(logits)
    losses = [F.cross_entropy(part, k[:, idx]) for idx, part in enumerate(parts)]
    return torch.stack(losses).mean()


def train_mixed_vae(
    spec: LoanMixedSpec,
    x_train: torch.Tensor,
    k_train: torch.Tensor,
    normalizer: ContinuousNormalizer,
    cfg: MixedMethodConfig,
) -> tuple[MixedVAE, dict[str, float]]:
    model = MixedVAE(
        spec.continuous_dim,
        spec.onehot_dim,
        spec.category_sizes,
        cfg.hidden_dim,
        cfg.latent_dim,
    ).to(x_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    x_norm = normalizer.transform(x_train)
    k_onehot = spec.one_hot(k_train)
    steps_per_epoch = max(1, math.ceil(x_train.shape[0] / cfg.batch_size))
    last = {}
    for _epoch in range(cfg.ae_epochs):
        for _ in range(steps_per_epoch):
            idx = _sample_batch(x_train.shape[0], cfg.batch_size, x_train.device)
            x_hat, logits, mu, logvar = model(x_norm[idx], k_onehot[idx])
            recon_cont = F.mse_loss(x_hat, x_norm[idx])
            recon_cat = _categorical_loss(spec, logits, k_train[idx])
            kld = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
            loss = recon_cont + recon_cat + 1e-3 * kld
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = {
                "vae_loss": float(loss.detach().cpu()),
                "vae_cont_loss": float(recon_cont.detach().cpu()),
                "vae_cat_loss": float(recon_cat.detach().cpu()),
                "vae_kld": float(kld.detach().cpu()),
            }
    return model, last


@torch.no_grad()
def encode_latents(
    spec: LoanMixedSpec,
    model: MixedVAE,
    x: torch.Tensor,
    k: torch.Tensor,
    normalizer: ContinuousNormalizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_norm = normalizer.transform(x)
    k_onehot = spec.one_hot(k)
    mu, logvar = model.encode(x_norm, k_onehot)
    return mu, logvar


def _latent_stats(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = latents.mean(dim=0, keepdim=True)
    std = latents.std(dim=0, keepdim=True).clamp_min(0.05)
    return mean, std


@torch.no_grad()
def sample_tabsyn_style(
    spec: LoanMixedSpec,
    model: MixedVAE,
    normalizer: ContinuousNormalizer,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = latent_mean + latent_std * torch.randn(n, latent_mean.shape[1], device=latent_mean.device)
    x_norm, logits = model.decode(z)
    k = torch.stack([part.argmax(dim=1) for part in spec.split_logits(logits)], dim=1)
    x = spec.clamp_continuous(normalizer.inverse(x_norm))
    return x, k


def sample_latent_constrained(
    spec: LoanMixedSpec,
    model: MixedVAE,
    normalizer: ContinuousNormalizer,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    n: int,
    cfg: MixedMethodConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = (
        latent_mean
        + latent_std * torch.randn(n, latent_mean.shape[1], device=latent_mean.device)
    ).detach()
    z.requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=cfg.latent_constraint_lr)
    scale = max(spec.violation_threshold, 1e-3)
    for _ in range(cfg.latent_constraint_steps):
        x_norm, logits = model.decode(z)
        x = spec.clamp_continuous(normalizer.inverse(x_norm))
        residual = spec.soft_residuals(x, logits)
        constraint_loss = (residual / scale).pow(2).mean()
        prior_loss = ((z - latent_mean) / latent_std).pow(2).mean()
        loss = cfg.latent_constraint_weight * constraint_loss + cfg.latent_prior_weight * prior_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        x_norm, logits = model.decode(z)
        k = torch.stack([part.argmax(dim=1) for part in spec.split_logits(logits)], dim=1)
        x = spec.clamp_continuous(normalizer.inverse(x_norm))
    return x, k


def train_original_space_flow(
    spec: LoanMixedSpec,
    x_train: torch.Tensor,
    k_train: torch.Tensor,
    normalizer: ContinuousNormalizer,
    cfg: MixedMethodConfig,
) -> tuple[ConditionalVelocity, dict[str, float]]:
    model = ConditionalVelocity(spec.continuous_dim, spec.onehot_dim, cfg.hidden_dim).to(x_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    x1_all = normalizer.transform(x_train)
    last = {}
    for _step in range(cfg.flow_steps):
        idx = _sample_batch(x_train.shape[0], cfg.batch_size, x_train.device)
        x1 = x1_all[idx]
        k = k_train[idx]
        k_onehot = spec.one_hot(k)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, device=x_train.device)
        x_t = (1.0 - t) * x0 + t * x1
        target_velocity = x1 - x0
        raw_velocity = model(x_t, k_onehot, t)
        fm_loss = F.mse_loss(raw_velocity, target_velocity)

        x_orig = normalizer.inverse(x_t)
        repair_norm = spec.repair_velocity(x_orig, k, cfg.direct_repair_rate) / normalizer.std.to(
            x_train.device
        )
        x_after = normalizer.inverse(x_t + 0.05 * (raw_velocity + repair_norm))
        residual_after = spec.residuals(x_after, k)
        struct_loss = (residual_after / spec.violation_threshold).pow(2).mean()
        loss = fm_loss + cfg.direct_struct_weight * struct_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = {
            "flow_loss": float(loss.detach().cpu()),
            "flow_fm_loss": float(fm_loss.detach().cpu()),
            "flow_struct_loss": float(struct_loss.detach().cpu()),
        }
    return model, last


@torch.no_grad()
def sample_original_space_flow(
    spec: LoanMixedSpec,
    model: ConditionalVelocity,
    normalizer: ContinuousNormalizer,
    k_train: torch.Tensor,
    n: int,
    cfg: MixedMethodConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = k_train.device
    idx = torch.randint(0, k_train.shape[0], (n,), device=device)
    k = k_train[idx].clone()
    k_onehot = spec.one_hot(k)
    x_norm = torch.randn(n, spec.continuous_dim, device=device)
    dt = 1.0 / max(1, cfg.sample_steps)
    for step in range(cfg.sample_steps):
        t = torch.full((n, 1), step / max(1, cfg.sample_steps - 1), device=device)
        raw_velocity = model(x_norm, k_onehot, t)
        x_orig = normalizer.inverse(x_norm)
        repair_norm = spec.repair_velocity(x_orig, k, cfg.direct_repair_rate) / normalizer.std.to(device)
        x_norm = x_norm + dt * (raw_velocity + repair_norm)
        x_norm = x_norm.clamp(-6.0, 6.0)
    x = spec.repair(spec.clamp_continuous(normalizer.inverse(x_norm)), k)
    return x, k


def _category_tv(k: np.ndarray, ref: np.ndarray, sizes: tuple[int, ...]) -> tuple[float, dict[str, float]]:
    per_col = {}
    values = []
    for idx, size in enumerate(sizes):
        p = np.bincount(k[:, idx], minlength=size).astype(np.float64)
        q = np.bincount(ref[:, idx], minlength=size).astype(np.float64)
        p /= max(1.0, p.sum())
        q /= max(1.0, q.sum())
        tv = 0.5 * np.abs(p - q).sum()
        per_col[f"cat_{idx}_tv"] = float(tv)
        values.append(tv)
    return float(np.mean(values)), per_col


def _joint_category_tv(k: np.ndarray, ref: np.ndarray) -> float:
    def counts(values: np.ndarray) -> dict[tuple[int, ...], int]:
        out: dict[tuple[int, ...], int] = {}
        for row in values:
            key = tuple(int(v) for v in row)
            out[key] = out.get(key, 0) + 1
        return out

    left = counts(k)
    right = counts(ref)
    keys = set(left) | set(right)
    n_left = max(1, len(k))
    n_right = max(1, len(ref))
    return float(0.5 * sum(abs(left.get(key, 0) / n_left - right.get(key, 0) / n_right) for key in keys))


def evaluate_mixed_samples(
    spec: LoanMixedSpec,
    name: str,
    x: torch.Tensor,
    k: torch.Tensor,
    ref_x: torch.Tensor,
    ref_k: torch.Tensor,
    pre_repair_x: torch.Tensor | None = None,
) -> dict[str, float | str | int]:
    x_np = x.detach().cpu().numpy().astype(np.float64)
    ref_x_np = ref_x.detach().cpu().numpy().astype(np.float64)
    k_np = k.detach().cpu().numpy().astype(np.int64)
    ref_k_np = ref_k.detach().cpu().numpy().astype(np.int64)
    residual = spec.residuals(x, k).detach().cpu().numpy()
    violation = np.any(np.abs(residual) > spec.violation_threshold, axis=1)
    cat_tv, per_col = _category_tv(k_np, ref_k_np, spec.category_sizes)
    out: dict[str, float | str | int] = {
        "model": name,
        "n_samples": int(x_np.shape[0]),
        "continuous_sliced_wasserstein": sliced_wasserstein(x_np, ref_x_np, seed=17),
        "continuous_mean_l2": float(np.linalg.norm(x_np.mean(axis=0) - ref_x_np.mean(axis=0))),
        "continuous_cov_frobenius": float(np.linalg.norm(np.cov(x_np.T) - np.cov(ref_x_np.T), ord="fro")),
        "categorical_tv_mean": cat_tv,
        "categorical_joint_tv": _joint_category_tv(k_np, ref_k_np),
        "constraint_mae": float(np.mean(np.abs(residual))),
        "constraint_payment_mae": float(np.mean(np.abs(residual[:, 0]))),
        "constraint_credit_mae": float(np.mean(np.abs(residual[:, 1]))),
        "violation_rate": float(np.mean(violation)),
    }
    out.update(per_col)
    if pre_repair_x is not None:
        shift = (x - pre_repair_x).detach().cpu().numpy()
        out["repair_l2_mean"] = float(np.linalg.norm(shift, axis=1).mean())
        out["repair_linf_mean"] = float(np.abs(shift).max(axis=1).mean())
    return out


def save_mixed_csv(path: str | Path, spec: LoanMixedSpec, x: torch.Tensor, k: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(spec.table_columns())
        rows = spec.to_numpy_table(x, k)
        for row in rows:
            writer.writerow([int(v) for v in row[: spec.categorical_dim]] + [float(v) for v in row[spec.categorical_dim :]])


def write_metric_csv(path: str | Path, rows: list[dict[str, float | str | int]]) -> None:
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


def run_mixed_method_experiment(cfg: MixedMethodConfig) -> dict[str, object]:
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = LoanMixedSpec(
        residual_std=cfg.residual_std,
        violation_threshold=cfg.violation_threshold,
    )

    train_x, train_k = spec.sample(cfg.n_train_rows, device)
    ref_x, ref_k = spec.sample(cfg.n_eval_rows, device)
    normalizer = ContinuousNormalizer(train_x)

    vae, vae_history = train_mixed_vae(spec, train_x, train_k, normalizer, cfg)
    latents, _ = encode_latents(spec, vae, train_x, train_k, normalizer)
    latent_mean, latent_std = _latent_stats(latents)

    tabsyn_raw_x, tabsyn_raw_k = sample_tabsyn_style(
        spec,
        vae,
        normalizer,
        latent_mean,
        latent_std,
        cfg.n_eval_rows,
    )
    tabsyn_repair_x = spec.repair(tabsyn_raw_x, tabsyn_raw_k)

    latent_x, latent_k = sample_latent_constrained(
        spec,
        vae,
        normalizer,
        latent_mean,
        latent_std,
        cfg.n_eval_rows,
        cfg,
    )

    flow, flow_history = train_original_space_flow(spec, train_x, train_k, normalizer, cfg)
    direct_x, direct_k = sample_original_space_flow(
        spec,
        flow,
        normalizer,
        train_k,
        cfg.n_eval_rows,
        cfg,
    )

    save_mixed_csv(out_dir / "train.csv", spec, train_x, train_k)
    save_mixed_csv(out_dir / "reference.csv", spec, ref_x, ref_k)
    save_mixed_csv(out_dir / "tabsyn_raw.csv", spec, tabsyn_raw_x, tabsyn_raw_k)
    save_mixed_csv(out_dir / "tabsyn_repair.csv", spec, tabsyn_repair_x, tabsyn_raw_k)
    save_mixed_csv(out_dir / "latent_constrained_flux.csv", spec, latent_x, latent_k)
    save_mixed_csv(out_dir / "original_space_constrained_flux.csv", spec, direct_x, direct_k)

    rows = [
        evaluate_mixed_samples(spec, "tabsyn_raw", tabsyn_raw_x, tabsyn_raw_k, ref_x, ref_k),
        evaluate_mixed_samples(
            spec,
            "tabsyn_repair",
            tabsyn_repair_x,
            tabsyn_raw_k,
            ref_x,
            ref_k,
            pre_repair_x=tabsyn_raw_x,
        ),
        evaluate_mixed_samples(spec, "latent_constrained_flux", latent_x, latent_k, ref_x, ref_k),
        evaluate_mixed_samples(spec, "original_space_constrained_flux", direct_x, direct_k, ref_x, ref_k),
    ]

    result: dict[str, object] = {
        "config": asdict(cfg),
        "columns": {
            "categorical": spec.categorical_columns,
            "continuous": spec.continuous_columns,
        },
        "category_names": spec.category_names,
        "normalizer": normalizer.to_dict(),
        "histories": {
            "tabsyn_style_vae": vae_history,
            "original_space_flow": flow_history,
        },
        "metrics": rows,
        "notes": {
            "tabsyn_repair": (
                "Lightweight TabSyn-style VAE proxy followed by hard original-space repair; "
                "intended to test post-hoc distribution shift before running full TabSyn."
            ),
            "latent_constrained_flux": (
                "Latent samples are optimized through the decoder with a soft decoded constraint; "
                "this exposes decoder and soft-categorical failure modes."
            ),
            "original_space_constrained_flux": (
                "Stage-A original-space conditional constrained flow/Flux proxy: hard categories "
                "are fixed and continuous variables are repaired during generation."
            ),
        },
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_metric_csv(out_dir / "metrics.csv", rows)
    return result
