from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from constrained_tabular_fm.continuous_original_space import make_free_flow_normalizer
from constrained_tabular_fm.metrics import sliced_wasserstein
from constrained_tabular_fm.mixed_methods import ContinuousNormalizer, MixedVAE, set_seed


ROOT = Path(__file__).resolve().parents[1]


CONTINUOUS_COLUMNS = (
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "revol_util",
    "total_acc",
    "installment",
    "loan_income_ratio",
    "fico_range_high",
)

FREE_CONTINUOUS_COLUMNS = (
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "revol_util",
    "total_acc",
)

DEPENDENT_CONTINUOUS_COLUMNS = (
    "installment",
    "loan_income_ratio",
    "fico_range_high",
)

CATEGORICAL_COLUMNS = (
    "term_months",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "loan_status",
)

PREFERRED_CATEGORY_ORDER = {
    "term_months": ("36", "60"),
    "grade": ("A", "B", "C", "D", "E", "F", "G"),
    "home_ownership": ("MORTGAGE", "RENT", "OWN", "OTHER"),
    "verification_status": ("Not Verified", "Source Verified", "Verified"),
    "application_type": ("Individual", "Joint App"),
    "loan_status": ("fully_paid", "charged_off", "current", "late_or_grace"),
}


@dataclass
class LendingClubConfig:
    train_csv: str = "data/lendingclub_preprocessed/train.csv"
    reference_csv: str = "data/lendingclub_preprocessed/reference.csv"
    output_dir: str = "outputs/lendingclub_constraint_methods"
    seed: int = 13
    train_rows: int = 50_000
    eval_rows: int = 10_000
    batch_size: int = 512
    hidden_dim: int = 160
    latent_dim: int = 16
    ae_epochs: int = 18
    flow_steps: int = 1200
    sample_steps: int = 80
    lr: float = 1e-3
    free_whiten_shrinkage: float = 0.5
    installment_tolerance: float = 0.05
    ratio_tolerance: float = 0.005
    fico_tolerance: float = 0.5
    device: str = "auto"


@dataclass
class LendingClubAllMethodsConfig(LendingClubConfig):
    ctgan_epochs: int = 20
    tvae_epochs: int = 30
    tabddpm_steps: int = 700
    tabddpm_timesteps: int = 80
    tabsyn_vae_epochs: int = 80
    tabsyn_diffusion_epochs: int = 160
    tabsyn_sample_steps: int = 50
    skip_ctgan: bool = False
    skip_tvae: bool = False
    skip_tabddpm: bool = False
    skip_tabsyn: bool = False
    skip_internal: bool = False


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def external_source_metadata() -> dict[str, object]:
    return {
        "repositories": {
            "tabsyn": {
                "url": "https://github.com/amazon-science/tabsyn",
                "local_path": str(ROOT / "external" / "tabsyn"),
                "commit": git_commit(ROOT / "external" / "tabsyn"),
                "paper_name": "TabSyn adapter",
            },
            "tabddpm": {
                "url": "https://github.com/yandex-research/tab-ddpm",
                "local_path": str(ROOT / "external" / "tab-ddpm"),
                "commit": git_commit(ROOT / "external" / "tab-ddpm"),
                "paper_name": "Official TabDDPM path",
            },
            "ctgan_tvae": {
                "url": "https://github.com/sdv-dev/CTGAN",
                "local_path": str(ROOT / "external" / "CTGAN"),
                "commit": git_commit(ROOT / "external" / "CTGAN"),
            },
        },
        "compatibility_patches": {
            "tabddpm_quantile_normalizer": (
                "Upstream TabDDPM passes subsample=1e9 to sklearn "
                "QuantileTransformer; local runners cast it to int for newer "
                "sklearn while preserving the upstream quantile normalization."
            )
        },
    }


class CategoryCodec:
    def __init__(self, train: pd.DataFrame, reference: pd.DataFrame) -> None:
        names: list[tuple[str, ...]] = []
        for column in CATEGORICAL_COLUMNS:
            values = pd.concat([train[column], reference[column]], ignore_index=True).dropna().astype(str)
            observed = set(values.unique().tolist())
            preferred = [value for value in PREFERRED_CATEGORY_ORDER.get(column, ()) if value in observed]
            tail = sorted(observed - set(preferred))
            names.append(tuple(preferred + tail))
        self.names = tuple(names)
        self.sizes = tuple(len(part) for part in self.names)
        self.maps = [{value: idx for idx, value in enumerate(part)} for part in self.names]

    @property
    def onehot_dim(self) -> int:
        return int(sum(self.sizes))

    def encode_frame(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = []
        for column, mapping in zip(CATEGORICAL_COLUMNS, self.maps):
            encoded.append(frame[column].astype(str).map(mapping).to_numpy(np.int64))
        out = np.stack(encoded, axis=1)
        if np.any(out < 0):
            raise ValueError("Encountered an unknown categorical value while encoding LendingClub data.")
        return out

    def decode_array(self, k: np.ndarray) -> dict[str, list[str]]:
        decoded: dict[str, list[str]] = {}
        for idx, column in enumerate(CATEGORICAL_COLUMNS):
            names = self.names[idx]
            decoded[column] = [names[int(value)] for value in k[:, idx]]
        return decoded


class LendingClubSpec:
    continuous_columns = CONTINUOUS_COLUMNS
    categorical_columns = CATEGORICAL_COLUMNS
    free_continuous_columns = FREE_CONTINUOUS_COLUMNS
    dependent_continuous_columns = DEPENDENT_CONTINUOUS_COLUMNS

    def __init__(
        self,
        category_names: tuple[tuple[str, ...], ...],
        lower: torch.Tensor,
        upper: torch.Tensor,
        thresholds: tuple[float, float, float],
    ) -> None:
        self.category_names = category_names
        self.category_sizes = tuple(len(part) for part in category_names)
        self.lower = lower
        self.upper = upper
        self.thresholds = thresholds

    @property
    def continuous_dim(self) -> int:
        return len(self.continuous_columns)

    @property
    def free_dim(self) -> int:
        return len(self.free_continuous_columns)

    @property
    def categorical_dim(self) -> int:
        return len(self.categorical_columns)

    @property
    def onehot_dim(self) -> int:
        return int(sum(self.category_sizes))

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

    def term_months(self, k: torch.Tensor) -> torch.Tensor:
        values = [float(value) for value in self.category_names[0]]
        lookup = torch.tensor(values, device=k.device, dtype=torch.float32)
        return lookup[k[:, 0]]

    def amortized_installment(
        self,
        loan_amnt: torch.Tensor,
        int_rate: torch.Tensor,
        term_months: torch.Tensor,
    ) -> torch.Tensor:
        monthly = (int_rate.clamp(0.0, 40.0) / 100.0) / 12.0
        periods = term_months.clamp_min(1.0)
        denom = 1.0 - torch.pow(1.0 + monthly, -periods)
        payment = loan_amnt * monthly / denom.clamp_min(1e-6)
        zero_payment = loan_amnt / periods
        return torch.where(monthly.abs() < 1e-8, zero_payment, payment)

    def targets(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        loan_amnt = x[:, 0].clamp_min(1.0)
        int_rate = x[:, 1].clamp(0.0, 40.0)
        annual_inc = x[:, 2].clamp_min(1.0)
        fico_low = x[:, 4]
        installment = self.amortized_installment(loan_amnt, int_rate, self.term_months(k))
        ratio = loan_amnt / annual_inc
        fico_high = fico_low + 4.0
        return torch.stack([installment, ratio, fico_high], dim=1)

    def residuals(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        target = self.targets(x, k)
        return torch.stack(
            [
                x[:, 8] - target[:, 0],
                x[:, 9] - target[:, 1],
                x[:, 10] - target[:, 2],
            ],
            dim=1,
        )

    def clamp_continuous(self, x: torch.Tensor) -> torch.Tensor:
        lower = self.lower.to(x.device, x.dtype)
        upper = self.upper.to(x.device, x.dtype)
        return torch.minimum(torch.maximum(x, lower), upper)

    def repair(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x_fixed = self.clamp_continuous(x)
        target = self.targets(x_fixed, k)
        return torch.cat([x_fixed[:, :8], target], dim=1)

    def decode_free(self, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        full = torch.cat([u, torch.zeros(u.shape[0], 3, device=u.device, dtype=u.dtype)], dim=1)
        return self.repair(full, k)


class ConditionalFreeVelocity(nn.Module):
    def __init__(self, free_dim: int, onehot_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(free_dim + onehot_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, free_dim),
        )

    def forward(self, x_norm: torch.Tensor, k_onehot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_norm, k_onehot, t], dim=1))


def load_lendingclub_data(cfg: LendingClubConfig) -> tuple[pd.DataFrame, pd.DataFrame, CategoryCodec]:
    train = pd.read_csv(cfg.train_csv)
    reference = pd.read_csv(cfg.reference_csv)
    train = train.sample(frac=1.0, random_state=cfg.seed).head(cfg.train_rows).reset_index(drop=True)
    reference = reference.sample(frac=1.0, random_state=cfg.seed + 1).head(cfg.eval_rows).reset_index(drop=True)
    codec = CategoryCodec(train, reference)
    return train, reference, codec


def tensors_from_frames(
    train: pd.DataFrame,
    reference: pd.DataFrame,
    codec: CategoryCodec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = torch.tensor(train[list(CONTINUOUS_COLUMNS)].to_numpy(np.float32), device=device)
    ref_x = torch.tensor(reference[list(CONTINUOUS_COLUMNS)].to_numpy(np.float32), device=device)
    train_k = torch.tensor(codec.encode_frame(train), device=device)
    ref_k = torch.tensor(codec.encode_frame(reference), device=device)
    return train_x, train_k, ref_x, ref_k


def generated_frame_to_tensors(
    frame: pd.DataFrame,
    codec: CategoryCodec,
    spec: LendingClubSpec,
    medians: pd.Series,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    continuous = pd.DataFrame(index=frame.index)
    for column in CONTINUOUS_COLUMNS:
        values = pd.to_numeric(frame.get(column), errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(float(medians[column]))
        continuous[column] = values
    x = torch.tensor(continuous[list(CONTINUOUS_COLUMNS)].to_numpy(np.float32), device=device)
    x = spec.clamp_continuous(x)

    encoded_cols = []
    for col_idx, column in enumerate(CATEGORICAL_COLUMNS):
        names = codec.names[col_idx]
        mapping = codec.maps[col_idx]
        raw = frame.get(column)
        if raw is None:
            encoded_cols.append(np.zeros(len(frame), dtype=np.int64))
            continue
        values = raw.astype(str).str.strip()
        if column == "term_months":
            values = values.str.extract(r"(\d+)", expand=False).fillna(values)
        mapped = values.map(mapping)
        mapped = mapped.fillna(0).astype(np.int64).to_numpy()
        mapped = np.clip(mapped, 0, len(names) - 1)
        encoded_cols.append(mapped)
    k = torch.tensor(np.stack(encoded_cols, axis=1), device=device, dtype=torch.long)
    return x, k


def build_spec(
    train_x: torch.Tensor,
    codec: CategoryCodec,
    cfg: LendingClubConfig,
) -> LendingClubSpec:
    lower = train_x.quantile(0.001, dim=0)
    upper = train_x.quantile(0.999, dim=0)
    return LendingClubSpec(
        codec.names,
        lower.detach().cpu(),
        upper.detach().cpu(),
        (cfg.installment_tolerance, cfg.ratio_tolerance, cfg.fico_tolerance),
    )


def sample_batch(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, n, (batch_size,), device=device)


def categorical_loss(spec: LendingClubSpec, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    losses = [F.cross_entropy(part, k[:, idx]) for idx, part in enumerate(spec.split_logits(logits))]
    return torch.stack(losses).mean()


def train_mixed_vae(
    spec: LendingClubSpec,
    x_train: torch.Tensor,
    k_train: torch.Tensor,
    normalizer: ContinuousNormalizer,
    cfg: LendingClubConfig,
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
    last: dict[str, float] = {}
    for _epoch in range(cfg.ae_epochs):
        for _ in range(steps_per_epoch):
            idx = sample_batch(x_train.shape[0], cfg.batch_size, x_train.device)
            x_hat, logits, mu, logvar = model(x_norm[idx], k_onehot[idx])
            recon_cont = F.mse_loss(x_hat, x_norm[idx])
            recon_cat = categorical_loss(spec, logits, k_train[idx])
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
def encode_latent_mean(
    spec: LendingClubSpec,
    model: MixedVAE,
    x: torch.Tensor,
    k: torch.Tensor,
    normalizer: ContinuousNormalizer,
) -> torch.Tensor:
    mu, _ = model.encode(normalizer.transform(x), spec.one_hot(k))
    return mu


@torch.no_grad()
def sample_vae_raw(
    spec: LendingClubSpec,
    model: MixedVAE,
    normalizer: Any,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = latent_mean + latent_std * torch.randn(n, latent_mean.shape[1], device=latent_mean.device)
    x_norm, logits = model.decode(z)
    k = torch.stack([part.argmax(dim=1) for part in spec.split_logits(logits)], dim=1)
    x = spec.clamp_continuous(normalizer.inverse(x_norm))
    return x, k


def train_original_space_free_flow(
    spec: LendingClubSpec,
    x_train: torch.Tensor,
    k_train: torch.Tensor,
    normalizer: Any,
    cfg: LendingClubConfig,
) -> tuple[ConditionalFreeVelocity, dict[str, float]]:
    model = ConditionalFreeVelocity(spec.free_dim, spec.onehot_dim, cfg.hidden_dim).to(x_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    u1_all = normalizer.transform(x_train[:, : spec.free_dim])
    k_onehot = spec.one_hot(k_train)
    last: dict[str, float] = {}
    for _step in range(cfg.flow_steps):
        idx = sample_batch(x_train.shape[0], cfg.batch_size, x_train.device)
        u1 = u1_all[idx]
        u0 = torch.randn_like(u1)
        t = torch.rand(u1.shape[0], 1, device=x_train.device)
        u_t = (1.0 - t) * u0 + t * u1
        target_velocity = u1 - u0
        pred = model(u_t, k_onehot[idx], t)
        # Engineering objective: category-conditioned rectified-flow regression
        # on original-space free coordinates. This is Flux/v-Flux motivated, but
        # it is not the full KDE/projected-Fisher Flux Matching loss.
        loss = F.mse_loss(pred, target_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = {"flow_loss": float(loss.detach().cpu())}
    return model, last


@torch.no_grad()
def sample_original_space_free_flow(
    spec: LendingClubSpec,
    model: ConditionalFreeVelocity,
    normalizer: ContinuousNormalizer,
    k_train: torch.Tensor,
    n: int,
    cfg: LendingClubConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = k_train.device
    idx = torch.randint(0, k_train.shape[0], (n,), device=device)
    k = k_train[idx].clone()
    k_onehot = spec.one_hot(k)
    u_norm = torch.randn(n, spec.free_dim, device=device)
    dt = 1.0 / max(1, cfg.sample_steps)
    for step in range(cfg.sample_steps):
        t = torch.full((n, 1), step / max(1, cfg.sample_steps - 1), device=device)
        u_norm = u_norm + dt * model(u_norm, k_onehot, t)
        u_norm = u_norm.clamp(-6.0, 6.0)
    u = normalizer.inverse(u_norm)
    lower = spec.lower[: spec.free_dim].to(device, u.dtype)
    upper = spec.upper[: spec.free_dim].to(device, u.dtype)
    u = torch.minimum(torch.maximum(u, lower), upper)
    return spec.decode_free(u, k), k


@torch.no_grad()
def sample_train_bootstrap(
    x_train: torch.Tensor,
    k_train: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x_train.shape[0], (n,), device=x_train.device)
    return x_train[idx].clone(), k_train[idx].clone()


@torch.no_grad()
def sample_independent_columns(
    x_train: torch.Tensor,
    k_train: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_cols = []
    for col in range(x_train.shape[1]):
        idx = torch.randint(0, x_train.shape[0], (n,), device=x_train.device)
        x_cols.append(x_train[idx, col])
    k_cols = []
    for col in range(k_train.shape[1]):
        idx = torch.randint(0, k_train.shape[0], (n,), device=k_train.device)
        k_cols.append(k_train[idx, col])
    return torch.stack(x_cols, dim=1), torch.stack(k_cols, dim=1)


def category_tv(k: np.ndarray, ref: np.ndarray, sizes: tuple[int, ...]) -> tuple[float, dict[str, float]]:
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


def joint_category_tv(k: np.ndarray, ref: np.ndarray) -> float:
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


def evaluate_lendingclub_samples(
    spec: LendingClubSpec,
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
    ref_mean = ref_x_np.mean(axis=0, keepdims=True)
    ref_std = ref_x_np.std(axis=0, keepdims=True).clip(min=1e-6)
    residual = spec.residuals(x, k).detach().cpu().numpy().astype(np.float64)
    thresholds = np.asarray(spec.thresholds, dtype=np.float64)
    violation = np.any(np.abs(residual) > thresholds[None, :], axis=1)
    cat_tv, per_col = category_tv(k_np, ref_k_np, spec.category_sizes)
    out: dict[str, float | str | int] = {
        "model": name,
        "n_samples": int(x_np.shape[0]),
        "continuous_sliced_wasserstein": sliced_wasserstein(x_np, ref_x_np, seed=17),
        "continuous_sliced_wasserstein_norm": sliced_wasserstein(
            (x_np - ref_mean) / ref_std,
            (ref_x_np - ref_mean) / ref_std,
            seed=17,
        ),
        "free_sliced_wasserstein_norm": sliced_wasserstein(
            (x_np[:, :8] - ref_mean[:, :8]) / ref_std[:, :8],
            (ref_x_np[:, :8] - ref_mean[:, :8]) / ref_std[:, :8],
            seed=19,
        ),
        "dependent_sliced_wasserstein_norm": sliced_wasserstein(
            (x_np[:, 8:] - ref_mean[:, 8:]) / ref_std[:, 8:],
            (ref_x_np[:, 8:] - ref_mean[:, 8:]) / ref_std[:, 8:],
            seed=23,
        ),
        "continuous_mean_l2_norm": float(
            np.linalg.norm(((x_np - ref_mean) / ref_std).mean(axis=0) - ((ref_x_np - ref_mean) / ref_std).mean(axis=0))
        ),
        "categorical_tv_mean": cat_tv,
        "categorical_joint_tv": joint_category_tv(k_np, ref_k_np),
        "constraint_nmae": float(np.mean(np.abs(residual) / thresholds[None, :])),
        "installment_mae": float(np.mean(np.abs(residual[:, 0]))),
        "loan_income_ratio_mae": float(np.mean(np.abs(residual[:, 1]))),
        "fico_range_high_mae": float(np.mean(np.abs(residual[:, 2]))),
        "installment_violation_rate": float(np.mean(np.abs(residual[:, 0]) > thresholds[0])),
        "loan_income_ratio_violation_rate": float(np.mean(np.abs(residual[:, 1]) > thresholds[1])),
        "fico_range_high_violation_rate": float(np.mean(np.abs(residual[:, 2]) > thresholds[2])),
        "violation_rate": float(np.mean(violation)),
    }
    out.update(per_col)
    if pre_repair_x is not None:
        shift = (x - pre_repair_x).detach().cpu().numpy().astype(np.float64)
        out["repair_l2_mean"] = float(np.linalg.norm(shift, axis=1).mean())
        out["repair_l2_norm_mean"] = float(np.linalg.norm(shift / ref_std, axis=1).mean())
        out["repair_linf_mean"] = float(np.abs(shift).max(axis=1).mean())
    return out


def save_lendingclub_csv(path: str | Path, spec: LendingClubSpec, codec: CategoryCodec, x: torch.Tensor, k: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_np = x.detach().cpu().numpy()
    k_np = k.detach().cpu().numpy()
    decoded = codec.decode_array(k_np)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(CONTINUOUS_COLUMNS) + list(CATEGORICAL_COLUMNS))
        for row_idx in range(x_np.shape[0]):
            row = [float(value) for value in x_np[row_idx]]
            row.extend(decoded[column][row_idx] for column in CATEGORICAL_COLUMNS)
            writer.writerow(row)


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


def write_main_table(path: str | Path, rows: list[dict[str, float | str | int]]) -> None:
    columns = [
        "model",
        "continuous_sliced_wasserstein_norm",
        "free_sliced_wasserstein_norm",
        "dependent_sliced_wasserstein_norm",
        "categorical_tv_mean",
        "categorical_joint_tv",
        "constraint_nmae",
        "violation_rate",
        "repair_l2_norm_mean",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if "rate" in column or "tv" in column:
                    values.append(f"{100.0 * value:.2f}%")
                else:
                    values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_tabddpm_quantile_normalizer() -> None:
    try:
        import lib
        import lib.data as tabddpm_data
    except Exception:
        return

    original_normalize = tabddpm_data.normalize
    if getattr(original_normalize, "_flux_quantile_compat", False):
        return

    def normalize_compat(X, normalization, seed, return_normalizer=False):
        if normalization != "quantile":
            return original_normalize(X, normalization, seed, return_normalizer)
        normalizer = tabddpm_data.sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X["train"].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
        normalizer.fit(X["train"])
        transformed = {k: normalizer.transform(v) for k, v in X.items()}
        if return_normalizer:
            return transformed, normalizer
        return transformed

    normalize_compat._flux_quantile_compat = True
    tabddpm_data.normalize = normalize_compat
    lib.normalize = normalize_compat


def run_ctgan_or_tvae(
    method: str,
    train_df: pd.DataFrame,
    codec: CategoryCodec,
    spec: LendingClubSpec,
    cfg: LendingClubAllMethodsConfig,
    out_dir: Path,
    n: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
    sys.path.insert(0, str(ROOT / "external" / "CTGAN"))
    try:
        from ctgan import CTGAN, TVAE

        enable_gpu = choose_device(cfg.device).type == "cuda"
        if method == "ctgan":
            batch_size = max(10, (cfg.batch_size // 10) * 10)
            model = CTGAN(
                epochs=cfg.ctgan_epochs,
                batch_size=batch_size,
                generator_dim=(128, 128),
                discriminator_dim=(128, 128),
                verbose=False,
                enable_gpu=enable_gpu,
            )
        elif method == "tvae":
            model = TVAE(
                epochs=cfg.tvae_epochs,
                batch_size=cfg.batch_size,
                compress_dims=(128, 128),
                decompress_dims=(128, 128),
                verbose=False,
                enable_gpu=enable_gpu,
            )
        else:
            raise ValueError(method)

        model.fit(train_df[list(CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS)], discrete_columns=list(CATEGORICAL_COLUMNS))
        sampled = model.sample(n)
        sampled.to_csv(out_dir / f"{method}_raw.csv", index=False)
        medians = train_df[list(CONTINUOUS_COLUMNS)].median(numeric_only=True)
        x, k = generated_frame_to_tensors(sampled, codec, spec, medians, choose_device(cfg.device))
        save_lendingclub_csv(out_dir / f"{method}.csv", spec, codec, x, k)
        return x, k, None
    except Exception as exc:
        return None, None, repr(exc)


def prepare_lendingclub_tabddpm_data(
    data_dir: Path,
    x: torch.Tensor,
    k: torch.Tensor,
    seed: int,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    x_np = x.detach().cpu().numpy().astype(np.float32)
    k_np = k.detach().cpu().numpy().astype(np.int64)
    n = x_np.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx = idx[:n_val]
    test_idx = idx[n_val : 2 * n_val]
    train_idx = idx[2 * n_val :]
    if len(test_idx) == 0:
        test_idx = val_idx
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split, split_idx in splits.items():
        np.save(data_dir / f"X_num_{split}.npy", x_np[split_idx, :-1])
        np.save(data_dir / f"X_cat_{split}.npy", k_np[split_idx])
        np.save(data_dir / f"y_{split}.npy", x_np[split_idx, -1])
    info = {
        "name": "lendingclub_structured",
        "id": "lendingclub_structured",
        "task_type": "regression",
        "n_classes": None,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "n_num_features": int(x_np.shape[1] - 1),
        "n_cat_features": int(k_np.shape[1]),
    }
    (data_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return data_dir


def run_lendingclub_tabddpm(
    train_x: torch.Tensor,
    train_k: torch.Tensor,
    codec: CategoryCodec,
    spec: LendingClubSpec,
    cfg: LendingClubAllMethodsConfig,
    out_dir: Path,
    n: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
    source_root = ROOT / "external" / "tab-ddpm"
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "scripts"))
    try:
        from scripts.sample import sample as tabddpm_sample
        from scripts.train import train as tabddpm_train

        patch_tabddpm_quantile_normalizer()
        device = choose_device(cfg.device)
        run_dir = out_dir / "tabddpm_official_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        data_dir = prepare_lendingclub_tabddpm_data(run_dir / "tabddpm_data", train_x, train_k, cfg.seed)
        model_params = {
            "num_classes": 0,
            "is_y_cond": False,
            "rtdl_params": {
                "d_layers": [cfg.hidden_dim, cfg.hidden_dim],
                "dropout": 0.0,
            },
        }
        transform = {
            "seed": cfg.seed,
            "normalization": "quantile",
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default",
        }
        tabddpm_train(
            parent_dir=str(run_dir),
            real_data_path=str(data_dir),
            steps=cfg.tabddpm_steps,
            lr=2e-3,
            weight_decay=0.0,
            batch_size=cfg.batch_size,
            model_type="mlp",
            model_params=model_params.copy(),
            num_timesteps=cfg.tabddpm_timesteps,
            gaussian_loss_type="mse",
            scheduler="cosine",
            T_dict=transform,
            num_numerical_features=spec.continuous_dim - 1,
            device=device,
            seed=cfg.seed,
        )
        tabddpm_sample(
            parent_dir=str(run_dir),
            real_data_path=str(data_dir),
            batch_size=min(cfg.batch_size, n),
            num_samples=n,
            model_type="mlp",
            model_params=model_params.copy(),
            model_path=str(run_dir / "model.pt"),
            num_timesteps=cfg.tabddpm_timesteps,
            gaussian_loss_type="mse",
            scheduler="cosine",
            T_dict=transform,
            num_numerical_features=spec.continuous_dim - 1,
            device=device,
            seed=cfg.seed,
        )
        x_num = np.load(run_dir / "X_num_train.npy", allow_pickle=True)[:n].astype(np.float32)
        y = np.load(run_dir / "y_train.npy", allow_pickle=True)[:n].reshape(-1, 1).astype(np.float32)
        x = spec.clamp_continuous(torch.tensor(np.concatenate([x_num, y], axis=1), device=device))
        k_np = np.load(run_dir / "X_cat_train.npy", allow_pickle=True)[:n].astype(np.int64)
        clipped = []
        for idx, size in enumerate(spec.category_sizes):
            clipped.append(np.clip(k_np[:, idx], 0, size - 1))
        k = torch.tensor(np.stack(clipped, axis=1), device=device, dtype=torch.long)
        save_lendingclub_csv(out_dir / "tabddpm_official.csv", spec, codec, x, k)
        return x, k, None
    except Exception as exc:
        return None, None, repr(exc)


def run_lendingclub_tabsyn_adapter(
    train_x: torch.Tensor,
    train_k: torch.Tensor,
    codec: CategoryCodec,
    spec: LendingClubSpec,
    cfg: LendingClubAllMethodsConfig,
    out_dir: Path,
    n: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
    source_root = ROOT / "external" / "tabsyn"
    sys.path.insert(0, str(source_root))
    try:
        from torch.utils.data import DataLoader, TensorDataset

        from tabsyn.diffusion_utils import sample as tabsyn_sample
        from tabsyn.model import MLPDiffusion, Model
        from tabsyn.vae.model import Decoder_model, Encoder_model, Model_VAE

        device = choose_device(cfg.device)
        categories = list(spec.category_sizes)
        d_token = 4
        num_layers = 2
        n_head = 1
        factor = 32

        normalizer = ContinuousNormalizer(train_x)
        x_num = normalizer.transform(train_x)
        x_cat = train_k.long()
        loader = DataLoader(
            TensorDataset(x_num, x_cat),
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        vae = Model_VAE(num_layers, spec.continuous_dim, categories, d_token, n_head=n_head, factor=factor).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        ce = torch.nn.CrossEntropyLoss()
        for _ in range(cfg.tabsyn_vae_epochs):
            for batch_num, batch_cat in loader:
                batch_num = batch_num.to(device)
                batch_cat = batch_cat.to(device)
                recon_num, recon_cat, mu_z, logvar_z = vae(batch_num, batch_cat)
                mse = (batch_num - recon_num).pow(2).mean()
                cat_loss = torch.stack(
                    [ce(logits, batch_cat[:, idx]) for idx, logits in enumerate(recon_cat)]
                ).mean()
                kld = -0.5 * torch.mean((1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).mean(-1))
                loss = mse + cat_loss + 1e-3 * kld
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        encoder = Encoder_model(num_layers, spec.continuous_dim, categories, d_token, n_head, factor).to(device)
        decoder = Decoder_model(num_layers, spec.continuous_dim, categories, d_token, n_head, factor).to(device)
        encoder.load_weights(vae)
        decoder.load_weights(vae)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z = encoder(x_num, x_cat)[:, 1:, :].reshape(train_x.shape[0], -1)
        latent_mean = z.mean(0, keepdim=True)
        z_train = (z - latent_mean) / 2.0

        denoise_fn = MLPDiffusion(z_train.shape[1], 256).to(device)
        diffusion = Model(denoise_fn=denoise_fn, hid_dim=z_train.shape[1]).to(device)
        optimizer_diff = torch.optim.Adam(diffusion.parameters(), lr=1e-3)
        latent_loader = DataLoader(
            TensorDataset(z_train.detach()),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        for _ in range(cfg.tabsyn_diffusion_epochs):
            for (batch,) in latent_loader:
                loss = diffusion(batch.to(device)).mean()
                optimizer_diff.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_diff.step()

        diffusion.eval()
        latent = tabsyn_sample(
            diffusion.denoise_fn_D,
            n,
            z_train.shape[1],
            num_steps=cfg.tabsyn_sample_steps,
            device=str(device),
        )
        latent = latent * 2.0 + latent_mean
        with torch.no_grad():
            decoded_num, decoded_cat = decoder(latent.reshape(n, -1, d_token))
        x = spec.clamp_continuous(normalizer.inverse(decoded_num))
        k = torch.stack([logits.argmax(dim=1) for logits in decoded_cat], dim=1).long()
        save_lendingclub_csv(out_dir / "tabsyn_adapter.csv", spec, codec, x, k)
        return x, k, None
    except Exception as exc:
        return None, None, repr(exc)


def run_lendingclub_experiment(cfg: LendingClubConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, ref_df, codec = load_lendingclub_data(cfg)
    train_x, train_k, ref_x, ref_k = tensors_from_frames(train_df, ref_df, codec, device)
    spec = build_spec(train_x, codec, cfg)

    full_normalizer = ContinuousNormalizer(train_x)
    vae, vae_history = train_mixed_vae(spec, train_x, train_k, full_normalizer, cfg)
    latents = encode_latent_mean(spec, vae, train_x, train_k, full_normalizer)
    latent_mean = latents.mean(dim=0, keepdim=True)
    latent_std = latents.std(dim=0, keepdim=True).clamp_min(0.05)
    vae_raw_x, vae_raw_k = sample_vae_raw(spec, vae, full_normalizer, latent_mean, latent_std, ref_x.shape[0])
    vae_repair_x = spec.repair(vae_raw_x, vae_raw_k)

    free_normalizer = make_free_flow_normalizer(
        train_x[:, : spec.free_dim],
        whiten_shrinkage=cfg.free_whiten_shrinkage,
    )
    flow, flow_history = train_original_space_free_flow(spec, train_x, train_k, free_normalizer, cfg)
    original_x, original_k = sample_original_space_free_flow(
        spec,
        flow,
        free_normalizer,
        train_k,
        ref_x.shape[0],
        cfg,
    )

    bootstrap_x, bootstrap_k = sample_train_bootstrap(train_x, train_k, ref_x.shape[0])
    independent_x, independent_k = sample_independent_columns(train_x, train_k, ref_x.shape[0])

    methods = {
        "independent_columns": (independent_x, independent_k),
        "vae_raw": (vae_raw_x, vae_raw_k),
        "vae_posthoc_repair": (vae_repair_x, vae_raw_k),
        "original_space_constrained_flow": (original_x, original_k),
        "train_bootstrap_sanity": (bootstrap_x, bootstrap_k),
    }
    for name, (x, k) in methods.items():
        save_lendingclub_csv(out_dir / f"{name}.csv", spec, codec, x, k)

    rows = [
        evaluate_lendingclub_samples(spec, "independent_columns", independent_x, independent_k, ref_x, ref_k),
        evaluate_lendingclub_samples(spec, "vae_raw", vae_raw_x, vae_raw_k, ref_x, ref_k),
        evaluate_lendingclub_samples(
            spec,
            "vae_posthoc_repair",
            vae_repair_x,
            vae_raw_k,
            ref_x,
            ref_k,
            pre_repair_x=vae_raw_x,
        ),
        evaluate_lendingclub_samples(
            spec,
            "original_space_constrained_flow",
            original_x,
            original_k,
            ref_x,
            ref_k,
        ),
        evaluate_lendingclub_samples(spec, "train_bootstrap_sanity", bootstrap_x, bootstrap_k, ref_x, ref_k),
    ]
    write_metric_csv(out_dir / "metrics.csv", rows)
    write_main_table(out_dir / "main_table.md", rows)

    result: dict[str, Any] = {
        "config": asdict(cfg),
        "device": str(device),
        "columns": {
            "continuous": list(CONTINUOUS_COLUMNS),
            "free_continuous": list(FREE_CONTINUOUS_COLUMNS),
            "dependent_continuous": list(DEPENDENT_CONTINUOUS_COLUMNS),
            "categorical": list(CATEGORICAL_COLUMNS),
        },
        "category_names": codec.names,
        "thresholds": {
            "installment": cfg.installment_tolerance,
            "loan_income_ratio": cfg.ratio_tolerance,
            "fico_range_high": cfg.fico_tolerance,
        },
        "histories": {
            "vae": vae_history,
            "original_space_free_flow": flow_history,
        },
        "source_metadata": external_source_metadata(),
        "method_positioning": {
            "implemented_main_method": "original_space_structural_coordinate_flow",
            "objective": "rectified-flow velocity regression over structural free coordinates",
            "not_implemented": "full KDE/projected-Fisher Algorithm 1 Flux Matching loss",
        },
        "metrics": rows,
        "notes": {
            "vae_raw": "A lightweight mixed VAE proxy trained on the full table without structural enforcement.",
            "vae_posthoc_repair": "The same VAE samples after deterministic original-space repair of dependent columns.",
            "original_space_constrained_flow": (
                "Hard categories are sampled from the empirical training joint distribution; "
                "a conditional rectified-flow model generates free continuous coordinates; "
                "dependent columns are reconstructed by the LendingClub structural equations. "
                "This is an original-space structural-coordinate constrained flow motivated "
                "by Flux/v-Flux, not a full projected-Fisher Flux-loss implementation."
            ),
            "train_bootstrap_sanity": "A non-generative sanity upper bound obtained by resampling training rows.",
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_lendingclub_all_methods(cfg: LendingClubAllMethodsConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, ref_df, codec = load_lendingclub_data(cfg)
    train_x, train_k, ref_x, ref_k = tensors_from_frames(train_df, ref_df, codec, device)
    spec = build_spec(train_x, codec, cfg)

    rows: list[dict[str, float | str | int]] = []
    failures: dict[str, str] = {}

    if not cfg.skip_internal:
        full_normalizer = ContinuousNormalizer(train_x)
        vae, vae_history = train_mixed_vae(spec, train_x, train_k, full_normalizer, cfg)
        latents = encode_latent_mean(spec, vae, train_x, train_k, full_normalizer)
        latent_mean = latents.mean(dim=0, keepdim=True)
        latent_std = latents.std(dim=0, keepdim=True).clamp_min(0.05)
        vae_raw_x, vae_raw_k = sample_vae_raw(spec, vae, full_normalizer, latent_mean, latent_std, ref_x.shape[0])
        vae_repair_x = spec.repair(vae_raw_x, vae_raw_k)

        free_normalizer = make_free_flow_normalizer(
            train_x[:, : spec.free_dim],
            whiten_shrinkage=cfg.free_whiten_shrinkage,
        )
        flow, flow_history = train_original_space_free_flow(spec, train_x, train_k, free_normalizer, cfg)
        original_x, original_k = sample_original_space_free_flow(
            spec,
            flow,
            free_normalizer,
            train_k,
            ref_x.shape[0],
            cfg,
        )

        save_lendingclub_csv(out_dir / "vae_raw.csv", spec, codec, vae_raw_x, vae_raw_k)
        save_lendingclub_csv(out_dir / "vae_posthoc_repair.csv", spec, codec, vae_repair_x, vae_raw_k)
        save_lendingclub_csv(out_dir / "original_space_constrained_flow.csv", spec, codec, original_x, original_k)

        rows.extend(
            [
                evaluate_lendingclub_samples(spec, "vae_raw", vae_raw_x, vae_raw_k, ref_x, ref_k),
                evaluate_lendingclub_samples(
                    spec,
                    "vae_posthoc_repair",
                    vae_repair_x,
                    vae_raw_k,
                    ref_x,
                    ref_k,
                    pre_repair_x=vae_raw_x,
                ),
                evaluate_lendingclub_samples(
                    spec,
                    "original_space_constrained_flow",
                    original_x,
                    original_k,
                    ref_x,
                    ref_k,
                ),
            ]
        )
        histories: dict[str, Any] = {
            "vae": vae_history,
            "original_space_free_flow": flow_history,
        }
    else:
        histories = {}

    independent_x, independent_k = sample_independent_columns(train_x, train_k, ref_x.shape[0])
    bootstrap_x, bootstrap_k = sample_train_bootstrap(train_x, train_k, ref_x.shape[0])
    save_lendingclub_csv(out_dir / "independent_columns.csv", spec, codec, independent_x, independent_k)
    save_lendingclub_csv(out_dir / "train_bootstrap_sanity.csv", spec, codec, bootstrap_x, bootstrap_k)
    rows.insert(0, evaluate_lendingclub_samples(spec, "independent_columns", independent_x, independent_k, ref_x, ref_k))
    rows.append(evaluate_lendingclub_samples(spec, "train_bootstrap_sanity", bootstrap_x, bootstrap_k, ref_x, ref_k))

    if not cfg.skip_ctgan:
        x, k, error = run_ctgan_or_tvae("ctgan", train_df, codec, spec, cfg, out_dir, ref_x.shape[0])
        if error:
            failures["ctgan"] = error
        elif x is not None and k is not None:
            rows.append(evaluate_lendingclub_samples(spec, "ctgan", x, k, ref_x, ref_k))

    if not cfg.skip_tvae:
        x, k, error = run_ctgan_or_tvae("tvae", train_df, codec, spec, cfg, out_dir, ref_x.shape[0])
        if error:
            failures["tvae"] = error
        elif x is not None and k is not None:
            rows.append(evaluate_lendingclub_samples(spec, "tvae", x, k, ref_x, ref_k))

    if not cfg.skip_tabsyn:
        x, k, error = run_lendingclub_tabsyn_adapter(train_x, train_k, codec, spec, cfg, out_dir, ref_x.shape[0])
        if error:
            failures["tabsyn_adapter"] = error
        elif x is not None and k is not None:
            rows.append(evaluate_lendingclub_samples(spec, "tabsyn_adapter", x, k, ref_x, ref_k))
            x_repair = spec.repair(x, k)
            save_lendingclub_csv(out_dir / "tabsyn_adapter_posthoc_repair.csv", spec, codec, x_repair, k)
            rows.append(
                evaluate_lendingclub_samples(
                    spec,
                    "tabsyn_adapter_posthoc_repair",
                    x_repair,
                    k,
                    ref_x,
                    ref_k,
                    pre_repair_x=x,
                )
            )

    if not cfg.skip_tabddpm:
        x, k, error = run_lendingclub_tabddpm(train_x, train_k, codec, spec, cfg, out_dir, ref_x.shape[0])
        if error:
            failures["tabddpm_official"] = error
        elif x is not None and k is not None:
            rows.append(evaluate_lendingclub_samples(spec, "tabddpm_official", x, k, ref_x, ref_k))

    order = {
        "independent_columns": 0,
        "vae_raw": 1,
        "vae_posthoc_repair": 2,
        "ctgan": 3,
        "tvae": 4,
        "tabsyn_adapter": 5,
        "tabsyn_adapter_posthoc_repair": 6,
        "tabddpm_official": 7,
        "original_space_constrained_flow": 8,
        "train_bootstrap_sanity": 9,
    }
    rows = sorted(rows, key=lambda row: order.get(str(row["model"]), 999))
    write_metric_csv(out_dir / "metrics.csv", rows)
    write_main_table(out_dir / "main_table.md", rows)

    result: dict[str, Any] = {
        "config": asdict(cfg),
        "device": str(device),
        "columns": {
            "continuous": list(CONTINUOUS_COLUMNS),
            "free_continuous": list(FREE_CONTINUOUS_COLUMNS),
            "dependent_continuous": list(DEPENDENT_CONTINUOUS_COLUMNS),
            "categorical": list(CATEGORICAL_COLUMNS),
        },
        "category_names": codec.names,
        "thresholds": {
            "installment": cfg.installment_tolerance,
            "loan_income_ratio": cfg.ratio_tolerance,
            "fico_range_high": cfg.fico_tolerance,
        },
        "histories": histories,
        "source_metadata": external_source_metadata(),
        "method_positioning": {
            "implemented_main_method": "original_space_structural_coordinate_flow",
            "objective": "category-conditioned rectified-flow velocity regression over structural free coordinates",
            "not_implemented": "full KDE/projected-Fisher Algorithm 1 Flux Matching loss",
            "baseline_names": {
                "tabsyn_adapter": "Uses external TabSyn model/diffusion classes under local schema and budget; not a complete official CLI pipeline.",
                "tabddpm_official": "Uses official TabDDPM train/sample code path with the local schema encoding.",
            },
        },
        "metrics": rows,
        "failures": failures,
        "notes": {
            "ctgan_tvae": "CTGAN/TVAE receive categorical LendingClub columns as discrete columns and receive no constraint repair.",
            "tabddpm_official": (
                "Official TabDDPM mixed path with fico_range_high encoded as regression y and is_y_cond=false; "
                "outputs receive only domain/category clamping, not structural repair. "
                "Report train steps, diffusion timesteps, external commit, and the quantile "
                "normalizer compatibility patch for paper tables."
            ),
            "tabsyn_adapter": (
                "Uses the external TabSyn VAE and EDM diffusion classes on LendingClub mixed columns, "
                "with shortened local budgets; outputs receive no structural repair. "
                "Do not describe this as a full official TabSyn CLI pipeline."
            ),
            "original_space_constrained_flow": (
                "Hard categories are sampled from the empirical training joint distribution; "
                "a conditional rectified-flow model generates free continuous coordinates; "
                "dependent columns are reconstructed. This is the implemented "
                "original-space structural-coordinate constrained flow."
            ),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return result
