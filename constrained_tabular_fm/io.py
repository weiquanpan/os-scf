from __future__ import annotations

from pathlib import Path

import numpy as np


def save_array_bundle(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_table(path: str | Path, columns: tuple[str, ...] | None = None) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if "samples" in data:
            return data["samples"]
        if "generated" in data:
            return data["generated"]
        if "train" in data:
            return data["train"]
        first_key = list(data.keys())[0]
        return data[first_key]
    if suffix == ".csv":
        raw = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        if raw.dtype.names is None:
            return np.genfromtxt(path, delimiter=",", dtype=float)
        names = columns or raw.dtype.names
        return np.stack([raw[name] for name in names], axis=1)
    raise ValueError(f"Unsupported table file type: {path}")

