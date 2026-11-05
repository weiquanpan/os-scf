from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTINUOUS_COLUMNS = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]

CATEGORICAL_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]

OUTPUT_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    counts = series.astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def residual_summary(df: pd.DataFrame) -> dict[str, Any]:
    residual = df["cnt"].to_numpy(float) - (
        df["casual"].to_numpy(float) + df["registered"].to_numpy(float)
    )
    abs_values = np.abs(residual)
    workingday_target = (
        (df["holiday"].astype(int) == 0)
        & (~df["weekday"].astype(int).isin([0, 6]))
    ).astype(int)
    workingday_violation = df["workingday"].astype(int).to_numpy() != workingday_target.to_numpy()
    return {
        "cnt_equals_casual_plus_registered": {
            "mean_abs": float(abs_values.mean()),
            "p50_abs": float(np.quantile(abs_values, 0.50)),
            "p95_abs": float(np.quantile(abs_values, 0.95)),
            "p99_abs": float(np.quantile(abs_values, 0.99)),
            "max_abs": float(abs_values.max()),
        },
        "workingday_logic_violation_rate": float(np.mean(workingday_violation)),
    }


def write_summary(output_dir: Path, metadata: dict[str, Any], train_rows: int, reference_rows: int) -> None:
    lines = [
        "# Bike Sharing Structured Preprocessing Summary",
        "",
        f"- input: `{metadata['input']}`",
        f"- source file: `{metadata['source_file']}`",
        f"- rows: {metadata['rows']}",
        f"- train rows: {train_rows}",
        f"- reference rows: {reference_rows}",
        f"- seed: {metadata['seed']}",
        "",
        "## Constraints",
        "",
    ]
    for item in metadata["constraints"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Residual Summary", ""])
    residual = metadata["residual_summary"]["cnt_equals_casual_plus_registered"]
    stat_text = ", ".join(f"{key}={value:.6g}" for key, value in residual.items())
    lines.append(f"- cnt_equals_casual_plus_registered: {stat_text}")
    lines.append(
        "- workingday_logic_violation_rate: "
        f"{metadata['residual_summary']['workingday_logic_violation_rate']:.6g}"
    )
    lines.extend(["", "## Category Counts", ""])
    for column, counts in metadata["category_counts"].items():
        compact = ", ".join(f"{key}={value}" for key, value in list(counts.items())[:30])
        lines.append(f"- {column}: {compact}")
    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preprocess(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    missing = [column for column in OUTPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected BikeSharing columns: {missing}")

    out = df[OUTPUT_COLUMNS].copy()
    for column in CONTINUOUS_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64").astype(str)

    valid = out[CONTINUOUS_COLUMNS].notna().all(axis=1)
    valid &= out[["casual", "registered", "cnt"]].ge(0).all(axis=1)
    out = out.loc[valid].reset_index(drop=True)

    if args.max_rows is not None and len(out) > args.max_rows:
        out = out.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    else:
        out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_size = int(round(len(out) * args.train_frac))
    train = out.iloc[:train_size].reset_index(drop=True)
    reference = out.iloc[train_size:].reset_index(drop=True)

    out.to_csv(output_dir / "bike_sharing_structured.csv", index=False)
    train.to_csv(output_dir / "train.csv", index=False)
    reference.to_csv(output_dir / "reference.csv", index=False)

    metadata: dict[str, Any] = {
        "input": str(input_path),
        "source_file": "hour.csv",
        "output_dir": str(output_dir),
        "seed": args.seed,
        "max_rows": args.max_rows,
        "rows": int(len(out)),
        "train_frac": args.train_frac,
        "columns": {
            "continuous": CONTINUOUS_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS,
            "all": OUTPUT_COLUMNS,
        },
        "constraints": [
            "cnt = casual + registered",
            "workingday = 1[holiday = 0 and weekday not in {0, 6}]",
        ],
        "residual_summary": residual_summary(out),
        "category_counts": {column: value_counts_dict(out[column]) for column in CATEGORICAL_COLUMNS},
        "source_citation": (
            "Fanaee-T, Hadi, and Gama, Joao, Event labeling combining ensemble "
            "detectors and background knowledge, Progress in Artificial Intelligence, 2013."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(output_dir, metadata, len(train), len(reference))

    print(f"Wrote {len(out)} rows to {output_dir / 'bike_sharing_structured.csv'}")
    print(f"Wrote train/reference split: {len(train)} / {len(reference)}")
    print(f"Wrote metadata to {output_dir / 'metadata.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess UCI Bike Sharing hour.csv into a structured benchmark.")
    parser.add_argument("--input", default="data/bike_sharing/hour.csv")
    parser.add_argument("--output-dir", default="data/bike_sharing_preprocessed")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.8)
    return parser


if __name__ == "__main__":
    preprocess(build_parser().parse_args())
