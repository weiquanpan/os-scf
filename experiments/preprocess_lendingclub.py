from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "open_acc",
    "revol_util",
    "total_acc",
    "application_type",
]

OUTPUT_COLUMNS = [
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
    "term_months",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "loan_status",
]

CONTINUOUS_COLUMNS = [
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
]

CATEGORICAL_COLUMNS = [
    "term_months",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "loan_status",
]


def parse_term_months(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")


def parse_percent(series: pd.Series) -> pd.Series:
    clean = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(clean, errors="coerce")


def amortized_installment(loan_amnt: np.ndarray, int_rate_pct: np.ndarray, term_months: np.ndarray) -> np.ndarray:
    annual_rate = int_rate_pct / 100.0
    monthly_rate = annual_rate / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        payment = loan_amnt * monthly_rate / (1.0 - np.power(1.0 + monthly_rate, -term_months))
    zero_rate = np.isclose(monthly_rate, 0.0) | ~np.isfinite(payment)
    payment = np.where(zero_rate, loan_amnt / term_months, payment)
    return payment


def simplify_loan_status(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value)
    if "Fully Paid" in text:
        return "fully_paid"
    if "Charged Off" in text or text == "Default":
        return "charged_off"
    if text == "Current":
        return "current"
    if "Late" in text or text == "In Grace Period":
        return "late_or_grace"
    return "other"


def clean_home_ownership(series: pd.Series) -> pd.Series:
    clean = series.fillna("OTHER").astype(str).str.strip().str.upper()
    return clean.where(clean.isin(["MORTGAGE", "RENT", "OWN"]), "OTHER")


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    counts = series.fillna("<NA>").value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def clean_chunk(chunk: pd.DataFrame, formula_tolerance: float, final_status_only: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {
        "raw_rows": int(len(chunk)),
        "valid_numeric_rows": 0,
        "formula_clean_rows": 0,
        "quality_clean_rows": 0,
        "status_clean_rows": 0,
    }

    numeric = pd.DataFrame(
        {
            "loan_amnt": pd.to_numeric(chunk["loan_amnt"], errors="coerce"),
            "int_rate": parse_percent(chunk["int_rate"]),
            "installment": pd.to_numeric(chunk["installment"], errors="coerce"),
            "annual_inc": pd.to_numeric(chunk["annual_inc"], errors="coerce"),
            "dti": pd.to_numeric(chunk["dti"], errors="coerce"),
            "fico_range_low": pd.to_numeric(chunk["fico_range_low"], errors="coerce"),
            "fico_range_high": pd.to_numeric(chunk["fico_range_high"], errors="coerce"),
            "open_acc": pd.to_numeric(chunk["open_acc"], errors="coerce"),
            "revol_util": pd.to_numeric(chunk["revol_util"], errors="coerce"),
            "total_acc": pd.to_numeric(chunk["total_acc"], errors="coerce"),
            "term_months": parse_term_months(chunk["term"]),
        }
    )

    valid = numeric.notna().all(axis=1)
    valid &= numeric["loan_amnt"].gt(0)
    valid &= numeric["annual_inc"].gt(0)
    valid &= numeric["term_months"].isin([36, 60])
    valid &= numeric["int_rate"].gt(0) & numeric["int_rate"].lt(40)
    stats["valid_numeric_rows"] = int(valid.sum())

    formula = amortized_installment(
        numeric["loan_amnt"].to_numpy(float),
        numeric["int_rate"].to_numpy(float),
        numeric["term_months"].to_numpy(float),
    )
    installment_residual = numeric["installment"].to_numpy(float) - formula
    formula_clean = valid & (np.abs(installment_residual) <= formula_tolerance)
    stats["formula_clean_rows"] = int(formula_clean.sum())

    quality = formula_clean.copy()
    quality &= numeric["annual_inc"].le(1_000_000)
    quality &= numeric["dti"].between(0, 80)
    quality &= numeric["revol_util"].between(0, 180)
    quality &= numeric["open_acc"].between(0, 100)
    quality &= numeric["total_acc"].between(0, 250)
    quality &= (numeric["fico_range_high"] - numeric["fico_range_low"] - 4.0).abs().le(0.01)
    stats["quality_clean_rows"] = int(quality.sum())

    status = chunk["loan_status"].map(simplify_loan_status)
    if final_status_only:
        status_mask = status.isin(["fully_paid", "charged_off"])
    else:
        status_mask = status.notna() & ~status.eq("other")
    keep = quality & status_mask
    stats["status_clean_rows"] = int(keep.sum())

    out = pd.DataFrame(
        {
            "loan_amnt": numeric.loc[keep, "loan_amnt"],
            "int_rate": numeric.loc[keep, "int_rate"],
            "annual_inc": numeric.loc[keep, "annual_inc"],
            "dti": numeric.loc[keep, "dti"],
            "fico_range_low": numeric.loc[keep, "fico_range_low"],
            "open_acc": numeric.loc[keep, "open_acc"],
            "revol_util": numeric.loc[keep, "revol_util"],
            "total_acc": numeric.loc[keep, "total_acc"],
            "installment": numeric.loc[keep, "installment"],
            "loan_income_ratio": numeric.loc[keep, "loan_amnt"] / numeric.loc[keep, "annual_inc"],
            "fico_range_high": numeric.loc[keep, "fico_range_high"],
            "term_months": numeric.loc[keep, "term_months"].astype("int64").astype(str),
            "grade": chunk.loc[keep, "grade"].astype(str).str.strip(),
            "home_ownership": clean_home_ownership(chunk.loc[keep, "home_ownership"]),
            "verification_status": chunk.loc[keep, "verification_status"].astype(str).str.strip(),
            "purpose": chunk.loc[keep, "purpose"].astype(str).str.strip(),
            "application_type": chunk.loc[keep, "application_type"].astype(str).str.strip(),
            "loan_status": status.loc[keep],
        }
    )
    return out[OUTPUT_COLUMNS], stats


def update_totals(totals: dict[str, int], add: dict[str, int]) -> None:
    for key, value in add.items():
        totals[key] = totals.get(key, 0) + int(value)


def keep_priority_sample(current: pd.DataFrame, incoming: pd.DataFrame, max_rows: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if incoming.empty:
        return current
    incoming = incoming.copy()
    incoming["_sample_priority"] = rng.random(len(incoming))
    combined = incoming if current.empty else pd.concat([current, incoming], ignore_index=True)
    if max_rows is not None and len(combined) > max_rows:
        combined = combined.nsmallest(max_rows, "_sample_priority").reset_index(drop=True)
    return combined


def residual_summary(df: pd.DataFrame) -> dict[str, Any]:
    formula = amortized_installment(
        df["loan_amnt"].to_numpy(float),
        df["int_rate"].to_numpy(float),
        df["term_months"].astype(float).to_numpy(),
    )
    installment_residual = df["installment"].to_numpy(float) - formula
    ratio_residual = df["loan_income_ratio"].to_numpy(float) - (
        df["loan_amnt"].to_numpy(float) / df["annual_inc"].to_numpy(float)
    )
    fico_residual = df["fico_range_high"].to_numpy(float) - (df["fico_range_low"].to_numpy(float) + 4.0)

    def summarize_abs(values: np.ndarray) -> dict[str, float]:
        abs_values = np.abs(values)
        return {
            "mean_abs": float(abs_values.mean()),
            "p50_abs": float(np.quantile(abs_values, 0.50)),
            "p95_abs": float(np.quantile(abs_values, 0.95)),
            "p99_abs": float(np.quantile(abs_values, 0.99)),
            "max_abs": float(abs_values.max()),
        }

    return {
        "installment": summarize_abs(installment_residual),
        "loan_income_ratio": summarize_abs(ratio_residual),
        "fico_range_high": summarize_abs(fico_residual),
    }


def write_summary(output_dir: Path, metadata: dict[str, Any], train_rows: int, reference_rows: int) -> None:
    lines = [
        "# LendingClub Structured Preprocessing Summary",
        "",
        f"- input: `{metadata['input']}`",
        f"- sampled clean rows: {metadata['sampled_rows']}",
        f"- train rows: {train_rows}",
        f"- reference rows: {reference_rows}",
        f"- final status only: {metadata['final_status_only']}",
        f"- installment formula tolerance: {metadata['formula_tolerance']}",
        "",
        "## Raw-to-clean Counts",
        "",
    ]
    for key, value in metadata["row_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Constraints", ""])
    for item in metadata["constraints"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Sample Residual Summary", ""])
    for name, stats in metadata["residual_summary"].items():
        stat_text = ", ".join(f"{k}={v:.6g}" for k, v in stats.items())
        lines.append(f"- {name}: {stat_text}")
    lines.extend(["", "## Category Counts", ""])
    for col, counts in metadata["category_counts"].items():
        compact = ", ".join(f"{k}={v}" for k, v in list(counts.items())[:20])
        lines.append(f"- {col}: {compact}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preprocess(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled = pd.DataFrame()
    totals: dict[str, int] = {}
    usecols = set(RAW_COLUMNS)
    for chunk in pd.read_csv(
        input_path,
        usecols=lambda column: column in usecols,
        chunksize=args.chunk_size,
        low_memory=False,
    ):
        clean, stats = clean_chunk(
            chunk,
            formula_tolerance=args.formula_tolerance,
            final_status_only=not args.include_current_statuses,
        )
        update_totals(totals, stats)
        sampled = keep_priority_sample(sampled, clean, args.max_rows, rng)

    if sampled.empty:
        raise RuntimeError("No rows survived preprocessing filters.")

    sampled = sampled.drop(columns=["_sample_priority"]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train_size = int(round(len(sampled) * args.train_frac))
    train = sampled.iloc[:train_size].reset_index(drop=True)
    reference = sampled.iloc[train_size:].reset_index(drop=True)

    sampled.to_csv(output_dir / "lendingclub_structured.csv", index=False)
    train.to_csv(output_dir / "train.csv", index=False)
    reference.to_csv(output_dir / "reference.csv", index=False)

    category_counts = {column: value_counts_dict(sampled[column]) for column in CATEGORICAL_COLUMNS}
    metadata: dict[str, Any] = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "max_rows": args.max_rows,
        "sampled_rows": int(len(sampled)),
        "train_frac": args.train_frac,
        "final_status_only": not args.include_current_statuses,
        "formula_tolerance": args.formula_tolerance,
        "row_counts": totals,
        "columns": {
            "continuous": CONTINUOUS_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS,
            "all": OUTPUT_COLUMNS,
        },
        "constraints": [
            "installment = amortized_payment(loan_amnt, int_rate, term_months)",
            "loan_income_ratio = loan_amnt / annual_inc",
            "fico_range_high = fico_range_low + 4",
        ],
        "residual_summary": residual_summary(sampled),
        "category_counts": category_counts,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(output_dir, metadata, len(train), len(reference))

    print(f"Wrote {len(sampled)} clean rows to {output_dir / 'lendingclub_structured.csv'}")
    print(f"Wrote train/reference split: {len(train)} / {len(reference)}")
    print(f"Wrote metadata to {output_dir / 'metadata.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess LendingClub accepted loans into a structured benchmark table.")
    parser.add_argument("--input", default="data/accepted_2007_to_2018Q4.CSV")
    parser.add_argument("--output-dir", default="data/lendingclub_preprocessed")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--formula-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--include-current-statuses",
        action="store_true",
        help="Keep current/late loans as categorical statuses instead of restricting to final paid/charged-off outcomes.",
    )
    return parser


if __name__ == "__main__":
    preprocess(build_parser().parse_args())
