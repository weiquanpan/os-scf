from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constrained_tabular_fm.config import ExperimentConfig
from constrained_tabular_fm.runner import run_tabular_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["accounting", "complex", "causal", "panel"],
        default=ExperimentConfig.dataset,
    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--steps", type=int, default=ExperimentConfig.n_train_steps)
    parser.add_argument("--train-rows", type=int, default=ExperimentConfig.n_train_rows)
    parser.add_argument("--eval-rows", type=int, default=ExperimentConfig.n_eval_rows)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--sample-steps", type=int, default=ExperimentConfig.sample_steps)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument(
        "--model-type",
        choices=["linear", "mlp"],
        default=ExperimentConfig.model_type,
    )
    parser.add_argument("--hidden-dim", type=int, default=ExperimentConfig.hidden_dim)
    parser.add_argument("--depth", type=int, default=ExperimentConfig.depth)
    parser.add_argument("--residual-std", type=float, default=ExperimentConfig.residual_std)
    parser.add_argument(
        "--violation-threshold",
        type=float,
        default=ExperimentConfig.violation_threshold,
    )
    parser.add_argument("--repair-rate", type=float, default=ExperimentConfig.repair_rate)
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=ExperimentConfig.constraint_weight,
    )
    parser.add_argument("--device", type=str, default=ExperimentConfig.device)
    parser.add_argument("--output-dir", type=str, default=ExperimentConfig.output_dir)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        seed=args.seed,
        dataset=args.dataset,
        n_train_rows=args.train_rows,
        n_eval_rows=args.eval_rows,
        n_train_steps=args.steps,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        lr=args.lr,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        residual_std=args.residual_std,
        violation_threshold=args.violation_threshold,
        repair_rate=args.repair_rate,
        constraint_weight=args.constraint_weight,
        device=args.device,
        output_dir=args.output_dir,
    )
    results = run_tabular_experiment(cfg)
    print(json.dumps(results["metrics"], indent=2))
    print(f"\nWrote results to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
