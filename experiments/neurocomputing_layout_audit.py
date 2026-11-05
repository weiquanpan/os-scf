"""Recreate original row-contiguous FP32 evaluation; audit FP64 sensitivity.

Pandas column selection is column-contiguous. CUDA reduction order can then
change despite bit-identical samples. No recorded result or threshold is edited.
"""
from experiments.neurocomputing.common import *
from experiments.neurocomputing import audit as frozen_audit
import argparse


def main(out):
    out = Path(out)
    archive = out / "audit_initial"
    if (out / "audit/report.json").exists() and not archive.exists():
        shutil.copytree(out / "audit", archive)
    paths = sorted(list(out.glob("split*/final/*/seed*/*.metrics.json")) +
                   list(out.glob("split*/ablation/*/seed*/*.metrics.json")))
    paths += [p.with_name(p.name.replace(".metrics.json", ".csv")) for p in paths]
    before = {str(p.relative_to(out)): sha(p) for p in paths}
    original_metrics = metrics
    sensitivity = []

    def canonical(b, x, k):
        x = x.contiguous()
        result = original_metrics(b, x, k)
        wide = original_metrics(b, x.double(), k)
        sensitivity.append({"fp32_violation_rate": result["violation_rate"],
                            "fp64_violation_rate": wide["violation_rate"],
                            "absolute_rate_difference": abs(result["violation_rate"] - wide["violation_rate"]),
                            "fp32_constraint_mae": result["constraint_mae"],
                            "fp64_constraint_mae": wide["constraint_mae"]})
        return result

    frozen_audit.metrics = canonical
    try:
        result = frozen_audit.audit(out, require_complete=True)
    finally:
        frozen_audit.metrics = original_metrics
    replay = pd.read_csv(out / "audit/metric_replay.csv")
    assert len(replay) == len(sensitivity)
    precision = pd.concat([replay[["dataset", "split", "seed", "method"]], pd.DataFrame(sensitivity)], axis=1)
    precision.to_csv(out / "audit/precision_sensitivity.csv", index=False)
    changed = [p for p, digest in before.items() if sha(out / p) != digest]
    if changed:
        raise RuntimeError(f"Saved evidence changed during replay: {changed}")
    hard = precision[precision.method.str.replace(r"_rounded$", "", regex=True).isin(
        ["osscf", "direct_repair", "tabsyn_repair", "free_tabsyn"])]
    info = {"status": "PASS", "source_sha256": sha(__file__),
            "frozen_audit_sha256": sha(Path(frozen_audit.__file__)),
            "correction": "Restore original row-contiguous FP32 layout; no recorded metric or threshold is changed.",
            "initial_failure_preserved_in": "audit_initial/report.json" if archive.exists() else None,
            "evidence_hashes": before, "evidence_unchanged": True,
            "precision_sensitivity_rows": len(precision),
            "max_fp32_fp64_rate_difference": float(precision.absolute_rate_difference.max()),
            "fp64_hard_equation_max_violation_rate": float(hard.fp64_violation_rate.max())}
    dump(out / "audit/layout_replay.json", info)
    print(json.dumps({"status": result["status"], "rows": len(precision),
                      "max_replay_error": result["max_replay_error"],
                      "max_fp32_fp64_rate_difference": info["max_fp32_fp64_rate_difference"],
                      "evidence_unchanged": True}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    main(p.parse_args().out)
