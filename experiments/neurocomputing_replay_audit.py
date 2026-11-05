"""Additional post-hoc verification of ablations and saved-model sampling.

The core training source snapshot remains immutable. This separate audit writes
only its own results and never replaces saved samples or selects checkpoints.
"""
from experiments.neurocomputing.common import *
from experiments.neurocomputing.models import TabSynAdapter, load
import argparse


def main(out):
    out = Path(out)
    if not (out / "COMPLETE.json").exists():
        raise RuntimeError("Run the complete protocol and core audit first")
    config = json.loads((out / "protocol.json").read_text())
    rows, issues, inputs = [], [], {}
    provenance_data = json.loads((out / "provenance.json").read_text())
    identity = hashlib.sha256(json.dumps({"config": config, "sources": provenance_data["sources"]},
                                        sort_keys=True).encode()).hexdigest()
    if identity != provenance_data["identity"]:
        issues.append("Archived protocol does not match the frozen experiment identity")
    for relative, expected_hash in provenance_data["sources"].items():
        if sha(ROOT / relative) != expected_hash:
            issues.append(f"Current source changed since training: {relative}")
    inputs["protocol.json"] = sha(out / "protocol.json")
    inputs["provenance.json"] = sha(out / "provenance.json")
    # Replay every ablation metric, not just its presence in the core audit.
    bundles = {}
    for path in sorted(out.glob("split*/ablation/*/seed*/*.metrics.json")):
        old = json.loads(path.read_text())
        key = (old["dataset"], old["split"])
        if key not in bundles:
            bundles[key] = bundle(*key, out, "test")
        b = bundles[key]
        sample = path.with_name(path.name.replace(".metrics.json", ".csv"))
        digest = sha(sample)
        if digest != old["sample_sha256"]:
            issues.append(f"Ablation sample hash mismatch: {sample}")
        f = pd.read_csv(sample)
        x = torch.tensor(f[b.cols].to_numpy(np.float32), device=DEVICE).contiguous()
        k = torch.tensor(b.codec.encode(f), device=DEVICE)
        actual = metrics(b, x, k)
        error = max(abs(old[c] - v) for c, v in actual.items())
        if error > 1e-5:
            issues.append(f"Ablation metric mismatch: {sample}: {error}")
        if old["method"].startswith(("osscf_", "direct_repair_")) and actual["violation_rate"] != 0:
            issues.append(f"Ablation hard-constraint failure: {sample}")
        if b.name == "bikesharing" and old["method"].startswith(("osscf_", "direct_repair_")) and actual["logic_violation"] != 0:
            issues.append(f"Ablation working-day logic failure: {sample}")
        rows.append({"kind": "ablation_metrics", "path": str(sample.relative_to(out)), "max_error": error})
        inputs[str(sample.relative_to(out))] = digest
    expected = (len(config["split_seeds"]) * len(config["training_seeds"]) *
                sum(d != "taxi" for d in config["datasets"]) * len(config["flow"]["alphas"]) * 3)
    if len(rows) != expected:
        issues.append(f"Expected {expected} ablation rows, got {len(rows)}")

    # The core audit already recomputes every field in these records. Explicitly
    # test the logical/lattice guarantees as well as the numerical equations.
    hard_methods = {"osscf", "direct_repair", "tabsyn_repair", "free_tabsyn"}
    structure_checks = 0
    for path in sorted(out.glob("split*/final/*/seed*/*.metrics.json")):
        record = json.loads(path.read_text())
        base = record["method"].removesuffix("_rounded")
        if base not in hard_methods:
            continue
        keys = ["violation_rate"]
        if record["dataset"] == "bikesharing":
            keys.append("logic_violation")
        if record["method"].endswith("_rounded"):
            keys.append("integer_violation")
            if record["dataset"] == "lendingclub":
                keys.append("fico_grid_violation")
        for key in keys:
            if record[key] != 0:
                issues.append(f"Declared structural guarantee failed: {path}: {key}={record[key]}")
        structure_checks += 1
        inputs[str(path.relative_to(out))] = sha(path)

    # One fixed (first declared) split/seed per dataset, chosen before inspecting
    # the outputs: saved weights + preprocessing must regenerate the same rows.
    split, s = config["split_seeds"][0], config["training_seeds"][0]
    for ds in config["datasets"]:
        b = bundle(ds, split, out, "val")
        folder = out / f"split{split}" / "final" / ds / f"seed{s}"
        n = config["sizes"][ds][2]
        for method in ["osscf", "direct", "tabsyn", "free_tabsyn"]:
            path = folder / f"{method}.csv"
            if method in ["osscf", "direct"]:
                checkpoint_path = folder / f"{method}.pt"
                state = load(checkpoint_path)
                kind = "hard" if method == "osscf" else "direct"
                dim = len(b.free) if kind == "hard" else len(b.cols)
                model = ConditionalVelocity(dim, sum(b.codec.sizes), config["flow"]["width"]).to(DEVICE)
                model.load_state_dict(state["model"])
                k = empirical_k(b, n, s + 500)
                x = sample_flow(b, model, state["normalizer"], k, s + 600, kind, config["flow"]["sample_steps"])
            else:
                checkpoint_path = folder / method / "model.pt"
                model = TabSynAdapter(b, method == "free_tabsyn", config["tabsyn"])
                model.load_bundle(checkpoint_path)
                x, k = model.sample(n, s + 50000)
            saved = pd.read_csv(path)
            # CSV decimal round-tripping back to the original float32 values.
            expected_x = saved[b.cols].to_numpy(np.float32)
            actual_x = x.detach().cpu().numpy()
            error = float(np.abs(actual_x - expected_x).max())
            numeric_ok = bool(np.allclose(actual_x, expected_x, rtol=1e-6, atol=1e-5))
            categorical_ok = bool(np.array_equal(k.cpu().numpy(), b.codec.encode(saved)))
            if not numeric_ok or not categorical_ok:
                issues.append(f"Saved-model sampling mismatch: {ds}/{method}: {error}, cats={categorical_ok}")
            rows.append({"kind": "saved_model_sampling", "path": str(path.relative_to(out)),
                         "max_error": error, "numeric_ok": numeric_ok, "categorical_ok": categorical_ok})
            inputs[str(path.relative_to(out))] = sha(path)
            inputs[str(checkpoint_path.relative_to(out))] = sha(checkpoint_path)
            del model
    dest = out / "replay_audit"
    dest.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / "checks.csv", index=False)
    result = {"status": "PASS" if not issues else "FAIL", "ablation_metric_rows": expected,
              "saved_model_checks": len(config["datasets"]) * 4, "issues": issues,
              "supplemental_structure_records": structure_checks,
              "source_sha256": sha(__file__), "inputs": inputs,
              "device": str(DEVICE), "interpretation": "Checkpoint replay uses fixed first split/seed; this is not a statistical test or independent code review."}
    dump(dest / "report.json", result)
    print(json.dumps({k: result[k] for k in ["status", "ablation_metric_rows", "saved_model_checks", "issues"]}), flush=True)
    if issues:
        raise RuntimeError("Replay failed; see replay_audit/report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.out)
