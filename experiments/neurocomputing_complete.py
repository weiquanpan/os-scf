"""Durable training, corrected layout replay, diagnostics/report coordinator.

The frozen training package remains verbatim. Use --finalize-only to audit
existing evidence without training or changing saved samples and metrics.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def write(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=24)
    args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"; logs.mkdir(exist_ok=True)
    lock = out / "driver.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    started = time.time()
    try:
        history = out / "recovery_history"; history.mkdir(exist_ok=True)
        for name in ["RUN_STATUS.json", "FINALIZE_STATUS.json"]:
            target = history / ("initial_" + name)
            if (out / name).exists() and not target.exists():
                shutil.copy2(out / name, target)
        if not (out / "paper_v6_before").exists():
            shutil.copytree(ROOT / "paper-v6", out / "paper_v6_before")
        stages = []
        if not args.finalize_only:
            stages += [(s, "experiments.neurocomputing.run", ["--stage", s])
                       for s in ["prepare", "flow", "ablation", "tabsyn", "tabddpm", "references"]]
        stages += [(s, "experiments." + s, []) for s in ["neurocomputing_layout_audit",
                   "neurocomputing_raw_category_audit", "neurocomputing_replay_audit", "neurocomputing_report"]]
        for index, (stage, module, extra) in enumerate(stages):
            logpath = logs / f"complete_{stage}.log"
            with logpath.open("a", encoding="utf8") as log:
                child = subprocess.Popen([sys.executable, "-u", "-m", module, "--out", str(out), *extra],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                while True:
                    state = {"status": "RUNNING", "stage": stage, "driver_pid": os.getpid(),
                             "worker_pid": child.pid, "pid": os.getpid(), "detail": stage,
                             "stage_index": index + 1, "stages": len(stages), "log": str(logpath),
                             "elapsed_seconds": round(time.time() - started, 1),
                             "scope": "finalization only" if args.finalize_only else "complete pipeline"}
                    write(out / "RUN_STATUS.json", state); write(out / "FINALIZE_STATUS.json", state)
                    try:
                        code = child.wait(timeout=30); break
                    except subprocess.TimeoutExpired:
                        if time.time() - started > args.timeout_hours * 3600:
                            child.terminate(); child.wait()
                            state.update(status="TIMED_OUT")
                            write(out / "RUN_STATUS.json", state); write(out / "FINALIZE_STATUS.json", state)
                            return 2
                if code:
                    state.update(status="FAILED", exit_code=code)
                    write(out / "RUN_STATUS.json", state); write(out / "FINALIZE_STATUS.json", state)
                    return code
        state.update(status="COMPLETE", elapsed_seconds=round(time.time() - started, 1))
        write(out / "RUN_STATUS.json", state); write(out / "FINALIZE_STATUS.json", state)
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
