"""Run uniform, adaptive L1 and adaptive L2 serially, then evaluate their saved curves."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def reuse_uniform_run(source, target, contract, duration):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents:
        raise ValueError("The reused baseline must be outside its destination")
    run = json.loads((source / "run.json").read_text())
    saved = json.loads((source / "executed-model-contract.json").read_text())
    sections = (
        "units",
        "physics",
        "geometry",
        "materials",
        "initial_conditions",
        "boundary_conditions",
        "contact_or_coupling",
        "time",
    )
    if any(saved[key] != contract[key] for key in sections):
        raise ValueError("The uniform baseline has different physical parameters")
    contact = dict(law=contract["contact_or_coupling"]["law"])
    contact.update(
        {key: contract["contact_or_coupling"]["parameters"][key] for key in ("friction", "cut_off", "penalty")}
    )
    if (
        run["status"] != "completed"
        or run["level"] != 0
        or run.get("smoke", False)
        or abs(run.get("final_time", -1) - duration) > 1e-8
        or run["dt"] != contract["time"]["timestep"]
        or run["fine_h"] != contract["discretization"]["grid_or_element_size"]
        or run["coarse_h"] != run["fine_h"]
        or run["contact"] != contact
        or run["precision"] != "float64"
        or run["gauss_number"] != contract["api"]["configuration"]["gauss_number"]
        or run["stabilize"] != contract["api"]["configuration"]["stabilize"]
    ):
        raise ValueError("The uniform baseline is incomplete or has different numerical parameters")
    if not json.loads((source / "evaluation.json").read_text())["passed"]:
        raise ValueError("The uniform baseline failed numerical checks")

    def copy_file(old, new):
        if Path(old).suffix in (".npz", ".vtu"):
            try:
                os.link(old, new)
                return new
            except OSError:
                pass
        return shutil.copy2(old, new)

    shutil.copytree(source, target, copy_function=copy_file)
    provenance = {
        "source": str(source),
        "source_run_sha256": hashlib.sha256((source / "run.json").read_bytes()).hexdigest(),
        "timing_comparable": False,
        "timing_note": "Reused runtime is not a new controlled measurement; the source run had GPU contention",
    }
    (target / "reuse.json").write_text(json.dumps(provenance, indent=2) + "\n")
    run.update(reused_from=str(source), timing_comparable=False)
    (target / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return provenance


def wait_for_dependencies(pids, state, save):
    import psutil

    dependencies = []
    for pid in pids:
        if pid <= 0 or pid == os.getpid():
            raise ValueError(f"Invalid dependency PID: {pid}")
        try:
            process = psutil.Process(pid)
            dependencies.append((process, process.create_time()))
        except psutil.NoSuchProcess:
            continue
    state["dependencies"] = [{"pid": p.pid, "created": created} for p, created in dependencies]
    while dependencies:
        live = []
        for process, created in dependencies:
            try:
                if (
                    process.is_running()
                    and process.status() != psutil.STATUS_ZOMBIE
                    and process.create_time() == created
                ):
                    live.append((process, created))
            except psutil.NoSuchProcess:
                pass
        dependencies = live
        state.update(status="waiting_for_dependencies", waiting_for_pids=[p.pid for p, _ in live])
        save()
        if live:
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "OutputData")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--arch", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--device-memory-gb", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=17.5)
    parser.add_argument(
        "--reuse-uniform", type=Path, help="Reuse a completed matching uniform run; calculate adaptive L1/L2 only"
    )
    parser.add_argument("--wait-gpu", action="store_true", help="Wait for existing GPU processes before starting")
    parser.add_argument(
        "--after-pid", type=int, nargs="+", default=[], help="Wait for these processes, including CPU postprocessing"
    )
    args = parser.parse_args()
    if not args.reference.is_file():
        parser.error(f"Missing historical reference: {args.reference}")
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    state_file = args.output / "queue.json"
    if state_file.exists():
        raise FileExistsError(f"Queue already exists: {state_file}")
    reused = None
    if args.reuse_uniform is not None:
        contract = json.loads((here / "model-contract.json").read_text())
        reused = reuse_uniform_run(args.reuse_uniform, args.output / "uniform", contract, args.duration)
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "source_sha256": hashlib.sha256((here / "strip_footing.py").read_bytes()).hexdigest(),
        "cases": [
            {"name": name, "level": level, "status": "queued"}
            for level, name in enumerate(["uniform", "adaptive_l1", "adaptive_l2"])
        ],
        "reference": str(args.reference),
    }

    def save():
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_file)

    save()
    if reused is not None:
        state["cases"][0].update(status="reused", source=reused["source"], timing_comparable=False)
        state["reused_uniform"] = reused
        save()
    wait_for_dependencies(args.after_pid, state, save)
    if args.wait_gpu and args.arch == "gpu":
        while True:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            )
            pids = [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]
            if not pids:
                break
            state.update(status="waiting_for_gpu", waiting_for_pids=pids)
            save()
            time.sleep(30)
        state.update(status="running", waiting_for_pids=[])
        save()
    failed = False
    state.update(status="running", waiting_for_pids=[])
    production_cases = [row for row in state["cases"] if row["status"] != "reused"]
    state["preflight_cases"] = [dict(row) for row in production_cases]
    scheduled = [(True, row) for row in state["preflight_cases"]] + [(False, row) for row in production_cases]
    for preflight, row in scheduled:
        level, name = row["level"], row["name"]
        target = args.output / "preflight" / name if preflight else args.output / name
        duration = min(0.1, args.duration) if preflight else args.duration
        command = [
            sys.executable,
            "-u",
            str(here / "strip_footing.py"),
            "--level",
            str(level),
            "--arch",
            args.arch,
            "--duration",
            str(duration),
            "--device-memory-gb",
            str(args.device_memory_gb),
            "--output",
            str(target),
        ]
        if preflight:
            command += ["--save-interval", str(duration / 5)]
        row.update(
            status="running", command=command, log=str(args.output / f"{'preflight_' if preflight else ''}{name}.log")
        )
        save()
        with Path(row["log"]).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            save()
            row["exit_code"] = process.wait()
        row["status"] = "completed" if row["exit_code"] == 0 else "failed"
        failed |= row["exit_code"] != 0
        save()
        if preflight and not failed:
            with (args.output / f"preflight_{name}_evaluation.log").open("w") as log:
                result = subprocess.run(
                    [sys.executable, str(here / "draw/evaluate.py"), "preflight", str(target)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            row["evaluation_exit_code"] = result.returncode
            failed |= result.returncode != 0
            row["status"] = "passed" if result.returncode == 0 else "failed_preflight"
            save()
        if preflight and failed:
            break
    if not failed:
        command = [
            sys.executable,
            str(here / "draw/evaluate.py"),
            "compare",
            str(args.output),
            "--reference",
            str(args.reference),
        ]
        with (args.output / "comparison.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        state["evaluation_exit_code"] = result.returncode
        state["status"] = "passed" if result.returncode == 0 else "failed_comparison"
    else:
        state["status"] = "failed_run"
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    if state["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
