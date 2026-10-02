#!/usr/bin/env python3
"""Time-step energy convergence for three FEM--FEM/FEM--LSDEM impacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_SCRIPT = Path(__file__).resolve().with_name("shape_collision.py")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "OutputData/collision_timestep_convergence"
TIMESTEPS = (1.0e-5, 5.0e-6, 2.5e-6, 1.25e-6)
SIMULATION_TIME = 1.6e-1
CASES = (
    "fem_fem_sphere_sphere",
    "fem_lsdem_sphere_sphere",
    "fem_fem_sphere_cube",
    "fem_lsdem_sphere_cube",
    "fem_fem_cube_cube",
    "fem_lsdem_cube_cube",
)


def _case_parts(case_id: str) -> tuple[str, str]:
    contact_kind, shape_pair = case_id.rsplit("_", 2)[0], "_".join(case_id.rsplit("_", 2)[1:])
    return contact_kind, shape_pair


def _case_root(root: Path, case_id: str, *, preflight: bool) -> Path:
    contact_kind, shape_pair = _case_parts(case_id)
    prefix = root / "preflight" if preflight else root
    return prefix / shape_pair / contact_kind


def _timestep_name(timestep: float) -> str:
    return f"dt_{timestep:.3e}".replace(".", "p").replace("+", "")


def _separated_state_energy_difference(path: Path) -> dict[str, float]:
    """Compare time-level energy before and after complete separation."""

    history_path = path / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    active_indices = [index for index, row in enumerate(rows) if int(float(row["active_contact_count"])) > 0]
    if len(active_indices) < 2:
        raise RuntimeError(f"at least two active-contact samples are required in {history_path}")
    before = rows[0]
    after = rows[-1]
    if int(float(before["active_contact_count"])) != 0:
        raise RuntimeError(f"initial state is not separated in {history_path}")
    if int(float(after["active_contact_count"])) != 0:
        raise RuntimeError(f"final state is not separated in {history_path}")
    if active_indices[-1] >= len(rows) - 1:
        raise RuntimeError(f"contact has not ended before the final state in {history_path}")
    before_energy = float(before["raw_time_level_total_energy"])
    after_energy = float(after["raw_time_level_total_energy"])
    relative_difference = abs(after_energy - before_energy) / max(abs(before_energy), 1.0e-30)
    return {
        "separated_state_relative_energy_difference": relative_difference,
        "before_separation_time": float(before["time"]),
        "after_separation_time": float(after["time"]),
        "before_separation_total_energy": before_energy,
        "after_separation_total_energy": after_energy,
    }


def _read_result(path: Path, timestep: float, command: list[str], returncode: int) -> dict:
    result_path = path / "metrics.json"
    row = {
        "requested_dt": timestep,
        "returncode": returncode,
        "path": str(path.relative_to(REPO_ROOT)),
        "command": command,
        "passed": False,
    }
    if not result_path.is_file():
        return row
    result = json.loads(result_path.read_text(encoding="utf-8"))
    row["passed"] = bool(result.get("passed"))
    summaries = result.get("summaries", [])
    if summaries:
        summary = summaries[0]
        row.update(
            {
                "effective_dt": float(summary["effective_dt"]),
                "maximum_relative_energy_error": float(summary["maximum_energy_residual"]),
                "maximum_raw_time_level_energy_error": float(summary["maximum_raw_time_level_energy_residual"]),
                "steps": int(summary["steps"]),
            }
        )
        row.update(_separated_state_energy_difference(path))
    return row


def _run_case(case_id: str, root: Path, args) -> list[dict]:
    case_root = _case_root(root, case_id, preflight=args.preflight)
    rows = []
    for timestep in TIMESTEPS:
        output = case_root / _timestep_name(timestep)
        command = [
            sys.executable,
            str(CASE_SCRIPT),
            "--arch",
            args.arch,
            "--default-fp",
            args.default_fp,
            "--dt",
            str(timestep),
            "--fem-fem-dt",
            str(timestep),
            "--simulation-time",
            str(args.simulation_time),
            "--only",
            case_id,
            "--output",
            str(output),
        ]
        if args.preflight:
            command.append("--preflight")
        output.mkdir(parents=True, exist_ok=True)
        with (output / ("preflight.log" if args.preflight else "run.log")).open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if args.preflight:
            result_path = output / "preflight.json"
            passed = False
            if result_path.is_file():
                passed = bool(json.loads(result_path.read_text(encoding="utf-8")).get("passed"))
            rows.append(
                {
                    "requested_dt": timestep,
                    "returncode": completed.returncode,
                    "path": str(output.relative_to(REPO_ROOT)),
                    "command": command,
                    "passed": passed,
                }
            )
        else:
            rows.append(_read_result(output, timestep, command, completed.returncode))
    return rows


def _reuse_sphere_fem_fem(seed: Path, root: Path) -> list[dict]:
    source = seed / "timestep_study.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for row in payload["runs"]:
        copied = dict(row)
        copied["reused_existing_calculation"] = True
        copied["source_study"] = str(source.relative_to(REPO_ROOT))
        rows.append(copied)
    requested = [row["requested_dt"] for row in rows]
    if len(rows) != len(TIMESTEPS) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15)
        for actual, expected in zip(requested, TIMESTEPS)
    ):
        raise RuntimeError("the reusable sphere study does not contain the four requested time steps")
    return rows


def _study_payload(case_id: str, rows: list[dict], args) -> dict:
    complete = [row for row in rows if "maximum_relative_energy_error" in row]
    orders = []
    for coarse, fine in zip(complete, complete[1:]):
        coarse_error = coarse["maximum_relative_energy_error"]
        fine_error = fine["maximum_relative_energy_error"]
        if min(coarse_error, fine_error) > 0.0:
            orders.append(math.log(coarse_error / fine_error) / math.log(coarse["effective_dt"] / fine["effective_dt"]))
    errors = [row["maximum_relative_energy_error"] for row in complete]
    separated_energy_differences = [row["separated_state_relative_energy_difference"] for row in complete]
    separated_energy_orders = []
    for coarse, fine in zip(complete, complete[1:]):
        coarse_error = coarse["separated_state_relative_energy_difference"]
        fine_error = fine["separated_state_relative_energy_difference"]
        if min(coarse_error, fine_error) > 0.0:
            separated_energy_orders.append(
                math.log(coarse_error / fine_error) / math.log(coarse["effective_dt"] / fine["effective_dt"])
            )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case": case_id,
        "simulation_time": args.simulation_time,
        "precision": args.default_fp,
        "arch": args.arch,
        "preflight_only": args.preflight,
        "energy_definition": (
            "ordinary time-level kinetic, strain, rigid-body, and contact "
            "energies in completely separated endpoint states; no velocity "
            "or energy projection"
        ),
        "runs": rows,
        "pairwise_observed_orders": orders,
        "separated_state_energy_difference_definition": (
            "absolute difference between ordinary time-level total energy "
            "in the initial and final completely separated states, normalized "
            "by the initial value"
        ),
        "separated_state_energy_difference_pairwise_orders": (separated_energy_orders),
        "separated_state_energy_difference_monotonically_decreases": (
            len(separated_energy_differences) == len(TIMESTEPS)
            and all(
                fine < coarse
                for coarse, fine in zip(
                    separated_energy_differences,
                    separated_energy_differences[1:],
                )
            )
        ),
        "energy_error_monotonically_decreases": (
            len(errors) == len(TIMESTEPS) and all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
        ),
        "passed": all(row.get("passed") is True for row in rows),
    }


def _reanalyze_existing(root: Path, selected: list[str], args) -> list[dict]:
    outcomes = []
    for case_id in selected:
        case_root = _case_root(root, case_id, preflight=False)
        study_path = case_root / "timestep_study.json"
        study = json.loads(study_path.read_text(encoding="utf-8"))
        rows = study["runs"]
        for row in rows:
            run_path = REPO_ROOT / row["path"]
            row.update(_separated_state_energy_difference(run_path))
            row.pop("contact_event_time_level_energy_rms", None)
        payload = _study_payload(case_id, rows, args)
        study_path.write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")
        outcomes.append(payload)
    _write_matrix_summary(root)
    return outcomes


def _write_matrix_summary(root: Path) -> None:
    studies = []
    for case_id in CASES:
        case_root = _case_root(root, case_id, preflight=False)
        path = case_root / "timestep_study.json"
        if path.is_file():
            studies.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_cases": list(CASES),
        "completed_case_count": len(studies),
        "studies": studies,
        "passed": len(studies) == len(CASES) and all(study.get("passed") is True for study in studies),
    }
    (root / "matrix_study.json").write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--default-fp", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simulation-time", type=float, default=SIMULATION_TIME)
    parser.add_argument("--only-case", choices=CASES)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--reanalyze-existing",
        action="store_true",
        help="Derive separated-state energy differences from archived histories.",
    )
    parser.add_argument(
        "--reuse-fem-fem-sphere-study",
        type=Path,
        help="Reuse the already completed four-step FEM--FEM sphere study.",
    )
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = [args.only_case] if args.only_case else list(CASES)
    if args.reanalyze_existing:
        if args.preflight:
            parser.error("--reanalyze-existing cannot be combined with --preflight")
        outcomes = _reanalyze_existing(root, selected, args)
        print(json.dumps({"studies": outcomes}, indent=2), flush=True)
        return 0 if all(item["passed"] for item in outcomes) else 2
    outcomes = []
    for case_id in selected:
        if not args.preflight and case_id == "fem_fem_sphere_sphere" and args.reuse_fem_fem_sphere_study is not None:
            rows = _reuse_sphere_fem_fem(args.reuse_fem_fem_sphere_study.expanduser().resolve(), root)
        else:
            rows = _run_case(case_id, root, args)
        payload = _study_payload(case_id, rows, args)
        case_root = _case_root(root, case_id, preflight=args.preflight)
        case_root.mkdir(parents=True, exist_ok=True)
        filename = "preflight_study.json" if args.preflight else "timestep_study.json"
        (case_root / filename).write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")
        outcomes.append(payload)
    if not args.preflight:
        _write_matrix_summary(root)
    print(json.dumps({"studies": outcomes}, indent=2), flush=True)
    return 0 if all(item["passed"] for item in outcomes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
