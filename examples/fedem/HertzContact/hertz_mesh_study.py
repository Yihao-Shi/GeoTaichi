#!/usr/bin/env python3
"""Run a four-level locally refined Hertz mesh-convergence study."""

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
CASE_SCRIPT = Path(__file__).resolve().with_name("hertz_contact.py")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "OutputData/hertz_mesh_convergence"
DT = 2.5e-6
NORMAL_STIFFNESS = 2.0e9
MESH_LEVELS = (
    {
        "id": "mesh_coarse",
        "coarse_spacing_over_radius": 0.24,
        "contact_spacing_over_radius": 0.060,
        "surface_spacing_over_radius": 0.020,
    },
    {
        "id": "mesh_medium",
        "coarse_spacing_over_radius": 0.18,
        "contact_spacing_over_radius": 0.045,
        "surface_spacing_over_radius": 0.015,
    },
    {
        "id": "mesh_fine",
        "coarse_spacing_over_radius": 0.14,
        "contact_spacing_over_radius": 0.035,
        "surface_spacing_over_radius": 0.012,
    },
    {
        "id": "mesh_ultrafine",
        "coarse_spacing_over_radius": 0.11,
        "contact_spacing_over_radius": 0.0275,
        "surface_spacing_over_radius": 0.009,
    },
)


def _force_indentation_relative_l2(case: Path) -> float:
    config = json.loads((case / "config.json").read_text(encoding="utf-8"))
    parameters = config["parameters"]
    radius = float(parameters["radius"])
    effective_modulus = float(parameters["effective_modulus"])
    ramp_time = float(parameters["ramp_time"])
    numerical = []
    analytical = []
    with (case / "history.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            time_value = float(row["time"])
            indentation = float(row["indentation"])
            reaction = float(row["reaction_force"])
            if (
                time_value <= ramp_time
                and int(float(row["active_contact_count"])) > 0
                and indentation > 0.0
                and reaction > 0.0
            ):
                numerical.append(reaction)
                analytical.append((4.0 / 3.0) * effective_modulus * radius**0.5 * indentation**1.5)
    if len(numerical) < 10:
        raise RuntimeError(f"fewer than ten loading samples are available in {case}")
    numerator = sum((computed - exact) ** 2 for computed, exact in zip(numerical, analytical))
    denominator = sum(exact**2 for exact in analytical)
    if denominator <= 0.0:
        raise RuntimeError(f"zero analytical force norm in {case}")
    return (numerator / denominator) ** 0.5


def _force_contact_radius_relative_l2(case: Path) -> float:
    """Compare the computed loading path with the Hertz F--a relation."""
    config = json.loads((case / "config.json").read_text(encoding="utf-8"))
    parameters = config["parameters"]
    radius = float(parameters["radius"])
    effective_modulus = float(parameters["effective_modulus"])
    ramp_time = float(parameters["ramp_time"])
    target_load = float(parameters["target_load"])
    numerical = []
    analytical = []
    with (case / "history.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            time_value = float(row["time"])
            reaction = float(row["reaction_force"])
            contact_radius = float(row["equivalent_contact_radius"])
            if (
                time_value <= ramp_time
                and int(float(row["active_contact_count"])) > 0
                and reaction >= 0.20 * target_load
                and contact_radius > 0.0
            ):
                numerical.append(reaction)
                analytical.append((4.0 / 3.0) * effective_modulus * contact_radius**3 / radius)
    if len(numerical) < 10:
        raise RuntimeError(f"fewer than ten resolved-contact loading samples are available in {case}")
    numerator = sum((computed - exact) ** 2 for computed, exact in zip(numerical, analytical))
    denominator = sum(exact**2 for exact in analytical)
    if denominator <= 0.0:
        raise RuntimeError(f"zero analytical force norm in {case}")
    return math.sqrt(numerator / denominator)


def _command(level, args, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(CASE_SCRIPT),
        "--arch",
        args.arch,
        "--default-fp",
        args.default_fp,
        "--dt",
        str(args.dt),
        "--mesh-generator",
        "gmsh",
        "--normal-stiffness",
        str(args.normal_stiffness),
        "--contact-model",
        args.contact_model,
        "--coarse-spacing-over-radius",
        str(level["coarse_spacing_over_radius"]),
        "--contact-spacing-over-radius",
        str(level["contact_spacing_over_radius"]),
        "--surface-spacing-over-radius",
        str(level["surface_spacing_over_radius"]),
        "--output",
        str(output),
    ]
    if args.contact_model == "barrier":
        command.extend(("--barrier-cutoff", str(args.barrier_cutoff)))
    if args.preflight:
        command.append("--preflight")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--default-fp", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--normal-stiffness", type=float, default=NORMAL_STIFFNESS)
    parser.add_argument("--contact-model", choices=("linear", "barrier"), default="linear")
    parser.add_argument("--barrier-cutoff", type=float, default=5.0e-6)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild mesh_study.json from complete archived level outputs.",
    )
    parser.add_argument("--only", choices=tuple(level["id"] for level in MESH_LEVELS))
    args = parser.parse_args()
    if args.barrier_cutoff <= 0.0:
        parser.error("--barrier-cutoff must be positive")
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    levels = [level for level in MESH_LEVELS if args.only in (None, level["id"])]

    executions = []
    if args.summarize_only:
        if args.preflight or args.only is not None:
            parser.error("--summarize-only requires all four full-run levels")
        missing = [
            level["id"]
            for level in levels
            if not (root / level["id"] / "metrics.json").is_file()
            or not (root / level["id"] / "performance.json").is_file()
        ]
        if missing:
            parser.error("--summarize-only is missing archived levels: " + ", ".join(missing))
        executions = [
            {
                **level,
                "returncode": 0,
                "command": ["archived-level", str(root / level["id"])],
            }
            for level in levels
        ]
    else:
        for level in levels:
            case_output = root / level["id"]
            case_output.mkdir(parents=True, exist_ok=True)
            command = _command(level, args, case_output)
            log_name = "preflight.log" if args.preflight else "run.log"
            with (case_output / log_name).open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            executions.append(
                {
                    **level,
                    "returncode": completed.returncode,
                    "command": command,
                }
            )
            if completed.returncode != 0:
                break

    study = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dt": args.dt,
        "normal_stiffness": args.normal_stiffness,
        "contact_model": args.contact_model,
        "barrier_cutoff": (args.barrier_cutoff if args.contact_model == "barrier" else None),
        "mesh_generator": "gmsh",
        "precision": args.default_fp,
        "arch": args.arch,
        "preflight_only": args.preflight,
        "summarize_only": args.summarize_only,
        "executions": executions,
    }
    if not args.preflight and len(executions) == len(levels):
        summaries = []
        source_fingerprints = []
        for level in levels:
            case = root / level["id"]
            config = json.loads((case / "config.json").read_text(encoding="utf-8"))
            metrics = json.loads((case / "metrics.json").read_text(encoding="utf-8"))
            performance = json.loads((case / "performance.json").read_text(encoding="utf-8"))
            summaries.append(
                {
                    **level,
                    "passed": metrics["passed"],
                    "fem_nodes": performance["fem_nodes"],
                    "fem_elements": performance["fem_elements"],
                    "pressure_relative_l2": metrics["pressure_profile"]["relative_l2"],
                    "pressure_annular_profile_relative_l2": metrics["pressure_profile"]["annular_profile_relative_l2"],
                    "force_indentation_relative_l2": (_force_indentation_relative_l2(case)),
                    "force_contact_radius_relative_l2": (_force_contact_radius_relative_l2(case)),
                    "contact_radius_relative_error": metrics["mean_final_contact_radius_relative_error"],
                    "indentation_relative_error": metrics["mean_final_indentation_relative_error"],
                    "pressure_annulus_count": metrics["pressure_profile"]["independent_annulus_count"],
                    "pressure_diameter_point_count": metrics["pressure_profile"]["diameter_point_count"],
                    "saved_frame_count": performance["saved_frame_count"],
                    "native_output_complete": metrics["native_output"]["complete"],
                }
            )
            source_fingerprints.append(config.get("execution", {}).get("source_fingerprint"))
        errors = [item["pressure_relative_l2"] for item in summaries]
        force_errors = [item["force_indentation_relative_l2"] for item in summaries]
        force_contact_radius_errors = [item["force_contact_radius_relative_l2"] for item in summaries]
        contact_radius_errors = [item["contact_radius_relative_error"] for item in summaries]
        monotonic = all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
        force_monotonic = all(fine < coarse for coarse, fine in zip(force_errors, force_errors[1:]))
        force_contact_radius_monotonic = all(
            fine < coarse for coarse, fine in zip(force_contact_radius_errors, force_contact_radius_errors[1:])
        )
        contact_radius_monotonic = all(
            fine < coarse for coarse, fine in zip(contact_radius_errors, contact_radius_errors[1:])
        )
        homogeneous_solver_source = bool(
            source_fingerprints and all(source_fingerprints) and len(set(source_fingerprints)) == 1
        )
        study.update(
            {
                "summaries": summaries,
                "pressure_error_monotonically_decreases": monotonic,
                "force_indentation_error_monotonically_decreases": (force_monotonic),
                "force_contact_radius_error_monotonically_decreases": (force_contact_radius_monotonic),
                "contact_radius_error_monotonically_decreases": (contact_radius_monotonic),
                "source_fingerprints": source_fingerprints,
                "homogeneous_solver_source": homogeneous_solver_source,
                "coarse_to_fine_pressure_error_reduction": (
                    (errors[0] - errors[-1]) / errors[0] if errors[0] > 0.0 else 0.0
                ),
                "passed": (all(item["passed"] for item in summaries) and homogeneous_solver_source and monotonic),
            }
        )
    else:
        study["passed"] = all(item["returncode"] == 0 for item in executions)

    filename = "preflight_study.json" if args.preflight else "mesh_study.json"
    (root / filename).write_text(json.dumps(study, indent=2) + os.linesep, encoding="utf-8")
    print(json.dumps(study, indent=2))
    return 0 if study["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
