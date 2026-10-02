#!/usr/bin/env python3
"""Fully implicit FEM soft-particle triaxial compression with pressure-controlled ABD walls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
PLATE_MESH = ROOT / "assets/mesh/AffineBody/abd_compression_plate.obj"
PLATE_AREA = 0.30 * 0.30


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--fem-divisions", type=int, default=6)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--output-every", type=int, default=10)
    parser.add_argument("--axial-speed", type=float, default=0.04)
    parser.add_argument("--confining-pressure", type=float, default=1000.0)
    parser.add_argument("--servo-gain", type=float, default=0.012)
    parser.add_argument("--servo-max-speed", type=float, default=0.025)
    parser.add_argument("--smoke", action="store_true", help="Compile gate without contact-effect acceptance")
    parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData/pressure_controlled"))
    args = parser.parse_args()
    values = (
        args.fem_divisions,
        args.steps,
        args.dt,
        args.output_every,
        args.axial_speed,
        args.confining_pressure,
        args.servo_gain,
        args.servo_max_speed,
    )
    if not all(math.isfinite(float(value)) and value > 0 for value in values):
        parser.error("mesh, time, load and servo controls must be finite and positive")
    return args


def plate_specs():
    return (
        ([0.318, 0.500, 0.500], [0.0, 90.0, 0.0]),
        ([0.682, 0.500, 0.500], [0.0, 90.0, 0.0]),
        ([0.500, 0.318, 0.500], [90.0, 0.0, 0.0]),
        ([0.500, 0.682, 0.500], [90.0, 0.0, 0.0]),
        ([0.500, 0.500, 0.368], [0.0, 0.0, 0.0]),
        ([0.500, 0.500, 0.632], [0.0, 0.0, 0.0]),
    )


def main():
    args = arguments()
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import geotaichi as gt
    from examples.fedem.FEMAffineTriaxial.fem_affine_triaxial import _add_triaxial_fem

    gt.init(arch=args.arch, default_fp="float64", log=True, offline_cache=False)
    dem = gt.DEM(log=True)
    dem.set_configuration(
        domain=[1.0, 1.0, 1.0],
        scheme="AffineBody",
        search="BVH",
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        log=True,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=1.0e9,
        local_damping=0.0,
        contact_damping_stiffness=0.0,
        hessian_shift=0.0,
        friction_mode="lagged",
        friction_iterations=1,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": 6,
            "surface_node_number": 256,
            "max_point_triangle_pairs": 1024,
            "max_edge_edge_pairs": 2048,
            "body_coordination_number": 32,
            "wall_coordination_number": 1,
            "compaction_ratio": [1.0, 1.0],
        },
        log=True,
    )
    dem.add_attribute(0, {"Density": 2500.0})
    dem.add_template({"Name": "plate", "TemplateType": "AffineBody", "Object": gt.polyhedron(file=str(PLATE_MESH))})
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": [
                {
                    "Name": "plate",
                    "GroupID": 1,
                    "MaterialID": 0,
                    "BodyPoint": center,
                    "ScaleFactor": 1.0,
                    "BodyOrientation": orientation,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "YoungModulus": 1.0e9,
                    "Friction": 0.25,
                }
                for center, orientation in plate_specs()
            ],
        }
    )
    dem.add_property(0, 0, {"Dhat": 0.012, "BarrierStiffness": 4.0e4, "Friction": 0.25}, dType="all")

    fem = gt.FEM(log=True)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    _add_triaxial_fem(fem, args.fem_divisions)

    coupling = gt.FEDEM(dem=dem, fem=fem, log=True)
    coupling.set_configuration(domain=[1.0, 1.0, 1.0], search="BVH", gravity=[0.0, 0.0, 0.0], log=True)
    coupling.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.steps * args.dt,
            "SaveInterval": args.output_every * args.dt,
            "SavePath": args.output_dir,
            "assemble_type": "HashTriplet",
            "linear_solver": "PCG",
            "residual_tolerance": 1.0e-7,
            "absolute_tolerance": 1.0e-10,
            "correction_velocity_tolerance": 1.0e-3,
            "max_iterations": 100,
            "linear_solver_tolerance": 1.0e-8,
            "linear_solver_relative_tolerance": 1.0e-8,
            "linear_solver_max_iters": 3000,
            "project_pd": True,
            "enable_step_retry": True,
            "step_retry_max_retries": 3,
            "step_retry_reduction": 0.5,
            "step_retry_minimum_timestep": args.dt / 8.0,
        },
        log=True,
    )
    coupling.add_surface()
    surface_facets = int(coupling.surface_faces.shape[0])
    capacity = max(4096, 6 * surface_facets * 4)
    coupling.memory_allocate(
        {
            "max_contact_pairs": capacity,
            "max_point_triangle_pairs": capacity,
            "max_edge_edge_pairs": 2 * capacity,
            "contact_coordination_number": 64,
        }
    )
    coupling.choose_contact_model(
        "IPC",
        dhat=0.012,
        dmin=0.0,
        kappa=4.0e4,
        friction_coefficient=0.25,
        epsv=1.0e-3,
        friction_mode="lagged",
        friction_iterations=2,
    )
    for body_id in range(6):
        coupling.add_ipc_property(
            AffineBody=body_id,
            FEMbody=0,
            property={"dhat": 0.012, "kappa": 4.0e4, "friction_coefficient": 0.25, "epsv": 1.0e-3},
        )
    inward = ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0])
    for body_id, normal in enumerate(inward):
        coupling.add_affine_body_pressure_servo(
            body_id,
            normal,
            PLATE_AREA,
            args.confining_pressure,
            velocity_gain=args.servo_gain,
            max_velocity=args.servo_max_speed,
        )
    coupling.prescribe_affine_body_velocity(4, [0.0, 0.0, 0.0])
    coupling.prescribe_affine_body_velocity(5, [0.0, 0.0, -args.axial_speed])

    curve = []

    def record(engine):
        latest = engine.affine_pressure_history[-1] if engine.affine_pressure_history else {"walls": []}
        pressures = [wall["measured_pressure"] for wall in latest["walls"]]
        curve.append(
            {
                "step": int(engine.step_count),
                "time": float(engine.time),
                "axial_strain": float(args.axial_speed * engine.time / 0.20),
                "mean_confining_pressure": float(np.mean(pressures)) if pressures else 0.0,
                "maximum_active_contacts": int(engine.last_step_record["contact"]["active_contacts"]),
            }
        )

    result = coupling.run(verbose=False, postprocessing=[record])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pressure_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve[0])
        writer.writeheader()
        writer.writerows(curve)
    tail = curve[-max(5, len(curve) // 10) :]
    tail_pressure = float(np.mean([row["mean_confining_pressure"] for row in tail]))
    summary = {
        "case": "fully_implicit_fem_soft_particle_affine_pressure_triaxial",
        "converged": bool(result["converged"]),
        "steps": int(result["step"]),
        "completed_time": float(result["time"]),
        "target_confining_pressure": args.confining_pressure,
        "tail_mean_confining_pressure": tail_pressure,
        "relative_tail_pressure_error": abs(tail_pressure - args.confining_pressure) / args.confining_pressure,
        "final_axial_strain": curve[-1]["axial_strain"],
        "maximum_active_contacts": max(row["maximum_active_contacts"] for row in curve),
        "finite": bool(np.isfinite(coupling.enginer.fem.state.position.to_numpy()).all()),
    }
    summary["passed"] = bool(
        summary["converged"]
        and summary["finite"]
        and (args.smoke or (summary["maximum_active_contacts"] > 0 and summary["relative_tail_pressure_error"] <= 0.35))
    )
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["passed"]:
        raise RuntimeError(f"pressure-controlled triaxial validation failed: {summary}")


if __name__ == "__main__":
    main()
