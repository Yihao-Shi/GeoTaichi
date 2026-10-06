"""Ten Cate E4 oil-sphere settling with semi-resolved incompressible CFDEM."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.cfdem.SemiResolved.SphereFallingOil.draw.evaluate_sphere import (
    evaluate,
)


def case_parameters(cells_per_diameter=None):
    return {
        "case": "semi-sphere",
        "domain": [0.1, 0.1, 0.16],
        "diameter": 0.015,
        "particle_density": 1120.0,
        "fluid_density": 960.0,
        "viscosity": 0.058,
        "centers": [[0.05, 0.05, 0.1275]],
        "reference_speed": 0.955 * 0.128,
        "unbounded_reference_speed": 0.128,
        "reference_statistic": "peak",
        "reference": "ten Cate et al. (2002), case E4 confined maximum speed",
        "cells_per_diameter": cells_per_diameter or 3.0,
        "dt": 0.00025,
        "time": 1.25,
        "ppc": 2,
        "no_slip_walls": True,
        "tank_contact": True,
        "contact_stiffness": 2000000.0,
        "dem_dt": 1e-05,
        "added_mass_coefficient": 2.0,
        "wall_lubrication_cutoff_cells": 1.0,
        "wall_lubrication_minimum_gap": 1.0e-4,
        "contact_damping_ratio": math.sqrt(1.0 + 2.0 * 960.0 / 1120.0),
        "velocity_experiment": "examples/cfdem/SemiResolved/SphereFallingOil/experiment.csv",
        "experiment_rmse_tolerance": 0.1,
        "experiment_pre_wall_cutoff_s": 1.0,
    }


def grid_parameters(config):
    dx = config["diameter"] / config["cells_per_diameter"]
    counts = [max(8, round(length / dx)) for length in config["domain"]]
    spacing = [config["domain"][d] / counts[d] for d in range(3)]
    return (counts, spacing)


def validate_description(config):
    counts, spacing = grid_parameters(config)
    assert all((count > 0 for count in counts))
    assert all((size > 0.0 for size in spacing))
    assert all((0.0 < center[d] < config["domain"][d] for center in config["centers"] for d in range(3)))
    return {**config, "cell_counts": counts, "element_size": spacing}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(config, args):
    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    from geotaichi import DEMPM, init
    from src.utils.SolverRuntime import python_callback
    import taichi as ti

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts, spacing = grid_parameters(config)
    dt = args.dt or config["dt"]
    simulation_time = args.time or config["time"]
    config = {
        **config,
        "time": simulation_time,
        "dt": dt,
        "wall_lubrication_cutoff": config["wall_lubrication_cutoff_cells"] * min(spacing),
    }
    write_json(output / "configuration.json", config)
    body_count = len(config["centers"])
    init(
        dim=3,
        arch=args.arch,
        default_fp="float64",
        default_ip="int32",
        device_memory_GB=args.device_memory,
        offline_cache=True,
        debug=False,
        kernel_profiler=False,
        log=False,
    )
    dempm = DEMPM()
    dempm.set_configuration(
        domain=config["domain"],
        coupling_scheme="CFDEM",
        cfdem_resolution="SemiResolved",
        particle_interaction=False,
        wall_interaction=False,
        CFD_coupling_domain=[3, 6],
        gravity=[0.0, 0.0, -9.81],
        visualize=args.write_vtu,
    )
    dempm.mpm.set_configuration(
        dimension=3,
        background_damping=0.0,
        alphaPIC=args.alpha_pic,
        mapping="USL",
        shape_function="QuadBSpline",
        gravity=[0.0, 0.0, -9.81],
        material_type="Fluid",
        velocity_projection="PIC",
        solver_type="Implicit",
        discretization="FDM",
        fluid_level_set=False,
        fluid_domain_volume_fraction=0.1,
        particle_shifting=True,
        density_projection=True,
        density_projection_interior_only=True,
        solid_sdf_cut_cell=True,
        fluid_wall_no_slip=True,
        visualize=args.write_vtu,
    )
    implicit = {
        "linear_solver": args.linear_solver,
        "max_iteration_number": 200,
        "residual_tolerance": 1e-8,
        "linear_solver_relative_tolerance": 1e-10,
    }
    if args.linear_solver == "MGPCG":
        multilevel = 3
        implicit.update(multilevel=multilevel, pre_and_post_smoothing=2, bottom_smoothing=20)
    dempm.mpm.set_implicit_solver_parameters(**implicit)
    dempm.dem.set_configuration(
        scheme="DEM",
        boundary=["Destroy", "Destroy", "Destroy"],
        gravity=[0.0, 0.0, -9.81],
        engine="VelocityVerlet",
        search="LinkedCell",
        visualize=args.write_vtu,
    )
    dempm.set_solver(
        {
            "Timestep": dt,
            "DEMTimestep": 1e-05,
            "SimulationTime": simulation_time,
            "SaveInterval": args.save_interval or max(simulation_time / 20.0 - 0.1 * dt, dt),
            "SavePath": str(output),
            "CFL": 0.5,
        }
    )
    dempm.dem.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": body_count,
            "max_sphere_number": body_count,
            "max_clump_number": 0,
            "max_plane_number": 6,
            "body_coordination_number": 0,
            "wall_coordination_number": 3,
            "verlet_distance_multiplier": 0.2,
        }
    )
    allocated_cell_counts = list(counts)
    if args.linear_solver == "MGPCG":
        multiplier = 2 ** (implicit["multilevel"] - 1)
        allocated_cell_counts = [multiplier * math.ceil(count / multiplier) for count in counts]
    max_mpm_particles = math.prod(allocated_cell_counts) * 2**3
    dempm.mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": max_mpm_particles,
            "verlet_distance_multiplier": 0.5,
            "max_constraint_number": {},
        }
    )
    dempm.memory_allocate(
        memory={
            "body_coordination_number": 2 if body_count > 1 else 1,
            "wall_coordination_number": 0,
            "compaction_ratio": [0.2, 0.1],
        }
    )
    dempm.dem.add_attribute(
        materialID=0,
        attribute={"Density": config["particle_density"], "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0},
    )
    dempm.dem.create_body(
        body={
            "BodyType": "Sphere",
            "Template": [
                {
                    "GroupID": body_id,
                    "MaterialID": 0,
                    "BodyPoint": center,
                    "Radius": 0.5 * config["diameter"],
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                    "FixAngularVelocity": ["Free", "Free", "Free"],
                    "BodyOrientation": "uniform",
                }
                for body_id, center in enumerate(config["centers"])
            ],
        }
    )
    dempm.dem.choose_contact_model(particle_particle_contact_model=None, particle_wall_contact_model="Linear Model")
    contact_stiffness = config["contact_stiffness"]
    dempm.dem.add_property(
        materialID1=0,
        materialID2=0,
        property={
            "NormalStiffness": contact_stiffness,
            "TangentialStiffness": 1000000.0,
            "Friction": 0.0,
            "NormalViscousDamping": config["contact_damping_ratio"],
            "TangentialViscousDamping": 0.0,
        },
    )
    walls = []
    for axis in range(3):
        for side in (0, 1):
            center = [0.5 * length for length in config["domain"]]
            center[axis] = side * config["domain"][axis]
            normal = [0.0, 0.0, 0.0]
            normal[axis] = 1.0 if side == 0 else -1.0
            walls.append({"WallType": "Plane", "MaterialID": 0, "WallCenter": center, "OuterNormal": normal})
    dempm.dem.add_wall(body=walls)
    dempm.mpm.add_material(
        model="Newtonian",
        material={
            "MaterialID": 1,
            "Density": config["fluid_density"],
            "Modulus": 2000000.0,
            "Viscosity": config["viscosity"],
            "ElementLength": min(spacing),
            "cL": 1.5,
            "cQ": 2.0,
            "atmospheric_pressure": 0.0,
            "SurfaceTension": 0.0,
        },
    )
    dempm.mpm.add_element(element={"ElementType": "Staggered", "ElementSize": spacing, "GhostCell": 1})
    dempm.mpm.add_region(
        region=[
            {
                "Name": "fluid",
                "Type": "Rectangle",
                "BoundingBoxPoint": [0.0, 0.0, 0.0],
                "BoundingBoxSize": config["domain"],
            }
        ]
    )
    dempm.mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "fluid",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                }
            ]
        }
    )
    boundaries = []
    for direction in range(3):
        for side in (0, 1):
            normal = [0.0, 0.0, 0.0]
            normal[direction] = -1.0 if side == 0 else 1.0
            start = [0.0, 0.0, 0.0]
            end = list(config["domain"])
            start[direction] = side * config["domain"][direction]
            end[direction] = start[direction]
            boundaries.append(
                {"BoundaryType": "SolidCell", "Norm": normal, "StartPoint": start, "EndPoint": end, "CellThickness": 1}
            )
    dempm.mpm.add_boundary_condition(boundary=boundaries)
    dempm.mpm.select_save_data(particle=True, grid=True)
    dempm.dem.select_save_data(particle=True, sphere=True)
    dempm.select_save_data()
    dempm.choose_contact_model(None, None)
    history_t, history_x, history_v = ([], [], [])

    @python_callback
    def record_trajectory():
        if int(dempm.sims.current_step) % args.sample_every:
            return
        positions = dempm.dem.scene.particle.x.to_numpy()[:body_count]
        velocities = dempm.dem.scene.particle.v.to_numpy()[:body_count]
        history_t.append(float(dempm.sims.current_time + dempm.sims.delta))
        history_x.append(positions.copy())
        history_v.append(velocities.copy())

    run_options = {"mpm_gravity_field": True, "function": record_trajectory}
    run_options["drag_model"] = {
        "DragForceModel": "SchillerNaumannModel",
        "DragLaw": "Quadratic",
        "AddedMassCoefficient": config["added_mass_coefficient"],
        "WallLubricationCutoff": config["wall_lubrication_cutoff"],
        "WallLubricationMinimumGap": config["wall_lubrication_minimum_gap"],
    }
    try:
        dempm.run(**run_options)
    finally:
        if history_t:
            np.savez(
                output / "trajectory.npz",
                time=np.asarray(history_t),
                center=np.asarray(history_x),
                velocity=np.asarray(history_v),
            )
    times = np.asarray(history_t)
    centers = np.asarray(history_x)
    velocities = np.asarray(history_v)
    if len(times) < 2:
        raise RuntimeError("trajectory recorder produced fewer than two samples")
    solid_volume_error = 0.0
    ibm_l2_error = 0.0
    actual_counts = [int(count) - 2 * dempm.mpm.scene.element.ghost_cell for count in dempm.mpm.scene.element.cnum]
    actual_spacing = [float(size) for size in dempm.mpm.scene.element.grid_size]
    metrics = evaluate(config, times, centers, velocities, solid_volume_error, ibm_l2_error)
    metrics.update(
        fluid_solver="semi-implicit-incompressible-fdm",
        cell_counts=actual_counts,
        element_size=actual_spacing,
        cells_per_diameter=config["diameter"] / min(actual_spacing),
        requested_timestep=dt,
        alpha_pic=args.alpha_pic,
        timestep=float(np.median(np.diff(times)) / args.sample_every),
    )
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, sort_keys=True))
    if not args.skip_postprocess:
        dempm.mpm.postprocessing()
        dempm.dem.postprocessing()
    if args.strict and (not metrics["passed"]):
        raise SystemExit("Ten Cate E4 validation criteria failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-per-diameter", type=float)
    parser.add_argument("--arch", default=os.environ.get("GT_IBM_VALIDATION_ARCH", "gpu"))
    parser.add_argument("--linear-solver", choices=("PCG", "MGPCG"), default="MGPCG")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "OutputData")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--time", type=float)
    parser.add_argument("--alpha-pic", type=float, default=0.01)
    parser.add_argument("--save-interval", type=float)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--device-memory", type=float, default=8.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--write-vtu", action="store_true")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.sample_every <= 0:
        parser.error("--sample-every must be positive")
    if not 0.0 <= args.alpha_pic <= 1.0:
        parser.error("--alpha-pic must be in [0, 1]")
    return args


def main():
    args = parse_args()
    config = validate_description(case_parameters(cells_per_diameter=args.cells_per_diameter))
    if args.describe:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    run(config, args)


if __name__ == "__main__":
    main()
