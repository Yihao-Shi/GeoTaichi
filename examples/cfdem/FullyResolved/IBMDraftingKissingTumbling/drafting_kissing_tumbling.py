"""Lai drafting--kissing--tumbling benchmark with LSDEM--IBM."""

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

from examples.cfdem.FullyResolved.IBMDraftingKissingTumbling.draw.evaluate_drafting_kissing_tumbling import (
    evaluate,
)


def case_parameters(cells_per_diameter=None):
    diameter = 0.00167
    requested_resolution = cells_per_diameter or 8.35
    if requested_resolution <= 0.0:
        raise ValueError("cells_per_diameter must be positive")
    transverse_cells = max(1, round(0.01 * requested_resolution / diameter))
    cell_counts = [transverse_cells, transverse_cells, 4 * transverse_cells]
    actual_resolution = diameter * transverse_cells / 0.01
    config = {
        "case": "dkt",
        "domain": [0.01, 0.01, 0.04],
        "diameter": diameter,
        "particle_density": 1140.0,
        "fluid_density": 1000.0,
        "viscosity": 0.001,
        "centers": [[0.005, 0.005, 0.0316], [0.005 + 0.01 * diameter, 0.005, 0.035]],
        "reference_contact_time": 0.34,
        "reference_collision_speed": 0.07139,
        "reference": "Lai et al. (2023) DKT; drafting lasts about 0.34 s and the velocity peak marks collision",
        "requested_cells_per_diameter": requested_resolution,
        "cells_per_diameter": actual_resolution,
        "cell_counts": cell_counts,
        "dt": 5e-05,
        "dem_dt": 1e-07,
        "adaptive_contact_subcycling": True,
        "contact_guard_ratio": 0.005,
        "time": 0.60,
        "contact_time_relative_tolerance": 0.05,
        "collision_speed_relative_tolerance": 0.05,
        "solid_volume_relative_tolerance": 0.04,
        "ibm_velocity_relative_tolerance": 0.01,
        "maximum_overlap_ratio": 0.015,
        "minimum_final_distance_ratio": 1.10,
        "minimum_post_kissing_separation_ratio": 0.15,
        "minimum_lateral_growth_ratio": 1.0,
        "maximum_sample_interval": 0.002,
    }
    return config


def grid_parameters(config):
    counts = config["cell_counts"]
    spacing = [config["domain"][d] / counts[d] for d in range(3)]
    return (counts, spacing)


def validate_description(config):
    counts, spacing = grid_parameters(config)
    assert all((count > 0 for count in counts))
    assert all((size > 0.0 for size in spacing))
    assert all((0.0 < center[d] < config["domain"][d] for center in config["centers"] for d in range(3)))
    separation = config["centers"][1][2] - config["centers"][0][2]
    lateral = math.dist(config["centers"][0][:2], config["centers"][1][:2])
    assert math.isclose(separation, 0.0034, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(lateral, 0.01 * config["diameter"], rel_tol=0.0, abs_tol=1e-12)
    assert counts[0] == counts[1] and counts[2] == 4 * counts[0]
    assert math.isclose(config["cells_per_diameter"], config["diameter"] / spacing[0])
    return {**config, "cell_counts": counts, "element_size": spacing}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def contact_substep_required(config, coupled_dt, positions, velocities):
    separation = positions[1] - positions[0]
    distance = float(np.linalg.norm(separation))
    radial_speed = float(np.dot(separation, velocities[1] - velocities[0]) / max(distance, 1.0e-12))
    guard = max(config["contact_guard_ratio"] * config["diameter"], 2.0 * abs(radial_speed) * coupled_dt)
    return distance <= config["diameter"] + guard


def adaptive_dem_timestep(config, dem_dt, coupled_dt, positions, velocities):
    if contact_substep_required(config, coupled_dt, positions, velocities):
        return min(dem_dt, coupled_dt)
    return coupled_dt


def run(config, args):
    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    from geotaichi import DEMPM, init, polyhedron
    from src.mpdem.fluid_dynamics.IncompressibleCouplingKernel import kernel_ibm_velocity_l2_error
    from src.utils.SolverRuntime import python_callback

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    restart = args.restart_from.resolve() if args.restart_from is not None else None
    counts, spacing = grid_parameters(config)
    dt = args.dt or config["dt"]
    dem_dt = args.dem_dt or config["dem_dt"]
    simulation_time = args.time or config["time"]
    config = {**config, "time": simulation_time, "dt": dt, "dem_dt": dem_dt}
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
        cfdem_resolution="FullyResolved",
        particle_interaction=True,
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
        velocity_projection="Affine",
        solver_type="Implicit",
        discretization="FDM",
        fluid_level_set=False,
        fluid_domain_volume_fraction=0.1,
        particle_shifting=True,
        density_projection=True,
        density_projection_interior_only=True,
        solid_sdf_cut_cell=False,
        fluid_wall_no_slip=False,
        visualize=args.write_vtu,
    )
    implicit = {"linear_solver": args.linear_solver, "max_iteration_number": 200, "residual_tolerance": 1e-08}
    if args.linear_solver == "MGPCG":
        multilevel = 2
        implicit.update(multilevel=multilevel, pre_and_post_smoothing=2, bottom_smoothing=20)
    dempm.mpm.set_implicit_solver_parameters(**implicit)
    dempm.dem.set_configuration(
        scheme="LSDEM",
        boundary=["Destroy", "Destroy", "Destroy"],
        gravity=[0.0, 0.0, -9.81],
        engine="VelocityVerlet",
        search="LinkedCell",
        visualize=args.write_vtu,
    )
    dempm.set_solver(
        {
            "Timestep": dt,
            # Start with the stable contact step so the framework keeps the
            # requested outer step; the callback relaxes it after the first step.
            "DEMTimestep": dem_dt,
            "SimulationTime": simulation_time,
            "SaveInterval": args.save_interval or max(simulation_time / 20.0 - 0.1 * dt, dt),
            "SavePath": str(output),
            "CFL": 0.5,
        }
    )
    dempm.dem.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_rigid_body_number": body_count,
            "max_rigid_template_number": 1,
            "levelset_grid_number": 250000,
            "surface_node_number": 20000,
            "max_plane_number": 0,
            "body_coordination_number": 4 if body_count > 1 else 0,
            "wall_coordination_number": 0,
            "verlet_distance_multiplier": [0.15, 0.1],
            "point_coordination_number": [4, 2],
            "compaction_ratio": [0.15, 0.15],
        }
    )
    allocated_cell_counts = list(counts)
    if args.linear_solver == "MGPCG":
        multiplier = 2 ** (implicit["multilevel"] - 1)
        allocated_cell_counts = [multiplier * math.ceil(count / multiplier) for count in counts]
    max_mpm_particles = math.prod(allocated_cell_counts) * 1**3
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
    shape = polyhedron(file=str(ROOT / "assets/mesh/LSDEM/sphere.stl")).grids(space=0.05, extent=3)
    size = {"Radius": 0.5 * config["diameter"]}
    template = {"Name": "particle", "Object": shape, "WriteFile": False}
    dempm.dem.add_template(template=template)
    templates = []
    for center in config["centers"]:
        templates.append(
            {
                "Name": "particle",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": center,
                "BodyOrientation": "constant",
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixMotion": ["Free", "Free", "Free"],
                **size,
            }
        )
    if restart is None:
        dempm.dem.create_body(body={"GenerateType": "Create", "BodyType": "RigidBody", "Template": templates})
    else:
        dempm.dem.read_restart(
            file_number=args.restart_file_number,
            file_path=str(restart),
            particle=True,
            wall=False,
            ppcontact=False,
            pwcontact=False,
            is_continue=True,
        )
    dempm.dem.choose_contact_model(particle_particle_contact_model="Linear Model", particle_wall_contact_model=None)
    contact_stiffness = 2000000.0
    dempm.dem.add_property(
        materialID1=0,
        materialID2=0,
        property={
            "NormalStiffness": contact_stiffness,
            "TangentialStiffness": 1000000.0,
            "Friction": 0.0,
            "NormalViscousDamping": 0.0,
            "TangentialViscousDamping": 0.0,
        },
    )
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
    if restart is None:
        dempm.mpm.add_body(
            body={
                "Template": [
                    {
                        "RegionName": "fluid",
                        "nParticlesPerCell": 1,
                        "BodyID": 0,
                        "MaterialID": 1,
                        "InitialVelocity": [0.0, 0.0, 0.0],
                        "FixVelocity": ["Free", "Free", "Free"],
                    }
                ]
            }
        )
    else:
        dempm.mpm.read_restart(args.restart_file_number, str(restart), is_continue=True)
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
    dempm.dem.select_save_data(surface=True, grid=True, bounding=True)
    dempm.select_save_data()
    dempm.choose_contact_model(None, None)
    if restart is None:
        history_t, history_x, history_v = ([], [], [])
    else:
        dempm.read_restart(args.restart_file_number, str(restart), ppcontact=False, pwcontact=False)
        with np.load(restart / "trajectory.npz") as trajectory:
            history_t = list(np.asarray(trajectory["time"]))
            history_x = list(np.asarray(trajectory["center"]))
            history_v = list(np.asarray(trajectory["velocity"]))
    subcycled_steps = [0]
    active_dem_dt = [dem_dt]

    @python_callback
    def record_trajectory():
        positions = dempm.dem.scene.rigid.mass_center.to_numpy()[:body_count]
        velocities = dempm.dem.scene.rigid.v.to_numpy()[:body_count]
        if config["adaptive_contact_subcycling"]:
            coupled_dt = float(dempm.sims.delta)
            subcycled_steps[0] += int(active_dem_dt[0] < coupled_dt)
            next_dem_dt = adaptive_dem_timestep(config, dem_dt, coupled_dt, positions, velocities)
            dempm.sims.set_dem_timestep(next_dem_dt)
            dempm.dem.sims.set_timestep(next_dem_dt)
            active_dem_dt[0] = next_dem_dt
        if int(dempm.sims.current_step) % args.sample_every:
            return
        history_t.append(float(dempm.sims.current_time + dempm.sims.delta))
        history_x.append(positions.copy())
        history_v.append(velocities.copy())

    run_options = {"mpm_gravity_field": True, "function": record_trajectory}
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
    coupler = dempm.enginer.incompressible_coupler
    mapped_volume = float(np.sum(coupler.solid_fraction.to_numpy()) * math.prod(actual_spacing))
    expected_volume = body_count * 4.0 / 3.0 * math.pi * (0.5 * config["diameter"]) ** 3
    solid_volume_error = abs(mapped_volume - expected_volume) / expected_volume

    active_cnum = dempm.mpm.scene.element.cnum - 2 * dempm.mpm.scene.element.ghost_cell
    ibm_l2_error = float(
        kernel_ibm_velocity_l2_error(
            active_cnum, coupler.solid_fraction, coupler.solid_velocity_cell, dempm.mpm.scene.node
        )
    )
    metrics = evaluate(config, times, centers, velocities, solid_volume_error, ibm_l2_error)
    metrics.update(
        fluid_solver="semi-implicit-incompressible-fdm",
        cell_counts=actual_counts,
        element_size=actual_spacing,
        cells_per_diameter=config["diameter"] / min(actual_spacing),
        requested_timestep=dt,
        alpha_pic=args.alpha_pic,
        timestep=float(np.median(np.diff(times)) / args.sample_every),
        dem_contact_timestep=dem_dt,
        adaptive_contact_subcycling=config["adaptive_contact_subcycling"],
        subcycled_coupled_steps=subcycled_steps[0],
        restart_from=str(restart) if restart is not None else None,
        restart_file_number=args.restart_file_number if restart is not None else None,
    )
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, sort_keys=True))
    if not args.skip_postprocess:
        dempm.postprocessing(scheme="LSDEM")
    if args.strict and (not metrics["passed"]):
        raise SystemExit("drafting-kissing-tumbling validation criteria failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-per-diameter", type=float)
    parser.add_argument("--arch", default=os.environ.get("GT_IBM_VALIDATION_ARCH", "gpu"))
    parser.add_argument("--linear-solver", choices=("PCG", "MGPCG"), default="MGPCG")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "OutputData")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--dem-dt", type=float)
    parser.add_argument("--time", type=float)
    parser.add_argument("--alpha-pic", type=float, default=0.01)
    parser.add_argument("--save-interval", type=float)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--device-memory", type=float, default=8.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--write-vtu", action="store_true")
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--restart-file-number", type=int)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.sample_every <= 0:
        parser.error("--sample-every must be positive")
    if args.dem_dt is not None and args.dem_dt <= 0.0:
        parser.error("--dem-dt must be positive")
    if args.dem_dt is not None and args.dt is not None and args.dem_dt > args.dt:
        parser.error("--dem-dt cannot exceed --dt")
    if not 0.0 <= args.alpha_pic <= 1.0:
        parser.error("--alpha-pic must be in [0, 1]")
    if (args.restart_from is None) != (args.restart_file_number is None):
        parser.error("--restart-from and --restart-file-number must be supplied together")
    if args.restart_file_number is not None and args.restart_file_number < 0:
        parser.error("--restart-file-number must be non-negative")
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
