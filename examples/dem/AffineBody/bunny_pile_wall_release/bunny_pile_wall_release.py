#!/usr/bin/env python3
"""Settle a pile of AffineBody bunnies, then remove its right wall."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = Path(__file__).resolve().parents[1]
# The sparse OBJ has open holes; use its existing capped, same-vertex surface.
MESH = ROOT / "assets/mesh/LSDEM/bunny_impact_watertight.ply"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--release-time", type=float, default=2.0)
    parser.add_argument("--barrier-stiffness", type=float, default=8.0e12)
    parser.add_argument(
        "--settle-speed", type=float, default=0.15, help="Maximum surface speed before wall release (m/s)"
    )
    parser.add_argument("--smoke", action="store_true", help="Compile gate without collapse-effect acceptance")
    parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData/bunny_pile_wall_release"))
    args = parser.parse_args()
    if (
        args.count < 4
        or args.steps <= 0
        or args.save_every <= 0
        or not all(
            math.isfinite(value) and value > 0.0 for value in (args.dt, args.barrier_stiffness, args.settle_speed)
        )
    ):
        parser.error("count >= 4 and positive time/output controls are required")
    if (
        not math.isfinite(args.release_time)
        or args.release_time <= 0
        or (not args.smoke and args.release_time >= args.steps * args.dt)
    ):
        parser.error("release time must lie inside the simulation")
    return args


# Enclose every rotated original surface in a circumsphere about its mass centre.
BUNNY_RADIUS = 0.125758006
PITCH = 2.0 * BUNNY_RADIUS + 0.012


def centers(count):
    side = math.ceil(count ** (1.0 / 3.0) - 1e-12)
    lower = np.asarray([0.30, 0.20, 0.06])
    return [
        (lower + PITCH * (np.asarray([i % side, (i // side) % side, i // side**2]) + 0.5)).tolist()
        for i in range(count)
    ]


def container_bounds(count):
    points = np.asarray(centers(count))
    lower = np.asarray([0.30, 0.20, 0.06])
    return np.stack((lower, points.max(axis=0) + PITCH / 2.0))


def main():
    args = arguments()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    os.environ.setdefault("GT_AFFINE_MAX_HASH_TRIPLETS", "8000000")
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    import geotaichi as gt
    from src.utils.SolverRuntime import python_callback

    gt.init(arch=args.arch, default_fp="float64", device_memory_GB=1.0, log=True, offline_cache=False)
    bounds = container_bounds(args.count)
    lower, upper = bounds
    release_distance = float(upper[0] - lower[0] + 1.0)
    dem = gt.DEM(log=True)
    dem.set_configuration(
        domain=[float(upper[0] + release_distance + 1.0), float(upper[1] + 1.0), float(upper[2] + 1.0)],
        scheme="AffineBody",
        search="BVH",
        gravity=[0.0, 0.0, -9.81],
        visualize=False,
        log=True,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=2.0e12,
        dhat=0.006,
        barrier_stiffness=args.barrier_stiffness,
        local_damping=0.08,
        max_newton_iteration=1000,
        newton_tolerance=1.0e-5,
        # The mass bound limits linear surface-velocity error to 9.9e-7 m/s at the minimum dt.
        linear_tolerance=1.0e-12,
        linear_max_iteration=12000,
        line_search_max_iteration=20,
        max_step=0.025,
        ccd=True,
        ccd_type="ccd",
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": args.count,
            "surface_node_number": args.count * 2503,
            "max_plane_number": 5,
            "body_coordination_number": 12,
            "wall_coordination_number": 5,
            "max_point_triangle_pairs": max(80000, args.count * 16000),
            "max_edge_edge_pairs": max(180000, args.count * 32000),
            "compaction_ratio": [1.0, 1.0],
        },
        log=True,
    )
    dem.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.steps * args.dt,
            "SaveInterval": args.save_every * args.dt,
            "SavePath": args.output_dir,
            "enable_step_retry": True,
            "step_retry_max_retries": 2,
            "step_retry_reduction": 0.5,
            "step_retry_minimum_timestep": args.dt / 4.0,
        },
        log=True,
    )
    dem.add_attribute(0, {"Density": 1150.0, "ForceLocalDamping": 0.08, "TorqueLocalDamping": 0.08})
    dem.add_template({"Name": "bunny", "TemplateType": "AffineBody", "Object": gt.polyhedron(file=str(MESH))})
    templates = []
    initial_centers = centers(args.count)
    for body_id, point in enumerate(initial_centers):
        templates.append(
            {
                "Name": "bunny",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": point,
                "ScaleFactor": 1.0,
                "BodyOrientation": [0.0, 17.0 * (body_id % 4), 31.0 * body_id],
                "InitialVelocity": [0.0, 0.0, 0.0],
                "Friction": 0.45,
            }
        )
    dem.create_body({"BodyType": "AffineBody", "Template": templates})
    walls = (
        ([0.0, 0.0, lower[2]], [0.0, 0.0, 1.0]),
        ([lower[0], 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([upper[0], 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([0.0, lower[1], 0.0], [0.0, 1.0, 0.0]),
        ([0.0, upper[1], 0.0], [0.0, -1.0, 0.0]),
    )
    for wall_id, (point, normal) in enumerate(walls):
        dem.add_wall(
            {
                "WallID": wall_id,
                "WallType": "Plane",
                "MaterialID": 0,
                "WallCenter": np.asarray(point),
                "OuterNormal": np.asarray(normal),
            }
        )
    dem.add_property(
        0,
        0,
        {
            "Dhat": 0.006,
            "BarrierStiffness": args.barrier_stiffness,
            "ContactDampingStiffness": 0.0,
            "Friction": 0.45,
        },
        dType="all",
    )

    released = [False]
    release_state = {}

    @python_callback
    def remove_right_wall():
        if not released[0] and dem.sims.current_time >= args.release_time:
            speed = float(dem.enginer.operator.device_maximum_surface_speed())
            if not math.isfinite(speed) or speed > args.settle_speed:
                return
            dem.enginer.translate_wall(2, [release_distance, 0.0, 0.0])
            dem.update_wall_status(2, "Position", [float(upper[0] + release_distance), 0.0, 0.0])
            released[0] = True
            release_state.update(time=float(dem.sims.current_time), maximum_surface_speed=speed)

    try:
        dem.run(function=remove_right_wall)
    except RuntimeError:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "solver_failure.json").write_text(json.dumps(dem.diagnostics_snapshot(), indent=2) + "\n")
        raise
    final_vertices = dem.enginer.state.world_vertices()
    final_centers = np.asarray([vertices.mean(axis=0) for vertices in final_vertices])
    all_final_vertices = np.concatenate(final_vertices, axis=0)
    retained_wall_gaps = []
    for wall_id in (0, 1, 3, 4):
        point, normal = walls[wall_id]
        retained_wall_gaps.append(np.min((all_final_vertices - np.asarray(point)) @ np.asarray(normal)))
    minimum_retained_wall_gap = float(min(retained_wall_gaps))
    initial = np.asarray(initial_centers)
    accepted_steps = dem.enginer.history
    accepted_dt = [row["step_retry"]["accepted_timestep"] for row in accepted_steps]
    retry_count = sum(row["step_retry"]["retry_count"] for row in accepted_steps)
    start_time = accepted_steps[0]["time"] - accepted_dt[0]
    fixed_timestep = bool(
        len(accepted_steps) == args.steps and retry_count == 0 and all(abs(dt - args.dt) <= 1e-12 for dt in accepted_dt)
    )
    timestep_policy = bool(
        all(0.0 < dt <= args.dt + 1e-12 for dt in accepted_dt)
        and abs(sum(accepted_dt) - (dem.sims.current_time - start_time)) < 1e-8
    )
    summary = {
        "case": "affine_bunny_pile_wall_release",
        "bunny_count": args.count,
        "container_bounds": bounds.tolist(),
        "initial_centers": initial.tolist(),
        "mesh_vertices_per_body": 2503,
        "mesh_triangles_per_body": 5002,
        "steps": int(dem.sims.current_step),
        "nominal_timestep": args.dt,
        "minimum_accepted_timestep": float(min(accepted_dt)),
        "maximum_accepted_timestep": float(max(accepted_dt)),
        "retry_count": retry_count,
        "fixed_timestep_verified": fixed_timestep,
        "timestep_policy_verified": timestep_policy,
        "start_time": start_time,
        "completed_time": float(dem.sims.current_time),
        "barrier_stiffness": args.barrier_stiffness,
        "wall_released": bool(released[0]),
        "release_state": release_state,
        "settle_speed_limit": args.settle_speed,
        "maximum_hash_triplets": dem.enginer.maximum_hash_triplets,
        "minimum_retained_wall_gap": minimum_retained_wall_gap,
        "maximum_horizontal_displacement": float(np.max(np.linalg.norm(final_centers[:, :2] - initial[:, :2], axis=1))),
        "finite": bool(np.isfinite(final_centers).all()),
        "passed": bool(
            timestep_policy
            and (args.smoke or released[0])
            and abs(dem.sims.current_time - args.steps * args.dt) < 1.0e-8
            and np.isfinite(all_final_vertices).all()
            and minimum_retained_wall_gap > 1.0e-8
            and (args.smoke or np.max(np.linalg.norm(final_centers[:, :2] - initial[:, :2], axis=1)) > 0.05)
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["passed"]:
        raise RuntimeError(f"bunny wall-release validation failed: {summary}")


if __name__ == "__main__":
    main()
