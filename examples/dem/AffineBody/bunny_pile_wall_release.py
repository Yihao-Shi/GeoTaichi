#!/usr/bin/env python3
"""Settle a pile of AffineBody bunnies, then remove its right wall."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
MESH = ROOT / "assets/bunny_sparse.obj"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--release-time", type=float, default=0.40)
    parser.add_argument("--barrier-stiffness", type=float, default=8.0e8)
    parser.add_argument("--smoke", action="store_true", help="Compile gate without collapse-effect acceptance")
    parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData/bunny_pile_wall_release"))
    args = parser.parse_args()
    if args.count < 4 or args.steps <= 0 or args.dt <= 0.0 or args.save_every <= 0 or args.barrier_stiffness <= 0.0:
        parser.error("count >= 4 and positive time/output controls are required")
    if not 0.0 < args.release_time < args.steps * args.dt:
        parser.error("release time must lie inside the simulation")
    return args


def centers(count):
    result = []
    for layer in range(math.ceil(count / 4)):
        for iy in range(2):
            for ix in range(2):
                result.append([0.48 + 0.20 * ix, 0.38 + 0.24 * iy, 0.18 + 0.18 * layer])
                if len(result) == count:
                    return result
    return result


def main():
    args = arguments()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import geotaichi as gt
    from src.utils.SolverRuntime import python_callback

    gt.init(arch=args.arch, default_fp="float64", log=True, offline_cache=False)
    dem = gt.DEM(log=True)
    dem.set_configuration(
        domain=[1.4, 1.0, 1.4],
        scheme="AffineBody",
        search="BVH",
        gravity=[0.0, 0.0, -9.81],
        visualize=False,
        log=True,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=3.0e6,
        dhat=0.006,
        barrier_stiffness=args.barrier_stiffness,
        local_damping=0.08,
        max_newton_iteration=25,
        linear_tolerance=1.0e-6,
        linear_max_iteration=3000,
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
            "max_point_triangle_pairs": 80000,
            "max_edge_edge_pairs": 180000,
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
            "step_retry_max_retries": 3,
            "step_retry_reduction": 0.5,
            "step_retry_minimum_timestep": args.dt / 8.0,
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
        ([0.0, 0.0, 0.06], [0.0, 0.0, 1.0]),
        ([0.30, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.86, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([0.0, 0.20, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.80, 0.0], [0.0, -1.0, 0.0]),
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

    @python_callback
    def remove_right_wall():
        if not released[0] and dem.sims.current_time >= args.release_time:
            dem.enginer.translate_wall(2, [2.0, 0.0, 0.0])
            dem.update_wall_status(2, "Position", [2.86, 0.0, 0.0])
            released[0] = True

    dem.run(function=remove_right_wall)
    final_vertices = dem.enginer.state.world_vertices()
    final_centers = np.asarray([vertices.mean(axis=0) for vertices in final_vertices])
    all_final_vertices = np.concatenate(final_vertices, axis=0)
    retained_wall_gaps = []
    for wall_id in (0, 1, 3, 4):
        point, normal = walls[wall_id]
        retained_wall_gaps.append(np.min((all_final_vertices - np.asarray(point)) @ np.asarray(normal)))
    minimum_retained_wall_gap = float(min(retained_wall_gaps))
    initial = np.asarray(initial_centers)
    summary = {
        "case": "affine_bunny_pile_wall_release",
        "bunny_count": args.count,
        "steps": int(dem.sims.current_step),
        "completed_time": float(dem.sims.current_time),
        "barrier_stiffness": args.barrier_stiffness,
        "wall_released": bool(released[0]),
        "minimum_retained_wall_gap": minimum_retained_wall_gap,
        "maximum_horizontal_displacement": float(np.max(np.linalg.norm(final_centers[:, :2] - initial[:, :2], axis=1))),
        "finite": bool(np.isfinite(final_centers).all()),
        "passed": bool(
            released[0]
            and np.isfinite(final_centers).all()
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
