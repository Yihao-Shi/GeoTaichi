#!/usr/bin/env python3
"""Impact a stack of more than one thousand AffineBody cubes with a heavy ball."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
CUBE_MESH = ROOT / "assets/mesh/AffineBody/cube.obj"
BALL_MESH = ROOT / "assets/mesh/AffineBody/lowpoly_sphere.obj"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--cube-count", type=int, default=1001)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="Compile/capacity gate without impact-effect acceptance")
    parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData/thousand_cubes_heavy_ball_impact"))
    args = parser.parse_args()
    if args.cube_count <= 1000:
        parser.error("--cube-count must be greater than 1000")
    if args.steps <= 0 or args.dt <= 0.0 or args.save_every <= 0:
        parser.error("positive time/output controls are required")
    return args


def cube_centers(count):
    nx = ny = 10
    pitch = 0.051
    points = []
    for iz in range(math.ceil(count / (nx * ny))):
        for iy in range(ny):
            for ix in range(nx):
                points.append([0.72 + pitch * ix, 0.30 + pitch * iy, 0.082 + pitch * iz])
                if len(points) == count:
                    return points
    return points


def main():
    args = arguments()
    # ponytail: cap broad-phase over-allocation; raise the env value only if the runtime reports overflow.
    os.environ.setdefault("GT_AFFINE_MAX_HASH_TRIPLETS", "1000000")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import geotaichi as gt

    gt.init(arch=args.arch, default_fp="float64", log=True, offline_cache=False)
    dem = gt.DEM(log=True)
    dem.set_configuration(
        domain=[1.6, 1.1, 1.1],
        scheme="AffineBody",
        search="LinkedCell",
        gravity=[0.0, 0.0, -9.81],
        visualize=False,
        log=True,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=5.0e6,
        dhat=0.004,
        barrier_stiffness=1.0e6,
        local_damping=0.03,
        max_newton_iteration=20,
        linear_tolerance=2.0e-5,
        linear_max_iteration=3000,
        line_search_max_iteration=16,
        max_step=0.02,
        ccd=True,
        ccd_type="ccd",
    )
    body_count = args.cube_count + 1
    dem.memory_allocate(
        {
            "max_material_number": 2,
            "max_affine_body_number": body_count,
            "surface_node_number": args.cube_count * 8 + 12,
            "max_plane_number": 3,
            "body_coordination_number": 40,
            "wall_coordination_number": 3,
            "wall_per_cell": 64,
            "max_point_triangle_pairs": 290000,
            "max_edge_edge_pairs": 720000,
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
    dem.add_attribute(0, {"Density": 900.0, "ForceLocalDamping": 0.03, "TorqueLocalDamping": 0.03})
    dem.add_attribute(1, {"Density": 30000.0, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0})
    dem.add_template({"Name": "cube", "TemplateType": "AffineBody", "Object": gt.polyhedron(file=str(CUBE_MESH))})
    dem.add_template({"Name": "heavy_ball", "TemplateType": "AffineBody", "Object": gt.polyhedron(file=str(BALL_MESH))})
    centers = cube_centers(args.cube_count)
    cubes = [
        {
            "Name": "cube",
            "GroupID": 0,
            "MaterialID": 0,
            "BodyPoint": point,
            "ScaleFactor": 0.045,
            "InitialVelocity": [0.0, 0.0, 0.0],
            "Friction": 0.40,
        }
        for point in centers
    ]
    dem.create_body({"BodyType": "AffineBody", "Template": cubes})
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": {
                "Name": "heavy_ball",
                "GroupID": 1,
                "MaterialID": 1,
                "BodyPoint": [0.25, 0.53, 0.34],
                "ScaleFactor": 0.105,
                "InitialVelocity": [3.2, 0.0, 0.0],
                "Friction": 0.25,
            },
        }
    )
    for wall_id, (point, normal) in enumerate(
        (
            ([0.0, 0.0, 0.05], [0.0, 0.0, 1.0]),
            ([0.0, 0.24, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.82, 0.0], [0.0, -1.0, 0.0]),
        )
    ):
        dem.add_wall(
            {
                "WallID": wall_id,
                "WallType": "Plane",
                "MaterialID": 0,
                "WallCenter": np.asarray(point),
                "OuterNormal": np.asarray(normal),
            }
        )
    contact = {"Dhat": 0.004, "BarrierStiffness": 1.0e6, "ContactDampingStiffness": 0.0, "Friction": 0.40}
    dem.add_property(0, 0, contact, dType="all")
    dem.add_property(0, 1, {**contact, "Friction": 0.25}, dType="all")
    dem.add_property(1, 1, {**contact, "Friction": 0.25}, dType="all")
    dem.run()

    vertices = dem.enginer.state.world_vertices()
    final_cube_centers = np.asarray([body.mean(axis=0) for body in vertices[: args.cube_count]])
    final_ball_center = vertices[-1].mean(axis=0)
    initial = np.asarray(centers)
    displaced = np.linalg.norm(final_cube_centers - initial, axis=1)
    summary = {
        "case": "affine_thousand_cubes_heavy_ball_impact",
        "cube_count": args.cube_count,
        "steps": int(dem.sims.current_step),
        "completed_time": float(dem.sims.current_time),
        "ball_x_displacement": float(final_ball_center[0] - 0.25),
        "moved_cube_count": int(np.count_nonzero(displaced > 0.01)),
        "maximum_cube_displacement": float(displaced.max()),
        "finite": bool(np.isfinite(final_cube_centers).all() and np.isfinite(final_ball_center).all()),
    }
    summary["passed"] = bool(
        summary["finite"]
        and (args.smoke or (summary["ball_x_displacement"] > 0.05 and summary["moved_cube_count"] > 10))
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["passed"]:
        raise RuntimeError(f"heavy-ball impact validation failed: {summary}")


if __name__ == "__main__":
    main()
