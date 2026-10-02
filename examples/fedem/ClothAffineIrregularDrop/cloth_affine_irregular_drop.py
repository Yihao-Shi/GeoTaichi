#!/usr/bin/env python3
"""Drop many irregular AffineBody grains onto a square cloth fixed only at four corners."""

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
GRAIN_MESH = ROOT / "assets/mesh/AffineBody/irregular_grain.obj"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--grain-count", type=int, default=36)
    parser.add_argument("--cloth-divisions", type=int, default=24)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="Compile gate without contact-effect acceptance")
    parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData/irregular_grains_four_corner_cloth"))
    args = parser.parse_args()
    if args.grain_count < 16 or args.cloth_divisions < 8:
        parser.error("at least 16 grains and 8 cloth divisions are required")
    if args.steps <= 0 or args.dt <= 0.0 or args.save_every <= 0:
        parser.error("positive time/output controls are required")
    return args


def grain_centers(count):
    side = math.ceil(math.sqrt(count))
    axis = np.linspace(0.20, 0.80, side)
    return [
        [float(axis[index % side]), float(axis[(index // side) % side]), 0.25 + 0.10 * (index // (side * side))]
        for index in range(count)
    ]


def main():
    args = arguments()
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import geotaichi as gt

    gt.init(arch=args.arch, default_fp="float64", log=True, offline_cache=False)
    dem = gt.DEM(log=True)
    dem.set_configuration(
        domain=[1.2, 1.2, 1.2],
        scheme="AffineBody",
        search="BVH",
        gravity=[0.0, 0.0, -9.81],
        visualize=False,
        log=True,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=2.0e6,
        dhat=0.010,
        barrier_stiffness=8.0e4,
        local_damping=0.02,
        contact_damping_stiffness=0.0,
        friction_mode="lagged",
        friction_iterations=1,
        max_newton_iteration=40,
        linear_tolerance=1.0e-7,
        linear_max_iteration=4000,
        line_search_max_iteration=20,
        max_step=0.02,
        ccd=True,
        ccd_type="ccd",
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": args.grain_count,
            "surface_node_number": 12 * args.grain_count,
            "max_point_triangle_pairs": max(8192, 256 * args.grain_count),
            "max_edge_edge_pairs": max(16384, 512 * args.grain_count),
            "body_coordination_number": 20,
            "wall_coordination_number": 1,
            "compaction_ratio": [1.0, 1.0],
        },
        log=True,
    )
    dem.add_attribute(0, {"Density": 1800.0})
    dem.add_template({"Name": "grain", "TemplateType": "AffineBody", "Object": gt.polyhedron(file=str(GRAIN_MESH))})
    initial_centers = grain_centers(args.grain_count)
    rng = np.random.default_rng(20261002)
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": [
                {
                    "Name": "grain",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": point,
                    "ScaleFactor": 0.035,
                    "BodyOrientation": rng.uniform(0.0, 360.0, 3).tolist(),
                    "InitialVelocity": [0.0, 0.0, -0.10],
                    "Friction": 0.35,
                }
                for point in initial_centers
            ],
        }
    )
    dem.add_property(0, 0, {"Dhat": 0.010, "BarrierStiffness": 8.0e4, "Friction": 0.35}, dType="all")

    fem = gt.FEM(log=True)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    cloth = fem.add_mesh(
        {
            "Geometry": "Rectangle",
            "Size": (1.0, 1.0),
            "Divisions": (args.cloth_divisions, args.cloth_divisions),
            "ElementType": "TRI3",
        }
    )
    fem.add_material(
        "ClothARAP",
        density=1.0,
        stretch_stiffness=2.0e4,
        compression_stiffness=2.0e4,
        thickness=0.03,
        bending_stiffness=2.0e-2,
        bending_model="Quadratic",
    )
    sets = cloth.node_sets
    corners = sorted(
        (set(sets["xmin"]) & set(sets["ymin"]))
        | (set(sets["xmin"]) & set(sets["ymax"]))
        | (set(sets["xmax"]) & set(sets["ymin"]))
        | (set(sets["xmax"]) & set(sets["ymax"]))
    )
    if len(corners) != 4:
        raise RuntimeError(f"expected four cloth corner nodes, got {corners}")
    fem.add_boundary_condition({"type": "Dirichlet", "nodes": corners, "components": "all", "value": 0.0})

    coupling = gt.FEDEM(dem=dem, fem=fem, log=True)
    coupling.set_configuration(domain=[1.2, 1.2, 1.2], gravity=[0.0, 0.0, -9.81], search="BVH", log=True)
    coupling.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.steps * args.dt,
            "SaveInterval": args.save_every * args.dt,
            "SavePath": args.output_dir,
            "assemble_type": "HashTriplet",
            "linear_solver": "PCG",
            "project_pd": True,
            "project_bending_pd": True,
            "max_iterations": 150,
            "correction_velocity_tolerance": 1.0e-3,
            "linear_solver_tolerance": 1.0e-9,
            "linear_solver_relative_tolerance": 1.0e-7,
            "linear_solver_max_iters": 8000,
            "enable_step_retry": True,
            "step_retry_max_retries": 3,
            "step_retry_reduction": 0.5,
            "step_retry_minimum_timestep": args.dt / 8.0,
        },
        log=True,
    )
    coupling.add_surface()
    mixed_pairs = max(12288, args.grain_count * 384)
    coupling.memory_allocate(
        {
            "max_contact_pairs": mixed_pairs,
            "max_point_triangle_pairs": mixed_pairs,
            "max_edge_edge_pairs": 2 * mixed_pairs,
            "max_facet_cell_pairs": 2 * mixed_pairs,
            "contact_coordination_number": 96,
        }
    )
    coupling.choose_contact_model(
        "BarrierIPC",
        dhat=0.012,
        dmin=1.0e-3,
        kappa=4.0e4,
        friction_coefficient=0.25,
        epsv=1.0e-3,
        friction_mode="lagged",
        friction_iterations=2,
    )
    peak_contacts = [0]

    def record(engine):
        peak_contacts[0] = max(peak_contacts[0], int(engine.last_step_record["contact"]["active_contacts"]))

    result = coupling.run(verbose=False, postprocessing=[record])
    cloth_position = coupling.enginer.fem.state.position.to_numpy()
    body_vertices = coupling.enginer.affine.state.world_vertices()
    final_centers = np.asarray([vertices.mean(axis=0) for vertices in body_vertices])
    summary = {
        "case": "fully_implicit_cloth_affine_irregular_grain_drop",
        "grain_count": args.grain_count,
        "cloth_nodes": int(cloth.number_of_nodes),
        "cloth_triangles": int(cloth.number_of_cells),
        "fixed_corner_nodes": [int(node) for node in corners],
        "steps": int(result["step"]),
        "completed_time": float(result["time"]),
        "converged": bool(result["converged"]),
        "maximum_active_contacts": int(peak_contacts[0]),
        "minimum_cloth_z": float(cloth_position[:, 2].min()),
        "mean_grain_drop": float(np.mean(np.asarray(initial_centers)[:, 2] - final_centers[:, 2])),
        "finite": bool(np.isfinite(cloth_position).all() and np.isfinite(final_centers).all()),
    }
    summary["passed"] = bool(
        summary["converged"] and summary["finite"] and (args.smoke or summary["maximum_active_contacts"] > 0)
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["passed"]:
        raise RuntimeError(f"cloth--AffineBody validation failed: {summary}")


if __name__ == "__main__":
    main()
