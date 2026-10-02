#!/usr/bin/env python3
"""FEM--ABD container deposition example."""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
AFFINE_MESH = REPO_ROOT / "assets/mesh/AffineBody/lowpoly_sphere.obj"

DOMAIN = (1.0, 1.0, 1.0)
PARTICLE_RADIUS = 0.045


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--particle-count", type=int, default=18)
    parser.add_argument("--fem-divisions", type=int, default=6)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--output-interval", type=int, default=8)
    parser.add_argument("--drop-speed", type=float, default=0.80)
    parser.add_argument(
        "--output-dir",
        default=str(CASE_DIR / "OutputData"),
    )
    arguments = parser.parse_args()
    positive = {
        "particle count": arguments.particle_count,
        "FEM divisions": arguments.fem_divisions,
        "steps": arguments.steps,
        "dt": arguments.dt,
        "output interval": arguments.output_interval,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if arguments.output_interval > arguments.steps:
        parser.error("--output-interval cannot exceed --steps")
    if not math.isfinite(arguments.drop_speed) or arguments.drop_speed <= 0.0:
        parser.error("--drop-speed must be finite and positive")
    return arguments


def _deposition_centers(count):
    axes = (
        np.array([0.36, 0.50, 0.64]),
        np.array([0.36, 0.50, 0.64]),
        np.array([0.28, 0.40, 0.52, 0.64]),
    )
    centers = np.asarray(
        [[x, y, z] for z, y, x in itertools.product(axes[2], axes[1], axes[0])],
        dtype=np.float64,
    )
    if count > centers.shape[0]:
        raise ValueError(f"deposition scene supports at most {centers.shape[0]} ABD particles")
    return centers[:count]


def _add_deposition_fem(fem, divisions):
    side_divisions = max(2, divisions)
    boxes = (
        ([0.16, 0.16, 0.12], [0.68, 0.68, 0.04], [side_divisions, side_divisions, 1]),
        ([0.12, 0.16, 0.12], [0.04, 0.68, 0.72], [1, side_divisions, side_divisions]),
        ([0.84, 0.16, 0.12], [0.04, 0.68, 0.72], [1, side_divisions, side_divisions]),
        ([0.16, 0.12, 0.12], [0.68, 0.04, 0.72], [side_divisions, 1, side_divisions]),
        ([0.16, 0.84, 0.12], [0.68, 0.04, 0.72], [side_divisions, 1, side_divisions]),
    )
    for origin, size, mesh_divisions in boxes:
        fem.add_soft_particle(
            fem.create_mesh(
                "box",
                origin=origin,
                size=size,
                divisions=mesh_divisions,
                element_type="TET4",
            )
        )
    fem.add_material(
        "NeoHookean",
        density=2500.0,
        young_modulus=2.0e6,
        poisson_ratio=0.25,
    )
    all_nodes = np.arange(fem.scene.mesh.number_of_nodes, dtype=np.int32)
    fem.add_boundary_condition(
        {
            "type": "Dirichlet",
            "nodes": all_nodes,
            "components": "all",
            "value": 0.0,
        }
    )


def _affine_templates(arguments):
    return [
        {
            "Name": "affine_sphere",
            "GroupID": 0,
            "MaterialID": 0,
            "BodyPoint": center.tolist(),
            "BoundingRadius": PARTICLE_RADIUS,
            "InitialVelocity": [0.0, 0.0, -arguments.drop_speed],
            "InitialAngularVelocity": [0.0, 0.0, 0.0],
            "YoungModulus": 1.0e8,
            "Friction": 0.30,
        }
        for center in _deposition_centers(arguments.particle_count)
    ]


def main():
    arguments = parse_arguments()
    os.environ["GEOTAICHI_REAL_DTYPE"] = arguments.default_fp
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import geotaichi as gt

    gt.init(
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        log=True,
        debug=False,
        offline_cache=False,
    )

    affine_templates = _affine_templates(arguments)
    dem = gt.DEM(log=True)
    dem.set_configuration(
        domain=list(DOMAIN),
        scheme="AffineBody",
        search="BVH",
        gravity=[0.0, 0.0, -9.81],
        visualize=False,
        track_energy=False,
        log=True,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=1.0e8,
        local_damping=0.01,
        contact_damping_stiffness=0.0,
        hessian_shift=0.0,
        friction_mode="lagged",
        friction_iterations=1,
    )
    body_count = len(affine_templates)
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": body_count,
            "surface_node_number": max(256, body_count * 32),
            "max_point_triangle_pairs": max(4096, body_count * body_count * 128),
            "max_edge_edge_pairs": max(8192, body_count * body_count * 256),
            "body_coordination_number": max(32, 4 * body_count),
            "wall_coordination_number": 1,
            "compaction_ratio": [1.0, 1.0],
        },
        log=True,
    )
    dem.add_attribute(materialID=0, attribute={"Density": 2500.0})
    dem.add_template(
        {
            "Name": "affine_sphere",
            "TemplateType": "AffineBody",
            "Object": gt.polyhedron(file=str(AFFINE_MESH)),
        }
    )
    dem.create_body({"BodyType": "AffineBody", "Template": affine_templates})
    dem.add_property(
        0,
        0,
        {
            "Dhat": 0.012,
            "BarrierStiffness": 4.0e4,
            "ContactDampingStiffness": 0.0,
            "Friction": 0.30,
        },
        dType="all",
    )

    fem = gt.FEM(log=True)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    _add_deposition_fem(fem, arguments.fem_divisions)

    coupling = gt.FEDEM(dem=dem, fem=fem, log=True)
    coupling.set_configuration(
        domain=list(DOMAIN),
        search="BVH",
        gravity=[0.0, 0.0, -9.81],
        log=True,
    )
    coupling.set_solver(
        {
            "Timestep": arguments.dt,
            "SimulationTime": arguments.steps * arguments.dt,
            "SaveInterval": arguments.output_interval * arguments.dt,
            "SavePath": arguments.output_dir,
            "assemble_type": "HashTriplet",
            "linear_solver": "PCG",
            "residual_tolerance": 1.0e-7,
            "absolute_tolerance": 1.0e-10,
            "correction_velocity_tolerance": 1.0e-2,
            "max_iterations": 100,
            "linear_solver_tolerance": 1.0e-8,
            "linear_solver_relative_tolerance": 1.0e-8,
            "linear_solver_max_iters": 3000,
            "project_pd": True,
            "enable_step_retry": True,
            "step_retry_max_retries": 3,
            "step_retry_reduction": 0.5,
            "step_retry_minimum_timestep": arguments.dt / 8.0,
        },
        log=True,
    )
    coupling.add_surface()
    surface_facets = int(coupling.surface_faces.shape[0])
    pair_capacity = max(16384, body_count * max(surface_facets, 1) * 8)
    coupling.memory_allocate(
        {
            "max_contact_pairs": pair_capacity,
            "max_point_triangle_pairs": pair_capacity,
            "max_edge_edge_pairs": 2 * pair_capacity,
            "contact_coordination_number": max(64, 8 * body_count),
        }
    )
    coupling.choose_contact_model(
        "IPC",
        dhat=0.012,
        dmin=0.0,
        kappa=4.0e4,
        friction_coefficient=0.30,
        epsv=1.0e-3,
        friction_mode="lagged",
        friction_iterations=2,
    )
    for body_id in fem.scene.mesh.body_ids:
        coupling.add_ipc_property(
            AffineBody=0,
            FEMbody=int(body_id),
            property={
                "dhat": 0.012,
                "kappa": 4.0e4,
                "friction_coefficient": 0.30,
                "epsv": 1.0e-3,
            },
        )
    result = coupling.run()
    if not result["converged"]:
        raise RuntimeError("FEM-ABD deposition did not converge")
    print(
        "FEM-ABD deposition finished: "
        f"affine={body_count}, "
        f"frames={result['step']}, output={arguments.output_dir}"
    )


if __name__ == "__main__":
    main()
