#!/usr/bin/env python3
"""FEM--ABD isotropic compression example."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
PLATE_MESH = REPO_ROOT / "assets/mesh/AffineBody/abd_compression_plate.obj"

DOMAIN = (1.0, 1.0, 1.0)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--fem-divisions", type=int, default=6)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--output-interval", type=int, default=8)
    parser.add_argument("--compression-speed", type=float, default=0.10)
    parser.add_argument(
        "--output-dir",
        default=str(CASE_DIR / "OutputData"),
    )
    arguments = parser.parse_args()
    positive = {
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
    if not math.isfinite(arguments.compression_speed) or arguments.compression_speed <= 0.0:
        parser.error("--compression-speed must be finite and positive")
    return arguments


def _compression_plates(speed):
    return (
        ([0.315, 0.500, 0.500], [0.0, 90.0, 0.0], [speed, 0.0, 0.0]),
        ([0.685, 0.500, 0.500], [0.0, 90.0, 0.0], [-speed, 0.0, 0.0]),
        ([0.500, 0.315, 0.500], [90.0, 0.0, 0.0], [0.0, speed, 0.0]),
        ([0.500, 0.685, 0.500], [90.0, 0.0, 0.0], [0.0, -speed, 0.0]),
        ([0.500, 0.500, 0.365], [0.0, 0.0, 0.0], [0.0, 0.0, speed]),
        ([0.500, 0.500, 0.635], [0.0, 0.0, 0.0], [0.0, 0.0, -speed]),
    )


def _add_triaxial_fem(fem, divisions):
    specimen = fem.add_soft_particle(
        fem.create_mesh(
            "box",
            origin=[0.35, 0.35, 0.40],
            size=[0.30, 0.30, 0.20],
            divisions=[divisions, divisions, max(2, 2 * divisions // 3)],
            element_type="TET4",
        )
    )
    fem.add_material(
        "NeoHookean",
        density=1200.0,
        young_modulus=2.0e4,
        poisson_ratio=0.30,
    )
    return specimen


def _affine_templates(arguments):
    return [
        {
            "Name": "abd_plate",
            "GroupID": 0,
            "MaterialID": 0,
            "BodyPoint": center,
            "ScaleFactor": 1.0,
            "BodyOrientation": orientation,
            "InitialVelocity": velocity,
            "InitialAngularVelocity": [0.0, 0.0, 0.0],
            "YoungModulus": 1.0e9,
            "Friction": 0.25,
        }
        for center, orientation, velocity in _compression_plates(arguments.compression_speed)
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
        gravity=[0.0, 0.0, 0.0],
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
            "Name": "abd_plate",
            "TemplateType": "AffineBody",
            "Object": gt.polyhedron(file=str(PLATE_MESH)),
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
    _add_triaxial_fem(fem, arguments.fem_divisions)

    coupling = gt.FEDEM(dem=dem, fem=fem, log=True)
    coupling.set_configuration(
        domain=list(DOMAIN),
        search="BVH",
        gravity=[0.0, 0.0, 0.0],
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
    for affine_body in range(body_count):
        for body_id in fem.scene.mesh.body_ids:
            coupling.add_ipc_property(
                AffineBody=affine_body,
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
        raise RuntimeError("FEM-ABD isotropic compression did not converge")
    print(
        "FEM-ABD isotropic compression finished: "
        f"affine={body_count}, "
        f"frames={result['step']}, output={arguments.output_dir}"
    )


if __name__ == "__main__":
    main()
