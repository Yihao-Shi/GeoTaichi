#!/usr/bin/env python3
"""MPM--ABD mixed-particle container deposition example."""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = Path(__file__).resolve().parent
SOFT_MESH = REPO_ROOT / "assets/mesh/LSDEM/sphere.stl"
AFFINE_MESH = REPO_ROOT / "assets/mesh/AffineBody/lowpoly_sphere.obj"
CONTAINER_PLATE_MESH = REPO_ROOT / "assets/mesh/AffineBody/abd_plate.obj"

DOMAIN = (1.0, 1.0, 1.0)
PARTICLE_RADIUS = 0.045
PARTICLE_MATERIAL = 0
BOUNDARY_MATERIAL = 1


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--soft-count", type=int, default=8)
    parser.add_argument("--affine-count", type=int, default=8)
    parser.add_argument("--particles-per-cell", type=int, default=1)
    parser.add_argument("--mixed-contact-capacity", type=int)
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
        "soft count": arguments.soft_count,
        "affine count": arguments.affine_count,
        "particles per cell": arguments.particles_per_cell,
        "steps": arguments.steps,
        "dt": arguments.dt,
        "output interval": arguments.output_interval,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if arguments.output_interval > arguments.steps:
        parser.error("--output-interval cannot exceed --steps")
    if arguments.mixed_contact_capacity is not None and arguments.mixed_contact_capacity <= 0:
        parser.error("--mixed-contact-capacity must be positive")
    if not math.isfinite(arguments.drop_speed) or arguments.drop_speed <= 0.0:
        parser.error("--drop-speed must be finite and positive")
    return arguments


def _packing_centers(count):
    axes = (
        np.array([0.36, 0.50, 0.64]),
        np.array([0.36, 0.50, 0.64]),
        np.array([0.30, 0.43, 0.56, 0.69]),
    )
    centers = np.asarray(
        [[x, y, z] for z, y, x in itertools.product(axes[2], axes[1], axes[0])],
        dtype=np.float64,
    )
    if count > centers.shape[0]:
        raise ValueError(f"deposition supports at most {centers.shape[0]} mixed particles")
    return centers[:count]


def _plate_specs():
    return (
        ([0.500, 0.500, 0.140], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.140, 0.500, 0.500], [0.0, 90.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.860, 0.500, 0.500], [0.0, 90.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.500, 0.140, 0.500], [90.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.500, 0.860, 0.500], [90.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    )


def _particle_templates(arguments):
    total = arguments.soft_count + arguments.affine_count
    centers = _packing_centers(total)
    particle_velocity = [0.0, 0.0, -arguments.drop_speed]
    soft = []
    affine = []
    for index, center in enumerate(centers):
        entry = {
            "GroupID": 0,
            "MaterialID": PARTICLE_MATERIAL,
            "BodyPoint": center.tolist(),
            "BoundingRadius": PARTICLE_RADIUS,
            "InitialVelocity": particle_velocity,
            "InitialAngularVelocity": [0.0, 0.0, 0.0],
            "Friction": 0.30,
        }
        if index % 2 == 0 and len(soft) < arguments.soft_count:
            soft.append(
                {
                    **entry,
                    "Name": "soft_sphere",
                    "MaterialPointsPerCell": arguments.particles_per_cell,
                    "FixMotion": ["Free", "Free", "Free"],
                }
            )
        else:
            affine.append({**entry, "Name": "affine_sphere"})
    while len(soft) < arguments.soft_count:
        entry = affine.pop(0)
        entry["Name"] = "soft_sphere"
        entry["MaterialPointsPerCell"] = arguments.particles_per_cell
        entry["FixMotion"] = ["Free", "Free", "Free"]
        soft.append(entry)
    return soft, affine


def main():
    arguments = parse_arguments()
    os.environ["GEOTAICHI_REAL_DTYPE"] = arguments.default_fp
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from geotaichi import MPDEM, init, polyhedron

    init(
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        log=True,
        debug=False,
        offline_cache=False,
    )

    mpdem = MPDEM(log=True)
    mpdem.set_configuration(
        domain=list(DOMAIN),
        coupling_scheme="MPDEM",
        particle_interaction=True,
        wall_interaction=False,
        gravity=[0.0, 0.0, 0.0],
        search="LinkedCell",
        visualize=True,
        track_energy=False,
        log=True,
    )

    dem = mpdem.dem
    dem.set_configuration(
        domain=list(DOMAIN),
        boundary=["Destroy", "Destroy", "Destroy"],
        gravity=[0.0, 0.0, 0.0],
        search="LinkedCell",
        scheme="LSMPM",
        soft_rigid_contact="IPC",
        shape_function="QuadBSpline",
        visualize=True,
        track_energy=False,
        log=True,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        dhat=0.012,
        barrier_stiffness=4.0e5,
        contact_damping_stiffness=0.0,
        friction_epsv=1.0e-4,
        soft_background_damping=0.02,
        max_newton_iteration=30,
        linear_tolerance=1.0e-6,
        linear_max_iteration=2000,
        line_search_max_iteration=16,
        max_step=0.02,
        ccd=True,
        ccd_type="ccd",
        ccd_eta=0.2,
        ccd_max_iteration=10000,
    )

    plate_count = len(_plate_specs())
    total_affine = arguments.affine_count + plate_count
    soft_object = polyhedron(file=str(SOFT_MESH)).grids(space=0.20, extent=1)
    affine_object = polyhedron(file=str(AFFINE_MESH))
    plate_object = polyhedron(file=str(CONTAINER_PLATE_MESH))
    dem.add_template({"Name": "soft_sphere", "Object": soft_object})
    dem.add_template(
        {
            "Name": "affine_sphere",
            "TemplateType": "AffineBody",
            "Object": affine_object,
        }
    )
    dem.add_template(
        {
            "Name": "abd_plate",
            "TemplateType": "AffineBody",
            "Object": plate_object,
        }
    )
    soft_preprocess = dem.preprocess_soft_grid_template(
        "soft_sphere",
        points_per_cell=arguments.particles_per_cell,
    )
    soft_point_capacity = arguments.soft_count * soft_preprocess["material_point_number"]
    mixed_contact_capacity = arguments.mixed_contact_capacity or soft_point_capacity
    dem.set_affine_body_parameters(
        mixed_contact_capacity=mixed_contact_capacity,
        mixed_friction_capacity=mixed_contact_capacity,
    )
    surface_node_capacity = (
        arguments.soft_count * soft_object.mesh.vertices.shape[0]
        + arguments.affine_count * affine_object.mesh.vertices.shape[0]
        + plate_count * plate_object.mesh.vertices.shape[0]
    )
    dem.memory_allocate(
        memory={
            "max_material_number": 2,
            "max_rigid_body_number": 0,
            "max_soft_body_number": arguments.soft_count,
            "max_material_point_number": soft_point_capacity,
            "max_rigid_template_number": 1,
            "levelset_grid_number": (arguments.soft_count * soft_preprocess["levelset_grid_number"]),
            "soft_grid_number": (arguments.soft_count * soft_preprocess["soft_grid_number"]),
            "surface_node_number": surface_node_capacity,
            "body_coordination_number": 64,
            "wall_coordination_number": 0,
            "verlet_distance_multiplier": [0.15, 0.15],
            "point_coordination_number": [24, 12],
            "compaction_ratio": [1.0, 1.0],
            "max_point_triangle_pairs": max(4096, total_affine * total_affine * 128),
            "max_edge_edge_pairs": max(8192, total_affine * total_affine * 256),
        },
        log=True,
    )
    mpdem.set_solver(
        {
            "Timestep": arguments.dt,
            "SimulationTime": arguments.steps * arguments.dt,
            "SaveInterval": arguments.output_interval * arguments.dt,
            "SavePath": arguments.output_dir,
            "enable_step_retry": True,
            "step_retry_max_retries": 3,
            "step_retry_reduction": 0.5,
            "step_retry_minimum_timestep": arguments.dt / 8.0,
        },
        log=True,
    )

    dem.add_attribute(
        materialID=PARTICLE_MATERIAL,
        attribute={
            "Density": 1200.0,
            "ConstitutiveModel": "NeoHookean",
            "YoungModulus": 1.0e5,
            "PoissonRatio": 0.30,
            "ForceLocalDamping": 0.02,
            "TorqueLocalDamping": 0.02,
        },
    )
    dem.add_attribute(
        materialID=BOUNDARY_MATERIAL,
        attribute={
            "Density": 2.0e5,
            "ConstitutiveModel": "NeoHookean",
            "YoungModulus": 1.0e9,
            "PoissonRatio": 0.25,
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    soft, affine = _particle_templates(arguments)
    dem.create_body({"BodyType": "SoftBody", "Template": soft})
    affine.extend(
        {
            "Name": "abd_plate",
            "GroupID": 1,
            "MaterialID": BOUNDARY_MATERIAL,
            "BodyPoint": center,
            "ScaleFactor": 1.0,
            "BodyOrientation": orientation,
            "InitialVelocity": velocity,
            "InitialAngularVelocity": [0.0, 0.0, 0.0],
            "Friction": 0.30,
            "YoungModulus": 1.0e9,
        }
        for center, orientation, velocity in _plate_specs()
    )
    dem.create_body({"BodyType": "AffineBody", "Template": affine})

    contact = {
        "Dhat": 0.012,
        "BarrierStiffness": 4.0e5,
        "ContactDampingStiffness": 0.0,
        "Friction": 0.30,
    }
    dem.add_property(PARTICLE_MATERIAL, PARTICLE_MATERIAL, contact, dType="all")
    dem.add_property(PARTICLE_MATERIAL, BOUNDARY_MATERIAL, contact, dType="all")
    dem.add_property(BOUNDARY_MATERIAL, BOUNDARY_MATERIAL, contact, dType="all")
    dem.select_save_data(
        particle=True,
        surface=True,
        bounding=True,
        wall=False,
    )
    mpdem.run()

    print(
        "MPM-ABD deposition finished: "
        f"soft={int(dem.scene.softNum[0])}, "
        f"affine={len(dem.scene.affine_bodies)}, output={arguments.output_dir}"
    )


if __name__ == "__main__":
    main()
