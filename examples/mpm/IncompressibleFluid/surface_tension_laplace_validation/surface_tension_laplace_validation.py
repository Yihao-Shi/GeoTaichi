import argparse
import math
import subprocess
import os
import sys

from pathlib import Path

import numpy as np
import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.mpm.IncompressibleFluid.surface_tension_laplace_validation.draw.evaluate_surface_tension_laplace_validation import (
    validate_laplace,
)


from geotaichi import *  # noqa: E402,F403


@ti.kernel
def kernel_set_analytic_droplet_level_set_2d(
    center: ti.types.vector(2, float), radius: float, grid_size: ti.types.vector(2, float), fluid_sdf: ti.template()
):
    for I in ti.grouped(fluid_sdf):
        position = (I.cast(float) + 0.5) * grid_size
        fluid_sdf[I] = (position - center).norm() - radius


@ti.kernel
def kernel_set_analytic_droplet_level_set_3d(
    center: ti.types.vector(3, float), radius: float, grid_size: ti.types.vector(3, float), fluid_sdf: ti.template()
):
    for I in ti.grouped(fluid_sdf):
        position = (I.cast(float) + 0.5) * grid_size
        fluid_sdf[I] = (position - center).norm() - radius


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an MPM surface-tension droplet example and validate Laplace pressure."
    )
    parser.add_argument("--dim", choices=("2", "3", "both"), default="both")
    parser.add_argument("--sigma", type=float, default=0.0728)
    parser.add_argument("--radius", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--no-post", action="store_true")
    return parser.parse_args()


def run_child_examples(args):
    base_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--sigma",
        str(args.sigma),
        "--radius",
        str(args.radius),
        "--tolerance",
        str(args.tolerance),
        "--cpu-threads",
        str(args.cpu_threads),
    ]
    if args.no_post:
        base_cmd.append("--no-post")

    status = 0
    for dim in ("2", "3"):
        result = subprocess.run(base_cmd + ["--dim", dim], check=False)
        status = max(status, result.returncode)
    raise SystemExit(status)


def make_droplet_region(dim, center, radius):
    center_array = np.array(center, dtype=float)
    center_values = tuple(float(value) for value in center_array)

    def droplet_volume():
        if dim == 2:
            return math.pi * radius * radius
        return 4.0 / 3.0 * math.pi * radius**3

    def droplet_function(position, particle_radius=0.0):
        local_radius = radius - particle_radius
        distance2 = 0.0
        for d in ti.static(range(dim)):
            distance2 += (position[d] - center_values[d]) ** 2
        return distance2 <= local_radius * local_radius

    lower = center_array - radius
    size = np.full(dim, 2.0 * radius)
    return {
        "Name": "droplet",
        "Type": "UserDefined",
        "BoundingBoxPoint": lower.tolist(),
        "BoundingBoxSize": size.tolist(),
        "RegionVolume": droplet_volume,
        "RegionFunction": droplet_function,
    }


def run_example(args):
    dim = int(args.dim)
    if dim == 2:
        domain = [0.4, 0.4]
        center = [0.2, 0.2]
        radius = args.radius if args.radius > 0.0 else 0.09
        element_size = [0.01, 0.01]
        max_particle_number = 20000
        particles_per_cell = 3
        multilevel = 3
    else:
        domain = [0.3, 0.3, 0.3]
        center = [0.15, 0.15, 0.15]
        radius = args.radius if args.radius > 0.0 else 0.07
        element_size = [0.015, 0.015, 0.015]
        max_particle_number = 30000
        particles_per_cell = 2
        multilevel = 3

    init(
        dim=dim, arch="cpu", cpu_max_num_threads=args.cpu_threads, device_memory_GB=2, debug=False, kernel_profiler=True
    )

    mpm = MPM()
    mpm.set_configuration(
        domain=domain,
        background_damping=0.0,
        alphaPIC=1.0,
        mapping="USL",
        shape_function="QuadBSpline",
        gravity=[0.0 for _ in range(dim)],
        material_type="Fluid",
        velocity_projection="PIC",
        solver_type="Implicit",
        discretization="FDM",
        fluid_level_set=True,
        visualize=False,
    )

    mpm.set_implicit_solver_parameters(
        linear_solver="MGPCG", multilevel=multilevel, pre_and_post_smoothing=2, bottom_smoothing=20
    )

    save_path = f"surface_tension_laplace_{dim}d"
    mpm.set_solver(
        {
            "Timestep": 2.0e-4,
            "SimulationTime": 4.0e-4,
            "SaveInterval": 2.0e-4,
            "SavePath": save_path,
        }
    )

    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": max_particle_number,
            "verlet_distance_multiplier": 1.0,
            "max_constraint_number": {},
        }
    )

    mpm.add_material(
        model="Newtonian",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "Modulus": 2.0e6,
            "Viscosity": 1.0e-3,
            "ElementLength": min(element_size),
            "cL": 1.5,
            "cQ": 2.0,
            "atmospheric_pressure": 0.0,
            "SurfaceTension": args.sigma,
        },
    )

    mpm.add_element(
        element={
            "ElementType": "Staggered",
            "ElementSize": element_size,
            "GhostCell": 2,
        }
    )

    mpm.add_region(region=[make_droplet_region(dim, center, radius)])
    mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "droplet",
                    "nParticlesPerCell": particles_per_cell,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                }
            ]
        }
    )

    center_vector = ti.Vector(center)

    def apply_analytic_fluid_level_set(scene):
        if dim == 2:
            kernel_set_analytic_droplet_level_set_2d(
                center_vector, radius, scene.element.grid_size, scene.element.cell.fluid_sdf
            )
        else:
            kernel_set_analytic_droplet_level_set_3d(
                center_vector, radius, scene.element.grid_size, scene.element.cell.fluid_sdf
            )

    mpm.select_save_data(particle=True, grid=True)
    mpm.run(gravity_field=False, fluid_level_set_function=apply_analytic_fluid_level_set)

    passed = validate_laplace(mpm, dim, center, radius, args.sigma, args.tolerance)
    if not args.no_post:
        mpm.postprocessing()
    return 0 if passed else 1


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.dim == "both":
        run_child_examples(parsed_args)
    raise SystemExit(run_example(parsed_args))
