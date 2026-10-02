"""Axisymmetric MPM annulus using native explicit or Direct implicit ULMPM.

The explicit route intentionally reuses the mature native axisymmetric MPM
engine.  The implicit route uses Direct ULMPM, whose no-swirl tangent embeds
the meridional unknowns in a three-dimensional material map.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


DOMAIN = (1.0, 0.6)
BODY_ORIGIN = (0.4, 0.1)
BODY_SIZE = (0.4, 0.4)
MODE_DEFAULTS = {
    "explicit": {"steps": 2_000, "dt": 2.0e-5},
    "implicit": {"steps": 20, "dt": 1.0e-2},
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("explicit", "implicit"), default="explicit")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--cell-size", type=float, default=0.01)
    parser.add_argument("--particles-per-cell", type=int, default=2)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--output-interval", type=int)
    parser.add_argument("--radial-acceleration", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default=str(CASE_DIR / "OutputData" / "axisymmetric_annulus"),
    )
    arguments = parser.parse_args()
    defaults = MODE_DEFAULTS[arguments.solver]
    if arguments.steps is None:
        arguments.steps = defaults["steps"]
    if arguments.dt is None:
        arguments.dt = defaults["dt"]
    if arguments.output_interval is None:
        arguments.output_interval = max(1, arguments.steps // 20)
    positive = {
        "cell size": arguments.cell_size,
        "particles per cell": arguments.particles_per_cell,
        "steps": arguments.steps,
        "dt": arguments.dt,
        "output interval": arguments.output_interval,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(arguments.radial_acceleration):
        raise ValueError("radial acceleration must be finite")
    return arguments


def _particle_capacity(cell_size, particles_per_cell):
    cells = np.ceil(np.asarray(BODY_SIZE) / float(cell_size)).astype(int)
    count = int(np.prod(cells)) * int(particles_per_cell) ** 2
    return count + (count + 9) // 10


def _configure_native_explicit(gt, arguments):
    mpm = gt.MPM(log=True)
    mpm.set_configuration(
        dimension=2,
        domain=list(DOMAIN),
        axisymmetric=True,
        axis_offset=0.0,
        solver_type="Explicit",
        configuration="ULMPM",
        material_type="Solid",
        gravity=[arguments.radial_acceleration, 0.0],
        background_damping=0.1,
        alphaPIC=0.05,
        mapping="USF",
        shape_function="GIMP",
        visualize=False,
        log=True,
    )
    mpm.set_solver(
        {
            "Timestep": arguments.dt,
            "SimulationTime": arguments.steps * arguments.dt,
            "SaveInterval": arguments.output_interval * arguments.dt,
            "SavePath": arguments.output_dir,
        },
        log=True,
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": _particle_capacity(arguments.cell_size, arguments.particles_per_cell),
            "max_constraint_number": {
                "max_velocity_constraint": 100_000,
            },
        },
        log=True,
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "YoungModulus": 2.0e5,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(
        {
            "ElementType": "Q4N2D",
            "ElementSize": [arguments.cell_size] * 2,
        }
    )
    mpm.add_region(
        {
            "Name": "axisymmetric_annulus",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": list(BODY_ORIGIN),
            "BoundingBoxSize": list(BODY_SIZE),
        }
    )
    mpm.add_body(
        {
            "Template": {
                "RegionName": "axisymmetric_annulus",
                "nParticlesPerCell": arguments.particles_per_cell,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
        }
    )
    mpm.add_boundary_condition(
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, 0.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [DOMAIN[0], 0.0],
        }
    )
    mpm.select_save_data(particle=True, grid=False, object=False)
    return mpm


def _configure_direct_implicit(gt, arguments):
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    mpm = gt.MPM(log=True)
    mpm.set_configuration(
        dimension=2,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=list(DOMAIN),
        axisymmetric=True,
        axis_offset=0.0,
        gravity=[arguments.radial_acceleration, 0.0],
        background_damping=0.1,
        visualize=False,
        log=True,
    )
    body = mpm.create_body()
    body.add_rectangle(
        BODY_ORIGIN,
        tuple(np.asarray(BODY_ORIGIN) + np.asarray(BODY_SIZE)),
        arguments.cell_size,
        arguments.particles_per_cell,
        init_v=[0.0, 0.0],
        name="axisymmetric_annulus",
        grid_size=arguments.cell_size,
        xmin=[0.0, 0.0],
        xmax=list(DOMAIN),
    )
    mpm.add_body(body)
    mpm.memory_allocate(
        {"max_particle_number": _particle_capacity(arguments.cell_size, arguments.particles_per_cell)},
        log=False,
    )
    mpm.add_material(
        model="NeoHookean",
        density=1000.0,
        young_modulus=2.0e5,
        poisson_ratio=0.3,
    )
    mpm.add_element({"ElementSize": arguments.cell_size, "ShapeFunction": "Linear"})
    grid_num = np.ceil(np.asarray(DOMAIN) / arguments.cell_size).astype(int) + 1
    node = np.arange(int(np.prod(grid_num)), dtype=np.int32).reshape(int(grid_num[1]), int(grid_num[0]))
    bottom = node[0, :].reshape(-1)
    boundary = DirichletBoundary()
    boundary.append([list(2 * bottom + 1)], [0.0] * bottom.size)
    mpm.add_boundary_condition(dirichlet=boundary)
    mpm.set_solver(
        {
            "dt": arguments.dt,
            "step": arguments.steps,
            "interval": arguments.output_interval,
            "path": arguments.output_dir,
            "newmark": [1.0, 0.5, 1.0],
            "residual": 1.0e-7,
            "max_iters": 30,
            "scale": 1.0,
            "project_pd": True,
        }
    )
    return mpm


def main():
    arguments = parse_arguments()

    import geotaichi as gt

    gt.init(
        dim=2,
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        debug=False,
        log=True,
    )
    constructors = {
        "explicit": _configure_native_explicit,
        "implicit": _configure_direct_implicit,
    }
    mpm = constructors[arguments.solver](gt, arguments)
    mpm.run(verbose=False)


if __name__ == "__main__":
    main()
