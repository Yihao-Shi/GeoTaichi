"""Explicit IGA--MPM point--NURBS contact with the shared DEM linear law."""

import argparse
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from geotaichi import IGA, IGAMPM, MPM, init
from src.iga import Cube, DirichletBoundary, Primitives


def _default_arch():
    if platform.system() == "Darwin" or os.path.exists("/dev/nvidia0"):
        return "gpu"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default=_default_arch())
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--dt", type=float, default=5.0e-5)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--output-interval", type=int, default=120)
    parser.add_argument("--element-size", type=float, default=0.05)
    parser.add_argument("--particles-per-cell", type=int, default=2)
    parser.add_argument("--iga-refinement", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default=str(CASE_DIR / "OutputData"),
    )
    arguments = parser.parse_args()
    if any(
        value <= 0
        for value in (
            arguments.dt,
            arguments.steps,
            arguments.output_interval,
            arguments.element_size,
            arguments.particles_per_cell,
            arguments.iga_refinement,
        )
    ):
        raise ValueError("time, resolution, particle, and output arguments must be positive")

    block_origin = [0.35, 0.35, 0.04]
    block_size = [0.30, 0.30, 0.10]
    particle_volume = arguments.element_size**3 / arguments.particles_per_cell**3
    max_particles = math.ceil(math.prod(block_size) / particle_volume)

    init(
        dim=3,
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        debug=False,
        log=True,
    )

    iga = IGA(log=True)
    mpm = MPM(log=True)
    # Construct before MPM memory allocation so ParticleCoupling and the
    # Lagrangian transfer path are selected.
    coupling = IGAMPM(
        iga=iga,
        mpm=mpm,
        contact_model="Linear",
        log=True,
    )

    mpm.set_configuration(
        domain=[2.0, 2.0, 2.0],
        gravity=[0.0, 0.0, 0.0],
        alphaPIC=0.0,
        mapping="USL",
        shape_function="Linear",
        configuration="ULMPM",
        solver_type="Explicit",
        material_type="Solid",
        visualize=True,
        log=True,
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": max_particles,
            "max_constraint_number": {},
        },
        log=True,
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "YoungModulus": 1.0e4,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(
        element={
            "ElementType": "R8N3D",
            "ElementSize": [arguments.element_size] * 3,
        }
    )
    mpm.add_region(
        region={
            "Name": "contact_point",
            "Type": "Rectangle",
            "BoundingBoxPoint": block_origin,
            "BoundingBoxSize": block_size,
        }
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "contact_point",
                "nParticlesPerCell": arguments.particles_per_cell,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.25, 0.0, 1.0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        }
    )
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.set_solver(
        {
            "Timestep": arguments.dt,
            "SimulationTime": arguments.steps * arguments.dt,
            "SaveInterval": arguments.output_interval * arguments.dt,
            "SavePath": arguments.output_dir,
        },
        log=True,
    )

    iga.set_configuration(dimension=3, solver_type="Explicit")
    cube = Cube()
    cube.set_parameters(
        start_point=[0.0, 0.0, 0.2],
        size=[1.0, 1.0, 0.2],
    )
    cube.generate_knot_u(degree=2, num_ctrlpts=6 * arguments.iga_refinement + 1)
    cube.generate_knot_v(degree=2, num_ctrlpts=6 * arguments.iga_refinement + 1)
    cube.generate_knot_w(degree=2, num_ctrlpts=arguments.iga_refinement + 2)
    cube.generate_ctrlpts()
    cube.generate_weights()
    fixed = np.flatnonzero(np.isclose(cube.control_points[:, 2], 0.4))
    dirichlet = DirichletBoundary()
    dirichlet.append(
        [list(3 * fixed + direction) for direction in range(3)],
        [0.0] * (3 * len(fixed)),
    )
    primitives = Primitives()
    primitives.append(cube, "contact_block")
    primitives.finialize()
    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=dirichlet)
    iga.add_material(
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0, 0.0],
    )
    iga.add_element([2, 2, 2])
    iga.set_solver(
        dt=arguments.dt,
        step=arguments.steps,
        interval=arguments.output_interval,
        path=arguments.output_dir,
    )

    coupling.set_configuration(dimension=3, contact_model="Linear")
    coupling.add_property(
        MPMmaterial=1,
        IGAbody=0,
        property={
            "NormalStiffness": 1.0e5,
            "TangentialStiffness": 5.0e4,
            "StaticFriction": 0.4,
            "DynamicFriction": 0.3,
            "NormalViscousDamping": 0.05,
            "TangentialViscousDamping": 0.05,
        },
    )
    coupling.run(steps=arguments.steps, verbose=True)


if __name__ == "__main__":
    main()
