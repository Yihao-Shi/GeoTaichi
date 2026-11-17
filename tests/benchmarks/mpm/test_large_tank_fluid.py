"""Opt-in large-tank MPM fluid benchmark.

The original ``test_fluid.py`` allocated a 56k-particle Taichi simulation and
started defining kernels during module import.  Keeping the benchmark useful
requires the opposite contract: collection is side-effect free, while an
explicit benchmark run constructs the full problem inside the test.
"""

from __future__ import annotations

import os
from time import perf_counter

import numpy as np
import pytest


pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.mpm,
    pytest.mark.slow,
    pytest.mark.serial,
]


def _run_large_tank(output_path, *, steps, arch):
    from geotaichi import MPM, init

    init(
        dim=2,
        arch=arch,
        device_memory_GB=3.7,
        kernel_profiler=True,
        offline_cache=False,
        log=False,
    )
    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[8.0, 1.0],
        background_damping=0.0,
        alphaPIC=0.0,
        mapping="USL",
        shape_function="QuadBSpline",
        gravity=[0.0, -9.8],
        material_type="Fluid",
        log=False,
    )
    mpm.set_solver(
        {
            "Timestep": 1.0e-5,
            "SimulationTime": steps * 1.0e-5,
            "SaveInterval": steps * 1.0e-5,
            "SavePath": str(output_path),
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 60000,
            "max_constraint_number": {
                "max_reflection_constraint": 550000,
                "max_friction_constraint": 0,
                "max_velocity_constraint": 0,
            },
        },
        log=False,
    )
    mpm.add_material(
        model="Newtonian",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "Modulus": 2.0e7,
            "Viscosity": 1.0e-3,
            "ElementLength": 0.01,
            "cL": 0.7,
            "cQ": 2.0,
        },
    )
    mpm.add_element(
        element={
            "ElementType": "Q4N2D",
            "ElementSize": [0.01, 0.01],
        }
    )
    mpm.add_region(
        region={
            "Name": "water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": [7.0, 0.4],
        }
    )
    mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "water",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                }
            ]
        }
    )
    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "ReflectionConstraint",
                "Norm": [0.0, -1.0],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [8.0, 0.0],
            },
            {
                "BoundaryType": "ReflectionConstraint",
                "Norm": [-1.0, 0.0],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.0, 1.0],
            },
            {
                "BoundaryType": "ReflectionConstraint",
                "Norm": [1.0, 0.0],
                "StartPoint": [8.0, 0.0],
                "EndPoint": [8.0, 1.0],
            },
        ]
    )

    mpm.add_essentials(gravity_field=True)
    mpm.enginer.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    particle_count = int(mpm.scene.particleNum[0])
    start = perf_counter()
    for _ in range(steps):
        mpm.solver.core(mpm.scene, mpm.neighbor)
        mpm.sims.current_time += mpm.sims.delta
        mpm.sims.current_step += 1
    elapsed = perf_counter() - start

    position = mpm.scene.particle.x.to_numpy()[:particle_count]
    return particle_count, position, elapsed


def test_large_tank_fluid_throughput(tmp_path):
    steps = int(os.environ.get("GEOTAICHI_BENCHMARK_STEPS", "20"))
    arch = os.environ.get("GEOTAICHI_BENCHMARK_ARCH", "gpu")
    particle_count, position, elapsed = _run_large_tank(
        tmp_path / "large_tank", steps=steps, arch=arch
    )

    assert 50000 <= particle_count <= 60000
    assert np.isfinite(position).all()
    assert elapsed > 0.0
    particle_steps_per_second = particle_count * steps / elapsed
    print(
        f"large-tank: arch={arch} particles={particle_count} steps={steps} "
        f"elapsed={elapsed:.6f}s particle-steps/s={particle_steps_per_second:.3e}"
    )
