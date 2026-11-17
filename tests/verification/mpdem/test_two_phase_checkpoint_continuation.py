"""Compare uninterrupted and fresh-process two-phase continuation."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def run_stage(dimension, output, steps, restart):
    import src.utils.GlobalVariable as gv

    gv.DIMENSION = dimension
    from geotaichi import MPM, init

    init(dim=dimension, arch="cpu", default_fp="float64", cpu_max_num_threads=2, offline_cache=True, log=False)
    mpm = MPM(log=False)
    domain = [0.4] * dimension
    gravity = [0.0] * (dimension - 1) + [-9.8]
    mpm.set_configuration(
        domain=domain,
        background_damping=0.0,
        gravity=gravity,
        alphaPIC=0.5,
        mapping="USL",
        shape_function="Linear",
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
    )
    mpm.set_solver(
        {"Timestep": 1e-4, "SimulationTime": steps * 1e-4, "SaveInterval": 1e-4, "SavePath": str(output)}, log=False
    )
    mpm.set_semi_implicit_solver_parameters(
        {
            "pressure_solver": "MGPCG",
            "linear_solver": "MGPCG",
            "assemble_type": "MatrixFree",
            "max_iteration_number": 100,
            "residual_tolerance": 1e-10,
            "multilevel": 2,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": 64,
            "max_constraint_number": {"max_velocity_constraint": 512},
        },
        log=False,
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "SolidDensity": 2650.0,
            "FluidDensity": 1000.0,
            "Porosity": 0.4,
            "FluidBulkModulus": 2.2e8,
            "Permeability": 1e-4,
            "FluidViscosity": 1e-3,
            "GrainDiameter": 1e-2,
            "YoungModulus": 1e6,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element({"ElementType": "Q4N2D" if dimension == 2 else "R8N3D", "ElementSize": [0.1] * dimension})
    if restart:
        mpm.read_restart(1, str(restart), is_continue=True)
        assert mpm.scene.two_phase_fluid_mass_initialized
    else:
        mpm.add_region(
            {
                "Name": "sample",
                "Type": "Rectangle2D" if dimension == 2 else "Rectangle",
                "BoundingBoxPoint": [0.1] * dimension,
                "BoundingBoxSize": [0.1] * dimension,
            }
        )
        mpm.add_body(
            {
                "Template": [
                    {
                        "RegionName": "sample",
                        "nParticlesPerCell": 2,
                        "BodyID": 0,
                        "MaterialID": 1,
                        "Phase": phase,
                        "InitialVelocity": [speed] + [0.0] * (dimension - 1),
                        "FixVelocity": ["Free"] * dimension,
                    }
                    for phase, speed in [("Solid", 0.02), ("Fluid", -0.01)]
                ]
            }
        )
    boundaries = []
    for axis in range(dimension):
        for side in (0, 1):
            start = [0.0] * dimension
            end = list(domain)
            start[axis] = end[axis] = domain[axis] * side
            velocity = [None] * dimension
            velocity[axis] = 0.0
            boundaries.append(
                {"BoundaryType": "VelocityConstraint", "Velocity": velocity, "StartPoint": start, "EndPoint": end}
            )
    mpm.add_boundary_condition(boundaries)
    mpm.select_save_data(particle=True, grid=True, object=False)
    mpm.run()
    assert mpm.scene.two_phase_fluid_mass_initialized
    assert abs(mpm.sims.current_time - steps * 1e-4) < 1e-12


@pytest.mark.parametrize("dimension", [2, 3])
def test_two_phase_fresh_process_continuation(tmp_path, dimension):
    for label, steps, restart in [("baseline", 3, ""), ("prefix", 1, ""), ("resumed", 3, str(tmp_path / "prefix"))]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.verification.mpdem.test_two_phase_checkpoint_continuation",
                str(dimension),
                str(tmp_path / label),
                str(steps),
                restart,
            ],
            cwd=ROOT,
            env={**os.environ, "GEOTAICHI_REAL_DTYPE": "float64"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
        (tmp_path / (label + ".log")).write_text(result.stdout)
        assert result.returncode == 0, result.stdout[-10000:]
    baseline = tmp_path / "baseline/particles/MPMParticle000003.npz"
    resumed = tmp_path / "resumed/particles/MPMParticle000003.npz"
    with np.load(baseline) as a, np.load(resumed) as b:
        for key in [
            "position",
            "velocity",
            "solid_velocity",
            "fluid_velocity",
            "solid_mass",
            "fluid_mass",
            "pressure",
            "porosity",
            "solid_velocity_gradient",
            "fluid_velocity_gradient",
            "stress",
        ]:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-12, err_msg=key)
        assert int(b["current_step"]) == 3 and bool(b["two_phase_fluid_mass_initialized"])


if __name__ == "__main__":
    run_stage(int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
