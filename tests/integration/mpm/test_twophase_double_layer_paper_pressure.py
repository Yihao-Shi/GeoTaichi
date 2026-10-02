import os
import tempfile

import numpy as np
import pytest
import taichi as ti


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from geotaichi import MPM, init


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.serial]


def run_case(save_path):
    save_path = os.fspath(save_path)
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)

    shape_function = os.environ.get("GEOTAICHI_DOUBLE_LAYER_SHAPE", "QuadBSpline")
    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.4, 0.4],
        background_damping=0.0,
        gravity=[0.0, -9.8],
        alphaPIC=1.0,
        mapping="USL",
        shape_function=shape_function,
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
        velocity_projection="Affine",
    )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-4,
            "SimulationTime": 1.0e-4,
            "SaveInterval": 1.0e-4,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.set_semi_implicit_solver_parameters(
        {
            "linear_solver": "MGPCG",
            "pressure_solver": "MGPCG",
            "multilevel": 2,
            "max_iteration_number": 100,
            "residual_tolerance": 1.0e-6,
        }
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 64,
            "max_constraint_number": {"max_velocity_constraint": 128},
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
            "Permeability": 1.0e-4,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 1.0e-2,
            "YoungModulus": 1.0e6,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [0.1, 0.1]})
    mpm.add_region(
        region={
            "Name": "sample",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.15, 0.15],
            "BoundingBoxSize": [0.1, 0.1],
        }
    )
    mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "sample",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Solid",
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                },
                {
                    "RegionName": "sample",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                },
            ]
        }
    )
    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.4, 0.0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.0, 0.4],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [0.4, 0.0],
                "EndPoint": [0.4, 0.4],
            },
        ]
    )
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run()

    particle_num = int(mpm.scene.particleNum[0])
    position = mpm.scene.particle.x.to_numpy()[:particle_num]
    velocity = mpm.scene.particle.v.to_numpy()[:particle_num]
    pressure = mpm.scene.particle.pressure.to_numpy()[:particle_num]
    phase = mpm.scene.particle.phase.to_numpy()[:particle_num]
    assert particle_num == 8
    assert set(np.unique(phase).tolist()) == {1, 2}
    assert np.isfinite(position).all()
    assert np.isfinite(velocity).all()
    assert np.isfinite(pressure).all()
    assert pressure[phase == 2].size > 0
    assert np.isfinite(pressure[phase == 2]).all()
    saved_particle = np.load(os.path.join(save_path, "particles", "MPMParticle000001.npz"))
    assert "pressure" in saved_particle.files
    assert "phase" in saved_particle.files
    assert np.isfinite(saved_particle["pressure"][saved_particle["phase"] == 2]).all()
    assert mpm.enginer.poisson_solver is not None
    assert os.path.exists(os.path.join(save_path, "vtks", "GraphicMPMSolidParticle000001.vtu"))
    assert os.path.exists(os.path.join(save_path, "vtks", "GraphicMPMFluidParticle000001.vtu"))


def test_twophase_double_layer_pressure_one_step_and_particle_output(tmp_path):
    run_case(tmp_path / "double-layer-paper")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(
        prefix="geotaichi-double-layer-paper-"
    ) as temporary_directory:
        run_case(os.path.join(temporary_directory, "output"))
