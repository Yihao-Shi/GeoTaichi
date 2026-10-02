import os
import tempfile

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.serial]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _test_arch():
    arch = os.environ.get("GEOTAICHI_TEST_ARCH", "cpu").lower()
    return "gpu" if arch == "cuda" else arch


def run_case(save_path):
    from geotaichi import MPM, init

    save_path = os.fspath(save_path)

    init(dim=3, arch=_test_arch(), cpu_max_num_threads=2, offline_cache=False, log=False)

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.3, 0.3, 0.3],
        background_damping=0.0,
        gravity=[0.0, 0.0, 0.0],
        alphaPIC=0.5,
        mapping="USL",
        shape_function="Linear",
        material_type="TwoPhaseSingleLayer",
        solver_type="SemiImplicit",
        velocity_projection="PIC/FLIP",
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
            "assemble_type": "MatrixFreeMGPCG",
            "linear_solver": "MGPCG",
            "max_iteration_number": 50,
            "residual_tolerance": 1.0e-8,
            "multilevel": 2,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 256,
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
            "Permeability": 1.0e-4,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 1.0e-2,
            "YoungModulus": 1.0e6,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": [0.1, 0.1, 0.1]})
    mpm.add_region(
        region={
            "Name": "sample",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.1, 0.1, 0.1],
            "BoundingBoxSize": [0.1, 0.1, 0.1],
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
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                }
            ]
        }
    )
    mpm.add_boundary_condition(
        boundary=[
            {"BoundaryType": "VelocityConstraint", "Velocity": [0.0, None, None], "StartPoint": [0.0, 0.0, 0.0], "EndPoint": [0.0, 0.3, 0.3]},
            {"BoundaryType": "VelocityConstraint", "Velocity": [0.0, None, None], "StartPoint": [0.3, 0.0, 0.0], "EndPoint": [0.3, 0.3, 0.3]},
            {"BoundaryType": "VelocityConstraint", "Velocity": [None, 0.0, None], "StartPoint": [0.0, 0.0, 0.0], "EndPoint": [0.3, 0.0, 0.3]},
            {"BoundaryType": "VelocityConstraint", "Velocity": [None, None, 0.0], "StartPoint": [0.0, 0.0, 0.0], "EndPoint": [0.3, 0.3, 0.0]},
            {"BoundaryType": "VelocityConstraint", "Velocity": [None, None, 0.0], "StartPoint": [0.0, 0.0, 0.3], "EndPoint": [0.3, 0.3, 0.3]},
        ]
    )
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run()

    particle_num = int(mpm.scene.particleNum[0])
    position = mpm.scene.particle.x.to_numpy()[:particle_num]
    velocity = mpm.scene.particle.v.to_numpy()[:particle_num]
    pressure = mpm.scene.particle.pressure.to_numpy()[:particle_num]
    assert particle_num > 0
    assert np.isfinite(position).all()
    assert np.isfinite(velocity).all()
    assert np.isfinite(pressure).all()
    assert os.path.exists(os.path.join(save_path, "particles", "MPMParticle000001.npz"))


@pytest.mark.isolated_dimension(3)
def test_single_layer_3d_mgpcg(tmp_path):
    run_case(tmp_path / "single-layer-3d")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(
        prefix="geotaichi-single-layer-3d-"
    ) as temporary_directory:
        run_case(os.path.join(temporary_directory, "output"))
