import os
import platform
import tempfile

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.serial]

def _test_arch():
    arch = os.environ.get("GEOTAICHI_TEST_ARCH", "cpu").lower()
    return "gpu" if arch == "cuda" else arch


def _skip_if_local_backend_cannot_allocate_sparse_grid():
    if _test_arch() == "cpu" and platform.system() == "Darwin" and platform.machine() == "arm64":
        pytest.skip("BlockScan sparse_grid requires Taichi CPU or CUDA; GeoTaichi maps Apple Silicon cpu to Metal.")


def run_case(save_path, sparse_grid=False):
    from geotaichi import MPM, init

    save_path = os.fspath(save_path)

    init(dim=3, arch=_test_arch(), cpu_max_num_threads=2, offline_cache=False, log=False)

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.3, 0.3, 0.3],
        background_damping=0.0,
        gravity=[0.0, 0.0, 0.0],
        alphaPIC=0.5,
        mapping="MUSL",
        shape_function="Linear",
        material_type="TwoPhaseSingleLayer",
        solver_type="Explicit",
        velocity_projection="PIC/FLIP",
        sparse_grid={"Enabled": True, "Backend": "BlockScan", "BlockSize": 2, "MaxActiveBlocks": 8} if sparse_grid else None,
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
    total_velocity = mpm.scene.particle.v.to_numpy()[:particle_num]
    solid_velocity = mpm.scene.particle.vs.to_numpy()[:particle_num]
    fluid_velocity = mpm.scene.particle.vf.to_numpy()[:particle_num]
    pressure = mpm.scene.particle.pressure.to_numpy()[:particle_num]
    assert particle_num > 0
    assert np.isfinite(position).all()
    assert np.isfinite(total_velocity).all()
    assert np.isfinite(solid_velocity).all()
    assert np.isfinite(fluid_velocity).all()
    assert np.isfinite(pressure).all()
    if sparse_grid:
        assert mpm.scene.sparse_grid is not None
        assert mpm.scene.sparse_grid.get_active_blocks() > 0
        assert mpm.scene.sparse_grid.get_active_node_slots() > 0
    assert os.path.exists(os.path.join(save_path, "particles", "MPMParticle000001.npz"))


@pytest.mark.isolated_dimension(3)
def test_single_layer_explicit_3d_musl(tmp_path):
    run_case(tmp_path / "dense")


@pytest.mark.isolated_dimension(3)
def test_single_layer_explicit_3d_musl_sparse_grid(tmp_path):
    _skip_if_local_backend_cannot_allocate_sparse_grid()
    run_case(tmp_path / "sparse", sparse_grid=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(
        prefix="geotaichi-single-layer-explicit-3d-"
    ) as temporary_directory:
        run_case(os.path.join(temporary_directory, "output"))
