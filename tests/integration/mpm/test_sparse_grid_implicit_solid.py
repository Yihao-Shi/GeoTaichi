import os
import platform

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.serial]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _test_arch():
    arch = os.environ.get("GEOTAICHI_TEST_ARCH", "cpu").lower()
    return "gpu" if arch == "cuda" else arch


def _skip_if_local_backend_cannot_allocate_sparse_grid():
    if _test_arch() == "cpu" and platform.system() == "Darwin" and platform.machine() == "arm64":
        pytest.skip("BlockScan sparse_grid requires Taichi CPU or CUDA; GeoTaichi maps Apple Silicon cpu to Metal.")


def _run_sparse_implicit_solid_case(assemble_type, output_root):
    from geotaichi import MPM

    save_path = os.path.join(
        os.fspath(output_root), f"sparse_grid_implicit_{assemble_type}")

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.2, 0.2, 0.2],
        gravity=[0.0, 0.0, -9.8],
        alphaPIC=0.0,
        mapping="MUSL",
        shape_function="Linear",
        configuration="ULMPM",
        solver_type="Implicit",
        material_type="Solid",
        sparse_grid={"Enabled": True, "Backend": "BlockScan", "BlockSize": 2, "MaxActiveBlocks": 8},
        visualize=False,
        log=False,
    )
    mpm.set_implicit_solver_parameters(
        assemble_type=assemble_type,
        max_iteration_number=2,
        displacement_tolerance=1.0e-3,
        residual_tolerance=1.0e-8,
    )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-3,
            "SimulationTime": 1.0e-3,
            "SaveInterval": 1.0e-3,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 32,
            "max_constraint_number": {"max_displacement_constraint": 128},
        },
        log=False,
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "Density": 1800.0,
            "YoungModulus": 2.0e5,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": [0.1, 0.1, 0.1]})
    mpm.add_region(
        region={
            "Name": "block",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.05, 0.05, 0.05],
            "BoundingBoxSize": [0.1, 0.1, 0.1],
        }
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "block",
                "nParticlesPerCell": 1,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        }
    )
    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "DisplacementConstraint",
                "Displacement": [0.0, 0.0, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [0.2, 0.2, 0.0],
            }
        ]
    )
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run()

    sparse_grid = mpm.scene.sparse_grid
    iterator = mpm.enginer.iterator
    particle_num = int(mpm.scene.particleNum[0])
    assert sparse_grid is not None
    assert sparse_grid.get_active_blocks() > 0
    assert sparse_grid.get_active_node_slots() > 0
    assert iterator.node_stride(mpm.scene) == sparse_grid.node_capacity
    assert int(iterator.operator.active_dofs) > 0
    assert particle_num > 0
    assert np.isfinite(mpm.scene.particle.x.to_numpy()[:particle_num]).all()
    assert np.isfinite(mpm.scene.particle.v.to_numpy()[:particle_num]).all()


@pytest.mark.parametrize("assemble_type", ["MatrixFree", "COO", "HashTriplet"])
@pytest.mark.isolated_dimension(3)
def test_sparse_grid_implicit_solid_3d_assemblies(assemble_type, tmp_path):
    _skip_if_local_backend_cannot_allocate_sparse_grid()

    from geotaichi import init

    init(dim=3, arch=_test_arch(), cpu_max_num_threads=2, offline_cache=False, log=False)
    _run_sparse_implicit_solid_case(assemble_type, tmp_path)
