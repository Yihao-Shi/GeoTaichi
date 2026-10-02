import os
import platform
import subprocess
import sys

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


def _output_path(name, output_root=None):
    root = output_root
    if root is None:
        root = os.environ.get("GT_TEST_OUTPUT_ROOT")
    if not root:
        raise RuntimeError("GT_TEST_OUTPUT_ROOT is required for isolated test output")
    return os.path.join(os.fspath(root), name)


def _run_subprocess(code, output_root):
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["GT_TEST_OUTPUT_ROOT"] = os.fspath(output_root)
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)


def _run_sparse_solid_case(configuration, material_model, output_root=None):
    from geotaichi import MPM

    save_path = _output_path(f"sparse_grid_{configuration}", output_root)

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.2, 0.2],
        gravity=[0.0, -9.8],
        alphaPIC=0.0,
        mapping="MUSL",
        shape_function="Linear",
        configuration=configuration,
        solver_type="Explicit",
        material_type="Solid",
        sparse_grid={"Enabled": True, "Backend": "BlockScan", "BlockSize": 2, "MaxActiveBlocks": 4},
        visualize=False,
        log=False,
    )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-5,
            "SimulationTime": 1.0e-5,
            "SaveInterval": 1.0e-5,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 64,
            "max_constraint_number": {"max_velocity_constraint": 64},
        },
        log=False,
    )
    mpm.add_material(
        model=material_model,
        material={
            "MaterialID": 1,
            "Density": 1800.0,
            "YoungModulus": 2.0e5,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [0.1, 0.1]})
    mpm.add_region(
        region={
            "Name": "block",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.05, 0.05],
            "BoundingBoxSize": [0.1, 0.1],
        }
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "block",
                "nParticlesPerCell": 1,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
        }
    )
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run()

    sparse_grid = mpm.scene.sparse_grid
    particle_num = int(mpm.scene.particleNum[0])
    assert sparse_grid is not None
    assert sparse_grid.get_active_blocks() > 0
    assert sparse_grid.get_active_node_slots() > 0
    assert particle_num > 0
    assert np.isfinite(mpm.scene.particle.x.to_numpy()[:particle_num]).all()
    assert np.isfinite(mpm.scene.particle.v.to_numpy()[:particle_num]).all()


def _run_sparse_solid_3d_case(configuration, material_model, output_root=None):
    from geotaichi import MPM

    save_path = _output_path(f"sparse_grid_3d_{configuration}", output_root)

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.2, 0.2, 0.2],
        gravity=[0.0, 0.0, -9.8],
        alphaPIC=0.0,
        mapping="MUSL",
        shape_function="Linear",
        configuration=configuration,
        solver_type="Explicit",
        material_type="Solid",
        sparse_grid={"Enabled": True, "Backend": "BlockScan", "BlockSize": 2, "MaxActiveBlocks": 8},
        visualize=False,
        log=False,
    )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-5,
            "SimulationTime": 1.0e-5,
            "SaveInterval": 1.0e-5,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 64,
            "max_constraint_number": {"max_velocity_constraint": 64},
        },
        log=False,
    )
    mpm.add_material(
        model=material_model,
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
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run()

    sparse_grid = mpm.scene.sparse_grid
    particle_num = int(mpm.scene.particleNum[0])
    assert sparse_grid is not None
    assert sparse_grid.get_active_blocks() > 0
    assert sparse_grid.get_active_node_slots() > 0
    assert particle_num > 0
    assert np.isfinite(mpm.scene.particle.x.to_numpy()[:particle_num]).all()
    assert np.isfinite(mpm.scene.particle.v.to_numpy()[:particle_num]).all()


def _run_sparse_fluid_3d_case(output_root=None):
    from geotaichi import MPM

    save_path = _output_path("sparse_grid_3d_fluid", output_root)

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.2, 0.2, 0.2],
        gravity=[0.0, 0.0, -9.8],
        alphaPIC=0.0,
        mapping="MUSL",
        shape_function="Linear",
        configuration="ULMPM",
        solver_type="Explicit",
        material_type="Fluid",
        sparse_grid={"Enabled": True, "Backend": "BlockScan", "BlockSize": 2, "MaxActiveBlocks": 8},
        visualize=False,
        log=False,
    )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-5,
            "SimulationTime": 1.0e-5,
            "SaveInterval": 1.0e-5,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 64,
            "max_constraint_number": {"max_velocity_constraint": 64},
        },
        log=False,
    )
    mpm.add_material(
        model="Newtonian",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "Modulus": 2.0e5,
            "Viscosity": 1.0e-3,
        },
    )
    mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": [0.1, 0.1, 0.1]})
    mpm.add_region(
        region={
            "Name": "fluid",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.05, 0.05, 0.05],
            "BoundingBoxSize": [0.1, 0.1, 0.1],
        }
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "fluid",
                "nParticlesPerCell": 1,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        }
    )
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run()

    sparse_grid = mpm.scene.sparse_grid
    particle_num = int(mpm.scene.particleNum[0])
    assert sparse_grid is not None
    assert sparse_grid.get_active_blocks() > 0
    assert sparse_grid.get_active_node_slots() > 0
    assert particle_num > 0
    assert np.isfinite(mpm.scene.particle.x.to_numpy()[:particle_num]).all()
    assert np.isfinite(mpm.scene.particle.v.to_numpy()[:particle_num]).all()


def test_sparse_grid_explicit_ulmpm_and_tlmpm_solid(tmp_path):
    _skip_if_local_backend_cannot_allocate_sparse_grid()

    from geotaichi import init

    init(dim=2, arch=_test_arch(), cpu_max_num_threads=2, offline_cache=False, log=False)
    _run_sparse_solid_case("ULMPM", "LinearElastic", tmp_path)
    _run_sparse_solid_case("TLMPM", "NeoHookean", tmp_path)


def test_sparse_grid_explicit_3d_solid(tmp_path):
    _skip_if_local_backend_cannot_allocate_sparse_grid()

    code = (
        "from geotaichi import init\n"
        "from tests.integration.mpm.test_sparse_grid_explicit import _run_sparse_solid_3d_case\n"
        f"init(dim=3, arch={_test_arch()!r}, cpu_max_num_threads=2, offline_cache=False, log=False)\n"
        "_run_sparse_solid_3d_case('ULMPM', 'LinearElastic')\n"
        "_run_sparse_solid_3d_case('TLMPM', 'NeoHookean')\n"
    )
    _run_subprocess(code, tmp_path)


def test_sparse_grid_explicit_3d_fluid(tmp_path):
    _skip_if_local_backend_cannot_allocate_sparse_grid()

    code = (
        "from geotaichi import init\n"
        "from tests.integration.mpm.test_sparse_grid_explicit import _run_sparse_fluid_3d_case\n"
        f"init(dim=3, arch={_test_arch()!r}, cpu_max_num_threads=2, offline_cache=False, log=False)\n"
        "_run_sparse_fluid_3d_case()\n"
    )
    _run_subprocess(code, tmp_path)
