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
        pytest.skip("Column-collapse GPU regression is run on Taichi CPU/CUDA, not Apple Metal.")


def _run_subprocess(case_name, output_root):
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["GT_TEST_OUTPUT_ROOT"] = os.fspath(output_root)
    code = (
        "from tests.integration.mpm.test_column_collapse_hash_triplet_gpu "
        "import run_column_case\n"
        f"run_column_case({case_name!r})\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)


def _case_options(case_name):
    cases = {
        "hashtriplet_2d": {"dim": 2},
        "hashtriplet_2d_sparse": {"dim": 2, "sparse": True},
        "hashtriplet_2d_adaptive": {"dim": 2, "adaptive": True},
        "hashtriplet_2d_adaptive_sparse": {"dim": 2, "adaptive": True, "sparse": True},
        "hashtriplet_3d": {"dim": 3},
        "hashtriplet_3d_sparse": {"dim": 3, "sparse": True},
    }
    return cases[case_name]


def run_column_case(case_name):
    from geotaichi import MPM, init

    options = _case_options(case_name)
    dim = options["dim"]
    sparse = bool(options.get("sparse", False))
    adaptive = bool(options.get("adaptive", False))
    arch = _test_arch()

    dx = 0.1
    domain = [0.6, 0.5] if dim == 2 else [0.6, 0.5, 4 * dx]
    column_size = [0.2, 0.2] if dim == 2 else [0.2, 0.2, 4 * dx]
    gravity = [0.0, -9.8] if dim == 2 else [0.0, -9.8, 0.0]
    element_size = [dx, dx] if dim == 2 else [dx, dx, dx]
    initial_velocity = [0.0, 0.0] if dim == 2 else [0.0, 0.0, 0.0]
    fix_velocity = ["Free", "Free"] if dim == 2 else ["Free", "Free", "Free"]
    element_type = "Q4N2D" if dim == 2 else "R8N3D"
    region_type = "Rectangle2D" if dim == 2 else "Rectangle"
    output_root = os.environ.get("GT_TEST_OUTPUT_ROOT")
    if not output_root:
        raise RuntimeError(
            "GT_TEST_OUTPUT_ROOT is required for isolated test output")
    save_path = os.path.join(output_root, f"column_hash_{case_name}")

    init(dim=dim, arch=arch, device_memory_GB=3, default_fp="float32", cpu_max_num_threads=2, offline_cache=False, log=False)

    sparse_grid = None
    if sparse:
        sparse_grid = {"Enabled": True, "Backend": "BlockScan", "BlockSize": 2, "MaxActiveBlocks": 0}

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=domain,
        background_damping=0.0,
        gravity=gravity,
        alphaPIC=0.005,
        mapping="MUSL",
        shape_function="Linear",
        solver_type="Implicit",
        material_type="Solid",
        sparse_grid=sparse_grid,
        visualize=False,
        log=False,
    )
    mpm.set_implicit_solver_parameters(
        assemble_type="HashTriplet",
        quasi_static=False,
        max_iteration_number=2,
        displacement_tolerance=1.0e-4,
        residual_tolerance=1.0e-6,
    )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-4,
            "SimulationTime": 2.0e-4,
            "SaveInterval": 1.0,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 4096,
            "max_constraint_number": {"max_displacement_constraint": 10000},
            "dof_multiplier": 4,
        },
        log=False,
    )
    mpm.add_material(
        model="DruckerPrager",
        material={
            "MaterialID": 1,
            "Density": 2500.0,
            "YoungModulus": 8.6e5,
            "PoissonRatio": 0.3,
            "Friction": 19.0,
            "Cohesion": 10.0,
            "Dilation": 0.0,
        },
    )

    element = {"ElementType": element_type, "ElementSize": element_size}
    if adaptive:
        element["AdaptiveGrid"] = {
            "RefineInterval": 1,
            "RefineThreshold": 1.0e-12,
            "RefineCriterion": "EquivalentStress",
            "RefineRatio": 1.0,
            "BufferCells": 0,
            "HangingConstraintMode": "ShapeFunction",
            "HangingShapeCapacityFactor": 4 if dim == 2 else 8,
        }
    mpm.add_element(element=element)
    mpm.add_region(
        region={
            "Name": "region1",
            "Type": region_type,
            "BoundingBoxPoint": [0.0, 0.0] if dim == 2 else [0.0, 0.0, 0.0],
            "BoundingBoxSize": column_size,
        }
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "region1",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": initial_velocity,
                "FixVelocity": fix_velocity,
            }
        }
    )

    if dim == 2:
        boundary = [
            {
                "BoundaryType": "DisplacementConstraint",
                "Displacement": [0.0, 0.0],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [domain[0], 0.0],
            },
            {
                "BoundaryType": "DisplacementConstraint",
                "Displacement": [0.0, 0.0],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.0, domain[1]],
            },
        ]
    else:
        boundary = [
            {
                "BoundaryType": "DisplacementConstraint",
                "Displacement": [0.0, 0.0, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [domain[0], 0.0, domain[2]],
            },
            {
                "BoundaryType": "DisplacementConstraint",
                "Displacement": [0.0, 0.0, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [0.0, domain[1], domain[2]],
            },
        ]
    mpm.add_boundary_condition(boundary=boundary)
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run(gravity_field=lambda points: column_size[1] - points[:, 1])

    particle_num = int(mpm.scene.particleNum[0])
    iterator = mpm.enginer.iterator
    assert particle_num > 0
    assert int(iterator.operator.active_dofs) > 0
    assert iterator.hash_triplet is not None
    assert iterator.hash_triplet.solver == "BiCGSTAB"
    assert np.isfinite(mpm.scene.particle.x.to_numpy()[:particle_num]).all()
    assert np.isfinite(mpm.scene.particle.v.to_numpy()[:particle_num]).all()

    if sparse and not adaptive:
        assert mpm.scene.sparse_grid is not None
        assert mpm.scene.sparse_grid.get_active_blocks() > 0
        assert mpm.scene.sparse_grid.get_active_node_slots() > 0
        assert iterator.node_stride(mpm.scene) == mpm.scene.sparse_grid.node_capacity
    if adaptive:
        assert getattr(mpm.scene.element, "adaptive", False)
        assert getattr(mpm.scene.element, "node_map", None) is not None
        assert int(mpm.scene.element.gridSum) > 0
        if sparse:
            assert mpm.sims.sparse_grid is True
            assert mpm.scene.sparse_grid is None


@pytest.mark.parametrize(
    "case_name",
    [
        "hashtriplet_2d",
        "hashtriplet_2d_sparse",
        "hashtriplet_2d_adaptive",
        "hashtriplet_2d_adaptive_sparse",
        "hashtriplet_3d",
        "hashtriplet_3d_sparse",
    ],
)
def test_column_collapse_hash_triplet_gpu_variants(case_name, tmp_path):
    _skip_if_local_backend_cannot_allocate_sparse_grid()
    _run_subprocess(case_name, tmp_path)
