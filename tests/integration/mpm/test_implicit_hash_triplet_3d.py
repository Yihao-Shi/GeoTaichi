import os

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.slow, pytest.mark.serial]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def run_case(output_root):
    from geotaichi import MPM, init
    from src.mpm.engines.AssembleMatrixKernel import kernel_moment_balance_direct

    save_path = os.path.join(
        os.fspath(output_root), "implicit_hash_triplet_3d")

    init(dim=3, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.2, 0.2, 0.2],
        gravity=[0.0, 0.0, -9.8],
        alphaPIC=0.0,
        mapping="MUSL",
        shape_function="Linear",
        solver_type="Implicit",
        material_type="Solid",
        visualize=False,
        log=False,
    )
    mpm.set_implicit_solver_parameters(
        assemble_type="HashTriplet",
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

    iterator = mpm.enginer.iterator
    active_dofs = int(iterator.operator.active_dofs)
    active_nodes = active_dofs // mpm.sims.dimension
    assert active_dofs > 0

    assert iterator.local_stiffness is None
    iterator.preconditioning_matrix(mpm.sims, mpm.scene)
    iterator.assemble_global_matrix(mpm.sims, mpm.scene)

    rng = np.random.default_rng(11)
    x_np = np.zeros(iterator.unknow_vector.shape[0], dtype=np.float64)
    x_np[:active_dofs] = rng.normal(size=active_dofs)
    x = ti.field(float, shape=iterator.unknow_vector.shape[0])
    ax = ti.field(float, shape=iterator.unknow_vector.shape[0])
    x.from_numpy(x_np)

    kernel_moment_balance_direct(
        mpm.scene.element.gridSum,
        mpm.scene.element.grid_nodes,
        active_dofs,
        int(mpm.scene.particleNum[0]),
        mpm.scene.particle,
        mpm.scene.element.dshape_fn,
        mpm.scene.element.node_size,
        mpm.scene.element.LnID,
        mpm.scene.element.flag,
        mpm.scene.material.stiffness_matrix,
        iterator.assemble_stiffness_matrix,
        iterator.mass_matrix,
        x,
        ax,
    )

    matrix = iterator.hash_triplet.to_scipy(active_nodes)
    ax_ref = matrix @ x_np[:active_dofs]
    ax_hash = ax.to_numpy()[:active_dofs]
    rel = np.linalg.norm(ax_hash - ax_ref) / max(np.linalg.norm(ax_ref), 1.0e-30)
    assert rel < 1.0e-10, rel

    particle_num = int(mpm.scene.particleNum[0])
    assert particle_num > 0
    assert np.isfinite(mpm.scene.particle.x.to_numpy()[:particle_num]).all()
    assert np.isfinite(mpm.scene.particle.v.to_numpy()[:particle_num]).all()


@pytest.mark.isolated_dimension(3)
def test_implicit_hash_triplet_3d(tmp_path):
    run_case(tmp_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
