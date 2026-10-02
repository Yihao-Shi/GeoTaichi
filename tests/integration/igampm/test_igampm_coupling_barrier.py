"""Three-dimensional IGA-MPM barrier integration check."""

import numpy as np
import pytest

import src.igampm.config as config

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.serial]


def _build_iga_cube(output_path):
    from src.iga import Cube, ImplicitIGA, Primitives

    cube = Cube()
    cube.set_parameters(start_point=[0.0, 0.0, 0.0], size=[1.0, 1.0, 0.2])
    cube.generate_knot_u(degree=2, num_ctrlpts=3)
    cube.generate_knot_v(degree=2, num_ctrlpts=3)
    cube.generate_knot_w(degree=2, num_ctrlpts=3)
    cube.generate_ctrlpts()
    cube.generate_weights()

    primitives = Primitives()
    primitives.append(cube, "cube")
    primitives.finialize()
    return ImplicitIGA(
        primitives=primitives,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        degree=[2, 2, 2],
        path=str(output_path),
    )


def _build_mpm_particle(output_path):
    from src.mpm.generator.Body import Body
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    body = Body()
    body.add_particles(
        [[0.5, 0.5, -0.031]],
        volume=1.0e-3,
        xmin=[-0.1, -0.1, -0.1],
        xmax=[1.1, 1.1, 0.3],
        boundary_ids=[0],
    )
    mpm = ImplicitULMPM(
        domain=[1.2, 1.2, 0.5],
        dx=0.1,
        dt=1.0e-3,
        bodies=body,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        scale=1.0,
        line_search=False,
        shape_function="linear",
        visualize=False,
        path=str(output_path),
    )
    mpm.init_F0()
    mpm.mass_vec.fill(0.0)
    mpm.grid_reset()
    mpm.compute_shapefn()
    mpm.mass_vel_acc_p2g()
    mpm.find_active_node()
    mpm.prefix_sum_executor.run(mpm.node2dof)
    mpm.active_dof = mpm.set_active_dof()
    mpm.compute_nodal_vel_acc()
    assert mpm.active_dof > 0
    return mpm


@pytest.mark.isolated_dimension(3)
def test_igampm_semi_ipc_keeps_contact_ccd(taichi_runtime, tmp_path):
    config.set_dimension(3)

    from src.igampm import IGAMPM

    iga = _build_iga_cube(tmp_path / "iga-semi")
    mpm = _build_mpm_particle(tmp_path / "mpm-semi")
    coupling = IGAMPM(
        iga,
        mpm,
        contact_model="SemiIPC",
        penalty=1.0e4,
        dhat=0.08,
        barrier_nnz=20_000,
    )
    coupling.initialize_barrier()
    iga.incre_resolution.fill(0.0)
    direction = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    direction[: mpm.active_dof].reshape((-1, 3))[:, 2] = 0.1
    mpm.incre_resolution.from_numpy(direction)

    step = coupling.conservative_contact_step_device(
        max_step=1.0,
        safety=0.9,
        verify=False,
    )
    coupling.assemble_barrier_system()

    assert coupling.barrier.model == "SemiIPC"
    assert 0.0 < step < 1.0
    assert coupling.curr_barrier_contact_num > 0
    assert np.isfinite(coupling.barrier_grad.to_numpy()).all()


@pytest.mark.isolated_dimension(3)
def test_igampm_coupling_barrier_3d_assembles_symmetric_system(
    taichi_runtime,
    tmp_path,
):
    config.set_dimension(3)

    from src.igampm import IGAMPM

    iga = _build_iga_cube(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    coupling = IGAMPM(iga, mpm, kappa=1.0e4, dhat=0.08, barrier_nnz=20_000)

    coupling.initialize_barrier()
    print(f"IGA-MPM barrier contacts: {coupling.curr_barrier_contact_num}")
    assert coupling.curr_barrier_contact_num > 0

    iga.incre_resolution.fill(0.0)
    mpm_direction = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    mpm_direction[: mpm.active_dof].reshape((-1, 3))[:, 2] = 0.1
    mpm.incre_resolution.from_numpy(mpm_direction)
    control_points_before = coupling.contact_surface.control_points_hat.to_numpy().copy()
    contact_step = coupling.conservative_contact_step_device(max_step=1.0, safety=0.9)
    expected_pairs = int(mpm.total_surface_num) * int(coupling.contact_surface.num_surfaces)
    assert 0.0 < contact_step < 1.0
    assert np.isclose(
        contact_step,
        np.min(coupling.contact_accd_toc.to_numpy()[:expected_pairs]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        coupling.contact_surface.control_points_hat.to_numpy(),
        control_points_before,
    )

    coupling.assemble_barrier_system()
    K = coupling.barrier_matrix()
    nnz_count = int(coupling.barrier_nnz_count[0])
    grad = coupling.barrier_grad.to_numpy()
    sym_diff = (K - K.T).tocoo()
    max_sym = float(np.max(np.abs(sym_diff.data))) if sym_diff.nnz else 0.0
    print(
        "IGA-MPM barrier assembly:",
        f"raw_triplets={nnz_count}",
        f"reduced_nnz={K.nnz}",
        f"grad_norm={np.linalg.norm(grad):.3e}",
        f"sym_max={max_sym:.3e}",
    )
    blocks = coupling.barrier_blocks()
    forces = coupling.barrier_contact_forces()
    monolithic = coupling.assemble_monolithic_newton_system()
    total_active = iga.degree_of_freedom + mpm.active_dof
    monolithic_matrix = monolithic["matrix"].to_scipy(monolithic["active_nodes"])
    monolithic_rhs = monolithic["rhs"].to_numpy()[:total_active]
    assert nnz_count > 0
    assert K.nnz > 0
    assert np.linalg.norm(grad) > 0.0
    assert max_sym < 1.0e-7
    assert blocks["K_im"].shape == (iga.degree_of_freedom, mpm.active_dof)
    assert blocks["K_mi"].shape == (mpm.active_dof, iga.degree_of_freedom)
    assert forces["iga"].shape == (iga.degree_of_freedom,)
    assert forces["mpm"].shape == (mpm.active_dof,)
    assert np.linalg.norm(forces["mpm"]) > 0.0
    assert monolithic_matrix.shape == (total_active, total_active)
    assert monolithic_rhs.shape == (total_active,)
