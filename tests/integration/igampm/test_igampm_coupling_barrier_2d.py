"""Two-dimensional IGA-MPM barrier integration check."""

import numpy as np
import pytest

import src.igampm.config as config


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.serial]


def _build_iga_rectangle(output_path, axisymmetric=False):
    from src.iga import ImplicitIGA, Primitives, Rectangle

    rectangle = Rectangle()
    rectangle.set_parameters(
        start_point=[0.5, 0.0] if axisymmetric else [0.0, 0.0],
        size=[1.0, 0.2],
    )
    rectangle.generate_knot_u(degree=2, num_ctrlpts=3)
    rectangle.generate_knot_v(degree=2, num_ctrlpts=3)
    rectangle.generate_ctrlpts()
    rectangle.generate_weights()

    primitives = Primitives()
    primitives.append(rectangle, "rectangle")
    primitives.finialize()
    return ImplicitIGA(
        primitives=primitives,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        degree=[2, 2],
        axisymmetric=axisymmetric,
        axis_offset=0.0,
        path=str(output_path),
    )


def _build_mpm_particle(output_path, axisymmetric=False):
    from src.mpm.generator.Body import Body
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    body = Body()
    body.add_particles(
        [[1.0 if axisymmetric else 0.5, -0.031]],
        volume=1.0e-3,
        xmin=[-0.1, -0.1],
        xmax=[1.1, 0.3],
        boundary_ids=[0],
    )
    mpm = ImplicitULMPM(
        domain=[1.2, 0.5],
        dx=0.1,
        dt=1.0e-3,
        bodies=body,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        scale=1.0,
        line_search=False,
        shape_function="linear",
        visualize=False,
        axisymmetric=axisymmetric,
        axis_offset=0.0,
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


def test_igampm_coupling_barrier_2d_assembles_symmetric_system(
    taichi_runtime,
    monkeypatch,
    tmp_path,
):
    config.set_dimension(2)

    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    coupling = IGAMPM(iga, mpm, kappa=1.0e4, dhat=0.08, barrier_nnz=20_000)

    coupling.initialize_barrier()
    print(f"2D IGA-MPM barrier contacts: {coupling.curr_barrier_contact_num}")
    assert coupling.curr_barrier_contact_num > 0

    coupling.assemble_barrier_system()
    K = coupling.barrier_matrix()
    nnz_count = int(coupling.barrier_nnz_count[0])
    grad = coupling.barrier_grad.to_numpy()
    sym_diff = (K - K.T).tocoo()
    max_sym = float(np.max(np.abs(sym_diff.data))) if sym_diff.nnz else 0.0
    print(
        "2D IGA-MPM barrier assembly:",
        f"raw_triplets={nnz_count}",
        f"reduced_nnz={K.nnz}",
        f"grad_norm={np.linalg.norm(grad):.3e}",
        f"sym_max={max_sym:.3e}",
    )
    blocks = coupling.barrier_blocks()
    forces = coupling.barrier_contact_forces()
    from src.linear_solver.BuildTriplet import BuildTriplet

    with monkeypatch.context() as patch:
        patch.setattr(BuildTriplet, "append_reduced_from", BuildTriplet.append_raw_from)
        baseline = coupling.assemble_monolithic_newton_system()
        reference = baseline["matrix"].to_scipy(baseline["active_nodes"]).toarray()
        reference_rhs = baseline["rhs"].to_numpy().copy()
    monolithic = coupling.assemble_monolithic_newton_system()
    np.testing.assert_allclose(
        monolithic["matrix"].to_scipy(monolithic["active_nodes"]).toarray(), reference, rtol=1e-12, atol=1e-7
    )
    np.testing.assert_allclose(monolithic["rhs"].to_numpy(), reference_rhs, rtol=1e-12, atol=1e-7)
    total_active = iga.degree_of_freedom + mpm.active_dof
    monolithic_matrix = monolithic["matrix"].to_scipy(monolithic["active_nodes"])
    monolithic_rhs = monolithic["rhs"].to_numpy()[:total_active]
    assert nnz_count > 0
    assert K.nnz > 0
    assert np.all(np.isfinite(K.data))
    assert np.all(np.isfinite(grad))
    assert np.linalg.norm(grad) > 0.0
    assert max_sym < 1.0e-7
    assert blocks["K_im"].shape == (iga.degree_of_freedom, mpm.active_dof)
    assert blocks["K_mi"].shape == (mpm.active_dof, iga.degree_of_freedom)
    assert forces["iga"].shape == (iga.degree_of_freedom,)
    assert forces["mpm"].shape == (mpm.active_dof,)
    assert np.linalg.norm(forces["mpm"]) > 0.0
    assert monolithic_matrix.shape == (total_active, total_active)
    assert monolithic_rhs.shape == (total_active,)


def test_igampm_axisymmetric_ipc_uses_revolved_measure_and_symmetric_system(
    taichi_runtime,
    tmp_path,
):
    config.set_dimension(2)

    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / "iga-axisymmetric", axisymmetric=True)
    mpm = _build_mpm_particle(tmp_path / "mpm-axisymmetric", axisymmetric=True)
    coupling = IGAMPM(
        iga,
        mpm,
        kappa=1.0e4,
        dhat=0.08,
        barrier_nnz=20_000,
        axisymmetric=True,
        axis_offset=0.0,
    )

    iga.precompute()
    expected_iga_volume = np.pi * (1.5**2 - 0.5**2) * 0.2
    assert np.sum(iga.patch.volume.to_numpy()) == pytest.approx(expected_iga_volume, rel=2.0e-12)
    assert mpm.particle.vol0.to_numpy()[0] == pytest.approx(2.0 * np.pi * 1.0e-3, rel=2.0e-12)
    initial_F = iga.initial_deformation_gradients.to_numpy().reshape(-1, 3, 3)
    assert np.allclose(initial_F, np.eye(3), rtol=0.0, atol=2.0e-12)

    iga.reset_linear_system()
    iga.rhs.fill(0.0)
    iga.assemble_body_matrix(project_spd=False)
    body_matrix = iga.hash_matrix.to_scipy(iga.degree_of_freedom // 2)
    body_symmetry_error = (body_matrix - body_matrix.T).tocoo()
    maximum_body_error = float(np.max(np.abs(body_symmetry_error.data))) if body_symmetry_error.nnz else 0.0
    assert body_matrix.nnz > 0
    assert np.all(np.isfinite(body_matrix.data))
    assert maximum_body_error < 1.0e-7

    coupling.initialize_barrier()
    assert mpm.surface_measure.to_numpy()[0] > 0.0
    assert coupling.curr_barrier_contact_num > 0
    coupling.assemble_barrier_system()
    matrix = coupling.barrier_matrix()
    symmetry_error = (matrix - matrix.T).tocoo()
    maximum_error = float(np.max(np.abs(symmetry_error.data))) if symmetry_error.nnz else 0.0
    assert matrix.nnz > 0
    assert np.all(np.isfinite(matrix.data))
    assert maximum_error < 1.0e-7
