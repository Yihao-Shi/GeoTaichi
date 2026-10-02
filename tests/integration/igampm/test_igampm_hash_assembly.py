"""Integration checks for hash assembly across IGA and MPM."""

import numpy as np
import pytest

import src.igampm.config as config


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.serial]


def _assert_same_matrix(label, raw, reduced, atol=1.0e-8):
    diff = (raw - reduced).tocoo()
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    scale = max(float(np.max(np.abs(raw.data))) if raw.nnz else 0.0, 1.0e-30)
    rel = max_abs / scale
    print(f"{label}: raw_nnz={raw.nnz}, hash_nnz={reduced.nnz}, max_abs={max_abs:.3e}, rel={rel:.3e}")
    assert raw.shape == reduced.shape
    assert max_abs <= atol


def _assert_same_vector(label, raw, reduced, atol=1.0e-8):
    raw = np.asarray(raw, dtype=np.float64)
    reduced = np.asarray(reduced, dtype=np.float64)
    max_abs = float(np.max(np.abs(raw - reduced))) if raw.size else 0.0
    scale = max(float(np.max(np.abs(raw))) if raw.size else 0.0, 1.0e-30)
    rel = max_abs / scale
    print(f"{label}: max_abs={max_abs:.3e}, rel={rel:.3e}")
    assert raw.shape == reduced.shape
    assert max_abs <= atol


def _check_mpm_ul_assembly(output_path):
    from src.mpm.generator.Body import Body
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    body = Body()
    body.add_rectangle([0.11, 0.11], [0.31, 0.31], 0.1, ppc=1)
    mpm = ImplicitULMPM(
        domain=[0.4, 0.4],
        dx=0.1,
        dt=1.0e-3,
        bodies=body,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, -9.8],
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
    mpm.compute_mass_list(mpm.integration)
    constrained_grid = int(mpm.dof2node.to_numpy()[0])
    constrained_dof = config.DIM * constrained_grid
    dirichlet = DirichletBoundary()
    dirichlet.append([[constrained_dof]], [1.0e-5])
    dirichlet.finalize(config.DIM * mpm.total_background_grid_num)
    mpm.dirichlet = dirichlet

    mpm.grid_disp.fill(0.0)
    mpm.matrix_reset()
    mpm.hash_matrix.reset_system()
    mpm.assemble_inertia_force(mpm.active_dof, mpm.damping, mpm.gravity, mpm.integration, mpm.grid_disp)
    mpm.assemble_material_force(mpm.active_dof, mpm.grid_disp)
    mpm.assemble_stiffness_matrix(mpm.active_dof, mpm.grid_disp)
    mpm.assemble_mass_matrix(mpm.stiffness_nnz)
    mpm.apply_dirichlet(mpm.stiffness_nnz, mpm.active_dof)
    mpm.hash_matrix.finalize_taichi_assembly()
    raw = mpm.hash_matrix.to_scipy(mpm.active_dof // config.DIM).tocsr()
    raw_rhs = mpm.rhs.to_numpy()[:mpm.active_dof]

    mpm.matrix_reset()
    mpm.hash_matrix.reset_system()
    mpm.assemble_inertia_force(mpm.active_dof, mpm.damping, mpm.gravity, mpm.integration, mpm.grid_disp)
    mpm.assemble_material_force(mpm.active_dof, mpm.grid_disp)
    mpm.assemble_stiffness_matrix_hash(mpm.active_dof, mpm.grid_disp)
    mpm.assemble_mass_matrix_hash()
    mpm.apply_dirichlet_hash(mpm.active_dof)
    mpm.hash_matrix.finalize_taichi_assembly()
    reduced = mpm.hash_matrix.to_scipy(mpm.active_dof // config.DIM).tocsr()
    reduced_rhs = mpm.rhs.to_numpy()[:mpm.active_dof]

    _assert_same_matrix("ImplicitULMPM block hash assembly", raw, reduced)
    _assert_same_vector("ImplicitULMPM block hash rhs", raw_rhs, reduced_rhs)


def _check_iga_assembly(output_path):
    from src.iga import ImplicitIGA, Primitives, Rectangle

    body = Rectangle()
    body.set_parameters(size=[1.0, 0.4])
    body.generate_knot_u(degree=2, num_ctrlpts=3)
    body.generate_knot_v(degree=2, num_ctrlpts=3)
    body.generate_ctrlpts()
    body.generate_weights()
    body.activate_boundary()
    body.gather_boundary_ctrlpts()

    primitives = Primitives()
    primitives.append(body, "patch")
    primitives.finialize()

    iga = ImplicitIGA(
        primitives=primitives,
        dirichlet=None,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, -9.8],
        residual=1.0e-8,
        interval=1,
        step=1,
        degree=[2, 2],
        path=str(output_path),
    )

    iga.precompute()
    iga.rhs.fill(0.0)
    iga.incre_resolution.fill(0.0)
    iga.hash_matrix.reset_system()
    for patch_id in range(iga.patch.primitive.num_primitives):
        iga.assemble_stiffness_matrix(
            iga.prefix_nnz[patch_id],
            iga.patch.total_num_ctrlpts[patch_id + 1],
            iga.patch.total_num_element[patch_id + 1],
            iga.patch.prefix_total_num_ctrlpts[patch_id],
            iga.patch.prefix_num_knot[patch_id],
            iga.patch.prefix_num_element[patch_id],
            iga.patch.num_knot[patch_id + 1],
            iga.patch.num_element[patch_id + 1],
            iga.patch.num_ctrlpts[patch_id + 1],
            iga.integration,
            iga.gravity,
            iga.grid_disp,
        )

    iga.hash_matrix.finalize_taichi_assembly()
    raw = iga.hash_matrix.to_scipy(iga.degree_of_freedom // config.DIM).tocsr()
    raw_rhs = iga.rhs.to_numpy()

    iga.rhs.fill(0.0)
    iga.incre_resolution.fill(0.0)
    iga.hash_matrix.reset_system()
    iga.assemble_body_matrix()
    iga.hash_matrix.finalize_taichi_assembly()
    reduced = iga.hash_matrix.to_scipy(iga.degree_of_freedom // config.DIM).tocsr()
    reduced_rhs = iga.rhs.to_numpy()

    _assert_same_matrix("ImplicitIGA block hash assembly", raw, reduced)
    _assert_same_vector("ImplicitIGA block hash rhs", raw_rhs, reduced_rhs)


def test_igampm_hash_assembly_matches_raw_triplets(taichi_runtime, tmp_path):
    config.set_dimension(2)
    _check_mpm_ul_assembly(tmp_path / "mpm")
    _check_iga_assembly(tmp_path / "iga")
