"""Device COO assembly contracts for the coupled IGA--MPM solver."""

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.iga, pytest.mark.cpu]


@pytest.mark.parametrize("solver", ["PCG", "BiCGSTAB"])
def test_igampm_coo_scatter_and_krylov_remain_on_device(taichi_runtime, solver):
    import src.igampm.config as config
    from src.igampm.engines import Engine
    from src.linear_solver.BuildTriplet import BuildTriplet
    from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix

    config.set_dimension(2)
    dense = np.array(
        [
            [4.0, 1.0, 0.2, 0.3],
            [1.0, 3.0, 0.4, 0.5],
            [0.2, 0.4, 2.0, 0.5],
            [0.3, 0.5, 0.5, 5.0],
        ],
        dtype=np.float64,
    )
    if solver == "BiCGSTAB":
        dense[0, 2] = 0.7
    rows, columns = np.nonzero(dense)
    source = BuildTriplet(
        dim=2,
        max_pairs_num=32,
        max_nonzeros=32,
        max_active_nodes=2,
        symmetric=False,
        solver="PCG",
    )
    source.reset_system()
    source.assemble_scalar_triplets(
        rows.astype(np.int32),
        columns.astype(np.int32),
        dense[rows, columns],
    )

    engine = object.__new__(Engine)
    engine.assemble_type = "COO"
    engine.friction_mode = "lagged"
    engine.monolithic_solver_name = solver
    engine.monolithic_coo_matrix = CoordinateSparseMatrix(128, 4, preconditioned=True, symmetry=solver == "PCG")
    engine.monolithic_coo_count = ti.field(ti.i32, shape=())
    engine.monolithic_coo_overflow = ti.field(ti.i32, shape=())
    engine.monolithic_coo_diagonal = ti.field(ti.f64, shape=4)
    engine.monolithic_rhs = ti.field(ti.f64, shape=4)
    engine.monolithic_correction = ti.field(ti.f64, shape=4)
    engine.monolithic_fixed = ti.field(ti.i32, shape=4)
    engine.monolithic_fixed_correction = ti.field(ti.f64, shape=4)
    engine.monolithic_linear_solver_tolerance = 1.0e-12
    engine.monolithic_linear_solver_relative_tolerance = 0.0
    engine.monolithic_linear_solver_max_iters = 200

    engine.monolithic_coo_matrix.reset()
    engine.monolithic_coo_count[None] = 0
    engine.monolithic_coo_overflow[None] = 0
    engine._append_hash_source_to_monolithic_coo(source, 2, 0)
    active_nnz = int(engine.monolithic_coo_count[None])
    assembled = engine.monolithic_coo_matrix._to_scipy().toarray()
    np.testing.assert_allclose(assembled, dense, rtol=0.0, atol=1.0e-12)
    assert int(engine.monolithic_coo_overflow[None]) == 0

    rhs = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    engine.monolithic_rhs.from_numpy(rhs)
    fixed_dof = 1
    fixed_correction = 0.25
    engine.monolithic_fixed[fixed_dof] = 1
    engine.monolithic_fixed_correction[fixed_dof] = fixed_correction
    engine._eliminate_device_monolithic_dirichlet_coo(4)
    active_nnz = int(engine.monolithic_coo_count[None])
    engine.monolithic_coo_matrix.linear_operator.update_active_dofs(4)
    engine.monolithic_coo_matrix.linear_operator.update_nnz(active_nnz)
    engine._build_device_monolithic_coo_diagonal(4)

    expected_matrix = dense.copy()
    expected_rhs = rhs - dense[:, fixed_dof] * fixed_correction
    expected_matrix[:, fixed_dof] = 0.0
    expected_matrix[fixed_dof, :] = 0.0
    expected_matrix[fixed_dof, fixed_dof] = 1.0
    expected_rhs[fixed_dof] = fixed_correction
    np.testing.assert_allclose(
        engine.monolithic_coo_matrix._to_scipy().toarray(),
        expected_matrix,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(engine.monolithic_rhs.to_numpy(), expected_rhs, rtol=0.0, atol=1.0e-12)

    result = engine._solve_monolithic_linear_system({"active_dof": 4, "active_nodes": 2})
    assert result["converged"]
    assert result["backend"] == f"taichi_coo_{solver.lower()}"
    np.testing.assert_allclose(
        engine.monolithic_correction.to_numpy(),
        np.linalg.solve(expected_matrix, expected_rhs),
        rtol=1.0e-9,
        atol=1.0e-11,
    )
