"""Keep each affine body's four control blocks together in the preconditioner."""

import numpy as np
import pytest
import taichi as ti
from scipy.sparse import csr_matrix

from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.BlockJacobi import BlockJacobi
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.linear_solver.LinearOperator import LinearOperator


def grouped_system(nodes=8):
    return BuildTriplet(
        3,
        nodes * nodes,
        nodes * nodes,
        nodes,
        symmetric=False,
        matrix_symmetric=True,
        solver="PCG",
        preconditioner_group_size=4,
    )


def test_grouped_principal_inverse_scales_and_reuses_active_prefix(taichi_runtime):
    rng = np.random.default_rng(36)
    factor = rng.normal(size=(24, 24))
    matrix = factor @ factor.T + np.eye(24)
    rhs = rng.normal(size=(8, 3))
    system = grouped_system()
    system.rhs.from_numpy(rhs)
    for nodes, scale in [(8, 1e-12), (4, 1e12), (8, 1.0)]:
        active = scale * matrix[: nodes * 3, : nodes * 3]
        system.load_from_scipy_blocks(csr_matrix(active), active_nodes=nodes)
        system._build_group_jacobi(nodes, int(system.non_diag.element_pair_num[0]))
        assert system.group_preconditioner.error[None] == 0
        system._apply_group_jacobi(nodes, system.rhs, system.z)
        actual = scale * system.z.to_numpy()[:nodes].ravel()
        expected = np.concatenate(
            [np.linalg.solve(matrix[i : i + 12, i : i + 12], rhs.ravel()[i : i + 12]) for i in range(0, nodes * 3, 12)]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", [False, True])
def test_grouped_pcg_resolves_stiff_body_modes_and_true_residual(taichi_runtime, device):
    rng = np.random.default_rng(44)
    matrix = np.zeros((24, 24))
    for start in [0, 12]:
        q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        matrix[start : start + 12, start : start + 12] = (q * np.geomspace(1e-3, 1e3, 12)) @ q.T
    coupling = rng.normal(size=24)
    matrix += 0.1 * np.outer(coupling, coupling)
    expected = rng.normal(scale=1e-4, size=24)
    rhs = matrix @ expected
    system = grouped_system()
    system.device_execution = device
    system.load_from_scipy_blocks(csr_matrix(matrix))
    system.rhs.from_numpy(rhs.reshape(8, 3))
    result = system.solve(tol=1e-12, maxiter=100, return_solution=False)
    assert result["converged"], result
    actual = system.x.to_numpy().ravel()
    assert np.linalg.norm(rhs - matrix @ actual) <= 1e-12
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    assert result["iterations"] < 30


@pytest.mark.parametrize("diagonal", [-1.0, np.nan])
def test_grouped_pcg_rejects_invalid_principal_block(taichi_runtime, diagonal):
    matrix = np.eye(12)
    matrix[0, 0] = diagonal
    system = grouped_system(4)
    system.load_from_scipy_blocks(csr_matrix(matrix))
    system.rhs.from_numpy(np.ones((4, 3)))
    with pytest.raises(RuntimeError, match="Cholesky pivot"):
        system.solve(tol=1e-12, maxiter=10, return_solution=False)


def mixed_system(symmetric=True):
    return BuildTriplet(
        3,
        121,
        121,
        11,
        symmetric=False,
        matrix_symmetric=symmetric,
        solver="PCG" if symmetric else "BiCGSTAB",
        preconditioner_group_size=4,
        preconditioner_group_nodes=8,
    )


def principal_reference(matrix, rhs, transpose=False):
    bounds = [0, 12, 24, *range(27, matrix.shape[0] + 1, 3)]
    return np.concatenate(
        [
            np.linalg.solve(matrix[a:b, a:b].T if transpose else matrix[a:b, a:b], rhs[a:b])
            for a, b in zip(bounds, bounds[1:])
        ]
    )


@pytest.mark.parametrize("symmetric", [True, False])
def test_mixed_principal_blocks_include_all_terms_and_follow_active_grid(taichi_runtime, symmetric):
    rng = np.random.default_rng(713)
    matrix = rng.normal(size=(33, 33))
    if symmetric:
        matrix = matrix @ matrix.T + np.eye(33)
    else:
        matrix += 8 * np.eye(33)
        matrix[[0, 7]] = matrix[[7, 0]]  # Requires a row pivot within the ABD block.
    rhs = rng.normal(size=(11, 3))
    system = mixed_system(symmetric)
    for active, scale in [(11, 1e-12), (8, 1e12), (9, 1.0)]:
        current = scale * matrix[: 3 * active, : 3 * active]
        system.load_from_scipy_blocks(csr_matrix(current), active_nodes=active)
        system.rhs.from_numpy(rhs)
        system.group_preconditioner.build_triplet(
            system, active, int(system.non_diag.element_pair_num[0]), cholesky=symmetric
        )
        for transpose in (False, True):
            system.group_preconditioner.apply(active, system.rhs, system.z, transpose=transpose)
            np.testing.assert_allclose(
                scale * system.z.to_numpy()[:active].ravel(),
                principal_reference(matrix[: 3 * active, : 3 * active], rhs[:active].ravel(), transpose),
                rtol=1e-11,
                atol=1e-11,
            )
        np.testing.assert_allclose(system.to_scipy(active).toarray(), current, rtol=1e-14, atol=1e-24)


@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("symmetric,transpose", [(True, False), (False, False), (False, True)])
def test_mixed_krylov_solves_complete_coupled_matrix(taichi_runtime, device, symmetric, transpose):
    rng = np.random.default_rng(714)
    matrix = rng.normal(size=(33, 33))
    matrix = matrix @ matrix.T + 5 * np.eye(33) if symmetric else matrix + 12 * np.eye(33)
    expected = rng.normal(size=33)
    rhs = (matrix.T if transpose else matrix) @ expected
    system = mixed_system(symmetric)
    system.device_execution = device
    system.load_from_scipy_blocks(csr_matrix(matrix))
    system.rhs.from_numpy(rhs.reshape(11, 3))
    result = system.solve(tol=1e-11, maxiter=300, return_solution=False, transpose=transpose)
    assert result["converged"], result
    actual = system.x.to_numpy().ravel()
    assert np.linalg.norm((matrix.T if transpose else matrix) @ actual - rhs) <= 1e-11
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)


@pytest.mark.parametrize("device", [False, True])
def test_exact_indefinite_adjoint_uses_requested_lu_bicgstab_fallback(taichi_runtime, device):
    matrix = np.diag(np.linspace(1, 3, 33))
    matrix[0, 0] = -1
    matrix[0, 24] = matrix[24, 0] = 0.2
    rhs = np.linspace(-1, 1, 33)
    field = ti.field(ti.f64, shape=33)
    field.from_numpy(rhs)
    system = mixed_system()
    system.device_execution = device
    system.load_from_scipy_blocks(csr_matrix(matrix))
    result = system.solve_flat_system(field, tol=1e-12, maxiter=100, fallback_to_bicgstab=True)
    assert result["converged"], result
    assert result["fallback_from"] == "PCG"
    assert "Cholesky" in result["primary_preconditioner_failure"]
    np.testing.assert_allclose(system.x.to_numpy().ravel(), np.linalg.solve(matrix, rhs), rtol=0, atol=1e-12)


@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("symmetric", [True, False])
def test_coo_mixed_preconditioner_preserves_duplicate_terms_and_scalar_krylov(taichi_runtime, device, symmetric):
    rng = np.random.default_rng(715)
    matrix = rng.normal(size=(33, 33))
    matrix = matrix @ matrix.T + 5 * np.eye(33) if symmetric else matrix + 12 * np.eye(33)
    coordinate = CoordinateSparseMatrix(2 * matrix.size, 33, symmetry=symmetric)
    row, col = np.indices(matrix.shape)
    coordinate.rows.from_numpy(np.tile(row.ravel(), 2).astype(np.int32))
    coordinate.cols.from_numpy(np.tile(col.ravel(), 2).astype(np.int32))
    coordinate.data.from_numpy(np.tile(0.5 * matrix.ravel(), 2))
    preconditioner = BlockJacobi(3, 11, 4, 8)
    preconditioner.build_coo(coordinate, 11, cholesky=symmetric)
    rhs, solution = ti.field(ti.f64, shape=33), ti.field(ti.f64, shape=33)
    values = rng.normal(size=33)
    rhs.from_numpy(values)
    preconditioner.apply(11, rhs, solution, flat=True)
    np.testing.assert_allclose(solution.to_numpy(), principal_reference(matrix, values), rtol=1e-12, atol=1e-12)
    solution.fill(0.0)
    coordinate.linear_solver.device_execution = device
    inverse = LinearOperator(lambda src, dst: preconditioner.apply(11, src, dst, flat=True))
    assert coordinate.solve(rhs, solution, inverse, tol=1e-11, maxiter=300)
    assert np.linalg.norm(matrix @ solution.to_numpy() - values) <= 1e-11


def test_mixed_prefix_rejects_partial_or_out_of_range_bodies(taichi_runtime):
    for nodes in (-4, 5, 12, 4.5):
        with pytest.raises(ValueError):
            BuildTriplet(
                3,
                121,
                121,
                11,
                symmetric=False,
                matrix_symmetric=True,
                solver="PCG",
                preconditioner_group_size=4,
                preconditioner_group_nodes=nodes,
            )
    system = mixed_system()
    with pytest.raises(ValueError, match="affine prefix"):
        system.group_preconditioner.grouped_nodes(7)


@pytest.mark.parametrize("diagonal", [0.0, np.nan])
def test_mixed_lu_rejects_singular_or_nonfinite_body_without_updating_solution(taichi_runtime, diagonal):
    matrix = np.eye(33)
    matrix[0, 0] = diagonal
    system = mixed_system(symmetric=False)
    system.load_from_scipy_blocks(csr_matrix(matrix))
    system.rhs.fill(1.0)
    before = system.x.to_numpy().copy()
    with pytest.raises(RuntimeError, match="LU pivot"):
        system.solve(tol=1e-12, return_solution=False)
    np.testing.assert_array_equal(system.x.to_numpy(), before)
