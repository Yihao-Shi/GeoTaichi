"""Regression contracts for matrix-free Krylov convergence state.

The stiff-system cases are intentionally scaled like an IPC barrier Hessian:
the Jacobi-preconditioned recurrence norm is tiny even though the physical
residual is still above the requested absolute tolerance.
"""

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.linear_solver.MatrixFreeBICGSTAB import MatrixFreeBICGSTAB
from src.linear_solver.MatrixFreeCG import MatrixFreeCG
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB
from src.linear_solver.MatrixFreePCG import MatrixFreePCG

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.linear_solver,
]


@ti.data_oriented
class _DenseOperator:
    def __init__(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        self.size = int(matrix.shape[0])
        self.matrix = ti.field(dtype=ti.f64, shape=matrix.shape)
        self.matrix.from_numpy(matrix)

    def matvec(self, vector, result):
        self._matvec(vector, result)

    @ti.kernel
    def _matvec(self, vector: ti.template(), result: ti.template()):
        for row in range(self.size):
            value = 0.0
            for column in range(self.size):
                value += self.matrix[row, column] * vector[column]
            result[row] = value


def _scalar_fields(rhs, initial=None):
    rhs = np.asarray(rhs, dtype=np.float64)
    if initial is None:
        initial = np.zeros_like(rhs)
    b = ti.field(dtype=ti.f64, shape=rhs.size)
    x = ti.field(dtype=ti.f64, shape=rhs.size)
    b.from_numpy(rhs)
    x.from_numpy(np.asarray(initial, dtype=np.float64))
    return b, x


@pytest.mark.matrix_free
def test_pcg_uses_true_residual_for_stiff_jacobi_system(taichi_runtime):
    matrix = np.asarray([[1.0e10, 2.0e8], [2.0e8, 2.0e10]], dtype=np.float64)
    expected = np.asarray([1.25e-15, -0.75e-15])
    rhs = matrix @ expected
    operator = _DenseOperator(matrix)
    b, x = _scalar_fields(rhs)
    diagonal = ti.field(dtype=ti.f64, shape=2)
    diagonal.from_numpy(np.diag(matrix))
    solver = MatrixFreePCG(2)

    # sqrt(r.T M^-1 r) is below 1e-9 for this system, while ||r||_2 is
    # about 2e-5.  The former stopping criterion returned the untouched zero
    # initial guess.
    preconditioned_norm = np.sqrt(np.dot(rhs, rhs / np.diag(matrix)))
    assert preconditioned_norm < 1.0e-9
    assert np.linalg.norm(rhs) > 1.0e-9

    succeeded = solver.solve(operator, b, x, diagonal, size=2, tol=1.0e-9, maxiter=20)

    assert succeeded
    assert solver.last_converged
    assert solver.last_iterations > 0
    assert solver.last_breakdown_reason == ""
    assert solver.last_initial_residual == pytest.approx(np.linalg.norm(rhs), rel=2.0e-15)
    assert solver.last_residual <= 1.0e-9
    np.testing.assert_allclose(x.to_numpy(), expected, rtol=2.0e-13, atol=1.0e-28)
    assert np.linalg.norm(matrix @ x.to_numpy() - rhs) <= 1.0e-9


@pytest.mark.matrix_free
def test_pcg_verifies_recursive_residual_before_reporting_convergence(
    taichi_runtime,
):
    """An unattainable absolute tolerance must not accept residual drift."""

    indices = np.arange(5, dtype=np.float64)
    matrix = 1.0e20 / (indices[:, None] + indices[None, :] + 1.0)
    expected = np.linspace(-1.0, 1.0, 5) * 1.0e-12
    rhs = matrix @ expected
    operator = _DenseOperator(matrix)
    b, x = _scalar_fields(rhs)
    diagonal = ti.field(dtype=ti.f64, shape=5)
    diagonal.from_numpy(np.diag(matrix))
    solver = MatrixFreePCG(5)

    succeeded = solver.solve(operator, b, x, diagonal, size=5, tol=1.0e-9, maxiter=40)
    actual_residual = np.linalg.norm(matrix @ x.to_numpy() - rhs)

    assert not succeeded
    assert not solver.last_converged
    assert solver.last_residual_restarts > 0
    assert actual_residual > 1.0e-9
    assert solver.last_residual > 1.0e-9
    # At this deliberately extreme 1e20 scale, NumPy and the Taichi kernel
    # sum each dense row in a different order, so their cancellation floors
    # need not be bitwise-close.  The solver's reported value must match the
    # freshly recomputed r=b-Ax field that actually made the convergence
    # decision; the independently evaluated NumPy residual above still proves
    # that neither arithmetic path meets the requested tolerance.
    assert solver.last_residual == pytest.approx(np.linalg.norm(solver.r.to_numpy()), rel=2.0e-15)


@pytest.mark.matrix_free
@pytest.mark.parametrize(
    ("solver_class", "preconditioned"),
    (
        (MatrixFreeCG, False),
        (MatrixFreeBICGSTAB, False),
        (MatrixFreePBICGSTAB, True),
    ),
)
def test_krylov_siblings_reject_false_recursive_convergence(taichi_runtime, solver_class, preconditioned):
    matrix = np.asarray([[1.0, -0.99999999], [-0.99999999, 1.0]], dtype=np.float64)
    rhs = np.asarray([1.0, 0.3], dtype=np.float64)
    operator = _DenseOperator(matrix)
    b, x = _scalar_fields(rhs)
    solver = solver_class(2)

    if preconditioned:
        diagonal = ti.field(dtype=ti.f64, shape=2)
        diagonal.fill(1.0)
        succeeded = solver.solve(operator, b, x, diagonal, size=2, tol=1.0e-12, maxiter=30)
    else:
        succeeded = solver.solve(operator, b, x, size=2, tol=1.0e-12, maxiter=30)

    actual_residual = np.linalg.norm(matrix @ x.to_numpy() - rhs)
    assert not succeeded
    assert solver.last_residual_restarts > 0
    assert actual_residual > 1.0e-12
    assert solver.last_residual > 1.0e-12
    # As in the stiff PCG test above, cancellation depends on matvec ordering.
    # Verify the final state with a fresh application of the actual operator,
    # while the independent NumPy residual still forbids false convergence.
    operator.matvec(x, solver.Ax)
    device_residual = np.linalg.norm(b.to_numpy() - solver.Ax.to_numpy())
    assert solver.last_residual == pytest.approx(device_residual, rel=2.0e-15)
    assert solver.last_residual == pytest.approx(np.linalg.norm(solver.r.to_numpy()), rel=2.0e-15)


@pytest.mark.coo
def test_coordinate_sparse_propagates_pcg_failure_state(taichi_runtime):
    matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    rows, columns = np.nonzero(matrix)
    sparse = CoordinateSparseMatrix(
        nonzeros=rows.size,
        degree_of_freedom=2,
        preconditioned=True,
        symmetry=True,
    )
    sparse.rows.from_numpy(rows.astype(np.int32))
    sparse.cols.from_numpy(columns.astype(np.int32))
    sparse.data.from_numpy(matrix[rows, columns])
    sparse.linear_operator.update_nnz(rows.size)
    b, x = _scalar_fields(np.asarray([1.0, -2.0]))
    diagonal = ti.field(dtype=ti.f64, shape=2)
    diagonal.from_numpy(np.diag(matrix))

    succeeded = sparse.solve(b, x, diagonal, tol=1.0e-14, maxiter=0)

    assert succeeded is False
    assert not sparse.linear_solver.last_converged
    assert sparse.linear_solver.last_iterations == 0
    assert sparse.linear_solver.last_breakdown_reason == "maximum_iterations"
    assert sparse.linear_solver.last_residual == pytest.approx(np.sqrt(5.0))
    np.testing.assert_array_equal(x.to_numpy(), np.zeros(2))


@pytest.mark.coo
def test_coordinate_sparse_forwards_pcg_relative_tolerance(taichi_runtime):
    matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    rows, columns = np.nonzero(matrix)
    sparse = CoordinateSparseMatrix(
        nonzeros=rows.size,
        degree_of_freedom=2,
        preconditioned=True,
        symmetry=True,
    )
    sparse.rows.from_numpy(rows.astype(np.int32))
    sparse.cols.from_numpy(columns.astype(np.int32))
    sparse.data.from_numpy(matrix[rows, columns])
    sparse.linear_operator.update_nnz(rows.size)
    b, x = _scalar_fields(np.asarray([1.0, -2.0]))
    diagonal = ti.field(dtype=ti.f64, shape=2)
    diagonal.from_numpy(np.diag(matrix))

    succeeded = sparse.solve(
        b,
        x,
        diagonal,
        tol=1.0e-12,
        rel_tol=0.3,
        maxiter=1,
    )

    assert succeeded
    assert sparse.linear_solver.last_converged
    assert sparse.linear_solver.last_iterations == 1
    assert sparse.linear_solver.last_residual <= (0.3 * sparse.linear_solver.last_initial_residual)


@pytest.mark.hash_triplet
def test_hash_triplet_pcg_uses_same_true_residual_contract(
    taichi_runtime,
):
    diagonal_values = np.asarray([1.0e10, 2.0e10])
    expected = np.asarray([1.25e-15, -0.75e-15])
    rhs = diagonal_values * expected
    matrix = BuildTriplet(
        dim=1,
        max_pairs_num=1,
        max_nonzeros=1,
        max_active_nodes=2,
        symmetric=False,
        solver="PCG",
        matrix_symmetric=True,
        device_reduction=False,
    )
    matrix.reset_system()
    matrix.assemble_scalar_triplets(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        diagonal_values,
    )
    matrix.finalize_taichi_assembly()

    result = matrix.solve(
        rhs=rhs.reshape((2, 1)),
        active_nodes=2,
        tol=1.0e-9,
        maxiter=20,
    )

    assert result["converged"]
    assert result["iterations"] > 0
    assert result["residual"] <= 1.0e-9
    np.testing.assert_allclose(result["x"].reshape(-1), expected, rtol=2.0e-15, atol=1.0e-30)


@pytest.mark.hash_triplet
def test_hash_triplet_pcg_supports_scale_invariant_relative_tolerance(
    taichi_runtime,
):
    dense = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    rhs = np.asarray([1.0, -2.0], dtype=np.float64)
    matrix = BuildTriplet(
        dim=1,
        max_pairs_num=3,
        max_nonzeros=3,
        max_active_nodes=2,
        symmetric=False,
        solver="PCG",
        matrix_symmetric=True,
        device_reduction=False,
    )
    matrix.reset_system()
    matrix.assemble_scalar_triplets(
        np.asarray([0, 0, 1], dtype=np.int32),
        np.asarray([0, 1, 1], dtype=np.int32),
        np.asarray([4.0, 1.0, 3.0], dtype=np.float64),
    )
    matrix.finalize_taichi_assembly()

    absolute_only = matrix.solve(
        rhs=rhs.reshape((2, 1)),
        active_nodes=2,
        tol=1.0e-12,
        maxiter=1,
    )
    relative = matrix.solve(
        rhs=rhs.reshape((2, 1)),
        active_nodes=2,
        tol=1.0e-12,
        rel_tol=0.3,
        maxiter=1,
    )

    assert not absolute_only["converged"]
    assert relative["converged"]
    assert relative["iterations"] == 1
    assert relative["initial_residual"] == pytest.approx(np.linalg.norm(rhs))
    assert relative["convergence_tolerance"] == pytest.approx(0.3 * np.linalg.norm(rhs))
    assert relative["residual"] <= relative["convergence_tolerance"]


@pytest.mark.hash_triplet
def test_hash_triplet_pcg_curvature_check_is_scale_invariant(taichi_runtime):
    diagonal = np.asarray([1.0e-40, 2.0e-40])
    expected = np.asarray([1.0, -2.0])
    rhs = diagonal * expected
    matrix = BuildTriplet(
        dim=1,
        max_pairs_num=1,
        max_nonzeros=1,
        max_active_nodes=2,
        symmetric=False,
        solver="PCG",
        matrix_symmetric=True,
        device_reduction=False,
    )
    matrix.reset_system()
    matrix.assemble_scalar_triplets(
        np.arange(2, dtype=np.int32),
        np.arange(2, dtype=np.int32),
        diagonal,
    )
    matrix.finalize_taichi_assembly()

    result = matrix.solve(
        rhs=rhs.reshape((2, 1)),
        active_nodes=2,
        tol=0.0,
        rel_tol=1.0e-12,
        maxiter=2,
    )

    assert result["converged"]
    assert result["iterations"] == 1
    np.testing.assert_allclose(result["x"].reshape(-1), expected, rtol=2.0e-15)


@pytest.mark.hash_triplet
def test_hash_triplet_exact_symmetric_solve_falls_back_after_negative_curvature(taichi_runtime):
    matrix = BuildTriplet(
        dim=1,
        max_pairs_num=1,
        max_nonzeros=1,
        max_active_nodes=2,
        symmetric=False,
        solver="PCG",
        matrix_symmetric=True,
        device_reduction=True,
    )
    matrix.reset_system()
    matrix.assemble_scalar_triplets(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([-2.0, 1.0], dtype=np.float64),
    )
    matrix.finalize_taichi_assembly()

    rhs = ti.field(ti.f64, shape=2)
    solution = ti.field(ti.f64, shape=2)
    rhs.from_numpy(np.asarray([1.0, 0.0]))
    result = matrix.solve_flat_system(
        rhs,
        solution,
        active_nodes=2,
        tol=1.0e-12,
        maxiter=20,
        return_solution=False,
        fallback_to_bicgstab=True,
    )

    assert result["converged"]
    assert result["fallback_from"] == "PCG"
    assert matrix.solver == "PCG"
    np.testing.assert_allclose(solution.to_numpy(), [-0.5, 0.0], rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.hash_triplet
def test_hash_triplet_pcg_verifies_recursive_residual(
    taichi_runtime,
):
    indices = np.arange(5, dtype=np.float64)
    dense = 1.0e20 / (indices[:, None] + indices[None, :] + 1.0)
    expected = np.linspace(-1.0, 1.0, 5) * 1.0e-12
    rhs = dense @ expected
    rows, columns = np.indices(dense.shape)
    matrix = BuildTriplet(
        dim=1,
        max_pairs_num=dense.size,
        max_nonzeros=dense.size,
        max_active_nodes=5,
        symmetric=False,
        solver="PCG",
        matrix_symmetric=False,
        device_reduction=False,
    )
    matrix.reset_system()
    matrix.assemble_scalar_triplets(
        rows.reshape(-1).astype(np.int32),
        columns.reshape(-1).astype(np.int32),
        dense.reshape(-1),
    )
    matrix.finalize_taichi_assembly()

    result = matrix.solve(
        rhs=rhs.reshape((5, 1)),
        active_nodes=5,
        tol=1.0e-9,
        maxiter=40,
    )
    actual_residual = np.linalg.norm(dense @ result["x"].reshape(-1) - rhs)

    assert not result["converged"]
    assert actual_residual > 1.0e-9
    assert result["residual"] > 1.0e-9
    roundoff_bound = (
        10.0 * np.finfo(np.float64).eps * (np.linalg.norm(dense) * np.linalg.norm(result["x"]) + np.linalg.norm(rhs))
    )
    assert abs(result["residual"] - actual_residual) <= roundoff_bound


@pytest.mark.hash_triplet
def test_hash_triplet_bicgstab_solves_transposed_block_system(
    taichi_runtime,
):
    dense = np.asarray(
        [
            [6.0, 1.0, 0.5, -0.2],
            [-0.4, 5.0, 0.3, 0.1],
            [0.2, -0.1, 4.0, 0.7],
            [0.6, 0.2, -0.3, 3.5],
        ],
        dtype=np.float64,
    )
    expected = np.asarray([0.3, -0.7, 1.1, 0.2])
    rows, columns = np.indices(dense.shape)
    matrix = BuildTriplet(
        dim=2,
        max_pairs_num=dense.size,
        max_nonzeros=4,
        max_active_nodes=2,
        symmetric=False,
        solver="BiCGSTAB",
        matrix_symmetric=False,
        device_reduction=False,
    )
    matrix.reset_system()
    matrix.assemble_scalar_triplets(
        rows.reshape(-1).astype(np.int32),
        columns.reshape(-1).astype(np.int32),
        dense.reshape(-1),
    )
    matrix.finalize_taichi_assembly()

    result = matrix.solve(
        rhs=(dense.T @ expected).reshape((2, 2)),
        active_nodes=2,
        tol=1.0e-12,
        maxiter=40,
        transpose=True,
    )

    assert result["converged"]
    np.testing.assert_allclose(result["x"].reshape(-1), expected, rtol=1.0e-11, atol=1.0e-12)


@pytest.mark.hash_triplet
def test_hash_triplet_bicgstab_rejects_false_recursive_convergence(
    taichi_runtime,
):
    dense = np.asarray([[1.0, -0.99999999], [-0.99999999, 1.0]], dtype=np.float64)
    rhs = np.asarray([1.0, 0.3], dtype=np.float64)
    rows, columns = np.indices(dense.shape)
    matrix = BuildTriplet(
        dim=1,
        max_pairs_num=dense.size,
        max_nonzeros=dense.size,
        max_active_nodes=2,
        symmetric=False,
        solver="BiCGSTAB",
        matrix_symmetric=False,
        device_reduction=False,
    )
    matrix.reset_system()
    matrix.assemble_scalar_triplets(
        rows.reshape(-1).astype(np.int32),
        columns.reshape(-1).astype(np.int32),
        dense.reshape(-1),
    )
    matrix.finalize_taichi_assembly()

    result = matrix.solve(
        rhs=rhs.reshape((2, 1)),
        active_nodes=2,
        tol=1.0e-12,
        maxiter=30,
    )
    actual_residual = np.linalg.norm(dense @ result["x"].reshape(-1) - rhs)

    assert not result["converged"]
    assert actual_residual > 1.0e-12
    assert result["residual"] > 1.0e-12
    matrix.matvec(2, int(matrix.non_diag.element_pair_num[0]), matrix.x, matrix.Ax)
    device_residual = np.linalg.norm(matrix.rhs.to_numpy() - matrix.Ax.to_numpy())
    assert result["residual"] == pytest.approx(device_residual, rel=2.0e-15)
    assert result["residual"] == pytest.approx(np.linalg.norm(matrix.r.to_numpy()), rel=2.0e-15)


@pytest.mark.hash_triplet
def test_hash_triplet_pcg_preserves_conjugacy_past_32_iterations(
    taichi_runtime,
):
    size = 96
    diagonal = np.arange(size, dtype=np.int32)
    upper = np.arange(size - 1, dtype=np.int32)
    matrix = BuildTriplet(
        dim=1,
        max_pairs_num=2 * size - 1,
        max_nonzeros=2 * size - 1,
        max_active_nodes=size,
        symmetric=False,
        solver="PCG",
        matrix_symmetric=True,
        device_reduction=False,
    )
    matrix.reset_system()
    matrix.assemble_scalar_triplets(
        np.concatenate((diagonal, upper)),
        np.concatenate((diagonal, upper + 1)),
        np.concatenate((np.full(size, 2.0), np.full(size - 1, -1.0))),
    )
    matrix.finalize_taichi_assembly()

    result = matrix.solve(
        rhs=np.ones((size, 1)),
        active_nodes=size,
        tol=1.0e-10,
        maxiter=size,
    )

    assert result["converged"]
    assert 32 < result["iterations"] <= size
    assert result["residual"] <= 1.0e-10


@pytest.mark.matrix_free
def test_bicgstab_nonzero_initial_guess_uses_actual_shadow_residual(
    taichi_runtime,
):
    matrix = np.asarray(
        [[4.0, -1.0, 0.5], [2.0, 5.0, -0.75], [0.25, 1.5, 3.0]],
        dtype=np.float64,
    )
    expected = np.asarray([0.75, -1.25, 2.0])
    initial = np.asarray([0.2, -0.1, 0.4])
    rhs = matrix @ expected
    initial_residual = rhs - matrix @ initial
    operator = _DenseOperator(matrix)
    b, x = _scalar_fields(rhs, initial)
    solver = MatrixFreeBICGSTAB(3)

    succeeded = solver.solve(operator, b, x, size=3, tol=1.0e-12, maxiter=30)

    assert succeeded
    assert solver.last_converged
    assert solver.last_breakdown_reason == ""
    assert solver.last_initial_residual == pytest.approx(np.linalg.norm(initial_residual), rel=2.0e-15)
    np.testing.assert_allclose(solver.r_tld.to_numpy(), initial_residual, rtol=2.0e-15, atol=1.0e-15)
    np.testing.assert_allclose(x.to_numpy(), expected, rtol=2.0e-12, atol=2.0e-12)
    assert np.linalg.norm(matrix @ x.to_numpy() - rhs) <= 1.0e-12


@pytest.mark.matrix_free
def test_preconditioned_bicgstab_reports_true_last_state(taichi_runtime):
    matrix = np.asarray(
        [[4.0, -1.0, 0.5], [2.0, 5.0, -0.75], [0.25, 1.5, 3.0]],
        dtype=np.float64,
    )
    expected = np.asarray([0.75, -1.25, 2.0])
    initial = np.asarray([0.2, -0.1, 0.4])
    rhs = matrix @ expected
    initial_residual = rhs - matrix @ initial
    operator = _DenseOperator(matrix)
    b, x = _scalar_fields(rhs, initial)
    diagonal = ti.field(dtype=ti.f64, shape=3)
    diagonal.from_numpy(np.abs(np.diag(matrix)))
    solver = MatrixFreePBICGSTAB(3)

    succeeded = solver.solve(
        operator,
        b,
        x,
        diagonal,
        size=3,
        tol=1.0e-12,
        maxiter=30,
    )

    assert succeeded
    assert solver.last_converged
    assert solver.last_breakdown_reason == ""
    assert solver.last_initial_residual == pytest.approx(np.linalg.norm(initial_residual), rel=2.0e-15)
    assert solver.last_residual <= 1.0e-12
    np.testing.assert_allclose(solver.r_tld.to_numpy(), initial_residual, rtol=2.0e-15, atol=1.0e-15)
    np.testing.assert_allclose(x.to_numpy(), expected, rtol=2.0e-12, atol=2.0e-12)
