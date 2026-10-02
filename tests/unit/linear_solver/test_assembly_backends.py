"""Numerical contracts shared by the implicit assembly backends.

These tests deliberately use a two-node, two-degree-of-freedom system.  The
system is small enough to compare against a dense NumPy oracle while still
distinguishing scalar diagonal Jacobi from node-block Jacobi.
"""

from typing import Tuple

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.linear_solver.MatrixFreePCG import MatrixFreePCG


pytestmark = [
    pytest.mark.assembly,
    pytest.mark.linear_solver,
    pytest.mark.cpu,
]


@ti.data_oriented
class _DenseReferenceOperator:
    """Matrix-free operator used only as a small independent test oracle."""

    def __init__(self, matrix: np.ndarray):
        self.size = int(matrix.shape[0])
        self.matrix = ti.field(dtype=ti.f64, shape=matrix.shape)
        self.matrix.from_numpy(np.asarray(matrix, dtype=np.float64))

    def matvec(self, vector, result) -> None:
        self._matvec(vector, result)

    @ti.kernel
    def _matvec(self, vector: ti.template(), result: ti.template()):
        for row in range(self.size):
            value = 0.0
            for column in range(self.size):
                value += self.matrix[row, column] * vector[column]
            result[row] = value


def _scalar_triplets(matrix: np.ndarray):
    rows, columns = np.nonzero(matrix)
    values = matrix[rows, columns]
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
    )


def _solve_matrix_free(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    operator = _DenseReferenceOperator(matrix)
    size = int(rhs.size)
    rhs_field = ti.field(dtype=ti.f64, shape=size)
    solution = ti.field(dtype=ti.f64, shape=size)
    diagonal = ti.field(dtype=ti.f64, shape=size)
    rhs_field.from_numpy(np.asarray(rhs, dtype=np.float64))
    solution.fill(0.0)
    diagonal.from_numpy(np.diag(matrix).astype(np.float64))
    solver = MatrixFreePCG(size)
    solver.solve(
        operator,
        rhs_field,
        solution,
        diagonal,
        size,
        tol=1.0e-12,
        maxiter=100,
    )
    return solution.to_numpy()


def _solve_coo(matrix: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows, columns, values = _scalar_triplets(matrix)
    sparse = CoordinateSparseMatrix(
        nonzeros=values.size,
        degree_of_freedom=rhs.size,
        preconditioned=True,
        symmetry=True,
    )
    sparse.rows.from_numpy(rows)
    sparse.cols.from_numpy(columns)
    sparse.data.from_numpy(values)
    sparse.linear_operator.update_nnz(values.size)

    rhs_field = ti.field(dtype=ti.f64, shape=rhs.size)
    solution = ti.field(dtype=ti.f64, shape=rhs.size)
    diagonal = ti.field(dtype=ti.f64, shape=rhs.size)
    rhs_field.from_numpy(np.asarray(rhs, dtype=np.float64))
    solution.fill(0.0)
    diagonal.from_numpy(np.diag(matrix).astype(np.float64))
    sparse.solve(
        rhs_field,
        solution,
        diagonal,
        tol=1.0e-12,
        maxiter=100,
    )
    return sparse._to_scipy().toarray(), solution.to_numpy()


def _solve_hash_triplet(
    matrix: np.ndarray, rhs: np.ndarray, *, tol: float = 1.0e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rows, columns, values = _scalar_triplets(matrix)
    sparse = BuildTriplet(
        dim=2,
        max_pairs_num=max(1, values.size),
        max_nonzeros=max(1, values.size),
        max_active_nodes=2,
        symmetric=False,
        solver="PCG",
        matrix_symmetric=False,
        device_reduction=False,
    )
    sparse.reset_system()
    sparse.assemble_scalar_triplets(rows, columns, values)
    sparse.finalize_taichi_assembly()
    result = sparse.solve(
        rhs=np.asarray(rhs, dtype=np.float64).reshape((2, 2)),
        active_nodes=2,
        tol=tol,
        maxiter=100,
    )
    block_inverse = sparse.diag_inverse.to_numpy().reshape((2, 2, 2))
    return (
        sparse.to_scipy(active_nodes=2).toarray(),
        result["x"],
        block_inverse,
        result,
    )


@pytest.mark.matrix_free
@pytest.mark.coo
@pytest.mark.hash_triplet
def test_matrix_free_coo_and_hash_triplet_match_dense_oracle(taichi_runtime):
    """All assembly representations must produce the same linear solution."""

    matrix = np.asarray(
        [
            [5.0, 1.2, -0.8, 0.1],
            [1.2, 4.0, 0.2, -0.5],
            [-0.8, 0.2, 3.5, 0.6],
            [0.1, -0.5, 0.6, 2.8],
        ],
        dtype=np.float64,
    )
    expected_solution = np.asarray([0.4, -0.7, 1.1, 0.25])
    rhs = matrix @ expected_solution

    matrix_free_solution = _solve_matrix_free(matrix, rhs)
    coo_matrix, coo_solution = _solve_coo(matrix, rhs)
    hash_matrix, hash_solution, _, _ = _solve_hash_triplet(matrix, rhs)

    assert np.allclose(coo_matrix, matrix, rtol=0.0, atol=1.0e-14)
    assert np.allclose(hash_matrix, matrix, rtol=0.0, atol=1.0e-14)
    assert np.allclose(matrix_free_solution, expected_solution, rtol=1.0e-11)
    assert np.allclose(coo_solution, expected_solution, rtol=1.0e-11)
    assert np.allclose(
        np.asarray(hash_solution).reshape(-1),
        expected_solution,
        rtol=1.0e-11,
    )


@pytest.mark.hash_triplet
def test_hash_triplet_uses_full_node_block_jacobi(taichi_runtime):
    """HashTriplet must retain coupling inside every node diagonal block."""

    matrix = np.asarray(
        [
            [5.0, 1.2, -0.8, 0.1],
            [1.2, 4.0, 0.2, -0.5],
            [-0.8, 0.2, 3.5, 0.6],
            [0.1, -0.5, 0.6, 2.8],
        ],
        dtype=np.float64,
    )
    rhs = matrix @ np.asarray([0.4, -0.7, 1.1, 0.25])
    _, _, block_inverse, _ = _solve_hash_triplet(matrix, rhs)

    expected = np.stack(
        (np.linalg.inv(matrix[:2, :2]), np.linalg.inv(matrix[2:, 2:]))
    )
    diagonal_only = np.stack(
        (
            np.diag(1.0 / np.diag(matrix[:2, :2])),
            np.diag(1.0 / np.diag(matrix[2:, 2:])),
        )
    )
    assert np.allclose(block_inverse, expected, rtol=1.0e-12, atol=1.0e-12)
    assert not np.allclose(block_inverse, diagonal_only)


@pytest.mark.hash_triplet
def test_hash_triplet_pcg_absolute_tolerance_uses_true_residual(
    taichi_runtime,
):
    """A large Jacobi scale must not make a nonzero residual look converged.

    ``sqrt(r.T @ M^-1 @ r)`` changes when the same linear system is scaled and
    therefore cannot be compared directly with the solver's absolute residual
    tolerance.  This high-stiffness diagonal system makes that preconditioned
    quantity smaller than ``tol`` even though ``||r||_2`` is four orders of
    magnitude larger.
    """

    diagonal = 4.0e9 * np.arange(1.0, 5.0)
    matrix = np.diag(diagonal)
    rhs = 1.0e-5 * np.asarray([1.0, -2.0, 3.0, -4.0])
    expected_solution = rhs / diagonal

    assembled, solution, _, diagnostics = _solve_hash_triplet(
        matrix, rhs, tol=1.0e-9
    )
    solution = np.asarray(solution).reshape(-1)
    true_residual = np.linalg.norm(assembled @ solution - rhs)

    assert np.linalg.norm(rhs) > 1.0e4 * 1.0e-9
    assert np.allclose(
        solution,
        expected_solution,
        rtol=1.0e-12,
        atol=1.0e-27,
    )
    assert diagnostics["converged"]
    assert diagnostics["iterations"] == 1
    assert diagnostics["residual"] == pytest.approx(
        true_residual, rel=1.0e-12, abs=1.0e-30
    )


@pytest.mark.coo
def test_coo_sums_duplicate_scalar_triplets(taichi_runtime):
    """COO permits duplicate scalar entries; conversion must add them."""

    sparse = CoordinateSparseMatrix(
        nonzeros=5,
        degree_of_freedom=2,
        preconditioned=False,
        symmetry=True,
        linear_solver=False,
    )
    sparse.rows.from_numpy(np.asarray([0, 0, 0, 1, 1], dtype=np.int32))
    sparse.cols.from_numpy(np.asarray([0, 1, 1, 0, 1], dtype=np.int32))
    sparse.data.from_numpy(np.asarray([4.0, 0.25, 0.75, 1.0, 3.0]))
    assert np.array_equal(
        sparse._to_scipy().toarray(),
        np.asarray([[4.0, 1.0], [1.0, 3.0]]),
    )


@pytest.mark.hash_triplet
def test_hash_triplet_raw_only_source_skips_reduced_workspace(
    taichi_runtime,
):
    source = BuildTriplet(
        dim=3,
        max_pairs_num=32,
        max_nonzeros=32,
        max_active_nodes=4,
        symmetric=False,
        raw_only=True,
    )
    assert source.non_diag.max_pairs_num == 32
    assert source.max_nonzeros == 1
    with pytest.raises(RuntimeError, match="raw-only"):
        source.finalize_taichi_assembly()


@pytest.mark.hash_triplet
def test_hash_triplet_rejects_oversized_device_hash_before_allocation(
    taichi_runtime,
):
    with pytest.raises(ValueError, match="hash-table capacity"):
        BuildTriplet(
            dim=1,
            max_pairs_num=1,
            max_nonzeros=(1 << 29) + 1,
            max_active_nodes=1,
            symmetric=False,
        )
