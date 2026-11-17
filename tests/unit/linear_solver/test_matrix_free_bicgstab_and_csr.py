"""Oracles extracted from the legacy print-only linear-solver probes."""

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.CompressedSparseRow import CompressedSparseRow
from src.linear_solver.LinearOperator import LinearOperator
from src.linear_solver.MatrixFreeBICGSTAB import MatrixFreeBICGSTAB


pytestmark = [
    pytest.mark.cpu,
    pytest.mark.linear_solver,
]


_MATRIX = np.asarray(
    [[4.0, -1.0, 0.5], [2.0, 5.0, -0.75], [0.25, 1.5, 3.0]],
    dtype=np.float64,
)


@ti.kernel
def _nonsymmetric_matvec(vector: ti.template(), result: ti.template()):
    result[0] = 4.0 * vector[0] - vector[1] + 0.5 * vector[2]
    result[1] = 2.0 * vector[0] + 5.0 * vector[1] - 0.75 * vector[2]
    result[2] = 0.25 * vector[0] + 1.5 * vector[1] + 3.0 * vector[2]


@pytest.mark.matrix_free
def test_matrix_free_bicgstab_matches_dense_oracle(taichi_runtime):
    expected = np.asarray([0.75, -1.25, 2.0])
    rhs = _MATRIX @ expected
    rhs_field = ti.field(ti.f64, shape=3)
    solution = ti.field(ti.f64, shape=3)
    rhs_field.from_numpy(rhs)
    solution.fill(0.0)
    solver = MatrixFreeBICGSTAB(3)

    succeeded = solver.solve(
        LinearOperator(_nonsymmetric_matvec),
        rhs_field,
        solution,
        size=3,
        tol=1.0e-12,
        maxiter=30,
    )

    assert succeeded
    np.testing.assert_allclose(
        solution.to_numpy(), expected, rtol=1.0e-11, atol=1.0e-11
    )
    np.testing.assert_allclose(
        _MATRIX @ solution.to_numpy(), rhs, rtol=1.0e-11, atol=1.0e-11
    )


def test_compressed_sparse_row_cpu_solve_matches_numpy(taichi_runtime):
    matrix = np.asarray(
        [[4.0, 1.0, 0.0], [1.0, 3.0, -0.5], [0.0, -0.5, 2.0]],
        dtype=np.float64,
    )
    expected = np.asarray([1.25, -0.5, 2.0])
    rhs = matrix @ expected
    sparse = CompressedSparseRow(
        nonzeros=np.count_nonzero(matrix),
        degree_of_freedom=matrix.shape[0],
        preconditioned=True,
        symmetry=True,
    )

    sparse._from_numpy(matrix)
    actual = sparse.spsolve(rhs)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)
