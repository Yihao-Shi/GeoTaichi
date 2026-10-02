"""Small dense-matrix oracles for Taichi matrix helpers."""

import numpy as np
import pytest
import taichi as ti

from src.utils.MatrixFunction import get_eigenvalue_3x3, get_jacobian_inverse5


pytestmark = [pytest.mark.cpu, pytest.mark.linear_solver]


@ti.kernel
def _inverse_5x5(
    source: ti.types.ndarray(dtype=ti.f64, ndim=2),
    destination: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    matrix = ti.Matrix.zero(ti.f64, 5, 5)
    row = 0
    while row < 5:
        column = 0
        while column < 5:
            matrix[row, column] = source[row, column]
            column += 1
        row += 1
    inverse = get_jacobian_inverse5(matrix)
    row = 0
    while row < 5:
        column = 0
        while column < 5:
            destination[row, column] = inverse[row, column]
            column += 1
        row += 1


@ti.kernel
def _symmetric_eigenvalues(
    source: ti.types.ndarray(dtype=ti.f64, ndim=2),
    destination: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    matrix = ti.Matrix.zero(ti.f64, 3, 3)
    for row, column in ti.static(ti.ndrange(3, 3)):
        matrix[row, column] = source[row, column]
    eigenvalues = get_eigenvalue_3x3(matrix)
    for index in ti.static(range(3)):
        destination[index] = eigenvalues[index]


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(
            [
                [6.0, 1.0, -0.5, 0.25, 0.75],
                [0.4, 5.0, 0.8, -0.2, 0.1],
                [-0.3, 0.6, 4.0, 0.5, -0.4],
                [0.2, -0.1, 0.7, 3.5, 0.9],
                [0.5, 0.2, -0.6, 0.4, 4.5],
            ]
        ),
        np.asarray(
            [
                [0.0, 2.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.5, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.0],
                [1.0, 0.0, 0.0, 6.0, 0.0],
            ]
        ),
    ],
    ids=("diagonally-dominant", "requires-row-pivoting"),
)
def test_explicit_five_by_five_inverse_matches_numpy(taichi_runtime, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    actual = np.zeros_like(matrix)

    _inverse_5x5(matrix, actual)

    np.testing.assert_allclose(
        actual, np.linalg.inv(matrix), rtol=2.0e-12, atol=2.0e-12
    )
    np.testing.assert_allclose(
        actual @ matrix, np.eye(5), rtol=2.0e-12, atol=2.0e-12
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray(
            [[4.0, 0.8, -0.2], [0.8, 2.5, 0.4], [-0.2, 0.4, 1.25]]
        ),
        np.diag([3.0, 3.0, 3.0]),
    ],
    ids=("distinct", "isotropic"),
)
def test_symmetric_eigenvalues_match_numpy(taichi_runtime, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    actual = np.zeros(3, dtype=np.float64)

    _symmetric_eigenvalues(matrix, actual)

    np.testing.assert_allclose(
        np.sort(actual), np.linalg.eigvalsh(matrix), rtol=3.0e-7, atol=3.0e-7
    )
