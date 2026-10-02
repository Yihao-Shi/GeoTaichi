import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from src.linear_solver.ScipyKrylov import checked_scipy_krylov


def test_checked_scipy_krylov_rejects_false_convergence():
    matrix = csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    rhs = np.array([1.0, 0.0])

    def false_solver(_matrix, _rhs, **_kwargs):
        return np.zeros(2), 0

    with pytest.raises(RuntimeError, match="true-residual verification"):
        checked_scipy_krylov(false_solver, matrix, rhs, rtol=1.0e-12)

    solution = checked_scipy_krylov(cg, matrix, rhs, rtol=1.0e-12, atol=0.0)
    assert np.linalg.norm(rhs - matrix @ solution) <= 1.0e-12 * np.linalg.norm(rhs)
