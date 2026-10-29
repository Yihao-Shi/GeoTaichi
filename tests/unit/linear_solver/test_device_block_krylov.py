"""Block Jacobi and transpose semantics survive device scalar recurrences."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.linear_solver.BuildTriplet import BuildTriplet


@pytest.mark.parametrize("dimension", [2, 3, 4])
@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_device_block_solve_reuses_transpose_and_initial_guess(taichi_runtime, dimension, nonsymmetric):
    nodes = 5
    size = nodes * dimension
    rng = np.random.default_rng(91)
    matrix = rng.normal(scale=0.05, size=(size, size))
    if not nonsymmetric:
        matrix = matrix + matrix.T
    matrix += 3.0 * np.eye(size)
    system = BuildTriplet(
        dim=dimension,
        max_pairs_num=nodes**2,
        max_nonzeros=nodes**2,
        max_active_nodes=nodes,
        symmetric=False,
        matrix_symmetric=not nonsymmetric,
        solver="BiCGSTAB" if nonsymmetric else "PCG",
    )
    system.load_from_scipy_blocks(csr_matrix(matrix), active_nodes=nodes)
    for transpose in [False, True, False] if nonsymmetric else [False]:
        operator = matrix.T if transpose else matrix
        expected = rng.normal(size=size)
        rhs = operator @ expected
        initial = rng.normal(size=(nodes, dimension))
        for device in (False, True):
            system.device_execution = device
            system.rhs.from_numpy(rhs.reshape(nodes, dimension))
            method = system.BiCGSTABSolver if nonsymmetric else system.PCGSolver
            options = {"transpose": transpose} if nonsymmetric else {}
            result = method(x=initial, active_nodes=nodes, tol=1e-11, maxiter=100, return_solution=False, **options)
            assert result["converged"], result
            actual = system.x.to_numpy().reshape(-1)
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
            residual = np.linalg.norm(operator @ actual - rhs)
            assert residual <= 1e-11 * (1 + 1e-4)
            assert result["residual"] == pytest.approx(residual, abs=1e-13)


def test_device_block_bicgstab_retains_conjugacy_past_32_iterations(taichi_runtime):
    size = 128
    matrix = 2.0 * np.eye(size) - 1.01 * np.eye(size, k=-1) - 0.99 * np.eye(size, k=1)
    system = BuildTriplet(
        dim=1,
        max_pairs_num=3 * size,
        max_nonzeros=3 * size,
        max_active_nodes=size,
        symmetric=False,
        matrix_symmetric=False,
        solver="BiCGSTAB",
    )
    system.load_from_scipy_blocks(csr_matrix(matrix), active_nodes=size)
    system.device_execution = True
    system.rhs.from_numpy(np.ones((size, 1)))
    first = system.solve(tol=1e-14, maxiter=33, return_solution=False)
    assert first["iterations"] == 33 and not first["converged"]
    assert system.last_residual_restarts == 0
    actual = system.x.to_numpy().ravel()
    assert first["residual"] == pytest.approx(np.linalg.norm(matrix @ actual - 1), abs=1e-12)
    final = system.solve(tol=1e-8, maxiter=200, return_solution=False)
    assert final["converged"]
    assert np.linalg.norm(matrix @ system.x.to_numpy().ravel() - 1) <= 1e-8
