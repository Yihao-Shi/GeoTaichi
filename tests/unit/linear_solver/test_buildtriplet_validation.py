import numpy as np
import pytest
from scipy.sparse.linalg import bicgstab, cg

from src.linear_solver.BuildTriplet import BuildTriplet

pytestmark = [pytest.mark.unit, pytest.mark.linear_solver, pytest.mark.cpu]


def _assert_matches_scipy(system, result, reference, info, rhs):
    relative_error = np.linalg.norm(result["x"].reshape(-1) - reference) / max(np.linalg.norm(reference), 1.0e-30)
    matrix = system.to_scipy(rhs.shape[0])
    residual = np.linalg.norm(matrix @ result["x"].reshape(-1) - rhs.reshape(-1))
    assert result["converged"]
    assert info == 0
    assert relative_error < 1.0e-7
    assert np.isfinite(residual)


@pytest.mark.parametrize("dimension", (2, 3))
def test_buildtriplet_pcg_matches_scipy(taichi_runtime, dimension):
    system = BuildTriplet(
        dim=dimension,
        max_pairs_num=512,
        max_nonzeros=512,
        max_active_nodes=32,
        symmetric=True,
    )
    rng = np.random.default_rng(0)
    active_nodes = 16
    diagonal, block_i, block_j, block_h = system._make_symmetric_spd_case(rng, active_nodes, 64)
    system.diag.from_numpy(diagonal)
    system._set_reduced_non_diag(block_i, block_j, block_h)
    rhs = rng.normal(size=(active_nodes, dimension))
    result = system.PCGSolver(rhs=rhs, active_nodes=active_nodes, tol=1.0e-9, maxiter=500)
    reference, info = cg(system.to_scipy(active_nodes), rhs.reshape(-1), rtol=1.0e-9, atol=0.0, maxiter=500)
    _assert_matches_scipy(system, result, reference, info, rhs)


@pytest.mark.parametrize("dimension", (2, 3, 4))
def test_buildtriplet_bicgstab_matches_scipy(taichi_runtime, dimension):
    system = BuildTriplet(
        dim=dimension,
        max_pairs_num=512,
        max_nonzeros=512,
        max_active_nodes=32,
        symmetric=False,
    )
    rng = np.random.default_rng(0)
    active_nodes = 16
    diagonal, block_i, block_j, block_h = system._make_nonsymmetric_case(rng, active_nodes, 96)
    system.diag.from_numpy(diagonal)
    system._set_reduced_non_diag(block_i, block_j, block_h)
    rhs = rng.normal(size=(active_nodes, dimension))
    result = system.BiCGSTABSolver(rhs=rhs, active_nodes=active_nodes, tol=1.0e-9, maxiter=500)
    reference, info = bicgstab(system.to_scipy(active_nodes), rhs.reshape(-1), rtol=1.0e-9, atol=0.0, maxiter=500)
    _assert_matches_scipy(system, result, reference, info, rhs)
