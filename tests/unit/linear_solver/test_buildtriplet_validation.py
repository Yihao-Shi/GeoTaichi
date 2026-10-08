import numpy as np
import pytest
from scipy.sparse import csr_matrix
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


@pytest.mark.parametrize("dimension", (2, 3))
def test_pcg_reuses_dirty_workspace_with_nonzero_initial_guess(taichi_runtime, dimension):
    system = BuildTriplet(
        dim=dimension,
        max_pairs_num=8,
        max_nonzeros=8,
        max_active_nodes=8,
        symmetric=False,
        matrix_symmetric=True,
    )
    active_nodes = 5
    size = active_nodes * dimension
    matrix = np.diag(np.linspace(3.0, 5.0, size))
    for i in range(size - dimension):
        matrix[i, i + dimension] = matrix[i + dimension, i] = -0.5
    system.load_from_scipy_blocks(csr_matrix(matrix), active_nodes=active_nodes)
    rng = np.random.default_rng(42)
    for scale in (1.0, 1e-5, 2.0):
        for workspace in (system.r, system.z, system.p, system.Ap, system.Ax):
            workspace.fill(123.0)
        expected = scale * rng.normal(size=(active_nodes, dimension))
        rhs = (matrix @ expected.ravel()).reshape(active_nodes, dimension)
        initial = scale * rng.normal(size=(active_nodes, dimension))
        result = system.PCGSolver(rhs=rhs, x=initial, active_nodes=active_nodes, tol=1e-12, maxiter=100)
        np.testing.assert_allclose(result["x"], expected, rtol=1e-10, atol=1e-12)
        true_residual = np.linalg.norm(matrix @ result["x"].ravel() - rhs.ravel())
        assert result["converged"] and true_residual <= 1e-12
        assert result["residual"] == pytest.approx(true_residual, abs=2e-14)


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


@pytest.mark.parametrize("symmetric", [False, True])
def test_fixed_slots_survive_dynamic_pattern_rebuilds(taichi_runtime, symmetric):
    import taichi as ti

    system = BuildTriplet(
        dim=2,
        max_pairs_num=2,
        max_nonzeros=3,
        max_active_nodes=4,
        symmetric=False,
        matrix_symmetric=symmetric,
        pattern_cache_extra_fraction=0.0,
        pattern_cache_max_age=0,
    )
    system.install_fixed_pattern(np.array([[0, 1]], dtype=np.int32))

    @ti.kernel
    def assemble(other_i: ti.i32, other_j: ti.i32, value: ti.f64):
        system.add_fixed_block(0, 0, 1, ti.Matrix([[value, 0.2], [-0.3, value + 1.0]]))
        system.raw_non_diag_count[0] = 2
        system.non_diag.blockI[0] = 0
        system.non_diag.blockJ[0] = 1
        system.non_diag.blockH[0] = ti.Vector([0.5, 0.0, 0.0, 0.5])
        system.non_diag.blockI[1] = other_i
        system.non_diag.blockJ[1] = other_j
        system.non_diag.blockH[1] = ti.Vector([2.0, 0.0, 0.0, 3.0])

    for step, (i, j) in enumerate(((1, 2), (2, 3), (1, 3), (1, 2))):
        system.reset_system()
        assemble(i, j, float(step + 1))
        system.finalize_taichi_assembly()
        system.finalize_taichi_assembly()  # Querying a prepared matrix is idempotent.
        expected = np.zeros((8, 8))
        expected[:2, 2:4] = [[step + 1.5, 0.2], [-0.3, step + 2.5]]
        expected[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = np.diag([2.0, 3.0])
        if symmetric:
            expected += expected.T
        np.testing.assert_allclose(system.to_scipy(4).toarray(), expected, rtol=0, atol=1e-13)
        assert system.non_diag.tripletI[0] == 0 and system.non_diag.tripletJ[0] == 1
