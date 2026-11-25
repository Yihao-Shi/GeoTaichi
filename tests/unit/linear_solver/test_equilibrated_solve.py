import numpy as np
import pytest
from scipy.sparse import csr_matrix
import taichi as ti

from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.EquilibratedSolve import EquilibratedSolve
from src.utils.errors import NonfiniteStateError


@pytest.mark.parametrize("relative_tolerance", [0.0, 1e-8])
def test_equilibrated_solve_preserves_physical_matrix_and_true_residual(
    taichi_runtime, monkeypatch, relative_tolerance
):
    matrix = BuildTriplet(
        dim=3,
        max_pairs_num=64,
        max_nonzeros=64,
        max_active_nodes=6,
        symmetric=False,
        solver="BiCGSTAB",
        matrix_symmetric=False,
        preconditioner_group_size=4,
        preconditioner_group_nodes=4,
    )
    solver = EquilibratedSolve(matrix)
    scaled = solver.scaled_matrix
    assert scaled.non_diag.tripletI is matrix.non_diag.tripletI
    assert scaled.non_diag.tripletJ is matrix.non_diag.tripletJ
    assert scaled.non_diag.tripletH is not matrix.non_diag.tripletH
    assert not hasattr(scaled.non_diag, "blockH")
    rhs, solution = ti.field(ti.f64, shape=18), ti.field(ti.f64, shape=18)
    rng = np.random.default_rng(8061)
    for nodes in (5, 4, 5):
        n = 3 * nodes
        diagonal = np.logspace(-3, 3, n)
        core = np.eye(n) + 0.03 * rng.normal(size=(n, n))
        physical = diagonal[:, None] * core * diagonal[None, :]
        expected = rng.normal(size=n) / diagonal
        force = physical @ expected
        matrix.load_from_scipy_blocks(csr_matrix(physical), nodes)
        before = matrix.to_scipy(nodes).toarray()
        rhs.from_numpy(np.pad(force, (0, 18 - n)))
        with monkeypatch.context() as patch:
            patch.setattr(BuildTriplet, "to_scipy", lambda *_: pytest.fail("solve crossed the host matrix boundary"))
            result = solver.solve(rhs, solution, active_nodes=nodes, tol=1e-9, rel_tol=relative_tolerance, maxiter=200)
        actual = solution.to_numpy()[:n]
        residual = np.linalg.norm(force - physical @ actual)
        assert result["converged"] and result["equilibrated"]
        assert residual <= max(1e-9, relative_tolerance * np.linalg.norm(force))
        assert result["residual"] == pytest.approx(residual, abs=1e-11)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(matrix.to_scipy(nodes).toarray(), before)
        assert scaled._row_entries is matrix._row_entries
    rejected = solver.solve(rhs, solution, active_nodes=nodes, tol=1e-12, maxiter=0)
    assert not rejected["converged"]
    assert rejected["residual"] == pytest.approx(np.linalg.norm(force))
    original_solve = solver.scaled_matrix.solve_flat_system
    calls = []

    def premature_convergence(*args, **kwargs):
        calls.append(kwargs["tol"])
        if len(calls) == 1:
            solver.scaled_solution.fill(0.0)
            return dict(converged=True, residual=1e-30, iterations=1, convergence_tolerance=kwargs["tol"])
        return original_solve(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(solver.scaled_matrix, "solve_flat_system", premature_convergence)
        refined = solver.solve(rhs, solution, active_nodes=nodes, tol=1e-9, maxiter=200)
    assert refined["converged"] and refined["equilibration_refinements"] >= 1
    assert np.linalg.norm(force - physical @ solution.to_numpy()[:n]) <= 1e-9
    with pytest.raises(ValueError, match="tolerances"):
        solver.solve(rhs, solution, active_nodes=nodes, tol=1e-12, rel_tol=-1.0, maxiter=1)
    with pytest.raises(ValueError, match="maxiter"):
        solver.solve(rhs, solution, active_nodes=nodes, tol=1e-12, maxiter=-1)
    rhs[0] = float("nan")
    with pytest.raises(NonfiniteStateError):
        solver.solve(rhs, solution, active_nodes=nodes, tol=1e-12, maxiter=1)
