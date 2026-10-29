"""Device recurrences preserve true residuals, active prefixes and error states."""

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.MatrixFreeCG import MatrixFreeCG
from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.linear_solver.MatrixFreeBICGSTAB import MatrixFreeBICGSTAB
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB
from src.linear_solver.LinearOperator import LinearOperator
from src.linear_solver.MatrixFreeKrylovKernel import reduce_shared
from src.linear_solver.MultiGridPCG_mixture import MGPCGMixPoissonSolver
from src.linear_solver.MultiGridPCG_mixture_Axi import MGPCGMixPoissonSolver_Axi


@ti.data_oriented
class PrefixOperator:
    def __init__(self, nonsymmetric=False):
        self.size = 0
        self.left = -0.6 if nonsymmetric else -0.9

    def matvec(self, x, result):
        self.multiply(self.size, x, result)

    @ti.kernel
    def multiply(self, size: int, x: ti.template(), result: ti.template()):
        for i in range(size):
            value = 2.1 * x[i]
            if i > 0:
                value += self.left * x[i - 1]
            if i + 1 < size:
                value -= 0.9 * x[i + 1]
            result[i] = value


@pytest.mark.parametrize(
    "solver_type,preconditioned,nonsymmetric",
    [
        (MatrixFreeCG, False, False),
        (MatrixFreePCG, True, False),
        (MatrixFreePCG, "operator", False),
        (MatrixFreeBICGSTAB, False, True),
        (MatrixFreePBICGSTAB, True, True),
    ],
)
def test_device_and_host_recurrences_match_on_reused_active_prefixes(
    taichi_runtime, solver_type, preconditioned, nonsymmetric
):
    capacity = 131
    solver = solver_type(capacity)
    fields = [ti.field(ti.f64, shape=capacity) for _ in range(3)]
    b, x, diagonal = fields
    diagonal.fill(2.1)
    preconditioner = LinearOperator(lambda src, dst: dst.copy_from(src)) if preconditioned == "operator" else diagonal
    operator = PrefixOperator(nonsymmetric)
    rng = np.random.default_rng(17)
    for size in (3, 65, 31, 64, 131):
        operator.size = size
        matrix = 2.1 * np.eye(size) + operator.left * np.eye(size, k=-1) - 0.9 * np.eye(size, k=1)
        rhs = rng.normal(size=size)
        reference = np.linalg.solve(matrix, rhs)
        initial = rng.normal(size=capacity)
        b.from_numpy(np.pad(rhs, (0, capacity - size), constant_values=1.0e9))
        for device in (False, True):
            solver.device_execution = device
            x.from_numpy(initial)
            args = (operator, b, x, preconditioner, size) if preconditioned else (operator, b, x, size)
            assert solver.solve(*args, tol=1e-10, rel_tol=1e-11, maxiter=300)
            solution = x.to_numpy()
            np.testing.assert_array_equal(solution[size:], initial[size:])
            np.testing.assert_allclose(solution[:size], reference, atol=2e-10, rtol=2e-10)
            actual = np.linalg.norm(matrix @ solution[:size] - rhs)
            assert actual <= max(1e-10, 1e-11 * solver.last_initial_residual) * (1 + 1e-4)
            assert solver.last_residual == pytest.approx(np.linalg.norm(solver.r.to_numpy()[:size]), abs=1e-13)
            if device:
                reads = solver._device_solver.status_reads
                per_iteration = 2 if nonsymmetric else 1
                assert reads <= per_iteration * solver.last_iterations + solver.last_residual_restarts + 4


@pytest.mark.parametrize("solver_type", [MatrixFreeCG, MatrixFreePCG, MatrixFreeBICGSTAB, MatrixFreePBICGSTAB])
def test_device_breakdown_and_maxiter_never_claim_convergence(taichi_runtime, solver_type):
    size = 3
    solver = solver_type(size)
    solver.device_execution = True
    b, x, diagonal = [ti.field(ti.f64, shape=size) for _ in range(3)]
    b.fill(1.0)
    x.fill(0.0)
    diagonal.fill(1.0)
    operator = PrefixOperator()
    operator.size = size
    args = (
        (operator, b, x, diagonal, size)
        if solver_type in (MatrixFreePCG, MatrixFreePBICGSTAB)
        else (operator, b, x, size)
    )
    assert not solver.solve(*args, tol=1e-15, maxiter=0)
    assert solver.last_breakdown_reason == "maximum_iterations"
    assert solver.last_residual == pytest.approx(np.sqrt(3.0), rel=1e-14)
    operator.matvec = lambda src, dst: dst.fill(0.0)
    assert not solver.solve(*args, tol=1e-15, maxiter=3)
    assert solver.last_breakdown_reason in ("non_positive_curvature", "zero_alpha_denominator")
    np.testing.assert_array_equal(x.to_numpy(), np.zeros(size))


def test_reduction_handles_partial_blocks_in_double_precision(taichi_runtime):
    a, b = [ti.field(ti.f64, shape=131) for _ in range(2)]
    rng = np.random.default_rng(43)
    left, right = rng.normal(size=(2, 131))
    a.from_numpy(left)
    b.from_numpy(right)
    for size in (1, 3, 63, 64, 65, 127, 131):
        assert reduce_shared(size, a, b) == pytest.approx(left[:size] @ right[:size], rel=1e-13, abs=1e-13)


@pytest.mark.parametrize(
    "dimension,axisymmetric,sign", [(2, False, 1), (3, False, 1), (2, True, 1), (2, False, -1), (2, True, -1)]
)
def test_mixture_multigrid_device_dispatch_retains_its_operator(taichi_runtime, dimension, axisymmetric, sign):
    shape = (8,) * dimension
    cls = MGPCGMixPoissonSolver_Axi if axisymmetric else MGPCGMixPoissonSolver
    kwargs = {"dr": 0.1} if axisymmetric else {}
    solver = cls(dimension, shape, n_mg_levels=2, bottom_smoothing=10, **kwargs)
    for level in range(2):
        solver.grid_type[level].fill(solver.FLUID)
        solver.Adiag[level].fill(sign * (2.0 * dimension + 1.0))
        positions = np.indices(solver.Adiag[level].shape)
        links = sign * np.stack([-0.2 - 0.01 * positions[d] for d in range(dimension)], axis=-1)
        solver.Ax[level].from_numpy(links)
        if axisymmetric:
            negative = np.stack([np.roll(links[..., d], 1, axis=d) for d in range(dimension)], axis=-1)
            solver.Ax_neg[level].from_numpy(negative)
    matrix = np.eye(int(np.prod(shape))) * sign * (2.0 * dimension + 1.0)
    for index in np.ndindex(shape):
        row = np.ravel_multi_index(index, shape)
        for d in range(dimension):
            neighbor = list(index)
            neighbor[d] += 1
            if neighbor[d] < shape[d]:
                col = np.ravel_multi_index(tuple(neighbor), shape)
                matrix[row, col] = matrix[col, row] = sign * (-0.2 - 0.01 * index[d])
    rng = np.random.default_rng(13)
    for repeat in range(2):
        rhs = rng.normal(size=shape)
        expected = np.linalg.solve(matrix, rhs.ravel()).reshape(shape)
        for graph in (False, True):
            solver.graph_dispatch = graph
            solver.b.from_numpy(rhs)
            solver.x.fill(0.0)
            for z in solver.z:
                z.fill(7.0)
            assert solver.solve(200, rel_tol=1e-20, abs_tol=1e-26, eps=0.0), solver.breakdown_reason
            actual = solver.x.to_numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
            residual = rhs.ravel() - matrix @ actual.ravel()
            assert residual @ residual <= solver.initial_residual * 1e-20 * (1 + 1e-4)
