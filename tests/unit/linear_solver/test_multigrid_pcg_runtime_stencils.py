"""Reference checks for runtime-enumerated 3D multigrid stencils."""

from itertools import product
from math import ceil

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.MultiGridPCG import MGPCGPoissonSolver

pytestmark = [
    pytest.mark.unit,
    pytest.mark.linear_solver,
    pytest.mark.cpu,
    pytest.mark.serial,
]


def setup_module():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)


def teardown_module():
    ti.reset()


def _coarse_shape(shape):
    return tuple(ceil(size / 2) for size in shape)


def _inside(index, shape):
    return all(0 <= index[d] < shape[d] for d in range(3))


def _apply_stencil(diagonal, positive_axis, values, index):
    result = diagonal[index] * values[index]
    for axis in range(3):
        left = list(index)
        left[axis] -= 1
        left = tuple(left)
        if _inside(left, values.shape):
            result += positive_axis[left + (axis,)] * values[left]

        right = list(index)
        right[axis] += 1
        right = tuple(right)
        if _inside(right, values.shape):
            result += positive_axis[index + (axis,)] * values[right]
    return result


def test_3d_grid_type_restriction_matches_runtime_block_reference():
    shape = (5, 4, 3)
    solver = MGPCGPoissonSolver(3, shape, n_mg_levels=2)
    fine = np.full(shape, solver.SOLID, dtype=np.int32)
    fine[2, 0, 0] = solver.FLUID
    fine[4, 0, 0] = solver.AIR
    fine[0, 2, 0] = solver.FLUID
    fine[1, 3, 1] = solver.AIR
    fine[3, 2, 2] = solver.FLUID
    solver.grid_type[0].from_numpy(fine)

    solver.init_gridtype(solver.grid_type[0], solver.grid_type[1])

    expected = np.empty(_coarse_shape(shape), dtype=np.int32)
    for coarse in np.ndindex(expected.shape):
        attributes = []
        for offset in product((0, 1), repeat=3):
            index = tuple(2 * coarse[d] + offset[d] for d in range(3))
            if _inside(index, shape):
                attributes.append(fine[index])
        if solver.AIR in attributes:
            expected[coarse] = solver.AIR
        elif solver.FLUID in attributes:
            expected[coarse] = solver.FLUID
        else:
            expected[coarse] = solver.SOLID

    np.testing.assert_array_equal(solver.grid_type[1].to_numpy(), expected)


def test_3d_restriction_prolongation_and_jacobi_match_numpy_reference():
    shape = (5, 4, 3)
    solver = MGPCGPoissonSolver(3, shape, n_mg_levels=2, smoother="jacobi")
    coordinates = np.indices(shape, dtype=np.float64)
    rhs = 0.3 + 0.7 * coordinates[0] - 0.2 * coordinates[1] + 0.11 * coordinates[2]
    values = -0.4 + 0.13 * coordinates[0] + 0.17 * coordinates[1] - 0.09 * coordinates[2]
    diagonal = 4.5 + 0.05 * sum(coordinates)
    positive_axis = np.empty(shape + (3,), dtype=np.float64)
    positive_axis[..., 0] = -0.31 - 0.01 * coordinates[0]
    positive_axis[..., 1] = -0.27 - 0.02 * coordinates[1]
    positive_axis[..., 2] = -0.23 - 0.03 * coordinates[2]
    grid_type = np.full(shape, solver.FLUID, dtype=np.int32)
    grid_type[0, 0, 0] = solver.SOLID
    grid_type[4, 3, 2] = solver.AIR

    solver.r[0].from_numpy(rhs)
    solver.z[0].from_numpy(values)
    solver.Adiag[0].from_numpy(diagonal)
    solver.Ax[0].from_numpy(positive_axis)
    solver.grid_type[0].from_numpy(grid_type)
    solver.restrict_full_weighting(0)

    residual = np.empty(shape, dtype=np.float64)
    for index in np.ndindex(shape):
        residual[index] = rhs[index] - _apply_stencil(diagonal, positive_axis, values, index)
    expected_coarse = np.zeros(_coarse_shape(shape), dtype=np.float64)
    for coarse in np.ndindex(expected_coarse.shape):
        for offset in product((-1, 0, 1), repeat=3):
            fine = tuple(2 * coarse[d] + offset[d] for d in range(3))
            if _inside(fine, shape) and grid_type[fine] == solver.FLUID:
                weight = np.prod([0.5 if component == 0 else 0.25 for component in offset])
                expected_coarse[coarse] += weight * residual[fine]
    np.testing.assert_allclose(solver.r[1].to_numpy(), expected_coarse, rtol=0.0, atol=2.0e-14)

    coarse_correction = np.arange(np.prod(expected_coarse.shape), dtype=np.float64).reshape(expected_coarse.shape) / 7.0
    solver.z[0].from_numpy(values)
    solver.z[1].from_numpy(coarse_correction)
    solver.prolongate(0)
    expected_fine = values.copy()
    for index in np.ndindex(shape):
        coarse = tuple(index[d] // 2 for d in range(3))
        expected_fine[index] += coarse_correction[coarse]
    np.testing.assert_allclose(solver.z[0].to_numpy(), expected_fine, rtol=0.0, atol=2.0e-14)

    solver.z[0].from_numpy(values)
    solver.smooth_jacobi(0)
    expected_smooth = np.zeros(shape, dtype=np.float64)
    omega = solver.jacobi_omega
    for index in np.ndindex(shape):
        if grid_type[index] == solver.FLUID:
            neighbors = _apply_stencil(np.zeros(shape), positive_axis, values, index)
            correction = (rhs[index] - neighbors) / diagonal[index]
            expected_smooth[index] = (1.0 - omega) * values[index] + omega * correction
    np.testing.assert_allclose(solver.z[0].to_numpy(), expected_smooth, rtol=0.0, atol=2.0e-14)


def test_solver_reports_iteration_exhaustion():
    solver = MGPCGPoissonSolver(2, (2, 2), n_mg_levels=1)
    solver.grid_type[0].fill(solver.FLUID)
    solver.Adiag[0].fill(1.0)
    solver.Ax[0].fill(0.0)
    solver.b.fill(1.0)

    assert not solver.solve(max_iters=0)
    assert solver.breakdown_reason == "max_iterations"


def test_solver_accepts_positive_small_scale_search_products():
    solver = MGPCGPoissonSolver(2, (2, 2), n_mg_levels=1, bottom_smoothing=2)
    solver.grid_type[0].fill(solver.FLUID)
    solver.Adiag[0].fill(1.0)
    solver.Ax[0].fill(0.0)
    solver.b.fill(1.0e-7)

    assert solver.solve(max_iters=2)
    np.testing.assert_allclose(solver.x.to_numpy(), 1.0e-7, rtol=0.0, atol=1.0e-20)
    assert solver.breakdown_reason == ""


def test_solver_rejects_false_recursive_convergence():
    solver = MGPCGPoissonSolver(2, (2, 1), n_mg_levels=1, bottom_smoothing=2)
    solver.grid_type[0].fill(solver.FLUID)
    solver.Adiag[0].fill(1.0)
    off_diagonal = np.zeros((2, 1, 2), dtype=np.float64)
    off_diagonal[0, 0, 0] = -0.999999999999
    solver.Ax[0].from_numpy(off_diagonal)
    solver.b.from_numpy(np.asarray([[1.0], [0.5]], dtype=np.float64))

    assert not solver.solve(max_iters=3, rel_tol=1.0e-10, abs_tol=1.0e-14)
    solver.compute_true_residual()
    actual_residual_squared = float(np.dot(solver.r[0].to_numpy().ravel(), solver.r[0].to_numpy().ravel()))
    assert actual_residual_squared > max(1.0e-14, 1.25e-10)
    assert solver.final_residual == pytest.approx(actual_residual_squared)
    assert solver.breakdown_reason == "max_iterations"
