"""Default solve accuracy must resolve physical motion without requesting lost coordinate bits."""

import numpy as np
import pytest
import taichi as ti

from src.dem.Simulation import Simulation
from src.dem.engines.AffineBodyEngine import AffineBodyEngine
from src.dem.engines.AffineBodyOperator import TaichiAffineBodyOperator
from src.linear_solver.BuildTriplet import BuildTriplet


@ti.data_oriented
class LinearAccuracyProbe(TaichiAffineBodyOperator):
    def __init__(self, dt):
        self.body_num = 1
        self.control_num = self.vertex_num = 4
        self.fully_implicit = False
        self.dt = dt
        self.mass = ti.field(ti.f64, shape=(1, 4, 4))
        self.basis = ti.field(ti.f64, shape=(4, 4))
        self.node2body = ti.field(ti.i32, shape=4)
        self.grad = ti.Vector.field(3, ti.f64, shape=4)
        self.direction_y = ti.Vector.field(3, ti.f64, shape=4)
        self.mass.from_numpy((1.0e-4 * np.eye(4))[None])
        self.basis.from_numpy(np.eye(4))


@pytest.mark.parametrize("dt", [1.0e-3, 1.0e-5])
def test_default_linear_accuracy_resolves_small_mass_newton_correction(taichi_runtime, dt):
    operator = LinearAccuracyProbe(dt)
    expected = dt * 1.0e-5 * np.arange(1, 13, dtype=np.float64).reshape((4, 3))
    operator.grad.from_numpy(-1.0e-4 * expected)
    matrix = BuildTriplet(3, 1, 1, 4, symmetric=False, solver="PCG", matrix_symmetric=True)
    matrix.diag.from_numpy(np.tile((1.0e-4 * np.eye(3)).ravel(), (4, 1)))
    engine = AffineBodyEngine()
    engine.operator, engine.hash_triplet = operator, matrix
    engine._ensure_linear_solver = lambda _sims: None
    sims = Simulation()
    sims.affine_assemble_type = "HashTriplet"
    assert sims.affine_newton_tolerance < sims.affine_friction_tolerance
    assert np.linalg.norm(operator.grad.to_numpy()) < 1.0e-9
    engine._solve_hash_direction_device(sims)
    np.testing.assert_allclose(operator.direction_y.to_numpy(), expected, rtol=1e-12, atol=1e-15)
    assert engine.last_linear_iterations > 0
    assert operator.device_surface_direction_inf_norm() / dt > sims.affine_newton_tolerance


def test_adjoint_retains_complete_body_preconditioner_and_true_accuracy(taichi_runtime):
    from types import SimpleNamespace

    rng = np.random.default_rng(44)
    rotation, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    dense = np.einsum("ik,k,jk->ij", rotation, np.geomspace(1e-3, 1e3, 12), rotation)
    expected = rng.normal(scale=1e-4, size=12)
    rhs = np.einsum("ij,j->i", dense, expected)
    matrix = BuildTriplet(
        3,
        16,
        16,
        4,
        symmetric=False,
        solver="BiCGSTAB",
        matrix_symmetric=True,
        preconditioner_group_size=4,
    )
    linear_rhs = ti.field(ti.f64, shape=12)
    linear_x = ti.field(ti.f64, shape=12)
    linear_rhs.from_numpy(rhs)

    @ti.kernel
    def assemble(values: ti.types.ndarray()):
        for row, col in ti.ndrange(4, 4):
            if row <= col:
                block = ti.Matrix.zero(ti.f64, 3, 3)
                for i, j in ti.static(ti.ndrange(3, 3)):
                    block[i, j] = values[3 * row + i, 3 * col + j]
                matrix.add_block_entry(row, col, block)

    engine = AffineBodyEngine()
    engine.adjoint_triplet = matrix
    engine.sims = SimpleNamespace(
        affine_linear_tolerance=1e-12, affine_linear_relative_tolerance=0.0, affine_linear_max_iteration=4
    )
    engine.operator = SimpleNamespace(
        control_num=4,
        linear_rhs=linear_rhs,
        linear_x=linear_x,
        device_restore_lagged_friction_for_adjoint=lambda: None,
        device_enter_equilibrium_adjoint=lambda: None,
        bind_hash_triplet=lambda _matrix: None,
        assemble_device=lambda **kwargs: assemble(np.ascontiguousarray(dense)),
        device_gradient_inf_norm=lambda: 0.0,
    )
    engine._solve_adjoint_device()
    actual = linear_x.to_numpy()
    assert np.linalg.norm(np.einsum("ji,j->i", dense, actual) - rhs) <= engine.sims.affine_linear_tolerance
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    assert matrix.preconditioner_group_size == 4 and matrix.solver == "BiCGSTAB"


def test_explicit_linear_accuracy_remains_authoritative(taichi_runtime):
    sims = Simulation()
    sims.set_affine_body_parameters(linear_relative_tolerance=0.1)
    sims.set_affine_body_parameters(linear_tolerance=2.0e-5, newton_tolerance=3.0e-6, friction_tolerance=4.0e-7)
    assert sims.affine_linear_relative_tolerance == 0.0
    engine = AffineBodyEngine()
    engine.operator = type("LaggedOperator", (), {"fully_implicit": False})()
    assert engine._linear_tolerance(sims, 1.0e-10) == 2.0e-5
    assert sims.affine_newton_tolerance == 3.0e-6
    assert sims.affine_friction_tolerance == 4.0e-7


@pytest.mark.parametrize("fully_implicit", [False, True])
def test_relative_linear_accuracy_reaches_device_solve_without_bypassing_force_target(taichi_runtime, fully_implicit):
    operator = LinearAccuracyProbe(1e-3)
    operator.fully_implicit = fully_implicit
    expected = 1e-8 * np.arange(1, 13, dtype=np.float64).reshape(4, 3)
    operator.grad.from_numpy(-1e-4 * expected)
    matrix = BuildTriplet(3, 1, 1, 4, symmetric=False, solver="PCG", matrix_symmetric=True)
    matrix.diag.from_numpy(np.tile((1e-4 * np.eye(3)).ravel(), (4, 1)))
    engine = AffineBodyEngine()
    engine.operator, engine.hash_triplet = operator, matrix
    engine._ensure_linear_solver = lambda _sims: None
    sims = Simulation()
    sims.set_affine_body_parameters(assemble_type="HashTriplet", linear_tolerance=0.0, linear_relative_tolerance=0.25)
    result = engine._solve_hash_direction_device(sims)
    target = 0.25 * np.linalg.norm(1e-4 * expected)
    if fully_implicit:
        target = min(
            target, 0.1 * engine._fully_implicit_force_target(sims, float(np.max(1e-4 * expected)), operator.dt)
        )
    assert result["convergence_tolerance"] == pytest.approx(target, rel=1e-12, abs=0)
    np.testing.assert_allclose(operator.direction_y.to_numpy(), expected, rtol=1e-12, atol=0)
    assert sims.affine_linear_relative_tolerance == 0.25


@pytest.mark.parametrize("relative", [-1.0, np.nan, np.inf])
def test_invalid_relative_linear_tolerance_is_rejected(taichi_runtime, relative):
    sims = Simulation()
    with pytest.raises(RuntimeError, match="linear_relative_tolerance"):
        sims.set_affine_body_parameters(linear_relative_tolerance=relative)


@pytest.mark.parametrize("integration", [(1.0, 0.5, 1.0), (0.5, 0.25, 0.5)])
def test_default_force_accuracy_fits_coordinate_and_velocity_resolution(taichi_runtime, integration):
    sims = Simulation()
    sims.set_affine_body_parameters(newmark=integration)
    dt, mass, coordinate = 5e-4, 1e-2, 1.25
    scale = 2 * integration[0] * integration[1] * dt**2
    threshold = AffineBodyEngine._fully_implicit_force_target(sims, scale * 1e-4, dt)
    # Each stored absolute coordinate can have half an ULP of rounding error.
    rounding_force = mass * np.spacing(coordinate) / scale
    assert threshold / scale > 2 * rounding_force
    endpoint_velocity_factor = integration[2] / (2 * integration[0] * integration[1] * dt)
    assert endpoint_velocity_factor * threshold / mass < 0.1 * sims.affine_newton_tolerance


def test_explicit_force_accuracy_remains_authoritative(taichi_runtime):
    sims = Simulation()
    sims.set_affine_body_parameters(fully_implicit_force_atol=1e-12, fully_implicit_force_rtol=0.0)
    assert AffineBodyEngine._fully_implicit_force_target(sims, 1e-10, 5e-4) == pytest.approx(2.5e-19, rel=1e-14, abs=0)
