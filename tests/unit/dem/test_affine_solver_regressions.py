from types import SimpleNamespace
import importlib

import numpy as np
import pytest
import taichi as ti
import trimesh

from src.dem.DEMBase import AffineBodySolver
from src.dem.engines.AffineBodyEngine import AffineBodyEngine
from src.dem.engines.AffineBodyOperator import TaichiAffineBodyOperator
from src.dem.engines.AffineBodyState import AffineBodyState


@ti.data_oriented
class BodyProbe(TaichiAffineBodyOperator):
    def __init__(self, newmark=False):
        self.use_newmark = newmark
        self.integration = (0.5, 0.25, 0.5) if newmark else (1.0, 0.5, 1.0)
        self.body_num = 1
        self.control_num = 4
        self.dof = 12
        self.max_coo_entries = 12
        self.fully_implicit = True
        for name in (
            "y",
            "hat_y",
            "velocity_origin_y",
            "grad",
            "velocity_y",
            "previous_y",
            "previous_velocity_y",
            "external_generalized_force",
            "direction_y",
            "line_search_base_y",
        ):
            setattr(self, name, ti.Vector.field(3, ti.f64, shape=4))
        for name in ("energy", "dt_device", "scale_device"):
            setattr(self, name, ti.field(ti.f64, shape=()))
        for name in ("body_mass", "force_damp", "torque_damp"):
            setattr(self, name, ti.field(ti.f64, shape=1))
        self.mass = ti.field(ti.f64, shape=(1, 4, 4))
        self.gravity = ti.Vector.field(3, ti.f64, shape=())
        self.accepted_translation_velocity = ti.Vector.field(3, ti.f64, shape=())
        self.hessian = ti.field(ti.f64, shape=(12, 12))
        self.linear_x = ti.field(ti.f64, shape=12)
        self.linear_rhs = ti.field(ti.f64, shape=12)
        self.dt_device[None] = 1.0
        self.scale_device[None] = 0.25 if newmark else 1.0
        self.body_mass[0] = 1.0
        mesh = trimesh.creation.box()
        material = mesh.vertices + 0.25
        basis = np.column_stack((1 - material.sum(axis=1), material))
        mass = AffineBodyState._mass_matrix(mesh.vertices, mesh.faces, basis, 1.0, 1.0)
        self.mass.from_numpy(mass[None])

    @ti.func
    def _add_matrix_block(self, row, col, block, matrix_target: ti.template()):
        for i, j in ti.static(ti.ndrange(3, 3)):
            ti.atomic_add(matrix_target[3 * row + i, 3 * col + j], block[i, j])

    def bind_coo_matrix(self, matrix):
        self.coo_matrix = matrix

    def _active_jacobian_shift(self):
        return 0.0

    def prepare_line_search_neighbors_device(self, _alpha):
        return False

    @ti.kernel
    def friction_curvature(self, displacement: ti.f64, majorize: ti.template()) -> ti.types.matrix(3, 3, ti.f64):
        return self._lagged_friction_hessian(
            ti.Vector([displacement, 0.0, 0.0]),
            ti.Vector.zero(ti.f64, 3),
            ti.Vector([0.0, 0.0, 1.0]),
            0.25,
            majorize,
        )


@pytest.mark.parametrize("speed", [0.0, 0.002, 0.01, 0.04])
def test_friction_majorizer_bounds_the_same_potential(taichi_runtime, speed):
    operator = BodyProbe()
    operator.epsv = 0.01
    dt = 0.001
    operator.dt_device[None] = dt
    exact = operator.friction_curvature(speed * dt, False)
    major = operator.friction_curvature(speed * dt, True)
    f1 = 2 / operator.epsv - speed / operator.epsv**2 if speed < operator.epsv else 1 / speed
    np.testing.assert_allclose(major, np.diag([1, 1, 0]) * 0.25 * f1 / dt, atol=1e-10)
    assert np.linalg.eigvalsh(major - exact).min() >= -1e-10

    def potential(radius):
        return (
            (radius**2 * (-radius / 3 + operator.epsv) / operator.epsv**2 + operator.epsv / 3)
            if radius < operator.epsv
            else radius
        )

    for trial in (0.0, 0.001, 0.01, 0.1):
        assert potential(trial) <= potential(speed) + 0.5 * f1 * (trial**2 - speed**2) + 1e-15


def test_friction_search_breaks_sliding_cycle_and_keeps_physical_newton_stop():
    engine = AffineBodyEngine()
    coefficient, epsilon, dt = 2.58e-4, 1e-7, 0.001
    alpha = 1 / 256
    state = SimpleNamespace(x=coefficient * alpha / (2 - alpha), p=0.0, g=0.0, h=0.0, base=0.0)
    models = []

    def assemble(_sims, need_matrix=True, friction_majorizer=False, reuse_neighbor=False):
        radius = abs(state.x)
        f1 = 2 / epsilon - radius / epsilon**2 if radius < epsilon else 1 / radius
        radial = 2 * (epsilon - radius) / epsilon**2 if radius < epsilon else 0.0
        state.g = state.x + coefficient * f1 * state.x
        state.h = 1 + coefficient * (f1 if friction_majorizer else radial)
        if need_matrix:
            models.append(friction_majorizer)
        phi = radius**2 * (-radius / 3 + epsilon) / epsilon**2 + epsilon / 3 if radius < epsilon else radius
        return 0.5 * state.x**2 + coefficient * phi

    def backup():
        state.base = state.x

    engine.operator = SimpleNamespace(
        dt=dt,
        correction_velocity_factor=1 / dt,
        device_gradient_has_nonfinite=lambda: 0,
        device_direction_has_nonfinite=lambda: 0,
        device_direction_inf_norm=lambda: abs(state.p),
        device_gradient_direction_dot=lambda: state.g * state.p,
        device_backup_line_search_base=backup,
        device_restore_line_search_base=lambda: setattr(state, "x", state.base),
        device_set_line_search_trial=lambda step: setattr(state, "x", state.base + step * state.p),
        prepare_line_search_neighbors_device=lambda _alpha: False,
        has_lagged_friction_contacts=lambda: True,
    )
    engine._assemble_system_device = assemble
    engine._solve_hash_direction_device = lambda _sims: setattr(state, "p", -state.g / state.h)
    engine._clamp_device_direction = lambda _maximum: None
    engine._device_surface_direction_inf_norm = lambda _norm: abs(state.p)
    engine._ccd_step_size_device = lambda _sims: 1.0
    sims = SimpleNamespace(
        affine_newton_tolerance=1e-7,
        affine_max_newton_iteration=20,
        affine_line_search_max_iteration=20,
        affine_max_step=1.0,
    )
    energy, iterations = engine._solve_lagged_inner_cuda(sims, assemble(sims))
    assert engine.last_inner_converged and iterations < 10
    assert any(models) and models[-1] is False
    assert abs(state.g / state.h) / dt < sims.affine_newton_tolerance
    assert abs(state.x) < 1e-12 and np.isfinite(energy)


@pytest.mark.parametrize("omega", [[0.0, 0.0, 1.0], [1.0, -2.0, 3.0]])
@pytest.mark.parametrize("force_damping", [0.0, 2.0])
@pytest.mark.parametrize("newmark", [False, True])
def test_affine_rotational_damping_uses_solid_inertia(taichi_runtime, omega, force_damping, newmark):
    operator = BodyProbe(newmark=newmark)
    controls = np.vstack((np.zeros(3), np.eye(3))) - 0.25
    translation = np.array([1.0, -0.5, 2.0])
    operator.hat_y.from_numpy(controls)
    operator.velocity_origin_y.from_numpy(controls)
    factor = 2.0 if newmark else 1.0
    operator.y.from_numpy(controls + (np.cross(omega, controls) + translation) / factor)
    operator.force_damp[0] = force_damping
    operator.torque_damp[0] = 1.0
    operator._assemble_local_damping(True, operator.hessian)
    forces = -operator.grad.to_numpy() / float(operator.scale_device[None])
    np.testing.assert_allclose(forces.sum(axis=0), -force_damping * translation, atol=1e-14)
    np.testing.assert_allclose(np.cross(controls, forces).sum(axis=0), -np.asarray(omega) / 6.0, atol=1e-14)
    np.testing.assert_allclose(operator.hessian.to_numpy(), operator.hessian.to_numpy().T, atol=1e-14)
    assert np.linalg.eigvalsh(operator.hessian.to_numpy()).min() > -1e-14


def test_world_joint_preserves_converged_anchor_reaction(monkeypatch, taichi_runtime):
    module = importlib.import_module("src.dem.engines.AffineBodyEngine")
    operator = BodyProbe()
    operator.wall_num = 0
    operator.joint_num = 1
    operator.joint_body_np = np.array([[0, -1]])
    operator.levelset_contact = False
    state = SimpleNamespace(bodies=[{"force_damping": 0.0}])
    monkeypatch.setattr(module.AffineBodyState, "from_scene", lambda *args: state)
    monkeypatch.setattr(module, "TaichiAffineBodyOperator", lambda *args: operator)
    sims = SimpleNamespace(
        affine_assemble_type="COO",
        freeze_affine_body_parameters=lambda: None,
        search="LinkedCell",
        wall_type=None,
        enable_step_retry=False,
        step_retry_max_retries=0,
        step_retry_reduction=0.5,
        step_retry_minimum_timestep=0.0,
    )
    engine = AffineBodyEngine()
    engine.initialize(sims, SimpleNamespace(affine_bodies=[object()]))
    assert not engine.enforce_free_translation
    controls = np.vstack((np.zeros(3), np.eye(3))) - 0.25
    dt = 0.1
    delta = np.array([0.0, 0.0, -dt * dt / (1.0 + dt * dt * 1.0e6)])
    operator.hat_y.from_numpy(controls)
    operator.y.from_numpy(controls + delta)
    operator.gravity[None] = [0.0, 0.0, -1.0]
    engine._accept_device_state(dt)
    np.testing.assert_allclose(operator.y.to_numpy(), controls + delta, atol=1e-14)


@pytest.mark.parametrize("backend", ["COO", "MatrixFree"])
def test_fully_implicit_device_line_search_uses_selected_matrix(taichi_runtime, backend):
    engine = AffineBodyEngine()
    operator = BodyProbe()
    engine.operator = operator
    sims = SimpleNamespace(
        affine_assemble_type=backend,
        affine_line_search_max_iteration=4,
        affine_fully_implicit_armijo=1e-4,
        affine_fully_implicit_line_search_contraction=0.5,
    )
    engine._ensure_linear_solver(sims)
    matrix = engine.coo_matrix
    matrix.rows.from_numpy(np.arange(12, dtype=np.int32))
    matrix.cols.from_numpy(np.arange(12, dtype=np.int32))
    matrix.data.fill(2.0)
    operator.grad.fill(1.0)
    operator.direction_y.fill(-0.5)
    engine._ccd_step_size_device = lambda _sims: 1.0

    assembled = []

    def assemble(_sims, need_matrix=True, reuse_neighbor=False):
        assembled.append((need_matrix, reuse_neighbor))
        operator.grad.fill(0.0)
        return 0.0

    engine._assemble_system_device = assemble
    alpha = engine._fully_implicit_line_search_cuda(sims)
    assert alpha == 1.0
    assert engine.last_fully_implicit_merit_slope == pytest.approx(-12.0)
    assert engine.hash_triplet is None
    assert assembled == [(False, False), (True, True)]
    np.testing.assert_allclose(operator.y.to_numpy(), -0.5)


@pytest.mark.parametrize("dt", [0.001, 0.1, 1.0])
def test_fully_implicit_absolute_force_tolerance_is_independent_of_timestep(dt):
    sims = SimpleNamespace(affine_fully_implicit_force_atol=2.0e-6, affine_fully_implicit_force_rtol=3.0e-4)
    tolerance = AffineBodyEngine._fully_implicit_force_target(sims, 10.0 * dt**2, dt)
    assert tolerance / dt**2 == pytest.approx(3.002e-3)


@pytest.mark.parametrize("fully_implicit", [False, True])
def test_linear_tolerance_cannot_mask_fully_implicit_newton_residual(fully_implicit):
    engine = AffineBodyEngine()
    engine.operator = SimpleNamespace(fully_implicit=fully_implicit, dt=0.001)
    sims = SimpleNamespace(
        affine_linear_tolerance=1.0e-9,
        affine_fully_implicit_force_atol=1.0e-12,
        affine_fully_implicit_force_rtol=1.0e-8,
    )
    tolerance = engine._linear_tolerance(sims, 1.0e-10)
    assert tolerance == pytest.approx(2.0e-19 if fully_implicit else 1.0e-9, rel=1.0e-12, abs=0.0)


class Timer:
    def begin(self, *args):
        pass

    def end(self, *args):
        pass

    def profile1(self):
        pass


@pytest.mark.parametrize("retry", [False, True])
def test_affine_solver_counts_accepted_steps_and_stops_at_requested_time(monkeypatch, retry):
    monkeypatch.setattr(ti, "sync", lambda: None)
    sims = SimpleNamespace(
        scheme="AffineBody", delta=0.1, time=0.15, current_time=0.0, current_step=0, timer=Timer(), save_interval=0.05
    )
    sims.set_timestep = lambda dt: setattr(sims, "delta", dt)
    accepted = []

    def step(sims, scene):
        if retry and not accepted:
            sims.set_timestep(0.05)
        accepted.append(sims.delta)

    engine = SimpleNamespace(initialize=lambda *args: None, step=step)
    solver = AffineBodySolver(sims, None, None, engine, None)
    saved = []

    def save_file(_scene):
        saved.append(sims.current_time)
        solver.last_save_time = sims.current_time

    solver.save_file = save_file
    solver.Solver(None)
    assert sims.current_time == pytest.approx(0.15)
    assert sum(accepted) == pytest.approx(sims.current_time)
    assert sims.current_step == len(accepted)
    assert saved == pytest.approx([0.0, 0.05, 0.1, 0.15] if retry else [0.0, 0.1, 0.15])
