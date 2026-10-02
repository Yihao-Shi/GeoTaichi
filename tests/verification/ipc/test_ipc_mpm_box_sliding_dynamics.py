"""Production box-slide checks for fully implicit point-plane friction.

The single material point is a translation-only surrogate for a rigid box.
Its contact gap is chosen on the host by balancing the IPC barrier
force against its weight.  Consequently the two-dimensional production solve
reduces to the one-dimensional backward-Euler friction equation used as the
independent oracle below, without mocking contact assembly or Newton's method.
"""

import os

import numpy as np
import pytest
import taichi as ti
from scipy.optimize import brentq


GRAVITY = 9.81
MASS = 1.0
SURFACE_MEASURE = 1.0
MU_STATIC = 0.6
MU_DYNAMIC = 0.4
EPSV = 0.05
STRIBECK_VELOCITY = 0.2
DT = 1.0e-2
DHAT = 8.0e-2
KAPPA = 1.0e4


@pytest.fixture(scope="module", autouse=True)
def _initialize_taichi():
    """Own a deterministic CPU runtime for the sub-nanometric oracle checks."""
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        debug=False,
        offline_cache=False,
        cpu_max_num_threads=1,
    )
    try:
        yield
    finally:
        ti.sync()
        ti.reset()


def test_paper_box_slide_closed_form_reference():
    """Document the continuous Coulomb result reported by the paper."""
    slope = np.deg2rad(10.0)
    friction = 0.177
    initial_velocity = 0.1
    acceleration = GRAVITY * (np.sin(slope) - friction * np.cos(slope))
    stopping_time = -initial_velocity / acceleration
    stopping_distance = -(initial_velocity * initial_velocity) / (2.0 * acceleration)

    assert acceleration == pytest.approx(-0.006502015, abs=5.0e-10)
    assert stopping_time == pytest.approx(15.37985, abs=5.0e-5)
    assert stopping_distance == pytest.approx(0.768992, abs=5.0e-7)


def _barrier_normal_force(gap):
    """Host form of the dmin=0 clamped-log IPC barrier force."""
    distance2 = gap * gap
    active_distance2 = DHAT * DHAT
    if not 0.0 < distance2 < active_distance2:
        return 0.0 if distance2 >= active_distance2 else np.inf
    difference = distance2 - active_distance2
    log_term = np.log(distance2 / active_distance2)
    gradient_distance2 = -KAPPA * (2.0 * difference * log_term + difference * difference / distance2)
    gradient_distance = 2.0 * gap * gradient_distance2
    return -SURFACE_MEASURE * gradient_distance


def _equilibrium_gap():
    return brentq(
        lambda gap: _barrier_normal_force(gap) - MASS * GRAVITY,
        DHAT * 1.0e-6,
        DHAT * (1.0 - 1.0e-12),
        xtol=1.0e-14,
        rtol=1.0e-14,
    )


def _smooth_step(speed):
    if speed < EPSV:
        ratio = speed / EPSV
        return ratio * (2.0 - ratio)
    return 1.0


def _stribeck_falloff(speed):
    if speed > STRIBECK_VELOCITY:
        return 0.0
    ratio = speed / STRIBECK_VELOCITY
    return (2.0 * ratio + 1.0) * (ratio - 1.0) ** 2


def _friction_acceleration(speed):
    effective_mu = MU_DYNAMIC + (MU_STATIC - MU_DYNAMIC) * _stribeck_falloff(speed)
    return GRAVITY * effective_mu * _smooth_step(speed)


def _backward_euler_velocity(previous_velocity, load_ratio):
    """Solve the paper's scalar BE momentum residual for positive motion."""
    external_acceleration = load_ratio * MU_STATIC * GRAVITY
    free_velocity = previous_velocity + DT * external_acceleration
    if free_velocity <= 0.0:
        return 0.0

    def residual(velocity):
        return velocity - previous_velocity - DT * (external_acceleration - _friction_acceleration(velocity))

    return brentq(residual, 0.0, free_velocity, xtol=2.0e-14, rtol=1.0e-14)


def _steady_regularized_stick_velocity(load_ratio):
    external_acceleration = load_ratio * MU_STATIC * GRAVITY
    return brentq(
        lambda velocity: _friction_acceleration(velocity) - external_acceleration,
        0.0,
        EPSV,
        xtol=2.0e-14,
        rtol=1.0e-14,
    )


def _build_box(initial_velocity, load_ratio, output_path):
    import src.mpm.config as config
    from src.mpm.generator.Body import Body
    from src.mpm.generator.Ground import Ground
    from src.mpm.soft_particle.IPCULMPM import IPCULMPM

    config.set_dimension(2)
    gap = _equilibrium_gap()
    bodies = Body()
    bodies.add_particles(
        [[0.35, gap]],
        volume=1.0e-3,
        init_v=[initial_velocity, 0.0],
        boundary_ids=[0],
        surface_measure=SURFACE_MEASURE,
        xmin=[0.0, 0.0],
        xmax=[1.0, 1.0],
    )
    ground = Ground()
    ground.append([0.0, 0.0], [0.0, 1.0])
    solver = IPCULMPM(
        bodies,
        ground,
        domain=[1.0, 1.0],
        dx=0.1,
        dt=DT,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=MASS / 1.0e-3,
        gravity=[load_ratio * MU_STATIC * GRAVITY, -GRAVITY],
        residual=1.0e-10,
        max_iters=30,
        interval=1,
        step=1,
        scale=1.0,
        line_search=False,
        shape_function="linear",
        visualize=False,
        path=str(output_path),
        kappa=KAPPA,
        dhat=DHAT,
        dynamic_friction=MU_DYNAMIC,
        static_friction=MU_STATIC,
        viscous_friction=0.0,
        stribeck_velocity=STRIBECK_VELOCITY,
        epsv=EPSV,
        friction_profile="quadratic",
        activate_friction=True,
        friction_mode="fully_implicit",
        coordination_number=[1, 1],
        friction_set=[4, 4],
        barrier_set=[4, 4],
        # The trajectory assertions below resolve the scalar backward-Euler
        # oracle to roughly 1e-10 in velocity.  Use a correspondingly tighter
        # nonlinear residual target so the test measures the friction law,
        # rather than stopping two Newton digits before its own oracle bound.
        fully_implicit_residual_atol=1.0e-14,
        fully_implicit_residual_rtol=1.0e-12,
    )
    solver.initial_simulation()
    assert float(solver.mpm.particle[0].m) == pytest.approx(MASS)
    assert _barrier_normal_force(gap) == pytest.approx(MASS * GRAVITY)
    return solver, gap


def _advance(solver, load_ratio):
    solver.mpm.gravity = [load_ratio * MU_STATIC * GRAVITY, -GRAVITY]
    solver.substep(verbose=False)
    position = solver.mpm.particle.x.to_numpy()[0].copy()
    velocity = solver.mpm.particle.v.to_numpy()[0].copy()
    assert solver.ipc.last_friction_converged
    return position, velocity


@pytest.mark.isolated_dimension(2)
def test_regularized_stick_to_dynamic_slip_on_one_production_instance(
    tmp_path,
):
    """A subcritical q=0.8 load sticks, then q=1.2 initiates sliding."""
    solver, equilibrium_gap = _build_box(
        initial_velocity=0.0,
        load_ratio=0.8,
        output_path=tmp_path / "box",
    )
    oracle_velocity = 0.0
    oracle_position = 0.35

    for _ in range(20):
        oracle_velocity = _backward_euler_velocity(oracle_velocity, 0.8)
        oracle_position += DT * oracle_velocity
        position, velocity = _advance(solver, 0.8)
        assert velocity[0] == pytest.approx(oracle_velocity, rel=2.0e-8, abs=2.0e-10)
        assert position[0] == pytest.approx(oracle_position, rel=2.0e-9, abs=2.0e-10)
        assert position[1] == pytest.approx(equilibrium_gap, rel=0.0, abs=2.0e-10)
        assert velocity[1] == pytest.approx(0.0, abs=2.0e-9)

    stick_velocity = _steady_regularized_stick_velocity(0.8)
    assert oracle_velocity < EPSV
    assert oracle_velocity == pytest.approx(stick_velocity, rel=2.0e-6)

    # Change only the applied load on this same production object.  A load
    # above the peak static-friction capacity has no stationary root, crosses
    # the C1 threshold, and ultimately reaches the dynamic Stribeck branch.
    crossed_regularization = False
    reached_dynamic_branch = False
    for _ in range(18):
        previous_oracle = oracle_velocity
        oracle_velocity = _backward_euler_velocity(oracle_velocity, 1.2)
        oracle_position += DT * oracle_velocity
        position, velocity = _advance(solver, 1.2)
        assert velocity[0] == pytest.approx(oracle_velocity, rel=2.0e-8, abs=3.0e-10)
        assert position[0] == pytest.approx(oracle_position, rel=2.0e-9, abs=3.0e-10)
        assert position[1] == pytest.approx(equilibrium_gap, rel=0.0, abs=3.0e-10)
        assert velocity[0] > previous_oracle
        crossed_regularization |= velocity[0] >= EPSV
        reached_dynamic_branch |= velocity[0] >= STRIBECK_VELOCITY

    assert crossed_regularization
    assert reached_dynamic_branch
    observed_friction_mu = (1.2 * MU_STATIC * GRAVITY - (oracle_velocity - previous_oracle) / DT) / GRAVITY
    assert observed_friction_mu == pytest.approx(MU_DYNAMIC, rel=2.0e-12)


@pytest.mark.isolated_dimension(2)
def test_dynamic_stopping_trajectory_matches_backward_euler_oracle(tmp_path):
    """Kinetic, Stribeck, and regularized stopping all follow the BE residual."""
    initial_velocity = 0.6
    solver, equilibrium_gap = _build_box(
        initial_velocity=initial_velocity,
        load_ratio=0.0,
        output_path=tmp_path / "box",
    )
    oracle_velocity = initial_velocity
    oracle_position = 0.35
    production_positions = []
    production_velocities = []
    oracle_positions = []
    oracle_velocities = []

    for _ in range(30):
        oracle_velocity = _backward_euler_velocity(oracle_velocity, 0.0)
        oracle_position += DT * oracle_velocity
        position, velocity = _advance(solver, 0.0)
        production_positions.append(position[0])
        production_velocities.append(velocity[0])
        oracle_positions.append(oracle_position)
        oracle_velocities.append(oracle_velocity)
        assert position[1] == pytest.approx(equilibrium_gap, rel=0.0, abs=3.0e-10)
        assert velocity[1] == pytest.approx(0.0, abs=3.0e-9)

    np.testing.assert_allclose(
        production_velocities,
        oracle_velocities,
        rtol=2.0e-8,
        atol=3.0e-10,
    )
    np.testing.assert_allclose(
        production_positions,
        oracle_positions,
        rtol=2.0e-9,
        atol=4.0e-10,
    )

    # Before reaching the Stribeck interval, BE with constant kinetic
    # friction has an exact arithmetic velocity sequence and an exact summed
    # displacement sequence.
    kinetic_steps = sum(velocity >= STRIBECK_VELOCITY for velocity in oracle_velocities)
    step_ids = np.arange(1, kinetic_steps + 1, dtype=np.float64)
    analytic_velocity = initial_velocity - step_ids * DT * MU_DYNAMIC * GRAVITY
    analytic_position = (
        0.35 + step_ids * DT * initial_velocity - 0.5 * MU_DYNAMIC * GRAVITY * DT * DT * step_ids * (step_ids + 1.0)
    )
    np.testing.assert_allclose(
        production_velocities[:kinetic_steps],
        analytic_velocity,
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        production_positions[:kinetic_steps],
        analytic_position,
        rtol=2.0e-9,
        atol=3.0e-10,
    )
    assert production_velocities[-1] < EPSV
    assert np.all(np.diff(production_velocities) < 0.0)
