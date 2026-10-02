import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.physics_model.contact_model.ipc.IPC import (
    Barrier,
    Friction,
    clamped_log_barrier_second_derivative,
    initial_barrier_stiffness,
    ipc_barrier_distance_terms_py,
    ipc_barrier_force_magnitude,
    ipc_barrier_force_magnitude_gradient,
    ipc_toolkit_barrier_distance2_terms,
    ipc_toolkit_barrier_distance2_offset_terms,
    ipc_toolkit_barrier_distance_offset_terms,
    ipc_toolkit_barrier_distance_terms,
    ipc_friction_f0,
    ipc_friction_f1,
    ipc_friction_f1_derivative,
    ipc_friction_f1_over_speed,
    ipc_friction_hessian_term,
    ipc_fully_implicit_point_plane_friction,
    ipc_fully_implicit_point_plane_stribeck_friction,
    ipc_fully_implicit_scalar_law_py,
    ipc_stribeck_falloff,
    ipc_stribeck_falloff_derivative,
    semi_ipc_find,
    semi_ipc_find_or_insert,
    semi_ipc_terms,
    semi_ipc_terms_py,
    semi_ipc_update_multiplier,
    semi_ipc_update_multiplier_py,
    update_barrier_stiffness,
)


@pytest.fixture(scope="module", autouse=True)
def _initialize_taichi():
    ti.init(
        arch=ti.cpu,
        cpu_max_num_threads=2,
        offline_cache=False,
        default_fp=ti.f64,
        debug=False,
    )


@ti.kernel
def _eval_official_distance_terms(
    distance: ti.f64, active_distance: ti.f64, kappa: ti.f64
) -> ti.types.vector(6, ti.f64):
    energy2, gradient2, hessian2 = ipc_toolkit_barrier_distance2_terms(
        distance * distance, active_distance * active_distance, kappa
    )
    energy, gradient, hessian = ipc_toolkit_barrier_distance_terms(distance, active_distance, kappa)
    return ti.Vector([energy2, gradient2, hessian2, energy, gradient, hessian])


@ti.kernel
def _eval_official_offset_terms(
    distance: ti.f64,
    dhat: ti.f64,
    dmin: ti.f64,
    kappa: ti.f64,
    physical: ti.i32,
) -> ti.types.vector(6, ti.f64):
    energy2, gradient2, hessian2 = ipc_toolkit_barrier_distance2_offset_terms(
        distance * distance, dhat, dmin, kappa, physical
    )
    energy, gradient, hessian = ipc_toolkit_barrier_distance_offset_terms(distance, dhat, dmin, kappa, physical)
    return ti.Vector([energy2, gradient2, hessian2, energy, gradient, hessian])


@ti.kernel
def _eval_point_point_force_magnitude(
    p0x: ti.f64,
    p0y: ti.f64,
    p1x: ti.f64,
    p1y: ti.f64,
    dhat: ti.f64,
    dmin: ti.f64,
    kappa: ti.f64,
    physical: ti.i32,
) -> ti.types.vector(5, ti.f64):
    delta = ti.Vector([p0x - p1x, p0y - p1y])
    distance2 = delta.dot(delta)
    distance2_gradient = ti.Vector([2.0 * delta[0], 2.0 * delta[1], -2.0 * delta[0], -2.0 * delta[1]])
    magnitude = ipc_barrier_force_magnitude(distance2, dhat, dmin, kappa, physical)
    gradient = ipc_barrier_force_magnitude_gradient(
        distance2,
        distance2_gradient,
        dhat,
        dmin,
        kappa,
        physical,
    )
    return ti.Vector([magnitude, gradient[0], gradient[1], gradient[2], gradient[3]])


@ti.kernel
def _eval_friction_terms(speed: ti.f64, epsv: ti.f64, timestep: ti.f64) -> ti.types.vector(5, ti.f64):
    return ti.Vector(
        [
            ipc_friction_f0(speed, epsv, timestep),
            ipc_friction_f1(speed, epsv),
            ipc_friction_f1_derivative(speed, epsv),
            ipc_friction_f1_over_speed(speed, epsv),
            ipc_friction_hessian_term(speed, epsv),
        ]
    )


@ti.kernel
def _eval_semi_ipc_terms(gap: ti.f64, multiplier: ti.f64, penalty: ti.f64) -> ti.types.vector(4, ti.f64):
    energy, gradient, hessian = semi_ipc_terms(gap, multiplier, penalty)
    updated = semi_ipc_update_multiplier(gap, multiplier, penalty)
    return ti.Vector([energy, gradient, hessian, updated])


@ti.kernel
def _insert_colliding_semi_ipc_keys(
    states: ti.template(),
    keys: ti.template(),
    multipliers: ti.template(),
    count: ti.template(),
    slots: ti.template(),
    key_count: ti.i32,
    capacity: ti.i32,
):
    for index in range(key_count):
        key = ti.Vector([index * capacity, 0, 0, 0])
        slots[index] = semi_ipc_find_or_insert(states, keys, multipliers, count, key, capacity)


@ti.kernel
def _find_colliding_semi_ipc_keys(
    states: ti.template(),
    keys: ti.template(),
    slots: ti.template(),
    key_count: ti.i32,
    capacity: ti.i32,
):
    for index in range(key_count):
        key = ti.Vector([index * capacity, 0, 0, 0])
        slots[index] = semi_ipc_find(states, keys, key, capacity)


@ti.kernel
def _eval_friction_object(
    friction: ti.template(),
    displacement_x: ti.f64,
    displacement_y: ti.f64,
    mu_lambda: ti.f64,
    timestep: ti.f64,
) -> ti.types.vector(7, ti.f64):
    displacement = ti.Vector([displacement_x, displacement_y])
    normal = ti.Vector([0.0, 1.0])
    dvelocity_ddisplacement = ti.Matrix.identity(ti.f64, 2) / timestep
    energy = friction.energy(displacement, normal, mu_lambda, timestep)
    gradient = friction.gradient(
        displacement,
        normal,
        mu_lambda,
        timestep,
        dvelocity_ddisplacement,
    )
    hessian = friction.hessian(
        displacement,
        normal,
        mu_lambda,
        timestep,
        dvelocity_ddisplacement,
        dvelocity_ddisplacement,
    )
    return ti.Vector(
        [
            energy,
            gradient[0],
            gradient[1],
            hessian[0, 0],
            hessian[0, 1],
            hessian[1, 0],
            hessian[1, 1],
        ]
    )


@ti.kernel
def _eval_barrier_activation_distance(
    barrier: ti.template(),
) -> ti.f64:
    return barrier.activation_distance_term()


@ti.kernel
def _eval_fully_implicit_point_plane(
    velocity_x: ti.f64,
    velocity_y: ti.f64,
    normal_x: ti.f64,
    normal_y: ti.f64,
    mu_lambda: ti.f64,
    mu_lambda_gradient_x: ti.f64,
    mu_lambda_gradient_y: ti.f64,
    epsv: ti.f64,
    velocity_displacement_scale: ti.f64,
) -> ti.types.vector(6, ti.f64):
    force, jacobian = ipc_fully_implicit_point_plane_friction(
        ti.Vector([velocity_x, velocity_y]),
        ti.Vector([normal_x, normal_y]),
        mu_lambda,
        ti.Vector([mu_lambda_gradient_x, mu_lambda_gradient_y]),
        epsv,
        velocity_displacement_scale,
    )
    return ti.Vector(
        [
            force[0],
            force[1],
            jacobian[0, 0],
            jacobian[0, 1],
            jacobian[1, 0],
            jacobian[1, 1],
        ]
    )


@ti.kernel
def _eval_stribeck_falloff(speed: ti.f64, stribeck_velocity: ti.f64) -> ti.types.vector(2, ti.f64):
    return ti.Vector(
        [
            ipc_stribeck_falloff(speed, stribeck_velocity),
            ipc_stribeck_falloff_derivative(speed, stribeck_velocity),
        ]
    )


@ti.kernel
def _eval_fully_implicit_stribeck_point_plane(
    velocity_x: ti.f64,
    velocity_y: ti.f64,
    normal_force: ti.f64,
    normal_force_gradient_x: ti.f64,
    normal_force_gradient_y: ti.f64,
    mu_dynamic: ti.f64,
    mu_static: ti.f64,
    mu_viscous: ti.f64,
    stribeck_velocity: ti.f64,
    epsv: ti.f64,
    profile_id: ti.i32,
    velocity_displacement_scale: ti.f64,
) -> ti.types.vector(6, ti.f64):
    force, jacobian = ipc_fully_implicit_point_plane_stribeck_friction(
        ti.Vector([velocity_x, velocity_y]),
        ti.Vector([0.0, 1.0]),
        normal_force,
        ti.Vector([normal_force_gradient_x, normal_force_gradient_y]),
        mu_dynamic,
        mu_static,
        mu_viscous,
        stribeck_velocity,
        epsv,
        profile_id,
        velocity_displacement_scale,
    )
    return ti.Vector(
        [
            force[0],
            force[1],
            jacobian[0, 0],
            jacobian[0, 1],
            jacobian[1, 0],
            jacobian[1, 1],
        ]
    )


def _gap_reference(gap, active_gap, kappa):
    safe_gap = max(float(gap), max(1.0e-12 * active_gap, 1.0e-14))
    if safe_gap >= active_gap:
        return np.zeros(3, dtype=np.float64)
    diff = safe_gap - active_gap
    log_term = np.log(safe_gap / active_gap)
    return np.array(
        [
            -kappa * diff * diff * log_term,
            -kappa * (2.0 * diff * log_term + diff * diff / safe_gap),
            -kappa * (2.0 * log_term + 4.0 * diff / safe_gap - diff * diff / (safe_gap * safe_gap)),
        ],
        dtype=np.float64,
    )


def _friction_reference(speed, epsv, timestep):
    if speed < epsv:
        f0 = timestep * (speed * speed * (epsv - speed / 3.0) / (epsv * epsv) + epsv / 3.0)
        f1 = speed * (2.0 * epsv - speed) / (epsv * epsv)
        f1_derivative = 2.0 * (epsv - speed) / (epsv * epsv)
        f1_over_speed = (2.0 * epsv - speed) / (epsv * epsv)
        hessian_term = -1.0 / (epsv * epsv)
    else:
        f0 = timestep * speed
        f1 = 1.0
        f1_derivative = 0.0
        f1_over_speed = 1.0 / speed
        hessian_term = -1.0 / (speed * speed)
    return np.array(
        [f0, f1, f1_derivative, f1_over_speed, hessian_term],
        dtype=np.float64,
    )


def _central_first(func, x, step):
    return (func(x + step) - func(x - step)) / (2.0 * step)


def _central_second(func, x, step):
    return (func(x + step) - 2.0 * func(x) + func(x - step)) / (step * step)


@pytest.mark.required
def test_official_barrier_uses_unnormalized_squared_distance_and_chain_rule():
    distance = 0.31
    active_distance = 0.7
    kappa = 4.25
    actual = np.asarray(_eval_official_distance_terms(distance, active_distance, kappa))
    reference_distance2 = _gap_reference(distance * distance, active_distance * active_distance, kappa)
    np.testing.assert_allclose(actual[:3], reference_distance2, rtol=3.0e-13, atol=1.0e-13)
    expected_distance = np.asarray(
        [
            reference_distance2[0],
            2.0 * distance * reference_distance2[1],
            2.0 * reference_distance2[1] + 4.0 * distance * distance * reference_distance2[2],
        ]
    )
    np.testing.assert_allclose(actual[3:], expected_distance, rtol=3.0e-13, atol=1.0e-13)


def test_official_barrier_force_keeps_growing_at_sub_micron_gaps():
    active_distance = 1.0e-3
    force_magnitudes = [
        -float(_eval_official_distance_terms(distance, active_distance, 1.0)[4])
        for distance in (1.0e-5, 1.0e-8, 1.0e-11)
    ]
    assert np.all(np.isfinite(force_magnitudes))
    assert 0.0 < force_magnitudes[0] < force_magnitudes[1] < force_magnitudes[2]


@pytest.mark.parametrize("physical", [0, 1])
def test_official_barrier_dmin_and_physical_scaling(physical):
    distance = 0.37
    dmin = 0.2
    dhat = 0.4
    kappa = 3.25
    actual = np.asarray(_eval_official_offset_terms(distance, dhat, dmin, kappa, physical))
    shifted_distance2 = distance * distance - dmin * dmin
    shifted_dhat2 = (2.0 * dmin + dhat) * dhat
    reference = _gap_reference(shifted_distance2, shifted_dhat2, kappa)
    if physical:
        reference *= dhat / (shifted_dhat2 * shifted_dhat2)
    np.testing.assert_allclose(actual[:3], reference, rtol=3.0e-13, atol=1.0e-13)
    expected_distance = np.asarray(
        [
            reference[0],
            2.0 * distance * reference[1],
            2.0 * reference[1] + 4.0 * distance * distance * reference[2],
        ]
    )
    np.testing.assert_allclose(actual[3:], expected_distance, rtol=3.0e-13, atol=1.0e-12)

    inactive = np.asarray(_eval_official_offset_terms(dmin + dhat, dhat, dmin, kappa, physical))
    np.testing.assert_array_equal(inactive, np.zeros(6))


@pytest.mark.parametrize("physical", [0, 1])
def test_host_barrier_terms_match_taichi_official_kernel(physical):
    parameters = (0.37, 0.4, 0.2, 3.25)
    expected = np.asarray(_eval_official_offset_terms(*parameters, physical), dtype=np.float64)[3:]
    actual = np.asarray(ipc_barrier_distance_terms_py(*parameters, use_physical_barrier=bool(physical)))
    np.testing.assert_allclose(actual, expected, rtol=3.0e-13, atol=1.0e-12)

    invalid = ipc_barrier_distance_terms_py(0.2, 0.4, 0.2, 3.25, bool(physical))
    assert np.isinf(invalid[0])
    np.testing.assert_array_equal(invalid[1:], [0.0, 0.0])


@pytest.mark.parametrize("gap", [-0.2, 0.1, 0.4])
def test_semi_ipc_projected_augmented_lagrangian_matches_host_law(gap):
    multiplier = 3.0
    penalty = 10.0
    expected = np.asarray(
        [
            *semi_ipc_terms_py(gap, multiplier, penalty),
            semi_ipc_update_multiplier_py(gap, multiplier, penalty),
        ]
    )
    actual = np.asarray(_eval_semi_ipc_terms(gap, multiplier, penalty))
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=1.0e-14)


def test_semi_ipc_energy_derivatives_on_active_branch():
    gap = -0.2
    multiplier = 3.0
    penalty = 10.0
    step = 1.0e-5
    energy = lambda value: semi_ipc_terms_py(value, multiplier, penalty)[0]
    _, gradient, hessian = semi_ipc_terms_py(gap, multiplier, penalty)

    assert gradient == pytest.approx(_central_first(energy, gap, step), rel=1.0e-9)
    assert hessian == pytest.approx(_central_second(energy, gap, step), rel=1.0e-6)


def test_semi_ipc_hash_resolves_many_concurrent_collisions():
    capacity = 512
    key_count = 256
    states = ti.field(dtype=ti.i32, shape=capacity)
    keys = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
    multipliers = ti.field(dtype=ti.f64, shape=capacity)
    count = ti.field(dtype=ti.i32, shape=())
    slots = ti.field(dtype=ti.i32, shape=key_count)
    states.fill(2)
    count[None] = 0

    _insert_colliding_semi_ipc_keys(states, keys, multipliers, count, slots, key_count, capacity)
    inserted_slots = slots.to_numpy()
    assert int(count[None]) == key_count
    assert np.all(inserted_slots >= 0)
    assert np.unique(inserted_slots).size == key_count

    slots.fill(-1)
    _find_colliding_semi_ipc_keys(states, keys, slots, key_count, capacity)
    assert np.all(slots.to_numpy() >= 0)


def test_official_adaptive_barrier_stiffness_rules():
    bbox_diagonal = 1.0
    dhat = 1.0e-3
    average_mass = 1.0
    grad_energy = np.asarray([100.0])
    grad_barrier = np.asarray([-100.0])
    stiffness, maximum = initial_barrier_stiffness(
        bbox_diagonal,
        dhat,
        average_mass,
        grad_energy,
        grad_barrier,
    )
    expected_minimum = (
        1.0e11 * average_mass / (4.0 * 1.0e-16 * clamped_log_barrier_second_derivative(1.0e-16, dhat * dhat))
    )
    assert stiffness == pytest.approx(expected_minimum)
    assert maximum == pytest.approx(100.0 * expected_minimum)

    updated = update_barrier_stiffness(
        1.0e-20,
        1.0e-22,
        max_barrier_stiffness=10.0,
        barrier_stiffness=1.0,
        bbox_diagonal=1.0,
    )
    assert updated == pytest.approx(2.0)
    unchanged = update_barrier_stiffness(
        1.0e-22,
        1.0e-20,
        max_barrier_stiffness=10.0,
        barrier_stiffness=2.0,
        bbox_diagonal=1.0,
    )
    assert unchanged == pytest.approx(2.0)


@pytest.mark.parametrize("physical", [0, 1])
def test_official_barrier_force_magnitude_gradient(physical):
    position = np.asarray([0.0, 1.0e-4, 0.0, 0.0], dtype=np.float64)
    dhat = 1.0e-3
    dmin = 2.0e-5
    kappa = 1.0e2

    def evaluate(value):
        return np.asarray(_eval_point_point_force_magnitude(*value, dhat, dmin, kappa, physical))

    actual = evaluate(position)
    step = 1.0e-9
    finite_difference = np.zeros(4)
    for coordinate in range(4):
        direction = np.zeros(4)
        direction[coordinate] = step
        finite_difference[coordinate] = (evaluate(position + direction)[0] - evaluate(position - direction)[0]) / (
            2.0 * step
        )
    assert actual[0] > 0.0
    np.testing.assert_allclose(actual[1:], finite_difference, rtol=2.0e-7, atol=1.0e-5)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (Barrier, {"kappa": 0.0}),
        (Barrier, {"dhat": -1.0}),
        (Barrier, {"dmin": -1.0}),
        (Barrier, {"dmin": 0.1, "barrier_form": "legacy_gap"}),
        (
            Barrier,
            {"use_physical_barrier": True, "barrier_form": "legacy_gap"},
        ),
        (Barrier, {"kappa": np.nan}),
        (Friction, {"mu": -0.1}),
        (Friction, {"epsv": 0.0}),
        (Friction, {"epsv": np.inf}),
    ],
)
def test_contact_parameter_validation(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_barrier_public_threshold_includes_dmin():
    barrier = Barrier(dhat=0.4, dmin=0.2, kappa=3.0)
    assert float(barrier.threshold) == pytest.approx(0.6)
    assert float(barrier.activation_gap) == pytest.approx(0.4)
    barrier.threshold = 0.75
    assert float(barrier.activation_gap) == pytest.approx(0.55)


def test_barrier_activation_distance_is_available_inside_taichi_kernels():
    barrier = Barrier(dhat=0.4, dmin=0.2, kappa=3.0)
    assert float(_eval_barrier_activation_distance(barrier)) == pytest.approx(0.6)


def test_c1_friction_values_at_zero_inside_and_outside_mollifier():
    epsv = 0.4
    timestep = 0.05
    for speed in (0.0, 0.5 * epsv, 2.0 * epsv):
        actual = np.asarray(_eval_friction_terms(speed, epsv, timestep))
        expected = _friction_reference(speed, epsv, timestep)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=1.0e-14)

    at_zero = np.asarray(_eval_friction_terms(0.0, epsv, timestep))
    assert at_zero[0] == pytest.approx(epsv * timestep / 3.0)
    assert at_zero[1] == 0.0
    assert at_zero[2] == pytest.approx(2.0 / epsv)
    assert at_zero[3] == pytest.approx(2.0 / epsv)
    assert at_zero[4] == pytest.approx(-1.0 / (epsv * epsv))


def test_ipc_toolkit_smooth_friction_mollifier_log_grid():
    """Check the smooth-friction mollifier on an 8-by-8 logarithmic grid."""
    for eps_power in range(-8, 0):
        epsv = 10.0**eps_power
        for speed_power in range(-8, 0):
            speed = 10.0**speed_power
            if speed == 1.0e-8 and epsv == 1.0e-8:
                continue
            actual = np.asarray(_eval_friction_terms(speed, epsv, 1.0))
            expected = _friction_reference(speed, epsv, 1.0)
            np.testing.assert_allclose(actual, expected, rtol=3.0e-13, atol=1.0e-12)
            assert actual[3] * speed == pytest.approx(actual[1], rel=3.0e-13, abs=1.0e-13)


@pytest.mark.parametrize("speed_ratio", [0.0, 0.37, 1.0, 2.0])
def test_friction_mollifier_keeps_exact_scaling_below_old_numeric_floor(
    speed_ratio,
):
    """Positive IPC parameters must not be silently clamped to 1e-30."""
    epsv = 1.0e-20
    timestep = 1.0e-20
    speed = speed_ratio * epsv
    actual = np.asarray(_eval_friction_terms(speed, epsv, timestep))
    expected = _friction_reference(speed, epsv, timestep)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-13, atol=0.0)


@pytest.mark.parametrize("speed_ratio", [0.37, 1.7])
def test_c1_friction_first_and_second_derivatives_by_finite_difference(
    speed_ratio,
):
    epsv = 0.4
    timestep = 0.05
    speed = speed_ratio * epsv
    step = 2.0e-6 * epsv
    second_step = 2.0e-4 * epsv

    def f0(value):
        return float(_eval_friction_terms(value, epsv, timestep)[0])

    def f1(value):
        return float(_eval_friction_terms(value, epsv, timestep)[1])

    def f1_over_speed(value):
        return float(_eval_friction_terms(value, epsv, timestep)[3])

    terms = np.asarray(_eval_friction_terms(speed, epsv, timestep))
    np.testing.assert_allclose(
        _central_first(f0, speed, step),
        timestep * terms[1],
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        _central_second(f0, speed, second_step),
        timestep * terms[2],
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        _central_first(f1, speed, step),
        terms[2],
        rtol=3.0e-10,
        atol=3.0e-10,
    )
    np.testing.assert_allclose(
        _central_first(f1_over_speed, speed, step),
        terms[4],
        rtol=3.0e-10,
        atol=3.0e-9,
    )


@pytest.mark.parametrize("speed_ratio", [0.0, 0.37, 1.0, 1.7])
def test_friction_object_displacement_energy_gradient_hessian(speed_ratio):
    """The field-backed API uses displacement unknowns and velocity friction."""
    epsv = 0.4
    timestep = 0.05
    mu_lambda = 7.5
    friction = Friction(mu=0.6, epsv=epsv)
    displacement = np.array([speed_ratio * epsv * timestep, 0.13 * timestep], dtype=np.float64)

    def evaluate(value):
        return np.asarray(
            _eval_friction_object(
                friction,
                float(value[0]),
                float(value[1]),
                mu_lambda,
                timestep,
            )
        )

    actual = evaluate(displacement)
    step = 2.0e-7 * epsv * timestep
    finite_difference_gradient = np.zeros(2)
    finite_difference_hessian = np.zeros((2, 2))
    for column in range(2):
        direction = np.zeros(2)
        direction[column] = step
        plus = evaluate(displacement + direction)
        minus = evaluate(displacement - direction)
        finite_difference_gradient[column] = (plus[0] - minus[0]) / (2.0 * step)
        finite_difference_hessian[:, column] = (plus[1:3] - minus[1:3]) / (2.0 * step)

    np.testing.assert_allclose(
        actual[1:3],
        finite_difference_gradient,
        rtol=2.0e-7,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        actual[3:].reshape((2, 2)),
        finite_difference_hessian,
        rtol=2.0e-6,
        # The exact seam is only C1; a centered finite difference straddles
        # the two analytic branches and has an O(step) residual.
        atol=5.0e-5,
    )


def test_c1_friction_epsilon_sides_and_seam_continuity():
    epsv = 0.4
    timestep = 0.05
    delta = 1.0e-7 * epsv
    below = np.asarray(_eval_friction_terms(epsv - delta, epsv, timestep))
    at_seam = np.asarray(_eval_friction_terms(epsv, epsv, timestep))
    above = np.asarray(_eval_friction_terms(epsv + delta, epsv, timestep))

    assert at_seam[0] == pytest.approx(epsv * timestep)
    assert at_seam[1] == 1.0
    assert at_seam[2] == 0.0
    assert at_seam[3] == pytest.approx(1.0 / epsv)
    assert at_seam[4] == pytest.approx(-1.0 / (epsv * epsv))

    assert abs(below[0] - at_seam[0]) <= 1.1 * timestep * delta
    assert abs(above[0] - at_seam[0]) <= 1.1 * timestep * delta
    assert abs(below[1] - at_seam[1]) < 2.0e-14
    assert above[1] == at_seam[1]
    assert abs(below[2] - at_seam[2]) < 6.0e-7
    assert above[2] == at_seam[2]
    assert abs(below[3] - at_seam[3]) < 7.0e-7
    assert abs(above[3] - at_seam[3]) < 7.0e-7
    assert abs(below[4] - at_seam[4]) < 1.0e-12
    assert abs(above[4] - at_seam[4]) < 2.0e-6


def test_c1_friction_force_saturation_and_dissipation():
    epsv = 0.4
    timestep = 0.05
    coulomb_limit = 7.5
    zero_potential = float(_eval_friction_terms(0.0, epsv, timestep)[0])
    signed_speeds = np.array([-2.0 * epsv, -epsv, -0.4 * epsv, 0.0, 0.4 * epsv, epsv, 2.0 * epsv])

    positive_force_factors = []
    for signed_speed in signed_speeds:
        speed = abs(float(signed_speed))
        terms = np.asarray(_eval_friction_terms(speed, epsv, timestep))
        force = -coulomb_limit * np.sign(signed_speed) * terms[1]
        positive_force_factors.append(terms[1])

        assert 0.0 <= terms[1] <= 1.0
        assert abs(force) <= coulomb_limit
        assert force * signed_speed <= 0.0
        assert terms[0] >= zero_potential
        if speed >= epsv:
            assert abs(force) == pytest.approx(coulomb_limit)

    assert positive_force_factors[3] == 0.0
    np.testing.assert_allclose(
        positive_force_factors,
        positive_force_factors[::-1],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.required
def test_stick_slip_threshold_for_half_tangent_to_normal_load():
    """Check slope cases around the Coulomb threshold."""
    epsv = 0.4
    timestep = 0.05
    load_ratio = 0.5

    # Below the required coefficient the Coulomb force saturates before it can
    # balance the tangential load, so the contact must slip.
    assert 0.49 < load_ratio
    saturated = float(_eval_friction_terms(2.0 * epsv, epsv, timestep)[1])
    assert saturated == 1.0
    assert 0.49 * saturated < load_ratio

    # At mu=0.5 the equilibrium lies exactly at the smooth stick/slip seam.
    seam = float(_eval_friction_terms(epsv, epsv, timestep)[1])
    assert seam == 1.0
    assert 0.50 * seam == pytest.approx(load_ratio)

    # Above the threshold there is a unique speed inside the regularized
    # sticking branch whose friction balances the load.
    mu = 0.51
    demand = load_ratio / mu
    sticking_speed = epsv * (1.0 - np.sqrt(1.0 - demand))
    assert 0.0 < sticking_speed < epsv
    sticking_factor = float(_eval_friction_terms(sticking_speed, epsv, timestep)[1])
    assert mu * sticking_factor == pytest.approx(load_ratio, rel=1.0e-13)


@pytest.mark.parametrize("speed_ratio", [0.0, 0.5, 1.0 - 1.0e-6, 1.0 + 1.0e-6, 2.0])
def test_fully_implicit_point_plane_jacobian_by_finite_difference(speed_ratio):
    epsv = 0.4
    timestep_velocity_scale = 2.3
    normal = np.array([0.0, 1.0])
    mu = 0.6
    active_gap = 0.8
    gap = 0.31
    kappa = 3.2
    barrier = _gap_reference(gap, active_gap, kappa)
    base_velocity = np.array([speed_ratio * epsv, 0.17])

    def force(displacement):
        current_gap = gap + normal @ displacement
        current_barrier = _gap_reference(current_gap, active_gap, kappa)
        mu_lambda = -mu * current_barrier[1]
        mu_lambda_gradient = -mu * current_barrier[2] * normal
        current_velocity = base_velocity + timestep_velocity_scale * displacement
        result = np.asarray(
            _eval_fully_implicit_point_plane(
                current_velocity[0],
                current_velocity[1],
                normal[0],
                normal[1],
                mu_lambda,
                mu_lambda_gradient[0],
                mu_lambda_gradient[1],
                epsv,
                timestep_velocity_scale,
            )
        )
        return result[:2], result[2:].reshape((2, 2))

    _, analytic = force(np.zeros(2))
    step = 2.0e-7
    finite_difference = np.column_stack(
        [
            (force(step * np.eye(2)[column])[0] - force(-step * np.eye(2)[column])[0]) / (2.0 * step)
            for column in range(2)
        ]
    )
    np.testing.assert_allclose(analytic, finite_difference, rtol=1.0e-6, atol=3.0e-7)


def test_fully_implicit_normal_force_term_makes_jacobian_nonsymmetric():
    epsv = 0.4
    normal = np.array([0.0, 1.0])
    mu = 0.6
    barrier = _gap_reference(0.31, 0.8, 3.2)
    mu_lambda = -mu * barrier[1]
    mu_lambda_gradient = -mu * barrier[2] * normal
    result = np.asarray(
        _eval_fully_implicit_point_plane(
            2.0 * epsv,
            0.0,
            normal[0],
            normal[1],
            mu_lambda,
            mu_lambda_gradient[0],
            mu_lambda_gradient[1],
            epsv,
            2.3,
        )
    )
    jacobian = result[2:].reshape((2, 2))
    assert abs(jacobian[0, 1]) > 1.0e-8
    assert jacobian[1, 0] == pytest.approx(0.0, abs=1.0e-14)
    assert not np.allclose(jacobian, jacobian.T)


def test_stribeck_falloff_derivative_is_the_derivative_of_paper_equation():
    """Catch a missing 1/vs factor in the Stribeck derivative."""
    stribeck_velocity = 0.2
    speed = 0.073
    value, derivative = np.asarray(_eval_stribeck_falloff(speed, stribeck_velocity))
    ratio = speed / stribeck_velocity
    expected = (2.0 * ratio + 1.0) * (ratio - 1.0) ** 2
    expected_derivative = 6.0 * speed * (speed - stribeck_velocity) / stribeck_velocity**3
    assert value == pytest.approx(expected, rel=1.0e-14)
    assert derivative == pytest.approx(expected_derivative, rel=1.0e-14)

    step = 1.0e-7
    plus = float(_eval_stribeck_falloff(speed + step, stribeck_velocity)[0])
    minus = float(_eval_stribeck_falloff(speed - step, stribeck_velocity)[0])
    assert derivative == pytest.approx((plus - minus) / (2.0 * step), rel=2.0e-10)


def test_stribeck_falloff_is_c1_at_static_dynamic_seam():
    """The paper's compact falloff joins the dynamic branch in C1."""
    stribeck_velocity = 0.2
    delta = 1.0e-8 * stribeck_velocity
    below = np.asarray(_eval_stribeck_falloff(stribeck_velocity - delta, stribeck_velocity))
    at_seam = np.asarray(_eval_stribeck_falloff(stribeck_velocity, stribeck_velocity))
    above = np.asarray(_eval_stribeck_falloff(stribeck_velocity + delta, stribeck_velocity))

    np.testing.assert_allclose(at_seam, [0.0, 0.0], atol=1.0e-15)
    assert below[0] == pytest.approx(0.0, abs=4.0e-16)
    assert below[1] == pytest.approx(0.0, abs=4.0e-7)
    np.testing.assert_array_equal(above, [0.0, 0.0])


@pytest.mark.parametrize("profile_id", [0, 1])
def test_full_stribeck_point_plane_jacobian_by_finite_difference(profile_id):
    """Exercise mu_s != mu_d inside the Stribeck transition."""
    normal = np.array([0.0, 1.0])
    active_gap = 0.8
    gap = 0.31
    kappa = 3.2
    velocity_scale = 1.7
    base_velocity = np.array([0.073, -0.04])
    mu_dynamic = 0.31
    mu_static = 0.82
    mu_viscous = 0.07
    stribeck_velocity = 0.2
    epsv = 0.02

    def evaluate(displacement):
        current_gap = gap + normal @ displacement
        barrier = _gap_reference(current_gap, active_gap, kappa)
        normal_force = -barrier[1]
        normal_force_gradient = -barrier[2] * normal
        velocity = base_velocity + velocity_scale * displacement
        result = np.asarray(
            _eval_fully_implicit_stribeck_point_plane(
                velocity[0],
                velocity[1],
                normal_force,
                normal_force_gradient[0],
                normal_force_gradient[1],
                mu_dynamic,
                mu_static,
                mu_viscous,
                stribeck_velocity,
                epsv,
                profile_id,
                velocity_scale,
            )
        )
        return result[:2], result[2:].reshape((2, 2))

    _, analytic = evaluate(np.zeros(2))
    step = 2.0e-7
    finite_difference = np.column_stack(
        [
            (evaluate(step * np.eye(2)[column])[0] - evaluate(-step * np.eye(2)[column])[0]) / (2.0 * step)
            for column in range(2)
        ]
    )
    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-7, atol=2.0e-7)
    assert not np.allclose(analytic, analytic.T)


@pytest.mark.parametrize("profile_id", [0, 1])
@pytest.mark.parametrize("speed", [0.007, 0.073, 0.31])
def test_host_fully_implicit_scalar_law_matches_taichi_and_fd(profile_id, speed):
    normal_force = 2.7
    mu_dynamic = 0.31
    mu_static = 0.82
    mu_viscous = 0.07
    stribeck_velocity = 0.2
    epsv = 0.02
    radial, radial_speed, radial_normal = ipc_fully_implicit_scalar_law_py(
        speed,
        normal_force,
        mu_dynamic,
        mu_static,
        mu_viscous,
        stribeck_velocity,
        epsv,
        profile_id,
    )
    taichi_terms = np.asarray(
        _eval_fully_implicit_stribeck_point_plane(
            speed,
            0.0,
            normal_force,
            0.0,
            0.0,
            mu_dynamic,
            mu_static,
            mu_viscous,
            stribeck_velocity,
            epsv,
            profile_id,
            1.0,
        )
    )
    assert taichi_terms[0] == pytest.approx(radial * speed, rel=2.0e-13)
    assert taichi_terms[2] == pytest.approx(radial + radial_speed * speed, rel=2.0e-13)

    speed_step = 1.0e-7
    lambda_step = 1.0e-7

    def factor(test_speed, test_normal_force):
        return ipc_fully_implicit_scalar_law_py(
            test_speed,
            test_normal_force,
            mu_dynamic,
            mu_static,
            mu_viscous,
            stribeck_velocity,
            epsv,
            profile_id,
        )[0]

    assert radial_speed == pytest.approx(
        (factor(speed + speed_step, normal_force) - factor(speed - speed_step, normal_force)) / (2.0 * speed_step),
        rel=2.0e-8,
        abs=2.0e-8,
    )
    assert radial_normal == pytest.approx(
        (factor(speed, normal_force + lambda_step) - factor(speed, normal_force - lambda_step)) / (2.0 * lambda_step),
        rel=2.0e-8,
        abs=2.0e-8,
    )
