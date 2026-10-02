"""Pinned scalar barrier-potential conformance tests.

The suite checks the ClampedLog potential, barrier force magnitude, analytical
derivatives, activation boundary, and finite-difference consistency.
"""

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.physics_model.contact_model.ipc.IPC import (
    ipc_barrier_force_magnitude,
    ipc_barrier_force_magnitude_gradient,
    ipc_toolkit_barrier_distance2_offset_terms,
    ipc_toolkit_barrier_distance2_terms,
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
def _eval_clamped_log(
    distance_argument: ti.f64,
    activation_argument: ti.f64,
    stiffness: ti.f64,
) -> ti.types.vector(3, ti.f64):
    value, gradient, hessian = ipc_toolkit_barrier_distance2_terms(distance_argument, activation_argument, stiffness)
    return ti.Vector([value, gradient, hessian])


@ti.kernel
def _eval_barrier_potential(
    distance_squared: ti.f64,
    dhat: ti.f64,
    dmin: ti.f64,
    stiffness: ti.f64,
    use_physical_barrier: ti.i32,
) -> ti.types.vector(3, ti.f64):
    value, gradient, hessian = ipc_toolkit_barrier_distance2_offset_terms(
        distance_squared,
        dhat,
        dmin,
        stiffness,
        use_physical_barrier,
    )
    return ti.Vector([value, gradient, hessian])


@ti.kernel
def _eval_force_magnitude(
    distance_squared: ti.f64,
    distance_squared_gradient: ti.types.vector(12, ti.f64),
    dhat: ti.f64,
    dmin: ti.f64,
    stiffness: ti.f64,
    use_physical_barrier: ti.i32,
) -> ti.types.vector(13, ti.f64):
    magnitude = ipc_barrier_force_magnitude(
        distance_squared,
        dhat,
        dmin,
        stiffness,
        use_physical_barrier,
    )
    gradient = ipc_barrier_force_magnitude_gradient(
        distance_squared,
        distance_squared_gradient,
        dhat,
        dmin,
        stiffness,
        use_physical_barrier,
    )
    output = ti.Vector.zero(ti.f64, 13)
    output[0] = magnitude
    i = 0
    while i < 12:
        output[i + 1] = gradient[i]
        i += 1
    return output


def _clamped_log_reference(distance_argument, activation_argument, stiffness):
    """Evaluate the ClampedLogBarrier value and first two derivatives."""
    distance_argument = float(distance_argument)
    activation_argument = float(activation_argument)
    if distance_argument <= 0.0:
        return np.array([np.inf, 0.0, 0.0], dtype=np.float64)
    if distance_argument >= activation_argument:
        return np.zeros(3, dtype=np.float64)

    difference = distance_argument - activation_argument
    log_ratio = np.log(distance_argument / activation_argument)
    activation_over_distance = activation_argument / distance_argument
    return stiffness * np.array(
        [
            -difference * difference * log_ratio,
            (activation_argument - distance_argument) * (2.0 * log_ratio - activation_over_distance + 1.0),
            (activation_over_distance + 2.0) * activation_over_distance - 2.0 * log_ratio - 3.0,
        ],
        dtype=np.float64,
    )


def _barrier_potential_reference(distance_squared, dhat, dmin, stiffness, use_physical_barrier):
    shifted_distance_squared = distance_squared - dmin * dmin
    shifted_dhat_squared = (2.0 * dmin + dhat) * dhat
    terms = _clamped_log_reference(shifted_distance_squared, shifted_dhat_squared, stiffness)
    if use_physical_barrier:
        # ClampedLogBarrier::units(x) == x^2.
        terms *= dhat / (shifted_dhat_squared * shifted_dhat_squared)
    return terms


def _force_reference(
    distance_squared,
    distance_squared_gradient,
    dhat,
    dmin,
    stiffness,
    use_physical_barrier,
):
    _, first_derivative, second_derivative = _barrier_potential_reference(
        distance_squared,
        dhat,
        dmin,
        stiffness,
        use_physical_barrier,
    )
    distance = np.sqrt(distance_squared)
    magnitude = -2.0 * distance * first_derivative
    gradient = -(2.0 * distance * second_derivative + first_derivative / distance) * distance_squared_gradient
    return magnitude, gradient


def _point_point_distance_squared(x, dim):
    p0, p1 = np.split(np.asarray(x, dtype=np.float64), 2)
    assert p0.size == dim and p1.size == dim
    delta = p0 - p1
    return float(delta @ delta), np.concatenate((2.0 * delta, -2.0 * delta))


def _point_edge_distance_squared(x, dim):
    p, e0, e1 = np.split(np.asarray(x, dtype=np.float64), 3)
    assert p.size == dim and e0.size == dim and e1.size == dim
    edge = e1 - e0
    alpha = float((p - e0) @ edge / (edge @ edge))
    assert 0.0 < alpha < 1.0
    residual = p - ((1.0 - alpha) * e0 + alpha * e1)
    gradient = np.concatenate(
        (
            2.0 * residual,
            -2.0 * (1.0 - alpha) * residual,
            -2.0 * alpha * residual,
        )
    )
    return float(residual @ residual), gradient


def _edge_edge_distance_squared(x, dim):
    ea0, ea1, eb0, eb1 = np.split(np.asarray(x, dtype=np.float64), 4)
    assert ea0.size == dim
    direction_a = ea1 - ea0
    direction_b = eb1 - eb0
    rhs = eb0 - ea0
    system = np.array(
        [
            [direction_a @ direction_a, -(direction_a @ direction_b)],
            [direction_a @ direction_b, -(direction_b @ direction_b)],
        ],
        dtype=np.float64,
    )
    alpha, beta = np.linalg.solve(
        system,
        np.array([direction_a @ rhs, direction_b @ rhs], dtype=np.float64),
    )
    assert 0.0 < alpha < 1.0 and 0.0 < beta < 1.0
    residual = (1.0 - alpha) * ea0 + alpha * ea1 - (1.0 - beta) * eb0 - beta * eb1
    gradient = np.concatenate(
        (
            2.0 * (1.0 - alpha) * residual,
            2.0 * alpha * residual,
            -2.0 * (1.0 - beta) * residual,
            -2.0 * beta * residual,
        )
    )
    return float(residual @ residual), gradient


def _point_triangle_distance_squared(x, dim):
    p, t0, t1, t2 = np.split(np.asarray(x, dtype=np.float64), 4)
    assert dim == 3 and p.size == dim
    tangent = np.column_stack((t1 - t0, t2 - t0))
    uv = np.linalg.solve(tangent.T @ tangent, tangent.T @ (p - t0))
    barycentric = np.array([1.0 - uv.sum(), uv[0], uv[1]])
    assert np.all(barycentric > 0.0)
    residual = p - (barycentric[0] * t0 + barycentric[1] * t1 + barycentric[2] * t2)
    gradient = np.concatenate(
        (
            2.0 * residual,
            -2.0 * barycentric[0] * residual,
            -2.0 * barycentric[1] * residual,
            -2.0 * barycentric[2] * residual,
        )
    )
    return float(residual @ residual), gradient


UPSTREAM_FORCE_CASES = [
    pytest.param(
        _point_triangle_distance_squared,
        3,
        np.array(
            [
                0.0,
                1.0e-4,
                0.0,
                -1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                -1.0,
            ]
        ),
        id="point-triangle-3d",
    ),
    pytest.param(
        _edge_edge_distance_squared,
        3,
        np.array(
            [
                -1.0,
                -1.0e-4,
                0.0,
                1.0,
                -1.0e-4,
                0.0,
                0.0,
                1.0e-4,
                -1.0,
                0.0,
                1.0e-4,
                1.0,
            ]
        ),
        id="edge-edge-3d",
    ),
    pytest.param(
        _point_edge_distance_squared,
        3,
        np.array([0.0, 1.0e-4, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        id="point-edge-3d",
    ),
    pytest.param(
        _point_point_distance_squared,
        3,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-4]),
        id="point-point-3d",
    ),
    pytest.param(
        _point_edge_distance_squared,
        2,
        np.array([0.0, 1.0e-4, -1.0, 0.0, 1.0, 0.0]),
        id="point-edge-2d",
    ),
    pytest.param(
        _point_point_distance_squared,
        2,
        np.array([0.0, 0.0, 0.0, 1.0e-4]),
        id="point-point-2d",
    ),
]


@pytest.mark.parametrize("use_distance_squared", [False, True])
def test_clamped_log_value_gradient_hessian_on_upstream_grid(
    use_distance_squared,
):
    """Check ClampedLog values and derivatives on two 10-sample grids."""
    exponent_range = range(-2, 0) if use_distance_squared else range(-5, 0)
    for exponent in exponent_range:
        raw_dhat = 10.0**exponent
        for index in range(10):
            raw_distance = (0.5 + 0.04 * index) * raw_dhat
            distance_argument = raw_distance
            activation_argument = raw_dhat
            if use_distance_squared:
                distance_argument *= distance_argument
                activation_argument *= activation_argument
            expected = _clamped_log_reference(distance_argument, activation_argument, 1.0)
            actual = np.asarray(_eval_clamped_log(distance_argument, activation_argument, 1.0))
            for component, absolute_tolerance in enumerate((1.0e-25, 1.0e-19, 1.0e-12)):
                assert actual[component] == pytest.approx(
                    expected[component],
                    rel=2.0e-12,
                    abs=absolute_tolerance,
                )


@pytest.mark.parametrize("distance_argument", [0.0, -1.0e-8])
def test_clamped_log_infeasible_boundary_matches_pinned_semantics(
    distance_argument,
):
    """Upstream returns +inf energy but zero derivatives for d <= 0."""
    actual = np.asarray(_eval_clamped_log(distance_argument, 1.0e-6, 100.0))
    assert np.isposinf(actual[0])
    np.testing.assert_array_equal(actual[1:], np.zeros(2))


@pytest.mark.parametrize("use_physical_barrier", [0, 1])
def test_dmin_shift_and_physical_scaling(use_physical_barrier):
    dhat = 0.4
    dmin = 0.2
    stiffness = 3.25
    for distance in (0.21, 0.37, 0.59):
        distance_squared = distance * distance
        expected = _barrier_potential_reference(
            distance_squared,
            dhat,
            dmin,
            stiffness,
            use_physical_barrier,
        )
        actual = np.asarray(
            _eval_barrier_potential(
                distance_squared,
                dhat,
                dmin,
                stiffness,
                use_physical_barrier,
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("use_physical_barrier", [0, 1])
def test_activation_boundary_is_inactive_at_and_above_dmin_plus_dhat(
    use_physical_barrier,
):
    dhat = 0.4
    dmin = 0.2
    stiffness = 3.25
    cutoff = dmin + dhat
    for distance in (cutoff, np.nextafter(cutoff, np.inf), 1.25 * cutoff):
        actual = np.asarray(
            _eval_barrier_potential(
                distance * distance,
                dhat,
                dmin,
                stiffness,
                use_physical_barrier,
            )
        )
        np.testing.assert_array_equal(actual, np.zeros(3))

    below = np.asarray(
        _eval_barrier_potential(
            np.nextafter(cutoff, 0.0) ** 2,
            dhat,
            dmin,
            stiffness,
            use_physical_barrier,
        )
    )
    assert below[0] > 0.0
    assert below[1] < 0.0
    assert below[2] > 0.0


@pytest.mark.parametrize("use_physical_barrier", [0, 1])
@pytest.mark.parametrize(("distance_function", "dimension", "positions"), UPSTREAM_FORCE_CASES)
def test_upstream_force_magnitude_and_gradient_cases(
    distance_function,
    dimension,
    positions,
    use_physical_barrier,
):
    dhat = 1.0e-3
    dmin = 0.0
    stiffness = 1.0e2
    distance_squared, distance_squared_gradient = distance_function(positions, dimension)
    padded_gradient = np.zeros(12, dtype=np.float64)
    padded_gradient[: distance_squared_gradient.size] = distance_squared_gradient

    actual = np.asarray(
        _eval_force_magnitude(
            distance_squared,
            padded_gradient,
            dhat,
            dmin,
            stiffness,
            use_physical_barrier,
        )
    )
    expected_magnitude, expected_gradient = _force_reference(
        distance_squared,
        distance_squared_gradient,
        dhat,
        dmin,
        stiffness,
        use_physical_barrier,
    )
    assert actual[0] == pytest.approx(expected_magnitude, rel=2.0e-12)
    np.testing.assert_allclose(
        actual[1 : 1 + positions.size],
        expected_gradient,
        rtol=2.0e-12,
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(actual[1 + positions.size :], 0.0)

    # Use an independent finite-difference check through
    # the complete position -> distance^2 -> force-magnitude chain.
    finite_difference = np.zeros_like(positions)
    step = 1.0e-8
    for coordinate in range(positions.size):
        direction = np.zeros_like(positions)
        direction[coordinate] = step
        plus_distance_squared, plus_gradient = distance_function(positions + direction, dimension)
        minus_distance_squared, minus_gradient = distance_function(positions - direction, dimension)
        plus_padded_gradient = np.zeros(12, dtype=np.float64)
        minus_padded_gradient = np.zeros(12, dtype=np.float64)
        plus_padded_gradient[: plus_gradient.size] = plus_gradient
        minus_padded_gradient[: minus_gradient.size] = minus_gradient
        plus_magnitude = _eval_force_magnitude(
            plus_distance_squared,
            plus_padded_gradient,
            dhat,
            dmin,
            stiffness,
            use_physical_barrier,
        )[0]
        minus_magnitude = _eval_force_magnitude(
            minus_distance_squared,
            minus_padded_gradient,
            dhat,
            dmin,
            stiffness,
            use_physical_barrier,
        )[0]
        finite_difference[coordinate] = (plus_magnitude - minus_magnitude) / (2.0 * step)

    np.testing.assert_allclose(
        actual[1 : 1 + positions.size],
        finite_difference,
        rtol=2.0e-6,
        atol=2.0e-5,
    )
