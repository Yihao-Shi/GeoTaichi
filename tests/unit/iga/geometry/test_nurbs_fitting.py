"""Deterministic curve and surface fitting checks for the IGA layer."""

import numpy as np

from src.nurbs.Fitting import (
    compute_params_curve,
    compute_params_surface,
    global_approximate_bspline_curve,
    global_approximate_bspline_surface,
    global_interpolate_bspline_curve,
    global_interpolate_bspline_surface,
)


def test_global_curve_interpolation_reproduces_every_data_point():
    points = np.asarray(
        [[0.0, 0.0], [3.0, 4.0], [-1.0, 4.0], [-4.0, 0.0], [-5.0, -1.0]]
    )
    parameters = compute_params_curve(points)
    curve = global_interpolate_bspline_curve(points=points, degree=3)

    interpolated = np.asarray(
        [curve.single_point(parameter) for parameter in parameters]
    )
    np.testing.assert_allclose(interpolated, points, rtol=0.0, atol=5.0e-12)


def test_global_curve_approximation_preserves_fixed_endpoints():
    points = np.asarray(
        [[0.0, 0.0], [3.0, 4.0], [-1.0, 4.0], [-4.0, 0.0], [-5.0, -1.0]]
    )
    curve = global_approximate_bspline_curve(points=points, degree=3)
    distances = curve.distance(points)

    assert curve.control_points.shape == (4, 2)
    np.testing.assert_allclose(curve.single_point(0.0), points[0], atol=2.0e-13)
    np.testing.assert_allclose(curve.single_point(1.0), points[-1], atol=2.0e-12)
    assert np.all(np.isfinite(distances))
    assert distances[0] < 2.0e-12
    assert distances[-1] < 2.0e-12


def _bilinear_data_grid():
    return np.asarray(
        [
            [u, v, 0.2 * u * v]
            for v in (0.0, 1.0, 2.0, 4.0)
            for u in (0.0, 1.0, 2.0, 3.0)
        ]
    )


def test_global_surface_interpolation_reproduces_every_data_point():
    points = _bilinear_data_grid()
    reshaped = points.reshape((4, 4, 3))
    parameters_u, parameters_v = compute_params_surface(reshaped, 4, 4)
    surface = global_interpolate_bspline_surface(
        degree_u=2,
        degree_v=2,
        num_datapt_u=4,
        num_datapt_v=4,
        points=points,
    )

    interpolated = np.asarray(
        [
            surface.single_point(parameter_u, parameter_v)
            for parameter_v in parameters_v
            for parameter_u in parameters_u
        ]
    )
    np.testing.assert_allclose(interpolated, points, rtol=0.0, atol=5.0e-12)


def test_global_surface_approximation_uses_consistent_u_v_dimensions():
    parameters_u = (0.0, 0.5, 1.5, 2.5, 3.0)
    parameters_v = (0.0, 1.0, 2.0, 4.0)
    points = np.asarray(
        [
            [u, v, 0.2 * u * v]
            for v in parameters_v
            for u in parameters_u
        ]
    )
    surface = global_approximate_bspline_surface(
        degree_u=2,
        degree_v=2,
        num_datapt_u=5,
        num_datapt_v=4,
        points=points,
    )

    assert surface.control_points.shape == (12, 3)
    np.testing.assert_allclose(
        surface.single_point(0.0, 0.0), points[0], rtol=0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        surface.single_point(1.0, 1.0), points[-1], rtol=0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        surface.distance(points), 0.0, rtol=0.0, atol=3.0e-12
    )
