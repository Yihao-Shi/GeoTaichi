"""Deterministic geometry checks extracted from the legacy IGA plot scripts."""

from math import sqrt

import numpy as np

from src.nurbs.NurbsBasis import NurbsBasis
from src.nurbs.NurbsPrimitives import NurbsCurve, NurbsSurface


def _unit_circle_curve():
    control_points = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    knot_vector = np.asarray(
        [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0]
    )
    weights = np.ones(9)
    weights[1::2] = sqrt(2.0) / 2.0

    curve = NurbsCurve(closed=True)
    curve.degree = 2
    curve.knot_vector = knot_vector
    curve.control_points = control_points
    curve.weights = weights
    return curve


def test_rational_basis_is_partition_of_unity_with_zero_derivative_sum():
    curve = _unit_circle_curve()
    sample_knots = (0.01, 0.13, 0.24, 0.26, 0.37, 0.49, 0.51, 0.63, 0.74, 0.76, 0.88, 0.99)

    for knot in sample_knots:
        values_and_derivatives = np.asarray(
            [
                NurbsBasis(
                    control_point,
                    curve.degree,
                    knot,
                    curve.knot_vector,
                    curve.weights,
                )
                for control_point in range(curve.control_points.shape[0])
            ]
        )
        values = values_and_derivatives[:, 0]
        derivatives = values_and_derivatives[:, 1]

        assert np.all(values >= -1.0e-15)
        assert np.isclose(values.sum(), 1.0, rtol=0.0, atol=2.0e-14)
        assert np.isclose(derivatives.sum(), 0.0, rtol=0.0, atol=2.0e-13)


def test_rational_knot_refinement_preserves_curve_geometry():
    curve = NurbsCurve()
    curve.degree = 2
    curve.knot_vector = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    curve.control_points = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    curve.weights = np.asarray([1.0, sqrt(2.0) / 2.0, 1.0])

    sample_knots = np.linspace(0.0, 1.0, 33)
    original = np.asarray([curve.single_point(knot) for knot in sample_knots])
    curve.refine_knot(density=2)
    refined = np.asarray([curve.single_point(knot) for knot in sample_knots])

    assert curve.control_points.shape == (6, 3)
    assert curve.knot_vector.shape == (9,)
    assert np.all(curve.weights > 0.0)
    np.testing.assert_allclose(refined, original, rtol=0.0, atol=3.0e-15)


def test_closed_unit_circle_distance_and_curvature_match_analytic_values():
    curve = _unit_circle_curve()
    query = np.asarray([1.0, 1.0 + sqrt(2.0) / 2.0, 0.0])

    distance = float(curve.distance(query, closed=True)[0])

    assert np.isclose(distance, 1.0 - sqrt(2.0) / 2.0, atol=2.0e-12)
    assert np.isclose(curve.curvature(0.5), 1.0, atol=2.0e-12)
    np.testing.assert_allclose(
        curve.single_point(0.5), [2.0, 1.0, 0.0], rtol=0.0, atol=2.0e-14
    )


def test_planar_surface_projection_and_distance_match_analytic_values():
    surface = NurbsSurface()
    surface.degree = [2, 2]
    surface.knot_vector_u = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    surface.knot_vector_v = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    surface.control_points = np.asarray(
        [[u, v, 0.0] for v in (0.0, 0.5, 1.0) for u in (0.0, 0.5, 1.0)]
    )
    surface.weights = np.ones(9)
    query = np.asarray([0.25, 0.75, 2.0])

    assert np.isclose(float(surface.distance(query)[0]), 2.0, atol=2.0e-13)
    np.testing.assert_allclose(
        surface.projection(query)[0],
        [0.25, 0.75, 0.0],
        rtol=0.0,
        atol=2.0e-13,
    )

