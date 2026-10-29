"""Boundary interpolation and its gradient must agree at decimal grid endpoints."""

import numpy as np
import pytest
import taichi as ti

from src.utils.ShapeFunctions import ShapeBsplineQ, GShapeBsplineQ


@pytest.mark.parametrize(
    "boundary,edge,weight,gradient", [(1, 0.0, 1.0, -1.0), (2, -1.0, 0.0, 1.0), (3, 1.0, 0.0, -1.0), (4, 0.0, 1.0, 1.0)]
)
def test_quadratic_boundary_value_and_gradient_resist_roundoff(taichi_runtime, boundary, edge, weight, gradient):
    result = ti.Vector.field(2, ti.f64, shape=())

    @ti.kernel
    def sample(position: ti.f64):
        result[None] = ti.Vector(
            [ShapeBsplineQ(position, 0.2, 200.0, boundary), GShapeBsplineQ(position, 0.2, 200.0, boundary)]
        )

    endpoint = 0.2 + edge / 200.0
    for position in [endpoint, np.nextafter(endpoint, -np.inf), np.nextafter(endpoint, np.inf)]:
        sample(position)
        np.testing.assert_allclose(result[None], [weight, gradient * 200.0], atol=1e-12, rtol=0.0)
    sample(endpoint + (-1.0 if boundary in [1, 2] else 1.0) * 0.0001)
    np.testing.assert_array_equal(result[None], [0.0, 0.0])
