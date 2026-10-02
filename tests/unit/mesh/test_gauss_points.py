"""Analytical integration contracts for rectangle/simplex quadrature."""

import numpy as np
import pytest

from src.mesh.GaussPoint import (
    GaussPointInRectangle,
    GaussPointInTriangle,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.required,
    pytest.mark.geometry,
    pytest.mark.cpu,
]


@pytest.mark.parametrize("point_count", [1, 2, 3, 5, 10])
def test_gauss_legendre_integrates_maximal_monomial(point_count):
    quadrature = GaussPointInRectangle(
        gauss_point=point_count, dimemsion=1
    )
    quadrature.create_gauss_point(taichi_field=False)
    degree = 2 * point_count - 1

    value = np.sum(
        quadrature.weight * quadrature.gpcoords[:, 0] ** degree
    )
    exact = 0.0 if degree % 2 else 2.0 / (degree + 1)

    assert value == pytest.approx(exact, abs=2.0e-13)
    assert sum(quadrature.weight) == pytest.approx(2.0)


def test_rectangle_tensor_product_integrates_polynomial():
    quadrature = GaussPointInRectangle(
        gauss_point=[2, 3], dimemsion=2
    )
    quadrature.create_gauss_point(taichi_field=False)

    x = quadrature.gpcoords[:, 0]
    y = quadrature.gpcoords[:, 1]
    value = np.sum(quadrature.weight * x**2 * y**4)

    assert value == pytest.approx(4.0 / 15.0, rel=1.0e-13)
    assert quadrature.get_ith_weight(0) == quadrature.weight[0]
    assert quadrature.get_ith_coord(0) == tuple(
        quadrature.gpcoords[0]
    )


@pytest.mark.parametrize("point_count", [1, 3, 7, 13])
def test_triangle_rules_integrate_constant_and_linear(point_count):
    quadrature = GaussPointInTriangle(
        gauss_point=point_count, dimemsion=2
    )
    quadrature.create_gauss_point()

    assert sum(quadrature.weight) == pytest.approx(1.0, abs=2.0e-12)
    np.testing.assert_allclose(
        np.sum(
            quadrature.gpcoords
            * quadrature.weight[:, np.newaxis],
            axis=0,
        ),
        [1.0 / 3.0, 1.0 / 3.0],
        rtol=0.0,
        atol=2.0e-12,
    )
    assert quadrature.get_ith_weight(0) == quadrature.weight[0]
    assert quadrature.get_ith_coord(0) == tuple(
        quadrature.gpcoords[0]
    )


@pytest.mark.parametrize("point_count", [1, 4, 5])
def test_tetrahedron_rules_integrate_constant_and_linear(point_count):
    quadrature = GaussPointInTriangle(
        gauss_point=point_count, dimemsion=3
    )
    quadrature.create_gauss_point()

    assert quadrature.gpcoords.shape == (point_count, 3)
    assert sum(quadrature.weight) == pytest.approx(1.0, abs=1.0e-12)
    np.testing.assert_allclose(
        np.sum(
            quadrature.gpcoords
            * quadrature.weight[:, np.newaxis],
            axis=0,
        ),
        [0.25, 0.25, 0.25],
        rtol=0.0,
        atol=2.0e-8,
    )


def test_quadrature_rejects_invalid_configuration():
    with pytest.raises(RuntimeError, match="gauss points"):
        GaussPointInTriangle(2, dimemsion=2)
    with pytest.raises(RuntimeError, match="gauss points"):
        GaussPointInTriangle(3, dimemsion=3)
    with pytest.raises(RuntimeError, match="dimension"):
        GaussPointInRectangle([2, 2], dimemsion=3)
