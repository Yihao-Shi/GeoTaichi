"""Mathematical contracts for the surface FEM used by NURBS fitting."""

import numpy as np
import pytest

from src.nurbs.element.LinearQuadrilateralElement import (
    LinearQuadrilateralElement,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.fem,
    pytest.mark.geometry,
    pytest.mark.cpu,
]


def test_linear_quadrilateral_reproduces_affine_map_and_reference_area():
    reference = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    affine_gradient = np.asarray(
        [[1.4, 0.2], [-0.3, 0.8]], dtype=np.float64
    )
    translation = np.asarray([0.4, -0.2], dtype=np.float64)
    current = reference @ affine_gradient.T + translation

    element = LinearQuadrilateralElement(
        current,
        reference,
        np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        gauss_point_number=2,
    )

    for natural_coordinates in element.gauss_points:
        shape = element.shape_function(natural_coordinates)
        shape_gradient = element.dshape_dnat(natural_coordinates)
        np.testing.assert_allclose(np.sum(shape), 1.0, atol=2.0e-15)
        np.testing.assert_allclose(
            np.sum(shape_gradient, axis=0),
            np.zeros(2),
            atol=2.0e-15,
        )

    deformation_gradients = element.compute_deformation_gradient(current)
    expected_gradients = np.repeat(
        affine_gradient[None, :, :], element.gauss_number, axis=0
    )
    np.testing.assert_allclose(
        deformation_gradients,
        expected_gradients,
        rtol=2.0e-15,
        atol=2.0e-15,
    )

    # The 2x2 Gauss rule has four unit weights on [-1, 1]^2.
    np.testing.assert_allclose(
        element.reference_area,
        np.ones(4),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.sum(element.reference_area), 4.0, rtol=0.0, atol=2.0e-15
    )
