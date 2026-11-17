"""Shared Python scalar-validation contracts."""

import pytest
import numpy as np

from src.utils.linalg import positive, _quadratic_bspline, _cubic_bspline, _tetra_barycentric
from src.utils.linalg import newton

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def test_positive_accepts_finite_values_and_optional_zero():
    assert positive("density", "1.5") == 1.5
    assert positive("bending_stiffness", 0.0, allow_zero=True) == 0.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_positive_rejects_nonpositive_and_nonfinite_values(value):
    with pytest.raises(ValueError, match="density must be finite and positive"):
        positive("density", value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_positive_allow_zero_still_rejects_negative_and_nonfinite_values(value):
    with pytest.raises(ValueError, match="bending_stiffness must be finite and non-negative"):
        positive("bending_stiffness", value, allow_zero=True)


@pytest.mark.parametrize("basis", [_quadratic_bspline, _cubic_bspline])
@pytest.mark.parametrize("boundary_type", range(5))
def test_preprocessing_bspline_gradient_matches_its_weight(basis, boundary_type):
    for point in (-1.21, -0.84, -0.23, 0.17, 0.78, 1.31):
        _, gradient = basis(point, 0.0, 1.0, boundary_type)
        h = 1e-6
        finite_difference = (
            basis(point + h, 0.0, 1.0, boundary_type)[0] - basis(point - h, 0.0, 1.0, boundary_type)[0]
        ) / (2 * h)
        assert gradient == pytest.approx(finite_difference, abs=1e-8)


@pytest.mark.parametrize("basis", [_quadratic_bspline, _cubic_bspline])
def test_interior_bspline_partition_of_unity(basis):
    weights_and_gradients = [basis(0.17, node, 2.0, 0) for node in np.arange(-2, 2.5, 0.5)]
    np.testing.assert_allclose(np.sum(weights_and_gradients, axis=0), [1.0, 0.0], atol=1e-14)


def test_tetrahedral_barycentric_coordinates_reproduce_affine_geometry():
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    point = np.array([0.4, 0.6, 0.8])
    weights, gradients = _tetra_barycentric(point, vertices)
    np.testing.assert_allclose(weights @ vertices, point)
    np.testing.assert_allclose(gradients.T @ vertices, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(gradients.sum(axis=0), 0.0, atol=1e-14)


def test_array_newton_full_output_preserves_roots_and_convergence_flags():
    targets = np.array([2.0, 9.0])
    result = newton(
        lambda values: values * values - targets,
        np.array([1.0, 2.0]),
        fprime=lambda values: 2.0 * values,
        full_output=True,
    )
    np.testing.assert_allclose(result.root, np.sqrt(targets), atol=1e-12)
    assert result.converged.all()
    assert not result.zero_der.any()
