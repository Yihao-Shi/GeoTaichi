"""Dense oracles for runtime finite-strain tangent assembly."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.MaterialKernel import (
    compute_d2I2dF2,
    get_invariant_hessian,
)


pytestmark = [pytest.mark.materials, pytest.mark.cpu]

_LINEAR = np.array([0.37, -0.19, 0.43], dtype=np.float64)
_QUADRATIC = np.array(
    [
        [0.11, -0.07, 0.05],
        [-0.07, 0.13, -0.09],
        [0.05, -0.09, 0.17],
    ],
    dtype=np.float64,
)


def _invariants(deformation_gradient):
    right_cauchy_green = deformation_gradient.T @ deformation_gradient
    invariant_i1 = np.sum(deformation_gradient**2)
    invariant_i2 = 0.5 * (
        invariant_i1**2
        - np.sum(right_cauchy_green * right_cauchy_green)
    )
    return np.array(
        [
            invariant_i1,
            invariant_i2,
            np.linalg.det(deformation_gradient),
        ]
    )


def _invariant_pk1(deformation_gradient):
    invariants = _invariants(deformation_gradient)
    energy_gradient = _LINEAR + _QUADRATIC @ invariants
    invariant_i1 = invariants[0]
    gradient_i1 = 2.0 * deformation_gradient
    gradient_i2 = 2.0 * (
        invariant_i1 * deformation_gradient
        - deformation_gradient
        @ deformation_gradient.T
        @ deformation_gradient
    )
    gradient_j = (
        invariants[2] * np.linalg.inv(deformation_gradient).T
    )
    return (
        energy_gradient[0] * gradient_i1
        + energy_gradient[1] * gradient_i2
        + energy_gradient[2] * gradient_j
    )


def _dense_jacobian(vector_function, deformation_gradient, step):
    dimension = deformation_gradient.shape[0]
    size = dimension * dimension
    jacobian = np.empty((size, size), dtype=np.float64)
    for column in range(size):
        perturbation = np.zeros_like(deformation_gradient)
        row_index = column % dimension
        column_index = column // dimension
        perturbation[row_index, column_index] = step
        plus = vector_function(deformation_gradient + perturbation)
        minus = vector_function(deformation_gradient - perturbation)
        jacobian[:, column] = (
            (plus - minus) / (2.0 * step)
        ).flatten(order="F")
    return jacobian


def test_runtime_invariant_hessians_match_random_dense_oracles(
    taichi_material_cpu,
):
    hessian_i2 = ti.Matrix.field(9, 9, dtype=ti.f64, shape=())
    invariant_tangent = ti.Matrix.field(
        9, 9, dtype=ti.f64, shape=()
    )

    @ti.kernel
    def evaluate(
        deformation_gradient: ti.types.ndarray(
            dtype=ti.f64, ndim=2
        ),
    ):
        deformation = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            deformation[row, column] = deformation_gradient[row, column]
        invariants = ti.Vector(
            [
                (
                    deformation.transpose() @ deformation
                ).trace(),
                0.0,
                deformation.determinant(),
            ]
        )
        right_cauchy_green = (
            deformation.transpose() @ deformation
        )
        invariants[1] = 0.5 * (
            invariants[0] * invariants[0]
            - (
                right_cauchy_green @ right_cauchy_green
            ).trace()
        )
        derivatives = ti.Vector(
            [
                _LINEAR[0]
                + _QUADRATIC[0, 0] * invariants[0]
                + _QUADRATIC[0, 1] * invariants[1]
                + _QUADRATIC[0, 2] * invariants[2],
                _LINEAR[1]
                + _QUADRATIC[1, 0] * invariants[0]
                + _QUADRATIC[1, 1] * invariants[1]
                + _QUADRATIC[1, 2] * invariants[2],
                _LINEAR[2]
                + _QUADRATIC[2, 0] * invariants[0]
                + _QUADRATIC[2, 1] * invariants[1]
                + _QUADRATIC[2, 2] * invariants[2],
            ]
        )
        hessian_i2[None] = compute_d2I2dF2(deformation)
        invariant_tangent[None] = get_invariant_hessian(
            deformation,
            derivatives[0],
            derivatives[1],
            derivatives[2],
            _QUADRATIC[0, 0],
            _QUADRATIC[0, 1],
            _QUADRATIC[0, 2],
            _QUADRATIC[1, 1],
            _QUADRATIC[1, 2],
            _QUADRATIC[2, 2],
        )

    generator = np.random.default_rng(20260724)
    for _ in range(4):
        deformation = (
            np.eye(3) + 0.12 * generator.standard_normal((3, 3))
        )
        assert np.linalg.det(deformation) > 0.0
        evaluate(np.ascontiguousarray(deformation))

        expected_i2_hessian = _dense_jacobian(
            lambda value: 2.0
            * (
                np.sum(value**2) * value
                - value @ value.T @ value
            ),
            deformation,
            1.0e-6,
        )
        expected_tangent = _dense_jacobian(
            _invariant_pk1, deformation, 1.0e-6
        )

        np.testing.assert_allclose(
            hessian_i2.to_numpy()[()],
            expected_i2_hessian,
            rtol=2.0e-8,
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            invariant_tangent.to_numpy()[()],
            expected_tangent,
            rtol=2.0e-8,
            atol=3.0e-8,
        )
        np.testing.assert_allclose(
            invariant_tangent.to_numpy()[()],
            invariant_tangent.to_numpy()[()].T,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
