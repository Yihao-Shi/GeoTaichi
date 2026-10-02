"""Finite-difference oracles for stress invariants and the Lode angle."""

import math

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import (
    ComputeLodeAngle,
    ComputeStressInvariantJ2,
    ComputeStressInvariantJ3,
    DqDsigma,
    Dj2Dsigma,
    Dj3Dsigma,
    DlodeDsigma,
    EquivalentDeviatoricStress,
)
from src.utils.TypeDefination import vec6f


pytestmark = [pytest.mark.cpu, pytest.mark.materials]


@ti.kernel
def _evaluate_invariants(
    stress_values: ti.types.ndarray(dtype=ti.f64, ndim=1),
    invariant_values: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gradients: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    stress = vec6f(
        stress_values[0],
        stress_values[1],
        stress_values[2],
        stress_values[3],
        stress_values[4],
        stress_values[5],
    )
    invariant_values[0] = ComputeStressInvariantJ2(stress)
    invariant_values[1] = ComputeStressInvariantJ3(stress)
    invariant_values[2] = ComputeLodeAngle(stress)
    invariant_values[3] = EquivalentDeviatoricStress(stress)
    gradient_j2 = Dj2Dsigma(stress)
    gradient_j3 = Dj3Dsigma(stress)
    gradient_lode = DlodeDsigma(stress)
    gradient_q = DqDsigma(stress)
    for component in ti.static(range(6)):
        gradients[0, component] = gradient_j2[component]
        gradients[1, component] = gradient_j3[component]
        gradients[2, component] = gradient_lode[component]
        gradients[3, component] = gradient_q[component]


def _deviator(stress):
    result = np.asarray(stress, dtype=np.float64).copy()
    mean = np.mean(result[:3])
    result[:3] -= mean
    return result


def _j2(stress):
    deviator = _deviator(stress)
    return 0.5 * np.dot(deviator[:3], deviator[:3]) + np.dot(
        deviator[3:], deviator[3:]
    )


def _j3(stress):
    s11, s22, s33, s12, s23, s13 = _deviator(stress)
    return (
        s11 * s22 * s33
        + 2.0 * s12 * s23 * s13
        - s33 * s12 * s12
        - s11 * s23 * s23
        - s22 * s13 * s13
    )


def _lode_angle(stress):
    j2 = _j2(stress)
    argument = 1.5 * math.sqrt(3.0) * _j3(stress) / (j2 ** 1.5)
    return math.acos(np.clip(argument, -1.0, 1.0)) / 3.0


def _equivalent_deviatoric_stress(stress):
    return math.sqrt(3.0 * _j2(stress))


def _central_gradient(function, stress, step=2.0e-4):
    gradient = np.zeros(6, dtype=np.float64)
    for component in range(6):
        plus = np.asarray(stress, dtype=np.float64).copy()
        minus = plus.copy()
        plus[component] += step
        minus[component] -= step
        gradient[component] = (function(plus) - function(minus)) / (2.0 * step)
    return gradient


def test_stress_invariant_gradients_match_finite_difference(taichi_material_cpu):
    stress = np.asarray([1.1, -0.7, 0.4, 0.35, -0.22, 0.17])
    invariants = np.zeros(4, dtype=np.float64)
    gradients = np.zeros((4, 6), dtype=np.float64)

    _evaluate_invariants(stress, invariants, gradients)

    expected_values = np.asarray(
        [
            _j2(stress),
            _j3(stress),
            _lode_angle(stress),
            _equivalent_deviatoric_stress(stress),
        ]
    )
    expected_gradients = np.stack(
        (
            _central_gradient(_j2, stress),
            _central_gradient(_j3, stress),
            _central_gradient(_lode_angle, stress),
            _central_gradient(_equivalent_deviatoric_stress, stress),
        )
    )
    # Production constitutive derivatives are represented as symmetric
    # second-order tensors and are contracted with ``voigt_tensor_dot``.
    # A coordinate derivative with respect to an independently perturbed
    # stored shear component is therefore twice the corresponding tensor
    # entry.
    tensor_gradients_as_coordinate_gradients = gradients.copy()
    tensor_gradients_as_coordinate_gradients[:, 3:] *= 2.0
    np.testing.assert_allclose(
        invariants, expected_values, rtol=4.0e-6, atol=4.0e-7
    )
    np.testing.assert_allclose(
        tensor_gradients_as_coordinate_gradients,
        expected_gradients,
        rtol=3.0e-4,
        atol=3.0e-5,
    )
