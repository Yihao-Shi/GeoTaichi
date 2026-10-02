"""Unit tests for the finite-strain Neo-Hookean energy API."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.NeoHookean import (
    NeoHookeanModel,
)


pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def _neo_hookean(**kwargs):
    return NeoHookeanModel().initialize_from_kwargs(**kwargs)


def _make_evaluator(model, dimension):
    energy = ti.field(dtype=ti.f64, shape=())
    pk1 = ti.Matrix.field(
        dimension, dimension, dtype=ti.f64, shape=()
    )
    tangent = ti.Matrix.field(
        dimension * dimension,
        dimension * dimension,
        dtype=ti.f64,
        shape=(),
    )
    von_mises = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate_kernel(
        deformation_gradient: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        deformation = ti.Matrix.zero(
            ti.f64, dimension, dimension
        )
        for row, column in ti.static(
            ti.ndrange(dimension, dimension)
        ):
            deformation[row, column] = deformation_gradient[row, column]
        energy[None] = model.Psi(deformation)
        pk1[None] = model.first_piola_stress(deformation)
        tangent[None] = model.first_piola_tangent(deformation)
        von_mises[None] = model.VonMises(deformation)

    def evaluate(deformation_gradient):
        evaluate_kernel(
            np.ascontiguousarray(deformation_gradient, dtype=np.float64)
        )
        return (
            float(energy[None]),
            pk1.to_numpy()[()],
            tangent.to_numpy()[()],
            float(von_mises[None]),
        )

    return evaluate


def _closed_form_response(model, deformation_gradient):
    dimension = deformation_gradient.shape[0]
    jacobian = np.linalg.det(deformation_gradient)
    inverse_transpose = np.linalg.inv(deformation_gradient).T
    log_j = np.log(jacobian)
    energy = (
        0.5
        * model.shear
        * (np.sum(deformation_gradient**2) - dimension)
        - model.shear * log_j
        + 0.5 * model.lame_lambda * log_j**2
    )
    pk1 = model.shear * (
        deformation_gradient - inverse_transpose
    ) + model.lame_lambda * log_j * inverse_transpose
    return energy, pk1


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "density": 1100.0,
            "young_modulus": 2.4e5,
            "poisson_ratio": 0.28,
        },
        {
            "material_parameters": {
                "Density": 1100.0,
                "YoungModulus": 2.4e5,
                "PoissonRatio": 0.28,
            }
        },
    ],
    ids=["keyword-aliases", "canonical-material-dict"],
)
def test_initialize_from_kwargs_builds_neo_hookean_model(kwargs):
    model = _neo_hookean(**kwargs)

    assert isinstance(model, NeoHookeanModel)
    assert model.density == 1100.0
    assert model.young_modulus == 2.4e5
    assert model.poisson_ratio == 0.28
    assert model.mu_ == pytest.approx(2.4e5 / (2.0 * 1.28))
    assert model.lambda_ == pytest.approx(
        2.4e5 * 0.28 / (1.28 * (1.0 - 2.0 * 0.28))
    )


def test_initialize_from_kwargs_requires_young_modulus():
    with pytest.raises(KeyError, match="YoungModulus"):
        _neo_hookean(density=1000.0, poisson_ratio=0.3)


@pytest.mark.parametrize("dimension", [2, 3])
def test_identity_has_zero_energy_and_zero_pk1(
    taichi_material_cpu, dimension
):
    model = _neo_hookean(
        density=1000.0,
        young_modulus=2.4e5,
        poisson_ratio=0.28,
    )
    evaluate = _make_evaluator(model, dimension)

    energy, pk1, tangent, von_mises = evaluate(np.eye(dimension))

    assert energy == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(pk1, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(tangent, tangent.T, atol=1.0e-12)
    assert von_mises == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize(
    "deformation_gradient",
    [
        np.array([[1.12, 0.04], [0.02, 0.91]]),
        np.array(
            [
                [1.08, 0.03, 0.00],
                [0.01, 0.94, 0.02],
                [0.00, 0.01, 1.04],
            ]
        ),
    ],
    ids=["2d", "3d"],
)
def test_energy_and_pk1_match_closed_form(
    taichi_material_cpu, deformation_gradient
):
    model = _neo_hookean(
        density=1000.0,
        young_modulus=2.4e5,
        poisson_ratio=0.28,
    )
    evaluate = _make_evaluator(model, deformation_gradient.shape[0])

    energy, pk1, tangent, von_mises = evaluate(deformation_gradient)
    expected_energy, expected_pk1 = _closed_form_response(
        model, deformation_gradient
    )

    assert energy == pytest.approx(expected_energy, rel=1.0e-12)
    np.testing.assert_allclose(
        pk1, expected_pk1, rtol=1.0e-12, atol=1.0e-10
    )
    np.testing.assert_allclose(
        tangent, tangent.T, rtol=1.0e-12, atol=1.0e-10
    )
    assert von_mises > 0.0


@pytest.mark.parametrize(
    "deformation_gradient",
    [
        np.array([[1.12, 0.04], [0.02, 0.91]]),
        np.array(
            [
                [1.08, 0.03, 0.00],
                [0.01, 0.94, 0.02],
                [0.00, 0.01, 1.04],
            ]
        ),
    ],
    ids=["2d", "3d"],
)
def test_pk1_and_tangent_match_central_finite_differences(
    taichi_material_cpu, deformation_gradient
):
    model = _neo_hookean(
        density=1000.0,
        young_modulus=2.4e5,
        poisson_ratio=0.28,
    )
    dimension = deformation_gradient.shape[0]
    evaluate = _make_evaluator(model, dimension)
    energy, pk1, tangent, _ = evaluate(deformation_gradient)
    del energy

    step = 1.0e-6
    energy_gradient = np.zeros(dimension * dimension)
    stress_jacobian = np.zeros_like(tangent)
    for column in range(dimension * dimension):
        row_index = column % dimension
        column_index = column // dimension
        plus = deformation_gradient.copy()
        minus = deformation_gradient.copy()
        plus[row_index, column_index] += step
        minus[row_index, column_index] -= step
        energy_plus, pk1_plus, _, _ = evaluate(plus)
        energy_minus, pk1_minus, _, _ = evaluate(minus)
        energy_gradient[column] = (
            energy_plus - energy_minus
        ) / (2.0 * step)
        stress_jacobian[:, column] = (
            pk1_plus.flatten(order="F")
            - pk1_minus.flatten(order="F")
        ) / (2.0 * step)

    np.testing.assert_allclose(
        pk1.flatten(order="F"),
        energy_gradient,
        rtol=2.0e-8,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        tangent,
        stress_jacobian,
        rtol=2.0e-8,
        atol=2.0e-5,
    )


def test_neo_hookean_records_maximum_sound_speed():
    model = _neo_hookean(
        density=1000.0,
        young_modulus=2.4e5,
        poisson_ratio=0.28,
    )
    expected = np.sqrt(
        model.young
        * (1.0 - model.poisson)
        / (
            (1.0 + model.poisson)
            * (1.0 - 2.0 * model.poisson)
            * model.density
        )
    )

    assert model.max_sound_speed == pytest.approx(expected)
