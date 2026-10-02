"""Unit tests for the remaining finite-strain elastic material models."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.Gent import Gent
from src.physics_model.consititutive_model.finite_strain.HenckyElastic import (
    HenckyElasticModel,
)
from src.physics_model.consititutive_model.finite_strain.Hydrogel import (
    Hydrogel,
)
from src.physics_model.consititutive_model.finite_strain.MooneyRivlin import (
    MooneyRivlin,
)


pytestmark = [pytest.mark.materials, pytest.mark.cpu]

_DENSITY = 1100.0
_YOUNG = 2.4e5
_POISSON = 0.28


def _shear_modulus():
    return _YOUNG / (2.0 * (1.0 + _POISSON))


def _make_model(model_name, *, secondary=False):
    common = {
        "Density": _DENSITY,
        "YoungModulus": _YOUNG,
        "PoissonRatio": _POISSON,
    }
    if model_name == "mooney_rivlin":
        shear = _shear_modulus()
        coefficient = [[0.0, 0.0], [0.5 * shear, 0.0]]
        if secondary:
            coefficient[0][1] = 0.2 * shear
            coefficient[1][1] = 0.02 * shear
        model = MooneyRivlin()
        model.model_initialize(
            {**common, "Coefficient": coefficient}
        )
        return model
    if model_name == "gent":
        model = Gent()
        model.model_initialize(
            {
                **common,
                "Tensile1": 30.0,
                "Tensile2": 25.0 if secondary else 0.0,
            }
        )
        return model
    if model_name == "hencky":
        model = HenckyElasticModel()
        model.model_initialize(common)
        return model
    if model_name == "hydrogel":
        model = Hydrogel()
        model.model_initialize({**common, "Tensile": 30.0})
        return model
    raise AssertionError(f"unknown model {model_name}")


def _make_pk1_evaluator(model):
    output = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate_kernel(
        deformation_gradient: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        deformation = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            deformation[row, column] = deformation_gradient[row, column]
        output[None] = model.first_piola_stress(deformation)

    def evaluate(deformation_gradient):
        evaluate_kernel(
            np.ascontiguousarray(deformation_gradient, dtype=np.float64)
        )
        return output.to_numpy()[()]

    return evaluate


def _make_tangent_evaluator(model):
    output = ti.Matrix.field(9, 9, dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate_kernel(
        deformation_gradient: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        deformation = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            deformation[row, column] = deformation_gradient[row, column]
        output[None] = model.first_piola_tangent(deformation)

    def evaluate(deformation_gradient):
        evaluate_kernel(
            np.ascontiguousarray(deformation_gradient, dtype=np.float64)
        )
        return output.to_numpy()[()]

    return evaluate


def _invariants(deformation_gradient):
    right_cauchy_green = deformation_gradient.T @ deformation_gradient
    first = np.trace(right_cauchy_green)
    second = 0.5 * (
        first**2 - np.trace(right_cauchy_green @ right_cauchy_green)
    )
    return first, second


def _volumetric_log_energy(model, jacobian):
    log_j = np.log(jacobian)
    return (
        0.5 * model.lame_lambda * log_j**2
        - model.shear * log_j
    )


def _quadratic_volumetric_energy(model, jacobian):
    return (
        0.25 * model.lame_lambda * (jacobian**2 - 1.0)
        - 0.5 * model.lame_lambda * np.log(jacobian)
    )


def _reference_energy(model_name, model, deformation_gradient):
    jacobian = np.linalg.det(deformation_gradient)
    if jacobian <= 0.0:
        raise ValueError("reference energy requires det(F) > 0")
    first, second = _invariants(deformation_gradient)

    if model_name == "mooney_rivlin":
        coefficient = model.coeff.to_numpy()
        energy = 0.0
        for i in range(coefficient.shape[0]):
            for j in range(coefficient.shape[1]):
                energy += (
                    coefficient[i, j]
                    * (first - 3.0) ** i
                    * (second - 3.0) ** j
                )
        return energy + _volumetric_log_energy(model, jacobian)

    if model_name == "gent":
        isochoric_first = jacobian ** (-2.0 / 3.0) * first
        isochoric_second = jacobian ** (-4.0 / 3.0) * second
        energy = 0.0
        if model.Jm1 > 0.0:
            energy -= (
                0.5
                * model.shear
                * model.Jm1
                * np.log(
                    1.0
                    - (isochoric_first - 3.0) / model.Jm1
                )
            )
        if model.Jm2 > 0.0:
            energy -= (
                0.5
                * model.shear
                * model.Jm2
                * np.log(
                    1.0
                    - (isochoric_second - 3.0) / model.Jm2
                )
            )
        return energy + _quadratic_volumetric_energy(model, jacobian)

    if model_name == "hydrogel":
        isochoric_first = jacobian ** (-2.0 / 3.0) * first
        energy = (
            -0.5
            * model.shear
            * model.Jm
            * np.log(1.0 - (isochoric_first - 3.0) / model.Jm)
        )
        return energy + _quadratic_volumetric_energy(model, jacobian)

    if model_name == "hencky":
        principal_strain = np.log(
            np.linalg.svd(
                deformation_gradient, compute_uv=False
            )
        )
        normal_diagonal = model.bulk + 4.0 * model.shear / 3.0
        normal_off_diagonal = model.bulk - 2.0 * model.shear / 3.0
        elasticity = np.full((3, 3), normal_off_diagonal)
        np.fill_diagonal(elasticity, normal_diagonal)
        return 0.5 * principal_strain @ elasticity @ principal_strain

    raise AssertionError(f"unknown model {model_name}")


def _energy_gradient(energy, deformation_gradient, step):
    gradient = np.zeros(9, dtype=np.float64)
    for component in range(9):
        row = component % 3
        column = component // 3
        plus = deformation_gradient.copy()
        minus = deformation_gradient.copy()
        plus[row, column] += step
        minus[row, column] -= step
        gradient[component] = (energy(plus) - energy(minus)) / (
            2.0 * step
        )
    return gradient


def _energy_hessian(energy, deformation_gradient, step):
    hessian = np.zeros((9, 9), dtype=np.float64)
    base = energy(deformation_gradient)
    for first_component in range(9):
        first_row = first_component % 3
        first_column = first_component // 3
        plus = deformation_gradient.copy()
        minus = deformation_gradient.copy()
        plus[first_row, first_column] += step
        minus[first_row, first_column] -= step
        hessian[first_component, first_component] = (
            energy(plus) - 2.0 * base + energy(minus)
        ) / step**2
        for second_component in range(first_component + 1, 9):
            second_row = second_component % 3
            second_column = second_component // 3
            plus_plus = deformation_gradient.copy()
            plus_minus = deformation_gradient.copy()
            minus_plus = deformation_gradient.copy()
            minus_minus = deformation_gradient.copy()
            plus_plus[first_row, first_column] += step
            plus_plus[second_row, second_column] += step
            plus_minus[first_row, first_column] += step
            plus_minus[second_row, second_column] -= step
            minus_plus[first_row, first_column] -= step
            minus_plus[second_row, second_column] += step
            minus_minus[first_row, first_column] -= step
            minus_minus[second_row, second_column] -= step
            value = (
                energy(plus_plus)
                - energy(plus_minus)
                - energy(minus_plus)
                + energy(minus_minus)
            ) / (4.0 * step**2)
            hessian[first_component, second_component] = value
            hessian[second_component, first_component] = value
    return hessian


def _pk1_jacobian(evaluate_pk1, deformation_gradient, step):
    jacobian = np.zeros((9, 9), dtype=np.float64)
    for component in range(9):
        row = component % 3
        column = component // 3
        plus = deformation_gradient.copy()
        minus = deformation_gradient.copy()
        plus[row, column] += step
        minus[row, column] -= step
        jacobian[:, component] = (
            evaluate_pk1(plus).flatten(order="F")
            - evaluate_pk1(minus).flatten(order="F")
        ) / (2.0 * step)
    return jacobian


@pytest.mark.parametrize(
    "model_name",
    ["mooney_rivlin", "gent", "hencky", "hydrogel"],
)
def test_model_initialize_sets_elastic_parameters_and_state_schema(
    model_name,
):
    model = _make_model(model_name)

    assert model.density == _DENSITY
    assert model.young == _YOUNG
    assert model.poisson == _POISSON
    assert model.shear == pytest.approx(_shear_modulus())
    assert model.bulk == pytest.approx(
        _YOUNG / (3.0 * (1.0 - 2.0 * _POISSON))
    )
    assert set(model.define_state_vars()) == {
        "stress0",
        "deformation_gradient",
    }

    if model_name == "mooney_rivlin":
        assert model.coeff.to_numpy()[1, 0] == pytest.approx(
            0.5 * _shear_modulus()
        )
    elif model_name == "gent":
        assert model.Jm1 == 30.0
        assert model.Jm2 == 0.0
    elif model_name == "hydrogel":
        assert model.Jm == 30.0


@pytest.mark.parametrize(
    ("model", "incomplete_parameters"),
    [
        (MooneyRivlin, {"Coefficient": [[0.0, 0.0], [1.0, 0.0]]}),
        (Gent, {"Tensile1": 10.0}),
        (HenckyElasticModel, {}),
        (Hydrogel, {"Tensile": 10.0}),
    ],
)
def test_model_initialize_requires_young_modulus(
    model, incomplete_parameters
):
    with pytest.raises(KeyError, match="YoungModulus"):
        model().model_initialize(incomplete_parameters)


def test_mooney_rivlin_requires_two_by_two_coefficient_rows():
    with pytest.raises(ValueError, match="dimension"):
        MooneyRivlin().model_initialize(
            {
                "YoungModulus": _YOUNG,
                "Coefficient": [[0.0, 0.0]],
            }
        )


@pytest.mark.parametrize(
    "model_name", ["mooney_rivlin", "hencky", "hydrogel"]
)
def test_identity_is_stress_free(
    taichi_material_cpu, model_name
):
    model = _make_model(model_name)
    evaluate_pk1 = _make_pk1_evaluator(model)

    np.testing.assert_allclose(
        evaluate_pk1(np.eye(3)), 0.0, atol=2.0e-5
    )


def test_gent_identity_is_stress_free(taichi_material_cpu):
    model = _make_model("gent")
    evaluate_pk1 = _make_pk1_evaluator(model)

    np.testing.assert_allclose(
        evaluate_pk1(np.eye(3)), 0.0, atol=2.0e-5
    )


@pytest.mark.parametrize(
    ("model_name", "deformation_gradient"),
    [
        (
            "mooney_rivlin",
            np.array(
                [
                    [1.08, 0.05, 0.01],
                    [0.02, 0.94, 0.03],
                    [0.01, 0.02, 1.03],
                ]
            ),
        ),
        (
            "gent",
            np.array(
                [
                    [1.08, 0.05, 0.01],
                    [0.02, 0.94, 0.03],
                    [0.01, 0.02, 1.03],
                ]
            ),
        ),
        (
            "hencky",
            np.array(
                [
                    [1.12, 0.07, 0.01],
                    [0.02, 0.91, 0.04],
                    [0.01, 0.03, 1.05],
                ]
            ),
        ),
        (
            "hydrogel",
            np.array(
                [
                    [1.08, 0.05, 0.01],
                    [0.02, 0.94, 0.03],
                    [0.01, 0.02, 1.03],
                ]
            ),
        ),
    ],
)
def test_pk1_and_numerical_tangent_are_energy_derivatives(
    taichi_material_cpu, model_name, deformation_gradient
):
    model = _make_model(model_name)
    evaluate_pk1 = _make_pk1_evaluator(model)
    energy = lambda deformation: _reference_energy(
        model_name, model, deformation
    )

    pk1 = evaluate_pk1(deformation_gradient).flatten(order="F")
    energy_gradient = _energy_gradient(
        energy, deformation_gradient, 1.0e-6
    )
    numerical_tangent = _pk1_jacobian(
        evaluate_pk1, deformation_gradient, 2.0e-4
    )
    energy_hessian = _energy_hessian(
        energy, deformation_gradient, 2.0e-4
    )

    np.testing.assert_allclose(
        pk1, energy_gradient, rtol=3.0e-5, atol=0.15
    )
    tangent_error = np.linalg.norm(
        numerical_tangent - energy_hessian
    ) / max(np.linalg.norm(energy_hessian), 1.0)
    symmetry_error = np.linalg.norm(
        numerical_tangent - numerical_tangent.T
    ) / max(np.linalg.norm(numerical_tangent), 1.0)
    assert tangent_error < 4.0e-3
    assert symmetry_error < 4.0e-3


def test_mooney_rivlin_secondary_invariant_is_energy_consistent(
    taichi_material_cpu,
):
    model = _make_model("mooney_rivlin", secondary=True)
    deformation = np.array(
        [
            [1.08, 0.12, 0.01],
            [0.02, 0.94, 0.07],
            [0.03, 0.01, 1.03],
        ]
    )
    evaluate_pk1 = _make_pk1_evaluator(model)
    energy = lambda value: _reference_energy(
        "mooney_rivlin", model, value
    )

    np.testing.assert_allclose(
        evaluate_pk1(deformation).flatten(order="F"),
        _energy_gradient(energy, deformation, 1.0e-6),
        rtol=3.0e-5,
        atol=0.15,
    )


def test_gent_secondary_invariant_is_energy_consistent(
    taichi_material_cpu,
):
    model = _make_model("gent", secondary=True)
    deformation = np.array(
        [
            [1.08, 0.12, 0.01],
            [0.02, 0.94, 0.07],
            [0.03, 0.01, 1.03],
        ]
    )
    evaluate_pk1 = _make_pk1_evaluator(model)
    energy = lambda value: _reference_energy("gent", model, value)

    np.testing.assert_allclose(
        evaluate_pk1(deformation).flatten(order="F"),
        _energy_gradient(energy, deformation, 1.0e-6),
        rtol=3.0e-5,
        atol=0.15,
    )


@pytest.mark.parametrize("model_name", ["gent", "hydrogel"])
def test_locking_stress_grows_near_valid_extensibility_limit(
    taichi_material_cpu, model_name
):
    model = _make_model(model_name)
    evaluate_pk1 = _make_pk1_evaluator(model)
    moderate = np.diag([1.2, 1.0 / np.sqrt(1.2), 1.0 / np.sqrt(1.2)])
    near_limit = np.diag([5.5, 1.0 / np.sqrt(5.5), 1.0 / np.sqrt(5.5)])

    moderate_norm = np.linalg.norm(evaluate_pk1(moderate))
    near_limit_norm = np.linalg.norm(evaluate_pk1(near_limit))

    assert np.isfinite(moderate_norm)
    assert np.isfinite(near_limit_norm)
    assert near_limit_norm > 5.0 * moderate_norm


@pytest.mark.parametrize(
    "model_name", ["mooney_rivlin", "gent", "hencky", "hydrogel"]
)
def test_strong_positive_jacobian_compression_remains_finite(
    taichi_material_cpu, model_name
):
    model = _make_model(model_name)
    evaluate_pk1 = _make_pk1_evaluator(model)
    deformation = np.diag([0.2, 0.9, 1.1])

    assert np.all(np.isfinite(evaluate_pk1(deformation)))


@pytest.mark.parametrize(
    "model_name", ["mooney_rivlin", "gent", "hencky", "hydrogel"]
)
def test_model_rejects_nonpositive_jacobian(
    taichi_material_cpu, model_name
):
    model = _make_model(model_name)
    evaluate_pk1 = _make_pk1_evaluator(model)

    with pytest.raises((ValueError, RuntimeError)):
        evaluate_pk1(np.diag([-1.0, 1.0, 1.0]))


@pytest.mark.parametrize("model_name", ["gent", "hydrogel"])
def test_locking_model_rejects_nonpositive_extensibility(model_name):
    common = {
        "YoungModulus": _YOUNG,
        "PoissonRatio": _POISSON,
    }
    with pytest.raises(ValueError):
        if model_name == "gent":
            Gent().model_initialize(
                {**common, "Tensile1": 0.0, "Tensile2": 0.0}
            )
        else:
            Hydrogel().model_initialize({**common, "Tensile": 0.0})


@pytest.mark.parametrize(
    "model_name", ["mooney_rivlin", "gent", "hencky", "hydrogel"]
)
def test_model_exposes_strain_energy_density(
    taichi_material_cpu, model_name
):
    model = _make_model(model_name)
    output = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate():
        output[None] = model.strain_energy_density(
            ti.Matrix.identity(ti.f64, 3)
        )

    evaluate()
    assert np.isfinite(output[None])


@pytest.mark.parametrize(
    ("model_name", "secondary", "deformation_gradient"),
    [
        (
            "mooney_rivlin",
            True,
            np.array(
                [
                    [1.08, 0.12, 0.01],
                    [0.02, 0.94, 0.07],
                    [0.03, 0.01, 1.03],
                ]
            ),
        ),
        (
            "gent",
            True,
            np.array(
                [
                    [1.08, 0.12, 0.01],
                    [0.02, 0.94, 0.07],
                    [0.03, 0.01, 1.03],
                ]
            ),
        ),
        (
            "hencky",
            False,
            np.array(
                [
                    [1.12, 0.07, 0.01],
                    [0.02, 0.91, 0.04],
                    [0.01, 0.03, 1.05],
                ]
            ),
        ),
        (
            "hydrogel",
            False,
            np.array(
                [
                    [1.08, 0.05, 0.01],
                    [0.02, 0.94, 0.03],
                    [0.01, 0.02, 1.03],
                ]
            ),
        ),
    ],
)
def test_analytic_tangent_matches_pk1_finite_difference_oracle(
    taichi_material_cpu,
    model_name,
    secondary,
    deformation_gradient,
):
    model = _make_model(model_name, secondary=secondary)
    evaluate_pk1 = _make_pk1_evaluator(model)
    evaluate_tangent = _make_tangent_evaluator(model)

    analytic_tangent = evaluate_tangent(deformation_gradient)
    finite_difference_oracle = _pk1_jacobian(
        evaluate_pk1, deformation_gradient, 1.0e-6
    )

    np.testing.assert_allclose(
        analytic_tangent,
        analytic_tangent.T,
        rtol=2.0e-12,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        analytic_tangent,
        finite_difference_oracle,
        rtol=5.0e-6,
        atol=0.1,
    )


def test_hencky_tangent_has_exact_repeated_stretch_limit(
    taichi_material_cpu,
):
    model = _make_model("hencky")
    evaluate_tangent = _make_tangent_evaluator(model)
    tangent = evaluate_tangent(np.eye(3))
    expected = np.zeros((9, 9), dtype=np.float64)
    for a in range(3):
        for i in range(3):
            row = a * 3 + i
            for b in range(3):
                for j in range(3):
                    column = b * 3 + j
                    expected[row, column] = (
                        model.lame_lambda * (i == a) * (j == b)
                        + model.shear * (i == j) * (a == b)
                        + model.shear * (i == b) * (j == a)
                    )

    np.testing.assert_allclose(
        tangent, expected, rtol=2.0e-12, atol=2.0e-8
    )

    repeated_stretch = np.diag([1.2, 1.2, 0.9])
    repeated_tangent = evaluate_tangent(repeated_stretch)
    finite_difference_oracle = _pk1_jacobian(
        _make_pk1_evaluator(model), repeated_stretch, 1.0e-6
    )
    np.testing.assert_allclose(
        repeated_tangent,
        repeated_tangent.T,
        rtol=2.0e-12,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        repeated_tangent,
        finite_difference_oracle,
        rtol=5.0e-6,
        atol=0.1,
    )


@pytest.mark.parametrize(
    "model_name", ["mooney_rivlin", "gent", "hencky", "hydrogel"]
)
def test_model_records_maximum_sound_speed(model_name):
    model = _make_model(model_name)
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
