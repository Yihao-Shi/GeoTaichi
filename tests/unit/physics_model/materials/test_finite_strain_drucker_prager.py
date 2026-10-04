"""Finite-strain Drucker--Prager parameter and analytic-tangent tests."""

import math

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.DruckerPrager import (
    FiniteStrainDruckerPragerModel,
)

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def _model(**overrides):
    parameters = {
        "Density": 1800.0,
        "YoungModulus": 2.0e5,
        "PoissonRatio": 0.3,
        "FrictionAngle": 30.0,
        "DilationAngle": 30.0,
        "Cohesion": 1200.0,
        "dpType": "Circumscribed",
    }
    parameters.update(overrides)
    model = FiniteStrainDruckerPragerModel()
    model.model_initialize(parameters)
    model.allocate_state(1)
    return model


def _evaluator(model):
    energy = ti.field(ti.f64, shape=())
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())
    projected = ti.Matrix.field(3, 3, ti.f64, shape=())

    @ti.kernel
    def evaluate_kernel(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        deformation = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            deformation[row, column] = values[row, column]
        energy[None] = model.Psi_at(0, deformation)
        stress[None] = model.first_piola_stress_at(0, deformation)
        tangent[None] = model.first_piola_tangent_at(0, deformation)
        projected[None] = model.project_elastic_deformation(0, deformation)

    def evaluate(deformation):
        evaluate_kernel(np.ascontiguousarray(deformation, dtype=np.float64))
        return (
            float(energy[None]),
            stress.to_numpy()[()],
            tangent.to_numpy()[()],
            projected.to_numpy()[()],
        )

    return evaluate


def _flatten_column_major(matrix):
    return np.asarray(matrix, dtype=np.float64).T.reshape(-1)


def test_classical_circumscribed_cone_parameters(taichi_material_cpu):
    model = _model()
    phi = math.radians(30.0)
    expected_q = 6.0 * math.sin(phi) / (math.sqrt(3.0) * (3.0 - math.sin(phi)))
    expected_k = 6.0 * 1200.0 * math.cos(phi) / (math.sqrt(3.0) * (3.0 - math.sin(phi)))
    assert model.q_friction == pytest.approx(expected_q)
    assert model.k_cohesion == pytest.approx(expected_k)
    assert model.alpha == pytest.approx(math.sqrt(2.0) * expected_q / 3.0)
    assert model.cohesive_yield_stress == pytest.approx(math.sqrt(2.0) * expected_k)


def test_nonassociated_flow_is_not_silently_accepted():
    with pytest.raises(ValueError, match="associated flow"):
        _model(DilationAngle=5.0)


@pytest.mark.parametrize(
    "deformation",
    [
        np.diag(np.exp([-0.001, 0.0005, -0.002])),
        np.array([[1.12, 0.06, -0.01], [0.02, 0.92, 0.04], [0.01, -0.03, 0.84]]),
        np.eye(3) * np.exp(0.05),
    ],
    ids=["elastic", "cone", "apex"],
)
def test_associated_incremental_energy_has_analytic_symmetric_tangent(
    taichi_material_cpu,
    deformation,
):
    model = _model()
    evaluate = _evaluator(model)
    energy, pk1, tangent, projected = evaluate(deformation)

    assert np.isfinite(energy)
    assert np.linalg.det(projected) > 0.0
    np.testing.assert_allclose(tangent, tangent.T, rtol=2.0e-10, atol=2.0e-8)

    step = 2.0e-7
    numerical_energy_gradient = np.zeros(9, dtype=np.float64)
    numerical = np.zeros((9, 9), dtype=np.float64)
    for column in range(9):
        component = column // 3
        row = column % 3
        plus = deformation.copy()
        minus = deformation.copy()
        plus[row, component] += step
        minus[row, component] -= step
        plus_energy, plus_stress = evaluate(plus)[:2]
        minus_energy, minus_stress = evaluate(minus)[:2]
        numerical_energy_gradient[column] = (plus_energy - minus_energy) / (2.0 * step)
        numerical[:, column] = (_flatten_column_major(plus_stress) - _flatten_column_major(minus_stress)) / (2.0 * step)
    np.testing.assert_allclose(
        _flatten_column_major(pk1),
        numerical_energy_gradient,
        rtol=5.0e-7,
        atol=2.0e-4,
    )
    np.testing.assert_allclose(tangent, numerical, rtol=3.0e-4, atol=2.0e-2)
    assert np.linalg.norm(pk1) > 0.0


def test_returned_stress_satisfies_classical_dp_cone(taichi_material_cpu):
    model = _model()
    evaluate = _evaluator(model)
    deformation = np.diag(np.exp([0.20, -0.12, -0.25]))
    _, _, _, projected = evaluate(deformation)

    singular_values = np.linalg.svd(projected, compute_uv=False)
    strain = np.log(singular_values)
    trace = strain.sum()
    deviatoric = strain - trace / 3.0
    kirchhoff = 2.0 * model.shear * deviatoric + model.bulk * trace
    mean = kirchhoff.mean()
    sqrt_j2 = np.linalg.norm(kirchhoff - mean) / math.sqrt(2.0)
    yield_value = sqrt_j2 + model.q_friction * mean - model.k_cohesion
    assert yield_value <= 1.0e-7 * max(model.k_cohesion, 1.0)


def test_hydrostatic_apex_commit_uses_associated_multiplier(
    taichi_material_cpu,
):
    model = _model()
    committed = ti.Matrix.field(3, 3, ti.f64, shape=())
    trial_log_stretch = 0.05

    @ti.kernel
    def commit():
        trial = ti.Matrix.identity(ti.f64, 3) * ti.exp(trial_log_stretch)
        committed[None] = model.commit_state(0, trial)

    commit()
    projected_singular_values = np.linalg.svd(committed.to_numpy()[()], compute_uv=False)
    projected_trace = float(np.log(projected_singular_values).sum())
    trial_trace = 3.0 * trial_log_stretch
    expected_multiplier = (trial_trace - model.trace_apex) / (3.0 * model.alpha)

    assert projected_trace == pytest.approx(model.trace_apex, abs=1.0e-12)
    assert float(model.equivalent_plastic_strain[0]) == pytest.approx(math.sqrt(2.0 / 3.0) * expected_multiplier)
    assert float(model.volumetric_plastic_strain[0]) == pytest.approx(trial_trace - model.trace_apex)


def test_total_deformation_contract_preserves_plastic_part_and_tangent(
    taichi_material_cpu,
):
    model = _model()
    committed_total = ti.Matrix.field(3, 3, ti.f64, shape=())
    energy = ti.field(ti.f64, shape=())
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())

    @ti.kernel
    def commit(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        total = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            total[row, column] = values[row, column]
        committed_total[None] = model.commit_total_state(0, total)

    @ti.kernel
    def evaluate(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        total = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            total[row, column] = values[row, column]
        energy[None] = model.total_strain_energy_density_at(0, total)
        stress[None] = model.total_first_piola_stress_at(0, total)
        tangent[None] = model.total_first_piola_tangent_at(0, total)

    first_total = np.diag(np.exp([0.20, -0.12, -0.25]))
    commit(np.ascontiguousarray(first_total))
    np.testing.assert_allclose(committed_total[None], first_total, atol=1.0e-14)
    plastic_inverse = model.plastic_deformation_inverse.to_numpy()[0]
    assert not np.allclose(plastic_inverse, np.eye(3))

    trial_total = first_total @ np.array([[1.01, 0.015, 0.0], [0.0, 0.995, 0.01], [0.0, 0.0, 1.0]])
    evaluate(np.ascontiguousarray(trial_total))
    analytic = tangent.to_numpy()[()]
    total_pk1 = stress.to_numpy()[()]

    step = 2.0e-7
    numerical = np.zeros((9, 9), dtype=np.float64)
    numerical_energy_gradient = np.zeros(9, dtype=np.float64)
    for column in range(9):
        material_axis = column // 3
        spatial_axis = column % 3
        plus = trial_total.copy()
        minus = trial_total.copy()
        plus[spatial_axis, material_axis] += step
        minus[spatial_axis, material_axis] -= step
        evaluate(np.ascontiguousarray(plus))
        plus_energy = float(energy[None])
        plus_stress = stress.to_numpy()[()].copy()
        evaluate(np.ascontiguousarray(minus))
        minus_energy = float(energy[None])
        minus_stress = stress.to_numpy()[()].copy()
        numerical_energy_gradient[column] = (plus_energy - minus_energy) / (2.0 * step)
        numerical[:, column] = (_flatten_column_major(plus_stress) - _flatten_column_major(minus_stress)) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=5.0e-4, atol=3.0e-2)
    np.testing.assert_allclose(_flatten_column_major(total_pk1), numerical_energy_gradient, rtol=5.0e-7, atol=2.0e-4)

    elastic_trial = trial_total @ plastic_inverse
    evaluate_elastic = _evaluator(model)
    elastic_pk1 = evaluate_elastic(elastic_trial)[1]
    total_cauchy = total_pk1 @ trial_total.T / np.linalg.det(trial_total)
    elastic_cauchy = elastic_pk1 @ elastic_trial.T / np.linalg.det(elastic_trial)
    np.testing.assert_allclose(total_cauchy, elastic_cauchy, rtol=2.0e-12, atol=2.0e-10)
