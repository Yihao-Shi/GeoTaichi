"""Finite-strain Modified Cam-Clay return-map checks."""

import math

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.ModifiedCamClay import (
    FiniteStrainModifiedCamClayModel,
)

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def _model():
    model = FiniteStrainModifiedCamClayModel().initialize_from_kwargs(
        Density=1800.0,
        PoissonRatio=0.3,
        StressRatio=1.2,
        **{
            "lambda": 0.20,
            "kappa": 0.05,
            "void_ratio_ref": 0.8,
            "pc0": 2.0e5,
            "OverConsolidationRatio": 2.0,
        },
    )
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


def _yield_value(model, elastic_deformation, pc):
    strain = np.log(np.linalg.svd(elastic_deformation, compute_uv=False))
    trace = strain.sum()
    deviatoric = strain - trace / 3.0
    scale = math.exp(
        -trace / model.kappa_star
        + model.reference_shear_modulus * deviatoric.dot(deviatoric) / (model.kappa_star * model.initial_pressure)
    )
    pressure = model.initial_pressure * scale
    q = math.sqrt(6.0) * model.reference_shear_modulus * scale * np.linalg.norm(deviatoric)
    return q * q / model.stress_ratio**2 + pressure * (pressure - pc)


def _converge_lagged_hardening(model, deformation):
    @ti.kernel
    def begin():
        model.begin_lagged_incremental_potential(0)

    @ti.kernel
    def refresh(values: ti.types.ndarray(dtype=ti.f64, ndim=2)) -> ti.f64:
        trial = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            trial[row, column] = values[row, column]
        return model.refresh_lagged_incremental_potential(0, trial)

    begin()
    error = math.inf
    for _ in range(model.lagged_max_iterations):
        error = float(refresh(np.ascontiguousarray(deformation, dtype=np.float64)))
        if error <= model.lagged_tolerance:
            return error
    raise AssertionError(f"lagged MCC hardening did not converge: {error}")


def test_mcc_parameters_use_double_logarithmic_indices(taichi_material_cpu):
    model = _model()
    assert model.lambda_star == pytest.approx(0.20 / 1.8)
    assert model.kappa_star == pytest.approx(0.05 / 1.8)
    assert model.initial_pressure == pytest.approx(1.0e5)
    assert model.initial_preconsolidation_pressure == pytest.approx(2.0e5)
    assert model.history_state_size == 12
    assert model.has_incremental_potential
    assert model.has_symmetric_tangent


def test_mcc_plastic_return_has_9x9_consistent_tangent(taichi_material_cpu):
    model = _model()
    evaluate = _evaluator(model)
    deformation = np.diag(np.exp([0.01, -0.02, -0.04]))
    energy, pk1, tangent, projected = evaluate(deformation)

    assert np.isfinite(energy)
    assert tangent.shape == (9, 9)
    np.testing.assert_allclose(tangent, tangent.T, rtol=2.0e-8, atol=1.0e-4)
    assert np.linalg.det(projected) > 0.0
    assert (
        abs(
            _yield_value(
                model,
                projected,
                float(model.lagged_preconsolidation_pressure[0]),
            )
        )
        < 2.0e-5 * model.initial_preconsolidation_pressure**2
    )

    step = 2.0e-7
    energy_gradient = np.zeros(9, dtype=np.float64)
    numerical = np.zeros((9, 9), dtype=np.float64)
    for column in range(9):
        component = column // 3
        row = column % 3
        plus = deformation.copy()
        minus = deformation.copy()
        plus[row, component] += step
        minus[row, component] -= step
        plus_energy, plus_stress, _, _ = evaluate(plus)
        minus_energy, minus_stress, _, _ = evaluate(minus)
        energy_gradient[column] = (plus_energy - minus_energy) / (2.0 * step)
        numerical[:, column] = (_flatten_column_major(plus_stress) - _flatten_column_major(minus_stress)) / (2.0 * step)
    np.testing.assert_allclose(energy_gradient, _flatten_column_major(pk1), rtol=2.0e-5, atol=2.0e-2)
    np.testing.assert_allclose(tangent, numerical, rtol=2.0e-3, atol=2.0e2)
    assert np.linalg.norm(pk1) > 0.0


def test_mcc_commit_hardens_with_plastic_compression(taichi_material_cpu):
    model = _model()
    committed = ti.Matrix.field(3, 3, ti.f64, shape=())
    deformation = np.diag(np.exp([0.01, -0.02, -0.04]))
    _converge_lagged_hardening(model, deformation)

    @ti.kernel
    def commit():
        trial = ti.Matrix.zero(ti.f64, 3, 3)
        trial[0, 0] = ti.exp(0.01)
        trial[1, 1] = ti.exp(-0.02)
        trial[2, 2] = ti.exp(-0.04)
        committed[None] = model.commit_state(0, trial)

    commit()
    assert float(model.preconsolidation_pressure[0]) > 2.0e5
    assert float(model.volumetric_plastic_strain[0]) > 0.0
    assert float(model.equivalent_plastic_strain[0]) > 0.0
    assert np.linalg.det(committed.to_numpy()[()]) > 0.0
    assert (
        abs(
            _yield_value(
                model,
                committed.to_numpy()[()],
                float(model.preconsolidation_pressure[0]),
            )
        )
        < 2.0e-5 * float(model.preconsolidation_pressure[0]) ** 2
    )


def test_mcc_hydrostatic_cap_return_handles_repeated_stretches(
    taichi_material_cpu,
):
    model = _model()
    evaluate = _evaluator(model)
    deformation = np.eye(3) * math.exp(-0.01)
    _, _, tangent, projected = evaluate(deformation)
    pc = float(model.lagged_preconsolidation_pressure[0])

    assert np.all(np.isfinite(tangent))
    assert abs(_yield_value(model, projected, pc)) < 2.0e-5 * pc * pc


def test_mcc_lagged_volume_matches_committed_total_stress(taichi_material_cpu):
    model = _model()
    deformation = np.diag(np.exp([0.01, -0.02, -0.04]))
    _converge_lagged_hardening(model, deformation)
    stress = ti.Matrix.field(3, 3, ti.f64, shape=2)

    @ti.kernel
    def commit_and_compare(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        total = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            total[row, column] = values[row, column]
        stress[0] = model.total_first_piola_stress_at(0, total)
        ignored = model.commit_total_state(0, total)
        stress[1] = model.total_first_piola_stress_at(0, total)

    commit_and_compare(np.ascontiguousarray(deformation))
    assert model.lagged_plastic_volume_active[0] == 0
    actual_jp = 1.0 / np.linalg.det(model.plastic_deformation_inverse.to_numpy()[0])
    assert actual_jp < 1.0
    assert model.lagged_plastic_jacobian[0] == pytest.approx(actual_jp, rel=1.0e-8)
    np.testing.assert_allclose(stress.to_numpy()[0], stress.to_numpy()[1], rtol=2.0e-8, atol=0.01)
