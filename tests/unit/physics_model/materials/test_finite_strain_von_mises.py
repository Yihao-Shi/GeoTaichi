"""Finite-strain von Mises return-mapping tests."""

import math

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.VonMises import (
    FiniteStrainVonMisesModel,
)
from src.physics_model.consititutive_model.finite_strain.DruckerPrager import (
    FiniteStrainDruckerPragerModel,
)
from src.physics_model.consititutive_model.finite_strain.HenckyPlasticity import (
    HenckyAssociatedPlasticityModel,
)

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def test_von_mises_and_dp_are_parallel_models():
    assert issubclass(FiniteStrainVonMisesModel, HenckyAssociatedPlasticityModel)
    assert issubclass(FiniteStrainDruckerPragerModel, HenckyAssociatedPlasticityModel)
    assert not issubclass(FiniteStrainVonMisesModel, FiniteStrainDruckerPragerModel)


def _model(hardening=0.0):
    model = FiniteStrainVonMisesModel().initialize_from_kwargs(
        density=7800.0,
        young_modulus=2.0e5,
        poisson_ratio=0.3,
        YieldStress=1200.0,
        HardeningModulus=hardening,
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


def test_von_mises_uses_standard_equivalent_yield_stress(taichi_material_cpu):
    model = _model()
    assert model.reference_yield_intercept == pytest.approx(math.sqrt(2.0 / 3.0) * 1200.0)


def test_von_mises_return_and_analytic_tangent(taichi_material_cpu):
    model = _model(hardening=5000.0)
    evaluate = _evaluator(model)
    deformation = np.array(
        [
            [1.14, 0.05, -0.02],
            [0.01, 0.95, 0.03],
            [0.02, -0.01, 0.91],
        ],
        dtype=np.float64,
    )
    energy, pk1, tangent, projected = evaluate(deformation)
    assert energy > 0.0
    assert np.linalg.det(projected) > 0.0
    np.testing.assert_allclose(tangent, tangent.T, rtol=2.0e-10, atol=2.0e-8)

    step = 2.0e-7
    numerical = np.zeros((9, 9), dtype=np.float64)
    for column in range(9):
        component = column // 3
        row = column % 3
        plus = deformation.copy()
        minus = deformation.copy()
        plus[row, component] += step
        minus[row, component] -= step
        numerical[:, column] = (
            _flatten_column_major(evaluate(plus)[1]) - _flatten_column_major(evaluate(minus)[1])
        ) / (2.0 * step)
    np.testing.assert_allclose(tangent, numerical, rtol=4.0e-4, atol=3.0e-2)
    assert np.linalg.norm(pk1) > 0.0


def test_von_mises_commit_updates_hardening_state(taichi_material_cpu):
    model = _model(hardening=5000.0)
    committed = ti.Matrix.field(3, 3, ti.f64, shape=())

    @ti.kernel
    def commit():
        trial = ti.Matrix([[1.20, 0.0, 0.0], [0.0, 0.90, 0.0], [0.0, 0.0, 0.92]])
        committed[None] = model.commit_state(0, trial)

    commit()
    equivalent_plastic_strain = float(model.equivalent_plastic_strain[0])
    assert equivalent_plastic_strain > 0.0
    assert np.linalg.det(committed.to_numpy()[()]) > 0.0
    hardened_intercept = math.sqrt(2.0 / 3.0) * (
        model.yield_stress_equivalent + model.hardening_modulus * equivalent_plastic_strain
    )
    assert hardened_intercept > model.reference_yield_intercept


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: _model(hardening=5000.0),
        lambda: FiniteStrainDruckerPragerModel().initialize_from_kwargs(
            density=1800.0,
            young_modulus=2.0e5,
            poisson_ratio=0.3,
            Cohesion=1200.0,
            FrictionAngle=30.0,
            DilationAngle=30.0,
        ),
    ],
    ids=["von_mises", "drucker_prager"],
)
def test_accepted_plastic_state_vjp_crosses_two_steps(taichi_material_cpu, model_factory):
    model = model_factory()
    if model.plastic_deformation_inverse is None:
        model.allocate_state(1)

    increments = ti.Matrix.field(3, 3, ti.f64, shape=2)
    total_before = ti.Matrix.field(3, 3, ti.f64, shape=2)
    trial_total = ti.Matrix.field(3, 3, ti.f64, shape=2)
    history_before = ti.Vector.field(11, ti.f64, shape=2)
    increment_vjp = ti.Matrix.field(3, 3, ti.f64, shape=2)
    initial_total = ti.Matrix.field(3, 3, ti.f64, shape=())
    initial_total_vjp = ti.Matrix.field(3, 3, ti.f64, shape=())
    initial_plastic_vjp = ti.Matrix.field(3, 3, ti.f64, shape=())
    initial_equivalent_vjp = ti.field(ti.f64, shape=())
    initial_volumetric_vjp = ti.field(ti.f64, shape=())
    loss = ti.field(ti.f64, shape=())

    total_seed = ti.Matrix([[0.13, -0.07, 0.04], [0.02, 0.09, -0.05], [-0.03, 0.06, 0.11]])
    plastic_seed = ti.Matrix([[-0.08, 0.03, 0.05], [0.07, 0.12, -0.04], [0.01, -0.06, 0.10]])
    equivalent_seed = 0.17
    volumetric_seed = -0.09

    @ti.kernel
    def reset_state(
        total: ti.types.ndarray(dtype=ti.f64, ndim=2),
        plastic_inverse: ti.types.ndarray(dtype=ti.f64, ndim=2),
        equivalent: ti.f64,
        volumetric: ti.f64,
    ):
        for row, column in ti.static(ti.ndrange(3, 3)):
            initial_total[None][row, column] = total[row, column]
            model.plastic_deformation_inverse[0][row, column] = plastic_inverse[row, column]
        model.equivalent_plastic_strain[0] = equivalent
        model.volumetric_plastic_strain[0] = volumetric

    @ti.kernel
    def forward():
        current_total = initial_total[None]
        for step in ti.static(range(2)):
            total_before[step] = current_total
            history_before[step] = model.get_history_state(0)
            trial_total[step] = increments[step] @ current_total
            current_total = model.commit_total_state(0, trial_total[step])
        value = 0.0
        final_plastic_inverse = model.plastic_deformation_inverse[0]
        for row, column in ti.static(ti.ndrange(3, 3)):
            value += total_seed[row, column] * current_total[row, column]
            value += plastic_seed[row, column] * final_plastic_inverse[row, column]
        value += equivalent_seed * model.equivalent_plastic_strain[0]
        value += volumetric_seed * model.volumetric_plastic_strain[0]
        loss[None] = value

    @ti.kernel
    def reverse():
        total_vjp = total_seed
        plastic_vjp = plastic_seed
        equivalent_vjp = equivalent_seed
        volumetric_vjp = volumetric_seed
        for reverse_index in ti.static(range(2)):
            step = 1 - reverse_index
            model.set_history_state(0, history_before[step])
            (
                trial_vjp,
                plastic_vjp,
                equivalent_vjp,
                volumetric_vjp,
            ) = model.commit_total_state_vjp(
                0,
                trial_total[step],
                total_vjp,
                plastic_vjp,
                equivalent_vjp,
                volumetric_vjp,
            )
            increment_vjp[step] = trial_vjp @ total_before[step].transpose()
            total_vjp = increments[step].transpose() @ trial_vjp
        initial_total_vjp[None] = total_vjp
        initial_plastic_vjp[None] = plastic_vjp
        initial_equivalent_vjp[None] = equivalent_vjp
        initial_volumetric_vjp[None] = volumetric_vjp

    base_increments = np.array(
        [
            [[1.16, 0.04, -0.02], [0.01, 0.93, 0.03], [0.02, -0.01, 0.90]],
            [[1.05, -0.03, 0.01], [0.02, 0.97, 0.04], [-0.01, 0.02, 0.96]],
        ],
        dtype=np.float64,
    )
    base_total = np.array(
        [[1.01, 0.01, 0.0], [0.0, 0.99, -0.01], [0.0, 0.01, 1.0]],
        dtype=np.float64,
    )
    base_plastic = np.array(
        [[0.99, 0.01, 0.0], [0.0, 1.01, -0.01], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    base_equivalent = 0.013
    base_volumetric = -0.007

    def evaluate(current_increments, current_total, current_plastic, equivalent, volumetric):
        increments.from_numpy(np.ascontiguousarray(current_increments))
        reset_state(
            np.ascontiguousarray(current_total),
            np.ascontiguousarray(current_plastic),
            equivalent,
            volumetric,
        )
        forward()
        return float(loss[None])

    evaluate(
        base_increments,
        base_total,
        base_plastic,
        base_equivalent,
        base_volumetric,
    )
    reverse()

    step = 2.0e-7
    increment_direction = np.linspace(-0.4, 0.5, 18).reshape(2, 3, 3)
    total_direction = np.linspace(0.3, -0.2, 9).reshape(3, 3)
    plastic_direction = np.linspace(-0.25, 0.35, 9).reshape(3, 3)
    equivalent_direction = 0.23
    volumetric_direction = -0.17
    analytic_directional_derivative = (
        np.sum(increment_vjp.to_numpy() * increment_direction)
        + np.sum(initial_total_vjp.to_numpy()[()] * total_direction)
        + np.sum(initial_plastic_vjp.to_numpy()[()] * plastic_direction)
        + float(initial_equivalent_vjp[None]) * equivalent_direction
        + float(initial_volumetric_vjp[None]) * volumetric_direction
    )
    plus_loss = evaluate(
        base_increments + step * increment_direction,
        base_total + step * total_direction,
        base_plastic + step * plastic_direction,
        base_equivalent + step * equivalent_direction,
        base_volumetric + step * volumetric_direction,
    )
    minus_loss = evaluate(
        base_increments - step * increment_direction,
        base_total - step * total_direction,
        base_plastic - step * plastic_direction,
        base_equivalent - step * equivalent_direction,
        base_volumetric - step * volumetric_direction,
    )
    numerical_directional_derivative = (plus_loss - minus_loss) / (2.0 * step)
    assert analytic_directional_derivative == pytest.approx(numerical_directional_derivative, rel=4.0e-5, abs=2.0e-7)
