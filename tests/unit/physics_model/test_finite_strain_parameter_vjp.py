"""Finite-difference checks for device DP/VM material-parameter VJPs."""

import math

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.DruckerPrager import (
    FiniteStrainDruckerPragerModel,
)
from src.physics_model.consititutive_model.finite_strain.VonMises import (
    FiniteStrainVonMisesModel,
)

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def _model(kind, values):
    if kind == "dp":
        model = FiniteStrainDruckerPragerModel().initialize_from_kwargs(
            density=1800.0,
            young_modulus=values[0],
            poisson_ratio=values[1],
            Cohesion=values[2],
            FrictionAngle=values[3],
            DilationAngle=values[3],
        )
    else:
        model = FiniteStrainVonMisesModel().initialize_from_kwargs(
            density=7800.0,
            young_modulus=values[0],
            poisson_ratio=values[1],
            YieldStress=values[2],
            HardeningModulus=values[3],
        )
    model.allocate_state(1)
    return model


def _evaluate(model, deformation, inverse_seed, equivalent_seed, volumetric_seed):
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    parameter_derivative = ti.Matrix.field(3, 3, ti.f64, shape=4)
    state_vjp = ti.Vector.field(4, ti.f64, shape=())
    objective = ti.field(ti.f64, shape=())

    @ti.kernel
    def evaluate_stress():
        F = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            F[row, column] = deformation[row, column]
        stress[None] = model.first_piola_stress_at(0, F)
        all_derivatives = model.first_piola_parameter_derivatives_at(0, F)
        for parameter in ti.static(range(4)):
            for column, row in ti.static(ti.ndrange(3, 3)):
                parameter_derivative[parameter][row, column] = all_derivatives[3 * column + row, parameter]

    @ti.kernel
    def evaluate_commit():
        F = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            F[row, column] = deformation[row, column]
        state_vjp[None] = model.commit_total_state_parameter_vjp(0, F, inverse_seed, equivalent_seed, volumetric_seed)

    evaluate_stress()
    evaluate_commit()

    # Commit is intentionally a separate kernel: the accepted-state VJP is
    # independent of the equilibrium stress kernel and should not multiply
    # CUDA specialization size.
    @ti.kernel
    def commit_objective():
        F = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            F[row, column] = deformation[row, column]
        model.commit_total_state(0, F)
        value = 0.0
        for row, column in ti.static(ti.ndrange(3, 3)):
            value += inverse_seed[row, column] * model.plastic_deformation_inverse[0][row, column]
        objective[None] = (
            value
            + equivalent_seed * model.equivalent_plastic_strain[0]
            + volumetric_seed * model.volumetric_plastic_strain[0]
        )

    commit_objective()
    return (
        stress.to_numpy()[()],
        parameter_derivative.to_numpy(),
        state_vjp.to_numpy()[()],
        float(objective[None]),
    )


@pytest.mark.parametrize("kind", ["dp", "vm"])
def test_device_material_parameter_vjp_matches_fd(taichi_runtime, kind):
    values = np.array([2.0e4, 0.3, 120.0, 28.0 if kind == "dp" else 900.0])
    deformation = np.array(
        [[1.24, 0.03, 0.0], [0.0, 0.86, 0.02], [0.0, 0.0, 0.82]],
        dtype=np.float64,
    )
    inverse_seed = ti.Matrix([[0.3, -0.2, 0.1], [0.0, 0.4, -0.1], [0.2, 0.0, -0.25]])
    equivalent_seed = 0.7
    volumetric_seed = -0.4

    baseline = _evaluate(_model(kind, values), deformation, inverse_seed, equivalent_seed, volumetric_seed)
    analytic_stress = baseline[0]
    analytic_parameter_derivative = baseline[1]
    analytic_state = baseline[2]
    eps = np.array([1.0e-3, 1.0e-6, 1.0e-3, 1.0e-3])
    fd_stress = np.zeros((4, 3, 3))
    fd_state = np.zeros(4)
    for parameter in range(4):
        plus = values.copy()
        minus = values.copy()
        plus[parameter] += eps[parameter]
        minus[parameter] -= eps[parameter]
        stress_plus, _, _, _ = _evaluate(
            _model(kind, plus), deformation, inverse_seed, equivalent_seed, volumetric_seed
        )
        stress_minus, _, _, _ = _evaluate(
            _model(kind, minus), deformation, inverse_seed, equivalent_seed, volumetric_seed
        )
        fd_stress[parameter] = (stress_plus - stress_minus) / (2.0 * eps[parameter])

        model_plus = _model(kind, plus)
        model_minus = _model(kind, minus)
        _, _, _, objective_plus = _evaluate(model_plus, deformation, inverse_seed, equivalent_seed, volumetric_seed)
        _, _, _, objective_minus = _evaluate(model_minus, deformation, inverse_seed, equivalent_seed, volumetric_seed)
        fd_state[parameter] = (objective_plus - objective_minus) / (2.0 * eps[parameter])

    np.testing.assert_allclose(analytic_parameter_derivative, fd_stress, rtol=2.0e-3, atol=2.0e-4)
    np.testing.assert_allclose(analytic_state, fd_state, rtol=4.0e-3, atol=2.0e-4)
