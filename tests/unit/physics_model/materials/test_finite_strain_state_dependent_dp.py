"""State-law equivalence, physical tangent and accepted-history checks."""

import math

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.StateDependentDruckerPrager import (
    StateDependentDruckerPragerModel,
)

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def _model(**overrides):
    parameters = dict(
        YoungModulus=2e5,
        Density=1600.0,
        PoissonRatio=0.3,
        Cohesion=1200.0,
        e0=0.62,
        e_Tao=0.9,
        lambda_c=0.119,
        ksi=0.23,
        nd=1.7,
        nf=2.68,
        fai_c=30.0,
    )
    parameters.update(overrides)
    model = StateDependentDruckerPragerModel().initialize_from_kwargs(**parameters)
    model.allocate_state(1)
    return model


def test_state_law_matches_infinitesimal_sdmc_and_volume_ratio(taichi_material_cpu):
    model = _model()
    result = ti.Vector.field(6, ti.f64, shape=())

    @ti.kernel
    def evaluate(jacobian: ti.f64):
        result[None] = ti.Vector(model.state_parameters(0, ti.log(jacobian)))

    assert model.dp_type == "MiddleCircumscribed"
    assert model.is_nonassociated and not model.has_incremental_potential
    for void in [0.4, 0.62, 1.1]:
        model.void_ratio[0] = void
        for pressure in [1000.0, 150e3]:
            model.state_pressure[0] = pressure
            for jacobian in [0.5, 0.95, 1.0, 1.03, 2.0]:
                evaluate(jacobian)
                coefficients = result.to_numpy()[()].copy()
                e = np.clip((1 + void) * jacobian - 1, 0.1, 1.5)
                ec = model.e_Tao - model.lambda_c * (pressure / 101000) ** model.ksi
                state = e - ec
                phi = max(math.radians(30), math.atan(math.tan(math.radians(30)) * math.exp(-model.nf * state)))
                psi = max(0, math.atan(-model.nd * state))
                qphi, kphi = model._cone_parameters(model.cohesion, phi, model.dp_type)
                qpsi = model._cone_parameters(0, psi, model.dp_type)[0]
                np.testing.assert_allclose(
                    coefficients[:3],
                    [math.sqrt(2) * qphi / 3, math.sqrt(2) * qpsi / 3, math.sqrt(2) * kphi],
                    rtol=1e-12,
                )
                evaluate(jacobian * math.exp(2e-6))
                plus = result.to_numpy()[()][:3].copy()
                evaluate(jacobian * math.exp(-2e-6))
                numerical = (plus - result.to_numpy()[()][:3]) / 4e-6
                np.testing.assert_allclose(coefficients[3:], numerical, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dp_type", ["Circumscribed", "MiddleCircumscribed", "Inscribed"])
@pytest.mark.parametrize("e0", [0.62, 1.1])
def test_analytic_tangent_and_commit_are_consistent(taichi_material_cpu, dp_type, e0):
    model = _model(dpType=dp_type, e0=e0)
    total = ti.Matrix.field(3, 3, ti.f64, shape=1)
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())
    history = ti.Vector.field(14, ti.f64, shape=())
    region = ti.field(ti.i32, shape=())

    @ti.kernel
    def evaluate():
        stress[None] = model.total_first_piola_stress_at(0, total[0])
        tangent[None] = model.total_first_piola_tangent_at(0, total[0])
        elastic = model.trial_elastic_deformation(0, total[0])
        region[None] = model._principal_response(0, elastic)[9]

    @ti.kernel
    def save_history():
        history[None] = model.get_history_state(0)

    @ti.kernel
    def restore_history():
        model.set_history_state(0, history[None])

    @ti.kernel
    def commit():
        ignored = model.commit_total_state(0, total[0])

    total[0] = np.eye(3) * np.exp(-0.002)
    model.prepare_step_state(total, 1)
    np.testing.assert_allclose(model.committed_jacobian[0], np.exp(-0.006))
    assert model.state_pressure[0] > 1000
    save_history()
    initial = history.to_numpy()[()].copy()
    theta = 0.37
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    for branch, strain in [(0, [-0.002, -0.004, -0.007]), (1, [0.20, -0.12, -0.10]), (2, [0.01, 0.015, 0.02])]:
        restore_history()
        deformation = rotation @ np.diag(np.exp(strain))
        total[0] = deformation
        evaluate()
        assert region[None] == branch
        expected_stress = stress.to_numpy()[()].copy()
        analytic = tangent.to_numpy()[()].copy()
        numerical = np.zeros((9, 9))
        for j in range(9):
            plus, minus = deformation.copy(), deformation.copy()
            plus[j % 3, j // 3] += 2e-6
            minus[j % 3, j // 3] -= 2e-6
            total[0] = plus
            evaluate()
            first = stress.to_numpy()[()].copy()
            total[0] = minus
            evaluate()
            numerical[:, j] = (first - stress.to_numpy()[()]).T.reshape(-1) / 4e-6
        assert np.linalg.norm(numerical - analytic) / np.linalg.norm(analytic) < 5e-6
        # Newton and line-search evaluations must not accumulate evolution.
        np.testing.assert_array_equal(model.void_ratio[0], initial[11])
        np.testing.assert_array_equal(model.equivalent_plastic_strain[0], initial[0])
        total[0] = deformation
        commit()
        assert model.void_ratio[0] == pytest.approx(
            np.clip((1 + initial[11]) * np.linalg.det(deformation) / initial[12] - 1, 0.1, 1.5)
        )
        evaluate()
        assert np.linalg.norm(stress.to_numpy()[()] - expected_stress) / max(np.linalg.norm(expected_stress), 1) < 1e-9
        assert np.linalg.det(model.plastic_deformation_inverse.to_numpy()[0]) > 0


@pytest.mark.parametrize("overrides", [dict(e0=0), dict(nd=-1), dict(ksi=0), dict(nf=float("nan")), dict(fai_c=90)])
def test_invalid_parameters_fail_before_allocation(overrides):
    with pytest.raises((ValueError, KeyError)):
        _model(**overrides)


@pytest.mark.parametrize("key", ["dpType", "DPType", "yield_surface_type"])
def test_cone_mapping_aliases(key):
    model = StateDependentDruckerPragerModel().initialize_from_kwargs(YoungModulus=2e5, fai_c=30, **{key: "Inscribed"})
    assert model.dp_type == "Inscribed"
