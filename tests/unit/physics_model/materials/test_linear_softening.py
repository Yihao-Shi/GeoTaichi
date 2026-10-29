"""Endpoint-controlled bounded linear softening and consistent tangents."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.SoftenModel import LinearSoft
from src.physics_model.consititutive_model.infinitesimal_strain.DruckerPrager import DruckerPragerModel

pytestmark = [pytest.mark.cpu, pytest.mark.materials]


@ti.kernel
def sample(
    law: ti.template(),
    peak: ti.f64,
    residual: ti.f64,
    start: ti.f64,
    end: ti.f64,
    strain: ti.types.ndarray(),
    result: ti.types.ndarray(),
):
    for i in range(strain.shape[0]):
        result[i, 0] = law.soft(peak, residual, start, end, strain[i])
        result[i, 1] = law.soft_deriv(peak, residual, start, end, strain[i])


@ti.kernel
def sample_dp_plastic_modulus(
    model: ti.template(),
    state_vars: ti.template(),
    material_params: ti.template(),
    result: ti.template(),
):
    stress = ti.Vector([0.0, 0.0, 0.0, 10000.0, 0.0, 0.0])
    internal = ti.Vector([state_vars[0].epdstrain])
    flow = model.ComputeDgDsigma(1, stress, internal, material_params[None])
    result[None] = model.ComputePlasticModulus(
        1,
        flow,
        stress,
        internal,
        state_vars[0],
        material_params[None],
    )


def evaluate(law, strain, peak=20000.0, residual=4000.0, start=0.0, end=0.02):
    strain = np.asarray(strain, dtype=np.float64)
    result = np.empty((len(strain), 2))
    sample(law, peak, residual, start, end, strain, result)
    return result


def test_linear_softening_stops_at_residual(taichi_material_cpu):
    result = evaluate(LinearSoft(), [-0.01, 0, 0.01, 0.02, 0.025, 0.05, 0.099, 0.1, 0.2])
    np.testing.assert_allclose(result[:, 0], [20000, 20000, 12000, 4000, 4000, 4000, 4000, 4000, 4000])
    np.testing.assert_allclose(result[[0, 1, 3, 4, 5, 6, 7, 8], 1], 0)
    assert result[2, 1] == pytest.approx(-800000)


@pytest.mark.parametrize("end", [0.02, 0.1])
@pytest.mark.parametrize("peak,residual", [(20000.0, 4000.0), (4000.0, 20000.0), (4000.0, 4000.0)])
def test_tangent_matches_finite_difference_in_ramp_and_plateaus(taichi_material_cpu, end, peak, residual):
    law = LinearSoft()
    strains = np.array([-0.01, 0.007, 0.015, 0.03, 0.08, 0.12])
    h = 1e-7
    result = evaluate(law, strains, peak, residual, end=end)
    fd = (
        evaluate(law, strains + h, peak, residual, end=end)[:, 0]
        - evaluate(law, strains - h, peak, residual, end=end)[:, 0]
    ) / (2 * h)
    np.testing.assert_allclose(result[:, 1], fd, rtol=1e-9, atol=1e-5)
    assert np.all(result[:, 0] >= min(peak, residual))
    assert np.all(result[:, 0] <= max(peak, residual))


def test_endpoint_migration_preserves_previous_clipped_fivefold_curve(taichi_material_cpu):
    strains = np.linspace(-0.01, 0.12, 101)
    result = evaluate(LinearSoft(), strains)
    old_strength = 20000.0 - 16000.0 * np.clip(5.0 * strains / 0.1, 0, 1)
    old_tangent = np.where((strains > 0) & (strains < 0.02), -800000.0, 0.0)
    np.testing.assert_allclose(result[:, 0], old_strength, rtol=1e-14, atol=1e-11)
    np.testing.assert_allclose(result[:, 1], old_tangent)


def test_material_selection_does_not_read_obsolete_scale(taichi_material_cpu):
    model = DruckerPragerModel()
    model.choose_soft_function({"SoftType": "Linear", "SoftenParameter": 5})
    assert not hasattr(model.soft_function, "parameter")
    result = evaluate(model.soft_function, [0.01, 0.05], end=0.1)
    np.testing.assert_allclose(result[:, 0], [18400, 12000])


def test_nonzero_strain_origin_and_instantaneous_drop(taichi_material_cpu):
    result = evaluate(LinearSoft(), [0.02, 0.03, 0.04, 0.05], start=0.03, end=0.05)
    np.testing.assert_allclose(result[:, 0], [20000, 20000, 12000, 4000])
    result = evaluate(LinearSoft(), [0.02, 0.03, 0.04], start=0.03, end=0.03)
    np.testing.assert_allclose(result[:, 0], [20000, 20000, 4000])
    np.testing.assert_allclose(result[:, 1], 0)


def test_drucker_prager_softening_modulus_has_consistency_sign_and_strain_scale(taichi_material_cpu):
    model = DruckerPragerModel()
    model.model_initialize(
        {
            "SoftType": "Linear",
            "Density": 2000.0,
            "YoungModulus": 1.0e6,
            "PoissonRatio": 0.33,
            "Cohesion": 20000.0,
            "Friction": 1.0,
            "ResidualCohesion": 4000.0,
            "ResidualFriction": 1.0,
            "PlasticDevStrain": 0.0,
            "ResidualPlasticDevStrain": 0.04,
        }
    )
    state_vars = ti.Struct.field(dict(epdstrain=ti.f64), shape=1)
    state_vars[0].epdstrain = 0.02
    material_params = ti.Vector.field(10, ti.f64, shape=())
    material_params[None] = [
        model.bulk,
        model.shear,
        model.q_fai,
        model.k_fai,
        model.q_psi,
        model.tensile,
        model.fai_peak,
        12000.0,
        model.psi_peak,
        0.0,
    ]
    result = ti.field(ti.f64, shape=())

    sample_dp_plastic_modulus(model, state_vars, material_params, result)

    dk_dc = 6.0 * np.cos(np.deg2rad(1.0)) / (np.sqrt(3.0) * (3.0 + np.sin(np.deg2rad(1.0))))
    expected = dk_dc * 400000.0 * np.sqrt(1.0 / 3.0)
    assert result[None] == pytest.approx(expected)
