"""Pressure-independent DP softening: current history and analytical shear path."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.DruckerPrager import DruckerPragerModel

pytestmark = [pytest.mark.cpu, pytest.mark.materials]


def make_model():
    model = DruckerPragerModel(stress_integration="SubStepping")
    model.model_initialize(
        {
            "Density": 2000.0,
            "YoungModulus": 1e6,
            "PoissonRatio": 0.33,
            "Friction": 0.0,
            "Dilation": 0.0,
            "ResidualFriction": 0.0,
            "ResidualDilation": 0.0,
            "Cohesion": 20000.0,
            "ResidualCohesion": 4000.0,
            "SoftType": "Linear",
            "dpType": "Inscribed",
            "PlasticDevStrain": 0.0,
            "ResidualPlasticDevStrain": 0.32,
        }
    )
    return model


def test_zero_friction_and_local_history_update(taichi_material_cpu):
    model = make_model()
    assert model.q_fai == 0.0
    assert model.k_fai == pytest.approx(20000.0)
    state = ti.Struct.field({"epdstrain": ti.f64}, shape=1)
    out = ti.Vector.field(10, ti.f64, shape=())

    @ti.kernel
    def evaluate():
        state[0].epdstrain = 0.05
        stress = ti.Vector([-60000.0, -80000.0, -70000.0, 10000.0, 0.0, 0.0])
        initial = model.GetMaterialParameter(stress, state[0])
        out[None] = model.UpdateMaterialParameter(stress, ti.Vector([0.15]), state[0], initial)

    evaluate()
    assert out[None][7] == pytest.approx(12500.0)
    assert out[None][3] == pytest.approx(12500.0)
    assert state[0].epdstrain == pytest.approx(0.05)


def test_simple_shear_matches_bounded_linear_softening_oracle(taichi_material_cpu):
    model = make_model()
    state = ti.Struct.field({"epdstrain": ti.f64}, shape=1)
    history = ti.Vector.field(2, ti.f64, shape=81)

    @ti.kernel
    def integrate():
        stress = ti.Vector.zero(ti.f64, 6)
        state[0].epdstrain = 0.0
        history[0] = ti.Vector([0.0, 0.0])
        ti.loop_config(serialize=True)
        for i in range(80):
            stress = model.core(
                0, stress, ti.Vector([0.0, 0.0, 0.0, 0.005, 0.0, 0.0]), ti.Vector.zero(ti.f64, 3), state
            )
            history[i + 1] = ti.Vector([stress[3], state[0].epdstrain])

    integrate()
    result = history.to_numpy()
    gamma = np.arange(81) * 0.01
    shear = 1e6 / (2 * 1.33)
    plastic = np.maximum(0, (shear * gamma - 20000) / (shear * np.sqrt(3) - 50000))
    tau = np.where(plastic < 0.32, np.minimum(shear * gamma, 20000 - 50000 * plastic), 4000.0)
    expected_strain = (gamma - tau / shear) / np.sqrt(3)
    np.testing.assert_allclose(result[:, 0], tau, atol=2.0, rtol=0)
    np.testing.assert_allclose(result[:, 1], expected_strain, atol=4e-6, rtol=0)
