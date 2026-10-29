"""Signed yield distances and history-chain derivatives of small-strain plasticity."""

import numpy as np
import pytest
import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.consititutive_model.infinitesimal_strain.ElasticPerfectlyPlastic import (
    ElasticPerfectlyPlasticModel,
)
from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import (
    DeqepsilonqDepsilon,
    EquivalentDeviatoricStrain,
)
from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel
from src.physics_model.consititutive_model.infinitesimal_strain.ModifiedCamClay import ModifiedCamClayModel
from src.utils.VectorFunction import voigt_tensor_dot

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def soft_model(kind, integration="SubStepping"):
    cls = MohrCoulombModel if kind == "mc" else ElasticPerfectlyPlasticModel
    model = cls(stress_integration=integration)
    model.model_initialize(
        dict(
            Density=2000.0,
            YoungModulus=1e6,
            PoissonRatio=0.33,
            Cohesion=20000.0,
            ResidualCohesion=4000.0,
            Friction=30.0,
            ResidualFriction=15.0,
            Dilation=0.0,
            YieldStress=20000.0,
            ResidualYieldStress=4000.0,
            SoftType="Linear",
            PlasticDevStrain=0.0,
            ResidualPlasticDevStrain=0.5,
        )
    )
    return model


def test_tensor_history_gradient_matches_directional_difference(taichi_material_cpu):
    out = ti.field(ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        strain = ti.Vector([0.07, -0.02, -0.05, 0.04, 0.03, -0.02])
        direction = ti.Vector([0.1, -0.05, -0.05, 0.2, 0.1, -0.03])
        out[0] = voigt_tensor_dot(DeqepsilonqDepsilon(strain), direction)
        out[1] = (
            EquivalentDeviatoricStrain(strain + 1e-6 * direction)
            - EquivalentDeviatoricStrain(strain - 1e-6 * direction)
        ) / 2e-6

    evaluate()
    assert out[0] == pytest.approx(out[1], rel=1e-8)


@pytest.mark.parametrize("kind", ["mc", "mises"])
def test_soft_strength_uses_local_history_without_committing(taichi_material_cpu, kind):
    model = soft_model(kind)
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)
    out = ti.field(ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        state[0].strain = ti.Vector([0.0, 0.0, 0.0, 0.05 * ti.sqrt(1.5), 0.0, 0.0])
        value = ti.Vector([-60000.0, -80000.0, -70000.0, 10000.0, 0.0, 0.0])
        params = model.GetMaterialParameter(value, state[0])
        local = ti.Vector([0.0, 0.0, 0.0, 0.15 * ti.sqrt(1.5), 0.0, 0.0])
        params = model.UpdateMaterialParameter(value, local, state[0], params)
        if ti.static(kind == "mc"):
            out[0] = params[4]
        else:
            out[0] = params[2]
        out[1] = EquivalentDeviatoricStrain(state[0].strain)

    evaluate()
    assert out[0] == pytest.approx(15200.0)
    assert out[1] == pytest.approx(0.05)


@pytest.mark.parametrize("kind", ["mc", "mises"])
@pytest.mark.parametrize("random_field", [False, True])
def test_tensor_softening_modulus_matches_history_direction(taichi_material_cpu, monkeypatch, kind, random_field):
    monkeypatch.setattr(GlobalVariable, "RANDOMFIELD", random_field)
    model = soft_model(kind)
    model.residual_frac = 0.3
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)
    out = ti.field(ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        if ti.static(random_field):
            state[0].bulk, state[0].shear = model.bulk, model.shear
            if ti.static(kind == "mc"):
                state[0].c_peak, state[0].fai_peak = 25000.0, 0.4
                state[0].psi_peak, state[0].tensile = 0.0, 1000.0
            else:
                state[0]._yield_peak = 25000.0
        state[0].strain = ti.Vector([0.07, -0.02, -0.05, 0.04, 0.03, -0.02])
        stress = ti.Vector([-60000.0, -80000.0, -70000.0, 10000.0, 1000.0, -3000.0])
        flow = ti.Vector([0.1, -0.05, -0.05, 0.2, 0.1, -0.03])
        params = model.GetMaterialParameter(stress, state[0])
        out[0] = model.ComputePlasticModulus(1, flow, stress, model.GetInternalVariables(state[0]), state[0], params)
        plus, minus = state[0], state[0]
        plus.strain += 1e-6 * flow
        minus.strain -= 1e-6 * flow
        fp, fm = 0.0, 0.0
        if ti.static(kind == "mc"):
            fp, _ = model.ComputeYieldFunction(
                stress, model.GetInternalVariables(plus), model.GetMaterialParameter(stress, plus)
            )
            fm, _ = model.ComputeYieldFunction(
                stress, model.GetInternalVariables(minus), model.GetMaterialParameter(stress, minus)
            )
        else:
            fp = model.ComputeYieldFunction(
                stress, model.GetInternalVariables(plus), model.GetMaterialParameter(stress, plus)
            )
            fm = model.ComputeYieldFunction(
                stress, model.GetInternalVariables(minus), model.GetMaterialParameter(stress, minus)
            )
        out[1] = (fp - fm) / 2e-6

    evaluate()
    assert abs(out[1]) > 1.0
    assert out[0] == pytest.approx(out[1], rel=1e-7)


def test_mc_elastic_interior_retains_negative_distance(taichi_material_cpu):
    model = soft_model("mc")
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)
    out = ti.field(ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        stress = ti.Vector([-60000.0, -80000.0, -70000.0, 1000.0, 0.0, 0.0])
        params = model.GetMaterialParameter(stress, state[0])
        internal = model.GetInternalVariables(state[0])
        branch, out[0] = model.ComputeYieldState(stress, internal, params)
        fs, ft = model.ComputeYieldFunction(stress, internal, params)
        out[1] = ti.max(fs, ft)
        assert branch == 0

    evaluate()
    assert out[1] < 0
    assert out[0] == pytest.approx(out[1])


@pytest.mark.parametrize("bonding,subloading", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("r0", [0.7, 1.0])
def test_mcc_modulus_matches_local_history_rate(taichi_material_cpu, bonding, subloading, r0):
    model = ModifiedCamClayModel()
    model.model_initialize(
        dict(
            StressRatio=1.2,
            **{"lambda": 0.1, "kappa": 0.03},
            void_ratio_ref=0.7,
            pc=200000.0,
            bonding=bonding,
            subloading=subloading,
            mc_a=50000.0,
            mc_b=1.2,
            mc_c=10000.0,
            mc_d=1.5,
            s_h=0.8,
            m_degradation=0.3,
            subloading_u=0.5,
        )
    )
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)
    out = ti.field(ti.f64, shape=3)

    @ti.kernel
    def evaluate():
        state[0].pc, state[0].void_ratio = 200000.0, 0.7
        if ti.static(bonding):
            state[0].chi, state[0].pcd, state[0].pcc = model.ComputeBondingParameters(0.8)
        if ti.static(subloading):
            state[0].subloading_r = r0
        stress = ti.Vector([-60000.0, -100000.0, -80000.0, 12000.0, 1000.0, -3000.0])
        params = model.GetMaterialParameter(stress, state[0])
        internal = model.GetInternalVariables(state[0])
        flow = model.ComputeDgDsigma(1, stress, internal, params)
        out[2] = model.ComputeInternalVariables(0.0, flow, internal, params).norm()
        out[0] = model.ComputePlasticModulus(1, flow, stress, internal, state[0], params)
        h = 1e-11
        shifted = internal + model.ComputeInternalVariables(h, flow, internal, params)
        half_shifted = internal + model.ComputeInternalVariables(0.5 * h, flow, internal, params)
        f0 = model.ComputeYieldFunction(stress, internal, params)
        f1 = model.ComputeYieldFunction(stress, shifted, params)
        fh = model.ComputeYieldFunction(stress, half_shifted, params)
        # Forward differences avoid negative accumulated plastic strain; Richardson cancels O(h).
        out[1] = (4 * fh - f1 - 3 * f0) / h

    evaluate()
    assert out[2] == pytest.approx(0.0, abs=1e-10)
    assert out[0] == pytest.approx(out[1], rel=2e-6)


@pytest.mark.parametrize("kind", ["mc", "mises"])
@pytest.mark.parametrize("integration", ["ReturnMapping", "SubStepping"])
def test_tensor_models_match_softening_shear_curve(taichi_material_cpu, kind, integration):
    model = soft_model(kind, integration)
    if kind == "mc":
        model.model_initialize(dict(model.material, Friction=0.0, ResidualFriction=0.0))
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)
    result = ti.Vector.field(2, ti.f64, shape=71)

    @ti.kernel
    def integrate():
        stress = ti.Vector([-80000.0, -80000.0, -80000.0, 0.0, 0.0, 0.0])
        ti.loop_config(serialize=True)
        for i in range(70):
            stress = model.core(0, stress, ti.Vector([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]), ti.Vector.zero(ti.f64, 3), state)
            result[i + 1] = ti.Vector([stress[3], EquivalentDeviatoricStrain(state[0].strain)])

    integrate()
    gamma = np.arange(71) * 0.02
    shear = 1e6 / (2 * 1.33)
    stress_factor = 1.0 if kind == "mc" else np.sqrt(3)
    kappa = np.maximum(0.0, (stress_factor * shear * gamma - 20000.0) / (stress_factor * shear * np.sqrt(6) - 32000.0))
    tau = np.where(
        kappa < 0.5, np.minimum(shear * gamma, (20000.0 - 32000.0 * kappa) / stress_factor), 4000.0 / stress_factor
    )
    expected_history = (gamma - tau / shear) / np.sqrt(6)
    np.testing.assert_allclose(result.to_numpy()[:, 0], tau, atol=2.0, rtol=0)
    np.testing.assert_allclose(result.to_numpy()[:, 1], expected_history, atol=4e-6, rtol=0)
