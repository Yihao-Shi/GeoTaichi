"""History derivatives for tensile flow and particle-specific DP softening."""

import pytest
import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.consititutive_model.infinitesimal_strain.DruckerPrager import DruckerPragerModel
from src.physics_model.consititutive_model.infinitesimal_strain.ElasticPerfectlyPlastic import (
    ElasticPerfectlyPlasticModel,
)
from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel
from src.utils.VectorFunction import voigt_tensor_dot

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def make_model(kind):
    cls = DruckerPragerModel if kind == "dp" else MohrCoulombModel
    model = cls(stress_integration="SubStepping")
    model.model_initialize(
        dict(
            YoungModulus=2e5,
            Cohesion=100.0,
            ResidualCohesion=50.0,
            Friction=30.0,
            ResidualFriction=30.0,
            Dilation=0.0,
            ResidualDilation=0.0,
            Tensile=1000.0,
            SoftType="Linear",
            PlasticDevStrain=0.0,
            ResidualPlasticDevStrain=0.5,
        )
    )
    return model


def test_dp_history_rate_matches_tensile_flow_norm(taichi_material_cpu):
    model = make_model("dp")
    out = ti.field(ti.f64, shape=2)
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)

    @ti.kernel
    def evaluate():
        stress = ti.Vector([150.0, 130.0, 110.0, 10.0, 3.0, -5.0])
        params = model.GetMaterialParameter(stress, state[0])
        internal = model.GetInternalVariables(state[0])
        flow = model.ComputeDgDsigma(2, stress, internal, params)
        out[0] = model.ComputeInternalVariables(1.0, flow, internal, params)[0]
        out[1] = ti.sqrt(2.0 / 3.0 * voigt_tensor_dot(flow, flow))

    evaluate()
    assert out[0] == pytest.approx(out[1], rel=1e-12)


@pytest.mark.parametrize("kind", ["dp", "mc"])
def test_softened_tensile_cap_modulus_matches_history_direction(taichi_material_cpu, kind):
    model = make_model(kind)
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)
    out = ti.field(ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        if ti.static(kind == "dp"):
            state[0].epdstrain = 0.2
        else:
            state[0].strain = ti.Vector([0.0, 0.0, 0.0, 0.2 * ti.sqrt(1.5), 0.0, 0.0])
        stress = ti.Vector([150.0, 130.0, 110.0, 10.0, 3.0, -5.0])
        params = model.GetMaterialParameter(stress, state[0])
        internal = model.GetInternalVariables(state[0])
        flow = model.ComputeDgDsigma(2, stress, internal, params)
        out[0] = model.ComputePlasticModulus(2, flow, stress, internal, state[0], params)
        plus, minus = state[0], state[0]
        if ti.static(kind == "dp"):
            rate = model.ComputeInternalVariables(1.0, flow, internal, params)[0]
            plus.epdstrain += 1e-6 * rate
            minus.epdstrain -= 1e-6 * rate
        else:
            plus.strain += 1e-6 * flow
            minus.strain -= 1e-6 * flow
        _, fp = model.ComputeYieldFunction(
            stress, model.GetInternalVariables(plus), model.GetMaterialParameter(stress, plus)
        )
        _, fm = model.ComputeYieldFunction(
            stress, model.GetInternalVariables(minus), model.GetMaterialParameter(stress, minus)
        )
        out[1] = (fp - fm) / 2e-6

    evaluate()
    assert abs(out[1]) > 1.0
    assert out[0] == pytest.approx(out[1], rel=1e-7)


def test_dp_random_field_modulus_uses_particle_strengths(taichi_material_cpu, monkeypatch):
    monkeypatch.setattr(GlobalVariable, "RANDOMFIELD", True)
    model = make_model("dp")
    model.residual_frac = 0.3
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)
    out = ti.field(ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        state[0].epdstrain = 0.2
        state[0].bulk, state[0].shear = model.bulk, model.shear
        state[0].fai_peak, state[0].psi_peak, state[0].c_peak, state[0].tensile = 0.4, 0.0, 250.0, 0.0
        stress = ti.Vector([-1200.0, -950.0, -850.0, 310.0, -70.0, 80.0])
        params = model.GetMaterialParameter(stress, state[0])
        internal = model.GetInternalVariables(state[0])
        flow = model.ComputeDgDsigma(1, stress, internal, params)
        out[0] = model.ComputePlasticModulus(1, flow, stress, internal, state[0], params)
        rate = model.ComputeInternalVariables(1.0, flow, internal, params)[0]
        plus, minus = state[0], state[0]
        plus.epdstrain += 1e-6 * rate
        minus.epdstrain -= 1e-6 * rate
        fp, _ = model.ComputeYieldFunction(stress, internal, model.GetMaterialParameter(stress, plus))
        fm, _ = model.ComputeYieldFunction(stress, internal, model.GetMaterialParameter(stress, minus))
        out[1] = (fp - fm) / 2e-6

    evaluate()
    assert abs(out[1]) > 1.0
    assert out[0] == pytest.approx(out[1], rel=1e-7)


def test_mises_random_field_initialization_preserves_softening_endpoints(taichi_material_cpu, monkeypatch, tmp_path):
    monkeypatch.setattr(GlobalVariable, "RANDOMFIELD", True)
    path = tmp_path / "material.txt"
    path.write_text("2000 1000000 .33 25000\n2000 1000000 .33 22000\n")
    model = ElasticPerfectlyPlasticModel()
    model.random_field_initialize(
        dict(
            MaterialFile=str(path),
            ResidualYieldStress=4000.0,
            ResidualFraction=0.3,
            SoftType="Linear",
            PlasticDevStrain=0.1,
            ResidualPlasticDevStrain=0.6,
        )
    )
    state = ti.types.struct(**model.get_state_vars()).field(shape=2)
    model.read_random_field(0, 2, state)
    out = ti.field(ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        for i in range(2):
            state[i].strain = ti.Vector([0.0, 0.0, 0.0, 0.2 * ti.sqrt(1.5), 0.0, 0.0])
            params = model.GetMaterialParameter(ti.Vector.zero(ti.f64, 6), state[i])
            out[i] = params[2]

    evaluate()
    # At kappa .2, weight=(.2-.1)/(.6-.1)=.2, using each particle's own peak.
    assert out[0] == pytest.approx(21500.0)
    assert out[1] == pytest.approx(18920.0)
