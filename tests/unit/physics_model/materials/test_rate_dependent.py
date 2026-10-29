"""mu(I) scalar laws and coupled explicit constitutive updates."""

import math

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.DruckerPrager import DruckerPragerModel
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.physics_model.consititutive_model.infinitesimal_strain.ElasticPerfectlyPlastic import (
    ElasticPerfectlyPlasticModel,
)
from src.physics_model.consititutive_model.infinitesimal_strain.GranularMaterial import GranularMaterial
from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel
from src.physics_model.consititutive_model.infinitesimal_strain.ModifiedCamClay import ModifiedCamClayModel
from src.physics_model.consititutive_model.infinitesimal_strain.NorSandModel import NorSandModel
from src.physics_model.consititutive_model.infinitesimal_strain.RateDependent import RateDependent

pytestmark = [pytest.mark.materials, pytest.mark.cpu]

PARAMETERS = dict(
    Density=1800.0,
    GrainDensity=2650.0,
    YoungModulus=2.0e5,
    PoissonRatio=0.3,
    StaticFriction=30.0,
    DynamicFriction=45.0,
    AverageDiameter=0.002,
    InertialNumber=0.03,
    eps=0.0,
    Cohesion=100.0,
    Dilation=0.0,
    RateDependent=True,
)


def _mu(pressure, rate, parameters=PARAMETERS):
    static = math.tan(math.radians(parameters["StaticFriction"]))
    dynamic = math.tan(math.radians(parameters.get("DynamicFriction", parameters["StaticFriction"])))
    if pressure <= 0.0:
        return static
    c = parameters["InertialNumber"] * math.sqrt(
        pressure / (parameters["GrainDensity"] * parameters["AverageDiameter"] ** 2)
    )
    return static + abs(rate) * (dynamic - static) / (c + math.hypot(rate, parameters["eps"]))


def _model(model_class, integration="ReturnMapping", **parameters):
    model = model_class(solver_type="Explicit", stress_integration=integration)
    material = dict(PARAMETERS)
    if model_class is ModifiedCamClayModel:
        material.update(StressRatio=1.2, **{"lambda": 0.1, "kappa": 0.03}, void_ratio_ref=0.7, pc=2000.0)
    material.update(parameters)
    model.model_initialize(material)
    return model


def _update(model, *, dt_value=0.01, shear_rate=10.0, mean=-1000.0, dimension=2):
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)
    particle = ti.types.struct(stress=ti.types.vector(6, ti.f64)).field(shape=1)
    dt = ti.field(ti.f64, shape=())
    stress = ti.Vector.field(6, ti.f64, shape=())
    elastic = ti.field(ti.f64, shape=2)
    dt[None] = dt_value
    particle.stress[0] = [mean, mean, mean, 0.0, 0.0, 0.0]
    granular = isinstance(model, GranularMaterial)

    @ti.kernel
    def update():
        model._initialize_vars(0, particle, state)
        if ti.static(granular):
            elastic[0], elastic[1] = model.bulk, model.shear
        else:
            parameters = model.GetMaterialParameter(particle[0].stress, state[0])
            elastic[0], elastic[1] = model.ComputeElasticModulus(particle[0].stress, parameters)
        gradient = ti.Matrix.zero(ti.f64, dimension, dimension)
        gradient[0, 1], gradient[1, 0] = 0.5 * shear_rate, 0.5 * shear_rate
        if ti.static(dimension == 2):
            stress[None] = model.ComputeStress2D(0, particle[0].stress, gradient, state, dt)
        else:
            stress[None] = model.ComputeStress(0, particle[0].stress, gradient, state, dt)

    update()
    return stress.to_numpy(), state, elastic.to_numpy()


@pytest.mark.parametrize("integration", ["ReturnMapping", "ImplicitIntegration", "SubStepping"])
def test_dp_dispatch_separates_closed_form_from_rate_coupling(integration):
    model = _model(DruckerPragerModel, integration, RateDependent=False)
    static_core = model.ExplicitIntegration if integration == "SubStepping" else model.ClosedFormIntegration
    assert model.core == static_core
    model.set_rate_dependent_model(PARAMETERS)
    assert model.core == model.ImplicitIntegration
    assert DruckerPragerModel.ImplicitIntegration is PlasticMaterial.ImplicitIntegration
    model.set_rate_dependent_model(dict(PARAMETERS, RateDependent=False))
    assert model.core == static_core


def test_default_dynamic_friction_and_grain_density():
    parameters = dict(PARAMETERS)
    parameters.pop("DynamicFriction")
    parameters.pop("Density")
    rate = RateDependent(parameters)
    assert rate.phi2 == rate.phi1 == pytest.approx(math.tan(math.pi / 6.0))
    assert rate.inertial_density == 2650.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AverageDiameter", 0.0),
        ("InertialNumber", -1.0),
        ("GrainDensity", float("nan")),
        ("eps", -1.0),
        ("DynamicFriction", 20.0),
        ("StaticFriction", 90.0),
    ],
)
def test_invalid_rate_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        RateDependent(dict(PARAMETERS, **{name: value}))


@pytest.mark.parametrize("pressure", [0.1, 50.0, 1000.0])
@pytest.mark.parametrize("rate_value", [0.0, 1.0, -1.0])
def test_mu_and_pressure_derivatives(taichi_material_cpu, pressure, rate_value):
    rate = RateDependent(PARAMETERS)
    output = ti.field(ti.f64, shape=4)
    h = pressure * 1.0e-5

    @ti.kernel
    def evaluate():
        output[0] = rate.GetMuI(pressure, rate_value)
        output[1] = rate.DMuIDpressure(pressure, rate_value)
        output[2] = rate.GetInertialNumber(pressure, rate_value)
        output[3] = rate.DInertialNumberDpressure(pressure, rate_value)

    evaluate()
    result = output.to_numpy()
    assert result[0] == pytest.approx(_mu(pressure, rate_value))
    assert result[1] == pytest.approx(
        (_mu(pressure + h, rate_value) - _mu(pressure - h, rate_value)) / (2.0 * h), rel=1.0e-7, abs=1.0e-12
    )
    inertial_number = abs(rate_value) * PARAMETERS["AverageDiameter"] * math.sqrt(PARAMETERS["GrainDensity"] / pressure)
    assert result[2] == pytest.approx(inertial_number)
    assert result[3] == pytest.approx(-0.5 * inertial_number / pressure)


@pytest.mark.parametrize(
    ("pressure", "tau"), [(1000.0, 0.0), (1000.0, 100.0), (1000.0, 1000.0), (0.0, 0.0), (0.0, 100.0), (-100.0, 100.0)]
)
@pytest.mark.parametrize("eps", [0.0, 0.01])
def test_granular_consistency_and_zero_pressure(taichi_material_cpu, pressure, tau, eps):
    parameters = dict(PARAMETERS, eps=eps)
    rate = RateDependent(parameters)
    stress = ti.Vector.field(6, ti.f64, shape=())
    increment = ti.field(ti.f64, shape=())
    dt = ti.field(ti.f64, shape=())
    dt[None] = 0.01

    @ti.kernel
    def solve():
        increment[None], stress[None] = rate.solve(
            10000.0, ti.Vector([-pressure, -pressure, -pressure, tau, 0.0, 0.0]), dt
        )

    solve()
    value = stress.to_numpy()
    assert np.isfinite(value).all() and np.isfinite(increment[None])
    if pressure < 0.0:
        np.testing.assert_allclose(value, 0.0)
    elif tau <= rate.phi1 * pressure:
        np.testing.assert_allclose(value, [-pressure, -pressure, -pressure, tau, 0.0, 0.0])
        assert increment[None] == 0.0
    else:
        plastic_rate = increment[None] / dt[None]
        assert value[3] == pytest.approx(pressure * _mu(pressure, plastic_rate, parameters), abs=1.0e-9)
        assert tau - value[3] == pytest.approx(10000.0 * increment[None])


@pytest.mark.parametrize("integration", ["ReturnMapping", "ImplicitIntegration", "SubStepping"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_dp_rate_return_mapping_matches_scalar_oracle(taichi_material_cpu, integration, dimension):
    model = _model(DruckerPragerModel, integration)
    stress, state, moduli = _update(model, dimension=dimension)
    shear = moduli[1]
    tau_trial = shear * 0.1
    lower, upper = 0.0, tau_trial
    for _ in range(80):
        tau = 0.5 * (lower + upper)
        rate = (tau_trial - tau) / (shear * 0.01)
        angle = math.atan(_mu(1100.0, rate))
        q, k, _, _ = model.choose_yield_surface_type(100.0, angle, 0.0, 1.0e22)
        if tau > 1000.0 * q + k:
            upper = tau
        else:
            lower = tau
    expected = 0.5 * (lower + upper)
    assert stress[3] == pytest.approx(expected, rel=2.0e-8)
    assert state.strain_rate[0] == pytest.approx((tau_trial - expected) / (shear * 0.01), rel=2.0e-8)
    assert state.epdstrain[0] == pytest.approx((tau_trial - expected) / shear / math.sqrt(3.0), rel=2.0e-8)


@pytest.mark.parametrize("dimension", [2, 3])
def test_dp_unified_rate_return_couples_pressure_and_dilation(taichi_material_cpu, dimension):
    model = _model(DruckerPragerModel, Dilation=5.0)
    stress, state, moduli = _update(model, dimension=dimension)
    bulk, shear = moduli
    _, _, q_psi, _ = model.choose_yield_surface_type(100.0, math.pi / 6.0, math.radians(5.0), 1.0e22)
    lower, upper = 0.0, 0.1
    for _ in range(80):
        multiplier = 0.5 * (lower + upper)
        pressure = 1000.0 + bulk * q_psi * multiplier
        angle = math.atan(_mu(pressure + 100.0, multiplier / 0.01))
        q, k, _, _ = model.choose_yield_surface_type(100.0, angle, math.radians(5.0), 1.0e22)
        residual = shear * (0.1 - multiplier) - q * pressure - k
        if residual > 0.0:
            lower = multiplier
        else:
            upper = multiplier
    multiplier = 0.5 * (lower + upper)
    pressure = 1000.0 + bulk * q_psi * multiplier
    np.testing.assert_allclose(stress[:3], -pressure, rtol=2.0e-8)
    assert stress[3] == pytest.approx(shear * (0.1 - multiplier), rel=2.0e-8)
    assert state.strain_rate[0] == pytest.approx(multiplier / 0.01, rel=2.0e-8)
    assert state.rate_pressure[0] == pytest.approx(pressure, rel=2.0e-8)
    assert state.epdstrain[0] == pytest.approx(multiplier * math.sqrt(1.0 / 3.0 + 2.0 * q_psi**2 / 9.0), rel=2.0e-8)


@pytest.mark.parametrize("model_class", [DruckerPragerModel, MohrCoulombModel, ModifiedCamClayModel])
def test_rate_strength_and_physical_time_step(taichi_material_cpu, model_class):
    model = _model(model_class)
    slow, slow_state, _ = _update(model, dt_value=0.01, shear_rate=10.0)
    fast, fast_state, _ = _update(model, dt_value=0.001, shear_rate=100.0)
    assert np.isfinite(slow).all() and np.isfinite(fast).all()
    assert fast[3] > slow[3]
    assert fast_state.strain_rate[0] > slow_state.strain_rate[0]


@pytest.mark.parametrize("model_class", [DruckerPragerModel, MohrCoulombModel, ModifiedCamClayModel])
def test_elastic_step_has_zero_plastic_rate(taichi_material_cpu, model_class):
    model = _model(model_class)
    stress, state, moduli = _update(model, shear_rate=0.001)
    assert stress[3] == pytest.approx(moduli[1] * 0.001 * 0.01)
    assert state.strain_rate[0] == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize("model_class", [DruckerPragerModel, MohrCoulombModel, ModifiedCamClayModel])
def test_equal_friction_limits_recover_static_material(taichi_material_cpu, model_class):
    rate_model = _model(model_class, DynamicFriction=30.0)
    static_model = _model(model_class, RateDependent=False)
    actual, actual_state, _ = _update(rate_model)
    expected, expected_state, _ = _update(static_model)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-8, atol=1.0e-8)
    assert actual_state.epdstrain[0] == pytest.approx(expected_state.epdstrain[0])


@pytest.mark.parametrize("surface", ["Circumscribed", "MiddleCircumscribed", "Inscribed"])
def test_dp_strength_derivatives(taichi_material_cpu, surface):
    model = _model(DruckerPragerModel, dpType=surface)
    result = ti.field(ti.f64, shape=3)

    @ti.kernel
    def evaluate():
        result[0], result[1], result[2] = model.get_strength_derivative(100.0, 0.5)

    evaluate()
    h = 1.0e-6
    qp, kp, _, _ = model.choose_yield_surface_type(100.0, 0.5 + h, 0.0, 1.0e22)
    qm, km, _, _ = model.choose_yield_surface_type(100.0, 0.5 - h, 0.0, 1.0e22)
    np.testing.assert_allclose(result.to_numpy()[:2], [(qp - qm) / (2 * h), (kp - km) / (2 * h)], rtol=1.0e-8)


def test_granular_material_update_uses_shared_rate_solve(taichi_material_cpu):
    model = _model(GranularMaterial)
    stress, state, moduli = _update(model)
    rate = state.epdstrain[0] / 0.01
    assert stress[3] == pytest.approx(1000.0 * _mu(1000.0, rate), rel=1.0e-9)
    assert moduli[1] * 0.1 - stress[3] == pytest.approx(moduli[1] * state.epdstrain[0])


@pytest.mark.parametrize("model_class", [DruckerPragerModel, MohrCoulombModel, ModifiedCamClayModel])
def test_converged_rate_pressure_and_yield_surface(taichi_material_cpu, model_class):
    model = _model(model_class)
    stress, state, moduli = _update(model)
    output = ti.field(ti.f64, shape=())
    value = ti.Vector(stress)

    @ti.kernel
    def evaluate():
        parameters = model.GetMaterialParameter(value, state[0])
        internal = model.GetInternalVariables(state[0])
        branch, output[None] = model.ComputeYieldState(value, internal, parameters)

    evaluate()
    assert state.rate_pressure[0] == pytest.approx(-np.mean(stress[:3]), rel=1.0e-8)
    dev_delta = np.array([0.0, 0.0, 0.0, moduli[1] * 0.1, 0.0, 0.0]) - stress
    dev_delta[:3] -= np.mean(dev_delta[:3])
    plastic_rate = math.sqrt(
        2.0 * (np.dot(dev_delta[:3], dev_delta[:3]) + 2.0 * np.dot(dev_delta[3:], dev_delta[3:]))
    ) / (2.0 * moduli[1] * 0.01)
    assert state.strain_rate[0] == pytest.approx(plastic_rate, rel=1.0e-8)
    assert abs(output[None]) < 1.0e-3


def test_stiff_rate_feedback_near_initial_yield(taichi_material_cpu):
    model = _model(DruckerPragerModel, AverageDiameter=0.3, InertialNumber=0.001)
    shear_rate = 600.0 / (model.shear * 0.01)
    stress, state, _ = _update(model, shear_rate=shear_rate)
    assert 0.0 < stress[3] < 600.0
    assert 0.0 < state.strain_rate[0] < shear_rate


@pytest.mark.parametrize("integration", ["ReturnMapping", "SubStepping", "ImplicitIntegrationAL"])
def test_mcc_rate_with_supported_local_integrators(taichi_material_cpu, integration):
    stress, state, _ = _update(_model(ModifiedCamClayModel, integration))
    assert np.isfinite(stress).all() and state.strain_rate[0] > 0.0


def test_mcc_infers_static_friction_from_stress_ratio():
    parameters = dict(PARAMETERS)
    parameters.pop("StaticFriction")
    parameters.update(StressRatio=1.2, **{"lambda": 0.1, "kappa": 0.03}, void_ratio_ref=0.7, pc=2000.0)
    model = ModifiedCamClayModel()
    model.model_initialize(parameters)
    assert model.rate_dependent_function.phi1 == pytest.approx(math.tan(math.pi / 6.0))


@pytest.mark.parametrize("model_class", [DruckerPragerModel, MohrCoulombModel, ModifiedCamClayModel])
def test_implicit_global_solver_rejects_unimplemented_rate_tangent(model_class):
    model = model_class(solver_type="Implicit")
    with pytest.raises(ValueError, match="consistent rate-dependent tangent"):
        model.set_rate_dependent_model(PARAMETERS)
    assert not model.is_rate_dependent


@pytest.mark.parametrize(
    "model_class,extra",
    [
        (ElasticPerfectlyPlasticModel, dict(YieldStress=100.0)),
        (
            NorSandModel,
            dict(G0=100.0, kappa=0.01, **{"lambda": 0.05}, M=1.25, N=0.3, beta=3.5, vc0=1.9, v0=1.8, h=150.0),
        ),
    ],
)
@pytest.mark.parametrize("solver_type", ["Explicit", "Implicit"])
def test_unsupported_rate_combinations_fail_during_material_initialization(model_class, extra, solver_type):
    material = dict(PARAMETERS, **extra)
    model = model_class(solver_type=solver_type)
    expected = (
        "does not implement mu\\(I\\) RateDependent"
        if solver_type == "Explicit"
        else "consistent rate-dependent tangent"
    )
    with pytest.raises(ValueError, match=expected):
        model.model_initialize(material)


@pytest.mark.parametrize("model_class", [DruckerPragerModel, MohrCoulombModel, ModifiedCamClayModel])
def test_implicit_rate_rejection_happens_at_initialization(model_class):
    material = dict(PARAMETERS)
    if model_class is ModifiedCamClayModel:
        material.update(StressRatio=1.2, **{"lambda": 0.1, "kappa": 0.03}, void_ratio_ref=0.7, pc=2000.0)
    with pytest.raises(ValueError, match="consistent rate-dependent tangent"):
        model_class(solver_type="Implicit").model_initialize(material)


@pytest.mark.parametrize("name", ["DruckerPrager", "MohrCoulomb", "ModifiedCamClay"])
def test_native_implicit_mpm_rejects_rate_material_before_running(taichi_material_cpu, name):
    from src.mpm.mainMPM import MPM

    mpm = MPM(log=False)
    mpm.set_configuration(dimension=2, domain=[0.1, 0.1], solver_type="Implicit", log=False)
    mpm.memory_allocate(memory={"max_material_number": 1, "max_particle_number": 1}, log=False)
    material = dict(PARAMETERS, MaterialID=1)
    if name == "ModifiedCamClay":
        material.update(StressRatio=1.2, **{"lambda": 0.1, "kappa": 0.03}, void_ratio_ref=0.7, pc=2000.0)
    with pytest.raises(ValueError, match="consistent rate-dependent tangent"):
        mpm.add_material(model=name, material=material)


@pytest.mark.parametrize("model_class", [DruckerPragerModel, MohrCoulombModel, ModifiedCamClayModel])
def test_frozen_rate_yield_gradient_matches_finite_difference(taichi_material_cpu, model_class):
    model = _model(model_class)
    stress, state, _ = _update(model)
    output = ti.field(ti.f64, shape=(6, 2))
    cam_clay = model_class is ModifiedCamClayModel

    @ti.kernel
    def evaluate():
        value = ti.Vector([-1200.0, -950.0, -850.0, 310.0, -70.0, 80.0])
        parameters = model.GetMaterialParameter(value, state[0])
        internal = model.GetInternalVariables(state[0])
        grad = model.ComputeDfDsigma(1, value, internal, parameters)
        for component in ti.static(range(6)):
            plus, minus = value, value
            plus[component] += 0.001
            minus[component] -= 0.001
            fp, fm = 0.0, 0.0
            if ti.static(cam_clay):
                fp = model.ComputeYieldFunction(plus, internal, parameters)
                fm = model.ComputeYieldFunction(minus, internal, parameters)
            else:
                fp, ft = model.ComputeYieldFunction(plus, internal, parameters)
                fm, ft = model.ComputeYieldFunction(minus, internal, parameters)
            output[component, 0] = grad[component] * (1.0 if component < 3 else 2.0)
            output[component, 1] = (fp - fm) / 0.002

    evaluate()
    result = output.to_numpy()
    np.testing.assert_allclose(result[:, 0], result[:, 1], rtol=2.0e-6, atol=1.0e-7)


@pytest.mark.parametrize("name", ["DruckerPrager", "MohrCoulomb", "ModifiedCamClay"])
def test_direct_explicit_kernel_initializes_matrix_stress_and_updates_rate(taichi_material_cpu, name):
    from src.mpm.engines.direct.ExplicitULMPM import ExplicitULMPM

    parameters = dict(PARAMETERS)
    if name == "ModifiedCamClay":
        parameters.update(StressRatio=1.2, **{"lambda": 0.1, "kappa": 0.03}, void_ratio_ref=0.7, pc=2000.0)
    engine = object.__new__(ExplicitULMPM)
    engine.configuration = "UL"
    engine.handle_material(material=name, material_parameters=parameters)
    engine.stateVars = ti.types.struct(**engine.material.get_state_vars()).field(shape=1)
    engine.particleNum = ti.field(ti.i32, shape=1)
    engine.particleNum[0] = 1
    engine.particle = ti.types.struct(stress=ti.types.matrix(3, 3, ti.f64), vol0=ti.f64).field(shape=1)
    engine.particle.stress[0] = -1000.0 * np.eye(3)
    engine.particle.vol0[0] = 1.0
    engine.grid = ti.types.struct(v=ti.types.vector(2, ti.f64)).field(shape=2)
    engine.grid.v[0], engine.grid.v[1] = [0.0, 5.0], [5.0, 0.0]
    engine.offset = ti.field(ti.i32, shape=1)
    engine.offset[0] = 2
    engine.LnID = ti.field(ti.i32, shape=(1, 2))
    engine.LnID[0, 0], engine.LnID[0, 1] = 0, 1
    engine.dshape = ti.Vector.field(2, ti.f64, shape=(1, 2))
    engine.dshape[0, 0], engine.dshape[0, 1] = [1.0, 0.0], [0.0, 1.0]
    engine.TIdt = ti.field(ti.f64, shape=())
    engine.TIdt[None] = 0.01
    engine.initial_material()
    engine.stress_update()
    stress = engine.particle.stress.to_numpy()[0]
    assert np.isfinite(stress).all()
    assert engine.stateVars.strain_rate[0] > 0.0
    assert engine.stateVars.rate_pressure[0] == pytest.approx(-np.trace(stress) / 3.0, rel=1.0e-8)


@pytest.mark.parametrize("model_class", [DruckerPragerModel, MohrCoulombModel])
def test_random_field_initialization_enables_rate_extension(tmp_path, model_class):
    path = tmp_path / "material.txt"
    path.write_text("1800 200000 .3 100 30 0 0\n")
    model = model_class()
    model.random_field_initialize(dict(PARAMETERS, MaterialFile=str(path)))
    assert model.is_rate_dependent
    assert {"strain_rate", "rate_pressure"} <= model.get_state_vars().keys()


def test_unsupported_solid_rejects_rate_flag():
    from src.physics_model.consititutive_model.infinitesimal_strain.LinearElastic import LinearElasticModel

    with pytest.raises(ValueError, match="does not implement"):
        LinearElasticModel().model_initialize(PARAMETERS)


@pytest.mark.parametrize("model_class", [DruckerPragerModel, MohrCoulombModel])
@pytest.mark.parametrize("rate_dependent", [False, True])
def test_softening_rate_chain_matches_history_finite_difference(taichi_material_cpu, model_class, rate_dependent):
    from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import DeqepsilonqDepsilon
    from src.utils.VectorFunction import voigt_tensor_dot

    model = _model(
        model_class,
        RateDependent=rate_dependent,
        SoftType="Linear",
        ResidualFriction=15.0,
        ResidualCohesion=50.0,
        PlasticDevStrain=0.0,
        ResidualPlasticDevStrain=0.5,
    )
    state = ti.types.struct(**model.get_state_vars()).field(shape=1)
    output = ti.field(ti.f64, shape=2)
    dp = model_class is DruckerPragerModel

    @ti.kernel
    def evaluate():
        if ti.static(dp):
            state[0].epdstrain = 0.1
        else:
            state[0].strain = ti.Vector([0.0, 0.0, 0.0, 0.1 * ti.sqrt(1.5), 0.0, 0.0])
        if ti.static(rate_dependent):
            state[0].strain_rate, state[0].rate_pressure = 3.0, 1000.0
        value = ti.Vector([-1200.0, -950.0, -850.0, 310.0, -70.0, 80.0])
        flow = ti.Vector([0.1, -0.05, -0.05, 0.2, 0.1, -0.03])
        parameters = model.GetMaterialParameter(value, state[0])
        internal = model.GetInternalVariables(state[0])
        modulus = model.ComputePlasticModulus(1, flow, value, internal, state[0], parameters)
        if ti.static(dp):
            output[0] = modulus
        else:
            output[0] = modulus / voigt_tensor_dot(DeqepsilonqDepsilon(state[0].strain), flow)
        plus, minus = state[0], state[0]
        if ti.static(dp):
            # Differentiate along the plastic multiplier, including d(kappa)/d(lambda).
            history_rate = model.ComputeInternalVariables(1, flow, internal, parameters)[0]
            plus.epdstrain += 1.0e-6 * history_rate
            minus.epdstrain -= 1.0e-6 * history_rate
        else:
            plus.strain *= 1.0 + 1.0e-5
            minus.strain *= 1.0 - 1.0e-5
        fp, ft = model.ComputeYieldFunction(value, internal, model.GetMaterialParameter(value, plus))
        fm, ft = model.ComputeYieldFunction(value, internal, model.GetMaterialParameter(value, minus))
        output[1] = (fp - fm) / 2.0e-6

    evaluate()
    assert output[0] == pytest.approx(output[1], rel=1.0e-7)
