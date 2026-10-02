"""Small constitutive contracts for the classical elastoplastic models.

Each model is checked independently for its input contract, state schema,
elastic and plastic branches, history evolution, and algorithmic tangent.
"""

import math

import numpy as np
import pytest
import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.consititutive_model.infinitesimal_strain.DruckerPrager import (
    DruckerPragerModel,
)
from src.physics_model.consititutive_model.infinitesimal_strain.ElasticPerfectlyPlastic import (
    ElasticPerfectlyPlasticModel,
)
from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import (
    MohrCoulombModel,
)

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


@ti.dataclass
class _VonMisesState:
    epstrain: ti.f64
    yield_state: ti.u8


@ti.dataclass
class _FrictionalState:
    epdstrain: ti.f64
    yield_state: ti.u8


def _elastic_stiffness(bulk: float, shear: float) -> np.ndarray:
    """Return the 3-D isotropic stiffness for engineering Voigt strain."""

    normal_diagonal = bulk + 4.0 * shear / 3.0
    normal_off_diagonal = bulk - 2.0 * shear / 3.0
    stiffness = np.zeros((6, 6), dtype=np.float64)
    stiffness[:3, :3] = normal_off_diagonal
    np.fill_diagonal(stiffness[:3, :3], normal_diagonal)
    np.fill_diagonal(stiffness[3:, 3:], shear)
    return stiffness


def _von_mises_equivalent(stress: np.ndarray) -> float:
    return math.sqrt(
        0.5 * ((stress[0] - stress[1]) ** 2 + (stress[1] - stress[2]) ** 2 + (stress[2] - stress[0]) ** 2)
        + 3.0 * np.dot(stress[3:], stress[3:])
    )


def _make_von_mises(*, yield_stress: float = 100.0):
    model = ElasticPerfectlyPlasticModel(
        configuration="UL",
        solver_type="Implicit",
        stress_integration="ReturnMapping",
    )
    model.model_initialize(
        {
            "Density": 1800.0,
            "YoungModulus": 2.0e5,
            "PoissonRatio": 0.3,
            "YieldStress": yield_stress,
        }
    )
    return model


def _make_mohr_coulomb(*, cohesion: float = 100.0):
    model = MohrCoulombModel(
        configuration="UL",
        solver_type="Implicit",
        stress_integration="ReturnMapping",
    )
    model.model_initialize(
        {
            "Density": 1800.0,
            "YoungModulus": 2.0e5,
            "PoissonRatio": 0.3,
            "Cohesion": cohesion,
            "Friction": 30.0,
            "Dilation": 5.0,
            "Tensile": cohesion,
        }
    )
    return model


def _make_drucker_prager(*, cohesion: float = 100.0):
    model = DruckerPragerModel(
        configuration="UL",
        solver_type="Implicit",
        stress_integration="ReturnMapping",
    )
    model.model_initialize(
        {
            "Density": 1800.0,
            "YoungModulus": 2.0e5,
            "PoissonRatio": 0.3,
            "Cohesion": cohesion,
            "Friction": 30.0,
            "Dilation": 5.0,
            "Tensile": cohesion,
            "dpType": "MiddleCircumscribed",
        }
    )
    return model


_MODEL_CASES = [
    pytest.param("von-mises", _make_von_mises, id="elastic-perfectly-plastic"),
    pytest.param("frictional", _make_mohr_coulomb, id="mohr-coulomb"),
    pytest.param("frictional", _make_drucker_prager, id="drucker-prager"),
]

_PLASTIC_MODEL_CASES = [
    pytest.param(
        "von-mises",
        _make_von_mises,
        id="elastic-perfectly-plastic",
    ),
    pytest.param("frictional", _make_mohr_coulomb, id="mohr-coulomb"),
    pytest.param("frictional", _make_drucker_prager, id="drucker-prager"),
]


def _evaluate_von_mises(
    model,
    engineering_increments,
    previous_stresses,
    initial_history,
):
    increments = np.asarray(engineering_increments, dtype=np.float64)
    previous = np.asarray(previous_stresses, dtype=np.float64)
    history = np.broadcast_to(np.asarray(initial_history, dtype=np.float64), (increments.shape[0],)).copy()
    count = increments.shape[0]
    state = _VonMisesState.field(shape=count)
    stress = ti.Vector.field(6, dtype=ti.f64, shape=count)
    tangent = ti.Matrix.field(6, 6, dtype=ti.f64, shape=count)
    final_yield_state = ti.field(dtype=ti.i32, shape=count)
    final_yield_value = ti.field(dtype=ti.f64, shape=count)

    @ti.kernel
    def evaluate(
        strain: ti.types.ndarray(dtype=ti.f64, ndim=2),
        old_stress: ti.types.ndarray(dtype=ti.f64, ndim=2),
        old_history: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for sample in range(count):
            de = ti.Vector.zero(ti.f64, 6)
            sigma0 = ti.Vector.zero(ti.f64, 6)
            for component in ti.static(range(6)):
                de[component] = strain[sample, component]
                sigma0[component] = old_stress[sample, component]
            # The constitutive core consumes tensorial shear strain, while its
            # 6x6 tangent is expressed against engineering shear strain.
            for component in ti.static(range(3, 6)):
                de[component] *= 0.5

            state[sample].epstrain = old_history[sample]
            state[sample].yield_state = ti.u8(0)
            sigma = model.core(
                sample,
                sigma0,
                de,
                ti.Vector.zero(ti.f64, 3),
                state,
            )
            stress[sample] = sigma
            tangent[sample] = model.compute_stiffness_tensor(sample, sigma, state)
            internal = model.GetInternalVariables(state[sample])
            parameters = model.GetMaterialParameter(sigma, state[sample])
            branch, value = model.ComputeYieldState(sigma, internal, parameters)
            final_yield_state[sample] = branch
            final_yield_value[sample] = value

    evaluate(increments, previous, history)
    return {
        "stress": stress.to_numpy(),
        "tangent": tangent.to_numpy(),
        "history": state.epstrain.to_numpy(),
        "trial_branch": state.yield_state.to_numpy(),
        "final_branch": final_yield_state.to_numpy(),
        "yield_value": final_yield_value.to_numpy(),
    }


def _evaluate_frictional(
    model,
    engineering_increments,
    previous_stresses,
    initial_history,
):
    increments = np.asarray(engineering_increments, dtype=np.float64)
    previous = np.asarray(previous_stresses, dtype=np.float64)
    history = np.broadcast_to(np.asarray(initial_history, dtype=np.float64), (increments.shape[0],)).copy()
    count = increments.shape[0]
    state = _FrictionalState.field(shape=count)
    stress = ti.Vector.field(6, dtype=ti.f64, shape=count)
    tangent = ti.Matrix.field(6, 6, dtype=ti.f64, shape=count)
    final_yield_state = ti.field(dtype=ti.i32, shape=count)
    final_yield_value = ti.field(dtype=ti.f64, shape=count)

    @ti.kernel
    def evaluate(
        strain: ti.types.ndarray(dtype=ti.f64, ndim=2),
        old_stress: ti.types.ndarray(dtype=ti.f64, ndim=2),
        old_history: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for sample in range(count):
            de = ti.Vector.zero(ti.f64, 6)
            sigma0 = ti.Vector.zero(ti.f64, 6)
            for component in ti.static(range(6)):
                de[component] = strain[sample, component]
                sigma0[component] = old_stress[sample, component]
            for component in ti.static(range(3, 6)):
                de[component] *= 0.5

            state[sample].epdstrain = old_history[sample]
            state[sample].yield_state = ti.u8(0)
            sigma = model.core(
                sample,
                sigma0,
                de,
                ti.Vector.zero(ti.f64, 3),
                state,
            )
            stress[sample] = sigma
            tangent[sample] = model.compute_stiffness_tensor(sample, sigma, state)
            internal = model.GetInternalVariables(state[sample])
            parameters = model.GetMaterialParameter(sigma, state[sample])
            branch, value = model.ComputeYieldState(sigma, internal, parameters)
            final_yield_state[sample] = branch
            final_yield_value[sample] = value

    evaluate(increments, previous, history)
    return {
        "stress": stress.to_numpy(),
        "tangent": tangent.to_numpy(),
        "history": state.epdstrain.to_numpy(),
        "trial_branch": state.yield_state.to_numpy(),
        "final_branch": final_yield_state.to_numpy(),
        "yield_value": final_yield_value.to_numpy(),
    }


def _evaluate(
    kind,
    model,
    engineering_increments,
    previous_stresses=None,
    initial_history=0.0,
):
    increments = np.atleast_2d(np.asarray(engineering_increments, dtype=np.float64))
    if previous_stresses is None:
        previous = np.zeros_like(increments)
    else:
        previous = np.broadcast_to(np.asarray(previous_stresses, dtype=np.float64), increments.shape).copy()
    if kind == "von-mises":
        return _evaluate_von_mises(model, increments, previous, initial_history)
    return _evaluate_frictional(model, increments, previous, initial_history)


def test_elastic_perfectly_plastic_parameters_and_defaults():
    model = ElasticPerfectlyPlasticModel()
    model.model_initialize({"YoungModulus": 3.0e6, "YieldStress": 1200.0})

    assert model.density == 2650
    assert model.poisson == 0.3
    assert model.shear == pytest.approx(3.0e6 / 2.6)
    assert model.bulk == pytest.approx(3.0e6 / 1.2)
    assert model._yield_peak == 1200.0
    assert model._yield_residual == 1200.0


def test_mohr_coulomb_parameters_use_radians_and_limit_tension():
    model = MohrCoulombModel()
    model.model_initialize(
        {
            "YoungModulus": 3.0e6,
            "Cohesion": 100.0,
            "Friction": 30.0,
            "Dilation": 10.0,
            "Tensile": 1.0e9,
        }
    )

    assert model.fai_peak == pytest.approx(math.pi / 6.0)
    assert model.psi_peak == pytest.approx(math.pi / 18.0)
    assert model.tensile == pytest.approx(100.0 / math.tan(math.pi / 6.0))


def test_double_layer_material_validates_maximum_porosity():
    parameters = {
        "YoungModulus": 3.0e6,
        "Porosity": 0.38,
        "MaximumPorosity": 0.50,
        "CavitationPressure": 0.0,
        "FluidBulkModulus": 2.2e8,
        "Permeability": 1.0e-3,
    }
    model = MohrCoulombModel(material_type="TwoPhaseDoubleLayer")
    model.initialize_coupling()
    model.model_initialize(parameters)
    assert model.maximum_porosity == pytest.approx(0.50)
    assert model.cavitation_pressure == pytest.approx(0.0)

    parameters["MaximumPorosity"] = 0.37
    with pytest.raises(ValueError, match="MaximumPorosity"):
        model.model_initialize(parameters)


def test_single_layer_zeroes_effective_stress_above_maximum_porosity(
    taichi_material_cpu,
):
    model = MohrCoulombModel(material_type="TwoPhaseSingleLayer")
    model.initialize_coupling()
    model.model_initialize(
        {
            "YoungModulus": 3.0e6,
            "Cohesion": 1000.0,
            "Friction": 30.0,
            "Porosity": 0.38,
            "MaximumPorosity": 0.50,
            "FluidBulkModulus": 2.2e8,
            "Permeability": 1.0e-3,
        }
    )
    dt = ti.field(ti.f64, shape=())
    dt[None] = 1.0e-4
    state = _FrictionalState.field(shape=2)
    stress = ti.Vector.field(6, ti.f64, shape=2)

    @ti.kernel
    def update():
        previous = ti.Vector([-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0])
        for sample in range(2):
            porosity = 0.49 + 0.02 * sample
            stress[sample] = model.compute_single_layer_effective_stress_2d(
                sample,
                previous,
                ti.Matrix.zero(ti.f64, 2, 2),
                porosity,
                state,
                dt,
            )

    update()
    np.testing.assert_allclose(stress.to_numpy()[0], -1000.0 * np.array([1, 1, 1, 0, 0, 0]))
    np.testing.assert_allclose(stress.to_numpy()[1], 0.0)


@pytest.mark.parametrize(
    ("surface", "index"),
    [
        ("Circumscribed", 0),
        ("MiddleCircumscribed", 1),
        ("Inscribed", 2),
    ],
)
def test_drucker_prager_surface_parameters(surface, index):
    model = DruckerPragerModel()
    model.model_initialize(
        {
            "YoungModulus": 3.0e6,
            "Cohesion": 100.0,
            "Friction": 30.0,
            "Dilation": 10.0,
            "dpType": surface,
        }
    )

    assert model.yield_surface_type == index
    assert model.fai_peak == pytest.approx(math.pi / 6.0)
    assert model.psi_peak == pytest.approx(math.pi / 18.0)
    assert model.q_fai > 0.0
    assert model.k_fai > 0.0
    assert model.q_psi > 0.0


@pytest.mark.parametrize(
    ("model_class", "history_name"),
    [
        (ElasticPerfectlyPlasticModel, "epstrain"),
        (MohrCoulombModel, "epdstrain"),
        (DruckerPragerModel, "epdstrain"),
    ],
)
def test_state_schema_is_minimal_and_implicit_adds_trial_branch(model_class, history_name):
    explicit = model_class(configuration="UL", solver_type="Explicit")
    implicit = model_class(configuration="UL", solver_type="Implicit")

    assert explicit.get_state_vars() == {history_name: float}
    assert implicit.get_state_vars() == {
        history_name: float,
        "yield_state": ti.u8,
    }
    assert implicit._initialize_vars == implicit._initialize_vars_update_lagrangian


@pytest.mark.parametrize(
    ("model_class", "parameters", "missing_name"),
    [
        (ElasticPerfectlyPlasticModel, {"YieldStress": 100.0}, "YoungModulus"),
        (
            ElasticPerfectlyPlasticModel,
            {"YoungModulus": 2.0e5},
            "YieldStress",
        ),
        (MohrCoulombModel, {"Cohesion": 100.0}, "YoungModulus"),
        (DruckerPragerModel, {"Friction": 30.0}, "YoungModulus"),
        (
            DruckerPragerModel,
            {"YoungModulus": 2.0e5},
            "Friction|StaticFriction",
        ),
    ],
)
def test_missing_required_parameters_are_rejected(model_class, parameters, missing_name):
    with pytest.raises(KeyError, match=missing_name):
        model_class().model_initialize(parameters)


@pytest.mark.parametrize(
    ("model_class", "parameters"),
    [
        (
            ElasticPerfectlyPlasticModel,
            {
                "Density": -1.0,
                "YoungModulus": 2.0e5,
                "YieldStress": -100.0,
            },
        ),
        (
            MohrCoulombModel,
            {
                "YoungModulus": -2.0e5,
                "Cohesion": -100.0,
                "Friction": 95.0,
            },
        ),
        (
            DruckerPragerModel,
            {
                "YoungModulus": 2.0e5,
                "PoissonRatio": -1.1,
                "Cohesion": -100.0,
                "Friction": -5.0,
            },
        ),
    ],
)
def test_nonphysical_parameters_are_rejected(model_class, parameters):
    with pytest.raises(ValueError):
        model_class().model_initialize(parameters)


@pytest.mark.parametrize(("kind", "factory"), _MODEL_CASES)
def test_zero_increment_preserves_stress_and_history(taichi_material_cpu, kind, factory):
    # A zero increment preserves an admissible state.  The common previous
    # stress lies outside the default 100 Pa von-Mises surface, so use a wide
    # surface for that branch instead of asking return mapping to retain an
    # invalid initial stress.
    model = factory(yield_stress=1.0e9) if kind == "von-mises" else factory()
    previous = np.array(
        [-800.0, -700.0, -600.0, 5.0, -3.0, 2.0],
        dtype=np.float64,
    )
    result = _evaluate(
        kind,
        model,
        np.zeros(6),
        previous_stresses=previous,
        initial_history=0.125,
    )

    np.testing.assert_allclose(result["stress"][0], previous, atol=1.0e-12)
    assert result["history"][0] == pytest.approx(0.125)
    assert result["trial_branch"][0] == 0


@pytest.mark.parametrize(("kind", "factory"), _MODEL_CASES)
def test_clear_elastic_branch_matches_isotropic_oracle(taichi_material_cpu, kind, factory):
    model = factory(**({"yield_stress": 1.0e9} if kind == "von-mises" else {"cohesion": 1.0e9}))
    strain = np.array([2.0e-5, -1.0e-5, 0.5e-5, 3.0e-5, -2.0e-5, 1.0e-5])
    previous = np.array(
        [-800.0, -700.0, -600.0, 5.0, -3.0, 2.0],
        dtype=np.float64,
    )
    result = _evaluate(kind, model, strain, previous_stresses=previous)
    expected_tangent = _elastic_stiffness(model.bulk, model.shear)
    expected_stress = previous + expected_tangent @ strain

    assert result["trial_branch"][0] == 0
    assert result["history"][0] == 0.0
    np.testing.assert_allclose(result["stress"][0], expected_stress, rtol=1.0e-7, atol=5.0e-6)
    np.testing.assert_allclose(
        result["tangent"][0],
        expected_tangent,
        rtol=1.0e-12,
        atol=1.0e-10,
    )


@pytest.mark.parametrize(("kind", "factory"), _MODEL_CASES)
def test_elastic_algorithmic_tangent_matches_central_difference(taichi_material_cpu, kind, factory):
    model = factory(**({"yield_stress": 1.0e9} if kind == "von-mises" else {"cohesion": 1.0e9}))
    base = np.array([2.0e-5, -1.0e-5, 0.5e-5, 3.0e-5, -2.0e-5, 1.0e-5])
    # Several legacy constitutive helpers still create f32 temporaries even
    # under an f64 runtime, so use a step large enough to avoid cancellation.
    step = 1.0e-5
    samples = [base]
    for component in range(6):
        positive = base.copy()
        negative = base.copy()
        positive[component] += step
        negative[component] -= step
        samples.extend((positive, negative))

    result = _evaluate(kind, model, np.asarray(samples))
    finite_difference = np.empty((6, 6), dtype=np.float64)
    for component in range(6):
        finite_difference[:, component] = (
            result["stress"][1 + 2 * component] - result["stress"][2 + 2 * component]
        ) / (2.0 * step)

    np.testing.assert_allclose(
        result["tangent"][0],
        finite_difference,
        rtol=2.0e-6,
        atol=2.0e-2,
    )


@pytest.mark.parametrize(("kind", "factory"), _PLASTIC_MODEL_CASES)
def test_clear_plastic_branch_returns_to_surface_and_updates_history(taichi_material_cpu, kind, factory):
    model = factory()
    previous = np.array(
        [-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    shear_increment = 1.5 if kind == "von-mises" else 4.0e-2
    loading = np.array([0.0, 0.0, 0.0, shear_increment, 0.0, 0.0])
    result = _evaluate(kind, model, loading, previous_stresses=previous)

    assert np.all(np.isfinite(result["stress"][0]))
    assert result["final_branch"][0] > 0
    assert result["history"][0] > 0.0
    assert result["yield_value"][0] <= 1.1e-1


@pytest.mark.parametrize(("kind", "factory"), _PLASTIC_MODEL_CASES)
def test_unloading_does_not_erase_accumulated_plastic_history(taichi_material_cpu, kind, factory):
    model = factory()
    previous = np.array(
        [-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    shear_increment = 1.5 if kind == "von-mises" else 4.0e-2
    loading = np.array([0.0, 0.0, 0.0, shear_increment, 0.0, 0.0])
    loaded = _evaluate(kind, model, loading, previous_stresses=previous)
    unloading = np.array([0.0, 0.0, 0.0, -1.0e-4, 0.0, 0.0])
    unloaded = _evaluate(
        kind,
        model,
        unloading,
        previous_stresses=loaded["stress"][0],
        initial_history=loaded["history"][0],
    )

    assert unloaded["trial_branch"][0] == 0
    assert unloaded["history"][0] == pytest.approx(loaded["history"][0], rel=1.0e-12, abs=1.0e-14)
    assert unloaded["history"][0] >= 0.0


def test_von_mises_return_surface_uses_configured_yield_stress(
    taichi_material_cpu,
):
    model = _make_von_mises(yield_stress=100.0)
    result = _evaluate(
        "von-mises",
        model,
        np.array([0.0, 0.0, 0.0, 2.0e-2, 0.0, 0.0]),
    )

    assert result["history"][0] > 0.0
    assert _von_mises_equivalent(result["stress"][0]) == pytest.approx(model._yield_peak, rel=2.0e-5, abs=2.0e-3)


def test_drucker_prager_plastic_step_records_active_branch(
    taichi_material_cpu,
):
    model = _make_drucker_prager()
    result = _evaluate(
        "frictional",
        model,
        np.array([0.0, 0.0, 0.0, 4.0e-2, 0.0, 0.0]),
        previous_stresses=np.array([-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0]),
    )

    assert result["history"][0] > 0.0
    assert result["trial_branch"][0] > 0
