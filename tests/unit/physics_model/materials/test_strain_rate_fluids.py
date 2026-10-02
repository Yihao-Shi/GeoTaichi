"""Unit tests for Newtonian and Bingham strain-rate materials."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.strain_rate.Bingham import (
    BinghamModel,
)
from src.physics_model.consititutive_model.strain_rate.Newtonian import (
    NewtonianModel,
)


pytestmark = [pytest.mark.materials, pytest.mark.cpu]


@ti.dataclass
class _FluidState:
    pressure: ti.f64
    rho: ti.f64


def _evaluate_shear_stress(model, strain_rate):
    output = ti.Vector.field(6, dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate(rate: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        value = ti.Vector.zero(ti.f64, 6)
        for component in ti.static(range(6)):
            value[component] = rate[component]
        output[None] = model.shear_stress(value)

    evaluate(np.asarray(strain_rate, dtype=np.float64))
    return output.to_numpy()[()]


def _evaluate_core(
    model,
    strain_rate,
    *,
    time_step,
    initial_pressure=0.0,
    density=None,
):
    state = _FluidState.field(shape=1)
    dt = ti.field(dtype=ti.f64, shape=())
    output = ti.Vector.field(6, dtype=ti.f64, shape=())
    dt[None] = time_step
    state.pressure[0] = initial_pressure
    state.rho[0] = model.density if density is None else density

    @ti.kernel
    def evaluate(rate: ti.types.ndarray(dtype=ti.f64, ndim=1)):
        value = ti.Vector.zero(ti.f64, 6)
        for component in ti.static(range(6)):
            value[component] = rate[component]
        output[None] = model.core(0, value, state, dt)

    evaluate(np.asarray(strain_rate, dtype=np.float64))
    return (
        output.to_numpy()[()],
        float(state.pressure.to_numpy()[0]),
        float(state.rho.to_numpy()[0]),
    )


@pytest.mark.parametrize(
    ("model_class", "solver_type", "expected"),
    [
        (NewtonianModel, "Explicit", {"pressure": float, "rho": float}),
        (NewtonianModel, "Implicit", {}),
        (BinghamModel, "Explicit", {"pressure": float, "rho": float}),
        (BinghamModel, "Implicit", {}),
    ],
)
def test_fluid_state_schema_depends_on_solver_type(model_class, solver_type, expected):
    model = model_class(solver_type=solver_type)

    assert model.get_state_vars() == expected
    assert model._initialize_vars == model._initialize_vars_


def test_newtonian_model_initialize_reads_properties_and_defaults():
    model = NewtonianModel()
    model.model_initialize(
        {
            "Density": 998.0,
            "Modulus": 2.2e6,
            "Viscosity": 8.9e-4,
            "ElementLength": 0.02,
            "cL": 0.7,
            "cQ": 1.4,
            "atmospheric_pressure": 101325.0,
            "surface_tension": 0.073,
        }
    )

    assert model.density == 998.0
    assert model.modulus == 2.2e6
    assert model.viscosity == 8.9e-4
    assert model.element_length == 0.02
    assert model.cl == 0.7
    assert model.cq == 1.4
    assert model.atmospheric_pressure == 101325.0
    assert model.surface_tension == 0.073
    assert model.gamma == 1.0
    assert model.max_sound_speed == pytest.approx(np.sqrt(2.2e6 / 998.0))


@pytest.mark.parametrize(
    "strain_rate",
    [
        np.array([0.4, 0.4, 0.4, 0.0, 0.0, 0.0]),
        np.array([0.6, -0.2, 0.1, 0.3, -0.1, 0.2]),
    ],
    ids=["pure-volumetric", "general"],
)
def test_newtonian_shear_stress_matches_deviatoric_oracle(taichi_material_cpu, strain_rate):
    model = NewtonianModel()
    model.add_material(
        density=1000.0,
        modulus=2.0e6,
        viscosity=0.35,
        element_length=0.0,
        cl=0.0,
        cq=0.0,
        atmospheric_pressure=0.0,
    )

    actual = _evaluate_shear_stress(model, strain_rate)
    mean_rate = np.sum(strain_rate[:3]) / 3.0
    expected = 2.0 * model.viscosity * strain_rate
    expected[:3] -= 2.0 * model.viscosity * mean_rate

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    assert np.sum(actual[:3]) == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize(
    ("volumetric_rate", "expected_pressure"),
    [(-0.6, 12000.0), (0.6, 0.0)],
    ids=["compression", "expansion-cavitation-clamp"],
)
def test_newtonian_core_updates_pressure_and_applies_cavitation_clamp(
    taichi_material_cpu, volumetric_rate, expected_pressure
):
    model = NewtonianModel()
    model.add_material(
        density=1000.0,
        modulus=2.0e5,
        viscosity=0.1,
        element_length=0.0,
        cl=0.0,
        cq=0.0,
        atmospheric_pressure=0.0,
    )
    isotropic_rate = np.array([volumetric_rate / 3.0] * 3 + [0.0] * 3)

    stress, stored_pressure, rho = _evaluate_core(model, isotropic_rate, time_step=0.1)

    np.testing.assert_allclose(stress[:3], -expected_pressure, atol=1.0e-12)
    np.testing.assert_allclose(stress[3:], 0.0, atol=1.0e-12)
    assert stored_pressure == pytest.approx(-expected_pressure)
    assert rho == 1000.0


def test_artificial_viscosity_activates_only_in_compression(
    taichi_material_cpu,
):
    model = NewtonianModel()
    model.add_material(
        density=1000.0,
        modulus=2.0e5,
        viscosity=0.1,
        element_length=0.2,
        cl=0.5,
        cq=1.5,
        atmospheric_pressure=0.0,
    )
    state = _FluidState.field(shape=1)
    output = ti.field(dtype=ti.f64, shape=2)
    state.rho[0] = 950.0

    @ti.kernel
    def evaluate():
        output[0] = model.artifical_viscosity(0, -2.0, state)
        output[1] = model.artifical_viscosity(0, 2.0, state)

    evaluate()
    values = output.to_numpy()
    expected_compression = -950.0 * 0.5 * 0.2 * -2.0 + 950.0 * 1.5 * 0.2**2 * (-2.0) ** 2

    assert values[0] == pytest.approx(expected_compression)
    assert values[1] == 0.0


@pytest.mark.parametrize(
    ("strain_rate", "expected"),
    [
        (np.zeros(6), np.zeros(6)),
        (
            np.array([0.4, -0.4, 0.0, 0.0, 0.0, 0.0]),
            np.array([12.0, -12.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
    ids=["zero-rate", "yielded-extension"],
)
def test_bingham_normal_response_matches_ideal_plastic_viscosity(taichi_material_cpu, strain_rate, expected):
    model = BinghamModel()
    model.add_material(
        density=1000.0,
        modulus=2.0e5,
        viscosity=5.0,
        _yield=8.0,
        critical_rate=1.0e-4,
        atmospheric_pressure=0.0,
        gamma=7.0,
    )

    actual = _evaluate_shear_stress(model, strain_rate)

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_bingham_mixed_voigt_response_matches_cbgeo_shear_conversion(
    taichi_material_cpu,
):
    model = BinghamModel()
    model.add_material(
        density=1000.0,
        modulus=2.0e5,
        viscosity=5.0,
        _yield=8.0,
        critical_rate=1.0e-4,
        atmospheric_pressure=0.0,
        gamma=7.0,
    )

    # Convert the engineering-shear vector to tensorial shear before
    # evaluating the Bingham law.
    engineering_rate = np.array([0.4, -0.4, 0.0, 0.6, -0.2, 0.4], dtype=np.float64)
    tensor_rate = engineering_rate.copy()
    tensor_rate[3:] *= 0.5

    shear_rate = np.sqrt(2.0 * (np.dot(tensor_rate, tensor_rate) + np.dot(tensor_rate[3:], tensor_rate[3:])))
    apparent_viscosity = 2.0 * (model._yield / shear_rate + model.viscosity)
    expected = apparent_viscosity * tensor_rate
    assert 0.5 * np.dot(expected[:3], expected[:3]) >= model._yield**2

    actual = _evaluate_shear_stress(model, tensor_rate)

    # In GeoTaichi calculate_strain_rate() already returns tensor shear, and
    # voigt_tensor_dot() supplies the duplicate off-diagonal contraction.
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_bingham_core_combines_pressure_and_deviatoric_stress(
    taichi_material_cpu,
):
    model = BinghamModel()
    model.add_material(
        density=1000.0,
        modulus=2.0e5,
        viscosity=5.0,
        _yield=8.0,
        critical_rate=1.0e-4,
        atmospheric_pressure=0.0,
        gamma=7.0,
    )
    strain_rate = np.array([0.2, -0.5, -0.3, 0.0, 0.0, 0.0])

    stress, stored_pressure, _ = _evaluate_core(model, strain_rate, time_step=0.1)
    shear = _evaluate_shear_stress(model, strain_rate)
    expected_pressure = 12000.0
    expected = shear.copy()
    expected[:3] -= expected_pressure

    np.testing.assert_allclose(stress, expected, rtol=1.0e-12, atol=1.0e-10)
    assert stored_pressure == pytest.approx(-expected_pressure)


def test_bingham_model_initialize_preserves_gamma_and_atmospheric_pressure():
    model = BinghamModel()
    model.model_initialize(
        {
            "Density": 1000.0,
            "Modulus": 2.0e5,
            "Viscosity": 5.0,
            "YieldStress": 8.0,
            "CriticalStrainRate": 1.0e-4,
            "gamma": 7.0,
            "atmospheric_pressure": 101325.0,
        }
    )

    assert model.gamma == 7.0
    assert model.atmospheric_pressure == 101325.0


def test_bingham_pure_shear_matches_cbgeo_normal_invariant(taichi_material_cpu):
    model = BinghamModel()
    model.add_material(
        density=1000.0,
        modulus=2.0e5,
        viscosity=5.0,
        _yield=8.0,
        critical_rate=1.0e-4,
        atmospheric_pressure=0.0,
        gamma=7.0,
    )
    strain_rate = np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.0])
    actual = _evaluate_shear_stress(model, strain_rate)

    # The Bingham implementation performs the final yield check using only
    # tau[0:3], so a state with only shear components is cleared
    # even when its shear rate exceeds the critical rate.
    np.testing.assert_allclose(actual, 0.0, atol=0.0)


def test_bingham_configured_critical_rate_controls_activation(
    taichi_material_cpu,
):
    strain_rate = np.array([0.4, -0.4, 0.0, 0.0, 0.0, 0.0])
    model = BinghamModel()
    model.add_material(
        density=1000.0,
        modulus=2.0e5,
        viscosity=5.0,
        _yield=8.0,
        critical_rate=1.0,
        atmospheric_pressure=0.0,
        gamma=7.0,
    )
    np.testing.assert_allclose(_evaluate_shear_stress(model, strain_rate), 0.0, atol=0.0)

    model.critical_rate = 0.2
    np.testing.assert_allclose(
        _evaluate_shear_stress(model, strain_rate),
        np.array([12.0, -12.0, 0.0, 0.0, 0.0, 0.0]),
        atol=1.0e-12,
    )
