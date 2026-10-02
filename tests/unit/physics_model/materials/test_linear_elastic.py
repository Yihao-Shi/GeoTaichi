"""Unit tests for the infinitesimal-strain linear-elastic material."""

import numpy as np
import pytest
import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.consititutive_model.infinitesimal_strain.LinearElastic import (
    LinearElasticModel,
)


pytestmark = [pytest.mark.materials, pytest.mark.cpu]


@ti.dataclass
class _LinearElasticState:
    estress: ti.f64


def _isotropic_stiffness(bulk, shear):
    normal_diagonal = bulk + 4.0 * shear / 3.0
    normal_off_diagonal = bulk - 2.0 * shear / 3.0
    stiffness = np.zeros((6, 6), dtype=np.float64)
    stiffness[:3, :3] = normal_off_diagonal
    np.fill_diagonal(stiffness[:3, :3], normal_diagonal)
    np.fill_diagonal(stiffness[3:, 3:], shear)
    return stiffness


def _evaluate_model(model, strain_increment, previous_stress):
    state = _LinearElasticState.field(shape=1)
    stress = ti.Vector.field(6, dtype=ti.f64, shape=())
    stiffness = ti.Matrix.field(6, 6, dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate(
        strain: ti.types.ndarray(dtype=ti.f64, ndim=1),
        previous: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        de = ti.Vector.zero(ti.f64, 6)
        old_stress = ti.Vector.zero(ti.f64, 6)
        for component in ti.static(range(6)):
            de[component] = strain[component]
            old_stress[component] = previous[component]
        state[0].estress = 0.0
        stress[None] = model.cores(
            0,
            old_stress,
            de,
            ti.Vector.zero(ti.f64, 3),
            state,
        )
        stiffness[None] = model.compute_stiffness_tensor(
            0, stress[None], state
        )

    old_random_field = GlobalVariable.RANDOMFIELD
    GlobalVariable.RANDOMFIELD = False
    try:
        evaluate(
            np.asarray(strain_increment, dtype=np.float64),
            np.asarray(previous_stress, dtype=np.float64),
        )
    finally:
        GlobalVariable.RANDOMFIELD = old_random_field

    return (
        stress.to_numpy()[()],
        stiffness.to_numpy()[()],
        float(state.estress.to_numpy()[0]),
    )


@pytest.mark.parametrize(
    ("density", "young", "poisson"),
    [
        (1000.0, 1.0e5, 0.0),
        (1800.0, 2.5e6, 0.25),
        (2650.0, 7.0e7, 0.45),
    ],
)
def test_add_material_derives_isotropic_moduli_and_sound_speed(
    density, young, poisson
):
    model = LinearElasticModel()
    model.add_material(density, young, poisson)

    expected_shear = young / (2.0 * (1.0 + poisson))
    expected_bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    expected_wave_speed = np.sqrt(
        young
        * (1.0 - poisson)
        / ((1.0 + poisson) * (1.0 - 2.0 * poisson) * density)
    )

    assert model.density == density
    assert model.young == young
    assert model.poisson == poisson
    assert model.shear == pytest.approx(expected_shear)
    assert model.bulk == pytest.approx(expected_bulk)
    assert model.max_sound_speed == pytest.approx(expected_wave_speed)


def test_model_initialize_applies_defaults_and_requires_young_modulus():
    model = LinearElasticModel()
    model.model_initialize({"YoungModulus": 3.0e6})

    assert model.density == 2650
    assert model.young == 3.0e6
    assert model.poisson == 0.3
    assert model.material == {"YoungModulus": 3.0e6}

    with pytest.raises(KeyError, match="YoungModulus"):
        LinearElasticModel().model_initialize({"Density": 1800.0})


@pytest.mark.parametrize("configuration", ["UL", "ULMPM"])
def test_state_schema_contains_only_equivalent_stress(configuration):
    model = LinearElasticModel(configuration=configuration)

    assert model.get_state_vars() == {"estress": float}
    assert model._initialize_vars == model._initialize_vars_update_lagrangian


@pytest.mark.parametrize(
    "strain_increment",
    [
        np.array([1.0e-3, 1.0e-3, 1.0e-3, 0.0, 0.0, 0.0]),
        np.array([8.0e-4, -4.0e-4, -4.0e-4, 0.0, 0.0, 0.0]),
        np.array([5.0e-4, -2.0e-4, 1.0e-4, 3.0e-4, -1.0e-4, 2.0e-4]),
    ],
    ids=["hydrostatic", "deviatoric", "general-with-shear"],
)
def test_stress_update_and_stiffness_match_isotropic_oracle(
    taichi_material_cpu, strain_increment
):
    model = LinearElasticModel()
    model.add_material(density=1800.0, young=2.0e5, poisson=0.3)
    previous_stress = np.array(
        [-1200.0, -1000.0, -900.0, 10.0, -5.0, 7.0],
        dtype=np.float64,
    )

    actual_stress, actual_stiffness, equivalent_stress = _evaluate_model(
        model, strain_increment, previous_stress
    )
    expected_stiffness = _isotropic_stiffness(model.bulk, model.shear)
    tensor_strain_operator = expected_stiffness.copy()
    tensor_strain_operator[3:, 3:] *= 2.0
    expected_stress = (
        previous_stress + tensor_strain_operator @ strain_increment
    )
    expected_equivalent = np.sqrt(
        0.5
        * (
            (expected_stress[0] - expected_stress[1]) ** 2
            + (expected_stress[1] - expected_stress[2]) ** 2
            + (expected_stress[0] - expected_stress[2]) ** 2
        )
    )

    np.testing.assert_allclose(
        actual_stress, expected_stress, rtol=5.0e-8, atol=1.0e-5
    )
    np.testing.assert_allclose(
        actual_stiffness,
        expected_stiffness,
        rtol=5.0e-8,
        atol=1.0e-5,
    )
    assert equivalent_stress == pytest.approx(
        expected_equivalent, rel=5.0e-8, abs=1.0e-5
    )
