import numpy as np
import pytest

from src.physics_model.consititutive_model.finite_strain import (
    ClothARAP,
    ClothNeoHookean,
)
from src.fem.MaterialManager import FEMMaterialManager


def create_material(model, **parameters):
    return FEMMaterialManager().material_handle(model, **parameters)


pytestmark = [pytest.mark.unit, pytest.mark.fem, pytest.mark.materials, pytest.mark.cpu]


@pytest.mark.parametrize(
    "material,deformation_gradient",
    [
        (create_material("StVK", young_modulus=2.0e4, poisson_ratio=0.27), np.asarray(((1.1, 0.1, 0.0), (0.0, 0.9, 0.1), (0.0, 0.0, 1.05)))),
        (create_material("StVK", young_modulus=2.0e4, poisson_ratio=0.27), np.asarray(((1.1, 0.1), (0.0, 0.9), (0.05, 0.0)))),
        (create_material("NeoHookean", young_modulus=2.0e4, poisson_ratio=0.27), np.asarray(((1.1, 0.1, 0.0), (0.0, 0.9, 0.1), (0.0, 0.0, 1.05)))),
        (ClothARAP(2.0e4), np.asarray(((1.1, 0.1), (0.0, 0.9), (0.05, 0.0)))),
        (ClothNeoHookean(2.0e4, 0.27), np.asarray(((1.1, 0.1), (0.0, 0.9), (0.05, 0.0)))),
    ],
)
def test_consistent_material_tangent_matches_directional_difference(material, deformation_gradient):
    rng = np.random.default_rng(18)
    increment = rng.normal(size=deformation_gradient.shape)
    _, _, tangent = material.evaluate(deformation_gradient)
    epsilon = 1.0e-7
    plus = material.evaluate(deformation_gradient + epsilon * increment, need_tangent=False)[1]
    minus = material.evaluate(deformation_gradient - epsilon * increment, need_tangent=False)[1]
    finite_difference = (plus - minus) / (2.0 * epsilon)
    analytical = np.einsum("iJkL,kL->iJ", tangent, increment)

    np.testing.assert_allclose(analytical, finite_difference, rtol=2.0e-6, atol=2.0e-4)


def test_cloth_factory_selects_surface_formulation():
    arap = create_material(
        "Cloth",
        stretch_stiffness=3.0e4,
        compression_stiffness=5.0e4,
        density=2.0,
        thickness=0.02,
        bending_stiffness=12.0,
        bending_poisson_ratio=0.3,
    )
    neo_hookean = create_material(
        "ClothNeoHookean",
        young_modulus=3.0e4,
        poisson_ratio=0.25,
        density=2.0,
        thickness=0.02,
    )

    assert arap.is_cloth and arap.cloth_model == "arap"
    assert arap.stretch_stiffness == 3.0e4
    assert arap.compression_stiffness == 5.0e4
    assert arap.quadratic_bending_modulus > 0.0
    assert neo_hookean.is_cloth and neo_hookean.cloth_model == "neo_hookean"


def test_cloth_energies_are_zero_for_a_rigid_surface_map():
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0), (0.0, 0.0)))

    for material in (ClothARAP(1.0e4), ClothNeoHookean(1.0e4, 0.3)):
        energy, first_piola, _ = material.evaluate(rotation, need_tangent=False)
        assert energy == pytest.approx(0.0, abs=1.0e-12)
        np.testing.assert_allclose(first_piola, 0.0, atol=1.0e-12)
