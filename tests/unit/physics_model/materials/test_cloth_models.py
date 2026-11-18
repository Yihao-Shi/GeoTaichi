"""Cloth model ownership and shared validation contracts."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.constitutive_model.cloth_model.ClothARAP import ClothARAP
from src.physics_model.constitutive_model.cloth_model.ClothBending import ClothBending
from src.physics_model.constitutive_model.cloth_model.ClothNeoHookean import ClothNeoHookean
from tests.helpers.cloth_reference import cloth_response

pytestmark = [pytest.mark.unit, pytest.mark.materials, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


def make_evaluator(material):
    energy = ti.field(dtype=ti.f64, shape=())
    piola = ti.Matrix.field(3, 2, dtype=ti.f64, shape=())
    tangent = ti.Matrix.field(6, 6, dtype=ti.f64, shape=())
    stress = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate_kernel(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        deformation = ti.Matrix.zero(ti.f64, 3, 2)
        for row, column in ti.static(ti.ndrange(3, 2)):
            deformation[row, column] = values[row, column]
        energy[None] = material.strain_energy_density(deformation)
        piola[None] = material.first_piola_stress(deformation)
        tangent[None] = material.first_piola_tangent(deformation)
        stress[None] = material.cauchy_stress(deformation)

    def evaluate(deformation):
        evaluate_kernel(np.ascontiguousarray(deformation, dtype=np.float64))
        return (
            float(energy[None]),
            piola.to_numpy()[()],
            tangent.to_numpy()[()].reshape(2, 3, 2, 3).transpose(1, 0, 3, 2),
            stress.to_numpy()[()],
        )

    return evaluate


def test_fem_material_exports_resolve_to_implementation_classes():
    from src.fem import ClothARAP as FEMClothARAP, ClothNeoHookean as FEMClothNeoHookean

    assert FEMClothARAP is ClothARAP
    assert FEMClothNeoHookean is ClothNeoHookean


@pytest.mark.parametrize("name", ["None", "Quadratic", "Dihedral"])
def test_bending_model_canonical_names(name):
    assert ClothBending.validate_model(name) == name


@pytest.mark.parametrize("name", ["off", "cotangent", "ipc", "dihedral", "DihedralAngle", " Quadratic", None])
def test_bending_model_rejects_aliases_and_misspellings(name):
    with pytest.raises(ValueError, match="bending_model"):
        ClothBending.validate_model(name)


@pytest.mark.parametrize("name", ["Cloth", "ARAP", "ClothNH", "clotharap", "ClothNeoHookean "])
def test_cloth_material_factory_rejects_aliases_and_misspellings(name):
    from src.fem.MaterialManager import FEMMaterialManager

    with pytest.raises(ValueError, match="material"):
        FEMMaterialManager().material_handle(name)


@pytest.mark.parametrize("model,args", [(ClothARAP, (1.0e3,)), (ClothNeoHookean, (1.0e3, 0.3))])
def test_cloth_shared_validation_and_bending_modulus(model, args):
    material = model(*args)
    assert material.quadratic_bending_modulus == 0.0
    with pytest.raises(ValueError, match="bending_stiffness"):
        model(*args, bending_stiffness=-1.0)
    with pytest.raises(ValueError, match="bending_poisson_ratio"):
        model(*args, bending_poisson_ratio=1.0)
    with pytest.raises(ValueError, match="bending_model"):
        model(*args, bending_model="invalid")

    material = model(*args, bending_stiffness=12.0, thickness=0.2, bending_poisson_ratio=0.3, bending_model="Dihedral")
    assert material.bending_model == "Dihedral"
    assert material.quadratic_bending_modulus == pytest.approx(12.0 * 0.2**3 / (24.0 * (1.0 - 0.3**2)))


@pytest.mark.parametrize(
    "material",
    [ClothARAP(2.0e4, compression_stiffness=3.0e4), ClothNeoHookean(2.0e4, 0.27)],
)
def test_device_cloth_response_matches_independent_numpy_reference_and_directional_derivative(material):
    evaluate = make_evaluator(material)
    deformation = np.asarray(((1.1, 0.1), (0.0, 0.9), (0.05, 0.0)))
    for value in (deformation, np.asarray(((1.1, 0.0), (0.0, 1.1), (0.0, 0.0)))):
        energy, piola, tangent, stress = evaluate(value)
        reference_energy, reference_piola, reference_tangent = cloth_response(material, value)
        assert energy == pytest.approx(reference_energy, rel=2.0e-12, abs=1.0e-10)
        np.testing.assert_allclose(piola, reference_piola, rtol=2.0e-12, atol=1.0e-9)
        np.testing.assert_allclose(tangent, reference_tangent, rtol=2.0e-12, atol=1.0e-9)
        jacobian = np.sqrt(np.linalg.det(value.T @ value))
        np.testing.assert_allclose(stress, reference_piola @ value.T / jacobian, rtol=2.0e-12, atol=1.0e-9)

    _, _, tangent, _ = evaluate(deformation)
    direction = np.random.default_rng(18).normal(size=deformation.shape)
    epsilon = 1.0e-7
    plus = evaluate(deformation + epsilon * direction)[1]
    minus = evaluate(deformation - epsilon * direction)[1]
    np.testing.assert_allclose(
        np.einsum("iJkL,kL->iJ", tangent, direction),
        (plus - minus) / (2.0 * epsilon),
        rtol=2.0e-6,
        atol=2.0e-4,
    )


@pytest.mark.parametrize("material", [ClothARAP(1.0e4), ClothNeoHookean(1.0e4, 0.3)])
def test_device_cloth_energy_and_stress_are_zero_for_a_rigid_surface_map(material):
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0), (0.0, 0.0)))
    energy, piola, _, stress = make_evaluator(material)(rotation)
    assert energy == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(piola, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(stress, 0.0, atol=1.0e-12)
