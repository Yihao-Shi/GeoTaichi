"""Spectral reconstruction must not introduce a constitutive force floor."""

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.materials, pytest.mark.serial]


def test_hencky_trial_basis_and_elastic_stress_match_double_precision_reference(taichi_runtime):
    from src.physics_model.consititutive_model.finite_strain.DruckerPrager import FiniteStrainDruckerPragerModel

    model = FiniteStrainDruckerPragerModel()
    model.model_initialize(
        {
            "Density": 1600.0,
            "YoungModulus": 6e7,
            "PoissonRatio": 0.3,
            "FrictionAngle": 30.0,
            "DilationAngle": 0.0,
            "Cohesion": 1e12,
            "dpType": "Circumscribed",
        }
    )
    model.allocate_state(1)
    rng = np.random.default_rng(4)
    values = np.tile(np.eye(3), (102, 1, 1))
    values[:100, :2, :2] += rng.normal(0.0, 0.2, (100, 2, 2))
    values[100] *= 0.997
    values[101] = 0.997 * np.eye(3) + rng.normal(0.0, 1e-8, (3, 3))
    deformation = ti.Matrix.field(3, 3, ti.f64, shape=len(values))
    reconstructed = ti.Matrix.field(3, 3, ti.f64, shape=len(values))
    orthogonal = ti.Matrix.field(3, 3, ti.f64, shape=len(values))
    stress = ti.Matrix.field(3, 3, ti.f64, shape=len(values))
    deformation.from_numpy(values)

    @ti.kernel
    def evaluate():
        for i in deformation:
            left, right, singular, strain, trace, dev, norm = model._principal_trial_state(deformation[i])
            diagonal = ti.Matrix.zero(ti.f64, 3, 3)
            for j in ti.static(range(3)):
                diagonal[j, j] = singular[j]
            reconstructed[i] = left @ diagonal @ right.transpose()
            orthogonal[i] = left.transpose() @ left
            stress[i] = model.first_piola_stress_at(0, deformation[i])

    evaluate()
    np.testing.assert_allclose(reconstructed.to_numpy(), values, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(orthogonal.to_numpy(), np.tile(np.eye(3), (len(values), 1, 1)), rtol=0.0, atol=3e-15)
    left, singular, right = np.linalg.svd(values)
    strain = np.log(singular)
    trace = np.log(np.linalg.det(values))
    principal_stress = 2 * model.shear * (strain - trace[:, None] / 3) + model.bulk * trace[:, None]
    reference = (left * (principal_stress / singular)[:, None, :]) @ right
    np.testing.assert_allclose(stress.to_numpy(), reference, rtol=0.0, atol=2e-7)
