"""The native velocity gradient must rotate Cauchy stress with W sigma - sigma W."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import (
    Sigrot,
    calculate_vorticity_increment,
    calculate_vorticity_increment2D,
)
from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel
from src.physics_model.consititutive_model.infinitesimal_strain.NorSandModel import NorSandModel
from src.physics_model.consititutive_model.infinitesimal_strain.LinearElastic import LinearElasticModel
from src.utils.VectorFunction import outer_product, outer_product2D

pytestmark = [pytest.mark.materials, pytest.mark.cpu]

STRESS = np.array([[-200.0, 10.0, 30.0], [10.0, -300.0, 20.0], [30.0, 20.0, -400.0]])
VOIGT = ((0, 1, 2, 0, 1, 0), (0, 1, 2, 1, 2, 2))


@ti.dataclass
class _ElasticState:
    estress: ti.f64


@pytest.mark.parametrize("dimension,axis", [(2, 0), (3, 0), (3, 1), (3, 2)])
def test_material_entry_follows_a_rigidly_rotating_stress(taichi_material_cpu, monkeypatch, dimension, axis):
    monkeypatch.setattr("src.utils.GlobalVariable.RANDOMFIELD", False)
    model = LinearElasticModel()
    model.add_material(1000.0, 1e5, 0.3)
    state = _ElasticState.field(shape=1)
    out = ti.Vector.field(6, ti.f64, shape=())
    dt = ti.field(ti.f64, shape=())
    angle = 1e-6
    dt[None] = angle
    i, j = [(0, 1), (1, 2), (0, 2)][axis]
    w = np.zeros((3, 3))
    w[i, j], w[j, i] = -1.0, 1.0

    @ti.kernel
    def evaluate():
        sigma = ti.Vector([STRESS[i, j] for i, j in zip(*VOIGT)])
        gradient = ti.Matrix.zero(ti.f64, dimension, dimension)
        for k in ti.static(range(dimension)):
            basis = ti.Vector([float(d == k) for d in range(dimension)])
            velocity = ti.Matrix(w[:dimension, :dimension]) @ basis
            if ti.static(dimension == 2):
                gradient += outer_product2D(velocity, basis)
            else:
                gradient += outer_product(velocity, basis)
        if ti.static(dimension == 2):
            out[None] = model.ComputeStress2D(0, sigma, gradient, state, dt)
        else:
            out[None] = model.ComputeStress(0, sigma, gradient, state, dt)

    evaluate()
    rotation = np.eye(3)
    rotation[i, i] = rotation[j, j] = np.cos(angle)
    rotation[i, j], rotation[j, i] = -np.sin(angle), np.sin(angle)
    expected = rotation @ STRESS @ rotation.T
    # The production update is first order; its finite-rotation error is O(angle**2).
    np.testing.assert_allclose(out.to_numpy(), expected[VOIGT], rtol=0, atol=3e-10)


@pytest.mark.parametrize("dimension,axis", [(2, 0), (3, 0), (3, 1), (3, 2)])
def test_spin_matches_rigid_rotation_commutator(taichi_material_cpu, dimension, axis):
    w = np.zeros((3, 3))
    i, j = [(0, 1), (1, 2), (0, 2)][axis]
    w[i, j], w[j, i] = -1.0, 1.0
    dt = ti.field(ti.f64, shape=())
    dt[None] = 0.001
    out = ti.Vector.field(6, ti.f64, shape=())

    @ti.kernel
    def evaluate():
        sigma = ti.Vector([STRESS[i, j] for i, j in zip(*VOIGT)])
        if ti.static(dimension == 2):
            gradient = ti.Matrix(w[:2, :2])
            spin = calculate_vorticity_increment2D(gradient, dt)
            out[None] = Sigrot(sigma, spin)
        else:
            gradient = ti.Matrix(w)
            spin = calculate_vorticity_increment(gradient, dt)
            out[None] = Sigrot(sigma, spin)

    evaluate()
    expected = 0.001 * (w @ STRESS - STRESS @ w)
    np.testing.assert_allclose(out.to_numpy(), expected[VOIGT], atol=1e-12)


@pytest.mark.parametrize("model_class", [MohrCoulombModel, NorSandModel])
def test_material_rotation_uses_the_objective_direction(taichi_material_cpu, model_class):
    model = model_class()
    out = ti.Vector.field(6, ti.f64, shape=())

    @ti.kernel
    def evaluate():
        sigma = ti.Vector([STRESS[i, j] for i, j in zip(*VOIGT)])
        out[None] = model.ComputeSigrotStress(ti.Vector([0.001, 0.0, 0.0]), sigma)

    evaluate()
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    expected = 0.001 * (w @ STRESS - STRESS @ w)
    np.testing.assert_allclose(out.to_numpy(), expected[VOIGT], atol=1e-12)
