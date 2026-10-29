"""Sanisand branch selection preserves the elastic/plastic handoff and history."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.SanisandMS import SanisandMSModel

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


@ti.data_oriented
class _HandoffProbe(SanisandMSModel):
    def __init__(self):
        super().__init__()
        self.rk_calls = ti.field(ti.i32, shape=())
        self.rk_stress = ti.Vector.field(6, ti.f64, shape=())
        self.rk_strain = ti.Vector.field(6, ti.f64, shape=())

    @ti.func
    def GetF(self, stress, alpha):
        return ti.abs(stress[3]) - 1.0

    @ti.func
    def GetElasticModuli(self, sigma, en, ElastFlag):
        return 1.0, 2.0

    @ti.func
    def GetNormalToYield(self, stress, alpha):
        return ti.Vector([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    @ti.func
    def RungeKutta4(self, stress, strain, void_ratio, alpha, alpha_m, mm_plus, mm_minus, alpha_in):
        self.rk_calls[None] += 1
        self.rk_stress[None] = stress
        self.rk_strain[None] = strain
        return stress + 2.0 * strain, alpha + alpha_in, alpha_m - alpha_in, mm_plus + 0.3, mm_minus - 0.4, void_ratio


@pytest.mark.parametrize(
    "shear,increment,plastic_start,plastic_increment",
    [
        pytest.param(0.2, 0.1, None, None, id="elastic"),
        pytest.param(2.0, 0.1, 2.0, 0.1, id="outside-yield-surface"),
        pytest.param(0.0, 2.0, 1.0, 1.0, id="elastic-to-plastic"),
        pytest.param(1.0, 2.0, 1.0, 2.0, id="plastic-loading"),
        pytest.param(1.0, -3.0, -1.0, -1.0, id="unloading-and-reverse-loading"),
    ],
)
def test_sanisand_elastic_plastic_handoff(taichi_material_cpu, shear, increment, plastic_start, plastic_increment):
    model = _HandoffProbe()
    stress = ti.Vector.field(6, ti.f64, shape=())
    alpha = ti.Vector.field(6, ti.f64, shape=())
    alpha_m = ti.Vector.field(6, ti.f64, shape=())
    memory = ti.Vector.field(2, ti.f64, shape=())

    @ti.kernel
    def advance():
        stress[None], alpha[None], alpha_m[None], memory[None][0], memory[None][1] = model.explicit_integrator(
            ti.Vector([100.0, 100.0, 100.0, shear, 0.0, 0.0]),
            ti.Vector([0.0, 0.0, 0.0, increment, 0.0, 0.0]),
            0.7,
            ti.Vector([0.1] * 6),
            ti.Vector([0.2] * 6),
            0.5,
            0.05,
            ti.Vector([0.01] * 6),
            1.0,
            2.0,
        )

    advance()
    if plastic_start is None:
        assert model.rk_calls[None] == 0
        np.testing.assert_allclose(stress[None], [100.0, 100.0, 100.0, shear + increment, 0.0, 0.0])
        np.testing.assert_allclose(alpha[None], 0.1)
        np.testing.assert_allclose(alpha_m[None], 0.2)
        np.testing.assert_allclose(memory[None], [0.5, 0.05])
    else:
        assert model.rk_calls[None] == 1
        np.testing.assert_allclose(model.rk_stress[None], [100.0, 100.0, 100.0, plastic_start, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(model.rk_strain[None], [0.0, 0.0, 0.0, plastic_increment, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            stress[None], [100.0, 100.0, 100.0, plastic_start + 2.0 * plastic_increment, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(alpha[None], 0.11)
        np.testing.assert_allclose(alpha_m[None], 0.19)
        np.testing.assert_allclose(memory[None], [0.8, -0.35])
