"""Paper section 4.3: Tresca strength and plastic-invariant normalization."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import EquivalentDeviatoricStrain
from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel

pytestmark = [pytest.mark.cpu, pytest.mark.materials]


def test_zero_friction_is_tresca_and_paper_softening_slope_is_preserved(taichi_material_cpu):
    model = MohrCoulombModel()
    model.model_initialize(
        {
            "Density": 2000.0,
            "YoungModulus": 1e6,
            "PoissonRatio": 0.33,
            "Friction": 0.0,
            "Dilation": 0.0,
            "Cohesion": 20000.0,
            "ResidualCohesion": 4000.0,
            "SoftType": "Linear",
            "PlasticDevStrain": 0.0,
            "ResidualPlasticDevStrain": 0.32 / np.sqrt(2.0),
        }
    )
    tensors = np.array(
        [
            [[-80000, 10000, 0], [10000, -40000, 0], [0, 0, -60000]],
            [[0, 30000, 0], [30000, 0, 0], [0, 0, 0]],
            [[-10000, 0, 0], [0, -10000, 0], [0, 0, -90000]],
        ],
        dtype=np.float64,
    )
    strain = np.array([0.06, -0.02, -0.04, 0.08, 0.01, -0.03])
    results = ti.field(ti.f64, shape=5)
    state = ti.Struct.field({"strain": ti.types.vector(6, ti.f64)}, shape=1)

    @ti.kernel
    def sample(data: ti.types.ndarray(), ep: ti.types.ndarray()):
        for i in range(3):
            stress = ti.Vector(
                [data[i, 0, 0], data[i, 1, 1], data[i, 2, 2], data[i, 0, 1], data[i, 1, 2], data[i, 0, 2]]
            )
            e, q, theta = model.ComputeStressInvariant(stress)
            results[i] = model.ComputeShearFunction(e, q, theta, 0.0, 20000.0)
        native = EquivalentDeviatoricStrain(ti.Vector([ep[j] for j in ti.static(range(6))]))
        results[3] = ti.sqrt(2.0) * native
        state[0].strain = ti.Vector([ep[j] for j in ti.static(range(6))])
        parameters = model.GetMaterialParameter(ti.Vector.zero(ti.f64, 6), state[0])
        results[4] = parameters[4]

    sample(tensors, strain)
    eigenvalues = np.linalg.eigvalsh(tensors)
    np.testing.assert_allclose(
        results.to_numpy()[:3], 0.5 * (eigenvalues[:, -1] - eigenvalues[:, 0]) - 20000, atol=1e-6
    )
    ep = np.array(
        [[strain[0], strain[3], strain[5]], [strain[3], strain[1], strain[4]], [strain[5], strain[4], strain[2]]]
    )
    ep -= np.eye(3) * np.trace(ep) / 3
    paper = np.sqrt(2 / 3 * np.sum(ep * ep))
    assert results[3] == pytest.approx(paper, rel=1e-7)
    assert results[4] == pytest.approx(20000 - 50000 * paper, rel=1e-7)
