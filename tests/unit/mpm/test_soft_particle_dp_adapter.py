import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.mpm.MaterialManager import SoftParticleSingleMaterialAdapter
from src.physics_model.consititutive_model.finite_strain.DruckerPrager import (
    DruckerPragerModel,
)

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.cpu, pytest.mark.serial]


def test_soft_mpm_and_levelset_grids_have_independent_capacities():
    from src.mpm.Simulation import Simulation

    sims = Simulation.__new__(Simulation)
    sims.scheme = "LSMPM"
    sims.max_soft_body_num = 1
    sims.max_material_point_num = 8
    sims.max_soft_grid_num = 27
    sims.max_level_grid_num = 64

    sims.validate_soft_particle_configuration(require_memory=True)


def test_soft_affine_dp_uses_particle_history_and_commits_plastic_state():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    model = DruckerPragerModel()
    model.model_initialize(
        {
            "MaterialID": 0,
            "Density": 1800.0,
            "YoungModulus": 1.0e5,
            "PoissonRatio": 0.3,
            "FrictionAngle": 30.0,
            "DilationAngle": 30.0,
            "Cohesion": 100.0,
            "dpType": "Circumscribed",
        }
    )
    model.allocate_state(1)
    adapter = SoftParticleSingleMaterialAdapter(model)
    state = ti.Struct.field({"estress": ti.f64}, shape=1)
    trial_field = ti.Matrix.field(3, 3, ti.f64, shape=())
    committed_field = ti.Matrix.field(3, 3, ti.f64, shape=())
    stress_field = ti.Matrix.field(3, 3, ti.f64, shape=())
    energy = ti.field(ti.f64, shape=())
    tangent_norm = ti.field(ti.f64, shape=())

    @ti.kernel
    def evaluate():
        initial = ti.Matrix.identity(ti.f64, 3)
        displacement_gradient = ti.Matrix.zero(ti.f64, 3, 3)
        displacement_gradient[0, 1] = 0.15
        trial = adapter.trial_deformation_gradient(0, 0, initial, displacement_gradient)
        trial_field[None] = trial
        energy[None] = adapter.Psi_at(0, 0, trial)
        tangent = adapter.d2Psi_div_d2F_at(0, 0, trial)
        tangent_norm[None] = tangent.norm()
        committed, stress = adapter.commit_soft_particle_state(0, 0, trial, state)
        committed_field[None] = committed
        stress_field[None] = stress

    evaluate()

    expected = np.eye(3)
    expected[0, 1] = 0.15
    np.testing.assert_allclose(trial_field[None], expected, rtol=0.0, atol=1.0e-14)
    assert np.isfinite(float(energy[None]))
    assert float(energy[None]) > 0.0
    assert np.isfinite(float(tangent_norm[None]))
    assert float(tangent_norm[None]) > 0.0
    assert np.isfinite(stress_field[None]).all()
    assert np.linalg.det(committed_field[None]) > 0.0
    np.testing.assert_allclose(committed_field[None], trial_field[None], atol=1.0e-14)
    assert not np.allclose(model.plastic_deformation_inverse.to_numpy()[0], np.eye(3))
    assert float(model.equivalent_plastic_strain[0]) > 0.0
    assert float(state.estress[0]) > 0.0

    ti.reset()
