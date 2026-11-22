from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpdem.engines.DirectAffineIPCSystem import DirectAffineIPCSystem
from src.mpdem.engines.DifferentiableMPMAffine import DifferentiableMPMAffine
from src.utils.errors import SimulationStateError


@ti.data_oriented
class Affine:
    def __init__(self):
        self.y = ti.Vector.field(3, ti.f64, shape=1)
        self.direction_y = ti.Vector.field(3, ti.f64, shape=1)
        self.base = ti.Vector.field(3, ti.f64, shape=1)

    @ti.kernel
    def device_backup_line_search_base(self):
        self.base[0] = self.y[0]

    @ti.kernel
    def device_set_line_search_trial(self, alpha: ti.f64):
        self.y[0] = self.base[0] + alpha * self.direction_y[0]


@ti.data_oriented
class MPM:
    def __init__(self):
        self.degree_of_freedom = 3
        self.grid_disp = ti.field(ti.f64, shape=3)
        self.grid_disp_temp = ti.field(ti.f64, shape=3)
        self.incre_resolution = ti.field(ti.f64, shape=3)

    @ti.kernel
    def update_grid_disp(self, active: ti.i32, alpha: ti.f64):
        for dof in range(active):
            self.grid_disp_temp[dof] = self.grid_disp[dof] + alpha * self.incre_resolution[dof]


@pytest.mark.parametrize("limit", [0.1, 1.0])
def test_relaxed_iterate_uses_ccd_and_keeps_the_next_reference(taichi_runtime, limit):
    system = object.__new__(DirectAffineIPCSystem)
    system.affine, system.mpm = Affine(), MPM()
    system.affine_controls = 1
    system.friction_relaxation = 0.8
    system.correction = ti.field(ti.f64, shape=6)
    system.relaxation_position = ti.Vector.field(3, ti.f64, shape=1)
    system.relaxation_displacement = ti.field(ti.f64, shape=3)
    old_y, old_u = np.array([[2.0, 3.0, 4.0]]), np.array([1.0, 2.0, 3.0])
    new_y, new_u = np.array([[4.0, 7.0, 8.0]]), np.array([3.0, 4.0, 5.0])
    system.affine.y.from_numpy(old_y)
    system.mpm.grid_disp.from_numpy(old_u)
    system._store_relaxation_state(3)
    system.affine.y.from_numpy(new_y)
    system.mpm.grid_disp.from_numpy(new_u)
    checked = []
    system.maximum_step_device = lambda **kwargs: checked.append(kwargs) or limit
    sims = SimpleNamespace(
        affine_ccd_type="accd",
        affine_ccd_eta=0.2,
        affine_accd_tolerance=1e-7,
        affine_ccd_max_iteration=100,
    )
    system._relax_friction_iterate_device(sims, 3)
    fraction = min(0.2, limit)
    np.testing.assert_allclose(system.affine.y.to_numpy(), new_y + fraction * (old_y - new_y))
    np.testing.assert_allclose(system.mpm.grid_disp.to_numpy(), new_u + fraction * (old_u - new_u))
    np.testing.assert_array_equal(system.relaxation_position.to_numpy(), system.affine.y.to_numpy())
    np.testing.assert_array_equal(system.relaxation_displacement.to_numpy(), system.mpm.grid_disp.to_numpy())
    assert checked[0]["active_mpm_dof"] == 3
    before = system.affine.y.to_numpy().copy()
    system.maximum_step_device = lambda **_: 0.0
    with pytest.raises(SimulationStateError, match="CCD"):
        system._relax_friction_iterate_device(sims, 3)
    np.testing.assert_array_equal(system.affine.y.to_numpy(), before)


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_friction_relaxation_rejects_invalid_weights(weight):
    mpm = SimpleNamespace(dimension=3, material=SimpleNamespace(), configure_nonassociated_newton=lambda: True)
    with pytest.raises(ValueError, match="friction_relaxation"):
        DirectAffineIPCSystem(SimpleNamespace(), SimpleNamespace(mpm=mpm), None, friction_relaxation=weight)


@pytest.mark.parametrize("record,requested", [(True, -1), (False, 3)])
def test_relaxed_friction_rejects_unsupported_recording_and_fixed_counts(record, requested):
    system = object.__new__(DirectAffineIPCSystem)
    system.friction_relaxation = 0.8
    with pytest.raises(ValueError, match="converged forward"):
        system._solve_nonassociated_equilibrium_device(
            None, max_outer=1, requested_outer=requested, record_adjoint=record
        )
    with pytest.raises(ValueError, match="friction_relaxation"):
        DifferentiableMPMAffine(SimpleNamespace(enginer=system), 1)
