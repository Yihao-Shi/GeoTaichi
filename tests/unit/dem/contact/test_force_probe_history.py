"""A Verlet force probe must not commit friction or rolling return maps."""

import numpy as np
import pytest
import taichi as ti

from src.utils import GlobalVariable
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty
from src.physics_model.contact_model.HertzMindlinModel import HertzMindlinSurfaceProperty
from src.physics_model.contact_model.EnergyConservingModel import PenaltyProperty
from src.physics_model.contact_model.RollingModel import JiangRollingSurfaceProperty
from src.physics_model.contact_model.LinearRollingModel import LinearRollingSurfaceProperty
from src.physics_model.contact_model.BarrierModel import BarrierProperty

pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.cpu]


@pytest.mark.parametrize("model", ["linear", "hertz", "penalty", "rolling", "linear_rolling", "barrier"])
def test_zero_duration_probe_preserves_history_and_dissipation(taichi_runtime, monkeypatch, model):
    monkeypatch.setattr(GlobalVariable, "TRACKENERGY", True)
    monkeypatch.setattr(GlobalVariable, "ADAPTIVESTIFF", False)
    types = dict(
        linear=LinearSurfaceProperty,
        hertz=HertzMindlinSurfaceProperty,
        penalty=PenaltyProperty,
        rolling=JiangRollingSurfaceProperty,
        linear_rolling=LinearRollingSurfaceProperty,
        barrier=BarrierProperty,
    )
    prop = types[model].field(shape=1)
    if model == "linear":
        prop[0].add_surface_property(100.0, 100.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0)
    elif model == "hertz":
        prop[0].add_surface_property(1000.0, 1000.0, 0.3, 0.3, 0.0, 0.0)
    elif model == "penalty":
        prop[0].add_surface_property(100.0, 100.0, 2.0, 0.3, 0.0, 0.0)
    elif model == "barrier":
        prop[0].add_surface_property(100.0, 0.02, 1.0, 0.3, 0.0, 0.0)
    elif model == "linear_rolling":
        prop[0].add_surface_property(100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0)
    else:
        prop[0].add_surface_property(100.0, 1.0, 0.3, 1.0, 0.3, 0.0, 0.0)
    dt = ti.field(float, shape=())
    history = ti.Vector.field(3, float, shape=3)
    forces = ti.Vector.field(3, float, shape=2)

    @ti.kernel
    def evaluate():
        normal, zero = ti.Vector([0.0, 0.0, 1.0]), ti.Vector([0.0, 0.0, 0.0])
        fn, ft = zero, zero
        if ti.static(model == "penalty" or model == "barrier"):
            fn, ft, h = prop[0]._force_assemble(1.0, 1.0, -0.01, 1.0, normal, zero, history[0], dt)
            history[0] = h
        elif ti.static(model == "rolling" or model == "linear_rolling"):
            fn, ft, _, h, r, t = prop[0]._force_assemble(
                1.0, 1.0, -0.01, 1.0, 1.0, normal, zero, zero, zero, history[0], history[1], history[2], dt
            )
            history[0], history[1], history[2] = h, r, t
        else:
            fn, ft, _, h = prop[0]._force_assemble(1.0, 1.0, -0.01, 1.0, 1.0, normal, zero, zero, history[0], dt)
            history[0] = h
        forces[0], forces[1] = fn, ft

    old = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]])
    history.from_numpy(old)
    dt[None] = 0.0
    evaluate()
    np.testing.assert_array_equal(history.to_numpy(), old)
    assert prop[0].friction_energy == prop[0].damp_energy == 0.0
    probe_force = forces.to_numpy()
    dt[None] = 0.01
    evaluate()
    # At zero relative speed the same force is returned, but only the actual
    # step commits slip caused by a reduced Coulomb limit.
    np.testing.assert_allclose(forces.to_numpy(), probe_force, atol=1e-13, rtol=0)
    assert np.linalg.norm(history[0]) < np.linalg.norm(old[0])
    assert prop[0].friction_energy < 0.0
