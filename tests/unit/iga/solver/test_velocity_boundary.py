"""A retried prescribed motion must retain velocity, not the original increment."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.iga, pytest.mark.isolated_dimension(2)]


def test_prescribed_velocity_tracks_coupled_timestep(taichi_runtime):
    import src.iga.config as config

    config.set_dimension(2)
    from src.iga import DirichletBoundary, ImplicitIGA
    from src.iga.engines.ExplicitIGA import ExplicitIGA
    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin

    boundary = DirichletBoundary()
    boundary.append([[0]], [0.002])
    boundary.append_velocity([[1]], [-0.1])
    boundary.finalize(2)
    engine = object.__new__(ImplicitIGA)
    engine.TIdt = ti.field(ti.f64, shape=())
    engine.dirichlet = boundary
    engine.grid_disp = ti.field(ti.f64, shape=2)
    engine.patch = SimpleNamespace(
        **{name: ti.Vector.field(2, ti.f64, shape=1) for name in ("control_points", "velocitys", "accelerations")}
    )
    engine.patch.velocitys.from_numpy(np.array([[0.0, -0.1]]))
    coupled = SimpleNamespace(iga=engine, mpm=SimpleNamespace(dt=0))
    elapsed = 0.0
    for dt in (0.001, 0.0005, 0.000125, 0.001):
        ImplicitEngineMixin._set_implicit_timestep(coupled, dt)
        values = boundary.value.to_numpy()
        np.testing.assert_allclose(values, [0.002, -0.1 * dt], atol=1e-16)
        assert coupled.mpm.dt == pytest.approx(dt)
        engine.grid_disp.from_numpy(values)
        engine.dynamic_advance(1, 0, [1.0, 0.5, 1.0])
        elapsed += dt
        assert engine.patch.control_points.to_numpy()[0, 1] == pytest.approx(-0.1 * elapsed, abs=1e-15)
        assert engine.patch.velocitys.to_numpy()[0, 1] == pytest.approx(-0.1, abs=1e-15)
        assert engine.patch.accelerations.to_numpy()[0, 1] == pytest.approx(0, abs=1e-11)
    with pytest.raises(ValueError, match="implicit IGA only"):
        ExplicitIGA(None, dirichlet=boundary)
    with pytest.raises(ValueError, match="unique"):
        boundary.append_velocity([[1]], [-0.1])
    with pytest.raises(ValueError, match="finite"):
        DirichletBoundary().append_velocity([[1]], [np.nan])
