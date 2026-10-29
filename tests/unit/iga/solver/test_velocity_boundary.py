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


@pytest.mark.parametrize("assembly", ["HashTriplet", "COO"])
def test_nonzero_boundary_increment_can_do_work(taichi_runtime, tmp_path, assembly):
    from src.iga import DirichletBoundary, ImplicitIGA, Primitives, Rectangle, set_dimension

    set_dimension(2)
    square = Rectangle()
    square.set_parameters(size=[1.0, 1.0])
    square.generate_knot_u(degree=1, num_ctrlpts=3)
    square.generate_knot_v(degree=1, num_ctrlpts=3)
    square.generate_ctrlpts()
    square.generate_weights()
    reference = square.control_points.copy()
    bottom = np.flatnonzero(np.isclose(reference[:, 1], 0.0))
    top = np.flatnonzero(np.isclose(reference[:, 1], 1.0))
    fixed = np.concatenate((bottom, top))
    dofs = (2 * fixed[:, None] + np.arange(2)).ravel().astype(np.int32)
    top_dofs = (2 * top[:, None] + np.arange(2)).ravel().astype(np.int32)
    boundary = DirichletBoundary()
    boundary.append([dofs.tolist()], np.zeros(dofs.size).tolist())
    primitives = Primitives()
    primitives.append(square)
    primitives.finialize()
    engine = ImplicitIGA(
        primitives,
        dirichlet=boundary,
        degree=[1, 1],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0],
        dt=0.01,
        newmark=[1.0, 0.5, 1.0],
        residual=1.0e-7,
        assemble_type=assembly,
        path=str(tmp_path),
    )
    engine.precompute()
    total = 0.0
    for increment in (0.01, -0.005):
        boundary.set_constraints(top_dofs, np.tile([increment, 0.0], len(top)))
        engine.substep(verbose=False)
        total += increment
        current = engine.patch.control_points.to_numpy()
        np.testing.assert_allclose(current[bottom], reference[bottom], atol=1e-12)
        np.testing.assert_allclose(current[top], reference[top] + [total, 0], atol=1e-12)
        assert np.all(np.isfinite(current))
        assert engine.history[-1]["converged"]
    interior = np.setdiff1d(np.arange(len(reference)), fixed)
    assert np.max(np.abs(current[interior] - reference[interior])) > 1e-6
