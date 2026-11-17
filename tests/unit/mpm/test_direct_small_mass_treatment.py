"""Small-mass removal retains complete, valid particle--node connections."""

import numpy as np
import pytest
import taichi as ti
from types import SimpleNamespace

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.serial]


@pytest.mark.parametrize("ratio", [-1.0, 1.0, float("nan"), float("inf")])
def test_invalid_cutoff_fails_before_solver_allocation(monkeypatch, ratio):
    from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    monkeypatch.setattr(ImplicitMPM, "__init__", lambda *args, **kwargs: pytest.fail("allocated before validation"))
    with pytest.raises(ValueError, match="small_mass_ratio"):
        ImplicitULMPM(None, small_mass_ratio=ratio)


@pytest.mark.parametrize(
    "direct,solver_type,configuration",
    [
        (False, "Implicit", "ULMPM"),
        (True, "Explicit", "ULMPM"),
        (True, "Implicit", "TLMPM"),
    ],
)
def test_public_solver_rejects_cutoff_on_unsupported_routes(direct, solver_type, configuration):
    from src.mpm.mainMPM import MPM

    mpm = object.__new__(MPM)
    mpm.sims = SimpleNamespace(is_direct_backend=lambda: direct, solver_type=solver_type, configuration=configuration)
    with pytest.raises(ValueError, match="Direct implicit ULMPM"):
        mpm.set_solver({"small_mass_ratio": 1e-5}, log=False)


@pytest.mark.parametrize(
    "dimension",
    [
        pytest.param(2, marks=pytest.mark.isolated_dimension(2)),
        pytest.param(3, marks=pytest.mark.isolated_dimension(3)),
    ],
)
@pytest.mark.parametrize("ratio", [0.0, 1e-5, 1e-4])
def test_cutoff_compacts_support_and_reports_exact_discarded_mass_and_momentum(taichi_runtime, dimension, ratio):
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    engine = object.__new__(ImplicitULMPM)
    engine.small_mass_ratio = ratio
    engine.last_small_mass_treatment = None
    engine.val_lim = 1e-12
    engine.total_background_grid_num = 4
    engine._mass_vel_acc_p2g = lambda: None
    grid = ti.types.struct(m=ti.f64, v=ti.types.vector(dimension, ti.f64), a=ti.types.vector(dimension, ti.f64))
    engine.grid = grid.field(shape=4)
    mass = np.array([5.0, 1e-7, 2.0, 1e-8])
    momentum = mass[:, None] * np.arange(1, 4 * dimension + 1).reshape(4, dimension)
    engine.grid.m.from_numpy(mass)
    engine.grid.v.from_numpy(momentum)
    engine.grid.a.from_numpy(momentum * 2)
    engine.particleNum = ti.field(ti.i32, shape=1)
    engine.particleNum[0] = 2
    engine.offset = ti.field(ti.i32, shape=2)
    engine.offset.from_numpy(np.array([3, 2], dtype=np.int32))
    engine.LnID = ti.field(ti.i32, shape=(2, 3))
    engine.LnID.from_numpy(np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32))
    engine.shape = ti.field(ti.f64, shape=(2, 3))
    weights = np.array([[0.3, 1e-8, 0.7 - 1e-8], [1 - 1e-8, 1e-8, 0]])
    engine.shape.from_numpy(weights)
    engine.dshape = ti.Vector.field(dimension, ti.f64, shape=(2, 3))
    gradients = np.arange(1, 6 * dimension + 1, dtype=np.float64).reshape(2, 3, dimension)
    engine.dshape.from_numpy(gradients)
    engine.invalid_stencil_particle = ti.field(ti.i32, shape=())

    engine.mass_vel_acc_p2g()
    if ratio == 0:
        np.testing.assert_array_equal(engine.grid.m.to_numpy(), mass)
        np.testing.assert_array_equal(engine.offset.to_numpy(), [3, 2])
        assert engine.last_small_mass_treatment is None
        return
    np.testing.assert_array_equal(engine.offset.to_numpy(), [2, 1])
    np.testing.assert_array_equal(engine.LnID.to_numpy()[0, :2], [0, 2])
    assert engine.LnID[1, 0] == 2
    np.testing.assert_array_equal(engine.shape.to_numpy()[0, :2], weights[0, [0, 2]])
    np.testing.assert_array_equal(engine.dshape.to_numpy()[0, :2], gradients[0, [0, 2]])
    np.testing.assert_array_equal(engine.grid.m.to_numpy(), [5, 0, 2, 0])
    np.testing.assert_array_equal(engine.grid.v.to_numpy()[[0, 2]], momentum[[0, 2]])
    np.testing.assert_array_equal(engine.grid.v.to_numpy()[[1, 3]], 0)
    np.testing.assert_array_equal(engine.grid.a.to_numpy()[[1, 3]], 0)
    stats = engine.last_small_mass_treatment
    assert stats["threshold"] == pytest.approx(ratio * mass.mean())
    assert stats["removed_nodes"] == 2
    assert stats["removed_mass"] == pytest.approx(mass[[1, 3]].sum())
    assert stats["removed_mass_fraction"] == pytest.approx(mass[[1, 3]].sum() / mass.sum())
    np.testing.assert_allclose(stats["removed_momentum"], momentum[[1, 3]].sum(axis=0), rtol=1e-14)


def test_cutoff_rejects_particles_without_any_remaining_support(taichi_runtime):
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    engine = object.__new__(ImplicitULMPM)
    engine.small_mass_ratio, engine.val_lim, engine.total_background_grid_num = 1e-4, 1e-12, 2
    engine._mass_vel_acc_p2g = lambda: None
    engine.grid = ti.types.struct(m=ti.f64).field(shape=2)
    engine.grid.m.from_numpy(np.array([1e-8, 1.0]))
    engine.particleNum = ti.field(ti.i32, shape=1)
    engine.particleNum[0] = 1
    engine.offset = ti.field(ti.i32, shape=1)
    engine.offset[0] = 1
    engine.LnID = ti.field(ti.i32, shape=(1, 1))
    engine.shape = ti.field(ti.f64, shape=(1, 1))
    engine.dshape = ti.Vector.field(2, ti.f64, shape=(1, 1))
    engine.invalid_stencil_particle = ti.field(ti.i32, shape=())
    with pytest.raises(ValueError, match="every support node"):
        engine.mass_vel_acc_p2g()
