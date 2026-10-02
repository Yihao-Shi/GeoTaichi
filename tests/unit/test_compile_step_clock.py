from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.mpm
def test_mpm_compile_counts_the_physical_step():
    from src.mpm.MPMBase import Solver

    core_calls = []
    solver = object.__new__(Solver)
    solver.sims = SimpleNamespace(
        current_time=1.25,
        current_step=10,
        delta=0.125,
        timer=SimpleNamespace(profile1=lambda: None),
    )
    solver.core = lambda scene, neighbor: core_calls.append((scene, neighbor))

    solver.compile("scene", "neighbor")

    assert core_calls == [("scene", "neighbor")]
    assert solver.sims.current_time == pytest.approx(1.375)
    assert solver.sims.current_step == 11


@pytest.mark.mpm
def test_mpm_compile_uses_timestep_that_was_integrated():
    from src.mpm.MPMBase import Solver

    solver = object.__new__(Solver)
    solver.sims = SimpleNamespace(
        current_time=0.0,
        current_step=0,
        delta=0.1,
        timer=SimpleNamespace(profile1=lambda: None),
    )
    solver.core = lambda scene, neighbor: setattr(solver.sims, "delta", 0.025)

    solver.compile(None, None)

    assert solver.sims.current_time == pytest.approx(0.1)


@pytest.mark.mpm
def test_mpm_loop_saves_post_step_state_without_overshoot(monkeypatch):
    import src.mpm.MPMBase as mpm_base

    solver = object.__new__(mpm_base.Solver)
    timer = SimpleNamespace(profile1=lambda: None)
    solver.sims = SimpleNamespace(
        current_time=0.0,
        current_step=0,
        current_print=0,
        delta=1.0,
        time=3.0,
        save_interval=1.0,
        timer=timer,
        set_timestep=lambda value: setattr(solver.sims, "delta", value),
    )
    solver.engine = SimpleNamespace(pre_calculation=lambda sims, scene, neighbor: None)
    solver.generator = SimpleNamespace(regenerate=lambda scene: False)
    solver.last_save_time = 0.0
    stepped_from = []
    saved = []
    solver.core = lambda scene, neighbor: stepped_from.append(solver.sims.current_time)

    def save_file(scene):
        saved.append(solver.sims.current_time)
        solver.last_save_time = solver.sims.current_time
        solver.sims.current_print += 1

    solver.save_file = save_file
    monkeypatch.setattr(mpm_base.ti, "sync", lambda: None)
    monkeypatch.setattr(mpm_base, "runtime_checkpoint", lambda: None)
    monkeypatch.setattr(mpm_base, "print_simulation_start", lambda name: None)

    solver.Solver(None, None)

    assert stepped_from == [0.0, 1.0, 2.0]
    assert saved == [0.0, 1.0, 2.0, 3.0]
    assert solver.sims.current_time == pytest.approx(3.0)
    assert solver.sims.current_step == 3


@pytest.mark.dem
def test_dem_compile_counts_the_physical_step(monkeypatch):
    import src.dem.DEMBase as dem_base

    core_calls = []
    solver = object.__new__(dem_base.Solver)
    solver.sims = SimpleNamespace(
        current_time=2.0,
        current_step=7,
        delta=0.05,
        timer=SimpleNamespace(profile1=lambda: None),
    )
    solver.core = lambda scene: core_calls.append(scene)
    monkeypatch.setattr(dem_base.ti, "sync", lambda: None)

    solver.compile("scene")

    assert core_calls == ["scene"]
    assert solver.sims.current_time == pytest.approx(2.05)
    assert solver.sims.current_step == 8


@pytest.mark.dem
def test_dem_compile_uses_timestep_that_was_integrated(monkeypatch):
    import src.dem.DEMBase as dem_base

    solver = object.__new__(dem_base.Solver)
    solver.sims = SimpleNamespace(
        current_time=0.0,
        current_step=0,
        delta=0.1,
        timer=SimpleNamespace(profile1=lambda: None),
    )
    solver.core = lambda scene: setattr(solver.sims, "delta", 0.025)
    monkeypatch.setattr(dem_base.ti, "sync", lambda: None)

    solver.compile(None)

    assert solver.sims.current_time == pytest.approx(0.1)


@pytest.mark.coupling
def test_dempm_compile_advances_every_coupled_clock(monkeypatch):
    import src.mpdem.DEMPMBase as dempm_base

    solver = object.__new__(dempm_base.Solver)
    solver.sims = SimpleNamespace(
        current_time=3.0,
        current_step=21,
        delta=0.02,
        timer=SimpleNamespace(profile1=lambda: None),
    )
    solver.msims = SimpleNamespace(current_time=3.0)
    solver.dsims = SimpleNamespace(current_time=3.0)
    core_calls = []
    solver.core = lambda: core_calls.append(True)
    monkeypatch.setattr(dempm_base.ti, "sync", lambda: None)

    solver.compile()

    assert core_calls == [True]
    assert solver.sims.current_time == pytest.approx(3.02)
    assert solver.msims.current_time == pytest.approx(3.02)
    assert solver.dsims.current_time == pytest.approx(3.02)
    assert solver.sims.current_step == 22


@pytest.mark.coupling
def test_dempm_compile_uses_timestep_that_was_integrated(monkeypatch):
    import src.mpdem.DEMPMBase as dempm_base

    solver = object.__new__(dempm_base.Solver)
    solver.sims = SimpleNamespace(
        current_time=0.0,
        current_step=0,
        delta=0.1,
        timer=SimpleNamespace(profile1=lambda: None),
    )
    solver.msims = SimpleNamespace(current_time=0.0)
    solver.dsims = SimpleNamespace(current_time=0.0)
    solver.core = lambda: setattr(solver.sims, "delta", 0.025)
    monkeypatch.setattr(dempm_base.ti, "sync", lambda: None)

    solver.compile()

    assert solver.sims.current_time == pytest.approx(0.1)
    assert solver.msims.current_time == pytest.approx(0.1)
    assert solver.dsims.current_time == pytest.approx(0.1)


@pytest.mark.parametrize("module_name", ("src.mpm.MPMBase", "src.dem.DEMBase", "src.mpdem.DEMPMBase"))
@pytest.mark.parametrize("target, dt, count", ((0.65, 2.0e-4, 3250), (0.12, 1.0e-5, 12000), (0.120003, 1.0e-5, 12001)))
@pytest.mark.parametrize("save_interval_fraction", (1.0, 0.85))
def test_loops_skip_roundoff_but_integrate_real_fractional_steps(
    monkeypatch, module_name, target, dt, count, save_interval_fraction
):
    import importlib

    base = importlib.import_module(module_name)
    solver = object.__new__(base.Solver)
    solver.sims = SimpleNamespace(
        coupling_scheme="CFDEM",
        current_time=0.0,
        current_step=0,
        current_print=0,
        delta=dt,
        time=target,
        save_interval=save_interval_fraction * target,
        timer=SimpleNamespace(profile1=lambda: None),
        set_timestep=lambda value: setattr(solver.sims, "delta", value),
    )
    solver.msims = SimpleNamespace(
        current_time=0.0,
        set_timestep=lambda value: setattr(solver.msims, "delta", value),
    )
    solver.dsims = SimpleNamespace(
        current_time=0.0,
        set_timestep=lambda value: setattr(solver.dsims, "delta", value),
    )
    solver.engine = SimpleNamespace(
        pre_calculate=lambda: None, pre_calculation=lambda *args: None, reset_message=lambda: None
    )
    solver.contact = SimpleNamespace(neighbor=None)
    solver.generator = SimpleNamespace(regenerate=lambda *args: False)
    solver.last_save_time = 0.0
    saved = []

    def save_file(*args):
        saved.append(solver.sims.current_time)
        solver.last_save_time = solver.sims.current_time

    solver.save_file = save_file
    calls = []
    solver.core = lambda *args: calls.append(solver.sims.delta)
    monkeypatch.setattr(base.ti, "sync", lambda: None)
    monkeypatch.setattr(base, "runtime_checkpoint", lambda: None)
    monkeypatch.setattr(base, "print_simulation_start", lambda name: None)

    if module_name == "src.mpdem.DEMPMBase":
        solver.CouplingSolver(None, None)
        assert abs(solver.msims.current_time - target) < 1.0e-14
        assert abs(solver.dsims.current_time - target) < 1.0e-14
    elif module_name == "src.mpm.MPMBase":
        solver.Solver(None, None)
    else:
        solver.Solver(None)

    assert len(calls) == count
    assert min(calls) > 2.9e-6
    assert solver.sims.current_step == count
    assert abs(solver.sims.current_time - target) < 1.0e-14
    assert saved[0] == 0.0 and abs(saved[-1] - target) < 1.0e-14
