import pytest

pytest.importorskip("taichi")

from src.dem.engines.AffineBodyEngine import AffineBodyEngine
from src.fem.engines.ImplicitFEM import (
    ImplicitFEM,
    NewtonConvergenceError,
)
from src.iga.engines.ImplicitIGA import IGAConvergenceError, ImplicitIGA
from src.igampm.engines.CoupledEngine import Engine as IGAMPMEngine
from src.mpm.engines.direct.MPMSolver import (
    MPMConvergenceError,
    MPMSolver,
)
from src.mpdem.engines.SoftAffineIPCEngine import SoftAffineIPCEngine
from src.utils.StepRetry import StepRetryPolicy
from src.utils.SolverRuntime import StepSchedule


class _Scalar:
    def __init__(self, value):
        self.value = float(value)

    def __getitem__(self, index):
        return self.value

    def __setitem__(self, index, value):
        self.value = float(value)


class _Simulation:
    def __init__(self):
        self.dt = _Scalar(1.0)
        self.delta = 1.0
        self.current_time = 0.0
        self.current_step = 0

    def set_timestep(self, value):
        self.dt[None] = value
        self.delta = float(value)


def _retry_policy():
    return StepRetryPolicy(enabled=True, maximum_retries=2, reduction=0.5)


def test_standalone_fem_retry_accepts_reduced_timestep():
    engine = object.__new__(ImplicitFEM)
    engine.dt = 1.0
    engine.step_retry = _retry_policy()
    engine.history = []
    engine.last_step_record = None
    engine.last_failure = None
    engine.time = 0.0
    engine.step_count = 0
    calls = []

    def attempt(record_history=True):
        assert record_history
        calls.append(engine.dt)
        if len(calls) == 1:
            raise NewtonConvergenceError("Newton did not converge")
        engine.history.append({"step": 1})
        engine.last_step_record = engine.history[-1]
        return True

    engine._substep_once = attempt
    assert ImplicitFEM.substep(engine)
    assert calls == [1.0, 0.5]
    assert engine.history[-1]["step_retry"]["retry_count"] == 1


def test_standalone_iga_retry_accepts_reduced_timestep():
    engine = object.__new__(ImplicitIGA)
    engine.dt = 1.0
    engine.step_retry = _retry_policy()
    engine.history = []
    engine.last_step_record = None
    engine.last_failure = None
    engine.time = 0.0
    engine.step_count = 0
    calls = []

    def attempt(verbose=True):
        calls.append(engine.dt)
        if len(calls) == 1:
            raise IGAConvergenceError("IGA did not converge")
        engine.history.append({"step": 1})
        engine.last_step_record = engine.history[-1]

    engine._substep_once = attempt
    ImplicitIGA.substep(engine, verbose=False)
    assert calls == [1.0, 0.5]
    assert engine.history[-1]["step_retry"]["accepted_timestep"] == 0.5


def test_direct_mpm_retry_accepts_reduced_timestep():
    engine = object.__new__(MPMSolver)
    engine.dt = 1.0
    engine.step_retry = _retry_policy()
    engine.history = []
    engine.last_failure = None
    engine.time = 0.0
    engine.step_count = 0
    calls = []

    def attempt(verbose=True):
        calls.append(engine.dt)
        if len(calls) == 1:
            raise MPMConvergenceError("MPM did not converge")
        return {"residual": 1.0e-8}

    MPMSolver.run_substep(engine, attempt, verbose=False)
    assert calls == [1.0, 0.5]
    assert engine.time == 0.5
    assert engine.history[-1]["residual"] == 1.0e-8


def test_igampm_retry_uses_transactional_substep():
    engine = object.__new__(IGAMPMEngine)
    engine.iga = type("IGA", (), {"dt": 1.0})()
    engine.mpm = type("MPM", (), {"dt": 1.0})()
    engine.step_retry = _retry_policy()
    engine.history = []
    engine.last_failure = None
    engine.time = 0.0
    engine.implicit_step_index = 0
    engine.step_schedule = StepSchedule()
    engine.last_step_record = None
    engine.track_energy = False
    engine.add_implicit_energy_record = lambda _record, _result: None
    engine.last_contact_ccd_step = 1.0
    engine.last_contact_ccd_min_distance = 1.0
    engine.barrier = type("Barrier", (), {"model": "BarrierIPC"})()
    engine.is_semi = False
    calls = []

    def attempt(**kwargs):
        calls.append(engine.iga.dt)
        if len(calls) == 1:
            raise RuntimeError("monolithic Newton did not converge")
        engine.implicit_step_index = 1
        return {"converged": True, "minimum_distance": 1.0}

    engine._implicit_ipc_substep_once = attempt
    result = IGAMPMEngine.implicit_ipc_substep(engine)
    assert calls == [1.0, 0.5]
    assert result["step_retry"]["retry_count"] == 1
    assert engine.time == 0.5


@pytest.mark.parametrize("engine_type", [AffineBodyEngine, SoftAffineIPCEngine])
def test_affine_retry_controllers_accept_reduced_timestep(engine_type):
    engine = object.__new__(engine_type)
    engine.step_retry = _retry_policy()
    engine.history = []
    engine.last_failure = None
    engine.last_newton_iterations = 1
    engine.last_friction_residual = 0.0
    engine.last_ccd_step = 1.0
    engine.last_candidate_pairs = 0
    engine.operator = type(
        "Operator",
        (),
        {
            "dt": 1.0,
            "cuda_hot_loop": True,
            "rollback_step_device": lambda self: None,
        },
    )()
    engine.initialize = lambda sims, scene: None
    calls = []

    def attempt(sims, scene):
        calls.append(sims.delta)
        if len(calls) == 1:
            raise RuntimeError("Newton solve did not converge")

    if engine_type is SoftAffineIPCEngine:
        engine.requested_friction_iterations = 1
        engine.advance = lambda sims, _requested: attempt(sims, None)
    else:
        engine._step_once = attempt
    simulation = _Simulation()
    engine_type.step(engine, simulation, None)

    assert calls == [1.0, 0.5]
    assert simulation.delta == 0.5
    assert engine.history[-1]["step_retry"]["retry_count"] == 1
