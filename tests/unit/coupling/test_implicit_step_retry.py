from types import SimpleNamespace

import pytest

pytest.importorskip("taichi")

from src.fedem.AffineIPCEngine import FEMAffineIPCEngine
from src.fem.engines.ImplicitFEM import NewtonConvergenceError
from src.fempm.ImplicitEngine import FEMPMImplicitEngine
from src.utils.StepRetry import StepRetryPolicy


def test_fempm_run_does_not_take_a_floating_point_extra_step():
    engine = object.__new__(FEMPMImplicitEngine)
    engine.time = 0.5 - 1.2e-14
    engine.step_count = 10
    engine.compile_seconds = 0.0
    engine.history = []
    engine.simulation = SimpleNamespace(time=0.5, current_time=engine.time)
    engine.fem = SimpleNamespace(time=engine.time)
    engine.mpm_wrapper = SimpleNamespace(sims=SimpleNamespace(current_time=engine.time))
    engine.substep = lambda **kwargs: pytest.fail("floating-point drift triggered an extra step")

    result = FEMPMImplicitEngine.run(engine, verbose=False)

    assert result["time"] == 0.5
    assert engine.simulation.current_time == 0.5
    assert engine.fem.time == 0.5
    assert engine.mpm_wrapper.sims.current_time == 0.5


@pytest.mark.parametrize(
    ("engine_type", "public_method", "attempt_method"),
    [
        (FEMPMImplicitEngine, "substep", "_substep_once"),
        (FEMAffineIPCEngine, "step", "_step_once"),
    ],
)
def test_implicit_coupling_retry_accepts_a_reduced_timestep(engine_type, public_method, attempt_method):
    engine = object.__new__(engine_type)
    engine.dt = 1.0
    engine.step_retry = StepRetryPolicy(enabled=True, maximum_retries=2, reduction=0.5)
    engine.history = []
    engine.last_step_record = None
    engine.last_failure = None
    applied = []
    calls = []

    def set_timestep(value):
        applied.append(float(value))
        engine.dt = float(value)

    def diagnostics(exception, attempt, timestep):
        return {
            "kind": "newton_nonconvergence",
            "exception": type(exception).__name__,
            "message": str(exception),
            "attempt": attempt,
            "timestep": timestep,
        }

    def attempt(verbose=False):
        calls.append(engine.dt)
        if len(calls) == 1:
            raise NewtonConvergenceError("Newton did not converge")
        engine.history.append({"step": 1})
        engine.last_step_record = engine.history[-1]
        return True

    engine._set_timestep = set_timestep
    engine._failure_diagnostics = diagnostics
    setattr(engine, attempt_method, attempt)

    assert getattr(engine_type, public_method)(engine, verbose=False)
    assert calls == [1.0, 0.5]
    assert applied == [1.0, 0.5]
    assert engine.dt == pytest.approx(0.5)
    assert engine.history[-1]["step_retry"]["retry_count"] == 1
    assert engine.history[-1]["step_retry"]["accepted_timestep"] == pytest.approx(0.5)
    assert engine.last_failure is None


@pytest.mark.parametrize(
    ("engine_type", "public_method", "attempt_method"),
    [
        (FEMPMImplicitEngine, "substep", "_substep_once"),
        (FEMAffineIPCEngine, "step", "_step_once"),
    ],
)
def test_implicit_coupling_retry_is_bounded_and_restores_configured_timestep(
    engine_type, public_method, attempt_method
):
    engine = object.__new__(engine_type)
    engine.dt = 1.0
    engine.step_retry = StepRetryPolicy(enabled=True, maximum_retries=1, reduction=0.25)
    engine.history = []
    engine.last_step_record = None
    engine.last_failure = None
    calls = []

    def set_timestep(value):
        engine.dt = float(value)

    def attempt(verbose=False):
        calls.append(engine.dt)
        raise NewtonConvergenceError("linear solve did not converge")

    engine._set_timestep = set_timestep
    engine._failure_diagnostics = lambda exc, index, dt: {
        "kind": "linear_solver_nonconvergence",
        "attempt": index,
        "timestep": dt,
    }
    setattr(engine, attempt_method, attempt)

    with pytest.raises(NewtonConvergenceError):
        getattr(engine_type, public_method)(engine, verbose=False)

    assert calls == [1.0, 0.25]
    assert engine.dt == pytest.approx(1.0)
    assert engine.last_failure["original_timestep"] == pytest.approx(1.0)
    assert len(engine.last_failure["attempts"]) == 2


@pytest.mark.parametrize(
    ("engine_type", "public_method", "attempt_method"),
    [
        (FEMPMImplicitEngine, "substep", "_substep_once"),
        (FEMAffineIPCEngine, "step", "_step_once"),
    ],
)
def test_implicit_coupling_does_not_retry_non_solver_errors(engine_type, public_method, attempt_method):
    engine = object.__new__(engine_type)
    engine.dt = 1.0
    engine.step_retry = StepRetryPolicy(enabled=True, maximum_retries=3, reduction=0.5)
    engine.history = []
    engine.last_step_record = None
    engine.last_failure = None
    calls = []
    engine._set_timestep = lambda value: setattr(engine, "dt", float(value))

    def attempt(verbose=False):
        calls.append(engine.dt)
        raise RuntimeError("contact capacity overflow")

    setattr(engine, attempt_method, attempt)
    with pytest.raises(RuntimeError, match="capacity overflow"):
        getattr(engine_type, public_method)(engine, verbose=False)
    assert calls == [1.0]
