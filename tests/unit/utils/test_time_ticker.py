from src.utils import GlobalVariable
from math import fsum
from types import SimpleNamespace

from src.utils.TimeTicker import Timer, advance_time, has_remaining_time
import src.utils.TimeTicker as time_ticker


def test_gpu_timer_end_always_synchronizes(monkeypatch):
    calls = []
    monkeypatch.setattr(GlobalVariable, "USEGPU", True)
    monkeypatch.setattr(time_ticker.ti, "sync", lambda: calls.append("sync"))

    timer = Timer()
    timer.begin("asynchronous")
    timer.end("asynchronous")
    assert calls == ["sync"]


def test_gpu_timer_section_preserves_body_error_when_sync_also_fails(monkeypatch):
    monkeypatch.setattr(GlobalVariable, "USEGPU", True)
    monkeypatch.setattr(
        time_ticker.ti,
        "sync",
        lambda: (_ for _ in ()).throw(RuntimeError("late CUDA error")),
    )

    timer = Timer()
    try:
        with timer.section("failing"):
            raise ValueError("specific stage")
    except ValueError as exception:
        assert str(exception) == "specific stage"
        assert isinstance(exception.__cause__, RuntimeError)
    else:
        raise AssertionError("section swallowed the body exception")


def test_roundoff_remainder_does_not_create_a_tiny_terminal_step():
    assert not has_remaining_time(0.65 - 3.9e-14, 0.65, 2.0e-4)
    assert has_remaining_time(0.65 - 1.0e-6, 0.65, 2.0e-4)


def test_compensated_clock_keeps_whole_and_fractional_physical_steps():
    # Plain += takes a spurious 12001st step of 1.38e-14 s for target 0.12.
    for target, expected_steps in ((0.12, 12000), (0.120003, 12001), (0.5, 50000)):
        sims = SimpleNamespace(current_time=0.0)
        steps = []
        while has_remaining_time(sims.current_time, target, 1.0e-5):
            step_dt = min(1.0e-5, target - sims.current_time)
            steps.append(step_dt)
            advance_time(sims, step_dt)
        assert len(steps) == expected_steps
        assert min(steps) > 2.9e-6
        assert abs(sims.current_time - target) < 1.0e-14


def test_compensated_clock_handles_adaptive_steps_and_external_restart():
    sims = SimpleNamespace(current_time=0.0)
    steps = [1.0e-5, 3.0e-6, 7.0e-6] * 10000
    for step in steps:
        advance_time(sims, step)
    assert sims.current_time == fsum(steps)
    sims.current_time = 0.03
    advance_time(sims, 2.0e-6)
    assert sims.current_time == 0.03 + 2.0e-6
