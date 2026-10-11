import importlib

import pytest

from src.utils.SolverRuntime import (
    StepSchedule,
    inexact_newton_absolute_tolerance,
    inexact_newton_relative_tolerance,
    normalize_callbacks,
    python_callback,
)


def test_inexact_newton_forcing_restarts_caps_and_tightens():
    assert inexact_newton_relative_tolerance(1.0, None) == 0.01
    assert inexact_newton_relative_tolerance(2.0, 1.0) == 0.01
    assert inexact_newton_relative_tolerance(0.01, 1.0) == pytest.approx(0.0009)
    assert inexact_newton_relative_tolerance(1e-12, 1.0) == 1e-7


def test_inexact_linear_error_uses_physical_force_and_preserves_absolute_floor():
    assert inexact_newton_absolute_tolerance(7.0, 0.01, 1e-8) == pytest.approx(0.07)
    assert inexact_newton_absolute_tolerance(0.0, 0.01, 1e-8) == 1e-8
    assert inexact_newton_absolute_tolerance(1e-9, 0.01, 1e-8) == 1e-8


@pytest.mark.parametrize("residual", [-1.0, float("nan"), float("inf")])
def test_inexact_linear_error_rejects_invalid_physical_force(residual):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        inexact_newton_absolute_tolerance(residual, 0.01, 1e-8)


KERNEL_CALLBACK_SOLVERS = (
    ("src.dem.DEMBase", "Solver"),
    ("src.mpm.MPMBase", "Solver"),
    ("src.mpdem.DEMPMBase", "Solver"),
    ("src.fedem.FEDEMBase", "Solver"),
    ("src.fempm.FEMPMBase", "Solver"),
    ("src.mpdem.engines.SoftAffineIPCBase", "SoftAffineIPCSolver"),
)


def test_callback_normalization_preserves_mapping_order_and_validates_entries():
    first = lambda: None
    second = lambda: None

    assert normalize_callbacks(None) == ()
    assert normalize_callbacks(first) == (first,)
    assert normalize_callbacks({"first": first, "second": second}) == (
        first,
        second,
    )
    assert normalize_callbacks([first, second], transform=lambda f: (f,)) == (
        (first,),
        (second,),
    )
    assert normalize_callbacks(python_callback(first), transform=lambda f: (f,)) == (first,)
    with pytest.raises(TypeError, match="every solver callback"):
        normalize_callbacks([first, 3])


@pytest.mark.parametrize(("module_name", "class_name"), KERNEL_CALLBACK_SOLVERS)
def test_solver_python_callback_bypasses_taichi_kernel_wrapping(module_name, class_name):
    solver_class = getattr(importlib.import_module(module_name), class_name)
    solver = solver_class.__new__(solver_class)
    solver.postprocess = []
    callback = python_callback(lambda: None)

    solver.set_callback_function(callback)

    assert solver.postprocess == [callback]


def test_step_schedule_separates_diagnostics_jacobian_and_history():
    schedule = StepSchedule(
        diagnostic_interval=4,
        jacobian_interval=2,
        history_interval=3,
        max_history_entries=2,
    )

    assert not schedule.diagnostics_due(1)
    assert schedule.diagnostics_due(1, callback_requires_diagnostics=True)
    assert schedule.diagnostics_due(4)
    assert schedule.jacobian_due(2)
    assert not schedule.jacobian_due(3)
    assert schedule.history_due(3)
    assert schedule.history_due(1, output=True)

    history = []
    for step in range(4):
        schedule.append_history(history, {"step": step})
    assert history == [{"step": 2}, {"step": 3}]


@pytest.mark.parametrize(
    "keyword",
    ("diagnostic_interval", "jacobian_interval", "history_interval"),
)
def test_step_schedule_rejects_nonpositive_intervals(keyword):
    with pytest.raises(ValueError, match=keyword):
        StepSchedule(**{keyword: 0})
