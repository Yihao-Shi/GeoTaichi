"""Small setup-time policies shared by solver drivers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def inexact_newton_relative_tolerance(residual, previous_residual, minimum=1.0e-7):
    """Bound the adaptive Krylov forcing term without relaxing Newton convergence."""
    if previous_residual is None or previous_residual <= 0.0:
        return 1.0e-2
    return min(1.0e-2, max(minimum, 0.9 * (residual / previous_residual) ** 1.5))


def python_callback(function):
    """Mark a solver callback to run in Python instead of as a Taichi kernel."""
    if not callable(function):
        raise TypeError("python_callback expects a callable")
    function.__geotaichi_python_callback__ = True
    return function


def normalize_callbacks(functions, transform=None):
    """Return one validated callback tuple from a public callback argument."""
    if functions is None:
        return ()
    if isinstance(functions, Mapping):
        selected = tuple(functions.values())
    elif callable(functions):
        selected = (functions,)
    else:
        try:
            selected = tuple(functions)
        except TypeError as exception:
            raise TypeError("solver callbacks must be a callable, mapping, or iterable") from exception
    if not all(callable(function) for function in selected):
        raise TypeError("every solver callback must be callable")
    if transform is not None:
        selected = tuple(
            function if getattr(function, "__geotaichi_python_callback__", False) else transform(function)
            for function in selected
        )
    return selected


def _positive_interval(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class StepSchedule:
    """Accepted-step schedule for diagnostics, safety checks, and history."""

    diagnostic_interval: int = 1
    jacobian_interval: int = 1
    history_interval: int = 1
    max_history_entries: int | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "diagnostic_interval",
            _positive_interval(self.diagnostic_interval, "diagnostic_interval"),
        )
        object.__setattr__(
            self,
            "jacobian_interval",
            _positive_interval(self.jacobian_interval, "jacobian_interval"),
        )
        object.__setattr__(
            self,
            "history_interval",
            _positive_interval(self.history_interval, "history_interval"),
        )
        if self.max_history_entries is not None:
            capacity = int(self.max_history_entries)
            if capacity <= 0:
                raise ValueError("max_history_entries must be positive or None")
            object.__setattr__(self, "max_history_entries", capacity)

    @classmethod
    def from_options(cls, options, *, output_interval=1):
        diagnostic_interval = options.get("diagnostic_interval", output_interval)
        return cls(
            diagnostic_interval=diagnostic_interval,
            jacobian_interval=options.get("jacobian_interval", 1),
            history_interval=options.get("history_interval", diagnostic_interval),
            max_history_entries=options.get("max_history_entries"),
        )

    @staticmethod
    def _due(step, interval):
        return int(step) % interval == 0

    def diagnostics_due(
        self,
        step,
        *,
        output=False,
        callback_requires_diagnostics=False,
        final=False,
    ):
        return bool(output or callback_requires_diagnostics or final or self._due(step, self.diagnostic_interval))

    def jacobian_due(self, step, *, output=False, final=False):
        return bool(output or final or self._due(step, self.jacobian_interval))

    def history_due(self, step, *, output=False, final=False):
        return bool(output or final or self._due(step, self.history_interval))

    def append_history(self, history, record):
        history.append(record)
        if self.max_history_entries is not None and len(history) > self.max_history_entries:
            del history[: len(history) - self.max_history_entries]


__all__ = ["StepSchedule", "inexact_newton_relative_tolerance", "normalize_callbacks", "python_callback"]
