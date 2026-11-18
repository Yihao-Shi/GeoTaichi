"""Validated accepted-step scheduling for diagnostics, Jacobians, and history."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class StepSchedule:
    """Accepted-step schedule for diagnostics, safety checks, and history."""

    @staticmethod
    def _positive_interval(value, name):
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    diagnostic_interval: int = 1
    jacobian_interval: int = 1
    history_interval: int = 1
    max_history_entries: int | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "diagnostic_interval",
            StepSchedule._positive_interval(self.diagnostic_interval, "diagnostic_interval"),
        )
        object.__setattr__(
            self,
            "jacobian_interval",
            StepSchedule._positive_interval(self.jacobian_interval, "jacobian_interval"),
        )
        object.__setattr__(
            self,
            "history_interval",
            StepSchedule._positive_interval(self.history_interval, "history_interval"),
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

    def diagnostics_due(
        self,
        step,
        *,
        output=False,
        callback_requires_diagnostics=False,
        final=False,
    ):
        return bool(output or callback_requires_diagnostics or final or int(step) % self.diagnostic_interval == 0)

    def jacobian_due(self, step, *, output=False, final=False):
        return bool(output or final or int(step) % self.jacobian_interval == 0)

    def history_due(self, step, *, output=False, final=False):
        return bool(output or final or int(step) % self.history_interval == 0)

    def append_history(self, history, record):
        history.append(record)
        if self.max_history_entries is not None and len(history) > self.max_history_entries:
            del history[: len(history) - self.max_history_entries]
