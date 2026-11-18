"""Dependency-free diagnostics shared by every public solver facade.

The adapter intentionally observes existing solver state without importing
Taichi or NumPy.  This keeps it usable from the MCP worker even when a task
fails before the numerical backend has finished initializing.
"""

from __future__ import annotations
import math


class SolverDiagnosticsMixin:
    """Opt-in facade mixin exposing the common diagnostics contract."""

    @staticmethod
    def _plain_scalar(value):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if callable(value):
            return None
        for index in (None, 0):
            try:
                candidate = value[index]
            except (IndexError, KeyError, TypeError, AttributeError):
                continue
            if candidate is not value:
                return SolverDiagnosticsMixin._plain_scalar(candidate)
        try:
            candidate = value.item()
        except (AttributeError, TypeError, ValueError):
            return None
        return SolverDiagnosticsMixin._plain_scalar(candidate)

    @staticmethod
    def _first_scalar(owners, names):
        for owner in owners:
            if owner is None:
                continue
            for name in names:
                try:
                    value = getattr(owner, name)
                except (AttributeError, RuntimeError, TypeError):
                    continue
                value = SolverDiagnosticsMixin._plain_scalar(value)
                if value is not None:
                    return value
        return None

    def diagnostics_snapshot(self):
        """Return a stable, JSON-friendly snapshot for a facade or engine."""

        engine = getattr(self, "enginer", None) or getattr(self, "engine", None)
        if engine is None:
            engine = self
        delegated = getattr(engine, "diagnostics_snapshot", None)
        if engine is not self and callable(delegated):
            snapshot = delegated()
            if isinstance(snapshot, dict):
                snapshot.setdefault("schema_version", 1)
                snapshot.setdefault("solver", type(self).__name__)
                snapshot.setdefault("engine", type(engine).__name__)
                return snapshot

        simulation = (
            getattr(self, "sims", None)
            or getattr(self, "simulation", None)
            or getattr(engine, "sims", None)
            or getattr(engine, "simulation", None)
        )
        state_owners = (engine, simulation, self)
        state = {}
        for key, names in (
            ("time", ("current_time", "time")),
            ("step", ("step_count", "current_step")),
            ("timestep", ("dt", "delta", "time_step")),
        ):
            value = SolverDiagnosticsMixin._first_scalar(state_owners, names)
            if value is not None:
                state[key] = value
        target_time = SolverDiagnosticsMixin._first_scalar(
            (simulation, self),
            ("target_time", "simulation_time", "time"),
        )
        if target_time is not None:
            state["target_time"] = target_time

        snapshot = {
            "schema_version": 1,
            "solver": type(self).__name__,
            "engine": type(engine).__name__,
            "state": state,
        }
        failure = getattr(engine, "last_failure", None)
        if isinstance(failure, dict):
            snapshot["last_failure"] = failure
        history = getattr(engine, "history", None)
        if isinstance(history, (list, tuple)) and history:
            record = history[-1]
            if isinstance(record, dict):
                snapshot["last_step"] = record
        return snapshot
