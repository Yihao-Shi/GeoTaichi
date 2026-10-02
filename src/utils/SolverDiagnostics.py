"""Dependency-free diagnostics shared by every public solver facade.

The adapter intentionally observes existing solver state without importing
Taichi or NumPy.  This keeps it usable from the MCP worker even when a task
fails before the numerical backend has finished initializing.
"""

from __future__ import annotations

import math


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
            return _plain_scalar(candidate)
    try:
        candidate = value.item()
    except (AttributeError, TypeError, ValueError):
        return None
    return _plain_scalar(candidate)


def _first_scalar(owners, names):
    for owner in owners:
        if owner is None:
            continue
        for name in names:
            try:
                value = getattr(owner, name)
            except (AttributeError, RuntimeError, TypeError):
                continue
            value = _plain_scalar(value)
            if value is not None:
                return value
    return None


def _last_history(engine):
    history = getattr(engine, "history", None)
    if not isinstance(history, (list, tuple)) or not history:
        return None
    record = history[-1]
    return record if isinstance(record, dict) else None


def solver_diagnostics_snapshot(owner):
    """Return a stable, JSON-friendly snapshot for a facade or engine."""

    engine = getattr(owner, "enginer", None) or getattr(owner, "engine", None)
    if engine is None:
        engine = owner
    delegated = getattr(engine, "diagnostics_snapshot", None)
    if engine is not owner and callable(delegated):
        snapshot = delegated()
        if isinstance(snapshot, dict):
            snapshot.setdefault("schema_version", 1)
            snapshot.setdefault("solver", type(owner).__name__)
            snapshot.setdefault("engine", type(engine).__name__)
            return snapshot

    simulation = (
        getattr(owner, "sims", None)
        or getattr(owner, "simulation", None)
        or getattr(engine, "sims", None)
        or getattr(engine, "simulation", None)
    )
    state_owners = (engine, simulation, owner)
    state = {}
    for key, names in (
        ("time", ("time", "current_time")),
        ("step", ("step_count", "current_step")),
        ("timestep", ("dt", "delta", "time_step")),
    ):
        value = _first_scalar(state_owners, names)
        if value is not None:
            state[key] = value
    target_time = _first_scalar(
        (simulation, owner),
        ("target_time", "simulation_time", "time"),
    )
    if target_time is not None:
        state["target_time"] = target_time

    snapshot = {
        "schema_version": 1,
        "solver": type(owner).__name__,
        "engine": type(engine).__name__,
        "state": state,
    }
    failure = getattr(engine, "last_failure", None)
    if isinstance(failure, dict):
        snapshot["last_failure"] = failure
    history = _last_history(engine)
    if history is not None:
        snapshot["last_step"] = history
    return snapshot


class SolverDiagnosticsMixin:
    """Opt-in facade mixin exposing the common diagnostics contract."""

    def diagnostics_snapshot(self):
        return solver_diagnostics_snapshot(self)


__all__ = ["SolverDiagnosticsMixin", "solver_diagnostics_snapshot"]
