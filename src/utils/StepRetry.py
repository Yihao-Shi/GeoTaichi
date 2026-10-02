"""Host-side bounded retry policy for recoverable implicit-step failures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StepRetryPolicy:
    """Validated geometric timestep reduction policy.

    ``maximum_retries`` counts additional attempts after the original step.
    A zero ``minimum_timestep`` disables the optional lower bound.
    """

    enabled: bool = False
    maximum_retries: int = 2
    reduction: float = 0.5
    minimum_timestep: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enable_step_retry must be a boolean")
        try:
            maximum_retries = int(self.maximum_retries)
        except (TypeError, ValueError) as exc:
            raise ValueError("step_retry_max_retries must be an integer") from exc
        if isinstance(self.maximum_retries, bool) or maximum_retries != self.maximum_retries:
            raise ValueError("step_retry_max_retries must be an integer")
        try:
            reduction = float(self.reduction)
            minimum_timestep = float(self.minimum_timestep)
        except (TypeError, ValueError) as exc:
            raise ValueError("step retry timestep values must be numeric") from exc
        if maximum_retries < 0:
            raise ValueError("step_retry_max_retries must be non-negative")
        if not math.isfinite(reduction) or not 0.0 < reduction < 1.0:
            raise ValueError("step_retry_reduction must be finite and in (0, 1)")
        if not math.isfinite(minimum_timestep) or minimum_timestep < 0.0:
            raise ValueError("step_retry_minimum_timestep must be finite and non-negative")
        object.__setattr__(self, "maximum_retries", maximum_retries)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "minimum_timestep", minimum_timestep)

    def next_timestep(self, current: float, retries_completed: int) -> Optional[float]:
        """Return the next strictly smaller timestep, or ``None`` when exhausted."""
        if not self.enabled or retries_completed >= self.maximum_retries:
            return None
        current = float(current)
        if not math.isfinite(current) or current <= 0.0:
            raise ValueError("current retry timestep must be finite and positive")
        candidate = current * self.reduction
        if self.minimum_timestep > 0.0:
            candidate = max(candidate, self.minimum_timestep)
        if not candidate < current:
            return None
        return candidate


def nonlinear_failure_kind(exception: BaseException) -> str:
    """Map implicit solver messages onto stable diagnostic categories."""
    message = str(exception).lower()
    if "linear solve" in message or "pcg" in message or "bicg" in message:
        return "linear_solver_nonconvergence"
    if "line search" in message or "no strictly feasible step" in message:
        return "line_search_failure"
    if "descent" in message or "descending" in message:
        return "non_descent_direction"
    if "did not converge" in message or "failed to converge" in message:
        return "newton_nonconvergence"
    return "nonlinear_solver_failure"


def is_recoverable_nonlinear_failure(exception: BaseException) -> bool:
    """Recognize only failures emitted by an active nonlinear/linear solve."""

    message = str(exception).lower()
    return any(
        phrase in message
        for phrase in (
            "did not converge",
            "failed to converge",
            "line search failed",
            "not descending",
            "descent direction",
            "non-descent direction",
            "no finite descent direction",
            "no strictly feasible step",
        )
    )


__all__ = [
    "StepRetryPolicy",
    "is_recoverable_nonlinear_failure",
    "nonlinear_failure_kind",
]
