"""Pure state transitions for the Blender SolverJob workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class Phase(str, Enum):
    IDLE = "IDLE"
    EXPORTED = "EXPORTED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class Event(str, Enum):
    EXPORT_SUCCEEDED = "EXPORT_SUCCEEDED"
    VALIDATION_SUCCEEDED = "VALIDATION_SUCCEEDED"
    SUBMISSION_SUCCEEDED = "SUBMISSION_SUCCEEDED"
    STATUS_RECEIVED = "STATUS_RECEIVED"
    FETCH_SUCCEEDED = "FETCH_SUCCEEDED"
    OPERATION_FAILED = "OPERATION_FAILED"
    RESET = "RESET"


@dataclass(frozen=True)
class AppState:
    phase: Phase = Phase.IDLE
    job_path: str = ""
    task_id: str = ""
    task_status: str = ""
    output_directory: str = ""
    cache_directory: str = ""
    message: str = ""
    error: str = ""


def transition(
    state: AppState,
    event: Event,
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[AppState, Tuple[str, ...]]:
    """Return a new state and declarative effects for one workflow event."""
    data = payload or {}
    if event == Event.RESET:
        return AppState(), ("redraw",)
    if event == Event.OPERATION_FAILED:
        return replace(state, error=str(data.get("error", "Operation failed")), message=""), ("redraw",)
    if event == Event.EXPORT_SUCCEEDED:
        job_path = str(data.get("job_path", ""))
        if not job_path:
            raise ValueError("EXPORT_SUCCEEDED requires job_path")
        return (
            replace(
                state,
                phase=Phase.EXPORTED,
                job_path=job_path,
                task_id="",
                task_status="",
                output_directory="",
                cache_directory="",
                message="SolverJob exported",
                error="",
            ),
            ("redraw",),
        )
    if event == Event.VALIDATION_SUCCEEDED:
        if not state.job_path:
            raise ValueError("cannot validate before exporting a SolverJob")
        return replace(state, phase=Phase.VALIDATED, message="SolverJob validated", error=""), ("redraw",)
    if event == Event.SUBMISSION_SUCCEEDED:
        task_id = str(data.get("task_id", ""))
        if not task_id:
            raise ValueError("SUBMISSION_SUCCEEDED requires task_id")
        return (
            replace(
                state,
                phase=Phase.SUBMITTED,
                task_id=task_id,
                task_status=str(data.get("task_status", "pending")),
                output_directory=str(data.get("output_directory", "")),
                message="SolverJob submitted",
                error="",
            ),
            ("redraw", "schedule_status_poll"),
        )
    if event == Event.STATUS_RECEIVED:
        status = str(data.get("task_status", "")).lower()
        phase_by_status = {
            "pending": Phase.SUBMITTED,
            "running": Phase.RUNNING,
            "completed": Phase.COMPLETED,
            "failed": Phase.FAILED,
            "interrupted": Phase.INTERRUPTED,
        }
        if status not in phase_by_status:
            raise ValueError("unsupported task status %r" % status)
        return (
            replace(
                state,
                phase=phase_by_status[status],
                task_status=status,
                output_directory=str(data.get("output_directory", state.output_directory)),
                message=str(data.get("message", "Task status: %s" % status)),
                error=str(data.get("error", "")),
            ),
            ("redraw",) if status in {"completed", "failed", "interrupted"} else ("redraw", "schedule_status_poll"),
        )
    if event == Event.FETCH_SUCCEEDED:
        if state.phase not in {Phase.COMPLETED, Phase.FAILED, Phase.INTERRUPTED}:
            raise ValueError("artifacts can be fetched only after the task reaches a terminal state")
        return (
            replace(
                state,
                cache_directory=str(data.get("cache_directory", "")),
                message="Task artifacts cached",
                error="",
            ),
            ("redraw",),
        )
    raise ValueError("unsupported workflow event %r" % event)
