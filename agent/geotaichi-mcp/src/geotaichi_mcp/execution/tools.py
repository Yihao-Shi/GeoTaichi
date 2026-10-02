"""MCP business functions for synchronous code and persistent tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.config import find_repo_root, resolve_user_path
from ..core.contracts import build_error, build_ok
from ..core.jobs import prepare_solver_job
from .task_manager import TaskManager


_task_manager: Optional[TaskManager] = None


def configure_task_manager(manager: Optional[TaskManager] = None) -> TaskManager:
    """Install a manager for tests or rebuild it from current environment settings."""
    global _task_manager
    _task_manager = manager or TaskManager()
    return _task_manager


def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


def geotaichi_execute_code(
    code: str,
    timeout: float = 10.0,
    working_directory: str = "",
    task_id: str = "",
) -> Dict[str, Any]:
    """Execute code in a running task namespace, or in an isolated process when no task ID is supplied."""
    try:
        if task_id:
            if working_directory:
                return build_error(
                    "invalid_live_working_directory",
                    "working_directory cannot be used together with task_id; live code uses the task directory.",
                    {"task_id": task_id},
                )
            result = get_task_manager().execute_code_in_task(task_id, code, timeout)
        else:
            result = get_task_manager().execute_code(code, timeout, working_directory or None)
        if result["status"] == "completed":
            return build_ok(result)
        return build_error(
            "code_%s" % result["status"],
            "%s code execution %s." % (result.get("execution_mode", "isolated").capitalize(), result["status"]),
            {
                "task_id": result.get("task_id"),
                "return_code": result.get("return_code"),
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            },
            result,
        )
    except KeyError:
        return build_error("task_not_found", "Task %r was not found." % task_id)
    except Exception as exc:
        return build_error("code_execution_failed", str(exc), {"task_id": task_id or None})


def geotaichi_execute_task(
    entry_script: str,
    description: str,
    arguments: Optional[List[str]] = None,
    working_directory: str = "",
) -> Dict[str, Any]:
    """Submit a Python model script for isolated, asynchronous execution and return a task ID."""
    try:
        metadata = get_task_manager().submit(
            entry_script,
            description,
            arguments,
            working_directory or None,
        )
        return build_ok(
            {
                "task_id": metadata["task_id"],
                "task_status": metadata["status"],
                "entry_script": metadata["entry_script"],
                "description": metadata["description"],
                "task_directory": metadata["task_directory"],
                "output_directory": metadata["output_directory"],
                "stdout_path": metadata["stdout_path"],
                "stderr_path": metadata["stderr_path"],
            }
        )
    except FileNotFoundError as exc:
        return build_error("entry_script_not_found", str(exc))
    except Exception as exc:
        return build_error("task_submission_failed", str(exc))


def geotaichi_validate_solver_job(job_path: str) -> Dict[str, Any]:
    """Validate and resolve a SolverJob and its optional model/scene contracts without executing it."""
    try:
        repo = find_repo_root(required=False)
        path = resolve_user_path(job_path, repo, must_exist=True)
        context, validation = prepare_solver_job(path, repo)
        return build_ok(
            {
                "valid": True,
                "job_path": str(path),
                "fingerprint": context["fingerprint"],
                "entry_script": context["entry_script"],
                "working_directory": context["working_directory"],
                "scene_manifest": context.get("scene_source_path"),
                "scene_fingerprint": context.get("scene_fingerprint"),
                "model_contract": context.get("contract_source_path"),
                "model_contract_fingerprint": context.get("contract_fingerprint"),
                "warnings": validation["warnings"],
            }
        )
    except FileNotFoundError as exc:
        return build_error("solver_job_input_not_found", str(exc))
    except (ValueError, NotADirectoryError) as exc:
        return build_error("invalid_solver_job", str(exc))
    except Exception as exc:
        return build_error("solver_job_validation_failed", str(exc))


def geotaichi_submit_solver_job(
    job_path: str,
    confirm_trusted_execution: bool = False,
) -> Dict[str, Any]:
    """Submit a validated SolverJob; trusted script execution requires explicit caller confirmation."""
    if confirm_trusted_execution is not True:
        return build_error(
            "trusted_execution_not_confirmed",
            "SolverJob runs a local Python script. Set confirm_trusted_execution=true after reviewing the job.",
        )
    try:
        repo = find_repo_root(required=False)
        path = resolve_user_path(job_path, repo, must_exist=True)
        context, validation = prepare_solver_job(path, repo)
        metadata = get_task_manager().submit(
            context["entry_script"],
            context["document"].get("description", ""),
            context["arguments"],
            context["working_directory"],
            job_context=context,
        )
        return build_ok(
            {
                "task_id": metadata["task_id"],
                "task_status": metadata["status"],
                "job_fingerprint": context["fingerprint"],
                "task_directory": metadata["task_directory"],
                "output_directory": metadata["output_directory"],
                "diagnostics_path": metadata["diagnostics_path"],
                "warnings": validation["warnings"],
            }
        )
    except FileNotFoundError as exc:
        return build_error("solver_job_input_not_found", str(exc))
    except (ValueError, NotADirectoryError) as exc:
        return build_error("invalid_solver_job", str(exc))
    except Exception as exc:
        return build_error("solver_job_submission_failed", str(exc))


def geotaichi_check_task_status(
    task_id: str,
    stdout_offset: int = 0,
    stderr_offset: int = 0,
    max_output_chars: int = 2500,
) -> Dict[str, Any]:
    """Poll task metadata and bounded stdout/stderr chunks using byte offsets."""
    try:
        if max_output_chars < 1 or max_output_chars > 3000:
            return build_error("invalid_output_limit", "max_output_chars must be between 1 and 3000 per stream")
        return build_ok(get_task_manager().status(task_id, stdout_offset, stderr_offset, max_output_chars))
    except KeyError:
        return build_error("task_not_found", "Task %r was not found." % task_id)
    except Exception as exc:
        return build_error("task_status_failed", str(exc), {"task_id": task_id})


def geotaichi_list_tasks(skip_newest: int = 0, limit: int = 16) -> Dict[str, Any]:
    """List persistent task history from newest to oldest with pagination."""
    try:
        return build_ok(get_task_manager().list(skip_newest, limit))
    except Exception as exc:
        return build_error("task_list_failed", str(exc))


def geotaichi_list_task_artifacts(task_id: str, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
    """List task-owned contracts, diagnostics, and output artifacts without reading file contents."""
    try:
        return build_ok(get_task_manager().artifacts(task_id, skip, limit))
    except KeyError:
        return build_error("task_not_found", "Task %r was not found." % task_id)
    except Exception as exc:
        return build_error("task_artifact_list_failed", str(exc), {"task_id": task_id})


def geotaichi_interrupt_task(task_id: str, grace_seconds: float = 2.0, force: bool = False) -> Dict[str, Any]:
    """Request process-group termination, optionally forcing a kill after a bounded grace period."""
    try:
        result = get_task_manager().interrupt(task_id, grace_seconds, force)
        if not result["interrupt_requested"]:
            return build_error(
                "task_not_running",
                "Task %r is already in terminal state %s." % (task_id, result["status"]),
                result,
            )
        return build_ok(result)
    except KeyError:
        return build_error("task_not_found", "Task %r was not found." % task_id)
    except Exception as exc:
        return build_error("task_interrupt_failed", str(exc), {"task_id": task_id})
