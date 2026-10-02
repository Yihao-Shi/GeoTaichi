"""Stable public tool facade and FastMCP registration."""

from __future__ import annotations

from typing import Any

from .execution.tools import (
    configure_task_manager,
    geotaichi_check_task_status,
    geotaichi_execute_code,
    geotaichi_execute_task,
    geotaichi_interrupt_task,
    geotaichi_list_task_artifacts,
    geotaichi_list_tasks,
    geotaichi_submit_solver_job,
    geotaichi_validate_solver_job,
    get_task_manager,
)
from .knowledge.tools import (
    geotaichi_audit_api_docs,
    geotaichi_browse_capabilities,
    geotaichi_get_model_template,
    geotaichi_inspect_model,
    geotaichi_query_capabilities,
    geotaichi_review_model,
    geotaichi_score_physics,
)


READ_ONLY_TOOLS = (
    geotaichi_browse_capabilities,
    geotaichi_query_capabilities,
    geotaichi_get_model_template,
    geotaichi_inspect_model,
    geotaichi_score_physics,
    geotaichi_review_model,
    geotaichi_audit_api_docs,
    geotaichi_validate_solver_job,
    geotaichi_check_task_status,
    geotaichi_list_tasks,
    geotaichi_list_task_artifacts,
)

LIFECYCLE_TOOLS = (
    geotaichi_interrupt_task,
)

TRUSTED_EXECUTION_TOOLS = (
    geotaichi_submit_solver_job,
    geotaichi_execute_task,
    geotaichi_execute_code,
)

PUBLIC_TOOLS = READ_ONLY_TOOLS + LIFECYCLE_TOOLS + TRUSTED_EXECUTION_TOOLS


TOOL_ANNOTATIONS = {
    **{
        tool.__name__: {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
        for tool in READ_ONLY_TOOLS
    },
    geotaichi_interrupt_task.__name__: {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    geotaichi_submit_solver_job.__name__: {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
    geotaichi_execute_task.__name__: {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
    geotaichi_execute_code.__name__: {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
}


def register_tools(mcp: Any, profile: str = "trusted") -> None:
    """Register a read/lifecycle-only safe profile or the full trusted profile."""
    if profile not in {"safe", "trusted"}:
        raise ValueError("tool profile must be 'safe' or 'trusted'")
    selected = READ_ONLY_TOOLS + LIFECYCLE_TOOLS
    if profile == "trusted":
        selected += TRUSTED_EXECUTION_TOOLS
    for tool in selected:
        mcp.tool(annotations=TOOL_ANNOTATIONS[tool.__name__])(tool)


__all__ = [
    "PUBLIC_TOOLS",
    "READ_ONLY_TOOLS",
    "TRUSTED_EXECUTION_TOOLS",
    "configure_task_manager",
    "geotaichi_audit_api_docs",
    "geotaichi_browse_capabilities",
    "geotaichi_check_task_status",
    "geotaichi_execute_code",
    "geotaichi_execute_task",
    "geotaichi_get_model_template",
    "geotaichi_inspect_model",
    "geotaichi_interrupt_task",
    "geotaichi_list_task_artifacts",
    "geotaichi_list_tasks",
    "geotaichi_query_capabilities",
    "geotaichi_review_model",
    "geotaichi_score_physics",
    "geotaichi_submit_solver_job",
    "geotaichi_validate_solver_job",
    "get_task_manager",
    "register_tools",
]
