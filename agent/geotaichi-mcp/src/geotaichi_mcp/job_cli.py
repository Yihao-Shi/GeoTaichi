"""JSON CLI for SolverJob validation and local task lifecycle operations."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from .execution.task_manager import TaskManager
from .execution.tools import (
    configure_task_manager,
    geotaichi_check_task_status,
    geotaichi_interrupt_task,
    geotaichi_list_task_artifacts,
    geotaichi_list_tasks,
    geotaichi_submit_solver_job,
    geotaichi_validate_solver_job,
)


def _emit(payload: Dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=False))
    return 0 if payload.get("ok") else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="geotaichi-job", description=__doc__)
    parser.add_argument("--repo-root", help="GeoTaichi checkout; otherwise auto-detected")
    parser.add_argument("--workspace", help="persistent task workspace")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate and resolve a SolverJob")
    validate.add_argument("job_path")

    submit = subparsers.add_parser("submit", help="submit a trusted SolverJob")
    submit.add_argument("job_path")
    submit.add_argument("--confirm-trusted-execution", action="store_true")

    status = subparsers.add_parser("status", help="read task status and bounded logs")
    status.add_argument("task_id")
    status.add_argument("--stdout-offset", type=int, default=0)
    status.add_argument("--stderr-offset", type=int, default=0)
    status.add_argument("--max-output-chars", type=int, default=2500)

    listing = subparsers.add_parser("list", help="list persistent tasks")
    listing.add_argument("--skip-newest", type=int, default=0)
    listing.add_argument("--limit", type=int, default=16)

    cancel = subparsers.add_parser("cancel", help="interrupt a running task")
    cancel.add_argument("task_id")
    cancel.add_argument("--grace-seconds", type=float, default=2.0)
    cancel.add_argument("--force", action="store_true")

    artifacts = subparsers.add_parser("artifacts", help="list task-owned output artifacts")
    artifacts.add_argument("task_id")
    artifacts.add_argument("--skip", type=int, default=0)
    artifacts.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repo_root:
        os.environ["GEOTAICHI_REPO_ROOT"] = os.path.abspath(os.path.expanduser(args.repo_root))
    if args.workspace:
        os.environ["GEOTAICHI_MCP_WORKSPACE"] = os.path.abspath(os.path.expanduser(args.workspace))
    if args.command == "validate":
        return _emit(geotaichi_validate_solver_job(args.job_path))
    configure_task_manager(TaskManager())
    if args.command == "submit":
        return _emit(geotaichi_submit_solver_job(args.job_path, args.confirm_trusted_execution))
    if args.command == "status":
        return _emit(
            geotaichi_check_task_status(
                args.task_id,
                args.stdout_offset,
                args.stderr_offset,
                args.max_output_chars,
            )
        )
    if args.command == "list":
        return _emit(geotaichi_list_tasks(args.skip_newest, args.limit))
    if args.command == "cancel":
        return _emit(geotaichi_interrupt_task(args.task_id, args.grace_seconds, args.force))
    if args.command == "artifacts":
        return _emit(geotaichi_list_task_artifacts(args.task_id, args.skip, args.limit))
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
