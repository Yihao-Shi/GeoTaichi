"""Subprocess adapter for the shared ``geotaichi-job`` JSON CLI."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List


class JobClientError(RuntimeError):
    pass


class JobClient:
    def __init__(self, python_executable: str, repo_root: str = "", workspace: str = "") -> None:
        if not python_executable.strip():
            raise ValueError("a Python executable with GeoTaichi installed is required")
        self.python_executable = python_executable
        self.repo_root = repo_root
        self.workspace = workspace

    def _base_command(self) -> List[str]:
        command = [self.python_executable, "-m", "geotaichi_mcp.job_cli"]
        if self.repo_root:
            command.extend(("--repo-root", self.repo_root))
        if self.workspace:
            command.extend(("--workspace", self.workspace))
        return command

    def call(self, arguments: List[str], timeout: float = 30.0) -> Dict[str, Any]:
        completed = subprocess.run(
            [*self._base_command(), *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            detail = completed.stderr.strip() or "job CLI returned no JSON response"
            raise JobClientError(detail)
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise JobClientError("job CLI returned invalid JSON: %s" % lines[-1]) from exc
        if not isinstance(payload, dict):
            raise JobClientError("job CLI response must be a JSON object")
        if not payload.get("ok"):
            error = payload.get("error") or {}
            raise JobClientError(str(error.get("message") or "GeoTaichi job operation failed"))
        return payload["data"]

    def validate(self, job_path: str) -> Dict[str, Any]:
        return self.call(["validate", job_path])

    def submit(self, job_path: str, confirm_trusted_execution: bool) -> Dict[str, Any]:
        arguments = ["submit", job_path]
        if confirm_trusted_execution:
            arguments.append("--confirm-trusted-execution")
        return self.call(arguments)

    def status(self, task_id: str) -> Dict[str, Any]:
        return self.call(["status", task_id, "--max-output-chars", "3000"])

    def cancel(self, task_id: str) -> Dict[str, Any]:
        return self.call(["cancel", task_id])

    def artifacts(self, task_id: str) -> Dict[str, Any]:
        records = []
        skip = 0
        task_directory = ""
        task_status = ""
        while True:
            page = self.call(
                [
                    "artifacts",
                    task_id,
                    "--skip",
                    str(skip),
                    "--limit",
                    "10",
                ]
            )
            task_directory = str(page.get("task_directory", task_directory))
            task_status = str(page.get("task_status", task_status))
            current = list(page.get("artifacts") or [])
            records.extend(current)
            skip += len(current)
            if not page.get("has_more"):
                return {
                    "task_id": task_id,
                    "task_status": task_status,
                    "task_directory": task_directory,
                    "total_count": len(records),
                    "displayed_count": len(records),
                    "has_more": False,
                    "artifacts": records,
                }
            if not current:
                raise JobClientError("artifact pagination made no progress")
