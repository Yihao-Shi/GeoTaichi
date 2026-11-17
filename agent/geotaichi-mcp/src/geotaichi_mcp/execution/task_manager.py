"""Persistent subprocess task lifecycle for isolated GeoTaichi runs."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..core.config import default_workspace, find_repo_root, resolve_user_path
from ..core.jobs import resolve_solver_job_arguments
from .live import (
    cancel_live_request,
    enqueue_live_request,
    ensure_live_directories,
    read_live_result,
)


ACTIVE_STATUSES = {"pending", "running"}
TERMINAL_STATUSES = {"completed", "failed", "interrupted"}
TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")
METADATA_NAME = "task.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@contextmanager
def _metadata_lock(task_dir: Path) -> Iterator[None]:
    lock_path = task_dir / ".task.lock"
    task_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def read_task_metadata(task_dir: Path) -> Dict[str, Any]:
    with (task_dir / METADATA_NAME).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_task_metadata(task_dir: Path, metadata: Dict[str, Any]) -> None:
    """Atomically replace task metadata while coordinating with the worker."""
    with _metadata_lock(task_dir):
        temporary = task_dir / (".task.%s.%s.tmp" % (os.getpid(), threading.get_ident()))
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(task_dir / METADATA_NAME))


def update_task_metadata(task_dir: Path, **changes: Any) -> Dict[str, Any]:
    """Merge changes into the current task metadata atomically."""
    with _metadata_lock(task_dir):
        metadata_path = task_dir / METADATA_NAME
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        metadata.update(changes)
        temporary = task_dir / (".task.%s.%s.tmp" % (os.getpid(), threading.get_ident()))
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(metadata_path))
        return metadata


def _read_log(path: Path, offset: int, max_bytes: int) -> Dict[str, Any]:
    if offset < 0:
        raise ValueError("log offset must be non-negative")
    if max_bytes < 1 or max_bytes > 20000:
        raise ValueError("max_output_chars must be between 1 and 20000")
    if not path.exists():
        return {"text": "", "offset": offset, "next_offset": offset, "has_more": False, "size": 0}
    size = path.stat().st_size
    start = min(offset, size)
    with path.open("rb") as stream:
        stream.seek(start)
        payload = stream.read(max_bytes)
    next_offset = start + len(payload)
    return {
        "text": payload.decode("utf-8", errors="replace"),
        "offset": start,
        "next_offset": next_offset,
        "has_more": next_offset < size,
        "size": size,
    }


class TaskManager:
    """Launch and supervise GeoTaichi scripts in clean Python processes."""

    def __init__(self, workspace: Optional[Path] = None, repo_root: Optional[Path] = None) -> None:
        self.repo_root = repo_root or find_repo_root(required=False)
        if self.repo_root is not None:
            self.repo_root = self.repo_root.resolve()
        self.workspace = (workspace or default_workspace(self.repo_root)).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._processes: Dict[str, subprocess.Popen] = {}
        self._process_lock = threading.RLock()

    def _task_dir(self, task_id: str) -> Path:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("task_id must be a 12-character lowercase hexadecimal identifier")
        return self.workspace / task_id

    def _load(self, task_id: str) -> Tuple[Path, Dict[str, Any]]:
        task_dir = self._task_dir(task_id)
        if not (task_dir / METADATA_NAME).is_file():
            raise KeyError(task_id)
        return task_dir, read_task_metadata(task_dir)

    def _base_environment(self) -> Dict[str, str]:
        environment = os.environ.copy()
        if self.repo_root is not None:
            existing = environment.get("PYTHONPATH", "")
            entries = [str(self.repo_root)]
            if existing:
                entries.append(existing)
            environment["PYTHONPATH"] = os.pathsep.join(entries)
        environment.setdefault("PYTHONUNBUFFERED", "1")
        return environment

    def _resolve_working_directory(self, value: Optional[str], script: Path) -> Path:
        if value:
            directory = resolve_user_path(value, self.repo_root, must_exist=True)
        elif self.repo_root is not None:
            directory = self.repo_root
        else:
            directory = script.parent
        if not directory.is_dir():
            raise NotADirectoryError(str(directory))
        return directory

    def submit(
        self,
        entry_script: str,
        description: str = "",
        arguments: Optional[Sequence[str]] = None,
        working_directory: Optional[str] = None,
        job_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit a script and return persistent task metadata immediately."""
        script = resolve_user_path(entry_script, self.repo_root, must_exist=True)
        if not script.is_file():
            raise FileNotFoundError(str(script))
        if script.suffix.lower() != ".py":
            raise ValueError("entry_script must name a Python .py file")
        cwd = self._resolve_working_directory(working_directory, script)
        source_arguments = [str(item) for item in (arguments or [])]
        normalized_description = description.strip()
        if len(normalized_description) > 2000:
            raise ValueError("description must contain at most 2000 characters")
        if len(source_arguments) > 64:
            raise ValueError("arguments must contain at most 64 entries")
        if any(len(item) > 2048 for item in source_arguments):
            raise ValueError("each task argument must contain at most 2048 characters")

        task_id = uuid.uuid4().hex[:12]
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=False, exist_ok=False)
        output_dir = task_dir / "output"
        output_dir.mkdir()
        stdout_path = task_dir / "stdout.log"
        stderr_path = task_dir / "stderr.log"
        stdout_path.touch()
        stderr_path.touch()
        live_paths = ensure_live_directories(task_dir)

        solver_job_metadata = None
        staged_job_path = task_dir / "solver-job.json" if job_context is not None else None
        staged_scene_path = (
            task_dir / "scene-manifest.json"
            if job_context is not None and job_context.get("scene_document") is not None
            else None
        )
        staged_contract_path = (
            task_dir / "model-contract.json"
            if job_context is not None and job_context.get("contract_document") is not None
            else None
        )
        args = source_arguments
        if staged_job_path is not None:
            args = resolve_solver_job_arguments(
                source_arguments,
                solver_job_path=staged_job_path,
                output_directory=output_dir,
                scene_manifest_path=staged_scene_path,
                model_contract_path=staged_contract_path,
            )
        if any(len(item) > 2048 for item in args):
            raise ValueError("each resolved task argument must contain at most 2048 characters")
        if job_context is not None:
            staged_document = deepcopy(job_context["document"])
            if job_context.get("contract_document") is not None:
                with staged_contract_path.open("w", encoding="utf-8") as stream:
                    json.dump(job_context["contract_document"], stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
                staged_document["model"]["contract_path"] = staged_contract_path.name
            if job_context.get("scene_document") is not None:
                with staged_scene_path.open("w", encoding="utf-8") as stream:
                    json.dump(job_context["scene_document"], stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
                staged_document.setdefault("scene", {})["manifest_path"] = staged_scene_path.name
            staged_document["model"]["arguments"] = args
            with staged_job_path.open("w", encoding="utf-8") as stream:
                json.dump(staged_document, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            solver_job_metadata = {
                "path": str(staged_job_path),
                "source_path": job_context.get("source_path"),
                "fingerprint": job_context["fingerprint"],
                "scene_manifest_path": str(staged_scene_path) if staged_scene_path else None,
                "scene_fingerprint": job_context.get("scene_fingerprint"),
                "model_contract_path": str(staged_contract_path) if staged_contract_path else None,
                "model_contract_fingerprint": job_context.get("contract_fingerprint"),
            }

        metadata: Dict[str, Any] = {
            "schema_version": 3,
            "task_id": task_id,
            "status": "pending",
            "entry_script": str(script),
            "description": normalized_description,
            "arguments": args,
            "working_directory": str(cwd),
            "task_directory": str(task_dir),
            "output_directory": str(output_dir),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "created_at": utc_now(),
            "started_at": None,
            "ended_at": None,
            "pid": None,
            "return_code": None,
            "live_execution": {
                **live_paths,
                "enabled": False,
                "namespace": "__main__",
                "checkpoint_mode": "cooperative",
            },
            "solver_job": solver_job_metadata,
            "diagnostics_path": str(task_dir / "diagnostics.json"),
            "physics_score_path": str(task_dir / "physics-score.json"),
        }
        write_task_metadata(task_dir, metadata)

        worker_path = Path(__file__).with_name("worker.py")
        command = [
            sys.executable,
            "-u",
            str(worker_path),
            "--task-dir",
            str(task_dir),
            "--",
            str(script),
            *args,
        ]
        environment = self._base_environment()
        environment["GEOTAICHI_MCP_TASK_ID"] = task_id
        environment["GEOTAICHI_MCP_TASK_DIR"] = str(task_dir)
        environment["GEOTAICHI_DIAGNOSTICS_PATH"] = str(task_dir / "diagnostics.json")
        uses_output_argument = staged_job_path is not None and "{output_directory}" in source_arguments
        if uses_output_argument:
            environment.pop("GEOTAICHI_MCP_OUTPUT_DIR", None)
        else:
            environment["GEOTAICHI_MCP_OUTPUT_DIR"] = str(output_dir)
        if staged_job_path is not None:
            environment["GEOTAICHI_SOLVER_JOB"] = str(staged_job_path)
        if staged_scene_path is not None:
            if "{scene_manifest}" in source_arguments:
                environment.pop("GEOTAICHI_SCENE_MANIFEST", None)
            else:
                environment["GEOTAICHI_SCENE_MANIFEST"] = str(staged_scene_path)
        if staged_contract_path is not None:
            if "{model_contract}" in source_arguments:
                environment.pop("GEOTAICHI_MODEL_CONTRACT", None)
            else:
                environment["GEOTAICHI_MODEL_CONTRACT"] = str(staged_contract_path)

        stdout_stream = stdout_path.open("ab", buffering=0)
        stderr_stream = stderr_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=environment,
                stdout=stdout_stream,
                stderr=stderr_stream,
                start_new_session=(os.name != "nt"),
            )
        except Exception:
            stdout_stream.close()
            stderr_stream.close()
            update_task_metadata(task_dir, status="failed", ended_at=utc_now())
            raise

        update_task_metadata(task_dir, pid=process.pid, command=command)
        with self._process_lock:
            self._processes[task_id] = process
        watcher = threading.Thread(
            target=self._watch_process,
            args=(task_id, process, stdout_stream, stderr_stream),
            name="geotaichi-mcp-%s" % task_id,
            daemon=True,
        )
        watcher.start()
        return read_task_metadata(task_dir)

    def _watch_process(self, task_id: str, process: subprocess.Popen, stdout_stream: Any, stderr_stream: Any) -> None:
        return_code = process.wait()
        stdout_stream.close()
        stderr_stream.close()
        with self._process_lock:
            self._processes.pop(task_id, None)
        task_dir = self._task_dir(task_id)
        try:
            metadata = read_task_metadata(task_dir)
            if metadata.get("status") in ACTIVE_STATUSES:
                if metadata.get("interrupt_requested_at"):
                    status = "interrupted"
                else:
                    status = "completed" if return_code == 0 else "failed"
                update_task_metadata(
                    task_dir,
                    status=status,
                    ended_at=utc_now(),
                    return_code=return_code,
                )
        except (FileNotFoundError, json.JSONDecodeError):
            return

    def _is_alive(self, task_id: str, pid: Optional[int]) -> bool:
        if not pid:
            return False
        with self._process_lock:
            process = self._processes.get(task_id)
            if process is not None:
                return process.poll() is None
        try:
            import psutil

            candidate = psutil.Process(int(pid))
            if not candidate.is_running():
                return False
            command = " ".join(candidate.cmdline())
            return "worker.py" in command and task_id in command
        except Exception:
            return False

    def _reconcile(self, task_dir: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if metadata.get("status") not in ACTIVE_STATUSES:
            return metadata
        if self._is_alive(metadata["task_id"], metadata.get("pid")):
            return metadata
        status = "interrupted" if metadata.get("interrupt_requested_at") else "failed"
        return update_task_metadata(
            task_dir,
            status=status,
            ended_at=metadata.get("ended_at") or utc_now(),
            failure_reason=metadata.get("failure_reason") or "worker process exited without a terminal metadata update",
        )

    def status(
        self,
        task_id: str,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        max_output_chars: int = 4000,
    ) -> Dict[str, Any]:
        task_dir, metadata = self._load(task_id)
        metadata = self._reconcile(task_dir, metadata)
        visible_metadata = {key: value for key, value in metadata.items() if key != "command"}
        diagnostics = None
        diagnostics_path = Path(metadata.get("diagnostics_path") or task_dir / "diagnostics.json")
        if diagnostics_path.is_file():
            try:
                with diagnostics_path.open("r", encoding="utf-8") as stream:
                    diagnostics = json.load(stream)
            except (OSError, json.JSONDecodeError):
                diagnostics = {"schema_version": 1, "status": "unreadable"}
        physics_score = None
        physics_score_path = Path(metadata.get("physics_score_path") or task_dir / "physics-score.json")
        if physics_score_path.is_file():
            try:
                with physics_score_path.open("r", encoding="utf-8") as stream:
                    physics_score = json.load(stream)
            except (OSError, json.JSONDecodeError):
                physics_score = {"schema_version": 1, "status": "unreadable"}
        return {
            "task": visible_metadata,
            "stdout": _read_log(Path(metadata["stdout_path"]), stdout_offset, max_output_chars),
            "stderr": _read_log(Path(metadata["stderr_path"]), stderr_offset, max_output_chars),
            "diagnostics": diagnostics,
            "physics_score": physics_score,
        }

    def artifacts(self, task_id: str, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """List bounded task-owned artifacts without reading their contents."""
        if skip < 0:
            raise ValueError("skip must be non-negative")
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        task_dir, metadata = self._load(task_id)
        metadata = self._reconcile(task_dir, metadata)
        candidates = []
        for name in (
            "solver-job.json",
            "scene-manifest.json",
            "model-contract.json",
            "diagnostics.json",
            "physics-score.json",
        ):
            path = task_dir / name
            if path.is_file():
                candidates.append(path)
        output_dir = task_dir / "output"
        if output_dir.is_dir():
            candidates.extend(
                path
                for path in output_dir.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and not any(part.startswith("._") for part in path.relative_to(output_dir).parts)
            )
        candidates.sort(key=lambda path: path.relative_to(task_dir).as_posix())
        page = candidates[skip : skip + limit]
        records = []
        for path in page:
            stat = path.stat()
            records.append(
                {
                    "relative_path": path.relative_to(task_dir).as_posix(),
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat().replace("+00:00", "Z"),
                }
            )
        return {
            "task_id": task_id,
            "task_status": metadata.get("status"),
            "task_directory": str(task_dir),
            "total_count": len(candidates),
            "displayed_count": len(records),
            "has_more": skip + len(records) < len(candidates),
            "artifacts": records,
        }

    def list(self, skip_newest: int = 0, limit: int = 32) -> Dict[str, Any]:
        if skip_newest < 0:
            raise ValueError("skip_newest must be non-negative")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        records: List[Dict[str, Any]] = []
        for metadata_path in self.workspace.glob("*/%s" % METADATA_NAME):
            try:
                task_dir = metadata_path.parent
                metadata = self._reconcile(task_dir, read_task_metadata(task_dir))
                records.append(metadata)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                continue
        records.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        page = records[skip_newest : skip_newest + limit]
        summary_fields = (
            "task_id",
            "status",
            "entry_script",
            "description",
            "created_at",
            "started_at",
            "ended_at",
            "pid",
            "return_code",
            "output_directory",
        )
        summaries = [{key: item.get(key) for key in summary_fields} for item in page]
        return {
            "total_count": len(records),
            "displayed_count": len(summaries),
            "has_more": skip_newest + len(summaries) < len(records),
            "tasks": summaries,
        }

    def interrupt(self, task_id: str, grace_seconds: float = 2.0, force: bool = False) -> Dict[str, Any]:
        if grace_seconds < 0 or grace_seconds > 30:
            raise ValueError("grace_seconds must be between 0 and 30")
        task_dir, metadata = self._load(task_id)
        metadata = self._reconcile(task_dir, metadata)
        if metadata.get("status") in TERMINAL_STATUSES:
            return {"task_id": task_id, "interrupt_requested": False, "status": metadata["status"]}
        pid = metadata.get("pid")
        if not self._is_alive(task_id, pid):
            metadata = self._reconcile(task_dir, metadata)
            return {"task_id": task_id, "interrupt_requested": False, "status": metadata["status"]}

        requested_at = utc_now()
        update_task_metadata(task_dir, interrupt_requested_at=requested_at)
        self._signal(task_id, int(pid), signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while grace_seconds and time.monotonic() < deadline and self._is_alive(task_id, pid):
            time.sleep(0.05)
        forced = False
        if force and self._is_alive(task_id, pid):
            kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
            self._signal(task_id, int(pid), kill_signal)
            forced = True
        return {
            "task_id": task_id,
            "interrupt_requested": True,
            "forced": forced,
            "status": self.status(task_id, max_output_chars=1)["task"]["status"],
        }

    def _signal(self, task_id: str, pid: int, requested_signal: signal.Signals) -> None:
        if not self._is_alive(task_id, pid):
            return
        if os.name != "nt":
            os.killpg(os.getpgid(pid), requested_signal)
            return
        with self._process_lock:
            process = self._processes.get(task_id)
        if process is not None:
            if requested_signal == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()

    def execute_code(
        self,
        code: str,
        timeout: float = 10.0,
        working_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an isolated Python snippet synchronously."""
        if not code.strip():
            raise ValueError("code must not be empty")
        if timeout <= 0 or timeout > 600:
            raise ValueError("timeout must be greater than 0 and at most 600 seconds")
        if working_directory:
            cwd = resolve_user_path(working_directory, self.repo_root, must_exist=True)
        else:
            cwd = self.repo_root or Path.cwd()
        if not cwd.is_dir():
            raise NotADirectoryError(str(cwd))
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, "-u", "-c", code],
                cwd=str(cwd),
                env=self._base_environment(),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "timeout",
                "execution_mode": "isolated",
                "return_code": None,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "stdout": (
                    (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
                ),
                "stderr": (
                    (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
                ),
            }
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "execution_mode": "isolated",
            "return_code": completed.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def execute_code_in_task(self, task_id: str, code: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Execute *code* in a running task when it reaches a safe checkpoint."""
        if not code.strip():
            raise ValueError("code must not be empty")
        if timeout <= 0 or timeout > 600:
            raise ValueError("timeout must be greater than 0 and at most 600 seconds")
        task_dir, metadata = self._load(task_id)
        metadata = self._reconcile(task_dir, metadata)
        if metadata.get("status") not in ACTIVE_STATUSES:
            raise RuntimeError("task %s is not running (status: %s)" % (task_id, metadata.get("status")))
        live = metadata.get("live_execution") or {}
        if metadata.get("status") == "running" and not live.get("enabled"):
            raise RuntimeError("task %s does not have an active live execution runtime" % task_id)

        request_id, result_path = enqueue_live_request(task_dir, code, timeout)
        started = time.monotonic()
        deadline = started + timeout + min(0.25, timeout * 0.1)
        while time.monotonic() < deadline:
            if result_path.is_file():
                result = read_live_result(result_path)
                result["task_id"] = task_id
                return result
            current = self._reconcile(task_dir, read_task_metadata(task_dir))
            if current.get("status") in TERMINAL_STATUSES:
                cancel_live_request(
                    task_dir,
                    request_id,
                    "task reached terminal status %s before handling the request" % current.get("status"),
                )
                result = read_live_result(result_path)
                result["task_id"] = task_id
                result["task_status"] = current.get("status")
                return result
            time.sleep(0.01)

        cancel_live_request(
            task_dir,
            request_id,
            "task did not reach a cooperative checkpoint within %.3f seconds" % timeout,
        )
        result = read_live_result(result_path)
        result["task_id"] = task_id
        result["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return result
