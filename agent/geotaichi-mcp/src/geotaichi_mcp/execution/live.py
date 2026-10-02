"""Cooperative live execution inside a running GeoTaichi task.

Requests and results are durable JSON files below the task directory.  The
worker drains them only at explicit checkpoints, so snippets run on the model
thread and share the task's ``__main__`` namespace.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, TextIO, Tuple


CONTROL_DIRECTORY = "control"
REQUEST_DIRECTORY = "requests"
RESULT_DIRECTORY = "results"
CANCEL_DIRECTORY = "cancelled"
MAX_CODE_CHARS = 200_000
MAX_CAPTURE_CHARS = 100_000


class LiveExecutionTimeout(BaseException):
    """Raised by the trace guard when Python code exceeds its deadline."""


class CooperativeTaskInterrupted(BaseException):
    """Task-control interruption that must pass through snippet handling."""


class _Tee(io.TextIOBase):
    def __init__(self, original: TextIO, capture: io.StringIO) -> None:
        self.original = original
        self.capture = capture

    @property
    def encoding(self) -> Optional[str]:
        return getattr(self.original, "encoding", None)

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        self.original.write(value)
        remaining = MAX_CAPTURE_CHARS - self.capture.tell()
        if remaining > 0:
            self.capture.write(value[:remaining])
        return len(value)

    def flush(self) -> None:
        self.original.flush()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.%s.tmp" % (path.name, os.getpid(), threading.get_ident()))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("live execution record must be a JSON object")
    return value


def live_directories(task_dir: Path) -> Tuple[Path, Path, Path]:
    control = task_dir / CONTROL_DIRECTORY
    return control / REQUEST_DIRECTORY, control / RESULT_DIRECTORY, control / CANCEL_DIRECTORY


def ensure_live_directories(task_dir: Path) -> Dict[str, str]:
    requests, results, cancelled = live_directories(task_dir)
    for directory in (requests, results, cancelled):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "transport": "filesystem-checkpoint",
        "request_directory": str(requests),
        "result_directory": str(results),
        "cancel_directory": str(cancelled),
    }


def enqueue_live_request(task_dir: Path, code: str, timeout: float) -> Tuple[str, Path]:
    if not code.strip():
        raise ValueError("code must not be empty")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError("code must contain at most %s characters" % MAX_CODE_CHARS)
    paths = ensure_live_directories(task_dir)
    request_id = uuid.uuid4().hex
    now = time.time()
    request = {
        "schema_version": 1,
        "request_id": request_id,
        "code": code,
        "created_at_unix": now,
        "expires_at_unix": now + timeout,
        "execution_timeout_seconds": timeout,
    }
    request_path = Path(paths["request_directory"]) / (request_id + ".json")
    result_path = Path(paths["result_directory"]) / (request_id + ".json")
    _atomic_json(request_path, request)
    return request_id, result_path


def cancel_live_request(task_dir: Path, request_id: str, reason: str) -> None:
    requests, results, cancelled = live_directories(task_dir)
    _atomic_json(cancelled / (request_id + ".json"), {"request_id": request_id, "reason": reason})
    request_path = requests / (request_id + ".json")
    try:
        request_path.unlink()
    except FileNotFoundError:
        pass
    if not (results / (request_id + ".json")).exists():
        _atomic_json(
            results / (request_id + ".json"),
            {
                "schema_version": 1,
                "request_id": request_id,
                "status": "timeout",
                "execution_mode": "live",
                "stdout": "",
                "stderr": "",
                "result": None,
                "error_type": "LiveExecutionTimeout",
                "error_message": reason,
                "elapsed_seconds": 0.0,
                "executed": False,
            },
        )


def read_live_result(path: Path) -> Dict[str, Any]:
    return _read_json(path)


def _bounded_value(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:10_000]
    if depth >= 4:
        return repr(value)[:2_000]
    if isinstance(value, Mapping):
        bounded: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 100:
                bounded["..."] = "truncated"
                break
            bounded[str(key)[:500]] = _bounded_value(item, depth + 1)
        return bounded
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        bounded_items = [_bounded_value(item, depth + 1) for item in items[:100]]
        if len(items) > 100:
            bounded_items.append("... truncated ...")
        return bounded_items
    try:
        return repr(value)[:10_000]
    except Exception:
        return "<unrepresentable %s>" % type(value).__name__


def _compile_snippet(code: str) -> Tuple[Any, str]:
    try:
        return compile(code, "<geotaichi-live>", "eval"), "eval"
    except SyntaxError:
        return compile(code, "<geotaichi-live>", "exec"), "exec"


class LiveExecutionRuntime:
    """Drain queued snippets in one persistent task namespace."""

    def __init__(self, task_dir: Path, namespace: Dict[str, Any]) -> None:
        self.task_dir = task_dir.resolve()
        self.namespace = namespace
        self.paths = ensure_live_directories(self.task_dir)
        self._lock = threading.RLock()
        self._draining = False

    def publish(self, values: Mapping[str, Any]) -> Iterable[str]:
        with self._lock:
            self.namespace.update(values)
        return tuple(sorted(values))

    def checkpoint(self, max_requests: int = 8) -> int:
        if max_requests < 1 or max_requests > 128:
            raise ValueError("max_requests must be between 1 and 128")
        with self._lock:
            if self._draining:
                return 0
            self._draining = True
            try:
                return self._drain(max_requests)
            finally:
                self._draining = False

    def _drain(self, max_requests: int) -> int:
        requests, results, cancelled = live_directories(self.task_dir)
        handled = 0
        for request_path in sorted(requests.glob("*.json")):
            if handled >= max_requests:
                break
            request_id = request_path.stem
            result_path = results / request_path.name
            cancel_path = cancelled / request_path.name
            if result_path.exists() or cancel_path.exists():
                try:
                    request_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            try:
                request = _read_json(request_path)
                if request.get("request_id") != request_id:
                    raise ValueError("request_id does not match its filename")
                if time.time() >= float(request["expires_at_unix"]):
                    result = self._expired_result(request_id)
                else:
                    result = self._execute(request)
            except CooperativeTaskInterrupted:
                raise
            except BaseException as exc:
                result = self._failure_result(request_id, exc)
            if not cancel_path.exists():
                _atomic_json(result_path, result)
            try:
                request_path.unlink()
            except FileNotFoundError:
                pass
            handled += 1
        return handled

    @staticmethod
    def _expired_result(request_id: str) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": request_id,
            "status": "expired",
            "execution_mode": "live",
            "stdout": "",
            "stderr": "",
            "result": None,
            "error_type": "LiveRequestExpired",
            "error_message": "request expired before the task reached a checkpoint",
            "elapsed_seconds": 0.0,
            "executed": False,
        }

    @staticmethod
    def _failure_result(request_id: str, exc: BaseException) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": request_id,
            "status": "error",
            "execution_mode": "live",
            "stdout": "",
            "stderr": "",
            "result": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "elapsed_seconds": 0.0,
            "executed": False,
        }

    def _execute(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = str(request["request_id"])
        code = str(request["code"])
        timeout = min(float(request["execution_timeout_seconds"]), 600.0)
        deadline = time.monotonic() + timeout
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_trace = sys.gettrace()
        started = time.monotonic()

        def deadline_trace(frame: object, event: str, arg: object) -> Any:
            if time.monotonic() >= deadline:
                raise LiveExecutionTimeout("live Python snippet exceeded %.3f seconds" % timeout)
            return deadline_trace

        result_value: Any = None
        status = "completed"
        error_type: Optional[str] = None
        error_message: Optional[str] = None
        error_traceback: Optional[str] = None
        self.namespace.pop("_geotaichi_mcp_result", None)
        try:
            compiled, mode = _compile_snippet(code)
            sys.stdout = _Tee(old_stdout, stdout_capture)
            sys.stderr = _Tee(old_stderr, stderr_capture)
            sys.settrace(deadline_trace)
            if mode == "eval":
                result_value = eval(compiled, self.namespace, self.namespace)
            else:
                exec(compiled, self.namespace, self.namespace)
                result_value = self.namespace.pop("_geotaichi_mcp_result", None)
        except CooperativeTaskInterrupted:
            raise
        except LiveExecutionTimeout as exc:
            status = "timeout"
            error_type = type(exc).__name__
            error_message = str(exc)
        except BaseException as exc:
            status = "error"
            error_type = type(exc).__name__
            error_message = str(exc)
            error_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-20_000:]
        finally:
            sys.settrace(old_trace)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if status != "completed":
                self.namespace.pop("_geotaichi_mcp_result", None)

        response: Dict[str, Any] = {
            "schema_version": 1,
            "request_id": request_id,
            "status": status,
            "execution_mode": "live",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": _bounded_value(result_value),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "executed": True,
        }
        if error_type is not None:
            response["error_type"] = error_type
            response["error_message"] = error_message
        if error_traceback is not None:
            response["traceback"] = error_traceback
        return response


_runtime_lock = threading.RLock()
_runtime: Optional[LiveExecutionRuntime] = None


def install_runtime(task_dir: Path, namespace: Dict[str, Any]) -> LiveExecutionRuntime:
    global _runtime
    runtime = LiveExecutionRuntime(task_dir, namespace)
    with _runtime_lock:
        _runtime = runtime
    return runtime


def uninstall_runtime(runtime: Optional[LiveExecutionRuntime] = None) -> None:
    global _runtime
    with _runtime_lock:
        if runtime is None or _runtime is runtime:
            _runtime = None


def checkpoint(max_requests: int = 8) -> int:
    """Process pending requests at an application-defined safe point."""
    with _runtime_lock:
        runtime = _runtime
    if runtime is None:
        return 0
    return runtime.checkpoint(max_requests)


def publish(**values: Any) -> Iterable[str]:
    """Publish function-local model objects into the live task namespace."""
    with _runtime_lock:
        runtime = _runtime
    if runtime is None:
        return ()
    return runtime.publish(values)
