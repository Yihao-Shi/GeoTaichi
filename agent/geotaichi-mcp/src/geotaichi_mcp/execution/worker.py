"""Worker process that runs one model and persists its terminal status."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tokenize
import traceback
import types
from pathlib import Path
from typing import List, Optional


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geotaichi_mcp.execution.task_manager import read_task_metadata, update_task_metadata, utc_now
from geotaichi_mcp.execution.live import CooperativeTaskInterrupted, install_runtime, uninstall_runtime


class TaskInterrupted(CooperativeTaskInterrupted):
    pass


def _handle_interrupt(signum: int, frame: object) -> None:
    raise TaskInterrupted("received signal %s" % signum)


def _exit_code(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    print(value, file=sys.stderr)
    return 1


def _json_safe(value: object, depth: int = 0) -> object:
    """Convert bounded solver diagnostics to JSON-compatible values."""
    if depth > 5:
        return "<maximum diagnostic depth reached>"
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        if value != value:
            return "NaN"
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in list(value.items())[:100]}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in list(value)[:100]]
    return str(value)


def _solver_diagnostics(namespace: dict) -> Optional[dict]:
    """Find the public/engine diagnostic surface without importing a solver."""
    for name in (
        "model",
        "coupling",
        "solver",
        "engine",
        "fempm",
        "fedem",
        "igampm",
        "dempm",
        "mpdem",
        "problem",
        "simulation",
        "mpm",
        "dem",
        "fem",
        "iga",
    ):
        owner = namespace.get(name)
        if owner is None:
            continue
        snapshot = getattr(owner, "diagnostics_snapshot", None)
        if callable(snapshot):
            try:
                return _json_safe(snapshot())  # type: ignore[return-value]
            except Exception as exc:
                return {"status": "unavailable", "error": "%s: %s" % (type(exc).__name__, exc)}
        engine = getattr(owner, "enginer", None) or getattr(owner, "engine", None) or owner
        snapshot = getattr(engine, "diagnostics_snapshot", None)
        if callable(snapshot):
            try:
                return _json_safe(snapshot())  # type: ignore[return-value]
            except Exception as exc:
                return {"status": "unavailable", "error": "%s: %s" % (type(exc).__name__, exc)}
        last_failure = getattr(engine, "last_failure", None)
        if last_failure is not None:
            return {"last_failure": _json_safe(last_failure)}
    return None


def _write_diagnostics(
    task_dir: Path,
    metadata: dict,
    status: str,
    return_code: int,
    failure_reason: Optional[str],
    failure: Optional[BaseException],
    namespace: dict,
) -> dict:
    """Atomically persist one task-level diagnostic summary."""
    payload = {
        "schema_version": 1,
        "task_id": metadata.get("task_id"),
        "status": status,
        "return_code": return_code,
        "ended_at": utc_now(),
        "failure": None,
        "solver": _solver_diagnostics(namespace),
    }
    if failure is not None or failure_reason:
        payload["failure"] = {
            "kind": type(failure).__name__ if failure is not None else "TaskFailure",
            "message": str(failure) if failure is not None else failure_reason,
        }
    path = Path(os.environ.get("GEOTAICHI_DIAGNOSTICS_PATH", str(task_dir / "diagnostics.json")))
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
    return payload


def _write_physics_score(
    task_dir: Path,
    metadata: dict,
    status: str,
    diagnostics: dict,
    namespace: dict,
) -> None:
    """Score staged validation evidence without letting review I/O change task status."""
    solver_job = metadata.get("solver_job") or {}
    contract_value = solver_job.get("model_contract_path") or os.environ.get("GEOTAICHI_MODEL_CONTRACT")
    if not contract_value:
        return
    contract_path = Path(contract_value)
    if not contract_path.is_file():
        return

    evidence = namespace.get("validation_evidence")
    evidence_source = "namespace:validation_evidence"
    if not isinstance(evidence, dict):
        candidates = [task_dir / "output" / "validation-evidence.json", task_dir / "validation-evidence.json"]
        evidence = {}
        evidence_source = "none"
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                with candidate.open("r", encoding="utf-8") as stream:
                    loaded = json.load(stream)
                if isinstance(loaded, dict):
                    evidence = loaded
                    evidence_source = str(candidate)
                    break
            except (OSError, json.JSONDecodeError):
                continue
    evidence = dict(evidence)
    evidence.setdefault("task_status", status)
    evidence.setdefault("solver_diagnostics", diagnostics.get("solver"))

    path = task_dir / "physics-score.json"
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    try:
        from geotaichi_mcp.core.resources import load_physics_validation_rubric
        from geotaichi_mcp.knowledge.physics_validation import score_physics_validation

        with contract_path.open("r", encoding="utf-8") as stream:
            contract = json.load(stream)
        score = score_physics_validation(contract, evidence, load_physics_validation_rubric())
        payload = {
            "schema_version": 1,
            "task_id": metadata.get("task_id"),
            "contract_path": str(contract_path),
            "evidence_source": evidence_source,
            "score": score,
        }
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    except Exception as exc:
        # Scoring is a terminal review artifact. Its failure must be visible but
        # must not replace the already determined numerical task status.
        payload = {
            "schema_version": 1,
            "task_id": metadata.get("task_id"),
            "status": "unavailable",
            "error": {"kind": type(exc).__name__, "message": str(exc)},
        }
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(str(temporary), str(path))
        except OSError:
            try:
                temporary.unlink()
            except OSError:
                pass


def run_task(task_dir: Path, entry_script: Path, arguments: List[str]) -> int:
    signal.signal(signal.SIGTERM, _handle_interrupt)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handle_interrupt)
    metadata = read_task_metadata(task_dir)
    sys.argv = [str(entry_script), *arguments]
    status = "completed"
    return_code = 0
    failure_reason: Optional[str] = None
    failure: Optional[BaseException] = None
    task_module = types.ModuleType("__main__")
    namespace = task_module.__dict__
    namespace.update(
        {
            "__file__": str(entry_script),
            "__cached__": None,
            "__package__": None,
        }
    )
    previous_main = sys.modules.get("__main__")
    sys.modules["__main__"] = task_module
    runtime = install_runtime(task_dir, namespace)
    hook_callback = runtime.checkpoint
    hook_available = False
    try:
        from src.utils.RuntimeHook import set_runtime_checkpoint

        set_runtime_checkpoint(hook_callback)
        hook_available = True
    except ImportError:
        pass
    live_execution = {
        **runtime.paths,
        "enabled": True,
        "namespace": "__main__",
        "checkpoint_mode": "cooperative",
        "solver_hook_available": hook_available,
    }
    update_task_metadata(
        task_dir,
        status="running",
        pid=os.getpid(),
        started_at=metadata.get("started_at") or utc_now(),
        live_execution=live_execution,
    )
    try:
        with tokenize.open(str(entry_script)) as stream:
            source = stream.read()
        compiled = compile(source, str(entry_script), "exec")
        exec(compiled, namespace, namespace)
    except TaskInterrupted as exc:
        failure = exc
        status = "interrupted"
        return_code = 130
        failure_reason = str(exc)
        print("GeoTaichi MCP task interrupted: %s" % exc, file=sys.stderr)
    except SystemExit as exc:
        return_code = _exit_code(exc.code)
        if return_code != 0:
            failure = exc
            status = "failed"
            failure_reason = "model exited with status %s" % return_code
    except BaseException as exc:
        failure = exc
        status = "failed"
        return_code = 1
        failure_reason = "%s: %s" % (type(exc).__name__, exc)
        traceback.print_exc()
    finally:
        if hook_available:
            try:
                from src.utils.RuntimeHook import clear_runtime_checkpoint

                clear_runtime_checkpoint(hook_callback)
            except ImportError:
                pass
        uninstall_runtime(runtime)
        if previous_main is not None:
            sys.modules["__main__"] = previous_main
        else:
            sys.modules.pop("__main__", None)
        live_execution["enabled"] = False
        diagnostics = _write_diagnostics(
            task_dir,
            metadata,
            status,
            return_code,
            failure_reason,
            failure,
            namespace,
        )
        _write_physics_score(task_dir, metadata, status, diagnostics, namespace)
        update_task_metadata(
            task_dir,
            status=status,
            ended_at=utc_now(),
            return_code=return_code,
            failure_reason=failure_reason,
            live_execution=live_execution,
        )
    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("remainder", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.remainder and args.remainder[0] == "--":
        args.remainder = args.remainder[1:]
    if not args.remainder:
        parser.error("an entry script is required after --")
    return args


def main() -> int:
    args = parse_args()
    entry_script = Path(args.remainder[0]).resolve()
    return run_task(args.task_dir.resolve(), entry_script, list(args.remainder[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
