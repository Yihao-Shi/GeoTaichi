import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from geotaichi_mcp.execution.task_manager import TaskManager
from src.utils.RuntimeHook import clear_runtime_checkpoint, runtime_checkpoint, set_runtime_checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def wait_for_status(manager, task_id, expected, timeout=10.0):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = manager.status(task_id, max_output_chars=20000)
        if latest["task"]["status"] in expected:
            return latest
        time.sleep(0.02)
    raise AssertionError("task did not reach %s: %s" % (expected, latest))


def wait_for_output(manager, task_id, text, timeout=10.0):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = manager.status(task_id, max_output_chars=20000)
        if text in latest["stdout"]["text"]:
            return latest
        time.sleep(0.02)
    raise AssertionError("task did not print %r: %s" % (text, latest))


def test_background_task_persists_logs_outputs_and_history(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(
        """\
import os
import pathlib
import sys

print("model-start", flush=True)
print("diagnostic", file=sys.stderr, flush=True)
output = pathlib.Path(os.environ["GEOTAICHI_MCP_OUTPUT_DIR"])
(output / "result.marker").write_text(os.environ["GEOTAICHI_MCP_TASK_ID"], encoding="utf-8")
""",
        encoding="utf-8",
    )
    workspace = tmp_path / "tasks"
    manager = TaskManager(workspace=workspace, repo_root=REPOSITORY_ROOT)

    submitted = manager.submit(str(script), "unit task")
    result = wait_for_status(manager, submitted["task_id"], {"completed"})

    assert result["task"]["return_code"] == 0
    assert "model-start" in result["stdout"]["text"]
    assert "diagnostic" in result["stderr"]["text"]
    marker = Path(result["task"]["output_directory"]) / "result.marker"
    assert marker.read_text(encoding="utf-8") == submitted["task_id"]
    assert result["diagnostics"]["status"] == "completed"
    assert result["diagnostics"]["solver"] is None
    artifacts = manager.artifacts(submitted["task_id"])
    assert {item["relative_path"] for item in artifacts["artifacts"]} == {
        "diagnostics.json",
        "output/result.marker",
    }

    recovered = TaskManager(workspace=workspace, repo_root=REPOSITORY_ROOT)
    history = recovered.list()
    assert history["total_count"] == 1
    assert history["tasks"][0]["status"] == "completed"
    assert "command" not in history["tasks"][0]


def test_task_interrupt_updates_terminal_status(tmp_path):
    script = tmp_path / "long_model.py"
    script.write_text(
        """\
import time

print("ready", flush=True)
while True:
    time.sleep(0.05)
""",
        encoding="utf-8",
    )
    manager = TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT)
    submitted = manager.submit(str(script), "interrupt unit task")
    wait_for_output(manager, submitted["task_id"], "ready")

    interruption = manager.interrupt(submitted["task_id"], grace_seconds=2.0)
    result = wait_for_status(manager, submitted["task_id"], {"interrupted"})

    assert interruption["interrupt_requested"]
    assert result["task"]["return_code"] == 130
    assert "interrupted" in result["stderr"]["text"].lower()


def test_execute_code_reports_success_failure_and_timeout(tmp_path):
    manager = TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT)

    success = manager.execute_code("print('ok')", timeout=2)
    failure = manager.execute_code("raise RuntimeError('boom')", timeout=2)
    timeout = manager.execute_code("import time; time.sleep(1)", timeout=0.05)

    assert success["status"] == "completed"
    assert success["stdout"].strip() == "ok"
    assert failure["status"] == "failed"
    assert "RuntimeError: boom" in failure["stderr"]
    assert timeout["status"] == "timeout"


def test_log_offsets_return_only_unread_bytes(tmp_path):
    script = tmp_path / "output.py"
    script.write_text("print('abcdefghij', flush=True)\n", encoding="utf-8")
    manager = TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT)
    submitted = manager.submit(str(script), "offset unit task")
    wait_for_status(manager, submitted["task_id"], {"completed"})

    first = manager.status(submitted["task_id"], max_output_chars=4)["stdout"]
    second = manager.status(
        submitted["task_id"],
        stdout_offset=first["next_offset"],
        max_output_chars=20,
    )["stdout"]

    assert first["text"] == "abcd"
    assert second["text"] == "efghij\n"
    assert not second["has_more"]


def test_live_code_shares_and_mutates_running_task_namespace(tmp_path):
    script = tmp_path / "live_model.py"
    script.write_text(
        """\
import time
from geotaichi_mcp.execution.live import checkpoint, publish

state = {"value": 1, "stop": False}
publish(state=state)
print("live-ready", flush=True)
while not state["stop"]:
    checkpoint()
    time.sleep(0.005)
checkpoint()
print("final-value=%s" % state["value"], flush=True)
""",
        encoding="utf-8",
    )
    workspace = tmp_path / "tasks"
    manager = TaskManager(workspace=workspace, repo_root=REPOSITORY_ROOT)
    submitted = manager.submit(str(script), "live namespace task")
    wait_for_output(manager, submitted["task_id"], "live-ready")

    recovered = TaskManager(workspace=workspace, repo_root=REPOSITORY_ROOT)
    mutation = recovered.execute_code_in_task(
        submitted["task_id"],
        "state['value'] += 41\nprint('snippet-value=%s' % state['value'])\n_geotaichi_mcp_result = state['value']",
        timeout=2,
    )
    observed = recovered.execute_code_in_task(submitted["task_id"], "state['value']", timeout=2)
    stopped = recovered.execute_code_in_task(submitted["task_id"], "state.__setitem__('stop', True)", timeout=2)
    result = wait_for_status(manager, submitted["task_id"], {"completed"})

    assert mutation["status"] == "completed"
    assert mutation["execution_mode"] == "live"
    assert mutation["result"] == 42
    assert "snippet-value=42" in mutation["stdout"]
    assert observed["result"] == 42
    assert stopped["status"] == "completed"
    assert "snippet-value=42" in result["stdout"]["text"]
    assert "final-value=42" in result["stdout"]["text"]
    assert result["task"]["live_execution"]["enabled"] is False


def test_live_request_timeout_is_not_executed_at_a_later_checkpoint(tmp_path):
    script = tmp_path / "late_checkpoint.py"
    script.write_text(
        """\
import time
from geotaichi_mcp.execution.live import checkpoint

print("sleeping", flush=True)
time.sleep(0.25)
checkpoint()
print("finished", flush=True)
""",
        encoding="utf-8",
    )
    manager = TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT)
    submitted = manager.submit(str(script), "expired live request")
    wait_for_output(manager, submitted["task_id"], "sleeping")

    result = manager.execute_code_in_task(
        submitted["task_id"],
        "import pathlib, os; pathlib.Path(os.environ['GEOTAICHI_MCP_OUTPUT_DIR'], 'late.marker').touch()",
        timeout=0.05,
    )
    terminal = wait_for_status(manager, submitted["task_id"], {"completed"})

    assert result["status"] == "timeout"
    assert result["executed"] is False
    assert not (Path(terminal["task"]["output_directory"]) / "late.marker").exists()


def test_live_python_timeout_and_error_do_not_terminate_task(tmp_path):
    script = tmp_path / "resilient.py"
    script.write_text(
        """\
import time
from geotaichi_mcp.execution.live import checkpoint, publish

state = {"stop": False}
publish(state=state)
print("ready", flush=True)
while not state["stop"]:
    checkpoint()
    time.sleep(0.005)
""",
        encoding="utf-8",
    )
    manager = TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT)
    submitted = manager.submit(str(script), "resilient live task")
    wait_for_output(manager, submitted["task_id"], "ready")

    failed = manager.execute_code_in_task(
        submitted["task_id"],
        "_geotaichi_mcp_result = 'stale'\nraise ValueError('bad snippet')",
        timeout=1,
    )
    after_failure = manager.execute_code_in_task(submitted["task_id"], "pass", timeout=1)
    timed_out = manager.execute_code_in_task(submitted["task_id"], "while True:\n    pass", timeout=0.05)
    alive = manager.execute_code_in_task(submitted["task_id"], "state['stop'] = True", timeout=1)
    terminal = wait_for_status(manager, submitted["task_id"], {"completed"})

    assert failed["status"] == "error"
    assert failed["error_type"] == "ValueError"
    assert after_failure["status"] == "completed"
    assert after_failure["result"] is None
    assert timed_out["status"] == "timeout"
    assert timed_out["executed"] is True
    assert alive["status"] == "completed"
    assert terminal["task"]["return_code"] == 0


def test_runtime_hook_is_noop_until_callback_is_installed():
    clear_runtime_checkpoint()
    assert runtime_checkpoint() == 0
    calls = []

    def callback():
        calls.append(True)
        return 3

    set_runtime_checkpoint(callback)
    try:
        assert runtime_checkpoint() == 3
        assert calls == [True]
    finally:
        clear_runtime_checkpoint(callback)
    assert runtime_checkpoint() == 0


def test_task_interrupt_passes_through_a_busy_live_snippet(tmp_path):
    script = tmp_path / "interrupt_live.py"
    script.write_text(
        """\
import time
from geotaichi_mcp.execution.live import checkpoint

print("checkpoint-ready", flush=True)
while True:
    checkpoint()
    time.sleep(0.005)
""",
        encoding="utf-8",
    )
    manager = TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT)
    submitted = manager.submit(str(script), "interrupt busy live snippet")
    wait_for_output(manager, submitted["task_id"], "checkpoint-ready")

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            manager.execute_code_in_task,
            submitted["task_id"],
            "print('snippet-busy', flush=True)\nwhile True:\n    pass",
            10,
        )
        wait_for_output(manager, submitted["task_id"], "snippet-busy")
        interruption = manager.interrupt(submitted["task_id"], grace_seconds=2.0)
        live_result = pending.result(timeout=5)

    terminal = wait_for_status(manager, submitted["task_id"], {"interrupted"})
    assert interruption["interrupt_requested"]
    assert live_result["status"] == "timeout"
    assert live_result["executed"] is False
    assert live_result["task_status"] == "interrupted"
    assert terminal["task"]["return_code"] == 130
