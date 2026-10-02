import time
from pathlib import Path

import pytest


pytest.importorskip("taichi")

from geotaichi_mcp.execution.task_manager import TaskManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def wait_for_output(manager, task_id, text, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        latest = manager.status(task_id, max_output_chars=20000)
        if text in latest["stdout"]["text"]:
            return latest
        if latest["task"]["status"] in {"completed", "failed", "interrupted"}:
            raise AssertionError("task ended before expected output: %s" % latest)
        time.sleep(0.02)
    raise AssertionError("task did not print %r" % text)


def wait_for_terminal(manager, task_id, timeout=20.0):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = manager.status(task_id, max_output_chars=20000)
        if latest["task"]["status"] in {"completed", "failed", "interrupted"}:
            return latest
        time.sleep(0.02)
    raise AssertionError("task did not reach a terminal status: %s" % latest)


@pytest.mark.integration
def test_live_checkpoint_observes_and_mutates_a_real_taichi_field(tmp_path):
    script = tmp_path / "taichi_checkpoint.py"
    script.write_text(
        """\
import time
import taichi as ti
from geotaichi_mcp.execution.live import checkpoint, publish

ti.init(arch=ti.cpu, offline_cache=False)
value = ti.field(dtype=ti.i32, shape=())
stop = False

@ti.kernel
def advance():
    value[None] += 1

publish(value=value)
print("taichi-live-ready", flush=True)
while not stop:
    advance()
    checkpoint()
    time.sleep(0.005)
print("taichi-live-final=%s" % int(value[None]), flush=True)
""",
        encoding="utf-8",
    )
    manager = TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT)
    submitted = manager.submit(str(script), "real Taichi live checkpoint")
    wait_for_output(manager, submitted["task_id"], "taichi-live-ready")

    mutation = manager.execute_code_in_task(
        submitted["task_id"],
        "before = int(value[None])\nvalue[None] = 100\nstop = True\n"
        "_geotaichi_mcp_result = {'before': before, 'after': int(value[None])}",
        timeout=5,
    )
    wait_for_output(manager, submitted["task_id"], "taichi-live-final=100")
    terminal = wait_for_terminal(manager, submitted["task_id"])

    assert mutation["status"] == "completed"
    assert mutation["result"]["before"] >= 1
    assert mutation["result"]["after"] == 100
    assert terminal["task"]["status"] == "completed"
