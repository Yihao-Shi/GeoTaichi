"""Queue dependencies and the short-run gate must precede production work."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

SPEC = importlib.util.spec_from_file_location(
    "footing_queue",
    Path(__file__).resolve().parents[3] / "examples/mpm/Footing/strip_footing_adaptive/run_comparison.py",
)
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def make_uniform_baseline(path):
    contract = json.loads(Path(QUEUE.__file__).with_name("model-contract.json").read_text())
    path.mkdir()
    contact = contract["contact_or_coupling"]
    run = dict(
        status="completed",
        level=0,
        smoke=False,
        final_time=17.5,
        dt=contract["time"]["timestep"],
        fine_h=0.1,
        coarse_h=0.1,
        contact=dict(law=contact["law"], **{k: contact["parameters"][k] for k in ("friction", "cut_off", "penalty")}),
        precision="float64",
        gauss_number=0,
        stabilize="B-Bar Method",
        wall_seconds=9000,
    )
    (path / "run.json").write_text(json.dumps(run))
    (path / "executed-model-contract.json").write_text(json.dumps(contract))
    (path / "evaluation.json").write_text(json.dumps({"passed": True}))
    (path / "particles.npz").write_bytes(b"immutable particle snapshot")
    return contract


def test_reuse_preserves_original_metadata_and_marks_runtime_incomparable(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    contract = make_uniform_baseline(source)
    original = (source / "run.json").read_bytes()
    provenance = QUEUE.reuse_uniform_run(source, target, contract, 17.5)
    run = json.loads((target / "run.json").read_text())
    assert (source / "run.json").read_bytes() == original
    assert run["reused_from"] == str(source)
    assert run["wall_seconds"] == 9000
    assert run["timing_comparable"] is False
    assert provenance["source"] == str(source)
    assert (target / "particles.npz").read_bytes() == (source / "particles.npz").read_bytes()


@pytest.mark.parametrize("change", [{"final_time": 1.0}, {"contact": {"law": "GeoContact", "cut_off": 1.5}}])
def test_reuse_rejects_incomplete_or_different_baseline(tmp_path, change):
    source, target = tmp_path / "source", tmp_path / "target"
    contract = make_uniform_baseline(source)
    run = json.loads((source / "run.json").read_text())
    run.update(change)
    (source / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="numerical parameters"):
        QUEUE.reuse_uniform_run(source, target, contract, 17.5)
    assert not target.exists()


def test_reused_uniform_schedules_only_adaptive_cases(tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_uniform_baseline(source)
    reference = tmp_path / "reference.csv"
    reference.write_text("saved normal result")
    output = tmp_path / "output"
    monkeypatch.setattr(
        QUEUE.sys,
        "argv",
        [
            "queue",
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--arch",
            "cpu",
            "--reuse-uniform",
            str(source),
        ],
    )
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=456, wait=lambda: 0)

    monkeypatch.setattr(QUEUE.subprocess, "Popen", launch)
    monkeypatch.setattr(QUEUE.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    QUEUE.main()
    state = json.loads((output / "queue.json").read_text())
    assert [command[command.index("--level") + 1] for command in commands] == ["1", "2", "1", "2"]
    assert state["cases"][0]["status"] == "reused"
    assert state["status"] == "passed"


def test_dependency_wait_includes_cpu_postprocessing_and_stops_on_zombie(monkeypatch):
    statuses = iter(["running", psutil.STATUS_ZOMBIE])
    process = SimpleNamespace(pid=123, create_time=lambda: 42, is_running=lambda: True, status=lambda: next(statuses))
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    sleeps, snapshots = [], []
    monkeypatch.setattr(QUEUE.time, "sleep", sleeps.append)
    state = {}
    QUEUE.wait_for_dependencies([123], state, lambda: snapshots.append(list(state["waiting_for_pids"])))
    assert snapshots == [[123], []]
    assert sleeps == [30]
    assert state["dependencies"] == [{"pid": 123, "created": 42}]


def test_dependency_wait_does_not_follow_reused_pid(monkeypatch):
    created = iter([42, 43])
    process = SimpleNamespace(
        pid=123, create_time=lambda: next(created), is_running=lambda: True, status=lambda: "running"
    )
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    state = {}
    QUEUE.wait_for_dependencies([123], state, lambda: None)
    assert state["waiting_for_pids"] == []


@pytest.mark.parametrize("evaluation_exit_code", [0, 1])
def test_preflight_gates_production_cases(tmp_path, monkeypatch, evaluation_exit_code):
    reference = tmp_path / "reference.csv"
    reference.write_text("saved normal result")
    output = tmp_path / "output"
    monkeypatch.setattr(
        QUEUE.sys, "argv", ["queue", "--reference", str(reference), "--output", str(output), "--arch", "cpu"]
    )
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=456, wait=lambda: 0)

    monkeypatch.setattr(QUEUE.subprocess, "Popen", launch)
    monkeypatch.setattr(
        QUEUE.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=evaluation_exit_code)
    )
    if evaluation_exit_code:
        with pytest.raises(SystemExit):
            QUEUE.main()
    else:
        QUEUE.main()
    state = json.loads((output / "queue.json").read_text())
    assert "preflight/uniform" in commands[0][-3]
    if evaluation_exit_code:
        assert len(commands) == 1
        assert all(row["status"] == "queued" for row in state["cases"])
        assert state["status"] == "failed_run"
    else:
        assert len(commands) == 6
        assert all(row["status"] == "completed" for row in state["cases"])
        assert state["status"] == "passed"
