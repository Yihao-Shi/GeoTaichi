import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from blender.core.async_service import SingleFlightService
from blender.core.cache import cache_task_artifacts
from blender.core.job_client import JobClient
from blender.core.manifest import build_scene_manifest, build_solver_job, topology_fingerprint
from blender.core.state import AppState, Event, Phase, transition
from blender.install_blender_addon import parse_args as parse_install_args


def test_manifest_and_solver_job_are_deterministic_and_versioned():
    first = topology_fingerprint([(0, 0, 0), (1, 0, 0)], [(0, 1)], [])
    second = topology_fingerprint([(0, 0, 0), (1, 0, 0)], [(0, 1)], [])
    changed = topology_fingerprint([(0, 0, 0), (2, 0, 0)], [(0, 1)], [])
    assert first == second
    assert first != changed

    manifest = build_scene_manifest(
        scene_id="scene-1",
        source_version="4.2",
        source_file="scene.blend",
        meters_per_unit=1.0,
        frame_start=1,
        frame_end=2,
        fps=24.0,
        objects=[],
    )
    job = build_solver_job(
        name="unit",
        entry_script="model.py",
        working_directory=".",
        scene_manifest_path="scene-manifest.json",
        solver_family="FEMPM",
        solver_mode="IPC",
        spatial_dimension="3",
    )
    assert manifest["schema_version"] == 1
    assert job["schema_version"] == 1
    assert job["execution"] == {"profile": "trusted", "requires_confirmation": True}
    assert job["metadata"]["solver_family"] == "FEMPM"
    assert job["metadata"]["solver_mode"] == "IPC"
    assert job["metadata"]["spatial_dimension"] == "3"
    assert job["model"]["arguments"] == [
        "--scene-manifest",
        "{scene_manifest}",
        "--output-dir",
        "{output_directory}",
    ]


def test_solver_job_supports_standard_and_additional_model_arguments():
    job = build_solver_job(
        name="unit",
        entry_script="model.py",
        working_directory=".",
        contract_path="contract.json",
        scene_manifest_path="scene.json",
        model_arguments=("--steps", "4"),
    )

    assert job["model"]["arguments"] == [
        "--contract",
        "{model_contract}",
        "--scene-manifest",
        "{scene_manifest}",
        "--output-dir",
        "{output_directory}",
        "--steps",
        "4",
    ]

    with pytest.raises(ValueError, match="must not repeat"):
        build_solver_job(
            name="unit",
            entry_script="model.py",
            working_directory=".",
            scene_manifest_path="scene.json",
            model_arguments=("--output-dir", "other"),
        )

    with pytest.raises(ValueError, match="not available"):
        build_solver_job(
            name="unit",
            entry_script="model.py",
            working_directory=".",
            scene_manifest_path="scene.json",
            solver_family="IGA",
            solver_mode="LSDEM",
        )


@pytest.mark.parametrize("family", ("FEM", "IGA", "MPM"))
@pytest.mark.parametrize("mode", ("EXPLICIT", "IMPLICIT"))
def test_solver_job_routes_standalone_axisymmetric_integrators(family, mode):
    job = build_solver_job(
        name="axisymmetric",
        entry_script="axisymmetric_annulus.py",
        working_directory=".",
        scene_manifest_path="scene.json",
        solver_family=family,
        solver_mode=mode,
        spatial_dimension="AXISYMMETRIC",
    )

    assert job["metadata"] == {
        "source": "blender",
        "solver_family": family,
        "solver_mode": mode,
        "spatial_dimension": "AXISYMMETRIC",
    }


def test_legacy_addon_installer_requires_an_explicit_addon_path(tmp_path):
    destination = tmp_path / "scripts" / "addons"

    arguments = parse_install_args(["--addon-path", str(destination), "--yes"])

    assert arguments.addon_path == str(destination)
    assert arguments.yes is True
    with pytest.raises(SystemExit):
        parse_install_args([])


def test_state_machine_keeps_task_identity_across_polling():
    exported, _ = transition(AppState(), Event.EXPORT_SUCCEEDED, {"job_path": "job.json"})
    validated, _ = transition(exported, Event.VALIDATION_SUCCEEDED)
    submitted, effects = transition(
        validated,
        Event.SUBMISSION_SUCCEEDED,
        {"task_id": "0123456789ab", "output_directory": "/tmp/output"},
    )
    running, _ = transition(submitted, Event.STATUS_RECEIVED, {"task_status": "running"})
    completed, _ = transition(running, Event.STATUS_RECEIVED, {"task_status": "completed"})

    assert submitted.phase == Phase.SUBMITTED
    assert effects == ("redraw", "schedule_status_poll")
    assert running.task_id == submitted.task_id
    assert completed.phase == Phase.COMPLETED


def test_single_flight_service_serializes_background_work():
    service = SingleFlightService()
    try:
        assert service.submit("first", lambda: 42)
        assert not service.submit("second", lambda: 0)
        deadline = time.monotonic() + 2.0
        result = None
        while result is None and time.monotonic() < deadline:
            result = service.poll()
            time.sleep(0.001)
        assert result is not None
        assert result.name == "first"
        assert result.value == 42
    finally:
        service.shutdown()


def test_artifact_cache_preserves_relative_paths_and_is_immutable(tmp_path):
    task = tmp_path / "task"
    output = task / "output"
    output.mkdir(parents=True)
    diagnostic = task / "diagnostics.json"
    result_file = output / "result.vtu"
    diagnostic.write_text("{}", encoding="utf-8")
    result_file.write_text("first", encoding="utf-8")
    artifacts = [
        {"relative_path": "diagnostics.json", "path": str(diagnostic)},
        {"relative_path": "output/result.vtu", "path": str(result_file)},
    ]

    destination = cache_task_artifacts(artifacts, task, tmp_path / "cache", "scene-1", "0123456789ab")
    result_file.write_text("changed", encoding="utf-8")
    again = cache_task_artifacts(artifacts, task, tmp_path / "cache", "scene-1", "0123456789ab")

    assert again == destination
    assert (destination / "diagnostics.json").is_file()
    assert (destination / "output" / "result.vtu").read_text(encoding="utf-8") == "first"

    with pytest.raises(ValueError):
        cache_task_artifacts(
            [{"relative_path": "../escape", "path": str(result_file)}],
            task,
            tmp_path / "other-cache",
            "scene-1",
            "0123456789ab",
        )


def test_job_client_paginates_artifacts(monkeypatch):
    pages = {
        0: {
            "task_id": "0123456789ab",
            "task_status": "completed",
            "task_directory": "/tmp/task",
            "has_more": True,
            "artifacts": [{"relative_path": "output/a", "path": "/tmp/task/output/a"}],
        },
        1: {
            "task_id": "0123456789ab",
            "task_status": "completed",
            "task_directory": "/tmp/task",
            "has_more": False,
            "artifacts": [{"relative_path": "output/b", "path": "/tmp/task/output/b"}],
        },
    }

    def run(command, **kwargs):
        skip = int(command[command.index("--skip") + 1])
        return SimpleNamespace(
            stdout=json.dumps({"ok": True, "data": pages[skip]}),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("blender.core.job_client.subprocess.run", run)
    result = JobClient("python3").artifacts("0123456789ab")
    assert result["total_count"] == 2
    assert [item["relative_path"] for item in result["artifacts"]] == [
        "output/a",
        "output/b",
    ]
