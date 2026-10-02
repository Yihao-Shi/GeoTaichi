import importlib.util
import json
import time
from pathlib import Path

import pytest

from geotaichi_mcp.core.jobs import (
    document_fingerprint,
    prepare_solver_job,
    validate_scene_manifest,
    validate_solver_job,
)
from geotaichi_mcp.execution.task_manager import TaskManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _scene_manifest():
    return {
        "schema_version": 1,
        "scene_id": "scene-1",
        "source": {"application": "Blender", "version": "4.2", "file": "scene.blend"},
        "coordinate_system": {
            "up_axis": "Z",
            "handedness": "right",
            "length_unit": "m",
            "meters_per_unit": 1.0,
        },
        "frame": {"start": 1, "end": 2, "fps": 24.0},
        "objects": [
            {
                "object_id": "object-1",
                "name": "Cube",
                "role": "FEM",
                "enabled": True,
                "transform": {"matrix_world": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
                "geometry": {
                    "type": "MESH",
                    "vertex_count": 8,
                    "edge_count": 12,
                    "face_count": 6,
                    "topology_hash": "abc123",
                },
            }
        ],
        "metadata": {},
    }


def _solver_job():
    return {
        "schema_version": 1,
        "name": "unit-job",
        "description": "contract staging test",
        "model": {
            "entry_script": "model.py",
            "arguments": [
                "--unit",
                "--contract",
                "{model_contract}",
                "--scene-manifest",
                "{scene_manifest}",
                "--output-dir",
                "{output_directory}",
            ],
            "working_directory": ".",
            "contract_path": "model-contract.json",
        },
        "scene": {"manifest_path": "scene-manifest.json"},
        "execution": {"profile": "trusted", "requires_confirmation": True},
        "outputs": {"requested_artifacts": ["environment.json"]},
        "metadata": {"source": "unit"},
    }


def _wait(manager, task_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = manager.status(task_id, max_output_chars=20000)
        if result["task"]["status"] in {"completed", "failed", "interrupted"}:
            return result
        time.sleep(0.02)
    raise AssertionError("task did not finish")


def test_solver_job_and_scene_manifest_validation_is_strict():
    scene = _scene_manifest()
    job = _solver_job()

    assert validate_scene_manifest(scene)["valid"]
    assert validate_solver_job(job)["valid"]
    assert document_fingerprint(job) == document_fingerprint(dict(reversed(list(job.items()))))

    scene["objects"][0]["transform"]["matrix_world"][3] = float("inf")
    job["execution"]["requires_confirmation"] = False
    assert not validate_scene_manifest(scene)["valid"]
    assert not validate_solver_job(job)["valid"]


def test_solver_job_rejects_unbound_or_embedded_staging_placeholders():
    missing_contract = _solver_job()
    missing_contract["model"]["contract_path"] = ""
    embedded = _solver_job()
    embedded["model"]["arguments"] = ["--output={output_directory}"]

    assert not validate_solver_job(missing_contract)["valid"]
    assert not validate_solver_job(embedded)["valid"]


@pytest.mark.parametrize(
    ("field", "value"),
    (("arguments", None), ("arguments", 3), ("scene", "scene.json")),
)
def test_solver_job_reports_malformed_placeholder_containers(field, value):
    job = _solver_job()
    if field == "scene":
        job["scene"] = value
    else:
        job["model"][field] = value

    validation = validate_solver_job(job)

    assert validation["valid"] is False
    assert validation["errors"]


def test_all_model_templates_require_named_contract_arguments():
    asset_root = REPOSITORY_ROOT / "agent/geotaichi-model-builder/assets"
    templates = sorted(
        path for path in asset_root.glob("*_model_template.py") if not path.name.startswith("._")
    )

    assert templates
    for index, template in enumerate(templates):
        specification = importlib.util.spec_from_file_location("geotaichi_template_%d" % index, template)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        arguments = module.parse_args(
            [
                "--contract",
                "contract.json",
                "--scene-manifest",
                "scene.json",
                "--output-dir",
                "output",
            ]
        )
        assert arguments.contract == "contract.json"
        assert arguments.scene_manifest == "scene.json"
        assert arguments.output_dir == "output"
        with pytest.raises(SystemExit):
            module.parse_args([])


def test_prepared_solver_job_stages_immutable_inputs_and_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTAICHI_MODEL_CONTRACT", "/stale/model-contract.json")
    monkeypatch.setenv("GEOTAICHI_SCENE_MANIFEST", "/stale/scene-manifest.json")
    monkeypatch.setenv("GEOTAICHI_MCP_OUTPUT_DIR", "/stale/output")
    script = tmp_path / "model.py"
    script.write_text(
        """\
import json
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--unit", action="store_true")
parser.add_argument("--contract", required=True)
parser.add_argument("--scene-manifest", required=True)
parser.add_argument("--output-dir", required=True)
arguments = parser.parse_args()
paths = {
    key: os.environ.get(key)
    for key in (
        "GEOTAICHI_SOLVER_JOB",
        "GEOTAICHI_SCENE_MANIFEST",
        "GEOTAICHI_DIAGNOSTICS_PATH",
    )
}
paths["ARG_CONTRACT"] = arguments.contract
paths["ARG_SCENE"] = arguments.scene_manifest
paths["ARG_OUTPUT"] = arguments.output_dir
paths["ARG_UNIT"] = arguments.unit
output = Path(arguments.output_dir)
(output / "environment.json").write_text(json.dumps(paths), encoding="utf-8")
""",
        encoding="utf-8",
    )
    scene_path = tmp_path / "scene-manifest.json"
    scene_path.write_text(json.dumps(_scene_manifest()), encoding="utf-8")
    contract_path = tmp_path / "model-contract.json"
    contract_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    job_path = tmp_path / "solver-job.json"
    job_path.write_text(json.dumps(_solver_job()), encoding="utf-8")

    context, validation = prepare_solver_job(job_path, REPOSITORY_ROOT)
    assert validation["valid"]
    manager = TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT)
    task = manager.submit(
        context["entry_script"],
        "job staging",
        context["arguments"],
        context["working_directory"],
        job_context=context,
    )
    result = _wait(manager, task["task_id"])

    assert result["task"]["status"] == "completed", result
    task_directory = Path(result["task"]["task_directory"])
    assert (task_directory / "solver-job.json").is_file()
    assert (task_directory / "scene-manifest.json").is_file()
    assert (task_directory / "model-contract.json").is_file()
    assert result["diagnostics"]["status"] == "completed"
    assert result["physics_score"]["score"]["decision"] == "insufficient_evidence"
    environment = json.loads(
        (task_directory / "output" / "environment.json").read_text(encoding="utf-8")
    )
    assert Path(environment["GEOTAICHI_SOLVER_JOB"]).parent == task_directory
    assert environment["GEOTAICHI_SCENE_MANIFEST"] is None
    assert environment.get("GEOTAICHI_MODEL_CONTRACT") is None
    assert environment.get("GEOTAICHI_MCP_OUTPUT_DIR") is None
    assert Path(environment["ARG_CONTRACT"]).parent == task_directory
    assert Path(environment["ARG_SCENE"]).parent == task_directory
    assert Path(environment["ARG_OUTPUT"]) == task_directory / "output"
    assert environment["ARG_UNIT"] is True
    staged_job = json.loads((task_directory / "solver-job.json").read_text(encoding="utf-8"))
    assert "{model_contract}" not in staged_job["model"]["arguments"]
    assert staged_job["model"]["arguments"] == result["task"]["arguments"]
    artifacts = manager.artifacts(task["task_id"])
    assert {record["relative_path"] for record in artifacts["artifacts"]} == {
        "diagnostics.json",
        "model-contract.json",
        "output/environment.json",
        "physics-score.json",
        "scene-manifest.json",
        "solver-job.json",
    }
