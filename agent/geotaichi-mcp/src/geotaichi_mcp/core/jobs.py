"""Canonical SolverJob and SceneManifest validation and staging helpers."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import resolve_user_path


SOLVER_JOB_SCHEMA_VERSION = 1
SCENE_MANIFEST_SCHEMA_VERSION = 1
OBJECT_ROLES = {"REFERENCE", "COLLIDER", "FEM", "MPM", "DEM", "IGA"}
UP_AXES = {"X", "Y", "Z"}
HANDEDNESS = {"left", "right"}
SOLVER_JOB_ARGUMENT_PLACEHOLDERS = {
    "{solver_job}",
    "{scene_manifest}",
    "{model_contract}",
    "{output_directory}",
}


def _issue(path: str, message: str) -> Dict[str, str]:
    return {"path": path, "message": message}


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _canonical_json(document: Dict[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def document_fingerprint(document: Dict[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for one JSON document."""
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def load_json_document(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("JSON document must be an object")
    return document


def validate_scene_manifest(document: Any) -> Dict[str, Any]:
    """Validate a decoded SceneManifest without importing Blender."""
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    if not isinstance(document, dict):
        return {
            "valid": False,
            "errors": [_issue("$", "SceneManifest must be a JSON object.")],
            "warnings": [],
        }
    if document.get("schema_version") != SCENE_MANIFEST_SCHEMA_VERSION:
        errors.append(_issue("schema_version", "Expected SceneManifest schema_version 1."))
    if not _nonempty_string(document.get("scene_id")):
        errors.append(_issue("scene_id", "Expected a stable non-empty scene identifier."))

    source = document.get("source")
    if not isinstance(source, dict):
        errors.append(_issue("source", "Expected a source object."))
    elif not _nonempty_string(source.get("application")):
        errors.append(_issue("source.application", "Expected a non-empty source application."))

    coordinate = document.get("coordinate_system")
    if not isinstance(coordinate, dict):
        errors.append(_issue("coordinate_system", "Expected a coordinate-system object."))
    else:
        if coordinate.get("up_axis") not in UP_AXES:
            errors.append(_issue("coordinate_system.up_axis", "Expected X, Y, or Z."))
        if coordinate.get("handedness") not in HANDEDNESS:
            errors.append(_issue("coordinate_system.handedness", "Expected 'left' or 'right'."))
        if not _nonempty_string(coordinate.get("length_unit")):
            errors.append(_issue("coordinate_system.length_unit", "Expected a non-empty length unit."))
        scale = coordinate.get("meters_per_unit")
        if not _finite_number(scale) or float(scale) <= 0.0:
            errors.append(_issue("coordinate_system.meters_per_unit", "Expected a finite positive scale."))

    frame = document.get("frame")
    if not isinstance(frame, dict):
        errors.append(_issue("frame", "Expected a frame object."))
    else:
        start = frame.get("start")
        end = frame.get("end")
        fps = frame.get("fps")
        if not isinstance(start, int) or isinstance(start, bool):
            errors.append(_issue("frame.start", "Expected an integer start frame."))
        if not isinstance(end, int) or isinstance(end, bool):
            errors.append(_issue("frame.end", "Expected an integer end frame."))
        if isinstance(start, int) and isinstance(end, int) and end < start:
            errors.append(_issue("frame.end", "End frame must not precede start frame."))
        if not _finite_number(fps) or float(fps) <= 0.0:
            errors.append(_issue("frame.fps", "Expected a finite positive frame rate."))

    objects = document.get("objects")
    if not isinstance(objects, list):
        errors.append(_issue("objects", "Expected an object list."))
        objects = []
    seen_ids = set()
    for index, item in enumerate(objects):
        path = "objects[%d]" % index
        if not isinstance(item, dict):
            errors.append(_issue(path, "Expected an object record."))
            continue
        object_id = item.get("object_id")
        if not _nonempty_string(object_id):
            errors.append(_issue(path + ".object_id", "Expected a stable non-empty object identifier."))
        elif object_id in seen_ids:
            errors.append(_issue(path + ".object_id", "Object identifiers must be unique."))
        else:
            seen_ids.add(object_id)
        if not _nonempty_string(item.get("name")):
            errors.append(_issue(path + ".name", "Expected a non-empty object name."))
        if item.get("role") not in OBJECT_ROLES:
            errors.append(_issue(path + ".role", "Expected one of: %s." % ", ".join(sorted(OBJECT_ROLES))))
        transform = item.get("transform")
        matrix = transform.get("matrix_world") if isinstance(transform, dict) else None
        if not isinstance(matrix, list) or len(matrix) != 16 or not all(_finite_number(value) for value in matrix):
            errors.append(_issue(path + ".transform.matrix_world", "Expected 16 finite row-major values."))
        geometry = item.get("geometry")
        if not isinstance(geometry, dict):
            errors.append(_issue(path + ".geometry", "Expected a geometry summary object."))
            continue
        for count_name in ("vertex_count", "edge_count", "face_count"):
            value = geometry.get(count_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(_issue(path + ".geometry." + count_name, "Expected a non-negative integer."))
        if item.get("enabled", True) and geometry.get("type") == "MESH" and geometry.get("vertex_count") == 0:
            warnings.append(_issue(path + ".geometry.vertex_count", "Enabled mesh has no vertices."))
        if not _nonempty_string(geometry.get("topology_hash")):
            warnings.append(_issue(path + ".geometry.topology_hash", "Geometry has no topology fingerprint."))

    if not objects:
        warnings.append(_issue("objects", "SceneManifest contains no exported objects."))
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def validate_solver_job(document: Any) -> Dict[str, Any]:
    """Validate a decoded SolverJob independently of filesystem state."""
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    if not isinstance(document, dict):
        return {
            "valid": False,
            "errors": [_issue("$", "SolverJob must be a JSON object.")],
            "warnings": [],
        }
    if document.get("schema_version") != SOLVER_JOB_SCHEMA_VERSION:
        errors.append(_issue("schema_version", "Expected SolverJob schema_version 1."))
    if not _nonempty_string(document.get("name")):
        errors.append(_issue("name", "Expected a non-empty job name."))
    description = document.get("description", "")
    if not isinstance(description, str) or len(description) > 2000:
        errors.append(_issue("description", "Expected a string containing at most 2000 characters."))

    model_arguments: List[str] = []
    model = document.get("model")
    if not isinstance(model, dict):
        errors.append(_issue("model", "Expected a model execution object."))
    else:
        if not _nonempty_string(model.get("entry_script")):
            errors.append(_issue("model.entry_script", "Expected a Python entry-script path."))
        arguments = model.get("arguments", [])
        if (
            not isinstance(arguments, list)
            or len(arguments) > 64
            or not all(isinstance(value, str) for value in arguments)
        ):
            errors.append(_issue("model.arguments", "Expected at most 64 string arguments."))
        else:
            model_arguments = arguments
        for index, argument in enumerate(model_arguments):
            for placeholder in SOLVER_JOB_ARGUMENT_PLACEHOLDERS:
                if placeholder in argument and argument != placeholder:
                    errors.append(
                        _issue(
                            "model.arguments[%d]" % index,
                            "%s must occupy a complete argument." % placeholder,
                        )
                    )
        if "{model_contract}" in model_arguments and not _nonempty_string(model.get("contract_path")):
            errors.append(
                _issue(
                    "model.arguments",
                    "{model_contract} requires model.contract_path.",
                )
            )
        for optional_path in ("working_directory", "contract_path"):
            if (
                optional_path in model
                and model[optional_path] not in (None, "")
                and not _nonempty_string(model[optional_path])
            ):
                errors.append(_issue("model." + optional_path, "Expected an empty value or path string."))

    scene = document.get("scene", {})
    if not isinstance(scene, dict):
        errors.append(_issue("scene", "Expected a scene object."))
    elif scene.get("manifest_path") not in (None, "") and not _nonempty_string(scene.get("manifest_path")):
        errors.append(_issue("scene.manifest_path", "Expected an empty value or SceneManifest path."))
    if (
        "{scene_manifest}" in model_arguments
        and (
            not isinstance(scene, dict)
            or not _nonempty_string(scene.get("manifest_path"))
        )
    ):
        errors.append(
            _issue(
                "model.arguments",
                "{scene_manifest} requires scene.manifest_path.",
            )
        )

    execution = document.get("execution")
    if not isinstance(execution, dict):
        errors.append(_issue("execution", "Expected an execution-policy object."))
    else:
        if execution.get("profile") not in {"trusted"}:
            errors.append(_issue("execution.profile", "Only the explicit 'trusted' execution profile is supported."))
        if execution.get("requires_confirmation") is not True:
            errors.append(
                _issue(
                    "execution.requires_confirmation",
                    "Trusted script execution must explicitly require caller confirmation.",
                )
            )

    outputs = document.get("outputs", {})
    if not isinstance(outputs, dict):
        errors.append(_issue("outputs", "Expected an outputs object."))
    else:
        artifacts = outputs.get("requested_artifacts", [])
        if not isinstance(artifacts, list) or not all(_nonempty_string(value) for value in artifacts):
            errors.append(_issue("outputs.requested_artifacts", "Expected a list of non-empty artifact labels."))

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def resolve_solver_job_arguments(
    arguments: List[str],
    *,
    solver_job_path: Path,
    output_directory: Path,
    scene_manifest_path: Optional[Path] = None,
    model_contract_path: Optional[Path] = None,
) -> List[str]:
    """Resolve task-owned path placeholders after immutable inputs are staged."""
    bindings = {
        "{solver_job}": solver_job_path,
        "{output_directory}": output_directory,
        "{scene_manifest}": scene_manifest_path,
        "{model_contract}": model_contract_path,
    }
    resolved = []
    for argument in arguments:
        if argument not in SOLVER_JOB_ARGUMENT_PLACEHOLDERS:
            resolved.append(argument)
            continue
        path = bindings[argument]
        if path is None:
            raise ValueError("SolverJob argument %s has no staged input" % argument)
        resolved.append(str(path))
    return resolved


def _resolve_job_path(value: str, job_path: Path, repo_root: Optional[Path], must_exist: bool) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        adjacent = (job_path.parent / candidate).resolve()
        if adjacent.exists() or repo_root is None:
            candidate = adjacent
        else:
            candidate = resolve_user_path(candidate, repo_root)
    else:
        candidate = candidate.resolve()
    if must_exist and not candidate.exists():
        raise FileNotFoundError(str(candidate))
    return candidate


def prepare_solver_job(job_path: Path, repo_root: Optional[Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load, validate, resolve, and fingerprint one SolverJob.

    The returned context is suitable for :class:`TaskManager`; referenced JSON
    documents are decoded before task launch so the task directory can contain
    a self-contained, immutable copy.
    """
    job_path = job_path.expanduser().resolve()
    document = load_json_document(job_path)
    result = validate_solver_job(document)
    if not result["valid"]:
        raise ValueError("SolverJob failed validation: %s" % json.dumps(result["errors"], ensure_ascii=False))

    normalized = deepcopy(document)
    model = normalized["model"]
    entry_script = _resolve_job_path(model["entry_script"], job_path, repo_root, True)
    if not entry_script.is_file() or entry_script.suffix.lower() != ".py":
        raise ValueError("model.entry_script must resolve to a Python .py file")
    working_value = model.get("working_directory") or str(entry_script.parent)
    working_directory = _resolve_job_path(working_value, job_path, repo_root, True)
    if not working_directory.is_dir():
        raise NotADirectoryError(str(working_directory))

    contract_document = None
    contract_path = None
    if model.get("contract_path"):
        contract_path = _resolve_job_path(model["contract_path"], job_path, repo_root, True)
        contract_document = load_json_document(contract_path)

    scene_document = None
    scene_path = None
    scene = normalized.get("scene") or {}
    if scene.get("manifest_path"):
        scene_path = _resolve_job_path(scene["manifest_path"], job_path, repo_root, True)
        scene_document = load_json_document(scene_path)
        scene_result = validate_scene_manifest(scene_document)
        if not scene_result["valid"]:
            raise ValueError(
                "SceneManifest failed validation: %s"
                % json.dumps(scene_result["errors"], ensure_ascii=False)
            )
        result["warnings"].extend(scene_result["warnings"])

    context = {
        "source_path": str(job_path),
        "document": normalized,
        "fingerprint": document_fingerprint(normalized),
        "entry_script": str(entry_script),
        "working_directory": str(working_directory),
        "arguments": list(model.get("arguments") or []),
        "contract_document": contract_document,
        "contract_source_path": str(contract_path) if contract_path else None,
        "scene_document": scene_document,
        "scene_source_path": str(scene_path) if scene_path else None,
    }
    if contract_document is not None:
        context["contract_fingerprint"] = document_fingerprint(contract_document)
    if scene_document is not None:
        context["scene_fingerprint"] = document_fingerprint(scene_document)
    return context, result
