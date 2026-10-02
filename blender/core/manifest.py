"""Deterministic SceneManifest and SolverJob construction."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from ..defaults import validate_solver_route


def topology_fingerprint(
    vertices: Iterable[Sequence[float]],
    edges: Iterable[Sequence[int]],
    faces: Iterable[Sequence[int]],
) -> str:
    """Hash local-space coordinates and topology in deterministic iteration order."""
    digest = hashlib.sha256()
    for vertex in vertices:
        values = tuple(float(value) for value in vertex)
        digest.update(struct.pack("<I", len(values)))
        digest.update(struct.pack("<%sd" % len(values), *values))
    digest.update(b"|edges|")
    for edge in edges:
        values = tuple(int(value) for value in edge)
        digest.update(struct.pack("<I", len(values)))
        digest.update(struct.pack("<%sq" % len(values), *values))
    digest.update(b"|faces|")
    for face in faces:
        values = tuple(int(value) for value in face)
        digest.update(struct.pack("<I", len(values)))
        digest.update(struct.pack("<%sq" % len(values), *values))
    return digest.hexdigest()


def build_scene_manifest(
    *,
    scene_id: str,
    source_version: str,
    source_file: str,
    meters_per_unit: float,
    frame_start: int,
    frame_end: int,
    fps: float,
    objects: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not scene_id:
        raise ValueError("scene_id is required")
    if meters_per_unit <= 0.0:
        raise ValueError("meters_per_unit must be positive")
    if frame_end < frame_start:
        raise ValueError("frame_end must not precede frame_start")
    object_ids = [str(item.get("object_id", "")) for item in objects]
    if not all(object_ids) or len(object_ids) != len(set(object_ids)):
        raise ValueError("every exported object needs a unique stable object_id")
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "source": {"application": "Blender", "version": source_version, "file": source_file},
        "coordinate_system": {
            "up_axis": "Z",
            "handedness": "right",
            "length_unit": "m",
            "meters_per_unit": float(meters_per_unit),
        },
        "frame": {"start": int(frame_start), "end": int(frame_end), "fps": float(fps)},
        "objects": list(objects),
        "metadata": {},
    }


def build_solver_job(
    *,
    name: str,
    entry_script: str,
    working_directory: str,
    scene_manifest_path: str,
    contract_path: str = "",
    description: str = "",
    solver_family: str = "",
    solver_mode: str = "",
    spatial_dimension: str = "",
    model_arguments: Sequence[str] = (),
    use_standard_arguments: bool = True,
) -> Dict[str, Any]:
    if not name.strip():
        raise ValueError("job name is required")
    if not entry_script.strip():
        raise ValueError("entry_script is required")
    arguments = [str(value) for value in model_arguments]
    standard_flags = {"--contract", "--scene-manifest", "--output-dir"}
    if use_standard_arguments and any(value in standard_flags for value in arguments):
        raise ValueError("additional arguments must not repeat standard model argument flags")
    if use_standard_arguments:
        standard_arguments = []
        if contract_path:
            standard_arguments.extend(("--contract", "{model_contract}"))
        if scene_manifest_path:
            standard_arguments.extend(("--scene-manifest", "{scene_manifest}"))
        standard_arguments.extend(("--output-dir", "{output_directory}"))
        arguments = [*standard_arguments, *arguments]
    route_metadata = {}
    if solver_family.strip() or solver_mode.strip():
        if not solver_family.strip() or not solver_mode.strip():
            raise ValueError("solver_family and solver_mode must be provided together")
        family, mode = validate_solver_route(solver_family, solver_mode)
        route_metadata = {"solver_family": family, "solver_mode": mode}
    if spatial_dimension:
        dimension = str(spatial_dimension).strip().upper()
        if dimension not in {"2", "3", "AXISYMMETRIC"}:
            raise ValueError("spatial_dimension must be 2, 3, or AXISYMMETRIC")
        route_metadata["spatial_dimension"] = dimension
    return {
        "schema_version": 1,
        "name": name.strip(),
        "description": description.strip(),
        "model": {
            "entry_script": entry_script,
            "arguments": arguments,
            "working_directory": working_directory,
            "contract_path": contract_path,
        },
        "scene": {"manifest_path": scene_manifest_path},
        "execution": {"profile": "trusted", "requires_confirmation": True},
        "outputs": {"requested_artifacts": []},
        "metadata": {
            "source": "blender",
            **route_metadata,
        },
    }


def write_json_atomic(path: Path, document: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
