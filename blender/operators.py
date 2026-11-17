"""Blender operators for exporting and running canonical SolverJobs."""

from __future__ import annotations

import hashlib
import os
import shlex
import uuid
from pathlib import Path

import bpy
from bpy.types import Operator

from .core.async_service import SingleFlightService
from .core.cache import cache_task_artifacts
from .core.job_client import JobClient
from .core.manifest import build_scene_manifest, build_solver_job, topology_fingerprint, write_json_atomic
from .core.state import AppState, Event, Phase, transition


_SERVICE = None
_PENDING_SCENE = ""
_AUTO_POLL_SCENES = set()


def _ensure_scene_id(settings) -> str:
    if not settings.scene_id:
        settings.scene_id = uuid.uuid4().hex
    return settings.scene_id


def _ensure_object_id(obj) -> str:
    object_id = str(obj.get("_geotaichi_uuid", ""))
    if not object_id:
        object_id = uuid.uuid4().hex
        obj["_geotaichi_uuid"] = object_id
    return object_id


def _state_from_settings(settings) -> AppState:
    try:
        phase = Phase(settings.phase)
    except ValueError:
        phase = Phase.IDLE
    return AppState(
        phase=phase,
        job_path=settings.job_path,
        task_id=settings.task_id,
        task_status=settings.task_status,
        output_directory=settings.output_directory,
        cache_directory=settings.cache_directory,
        message=settings.last_message,
        error=settings.last_error,
    )


def _store_state(settings, state: AppState) -> None:
    settings.phase = state.phase.value
    settings.job_path = state.job_path
    settings.task_id = state.task_id
    settings.task_status = state.task_status
    settings.output_directory = state.output_directory
    settings.cache_directory = state.cache_directory
    settings.last_message = state.message
    settings.last_error = state.error


def _apply(settings, event: Event, payload=None) -> AppState:
    state, _ = transition(_state_from_settings(settings), event, payload)
    _store_state(settings, state)
    return state


def _absolute_blender_path(value: str) -> str:
    return str(Path(bpy.path.abspath(value)).expanduser().resolve()) if value else ""


def _python_command(value: str) -> str:
    expanded = os.path.expanduser(value.strip())
    if not expanded:
        return ""
    if expanded.startswith("//"):
        return _absolute_blender_path(expanded)
    if os.path.isabs(expanded) or os.sep in expanded or (os.altsep and os.altsep in expanded):
        return str(Path(expanded).resolve())
    return expanded


def _job_client(settings) -> JobClient:
    return JobClient(
        _python_command(settings.python_executable),
        _absolute_blender_path(settings.repo_root),
        _absolute_blender_path(settings.workspace),
    )


def _begin_background(scene, name, function) -> bool:
    global _PENDING_SCENE
    if _SERVICE is None or not _SERVICE.submit(name, function):
        return False
    _PENDING_SCENE = scene.name
    scene.geotaichi.operation = name
    if not bpy.app.timers.is_registered(_poll_background):
        bpy.app.timers.register(_poll_background, first_interval=0.1)
    return True


def _apply_status(settings, result) -> None:
    task = result["task"]
    _apply(
        settings,
        Event.STATUS_RECEIVED,
        {
            "task_status": task["status"],
            "output_directory": task.get("output_directory", ""),
            "error": task.get("failure_reason") or "",
        },
    )


def _finish_background(scene, result) -> None:
    settings = scene.geotaichi
    settings.operation = ""
    if result.error is not None:
        _apply(
            settings,
            Event.OPERATION_FAILED,
            {"error": "%s: %s" % (type(result.error).__name__, result.error)},
        )
        return
    if result.name == "validate":
        _apply(settings, Event.VALIDATION_SUCCEEDED)
    elif result.name == "submit":
        _apply(settings, Event.SUBMISSION_SUCCEEDED, result.value)
        _AUTO_POLL_SCENES.add(scene.name)
    elif result.name in {"status", "cancel"}:
        _apply_status(settings, result.value)
        if settings.phase in {"SUBMITTED", "RUNNING"}:
            _AUTO_POLL_SCENES.add(scene.name)
        else:
            _AUTO_POLL_SCENES.discard(scene.name)
    elif result.name == "fetch":
        _apply(settings, Event.FETCH_SUCCEEDED, {"cache_directory": str(result.value)})


def _poll_background():
    global _PENDING_SCENE
    if _SERVICE is None:
        return None
    result = _SERVICE.poll()
    if result is None:
        return 0.1
    scene = bpy.data.scenes.get(_PENDING_SCENE)
    _PENDING_SCENE = ""
    if scene is not None:
        _finish_background(scene, result)
    if _AUTO_POLL_SCENES and not bpy.app.timers.is_registered(_auto_poll_status):
        bpy.app.timers.register(_auto_poll_status, first_interval=1.0)
    return None


def _auto_poll_status():
    if _SERVICE is None or _SERVICE.busy:
        return 0.5 if _AUTO_POLL_SCENES else None
    while _AUTO_POLL_SCENES:
        scene_name = next(iter(_AUTO_POLL_SCENES))
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            _AUTO_POLL_SCENES.discard(scene_name)
            continue
        settings = scene.geotaichi
        if settings.phase not in {"SUBMITTED", "RUNNING"} or not settings.task_id:
            _AUTO_POLL_SCENES.discard(scene_name)
            continue
        client = _job_client(settings)
        task_id = settings.task_id
        if _begin_background(scene, "status", lambda: client.status(task_id)):
            return None
        return 0.5
    return None


def _mesh_geometry(obj):
    if obj.type != "MESH" or obj.data is None:
        digest = hashlib.sha256((obj.type + "\0" + obj.name).encode("utf-8")).hexdigest()
        return {
            "type": obj.type,
            "vertex_count": 0,
            "edge_count": 0,
            "face_count": 0,
            "topology_hash": digest,
        }
    mesh = obj.data
    vertices = ((vertex.co.x, vertex.co.y, vertex.co.z) for vertex in mesh.vertices)
    edges = ((edge.vertices[0], edge.vertices[1]) for edge in mesh.edges)
    faces = (tuple(polygon.vertices) for polygon in mesh.polygons)
    return {
        "type": "MESH",
        "vertex_count": len(mesh.vertices),
        "edge_count": len(mesh.edges),
        "face_count": len(mesh.polygons),
        "topology_hash": topology_fingerprint(vertices, edges, faces),
    }


def _object_record(obj):
    matrix = [float(obj.matrix_world[row][column]) for row in range(4) for column in range(4)]
    return {
        "object_id": _ensure_object_id(obj),
        "name": obj.name,
        "role": obj.geotaichi_role,
        "enabled": not obj.hide_render,
        "transform": {"matrix_world": matrix},
        "geometry": _mesh_geometry(obj),
        "properties": {},
    }


class GEOTAICHI_OT_export_solver_job(Operator):
    bl_idname = "geotaichi.export_solver_job"
    bl_label = "Export SolverJob"
    bl_description = "Write a deterministic SceneManifest and trusted SolverJob"

    def execute(self, context):
        settings = context.scene.geotaichi
        try:
            scene_id = _ensure_scene_id(settings)
            export_root = Path(_absolute_blender_path(settings.export_directory))
            job_directory = export_root / scene_id
            scene_path = job_directory / "scene-manifest.json"
            job_path = job_directory / "solver-job.json"
            unit_settings = context.scene.unit_settings
            meters_per_unit = float(unit_settings.scale_length or 1.0)
            manifest = build_scene_manifest(
                scene_id=scene_id,
                source_version=".".join(str(value) for value in bpy.app.version),
                source_file=Path(bpy.data.filepath).name,
                meters_per_unit=meters_per_unit,
                frame_start=context.scene.frame_start,
                frame_end=context.scene.frame_end,
                fps=context.scene.render.fps / context.scene.render.fps_base,
                objects=[_object_record(obj) for obj in context.scene.objects],
            )
            working_directory = _absolute_blender_path(settings.repo_root)
            entry_script = _absolute_blender_path(settings.entry_script)
            contract_path = _absolute_blender_path(settings.contract_path)
            job = build_solver_job(
                name=context.scene.name or "GeoTaichi Blender Job",
                description="Exported from Blender scene %s" % context.scene.name,
                entry_script=entry_script,
                working_directory=working_directory or str(Path(entry_script).parent),
                contract_path=contract_path,
                scene_manifest_path=scene_path.name,
                solver_family=settings.solver_family,
                solver_mode=settings.solver_mode,
                spatial_dimension=settings.spatial_dimension,
                model_arguments=shlex.split(settings.model_arguments),
                use_standard_arguments=settings.use_standard_arguments,
            )
            write_json_atomic(scene_path, manifest)
            write_json_atomic(job_path, job)
            state = _apply(settings, Event.EXPORT_SUCCEEDED, {"job_path": str(job_path)})
            self.report({"INFO"}, state.message)
            return {"FINISHED"}
        except Exception as exc:
            _apply(settings, Event.OPERATION_FAILED, {"error": "%s: %s" % (type(exc).__name__, exc)})
            self.report({"ERROR"}, settings.last_error)
            return {"CANCELLED"}


class GEOTAICHI_OT_validate_solver_job(Operator):
    bl_idname = "geotaichi.validate_solver_job"
    bl_label = "Validate"

    def execute(self, context):
        settings = context.scene.geotaichi
        client = _job_client(settings)
        job_path = settings.job_path
        if not _begin_background(context.scene, "validate", lambda: client.validate(job_path)):
            self.report({"WARNING"}, "Another GeoTaichi operation is already running")
            return {"CANCELLED"}
        return {"FINISHED"}


class GEOTAICHI_OT_submit_solver_job(Operator):
    bl_idname = "geotaichi.submit_solver_job"
    bl_label = "Submit"

    def execute(self, context):
        settings = context.scene.geotaichi
        client = _job_client(settings)
        job_path = settings.job_path
        confirmed = bool(settings.confirm_trusted_execution)
        if not _begin_background(
            context.scene,
            "submit",
            lambda: client.submit(job_path, confirmed),
        ):
            self.report({"WARNING"}, "Another GeoTaichi operation is already running")
            return {"CANCELLED"}
        return {"FINISHED"}


class GEOTAICHI_OT_refresh_task(Operator):
    bl_idname = "geotaichi.refresh_task"
    bl_label = "Refresh Status"

    def execute(self, context):
        settings = context.scene.geotaichi
        client = _job_client(settings)
        task_id = settings.task_id
        if not _begin_background(context.scene, "status", lambda: client.status(task_id)):
            self.report({"WARNING"}, "Another GeoTaichi operation is already running")
            return {"CANCELLED"}
        return {"FINISHED"}


class GEOTAICHI_OT_cancel_task(Operator):
    bl_idname = "geotaichi.cancel_task"
    bl_label = "Cancel Task"

    def execute(self, context):
        settings = context.scene.geotaichi
        client = _job_client(settings)
        task_id = settings.task_id

        def cancel_and_status():
            client.cancel(task_id)
            return client.status(task_id)

        if not _begin_background(context.scene, "cancel", cancel_and_status):
            self.report({"WARNING"}, "Another GeoTaichi operation is already running")
            return {"CANCELLED"}
        return {"FINISHED"}


class GEOTAICHI_OT_fetch_artifacts(Operator):
    bl_idname = "geotaichi.fetch_artifacts"
    bl_label = "Fetch to Cache"

    def execute(self, context):
        settings = context.scene.geotaichi
        client = _job_client(settings)
        task_id = settings.task_id
        scene_id = _ensure_scene_id(settings)
        cache_root = Path(_absolute_blender_path(settings.cache_root))

        def fetch():
            listing = client.artifacts(task_id)
            return cache_task_artifacts(
                listing["artifacts"],
                Path(listing["task_directory"]),
                cache_root,
                scene_id,
                task_id,
            )

        if not _begin_background(context.scene, "fetch", fetch):
            self.report({"WARNING"}, "Another GeoTaichi operation is already running")
            return {"CANCELLED"}
        return {"FINISHED"}


class GEOTAICHI_OT_reset_job_state(Operator):
    bl_idname = "geotaichi.reset_job_state"
    bl_label = "Reset Workflow"

    def execute(self, context):
        _apply(context.scene.geotaichi, Event.RESET)
        return {"FINISHED"}


CLASSES = (
    GEOTAICHI_OT_export_solver_job,
    GEOTAICHI_OT_validate_solver_job,
    GEOTAICHI_OT_submit_solver_job,
    GEOTAICHI_OT_refresh_task,
    GEOTAICHI_OT_cancel_task,
    GEOTAICHI_OT_fetch_artifacts,
    GEOTAICHI_OT_reset_job_state,
)


def register() -> None:
    global _SERVICE
    _SERVICE = SingleFlightService()
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    global _SERVICE, _PENDING_SCENE
    _AUTO_POLL_SCENES.clear()
    if _SERVICE is not None:
        _SERVICE.shutdown()
    _SERVICE = None
    _PENDING_SCENE = ""
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
