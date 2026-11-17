"""Import numbered GeoTaichi VTU files and bind them to Blender's timeline."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.types import Operator

from .core.vtu import VTUSequence, discover_vtu_sequences, read_vtu_geometry


_COLLECTION_NAME = "GeoTaichi Results"
_SEQUENCE_TAG = "_geotaichi_vtu_sequence"
_POINT_MODIFIER_NAME = "GeoTaichi Points"


def _absolute_blender_path(value: str) -> Path:
    return Path(bpy.path.abspath(value)).expanduser().resolve()


def _result_directory(settings) -> Path:
    value = settings.result_directory or settings.cache_directory or settings.output_directory
    if not value:
        raise ValueError("Select a result directory first")
    return _absolute_blender_path(value)


def _matching_sequences(sequences, prefix_filter: str):
    query = prefix_filter.strip().casefold()
    if not query:
        return sequences
    exact = tuple(sequence for sequence in sequences if sequence.prefix.casefold() == query)
    if exact:
        return exact
    return tuple(sequence for sequence in sequences if query in sequence.prefix.casefold())


def _result_collection(scene):
    collection = scene.collection.children.get(_COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(_COLLECTION_NAME)
        scene.collection.children.link(collection)
    return collection


def _sequence_name(sequence: VTUSequence) -> str:
    name = sequence.prefix.rstrip("_.- ") or "VTU"
    return "GeoTaichi %s" % name


def _find_sequence_object(scene, sequence: VTUSequence):
    directory = str(sequence.directory)
    for obj in scene.objects:
        if not obj.get(_SEQUENCE_TAG):
            continue
        if obj.get("_geotaichi_vtu_directory") == directory and obj.get("_geotaichi_vtu_prefix") == sequence.prefix:
            return obj
    return None


def _new_sequence_object(scene, sequence: VTUSequence):
    mesh = bpy.data.meshes.new(_sequence_name(sequence))
    obj = bpy.data.objects.new(_sequence_name(sequence), mesh)
    _result_collection(scene).objects.link(obj)
    return obj


def _store_sequence(obj, sequence: VTUSequence, start_frame: int, frame_step: int) -> None:
    obj[_SEQUENCE_TAG] = True
    obj["_geotaichi_vtu_directory"] = str(sequence.directory)
    obj["_geotaichi_vtu_prefix"] = sequence.prefix
    obj["_geotaichi_vtu_frame_numbers"] = json.dumps(sequence.frame_numbers, separators=(",", ":"))
    obj["_geotaichi_vtu_frame_width"] = sequence.frame_width
    obj["_geotaichi_vtu_start_frame"] = start_frame
    obj["_geotaichi_vtu_frame_step"] = frame_step
    obj["_geotaichi_vtu_current_frame"] = -1


def _sequence_from_object(obj) -> VTUSequence:
    frame_numbers = tuple(int(value) for value in json.loads(obj["_geotaichi_vtu_frame_numbers"]))
    return VTUSequence(
        directory=Path(obj["_geotaichi_vtu_directory"]),
        prefix=str(obj["_geotaichi_vtu_prefix"]),
        frame_numbers=frame_numbers,
        frame_width=int(obj["_geotaichi_vtu_frame_width"]),
    )


def _point_modifier(obj):
    for modifier in obj.modifiers:
        if modifier.type == "NODES" and modifier.name == _POINT_MODIFIER_NAME:
            return modifier
    return None


def _enable_point_display(obj, radius: float) -> None:
    modifier = _point_modifier(obj)
    if modifier is None:
        modifier = obj.modifiers.new(name=_POINT_MODIFIER_NAME, type="NODES")
        node_group = bpy.data.node_groups.new(name=_POINT_MODIFIER_NAME, type="GeometryNodeTree")
        node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        input_node = node_group.nodes.new("NodeGroupInput")
        output_node = node_group.nodes.new("NodeGroupOutput")
        points_node = node_group.nodes.new("GeometryNodeMeshToPoints")
        points_node.name = "GeoTaichi Mesh to Points"
        points_node.mode = "VERTICES"
        node_group.links.new(input_node.outputs["Geometry"], points_node.inputs["Mesh"])
        node_group.links.new(points_node.outputs["Points"], output_node.inputs["Geometry"])
        modifier.node_group = node_group
    points_node = modifier.node_group.nodes.get("GeoTaichi Mesh to Points")
    if points_node is not None:
        points_node.inputs["Radius"].default_value = radius
    modifier.show_viewport = True
    modifier.show_render = True


def _disable_point_display(obj) -> None:
    modifier = _point_modifier(obj)
    if modifier is not None:
        modifier.show_viewport = False
        modifier.show_render = False


def _replace_geometry(obj, frame, point_radius: float) -> None:
    mesh = obj.data
    mesh.clear_geometry()
    mesh.from_pydata(frame.points, frame.edges, frame.faces)
    mesh.update()
    if frame.is_point_cloud:
        _enable_point_display(obj, point_radius)
    else:
        _disable_point_display(obj)


def _frame_index(obj, blender_frame: int, frame_count: int) -> int:
    start_frame = int(obj["_geotaichi_vtu_start_frame"])
    frame_step = int(obj["_geotaichi_vtu_frame_step"])
    return max(0, min(frame_count - 1, (blender_frame - start_frame) // frame_step))


def _update_object_frame(obj, blender_frame: int, point_radius: float, force: bool = False) -> bool:
    sequence = _sequence_from_object(obj)
    if not sequence.frame_numbers:
        return False
    source_index = _frame_index(obj, blender_frame, len(sequence.frame_numbers))
    source_frame = sequence.frame_numbers[source_index]
    if not force and int(obj.get("_geotaichi_vtu_current_frame", -1)) == source_frame:
        return False
    frame = read_vtu_geometry(sequence.path_at(source_index))
    _replace_geometry(obj, frame, point_radius)
    obj["_geotaichi_vtu_current_frame"] = source_frame
    return True


def _update_scene_sequences(scene, force: bool = False) -> int:
    settings = getattr(scene, "geotaichi", None)
    if settings is None:
        return 0
    updated = 0
    errors = []
    for obj in scene.objects:
        if not obj.get(_SEQUENCE_TAG) or obj.type != "MESH":
            continue
        try:
            updated += int(_update_object_frame(obj, scene.frame_current, settings.result_point_radius, force))
        except Exception as exc:
            errors.append("%s: %s" % (obj.name, exc))
    settings.result_error = "; ".join(errors)
    return updated


@persistent
def _frame_change_pre(scene, _depsgraph=None):
    _update_scene_sequences(scene)


@persistent
def _load_post(_unused):
    for scene in bpy.data.scenes:
        _update_scene_sequences(scene, force=True)


class GEOTAICHI_OT_import_vtu_sequences(Operator):
    bl_idname = "geotaichi.import_vtu_sequences"
    bl_label = "Import VTU Sequences"
    bl_description = "Discover numbered VTU files, create result objects, and bind them to the timeline"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.geotaichi
        try:
            directory = _result_directory(settings)
            sequences = discover_vtu_sequences(directory, recursive=settings.result_recursive)
            sequences = _matching_sequences(sequences, settings.result_prefix_filter)
            if not sequences:
                raise ValueError("No numbered VTU sequences match the selected directory and prefix")
            imported_frames = 0
            for sequence in sequences:
                obj = _find_sequence_object(context.scene, sequence) if settings.result_replace_existing else None
                if obj is None:
                    obj = _new_sequence_object(context.scene, sequence)
                _store_sequence(obj, sequence, settings.result_start_frame, settings.result_frame_step)
                _update_object_frame(obj, settings.result_start_frame, settings.result_point_radius, force=True)
                imported_frames = max(imported_frames, len(sequence.frame_numbers))
            if settings.result_update_timeline:
                context.scene.frame_start = settings.result_start_frame
                context.scene.frame_end = (
                    settings.result_start_frame + (imported_frames - 1) * settings.result_frame_step
                )
                context.scene.frame_set(settings.result_start_frame)
            settings.result_directory = str(directory)
            settings.result_error = ""
            settings.result_message = "Imported %d VTU sequence(s), up to %d frame(s)" % (
                len(sequences),
                imported_frames,
            )
            self.report({"INFO"}, settings.result_message)
            return {"FINISHED"}
        except Exception as exc:
            settings.result_error = "%s: %s" % (type(exc).__name__, exc)
            self.report({"ERROR"}, settings.result_error)
            return {"CANCELLED"}


class GEOTAICHI_OT_reload_vtu_frame(Operator):
    bl_idname = "geotaichi.reload_vtu_frame"
    bl_label = "Reload Current Frame"
    bl_description = "Reload the current source files for all imported VTU sequences"

    def execute(self, context):
        updated = _update_scene_sequences(context.scene, force=True)
        settings = context.scene.geotaichi
        if settings.result_error:
            self.report({"ERROR"}, settings.result_error)
            return {"CANCELLED"}
        settings.result_message = "Reloaded %d VTU sequence(s)" % updated
        self.report({"INFO"}, settings.result_message)
        return {"FINISHED"}


CLASSES = (
    GEOTAICHI_OT_import_vtu_sequences,
    GEOTAICHI_OT_reload_vtu_frame,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    if _frame_change_pre not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_frame_change_pre)
    if _load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_post)


def unregister() -> None:
    if _load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post)
    if _frame_change_pre in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_frame_change_pre)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
