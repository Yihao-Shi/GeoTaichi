"""Persistent Blender properties for the thin SolverJob client."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import PropertyGroup

from .defaults import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DIMENSION,
    DEFAULT_EXPORT_DIRECTORY,
    DEFAULT_OBJECT_ROLE,
    DEFAULT_PHASE,
    DEFAULT_RESULT_FRAME_STEP,
    DEFAULT_RESULT_POINT_RADIUS,
    DEFAULT_RESULT_RECURSIVE,
    DEFAULT_RESULT_REPLACE_EXISTING,
    DEFAULT_RESULT_START_FRAME,
    DEFAULT_RESULT_UPDATE_TIMELINE,
    DEFAULT_SOLVER_FAMILY,
    DEFAULT_SOLVER_MODE,
    DEFAULT_USE_STANDARD_ARGUMENTS,
    DIMENSION_ITEMS,
    PHASE_ITEMS,
    ROLE_ITEMS,
    SOLVER_ITEMS,
    SOLVER_MODE_ITEMS,
    default_python_executable,
)


class GeoTaichiSceneSettings(PropertyGroup):
    scene_id: StringProperty(name="Scene ID", default="")
    phase: EnumProperty(name="Phase", items=PHASE_ITEMS, default=DEFAULT_PHASE)
    solver_family: EnumProperty(
        name="Solver",
        items=SOLVER_ITEMS,
        default=DEFAULT_SOLVER_FAMILY,
        description="Public GeoTaichi facade used by the selected model script",
    )
    solver_mode: EnumProperty(
        name="Mode",
        items=SOLVER_MODE_ITEMS,
        default=DEFAULT_SOLVER_MODE,
        description="Configured numerical or coupling path selected by the model script",
    )
    spatial_dimension: EnumProperty(
        name="Dimension",
        items=DIMENSION_ITEMS,
        default=DEFAULT_DIMENSION,
    )
    python_executable: StringProperty(
        name="Python",
        description="Python executable where GeoTaichi and geotaichi_mcp are installed",
        default=default_python_executable(),
        subtype="FILE_PATH",
    )
    repo_root: StringProperty(name="Repository", default="", subtype="DIR_PATH")
    workspace: StringProperty(name="Task Workspace", default="", subtype="DIR_PATH")
    entry_script: StringProperty(name="Entry Script", default="", subtype="FILE_PATH")
    contract_path: StringProperty(name="Model Contract", default="", subtype="FILE_PATH")
    use_standard_arguments: BoolProperty(
        name="Standard Model Arguments",
        description="Pass staged contract, scene manifest, and output paths as named arguments",
        default=DEFAULT_USE_STANDARD_ARGUMENTS,
    )
    model_arguments: StringProperty(
        name="Additional Arguments",
        description="Shell-style arguments appended after the standard model arguments",
        default="",
    )
    export_directory: StringProperty(name="Job Directory", default=DEFAULT_EXPORT_DIRECTORY, subtype="DIR_PATH")
    cache_root: StringProperty(name="Cache Root", default=DEFAULT_CACHE_ROOT, subtype="DIR_PATH")
    confirm_trusted_execution: BoolProperty(
        name="Allow Trusted Script Execution",
        description="The selected Python model can execute arbitrary local code",
        default=False,
    )
    job_path: StringProperty(name="SolverJob", default="", subtype="FILE_PATH")
    task_id: StringProperty(name="Task ID", default="")
    task_status: StringProperty(name="Task Status", default="")
    output_directory: StringProperty(name="Task Output", default="", subtype="DIR_PATH")
    cache_directory: StringProperty(name="Cached Output", default="", subtype="DIR_PATH")
    operation: StringProperty(name="Background Operation", default="")
    last_message: StringProperty(name="Message", default="")
    last_error: StringProperty(name="Error", default="")
    result_directory: StringProperty(
        name="Result Directory",
        description="Directory containing VTU files; fetched task output is used when this is empty",
        default="",
        subtype="DIR_PATH",
    )
    result_prefix_filter: StringProperty(
        name="Prefix Filter",
        description="Optional VTU filename prefix; leave empty to import every numbered sequence",
        default="",
    )
    result_start_frame: IntProperty(
        name="Start",
        description="Blender frame assigned to the first VTU file",
        default=DEFAULT_RESULT_START_FRAME,
    )
    result_frame_step: IntProperty(
        name="Step",
        description="Number of Blender frames between consecutive VTU files",
        default=DEFAULT_RESULT_FRAME_STEP,
        min=1,
    )
    result_point_radius: FloatProperty(
        name="Point Radius",
        description="Display radius for point-only MPM and DEM VTU results",
        default=DEFAULT_RESULT_POINT_RADIUS,
        min=1.0e-9,
        soft_max=1.0,
        subtype="DISTANCE",
    )
    result_recursive: BoolProperty(
        name="Search Subdirectories",
        description="Discover VTU sequences below the selected result directory",
        default=DEFAULT_RESULT_RECURSIVE,
    )
    result_replace_existing: BoolProperty(
        name="Update Existing",
        description="Reuse an imported object when its directory and prefix match",
        default=DEFAULT_RESULT_REPLACE_EXISTING,
    )
    result_update_timeline: BoolProperty(
        name="Set Timeline Range",
        description="Set the scene start and end frames from the imported sequences",
        default=DEFAULT_RESULT_UPDATE_TIMELINE,
    )
    result_message: StringProperty(name="Result Message", default="")
    result_error: StringProperty(name="Result Error", default="")


CLASSES = (GeoTaichiSceneSettings,)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.geotaichi = PointerProperty(type=GeoTaichiSceneSettings)
    bpy.types.Object.geotaichi_role = EnumProperty(name="GeoTaichi Role", items=ROLE_ITEMS, default=DEFAULT_OBJECT_ROLE)


def unregister() -> None:
    if hasattr(bpy.types.Object, "geotaichi_role"):
        del bpy.types.Object.geotaichi_role
    if hasattr(bpy.types.Scene, "geotaichi"):
        del bpy.types.Scene.geotaichi
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
