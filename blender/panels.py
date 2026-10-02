"""GeoTaichi SolverJob panel in Blender's 3D View sidebar."""

from __future__ import annotations

import bpy
from bpy.types import Panel


class GEOTAICHI_PT_solver_job(Panel):
    bl_label = "GeoTaichi"
    bl_idname = "GEOTAICHI_PT_solver_job"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GeoTaichi"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.geotaichi

        status = layout.box()
        status.label(text="Phase: %s" % settings.phase.title())
        if settings.task_id:
            status.label(text="Task: %s" % settings.task_id)
        if settings.last_message:
            status.label(text=settings.last_message, icon="INFO")
        if settings.last_error:
            status.label(text=settings.last_error, icon="ERROR")
        if settings.operation:
            status.label(text="Working: %s" % settings.operation, icon="TIME")

        configuration = layout.box()
        configuration.label(text="Execution")
        configuration.prop(settings, "solver_family")
        configuration.prop(settings, "solver_mode")
        configuration.prop(settings, "spatial_dimension")
        configuration.prop(settings, "python_executable")
        configuration.prop(settings, "repo_root")
        configuration.prop(settings, "workspace")
        configuration.prop(settings, "entry_script")
        configuration.prop(settings, "contract_path")
        configuration.prop(settings, "use_standard_arguments")
        configuration.prop(settings, "model_arguments")
        configuration.prop(settings, "export_directory")
        configuration.prop(settings, "cache_root")

        active = context.active_object
        if active is not None:
            object_box = layout.box()
            object_box.label(text="Active Object")
            object_box.prop(active, "geotaichi_role")

        workflow = layout.box()
        workflow.label(text="SolverJob Workflow")
        workflow.enabled = not bool(settings.operation)
        workflow.operator("geotaichi.export_solver_job", icon="EXPORT")
        row = workflow.row(align=True)
        row.enabled = bool(settings.job_path)
        row.operator("geotaichi.validate_solver_job", icon="CHECKMARK")
        workflow.prop(settings, "confirm_trusted_execution")
        submit = workflow.row()
        submit.enabled = bool(settings.job_path and settings.confirm_trusted_execution)
        submit.operator("geotaichi.submit_solver_job", icon="PLAY")

        task_row = workflow.row(align=True)
        task_row.enabled = bool(settings.task_id)
        task_row.operator("geotaichi.refresh_task", icon="FILE_REFRESH")
        cancel = task_row.row(align=True)
        cancel.enabled = settings.phase in {"SUBMITTED", "RUNNING"}
        cancel.operator("geotaichi.cancel_task", icon="CANCEL")
        fetch = workflow.row()
        fetch.enabled = settings.phase in {"COMPLETED", "FAILED", "INTERRUPTED"} and bool(settings.output_directory)
        fetch.operator("geotaichi.fetch_artifacts", icon="IMPORT")
        workflow.operator("geotaichi.reset_job_state", icon="LOOP_BACK")


class GEOTAICHI_PT_results(Panel):
    bl_label = "VTU Results"
    bl_idname = "GEOTAICHI_PT_results"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GeoTaichi"
    bl_parent_id = "GEOTAICHI_PT_solver_job"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.geotaichi

        if settings.result_message:
            layout.label(text=settings.result_message, icon="INFO")
        if settings.result_error:
            layout.label(text=settings.result_error, icon="ERROR")
        layout.prop(settings, "result_directory")
        layout.prop(settings, "result_prefix_filter")
        row = layout.row(align=True)
        row.prop(settings, "result_start_frame")
        row.prop(settings, "result_frame_step")
        layout.prop(settings, "result_point_radius")
        options = layout.column(align=True)
        options.prop(settings, "result_recursive")
        options.prop(settings, "result_replace_existing")
        options.prop(settings, "result_update_timeline")
        layout.operator("geotaichi.import_vtu_sequences", icon="IMPORT")
        layout.operator("geotaichi.reload_vtu_frame", icon="FILE_REFRESH")


CLASSES = (GEOTAICHI_PT_solver_job, GEOTAICHI_PT_results)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
