import taichi as ti
import numpy as np

from src.dem.structs.BaseStruct import BoundingBox, BoundingSphere, DeformableBoundingSphere
from src.mpm.soft_particle.Structs import DeformableGrid, SoftBody, SoftMaterialPoint
from src.utils.FieldIO import runtime_float_numpy_dtype
from src.utils.constants import Threshold


@ti.kernel
def count_soft_grid_allocated_slots_(soft_grid_num: int) -> int:
    return soft_grid_num


@ti.kernel
def count_soft_grid_mass_nodes_(soft_grid_num: int, soft_grid: ti.template()) -> int:
    count = 0
    for node in range(soft_grid_num):
        if soft_grid[node].m > Threshold:
            count += 1
    return count


def level_body_count(sims):
    if sims.scheme == "LSMPM":
        return sims.max_rigid_body_num + sims.max_soft_body_num
    return sims.max_rigid_body_num


def levelset_template_grid_count(sims):
    extra_soft_grids = sims.max_soft_body_num if sims.scheme == "LSMPM" else 0
    return sims.max_rigid_template_num + extra_soft_grids


def activate_soft_levelset_grid(scene, sims):
    if sims.max_soft_body_num <= 0:
        return
    soft_grid_capacity = max(sims.max_soft_grid_num * sims.max_soft_body_num, 1)
    scene.soft_grid = DeformableGrid.field(shape=soft_grid_capacity)
    scene.soft_grid_storage = sims.soft_grid_storage
    scene.soft_grid_capacity = soft_grid_capacity
    scene.soft_grid_owner = ti.field(int, shape=soft_grid_capacity)
    scene.soft_grid_local = ti.field(int, shape=soft_grid_capacity)
    scene.soft_velocity_constraint = ti.field(int, shape=max(sims.max_soft_velocity_constraint_num, 1))
    scene.levelset = None


def activate_soft_body(scene, sims):
    if sims.scheme != "LSMPM" or scene.soft is not None or sims.max_soft_body_num <= 0:
        return
    scene.soft = SoftBody.field(shape=sims.max_soft_body_num)
    if sims.max_level_grid_num <= 0:
        raise RuntimeError("Keyword:: /max_levelset_grid_num/ should be larger than 0")


def activate_soft_material_points(scene, sims):
    if scene.soft_point is not None:
        return
    if sims.max_material_point_num <= 0:
        raise RuntimeError("Keyword:: /max_material_point_number/ should be larger than 0")
    scene.soft_point = SoftMaterialPoint.field(shape=sims.max_material_point_num)
    scene.soft_support_shared = True
    soft_shape_nodes = sims.soft_shape_nodes
    template_point_capacity = max(sims.max_soft_template_point_num, 1)
    scene.soft_shape_node = ti.field(int, shape=(template_point_capacity, soft_shape_nodes))
    scene.soft_shape = ti.field(float, shape=(template_point_capacity, soft_shape_nodes))
    scene.soft_dshape = ti.Vector.field(3, float, shape=(template_point_capacity, soft_shape_nodes))
    scene.soft_shape_count = ti.field(int, shape=template_point_capacity)
    scene.soft_surface_point_id = ti.field(int, shape=sims.max_material_point_num)
    scene.soft_levelset_grad_error = ti.field(float, shape=())
    scene.soft_levelset_reference_sdf_volume = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_reference_material_volume = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_material_volume = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_target_volume = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_current_volume = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_interface_measure = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_volume_shift = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_volume_lower_shift = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_volume_upper_shift = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_volume_trial_shift = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_volume_cumulative_shift = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_volume_error = ti.field(float, shape=sims.max_soft_body_num)
    scene.soft_levelset_volume_max_error = ti.field(float, shape=())
    max_surface_nodes = max(sims.max_soft_template_surface_num, 1)
    surface_shape_nodes = 8
    scene.surface_shape_node = ti.field(int, shape=(max_surface_nodes, surface_shape_nodes))
    scene.surface_shape = ti.field(float, shape=(max_surface_nodes, surface_shape_nodes))
    scene.surface_shape_count = ti.field(int, shape=max_surface_nodes)
    max_sdf_nodes = max(sims.max_soft_template_sdf_num, 1)
    scene.sdf_shape_node = ti.field(int, shape=(max_sdf_nodes, surface_shape_nodes))
    scene.sdf_shape = ti.field(float, shape=(max_sdf_nodes, surface_shape_nodes))
    scene.sdf_shape_count = ti.field(int, shape=max_sdf_nodes)
    scene.soft_levelset_velocity = ti.Vector.field(3, float, shape=scene.rigid_grid.shape[0])
    scene.soft_levelset_initial_sdf = ti.field(float, shape=scene.rigid_grid.shape[0])
    scene.soft_levelset_initial_sdf_initialized = ti.field(int, shape=sims.max_soft_body_num)
    scene.soft_levelset_inverse_derivative = None
    if sims.soft_levelset_advection_scheme == "ReferenceMap":
        scene.soft_levelset_inverse_derivative = ti.Matrix.field(3, 3, float, shape=scene.rigid_grid.shape[0])
    scene.soft_levelset_projection_weight = ti.field(float, shape=scene.rigid_grid.shape[0])
    scene.soft_contact_trace_force = ti.Vector.field(3, float, shape=scene.rigid_grid.shape[0])
    scene.soft_contact_trace_uncovered = ti.field(int, shape=())
    scene.soft_levelset_projection_band_nodes = ti.field(int, shape=())
    scene.soft_levelset_projection_uncovered_nodes = ti.field(int, shape=())
    scene.soft_levelset_projection_max_support_loss = ti.field(float, shape=())
    level_body_num = sims.max_rigid_body_num + sims.max_soft_body_num
    max_contact_nodes = max(sims.max_surface_node_num * level_body_num, 1)
    scene.ls_contact_body = ti.field(int, shape=max_contact_nodes)
    scene.ls_contact_kind = ti.field(ti.u8, shape=max_contact_nodes)
    scene.ls_contact_ref = ti.field(int, shape=max_contact_nodes)
    scene.ls_contact_body_start = ti.field(int, shape=max(sims.max_particle_num, 1))
    scene.ls_contact_body_end = ti.field(int, shape=max(sims.max_particle_num, 1))
    scene.ls_contact_count = ti.field(int, shape=())


@ti.kernel
def upload_soft_point_support_(
    start: int,
    node: ti.types.ndarray(),
    shape: ti.types.ndarray(),
    dshape: ti.types.ndarray(),
    count: ti.types.ndarray(),
    target_node: ti.template(),
    target_shape: ti.template(),
    target_dshape: ti.template(),
    target_count: ti.template(),
):
    for i in range(node.shape[0]):
        target_count[start + i] = count[i]
        for a in range(node.shape[1]):
            target_node[start + i, a] = node[i, a]
            target_shape[start + i, a] = shape[i, a]
            target_dshape[start + i, a] = ti.Vector([dshape[i, a, 0], dshape[i, a, 1], dshape[i, a, 2]])


@ti.kernel
def upload_soft_trace_support_(
    start: int,
    node: ti.types.ndarray(),
    shape: ti.types.ndarray(),
    count: ti.types.ndarray(),
    target_node: ti.template(),
    target_shape: ti.template(),
    target_count: ti.template(),
):
    for i in range(node.shape[0]):
        target_count[start + i] = count[i]
        for a in range(node.shape[1]):
            target_node[start + i, a] = node[i, a]
            target_shape[start + i, a] = shape[i, a]


def register_soft_template_support(scene, support):
    key = id(support)
    registered = scene.soft_template_support_registry.get(key)
    if registered is not None:
        return registered

    point_start = int(scene.softTemplatePointNum[0])
    surface_start = int(scene.softTemplateSurfaceNum[0])
    sdf_start = int(scene.softTemplateSdfNum[0])
    point_end = point_start + support.material_point_number
    surface_end = surface_start + support.surface_node_number
    sdf_end = sdf_start + support.levelset_node_number
    if point_end > scene.soft_shape_count.shape[0]:
        raise ValueError(
            "The shared soft-template material-point support storage should " f"be enlarged to {point_end}"
        )
    if surface_end > scene.surface_shape_count.shape[0]:
        raise ValueError("The shared soft-template surface support storage should be " f"enlarged to {surface_end}")
    if sdf_end > scene.sdf_shape_count.shape[0]:
        raise ValueError("The shared soft-template SDF support storage should be enlarged " f"to {sdf_end}")
    if support.point_node.shape[1] > scene.soft_shape_node.shape[1]:
        raise ValueError("The configured soft shape-function support is too small")

    float_dtype = runtime_float_numpy_dtype()
    point_node = np.ascontiguousarray(support.point_node, dtype=np.int32)
    point_shape = np.ascontiguousarray(support.point_shape, dtype=float_dtype)
    point_dshape = np.ascontiguousarray(support.point_dshape, dtype=float_dtype)
    point_count = np.ascontiguousarray(support.point_count, dtype=np.int32)
    upload_soft_point_support_(
        point_start,
        point_node,
        point_shape,
        point_dshape,
        point_count,
        scene.soft_shape_node,
        scene.soft_shape,
        scene.soft_dshape,
        scene.soft_shape_count,
    )
    upload_soft_trace_support_(
        surface_start,
        np.ascontiguousarray(support.surface_node, dtype=np.int32),
        np.ascontiguousarray(support.surface_shape, dtype=float_dtype),
        np.ascontiguousarray(support.surface_count, dtype=np.int32),
        scene.surface_shape_node,
        scene.surface_shape,
        scene.surface_shape_count,
    )
    upload_soft_trace_support_(
        sdf_start,
        np.ascontiguousarray(support.sdf_node, dtype=np.int32),
        np.ascontiguousarray(support.sdf_shape, dtype=float_dtype),
        np.ascontiguousarray(support.sdf_count, dtype=np.int32),
        scene.sdf_shape_node,
        scene.sdf_shape,
        scene.sdf_shape_count,
    )
    registered = (point_start, surface_start, sdf_start, support.grid_type)
    scene.soft_template_support_registry[key] = registered
    scene.softTemplatePointNum[0] = point_end
    scene.softTemplateSurfaceNum[0] = surface_end
    scene.softTemplateSdfNum[0] = sdf_end
    return registered


def activate_soft_bounding_sphere(scene, sims):
    if scene.particle is not None or not (sims.max_rigid_body_num > 0 or sims.max_soft_body_num > 0):
        return
    if sims.max_soft_body_num > 0:
        scene.particle = DeformableBoundingSphere.field(shape=(sims.max_rigid_body_num + sims.max_soft_body_num))
    else:
        scene.particle = BoundingSphere.field(shape=sims.max_rigid_body_num)


def activate_soft_bounding_box(scene, sims):
    if scene.box is None and (sims.max_rigid_body_num > 0 or sims.max_soft_body_num > 0):
        scene.box = BoundingBox.field(shape=(sims.max_rigid_body_num + sims.max_soft_body_num))


def check_soft_body_number(scene, sims, soft_body_number):
    if scene.softNum[0] + soft_body_number > sims.max_soft_body_num:
        raise ValueError("The soft bodies should be set as: ", scene.softNum[0] + soft_body_number)


def check_material_point_number(scene, sims, material_point_number):
    if scene.softPointNum[0] + material_point_number > sims.max_material_point_num:
        raise ValueError("The soft material points should be set as: ", scene.softPointNum[0] + material_point_number)
