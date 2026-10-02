import math

import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.utils.constants import Threshold, WENO_EPS, ZEROVEC3f
from src.utils.Quaternion import SetDQ, SetToRotate
from src.utils.ShapeFunctions import ShapeBsplineC, ShapeBsplineQ, ShapeLinear
from src.utils.TypeDefination import vec3f, vec3i, mat3x3
from src.utils.VectorFunction import Normalize
from src.mpm.soft_particle.LevelSet import (
    monitor_soft_levelset_signed_distance_error_,
    reinitialize_soft_levelset_step_,
    sample_levelset_reference_bounds_unscaled,
    sample_levelset_unscaled,
    split_soft_grid_slot,
    store_soft_levelset_reinit_reference_,
)


@ti.func
def soft_point_support_index_(point, material_point, soft, rigid):
    bodyID = material_point[point].bodyID
    sb = rigid[bodyID].softID
    return sb, soft[sb].templatePointStart + point - soft[sb].startIndex


@ti.func
def soft_point_support_node_(sb, support, basis, soft, shape_node):
    return soft[sb].mpmGridStart + shape_node[support, basis]


@ti.func
def soft_surface_support_index_(surface_node, bodyID, soft, rigid):
    sb = rigid[bodyID].softID
    return sb, (soft[sb].templateSurfaceStart + surface_node - soft[sb].startNode)


@ti.func
def soft_surface_levelset_error_at_node_(bodyID, local_node, box, grid, vertice):
    local_position = box[bodyID].scale * vertice[local_node].x
    return ti.abs(box[bodyID].distance(local_position, grid))


@ti.kernel
def soft_surface_force_reset_(
    softNum: int,
    surfaceNum: int,
    soft: ti.template(),
    surface: ti.template(),
    rigid: ti.template(),
    vertice: ti.template(),
):
    for sb in range(softNum):
        soft[sb].contact_force = ZEROVEC3f
    for ns in range(surfaceNum):
        bodyID = surface[ns]
        local_node = rigid[bodyID].global_node_to_local(ns)
        vertice[local_node]._reset_contact()


@ti.kernel
def soft_material_point_force_reset_(pointNum: int, material_point: ti.template()):
    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            material_point[p]._reset_contact()


@ti.kernel
def max_soft_surface_levelset_error_(
    softNum: int,
    soft: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    grid: ti.template(),
    vertice: ti.template(),
) -> float:
    max_error = 0.0
    for sb in range(softNum):
        bodyID = soft[sb].bodyID
        scale = box[bodyID].scale
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            local_position = scale * vertice[local_node].x
            ti.atomic_max(max_error, ti.abs(box[bodyID].distance(local_position, grid)))
    return max_error


@ti.kernel
def sum_soft_surface_levelset_error_(
    softNum: int,
    soft: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    grid: ti.template(),
    vertice: ti.template(),
) -> float:
    error_sum = 0.0
    for sb in range(softNum):
        bodyID = soft[sb].bodyID
        scale = box[bodyID].scale
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            local_position = scale * vertice[local_node].x
            ti.atomic_add(error_sum, ti.abs(box[bodyID].distance(local_position, grid)))
    return error_sum


@ti.kernel
def sum_soft_surface_levelset_area_weight_(
    softNum: int, soft: ti.template(), box: ti.template(), vertice: ti.template()
) -> float:
    area_sum = 0.0
    for sb in range(softNum):
        bodyID = soft[sb].bodyID
        physical_area = box[bodyID].reference_surface_area * box[bodyID].scale ** 2
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            ti.atomic_add(
                area_sum,
                physical_area * ti.max(vertice[local_node].parameter, 0.0),
            )
    return area_sum


@ti.kernel
def sum_soft_surface_levelset_area_weighted_error_(
    softNum: int, soft: ti.template(), box: ti.template(), grid: ti.template(), vertice: ti.template()
) -> float:
    error_sum = 0.0
    for sb in range(softNum):
        bodyID = soft[sb].bodyID
        scale = box[bodyID].scale
        physical_area = box[bodyID].reference_surface_area * scale**2
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            local_position = scale * vertice[local_node].x
            area = physical_area * ti.max(vertice[local_node].parameter, 0.0)
            error = ti.abs(box[bodyID].distance(local_position, grid))
            ti.atomic_add(error_sum, area * error)
    return error_sum


@ti.kernel
def sum_soft_surface_levelset_area_weighted_squared_error_(
    softNum: int, soft: ti.template(), box: ti.template(), grid: ti.template(), vertice: ti.template()
) -> float:
    error_sum = 0.0
    for sb in range(softNum):
        bodyID = soft[sb].bodyID
        scale = box[bodyID].scale
        physical_area = box[bodyID].reference_surface_area * scale**2
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            local_position = scale * vertice[local_node].x
            area = physical_area * ti.max(vertice[local_node].parameter, 0.0)
            error = box[bodyID].distance(local_position, grid)
            ti.atomic_add(error_sum, area * error * error)
    return error_sum


@ti.kernel
def min_soft_surface_active_shape_sum_(
    surfaceNum: int,
    surface: ti.template(),
    rigid: ti.template(),
    soft: ti.template(),
    soft_grid: ti.template(),
    surface_shape_node: ti.template(),
    surface_shape: ti.template(),
    surface_shape_count: ti.template(),
) -> float:
    minimum = 1.0e30
    for ns in range(surfaceNum):
        bodyID = surface[ns]
        if int(rigid[bodyID].is_soft) == 1:
            sb, support = soft_surface_support_index_(ns, bodyID, soft, rigid)
            active_weight = 0.0
            for n in range(surface_shape_count[support]):
                node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                if soft_grid[node].m > Threshold:
                    active_weight += surface_shape[support, n]
            ti.atomic_min(minimum, active_weight)
    return minimum


@ti.kernel
def count_soft_surface_without_active_support_(
    surfaceNum: int,
    surface: ti.template(),
    rigid: ti.template(),
    soft: ti.template(),
    soft_grid: ti.template(),
    surface_shape_node: ti.template(),
    surface_shape: ti.template(),
    surface_shape_count: ti.template(),
) -> int:
    unsupported = 0
    for ns in range(surfaceNum):
        bodyID = surface[ns]
        if int(rigid[bodyID].is_soft) == 1:
            sb, support = soft_surface_support_index_(ns, bodyID, soft, rigid)
            active_weight = 0.0
            for n in range(surface_shape_count[support]):
                node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                if soft_grid[node].m > Threshold:
                    active_weight += surface_shape[support, n]
            if active_weight <= Threshold:
                ti.atomic_add(unsupported, 1)
    return unsupported


@ti.kernel
def min_soft_levelset_domain_margin_(
    softNum: int,
    soft: ti.template(),
    box: ti.template(),
) -> float:
    """Minimum deformed-body clearance from the fixed SDF box, in cells."""
    minimum_margin = 1.0e30
    for sb in range(softNum):
        bodyID = soft[sb].bodyID
        spacing = ti.max(box[bodyID].grid_space, Threshold)
        for d in ti.static(range(3)):
            lower_margin = (box[bodyID].shape_min[d] - box[bodyID].xmin[d]) / spacing
            upper_margin = (box[bodyID].xmax[d] - box[bodyID].shape_max[d]) / spacing
            ti.atomic_min(minimum_margin, lower_margin)
            ti.atomic_min(minimum_margin, upper_margin)
    return minimum_margin


@ti.func
def update_soft_grid_node_(grid_slot, softNum, dt, soft, soft_grid, soft_grid_owner, soft_grid_local, box, material):
    sb, local_grid = split_soft_grid_slot(grid_slot, soft_grid_owner, soft_grid_local)
    if sb < softNum and local_grid < soft[sb].mpmGridNum:
        matID = int(soft[sb].materialID)
        damp = material[matID]._get_force_damping_coefficient()
        if soft_grid[grid_slot].m > Threshold:
            velocity = soft_grid[grid_slot].v
            force = soft_grid[grid_slot].f
            if damp > 0.0:
                density = material[matID]._get_density()
                young = material[matID]._get_young_modulus()
                bodyID = soft[sb].bodyID
                h = soft[sb].scale * soft[sb].gridSpace
                damping_force = -damp * soft_grid[grid_slot].m * ti.sqrt(young / (density * h * h)) * velocity
                force += damping_force
                if ti.static(GlobalVariable.TRACKENERGY):
                    ti.atomic_add(
                        soft[sb].damp_energy,
                        damping_force.dot(velocity) * dt[None],
                    )
            acceleration = force / soft_grid[grid_slot].m
            soft_grid[grid_slot].f = force
            soft_grid[grid_slot].v = velocity + acceleration * dt[None]


@ti.kernel
def precompute_soft_grid_reference_mass_(
    softNum: int,
    softGridNum: int,
    pointNum: int,
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_shape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
):
    """Assemble invariant mass and the initial cross-step grid velocity."""
    for node in range(softGridNum):
        soft_grid[node].m = 0.0
        soft_grid[node].v = ZEROVEC3f

    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index_(p, material_point, soft, rigid)
            if sb < softNum:
                for n in range(soft_shape_count[support]):
                    node = soft_point_support_node_(sb, support, n, soft, soft_shape_node)
                    ti.atomic_add(
                        soft_grid[node].m,
                        soft_shape[support, n] * material_point[p].m,
                    )
                    ti.atomic_add(
                        soft_grid[node].v,
                        soft_shape[support, n] * material_point[p].m * material_point[p].v,
                    )

    for node in range(softGridNum):
        if soft_grid[node].m > Threshold:
            soft_grid[node].v /= soft_grid[node].m


@ti.kernel
def initialize_soft_particle_constitutive_state_(
    pointNum: int,
    material_point: ti.template(),
    soft_matProps: ti.template(),
    soft_stateVars: ti.template(),
):
    """Initialize the stress cache consumed by the first explicit step."""
    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            material_id = material_point[p].materialID
            stress = soft_matProps.soft_particle_pk1(material_id, material_point[p].F)
            material_point[p].stress = stress
            if ti.static(GlobalVariable.TRACKENERGY):
                material_point[p].strain_energy = material_point[p].vol0 * soft_matProps.Psi(
                    material_id, material_point[p].F
                )
            soft_matProps.update_soft_particle_state(p, material_id, material_point[p].F, stress, soft_stateVars)


@ti.kernel
def reset_soft_grid_step_(
    softGridNum: int,
    soft_grid: ti.template(),
):
    for ng in range(softGridNum):
        soft_grid[ng]._tlmpm_step_reset()


@ti.kernel
def soft_body_force_p2g_(
    pointNum: int,
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_shape: ti.template(),
    soft_dshape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
    gravity: ti.types.vector(3, float),
):
    # The stress evaluated after the previous F update is the stress at the
    # current time level.  The first step receives the same cache from
    # initialize_soft_particle_constitutive_state_.
    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index_(p, material_point, soft, rigid)
            stress = material_point[p].stress
            template_stress = stress @ soft[sb].referenceRotation / soft[sb].scale
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node_(sb, support, n, soft, soft_shape_node)
                weight = soft_shape[support, n]
                template_grad = soft_dshape[support, n]
                body_force = weight * material_point[p].m * gravity
                internal_force = -material_point[p].vol0 * (template_stress @ template_grad)
                contact_force = weight * material_point[p].contact_force
                bodyID = material_point[p].bodyID
                external_force = (
                    weight * soft[rigid[bodyID].softID].external_load_factor * material_point[p].external_force
                )
                ti.atomic_add(
                    soft_grid[node].f,
                    body_force + internal_force + contact_force + external_force,
                )


@ti.kernel
def soft_surface_force_p2g_(
    surfaceNum: int,
    soft: ti.template(),
    soft_grid: ti.template(),
    surface_shape_node: ti.template(),
    surface_shape: ti.template(),
    surface_shape_count: ti.template(),
    vertice: ti.template(),
    surface: ti.template(),
    rigid: ti.template(),
):
    for ns in range(surfaceNum):
        bodyID = surface[ns]
        if int(rigid[bodyID].is_soft) == 1:
            sb, support = soft_surface_support_index_(ns, bodyID, soft, rigid)
            if surface_shape_count[support] > 0:
                local_node = rigid[bodyID].global_node_to_local(ns)
                cforce = (
                    vertice[local_node].contact_force
                    + soft[rigid[bodyID].softID].external_load_factor * vertice[local_node].external_force
                )
                if cforce.dot(cforce) > Threshold * Threshold:
                    active_weight = 0.0
                    for n in range(surface_shape_count[support]):
                        node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                        if soft_grid[node].m > Threshold:
                            active_weight += surface_shape[support, n]
                    # Boundary traces can include empty exterior nodes. Renormalize
                    # over the mass-carrying support so no contact resultant is lost.
                    for n in range(surface_shape_count[support]):
                        node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                        if soft_grid[node].m > Threshold and active_weight > Threshold:
                            ti.atomic_add(
                                soft_grid[node].f,
                                surface_shape[support, n] / active_weight * cforce,
                            )


@ti.kernel
def update_soft_grid_kinematic_(
    softNum: int,
    softGridNum: int,
    fixedGridNum: int,
    dt: ti.template(),
    soft: ti.template(),
    soft_grid: ti.template(),
    soft_grid_owner: ti.template(),
    soft_grid_local: ti.template(),
    fixed_grid: ti.template(),
    box: ti.template(),
    material: ti.template(),
):
    for grid_slot in range(softGridNum):
        update_soft_grid_node_(grid_slot, softNum, dt, soft, soft_grid, soft_grid_owner, soft_grid_local, box, material)

    # As in the standard MPM VelocityConstraint, only the precomputed sparse
    # node list is traversed; no boundary flag is stored on grid nodes.
    for boundary in range(fixedGridNum):
        node = fixed_grid[boundary]
        if node < softGridNum and soft_grid[node].m > Threshold:
            soft_grid[node].v = ZEROVEC3f
            soft_grid[node].f = ZEROVEC3f


@ti.kernel
def soft_body_g2p_(
    pointNum: int,
    pic_fraction: float,
    dt: ti.template(),
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_shape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
):
    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index_(p, material_point, soft, rigid)
            acceleration = ZEROVEC3f
            velocity = ZEROVEC3f
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node_(sb, support, n, soft, soft_shape_node)
                if soft_grid[node].m > Threshold:
                    weight = soft_shape[support, n]
                    acceleration += weight * soft_grid[node].f / soft_grid[node].m
                    velocity += weight * soft_grid[node].v
            velocity_flip = material_point[p].v + acceleration * dt[None]
            material_point[p].v = pic_fraction * velocity + (1.0 - pic_fraction) * velocity_flip
            material_point[p].x += velocity * dt[None]


@ti.kernel
def reset_soft_body_kinematic_reduction_(
    softNum: int,
    soft: ti.template(),
):
    for sb in range(softNum):
        soft[sb].previous_center = ZEROVEC3f
        soft[sb].v = ZEROVEC3f


@ti.kernel
def reduce_soft_body_kinematics_(
    pointNum: int,
    soft: ti.template(),
    material_point: ti.template(),
    rigid: ti.template(),
):
    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            bodyID = material_point[p].bodyID
            sb = rigid[bodyID].softID
            ti.atomic_add(
                soft[sb].previous_center,
                material_point[p].m * material_point[p].x,
            )
            ti.atomic_add(
                soft[sb].v,
                material_point[p].m * material_point[p].v,
            )


@ti.kernel
def apply_soft_body_translation_constraint_(
    pointNum: int,
    soft: ti.template(),
    material_point: ti.template(),
    rigid: ti.template(),
):
    # Soft bodies do not pass through the rigid-body integrator. Remove only
    # their constrained rigid-translation mode, in parallel over points.
    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            bodyID = material_point[p].bodyID
            sb = rigid[bodyID].softID
            inv_mass = 1.0 / ti.max(soft[sb].m, Threshold)
            center = soft[sb].previous_center * inv_mass
            velocity = soft[sb].v * inv_mass
            fixed = vec3f(
                1.0 - ti.cast(rigid[bodyID].is_fix[0], float),
                1.0 - ti.cast(rigid[bodyID].is_fix[1], float),
                1.0 - ti.cast(rigid[bodyID].is_fix[2], float),
            )
            material_point[p].x += fixed * (soft[sb].mass_center0 - center)
            material_point[p].v -= fixed * velocity


@ti.kernel
def remap_soft_grid_velocity_(
    softGridNum: int,
    fixedGridNum: int,
    pointNum: int,
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    fixed_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_shape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
):
    for ng in range(softGridNum):
        if soft_grid[ng].m > Threshold:
            soft_grid[ng].v = ZEROVEC3f

    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index_(p, material_point, soft, rigid)
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node_(sb, support, n, soft, soft_shape_node)
                weight = soft_shape[support, n]
                ti.atomic_add(soft_grid[node].v, weight * material_point[p].m * material_point[p].v)

    for ng in range(softGridNum):
        if soft_grid[ng].m > Threshold:
            soft_grid[ng].v /= soft_grid[ng].m

    # This corrected velocity field is used below for F and surface updates,
    # then retained as the initial grid velocity of the next time step.

    for boundary in range(fixedGridNum):
        node = fixed_grid[boundary]
        if node < softGridNum and soft_grid[node].m > Threshold:
            soft_grid[node].v = ZEROVEC3f


@ti.kernel
def update_soft_particle_stress_(
    pointNum: int,
    dt: ti.template(),
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_dshape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
    soft_matProps: ti.template(),
    soft_stateVars: ti.template(),
):
    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index_(p, material_point, soft, rigid)
            template_F_rate = mat3x3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node_(sb, support, n, soft, soft_shape_node)
                if soft_grid[node].m > Threshold:
                    template_F_rate += soft_grid[node].v.outer_product(soft_dshape[support, n])
            F_rate = template_F_rate @ soft[sb].referenceRotation.transpose() / soft[sb].scale
            material_point[p].F += F_rate * dt[None]
            material_id = material_point[p].materialID
            stress = soft_matProps.soft_particle_pk1(material_id, material_point[p].F)
            material_point[p].stress = stress
            if ti.static(GlobalVariable.TRACKENERGY):
                material_point[p].strain_energy = material_point[p].vol0 * soft_matProps.Psi(
                    material_id, material_point[p].F
                )
            soft_matProps.update_soft_particle_state(p, material_id, material_point[p].F, stress, soft_stateVars)


@ti.kernel
def prepare_soft_body_surface_frame_(
    softNum: int,
    soft: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
):
    for sb in range(softNum):
        bodyID = soft[sb].bodyID
        old_center = soft[sb].mass_center
        inv_mass = 1.0 / ti.max(soft[sb].m, Threshold)
        center = soft[sb].previous_center * inv_mass
        velocity = soft[sb].v * inv_mass
        fixed = vec3f(
            1.0 - ti.cast(rigid[bodyID].is_fix[0], float),
            1.0 - ti.cast(rigid[bodyID].is_fix[1], float),
            1.0 - ti.cast(rigid[bodyID].is_fix[2], float),
        )
        center += fixed * (soft[sb].mass_center0 - center)
        velocity -= fixed * velocity
        soft[sb].previous_center = old_center
        soft[sb].mass_center = center
        soft[sb].v = velocity
        rigid[bodyID].mass_center = center
        rigid[bodyID].v = velocity
        rigid[bodyID].angmoment = ZEROVEC3f
        soft[sb].surface_inertia = ti.Matrix.zero(float, 3, 3)
        box[bodyID].shape_min = vec3f(1.0e30, 1.0e30, 1.0e30)
        box[bodyID].shape_max = vec3f(-1.0e30, -1.0e30, -1.0e30)
        box[bodyID].shape_radius = 0.0


@ti.kernel
def reduce_soft_body_surface_frame_(
    pointNum: int,
    soft: ti.template(),
    material_point: ti.template(),
    rigid: ti.template(),
):
    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            bodyID = material_point[p].bodyID
            sb = rigid[bodyID].softID
            relative_position = material_point[p].x - soft[sb].mass_center
            relative_velocity = material_point[p].v - soft[sb].v
            angular_momentum = relative_position.cross(
                material_point[p].m * relative_velocity
            )
            rr = relative_position.dot(relative_position)
            inertia = material_point[p].m * (
                rr * ti.Matrix.identity(float, 3)
                - relative_position.outer_product(relative_position)
            )
            ti.atomic_add(rigid[bodyID].angmoment, angular_momentum)
            for i, j in ti.static(ti.ndrange(3, 3)):
                ti.atomic_add(soft[sb].surface_inertia[i, j], inertia[i, j])


@ti.kernel
def finalize_soft_body_rotation_(
    softNum: int,
    dt: ti.template(),
    soft: ti.template(),
    rigid: ti.template(),
):
    for sb in range(softNum):
        bodyID = soft[sb].bodyID
        old_q = rigid[bodyID].q
        soft[sb].previous_rotation = SetToRotate(old_q)
        inertia = soft[sb].surface_inertia
        inertia_scale = ti.max(inertia.trace(), 1.0e-30)
        omega = (
            inertia
            + 1.0e-9 * inertia_scale * ti.Matrix.identity(float, 3)
        ).inverse() @ rigid[bodyID].angmoment
        omega *= ti.cast(rigid[bodyID].is_fix, float)
        rigid[bodyID].w = omega
        rigid[bodyID].q = Normalize(old_q + dt[None] * SetDQ(old_q, omega))


@ti.func
def accumulate_soft_body_bounds_(bodyID, point_min, point_max, point_radius, box):
    for d in ti.static(range(3)):
        ti.atomic_min(box[bodyID].shape_min[d], point_min[d])
        ti.atomic_max(box[bodyID].shape_max[d], point_max[d])
    ti.atomic_max(box[bodyID].shape_radius, point_radius)


@ti.kernel
def track_soft_surface_points_(
    surfaceNum: int,
    dt: ti.template(),
    soft: ti.template(),
    soft_grid: ti.template(),
    surface_shape_node: ti.template(),
    surface_shape: ti.template(),
    surface_shape_count: ti.template(),
    vertice: ti.template(),
    surface: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
):
    # The closed contact surface is the authoritative deformed geometry, so
    # its nodes also provide the broad-phase bounds without rescanning every
    # volumetric material point.
    for ns in range(surfaceNum):
        bodyID = surface[ns]
        if int(rigid[bodyID].is_soft) == 1:
            sb = rigid[bodyID].softID
            support = soft[sb].templateSurfaceStart + ns - soft[sb].startNode
            local_node = rigid[bodyID].global_node_to_local(ns)
            scale = box[bodyID].scale
            old_global = (
                soft[sb].previous_center
                + soft[sb].previous_rotation @ (scale * vertice[local_node].x)
            )
            node_velocity = ZEROVEC3f
            active_weight = 0.0
            for n in range(surface_shape_count[support]):
                node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                if soft_grid[node].m > Threshold:
                    active_weight += surface_shape[support, n]
                    node_velocity += surface_shape[support, n] * soft_grid[node].v
            if active_weight > Threshold:
                node_velocity /= active_weight
            inv_rotate = SetToRotate(rigid[bodyID].q).transpose()
            new_global = old_global + node_velocity * dt[None]
            new_local = inv_rotate @ (
                new_global - soft[sb].mass_center
            ) / scale
            vertice[local_node]._update_kinematics(new_local, node_velocity)
            new_local_scaled = scale * new_local
            accumulate_soft_body_bounds_(
                bodyID,
                new_local_scaled,
                new_local_scaled,
                new_local_scaled.norm(),
                box,
            )


@ti.kernel
def finalize_soft_body_bounds_(
    softNum: int,
    soft: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    bounding_sphere: ti.template(),
):
    for sb in range(softNum):
        bodyID = soft[sb].bodyID
        bounding_sphere[bodyID]._follow_deformed_shape(
            soft[sb].mass_center,
            SetToRotate(rigid[bodyID].q),
            box[bodyID].shape_min,
            box[bodyID].shape_max,
            box[bodyID].shape_radius,
            box[bodyID].grid_space,
        )


@ti.kernel
def rebuild_soft_grid_velocity_from_current_points_(
    softGridNum: int,
    pointNum: int,
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_shape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
):
    """Rebuild current TLMPM nodal velocities without advancing mechanics."""
    for node in range(softGridNum):
        soft_grid[node].m = 0.0
        soft_grid[node].v = ZEROVEC3f

    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index_(p, material_point, soft, rigid)
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node_(sb, support, n, soft, soft_shape_node)
                weighted_mass = soft_shape[support, n] * material_point[p].m
                ti.atomic_add(soft_grid[node].m, weighted_mass)
                ti.atomic_add(
                    soft_grid[node].v,
                    weighted_mass * material_point[p].v,
                )

    for node in range(softGridNum):
        if soft_grid[node].m > Threshold:
            soft_grid[node].v /= soft_grid[node].m


@ti.func
def soft_sdf_projection_boundary_type_(node, node_count):
    boundary_type = ti.min(2, node) - ti.min(node_count - 1 - node, 2)
    if boundary_type < 0:
        boundary_type += 3
    elif boundary_type > 0:
        boundary_type += 2
    return boundary_type


@ti.func
def soft_sdf_projection_weight_1d_(
    shape_function_type: ti.template(),
    point,
    node_position,
    inv_dx,
    boundary_type,
):
    weight = 0.0
    if ti.static(shape_function_type == 0):
        weight = ShapeLinear(point, node_position, inv_dx, 0.0)
    elif ti.static(shape_function_type == 1):
        weight = ShapeBsplineQ(point, node_position, inv_dx, boundary_type)
    else:
        weight = ShapeBsplineC(point, node_position, inv_dx, boundary_type)
    return weight


@ti.func
def scatter_soft_point_to_sdf_node_(
    shape_function_type: ti.template(),
    sb,
    bodyID,
    node_ijk,
    local_position,
    local_velocity,
    local_velocity_gradient,
    point_mass,
    xmin,
    dx,
    inv_dx,
    soft,
    levelset_velocity,
    projection_weight,
    box,
):
    shape_sum = 0.0
    inside = True
    for d in ti.static(range(3)):
        inside = inside and node_ijk[d] >= 0 and node_ijk[d] < box[bodyID].gnum[d]
    if inside:
        node_position = xmin + ti.cast(node_ijk, float) * dx
        wx = soft_sdf_projection_weight_1d_(
            shape_function_type,
            local_position[0],
            node_position[0],
            inv_dx,
            soft_sdf_projection_boundary_type_(node_ijk[0], box[bodyID].gnum[0]),
        )
        wy = soft_sdf_projection_weight_1d_(
            shape_function_type,
            local_position[1],
            node_position[1],
            inv_dx,
            soft_sdf_projection_boundary_type_(node_ijk[1], box[bodyID].gnum[1]),
        )
        wz = soft_sdf_projection_weight_1d_(
            shape_function_type,
            local_position[2],
            node_position[2],
            inv_dx,
            soft_sdf_projection_boundary_type_(node_ijk[2], box[bodyID].gnum[2]),
        )
        shape = wx * wy * wz
        if shape > Threshold:
            local_grid = (
                node_ijk[0]
                + node_ijk[1] * box[bodyID].gnum[0]
                + node_ijk[2] * box[bodyID].gnum[0] * box[bodyID].gnum[1]
            )
            sdf_grid = soft[sb].gridStart + local_grid
            projected_mass = shape * point_mass
            affine_velocity = local_velocity + local_velocity_gradient @ (node_position - local_position)
            ti.atomic_add(projection_weight[sdf_grid], projected_mass)
            ti.atomic_add(
                levelset_velocity[sdf_grid],
                projected_mass * affine_velocity,
            )
            shape_sum = shape
    return shape_sum


@ti.kernel
def reset_soft_levelset_velocity_projection_(
    softNum: int,
    maxGridNum: int,
    soft: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
):
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            levelset_velocity[sdf_grid] = ZEROVEC3f
            projection_weight[sdf_grid] = 0.0


@ti.kernel
def project_current_soft_points_to_levelset_(
    pointNum: int,
    shape_function_type: ti.template(),
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_dshape: ti.template(),
    soft_shape_count: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    projection_max_support_loss: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
):
    projection_max_support_loss[None] = 0.0
    for p in range(pointNum):
        if int(material_point[p].active) == 1:
            bodyID = material_point[p].bodyID
            sb, support = soft_point_support_index_(p, material_point, soft, rigid)
            scale = soft[sb].scale
            rotate = SetToRotate(rigid[bodyID].q)
            relative_position = material_point[p].x - rigid[bodyID].mass_center
            local_position = rotate.transpose() @ relative_position / scale
            frame_velocity = soft[sb].v + rigid[bodyID].w.cross(relative_position)
            local_velocity = rotate.transpose() @ (material_point[p].v - frame_velocity) / scale
            template_f_rate = ti.Matrix.zero(float, 3, 3)
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node_(sb, support, n, soft, soft_shape_node)
                if soft_grid[node].m > Threshold:
                    template_f_rate += soft_grid[node].v.outer_product(soft_dshape[support, n])
            f_rate = template_f_rate @ soft[sb].referenceRotation.transpose() / scale
            velocity_gradient = ti.Matrix.zero(float, 3, 3)
            if ti.abs(material_point[p].F.determinant()) > Threshold:
                velocity_gradient = f_rate @ material_point[p].F.inverse()
            frame_spin = mat3x3(
                [0.0, -rigid[bodyID].w[2], rigid[bodyID].w[1]],
                [rigid[bodyID].w[2], 0.0, -rigid[bodyID].w[0]],
                [-rigid[bodyID].w[1], rigid[bodyID].w[0], 0.0],
            )
            local_velocity_gradient = rotate.transpose() @ (velocity_gradient - frame_spin) @ rotate

            xmin = box[bodyID].xmin / scale
            dx = box[bodyID].grid_space / scale
            inv_dx = 1.0 / dx
            base = ti.Vector([0, 0, 0])
            offset_cells = 0.0
            if ti.static(shape_function_type == 1):
                offset_cells = 0.5
            elif ti.static(shape_function_type == 2):
                offset_cells = 1.0
            for d in ti.static(range(3)):
                base[d] = int(ti.floor((local_position[d] - xmin[d]) * inv_dx - offset_cells))

            point_support_sum = 0.0
            if ti.static(shape_function_type == 0):
                for i, j, k in ti.ndrange(2, 2, 2):
                    point_support_sum += scatter_soft_point_to_sdf_node_(
                        shape_function_type,
                        sb,
                        bodyID,
                        base + ti.Vector([i, j, k]),
                        local_position,
                        local_velocity,
                        local_velocity_gradient,
                        material_point[p].m,
                        xmin,
                        dx,
                        inv_dx,
                        soft,
                        levelset_velocity,
                        projection_weight,
                        box,
                    )
            elif ti.static(shape_function_type == 1):
                for i, j, k in ti.ndrange(3, 3, 3):
                    point_support_sum += scatter_soft_point_to_sdf_node_(
                        shape_function_type,
                        sb,
                        bodyID,
                        base + ti.Vector([i, j, k]),
                        local_position,
                        local_velocity,
                        local_velocity_gradient,
                        material_point[p].m,
                        xmin,
                        dx,
                        inv_dx,
                        soft,
                        levelset_velocity,
                        projection_weight,
                        box,
                    )
            else:
                for i, j, k in ti.ndrange(4, 4, 4):
                    point_support_sum += scatter_soft_point_to_sdf_node_(
                        shape_function_type,
                        sb,
                        bodyID,
                        base + ti.Vector([i, j, k]),
                        local_position,
                        local_velocity,
                        local_velocity_gradient,
                        material_point[p].m,
                        xmin,
                        dx,
                        inv_dx,
                        soft,
                        levelset_velocity,
                        projection_weight,
                        box,
                    )
            ti.atomic_max(
                projection_max_support_loss[None],
                ti.abs(1.0 - point_support_sum),
            )


@ti.kernel
def normalize_soft_levelset_velocity_projection_(
    softNum: int,
    maxGridNum: int,
    soft: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
):
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            if projection_weight[sdf_grid] > Threshold:
                levelset_velocity[sdf_grid] /= projection_weight[sdf_grid]
            else:
                levelset_velocity[sdf_grid] = ZEROVEC3f


@ti.kernel
def extend_soft_levelset_velocity_projection_(
    softNum: int,
    maxGridNum: int,
    color: int,
    soft: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
):
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            bodyID = soft[sb].bodyID
            gnum = box[bodyID].gnum
            i = local_grid % gnum[0]
            j = (local_grid % (gnum[0] * gnum[1])) // gnum[0]
            k = local_grid // (gnum[0] * gnum[1])
            sdf_grid = soft[sb].gridStart + local_grid
            if (
                (i + j + k) % 2 == color
                and i > 0
                and i < gnum[0] - 1
                and j > 0
                and j < gnum[1] - 1
                and k > 0
                and k < gnum[2] - 1
                and projection_weight[sdf_grid] <= Threshold
            ):
                accumulated_weight = 0.0
                accumulated_velocity = ZEROVEC3f
                mean_neighbor_weight = 0.0
                neighbor_count = 0
                for d in ti.static(range(3)):
                    for side in ti.static(range(2)):
                        neighbor_ijk = ti.Vector([i, j, k])
                        neighbor_ijk[d] += -1 if side == 0 else 1
                        inside = True
                        for axis in ti.static(range(3)):
                            inside = inside and neighbor_ijk[axis] >= 0 and neighbor_ijk[axis] < gnum[axis]
                        if inside:
                            neighbor_local = (
                                neighbor_ijk[0] + neighbor_ijk[1] * gnum[0] + neighbor_ijk[2] * gnum[0] * gnum[1]
                            )
                            neighbor_grid = soft[sb].gridStart + neighbor_local
                            neighbor_weight = projection_weight[neighbor_grid]
                            if neighbor_weight > Threshold:
                                accumulated_weight += neighbor_weight
                                accumulated_velocity += neighbor_weight * levelset_velocity[neighbor_grid]
                                mean_neighbor_weight += neighbor_weight
                                neighbor_count += 1
                if accumulated_weight > Threshold:
                    levelset_velocity[sdf_grid] = accumulated_velocity / accumulated_weight
                    projection_weight[sdf_grid] = mean_neighbor_weight / ti.max(neighbor_count, 1)


@ti.kernel
def audit_soft_levelset_velocity_projection_(
    softNum: int,
    maxGridNum: int,
    monitor_band_cells: float,
    soft: ti.template(),
    grid: ti.template(),
    projection_weight: ti.template(),
    projection_band_nodes: ti.template(),
    projection_uncovered_nodes: ti.template(),
    box: ti.template(),
):
    projection_band_nodes[None] = 0
    projection_uncovered_nodes[None] = 0
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            bodyID = soft[sb].bodyID
            sdf_grid = soft[sb].gridStart + local_grid
            dx = box[bodyID].grid_space / soft[sb].scale
            if ti.abs(grid[sdf_grid].distance_field) <= monitor_band_cells * dx:
                ti.atomic_add(projection_band_nodes[None], 1)
                if projection_weight[sdf_grid] <= Threshold:
                    ti.atomic_add(projection_uncovered_nodes[None], 1)


@ti.func
def soft_levelset_local_velocity_(
    sb,
    local_pos,
    soft,
    levelset_velocity,
    projection_weight,
    box,
):
    bodyID = soft[sb].bodyID
    scale = soft[sb].scale
    xmin = box[bodyID].xmin / scale
    xmax = box[bodyID].xmax / scale
    dx = box[bodyID].grid_space / scale
    query = local_pos
    for d in ti.static(range(3)):
        query[d] = ti.min(ti.max(query[d], xmin[d]), xmax[d])
    base = ti.Vector([0, 0, 0])
    for d in ti.static(range(3)):
        base[d] = ti.min(
            box[bodyID].gnum[d] - 2,
            ti.max(0, int((query[d] - xmin[d]) / dx)),
        )
    origin = xmin + vec3f(base[0], base[1], base[2]) * dx
    fraction = (query - origin) / dx
    local_velocity = ZEROVEC3f
    active_weight = 0.0
    for i in ti.static(range(2)):
        wx = (1.0 - fraction[0]) if i == 0 else fraction[0]
        for j in ti.static(range(2)):
            wy = (1.0 - fraction[1]) if j == 0 else fraction[1]
            for k in ti.static(range(2)):
                wz = (1.0 - fraction[2]) if k == 0 else fraction[2]
                weight = wx * wy * wz
                logical = (
                    base[0]
                    + i
                    + (base[1] + j) * box[bodyID].gnum[0]
                    + (base[2] + k) * box[bodyID].gnum[0] * box[bodyID].gnum[1]
                )
                sdf_grid = soft[sb].gridStart + logical
                nodal_weight = weight * projection_weight[sdf_grid]
                active_weight += nodal_weight
                local_velocity += nodal_weight * levelset_velocity[sdf_grid]
    if active_weight > Threshold:
        local_velocity /= active_weight
    return local_velocity


@ti.func
def soft_levelset_rk2_departure_(
    sb,
    local_pos,
    direction,
    dt,
    soft,
    levelset_velocity,
    projection_weight,
    box,
):
    velocity0 = soft_levelset_local_velocity_(sb, local_pos, soft, levelset_velocity, projection_weight, box)
    midpoint = local_pos + 0.5 * direction * dt * velocity0
    velocity_midpoint = soft_levelset_local_velocity_(sb, midpoint, soft, levelset_velocity, projection_weight, box)
    return local_pos + direction * dt * velocity_midpoint


@ti.kernel
def predict_soft_levelset_maccormack_(
    softNum: int,
    maxGridNum: int,
    dt: float,
    soft: ti.template(),
    grid: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
) -> float:
    maximum_departure_excess = 0.0
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            bodyID = soft[sb].bodyID
            scale = box[bodyID].scale
            i = local_grid % box[bodyID].gnum[0]
            j = (local_grid % (box[bodyID].gnum[0] * box[bodyID].gnum[1])) // box[bodyID].gnum[0]
            k = local_grid // (box[bodyID].gnum[0] * box[bodyID].gnum[1])
            local_scaled = box[bodyID].xmin + vec3f(i, j, k) * box[bodyID].grid_space
            local_unscaled = local_scaled / scale
            sdf_grid = soft[sb].gridStart + local_grid
            backtrace = soft_levelset_rk2_departure_(
                sb,
                local_unscaled,
                -1.0,
                dt,
                soft,
                levelset_velocity,
                projection_weight,
                box,
            )
            xmin = box[bodyID].xmin / scale
            xmax = box[bodyID].xmax / scale
            spacing = ti.max(box[bodyID].grid_space / scale, Threshold)
            for d in ti.static(range(3)):
                excess = (
                    ti.max(
                        ti.max(
                            xmin[d] - backtrace[d],
                            backtrace[d] - xmax[d],
                        ),
                        0.0,
                    )
                    / spacing
                )
                ti.atomic_max(maximum_departure_excess, excess)
            grid[sdf_grid].distance_field0 = grid[sdf_grid].distance_field
            grid[sdf_grid].distance_field_temp = sample_levelset_unscaled(backtrace, box[bodyID], grid)

    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            grid[sdf_grid].distance_field = grid[sdf_grid].distance_field_temp
    return maximum_departure_excess


@ti.kernel
def correct_soft_levelset_maccormack_(
    softNum: int,
    maxGridNum: int,
    dt: float,
    soft: ti.template(),
    grid: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
) -> float:
    maximum_departure_excess = 0.0
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            bodyID = soft[sb].bodyID
            scale = box[bodyID].scale
            i = local_grid % box[bodyID].gnum[0]
            j = (local_grid % (box[bodyID].gnum[0] * box[bodyID].gnum[1])) // box[bodyID].gnum[0]
            k = local_grid // (box[bodyID].gnum[0] * box[bodyID].gnum[1])
            local_scaled = box[bodyID].xmin + vec3f(i, j, k) * box[bodyID].grid_space
            local_unscaled = local_scaled / scale
            reverse_trace = soft_levelset_rk2_departure_(
                sb,
                local_unscaled,
                1.0,
                dt,
                soft,
                levelset_velocity,
                projection_weight,
                box,
            )
            xmin = box[bodyID].xmin / scale
            xmax = box[bodyID].xmax / scale
            spacing = ti.max(box[bodyID].grid_space / scale, Threshold)
            for d in ti.static(range(3)):
                excess = (
                    ti.max(
                        ti.max(
                            xmin[d] - reverse_trace[d],
                            reverse_trace[d] - xmax[d],
                        ),
                        0.0,
                    )
                    / spacing
                )
                ti.atomic_max(maximum_departure_excess, excess)
            sdf_grid = soft[sb].gridStart + local_grid
            grid[sdf_grid].distance_field_temp = sample_levelset_unscaled(reverse_trace, box[bodyID], grid)

    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            bodyID = soft[sb].bodyID
            scale = box[bodyID].scale
            i = local_grid % box[bodyID].gnum[0]
            j = (local_grid % (box[bodyID].gnum[0] * box[bodyID].gnum[1])) // box[bodyID].gnum[0]
            k = local_grid // (box[bodyID].gnum[0] * box[bodyID].gnum[1])
            local_scaled = box[bodyID].xmin + vec3f(i, j, k) * box[bodyID].grid_space
            local_unscaled = local_scaled / scale
            backtrace = soft_levelset_rk2_departure_(
                sb,
                local_unscaled,
                -1.0,
                dt,
                soft,
                levelset_velocity,
                projection_weight,
                box,
            )
            xmin = box[bodyID].xmin / scale
            xmax = box[bodyID].xmax / scale
            spacing = ti.max(box[bodyID].grid_space / scale, Threshold)
            for d in ti.static(range(3)):
                excess = (
                    ti.max(
                        ti.max(
                            xmin[d] - backtrace[d],
                            backtrace[d] - xmax[d],
                        ),
                        0.0,
                    )
                    / spacing
                )
                ti.atomic_max(maximum_departure_excess, excess)
            sdf_grid = soft[sb].gridStart + local_grid
            corrected = grid[sdf_grid].distance_field + 0.5 * (
                grid[sdf_grid].distance_field0 - grid[sdf_grid].distance_field_temp
            )
            bounds = sample_levelset_reference_bounds_unscaled(backtrace, box[bodyID], grid)
            grid[sdf_grid].distance_field = ti.min(ti.max(corrected, bounds[0]), bounds[1])
    return maximum_departure_excess


@ti.func
def soft_levelset_weno5_phi_offset_(
    sb,
    ijk,
    axis,
    offset,
    gnum,
    soft,
    grid,
):
    neighbor = vec3i(ijk[0], ijk[1], ijk[2])
    neighbor[axis] += offset
    local_grid = neighbor[0] + neighbor[1] * gnum[0] + neighbor[2] * gnum[0] * gnum[1]
    return grid[soft[sb].gridStart + local_grid].distance_field


@ti.func
def soft_levelset_weno5_left_flux_(fm2, fm1, f0, fp1, fp2):
    candidate0 = (2.0 * fm2 - 7.0 * fm1 + 11.0 * f0) / 6.0
    candidate1 = (-fm1 + 5.0 * f0 + 2.0 * fp1) / 6.0
    candidate2 = (2.0 * f0 + 5.0 * fp1 - fp2) / 6.0
    beta0 = (13.0 / 12.0) * (fm2 - 2.0 * fm1 + f0) ** 2 + 0.25 * (fm2 - 4.0 * fm1 + 3.0 * f0) ** 2
    beta1 = (13.0 / 12.0) * (fm1 - 2.0 * f0 + fp1) ** 2 + 0.25 * (fm1 - fp1) ** 2
    beta2 = (13.0 / 12.0) * (f0 - 2.0 * fp1 + fp2) ** 2 + 0.25 * (3.0 * f0 - 4.0 * fp1 + fp2) ** 2
    epsilon = WENO_EPS * WENO_EPS
    alpha0 = 0.1 / (epsilon + beta0) ** 2
    alpha1 = 0.6 / (epsilon + beta1) ** 2
    alpha2 = 0.3 / (epsilon + beta2) ** 2
    alpha_sum = alpha0 + alpha1 + alpha2
    return (alpha0 * candidate0 + alpha1 * candidate1 + alpha2 * candidate2) / alpha_sum


@ti.func
def soft_levelset_weno5_derivative_(sb, ijk, axis, gnum, h, velocity, soft, grid):
    phi_0 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, 0, gnum, soft, grid)
    derivative = 0.0
    if ijk[axis] >= 3 and ijk[axis] < gnum[axis] - 3:
        phi_m3 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, -3, gnum, soft, grid)
        phi_m2 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, -2, gnum, soft, grid)
        phi_m1 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, -1, gnum, soft, grid)
        phi_p1 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, 1, gnum, soft, grid)
        phi_p2 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, 2, gnum, soft, grid)
        phi_p3 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, 3, gnum, soft, grid)
        flux_plus = 0.0
        flux_minus = 0.0
        if velocity >= 0.0:
            flux_plus = soft_levelset_weno5_left_flux_(phi_m2, phi_m1, phi_0, phi_p1, phi_p2)
            flux_minus = soft_levelset_weno5_left_flux_(phi_m3, phi_m2, phi_m1, phi_0, phi_p1)
        else:
            flux_plus = soft_levelset_weno5_left_flux_(phi_p3, phi_p2, phi_p1, phi_0, phi_m1)
            flux_minus = soft_levelset_weno5_left_flux_(phi_p2, phi_p1, phi_0, phi_m1, phi_m2)
        derivative = (flux_plus - flux_minus) / h
    elif velocity >= 0.0:
        if ijk[axis] > 0:
            phi_m1 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, -1, gnum, soft, grid)
            derivative = (phi_0 - phi_m1) / h
        elif gnum[axis] > 1:
            phi_p1 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, 1, gnum, soft, grid)
            derivative = (phi_p1 - phi_0) / h
    else:
        if ijk[axis] < gnum[axis] - 1:
            phi_p1 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, 1, gnum, soft, grid)
            derivative = (phi_p1 - phi_0) / h
        elif gnum[axis] > 1:
            phi_m1 = soft_levelset_weno5_phi_offset_(sb, ijk, axis, -1, gnum, soft, grid)
            derivative = (phi_0 - phi_m1) / h
    return derivative


@ti.kernel
def store_soft_levelset_weno5_reference_(
    softNum: int,
    maxGridNum: int,
    soft: ti.template(),
    grid: ti.template(),
):
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            grid[sdf_grid].distance_field0 = grid[sdf_grid].distance_field
            grid[sdf_grid].distance_field_temp = grid[sdf_grid].distance_field


@ti.kernel
def maximum_soft_levelset_weno5_cfl_(
    softNum: int,
    maxGridNum: int,
    dt: float,
    soft: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
) -> float:
    maximum_cfl = 0.0
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            if projection_weight[sdf_grid] > Threshold:
                bodyID = soft[sb].bodyID
                h = box[bodyID].grid_space / ti.max(box[bodyID].scale, Threshold)
                velocity = levelset_velocity[sdf_grid]
                local_cfl = dt * (ti.abs(velocity[0]) + ti.abs(velocity[1]) + ti.abs(velocity[2])) / h
                ti.atomic_max(maximum_cfl, local_cfl)
    return maximum_cfl


@ti.kernel
def advance_soft_levelset_weno5_stage_(
    softNum: int,
    maxGridNum: int,
    dt: float,
    stage: int,
    soft: ti.template(),
    grid: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
):
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            bodyID = soft[sb].bodyID
            gnum = box[bodyID].gnum
            i = local_grid % gnum[0]
            j = (local_grid % (gnum[0] * gnum[1])) // gnum[0]
            k = local_grid // (gnum[0] * gnum[1])
            ijk = vec3i(i, j, k)
            sdf_grid = soft[sb].gridStart + local_grid
            phi = grid[sdf_grid].distance_field
            euler_value = phi
            if projection_weight[sdf_grid] > Threshold:
                h = box[bodyID].grid_space / ti.max(box[bodyID].scale, Threshold)
                velocity = levelset_velocity[sdf_grid]
                dphi_x = soft_levelset_weno5_derivative_(sb, ijk, 0, gnum, h, velocity[0], soft, grid)
                dphi_y = soft_levelset_weno5_derivative_(sb, ijk, 1, gnum, h, velocity[1], soft, grid)
                dphi_z = soft_levelset_weno5_derivative_(sb, ijk, 2, gnum, h, velocity[2], soft, grid)
                euler_value = phi - dt * (velocity[0] * dphi_x + velocity[1] * dphi_y + velocity[2] * dphi_z)
            phi0 = grid[sdf_grid].distance_field0
            phi_new = euler_value
            if stage == 2:
                phi_new = 0.75 * phi0 + 0.25 * euler_value
            elif stage == 3:
                phi_new = (1.0 / 3.0) * phi0 + (2.0 / 3.0) * euler_value
            grid[sdf_grid].distance_field_temp = phi_new

    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            grid[sdf_grid].distance_field = grid[sdf_grid].distance_field_temp


def advect_projected_soft_levelset_weno5_(
    softNum,
    maxGridNum,
    dt,
    soft,
    grid,
    levelset_velocity,
    projection_weight,
    box,
    maximum_cfl=0.20,
):
    if maximum_cfl <= 0.0:
        raise ValueError("WENO5 maximum Courant number must be positive")
    observed_cfl = float(
        maximum_soft_levelset_weno5_cfl_(
            softNum,
            maxGridNum,
            dt,
            soft,
            levelset_velocity,
            projection_weight,
            box,
        )
    )
    substeps = max(1, int(math.ceil(observed_cfl / maximum_cfl)))
    substep_dt = dt / substeps
    for _ in range(substeps):
        store_soft_levelset_weno5_reference_(softNum, maxGridNum, soft, grid)
        for stage in (1, 2, 3):
            advance_soft_levelset_weno5_stage_(
                softNum,
                maxGridNum,
                substep_dt,
                stage,
                soft,
                grid,
                levelset_velocity,
                projection_weight,
                box,
            )


def project_soft_points_to_levelset_velocity_(
    softNum,
    maxGridNum,
    pointNum,
    shape_function_type,
    projection_monitor_band,
    projection_extension_iterations,
    soft,
    grid,
    material_point,
    soft_grid,
    soft_shape_node,
    soft_dshape,
    soft_shape_count,
    levelset_velocity,
    projection_weight,
    projection_band_nodes,
    projection_uncovered_nodes,
    projection_max_support_loss,
    rigid,
    box,
):
    reset_soft_levelset_velocity_projection_(
        softNum,
        maxGridNum,
        soft,
        levelset_velocity,
        projection_weight,
    )
    project_current_soft_points_to_levelset_(
        pointNum,
        shape_function_type,
        soft,
        material_point,
        soft_grid,
        soft_shape_node,
        soft_dshape,
        soft_shape_count,
        levelset_velocity,
        projection_weight,
        projection_max_support_loss,
        rigid,
        box,
    )
    normalize_soft_levelset_velocity_projection_(
        softNum,
        maxGridNum,
        soft,
        levelset_velocity,
        projection_weight,
    )
    for _ in range(max(int(projection_extension_iterations), 0)):
        extend_soft_levelset_velocity_projection_(
            softNum,
            maxGridNum,
            0,
            soft,
            levelset_velocity,
            projection_weight,
            box,
        )
        extend_soft_levelset_velocity_projection_(
            softNum,
            maxGridNum,
            1,
            soft,
            levelset_velocity,
            projection_weight,
            box,
        )
    audit_soft_levelset_velocity_projection_(
        softNum,
        maxGridNum,
        projection_monitor_band,
        soft,
        grid,
        projection_weight,
        projection_band_nodes,
        projection_uncovered_nodes,
        box,
    )


def advect_soft_levelset_(
    softNum,
    maxGridNum,
    pointNum,
    shape_function_type,
    projection_monitor_band,
    projection_extension_iterations,
    dt,
    soft,
    grid,
    material_point,
    soft_grid,
    soft_shape_node,
    soft_dshape,
    soft_shape_count,
    levelset_velocity,
    projection_weight,
    projection_band_nodes,
    projection_uncovered_nodes,
    projection_max_support_loss,
    rigid,
    box,
):
    project_soft_points_to_levelset_velocity_(
        softNum,
        maxGridNum,
        pointNum,
        shape_function_type,
        projection_monitor_band,
        projection_extension_iterations,
        soft,
        grid,
        material_point,
        soft_grid,
        soft_shape_node,
        soft_dshape,
        soft_shape_count,
        levelset_velocity,
        projection_weight,
        projection_band_nodes,
        projection_uncovered_nodes,
        projection_max_support_loss,
        rigid,
        box,
    )
    prediction_excess = predict_soft_levelset_maccormack_(
        softNum,
        maxGridNum,
        dt,
        soft,
        grid,
        levelset_velocity,
        projection_weight,
        box,
    )
    correction_excess = correct_soft_levelset_maccormack_(
        softNum,
        maxGridNum,
        dt,
        soft,
        grid,
        levelset_velocity,
        projection_weight,
        box,
    )
    return max(float(prediction_excess), float(correction_excess))


def advect_soft_levelset_weno5_(
    softNum,
    maxGridNum,
    pointNum,
    shape_function_type,
    projection_monitor_band,
    projection_extension_iterations,
    dt,
    soft,
    grid,
    material_point,
    soft_grid,
    soft_shape_node,
    soft_dshape,
    soft_shape_count,
    levelset_velocity,
    projection_weight,
    projection_band_nodes,
    projection_uncovered_nodes,
    projection_max_support_loss,
    rigid,
    box,
    maximum_cfl=0.20,
):
    project_soft_points_to_levelset_velocity_(
        softNum,
        maxGridNum,
        pointNum,
        shape_function_type,
        projection_monitor_band,
        projection_extension_iterations,
        soft,
        grid,
        material_point,
        soft_grid,
        soft_shape_node,
        soft_dshape,
        soft_shape_count,
        levelset_velocity,
        projection_weight,
        projection_band_nodes,
        projection_uncovered_nodes,
        projection_max_support_loss,
        rigid,
        box,
    )
    advect_projected_soft_levelset_weno5_(
        softNum,
        maxGridNum,
        dt,
        soft,
        grid,
        levelset_velocity,
        projection_weight,
        box,
        maximum_cfl,
    )
    return 0.0
