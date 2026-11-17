import math

import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.utils.constants import THRESHOLD, WENO_EPS, ZEROVEC3F
from src.utils.Quaternion import set_dq, set_to_rotate
from src.utils.ShapeFunctions import shape_bspline_c, shape_bspline_q, shape_linear
from src.utils.TypeDefinition import vec3f, vec3i, mat3x3
from src.utils.VectorFunction import normalize
from src.mpm.soft_particle.LevelSet import (
    monitor_soft_levelset_signed_distance_error,
    reinitialize_soft_levelset_step,
    sample_levelset_reference_bounds_unscaled,
    sample_levelset_unscaled,
    split_soft_grid_slot,
    store_soft_levelset_reinit_reference,
)


@ti.func
def soft_point_support_index(point, material_point, soft, rigid):
    body_id = material_point[point].bodyID
    sb = rigid[body_id].softID
    return sb, soft[sb].templatePointStart + point - soft[sb].startIndex


@ti.func
def soft_point_support_node(sb, support, basis, soft, shape_node):
    return soft[sb].mpmGridStart + shape_node[support, basis]


@ti.func
def soft_surface_support_index(surface_node, body_id, soft, rigid):
    sb = rigid[body_id].softID
    return sb, (soft[sb].templateSurfaceStart + surface_node - soft[sb].startNode)


@ti.func
def soft_surface_levelset_error_at_node(body_id, local_node, box, grid, vertice):
    local_position = box[body_id].scale * vertice[local_node].x
    return ti.abs(box[body_id].distance(local_position, grid))


@ti.kernel
def soft_surface_force_reset(
    soft_num: int,
    surface_num: int,
    soft: ti.template(),
    surface: ti.template(),
    rigid: ti.template(),
    vertice: ti.template(),
):
    for sb in range(soft_num):
        soft[sb].contact_force = ZEROVEC3F
    for ns in range(surface_num):
        body_id = surface[ns]
        local_node = rigid[body_id].global_node_to_local(ns)
        vertice[local_node]._reset_contact()


@ti.kernel
def soft_material_point_force_reset(point_num: int, material_point: ti.template()):
    for p in range(point_num):
        if int(material_point[p].active) == 1:
            material_point[p]._reset_contact()


@ti.kernel
def max_soft_surface_levelset_error(
    soft_num: int,
    soft: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    grid: ti.template(),
    vertice: ti.template(),
) -> float:
    max_error = 0.0
    for sb in range(soft_num):
        body_id = soft[sb].bodyID
        scale = box[body_id].scale
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            local_position = scale * vertice[local_node].x
            ti.atomic_max(max_error, ti.abs(box[body_id].distance(local_position, grid)))
    return max_error


@ti.kernel
def sum_soft_surface_levelset_error(
    soft_num: int,
    soft: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    grid: ti.template(),
    vertice: ti.template(),
) -> float:
    error_sum = 0.0
    for sb in range(soft_num):
        body_id = soft[sb].bodyID
        scale = box[body_id].scale
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            local_position = scale * vertice[local_node].x
            ti.atomic_add(error_sum, ti.abs(box[body_id].distance(local_position, grid)))
    return error_sum


@ti.kernel
def sum_soft_surface_levelset_area_weight(
    soft_num: int, soft: ti.template(), box: ti.template(), vertice: ti.template()
) -> float:
    area_sum = 0.0
    for sb in range(soft_num):
        body_id = soft[sb].bodyID
        physical_area = box[body_id].reference_surface_area * box[body_id].scale ** 2
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            ti.atomic_add(
                area_sum,
                physical_area * ti.max(vertice[local_node].parameter, 0.0),
            )
    return area_sum


@ti.kernel
def sum_soft_surface_levelset_area_weighted_error(
    soft_num: int, soft: ti.template(), box: ti.template(), grid: ti.template(), vertice: ti.template()
) -> float:
    error_sum = 0.0
    for sb in range(soft_num):
        body_id = soft[sb].bodyID
        scale = box[body_id].scale
        physical_area = box[body_id].reference_surface_area * scale**2
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            local_position = scale * vertice[local_node].x
            area = physical_area * ti.max(vertice[local_node].parameter, 0.0)
            error = ti.abs(box[body_id].distance(local_position, grid))
            ti.atomic_add(error_sum, area * error)
    return error_sum


@ti.kernel
def sum_soft_surface_levelset_area_weighted_squared_error(
    soft_num: int, soft: ti.template(), box: ti.template(), grid: ti.template(), vertice: ti.template()
) -> float:
    error_sum = 0.0
    for sb in range(soft_num):
        body_id = soft[sb].bodyID
        scale = box[body_id].scale
        physical_area = box[body_id].reference_surface_area * scale**2
        for ns in range(soft[sb].startNode, soft[sb].endNode):
            local_node = soft[sb].global_node_to_local(ns)
            local_position = scale * vertice[local_node].x
            area = physical_area * ti.max(vertice[local_node].parameter, 0.0)
            error = box[body_id].distance(local_position, grid)
            ti.atomic_add(error_sum, area * error * error)
    return error_sum


@ti.kernel
def min_soft_surface_active_shape_sum(
    surface_num: int,
    surface: ti.template(),
    rigid: ti.template(),
    soft: ti.template(),
    soft_grid: ti.template(),
    surface_shape_node: ti.template(),
    surface_shape: ti.template(),
    surface_shape_count: ti.template(),
) -> float:
    minimum = 1.0e30
    for ns in range(surface_num):
        body_id = surface[ns]
        if int(rigid[body_id].is_soft) == 1:
            sb, support = soft_surface_support_index(ns, body_id, soft, rigid)
            active_weight = 0.0
            for n in range(surface_shape_count[support]):
                node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                if soft_grid[node].m > THRESHOLD:
                    active_weight += surface_shape[support, n]
            ti.atomic_min(minimum, active_weight)
    return minimum


@ti.kernel
def count_soft_surface_without_active_support(
    surface_num: int,
    surface: ti.template(),
    rigid: ti.template(),
    soft: ti.template(),
    soft_grid: ti.template(),
    surface_shape_node: ti.template(),
    surface_shape: ti.template(),
    surface_shape_count: ti.template(),
) -> int:
    unsupported = 0
    for ns in range(surface_num):
        body_id = surface[ns]
        if int(rigid[body_id].is_soft) == 1:
            sb, support = soft_surface_support_index(ns, body_id, soft, rigid)
            active_weight = 0.0
            for n in range(surface_shape_count[support]):
                node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                if soft_grid[node].m > THRESHOLD:
                    active_weight += surface_shape[support, n]
            if active_weight <= THRESHOLD:
                ti.atomic_add(unsupported, 1)
    return unsupported


@ti.kernel
def min_soft_levelset_domain_margin(
    soft_num: int,
    soft: ti.template(),
    box: ti.template(),
) -> float:
    """Minimum deformed-body clearance from the fixed SDF box, in cells."""
    minimum_margin = 1.0e30
    for sb in range(soft_num):
        body_id = soft[sb].bodyID
        spacing = ti.max(box[body_id].grid_space, THRESHOLD)
        for d in ti.static(range(3)):
            lower_margin = (box[body_id].shape_min[d] - box[body_id].xmin[d]) / spacing
            upper_margin = (box[body_id].xmax[d] - box[body_id].shape_max[d]) / spacing
            ti.atomic_min(minimum_margin, lower_margin)
            ti.atomic_min(minimum_margin, upper_margin)
    return minimum_margin


@ti.func
def update_soft_grid_node(grid_slot, soft_num, dt, soft, soft_grid, soft_grid_owner, soft_grid_local, box, material):
    sb, local_grid = split_soft_grid_slot(grid_slot, soft_grid_owner, soft_grid_local)
    if sb < soft_num and local_grid < soft[sb].mpmGridNum:
        mat_id = int(soft[sb].materialID)
        damp = material[mat_id]._get_force_damping_coefficient()
        if soft_grid[grid_slot].m > THRESHOLD:
            velocity = soft_grid[grid_slot].v
            force = soft_grid[grid_slot].f
            if damp > 0.0:
                density = material[mat_id]._get_density()
                young = material[mat_id]._get_young_modulus()
                body_id = soft[sb].bodyID
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
def precompute_soft_grid_reference_mass(
    soft_num: int,
    soft_grid_num: int,
    point_num: int,
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_shape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
):
    """Assemble invariant mass and the initial cross-step grid velocity."""
    for node in range(soft_grid_num):
        soft_grid[node].m = 0.0
        soft_grid[node].v = ZEROVEC3F

    for p in range(point_num):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index(p, material_point, soft, rigid)
            if sb < soft_num:
                for n in range(soft_shape_count[support]):
                    node = soft_point_support_node(sb, support, n, soft, soft_shape_node)
                    ti.atomic_add(
                        soft_grid[node].m,
                        soft_shape[support, n] * material_point[p].m,
                    )
                    ti.atomic_add(
                        soft_grid[node].v,
                        soft_shape[support, n] * material_point[p].m * material_point[p].v,
                    )

    for node in range(soft_grid_num):
        if soft_grid[node].m > THRESHOLD:
            soft_grid[node].v /= soft_grid[node].m


@ti.kernel
def initialize_soft_particle_constitutive_state(
    point_num: int,
    material_point: ti.template(),
    soft_mat_props: ti.template(),
    soft_state_vars: ti.template(),
):
    """Initialize the stress cache consumed by the first explicit step."""
    for p in range(point_num):
        if int(material_point[p].active) == 1:
            material_id = material_point[p].materialID
            stress = soft_mat_props.soft_particle_pk1(material_id, material_point[p].F)
            material_point[p].stress = stress
            if ti.static(GlobalVariable.TRACKENERGY):
                material_point[p].strain_energy = material_point[p].vol0 * soft_mat_props.psi(
                    material_id, material_point[p].F
                )
            soft_mat_props.update_soft_particle_state(p, material_id, material_point[p].F, stress, soft_state_vars)


@ti.kernel
def reset_soft_grid_step(
    soft_grid_num: int,
    soft_grid: ti.template(),
):
    for ng in range(soft_grid_num):
        soft_grid[ng]._tlmpm_step_reset()


@ti.kernel
def soft_body_force_p2g(
    point_num: int,
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
    for p in range(point_num):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index(p, material_point, soft, rigid)
            stress = material_point[p].stress
            template_stress = stress @ soft[sb].referenceRotation / soft[sb].scale
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node(sb, support, n, soft, soft_shape_node)
                weight = soft_shape[support, n]
                template_grad = soft_dshape[support, n]
                body_force = weight * material_point[p].m * gravity
                internal_force = -material_point[p].vol0 * (template_stress @ template_grad)
                contact_force = weight * material_point[p].contact_force
                body_id = material_point[p].bodyID
                external_force = (
                    weight * soft[rigid[body_id].softID].external_load_factor * material_point[p].external_force
                )
                ti.atomic_add(
                    soft_grid[node].f,
                    body_force + internal_force + contact_force + external_force,
                )


@ti.kernel
def soft_surface_force_p2g(
    surface_num: int,
    soft: ti.template(),
    soft_grid: ti.template(),
    surface_shape_node: ti.template(),
    surface_shape: ti.template(),
    surface_shape_count: ti.template(),
    vertice: ti.template(),
    surface: ti.template(),
    rigid: ti.template(),
):
    for ns in range(surface_num):
        body_id = surface[ns]
        if int(rigid[body_id].is_soft) == 1:
            sb, support = soft_surface_support_index(ns, body_id, soft, rigid)
            if surface_shape_count[support] > 0:
                local_node = rigid[body_id].global_node_to_local(ns)
                cforce = (
                    vertice[local_node].contact_force
                    + soft[rigid[body_id].softID].external_load_factor * vertice[local_node].external_force
                )
                if cforce.dot(cforce) > THRESHOLD * THRESHOLD:
                    active_weight = 0.0
                    for n in range(surface_shape_count[support]):
                        node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                        if soft_grid[node].m > THRESHOLD:
                            active_weight += surface_shape[support, n]
                    # Boundary traces can include empty exterior nodes. Renormalize
                    # over the mass-carrying support so no contact resultant is lost.
                    for n in range(surface_shape_count[support]):
                        node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                        if soft_grid[node].m > THRESHOLD and active_weight > THRESHOLD:
                            ti.atomic_add(
                                soft_grid[node].f,
                                surface_shape[support, n] / active_weight * cforce,
                            )


@ti.kernel
def update_soft_grid_kinematic(
    soft_num: int,
    soft_grid_num: int,
    fixed_grid_num: int,
    dt: ti.template(),
    soft: ti.template(),
    soft_grid: ti.template(),
    soft_grid_owner: ti.template(),
    soft_grid_local: ti.template(),
    fixed_grid: ti.template(),
    box: ti.template(),
    material: ti.template(),
):
    for grid_slot in range(soft_grid_num):
        update_soft_grid_node(grid_slot, soft_num, dt, soft, soft_grid, soft_grid_owner, soft_grid_local, box, material)

    # As in the standard MPM VelocityConstraint, only the precomputed sparse
    # node list is traversed; no boundary flag is stored on grid nodes.
    for boundary in range(fixed_grid_num):
        node = fixed_grid[boundary]
        if node < soft_grid_num and soft_grid[node].m > THRESHOLD:
            soft_grid[node].v = ZEROVEC3F
            soft_grid[node].f = ZEROVEC3F


@ti.kernel
def soft_body_g2p(
    point_num: int,
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
    for p in range(point_num):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index(p, material_point, soft, rigid)
            acceleration = ZEROVEC3F
            velocity = ZEROVEC3F
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node(sb, support, n, soft, soft_shape_node)
                if soft_grid[node].m > THRESHOLD:
                    weight = soft_shape[support, n]
                    acceleration += weight * soft_grid[node].f / soft_grid[node].m
                    velocity += weight * soft_grid[node].v
            velocity_flip = material_point[p].v + acceleration * dt[None]
            material_point[p].v = pic_fraction * velocity + (1.0 - pic_fraction) * velocity_flip
            material_point[p].x += velocity * dt[None]


@ti.kernel
def reset_soft_body_kinematic_reduction(
    soft_num: int,
    soft: ti.template(),
):
    for sb in range(soft_num):
        soft[sb].previous_center = ZEROVEC3F
        soft[sb].v = ZEROVEC3F


@ti.kernel
def reduce_soft_body_kinematics(
    point_num: int,
    soft: ti.template(),
    material_point: ti.template(),
    rigid: ti.template(),
):
    for p in range(point_num):
        if int(material_point[p].active) == 1:
            body_id = material_point[p].bodyID
            sb = rigid[body_id].softID
            ti.atomic_add(
                soft[sb].previous_center,
                material_point[p].m * material_point[p].x,
            )
            ti.atomic_add(
                soft[sb].v,
                material_point[p].m * material_point[p].v,
            )


@ti.kernel
def apply_soft_body_translation_constraint(
    point_num: int,
    soft: ti.template(),
    material_point: ti.template(),
    rigid: ti.template(),
):
    # Soft bodies do not pass through the rigid-body integrator. Remove only
    # their constrained rigid-translation mode, in parallel over points.
    for p in range(point_num):
        if int(material_point[p].active) == 1:
            body_id = material_point[p].bodyID
            sb = rigid[body_id].softID
            inv_mass = 1.0 / ti.max(soft[sb].m, THRESHOLD)
            center = soft[sb].previous_center * inv_mass
            velocity = soft[sb].v * inv_mass
            fixed = vec3f(
                1.0 - ti.cast(rigid[body_id].is_fix[0], float),
                1.0 - ti.cast(rigid[body_id].is_fix[1], float),
                1.0 - ti.cast(rigid[body_id].is_fix[2], float),
            )
            material_point[p].x += fixed * (soft[sb].mass_center0 - center)
            material_point[p].v -= fixed * velocity


@ti.kernel
def remap_soft_grid_velocity(
    soft_grid_num: int,
    fixed_grid_num: int,
    point_num: int,
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    fixed_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_shape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
):
    for ng in range(soft_grid_num):
        if soft_grid[ng].m > THRESHOLD:
            soft_grid[ng].v = ZEROVEC3F

    for p in range(point_num):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index(p, material_point, soft, rigid)
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node(sb, support, n, soft, soft_shape_node)
                weight = soft_shape[support, n]
                ti.atomic_add(soft_grid[node].v, weight * material_point[p].m * material_point[p].v)

    for ng in range(soft_grid_num):
        if soft_grid[ng].m > THRESHOLD:
            soft_grid[ng].v /= soft_grid[ng].m

    # This corrected velocity field is used below for F and surface updates,
    # then retained as the initial grid velocity of the next time step.

    for boundary in range(fixed_grid_num):
        node = fixed_grid[boundary]
        if node < soft_grid_num and soft_grid[node].m > THRESHOLD:
            soft_grid[node].v = ZEROVEC3F


@ti.kernel
def update_soft_particle_stress(
    point_num: int,
    dt: ti.template(),
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_dshape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
    soft_mat_props: ti.template(),
    soft_state_vars: ti.template(),
):
    for p in range(point_num):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index(p, material_point, soft, rigid)
            template_F_rate = mat3x3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node(sb, support, n, soft, soft_shape_node)
                if soft_grid[node].m > THRESHOLD:
                    template_F_rate += soft_grid[node].v.outer_product(soft_dshape[support, n])
            F_rate = template_F_rate @ soft[sb].referenceRotation.transpose() / soft[sb].scale
            material_point[p].F += F_rate * dt[None]
            material_id = material_point[p].materialID
            stress = soft_mat_props.soft_particle_pk1(material_id, material_point[p].F)
            material_point[p].stress = stress
            if ti.static(GlobalVariable.TRACKENERGY):
                material_point[p].strain_energy = material_point[p].vol0 * soft_mat_props.psi(
                    material_id, material_point[p].F
                )
            soft_mat_props.update_soft_particle_state(p, material_id, material_point[p].F, stress, soft_state_vars)


@ti.kernel
def prepare_soft_body_surface_frame(
    soft_num: int,
    soft: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
):
    for sb in range(soft_num):
        body_id = soft[sb].bodyID
        old_center = soft[sb].mass_center
        inv_mass = 1.0 / ti.max(soft[sb].m, THRESHOLD)
        center = soft[sb].previous_center * inv_mass
        velocity = soft[sb].v * inv_mass
        fixed = vec3f(
            1.0 - ti.cast(rigid[body_id].is_fix[0], float),
            1.0 - ti.cast(rigid[body_id].is_fix[1], float),
            1.0 - ti.cast(rigid[body_id].is_fix[2], float),
        )
        center += fixed * (soft[sb].mass_center0 - center)
        velocity -= fixed * velocity
        soft[sb].previous_center = old_center
        soft[sb].mass_center = center
        soft[sb].v = velocity
        rigid[body_id].mass_center = center
        rigid[body_id].v = velocity
        rigid[body_id].angmoment = ZEROVEC3F
        soft[sb].surface_inertia = ti.Matrix.zero(float, 3, 3)
        box[body_id].shape_min = vec3f(1.0e30, 1.0e30, 1.0e30)
        box[body_id].shape_max = vec3f(-1.0e30, -1.0e30, -1.0e30)
        box[body_id].shape_radius = 0.0


@ti.kernel
def reduce_soft_body_surface_frame(
    point_num: int,
    soft: ti.template(),
    material_point: ti.template(),
    rigid: ti.template(),
):
    for p in range(point_num):
        if int(material_point[p].active) == 1:
            body_id = material_point[p].bodyID
            sb = rigid[body_id].softID
            relative_position = material_point[p].x - soft[sb].mass_center
            relative_velocity = material_point[p].v - soft[sb].v
            angular_momentum = relative_position.cross(material_point[p].m * relative_velocity)
            rr = relative_position.dot(relative_position)
            inertia = material_point[p].m * (
                rr * ti.Matrix.identity(float, 3) - relative_position.outer_product(relative_position)
            )
            ti.atomic_add(rigid[body_id].angmoment, angular_momentum)
            for i, j in ti.static(ti.ndrange(3, 3)):
                ti.atomic_add(soft[sb].surface_inertia[i, j], inertia[i, j])


@ti.kernel
def finalize_soft_body_rotation(
    soft_num: int,
    dt: ti.template(),
    soft: ti.template(),
    rigid: ti.template(),
):
    for sb in range(soft_num):
        body_id = soft[sb].bodyID
        old_q = rigid[body_id].q
        soft[sb].previous_rotation = set_to_rotate(old_q)
        inertia = soft[sb].surface_inertia
        inertia_scale = ti.max(inertia.trace(), 1.0e-30)
        omega = (inertia + 1.0e-9 * inertia_scale * ti.Matrix.identity(float, 3)).inverse() @ rigid[body_id].angmoment
        omega *= ti.cast(rigid[body_id].is_fix, float)
        rigid[body_id].w = omega
        rigid[body_id].q = normalize(old_q + dt[None] * set_dq(old_q, omega))


@ti.func
def accumulate_soft_body_bounds(body_id, point_min, point_max, point_radius, box):
    for d in ti.static(range(3)):
        ti.atomic_min(box[body_id].shape_min[d], point_min[d])
        ti.atomic_max(box[body_id].shape_max[d], point_max[d])
    ti.atomic_max(box[body_id].shape_radius, point_radius)


@ti.kernel
def track_soft_surface_points(
    surface_num: int,
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
    for ns in range(surface_num):
        body_id = surface[ns]
        if int(rigid[body_id].is_soft) == 1:
            sb = rigid[body_id].softID
            support = soft[sb].templateSurfaceStart + ns - soft[sb].startNode
            local_node = rigid[body_id].global_node_to_local(ns)
            scale = box[body_id].scale
            old_global = soft[sb].previous_center + soft[sb].previous_rotation @ (scale * vertice[local_node].x)
            node_velocity = ZEROVEC3F
            active_weight = 0.0
            for n in range(surface_shape_count[support]):
                node = soft[sb].mpmGridStart + surface_shape_node[support, n]
                if soft_grid[node].m > THRESHOLD:
                    active_weight += surface_shape[support, n]
                    node_velocity += surface_shape[support, n] * soft_grid[node].v
            if active_weight > THRESHOLD:
                node_velocity /= active_weight
            inv_rotate = set_to_rotate(rigid[body_id].q).transpose()
            new_global = old_global + node_velocity * dt[None]
            new_local = inv_rotate @ (new_global - soft[sb].mass_center) / scale
            vertice[local_node]._update_kinematics(new_local, node_velocity)
            new_local_scaled = scale * new_local
            accumulate_soft_body_bounds(
                body_id,
                new_local_scaled,
                new_local_scaled,
                new_local_scaled.norm(),
                box,
            )


@ti.kernel
def finalize_soft_body_bounds(
    soft_num: int,
    soft: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    bounding_sphere: ti.template(),
):
    for sb in range(soft_num):
        body_id = soft[sb].bodyID
        bounding_sphere[body_id]._follow_deformed_shape(
            soft[sb].mass_center,
            set_to_rotate(rigid[body_id].q),
            box[body_id].shape_min,
            box[body_id].shape_max,
            box[body_id].shape_radius,
            box[body_id].grid_space,
        )


@ti.kernel
def rebuild_soft_grid_velocity_from_current_points(
    soft_grid_num: int,
    point_num: int,
    soft: ti.template(),
    material_point: ti.template(),
    soft_grid: ti.template(),
    soft_shape_node: ti.template(),
    soft_shape: ti.template(),
    soft_shape_count: ti.template(),
    rigid: ti.template(),
):
    """Rebuild current TLMPM nodal velocities without advancing mechanics."""
    for node in range(soft_grid_num):
        soft_grid[node].m = 0.0
        soft_grid[node].v = ZEROVEC3F

    for p in range(point_num):
        if int(material_point[p].active) == 1:
            sb, support = soft_point_support_index(p, material_point, soft, rigid)
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node(sb, support, n, soft, soft_shape_node)
                weighted_mass = soft_shape[support, n] * material_point[p].m
                ti.atomic_add(soft_grid[node].m, weighted_mass)
                ti.atomic_add(
                    soft_grid[node].v,
                    weighted_mass * material_point[p].v,
                )

    for node in range(soft_grid_num):
        if soft_grid[node].m > THRESHOLD:
            soft_grid[node].v /= soft_grid[node].m


@ti.func
def soft_sdf_projection_boundary_type(node, node_count):
    boundary_type = ti.min(2, node) - ti.min(node_count - 1 - node, 2)
    if boundary_type < 0:
        boundary_type += 3
    elif boundary_type > 0:
        boundary_type += 2
    return boundary_type


@ti.func
def soft_sdf_projection_weight_1d(
    shape_function_type: ti.template(),
    point,
    node_position,
    inv_dx,
    boundary_type,
):
    weight = 0.0
    if ti.static(shape_function_type == 0):
        weight = shape_linear(point, node_position, inv_dx, 0.0)
    elif ti.static(shape_function_type == 1):
        weight = shape_bspline_q(point, node_position, inv_dx, boundary_type)
    else:
        weight = shape_bspline_c(point, node_position, inv_dx, boundary_type)
    return weight


@ti.func
def scatter_soft_point_to_sdf_node(
    shape_function_type: ti.template(),
    sb,
    body_id,
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
        inside = inside and node_ijk[d] >= 0 and node_ijk[d] < box[body_id].gnum[d]
    if inside:
        node_position = xmin + ti.cast(node_ijk, float) * dx
        wx = soft_sdf_projection_weight_1d(
            shape_function_type,
            local_position[0],
            node_position[0],
            inv_dx,
            soft_sdf_projection_boundary_type(node_ijk[0], box[body_id].gnum[0]),
        )
        wy = soft_sdf_projection_weight_1d(
            shape_function_type,
            local_position[1],
            node_position[1],
            inv_dx,
            soft_sdf_projection_boundary_type(node_ijk[1], box[body_id].gnum[1]),
        )
        wz = soft_sdf_projection_weight_1d(
            shape_function_type,
            local_position[2],
            node_position[2],
            inv_dx,
            soft_sdf_projection_boundary_type(node_ijk[2], box[body_id].gnum[2]),
        )
        shape = wx * wy * wz
        if shape > THRESHOLD:
            local_grid = (
                node_ijk[0]
                + node_ijk[1] * box[body_id].gnum[0]
                + node_ijk[2] * box[body_id].gnum[0] * box[body_id].gnum[1]
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
def reset_soft_levelset_velocity_projection(
    soft_num: int,
    max_grid_num: int,
    soft: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
):
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            levelset_velocity[sdf_grid] = ZEROVEC3F
            projection_weight[sdf_grid] = 0.0


@ti.kernel
def project_current_soft_points_to_levelset(
    point_num: int,
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
    for p in range(point_num):
        if int(material_point[p].active) == 1:
            body_id = material_point[p].bodyID
            sb, support = soft_point_support_index(p, material_point, soft, rigid)
            scale = soft[sb].scale
            rotate = set_to_rotate(rigid[body_id].q)
            relative_position = material_point[p].x - rigid[body_id].mass_center
            local_position = rotate.transpose() @ relative_position / scale
            frame_velocity = soft[sb].v + rigid[body_id].w.cross(relative_position)
            local_velocity = rotate.transpose() @ (material_point[p].v - frame_velocity) / scale
            template_f_rate = ti.Matrix.zero(float, 3, 3)
            for n in range(soft_shape_count[support]):
                node = soft_point_support_node(sb, support, n, soft, soft_shape_node)
                if soft_grid[node].m > THRESHOLD:
                    template_f_rate += soft_grid[node].v.outer_product(soft_dshape[support, n])
            f_rate = template_f_rate @ soft[sb].referenceRotation.transpose() / scale
            velocity_gradient = ti.Matrix.zero(float, 3, 3)
            if ti.abs(material_point[p].F.determinant()) > THRESHOLD:
                velocity_gradient = f_rate @ material_point[p].F.inverse()
            frame_spin = mat3x3(
                [0.0, -rigid[body_id].w[2], rigid[body_id].w[1]],
                [rigid[body_id].w[2], 0.0, -rigid[body_id].w[0]],
                [-rigid[body_id].w[1], rigid[body_id].w[0], 0.0],
            )
            local_velocity_gradient = rotate.transpose() @ (velocity_gradient - frame_spin) @ rotate

            xmin = box[body_id].xmin / scale
            dx = box[body_id].grid_space / scale
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
                    point_support_sum += scatter_soft_point_to_sdf_node(
                        shape_function_type,
                        sb,
                        body_id,
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
                    point_support_sum += scatter_soft_point_to_sdf_node(
                        shape_function_type,
                        sb,
                        body_id,
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
                    point_support_sum += scatter_soft_point_to_sdf_node(
                        shape_function_type,
                        sb,
                        body_id,
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
def normalize_soft_levelset_velocity_projection(
    soft_num: int,
    max_grid_num: int,
    soft: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
):
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            if projection_weight[sdf_grid] > THRESHOLD:
                levelset_velocity[sdf_grid] /= projection_weight[sdf_grid]
            else:
                levelset_velocity[sdf_grid] = ZEROVEC3F


@ti.kernel
def extend_soft_levelset_velocity_projection(
    soft_num: int,
    max_grid_num: int,
    color: int,
    soft: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
):
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            body_id = soft[sb].bodyID
            gnum = box[body_id].gnum
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
                and projection_weight[sdf_grid] <= THRESHOLD
            ):
                accumulated_weight = 0.0
                accumulated_velocity = ZEROVEC3F
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
                            if neighbor_weight > THRESHOLD:
                                accumulated_weight += neighbor_weight
                                accumulated_velocity += neighbor_weight * levelset_velocity[neighbor_grid]
                                mean_neighbor_weight += neighbor_weight
                                neighbor_count += 1
                if accumulated_weight > THRESHOLD:
                    levelset_velocity[sdf_grid] = accumulated_velocity / accumulated_weight
                    projection_weight[sdf_grid] = mean_neighbor_weight / ti.max(neighbor_count, 1)


@ti.kernel
def audit_soft_levelset_velocity_projection(
    soft_num: int,
    max_grid_num: int,
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
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            body_id = soft[sb].bodyID
            sdf_grid = soft[sb].gridStart + local_grid
            dx = box[body_id].grid_space / soft[sb].scale
            if ti.abs(grid[sdf_grid].distance_field) <= monitor_band_cells * dx:
                ti.atomic_add(projection_band_nodes[None], 1)
                if projection_weight[sdf_grid] <= THRESHOLD:
                    ti.atomic_add(projection_uncovered_nodes[None], 1)


@ti.func
def soft_levelset_local_velocity(
    sb,
    local_pos,
    soft,
    levelset_velocity,
    projection_weight,
    box,
):
    body_id = soft[sb].bodyID
    scale = soft[sb].scale
    xmin = box[body_id].xmin / scale
    xmax = box[body_id].xmax / scale
    dx = box[body_id].grid_space / scale
    query = local_pos
    for d in ti.static(range(3)):
        query[d] = ti.min(ti.max(query[d], xmin[d]), xmax[d])
    base = ti.Vector([0, 0, 0])
    for d in ti.static(range(3)):
        base[d] = ti.min(
            box[body_id].gnum[d] - 2,
            ti.max(0, int((query[d] - xmin[d]) / dx)),
        )
    origin = xmin + vec3f(base[0], base[1], base[2]) * dx
    fraction = (query - origin) / dx
    local_velocity = ZEROVEC3F
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
                    + (base[1] + j) * box[body_id].gnum[0]
                    + (base[2] + k) * box[body_id].gnum[0] * box[body_id].gnum[1]
                )
                sdf_grid = soft[sb].gridStart + logical
                nodal_weight = weight * projection_weight[sdf_grid]
                active_weight += nodal_weight
                local_velocity += nodal_weight * levelset_velocity[sdf_grid]
    if active_weight > THRESHOLD:
        local_velocity /= active_weight
    return local_velocity


@ti.func
def soft_levelset_rk2_departure(
    sb,
    local_pos,
    direction,
    dt,
    soft,
    levelset_velocity,
    projection_weight,
    box,
):
    velocity0 = soft_levelset_local_velocity(sb, local_pos, soft, levelset_velocity, projection_weight, box)
    midpoint = local_pos + 0.5 * direction * dt * velocity0
    velocity_midpoint = soft_levelset_local_velocity(sb, midpoint, soft, levelset_velocity, projection_weight, box)
    return local_pos + direction * dt * velocity_midpoint


@ti.kernel
def predict_soft_levelset_maccormack(
    soft_num: int,
    max_grid_num: int,
    dt: float,
    soft: ti.template(),
    grid: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
) -> float:
    maximum_departure_excess = 0.0
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            body_id = soft[sb].bodyID
            scale = box[body_id].scale
            i = local_grid % box[body_id].gnum[0]
            j = (local_grid % (box[body_id].gnum[0] * box[body_id].gnum[1])) // box[body_id].gnum[0]
            k = local_grid // (box[body_id].gnum[0] * box[body_id].gnum[1])
            local_scaled = box[body_id].xmin + vec3f(i, j, k) * box[body_id].grid_space
            local_unscaled = local_scaled / scale
            sdf_grid = soft[sb].gridStart + local_grid
            backtrace = soft_levelset_rk2_departure(
                sb,
                local_unscaled,
                -1.0,
                dt,
                soft,
                levelset_velocity,
                projection_weight,
                box,
            )
            xmin = box[body_id].xmin / scale
            xmax = box[body_id].xmax / scale
            spacing = ti.max(box[body_id].grid_space / scale, THRESHOLD)
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
            grid[sdf_grid].distance_field_temp = sample_levelset_unscaled(backtrace, box[body_id], grid)

    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            grid[sdf_grid].distance_field = grid[sdf_grid].distance_field_temp
    return maximum_departure_excess


@ti.kernel
def correct_soft_levelset_maccormack(
    soft_num: int,
    max_grid_num: int,
    dt: float,
    soft: ti.template(),
    grid: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
) -> float:
    maximum_departure_excess = 0.0
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            body_id = soft[sb].bodyID
            scale = box[body_id].scale
            i = local_grid % box[body_id].gnum[0]
            j = (local_grid % (box[body_id].gnum[0] * box[body_id].gnum[1])) // box[body_id].gnum[0]
            k = local_grid // (box[body_id].gnum[0] * box[body_id].gnum[1])
            local_scaled = box[body_id].xmin + vec3f(i, j, k) * box[body_id].grid_space
            local_unscaled = local_scaled / scale
            reverse_trace = soft_levelset_rk2_departure(
                sb,
                local_unscaled,
                1.0,
                dt,
                soft,
                levelset_velocity,
                projection_weight,
                box,
            )
            xmin = box[body_id].xmin / scale
            xmax = box[body_id].xmax / scale
            spacing = ti.max(box[body_id].grid_space / scale, THRESHOLD)
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
            grid[sdf_grid].distance_field_temp = sample_levelset_unscaled(reverse_trace, box[body_id], grid)

    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            body_id = soft[sb].bodyID
            scale = box[body_id].scale
            i = local_grid % box[body_id].gnum[0]
            j = (local_grid % (box[body_id].gnum[0] * box[body_id].gnum[1])) // box[body_id].gnum[0]
            k = local_grid // (box[body_id].gnum[0] * box[body_id].gnum[1])
            local_scaled = box[body_id].xmin + vec3f(i, j, k) * box[body_id].grid_space
            local_unscaled = local_scaled / scale
            backtrace = soft_levelset_rk2_departure(
                sb,
                local_unscaled,
                -1.0,
                dt,
                soft,
                levelset_velocity,
                projection_weight,
                box,
            )
            xmin = box[body_id].xmin / scale
            xmax = box[body_id].xmax / scale
            spacing = ti.max(box[body_id].grid_space / scale, THRESHOLD)
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
            bounds = sample_levelset_reference_bounds_unscaled(backtrace, box[body_id], grid)
            grid[sdf_grid].distance_field = ti.min(ti.max(corrected, bounds[0]), bounds[1])
    return maximum_departure_excess


@ti.func
def soft_levelset_weno5_phi_offset(
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
def soft_levelset_weno5_left_flux(fm2, fm1, f0, fp1, fp2):
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
def soft_levelset_weno5_derivative(sb, ijk, axis, gnum, h, velocity, soft, grid):
    phi_0 = soft_levelset_weno5_phi_offset(sb, ijk, axis, 0, gnum, soft, grid)
    derivative = 0.0
    if ijk[axis] >= 3 and ijk[axis] < gnum[axis] - 3:
        phi_m3 = soft_levelset_weno5_phi_offset(sb, ijk, axis, -3, gnum, soft, grid)
        phi_m2 = soft_levelset_weno5_phi_offset(sb, ijk, axis, -2, gnum, soft, grid)
        phi_m1 = soft_levelset_weno5_phi_offset(sb, ijk, axis, -1, gnum, soft, grid)
        phi_p1 = soft_levelset_weno5_phi_offset(sb, ijk, axis, 1, gnum, soft, grid)
        phi_p2 = soft_levelset_weno5_phi_offset(sb, ijk, axis, 2, gnum, soft, grid)
        phi_p3 = soft_levelset_weno5_phi_offset(sb, ijk, axis, 3, gnum, soft, grid)
        flux_plus = 0.0
        flux_minus = 0.0
        if velocity >= 0.0:
            flux_plus = soft_levelset_weno5_left_flux(phi_m2, phi_m1, phi_0, phi_p1, phi_p2)
            flux_minus = soft_levelset_weno5_left_flux(phi_m3, phi_m2, phi_m1, phi_0, phi_p1)
        else:
            flux_plus = soft_levelset_weno5_left_flux(phi_p3, phi_p2, phi_p1, phi_0, phi_m1)
            flux_minus = soft_levelset_weno5_left_flux(phi_p2, phi_p1, phi_0, phi_m1, phi_m2)
        derivative = (flux_plus - flux_minus) / h
    elif velocity >= 0.0:
        if ijk[axis] > 0:
            phi_m1 = soft_levelset_weno5_phi_offset(sb, ijk, axis, -1, gnum, soft, grid)
            derivative = (phi_0 - phi_m1) / h
        elif gnum[axis] > 1:
            phi_p1 = soft_levelset_weno5_phi_offset(sb, ijk, axis, 1, gnum, soft, grid)
            derivative = (phi_p1 - phi_0) / h
    else:
        if ijk[axis] < gnum[axis] - 1:
            phi_p1 = soft_levelset_weno5_phi_offset(sb, ijk, axis, 1, gnum, soft, grid)
            derivative = (phi_p1 - phi_0) / h
        elif gnum[axis] > 1:
            phi_m1 = soft_levelset_weno5_phi_offset(sb, ijk, axis, -1, gnum, soft, grid)
            derivative = (phi_0 - phi_m1) / h
    return derivative


@ti.kernel
def store_soft_levelset_weno5_reference(
    soft_num: int,
    max_grid_num: int,
    soft: ti.template(),
    grid: ti.template(),
):
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            grid[sdf_grid].distance_field0 = grid[sdf_grid].distance_field
            grid[sdf_grid].distance_field_temp = grid[sdf_grid].distance_field


@ti.kernel
def maximum_soft_levelset_weno5_cfl(
    soft_num: int,
    max_grid_num: int,
    dt: float,
    soft: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
) -> float:
    maximum_cfl = 0.0
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            if projection_weight[sdf_grid] > THRESHOLD:
                body_id = soft[sb].bodyID
                h = box[body_id].grid_space / ti.max(box[body_id].scale, THRESHOLD)
                velocity = levelset_velocity[sdf_grid]
                local_cfl = dt * (ti.abs(velocity[0]) + ti.abs(velocity[1]) + ti.abs(velocity[2])) / h
                ti.atomic_max(maximum_cfl, local_cfl)
    return maximum_cfl


@ti.kernel
def advance_soft_levelset_weno5_stage(
    soft_num: int,
    max_grid_num: int,
    dt: float,
    stage: int,
    soft: ti.template(),
    grid: ti.template(),
    levelset_velocity: ti.template(),
    projection_weight: ti.template(),
    box: ti.template(),
):
    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            body_id = soft[sb].bodyID
            gnum = box[body_id].gnum
            i = local_grid % gnum[0]
            j = (local_grid % (gnum[0] * gnum[1])) // gnum[0]
            k = local_grid // (gnum[0] * gnum[1])
            ijk = vec3i(i, j, k)
            sdf_grid = soft[sb].gridStart + local_grid
            phi = grid[sdf_grid].distance_field
            euler_value = phi
            if projection_weight[sdf_grid] > THRESHOLD:
                h = box[body_id].grid_space / ti.max(box[body_id].scale, THRESHOLD)
                velocity = levelset_velocity[sdf_grid]
                dphi_x = soft_levelset_weno5_derivative(sb, ijk, 0, gnum, h, velocity[0], soft, grid)
                dphi_y = soft_levelset_weno5_derivative(sb, ijk, 1, gnum, h, velocity[1], soft, grid)
                dphi_z = soft_levelset_weno5_derivative(sb, ijk, 2, gnum, h, velocity[2], soft, grid)
                euler_value = phi - dt * (velocity[0] * dphi_x + velocity[1] * dphi_y + velocity[2] * dphi_z)
            phi0 = grid[sdf_grid].distance_field0
            phi_new = euler_value
            if stage == 2:
                phi_new = 0.75 * phi0 + 0.25 * euler_value
            elif stage == 3:
                phi_new = (1.0 / 3.0) * phi0 + (2.0 / 3.0) * euler_value
            grid[sdf_grid].distance_field_temp = phi_new

    for sb, local_grid in ti.ndrange(soft_num, max_grid_num):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            grid[sdf_grid].distance_field = grid[sdf_grid].distance_field_temp


def advect_projected_soft_levelset_weno5(
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
        maximum_soft_levelset_weno5_cfl(
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
        store_soft_levelset_weno5_reference(softNum, maxGridNum, soft, grid)
        for stage in (1, 2, 3):
            advance_soft_levelset_weno5_stage(
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


def project_soft_points_to_levelset_velocity(
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
    reset_soft_levelset_velocity_projection(
        softNum,
        maxGridNum,
        soft,
        levelset_velocity,
        projection_weight,
    )
    project_current_soft_points_to_levelset(
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
    normalize_soft_levelset_velocity_projection(
        softNum,
        maxGridNum,
        soft,
        levelset_velocity,
        projection_weight,
    )
    for _ in range(max(int(projection_extension_iterations), 0)):
        extend_soft_levelset_velocity_projection(
            softNum,
            maxGridNum,
            0,
            soft,
            levelset_velocity,
            projection_weight,
            box,
        )
        extend_soft_levelset_velocity_projection(
            softNum,
            maxGridNum,
            1,
            soft,
            levelset_velocity,
            projection_weight,
            box,
        )
    audit_soft_levelset_velocity_projection(
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


def advect_soft_levelset(
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
    project_soft_points_to_levelset_velocity(
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
    prediction_excess = predict_soft_levelset_maccormack(
        softNum,
        maxGridNum,
        dt,
        soft,
        grid,
        levelset_velocity,
        projection_weight,
        box,
    )
    correction_excess = correct_soft_levelset_maccormack(
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


def advect_soft_levelset_weno5(
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
    project_soft_points_to_levelset_velocity(
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
    advect_projected_soft_levelset_weno5(
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
