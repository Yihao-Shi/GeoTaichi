import taichi as ti
from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import (
    EquivalentDeviatoricStrain,
    EquivalentDeviatoricStress,
    EquivalentStrain,
)
from src.utils.constants import Threshold
from src.utils.ScalarFunction import linearize, vectorize_id
from src.utils.TypeDefination import vec3f, vec3i


@ti.kernel
def initialize_adaptive_particle_size(
    particleNum: int,
    characteristic_mode: int,
    coarse_grid_size: ti.types.vector(3, float),
    psize: ti.types.ndarray(),
    particle_size: ti.template(),
    calLength: ti.template(),
):
    for np in range(particleNum):
        particle_size[np] = vec3f(psize[np, 0], psize[np, 1], psize[np, 2])
        calLength[np] = vec3f(0.0, 0.0, 0.0)
        if characteristic_mode == 1:
            calLength[np] = particle_size[np]
        elif characteristic_mode == 2:
            calLength[np] = 0.5 * coarse_grid_size
        elif characteristic_mode == 3:
            calLength[np] = coarse_grid_size


@ti.func
def adaptive_bspline_boundary_type(node_index: int, node_count: int):
    boundary_type = min(2, node_index) - min(node_count - 1 - node_index, 2)
    if boundary_type < 0:
        boundary_type += 3
    elif boundary_type > 0:
        boundary_type += 2
    return boundary_type


@ti.func
def scaled_penalty_relaxation(
    penalty_cap: float,
    penalty_beta: float,
    penalty_young: float,
    penalty_length: float,
    reference_volume: float,
    dt: ti.template(),
    inverse_effective_mass: float,
):
    relaxation = penalty_cap
    if penalty_beta > Threshold and penalty_young > Threshold and penalty_length > Threshold:
        stiffness = penalty_beta * penalty_young * ti.sqrt(ti.max(reference_volume, Threshold)) / penalty_length
        scaled = stiffness * dt[None] * dt[None] * inverse_effective_mass
        relaxation = scaled / (1.0 + scaled)
        relaxation = ti.min(relaxation, penalty_cap)
    return ti.min(ti.max(relaxation, 0.0), 1.0)


@ti.func
def append_adaptive_support_node(
    active_id: int,
    row_end: int,
    compact_id: int,
    shape_value: float,
    dshape_value: ti.types.vector(3, float),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
):
    if compact_id >= 0 and active_id < row_end:
        LnID[active_id] = compact_id
        shape_fn[active_id] = shape_value
        dshape_fn[active_id] = dshape_value
        active_id += 1
    return active_id


@ti.func
def append_or_accumulate_adaptive_support_node(
    row_start: int,
    active_id: int,
    row_end: int,
    compact_id: int,
    shape_value: float,
    dshape_value: ti.types.vector(3, float),
    support_overflow: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
):
    found = 0
    for slot in range(row_start, active_id):
        if found == 0 and LnID[slot] == compact_id:
            shape_fn[slot] += shape_value
            dshape_fn[slot] += dshape_value
            found = 1

    if found == 0:
        if compact_id >= 0 and active_id < row_end:
            LnID[active_id] = compact_id
            shape_fn[active_id] = shape_value
            dshape_fn[active_id] = dshape_value
            active_id += 1
        else:
            support_overflow[None] = 1
    return active_id


@ti.func
def append_projected_hanging_node(
    active_id: int,
    row_start: int,
    row_end: int,
    compact_id: int,
    shape_value: float,
    dshape_value: ti.types.vector(3, float),
    strong_hanging: int,
    hanging_lookup: ti.template(),
    hanging_master_id: ti.template(),
    hanging_master_weight: ti.template(),
    support_overflow: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
):
    if strong_hanging == 1:
        constraint_id = hanging_lookup[compact_id]
        if constraint_id >= 0:
            for master in ti.static(range(8)):
                weight = hanging_master_weight[constraint_id][master]
                if weight > Threshold:
                    active_id = append_or_accumulate_adaptive_support_node(
                        row_start,
                        active_id,
                        row_end,
                        hanging_master_id[constraint_id][master],
                        weight * shape_value,
                        weight * dshape_value,
                        support_overflow,
                        LnID,
                        shape_fn,
                        dshape_fn,
                    )
        else:
            active_id = append_or_accumulate_adaptive_support_node(
                row_start,
                active_id,
                row_end,
                compact_id,
                shape_value,
                dshape_value,
                support_overflow,
                LnID,
                shape_fn,
                dshape_fn,
            )
    else:
        active_id = append_adaptive_support_node(
            active_id,
            row_end,
            compact_id,
            shape_value,
            dshape_value,
            LnID,
            shape_fn,
            dshape_fn,
        )
    return active_id


@ti.kernel
def adaptive_global_update(
    total_nodes: int,
    influenced_node: int,
    shape_mode: int,
    strong_hanging: int,
    refinement_ratio: int,
    max_refinement_level: int,
    finest_ratio: int,
    coarse_grid_size: ti.types.vector(3, float),
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    fine_grid_size: ti.types.vector(3, float),
    fine_igrid_size: ti.types.vector(3, float),
    fine_gnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    calLength: ti.template(),
    refined_cell: ti.template(),
    particle_level: ti.template(),
    logical_to_compact: ti.template(),
    hanging_lookup: ti.template(),
    hanging_master_id: ti.template(),
    hanging_master_weight: ti.template(),
    support_overflow: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
):
    for np in range(particleNum):
        position = particle[np].x
        coarse_cell = ti.floor(position * coarse_igrid_size, int)
        coarse_cell = ti.min(ti.max(coarse_cell, 0), coarse_cnum - 1)
        coarse_cell_id = linearize(coarse_cell, coarse_cnum)
        level = int(refined_cell[coarse_cell_id])
        level = ti.min(level, max_refinement_level)
        particle_level[np] = ti.cast(level, ti.u8)

        element_size = coarse_grid_size
        ielement_size = coarse_igrid_size
        level_gnum = coarse_cnum + 1
        level_divisions = 1
        if level == 1:
            level_divisions = refinement_ratio
        elif level >= 2:
            level_divisions = refinement_ratio * refinement_ratio
        element_size = coarse_grid_size / float(level_divisions)
        ielement_size = coarse_igrid_size * float(level_divisions)
        level_gnum = coarse_cnum * level_divisions + 1
        node_stride = finest_ratio // level_divisions

        psize = calLength[np]
        support_shift = psize
        if shape_mode == 1:
            support_shift = 0.5 * element_size
        elif shape_mode == 2:
            support_shift = element_size
        base_bound = ti.floor((position - support_shift) * ielement_size, int)
        row_start = np * total_nodes
        row_end = (np + 1) * total_nodes
        activeID = row_start
        for k in range(base_bound[2], base_bound[2] + influenced_node):
            for j in range(base_bound[1], base_bound[1] + influenced_node):
                for i in range(base_bound[0], base_bound[0] + influenced_node):
                    level_node = vec3i(i, j, k)
                    fine_node = level_node * node_stride
                    if all(fine_node >= 0) and all(fine_node < fine_gnum):
                        node_coords = level_node * element_size
                        shape_parameter = psize
                        if shape_mode > 0:
                            shape_parameter = vec3f(
                                adaptive_bspline_boundary_type(level_node[0], level_gnum[0]),
                                adaptive_bspline_boundary_type(level_node[1], level_gnum[1]),
                                adaptive_bspline_boundary_type(level_node[2], level_gnum[2]),
                            )
                        shapen0 = shape_function(position[0], node_coords[0], ielement_size[0], shape_parameter[0])
                        shapen1 = shape_function(position[1], node_coords[1], ielement_size[1], shape_parameter[1])
                        shapen2 = shape_function(position[2], node_coords[2], ielement_size[2], shape_parameter[2])
                        shapeval = shapen0 * shapen1 * shapen2
                        if shapeval > Threshold:
                            logical_id = linearize(fine_node, fine_gnum)
                            compact_id = logical_to_compact[logical_id]
                            if compact_id >= 0:
                                dshapen0 = grad_shape_function(
                                    position[0], node_coords[0], ielement_size[0], shape_parameter[0]
                                )
                                dshapen1 = grad_shape_function(
                                    position[1], node_coords[1], ielement_size[1], shape_parameter[1]
                                )
                                dshapen2 = grad_shape_function(
                                    position[2], node_coords[2], ielement_size[2], shape_parameter[2]
                                )
                                activeID = append_projected_hanging_node(
                                    activeID,
                                    row_start,
                                    row_end,
                                    compact_id,
                                    shapeval,
                                    vec3f(
                                        dshapen0 * shapen1 * shapen2,
                                        shapen0 * dshapen1 * shapen2,
                                        shapen0 * shapen1 * dshapen2,
                                    ),
                                    strong_hanging,
                                    hanging_lookup,
                                    hanging_master_id,
                                    hanging_master_weight,
                                    support_overflow,
                                    LnID,
                                    shape_fn,
                                    dshape_fn,
                                )
        node_size[np] = ti.cast(activeID - row_start, ti.u8)


@ti.func
def append_bridging_support(
    active_id,
    row_end,
    position,
    psize,
    weight,
    influenced_node,
    shape_mode,
    element_size,
    ielement_size,
    level_gnum,
    node_stride,
    fine_gnum,
    level_to_compact: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
):
    support_shift = psize
    if shape_mode == 1:
        support_shift = 0.5 * element_size
    elif shape_mode == 2:
        support_shift = element_size
    base_bound = ti.floor((position - support_shift) * ielement_size, int)

    for k in range(base_bound[2], base_bound[2] + influenced_node):
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                level_node = vec3i(i, j, k)
                fine_node = level_node * node_stride
                if all(fine_node >= 0) and all(fine_node < fine_gnum):
                    node_coords = level_node * element_size
                    shape_parameter = psize
                    if shape_mode > 0:
                        shape_parameter = vec3f(
                            adaptive_bspline_boundary_type(level_node[0], level_gnum[0]),
                            adaptive_bspline_boundary_type(level_node[1], level_gnum[1]),
                            adaptive_bspline_boundary_type(level_node[2], level_gnum[2]),
                        )
                    shapen0 = shape_function(
                        position[0],
                        node_coords[0],
                        ielement_size[0],
                        shape_parameter[0],
                    )
                    shapen1 = shape_function(
                        position[1],
                        node_coords[1],
                        ielement_size[1],
                        shape_parameter[1],
                    )
                    shapen2 = shape_function(
                        position[2],
                        node_coords[2],
                        ielement_size[2],
                        shape_parameter[2],
                    )
                    shapeval = shapen0 * shapen1 * shapen2
                    if shapeval > Threshold:
                        logical_id = linearize(fine_node, fine_gnum)
                        compact_id = level_to_compact[logical_id]
                        assert compact_id >= 0, "AdaptiveGrid bridging node map is incomplete"
                        if compact_id >= 0:
                            dshapen0 = grad_shape_function(
                                position[0],
                                node_coords[0],
                                ielement_size[0],
                                shape_parameter[0],
                            )
                            dshapen1 = grad_shape_function(
                                position[1],
                                node_coords[1],
                                ielement_size[1],
                                shape_parameter[1],
                            )
                            dshapen2 = grad_shape_function(
                                position[2],
                                node_coords[2],
                                ielement_size[2],
                                shape_parameter[2],
                            )
                            active_id = append_adaptive_support_node(
                                active_id,
                                row_end,
                                compact_id,
                                weight * shapeval,
                                weight
                                * vec3f(
                                    dshapen0 * shapen1 * shapen2,
                                    shapen0 * dshapen1 * shapen2,
                                    shapen0 * shapen1 * dshapen2,
                                ),
                                LnID,
                                shape_fn,
                                dshape_fn,
                            )
    return active_id


@ti.kernel
def adaptive_bridging_global_update(
    total_nodes: int,
    level_nodes: int,
    influenced_node: int,
    shape_mode: int,
    refinement_ratio: int,
    coarse_grid_size: ti.types.vector(3, float),
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    fine_grid_size: ti.types.vector(3, float),
    fine_igrid_size: ti.types.vector(3, float),
    fine_gnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    calLength: ti.template(),
    refined_cell: ti.template(),
    particle_level: ti.template(),
    logical_to_compact: ti.template(),
    fine_logical_to_compact: ti.template(),
    bridge_coarse_weight: ti.template(),
    bridge_alpha: ti.template(),
    bridge_coarse_size: ti.template(),
    bridge_body_id: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
):
    for np in range(particleNum):
        position = particle[np].x
        coarse_cell = ti.floor(position * coarse_igrid_size, int)
        coarse_cell = ti.min(ti.max(coarse_cell, 0), coarse_cnum - 1)
        coarse_cell_id = linearize(coarse_cell, coarse_cnum)
        level = int(refined_cell[coarse_cell_id])
        particle_level[np] = ti.cast(level, ti.u8)
        bridge_body_id[np] = particle[np].bodyID

        alpha = 1.0
        if level == 1:
            alpha = 0.0
            local_coord = position * coarse_igrid_size - coarse_cell.cast(float)
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                weight = 1.0
                for d in ti.static(range(3)):
                    weight *= local_coord[d] if offset[d] == 1 else 1.0 - local_coord[d]
                coarse_node = coarse_cell + offset
                alpha += weight * bridge_coarse_weight[linearize(coarse_node, coarse_cnum + 1)]
            alpha = ti.min(ti.max(alpha, 0.0), 1.0)
        bridge_alpha[np] = alpha

        active_id = np * total_nodes
        psize = calLength[np]
        if alpha > Threshold:
            active_id = append_bridging_support(
                active_id,
                (np + 1) * total_nodes,
                position,
                psize,
                alpha,
                influenced_node,
                shape_mode,
                coarse_grid_size,
                coarse_igrid_size,
                coarse_cnum + 1,
                refinement_ratio,
                fine_gnum,
                logical_to_compact,
                LnID,
                shape_fn,
                dshape_fn,
                shape_function,
                grad_shape_function,
            )
        coarse_size = active_id - np * total_nodes
        bridge_coarse_size[np] = ti.cast(coarse_size, ti.u8)

        if level == 1 and 1.0 - alpha > Threshold:
            active_id = append_bridging_support(
                active_id,
                (np + 1) * total_nodes,
                position,
                psize,
                1.0 - alpha,
                influenced_node,
                shape_mode,
                fine_grid_size,
                fine_igrid_size,
                fine_gnum,
                1,
                fine_gnum,
                fine_logical_to_compact,
                LnID,
                shape_fn,
                dshape_fn,
                shape_function,
                grad_shape_function,
            )
        assert active_id - np * total_nodes <= 2 * level_nodes
        node_size[np] = ti.cast(active_id - np * total_nodes, ti.u8)


@ti.kernel
def assemble_bridging_penalty_impulse(
    cutoff: float,
    penalty: float,
    penalty_beta: float,
    penalty_young: float,
    penalty_length: float,
    dt: ti.template(),
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    bridge_alpha: ti.template(),
    bridge_coarse_size: ti.template(),
    bridge_body_id: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    node: ti.template(),
    penalty_impulse: ti.template(),
):
    for np in range(particleNum):
        alpha = bridge_alpha[np]
        coarse_size = int(bridge_coarse_size[np])
        support_size = int(node_size[np])
        if alpha > Threshold and alpha < 1.0 - Threshold and coarse_size > 0 and support_size > coarse_size:
            body_id = int(bridge_body_id[np])
            active_id = np * total_nodes
            coarse_velocity = vec3f(0.0, 0.0, 0.0)
            fine_velocity = vec3f(0.0, 0.0, 0.0)
            inverse_effective_mass = 0.0
            support_is_active = True

            for local_id in range(support_size):
                shape = shape_fn[active_id + local_id]
                if local_id < coarse_size:
                    shape /= alpha
                else:
                    shape /= 1.0 - alpha
                node_id = LnID[active_id + local_id]
                mass = node[node_id, body_id].m
                if mass <= cutoff:
                    support_is_active = False
                else:
                    inverse_effective_mass += shape * shape / mass
                    if local_id < coarse_size:
                        coarse_velocity += shape * node[node_id, body_id].momentum
                    else:
                        fine_velocity += shape * node[node_id, body_id].momentum

            if support_is_active and inverse_effective_mass > Threshold:
                relaxation = scaled_penalty_relaxation(
                    penalty,
                    penalty_beta,
                    penalty_young,
                    penalty_length,
                    particle[np].vol,
                    dt,
                    inverse_effective_mass,
                )
                impulse = relaxation * (coarse_velocity - fine_velocity) / inverse_effective_mass
                for local_id in range(support_size):
                    shape = shape_fn[active_id + local_id]
                    direction = 1.0
                    if local_id < coarse_size:
                        shape /= alpha
                        direction = -1.0
                    else:
                        shape /= 1.0 - alpha
                    node_id = LnID[active_id + local_id]
                    for d in ti.static(range(3)):
                        ti.atomic_add(
                            penalty_impulse[node_id, body_id][d],
                            direction * shape * impulse[d],
                        )


@ti.kernel
def mark_refined_cells_epdstrain(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    stateVars: ti.template(),
    current_refined_cell: ti.template(),
    refined_cell: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            current = int(current_refined_cell[cell_id])
            if current < max_refinement_level and stateVars[np].epdstrain >= threshold:
                refined_cell[cell_id] = ti.cast(current + 1, ti.u8)


@ti.kernel
def mark_refined_cells_epstrain(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    stateVars: ti.template(),
    current_refined_cell: ti.template(),
    refined_cell: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            current = int(current_refined_cell[cell_id])
            if current < max_refinement_level and stateVars[np].epstrain >= threshold:
                refined_cell[cell_id] = ti.cast(current + 1, ti.u8)


@ti.kernel
def mark_refined_cells_strain(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    stateVars: ti.template(),
    current_refined_cell: ti.template(),
    refined_cell: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            current = int(current_refined_cell[cell_id])
            if current < max_refinement_level and EquivalentStrain(stateVars[np].strain) >= threshold:
                refined_cell[cell_id] = ti.cast(current + 1, ti.u8)


@ti.kernel
def mark_refined_cells_deviatoric_strain(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    stateVars: ti.template(),
    current_refined_cell: ti.template(),
    refined_cell: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            current = int(current_refined_cell[cell_id])
            if current < max_refinement_level and EquivalentDeviatoricStrain(stateVars[np].strain) >= threshold:
                refined_cell[cell_id] = ti.cast(current + 1, ti.u8)


@ti.kernel
def mark_refined_cells_stress(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    stateVars: ti.template(),
    current_refined_cell: ti.template(),
    refined_cell: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            current = int(current_refined_cell[cell_id])
            if current < max_refinement_level and EquivalentDeviatoricStress(particle[np].stress) >= threshold:
                refined_cell[cell_id] = ti.cast(current + 1, ti.u8)


@ti.func
def matrix_equivalent_deviatoric_stress(stress):
    mean_stress = 0.0
    for i in ti.static(range(stress.n)):
        mean_stress += stress[i, i]
    mean_stress /= stress.n

    dev_norm2 = 0.0
    for i, j in ti.static(ti.ndrange(stress.n, stress.m)):
        value = stress[i, j]
        if ti.static(i == j):
            value -= mean_stress
        dev_norm2 += value * value
    return ti.sqrt(1.5 * dev_norm2)


@ti.kernel
def mark_refined_cells_matrix_stress(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    stateVars: ti.template(),
    current_refined_cell: ti.template(),
    refined_cell: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            current = int(current_refined_cell[cell_id])
            if current < max_refinement_level and matrix_equivalent_deviatoric_stress(particle[np].stress) >= threshold:
                refined_cell[cell_id] = ti.cast(current + 1, ti.u8)


@ti.kernel
def mark_refined_cells_softening(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    stateVars: ti.template(),
    current_refined_cell: ti.template(),
    refined_cell: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            current = int(current_refined_cell[cell_id])
            if current < max_refinement_level and stateVars[np].epdstrain >= threshold:
                refined_cell[cell_id] = ti.cast(current + 1, ti.u8)


@ti.kernel
def mark_refined_cells_migrated_particles(
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    current_refined_cell: ti.template(),
    particle_refined: ti.template(),
    refined_cell: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0 and particle_refined[np] > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            target_level = ti.max(int(current_refined_cell[cell_id]), int(particle_refined[np]))
            if target_level > int(current_refined_cell[cell_id]):
                refined_cell[cell_id] = ti.cast(target_level, ti.u8)


@ti.kernel
def dilate_refined_cells(
    coarse_cnum: ti.types.vector(3, int), refined_cell: ti.template(), refined_cell_buffer: ti.template()
):
    for cell_id in range(refined_cell.shape[0]):
        level = int(refined_cell[cell_id])
        if level > 0:
            cell = ti.Vector(vectorize_id(cell_id, coarse_cnum))
            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                neighbor = cell + offset
                if all(neighbor >= 0) and all(neighbor < coarse_cnum):
                    ti.atomic_max(refined_cell_buffer[linearize(neighbor, coarse_cnum)], refined_cell[cell_id])


@ti.kernel
def merge_refined_cells(refined_cell: ti.template(), refined_cell_buffer: ti.template()):
    for cell_id in range(refined_cell.shape[0]):
        if refined_cell_buffer[cell_id] > refined_cell[cell_id]:
            refined_cell[cell_id] = refined_cell_buffer[cell_id]


@ti.kernel
def count_refined_cells(refined_cell: ti.template(), refined_cell_count: ti.template()):
    refined_cell_count[None] = 0
    for cell_id in range(refined_cell.shape[0]):
        if refined_cell[cell_id] > 0:
            ti.atomic_add(refined_cell_count[None], 1)


@ti.kernel
def count_new_refined_cells(
    refined_cell: ti.template(), refinement_seed: ti.template(), new_refined_cell_count: ti.template()
):
    new_refined_cell_count[None] = 0
    for cell_id in range(refined_cell.shape[0]):
        if refinement_seed[cell_id] > refined_cell[cell_id]:
            ti.atomic_add(new_refined_cell_count[None], 1)


@ti.kernel
def assign_particle_levels(
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    refined_cell: ti.template(),
    particle_level: ti.template(),
):
    for np in range(particleNum):
        cell = ti.floor(particle[np].x * coarse_igrid_size, int)
        cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
        particle_level[np] = refined_cell[linearize(cell, coarse_cnum)]


@ti.kernel
def accumulate_split_cell_volume(
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    split_cell_volume: ti.template(),
):
    for cell_id in range(split_cell_volume.shape[0]):
        split_cell_volume[cell_id] = 0.0

    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            ti.atomic_add(
                split_cell_volume[linearize(cell, coarse_cnum)],
                particle[np].vol,
            )


@ti.kernel
def collect_particles_to_split(
    coarse_igrid_size: ti.types.vector(3, float),
    coarse_cnum: ti.types.vector(3, int),
    particleNum: int,
    max_split_parents: int,
    min_split_cell_volume: float,
    particle: ti.template(),
    refined_cell: ti.template(),
    split_cell_volume: ti.template(),
    particle_refined: ti.template(),
    unrefined_particle_count: ti.template(),
    split_particle_count: ti.template(),
    split_parent_id: ti.template(),
):
    split_particle_count[None] = 0
    unrefined_particle_count[None] = 0
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            ti.atomic_add(unrefined_particle_count[None], 1)
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            can_split_cell = min_split_cell_volume <= Threshold or split_cell_volume[cell_id] >= min_split_cell_volume
            if refined_cell[cell_id] > particle_refined[np] and can_split_cell:
                split_id = ti.atomic_add(split_particle_count[None], 1)
                if split_id < max_split_parents:
                    split_parent_id[split_id] = np


@ti.kernel
def copy_refined_particle_children(
    split_parent_num: int,
    old_particle_num: int,
    split_parent_id: ti.template(),
    particle: ti.template(),
    stateVars: ti.template(),
    particle_capacity: int,
    particle_split_overflow: ti.template(),
):
    for append_id in range(7 * split_parent_num):
        target = old_particle_num + append_id
        if target < particle_capacity:
            parent = split_parent_id[append_id // 7]
            particle[target] = particle[parent]
            stateVars[target] = stateVars[parent]
        else:
            ti.atomic_add(particle_split_overflow[None], 1)


@ti.kernel
def cache_refined_particle_parents(
    split_parent_num: int,
    split_parent_id: ti.template(),
    particle: ti.template(),
    particle_size: ti.template(),
    calLength: ti.template(),
    parent_position: ti.template(),
    parent_velocity: ti.template(),
    parent_velocity_gradient: ti.template(),
    child_size: ti.template(),
    child_cal_length: ti.template(),
    child_mass: ti.template(),
    child_volume: ti.template(),
    child_level: ti.template(),
    particle_refined: ti.template(),
):
    for split_id in range(split_parent_num):
        parent = split_parent_id[split_id]
        parent_position[split_id] = particle[parent].x
        parent_velocity[split_id] = particle[parent].v
        parent_velocity_gradient[split_id] = particle[parent].velocity_gradient
        child_size[split_id] = 0.5 * particle_size[parent]
        child_cal_length[split_id] = 0.5 * calLength[parent]
        child_mass[split_id] = 0.125 * particle[parent].m
        child_volume[split_id] = 0.125 * particle[parent].vol
        child_level[split_id] = ti.cast(int(particle_refined[parent]) + 1, ti.u8)


@ti.kernel
def split_refined_particles(
    split_parent_num: int,
    old_particle_num: int,
    split_parent_id: ti.template(),
    particle: ti.template(),
    parent_position: ti.template(),
    parent_velocity: ti.template(),
    parent_velocity_gradient: ti.template(),
    child_size: ti.template(),
    child_cal_length: ti.template(),
    child_mass: ti.template(),
    child_volume: ti.template(),
    child_level: ti.template(),
    particle_size: ti.template(),
    calLength: ti.template(),
    particle_level: ti.template(),
    particle_refined: ti.template(),
):
    for child_task in range(8 * split_parent_num):
        split_id = child_task // 8
        child_index = child_task - 8 * split_id
        target = split_parent_id[split_id]
        appended_start = old_particle_num + 7 * split_id
        if child_index > 0:
            target = appended_start + child_index - 1

        offset = vec3f(
            ti.cast(2 * (child_index % 2) - 1, float),
            ti.cast(2 * ((child_index // 2) % 2) - 1, float),
            ti.cast(2 * (child_index // 4) - 1, float),
        )
        child_offset = 0.999 * offset * child_size[split_id]
        particle[target].particleID = target
        particle[target].m = child_mass[split_id]
        particle[target].vol = child_volume[split_id]
        particle[target].x = parent_position[split_id] + child_offset
        particle[target].v = parent_velocity[split_id]
        particle[target].velocity_gradient = ti.Matrix.zero(float, 3, 3)
        particle_size[target] = child_size[split_id]
        calLength[target] = child_cal_length[split_id]
        particle_level[target] = child_level[split_id]
        particle_refined[target] = child_level[split_id]


@ti.kernel
def reset_hanging_penalty_impulse_list(
    touched_count: int, touched_node_id: ti.template(), penalty_impulse: ti.template()
):
    for touched_id, body_id in ti.ndrange(touched_count, penalty_impulse.shape[1]):
        node_id = touched_node_id[touched_id]
        penalty_impulse[node_id, body_id] = vec3f(0.0, 0.0, 0.0)


@ti.kernel
def load_hanging_constraint_table(
    hanging_count: int,
    touched_count: int,
    slave_ids: ti.types.ndarray(),
    master_ids: ti.types.ndarray(),
    master_weights: ti.types.ndarray(),
    touched_ids: ti.types.ndarray(),
    hanging_node_id: ti.template(),
    hanging_lookup: ti.template(),
    hanging_master_id: ti.template(),
    hanging_master_weight: ti.template(),
    hanging_touched_node_id: ti.template(),
):
    for constraint_id in range(hanging_count):
        node_id = slave_ids[constraint_id]
        hanging_node_id[constraint_id] = node_id
        hanging_lookup[node_id] = constraint_id
        for master in ti.static(range(8)):
            hanging_master_id[constraint_id][master] = master_ids[constraint_id, master]
            hanging_master_weight[constraint_id][master] = master_weights[constraint_id, master]

    for touched_id in range(touched_count):
        hanging_touched_node_id[touched_id] = touched_ids[touched_id]


@ti.kernel
def assemble_hanging_penalty_impulse_list(
    cutoff: float,
    penalty: float,
    penalty_beta: float,
    penalty_young: float,
    penalty_length: float,
    reference_volume: float,
    dt: ti.template(),
    hanging_count: int,
    hanging_node_id: ti.template(),
    hanging_master_id: ti.template(),
    hanging_master_weight: ti.template(),
    node: ti.template(),
    penalty_impulse: ti.template(),
):
    for constraint_id, body_id in ti.ndrange(hanging_count, node.shape[1]):
        node_id = hanging_node_id[constraint_id]
        slave_mass = node[node_id, body_id].m
        if slave_mass > cutoff:
            master_velocity = vec3f(0.0, 0.0, 0.0)
            inverse_effective_mass = 1.0 / slave_mass
            masters_are_active = True

            for master in ti.static(range(8)):
                weight = hanging_master_weight[constraint_id][master]
                if weight > Threshold:
                    master_id = hanging_master_id[constraint_id][master]
                    master_mass = node[master_id, body_id].m
                    if master_mass <= cutoff:
                        masters_are_active = False
                    else:
                        master_velocity += weight * node[master_id, body_id].momentum
                        inverse_effective_mass += weight * weight / master_mass

            if masters_are_active:
                mismatch = node[node_id, body_id].momentum - master_velocity
                relaxation = scaled_penalty_relaxation(
                    penalty,
                    penalty_beta,
                    penalty_young,
                    penalty_length,
                    reference_volume,
                    dt,
                    inverse_effective_mass,
                )
                impulse = relaxation * mismatch / inverse_effective_mass
                for d in ti.static(range(3)):
                    ti.atomic_add(penalty_impulse[node_id, body_id][d], -impulse[d])

                for master in ti.static(range(8)):
                    weight = hanging_master_weight[constraint_id][master]
                    if weight > Threshold:
                        master_id = hanging_master_id[constraint_id][master]
                        for d in ti.static(range(3)):
                            ti.atomic_add(
                                penalty_impulse[master_id, body_id][d],
                                weight * impulse[d],
                            )


@ti.kernel
def apply_hanging_penalty_velocity_list(
    cutoff: float,
    touched_count: int,
    touched_node_id: ti.template(),
    node: ti.template(),
    penalty_impulse: ti.template(),
):
    for touched_id, body_id in ti.ndrange(touched_count, node.shape[1]):
        node_id = touched_node_id[touched_id]
        mass = node[node_id, body_id].m
        if mass > cutoff:
            node[node_id, body_id].momentum += penalty_impulse[node_id, body_id] / mass
        penalty_impulse[node_id, body_id] = vec3f(0.0, 0.0, 0.0)


@ti.kernel
def apply_hanging_penalty_impulse_list(
    cutoff: float,
    dt: ti.template(),
    touched_count: int,
    touched_node_id: ti.template(),
    node: ti.template(),
    penalty_impulse: ti.template(),
):
    for touched_id, body_id in ti.ndrange(touched_count, node.shape[1]):
        node_id = touched_node_id[touched_id]
        mass = node[node_id, body_id].m
        if mass > cutoff:
            velocity_correction = penalty_impulse[node_id, body_id] / mass
            node[node_id, body_id].momentum += velocity_correction
            node[node_id, body_id].force += velocity_correction / dt[None]
        penalty_impulse[node_id, body_id] = vec3f(0.0, 0.0, 0.0)


@ti.kernel
def apply_hanging_penalty_velocity(cutoff: float, node: ti.template(), penalty_impulse: ti.template()):
    for node_id, body_id in node:
        mass = node[node_id, body_id].m
        if mass > cutoff:
            node[node_id, body_id].momentum += penalty_impulse[node_id, body_id] / mass


@ti.kernel
def apply_hanging_penalty_impulse(
    cutoff: float, dt: ti.template(), node: ti.template(), penalty_impulse: ti.template()
):
    for node_id, body_id in node:
        mass = node[node_id, body_id].m
        if mass > cutoff:
            velocity_correction = penalty_impulse[node_id, body_id] / mass
            node[node_id, body_id].momentum += velocity_correction
            node[node_id, body_id].force += velocity_correction / dt[None]
