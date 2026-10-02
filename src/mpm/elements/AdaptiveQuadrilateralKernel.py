import taichi as ti
from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import (
    EquivalentDeviatoricStrain,
    EquivalentDeviatoricStress,
    EquivalentStrain,
)
from src.utils.constants import Threshold
from src.utils.ScalarFunction import linearize, vectorize_id
from src.utils.TypeDefination import vec2f, vec2i


@ti.kernel
def initialize_adaptive_particle_size_2d(
    particleNum: int,
    characteristic_mode: int,
    coarse_grid_size: ti.types.vector(2, float),
    psize: ti.types.ndarray(),
    particle_size: ti.template(),
    calLength: ti.template(),
):
    for np in range(particleNum):
        particle_size[np] = vec2f(psize[np, 0], psize[np, 1])
        calLength[np] = vec2f(0.0, 0.0)
        if characteristic_mode == 1:
            calLength[np] = particle_size[np]
        elif characteristic_mode == 2:
            calLength[np] = 0.5 * coarse_grid_size
        elif characteristic_mode == 3:
            calLength[np] = coarse_grid_size


@ti.func
def adaptive_bspline_boundary_type_2d(node_index: int, node_count: int):
    boundary_type = min(2, node_index) - min(node_count - 1 - node_index, 2)
    if boundary_type < 0:
        boundary_type += 3
    elif boundary_type > 0:
        boundary_type += 2
    return boundary_type


@ti.func
def scaled_penalty_relaxation_2d(
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
def append_or_accumulate_adaptive_support_node_2d(
    row_start: int,
    active_id: int,
    row_end: int,
    compact_id: int,
    shape_value: float,
    dshape_value: ti.types.vector(2, float),
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
def append_or_accumulate_adaptive_support_node_bbar_2d(
    row_start: int,
    active_id: int,
    row_end: int,
    compact_id: int,
    shape_value: float,
    dshape_value: ti.types.vector(2, float),
    dshape_center_value: ti.types.vector(2, float),
    support_overflow: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    dshape_fnc: ti.template(),
):
    found = 0
    for slot in range(row_start, active_id):
        if found == 0 and LnID[slot] == compact_id:
            shape_fn[slot] += shape_value
            dshape_fn[slot] += dshape_value
            dshape_fnc[slot] += dshape_center_value
            found = 1

    if found == 0:
        if compact_id >= 0 and active_id < row_end:
            LnID[active_id] = compact_id
            shape_fn[active_id] = shape_value
            dshape_fn[active_id] = dshape_value
            dshape_fnc[active_id] = dshape_center_value
            active_id += 1
        else:
            support_overflow[None] = 1
    return active_id


@ti.func
def append_projected_hanging_node_2d(
    active_id: int,
    row_start: int,
    row_end: int,
    compact_id: int,
    shape_value: float,
    dshape_value: ti.types.vector(2, float),
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
            for master in ti.static(range(4)):
                weight = hanging_master_weight[constraint_id][master]
                if weight > Threshold:
                    active_id = append_or_accumulate_adaptive_support_node_2d(
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
            active_id = append_or_accumulate_adaptive_support_node_2d(
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
        LnID[active_id] = compact_id
        shape_fn[active_id] = shape_value
        dshape_fn[active_id] = dshape_value
        active_id += 1
    return active_id


@ti.func
def append_projected_hanging_node_bbar_2d(
    active_id: int,
    row_start: int,
    row_end: int,
    compact_id: int,
    shape_value: float,
    dshape_value: ti.types.vector(2, float),
    dshape_center_value: ti.types.vector(2, float),
    strong_hanging: int,
    hanging_lookup: ti.template(),
    hanging_master_id: ti.template(),
    hanging_master_weight: ti.template(),
    support_overflow: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    dshape_fnc: ti.template(),
):
    if strong_hanging == 1:
        constraint_id = hanging_lookup[compact_id]
        if constraint_id >= 0:
            for master in ti.static(range(4)):
                weight = hanging_master_weight[constraint_id][master]
                if weight > Threshold:
                    active_id = append_or_accumulate_adaptive_support_node_bbar_2d(
                        row_start,
                        active_id,
                        row_end,
                        hanging_master_id[constraint_id][master],
                        weight * shape_value,
                        weight * dshape_value,
                        weight * dshape_center_value,
                        support_overflow,
                        LnID,
                        shape_fn,
                        dshape_fn,
                        dshape_fnc,
                    )
        else:
            active_id = append_or_accumulate_adaptive_support_node_bbar_2d(
                row_start,
                active_id,
                row_end,
                compact_id,
                shape_value,
                dshape_value,
                dshape_center_value,
                support_overflow,
                LnID,
                shape_fn,
                dshape_fn,
                dshape_fnc,
            )
    else:
        LnID[active_id] = compact_id
        shape_fn[active_id] = shape_value
        dshape_fn[active_id] = dshape_value
        dshape_fnc[active_id] = dshape_center_value
        active_id += 1
    return active_id


@ti.kernel
def adaptive_global_update_2d(
    total_nodes: int,
    influenced_node: int,
    shape_mode: int,
    strong_hanging: int,
    refinement_ratio: int,
    max_refinement_level: int,
    finest_ratio: int,
    coarse_grid_size: ti.types.vector(2, float),
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
    fine_grid_size: ti.types.vector(2, float),
    fine_igrid_size: ti.types.vector(2, float),
    fine_gnum: ti.types.vector(2, int),
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
    shape_function_r: ti.template(),
    shape_function_z: ti.template(),
    grad_shape_function_r: ti.template(),
    grad_shape_function_z: ti.template(),
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
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                level_node = vec2i(i, j)
                fine_node = level_node * node_stride
                if all(fine_node >= 0) and all(fine_node < fine_gnum):
                    node_coords = level_node * element_size
                    shape_parameter = psize
                    if shape_mode > 0:
                        shape_parameter = vec2f(
                            adaptive_bspline_boundary_type_2d(level_node[0], level_gnum[0]),
                            adaptive_bspline_boundary_type_2d(level_node[1], level_gnum[1]),
                        )
                    shapen0 = shape_function_r(
                        position[0],
                        node_coords[0],
                        ielement_size[0],
                        shape_parameter[0],
                    )
                    shapen1 = shape_function_z(
                        position[1],
                        node_coords[1],
                        ielement_size[1],
                        shape_parameter[1],
                    )
                    shapeval = shapen0 * shapen1
                    if shapeval > Threshold:
                        logical_id = linearize(fine_node, fine_gnum)
                        compact_id = logical_to_compact[logical_id]
                        assert compact_id >= 0, "AdaptiveGrid node map is incomplete"
                        dshapen0 = grad_shape_function_r(
                            position[0],
                            node_coords[0],
                            ielement_size[0],
                            shape_parameter[0],
                        )
                        dshapen1 = grad_shape_function_z(
                            position[1],
                            node_coords[1],
                            ielement_size[1],
                            shape_parameter[1],
                        )
                        activeID = append_projected_hanging_node_2d(
                            activeID,
                            row_start,
                            row_end,
                            compact_id,
                            shapeval,
                            vec2f(
                                dshapen0 * shapen1,
                                shapen0 * dshapen1,
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


@ti.kernel
def adaptive_global_update_bbar_2d(
    total_nodes: int,
    influenced_node: int,
    strong_hanging: int,
    refinement_ratio: int,
    max_refinement_level: int,
    finest_ratio: int,
    coarse_grid_size: ti.types.vector(2, float),
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
    fine_gnum: ti.types.vector(2, int),
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
    dshape_fnc: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
    shape_function_center: ti.template(),
):
    for np in range(particleNum):
        position = particle[np].x
        coarse_cell = ti.floor(position * coarse_igrid_size, int)
        coarse_cell = ti.min(ti.max(coarse_cell, 0), coarse_cnum - 1)
        coarse_cell_id = linearize(coarse_cell, coarse_cnum)
        level = int(refined_cell[coarse_cell_id])
        level = ti.min(level, max_refinement_level)
        particle_level[np] = ti.cast(level, ti.u8)

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
        base_bound = ti.floor((position - psize) * ielement_size, int)
        row_start = np * total_nodes
        row_end = (np + 1) * total_nodes
        activeID = row_start
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                level_node = vec2i(i, j)
                fine_node = level_node * node_stride
                if all(level_node >= 0) and all(level_node < level_gnum) and all(fine_node < fine_gnum):
                    node_coords = level_node * element_size
                    shapen0 = shape_function(
                        position[0],
                        node_coords[0],
                        ielement_size[0],
                        psize[0],
                    )
                    shapen1 = shape_function(
                        position[1],
                        node_coords[1],
                        ielement_size[1],
                        psize[1],
                    )
                    shapeval = shapen0 * shapen1
                    if shapeval > Threshold:
                        logical_id = linearize(fine_node, fine_gnum)
                        compact_id = logical_to_compact[logical_id]
                        assert compact_id >= 0, "AdaptiveGrid node map is incomplete"
                        dshapen0 = grad_shape_function(
                            position[0],
                            node_coords[0],
                            ielement_size[0],
                            psize[0],
                        )
                        dshapen1 = grad_shape_function(
                            position[1],
                            node_coords[1],
                            ielement_size[1],
                            psize[1],
                        )
                        shapenc0 = shape_function_center(
                            position[0],
                            node_coords[0],
                            ielement_size[0],
                            psize[0],
                        )
                        shapenc1 = shape_function_center(
                            position[1],
                            node_coords[1],
                            ielement_size[1],
                            psize[1],
                        )
                        activeID = append_projected_hanging_node_bbar_2d(
                            activeID,
                            row_start,
                            row_end,
                            compact_id,
                            shapeval,
                            vec2f(dshapen0 * shapen1, shapen0 * dshapen1),
                            vec2f(dshapen0 * shapenc1, shapenc0 * dshapen1),
                            strong_hanging,
                            hanging_lookup,
                            hanging_master_id,
                            hanging_master_weight,
                            support_overflow,
                            LnID,
                            shape_fn,
                            dshape_fn,
                            dshape_fnc,
                        )
        node_size[np] = ti.cast(activeID - row_start, ti.u8)


@ti.kernel
def adaptive_coarse_global_update_2d(
    total_nodes: int,
    influenced_node: int,
    shape_mode: int,
    coarse_grid_size: ti.types.vector(2, float),
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_gnum: ti.types.vector(2, int),
    particleNum: int,
    particle: ti.template(),
    calLength: ti.template(),
    particle_level: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function_r: ti.template(),
    shape_function_z: ti.template(),
    grad_shape_function_r: ti.template(),
    grad_shape_function_z: ti.template(),
):
    for np in range(particleNum):
        position = particle[np].x
        particle_level[np] = ti.u8(0)
        psize = calLength[np]
        support_shift = psize
        if shape_mode == 1:
            support_shift = 0.5 * coarse_grid_size
        elif shape_mode == 2:
            support_shift = coarse_grid_size
        base_bound = ti.floor((position - support_shift) * coarse_igrid_size, int)
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                node_index = vec2i(i, j)
                if all(node_index >= 0) and all(node_index < coarse_gnum):
                    node_coords = node_index * coarse_grid_size
                    shape_parameter = psize
                    if shape_mode > 0:
                        shape_parameter = vec2f(
                            adaptive_bspline_boundary_type_2d(node_index[0], coarse_gnum[0]),
                            adaptive_bspline_boundary_type_2d(node_index[1], coarse_gnum[1]),
                        )
                    shapen0 = shape_function_r(
                        position[0],
                        node_coords[0],
                        coarse_igrid_size[0],
                        shape_parameter[0],
                    )
                    shapen1 = shape_function_z(
                        position[1],
                        node_coords[1],
                        coarse_igrid_size[1],
                        shape_parameter[1],
                    )
                    shapeval = shapen0 * shapen1
                    if shapeval > Threshold:
                        dshapen0 = grad_shape_function_r(
                            position[0],
                            node_coords[0],
                            coarse_igrid_size[0],
                            shape_parameter[0],
                        )
                        dshapen1 = grad_shape_function_z(
                            position[1],
                            node_coords[1],
                            coarse_igrid_size[1],
                            shape_parameter[1],
                        )
                        LnID[active_id] = linearize(node_index, coarse_gnum)
                        shape_fn[active_id] = shapeval
                        dshape_fn[active_id] = vec2f(
                            dshapen0 * shapen1,
                            shapen0 * dshapen1,
                        )
                        active_id += 1
        assert active_id - np * total_nodes <= total_nodes
        node_size[np] = ti.cast(active_id - np * total_nodes, ti.u8)


@ti.kernel
def adaptive_coarse_global_update_bbar_2d(
    total_nodes: int,
    influenced_node: int,
    coarse_grid_size: ti.types.vector(2, float),
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_gnum: ti.types.vector(2, int),
    particleNum: int,
    particle: ti.template(),
    calLength: ti.template(),
    particle_level: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    dshape_fnc: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
    shape_function_center: ti.template(),
):
    for np in range(particleNum):
        position = particle[np].x
        particle_level[np] = ti.u8(0)
        psize = calLength[np]
        base_bound = ti.floor((position - psize) * coarse_igrid_size, int)
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                node_index = vec2i(i, j)
                if all(node_index >= 0) and all(node_index < coarse_gnum):
                    node_coords = node_index * coarse_grid_size
                    shapen0 = shape_function(
                        position[0],
                        node_coords[0],
                        coarse_igrid_size[0],
                        psize[0],
                    )
                    shapen1 = shape_function(
                        position[1],
                        node_coords[1],
                        coarse_igrid_size[1],
                        psize[1],
                    )
                    shapeval = shapen0 * shapen1
                    if shapeval > Threshold:
                        dshapen0 = grad_shape_function(
                            position[0],
                            node_coords[0],
                            coarse_igrid_size[0],
                            psize[0],
                        )
                        dshapen1 = grad_shape_function(
                            position[1],
                            node_coords[1],
                            coarse_igrid_size[1],
                            psize[1],
                        )
                        shapenc0 = shape_function_center(
                            position[0],
                            node_coords[0],
                            coarse_igrid_size[0],
                            psize[0],
                        )
                        shapenc1 = shape_function_center(
                            position[1],
                            node_coords[1],
                            coarse_igrid_size[1],
                            psize[1],
                        )
                        LnID[active_id] = linearize(node_index, coarse_gnum)
                        shape_fn[active_id] = shapeval
                        dshape_fn[active_id] = vec2f(
                            dshapen0 * shapen1,
                            shapen0 * dshapen1,
                        )
                        dshape_fnc[active_id] = vec2f(
                            dshapen0 * shapenc1,
                            shapenc0 * dshapen1,
                        )
                        active_id += 1
        assert active_id - np * total_nodes <= total_nodes
        node_size[np] = ti.cast(active_id - np * total_nodes, ti.u8)


@ti.func
def append_bridging_support_2d(
    active_id,
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
    shape_function_r: ti.template(),
    shape_function_z: ti.template(),
    grad_shape_function_r: ti.template(),
    grad_shape_function_z: ti.template(),
):
    support_shift = psize
    if shape_mode == 1:
        support_shift = 0.5 * element_size
    elif shape_mode == 2:
        support_shift = element_size
    base_bound = ti.floor((position - support_shift) * ielement_size, int)

    for j in range(base_bound[1], base_bound[1] + influenced_node):
        for i in range(base_bound[0], base_bound[0] + influenced_node):
            level_node = vec2i(i, j)
            fine_node = level_node * node_stride
            if all(fine_node >= 0) and all(fine_node < fine_gnum):
                node_coords = level_node * element_size
                shape_parameter = psize
                if shape_mode > 0:
                    shape_parameter = vec2f(
                        adaptive_bspline_boundary_type_2d(level_node[0], level_gnum[0]),
                        adaptive_bspline_boundary_type_2d(level_node[1], level_gnum[1]),
                    )
                shapen0 = shape_function_r(
                    position[0],
                    node_coords[0],
                    ielement_size[0],
                    shape_parameter[0],
                )
                shapen1 = shape_function_z(
                    position[1],
                    node_coords[1],
                    ielement_size[1],
                    shape_parameter[1],
                )
                shapeval = shapen0 * shapen1
                if shapeval > Threshold:
                    logical_id = linearize(fine_node, fine_gnum)
                    compact_id = level_to_compact[logical_id]
                    assert compact_id >= 0, "AdaptiveGrid bridging node map is incomplete"
                    dshapen0 = grad_shape_function_r(
                        position[0],
                        node_coords[0],
                        ielement_size[0],
                        shape_parameter[0],
                    )
                    dshapen1 = grad_shape_function_z(
                        position[1],
                        node_coords[1],
                        ielement_size[1],
                        shape_parameter[1],
                    )
                    LnID[active_id] = compact_id
                    shape_fn[active_id] = weight * shapeval
                    dshape_fn[active_id] = weight * vec2f(
                        dshapen0 * shapen1,
                        shapen0 * dshapen1,
                    )
                    active_id += 1
    return active_id


@ti.kernel
def adaptive_bridging_global_update_2d(
    total_nodes: int,
    level_nodes: int,
    influenced_node: int,
    shape_mode: int,
    refinement_ratio: int,
    coarse_grid_size: ti.types.vector(2, float),
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
    fine_grid_size: ti.types.vector(2, float),
    fine_igrid_size: ti.types.vector(2, float),
    fine_gnum: ti.types.vector(2, int),
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
    shape_function_r: ti.template(),
    shape_function_z: ti.template(),
    grad_shape_function_r: ti.template(),
    grad_shape_function_z: ti.template(),
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
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2))):
                weight = 1.0
                for d in ti.static(range(2)):
                    weight *= local_coord[d] if offset[d] == 1 else 1.0 - local_coord[d]
                coarse_node = coarse_cell + offset
                alpha += weight * bridge_coarse_weight[linearize(coarse_node, coarse_cnum + 1)]
            alpha = ti.min(ti.max(alpha, 0.0), 1.0)
        bridge_alpha[np] = alpha

        active_id = np * total_nodes
        psize = calLength[np]
        if alpha > Threshold:
            active_id = append_bridging_support_2d(
                active_id,
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
                shape_function_r,
                shape_function_z,
                grad_shape_function_r,
                grad_shape_function_z,
            )
        coarse_size = active_id - np * total_nodes
        bridge_coarse_size[np] = ti.cast(coarse_size, ti.u8)

        if level == 1 and 1.0 - alpha > Threshold:
            active_id = append_bridging_support_2d(
                active_id,
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
                shape_function_r,
                shape_function_z,
                grad_shape_function_r,
                grad_shape_function_z,
            )
        assert active_id - np * total_nodes <= 2 * level_nodes
        node_size[np] = ti.cast(active_id - np * total_nodes, ti.u8)


@ti.kernel
def assemble_bridging_penalty_impulse_2d(
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
            coarse_velocity = vec2f(0.0, 0.0)
            fine_velocity = vec2f(0.0, 0.0)
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
                relaxation = scaled_penalty_relaxation_2d(
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
                    for d in ti.static(range(2)):
                        ti.atomic_add(
                            penalty_impulse[node_id, body_id][d],
                            direction * shape * impulse[d],
                        )


@ti.kernel
def mark_refined_cells_epdstrain_2d(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
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
def mark_refined_cells_epstrain_2d(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
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
def mark_refined_cells_strain_2d(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
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
def mark_refined_cells_deviatoric_strain_2d(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
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
def mark_refined_cells_stress_2d(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
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
def matrix_equivalent_deviatoric_stress_2d(stress):
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
def mark_refined_cells_matrix_stress_2d(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
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
            if (
                current < max_refinement_level
                and matrix_equivalent_deviatoric_stress_2d(particle[np].stress) >= threshold
            ):
                refined_cell[cell_id] = ti.cast(current + 1, ti.u8)


@ti.kernel
def mark_refined_cells_softening_2d(
    threshold: float,
    max_refinement_level: int,
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
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
def mark_refined_cells_migrated_particles_2d(
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
    particleNum: int,
    particle: ti.template(),
    current_refined_cell: ti.template(),
    particle_refined: ti.template(),
    refined_cell: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0 and particle_refined[np] == 1:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_id = linearize(cell, coarse_cnum)
            target_level = ti.max(int(current_refined_cell[cell_id]), int(particle_refined[np]))
            if target_level > int(current_refined_cell[cell_id]):
                refined_cell[cell_id] = ti.cast(target_level, ti.u8)


@ti.kernel
def dilate_refined_cells_2d(
    coarse_cnum: ti.types.vector(2, int), refined_cell: ti.template(), refined_cell_buffer: ti.template()
):
    for cell_id in range(refined_cell.shape[0]):
        level = int(refined_cell[cell_id])
        if level > 0:
            cell = vec2i(vectorize_id(cell_id, coarse_cnum))
            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                neighbor = cell + offset
                if all(neighbor >= 0) and all(neighbor < coarse_cnum):
                    ti.atomic_max(refined_cell_buffer[linearize(neighbor, coarse_cnum)], refined_cell[cell_id])


@ti.kernel
def merge_refined_cells_2d(refined_cell: ti.template(), refined_cell_buffer: ti.template()):
    for cell_id in range(refined_cell.shape[0]):
        if refined_cell_buffer[cell_id] > refined_cell[cell_id]:
            refined_cell[cell_id] = refined_cell_buffer[cell_id]


@ti.kernel
def count_refined_cells_2d(refined_cell: ti.template(), refined_cell_count: ti.template()):
    refined_cell_count[None] = 0
    for cell_id in range(refined_cell.shape[0]):
        if refined_cell[cell_id] > 0:
            ti.atomic_add(refined_cell_count[None], 1)


@ti.kernel
def count_new_refined_cells_2d(
    refined_cell: ti.template(), refinement_seed: ti.template(), new_refined_cell_count: ti.template()
):
    new_refined_cell_count[None] = 0
    for cell_id in range(refined_cell.shape[0]):
        if refinement_seed[cell_id] > refined_cell[cell_id]:
            ti.atomic_add(new_refined_cell_count[None], 1)


@ti.kernel
def assign_particle_levels_2d(
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
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
def count_particles_to_split_2d(
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
    particleNum: int,
    particle: ti.template(),
    refined_cell: ti.template(),
    particle_refined: ti.template(),
    unrefined_particle_count: ti.template(),
) -> int:
    split_count = 0
    unrefined_particle_count[None] = 0
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            ti.atomic_add(unrefined_particle_count[None], 1)
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            if refined_cell[linearize(cell, coarse_cnum)] > particle_refined[np]:
                split_count += 1
    return split_count


@ti.kernel
def build_particle_traction_linked_list_2d(
    particleNum: int,
    ptraction_count: int,
    particle_traction: ti.template(),
    particle_traction_head: ti.template(),
    particle_traction_next: ti.template(),
):
    for pid in range(particleNum):
        particle_traction_head[pid] = -1
    for traction_id in range(ptraction_count):
        particle_traction_next[traction_id] = -1
    ti.loop_config(serialize=True)
    for traction_id in range(ptraction_count):
        pid = particle_traction[traction_id].pid
        if 0 <= pid and pid < particleNum:
            previous = particle_traction_head[pid]
            particle_traction_head[pid] = traction_id
            particle_traction_next[traction_id] = previous


@ti.kernel
def count_particle_traction_constraints_to_split_2d(
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
    particleNum: int,
    particle: ti.template(),
    particle_traction_head: ti.template(),
    particle_traction_next: ti.template(),
    refined_cell: ti.template(),
    particle_refined: ti.template(),
) -> int:
    split_constraint_count = 0
    for pid in range(particleNum):
        if int(particle[pid].active) == 1 and int(particle[pid].materialID) > 0:
            cell = ti.floor(particle[pid].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            if refined_cell[linearize(cell, coarse_cnum)] > particle_refined[pid]:
                traction_id = particle_traction_head[pid]
                while traction_id >= 0:
                    split_constraint_count += 1
                    traction_id = particle_traction_next[traction_id]
    return split_constraint_count


@ti.func
def adaptive_split_particle_traction_psize_2d(
    use_direct_psize: ti.template(), pid: int, particle: ti.template(), particle_size: ti.template()
):
    psize = particle_size[pid]
    if ti.static(not use_direct_psize):
        psize = (
            0.25
            * particle[pid].vol
            / vec2f(
                ti.max(particle_size[pid][1], Threshold),
                ti.max(particle_size[pid][0], Threshold),
            )
        )
    return psize


@ti.kernel
def split_refined_particles_2d(
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
    particleNum: int,
    is_axisymmetric: ti.template(),
    refined_cell: ti.template(),
    particle: ti.template(),
    stateVars: ti.template(),
    particle_size: ti.template(),
    calLength: ti.template(),
    particle_level: ti.template(),
    particle_refined: ti.template(),
    new_particle_num: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_level = int(refined_cell[linearize(cell, coarse_cnum)])
            current_level = int(particle_refined[np])
            if cell_level > current_level:
                appended_start = ti.atomic_add(new_particle_num[None], 3)
                parent_position = particle[np].x
                parent_velocity = particle[np].v
                parent_velocity_gradient = particle[np].velocity_gradient
                parent_mass = particle[np].m
                parent_volume = particle[np].vol
                child_size = 0.5 * particle_size[np]
                child_cal_length = 0.5 * calLength[np]
                child_level = ti.cast(ti.min(current_level + 1, cell_level), ti.u8)

                for offset in ti.static(ti.grouped(ti.ndrange(2, 2))):
                    child_index = offset[0] + 2 * offset[1]
                    target = np
                    if child_index > 0:
                        target = appended_start + child_index - 1
                        particle[target] = particle[np]
                        stateVars[target] = stateVars[np]

                    child_offset = 0.999 * (2 * offset.cast(float) - 1.0) * child_size
                    child_position = parent_position + child_offset
                    radial_factor = 1.0
                    if ti.static(is_axisymmetric):
                        radial_factor = ti.max(child_position[0], Threshold) / ti.max(parent_position[0], Threshold)
                    particle[target].particleID = target
                    particle[target].m = 0.25 * parent_mass * radial_factor
                    particle[target].vol = 0.25 * parent_volume * radial_factor
                    particle[target].x = child_position
                    particle[target].v = parent_velocity
                    particle[target].velocity_gradient = 0.0 * parent_velocity_gradient
                    particle_size[target] = child_size
                    calLength[target] = child_cal_length
                    particle_level[target] = child_level
                    particle_refined[target] = child_level


@ti.kernel
def split_refined_particles_with_traction_2d(
    coarse_igrid_size: ti.types.vector(2, float),
    coarse_cnum: ti.types.vector(2, int),
    particleNum: int,
    ptraction_count: int,
    is_axisymmetric: ti.template(),
    use_direct_particle_traction_psize: ti.template(),
    refined_cell: ti.template(),
    particle: ti.template(),
    stateVars: ti.template(),
    particle_size: ti.template(),
    calLength: ti.template(),
    particle_level: ti.template(),
    particle_refined: ti.template(),
    particle_traction: ti.template(),
    particle_traction_head: ti.template(),
    particle_traction_next: ti.template(),
    new_particle_num: ti.template(),
    new_particle_traction_num: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            cell = ti.floor(particle[np].x * coarse_igrid_size, int)
            cell = ti.min(ti.max(cell, 0), coarse_cnum - 1)
            cell_level = int(refined_cell[linearize(cell, coarse_cnum)])
            current_level = int(particle_refined[np])
            if cell_level > current_level:
                appended_start = ti.atomic_add(new_particle_num[None], 3)
                parent_position = particle[np].x
                parent_velocity = particle[np].v
                parent_velocity_gradient = particle[np].velocity_gradient
                parent_mass = particle[np].m
                parent_volume = particle[np].vol
                child_size = 0.5 * particle_size[np]
                child_cal_length = 0.5 * calLength[np]
                child_level = ti.cast(ti.min(current_level + 1, cell_level), ti.u8)

                for offset in ti.static(ti.grouped(ti.ndrange(2, 2))):
                    child_index = offset[0] + 2 * offset[1]
                    target = np
                    if child_index > 0:
                        target = appended_start + child_index - 1
                        particle[target] = particle[np]
                        stateVars[target] = stateVars[np]

                    child_offset = 0.999 * (2 * offset.cast(float) - 1.0) * child_size
                    child_position = parent_position + child_offset
                    radial_factor = 1.0
                    if ti.static(is_axisymmetric):
                        radial_factor = ti.max(child_position[0], Threshold) / ti.max(parent_position[0], Threshold)
                    particle[target].particleID = target
                    particle[target].m = 0.25 * parent_mass * radial_factor
                    particle[target].vol = 0.25 * parent_volume * radial_factor
                    particle[target].x = child_position
                    particle[target].v = parent_velocity
                    particle[target].velocity_gradient = 0.0 * parent_velocity_gradient
                    particle_size[target] = child_size
                    calLength[target] = child_cal_length
                    particle_level[target] = child_level
                    particle_refined[target] = child_level

                traction_id = particle_traction_head[np]
                while traction_id >= 0:
                    particle_traction[traction_id].psize = adaptive_split_particle_traction_psize_2d(
                        use_direct_particle_traction_psize,
                        np,
                        particle,
                        particle_size,
                    )
                    for child_index in ti.static(range(1, 4)):
                        child_pid = appended_start + child_index - 1
                        target_traction = ti.atomic_add(new_particle_traction_num[None], 1)
                        particle_traction[target_traction] = particle_traction[traction_id]
                        particle_traction[target_traction].pid = child_pid
                        particle_traction[target_traction].psize = adaptive_split_particle_traction_psize_2d(
                            use_direct_particle_traction_psize,
                            child_pid,
                            particle,
                            particle_size,
                        )
                    traction_id = particle_traction_next[traction_id]


@ti.kernel
def reset_hanging_penalty_impulse_list_2d(
    touched_count: int, touched_node_id: ti.template(), penalty_impulse: ti.template()
):
    for touched_id, body_id in ti.ndrange(touched_count, penalty_impulse.shape[1]):
        node_id = touched_node_id[touched_id]
        penalty_impulse[node_id, body_id] = vec2f(0.0, 0.0)


@ti.kernel
def assemble_hanging_penalty_impulse_list_2d(
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
            master_velocity = vec2f(0.0, 0.0)
            inverse_effective_mass = 1.0 / slave_mass
            masters_are_active = True

            for master in ti.static(range(4)):
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
                relaxation = scaled_penalty_relaxation_2d(
                    penalty,
                    penalty_beta,
                    penalty_young,
                    penalty_length,
                    reference_volume,
                    dt,
                    inverse_effective_mass,
                )
                impulse = relaxation * mismatch / inverse_effective_mass
                for d in ti.static(range(2)):
                    ti.atomic_add(penalty_impulse[node_id, body_id][d], -impulse[d])

                for master in ti.static(range(4)):
                    weight = hanging_master_weight[constraint_id][master]
                    if weight > Threshold:
                        master_id = hanging_master_id[constraint_id][master]
                        for d in ti.static(range(2)):
                            ti.atomic_add(
                                penalty_impulse[master_id, body_id][d],
                                weight * impulse[d],
                            )


@ti.kernel
def apply_hanging_penalty_velocity_list_2d(
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
        penalty_impulse[node_id, body_id] = vec2f(0.0, 0.0)


@ti.kernel
def apply_hanging_penalty_impulse_list_2d(
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
        penalty_impulse[node_id, body_id] = vec2f(0.0, 0.0)


@ti.kernel
def apply_hanging_penalty_velocity_2d(cutoff: float, node: ti.template(), penalty_impulse: ti.template()):
    for node_id, body_id in node:
        mass = node[node_id, body_id].m
        if mass > cutoff:
            node[node_id, body_id].momentum += penalty_impulse[node_id, body_id] / mass


@ti.kernel
def apply_hanging_penalty_impulse_2d(
    cutoff: float, dt: ti.template(), node: ti.template(), penalty_impulse: ti.template()
):
    for node_id, body_id in node:
        mass = node[node_id, body_id].m
        if mass > cutoff:
            velocity_correction = penalty_impulse[node_id, body_id] / mass
            node[node_id, body_id].momentum += velocity_correction
            node[node_id, body_id].force += velocity_correction / dt[None]
