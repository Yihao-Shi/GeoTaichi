import taichi as ti


from src.utils.constants import ZEROMAT2X2, ZEROVEC2F, THRESHOLD
from src.utils.ShapeFunctions import local_linear_shapefn
from src.utils.TypeDefinition import vec2f, vec2i, vec2u8
from src.utils.ScalarFunction import linearize2d, vectorize_id, linearize
import src.utils.GlobalVariable as GlobalVariable


@ti.func
def set_connectivity(cell_id, gnum, node_connectivity):
    xgrid = gnum[0]

    node_connectivity[cell_id][0] = cell_id
    node_connectivity[cell_id][1] = cell_id + 1
    node_connectivity[cell_id][2] = cell_id + xgrid + 1
    node_connectivity[cell_id][3] = cell_id + xgrid


# ========================================================= #
#                  Get Node ID & Index                      #
# ========================================================= #
@ti.kernel
def set_particle_characteristic_length_cpdi(
    particle_num: int, cal_length: ti.template(), particle: ti.template(), psize: ti.types.ndarray()
):
    for np in range(particle_num):
        psize = psize[np]
        cal_length[np]._set(vec2f(psize[np, 0], 0), vec2f(0, psize[np, 1], 0))


@ti.kernel
def set_particle_characteristic_length_gimp(
    particle_num: int, factor: float, cal_length: ti.template(), particle: ti.template(), psize: ti.types.ndarray()
):
    for np in range(particle_num):
        body_id = particle[np].bodyID
        cal_length[body_id] = factor * vec2f(psize[np, 0], psize[np, 1])


@ti.kernel
def set_node_position(position: ti.template(), gnum: ti.types.vector(2, int), grid_size: ti.types.vector(2, float)):
    for ng in position:
        # TODO: sorted by morton code
        ig, jg = get_node_index(ng, gnum)
        pos = vec2f([ig, jg]) * grid_size
        position[ng] = pos


@ti.func
def get_node_index(ng, gnum):
    ig = ng % gnum[0]
    jg = ng // gnum[0]
    return ig, jg


@ti.func
def get_node_id(i, j, gnum):
    return int(i + j * gnum[0])


@ti.kernel
def find_nodes_per_element(
    current_offset: int,
    cell_sum: int,
    gnum: ti.types.vector(2, int),
    node_connectivity: ti.template(),
    set_connectivity: ti.template(),
):
    for nc in range(current_offset, current_offset + cell_sum):
        set_connectivity(nc, gnum, node_connectivity)


@ti.func
def calc_base_cell(ielement_size, particle_size, position):
    return ti.floor((position - particle_size) * ielement_size, int)


@ti.func
def calc_natural_size(ielement_size, local_element_size, particle_size):
    return particle_size * ielement_size * local_element_size


@ti.func
def calc_natural_position(start_natural_coords, position, base_bound, ielement_size, local_element_size):
    #     4 *_  _ _ _ _ _* 3
    #       |            |
    #       |            |
    #       |            |
    #       |            |
    #       |            |
    #       *_ _ _ _ _ _ *
    #     0               1
    return start_natural_coords + (position * ielement_size - base_bound) * local_element_size


@ti.func
def shapefn(
    natural_particle_position, natural_coords, ielement_size, natural_particle_size, shape_function: ti.template()
):
    shapen0 = shape_function(
        natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0]
    )
    shapen1 = shape_function(
        natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1]
    )
    return shapen0, shapen1


@ti.func
def shapefn2d_axisy(
    natural_particle_position,
    natural_coords,
    ielement_size,
    natural_particle_size,
    shape_function_r: ti.template(),
    shape_function_z: ti.template(),
):
    shapen0 = shape_function_r(
        natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0]
    )
    shapen1 = shape_function_z(
        natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1]
    )
    return shapen0, shapen1


@ti.func
def grad_shapefn(
    natural_particle_position, natural_coords, ielement_size, natural_particle_size, grad_shape_function: ti.template()
):
    dshapen0 = grad_shape_function(
        natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0]
    )
    dshapen1 = grad_shape_function(
        natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1]
    )
    return dshapen0, dshapen1


@ti.func
def grad_shapefn_thb(
    natural_particle_position, natural_coords, element_size, nlevel, ntype, grad_shape_function: ti.template()
):
    dshapen0 = grad_shape_function(
        natural_particle_position[0], natural_coords[0], element_size[0], nlevel[0], ntype[0]
    )
    dshapen1 = grad_shape_function(
        natural_particle_position[1], natural_coords[1], element_size[1], nlevel[1], ntype[1]
    )
    return dshapen0, dshapen1


@ti.func
def grad_shapefn2d_axisy(
    natural_particle_position,
    natural_coords,
    ielement_size,
    natural_particle_size,
    grad_shape_function_r: ti.template(),
    grad_shape_function_z: ti.template(),
):
    dshapen0 = grad_shape_function_r(
        natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0]
    )
    dshapen1 = grad_shape_function_z(
        natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1]
    )
    return dshapen0, dshapen1


@ti.func
def shapefnc(
    natural_particle_position,
    natural_coords,
    ielement_size,
    natural_particle_size,
    shape_function_center: ti.template(),
):
    shapen0 = shape_function_center(
        natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0]
    )
    shapen1 = shape_function_center(
        natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1]
    )
    return shapen0, shapen1


@ti.func
def shapefnc2d_axisy(
    natural_particle_position,
    natural_coords,
    ielement_size,
    natural_particle_size,
    shape_function_center_r: ti.template(),
    shape_function_center_z: ti.template(),
):
    shapen0 = shape_function_center_r(
        natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0]
    )
    shapen1 = shape_function_center_z(
        natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1]
    )
    return shapen0, shapen1


@ti.func
def get_jacobian(nodal_coords, local_grad_shapefn, jacobian: ti.template()):
    jacobian[0, 0] += local_grad_shapefn[0] * nodal_coords[0]
    jacobian[0, 1] += local_grad_shapefn[0] * nodal_coords[1]
    jacobian[1, 0] += local_grad_shapefn[1] * nodal_coords[0]
    jacobian[1, 1] += local_grad_shapefn[1] * nodal_coords[1]


@ti.func
def transform_local_gshape_to_global(local_grad_shapefn, jacobian):
    grad_sf = ZEROVEC2F
    grad_sf[0] = local_grad_shapefn[0] * jacobian[0, 0] + local_grad_shapefn[1] * jacobian[0, 1]
    grad_sf[1] = local_grad_shapefn[1] * jacobian[1, 0] + local_grad_shapefn[1] * jacobian[1, 1]
    return grad_sf


@ti.func
def dn(offset, shape0, shape1, local_shapefn):
    local_shapefn[offset] = shape0 * shape1


@ti.func
def dn_dx(offset, shapen0, shapen1, dshapen0, dshapen1, local_grad_shapefn):
    local_grad_shapefn[offset][0] = dshapen0 * shapen1
    local_grad_shapefn[offset][1] = dshapen1 * shapen0


@ti.func
def assemble(np, nodes, jacobian, dshape_fn, node_size):
    for offset in range(np * nodes, int(node_size[np])):
        local_grad_shapefn = dshape_fn[offset]
        grad_sf = transform_local_gshape_to_global(local_grad_shapefn, jacobian.inverse())

        dshape_fn[offset] = grad_sf


@ti.func
def assemble_bbar(np, total_nodes, jacobian, dshape_fn, dshape_fnc, node_size):
    for offset in range(np * total_nodes, int(node_size[np])):
        local_grad_shapefn = dshape_fn[offset]
        local_grad_shapefnc = dshape_fnc[offset]

        grad_sf = transform_local_gshape_to_global(local_grad_shapefn, jacobian.inverse())
        grad_sfc = transform_local_gshape_to_global(local_grad_shapefnc, jacobian.inverse())

        dshape_fn[offset] = grad_sf
        dshape_fnc[offset] = grad_sfc


@ti.func
def set_node_index(gnum, base_bound, natural_coords):
    index = base_bound + 0.5 * (natural_coords - vec2f([-1, -1]))
    return int(index[0] + index[1] * gnum[0])


# Used for Classical MPM
@ti.kernel
def update(
    total_nodes: int,
    influenced_node: int,
    element_size: ti.types.vector(2, float),
    ielement_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    start_natural_coords: ti.types.vector(2, float),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    ln_id: ti.template(),
    node_size: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
):
    local_element_size, ilocal_element_size = vec2f([2.0, 2.0]), vec2f([0.5, 0.5])
    for np in range(particle_num):
        position, psize = particle[np].x, cal_length[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        natural_particle_position = calc_natural_position(
            start_natural_coords, position, base_bound, ielement_size, local_element_size
        )
        natural_particle_size = calc_natural_size(ielement_size, local_element_size, psize)
        jacobian = ZEROMAT2X2
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            jp = j
            if ti.static(GlobalVariable.MPMYPBC):
                jp = j % gnum[1] if j > 0 else (j - 1) % gnum[1]
            else:
                if j < 0 or j >= gnum[1]:
                    continue
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                ip = i
                if ti.static(GlobalVariable.MPMXPBC):
                    ip = i % gnum[0] if i > 0 else (i - 1) % gnum[0]
                else:
                    if i < 0 or i >= gnum[0]:
                        continue

                natural_coords = start_natural_coords + vec2i([ip, jp]) * local_element_size
                linear_index = set_node_index(gnum, base_bound, natural_coords)
                linear_id = linearize2d(linear_index[0], linear_index[1], gnum)
                nodal_coords = linear_index * element_size
                shapen0, shapen1 = shapefn(
                    natural_particle_position,
                    natural_coords,
                    ilocal_element_size,
                    natural_particle_size,
                    shape_function,
                )
                shapeval = shapen0 * shapen1
                if shapeval > THRESHOLD:
                    dshapen0, dshapen1 = grad_shapefn(
                        natural_particle_position,
                        natural_coords,
                        ilocal_element_size,
                        natural_particle_size,
                        grad_shape_function,
                    )
                    local_grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                    ln_id[active_id] = linear_id
                    shape_fn[active_id] = shapeval
                    dshape_fn[active_id] = local_grad_shapeval
                    get_jacobian(nodal_coords, local_grad_shapeval, jacobian)
                    active_id += 1
        node_size[np] = ti.u8(active_id - np * total_nodes)
        assemble(np, total_nodes, jacobian, dshape_fn, node_size)


# Used for Anti-Locking Classical MPM
@ti.kernel
def updatebbar(
    total_nodes: int,
    influenced_node: int,
    element_size: ti.types.vector(2, float),
    ielement_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    start_natural_coords: ti.types.vector(2, float),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    ln_id: ti.template(),
    node_size: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    dshape_fnc: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
    shape_function_center: ti.template(),
):
    local_element_size, ilocal_element_size = vec2f([2.0, 2.0]), vec2f([0.5, 0.5])
    for np in range(particle_num):
        position, psize = particle[np].x, cal_length[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        natural_particle_position = calc_natural_position(
            start_natural_coords, position, base_bound, ielement_size, local_element_size
        )
        natural_particle_size = calc_natural_size(ielement_size, local_element_size, psize)
        jacobian = ZEROMAT2X2
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            jp = j
            if ti.static(GlobalVariable.MPMYPBC):
                jp = j % gnum[1] if j > 0 else (j - 1) % gnum[1]
            else:
                if j < 0 or j >= gnum[1]:
                    continue
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                ip = i
                if ti.static(GlobalVariable.MPMXPBC):
                    ip = i % gnum[0] if i > 0 else (i - 1) % gnum[0]
                else:
                    if i < 0 or i >= gnum[0]:
                        continue

                natural_coords = start_natural_coords + vec2i([ip, jp]) * local_element_size
                linear_index = set_node_index(gnum, base_bound, natural_coords)
                linear_id = linearize2d(linear_index[0], linear_index[1], gnum)
                nodal_coords = linear_index * element_size
                shapen0, shapen1 = shapefn(
                    natural_particle_position,
                    natural_coords,
                    ilocal_element_size,
                    natural_particle_size,
                    shape_function,
                )
                shapeval = shapen0 * shapen1
                if shapeval > THRESHOLD:
                    dshapen0, dshapen1 = grad_shapefn(
                        natural_particle_position,
                        natural_coords,
                        ilocal_element_size,
                        natural_particle_size,
                        grad_shape_function,
                    )
                    shapenc0, shapenc1 = shapefnc(
                        natural_particle_position,
                        natural_coords,
                        ilocal_element_size,
                        natural_particle_size,
                        shape_function_center,
                    )
                    local_grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                    local_grad_shapevalc = vec2f([dshapen0 * shapenc1, shapenc0 * dshapen1])
                    ln_id[active_id] = linear_id
                    shape_fn[active_id] = shapeval
                    dshape_fn[active_id] = local_grad_shapeval
                    dshape_fnc[active_id] = local_grad_shapevalc
                    get_jacobian(nodal_coords, local_grad_shapeval, jacobian)
                    active_id += 1
        node_size[np] = ti.u8(active_id - np * total_nodes)
        assemble_bbar(np, total_nodes, jacobian, dshape_fn, dshape_fnc, node_size)


@ti.func
def assemble_shape(
    particle_position,
    node_position,
    element_size,
    natural_particle_size,
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
):
    shapen0, shapen1 = shapefn(particle_position, node_position, element_size, natural_particle_size, shape_function)
    dshapen0, dshapen1 = grad_shapefn(
        particle_position, node_position, element_size, natural_particle_size, grad_shape_function
    )

    shape = shapen0 * shapen1
    grad_shape = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
    return shape, grad_shape


@ti.kernel
def global_update(
    total_nodes: int,
    influenced_node: int,
    element_size: ti.types.vector(2, float),
    ielement_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    node_size: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
):
    for np in range(particle_num):
        position, psize = particle[np].x, cal_length[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            jp = j
            if ti.static(GlobalVariable.MPMYPBC):
                jp = j % gnum[1] if j > 0 else (j - 1) % gnum[1]
            else:
                if j < 0 or j >= gnum[1]:
                    continue
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                ip = i
                if ti.static(GlobalVariable.MPMXPBC):
                    ip = i % gnum[0] if i > 0 else (i - 1) % gnum[0]
                else:
                    if i < 0 or i >= gnum[0]:
                        continue
                node_id = int(ip + jp * gnum[0])
                node_coords = vec2i(ip, jp) * element_size
                shapen0, shapen1 = shapefn(particle[np].x, node_coords, ielement_size, psize, shape_function)
                shapeval = shapen0 * shapen1
                if shapeval > THRESHOLD:
                    dshapen0, dshapen1 = grad_shapefn(
                        particle[np].x, node_coords, ielement_size, psize, grad_shape_function
                    )
                    grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                    ln_id[active_id] = node_id
                    shape_fn[active_id] = shapeval
                    dshape_fn[active_id] = grad_shapeval
                    active_id += 1
        node_size[np] = ti.u8(active_id - np * total_nodes)


@ti.kernel
def global_update_thb(
    elem_nb_inf_node: ti.template(),
    elem_influen_node: ti.template(),
    element_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    n_level: int,
    elem_child_elem: ti.template(),
    nodal_coords: ti.template(),
    nlevel: ti.template(),
    ntype: ti.template(),
    node_size: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
):
    total_nodes = ti.static(elem_influen_node.shape[1])
    for np in range(particle_num):
        position = particle[np].x
        px = position[0]
        py = position[1]
        row_start = np * total_nodes
        row_end = row_start + total_nodes
        active_id = row_start
        # InWhichCell
        ix = int(px / element_size[0]) + 1
        iy = int(py / element_size[1]) + 1
        in_which_cell = (iy - 1) * (gnum[0] - 1) + ix
        center_x = (float(ix) - 0.5) * element_size[0]
        center_y = (float(iy) - 0.5) * element_size[1]
        valid_cell = 1
        if in_which_cell < 1 or in_which_cell > elem_child_elem.shape[0]:
            valid_cell = 0
        if valid_cell == 1:
            for l in range(1, n_level + 1):
                if in_which_cell >= 1 and in_which_cell <= elem_child_elem.shape[0]:
                    e = elem_child_elem[in_which_cell - 1, l - 1]
                    if e <= 0:
                        break
                    dis_x = px - center_x
                    dis_y = py - center_y
                    sub_len = element_size[0] / (2**l)
                    if dis_x <= 0 and dis_y <= 0:
                        center_x -= sub_len * 0.5
                        center_y -= sub_len * 0.5
                    elif dis_x >= 0 and dis_y <= 0:
                        in_which_cell = e
                        center_x += sub_len * 0.5
                        center_y -= sub_len * 0.5
                    elif dis_x >= 0 and dis_y >= 0:
                        in_which_cell = e + 1
                        center_x += sub_len * 0.5
                        center_y += sub_len * 0.5
                    elif dis_x <= 0 and dis_y >= 0:
                        in_which_cell = e + 2
                        center_x -= sub_len * 0.5
                        center_y += sub_len * 0.5
                else:
                    valid_cell = 0
        # calculate shape function
        if valid_cell == 1 and in_which_cell >= 1 and in_which_cell <= elem_nb_inf_node.shape[0]:
            for i in range(0, elem_nb_inf_node[in_which_cell - 1]):
                if active_id < row_end:
                    node_id = elem_influen_node[in_which_cell - 1, i]
                    if node_id >= 1 and node_id <= nodal_coords.shape[0]:
                        node_x = nodal_coords[node_id - 1][0]
                        node_y = nodal_coords[node_id - 1][1]
                        level_x = nlevel[node_id - 1, 0]
                        level_y = nlevel[node_id - 1, 1]
                        type_x = ntype[node_id - 1, 0]
                        type_y = ntype[node_id - 1, 1]
                        shapen0 = shape_function(px, node_x, element_size[0], level_x, type_x)
                        shapen1 = shape_function(py, node_y, element_size[1], level_y, type_y)
                        # >----for rigid using Linear shape function
                        """if np >= 41200:   # 395600
                            shapen0 = ShapeLinear(particle[np].x[0], node_coords[0], subLen)
                            shapen1 = ShapeLinear(particle[np].x[1], node_coords[1], subLen)"""
                        # ----<
                        shapeval = shapen0 * shapen1
                        if shapeval > THRESHOLD:
                            dshapen0 = grad_shape_function(px, node_x, element_size[0], level_x, type_x)
                            dshapen1 = grad_shape_function(py, node_y, element_size[1], level_y, type_y)
                            # >----for rigid using Linear shape function
                            """if np >= 41200:
                                dshapen0 = GShapeLinear(particle[np].x[0], node_coords[0], subLen)
                                dshapen1 = GShapeLinear(particle[np].x[1], node_coords[1], subLen)"""
                            # ----<
                            ln_id[active_id] = node_id - 1
                            shape_fn[active_id] = shapeval
                            dshape_fn[active_id][0] = dshapen0 * shapen1
                            dshape_fn[active_id][1] = shapen0 * dshapen1
                            active_id += 1
        node_size[np] = ti.u8(active_id - row_start)


@ti.kernel
def global_update_2d_axisy(
    total_nodes: int,
    influenced_node: int,
    element_size: ti.types.vector(2, float),
    ielement_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    node_size: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function_r: ti.template(),
    shape_function_z: ti.template(),
    grad_shape_function_r: ti.template(),
    grad_shape_function_z: ti.template(),
):
    for np in range(particle_num):
        position, psize = particle[np].x, cal_length[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            jp = j
            if ti.static(GlobalVariable.MPMYPBC):
                jp = j % gnum[1] if j > 0 else (j - 1) % gnum[1]
            else:
                if j < 0 or j >= gnum[1]:
                    continue
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                ip = i
                if ti.static(GlobalVariable.MPMXPBC):
                    ip = i % gnum[0] if i > 0 else (i - 1) % gnum[0]
                else:
                    if i < 0 or i >= gnum[0]:
                        continue
                node_id = int(ip + jp * gnum[0])
                node_coords = vec2i(ip, jp) * element_size
                shapen0, shapen1 = shapefn2d_axisy(
                    particle[np].x, node_coords, ielement_size, psize, shape_function_r, shape_function_z
                )
                shapeval = shapen0 * shapen1
                if shapeval > THRESHOLD:
                    dshapen0, dshapen1 = grad_shapefn2d_axisy(
                        particle[np].x, node_coords, ielement_size, psize, grad_shape_function_r, grad_shape_function_z
                    )
                    grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                    ln_id[active_id] = node_id
                    shape_fn[active_id] = shapeval
                    dshape_fn[active_id] = grad_shapeval
                    active_id += 1
        node_size[np] = ti.u8(active_id - np * total_nodes)


@ti.kernel
def global_update_2d_axisy_thb(
    elem_nb_inf_node: ti.template(),
    elem_influen_node: ti.template(),
    element_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    n_level: int,
    elem_child_elem: ti.template(),
    nodal_coords: ti.template(),
    nlevel: ti.template(),
    ntype: ti.template(),
    node_size: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
):
    total_nodes = ti.static(elem_influen_node.shape[1])
    for np in range(particle_num):
        position = particle[np].x
        px = position[0]
        py = position[1]
        row_start = np * total_nodes
        row_end = row_start + total_nodes
        active_id = row_start
        # InWhichCell
        ix = int(px / element_size[0]) + 1
        iy = int(py / element_size[1]) + 1
        in_which_cell = (iy - 1) * (gnum[0] - 1) + ix
        center_x = (float(ix) - 0.5) * element_size[0]
        center_y = (float(iy) - 0.5) * element_size[1]
        valid_cell = 1
        if in_which_cell < 1 or in_which_cell > elem_child_elem.shape[0]:
            valid_cell = 0
        if valid_cell == 1:
            for l in range(1, n_level + 1):
                if in_which_cell >= 1 and in_which_cell <= elem_child_elem.shape[0]:
                    e = elem_child_elem[in_which_cell - 1, l - 1]
                    if e <= 0:
                        break
                    dis_x = px - center_x
                    dis_y = py - center_y
                    sub_len = element_size[0] / (2**l)
                    if dis_x <= 0 and dis_y <= 0:
                        center_x -= sub_len * 0.5
                        center_y -= sub_len * 0.5
                    elif dis_x >= 0 and dis_y <= 0:
                        in_which_cell = e
                        center_x += sub_len * 0.5
                        center_y -= sub_len * 0.5
                    elif dis_x >= 0 and dis_y >= 0:
                        in_which_cell = e + 1
                        center_x += sub_len * 0.5
                        center_y += sub_len * 0.5
                    elif dis_x <= 0 and dis_y >= 0:
                        in_which_cell = e + 2
                        center_x -= sub_len * 0.5
                        center_y += sub_len * 0.5
                else:
                    valid_cell = 0
        # calculate shape function
        if valid_cell == 1 and in_which_cell >= 1 and in_which_cell <= elem_nb_inf_node.shape[0]:
            for i in range(0, elem_nb_inf_node[in_which_cell - 1]):
                if active_id < row_end:
                    node_id = elem_influen_node[in_which_cell - 1, i]
                    if node_id >= 1 and node_id <= nodal_coords.shape[0]:
                        node_x = nodal_coords[node_id - 1][0]
                        node_y = nodal_coords[node_id - 1][1]
                        level_x = nlevel[node_id - 1, 0]
                        level_y = nlevel[node_id - 1, 1]
                        type_x = ntype[node_id - 1, 0]
                        type_y = ntype[node_id - 1, 1]
                        shapen0 = shape_function(px, node_x, element_size[0], level_x, type_x)
                        shapen1 = shape_function(py, node_y, element_size[1], level_y, type_y)
                        shapeval = shapen0 * shapen1
                        if shapeval > THRESHOLD:
                            dshapen0 = grad_shape_function(px, node_x, element_size[0], level_x, type_x)
                            dshapen1 = grad_shape_function(py, node_y, element_size[1], level_y, type_y)
                            ln_id[active_id] = node_id - 1
                            shape_fn[active_id] = shapeval
                            dshape_fn[active_id][0] = dshapen0 * shapen1
                            dshape_fn[active_id][1] = shapen0 * dshapen1
                            active_id += 1
        node_size[np] = ti.u8(active_id - row_start)


@ti.kernel
def global_updatebbar(
    total_nodes: int,
    influenced_node: int,
    element_size: ti.types.vector(2, float),
    ielement_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    node_size: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    dshape_fnc: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
    shape_function_center: ti.template(),
):
    for np in range(particle_num):
        position, psize = particle[np].x, cal_length[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            jp = j
            if ti.static(GlobalVariable.MPMYPBC):
                jp = j % gnum[1] if j > 0 else (j - 1) % gnum[1]
            else:
                if j < 0 or j >= gnum[1]:
                    continue
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                ip = i
                if ti.static(GlobalVariable.MPMXPBC):
                    ip = i % gnum[0] if i > 0 else (i - 1) % gnum[0]
                else:
                    if i < 0 or i >= gnum[0]:
                        continue
                node_id = int(ip + jp * gnum[0])
                node_coords = vec2i(ip, jp) * element_size
                shapen0, shapen1 = shapefn(particle[np].x, node_coords, ielement_size, psize, shape_function)
                shapeval = shapen0 * shapen1
                if shapeval > THRESHOLD:
                    dshapen0, dshapen1 = grad_shapefn(
                        particle[np].x, node_coords, ielement_size, psize, grad_shape_function
                    )
                    grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                    shapenc0, shapenc1 = shapefnc(
                        particle[np].x, node_coords, ielement_size, psize, shape_function_center
                    )
                    grad_shapevalc = vec2f([dshapen0 * shapenc1, shapenc0 * dshapen1])
                    ln_id[active_id] = node_id
                    shape_fn[active_id] = shapeval
                    dshape_fn[active_id] = grad_shapeval
                    dshape_fnc[active_id] = grad_shapevalc
                    active_id += 1
        node_size[np] = ti.u8(active_id - np * total_nodes)


@ti.kernel
def global_updatebbar_axisy(
    total_nodes: int,
    influenced_node: int,
    element_size: ti.types.vector(2, float),
    ielement_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    node_size: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    shape_fnc: ti.template(),
    dshape_fn: ti.template(),
    dshape_fnc: ti.template(),
    shape_function_r: ti.template(),
    shape_function_z: ti.template(),
    grad_shape_function_r: ti.template(),
    grad_shape_function_z: ti.template(),
    shape_function_center_r: ti.template(),
    shape_function_center_z: ti.template(),
):
    for np in range(particle_num):
        position, psize = particle[np].x, cal_length[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            jp = j
            if ti.static(GlobalVariable.MPMYPBC):
                jp = j % gnum[1] if j > 0 else (j - 1) % gnum[1]
            else:
                if j < 0 or j >= gnum[1]:
                    continue
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                ip = i
                if ti.static(GlobalVariable.MPMXPBC):
                    ip = i % gnum[0] if i > 0 else (i - 1) % gnum[0]
                else:
                    if i < 0 or i >= gnum[0]:
                        continue
                node_id = int(ip + jp * gnum[0])
                node_coords = vec2i(ip, jp) * element_size
                shapen0, shapen1 = shapefn2d_axisy(
                    particle[np].x, node_coords, ielement_size, psize, shape_function_r, shape_function_z
                )
                shapeval = shapen0 * shapen1
                if shapeval > THRESHOLD:
                    dshapen0, dshapen1 = grad_shapefn2d_axisy(
                        particle[np].x, node_coords, ielement_size, psize, grad_shape_function_r, grad_shape_function_z
                    )
                    grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                    shapenc0, shapenc1 = shapefnc2d_axisy(
                        particle[np].x,
                        node_coords,
                        ielement_size,
                        psize,
                        shape_function_center_r,
                        shape_function_center_z,
                    )
                    grad_shapevalc = vec2f([dshapen0 * shapenc1, shapenc0 * dshapen1])
                    ln_id[active_id] = node_id
                    shape_fn[active_id] = shapeval
                    shape_fnc[active_id] = shapenc0 * shapenc1
                    dshape_fn[active_id] = grad_shapeval
                    dshape_fnc[active_id] = grad_shapevalc
                    active_id += 1
        node_size[np] = ti.u8(active_id - np * total_nodes)


@ti.kernel
def global_update_spline(
    total_nodes: int,
    influenced_node: int,
    element_size: ti.types.vector(2, float),
    ielement_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    node_size: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
    boundtype: ti.template(),
):
    for np in range(particle_num):
        body_id = int(particle[np].bodyID)
        position, psize = particle[np].x, cal_length[body_id]
        base_bound = calc_base_cell(ielement_size, psize, position)
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            jp = j
            if ti.static(GlobalVariable.MPMYPBC):
                jp = j % gnum[1] if j > 0 else (j - 1) % gnum[1]
            else:
                if j < 0 or j >= gnum[1]:
                    continue
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                ip = i
                if ti.static(GlobalVariable.MPMXPBC):
                    ip = i % gnum[0] if i > 0 else (i - 1) % gnum[0]
                else:
                    if i < 0 or i >= gnum[0]:
                        continue
                node_id = int(ip + jp * gnum[0])
                node_coords = vec2i(ip, jp) * element_size
                btype = boundtype[node_id, body_id]
                if ti.static(GlobalVariable.MPMXPBC):
                    btype[0] = 0
                if ti.static(GlobalVariable.MPMYPBC):
                    btype[1] = 0
                shapen0, shapen1 = shapefn(particle[np].x, node_coords, ielement_size, btype, shape_function)
                shapeval = shapen0 * shapen1
                if shapeval > THRESHOLD:
                    dshapen0, dshapen1 = grad_shapefn(
                        particle[np].x, node_coords, ielement_size, btype, grad_shape_function
                    )
                    grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                    ln_id[active_id] = node_id
                    shape_fn[active_id] = shapeval
                    dshape_fn[active_id] = grad_shapeval
                    active_id += 1
        node_size[np] = ti.u8(active_id - np * total_nodes)


@ti.kernel
def global_update_spline_fn(
    total_nodes: int,
    influenced_node: int,
    element_size: ti.types.vector(3, float),
    ielement_size: ti.types.vector(3, float),
    gnum: ti.types.vector(3, int),
    particle_num: int,
    particle: ti.template(),
    cal_length: ti.template(),
    node_size: ti.template(),
    ln_id: ti.template(),
    shape_fn: ti.template(),
    shape_function: ti.template(),
    boundtype: ti.template(),
):
    for np in range(particle_num):
        body_id = int(particle[np].bodyID)
        position, psize = particle[np].x, cal_length[body_id]
        base_bound = calc_base_cell(ielement_size, psize, position)
        active_id = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            jp = j
            if ti.static(GlobalVariable.MPMYPBC):
                jp = j % gnum[1] if j > 0 else (j - 1) % gnum[1]
            else:
                if j < 0 or j >= gnum[1]:
                    continue
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                ip = i
                if ti.static(GlobalVariable.MPMXPBC):
                    ip = i % gnum[0] if i > 0 else (i - 1) % gnum[0]
                else:
                    if i < 0 or i >= gnum[0]:
                        continue
                node_id = int(ip + jp * gnum[0])
                node_coords = vec2i(ip, jp) * element_size
                btype = boundtype[node_id, body_id]
                if ti.static(GlobalVariable.MPMXPBC):
                    btype[0] = 0
                if ti.static(GlobalVariable.MPMYPBC):
                    btype[1] = 0
                shapen0, shapen1 = shapefn(particle[np].x, node_coords, ielement_size, btype, shape_function)
                shapeval = shapen0 * shapen1
                if shapeval > THRESHOLD:
                    ln_id[active_id] = node_id
                    shape_fn[active_id] = shapeval
                    active_id += 1
        node_size[np] = ti.u8(active_id - np * total_nodes)


@ti.func
def transform_local_to_global(cell_id, node_connectivity, nodal_coords, guass_position):
    shape = local_linear_shapefn(guass_position)
    return (
        shape[0] * nodal_coords[node_connectivity[cell_id][0]]
        + shape[1] * nodal_coords[node_connectivity[cell_id][1]]
        + shape[2] * nodal_coords[node_connectivity[cell_id][2]]
        + shape[3] * nodal_coords[node_connectivity[cell_id][3]]
    )


@ti.kernel
def kernel_find_located_cell(
    igrid_size: ti.types.vector(2, float), grid_num: ti.types.vector(2, int), particle_num: int, particle: ti.template()
):
    for np in range(particle_num):
        index = int(particle[np].x * igrid_size)
        particle[np].cellID = index[0] + index[1] * grid_num[0]


@ti.kernel
def kernel_reset_cell_status(cell_active: ti.template()):
    ti.loop_config(bit_vectorize=True)
    for i in cell_active:
        cell_active[i] = 0


@ti.kernel
def activate_cell(cell: ti.template()):
    for i, j in cell:
        cell[i, j].active = ti.u8(1)


@ti.func
def get_boundary_type(grid_id, gnum, boundary_1, boundary1):
    btype = 0
    if grid_id - 1 < 0:
        btype = 1
    elif grid_id - 2 < 0:
        btype = 2
    elif grid_id + 1 >= gnum:
        btype = 4
    elif grid_id + 2 >= gnum:
        btype = 3
    if boundary_1 == -2 and grid_id < 0.5 * gnum:
        btype = 1
    elif boundary_1 == -1 and grid_id < 0.5 * gnum:
        btype = 2
    elif boundary1 == 2 and grid_id > 0.5 * gnum:
        btype = 4
    elif boundary1 == 1 and grid_id > 0.5 * gnum:
        btype = 3
    return btype


@ti.kernel
def kernel_set_boundary_type(
    grid_sum: int,
    grid_level: int,
    gnum: ti.types.vector(2, int),
    boundary_flag: ti.template(),
    boundary_type: ti.template(),
):
    ti.loop_config(bit_vectorize=True)
    for i in boundary_flag:
        if i < grid_sum * grid_level:
            ix, iy = vectorize_id(int(i % grid_sum), gnum)
            xtype = min(2, ix) - min(gnum[0] - 1 - ix, 2)
            ytype = min(2, iy) - min(gnum[1] - 1 - iy, 2)
            if xtype < 0:
                xtype += 3
            elif xtype > 0:
                xtype += 2
            if ytype < 0:
                ytype += 3
            elif ytype > 0:
                ytype += 2
            """Ind = vec2i(vectorize_id(i % gridSum, gnum))
            xtype, ytype = 0, 0
            if Ind[0] - 1 < 0:
                xtype = 1
            elif Ind[0] + 1 >= gnum[0]:
                xtype = 4
            else:
                ind_1 = int(boundary_flag[i-1])
                ind_0 = int(boundary_flag[i])
                ind1 = int(boundary_flag[i+1])
                xtype = get_boundary_type(Ind[0], gnum[0], -ind_0-ind_1, ind_0+ind1)
            if Ind[1] - 1 < 0:
                ytype = 1
            elif Ind[1] + 1 >= gnum[1]:
                ytype = 4
            else:
                ind_1 = int(boundary_flag[int(i // gridSum) * gridSum + linearize(Ind + vec2i(0, -1), gnum)])
                ind_0 = int(boundary_flag[int(i // gridSum) * gridSum + linearize(Ind + vec2i(0, 0), gnum)])
                ind1 = int(boundary_flag[int(i // gridSum) * gridSum + linearize(Ind + vec2i(0, 1), gnum)])
                ytype = get_boundary_type(Ind[1], gnum[1], -ind_0-ind_1, ind_0+ind1)"""
            boundary_type[int(i % grid_sum), int(i // grid_sum)] = vec2u8(xtype, ytype)


# ---- SemiImplicit TwoPhaseSingleLayer compatibility kernels ----
@ti.kernel
def find_active_node_byvolume(
    grid_sum: int,
    node: ti.template(),
    flag: ti.template(),
    cell_volumefrac: ti.template(),
    cnum: ti.types.vector(2, int),
):
    flag.fill(1)
    volume_fraction_tolerance = 0.1
    nb = 0
    for nc in cell_volumefrac:
        if cell_volumefrac[nc] <= volume_fraction_tolerance:
            ic, jc = get_node_index(nc, cnum)
            for ix in ti.static(range(0, 2)):  # 0,1
                for iy in ti.static(range(0, 2)):  # 0,1
                    ng = ic + ix + (jc + iy) * (cnum[0] + 1)
                    flag[ng + nb * grid_sum] = 0


@ti.kernel
def set_active_dofs_poisson(
    grid_sum: int,
    cutoff: float,
    node: ti.template(),
    flag: ti.template(),
    is_rigid: ti.template(),
    num_pressusre: ti.template(),
    pressure_list: ti.template(),
) -> int:
    ti.loop_config(serialize=True)
    dofs = -1
    nps = -1
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            node[ng, nb]._set_dofs(-1)
            if is_rigid[nb] == 0:
                if flag[ng + nb * grid_sum] == 1:
                    dofs += 1
                    node[ng, nb]._set_dofs(dofs)
                elif flag[ng + nb * grid_sum] == 0:
                    dofs += 1
                    node[ng, nb]._set_dofs(dofs)
                    nps += 1
                    pressure_list[nps] = ng + nb * grid_sum
                elif flag[ng + nb * grid_sum] == -1:
                    node[ng, nb]._set_dofs(-1)
                # print(ng, node[ng, nb].dof)
    num_pressusre[None] = nps + 1
    return dofs + 1


@ti.kernel
def set_active_dofs_poisson_1(
    grid_sum: int,
    cutoff: float,
    node: ti.template(),
    flag: ti.template(),
    is_rigid: ti.template(),
    nodal_coords: ti.template(),
    pcoord: float,
) -> int:
    dofs = -1
    ti.loop_config(serialize=True)
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            node[ng, nb]._set_dofs(-1)
            if is_rigid[nb] == 0:
                if flag[ng + 0 * grid_sum] == 1 and node[ng, nb].m > cutoff:
                    dofs += 1
                    node[ng, nb]._set_dofs(dofs)
                # elif nodal_coords[ng][0] <= 1.0 and nodal_coords[ng][1] >= 9.99 and nodal_coords[ng][1] <= 10.01:
                # elif flag[ng] == 0 and nodal_coords[ng][1] <= pcoord:
                #     dofs += 1
                #     node[ng, nb]._set_dofs(dofs)
            # print(ng, node[ng, nb].dof)
    return dofs + 1


@ti.kernel
def estimate_active_dofs_poisson(cutoff: float, node: ti.template()) -> int:
    total_active_nodes = 0
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                total_active_nodes += 1
    return total_active_nodes


@ti.kernel
def update_cell_type(
    cnum: ti.types.vector(2, int),
    cell_volumefrac: ti.template(),
    cell_rigid: ti.template(),
    cell_type: ti.template(),
    axis_offset: int,
    h0: int,
):
    cell_type.fill(0)
    volume_fraction_tolerance = 0.1
    for I in ti.grouped(cell_type):
        cell_id = I[1] * cnum[0] + I[0]
        if cell_rigid[I] == 1:
            cell_type[I] = 2
            if cell_volumefrac[cell_id] > 0.01:  # temp: avoid pile tip == 0
                cell_type[I] = 1
        if cell_volumefrac[cell_id] > volume_fraction_tolerance:
            cell_type[I] = 1
        # temp: avoid abrupt panartation
        # if cell_type[I] == 1 and cell_rigid[i-1, j] == 1 and cell_rigid[i+1, j] == 1 and cell_rigid[i, j-1] == 1 and cell_rigid[i, j+1] == 1:
        #     cell_type[I] = 0
        # Axisymmetric models may reserve cells inside the radial axis offset.
        if axis_offset > 0 and I[0] < axis_offset:
            cell_type[I] = 2
