import taichi as ti


from src.utils.constants import ZEROMAT2x2, ZEROVEC2f, Threshold
from src.utils.ShapeFunctions import local_linear_shapefn
from src.utils.TypeDefination import vec2f, vec2i

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
def set_node_position_(position: ti.template(), gnum: ti.types.vector(2, int), grid_size: ti.types.vector(2, float)):
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
def find_nodes_per_element_(current_offset: int, cellSum: int, gnum: ti.types.vector(2, int), 
                            node_connectivity: ti.template(), set_connectivity: ti.template()):
    for nc in range(current_offset, current_offset + cellSum):
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
def shapefn(natural_particle_position, natural_coords, ielement_size, natural_particle_size, shape_function: ti.template()):
    shapen0 = shape_function(natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0])
    shapen1 = shape_function(natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1])
    return shapen0, shapen1

@ti.func
def grad_shapefn(natural_particle_position, natural_coords, ielement_size, natural_particle_size, grad_shape_function: ti.template()):
    dshapen0 = grad_shape_function(natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0])
    dshapen1 = grad_shape_function(natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1])
    return dshapen0, dshapen1

@ti.func
def shapefnc(natural_particle_position, natural_coords, ielement_size, natural_particle_size, shape_function_center: ti.template()):
    shapen0 = shape_function_center(natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0])
    shapen1 = shape_function_center(natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1])
    return shapen0, shapen1

@ti.func
def get_jacobian(nodal_coords, local_grad_shapefn, jacobian: ti.template()):
    jacobian[0, 0] += local_grad_shapefn[0] * nodal_coords[0]
    jacobian[0, 1] += local_grad_shapefn[0] * nodal_coords[1]
    jacobian[1, 0] += local_grad_shapefn[1] * nodal_coords[0]
    jacobian[1, 1] += local_grad_shapefn[1] * nodal_coords[1]

@ti.func
def transform_local_gshape_to_global(local_grad_shapefn, jacobian):
    grad_sf = ZEROVEC2f
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
def update(total_nodes: int, influenced_node: int, ielement_size: ti.types.vector(2, float), gnum: ti.types.vector(2, int), start_natural_coords: ti.types.vector(2, float), particleNum: int, 
           particle: ti.template(), calLength: ti.template(), nodal_coords: ti.template(), LnID: ti.template(), node_size: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), shape_function: ti.template(), grad_shape_function: ti.template()):
    local_element_size, ilocal_element_size = vec2f([2., 2.]), vec2f([0.5, 0.5])
    for np in range(particleNum): 
        position, psize = particle[np].x, calLength[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        natural_particle_position = calc_natural_position(start_natural_coords, position, base_bound, ielement_size, local_element_size)
        natural_particle_size = calc_natural_size(ielement_size, local_element_size, psize)
        jacobian = ZEROMAT2x2
        activeID = np * total_nodes
        for j, i in ti.ndrange(influenced_node, influenced_node):
            if i < 0 or i >= gnum[0]: continue
            if j < 0 or j >= gnum[1]: continue

            natural_coords = start_natural_coords + vec2i([i, j]) * local_element_size
            linear_id = set_node_index(gnum, base_bound, natural_coords)
            shapen0, shapen1 = shapefn(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, shape_function)
            shapeval = shapen0 * shapen1
            if shapeval > Threshold:
                dshapen0, dshapen1 = grad_shapefn(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, grad_shape_function)
                local_grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                LnID[activeID] = linear_id
                shape_fn[activeID] = shapeval
                dshape_fn[activeID] = local_grad_shapeval
                get_jacobian(nodal_coords[linear_id], local_grad_shapeval, jacobian)
                activeID += 1
        node_size[np] = ti.u8(activeID - np * total_nodes)
        assemble(np, total_nodes, jacobian, dshape_fn, node_size)

# Used for Anti-Locking Classical MPM 
@ti.kernel
def updatebbar(total_nodes: int, influenced_node: int, ielement_size: ti.types.vector(2, float), gnum: ti.types.vector(2, int), start_natural_coords: ti.types.vector(2, float), particleNum: int, particle: ti.template(), 
               calLength: ti.template(), nodal_coords: ti.template(), LnID: ti.template(), node_size: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), dshape_fnc: ti.template(), shape_function: ti.template(), grad_shape_function: ti.template(), shape_function_center: ti.template()):
    local_element_size, ilocal_element_size = vec2f([2., 2.]), vec2f([0.5, 0.5])
    for np in range(particleNum):  
        position, psize = particle[np].x, calLength[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        natural_particle_position = calc_natural_position(start_natural_coords, position, base_bound, ielement_size, local_element_size)
        natural_particle_size = calc_natural_size(ielement_size, local_element_size, psize)
        jacobian = ZEROMAT2x2
        activeID = np * total_nodes
        for j, i in ti.ndrange(influenced_node, influenced_node):
            if i < 0 or i >= gnum[0]: continue
            if j < 0 or j >= gnum[1]: continue

            natural_coords = start_natural_coords + vec2i([i, j]) * local_element_size
            linear_id = set_node_index(gnum, base_bound, natural_coords)
            
            shapen0, shapen1 = shapefn(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, shape_function)
            shapeval = shapen0 * shapen1 
            if shapeval > Threshold:
                dshapen0, dshapen1 = grad_shapefn(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, grad_shape_function)
                shapenc0, shapenc1 = shapefnc(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, shape_function_center)
                local_grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                local_grad_shapevalc = vec2f([dshapen0 * shapenc1, shapenc0 * dshapen1])
                LnID[activeID] = linear_id
                shape_fn[activeID] = shapeval
                dshape_fn[activeID] = local_grad_shapeval
                dshape_fnc[activeID] = local_grad_shapevalc
                get_jacobian(nodal_coords[linear_id], local_grad_shapeval, jacobian)
                activeID += 1
        node_size[np] = ti.u8(activeID - np * total_nodes)
        assemble_bbar(np, total_nodes, jacobian, dshape_fn, dshape_fnc, node_size)

@ti.func
def assemble_shape(particle_position, node_position, element_size, natural_particle_size, shape_function: ti.template(), grad_shape_function: ti.template()):
    shapen0, shapen1 = shapefn(particle_position, node_position, element_size, natural_particle_size, shape_function)
    dshapen0, dshapen1 = grad_shapefn(particle_position, node_position, element_size, natural_particle_size, grad_shape_function)

    shape = shapen0 * shapen1
    grad_shape = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
    return shape, grad_shape

@ti.kernel
def global_update(total_nodes: int, influenced_node: int, ielement_size: ti.types.vector(2, float), gnum: ti.types.vector(2, int), particleNum: int, particle: ti.template(), calLength: ti.template(), 
                  node_coords: ti.template(), node_size: ti.template(), LnID: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), shape_function: ti.template(), grad_shape_function: ti.template()):
    for np in range(particleNum):
        position, psize = particle[np].x, calLength[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        activeID = np * total_nodes
        for j in range(base_bound[1], base_bound[1] + influenced_node):
            if j < 0 or j >= gnum[1]: continue
            for i in range(base_bound[0], base_bound[0] + influenced_node):
                if i < 0 or i >= gnum[0]: continue
                nodeID = int(i + j * gnum[0])
                shapen0, shapen1 = shapefn(particle[np].x, node_coords[nodeID], ielement_size, psize, shape_function)
                shapeval = shapen0 * shapen1
                if shapeval > Threshold:
                    dshapen0, dshapen1 = grad_shapefn(particle[np].x, node_coords[nodeID], ielement_size, psize, grad_shape_function)
                    grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                    LnID[activeID] = nodeID
                    shape_fn[activeID]=shapeval
                    dshape_fn[activeID]=grad_shapeval
                    activeID += 1
        node_size[np] = ti.u8(activeID - np * total_nodes)

@ti.kernel
def global_updatebbar(total_nodes: int, influenced_node: int, ielement_size: ti.types.vector(2, float), gnum: ti.types.vector(2, int), particleNum: int, particle: ti.template(), calLength: ti.template(),
                      node_coords: ti.template(), node_size: ti.template(), LnID: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), dshape_fnc: ti.template(), shape_function: ti.template(), grad_shape_function: ti.template(), shape_function_center: ti.template()):
    for np in range(particleNum):
        position, psize = particle[np].x, calLength[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        activeID = np * total_nodes
        for j, i in ti.ndrange((base_bound[1], base_bound[1] + influenced_node), (base_bound[0], base_bound[0] + influenced_node)):
            if i < 0 or i >= gnum[0]: continue
            if j < 0 or j >= gnum[1]: continue

            nodeID = int(i + j * gnum[0])
            shapen0, shapen1 = shapefn(particle[np].x, node_coords[nodeID], ielement_size, psize, shape_function, )
            shapeval = shapen0 * shapen1
            if shapeval > Threshold:
                dshapen0, dshapen1 = grad_shapefn(particle[np].x, node_coords[nodeID], ielement_size, psize, grad_shape_function, )
                grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                shapenc0, shapenc1 = shapefnc(particle[np].x, node_coords[nodeID], ielement_size, psize, shape_function_center)
                grad_shapevalc = vec2f([dshapen0 * shapenc1, shapenc0 * dshapen1])
                LnID[activeID] = nodeID
                shape_fn[activeID]=shapeval
                dshape_fn[activeID]=grad_shapeval
                dshape_fnc[activeID]=grad_shapevalc
                activeID += 1
        node_size[np] = ti.u8(activeID - np * total_nodes)
    

@ti.func
def transform_local_to_global(cell_id, node_connectivity, nodal_coords, guass_position):
    shape = local_linear_shapefn(guass_position)
    return shape[0] * nodal_coords[node_connectivity[cell_id][0]] + \
           shape[1] * nodal_coords[node_connectivity[cell_id][1]] + \
           shape[2] * nodal_coords[node_connectivity[cell_id][2]] + \
           shape[3] * nodal_coords[node_connectivity[cell_id][3]]

@ti.kernel
def kernel_find_located_cell(igrid_size: ti.types.vector(2, float), grid_num: ti.types.vector(2, int), particleNum: int, particle: ti.template()):
    for np in range(particleNum):
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

@ti.kernel
def find_active_node(gridSum: int, cutoff: float, node: ti.template(), flag: ti.template()):
    flag.fill(0)
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                flag[ng + nb * gridSum] = 1
                
@ti.kernel
def estimate_active_dofs(cutoff: float, node: ti.template()) -> int:
    total_active_nodes = 0
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                total_active_nodes += 1
    return 2 * total_active_nodes


@ti.kernel
def set_active_dofs(gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), active_node: ti.template()) -> int:
    ti.loop_config(serialize=True)
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dofs = flag[ng + nb * gridSum] - 1
                node[ng, nb]._set_dofs(dofs)
                active_node[dofs] = ng + node.shape[0] * nb
    return 2 * flag[flag.shape[0] - 1]