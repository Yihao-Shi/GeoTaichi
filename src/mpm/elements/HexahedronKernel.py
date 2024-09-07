import taichi as ti


from src.utils.constants import ZEROMAT3x3, ZEROVEC3f, Threshold
from src.utils.ShapeFunctions import local_linear_shapefn
from src.utils.ScalarFunction import linearize
from src.utils.TypeDefination import vec3f, vec3i

@ti.func
def set_connectivity(cell_id, gnum, node_connectivity):
    xgrid = gnum[0]
    xygrid = gnum[0] * gnum[1]

    node_connectivity[cell_id][0] = cell_id
    node_connectivity[cell_id][1] = cell_id + 1
    node_connectivity[cell_id][2] = cell_id + xgrid + 1
    node_connectivity[cell_id][3] = cell_id + xgrid
    node_connectivity[cell_id][4] = cell_id + xygrid
    node_connectivity[cell_id][5] = cell_id + xygrid + 1
    node_connectivity[cell_id][6] = cell_id + xygrid + xgrid + 1
    node_connectivity[cell_id][7] = cell_id + xygrid + xgrid

@ti.func
def get_cell_jacobian(point, coords, node_connectivity):
    node0 = coords[node_connectivity[0]]
    node1 = coords[node_connectivity[1]]
    node2 = coords[node_connectivity[2]]
    node3 = coords[node_connectivity[3]]
    node4 = coords[node_connectivity[4]]
    node5 = coords[node_connectivity[5]]
    node6 = coords[node_connectivity[6]]
    node7 = coords[node_connectivity[7]]
    return calc_jacobian(point, node0, node1, node2, node3, node4, node5, node6, node7)


@ti.func
def calc_jacobian(local_coord, node0, node1, node2, node3, node4, node5, node6, node7):
        #        6               7
        #          *_ _ _ _ _ _*
        #         /|           /|
        #        / |          / |
        #     4 *_ |_ _ _ _ _* 5|
        #       |  |         |  |
        #       |  |         |  |
        #       |  *_ _ _ _ _|_ *
        #       | / 2        | / 3
        #       |/           |/
        #       *_ _ _ _ _ _ *
        #     0               1
        xi, eta, kappa = local_coord[0], local_coord[1], local_coord[2]
        dN0dxi = -0.125 * (1 - eta) * (1 - kappa); dN0deta = -0.125 * (1 - xi) * (1 - kappa); dN0dkappa = -0.125 * (1 - xi) * (1 - eta)
        dN1dxi = 0.125 * (1 - eta) * (1 - kappa); dN1deta = -0.125 * (1 + xi) * (1 - kappa); dN1dkappa = -0.125 * (1 + xi) * (1 - eta)
        dN2dxi = -0.125 * (1 + eta) * (1 - kappa); dN2deta = 0.125 * (1 - xi) * (1 - kappa); dN2dkappa = -0.125 * (1 - xi) * (1 + eta)
        dN3dxi = 0.125 * (1 + eta) * (1 - kappa); dN3deta = 0.125 * (1 + xi) * (1 - kappa); dN3dkappa = -0.125 * (1 + xi) * (1 + eta)
        dN4dxi = -0.125 * (1 - eta) * (1 + kappa); dN4deta = -0.125 * (1 - xi) * (1 + kappa); dN4dkappa = 0.125 * (1 - xi) * (1 - eta)
        dN5dxi = 0.125 * (1 - eta) * (1 + kappa); dN5deta = -0.125 * (1 + xi) * (1 + kappa); dN5dkappa = 0.125 * (1 + xi) * (1 - eta)
        dN6dxi = -0.125 * (1 + eta) * (1 + kappa); dN6deta = 0.125 * (1 - xi) * (1 + kappa); dN6dkappa = 0.125 * (1 - xi) * (1 + eta)
        dN7dxi = 0.125 * (1 + eta) * (1 + kappa); dN7deta = 0.125 * (1 + xi) * (1 + kappa); dN7dkappa = 0.125 * (1 + xi) * (1 + eta)
        jacobian = ZEROMAT3x3
        jacobian[0, 0] = dN0dxi * node0[0] + dN1dxi * node1[0] + dN2dxi * node2[0] + dN3dxi * node3[0] + dN4dxi * node4[0] + dN5dxi * node5[0] + dN6dxi * node6[0] + dN7dxi * node7[0]
        jacobian[1, 0] = dN0dxi * node0[1] + dN1dxi * node1[1] + dN2dxi * node2[1] + dN3dxi * node3[1] + dN4dxi * node4[1] + dN5dxi * node5[1] + dN6dxi * node6[1] + dN7dxi * node7[1]
        jacobian[2, 0] = dN0dxi * node0[2] + dN1dxi * node1[2] + dN2dxi * node2[2] + dN3dxi * node3[2] + dN4dxi * node4[2] + dN5dxi * node5[2] + dN6dxi * node6[2] + dN7dxi * node7[2]
        jacobian[0, 1] = dN0deta * node1[0] + dN1deta * node1[0] + dN2deta * node2[0] + dN3deta * node3[0] + dN4deta * node4[0] + dN5deta * node5[0] + dN6deta * node6[0] + dN7deta * node7[0]
        jacobian[1, 1] = dN0deta * node1[1] + dN1deta * node1[1] + dN2deta * node2[1] + dN3deta * node3[1] + dN4deta * node4[1] + dN5deta * node5[1] + dN6deta * node6[1] + dN7deta * node7[1]
        jacobian[2, 1] = dN0deta * node1[2] + dN1deta * node1[2] + dN2deta * node2[2] + dN3deta * node3[2] + dN4deta * node4[2] + dN5deta * node5[2] + dN6deta * node6[2] + dN7deta * node7[2]
        jacobian[0, 2] = dN0dkappa * node0[0] + dN1dkappa * node1[0] + dN2dkappa * node2[0] + dN3dkappa * node3[0] + dN4dkappa * node4[0] + dN5dkappa * node5[0] + dN6dkappa * node6[0] + dN7dkappa * node7[0]
        jacobian[1, 2] = dN0dkappa * node0[1] + dN1dkappa * node1[1] + dN2dkappa * node2[1] + dN3dkappa * node3[1] + dN4dkappa * node4[1] + dN5dkappa * node5[1] + dN6dkappa * node6[1] + dN7dkappa * node7[1]
        jacobian[2, 2] = dN0dkappa * node0[2] + dN1dkappa * node1[2] + dN2dkappa * node2[2] + dN3dkappa * node3[2] + dN4dkappa * node4[2] + dN5dkappa * node5[2] + dN6dkappa * node6[2] + dN7dkappa * node7[2]
        return jacobian

# ========================================================= #
#                  Get Node ID & Index                      #
# ========================================================= # 
@ti.kernel
def set_node_position_(position: ti.template(), gnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float)):
    for ng in position:
        # TODO: sorted by morton code
        ig, jg, kg = get_node_index(ng, gnum)
        pos = vec3f([ig, jg, kg]) * grid_size
        position[ng] = pos


@ti.func
def get_node_index(ng, gnum):
    ig = (ng % (gnum[0] * gnum[1])) % gnum[0]
    jg = (ng % (gnum[0] * gnum[1])) // gnum[0]
    kg = ng // (gnum[0] * gnum[1])
    return ig, jg, kg


@ti.func
def get_node_id(i, j, k, gnum):
    return int(i + j * gnum[0] + k * gnum[0] * gnum[1])


@ti.kernel
def find_nodes_per_element_(current_offset: int, cellSum: int, gnum: ti.types.vector(3, int), 
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
    #        6               7
    #          *_ _ _ _ _ _*
    #         /|           /|
    #        / |          / |
    #     4 *_ |_ _ _ _ _* 5|
    #       |  |         |  |
    #       |  |         |  |
    #       |  *_ _ _ _ _|_ *
    #       | / 2        | / 3
    #       |/           |/
    #       *_ _ _ _ _ _ *
    #     0               1
    return start_natural_coords + (position * ielement_size - base_bound) * local_element_size

@ti.func
def shapefn(natural_particle_position, natural_coords, ielement_size, natural_particle_size, shape_function: ti.template()):
    shapen0 = shape_function(natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0])
    shapen1 = shape_function(natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1])
    shapen2 = shape_function(natural_particle_position[2], natural_coords[2], ielement_size[2], natural_particle_size[2])
    return shapen0, shapen1, shapen2

@ti.func
def grad_shapefn(natural_particle_position, natural_coords, ielement_size, natural_particle_size, grad_shape_function: ti.template()):
    dshapen0 = grad_shape_function(natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0])
    dshapen1 = grad_shape_function(natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1])
    dshapen2 = grad_shape_function(natural_particle_position[2], natural_coords[2], ielement_size[2], natural_particle_size[2])
    return dshapen0, dshapen1, dshapen2

@ti.func
def shapefnc(natural_particle_position, natural_coords, ielement_size, natural_particle_size, shape_function_center: ti.template()):
    shapen0 = shape_function_center(natural_particle_position[0], natural_coords[0], ielement_size[0], natural_particle_size[0])
    shapen1 = shape_function_center(natural_particle_position[1], natural_coords[1], ielement_size[1], natural_particle_size[1])
    shapen2 = shape_function_center(natural_particle_position[2], natural_coords[2], ielement_size[2], natural_particle_size[2])
    return shapen0, shapen1, shapen2

@ti.func
def get_jacobian(nodal_coords, local_grad_shapefn, jacobian: ti.template()):
    jacobian[0, 0] += local_grad_shapefn[0] * nodal_coords[0]
    jacobian[0, 1] += local_grad_shapefn[0] * nodal_coords[1]
    jacobian[0, 2] += local_grad_shapefn[0] * nodal_coords[2]
    jacobian[1, 0] += local_grad_shapefn[1] * nodal_coords[0]
    jacobian[1, 1] += local_grad_shapefn[1] * nodal_coords[1]
    jacobian[1, 2] += local_grad_shapefn[1] * nodal_coords[2]
    jacobian[2, 0] += local_grad_shapefn[2] * nodal_coords[0]
    jacobian[2, 1] += local_grad_shapefn[2] * nodal_coords[1]
    jacobian[2, 2] += local_grad_shapefn[2] * nodal_coords[2]

@ti.func
def transform_local_gshape_to_global(local_grad_shapefn, jacobian):
    grad_sf = ZEROVEC3f
    grad_sf[0] = local_grad_shapefn[0] * jacobian[0, 0] + local_grad_shapefn[1] * jacobian[0, 1] + local_grad_shapefn[2] * jacobian[0, 2]
    grad_sf[1] = local_grad_shapefn[1] * jacobian[1, 0] + local_grad_shapefn[1] * jacobian[1, 1] + local_grad_shapefn[2] * jacobian[1, 2]
    grad_sf[2] = local_grad_shapefn[2] * jacobian[2, 0] + local_grad_shapefn[1] * jacobian[2, 1] + local_grad_shapefn[2] * jacobian[2, 2]
    return grad_sf

@ti.func
def dn(offset, shape0, shape1, shape2, local_shapefn):
    local_shapefn[offset] = shape0 * shape1 * shape2

@ti.func
def dn_dx(offset, shapen0, shapen1, shapen2, dshapen0, dshapen1, dshapen2, local_grad_shapefn):
    local_grad_shapefn[offset][0] = dshapen0 * shapen1 * shapen2
    local_grad_shapefn[offset][1] = dshapen1 * shapen0 * shapen2
    local_grad_shapefn[offset][2] = dshapen2 * shapen0 * shapen1

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
    index = vec3i(base_bound + 0.5 * (natural_coords - vec3f([-1, -1, -1])))
    return index

# Used for Classical MPM
@ti.kernel
def update(total_nodes: int, influenced_node: int, element_size: ti.types.vector(3, float), ielement_size: ti.types.vector(3, float), gnum: ti.types.vector(3, int), start_natural_coords: ti.types.vector(3, float), particleNum: int, 
           particle: ti.template(), calLength: ti.template(), LnID: ti.template(), node_size: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), shape_function: ti.template(), grad_shape_function: ti.template()):
    local_element_size, ilocal_element_size = vec3f([2., 2., 2.]), vec3f([0.5, 0.5, 0.5])
    for np in range(particleNum): 
        position, psize = particle[np].x, calLength[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        natural_particle_position = calc_natural_position(start_natural_coords, position, base_bound, ielement_size, local_element_size)
        natural_particle_size = calc_natural_size(ielement_size, local_element_size, psize)
        jacobian = ZEROMAT3x3
        activeID = np * total_nodes
        for k, j, i in ti.ndrange(influenced_node, influenced_node, influenced_node):
            if i < 0 or i >= gnum[0]: continue
            if j < 0 or j >= gnum[1]: continue
            if k < 0 or k >= gnum[2]: continue

            natural_coords = start_natural_coords + vec3i([i, j, k]) * local_element_size
            linear_index = set_node_index(gnum, base_bound, natural_coords)
            linear_id = linearize(linear_index[0], linear_index[1], linear_index[2], gnum)
            nodal_coords = linear_index * element_size
            shapen0, shapen1, shapen2 = shapefn(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, shape_function)
            shapeval = shapen0 * shapen1 * shapen2
            if shapeval > Threshold:
                dshapen0, dshapen1, dshapen2 = grad_shapefn(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, grad_shape_function)
                local_grad_shapeval = vec3f([dshapen0 * shapen1 * shapen2, shapen0 * dshapen1 * shapen2, shapen0 * shapen1 * dshapen2])
                LnID[activeID] = linear_id
                shape_fn[activeID] = shapeval
                dshape_fn[activeID] = local_grad_shapeval
                get_jacobian(nodal_coords, local_grad_shapeval, jacobian)
                activeID += 1
        node_size[np] = ti.u8(activeID - np * total_nodes)
        assemble(np, total_nodes, jacobian, dshape_fn, node_size)

# Used for Anti-Locking Classical MPM 
@ti.kernel
def updatebbar(total_nodes: int, influenced_node: int, element_size: ti.types.vector(3, float), ielement_size: ti.types.vector(3, float), gnum: ti.types.vector(3, int), start_natural_coords: ti.types.vector(3, float), particleNum: int, particle: ti.template(), 
               calLength: ti.template(), LnID: ti.template(), node_size: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), dshape_fnc: ti.template(), shape_function: ti.template(), grad_shape_function: ti.template(), shape_function_center: ti.template()):
    local_element_size, ilocal_element_size = vec3f([2., 2., 2.]), vec3f([0.5, 0.5, 0.5])
    for np in range(particleNum):  
        position, psize = particle[np].x, calLength[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        natural_particle_position = calc_natural_position(start_natural_coords, position, base_bound, ielement_size, local_element_size)
        natural_particle_size = calc_natural_size(ielement_size, local_element_size, psize)
        jacobian = ZEROMAT3x3
        activeID = np * total_nodes
        for k, j, i in ti.ndrange(influenced_node, influenced_node, influenced_node):
            if i < 0 or i >= gnum[0]: continue
            if j < 0 or j >= gnum[1]: continue
            if k < 0 or k >= gnum[2]: continue

            natural_coords = start_natural_coords + vec3i([i, j, k]) * local_element_size
            linear_index = set_node_index(gnum, base_bound, natural_coords)
            linear_id = linearize(linear_index[0], linear_index[1], linear_index[2], gnum)
            nodal_coords = linear_index * element_size
            
            shapen0, shapen1, shapen2 = shapefn(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, shape_function)
            shapeval = shapen0 * shapen1 * shapen2
            if shapeval > Threshold:
                dshapen0, dshapen1, dshapen2 = grad_shapefn(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, grad_shape_function)
                shapenc0, shapenc1, shapenc2 = shapefnc(natural_particle_position, natural_coords, ilocal_element_size, natural_particle_size, shape_function_center)
                local_grad_shapeval = vec3f([dshapen0 * shapen1 * shapen2, shapen0 * dshapen1 * shapen2, shapen0 * shapen1 * dshapen2])
                local_grad_shapevalc = vec3f([dshapen0 * shapenc1 * shapenc2, shapenc0 * dshapen1 * shapenc2, shapenc0 * shapenc1 * dshapen2])
                LnID[activeID] = linear_id
                shape_fn[activeID] = shapeval
                dshape_fn[activeID] = local_grad_shapeval
                dshape_fnc[activeID] = local_grad_shapevalc
                get_jacobian(nodal_coords, local_grad_shapeval, jacobian)
                activeID += 1
        node_size[np] = ti.u8(activeID - np * total_nodes)
        assemble_bbar(np, total_nodes, jacobian, dshape_fn, dshape_fnc, node_size)

@ti.func
def assemble_shape(particle_position, node_position, element_size, natural_particle_size, shape_function: ti.template(), grad_shape_function: ti.template()):
    shapen0, shapen1, shapen2 = shapefn(particle_position, node_position, element_size, natural_particle_size, shape_function)
    dshapen0, dshapen1, dshapen2 = grad_shapefn(particle_position, node_position, element_size, natural_particle_size, grad_shape_function)

    shape = shapen0 * shapen1 * shapen2
    grad_shape = vec3f([dshapen0 * shapen1 * shapen2, shapen0 * dshapen1 * shapen2, shapen0 * shapen1 * dshapen2])
    return shape, grad_shape

@ti.kernel
def global_update(total_nodes: int, influenced_node: int, element_size: ti.types.vector(3, float), ielement_size: ti.types.vector(3, float), gnum: ti.types.vector(3, int), particleNum: int, particle: ti.template(), calLength: ti.template(), 
                  node_size: ti.template(), LnID: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), shape_function: ti.template(), grad_shape_function: ti.template()):
    for np in range(particleNum):
        position, psize = particle[np].x, calLength[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        activeID = np * total_nodes
        for k in range(base_bound[2], base_bound[2] + influenced_node):
            if k < 0 or k >= gnum[2]: continue
            for j in range(base_bound[1], base_bound[1] + influenced_node):
                if j < 0 or j >= gnum[1]: continue
                for i in range(base_bound[0], base_bound[0] + influenced_node):
                    if i < 0 or i >= gnum[0]: continue
                    nodeID = int(i + j * gnum[0] + k * gnum[0] * gnum[1])
                    node_coords = vec3i(i, j, k) * element_size
                    shapen0, shapen1, shapen2 = shapefn(particle[np].x, node_coords, ielement_size, psize, shape_function)
                    shapeval = shapen0 * shapen1 * shapen2
                    if shapeval > Threshold:
                        dshapen0, dshapen1, dshapen2 = grad_shapefn(particle[np].x, node_coords, ielement_size, psize, grad_shape_function)
                        grad_shapeval = vec3f([dshapen0 * shapen1 * shapen2, shapen0 * dshapen1 * shapen2, shapen0 * shapen1 * dshapen2])
                        LnID[activeID] = nodeID
                        shape_fn[activeID]=shapeval
                        dshape_fn[activeID]=grad_shapeval
                        activeID += 1
        node_size[np] = ti.u8(activeID - np * total_nodes)

@ti.kernel
def global_updatebbar(total_nodes: int, influenced_node: int, element_size: ti.types.vector(3, float), ielement_size: ti.types.vector(3, float), gnum: ti.types.vector(3, int), particleNum: int, particle: ti.template(), calLength: ti.template(),
                      node_size: ti.template(), LnID: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), dshape_fnc: ti.template(), shape_function: ti.template(), grad_shape_function: ti.template(), shape_function_center: ti.template()):
    for np in range(particleNum):
        position, psize = particle[np].x, calLength[int(particle[np].bodyID)]
        base_bound = calc_base_cell(ielement_size, psize, position)
        activeID = np * total_nodes
        for k, j, i in ti.ndrange((base_bound[2], base_bound[2] + influenced_node), (base_bound[1], base_bound[1] + influenced_node), (base_bound[0], base_bound[0] + influenced_node)):
            if i < 0 or i >= gnum[0]: continue
            if j < 0 or j >= gnum[1]: continue
            if k < 0 or k >= gnum[2]: continue

            nodeID = int(i + j * gnum[0] + k * gnum[0] * gnum[1])
            node_coords = vec3i(i, j, k) * element_size
            shapen0, shapen1, shapen2 = shapefn(particle[np].x, node_coords, ielement_size, psize, shape_function, )
            shapeval = shapen0 * shapen1 * shapen2
            if shapeval > Threshold:
                dshapen0, dshapen1, dshapen2 = grad_shapefn(particle[np].x, node_coords, ielement_size, psize, grad_shape_function, )
                grad_shapeval = vec3f([dshapen0 * shapen1 * shapen2, shapen0 * dshapen1 * shapen2, shapen0 * shapen1 * dshapen2])
                shapenc0, shapenc1, shapenc2 = shapefnc(particle[np].x, node_coords, ielement_size, psize, shape_function_center)
                grad_shapevalc = vec3f([dshapen0 * shapenc1 * shapenc2, shapenc0 * dshapen1 * shapenc2, shapenc0 * shapenc1 * dshapen2])
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
           shape[3] * nodal_coords[node_connectivity[cell_id][3]] + \
           shape[4] * nodal_coords[node_connectivity[cell_id][4]] + \
           shape[5] * nodal_coords[node_connectivity[cell_id][5]] + \
           shape[6] * nodal_coords[node_connectivity[cell_id][6]] + \
           shape[7] * nodal_coords[node_connectivity[cell_id][7]]

@ti.kernel
def kernel_find_located_cell(igrid_size: ti.types.vector(3, float), grid_num: ti.types.vector(3, int), particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        index = int(particle[np].x * igrid_size)
        particle[np].cellID = index[0] + index[1] * grid_num[0] + index[2] * grid_num[0] * grid_num[1]

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
    return 3 * total_active_nodes


@ti.kernel
def set_active_dofs(gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), active_node: ti.template()) -> int:
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dofs = flag[ng + nb * gridSum] - 1
                node[ng, nb]._set_dofs(dofs)
                active_node[dofs] = ng + node.shape[0] * nb
    return 3 * flag[flag.shape[0] - 1]