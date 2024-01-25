import taichi as ti

from src.utils.MaterialKernel import MeanStress, calculate_strain_increment
from src.utils.constants import Threshold, ZEROMAT4x4, ZEROMAT6x3, ZEROVEC3f, ZEROMAT3x3, DELTA, EYE
from src.utils.MatrixFunction import truncation
from src.utils.ShapeFunctions import ShapeLinear, GShapeLinear
from src.utils.TypeDefination import vec3f, vec4f, vec6f, mat4x4, vec3i
from src.utils.VectorFunction import Normalize, outer_product, MeanValue, voigt_form, packingIndex


@ti.func
def shape_mapping(shape_fn, vars):
    return shape_fn * vars


@ti.func
def gradshape_vector_mapping(bmatrix, vecs):
    return vec6f([bmatrix[0, 0] * vecs[0] + bmatrix[0, 1] * vecs[1] + bmatrix[0, 2] * vecs[2],
                  bmatrix[1, 0] * vecs[0] + bmatrix[1, 1] * vecs[1] + bmatrix[1, 2] * vecs[2],
                  bmatrix[2, 0] * vecs[0] + bmatrix[2, 1] * vecs[1] + bmatrix[2, 2] * vecs[2],
                  bmatrix[3, 0] * vecs[0] + bmatrix[3, 1] * vecs[1],
                  bmatrix[4, 1] * vecs[1] + bmatrix[4, 2] * vecs[2],
                  bmatrix[5, 0] * vecs[0] + bmatrix[5, 2] * vecs[2]])


@ti.func
def gradshape_scalar_mapping(bmatrix, scalar):
    vec1 = bmatrix[0, 0] * scalar 
    vec2 = bmatrix[1, 1] * scalar
    vec3 = bmatrix[2, 2] * scalar
    return vec3f([vec1, vec2, vec3])


@ti.kernel
def tlgrid_reset(cutoff: float, node: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            node[ng, nb]._tlgrid_reset()


@ti.kernel
def grid_reset(cutoff: float, node: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            node[ng, nb]._grid_reset()


@ti.kernel
def extra_grid_reset(cutoff: float, extra_node: ti.template()):
    for ng, nb in extra_node:
        if extra_node[ng, nb].vol > cutoff:
            extra_node[ng, nb]._grid_reset()


@ti.kernel
def grid_mass_reset(cutoff: float, node: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            node[ng, nb].m = 0.


@ti.kernel
def grid_internal_force_reset(cutoff: float, node: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            node[ng, nb]._reset_internal_force()

    
@ti.kernel
def gauss_cell_reset(cell: ti.template(), sub_cell: ti.template()):
    for nc, nb in cell:
        cell[nc, nb]._reset()

    for nc, nb in sub_cell:
        sub_cell[nc, nb]._reset()


@ti.kernel
def contact_force_reset(particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        particle[np]._reset_contact_force()


@ti.kernel
def particle_mass_density_reset(particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        particle[np]._reset_mass_density()


@ti.func
def contact_normal(ng, bodyID1, bodyID2, node):
    norm1, norm2 = node[ng, bodyID1].gradm, node[ng, bodyID2].gradm
    temp_norm = norm1 - norm2
    norm = Normalize(temp_norm)
    return norm


# ======================================== Explicit MPM ======================================== #
# ========================================================= #
#                  Moving Least Square                      #
# ========================================================= #
@ti.func
def polynomial(position):
    return vec4f(1., position[0], position[1], position[2])

@ti.func
def iMomentMatrix(xp, xg, igrid_size, kernel_function: ti.template()):
    w = kernel_function(xp, xg, igrid_size)
    return w * mat4x4([[1, xg[0], xg[1], xg[2]],
                       [xg[0], xg[0] * xg[0], xg[0] * xg[1], xg[0] * xg[2]],
                       [xg[1], xg[1] * xg[0], xg[1] * xg[1], xg[1] * xg[2]],
                       [xg[2], xg[2] * xg[0], xg[2] * xg[1], xg[2] * xg[2]]]).inverse()


# ========================================================= #
#             Particle Momentum to Grid (P2G)               #
# ========================================================= #
@ti.kernel
def kernel_volume_p2c(cnum: ti.types.vector(3, int), inv_dx: ti.types.vector(3, float), particleNum: int, cell: ti.template(), particle: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            cellID = ti.floor(particle[np].x * inv_dx, int)
            linear_cellID = int(cellID[0] + cellID[1] * cnum[0] + cellID[2] * cnum[0] * cnum[1])
            cell[linear_cellID, bodyID]._update_cell_volume(particle[np].vol)

@ti.kernel
def kernel_mass_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_mass(nmass)

@ti.kernel
def kernel_momentum_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            velocity = particle[np].v
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * velocity)

@ti.kernel
def kernel_mass_momentum_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            velocity = particle[np].v
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_mass(nmass)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * velocity)

@ti.kernel
def kernel_external_force_p2g(total_nodes: int, gravity: ti.types.vector(3, float), particleNum: int, node: ti.template(), particle: ti.template(), 
                              LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex = particle[np]._compute_external_force(gravity)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                external_force = shape_mapping(shapefn[ln], fex)
                node[nodeID, bodyID]._update_external_force(external_force)                

@ti.kernel
def kernel_internal_force_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec3f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]])
                node[nodeID, bodyID]._update_internal_force(internal_force)

@ti.kernel
def kernel_sum_cell_stress(gauss_num: int, dx: ti.types.vector(3, float), inv_dx: ti.types.vector(3, float), cnum: ti.types.vector(3, int), particleNum: int, particle: ti.template(), cell: ti.template(), sub_cell: ti.template()):
    gauss_point_num = gauss_num * gauss_num * gauss_num
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            element_id = ti.floor(particle[np].x * inv_dx, int)
            linear_element_id = int(element_id[0] + element_id[1] * cnum[0] + element_id[2] * cnum[0] * cnum[1])
            if int(cell[linear_element_id, bodyID].active) == 1:
                volume = particle[np].vol
                sub_element_id = ti.floor((particle[np].x - element_id * dx) * inv_dx * gauss_num, int)
                sub_linear_element_id = sub_element_id[0] + sub_element_id[1] * gauss_num + sub_element_id[2] * gauss_num * gauss_num
                sub_cell[linear_element_id * gauss_point_num + sub_linear_element_id, bodyID].stress += particle[np].stress * volume
                sub_cell[linear_element_id * gauss_point_num + sub_linear_element_id, bodyID].vol += volume

@ti.kernel
def kernel_internal_force_on_gauss_point_p2g(gauss_num: int, cnum: ti.types.vector(3, int), gnum: ti.types.vector(3, int), dx: ti.types.vector(3, float), inv_dx: ti.types.vector(3, float),  
                                             node: ti.template(), cell: ti.template(), sub_cell: ti.template(), gauss_point: ti.template(), weight: ti.template()):
    gauss_point_num = gauss_num * gauss_num * gauss_num
    for nc in range(cell.shape[0]):
        for nb in range(cell.shape[1]):
            if int(cell[nc, nb].active) == 1:
                base = vec3i(packingIndex(nc, cnum))
                volume = dx[0] * dx[1] * dx[2] / gauss_point_num
                for ngp in range(gauss_point_num):
                    gp = 0.5 * dx * (gauss_point[ngp] + 1) + base * dx
                    fInt = -weight[ngp] * sub_cell[nc * gauss_point_num + ngp, nb].stress * volume
                    for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                        nx, ny, nz = base[0] + i, base[1] + j, base[2] + k
                        nodeID = nx + ny * gnum[0] + nz * gnum[0] * gnum[1]
                        sx = ShapeLinear(gp[0], nx * dx[0], inv_dx[0], 0)
                        sy = ShapeLinear(gp[1], ny * dx[1], inv_dx[1], 0)
                        sz = ShapeLinear(gp[2], nz * dx[2], inv_dx[2], 0)
                        gsx = GShapeLinear(gp[0], nx * dx[0], inv_dx[0], 0)
                        gsy = GShapeLinear(gp[1], ny * dx[1], inv_dx[1], 0)
                        gsz = GShapeLinear(gp[2], nz * dx[2], inv_dx[2], 0)
                        dshape_fn = vec3f(gsx * sy * sz, gsy * sx * sz, gsz * sx * sy)
                        internal_force = vec3f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                                dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                                dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]]) 
                        node[nodeID, nb]._update_internal_force(internal_force)
    

@ti.kernel
def kernel_internal_force_on_material_point_p2g(cnum: ti.types.vector(3, int), gnum: ti.types.vector(3, int), dx: ti.types.vector(3, float), inv_dx: ti.types.vector(3, float), particleNum: int, node: ti.template(), particle: ti.template(),cell: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            element_id = ti.floor(particle[np].x * inv_dx, int)
            linear_element_id = element_id[0] + element_id[1] * cnum[0] + element_id[2] * cnum[0] * cnum[1]
            if int(cell[linear_element_id, bodyID].active) == 0:
                fInt = particle[np]._compute_internal_force()
                position = particle[np].x
                for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                    nx, ny, nz = element_id[0] + i, element_id[1] + j, element_id[2] + k
                    nodeID = nx + ny * gnum[0] + nz * gnum[0] * gnum[1]
                    sx = ShapeLinear(position[0], nx * dx[0], inv_dx[0], 0)
                    sy = ShapeLinear(position[1], ny * dx[1], inv_dx[1], 0)
                    sz = ShapeLinear(position[2], nz * dx[2], inv_dx[2], 0)
                    gsx = GShapeLinear(position[0], nx * dx[0], inv_dx[0], 0)
                    gsy = GShapeLinear(position[1], ny * dx[1], inv_dx[1], 0)
                    gsz = GShapeLinear(position[2], nz * dx[2], inv_dx[2], 0)
                    dshape_fn = vec3f(gsx * sy * sz, gsy * sx * sz, gsz * sx * sy)
                    internal_force = vec3f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                            dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                            dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]])
                    node[nodeID, bodyID]._update_internal_force(internal_force)

@ti.kernel
def kernel_internal_force_bbar_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                temp_dshape = (dshape_fnc - dshape_fn) / 3.
                internal_force = vec3f([(dshape_fn[0] + temp_dshape[0]) * fInt[0] + temp_dshape[0] * fInt[1] + temp_dshape[0] * fInt[2] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                        temp_dshape[1] * fInt[0] + (dshape_fn[1] + temp_dshape[1]) * fInt[1] + temp_dshape[1] * fInt[2] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                        temp_dshape[2] * fInt[0] + temp_dshape[2] * fInt[1] + (dshape_fn[2] + temp_dshape[2]) * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]])
                node[nodeID, bodyID]._update_internal_force(internal_force) 

@ti.kernel
def kernel_momentum_apic_p2g(total_nodes: int, particleNum: int, node: ti.template(), nodal_coords: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            xp = particle[np].x
            mass = particle[np].m
            velocity = particle[np].v
            gradv = particle[np].velocity_gradient
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_mass(nmass)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * (velocity + gradv @ (nodal_coords[nodeID] - xp)))

@ti.kernel
def kernel_volume_p2g(total_nodes: int, particleNum: int, extra_node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            volume = particle[np].vol 
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nvol = shape_mapping(shapefn[ln], volume)
                extra_node[nodeID, bodyID]._update_nodal_volume(nvol)

@ti.kernel
def kernel_jacobian_p2g(total_nodes: int, dt: ti.template(), particleNum: int, extra_node: ti.template(), particle: ti.template(), particle_fbar: ti.template(),
                        LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            djacobian = (DELTA + dt[None] * particle[np].velocity_gradient).determinant()
            jacobian = particle[np].vol * djacobian * particle_fbar[np].jacobian
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                extra_node[nodeID, bodyID]._update_nodal_jacobian(shape_mapping(shapefn[ln], jacobian))
            particle_fbar[np].djacobian = djacobian

@ti.kernel
def kernel_pressure_p2g(gnum: ti.types.vector(3, int), igrid_size: ti.types.vector(3, float), particleNum: int, extra_node: ti.template(), particle: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            pressure = particle[np].m * MeanStress(particle[np].stress)
            position = particle[np].x
            baseID = ti.floor(position * igrid_size, int)
            fx = position * igrid_size - baseID.cast(float)
            weight = [1 - fx, fx]
            for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                girdID = baseID + vec3f(i, j, k)
                SF = weight[i][0] * weight[j][1] * weight[k][2]
                linear_grid_id = int(girdID[0] + girdID[1] * gnum[0] + girdID[2] * gnum[0] * gnum[1])
                extra_node[linear_grid_id, bodyID]._update_nodal_pressure(shape_mapping(SF, pressure))

# ========================================================= #
#                Grid Projection Operator                   #
# ========================================================= #
@ti.kernel
def kernel_find_valid_element(cell_vol: float, cell: ti.template()):
    threshold = 0.9
    for nc in range(cell.shape[0]):
        for nb in range(cell.shape[1]):
            if int(cell[nc, nb].active) == 1:
                if cell[nc, nb].volume / cell_vol > threshold:
                    cell[nc, nb].active = ti.u8(1)
                else:
                    cell[nc, nb].active = ti.u8(0)

@ti.kernel
def kernel_compute_gauss_average_stress(gauss_num: int, cut_off: float, cell: ti.template(), sub_cell: ti.template()):
    gauss_number = gauss_num * gauss_num * gauss_num
    for nc in range(sub_cell.shape[0]):
        for nb in range(sub_cell.shape[1]):
            if int(cell[nc // gauss_number, nb].active) == 1 and sub_cell[nc, nb].vol > cut_off:
                sub_cell[nc, nb].stress /= sub_cell[nc, nb].vol

@ti.kernel
def kernel_average_pressure(gauss_num: int, cell: ti.template(), sub_cell: ti.template()):
    gauss_number = gauss_num * gauss_num * gauss_num
    for nc in range(cell.shape[0]):
        for nb in range(cell.shape[1]):
            if int(cell[nc, nb].active) == 1:
                pressure = 0.
                for ngp in range(gauss_number):
                    stress = sub_cell[nc * gauss_number + ngp, nb].stress
                    pressure += (stress[0] + stress[1] + stress[2]) / 3.
                pressure /= gauss_number

                for ngp in range(gauss_number):
                    stress = sub_cell[nc * gauss_number + ngp, nb].stress
                    p = (stress[0] + stress[1] + stress[2]) / 3.
                    ave_stress = stress - (p - pressure) * EYE
                    sub_cell[nc * gauss_number + ngp, nb].stress = ave_stress

@ti.kernel
def kernel_compute_grid_velocity(cutoff: float, node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._compute_nodal_velocity()

@ti.kernel
def kernel_compute_grid_kinematic(cutoff: float, damp: float, node: ti.template(), dt: ti.template()):
    # ti.block_local(dt)
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._compute_nodal_kinematic(damp, dt) 
    
@ti.kernel
def kernel_grid_kinematic_integration(cutoff: float, node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._update_nodal_kinematic() 

@ti.kernel
def kernel_grid_kinematic_recorrect(cutoff: float, node: ti.template(), dt: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._recorrect_nodal_kinematic(dt) 

@ti.kernel
def kernel_grid_jacobian(cutoff: float, is_rigid: ti.template(), extra_node: ti.template()):
    for ng in range(extra_node.shape[0]):
        for nb in range(extra_node.shape[1]):
            if extra_node[ng, nb].vol > cutoff and is_rigid[nb] == 0:
                extra_node[ng, nb].jacobian /= extra_node[ng, nb].vol

@ti.kernel
def kernel_grid_pressure(cutoff: float, is_rigid: ti.template(), node: ti.template(), extra_node: ti.template()):
    for ng in range(extra_node.shape[0]):
        for nb in range(extra_node.shape[1]):
            if node[ng, nb].m > cutoff and is_rigid[nb] == 0:
                extra_node[ng, nb].pressure /= node[ng, nb].m

# ========================================================= #
#                 Grid to Particle (G2P)                    #
# ========================================================= #
@ti.kernel
def kernel_kinemaitc_g2p(total_nodes: int, alpha: float, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPIC, vFLIP = ZEROVEC3f, ZEROVEC3f
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                velocity = node[nodeID, bodyID].momentum
                accleration = node[nodeID, bodyID].force
                vPIC += shape_mapping(shape_fn, velocity)
                vFLIP += shape_mapping(shape_fn, accleration) * dt[None]
            particle[np]._update_particle_state(dt, alpha, vPIC, vFLIP)

@ti.kernel
def kernel_mass_g2p(total_nodes: int, cell_volume: float, node_size: ti.template(), LnID: ti.template(), shapefn: ti.template(), node: ti.template(), particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mdensity = 0.
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                mass_density = shape_mapping(shapefn[ln], node[nodeID, bodyID].m / cell_volume)
                mdensity += mass_density
            particle[np].mass_density = mdensity

@ti.kernel
def kernel_pressure_g2p(gnum: ti.types.vector(3, int), igrid_size: ti.types.vector(3, float), extra_node: ti.template(), particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            pressure = 0.
            position = particle[np].x
            stress = particle[np].stress
            baseID = ti.floor(position * igrid_size, int)
            fx = position * igrid_size - baseID.cast(float)
            weight = [1 - fx, fx]
            for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                girdID = baseID + vec3f(i, j, k)
                SF = weight[i][0] * weight[j][1] * weight[k][2]
                linear_grid_id = int(girdID[0] + girdID[1] * gnum[0] + girdID[2] * gnum[0] * gnum[1])
                pressure += shape_mapping(SF, extra_node[linear_grid_id, bodyID].pressure)
            particle[np].stress = stress - (MeanStress(stress) - pressure) * EYE

# ========================================================= #
#                 Apply Constitutive Model                  #
# ========================================================= #
@ti.kernel
def kernel_compute_stress_strain(total_nodes: int, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), 
                                 matPorps: ti.template(), stateVars: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradient = update_velocity_gradient(np, total_nodes, node, particle, LnID, dshapefn, node_size)
            previous_stress = particle[np].stress
            particle[np].vol *= matPorps[materialID].update_particle_volume(np, velocity_gradient, stateVars, dt)
            particle[np].stress = matPorps[materialID].ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
            particle[np].velocity_gradient = velocity_gradient
            particle[np].strain += calculate_strain_increment(velocity_gradient, dt)

@ti.kernel
def kernel_compute_stress_strain_bbar(total_nodes: int, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), 
                                      matPorps: ti.template(), stateVars: ti.template(), LnID: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradient, strain_rate_trace = update_velocity_gradient_bbar(np, total_nodes, node, particle, LnID, dshapefn, dshapefnc, node_size)
            previous_stress = particle[np].stress
            particle[np].vol *= matPorps[materialID].update_particle_volume_bbar(np, strain_rate_trace, stateVars, dt)
            particle[np].stress = matPorps[materialID].ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
            particle[np].velocity_gradient = velocity_gradient
            particle[np].strain += calculate_strain_increment(velocity_gradient, dt)

@ti.kernel
def kernel_compute_stress_strain_fbar(total_nodes: int, dt: ti.template(), particleNum: int, extra_node: ti.template(), particle: ti.template(), particle_fbar: ti.template(),  
                                      matPorps: ti.template(), stateVars: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            jacobian = 0.
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                jacobian += shape_mapping(shape_fn, extra_node[nodeID, bodyID].jacobian)

            velocity_gradient = particle[np].velocity_gradient
            multiplier = (jacobian / (particle_fbar[np].jacobian * particle_fbar[np].djacobian)) ** (1./3.)
            updated_ddeformation_gradient = multiplier * (DELTA + dt[None] * velocity_gradient)
            updated_velocity_gradient = (updated_ddeformation_gradient - DELTA) / dt[None]
            particle_fbar[np].jacobian = jacobian
            previous_stress = particle[np].stress
            
            particle[np].vol *= matPorps[materialID].update_particle_volume(np, velocity_gradient, stateVars, dt)
            particle[np].stress = matPorps[materialID].ComputeStress(np, previous_stress, updated_velocity_gradient, stateVars, dt)
            particle[np].velocity_gradient = updated_velocity_gradient
            particle[np].strain += calculate_strain_increment(velocity_gradient, dt)

@ti.kernel
def kernel_compute_stress_strain_apic(total_nodes: int, dt: ti.template(), particleNum: int, node: ti.template(), nodal_coords: ti.template(), particle: ti.template(), 
                                      matPorps: ti.template(), stateVars: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradient = update_velocity_gradient_apic(np, total_nodes, node, nodal_coords, particle, LnID, shapefn, node_size)
            previous_stress = particle[np].stress
            particle[np].vol *= matPorps[materialID].update_particle_volume(np, velocity_gradient, stateVars, dt)
            particle[np].stress = matPorps[materialID].ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
            particle[np].velocity_gradient = velocity_gradient
            particle[np].strain += calculate_strain_increment(velocity_gradient, dt)
            

# ========================================================= #
#                 Update velocity gradient                  #
# ========================================================= #
@ti.func
def update_velocity_gradient_apic(np, total_nodes, node, nodal_coords, particle, LnID, shapefn, node_size):
    Wp = ZEROMAT3x3
    Bp = ZEROMAT3x3
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    position = particle[np].x
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        grid_coord = nodal_coords[nodeID]
        pointer = grid_coord - position
        gv = node[nodeID, bodyID].momentum
        shape_fn = shapefn[ln]

        Wp += shape_fn * outer_product(pointer, pointer)
        Bp += shape_fn * outer_product(gv, pointer)
    return truncation(Bp @ Wp.inverse())

@ti.func
def update_displacement_gradient(np, total_nodes, node, particle, LnID, dshapefn, node_size):
    displacement_gradient = ZEROMAT3x3
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        gu = node[nodeID, bodyID].displacement
        dshape_fn = dshapefn[ln]
        displacement_gradient += outer_product(dshape_fn, gu)
    return truncation(displacement_gradient)

@ti.func
def update_velocity_gradient(np, total_nodes, node, particle, LnID, dshapefn, node_size):
    velocity_gradient = ZEROMAT3x3
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        gv = node[nodeID, bodyID].momentum
        dshape_fn = dshapefn[ln]
        velocity_gradient += outer_product(dshape_fn, gv)
    return truncation(velocity_gradient)

@ti.func
def update_velocity_gradient_bbar(np, total_nodes, node, particle, LnID, dshapefn, dshapefnc, node_size):
    velocity_gradient = ZEROMAT3x3
    strain_rate_trace = ZEROVEC3f
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        gv = node[nodeID, bodyID].momentum
        dshape_fn = dshapefn[ln]
        dshape_fnc = dshapefnc[ln]
        temp_dshape = (dshape_fnc - dshape_fn) / 3.
        
        average_bmatrix = temp_dshape[0] * gv[0] + temp_dshape[1] * gv[1] + temp_dshape[2] * gv[2]
        velocity_gradient += outer_product(dshape_fn, gv)
        velocity_gradient[0, 0] += average_bmatrix
        velocity_gradient[1, 1] += average_bmatrix
        velocity_gradient[2, 2] += average_bmatrix

        strain_rate_trace[0] += dshape_fn[0] * gv[0]
        strain_rate_trace[1] += dshape_fn[1] * gv[1]
        strain_rate_trace[2] += dshape_fn[2] * gv[2]
    return truncation(velocity_gradient), strain_rate_trace


# ========================================================= #
#                           MUSL                            #
# ========================================================= #
@ti.kernel
def kernel_reset_grid_velocity(node: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > 0.:
            node[ng, nb].momentum = ZEROVEC3f

@ti.kernel
def kernel_postmapping_kinemaitc(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], particle[np].m)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * particle[np].v)

# ========================================================= #
#                   Compute B Matrix                        #
# ========================================================= #
@ti.func
def compute_Bmatrix(GS):
    temp = ZEROMAT6x3
    temp[0, 0] = GS[0]
    temp[1, 1] = GS[1]
    temp[2, 2] = GS[2]
    temp[3, 0], temp[3, 1] = GS[1], GS[0]
    temp[4, 1], temp[4, 2] = GS[2], GS[1]
    temp[5, 0], temp[5, 2] = GS[2], GS[0]
    return temp

# ========================================================= #
#               Anti-Locking (B-Bar Method)                 #
# ========================================================= #
@ti.func
def compute_Bmatrix_bbar(GS, GSC):
    temp = ZEROMAT6x3
    temp[0, 0] = GS[0] + (GSC[0] - GS[0]) / 3.
    temp[0, 1] = (GSC[1] - GS[1]) / 3.
    temp[0, 2] = (GSC[2] - GS[2]) / 3.
    temp[1, 0] = (GSC[0] - GS[0]) / 3.
    temp[1, 1] = GS[1] + (GSC[1] - GS[1]) / 3.
    temp[1, 2] = (GSC[2] - GS[2]) / 3.
    temp[2, 0] = (GSC[0] - GS[0]) / 3.
    temp[2, 1] = (GSC[1] - GS[1]) / 3.
    temp[2, 2] = GS[2] + (GSC[2] - GS[2]) / 3.
    temp[3, 0], temp[3, 1] = GS[1], GS[0]
    temp[4, 1], temp[4, 2] = GS[2], GS[1]
    temp[5, 0], temp[5, 2] = GS[2], GS[0]
    return temp

# ========================================================= #
#                          F bar                            #
# ========================================================= #
@ti.kernel
def kernel_calc_increment_velocity(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            velocity_gradient = update_velocity_gradient(np, total_nodes, node, particle, LnID, dshapefn, node_size)
            particle[np].velocity_gradient = velocity_gradient

@ti.func
def calc_deformation_grad_rate(np, total_nodes, node, particle, LnID, dshapefn, node_size, dt):
    deformation_gradient_rate = ZEROMAT3x3
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        dshape_fn = dshapefn[ln]
        gv = node[nodeID, bodyID].momentum
        deformation_gradient_rate[0, 0] += gv[0] * dshape_fn[0] * dt[None]
        deformation_gradient_rate[0, 1] += gv[0] * dshape_fn[1] * dt[None]
        deformation_gradient_rate[0, 2] += gv[0] * dshape_fn[2] * dt[None]
        deformation_gradient_rate[1, 0] += gv[1] * dshape_fn[0] * dt[None]
        deformation_gradient_rate[1, 1] += gv[1] * dshape_fn[1] * dt[None]
        deformation_gradient_rate[1, 2] += gv[1] * dshape_fn[2] * dt[None]
        deformation_gradient_rate[2, 0] += gv[2] * dshape_fn[0] * dt[None]
        deformation_gradient_rate[2, 1] += gv[2] * dshape_fn[1] * dt[None]
        deformation_gradient_rate[2, 2] += gv[2] * dshape_fn[2] * dt[None]
    return deformation_gradient_rate


# ========================================================= #
#                 Compute Domain Gradient                   #
# ========================================================= #
@ti.kernel
def kernel_calc_contact_normal(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                grad_domain = dshapefn[ln] * particle[np].vol 
                node[LnID[ln], bodyID]._update_nodal_grad_domain(grad_domain)  

@ti.kernel
def kernel_calc_contact_displacement(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                contact_pos = shapefn[ln] * particle[np].x * particle[np].m
                node[LnID[ln], bodyID]._update_nodal_contact_pos(contact_pos)  

# ========================================================= #
#               Grid Based Contact Detection                #
# ========================================================= #
@ti.kernel
def kernel_calc_friction_contact(cut_off: float, mu: float, position_offset: ti.types.vector(3, float), dt: ti.template(), is_rigid: ti.template(), node: ti.template()):
    # ti.block_local(dt)
    for ng in range(node.shape[0]):
        bodyID1, bodyID2 = 0, 1
        m1, m2 = node[ng, bodyID1].m, node[ng, bodyID2].m
        if m1 > cut_off and m2 > cut_off:
            contact_pos1, contact_pos2 = node[ng, bodyID1].contact_pos, node[ng, bodyID2].contact_pos
            mv1, mv2 = m1 * node[ng, bodyID1].momentum, m2 * node[ng, bodyID2].momentum

            norm = ZEROVEC3f
            g_mass = 0.
            norm1, norm2 = node[ng, bodyID1].grad_domain, node[ng, bodyID2].grad_domain
            if is_rigid[bodyID1] == 0 and is_rigid[bodyID2] == 0:
                norm = Normalize(norm1 - norm2)
                g_mass = (m1 + m2) * dt[None]
            elif is_rigid[bodyID1] == 1:
                norm = Normalize(norm1)
                g_mass = m1 * dt[None]
            elif is_rigid[bodyID2] == 1:
                norm = -Normalize(norm2)
                g_mass = m2 * dt[None]

            is_penetrate = (mv1 * m2 - m1 * mv2).dot(norm)
            is_contact = (m1 * contact_pos2 - m2 * contact_pos1).dot(norm) < (m1 * m2) * MeanValue(position_offset)
            if is_penetrate > Threshold and is_contact:
                inv_gmass = 1. / g_mass
                cforce = (mv1 * m2 - m1 * mv2) * inv_gmass
                norm_force = is_penetrate * inv_gmass
                if mu > Threshold:
                    trial_ft = cforce - norm_force * norm
                    fstick = trial_ft.norm()
                    fslip = mu * ti.abs(norm_force)
                    if fslip < fstick:
                        cforce = norm_force * norm + fslip * (trial_ft / fstick)
                else:
                    cforce = norm_force * norm
                node[ng, bodyID1]._update_contact_force(-cforce)
                node[ng, bodyID2]._update_contact_force(cforce)

@ti.kernel
def kernel_assemble_contact_force(cutoff: float, dt: ti.template(), node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._contact_force_assemble(dt) 

# ========================================================= #
#                    Guass Integration                      #
# ========================================================= #
@ti.kernel
def kernel_momentum_mlsp2g(total_nodes: int, igrid_size: ti.types.vector(3, float), particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), 
                           node_size: ti.template(), nodal_coords: ti.template(), kernel_function: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            particle_coord = particle[np].x
            mass = particle[np].m
            velocity = particle[np].v

            imoment_matrix = ZEROMAT4x4
            particle_polynomial = polynomial(particle_coord)
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                grid_coords = nodal_coords[nodeID]
                imoment_matrix += iMomentMatrix(particle_coord, grid_coords, igrid_size, kernel_function)

            for ln in range(offset, offset + int(node_size[np])):
                w = kernel_function(particle_coord, grid_coords, igrid_size)
                phi = particle_polynomial @ imoment_matrix * w @ polynomial(grid_coords)
                node[nodeID, bodyID]._update_nodal_momentum(shape_mapping(phi, mass * velocity))

# ======================================== Implicit MPM ======================================== #
# ========================================================= #
#            Particle Momentum to Grid (iP2G)               #
# ========================================================= #
@ti.kernel
def kernel_mass_momentum_acceleration_ip2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            velocity = particle[np].v
            acceleration = particle[np].a
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_mass(nmass)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * velocity)
                node[nodeID, bodyID]._update_nodal_acceleration(nmass * acceleration)


# ========================================================= #
#                Grid Projection Operator                   #
# ========================================================= #
@ti.kernel
def kernel_compute_grid_velocity_acceleration(cutoff: float, node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._compute_nodal_velocity()
                node[ng, nb]._compute_nodal_acceleration()
                
# ========================================================= #
#                 Grid to Particle (G2P)                    #
# ========================================================= #
@ti.kernel
def kernel_kinemaitc_ig2p(total_nodes: int, alpha: float, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            accleration, velocity, displacement = ZEROVEC3f, ZEROVEC3f, ZEROVEC3f
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                velocity += shape_mapping(shape_fn, node[nodeID, bodyID].momentum)
                accleration += shape_mapping(shape_fn, node[nodeID, bodyID].inertia)
                displacement += shape_mapping(shape_fn, node[nodeID, bodyID].displacement)
            particle[np]._update_particle_state(dt, alpha, velocity, accleration, displacement)