import taichi as ti

from src.consititutive_model.MaterialKernel import MeanStress, get_angular_velocity
from src.utils.constants import Threshold, ZEROMAT4x4, ZEROMAT6x3, ZEROVEC2f, ZEROVEC3f, ZEROMAT2x2, ZEROMAT3x3, DELTA2D, DELTA, EYE
from src.utils.MatrixFunction import truncation, trace
from src.utils.ScalarFunction import vectorize_id, linearize, sgn
from src.utils.ShapeFunctions import ShapeLinear, GShapeLinear, ShapeLinearCenter, ShapeGIMP, GShapeGIMP, ShapeGIMPCenter, ShapeBsplineQ, GShapeBsplineQ, ShapeBsplineC, GShapeBsplineC
from src.utils.TypeDefination import vec2f, vec3f, vec4f, vec6f, mat3x3, mat4x4, vec2i, vec3i, mat2x2
from src.utils.VectorFunction import Normalize, outer_product, MeanValue, Squared, outer_product2D, dot2
import src.utils.GlobalVariable as GlobalVariable


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
@ti.kernel
def lightweight_p2g(particleNum: int, gnum: ti.types.vector(GlobalVariable.DIMENSION, int), grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float), gravity: ti.types.vector(3, float), 
                    particle_lengths: ti.template(), boundary_types: ti.template(), node: ti.template(), particle: ti.template()):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.momentum.get_scalar_field(d))
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        position = particle[np].x
        velocity = particle[np].v
        p_mass = particle[np].m
        velocity_gradient = particle[np].velocity_gradient
        previous_stress, volume = particle[np].stress, particle[np].vol
        psize = particle_lengths[bodyID]

        internal_force = -volume * previous_stress
        external_force = particle[np]._compute_external_force(gravity)
        for offset in ti.static(ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE, ) * GlobalVariable.DIMENSION)))):
            base = ti.floor((position - psize) * igrid_size).cast(int)
            grid_id = base + offset
            if all(grid_id >= 0):
                nodeID = linearize(grid_id, gnum)
                grid_pos = grid_id * grid_size
                shape_fn, dshape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(float, GlobalVariable.DIMENSION)
                shape_fnc = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                if ti.static(GlobalVariable.SHAPEFUNCTION == 0): 
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        shape_fn[d] = ShapeLinear(position[d], grid_pos[d], igrid_size[d], 0)
                        dshape_fn[d] = GShapeLinear(position[d], grid_pos[d], igrid_size[d], 0)
                        if ti.static(GlobalVariable.BBAR):
                            shape_fnc[d] = ShapeLinearCenter(position[d], grid_pos[d], igrid_size[d], 0)
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 1): 
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        shape_fn[d] = ShapeGIMP(position[d], grid_pos[d], igrid_size[d], psize[d])
                        dshape_fn[d] = GShapeGIMP(position[d], grid_pos[d], igrid_size[d], psize[d])
                        if ti.static(GlobalVariable.BBAR):
                            shape_fnc[d] = ShapeGIMPCenter(position[d], grid_pos[d], igrid_size[d], psize[d])
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 2): 
                    boundary_type = boundary_types[nodeID, bodyID]
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        btypes = int(boundary_type[d])
                        shape_fn[d] = ShapeBsplineQ(position[d], grid_pos[d], igrid_size[d], btypes)
                        dshape_fn[d] = GShapeBsplineQ(position[d], grid_pos[d], igrid_size[d], btypes)
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 3): 
                    boundary_type = boundary_types[nodeID, bodyID]
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        btypes = int(boundary_type[d])
                        shape_fn[d] = ShapeBsplineC(position[d], grid_pos[d], igrid_size[d], btypes)
                        dshape_fn[d] = GShapeBsplineC(position[d], grid_pos[d], igrid_size[d], btypes)

                dpos = position - grid_pos
                weight = 1.0
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    weight *= shape_fn[d]
                pforce = weight * external_force
                
                if ti.static(GlobalVariable.DIMENSION == 2):
                    weight_grad = ti.Vector([dshape_fn[0] * shape_fn[1], shape_fn[0] * dshape_fn[1]]) 
                    if ti.static(GlobalVariable.BBAR):
                        weight_gradc = vec3f([dshape_fn[0] * shape_fnc[1], shape_fnc[0] * dshape_fn[1]])
                        temp_dshape = 0.5 * (weight_gradc - weight_grad)
                        pforce += vec2f([(weight_grad[0] + temp_dshape[0]) * internal_force[0] + temp_dshape[0] * internal_force[1] + temp_dshape[0] * internal_force[2] + weight_grad[1] * internal_force[3],
                                          temp_dshape[1] * internal_force[0] + (weight_grad[1] + temp_dshape[1]) * internal_force[1] + temp_dshape[1] * internal_force[2] + weight_grad[0] * internal_force[3]])
                    else:
                        pforce += vec2f([weight_grad[0] * internal_force[0] + weight_grad[1] * internal_force[3], weight_grad[1] * internal_force[1] + weight_grad[0] * internal_force[3]])
                elif ti.static(GlobalVariable.DIMENSION == 3):
                    weight_grad = ti.Vector([dshape_fn[0] * shape_fn[1] * shape_fn[2], 
                                             shape_fn[0] * dshape_fn[1] * shape_fn[2], 
                                             shape_fn[0] * shape_fn[1] * dshape_fn[2]]) 
                    if ti.static(GlobalVariable.BBAR):
                        weight_gradc = vec3f([dshape_fn[0] * shape_fnc[1] * shape_fnc[2], 
                                             shape_fnc[0] * dshape_fn[1] * shape_fnc[2], 
                                             shape_fnc[0] * shape_fnc[1] * dshape_fn[2]])
                        temp_dshape = (weight_gradc - weight_grad) / 3.
                        pforce += vec3f([(weight_grad[0] + temp_dshape[0]) * internal_force[0] + temp_dshape[0] * internal_force[1] + temp_dshape[0] * internal_force[2] + weight_grad[1] * internal_force[3] + weight_grad[2] * internal_force[5],
                                          temp_dshape[1] * internal_force[0] + (weight_grad[1] + temp_dshape[1]) * internal_force[1] + temp_dshape[1] * internal_force[2] + weight_grad[0] * internal_force[3] + weight_grad[2] * internal_force[4],
                                          temp_dshape[2] * internal_force[0] + temp_dshape[2] * internal_force[1] + (weight_grad[2] + temp_dshape[2]) * internal_force[2] + weight_grad[1] * internal_force[4] + weight_grad[0] * internal_force[5]])
                    else:
                        pforce += vec3f([weight_grad[0] * internal_force[0] + weight_grad[1] * internal_force[3] + weight_grad[2] * internal_force[5],
                                         weight_grad[1] * internal_force[1] + weight_grad[0] * internal_force[3] + weight_grad[2] * internal_force[4],
                                         weight_grad[2] * internal_force[2] + weight_grad[1] * internal_force[4] + weight_grad[0] * internal_force[5]])
                nmass = weight * p_mass
                momentum = nmass * velocity
                if ti.static(GlobalVariable.APIC or GlobalVariable.TPIC):
                    momentum -= nmass * velocity_gradient @ dpos
                node[nodeID, bodyID].m += nmass
                node[nodeID, bodyID].momentum += momentum
                node[nodeID, bodyID].force += pforce

@ti.kernel
def lightweight_g2p(particleNum: int, alpha: float, cutoff: float, fraction: float, gnum: ti.types.vector(GlobalVariable.DIMENSION, int), grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float), 
                    dt: ti.template(), particle_lengths: ti.template(), boundary_types: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.momentum.get_scalar_field(d))
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        #np = particleID[pid]
        bodyID = int(particle[np].bodyID)
        materialID = int(particle[np].materialID)
        position = particle[np].x
        vFLIP = particle[np].v
        psize = particle_lengths[bodyID]

        vPIC = ti.Vector.zero(float, GlobalVariable.DIMENSION)
        Wp = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
        velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
        velocity_gradient_bar = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
        for offset in ti.static(ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE, ) * GlobalVariable.DIMENSION)))):
            base = ti.floor((position - psize) * igrid_size).cast(int)
            grid_id = base + offset
            if all(grid_id >= 0):
                nodeID = linearize(grid_id, gnum)
                grid_pos = grid_id * grid_size
                shape_fn, dshape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(float, GlobalVariable.DIMENSION)
                shape_fnc = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                if ti.static(GlobalVariable.SHAPEFUNCTION == 0): 
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        shape_fn[d] = ShapeLinear(position[d], grid_pos[d], igrid_size[d], 0)
                        if ti.static(not GlobalVariable.APIC):
                            dshape_fn[d] = GShapeLinear(position[d], grid_pos[d], igrid_size[d], 0)
                        if ti.static(GlobalVariable.BBAR):
                            shape_fnc[d] = ShapeLinearCenter(position[d], grid_pos[d], igrid_size[d], 0)
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 1): 
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        shape_fn[d] = ShapeGIMP(position[d], grid_pos[d], igrid_size[d], psize[d])
                        if ti.static(not GlobalVariable.APIC):
                            dshape_fn[d] = GShapeGIMP(position[d], grid_pos[d], igrid_size[d], psize[d])
                        if ti.static(GlobalVariable.BBAR):
                            shape_fnc[d] = ShapeGIMPCenter(position[d], grid_pos[d], igrid_size[d], psize[d])
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 2): 
                    boundary_type = boundary_types[nodeID, bodyID]
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        btypes = int(boundary_type[d])
                        shape_fn[d] = ShapeBsplineQ(position[d], grid_pos[d], igrid_size[d], btypes)
                        if ti.static(not GlobalVariable.APIC):
                            dshape_fn[d] = GShapeBsplineQ(position[d], grid_pos[d], igrid_size[d], btypes)
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 3): 
                    boundary_type = boundary_types[nodeID, bodyID]
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        btypes = int(boundary_type[d])
                        shape_fn[d] = ShapeBsplineC(position[d], grid_pos[d], igrid_size[d], btypes)
                        if ti.static(not GlobalVariable.APIC):
                            dshape_fn[d] = GShapeBsplineC(position[d], grid_pos[d], igrid_size[d], btypes)

                velocity = node[nodeID, bodyID].momentum
                acceleration = node[nodeID, bodyID].force
                weight = 1.0
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    weight *= shape_fn[d]
                vPIC += weight * velocity
                vFLIP += weight * acceleration * dt[None]

                if ti.static(GlobalVariable.APIC):
                    dpos = position - grid_pos
                    if ti.static(GlobalVariable.DIMENSION == 2):
                        Wp += weight * outer_product2D(dpos, dpos)
                        velocity_gradient += weight * outer_product2D(dpos, velocity)
                    elif ti.static(GlobalVariable.DIMENSION == 3):
                        Wp += weight * outer_product(dpos, dpos)
                        velocity_gradient += weight * outer_product(dpos, velocity)
                else:
                    if ti.static(GlobalVariable.DIMENSION == 2):
                        weight_grad = ti.Vector([dshape_fn[0] * shape_fn[1], shape_fn[0] * dshape_fn[1]]) 
                        velocity_gradient_increment = outer_product2D(weight_grad, velocity)
                        velocity_gradient += velocity_gradient_increment
                        if ti.static(GlobalVariable.BBAR):
                            weight_gradc = vec3f([dshape_fn[0] * shape_fnc[1], shape_fnc[0] * dshape_fn[1]])
                            temp_dshape = 0.5 * (weight_gradc - weight_grad)
                            average_bmatrix = temp_dshape[0] * velocity[0] + temp_dshape[1] * velocity[1]
                            velocity_gradient_bar += velocity_gradient_increment
                            velocity_gradient_bar[0, 0] += average_bmatrix
                            velocity_gradient_bar[1, 1] += average_bmatrix
                    elif ti.static(GlobalVariable.DIMENSION == 3):
                        weight_grad = ti.Vector([dshape_fn[0] * shape_fn[1] * shape_fn[2], 
                                                shape_fn[0] * dshape_fn[1] * shape_fn[2], 
                                                shape_fn[0] * shape_fn[1] * dshape_fn[2]])
                        velocity_gradient_increment = outer_product(weight_grad, velocity)
                        velocity_gradient += velocity_gradient_increment
                        if ti.static(GlobalVariable.BBAR):
                            weight_gradc = vec3f([dshape_fn[0] * shape_fnc[1] * shape_fnc[2], 
                                                  shape_fnc[0] * dshape_fn[1] * shape_fnc[2], 
                                                  shape_fnc[0] * shape_fnc[1] * dshape_fn[2]])
                            temp_dshape = (weight_gradc - weight_grad) / 3.
                            average_bmatrix = temp_dshape[0] * velocity[0] + temp_dshape[1] * velocity[1] + temp_dshape[2] * velocity[2]
                            velocity_gradient_bar += velocity_gradient_increment
                            velocity_gradient_bar[0, 0] += average_bmatrix
                            velocity_gradient_bar[1, 1] += average_bmatrix
                            velocity_gradient_bar[2, 2] += average_bmatrix

        if ti.static(not GlobalVariable.BBAR):
            velocity_gradient_bar = velocity_gradient
                    
        if ti.static(GlobalVariable.APIC):
            velocity_gradient = Wp.inverse() @ velocity_gradient

        particle[np].velocity_gradient = velocity_gradient_bar
        particle[np].v = alpha * vPIC + (1. - alpha) * vFLIP
        particle[np].x += vPIC * dt[None]
        
        if ti.static(not GlobalVariable.FBAR):
            previous_stress, volume = particle[np].stress, particle[np].vol
            volume *= (ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity_gradient).determinant()
            stress = ti.Vector.zero(float, 6)
            for matID in ti.static(range(1, matProps.shape[0])):
                if materialID == matID:
                    if ti.static(GlobalVariable.DIMENSION == 2):
                        stress = matProps[1].ComputeStress2D(np, previous_stress, velocity_gradient_bar, stateVars, dt)
                    elif ti.static(GlobalVariable.DIMENSION == 3):
                        stress = matProps[1].ComputeStress(np, previous_stress, velocity_gradient_bar, stateVars, dt)
            particle[np].stress = stress
            particle[np].vol = volume

    if ti.static(GlobalVariable.FBAR):
        node.jacobian.fill(0)
        for np in range(particleNum):
            bodyID = int(particle[np].bodyID)
            mass = particle[np].m
            position = particle[np].x
            velocity_gradient_bar = particle[np].velocity_gradient
            psize = particle_lengths[bodyID]
            djacobian = (ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity_gradient_bar).determinant()
            transfer_var = mass * djacobian
            for offset in ti.static(ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE, ) * GlobalVariable.DIMENSION)))):
                base = ti.floor((position - psize) * igrid_size).cast(int)
                grid_id = base + offset
                if all(grid_id >= 0):
                    nodeID = linearize(grid_id, gnum)
                    grid_pos = grid_id * grid_size
                    shape_fn, dshape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(float, GlobalVariable.DIMENSION)
                    shape_fnc = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                    if ti.static(GlobalVariable.SHAPEFUNCTION == 0): 
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            shape_fn[d] = ShapeLinear(position[d], grid_pos[d], igrid_size[d], 0)
                            if ti.static(not GlobalVariable.APIC):
                                dshape_fn[d] = GShapeLinear(position[d], grid_pos[d], igrid_size[d], 0)
                            if ti.static(GlobalVariable.BBAR):
                                shape_fnc[d] = ShapeLinearCenter(position[d], grid_pos[d], igrid_size[d], 0)
                    elif ti.static(GlobalVariable.SHAPEFUNCTION == 1): 
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            shape_fn[d] = ShapeGIMP(position[d], grid_pos[d], igrid_size[d], psize[d])
                            if ti.static(not GlobalVariable.APIC):
                                dshape_fn[d] = GShapeGIMP(position[d], grid_pos[d], igrid_size[d], psize[d])
                            if ti.static(GlobalVariable.BBAR):
                                shape_fnc[d] = ShapeGIMPCenter(position[d], grid_pos[d], igrid_size[d], psize[d])
                    elif ti.static(GlobalVariable.SHAPEFUNCTION == 2): 
                        boundary_type = boundary_types[nodeID, bodyID]
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            btypes = int(boundary_type[d])
                            shape_fn[d] = ShapeBsplineQ(position[d], grid_pos[d], igrid_size[d], btypes)
                            if ti.static(not GlobalVariable.APIC):
                                dshape_fn[d] = GShapeBsplineQ(position[d], grid_pos[d], igrid_size[d], btypes)
                    elif ti.static(GlobalVariable.SHAPEFUNCTION == 3): 
                        boundary_type = boundary_types[nodeID, bodyID]
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            btypes = int(boundary_type[d])
                            shape_fn[d] = ShapeBsplineC(position[d], grid_pos[d], igrid_size[d], btypes)
                            if ti.static(not GlobalVariable.APIC):
                                dshape_fn[d] = GShapeBsplineC(position[d], grid_pos[d], igrid_size[d], btypes)

                    weight = 1.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        weight *= shape_fn[d]
                    node[nodeID, bodyID].jacobian += weight * transfer_var

        for ng in range(node.shape[0]):
            for nb in range(node.shape[1]):
                if node[ng, nb].m > cutoff:
                    node[ng, nb].jacobian /= node[ng, nb].m

        for np in range(particleNum):
            bodyID = int(particle[np].bodyID)
            mass = particle[np].m
            position = particle[np].x
            velocity_gradient_bar = particle[np].velocity_gradient
            psize = particle_lengths[bodyID]
            djacobian = (ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity_gradient_bar).determinant()
            transfer_var = mass * djacobian
            djacobian_bar = 0.
            for offset in ti.static(ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE, ) * GlobalVariable.DIMENSION)))):
                base = ti.floor((position - psize) * igrid_size).cast(int)
                grid_id = base + offset
                if all(grid_id >= 0):
                    nodeID = linearize(grid_id, gnum)
                    grid_pos = grid_id * grid_size
                    shape_fn, dshape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(float, GlobalVariable.DIMENSION)
                    shape_fnc = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                    if ti.static(GlobalVariable.SHAPEFUNCTION == 0): 
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            shape_fn[d] = ShapeLinear(position[d], grid_pos[d], igrid_size[d], 0)
                            if ti.static(not GlobalVariable.APIC):
                                dshape_fn[d] = GShapeLinear(position[d], grid_pos[d], igrid_size[d], 0)
                            if ti.static(GlobalVariable.BBAR):
                                shape_fnc[d] = ShapeLinearCenter(position[d], grid_pos[d], igrid_size[d], 0)
                    elif ti.static(GlobalVariable.SHAPEFUNCTION == 1): 
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            shape_fn[d] = ShapeGIMP(position[d], grid_pos[d], igrid_size[d], psize[d])
                            if ti.static(not GlobalVariable.APIC):
                                dshape_fn[d] = GShapeGIMP(position[d], grid_pos[d], igrid_size[d], psize[d])
                            if ti.static(GlobalVariable.BBAR):
                                shape_fnc[d] = ShapeGIMPCenter(position[d], grid_pos[d], igrid_size[d], psize[d])
                    elif ti.static(GlobalVariable.SHAPEFUNCTION == 2): 
                        boundary_type = boundary_types[nodeID, bodyID]
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            btypes = int(boundary_type[d])
                            shape_fn[d] = ShapeBsplineQ(position[d], grid_pos[d], igrid_size[d], btypes)
                            if ti.static(not GlobalVariable.APIC):
                                dshape_fn[d] = GShapeBsplineQ(position[d], grid_pos[d], igrid_size[d], btypes)
                    elif ti.static(GlobalVariable.SHAPEFUNCTION == 3): 
                        boundary_type = boundary_types[nodeID, bodyID]
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            btypes = int(boundary_type[d])
                            shape_fn[d] = ShapeBsplineC(position[d], grid_pos[d], igrid_size[d], btypes)
                            if ti.static(not GlobalVariable.APIC):
                                dshape_fn[d] = GShapeBsplineC(position[d], grid_pos[d], igrid_size[d], btypes)
                    
                    jacobian = node[nodeID, bodyID].jacobian
                    weight = 1.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        weight *= shape_fn[d]
                    djacobian_bar += jacobian

            velocity_gradient = particle[np].velocity_gradient
            ddeformation_gradient = ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity_gradient
            djacobian = ddeformation_gradient.determinant()
            djacobian_bar_new = fraction * djacobian_bar + (1. - fraction) * djacobian

            multiplier = (djacobian_bar_new / djacobian) ** (1. / GlobalVariable.DIMENSION)
            updated_velocity_gradient = (multiplier - 1.) * ti.Matrix.identity(float, GlobalVariable.DIMENSION) / dt[None] + multiplier * velocity_gradient
            
            particle[np].velocity_gradient = updated_velocity_gradient

    if ti.static(GlobalVariable.PARTICLESHIFTING):
        pass


@ti.kernel
def lightweight_grid_operation(cutoff: float, damp: float, node: ti.template(), dt: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                mass = node[ng, nb].m
                acceleration = node[ng, nb].force  / mass
                velocity = node[ng, nb].momentum / mass
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    if velocity[d] * acceleration[d] > 0.: 
                        acceleration[d] -= damp * ti.abs(acceleration[d]) * sgn(velocity[d])
                node[ng, nb].momentum = velocity + acceleration * dt[None]
                node[ng, nb].force = acceleration 

@ti.kernel
def g2p2g(particleNum: int, p_mass: float, grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float), gravity: ti.types.vector(3, float), 
          dt: ti.template(), node_in: ti.template(), node_out: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), particleID: ti.template()):
    for np in range(particleNum):
        pid = particleID[np]
        position = particle[pid].x
        velocity = particle[pid].v
        base = ti.floor(position * igrid_size - 0.5).cast(int)
        fx = position * igrid_size - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, GlobalVariable.DIMENSION)
        velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
        # Loop over 3x3 grid node neighborhood
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * GlobalVariable.DIMENSION)))):
            dpos = offset.cast(float) - fx
            g_v = node_in[base + offset, 0].momentum
            weight = 1.0
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                weight *= w[offset[d]][d]
            new_v += weight * g_v
            velocity_gradient += 4 * igrid_size * weight * g_v.outer_product(dpos)

        # P2G
        base = ti.floor(position * igrid_size - 0.5).cast(int)
        fx = position * igrid_size - float(base)
        w2 = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        w_grad = [fx - 1.5, -2 * (fx - 1), fx - 0.5]
        # Deformation gradient update
        previous_stress, volume = particle[np].stress, particle[np].vol
        volume *= (ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity).determinant()
        stress = ti.Vector.zero(float, 6)
        if ti.static(GlobalVariable.DIMENSION == 2):
            stress = matProps[1].ComputeStress2D(np, previous_stress, velocity_gradient, stateVars, dt)
        elif ti.static(GlobalVariable.DIMENSION == 3):
            stress = matProps[1].ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        particle[np].stress = stress
        particle[np].vol = volume
        fInt = volume * stress
        # Loop over 3x3 grid node neighborhood
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * GlobalVariable.DIMENSION)))):
            dpos = (offset.cast(float) - fx) * grid_size
            weight = 1.0
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                weight *= w2[offset[d]][d]
                
            pforce = particle[np]._compute_external_force(gravity)
            if ti.static(GlobalVariable.DIMENSION == 2):
                weight_grad = ti.Vector([w_grad[offset[0]][0] * w[offset[1]][1], w[offset[0]][0] * w_grad[offset[1]][1]]) * igrid_size
                pforce += vec2f([weight_grad[0] * fInt[0] + weight_grad[1] * fInt[3],
                                weight_grad[1] * fInt[1] + weight_grad[0] * fInt[3]]) + p_mass * vec2f(gravity[0], gravity[1]) 
            elif ti.static(GlobalVariable.DIMENSION == 3):
                weight_grad = ti.Vector([w_grad[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2], 
                                         w[offset[0]][0] * w_grad[offset[1]][1] * w[offset[2]][2], 
                                         w[offset[0]][0] * w[offset[1]][1] * w_grad[offset[2]][2]]) * igrid_size
                pforce += vec3f([weight_grad[0] * fInt[0] + weight_grad[1] * fInt[3] + weight_grad[2] * fInt[5],
                                weight_grad[1] * fInt[1] + weight_grad[0] * fInt[3] + weight_grad[2] * fInt[4],
                                weight_grad[2] * fInt[2] + weight_grad[1] * fInt[4] + weight_grad[0] * fInt[5]]) + p_mass * gravity
            node_out[base + offset, 0].momentum += weight * (p_mass * velocity + velocity_gradient @ dpos + pforce)
            node_out[base + offset, 0].mass += weight * p_mass

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
def kernel_angular_velocity_p2c(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            angular_velocity = get_angular_velocity(particle[np].velocity_gradient)
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_angular_velocity(nmass * angular_velocity)


@ti.kernel
def kernel_volume_p2c(cnum: ti.types.vector(3, int), inv_dx: ti.types.vector(3, float), particleNum: int, cell: ti.template(), particle: ti.template()):
    ti.block_local(cell.volume)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            cellID = ti.floor(particle[np].x * inv_dx, int)
            linear_cellID = linearize(cellID, cnum)
            cell[linear_cellID, bodyID]._update_cell_volume(particle[np].vol)

@ti.kernel
def kernel_volume_p2c_2D(cnum: ti.types.vector(2, int), inv_dx: ti.types.vector(2, float), particleNum: int, cell: ti.template(), particle: ti.template()):
    ti.block_local(cell.volume)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            cellID = ti.floor(particle[np].x * inv_dx, int)
            linear_cellID = linearize(cellID, cnum)
            cell[linear_cellID, bodyID]._update_cell_volume(particle[np].vol)

@ti.kernel
def kernel_mass_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    ti.block_local(node.m)
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
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.momentum.get_scalar_field(d))
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
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.momentum.get_scalar_field(d))
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
def kernel_mass_momentum_taylor_p2g(total_nodes: int, particleNum: int, gnum: ti.types.vector(GlobalVariable.DIMENSION, int), grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), 
                                    node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.momentum.get_scalar_field(d))
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
                nodal_coord = grid_size * ti.Vector(vectorize_id(nodeID, gnum))
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_mass(nmass)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * (velocity + gradv @ (nodal_coord - xp)))

@ti.kernel
def kernel_mass_momentum_taylor_p2g_2DAxisy(total_nodes: int, particleNum: int, gnum: ti.types.vector(GlobalVariable.DIMENSION, int), grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), 
                                    node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.momentum.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            xp = particle[np].x
            mass = particle[np].m
            velocity = particle[np].v
            velocity_gradient = particle[np].velocity_gradient
            gradv = mat2x2([velocity_gradient[0, 0], velocity_gradient[0, 1]], [velocity_gradient[1, 0], velocity_gradient[1, 1]])
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nodal_coord = grid_size * ti.Vector(vectorize_id(nodeID, gnum))
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_mass(nmass)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * (velocity + gradv @ (nodal_coord - xp)))

@ti.kernel
def kernel_mass_momentum_p2g_twophase(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.momentum.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            mass_s = particle[np].ms
            mass_f = particle[np].mf
            velocity = particle[np].v
            velocity_s = particle[np].vs
            velocity_f = particle[np].vf
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                nmass_s = shape_mapping(shapefn[ln], mass_s)
                nmass_f = shape_mapping(shapefn[ln], mass_f)
                node[nodeID, bodyID]._update_nodal_mass(nmass,nmass_s,nmass_f)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * velocity, nmass_s * velocity_s,  nmass_f * velocity_f)

@ti.kernel
def kernel_assemble_dem_contact_force(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex = particle[np].contact_traction
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                cforce = shape_mapping(shapefn[ln], fex)
                node[nodeID, bodyID]._update_contact_force(cforce)

@ti.kernel
def kernel_external_force_p2g(total_nodes: int, gravity: ti.types.vector(3, float), particleNum: int, node: ti.template(), particle: ti.template(), 
                              LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
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
def kernel_force_p2g(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(), 
                     LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fex = particle[np]._compute_external_force(gravity)
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = vec3f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)

@ti.kernel
def kernel_force_p2g_2D(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(), 
                        LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fex = particle[np]._compute_external_force(gravity)
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)

@ti.kernel
def kernel_viscous_force_p2g(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), 
                             LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template(), matProps: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.momentum.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            materialID = int(particle[np].materialID)
            offset = np * total_nodes
            velocity_gradient = particle[np].velocity_gradient
            viscous_stress = matProps[materialID].ComputeShearStress(velocity_gradient)
            fInt = particle[np].vol * viscous_stress
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec3f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]])
                node[nodeID, bodyID].momentum += internal_force * dt[None]

@ti.kernel
def kernel_viscous_force_p2g_2D(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), 
                                LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template(), matProps: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.momentum.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            materialID = int(particle[np].materialID)
            offset = np * total_nodes
            velocity_gradient = particle[np].velocity_gradient
            viscous_stress = matProps[materialID].ComputeShearStress2D(velocity_gradient)
            fInt = particle[np].vol * viscous_stress
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                node[nodeID, bodyID].momentum += internal_force * dt[None]
    
@ti.kernel
def kernel_force_p2g_2DAxisy(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(), 
                             LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            position = particle[np].x
            fex = particle[np]._compute_external_force(gravity)
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn=shapefn[ln]
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + fInt[2] * shape_fn / position[0],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)

@ti.kernel
def kernel_force_bbar_p2g_2DAxisy(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(), 
                                  LnID: ti.template(), shapefn: ti.template(), shapefnc: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            position = particle[np].x
            fex = particle[np]._compute_external_force(gravity)
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                shape_fnc = shapefnc[ln]
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]

                B0 = shape_fn / position[0]
                B1 = dshape_fn[0]
                B2 = dshape_fn[1]
                B0bar = shape_fnc / position[0]#((position[0] // grid_size[0] + 0.5) * grid_size[0])
                B1bar = dshape_fnc[0]
                B2bar = dshape_fnc[1]

                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = 1./3. * vec2f([(B1bar + 2. * B1 + B0bar - B0) * fInt[0] + (B1bar - B1 + B0bar - B0) * fInt[1] + (B1bar - B1 + B0bar + 2. * B0) * fInt[2] + 3. * B2 * fInt[3],
                                                (B2bar - B2) * fInt[0] + (B2bar + 2. * B2) * fInt[1] + (B2bar - B2) * fInt[2] + 3. * B1 * fInt[3]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)

@ti.kernel
def kernel_force_mls_p2g(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), gnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), inertia_tensor: ti.types.vector(3, float), 
                         node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            position = particle[np].x
            fex = particle[np]._compute_external_force(gravity)
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                pointer = grid_size * ti.Vector([*vectorize_id(nodeID, gnum)]) - position
                internal_force = vec3f(fInt[0] * inertia_tensor[0] * pointer[0] + fInt[3] * inertia_tensor[1] * pointer[1] + fInt[5] * inertia_tensor[2] * pointer[2],
                                       fInt[3] * inertia_tensor[0] * pointer[0] + fInt[1] * inertia_tensor[1] * pointer[1] + fInt[4] * inertia_tensor[2] * pointer[2],
                                       fInt[5] * inertia_tensor[0] * pointer[0] + fInt[4] * inertia_tensor[1] * pointer[1] + fInt[2] * inertia_tensor[2] * pointer[2])
                node[nodeID, bodyID]._update_nodal_force(shape_fn * (fex + internal_force))

@ti.kernel
def kernel_force_mls_p2g_2D(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), gnum: ti.types.vector(2, int), grid_size: ti.types.vector(2, float), inertia_tensor: ti.types.vector(2, float), 
                         node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            position = particle[np].x
            fex = particle[np]._compute_external_force(gravity)
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                pointer = grid_size * ti.Vector([*vectorize_id(nodeID, gnum)]) - position
                internal_force = vec2f(fInt[0] * inertia_tensor[0] * pointer[0] + fInt[3] * inertia_tensor[1] * pointer[1],
                                       fInt[3] * inertia_tensor[0] * pointer[0] + fInt[1] * inertia_tensor[1] * pointer[1])
                node[nodeID, bodyID]._update_nodal_force(shape_fn * (fex + internal_force))

@ti.kernel
def kernel_reference_force_p2g(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(), 
                               LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fex = particle[np]._compute_external_force(gravity)
            fInt = -particle[np].vol * particle[np].stress
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = fInt @ dshapefn[ln] 
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)

@ti.kernel
def kernel_external_force_p2g_twophase(total_nodes: int, gravity: ti.types.vector(3, float), particleNum: int, node: ti.template(), particle: ti.template(),
                              LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): 
        ti.block_local(node.force.get_scalar_field(d))
        ti.block_local(node.forcef.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex, fexf = particle[np]._compute_external_force(gravity)
            drag = particle[np]._compute_drag_force()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                external_force = shape_mapping(shapefn[ln], fex)
                external_forcef = shape_mapping(shapefn[ln], fexf)
                drag_force = shape_mapping(shapefn[ln], drag)
                node[nodeID, bodyID]._update_external_force(external_force, external_forcef + drag_force)

@ti.kernel
def kernel_force_p2g_twophase(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(),
                              LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): 
        ti.block_local(node.force.get_scalar_field(d))
        ti.block_local(node.forcef.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex, fexf = particle[np]._compute_external_force(gravity)
            fInt, fintf = particle[np]._compute_internal_force()
            drag = particle[np]._compute_drag_force()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shape_fn, fex)
                external_forcef = shape_mapping(shape_fn, fexf)
                drag_force = shape_mapping(shape_fn, drag)
                internal_force = vec3f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]])
                internal_forcef = vec3f([dshape_fn[0] * fintf[0],
                                         dshape_fn[1] * fintf[1],
                                         dshape_fn[2] * fintf[2]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force, external_forcef + drag_force + internal_forcef)

@ti.kernel
def kernel_force_p2g_twophase2D(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(),
                              LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): 
        ti.block_local(node.force.get_scalar_field(d))
        ti.block_local(node.forcef.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex, fexf = particle[np]._compute_external_force(gravity)
            fInt, fintf = particle[np]._compute_internal_force()
            drag = particle[np]._compute_drag_force()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shape_fn, fex)
                external_forcef = shape_mapping(shape_fn, fexf)
                drag_force = shape_mapping(shape_fn, drag)
                internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                internal_forcef = vec2f([dshape_fn[0] * fintf[0],
                                         dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force, external_forcef + drag_force + internal_forcef)

@ti.kernel
def kernel_force_bbar_p2g_twophase2D(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(),
                              LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): 
        ti.block_local(node.force.get_scalar_field(d))
        ti.block_local(node.forcef.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex, fexf = particle[np]._compute_external_force(gravity)
            fInt, fintf = particle[np]._compute_internal_force()
            drag = particle[np]._compute_drag_force()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                temp_dshape = 0.5 * (dshape_fnc - dshape_fn)
                external_force = shape_mapping(shape_fn, fex)
                external_forcef = shape_mapping(shape_fn, fexf)
                drag_force = shape_mapping(shape_fn, drag)
                internal_force = vec2f([(dshape_fn[0] + temp_dshape[0]) * fInt[0] + temp_dshape[0] * fInt[1] + temp_dshape[0] * fInt[2] + dshape_fn[1] * fInt[3],
                                        temp_dshape[1] * fInt[0] + (dshape_fn[1] + temp_dshape[1]) * fInt[1] + temp_dshape[1] * fInt[2] + dshape_fn[0] * fInt[3]])
                internal_forcef = vec2f([dshape_fn[0] * fintf[0],
                                         dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force, external_forcef + drag_force + internal_forcef)

@ti.kernel
def kernel_force_p2g_twophase_2DAxisy(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(),
                                     LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): 
        ti.block_local(node.force.get_scalar_field(d))
        ti.block_local(node.forcef.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            position = particle[np].x
            fex, fexf = particle[np]._compute_external_force(gravity)
            fInt, fintf = particle[np]._compute_internal_force()
            drag = particle[np]._compute_drag_force()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                
                external_force = shape_mapping(shape_fn, fex)
                external_forcef = shape_mapping(shape_fn, fexf)
                drag_force = shape_mapping(shape_fn, drag)
                internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + fInt[2] * shape_fn / position[0],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                internal_forcef = vec2f([dshape_fn[0] * fintf[0],
                                         dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force, external_forcef + drag_force + internal_forcef)

@ti.kernel
def kernel_force_bbar_p2g_twophase_2DAxisy(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(),
                                     LnID: ti.template(), shapefn: ti.template(), shapefnc: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): 
        ti.block_local(node.force.get_scalar_field(d))
        ti.block_local(node.forcef.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            position = particle[np].x
            fex, fexf = particle[np]._compute_external_force(gravity)
            fInt, fintf = particle[np]._compute_internal_force()
            drag = particle[np]._compute_drag_force()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                shape_fnc = shapefnc[ln]
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]

                B0 = shape_fn / position[0]
                B1 = dshape_fn[0]
                B2 = dshape_fn[1]
                B0bar = shape_fnc / position[0]
                B1bar = dshape_fnc[0]
                B2bar = dshape_fnc[1]
                
                external_force = shape_mapping(shape_fn, fex)
                external_forcef = shape_mapping(shape_fn, fexf)
                drag_force = shape_mapping(shape_fn, drag)
                internal_force = 1./3. * vec2f([(B1bar + 2. * B1 + B0bar - B0) * fInt[0] + (B1bar - B1 + B0bar - B0) * fInt[1] + (B1bar - B1 + B0bar + 2. * B0) * fInt[2] + 3. * B2 * fInt[3],
                                                (B2bar - B2) * fInt[0] + (B2bar + 2. * B2) * fInt[1] + (B2bar - B2) * fInt[2] + 3. * B1 * fInt[3]])
                internal_forcef = vec2f([dshape_fn[0] * fintf[0],
                                         dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force, external_forcef + drag_force + internal_forcef)

@ti.kernel
def kernel_sum_cell_stress(gauss_num: int, dx: ti.types.vector(3, float), inv_dx: ti.types.vector(3, float), cnum: ti.types.vector(3, int), particleNum: int, particle: ti.template(), cell: ti.template(), sub_cell: ti.template()):
    gauss_point_num = gauss_num * gauss_num * gauss_num
    for d in ti.static(range(6)): 
        ti.block_local(sub_cell.stress.get_scalar_field(d))
    ti.block_local(sub_cell.vol)
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
def kernel_sum_cell_stress_2D(gauss_num: int, dx: ti.types.vector(2, float), inv_dx: ti.types.vector(2, float), cnum: ti.types.vector(2, int), particleNum: int, particle: ti.template(), cell: ti.template(), sub_cell: ti.template()):
    gauss_point_num = gauss_num * gauss_num
    for d in ti.static(range(6)): 
        ti.block_local(sub_cell.stress.get_scalar_field(d))
    ti.block_local(sub_cell.vol)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            element_id = ti.floor(particle[np].x * inv_dx, int)
            linear_element_id = int(element_id[0] + element_id[1] * cnum[0])
            if int(cell[linear_element_id, bodyID].active) == 1:
                volume = particle[np].vol
                sub_element_id = ti.floor((particle[np].x - element_id * dx) * inv_dx * gauss_num, int)
                sub_linear_element_id = sub_element_id[0] + sub_element_id[1] * gauss_num
                sub_cell[linear_element_id * gauss_point_num + sub_linear_element_id, bodyID].stress += particle[np].stress * volume
                sub_cell[linear_element_id * gauss_point_num + sub_linear_element_id, bodyID].vol += volume

@ti.kernel
def kernel_internal_force_on_gauss_point_p2g(gauss_num: int, cnum: ti.types.vector(3, int), gnum: ti.types.vector(3, int), dx: ti.types.vector(3, float), inv_dx: ti.types.vector(3, float),  
                                             node: ti.template(), cell: ti.template(), sub_cell: ti.template(), gauss_point: ti.template(), weight: ti.template()):
    gauss_point_num = gauss_num * gauss_num * gauss_num
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for nc in range(cell.shape[0]):
        for nb in range(cell.shape[1]):
            if int(cell[nc, nb].active) == 1:
                base = vec3i(vectorize_id(nc, cnum))
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
                        node[nodeID, nb]._update_nodal_force(internal_force)
    
@ti.kernel
def kernel_internal_force_on_gauss_point_p2g_2D(gauss_num: int, cnum: ti.types.vector(2, int), gnum: ti.types.vector(2, int), dx: ti.types.vector(2, float), inv_dx: ti.types.vector(2, float),
                                                node: ti.template(), cell: ti.template(), sub_cell: ti.template(), gauss_point: ti.template(), weight: ti.template()):
    gauss_point_num = gauss_num * gauss_num
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for nc in range(cell.shape[0]):
        for nb in range(cell.shape[1]):
            if int(cell[nc, nb].active) == 1:
                base = vec2i(vectorize_id(nc, cnum))
                volume = dx[0] * dx[1] / gauss_point_num
                for ngp in range(gauss_point_num):
                    gp = 0.5 * dx * (gauss_point[ngp] + 1) + base * dx
                    fInt = -weight[ngp] * sub_cell[nc * gauss_point_num + ngp, nb].stress * volume
                    for i, j in ti.static(ti.ndrange(2, 2)):
                        nx, ny= base[0] + i, base[1] + j
                        nodeID = nx + ny * gnum[0]
                        sx = ShapeLinear(gp[0], nx * dx[0], inv_dx[0], 0)
                        sy = ShapeLinear(gp[1], ny * dx[1], inv_dx[1], 0)
                        gsx = GShapeLinear(gp[0], nx * dx[0], inv_dx[0], 0)
                        gsy = GShapeLinear(gp[1], ny * dx[1], inv_dx[1], 0)
                        dshape_fn = vec2f(gsx * sy, gsy * sx)
                        internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                                dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                        node[nodeID, nb]._update_nodal_force(internal_force)

@ti.kernel
def kernel_internal_force_on_material_point_p2g(cnum: ti.types.vector(3, int), gnum: ti.types.vector(3, int), dx: ti.types.vector(3, float), inv_dx: ti.types.vector(3, float), particleNum: int, node: ti.template(), particle: ti.template(),cell: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
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
                    node[nodeID, bodyID]._update_nodal_force(internal_force)

@ti.kernel
def kernel_internal_force_on_material_point_p2g_2D(cnum: ti.types.vector(2, int), gnum: ti.types.vector(2, int), dx: ti.types.vector(2, float), inv_dx: ti.types.vector(2, float), particleNum: int, node: ti.template(), particle: ti.template(),cell: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            element_id = ti.floor(particle[np].x * inv_dx, int)
            linear_element_id = element_id[0] + element_id[1] * cnum[0]
            if int(cell[linear_element_id, bodyID].active) == 0:
                fInt = particle[np]._compute_internal_force()
                position = particle[np].x
                for i, j in ti.static(ti.ndrange(2, 2)):
                    nx, ny = element_id[0] + i, element_id[1] + j
                    nodeID = nx + ny * gnum[0]
                    sx = ShapeLinear(position[0], nx * dx[0], inv_dx[0], 0)
                    sy = ShapeLinear(position[1], ny * dx[1], inv_dx[1], 0)
                    gsx = GShapeLinear(position[0], nx * dx[0], inv_dx[0], 0)
                    gsy = GShapeLinear(position[1], ny * dx[1], inv_dx[1], 0)
                    dshape_fn = vec2f(gsx * sy, gsy * sx)
                    internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                            dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                    node[nodeID, bodyID]._update_nodal_force(internal_force)

@ti.kernel
def kernel_internal_force_p2g_twophase2D(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt, fintf = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                internal_forcef = vec2f([dshape_fn[0] * fintf[0],
                                         dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_internal_force(internal_force, internal_forcef)

@ti.kernel
def kernel_internal_force_p2g_twophase(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt, fintf = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec3f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]])
                internal_forcef = vec3f([dshape_fn[0] * fintf[0],
                                         dshape_fn[1] * fintf[1],
                                         dshape_fn[2] * fintf[2]])
                node[nodeID, bodyID]._update_internal_force(internal_force, internal_forcef)

@ti.kernel
def kernel_force_bbar_p2g(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(), 
                                   LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fex = particle[np]._compute_external_force(gravity)
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                temp_dshape = (dshape_fnc - dshape_fn) / 3.
                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = vec3f([(dshape_fn[0] + temp_dshape[0]) * fInt[0] + temp_dshape[0] * fInt[1] + temp_dshape[0] * fInt[2] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                        temp_dshape[1] * fInt[0] + (dshape_fn[1] + temp_dshape[1]) * fInt[1] + temp_dshape[1] * fInt[2] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                        temp_dshape[2] * fInt[0] + temp_dshape[2] * fInt[1] + (dshape_fn[2] + temp_dshape[2]) * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force) 

@ti.kernel
def kernel_force_bbar_p2g_2D(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(), 
                                       LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fex = particle[np]._compute_external_force(gravity)
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                temp_dshape = 0.5 * (dshape_fnc - dshape_fn)
                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = vec2f([(dshape_fn[0] + temp_dshape[0]) * fInt[0] + temp_dshape[0] * fInt[1] + temp_dshape[0] * fInt[2] + dshape_fn[1] * fInt[3],
                                        temp_dshape[1] * fInt[0] + (dshape_fn[1] + temp_dshape[1]) * fInt[1] + temp_dshape[1] * fInt[2] + dshape_fn[0] * fInt[3]])
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)

@ti.kernel
def kernel_internal_force_bbar_p2g_twophase2D(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt, fintf = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                internal_forcef = vec2f([dshape_fn[0] * fintf[0],
                                         dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_internal_force(internal_force, internal_forcef)

@ti.kernel
def kernel_volume_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    ti.block_local(node.vol)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            volume = particle[np].vol 
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nvol = shape_mapping(shapefn[ln], volume)
                node[nodeID, bodyID].vol += nvol

@ti.kernel
def kernel_jacobian_p2g(total_nodes: int, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    ti.block_local(node.jacobian)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            mass = particle[np].m 
            velocity_gradient = particle[np].velocity_gradient
            djacobian = (ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity_gradient).determinant()
            transfer_var = mass * djacobian
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                node[nodeID, bodyID].jacobian += shape_mapping(shapefn[ln], transfer_var)

@ti.kernel
def kernel_pressure_p2g(particleNum: int, total_nodes: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    ti.block_local(node.pressure)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            pressure = particle[np].m * particle[np]._get_mean_stress()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                node[nodeID, bodyID].pressure += shape_mapping(shapefn[ln], pressure)

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
def kernel_compute_gauss_average_stress_2D(gauss_num: int, cut_off: float, cell: ti.template(), sub_cell: ti.template()):
    gauss_number = gauss_num * gauss_num
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
def kernel_average_pressure_2D(gauss_num: int, cell: ti.template(), sub_cell: ti.template()):
    gauss_number = gauss_num * gauss_num
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
def kernel_compute_grid_acceleration(cutoff: float, dt: ti.template(), node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._compute_nodal_acceleration(dt)

@ti.kernel
def kernel_compute_grid_velocity_twophase(cutoff: float, node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            node[ng, nb]._compute_nodal_velocity(cutoff)

@ti.kernel
def kernel_compute_grid_kinematic(cutoff: float, damp: float, node: ti.template(), dt: ti.template()):
    # ti.block_local(dt)
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._compute_nodal_kinematic(damp, dt) 

@ti.kernel
def kernel_compute_grid_kinematic_fluid(cutoff: float, damp: float, node: ti.template(), dt: ti.template()):
    # ti.block_local(dt)
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].mf > cutoff:
                node[ng, nb]._compute_nodal_kinematic_fluid(damp, dt)

@ti.kernel
def kernel_compute_grid_kinematic_solid(cutoff: float, damp: float, node: ti.template(), dt: ti.template()):
    # ti.block_local(dt)
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].ms > cutoff:
                node[ng, nb]._compute_nodal_kinematic_solid(damp, dt)

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
def kernel_grid_jacobian(cutoff: float, is_rigid: ti.template(), node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff and is_rigid[nb] == 0:
                node[ng, nb].jacobian /= node[ng, nb].m

@ti.kernel
def kernel_grid_pressure(cutoff: float, is_rigid: ti.template(), node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff and is_rigid[nb] == 0:
                node[ng, nb].pressure /= node[ng, nb].m

@ti.kernel
def kernel_grid_pressure_volume(cutoff: float, is_rigid: ti.template(), node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].vol > cutoff and is_rigid[nb] == 0:
                node[ng, nb].pressure /= node[ng, nb].vol

# ========================================================= #
#                 Grid to Particle (G2P)                    #
# ========================================================= #
@ti.kernel
def kernel_kinemaitc_g2p(total_nodes: int, alpha: float, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    ti.block_local(node.momentum)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPIC, vFLIP = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(float, GlobalVariable.DIMENSION)
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                velocity = node[nodeID, bodyID].momentum
                accleration = node[nodeID, bodyID].force
                vPIC += shape_mapping(shape_fn, velocity)
                vFLIP += shape_mapping(shape_fn, accleration) 
            particle[np]._update_particle_state(dt, alpha, vPIC, vFLIP)

@ti.kernel
def kernel_kinemaitc_g2p_twophase(total_nodes: int, alpha: float, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPICs, vFLIPs = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(float, GlobalVariable.DIMENSION)
            vPICf, vFLIPf = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(float, GlobalVariable.DIMENSION)
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                velocitys = node[nodeID, bodyID].momentums
                velocityf = node[nodeID, bodyID].momentumf
                acclerations = node[nodeID, bodyID].forces
                acclerationf = node[nodeID, bodyID].forcef
                vPICs += shape_mapping(shape_fn, velocitys)
                vFLIPs += shape_mapping(shape_fn, acclerations) * dt[None]
                vPICf += shape_mapping(shape_fn, velocityf)
                vFLIPf += shape_mapping(shape_fn, acclerationf) * dt[None]
            particle[np]._update_particle_state(dt, alpha, vPICs, vFLIPs, vPICs, vFLIPs, vPICf, vFLIPf)

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
def kernel_pressure_g2p(particleNum: int, total_nodes: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            pressure = 0.
            mean_stress = particle[np]._get_mean_stress()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                pressure += shape_mapping(shapefn[ln], node[nodeID, bodyID].pressure)
            particle[np]._update_stress(-(mean_stress - pressure) * EYE)

# ========================================================= #
#                 Apply Constitutive Model                  #
# ========================================================= #
@ti.kernel
def kernel_find_sound_speed(particleNum: int, particle: ti.template(), matProps: ti.template()):
    for np in range(particleNum):
        materialID = particle[np].materialID
        matProps[materialID]._set_modulus(Squared(particle[np].v))

@ti.kernel
def kernel_compute_reference_stress_strain(total_nodes: int, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), 
                                           matProps: ti.template(), stateVars: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient.transpose()
            stateVars[np].deformation_gradient += velocity_gradient * dt[None]
            deformation_gradient = stateVars[np].deformation_gradient
            velocity_gradient = velocity_gradient @ deformation_gradient.inverse()
            stress = particle[np].stress
            particle[np].stress = matProps[materialID].ComputePKStress(np, stress, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_compute_stress_strain(particleNum: int, dt: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress
            particle[np].stress = matProps[materialID].ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_compute_stress_strain_2D(particleNum: int, dt: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress
            particle[np].stress = matProps[materialID].ComputeStress2D(np, previous_stress, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_compute_stress_strain_twophase(particleNum: int, dt: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            solid_velocity_gradient, fluid_velocity_gradient = particle[np].solid_velocity_gradient, particle[np].fluid_velocity_gradient
            previous_stress, porosity = particle[np].stress, particle[np].porosity
            particle[np].pressure -= matProps[materialID].ComputePressure(solid_velocity_gradient, fluid_velocity_gradient, porosity, dt)
            particle[np].stress = matProps[materialID].ComputeStress(np, previous_stress, solid_velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_compute_stress_strain_twophase2D(particleNum: int, dt: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            solid_velocity_gradient, fluid_velocity_gradient = particle[np].solid_velocity_gradient, particle[np].fluid_velocity_gradient
            previous_stress, porosity = particle[np].stress, particle[np].porosity
            particle[np].pressure -= matProps[materialID].ComputePressure(solid_velocity_gradient, fluid_velocity_gradient, porosity, dt)
            particle[np].stress = matProps[materialID].ComputeStress2D(np, previous_stress, solid_velocity_gradient, stateVars, dt)

# ========================================================= #
#                 Update velocity gradient                  #
# ========================================================= #
@ti.kernel
def kernel_update_velocity_gradient_fbar(fraction: float, cutoff: float, total_nodes: int, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), 
                                         LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    eyes = ti.Matrix.identity(float, GlobalVariable.DIMENSION)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            djacobian_bar = 0.
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                djacobian_bar += shape_mapping(shape_fn, node[nodeID, bodyID].jacobian)

            velocity_gradient = particle[np].velocity_gradient
            ddeformation_gradient = eyes + dt[None] * velocity_gradient
            djacobian = ddeformation_gradient.determinant()
            djacobian_bar_new = fraction * djacobian_bar + (1. - fraction) * djacobian
            #jacobian_bar_new = clamp(0.01, 100, jacobian_bar_new)

            multiplier = (djacobian_bar_new / djacobian) ** (1. / GlobalVariable.DIMENSION)
            updated_velocity_gradient = (multiplier - 1.) * eyes / dt[None] + multiplier * velocity_gradient
            
            #particle[np].jacobian = jacobian_bar_new
            particle[np].velocity_gradient = updated_velocity_gradient

@ti.kernel
def kernel_update_velocity_gradient(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(),  
                                    stateVars: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            velocity_gradient = ZEROMAT3x3
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                dshape_fn = dshapefn[ln]
                velocity_gradient += outer_product(dshape_fn, gv)
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps[materialID].update_particle_volume(np, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_update_velocity_gradient_affine(total_nodes: int, particleNum: int, gnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), dt: ti.template(), node: ti.template(), particle: ti.template(), 
                                           matProps: ti.template(), stateVars: ti.template(),  LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            Wp = ZEROMAT3x3
            Bp = ZEROMAT3x3
            offset = np * total_nodes
            position = particle[np].x
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                grid_coord = grid_size * vec3f(vectorize_id(nodeID, gnum))
                pointer = grid_coord - position
                gv = node[nodeID, bodyID].momentum
                shape_fn = shapefn[ln]

                Wp += shape_fn * outer_product(pointer, pointer)
                Bp += shape_fn * outer_product(pointer, gv)
            velocity_gradient = truncation(Bp @ Wp.inverse())
            particle[np].velocity_gradient = velocity_gradient
            particle[np].vol *= matProps[materialID].update_particle_volume(np, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_update_velocity_gradient_bbar(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(),  
                                         stateVars: ti.template(), LnID: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            velocity_gradient = ZEROMAT3x3
            strain_rate_trace = ZEROVEC3f
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
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps[materialID].update_particle_volume_bbar(np, strain_rate_trace, stateVars, dt)

@ti.kernel
def kernel_update_velocity_gradient_2D(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), 
                                LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradient = ZEROMAT2x2
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                dshape_fn = dshapefn[ln]
                velocity_gradient += outer_product2D(dshape_fn, gv)
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps[materialID].update_particle_volume_2D(np, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_update_velocity_gradient_affine_2D(total_nodes: int, particleNum: int, gnum: ti.types.vector(2, int), grid_size: ti.types.vector(2, float), dt: ti.template(), node: ti.template(), particle: ti.template(), 
                                       matProps: ti.template(), stateVars: ti.template(),  LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            Wp = ZEROMAT2x2
            Bp = ZEROMAT2x2
            offset = np * total_nodes
            position = particle[np].x
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                grid_coord = grid_size * vec2f(vectorize_id(nodeID, gnum))
                pointer = grid_coord - position
                gv = node[nodeID, bodyID].momentum
                shape_fn = shapefn[ln]
                Wp += shape_fn * outer_product2D(pointer, pointer)
                Bp += shape_fn * outer_product2D(gv, pointer)
            velocity_gradient = truncation(Bp @ Wp.inverse())
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps[materialID].update_particle_volume_2D(np, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_update_velocity_gradient_2DAxisy(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), 
                                            LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            velocity_gradient = ZEROMAT3x3
            offset = np * total_nodes
            position = particle[np].x
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                velocity_gradient0 = outer_product2D(dshape_fn, gv)
                velocity_gradient += mat3x3([[velocity_gradient0[0, 0], velocity_gradient0[0, 1], 0],
                                            [velocity_gradient0[1, 0], velocity_gradient0[1, 1], 0],
                                            [0, 0, shape_fn * gv[0] / position[0]]])
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps[materialID].update_particle_volume(np, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_update_velocity_gradient_bbar_2D(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), 
                                     LnID: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            velocity_gradient = ZEROMAT2x2
            strain_rate_trace = ZEROVEC3f
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                temp_dshape = (dshape_fnc - dshape_fn) / 3.

                average_bmatrix = temp_dshape[0] * gv[0] + temp_dshape[1] * gv[1]
                velocity_gradient += outer_product2D(dshape_fn, gv)
                velocity_gradient[0, 0] += average_bmatrix
                velocity_gradient[1, 1] += average_bmatrix

                strain_rate_trace[0] += dshape_fn[0] * gv[0]
                strain_rate_trace[1] += dshape_fn[1] * gv[1]
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps[materialID].update_particle_volume_bbar(np, strain_rate_trace, stateVars, dt)

@ti.kernel
def kernel_update_velocity_gradient_bbar_2DAxisy(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), 
                                     LnID: ti.template(), shapefn: ti.template(), shapefnc: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            one_three = 1./3.
            velocity_gradient = ZEROMAT3x3
            strain_rate_trace = ZEROVEC3f
            offset = np * total_nodes
            position = particle[np].x
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                shape_fn = shapefn[ln]
                shape_fnc = shapefnc[ln]
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                
                B0 = shape_fn / position[0]
                B1 = dshape_fn[0]
                B2 = dshape_fn[1]
                B0bar = shape_fnc / position[0]#((position[0] // grid_size[0]) + 0.5 * grid_size[0])
                B1bar = dshape_fnc[0]
                B2bar = dshape_fnc[1]

                velocity_gradient += mat3x3([one_three * ((B1bar + 2. * B1 + B0bar - B0) * gv[0] + (B2bar - B2) * gv[1]), B2 * gv[0], 0],
                                            [B1 * gv[1], one_three * ((B1bar - B1 + B0bar - B0) * gv[0] + (B2bar + 2. * B2) * gv[1]), 0],
                                            [0, 0, one_three * ((B1bar - B1 + B0bar + 2. * B0) * gv[0] + (B2bar - B2) * gv[1])])
                strain_rate_trace += vec3f(B1 * gv[0], B2 * gv[1], B0 * gv[0])
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps[materialID].update_particle_volume_bbar(np, strain_rate_trace, stateVars, dt)

@ti.kernel
def kernel_update_velocity_gradient_twophase(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(),  
                                             stateVars: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            solid_velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
            fluid_velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gvs = node[nodeID, bodyID].momentums
                gvf = node[nodeID, bodyID].momentumf
                dshape_fn = dshapefn[ln]
                solid_velocity_gradient += outer_product(dshape_fn, gvs)
                fluid_velocity_gradient += outer_product(dshape_fn, gvf)
            porosity, pvolume = particle[np].porosity, particle[np].vol
            pvolume *= matProps[materialID].update_particle_volume(np, solid_velocity_gradient, stateVars, dt)
            porosity = matProps[materialID].update_particle_porosity(solid_velocity_gradient, porosity, dt) 
            particle[np].mf = matProps[materialID].update_particle_fluid_mass(pvolume, porosity)
            particle[np].m = particle[np].ms + particle[np].mf
            particle[np].vol, particle[np].porosity = pvolume, porosity
            particle[np].solid_velocity_gradient = solid_velocity_gradient
            particle[np].fluid_velocity_gradient = fluid_velocity_gradient

@ti.kernel
def kernel_update_velocity_gradient_twophase_2D(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), 
                                                LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            solid_velocity_gradient = ZEROMAT2x2
            fluid_velocity_gradient = ZEROMAT2x2
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gvs = node[nodeID, bodyID].momentums
                gvf = node[nodeID, bodyID].momentumf
                dshape_fn = dshapefn[ln]
                solid_velocity_gradient += outer_product2D(dshape_fn, gvs)
                fluid_velocity_gradient += outer_product2D(dshape_fn, gvf)
            porosity, pvolume = particle[np].porosity, particle[np].vol
            pvolume *= matProps[materialID].update_particle_volume_2D(np, solid_velocity_gradient, stateVars, dt)
            porosity = matProps[materialID].update_particle_porosity(solid_velocity_gradient, porosity, dt)
            particle[np].mf = matProps[materialID].update_particle_fluid_mass(pvolume, porosity)
            particle[np].m = particle[np].ms + particle[np].mf
            particle[np].vol, particle[np].porosity = pvolume, porosity
            particle[np].solid_velocity_gradient = truncation(solid_velocity_gradient)
            particle[np].fluid_velocity_gradient = truncation(fluid_velocity_gradient)

@ti.kernel
def kernel_update_velocity_gradient_bbar_twophase_2D(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), 
                                             LnID: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            solid_velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
            fluid_velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
            strain_rate_trace = ZEROVEC3f
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gvs = node[nodeID, bodyID].momentums
                gvf = node[nodeID, bodyID].momentumf
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                solid_velocity_gradient += mat2x2([[0.5*(dshape_fn[0] + dshape_fnc[0]) * gvs[0], 0.5*(dshape_fnc[1] - dshape_fn[1]) * gvs[0]],
                                            [0.5*(dshape_fnc[0] - dshape_fn[0]) * gvs[1], 0.5*(dshape_fn[1] + dshape_fnc[1]) * gvs[1]]])
                fluid_velocity_gradient += mat2x2([[dshape_fnc[0] * gvf[0], 0],
                                            [0, dshape_fnc[1] * gvf[1]]])
                strain_rate_trace[0] += dshape_fn[0] * gvs[0]
                strain_rate_trace[1] += dshape_fn[1] * gvs[1]
            porosity, pvolume = particle[np].porosity, particle[np].vol
            pvolume *= matProps[materialID].update_particle_volume_bbar(np, strain_rate_trace, stateVars, dt)
            porosity = matProps[materialID].update_particle_porosity(solid_velocity_gradient, porosity, dt)
            particle[np].mf = matProps[materialID].update_particle_fluid_mass(pvolume, porosity)
            particle[np].m = particle[np].ms + particle[np].mf
            particle[np].vol, particle[np].porosity = pvolume, porosity
            particle[np].solid_velocity_gradient = solid_velocity_gradient
            particle[np].fluid_velocity_gradient = fluid_velocity_gradient

# ========================================================= #
#                           MUSL                            #
# ========================================================= #
@ti.kernel
def kernel_reset_grid_velocity(node: ti.template()):
    node.momentum.fill(0)

@ti.kernel
def kernel_reset_grid_velocity_twophase2D(node: ti.template()):
    node.momentum.fill(0)
    node.momentums.fill(0)
    node.momentumf.fill(0)

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

@ti.kernel
def kernel_postmapping_kinemaitc_twophase2D(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], particle[np].m)
                nmass_s = shape_mapping(shapefn[ln], particle[np].ms)
                nmass_f = shape_mapping(shapefn[ln], particle[np].mf)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * particle[np].v, nmass_s * particle[np].vs, nmass_f * particle[np].vf)

@ti.kernel
def kernel_calc_contact_normal_2DAxisy(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            position=particle[np].x
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                grad_domain = dshapefn[ln] * particle[np].vol / position[0]
                node[LnID[ln], bodyID]._update_nodal_grad_domain(grad_domain)

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
#               velocity gradient projection                #
# ========================================================= #
@ti.kernel
def kernel_dilatational_velocity_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        dil_gradv = trace(particle[np].velocity_gradient)
        volume = particle[np].vol
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            node[nodeID, bodyID].vol += shape_mapping(shape_fn, volume)
            node[nodeID, bodyID].jacobian += shape_mapping(shape_fn, dil_gradv * volume)
    
@ti.kernel
def kernel_gradient_velocity_projection_correction_2D(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(),  LnID: ti.template(), 
                                                      shapefn: ti.template(), node_size: ti.template(), matProps: ti.template(), stateVars: ti.template(), dt: ti.template()):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        materialID = int(particle[np].materialID)
        velocity_gradient = particle[np].velocity_gradient
        volume = particle[np].vol

        dil_gradv_bar = 0.
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            dil_gradv_bar += shape_fn * node[nodeID, bodyID].jacobian
        velocity_gradient += 1./2. * (dil_gradv_bar - trace(velocity_gradient)) * DELTA2D
        pressureAV = matProps[materialID].ComputePressure2D(np, stateVars, velocity_gradient, dt)
        particle[np].stress = matProps[materialID].ComputeShearStress2D(velocity_gradient)

        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            node[nodeID, bodyID].pressure += shape_fn * pressureAV * volume
        
@ti.kernel
def kernel_gradient_velocity_projection_correction(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(),  LnID: ti.template(), 
                                                   shapefn: ti.template(), node_size: ti.template(), matProps: ti.template(), stateVars: ti.template(), dt: ti.template()):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        materialID = int(particle[np].materialID)
        velocity_gradient = particle[np].velocity_gradient
        volume = particle[np].vol

        dil_gradv_bar = 0.
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            dil_gradv_bar += shape_fn * node[nodeID, bodyID].jacobian
        velocity_gradient += 1./3. * (dil_gradv_bar - trace(velocity_gradient)) * DELTA
        pressureAV = matProps[materialID].ComputePressure(np, stateVars, velocity_gradient, dt)
        particle[np].stress = matProps[materialID].ComputeShearStress(velocity_gradient)

        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            node[nodeID, bodyID].pressure += shape_fn * pressureAV * volume
        
@ti.kernel
def kernel_pressure_correction(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(),  LnID: ti.template(), 
                               shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        pressure_bar = 0.
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            pressure_bar += shape_fn * node[nodeID, bodyID].pressure 
        particle[np].stress -= pressure_bar * EYE

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
def kernel_calc_contact_displacement(total_nodes: int, particleNum: int, cutoff: float, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb].contact_pos.fill(0)
    
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                contact_pos = shapefn[ln] * particle[np].x * particle[np].m
                node[LnID[ln], bodyID].contact_pos += contact_pos

# ========================================================= #
#               Grid Based Contact Detection                #
# ========================================================= #
@ti.kernel
def kernel_assemble_contact_force(cutoff: float, dt: ti.template(), node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._contact_force_assemble(dt) 

################## MPM contact ##################
@ti.kernel
def kernel_calc_friction_contact(cut_off: float, mu: float, dt: ti.template(), is_rigid: ti.template(), node: ti.template()):
    # ti.block_local(dt)
    for ng in range(node.shape[0]):
        bodyID1, bodyID2 = 0, 1
        m1, m2 = node[ng, bodyID1].m, node[ng, bodyID2].m
        if m1 > cut_off and m2 > cut_off:
            mv1, mv2 = m1 * node[ng, bodyID1].momentum, m2 * node[ng, bodyID2].momentum
            norm1, norm2 = node[ng, bodyID1].grad_domain, node[ng, bodyID2].grad_domain

            norm, g_mass = ti.Vector.zero(float, GlobalVariable.DIMENSION), 0.
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
            if is_penetrate > Threshold:
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

################## Geo contact ##################
@ti.kernel
def kernel_calc_geocontact(cut_off: float, mu: float, alpha: float, beta: float, offset: float, gnum: ti.types.vector(GlobalVariable.DIMENSION, int), grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
                            dt: ti.template(), is_rigid: ti.template(), node: ti.template()):
    # ti.block_local(dt)
    for ng in range(node.shape[0]):
        bodyID1, bodyID2 = 0, 1
        m1, m2 = node[ng, bodyID1].m, node[ng, bodyID2].m
        if m1 > cut_off and m2 > cut_off:
            contact_pos1, contact_pos2 = node[ng, bodyID1].contact_pos / m1, node[ng, bodyID2].contact_pos / m2
            mv1, mv2 = m1 * node[ng, bodyID1].momentum, m2 * node[ng, bodyID2].momentum
            norm1, norm2 = node[ng, bodyID1].grad_domain, node[ng, bodyID2].grad_domain

            norm, g_mass, xm = ti.Vector.zero(float, GlobalVariable.DIMENSION), 0., ti.Vector.zero(float, GlobalVariable.DIMENSION)
            if is_rigid[bodyID1] == 1:
                norm = Normalize(norm1)
                g_mass = m1 * dt[None]
                xm = contact_pos1
            elif is_rigid[bodyID2] == 1:
                norm = -Normalize(norm2)
                g_mass = m2 * dt[None]
                xm = contact_pos2

            is_penetrate = (mv1 * m2 - m1 * mv2).dot(norm)
            is_contact = (contact_pos2 - contact_pos1).dot(norm) < MeanValue(offset * grid_size)
            if is_penetrate > Threshold and is_contact:
                inv_gmass = 1. / g_mass
                cforce = (mv1 * m2 - m1 * mv2) * inv_gmass

                ############# Geo-contact #############
                gsize = MeanValue(grid_size)
                node_coord = grid_size * ti.Vector(vectorize_id(ng, gnum), dt=int)
                dext = (xm - node_coord).dot(norm)

                # Reference: Hammerquist, C. C., Nairn, J. A., 2018. Modeling nanoindentation using the material point method. J. Mater. Res. 33, 1369-1381
                dist = 0.
                if dext <= 0:
                    dist = ti.abs(1. - 2. * (-dext / (1.25 * gsize)) ** 0.58)
                elif dext > 0.:
                    dist = ti.abs(2. * (dext / (1.25 * gsize)) ** 0.58 - 1.)

                # Reference: Gao L., Guo N., Yang Z. X., Jardine R. J., MPM modeling of pile installation in sand: Contact improvement and quantitative analysis. Comput. Geotech.
                factor = (1. - alpha * dist ** beta) / (1. + alpha * dist ** beta)

                norm_force = factor * is_penetrate * inv_gmass
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

################## DEM contact ##################
@ti.kernel
def kernel_calc_demcontact_2D(total_nodes: int, particleNum: int, grid_size: ti.types.vector(2, float), velocity: ti.types.vector(2, float), particle: ti.template(), matProps: ti.template(),
                              dt: ti.template(), polygon_vertices: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template(), node: ti.template()):
    # ti.block_local(dt)
    v1, v2 = velocity, ZEROVEC2f
    normal, tangential = ZEROVEC2f, ZEROVEC2f
    
    circle_radius = MeanValue(grid_size) * 0.75
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            circle_center = particle[np].x
            v2 = particle[np].v
            m2 = particle[np].m
            materialID = int(particle[np].materialID)
            k_normal, k_tangential, mu = 0., 0., 0.
            k_normal, k_tangential, mu = matProps[materialID].kn, matProps[materialID].kt, matProps[materialID].friction
            # Calculate the minimum distance and normal vector
            distance, normal = circle_polygon_distance(circle_center, polygon_vertices)
            # Calculate the particle traction using linear relationship-------------
            if distance > circle_radius:
                particle[np].contact_traction = vec2f([0.0, 0.0])
            else:
                delta = circle_radius - distance
                normal = Normalize(normal)
                tangential = (v1 - v2) - (v1 - v2).dot(normal) * normal
                tangential = Normalize(tangential)
                # -- damping--
                cforce = delta * k_normal * normal + 2.0 * ti.sqrt(m2 * k_normal) *  (v1 - v2).dot(normal) * normal
                if mu > Threshold:
                    val_fslip = mu * delta * k_normal
                    val_fstick = ti.sqrt(dot2(particle[np].contact_traction)) - (v1 - v2).dot(tangential) * k_tangential * dt[None]
                    val_fstick = ti.abs(val_fstick)
                    cforce += ti.min(val_fslip, val_fstick) * tangential
                particle[np].contact_traction = cforce

                bodyID = int(particle[np].bodyID)
                offset = np * total_nodes
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    extf = shape_mapping(shapefn[ln], cforce)
                    node[nodeID, bodyID]._update_contact_force(extf)

    for nver in range(polygon_vertices.shape[0]):
        polygon_vertices[nver] += velocity * dt[None]

@ti.func
def circle_polygon_distance(circle_center, polygon_vertices):
    num_vertices = polygon_vertices.shape[0]
    min_distance = 1.0e10
    normal_vector, closest_point = ZEROVEC2f, ZEROVEC2f
    p1, p2 = ZEROVEC2f, ZEROVEC2f
    for i in range(num_vertices):
        p1[0] = polygon_vertices[i][0]
        p1[1] = polygon_vertices[i][1]
        p2[0] = polygon_vertices[(i + 1) % num_vertices][0]
        p2[1] = polygon_vertices[(i + 1) % num_vertices][1]
        # Calculate the distance from the circle center to the edge
        edge_vector = p2 - p1
        point_vector = circle_center - p1
        edge_length = ti.sqrt(edge_vector[0]**2 + edge_vector[1]**2)
        edge_unit_vector = edge_vector / edge_length
        projection_length = point_vector.dot(edge_unit_vector)
        current_closest_point = p1 + projection_length * edge_unit_vector
        # Clamp the closest point to the edge segment
        if projection_length < 0.0:
            current_closest_point = p1
        elif projection_length > edge_length:
            current_closest_point = p2
        # Calculate the distance from the circle center to the closest point
        distance = ti.sqrt(dot2(circle_center - current_closest_point))
        # Update the minimum distance and normal vector
        if distance < min_distance:
            min_distance = distance
            closest_point = current_closest_point
            # Calculate the normal vector (perpendicular to the edge)
            normal_vector = vec2f([-edge_unit_vector[1], edge_unit_vector[0]])
            # Ensure the normal vector points towards the circle
            if (normal_vector).dot(circle_center - closest_point) < 0.0:
                normal_vector = -normal_vector
    # Check if the circle center is inside the polygon
    is_inside = False
    j = num_vertices - 1
    for i in range(num_vertices):
        if ((polygon_vertices[i][1] > circle_center[1]) != (polygon_vertices[j][1] > circle_center[1])) and \
           (circle_center[0] < (polygon_vertices[j][0] - polygon_vertices[i][0]) * (circle_center[1] - polygon_vertices[i][1]) / 
           (polygon_vertices[j][1] - polygon_vertices[i][1]) + polygon_vertices[i][0]):
            is_inside = not is_inside
        j = i
    # Adjust for penetration
    if is_inside:
        min_distance = -min_distance
        normal_vector = -normal_vector
    return min_distance, normal_vector

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

# ========================================================= #
#                    Particle shifting                      #
# ========================================================= #
@ti.kernel
def kernel_particle_shifting_delta_correction(total_nodes: int, particleNum: int, grid_size: ti.types.vector(3, float), node: ti.template(), particle: ti.template(), 
                                              LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template(), ):
    # refer to A.S. Baumgarten, K. Kamrin, Analysis and mitigation of spatial integration errors for the material point method, Internat. J. Numer. Methods Engrg. (2023).
    E2 = 0.
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].vol > Threshold:
                EI = ti.max(0, -grid_size[0] * grid_size[1] * grid_size[2] + node[ng, nb].vol)
                E2 += EI * EI

    den = 0.
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            grad_E2 = vec3f(0., 0., 0.)
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                EI = ti.max(0, -grid_size[0] * grid_size[1] * grid_size[2] + node[nodeID, bodyID].vol)
                grad_E2 += dshape_fn * EI
            grad_E2 *= 2. * particle[np].vol
            den += grad_E2.dot(grad_E2)
            particle[np].grad_E2 = grad_E2
    
    if den > 0.:
        for np in range(particleNum):
            particle[np].x -= E2 / den * particle[np].grad_E2

# ======================================== Implicit MPM ======================================== #
# ========================================================= #
#            Particle Momentum to Grid (iP2G)               #
# ========================================================= #
@ti.kernel
def kernel_mass_momentum_acceleration_force_ip2g(total_nodes: int, particleNum: int, gravity: ti.types.vector(3, float), node: ti.template(), particle: ti.template(), 
                                                 LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)): 
        ti.block_local(node.momentum.get_scalar_field(d))
        ti.block_local(node.inertia.get_scalar_field(d))
        ti.block_local(node.ext_force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            velocity = particle[np].v
            acceleration = particle[np].a
            fex = particle[np]._compute_external_force(gravity)
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                nmass = shape_mapping(shape_fn, mass)
                external_force = shape_mapping(shapefn[ln], fex)
                node[nodeID, bodyID]._update_nodal_mass(nmass)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * velocity)
                node[nodeID, bodyID]._update_nodal_acceleration(nmass * acceleration)
                node[nodeID, bodyID]._update_external_force(external_force)

@ti.kernel
def kernel_internal_force_p2g(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), 
                              LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
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
def kernel_internal_force_p2g_2D(total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), 
                                 LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                node[nodeID, bodyID]._update_internal_force(internal_force)

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

@ti.kernel
def kernel_compute_nodal_kinematics_newmark(beta: float, gamma: float, cutoff: float, node: ti.template(), dt: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._update_nodal_kinematic_newmark(beta, gamma, dt)

@ti.kernel
def kernel_update_nodal_disp(cutoff: float, node: ti.template(), unknown_vector: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = node[ng, nb].dof
                disp = vec3f(unknown_vector[dof0], unknown_vector[dof0 + 1], unknown_vector[dof0 + 2])
                node[ng, nb]._update_nodal_disp(disp)

@ti.kernel
def kernel_update_nodal_disp_2D(cutoff: float, node: ti.template(), unknown_vector: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = node[ng, nb].dof
                disp = vec2f(unknown_vector[dof0], unknown_vector[dof0 + 1])
                node[ng, nb]._update_nodal_disp(disp)

# ========================================================= #
#                 Grid to Particle (G2P)                    #
# ========================================================= #
@ti.kernel
def kernel_kinemaitc_ig2p(total_nodes: int, alpha: float, dt: ti.template(), particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            accleration = ti.Vector.zero(float, GlobalVariable.DIMENSION) 
            velocity = ti.Vector.zero(float, GlobalVariable.DIMENSION) 
            displacement = ti.Vector.zero(float, GlobalVariable.DIMENSION) 
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                velocity += shape_mapping(shape_fn, node[nodeID, bodyID].momentum)
                accleration += shape_mapping(shape_fn, node[nodeID, bodyID].inertia)
                displacement += shape_mapping(shape_fn, node[nodeID, bodyID].displacement)
            particle[np]._update_particle_state(dt, alpha, velocity, accleration, displacement)

# ========================================================= #
#                 Grid to Particle (G2P)                    #
# ========================================================= #
@ti.kernel
def kernel_compute_stress_strain_newmark_2D(dt: ti.template(), particleNum: int, particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), stiffness_matrix: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress0
            stress = matProps[materialID].ComputeStress2D(np, previous_stress, velocity_gradient, stateVars, dt)
            stiffness_matrix[np] = matProps[materialID].compute_stiffness_tensor(np, stress, stateVars)
            particle[np].stress = stress

@ti.kernel
def kernel_compute_stress_strain_newmark(dt: ti.template(), particleNum: int, particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), stiffness_matrix: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress0
            stress = matProps[materialID].ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
            stiffness_matrix[np] = matProps[materialID].compute_stiffness_tensor(np, stress, stateVars)
            particle[np].stress = stress

@ti.kernel
def kernel_update_stress_strain_newmark(particleNum: int, particle: ti.template(), dt: ti.template()):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            particle[np].stress0 = particle[np].stress
            particle[np].vol0 = particle[np].vol 


@ti.kernel
def kernel_update_displacement_gradient(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(),
                                        stateVars: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            displacement_gradient = ZEROMAT3x3
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gu = node[nodeID, bodyID].displacement
                dshape_fn = dshapefn[ln]
                displacement_gradient += outer_product(dshape_fn, gu)
            velocity_gradient = displacement_gradient / dt[None]
            particle[np].velocity_gradient = truncation(velocity_gradient)

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps[materialID].update_particle_volume(np, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_update_displacement_gradient_affine(total_nodes: int, particleNum: int, gnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), dt: ti.template(), node: ti.template(), particle: ti.template(), 
                                               matProps: ti.template(), stateVars: ti.template(),  LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            Wp = ZEROMAT3x3
            Bp = ZEROMAT3x3
            offset = np * total_nodes
            position = particle[np].x
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                grid_coord = grid_size * vec3f(vectorize_id(nodeID, gnum))
                pointer = grid_coord - position
                gu = node[nodeID, bodyID].displacement
                shape_fn = shapefn[ln]

                Wp += shape_fn * outer_product(pointer, pointer)
                Bp += shape_fn * outer_product(pointer, gu)
            velocity_gradient = truncation(Bp @ Wp.inverse()) / dt[None]
            particle[np].velocity_gradient = velocity_gradient

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps[materialID].update_particle_volume(np, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_update_displacement_gradient_bbar(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(),  
                                         stateVars: ti.template(), LnID: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            displacement_gradient = ZEROMAT3x3
            strain_incre_trace = ZEROVEC3f
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].displacement
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                temp_dshape = (dshape_fnc - dshape_fn) / 3.
                
                average_bmatrix = temp_dshape[0] * gv[0] + temp_dshape[1] * gv[1] + temp_dshape[2] * gv[2]
                displacement_gradient += outer_product(dshape_fn, gv)
                displacement_gradient[0, 0] += average_bmatrix
                displacement_gradient[1, 1] += average_bmatrix
                displacement_gradient[2, 2] += average_bmatrix

                strain_incre_trace[0] += dshape_fn[0] * gv[0]
                strain_incre_trace[1] += dshape_fn[1] * gv[1]
                strain_incre_trace[2] += dshape_fn[2] * gv[2]
            particle[np].velocity_gradient = truncation(displacement_gradient) / dt[None]

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps[materialID].update_particle_volume_bbar(np, strain_incre_trace / dt[None], stateVars, dt)

@ti.kernel
def kernel_update_displacement_gradient_2D(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), 
                                LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            displacement_gradient = ZEROMAT2x2
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gu = node[nodeID, bodyID].displacement
                dshape_fn = dshapefn[ln]
                displacement_gradient += outer_product2D(dshape_fn, gu)
            velocity_gradient = displacement_gradient / dt[None]
            particle[np].velocity_gradient = truncation(velocity_gradient)

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps[materialID].update_particle_volume_2D(np, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_update_displacement_gradient_affine_2D(total_nodes: int, particleNum: int, gnum: ti.types.vector(2, int), grid_size: ti.types.vector(2, float), dt: ti.template(), node: ti.template(), particle: ti.template(), 
                                       matProps: ti.template(), stateVars: ti.template(),  LnID: ti.template(), shapefn: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            Wp = ZEROMAT2x2
            Bp = ZEROMAT2x2
            offset = np * total_nodes
            position = particle[np].x
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                grid_coord = grid_size * vec2f(vectorize_id(nodeID, gnum))
                pointer = grid_coord - position
                gu = node[nodeID, bodyID].displacement
                shape_fn = shapefn[ln]
                Wp += shape_fn * outer_product2D(pointer, pointer)
                Bp += shape_fn * outer_product2D(gu, pointer)
            velocity_gradient = truncation(Bp @ Wp.inverse()) / dt[None]
            particle[np].velocity_gradient = truncation(velocity_gradient)

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps[materialID].update_particle_volume_2D(np, velocity_gradient, stateVars, dt)

@ti.kernel
def kernel_update_displacement_gradient_bbar_2D(total_nodes: int, particleNum: int, dt: ti.template(), node: ti.template(), particle: ti.template(), matProps: ti.template(), stateVars: ti.template(), 
                                     LnID: ti.template(), dshapefn: ti.template(), dshapefnc: ti.template(), node_size: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            displacement_gradient = ZEROMAT2x2
            strain_incre_trace = ZEROVEC3f
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                temp_dshape = (dshape_fnc - dshape_fn) / 3.

                average_bmatrix = temp_dshape[0] * gv[0] + temp_dshape[1] * gv[1]
                displacement_gradient += outer_product2D(dshape_fn, gv)
                displacement_gradient[0, 0] += average_bmatrix
                displacement_gradient[1, 1] += average_bmatrix

                strain_incre_trace[0] += dshape_fn[0] * gv[0]
                strain_incre_trace[1] += dshape_fn[1] * gv[1]
            velocity_gradient = displacement_gradient / dt[None]
            particle[np].velocity_gradient = truncation(velocity_gradient)

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps[materialID].update_particle_volume_bbar(np, strain_incre_trace / dt[None], stateVars, dt)

# ========================================================= #
#                  Incompressible flows                     #
# ========================================================= #
@ti.kernel
def init_boundary(cnum: ti.types.vector(3, float), cell_type: ti.template()):         
    for I in ti.grouped(cell_type):
        if any(I <= 0) or any(I >= cnum - 1):
            cell_type[I] = 2    

@ti.kernel
def init_boundary2D(cnum: ti.types.vector(2, float), cell_type: ti.template()):         
    for I in ti.grouped(cell_type):
        if any(I <= 0) or any(I >= cnum - 1):
            cell_type[I] = 2    

@ti.func
def is_fluid(cell_type):
    return int(cell_type) == 1

@ti.func
def is_solid(cell_type):
    return int(cell_type) == 2

@ti.func
def is_air(cell_type):
    return int(cell_type) == 0

@ti.func
def get_node_id(index, gnum):
    return (linearize(index, gnum), 0)

@ti.kernel
def kernel_find_fluid_domain2D(particleNum: int, cnum: ti.types.vector(2, int), grid_size: ti.types.vector(2, float), cell_type: ti.template(), particle: ti.template()):
    for I in ti.grouped(ti.ndrange((1, cnum[0] - 1), (1, cnum[1] - 1))):
        if not is_solid(cell_type[I]):
            cell_type[I] = 0
    for np in range(particleNum):
        idx = ti.cast(ti.floor(particle[np].x / grid_size), int)
        if not is_solid(cell_type[idx]):
            cell_type[idx] = 1

@ti.kernel
def kernel_find_fluid_domain(particleNum: int, cnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), cell_type: ti.template(), particle: ti.template()):
    for I in ti.grouped(ti.ndrange((1, cnum[0] - 1), (1, cnum[1] - 1), (1, cnum[2] - 1))):
        if not is_solid(cell_type[I]):
            cell_type[I] = 0
    for np in range(particleNum):
        idx = ti.cast(ti.floor(particle[np].x / grid_size), int)
        if not is_solid(cell_type[idx]):
            cell_type[idx] = 1

@ti.kernel
def kernel_compute_grid_velocity_gravity(cutoff: float, gravity: ti.types.vector(3, float), dt: ti.template(), node: ti.template()):
    for I in ti.grouped(node):
        if node[I].m > cutoff:
            node[I]._compute_nodal_velocity()
            node[I]._add_gravity(gravity, dt)

@ti.kernel
def kernel_enforce_boundary2D(gnum: ti.types.vector(2, int), cell_type: ti.template(), node: ti.template()):
    for I in ti.grouped(cell_type):
        if is_solid(cell_type[I]):
            if I[0] <= 0 or I[0] >= gnum[0] - 2:
                node[get_node_id(I, gnum)].momentum[0] = 0
                node[get_node_id(I + vec2i(1, 0), gnum)].momentum[0] = 0
                node[get_node_id(I + vec2i(0, 1), gnum)].momentum[0] = 0
                node[get_node_id(I + vec2i(1, 1), gnum)].momentum[0] = 0
            elif I[1] <= 0 or I[1] >= gnum[1] - 2:
                node[get_node_id(I, gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(1, 0), gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(0, 1), gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(1, 1), gnum)].momentum[1] = 0
            else:
                node[get_node_id(I, gnum)].momentum = 0
                node[get_node_id(I + vec2i(1, 0), gnum)].momentum = 0
                node[get_node_id(I + vec2i(0, 1), gnum)].momentum = 0
                node[get_node_id(I + vec2i(1, 1), gnum)].momentum = 0

@ti.kernel
def kernel_enforce_boundary3D(gnum: ti.types.vector(3, int), cell_type: ti.template(), node: ti.template()):
    for I in ti.grouped(cell_type):
        if is_solid(cell_type[I]):
            if I[0] <= 0 or I[0] >= gnum[0] - 2:
                node[get_node_id(I, gnum)].momentum[0] = 0
                node[get_node_id(I + vec2i(1, 0), gnum)].momentum[0] = 0
                node[get_node_id(I + vec2i(0, 1), gnum)].momentum[0] = 0
                node[get_node_id(I + vec2i(1, 1), gnum)].momentum[0] = 0
            elif I[1] <= 0 or I[1] >= gnum[1] - 2:
                node[get_node_id(I, gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(1, 0), gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(0, 1), gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(1, 1), gnum)].momentum[1] = 0
            elif I[2] <= 0 or I[1] >= gnum[2] - 2:
                node[get_node_id(I, gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(1, 0), gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(0, 1), gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(1, 1), gnum)].momentum[1] = 0
            else:
                node[get_node_id(I, gnum)].momentum = 0
                node[get_node_id(I + vec2i(1, 0), gnum)].momentum = 0
                node[get_node_id(I + vec2i(0, 1), gnum)].momentum = 0
                node[get_node_id(I + vec2i(1, 1), gnum)].momentum = 0

@ti.kernel
def kernel_fill_atmospheric_pressure2D(cnum: ti.types.vector(2, int), pressure: ti.template(), cell_type: ti.template(), matProps: ti.template()):
    for I in ti.grouped(ti.ndrange((1, cnum[0] - 1), (1, cnum[1] - 1))):
        if is_air(cell_type[I]):
            pressure[I] = matProps[1].atmospheric_pressure

@ti.kernel
def kernel_fill_atmospheric_pressure3D(cnum: ti.types.vector(3, int), pressure: ti.template(), cell_type: ti.template(), matProps: ti.template()):
    for I in ti.grouped(ti.ndrange((1, cnum[0] - 1), (1, cnum[1] - 1), (1, cnum[2] - 1))):
        if is_air(cell_type[I]):
            pressure[I] = matProps[1].atmospheric_pressure

@ti.kernel
def kernel_correct_velocity2D(cnum: ti.types.vector(2, int), igrid_size: ti.types.vector(2, float), dt: ti.template(), matProps: ti.template(), pressure: ti.template(), cell_type: ti.template(), node: ti.template()):
    scale = dt[None] / matProps[1].density
    gnum = cnum + 1
    for I in ti.grouped(pressure):
        print(I, pressure[I])
    for I in ti.grouped(ti.ndrange((1, cnum[0]), (1, cnum[1]))):
        I_10 = I - vec2i(1, 0)
        I_01 = I - vec2i(0, 1)
        I_11 = I - vec2i(1, 1)
        cell_type_00 = cell_type[I]
        cell_type_10 = cell_type[I_10]
        cell_type_01 = cell_type[I_01]

        if is_fluid(cell_type_10) or is_fluid(cell_type_00):
            if is_solid(cell_type_10) or is_solid(cell_type_00): 
                node[get_node_id(I, gnum)].momentum[0] = 0
                node[get_node_id(I + vec2i(0, 1), gnum)].momentum[0] = 0
            else:
                node[get_node_id(I, gnum)].momentum[0] -= 0.5 * scale * (pressure[I] - pressure[I_10] + pressure[I_01] - pressure[I_11]) * igrid_size[0]

        if is_fluid(cell_type_01) or is_fluid(cell_type_00):
            if is_solid(cell_type_01) or is_solid(cell_type_00): 
                node[get_node_id(I, gnum)].momentum[1] = 0
                node[get_node_id(I + vec2i(1, 0), gnum)].momentum[1] = 0
            else:
                node[get_node_id(I, gnum)].momentum[1] -= 0.5 * scale * (pressure[I] - pressure[I_01] + pressure[I_10] - pressure[I_11]) * igrid_size[1]
