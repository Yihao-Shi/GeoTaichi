import taichi as ti

from src.physics_model.consititutive_model.MaterialKernel import get_angular_velocity
from src.utils.constants import (
    Threshold,
    ZEROMAT4x4,
    ZEROMAT6x3,
    ZEROVEC2f,
    ZEROVEC3f,
    ZEROMAT2x2,
    ZEROMAT3x3,
    DELTA2D,
    DELTA,
    EYE,
)
from src.utils.MatrixFunction import truncation, trace
from src.utils.ScalarFunction import vectorize_id, linearize, sgn
from src.utils.ShapeFunctions import (
    ShapeLinear,
    GShapeLinear,
    ShapeLinearCenter,
    ShapeGIMP,
    GShapeGIMP,
    ShapeGIMPCenter,
    ShapeBsplineQ,
    GShapeBsplineQ,
    ShapeBsplineC,
    GShapeBsplineC,
)
from src.utils.Quaternion import SetDQ, SetToRotate
from src.utils.TypeDefination import (
    real,
    vec2f,
    vec3f,
    vec4f,
    vec6f,
    vec8f,
    vec12f,
    mat3x3,
    mat4x4,
    vec2i,
    vec3i,
    mat2x2,
    vec4i,
    vec8i,
)
from src.utils.VectorFunction import Normalize, outer_product, MeanValue, Squared, outer_product2D, dot2
from src.mpm.sparse_grid.BlockSparseGrid import compact_node_grid_coord
from src.levelset.FluidLevelSetKernel import (
    min_grid_spacing,
    grid_cell_volume,
    equivalent_particle_radius,
    free_surface_theta,
    smoothed_heaviside as smoothed_fluid_heaviside,
    clamp_cell_index as clamp_offset_cell_index,
    surface_curvature as fluid_surface_curvature,
    sample_cell_centered_sdf as sample_cell_centered_fluid_sdf,
)
import src.utils.GlobalVariable as GlobalVariable


@ti.func
def shape_mapping(shape_fn, vars):
    return shape_fn * vars


@ti.func
def gradshape_vector_mapping(bmatrix, vecs):
    return vec6f(
        [
            bmatrix[0, 0] * vecs[0] + bmatrix[0, 1] * vecs[1] + bmatrix[0, 2] * vecs[2],
            bmatrix[1, 0] * vecs[0] + bmatrix[1, 1] * vecs[1] + bmatrix[1, 2] * vecs[2],
            bmatrix[2, 0] * vecs[0] + bmatrix[2, 1] * vecs[1] + bmatrix[2, 2] * vecs[2],
            bmatrix[3, 0] * vecs[0] + bmatrix[3, 1] * vecs[1],
            bmatrix[4, 1] * vecs[1] + bmatrix[4, 2] * vecs[2],
            bmatrix[5, 0] * vecs[0] + bmatrix[5, 2] * vecs[2],
        ]
    )


@ti.func
def gradshape_scalar_mapping(bmatrix, scalar):
    vec1 = bmatrix[0, 0] * scalar
    vec2 = bmatrix[1, 1] * scalar
    vec3 = bmatrix[2, 2] * scalar
    return vec3f([vec1, vec2, vec3])


@ti.func
def MeanStress(stress):
    return (stress[0] + stress[1] + stress[2]) / 3.0


@ti.func
def bbar_velocity_gradient_2d(grid_velocity, dshape_fn, dshape_fnc):
    temp_dshape = 0.5 * (dshape_fnc - dshape_fn)
    average_bmatrix = temp_dshape.dot(grid_velocity)
    velocity_gradient = outer_product2D(grid_velocity, dshape_fn)
    velocity_gradient[0, 0] += average_bmatrix
    velocity_gradient[1, 1] += average_bmatrix
    return velocity_gradient


@ti.func
def bbar_internal_force_2d(dshape_fn, dshape_fnc, internal_stress):
    temp_dshape = 0.5 * (dshape_fnc - dshape_fn)
    return vec2f(
        [
            (dshape_fn[0] + temp_dshape[0]) * internal_stress[0]
            + temp_dshape[0] * internal_stress[1]
            + dshape_fn[1] * internal_stress[3],
            temp_dshape[1] * internal_stress[0]
            + (dshape_fn[1] + temp_dshape[1]) * internal_stress[1]
            + dshape_fn[0] * internal_stress[3],
        ]
    )


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
            node[ng, nb].m = 0.0


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


@ti.kernel
def sdf_constraint(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    sdf: ti.template(),
):
    pass


# ======================================== Explicit MPM ======================================== #
@ti.kernel
def lightweight_p2g(
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    gravity: ti.types.vector(3, float),
    particle_lengths: ti.template(),
    boundary_types: ti.template(),
    node: ti.template(),
    particle: ti.template(),
):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
        for offset in ti.static(ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION)))):
            base = ti.floor((position - psize) * igrid_size).cast(int)
            grid_id = base + offset
            if all(grid_id >= 0) and all(grid_id < gnum):
                nodeID = linearize(grid_id, gnum)
                grid_pos = grid_id * grid_size
                shape_fn, dshape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(
                    float, GlobalVariable.DIMENSION
                )
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

                dpos = grid_pos - position
                weight = 1.0
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    weight *= shape_fn[d]
                pforce = weight * external_force

                if ti.static(GlobalVariable.DIMENSION == 2):
                    weight_grad = ti.Vector([dshape_fn[0] * shape_fn[1], shape_fn[0] * dshape_fn[1]])
                    if ti.static(GlobalVariable.BBAR):
                        weight_gradc = vec3f([dshape_fn[0] * shape_fnc[1], shape_fnc[0] * dshape_fn[1]])
                        temp_dshape = 0.5 * (weight_gradc - weight_grad)
                        pforce += vec2f(
                            [
                                (weight_grad[0] + temp_dshape[0]) * internal_force[0]
                                + temp_dshape[0] * internal_force[1]
                                + temp_dshape[0] * internal_force[2]
                                + weight_grad[1] * internal_force[3],
                                temp_dshape[1] * internal_force[0]
                                + (weight_grad[1] + temp_dshape[1]) * internal_force[1]
                                + temp_dshape[1] * internal_force[2]
                                + weight_grad[0] * internal_force[3],
                            ]
                        )
                    else:
                        pforce += vec2f(
                            [
                                weight_grad[0] * internal_force[0] + weight_grad[1] * internal_force[3],
                                weight_grad[1] * internal_force[1] + weight_grad[0] * internal_force[3],
                            ]
                        )
                elif ti.static(GlobalVariable.DIMENSION == 3):
                    weight_grad = ti.Vector(
                        [
                            dshape_fn[0] * shape_fn[1] * shape_fn[2],
                            shape_fn[0] * dshape_fn[1] * shape_fn[2],
                            shape_fn[0] * shape_fn[1] * dshape_fn[2],
                        ]
                    )
                    if ti.static(GlobalVariable.BBAR):
                        weight_gradc = vec3f(
                            [
                                dshape_fn[0] * shape_fnc[1] * shape_fnc[2],
                                shape_fnc[0] * dshape_fn[1] * shape_fnc[2],
                                shape_fnc[0] * shape_fnc[1] * dshape_fn[2],
                            ]
                        )
                        temp_dshape = (weight_gradc - weight_grad) / 3.0
                        pforce += vec3f(
                            [
                                (weight_grad[0] + temp_dshape[0]) * internal_force[0]
                                + temp_dshape[0] * internal_force[1]
                                + temp_dshape[0] * internal_force[2]
                                + weight_grad[1] * internal_force[3]
                                + weight_grad[2] * internal_force[5],
                                temp_dshape[1] * internal_force[0]
                                + (weight_grad[1] + temp_dshape[1]) * internal_force[1]
                                + temp_dshape[1] * internal_force[2]
                                + weight_grad[0] * internal_force[3]
                                + weight_grad[2] * internal_force[4],
                                temp_dshape[2] * internal_force[0]
                                + temp_dshape[2] * internal_force[1]
                                + (weight_grad[2] + temp_dshape[2]) * internal_force[2]
                                + weight_grad[1] * internal_force[4]
                                + weight_grad[0] * internal_force[5],
                            ]
                        )
                    else:
                        pforce += vec3f(
                            [
                                weight_grad[0] * internal_force[0]
                                + weight_grad[1] * internal_force[3]
                                + weight_grad[2] * internal_force[5],
                                weight_grad[1] * internal_force[1]
                                + weight_grad[0] * internal_force[3]
                                + weight_grad[2] * internal_force[4],
                                weight_grad[2] * internal_force[2]
                                + weight_grad[1] * internal_force[4]
                                + weight_grad[0] * internal_force[5],
                            ]
                        )
                nmass = weight * p_mass
                momentum = nmass * velocity
                if ti.static(GlobalVariable.APIC or GlobalVariable.TPIC):
                    momentum += nmass * velocity_gradient @ dpos
                node[nodeID, bodyID].m += nmass
                node[nodeID, bodyID].momentum += momentum
                node[nodeID, bodyID].force += pforce


@ti.kernel
def lightweight_g2p(
    particleNum: int,
    alpha: float,
    cutoff: float,
    fraction: float,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    particle_lengths: ti.template(),
    boundary_types: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        # np = particleID[pid]
        bodyID = int(particle[np].bodyID)
        materialID = int(particle[np].materialID)
        position = particle[np].x
        vFLIP = particle[np].v
        psize = particle_lengths[bodyID]

        vPIC = ti.Vector.zero(float, GlobalVariable.DIMENSION)
        Wp = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
        velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
        velocity_gradient_bar = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
        for offset in ti.static(ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION)))):
            base = ti.floor((position - psize) * igrid_size).cast(int)
            grid_id = base + offset
            if all(grid_id >= 0) and all(grid_id < gnum):
                nodeID = linearize(grid_id, gnum)
                grid_pos = grid_id * grid_size
                shape_fn, dshape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(
                    float, GlobalVariable.DIMENSION
                )
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
                        velocity_gradient += weight * outer_product2D(velocity, dpos)
                    elif ti.static(GlobalVariable.DIMENSION == 3):
                        Wp += weight * outer_product(dpos, dpos)
                        velocity_gradient += weight * outer_product(velocity, dpos)
                else:
                    if ti.static(GlobalVariable.DIMENSION == 2):
                        weight_grad = ti.Vector([dshape_fn[0] * shape_fn[1], shape_fn[0] * dshape_fn[1]])
                        velocity_gradient_increment = outer_product2D(velocity, weight_grad)
                        velocity_gradient += velocity_gradient_increment
                        if ti.static(GlobalVariable.BBAR):
                            weight_gradc = vec3f([dshape_fn[0] * shape_fnc[1], shape_fnc[0] * dshape_fn[1]])
                            temp_dshape = 0.5 * (weight_gradc - weight_grad)
                            average_bmatrix = temp_dshape[0] * velocity[0] + temp_dshape[1] * velocity[1]
                            velocity_gradient_bar += velocity_gradient_increment
                            velocity_gradient_bar[0, 0] += average_bmatrix
                            velocity_gradient_bar[1, 1] += average_bmatrix
                    elif ti.static(GlobalVariable.DIMENSION == 3):
                        weight_grad = ti.Vector(
                            [
                                dshape_fn[0] * shape_fn[1] * shape_fn[2],
                                shape_fn[0] * dshape_fn[1] * shape_fn[2],
                                shape_fn[0] * shape_fn[1] * dshape_fn[2],
                            ]
                        )
                        velocity_gradient_increment = outer_product(velocity, weight_grad)
                        velocity_gradient += velocity_gradient_increment
                        if ti.static(GlobalVariable.BBAR):
                            weight_gradc = vec3f(
                                [
                                    dshape_fn[0] * shape_fnc[1] * shape_fnc[2],
                                    shape_fnc[0] * dshape_fn[1] * shape_fnc[2],
                                    shape_fnc[0] * shape_fnc[1] * dshape_fn[2],
                                ]
                            )
                            temp_dshape = (weight_gradc - weight_grad) / 3.0
                            average_bmatrix = (
                                temp_dshape[0] * velocity[0]
                                + temp_dshape[1] * velocity[1]
                                + temp_dshape[2] * velocity[2]
                            )
                            velocity_gradient_bar += velocity_gradient_increment
                            velocity_gradient_bar[0, 0] += average_bmatrix
                            velocity_gradient_bar[1, 1] += average_bmatrix
                            velocity_gradient_bar[2, 2] += average_bmatrix

        if ti.static(not GlobalVariable.BBAR):
            velocity_gradient_bar = velocity_gradient

        if ti.static(GlobalVariable.APIC):
            velocity_gradient = Wp.inverse() @ velocity_gradient

        particle[np].velocity_gradient = velocity_gradient_bar
        particle[np].v = alpha * vPIC + (1.0 - alpha) * vFLIP
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
            djacobian = (
                ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity_gradient_bar
            ).determinant()
            transfer_var = mass * djacobian
            for offset in ti.static(
                ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION)))
            ):
                base = ti.floor((position - psize) * igrid_size).cast(int)
                grid_id = base + offset
                if all(grid_id >= 0) and all(grid_id < gnum):
                    nodeID = linearize(grid_id, gnum)
                    grid_pos = grid_id * grid_size
                    shape_fn, dshape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(
                        float, GlobalVariable.DIMENSION
                    )
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
            djacobian = (
                ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity_gradient_bar
            ).determinant()
            transfer_var = mass * djacobian
            djacobian_bar = 0.0
            for offset in ti.static(
                ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION)))
            ):
                base = ti.floor((position - psize) * igrid_size).cast(int)
                grid_id = base + offset
                if all(grid_id >= 0) and all(grid_id < gnum):
                    nodeID = linearize(grid_id, gnum)
                    grid_pos = grid_id * grid_size
                    shape_fn, dshape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(
                        float, GlobalVariable.DIMENSION
                    )
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
            djacobian_bar_new = fraction * djacobian_bar + (1.0 - fraction) * djacobian

            multiplier = (djacobian_bar_new / djacobian) ** (1.0 / GlobalVariable.DIMENSION)
            updated_velocity_gradient = (multiplier - 1.0) * ti.Matrix.identity(float, GlobalVariable.DIMENSION) / dt[
                None
            ] + multiplier * velocity_gradient

            particle[np].velocity_gradient = updated_velocity_gradient

    if ti.static(GlobalVariable.PARTICLESHIFTING):
        pass


@ti.kernel
def lightweight_grid_operation(cutoff: float, damp: float, node: ti.template(), dt: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                mass = node[ng, nb].m
                acceleration = node[ng, nb].force / mass
                velocity = node[ng, nb].momentum / mass
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    if velocity[d] * acceleration[d] > 0.0:
                        acceleration[d] -= damp * ti.abs(acceleration[d]) * sgn(velocity[d])
                node[ng, nb].momentum = velocity + acceleration * dt[None]
                node[ng, nb].force = acceleration


@ti.kernel
def g2p2g(
    particleNum: int,
    p_mass: float,
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    gravity: ti.types.vector(3, float),
    dt: ti.template(),
    node_in: ti.template(),
    node_out: ti.template(),
    particle: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    particleID: ti.template(),
):
    for np in range(particleNum):
        pid = particleID[np]
        position = particle[pid].x
        velocity = particle[pid].v
        base = ti.floor(position * igrid_size - 0.5).cast(int)
        fx = position * igrid_size - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, GlobalVariable.DIMENSION)
        velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
        # Loop over 3x3 grid node neighborhood
        for offset in ti.static(ti.grouped(ti.ndrange(*((3,) * GlobalVariable.DIMENSION)))):
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
        for offset in ti.static(ti.grouped(ti.ndrange(*((3,) * GlobalVariable.DIMENSION)))):
            dpos = (offset.cast(float) - fx) * grid_size
            weight = 1.0
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                weight *= w2[offset[d]][d]

            pforce = particle[np]._compute_external_force(gravity)
            if ti.static(GlobalVariable.DIMENSION == 2):
                weight_grad = (
                    ti.Vector([w_grad[offset[0]][0] * w[offset[1]][1], w[offset[0]][0] * w_grad[offset[1]][1]])
                    * igrid_size
                )
                pforce += vec2f(
                    [
                        weight_grad[0] * fInt[0] + weight_grad[1] * fInt[3],
                        weight_grad[1] * fInt[1] + weight_grad[0] * fInt[3],
                    ]
                ) + p_mass * vec2f(gravity[0], gravity[1])
            elif ti.static(GlobalVariable.DIMENSION == 3):
                weight_grad = (
                    ti.Vector(
                        [
                            w_grad[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2],
                            w[offset[0]][0] * w_grad[offset[1]][1] * w[offset[2]][2],
                            w[offset[0]][0] * w[offset[1]][1] * w_grad[offset[2]][2],
                        ]
                    )
                    * igrid_size
                )
                pforce += (
                    vec3f(
                        [
                            weight_grad[0] * fInt[0] + weight_grad[1] * fInt[3] + weight_grad[2] * fInt[5],
                            weight_grad[1] * fInt[1] + weight_grad[0] * fInt[3] + weight_grad[2] * fInt[4],
                            weight_grad[2] * fInt[2] + weight_grad[1] * fInt[4] + weight_grad[0] * fInt[5],
                        ]
                    )
                    + p_mass * gravity
                )
            node_out[base + offset, 0].momentum += weight * (p_mass * velocity + velocity_gradient @ dpos + pforce)
            node_out[base + offset, 0].mass += weight * p_mass


# ========================================================= #
#                  Moving Least Square                      #
# ========================================================= #
@ti.func
def polynomial(position):
    return vec4f(1.0, position[0], position[1], position[2])


@ti.func
def iMomentMatrix(xp, xg, igrid_size, kernel_function: ti.template()):
    w = kernel_function(xp, xg, igrid_size)
    return (
        w
        * mat4x4(
            [
                [1, xg[0], xg[1], xg[2]],
                [xg[0], xg[0] * xg[0], xg[0] * xg[1], xg[0] * xg[2]],
                [xg[1], xg[1] * xg[0], xg[1] * xg[1], xg[1] * xg[2]],
                [xg[2], xg[2] * xg[0], xg[2] * xg[1], xg[2] * xg[2]],
            ]
        ).inverse()
    )


# ========================================================= #
#             Particle Momentum to Grid (P2G)               #
# ========================================================= #
@ti.kernel
def kernel_angular_velocity_p2c(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
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
def kernel_volume_p2c(
    cnum: ti.types.vector(3, int),
    inv_dx: ti.types.vector(3, float),
    particleNum: int,
    cell: ti.template(),
    particle: ti.template(),
):
    ti.block_local(cell.volume)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            cellID = ti.floor(particle[np].x * inv_dx, int)
            linear_cellID = linearize(cellID, cnum)
            cell[linear_cellID, bodyID]._update_cell_volume(particle[np].vol)


@ti.kernel
def kernel_volume_p2c_2D(
    cnum: ti.types.vector(2, int),
    inv_dx: ti.types.vector(2, float),
    particleNum: int,
    cell: ti.template(),
    particle: ti.template(),
):
    ti.block_local(cell.volume)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            cellID = ti.floor(particle[np].x * inv_dx, int)
            linear_cellID = linearize(cellID, cnum)
            cell[linear_cellID, bodyID]._update_cell_volume(particle[np].vol)


@ti.kernel
def kernel_mass_p2g(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
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
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)


@ti.kernel
def kernel_momentum_p2g(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
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
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)


@ti.kernel
def kernel_mass_momentum_p2g(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
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
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)


@ti.kernel
def kernel_mass_momentum_taylor_p2g(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
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
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)


@ti.kernel
def kernel_mass_momentum_taylor_p2g_sparse(
    total_nodes: int,
    particleNum: int,
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    active_block_ids: ti.template(),
):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
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
                nodal_coord = grid_size * ti.cast(
                    compact_node_grid_coord(nodeID, block_count, block_size, block_volume, active_block_ids), float
                )
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_mass(nmass)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * (velocity + gradv @ (nodal_coord - xp)))


@ti.kernel
def kernel_mass_momentum_taylor_p2g_2DAxisy(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            xp = particle[np].x
            mass = particle[np].m
            velocity = particle[np].v
            velocity_gradient = particle[np].velocity_gradient
            gradv = mat2x2(
                [velocity_gradient[0, 0], velocity_gradient[0, 1]], [velocity_gradient[1, 0], velocity_gradient[1, 1]]
            )
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nodal_coord = grid_size * ti.Vector(vectorize_id(nodeID, gnum))
                nmass = shape_mapping(shapefn[ln], mass)
                node[nodeID, bodyID]._update_nodal_mass(nmass)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * (velocity + gradv @ (nodal_coord - xp)))
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * velocity)


@ti.kernel
def kernel_mass_momentum_p2g_phasefield(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            velocity = particle[np].v
            pf = particle[np].pf
            viscous = particle[np].viscous
            vol = particle[np].vol
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                nviscous = shape_mapping(shapefn[ln], viscous)
                node[nodeID, bodyID]._update_nodal_scale_pf(nmass, pf * nmass, nviscous * vol)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * velocity)


@ti.kernel
def kernel_mass_momentum_p2g_twophase(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
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
                node[nodeID, bodyID]._update_nodal_mass(nmass, nmass_s, nmass_f)
                node[nodeID, bodyID]._update_nodal_momentum(
                    nmass * velocity, nmass_s * velocity_s, nmass_f * velocity_f
                )
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass, nmass_s, nmass_f)
                            node[tideID, bodyID]._update_nodal_momentum(
                                nmass * velocity, nmass_s * velocity_s, nmass_f * velocity_f
                            )
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass, nmass_s, nmass_f)
                            node[tideID, bodyID]._update_nodal_momentum(
                                nmass * velocity, nmass_s * velocity_s, nmass_f * velocity_f
                            )
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_mass(nmass, nmass_s, nmass_f)
                            node[tideID, bodyID]._update_nodal_momentum(
                                nmass * velocity, nmass_s * velocity_s, nmass_f * velocity_f
                            )


@ti.kernel
def kernel_mass_momentum_taylor_p2g_twophase(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    ti.block_local(node.m)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
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
            xp = particle[np].x
            gradv = particle[np].velocity_gradient
            grdv_f = particle[np].velocity_gradientf
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nodal_coord = grid_size * vectorize_id(nodeID, gnum)
                xip = nodal_coord - xp
                nmass = shape_mapping(shapefn[ln], mass)
                nmass_s = shape_mapping(shapefn[ln], mass_s)
                nmass_f = shape_mapping(shapefn[ln], mass_f)
                node[nodeID, bodyID]._update_nodal_mass(nmass, nmass_s, nmass_f)
                node[nodeID, bodyID]._update_nodal_momentum(
                    nmass * (velocity + gradv @ xip),
                    nmass_s * (velocity_s + gradv @ xip),
                    nmass_f * (velocity_f + grdv_f @ xip),
                )


@ti.kernel
def kernel_assemble_dem_contact_force(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex = particle[np].contact_traction
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                cforce = shape_mapping(shapefn[ln], fex)
                node[nodeID, bodyID]._update_contact_force(cforce)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_contact_force(cforce)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_contact_force(cforce)


@ti.kernel
def kernel_external_force_p2g(
    total_nodes: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex = particle[np]._compute_external_force(gravity)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                external_force = shape_mapping(shapefn[ln], fex)
                node[nodeID, bodyID]._update_external_force(external_force)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_external_force(external_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_external_force(external_force)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_external_force(external_force)


@ti.kernel
def kernel_force_p2g(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                internal_force = vec3f(
                    [
                        dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5],
                    ]
                )
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)


@ti.kernel
def kernel_force_p2g_2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                internal_force = vec2f(
                    [dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3], dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]]
                )
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)


@ti.kernel
def kernel_viscous_force_p2g(
    total_nodes: int,
    start_index: int,
    end_index: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    matProps: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
    for np in range(start_index, end_index):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            velocity_gradient = particle[np].velocity_gradient
            viscous_stress = matProps.ComputeShearStress(velocity_gradient)
            fInt = particle[np].vol * viscous_stress
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec3f(
                    [
                        dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5],
                    ]
                )
                node[nodeID, bodyID].momentum += internal_force * dt[None]
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].momentum += internal_force * dt[None]
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].momentum += internal_force * dt[None]
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].momentum += internal_force * dt[None]


@ti.kernel
def kernel_viscous_force_p2g_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    matProps: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.momentum.get_scalar_field(d))
    for np in range(start_index, end_index):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            velocity_gradient = particle[np].velocity_gradient
            viscous_stress = matProps.ComputeShearStress2D(velocity_gradient)
            fInt = particle[np].vol * viscous_stress
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec2f(
                    [dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3], dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]]
                )
                node[nodeID, bodyID].momentum += internal_force * dt[None]
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].momentum += internal_force * dt[None]
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].momentum += internal_force * dt[None]


@ti.kernel
def kernel_force_p2g_2DAxisy(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = vec2f(
                    [
                        dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + fInt[2] * shape_fn / position[0],
                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3],
                    ]
                )
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)


@ti.kernel
def kernel_force_bbar_p2g_2DAxisy(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    shapefnc: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                B0bar = shape_fnc / position[0]  # ((position[0] // grid_size[0] + 0.5) * grid_size[0])
                B1bar = dshape_fnc[0]
                B2bar = dshape_fnc[1]

                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = (
                    1.0
                    / 3.0
                    * vec2f(
                        [
                            (B1bar + 2.0 * B1 + B0bar - B0) * fInt[0]
                            + (B1bar - B1 + B0bar - B0) * fInt[1]
                            + (B1bar - B1 + B0bar + 2.0 * B0) * fInt[2]
                            + 3.0 * B2 * fInt[3],
                            (B2bar - B2) * fInt[0]
                            + (B2bar + 2.0 * B2) * fInt[1]
                            + (B2bar - B2) * fInt[2]
                            + 3.0 * B1 * fInt[3],
                        ]
                    )
                )
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)


@ti.kernel
def kernel_force_mls_p2g(
    total_nodes: int,
    particleNum: int,
    gravity: ti.types.vector(3, float),
    gnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    inertia_tensor: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                internal_force = vec3f(
                    fInt[0] * inertia_tensor[0] * pointer[0]
                    + fInt[3] * inertia_tensor[1] * pointer[1]
                    + fInt[5] * inertia_tensor[2] * pointer[2],
                    fInt[3] * inertia_tensor[0] * pointer[0]
                    + fInt[1] * inertia_tensor[1] * pointer[1]
                    + fInt[4] * inertia_tensor[2] * pointer[2],
                    fInt[5] * inertia_tensor[0] * pointer[0]
                    + fInt[4] * inertia_tensor[1] * pointer[1]
                    + fInt[2] * inertia_tensor[2] * pointer[2],
                )
                node[nodeID, bodyID]._update_nodal_force(shape_fn * (fex + internal_force))
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(shape_fn * (fex + internal_force))
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(shape_fn * (fex + internal_force))
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(shape_fn * (fex + internal_force))


@ti.kernel
def kernel_force_mls_p2g_2D(
    total_nodes: int,
    particleNum: int,
    gravity: ti.types.vector(3, float),
    gnum: ti.types.vector(2, int),
    grid_size: ti.types.vector(2, float),
    inertia_tensor: ti.types.vector(2, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                internal_force = vec2f(
                    fInt[0] * inertia_tensor[0] * pointer[0] + fInt[3] * inertia_tensor[1] * pointer[1],
                    fInt[3] * inertia_tensor[0] * pointer[0] + fInt[1] * inertia_tensor[1] * pointer[1],
                )
                node[nodeID, bodyID]._update_nodal_force(shape_fn * (fex + internal_force))
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(shape_fn * (fex + internal_force))
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(shape_fn * (fex + internal_force))


@ti.kernel
def kernel_reference_force_p2g(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)


@ti.kernel
def kernel_reference_force_p2g_2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(2)):
        ti.block_local(node.force.get_scalar_field(d))
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
                if ti.static(GlobalVariable.MPMXPBC) or ti.static(GlobalVariable.MPMYPBC):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)


@ti.kernel
def kernel_external_force_p2g_twophase(
    total_nodes: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
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
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_external_force(external_force, external_forcef + drag_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_external_force(external_force, external_forcef + drag_force)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_external_force(external_force, external_forcef + drag_force)


@ti.kernel
def kernel_force_p2g_twophase(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
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
                internal_force = vec3f(
                    [
                        dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5],
                    ]
                )
                internal_forcef = vec3f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1], dshape_fn[2] * fintf[2]])
                node[nodeID, bodyID]._update_nodal_force(
                    external_force + internal_force, external_forcef + drag_force + internal_forcef
                )
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )


@ti.kernel
def kernel_force_p2g_twophase2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
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
                internal_force = vec2f(
                    [dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3], dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]]
                )
                internal_forcef = vec2f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_nodal_force(
                    external_force + internal_force, external_forcef + drag_force + internal_forcef
                )
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )


@ti.kernel
def kernel_force_bbar_p2g_twophase2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
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
                external_force = shape_mapping(shape_fn, fex)
                external_forcef = shape_mapping(shape_fn, fexf)
                drag_force = shape_mapping(shape_fn, drag)
                internal_force = bbar_internal_force_2d(dshape_fn, dshape_fnc, fInt)
                internal_forcef = vec2f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_nodal_force(
                    external_force + internal_force, external_forcef + drag_force + internal_forcef
                )
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )


@ti.kernel
def kernel_force_p2g_twophase_2DAxisy(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
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
                internal_force = vec2f(
                    [
                        dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + fInt[2] * shape_fn / position[0],
                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3],
                    ]
                )
                internal_forcef = vec2f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_nodal_force(
                    external_force + internal_force, external_forcef + drag_force + internal_forcef
                )
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )


@ti.kernel
def kernel_force_bbar_p2g_twophase_2DAxisy(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    shapefnc: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
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
                internal_force = (
                    1.0
                    / 3.0
                    * vec2f(
                        [
                            (B1bar + 2.0 * B1 + B0bar - B0) * fInt[0]
                            + (B1bar - B1 + B0bar - B0) * fInt[1]
                            + (B1bar - B1 + B0bar + 2.0 * B0) * fInt[2]
                            + 3.0 * B2 * fInt[3],
                            (B2bar - B2) * fInt[0]
                            + (B2bar + 2.0 * B2) * fInt[1]
                            + (B2bar - B2) * fInt[2]
                            + 3.0 * B1 * fInt[3],
                        ]
                    )
                )
                internal_forcef = vec2f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_nodal_force(
                    external_force + internal_force, external_forcef + drag_force + internal_forcef
                )
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(
                                external_force + internal_force, external_forcef + drag_force + internal_forcef
                            )


@ti.kernel
def kernel_sum_cell_stress(
    gauss_num: int,
    dx: ti.types.vector(3, float),
    inv_dx: ti.types.vector(3, float),
    cnum: ti.types.vector(3, int),
    particleNum: int,
    particle: ti.template(),
    cell: ti.template(),
    sub_cell: ti.template(),
):
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
                sub_linear_element_id = (
                    sub_element_id[0] + sub_element_id[1] * gauss_num + sub_element_id[2] * gauss_num * gauss_num
                )
                sub_cell[linear_element_id * gauss_point_num + sub_linear_element_id, bodyID].stress += (
                    particle[np].stress * volume
                )
                sub_cell[linear_element_id * gauss_point_num + sub_linear_element_id, bodyID].vol += volume


@ti.kernel
def kernel_sum_cell_stress_2D(
    gauss_num: int,
    dx: ti.types.vector(2, float),
    inv_dx: ti.types.vector(2, float),
    cnum: ti.types.vector(2, int),
    particleNum: int,
    particle: ti.template(),
    cell: ti.template(),
    sub_cell: ti.template(),
):
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
                sub_cell[linear_element_id * gauss_point_num + sub_linear_element_id, bodyID].stress += (
                    particle[np].stress * volume
                )
                sub_cell[linear_element_id * gauss_point_num + sub_linear_element_id, bodyID].vol += volume


@ti.kernel
def kernel_internal_force_on_gauss_point_p2g(
    gauss_num: int,
    cnum: ti.types.vector(3, int),
    gnum: ti.types.vector(3, int),
    dx: ti.types.vector(3, float),
    inv_dx: ti.types.vector(3, float),
    node: ti.template(),
    cell: ti.template(),
    sub_cell: ti.template(),
    gauss_point: ti.template(),
    weight: ti.template(),
):
    gauss_point_num = gauss_num * gauss_num * gauss_num
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                        internal_force = vec3f(
                            [
                                dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5],
                            ]
                        )
                        node[nodeID, nb]._update_nodal_force(internal_force)
                        if (
                            ti.static(GlobalVariable.MPMXPBC)
                            or ti.static(GlobalVariable.MPMYPBC)
                            or ti.static(GlobalVariable.MPMZPBC)
                        ):
                            index = ti.Vector([vectorize_id(nodeID, gnum)])
                            if ti.static(GlobalVariable.MPMXPBC):
                                xindex = -index[0] + gnum[0]
                                if xindex == 0 or xindex == gnum[0]:
                                    temp_index = index
                                    temp_index[0] = xindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, nb]._update_nodal_force(internal_force)
                            if ti.static(GlobalVariable.MPMYPBC):
                                yindex = -index[1] + gnum[1]
                                if yindex == 0 or yindex == gnum[1]:
                                    temp_index = index
                                    temp_index[1] = yindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, nb]._update_nodal_force(internal_force)
                            if ti.static(GlobalVariable.MPMZPBC):
                                zindex = -index[2] + gnum[2]
                                if zindex == 0 or zindex == gnum[2]:
                                    temp_index = index
                                    temp_index[2] = zindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, nb]._update_nodal_force(internal_force)


@ti.kernel
def kernel_internal_force_on_gauss_point_p2g_2D(
    gauss_num: int,
    cnum: ti.types.vector(2, int),
    gnum: ti.types.vector(2, int),
    dx: ti.types.vector(2, float),
    inv_dx: ti.types.vector(2, float),
    node: ti.template(),
    cell: ti.template(),
    sub_cell: ti.template(),
    gauss_point: ti.template(),
    weight: ti.template(),
):
    gauss_point_num = gauss_num * gauss_num
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
    for nc in range(cell.shape[0]):
        for nb in range(cell.shape[1]):
            if int(cell[nc, nb].active) == 1:
                base = vec2i(vectorize_id(nc, cnum))
                volume = dx[0] * dx[1] / gauss_point_num
                for ngp in range(gauss_point_num):
                    gp = 0.5 * dx * (gauss_point[ngp] + 1) + base * dx
                    fInt = -weight[ngp] * sub_cell[nc * gauss_point_num + ngp, nb].stress * volume
                    for i, j in ti.static(ti.ndrange(2, 2)):
                        nx, ny = base[0] + i, base[1] + j
                        nodeID = nx + ny * gnum[0]
                        sx = ShapeLinear(gp[0], nx * dx[0], inv_dx[0], 0)
                        sy = ShapeLinear(gp[1], ny * dx[1], inv_dx[1], 0)
                        gsx = GShapeLinear(gp[0], nx * dx[0], inv_dx[0], 0)
                        gsy = GShapeLinear(gp[1], ny * dx[1], inv_dx[1], 0)
                        dshape_fn = vec2f(gsx * sy, gsy * sx)
                        internal_force = vec2f(
                            [
                                dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3],
                            ]
                        )
                        node[nodeID, nb]._update_nodal_force(internal_force)
                        if (
                            ti.static(GlobalVariable.MPMXPBC)
                            or ti.static(GlobalVariable.MPMYPBC)
                            or ti.static(GlobalVariable.MPMZPBC)
                        ):
                            index = ti.Vector([vectorize_id(nodeID, gnum)])
                            if ti.static(GlobalVariable.MPMXPBC):
                                xindex = -index[0] + gnum[0]
                                if xindex == 0 or xindex == gnum[0]:
                                    temp_index = index
                                    temp_index[0] = xindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, nb]._update_nodal_force(internal_force)
                            if ti.static(GlobalVariable.MPMYPBC):
                                yindex = -index[1] + gnum[1]
                                if yindex == 0 or yindex == gnum[1]:
                                    temp_index = index
                                    temp_index[1] = yindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, nb]._update_nodal_force(internal_force)


@ti.kernel
def kernel_internal_force_on_material_point_p2g(
    cnum: ti.types.vector(3, int),
    gnum: ti.types.vector(3, int),
    dx: ti.types.vector(3, float),
    inv_dx: ti.types.vector(3, float),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    cell: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                    internal_force = vec3f(
                        [
                            dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                            dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                            dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5],
                        ]
                    )
                    node[nodeID, bodyID]._update_nodal_force(internal_force)
                    if (
                        ti.static(GlobalVariable.MPMXPBC)
                        or ti.static(GlobalVariable.MPMYPBC)
                        or ti.static(GlobalVariable.MPMZPBC)
                    ):
                        index = ti.Vector([vectorize_id(nodeID, gnum)])
                        if ti.static(GlobalVariable.MPMXPBC):
                            xindex = -index[0] + gnum[0]
                            if xindex == 0 or xindex == gnum[0]:
                                temp_index = index
                                temp_index[0] = xindex
                                tideID = linearize(temp_index, gnum)
                                node[tideID, bodyID]._update_nodal_force(internal_force)
                        if ti.static(GlobalVariable.MPMYPBC):
                            yindex = -index[1] + gnum[1]
                            if yindex == 0 or yindex == gnum[1]:
                                temp_index = index
                                temp_index[1] = yindex
                                tideID = linearize(temp_index, gnum)
                                node[tideID, bodyID]._update_nodal_force(internal_force)
                        if ti.static(GlobalVariable.MPMZPBC):
                            zindex = -index[2] + gnum[2]
                            if zindex == 0 or zindex == gnum[2]:
                                temp_index = index
                                temp_index[2] = zindex
                                tideID = linearize(temp_index, gnum)
                                node[tideID, bodyID]._update_nodal_force(internal_force)


@ti.kernel
def kernel_internal_force_on_material_point_p2g_2D(
    cnum: ti.types.vector(2, int),
    gnum: ti.types.vector(2, int),
    dx: ti.types.vector(2, float),
    inv_dx: ti.types.vector(2, float),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    cell: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                    internal_force = vec2f(
                        [
                            dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                            dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3],
                        ]
                    )
                    node[nodeID, bodyID]._update_nodal_force(internal_force)
                    if (
                        ti.static(GlobalVariable.MPMXPBC)
                        or ti.static(GlobalVariable.MPMYPBC)
                        or ti.static(GlobalVariable.MPMZPBC)
                    ):
                        index = ti.Vector([vectorize_id(nodeID, gnum)])
                        if ti.static(GlobalVariable.MPMXPBC):
                            xindex = -index[0] + gnum[0]
                            if xindex == 0 or xindex == gnum[0]:
                                temp_index = index
                                temp_index[0] = xindex
                                tideID = linearize(temp_index, gnum)
                                node[tideID, bodyID]._update_nodal_force(internal_force)
                        if ti.static(GlobalVariable.MPMYPBC):
                            yindex = -index[1] + gnum[1]
                            if yindex == 0 or yindex == gnum[1]:
                                temp_index = index
                                temp_index[1] = yindex
                                tideID = linearize(temp_index, gnum)
                                node[tideID, bodyID]._update_nodal_force(internal_force)


@ti.kernel
def kernel_internal_force_p2g_twophase2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt, fintf = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec2f(
                    [dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3], dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]]
                )
                internal_forcef = vec2f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_internal_force(internal_force, internal_forcef)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_internal_force(internal_force, internal_forcef)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_internal_force(internal_force, internal_forcef)


@ti.kernel
def kernel_internal_force_p2g_twophase(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt, fintf = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec3f(
                    [
                        dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5],
                    ]
                )
                internal_forcef = vec3f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1], dshape_fn[2] * fintf[2]])
                node[nodeID, bodyID]._update_internal_force(internal_force, internal_forcef)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_internal_force(internal_force, internal_forcef)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_internal_force(internal_force, internal_forcef)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_internal_force(internal_force, internal_forcef)


@ti.kernel
def kernel_force_bbar_p2g(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                temp_dshape = (dshape_fnc - dshape_fn) / 3.0
                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = vec3f(
                    [
                        (dshape_fn[0] + temp_dshape[0]) * fInt[0]
                        + temp_dshape[0] * fInt[1]
                        + temp_dshape[0] * fInt[2]
                        + dshape_fn[1] * fInt[3]
                        + dshape_fn[2] * fInt[5],
                        temp_dshape[1] * fInt[0]
                        + (dshape_fn[1] + temp_dshape[1]) * fInt[1]
                        + temp_dshape[1] * fInt[2]
                        + dshape_fn[0] * fInt[3]
                        + dshape_fn[2] * fInt[4],
                        temp_dshape[2] * fInt[0]
                        + temp_dshape[2] * fInt[1]
                        + (dshape_fn[2] + temp_dshape[2]) * fInt[2]
                        + dshape_fn[1] * fInt[4]
                        + dshape_fn[0] * fInt[5],
                    ]
                )
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)


@ti.kernel
def kernel_force_bbar_p2g_2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
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
                external_force = shape_mapping(shapefn[ln], fex)
                internal_force = bbar_internal_force_2d(dshape_fn, dshape_fnc, fInt)
                node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_force(external_force + internal_force)


@ti.kernel
def kernel_internal_force_bbar_p2g_twophase2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.force.get_scalar_field(d))
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt, fintf = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                internal_force = vec2f(
                    [dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3], dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]]
                )
                internal_forcef = vec2f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1]])
                node[nodeID, bodyID]._update_internal_force(internal_force, internal_forcef)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_internal_force(internal_force, internal_forcef)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_internal_force(internal_force, internal_forcef)


@ti.kernel
def kernel_volume_p2g(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
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
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].vol += nvol
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].vol += nvol
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].vol += nvol


@ti.kernel
def kernel_jacobian_p2g(
    total_nodes: int,
    dt: ti.template(),
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    ti.block_local(node.jacobian)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            mass = particle[np].m
            velocity_gradient = particle[np].velocity_gradient
            djacobian = (
                ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity_gradient
            ).determinant()
            transfer_var = mass * djacobian
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                node[nodeID, bodyID].jacobian += shape_mapping(shapefn[ln], transfer_var)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].jacobian += shape_mapping(shapefn[ln], transfer_var)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].jacobian += shape_mapping(shapefn[ln], transfer_var)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].jacobian += shape_mapping(shapefn[ln], transfer_var)


@ti.kernel
def kernel_pressure_p2g(
    particleNum: int,
    total_nodes: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    ti.block_local(node.pressure)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            pressure = particle[np].m * particle[np]._get_mean_stress()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                node[nodeID, bodyID].pressure += shape_mapping(shapefn[ln], pressure)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].pressure += shape_mapping(shapefn[ln], pressure)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].pressure += shape_mapping(shapefn[ln], pressure)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].pressure += shape_mapping(shapefn[ln], pressure)


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
def kernel_compute_gauss_average_stress_2D(
    gauss_num: int, cut_off: float, cell: ti.template(), sub_cell: ti.template()
):
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
                pressure = 0.0
                for ngp in range(gauss_number):
                    stress = sub_cell[nc * gauss_number + ngp, nb].stress
                    pressure += (stress[0] + stress[1] + stress[2]) / 3.0
                pressure /= gauss_number

                for ngp in range(gauss_number):
                    stress = sub_cell[nc * gauss_number + ngp, nb].stress
                    p = (stress[0] + stress[1] + stress[2]) / 3.0
                    ave_stress = stress - (p - pressure) * EYE
                    sub_cell[nc * gauss_number + ngp, nb].stress = ave_stress


@ti.kernel
def kernel_average_pressure_2D(gauss_num: int, cell: ti.template(), sub_cell: ti.template()):
    gauss_number = gauss_num * gauss_num
    for nc in range(cell.shape[0]):
        for nb in range(cell.shape[1]):
            if int(cell[nc, nb].active) == 1:
                pressure = 0.0
                for ngp in range(gauss_number):
                    stress = sub_cell[nc * gauss_number + ngp, nb].stress
                    pressure += (stress[0] + stress[1] + stress[2]) / 3.0
                pressure /= gauss_number

                for ngp in range(gauss_number):
                    stress = sub_cell[nc * gauss_number + ngp, nb].stress
                    p = (stress[0] + stress[1] + stress[2]) / 3.0
                    ave_stress = stress - (p - pressure) * EYE
                    sub_cell[nc * gauss_number + ngp, nb].stress = ave_stress


@ti.kernel
def kernel_compute_grid_velocity(cutoff: float, node: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            node[ng, nb]._compute_nodal_velocity()


@ti.kernel
def kernel_compute_grid_velocity_twophase(cutoff: float, node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            node[ng, nb]._compute_nodal_velocity(cutoff)


@ti.kernel
def kernel_compute_grid_kinematic(cutoff: float, damp: float, node: ti.template(), dt: ti.template()):
    # ti.block_local(dt)
    for ng, nb in node:
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
def kernel_kinemaitc_g2p(
    total_nodes: int,
    alpha: float,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    ti.block_local(node.momentum)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPIC, vFLIP = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(
                float, GlobalVariable.DIMENSION
            )
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
def kernel_compute_particle_Gpf(
    total_nodes: int,
    cut_off: float,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    ti.block_local(node.m)
    ti.block_local(node.pf)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            Gpf = ti.Vector.zero(float, GlobalVariable.DIMENSION)
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                pf = node[nodeID, bodyID].pf
                m = node[nodeID, bodyID].m
                if m > cut_off:
                    Gpf += shape_mapping(dshape_fn, pf / m)
            particle[np].gpf = Gpf


@ti.kernel
def kernel_kinemaitc_g2p_twophase(
    total_nodes: int,
    alpha: float,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    # ti.block_local(dt)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPICs, vFLIPs = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(
                float, GlobalVariable.DIMENSION
            )
            vPICf, vFLIPf = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(
                float, GlobalVariable.DIMENSION
            )
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
def kernel_mass_g2p(
    total_nodes: int,
    cell_volume: float,
    node_size: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node: ti.template(),
    particleNum: int,
    particle: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mdensity = 0.0
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                mass_density = shape_mapping(shapefn[ln], node[nodeID, bodyID].m / cell_volume)
                mdensity += mass_density
            particle[np].mass_density = mdensity


@ti.kernel
def kernel_pressure_g2p(
    particleNum: int,
    total_nodes: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            pressure = 0.0
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
def kernel_find_sound_speed(start_index: int, end_index: int, particle: ti.template(), matProps: ti.template()):
    for np in range(start_index, end_index):
        matProps._set_modulus(Squared(particle[np].v))


@ti.kernel
def kernel_compute_reference_stress_strain(
    start_index: int,
    end_index: int,
    dt: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient.transpose()
            stateVars[np].deformation_gradient += velocity_gradient * dt[None]
            deformation_gradient = stateVars[np].deformation_gradient
            velocity_gradient = velocity_gradient @ deformation_gradient.inverse()
            stress = particle[np].stress
            particle[np].stress = matProps.ComputePKStress(np, stress, velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_compute_reference_stress_strain_2D(
    start_index: int,
    end_index: int,
    dt: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient.transpose()
            stateVars[np].deformation_gradient += velocity_gradient * dt[None]
            deformation_gradient = stateVars[np].deformation_gradient
            velocity_gradient = velocity_gradient @ deformation_gradient.inverse()
            stress = particle[np].stress
            particle[np].stress = matProps.ComputePKStress2D(np, stress, velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_compute_stress_strain(
    start_index: int,
    end_index: int,
    dt: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress
            particle[np].stress = matProps.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_compute_stress_strain_2D(
    start_index: int,
    end_index: int,
    dt: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress
            particle[np].stress = matProps.ComputeStress2D(np, previous_stress, velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_compute_stress_strain_twophase(
    start_index: int,
    end_index: int,
    dt: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            solid_velocity_gradient, fluid_velocity_gradient = (
                particle[np].solid_velocity_gradient,
                particle[np].fluid_velocity_gradient,
            )
            previous_stress, porosity = particle[np].stress, particle[np].porosity
            particle[np].pressure -= matProps.ComputePressure(
                solid_velocity_gradient, fluid_velocity_gradient, porosity, dt
            )
            particle[np].stress = matProps.ComputeStress(np, previous_stress, solid_velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_compute_stress_strain_twophase2D(
    start_index: int,
    end_index: int,
    dt: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            solid_velocity_gradient, fluid_velocity_gradient = (
                particle[np].solid_velocity_gradient,
                particle[np].fluid_velocity_gradient,
            )
            previous_stress, porosity = particle[np].stress, particle[np].porosity
            particle[np].pressure -= matProps.ComputePressure(
                solid_velocity_gradient, fluid_velocity_gradient, porosity, dt
            )
            particle[np].stress = matProps.ComputeStress2D(np, previous_stress, solid_velocity_gradient, stateVars, dt)


# ========================================================= #
#                 Update velocity gradient                  #
# ========================================================= #
@ti.kernel
def kernel_update_velocity_gradient_fbar(
    fraction: float,
    cutoff: float,
    total_nodes: int,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            djacobian_bar = 0.0
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                djacobian_bar += shape_mapping(shape_fn, node[nodeID, bodyID].jacobian)

            velocity_gradient = particle[np].velocity_gradient
            ddeformation_gradient = ti.Matrix.identity(float, GlobalVariable.DIMENSION) + dt[None] * velocity_gradient
            djacobian = ddeformation_gradient.determinant()
            djacobian_bar_new = fraction * djacobian_bar + (1.0 - fraction) * djacobian
            # jacobian_bar_new = clamp(0.01, 100, jacobian_bar_new)

            multiplier = (djacobian_bar_new / djacobian) ** (1.0 / GlobalVariable.DIMENSION)
            # === split into volumetric + deviatoric ===
            trL = velocity_gradient.trace() / GlobalVariable.DIMENSION
            devL = velocity_gradient - trL * ti.Matrix.identity(float, GlobalVariable.DIMENSION)

            # adjust only volumetric part
            trL_new = (multiplier - 1.0) / dt[None] + multiplier * trL
            updated_velocity_gradient = devL + trL_new * ti.Matrix.identity(float, GlobalVariable.DIMENSION)

            particle[np].velocity_gradient = updated_velocity_gradient


@ti.kernel
def kernel_update_velocity_gradient(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            velocity_gradient = ZEROMAT3x3
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                dshape_fn = dshapefn[ln]
                velocity_gradient += outer_product(gv, dshape_fn)
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps.update_particle_volume(np, velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_update_velocity_gradient_affine(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
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
                Bp += shape_fn * outer_product(gv, pointer)
            velocity_gradient = truncation(Bp @ Wp.inverse()) if Wp.determinant() > Threshold else ZEROMAT3x3
            particle[np].velocity_gradient = velocity_gradient


@ti.kernel
def kernel_update_velocity_gradient_affine_sparse(
    total_nodes: int,
    particleNum: int,
    grid_size: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    active_block_ids: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            Wp = ZEROMAT3x3
            Bp = ZEROMAT3x3
            offset = np * total_nodes
            position = particle[np].x
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                grid_coord = grid_size * vec3f(
                    compact_node_grid_coord(nodeID, block_count, block_size, block_volume, active_block_ids)
                )
                pointer = grid_coord - position
                gv = node[nodeID, bodyID].momentum
                shape_fn = shapefn[ln]

                Wp += shape_fn * outer_product(pointer, pointer)
                Bp += shape_fn * outer_product(gv, pointer)
            velocity_gradient = truncation(Bp @ Wp.inverse()) if Wp.determinant() > Threshold else ZEROMAT3x3
            particle[np].velocity_gradient = velocity_gradient


@ti.kernel
def kernel_update_velocity_gradient_bbar(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            velocity_gradient = ZEROMAT3x3
            strain_rate_trace = ZEROVEC3f
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                temp_dshape = (dshape_fnc - dshape_fn) / 3.0

                average_bmatrix = temp_dshape[0] * gv[0] + temp_dshape[1] * gv[1] + temp_dshape[2] * gv[2]
                velocity_gradient += outer_product(gv, dshape_fn)
                velocity_gradient[0, 0] += average_bmatrix
                velocity_gradient[1, 1] += average_bmatrix
                velocity_gradient[2, 2] += average_bmatrix

                strain_rate_trace[0] += dshape_fn[0] * gv[0]
                strain_rate_trace[1] += dshape_fn[1] * gv[1]
                strain_rate_trace[2] += dshape_fn[2] * gv[2]
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps.update_particle_volume_bbar(np, strain_rate_trace, stateVars, dt)


@ti.kernel
def kernel_update_velocity_gradient_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            velocity_gradient = ZEROMAT2x2
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                dshape_fn = dshapefn[ln]
                velocity_gradient += outer_product2D(gv, dshape_fn)
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps.update_particle_volume_2D(np, velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_update_velocity_gradient_affine_2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(2, int),
    grid_size: ti.types.vector(2, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
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
            velocity_gradient = truncation(Bp @ Wp.inverse()) if Wp.determinant() > Threshold else ZEROMAT2x2
            particle[np].velocity_gradient = velocity_gradient


@ti.kernel
def kernel_update_velocity_gradient_affine_2D_sparse(
    total_nodes: int,
    particleNum: int,
    grid_size: ti.types.vector(2, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    active_block_ids: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            Wp = ZEROMAT2x2
            Bp = ZEROMAT2x2
            offset = np * total_nodes
            position = particle[np].x
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                grid_coord = grid_size * ti.cast(
                    compact_node_grid_coord(nodeID, block_count, block_size, block_volume, active_block_ids), float
                )
                pointer = grid_coord - position
                gv = node[nodeID, bodyID].momentum
                shape_fn = shapefn[ln]
                Wp += shape_fn * outer_product2D(pointer, pointer)
                Bp += shape_fn * outer_product2D(gv, pointer)
            velocity_gradient = truncation(Bp @ Wp.inverse()) if Wp.determinant() > Threshold else ZEROMAT2x2
            particle[np].velocity_gradient = velocity_gradient


@ti.kernel
def kernel_update_velocity_gradient_2DAxisy(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            velocity_gradient = ZEROMAT3x3
            offset = np * total_nodes
            position = particle[np].x
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                velocity_gradient0 = outer_product2D(gv, dshape_fn)
                velocity_gradient += mat3x3(
                    [
                        [velocity_gradient0[0, 0], velocity_gradient0[0, 1], 0],
                        [velocity_gradient0[1, 0], velocity_gradient0[1, 1], 0],
                        [0, 0, shape_fn * gv[0] / position[0]],
                    ]
                )
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps.update_particle_volume(np, velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_update_velocity_gradient_bbar_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            velocity_gradient = ZEROMAT2x2
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].momentum
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                velocity_gradient += bbar_velocity_gradient_2d(gv, dshape_fn, dshape_fnc)
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps.update_particle_volume_bbar_2D(np, velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_update_velocity_gradient_bbar_2DAxisy(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    shapefnc: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            one_three = 1.0 / 3.0
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
                B0bar = shape_fnc / position[0]  # ((position[0] // grid_size[0]) + 0.5 * grid_size[0])
                B1bar = dshape_fnc[0]
                B2bar = dshape_fnc[1]

                velocity_gradient += mat3x3(
                    [one_three * ((B1bar + 2.0 * B1 + B0bar - B0) * gv[0] + (B2bar - B2) * gv[1]), B2 * gv[0], 0],
                    [B1 * gv[1], one_three * ((B1bar - B1 + B0bar - B0) * gv[0] + (B2bar + 2.0 * B2) * gv[1]), 0],
                    [0, 0, one_three * ((B1bar - B1 + B0bar + 2.0 * B0) * gv[0] + (B2bar - B2) * gv[1])],
                )
                strain_rate_trace += vec3f(B1 * gv[0], B2 * gv[1], B0 * gv[0])
            particle[np].velocity_gradient = truncation(velocity_gradient)
            particle[np].vol *= matProps.update_particle_volume_bbar(np, strain_rate_trace, stateVars, dt)


@ti.kernel
def kernel_update_velocity_gradient_twophase(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            solid_velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
            fluid_velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gvs = node[nodeID, bodyID].momentums
                gvf = node[nodeID, bodyID].momentumf
                dshape_fn = dshapefn[ln]
                solid_velocity_gradient += outer_product(gvs, dshape_fn)
                fluid_velocity_gradient += outer_product(gvf, dshape_fn)
            porosity, pvolume = particle[np].porosity, particle[np].vol
            pvolume *= matProps.update_particle_volume(np, solid_velocity_gradient, stateVars, dt)
            porosity = matProps.update_particle_porosity(solid_velocity_gradient, porosity, dt)
            particle[np].mf = matProps.update_particle_fluid_mass(pvolume, porosity)
            particle[np].m = particle[np].ms + particle[np].mf
            particle[np].vol, particle[np].porosity = pvolume, porosity
            particle[np].solid_velocity_gradient = solid_velocity_gradient
            particle[np].fluid_velocity_gradient = fluid_velocity_gradient


@ti.kernel
def kernel_update_velocity_gradient_twophase_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            solid_velocity_gradient = ZEROMAT2x2
            fluid_velocity_gradient = ZEROMAT2x2
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gvs = node[nodeID, bodyID].momentums
                gvf = node[nodeID, bodyID].momentumf
                dshape_fn = dshapefn[ln]
                solid_velocity_gradient += outer_product2D(gvs, dshape_fn)
                fluid_velocity_gradient += outer_product2D(gvf, dshape_fn)
            porosity, pvolume = particle[np].porosity, particle[np].vol
            pvolume *= matProps.update_particle_volume_2D(np, solid_velocity_gradient, stateVars, dt)
            porosity = matProps.update_particle_porosity(solid_velocity_gradient, porosity, dt)
            particle[np].mf = matProps.update_particle_fluid_mass(pvolume, porosity)
            particle[np].m = particle[np].ms + particle[np].mf
            particle[np].vol, particle[np].porosity = pvolume, porosity
            particle[np].solid_velocity_gradient = truncation(solid_velocity_gradient)
            particle[np].fluid_velocity_gradient = truncation(fluid_velocity_gradient)


@ti.kernel
def kernel_update_velocity_gradient_bbar_twophase_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            solid_velocity_gradient = ZEROMAT2x2
            fluid_velocity_gradient = ZEROMAT2x2
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gvs = node[nodeID, bodyID].momentums
                gvf = node[nodeID, bodyID].momentumf
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                solid_velocity_gradient += bbar_velocity_gradient_2d(gvs, dshape_fn, dshape_fnc)
                fluid_velocity_gradient += mat2x2([[dshape_fnc[0] * gvf[0], 0], [0, dshape_fnc[1] * gvf[1]]])
            porosity, pvolume = particle[np].porosity, particle[np].vol
            pvolume *= matProps.update_particle_volume_bbar_2D(np, solid_velocity_gradient, stateVars, dt)
            porosity = matProps.update_particle_porosity(solid_velocity_gradient, porosity, dt)
            particle[np].mf = matProps.update_particle_fluid_mass(pvolume, porosity)
            particle[np].m = particle[np].ms + particle[np].mf
            particle[np].vol, particle[np].porosity = pvolume, porosity
            particle[np].solid_velocity_gradient = solid_velocity_gradient
            particle[np].fluid_velocity_gradient = fluid_velocity_gradient


# ========================================================= #
#                     Update Rotation                       #
# ========================================================= #
@ti.kernel
def update_coupling_quanternion(particleNum: int, particle: ti.template(), dt: ti.template()):
    for np in range(particleNum):
        old_q = particle[np].q
        omega = get_angular_velocity(particle[np].velocity_gradient)
        rotation_matrix = SetToRotate(old_q)
        dq = SetDQ(old_q, rotation_matrix.transpose() @ omega) * dt[None]
        q = Normalize(old_q + dq)
        particle[np].q = q


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
def kernel_postmapping_kinemaitc(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], particle[np].m)
                node[nodeID, bodyID]._update_nodal_momentum(nmass * particle[np].v)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * particle[np].v)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * particle[np].v)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_momentum(nmass * particle[np].v)


@ti.kernel
def kernel_postmapping_kinemaitc_twophase2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], particle[np].m)
                nmass_s = shape_mapping(shapefn[ln], particle[np].ms)
                nmass_f = shape_mapping(shapefn[ln], particle[np].mf)
                node[nodeID, bodyID]._update_nodal_momentum(
                    nmass * particle[np].v, nmass_s * particle[np].vs, nmass_f * particle[np].vf
                )
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_momentum(
                                nmass * particle[np].v, nmass_s * particle[np].vs, nmass_f * particle[np].vf
                            )
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_momentum(
                                nmass * particle[np].v, nmass_s * particle[np].vs, nmass_f * particle[np].vf
                            )
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_momentum(
                                nmass * particle[np].v, nmass_s * particle[np].vs, nmass_f * particle[np].vf
                            )


@ti.kernel
def kernel_calc_contact_normal_2DAxisy(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            position = particle[np].x
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                grad_domain = dshapefn[ln] * particle[np].vol / position[0]
                node[nodeID, bodyID]._update_nodal_grad_domain(grad_domain)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_grad_domain(grad_domain)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_grad_domain(grad_domain)


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
    temp[0, 0] = GS[0] + (GSC[0] - GS[0]) / 3.0
    temp[0, 1] = (GSC[1] - GS[1]) / 3.0
    temp[0, 2] = (GSC[2] - GS[2]) / 3.0
    temp[1, 0] = (GSC[0] - GS[0]) / 3.0
    temp[1, 1] = GS[1] + (GSC[1] - GS[1]) / 3.0
    temp[1, 2] = (GSC[2] - GS[2]) / 3.0
    temp[2, 0] = (GSC[0] - GS[0]) / 3.0
    temp[2, 1] = (GSC[1] - GS[1]) / 3.0
    temp[2, 2] = GS[2] + (GSC[2] - GS[2]) / 3.0
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
def kernel_dilatational_velocity_p2g(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
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
            if (
                ti.static(GlobalVariable.MPMXPBC)
                or ti.static(GlobalVariable.MPMYPBC)
                or ti.static(GlobalVariable.MPMZPBC)
            ):
                index = ti.Vector([vectorize_id(nodeID, gnum)])
                if ti.static(GlobalVariable.MPMXPBC):
                    xindex = -index[0] + gnum[0]
                    if xindex == 0 or xindex == gnum[0]:
                        temp_index = index
                        temp_index[0] = xindex
                        tideID = linearize(temp_index, gnum)
                        node[tideID, bodyID].vol += shape_mapping(shape_fn, volume)
                        node[tideID, bodyID].jacobian += shape_mapping(shape_fn, dil_gradv * volume)
                if ti.static(GlobalVariable.MPMYPBC):
                    yindex = -index[1] + gnum[1]
                    if yindex == 0 or yindex == gnum[1]:
                        temp_index = index
                        temp_index[1] = yindex
                        tideID = linearize(temp_index, gnum)
                        node[tideID, bodyID].vol += shape_mapping(shape_fn, volume)
                        node[tideID, bodyID].jacobian += shape_mapping(shape_fn, dil_gradv * volume)
                if ti.static(GlobalVariable.MPMZPBC):
                    zindex = -index[2] + gnum[2]
                    if zindex == 0 or zindex == gnum[2]:
                        temp_index = index
                        temp_index[2] = zindex
                        tideID = linearize(temp_index, gnum)
                        node[tideID, bodyID].vol += shape_mapping(shape_fn, volume)
                        node[tideID, bodyID].jacobian += shape_mapping(shape_fn, dil_gradv * volume)


@ti.kernel
def kernel_gradient_velocity_projection_correction_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    dt: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        bodyID = int(particle[np].bodyID)
        velocity_gradient = particle[np].velocity_gradient
        volume = particle[np].vol

        dil_gradv_bar = 0.0
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            dil_gradv_bar += shape_fn * node[nodeID, bodyID].jacobian
        velocity_gradient += 1.0 / 2.0 * (dil_gradv_bar - trace(velocity_gradient)) * DELTA2D
        pressureAV = matProps.ComputePressure2D(np, stateVars, velocity_gradient, dt)
        particle[np].stress = matProps.ComputeShearStress2D(velocity_gradient)

        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            node[nodeID, bodyID].pressure += shape_fn * pressureAV * volume
            if (
                ti.static(GlobalVariable.MPMXPBC)
                or ti.static(GlobalVariable.MPMYPBC)
                or ti.static(GlobalVariable.MPMZPBC)
            ):
                index = ti.Vector([vectorize_id(nodeID, gnum)])
                if ti.static(GlobalVariable.MPMXPBC):
                    xindex = -index[0] + gnum[0]
                    if xindex == 0 or xindex == gnum[0]:
                        temp_index = index
                        temp_index[0] = xindex
                        tideID = linearize(temp_index, gnum)
                        node[tideID, bodyID].pressure += shape_fn * pressureAV * volume
                if ti.static(GlobalVariable.MPMYPBC):
                    yindex = -index[1] + gnum[1]
                    if yindex == 0 or yindex == gnum[1]:
                        temp_index = index
                        temp_index[1] = yindex
                        tideID = linearize(temp_index, gnum)
                        node[tideID, bodyID].pressure += shape_fn * pressureAV * volume


@ti.kernel
def kernel_gradient_velocity_projection_correction(
    total_nodes: int,
    start_index: int,
    end_index: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    dt: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        bodyID = int(particle[np].bodyID)
        velocity_gradient = particle[np].velocity_gradient
        volume = particle[np].vol

        dil_gradv_bar = 0.0
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            dil_gradv_bar += shape_fn * node[nodeID, bodyID].jacobian
        velocity_gradient += 1.0 / 3.0 * (dil_gradv_bar - trace(velocity_gradient)) * DELTA
        pressureAV = matProps.ComputePressure(np, stateVars, velocity_gradient, dt)
        particle[np].stress = matProps.ComputeShearStress(velocity_gradient)

        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            node[nodeID, bodyID].pressure += shape_fn * pressureAV * volume
            if (
                ti.static(GlobalVariable.MPMXPBC)
                or ti.static(GlobalVariable.MPMYPBC)
                or ti.static(GlobalVariable.MPMZPBC)
            ):
                index = ti.Vector([vectorize_id(nodeID, gnum)])
                if ti.static(GlobalVariable.MPMXPBC):
                    xindex = -index[0] + gnum[0]
                    if xindex == 0 or xindex == gnum[0]:
                        temp_index = index
                        temp_index[0] = xindex
                        tideID = linearize(temp_index, gnum)
                        node[tideID, bodyID].pressure += shape_fn * pressureAV * volume
                if ti.static(GlobalVariable.MPMYPBC):
                    yindex = -index[1] + gnum[1]
                    if yindex == 0 or yindex == gnum[1]:
                        temp_index = index
                        temp_index[1] = yindex
                        tideID = linearize(temp_index, gnum)
                        node[tideID, bodyID].pressure += shape_fn * pressureAV * volume
                if ti.static(GlobalVariable.MPMZPBC):
                    zindex = -index[2] + gnum[2]
                    if zindex == 0 or zindex == gnum[2]:
                        temp_index = index
                        temp_index[2] = zindex
                        tideID = linearize(temp_index, gnum)
                        node[tideID, bodyID].pressure += shape_fn * pressureAV * volume


@ti.kernel
def kernel_pressure_correction(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        pressure_bar = 0.0
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            shape_fn = shapefn[ln]
            pressure_bar += shape_fn * node[nodeID, bodyID].pressure
        particle[np].stress -= pressure_bar * EYE


# ========================================================= #
#                 Compute Domain Gradient                   #
# ========================================================= #
@ti.kernel
def kernel_calc_contact_normal(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                grad_domain = dshapefn[ln] * particle[np].vol
                node[nodeID, bodyID]._update_nodal_grad_domain(grad_domain)
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_grad_domain(grad_domain)
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_grad_domain(grad_domain)
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID]._update_nodal_grad_domain(grad_domain)


@ti.kernel
def kernel_calc_contact_displacement(
    total_nodes: int,
    particleNum: int,
    cutoff: float,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb].contact_pos.fill(0)

    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                contact_pos = shapefn[ln] * particle[np].x * particle[np].m
                node[nodeID, bodyID].contact_pos += contact_pos
                if (
                    ti.static(GlobalVariable.MPMXPBC)
                    or ti.static(GlobalVariable.MPMYPBC)
                    or ti.static(GlobalVariable.MPMZPBC)
                ):
                    index = ti.Vector([vectorize_id(nodeID, gnum)])
                    if ti.static(GlobalVariable.MPMXPBC):
                        xindex = -index[0] + gnum[0]
                        if xindex == 0 or xindex == gnum[0]:
                            temp_index = index
                            temp_index[0] = xindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].contact_pos += contact_pos
                    if ti.static(GlobalVariable.MPMYPBC):
                        yindex = -index[1] + gnum[1]
                        if yindex == 0 or yindex == gnum[1]:
                            temp_index = index
                            temp_index[1] = yindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].contact_pos += contact_pos
                    if ti.static(GlobalVariable.MPMZPBC):
                        zindex = -index[2] + gnum[2]
                        if zindex == 0 or zindex == gnum[2]:
                            temp_index = index
                            temp_index[2] = zindex
                            tideID = linearize(temp_index, gnum)
                            node[tideID, bodyID].contact_pos += contact_pos


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
def kernel_calc_friction_contact(
    cut_off: float,
    mu: float,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    dt: ti.template(),
    is_rigid: ti.template(),
    node: ti.template(),
):
    # ti.block_local(dt)
    for ng in range(node.shape[0]):
        for bodyID1 in range(node.shape[1] - 1):
            for bodyID2 in range(bodyID1, node.shape[1]):
                m1, m2 = node[ng, bodyID1].m, node[ng, bodyID2].m
                if m1 > cut_off and m2 > cut_off:
                    mv1, mv2 = m1 * node[ng, bodyID1].momentum, m2 * node[ng, bodyID2].momentum
                    norm1, norm2 = node[ng, bodyID1].grad_domain, node[ng, bodyID2].grad_domain

                    norm, g_mass = ti.Vector.zero(float, GlobalVariable.DIMENSION), 0.0
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
                        inv_gmass = 1.0 / g_mass
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

                        if (
                            ti.static(GlobalVariable.MPMXPBC)
                            or ti.static(GlobalVariable.MPMYPBC)
                            or ti.static(GlobalVariable.MPMZPBC)
                        ):
                            index = ti.Vector([vectorize_id(ng, gnum)])
                            if ti.static(GlobalVariable.MPMXPBC):
                                xindex = -index[0] + gnum[0]
                                if xindex == 0 or xindex == gnum[0]:
                                    temp_index = index
                                    temp_index[0] = xindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, bodyID1]._update_contact_force(-cforce)
                                    node[tideID, bodyID2]._update_contact_force(cforce)
                            if ti.static(GlobalVariable.MPMYPBC):
                                yindex = -index[1] + gnum[1]
                                if yindex == 0 or yindex == gnum[1]:
                                    temp_index = index
                                    temp_index[1] = yindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, bodyID1]._update_contact_force(-cforce)
                                    node[tideID, bodyID2]._update_contact_force(cforce)
                            if ti.static(GlobalVariable.MPMZPBC):
                                zindex = -index[2] + gnum[2]
                                if zindex == 0 or zindex == gnum[2]:
                                    temp_index = index
                                    temp_index[2] = zindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, bodyID1]._update_contact_force(-cforce)
                                    node[tideID, bodyID2]._update_contact_force(cforce)


################## Geo contact ##################
@ti.kernel
def kernel_calc_geocontact(
    cut_off: float,
    mu: float,
    alpha: float,
    beta: float,
    offset: float,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    is_rigid: ti.template(),
    node: ti.template(),
):
    # ti.block_local(dt)
    for ng in range(node.shape[0]):
        for bodyID1 in range(node.shape[1] - 1):
            for bodyID2 in range(bodyID1, node.shape[1]):
                m1, m2 = node[ng, bodyID1].m, node[ng, bodyID2].m
                if m1 > cut_off and m2 > cut_off:
                    contact_pos1, contact_pos2 = node[ng, bodyID1].contact_pos / m1, node[ng, bodyID2].contact_pos / m2
                    mv1, mv2 = m1 * node[ng, bodyID1].momentum, m2 * node[ng, bodyID2].momentum
                    norm1, norm2 = node[ng, bodyID1].grad_domain, node[ng, bodyID2].grad_domain

                    norm, g_mass, xm = (
                        ti.Vector.zero(float, GlobalVariable.DIMENSION),
                        0.0,
                        ti.Vector.zero(float, GlobalVariable.DIMENSION),
                    )
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
                        inv_gmass = 1.0 / g_mass
                        cforce = (mv1 * m2 - m1 * mv2) * inv_gmass

                        ############# Geo-contact #############
                        gsize = MeanValue(grid_size)
                        node_coord = grid_size * ti.Vector(vectorize_id(ng, gnum), dt=int)
                        dext = (xm - node_coord).dot(norm)

                        # Reference: Hammerquist, C. C., Nairn, J. A., 2018. Modeling nanoindentation using the material point method. J. Mater. Res. 33, 1369-1381
                        dist = 0.0
                        if dext <= 0:
                            dist = ti.abs(1.0 - 2.0 * (-dext / (1.25 * gsize)) ** 0.58)
                        elif dext > 0.0:
                            dist = ti.abs(2.0 * (dext / (1.25 * gsize)) ** 0.58 - 1.0)

                        # Reference: Gao L., Guo N., Yang Z. X., Jardine R. J., MPM modeling of pile installation in sand: Contact improvement and quantitative analysis. Comput. Geotech.
                        factor = (1.0 - alpha * dist**beta) / (1.0 + alpha * dist**beta)

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

                        if (
                            ti.static(GlobalVariable.MPMXPBC)
                            or ti.static(GlobalVariable.MPMYPBC)
                            or ti.static(GlobalVariable.MPMZPBC)
                        ):
                            index = ti.Vector([vectorize_id(ng, gnum)])
                            if ti.static(GlobalVariable.MPMXPBC):
                                xindex = -index[0] + gnum[0]
                                if xindex == 0 or xindex == gnum[0]:
                                    temp_index = index
                                    temp_index[0] = xindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, bodyID1]._update_contact_force(-cforce)
                                    node[tideID, bodyID2]._update_contact_force(cforce)
                            if ti.static(GlobalVariable.MPMYPBC):
                                yindex = -index[1] + gnum[1]
                                if yindex == 0 or yindex == gnum[1]:
                                    temp_index = index
                                    temp_index[1] = yindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, bodyID1]._update_contact_force(-cforce)
                                    node[tideID, bodyID2]._update_contact_force(cforce)
                            if ti.static(GlobalVariable.MPMZPBC):
                                zindex = -index[2] + gnum[2]
                                if zindex == 0 or zindex == gnum[2]:
                                    temp_index = index
                                    temp_index[2] = zindex
                                    tideID = linearize(temp_index, gnum)
                                    node[tideID, bodyID1]._update_contact_force(-cforce)
                                    node[tideID, bodyID2]._update_contact_force(cforce)


################## DEM contact ##################
@ti.kernel
def kernel_calc_demcontact_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    gnum: ti.types.vector(2, int),
    grid_size: ti.types.vector(2, float),
    velocity: ti.types.vector(2, float),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    dt: ti.template(),
    polygon_vertices: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    node: ti.template(),
):
    # ti.block_local(dt)
    v1, v2 = velocity, ZEROVEC2f
    normal, tangential = ZEROVEC2f, ZEROVEC2f

    circle_radius = MeanValue(grid_size) * 0.75
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            circle_center = particle[np].x
            v2 = particle[np].v
            m2 = particle[np].m
            k_normal, k_tangential, mu = matProps.kn, matProps.kt, matProps.friction
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
                cforce = delta * k_normal * normal + 2.0 * ti.sqrt(m2 * k_normal) * (v1 - v2).dot(normal) * normal
                if mu > Threshold:
                    val_fslip = mu * delta * k_normal
                    val_fstick = (
                        ti.sqrt(dot2(particle[np].contact_traction))
                        - (v1 - v2).dot(tangential) * k_tangential * dt[None]
                    )
                    val_fstick = ti.abs(val_fstick)
                    cforce += ti.min(val_fslip, val_fstick) * tangential
                particle[np].contact_traction = cforce

                bodyID = int(particle[np].bodyID)
                offset = np * total_nodes
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    extf = shape_mapping(shapefn[ln], cforce)
                    node[nodeID, bodyID]._update_contact_force(extf)

                    if (
                        ti.static(GlobalVariable.MPMXPBC)
                        or ti.static(GlobalVariable.MPMYPBC)
                        or ti.static(GlobalVariable.MPMZPBC)
                    ):
                        index = ti.Vector([vectorize_id(nodeID, gnum)])
                        if ti.static(GlobalVariable.MPMXPBC):
                            xindex = -index[0] + gnum[0]
                            if xindex == 0 or xindex == gnum[0]:
                                temp_index = index
                                temp_index[0] = xindex
                                tideID = linearize(temp_index, gnum)
                                node[tideID, bodyID]._update_contact_force(extf)
                        if ti.static(GlobalVariable.MPMYPBC):
                            yindex = -index[1] + gnum[1]
                            if yindex == 0 or yindex == gnum[1]:
                                temp_index = index
                                temp_index[1] = yindex
                                tideID = linearize(temp_index, gnum)
                                node[tideID, bodyID]._update_contact_force(extf)
                        if ti.static(GlobalVariable.MPMZPBC):
                            zindex = -index[2] + gnum[2]
                            if zindex == 0 or zindex == gnum[2]:
                                temp_index = index
                                temp_index[2] = zindex
                                tideID = linearize(temp_index, gnum)
                                node[tideID, bodyID]._update_contact_force(extf)

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
        edge_length = ti.sqrt(edge_vector[0] ** 2 + edge_vector[1] ** 2)
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
        if ((polygon_vertices[i][1] > circle_center[1]) != (polygon_vertices[j][1] > circle_center[1])) and (
            circle_center[0]
            < (polygon_vertices[j][0] - polygon_vertices[i][0])
            * (circle_center[1] - polygon_vertices[i][1])
            / (polygon_vertices[j][1] - polygon_vertices[i][1])
            + polygon_vertices[i][0]
        ):
            is_inside = not is_inside
        j = i
    # Adjust for penetration
    if is_inside:
        min_distance = -min_distance
        normal_vector = -normal_vector
    return min_distance, normal_vector


# ========================================================= #
#                    Particle shifting                      #
# ========================================================= #
@ti.kernel
def kernel_particle_shifting_delta_correction(
    total_nodes: int,
    particleNum: int,
    grid_size: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    # refer to A.S. Baumgarten, K. Kamrin, Analysis and mitigation of spatial integration errors for the material point method, Internat. J. Numer. Methods Engrg. (2023).
    E2 = 0.0
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].vol > Threshold:
                EI = ti.max(0, -grid_size[0] * grid_size[1] * grid_size[2] + node[ng, nb].vol)
                E2 += EI * EI

    den = 0.0
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            grad_E2 = vec3f(0.0, 0.0, 0.0)
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                EI = ti.max(0, -grid_size[0] * grid_size[1] * grid_size[2] + node[nodeID, bodyID].vol)
                grad_E2 += dshape_fn * EI
            grad_E2 *= 2.0 * particle[np].vol
            den += grad_E2.dot(grad_E2)
            particle[np].grad_E2 = grad_E2

    if den > 0.0:
        for np in range(particleNum):
            particle[np].x -= E2 / den * particle[np].grad_E2


@ti.kernel
def kernel_volume_p2g_fdm_mac_shifting(
    total_nodes: int,
    particleNum: int,
    node_volume: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    node_volume.fill(0)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            volume = particle[np].vol
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                node_volume[nodeID, bodyID] += shape_mapping(shapefn[ln], volume)


@ti.func
def shift_incompressible_particle(np, shift, particle: ti.template()):
    # These are quadrature-position corrections, not physical advection.
    # Transport the local affine velocity with the sample location; otherwise
    # the next APIC transfer gains a spurious -grad(v) @ shift contribution.
    velocity_correction = particle[np].velocity_gradient @ shift
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        if int(particle[np].fix_v[d]) == 0:
            particle[np].v[d] += velocity_correction[d]
    particle[np].x += shift


@ti.kernel
def kernel_particle_shifting_delta_correction_fdm_mac(
    total_nodes: int,
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    max_shift_ratio: float,
    node_volume: ti.template(),
    reference_volume: ti.template(),
    cell_type: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    # FDM/MAC version of the explicit MPM particle shifting correction, using an auxiliary scalar volume field.
    max_shift = max_shift_ratio * grid_size[0]
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        max_shift = ti.min(max_shift, max_shift_ratio * grid_size[d])

    E2 = 0.0
    for ng in range(node_volume.shape[0]):
        for nb in range(node_volume.shape[1]):
            if node_volume[ng, nb] > Threshold:
                EI = ti.max(0.0, -reference_volume[ng, nb] + node_volume[ng, nb])
                E2 += EI * EI

    den = 0.0
    for np in range(particleNum):
        if (
            int(particle[np].materialID) > 0
            and int(particle[np].active) == 1
            and particle_shifting_is_active_fluid(particle[np].x, ghost_cell, cnum, grid_size, cell_type)
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            grad_E2 = ti.Vector.zero(float, GlobalVariable.DIMENSION)
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                EI = ti.max(0.0, -reference_volume[nodeID, bodyID] + node_volume[nodeID, bodyID])
                grad_E2 += dshapefn[ln] * EI
            grad_E2 *= 2.0 * particle[np].vol
            den += grad_E2.dot(grad_E2)
            particle[np].grad_E2 = grad_E2

    if den > 0.0:
        step_scale = E2 / den
        for np in range(particleNum):
            if (
                int(particle[np].materialID) > 0
                and int(particle[np].active) == 1
                and particle_shifting_is_active_fluid(particle[np].x, ghost_cell, cnum, grid_size, cell_type)
            ):
                shift = -step_scale * particle[np].grad_E2
                shift_norm = shift.norm()
                if shift_norm > max_shift:
                    shift *= max_shift / shift_norm
                shift_incompressible_particle(np, shift, particle)


@ti.kernel
def kernel_compute_particle_shifting_energy(node_volume: ti.template(), reference_volume: ti.template()) -> float:
    E2 = 0.0
    for ng in range(node_volume.shape[0]):
        for nb in range(node_volume.shape[1]):
            if node_volume[ng, nb] > Threshold:
                EI = ti.max(0.0, -reference_volume[ng, nb] + node_volume[ng, nb])
                E2 += EI * EI
    return E2


@ti.kernel
def kernel_compute_particle_shifting_gradient_fdm_mac(
    total_nodes: int,
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    reference_volume: ti.template(),
    node_volume: ti.template(),
    cell_type: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
) -> float:
    den = 0.0
    for np in range(particleNum):
        grad_E2 = ti.Vector.zero(float, GlobalVariable.DIMENSION)
        if (
            int(particle[np].materialID) > 0
            and int(particle[np].active) == 1
            and particle_shifting_is_active_fluid(particle[np].x, ghost_cell, cnum, grid_size, cell_type)
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                EI = ti.max(0.0, -reference_volume[nodeID, bodyID] + node_volume[nodeID, bodyID])
                grad_E2 += dshapefn[ln] * EI
            grad_E2 *= 2.0 * particle[np].vol
            den += grad_E2.dot(grad_E2)
        particle[np].grad_E2 = grad_E2
    return den


@ti.kernel
def kernel_apply_particle_shifting_gradient(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    max_shift_ratio: float,
    step_scale: float,
    cell_type: ti.template(),
    particle: ti.template(),
):
    max_shift = max_shift_ratio * grid_size[0]
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        max_shift = ti.min(max_shift, max_shift_ratio * grid_size[d])

    for np in range(particleNum):
        if (
            int(particle[np].materialID) > 0
            and int(particle[np].active) == 1
            and particle_shifting_is_active_fluid(particle[np].x, ghost_cell, cnum, grid_size, cell_type)
        ):
            shift = -step_scale * particle[np].grad_E2
            shift_norm = shift.norm()
            if shift_norm > max_shift:
                shift *= max_shift / shift_norm
            shift_incompressible_particle(np, shift, particle)


@ti.func
def fdm_regular_axis_shape_grad(position, node_coord, inv_grid_size, psize, btype: int):
    shape = 0.0
    dshape = 0.0
    if ti.static(GlobalVariable.SHAPEFUNCTION == 0):
        shape = ShapeLinear(position, node_coord, inv_grid_size, 0)
        dshape = GShapeLinear(position, node_coord, inv_grid_size, 0)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 1):
        shape = ShapeGIMP(position, node_coord, inv_grid_size, psize)
        dshape = GShapeGIMP(position, node_coord, inv_grid_size, psize)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 2):
        shape = ShapeBsplineQ(position, node_coord, inv_grid_size, btype)
        dshape = GShapeBsplineQ(position, node_coord, inv_grid_size, btype)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 3):
        shape = ShapeBsplineC(position, node_coord, inv_grid_size, btype)
        dshape = GShapeBsplineC(position, node_coord, inv_grid_size, btype)
    return shape, dshape


@ti.func
def fdm_regular_axis_mapped_index(index: int, count: int, ghost_cell: int, periodic: ti.template()):
    # Regular shifting nodes share the mesh origin at -ghost_cell * h.
    # Wrap storage indices, never the coordinates used to evaluate the basis.
    mapped = index + ghost_cell
    valid = True
    if ti.static(periodic):
        mapped = index % (count - 1 - 2 * ghost_cell) + ghost_cell
    else:
        valid = mapped >= 0 and mapped < count
    return mapped, valid


@ti.func
def fdm_regular_axis_basis_integral(lower, upper, half_length, btype):
    integral = 0.0
    if ti.static(GlobalVariable.SHAPEFUNCTION == 1):
        # GIMP is the box average of the linear hat. Its primitive is the
        # difference of two integrated hat primitives; no quadrature tuning.
        for endpoint in ti.static(range(2)):
            r = 1.0 * (lower if endpoint == 0 else upper)
            r = ti.max(-1.0 - half_length, ti.min(1.0 + half_length, r))
            cdf = 0.0
            if half_length > 0.0:
                hi, lo = r + half_length, r - half_length
                high = (ti.max(hi + 1.0, 0.0) ** 3 - 2.0 * ti.max(hi, 0.0) ** 3 + ti.max(hi - 1.0, 0.0) ** 3) / 6.0
                low = (ti.max(lo + 1.0, 0.0) ** 3 - 2.0 * ti.max(lo, 0.0) ** 3 + ti.max(lo - 1.0, 0.0) ** 3) / 6.0
                cdf = (high - low) / (2.0 * half_length)
            else:
                cdf = 0.5 * (ti.max(r + 1.0, 0.0) ** 2 - 2.0 * ti.max(r, 0.0) ** 2 + ti.max(r - 1.0, 0.0) ** 2)
            integral += (2 * endpoint - 1) * cdf
    else:
        # All Linear/Q2/C3 pieces (including modified boundary bases) have
        # degree <= 3, with knots on this half-cell partition. Two-point
        # Gauss integration on each clipped piece is therefore exact.
        for segment in range(8):
            lo = ti.max(lower, -2.0 + 0.5 * segment)
            hi = ti.min(upper, -1.5 + 0.5 * segment)
            if hi > lo:
                middle, radius = 0.5 * (lo + hi), 0.5 * (hi - lo)
                left, _ = fdm_regular_axis_shape_grad(middle - radius / ti.sqrt(3.0), 0.0, 1.0, half_length, btype)
                right, _ = fdm_regular_axis_shape_grad(middle + radius / ti.sqrt(3.0), 0.0, 1.0, half_length, btype)
                integral += radius * (left + right)
    return integral


@ti.kernel
def kernel_fdm_shifting_reference_volume(
    ghost_cell: int,
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    particle_lengths: ti.template(),
    boundtype: ti.template(),
    reference_volume: ti.template(),
):
    # Integrate over the physical mesh, not its exterior ghost padding. This
    # is intrinsic nodal volume, independent of current particle clustering
    # or AIR labels; it cannot fill voids by redefining them as fluid.
    counts = gnum - 1 - 2 * ghost_cell
    for node_id, body_id in reference_volume:
        index = vectorize_id(node_id, gnum)
        volume = 1.0
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            raw = index[d] - ghost_cell
            fraction = 1.0
            if ti.static((GlobalVariable.MPMXPBC, GlobalVariable.MPMYPBC, GlobalVariable.MPMZPBC)[d]):
                # Only the unique periodic nodes receive mapped volume.
                if raw < 0 or raw >= counts[d]:
                    fraction = 0.0
            else:
                btype = 0
                if ti.static(GlobalVariable.SHAPEFUNCTION == 2 or GlobalVariable.SHAPEFUNCTION == 3):
                    btype = int(boundtype[node_id, body_id][d])
                fraction = fdm_regular_axis_basis_integral(
                    -raw, counts[d] - raw, particle_lengths[body_id][d] / grid_size[d], btype
                )
            volume *= grid_size[d] * fraction
        reference_volume[node_id, body_id] = volume


@ti.kernel
def kernel_compute_particle_shifting_gradient_fdm_on_the_fly_3d(
    influenced_node: int,
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    gnum: ti.types.vector(3, int),
    reference_volume: ti.template(),
    node_volume: ti.template(),
    cell_type: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
) -> float:
    den = 0.0
    for np in range(particleNum):
        grad_E2 = vec3f(0.0, 0.0, 0.0)
        if (
            int(particle[np].materialID) > 0
            and int(particle[np].active) == 1
            and particle_shifting_is_active_fluid(particle[np].x, ghost_cell, cnum, grid_size, cell_type)
        ):
            bodyID = int(particle[np].bodyID)
            position, psize = particle[np].x, particle_lengths[bodyID]
            base_bound = ti.floor((position - psize) * igrid_size, int)
            axis_shape = ti.Matrix.zero(float, 3, GlobalVariable.INFLUENCENODE)
            axis_dshape = ti.Matrix.zero(float, 3, GlobalVariable.INFLUENCENODE)
            axis_index = ti.Matrix.zero(int, 3, GlobalVariable.INFLUENCENODE)
            axis_valid = ti.Matrix.zero(int, 3, GlobalVariable.INFLUENCENODE)

            for local_id in ti.static(range(GlobalVariable.INFLUENCENODE)):
                raw_index = base_bound[0] + local_id
                mapped, valid = fdm_regular_axis_mapped_index(raw_index, gnum[0], ghost_cell, GlobalVariable.MPMXPBC)
                btype = 0
                if ti.static(GlobalVariable.SHAPEFUNCTION == 2 or GlobalVariable.SHAPEFUNCTION == 3):
                    btype = mac_boundary_type_1d(mapped, gnum[0])
                    if ti.static(GlobalVariable.MPMXPBC):
                        btype = 0
                if valid:
                    shape, dshape = fdm_regular_axis_shape_grad(
                        position[0], raw_index * grid_size[0], igrid_size[0], psize[0], btype
                    )
                    axis_shape[0, local_id] = shape
                    axis_dshape[0, local_id] = dshape
                    axis_valid[0, local_id] = 1
                axis_index[0, local_id] = mapped

            for local_id in ti.static(range(GlobalVariable.INFLUENCENODE)):
                raw_index = base_bound[1] + local_id
                mapped, valid = fdm_regular_axis_mapped_index(raw_index, gnum[1], ghost_cell, GlobalVariable.MPMYPBC)
                btype = 0
                if ti.static(GlobalVariable.SHAPEFUNCTION == 2 or GlobalVariable.SHAPEFUNCTION == 3):
                    btype = mac_boundary_type_1d(mapped, gnum[1])
                    if ti.static(GlobalVariable.MPMYPBC):
                        btype = 0
                if valid:
                    shape, dshape = fdm_regular_axis_shape_grad(
                        position[1], raw_index * grid_size[1], igrid_size[1], psize[1], btype
                    )
                    axis_shape[1, local_id] = shape
                    axis_dshape[1, local_id] = dshape
                    axis_valid[1, local_id] = 1
                axis_index[1, local_id] = mapped

            for local_id in ti.static(range(GlobalVariable.INFLUENCENODE)):
                raw_index = base_bound[2] + local_id
                mapped, valid = fdm_regular_axis_mapped_index(raw_index, gnum[2], ghost_cell, GlobalVariable.MPMZPBC)
                btype = 0
                if ti.static(GlobalVariable.SHAPEFUNCTION == 2 or GlobalVariable.SHAPEFUNCTION == 3):
                    btype = mac_boundary_type_1d(mapped, gnum[2])
                    if ti.static(GlobalVariable.MPMZPBC):
                        btype = 0
                if valid:
                    shape, dshape = fdm_regular_axis_shape_grad(
                        position[2], raw_index * grid_size[2], igrid_size[2], psize[2], btype
                    )
                    axis_shape[2, local_id] = shape
                    axis_dshape[2, local_id] = dshape
                    axis_valid[2, local_id] = 1
                axis_index[2, local_id] = mapped

            for offset in ti.grouped(
                ti.ndrange(GlobalVariable.INFLUENCENODE, GlobalVariable.INFLUENCENODE, GlobalVariable.INFLUENCENODE)
            ):
                mapped_index = vec3i(0, 0, 0)
                shape = vec3f(0.0, 0.0, 0.0)
                dshape = vec3f(0.0, 0.0, 0.0)
                inside = True
                for d in ti.static(range(3)):
                    for local_id in ti.static(range(GlobalVariable.INFLUENCENODE)):
                        if offset[d] == local_id:
                            mapped_index[d] = axis_index[d, local_id]
                            shape[d] = axis_shape[d, local_id]
                            dshape[d] = axis_dshape[d, local_id]
                            inside = inside and axis_valid[d, local_id] == 1
                if inside:
                    nodeID = int(mapped_index[0] + mapped_index[1] * gnum[0] + mapped_index[2] * gnum[0] * gnum[1])
                    EI = ti.max(0.0, -reference_volume[nodeID, bodyID] + node_volume[nodeID, bodyID])
                    grad_E2 += (
                        vec3f(
                            [
                                dshape[0] * shape[1] * shape[2],
                                shape[0] * dshape[1] * shape[2],
                                shape[0] * shape[1] * dshape[2],
                            ]
                        )
                        * EI
                    )
            grad_E2 *= 2.0 * particle[np].vol
            den += grad_E2.dot(grad_E2)
        particle[np].grad_E2 = grad_E2
    return den


@ti.func
def fdm_regular_node_shape_grad_2d(position, node_coords, igrid_size, psize, btype):
    shape_fn = vec2f(0.0, 0.0)
    dshape_fn = vec2f(0.0, 0.0)
    if ti.static(GlobalVariable.SHAPEFUNCTION == 0):
        for d in ti.static(range(2)):
            shape_fn[d] = ShapeLinear(position[d], node_coords[d], igrid_size[d], 0)
            dshape_fn[d] = GShapeLinear(position[d], node_coords[d], igrid_size[d], 0)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 1):
        for d in ti.static(range(2)):
            shape_fn[d] = ShapeGIMP(position[d], node_coords[d], igrid_size[d], psize[d])
            dshape_fn[d] = GShapeGIMP(position[d], node_coords[d], igrid_size[d], psize[d])
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 2):
        for d in ti.static(range(2)):
            shape_fn[d] = ShapeBsplineQ(position[d], node_coords[d], igrid_size[d], int(btype[d]))
            dshape_fn[d] = GShapeBsplineQ(position[d], node_coords[d], igrid_size[d], int(btype[d]))
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 3):
        for d in ti.static(range(2)):
            shape_fn[d] = ShapeBsplineC(position[d], node_coords[d], igrid_size[d], int(btype[d]))
            dshape_fn[d] = GShapeBsplineC(position[d], node_coords[d], igrid_size[d], int(btype[d]))
    weight = shape_fn[0] * shape_fn[1]
    grad = vec2f([dshape_fn[0] * shape_fn[1], shape_fn[0] * dshape_fn[1]])
    return weight, grad


@ti.func
def fdm_regular_node_shape_grad_3d(position, node_coords, igrid_size, psize, btype):
    shape_fn = vec3f(0.0, 0.0, 0.0)
    dshape_fn = vec3f(0.0, 0.0, 0.0)
    if ti.static(GlobalVariable.SHAPEFUNCTION == 0):
        for d in ti.static(range(3)):
            shape_fn[d] = ShapeLinear(position[d], node_coords[d], igrid_size[d], 0)
            dshape_fn[d] = GShapeLinear(position[d], node_coords[d], igrid_size[d], 0)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 1):
        for d in ti.static(range(3)):
            shape_fn[d] = ShapeGIMP(position[d], node_coords[d], igrid_size[d], psize[d])
            dshape_fn[d] = GShapeGIMP(position[d], node_coords[d], igrid_size[d], psize[d])
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 2):
        for d in ti.static(range(3)):
            shape_fn[d] = ShapeBsplineQ(position[d], node_coords[d], igrid_size[d], int(btype[d]))
            dshape_fn[d] = GShapeBsplineQ(position[d], node_coords[d], igrid_size[d], int(btype[d]))
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 3):
        for d in ti.static(range(3)):
            shape_fn[d] = ShapeBsplineC(position[d], node_coords[d], igrid_size[d], int(btype[d]))
            dshape_fn[d] = GShapeBsplineC(position[d], node_coords[d], igrid_size[d], int(btype[d]))
    weight = shape_fn[0] * shape_fn[1] * shape_fn[2]
    grad = vec3f(
        [
            dshape_fn[0] * shape_fn[1] * shape_fn[2],
            shape_fn[0] * dshape_fn[1] * shape_fn[2],
            shape_fn[0] * shape_fn[1] * dshape_fn[2],
        ]
    )
    return weight, grad


@ti.kernel
def kernel_volume_p2g_fdm_shifting_on_the_fly_2d(
    influenced_node: int,
    particleNum: int,
    ghost_cell: int,
    grid_size: ti.types.vector(2, float),
    igrid_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    node_volume: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
    boundtype: ti.template(),
):
    node_volume.fill(0)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            position, psize = particle[np].x, particle_lengths[bodyID]
            base_bound = ti.floor((position - psize) * igrid_size, int)
            volume = particle[np].vol
            for j in range(base_bound[1], base_bound[1] + influenced_node):
                jp, valid_j = fdm_regular_axis_mapped_index(j, gnum[1], ghost_cell, GlobalVariable.MPMYPBC)
                if not valid_j:
                    continue
                for i in range(base_bound[0], base_bound[0] + influenced_node):
                    ip, valid_i = fdm_regular_axis_mapped_index(i, gnum[0], ghost_cell, GlobalVariable.MPMXPBC)
                    if not valid_i:
                        continue
                    nodeID = int(ip + jp * gnum[0])
                    node_coords = vec2i(i, j) * grid_size
                    btype = vec2i(0, 0)
                    if ti.static(GlobalVariable.SHAPEFUNCTION == 2 or GlobalVariable.SHAPEFUNCTION == 3):
                        btype = ti.cast(boundtype[nodeID, bodyID], ti.i32)
                        if ti.static(GlobalVariable.MPMXPBC):
                            btype[0] = 0
                        if ti.static(GlobalVariable.MPMYPBC):
                            btype[1] = 0
                    weight, _ = fdm_regular_node_shape_grad_2d(position, node_coords, igrid_size, psize, btype)
                    if weight > Threshold:
                        node_volume[nodeID, bodyID] += shape_mapping(weight, volume)


@ti.kernel
def kernel_particle_shifting_delta_correction_fdm_on_the_fly_2d(
    influenced_node: int,
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(2, int),
    grid_size: ti.types.vector(2, float),
    igrid_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    max_shift_ratio: float,
    node_volume: ti.template(),
    reference_volume: ti.template(),
    cell_type: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
    boundtype: ti.template(),
):
    max_shift = ti.min(max_shift_ratio * grid_size[0], max_shift_ratio * grid_size[1])

    E2 = 0.0
    for ng in range(node_volume.shape[0]):
        for nb in range(node_volume.shape[1]):
            if node_volume[ng, nb] > Threshold:
                EI = ti.max(0.0, -reference_volume[ng, nb] + node_volume[ng, nb])
                E2 += EI * EI

    den = 0.0
    for np in range(particleNum):
        if (
            int(particle[np].materialID) > 0
            and int(particle[np].active) == 1
            and particle_shifting_is_active_fluid(particle[np].x, ghost_cell, cnum, grid_size, cell_type)
        ):
            bodyID = int(particle[np].bodyID)
            position, psize = particle[np].x, particle_lengths[bodyID]
            base_bound = ti.floor((position - psize) * igrid_size, int)
            grad_E2 = vec2f(0.0, 0.0)
            for j in range(base_bound[1], base_bound[1] + influenced_node):
                jp, valid_j = fdm_regular_axis_mapped_index(j, gnum[1], ghost_cell, GlobalVariable.MPMYPBC)
                if not valid_j:
                    continue
                for i in range(base_bound[0], base_bound[0] + influenced_node):
                    ip, valid_i = fdm_regular_axis_mapped_index(i, gnum[0], ghost_cell, GlobalVariable.MPMXPBC)
                    if not valid_i:
                        continue
                    nodeID = int(ip + jp * gnum[0])
                    node_coords = vec2i(i, j) * grid_size
                    btype = vec2i(0, 0)
                    if ti.static(GlobalVariable.SHAPEFUNCTION == 2 or GlobalVariable.SHAPEFUNCTION == 3):
                        btype = ti.cast(boundtype[nodeID, bodyID], ti.i32)
                        if ti.static(GlobalVariable.MPMXPBC):
                            btype[0] = 0
                        if ti.static(GlobalVariable.MPMYPBC):
                            btype[1] = 0
                    _, dshape_fn = fdm_regular_node_shape_grad_2d(position, node_coords, igrid_size, psize, btype)
                    EI = ti.max(0.0, -reference_volume[nodeID, bodyID] + node_volume[nodeID, bodyID])
                    grad_E2 += dshape_fn * EI
            grad_E2 *= 2.0 * particle[np].vol
            den += grad_E2.dot(grad_E2)
            particle[np].grad_E2 = grad_E2

    if den > 0.0:
        step_scale = E2 / den
        for np in range(particleNum):
            if (
                int(particle[np].materialID) > 0
                and int(particle[np].active) == 1
                and particle_shifting_is_active_fluid(particle[np].x, ghost_cell, cnum, grid_size, cell_type)
            ):
                shift = -step_scale * particle[np].grad_E2
                shift_norm = shift.norm()
                if shift_norm > max_shift:
                    shift *= max_shift / shift_norm
                shift_incompressible_particle(np, shift, particle)


@ti.kernel
def kernel_volume_p2g_fdm_shifting_on_the_fly_3d(
    influenced_node: int,
    particleNum: int,
    ghost_cell: int,
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    gnum: ti.types.vector(3, int),
    node_volume: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
    boundtype: ti.template(),
):
    node_volume.fill(0)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            position, psize = particle[np].x, particle_lengths[bodyID]
            base_bound = ti.floor((position - psize) * igrid_size, int)
            volume = particle[np].vol
            for k in range(base_bound[2], base_bound[2] + influenced_node):
                kp, valid_k = fdm_regular_axis_mapped_index(k, gnum[2], ghost_cell, GlobalVariable.MPMZPBC)
                if not valid_k:
                    continue
                for j in range(base_bound[1], base_bound[1] + influenced_node):
                    jp, valid_j = fdm_regular_axis_mapped_index(j, gnum[1], ghost_cell, GlobalVariable.MPMYPBC)
                    if not valid_j:
                        continue
                    for i in range(base_bound[0], base_bound[0] + influenced_node):
                        ip, valid_i = fdm_regular_axis_mapped_index(i, gnum[0], ghost_cell, GlobalVariable.MPMXPBC)
                        if not valid_i:
                            continue
                        nodeID = int(ip + jp * gnum[0] + kp * gnum[0] * gnum[1])
                        node_coords = vec3i(i, j, k) * grid_size
                        btype = vec3i(0, 0, 0)
                        if ti.static(GlobalVariable.SHAPEFUNCTION == 2 or GlobalVariable.SHAPEFUNCTION == 3):
                            btype = ti.cast(boundtype[nodeID, bodyID], ti.i32)
                            if ti.static(GlobalVariable.MPMXPBC):
                                btype[0] = 0
                            if ti.static(GlobalVariable.MPMYPBC):
                                btype[1] = 0
                            if ti.static(GlobalVariable.MPMZPBC):
                                btype[2] = 0
                        weight, _ = fdm_regular_node_shape_grad_3d(position, node_coords, igrid_size, psize, btype)
                        if weight > Threshold:
                            node_volume[nodeID, bodyID] += shape_mapping(weight, volume)


@ti.kernel
def kernel_particle_shifting_delta_correction_fdm_on_the_fly_3d(
    influenced_node: int,
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    gnum: ti.types.vector(3, int),
    max_shift_ratio: float,
    node_volume: ti.template(),
    reference_volume: ti.template(),
    cell_type: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
    boundtype: ti.template(),
):
    max_shift = max_shift_ratio * grid_size[0]
    for d in ti.static(range(3)):
        max_shift = ti.min(max_shift, max_shift_ratio * grid_size[d])

    E2 = 0.0
    for ng in range(node_volume.shape[0]):
        for nb in range(node_volume.shape[1]):
            if node_volume[ng, nb] > Threshold:
                EI = ti.max(0.0, -reference_volume[ng, nb] + node_volume[ng, nb])
                E2 += EI * EI

    den = 0.0
    for np in range(particleNum):
        if (
            int(particle[np].materialID) > 0
            and int(particle[np].active) == 1
            and particle_shifting_is_active_fluid(particle[np].x, ghost_cell, cnum, grid_size, cell_type)
        ):
            bodyID = int(particle[np].bodyID)
            position, psize = particle[np].x, particle_lengths[bodyID]
            base_bound = ti.floor((position - psize) * igrid_size, int)
            grad_E2 = vec3f(0.0, 0.0, 0.0)
            for k in range(base_bound[2], base_bound[2] + influenced_node):
                kp, valid_k = fdm_regular_axis_mapped_index(k, gnum[2], ghost_cell, GlobalVariable.MPMZPBC)
                if not valid_k:
                    continue
                for j in range(base_bound[1], base_bound[1] + influenced_node):
                    jp, valid_j = fdm_regular_axis_mapped_index(j, gnum[1], ghost_cell, GlobalVariable.MPMYPBC)
                    if not valid_j:
                        continue
                    for i in range(base_bound[0], base_bound[0] + influenced_node):
                        ip, valid_i = fdm_regular_axis_mapped_index(i, gnum[0], ghost_cell, GlobalVariable.MPMXPBC)
                        if not valid_i:
                            continue
                        nodeID = int(ip + jp * gnum[0] + kp * gnum[0] * gnum[1])
                        node_coords = vec3i(i, j, k) * grid_size
                        btype = vec3i(0, 0, 0)
                        if ti.static(GlobalVariable.SHAPEFUNCTION == 2 or GlobalVariable.SHAPEFUNCTION == 3):
                            btype = ti.cast(boundtype[nodeID, bodyID], ti.i32)
                            if ti.static(GlobalVariable.MPMXPBC):
                                btype[0] = 0
                            if ti.static(GlobalVariable.MPMYPBC):
                                btype[1] = 0
                            if ti.static(GlobalVariable.MPMZPBC):
                                btype[2] = 0
                        _, dshape_fn = fdm_regular_node_shape_grad_3d(position, node_coords, igrid_size, psize, btype)
                        EI = ti.max(0.0, -reference_volume[nodeID, bodyID] + node_volume[nodeID, bodyID])
                        grad_E2 += dshape_fn * EI
            grad_E2 *= 2.0 * particle[np].vol
            den += grad_E2.dot(grad_E2)
            particle[np].grad_E2 = grad_E2

    if den > 0.0:
        step_scale = E2 / den
        for np in range(particleNum):
            if (
                int(particle[np].materialID) > 0
                and int(particle[np].active) == 1
                and particle_shifting_is_active_fluid(particle[np].x, ghost_cell, cnum, grid_size, cell_type)
            ):
                shift = -step_scale * particle[np].grad_E2
                shift_norm = shift.norm()
                if shift_norm > max_shift:
                    shift *= max_shift / shift_norm
                shift_incompressible_particle(np, shift, particle)


# ======================================== Implicit MPM ======================================== #
# ========================================================= #
#                    Preprocess dofs                        #
# ========================================================= #
@ti.kernel
def find_active_node(gridSum: int, cutoff: float, node: ti.template(), flag: ti.template()):
    flag.fill(0)
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                flag[ng + nb * gridSum] = 1


@ti.kernel
def estimate_active_grid_dofs(cutoff: float, node: ti.template()) -> int:
    total_active_nodes = 0
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                total_active_nodes += 1
    return GlobalVariable.DIMENSION * total_active_nodes


@ti.kernel
def set_active_dofs(gridSum: int, cutoff: float, node: ti.template(), flag: ti.template()) -> int:
    total_dof = GlobalVariable.DIMENSION * flag[flag.shape[0] - 1]
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dofs = GlobalVariable.DIMENSION * (flag[ng + nb * gridSum] - 1)
                flag[ng + nb * gridSum] = dofs
    return total_dof


# ========================================================= #
#            Particle Momentum to Grid (iP2G)               #
# ========================================================= #
@ti.kernel
def kernel_mass_momentum_acceleration_force_ip2g(
    total_nodes: int,
    particleNum: int,
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
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
def kernel_internal_force_p2g(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec3f(
                    [
                        dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                        dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                        dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5],
                    ]
                )
                node[nodeID, bodyID]._update_internal_force(internal_force)


@ti.kernel
def kernel_internal_force_p2g_2D(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fInt = particle[np]._compute_internal_force()
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape_fn = dshapefn[ln]
                internal_force = vec2f(
                    [dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3], dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]]
                )
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
def kernel_compute_nodal_kinematics_newmark(
    beta: float, gamma: float, cutoff: float, node: ti.template(), dt: ti.template()
):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb]._update_nodal_kinematic_newmark(beta, gamma, dt)


@ti.kernel
def kernel_update_nodal_disp(
    gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), unknown_vector: ti.template()
):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                disp = vec3f(unknown_vector[dof0], unknown_vector[dof0 + 1], unknown_vector[dof0 + 2])
                node[ng, nb]._update_nodal_disp(disp)


@ti.kernel
def kernel_update_nodal_disp_2D(
    gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), unknown_vector: ti.template()
):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                disp = vec2f(unknown_vector[dof0], unknown_vector[dof0 + 1])
                node[ng, nb]._update_nodal_disp(disp)


# ========================================================= #
#                 Grid to Particle (G2P)                    #
# ========================================================= #
@ti.kernel
def kernel_kinemaitc_ig2p(
    total_nodes: int,
    alpha: float,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
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
def kernel_compute_stress_strain_newmark_2D(
    dt: ti.template(),
    start_index: int,
    end_index: int,
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    stiffness_matrix: ti.template(),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress0
            stress = matProps.ComputeStress2D(np, previous_stress, velocity_gradient, stateVars, dt)
            stiffness_matrix[np] = matProps.compute_stiffness_tensor(np, stress, stateVars)
            particle[np].stress = stress


@ti.kernel
def kernel_compute_stress_strain_newmark_elastic_tangent_2D(
    dt: ti.template(),
    start_index: int,
    end_index: int,
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    stiffness_matrix: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress0
            stress = matProps.ComputeStress2D(np, previous_stress, velocity_gradient, stateVars, dt)
            stiffness_matrix[np] = matProps.compute_elastic_tensor(np, stress, stateVars)
            particle[np].stress = stress


@ti.kernel
def kernel_compute_stress_strain_newmark(
    dt: ti.template(),
    start_index: int,
    end_index: int,
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    stiffness_matrix: ti.template(),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress0
            stress = matProps.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
            stiffness_matrix[np] = matProps.compute_stiffness_tensor(np, stress, stateVars)
            particle[np].stress = stress


@ti.kernel
def kernel_compute_stress_strain_newmark_elastic_tangent(
    dt: ti.template(),
    start_index: int,
    end_index: int,
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    stiffness_matrix: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            velocity_gradient = particle[np].velocity_gradient
            previous_stress = particle[np].stress0
            stress = matProps.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
            stiffness_matrix[np] = matProps.compute_elastic_tensor(np, stress, stateVars)
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
def kernel_update_displacement_gradient(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            displacement_gradient = ZEROMAT3x3
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gu = node[nodeID, bodyID].displacement
                dshape_fn = dshapefn[ln]
                displacement_gradient += outer_product(gu, dshape_fn)
            velocity_gradient = displacement_gradient / dt[None]
            particle[np].velocity_gradient = truncation(velocity_gradient)

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps.update_particle_volume(np, velocity_gradient, stateVars, dt)


@ti.kernel
def kernel_update_displacement_gradient_affine(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
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
                Bp += shape_fn * outer_product(gu, pointer)
            velocity_gradient = truncation(Bp @ Wp.inverse()) / dt[None] if Wp.determinant() > Threshold else ZEROMAT3x3
            particle[np].velocity_gradient = velocity_gradient


@ti.kernel
def kernel_update_displacement_gradient_bbar(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            displacement_gradient = ZEROMAT3x3
            strain_incre_trace = ZEROVEC3f
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gv = node[nodeID, bodyID].displacement
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                temp_dshape = (dshape_fnc - dshape_fn) / 3.0

                average_bmatrix = temp_dshape[0] * gv[0] + temp_dshape[1] * gv[1] + temp_dshape[2] * gv[2]
                displacement_gradient += outer_product(gv, dshape_fn)
                displacement_gradient[0, 0] += average_bmatrix
                displacement_gradient[1, 1] += average_bmatrix
                displacement_gradient[2, 2] += average_bmatrix

                strain_incre_trace[0] += dshape_fn[0] * gv[0]
                strain_incre_trace[1] += dshape_fn[1] * gv[1]
                strain_incre_trace[2] += dshape_fn[2] * gv[2]
            particle[np].velocity_gradient = truncation(displacement_gradient) / dt[None]

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps.update_particle_volume_bbar(
                np, strain_incre_trace / dt[None], stateVars, dt
            )


@ti.kernel
def kernel_update_displacement_gradient_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            displacement_gradient = ZEROMAT2x2
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                gu = node[nodeID, bodyID].displacement
                dshape_fn = dshapefn[ln]
                displacement_gradient += outer_product2D(gu, dshape_fn)
            velocity_gradient = displacement_gradient / dt[None]
            particle[np].velocity_gradient = truncation(velocity_gradient)

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps.update_particle_volume_2D(
                np, velocity_gradient, stateVars, dt
            )


@ti.kernel
def kernel_update_displacement_gradient_affine_2D(
    total_nodes: int,
    particleNum: int,
    gnum: ti.types.vector(2, int),
    grid_size: ti.types.vector(2, float),
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
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
            velocity_gradient = truncation(Bp @ Wp.inverse()) / dt[None] if Wp.determinant() > Threshold else ZEROMAT2x2
            particle[np].velocity_gradient = velocity_gradient


@ti.kernel
def kernel_update_displacement_gradient_bbar_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    materialID: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    dshapefnc: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            displacement_gradient = ZEROMAT2x2
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                displacement = node[nodeID, bodyID].displacement
                dshape_fn = dshapefn[ln]
                dshape_fnc = dshapefnc[ln]
                displacement_gradient += bbar_velocity_gradient_2d(displacement, dshape_fn, dshape_fnc)
            velocity_gradient = displacement_gradient / dt[None]
            particle[np].velocity_gradient = truncation(velocity_gradient)

            previous_volume = particle[np].vol0
            particle[np].vol = previous_volume * matProps.update_particle_volume_bbar_2D(
                np, velocity_gradient, stateVars, dt
            )


# ======================================== Incompressible flows ======================================== #
# ========================================================= #
#                    Preprocess dofs                        #
# ========================================================= #
@ti.kernel
def find_active_fdm_cell(
    ghost_cell: int,
    cellSum: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    flag: ti.template(),
):
    flag.fill(0)
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, cnum[0] - 2 * ghost_cell), (0, cnum[1] - 2 * ghost_cell))):
            linear_cell_id = linearize(I, cnum - 2 * ghost_cell)
            flag[linear_cell_id + 0 * cellSum] = is_fluid(cell_type[I])
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(
            ti.ndrange((0, cnum[0] - 2 * ghost_cell), (0, cnum[1] - 2 * ghost_cell), (0, cnum[2] - 2 * ghost_cell))
        ):
            linear_cell_id = linearize(I, cnum - 2 * ghost_cell)
            flag[linear_cell_id + 0 * cellSum] = is_fluid(cell_type[I])


@ti.kernel
def estimate_active_fdm_cell_dofs(
    ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), cell_type: ti.template()
) -> int:
    total_active_cells = 0
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, cnum[0] - 2 * ghost_cell), (0, cnum[1] - 2 * ghost_cell))):
            if is_fluid(cell_type[I]):
                total_active_cells += 1
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(
            ti.ndrange((0, cnum[0] - 2 * ghost_cell), (0, cnum[1] - 2 * ghost_cell), (0, cnum[2] - 2 * ghost_cell))
        ):
            if is_fluid(cell_type[I]):
                total_active_cells += 1
    return total_active_cells


@ti.kernel
def set_active_fdm_cell_dofs(
    ghost_cell: int,
    cellSum: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    flag: ti.template(),
) -> int:
    total_dof = flag[flag.shape[0] - 1]
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, cnum[0] - 2 * ghost_cell), (0, cnum[1] - 2 * ghost_cell))):
            linear_cell_id = linearize(I, cnum - 2 * ghost_cell)
            dofs = flag[linear_cell_id + 0 * cellSum] - 1
            flag[linear_cell_id + 0 * cellSum] = dofs
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(
            ti.ndrange((0, cnum[0] - 2 * ghost_cell), (0, cnum[1] - 2 * ghost_cell), (0, cnum[2] - 2 * ghost_cell))
        ):
            linear_cell_id = linearize(I, cnum - 2 * ghost_cell)
            dofs = flag[linear_cell_id + 0 * cellSum] - 1
            flag[linear_cell_id + 0 * cellSum] = dofs
    return total_dof


@ti.kernel
def find_active_density_projection_cell(
    ghost_cell: int,
    cellSum: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    interior_only: ti.template(),
    flag: ti.template(),
):
    flag.fill(0)
    active_cnum = cnum - 2 * ghost_cell
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            linear_cell_id = linearize(I, active_cnum)
            flag[linear_cell_id + 0 * cellSum] = density_projection_cell_is_active(
                I, ghost_cell, cnum, cell_type, interior_only
            )
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            linear_cell_id = linearize(I, active_cnum)
            flag[linear_cell_id + 0 * cellSum] = density_projection_cell_is_active(
                I, ghost_cell, cnum, cell_type, interior_only
            )


@ti.kernel
def set_active_density_projection_cell_dofs(
    ghost_cell: int,
    cellSum: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    interior_only: ti.template(),
    flag: ti.template(),
) -> int:
    total_dof = flag[flag.shape[0] - 1]
    active_cnum = cnum - 2 * ghost_cell
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            linear_cell_id = linearize(I, active_cnum)
            if density_projection_cell_is_active(I, ghost_cell, cnum, cell_type, interior_only):
                flag[linear_cell_id + 0 * cellSum] = flag[linear_cell_id + 0 * cellSum] - 1
            else:
                flag[linear_cell_id + 0 * cellSum] = -1
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            linear_cell_id = linearize(I, active_cnum)
            if density_projection_cell_is_active(I, ghost_cell, cnum, cell_type, interior_only):
                flag[linear_cell_id + 0 * cellSum] = flag[linear_cell_id + 0 * cellSum] - 1
            else:
                flag[linear_cell_id + 0 * cellSum] = -1
    return total_dof


@ti.kernel
def kernel_copy_density_projection_mg_cell_type(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    interior_only: ti.template(),
    grid_type: ti.template(),
):
    for I in ti.grouped(grid_type):
        if density_projection_cell_is_active(I, ghost_cell, cnum, cell_type, interior_only):
            grid_type[I] = 1
        elif get_offset_cell_type(I, ghost_cell, cnum, cell_type) == 2:
            grid_type[I] = 2
        else:
            grid_type[I] = 0


@ti.func
def mac_boundary_type_1d(index: int, count: int):
    btype = 0
    side = ti.min(2, index) - ti.min(count - 1 - index, 2)
    if side < 0:
        btype = side + 3
    elif side > 0:
        btype = side + 2
    return btype


@ti.func
def mac_get_offset_cell_type(
    index, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), cell_type_field: ti.template()
):
    cell_type = 2
    inside = True
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        inside = inside and index[d] >= -ghost_cell and index[d] < cnum[d] - ghost_cell
    if inside:
        cell_type = int(cell_type_field[index])
    return cell_type


@ti.func
def mac_domain_boundary_is_solid(
    index,
    normal_dir: ti.template(),
    active_cnum,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type_field: ti.template(),
):
    solid = False
    cell = index
    if index[normal_dir] == 0:
        cell[normal_dir] = -1
        solid = mac_get_offset_cell_type(cell, ghost_cell, cnum, cell_type_field) == 2
    elif index[normal_dir] == active_cnum[normal_dir]:
        cell[normal_dir] = active_cnum[normal_dir]
        solid = mac_get_offset_cell_type(cell, ghost_cell, cnum, cell_type_field) == 2
    return solid


@ti.func
def mac_normal_ghost_is_solid(
    index,
    normal_dir: ti.template(),
    active_cnum,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type_field: ti.template(),
):
    solid = False
    cell = index
    if index[normal_dir] < 0:
        cell[normal_dir] = -1
        solid = mac_get_offset_cell_type(cell, ghost_cell, cnum, cell_type_field) == 2
    elif index[normal_dir] > active_cnum[normal_dir]:
        cell[normal_dir] = active_cnum[normal_dir]
        solid = mac_get_offset_cell_type(cell, ghost_cell, cnum, cell_type_field) == 2
    return solid


# ========================================================= #
#                       Main Kernel                         #
# ========================================================= #
INCOMPRESSIBLE_P2G_BLOCK_SIZE_2D = 32
INCOMPRESSIBLE_P2G_BLOCK_SIZE_3D = 128


@ti.func
def mac_vectorize_id(index, count):
    if ti.static(GlobalVariable.DIMENSION == 2):
        i, j = vectorize_id(index, count)
        return vec2i(i, j)
    else:
        i, j, k = vectorize_id(index, count)
        return vec3i(i, j, k)


@ti.func
def mac_face_is_inside(grid_id, normal_dir: ti.template(), active_cnum, ghost_cell: int):
    face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, normal_dir)
    inside_face = True
    for axis in ti.static(range(GlobalVariable.DIMENSION)):
        if ti.static(GlobalVariable.SHAPEFUNCTION == 2):
            if axis == normal_dir:
                inside_face = inside_face and grid_id[axis] >= 0 and grid_id[axis] < face_count[axis]
            else:
                inside_face = (
                    inside_face and grid_id[axis] >= -ghost_cell and grid_id[axis] < face_count[axis] + ghost_cell
                )
        elif axis == normal_dir:
            inside_face = inside_face and grid_id[axis] >= 0 and grid_id[axis] < face_count[axis]
        else:
            inside_face = inside_face and grid_id[axis] >= -ghost_cell and grid_id[axis] < face_count[axis] + ghost_cell
    return inside_face


@ti.func
def mac_face_mass_momentum(
    normal_dir: ti.template(),
    grid_id,
    active_cnum,
    grid_size,
    igrid_size,
    p_mass,
    position,
    velocity,
    velocity_gradient,
    psize,
):
    stagger = 0.5 * (1 - ti.Vector.unit(GlobalVariable.DIMENSION, normal_dir))
    face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, normal_dir)
    grid_pos = (grid_id + stagger) * grid_size
    shape_fn = ti.Vector.zero(real, GlobalVariable.DIMENSION)
    if ti.static(GlobalVariable.SHAPEFUNCTION == 0):
        for axis in ti.static(range(GlobalVariable.DIMENSION)):
            shape_fn[axis] = ShapeLinear(position[axis], grid_pos[axis], igrid_size[axis], 0)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 1):
        for axis in ti.static(range(GlobalVariable.DIMENSION)):
            shape_fn[axis] = ShapeGIMP(position[axis], grid_pos[axis], igrid_size[axis], psize[axis])
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 2):
        for axis in ti.static(range(GlobalVariable.DIMENSION)):
            boundary_type = 0
            if axis == normal_dir:
                boundary_type = mac_boundary_type_1d(grid_id[axis], face_count[axis])
            shape_fn[axis] = ShapeBsplineQ(position[axis], grid_pos[axis], igrid_size[axis], boundary_type)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 3):
        for axis in ti.static(range(GlobalVariable.DIMENSION)):
            boundary_type = 0
            if axis == normal_dir:
                boundary_type = mac_boundary_type_1d(grid_id[axis], face_count[axis])
            shape_fn[axis] = ShapeBsplineC(position[axis], grid_pos[axis], igrid_size[axis], boundary_type)

    weight = 1.0
    for axis in ti.static(range(GlobalVariable.DIMENSION)):
        weight *= shape_fn[axis]
    nodal_mass = weight * p_mass
    nodal_momentum = nodal_mass * velocity[normal_dir]
    if ti.static(GlobalVariable.APIC or GlobalVariable.TPIC):
        dpos = position - grid_pos
        affine_row = ti.Vector(
            [velocity_gradient[normal_dir, axis] for axis in ti.static(range(GlobalVariable.DIMENSION))]
        )
        nodal_momentum -= nodal_mass * affine_row.dot(dpos)
    return nodal_mass, nodal_momentum


@ti.kernel
def kernel_mass_momentum_mac_cell_p2g(
    total_nodes: int,
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    node: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
    cell_volumefrac: ti.template(),
    cell_volume: float,
    is_2DAxisy: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.velocity[d])
        ti.block_local(node.m[d])
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            p_mass = particle[np].m
            position = particle[np].x
            velocity = particle[np].v
            velocity_gradient = particle[np].velocity_gradient
            psize = particle_lengths[bodyID]
            cell_base = ti.floor(position / grid_size - 0.5).cast(int)
            if ti.static(GlobalVariable.DIMENSION == 2):
                for i, j in ti.static(ti.ndrange(2, 2)):
                    cell = cell_base + ti.Vector([i, j])
                    if 0 <= cell[0] < active_cnum[0] and 0 <= cell[1] < active_cnum[1]:
                        center = (cell.cast(float) + 0.5) * grid_size
                        cell_weight = ti.max(0.0, 1.0 - ti.abs(position[0] - center[0]) / grid_size[0]) * ti.max(
                            0.0, 1.0 - ti.abs(position[1] - center[1]) / grid_size[1]
                        )
                        cell_measure = cell_volume
                        if ti.static(is_2DAxisy):
                            cell_measure *= (cell[0] + 0.5) * grid_size[0]
                        ti.atomic_add(
                            cell_volumefrac[linearize(cell, cnum)],
                            cell_weight * particle[np].vol / cell_measure,
                        )
            elif ti.static(GlobalVariable.DIMENSION == 3):
                for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                    cell = cell_base + ti.Vector([i, j, k])
                    if (
                        0 <= cell[0] < active_cnum[0]
                        and 0 <= cell[1] < active_cnum[1]
                        and 0 <= cell[2] < active_cnum[2]
                    ):
                        center = (cell.cast(float) + 0.5) * grid_size
                        cell_weight = (
                            ti.max(0.0, 1.0 - ti.abs(position[0] - center[0]) / grid_size[0])
                            * ti.max(0.0, 1.0 - ti.abs(position[1] - center[1]) / grid_size[1])
                            * ti.max(0.0, 1.0 - ti.abs(position[2] - center[2]) / grid_size[2])
                        )
                        ti.atomic_add(
                            cell_volumefrac[linearize(cell, cnum)],
                            cell_weight * particle[np].vol / cell_volume,
                        )
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                stagger = 0.5 * (1 - ti.Vector.unit(GlobalVariable.DIMENSION, d))
                face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
                base = ti.floor((position - psize) * igrid_size - stagger).cast(int)
                for offset in ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION))):
                    grid_id = base + offset
                    inside_face = True
                    for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                        if ti.static(GlobalVariable.SHAPEFUNCTION == 2):
                            if ti.static(d1 == d):
                                inside_face = inside_face and grid_id[d1] >= 0 and grid_id[d1] < face_count[d1]
                            else:
                                inside_face = (
                                    inside_face
                                    and grid_id[d1] >= -ghost_cell
                                    and grid_id[d1] < face_count[d1] + ghost_cell
                                )
                        elif ti.static(d1 == d):
                            inside_face = inside_face and grid_id[d1] >= 0 and grid_id[d1] < face_count[d1]
                        else:
                            inside_face = (
                                inside_face and grid_id[d1] >= -ghost_cell and grid_id[d1] < face_count[d1] + ghost_cell
                            )

                    if inside_face:
                        grid_pos = (grid_id + stagger) * grid_size
                        shape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                        if ti.static(GlobalVariable.SHAPEFUNCTION == 0):
                            for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                                shape_fn[d1] = ShapeLinear(position[d1], grid_pos[d1], igrid_size[d1], 0)
                        elif ti.static(GlobalVariable.SHAPEFUNCTION == 1):
                            for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                                shape_fn[d1] = ShapeGIMP(position[d1], grid_pos[d1], igrid_size[d1], psize[d1])
                        elif ti.static(GlobalVariable.SHAPEFUNCTION == 2):
                            for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                                btypes = 0
                                if ti.static(d1 == d):
                                    btypes = mac_boundary_type_1d(grid_id[d1], face_count[d1])
                                shape_fn[d1] = ShapeBsplineQ(position[d1], grid_pos[d1], igrid_size[d1], btypes)
                        elif ti.static(GlobalVariable.SHAPEFUNCTION == 3):
                            for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                                btypes = 0
                                if ti.static(d1 == d):
                                    btypes = mac_boundary_type_1d(grid_id[d1], face_count[d1])
                                shape_fn[d1] = ShapeBsplineC(position[d1], grid_pos[d1], igrid_size[d1], btypes)

                        weight = 1.0
                        for d0 in ti.static(range(GlobalVariable.DIMENSION)):
                            weight *= shape_fn[d0]

                        """pforce = 0.
                        if ti.static(GlobalVariable.DIMENSION == 2):
                            weight_grad = ti.Vector([dshape_fn[0] * shape_fn[1], shape_fn[0] * dshape_fn[1]])
                            if ti.static(d == 0):
                                pforce += weight_grad[0] * internal_force[0] + weight_grad[1] * internal_force[3]
                            elif ti.static(d == 1):
                                pforce += weight_grad[1] * internal_force[1] + weight_grad[0] * internal_force[3]
                        elif ti.static(GlobalVariable.DIMENSION == 3):
                            weight_grad = ti.Vector([dshape_fn[0] * shape_fn[1] * shape_fn[2],
                                                    shape_fn[0] * dshape_fn[1] * shape_fn[2],
                                                    shape_fn[0] * shape_fn[1] * dshape_fn[2]])
                            if ti.static(d == 0):
                                pforce += weight_grad[0] * internal_force[0] + weight_grad[1] * internal_force[3] + weight_grad[2] * internal_force[5]
                            elif ti.static(d == 1):
                                pforce += weight_grad[1] * internal_force[1] + weight_grad[0] * internal_force[3] + weight_grad[2] * internal_force[4]
                            elif ti.static(d == 2):
                                pforce += weight_grad[2] * internal_force[2] + weight_grad[1] * internal_force[4] + weight_grad[0] * internal_force[5]"""
                        nmass = shape_mapping(weight, p_mass)
                        momentum = shape_mapping(nmass, velocity[d])
                        if ti.static(GlobalVariable.APIC or GlobalVariable.TPIC):
                            dpos = position - grid_pos
                            momentum -= nmass * ti.Vector(
                                [velocity_gradient[d, k] for k in ti.static(range(GlobalVariable.DIMENSION))]
                            ).dot(dpos)
                        node.m[d][grid_id] += nmass
                        node.velocity[d][grid_id] += momentum


@ti.kernel
def kernel_mass_momentum_mac_cell_p2g_cell_reduced(
    total_nodes: int,
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    node: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
    cell_offsets: ti.template(),
    particle_ids: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    patch_width = ti.static(GlobalVariable.INFLUENCENODE + 1)
    patch_volume = ti.static(patch_width**GlobalVariable.DIMENSION)
    patch_entries = ti.static(GlobalVariable.DIMENSION * patch_volume)
    patch_shape = ti.Vector([patch_width for _ in ti.static(range(GlobalVariable.DIMENSION))])
    patch_shift = ti.static(GlobalVariable.INFLUENCENODE // 2)
    cell_count = 1
    for axis in ti.static(range(GlobalVariable.DIMENSION)):
        cell_count *= cnum[axis]

    for cell_id in range(cell_count):
        reduced_mass = ti.Vector.zero(real, patch_entries)
        reduced_momentum = ti.Vector.zero(real, patch_entries)
        host_cell = mac_vectorize_id(cell_id, cnum)
        patch_origin = host_cell - patch_shift
        for sorted_id in range(cell_offsets[cell_id], cell_offsets[cell_id + 1]):
            np = particle_ids[sorted_id]
            if int(particle[np].active) == 1:
                body_id = int(particle[np].bodyID)
                p_mass = particle[np].m
                position = particle[np].x
                velocity = particle[np].v
                velocity_gradient = particle[np].velocity_gradient
                psize = particle_lengths[body_id]
                for normal_dir in ti.static(range(GlobalVariable.DIMENSION)):
                    stagger = 0.5 * (1 - ti.Vector.unit(GlobalVariable.DIMENSION, normal_dir))
                    base = ti.floor((position - psize) * igrid_size - stagger).cast(int)
                    for offset in ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION))):
                        grid_id = base + offset
                        if mac_face_is_inside(grid_id, normal_dir, active_cnum, ghost_cell):
                            local_coord = grid_id - patch_origin
                            inside_patch = True
                            for axis in ti.static(range(GlobalVariable.DIMENSION)):
                                inside_patch = (
                                    inside_patch and local_coord[axis] >= 0 and local_coord[axis] < patch_width
                                )
                            if inside_patch:
                                local_id = linearize(local_coord, patch_shape)
                                local_slot = normal_dir * patch_volume + local_id
                                nodal_mass, nodal_momentum = mac_face_mass_momentum(
                                    normal_dir,
                                    grid_id,
                                    active_cnum,
                                    grid_size,
                                    igrid_size,
                                    p_mass,
                                    position,
                                    velocity,
                                    velocity_gradient,
                                    psize,
                                )
                                reduced_mass[local_slot] += nodal_mass
                                reduced_momentum[local_slot] += nodal_momentum

        for normal_dir in ti.static(range(GlobalVariable.DIMENSION)):
            for local_id in range(patch_volume):
                local_slot = normal_dir * patch_volume + local_id
                local_coord = mac_vectorize_id(local_id, patch_shape)
                grid_id = patch_origin + local_coord
                if mac_face_is_inside(grid_id, normal_dir, active_cnum, ghost_cell):
                    nodal_mass = reduced_mass[local_slot]
                    nodal_momentum = reduced_momentum[local_slot]
                    if nodal_mass != 0.0 or nodal_momentum != 0.0:
                        node.m[normal_dir][grid_id] += nodal_mass
                        node.velocity[normal_dir][grid_id] += nodal_momentum


@ti.kernel
def kernel_mass_momentum_mac_cell_p2g_shared(
    total_nodes: int,
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    node: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
    cell_offsets: ti.template(),
    particle_ids: ti.template(),
):
    block_size = ti.static(
        INCOMPRESSIBLE_P2G_BLOCK_SIZE_2D if GlobalVariable.DIMENSION == 2 else INCOMPRESSIBLE_P2G_BLOCK_SIZE_3D
    )
    patch_width = ti.static(GlobalVariable.INFLUENCENODE + 1)
    patch_volume = ti.static(patch_width**GlobalVariable.DIMENSION)
    patch_entries = ti.static(GlobalVariable.DIMENSION * patch_volume)
    patch_shape = ti.Vector([patch_width for _ in ti.static(range(GlobalVariable.DIMENSION))])
    patch_shift = ti.static(GlobalVariable.INFLUENCENODE // 2)
    active_cnum = cnum - 2 * ghost_cell
    cell_count = 1
    for axis in ti.static(range(GlobalVariable.DIMENSION)):
        cell_count *= cnum[axis]

    ti.loop_config(block_dim=block_size)
    for worker in range(cell_count * block_size):
        thread_id = worker % block_size
        cell_id = worker // block_size
        reduced_mass = ti.simt.block.SharedArray((patch_entries,), real)
        reduced_momentum = ti.simt.block.SharedArray((patch_entries,), real)

        local_slot = thread_id
        while local_slot < patch_entries:
            reduced_mass[local_slot] = 0.0
            reduced_momentum[local_slot] = 0.0
            local_slot += block_size
        ti.simt.block.sync()

        host_cell = mac_vectorize_id(cell_id, cnum)
        patch_origin = host_cell - patch_shift
        sorted_id = cell_offsets[cell_id] + thread_id
        while sorted_id < cell_offsets[cell_id + 1]:
            np = particle_ids[sorted_id]
            if int(particle[np].active) == 1:
                body_id = int(particle[np].bodyID)
                p_mass = particle[np].m
                position = particle[np].x
                velocity = particle[np].v
                velocity_gradient = particle[np].velocity_gradient
                psize = particle_lengths[body_id]
                for normal_dir in ti.static(range(GlobalVariable.DIMENSION)):
                    stagger = 0.5 * (1 - ti.Vector.unit(GlobalVariable.DIMENSION, normal_dir))
                    base = ti.floor((position - psize) * igrid_size - stagger).cast(int)
                    for offset in ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION))):
                        grid_id = base + offset
                        if mac_face_is_inside(grid_id, normal_dir, active_cnum, ghost_cell):
                            local_coord = grid_id - patch_origin
                            inside_patch = True
                            for axis in ti.static(range(GlobalVariable.DIMENSION)):
                                inside_patch = (
                                    inside_patch and local_coord[axis] >= 0 and local_coord[axis] < patch_width
                                )
                            if inside_patch:
                                local_id = linearize(local_coord, patch_shape)
                                local_slot = normal_dir * patch_volume + local_id
                                nodal_mass, nodal_momentum = mac_face_mass_momentum(
                                    normal_dir,
                                    grid_id,
                                    active_cnum,
                                    grid_size,
                                    igrid_size,
                                    p_mass,
                                    position,
                                    velocity,
                                    velocity_gradient,
                                    psize,
                                )
                                ti.atomic_add(reduced_mass[local_slot], nodal_mass)
                                ti.atomic_add(reduced_momentum[local_slot], nodal_momentum)
            sorted_id += block_size
        ti.simt.block.sync()

        for normal_dir in ti.static(range(GlobalVariable.DIMENSION)):
            local_id = thread_id
            while local_id < patch_volume:
                local_slot = normal_dir * patch_volume + local_id
                local_coord = mac_vectorize_id(local_id, patch_shape)
                grid_id = patch_origin + local_coord
                if mac_face_is_inside(grid_id, normal_dir, active_cnum, ghost_cell):
                    nodal_mass = reduced_mass[local_slot]
                    nodal_momentum = reduced_momentum[local_slot]
                    if nodal_mass != 0.0 or nodal_momentum != 0.0:
                        ti.atomic_add(node.m[normal_dir][grid_id], nodal_mass)
                        ti.atomic_add(node.velocity[normal_dir][grid_id], nodal_momentum)
                local_id += block_size


@ti.func
def interpolate_mac_particle_kinematics(position, psize, ghost_cell, cnum, grid_size, igrid_size, node: ti.template()):
    active_cnum = cnum - 2 * ghost_cell
    velocity_gradient = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
    vPIC, vFLIP = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(float, GlobalVariable.DIMENSION)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        stagger = 0.5 * (1 - ti.Vector.unit(GlobalVariable.DIMENSION, d))
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
        base = ti.floor((position - psize) * igrid_size - stagger).cast(int)
        for offset in ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION))):
            grid_id = base + offset
            inside_face = True
            for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                if ti.static(GlobalVariable.SHAPEFUNCTION == 2):
                    if ti.static(d1 == d):
                        inside_face = inside_face and grid_id[d1] >= 0 and grid_id[d1] < face_count[d1]
                    else:
                        inside_face = (
                            inside_face and grid_id[d1] >= -ghost_cell and grid_id[d1] < face_count[d1] + ghost_cell
                        )
                elif ti.static(d1 == d):
                    inside_face = inside_face and grid_id[d1] >= 0 and grid_id[d1] < face_count[d1]
                else:
                    inside_face = (
                        inside_face and grid_id[d1] >= -ghost_cell and grid_id[d1] < face_count[d1] + ghost_cell
                    )

            if inside_face:
                grid_pos = (grid_id + stagger) * grid_size
                shape_fn, dshape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION), ti.Vector.zero(
                    float, GlobalVariable.DIMENSION
                )
                if ti.static(GlobalVariable.SHAPEFUNCTION == 0):
                    for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                        shape_fn[d1] = ShapeLinear(position[d1], grid_pos[d1], igrid_size[d1], 0)
                        dshape_fn[d1] = GShapeLinear(position[d1], grid_pos[d1], igrid_size[d1], 0)
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 1):
                    for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                        shape_fn[d1] = ShapeGIMP(position[d1], grid_pos[d1], igrid_size[d1], psize[d1])
                        dshape_fn[d1] = GShapeGIMP(position[d1], grid_pos[d1], igrid_size[d1], psize[d1])
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 2):
                    for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                        btypes = 0
                        if ti.static(d1 == d):
                            btypes = mac_boundary_type_1d(grid_id[d1], face_count[d1])
                        shape_fn[d1] = ShapeBsplineQ(position[d1], grid_pos[d1], igrid_size[d1], btypes)
                        dshape_fn[d1] = GShapeBsplineQ(position[d1], grid_pos[d1], igrid_size[d1], btypes)
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 3):
                    for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                        btypes = 0
                        if ti.static(d1 == d):
                            btypes = mac_boundary_type_1d(grid_id[d1], face_count[d1])
                        shape_fn[d1] = ShapeBsplineC(position[d1], grid_pos[d1], igrid_size[d1], btypes)
                        dshape_fn[d1] = GShapeBsplineC(position[d1], grid_pos[d1], igrid_size[d1], btypes)

                weight = 1.0
                for d0 in ti.static(range(GlobalVariable.DIMENSION)):
                    weight *= shape_fn[d0]

                velocity = node.velocity[d][grid_id]
                accleration = node.force[d][grid_id]
                vPIC[d] += shape_mapping(weight, velocity)
                vFLIP[d] += shape_mapping(weight, accleration)

                if ti.static(GlobalVariable.DIMENSION == 2):
                    weight_grad = ti.Vector([dshape_fn[0] * shape_fn[1], shape_fn[0] * dshape_fn[1]])
                    for k in ti.static(range(GlobalVariable.DIMENSION)):
                        velocity_gradient[d, k] += weight_grad[k] * velocity
                elif ti.static(GlobalVariable.DIMENSION == 3):
                    weight_grad = ti.Vector(
                        [
                            dshape_fn[0] * shape_fn[1] * shape_fn[2],
                            shape_fn[0] * dshape_fn[1] * shape_fn[2],
                            shape_fn[0] * shape_fn[1] * dshape_fn[2],
                        ]
                    )
                    for k in ti.static(range(GlobalVariable.DIMENSION)):
                        velocity_gradient[d, k] += weight_grad[k] * velocity
    return vPIC, vFLIP, velocity_gradient


@ti.func
def interpolate_mac_transport_velocity(position, ghost_cell, cnum, grid_size, node: ti.template()):
    # Compatible B2/B1 MAC reconstruction: differentiating the normal B2
    # basis gives differences of the transverse B1 cell basis. Thus its
    # divergence is the interpolation of the discrete MAC divergence.
    # This is trajectory sampling only; PIC/FLIP/APIC and wall forces retain
    # their configured transfer basis. Tangential extension in the outer
    # half-cell is constant (first-order near-wall interpolation).
    active = cnum - 2 * ghost_cell
    periodic = ti.static((GlobalVariable.MPMXPBC, GlobalVariable.MPMYPBC, GlobalVariable.MPMZPBC))
    velocity = ti.Vector.zero(float, GlobalVariable.DIMENSION)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        coordinate = position / grid_size
        base = ti.Vector.zero(int, GlobalVariable.DIMENSION)
        for axis in ti.static(range(GlobalVariable.DIMENSION)):
            if ti.static(axis == d):
                base[axis] = ti.floor(coordinate[axis] - 0.5, int)
                if ti.static(not periodic[axis]):
                    base[axis] = ti.min(active[axis] - 1, ti.max(0, base[axis]))
            else:
                coordinate[axis] -= 0.5
                if ti.static(not periodic[axis]):
                    coordinate[axis] = ti.min(active[axis] - 1.0, ti.max(0.0, coordinate[axis]))
                base[axis] = ti.floor(coordinate[axis], int)
        for offset in ti.grouped(ti.ndrange(3, *((2,) * (GlobalVariable.DIMENSION - 1)))):
            index = base
            weight = 1.0
            for local_axis in ti.static(range(GlobalVariable.DIMENSION)):
                axis = ti.static((d + local_axis) % GlobalVariable.DIMENSION)
                index[axis] += offset[local_axis]
                if ti.static(axis == d):
                    linear_boundary = False
                    if ti.static(not periodic[axis]):
                        linear_boundary = coordinate[axis] < 0.5 or coordinate[axis] > active[axis] - 0.5
                    if linear_boundary:
                        weight *= ShapeLinear(coordinate[axis], index[axis], 1.0, 0)
                    else:
                        weight *= ShapeBsplineQ(coordinate[axis], index[axis], 1.0, 0)
                else:
                    weight *= ShapeLinear(coordinate[axis], index[axis], 1.0, 0)
                if ti.static(periodic[axis]):
                    index[axis] %= active[axis]
            if weight > Threshold:
                velocity[d] += weight * node.velocity[d][index]
    return velocity


@ti.kernel
def kernel_kinemaitc_mac_cell_g2p(
    total_nodes: int,
    alpha: float,
    dt: ti.template(),
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    node: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        ti.block_local(node.velocity[d])
        ti.block_local(node.force[d])
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            position = particle[np].x
            psize = particle_lengths[int(particle[np].bodyID)]
            vPIC, vFLIP, velocity_gradient = interpolate_mac_particle_kinematics(
                position, psize, ghost_cell, cnum, grid_size, igrid_size, node
            )
            # Euler trajectories expand even a rigid rotation (det(I+dt*C)>1).
            # Midpoint advection changes only the particle trajectory, not the
            # PIC/FLIP momentum update, pressure projection or material model.
            fixed = particle[np].fix_v.cast(float)
            transport_velocity = interpolate_mac_transport_velocity(position, ghost_cell, cnum, grid_size, node)
            midpoint = position + 0.5 * dt[None] * (transport_velocity * (1.0 - fixed) + particle[np].v * fixed)
            transport_velocity = interpolate_mac_transport_velocity(midpoint, ghost_cell, cnum, grid_size, node)
            particle[np]._update_incompressible_particle_state(dt, alpha, vPIC, vFLIP, transport_velocity)
            particle[np].velocity_gradient = velocity_gradient


@ti.kernel
def kernel_compute_mac_grid_velocity_gravity(
    cutoff: float, gravity: ti.types.vector(GlobalVariable.DIMENSION, float), dt: ti.template(), node: ti.template()
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        for I in ti.grouped(node.m[d]):
            if node.m[d][I] > cutoff:
                mass = node.m[d][I]
                force = node.force[d][I]
                node.velocity[d][I] /= mass
                node.force[d][I] = node.velocity[d][I]
                acceleration = force / mass + gravity[d]
                node.velocity[d][I] += acceleration * dt[None]


@ti.kernel
def kernel_compute_mac_grid_velocity(cutoff: float, node: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        for I in ti.grouped(node.m[d]):
            if node.m[d][I] > cutoff:
                node.velocity[d][I] /= node.m[d][I]
                node.force[d][I] = node.velocity[d][I]


@ti.kernel
def kernel_add_mac_grid_gravity(
    gravity: ti.types.vector(GlobalVariable.DIMENSION, float), dt: ti.template(), node: ti.template()
):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        for I in ti.grouped(node.velocity[d]):
            if gravity[d] != 0:
                node.velocity[d][I] += gravity[d] * dt[None]


@ti.kernel
def kernel_compute_mac_grid_acceleration(cutoff: float, node: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        for I in ti.grouped(node.m[d]):
            if node.m[d][I] > cutoff:
                node.force[d][I] = node.velocity[d][I] - node.force[d][I]


@ti.kernel
def extrapolate_mac_boundary_tangent_velocity(
    ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), node: ti.template()
):
    active_cnum = cnum - 2 * ghost_cell
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
        for I in ti.grouped(node.velocity[d]):
            source = I
            has_ghost = False
            sign = 1.0
            domain_face = I[d] == 0 or I[d] == face_count[d] - 1
            if I[d] < 0:
                source[d] = -I[d]
                has_ghost = True
                sign = -1.0
            elif I[d] >= face_count[d]:
                source[d] = 2 * (face_count[d] - 1) - I[d]
                has_ghost = True
                sign = -1.0
            for k in ti.static(range(GlobalVariable.DIMENSION)):
                if ti.static(k != d):
                    if I[k] < 0:
                        source[k] = -I[k] - 1
                        has_ghost = True
                    elif I[k] >= face_count[k]:
                        source[k] = 2 * face_count[k] - 1 - I[k]
                        has_ghost = True

            if domain_face:
                node.velocity[d][I] = 0.0
                node.force[d][I] = 0.0
            elif has_ghost:
                inside_source = True
                for k in ti.static(range(GlobalVariable.DIMENSION)):
                    inside_source = inside_source and source[k] >= 0 and source[k] < face_count[k]
                if inside_source:
                    node.velocity[d][I] = sign * node.velocity[d][source]
                    node.force[d][I] = sign * node.force[d][source]
                else:
                    node.velocity[d][I] = 0.0
                    node.force[d][I] = 0.0


@ti.kernel
def enforce_boundary_and_extrapolate_mac_tangent_velocity(
    ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), cell_type: ti.template(), node: ti.template()
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(cell_type):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
        if inside_active and is_solid(cell_type[I]):
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                node.velocity[d][I] = 0.0
                node.velocity[d][I + offset] = 0.0
                node.force[d][I] = 0.0
                node.force[d][I + offset] = 0.0

    for d in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
        for I in ti.grouped(node.velocity[d]):
            source = I
            has_ghost = False
            sign = 1.0
            domain_face = I[d] == 0 or I[d] == face_count[d] - 1
            solid_domain_face = False
            if domain_face:
                solid_domain_face = mac_domain_boundary_is_solid(I, d, active_cnum, ghost_cell, cnum, cell_type)
            if I[d] < 0:
                source[d] = -I[d]
                has_ghost = True
                if mac_normal_ghost_is_solid(I, d, active_cnum, ghost_cell, cnum, cell_type):
                    sign = -1.0
            elif I[d] >= face_count[d]:
                source[d] = 2 * (face_count[d] - 1) - I[d]
                has_ghost = True
                if mac_normal_ghost_is_solid(I, d, active_cnum, ghost_cell, cnum, cell_type):
                    sign = -1.0
            for k in ti.static(range(GlobalVariable.DIMENSION)):
                if ti.static(k != d):
                    if I[k] < 0:
                        source[k] = -I[k] - 1
                        has_ghost = True
                    elif I[k] >= face_count[k]:
                        source[k] = 2 * face_count[k] - 1 - I[k]
                        has_ghost = True

            if solid_domain_face:
                node.velocity[d][I] = 0.0
                node.force[d][I] = 0.0
            elif has_ghost:
                inside_source = True
                for k in ti.static(range(GlobalVariable.DIMENSION)):
                    inside_source = inside_source and source[k] >= 0 and source[k] < face_count[k]
                if inside_source:
                    node.velocity[d][I] = sign * node.velocity[d][source]
                    node.force[d][I] = sign * node.force[d][source]
                else:
                    node.velocity[d][I] = 0.0
                    node.force[d][I] = 0.0


@ti.kernel
def enforce_boundary(
    ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), cell_type: ti.template(), node: ti.template()
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(cell_type):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
        if inside_active and is_solid(cell_type[I]):
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                node.velocity[d][I] = 0.0
                node.velocity[d][I + offset] = 0.0
                node.force[d][I] = 0.0
                node.force[d][I + offset] = 0.0

    for d in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
        for I in ti.grouped(node.velocity[d]):
            source = I
            has_ghost = False
            sign = 1.0
            domain_face = I[d] == 0 or I[d] == face_count[d] - 1
            solid_domain_face = False
            if domain_face:
                solid_domain_face = mac_domain_boundary_is_solid(I, d, active_cnum, ghost_cell, cnum, cell_type)
            if I[d] < 0:
                source[d] = -I[d]
                has_ghost = True
                if mac_normal_ghost_is_solid(I, d, active_cnum, ghost_cell, cnum, cell_type):
                    sign = -1.0
            elif I[d] >= face_count[d]:
                source[d] = 2 * (face_count[d] - 1) - I[d]
                has_ghost = True
                if mac_normal_ghost_is_solid(I, d, active_cnum, ghost_cell, cnum, cell_type):
                    sign = -1.0
            for k in ti.static(range(GlobalVariable.DIMENSION)):
                if ti.static(k != d):
                    if I[k] < 0:
                        source[k] = -I[k] - 1
                        has_ghost = True
                    elif I[k] >= face_count[k]:
                        source[k] = 2 * face_count[k] - 1 - I[k]
                        has_ghost = True

            if solid_domain_face:
                node.velocity[d][I] = 0.0
                node.force[d][I] = 0.0
            elif has_ghost:
                inside_source = True
                for k in ti.static(range(GlobalVariable.DIMENSION)):
                    inside_source = inside_source and source[k] >= 0 and source[k] < face_count[k]
                if inside_source:
                    node.velocity[d][I] = sign * node.velocity[d][source]
                    node.force[d][I] = sign * node.force[d][source]
                else:
                    node.velocity[d][I] = 0.0
                    node.force[d][I] = 0.0


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
def mac_cell_center_velocity(cell, node: ti.template()):
    velocity = ti.Vector.zero(float, GlobalVariable.DIMENSION)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
        velocity[d] = 0.5 * (node.velocity[d][cell] + node.velocity[d][cell + offset])
    return velocity


@ti.func
def ibm_mixed_density(phi_s: float, rho_f: float, rho_s: float):
    solid_density = rho_s
    if solid_density <= Threshold:
        solid_density = rho_f
    return (1.0 - phi_s) * rho_f + phi_s * solid_density


@ti.func
def ibm_cell_is_active(cell, active_cnum):
    inside = True
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        inside = inside and cell[d] >= 0 and cell[d] < active_cnum[d]
    return inside


@ti.func
def coupled_mac_face_material_3d(
    face,
    direction,
    active_cnum,
    rho_f,
    mode: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
):
    fluid_fraction = 0.0
    density = 0.0
    count = 0
    unit = ti.Vector.unit(3, direction)
    for side in ti.static((0, 1)):
        cell = face - side * unit
        inside = True
        for d in ti.static(range(3)):
            inside = inside and cell[d] >= 0 and cell[d] < active_cnum[d]
        if inside:
            phi_s = ti.min(1.0, ti.max(0.0, solid_fraction[cell]))
            local_fraction = 1.0
            local_density = rho_f
            if ti.static(mode == 1):
                local_fraction = ti.max(0.05, 1.0 - phi_s)
            elif ti.static(mode == 2):
                rho_s = solid_density[cell]
                if rho_s <= Threshold:
                    rho_s = rho_f
                local_density = (1.0 - phi_s) * rho_f + phi_s * rho_s
            fluid_fraction += local_fraction
            density += local_density
            count += 1
    return fluid_fraction / ti.max(count, 1), density / ti.max(count, 1)


@ti.kernel
def kernel_compute_mac_viscous_delta(
    cutoff: float,
    ghost_cell: int,
    cnum: ti.template(),
    grid_size: ti.template(),
    dt: ti.template(),
    matProps: ti.template(),
    cell_type: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
    mode: ti.template(),
    delta0: ti.template(),
    delta1: ti.template(),
    delta2: ti.template(),
    node: ti.template(),
    no_slip: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for direction in ti.static(range(GlobalVariable.DIMENSION)):
        face_cnum = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, direction)
        for I in ti.grouped(node.velocity[direction]):
            inside = True
            for axis in ti.static(range(GlobalVariable.DIMENSION)):
                inside = inside and 0 <= I[axis] < face_cnum[axis]
            delta = 0.0
            if inside:
                left_type = int(cell_type[I - ti.Vector.unit(GlobalVariable.DIMENSION, direction)])
                right_type = int(cell_type[I])
                if node.m[direction][I] > cutoff and (left_type == 1 or right_type == 1):
                    center_fraction, center_density = 1.0, matProps.density
                    if ti.static(mode > 0):
                        center_fraction, center_density = coupled_mac_face_material_3d(
                            I, direction, active_cnum, matProps.density, mode, solid_fraction, solid_density
                        )
                    diffusion = 0.0
                    center_velocity = node.velocity[direction][I]
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        unit = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                        left, right = I - unit, I + unit
                        # Legacy slip walls retain the zero-normal-gradient stencil.
                        # No-slip uses the reflected ghost velocity at the real wall.
                        if ti.static(not no_slip):
                            for axis in ti.static(range(GlobalVariable.DIMENSION)):
                                left[axis] = ti.max(0, left[axis])
                                right[axis] = ti.min(face_cnum[axis] - 1, right[axis])
                        left_fraction, right_fraction = 1.0, 1.0
                        if ti.static(mode > 0):
                            left_fraction, _ = coupled_mac_face_material_3d(
                                left, direction, active_cnum, matProps.density, mode, solid_fraction, solid_density
                            )
                            right_fraction, _ = coupled_mac_face_material_3d(
                                right, direction, active_cnum, matProps.density, mode, solid_fraction, solid_density
                            )
                        diffusion += (
                            0.5 * (center_fraction + left_fraction) * (node.velocity[direction][left] - center_velocity)
                            + 0.5
                            * (center_fraction + right_fraction)
                            * (node.velocity[direction][right] - center_velocity)
                        ) / (grid_size[d] * grid_size[d])
                    delta = (
                        dt[None] * matProps.viscosity * diffusion / ti.max(center_fraction * center_density, Threshold)
                    )
            if ti.static(direction == 0):
                delta0[I] = delta
            elif ti.static(direction == 1):
                delta1[I] = delta
            else:
                delta2[I] = delta


@ti.kernel
def kernel_apply_mac_viscous_delta(
    delta0: ti.template(), delta1: ti.template(), delta2: ti.template(), node: ti.template()
):
    for direction in ti.static(range(GlobalVariable.DIMENSION)):
        for I in ti.grouped(node.velocity[direction]):
            if ti.static(direction == 0):
                node.velocity[direction][I] += delta0[I]
            elif ti.static(direction == 1):
                node.velocity[direction][I] += delta1[I]
            else:
                node.velocity[direction][I] += delta2[I]


@ti.kernel
def kernel_apply_incompressible_ibm_mac_source(
    ghost_cell: int,
    cnum: ti.template(),
    dt: ti.template(),
    matProps: ti.template(),
    cell_type: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
    solid_velocity_cell: ti.template(),
    ibm_force_cell: ti.template(),
    node: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    rho_f = matProps.density
    ibm_force_cell.fill(0.0)

    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            phi_s = ti.min(1.0, ti.max(0.0, solid_fraction[I]))
            if phi_s > Threshold and int(cell_type[I]) == 1:
                rho = ibm_mixed_density(phi_s, rho_f, solid_density[I])
                rho_s = solid_density[I]
                if rho_s <= Threshold:
                    rho_s = rho_f
                varphi_s = ti.min(1.0, ti.max(0.0, phi_s * rho_s / ti.max(rho, Threshold)))
                fluid_velocity = mac_cell_center_velocity(I, node)
                ibm_force_cell[I] = (
                    rho * varphi_s * (solid_velocity_cell[I] - fluid_velocity) / ti.max(dt[None], Threshold)
                )
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            phi_s = ti.min(1.0, ti.max(0.0, solid_fraction[I]))
            if phi_s > Threshold and int(cell_type[I]) == 1:
                rho = ibm_mixed_density(phi_s, rho_f, solid_density[I])
                rho_s = solid_density[I]
                if rho_s <= Threshold:
                    rho_s = rho_f
                varphi_s = ti.min(1.0, ti.max(0.0, phi_s * rho_s / ti.max(rho, Threshold)))
                fluid_velocity = mac_cell_center_velocity(I, node)
                ibm_force_cell[I] = (
                    rho * varphi_s * (solid_velocity_cell[I] - fluid_velocity) / ti.max(dt[None], Threshold)
                )

    for k in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, k)
        if ti.static(GlobalVariable.DIMENSION == 2):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]))):
                delta_velocity = 0.0
                weight = 0.0
                for side in ti.static((0, 1)):
                    cell = I - side * ti.Vector.unit(GlobalVariable.DIMENSION, k)
                    if ibm_cell_is_active(cell, active_cnum):
                        phi_s = ti.min(1.0, ti.max(0.0, solid_fraction[cell]))
                        if phi_s > Threshold and int(cell_type[cell]) == 1:
                            rho = ibm_mixed_density(phi_s, rho_f, solid_density[cell])
                            delta_velocity += dt[None] * ibm_force_cell[cell][k] / ti.max(rho, Threshold)
                            weight += 1
                if weight > 0:
                    node.velocity[k][I] += delta_velocity / weight
        elif ti.static(GlobalVariable.DIMENSION == 3):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]), (0, face_count[2]))):
                delta_velocity = 0.0
                weight = 0.0
                for side in ti.static((0, 1)):
                    cell = I - side * ti.Vector.unit(GlobalVariable.DIMENSION, k)
                    if ibm_cell_is_active(cell, active_cnum):
                        phi_s = ti.min(1.0, ti.max(0.0, solid_fraction[cell]))
                        if phi_s > Threshold and int(cell_type[cell]) == 1:
                            rho = ibm_mixed_density(phi_s, rho_f, solid_density[cell])
                            delta_velocity += dt[None] * ibm_force_cell[cell][k] / ti.max(rho, Threshold)
                            weight += 1
                if weight > 0:
                    node.velocity[k][I] += delta_velocity / weight


@ti.func
def density_projection_cell_is_active(
    I,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    interior_only: ti.template(),
):
    active = get_offset_cell_type(I, ghost_cell, cnum, cell_type) == 1
    if ti.static(interior_only):
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            for s in ti.static((-1, 1)):
                active = (
                    active
                    and get_offset_cell_type(
                        I + ti.Vector.unit(GlobalVariable.DIMENSION, d) * s, ghost_cell, cnum, cell_type
                    )
                    == 1
                )
    return active


@ti.func
def cell_has_axis_neighbor_type(
    I, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), cell_type: ti.template(), target_type: int
):
    found = False
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        for s in ti.static((-1, 1)):
            found = (
                found
                or get_offset_cell_type(
                    I + ti.Vector.unit(GlobalVariable.DIMENSION, d) * s, ghost_cell, cnum, cell_type
                )
                == target_type
            )
    return found


@ti.func
def cell_has_neighbor_type(
    I, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), cell_type: ti.template(), target_type: int
):
    found = False
    if ti.static(GlobalVariable.DIMENSION == 2):
        for i in ti.static(range(-1, 2)):
            for j in ti.static(range(-1, 2)):
                if ti.static(i != 0 or j != 0):
                    found = (
                        found or get_offset_cell_type(I + ti.Vector([i, j]), ghost_cell, cnum, cell_type) == target_type
                    )
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for i in ti.static(range(-1, 2)):
            for j in ti.static(range(-1, 2)):
                for k in ti.static(range(-1, 2)):
                    if ti.static(i != 0 or j != 0 or k != 0):
                        found = (
                            found
                            or get_offset_cell_type(I + ti.Vector([i, j, k]), ghost_cell, cnum, cell_type)
                            == target_type
                        )
    return found


@ti.func
def particle_cell_in_closed_domain(position, active_cnum, igrid_size):
    scaled = position * igrid_size
    cell = ti.floor(scaled).cast(int)
    inside = particle_position_is_valid(position)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        inside = inside and scaled[d] >= -1.0e-6 and scaled[d] <= active_cnum[d] + 1.0e-6
        cell[d] = ti.min(active_cnum[d] - 1, ti.max(0, cell[d]))
    return cell, inside


@ti.func
def particle_near_fluid_interface(
    position,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    cell, inside_active = particle_cell_in_closed_domain(position, active_cnum, 1.0 / grid_size)
    near_interface = False
    if inside_active and get_offset_cell_type(cell, ghost_cell, cnum, cell_type) == 1:
        near_interface = cell_has_axis_neighbor_type(cell, ghost_cell, cnum, cell_type, 0)
    return near_interface


@ti.func
def particle_position_is_valid(position):
    valid = True
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        valid = valid and position[d] == position[d] and ti.abs(position[d]) < 1.0e20
    return valid


@ti.func
def particle_cell_is_active_fluid(
    position,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
    interior_only: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    cell, inside = particle_cell_in_closed_domain(position, active_cnum, 1.0 / grid_size)
    active = False
    if inside:
        active = density_projection_cell_is_active(cell, ghost_cell, cnum, cell_type, interior_only)
    return active


@ti.func
def particle_shifting_is_active_fluid(position, ghost_cell, cnum, grid_size, cell_type: ti.template()):
    # Shifting must not cross a free surface; a solid wall is not a free surface.
    # Keep the density-projection pressure space unchanged (its wall treatment differs).
    active = particle_cell_is_active_fluid(position, ghost_cell, cnum, grid_size, cell_type, False)
    if active:
        active = not particle_near_fluid_interface(position, ghost_cell, cnum, grid_size, cell_type)
    return active


@ti.func
def is_inside_field(index, field: ti.template()):
    inside = True
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        inside = inside and index[d] >= 0 and index[d] < field.shape[d]
    return inside


@ti.func
def get_mg_cell_type(index, grid_type: ti.template()):
    cell_type = 2
    if is_inside_field(index, grid_type):
        cell_type = int(grid_type[index])
    return cell_type


@ti.func
def get_offset_cell_type(
    index, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), cell_type_field: ti.template()
):
    cell_type = 2
    inside = True
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        inside = inside and index[d] >= -ghost_cell and index[d] < cnum[d] - ghost_cell
    if inside:
        cell_type = int(cell_type_field[index])
    return cell_type


@ti.func
def sample_fluid_sdf(
    index, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), fluid_sdf: ti.template()
):
    return fluid_sdf[clamp_offset_cell_index(index, ghost_cell, cnum)]


@ti.func
def sample_solid_sdf(
    index, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), solid_sdf: ti.template()
):
    return solid_sdf[clamp_offset_cell_index(index, ghost_cell, cnum)]


@ti.func
def interpolate_solid_sdf(
    position,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    solid_sdf: ti.template(),
):
    coord = position / grid_size - 0.5
    base = ti.floor(coord).cast(int)
    frac = coord - base.cast(float)
    phi = 0.0
    if ti.static(GlobalVariable.DIMENSION == 2):
        for i, j in ti.static(ti.ndrange(2, 2)):
            weight = (frac[0] if ti.static(i == 1) else 1.0 - frac[0]) * (
                frac[1] if ti.static(j == 1) else 1.0 - frac[1]
            )
            phi += weight * sample_solid_sdf(base + ti.Vector([i, j]), ghost_cell, cnum, solid_sdf)
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            weight = (
                (frac[0] if ti.static(i == 1) else 1.0 - frac[0])
                * (frac[1] if ti.static(j == 1) else 1.0 - frac[1])
                * (frac[2] if ti.static(k == 1) else 1.0 - frac[2])
            )
            phi += weight * sample_solid_sdf(base + ti.Vector([i, j, k]), ghost_cell, cnum, solid_sdf)
    return phi


@ti.func
def solid_sdf_position_normal(
    position,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    solid_sdf: ti.template(),
):
    cell = ti.floor(position / grid_size - 0.5).cast(int)
    normal = ti.Vector.zero(float, GlobalVariable.DIMENSION)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        direction = ti.Vector.unit(GlobalVariable.DIMENSION, d)
        normal[d] = (
            0.5
            * (
                sample_solid_sdf(cell + direction, ghost_cell, cnum, solid_sdf)
                - sample_solid_sdf(cell - direction, ghost_cell, cnum, solid_sdf)
            )
            / grid_size[d]
        )
    norm = normal.norm()
    if norm > Threshold:
        normal /= norm
    return normal


@ti.func
def fluid_sdf_normal(
    index,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    fluid_sdf: ti.template(),
):
    grad = ti.Vector.zero(float, GlobalVariable.DIMENSION)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        direction = ti.Vector.unit(GlobalVariable.DIMENSION, d)
        grad[d] = (
            0.5
            * (
                sample_fluid_sdf(index + direction, ghost_cell, cnum, fluid_sdf)
                - sample_fluid_sdf(index - direction, ghost_cell, cnum, fluid_sdf)
            )
            / grid_size[d]
        )

    normal = ti.Vector.zero(float, GlobalVariable.DIMENSION)
    norm = grad.norm()
    if norm > Threshold:
        normal = grad / norm
    return normal


@ti.func
def get_node_id(index, gnum):
    return (linearize(index, gnum), 0)


@ti.kernel
def init_boundary(
    ghost_cell: int,
    cnum: ti.template(),
    cell_type: ti.template(),
):
    # ``EngineKernel`` can be imported before ``geotaichi.init(dim=...)``.
    # Let Taichi specialize this argument from the actual scene so a 2D
    # integer cell-count vector is not frozen as the default 3D float type.
    for I in ti.grouped(cell_type):
        if any(I < 0) or any(I >= cnum - 2 * ghost_cell):
            cell_type[I] = 0


@ti.kernel
def mark_solid_cell_region(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    start_point: ti.types.vector(GlobalVariable.DIMENSION, float),
    end_point: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
):
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(cell_type):
            inside = True
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                cell_lower = I[d] * grid_size[d]
                cell_upper = (I[d] + 1) * grid_size[d]
                inside = inside and cell_lower < end_point[d] - 1e-12 and cell_upper > start_point[d] + 1e-12
            if inside:
                cell_type[I] = 2
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(cell_type):
            inside = True
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                cell_lower = I[d] * grid_size[d]
                cell_upper = (I[d] + 1) * grid_size[d]
                inside = inside and cell_lower < end_point[d] - 1e-12 and cell_upper > start_point[d] + 1e-12
            if inside:
                cell_type[I] = 2


@ti.kernel
def kernel_close_solid_cell_boundary_corners(
    ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), cell_type: ti.template()
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(cell_type):
        outside = False
        supported_by_walls = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            outside_d = I[d] < 0 or I[d] >= active_cnum[d]
            outside = outside or outside_d
            if outside_d:
                face_cell = I
                for k in ti.static(range(GlobalVariable.DIMENSION)):
                    if ti.static(k != d):
                        if face_cell[k] < 0:
                            face_cell[k] = 0
                        elif face_cell[k] >= active_cnum[k]:
                            face_cell[k] = active_cnum[k] - 1
                supported_by_walls = (
                    supported_by_walls and get_offset_cell_type(face_cell, ghost_cell, cnum, cell_type) == 2
                )

        if outside and supported_by_walls and is_air(cell_type[I]):
            cell_type[I] = 2


@ti.func
def box_signed_distance(position, lower, upper):
    q = ti.Vector.zero(float, GlobalVariable.DIMENSION)
    max_q = -1e30
    outside_norm2 = 0.0
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        q[d] = ti.max(lower[d] - position[d], position[d] - upper[d])
        max_q = ti.max(max_q, q[d])
        outside = ti.max(q[d], 0.0)
        outside_norm2 += outside * outside
    return ti.sqrt(outside_norm2) + ti.min(max_q, 0.0)


@ti.func
def solid_face_open_fraction(
    left,
    right,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    min_fraction: float,
    cell_type: ti.template(),
    solid_sdf: ti.template(),
):
    fraction = 0.0
    left_type = get_offset_cell_type(left, ghost_cell, cnum, cell_type)
    right_type = get_offset_cell_type(right, ghost_cell, cnum, cell_type)
    if left_type != 2 and right_type != 2:
        phi_l = solid_sdf[left]
        phi_r = solid_sdf[right]
        if phi_l > 0.0 and phi_r > 0.0:
            fraction = 1.0
        elif phi_l > 0.0 or phi_r > 0.0:
            positive_phi = ti.max(phi_l, phi_r)
            denom = ti.abs(phi_l) + ti.abs(phi_r)
            if denom > Threshold:
                fraction = ti.max(min_fraction, ti.min(1.0, positive_phi / denom))
    return fraction


@ti.func
def select_solid_face_fraction(
    direction: ti.template(),
    face,
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
):
    fraction = 1.0
    if ti.static(direction == 0):
        fraction = face_fraction0[face]
    elif ti.static(direction == 1):
        fraction = face_fraction1[face]
    elif ti.static(GlobalVariable.DIMENSION == 3 and direction == 2):
        fraction = face_fraction2[face]
    return fraction


@ti.func
def select_solid_face_velocity(
    direction: ti.template(),
    face,
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
):
    velocity = 0.0
    if ti.static(direction == 0):
        velocity = solid_velocity0[face]
    elif ti.static(direction == 1):
        velocity = solid_velocity1[face]
    elif ti.static(GlobalVariable.DIMENSION == 3 and direction == 2):
        velocity = solid_velocity2[face]
    return velocity


@ti.kernel
def enforce_boundary_cut_cell(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    node: ti.template(),
    no_slip: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(cell_type):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
        if inside_active and is_solid(cell_type[I]):
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                node.velocity[d][I] = select_solid_face_velocity(
                    d, I, solid_velocity0, solid_velocity1, solid_velocity2
                )
                node.velocity[d][I + offset] = select_solid_face_velocity(
                    d, I + offset, solid_velocity0, solid_velocity1, solid_velocity2
                )
                node.force[d][I] = 0.0
                node.force[d][I + offset] = 0.0

    for d in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
        for I in ti.grouped(node.velocity[d]):
            source = I
            has_ghost = False
            sign = 1.0
            wall_shift = 0.0
            domain_face = I[d] == 0 or I[d] == face_count[d] - 1
            solid_domain_face = False
            if domain_face:
                solid_domain_face = mac_domain_boundary_is_solid(I, d, active_cnum, ghost_cell, cnum, cell_type)
            if I[d] < 0:
                source[d] = -I[d]
                has_ghost = True
                if mac_normal_ghost_is_solid(I, d, active_cnum, ghost_cell, cnum, cell_type):
                    sign = -1.0
            elif I[d] >= face_count[d]:
                source[d] = 2 * (face_count[d] - 1) - I[d]
                has_ghost = True
                if mac_normal_ghost_is_solid(I, d, active_cnum, ghost_cell, cnum, cell_type):
                    sign = -1.0
            for k in ti.static(range(GlobalVariable.DIMENSION)):
                if ti.static(k != d):
                    if I[k] < 0:
                        source[k] = -I[k] - 1
                        has_ghost = True
                    elif I[k] >= face_count[k]:
                        source[k] = 2 * face_count[k] - 1 - I[k]
                        has_ghost = True
                    if ti.static(no_slip):
                        if I[k] < 0 or I[k] >= face_count[k]:
                            wall_cell = I
                            for axis in ti.static(range(GlobalVariable.DIMENSION)):
                                wall_cell[axis] = ti.min(active_cnum[axis] - 1, ti.max(0, wall_cell[axis]))
                            wall_cell[k] = -1 if I[k] < 0 else active_cnum[k]
                            if get_offset_cell_type(wall_cell, ghost_cell, cnum, cell_type) == 2:
                                sign = -sign
                                wall_shift = (
                                    2.0
                                    * select_solid_face_velocity(
                                        d, I, solid_velocity0, solid_velocity1, solid_velocity2
                                    )
                                    - wall_shift
                                )

            if solid_domain_face:
                node.velocity[d][I] = select_solid_face_velocity(
                    d, I, solid_velocity0, solid_velocity1, solid_velocity2
                )
                node.force[d][I] = 0.0
            elif has_ghost:
                inside_source = True
                for k in ti.static(range(GlobalVariable.DIMENSION)):
                    inside_source = inside_source and source[k] >= 0 and source[k] < face_count[k]
                if inside_source:
                    node.velocity[d][I] = sign * node.velocity[d][source] + wall_shift
                    node.force[d][I] = sign * node.force[d][source]
                else:
                    node.velocity[d][I] = 0.0
                    node.force[d][I] = 0.0


@ti.kernel
def kernel_reset_solid_sdf(grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), solid_sdf: ti.template()):
    far_distance = 1e6 * min_grid_spacing(grid_size)
    for I in ti.grouped(solid_sdf):
        solid_sdf[I] = far_distance


@ti.kernel
def kernel_update_solid_sdf_from_box(
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    start_point: ti.types.vector(GlobalVariable.DIMENSION, float),
    end_point: ti.types.vector(GlobalVariable.DIMENSION, float),
    solid_sdf: ti.template(),
):
    for I in ti.grouped(solid_sdf):
        center = (I.cast(float) + 0.5) * grid_size
        phi = box_signed_distance(center, start_point, end_point)
        ti.atomic_min(solid_sdf[I], phi)


@ti.kernel
def kernel_finalize_solid_sdf_from_cell_type(
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), cell_type: ti.template(), solid_sdf: ti.template()
):
    interface_band = 0.5 * min_grid_spacing(grid_size)
    for I in ti.grouped(solid_sdf):
        if is_solid(cell_type[I]):
            solid_sdf[I] = ti.min(solid_sdf[I], -interface_band)


@ti.kernel
def kernel_build_solid_face_open_fraction_direction(
    direction: ti.template(),
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    min_fraction: float,
    cell_type: ti.template(),
    solid_sdf: ti.template(),
    face_fraction: ti.template(),
):
    offset = ti.Vector.unit(GlobalVariable.DIMENSION, direction)
    for I in ti.grouped(face_fraction):
        face_fraction[I] = solid_face_open_fraction(I - offset, I, ghost_cell, cnum, min_fraction, cell_type, solid_sdf)


@ti.kernel
def kernel_find_fluid_domain(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
    particle: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(cell_type):
        if not is_solid(cell_type[I]):
            cell_type[I] = 0
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            idx, inside_active = particle_cell_in_closed_domain(particle[np].x, active_cnum, igrid_size)
            if inside_active and not is_solid(cell_type[idx]):
                cell_type[idx] = 1


@ti.kernel
def kernel_find_fluid_domain_by_volume(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    min_volume_fraction: float,
    cell_volumefrac: ti.template(),
    cell_type: ti.template(),
    particle: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(cell_type):
        if not is_solid(cell_type[I]):
            cell_type[I] = 0

    for I in ti.grouped(cell_type):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
        if inside_active and not is_solid(cell_type[I]):
            cell_id = linearize(I, cnum)
            if cell_volumefrac[cell_id] > min_volume_fraction:
                cell_type[I] = 1

    # Volume support determines the free-surface shape, but a threshold must
    # never turn a cell that actually contains fluid particles into AIR.
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            idx, inside_active = particle_cell_in_closed_domain(particle[np].x, active_cnum, igrid_size)
            if inside_active and not is_solid(cell_type[idx]):
                cell_type[idx] = 1


@ti.kernel
def kernel_fill_enclosed_fluid_cells(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
):
    # ponytail: this repairs isolated one-cell holes; use connectivity/flood-fill if multi-cell voids appear.
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(cell_type):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
        if inside_active and is_air(cell_type[I]):
            enclosed = True
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                unit = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                enclosed = enclosed and is_fluid(cell_type[I - unit]) and is_fluid(cell_type[I + unit])
            if enclosed:
                cell_type[I] = 3

    # Mark first so parallel scheduling cannot chain-fill adjacent air cells.
    for I in ti.grouped(cell_type):
        if int(cell_type[I]) == 3:
            cell_type[I] = 1


@ti.kernel
def kernel_build_fluid_sdf_from_particles(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    particle: ti.template(),
    fluid_sdf: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    min_dx = min_grid_spacing(grid_size)
    max_distance = 3.0 * min_dx
    for I in ti.grouped(fluid_sdf):
        fluid_sdf[I] = max_distance

    if ti.static(GlobalVariable.DIMENSION == 2):
        for np in range(particleNum):
            if int(particle[np].active) == 1:
                position = particle[np].x
                radius = equivalent_particle_radius(particle[np].vol)
                band = max_distance + radius
                lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                lo = ti.max(lo, -ghost_cell)
                hi = ti.min(hi, active_cnum + ghost_cell)
                for i, j in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1])):
                    cell = ti.Vector([i, j])
                    center = (cell.cast(float) + 0.5) * grid_size
                    phi = (center - position).norm() - radius
                    ti.atomic_min(fluid_sdf[cell], phi)
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for np in range(particleNum):
            if int(particle[np].active) == 1:
                position = particle[np].x
                radius = equivalent_particle_radius(particle[np].vol)
                band = max_distance + radius
                lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                lo = ti.max(lo, -ghost_cell)
                hi = ti.min(hi, active_cnum + ghost_cell)
                for i, j, k in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])):
                    cell = ti.Vector([i, j, k])
                    center = (cell.cast(float) + 0.5) * grid_size
                    phi = (center - position).norm() - radius
                    ti.atomic_min(fluid_sdf[cell], phi)


@ti.kernel
def kernel_find_fluid_sdf_bounds(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    particle: ti.template(),
    bounds: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    min_dx = min_grid_spacing(grid_size)
    max_distance = 3.0 * min_dx
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        bounds[0, d] = active_cnum[d] + ghost_cell
        bounds[1, d] = -ghost_cell

    for np in range(particleNum):
        if int(particle[np].active) == 1:
            position = particle[np].x
            radius = equivalent_particle_radius(particle[np].vol)
            band = max_distance + radius
            lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
            hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
            lo = ti.max(lo, -ghost_cell)
            hi = ti.min(hi, active_cnum + ghost_cell)
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                ti.atomic_min(bounds[0, d], lo[d])
                ti.atomic_max(bounds[1, d], hi[d])


@ti.kernel
def kernel_build_fluid_sdf_from_particles_bounded(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    particle: ti.template(),
    bounds: ti.template(),
    fluid_sdf: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    min_dx = min_grid_spacing(grid_size)
    max_distance = 3.0 * min_dx
    lower = ti.Vector.zero(int, GlobalVariable.DIMENSION)
    upper = ti.Vector.zero(int, GlobalVariable.DIMENSION)
    valid = True
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        lower[d] = bounds[0, d]
        upper[d] = bounds[1, d]
        valid = valid and lower[d] < upper[d]

    if valid:
        if ti.static(GlobalVariable.DIMENSION == 2):
            for i, j in ti.ndrange((lower[0], upper[0]), (lower[1], upper[1])):
                fluid_sdf[ti.Vector([i, j])] = max_distance
        elif ti.static(GlobalVariable.DIMENSION == 3):
            for i, j, k in ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2])):
                fluid_sdf[ti.Vector([i, j, k])] = max_distance

        if ti.static(GlobalVariable.DIMENSION == 2):
            for np in range(particleNum):
                if int(particle[np].active) == 1:
                    position = particle[np].x
                    radius = equivalent_particle_radius(particle[np].vol)
                    band = max_distance + radius
                    lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                    hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                    lo = ti.max(lo, -ghost_cell)
                    hi = ti.min(hi, active_cnum + ghost_cell)
                    for i, j in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1])):
                        cell = ti.Vector([i, j])
                        center = (cell.cast(float) + 0.5) * grid_size
                        phi = (center - position).norm() - radius
                        ti.atomic_min(fluid_sdf[cell], phi)
        elif ti.static(GlobalVariable.DIMENSION == 3):
            for np in range(particleNum):
                if int(particle[np].active) == 1:
                    position = particle[np].x
                    radius = equivalent_particle_radius(particle[np].vol)
                    band = max_distance + radius
                    lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                    hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                    lo = ti.max(lo, -ghost_cell)
                    hi = ti.min(hi, active_cnum + ghost_cell)
                    for i, j, k in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])):
                        cell = ti.Vector([i, j, k])
                        center = (cell.cast(float) + 0.5) * grid_size
                        phi = (center - position).norm() - radius
                        ti.atomic_min(fluid_sdf[cell], phi)


@ti.kernel
def kernel_build_fluid_sdf_from_interface_particles(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    particle: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    min_dx = min_grid_spacing(grid_size)
    max_distance = 3.0 * min_dx
    interface_band = 0.5 * min_dx

    if ti.static(GlobalVariable.DIMENSION == 2):
        for np in range(particleNum):
            if int(particle[np].active) == 1 and particle_near_fluid_interface(
                particle[np].x, ghost_cell, cnum, grid_size, cell_type
            ):
                position = particle[np].x
                radius = equivalent_particle_radius(particle[np].vol)
                band = max_distance + radius
                lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                lo = ti.max(lo, -ghost_cell)
                hi = ti.min(hi, active_cnum + ghost_cell)
                for i, j in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1])):
                    cell = ti.Vector([i, j])
                    center = (cell.cast(float) + 0.5) * grid_size
                    phi = (center - position).norm() - radius
                    if is_fluid(cell_type[cell]):
                        phi = ti.min(phi, -interface_band)
                    else:
                        phi = ti.max(phi, interface_band)
                    ti.atomic_min(fluid_sdf[cell], phi)
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for np in range(particleNum):
            if int(particle[np].active) == 1 and particle_near_fluid_interface(
                particle[np].x, ghost_cell, cnum, grid_size, cell_type
            ):
                position = particle[np].x
                radius = equivalent_particle_radius(particle[np].vol)
                band = max_distance + radius
                lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                lo = ti.max(lo, -ghost_cell)
                hi = ti.min(hi, active_cnum + ghost_cell)
                for i, j, k in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])):
                    cell = ti.Vector([i, j, k])
                    center = (cell.cast(float) + 0.5) * grid_size
                    phi = (center - position).norm() - radius
                    if is_fluid(cell_type[cell]):
                        phi = ti.min(phi, -interface_band)
                    else:
                        phi = ti.max(phi, interface_band)
                    ti.atomic_min(fluid_sdf[cell], phi)


@ti.kernel
def kernel_build_fluid_sdf_from_cell_type(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    use_particle_sdf: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    interface_band = 0.5 * min_grid_spacing(grid_size)
    outside_value = interface_band
    if ti.static(use_particle_sdf):
        outside_value = 3.0 * min_grid_spacing(grid_size)
    for I in ti.grouped(fluid_sdf):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
        if inside_active and is_fluid(cell_type[I]):
            fluid_sdf[I] = -interface_band
        else:
            fluid_sdf[I] = outside_value


@ti.kernel
def kernel_build_fluid_sdf_from_volume_fraction(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_volumefrac: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    distance_scale = min_grid_spacing(grid_size)
    # Keep ghost-fluid cuts away from a zero-length pressure stencil when a
    # particle-occupied surface cell has a reconstructed fraction below 0.5.
    minimum_distance = 0.1 * distance_scale
    for I in ti.grouped(fluid_sdf):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
        phi = 0.5 * distance_scale
        if inside_active:
            volume_fraction = ti.min(1.0, ti.max(0.0, cell_volumefrac[linearize(I, cnum)]))
            phi = (0.5 - volume_fraction) * distance_scale
            if is_fluid(cell_type[I]):
                phi = ti.min(phi, -minimum_distance)
            else:
                phi = ti.max(phi, minimum_distance)
        fluid_sdf[I] = phi


@ti.kernel
def kernel_finalize_fluid_sdf_from_cell_type_bounded(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
    bounds: ti.template(),
    fluid_sdf: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    interface_band = 0.5 * min_grid_spacing(grid_size)
    lower = ti.Vector.zero(int, GlobalVariable.DIMENSION)
    upper = ti.Vector.zero(int, GlobalVariable.DIMENSION)
    valid = True
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        lower[d] = bounds[0, d]
        upper[d] = bounds[1, d]
        valid = valid and lower[d] < upper[d]

    if valid:
        if ti.static(GlobalVariable.DIMENSION == 2):
            for i, j in ti.ndrange((lower[0], upper[0]), (lower[1], upper[1])):
                I = ti.Vector([i, j])
                inside_active = True
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
                if inside_active and is_fluid(cell_type[I]):
                    fluid_sdf[I] = ti.min(fluid_sdf[I], -interface_band)
                else:
                    fluid_sdf[I] = ti.max(fluid_sdf[I], interface_band)
        elif ti.static(GlobalVariable.DIMENSION == 3):
            for i, j, k in ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2])):
                I = ti.Vector([i, j, k])
                inside_active = True
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
                if inside_active and is_fluid(cell_type[I]):
                    fluid_sdf[I] = ti.min(fluid_sdf[I], -interface_band)
                else:
                    fluid_sdf[I] = ti.max(fluid_sdf[I], interface_band)


@ti.kernel
def kernel_mark_fluid_cells_from_sdf_bounded(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    bounds: ti.template(),
    fluid_sdf: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    lower = ti.Vector.zero(int, GlobalVariable.DIMENSION)
    upper = ti.Vector.zero(int, GlobalVariable.DIMENSION)
    valid = True
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        lower[d] = bounds[0, d]
        upper[d] = bounds[1, d]
        valid = valid and lower[d] < upper[d]

    if valid:
        if ti.static(GlobalVariable.DIMENSION == 2):
            for i, j in ti.ndrange((lower[0], upper[0]), (lower[1], upper[1])):
                I = ti.Vector([i, j])
                inside_active = True
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
                if inside_active and fluid_sdf[I] < 0.0 and not is_solid(cell_type[I]):
                    cell_type[I] = 1
        elif ti.static(GlobalVariable.DIMENSION == 3):
            for i, j, k in ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2])):
                I = ti.Vector([i, j, k])
                inside_active = True
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
                if inside_active and fluid_sdf[I] < 0.0 and not is_solid(cell_type[I]):
                    cell_type[I] = 1


@ti.kernel
def kernel_mark_fluid_sdf_particle_band(
    particleNum: int,
    step_tag: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    particle: ti.template(),
    fluid_sdf: ti.template(),
    fluid_sdf_stamp: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    min_dx = min_grid_spacing(grid_size)
    max_distance = 3.0 * min_dx

    if ti.static(GlobalVariable.DIMENSION == 2):
        for np in range(particleNum):
            if int(particle[np].active) == 1:
                position = particle[np].x
                radius = equivalent_particle_radius(particle[np].vol)
                band = max_distance + radius
                lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                lo = ti.max(lo, -ghost_cell)
                hi = ti.min(hi, active_cnum + ghost_cell)
                for i, j in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1])):
                    cell = ti.Vector([i, j])
                    if fluid_sdf_stamp[cell] != step_tag:
                        fluid_sdf[cell] = max_distance
                        fluid_sdf_stamp[cell] = step_tag
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for np in range(particleNum):
            if int(particle[np].active) == 1:
                position = particle[np].x
                radius = equivalent_particle_radius(particle[np].vol)
                band = max_distance + radius
                lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                lo = ti.max(lo, -ghost_cell)
                hi = ti.min(hi, active_cnum + ghost_cell)
                for i, j, k in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])):
                    cell = ti.Vector([i, j, k])
                    if fluid_sdf_stamp[cell] != step_tag:
                        fluid_sdf[cell] = max_distance
                        fluid_sdf_stamp[cell] = step_tag


@ti.kernel
def kernel_build_fluid_sdf_from_particles_tagged(
    particleNum: int,
    step_tag: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    particle: ti.template(),
    fluid_sdf: ti.template(),
    fluid_sdf_stamp: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    min_dx = min_grid_spacing(grid_size)
    max_distance = 3.0 * min_dx

    if ti.static(GlobalVariable.DIMENSION == 2):
        for np in range(particleNum):
            if int(particle[np].active) == 1:
                position = particle[np].x
                radius = equivalent_particle_radius(particle[np].vol)
                band = max_distance + radius
                lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                lo = ti.max(lo, -ghost_cell)
                hi = ti.min(hi, active_cnum + ghost_cell)
                for i, j in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1])):
                    cell = ti.Vector([i, j])
                    if fluid_sdf_stamp[cell] == step_tag:
                        center = (cell.cast(float) + 0.5) * grid_size
                        phi = (center - position).norm() - radius
                        ti.atomic_min(fluid_sdf[cell], phi)
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for np in range(particleNum):
            if int(particle[np].active) == 1:
                position = particle[np].x
                radius = equivalent_particle_radius(particle[np].vol)
                band = max_distance + radius
                lo = ti.floor((position - band) / grid_size - 0.5).cast(int)
                hi = ti.floor((position + band) / grid_size - 0.5).cast(int) + 1
                lo = ti.max(lo, -ghost_cell)
                hi = ti.min(hi, active_cnum + ghost_cell)
                for i, j, k in ti.ndrange((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])):
                    cell = ti.Vector([i, j, k])
                    if fluid_sdf_stamp[cell] == step_tag:
                        center = (cell.cast(float) + 0.5) * grid_size
                        phi = (center - position).norm() - radius
                        ti.atomic_min(fluid_sdf[cell], phi)


@ti.kernel
def kernel_mark_fluid_cells_from_sdf_tagged(
    step_tag: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    fluid_sdf_stamp: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            if fluid_sdf_stamp[I] == step_tag and fluid_sdf[I] < 0.0 and not is_solid(cell_type[I]):
                cell_type[I] = 1
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            if fluid_sdf_stamp[I] == step_tag and fluid_sdf[I] < 0.0 and not is_solid(cell_type[I]):
                cell_type[I] = 1


@ti.kernel
def kernel_finalize_fluid_sdf_from_cell_type_tagged(
    step_tag: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    fluid_sdf_stamp: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    interface_band = 0.5 * min_grid_spacing(grid_size)

    for I in ti.grouped(fluid_sdf):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
        if inside_active and is_fluid(cell_type[I]):
            if fluid_sdf_stamp[I] == step_tag:
                fluid_sdf[I] = ti.min(fluid_sdf[I], -interface_band)
            else:
                fluid_sdf[I] = -interface_band
        else:
            if fluid_sdf_stamp[I] == step_tag:
                fluid_sdf[I] = ti.max(fluid_sdf[I], interface_band)
            else:
                fluid_sdf[I] = interface_band


@ti.kernel
def kernel_compute_fluid_surface_tension(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    matProps: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    surface_tension: ti.template(),
):
    surface_tension.fill(0.0)
    sigma = ti.max(0.0, matProps.surface_tension)
    min_dx = min_grid_spacing(grid_size)
    curvature_limit = 2.0 / min_dx
    if ti.static(GlobalVariable.DIMENSION == 3):
        curvature_limit = 4.0 / min_dx

    for I in ti.grouped(surface_tension):
        if sigma > 0.0 and is_air(cell_type[I]):
            near_fluid = False
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                direction = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                near_fluid = near_fluid or get_offset_cell_type(I + direction, ghost_cell, cnum, cell_type) == 1
                near_fluid = near_fluid or get_offset_cell_type(I - direction, ghost_cell, cnum, cell_type) == 1

            if near_fluid:
                curvature = fluid_surface_curvature(I, ghost_cell, cnum, grid_size, fluid_sdf)
                curvature = ti.max(-curvature_limit, ti.min(curvature_limit, curvature))
                surface_tension[I] = sigma * curvature


@ti.kernel
def kernel_compute_fdm_cell_density_from_volume_fraction(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    rest_density: float,
    cell_type: ti.template(),
    cell_volumefrac: ti.template(),
    cell_density: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(cell_density):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and 0 <= I[d] < active_cnum[d]
        if inside_active and is_fluid(cell_type[I]):
            cell_density[I] = rest_density * cell_volumefrac[linearize(I, cnum)]
        else:
            cell_density[I] = 0.0


@ti.kernel
def kernel_compute_density_projection_fluid_fraction(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    cell_fraction: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    eps = min_grid_spacing(grid_size)
    gauss = 0.5773502691896258
    for I in ti.grouped(cell_fraction):
        fraction = 0.0
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and I[d] >= 0 and I[d] < active_cnum[d]
        if inside_active:
            if is_fluid(cell_type[I]) and not cell_has_axis_neighbor_type(I, ghost_cell, cnum, cell_type, 0):
                fraction = 1.0
            elif not is_solid(cell_type[I]) and cell_has_axis_neighbor_type(I, ghost_cell, cnum, cell_type, 1):
                if ti.static(GlobalVariable.DIMENSION == 2):
                    for i, j in ti.static(ti.ndrange(2, 2)):
                        xi = -gauss
                        eta = -gauss
                        if ti.static(i == 1):
                            xi = gauss
                        if ti.static(j == 1):
                            eta = gauss
                        position = (I.cast(float) + ti.Vector([0.5 + 0.5 * xi, 0.5 + 0.5 * eta])) * grid_size
                        phi = sample_cell_centered_fluid_sdf(position, ghost_cell, cnum, grid_size, fluid_sdf)
                        fraction += 0.25 * smoothed_fluid_heaviside(phi, eps)
                elif ti.static(GlobalVariable.DIMENSION == 3):
                    for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                        xi = -gauss
                        eta = -gauss
                        zeta = -gauss
                        if ti.static(i == 1):
                            xi = gauss
                        if ti.static(j == 1):
                            eta = gauss
                        if ti.static(k == 1):
                            zeta = gauss
                        position = (
                            I.cast(float) + ti.Vector([0.5 + 0.5 * xi, 0.5 + 0.5 * eta, 0.5 + 0.5 * zeta])
                        ) * grid_size
                        phi = sample_cell_centered_fluid_sdf(position, ghost_cell, cnum, grid_size, fluid_sdf)
                        fraction += 0.125 * smoothed_fluid_heaviside(phi, eps)
        cell_fraction[I] = ti.min(1.0, ti.max(0.0, fraction))


@ti.kernel
def kernel_assemble_density_projection_rhs(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    dt: ti.template(),
    flag: ti.template(),
    cell_type: ti.template(),
    cell_density: ti.template(),
    cell_fraction: ti.template(),
    matProps: ti.template(),
    tolerance: float,
    error_clamp: float,
    interior_only: ti.template(),
    use_fluid_fraction: ti.template(),
    right_hand_vector: ti.template(),
):
    right_hand_vector.fill(0.0)
    active_cnum = cnum - 2 * ghost_cell
    scale = matProps.density / (dt[None] * dt[None])
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            if density_projection_cell_is_active(I, ghost_cell, cnum, cell_type, interior_only):
                rho = cell_density[I]
                if rho > 0.0:
                    volume_fraction = 1.0
                    if ti.static(use_fluid_fraction):
                        volume_fraction = cell_fraction[I]
                    reference_density = matProps.density * ti.max(volume_fraction, 1.0e-4)
                    density_error = rho / reference_density - 1.0
                    if ti.abs(density_error) < tolerance:
                        density_error = 0.0
                    density_error = ti.min(error_clamp, ti.max(-error_clamp, density_error))
                    dof_index = flag[linearize(I, active_cnum)]
                    right_hand_vector[dof_index] = scale * volume_fraction * density_error
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            if density_projection_cell_is_active(I, ghost_cell, cnum, cell_type, interior_only):
                rho = cell_density[I]
                if rho > 0.0:
                    volume_fraction = 1.0
                    if ti.static(use_fluid_fraction):
                        volume_fraction = cell_fraction[I]
                    reference_density = matProps.density * ti.max(volume_fraction, 1.0e-4)
                    density_error = rho / reference_density - 1.0
                    if ti.abs(density_error) < tolerance:
                        density_error = 0.0
                    density_error = ti.min(error_clamp, ti.max(-error_clamp, density_error))
                    dof_index = flag[linearize(I, active_cnum)]
                    right_hand_vector[dof_index] = scale * volume_fraction * density_error


@ti.kernel
def kernel_density_projection_needs_solve(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    cell_density: ti.template(),
    cell_fraction: ti.template(),
    rest_density: float,
    tolerance: float,
    interior_only: ti.template(),
    use_fluid_fraction: ti.template(),
    needs_solve: ti.template(),
):
    needs_solve[None] = 0
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(cell_density):
        inside_active = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside_active = inside_active and 0 <= I[d] < active_cnum[d]
        if inside_active and density_projection_cell_is_active(I, ghost_cell, cnum, cell_type, interior_only):
            volume_fraction = 1.0
            if ti.static(use_fluid_fraction):
                volume_fraction = cell_fraction[I]
            reference_density = rest_density * ti.max(volume_fraction, 1.0e-4)
            if cell_density[I] > 0.0 and ti.abs(cell_density[I] / reference_density - 1.0) >= tolerance:
                ti.atomic_max(needs_solve[None], 1)


@ti.kernel
def kernel_assemble_density_projection_mg_b(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    dt: ti.template(),
    grid_type: ti.template(),
    cell_type: ti.template(),
    cell_density: ti.template(),
    cell_fraction: ti.template(),
    matProps: ti.template(),
    tolerance: float,
    error_clamp: float,
    interior_only: ti.template(),
    use_fluid_fraction: ti.template(),
    b: ti.template(),
):
    b.fill(0.0)
    scale = 1.0 / dt[None]
    for I in ti.grouped(grid_type):
        if grid_type[I] == 1 and density_projection_cell_is_active(I, ghost_cell, cnum, cell_type, interior_only):
            rho = cell_density[I]
            if rho > 0.0:
                volume_fraction = 1.0
                if ti.static(use_fluid_fraction):
                    volume_fraction = cell_fraction[I]
                reference_density = matProps.density * ti.max(volume_fraction, 1.0e-4)
                density_error = rho / reference_density - 1.0
                if ti.abs(density_error) < tolerance:
                    density_error = 0.0
                density_error = ti.min(error_clamp, ti.max(-error_clamp, density_error))
                b[I] = scale * volume_fraction * density_error


@ti.func
def density_projection_face_displacement(
    direction: ti.template(),
    face,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    matProps: ti.template(),
    pressure: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
):
    offset = ti.Vector.unit(GlobalVariable.DIMENSION, direction)
    left = face - offset
    right = face
    left_type = get_offset_cell_type(left, ghost_cell, cnum, cell_type)
    right_type = get_offset_cell_type(right, ghost_cell, cnum, cell_type)
    gradient = 0.0
    if left_type == 1 and right_type == 1:
        gradient = (pressure[right] - pressure[left]) / grid_size[direction]
    elif left_type == 1 and right_type == 0:
        theta = 1.0
        if ti.static(use_free_surface_theta):
            theta = free_surface_theta(left, right, fluid_sdf)
        gradient = (0.0 - pressure[left]) / (theta * grid_size[direction])
    elif left_type == 0 and right_type == 1:
        theta = 1.0
        if ti.static(use_free_surface_theta):
            theta = free_surface_theta(right, left, fluid_sdf)
        gradient = (pressure[right] - 0.0) / (theta * grid_size[direction])
    return -dt[None] * dt[None] / matProps.density * gradient


@ti.kernel
def kernel_compute_density_projection_face_displacement(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    matProps: ti.template(),
    pressure: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
    face_displacement0: ti.template(),
    face_displacement1: ti.template(),
    face_displacement2: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
        if ti.static(d == 0):
            for I in ti.grouped(face_displacement0):
                inside_face = True
                for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                    if ti.static(d1 == d):
                        inside_face = inside_face and I[d1] >= 0 and I[d1] < face_count[d1]
                    else:
                        inside_face = inside_face and I[d1] >= -ghost_cell and I[d1] < face_count[d1] + ghost_cell
                if inside_face:
                    face_displacement0[I] = density_projection_face_displacement(
                        d,
                        I,
                        ghost_cell,
                        cnum,
                        grid_size,
                        dt,
                        matProps,
                        pressure,
                        cell_type,
                        fluid_sdf,
                        use_free_surface_theta,
                    )
                else:
                    face_displacement0[I] = 0.0
        elif ti.static(d == 1):
            for I in ti.grouped(face_displacement1):
                inside_face = True
                for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                    if ti.static(d1 == d):
                        inside_face = inside_face and I[d1] >= 0 and I[d1] < face_count[d1]
                    else:
                        inside_face = inside_face and I[d1] >= -ghost_cell and I[d1] < face_count[d1] + ghost_cell
                if inside_face:
                    face_displacement1[I] = density_projection_face_displacement(
                        d,
                        I,
                        ghost_cell,
                        cnum,
                        grid_size,
                        dt,
                        matProps,
                        pressure,
                        cell_type,
                        fluid_sdf,
                        use_free_surface_theta,
                    )
                else:
                    face_displacement1[I] = 0.0
        else:
            for I in ti.grouped(face_displacement2):
                inside_face = True
                for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                    if ti.static(d1 == d):
                        inside_face = inside_face and I[d1] >= 0 and I[d1] < face_count[d1]
                    else:
                        inside_face = inside_face and I[d1] >= -ghost_cell and I[d1] < face_count[d1] + ghost_cell
                if inside_face:
                    face_displacement2[I] = density_projection_face_displacement(
                        d,
                        I,
                        ghost_cell,
                        cnum,
                        grid_size,
                        dt,
                        matProps,
                        pressure,
                        cell_type,
                        fluid_sdf,
                        use_free_surface_theta,
                    )
                else:
                    face_displacement2[I] = 0.0


@ti.func
def density_projection_face_shape_1d(
    pos, grid_pos, inv_dx, psize, grid_index: int, face_count: int, normal_axis: ti.template()
):
    value = 0.0
    if ti.static(GlobalVariable.SHAPEFUNCTION == 0):
        value = ShapeLinear(pos, grid_pos, inv_dx, 0)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 1):
        value = ShapeGIMP(pos, grid_pos, inv_dx, psize)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 2):
        btypes = 0
        if ti.static(normal_axis):
            btypes = mac_boundary_type_1d(grid_index, face_count)
        value = ShapeBsplineQ(pos, grid_pos, inv_dx, btypes)
    elif ti.static(GlobalVariable.SHAPEFUNCTION == 3):
        btypes = 0
        if ti.static(normal_axis):
            btypes = mac_boundary_type_1d(grid_index, face_count)
        value = ShapeBsplineC(pos, grid_pos, inv_dx, btypes)
    return value


@ti.func
def cached_density_projection_face_displacement(
    direction: ti.template(),
    grid_id,
    face_displacement0: ti.template(),
    face_displacement1: ti.template(),
    face_displacement2: ti.template(),
):
    value = 0.0
    if ti.static(direction == 0):
        value = face_displacement0[grid_id]
    elif ti.static(direction == 1):
        value = face_displacement1[grid_id]
    else:
        value = face_displacement2[grid_id]
    return value


@ti.kernel
def kernel_apply_density_projection_position_correction_cached(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    matProps: ti.template(),
    max_shift_ratio: float,
    cell_type: ti.template(),
    interior_only: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
    face_displacement0: ti.template(),
    face_displacement1: ti.template(),
    face_displacement2: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    max_shift = max_shift_ratio * min_grid_spacing(grid_size)
    for np in range(particleNum):
        if int(particle[np].active) == 1 and particle_cell_is_active_fluid(
            particle[np].x, ghost_cell, cnum, grid_size, cell_type, interior_only
        ):
            bodyID = int(particle[np].bodyID)
            pos = particle[np].x
            psize = particle_lengths[bodyID]
            delta_x = ti.Vector.zero(float, GlobalVariable.DIMENSION)
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                stagger = 0.5 * (1 - ti.Vector.unit(GlobalVariable.DIMENSION, d))
                face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
                base = ti.floor((pos - psize) * igrid_size - stagger).cast(int)
                shape_cache = ti.Matrix.zero(float, GlobalVariable.DIMENSION, GlobalVariable.INFLUENCENODE)
                for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                    for local_id in ti.static(range(GlobalVariable.INFLUENCENODE)):
                        axis_index = base[d1] + local_id
                        axis_pos = (axis_index + stagger[d1]) * grid_size[d1]
                        if ti.static(d1 == d):
                            shape_cache[d1, local_id] = density_projection_face_shape_1d(
                                pos[d1], axis_pos, igrid_size[d1], psize[d1], axis_index, face_count[d1], True
                            )
                        else:
                            shape_cache[d1, local_id] = density_projection_face_shape_1d(
                                pos[d1], axis_pos, igrid_size[d1], psize[d1], axis_index, face_count[d1], False
                            )

                for offset in ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION))):
                    grid_id = base + offset
                    inside_face = True
                    for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                        if ti.static(GlobalVariable.SHAPEFUNCTION == 2):
                            if ti.static(d1 == d):
                                inside_face = inside_face and grid_id[d1] >= 0 and grid_id[d1] < face_count[d1]
                            else:
                                inside_face = (
                                    inside_face
                                    and grid_id[d1] >= -ghost_cell
                                    and grid_id[d1] < face_count[d1] + ghost_cell
                                )
                        elif ti.static(d1 == d):
                            inside_face = inside_face and grid_id[d1] >= 0 and grid_id[d1] < face_count[d1]
                        else:
                            inside_face = (
                                inside_face and grid_id[d1] >= -ghost_cell and grid_id[d1] < face_count[d1] + ghost_cell
                            )

                    if inside_face:
                        weight = 1.0
                        for d0 in ti.static(range(GlobalVariable.DIMENSION)):
                            axis_shape = 0.0
                            for local_id in ti.static(range(GlobalVariable.INFLUENCENODE)):
                                if offset[d0] == local_id:
                                    axis_shape = shape_cache[d0, local_id]
                            weight *= axis_shape
                        if weight > Threshold:
                            delta_x[d] += weight * cached_density_projection_face_displacement(
                                d, grid_id, face_displacement0, face_displacement1, face_displacement2
                            )

            delta_norm = delta_x.norm()
            if delta_norm > max_shift and delta_norm > Threshold:
                delta_x *= max_shift / delta_norm
            shift_incompressible_particle(np, delta_x, particle)


@ti.kernel
def kernel_apply_density_projection_position_correction(
    total_nodes: int,
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    matProps: ti.template(),
    max_shift_ratio: float,
    pressure: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    interior_only: ti.template(),
    use_free_surface_theta: ti.template(),
    particle: ti.template(),
    particle_lengths: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    max_shift = max_shift_ratio * min_grid_spacing(grid_size)
    for np in range(particleNum):
        if int(particle[np].active) == 1 and particle_cell_is_active_fluid(
            particle[np].x, ghost_cell, cnum, grid_size, cell_type, interior_only
        ):
            bodyID = int(particle[np].bodyID)
            pos = particle[np].x
            psize = particle_lengths[bodyID]
            delta_x = ti.Vector.zero(float, GlobalVariable.DIMENSION)
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                stagger = 0.5 * (1 - ti.Vector.unit(GlobalVariable.DIMENSION, d))
                face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
                base = ti.floor((pos - psize) * igrid_size - stagger).cast(int)
                for offset in ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE,) * GlobalVariable.DIMENSION))):
                    grid_id = base + offset
                    inside_face = True
                    for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                        if ti.static(GlobalVariable.SHAPEFUNCTION == 2):
                            if ti.static(d1 == d):
                                inside_face = inside_face and grid_id[d1] >= 0 and grid_id[d1] < face_count[d1]
                            else:
                                inside_face = (
                                    inside_face
                                    and grid_id[d1] >= -ghost_cell
                                    and grid_id[d1] < face_count[d1] + ghost_cell
                                )
                        elif ti.static(d1 == d):
                            inside_face = inside_face and grid_id[d1] >= 0 and grid_id[d1] < face_count[d1]
                        else:
                            inside_face = (
                                inside_face and grid_id[d1] >= -ghost_cell and grid_id[d1] < face_count[d1] + ghost_cell
                            )

                    if inside_face:
                        grid_pos = (grid_id + stagger) * grid_size
                        shape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                        if ti.static(GlobalVariable.SHAPEFUNCTION == 0):
                            for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                                shape_fn[d1] = ShapeLinear(pos[d1], grid_pos[d1], igrid_size[d1], 0)
                        elif ti.static(GlobalVariable.SHAPEFUNCTION == 1):
                            for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                                shape_fn[d1] = ShapeGIMP(pos[d1], grid_pos[d1], igrid_size[d1], psize[d1])
                        elif ti.static(GlobalVariable.SHAPEFUNCTION == 2):
                            for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                                btypes = 0
                                if ti.static(d1 == d):
                                    btypes = mac_boundary_type_1d(grid_id[d1], face_count[d1])
                                shape_fn[d1] = ShapeBsplineQ(pos[d1], grid_pos[d1], igrid_size[d1], btypes)
                        elif ti.static(GlobalVariable.SHAPEFUNCTION == 3):
                            for d1 in ti.static(range(GlobalVariable.DIMENSION)):
                                btypes = 0
                                if ti.static(d1 == d):
                                    btypes = mac_boundary_type_1d(grid_id[d1], face_count[d1])
                                shape_fn[d1] = ShapeBsplineC(pos[d1], grid_pos[d1], igrid_size[d1], btypes)

                        weight = 1.0
                        for d0 in ti.static(range(GlobalVariable.DIMENSION)):
                            weight *= shape_fn[d0]
                        delta_x[d] += weight * density_projection_face_displacement(
                            d,
                            grid_id,
                            ghost_cell,
                            cnum,
                            grid_size,
                            dt,
                            matProps,
                            pressure,
                            cell_type,
                            fluid_sdf,
                            use_free_surface_theta,
                        )

            delta_norm = delta_x.norm()
            if delta_norm > max_shift and delta_norm > Threshold:
                delta_x *= max_shift / delta_norm
            shift_incompressible_particle(np, delta_x, particle)


@ti.kernel
def kernel_deactivate_particles_in_solid_cells(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
    particle: ti.template(),
) -> int:
    deactivated = 0
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            if not particle_position_is_valid(particle[np].x):
                particle[np].active = ti.u8(0)
                deactivated += 1
            else:
                cell = ti.floor(particle[np].x * igrid_size, int)
                if get_offset_cell_type(cell, ghost_cell, cnum, cell_type) == 2:
                    particle[np].active = ti.u8(0)
                    deactivated += 1
    return deactivated


@ti.kernel
def kernel_deactivate_particles_in_solid_sdf(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    solid_sdf: ti.template(),
    particle: ti.template(),
) -> int:
    deactivated = 0
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            if not particle_position_is_valid(particle[np].x):
                particle[np].active = ti.u8(0)
                deactivated += 1
            else:
                cell = ti.floor(particle[np].x / grid_size, int)
                if (
                    sample_solid_sdf(cell, ghost_cell, cnum, solid_sdf) < 0.0
                    and interpolate_solid_sdf(particle[np].x, ghost_cell, cnum, grid_size, solid_sdf) < 0.0
                ):
                    particle[np].active = ti.u8(0)
                    deactivated += 1
    return deactivated


@ti.kernel
def kernel_update_incompressible_particle_storage(particleNum: int, particle: ti.template()) -> int:
    remaining_particle = 0
    ti.loop_config(serialize=True)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            particle[remaining_particle] = particle[np]
            remaining_particle += 1
    return remaining_particle


@ti.kernel
def kernel_initialize_fdm_cell_pressure_from_particles(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
    pressure: ti.template(),
    pressure_weight: ti.template(),
    particle: ti.template(),
):
    pressure.fill(0)
    pressure_weight.fill(0)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            cell = ti.floor(particle[np].x * igrid_size, int)
            if get_offset_cell_type(cell, ghost_cell, cnum, cell_type) == 1:
                weight = particle[np].vol
                pressure[cell] -= particle[np].pressure * weight
                pressure_weight[cell] += weight

    for I in ti.grouped(cell_type):
        if is_fluid(cell_type[I]) and pressure_weight[I] > 0.0:
            pressure[I] /= pressure_weight[I]


@ti.kernel
def kernel_copy_incompressible_mg_cell_type(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    cell_type: ti.template(),
    grid_type: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(grid_type):
        inside = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside = inside and I[d] < active_cnum[d]
        if inside:
            grid_type[I] = cell_type[I]
        else:
            grid_type[I] = 2


@ti.kernel
def kernel_prepare_incompressible_mg_level0(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    node: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    matProps: ti.template(),
    use_free_surface_theta: ti.template(),
    grid_type: ti.template(),
    b: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    scale_A = dt[None] / matProps.density * igrid_size * igrid_size
    scale_b = igrid_size
    for I in ti.grouped(grid_type):
        inside = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside = inside and I[d] < active_cnum[d]
        cell_attr = 2
        if inside:
            cell_attr = int(cell_type[I])
        grid_type[I] = cell_attr

        b[I] = 0.0
        if cell_attr == 1:
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                b[I] += (node.velocity[d][I] - node.velocity[d][I + offset]) * scale_b[d]

            for d in ti.static(range(GlobalVariable.DIMENSION)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                    neighbor = I + offset
                    neighbor_type = get_offset_cell_type(neighbor, ghost_cell, cnum, cell_type)
                    if neighbor_type == 2:
                        if s < 0:
                            b[I] -= scale_b[d] * node.velocity[d][I]
                        else:
                            b[I] += scale_b[d] * node.velocity[d][I + offset]
                    elif neighbor_type == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I, neighbor, fluid_sdf)
                        b[I] += scale_A[d] / theta * (matProps.atmospheric_pressure + surface_tension[neighbor])


@ti.kernel
def kernel_assemble_incompressible_mg_b(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    node: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    matProps: ti.template(),
    grid_type: ti.template(),
    b: ti.template(),
):
    b.fill(0)
    scale_A = dt[None] / matProps.density * igrid_size * igrid_size
    scale_b = igrid_size
    for I in ti.grouped(grid_type):
        if grid_type[I] == 1:
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                b[I] += (node.velocity[d][I] - node.velocity[d][I + offset]) * scale_b[d]

    for I in ti.grouped(grid_type):
        if grid_type[I] == 1:
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                    neighbor = I + offset
                    neighbor_type = get_offset_cell_type(neighbor, ghost_cell, cnum, cell_type)
                    if neighbor_type == 2:
                        if s < 0:
                            b[I] -= scale_b[d] * node.velocity[d][I]
                        else:
                            b[I] += scale_b[d] * node.velocity[d][I + offset]
                    elif neighbor_type == 0:
                        b[I] += scale_A[d] * (matProps.atmospheric_pressure + surface_tension[neighbor])


@ti.kernel
def kernel_prepare_incompressible_mg_level0_cut_cell(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    node: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    solid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    matProps: ti.template(),
    use_free_surface_theta: ti.template(),
    grid_type: ti.template(),
    b: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    scale_A = dt[None] / matProps.density * igrid_size * igrid_size
    scale_b = igrid_size
    for I in ti.grouped(grid_type):
        inside = True
        for d in ti.static(range(GlobalVariable.DIMENSION)):
            inside = inside and I[d] < active_cnum[d]
        cell_attr = 2
        if inside:
            cell_attr = int(cell_type[I])
        grid_type[I] = cell_attr
        b[I] = 0.0
        if cell_attr == 1:
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                left_face = I
                right_face = I + offset
                left_fraction = select_solid_face_fraction(d, left_face, face_fraction0, face_fraction1, face_fraction2)
                right_fraction = select_solid_face_fraction(
                    d, right_face, face_fraction0, face_fraction1, face_fraction2
                )
                left_velocity = left_fraction * node.velocity[d][left_face] + (
                    1.0 - left_fraction
                ) * select_solid_face_velocity(d, left_face, solid_velocity0, solid_velocity1, solid_velocity2)
                right_velocity = right_fraction * node.velocity[d][right_face] + (
                    1.0 - right_fraction
                ) * select_solid_face_velocity(d, right_face, solid_velocity0, solid_velocity1, solid_velocity2)
                b[I] += (left_velocity - right_velocity) * scale_b[d]
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                    neighbor = I + offset
                    neighbor_type = get_offset_cell_type(neighbor, ghost_cell, cnum, cell_type)
                    face = I
                    if ti.static(s > 0):
                        face = I + offset
                    face_fraction = select_solid_face_fraction(d, face, face_fraction0, face_fraction1, face_fraction2)
                    if face_fraction > 0.0 and neighbor_type == 0 and solid_sdf[neighbor] >= 0.0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I, neighbor, fluid_sdf)
                        b[I] += (
                            face_fraction
                            * scale_A[d]
                            / theta
                            * (matProps.atmospheric_pressure + surface_tension[neighbor])
                        )


@ti.func
def mg_coupled_cell_fraction_3d(cell, mode: ti.template(), solid_fraction: ti.template()):
    fraction = 1.0
    if ti.static(mode == 1):
        fraction = ti.max(0.05, 1.0 - ti.min(1.0, ti.max(0.0, solid_fraction[cell])))
    return fraction


@ti.func
def mg_coupled_cell_density_3d(
    cell, rho_f, mode: ti.template(), solid_fraction: ti.template(), solid_density: ti.template()
):
    density = rho_f
    if ti.static(mode == 2):
        phi_s = ti.min(1.0, ti.max(0.0, solid_fraction[cell]))
        rho_s = solid_density[cell]
        if rho_s <= Threshold:
            rho_s = rho_f
        density = (1.0 - phi_s) * rho_f + phi_s * rho_s
    return density


@ti.func
def mg_coupled_face_terms_3d(
    cell,
    neighbor,
    rho_f,
    mode: ti.template(),
    cell_type: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
):
    flux_fraction = mg_coupled_cell_fraction_3d(cell, mode, solid_fraction)
    density = mg_coupled_cell_density_3d(cell, rho_f, mode, solid_fraction, solid_density)
    if int(cell_type[neighbor]) == 1:
        flux_fraction = 0.5 * (flux_fraction + mg_coupled_cell_fraction_3d(neighbor, mode, solid_fraction))
        density = 0.5 * (density + mg_coupled_cell_density_3d(neighbor, rho_f, mode, solid_fraction, solid_density))
    return flux_fraction, flux_fraction / ti.max(density, Threshold)


@ti.func
def mg_coupled_open_fraction_3d(
    direction: ti.template(),
    face,
    use_cut_cell: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
):
    fraction = 1.0
    if ti.static(use_cut_cell):
        fraction = select_solid_face_fraction(direction, face, face_fraction0, face_fraction1, face_fraction2)
    return fraction


@ti.kernel
def kernel_prepare_incompressible_mg_level0_coupled_3d(
    ghost_cell: int,
    cnum: ti.types.vector(3, int),
    igrid_size: ti.types.vector(3, float),
    dt: ti.template(),
    node: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    solid_fraction: ti.template(),
    previous_solid_fraction: ti.template(),
    solid_density: ti.template(),
    matProps: ti.template(),
    mode: ti.template(),
    use_cut_cell: ti.template(),
    use_free_surface_theta: ti.template(),
    grid_type: ti.template(),
    b: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(grid_type):
        inside = I[0] < active_cnum[0] and I[1] < active_cnum[1] and I[2] < active_cnum[2]
        cell_attr = 2
        if inside:
            cell_attr = int(cell_type[I])
        grid_type[I] = cell_attr
        b[I] = 0.0
        if cell_attr == 1:
            for d in ti.static(range(3)):
                unit = ti.Vector.unit(3, d)
                left_neighbor = I - unit
                right_neighbor = I + unit
                left_fraction, _ = mg_coupled_face_terms_3d(
                    I, left_neighbor, matProps.density, mode, cell_type, solid_fraction, solid_density
                )
                right_fraction, _ = mg_coupled_face_terms_3d(
                    I, right_neighbor, matProps.density, mode, cell_type, solid_fraction, solid_density
                )
                left_open = mg_coupled_open_fraction_3d(
                    d, I, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                )
                right_open = mg_coupled_open_fraction_3d(
                    d, I + unit, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                )
                left_velocity = node.velocity[d][I]
                right_velocity = node.velocity[d][I + unit]
                if ti.static(use_cut_cell):
                    left_velocity = left_open * left_velocity + (1.0 - left_open) * select_solid_face_velocity(
                        d, I, solid_velocity0, solid_velocity1, solid_velocity2
                    )
                    right_velocity = right_open * right_velocity + (1.0 - right_open) * select_solid_face_velocity(
                        d, I + unit, solid_velocity0, solid_velocity1, solid_velocity2
                    )
                else:
                    if int(cell_type[left_neighbor]) == 2:
                        left_velocity = 0.0
                    if int(cell_type[right_neighbor]) == 2:
                        right_velocity = 0.0
                b[I] += (left_fraction * left_velocity - right_fraction * right_velocity) * igrid_size[d]

            if ti.static(mode == 1):
                current_fraction = mg_coupled_cell_fraction_3d(I, mode, solid_fraction)
                previous_fraction = mg_coupled_cell_fraction_3d(I, mode, previous_solid_fraction)
                b[I] += (previous_fraction - current_fraction) / ti.max(dt[None], Threshold)

            for d in ti.static(range(3)):
                for side in ti.static((-1, 1)):
                    offset = side * ti.Vector.unit(3, d)
                    neighbor = I + offset
                    face = I
                    if ti.static(side > 0):
                        face = I + offset
                    face_open = mg_coupled_open_fraction_3d(
                        d, face, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                    )
                    _, coefficient = mg_coupled_face_terms_3d(
                        I, neighbor, matProps.density, mode, cell_type, solid_fraction, solid_density
                    )
                    if int(cell_type[neighbor]) == 0 and face_open > 0.0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I, neighbor, fluid_sdf)
                        b[I] += (
                            dt[None]
                            * face_open
                            * coefficient
                            * igrid_size[d]
                            * igrid_size[d]
                            / theta
                            * (matProps.atmospheric_pressure + surface_tension[neighbor])
                        )


@ti.kernel
def kernel_assemble_incompressible_mg_A_level0_coupled_3d(
    dt: ti.template(),
    igrid_size: ti.types.vector(3, float),
    matProps: ti.template(),
    grid_type: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
    mode: ti.template(),
    use_cut_cell: ti.template(),
    use_free_surface_theta: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
):
    for I in ti.grouped(grid_type):
        Adiag[I] = 0.0
        Ax[I] = ti.zero(Ax[I])
        if int(grid_type[I]) == 1:
            for d in ti.static(range(3)):
                for side in ti.static((-1, 1)):
                    offset = side * ti.Vector.unit(3, d)
                    neighbor = I + offset
                    neighbor_type = get_mg_cell_type(neighbor, grid_type)
                    face = I
                    if ti.static(side > 0):
                        face = I + offset
                    face_open = mg_coupled_open_fraction_3d(
                        d, face, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                    )
                    _, coefficient = mg_coupled_face_terms_3d(
                        I, neighbor, matProps.density, mode, cell_type, solid_fraction, solid_density
                    )
                    coefficient *= dt[None] * face_open * igrid_size[d] * igrid_size[d]
                    if neighbor_type == 1:
                        Adiag[I] += coefficient
                        if ti.static(side > 0):
                            Ax[I][d] = -coefficient
                    elif neighbor_type == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I, neighbor, fluid_sdf)
                        Adiag[I] += coefficient / theta
            if Adiag[I] <= 0.0:
                Adiag[I] = 1.0


@ti.kernel
def kernel_assemble_incompressible_mg_A(
    dt: ti.template(),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    matProps: ti.template(),
    grid_type: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
):
    scale_A = dt[None] / matProps.density * igrid_size * igrid_size
    for I in ti.grouped(grid_type):
        Adiag[I] = 0.0
        Ax[I] = ti.zero(Ax[I])
        if grid_type[I] == 1:
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                    neighbor_type = get_mg_cell_type(I + offset, grid_type)
                    if neighbor_type == 1:
                        Adiag[I] += scale_A[d]
                        if ti.static(s > 0):
                            Ax[I][d] = -scale_A[d]
                    elif neighbor_type == 0:
                        Adiag[I] += scale_A[d]
            if Adiag[I] <= 0.0:
                Adiag[I] = 1.0


@ti.kernel
def kernel_assemble_incompressible_mg_A_level0(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    dt: ti.template(),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    matProps: ti.template(),
    grid_type: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
):
    scale_A = dt[None] / matProps.density * igrid_size * igrid_size
    for I in ti.grouped(grid_type):
        Adiag[I] = 0.0
        Ax[I] = ti.zero(Ax[I])
        if grid_type[I] == 1:
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                    neighbor = I + offset
                    neighbor_type = get_offset_cell_type(neighbor, ghost_cell, cnum, cell_type)
                    if neighbor_type == 1:
                        Adiag[I] += scale_A[d]
                        if ti.static(s > 0):
                            Ax[I][d] = -scale_A[d]
                    elif neighbor_type == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I, neighbor, fluid_sdf)
                        Adiag[I] += scale_A[d] / theta
            if Adiag[I] <= 0.0:
                Adiag[I] = 1.0


@ti.kernel
def kernel_assemble_incompressible_mg_A_level0_cut_cell(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    dt: ti.template(),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    matProps: ti.template(),
    grid_type: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    solid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
):
    scale_A = dt[None] / matProps.density * igrid_size * igrid_size
    for I in ti.grouped(grid_type):
        Adiag[I] = 0.0
        Ax[I] = ti.zero(Ax[I])
        if grid_type[I] == 1:
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                    neighbor = I + offset
                    neighbor_type = get_offset_cell_type(neighbor, ghost_cell, cnum, cell_type)
                    face = I
                    if ti.static(s > 0):
                        face = I + offset
                    face_fraction = select_solid_face_fraction(d, face, face_fraction0, face_fraction1, face_fraction2)
                    if face_fraction > 0.0:
                        if neighbor_type == 1:
                            coefficient = face_fraction * scale_A[d]
                            Adiag[I] += coefficient
                            if ti.static(s > 0):
                                Ax[I][d] = -coefficient
                        elif neighbor_type == 0 and solid_sdf[neighbor] >= 0.0:
                            theta = 1.0
                            if ti.static(use_free_surface_theta):
                                theta = free_surface_theta(I, neighbor, fluid_sdf)
                            Adiag[I] += face_fraction * scale_A[d] / theta
            if Adiag[I] <= 0.0:
                Adiag[I] = 1.0


@ti.kernel
def kernel_update_cell_pressure_from_mg(pressure: ti.template(), grid_type: ti.template(), x: ti.template()):
    pressure.fill(0)
    for I in ti.grouped(grid_type):
        if grid_type[I] == 1:
            pressure[I] = x[I]


@ti.kernel
def kernel_correct_velocity_coupled_3d(
    ghost_cell: int,
    cnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    dt: ti.template(),
    matProps: ti.template(),
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
    mode: ti.template(),
    use_cut_cell: ti.template(),
    use_free_surface_theta: ti.template(),
    node: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for direction in ti.static(range(3)):
        face_cnum = active_cnum + ti.Vector.unit(3, direction)
        for I in ti.grouped(ti.ndrange((0, face_cnum[0]), (0, face_cnum[1]), (0, face_cnum[2]))):
            left = I - ti.Vector.unit(3, direction)
            right = I
            left_type = int(cell_type[left])
            right_type = int(cell_type[right])
            if left_type == 1 or right_type == 1:
                face_open = mg_coupled_open_fraction_3d(
                    direction, I, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                )
                if face_open <= Threshold or left_type == 2 or right_type == 2:
                    wall_velocity = 0.0
                    if ti.static(use_cut_cell):
                        wall_velocity = select_solid_face_velocity(
                            direction, I, solid_velocity0, solid_velocity1, solid_velocity2
                        )
                    node.velocity[direction][I] = wall_velocity
                    node.force[direction][I] = 0.0
                else:
                    fluid_cell = left
                    other_cell = right
                    if left_type != 1:
                        fluid_cell = right
                        other_cell = left
                    flux_fraction, pressure_coefficient = mg_coupled_face_terms_3d(
                        fluid_cell,
                        other_cell,
                        matProps.density,
                        mode,
                        cell_type,
                        solid_fraction,
                        solid_density,
                    )
                    inverse_density = pressure_coefficient / ti.max(flux_fraction, Threshold)
                    pressure_jump = 0.0
                    theta = 1.0
                    if right_type == 0:
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(left, right, fluid_sdf)
                        pressure_jump = matProps.atmospheric_pressure + surface_tension[right] - pressure[left]
                    elif left_type == 0:
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(right, left, fluid_sdf)
                        pressure_jump = pressure[right] - matProps.atmospheric_pressure - surface_tension[left]
                    else:
                        pressure_jump = pressure[right] - pressure[left]
                    node.velocity[direction][I] -= (
                        dt[None] * inverse_density * pressure_jump / (grid_size[direction] * theta)
                    )


@ti.kernel
def kernel_update_pressure_and_correct_velocity_from_mg(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    matProps: ti.template(),
    x: ti.template(),
    grid_type: ti.template(),
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
    node: ti.template(),
):
    pressure.fill(0)
    for I in ti.grouped(grid_type):
        if grid_type[I] == 1:
            pressure[I] = x[I]

    active_cnum = cnum - 2 * ghost_cell
    scale = dt[None] / (matProps.density * grid_size)
    atmospheric_pressure = matProps.atmospheric_pressure
    for k in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, k)
        if ti.static(GlobalVariable.DIMENSION == 2):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]))):
                I_1 = I - ti.Vector.unit(GlobalVariable.DIMENSION, k)
                cell_1 = mac_get_offset_cell_type(I_1, ghost_cell, cnum, cell_type)
                cell_2 = mac_get_offset_cell_type(I, ghost_cell, cnum, cell_type)
                if cell_1 == 1 or cell_2 == 1:
                    if cell_1 == 2 or cell_2 == 2:
                        node.velocity[k][I] = 0
                        node.force[k][I] = 0
                    elif cell_2 == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I_1, I, fluid_sdf)
                        node.velocity[k][I] -= (
                            scale[k] / theta * (atmospheric_pressure + surface_tension[I] - pressure[I_1])
                        )
                    elif cell_1 == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I, I_1, fluid_sdf)
                        node.velocity[k][I] -= (
                            scale[k] / theta * (pressure[I] - atmospheric_pressure - surface_tension[I_1])
                        )
                    else:
                        node.velocity[k][I] -= scale[k] * (pressure[I] - pressure[I_1])
        elif ti.static(GlobalVariable.DIMENSION == 3):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]), (0, face_count[2]))):
                I_1 = I - ti.Vector.unit(GlobalVariable.DIMENSION, k)
                cell_1 = mac_get_offset_cell_type(I_1, ghost_cell, cnum, cell_type)
                cell_2 = mac_get_offset_cell_type(I, ghost_cell, cnum, cell_type)
                if cell_1 == 1 or cell_2 == 1:
                    if cell_1 == 2 or cell_2 == 2:
                        node.velocity[k][I] = 0
                        node.force[k][I] = 0
                    elif cell_2 == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I_1, I, fluid_sdf)
                        node.velocity[k][I] -= (
                            scale[k] / theta * (atmospheric_pressure + surface_tension[I] - pressure[I_1])
                        )
                    elif cell_1 == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I, I_1, fluid_sdf)
                        node.velocity[k][I] -= (
                            scale[k] / theta * (pressure[I] - atmospheric_pressure - surface_tension[I_1])
                        )
                    else:
                        node.velocity[k][I] -= scale[k] * (pressure[I] - pressure[I_1])


@ti.func
def correct_velocity_cut_cell_face(
    k: ti.template(),
    I,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    scale,
    atmospheric_pressure: float,
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    solid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    use_free_surface_theta: ti.template(),
    node: ti.template(),
):
    I_1 = I - ti.Vector.unit(GlobalVariable.DIMENSION, k)
    cell_1 = mac_get_offset_cell_type(I_1, ghost_cell, cnum, cell_type)
    cell_2 = mac_get_offset_cell_type(I, ghost_cell, cnum, cell_type)
    if cell_1 == 1 or cell_2 == 1:
        face_fraction = select_solid_face_fraction(k, I, face_fraction0, face_fraction1, face_fraction2)
        immersed_1 = cell_1 == 0 and solid_sdf[I_1] < 0.0
        immersed_2 = cell_2 == 0 and solid_sdf[I] < 0.0
        if face_fraction <= Threshold or cell_1 == 2 or cell_2 == 2:
            node.velocity[k][I] = select_solid_face_velocity(k, I, solid_velocity0, solid_velocity1, solid_velocity2)
            node.force[k][I] = 0
        elif immersed_1 or immersed_2:
            wall_velocity = select_solid_face_velocity(k, I, solid_velocity0, solid_velocity1, solid_velocity2)
            node.velocity[k][I] = face_fraction * node.velocity[k][I] + (1.0 - face_fraction) * wall_velocity
        elif cell_2 == 0:
            theta = 1.0
            if ti.static(use_free_surface_theta):
                theta = free_surface_theta(I_1, I, fluid_sdf)
            node.velocity[k][I] -= scale[k] / theta * (atmospheric_pressure + surface_tension[I] - pressure[I_1])
        elif cell_1 == 0:
            theta = 1.0
            if ti.static(use_free_surface_theta):
                theta = free_surface_theta(I, I_1, fluid_sdf)
            node.velocity[k][I] -= scale[k] / theta * (pressure[I] - atmospheric_pressure - surface_tension[I_1])
        else:
            node.velocity[k][I] -= scale[k] * (pressure[I] - pressure[I_1])


@ti.kernel
def kernel_update_pressure_and_correct_velocity_from_mg_cut_cell(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    matProps: ti.template(),
    x: ti.template(),
    grid_type: ti.template(),
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    solid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    use_free_surface_theta: ti.template(),
    node: ti.template(),
):
    pressure.fill(0)
    for I in ti.grouped(grid_type):
        if grid_type[I] == 1:
            pressure[I] = x[I]

    active_cnum = cnum - 2 * ghost_cell
    scale = dt[None] / (matProps.density * grid_size)
    atmospheric_pressure = matProps.atmospheric_pressure
    for k in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, k)
        if ti.static(GlobalVariable.DIMENSION == 2):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]))):
                correct_velocity_cut_cell_face(
                    k,
                    I,
                    ghost_cell,
                    cnum,
                    scale,
                    atmospheric_pressure,
                    pressure,
                    surface_tension,
                    cell_type,
                    fluid_sdf,
                    solid_sdf,
                    face_fraction0,
                    face_fraction1,
                    face_fraction2,
                    solid_velocity0,
                    solid_velocity1,
                    solid_velocity2,
                    use_free_surface_theta,
                    node,
                )
        elif ti.static(GlobalVariable.DIMENSION == 3):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]), (0, face_count[2]))):
                correct_velocity_cut_cell_face(
                    k,
                    I,
                    ghost_cell,
                    cnum,
                    scale,
                    atmospheric_pressure,
                    pressure,
                    surface_tension,
                    cell_type,
                    fluid_sdf,
                    solid_sdf,
                    face_fraction0,
                    face_fraction1,
                    face_fraction2,
                    solid_velocity0,
                    solid_velocity1,
                    solid_velocity2,
                    use_free_surface_theta,
                    node,
                )


@ti.kernel
def kernel_correct_velocity(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    matProps: ti.template(),
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
    node: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    scale = dt[None] / (matProps.density * grid_size)
    atmospheric_pressure = matProps.atmospheric_pressure
    for k in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, k)
        if ti.static(GlobalVariable.DIMENSION == 2):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]))):
                I_1 = I - ti.Vector.unit(GlobalVariable.DIMENSION, k)
                cell_1 = mac_get_offset_cell_type(I_1, ghost_cell, cnum, cell_type)
                cell_2 = mac_get_offset_cell_type(I, ghost_cell, cnum, cell_type)
                if cell_1 == 1 or cell_2 == 1:
                    if cell_1 == 2 or cell_2 == 2:
                        node.velocity[k][I] = 0
                        node.force[k][I] = 0
                    elif cell_2 == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I_1, I, fluid_sdf)
                        node.velocity[k][I] -= (
                            scale[k] / theta * (atmospheric_pressure + surface_tension[I] - pressure[I_1])
                        )
                    elif cell_1 == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I, I_1, fluid_sdf)
                        node.velocity[k][I] -= (
                            scale[k] / theta * (pressure[I] - atmospheric_pressure - surface_tension[I_1])
                        )
                    else:
                        node.velocity[k][I] -= scale[k] * (pressure[I] - pressure[I_1])
        elif ti.static(GlobalVariable.DIMENSION == 3):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]), (0, face_count[2]))):
                I_1 = I - ti.Vector.unit(GlobalVariable.DIMENSION, k)
                cell_1 = mac_get_offset_cell_type(I_1, ghost_cell, cnum, cell_type)
                cell_2 = mac_get_offset_cell_type(I, ghost_cell, cnum, cell_type)
                if cell_1 == 1 or cell_2 == 1:
                    if cell_1 == 2 or cell_2 == 2:
                        node.velocity[k][I] = 0
                        node.force[k][I] = 0
                    elif cell_2 == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I_1, I, fluid_sdf)
                        node.velocity[k][I] -= (
                            scale[k] / theta * (atmospheric_pressure + surface_tension[I] - pressure[I_1])
                        )
                    elif cell_1 == 0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = free_surface_theta(I, I_1, fluid_sdf)
                        node.velocity[k][I] -= (
                            scale[k] / theta * (pressure[I] - atmospheric_pressure - surface_tension[I_1])
                        )
                    else:
                        node.velocity[k][I] -= scale[k] * (pressure[I] - pressure[I_1])


@ti.kernel
def kernel_correct_velocity_cut_cell(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    matProps: ti.template(),
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    solid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    use_free_surface_theta: ti.template(),
    node: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    scale = dt[None] / (matProps.density * grid_size)
    atmospheric_pressure = matProps.atmospheric_pressure
    for k in ti.static(range(GlobalVariable.DIMENSION)):
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, k)
        if ti.static(GlobalVariable.DIMENSION == 2):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]))):
                correct_velocity_cut_cell_face(
                    k,
                    I,
                    ghost_cell,
                    cnum,
                    scale,
                    atmospheric_pressure,
                    pressure,
                    surface_tension,
                    cell_type,
                    fluid_sdf,
                    solid_sdf,
                    face_fraction0,
                    face_fraction1,
                    face_fraction2,
                    solid_velocity0,
                    solid_velocity1,
                    solid_velocity2,
                    use_free_surface_theta,
                    node,
                )
        elif ti.static(GlobalVariable.DIMENSION == 3):
            for I in ti.grouped(ti.ndrange((0, face_count[0]), (0, face_count[1]), (0, face_count[2]))):
                correct_velocity_cut_cell_face(
                    k,
                    I,
                    ghost_cell,
                    cnum,
                    scale,
                    atmospheric_pressure,
                    pressure,
                    surface_tension,
                    cell_type,
                    fluid_sdf,
                    solid_sdf,
                    face_fraction0,
                    face_fraction1,
                    face_fraction2,
                    solid_velocity0,
                    solid_velocity1,
                    solid_velocity2,
                    use_free_surface_theta,
                    node,
                )


@ti.kernel
def enforce_particle_domain_collision(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    cell_type: ti.template(),
    particle: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            pos = particle[np].x
            pv = particle[np].v
            collision_vec = ti.Vector.zero(float, GlobalVariable.DIMENSION)
            if not particle_position_is_valid(pos):
                particle[np].active = ti.u8(0)

            for d in ti.static(range(GlobalVariable.DIMENSION)):
                if int(particle[np].active) == 1:
                    lower_cell = ti.floor(pos / grid_size, int)
                    upper_cell = lower_cell
                    lower_cell[d] = -ghost_cell
                    upper_cell[d] = active_cnum[d]
                    lower_limit = 0.0
                    upper_limit = active_cnum[d] * grid_size[d]

                    if is_solid(get_offset_cell_type(lower_cell, ghost_cell, cnum, cell_type)) and pos[d] < lower_limit:
                        collision_vec[d] -= 1.0
                        pos[d] = lower_limit
                    elif (
                        is_solid(get_offset_cell_type(upper_cell, ghost_cell, cnum, cell_type)) and pos[d] > upper_limit
                    ):
                        collision_vec[d] += 1.0
                        pos[d] = upper_limit

            if int(particle[np].active) == 1:
                cell = ti.floor(pos / grid_size, int)
                cell = ti.min(ti.max(cell, 0), active_cnum - 1)
                if get_offset_cell_type(cell, ghost_cell, cnum, cell_type) == 2:
                    found_open_side = False
                    best_distance = 1e30
                    best_position = 0.0
                    best_normal = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        for s in ti.static((-1, 1)):
                            searching = True
                            for step in range(1, cnum[d]):
                                if searching:
                                    probe = cell
                                    probe[d] += s * step
                                    if get_offset_cell_type(probe, ghost_cell, cnum, cell_type) != 2:
                                        face_position = 0.0
                                        normal = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                                        if ti.static(s < 0):
                                            face_position = (cell[d] - step + 1) * grid_size[d]
                                            normal[d] = 1.0
                                        else:
                                            face_position = (cell[d] + step) * grid_size[d]
                                            normal[d] = -1.0
                                        distance = ti.abs(pos[d] - face_position)
                                        if distance < best_distance:
                                            found_open_side = True
                                            best_distance = distance
                                            best_position = face_position + s * 1e-6 * grid_size[d]
                                            best_normal = normal
                                        searching = False

                    if found_open_side:
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            if ti.abs(best_normal[d]) > 0.0:
                                pos[d] = best_position
                        collision_vec += best_normal
                    else:
                        particle[np].active = ti.u8(0)

            if int(particle[np].active) == 1:
                collision_norm = collision_vec.norm()
                if collision_norm > 1e-6:
                    normal = collision_vec / collision_norm
                    normal_velocity = pv.dot(normal)
                    if normal_velocity > 0.0:
                        normal_component = normal_velocity * normal
                        pv -= normal_component

                particle[np].x = pos
                particle[np].v = pv


@ti.func
def resolve_solid_normal_velocity(particle_velocity, wall_velocity, normal):
    relative_normal_velocity = (particle_velocity - wall_velocity).dot(normal)
    if relative_normal_velocity < 0.0:
        particle_velocity -= relative_normal_velocity * normal
    return particle_velocity


@ti.func
def nearest_solid_face_velocity(
    position,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    velocity = ti.Vector.zero(float, GlobalVariable.DIMENSION)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        face = ti.floor(position / grid_size).cast(int)
        face[d] = ti.floor(position[d] / grid_size[d] + 0.5, int)
        face_count = active_cnum + ti.Vector.unit(GlobalVariable.DIMENSION, d)
        for axis in ti.static(range(GlobalVariable.DIMENSION)):
            face[axis] = ti.min(face_count[axis] - 1, ti.max(0, face[axis]))
        velocity[d] = select_solid_face_velocity(d, face, solid_velocity0, solid_velocity1, solid_velocity2)
    return velocity


@ti.kernel
def enforce_particle_solid_sdf_collision(
    particleNum: int,
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    solid_sdf: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    particle: ti.template(),
):
    clearance = 1.0e-4 * min_grid_spacing(grid_size)
    active_cnum = cnum - 2 * ghost_cell
    domain_upper = active_cnum.cast(float) * grid_size
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            pos = particle[np].x
            pv = particle[np].v
            inside_domain = particle_position_is_valid(pos)
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                inside_domain = inside_domain and pos[d] >= 0.0 and pos[d] <= domain_upper[d]
            if inside_domain:
                phi = interpolate_solid_sdf(pos, ghost_cell, cnum, grid_size, solid_sdf)
                if phi < clearance:
                    normal = solid_sdf_position_normal(pos, ghost_cell, cnum, grid_size, solid_sdf)
                    if normal.norm() > Threshold:
                        pos += (clearance - phi) * normal
                        for d in ti.static(range(GlobalVariable.DIMENSION)):
                            pos[d] = ti.min(ti.max(pos[d], 0.0), domain_upper[d])
                        wall_velocity = nearest_solid_face_velocity(
                            pos,
                            ghost_cell,
                            cnum,
                            grid_size,
                            solid_velocity0,
                            solid_velocity1,
                            solid_velocity2,
                        )
                        pv = resolve_solid_normal_velocity(pv, wall_velocity, normal)
                        particle[np].x = pos
                        particle[np].v = pv


@ti.kernel
def kernel_update_cell_pressure(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    pressure: ti.template(),
    flag: ti.template(),
    cell_type: ti.template(),
    unknown_vector: ti.template(),
):
    pressure.fill(0)
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, cnum[0] - 2 * ghost_cell), (0, cnum[1] - 2 * ghost_cell))):
            if cell_type[I] == 1:
                cell_id = linearize(I, cnum - 2 * ghost_cell)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    pressure[I] = unknown_vector[dof_index]
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(
            ti.ndrange((0, cnum[0] - 2 * ghost_cell), (0, cnum[1] - 2 * ghost_cell), (0, cnum[2] - 2 * ghost_cell))
        ):
            if cell_type[I] == 1:
                cell_id = linearize(I, cnum - 2 * ghost_cell)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    pressure[I] = unknown_vector[dof_index]


# ---- SemiImplicit TwoPhaseSingleLayer compatibility kernels ----
@ti.kernel
def cell_volume_reset(cell_volumefrac: ti.template()):
    for nc in cell_volumefrac:
        cell_volumefrac[nc] = 0.0


@ti.kernel
def kernel_mass_p2g_twophase(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mass = particle[np].m
            mass_s = particle[np].ms
            mass_f = particle[np].mf
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                nmass_s = shape_mapping(shapefn[ln], mass_s)
                nmass_f = shape_mapping(shapefn[ln], mass_f)
                node[nodeID, bodyID]._update_nodal_mass(nmass, nmass_s, nmass_f)


@ti.kernel
def kernel_mass_momentum_p2g_semitwophase(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    istep: int,
    is_rigid: ti.template(),
):
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
            pressure = particle[np].pressure
            if is_rigid[bodyID] == 1:
                if istep < 1000:
                    velocity = particle[np].v * istep / 1000
                    velocity_s = velocity
                else:
                    velocity = particle[np].v
                    velocity_s = velocity
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                nmass_s = shape_mapping(shapefn[ln], mass_s)
                nmass_f = shape_mapping(shapefn[ln], mass_f)
                node[nodeID, bodyID]._update_nodal_mass(nmass, nmass_s, nmass_f)
                node[nodeID, bodyID]._update_nodal_pressure_(nmass_s * pressure)
                node[nodeID, bodyID]._update_nodal_momentum(
                    nmass * velocity, nmass_s * velocity_s, nmass_f * velocity_f
                )


@ti.kernel
def kernel_mass_momentum_p2g_twophase_APIC(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    istep: int,
    is_rigid: ti.template(),
    grid_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
):
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
            pressure = particle[np].pressure
            xp = particle[np].x
            gradv = particle[np].solid_velocity_gradient
            grdv_f = particle[np].fluid_velocity_gradient
            # -- Linear loading--
            if is_rigid[bodyID] == 1:
                if istep < 1000:
                    velocity = particle[np].v * istep / 1000
                    velocity_s = velocity
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nodal_coord = grid_size * vec2f(vectorize_id(nodeID, gnum))
                xip = nodal_coord - xp
                nmass = shape_mapping(shapefn[ln], mass)
                nmass_s = shape_mapping(shapefn[ln], mass_s)
                nmass_f = shape_mapping(shapefn[ln], mass_f)
                node[nodeID, bodyID]._update_nodal_mass(nmass, nmass_s, nmass_f)
                node[nodeID, bodyID]._update_nodal_pressure_(nmass_s * pressure)
                node[nodeID, bodyID]._update_nodal_momentum(
                    nmass * (velocity + gradv @ xip),
                    nmass_s * (velocity_s + gradv @ xip),
                    nmass_f * (velocity_f + grdv_f @ xip),
                )


@ti.kernel
def kernel_mass_momentum_p2g_twophase_u_p(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    step: int,
    is_rigid: ti.template(),
):
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
            pressure = particle[np].pressure
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                nmass = shape_mapping(shapefn[ln], mass)
                nmass_s = shape_mapping(shapefn[ln], mass_s)
                nmass_f = shape_mapping(shapefn[ln], mass_f)
                node[nodeID, bodyID]._update_nodal_mass(nmass, nmass_s, nmass_f)
                node[nodeID, bodyID]._update_nodal_pressure_(nmass_s * pressure)
                node[nodeID, bodyID]._update_nodal_momentum(
                    nmass * velocity, nmass_s * velocity_s, nmass_f * velocity_f
                )


@ti.kernel
def kernel_pressure_tpic_p2g_correction_2D(
    total_nodes: int,
    particleNum: int,
    grid_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                node_position = grid_size * vec2f(vectorize_id(nodeID, gnum))
                correction = particle[np].pressure_gradient.dot(node_position - particle[np].x)
                node[nodeID, bodyID]._update_nodal_pressure_(shapefn[ln] * particle[np].ms * correction)


@ti.kernel
def kernel_update_particle_pressure_gradient_2D(
    total_nodes: int,
    particleNum: int,
    beta: float,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            gradient = ZEROVEC2f
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                pressure = beta * node[nodeID, bodyID].pressure + node[nodeID, bodyID].dpressure
                gradient += dshapefn[ln] * pressure
            particle[np].pressure_gradient = gradient


@ti.func
def single_point_nodal_pressure_gradient_2D(offset, size, bodyID, node, LnID, dshapefn):
    gradient = ZEROVEC2f
    for ln in range(offset, offset + size):
        gradient += dshapefn[ln] * node[LnID[ln], bodyID].pressure
    return gradient


@ti.kernel
def kernel_force_p2g_semitwophase2D(
    total_nodes: int,
    particleNum: int,
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    beta: float,
    nodal_pressure_gradient: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex, fexf = particle[np]._compute_external_force(gravity)
            fin, fintf = particle[np]._compute_internal_force_semi(beta)
            drag = particle[np]._compute_drag_force_semi()
            offset = np * total_nodes
            pressure_force = ZEROVEC2f
            if ti.static(nodal_pressure_gradient):
                # The FEM predictor and pressure-increment correction must use
                # the same gradient, including on partially filled cells.
                fin, fintf = particle[np]._compute_internal_force_semi(0.0)
                pressure_force = (
                    -beta
                    * particle[np].vol
                    * single_point_nodal_pressure_gradient_2D(offset, int(node_size[np]), bodyID, node, LnID, dshapefn)
                )
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shape_fn, fex)
                external_forcef = shape_mapping(shape_fn, fexf)
                drag_ = shape_mapping(shape_fn, drag)
                internal_force = vec2f(
                    [dshape_fn[0] * fin[0] + dshape_fn[1] * fin[3], dshape_fn[1] * fin[1] + dshape_fn[0] * fin[3]]
                )
                internal_forcef = vec2f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1]])
                if ti.static(nodal_pressure_gradient):
                    internal_force += shape_fn * pressure_force
                    internal_forcef += shape_fn * particle[np].porosity * pressure_force
                node[nodeID, bodyID]._update_nodal_dragval(drag_)
                node[nodeID, bodyID]._update_nodal_force(
                    external_force + internal_force, external_forcef + internal_forcef
                )


@ti.kernel
def kernel_force_p2g_semitwophase3D(
    total_nodes: int,
    particleNum: int,
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    beta: float,
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex, fexf = particle[np]._compute_external_force(gravity)
            fin, fintf = particle[np]._compute_internal_force_semi(beta)
            drag = particle[np]._compute_drag_force_semi()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shape_fn, fex)
                external_forcef = shape_mapping(shape_fn, fexf)
                drag_ = shape_mapping(shape_fn, drag)
                internal_force = vec3f(
                    [
                        dshape_fn[0] * fin[0] + dshape_fn[1] * fin[3] + dshape_fn[2] * fin[5],
                        dshape_fn[1] * fin[1] + dshape_fn[0] * fin[3] + dshape_fn[2] * fin[4],
                        dshape_fn[2] * fin[2] + dshape_fn[1] * fin[4] + dshape_fn[0] * fin[5],
                    ]
                )
                internal_forcef = vec3f([dshape_fn[0] * fintf[0], dshape_fn[1] * fintf[1], dshape_fn[2] * fintf[2]])
                node[nodeID, bodyID]._update_nodal_dragval(drag_)
                node[nodeID, bodyID]._update_nodal_force(
                    external_force + internal_force, external_forcef + internal_forcef
                )


@ti.kernel
def kernel_force_p2g_semitwophase2D_u_p(
    total_nodes: int,
    particleNum: int,
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    beta: float,
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            fex, fexf = particle[np]._compute_external_force(gravity)
            fin, fintf = particle[np]._compute_internal_force_semi(0.0)
            offset = np * total_nodes
            pressure_force = (
                -beta
                * particle[np].vol
                * single_point_nodal_pressure_gradient_2D(offset, int(node_size[np]), bodyID, node, LnID, dshapefn)
            )
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shape_fn, fex)
                internal_force = vec2f(
                    [dshape_fn[0] * fin[0] + dshape_fn[1] * fin[3], dshape_fn[1] * fin[1] + dshape_fn[0] * fin[3]]
                )
                node[nodeID, bodyID].force += external_force + internal_force + shape_fn * pressure_force


@ti.kernel
def kernel_force_p2g_semitwophase_2DAxisy(
    total_nodes: int,
    particleNum: int,
    gravity: ti.types.vector(3, float),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    beta: float,
    axis_offset: float,
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            position = particle[np].x - vec2f([axis_offset, 0.0])
            fex, fexf = particle[np]._compute_external_force(gravity)
            fint, fintf = particle[np]._compute_internal_force_semi(beta)
            drag = particle[np]._compute_drag_force_semi()
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                dshape_fn = dshapefn[ln]
                external_force = shape_mapping(shape_fn, fex)
                external_forcef = shape_mapping(shape_fn, fexf)
                drag_ = shape_mapping(shape_fn, drag)
                internal_force = vec2f(
                    [
                        dshape_fn[0] * fint[0] + dshape_fn[1] * fint[3] + fint[2] * shape_fn / position[0],
                        dshape_fn[1] * fint[1] + dshape_fn[0] * fint[3],
                    ]
                )
                internal_forcef = vec2f(
                    [dshape_fn[0] * fintf[0] + fintf[2] * shape_fn / position[0], dshape_fn[1] * fintf[1]]
                )
                node[nodeID, bodyID]._update_nodal_dragval(drag_)
                node[nodeID, bodyID]._update_nodal_force(
                    external_force + internal_force, external_forcef + internal_forcef
                )


@ti.kernel
def kernel_single_point_porosity_p2g(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    # Reuse nodal scratch fields; do not overwrite the old pressure predictor.
    node.porosity.fill(0.0)
    node.weight.fill(0.0)
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                volume = shapefn[ln] * particle[np].vol
                node[nodeID, bodyID].weight += volume
                node[nodeID, bodyID].porosity += volume * particle[np].porosity
    for ng, nb in node:
        if node[ng, nb].weight > 0.0:
            node[ng, nb].porosity /= node[ng, nb].weight


@ti.kernel
def kernel_pressure_p2g_twophase_2D(
    particleNum: int,
    extra_node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    total_nodes: int,
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            scale = particle[np].m * MeanStress(particle[np].stress)  # particle[np].porosity / particle[np].pressure
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                extra_node[nodeID, bodyID]._update_nodal_pressure(shape_mapping(shapefn[ln], scale))


@ti.kernel
def kernel_porosity_p2g(
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    total_nodes: int,
    node_size: ti.template(),
):
    node.pressure.fill(0.0)
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            poro = particle[np].m * particle[np].porosity
            pressure = particle[np].m * particle[np].pressure
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                node[nodeID, bodyID].porosity += shape_mapping(shapefn[ln], poro)
                node[nodeID, bodyID].pressure += shape_mapping(shapefn[ln], pressure)


@ti.kernel
def kernel_assemble_bc_map(
    node: ti.template(),
    element_size: ti.types.vector(2, float),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
    gnum: ti.types.vector(2, int),
    right_hand_vector: ti.template(),
    cell_type: ti.template(),
    cell_porosity: ti.template(),
    is_rigid: ti.template(),
):
    bodyID = 0 if is_rigid[0] == 0 else 1
    right_hand_vector.fill(0)
    for I in ti.grouped(cell_type):
        if cell_type[I] == 1:
            i, j = I
            position = vec2f(i + 0.5, j + 0.5) * element_size
            psize = vec2f(0, 0)
            base_bound = ti.floor((position - psize) / element_size, int)
            influenced_node = 2
            LnID = vec4i([0, 0, 0, 0])
            shape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            dxshape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            dyshape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            localID = 0
            for lj in range(base_bound[1], base_bound[1] + influenced_node):
                if lj < 0 or lj >= gnum[1]:
                    continue
                for li in range(base_bound[0], base_bound[0] + influenced_node):
                    if li < 0 or li >= gnum[0]:
                        continue
                    nodeID = int(li + lj * gnum[0])
                    node_coords = vec2i(li, lj) * element_size
                    shapen0 = shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                    shapen1 = shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                    shapeval = shapen0 * shapen1
                    if shapeval > Threshold:
                        dshapen0 = grad_shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                        dshapen1 = grad_shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                        grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                        LnID[localID] = nodeID
                        shape_fn[localID] = shapeval
                        dxshape_fn[localID] = grad_shapeval[0]
                        dyshape_fn[localID] = grad_shapeval[1]
                        localID += 1
            porosity = cell_porosity[I]
            total_nodes = influenced_node * influenced_node
            B1, B2 = 0.0, 0.0
            jdshape = vec2f([0.0, 0.0])
            for lnj in range(total_nodes):
                nodeIDj = LnID[lnj]
                jshape = shape_fn[lnj]
                jdshape[0] = dxshape_fn[lnj]
                jdshape[1] = dyshape_fn[lnj]
                B1 += (
                    jdshape[0] * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[0]
                    + jdshape[1] * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[1]
                )
                B2 += (
                    jdshape[0] * porosity * node[nodeIDj, bodyID].momentumf[0]
                    + jdshape[1] * porosity * node[nodeIDj, bodyID].momentumf[1]
                )
            right_hand_vector[I] += B1 + B2


@ti.kernel
def kernel_assemble_bc_map_3D(
    node: ti.template(),
    element_size: ti.types.vector(3, float),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
    gnum: ti.types.vector(3, int),
    right_hand_vector: ti.template(),
    cell_type: ti.template(),
    cell_porosity: ti.template(),
    is_rigid: ti.template(),
):
    bodyID = 0 if is_rigid[0] == 0 else 1
    right_hand_vector.fill(0)
    for I in ti.grouped(cell_type):
        if cell_type[I] == 1:
            i, j, k = I
            position = vec3f(i + 0.5, j + 0.5, k + 0.5) * element_size
            psize = vec3f(0, 0, 0)
            base_bound = ti.floor((position - psize) / element_size, int)
            influenced_node = 2
            LnID = vec8i([0, 0, 0, 0, 0, 0, 0, 0])
            shape_fn = vec8f([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            dxshape_fn = vec8f([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            dyshape_fn = vec8f([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            dzshape_fn = vec8f([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            localID = 0
            for lk in range(base_bound[2], base_bound[2] + influenced_node):
                if lk < 0 or lk >= gnum[2]:
                    continue
                for lj in range(base_bound[1], base_bound[1] + influenced_node):
                    if lj < 0 or lj >= gnum[1]:
                        continue
                    for li in range(base_bound[0], base_bound[0] + influenced_node):
                        if li < 0 or li >= gnum[0]:
                            continue
                        nodeID = linearize(vec3i(li, lj, lk), gnum)
                        node_coords = vec3f(li, lj, lk) * element_size
                        shapen0 = shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                        shapen1 = shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                        shapen2 = shape_function(position[2], node_coords[2], 1.0 / element_size[2], psize[2])
                        shapeval = shapen0 * shapen1 * shapen2
                        if shapeval > Threshold:
                            dshapen0 = grad_shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                            dshapen1 = grad_shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                            dshapen2 = grad_shape_function(position[2], node_coords[2], 1.0 / element_size[2], psize[2])
                            LnID[localID] = nodeID
                            shape_fn[localID] = shapeval
                            dxshape_fn[localID] = dshapen0 * shapen1 * shapen2
                            dyshape_fn[localID] = shapen0 * dshapen1 * shapen2
                            dzshape_fn[localID] = shapen0 * shapen1 * dshapen2
                            localID += 1
            porosity = cell_porosity[I]
            B1, B2 = 0.0, 0.0
            total_nodes = influenced_node * influenced_node * influenced_node
            jdshape = vec3f([0.0, 0.0, 0.0])
            for lnj in range(total_nodes):
                nodeIDj = LnID[lnj]
                jdshape[0] = dxshape_fn[lnj]
                jdshape[1] = dyshape_fn[lnj]
                jdshape[2] = dzshape_fn[lnj]
                B1 += (
                    jdshape[0] * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[0]
                    + jdshape[1] * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[1]
                    + jdshape[2] * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[2]
                )
                B2 += (
                    jdshape[0] * porosity * node[nodeIDj, bodyID].momentumf[0]
                    + jdshape[1] * porosity * node[nodeIDj, bodyID].momentumf[1]
                    + jdshape[2] * porosity * node[nodeIDj, bodyID].momentumf[2]
                )
            right_hand_vector[I] += B1 + B2


@ti.kernel
def kernel_assemble_bc_2DAxisy_map_(
    node: ti.template(),
    element_size: ti.types.vector(2, float),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
    gnum: ti.types.vector(2, int),
    right_hand_vector: ti.template(),
    cell_type: ti.template(),
    cell_porosity: ti.template(),
    is_rigid: ti.template(),
    axis_offset: float,
):
    bodyID = 0 if is_rigid[0] == 0 else 1
    right_hand_vector.fill(0)
    for I in ti.grouped(cell_type):
        if cell_type[I] == 1:
            i, j = I
            position = vec2f(i + 0.5, j + 0.5) * element_size
            psize = vec2f(0, 0)
            base_bound = ti.floor((position - psize) / element_size, int)
            influenced_node = 2
            LnID = vec4i([0, 0, 0, 0])
            shape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            dxshape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            dyshape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            localID = 0
            for lj in range(base_bound[1], base_bound[1] + influenced_node):
                if lj < 0 or lj >= gnum[1]:
                    continue
                for li in range(base_bound[0], base_bound[0] + influenced_node):
                    if li < 0 or li >= gnum[0]:
                        continue
                    nodeID = int(li + lj * gnum[0])
                    node_coords = vec2i(li, lj) * element_size
                    shapen0 = shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                    shapen1 = shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                    shapeval = shapen0 * shapen1
                    if shapeval > Threshold:
                        dshapen0 = grad_shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                        dshapen1 = grad_shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                        grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                        LnID[localID] = nodeID
                        shape_fn[localID] = shapeval
                        dxshape_fn[localID] = grad_shapeval[0]
                        dyshape_fn[localID] = grad_shapeval[1]
                        localID += 1
            porosity = cell_porosity[I]
            total_nodes = influenced_node * influenced_node
            B1, B2 = 0.0, 0.0
            jdshape = vec2f([0.0, 0.0])
            rc = position[0] - axis_offset
            rc_ = (position[0] - axis_offset) / element_size[0]
            for lnj in range(total_nodes):
                nodeIDj = LnID[lnj]
                jshape = shape_fn[lnj]
                jdshape[0] = dxshape_fn[lnj]
                jdshape[1] = dyshape_fn[lnj]
                B1 += (
                    jdshape[0] * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[0]
                    + jdshape[1] * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[1]
                    + jshape / rc * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[0]
                )
                B2 += (
                    jdshape[0] * porosity * node[nodeIDj, bodyID].momentumf[0]
                    + jdshape[1] * porosity * node[nodeIDj, bodyID].momentumf[1]
                    + jshape / rc * porosity * node[nodeIDj, bodyID].momentumf[0]
                )
            right_hand_vector[I] += (B1 + B2) * rc_


@ti.kernel
def kernel_assemble_FIC_bc_map(
    matProp: ti.template(),
    node: ti.template(),
    element_size: ti.types.vector(2, float),
    shape_function: ti.template(),
    grad_shape_function: ti.template(),
    gnum: ti.types.vector(2, int),
    right_hand_vector: ti.template(),
    cell_type: ti.template(),
    cell_porosity: ti.template(),
):
    right_hand_vector.fill(0)
    density_s, density_f = matProp.solid_density, matProp.fluid_density
    young, poisson = matProp.young, matProp.poisson
    Gmod = young / (1.0 + poisson) / 2.0
    Kmod = young / (1.0 - 2.0 * poisson) / 3.0
    tau_ = element_size[0] / ti.sqrt((Kmod + 4.0 * Gmod / 3.0) / density_s) * 10.0
    # tau_ = 0.0

    for I in ti.grouped(cell_type):
        if cell_type[I] == 1:
            i, j = I
            position = vec2f(i + 0.5, j + 0.5) * element_size
            psize = vec2f(0, 0)
            base_bound = ti.floor((position - psize) / element_size, int)
            influenced_node = 2
            LnID = vec4i([0, 0, 0, 0])
            shape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            dxshape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            dyshape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            localID = 0
            for lj in range(base_bound[1], base_bound[1] + influenced_node):
                if lj < 0 or lj >= gnum[1]:
                    continue
                for li in range(base_bound[0], base_bound[0] + influenced_node):
                    if li < 0 or li >= gnum[0]:
                        continue
                    nodeID = int(li + lj * gnum[0])
                    node_coords = vec2i(li, lj) * element_size
                    shapen0 = shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                    shapen1 = shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                    shapeval = shapen0 * shapen1
                    if shapeval > Threshold:
                        dshapen0 = grad_shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                        dshapen1 = grad_shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                        grad_shapeval = vec2f([dshapen0 * shapen1, shapen0 * dshapen1])
                        LnID[localID] = nodeID
                        shape_fn[localID] = shapeval
                        dxshape_fn[localID] = grad_shapeval[0]
                        dyshape_fn[localID] = grad_shapeval[1]
                        localID += 1
            bodyID = 0
            porosity = cell_porosity[I]
            H = tau_ * (porosity / density_f + (1.0 - porosity) / density_s)
            total_nodes = influenced_node * influenced_node
            B1, B2, B3, B4 = 0.0, 0.0, 0.0, 0.0
            jdshape = vec2f([0.0, 0.0])
            idshape = vec2f([0.0, 0.0])
            for lnj in range(total_nodes):
                nodeIDj = LnID[lnj]
                jshape = shape_fn[lnj]
                jdshape[0] = dxshape_fn[lnj]
                jdshape[1] = dyshape_fn[lnj]
                B1 += (
                    jdshape[0] * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[0]
                    + jdshape[1] * (1.0 - porosity) * node[nodeIDj, bodyID].momentums[1]
                )
                B2 += (
                    jdshape[0] * porosity * node[nodeIDj, bodyID].momentumf[0]
                    + jdshape[1] * porosity * node[nodeIDj, bodyID].momentumf[1]
                )
                for lni in range(total_nodes):
                    nodeIDi = LnID[lni]
                    ishape = shape_fn[lni]
                    idshape[0] = dxshape_fn[lni]
                    idshape[1] = dyshape_fn[lni]
                    B3 -= (
                        jdshape[0] * idshape[0] * H * node[nodeIDi, bodyID].pressure
                        + jdshape[1] * idshape[1] * H * node[nodeIDi, bodyID].pressure
                    )
                    B4 -= (
                        jdshape[0] * ishape * tau_ * node[nodeIDi, bodyID].extra_stabilize[0]
                        + jdshape[1] * ishape * tau_ * node[nodeIDi, bodyID].extra_stabilize[1]
                    )
            right_hand_vector[I] += B1 + B2 + B3 + B4


@ti.kernel
def kernel_assemble_A(
    level: int,
    dt: ti.template(),
    matProp: ti.template(),
    element_size: ti.template(),
    cell_porosity: ti.template(),
    cell_phi: ti.template(),
    grid_type: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
):
    dimension = ti.static(GlobalVariable.DIMENSION)
    density_s, density_f = matProp.solid_density, matProp.fluid_density
    cnum = ti.Vector.zero(int, dimension)
    for d in ti.static(range(dimension)):
        cnum[d] = grid_type.shape[d]

    for I in ti.grouped(grid_type):
        if grid_type[I] == 1:
            porosity = cell_porosity[I]
            mobility = (1.0 - porosity) / density_s + porosity / density_f
            for k in ti.static(range(dimension)):
                scale_A = dt[None] * mobility / (element_size[k] * element_size[k])
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(dimension, k) * s
                    neighbor = I + offset
                    # Outside the pressure array is a virtual solid ghost cell:
                    # it contributes a no-flux face and never consumes a physical cell.
                    if is_mg_valid_cell(neighbor, cnum):
                        if grid_type[neighbor] == 1:
                            Adiag[I] -= scale_A
                            if ti.static(s > 0):
                                Ax[I][k] = scale_A
                        elif grid_type[neighbor] == 0:
                            theta = 0.5
                            if level == 0:
                                theta = free_surface_theta(I, neighbor, cell_phi)
                            Adiag[I] -= scale_A / theta


@ti.kernel
def kernel_assemble_A_2DAxisy_multi(
    dt: ti.template(),
    matProp: ti.template(),
    element_size: ti.types.vector(2, float),
    cell_porosity: ti.template(),
    cell_phi: ti.template(),
    grid_type: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
    Ax_neg: ti.template(),
    axis_offset: float,
    level: int,
):
    dimension = ti.static(2)
    density_s, density_f = matProp.solid_density, matProp.fluid_density
    scale_A_ = dt[None] / (element_size[0] * element_size[1])
    factor = 2.0**level

    for I in ti.grouped(grid_type):
        if grid_type[I] == 1:
            porosity = cell_porosity[I]
            scale_A = scale_A_ * ((1.0 - porosity) / density_s + porosity / density_f)
            rc = float(I[0]) - axis_offset / (element_size[0] * factor * factor)
            for k in ti.static(range(dimension)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(dimension, k) * s
                    neighbor_type = get_mg_cell_type(I + offset, grid_type)
                    if neighbor_type == 1:
                        if k == 1:
                            if ti.static(s > 0):
                                Ax[I][k] = scale_A * (rc + 0.5) * factor
                                Adiag[I] -= scale_A * (rc + 0.5) * factor
                            else:
                                Ax_neg[I][k] = scale_A * (rc + 0.5) * factor
                                Adiag[I] -= scale_A * (rc + 0.5) * factor
                        elif k == 0:  # r-direction
                            if rc > 0.5:
                                if ti.static(s > 0):
                                    Ax[I][k] = scale_A * (rc + 1.0) * factor
                                    Adiag[I] -= scale_A * (rc + 1.0) * factor
                                else:
                                    Ax_neg[I][k] = scale_A * rc * factor
                                    Adiag[I] -= scale_A * rc * factor
                            else:
                                if ti.static(s > 0):
                                    Ax[I][k] = scale_A * (rc + 1.0) * factor
                                    Adiag[I] -= scale_A * (rc + 1.0) * factor
                                else:
                                    Ax_neg[I][k] = 0.0
                    elif neighbor_type == 0:
                        if k == 0:
                            if rc > 0.5:
                                if ti.static(s > 0):
                                    Adiag[I] -= scale_A * (rc + 1.0) * factor
                                else:
                                    Adiag[I] -= scale_A * rc * factor
                            else:
                                if ti.static(s > 0):
                                    Adiag[I] -= scale_A * (rc + 1.0) * factor
                        elif k == 1:
                            Adiag[I] -= scale_A * (rc + 0.5) * factor


@ti.kernel
def kernel_assemble_spare_A(
    grid_type: ti.template(), Adiag: ti.template(), Ax: ti.template(), Ax_: ti.template(), sparse_matrix: ti.template()
):
    infnode = 5
    # epsilon = 1e-14
    sparse_matrix.rows.fill(0)
    sparse_matrix.cols.fill(0)
    sparse_matrix.data.fill(0)
    ny, nx = grid_type.shape
    for i in range(ny):
        for j in range(nx):
            idx = j * ny + i
            if grid_type[i, j] == 1:
                # 对角线元素
                # data.append(Adiag[i, j])
                # row_ind.append(idx)
                # col_ind.append(idx)
                index = infnode * idx
                sparse_matrix.rows[index] = idx
                sparse_matrix.cols[index] = idx
                sparse_matrix.data[index] = Adiag[i, j]
                # if Adiag[i, j] < epsilon:
                #     sparse_matrix.data[index] += epsilon

                # x方向邻居
                if j < nx - 1 and grid_type[i, j + 1] == 1:
                    idx_right = (j + 1) * ny + i
                    # data.append(Ax[i, j, 1])
                    # row_ind.append(idx)
                    # col_ind.append(idx_right)
                    index = infnode * idx + 1
                    sparse_matrix.rows[index] = idx
                    sparse_matrix.cols[index] = idx_right
                    sparse_matrix.data[index] = Ax[i, j][1]
                if j > 0 and grid_type[i, j - 1] == 1:
                    idx_left = (j - 1) * ny + i
                    index = infnode * idx + 2
                    sparse_matrix.rows[index] = idx
                    sparse_matrix.cols[index] = idx_left
                    sparse_matrix.data[index] = Ax_[i, j][1]

                # y方向邻居
                if i < ny - 1 and grid_type[i + 1, j] == 1:
                    idx_up = j * ny + (i + 1)
                    index = infnode * idx + 3
                    sparse_matrix.rows[index] = idx
                    sparse_matrix.cols[index] = idx_up
                    sparse_matrix.data[index] = Ax[i, j][0]
                if i > 0 and grid_type[i - 1, j] == 1:
                    idx_down = j * ny + (i - 1)
                    index = infnode * idx + 4
                    sparse_matrix.rows[index] = idx
                    sparse_matrix.cols[index] = idx_down
                    sparse_matrix.data[index] = Ax_[i, j][0]


@ti.kernel
def kernel_assemble_A_FIC(
    dt: ti.template(),
    matProp: ti.template(),
    element_size: ti.types.vector(2, float),
    cell_porosity: ti.template(),
    cell_phi: ti.template(),
    grid_type: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
):
    dimension = ti.static(2)
    density_s, density_f = matProp.solid_density, matProp.fluid_density
    young, poisson = matProp.young, matProp.poisson
    Gmod = young / (1.0 + poisson) / 2.0
    Kmod = young / (1.0 - 2.0 * poisson) / 3.0
    tau_ = element_size[0] / ti.sqrt((Kmod + 4.0 * Gmod / 3.0) / density_s) * 10.0
    # tau_ = 0.0
    scale_A_ = (dt[None] + tau_) / (element_size[0] * element_size[1])

    for I in ti.grouped(grid_type):
        if grid_type[I] == 1:
            porosity = cell_porosity[I]
            scale_A = scale_A_ * ((1.0 - porosity) / density_s + porosity / density_f)
            for k in ti.static(range(dimension)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(dimension, k) * s
                    neighbor_type = get_mg_cell_type(I + offset, grid_type)
                    if neighbor_type == 1:
                        Adiag[I] -= scale_A
                        if ti.static(s > 0):
                            Ax[I][k] = scale_A
                    elif neighbor_type == 0:
                        Adiag[I] -= scale_A
                        phi_fluid = cell_phi[I]
                        phi_air = cell_phi[I + offset]
                        """theta = ti.abs(phi_fluid) / (ti.abs(phi_fluid) + ti.abs(phi_air))
                        theta = ti.max(0.001, ti.min(0.999, theta))                        
                        Adiag[I] += -scale_A - scale_A * (1.0 - theta) / theta    # 公式：-scale_A - (scale_A(1-θ)/θ)"""
                        """coeff_ratio  = ti.abs(phi_air) / ti.abs(phi_fluid)
                        coeff_ratio  = ti.max(0.001, ti.min(1000., coeff_ratio))
                        coeff_ratio  = phi_air / phi_fluid
                        # if I[0]==1:
                        #     print(I, coeff_ratio * scale_A)
                        Adiag[I] += -scale_A + coeff_ratio * scale_A"""


@ti.kernel  # MGPCG
def kernel_reset_cell_infor(
    cell_type: ti.template(),
    cell_volume: ti.template(),
    cell_phi: ti.template(),
    cell_porosity: ti.template(),
    cell_pressure: ti.template(),
    cell_dpressure: ti.template(),
):
    for I in ti.grouped(cell_type):
        cell_volume[I] = 0.0
        cell_phi[I] = 100.0
        cell_porosity[I] = 0.0
        cell_pressure[I] = 0.0
        cell_dpressure[I] = 0.0


@ti.kernel  # MGPCG
def kernel_update_cell_infor(
    istep: int,
    particleNum: int,
    particle: ti.template(),
    element_size: ti.template(),
    cell_volume: ti.template(),
    cell_porosity: ti.template(),
    cell_pressure: ti.template(),
):
    for np in range(particleNum):
        position = particle[np].x
        volume = particle[np].vol
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            cell_id = ti.floor((position) / element_size, int)
            valid = True
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                valid = valid and 0 <= cell_id[d] < cell_volume.shape[d]
            if valid:
                cell_volume[cell_id] += volume
            """cell_porosity[cell_id] += volume * porosity
            cell_pressure[cell_id] += volume * pressure
    for I in ti.grouped(cell_volume):
        if cell_volume[I] > Threshold:
            cell_porosity[I] /= cell_volume[I]
            cell_pressure[I] /= cell_volume[I]"""


@ti.kernel  # MGPCG
def kernel_update_cell_porosity(
    istep: int,
    cell_type: ti.template(),
    cell_porosity: ti.template(),
    cell_pressure: ti.template(),
    node: ti.template(),
    gnum: ti.template(),
    is_rigid: ti.template(),
):
    bodyID = 0 if is_rigid[0] == 0 else 1
    for I in ti.grouped(cell_type):
        if cell_type[I] == 1:
            poro = 0.0
            pressure = 0.0
            count = 0.0
            for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2),) * GlobalVariable.DIMENSION))):
                node_id = linearize(I + offset, gnum)
                poro += node[node_id, bodyID].porosity
                pressure += node[node_id, bodyID].pressure
                count += 1.0
            cell_porosity[I] = poro / count
            cell_pressure[I] = pressure / count


@ti.func
def is_mg_valid_cell(cell_id, cnum):
    valid = True
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        valid = valid and 0 <= cell_id[d] < cnum[d]
    return valid


@ti.func
def is_mg_fluid_cell(cell_id, cell_type):
    return int(cell_type[cell_id]) == 1


@ti.func
def is_mg_air_cell(cell_id, cell_type):
    return int(cell_type[cell_id]) == 0


@ti.func
def is_mg_solid_cell(cell_id, cell_type):
    return int(cell_type[cell_id]) == 2


@ti.kernel
def kernel_pre_update_phi(istep: int, cnum: ti.template(), cell_type: ti.template(), cell_free_surface: ti.template()):
    for I in ti.grouped(cell_free_surface):
        cell_free_surface[I] = ti.u8(0)
        for k in ti.static(range(GlobalVariable.DIMENSION)):
            for s in ti.static((-1, 1)):
                offset = ti.Vector.unit(GlobalVariable.DIMENSION, k) * s
                neigh = I + offset
                if (
                    is_mg_valid_cell(I, cnum)
                    and is_mg_fluid_cell(I, cell_type)
                    and is_mg_valid_cell(neigh, cnum)
                    and is_mg_air_cell(neigh, cell_type)
                ):
                    cell_free_surface[I] = ti.u8(1)
                elif (
                    is_mg_valid_cell(I, cnum)
                    and is_mg_air_cell(I, cell_type)
                    and is_mg_valid_cell(neigh, cnum)
                    and is_mg_fluid_cell(neigh, cell_type)
                ):
                    cell_free_surface[I] = ti.u8(1)


@ti.kernel
def kernel_compute_grid_kinematic_semitwophase(
    cutoff: float, damp: float, node: ti.template(), dt: ti.template(), nodal_coords: ti.template()
):
    delta_t = dt[None]
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].mf > cutoff and node[ng, nb].ms > cutoff:
                ms = node[ng, nb].ms
                mf = node[ng, nb].mf
                q = node[ng, nb].dragval
                accs = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                accf = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    rhs_total = node[ng, nb].force[d]
                    rhs_fluid = node[ng, nb].forcef[d] - q * (node[ng, nb].momentumf[d] - node[ng, nb].momentums[d])
                    det = ms * (mf + delta_t * q) + delta_t * q * mf
                    if ti.abs(det) > Threshold:
                        accs[d] = ((mf + delta_t * q) * rhs_total - mf * rhs_fluid) / det
                        accf[d] = (delta_t * q * rhs_total + ms * rhs_fluid) / det
                node[ng, nb]._compute_nodal_kinematic_semi(damp, dt, accs, accf)


@ti.kernel
def kernel_compute_grid_kinematic_semitwophase_(
    cutoff: float, damp: float, node: ti.template(), dt: ti.template(), nodal_coords: ti.template()
):
    delta_t = dt[None]
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].mf > cutoff and node[ng, nb].ms > cutoff:
                Ms = node[ng, nb].ms
                force = node[ng, nb].force
                accs = force / Ms
                accf = vec2f([0.0, 0.0])
                node[ng, nb]._compute_nodal_kinematic_semi(damp, dt, accs, accf)


@ti.kernel
def kernel_correct_grid_kinematic_semitwophase(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    dshapefn: ti.template(),
    shapefn: ti.template(),
    dt: ti.template(),
    cutoff: float,
):
    for np in range(particleNum):
        if int(particle[np].active) == 0:
            continue
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            ishape = shapefn[lni]
            Bs, Bf = ZEROVEC2f, ZEROVEC2f
            ms, mf = node[nodeIDi, bodyID].ms, node[nodeIDi, bodyID].mf
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                jdshape = dshapefn[lnj]
                Bs -= ishape * jdshape * node[nodeIDj, bodyID].dpressure * volume * (1.0 - porosity)
                Bf -= ishape * jdshape * node[nodeIDj, bodyID].dpressure * volume * porosity
            if node[nodeIDi, bodyID].ms > cutoff and node[nodeIDi, bodyID].mf > cutoff:
                node[nodeIDi, bodyID].forces += Bs / ms
                node[nodeIDi, bodyID].forcef += Bf / mf
                node[nodeIDi, bodyID]._correct_nodal_kinematic_semi(dt, Bs / ms, Bf / mf)


@ti.kernel
def kernel_correct_grid_kinematic_semitwophase_2DAxisy(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    dshapefn: ti.template(),
    shapefn: ti.template(),
    dt: ti.template(),
    cutoff: float,
):
    for np in range(particleNum):
        if int(particle[np].active) == 0:
            continue
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        position = particle[np].x
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            ishape = shapefn[lni]
            Bs, Bf = ZEROVEC2f, ZEROVEC2f
            ms, mf = node[nodeIDi, bodyID].ms, node[nodeIDi, bodyID].mf
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                jshape = shapefn[lnj]
                jdshape = dshapefn[lnj]
                Bs -= ishape * jdshape * node[nodeIDj, bodyID].dpressure * volume * (1.0 - porosity)
                Bs[0] -= ishape * jshape / position[0] * node[nodeIDj, bodyID].dpressure * volume * (1.0 - porosity)
                Bf -= ishape * jdshape * node[nodeIDj, bodyID].dpressure * volume * porosity
                Bf[0] -= ishape * jshape / position[0] * node[nodeIDj, bodyID].dpressure * volume * porosity
            if node[nodeIDi, bodyID].ms > cutoff and node[nodeIDi, bodyID].mf > cutoff:
                node[nodeIDi, bodyID].forces += Bs / ms
                node[nodeIDi, bodyID].forcef += Bf / mf
                node[nodeIDi, bodyID]._correct_nodal_kinematic_semi(dt, Bs / ms, Bf / mf)


@ti.kernel
def kernel_project_fic_pressure_gradient(
    total_nodes: int,
    start_index: int,
    end_index: int,
    material_mapping: ti.template(),
    mat_prop: ti.template(),
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    dshapefn: ti.template(),
    shapefn: ti.template(),
):
    # Store minus the volume-weighted L2 projection of C*grad(p_old).
    # C must be the same phase mobility as in the pressure Laplacian, not
    # inverse mixture density. Those coincide only for equal phase densities.
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].active) == 0 or int(particle[np].materialID) == 0:
            continue
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        mobility = porosity / mat_prop.fluid_density + (1.0 - porosity) / mat_prop.solid_density
        gradient = single_point_nodal_pressure_gradient_2D(
            offset,
            int(node_size[np]),
            bodyID,
            node,
            LnID,
            dshapefn,
        )
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            ishape = shapefn[lni]
            node[nodeIDi, bodyID].extra_stabilize -= ishape * volume * mobility * gradient


@ti.kernel
def kernel_normalize_fic_pressure_projection(node: ti.template()):
    for ng, nb in node:
        if node[ng, nb].weight > 0.0:
            node[ng, nb].extra_stabilize /= node[ng, nb].weight


@ti.kernel
def kernel_correct_grid_kinematic_semitwophase_u_p(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    dshapefn: ti.template(),
    shapefn: ti.template(),
    dt: ti.template(),
    cutoff: float,
):
    for np in range(particleNum):
        if int(particle[np].active) == 0:
            continue
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            ishape = shapefn[lni]
            idshape = dshapefn[lni]
            Bm = ZEROVEC2f
            m = node[nodeIDi, bodyID].m
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                jshape = shapefn[lnj]
                jdshape = dshapefn[lnj]
                Bm -= ishape * jdshape * node[nodeIDj, bodyID].dpressure * volume
                # Bm -= idshape * jshape * node[nodeIDj, bodyID].dpressure * volume
            if node[nodeIDi, bodyID].m > cutoff:
                node[nodeIDi, bodyID].force += Bm / m
                node[nodeIDi, bodyID].momentum += Bm / m * dt[None]


@ti.kernel
def kernel_correct_grid_kinematic_semitwophase_mg(
    istep: int,
    node: ti.template(),
    dt: ti.template(),
    cutoff: float,
    element_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    cell_type: ti.template(),
    cell_porosity: ti.template(),
    cell_dpressure: ti.template(),
    cell_phi: ti.template(),
    matProp: ti.template(),
    is_rigid: ti.template(),
):
    p0 = 0.0
    dimension = ti.static(2)
    inv_spacing = 1.0 / element_size
    bodyID = 0 if is_rigid[0] == 0 else 1
    density_s, density_f = matProp.solid_density, matProp.fluid_density
    pressure_cnum = ti.Vector([cell_type.shape[0], cell_type.shape[1]])
    for I in ti.grouped(cell_type):
        for k in ti.static(range(dimension)):
            I_1 = I - ti.Vector.unit(dimension, k)
            if all(I >= 0) and all(I < pressure_cnum) and all(I_1 >= 0) and all(I_1 < pressure_cnum):
                if (
                    (is_mg_fluid_cell(I, cell_type) or is_mg_fluid_cell(I_1, cell_type))
                    and not is_mg_solid_cell(I, cell_type)
                    and not is_mg_solid_cell(I_1, cell_type)
                ):
                    i, j = I
                    nodeID = int(i + j * gnum[0])
                    Bs, Bf = 0.0, 0.0
                    if is_mg_air_cell(I, cell_type):
                        theta = free_surface_theta(I_1, I, cell_phi)
                        Bs = -(p0 - cell_dpressure[I_1]) * inv_spacing[k] / (theta * density_s)
                        Bf = -(p0 - cell_dpressure[I_1]) * inv_spacing[k] / (theta * density_f)
                    elif is_mg_air_cell(I_1, cell_type):
                        theta = free_surface_theta(I, I_1, cell_phi)
                        Bs = -(cell_dpressure[I] - p0) * inv_spacing[k] / (theta * density_s)
                        Bf = -(cell_dpressure[I] - p0) * inv_spacing[k] / (theta * density_f)
                    else:
                        Bs = -(cell_dpressure[I] - cell_dpressure[I_1]) * inv_spacing[k] / density_s
                        Bf = -(cell_dpressure[I] - cell_dpressure[I_1]) * inv_spacing[k] / density_f
                    """p_I1 = get_pressure_with_gfm(I_1, cell_type, cell_pressure)
                    p_I  = get_pressure_with_gfm(I, cell_type, cell_pressure)
                    Bs = -(p_I - p_I1) * idx / density_s
                    Bf = -(p_I - p_I1) * idx / density_f"""
                    # node[nodeID, bodyID].forces[k] += Bs
                    # node[nodeID, bodyID].forcef[k] += Bf
                    # node[nodeID, bodyID].momentums[k] += Bs * dt[None]
                    # node[nodeID, bodyID].momentumf[k] += Bf * dt[None]
                    # print(I, nodeID, cell_pressure[I], cell_pressure[I_1], Bs, Bf)
                    if k == 0:
                        w1, w2 = 0.5, 0.5
                        if get_mg_cell_type(I - ti.Vector.unit(dimension, 1), cell_type) == 2:
                            w1, w2 = 1.0, 0.5
                        elif get_mg_cell_type(I + ti.Vector.unit(dimension, 1), cell_type) == 2:
                            w1, w2 = 0.5, 1.0
                        node[nodeID, bodyID].forces[k] += Bs * w1
                        node[nodeID, bodyID].forcef[k] += Bf * w1
                        node[nodeID, bodyID].momentums[k] += Bs * dt[None] * w1
                        node[nodeID, bodyID].momentumf[k] += Bf * dt[None] * w1
                        node[nodeID + gnum[0], bodyID].forces[k] += Bs * w2
                        node[nodeID + gnum[0], bodyID].forcef[k] += Bf * w2
                        node[nodeID + gnum[0], bodyID].momentums[k] += Bs * dt[None] * w2
                        node[nodeID + gnum[0], bodyID].momentumf[k] += Bf * dt[None] * w2
                    else:
                        w1, w2 = 0.5, 0.5
                        if get_mg_cell_type(I - ti.Vector.unit(dimension, 0), cell_type) == 2:
                            w1, w2 = 1.0, 0.5
                        elif get_mg_cell_type(I + ti.Vector.unit(dimension, 0), cell_type) == 2:
                            w1, w2 = 0.5, 1.0
                        node[nodeID, bodyID].forces[k] += Bs * w1
                        node[nodeID, bodyID].forcef[k] += Bf * w1
                        node[nodeID, bodyID].momentums[k] += Bs * dt[None] * w1
                        node[nodeID, bodyID].momentumf[k] += Bf * dt[None] * w1
                        node[nodeID + 1, bodyID].forces[k] += Bs * w2
                        node[nodeID + 1, bodyID].forcef[k] += Bf * w2
                        node[nodeID + 1, bodyID].momentums[k] += Bs * dt[None] * w2
                        node[nodeID + 1, bodyID].momentumf[k] += Bf * dt[None] * w2


@ti.kernel
def kernel_correct_grid_kinematic_semitwophase_mg_3D(
    istep: int,
    node: ti.template(),
    dt: ti.template(),
    cutoff: float,
    element_size: ti.types.vector(3, float),
    gnum: ti.types.vector(3, int),
    cell_type: ti.template(),
    cell_porosity: ti.template(),
    cell_dpressure: ti.template(),
    cell_phi: ti.template(),
    matProp: ti.template(),
    is_rigid: ti.template(),
):
    p0 = 0.0
    dimension = ti.static(3)
    inv_spacing = 1.0 / element_size
    bodyID = 0 if is_rigid[0] == 0 else 1
    density_s, density_f = matProp.solid_density, matProp.fluid_density
    cnum = ti.Vector([cell_type.shape[0], cell_type.shape[1], cell_type.shape[2]])
    for I in ti.grouped(cell_type):
        for k in ti.static(range(dimension)):
            I_1 = I - ti.Vector.unit(dimension, k)
            if is_mg_valid_cell(I, cnum) and is_mg_valid_cell(I_1, cnum):
                if (
                    (is_mg_fluid_cell(I, cell_type) or is_mg_fluid_cell(I_1, cell_type))
                    and not is_mg_solid_cell(I, cell_type)
                    and not is_mg_solid_cell(I_1, cell_type)
                ):
                    Bs, Bf = 0.0, 0.0
                    if is_mg_air_cell(I, cell_type):
                        theta = free_surface_theta(I_1, I, cell_phi)
                        Bs = -(p0 - cell_dpressure[I_1]) * inv_spacing[k] / (theta * density_s)
                        Bf = -(p0 - cell_dpressure[I_1]) * inv_spacing[k] / (theta * density_f)
                    elif is_mg_air_cell(I_1, cell_type):
                        theta = free_surface_theta(I, I_1, cell_phi)
                        Bs = -(cell_dpressure[I] - p0) * inv_spacing[k] / (theta * density_s)
                        Bf = -(cell_dpressure[I] - p0) * inv_spacing[k] / (theta * density_f)
                    else:
                        Bs = -(cell_dpressure[I] - cell_dpressure[I_1]) * inv_spacing[k] / density_s
                        Bf = -(cell_dpressure[I] - cell_dpressure[I_1]) * inv_spacing[k] / density_f
                    for offset in ti.static(ti.grouped(ti.ndrange((0, 2), (0, 2), (0, 2)))):
                        if offset[k] == 0:
                            node_index = I + offset
                            nodeID = linearize(node_index, gnum)
                            weight = 0.25
                            node[nodeID, bodyID].forces[k] += Bs * weight
                            node[nodeID, bodyID].forcef[k] += Bf * weight
                            node[nodeID, bodyID].momentums[k] += Bs * dt[None] * weight
                            node[nodeID, bodyID].momentumf[k] += Bf * dt[None] * weight


@ti.kernel
def kernel_correct_grid_kinematic_semitwophase_2DAxi_mg_fdm(
    istep: int,
    node: ti.template(),
    dt: ti.template(),
    cutoff: float,
    element_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    cell_type: ti.template(),
    cell_porosity: ti.template(),
    cell_dpressure: ti.template(),
    cell_phi: ti.template(),
    matProp: ti.template(),
    is_rigid: ti.template(),
    axi_offset: float,
):
    p0 = 0.0
    dimension = ti.static(2)
    dx = element_size[0]
    inv_spacing = 1.0 / element_size
    bodyID = 0 if is_rigid[0] == 0 else 1
    density_s, density_f = matProp.solid_density, matProp.fluid_density
    pressure_cnum = ti.Vector([cell_type.shape[0], cell_type.shape[1]])
    for I in ti.grouped(cell_type):
        for k in ti.static(range(dimension)):
            I_1 = I - ti.Vector.unit(dimension, k)
            if all(I >= 0) and all(I < pressure_cnum) and all(I_1 >= 0) and all(I_1 < pressure_cnum):
                if (
                    (is_mg_fluid_cell(I, cell_type) or is_mg_fluid_cell(I_1, cell_type))
                    and not is_mg_solid_cell(I, cell_type)
                    and not is_mg_solid_cell(I_1, cell_type)
                ):
                    i, j = I
                    nodeID = int(i + j * gnum[0])
                    Bs, Bf = 0.0, 0.0
                    if is_mg_air_cell(I, cell_type):
                        theta = free_surface_theta(I_1, I, cell_phi)
                        Bs = -(p0 - cell_dpressure[I_1]) * inv_spacing[k] / (theta * density_s)
                        Bf = -(p0 - cell_dpressure[I_1]) * inv_spacing[k] / (theta * density_f)
                        # if k==0:
                        #     Bs -= (p0 + cell_dpressure[I_1]) * 0.5 / density_s / (float(I[0]) * dx - axi_offset)
                        #     Bf -= (p0 + cell_dpressure[I_1]) * 0.5 / density_f / (float(I[0]) * dx - axi_offset)
                    elif is_mg_air_cell(I_1, cell_type):
                        theta = free_surface_theta(I, I_1, cell_phi)
                        Bs = -(cell_dpressure[I] - p0) * inv_spacing[k] / (theta * density_s)
                        Bf = -(cell_dpressure[I] - p0) * inv_spacing[k] / (theta * density_f)
                        # if k==0:
                        #     Bs -= (cell_dpressure[I] + p0) * 0.5 / density_s / (float(I[0]) * dx - axi_offset)
                        #     Bf -= (cell_dpressure[I] + p0) * 0.5 / density_f / (float(I[0]) * dx - axi_offset)
                    else:
                        Bs = -(cell_dpressure[I] - cell_dpressure[I_1]) * inv_spacing[k] / density_s
                        Bf = -(cell_dpressure[I] - cell_dpressure[I_1]) * inv_spacing[k] / density_f
                        # if k==0:
                        #     Bs -= (cell_dpressure[I] + cell_dpressure[I_1]) * 0.5 / density_s / (float(I[0]) * dx - axi_offset)
                        #     Bf -= (cell_dpressure[I] + cell_dpressure[I_1]) * 0.5 / density_f / (float(I[0]) * dx - axi_offset)
                    if k == 0:
                        w1, w2 = 0.5, 0.5
                        if get_mg_cell_type(I - ti.Vector.unit(dimension, 1), cell_type) == 2:
                            w1, w2 = 1.0, 0.5
                        elif get_mg_cell_type(I + ti.Vector.unit(dimension, 1), cell_type) == 2:
                            w1, w2 = 0.5, 1.0
                        node[nodeID, bodyID].forces[k] += Bs * w1
                        node[nodeID, bodyID].forcef[k] += Bf * w1
                        node[nodeID, bodyID].momentums[k] += Bs * dt[None] * w1
                        node[nodeID, bodyID].momentumf[k] += Bf * dt[None] * w1
                        node[nodeID + gnum[0], bodyID].forces[k] += Bs * w2
                        node[nodeID + gnum[0], bodyID].forcef[k] += Bf * w2
                        node[nodeID + gnum[0], bodyID].momentums[k] += Bs * dt[None] * w2
                        node[nodeID + gnum[0], bodyID].momentumf[k] += Bf * dt[None] * w2
                    else:
                        w1, w2 = 0.5, 0.5
                        if get_mg_cell_type(I - ti.Vector.unit(dimension, 0), cell_type) == 2:
                            w1, w2 = 1.0, 0.5
                        elif get_mg_cell_type(I + ti.Vector.unit(dimension, 0), cell_type) == 2:
                            w1, w2 = 0.5, 1.0
                        node[nodeID, bodyID].forces[k] += Bs * w1
                        node[nodeID, bodyID].forcef[k] += Bf * w1
                        node[nodeID, bodyID].momentums[k] += Bs * dt[None] * w1
                        node[nodeID, bodyID].momentumf[k] += Bf * dt[None] * w1
                        node[nodeID + 1, bodyID].forces[k] += Bs * w2
                        node[nodeID + 1, bodyID].forcef[k] += Bf * w2
                        node[nodeID + 1, bodyID].momentums[k] += Bs * dt[None] * w2
                        node[nodeID + 1, bodyID].momentumf[k] += Bf * dt[None] * w2


@ti.kernel
def kernel_correct_grid_kinematic_semitwophase_FIC_mg_fdm(
    istep: int,
    node: ti.template(),
    dt: ti.template(),
    cutoff: float,
    element_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    cell_type: ti.template(),
    cell_porosity: ti.template(),
    cell_pressure: ti.template(),
    cell_dpressure: ti.template(),
    cell_phi: ti.template(),
    matProp: ti.template(),
):
    p0 = 0.0
    bodyID = 0
    dimension = ti.static(2)
    inv_spacing = 1.0 / element_size
    density_s = matProp.solid_density
    density_f = matProp.fluid_density
    for I in ti.grouped(cell_type):
        for k in ti.static(range(dimension)):
            I_1 = I - ti.Vector.unit(dimension, k)
            if all(I >= 0) and all(I < gnum) and all(I_1 >= 0) and all(I_1 < gnum):
                if (
                    (is_mg_fluid_cell(I, cell_type) or is_mg_fluid_cell(I_1, cell_type))
                    and not is_mg_solid_cell(I, cell_type)
                    and not is_mg_solid_cell(I_1, cell_type)
                ):
                    i, j = I
                    nodeID = int(i + j * gnum[0])
                    porosity = cell_porosity[I]
                    Bs, Bf, fai = 0.0, 0.0, 0.0
                    if is_mg_air_cell(I, cell_type):
                        # p0 = get_ghost_pressure(cell_phi[I], cell_phi[I_1], cell_pressure[I_1])
                        Bs = -(p0 - cell_dpressure[I_1]) * inv_spacing[k] / density_s
                        Bf = -(p0 - cell_dpressure[I_1]) * inv_spacing[k] / density_f
                        pI, pI1 = cell_pressure[I] + p0, cell_pressure[I_1] + cell_dpressure[I]
                        fai = -(pI - pI1) * inv_spacing[k] / ((1.0 - porosity) * density_s + porosity * density_f)
                    elif is_mg_air_cell(I_1, cell_type):
                        # p0 = get_ghost_pressure(cell_phi[I_1], cell_phi[I], cell_pressure[I])
                        Bs = -(cell_dpressure[I] - p0) * inv_spacing[k] / density_s
                        Bf = -(cell_dpressure[I] - p0) * inv_spacing[k] / density_f
                        pI, pI1 = cell_pressure[I] + cell_dpressure[I], cell_pressure[I_1] + p0
                        fai = -(pI - pI1) * inv_spacing[k] / ((1.0 - porosity) * density_s + porosity * density_f)
                    else:
                        Bs = -(cell_dpressure[I] - cell_dpressure[I_1]) * inv_spacing[k] / density_s
                        Bf = -(cell_dpressure[I] - cell_dpressure[I_1]) * inv_spacing[k] / density_f
                        pI = cell_pressure[I] + cell_dpressure[I]
                        pI1 = cell_pressure[I_1] + cell_dpressure[I_1]
                        # fai = -(pI - pI1) * idx * ( (1.0 - porosity) / density_s  + porosity / density_f)
                        fai = -(pI - pI1) * inv_spacing[k] / ((1.0 - porosity) * density_s + porosity * density_f)
                    # if is_mg_fluid_cell(I, cell_type) and is_mg_fluid_cell(I_1, cell_type):
                    #     pI, pI1 = cell_dpressure[I] + cell_pressure[I], cell_dpressure[I_1] + cell_pressure[I_1]
                    #     fai = -(pI - pI1) * idx / ( (1.0 - porosity) * density_s  + porosity * density_f)
                    if k == 0:
                        w1, w2 = 0.5, 0.5
                        if get_mg_cell_type(I - ti.Vector.unit(dimension, 1), cell_type) == 2:
                            w1, w2 = 1.0, 0.5
                        elif get_mg_cell_type(I + ti.Vector.unit(dimension, 1), cell_type) == 2:
                            w1, w2 = 0.5, 1.0
                        node[nodeID, bodyID].forces[k] += Bs * w1
                        node[nodeID, bodyID].forcef[k] += Bf * w1
                        node[nodeID, bodyID].momentums[k] += Bs * dt[None] * w1
                        node[nodeID, bodyID].momentumf[k] += Bf * dt[None] * w1
                        node[nodeID + gnum[0], bodyID].forces[k] += Bs * w2
                        node[nodeID + gnum[0], bodyID].forcef[k] += Bf * w2
                        node[nodeID + gnum[0], bodyID].momentums[k] += Bs * dt[None] * w2
                        node[nodeID + gnum[0], bodyID].momentumf[k] += Bf * dt[None] * w2
                        node[nodeID, bodyID].extra_stabilize[k] += fai * w1
                        node[nodeID + gnum[0], bodyID].extra_stabilize[k] += fai * w2
                    else:
                        w1, w2 = 0.5, 0.5
                        if get_mg_cell_type(I - ti.Vector.unit(dimension, 0), cell_type) == 2:
                            w1, w2 = 1.0, 0.5
                        elif get_mg_cell_type(I + ti.Vector.unit(dimension, 0), cell_type) == 2:
                            w1, w2 = 0.5, 1.0
                        node[nodeID, bodyID].forces[k] += Bs * w1
                        node[nodeID, bodyID].forcef[k] += Bf * w1
                        node[nodeID, bodyID].momentums[k] += Bs * dt[None] * w1
                        node[nodeID, bodyID].momentumf[k] += Bf * dt[None] * w1
                        node[nodeID + 1, bodyID].forces[k] += Bs * w2
                        node[nodeID + 1, bodyID].forcef[k] += Bf * w2
                        node[nodeID + 1, bodyID].momentums[k] += Bs * dt[None] * w2
                        node[nodeID + 1, bodyID].momentumf[k] += Bf * dt[None] * w2
                        node[nodeID, bodyID].extra_stabilize[k] += fai * w1
                        node[nodeID + 1, bodyID].extra_stabilize[k] += fai * w2
                        """if cell_type[I - ti.Vector.unit(dimension, 1)] == 2:
                            node[nodeID, bodyID].extra_stabilize[k] = 0.0
                            node[nodeID + 1, bodyID].extra_stabilize[k] = 0.0
                            # node[nodeID, bodyID].extra_stabilize[0] = 0.0
                            # node[nodeID + 1, bodyID].extra_stabilize[0] = 0.0
                        elif cell_type[I + ti.Vector.unit(dimension, 1)] == 2:
                            node[nodeID + gnum[0], bodyID].extra_stabilize[k] = 0.0
                            node[nodeID + 1 + gnum[0], bodyID].extra_stabilize[k] = 0.0"""


@ti.kernel
def kernel_grid_porosity(cutoff: float, is_rigid: ti.template(), node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff and is_rigid[nb] == 0:
                node[ng, nb].porosity /= node[ng, nb].m
                node[ng, nb].pressure /= node[ng, nb].m


@ti.kernel
def kernel_kinemaitc_g2p_twophase2D(
    total_nodes: int,
    alpha: float,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    # ti.block_local(dt)
    # for ng in range(node.shape[0]):
    #     for nb in range(node.shape[1]):
    #         print(ng, node[ng, nb].forcef)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPICs, vFLIPs = ZEROVEC2f, ZEROVEC2f
            vPICf, vFLIPf = ZEROVEC2f, ZEROVEC2f
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
def kernel_kinemaitc_g2p_semitwophase2D(
    total_nodes: int,
    alpha: float,
    beta: float,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    # ti.block_local(dt)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPICs, vFLIPs = ZEROVEC2f, ZEROVEC2f
            vPICf, vFLIPf = ZEROVEC2f, ZEROVEC2f
            dpressure = 0.0
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                velocitys = node[nodeID, bodyID].momentums
                velocityf = node[nodeID, bodyID].momentumf
                acclerations = node[nodeID, bodyID].forces
                acclerationf = node[nodeID, bodyID].forcef
                ndpressure = beta * node[nodeID, bodyID].pressure + node[nodeID, bodyID].dpressure
                vPICs += shape_mapping(shape_fn, velocitys)
                vFLIPs += shape_mapping(shape_fn, acclerations) * dt[None]
                vPICf += shape_mapping(shape_fn, velocityf)
                vFLIPf += shape_mapping(shape_fn, acclerationf) * dt[None]
                dpressure += shape_mapping(shape_fn, ndpressure)
            particle[np]._update_particle_state(dt, alpha, vPICs, vFLIPs, vPICs, vFLIPs, vPICf, vFLIPf)
            # Reconstruct the solved nodal pressure; incremental particle
            # accumulation retains pressure modes invisible to the grid.
            particle[np]._update_particle_pressure(0.0, dpressure)


@ti.kernel
def kernel_kinemaitc_g2p_semitwophase3D(
    total_nodes: int,
    alpha: float,
    beta: float,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPICs = ti.Vector.zero(float, 3)
            vFLIPs = ti.Vector.zero(float, 3)
            vPICf = ti.Vector.zero(float, 3)
            vFLIPf = ti.Vector.zero(float, 3)
            dpressure = 0.0
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                velocitys = node[nodeID, bodyID].momentums
                velocityf = node[nodeID, bodyID].momentumf
                acclerations = node[nodeID, bodyID].forces
                acclerationf = node[nodeID, bodyID].forcef
                ndpressure = beta * node[nodeID, bodyID].pressure + node[nodeID, bodyID].dpressure
                vPICs += shape_mapping(shape_fn, velocitys)
                vFLIPs += shape_mapping(shape_fn, acclerations) * dt[None]
                vPICf += shape_mapping(shape_fn, velocityf)
                vFLIPf += shape_mapping(shape_fn, acclerationf) * dt[None]
                dpressure += shape_mapping(shape_fn, ndpressure)
            particle[np]._update_particle_state(dt, alpha, vPICs, vFLIPs, vPICs, vFLIPs, vPICf, vFLIPf)
            particle[np]._update_particle_pressure(0.0, dpressure)


@ti.kernel
def kernel_kinemaitc_g2p_semitwophase2D_FIC(
    total_nodes: int,
    alpha: float,
    beta: float,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    # ti.block_local(dt)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPICs, vFLIPs = ZEROVEC2f, ZEROVEC2f
            vPICf, vFLIPf = ZEROVEC2f, ZEROVEC2f
            dpressure = 0.0
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                velocitys = node[nodeID, bodyID].momentums
                velocityf = node[nodeID, bodyID].momentumf
                acclerations = node[nodeID, bodyID].forces
                acclerationf = node[nodeID, bodyID].forcef
                ndpressure = beta * node[nodeID, bodyID].pressure + node[nodeID, bodyID].dpressure
                vPICs += shape_mapping(shape_fn, velocitys)
                vFLIPs += shape_mapping(shape_fn, acclerations) * dt[None]
                vPICf += shape_mapping(shape_fn, velocityf)
                vFLIPf += shape_mapping(shape_fn, acclerationf) * dt[None]
                dpressure += shape_mapping(shape_fn, ndpressure)
            particle[np]._update_particle_state(dt, alpha, vPICs, vFLIPs, vPICs, vFLIPs, vPICf, vFLIPf)
            particle[np]._update_particle_pressure(0.0, dpressure)


@ti.kernel
def kernel_kinemaitc_g2p_semitwophase2D_u_p(
    total_nodes: int,
    alpha: float,
    beta: float,
    dt: ti.template(),
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    # ti.block_local(dt)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vPIC, vFLIP = ZEROVEC2f, ZEROVEC2f
            dpressure = 0.0
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape_fn = shapefn[ln]
                velocity = node[nodeID, bodyID].momentum
                accleration = node[nodeID, bodyID].force
                ndpressure = beta * node[nodeID, bodyID].pressure + node[nodeID, bodyID].dpressure
                vPIC += shape_mapping(shape_fn, velocity)
                vFLIP += shape_mapping(shape_fn, accleration) * dt[None]
                dpressure += shape_mapping(shape_fn, ndpressure)
            particle[np]._update_particle_state_u_p(dt, alpha, vPIC, vFLIP)
            particle[np].vs = particle[np].v
            particle[np].vf = particle[np].v
            # The storage RHS already retains the old particle pressure.
            # Transfer the solved field so prescribed nodal pressures survive.
            particle[np]._update_particle_pressure(0.0, dpressure)


@ti.kernel
def kernel_clamp_twophase_particle_pressure(
    start_index: int,
    end_index: int,
    material_mapping: ti.template(),
    minimum_pressure: float,
    particle: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].active) == 1:
            # free_surface marks a whole cell-neighbourhood, not particles
            # located on p=0. The pressure solve already imposes that boundary;
            # zeroing nearby interior points erases their hydrostatic pressure.
            particle[np].pressure = ti.max(particle[np].pressure, minimum_pressure)


@ti.kernel
def kernel_mass_g2p_poisson(
    total_nodes: int,
    cell_volume: float,
    node_size: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node: ti.template(),
    particleNum: int,
    particle: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            mdensity = 0.0
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                mass_density = shape_mapping(shapefn[ln], node[nodeID, bodyID].ms / cell_volume)
                mdensity += mass_density
            particle[np].mass_density = mdensity


@ti.kernel
def kernel_pressure_g2p_twophase_2D(
    extra_node: ti.template(),
    particleNum: int,
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    total_nodes: int,
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            scale = 0.0
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                scale += shape_mapping(shapefn[ln], extra_node[nodeID, bodyID].pressure)
            # particle[np].porosity = scale
            stress = particle[np].stress
            particle[np].stress = stress + (scale - MeanStress(stress)) * EYE


@ti.func
def update_velocity_gradient_affine_twophase2D(
    np, total_nodes, gnum, grid_size, node, particle, LnID, shapefn, node_size
):
    Wp = ZEROMAT2x2
    Bps = ZEROMAT2x2
    Bpf = ZEROMAT2x2
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    position = particle[np].x
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        grid_coord = grid_size * vec2f(vectorize_id(nodeID, gnum))
        pointer = grid_coord - position
        gvs = node[nodeID, bodyID].momentums
        gvf = node[nodeID, bodyID].momentumf
        shape_fn = shapefn[ln]

        Wp += shape_fn * outer_product2D(pointer, pointer)
        Bps += shape_fn * outer_product2D(pointer, gvs)
        Bpf += shape_fn * outer_product2D(pointer, gvf)
    return truncation(Bps @ Wp.inverse()), truncation(Bpf @ Wp.inverse())


@ti.func
def update_velocity_gradient_affine_twophase2DAxisy(
    np, total_nodes, gnum, grid_size, node, particle, LnID, shapefn, node_size, axis_offset
):
    Wp, Bps, Bpf = ZEROMAT2x2, ZEROMAT2x2, ZEROMAT2x2
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    position = particle[np].x
    vr = 0.0
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        grid_coord = grid_size * vec2f(vectorize_id(nodeID, gnum))
        pointer = grid_coord - position
        gvs = node[nodeID, bodyID].momentums
        gvf = node[nodeID, bodyID].momentumf
        shape_fn = shapefn[ln]

        Wp += shape_fn * outer_product2D(pointer, pointer)
        Bps += shape_fn * outer_product2D(pointer, gvs)
        Bpf += shape_fn * outer_product2D(pointer, gvf)
        vr += shape_fn * gvs[0]
    velocity_gradient0 = Bps @ Wp.inverse()
    velocity_gradients = mat3x3(
        [
            [velocity_gradient0[0, 0], velocity_gradient0[0, 1], 0],
            [velocity_gradient0[1, 0], velocity_gradient0[1, 1], 0],
            [0, 0, vr / (position[0] - axis_offset)],
        ]
    )
    velocity_gradient0 = Bpf @ Wp.inverse()
    velocity_gradientf = mat3x3(
        [
            [velocity_gradient0[0, 0], velocity_gradient0[0, 1], 0],
            [velocity_gradient0[1, 0], velocity_gradient0[1, 1], 0],
            [0, 0, 0],
        ]
    )
    return truncation(velocity_gradients), truncation(velocity_gradientf)


@ti.func
def update_velocity_gradient_2D(np, total_nodes, node, particle, LnID, dshapefn, node_size):
    velocity_gradient = ZEROMAT2x2
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        gv = node[nodeID, bodyID].momentum
        dshape_fn = dshapefn[ln]
        velocity_gradient += outer_product2D(dshape_fn, gv)
    return truncation(velocity_gradient)


@ti.func
def update_velocity_gradient_twophase2D(np, total_nodes, node, particle, LnID, dshapefn, node_size):
    velocity_gradients = ZEROMAT2x2
    velocity_gradientf = ZEROMAT2x2
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        gvs = node[nodeID, bodyID].momentums
        gvf = node[nodeID, bodyID].momentumf
        dshape_fn = dshapefn[ln]
        velocity_gradients += outer_product2D(dshape_fn, gvs)
        velocity_gradientf += outer_product2D(dshape_fn, gvf)
    return truncation(velocity_gradients), truncation(velocity_gradientf)


@ti.func
def update_velocity_gradient_twophase2DAxisy(
    np, total_nodes, node, particle, LnID, shapefn, dshapefn, node_size, position
):
    velocity_gradients = ZEROMAT3x3
    velocity_gradientf = ZEROMAT3x3
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        gvs = node[nodeID, bodyID].momentums
        gvf = node[nodeID, bodyID].momentumf
        shape_fn = shapefn[ln]
        dshape_fn = dshapefn[ln]
        velocity_gradient0 = outer_product2D(dshape_fn, gvs)
        velocity_gradients += mat3x3(
            [
                [velocity_gradient0[0, 0], velocity_gradient0[0, 1], 0],
                [velocity_gradient0[1, 0], velocity_gradient0[1, 1], 0],
                [0, 0, shape_fn * gvs[0] / position[0]],
            ]
        )
        velocity_gradient0 = outer_product2D(dshape_fn, gvf)
        velocity_gradientf += mat3x3(
            [
                [velocity_gradient0[0, 0], velocity_gradient0[0, 1], 0],
                [velocity_gradient0[1, 0], velocity_gradient0[1, 1], 0],
                [0, 0, shape_fn * gvf[0] / position[0]],
            ]
        )
    return truncation(velocity_gradients), truncation(velocity_gradientf)


@ti.kernel
def kernel_compute_stress_strain_semitwophase2D(
    total_nodes: int,
    dt: ti.template(),
    start_index: int,
    end_index: int,
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    matProp: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            velocity_gradients, velocity_gradientf = update_velocity_gradient_twophase2D(
                np, total_nodes, node, particle, LnID, dshapefn, node_size
            )
            previous_stress = particle[np].stress
            particle[np].vol *= matProp.update_particle_volume_2D(np, velocity_gradients, stateVars, dt)
            matProp.update_particle_porosity_2D(np, velocity_gradients, stateVars, particle, dt)
            matProp.update_particle_massf(np, stateVars, particle)
            particle[np].stress = matProp.compute_single_layer_effective_stress_2d(
                np, previous_stress, velocity_gradients, particle[np].porosity, stateVars, dt
            )
            particle[np].solid_velocity_gradient = velocity_gradients
            particle[np].fluid_velocity_gradient = velocity_gradientf


@ti.func
def update_velocity_gradient_twophase3D(np, total_nodes, node, particle, LnID, dshapefn, node_size):
    velocity_gradients = ZEROMAT3x3
    velocity_gradientf = ZEROMAT3x3
    bodyID = int(particle[np].bodyID)
    offset = np * total_nodes
    for ln in range(offset, offset + int(node_size[np])):
        nodeID = LnID[ln]
        gvs = node[nodeID, bodyID].momentums
        gvf = node[nodeID, bodyID].momentumf
        dshape_fn = dshapefn[ln]
        velocity_gradients += outer_product(dshape_fn, gvs)
        velocity_gradientf += outer_product(dshape_fn, gvf)
    return truncation(velocity_gradients), truncation(velocity_gradientf)


@ti.kernel
def kernel_compute_stress_strain_semitwophase3D(
    total_nodes: int,
    dt: ti.template(),
    start_index: int,
    end_index: int,
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    matProp: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            velocity_gradients, velocity_gradientf = update_velocity_gradient_twophase3D(
                np, total_nodes, node, particle, LnID, dshapefn, node_size
            )
            previous_stress = particle[np].stress
            particle[np].vol *= matProp.update_particle_volume(np, velocity_gradients, stateVars, dt)
            particle[np].porosity = matProp.update_particle_porosity(velocity_gradients, particle[np].porosity, dt)
            matProp.update_particle_massf(np, stateVars, particle)
            particle[np].stress = matProp.compute_single_layer_effective_stress(
                np, previous_stress, velocity_gradients, particle[np].porosity, stateVars, dt
            )
            particle[np].solid_velocity_gradient = velocity_gradients
            particle[np].fluid_velocity_gradient = velocity_gradientf


@ti.kernel
def kernel_compute_stress_strain_APIC_twophase2D(
    total_nodes: int,
    dt: ti.template(),
    start_index: int,
    end_index: int,
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    matProp: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    grid_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            # velocity_gradients, velocity_gradientf = update_velocity_gradient_affine_twophase2D(np, total_nodes, gnum, grid_size, node, particle, LnID, shapefn, node_size)
            velocity_gradients = particle[np].solid_velocity_gradient
            previous_stress = particle[np].stress
            particle[np].vol *= matProp.update_particle_volume_2D(np, velocity_gradients, stateVars, dt)
            matProp.update_particle_porosity_2D(np, velocity_gradients, stateVars, particle, dt)
            matProp.update_particle_massf(np, stateVars, particle)
            particle[np].stress = matProp.compute_single_layer_effective_stress_2d(
                np, previous_stress, velocity_gradients, particle[np].porosity, stateVars, dt
            )


@ti.kernel
def kernel_compute_affine_matrix_APIC_twophase2D(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    grid_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradients, velocity_gradientf = update_velocity_gradient_affine_twophase2D(
                np, total_nodes, gnum, grid_size, node, particle, LnID, shapefn, node_size
            )
            particle[np].solid_velocity_gradient = velocity_gradients
            particle[np].fluid_velocity_gradient = velocity_gradientf


@ti.kernel
def kernel_compute_stress_strain_twophase2D_u_p(
    total_nodes: int,
    dt: ti.template(),
    start_index: int,
    end_index: int,
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    matProp: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = material_mapping[i]
        # print(np, particle[np].free_surface)
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            velocity_gradient = update_velocity_gradient_2D(np, total_nodes, node, particle, LnID, dshapefn, node_size)
            previous_stress = particle[np].stress
            particle[np].vol *= matProp.update_particle_volume_2D(np, velocity_gradient, stateVars, dt)
            matProp.update_particle_porosity_2D_u_p(np, velocity_gradient, stateVars, particle, dt)
            matProp.update_particle_massf(np, stateVars, particle)
            particle[np].stress = matProp.compute_single_layer_effective_stress_2d(
                np, previous_stress, velocity_gradient, particle[np].porosity, stateVars, dt
            )
            particle[np].solid_velocity_gradient = velocity_gradient


@ti.kernel
def kernel_compute_stress_strain_twophase2DAxisy(
    total_nodes: int,
    dt: ti.template(),
    start_index: int,
    end_index: int,
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    matProp: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    axis_offset: float,
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            position = particle[np].x - vec2f([axis_offset, 0.0])
            velocity_gradients, velocity_gradientf = update_velocity_gradient_twophase2DAxisy(
                np, total_nodes, node, particle, LnID, shapefn, dshapefn, node_size, position
            )
            previous_stress = particle[np].stress
            particle[np].vol *= matProp.update_particle_volume(np, velocity_gradients, stateVars, dt)
            matProp.update_particle_porosity_axisy(np, velocity_gradients, stateVars, particle, dt)
            matProp.update_particle_massf(np, stateVars, particle)
            particle[np].stress = matProp.compute_single_layer_effective_stress(
                np, previous_stress, velocity_gradients, particle[np].porosity, stateVars, dt
            )
            particle[np].velocity_gradient = velocity_gradients


@ti.kernel
def kernel_compute_stress_strain_APIC_twophase2DAxisy(
    total_nodes: int,
    dt: ti.template(),
    start_index: int,
    end_index: int,
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    matProp: ti.template(),
    stateVars: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    grid_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    axis_offset: float,
):
    # ti.block_local(dt)
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            # velocity_gradients, velocity_gradientf = update_velocity_gradient_affine_twophase2DAxisy(np, total_nodes, gnum, grid_size, node, particle, LnID, shapefn, node_size, axis_offset)
            # velocity_gradients = particle[np].velocity_gradient
            position = particle[np].x - vec2f([axis_offset, 0.0])
            velocity_gradients, velocity_gradientf = update_velocity_gradient_twophase2DAxisy(
                np, total_nodes, node, particle, LnID, shapefn, dshapefn, node_size, position
            )
            previous_stress = particle[np].stress
            particle[np].vol *= matProp.update_particle_volume(np, velocity_gradients, stateVars, dt)
            matProp.update_particle_porosity_axisy(np, velocity_gradients, stateVars, particle, dt)
            matProp.update_particle_massf(np, stateVars, particle)
            particle[np].stress = matProp.compute_single_layer_effective_stress(
                np, previous_stress, velocity_gradients, particle[np].porosity, stateVars, dt
            )


@ti.kernel
def kernel_compute_affine_matrix_APIC_twophase2DAxisy(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    grid_size: ti.types.vector(2, float),
    gnum: ti.types.vector(2, int),
    axis_offset: float,
):
    # ti.block_local(dt)
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            velocity_gradients, velocity_gradientf = update_velocity_gradient_affine_twophase2DAxisy(
                np, total_nodes, gnum, grid_size, node, particle, LnID, shapefn, node_size, axis_offset
            )
            particle[np].velocity_gradient = velocity_gradients
            particle[np].velocity_gradientf = velocity_gradientf


@ti.kernel
def kernel_calc_contact_normal_twophase(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                grad_domain = dshapefn[ln] * particle[np].vol  # ms
                node[LnID[ln], bodyID]._update_nodal_grad_domain(grad_domain)


@ti.kernel
def kernel_calc_contact_normal_twophase_2DAxisy(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    axis_offset: float,
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            position = particle[np].x
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                grad_domain = dshapefn[ln] * particle[np].vol / (position[0] - axis_offset)
                node[LnID[ln], bodyID]._update_nodal_grad_domain(grad_domain)


@ti.kernel
def kernel_assemble_contact_force_solid(cutoff: float, dt: ti.template(), node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].ms > cutoff:
                node[ng, nb]._contact_force_assemble_solid(dt)


@ti.kernel
def kernel_calc_friction_contact_semi_2D(
    cut_off: float,
    mu: float,
    dt: ti.template(),
    is_rigid: ti.template(),
    node: ti.template(),
    nodal_coords: ti.template(),
    axis_offset: float,
):
    # ti.block_local(dt)
    cut_off_ = cut_off * 0.1
    for ng in range(node.shape[0]):
        bodyID1, bodyID2 = 0, 1

        m1, m2 = node[ng, bodyID1].m, node[ng, bodyID2].mf
        if m1 > cut_off_ and m2 > cut_off_:  # bodyID1 as rigid
            mv1, mv2 = m1 * node[ng, bodyID1].momentum, m2 * node[ng, bodyID2].momentumf
            norm1, norm2 = node[ng, bodyID1].grad_domain, node[ng, bodyID2].grad_domain

            norm, g_mass = ZEROVEC2f, 0.0
            if is_rigid[bodyID1] == 0 and is_rigid[bodyID2] == 0:
                norm = Normalize(norm1 - norm2)
                g_mass = (m1 + m2) * dt[None]
            elif is_rigid[bodyID1] == 1:
                norm = Normalize(norm1)
                g_mass = m1 * dt[None]
            elif is_rigid[bodyID2] == 1:
                norm = -Normalize(norm2)
                g_mass = m2 * dt[None]

            if nodal_coords[ng][0] <= 1.0e-5 + axis_offset:
                norm[0] = 0.0
                norm[1] = -1.0

            is_penetrate = (mv1 * m2 - m1 * mv2).dot(norm)
            if is_penetrate > Threshold:
                # print(ng, 'fluid contact')
                inv_gmass = 1.0 / g_mass
                cforce = (mv1 * m2 - m1 * mv2) * inv_gmass
                norm_force = is_penetrate * inv_gmass
                if mu > Threshold:  # 0.87 and norm[0] < 0.90
                    trial_ft = cforce - norm_force * norm
                    fstick = trial_ft.norm()
                    fslip = mu * ti.abs(norm_force)
                    if fslip < fstick:
                        cforce = norm_force * norm + fslip * (trial_ft / fstick)
                else:
                    cforce = norm_force * norm
                node[ng, bodyID1]._update_contact_force_fluid(-norm_force * norm)
                node[ng, bodyID2]._update_contact_force_fluid(norm_force * norm)
                node[ng, bodyID2].forcef += cforce / node[ng, bodyID2].mf
                node[ng, bodyID2].momentumf += cforce / node[ng, bodyID2].mf * dt[None]

        m1, m2 = node[ng, bodyID1].m, node[ng, bodyID2].ms
        if m1 > cut_off_ and m2 > cut_off_:  # bodyID1 as rigid
            mv1, mv2 = m1 * node[ng, bodyID1].momentum, m2 * node[ng, bodyID2].momentums
            norm1, norm2 = node[ng, bodyID1].grad_domain, node[ng, bodyID2].grad_domain

            norm, g_mass = ZEROVEC2f, 0.0
            if is_rigid[bodyID1] == 0 and is_rigid[bodyID2] == 0:
                norm = Normalize(norm1 - norm2)
                g_mass = (m1 + m2) * dt[None]
            elif is_rigid[bodyID1] == 1:
                norm = Normalize(norm1)
                g_mass = m1 * dt[None]
            elif is_rigid[bodyID2] == 1:
                norm = -Normalize(norm2)
                g_mass = m2 * dt[None]

            if nodal_coords[ng][0] <= 1.0e-5 + axis_offset:
                norm[0] = 0.0
                norm[1] = -1.0

            is_penetrate = (mv1 * m2 - m1 * mv2).dot(norm)
            if is_penetrate > Threshold:
                inv_gmass = 1.0 / g_mass
                cforce = (mv1 * m2 - m1 * mv2) * inv_gmass
                norm_force = is_penetrate * inv_gmass
                if mu > Threshold:  # 0.87 and norm[0] < 0.90
                    trial_ft = cforce - norm_force * norm
                    fstick = trial_ft.norm()
                    fslip = mu * ti.abs(norm_force)
                    if fslip < fstick:
                        cforce = norm_force * norm + fslip * (trial_ft / fstick)
                else:
                    cforce = norm_force * norm
                node[ng, bodyID1]._update_contact_force_solid(-cforce)
                node[ng, bodyID2]._update_contact_force_solid(cforce)
                node[ng, bodyID2].forces += cforce / node[ng, bodyID2].ms
                node[ng, bodyID2].momentums += cforce / node[ng, bodyID2].ms * dt[None]


@ti.kernel
def kernel_calc_normal_demcontact_semi_2D(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    matProps: ti.template(),
    grid_size: ti.types.vector(2, float),
    node: ti.template(),
    polygon_vertices: ti.template(),
    velocity: ti.types.vector(2, float),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    v1, v2 = ZEROVEC2f, ZEROVEC2f
    normal, tangential = ZEROVEC2f, ZEROVEC2f
    gsize = MeanValue(grid_size)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            circle_center = particle[np].x
            # Calculate minimum distance and normal vector
            distance, normal = circle_polygon_distance(circle_center, polygon_vertices)
            # Calculate particle traction using penalty factor
            if distance > 0:
                cforce = vec2f([0.0, 0.0])
            else:
                print("contact normal ", np, normal, distance)
                shear = 0.5 * 40.0e6 / (1.0 + 0.3)
                bulk = 40.0e6 / (3.0 * (1 - 2.0 * 0.3))
                vol = particle[np].vol
                beta = (bulk + 4.0 / 3.0 * shear) / gsize * ti.sqrt(vol)
                normal = Normalize(normal)
                nomforce = -distance * beta * normal * 100.0
                cforce = nomforce

                offset = np * total_nodes
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    extf = shape_mapping(shapefn[ln], cforce)
                    node[nodeID, bodyID]._update_contact_force_solid(extf)


@ti.kernel
def kernel_calc_tangential_demcontact_semi_2D(
    cut_off: float,
    mu: float,
    velocity: ti.types.vector(2, float),
    dt: ti.template(),
    node: ti.template(),
    nodal_coords: ti.template(),
    particle: ti.template(),
):
    # ti.block_local(dt)
    bodyID = bodyID = int(particle[0].bodyID)
    vr = velocity
    for ng in range(node.shape[0]):
        if node[ng, bodyID].m > cut_off:
            vd = node[ng, bodyID].momentums
            g_mass = node[ng, bodyID].ms / dt[None]
            norm_force = node[ng, bodyID].contact_force_s
            if norm_force.norm() > Threshold:
                cforce = ZEROVEC2f
                if mu > Threshold:
                    norm = Normalize(norm_force)
                    v_re = vr - vd
                    stick_force = g_mass * (v_re - v_re.dot(norm) * norm)
                    fstick = ti.sqrt(dot2(stick_force))
                    fslip = mu * ti.sqrt(dot2(norm_force))
                    if fslip < fstick:
                        cforce = fslip * (stick_force / fstick)
                    else:
                        cforce = stick_force
                    node[ng, bodyID]._update_contact_force_solid(cforce)


@ti.kernel
def kernel_apply_displacement_mixture(
    cut_off: float, dt: ti.template(), node: ti.template(), nodal_coords: ti.template()
):
    # ti.block_local(dt)
    v_constraint = vec2f([0.0, -0.05])
    for ng in range(node.shape[0]):
        bodyID = 0
        node_coord = nodal_coords[ng]
        if node_coord[0] <= 1.0 and node_coord[1] >= 9.99 and node_coord[1] <= 10.01:
            mf, vf = node[ng, bodyID].mf, node[ng, bodyID].momentumf
            if mf > cut_off:
                cforce = (v_constraint - vf) / dt[None]
                node[ng, bodyID]._update_contact_force_fluid(cforce)
                node[ng, bodyID].forcef += cforce
                node[ng, bodyID].momentumf += cforce * dt[None]

            ms, vs = node[ng, bodyID].ms, node[ng, bodyID].momentums
            if ms > cut_off:
                cforce = (v_constraint - vs) / dt[None]
                node[ng, bodyID]._update_contact_force_solid(cforce)
                node[ng, bodyID].forces += cforce
                node[ng, bodyID].momentums += cforce * dt[None]


@ti.kernel
def kernel_apply_node_traction(cut_off: float, dt: ti.template(), node: ti.template(), nodal_coords: ti.template()):
    # ti.block_local(dt)
    traction = vec2f([0.0, -1e4])
    for ng in range(node.shape[0]):
        bodyID = 0
        node_coord = nodal_coords[ng]
        if node_coord[0] <= 1.0 and node_coord[1] >= 9.99 and node_coord[1] <= 10.01:
            node[ng, bodyID].forces += traction * 0.1 * 0.5
            node[ng, bodyID].force += traction * 0.1 * 0.5


@ti.kernel
def kernel_update_nodal_pressure_2D(
    cutoff: float,
    beta: float,
    boundary_pressure: float,
    node: ti.template(),
    unknown_vector: ti.template(),
):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = node[ng, nb].dof
                if dof0 >= 0:
                    node[ng, nb]._update_nodal_dpressure(unknown_vector[dof0])
                else:
                    node[ng, nb]._update_nodal_dpressure(boundary_pressure - beta * node[ng, nb].pressure)


@ti.kernel
def kernel_limit_twophase_nodal_pressure_increment(
    cutoff: float,
    beta: float,
    minimum_pressure: float,
    node: ti.template(),
):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                node[ng, nb].dpressure = ti.max(
                    node[ng, nb].dpressure,
                    minimum_pressure - beta * node[ng, nb].pressure,
                )


@ti.kernel
def kernel_limit_twophase_cell_pressure_increment(
    beta: float,
    minimum_pressure: float,
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    cell_dpressure: ti.template(),
):
    for I in ti.grouped(cell_type):
        if cell_type[I] == 1:
            cell_dpressure[I] = ti.max(cell_dpressure[I], minimum_pressure - beta * cell_pressure[I])


@ti.func
def single_point_ghost_air_pressure(air_cell, cell_type, cell_pressure, cell_phi):
    dimension = ti.static(GlobalVariable.DIMENSION)
    cnum = ti.Vector.zero(int, dimension)
    for d in ti.static(range(dimension)):
        cnum[d] = cell_type.shape[d]
    pressure = 0.0
    count = 0.0
    for d in ti.static(range(dimension)):
        for s in ti.static((-1, 1)):
            fluid = air_cell + s * ti.Vector.unit(dimension, d)
            if is_mg_valid_cell(fluid, cnum) and is_mg_fluid_cell(fluid, cell_type):
                theta = free_surface_theta(fluid, air_cell, cell_phi)
                pressure -= (1.0 - theta) / theta * cell_pressure[fluid]
                count += 1.0
    if count > 0.0:
        pressure /= count
    return pressure


@ti.kernel
def kernel_update_nodal_dpressure_2D(
    node: ti.template(),
    element_size: ti.types.vector(2, float),
    shape_function: ti.template(),
    gnum: ti.types.vector(2, int),
    cell_type: ti.template(),
    cell_volume: ti.template(),
    cell_dpressure: ti.template(),
    is_free_surface: ti.template(),
    cell_phi: ti.template(),
    is_rigid: ti.template(),
):
    node.weight.fill(0.0)
    node.dpressure.fill(0.0)
    bodyID = 0 if is_rigid[0] == 0 else 1
    for I in ti.grouped(cell_volume):
        # if cell_volume[I] > Threshold:
        if cell_type[I] == 1 or is_free_surface[I] == 1:
            i, j = I
            position = vec2f(i + 0.5, j + 0.5) * element_size
            psize = vec2f(0, 0)
            base_bound = ti.floor((position - psize) / element_size, int)
            influenced_node = 2
            LnID = vec4i([0, 0, 0, 0])
            shape_fn = vec4f([0.0, 0.0, 0.0, 0.0])
            localID = 0
            for lj in range(base_bound[1], base_bound[1] + influenced_node):
                if lj < 0 or lj >= gnum[1]:
                    continue
                for li in range(base_bound[0], base_bound[0] + influenced_node):
                    if li < 0 or li >= gnum[0]:
                        continue
                    nodeID = int(li + lj * gnum[0])
                    node_coords = vec2i(li, lj) * element_size
                    shapen0 = shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                    shapen1 = shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                    shapeval = shapen0 * shapen1
                    if shapeval > Threshold:
                        LnID[localID] = nodeID
                        shape_fn[localID] = shapeval
                        localID += 1
            # vol = cell_volume[I]
            vol = element_size[0] * element_size[1]
            total_nodes = influenced_node * influenced_node
            dpressure = cell_dpressure[I]
            if cell_type[I] == 0:
                dpressure = single_point_ghost_air_pressure(I, cell_type, cell_dpressure, cell_phi)
            for lnj in range(total_nodes):
                nodeIDj = LnID[lnj]
                jshape = shape_fn[lnj]
                node[nodeIDj, bodyID].weight += vol * jshape
                node[nodeIDj, bodyID].dpressure += dpressure * vol * jshape


@ti.kernel
def kernel_update_nodal_dpressure_2D_(cutoff: float, node: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].weight > cutoff:
                node[ng, nb].dpressure /= node[ng, nb].weight


@ti.kernel
def kernel_update_nodal_dpressure_3D(
    node: ti.template(),
    element_size: ti.types.vector(3, float),
    shape_function: ti.template(),
    gnum: ti.types.vector(3, int),
    cell_type: ti.template(),
    cell_volume: ti.template(),
    cell_dpressure: ti.template(),
    is_free_surface: ti.template(),
    cell_phi: ti.template(),
    is_rigid: ti.template(),
):
    node.weight.fill(0.0)
    node.dpressure.fill(0.0)
    bodyID = 0 if is_rigid[0] == 0 else 1
    for I in ti.grouped(cell_volume):
        if cell_type[I] == 1 or is_free_surface[I] == 1:
            i, j, k = I
            position = vec3f(i + 0.5, j + 0.5, k + 0.5) * element_size
            psize = vec3f(0, 0, 0)
            base_bound = ti.floor((position - psize) / element_size, int)
            influenced_node = 2
            LnID = vec8i([0, 0, 0, 0, 0, 0, 0, 0])
            shape_fn = vec8f([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            localID = 0
            for lk in range(base_bound[2], base_bound[2] + influenced_node):
                if lk < 0 or lk >= gnum[2]:
                    continue
                for lj in range(base_bound[1], base_bound[1] + influenced_node):
                    if lj < 0 or lj >= gnum[1]:
                        continue
                    for li in range(base_bound[0], base_bound[0] + influenced_node):
                        if li < 0 or li >= gnum[0]:
                            continue
                        nodeID = linearize(vec3i(li, lj, lk), gnum)
                        node_coords = vec3f(li, lj, lk) * element_size
                        shapeval = (
                            shape_function(position[0], node_coords[0], 1.0 / element_size[0], psize[0])
                            * shape_function(position[1], node_coords[1], 1.0 / element_size[1], psize[1])
                            * shape_function(position[2], node_coords[2], 1.0 / element_size[2], psize[2])
                        )
                        if shapeval > Threshold:
                            LnID[localID] = nodeID
                            shape_fn[localID] = shapeval
                            localID += 1
            vol = element_size[0] * element_size[1] * element_size[2]
            dpressure = cell_dpressure[I]
            if cell_type[I] == 0:
                dpressure = single_point_ghost_air_pressure(I, cell_type, cell_dpressure, cell_phi)
            for lnj in range(influenced_node * influenced_node * influenced_node):
                nodeIDj = LnID[lnj]
                jshape = shape_fn[lnj]
                node[nodeIDj, bodyID].weight += vol * jshape
                node[nodeIDj, bodyID].dpressure += dpressure * vol * jshape


@ti.kernel
def calculate_cell_volume(
    cell_volumefrac: ti.template(),
    particleNum: int,
    particle: ti.template(),
    cell_volume: float,
    grid_size: ti.template(),
    cnum: ti.template(),
    is_2DAxisy: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            vol = particle[np].vol
            position = particle[np].x
            ic = ti.floor((position / grid_size), int)
            valid = True
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                valid = valid and 0 <= ic[d] < cnum[d]
            if not valid:
                continue
            icell = linearize(ic, cnum)
            cell_volume_ = cell_volume
            if ti.static(GlobalVariable.DIMENSION == 2):
                if is_2DAxisy:
                    cell_volume_ = cell_volume * (ic[0] + 0.5) * grid_size[0]
            cell_volumefrac[icell] += vol / cell_volume_


@ti.kernel
def calculate_cell_volume_weighted(
    cell_volumefrac: ti.template(),
    particleNum: int,
    ghost_cell: int,
    particle: ti.template(),
    cell_volume: float,
    grid_size: ti.template(),
    cnum: ti.template(),
    is_2DAxisy: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            position = particle[np].x
            base = ti.floor(position / grid_size - 0.5).cast(int)
            if ti.static(GlobalVariable.DIMENSION == 2):
                for i, j in ti.static(ti.ndrange(2, 2)):
                    cell = base + ti.Vector([i, j])
                    if 0 <= cell[0] < active_cnum[0] and 0 <= cell[1] < active_cnum[1]:
                        center = (cell.cast(float) + 0.5) * grid_size
                        weight = ti.max(0.0, 1.0 - ti.abs(position[0] - center[0]) / grid_size[0]) * ti.max(
                            0.0, 1.0 - ti.abs(position[1] - center[1]) / grid_size[1]
                        )
                        cell_measure = cell_volume
                        if is_2DAxisy:
                            cell_measure *= (cell[0] + 0.5) * grid_size[0]
                        ti.atomic_add(cell_volumefrac[linearize(cell, cnum)], weight * particle[np].vol / cell_measure)
            elif ti.static(GlobalVariable.DIMENSION == 3):
                for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                    cell = base + ti.Vector([i, j, k])
                    if (
                        0 <= cell[0] < active_cnum[0]
                        and 0 <= cell[1] < active_cnum[1]
                        and 0 <= cell[2] < active_cnum[2]
                    ):
                        center = (cell.cast(float) + 0.5) * grid_size
                        weight = (
                            ti.max(0.0, 1.0 - ti.abs(position[0] - center[0]) / grid_size[0])
                            * ti.max(0.0, 1.0 - ti.abs(position[1] - center[1]) / grid_size[1])
                            * ti.max(0.0, 1.0 - ti.abs(position[2] - center[2]) / grid_size[2])
                        )
                        ti.atomic_add(cell_volumefrac[linearize(cell, cnum)], weight * particle[np].vol / cell_volume)


@ti.kernel
def calculate_cell_volume_fraction(
    cell_volumefrac: ti.template(),
    particleNum: int,
    particle: ti.template(),
    cell_volume: float,
    grid_size: ti.template(),
    cnum: ti.template(),
    cell_rigid: ti.template(),
):
    cell_rigid.fill(0)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            position = particle[np].x
            ic = ti.floor((position / grid_size), int)
            valid = True
            for d in ti.static(range(GlobalVariable.DIMENSION)):
                valid = valid and 0 <= ic[d] < cnum[d]
            if not valid:
                continue
            icell = linearize(ic, cnum)
            vol = particle[np].vol
            if int(particle[np].materialID) > 0:
                cell_volumefrac[icell] += vol / cell_volume
            else:
                cell_rigid[ic] = 1


@ti.kernel
def calculate_cell_volume_fraction_2DAxisy_plane(
    cell_volumefrac: ti.template(),
    particleNum: int,
    particle: ti.template(),
    cell_volume: float,
    grid_size: ti.types.vector(2, float),
    cnum: ti.types.vector(2, int),
    cell_rigid: ti.template(),
    axis_offset: float,
):
    cell_rigid.fill(0)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            position = particle[np].x
            ic = ti.floor((position / grid_size), int)
            if ic[0] < 0 or ic[0] >= cnum[0] or ic[1] < 0 or ic[1] >= cnum[1]:
                continue
            icell = ic[0] + ic[1] * cnum[0]
            vol = particle[np].area
            if int(particle[np].materialID) > 0:
                cell_volumefrac[icell] += vol / cell_volume
            else:
                cell_rigid[ic] = 1
