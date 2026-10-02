import taichi as ti

from src.utils.constants import Threshold
from src.utils.TypeDefination import vec2f, vec3f, vec3i, vec2i
from src.utils.VectorFunction import SquareLen
from src.utils.ScalarFunction import linearize, vectorize_id, clamp
from src.utils.ShapeFunctions import ShapeLinear, GShapeLinear, ShapeLinearCenter, ShapeGIMP, GShapeGIMP, ShapeGIMPCenter, ShapeBsplineQ, GShapeBsplineQ, ShapeBsplineC, GShapeBsplineC
import src.utils.GlobalVariable as GlobalVariable


@ti.kernel
def kernel_initialize_boundary(constraint: ti.template()):
    for i in constraint:
        constraint[i].clear_boundary_condition() 


@ti.kernel
def add_boundary_flags(level: int, gridSum: int, inodes: ti.types.ndarray(), boundary_flag: ti.template()):
    for offset in range(inodes.shape[0]):
        boundary_flag[inodes[offset] + level * gridSum] = 1


@ti.kernel
def set_reflection_constraint(constraint: ti.template(), nodeID: ti.types.ndarray(), levels: ti.types.ndarray(), dirs: ti.types.ndarray(), signs: ti.types.ndarray()):
     for offset in range(nodeID.shape[0]):
        constraint[offset].set_boundary_condition(nodeID[offset], levels[offset], dirs[offset], signs[offset])


@ti.kernel
def set_friction_constraint(constraint: ti.template(), nodeID: ti.types.ndarray(), levels: ti.types.ndarray(), dirs: ti.types.ndarray(), signs: ti.types.ndarray(), mu: ti.types.ndarray()):
    for offset in range(nodeID.shape[0]):
        constraint[offset].set_boundary_condition(nodeID[offset], levels[offset], mu[offset], dirs[offset], signs[offset])


@ti.kernel
def set_contraints(constraint: ti.template(), nodeID: ti.types.ndarray(), levels: ti.types.ndarray(), dirs: ti.types.ndarray(), values: ti.types.ndarray()):
    for offset in range(nodeID.shape[0]):
        constraint[offset].set_boundary_condition(nodeID[offset], levels[offset], dirs[offset], values[offset])


@ti.func
def find_pre_location(start_index, pid, constraint):
    locate = -1
    for pre in range(start_index):
        if pid == constraint[pre].pid:
            locate = pre
            break
    return locate


@ti.kernel
def prefind_particle_traction_contraint(lists: ti.types.ndarray(), constraint: ti.template(), startNum: int, particleNum: int, particle: ti.template(), is_in_region: ti.template()) -> int:
    start_index = lists[0]
    pre_number = lists[0]
    for np in range(startNum, startNum + particleNum):
        if is_in_region(particle[np].x):
            temp = find_pre_location(pre_number, np, constraint)
            if temp == -1:
                temp = ti.atomic_add(start_index, 1)
    return start_index


@ti.kernel
def set_particle_traction_contraint(lists: ti.types.ndarray(), constraint: ti.template(), startNum: int, particleNum: int, particle: ti.template(), is_in_region: ti.template(), value: ti.types.vector(3, float), psize: ti.types.ndarray()):
    start_index = lists[0]
    pre_number = lists[0]
    for np in range(startNum, startNum + particleNum):
        if is_in_region(particle[np].x):
            temp = find_pre_location(pre_number, np, constraint)
            if temp == -1:
                temp = ti.atomic_add(start_index, 1)
            constraint[temp].set_boundary_condition(np, value, vec3f(psize[np, 0], psize[np, 1], psize[np, 2]))
    lists[0] = start_index


@ti.kernel
def set_particle_traction_contraint_2D(lists: ti.types.ndarray(), constraint: ti.template(), startNum: int, particleNum: int, particle: ti.template(), is_in_region: ti.template(), value: ti.types.vector(2, float), psize: ti.types.ndarray()):
    start_index = lists[0]
    pre_number = lists[0]
    for np in range(startNum, startNum + particleNum):
        if is_in_region(particle[np].x):
            temp = find_pre_location(pre_number, np, constraint)
            if temp == -1:
                temp = ti.atomic_add(start_index, 1)
            constraint[temp].set_boundary_condition(np, value, 0.25 * particle[np].vol / vec2f(psize[np, 1], psize[np, 0]))
    lists[0] = start_index


@ti.kernel
def set_particle_traction_contraint_twophase_2D(lists: ti.types.ndarray(), constraint: ti.template(), startNum: int, particleNum: int, particle: ti.template(), is_in_region: ti.template(), value: ti.types.vector(2, float), pvalue: ti.types.vector(2, float), psize: ti.types.ndarray()):
    start_index = lists[0]
    pre_number = lists[0]
    for np in range(startNum, startNum + particleNum):
        if is_in_region(particle[np].x):
            temp = find_pre_location(pre_number, np, constraint)
            if temp == -1:
                temp = ti.atomic_add(start_index, 1)
            constraint[temp].set_boundary_condition(np, value, pvalue, vec2f(psize[np, 0], psize[np, 1]))
    lists[0] = start_index


@ti.kernel
def clear_constraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), nlevel: int):
    for nodeID in range(inodes.shape[0]):
        for offset in range(lists[0]):
            if constraint[offset].node == inodes[nodeID] and constraint[offset].level == nlevel:
                constraint[offset].clear_boundary_condition()


@ti.kernel
def clear_displacement_constraint(constraint: ti.template(), inodes: ti.types.ndarray(), level: int):
    for i in range(inodes.shape[0]):
        constraint[3 * inodes[i], level].dof = ti.u8(0)
        constraint[3 * inodes[i] + 1, level].dof = ti.u8(0)
        constraint[3 * inodes[i] + 2, level].dof = ti.u8(0)


@ti.kernel
def copy_valid_constraint(lists: ti.types.ndarray(), constraint: ti.template()):   
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        if constraint[nboundary].level != ti.u8(255):
            constraint[remain_num] = constraint[nboundary]
            if remain_num < nboundary:
                constraint[nboundary].clear_boundary_condition()
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def apply_velocity_constraint(cut_off: float, lists: int, constraints: ti.template(), is_rigid: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if node[nodeID, grid_level].m > cut_off:
            direction = int(constraints[nboundary].dirs)

            if is_rigid[grid_level] == 0:
                prescribed_velocity = constraints[nboundary].velocity
                node[nodeID, grid_level].velocity_constraint(direction, prescribed_velocity)
            else:
                node[nodeID, grid_level].rigid_body_velocity_constraint(direction)


@ti.kernel
def apply_reflection_constraint(cut_off: float, lists: int, constraints: ti.template(), is_rigid: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if node[nodeID, grid_level].m > cut_off:
            direction = int(constraints[nboundary].dirs)
            signs = int(constraints[nboundary].signs)
            if is_rigid[grid_level] == 0:
                node[nodeID, grid_level].reflection_constraint(direction, signs)
            else:
                node[nodeID, grid_level].rigid_body_reflection_constraint(direction, signs)


@ti.kernel
def apply_contact_velocity_constraint(cut_off: float, lists: int, constraints: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if SquareLen(node[nodeID, grid_level].contact_force) > cut_off:
            direction = int(constraints[nboundary].dirs)
            node[nodeID, grid_level].contact_velocity_constraint(direction)


@ti.kernel
def apply_contact_reflection_constraint(cut_off: float, lists: int, constraints: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)

        if SquareLen(node[nodeID, grid_level].contact_force) > cut_off:
            direction = int(constraints[nboundary].dirs)
            signs = int(constraints[nboundary].signs)
            node[nodeID, grid_level].contact_reflection_constraint(direction, signs)


@ti.kernel
def apply_friction_constraint(cut_off: float, lists: int, constraints: ti.template(), is_rigid: ti.template(), node: ti.template(), dt: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if node[nodeID, grid_level].m > cut_off:
            direction = int(constraints[nboundary].dirs)
            signs = int(constraints[nboundary].signs)
            
            if is_rigid[grid_level] == 0:
                mu = constraints[nboundary].mu
                node[nodeID, grid_level].friction_constraint(mu, direction, signs, dt)
            else:
                node[nodeID, grid_level].rigid_friction_constraint(direction, signs)


@ti.kernel
def apply_absorbing_constraint(lists: int, constraints: ti.template(), material: ti.template(), node: ti.template(), extra_node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        norm = constraints[nboundary].norm
        delta = constraints[nboundary].delta
        h_min = constraints[nboundary].h_min
        a = constraints[nboundary].a
        b = constraints[nboundary].b
        position_type = constraints[nboundary].position_type

        mass = node[nodeID, grid_level].mass
        pwave_v = node[nodeID, grid_level].pwave_velocity
        swave_v = node[nodeID, grid_level].swave_velocity
        density = material[grid_level].density
        material_displacement = extra_node[nodeID, grid_level].displacement

        pwave_v /= mass
        swave_v /= mass
        density /= mass
        material_displacement /= mass

        wave_velocity = b * swave_v
        wave_velocity -= (wave_velocity.dot(norm) - a * pwave_v) * norm

        k_s = density * swave_v * swave_v / delta
        k_p = density * pwave_v * pwave_v / delta
        spring_constant = vec3f(k_s, k_s, k_s)
        spring_constant -= (spring_constant.dot(norm) - k_p) * norm

        absorbing_traction = node[nodeID, grid_level].momentum * wave_velocity * density + material_displacement * spring_constant
        if position_type == 0:
            absorbing_traction *= 0.25 * h_min * h_min 
        elif position_type == 1:
            absorbing_traction *= 0.5 * h_min * h_min 
        elif position_type == 2:
            absorbing_traction *= h_min * h_min 
        node[nodeID, grid_level]._update_nodal_force(-absorbing_traction)
            

@ti.kernel
def apply_traction_constraint(lists: int, constraints: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if node[nodeID, grid_level].m > Threshold:
            direction = int(constraints[nboundary].dirs)
            traction = constraints[nboundary].traction
            node[nodeID, grid_level].force += ti.Vector([(direction == j) * traction for j in ti.static(range(3))], float) 


@ti.kernel
def apply_traction_constraint_2D(lists: int, constraints: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if node[nodeID, grid_level].m > Threshold:
            direction = int(constraints[nboundary].dirs)
            traction = constraints[nboundary].traction
            node[nodeID, grid_level].force += ti.Vector([(direction == j) * traction for j in ti.static(range(2))], float) 


@ti.kernel
def lightweight_particle_traction_constraint(lists: int, gnum: ti.types.vector(GlobalVariable.DIMENSION, int), grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float), 
                                             constraints: ti.template(), dt: ti.template(), particle_lengths: ti.template(), boundary_types: ti.template(), node: ti.template(), particle: ti.template()):
    for d in ti.static(range(GlobalVariable.DIMENSION)): ti.block_local(node.force.get_scalar_field(d))
    for nboundary in range(lists):
        particleID = constraints[nboundary].pid
        bodyID = int(particle[particleID].bodyID)
        constraints[nboundary]._calc_psize_cp(dt, particle[particleID].velocity_gradient)
        position = particle[particleID].x
        psize = particle_lengths[bodyID]

        external_force = constraints[nboundary]._compute_traction_force()
        for offset in ti.static(ti.grouped(ti.ndrange(*((GlobalVariable.INFLUENCENODE, ) * GlobalVariable.DIMENSION)))):
            base = ti.floor((position - psize) * igrid_size).cast(int)
            grid_id = base + offset
            if all(grid_id >= 0):
                nodeID = linearize(grid_id, gnum)
                grid_pos = grid_id * grid_size
                shape_fn = ti.Vector.zero(float, GlobalVariable.DIMENSION)
                if ti.static(GlobalVariable.SHAPEFUNCTION == 0): 
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        shape_fn[d] = ShapeLinear(position[d], grid_pos[d], igrid_size[d], 0)
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 1): 
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        shape_fn[d] = ShapeGIMP(position[d], grid_pos[d], igrid_size[d], psize[d])
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 2): 
                    boundary_type = boundary_types[nodeID, bodyID]
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        btypes = int(boundary_type[d])
                        shape_fn[d] = ShapeBsplineQ(position[d], grid_pos[d], igrid_size[d], btypes)
                elif ti.static(GlobalVariable.SHAPEFUNCTION == 3): 
                    boundary_type = boundary_types[nodeID, bodyID]
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        btypes = int(boundary_type[d])
                        shape_fn[d] = ShapeBsplineC(position[d], grid_pos[d], igrid_size[d], btypes)

                weight = 1.0
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    weight *= shape_fn[d]
                pforce = weight * external_force
                node[nodeID, bodyID].force += pforce


@ti.kernel
def apply_particle_traction_constraint(lists: int, total_nodes: int, constraints: ti.template(), dt: ti.template(), node: ti.template(), particle: ti.template(), 
                                       LnID: ti.template(), shape_fn: ti.template(), node_size: ti.template()):
    for nboundary in range(lists):
        particleID = constraints[nboundary].pid
        bodyID = int(particle[particleID].bodyID)
        constraints[nboundary]._calc_psize_cp(dt, particle[particleID].velocity_gradient)
        traction = constraints[nboundary]._compute_traction_force()
        offset = particleID * total_nodes
        for ln in range(offset, offset + node_size[particleID]):
            nodeID = LnID[ln]
            node[nodeID, bodyID]._update_nodal_force(shape_fn[ln] * traction)


@ti.kernel
def apply_particle_traction_constraint_twophase(lists: int, total_nodes: int, constraints: ti.template(), dt: ti.template(), node: ti.template(), particle: ti.template(), 
                                       LnID: ti.template(), shape_fn: ti.template(), node_size: ti.template()):
    for nboundary in range(lists):
        particleID = constraints[nboundary].pid
        bodyID = int(particle[particleID].bodyID)
        constraints[nboundary]._calc_psize_cp(dt, particle[particleID].solid_velocity_gradient)
        exts, extf = constraints[nboundary]._compute_traction_force()
        offset = particleID * total_nodes
        for ln in range(offset, offset + node_size[particleID]):
            nodeID = LnID[ln]
            shapefn = shape_fn[ln]
            node[nodeID, bodyID]._update_nodal_force(shapefn * exts, shapefn * extf)


@ti.func
def is_subset_neighbor_cell(cell_id, body_id, cell, cnum):
    return int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, -1, -1)), cnum), body_id]) == 2 or int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, -1, -1)), cnum), body_id]) == 2 or int(cell[linearize(clamp(0, cnum, cell_id + vec3i(1, -1, -1)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 0, -1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, 0, -1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(1, 0, -1)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 1, -1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, 1, -1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(1, 1, -1)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, -1, 0)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, -1, 0)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(1, -1, 0)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 0, 0)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(1, 0, 0)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 1, 0)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, 1, 0)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(1, 1, 0)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, -1, 1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, -1, 1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(1, -1, 1)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 0, 1)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, 0, 1)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(1, 0, 1)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 1, 1)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, 1, 1)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(1, 1, 1)), cnum), body_id]) == 2

@ti.func
def is_boundary_nodes(cell_id, body_id, cell, cnum):
    return int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, -1, -1)), cnum), body_id]) == 2 or int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, -1, -1)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, -1, -1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, -1, -1)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 0, -1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 0, -1)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, 0, -1)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, 0, -1)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, -1, 0)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, -1, 0)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, -1, 0)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, -1, 0)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 0, 0)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec3i(-1, 0, 0)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, 0, 0)), cnum), body_id]) == 2 or    int(cell[linearize(clamp(0, cnum, cell_id + vec3i(0, 0, 0)), cnum), body_id]) == 3 

@ti.func
def is_subset_neighbor_cell_2D(cell_id, body_id, cell, cnum):
    return int(cell[linearize(clamp(0, cnum, cell_id + vec2i(-1, -1)), cnum), body_id]) == 2 or int(cell[linearize(clamp(0, cnum, cell_id + vec2i(0, -1)), cnum), body_id]) == 2 or int(cell[linearize(clamp(0, cnum, cell_id + vec2i(1, -1)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec2i(-1, 0)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec2i(1, 0)), cnum), body_id]) == 2 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec2i(-1, 1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec2i(0, 1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec2i(1, 1)), cnum), body_id]) == 2

@ti.func
def is_boundary_nodes_2D(cell_id, body_id, cell, cnum):
    return int(cell[linearize(clamp(0, cnum, cell_id + vec2i(-1, -1)), cnum), body_id]) == 2 or int(cell[linearize(clamp(0, cnum, cell_id + vec2i(-1, -1)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec2i(0, -1)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec2i(0 -1)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec2i(-1, 0)), cnum), body_id]) == 2 or  int(cell[linearize(clamp(0, cnum, cell_id + vec2i(-1, 0)), cnum), body_id]) == 3 or \
           int(cell[linearize(clamp(0, cnum, cell_id + vec2i(0, 0)), cnum), body_id]) == 2 or   int(cell[linearize(clamp(0, cnum, cell_id + vec2i(0, 0)), cnum), body_id]) == 3 

@ti.kernel
def apply_particle_virtual_traction_constraint_2D(particleNum: int, grid_size: ti.types.vector(2, float), cnum: ti.types.vector(2, int), 
                                               auxiliary_cell: ti.template(), auxiliary_node: ti.template(), node: ti.template(), particle: ti.template()):
    auxiliary_cell.fill(0)
    auxiliary_node.fill(0)
    for np in range(particleNum):
        cellID = ti.cast(particle[np].x / grid_size, int)
        bodyID = particle[np].bodyID
        linear_cellID = linearize(cellID, cnum)
        auxiliary_cell[linear_cellID, bodyID] = 1

    for nc, nb in auxiliary_cell:
        if int(auxiliary_cell[nc, nb]) == 0:
            for ndx, ndy in ti.static(ti.ndrange(2, 2)):
                gridID = linearize(vec2i(vectorize_id(nc, cnum)) + vec2i(ndx, ndy), cnum + 1)
                if node[gridID, nb].m > Threshold:
                    auxiliary_cell[nc, nb] = 2

    for nc, nb in auxiliary_cell:
        if int(auxiliary_cell[nc, nb]) > 0:
            if is_subset_neighbor_cell_2D(vec2i(vectorize_id(nc, cnum)), nb, auxiliary_cell, cnum):
                auxiliary_cell[nc, nb] = 3

    for ng, nb in node:
        if node[ng, nb].m > Threshold:
            if is_boundary_nodes_2D(vec2i(vectorize_id(ng, cnum + 1)), nb, auxiliary_cell, cnum):
                auxiliary_node[ng, nb] = 1

@ti.kernel
def apply_particle_virtual_traction_constraint(particleNum: int, grid_size: ti.types.vector(3, float), cnum: ti.types.vector(3, int), 
                                               auxiliary_cell: ti.template(), auxiliary_node: ti.template(), node: ti.template(), particle: ti.template()):
    auxiliary_cell.fill(0)
    auxiliary_node.fill(0)
    for np in range(particleNum):
        cellID = ti.cast(particle[np].x / grid_size, int)
        bodyID = particle[np].bodyID
        linear_cellID = linearize(cellID, cnum)
        auxiliary_cell[linear_cellID, bodyID] = 1

    for nc, nb in auxiliary_cell:
        if int(auxiliary_cell[nc, nb]) == 0:
            for ndx, ndy, ndz in ti.static(ti.ndrange(2, 2, 2)):
                gridID = linearize(vec3i(vectorize_id(nc, cnum)) + vec3i(ndx, ndy, ndz), cnum + 1)
                if node[gridID, nb].m > Threshold:
                    auxiliary_cell[nc, nb] = 2

    for nc, nb in auxiliary_cell:
        if int(auxiliary_cell[nc, nb]) == 1:
            if is_subset_neighbor_cell(vec3i(vectorize_id(nc, cnum)), nb, auxiliary_cell, cnum):
                auxiliary_cell[nc, nb] = 3

    for ng, nb in node:
        if node[ng, nb].m > Threshold:
            if is_boundary_nodes(vec3i(vectorize_id(ng, cnum + 1)), nb, auxiliary_cell, cnum):
                auxiliary_node[ng, nb] = 1

@ti.kernel
def apply_virtual_traction_field_2D(total_nodes: int, particleNum: int, virtual_force: ti.template(), virtual_stress: ti.template(), node: ti.template(), particle: ti.template(), 
                                 LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template(), auxiliary_node: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            position = particle[np].x
            volume = particle[np].vol
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fex = volume * virtual_force(position)
            fInt = volume * virtual_stress(position)
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                if int(auxiliary_node[nodeID, bodyID]) == 1:
                    dshape_fn = dshapefn[ln]
                    external_force = shapefn[ln] * fex
                    internal_force = vec2f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3],
                                            dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3]])
                    node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)

@ti.kernel
def apply_virtual_traction_field(total_nodes: int, particleNum: int, virtual_force: ti.template(), virtual_stress: ti.template(), node: ti.template(), particle: ti.template(), 
                                 LnID: ti.template(), shapefn: ti.template(), dshapefn: ti.template(), node_size: ti.template(), auxiliary_node: ti.template()):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            position = particle[np].x
            volume = particle[np].vol
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            fex = volume * virtual_force(position)
            fInt = volume * virtual_stress(position)
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                if int(auxiliary_node[nodeID, bodyID]) == 1:
                    dshape_fn = dshapefn[ln]
                    external_force = shapefn[ln] * fex
                    internal_force = vec3f([dshape_fn[0] * fInt[0] + dshape_fn[1] * fInt[3] + dshape_fn[2] * fInt[5],
                                            dshape_fn[1] * fInt[1] + dshape_fn[0] * fInt[3] + dshape_fn[2] * fInt[4],
                                            dshape_fn[2] * fInt[2] + dshape_fn[1] * fInt[4] + dshape_fn[0] * fInt[5]])
                    node[nodeID, bodyID]._update_nodal_force(external_force + internal_force)