import taichi as ti

from src.utils.constants import Threshold
from src.utils.TypeDefination import vec2f, vec3f, vec3i
from src.utils.VectorFunction import SquareLen
from src.utils.ScalarFunction import linearize


@ti.kernel
def kernel_initialize_boundary(constraint: ti.template()):
    for i in constraint:
        constraint[i].clear_boundary_condition() 


@ti.kernel
def add_boundary_flags(level: int, gridSum: int, inodes: ti.types.ndarray(), boundary_flag: ti.template()):
    for offset in range(inodes.shape[0]):
        boundary_flag[inodes[offset] + level * gridSum] = 1


@ti.kernel
def set_velocity_constraint(dimension: int, lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), active_direction: ti.types.vector(3, int), velocity: ti.types.vector(3, float), level: int, total_dofs: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + total_dofs * offset
        for d in range(dimension):
            if active_direction[d] == 1:
                constraint[locate].set_boundary_condition(inodes[offset], level, d, velocity[d])
                locate += 1
    lists[0] += total_dofs * inodes.shape[0]


@ti.kernel
def set_reflection_constraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), direction: int, signs: int, level: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        constraint[locate].set_boundary_condition(inodes[offset], level, direction, signs)
    lists[0] += inodes.shape[0]


@ti.kernel
def set_friction_constraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), mu: float, direction: int, signs: int, level: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        constraint[locate].set_boundary_condition(inodes[offset], level, mu, direction, signs)
    lists[0] += inodes.shape[0]


@ti.kernel
def set_absorbing_contraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), level: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
    lists[0] += inodes.shape[0]


@ti.kernel
def set_traction_contraint(dimension: int, lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), traction: ti.types.vector(3, float), dof: ti.types.vector(3, int), direction: int, level: int, fix_dofs: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + fix_dofs * offset
        for d in range(dimension):
            if dof[d] == 1:
                constraint[locate].set_boundary_condition(inodes[offset], level, traction[d], direction)
    lists[0] += fix_dofs * inodes.shape[0]


@ti.kernel
def set_displacement_contraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), dof: ti.types.vector(3, int), value: ti.types.vector(3, float), level: int, fix_dofs: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + fix_dofs * offset
        count = 0
        for j in ti.static(range(3)):
            if dof[j] == 1:
                constraint[locate + count].set_boundary_condition(inodes[offset], level, j, value[j])
                count += 1
    lists[0] += inodes.shape[0] * fix_dofs


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
        constraints[nboundary]._calc_psize_cp(dt, particle[particleID].velocity_gradient)
        exts, extf = constraints[nboundary]._compute_traction_force()
        offset = particleID * total_nodes
        for ln in range(offset, offset + node_size[particleID]):
            nodeID = LnID[ln]
            shapefn = shape_fn[ln]
            node[nodeID, bodyID]._update_nodal_force(shapefn * exts, shapefn * extf)


@ti.func
def is_inactive_neighbor_cell(c, cell, cnum):
    return cell[c + linearize(vec3i(-1, -1, -1), cnum)] == 0 or cell[c + linearize(vec3i(0, -1, -1), cnum)] == 0 or cell[c + linearize(vec3i(1, -1, -1), cnum)] == 0 or \
           cell[c + linearize(vec3i(-1, 0, -1), cnum)] == 0 or cell[c + linearize(vec3i(0, 0, -1), cnum)] == 0 or cell[c + linearize(vec3i(1, 0, -1), cnum)] == 0 or \
           cell[c + linearize(vec3i(-1, 1, -1), cnum)] == 0 or cell[c + linearize(vec3i(0, 1, -1), cnum)] == 0 or cell[c + linearize(vec3i(1, 1, -1), cnum)] == 0 or \
           cell[c + linearize(vec3i(-1, -1, 0), cnum)] == 0 or cell[c + linearize(vec3i(0, -1, 0), cnum)] == 0 or cell[c + linearize(vec3i(1, -1, 0), cnum)] == 0 or \
           cell[c + linearize(vec3i(-1, 0, 0), cnum)] == 0 or cell[c + linearize(vec3i(0, 0, 0), cnum)] == 0 or cell[c + linearize(vec3i(1, 0, 0), cnum)] == 0 or \
           cell[c + linearize(vec3i(-1, 1, 0), cnum)] == 0 or cell[c + linearize(vec3i(0, 1, 0), cnum)] == 0 or cell[c + linearize(vec3i(1, 1, 0), cnum)] == 0 or \
           cell[c + linearize(vec3i(-1, -1, 1), cnum)] == 0 or cell[c + linearize(vec3i(0, -1, 1), cnum)] == 0 or cell[c + linearize(vec3i(1, -1, 1), cnum)] == 0 or \
           cell[c + linearize(vec3i(-1, 0, 1), cnum)] == 0 or cell[c + linearize(vec3i(0, 0, 1), cnum)] == 0 or cell[c + linearize(vec3i(1, 0, 1), cnum)] == 0 or \
           cell[c + linearize(vec3i(-1, 1, 1), cnum)] == 0 or cell[c + linearize(vec3i(0, 1, 1), cnum)] == 0 or cell[c + linearize(vec3i(1, 1, 1), cnum)] == 0

@ti.kernel
def apply_particle_virtual_traction_constraint(lists: int, total_nodes: int, particleNum: int, grid_size: ti.types.vector(3, float), cnum: ti.types.vector(3, int), constraints: ti.template(), dt: ti.template(), cell: ti.template(), 
                                               node: ti.template(), particle: ti.template(), LnID: ti.template(), shape_fn: ti.template(), dshape_fn: ti.template(), node_size: ti.template()):
    ti.loop_config(bit_vectorize=True)
    for c in cell:
        cell[c] = 0

    for np in range(particleNum):
        cellID = ti.cast(particle[np].x / grid_size, int)
        linear_cellID = linearize(cellID, cnum)
        cell[linear_cellID] = 1

    ti.loop_config(bit_vectorize=True)
    for c in cell:
        if is_inactive_neighbor_cell(c, cell, cnum):
            for d in ti.static(8):
                pass
    
    for nboundary in range(lists):
        particleID = constraints[nboundary].pid
        bodyID = int(particle[particleID].bodyID)
        constraints[nboundary]._calc_psize_cp(dt, particle[particleID].velocity_gradient)
        traction = constraints[nboundary]._compute_traction_force()
        offset = particleID * total_nodes
        for ln in range(offset, offset + node_size[particleID]):
            nodeID = LnID[ln]
            node[nodeID, bodyID]._update_nodal_force(shape_fn[ln] * traction)