import taichi as ti

from src.utils.constants import Threshold
from src.utils.TypeDefination import vec2f, vec3f
from src.utils.VectorFunction import SquareLen


@ti.func
def find_pre_location(start_index, level, inodes, constraint, locate: ti.template()): # type: ignore
    for pre in range(start_index):
        if inodes == constraint[pre].node and level == constraint[pre].level:
            locate = pre
            break


@ti.func
def find_pre_location_with_direction(start_index, level, norm, inodes, constraint, locate: ti.template()): # type: ignore
    for pre in range(start_index):
        if inodes == constraint[pre].node and level == constraint[pre].level and all(norm == constraint[pre].norm):
            locate = pre
            break


@ti.kernel
def kernel_initialize_boundary(constraint: ti.template()):# type: ignore
    for i in constraint:
        constraint[i].node = -1
        constraint[i].level = ti.u8(255)


@ti.kernel
def set_velocity_constraint2D(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), active_direction: ti.types.vector(2, int), velocity: ti.types.vector(2, float), level: int):# type: ignore
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        find_pre_location(start_index, level, inodes[offset], constraint, locate)
        constraint[locate].set_boundary_condition(inodes[offset], level, active_direction, velocity)
    lists[0] += inodes.shape[0]


@ti.kernel
def set_velocity_constraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), active_direction: ti.types.vector(3, int), velocity: ti.types.vector(3, float), level: int):# type: ignore
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        find_pre_location(start_index, level, inodes[offset], constraint, locate)
        constraint[locate].set_boundary_condition(inodes[offset], level, active_direction, velocity)
    lists[0] += inodes.shape[0]


@ti.kernel
def set_reflection_constraint2D(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), direction: ti.types.vector(2, float), level: int):# type: ignore
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        find_pre_location(start_index, level, inodes[offset], constraint, locate)
        constraint[locate].set_boundary_condition(inodes[offset], level, direction)
    lists[0] += inodes.shape[0]


@ti.kernel
def set_reflection_constraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), direction: ti.types.vector(3, float), level: int):# type: ignore
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        find_pre_location(start_index, level, inodes[offset], constraint, locate)
        constraint[locate].set_boundary_condition(inodes[offset], level, direction)
    lists[0] += inodes.shape[0]


@ti.kernel
def set_friction_constraint2D(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), mu: float, norm: ti.types.vector(2, float), level: int):# type: ignore
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        find_pre_location_with_direction(start_index, level, norm.normalized(), inodes[offset], constraint, locate)
        constraint[locate].set_boundary_condition(inodes[offset], level, mu, norm.normalized())
    lists[0] += inodes.shape[0]


@ti.kernel
def set_friction_constraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), mu: float, norm: ti.types.vector(3, float), level: int):# type: ignore
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        find_pre_location_with_direction(start_index, level, norm.normalized(), inodes[offset], constraint, locate)
        constraint[locate].set_boundary_condition(inodes[offset], level, mu, norm.normalized())
    lists[0] += inodes.shape[0]


@ti.kernel
def set_absorbing_contraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), level: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        find_pre_location(start_index, level, inodes[offset], constraint, locate)
    lists[0] += inodes.shape[0]


@ti.kernel
def set_traction_contraint2D(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), traction: ti.types.vector(2, float), level: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        find_pre_location(start_index, level, inodes[offset], constraint, locate)
        constraint[locate].set_boundary_condition(inodes[offset], level, traction)
    lists[0] += inodes.shape[0]


@ti.kernel
def set_traction_contraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), traction: ti.types.vector(3, float), level: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + offset
        find_pre_location(start_index, level, inodes[offset], constraint, locate)
        constraint[locate].set_boundary_condition(inodes[offset], level, traction)
    lists[0] += inodes.shape[0]


@ti.kernel
def set_displacement_contraint(lists: ti.types.ndarray(), constraint: ti.template(), inodes: ti.types.ndarray(), dof: ti.types.vector(3, int), value: ti.types.vector(3, float), level: int, fix_dofs: int):
    start_index = lists[0]
    for offset in range(inodes.shape[0]):
        locate = start_index + fix_dofs * offset
        count = 0
        for j in range(3):
            if dof[j] == 1:
                constraint[locate + count].set_boundary_condition(inodes[offset], level, j, value[j])
                count += 1
    lists[0] += inodes.shape[0] * fix_dofs


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
def copy_valid_velocity_constraint2D(lists: ti.types.ndarray(), constraint: ti.template()):   
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        if constraint[nboundary].level != ti.u8(255):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].level = constraint[nboundary].level
            constraint[remain_num].fix_v = constraint[nboundary].fix_v
            constraint[remain_num].unfix_v = constraint[nboundary].unfix_v
            constraint[remain_num].velocity = constraint[nboundary].velocity

            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].level = ti.u8(255)
                constraint[nboundary].velocity = vec2f(0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def copy_valid_velocity_constraint(lists: ti.types.ndarray(), constraint: ti.template()):   
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        if constraint[nboundary].level != ti.u8(255):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].level = constraint[nboundary].level
            constraint[remain_num].fix_v = constraint[nboundary].fix_v
            constraint[remain_num].unfix_v = constraint[nboundary].unfix_v
            constraint[remain_num].velocity = constraint[nboundary].velocity

            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].level = ti.u8(255)
                constraint[nboundary].velocity = vec3f(0, 0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def copy_valid_reflection_constraint2D(lists: ti.types.ndarray(), constraint: ti.template()):    
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        if constraint[nboundary].level != ti.u8(255):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].level = constraint[nboundary].level
            constraint[remain_num].norm1 = constraint[nboundary].norm1
            constraint[remain_num].norm2 = constraint[nboundary].norm2
            constraint[remain_num].norm3 = constraint[nboundary].norm3

            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].level = ti.u8(255)
                constraint[nboundary].norm1 = vec2f(0, 0)
                constraint[nboundary].norm2 = vec2f(0, 0)
                constraint[nboundary].norm3 = vec2f(0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def copy_valid_reflection_constraint(lists: ti.types.ndarray(), constraint: ti.template()):    
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        if constraint[nboundary].level != ti.u8(255):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].level = constraint[nboundary].level
            constraint[remain_num].norm1 = constraint[nboundary].norm1
            constraint[remain_num].norm2 = constraint[nboundary].norm2
            constraint[remain_num].norm3 = constraint[nboundary].norm3

            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].level = ti.u8(255)
                constraint[nboundary].norm1 = vec3f(0, 0, 0)
                constraint[nboundary].norm2 = vec3f(0, 0, 0)
                constraint[nboundary].norm3 = vec3f(0, 0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def copy_valid_friction_constraint2D(lists: ti.types.ndarray(), constraint: ti.template()):    
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        if constraint[nboundary].level != ti.u8(255):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].level = constraint[nboundary].level
            constraint[remain_num].mu = constraint[nboundary].mu
            constraint[remain_num].norm = constraint[nboundary].norm
            
            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].level = ti.u8(255)
                constraint[nboundary].mu = 0.
                constraint[nboundary].norm = vec2f(0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def copy_valid_friction_constraint(lists: ti.types.ndarray(), constraint: ti.template()):    
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        if constraint[nboundary].level != ti.u8(255):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].level = constraint[nboundary].level
            constraint[remain_num].mu = constraint[nboundary].mu
            constraint[remain_num].norm = constraint[nboundary].norm
            
            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].level = ti.u8(255)
                constraint[nboundary].mu = 0.
                constraint[nboundary].norm = vec3f(0, 0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def copy_valid_absorbing_constraint(lists: ti.types.ndarray(), constraint: ti.template()):    
    pass


@ti.kernel
def copy_valid_traction_constraint2D(lists: ti.types.ndarray(), constraint: ti.template()):    
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        if constraint[nboundary].level != ti.u8(255):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].level = constraint[nboundary].level
            constraint[remain_num].traction = constraint[nboundary].traction
            
            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].level = ti.u8(255)
                constraint[nboundary].traction = vec2f(0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def copy_valid_traction_constraint(lists: ti.types.ndarray(), constraint: ti.template()):    
    remain_num = 0
    ti.loop_config(serialize=True)
    for nboundary in range(lists[0]):
        if constraint[nboundary].level != ti.u8(255):
            constraint[remain_num].node = constraint[nboundary].node
            constraint[remain_num].level = constraint[nboundary].level
            constraint[remain_num].traction = constraint[nboundary].traction
            
            if remain_num < nboundary:
                constraint[nboundary].node = -1
                constraint[nboundary].level = ti.u8(255)
                constraint[nboundary].traction = vec3f(0, 0, 0)
            remain_num += 1
    lists[0] = remain_num


@ti.kernel
def apply_velocity_constraint(cut_off: float, lists: int, constraints: ti.template(), is_rigid: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if node[nodeID, grid_level].m > cut_off:
            fix_v = int(constraints[nboundary].fix_v)
            unfix_v = int(constraints[nboundary].unfix_v)

            if is_rigid[grid_level] == 0:
                prescribed_velocity = constraints[nboundary].velocity
                node[nodeID, grid_level].velocity_constraint(fix_v, unfix_v, prescribed_velocity)
            else:
                node[nodeID, grid_level].rigid_body_velocity_constraint(fix_v, unfix_v)


@ti.kernel
def apply_reflection_constraint(cut_off: float, lists: int, constraints: ti.template(), is_rigid: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)

        if node[nodeID, grid_level].m > cut_off:
            norm1 = constraints[nboundary].norm1
            norm2 = constraints[nboundary].norm2
            norm3 = constraints[nboundary].norm3
            
            if is_rigid[grid_level] == 0:
                node[nodeID, grid_level].reflection_constraint(norm1, norm2, norm3)
            else:
                node[nodeID, grid_level].rigid_body_reflection_constraint(norm1, norm2, norm3)

@ti.kernel
def apply_contact_velocity_constraint(cut_off: float, lists: int, constraints: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if SquareLen(node[nodeID, grid_level].contact_force) > cut_off:
            fix_v = int(constraints[nboundary].fix_v)
            unfix_v = int(constraints[nboundary].unfix_v)
            node[nodeID, grid_level].contact_velocity_constraint(fix_v, unfix_v)

@ti.kernel
def apply_contact_reflection_constraint(cut_off: float, lists: int, constraints: ti.template(), node: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)

        if SquareLen(node[nodeID, grid_level].contact_force) > cut_off:
            norm1 = constraints[nboundary].norm1
            norm2 = constraints[nboundary].norm2
            norm3 = constraints[nboundary].norm3
            node[nodeID, grid_level].contact_reflection_constraint(norm1, norm2, norm3)

@ti.kernel
def apply_friction_constraint(cut_off: float, lists: int, constraints: ti.template(), is_rigid: ti.template(), node: ti.template(), dt: ti.template()):
    for nboundary in range(lists):
        nodeID = constraints[nboundary].node
        grid_level = int(constraints[nboundary].level)
        if node[nodeID, grid_level].m > cut_off:
            norm = constraints[nboundary].norm
            
            if is_rigid[grid_level] == 0:
                mu = constraints[nboundary].mu
                node[nodeID, grid_level].friction_constraint(mu, norm, dt)
            else:
                node[nodeID, grid_level].rigid_friction_constraint(norm)

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
            node[nodeID, grid_level].force += constraints[nboundary].traction