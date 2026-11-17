import taichi as ti

from src.utils.constants import M_THRESHOLD, ZEROVEC3F, L_THRESHOLD, THRESHOLD
from src.utils.Quaternion import set_from_two_vec, set_to_rotate, rodrigues_rotation_matrix
from src.utils.TypeDefinition import real, vec3f, vec4f, vec2i
from src.utils.VectorFunction import square_len
from src.utils.BitFunction import zero2_one_vector


@ti.kernel
def kernel_adaptive_timestep(particle_num: int, particle: ti.template()) -> float:
    min_dt = M_THRESHOLD
    for np in range(particle_num):
        ti.atomic_min(min_dt, 0.1 * particle[np].rad / (particle[np].v.norm() + THRESHOLD))
    return min_dt


@ti.kernel
def update_particle_storage(
    particle_num: ti.types.ndarray(),
    sphere_num: ti.types.ndarray(),
    clump_num: ti.types.ndarray(),
    particle: ti.template(),
    sphere: ti.template(),
    clump: ti.template(),
):
    active = True
    remaining_sphere = 0
    remaining_clump = 0
    remaining_particle = 0
    delete_particle = 0
    ti.loop_config(serialize=True)
    for np in range(particle_num[0]):
        multisphere_index = particle[np].multisphereIndex
        if ti.static(not sphere == None):
            if multisphere_index < 0:
                if int(particle[np].active) == 1:
                    particle[remaining_particle] = particle[np]
                    particle[remaining_particle].multisphereIndex += delete_particle
                    sphere[remaining_sphere] = sphere[-multisphere_index - 1]
                    sphere[remaining_sphere].sphereIndex -= delete_particle
                    remaining_particle += 1
                    remaining_sphere += 1
                else:
                    delete_particle += 1
        if ti.static(not clump == None):
            if multisphere_index >= 0:
                if active and int(particle[np].active) == 0:
                    active = False
                start_index = clump[multisphere_index].startIndex
                end_index = clump[multisphere_index].endIndex
                if np == end_index:
                    pebble_num = end_index - start_index + 1
                    if active:
                        for npebble in range(pebble_num):
                            particle[remaining_particle + npebble] = particle[np]
                            particle[remaining_particle + npebble].multisphereIndex -= delete_particle
                        clump[remaining_clump] = clump[multisphere_index]
                        clump[remaining_clump].startIndex -= delete_particle
                        clump[remaining_clump].endIndex -= delete_particle
                        remaining_particle += pebble_num
                        remaining_clump += 1
                    else:
                        delete_particle += pebble_num
                    active = True
    sphere_num[0] = remaining_sphere
    clump_num[0] = remaining_clump
    particle_num[0] = remaining_particle


@ti.kernel
def update_l_sparticle_storage(
    particle_num: ti.types.ndarray(),
    rigid_num: ti.types.ndarray(),
    surface_num: ti.types.ndarray(),
    particle: ti.template(),
    box: ti.template(),
    rigid: ti.template(),
):
    remaining_particle = 0
    remaining_node = 0
    delete_particle = 0
    delete_node = 0
    ti.loop_config(serialize=True)
    for np in range(particle_num[0]):
        if int(particle[np].active) == 1:
            vertice_num = rigid[np].endNode - rigid[np].startNode
            particle[remaining_particle] = particle[np]
            box[remaining_particle] = box[np]
            rigid[remaining_particle] = rigid[np]
            rigid[remaining_particle].startNode -= delete_node
            rigid[remaining_particle].endNode -= delete_node
            remaining_particle += 1
            remaining_node += vertice_num
        else:
            delete_particle += 1
            delete_node += rigid[np].endNode - rigid[np].startNode
    particle_num[0] = remaining_particle
    rigid_num[0] = remaining_particle
    surface_num[0] = remaining_node


@ti.kernel
def update_l_ssurface_storage(rigid_num: int, rigid: ti.template(), surface: ti.template()):
    for nb in range(rigid_num):
        for i in range(rigid[nb].startNode, rigid[nb].endNode):
            surface[i] = nb


@ti.kernel
def update_i_sparticle_storage(
    particle_num: ti.types.ndarray(), rigid_num: ti.types.ndarray(), particle: ti.template(), rigid: ti.template()
):
    remaining_particle = 0
    delete_particle = 0
    ti.loop_config(serialize=True)
    for np in range(particle_num[0]):
        if int(particle[np].active) == 1:
            particle[remaining_particle] = particle[np]
            rigid[remaining_particle] = rigid[np]
            remaining_particle += 1
        else:
            delete_particle += 1
    particle_num[0] = remaining_particle
    rigid_num[0] = remaining_particle


@ti.kernel
def update_servo_storage(servo_num: ti.types.ndarray(), wall: ti.template(), servo: ti.template()):
    remaining_servo = 0
    delete_servo = 0
    ti.loop_config(serialize=True)
    for ns in range(servo_num[0]):
        end_ind = int(servo[ns].endIndex)
        for nw in range(int(servo[ns].startIndex), int(servo[ns].endIndex)):
            if int(wall[nw].active) == 0:
                end_ind -= 1
        servo[ns].endIndex = ti.u8(end_ind)

    ti.loop_config(serialize=True)
    for ns in range(1, servo_num[0]):
        nwall = servo[ns].endIndex - servo[ns].startIndex
        servo[ns].startIndex = servo[ns - 1].endIndex
        servo[ns].endIndex = servo[ns].startIndex + nwall
        if nwall == 0:
            servo[ns].active = ti.u8(0)

    ti.loop_config(serialize=True)
    for ns in range(servo_num[0]):
        if int(servo[ns].active) == 1:
            servo[remaining_servo] = servo[ns]
            remaining_servo += 1
        else:
            delete_servo += 1
    servo_num[0] = remaining_servo


@ti.kernel
def update_wall_storage(wall_num: ti.types.ndarray(), wall: ti.template()):
    remaining_wall = 0
    delete_wall = 0
    ti.loop_config(serialize=True)
    for nw in range(wall_num[0]):
        if int(wall[nw].active) == 1:
            wall[remaining_wall] = wall[nw]
            remaining_wall += 1
        else:
            delete_wall += 1
    wall_num[0] = remaining_wall


@ti.kernel
def object_calm(particle_num: int, particle: ti.template()):
    for np in range(particle_num):
        particle[np].v = ZEROVEC3F
        particle[np].w = ZEROVEC3F


@ti.kernel
def reset_verlet_disp(object_num: int, object: ti.template()):
    for nobject in range(object_num):
        object[nobject]._renew_verlet()


@ti.kernel
def find_particle_verlet_disp(particle_num: int, particle: ti.template()) -> float:
    max_verlet_disp = 0.0
    for np in range(particle_num):
        verlet_disp = particle[np].verletDisp
        verlet_disp_len = (
            verlet_disp[0] * verlet_disp[0] + verlet_disp[1] * verlet_disp[1] + verlet_disp[2] * verlet_disp[2]
        )
        ti.atomic_max(max_verlet_disp, verlet_disp_len)
    return max_verlet_disp


@ti.kernel
def find_particle_min_mass(particle_num: int, particle: ti.template()) -> float:
    min_mass = M_THRESHOLD
    for np in range(particle_num):
        mass = particle[np]._get_mass()
        ti.atomic_min(min_mass, mass)
    return min_mass


@ti.kernel
def find_lsmpm_contact_min_mass(particle_num: int, rigid: ti.template()) -> float:
    min_mass = M_THRESHOLD
    for body_id in range(particle_num):
        mass = rigid[body_id]._get_mass()
        if int(rigid[body_id].is_soft) == 1:
            mass /= ti.max(rigid[body_id]._get_vertice_number(), 1)
        ti.atomic_min(min_mass, mass)
    return min_mass


@ti.kernel
def find_particle_total_mass(particle_num: int, particle: ti.template()) -> float:
    total_mass = 0.0
    for np in range(particle_num):
        mass = particle[np]._get_mass()
        ti.atomic_add(total_mass, mass)
    return total_mass


@ti.kernel
def find_left_bottom_scene(particle_num: int, particle: ti.template()) -> ti.types.vector(3, float):
    left_bottom_scene = vec3f(M_THRESHOLD, M_THRESHOLD, M_THRESHOLD)
    for np in range(particle_num):
        position = particle[np]._get_position()
        radius = particle[np]._get_radius()
        ti.atomic_min(left_bottom_scene[0], position[0] - radius)
        ti.atomic_min(left_bottom_scene[1], position[1] - radius)
        ti.atomic_min(left_bottom_scene[2], position[2] - radius)
    return left_bottom_scene


@ti.kernel
def find_right_top_scene(particle_num: int, particle: ti.template()) -> ti.types.vector(3, float):
    right_top_scene = vec3f(-M_THRESHOLD, -M_THRESHOLD, -M_THRESHOLD)
    for np in range(particle_num):
        position = particle[np]._get_position()
        radius = particle[np]._get_radius()
        ti.atomic_max(right_top_scene[0], position[0] + radius)
        ti.atomic_max(right_top_scene[1], position[1] + radius)
        ti.atomic_max(right_top_scene[2], position[2] + radius)
    return right_top_scene


@ti.kernel
def validate_displacement(limit: float, object_num: int, object: ti.template()) -> int:
    flag = 0
    for nobject in range(object_num):
        if flag == 0 and square_len(object[nobject].verletDisp) >= limit:
            flag = 1
    return flag


@ti.kernel
def validate_deformable_bounding_sphere(limit: float, object_num: int, particle: ti.template()) -> int:
    flag = 0
    for np in range(object_num):
        center_displacement = particle[np].verletDisp.norm()
        radial_growth = ti.max(0.0, particle[np].rad - particle[np].rad_verlet)
        if flag == 0 and (center_displacement + radial_growth) ** 2 >= limit:
            flag = 1
    return flag


@ti.kernel
def find_expect_extent(verlet_distance: float, body_num: ti.template(), box: ti.template()) -> ti.types.vector(2, int):
    pid, extent = -1, 0
    ti.loop_config(serialize=True)
    for np in range(body_num):
        if verlet_distance > box[np].grid_space * box[np].extent:
            if ti.ceil(verlet_distance / box[np].grid_space, int) > extent:
                pid = np
                extent = ti.ceil(verlet_distance / box[np].grid_space, int)
    return vec2i(pid, extent)


@ti.kernel
def find_min_grid_space(body_num: int, box: ti.template()) -> float:
    min_grid_space = 1e15
    for np in range(body_num):
        grid_space = box[np].grid_space
        ti.atomic_min(min_grid_space, grid_space)
    return min_grid_space


@ti.kernel
def find_min_extent(body_num: int, box: ti.template()) -> float:
    min_extent = 1e15
    for np in range(body_num):
        grid_space = box[np].grid_space * box[np].extent
        ti.atomic_min(min_extent, grid_space)
    return min_extent


@ti.kernel
def find_particle_max_radius(particle_num: int, particle: ti.template()) -> float:
    max_radius = 0.0
    for np in range(particle_num):
        radius = particle[np]._get_radius()
        ti.atomic_max(max_radius, radius)
    return max_radius


@ti.kernel
def find_particle_min_radius(particle_num: int, particle: ti.template()) -> float:
    min_radius = M_THRESHOLD
    for np in range(particle_num):
        radius = particle[np]._get_radius()
        ti.atomic_min(min_radius, radius)
    return min_radius


@ti.kernel
def find_patch_max_radius(wall_num: int, wall: ti.template()) -> float:
    max_radius = 0.0
    for nw in range(wall_num):
        radius = wall[nw]._get_bounding_radius()
        ti.atomic_max(max_radius, radius)
    return max_radius


@ti.kernel
def find_patch_min_radius(wall_num: int, wall: ti.template()) -> float:
    min_radius = M_THRESHOLD
    for nw in range(wall_num):
        radius = wall[nw]._get_bounding_radius()
        ti.atomic_min(min_radius, radius)
    return min_radius


@ti.kernel
def kernel_delete_particles(particle_num: int, particle: ti.template(), body_id: int):
    for np in range(particle_num):
        if particle[np].bodyID == body_id:
            particle[np].active = ti.u8(0)


@ti.kernel
def kernel_delete_lsparticles(particle_num: int, particle: ti.template(), body_id: int):
    for np in range(particle_num):
        if particle[np].bodyID == body_id:
            particle[np].active = ti.u8(0)


@ti.kernel
def kernel_delete_particles_in_region(particle_num: int, particle: ti.template(), is_in_region: ti.template()):
    for np in range(particle_num):
        if is_in_region(particle[np].x, particle[np].rad):
            particle[np].active = ti.u8(0)


@ti.kernel
def apply_boundary_conditions(
    domain: ti.types.vector(3, float),
    particle_num: int,
    particle: ti.template(),
    xboundary: ti.template(),
    yboundary: ti.template(),
    zboundary: ti.template(),
) -> int:
    not_in_xdomain, not_in_ydomain, not_in_zdomain = False, False, False
    for np in range(particle_num):
        if particle[np].active == 1:
            not_in_xdomain |= xboundary(np, 0, domain, particle)
            not_in_ydomain |= yboundary(np, 1, domain, particle)
            not_in_zdomain |= zboundary(np, 2, domain, particle)
    return not_in_xdomain | not_in_ydomain | not_in_zdomain


@ti.func
def none_boundary(np, axis, domain, particle):
    pass


@ti.func
def destroy_boundary(np, axis, domain, particle):
    in_domain = True
    position = particle[np].x
    if in_domain == True and position[axis] < 0.0:
        in_domain = False
        particle[np].active = ti.u8(0)
    elif in_domain == True and position[axis] > domain[axis]:
        in_domain = False
        particle[np].active = ti.u8(0)
    return not in_domain


@ti.func
def reflect_boundary(np, axis, domain, particle):
    in_domain = True
    position = particle[np].x
    if in_domain == True and position[axis] < 0.0:
        in_domain = False
        particle[np].x[axis] = L_THRESHOLD * domain[axis]
        particle[np].v = -particle[np].v
    elif in_domain == True and position[axis] > domain[axis]:
        in_domain = False
        particle[np].x[axis] = (1.0 - L_THRESHOLD) * domain[axis]
        particle[np].v = -particle[np].v
    return False


@ti.func
def period_boundary(np, axis, domain, particle):
    in_domain = True
    position = particle[np].x
    if in_domain == True and position[axis] < 0.0:
        in_domain = False
        particle[np].x[axis] += domain[axis]
    elif in_domain == True and position[axis] > domain[axis]:
        in_domain = False
        particle[np].x[axis] -= domain[axis]
    return False


@ti.kernel
def check_in_domain(domain: ti.types.vector(3, float), particle_num: int, particle: ti.template()):
    for np in range(particle_num):
        if particle[np].active == 1 and not is_in_domain(domain, particle[np].x):
            particle[np].active = ti.u8(0)


@ti.func
def is_in_domain(domain, position):
    in_domain = 1
    if in_domain == 1 and position[0] < 0.0:
        in_domain = 0
    elif in_domain == 1 and position[1] < 0.0:
        in_domain = 0
    elif in_domain == 1 and position[2] < 0.0:
        in_domain = 0
    elif in_domain == 1 and position[0] > domain[0]:
        in_domain = 0
    elif in_domain == 1 and position[1] > domain[1]:
        in_domain = 0
    elif in_domain == 1 and position[2] > domain[2]:
        in_domain = 0
    return in_domain


@ti.func
def update_value(factor, pre_val, val):
    return factor * pre_val + val


@ti.func
def replace(factor, pre_val, val):
    return val


@ti.func
def upscale(factor, pre_val, val):
    return pre_val * val


@ti.kernel
def modify_sphere_body_id_in_region(
    value: int, sphere_num: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            particle[particle_id].bodyID = ti.u8(value)


@ti.kernel
def modify_sphere_group_id_in_region(
    value: int, sphere_num: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            particle[particle_id].groupID = ti.u8(value)


@ti.kernel
def modify_levelset_group_id_in_region(value: int, rigid_num: int, rigid: ti.template(), is_in_region: ti.template()):
    for nrigid in range(rigid_num):
        if is_in_region(rigid[nrigid].mass_center):
            rigid[nrigid].groupID = ti.u8(value)


@ti.kernel
def modify_sphere_material_id_in_region(
    value: int,
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    material: ti.template(),
    is_in_region: ti.template(),
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            old_material_id = particle[particle_id].materialID
            old_density = material[old_material_id].density
            particle[particle_id].materialID = ti.u8(value)
            density = material[value].density
            ratio = density / old_density
            particle[particle_id].mass *= ratio
            sphere[nsphere].inv_I *= 1.0 / ratio


@ti.kernel
def modify_sphere_radius_in_region(
    value: float, sphere_num: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            ratio = value / particle[particle_id].rad
            particle[particle_id].rad = value
            particle[particle_id].m *= ratio * ratio * ratio
            sphere[nsphere].inv_I *= 1.0 / (ratio * ratio * ratio * ratio * ratio)


@ti.kernel
def modify_sphere_position_in_region(
    factor: int,
    value: ti.types.vector(3, float),
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            particle[particle_id].x = factor * particle[particle_id].x + value


@ti.kernel
def modify_sphere_velocity_in_region(
    factor: int,
    value: ti.types.vector(3, float),
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            particle[particle_id].v = factor * particle[particle_id].v + value


@ti.kernel
def modify_sphere_angular_velocity_in_region(
    factor: int,
    value: ti.types.vector(3, float),
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            particle[particle_id].w = factor * particle[particle_id].w + value


@ti.kernel
def modify_sphere_orientation_in_region(
    value: ti.types.vector(3, float),
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            sphere[nsphere].q = set_from_two_vec(vec3f(0.0, 0.0, 1), value)


@ti.kernel
def modify_sphere_fix_v_in_region(
    value: ti.types.vector(3, ti.u8),
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            sphere[nsphere].fix_v = zero2_one_vector(value)


@ti.kernel
def modify_sphere_fix_w_in_region(
    value: ti.types.vector(3, ti.u8),
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if is_in_region(particle[particle_id].x):
            sphere[nsphere].fix_w = zero2_one_vector(value)


@ti.kernel
def modify_clump_body_id_in_region(
    value: int, clump_num: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            for np in range(start_index, end_index):
                particle[np].bodyID = ti.u8(value)


@ti.kernel
def modify_clump_group_id_in_region(
    value: int, clump_num: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            for np in range(start_index, end_index):
                particle[np].groupID = ti.u8(value)


@ti.kernel
def modify_clump_material_id_in_region(
    value: int,
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    material: ti.template(),
    is_in_region: ti.template(),
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            for np in range(start_index, end_index):
                old_material_id = particle[np].materialID
                old_density = material[old_material_id].density
                particle[np].materialID = ti.u8(value)
                density = material[value].density
                ratio = density / old_density
                particle[np].mass *= ratio
            clump[nclump].inv_I *= 1.0 / ratio


@ti.kernel
def modify_clump_radius_in_region(
    value: float, clump_num: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        equivalent_radius = clump[nclump].equi_r
        if is_in_region == 1:
            for np in range(start_index, end_index):
                ratio = value / equivalent_radius
                particle[np].rad *= ratio
                particle[np].m *= ratio * ratio * ratio
            clump[nclump].inv_I *= 1.0 / (ratio * ratio * ratio * ratio * ratio)


@ti.kernel
def modify_clump_position_in_region(
    factor: int,
    value: ti.types.vector(3, float),
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            for np in range(start_index, end_index):
                particle[np].x = factor * particle[np].x + value
            clump[nclump].mass_center = factor * clump[nclump].mass_center + value


@ti.kernel
def modify_clump_velocity_in_region(
    factor: int,
    value: ti.types.vector(3, float),
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            for np in range(start_index, end_index):
                particle[np].v = factor * particle[np].v + value
            clump[nclump].v = factor * clump[nclump].v + value


@ti.kernel
def modify_clump_angular_velocity_in_region(
    factor: int,
    value: ti.types.vector(3, float),
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            center_vel = clump[nclump].v
            mass_center = clump[nclump].mass_center
            for np in range(start_index, end_index):
                angular_vel = factor * particle[np].w + value
                particle_vel = center_vel + angular_vel.cross(particle[np].x - mass_center)
                particle[np].v = particle_vel
                particle[np].w = angular_vel
            clump[nclump].w = factor * clump[nclump].w + value


@ti.kernel
def modify_clump_orientation_in_region(
    value: ti.types.vector(3, float),
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            rmatrix = set_to_rotate(clump[nclump].q)
            mass_center = clump[nclump].mass_center
            clump[nclump].q = set_from_two_vec(vec3f(0.0, 0.0, 1), value)
            update_rmatrix = set_to_rotate(clump[nclump].q)
            for np in range(start_index, end_index):
                particle[np].x = update_rmatrix @ (rmatrix.transpose() @ (particle[np].x - mass_center)) + mass_center


@ti.kernel
def modify_clump_fix_v_in_region(
    value: ti.types.vector(3, ti.u8),
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            clump[nclump].fix_v = zero2_one_vector(value)


@ti.kernel
def modify_clump_fix_w_in_region(
    value: ti.types.vector(3, ti.u8),
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    is_in_region: ti.template(),
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        is_in_region = 1
        for np in range(start_index, end_index):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            clump[nclump].fix_w = zero2_one_vector(value)


@ti.kernel
def modify_material_density(factor: int, value: float, material_id: int, material: ti.template()):
    density = material[material_id].density
    material[material_id].density = factor * density + value


@ti.kernel
def modify_material_fdamp(factor: int, value: float, material_id: int, material: ti.template()):
    fdamp = material[material_id].fdamp
    material[material_id].fdamp = factor * fdamp + value


@ti.kernel
def modify_material_tdamp(factor: int, value: float, material_id: int, material: ti.template()):
    tdamp = material[material_id].tdamp
    material[material_id].tdamp = factor * tdamp + value


@ti.kernel
def modify_sphere_body_id(value: int, sphere_num: int, sphere: ti.template(), particle: ti.template(), body_id: int):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if nsphere == body_id:
            particle[particle_id].bodyID = ti.u8(value)


@ti.kernel
def modify_sphere_group_id(value: int, sphere_num: int, sphere: ti.template(), particle: ti.template(), body_id: int):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if nsphere == body_id:
            particle[particle_id].groupID = ti.u8(value)


@ti.kernel
def modify_sphere_material_id(
    value: int, sphere_num: int, sphere: ti.template(), particle: ti.template(), material: ti.template(), body_id: int
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if nsphere == body_id:
            old_material_id = particle[particle_id].materialID
            old_density = material[old_material_id].density
            particle[particle_id].materialID = ti.u8(value)
            density = material[value].density
            ratio = density / old_density
            particle[particle_id].mass *= ratio
            sphere[nsphere].inv_I *= 1.0 / ratio


@ti.kernel
def modify_sphere_radius(value: float, sphere_num: int, sphere: ti.template(), particle: ti.template(), body_id: int):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if nsphere == body_id:
            ratio = value / particle[particle_id].rad
            particle[particle_id].rad = value
            particle[particle_id].m *= ratio * ratio * ratio
            sphere[nsphere].inv_I *= 1.0 / (ratio * ratio * ratio * ratio * ratio)


@ti.kernel
def modify_sphere_position(
    factor: int,
    value: ti.types.vector(3, float),
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    body_id: int,
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if nsphere == body_id:
            particle[particle_id].x = factor * particle[particle_id].x + value


@ti.kernel
def modify_sphere_velocity(
    factor: int,
    value: ti.types.vector(3, float),
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    body_id: int,
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if nsphere == body_id:
            particle[particle_id].v = factor * particle[particle_id].v + value


@ti.kernel
def modify_sphere_angular_velocity(
    factor: int,
    value: ti.types.vector(3, float),
    sphere_num: int,
    sphere: ti.template(),
    particle: ti.template(),
    body_id: int,
):
    for nsphere in range(sphere_num):
        particle_id = sphere[nsphere].sphereIndex
        if nsphere == body_id:
            particle[particle_id].w = factor * particle[particle_id].w + value


@ti.kernel
def modify_sphere_orientation(
    value: ti.types.vector(3, float), sphere_num: int, sphere: ti.template(), particle: ti.template(), body_id: int
):
    for nsphere in range(sphere_num):
        if nsphere == body_id:
            sphere[nsphere].q = set_from_two_vec(vec3f(0.0, 0.0, 1), value)


@ti.kernel
def modify_sphere_fix_v(
    value: ti.types.vector(3, ti.u8), sphere_num: int, sphere: ti.template(), particle: ti.template(), body_id: int
):
    for nsphere in range(sphere_num):
        if nsphere == body_id:
            sphere[nsphere].fix_v = zero2_one_vector(value)


@ti.kernel
def modify_sphere_fix_w(
    value: ti.types.vector(3, ti.u8), sphere_num: int, sphere: ti.template(), particle: ti.template(), body_id: int
):
    for nsphere in range(sphere_num):
        if nsphere == body_id:
            sphere[nsphere].fix_w = zero2_one_vector(value)


@ti.kernel
def modify_clump_body_id(value: int, clump_num: int, clump: ti.template(), particle: ti.template(), body_id: int):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        if nclump == body_id:
            for np in range(start_index, end_index):
                particle[np].bodyID = ti.u8(value)


@ti.kernel
def modify_clump_group_id(value: int, clump_num: int, clump: ti.template(), particle: ti.template(), body_id: int):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        if nclump == body_id:
            for np in range(start_index, end_index):
                particle[np].groupID = ti.u8(value)


@ti.kernel
def modify_clump_material_id(
    value: int, clump_num: int, clump: ti.template(), particle: ti.template(), material: ti.template(), body_id: int
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        if nclump == body_id:
            for np in range(start_index, end_index):
                old_material_id = particle[np].materialID
                old_density = material[old_material_id].density
                particle[np].materialID = ti.u8(value)
                density = material[value].density
                ratio = density / old_density
                particle[np].mass *= ratio
            clump[nclump].inv_I *= 1.0 / ratio


@ti.kernel
def modify_clump_radius(value: float, clump_num: int, clump: ti.template(), particle: ti.template(), body_id: int):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        equivalent_radius = clump[nclump].equi_r
        if nclump == body_id:
            for np in range(start_index, end_index):
                ratio = value / equivalent_radius
                particle[np].rad *= ratio
                particle[np].m *= ratio * ratio * ratio
            clump[nclump].inv_I *= 1.0 / (ratio * ratio * ratio * ratio * ratio)


@ti.kernel
def modify_clump_position(
    factor: int,
    value: ti.types.vector(3, float),
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    body_id: int,
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        if nclump == body_id:
            for np in range(start_index, end_index):
                particle[np].x = factor * particle[np].x + value
            clump[nclump].mass_center = factor * clump[nclump].mass_center + value


@ti.kernel
def modify_clump_velocity(
    factor: int,
    value: ti.types.vector(3, float),
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    body_id: int,
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        if nclump == body_id:
            for np in range(start_index, end_index):
                particle[np].v = factor * particle[np].v + value
            clump[nclump].v = factor * clump[nclump].v + value


@ti.kernel
def modify_clump_angular_velocity(
    factor: int,
    value: ti.types.vector(3, float),
    clump_num: int,
    clump: ti.template(),
    particle: ti.template(),
    body_id: int,
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        if nclump == body_id:
            center_vel = clump[nclump].v
            mass_center = clump[nclump].mass_center
            for np in range(start_index, end_index):
                angular_vel = factor * particle[np].w + value
                particle_vel = center_vel + angular_vel.cross(particle[np].x - mass_center)
                particle[np].v = particle_vel
                particle[np].w = angular_vel
            clump[nclump].w = factor * clump[nclump].w + value


@ti.kernel
def modify_clump_orientation(
    value: ti.types.vector(3, float), clump_num: int, clump: ti.template(), particle: ti.template(), body_id: int
):
    for nclump in range(clump_num):
        start_index = clump[nclump].startIndex
        end_index = clump[nclump].endIndex

        if nclump == body_id:
            rmatrix = set_to_rotate(clump[nclump].q)
            mass_center = clump[nclump].mass_center
            clump[nclump].q = set_from_two_vec(vec3f(0.0, 0.0, 1), value)
            update_rmatrix = set_to_rotate(clump[nclump].q)
            for np in range(start_index, end_index):
                particle[np].x = update_rmatrix @ (rmatrix.transpose() @ (particle[np].x - mass_center)) + mass_center


@ti.kernel
def modify_clump_fix_v(
    value: ti.types.vector(3, ti.u8), clump_num: int, clump: ti.template(), particle: ti.template(), body_id: int
):
    for nclump in range(clump_num):
        if nclump == body_id:
            clump[nclump].fix_v = zero2_one_vector(value)


@ti.kernel
def modify_clump_fix_w(
    value: ti.types.vector(3, ti.u8), clump_num: int, clump: ti.template(), particle: ti.template(), body_id: int
):
    for nclump in range(clump_num):
        if nclump == body_id:
            clump[nclump].fix_w = zero2_one_vector(value)


@ti.kernel
def modify_wall_activate_status(input_wall_id: int, status: int, wall_num: int, wall: ti.template()):
    for wall_id in range(wall_num):
        if wall[wall_id].wallID == input_wall_id:
            wall[wall_id].active = ti.u8(status)


@ti.kernel
def modify_wall_material_id(input_wall_id: int, value: int, wall_num: int, wall: ti.template()):
    for wall_id in range(wall_num):
        if wall[wall_id].wallID == input_wall_id:
            wall[wall_id].materialID = value


@ti.kernel
def modify_plane_position(
    factor: int, input_wall_id: int, value: ti.types.vector(3, float), wall_num: int, wall: ti.template()
):
    for wall_id in range(wall_num):
        if wall[wall_id].wallID == input_wall_id:
            wall[wall_id].point = wall[wall_id].point * factor + value


@ti.kernel
def modify_plane_orientation(
    factor: int, input_wall_id: int, value: ti.types.vector(3, float), wall_num: int, wall: ti.template()
):
    for wall_id in range(wall_num):
        if wall[wall_id].wallID == input_wall_id:
            wall[wall_id].norm = wall[wall_id].norm * factor + value


@ti.kernel
def modify_triangle_position(
    factor: int, input_wall_id: int, value: ti.types.vector(3, float), wall_num: int, wall: ti.template()
):
    for wall_id in range(wall_num):
        if wall[wall_id].wallID == input_wall_id:
            wall[wall_id].vertice1 = wall[wall_id].vertice1 * factor + value
            wall[wall_id].vertice2 = wall[wall_id].vertice2 * factor + value
            wall[wall_id].vertice3 = wall[wall_id].vertice3 * factor + value
            wall[wall_id].verletDisp = ZEROVEC3F


@ti.kernel
def modify_triangle_velocity(
    factor: int, input_wall_id: int, value: ti.types.vector(3, float), wall_num: int, wall: ti.template()
):
    for wall_id in range(wall_num):
        if wall[wall_id].wallID == input_wall_id:
            wall[wall_id].v = wall[wall_id].v * factor + value


@ti.kernel
def modify_triangle_orientation(
    mode: int,
    factor: int,
    input_wall_id: int,
    new_direction: ti.types.vector(3, float),
    rotation_center: ti.types.vector(3, float),
    wall_num: int,
    wall: ti.template(),
):
    for wall_id in range(wall_num):
        if wall[wall_id].wallID == input_wall_id:
            if mode == 1:
                rotation_center = wall[wall_id]._get_center()
            rotation_matrix = rodrigues_rotation_matrix(wall[wall_id].norm, factor * wall[wall_id].norm + new_direction)
            wall[wall_id].vertice1 = rotation_matrix @ (wall[wall_id].vertice1 - rotation_center) + rotation_center
            wall[wall_id].vertice2 = rotation_matrix @ (wall[wall_id].vertice2 - rotation_center) + rotation_center
            wall[wall_id].vertice3 = rotation_matrix @ (wall[wall_id].vertice3 - rotation_center) + rotation_center
            wall[wall_id].norm = factor * wall[wall_id].norm + new_direction


@ti.kernel
def create_level_set_grids(
    grid_num: int,
    grid_sum: int,
    grid: ti.template(),
    distance_fields: ti.types.ndarray(),
):
    for i in range(grid_sum):
        grid[i + grid_num]._set_grid(distance_fields[i])


@ti.kernel
def create_level_set_surface(
    surface_num: int,
    surface_sum: int,
    surface_node: ti.template(),
    surface_nodes: ti.types.ndarray(),
    parameters: ti.types.ndarray(),
):
    for i in range(surface_sum):
        surface_node[i + surface_num]._set_surface_node(
            vec3f(surface_nodes[i, 0], surface_nodes[i, 1], surface_nodes[i, 2])
        )
        surface_node[i + surface_num]._set_coefficient(parameters[i])


@ti.kernel
def kernel_add_polysuperellipsoid_parameter(
    surface_num: int,
    surface: ti.template(),
    xrad1: float,
    yrad1: float,
    zrad1: float,
    epsilon_e: float,
    epsilon_n: float,
    xrad2: float,
    yrad2: float,
    zrad2: float,
):
    surface[surface_num]._add_template_parameter(xrad1, yrad1, zrad1, epsilon_e, epsilon_n, xrad2, yrad2, zrad2)


@ti.kernel
def kernel_add_polysuperquadrics_parameter(
    surface_num: int,
    surface: ti.template(),
    xrad1: float,
    yrad1: float,
    zrad1: float,
    epsilon_x: float,
    epsilon_y: float,
    epsilon_z: float,
    xrad2: float,
    yrad2: float,
    zrad2: float,
):
    surface[surface_num]._add_template_parameter(
        xrad1, yrad1, zrad1, epsilon_x, epsilon_y, epsilon_z, xrad2, yrad2, zrad2
    )


@ti.kernel
def kernel_visualize_levelset_surface(
    rigid_num: int,
    surface_node: ti.template(),
    visualzie_surface_node: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
):
    for master in range(rigid_num):
        for node in range(rigid[master]._start_node(), rigid[master]._end_node()):
            gnode = rigid[master].local_node_to_global(node)
            rotate_matrix = set_to_rotate(rigid[master].q)
            visualzie_surface_node[gnode] = rigid[master].mass_center + rotate_matrix @ (
                box[master].scale * surface_node[node].x
            )


@ti.kernel
def kernel_visualize_implicit_surface(
    rigid_num: int,
    template_vertice_num: ti.types.ndarray(),
    stacked_vertices: ti.types.ndarray(),
    total_vertice_num: ti.types.ndarray(),
    visualzie_surface_node: ti.template(),
    rigid: ti.template(),
):
    for master in range(rigid_num):
        template_id = rigid[master].templateID
        scale = rigid[master].scale
        begin_index = total_vertice_num[master]
        for node in range(template_vertice_num[template_id], template_vertice_num[template_id + 1]):
            gnode = node - template_vertice_num[template_id]
            rotate_matrix = set_to_rotate(rigid[master].q)
            visualzie_surface_node[begin_index + gnode] = rigid[master].mass_center + rotate_matrix @ (
                scale * vec3f(stacked_vertices[node, 0], stacked_vertices[node, 1], stacked_vertices[node, 2])
            )


@ti.kernel
def kernel_postvisualize_surface(
    surface_num: int,
    surface_node: ti.types.ndarray(),
    position: ti.types.ndarray(),
    quanternion: ti.types.ndarray(),
    start_node: ti.types.ndarray(),
    start_local_node: ti.types.ndarray(),
    master_id: ti.types.ndarray(),
    scale: ti.types.ndarray(),
    vertices: ti.template(),
):
    for iterate in range(surface_num):
        master = master_id[iterate]
        local_node = iterate - start_node[master] + start_local_node[master]
        rotation_matrix = set_to_rotate(
            vec4f([quanternion[master, 0], quanternion[master, 1], quanternion[master, 2], quanternion[master, 3]])
        )
        vertices[iterate] = vec3f([position[master, 0], position[master, 1], position[master, 2]]) + rotation_matrix @ (
            scale[master]
            * vec3f([surface_node[local_node, 0], surface_node[local_node, 1], surface_node[local_node, 2]])
        )


@ti.kernel
def check_radius(rigid_num: int, rigid: ti.template()) -> bool:
    check = True
    for i in range(1, rigid_num):
        if check and rigid[i - 1].equi_r > rigid[i].equi_r:
            check = False
    return check


def get_connectivity(surface_node_number, faces, i):
    return faces + i * surface_node_number
