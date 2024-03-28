import taichi as ti

from src.utils.TypeDefination import vec3f, vec6f, vec3u8, mat3x3
from src.utils.Quaternion import RodriguesRotationMatrix


@ti.kernel
def kernel_calc_mass_of_center_(coords: ti.types.ndarray()) -> ti.types.vector(3, float): # type: ignore
    position = vec3f(0, 0, 0)
    for np in range(coords.shape[0]):
        position += vec3f(coords[np, 0], coords[np, 1], coords[np, 2])
    return position / coords.shape[0]

@ti.kernel
def kernel_position_rotate_(target: ti.types.vector(3, float), offset: ti.types.vector(3, float), body_coords: ti.template(), start_particle_num: int, end_particle_num: int): # type: ignore
    origin =vec3f([0, 0, 1]) 
    R = RodriguesRotationMatrix(origin, target)
    for nb in range(start_particle_num, end_particle_num):
        coords = body_coords[nb]
        coords -= offset
        coords = R @ coords
        coords += offset
        body_coords[nb] = coords

@ti.kernel
def kernel_position_rotate_for_array_(
    target: ti.types.vector(3, float), 
    offset: ti.types.vector(3, float), 
    body_coords:  ti.types.ndarray(), 
    start_particle_num: int, 
    end_particle_num: int): # type: ignore
    origin =vec3f([0, 0, 1]) 
    R = RodriguesRotationMatrix(origin, target)
    for nb in range(start_particle_num, end_particle_num):
        
        coords = vec3f(body_coords[nb, 0], body_coords[nb, 1], body_coords[nb, 2])
        coords -= offset
        coords = R @ coords
        coords += offset
        body_coords[nb, 0], body_coords[nb, 1], body_coords[nb, 2] = coords[0], coords[1], coords[2]

@ti.kernel
def kernel_apply_gravity_field_(density: float, start: int, end: int, k0: float, top_pos: float, gravity: ti.types.vector(3, float), particle: ti.template()): # type: ignore
    for np in range(start, end):
        materialID = particle[np].materialID
        if materialID > 0:
            gamma = -(particle[np].x[2] - top_pos) * density * gravity[2]
            particle[np]._add_gravity_field(k0, gamma)

@ti.kernel
def kernel_apply_stress_(start: int, end: int, stress_field: ti.types.matrix(3, 3, float), particle: ti.template()):
    for np in range(start, end):
        particle[np].stress += vec6f(stress_field[0, 0], stress_field[1, 1], stress_field[2, 2], 
                                     0.5 * (stress_field[0, 1] + stress_field[1, 0]), 
                                     0.5 * (stress_field[1, 2] + stress_field[2, 1]), 
                                     0.5 * (stress_field[0, 2] + stress_field[2, 0]))

@ti.kernel
def kernel_apply_vigot_stress_(start: int, end: int, stress_field: ti.types.vector(6, float), particle: ti.template()):
    for np in range(start, end):
        particle[np].stress += stress_field

@ti.kernel
def kernel_set_particle_traction_(start: int, end: int, funcs: ti.template(), fex: ti.types.vector(3, float), particle: ti.template()):
    for np in range(start, end):
        if funcs(particle[np].x):
            particle[np]._set_particle_traction(fex)
            
@ti.kernel
def kernel_activate_cell_(start_point: ti.types.vector(3, float), region_size: ti.types.vector(3, float), nodal_coords: ti.template(), 
                          node_connectivity: ti.template(), cell_active: ti.template(), is_in_region: ti.template()):
    for nc in range(cell_active.shape[0]):
        nodeID = node_connectivity[nc]
        cell_center = vec3f([0, 0, 0])
        for i in range(nodeID.n):
            cell_center += nodal_coords[nodeID[i]]
        cell_center /= nodeID.n

        if cell_center[0] < start_point[0] or cell_center[0] > start_point[0] + region_size[0]: continue
        if cell_center[1] < start_point[1] or cell_center[1] > start_point[1] + region_size[1]: continue
        if cell_center[2] < start_point[2] or cell_center[2] > start_point[2] + region_size[2]: continue
        if is_in_region(cell_center):
            cell_active[nc] = 1

@ti.kernel
def kernel_fill_particle_in_cell_(guass_point: ti.types.ndarray(), cell_active: ti.template(), nodal_coords: ti.template(), 
                                  node_connectivity: ti.template(), particle: ti.template(), insert_particle_num: ti.template(), transform_local_to_global: ti.template()):
    for nc in range(cell_active.shape[0]):
        if cell_active[nc] == 1:
            for nparticle in range(guass_point.shape[0]):
                natural_coords = vec3f(guass_point[nparticle, 0], guass_point[nparticle, 1], guass_point[nparticle, 2])
                particle_pos = transform_local_to_global(nc, node_connectivity, nodal_coords, natural_coords)
                old_particle = ti.atomic_add(insert_particle_num[None], 1)
                particle[old_particle] = particle_pos

@ti.func
def get_particle_offset2D(np, pnum):
    ip = np % pnum[0]
    jp = np // pnum[0]   
    kp = 0
    return ip, jp, kp       

@ti.func
def get_particle_offset3D(np, pnum):
    ip = (np % (pnum[0] * pnum[1])) % pnum[0]
    jp = (np % (pnum[0] * pnum[1])) // pnum[0]
    kp = np // (pnum[0] * pnum[1])
    return ip, jp, kp    

@ti.kernel
def kernel_place_particles_(grid_size: ti.types.vector(3, float), igrid_size: ti.types.vector(3, float), start_point: ti.types.vector(3, float), region_size: ti.types.vector(3, float), new_particle_num: int, npic: int,
                            particle: ti.template(), insert_particle_num: ti.template(), is_in_region: ti.template()):
    pnum = int(region_size * npic * igrid_size)
    ti.loop_config(serialize=True)
    for np in range(new_particle_num):
        ip = (np % (pnum[0] * pnum[1])) % pnum[0]
        jp = (np % (pnum[0] * pnum[1])) // pnum[0]
        kp = np // (pnum[0] * pnum[1])
        particle_pos = (vec3f([ip, jp, kp]) + 0.5) * grid_size / npic + start_point
        if is_in_region(particle_pos):
            old_particle = ti.atomic_add(insert_particle_num[None], 1)
            particle[old_particle] = particle_pos

@ti.kernel
def kernel_add_body_(particles: ti.template(), init_particleNum: int, start_particle_num: int, end_particle_num: int, particle: ti.template(), psize: ti.types.vector(3, float), 
                     particle_volume: float, bodyID: int, materialID: int, density: float, init_v: ti.types.vector(3, float), fix_v: ti.types.vector(3, ti.u8)):
    for np in range(end_particle_num - start_particle_num):
        particleID = start_particle_num + np
        particleNum = init_particleNum + np
        particles[particleNum]._set_essential(
            bodyID, 
            materialID, 
            density, 
            particle_volume, 
            psize, 
            particle[particleID], 
            init_v, 
            fix_v)

@ti.kernel
def kernel_read_particle_file_(
    particles: ti.template(),               # already exist particles
    particleNum: int,                       # already exist particle number
    particle_num: int,                      # new particle number 
    particle: ti.types.ndarray(),           # new particle position
    psize: ti.types.ndarray(), 
    particle_volume: ti.types.ndarray(), 
    bodyID: int, 
    materialID: int,
    density: float, 
    init_v: ti.types.vector(3, float), 
    fix_v: ti.types.vector(3, int)):
    for np in range(particle_num):
        particles[particleNum + np]._set_essential(
            bodyID, 
            materialID, 
            density, 
            particle_volume[np], 
            vec3f(psize[np, 0],psize[np, 1], psize[np, 2]), 
            vec3f(particle[np, 0], particle[np, 1], particle[np, 2]), 
            init_v, 
            fix_v)

@ti.kernel
def kernel_rebulid_particle(particle_number: int, particle: ti.template(), is_rigid: ti.template(), bodyID: ti.types.ndarray(), materialID: ti.types.ndarray(), active: ti.types.ndarray(),
                            mass: ti.types.ndarray(), position: ti.types.ndarray(), velocity: ti.types.ndarray(), volume: ti.types.ndarray(), traction: ti.types.ndarray(),
                            strain: ti.types.ndarray(), stress: ti.types.ndarray(), psize: ti.types.ndarray(), velocity_gradient: ti.types.ndarray(), fix_v: ti.types.ndarray()):
    for np in range(particle_number):
        if materialID[np] == 0:
            is_rigid[bodyID[np]] = 1
        particle[np]._restart(bodyID[np], materialID[np], active[np], mass[np], vec3f(position[np, 0], position[np, 1], position[np, 2]), vec3f(velocity[np, 0], velocity[np, 1], velocity[np, 2]), volume[np], 
                              vec3f(traction[np, 0], traction[np, 1], traction[np, 2]), vec6f(strain[np, 0], strain[np, 1], strain[np, 2], strain[np, 3], strain[np, 4], strain[np, 5]), 
                              vec6f(stress[np, 0], stress[np, 1], stress[np, 2], stress[np, 3], stress[np, 4], stress[np, 5]), vec3f(psize[np, 0], psize[np, 1], psize[np, 2]), 
                              mat3x3(velocity_gradient[np, 0, 0], velocity_gradient[np, 0, 1], velocity_gradient[np, 0, 2], 
                                     velocity_gradient[np, 1, 0], velocity_gradient[np, 1, 1], velocity_gradient[np, 1, 2], 
                                     velocity_gradient[np, 2, 0], velocity_gradient[np, 2, 1], velocity_gradient[np, 2, 2]), 
                              vec3u8(fix_v[np, 0], fix_v[np, 1], fix_v[np, 2]))
        

@ti.kernel
def kernel_rebulid_particle_coupling(particle_number: int, particle: ti.template(), is_rigid: ti.template(), bodyID: ti.types.ndarray(), materialID: ti.types.ndarray(), active: ti.types.ndarray(), free_surface: ti.types.ndarray(), 
                                     normal: ti.types.ndarray(), position: ti.types.ndarray(), velocity: ti.types.ndarray(), mass: ti.types.ndarray(), volume: ti.types.ndarray(), radius: ti.types.ndarray(), 
                                     traction: ti.types.ndarray(), strain: ti.types.ndarray(), stress: ti.types.ndarray(), psize: ti.types.ndarray(), velocity_gradient: ti.types.ndarray(), fix_v: ti.types.ndarray()):
    for np in range(particle_number):
        if materialID[np] == 0:
            is_rigid[bodyID[np]] = 1
        particle[np]._restart(bodyID[np], materialID[np], active[np], free_surface[np], vec3f(normal[np, 0], normal[np, 1], normal[np, 2]), mass[np], vec3f(position[np, 0], position[np, 1], position[np, 2]),
                              vec3f(velocity[np, 0], velocity[np, 1], velocity[np, 2]), volume[np], radius[np], vec3f(traction[np, 0], traction[np, 1], traction[np, 2]), vec6f(strain[np, 0], strain[np, 1], strain[np, 2], strain[np, 3], strain[np, 4], strain[np, 5]), 
                              vec6f(stress[np, 0], stress[np, 1], stress[np, 2], stress[np, 3], stress[np, 4], stress[np, 5]), vec3f(psize[np, 0], psize[np, 1], psize[np, 2]), 
                              mat3x3(velocity_gradient[np, 0, 0], velocity_gradient[np, 0, 1], velocity_gradient[np, 0, 2], 
                                     velocity_gradient[np, 1, 0], velocity_gradient[np, 1, 1], velocity_gradient[np, 1, 2], 
                                     velocity_gradient[np, 2, 0], velocity_gradient[np, 2, 1], velocity_gradient[np, 2, 2]), 
                              vec3u8(fix_v[np, 0], fix_v[np, 1], fix_v[np, 2]))