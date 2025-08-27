import taichi as ti

from src.utils.constants import DBL_EPSILON
from src.utils.TypeDefination import vec2f, vec3f, vec6f, vec3u8, mat3x3
from src.utils.Quaternion import ThetaToRotationMatrix, ThetaToRotationMatrix2D


@ti.kernel
def kernel_calc_mass_of_center_(coords: ti.types.ndarray()) -> ti.types.vector(3, float):
    position = vec3f(0, 0, 0)
    for np in range(coords.shape[0]):
        position += vec3f(coords[np, 0], coords[np, 1], coords[np, 2])
    return position / coords.shape[0]

@ti.kernel
def kernel_calc_mass_of_center_2D(coords: ti.types.ndarray()) -> ti.types.vector(2, float):
    position = vec2f(0, 0)
    for np in range(coords.shape[0]):
        position += vec2f(coords[np, 0], coords[np, 1])
    return position / coords.shape[0]

@ti.kernel
def kernel_position_rotate_(target: ti.types.vector(3, float), offset: ti.types.vector(3, float), body_coords: ti.types.ndarray(), start_particle_num: int, end_particle_num: int):
    R = ThetaToRotationMatrix(target)
    for nb in range(start_particle_num, end_particle_num):
        coords = vec3f(body_coords[nb, 0], body_coords[nb, 1], body_coords[nb, 2])
        coords -= offset
        coords = R @ coords
        coords += offset
        body_coords[nb, 0] = coords[0]
        body_coords[nb, 1] = coords[1]
        body_coords[nb, 2] = coords[2]

@ti.kernel
def kernel_position_rotate_2D(target: ti.types.vector(2, float), offset: ti.types.vector(2, float), body_coords: ti.types.ndarray(), start_particle_num: int, end_particle_num: int):
    R = ThetaToRotationMatrix2D(target)
    for nb in range(start_particle_num, end_particle_num):
        coords = vec2f(body_coords[nb, 0], body_coords[nb, 1])
        coords -= offset
        coords = R @ coords
        coords += offset
        body_coords[nb, 0] = coords[0]
        body_coords[nb, 1] = coords[1]

@ti.kernel
def kernel_apply_stress_from_file(start: int, end: int, stress_field: ti.types.ndarray(), particle: ti.template()):
    for np in range(start, end):
        particle[np]._update_stress(vec6f(stress_field[np, 0], stress_field[np, 1], stress_field[np, 2], stress_field[np, 3], stress_field[np, 4], stress_field[np, 5]))

@ti.kernel
def kernel_apply_vigot_stress_(start: int, end: int, stress_field: ti.types.vector(6, float), particle: ti.template()):
    for np in range(start, end):
        particle[np]._update_stress(stress_field)
            
@ti.kernel
def kernel_apply_pore_pressure_(start: int, end: int, pore_pressure: float, particle: ti.template()):
    for np in range(start, end):
        particle[np].pressure += pore_pressure

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
    pnum = int(region_size * npic * igrid_size + DBL_EPSILON * grid_size / npic)                                              # for numerical error
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
def kernel_place_particles_2D(grid_size: ti.types.vector(2, float), igrid_size: ti.types.vector(2, float), start_point: ti.types.vector(2, float), region_size: ti.types.vector(2, float), new_particle_num: int, npic: int,
                              particle: ti.template(), insert_particle_num: ti.template(), is_in_region: ti.template()):
    pnum = int(region_size * npic * igrid_size + DBL_EPSILON * grid_size / npic)
    ti.loop_config(serialize=True)
    for np in range(new_particle_num):
        ip = np % pnum[0]
        jp = np // pnum[0]
        particle_pos = (vec2f([ip, jp]) + 0.5) * grid_size / npic + start_point
        if is_in_region(particle_pos):
            old_particle = ti.atomic_add(insert_particle_num[None], 1)
            particle[old_particle] = particle_pos

@ti.kernel
def kernel_add_body_(particles: ti.template(), init_particleNum: int, start_particle_num: int, end_particle_num: int, particle: ti.template(), 
                     particle_volume: float, bodyID: int, materialID: int, density: ti.types.ndarray(), init_v: ti.types.vector(3, float), fix_v: ti.types.vector(3, ti.u8)):
    for np in range(end_particle_num - start_particle_num):
        particleID = start_particle_num + np
        particleNum = init_particleNum + np
        particles[particleNum]._set_essential(particleNum, bodyID, materialID, density[np], particle_volume, particle[particleID], init_v, fix_v)

@ti.kernel
def kernel_add_body_2D(particles: ti.template(), init_particleNum: int, start_particle_num: int, end_particle_num: int, particle: ti.template(), 
                     particle_volume: float, bodyID: int, materialID: int, density: ti.types.ndarray(), init_v: ti.types.vector(2, float), fix_v: ti.types.vector(2, ti.u8)):
    for np in range(end_particle_num - start_particle_num):
        particleID = start_particle_num + np
        particleNum = init_particleNum + np
        particles[particleNum]._set_essential(particleNum, bodyID, materialID, density[np], particle_volume, particle[particleID], init_v, fix_v)

@ti.kernel
def kernel_add_body_twophase2D(particles: ti.template(), init_particleNum: int, start_particle_num: int, end_particle_num: int, particle: ti.template(), 
                     particle_volume: float, bodyID: int, materialID: int, densitys: ti.types.ndarray(), densityf: float, porosity: float, permeability: float, init_v: ti.types.vector(2, float), fix_v: ti.types.vector(2, ti.u8)):
    for np in range(end_particle_num - start_particle_num):
        particleID = start_particle_num + np
        particleNum = init_particleNum + np
        particles[particleNum]._set_essential(particleNum, bodyID, materialID, densitys[np], densityf, porosity, particle_volume, particle[particleID], init_v, fix_v, permeability)

@ti.kernel
def kernel_read_particle_file_(particles: ti.template(), particleNum: int, particle_num: int, particle: ti.types.ndarray(), particle_volume: ti.types.ndarray(), 
                               bodyID: int, materialID: int, density: ti.types.ndarray(), init_v: ti.types.vector(3, float), fix_v: ti.types.vector(3, int)):
    for np in range(particle_num):
        i = particleNum + np
        particles[i]._set_essential(i, bodyID, materialID, density[np], particle_volume[np], vec3f(particle[np, 0], particle[np, 1], particle[np, 2]), init_v, fix_v)

@ti.kernel
def kernel_read_particle_file_2D(particles: ti.template(), particleNum: int, particle_num: int, particle: ti.types.ndarray(), particle_volume: ti.types.ndarray(), 
                                 bodyID: int, materialID: int, density: ti.types.ndarray(), init_v: ti.types.vector(2, float), fix_v: ti.types.vector(2, int)):
    for np in range(particle_num):
        i = particleNum + np
        particles[i]._set_essential(i, bodyID, materialID, density[np], particle_volume[np], vec2f(particle[np, 0], particle[np, 1]), init_v, fix_v)

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
