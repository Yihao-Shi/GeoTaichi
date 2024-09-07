import taichi as ti

from src.utils.TypeDefination import vec3f
from src.utils.VectorFunction import SquaredLength


@ti.kernel
def initial_object_object(object_object: ti.template(), hist_object_object: ti.template()):
    for i in object_object:
        hist_object_object[i] = object_object[i]

@ti.kernel
def no_operation():
    pass


@ti.kernel
def board_search_particle_particle_brust_(potential_particle_num: int, paritcleNum: ti.template(), verlet_distance: float, particle: ti.template(), 
                                          potential_list_particle_particle: ti.template(), particle_particle: ti.template(), iparticle_particle: ti.template()):
    for master in range(paritcleNum[None] -1):
        pos1 = particle[master].x
        rad1 = particle[master].rad
        index = particle[master].multisphereIndex

        sques = master * potential_particle_num
        isques = (master + 1) * potential_particle_num - 1
        for slave in range(master + 1, paritcleNum):
            pos2 = particle[slave].x
            rad2 = particle[slave].rad
            same = index == particle[slave].multisphereIndex
            valid = SquaredLength(pos2, pos1) <= 1.1 * (rad1 + rad2 + verlet_distance) * 1.1 * (rad1 + rad2 + verlet_distance)
            if valid and not same: 
                if master < slave:
                    potential_list_particle_particle[sques] = slave
                    sques += 1
                elif master > slave:
                    potential_list_particle_particle[isques] = slave
                    isques -= 1
        particle_particle[master + 1] = sques - master * potential_particle_num
        iparticle_particle[master + 1] = (master + 1) * potential_particle_num - 1 - isques 

@ti.kernel
def board_search_particle_wall_brust_(potential_wall_num: int, paritcleNum: ti.template(), wallNum: int, verlet_distance: float, particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template()):
    for particle_id in range(paritcleNum[None] -1):
        position = particle[particle_id].x
        radius = particle[particle_id].rad

        sques = particle_id * potential_wall_num
        for wall_id in range(wallNum):
            if wall[wall_id]._is_sphere_intersect(position, 1.1 * (radius + verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
        particle_wall[particle_id] = sques - particle_id * potential_wall_num

@ti.func
def get_cell_index(nc, cnum):
    ig = (nc % (cnum[0] * cnum[1])) % cnum[0]
    jg = (nc % (cnum[0] * cnum[1])) // cnum[0]
    kg = nc // (cnum[0] * cnum[1])
    return ig, jg, kg

@ti.func
def get_cell_id(i, j, k, cnum):
    return int(i + j * cnum[0] + k * cnum[0] * cnum[1])

@ti.func
def get_cell_center(nc, cnum, grid_size):
    return (vec3f([get_cell_index(nc, cnum)]) + 0.5) * grid_size

@ti.kernel
def calculate_particles_position_(particleNum: int, igrid_size: float, particle: ti.template(), particle_count: ti.template(), cnum: ti.types.vector(3, int)):
    # TODO: using morton code
    particle_count.fill(0)
    for np in range(particleNum):  
        grid_idx = ti.floor(particle[np].x * igrid_size , int)
        cellID = get_cell_id(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        ti.atomic_add(particle_count[cellID], 1)
    
@ti.kernel
def insert_particle_to_cell_(igrid_size: float, particleNum: int, particle: ti.template(), particle_count: ti.template(), particle_current: ti.template(), particleID: ti.template(), cnum: ti.types.vector(3, int)):
    particle_current.fill(0)
    for np in range(particleNum):
        grid_idx = ti.floor(particle[np].x * igrid_size , int)
        cellID = get_cell_id(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        grain_location = particle_count[cellID] - ti.atomic_add(particle_current[cellID], 1) - 1
        particleID[grain_location] = np

@ti.kernel
def board_search_particle_particle_linked_cell_(particleNum: int, potential_particle_num: int, verlet_distance: float, igrid_size: float, particle_count: ti.template(), 
                                                particle_current: ti.template(), particleID: ti.template(), particle: ti.template(), potential_list_particle_particle: ti.template(), 
                                                particle_particle: ti.template(), cnum: ti.types.vector(3, int)):
    for master in range(particleNum):
        position = particle[master].x
        radius = particle[master].rad
        index = particle[master].multisphereIndex
        grid_idx = ti.floor(position * igrid_size, int)
        
        x_begin = ti.max(grid_idx[0] - 1, 0)
        x_end = ti.min(grid_idx[0] + 2, cnum[0])
        y_begin = ti.max(grid_idx[1] - 1, 0)
        y_end = ti.min(grid_idx[1] + 2, cnum[1])
        z_begin = ti.max(grid_idx[2] - 1, 0)
        z_end = ti.min(grid_idx[2] + 2, cnum[2])

        sques = master * potential_particle_num
        # isques = (master + 1) * potential_particle_num - 1
        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = get_cell_id(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID] - particle_current[cellID], particle_count[cellID]):
                        slave = particleID[hash_index]
                        if master < slave and not index == particle[slave].multisphereIndex:
                            pos2 = particle[slave].x 
                            rad2 = particle[slave].rad
                            valid = SquaredLength(pos2, position) <= (radius + rad2 + 2 * verlet_distance) * (radius + rad2 + 2 * verlet_distance)
                            if valid: 
                                potential_list_particle_particle[sques] = slave
                                sques += 1
                            # elif master > slave:
                            #     potential_list_particle_particle[isques] = slave
                            #     isques -= 1
        particle_particle[master + 1] = sques - master * potential_particle_num
        # iparticle_particle[master + 1] = (master + 1) * potential_particle_num - 1 - isques 

@ti.kernel
def board_search_coupled_particle_linked_cell_(potential_particle_num: int, verlet_distance1: float, verlet_distance2: float, max_radius: float, igrid_size: float, particle_count: ti.template(), particle_current: ti.template(), 
                                               particleID: ti.template(), particle1: ti.template(), particle2: ti.template(), potential_list_particle_particle: ti.template(), 
                                               particle_particle: ti.template(), cnum: ti.types.vector(3, int), particleNum: int):
    for master in range(particleNum):
        position = particle1[master].x
        radius = particle1[master].rad

        grid_start = ti.floor((position - radius - verlet_distance1 - verlet_distance2 - max_radius) * igrid_size, int)
        grid_end = ti.ceil((position + radius + verlet_distance1 + verlet_distance2 + max_radius) * igrid_size, int)
        x_begin = ti.max(grid_start[0], 0)
        x_end = ti.min(grid_end[0], cnum[0])
        y_begin = ti.max(grid_start[1], 0)
        y_end = ti.min(grid_end[1], cnum[1])
        z_begin = ti.max(grid_start[2], 0)
        z_end = ti.min(grid_end[2], cnum[2])

        sques = master * potential_particle_num
        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = get_cell_id(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID] - particle_current[cellID], particle_count[cellID]):
                        slave = particleID[hash_index]
                        pos2 = particle2[slave].x 
                        rad2 = particle2[slave].rad
                        search_radius = (radius + rad2 + verlet_distance1 + verlet_distance2)
                        valid = SquaredLength(pos2, position) <= search_radius * search_radius
                        if valid: 
                            potential_list_particle_particle[sques] = slave
                            sques += 1
        particle_particle[master + 1] = sques - master * potential_particle_num


'''
@ti.kernel
def board_search_particle_particle_linked_cell_(particleNum: int, cellSum: int, potential_particle_num: int, verlet_distance: float, igrid_size: float, grid_size:float, particle_count: ti.template(), 
                                                particle_current: ti.template(), particleID: ti.template(), particle: ti.template(), potential_list_particle_particle: ti.template(), particle_particle: ti.template(), 
                                                iparticle_particle: ti.template(), cnum: ti.types.vector(3, int)):
    particle_particle.fill(0)
    iparticle_particle.fill(0)
    for master in range(particleNum):
        position = particle[master].x
        radius = particle[master].rad
        index = particle[master].multisphereIndex
        grid_idx = ti.floor(position * igrid_size, int)
        
        grid_x = (grid_idx + 0.5) * grid_size
        offset = vec3i([0, 0, 0])
        for k in ti.static(range(3)):
            d = position[k] - grid_x[k]
            if(d > 0): offset[k] = 1
            else: offset[k] = -1

        sques = master * potential_particle_num
        isques = (master + 1) * potential_particle_num - 1
        particle_particle_verlet_table_within_target_cell(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum, 
                                                          verlet_distance, master, grid_idx[0], grid_idx[1], grid_idx[2], position, radius, index, sques, isques)
        particle_particle_verlet_table(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum,
                                       cellSum, verlet_distance, master, grid_idx[0] + offset[0], grid_idx[1], grid_idx[2], position, radius, index, sques, isques)
        particle_particle_verlet_table(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum,
                                       cellSum, verlet_distance, master, grid_idx[0], grid_idx[1] + offset[1], grid_idx[2], position, radius, index, sques, isques)
        particle_particle_verlet_table(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum,
                                       cellSum, verlet_distance, master, grid_idx[0], grid_idx[1], grid_idx[2] + offset[2], position, radius, index, sques, isques)
        particle_particle_verlet_table(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum,
                                       cellSum, verlet_distance, master, grid_idx[0], grid_idx[1] + offset[1], grid_idx[2] + offset[2], position, radius, index, sques, isques)
        particle_particle_verlet_table(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum,
                                       cellSum, verlet_distance, master, grid_idx[0] + offset[0], grid_idx[1] + offset[1], grid_idx[2], position, radius, index, sques, isques)
        particle_particle_verlet_table(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum,
                                       cellSum, verlet_distance, master, grid_idx[0] + offset[0], grid_idx[1], grid_idx[2] + offset[2], position, radius, index, sques, isques)
        particle_particle_verlet_table(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum,
                                       cellSum, verlet_distance, master, grid_idx[0] + offset[0], grid_idx[1] + offset[1], grid_idx[2] + offset[2], position, radius, index, sques, isques)

@ti.func
def particle_particle_verlet_table_within_target_cell(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum,
                                                      verlet_distance, master, neigh_i, neigh_j, neigh_k, pos1, rad1, index, sques: ti.template(), isques: ti.template()):
    cellID = get_cell_id(neigh_i, neigh_j, neigh_k, cnum)
    for hash_index in range(particle_count[cellID] - particle_current[cellID], particle_count[cellID]):
        slave = particleID[hash_index]
        pos2 = particle[slave].x 
        rad2 = particle[slave].rad
        same = index == particle[slave].multisphereIndex
        valid = SquaredLength(pos2, pos1) <= 1.1 * (rad1 + rad2 + verlet_distance) * 1.1 * (rad1 + rad2 + verlet_distance)
        if valid and not same: 
            if master < slave:
                potential_list_particle_particle[sques] = slave
                sques += 1
            elif master > slave:
                potential_list_particle_particle[isques] = slave
                isques -= 1
            
@ti.func
def particle_particle_verlet_table(potential_list_particle_particle, particle_count, particle_current, particleID, particle, cnum,
                                   cellSum, verlet_distance, master, neigh_i, neigh_j, neigh_k, pos1, rad1, index, sques: ti.template(), isques: ti.template()):
    cellID = get_cell_id(neigh_i, neigh_j, neigh_k, cnum)
    if 0 <= cellID < cellSum: 
        for hash_index in range(particle_count[cellID] -particle_current[cellID], particle_count[cellID]):
            slave = particleID[hash_index]
            pos2 = particle[slave].x 
            rad2 = particle[slave].rad
            same = index == particle[slave].multisphereIndex
            valid = SquaredLength(pos2, pos1) <= 1.1 * (rad1 + rad2 + verlet_distance) * 1.1 * (rad1 + rad2 + verlet_distance)
            if valid and not same: 
                if master < slave:
                    potential_list_particle_particle[sques] = slave
                    sques += 1
                elif master > slave:
                    potential_list_particle_particle[isques] = slave
                    isques -= 1
'''        
# ============================================ Plane ================================================= #
@ti.kernel
def insert_plane_to_cell_(factor: float, wallNum: int, grid_size: float, cellSum: int, plane_in_cell: int, wall_count: ti.template(), wallID: ti.template(), wall: ti.template(), cnum: ti.types.vector(3, int)):
    wallID.fill(0)
    wall_count.fill(0)
    for nc in range(cellSum):
        cell_center = get_cell_center(nc, cnum, grid_size)
        for nw in range(wallNum):
            if int(wall[nw].active) == 1:
                distance = wall[nw]._point_to_wall_distance(cell_center)
                if distance <= factor * grid_size:
                    wall_location = nc * plane_in_cell + ti.atomic_add(wall_count[nc], 1)
                    wallID[wall_location] = nw

@ti.kernel
def board_search_particle_plane_linked_cell_(particleNum: int, potential_wall_num: int, plane_in_cell: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), wallID: ti.template(), 
                                             particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int)):
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_idx = ti.floor(position * igrid_size, int)
        
        sques = particle_id * potential_wall_num
        cellID = get_cell_id(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        to_beg, to_end = cellID * plane_in_cell, cellID * plane_in_cell + wall_count[cellID]
        for hash_index in range(to_beg, to_end):
            wall_id = wallID[hash_index]
            if wall[wall_id]._is_sphere_intersect(position, (radius + verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        particle_wall[particle_id + 1] = sques - particle_id * potential_wall_num

@ti.kernel
def board_search_coupled_particle_plane_linked_cell_(potential_wall_num: int, plane_in_cell: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), wallID: ti.template(), particle: ti.template(), wall: ti.template(), 
                                                     potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int), particleNum: int):
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_idx = ti.floor(position * igrid_size, int)
        
        sques = particle_id * potential_wall_num
        cellID = get_cell_id(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        to_beg, to_end = cellID * plane_in_cell, cellID * plane_in_cell + wall_count[cellID]
        for hash_index in range(to_beg, to_end):
            wall_id = wallID[hash_index]
            if wall[wall_id]._is_sphere_intersect(position, (radius + verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        particle_wall[particle_id + 1] = sques - particle_id * potential_wall_num

# ======================================== Facet ============================================= #
@ti.kernel
def insert_facet_to_cell_(wallNum: int, igrid_size: float, facet_in_cell: int, wall_count: ti.template(), wallID: ti.template(), wall: ti.template(), cnum: ti.types.vector(3, int)):
    wall_count.fill(0)
    for wall_id in range(wallNum):
        if int(wall[wall_id].active) == 1:
            wall[wall_id]._bounding_box()
            wall_min_coord = wall[wall_id].bound_beg
            wall_max_coord = wall[wall_id].bound_end

            minCoord = ti.max(ti.floor(wall_min_coord * igrid_size - 0.5, int), 0)
            maxCoord = ti.min(ti.ceil(wall_max_coord * igrid_size + 0.5, int) + 1, cnum)
            for neigh_i in range(minCoord[0], maxCoord[0]):
                for neigh_j in range(minCoord[1], maxCoord[1]): 
                    for neigh_k in range(minCoord[2], maxCoord[2]): 
                        cellID = get_cell_id(neigh_i, neigh_j, neigh_k, cnum)
                        wall_location = cellID * facet_in_cell + ti.atomic_add(wall_count[cellID], 1)                       
                        wallID[wall_location] = wall_id

@ti.kernel
def board_search_particle_facet_linked_cell_(particleNum: int, potential_wall_num: int, facet_in_cell: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), wallID: ti.template(), 
                                             particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int)):
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_idx = ti.floor(position * igrid_size, int)

        sques = particle_id * potential_wall_num
        cellID = get_cell_id(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        to_beg, to_end = cellID * facet_in_cell, cellID * facet_in_cell + wall_count[cellID]
        for hash_index in range(to_beg, to_end):
            wall_id = wallID[hash_index]
            if wall[wall_id]._is_sphere_intersect(position, (radius + 2 * verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        particle_wall[particle_id + 1] = sques - particle_id * potential_wall_num

@ti.kernel
def board_search_coupled_particle_facet_linked_cell_(potential_wall_num: int, facet_in_cell: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), wallID: ti.template(), particle: ti.template(), wall: ti.template(), 
                                                     potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int), particleNum: int):
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_idx = ti.floor(position * igrid_size, int)

        sques = particle_id * potential_wall_num
        cellID = get_cell_id(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        to_beg, to_end = cellID * facet_in_cell, cellID * facet_in_cell + wall_count[cellID]
        for hash_index in range(to_beg, to_end):
            wall_id = wallID[hash_index]
            if wall[wall_id]._is_sphere_intersect(position, (radius + 2 * verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        particle_wall[particle_id + 1] = sques - particle_id * potential_wall_num

# ======================================== Patch ============================================= #
@ti.kernel
def calculate_patch_position_(wallNum: int, igrid_size: float, wall: ti.template(), wall_count: ti.template(), cnum: ti.types.vector(3, int)):
    wall_count.fill(0)
    for wall_id in range(wallNum):  
        if int(wall[wall_id].active) == 1:
            grid_idx = ti.floor(wall[wall_id]._get_center() * igrid_size , int)
            cellID = get_cell_id(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
            ti.atomic_add(wall_count[cellID], 1)

@ti.kernel
def insert_patch_to_cell_(igrid_size: float, wallNum: int, wall: ti.template(), wall_count: ti.template(), patch_current: ti.template(), wallID: ti.template(), cnum: ti.types.vector(3, int)):
    patch_current.fill(0)
    for nw in range(wallNum):
        if int(wall[nw].active) == 1:
            grid_idx = ti.floor(wall[nw]._get_center() * igrid_size , int)
            cellID = get_cell_id(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
            patch_location = wall_count[cellID] - ti.atomic_add(patch_current[cellID], 1) - 1
            wallID[patch_location] = nw

@ti.kernel
def board_search_particle_patch_linked_cell_(particleNum: int, potential_wall_num: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), patch_current: ti.template(), wallID: ti.template(), 
                                             particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int)):
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad
        grid_idx = ti.floor(position * igrid_size, int)

        x_begin = ti.max(grid_idx[0] - 1, 0)
        x_end = ti.min(grid_idx[0] + 2, cnum[0])
        y_begin = ti.max(grid_idx[1] - 1, 0)
        y_end = ti.min(grid_idx[1] + 2, cnum[1])
        z_begin = ti.max(grid_idx[2] - 1, 0)
        z_end = ti.min(grid_idx[2] + 2, cnum[2])

        sques = particle_id * potential_wall_num
        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = get_cell_id(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(wall_count[cellID] - patch_current[cellID], wall_count[cellID]):
                        wall_id = wallID[hash_index]
                        valid = wall[wall_id]._is_sphere_intersect(position, (radius + verlet_distance))
                        if valid: 
                            potential_list_particle_wall[sques] = wall_id
                            sques += 1
        particle_wall[particle_id + 1] = sques - particle_id * potential_wall_num


@ti.kernel
def board_search_coupled_particle_patch_linked_cell_(potential_wall_num: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), patch_current: ti.template(), wallID: ti.template(), 
                                                     particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int),
                                                     particleNum: int):
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad
        grid_idx = ti.floor(position * igrid_size, int)
        
        x_begin = ti.max(grid_idx[0] - 1, 0)
        x_end = ti.min(grid_idx[0] + 2, cnum[0])
        y_begin = ti.max(grid_idx[1] - 1, 0)
        y_end = ti.min(grid_idx[1] + 2, cnum[1])
        z_begin = ti.max(grid_idx[2] - 1, 0)
        z_end = ti.min(grid_idx[2] + 2, cnum[2])

        sques = particle_id * potential_wall_num
        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = get_cell_id(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(wall_count[cellID] - patch_current[cellID], wall_count[cellID]):
                        wall_id = wallID[hash_index]
                        valid = wall[wall_id]._is_sphere_intersect(position, (radius + verlet_distance))
                        if valid: 
                            potential_list_particle_wall[sques] = wall_id
                            sques += 1
        particle_wall[particle_id + 1] = sques - particle_id * potential_wall_num

'''
@ti.kernel
def board_search_particle_patch_linked_cell_(potential_wall_num: int, particleNum: int, cellSum: int, verlet_distance: float, igrid_size: float, grid_size: float, wall_count: ti.template(), patch_current: ti.template(), wallID: ti.template(), 
                                             particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int)):
    particle_wall.fill(0)
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad
        grid_idx = ti.floor(position * igrid_size, int)

        grid_x = (grid_idx + 0.5) * grid_size
        offset = vec3i[(0, 0, 0)]
        for k in ti.static(range(3)):
            d = position[k] - grid_x[k]
            if(d > 0): offset[k] = 1
            else: offset[k] = -1

        sques = particle_id * potential_wall_num
        particle_patch_verlet_table(potential_list_particle_wall, wall_count, patch_current, wallID, particle, wall, cnum, cellSum,
                                    verlet_distance, grid_idx[0], grid_idx[1], grid_idx[2], position, radius, sques)
        particle_patch_verlet_table(potential_list_particle_wall, wall_count, patch_current, wallID, particle, wall, cnum, cellSum, 
                                    verlet_distance, grid_idx[0] + offset[0], grid_idx[1], grid_idx[2], position, radius, sques)
        particle_patch_verlet_table(potential_list_particle_wall, wall_count, patch_current, wallID, particle, wall, cnum, cellSum, 
                                    verlet_distance, grid_idx[0], grid_idx[1] + offset[1], grid_idx[2], position, radius, sques)
        particle_patch_verlet_table(potential_list_particle_wall, wall_count, patch_current, wallID, particle, wall, cnum, cellSum, 
                                    verlet_distance, grid_idx[0], grid_idx[1], grid_idx[2] + offset[2], position, radius, sques)
        particle_patch_verlet_table(potential_list_particle_wall, wall_count, patch_current, wallID, particle, wall, cnum, cellSum, 
                                    verlet_distance, grid_idx[0], grid_idx[1] + offset[1], grid_idx[2] + offset[2], position, radius, sques)
        particle_patch_verlet_table(potential_list_particle_wall, wall_count, patch_current, wallID, particle, wall, cnum, cellSum, 
                                    verlet_distance, grid_idx[0] + offset[0], grid_idx[1] + offset[1], grid_idx[2], position, radius, sques)
        particle_patch_verlet_table(potential_list_particle_wall, wall_count, patch_current, wallID, particle, wall, cnum, cellSum, 
                                    verlet_distance, grid_idx[0] + offset[0], grid_idx[1], grid_idx[2] + offset[2], position, radius, sques)
        particle_patch_verlet_table(potential_list_particle_wall, wall_count, patch_current, wallID, particle, wall, cnum, cellSum, 
                                    verlet_distance, grid_idx[0] + offset[0], grid_idx[1] + offset[1], grid_idx[2] + offset[2], position, radius, sques)
        particle_wall[particle_id + 1] = sques - particle_id * particle_id

@ti.func
def particle_patch_verlet_table(potential_list_particle_wall, wall_count, patch_current, wallID, wall, cnum,
                                cellSum, verlet_distance, neigh_i, neigh_j, neigh_k, position, radius, sques: ti.template()):
    cellID = get_cell_id(neigh_i, neigh_j, neigh_k, cnum)
    if 0 <= cellID < cellSum:
        for hash_index in range(wall_count[cellID] - patch_current[cellID], wall_count[cellID]):
            wall_id = wallID[hash_index]
            valid = wall[wall_id]._is_sphere_intersect(position, 1.1 * (radius + verlet_distance))
            if valid: 
                potential_list_particle_wall[sques] = wall_id
                sques += 1
'''
@ti.kernel
def get_intersection_type_(particle_wall: int, particleNum: int, potential_wall_num: int, potential_list_particle_wall: ti.template(), prefix_sum_particle_wall: ti.template(), 
                           potential_list_particle_particle: ti.template(), particle: ti.template(), wall: ti.template()):
    for cwlist in range(particle_wall):
        end1, end2 = potential_list_particle_wall[cwlist].end1, potential_list_particle_wall[cwlist].end2
        pos, rad = particle[end1].x, particle[end1].rad
        valid = wall[end2]._is_sphere_intersect(pos, rad) 
        potential_list_particle_wall[cwlist].wall_type = valid

    for particle_id in range(particleNum):
        to_beg = particle_id * potential_wall_num
        to_end = to_beg + prefix_sum_particle_wall[particle_id]

        for cwlist_i in range(to_beg, to_end):
            contact_type1 = potential_list_particle_wall[cwlist_i].wall_type
            if contact_type1 > 0:
                wall_id1 = potential_list_particle_particle[cwlist_i].end2
                norm1 = wall[wall_id1].norm
                for cwlist_j in range(cwlist_i + 1, to_end):
                    contact_type2 = potential_list_particle_wall[cwlist_j].wall_type
                    wall_id2 = potential_list_particle_particle[cwlist_j].end2
                    norm2 = wall[wall_id2].norm
                    if contact_type2 > 0 and SquaredLength(norm1, norm2) < 1e-6:
                        if contact_type1 == 1 and (contact_type2 == 2 or contact_type2 == 3):
                            contact_type2 = 0
                        elif contact_type1 == 2:
                            if contact_type2 == 2: contact_type2 = 0
                            elif contact_type2 == 1: contact_type1 = 0
                            else: contact_type2 = 0
                        elif contact_type1 == 3: contact_type1 = 0
                    potential_list_particle_wall[cwlist_j].wall_type = contact_type2
            potential_list_particle_wall[cwlist_i].wall_type = contact_type1
