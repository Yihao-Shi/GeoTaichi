import taichi as ti

from src.utils.constants import Threshold, ZEROVEC3f
from src.utils.GeometryFunction import intersectionOBBs
from src.utils.TypeDefination import vec3f, vec3i, vec2i
from src.utils.Quaternion import SetToRotate
from src.utils.VectorFunction import SquaredLength, SquareLen
from src.utils.ScalarFunction import linearize3D, vectorize_id
import src.utils.GlobalVariable as GlobalVariable


@ti.kernel
def reset_relative_displacement(particleNum: int, object_object: ti.template(), oblist: ti.template()):
    total_contact = ti.max(object_object[particleNum], 1)
    for nc in range(total_contact):
        oblist[nc].verletDisp = ZEROVEC3f

@ti.kernel
def validate_pprelative_displacement_(limit: float, particleNum: int, dt: ti.template(), particle: ti.template(), rigid: ti.template(), particle_particle: ti.template(), pplist: ti.template()) -> int:
    total_contact = ti.max(particle_particle[particleNum], 1)
    flag = 0 
    for nc in range(total_contact):
        end1, end2 = pplist[nc].endID1, pplist[nc].endID2
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad
        vel1, vel2 = rigid[end1]._get_velocity(), rigid[end2]._get_velocity()
        w1, w2 = rigid[end1]._get_angular_velocity(), rigid[end2]._get_angular_velocity()
        
        norm = (pos1 - pos2).normalized(Threshold)
        v_rel = vel1 + w1.cross(-norm) * rad1 - (vel2 + w2.cross(norm) * rad2)
        verletDisp = pplist[nc].verletDisp + v_rel * dt[None]
        if flag == 0 and SquareLen(verletDisp) >= limit:
            flag = 1
        pplist[nc].verletDisp = verletDisp
    return flag


@ti.kernel
def validate_pwrelative_displacement_(limit: float, particleNum: int, dt: ti.template(), particle: ti.template(), wall: ti.template(), rigid: ti.template(), particle_wall: ti.template(), pwlist: ti.template()) -> int:
    total_contact = ti.max(particle_wall[particleNum], 1)
    flag = 0 
    for nc in range(total_contact):
        end1, end2 = pwlist[nc].endID1, pwlist[nc].endID2
        particle_rad = particle[end1]._get_radius()
        vel1, vel2 = rigid[end1]._get_velocity(), wall[end2]._get_velocity()
        w1, norm = rigid[end1]._get_angular_velocity(), wall[end2].norm

        v_rel = vel1 + w1.cross(-norm) * particle_rad - vel2 
        verletDisp = pwlist[nc].verletDisp + v_rel * dt[None]
        if flag == 0 and SquareLen(verletDisp) >= limit:
            flag = 1
        pwlist[nc].verletDisp = verletDisp
    return flag

@ti.kernel
def initial_object_object(object_object: ti.template(), hist_object_object: ti.template()):
    for i in hist_object_object:
        hist_object_object[i] = object_object[i]

# ================================================================= #
#                                                                   #
#                         Brust Search                              #
#                                                                   #
# ================================================================= #
@ti.kernel
def board_search_particle_particle_brust_(potential_particle_num: int, particleNum: int, verlet_distance: float, particle: ti.template(), 
                                          potential_list_particle_particle: ti.template(), particle_particle: ti.template()):
    for master in range(particleNum -1):
        if int(particle[master].active) == 0: continue
        pos1 = particle[master].x
        rad1 = particle[master].rad
        index = particle[master]._get_multisphere_index1()

        sques = master * potential_particle_num
        for slave in range(master + 1, particleNum):
            if master < slave and not index == particle[slave]._get_multisphere_index2():
                pos2 = particle[slave].x
                rad2 = particle[slave].rad
                valid = SquaredLength(pos2, pos1) <= (rad1 + rad2 + 2 * verlet_distance) * (rad1 + rad2 + 2 * verlet_distance)
                if valid: 
                    potential_list_particle_particle[sques] = slave
                    sques += 1
        particle_particle[master + 1] = sques - master * potential_particle_num

@ti.kernel
def board_search_particle_wall_brust_(potential_wall_num: int, particleNum: int, wallNum: int, verlet_distance: float, particle: ti.template(), 
                                      wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template()):
    for particle_id in range(particleNum):
        if int(particle[particle_id].active) == 0: continue
        position = particle[particle_id].x
        radius = particle[particle_id].rad

        sques = particle_id * potential_wall_num
        for wall_id in range(wallNum):
            if wall[wall_id]._is_sphere_intersect(position, (radius + 2. * verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        neighbors = sques - particle_id * potential_wall_num
        assert neighbors <= potential_wall_num, f"Keyword:: DEMPM /body_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors


@ti.kernel
def board_search_particle_wall_brust_hierarchical_(particleNum: int, wallNum: int, verlet_distance: float, particle: ti.template(), wall: ti.template(), 
                                                   body: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template()):
    for particle_id in range(particleNum):
        if int(particle[particle_id].active) == 0: continue
        position = particle[particle_id].x
        radius = particle[particle_id].rad
        potential_wall_num = body[particle_id].potential_wall_num()

        sques = potential_wall_num
        for wall_id in range(wallNum):
            if wall[wall_id]._is_sphere_intersect(position, (radius + 2. * verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        neighbors = sques - potential_wall_num
        assert neighbors <= body[particle_id + 1].potential_wall_num() - body[particle_id].potential_wall_num(), f"Keyword:: DEMPM /body_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors


# ================================================================= #
#                                                                   #
#                          Linked Cell                              #
#                                                                   #
# ================================================================= #
@ti.func
def get_cell_center(nc, cnum, grid_size):
    return (vec3f([vectorize_id(nc, cnum)]) + 0.5) * grid_size

@ti.kernel
def calculate_particles_position_(particleNum: int, igrid_size: float, particle: ti.template(), particle_count: ti.template(), particle_current: ti.template(), cnum: ti.types.vector(3, int)):
    # TODO: using morton code
    particle_count.fill(0)
    for np in range(particleNum):  
        if int(particle[np].active) == 0: continue
        position = particle[np].x
        grid_idx = ti.floor(position * igrid_size , int)
        assert 0 <= grid_idx[0] < cnum[0] and 0 <= grid_idx[1] < cnum[1] and 0 <= grid_idx[2] < cnum[2], f"Particle {np} is located at [{position[0]}, {position[1]}, {position[2]}]. Out of simulation domain!"
        cellID = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        particle_current[np] = ti.atomic_add(particle_count[cellID + 1], 1)
    
@ti.kernel
def insert_particle_to_cell_(igrid_size: float, particleNum: int, particle: ti.template(), particle_count: ti.template(), particle_current: ti.template(), particleID: ti.template(), cnum: ti.types.vector(3, int)):
    for np in range(particleNum):
        if int(particle[np].active) == 0: continue
        grid_idx = ti.floor(particle[np].x * igrid_size, int)
        cellID = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        grain_location = particle_count[cellID] + particle_current[np]
        particleID[grain_location] = np

@ti.kernel
def board_search_particle_particle_linked_cell_(particleNum: int, potential_particle_num: int, verlet_distance: float, igrid_size: float, particle_count: ti.template(), 
                                                particleID: ti.template(), particle: ti.template(), potential_list_particle_particle: ti.template(), 
                                                particle_particle: ti.template(), cnum: ti.types.vector(3, int)):
    particle_particle.fill(0)
    for master in range(particleNum):
        if int(particle[master].active) == 0: continue
        position = particle[master].x
        radius = particle[master].rad
        index = particle[master]._get_multisphere_index1()
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
                    cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                        slave = particleID[hash_index]
                        if master < slave and not index == particle[slave]._get_multisphere_index2():
                            pos2 = particle[slave].x 
                            rad2 = particle[slave].rad
                            valid = SquaredLength(pos2, position) <= (radius + rad2 + 2 * verlet_distance) * (radius + rad2 + 2 * verlet_distance)
                            if valid: 
                                potential_list_particle_particle[sques] = slave
                                sques += 1
                            # elif master > slave:
                            #     potential_list_particle_particle[isques] = slave
                            #     isques -= 1
        neighbors = sques - master * potential_particle_num
        assert neighbors <= potential_particle_num, f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors
        # iparticle_particle[master + 1] = (master + 1) * potential_particle_num - 1 - isques 

@ti.kernel
def board_search_lsparticle_lsparticle_linked_cell_(particleNum: int, potential_point_num: int, verlet_distance: float, pplist: ti.template(), potential_list_point_particle: ti.template(), particle_particle: ti.template(), 
                                                    point_particle: ti.template(), rigid: ti.template(), box: ti.template(), vertice: ti.template(), grid: ti.template()):
    point_particle.fill(0)
    total_contact = particle_particle[particleNum]
    for nc in range(total_contact):
        master, slave = pplist[nc].endID1, pplist[nc].endID2
        rotate_matrix1, rotate_matrix2 = SetToRotate(rigid[master].q), SetToRotate(rigid[slave].q)
        mass_center1, mass_center2 = rigid[master]._get_position(), rigid[slave]._get_position()
        if intersectionOBBs(mass_center1, mass_center2, box[master]._get_dim(), box[slave]._get_dim(), rotate_matrix1, rotate_matrix2):
            scale = box[master].scale
            start_node, end_node, global_node = rigid[master]._start_node(), rigid[master]._end_node(), rigid[master].startNode
            for node in range(start_node, end_node):
                gnode = node - start_node + global_node
                surface_node = rotate_matrix2.transpose() @ (mass_center1 + rotate_matrix1 @ (scale * vertice[node].x) - mass_center2)
                if not box[slave]._in_box(surface_node): continue
                if box[slave].distance(surface_node, grid) < verlet_distance: 
                    sques = ti.atomic_add(point_particle[gnode + 1], 1)
                    potential_list_point_particle[sques + gnode * potential_point_num] = slave
                    assert sques < potential_point_num, f"Keyword:: /point_coordination_numbers[0]/ is too small, Node {gnode} on particle {master} has {sques+1} potential contact number"
    '''
    for master in range(particleNum):
        mass_center1 = rigid[master]._get_position()
        rotate_matrix1 = SetToRotate(rigid[master].q)
        scale = box[master].scale
        start_node, end_node, global_node = rigid[master]._start_node(), rigid[master]._end_node(), rigid[master].startNode
        sques = 0
        for neigh in range(particle_particle[master], particle_particle[master + 1]):
            slave = pplist[neigh].endID2
            mass_center2 = rigid[slave]._get_position()
            rotate_matrix2 = SetToRotate(rigid[slave].q)
            if intersectionOBBs(mass_center1, mass_center2, box[master]._get_dim(), box[slave]._get_dim(), rotate_matrix1, rotate_matrix2):
                for node in range(start_node, end_node):
                    gnode = node - start_node + global_node
                    surface_node = rotate_matrix2.transpose() @ (mass_center1 + rotate_matrix1 @ (scale * vertice[node].x) - mass_center2)
                    if not box[slave]._in_box(surface_node): continue
                    if box[slave].distance(surface_node, grid) < verlet_distance: 
                        potential_list_point_particle[sques + master * potential_point_num]._set(gnode, slave)
                        sques += 1
        assert sques <= potential_point_num, f"Keyword:: /point_coordination_numbers[0]/ is too small, Particle {master} has {sques} potential contact number"
        point_particle[master + 1] = sques'''

@ti.kernel
def board_search_coupled_particle_linked_cell_(potential_particle_num: int, verlet_distance1: float, verlet_distance2: float, max_radius: float, igrid_size: float, particle_count: ti.template(), 
                                               particleID: ti.template(), particle1: ti.template(), particle2: ti.template(), potential_list_particle_particle: ti.template(), 
                                               particle_particle: ti.template(), cnum: ti.types.vector(3, int), particleNum: int):
    particle_particle.fill(0)
    for master in range(particleNum):
        if int(particle1[master].active) != 0 or int(particle1[master].coupling) != 0:
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
                        cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                            slave = particleID[hash_index]
                            pos2 = particle2[slave].x 
                            rad2 = particle2[slave].rad
                            search_radius = (radius + rad2 + verlet_distance1 + verlet_distance2)
                            valid = SquaredLength(pos2, position) <= search_radius * search_radius
                            if valid: 
                                potential_list_particle_particle[sques] = slave
                                sques += 1
            neighbors = sques - master * potential_particle_num
            assert neighbors <= potential_particle_num, f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
            particle_particle[master + 1] = neighbors

@ti.kernel
def board_search_enhanced_coupled_particle_linked_cell_(potential_particle_num: int, verlet_distance1: float, verlet_distance2: float, max_radius: float, igrid_size: float, particle_count: ti.template(), 
                                               particleID: ti.template(), particle1: ti.template(), particle2: ti.template(), potential_list_particle_particle: ti.template(), 
                                               particle_particle: ti.template(), cnum: ti.types.vector(3, int), particleNum: int):
    particle_particle.fill(0)
    for master in range(particleNum):
        if int(particle1[master].active) == 0: continue
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
                    cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                        slave = particleID[hash_index]
                        pos2 = particle2[slave].x 
                        rad2 = particle2[slave].rad
                        norm = particle2[slave].normal
                        search_radius = (radius + rad2 + verlet_distance1 + verlet_distance2)
                        valid = (position - pos2).dot(norm) <= search_radius
                        if valid: 
                            potential_list_particle_particle[sques] = slave
                            sques += 1
        neighbors = sques - master * potential_particle_num
        assert neighbors <= potential_particle_num, f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors


@ti.kernel
def board_search_coupled_lsparticle_linked_cell_(potential_particle_num: int, verlet_distance1: float, verlet_distance2: float, max_radius: float, igrid_size: float, particle_count: ti.template(), 
                                                 particleID: ti.template(), particle1: ti.template(), rigid: ti.template(), box: ti.template(), grid: ti.template(),
                                                 potential_list_particle_particle: ti.template(), particle_particle: ti.template(), cnum: ti.types.vector(3, int), particleNum: int):
    particle_particle.fill(0)
    for master in range(particleNum):
        if int(particle1[master].active) == 0 or int(particle1[master].coupling) == 0: continue
        position = particle1[master].x
        radius = particle1[master].rad

        grid_start = ti.floor((position - radius - verlet_distance1 - verlet_distance2 - max_radius) * igrid_size, int)
        grid_end = ti.ceil((position + radius + verlet_distance1 + verlet_distance2 + max_radius) * igrid_size + 1, int)
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
                    cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                        slave = particleID[hash_index]
                        mass_center = rigid[slave]._get_position()
                        rotate_matrix = SetToRotate(rigid[slave].q)
                        surface_node = rotate_matrix.transpose() @ (position - mass_center)
                        verlet_distance = verlet_distance1 + verlet_distance2
                        if not box[slave]._in_box(surface_node): continue
                        if box[slave].distance(surface_node, grid) < verlet_distance + radius: 
                            potential_list_particle_particle[sques] = slave
                            sques += 1
        neighbors = sques - master * potential_particle_num
        assert neighbors <= potential_particle_num, f"Keyword:: DEMPM /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors

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
        index = particle[master]._get_multisphere_index1()
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
    cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
    for hash_index in range(particle_count[cellID] - particle_current[cellID], particle_count[cellID]):
        slave = particleID[hash_index]
        pos2 = particle[slave].x 
        rad2 = particle[slave].rad
        same = index == particle[slave]._get_multisphere_index2()
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
    cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
    if 0 <= cellID < cellSum: 
        for hash_index in range(particle_count[cellID] -particle_current[cellID], particle_count[cellID]):
            slave = particleID[hash_index]
            pos2 = particle[slave].x 
            rad2 = particle[slave].rad
            same = index == particle[slave]._get_multisphere_index2()
            valid = SquaredLength(pos2, pos1) <= 1.1 * (rad1 + rad2 + verlet_distance) * 1.1 * (rad1 + rad2 + verlet_distance)
            if valid and not same: 
                if master < slave:
                    potential_list_particle_particle[sques] = slave
                    sques += 1
                elif master > slave:
                    potential_list_particle_particle[isques] = slave
                    isques -= 1
'''        

# ============================================ Wall ================================================= #
@ti.kernel
def board_search_lsparticle_wall_linked_cell_(particleNum: int, potential_point_num: int, verlet_distance: float, pwlist: ti.template(), potential_list_point_wall: ti.template(), 
                                               particle_wall: ti.template(), point_wall: ti.template(), wall: ti.template(), rigid: ti.template(), vertice: ti.template(), box: ti.template()):
    point_wall.fill(0)
    total_contact = particle_wall[particleNum]
    for nc in range(total_contact):
        master, wall_id = pwlist[nc].endID1, pwlist[nc].endID2
        mass_center1 = rigid[master]._get_position()
        rotate_matrix1 = SetToRotate(rigid[master].q)
        start_node, end_node, global_node = rigid[master]._start_node(), rigid[master]._end_node(), rigid[master].startNode
        scale = box[master].scale
        for node in range(start_node, end_node):
            gnode = node - start_node + global_node
            surface_node = mass_center1 + rotate_matrix1 @ (scale * vertice[node].x)
            projected_point = wall[wall_id]._point_projection(surface_node)
            if wall[wall_id]._is_in_plane(projected_point) and wall[wall_id]._is_sphere_intersect(surface_node, verlet_distance) == 1:
                point_wall[gnode + 1] = 1
                sques = 0#ti.atomic_add(point_wall[gnode + 1], 1)
                potential_list_point_wall[sques + gnode * potential_point_num] = wall_id
                assert sques < potential_point_num, f"Keyword:: /point_coordination_number[1]/ is too small, Node {gnode} has {sques+1} potential contact number"
    
    '''
    for master in range(particleNum):
        mass_center1 = rigid[master]._get_position()
        rotate_matrix1 = SetToRotate(rigid[master].q)
        scale = box[master].scale
        start_node, end_node, global_node = rigid[master]._start_node(), rigid[master]._end_node(), rigid[master].startNode
        sques = 0

        for node in range(start_node, end_node):
            gnode = node - start_node + global_node
            surface_node = mass_center1 + rotate_matrix1 @ (scale * vertice[node].x)

            target_wall_id = -1
            min_dist = 0.
            for neigh in range(particle_wall[master], particle_wall[master + 1]):
                wall_id = pwlist[neigh].endID2
                projected_point = wall[wall_id]._point_projection(surface_node)
                if wall[wall_id]._is_in_plane(projected_point) and wall[wall_id]._is_sphere_intersect(surface_node, verlet_distance) == 1:
                    dist = wall[wall_id]._get_norm_distance(surface_node)
                    if min_dist < dist:
                        min_dist = dist
                        target_wall_id = wall_id
            if target_wall_id > -1:
                potential_list_point_wall[gnode * potential_point_num] = target_wall_id
                point_wall[master + 1] = 1'''

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
                    assert wall_count[nc] <= plane_in_cell, f"Keyword:: /wall_per_cell/ = {plane_in_cell} is too small, at least {wall_count[nc]}"

@ti.kernel
def board_search_particle_plane_linked_cell_(particleNum: int, potential_wall_num: int, plane_in_cell: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), wallID: ti.template(), 
                                             particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int)):
    particle_wall.fill(0)
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_idx = ti.floor(position * igrid_size, int)
        
        sques = particle_id * potential_wall_num
        cellID = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        to_beg, to_end = cellID * plane_in_cell, cellID * plane_in_cell + wall_count[cellID]
        for hash_index in range(to_beg, to_end):
            wall_id = wallID[hash_index]
            if wall[wall_id]._is_sphere_intersect(position, (radius + 2. * verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        neighbors = sques - particle_id * potential_wall_num
        assert neighbors <= potential_wall_num, f"Keyword:: /wall_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors

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
                        cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        wall_location = cellID * facet_in_cell + ti.atomic_add(wall_count[cellID], 1)                       
                        wallID[wall_location] = wall_id
                        assert wall_count[cellID] <= facet_in_cell, f"Keyword:: /wall_per_cell/ is too small. Cell {cellID} has at least {wall_count[cellID]} walls"

@ti.kernel
def board_search_particle_facet_linked_cell_(particleNum: int, potential_wall_num: int, facet_in_cell: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), wallID: ti.template(), 
                                             particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int)):
    particle_wall.fill(0)
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_idx = ti.floor(position * igrid_size, int)

        sques = particle_id * potential_wall_num
        cellID = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        to_beg, to_end = cellID * facet_in_cell, cellID * facet_in_cell + wall_count[cellID]
        for hash_index in range(to_beg, to_end):
            wall_id = wallID[hash_index]
            if wall[wall_id]._is_sphere_intersect(position, (radius + 2 * verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        neighbors = sques - particle_id * potential_wall_num
        assert neighbors <= potential_wall_num, f"Keyword:: /wall_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors

@ti.kernel
def board_search_coupled_particle_facet_linked_cell_(potential_wall_num: int, facet_in_cell: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), wallID: ti.template(), particle: ti.template(), wall: ti.template(), 
                                                     potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int), particleNum: int):
    particle_wall.fill(0)
    for particle_id in range(particleNum):
        if int(particle[particle_id].active) == 0 or int(particle[particle_id].coupling) == 0: continue
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_idx = ti.floor(position * igrid_size, int)

        sques = particle_id * potential_wall_num
        cellID = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        to_beg, to_end = cellID * facet_in_cell, cellID * facet_in_cell + wall_count[cellID]
        for hash_index in range(to_beg, to_end):
            wall_id = wallID[hash_index]
            if wall[wall_id]._is_sphere_intersect(position, (radius + 2 * verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        neighbors = sques - particle_id * potential_wall_num
        assert neighbors <= potential_wall_num, f"Keyword:: /wall_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors

@ti.kernel
def get_total_static_wall_length(cellSum: int, wall_count: ti.template()) -> int:
    return wall_count[cellSum - 1]

@ti.kernel
def calculate_static_facet_position_(wallNum: int, igrid_size: float, wall: ti.template(), wall_count: ti.template(), static_facet_current: ti.template(), cnum: ti.types.vector(3, int)):
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
                        cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        static_facet_current[wall_id] = ti.atomic_add(wall_count[cellID + 1], 1)

@ti.kernel
def insert_static_facet_to_cell_(igrid_size: float, wallNum: int, wall: ti.template(), wall_count: ti.template(), static_facet_current: ti.template(), wallID: ti.template(), cnum: ti.types.vector(3, int)):
    for nw in range(wallNum):
        if int(wall[nw].active) == 1:
            wall_min_coord = wall[nw].bound_beg
            wall_max_coord = wall[nw].bound_end

            minCoord = ti.max(ti.floor(wall_min_coord * igrid_size - 0.5, int), 0)
            maxCoord = ti.min(ti.ceil(wall_max_coord * igrid_size + 0.5, int), cnum)
            for neigh_i in range(minCoord[0], maxCoord[0]):
                for neigh_j in range(minCoord[1], maxCoord[1]): 
                    for neigh_k in range(minCoord[2], maxCoord[2]): 
                        cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        patch_location = wall_count[cellID] + static_facet_current[nw]
                        wallID[patch_location] = nw

@ti.kernel
def board_search_particle_static_facet_linked_cell_(particleNum: int, potential_wall_num: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), wallID: ti.template(), 
                                                    particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int)):
    particle_wall.fill(0)
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_idx = ti.floor(position * igrid_size, int)

        sques = particle_id * potential_wall_num
        cellID = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        for hash_index in range(wall_count[cellID], wall_count[cellID + 1]):
            wall_id = wallID[hash_index]
            valid = wall[wall_id]._is_sphere_intersect(position, (radius + 2 * verlet_distance))
            if valid: 
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        neighbors = sques - particle_id * potential_wall_num
        assert neighbors <= potential_wall_num, f"Keyword:: /wall_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors

# ======================================== Patch ============================================= #
@ti.kernel
def calculate_patch_position_(wallNum: int, igrid_size: float, wall: ti.template(), wall_count: ti.template(), patch_current: ti.template(), cnum: ti.types.vector(3, int)):
    wall_count.fill(0)
    for wall_id in range(wallNum):  
        if int(wall[wall_id].active) == 1:
            grid_idx = ti.floor(wall[wall_id]._get_center() * igrid_size , int)
            cellID = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
            patch_current[wall_id] = ti.atomic_add(wall_count[cellID + 1], 1)

@ti.kernel
def insert_patch_to_cell_(igrid_size: float, wallNum: int, wall: ti.template(), wall_count: ti.template(), patch_current: ti.template(), wallID: ti.template(), cnum: ti.types.vector(3, int)):
    for nw in range(wallNum):
        if int(wall[nw].active) == 1:
            grid_idx = ti.floor(wall[nw]._get_center() * igrid_size , int)
            cellID = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
            patch_location = wall_count[cellID] + patch_current[nw]
            wallID[patch_location] = nw

@ti.kernel
def board_search_particle_patch_linked_cell_(particleNum: int, potential_wall_num: int, verlet_distance: float, igrid_size: float, wall_count: ti.template(), wallID: ti.template(), 
                                             particle: ti.template(), wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), cnum: ti.types.vector(3, int)):
    particle_wall.fill(0)
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
                    cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(wall_count[cellID], wall_count[cellID + 1]):
                        wall_id = wallID[hash_index]
                        valid = wall[wall_id]._is_sphere_intersect(position, (radius + 2 * verlet_distance))
                        if valid: 
                            potential_list_particle_wall[sques] = wall_id
                            sques += 1
        neighbors = sques - particle_id * potential_wall_num
        assert neighbors <= potential_wall_num, f"Keyword:: /wall_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors

@ti.kernel
def insert_digital_elevation_facet_(igrid_size: float, wallNum: int, cnum: ti.types.vector(2, int), wall: ti.template(), wallID: ti.template()):
    for nw in range(wallNum):
        center = wall[nw]._get_center()
        xcell, ycell, _ = ti.floor(center * igrid_size, int)
        cellID = linearize3D(xcell, ycell, 0, cnum)
        wallID[cellID + 1] += 1

    cellSum = int(cnum[0] * cnum[1])
    ti.loop_config(serialize=True)
    for i in range(1, cellSum):
        wallID[i] = wallID[i] + wallID[i - 1]

# =========================================== Digital elevation model =========================================== #
@ti.kernel
def board_search_particle_digital_elevation_(particleNum: int, potential_wall_num: int, verlet_distance: float, icell_size: float, cnum: ti.types.vector(2, int), particle: ti.template(), wall: ti.template(),
                                             wallID: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template()):
    particle_wall.fill(0)
    for particle_id in range(particleNum):
        particle_pos, particle_rad = particle[particle_id].x, particle[particle_id].rad

        xStart, yStart, _ = ti.floor((particle_pos - particle_rad) * icell_size , int)
        xEnd, yEnd, _ = ti.ceil((particle_pos + particle_rad) * icell_size , int)
        
        sques = particle_id * potential_wall_num
        for neigh_x in range(xStart, xEnd):
            for neigh_y in range(yStart, yEnd):
                cellID = linearize3D(neigh_x, neigh_y, 0, cnum)
                startID, endID = wallID[cellID], wallID[cellID + 1]
                for wall_id in range(startID, endID):
                    valid = wall[wall_id]._is_sphere_intersect(particle_pos, (particle_rad + 2 * verlet_distance))
                    if valid: 
                        potential_list_particle_wall[sques] = wall_id
                        sques += 1
        neighbors = sques - particle_id * potential_wall_num
        assert neighbors <= potential_wall_num, f"Keyword:: /wall_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors

# ================================================================= #
#                   Hierarchical Linked Cell                        #
# ================================================================= #
@ti.kernel
def initialize_radius_range(particleNum: int, hierarchical_level: int, hierarchical_size: ti.types.ndarray(), particle_num_in_level: ti.types.ndarray(), body: ti.template(), grid: ti.template(), particle: ti.template()):
    for np in range(particleNum):
        radius = particle[np].rad
        level = -1
        for grid_level in range(hierarchical_level):
            if radius <= hierarchical_size[grid_level]:
                level = grid_level
                break
        assert level >= 0, f"Particle {np}, radius {radius} falied to inserted into a hash grid"
        body[np + 1]._set_level(level)
        particle_num_in_level[level] += 1

    for np in range(particleNum):
        ti.atomic_min(grid[int(body[np + 1].level)].rad_min, particle[np].rad)
        ti.atomic_max(grid[int(body[np + 1].level)].rad_max, particle[np].rad)

@ti.kernel
def initialize_body_information(particleNum: int, potential_particle_ratio: ti.types.ndarray(), body_coordination_number: ti.types.ndarray(), wall_coordination_number: ti.types.ndarray(), body: ti.template()):
    for np in range(particleNum):
        level = body[np + 1].level
        body[np + 1]._set(potential_particle_ratio[level], body_coordination_number[level], wall_coordination_number[level])

@ti.kernel
def initialize_wall_information(wallNum: int, hierarchical_level: int, hierarchical_size: ti.types.ndarray(), body: ti.template(), wall: ti.template()):
    for nw in range(wallNum):
        radius = wall[nw]._get_bounding_radius()
        level = -1
        for grid_level in range(hierarchical_level):
            if radius <= hierarchical_size[grid_level]:
                level = grid_level
                break
        assert level >= 0, f"Wall {nw} has a bounding sphere with a radius of {radius}, which is falied to inserted into a hash grid"
        body[nw + 1] = level

@ti.kernel
def initialize_grid_information(hierarchical_level: int, gsize: ti.types.ndarray(), cnum: ti.types.ndarray(), csum: ti.types.ndarray(), 
                                factor: ti.types.ndarray(), wall_per_cell: ti.types.ndarray(), grid: ti.template()) -> int:
    ti.loop_config(serialize=True)
    for i in range(hierarchical_level):
        grid[i]._set(gsize[i], factor[i], vec3i(cnum[i, 0], cnum[i, 1], cnum[i, 2]), wall_per_cell[i])
        if i < hierarchical_level - 1:
            grid[i + 1]._set_cell_index(csum[i])
            grid[i + 1]._set_wall_cells(wall_per_cell[i] * csum[i])

    ti.loop_config(serialize=True)
    for i in range(1, hierarchical_level):
        grid[i].cell_index += grid[i - 1].cell_index
        grid[i].wall_cells += grid[i - 1].wall_cells
    return grid[hierarchical_level - 1].wall_cells + wall_per_cell[hierarchical_level - 1] * csum[hierarchical_level - 1]

@ti.kernel
def get_potential_contact_pairs_num(particleNum: int, body: ti.template()) -> ti.types.vector(2, int):
    return vec2i(body[particleNum].potential_particle_num(), body[particleNum].potential_wall_num())

@ti.kernel
def calculate_particles_position_hierarchical_(particleNum: int, particle: ti.template(), particle_count: ti.template(), particle_current: ti.template(), body: ti.template(), grid: ti.template()):
    # TODO: using morton code
    particle_count.fill(0)
    for np in range(particleNum):  
        grid_level = body[np + 1].level
        start_index = grid[grid_level].cell_index
        grid_idx = ti.floor(particle[np].x * grid[grid_level].igrid_size, int)
        cellID = start_index + linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], grid[grid_level].cnum)
        particle_current[np] = ti.atomic_add(particle_count[cellID + 1], 1)
    
@ti.kernel
def insert_particle_to_cell_hierarchical_(particleNum: int, particle: ti.template(), particle_count: ti.template(), particle_current: ti.template(), particleID: ti.template(), body: ti.template(), grid: ti.template()):
    for np in range(particleNum):
        grid_level = body[np + 1].level
        start_index = grid[grid_level].cell_index
        grid_idx = ti.floor(particle[np].x * grid[grid_level].igrid_size , int)
        cellID = start_index + linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], grid[grid_level].cnum)
        grain_location = particle_count[cellID] + particle_current[np]
        particleID[grain_location] = np

@ti.kernel
def board_search_particle_particle_linked_cell_hierarchical_(particleNum: int, verlet_distance: float, particle_count: ti.template(), particleID: ti.template(), 
                                                             particle: ti.template(), potential_list_particle_particle: ti.template(), particle_particle: ti.template(), body: ti.template(), grid: ti.template()):
    particle_particle.fill(0)
    for master in range(particleNum):
        position = particle[master].x
        radius = particle[master].rad
        index = particle[master]._get_multisphere_index1()
        grid_level = body[master + 1].level
        cell_index = grid[grid_level].cell_index
        cnum = grid[grid_level].cnum
        potential_particle_num = body[master].potential_particle_num()
        grid_idx = ti.floor(position * grid[grid_level].igrid_size, int)
        
        x_begin = ti.max(grid_idx[0] - 1, 0)
        x_end = ti.min(grid_idx[0] + 2, cnum[0])
        y_begin = ti.max(grid_idx[1] - 1, 0)
        y_end = ti.min(grid_idx[1] + 2, cnum[1])
        z_begin = ti.max(grid_idx[2] - 1, 0)
        z_end = ti.min(grid_idx[2] + 2, cnum[2])

        sques = potential_particle_num
        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = cell_index + linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                        slave = particleID[hash_index]
                        if master < slave and not index == particle[slave]._get_multisphere_index2():
                            pos2 = particle[slave].x 
                            rad2 = particle[slave].rad
                            valid = SquaredLength(pos2, position) <= (radius + rad2 + 2 * verlet_distance) * (radius + rad2 + 2 * verlet_distance)
                            if valid: 
                                potential_list_particle_particle[sques] = slave
                                sques += 1

        for i in range(grid_level):
            cell_index = grid[i].cell_index
            cnum = grid[i].cnum
            grid_size = grid[i].grid_size
            igrid_size = grid[i].igrid_size
            grid_start = ti.floor((position - radius - 0.5 * grid_size) * igrid_size, int)
            grid_end = ti.ceil((position + radius + 0.5 * grid_size) * igrid_size, int)

            x_begin = ti.max(grid_start[0], 0)
            x_end = ti.min(grid_end[0], cnum[0])
            y_begin = ti.max(grid_start[1], 0)
            y_end = ti.min(grid_end[1], cnum[1])
            z_begin = ti.max(grid_start[2], 0)
            z_end = ti.min(grid_end[2], cnum[2])

            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cellID = cell_index + linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                            slave = particleID[hash_index]
                            if not index == particle[slave]._get_multisphere_index2():
                                pos2 = particle[slave].x 
                                rad2 = particle[slave].rad
                                valid = SquaredLength(pos2, position) <= (radius + rad2 + 2 * verlet_distance) * (radius + rad2 + 2 * verlet_distance)
                                if valid: 
                                    potential_list_particle_particle[sques] = slave
                                    sques += 1

        neighbors = sques - potential_particle_num
        assert neighbors <= body[master + 1].potential_particle_num() - body[master].potential_particle_num(), f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors

@ti.kernel
def board_search_particle_particle_linked_cell_hierarchical2_(particleNum: int, levels: int, verlet_distance: float, particle_count: ti.template(), particleID: ti.template(), 
                                                              particle: ti.template(), potential_list_particle_particle: ti.template(), particle_particle: ti.template(), body: ti.template(), grid: ti.template()):
    particle_particle.fill(0)
    for master in range(particleNum):
        position = particle[master].x
        radius = particle[master].rad
        index = particle[master]._get_multisphere_index1()
        grid_level = body[master + 1].level
        cell_index = grid[grid_level].cell_index
        cnum = grid[grid_level].cnum
        potential_particle_num = body[master].potential_particle_num()
        grid_idx = ti.floor(position * grid[grid_level].igrid_size, int)
        
        x_begin = ti.max(grid_idx[0] - 1, 0)
        x_end = ti.min(grid_idx[0] + 2, cnum[0])
        y_begin = ti.max(grid_idx[1] - 1, 0)
        y_end = ti.min(grid_idx[1] + 2, cnum[1])
        z_begin = ti.max(grid_idx[2] - 1, 0)
        z_end = ti.min(grid_idx[2] + 2, cnum[2])

        sques = potential_particle_num
        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = cell_index + linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                        slave = particleID[hash_index]
                        if master < slave and not index == particle[slave]._get_multisphere_index2():
                            pos2 = particle[slave].x 
                            rad2 = particle[slave].rad
                            valid = SquaredLength(pos2, position) <= (radius + rad2 + 2 * verlet_distance) * (radius + rad2 + 2 * verlet_distance)
                            if valid: 
                                potential_list_particle_particle[sques] = slave
                                sques += 1

        for i in range(grid_level + 1, levels):
            cell_index = grid[i].cell_index
            cnum = grid[i].cnum
            grid_size = grid[i].grid_size
            igrid_size = grid[i].igrid_size
            grid_start = ti.floor((position - radius - 0.5 * grid_size) * igrid_size, int)
            grid_end = ti.ceil((position + radius + 0.5 * grid_size) * igrid_size, int)

            x_begin = ti.max(grid_start[0], 0)
            x_end = ti.min(grid_end[0], cnum[0])
            y_begin = ti.max(grid_start[1], 0)
            y_end = ti.min(grid_end[1], cnum[1])
            z_begin = ti.max(grid_start[2], 0)
            z_end = ti.min(grid_end[2], cnum[2])

            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cellID = cell_index + linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                            slave = particleID[hash_index]
                            if not index == particle[slave]._get_multisphere_index2():
                                pos2 = particle[slave].x 
                                rad2 = particle[slave].rad
                                valid = SquaredLength(pos2, position) <= (radius + rad2 + 2 * verlet_distance) * (radius + rad2 + 2 * verlet_distance)
                                if valid: 
                                    potential_list_particle_particle[sques] = slave
                                    sques += 1

        neighbors = sques - potential_particle_num
        assert neighbors <= body[master + 1].potential_particle_num() - body[master].potential_particle_num(), f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors

@ti.kernel
def board_search_particle_particle_linked_cell_hierarchical_interlevel_(startID: int, endID: int, verlet_distance: float, particle_count: ti.template(), particleID: ti.template(), 
                                                                        particle: ti.template(), potential_list_particle_particle: ti.template(), particle_particle: ti.template(), body: ti.template(), grid: ti.template()):
    for master in range(startID, endID):
        position = particle[master].x
        radius = particle[master].rad
        index = particle[master]._get_multisphere_index1()
        grid_level = body[master + 1].level
        cell_index = grid[grid_level].cell_index
        cnum = grid[grid_level].cnum
        potential_particle_num = body[master].potential_particle_num() + particle_particle[master + 1]
        grid_idx = ti.floor(position * grid[grid_level].igrid_size, int)
        
        x_begin = ti.max(grid_idx[0] - 1, 0)
        x_end = ti.min(grid_idx[0] + 2, cnum[0])
        y_begin = ti.max(grid_idx[1] - 1, 0)
        y_end = ti.min(grid_idx[1] + 2, cnum[1])
        z_begin = ti.max(grid_idx[2] - 1, 0)
        z_end = ti.min(grid_idx[2] + 2, cnum[2])

        sques = potential_particle_num
        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = cell_index + linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                        slave = particleID[hash_index]
                        if master < slave and not index == particle[slave]._get_multisphere_index2():
                            pos2 = particle[slave].x 
                            rad2 = particle[slave].rad
                            valid = SquaredLength(pos2, position) <= (radius + rad2 + 2 * verlet_distance) * (radius + rad2 + 2 * verlet_distance)
                            if valid: 
                                potential_list_particle_particle[sques] = slave
                                sques += 1
        body[master].set_potential_particle_num(neighbors)
        neighbors = sques - potential_particle_num
        assert neighbors <= body[master + 1].potential_particle_num() - body[master].potential_particle_num(), f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors


@ti.kernel
def board_search_particle_particle_linked_cell_hierarchical_crosslevel_(startID: int, endID: int, verlet_distance: float, particle_count: ti.template(), particleID: ti.template(), 
                                                                        particle: ti.template(), potential_list_particle_particle: ti.template(), particle_particle: ti.template(), body: ti.template(), grid: ti.template()):
    for master in range(startID, endID):
        position = particle[master].x
        radius = particle[master].rad
        index = particle[master]._get_multisphere_index1()
        grid_level = body[master + 1].level
        cell_index = grid[grid_level].cell_index
        cnum = grid[grid_level].cnum
        potential_particle_num = body[master].potential_particle_num() + particle_particle[master + 1]

        sques = potential_particle_num
        i = grid_level - 1
        if i < 0: continue
        cell_index = grid[i].cell_index
        cnum = grid[i].cnum
        grid_size = grid[i].grid_size
        igrid_size = grid[i].igrid_size
        grid_start = ti.floor((position - radius - 0.5 * grid_size) * igrid_size, int)
        grid_end = ti.ceil((position + radius + 0.5 * grid_size) * igrid_size, int)

        x_begin = ti.max(grid_start[0], 0)
        x_end = ti.min(grid_end[0], cnum[0])
        y_begin = ti.max(grid_start[1], 0)
        y_end = ti.min(grid_end[1], cnum[1])
        z_begin = ti.max(grid_start[2], 0)
        z_end = ti.min(grid_end[2], cnum[2])

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = cell_index + linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                        slave = particleID[hash_index]
                        if not index == particle[slave]._get_multisphere_index2():
                            pos2 = particle[slave].x 
                            rad2 = particle[slave].rad
                            valid = SquaredLength(pos2, position) <= (radius + rad2 + 2 * verlet_distance) * (radius + rad2 + 2 * verlet_distance)
                            if valid: 
                                potential_list_particle_particle[sques] = slave
                                sques += 1

        neighbors = sques - potential_particle_num
        assert neighbors <= body[master + 1].potential_particle_num() - body[master].potential_particle_num(), f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors


@ti.kernel
def board_search_coupled_particle_linked_cell_hierarchical_(potential_particle_num: int, verlet_distance1: float, verlet_distance2: float, max_radius: float, particle_count: ti.template(),
                                                            particleID: ti.template(), particle1: ti.template(), particle2: ti.template(), potential_list_particle_particle: ti.template(), 
                                                            particle_particle: ti.template(), particleNum: int, levels: int, grid: ti.template()):
    particle_particle.fill(0)
    for master in range(particleNum):
        if int(particle1[master].active) == 0 or int(particle1[master].coupling) == 0: continue
        position = particle1[master].x
        radius = particle1[master].rad

        sques = master * potential_particle_num
        for grid_level in range(levels):
            cnum = grid[grid_level].cnum
            igrid_size = grid[grid_level].igrid_size

            grid_start = ti.floor((position - radius - verlet_distance1 - verlet_distance2 - max_radius) * igrid_size, int)
            grid_end = ti.ceil((position + radius + verlet_distance1 + verlet_distance2 + max_radius) * igrid_size, int)
            x_begin = ti.max(grid_start[0], 0)
            x_end = ti.min(grid_end[0], cnum[0])
            y_begin = ti.max(grid_start[1], 0)
            y_end = ti.min(grid_end[1], cnum[1])
            z_begin = ti.max(grid_start[2], 0)
            z_end = ti.min(grid_end[2], cnum[2])

            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                            slave = particleID[hash_index]
                            pos2 = particle2[slave].x 
                            rad2 = particle2[slave].rad
                            search_radius = (radius + rad2 + verlet_distance1 + verlet_distance2)
                            valid = SquaredLength(pos2, position) <= search_radius * search_radius
                            if valid: 
                                potential_list_particle_particle[sques] = slave
                                sques += 1
        neighbors = sques - master * potential_particle_num
        assert neighbors <= potential_particle_num, f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors

@ti.kernel
def board_search_coupled_lsparticle_linked_cell_hierarchical_(potential_particle_num: int, verlet_distance1: float, verlet_distance2: float, max_radius: float, particle_count: ti.template(), 
                                                              particleID: ti.template(), particle1: ti.template(), rigid: ti.template(), box: ti.template(), 
                                                              potential_list_particle_particle: ti.template(), particle_particle: ti.template(), particleNum: int, levels: int, grid: ti.template()):
    particle_particle.fill(0)
    for master in range(particleNum):
        if int(particle1[master].active) == 0 or int(particle1[master].coupling) == 0: continue
        position = particle1[master].x
        radius = particle1[master].rad

        sques = master * potential_particle_num
        for grid_level in range(levels):
            cell_index = grid[grid_level].cell_index
            cnum = grid[grid_level].cnum
            igrid_size = grid[grid_level].igrid_size

            grid_start = ti.floor((position - radius - verlet_distance1 - verlet_distance2 - max_radius) * igrid_size, int)
            grid_end = ti.ceil((position + radius + verlet_distance1 + verlet_distance2 + max_radius) * igrid_size, int)
            x_begin = ti.max(grid_start[0], 0)
            x_end = ti.min(grid_end[0], cnum[0])
            y_begin = ti.max(grid_start[1], 0)
            y_end = ti.min(grid_end[1], cnum[1])
            z_begin = ti.max(grid_start[2], 0)
            z_end = ti.min(grid_end[2], cnum[2])
            
            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cellID = cell_index + linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                            slave = particleID[hash_index]
                            mass_center = rigid[slave]._get_position()
                            rotate_matrix = SetToRotate(rigid[slave].q)
                            node = rotate_matrix.transpose() @ (position - mass_center)
                            verlet_distance = verlet_distance1 + verlet_distance2
                            if box[slave]._in_box(node) == 1: continue
                            if box[slave].distance(node, grid) < verlet_distance + radius: 
                                potential_list_particle_particle[sques] = slave
                                sques += 1
        neighbors = sques - master * potential_particle_num
        assert neighbors <= potential_particle_num, f"Keyword:: DEMPM /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors


# ============================================ Plane ================================================= #        
@ti.kernel
def insert_plane_to_cell_hierarchical_(wallNum: int, total_cell: int, levels: int, wall_count: ti.template(), wallID: ti.template(), wall: ti.template(), grid: ti.template()):
    wallID.fill(0)
    wall_count.fill(0)
    for nc in range(total_cell):
        grid_level = 0
        for i in range(levels):
            if nc < grid[i].cell_index + grid[i]._cell_sum():
                grid_level = i
                break
        
        cnum = grid[grid_level].cnum
        grid_size = grid[grid_level].grid_size
        factor = grid[grid_level].factor
        cell_index = grid[grid_level].cell_index
        plane_in_cell = grid[grid_level].wall_cells + (nc - cell_index) * grid[grid_level].wall_per_cell
        cell_center = get_cell_center(nc - cell_index, cnum, grid_size)
        for nw in range(wallNum):
            if int(wall[nw].active) == 1:
                distance = wall[nw]._point_to_wall_distance(cell_center)
                if distance <= factor * grid_size:
                    wall_location = plane_in_cell + ti.atomic_add(wall_count[nc], 1)
                    wallID[wall_location] = nw
                    assert wall_count[nc] <= grid[grid_level].wall_per_cell, f"Keyword:: /wall_per_cell/ = {grid[grid_level].wall_per_cell} at grid level {grid_level} is too small, at least {wall_count[nc]}"

@ti.kernel
def board_search_particle_plane_linked_cell_hierarchical_(particleNum: int, verlet_distance: float, wall_count: ti.template(), wallID: ti.template(), particle: ti.template(), 
                                                          wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), body: ti.template(), grid: ti.template()):
    particle_wall.fill(0)
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_level = body[particle_id + 1].level
        cell_index = grid[grid_level].cell_index
        cnum = grid[grid_level].cnum
        potential_wall_num = body[particle_id].potential_wall_num()
        grid_idx = ti.floor(position * grid[grid_level].igrid_size, int)
        
        sques = potential_wall_num
        local_cell = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        plane_in_cell = grid[grid_level].wall_cells + local_cell * grid[grid_level].wall_per_cell
        cellID = cell_index + local_cell
        to_beg, to_end = plane_in_cell, plane_in_cell + wall_count[cellID]
        for hash_index in range(to_beg, to_end):
            wall_id = wallID[hash_index]
            if wall[wall_id]._is_sphere_intersect(position, (radius + 2. * verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        neighbors = sques - potential_wall_num
        assert neighbors <= body[particle_id + 1].potential_wall_num() - body[particle_id].potential_wall_num(), f"Keyword:: /wall_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors

# ======================================== Facet ============================================= #
@ti.kernel
def insert_facet_to_cell_hierarchical_(wallNum: int, levels: int, wall_count: ti.template(), wallID: ti.template(), wall: ti.template(), grid: ti.template()):
    wall_count.fill(0)
    for wall_id in range(wallNum):
        if int(wall[wall_id].active) == 1:
            wall[wall_id]._bounding_box()
            wall_min_coord = wall[wall_id].bound_beg
            wall_max_coord = wall[wall_id].bound_end
            
            for grid_level in range(levels):
                igrid_size = grid[grid_level].igrid_size
                cnum = grid[grid_level].cnum
                cell_index = grid[grid_level].cell_index
                wall_cells = grid[grid_level].wall_cells 
                facet_in_cell = grid[grid_level].wall_per_cell
                minCoord = ti.max(ti.floor(wall_min_coord * igrid_size - 0.5, int), 0)
                maxCoord = ti.min(ti.ceil(wall_max_coord * igrid_size + 0.5, int) + 1, cnum)
                for neigh_i in range(minCoord[0], maxCoord[0]):
                    for neigh_j in range(minCoord[1], maxCoord[1]): 
                        for neigh_k in range(minCoord[2], maxCoord[2]): 
                            local_cell = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                            cellID = cell_index + local_cell
                            wall_location = wall_cells + local_cell * facet_in_cell + ti.atomic_add(wall_count[cellID], 1)                       
                            wallID[wall_location] = wall_id
                            assert wall_count[cellID] <= facet_in_cell, f"Keyword:: /wall_per_cell/ = {grid[grid_level].wall_per_cell} at grid level {grid_level} is too small, at least {wall_count[cellID]}."

@ti.kernel
def board_search_particle_facet_linked_cell_hierarchical_(particleNum: int, verlet_distance: float, wall_count: ti.template(), wallID: ti.template(), particle: ti.template(), 
                                                          wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), body: ti.template(), grid: ti.template()):
    particle_wall.fill(0)
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad 
        grid_level = body[particle_id + 1].level
        cell_index = grid[grid_level].cell_index
        cnum = grid[grid_level].cnum
        potential_wall_num = body[particle_id].potential_wall_num()
        grid_idx = ti.floor(position * grid[grid_level].igrid_size, int)

        sques = potential_wall_num
        local_cell = linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], cnum)
        facet_in_cell = grid[grid_level].wall_cells + local_cell * grid[grid_level].wall_per_cell
        cellID = cell_index + local_cell
        to_beg, to_end = facet_in_cell, facet_in_cell + wall_count[cellID]
        for hash_index in range(to_beg, to_end):
            wall_id = wallID[hash_index]
            if wall[wall_id]._is_sphere_intersect(position, (radius + 2 * verlet_distance)) == 1:
                potential_list_particle_wall[sques] = wall_id
                sques += 1
        neighbors = sques - potential_wall_num
        assert neighbors <= body[particle_id + 1].potential_wall_num() - body[particle_id].potential_wall_num(), f"Keyword:: /wall_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors

# ======================================== Patch ============================================= #
@ti.kernel
def calculate_patch_position_hierarchical_(wallNum: int, wall: ti.template(), wall_count: ti.template(), patch_current: ti.template(), grid: ti.template(), wallbody: ti.template()):
    wall_count.fill(0)
    for wall_id in range(wallNum):  
        if int(wall[wall_id].active) == 1:
            grid_level = wallbody[wall_id]
            start_index = grid[grid_level].cell_index
            grid_idx = ti.floor(wall[wall_id]._get_center() * grid[grid_level].igrid_size , int)
            cellID = start_index + linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], grid[grid_level].cnum)
            patch_current[wall_id] = ti.atomic_add(wall_count[cellID + 1], 1)

@ti.kernel
def insert_patch_to_cell_hierarchical_(wallNum: int, wall: ti.template(), wall_count: ti.template(), patch_current: ti.template(), wallID: ti.template(),  grid: ti.template(), wallbody: ti.template()):
    for nw in range(wallNum):
        if int(wall[nw].active) == 1:
            grid_level = wallbody[nw]
            start_index = grid[grid_level].cell_index
            wall_index = grid[grid_level].index
            grid_idx = ti.floor(wall[nw]._get_center() * grid[grid_level].igrid_size , int)
            cellID = start_index + linearize3D(grid_idx[0], grid_idx[1], grid_idx[2], grid[grid_level].cnum)
            patch_location = wall_index + wall_count[cellID] + patch_current[nw]
            wallID[patch_location] = nw

@ti.kernel
def board_search_particle_patch_linked_cell_hierarchical_(particleNum: int, levels: int, verlet_distance: float, wall_count: ti.template(), wallID: ti.template(), particle: ti.template(), 
                                                          wall: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template(), body: ti.template(), grid: ti.template()):
    particle_wall.fill(0)
    for particle_id in range(particleNum):
        position = particle[particle_id].x
        radius = particle[particle_id].rad
        grid_level = body[particle_id + 1].level
        cell_index = grid[grid_level].cell_index
        cnum = grid[grid_level].cnum
        wall_index = grid[grid_level].index
        potential_wall_num = body[particle_id].potential_wall_num()

        sques = potential_wall_num
        for i in range(levels):
            grid_start = ti.floor((position - radius - 2. * verlet_distance) * grid[i].igrid_size, int)
            grid_end = ti.ceil((position + radius + 2. * verlet_distance) * grid[i].igrid_size, int)
            x_begin = ti.max(grid_start[0], 0)
            x_end = ti.min(grid_end[0], cnum[0])
            y_begin = ti.max(grid_start[1], 0)
            y_end = ti.min(grid_end[1], cnum[1])
            z_begin = ti.max(grid_start[2], 0)
            z_end = ti.min(grid_end[2], cnum[2])

            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cellID = cell_index + linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(wall_count[cellID], wall_count[cellID + 1]):
                            wall_id = wallID[wall_index + hash_index]
                            valid = wall[wall_id]._is_sphere_intersect(position, (radius + 2 * verlet_distance))
                            if valid: 
                                potential_list_particle_wall[sques] = wall_id
                                sques += 1
        neighbors = sques - potential_wall_num
        assert neighbors <= body[particle_id + 1].potential_wall_num() - body[particle_id].potential_wall_num(), f"Keyword:: /wall_coordination_number/ is too small, Particle {particle_id} has {neighbors} potential contact number"
        particle_wall[particle_id + 1] = neighbors


# ================================================================= #
#                                                                   #
#                              BVH                                  #
#                                                                   #
# ================================================================= #
@ti.kernel
def board_search_particle_particle_bvh_(batch_id: int, particleNum: int, potential_particle_num: int, prefix_batch_size: ti.template(), nodes: ti.template(), morton_codes: ti.template(), aabbs: ti.template(), particle: ti.template(), potential_list_particle_particle: ti.template(), particle_particle: ti.template()):
    prefix_batch_num = prefix_batch_size[batch_id]
    batch_num = prefix_batch_size[batch_id + 1] - prefix_batch_num
    prefix_i_node = 2 * prefix_batch_num - batch_id
    particle_particle.fill(0)
    for master in range(particleNum):
        if int(particle[master].active) == 0: continue
        query_stack = ti.Vector.zero(ti.i32, 64)
        stack_depth = 1
        index = particle[master]._get_multisphere_index1()
        
        sques = master * potential_particle_num
        while stack_depth > 0:
            stack_depth -= 1
            node_idx = query_stack[stack_depth]
            node = nodes[prefix_i_node + node_idx]
            # Check if the AABB intersects with the node's bounding box
            if aabbs[prefix_batch_num + master].intersects(node.bound):
                # If it's a leaf node, add the AABB index to the query results
                if node.left == -1 and node.right == -1:
                    code = morton_codes[prefix_batch_num + node_idx - (batch_num - 1)]
                    slave = ti.i32(code & ti.u64(0xFFFFFFFF))
                    if master < slave and not index == particle[slave]._get_multisphere_index2():
                        potential_list_particle_particle[sques] = slave
                        sques += 1
                else:
                    # Push children onto the stack
                    if node.right != -1:
                        query_stack[stack_depth] = node.right
                        stack_depth += 1
                    if node.left != -1:
                        query_stack[stack_depth] = node.left
                        stack_depth += 1
        neighbors = sques - master * potential_particle_num
        assert neighbors <= potential_particle_num, f"Keyword:: /body_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_particle[master + 1] = neighbors


@ti.kernel
def board_search_particle_wall_bvh_(particle_batch_id: int, wall_batch_id: int, particleNum: int, potential_wall_num: int, prefix_batch_size: ti.template(), nodes: ti.template(), morton_codes: ti.template(), aabbs: ti.template(), particle: ti.template(), potential_list_particle_wall: ti.template(), particle_wall: ti.template()):
    prefix_particle_batch_num = prefix_batch_size[particle_batch_id]
    prefix_wall_batch_num = prefix_batch_size[wall_batch_id]
    wall_batch_num = prefix_batch_size[wall_batch_id + 1] - prefix_wall_batch_num
    prefix_i_node = 2 * prefix_wall_batch_num - wall_batch_id
    particle_wall.fill(0)
    for master in range(particleNum):
        if int(particle[master].active) == 0: continue
        query_stack = ti.Vector.zero(ti.i32, 64)
        stack_depth = 1
        
        sques = master * potential_wall_num
        while stack_depth > 0:
            stack_depth -= 1
            node_idx = query_stack[stack_depth]
            node = nodes[prefix_i_node + node_idx]
            # Check if the AABB intersects with the node's bounding box
            if aabbs[prefix_particle_batch_num + master].intersects(node.bound):
                # If it's a leaf node, add the AABB index to the query results
                if node.left == -1 and node.right == -1:
                    code = morton_codes[prefix_wall_batch_num + node_idx - (wall_batch_num - 1)]
                    slave = ti.i32(code & ti.u64(0xFFFFFFFF))
                    potential_list_particle_wall[sques] = slave
                    sques += 1
                else:
                    # Push children onto the stack
                    if node.right != -1:
                        query_stack[stack_depth] = node.right
                        stack_depth += 1
                    if node.left != -1:
                        query_stack[stack_depth] = node.left
                        stack_depth += 1
        neighbors = sques - master * potential_wall_num
        assert neighbors <= potential_wall_num, f"Keyword:: /wall_coordination_number/ is too small, Particle {master} has {neighbors} potential contact number"
        particle_wall[master + 1] = neighbors