import taichi as ti

from src.utils.constants import ZEROVEC3f, Threshold
from src.utils.Quaternion import SetToRotate
from src.utils.ScalarFunction import PairingMapping, EffectiveValue, linearize3D
from src.utils.TypeDefination import vec3f
from src.utils import GlobalVariable


@ti.kernel
def kernel_get_min_ratio(materialID: int, rigidNum: int, material: ti.template(), rigid: ti.template()) -> float:
    ratio = 1.
    for i in range(rigidNum):
        if rigid[i].materialID == materialID:
            temp = material[materialID].ncut / rigid[i].equi_r
            ti.atomic_min(ratio, temp)
            assert 0. < temp <= 1.
    return ratio

@ti.kernel
def kernel_find_max_stiffness(surfaceNum: int, rigid: ti.template(), surface: ti.template(), vertice: ti.template(), surfaceProps: ti.template()) -> float:
    max_stiff = 0.
    for i in range(surfaceNum):
        end1 = surface[i]
        local_node = rigid[end1].global_node_to_local(i)
        materialID = rigid[end1].materialID
        parameter = vertice[local_node].parameter
        ti.atomic_max(max_stiff, parameter * max(surfaceProps[materialID].kn, surfaceProps[materialID].ks))
    return max_stiff

@ti.func
def find_history(end1, end2, hist_cplist, hist_object_object):
    tangOverlapOld = ZEROVEC3f
    for offset in range(hist_object_object[end1], hist_object_object[end1+1]):
        if end2 == hist_cplist[offset].DstID:
            tangOverlapOld = hist_cplist[offset].oldTangOverlap
            break
    return tangOverlapOld

@ti.func
def find_addition_history(end1, end2, hist_cplist, hist_object_object):
    tangOverlapOld, rollAngleOld, twistAngleOld = ZEROVEC3f, ZEROVEC3f, ZEROVEC3f
    for offset in range(hist_object_object[end1], hist_object_object[end1+1]):
        if end2 == hist_cplist[offset].DstID:
            tangOverlapOld = hist_cplist[offset].oldTangOverlap
            rollAngleOld = hist_cplist[offset].oldRollAngle
            twistAngleOld = hist_cplist[offset].oldTwistAngle
            break
    return tangOverlapOld, rollAngleOld, twistAngleOld

# ========================================================= #
#                   Bit Table Resolve                       #
# ========================================================= # 
@ti.kernel
def bit_table_reset(contact_active: ti.template()):
    ti.loop_config(bit_vectorize=True)
    for i in contact_active:
        contact_active[i] = 0

@ti.func
def set_bit_table(end1, end2, max_row_num, contact_active):
    hash_number = PairingMapping(end1, end2, max_row_num)
    contact_active[hash_number] = 1

@ti.func
def clear_bit_table(end1, end2, max_row_num, contact_active):
    hash_number = PairingMapping(end1, end2, max_row_num)
    contact_active[hash_number] = 0

@ti.func
def get_bit_table(end1, end2, max_row_num, contact_active):
    hash_number = PairingMapping(end1, end2, max_row_num)
    return int(contact_active[hash_number])
            
@ti.kernel
def update_contact_bit_table_(potential_particle_num: int, max_particle_num: int, particleNum: int, particle_particle: ti.template(), hist_particle_particle: ti.template(), 
                              potential_list_particle_particle: ti.template(), cplist: ti.template(), hist_cplist: ti.template(), contact_active: ti.template(), inherit_overlap: ti.template()): 
    for i in range(particleNum * potential_particle_num):
        end1 = i // potential_particle_num
        offset = i % potential_particle_num
        particle_num = particle_particle[end1 + 1] - particle_particle[end1]
        if offset < particle_num:
            nc = particle_particle[end1] + offset
            end2 = potential_list_particle_particle[i]
            
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2

            exist = get_bit_table(end1, end2, max_particle_num, contact_active)
            if exist:
                for j in range(hist_particle_particle[end1], hist_particle_particle[end1 + 1]):
                    if end1 == hist_cplist[j].endID1:
                        inherit_overlap(nc, j, cplist, hist_cplist)
                        break
            else:
                clear_bit_table(end1, end2, max_particle_num, contact_active)

@ti.kernel
def update_contact_wall_bit_table_(potential_wall_num: int, max_particle_num: int, particleNum: int, particle_wall: ti.template(), hist_particle_wall: ti.template(), potential_list_particle_wall: ti.template(), 
                                   cplist: ti.template(), hist_cplist: ti.template(), contact_active: ti.template(), inherit_overlap: ti.template()): 
    for i in range(particleNum * potential_wall_num):
        end1 = i // potential_wall_num
        offset = i % potential_wall_num
        particle_num = particle_wall[end1 + 1] - particle_wall[end1]
        if offset < particle_num:
            nc = particle_wall[end1] + offset
            end2 = potential_list_particle_wall[i]     

            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2

            exist = get_bit_table(end1, end2, max_particle_num, contact_active)
            if exist:
                for j in range(hist_particle_wall[end1], hist_particle_wall[end1 + 1]):
                    if end1 == hist_cplist[j].endID1:
                        inherit_overlap(nc, j, cplist, hist_cplist)
                        break
            else:
                clear_bit_table(end1, end2, max_particle_num, contact_active)


# ========================================================= #
#              Particle Contact Matrix Resolve              #
# ========================================================= # 
@ti.kernel
def update_contact_table_(potential_object_num: int, particleNum: int, object_object: ti.template(), potential_list_object_object: ti.template(), cplist: ti.template()):
    assert object_object[particleNum] <= cplist.shape[0], f"Keyword:: /compaction_ratio/ is too small, at least {object_object[particleNum]} divided by {potential_list_object_object.shape[0]}"
    for i in range(particleNum * potential_object_num):
        end1 = i // potential_object_num
        offset = i % potential_object_num
        object_num = object_object[end1 + 1] - object_object[end1]
        if offset < object_num:
            nc = object_object[end1] + offset
            end2 = potential_list_object_object[i]
            
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2


@ti.kernel
def update_LScontact_table_(potential_object_num: int, surfaceNum: int, object_object: ti.template(), potential_list_object_object: ti.template(), cplist: ti.template()):
    assert object_object[surfaceNum] <= cplist.shape[0], f"Keyword:: /compaction_ratio/ is too small, at least {object_object[surfaceNum]} divided by {potential_list_object_object.shape[0]}"
    for i in range(surfaceNum * potential_object_num):
        nodeID = i // potential_object_num
        offset = i % potential_object_num
        particle_num = object_object[nodeID + 1] - object_object[nodeID]
        if offset < particle_num:
            nc = object_object[nodeID] + offset
            
            cplist[nc].endID1 = nodeID
            cplist[nc].endID2 = potential_list_object_object[i]


@ti.kernel
def update_contact_table_hierarchical_(particleNum: int, particle_particle: ti.template(), potential_list_particle_particle: ti.template(), cplist: ti.template(), body: ti.template()):
    assert particle_particle[particleNum] <= cplist.shape[0], f"Keyword:: /compaction_ratio[0]/ is too small, at least {particle_particle[particleNum]} divided by {potential_list_particle_particle.shape[0]}"
    for end1 in range(particleNum):
        particle_num = particle_particle[end1 + 1] - particle_particle[end1]
        potential_particle_num = body[end1].potential_particle_num()
        for offset in range(particle_num):
            i = potential_particle_num + offset
            end2 = potential_list_particle_particle[i]

            nc = particle_particle[end1] + offset
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2


@ti.kernel
def update_wall_contact_table_hierarchical_(particleNum: int, particle_wall: ti.template(), potential_list_particle_wall: ti.template(), cplist: ti.template(), body: ti.template()):
    assert particle_wall[particleNum] <= cplist.shape[0], f"Keyword:: /compaction_ratio[1]/ is too small, at least {particle_wall[particleNum]} divided by {potential_list_particle_wall.shape[0]}"
    for end1 in range(particleNum):
        particle_num = particle_wall[end1 + 1] - particle_wall[end1]
        potential_wall_num = body[end1].potential_wall_num()

        for offset in range(particle_num):
            i = potential_wall_num + offset
            end2 = potential_list_particle_wall[i]
            
            nc = particle_wall[end1] + offset
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2
      

@ti.kernel
def kernel_inherit_contact_history(particleNum: int, cplist: ti.template(), hist_cplist: ti.template(), object_object: ti.template(), hist_object_object: ti.template()):
    total_contact_num = object_object[particleNum]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        cplist[nc].oldTangOverlap = find_history(end1, end2, hist_cplist, hist_object_object)


@ti.kernel
def kernel_inherit_rolling_history(particleNum: int, cplist: ti.template(), hist_cplist: ti.template(), object_object: ti.template(), hist_object_object: ti.template()):
    total_contact_num = object_object[particleNum]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        tangOverlapOld, rollAngleOld, twistAngleOld = find_addition_history(end1, end2, hist_cplist, hist_object_object)
        cplist[nc].oldTangOverlap = tangOverlapOld
        cplist[nc].oldRollAngle = rollAngleOld
        cplist[nc].oldTwistAngle = twistAngleOld


@ti.kernel
def copy_contact_table(object_object: ti.template(), particleNum: int, cplist: ti.template(), hist_cplist: ti.template()):
    total_contact_num = object_object[particleNum]
    for nc in range(total_contact_num):
        hist_cplist[nc].DstID = cplist[nc].endID2
        hist_cplist[nc].oldTangOverlap = cplist[nc].oldTangOverlap


@ti.kernel
def copy_addition_contact_table(object_object: ti.template(), particleNum: int, cplist: ti.template(), hist_cplist: ti.template()):
    total_contact_num = object_object[particleNum]
    for nc in range(total_contact_num):
        hist_cplist[nc].DstID = cplist[nc].endID2
        hist_cplist[nc].oldTangOverlap = cplist[nc].oldTangOverlap
        hist_cplist[nc].oldRollAngle = cplist[nc].oldRollAngle
        hist_cplist[nc].oldTwistAngle = cplist[nc].oldTwistAngle


@ti.kernel
def kernel_update_active_collisions_(particleNum: int, particle: ti.template(), cplist: ti.template(), object_object: ti.template()):
    total_contact_num = object_object[particleNum]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)
        cplist[nc].avtice = ti.u8(gapn < 0.)


@ti.kernel
def kernel_particle_particle_force_assemble_(particleNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), particle1: ti.template(), particle2: ti.template(), 
                                             cplist: ti.template(), particle_particle: ti.template(), contact_model: ti.template()):
    total_contact_num = particle_particle[particleNum]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        matID1, matID2 = particle1[end1].materialID, particle2[end2].materialID
        pos1, pos2 = particle1[end1]._get_position(), particle2[end2]._get_position()
        rad1, rad2 = particle1[end1]._get_radius(), particle2[end2]._get_radius() 
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)  
        materialID = PairingMapping(matID1, matID2, max_material_num)
        contact_model(materialID, nc, end1, end2, gapn, pos1, pos2, rad1, rad2, particle1, particle2, surfaceProps, cplist, dt)


@ti.kernel
def kernel_LSparticle_LSparticle_force_assemble_(surfaceNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), rigid: ti.template(), grid: ti.template(), 
                                                 vertice: ti.template(), surface: ti.template(), box: ti.template(), cplist: ti.template(), particle_particle: ti.template(), contact_model: ti.template()):
    total_contact_num = particle_particle[surfaceNum]
    for nc in range(total_contact_num):
        global_node, end2 = cplist[nc].endID1, cplist[nc].endID2
        end1 = surface[global_node]
        local_node = rigid[end1].global_node_to_local(global_node)
        mass_center1, mass_center2 = rigid[end1]._get_position(), rigid[end2]._get_position()
        rotate_matrix1, rotate_matrix2 = SetToRotate(rigid[end1].q), SetToRotate(rigid[end2].q)
        matID1, matID2 = rigid[end1].materialID, rigid[end2].materialID
        materialID = PairingMapping(matID1, matID2, max_material_num)

        surface_node = mass_center1 + rotate_matrix1 @ (box[end1].scale * vertice[local_node].x) 
        surface_node_in_box = rotate_matrix2.transpose() @ (surface_node - mass_center2)
        if not box[end2]._in_box(surface_node_in_box): continue
        gapn = box[end2].distance(surface_node_in_box, grid)
        parameter = vertice[local_node].parameter
        contact_model(materialID, nc, parameter, end1, end2, 0., gapn, mass_center1, mass_center2, surface_node, surface_node_in_box, 
                      rotate_matrix2, rigid, grid, rigid, box, surfaceProps, cplist, dt)


@ti.kernel
def kernel_particle_LSparticle_force_assemble_(particleNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), particle: ti.template(), rigid: ti.template(), grid: ti.template(), 
                                               box: ti.template(), cplist: ti.template(), particle_particle: ti.template(), contact_model: ti.template()):
    total_contact_num = particle_particle[particleNum]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        mass_center1, mass_center2 = particle[end1]._get_position(), rigid[end2]._get_position()
        rad1, rotate_matrix2 = particle[end1]._get_radius(), SetToRotate(rigid[end2].q)
        matID1, matID2 = particle[end1].materialID, rigid[end2].materialID
        materialID = PairingMapping(matID1, matID2, max_material_num)
        
        surface_node_in_box = rotate_matrix2.transpose() @ (mass_center1 - mass_center2)
        if not box[end2]._in_box(surface_node_in_box): continue
        gapn = box[end2].distance(surface_node_in_box, grid) - rad1
        contact_model(materialID, nc, 1., end1, end2, rad1, gapn, mass_center1, mass_center2, mass_center1, surface_node_in_box, 
                      rotate_matrix2, particle, grid, rigid, box, surfaceProps, cplist, dt)

@ti.kernel
def kernel_particle_wall_force_assemble_(particleNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), particle: ti.template(), 
                                         wall: ti.template(), cplist: ti.template(), particle_wall: ti.template(), contact_model: ti.template()):
    total_contact_num = particle_wall[particleNum]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1, particle_rad = particle[end1]._get_position(), particle[end1]._get_radius()
        distance = wall[end2]._get_norm_distance(pos1)
        gapn = distance - particle_rad

        matID1, matID2 = particle[end1].materialID, wall[end2].materialID
        materialID = PairingMapping(matID1, matID2, max_material_num)
        contact_model(materialID, nc, end1, end2, gapn, distance, pos1, particle_rad, particle, wall, surfaceProps, cplist, dt)


@ti.kernel
def kernel_LSparticle_wall_force_assemble_(surfaceNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), rigid: ti.template(), vertice: ti.template(), surface: ti.template(),
                                           box: ti.template(), wall: ti.template(), cplist: ti.template(), particle_wall: ti.template(), contact_model: ti.template()):
    # ti.block_local(dt)
    total_contact_num = particle_wall[surfaceNum]
    for nc in range(total_contact_num):
        global_node, end2 = cplist[nc].endID1, cplist[nc].endID2
        end1 = surface[global_node]
        local_node = rigid[end1].global_node_to_local(global_node)
        mass_center1 = rigid[end1]._get_position()
        rotate_matrix1 = SetToRotate(rigid[end1].q)
        matID1, matID2 = rigid[end1].materialID, wall[end2].materialID
        materialID = PairingMapping(matID1, matID2, max_material_num)
        
        surface_node = mass_center1 + rotate_matrix1 @ (box[end1].scale * vertice[local_node].x)
        distance = wall[end2]._get_norm_distance(surface_node)
        parameter = vertice[local_node].parameter
        contact_model(materialID, nc, parameter, end1, end2, distance, mass_center1, surface_node, rigid, wall, surfaceProps, cplist, dt)


@ti.kernel
def kernel_particle_digital_elevation_force_assemble_(particleNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), particle: ti.template(), wall: ti.template(),
                                                      icell_size: float, cnum: ti.types.vector(2, int), wallID: ti.template(), cplist: ti.template(), contact_model: ti.template()):
    for end1 in range(particleNum):
        point, particle_rad = particle[end1].x, particle[end1].rad

        xStart, yStart, _ = ti.floor((point - particle_rad) * icell_size , int)
        xEnd, yEnd, _ = ti.ceil((point + particle_rad) * icell_size , int)

        for neigh_x in range(xStart, xEnd):
            for neigh_y in range(yStart, yEnd):
                cellID = linearize3D(neigh_x, neigh_y, 0, cnum)
                startID, endID = wallID[cellID], wallID[cellID + 1]

                for end2 in range(startID, endID):
                    distance = wall[end2]._get_norm_distance(point)
                    if distance > 0.:
                        gapn = distance - particle_rad
                        matID1, matID2 = particle[end1].materialID, wall[end2].materialID
                        materialID = PairingMapping(matID1, matID2, max_material_num)
                        contact_model(materialID, end1, end1, end2, gapn, distance, point, particle_rad, particle, wall, surfaceProps, cplist, dt)


@ti.kernel
def kernel_LSparticle_digital_elevation_force_assemble_(surfaceNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), rigid: ti.template(), vertice: ti.template(), surface: ti.template(), particle: ti.template(),
                                                        box: ti.template(), wall: ti.template(), icell_size: float, cnum: ti.types.vector(2, int), wallID: ti.template(), cplist: ti.template(), contact_model: ti.template()):
    for node in range(surfaceNum):
        end1 = surface[node]
        local_node = rigid[end1].global_node_to_local(node)
        mass_center, rotate_matrix = rigid[end1]._get_position(), SetToRotate(rigid[end1].q)
        point = mass_center + rotate_matrix @ (box[end1].scale * vertice[local_node].x)
        xCell, yCell, _ = ti.floor(point * icell_size , int)
        cellID = linearize3D(xCell, yCell, 0, cnum)

        end2 = -1
        gapn = 0.
        for wallid in range(wallID[cellID], wallID[cellID+1]):
            dist = wall[wallid]._get_norm_distance(point)
            if dist < 0.:
                projected_point = wall[wallid]._point_projection_by_distance(point, dist)
                if wall[wallid]._is_in_plane(projected_point):
                    end2 = wallid
                    gapn = dist

        if end2 != -1:
            matID1, matID2 = rigid[end1].materialID, wall[end2].materialID
            materialID = PairingMapping(matID1, matID2, max_material_num)
            parameter = vertice[local_node].parameter
            contact_model(materialID, end1, parameter, end1, end2, gapn, mass_center, point, rigid, wall, surfaceProps, cplist, dt)


@ti.kernel
def kernel_get_contact_tensor():
    pass


@ti.kernel
def kernel_compact_contact_table(total_num: int ,compact_table: ti.template(), active_contact: ti.template()):
    for i in range(1, total_num + 1):
        if active_contact[i] - active_contact[i - 1] == 1:
            compact_table[active_contact[i - 1]] = i - 1


@ti.kernel
def kernel_calculate_contact_force(particleNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), particle: ti.template(), cplist: ti.template(), 
                                   particle_particle: ti.template(), compact_table: ti.template(), active_contact: ti.template()):
    for i in range(active_contact[particle_particle[particleNum]]):
        nc = compact_table[i]
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        matID1, matID2 = particle[end1].materialID, particle[end2].materialID
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad 
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)  
        materialID = PairingMapping(matID1, matID2, max_material_num)

        if gapn < 0.:
            norm = (pos1 - pos2).normalized(Threshold)
            cpos = pos2 + (rad2 + 0.5 * gapn) * norm
            surfaceProps[materialID]._particle_particle_force_assemble(nc, end1, end2, gapn, norm, cpos, dt, particle, cplist)
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_rebulid_history_contact_list(cplist: ti.template(), hist_object_object: ti.template(), object_object: ti.types.ndarray(), 
                                        dst: ti.types.ndarray(), oldTangOverlap: ti.types.ndarray()):
    for i in range(object_object.shape[0]):
        hist_object_object[i] = object_object[i]
     
    for cp in range(object_object[object_object.shape[0] - 1]):
        cplist[cp].endID2 = dst[cp]
        cplist[cp].oldTangOverlap = vec3f(oldTangOverlap[cp, 0], oldTangOverlap[cp, 1], oldTangOverlap[cp, 2])


@ti.kernel
def kernel_rebulid_addition_history_contact_list(hist_cplist: ti.template(), hist_object_object: ti.template(), object_object: ti.types.ndarray(), 
                                                 dst: ti.types.ndarray(), oldTangOverlap: ti.types.ndarray(), oldRollAngle: ti.types.ndarray(), oldTwistAngle: ti.types.ndarray()):
    for i in range(object_object.shape[0]):
        hist_object_object[i] = object_object[i]

    for cp in range(object_object[object_object.shape[0] - 1]):
        hist_cplist[cp].DstID = dst[cp]
        hist_cplist[cp].oldTangOverlap = vec3f(oldTangOverlap[cp, 0], oldTangOverlap[cp, 1], oldTangOverlap[cp, 2])
        hist_cplist[cp].oldRollAngle = vec3f(oldRollAngle[cp, 0], oldRollAngle[cp, 1], oldRollAngle[cp, 2])
        hist_cplist[cp].oldTwistAngle = vec3f(oldTwistAngle[cp, 0], oldTwistAngle[cp, 1], oldTwistAngle[cp, 2])


@ti.func
def particle_contact_model_type1(materialID, nc, end1, end2, gapn, pos1, pos2, rad1, rad2, particle1, particle2, surfaceProps, cplist, dt):
    if gapn < surfaceProps[materialID].ncut:
        mass1, mass2 = particle1[end1]._get_mass(), particle2[end2]._get_mass()
        vel1, vel2 = particle1[end1]._get_velocity(), particle2[end2]._get_velocity()
        w1, w2 = particle1[end1]._get_angular_velocity(), particle2[end2]._get_angular_velocity()
        tangOverlapOld = cplist[nc].oldTangOverlap
        
        m_eff = EffectiveValue(mass1, mass2)
        rad_eff = EffectiveValue(rad1 + 0.5 * gapn, rad2 + 0.5 * gapn)
        norm = (pos1 - pos2).normalized(Threshold)
        cpos = pos2 + (rad2 - 0.5 * gapn) * norm
        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
        
        normal_force, tangential_force, tangOverTemp, elastic, viscous, friction = surfaceProps[materialID]._force_assemble(m_eff, rad_eff, gapn, 1., norm, v_rel, tangOverlapOld, dt)
        if ti.static(GlobalVariable.TRACKENERGY):
            elastic *= 0.5
            viscous *= 0.5
            friction *= 0.5
            particle1[end1].elastic_energy += elastic
            particle2[end2].elastic_energy += elastic
            particle1[end1].damp_energy += viscous
            particle2[end2].damp_energy += viscous
            particle1[end1].friction_energy += friction
            particle2[end2].friction_energy += friction
        Ftotal = normal_force + tangential_force
        moment1 = tangential_force.cross(pos1 - cpos)
        moment2 = tangential_force.cross(cpos - pos2)

        cplist[nc]._set_contact(normal_force, tangential_force, tangOverTemp)
        particle1[end1]._update_contact_interaction(Ftotal, moment1)
        particle2[end2]._update_contact_interaction(-Ftotal, moment2)
    else:
        cplist[nc]._no_contact()


@ti.func
def particle_contact_model_type2(materialID, nc, end1, end2, gapn, pos1, pos2, rad1, rad2, particle1, particle2, surfaceProps, cplist, dt):
    if gapn < surfaceProps[materialID].ncut:
        mass1, mass2 = particle1[end1]._get_mass(), particle2[end2]._get_mass()
        vel1, vel2 = particle1[end1]._get_velocity(), particle2[end2]._get_velocity()
        w1, w2 = particle1[end1]._get_angular_velocity(), particle2[end2]._get_angular_velocity()
        m_eff = EffectiveValue(mass1, mass2)
        rad_eff = EffectiveValue(rad1, rad2)

        norm = (pos1 - pos2).normalized(Threshold)
        cpos = pos2 + (rad2 - 0.5 * gapn) * norm
        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
        w_rel = w1 - w2
        wr_rel = norm.cross(w1) - norm.cross(w2)

        tangOverlapOld = cplist[nc].oldTangOverlap
        tangRollingOld = cplist[nc].oldRollAngle
        tangTwistingOld = cplist[nc].oldTwistAngle
        
        normal_force, tangential_force, momentum, tangOverTemp, tangRollingTemp, tangTwistingTemp = \
            surfaceProps[materialID]._force_assemble(m_eff, rad_eff, gapn, 1., norm, v_rel, w_rel, wr_rel, tangOverlapOld, tangRollingOld, tangTwistingOld, dt)
        Ftotal = normal_force + tangential_force
        resultant_momentum1 = tangential_force.cross(pos1 - cpos) + momentum
        resultant_momentum2 = tangential_force.cross(cpos - pos2) - momentum

        cplist[nc]._set_contact(normal_force, tangential_force, tangOverTemp, tangRollingTemp, tangTwistingTemp)
        particle1[end1]._update_contact_interaction(Ftotal, resultant_momentum1)
        particle2[end2]._update_contact_interaction(-Ftotal, resultant_momentum2)
    else:
        cplist[nc]._no_contact()


@ti.func
def LSparticle_contact_model_type0(materialID, nc, parameter, end1, end2, contact_radius, min_dist, mass_center1, mass_center2, global_intruding_node, local_intruding_node, 
                                   rotate_matrix2, object, grid, rigid, box, surfaceProps, cplist, dt):
    if min_dist < surfaceProps[materialID].ncut:
        vel1, vel2 = object[end1]._get_velocity(), rigid[end2]._get_velocity()
        w1, w2 = object[end1]._get_angular_velocity(), rigid[end2]._get_angular_velocity()

        dgdx = rotate_matrix2 @ box[end2].calculate_gradient(local_intruding_node, grid)
        norm = dgdx.normalized(Threshold)
        cpos = global_intruding_node - norm * (0.5 * min_dist + contact_radius)          # is right ??
        v_rel = vel1 + w1.cross(cpos - mass_center1) - (vel2 + w2.cross(cpos - mass_center2))
        tangOverlapOld = cplist[nc].oldTangOverlap
        mass1, mass2 = object[end1]._get_mass(), rigid[end2]._get_mass()   
        rad1, rad2 = object[end1]._get_contact_radius(cpos), rigid[end2]._get_contact_radius(cpos)  
        m_eff = EffectiveValue(mass1, mass2)
        rad_eff = EffectiveValue(rad1, rad2)

        normal_force, tangential_force, tangOverTemp, elastic, viscous, friction = surfaceProps[materialID]._force_assemble(m_eff, rad_eff, min_dist, parameter, dgdx, v_rel, tangOverlapOld, dt)
        if ti.static(GlobalVariable.TRACKENERGY):
            elastic *= 0.5
            viscous *= 0.5
            friction *= 0.5
            object[end1].elastic_energy += elastic
            rigid[end2].elastic_energy += elastic
            object[end1].damp_energy += viscous
            rigid[end2].damp_energy += viscous
            object[end1].friction_energy += friction
            rigid[end2].friction_energy += friction
        cplist[nc]._set_contact(normal_force, tangential_force, tangOverTemp)
        Ftotal = normal_force + tangential_force
        moment1 = Ftotal.cross(mass_center1 - cpos)
        moment2 = Ftotal.cross(cpos - mass_center2)
        object[end1]._update_contact_interaction(Ftotal, moment1)
        rigid[end2]._update_contact_interaction(-Ftotal, moment2)
    else:
        cplist[nc]._no_contact()


@ti.func
def LSparticle_contact_model_type1(materialID, nc, parameter, end1, end2, contact_radius, min_dist, mass_center1, mass_center2, global_intruding_node, local_intruding_node, 
                                   rotate_matrix2, object, grid, rigid, box, surfaceProps, cplist, dt):
    if min_dist < surfaceProps[materialID].ncut:
        vel1, vel2 = object[end1]._get_velocity(), rigid[end2]._get_velocity()
        w1, w2 = object[end1]._get_angular_velocity(), rigid[end2]._get_angular_velocity()
        
        norm = rotate_matrix2 @ box[end2].calculate_normal(local_intruding_node, grid)
        cpos = global_intruding_node - norm * (0.5 * min_dist + contact_radius)          # is right ?? 
        v_rel = vel1 + w1.cross(cpos - mass_center1) - (vel2 + w2.cross(cpos - mass_center2))
        tangOverlapOld = cplist[nc].oldTangOverlap  
        mass1, mass2 = object[end1]._get_mass(), rigid[end2]._get_mass()   
        rad1, rad2 = object[end1]._get_contact_radius(cpos), rigid[end2]._get_contact_radius(cpos)   
        m_eff = EffectiveValue(mass1, mass2)
        rad_eff = EffectiveValue(rad1, rad2)

        normal_force, tangential_force, tangOverTemp, elastic, viscous, friction = surfaceProps[materialID]._force_assemble(m_eff, rad_eff, min_dist, parameter, norm, v_rel, tangOverlapOld, dt)
        if ti.static(GlobalVariable.TRACKENERGY):
            elastic *= 0.5
            viscous *= 0.5
            friction *= 0.5
            object[end1].elastic_energy += elastic
            rigid[end2].elastic_energy += elastic
            object[end1].damp_energy += viscous
            rigid[end2].damp_energy += viscous
            object[end1].friction_energy += friction
            rigid[end2].friction_energy += friction
        cplist[nc]._set_contact(normal_force, tangential_force, tangOverTemp)
        Ftotal = normal_force + tangential_force
        moment1 = Ftotal.cross(mass_center1 - cpos)
        moment2 = Ftotal.cross(cpos - mass_center2)
        object[end1]._update_contact_interaction(Ftotal, moment1)
        rigid[end2]._update_contact_interaction(-Ftotal, moment2)
    else:
        cplist[nc]._no_contact()


@ti.func
def fluid_particle_contact_model(materialID, nc, end1, end2, gapn, pos1, pos2, rad1, rad2, particle1, particle2, surfaceProps, cplist, dt):
    if gapn < surfaceProps[materialID].ncut:
        mass1, mass2 = particle1[end1]._get_mass(), particle2[end2]._get_mass()
        vel1, vel2 = particle1[end1]._get_velocity(), particle2[end2]._get_velocity()
        w1, w2 = particle1[end1]._get_angular_velocity(), particle2[end2]._get_angular_velocity()
        m_eff = EffectiveValue(mass1, mass2)

        norm = (pos1 - pos2).normalized(Threshold)
        cpos = pos2 + (rad2 - 0.5 * gapn) * norm
        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
        
        normal_force, tangential_force = surfaceProps[materialID]._fluid_force_assemble(m_eff, gapn, 1., norm, v_rel, dt)
        Ftotal = normal_force + tangential_force
        resultant_momentum1 = tangential_force.cross(pos1 - cpos)
        resultant_momentum2 = tangential_force.cross(cpos - pos2)

        cplist[nc]._set_contact(normal_force, tangential_force, ZEROVEC3f)
        particle1[end1]._update_contact_interaction(Ftotal, resultant_momentum1)
        particle2[end2]._update_contact_interaction(-Ftotal, resultant_momentum2)
    else:
        cplist[nc]._no_contact()


@ti.func
def fluid_LSparticle_contact_model(materialID, nc, parameter, end1, end2, contact_radius, min_dist, mass_center1, mass_center2, global_intruding_node, local_intruding_node, 
                                   rotate_matrix2, object, grid, rigid, box, surfaceProps, cplist, dt):
    if min_dist < surfaceProps[materialID].ncut:
        vel1, vel2 = object[end1]._get_velocity(), rigid[end2]._get_velocity()
        w1, w2 = object[end1]._get_angular_velocity(), rigid[end2]._get_angular_velocity()
        
        norm = rotate_matrix2 @ box[end2].calculate_normal(local_intruding_node, grid)
        cpos = global_intruding_node - norm * (0.5 * min_dist + contact_radius)          # is right ?? 
        mass1, mass2 = object[end1]._get_mass(), rigid[end2]._get_mass()   
        m_eff = EffectiveValue(mass1, mass2)
        v_rel = vel1 + w1.cross(cpos - mass_center1) - (vel2 + w2.cross(cpos - mass_center2))

        normal_force, tangential_force = surfaceProps[materialID]._fluid_force_assemble(m_eff, min_dist, parameter, norm, v_rel, dt)
        cplist[nc]._set_contact(normal_force, tangential_force, ZEROVEC3f)
        Ftotal = normal_force + tangential_force
        moment1 = Ftotal.cross(mass_center1 - cpos)
        moment2 = Ftotal.cross(cpos - mass_center2)
        object[end1]._update_contact_interaction(Ftotal, moment1)
        rigid[end2]._update_contact_interaction(-Ftotal, moment2)
    else:
        cplist[nc]._no_contact()


@ti.func
def wall_contact_model_type1(materialID, nc, end1, end2, gapn, distance, pos1, particle_rad, particle, wall, surfaceProps, cplist, dt):
    fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
    if gapn < surfaceProps[materialID].ncut and fraction > Threshold:
        vel1, vel2 = particle[end1]._get_velocity(), wall[end2]._get_velocity()
        w1 = particle[end1]._get_angular_velocity()
        m_eff, rad_eff = particle[end1]._get_mass(), particle_rad + 0.5 * gapn
        tangOverlapOld = cplist[nc].oldTangOverlap

        norm = wall[end2].norm
        cpos = wall[end2]._point_projection(pos1) - 0.5 * gapn * norm
        v_rel = vel1 + w1.cross(cpos - pos1) - vel2 

        normal_force, tangential_force, tangOverTemp, elastic, viscous, friction = surfaceProps[materialID]._force_assemble(m_eff, rad_eff, gapn, 1., norm, v_rel, tangOverlapOld, dt)      
        if ti.static(GlobalVariable.TRACKENERGY):
            particle[end1].elastic_energy += elastic
            particle[end1].damp_energy += viscous
            particle[end1].friction_energy += friction
        Ftotal = fraction * (normal_force + tangential_force)
        resultant_momentum = Ftotal.cross(pos1 - cpos)
        cplist[nc]._set_contact(fraction * normal_force, fraction * tangential_force, tangOverTemp)
        particle[end1]._update_contact_interaction(Ftotal, resultant_momentum)
    else:
        cplist[nc]._no_contact()


@ti.func
def wall_contact_model_type2(materialID, nc, end1, end2, gapn, distance, pos1, particle_rad, particle, wall, surfaceProps, cplist, dt):
    fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
    if gapn < surfaceProps[materialID].ncut and fraction > Threshold:
        vel1, vel2 = particle[end1]._get_velocity(), wall[end2]._get_velocity()
        w1 = particle[end1]._get_angular_velocity()
        m_eff = particle[end1]._get_mass()
        rad_eff = particle_rad

        norm = wall[end2].norm
        cpos = wall[end2]._point_projection(pos1) - 0.5 * gapn * norm
        v_rel = vel1 + w1.cross(cpos - pos1) - vel2 
        w_rel = w1 
        wr_rel = norm.cross(w1) 

        tangOverlapOld = cplist[nc].oldTangOverlap
        tangRollingOld = cplist[nc].oldRollAngle
        tangTwistingOld = cplist[nc].oldTwistAngle

        normal_force, tangential_force, momentum, tangOverTemp, tangRollingTemp, tangTwistingTemp = \
            surfaceProps[materialID]._force_assemble(m_eff, rad_eff, gapn, 1., norm, v_rel, w_rel, wr_rel, tangOverlapOld, tangRollingOld, tangTwistingOld, dt)
        Ftotal = (normal_force + tangential_force)
        resultant_momentum = (Ftotal.cross(pos1 - cpos) + momentum)

        cplist[nc]._set_contact(fraction * normal_force, fraction * tangential_force, tangOverTemp, tangRollingTemp, tangTwistingTemp)
        particle[end1]._update_contact_interaction(fraction * Ftotal, fraction * resultant_momentum)
    else:
        cplist[nc]._no_contact()


@ti.func
def fluid_wall_contact_model(materialID, nc, end1, end2, gapn, distance, pos1, particle_rad, particle, wall, surfaceProps, cplist, dt):
    fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
    if gapn < surfaceProps[materialID].ncut and fraction > Threshold:
        vel1, vel2 = particle[end1]._get_velocity(), wall[end2]._get_velocity()
        w1 = particle[end1]._get_angular_velocity()
        m_eff, rad_eff = particle[end1]._get_mass(), particle_rad + 0.5 * gapn

        norm = wall[end2].norm
        cpos = wall[end2]._point_projection(pos1) - 0.5 * gapn * norm
        v_rel = vel1 + w1.cross(cpos - pos1) - vel2 

        normal_force, tangential_force = surfaceProps[materialID]._fluid_force_assemble(m_eff, gapn, 1., norm, v_rel, dt)
        Ftotal = fraction * (normal_force + tangential_force)
        resultant_momentum = Ftotal.cross(pos1 - cpos)

        cplist[nc]._set_contact(fraction * normal_force, fraction * tangential_force, ZEROVEC3f)
        particle[end1]._update_contact_interaction(Ftotal, resultant_momentum)
    else:
        cplist[nc]._no_contact()


@ti.func
def LSparticle_wall_contact_model(materialID, nc, parameter, end1, end2, min_dist, mass_center1, intruding_node, rigid, wall, surfaceProps, cplist, dt):
    if min_dist < surfaceProps[materialID].ncut:
        vel1, vel2 = rigid[end1]._get_velocity(), wall[end2]._get_velocity()
        w1 = rigid[end1]._get_angular_velocity()
        norm = wall[end2].norm
        cpos = intruding_node + 0.5 * norm * min_dist          # is right ?? 
        v_rel = vel1 + w1.cross(cpos - mass_center1) - vel2 

        tangOverlapOld = cplist[nc].oldTangOverlap
        m_eff, rad_eff = rigid[end1]._get_mass(), rigid[end1]._get_contact_radius(cpos)
        normal_force, tangential_force, tangOverTemp, elastic, viscous, friction = surfaceProps[materialID]._force_assemble(m_eff, rad_eff, min_dist, parameter, norm, v_rel, tangOverlapOld, dt)
        if ti.static(GlobalVariable.TRACKENERGY):
            rigid[end1].elastic_energy += elastic
            rigid[end1].damp_energy += viscous
            rigid[end1].friction_energy += friction
        Ftotal = normal_force + tangential_force
        resultant_momentum = Ftotal.cross(mass_center1 - cpos)
        cplist[nc]._set_contact(normal_force, tangential_force, tangOverTemp)
        rigid[end1]._update_contact_interaction(Ftotal, resultant_momentum)
    else:
        cplist[nc]._no_contact()
