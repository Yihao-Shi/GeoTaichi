import taichi as ti

from src.utils.constants import ZEROVEC3f
from src.utils.Quaternion import SetFromTwoVec, SetToRotate
from src.utils.ScalarFunction import PairingMapping
from src.utils.TypeDefination import vec3f


@ti.func
def inherit_history(nc, hist_nc, cplist, hist_cplist):
    cplist[nc].oldTangOverlap = hist_cplist[hist_nc].oldTangOverlap

@ti.func
def inherit_addition_history(nc, hist_nc, cplist, hist_cplist):
    cplist[nc].oldTangOverlap = hist_cplist[hist_nc].oldTangOverlap
    cplist[nc].oldRollAngle = hist_cplist[hist_nc].oldRollAngle
    cplist[nc].oldTwistAngle = hist_cplist[hist_nc].oldTwistAngle

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


@ti.kernel
def no_operation():
    pass

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
@ti.func
def append_contact_offset(np, max_coordination_num, particle_offset):
    offset = np * max_coordination_num + ti.atomic_add(particle_offset[np], 1)
    '''
    offset = ti.atomic_add(particle_offset[np], 1)
    if offset < max_coordination_num:
        offset =  np * max_coordination_num + offset
    else:
        print(f"The max_coordination_number should be set as: {max_coordination_num + 1}")
    '''
    return offset

@ti.kernel
def update_contact_table_(potential_particle_num: int, particleNum: int, particle_particle: ti.template(), potential_list_particle_particle: ti.template(), cplist: ti.template()):
    '''
    for end1 in range(particleNum):
        particle_num = particle_particle[end1 + 1] - particle_particle[end1]
        for offset in range(particle_num):
            i = end1 * potential_particle_num + offset
            end2 = potential_list_particle_particle[i]     

            nc = particle_particle[end1] + offset
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2

    # TODO: considering thread load balance
    '''
    for i in range(particleNum * potential_particle_num):
        end1 = i // potential_particle_num
        offset = i % potential_particle_num
        particle_num = particle_particle[end1 + 1] - particle_particle[end1]
        if offset < particle_num:
            nc = particle_particle[end1] + offset
            end2 = potential_list_particle_particle[i]
            
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2

            # particle_contact[i] = nc
            # Nj = potential_particle_num * (end2 + 1)
            # for j in range(Nj - iparticle_particle[end2 + 1], Nj):
            #     if potential_list_particle_particle[j] == end1:
            #         particle_contact[j] = nc
            #         break


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
def update_wall_contact_table_(potential_wall_num: int, particleNum: int, particle_wall: ti.template(), potential_list_particle_wall: ti.template(), cplist: ti.template()):
    '''
    for end1 in range(particleNum):
        particle_num = particle_wall[end1 + 1] - particle_wall[end1]
        for offset in range(particle_num):
            i = end1 * potential_wall_num + offset
            end2 = potential_list_particle_wall[i]     

            nc = particle_wall[end1] + offset
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2

    # TODO: considering thread load balance
    '''
    for i in range(particleNum * potential_wall_num):
        end1 = i // potential_wall_num
        offset = i % potential_wall_num
        particle_num = particle_wall[end1 + 1] - particle_wall[end1]
        if offset < particle_num:
            nc = particle_wall[end1] + offset
            end2 = potential_list_particle_wall[i]     

            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2
            # particle_contact[i] = nc
    

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
def kernel_flag2index(active_flag: ti.template(), active_index: ti.template()):
    pass


@ti.kernel
def kernel_particle_particle_force_assemble_(particleNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), particle: ti.template(), cplist: ti.template(), particle_particle: ti.template()):
    total_contact_num = particle_particle[particleNum]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        matID1, matID2 = particle[end1].materialID, particle[end2].materialID
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad 
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)  
        materialID = PairingMapping(matID1, matID2, max_material_num)

        if gapn < 0.:
            norm = (pos1 - pos2).normalized()
            cpos = pos2 + (rad2 + 0.5 * gapn) * norm
            surfaceProps[materialID]._particle_particle_force_assemble(nc, end1, end2, gapn, norm, cpos, dt, particle, cplist)
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_particle_wall_force_assemble_(particleNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), particle: ti.template(), 
                                         wall: ti.template(), cplist: ti.template(), particle_wall: ti.template()):
    total_contact_num = particle_wall[particleNum]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        matID1, matID2 = particle[end1].materialID, wall[end2].materialID
        pos1, particle_rad = particle[end1].x, particle[end1].rad
        distance = wall[end2]._get_norm_distance(pos1)
        gapn = distance - particle_rad
        materialID = PairingMapping(matID1, matID2, max_material_num)

        if gapn < 0.:
            norm = wall[end2].norm
            cpos = wall[end2]._point_projection(pos1) - 0.5 * gapn * norm
            surfaceProps[materialID]._particle_wall_force_assemble(nc, end1, end2, distance, gapn, norm, cpos, dt, particle, wall, cplist)      
        else:
            cplist[nc]._no_contact()

@ti.kernel
def kernel_get_contact_tensor():
    pass


@ti.kernel
def copy_histcp2cp(particleNum: int, object_object: ti.template(), hist_object_object: ti.template(), cplist: ti.template(), hist_cplist: ti.template()):
    contact_num = object_object[particleNum]
    for nc in range(contact_num):
        end1 = cplist[nc].endID1
        end2 = cplist[nc].endID2
        for j in range(hist_object_object[end1], hist_object_object[end1 + 1]):
            if hist_cplist[j].DstID == end2:
                cplist[nc].oldTangOverlap = hist_cplist[j].oldTangOverlap
                break


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
            norm = (pos1 - pos2).normalized()
            cpos = pos2 + (rad2 + 0.5 * gapn) * norm
            surfaceProps[materialID]._particle_particle_force_assemble(nc, end1, end2, gapn, norm, cpos, dt, particle, cplist)
        else:
            cplist[nc]._no_contact()

'''
@ti.kernel
def kernel_particle_particle_narrow_detection_(particleNum: int, particle: ti.template(), cplist: ti.template(), particle_particle: ti.template(), active_contact: ti.template()) -> int:
    total_contact_num = particle_particle[particleNum]
    # ti.block_local(dt)
    active_contact.fill(0)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad 
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)  
        if gapn < 0.:
            active_contact[nc + 1] = 1
        else:
            cplist[nc].cnforce = ZEROVEC3f
            cplist[nc].csforce = ZEROVEC3f
            cplist[nc].oldTangOverlap = ZEROVEC3f
    return total_contact_num
    
@ti.kernel
def kernel_particle_wall_narrow_detection_(particleNum: int, particle: ti.template(), wall: ti.template(), cplist: ti.template(), particle_particle: ti.template(), active_contact: ti.template()) -> int:
    total_contact_num = particle_particle[particleNum]
    # ti.block_local(dt)
    active_contact.fill(0)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        pos1, pos2 = particle[end1].x, particle[end2].x
        rad1, rad2 = particle[end1].rad, particle[end2].rad 
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)  
        if gapn < 0.:
            active_contact[nc + 1] = 1
        else:
            cplist[nc].cnforce = ZEROVEC3f
            cplist[nc].csforce = ZEROVEC3f
            cplist[nc].oldTangOverlap = ZEROVEC3f
    return total_contact_num
'''
