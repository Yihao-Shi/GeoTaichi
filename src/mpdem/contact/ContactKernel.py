import taichi as ti

from src.utils.constants import ZEROVEC3f
from src.utils.ScalarFunction import PairingMapping


@ti.kernel
def update_contact_table_(potential_particle_num: int, particle_particle: ti.template(), potential_list_particle_particle: ti.template(), cplist: ti.template(), particleNum: int):
    for i in range(particleNum * potential_particle_num):
        end1 = i // potential_particle_num
        offset = i % potential_particle_num
        particle_num = particle_particle[end1 + 1] - particle_particle[end1]
        if offset < particle_num:
            nc = particle_particle[end1] + offset
            end2 = potential_list_particle_particle[i]
            
            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2


@ti.kernel
def update_wall_contact_table_(potential_wall_num: int, particle_wall: ti.template(), potential_list_particle_wall: ti.template(), cplist: ti.template(), particleNum: int):
    for i in range(particleNum * potential_wall_num):
        end1 = i // potential_wall_num
        offset = i % potential_wall_num
        particle_num = particle_wall[end1 + 1] - particle_wall[end1]
        if offset < particle_num:
            nc = particle_wall[end1] + offset
            end2 = potential_list_particle_wall[i]     

            cplist[nc].endID1 = end1
            cplist[nc].endID2 = end2


@ti.kernel
def kernel_inherit_contact_history(particleNum: int, cplist: ti.template(), hist_cplist: ti.template(), object_object: ti.template(), hist_object_object: ti.template()):
    total_contact_num = object_object[particleNum]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        cplist[nc].oldTangOverlap = find_history(end1, end2, hist_cplist, hist_object_object)
    

@ti.kernel
def copy_contact_table(object_object: ti.template(), particleNum: int, cplist: ti.template(), hist_cplist: ti.template()):
    total_contact_num = object_object[particleNum]
    for nc in range(total_contact_num):
        hist_cplist[nc].DstID = cplist[nc].endID2
        hist_cplist[nc].oldTangOverlap = cplist[nc].oldTangOverlap


@ti.kernel
def kernel_particle_particle_force_assemble_(particleNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), particle1: ti.template(), particle2: ti.template(), 
                                             cplist: ti.template(), particle_particle: ti.template()):
    total_contact_num = particle_particle[particleNum]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        matID1, matID2 = particle1[end1].materialID, particle2[end2].materialID
        pos1, pos2 = particle1[end1].x, particle2[end2].x
        rad1, rad2 = particle1[end1].rad, particle2[end2].rad 
        gapn = (pos1 - pos2).norm() - (rad1 + rad2)  
        materialID = PairingMapping(matID2, matID1, max_material_num)
        
        if gapn < 0.:
            norm = (pos1 - pos2).normalized()
            cpos = pos2 + (rad2 + 0.5 * gapn) * norm
            surfaceProps[materialID]._coupled_particle_force_assemble(nc, end1, end2, gapn, norm, cpos, dt, particle1, particle2, cplist)
        else:
            cplist[nc]._no_contact()


@ti.kernel
def kernel_particle_wall_force_assemble_(particleNum: int, dt: ti.template(), max_material_num: int, surfaceProps: ti.template(), particle: ti.template(), wall: ti.template(), 
                                         cplist: ti.template(), particle_wall: ti.template()):
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
            surfaceProps[materialID]._mpm_wall_force_assemble(nc, end1, end2, distance, gapn, norm, dt, particle, wall, cplist) 
        else:
            cplist[nc]._no_contact()
        

@ti.func
def find_history(end1, end2, hist_cplist, hist_object_object):
    tangOverlapOld = ZEROVEC3f
    for offset in range(hist_object_object[end1], hist_object_object[end1+1]):
        if end2 == hist_cplist[offset].DstID:
            tangOverlapOld = hist_cplist[offset].oldTangOverlap
            break
    return tangOverlapOld  


@ti.kernel
def kernel_fluid_particle_force_assemble_(particleNum: int, max_material_num: int, surfaceProps: ti.template(), particle1: ti.template(), particle2: ti.template(), 
                                          cplist: ti.template(),particle_particle: ti.template(), dt: ti.template()):
    total_contact_num = particle_particle[particleNum]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        matID1, matID2 = particle1[end1].materialID, particle2[end2].materialID
        materialID = PairingMapping(matID2, matID1, max_material_num)
        surfaceProps[materialID]._coupled_particle_force_assemble(nc, particle1, particle2, cplist, dt)


@ti.kernel
def kernel_fluid_wall_force_assemble_(particleNum: int, max_material_num: int, surfaceProps: ti.template(), particle: ti.template(), wall: ti.template(), 
                                      cplist: ti.template(),particle_wall: ti.template(), dt: ti.template()):
    total_contact_num = particle_wall[particleNum]
    # ti.block_local(dt)
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        matID1, matID2 = particle[end1].materialID, wall[end2].materialID
        materialID = PairingMapping(matID1, matID2, max_material_num)
        surfaceProps[materialID]._mpm_wall_force_assemble(nc, particle, wall, cplist, dt) 
