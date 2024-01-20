import taichi as ti

from src.utils.constants import MThreshold, ZEROVEC3f
from src.utils.Quaternion import SetFromTwoVec, SetToRotate, RodriguesRotationMatrix
from src.utils.TypeDefination import vec3f
from src.utils.VectorFunction import Zero2OneVector, SquareLen


@ti.kernel
def particle_calm(particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        particle[np].v = ZEROVEC3f
        particle[np].w = ZEROVEC3f
                   
                       
@ti.kernel
def clump_calm(clumpNum: int, clump: ti.template()):
    for nclump in range(clumpNum):
        clump[nclump].v = ZEROVEC3f
        clump[nclump].w = ZEROVEC3f


@ti.kernel
def reset_verlet_disp_(objectNum: int, object: ti.template()):
    for nobject in range(objectNum):
        object[nobject]._renew_verlet()


@ti.kernel
def find_particle_verlet_disp_(particleNum: int, particle: ti.template()) -> float:
    max_verlet_disp = 0.
    for np in range(particleNum):
        verletDisp = particle[np].verletDisp
        verletDispLen = verletDisp[0] * verletDisp[0] + verletDisp[1] * verletDisp[1] + verletDisp[2] * verletDisp[2]
        ti.atomic_max(max_verlet_disp, verletDispLen)
    return max_verlet_disp


@ti.kernel
def find_particle_min_mass_(particleNum: int, particle: ti.template()) -> float:
    min_mass = MThreshold
    for np in range(particleNum):
        mass = particle[np].m
        ti.atomic_min(min_mass, mass)
    return min_mass


@ti.kernel
def validate_displacement_(limit: float, objectNum: int, object: ti.template()) -> int:
    flag = 0
    for nobject in range(objectNum):
        if flag == 0 and SquareLen(object[nobject].verletDisp) > limit:
            flag = 1
    return flag


@ti.kernel
def find_particle_max_radius_(particleNum: int, particle: ti.template()) -> float:
    max_radius = 0.
    for np in range(particleNum):
        radius = particle[np].rad
        ti.atomic_max(max_radius, radius)
    return max_radius


@ti.kernel
def find_particle_min_radius_(particleNum: int, particle: ti.template()) -> float:
    min_radius = MThreshold
    for np in range(particleNum):
        radius = particle[np].rad
        ti.atomic_min(min_radius, radius)
    return min_radius


@ti.kernel
def find_patch_max_radius_(wallNum: int, wall: ti.template()) -> float:
    max_radius = 0.
    for nw in range(wallNum):
        radius = wall[nw]._get_bounding_radius()
        ti.atomic_max(max_radius, radius)
    return max_radius


@ti.kernel
def find_patch_min_radius_(wallNum: int, wall: ti.template()) -> float:
    min_radius = MThreshold
    for nw in range(wallNum):
        radius = wall[nw]._get_bounding_radius()
        ti.atomic_min(min_radius, radius)
    return min_radius

@ti.kernel
def check_in_domain(domain: ti.types.vector(3, float), particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        if particle[np].active == 1 and not is_in_domain(domain, particle[np].x):
            particle[np].active = ti.u8(0)

@ti.func
def is_in_domain(domain, position):
    in_domain = 1
    if in_domain == 1 and position[0] < 0.: in_domain = 0
    elif in_domain == 1 and position[1] < 0.: in_domain = 0
    elif in_domain == 1 and position[2] < 0.: in_domain = 0
    elif in_domain == 1 and position[0] > domain[0]: in_domain = 0
    elif in_domain == 1 and position[1] > domain[1]: in_domain = 0
    elif in_domain == 1 and position[2] > domain[2]: in_domain = 0
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
def modify_sphere_bodyID_in_region(value: int, sphereNum: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            particle[particleID].bodyID = ti.u8(value)


@ti.kernel
def modify_sphere_groupID_in_region(value: int, sphereNum: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            particle[particleID].groupID = ti.u8(value)


@ti.kernel
def modify_sphere_materialID_in_region(value: int, sphereNum: int, sphere: ti.template(), particle: ti.template(), material: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            old_materialID = particle[particleID].materialID
            old_density = material[old_materialID].density
            particle[particleID].materialID = ti.u8(value)
            density = material[value].density
            ratio = density / old_density
            particle[particleID].mass *= ratio 
            sphere[nsphere].inv_I *= 1. / ratio 


@ti.kernel
def modify_sphere_radius_in_region(value: float, sphereNum: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            ratio = value / particle[particleID].rad
            particle[particleID].rad = value
            particle[particleID].m *= ratio * ratio * ratio
            sphere[nsphere].inv_I *= 1. / (ratio * ratio * ratio * ratio * ratio)


@ti.kernel
def modify_sphere_position_in_region(factor: int, value: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            particle[particleID].x = factor * particle[particleID].x + value


@ti.kernel
def modify_sphere_velocity_in_region(factor: int, value: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            particle[particleID].v = factor * particle[particleID].v + value


@ti.kernel
def modify_sphere_angular_velocity_in_region(factor: int, value: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            particle[particleID].w = factor * particle[particleID].w + value


@ti.kernel
def modify_sphere_orientation_in_region(value: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            sphere[nsphere].q = SetFromTwoVec(vec3f(0., 0., 1), value)


@ti.kernel
def modify_sphere_fix_v_in_region(value: ti.types.vector(3, ti.u8), sphereNum: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            sphere[nsphere].fix_v = Zero2OneVector(value)

        
@ti.kernel
def modify_sphere_fix_w_in_region(value: ti.types.vector(3, ti.u8), sphereNum: int, sphere: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if is_in_region(particle[particleID].x):
            sphere[nsphere].fix_w = Zero2OneVector(value)


@ti.kernel
def modify_clump_bodyID_in_region(value: int, clumpNum: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break
        
        if is_in_region == 1:
            for np in range(startIndex, endIndex):
                particle[np].bodyID = ti.u8(value)


@ti.kernel
def modify_clump_groupID_in_region(value: int, clumpNum: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break
        
        if is_in_region == 1:
            for np in range(startIndex, endIndex):
                particle[np].groupID = ti.u8(value)


@ti.kernel
def modify_clump_materialID_in_region(value: int, clumpNum: int, clump: ti.template(), particle: ti.template(), material: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break
        
        if is_in_region == 1:
            for np in range(startIndex, endIndex):
                old_materialID = particle[np].materialID
                old_density = material[old_materialID].density
                particle[np].materialID = ti.u8(value)
                density = material[value].density
                ratio = density / old_density
                particle[np].mass *= ratio 
            clump[nclump].inv_I *= 1. / ratio 


@ti.kernel
def modify_clump_radius_in_region(value: float, clumpNum: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break
        
        equivalent_radius = clump[nclump].equi_r
        if is_in_region == 1:
            for np in range(startIndex, endIndex):
                ratio = value / equivalent_radius
                particle[np].rad *= ratio
                particle[np].m *= ratio * ratio * ratio
            clump[nclump].inv_I *= 1. / (ratio * ratio * ratio * ratio * ratio)


@ti.kernel
def modify_clump_position_in_region(factor: int, value: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break
        
        if is_in_region == 1:
            for np in range(startIndex, endIndex):
                particle[np].x = factor * particle[np].x + value
            clump[nclump].mass_center = factor * clump[nclump].mass_center + value


@ti.kernel
def modify_clump_velocity_in_region(factor: int, value: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break
        
        if is_in_region == 1:
            for np in range(startIndex, endIndex):
                particle[np].v = factor * particle[np].v + value
            clump[nclump].v = factor * clump[nclump].v + value


@ti.kernel
def modify_clump_angular_velocity_in_region(factor: int, value: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break
        
        if is_in_region == 1:
            center_vel = clump[nclump].v
            mass_center = clump[nclump].mass_center
            for np in range(startIndex, endIndex):
                angular_vel = factor * particle[np].w + value
                particle_vel = center_vel + angular_vel.cross(particle[np].x - mass_center)
                particle[np].v = particle_vel
                particle[np].w = angular_vel
            clump[nclump].w = factor * clump[nclump].w  + value


@ti.kernel
def modify_clump_orientation_in_region(value: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break
        
        if is_in_region == 1:
            rmatrix = SetToRotate(clump[nclump].q)
            mass_center = clump[nclump].mass_center
            clump[nclump].q = SetFromTwoVec(vec3f(0., 0., 1), value)
            update_rmatrix = SetToRotate(clump[nclump].q)
            for np in range(startIndex, endIndex):
                particle[np].x = update_rmatrix @ (rmatrix.transpose() @ (particle[np].x - mass_center)) + mass_center


@ti.kernel
def modify_clump_fix_v_in_region(value: ti.types.vector(3, ti.u8), clumpNum: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            clump[nclump].fix_v = Zero2OneVector(value)

        
@ti.kernel
def modify_clump_fix_w_in_region(value: ti.types.vector(3, ti.u8), clumpNum: int, clump: ti.template(), particle: ti.template(), is_in_region: ti.template()):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        is_in_region = 1
        for np in range(startIndex, endIndex):
            if not is_in_region(particle[np].x):
                is_in_region = 0
                break

        if is_in_region == 1:
            clump[nclump].fix_w = Zero2OneVector(value)


@ti.kernel
def modify_sphere_bodyID(value: int, sphereNum: int, sphere: ti.template(), particle: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if nsphere == bodyID:
            particle[particleID].bodyID = ti.u8(value)


@ti.kernel
def modify_sphere_groupID(value: int, sphereNum: int, sphere: ti.template(), particle: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if nsphere == bodyID:
            particle[particleID].groupID = ti.u8(value)


@ti.kernel
def modify_sphere_materialID(value: int, sphereNum: int, sphere: ti.template(), particle: ti.template(), material: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if nsphere == bodyID:
            old_materialID = particle[particleID].materialID
            old_density = material[old_materialID].density
            particle[particleID].materialID = ti.u8(value)
            density = material[value].density
            ratio = density / old_density
            particle[particleID].mass *= ratio 
            sphere[nsphere].inv_I *= 1. / ratio 


@ti.kernel
def modify_sphere_radius(value: float, sphereNum: int, sphere: ti.template(), particle: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if nsphere == bodyID:
            ratio = value / particle[particleID].rad
            particle[particleID].rad = value
            particle[particleID].m *= ratio * ratio * ratio
            sphere[nsphere].inv_I *= 1. / (ratio * ratio * ratio * ratio * ratio)


@ti.kernel
def modify_sphere_position(factor: int, value: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if nsphere == bodyID:
            particle[particleID].x = factor * particle[particleID].x + value


@ti.kernel
def modify_sphere_velocity(factor: int, value: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if nsphere == bodyID:
            particle[particleID].v = factor * particle[particleID].v + value


@ti.kernel
def modify_sphere_angular_velocity(factor: int, value: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        particleID = sphere[nsphere].sphereIndex
        if nsphere == bodyID:
            particle[particleID].w = factor * particle[particleID].w + value


@ti.kernel
def modify_sphere_orientation(value: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        if nsphere == bodyID:
            sphere[nsphere].q = SetFromTwoVec(vec3f(0., 0., 1), value)


@ti.kernel
def modify_sphere_fix_v(value: ti.types.vector(3, ti.u8), sphereNum: int, sphere: ti.template(), particle: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        if nsphere == bodyID:
            sphere[nsphere].fix_v = Zero2OneVector(value)

        
@ti.kernel
def modify_sphere_fix_w(value: ti.types.vector(3, ti.u8), sphereNum: int, sphere: ti.template(), particle: ti.template(), bodyID: int):
    for nsphere in range(sphereNum):
        if nsphere == bodyID:
            sphere[nsphere].fix_w = Zero2OneVector(value)


@ti.kernel
def modify_clump_bodyID(value: int, clumpNum: int, clump: ti.template(), particle: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        if nclump == bodyID:
            for np in range(startIndex, endIndex):
                particle[np].bodyID = ti.u8(value)


@ti.kernel
def modify_clump_groupID(value: int, clumpNum: int, clump: ti.template(), particle: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        if nclump == bodyID:
            for np in range(startIndex, endIndex):
                particle[np].groupID = ti.u8(value)


@ti.kernel
def modify_clump_materialID(value: int, clumpNum: int, clump: ti.template(), particle: ti.template(), material: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        if nclump == bodyID:
            for np in range(startIndex, endIndex):
                old_materialID = particle[np].materialID
                old_density = material[old_materialID].density
                particle[np].materialID = ti.u8(value)
                density = material[value].density
                ratio = density / old_density
                particle[np].mass *= ratio 
            clump[nclump].inv_I *= 1. / ratio 


@ti.kernel
def modify_clump_radius(value: float, clumpNum: int, clump: ti.template(), particle: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex
        
        equivalent_radius = clump[nclump].equi_r
        if nclump == bodyID:
            for np in range(startIndex, endIndex):
                ratio = value / equivalent_radius
                particle[np].rad *= ratio
                particle[np].m *= ratio * ratio * ratio
            clump[nclump].inv_I *= 1. / (ratio * ratio * ratio * ratio * ratio)


@ti.kernel
def modify_clump_position(factor: int, value: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        if nclump == bodyID:
            for np in range(startIndex, endIndex):
                particle[np].x = factor * particle[np].x + value
            clump[nclump].mass_center = factor * clump[nclump].mass_center + value


@ti.kernel
def modify_clump_velocity(factor: int, value: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        if nclump == bodyID:
            for np in range(startIndex, endIndex):
                particle[np].v = factor * particle[np].v + value
            clump[nclump].v = factor * clump[nclump].v + value


@ti.kernel
def modify_clump_angular_velocity(factor: int, value: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        if nclump == bodyID:
            center_vel = clump[nclump].v
            mass_center = clump[nclump].mass_center
            for np in range(startIndex, endIndex):
                angular_vel = factor * particle[np].w + value
                particle_vel = center_vel + angular_vel.cross(particle[np].x - mass_center)
                particle[np].v = particle_vel
                particle[np].w = angular_vel
            clump[nclump].w = factor * clump[nclump].w  + value


@ti.kernel
def modify_clump_orientation(value: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        startIndex = clump[nclump].startIndex
        endIndex = clump[nclump].endIndex

        if nclump == bodyID:
            rmatrix = SetToRotate(clump[nclump].q)
            mass_center = clump[nclump].mass_center
            clump[nclump].q = SetFromTwoVec(vec3f(0., 0., 1), value)
            update_rmatrix = SetToRotate(clump[nclump].q)
            for np in range(startIndex, endIndex):
                particle[np].x = update_rmatrix @ (rmatrix.transpose() @ (particle[np].x - mass_center)) + mass_center


@ti.kernel
def modify_clump_fix_v(value: ti.types.vector(3, ti.u8), clumpNum: int, clump: ti.template(), particle: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        if nclump == bodyID:
            clump[nclump].fix_v = Zero2OneVector(value)

        
@ti.kernel
def modify_clump_fix_w(value: ti.types.vector(3, ti.u8), clumpNum: int, clump: ti.template(), particle: ti.template(), bodyID: int):
    for nclump in range(clumpNum):
        if nclump == bodyID:
            clump[nclump].fix_w = Zero2OneVector(value)


@ti.kernel
def modify_wall_activate_status(wallID: int, status: int, wallNum: int, wall: ti.template()):
    for wall_id in range(wallNum):
        if wall[wall_id].wallID == wallID:
            wall[wall_id].active = ti.u8(status)


@ti.kernel
def modify_wall_materialID(wallID: int, value: int, wallNum: int, wall: ti.template()):
    for wall_id in range(wallNum):
        if wall[wall_id].wallID == wallID:
            wall[wall_id].materialID = value


@ti.kernel
def modify_plane_position(factor: int, wallID: int, value: ti.types.vector(3, float), wallNum: int, wall: ti.template()):
    for wall_id in range(wallNum):
        if wall[wall_id].wallID == wallID:
            wall[wall_id].point = wall[wall_id].point * factor + value


@ti.kernel
def modify_plane_orientation(factor: int, wallID: int, value: ti.types.vector(3, float), wallNum: int, wall: ti.template()):
    for wall_id in range(wallNum):
        if wall[wall_id].wallID == wallID:
            wall[wall_id].norm = wall[wall_id].norm * factor + value


@ti.kernel
def modify_triangle_position(factor: int, wallID: int, value: ti.types.vector(3, float), wallNum: int, wall: ti.template()):
    for wall_id in range(wallNum):
        if wall[wall_id].wallID == wallID:
            wall[wall_id].vertice1 = wall[wall_id].vertice1 * factor + value
            wall[wall_id].vertice2 = wall[wall_id].vertice2 * factor + value
            wall[wall_id].vertice3 = wall[wall_id].vertice3 * factor + value
            wall[wall_id].verletDisp = ZEROVEC3f


@ti.kernel
def modify_triangle_velocity(factor: int, wallID: int, value: ti.types.vector(3, float), wallNum: int, wall: ti.template()):
    for wall_id in range(wallNum):
        if wall[wall_id].wallID == wallID:
            wall[wall_id].v = wall[wall_id].v * factor + value


@ti.kernel
def modify_triangle_orientation(mode: int, factor: int, wallID: int, new_direction: ti.types.vector(3, float), rotation_center: ti.types.vector(3, float), wallNum: int, wall: ti.template()):
    for wall_id in range(wallNum):
        if wall[wall_id].wallID == wallID:
            if mode == 1:
                rotation_center = wall[wall_id]._get_center()
            rotation_matrix = RodriguesRotationMatrix(wall[wall_id].norm, factor * wall[wall_id].norm + new_direction)
            wall[wall_id].vertice1 = rotation_matrix @ (wall[wall_id].vertice1 - rotation_center) + rotation_center
            wall[wall_id].vertice2 = rotation_matrix @ (wall[wall_id].vertice2 - rotation_center) + rotation_center
            wall[wall_id].vertice3 = rotation_matrix @ (wall[wall_id].vertice3 - rotation_center) + rotation_center
            wall[wall_id].norm = factor * wall[wall_id].norm + new_direction
