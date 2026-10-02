import taichi as ti

from src.utils.constants import MThreshold, ZEROVEC3f, LThreshold
from src.utils.Quaternion import SetFromTwoVec, SetToRotate, RodriguesRotationMatrix
from src.utils.TypeDefination import vec3f, vec4f, vec2i
from src.utils.VectorFunction import SquareLen
from src.utils.BitFunction import Zero2OneVector


@ti.kernel
def update_particle_storage_(particleNum: ti.types.ndarray(), sphereNum: ti.types.ndarray(), clumpNum: ti.types.ndarray(), particle: ti.template(), sphere: ti.template(), clump: ti.template()):
    active = True
    remaining_sphere = 0
    remaining_clump = 0
    remaining_particle = 0
    delete_particle = 0
    ti.loop_config(serialize=True)
    for np in range(particleNum[0]):
        multisphereIndex = particle[np].multisphereIndex
        if ti.static(not sphere == None):
            if multisphereIndex < 0:
                if int(particle[np].active) == 1:
                    particle[remaining_particle] = particle[np]
                    particle[remaining_particle].multisphereIndex += delete_particle
                    sphere[remaining_sphere] = sphere[-multisphereIndex - 1]
                    sphere[remaining_sphere].sphereIndex -= delete_particle
                    remaining_particle += 1
                    remaining_sphere += 1
                else:
                    delete_particle += 1
        if ti.static(not clump == None):
            if multisphereIndex >= 0:
                if active and int(particle[np].active) == 0:
                    active = False
                startIndex = clump[multisphereIndex].startIndex
                endIndex = clump[multisphereIndex].endIndex
                if np == endIndex:
                    pebble_num = endIndex - startIndex + 1
                    if active:
                        for npebble in range(pebble_num):
                            particle[remaining_particle + npebble] = particle[np]
                            particle[remaining_particle + npebble].multisphereIndex -= delete_particle
                        clump[remaining_clump] = clump[multisphereIndex]
                        clump[remaining_clump].startIndex -= delete_particle
                        clump[remaining_clump].endIndex -= delete_particle
                        remaining_particle += pebble_num
                        remaining_clump += 1
                    else:
                        delete_particle += pebble_num
                    active = True
    sphereNum[0] = remaining_sphere
    clumpNum[0] = remaining_clump
    particleNum[0] = remaining_particle

@ti.kernel
def update_LSparticle_storage_(particleNum: ti.types.ndarray(), rigidNum: ti.types.ndarray(), surfaceNum: ti.types.ndarray(), particle: ti.template(), box: ti.template(), rigid: ti.template()):
    remaining_particle = 0
    remaining_node = 0
    delete_particle = 0
    delete_node = 0
    ti.loop_config(serialize=True)
    for np in range(particleNum[0]):
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
    particleNum[0] = remaining_particle
    rigidNum[0] = remaining_particle
    surfaceNum[0] = remaining_node

@ti.kernel
def update_LSsurface_storage_(rigidNum: int, rigid: ti.template(), surface: ti.template()):
    for nb in range(rigidNum):
        for i in range(rigid[nb].startNode, rigid[nb].endNode):
            surface[i] = nb

@ti.kernel
def update_ISparticle_storage_(particleNum: ti.types.ndarray(), rigidNum: ti.types.ndarray(), particle: ti.template(), rigid: ti.template()):
    remaining_particle = 0
    delete_particle = 0
    ti.loop_config(serialize=True)
    for np in range(particleNum[0]):
        if int(particle[np].active) == 1:
            particle[remaining_particle] = particle[np]
            rigid[remaining_particle] = rigid[np]
            remaining_particle += 1
        else:
            delete_particle += 1
    particleNum[0] = remaining_particle
    rigidNum[0] = remaining_particle

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
        mass = particle[np]._get_mass()
        ti.atomic_min(min_mass, mass)
    return min_mass


@ti.kernel
def find_left_bottom_scene_(particleNum: int, particle: ti.template()) -> ti.types.vector(3, float):
    left_bottom_scene = vec3f(MThreshold, MThreshold, MThreshold)
    for np in range(particleNum):
        position = particle[np]._get_position()
        ti.atomic_min(left_bottom_scene[0], position[0])
        ti.atomic_min(left_bottom_scene[1], position[1])
        ti.atomic_min(left_bottom_scene[2], position[2])
    return left_bottom_scene


@ti.kernel
def find_right_top_scene_(particleNum: int, particle: ti.template()) -> ti.types.vector(3, float):
    right_top_scene = vec3f(-MThreshold, -MThreshold, -MThreshold)
    for np in range(particleNum):
        position = particle[np]._get_position()
        ti.atomic_max(right_top_scene[0], position[0])
        ti.atomic_max(right_top_scene[1], position[1])
        ti.atomic_max(right_top_scene[2], position[2])
    return right_top_scene


@ti.kernel
def validate_displacement_(limit: float, objectNum: int, object: ti.template()) -> int:
    flag = 0
    for nobject in range(objectNum):
        if flag == 0 and SquareLen(object[nobject].verletDisp) >= limit:
            flag = 1
    return flag


@ti.kernel
def find_expect_extent_(verlet_distance: float, bodyNum: ti.template(), box: ti.template()) -> ti.types.vector(2, int):
    pid, extent = -1, 0
    ti.loop_config(serialize=True)
    for np in range(bodyNum):
        if verlet_distance > box[np].grid_space * box[np].extent:
            if ti.ceil(verlet_distance / box[np].grid_space, int) > extent:
                pid = np
                extent = ti.ceil(verlet_distance / box[np].grid_space, int)
    return vec2i(pid, extent)


@ti.kernel
def find_min_grid_space_(bodyNum: int, box: ti.template()) -> float:
    min_grid_space = 1e15
    for np in range(bodyNum):
        grid_space = box[np].grid_space
        ti.atomic_min(min_grid_space, grid_space)
    return min_grid_space


@ti.kernel
def find_min_extent_(bodyNum: int, box: ti.template()) -> float:
    min_extent = 1e15
    for np in range(bodyNum):
        grid_space = box[np].grid_space * box[np].extent
        ti.atomic_min(min_extent, grid_space)
    return min_extent


@ti.kernel
def find_particle_max_radius_(particleNum: int, particle: ti.template()) -> float:
    max_radius = 0.
    for np in range(particleNum):
        radius = particle[np]._get_radius()
        ti.atomic_max(max_radius, radius)
    return max_radius


@ti.kernel
def find_particle_min_radius_(particleNum: int, particle: ti.template()) -> float:
    min_radius = MThreshold
    for np in range(particleNum):
        radius = particle[np]._get_radius()
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
def kernel_delete_particles(particleNum: int, particle: ti.template(), bodyID: int):
    for np in range(particleNum):
        if particle[np].bodyID == bodyID:
            particle[np].active = ti.u8(0)


@ti.kernel
def kernel_delete_particles_in_region(particleNum: int, particle: ti.template(), is_in_region: ti.template()):
    for np in range(particleNum):
        if is_in_region(particle[np].x):
            particle[np].active = ti.u8(0)

@ti.kernel
def apply_boundary_conditions(domain: ti.types.vector(3, float), particleNum: int, particle: ti.template(), 
                              xboundary: ti.template(), yboundary: ti.template(), zboundary: ti.template()) -> int:
    not_in_xdomain, not_in_ydomain, not_in_zdomain = False, False, False
    for np in range(particleNum):
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
    if in_domain == True and position[axis] < 0.: 
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
    if in_domain == True and position[axis] < 0.: 
        in_domain = False
        particle[np].x[axis] = LThreshold * domain[axis]
        particle[np].v = -particle[np].v
    elif in_domain == True and position[axis] > domain[axis]: 
        in_domain = False
        particle[np].x[axis] = (1. - LThreshold) * domain[axis]
        particle[np].v = -particle[np].v
    return False

@ti.func
def period_boundary(np, axis, domain, particle):
    in_domain = True
    position = particle[np].x
    if in_domain == True and position[axis] < 0.: 
        in_domain = False
        particle[np].x[axis] += domain[axis]
    elif in_domain == True and position[axis] > domain[axis]: 
        in_domain = False
        particle[np].x[axis] -= domain[axis]
    return False

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
def modify_levelset_groupID_in_region(value: int, rigidNum: int, rigid: ti.template(), is_in_region: ti.template()):
    for nrigid in range(rigidNum):
        if is_in_region(rigid[nrigid].mass_center):
            rigid[nrigid].groupID = ti.u8(value)


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
def modify_material_density(factor: int, value: float, materialID: int, material: ti.template()):
    density = material[materialID].density
    material[materialID].density = factor * density + value


@ti.kernel
def modify_material_fdamp(factor: int, value: float, materialID: int, material: ti.template()):
    fdamp = material[materialID].fdamp
    material[materialID].fdamp = factor * fdamp + value


@ti.kernel
def modify_material_tdamp(factor: int, value: float, materialID: int, material: ti.template()):
    tdamp = material[materialID].tdamp
    material[materialID].tdamp = factor * tdamp + value


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


@ti.kernel
def create_level_set_grids_(gridNum: int, gridSum: int, grid: ti.template(), distance_fields: ti.types.ndarray()):
    for i in range(gridSum):
        grid[i + gridNum]._set_grid(distance_fields[i])


@ti.kernel
def create_level_set_surface(surfaceNum: int, surfaceSum: int, surface_node: ti.template(), surface_nodes: ti.types.ndarray(), parameters: ti.types.ndarray()):
    for i in range(surfaceSum):
        surface_node[i + surfaceNum]._set_surface_node(vec3f(surface_nodes[i, 0], surface_nodes[i, 1], surface_nodes[i, 2]))
        surface_node[i + surfaceNum]._set_coefficient(parameters[i])


@ti.kernel
def kernel_add_polysuperellipsoid_parameter(surfaceNum: int, surface: ti.template(), xrad1: float, yrad1: float, zrad1: float, epsilon_e: float, epsilon_n: float, xrad2: float, yrad2: float, zrad2: float):
    surface[surfaceNum]._add_template_parameter(xrad1, yrad1, zrad1, epsilon_e, epsilon_n, xrad2, yrad2, zrad2)


@ti.kernel
def kernel_add_polysuperquadrics_parameter(surfaceNum: int, surface: ti.template(), xrad1: float, yrad1: float, zrad1: float, epsilon_x: float, epsilon_y: float, epsilon_z: float, xrad2: float, yrad2: float, zrad2: float):
    surface[surfaceNum]._add_template_parameter(xrad1, yrad1, zrad1, epsilon_x, epsilon_y, epsilon_z, xrad2, yrad2, zrad2)

@ti.kernel
def kernel_visualize_levelset_surface_(rigidNum: int, surface_node: ti.template(), visualzie_surface_node: ti.template(), rigid: ti.template(), box: ti.template()):
    for master in range(rigidNum):
        for node in range(rigid[master]._start_node(), rigid[master]._end_node()):
            gnode = rigid[master].local_node_to_global(node)
            rotate_matrix = SetToRotate(rigid[master].q)
            visualzie_surface_node[gnode] = rigid[master].mass_center + rotate_matrix @ (box[master].scale * surface_node[node].x)

@ti.kernel
def kernel_visualize_implicit_surface_(rigidNum: int, template_vertice_num: ti.types.ndarray(), stacked_vertices: ti.types.ndarray(), total_vertice_num: ti.types.ndarray(), visualzie_surface_node: ti.template(), rigid: ti.template()):
    for master in range(rigidNum):
        templateID = rigid[master].templateID
        scale = rigid[master].scale
        begin_index = total_vertice_num[master]
        for node in range(template_vertice_num[templateID], template_vertice_num[templateID + 1]):
            gnode = node - template_vertice_num[templateID]
            rotate_matrix = SetToRotate(rigid[master].q)
            visualzie_surface_node[begin_index + gnode] = rigid[master].mass_center + rotate_matrix @ (scale * vec3f(stacked_vertices[node, 0], stacked_vertices[node, 1], stacked_vertices[node, 2]))

@ti.kernel
def kernel_postvisualize_surface_(surface_num: int, surface_node: ti.types.ndarray(), position: ti.types.ndarray(), quanternion: ti.types.ndarray(), start_node: ti.types.ndarray(), 
                                  start_local_node: ti.types.ndarray(), masterID: ti.types.ndarray(), scale: ti.types.ndarray(), vertices: ti.template()):
    for iterate in range(surface_num):
        master = masterID[iterate]
        local_node = iterate - start_node[master] + start_local_node[master]
        rotation_matrix = SetToRotate(vec4f([quanternion[master, 0], quanternion[master, 1], quanternion[master, 2], quanternion[master, 3]]))
        vertices[iterate] = vec3f([position[master, 0], position[master, 1], position[master, 2]]) + rotation_matrix @ (scale[master] * vec3f([surface_node[local_node, 0], surface_node[local_node, 1], surface_node[local_node, 2]]))

@ti.kernel
def check_radius_(rigidNum: int, rigid: ti.template()) -> bool:
    check = True
    for i in range(1, rigidNum):
        if check and rigid[i-1].equi_r > rigid[i].equi_r:
            check = False
    return check

def GetConnectivity(surface_node_number, faces, i):
    return faces + i * surface_node_number