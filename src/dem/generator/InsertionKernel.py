import taichi as ti

from src.utils.constants import PI
from src.utils.Quaternion import ThetaToRotationMatrix, RandomGenerator, SetFromEuler, SetToRotate
from src.utils.TypeDefination import vec3f, vec2f, vec2i, vec4f, vec3u8, vec3i
from src.utils.ScalarFunction import vectorize_id, linearize3D, equal_to


@ti.func
def pebble_rotate(local_pebble_coords, target):
    R = ThetaToRotationMatrix(target)
    local_pebble_coords = R @ local_pebble_coords
    return local_pebble_coords

@ti.func
def get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble: ti.template(), rad_pebble: ti.template()):
    local_pebble_coords = vec3f(x_pebble[pebble, 0], x_pebble[pebble, 1], x_pebble[pebble, 2]) * scale_factor
    pebble_radius = rad_pebble[pebble] * scale_factor
    global_pebble_coords = pebble_rotate(local_pebble_coords, clump_orient) + com_pos
    return global_pebble_coords, pebble_radius

@ti.func
def validate_clump(start_point, com_pos, scale_factor, clump_orient, nspheres, x_pebble, rad_pebble, is_overlap, check_in_domain, insert_particle_in_neighbor):
    invalid = 0
    for pebble in range(nspheres):
        pcoord, pradius = get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble)
        if is_overlap(pcoord - start_point, pradius, insert_particle_in_neighbor) == 1 or not check_in_domain(pcoord, pradius):
            invalid = 1
            break
    return invalid

@ti.kernel
def kernel_create_sphere_(particle: ti.template(), sphere: ti.template(), material: ti.template(), bodyNum: int, particleNum: int, position: ti.types.vector(3, float), radius: float,
                          groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), fix_v: ti.types.vector(3, int), fix_w: ti.types.vector(3, int)):
    density = material[matID]._get_density()
    mass = density * 4./3. * PI * radius * radius * radius
    inertia = 2./5. * mass * radius * radius
    inv_inertia = 1. / inertia
    q = RandomGenerator()

    particle[particleNum]._add_index(-1-bodyNum)
    particle[particleNum]._add_particle_proporities(matID, groupID, radius, mass)
    particle[particleNum]._add_particle_kinematics(position, init_v, init_w)
    sphere[bodyNum]._add_index(bodyNum, particleNum)
    sphere[bodyNum]._add_sphere_attribute(init_w, inv_inertia, q, fix_v, fix_w)

@ti.kernel
def kernel_create_multisphere_(particle: ti.template(), clump: ti.template(), material: ti.template(), bodyNum: int, particleNum: int, nspheres: int, scale_factor: float, inertia: ti.types.vector(3, float),
                               x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray(), com_pos: ti.types.vector(3, float), equiv_rad: float, get_orientation: ti.template(), 
                               groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float)):
    nsphere = nspheres
    density = material[matID]._get_density()
    mass = density * 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
    inv_inertia = 1. / (inertia * density * scale_factor ** 5)
    orientation = get_orientation()
    q = SetFromEuler(*orientation)

    clump[bodyNum]._add_index(bodyNum, particleNum, particleNum + nsphere - 1)
    clump[bodyNum]._add_clump_kinematic(com_pos, init_v, init_w, inv_inertia)
    clump[bodyNum]._add_clump_attribute(mass, equiv_rad, inv_inertia, q)

    for pebble in range(nsphere):
        particleID = pebble + particleNum
        pebble_coord, pebble_radius = get_actual_clump(pebble, com_pos, scale_factor, orientation, x_pebble, rad_pebble) 
        pebble_mass = 4./3. * PI * pebble_radius * pebble_radius * pebble_radius * density
        particle[particleID]._add_index(bodyNum)
        particle[particleID]._add_particle_proporities(matID, groupID, pebble_radius, pebble_mass)
        particle[particleID]._add_particle_kinematics(pebble_coord, init_v + init_w.cross(pebble_coord - com_pos), init_w)

@ti.func
def create_bounding_sphere_(rigidNum, scale, com_pos, bounding_sphere, r_bound, x_bound):
    bounding_sphere[rigidNum]._add_bounding_sphere(x_bound + com_pos, r_bound)
    bounding_sphere[rigidNum]._scale(scale, com_pos)

@ti.func
def create_bounding_box(rigidNum, scale, bounding_box, minBox, maxBox, gridNum, space, gnum, extent):
    bounding_box[rigidNum]._set_bounding_box(minBox, maxBox)
    bounding_box[rigidNum]._scale(scale, vec3f(0, 0, 0))
    bounding_box[rigidNum]._add_grid(gridNum, scale * space, gnum, scale, extent)

@ti.func
def create_deformable_grids_(gridNum, gridSum, grid, scale, distance_fields: ti.types.ndarray()):
    for i in range(gridSum):
        grid[i + gridNum]._set_grid(distance_fields[i])
        grid[i + gridNum]._scale(scale)

@ti.func
def create_deformable_surface(rigidNum, surfaceNum, surfaceSum, surface_node, scale, surface_nodes: ti.types.ndarray(), parameters: ti.types.ndarray()):
    for i in range(surfaceSum):
        surface_node[i + surfaceNum]._set_master(rigidNum)
        surface_node[i + surfaceNum]._set_surface_node(vec3f(surface_nodes[i, 0], surface_nodes[i, 1], surface_nodes[i, 2]))
        surface_node[i + surfaceNum]._set_coefficient(parameters[i])
        surface_node[i + surfaceNum]._scale(scale, vec3f(0, 0, 0))

@ti.kernel
def kernel_create_level_set_rigid_body_(rigid_body: ti.template(), bounding_box: ti.template(), bounding_sphere: ti.template(), master: ti.template(), material: ti.template(), rigidNum: int, gridNum: int, verticeNum: int, surfaceNum: int, minBox: ti.types.vector(3, float), 
                                        maxBox: ti.types.vector(3, float), r_bound: float, x_bound: ti.types.vector(3, float), surfaceSum: int, space: float, gnum: ti.types.vector(3, int), extent: int, scale_factor: float, inertia: ti.types.vector(3, float), 
                                        com_pos: ti.types.vector(3, float), equiv_rad: float, get_orientation: ti.template(), groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), is_fix: ti.types.vector(3, int)):
    density = material[matID]._get_density()
    mass = 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
    inv_inertia = 1. / (inertia * scale_factor ** 5)
    orientation = get_orientation()
    q = SetFromEuler(*orientation)
    rotation_matrix = SetToRotate(q)
    
    create_bounding_sphere_(rigidNum, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound)
    create_bounding_box(rigidNum, scale_factor, bounding_box, minBox, maxBox, gridNum, space, gnum, extent)
    for i in range(surfaceSum):
        master[i + surfaceNum] = rigidNum

    rigid_body[rigidNum]._add_body_attribute(com_pos, mass, equiv_rad, inv_inertia, q)
    rigid_body[rigidNum]._add_surface_index(surfaceNum, surfaceNum + surfaceSum, verticeNum)
    rigid_body[rigidNum]._add_body_properties(matID, groupID, density)
    rigid_body[rigidNum]._add_body_kinematic(init_v, init_w, is_fix)

@ti.kernel
def kernel_create_implicit_surface_rigid_body_(rigid_body: ti.template(), bounding_sphere: ti.template(), material: ti.template(), rigidNum: int, templateID: int, r_bound: float, x_bound: ti.types.vector(3, float), scale_factor: float, inertia: ti.types.vector(3, float), 
                                        com_pos: ti.types.vector(3, float), equiv_rad: float, get_orientation: ti.template(), groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), is_fix: ti.types.vector(3, int)):
    density = material[matID]._get_density()
    mass = 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
    inv_inertia = 1. / (inertia * scale_factor ** 5)
    orientation = get_orientation()
    q = SetFromEuler(*orientation)
    rotation_matrix = SetToRotate(q)
    print(SetToRotate(q))
    
    create_bounding_sphere_(rigidNum, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound)
    rigid_body[rigidNum]._add_body_attribute(scale_factor, com_pos, mass, equiv_rad, inv_inertia, q)
    rigid_body[rigidNum]._add_body_properties(matID, groupID, templateID, density)
    rigid_body[rigidNum]._add_body_kinematic(init_v, init_w, is_fix)

@ti.kernel
def kernel_create_deformable_body_(rigid_body: ti.template(), surface_node: ti.template(), grid: ti.template(), bounding_box: ti.template(), bounding_sphere: ti.template(), material: ti.template(), rigidNum: int, gridNum: int, surfaceNum: int, minBox: ti.types.vector(3, float), maxBox: ti.types.vector(3, float), 
                                  r_bound: float, x_bound: ti.types.vector(3, float), surfaceSum: int, surface_nodes: ti.types.ndarray(), parameters: ti.types.ndarray(), gridSum: int, space: float, gnum: ti.types.vector(3, int), start_point: ti.types.vector(3, float), distance_fields: ti.types.ndarray(), 
                                  scale_factor: float, inertia: ti.types.vector(3, float), com_pos: ti.types.vector(3, float), equiv_rad: float, get_orientation: ti.template(), groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), is_fix: ti.types.vector(3, int)):
    create_bounding_sphere_(rigidNum, scale_factor, com_pos, bounding_sphere, r_bound, x_bound)
    create_bounding_box(rigidNum, scale_factor, bounding_box, minBox, maxBox, gridNum, space, gnum, start_point)
    create_deformable_grids_(gridNum, gridSum, grid, scale_factor, distance_fields)
    create_deformable_surface(rigidNum, surfaceNum, surfaceSum, surface_node, scale_factor, surface_nodes, parameters)
    bounding_box[rigidNum].scale = 1.

    density = material[matID]._get_density()
    mass = density * 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
    inv_inertia = 1. / (inertia * density * scale_factor ** 5)
    orientation = get_orientation()
    q = SetFromEuler(*orientation)

    rigid_body[rigidNum]._add_surface_index(surfaceNum, surfaceNum + surfaceSum)
    rigid_body[rigidNum]._add_body_properties(matID, groupID, mass, equiv_rad, inv_inertia, q)
    rigid_body[rigidNum]._add_body_kinematic(com_pos, init_v, init_w, inv_inertia, is_fix)

@ti.kernel
def kernel_get_orient(start_body_num: int, end_body_num: int, get_orientation: ti.template(), orients: ti.template()):
    for nb in range(end_body_num - start_body_num):
        orients[start_body_num + nb] = get_orientation()

@ti.kernel
def kernel_add_sphere_packing(particle: ti.template(), sphere: ti.template(), material: ti.template(), init_bodyNum: int, init_particleNum: int, start_body_num: int, end_body_num: int, coords: ti.template(), radii: ti.template(),
                              groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), fix_v: ti.types.vector(3, int), fix_w: ti.types.vector(3, int)):
    density = material[matID]._get_density()
    for nb in range(end_body_num - start_body_num):
        bodyNum = nb + init_bodyNum
        particleNum = nb * 1 + init_particleNum
        position, radius = coords[start_body_num + nb], radii[start_body_num + nb]
        mass = density * 4./3. * PI * radius * radius * radius
        inertia = 2./5. * mass * radius * radius
        inv_inertia = 1. / inertia
        q = RandomGenerator()

        particle[particleNum]._add_index(-1-bodyNum)
        particle[particleNum]._add_particle_proporities(matID, groupID, radius, mass)
        particle[particleNum]._add_particle_kinematics(position, init_v, init_w)
        sphere[bodyNum]._add_index(bodyNum, particleNum)
        sphere[bodyNum]._add_sphere_attribute(init_w, inv_inertia, q, fix_v, fix_w)
                
@ti.kernel
def kernel_add_sphere_files(particle: ti.template(), sphere: ti.template(), material: ti.template(), init_bodyNum: int, init_particleNum: int, body_num: int, coords: ti.types.ndarray(), radii: ti.types.ndarray(),
                            groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), fix_v: ti.types.vector(3, int), fix_w: ti.types.vector(3, int)):
    density = material[matID]._get_density()
    for nb in range(body_num):
        bodyNum = nb + init_bodyNum
        particleNum = nb * 1 + init_particleNum
        position, radius = vec3f([coords[nb, 0], coords[nb, 1], coords[nb, 2]]), radii[nb]
        mass = density * 4./3. * PI * radius * radius * radius
        inertia = 2./5. * mass * radius * radius
        inv_inertia = 1. / inertia
        q = RandomGenerator()

        particle[particleNum]._add_index(-1-bodyNum)
        particle[particleNum]._add_particle_proporities(matID, groupID, radius, mass)
        particle[particleNum]._add_particle_kinematics(position, init_v, init_w)
        sphere[bodyNum]._add_index(bodyNum, particleNum)
        sphere[bodyNum]._add_sphere_attribute(init_w, inv_inertia, q, fix_v, fix_w)

@ti.kernel
def kernel_add_multisphere_packing(particle: ti.template(), clump: ti.template(), material: ti.template(), init_bodyNum: int, init_particleNum: int, start_body_num: int, end_body_num: int, start_pebble_num: int, 
                                   template_ptr: ti.template(), clump_coords: ti.template(), clump_radii: ti.template(), clump_orients: ti.template(), pebble_coords: ti.template(), pebble_radii: ti.template(), 
                                   groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float)):
    density = material[matID]._get_density()
    for nb in range(end_body_num - start_body_num):
        nsphere = template_ptr.nspheres
        bodyNum = nb + init_bodyNum
        particleNum = nb * nsphere + init_particleNum
        clump_com, clump_rad, clump_orient = clump_coords[start_body_num + nb], clump_radii[start_body_num + nb], clump_orients[start_body_num + nb]
        scale_factor = clump_rad / template_ptr.r_equiv

        mass = density * 4./3. * PI * clump_rad * clump_rad * clump_rad
        inv_inertia = 1. / (scale_factor ** 5 * template_ptr.inertia * density)
        q = SetFromEuler(*clump_orient)

        clump[bodyNum]._add_index(bodyNum, particleNum, particleNum + nsphere - 1)
        clump[bodyNum]._add_clump_kinematic(clump_com, init_v, init_w, inv_inertia)
        clump[bodyNum]._add_clump_attribute(mass, clump_rad, inv_inertia, q)
    
        for pebble in range(nsphere):
            particleID = pebble + particleNum
            pebble_coord, pebble_radius = pebble_coords[start_pebble_num + nb * nsphere + pebble], pebble_radii[start_pebble_num + nb * nsphere + pebble]
            pebble_mass = 4/3. * PI * pebble_radius * pebble_radius * pebble_radius * density
            particle[particleID]._add_index(bodyNum)
            particle[particleID]._add_particle_proporities(matID, groupID, pebble_radius, pebble_mass)
            particle[particleID]._add_particle_kinematics(pebble_coord, init_v + init_w.cross(pebble_coord - clump_com), init_w)

@ti.kernel
def kernel_add_multisphere_files(particle: ti.template(), clump: ti.template(), material: ti.template(), init_bodyNum: int, init_particleNum: int, body_num: int, pebble_num: int, 
                                 clump_coords: ti.types.ndarray(), clump_radii: ti.types.ndarray(), clump_orients: ti.types.ndarray(), clump_inertia_vol: ti.types.ndarray(), startIndics: ti.types.ndarray(), endIndics: ti.types.ndarray(),
                                 pebble_coords: ti.types.ndarray(), pebble_radii: ti.types.ndarray(), multisphereIndics: ti.types.ndarray(), 
                                 groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float)):
    density = material[matID]._get_density()
    for nb in range(body_num):
        bodyNum = nb + init_bodyNum
        clump_com, clump_rad = vec3f([clump_coords[nb, 0], clump_coords[nb, 1], clump_coords[nb, 2]]), clump_radii[nb]
        clump_orient = vec3f([clump_orients[nb, 0], clump_orients[nb, 1], clump_orients[nb, 2]])
        clump_inv_inertia = 1. / (vec3f([clump_inertia_vol[nb, 0], clump_inertia_vol[nb, 1], clump_inertia_vol[nb, 2]]) * density)
        startIndex, endIndex = init_particleNum + int(startIndics[nb]), init_particleNum + int(endIndics[nb])

        mass = density * 4./3. * PI * clump_rad * clump_rad * clump_rad
        q = SetFromEuler(*clump_orient)

        clump[bodyNum]._add_index(bodyNum, startIndex, endIndex)
        clump[bodyNum]._add_clump_kinematic(clump_com, init_v, init_w, clump_inv_inertia)
        clump[bodyNum]._add_clump_attribute(mass, clump_rad, clump_inv_inertia, q)
    
    for pebble in range(pebble_num):
        particleID = pebble + init_particleNum
        pebble_coord = vec3f([pebble_coords[pebble, 0], pebble_coords[pebble, 1], pebble_coords[pebble, 2]])
        pebble_radius, multisphereIndex = pebble_radii[pebble], int(multisphereIndics[pebble])
        pebble_mass = 4/3. * PI * pebble_radius * pebble_radius * pebble_radius * density
        clump_com = vec3f([clump_coords[multisphereIndex, 0], clump_coords[multisphereIndex, 1], clump_coords[multisphereIndex, 2]])
        particle[particleID]._add_index(multisphereIndex)
        particle[particleID]._add_particle_proporities(matID, groupID, pebble_radius, pebble_mass)
        particle[particleID]._add_particle_kinematics(pebble_coord, init_v + init_w.cross(pebble_coord - clump_com), init_w)

@ti.kernel
def kernel_add_levelset_packing(rigid_body: ti.template(), bounding_box: ti.template(), bounding_sphere: ti.template(), master: ti.template(), material: ti.template(), rigidNum: int, gridNum: int, surfaceNum: int, verticeNum: int, minBox: ti.types.vector(3, float), 
                                maxBox: ti.types.vector(3, float), r_bound: float, x_bound: ti.types.vector(3, float), surfaceSum: int, space: float, gnum: ti.types.vector(3, int), extent: int, inertia: ti.types.vector(3, float), eqradius: float, 
                                groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), is_fix: ti.types.vector(3, int), start_body_num: int, end_body_num: int, coords: ti.template(), radii: ti.template(), orients: ti.template()):
    density = material[matID]._get_density()
    for nb in range(end_body_num - start_body_num):
        bounding_x, bounding_r = coords[start_body_num + nb], radii[start_body_num + nb] 
        scale_factor = bounding_r / r_bound
        orientation = orients[start_body_num + nb]
        q = SetFromEuler(*orientation)
        rotation_matrix = SetToRotate(q)
        com_pos, equiv_rad = bounding_x - scale_factor * rotation_matrix @ x_bound, scale_factor * eqradius
        mass = 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1. / (inertia * scale_factor ** 5)

        create_bounding_sphere_(rigidNum + nb, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound)
        create_bounding_box(rigidNum + nb, scale_factor, bounding_box, minBox, maxBox, gridNum, space, gnum, extent)
        for i in range(surfaceSum):
            master[i + surfaceNum + nb * surfaceSum] = rigidNum + nb

        rigid_body[rigidNum + nb]._add_body_attribute(com_pos, mass, equiv_rad, inv_inertia, q)
        rigid_body[rigidNum + nb]._add_surface_index(surfaceNum + nb * surfaceSum, surfaceNum + (nb + 1) * surfaceSum, verticeNum)
        rigid_body[rigidNum + nb]._add_body_properties(matID, groupID, density)
        rigid_body[rigidNum + nb]._add_body_kinematic(init_v, init_w, is_fix)

@ti.kernel
def kernel_add_implicit_surface_packing(rigid_body: ti.template(), bounding_sphere: ti.template(), material: ti.template(), rigidNum: int, templateID: int, r_bound: float, x_bound: ti.types.vector(3, float), inertia: ti.types.vector(3, float), eqradius: float, 
                                        groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), is_fix: ti.types.vector(3, int), start_body_num: int, end_body_num: int, coords: ti.template(), radii: ti.template(), orients: ti.template()):
    density = material[matID]._get_density()
    for nb in range(end_body_num - start_body_num):
        bounding_x, bounding_r = coords[start_body_num + nb], radii[start_body_num + nb] 
        scale_factor = bounding_r / r_bound
        orientation = orients[start_body_num + nb]
        q = SetFromEuler(*orientation)
        rotation_matrix = SetToRotate(q)
        com_pos, equiv_rad = bounding_x - scale_factor * rotation_matrix @ x_bound, scale_factor * eqradius
        mass = 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1. / (inertia * scale_factor ** 5)

        create_bounding_sphere_(rigidNum + nb, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound)
        rigid_body[rigidNum + nb]._add_body_attribute(scale_factor, com_pos, mass, equiv_rad, inv_inertia, q)
        rigid_body[rigidNum + nb]._add_body_properties(matID, groupID, templateID, density)
        rigid_body[rigidNum + nb]._add_body_kinematic(init_v, init_w, is_fix)

@ti.kernel
def kernel_add_levelset_files(rigid_body: ti.template(), bounding_box: ti.template(), bounding_sphere: ti.template(), master: ti.template(), rigidNum: int, gridNum: int, surfaceNum: int, 
                              r_bound: float, x_bound: ti.types.vector(3, float), minBox: ti.types.vector(3, float), maxBox: ti.types.vector(3, float), surfaceSum: int, inertia: ti.types.vector(3, float), eqradius: float, 
                              space: float, gnum: ti.types.vector(3, int), extent: int, body_num: int, coords: ti.types.ndarray(), radii: ti.types.ndarray(), orients: ti.types.ndarray()):
    for nb in range(body_num):
        bounding_x, bounding_r = vec3f([coords[nb, 0], coords[nb, 1], coords[nb, 2]]), radii[nb] 
        scale_factor = bounding_r / r_bound
        orientation = vec3f([orients[nb, 0], orients[nb, 1], orients[nb, 2]])
        q = SetFromEuler(*orientation)
        rotation_matrix = SetToRotate(q)
        com_pos, equiv_rad = bounding_x - scale_factor * rotation_matrix @ x_bound, scale_factor * eqradius
        mass = 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1. / (inertia * scale_factor ** 5)

        create_bounding_sphere_(rigidNum + nb, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound)
        create_bounding_box(rigidNum + nb, scale_factor, bounding_box, minBox, maxBox, gridNum, space, gnum, extent)
        for i in range(surfaceSum):
            master[i + surfaceNum + nb * surfaceSum] = rigidNum + nb

        rigid_body[rigidNum + nb]._add_body_attribute(com_pos, mass, equiv_rad, inv_inertia, q)

@ti.kernel
def kernel_add_rigid_body(rigid_body: ti.template(), material: ti.template(), rigidNum: int, surfaceNum: int, verticeNum: int, body_num: int, surfaceSum: int, groupID: int, matID: int, 
                          init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), is_fix: ti.types.vector(3, int)):
    density = material[matID]._get_density()
    for nb in range(body_num):
        rigid_body[rigidNum + nb]._add_surface_index(surfaceNum + nb * surfaceSum, surfaceNum + (nb + 1) * surfaceSum, verticeNum)
        rigid_body[rigidNum + nb]._add_body_properties(matID, groupID, density)
        rigid_body[rigidNum + nb]._add_body_kinematic(init_v, init_w, is_fix)

@ti.kernel
def kernel_add_implicit_surface_files(rigid_body: ti.template(), bounding_sphere: ti.template(), material: ti.template(), rigidNum: int, templateID: int, 
                              r_bound: float, x_bound: ti.types.vector(3, float), inertia: ti.types.vector(3, float), eqradius: float, 
                              body_num: int, coords: ti.types.ndarray(), radii: ti.types.ndarray(), orients: ti.types.ndarray(), groupID: int, matID: int, 
                              init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), is_fix: ti.types.vector(3, int)):
    density = material[matID]._get_density()
    for nb in range(body_num):
        bounding_x, bounding_r = vec3f([coords[nb, 0], coords[nb, 1], coords[nb, 2]]), radii[nb] 
        scale_factor = bounding_r / r_bound
        orientation = vec3f([orients[nb, 0], orients[nb, 1], orients[nb, 2]])
        q = SetFromEuler(*orientation)
        rotation_matrix = SetToRotate(q)
        com_pos, equiv_rad = bounding_x - scale_factor * rotation_matrix @ x_bound, scale_factor * eqradius
        mass = 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1. / (inertia * scale_factor ** 5)

        create_bounding_sphere_(rigidNum + nb, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound)
        rigid_body[rigidNum + nb]._add_body_attribute(scale_factor, com_pos, mass, equiv_rad, inv_inertia, q)
        rigid_body[rigidNum + nb]._add_body_properties(matID, groupID, templateID, density)
        rigid_body[rigidNum + nb]._add_body_kinematic(init_v, init_w, is_fix)

@ti.kernel
def kernel_add_deformable_packing(rigid_body: ti.template(), surface_node: ti.template(), grid: ti.template(), bounding_box: ti.template(), bounding_sphere: ti.template(), material: ti.template(), rigidNum: int, gridNum: int, surfaceNum: int, minBox: ti.types.vector(3, float), maxBox: ti.types.vector(3, float), 
                                r_bound: float, x_bound: ti.types.vector(3, float), surfaceSum: int, surface_nodes: ti.types.ndarray(), parameters: ti.types.ndarray(), gridSum: int, space: float, gnum: ti.types.vector(3, int), start_point: ti.types.vector(3, float), distance_fields: ti.types.ndarray(), 
                                inertia: ti.types.vector(3, float), eqradius: float, groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float), start_body_num: int, end_body_num: int, coords: ti.template(), radii: ti.template(), orients: ti.template(), is_fix: ti.types.vector(3, int)):
    density = material[matID]._get_density()
    for nb in range(end_body_num - start_body_num):
        bounding_x, bounding_r = coords[start_body_num + nb], radii[start_body_num + nb] 
        scale_factor = bounding_r / r_bound
        com_pos, equiv_rad = bounding_x - scale_factor * x_bound, scale_factor * eqradius
        create_bounding_sphere_(rigidNum + nb, scale_factor, com_pos, bounding_sphere, r_bound, x_bound)
        create_bounding_box(rigidNum + nb, scale_factor, bounding_box, minBox, maxBox, gridNum, space, gnum, start_point)
        create_deformable_grids_(gridNum + nb * gridSum, gridSum, grid, scale_factor, distance_fields)
        create_deformable_surface(rigidNum + nb, surfaceNum + nb * surfaceSum, surfaceSum, surface_node, scale_factor, surface_nodes, parameters)
        bounding_box[rigidNum].scale = 1.

        mass = density * 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1. / (inertia * density * scale_factor ** 5)
        orientation = orients[start_body_num + nb]
        q = SetFromEuler(*orientation)

        rigid_body[rigidNum + nb]._add_surface_index(surfaceNum + nb * surfaceSum, surfaceNum + (nb + 1) * surfaceSum)
        rigid_body[rigidNum + nb]._add_body_properties(matID, groupID, mass, equiv_rad, inv_inertia, q)
        rigid_body[rigidNum + nb]._add_body_kinematic(com_pos, init_v, init_w, inv_inertia, is_fix)

@ti.kernel
def kernel_position_rotate_(target: ti.types.vector(3, float), offset: ti.types.vector(3, float), body_coords: ti.template(), start_body_num: int, end_body_num: int):
    R = ThetaToRotationMatrix(target)
    for nb in range(start_body_num, end_body_num):
        coords = body_coords[nb]
        coords -= offset
        coords = R @ coords
        coords += offset
        body_coords[nb] = coords

@ti.kernel
def kernel_update_particle_number_by_sphere_(bodyNum: int, sphere: ti.template(), particle: ti.template(), check_in_region: ti.template()) -> ti.types.vector(2, int):
    inserted_body = 0
    inserted_particle = 0
    for nb in range(bodyNum):
        index = sphere[nb].sphereIndex
        position = particle[index].x
        radius = particle[index].rad
        if check_in_region(position, radius):
            inserted_body += 1
            inserted_particle += 1
    return vec2i([inserted_body, inserted_particle])

@ti.kernel
def kernel_update_particle_volume_by_sphere_(bodyNum: int, sphere: ti.template(), particle: ti.template(), check_in_region: ti.template()) -> ti.types.vector(2, float):
    inserted_volume = 0.
    inserted_particle = 0
    for nb in range(bodyNum):
        index = sphere[nb].sphereIndex
        position = particle[index].x
        radius = particle[index].rad
        if check_in_region(position, radius):
            inserted_volume += particle[index]._get_volume()
            inserted_particle += 1
    return vec2f([inserted_volume, float(inserted_particle)])

@ti.kernel
def kernel_update_particle_number_by_levelset_(rigidNum: int, bounding_sphere: ti.template(), check_in_region: ti.template()) -> ti.types.vector(2, int):
    inserted_body = 0
    inserted_particle = 0
    for index in range(rigidNum):
        position = bounding_sphere[index].x
        radius = bounding_sphere[index].rad
        if check_in_region(position, radius):
            inserted_body += 1
            inserted_particle += 1
    return vec2i([inserted_body, inserted_particle])

@ti.kernel
def kernel_update_particle_volume_by_levelset_(rigidNum: int, bounding_sphere: ti.template(), rigid: ti.template(), check_in_region: ti.template()) -> ti.types.vector(2, float):
    inserted_volume = 0.
    inserted_particle = 0
    for index in range(rigidNum):
        position = bounding_sphere[index].x
        radius = bounding_sphere[index].rad
        if check_in_region(position, radius):
            inserted_volume += rigid[index]._get_volume()
            inserted_particle += 1
    return vec2f([inserted_volume, inserted_particle])

@ti.kernel
def kernel_update_pebble_number_by_clump_(bodyNum: int, clump: ti.template(), particle: ti.template(), check_in_region: ti.template()) -> ti.types.vector(2, int):
    inserted_body = 0
    inserted_particle = 0
    for nb in range(bodyNum):
        start, end = clump[nb].startIndex, clump[nb].endIndex
        is_in_region = 1
        for npebble in range(start, end):
            position = particle[npebble].x
            radius = particle[npebble].rad
            if not check_in_region(position, radius):
                is_in_region = 0
                break
        if is_in_region:
            nsphere = end - start + 1
            inserted_body += 1
            inserted_particle += nsphere
    return vec2i([inserted_body, inserted_particle])

@ti.kernel
def kernel_update_particle_volume_by_clump_(bodyNum: int, clump: ti.template(), particle: ti.template(), check_in_region: ti.template()) -> ti.types.vector(2, float):
    inserted_volume = 0.
    inserted_particle = 0
    for nb in range(bodyNum):
        start, end = clump[nb].startIndex, clump[nb].endIndex
        is_in_region = 1
        for npebble in range(start, end):
            position = particle[npebble].x
            radius = particle[npebble].rad
            if not check_in_region(position, radius):
                is_in_region = 0
                break
        if is_in_region:
            nsphere = end - start + 1
            inserted_volume += clump[nb]._get_volume()
            inserted_particle += nsphere
    return vec2f([inserted_volume, float(inserted_particle)])

@ti.kernel
def kernel_insert_first_sphere_(start_point: ti.types.vector(3, float), position: ti.types.vector(3, float), radius: float, insert_body_num: ti.template(), 
                                insert_particle_in_neighbor: ti.template(), sphere_coords: ti.template(), sphere_radii: ti.template(), cell_num: ti.types.vector(3, int), cell_size: float, 
                                neighbor_position: ti.template(), neighbor_radius: ti.template(), num_particle_in_cell: ti.template(), particle_neighbor: ti.template(), insert_particle: ti.template()):
    sphere_coords[insert_body_num[None]] = position
    sphere_radii[insert_body_num[None]] = radius
    insert_particle(cell_num, cell_size, position - start_point, radius, insert_particle_in_neighbor, neighbor_position, neighbor_radius, num_particle_in_cell, particle_neighbor)
    insert_body_num[None] += 1

@ti.kernel
def kernel_sphere_poisson_sampling_(min_rad: float, max_rad: float, tries_default: int, expected_body_num: int, start_point: ti.types.vector(3, float), insert_body_num: ti.template(), insert_particle_in_neighbor: ti.template(), 
                                    sphere_coords: ti.template(), sphere_radii: ti.template(), cell_num: ti.types.vector(3, int), cell_size: float, position: ti.template(), radius: ti.template(), num_particle_in_cell: ti.template(), 
                                    particle_neighbor: ti.template(), check_in_domain: ti.template(), overlap: ti.template(), insert_particle: ti.template()):
    tries = 0
    while tries < insert_body_num[None] and tries < expected_body_num:
        source_x, source_rad = sphere_coords[tries], sphere_radii[tries]
        for _ in range(tries_default):
            sphere_radius = min_rad + ti.random() * (max_rad - min_rad)
            u, v = ti.random(), ti.random()
            theta, phi = 2 * PI * u, ti.acos(2 * v - 1)
            randvector = vec3f([ti.sin(theta) * ti.sin(phi), ti.cos(theta) * ti.sin(phi), ti.cos(phi)]).normalized()
            offset = randvector * ((1 + ti.random()) * sphere_radius + source_rad)
            sphere_coord = source_x + offset
    
            if check_in_domain(sphere_coord, sphere_radius) and insert_body_num[None] < expected_body_num and \
               overlap(cell_num, cell_size, sphere_coord - start_point, sphere_radius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor) == 0:   
                sphere_coords[insert_body_num[None]] = sphere_coord
                sphere_radii[insert_body_num[None]] = sphere_radius
                insert_particle(cell_num, cell_size, sphere_coord - start_point, sphere_radius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor)
                insert_body_num[None] += 1
        tries += 1

@ti.kernel                
def kernel_sphere_generate_without_overlap_(min_rad: float, max_rad: float, tries_default: int, expected_body_num: int, start_point: ti.types.vector(3, float), region_size: ti.types.vector(3, float),
                                            insert_body_num: ti.template(), insert_particle_in_neighbor: ti.template(),  sphere_coords: ti.template(), sphere_radii: ti.template(), 
                                            cell_num: ti.types.vector(3, int), cell_size: float, position: ti.template(), radius: ti.template(), num_particle_in_cell: ti.template(), 
                                            particle_neighbor: ti.template(), check_in_domain: ti.template(), overlap: ti.template(), insert_particle: ti.template()):
    while insert_body_num[None] < expected_body_num:
        count = 0
        for _ in range(tries_default):
            sphere_radius = min_rad + ti.random() * (max_rad - min_rad)
            offset = vec3f([ti.random(), ti.random(), ti.random()]) * region_size
            sphere_coord = start_point + offset 
            
            if check_in_domain(sphere_coord, sphere_radius) and \
               overlap(cell_num, cell_size, sphere_coord - start_point, sphere_radius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor) == 0: 
                sphere_coords[insert_body_num[None]] = sphere_coord
                sphere_radii[insert_body_num[None]] = sphere_radius
                insert_particle(cell_num, cell_size, sphere_coord - start_point, sphere_radius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor)
                insert_body_num[None] += 1
                break
            count += 1
        if count == tries_default:
            break

@ti.kernel                
def kernel_sphere_generate_lattice_(min_rad: float, max_rad: float, expected_body_num: int, position_distribution: ti.types.vector(3, int), start_point: ti.types.vector(3, float), valid: ti.template(),
                                    insert_body_num: ti.template(), insert_particle_in_neighbor: ti.template(),  sphere_coords: ti.template(), sphere_radii: ti.template(), 
                                    cell_num: ti.types.vector(3, int), cell_size: float, position: ti.template(), radius: ti.template(), num_particle_in_cell: ti.template(), 
                                    particle_neighbor: ti.template(), check_in_domain: ti.template(), overlap: ti.template(), insert_particle: ti.template()):
    tries = insert_body_num[None]
    while tries < expected_body_num:
        sphere_radius = min_rad + ti.random() * (max_rad - min_rad)
        randomID = expected_body_num - insert_body_num[None] - 1
        offset = vec3f([ti.random(), ti.random(), ti.random()]) * (max_rad - sphere_radius)
        sphere_coord = start_point + (vec3f(vectorize_id(valid[randomID], position_distribution)) + 0.5) * 2. * max_rad + offset
        if check_in_domain(sphere_coord, sphere_radius):
            if overlap(cell_num, cell_size, sphere_coord - start_point, sphere_radius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor) == 0: 
                sphere_coords[insert_body_num[None]] = sphere_coord
                sphere_radii[insert_body_num[None]] = sphere_radius
                insert_particle(cell_num, cell_size, sphere_coord - start_point, sphere_radius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor)
                insert_body_num[None] += 1
        valid[randomID] = -1
        tries += 1

@ti.kernel
def update_valid(number: int, valid: ti.template()) -> int:
    remains = 0
    ti.loop_config(serialize=True)
    for i in range(number):
        if valid[i] != -1:
            valid[remains] = valid[i]
            remains += 1
    return remains

@ti.kernel
def fill_valid(valid: ti.template()):
    for i in valid:
        valid[i] = i

@ti.kernel
def kernel_distribute_sphere_(min_rad: float, max_rad: float, volume_expect: float, insert_body_num: ti.template(), insert_particle_in_neighbor: ti.template(), 
                              sphere_coords: ti.template(), sphere_radii: ti.template(), start_point: ti.types.vector(3, float), region_size: ti.types.vector(3, float), check_in_domain: ti.template()) -> float:
    inserted_volume = 0.
    while inserted_volume < volume_expect:
        sphere_radius = min_rad + ti.random() * (max_rad - min_rad)
        offset = vec3f([ti.random(), ti.random(), ti.random()]) * region_size
        sphere_coord = start_point + offset 
        
        if check_in_domain(sphere_coord, sphere_radius):        
            pvol = 4./3. * PI * sphere_radius * sphere_radius * sphere_radius
            sphere_coords[insert_body_num[None]] = sphere_coord
            sphere_radii[insert_body_num[None]] = sphere_radius
            inserted_volume += pvol
            insert_body_num[None] += 1
            insert_particle_in_neighbor[None] += 1
    return inserted_volume

@ti.kernel
def kernel_insert_first_multisphere_(start_point: ti.types.vector(3, float), nspheres: int, r_equiv: float, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray(), 
                                     com_pos: ti.types.vector(3, float), equiv_rad: float, insert_body_num: ti.template(), insert_particle_in_neighbor: ti.template(), 
                                     clump_coords: ti.template(), clump_radii: ti.template(), clump_orients: ti.template(), get_orientation: ti.template(),
                                     pebble_coords: ti.template(), pebble_radii: ti.template(), cell_num: ti.types.vector(3, int), cell_size: float, neighbor_position: ti.template(), 
                                     neighbor_radius: ti.template(), num_particle_in_cell: ti.template(), particle_neighbor: ti.template(), insert_particle: ti.template()):
    scale_factor = equiv_rad / r_equiv
    clump_orient = get_orientation()
    iclump = insert_body_num[None]
    clump_coords[iclump] = com_pos
    clump_radii[iclump] = equiv_rad
    clump_orients[iclump] = clump_orient
    
    for pebble in range(nspheres):
        ipebble = insert_particle_in_neighbor[None] 
        pebble_coord, pebble_radius = get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble) 
        pebble_coords[ipebble] = pebble_coord
        pebble_radii[ipebble] = pebble_radius
        insert_particle(cell_num, cell_size, pebble_coord - start_point, pebble_radius, insert_particle_in_neighbor, neighbor_position, neighbor_radius, num_particle_in_cell, particle_neighbor)
    insert_body_num[None] += 1

@ti.kernel
def kernel_multisphere_poisson_sampling_(nspheres: int, r_equiv: float, r_bound: float, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray(), min_rad: float, max_rad: float, tries_default: int, expected_body_num: int, 
                                         start_point: ti.types.vector(3, float), insert_body_num: ti.template(), insert_particle_in_neighbor: ti.template(), clump_coords: ti.template(), clump_radii: ti.template(), 
                                         clump_orients: ti.template(), get_orientation: ti.template(), pebble_coords: ti.template(), pebble_radii: ti.template(), cell_num: ti.types.vector(3, int), cell_size: float, position: ti.template(), 
                                         radius: ti.template(), num_particle_in_cell: ti.template(), particle_neighbor: ti.template(), check_in_domain: ti.template(), overlap: ti.template(), insert_particle: ti.template()):
    tries = 0
    while tries < insert_body_num[None] and tries < expected_body_num:
        source_x, source_rad = clump_coords[tries], clump_radii[tries]
        for _ in range(tries_default):
            nsphere = nspheres
            equiv_rad = min_rad + ti.random() * (max_rad - min_rad)
            scale_factor = equiv_rad / r_equiv

            bound_rad = scale_factor * r_bound
            u, v = ti.random(), ti.random()
            theta, phi = 2 * PI * u, ti.acos(2 * v - 1)
            randvector = vec3f([ti.sin(theta) * ti.sin(phi), ti.cos(theta) * ti.sin(phi), ti.cos(phi)]).normalized()
            offset = randvector * ((1 + ti.random()) * bound_rad + source_rad)
            com_pos = source_x + offset
            clump_orient = get_orientation()
            invalid = 0
            for pebble in range(nspheres):
                pcoord, pradius = get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble)
                if overlap(cell_num, cell_size, pcoord - start_point, pradius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor) == 1 or not check_in_domain(pcoord, pradius):
                    invalid = 1
                    break
            #invalid = validate_clump(start_point, com_pos, scale_factor, clump_orient, nspheres, x_pebble, rad_pebble, is_overlap, check_in_domain, insert_particle_in_neighbor)

            if not invalid and insert_body_num[None] < expected_body_num: 
                iclump = insert_body_num[None]
                clump_coords[iclump] = com_pos
                clump_radii[iclump] = equiv_rad
                clump_orients[iclump] = clump_orient
                for pebble in range(nsphere):
                    ipebble = insert_particle_in_neighbor[None] 
                    pebble_coord, pebble_radius = get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble)
                    pebble_coords[ipebble] = pebble_coord
                    pebble_radii[ipebble] = pebble_radius                      
                    insert_particle(cell_num, cell_size, pebble_coord - start_point, pebble_radius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor)
                insert_body_num[None] += 1
        tries += 1

@ti.kernel
def kernel_multisphere_generate_without_overlap_(nspheres: int, r_equiv: float, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray(), min_rad: float, max_rad: float, tries_default: int, expected_body_num: int, 
                                                 insert_body_num: ti.template(), insert_particle_in_neighbor: ti.template(), clump_coords: ti.template(), clump_radii: ti.template(), 
                                                 clump_orients: ti.template(), get_orientation: ti.template(), pebble_coords: ti.template(), pebble_radii: ti.template(), cell_num: ti.types.vector(3, int), cell_size: float, 
                                                 position: ti.template(), radius: ti.template(), num_particle_in_cell: ti.template(), particle_neighbor: ti.template(), check_in_domain: ti.template(), 
                                                 start_point: ti.types.vector(3, float), region_size: ti.types.vector(3, float), overlap: ti.template(), insert_particle: ti.template()):
    while insert_body_num[None] < expected_body_num:
        count = 0
        for _ in range(tries_default):
            equiv_rad = min_rad + ti.random() * (max_rad - min_rad)
            scale_factor = equiv_rad / r_equiv
            
            offset = vec3f([ti.random(), ti.random(), ti.random()]) * region_size
            com_pos = start_point + offset 
            clump_orient = get_orientation()
            invalid = 0
            for pebble in range(nspheres):
                pcoord, pradius = get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble)
                if overlap(cell_num, cell_size, pcoord - start_point, pradius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor) == 1 or not check_in_domain(pcoord, pradius):
                    invalid = 1
                    break
            #invalid = validate_clump(start_point, com_pos, scale_factor, clump_orient, nspheres, x_pebble, rad_pebble, is_overlap, check_in_domain, insert_particle_in_neighbor)

            if invalid == 0: 
                iclump = insert_body_num[None]
                clump_coords[iclump] = com_pos
                clump_radii[iclump] = equiv_rad
                clump_orients[iclump] = clump_orient
                for pebble in range(nspheres):
                    ipebble = insert_particle_in_neighbor[None] 
                    pebble_coord, pebble_radius = get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble)
                    pebble_coords[ipebble] = pebble_coord
                    pebble_radii[ipebble] = pebble_radius  
                    insert_particle(cell_num, cell_size, pebble_coord - start_point, pebble_radius, insert_particle_in_neighbor, position, radius, num_particle_in_cell, particle_neighbor)
                insert_body_num[None] += 1
                break
            count += 1
        if count == tries_default:
            break

@ti.kernel
def kernel_distribute_multisphere_(nspheres: int, r_equiv: float, volume_expect: float, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray(), min_rad: float, max_rad: float, 
                                   expected_particle_volume: float, insert_body_num: ti.template(), insert_particle_in_neighbor: ti.template(), 
                                   clump_coords: ti.template(), clump_radii: ti.template(), clump_orients: ti.template(), get_orientation: ti.template(), pebble_coords: ti.template(), pebble_radii: ti.template(), 
                                   start_point: ti.types.vector(3, float), region_size: ti.types.vector(3, float), check_in_domain: ti.template()) -> float:
    inserted_volume = 0.
    while inserted_volume < expected_particle_volume:
        equiv_rad = min_rad + ti.random() * (max_rad - min_rad)
        scale_factor = equiv_rad / r_equiv

        offset = vec3f([ti.random(), ti.random(), ti.random()]) * region_size
        com_pos = start_point + offset 
        clump_orient = get_orientation()

        invalid = 0
        for pebble in range(nspheres):
            pcoord, pradius = get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble)
            if not check_in_domain(pcoord, pradius):
                invalid = 1
                break

        if not invalid: 
            iclump = insert_body_num[None]
            clump_coords[iclump] = com_pos
            clump_radii[iclump] = equiv_rad
            clump_orients[iclump] = clump_orient
            for pebble in range(nspheres):
                ipebble = insert_particle_in_neighbor[None] 
                pebble_coord, pebble_radius = get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble)
                pebble_coords[ipebble + pebble] = pebble_coord
                pebble_radii[ipebble + pebble] = pebble_radius
            pvol = scale_factor * scale_factor * scale_factor * volume_expect
            inserted_volume += pvol
            insert_particle_in_neighbor[None] += nspheres
            insert_body_num[None] += 1
    return inserted_volume


@ti.kernel
def kernel_rebulid_levelset_body(start: int, number: int, rigid: ti.template(), groupID: ti.types.ndarray(), materialID: ti.types.ndarray(), startNode: ti.types.ndarray(), endNode: ti.types.ndarray(), localNode: ti.types.ndarray(), 
                            mass: ti.types.ndarray(), radius: ti.types.ndarray(), mass_center: ti.types.ndarray(), acceleration: ti.types.ndarray(), angular_moment: ti.types.ndarray(), 
                            velocity: ti.types.ndarray(), omega: ti.types.ndarray(), quanternion: ti.types.ndarray(), inverse_inertia: ti.types.ndarray(), is_fix: ti.types.ndarray()):
    for sp in range(start, start + number):
        np = sp - start
        rigid[sp]._restart(groupID[np], materialID[np], startNode[np], endNode[np], localNode[np], mass[np], radius[np], 
                           vec3f(mass_center[np, 0], mass_center[np, 1], mass_center[np, 2]), vec3f(acceleration[np, 0], acceleration[np, 1], acceleration[np, 2]), vec3f(angular_moment[np, 0], angular_moment[np, 1], angular_moment[np, 2]),
                           vec3f(velocity[np, 0], velocity[np, 1], velocity[np, 2]), vec3f(omega[np, 0], omega[np, 1], omega[np, 2]), vec4f(quanternion[np, 0], quanternion[np, 1], quanternion[np, 2], quanternion[np, 3]), 
                           vec3f(inverse_inertia[np, 0], inverse_inertia[np, 1], inverse_inertia[np, 2]), vec3u8(is_fix[np, 0], is_fix[np, 1], is_fix[np, 2]))


@ti.kernel
def kernel_rebulid_implicit_surface_body(start: int, number: int, rigid: ti.template(), groupID: ti.types.ndarray(), materialID: ti.types.ndarray(), templateID: ti.types.ndarray(), scale: ti.types.ndarray(), 
                            mass: ti.types.ndarray(), radius: ti.types.ndarray(), mass_center: ti.types.ndarray(), acceleration: ti.types.ndarray(), angular_moment: ti.types.ndarray(), 
                            velocity: ti.types.ndarray(), omega: ti.types.ndarray(), quanternion: ti.types.ndarray(), inverse_inertia: ti.types.ndarray(), is_fix: ti.types.ndarray()):
    for sp in range(start, start + number):
        np = sp - start
        rigid[sp]._restart(groupID[np], materialID[np], templateID[np], scale[np], mass[np], radius[np], 
                           vec3f(mass_center[np, 0], mass_center[np, 1], mass_center[np, 2]), vec3f(acceleration[np, 0], acceleration[np, 1], acceleration[np, 2]), vec3f(angular_moment[np, 0], angular_moment[np, 1], angular_moment[np, 2]),
                           vec3f(velocity[np, 0], velocity[np, 1], velocity[np, 2]), vec3f(omega[np, 0], omega[np, 1], omega[np, 2]), vec4f(quanternion[np, 0], quanternion[np, 1], quanternion[np, 2], quanternion[np, 3]), 
                           vec3f(inverse_inertia[np, 0], inverse_inertia[np, 1], inverse_inertia[np, 2]), vec3u8(is_fix[np, 0], is_fix[np, 1], is_fix[np, 2]))


@ti.kernel
def kernel_rebulid_levelset_grid(start: int, number: int, grid: ti.template(), distance_field: ti.types.ndarray()):
    for sp in range(start, start + number):
        np = sp - start
        grid[sp]._set_grid(distance_field[np])


@ti.kernel
def kernel_rebulid_bounding_sphere(start: int, number: int, particle: ti.template(), active: ti.types.ndarray(), radius: ti.types.ndarray(), center: ti.types.ndarray()):
    for sp in range(start, start + number):
        np = sp - start
        particle[sp]._restart(active[np], vec3f(center[np, 0], center[np, 1], center[np, 2]), radius[np])


@ti.kernel
def kernel_rebulid_bounding_box(start: int, number: int, box: ti.template(), min_box: ti.types.ndarray(), max_box: ti.types.ndarray(), startGrid: ti.types.ndarray(), grid_num: ti.types.ndarray(),
                                grid_space: ti.types.ndarray(), scale: ti.types.ndarray(), extent: ti.types.ndarray()):
    for sp in range(start, start + number):
        np = sp - start
        box[sp]._restart(vec3i(grid_num[np, 0], grid_num[np, 1], grid_num[np, 2]), vec3f(min_box[np, 0], min_box[np, 1], min_box[np, 2]), vec3f(max_box[np, 0], max_box[np, 1], max_box[np, 2]), 
                         startGrid[np], grid_space[np], scale[np], extent[np])


@ti.kernel
def kernel_rebulid_surface_node(start: int, number: int, vertice: ti.template(), vertices: ti.types.ndarray(), parameters: ti.types.ndarray()):
    for sp in range(start, start + number):
        np = sp - start
        vertice[sp]._restart(parameters[np], vec3f(vertices[np, 0], vertices[np, 1], vertices[np, 2]))


@ti.kernel
def kernel_rebulid_particle(start: int, number: int, particle: ti.template(), active: ti.types.ndarray(), multisphereIndex: ti.types.ndarray(), groupID: ti.types.ndarray(), 
                            materialID: ti.types.ndarray(), mass: ti.types.ndarray(), radius: ti.types.ndarray(), position: ti.types.ndarray(), 
                            velocity: ti.types.ndarray(), omega: ti.types.ndarray()):
    for sp in range(start, start + number):
        np = sp - start
        particle[sp]._restart(active[np], multisphereIndex[np], groupID[np], materialID[np], mass[np], radius[np], vec3f(position[np, 0], position[np, 1], position[np, 2]), 
                              vec3f(velocity[np, 0], velocity[np, 1], velocity[np, 2]), vec3f(omega[np, 0], omega[np, 1], omega[np, 2]))


@ti.kernel
def kernel_rebuild_sphere(start: int, number: int, sphere: ti.template(), sphereIndex: ti.types.ndarray(), inv_I: ti.types.ndarray(), q: ti.types.ndarray(), 
                          a: ti.types.ndarray(), angmoment: ti.types.ndarray(), fix_v: ti.types.ndarray(), fix_w: ti.types.ndarray()):
    for ssphere in range(start, start + number):
        nsphere = ssphere - start
        sphere[ssphere]._restart(sphereIndex[nsphere], inv_I[nsphere], vec4f(q[nsphere, 0], q[nsphere, 1], q[nsphere, 2], q[nsphere, 3]), vec3f(a[nsphere, 0], a[nsphere, 1], a[nsphere, 2]),
                                 vec3f(angmoment[nsphere, 0], angmoment[nsphere, 1], angmoment[nsphere, 2]), vec3u8(fix_v[nsphere, 0], fix_v[nsphere, 1], fix_v[nsphere, 2]), vec3u8(fix_w[nsphere, 0], fix_w[nsphere, 1], fix_w[nsphere, 2]))


@ti.kernel
def kernel_rebuild_clump(start: int, number: int, clump: ti.template(), startIndex: ti.types.ndarray(), endIndex: ti.types.ndarray(), mass: ti.types.ndarray(), equi_r: ti.types.ndarray(), mass_center: ti.types.ndarray(), 
                         v: ti.types.ndarray(), w: ti.types.ndarray(), a: ti.types.ndarray(), angmoment: ti.types.ndarray(), q: ti.types.ndarray(), inv_I: ti.types.ndarray()):
    for sclump in range(start, start + number):
        nclump = sclump - start
        clump[sclump]._restart(startIndex[nclump], endIndex[nclump], mass[nclump], equi_r[nclump], vec3f(mass_center[nclump, 0], mass_center[nclump, 1], mass_center[nclump, 2]),
                               vec3f(v[nclump, 0], v[nclump, 1], v[nclump, 2]), vec3f(w[nclump, 0], w[nclump, 1], w[nclump, 2]), vec3f(a[nclump, 0], a[nclump, 1], a[nclump, 2]),
                               vec3f(angmoment[nclump, 0], angmoment[nclump, 1], angmoment[nclump, 2]), vec4f(q[nclump, 0], q[nclump, 1], q[nclump, 2], q[nclump, 3]), vec3f(inv_I[nclump, 0], inv_I[nclump, 1], inv_I[nclump, 2]))


@ti.kernel
def kernel_rebuild_plane(start: int, number: int, wall: ti.template(), active: ti.types.ndarray(), wallID: ti.types.ndarray(), materialID: ti.types.ndarray(), point: ti.types.ndarray(), norm: ti.types.ndarray()):
    for swall in range(start, start + number):
        nwall = swall - start
        wall[swall]._restart(active[nwall], wallID[nwall], materialID[nwall], vec3f(point[nwall, 0], point[nwall, 1], point[nwall, 2]), vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2]))


@ti.kernel
def kernel_rebuild_facet(start: int, number: int, wall: ti.template(), active: ti.types.ndarray(), wallID: ti.types.ndarray(), materialID: ti.types.ndarray(), point1: ti.types.ndarray(), 
                              point2: ti.types.ndarray(), point3: ti.types.ndarray(), norm: ti.types.ndarray(), velocity: ti.types.ndarray()):
    for swall in range(start, start + number):
        nwall = swall - start
        wall[swall]._restart(active[nwall], wallID[nwall], materialID[nwall], vec3f(point1[nwall, 0], point1[nwall, 1], point1[nwall, 2]), vec3f(point2[nwall, 0], point2[nwall, 1], point2[nwall, 2]), 
                             vec3f(point3[nwall, 0], point3[nwall, 1], point3[nwall, 2]), vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2]), vec3f(velocity[nwall, 0], velocity[nwall, 1], velocity[nwall, 2]))


@ti.kernel
def kernel_rebuild_patch(start: int, number: int, wall: ti.template(), active: ti.types.ndarray(), wallID: ti.types.ndarray(), materialID: ti.types.ndarray(), point1: ti.types.ndarray(), 
                              point2: ti.types.ndarray(), point3: ti.types.ndarray(), norm: ti.types.ndarray()):
    for swall in range(start, start + number):
        nwall = swall - start
        wall[swall]._restart(active[nwall], wallID[nwall], materialID[nwall], vec3f(point1[nwall, 0], point1[nwall, 1], point1[nwall, 2]), vec3f(point2[nwall, 0], point2[nwall, 1], point2[nwall, 2]), 
                             vec3f(point3[nwall, 0], point3[nwall, 1], point3[nwall, 2]), vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2]))


@ti.kernel
def kernel_rebuild_servo(start: int, number: int, servo: ti.template(), active: ti.types.ndarray(), startIndex: ti.types.ndarray(), endIndex: ti.types.ndarray(), alpha: ti.types.ndarray(), 
                         target_stress: ti.types.ndarray(), max_velocity: ti.types.ndarray()):
    for sservo in range(start, start + number):
        nservo = sservo - start
        servo[sservo]._restart(active[nservo], startIndex[nservo], endIndex[nservo], alpha[nservo], target_stress[nservo], max_velocity[nservo])


@ti.kernel
def generate_sphere_from_file(min_rad: float, max_rad: float, groupID: int, matID: int, voxelized_points_np: ti.types.ndarray(), init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float)):
    for voxelize in voxelized_points_np:
        radius = min_rad + ti.random() * (max_rad - min_rad)
        new_pos = voxelized_points_np[voxelize]
        

@ti.kernel
def kernel_add_facet(start: int, wallID: int, matID: int, vertices: ti.types.ndarray(), faces: ti.types.ndarray(), init_v: ti.types.vector(3, float), wall: ti.template()):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2]) 
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2]) 
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2]) 

        wall[swall].add_materialID(matID)
        wall[swall].add_wall_geometry_(wallID, vertice1, vertice2, vertice3, init_v)


@ti.kernel
def kernel_add_facet_files(start: int, wallID: int, matID: int, vertices: ti.types.ndarray(), faces: ti.types.ndarray(), norm: ti.types.ndarray(), init_v: ti.types.vector(3, float), wall: ti.template()):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2]) 
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2]) 
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2]) 

        wall[swall].add_materialID(matID)
        wall[swall].add_wall_geometry_(wallID, vertice1, vertice2, vertice3, init_v)
        wall[nwall].norm = vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2]) 

@ti.kernel
def kernel_add_facet_files_autonorm(iscounterclockwise: int, start: int, wallID: int, matID: int, vertices: ti.types.ndarray(), faces: ti.types.ndarray(), init_v: ti.types.vector(3, float), wall: ti.template()):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2]) 
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2]) 
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2]) 

        if iscounterclockwise == 0:
            vertice1, vertice3 = vertice3, vertice1

        wall[swall].add_materialID(matID)
        wall[swall].add_wall_geometry_(wallID, vertice1, vertice2, vertice3, init_v)

@ti.kernel
def kernel_add_patch(start: int, wallID: int, matID: int, vertices: ti.types.ndarray(), faces: ti.types.ndarray(), norm: ti.types.ndarray(), wall: ti.template()):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2]) 
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2]) 
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2]) 
        normal = vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2]) 

        wall[swall].add_materialID(matID)
        wall[swall].add_wall_geometry(wallID, vertice1, vertice2, vertice3)
        wall[swall].norm = normal

@ti.kernel
def kernel_add_patch_autonorm(iscounterclockwise: int, start: int, wallID: int, matID: int, vertices: ti.types.ndarray(), faces: ti.types.ndarray(), wall: ti.template()):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2]) 
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2]) 
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2]) 

        if iscounterclockwise == 0:
            vertice1, vertice3 = vertice3, vertice1

        wall[swall].add_materialID(matID)
        wall[swall].add_wall_geometry(wallID, vertice1, vertice2, vertice3)


@ti.kernel
def kernel_set_node_coords(gridSum: int, grid_size: float, gnum: ti.types.vector(3, int), start_point: ti.types.vector(3, float), node: ti.template()):
    for ng in ti.ndrange(gridSum):
        ig, jg, kg = vectorize_id(ng, gnum)
        node[ng].x = start_point + vec3f(ig * grid_size, jg * grid_size, kg * grid_size)


@ti.kernel
def kernel_place_particles_(grid_size: float, igrid_size: float, start_point: ti.types.vector(3, float), region_size: ti.types.vector(3, float), new_particle_num: int, 
                            npic: int, particle: ti.template(), insert_particle_num: ti.template(), is_in_region: ti.template()):
    pnum = int(region_size * npic * igrid_size)
    ti.loop_config(serialize=True)
    for np in range(new_particle_num):
        ip, jp, kp = vectorize_id(np, pnum)
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
        particles[particleNum]._set_essential(bodyID, materialID, density, particle_volume, psize, particle[particleID], init_v, fix_v)


@ti.kernel
def kernel_read_particle_file_(particles: ti.template(), particleNum: ti.template(), particle_num: int, particle: ti.types.ndarray(), psize: ti.types.ndarray(), particle_volume: ti.types.ndarray(), 
                               bodyID: int, materialID: int, density: float, init_v: ti.types.vector(3, float), fix_v: ti.types.vector(3, int)):
    offset = particleNum[None]
    for np in range(particle_num):
        particleNum = offset + np
        particles[particleNum]._set_essential(bodyID, materialID, density, particle_volume, psize, particle[np], init_v, fix_v)


@ti.kernel
def kernel_reload_particle(bodyID: int, materialID: int, active: int, mass: ti.types.ndarray(), volume: ti.types.ndarray(), psize: ti.types.ndarray(), position: ti.types.ndarray(), velocity: ti.types.ndarray(), velocity_gradient: ti.types.ndarray(),
                           stress: ti.types.ndarray(), external_force: ti.types.ndarray(), fix_v: ti.types.vector(3, int), particle: ti.template(), particleNum: ti.template(), material: ti.template()):
    particle_num = bodyID.shape[0]
    for np in range(particle_num):
        particle[np]._reload_essential(bodyID[np], materialID[np], active[np], mass[np], volume[np], psize[np], position[np], velocity[np], velocity_gradient[np], stress[np], external_force[np], fix_v)
    particleNum[None] = particle_num


@ti.kernel
def kernel_add_dem_wall(wallNum: int, no_data: float, cell_size: float, cell_number: ti.types.vector(2, int), digital_elevation: ti.types.ndarray(), wallID: int, matID: int, wall: ti.template()) -> int:
    wall_num = 0
    cellSum = int(cell_number[0] * cell_number[1])
    ti.loop_config(serialize=True)
    for i in range(cellSum):
        xInd, yInd = vectorize_id(i, cell_number)
        if xInd >= cell_number[0] or yInd >= cell_number[1]: continue

        Ind00 = linearize3D(xInd, yInd, 0, cell_number + 1)
        Ind10 = linearize3D(xInd + 1, yInd, 0, cell_number + 1)
        Ind01 = linearize3D(xInd, yInd + 1, 0, cell_number + 1)
        Ind11 = linearize3D(xInd + 1, yInd + 1, 0, cell_number + 1)

        height00 = digital_elevation[Ind00]
        height10 = digital_elevation[Ind10]
        height01 = digital_elevation[Ind01]
        height11 = digital_elevation[Ind11]

        xyCoord00 = vec3f(xInd * cell_size, yInd * cell_size, height00)
        xyCoord10 = vec3f((xInd + 1) * cell_size, yInd * cell_size, height10)
        xyCoord01 = vec3f(xInd * cell_size, (yInd + 1) * cell_size, height01)
        xyCoord11 = vec3f((xInd + 1) * cell_size, (yInd + 1) * cell_size, height11)

        if not equal_to(height00, no_data) and not equal_to(height10, no_data) and not equal_to(height01, no_data) and not equal_to(height11, no_data):
            wall[wallNum + wall_num].add_materialID(matID)
            wall[wallNum + wall_num].add_wall_geometry(wallID, xyCoord00, xyCoord10, xyCoord11)
            wall[wallNum + wall_num + 1].add_materialID(matID)
            wall[wallNum + wall_num + 1].add_wall_geometry(wallID, xyCoord11, xyCoord01, xyCoord00)
            wall_num += 2
            continue
        if equal_to(height00, no_data) and not equal_to(height10, no_data) and not equal_to(height01, no_data) and not equal_to(height11, no_data):
            wall[wallNum + wall_num].add_materialID(matID)
            wall[wallNum + wall_num].add_wall_geometry(wallID, xyCoord10, xyCoord11, xyCoord01)
            wall_num += 1
            continue
        if not equal_to(height00, no_data) and equal_to(height10, no_data) and not equal_to(height01, no_data) and not equal_to(height11, no_data):
            wall[wallNum + wall_num].add_materialID(matID)
            wall[wallNum + wall_num].add_wall_geometry(wallID, xyCoord00, xyCoord11, xyCoord01)
            wall_num += 1
            continue
        if not equal_to(height00, no_data) and not equal_to(height10, no_data) and equal_to(height01, no_data) and not equal_to(height11, no_data):
            wall[wallNum + wall_num].add_materialID(matID)
            wall[wallNum + wall_num].add_wall_geometry(wallID, xyCoord00, xyCoord10, xyCoord11)
            wall_num += 1
            continue
        if not equal_to(height00, no_data) and not equal_to(height10, no_data) and not equal_to(height01, no_data) and equal_to(height11, no_data):
            wall[wallNum + wall_num].add_materialID(matID)
            wall[wallNum + wall_num].add_wall_geometry(wallID, xyCoord00, xyCoord10, xyCoord01)
            wall_num += 1
            continue
    return wall_num