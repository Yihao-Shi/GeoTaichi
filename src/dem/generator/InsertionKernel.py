import taichi as ti

from src.utils.constants import PI, ZEROVEC3f, ZEROVEC6f
from src.utils.ShapeFunctions import SmoothedHeavisideFunction
from src.utils.Quaternion import RodriguesRotationMatrix, RandomGenerator, SetFromTwoVec
from src.utils.TypeDefination import vec3f, vec2f, vec2i, vec4f, vec3u8
from src.utils.ScalarFunction import vectorize_id


@ti.func
def pebble_rotate(local_pebble_coords, target):
    origin=vec3f([0, 0, 1])
    R = RodriguesRotationMatrix(origin, target)
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
    sphere[bodyNum]._add_index(particleNum)
    sphere[bodyNum]._add_sphere_attribute(init_w, inv_inertia, q, fix_v, fix_w)

@ti.kernel
def kernel_create_multisphere_(particle: ti.template(), clump: ti.template(), material: ti.template(), bodyNum: int, particleNum: int, nspheres: int, r_equiv: float, inertia: ti.types.vector(3, float),
                               x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray(), com_pos: ti.types.vector(3, float), equiv_rad: float, get_orientation: ti.template(), 
                               groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float)):
    nsphere = nspheres
    scale_factor = equiv_rad / r_equiv

    density = material[matID]._get_density()
    mass = density * 4./3. * PI * equiv_rad * equiv_rad * equiv_rad
    inv_inertia = 1. / (inertia * density * scale_factor ** 5)
    orientation = get_orientation()
    q = SetFromTwoVec(vec3f([0, 0, 1]), orientation)

    clump[bodyNum]._add_index(particleNum, particleNum + nsphere - 1)
    clump[bodyNum]._add_clump_kinematic(com_pos, init_v, init_w, inv_inertia)
    clump[bodyNum]._add_clump_attribute(mass, equiv_rad, inv_inertia, q)

    for pebble in range(nsphere):
        particleID = pebble + particleNum
        pebble_coord, pebble_radius = get_actual_clump(pebble, com_pos, scale_factor, orientation, x_pebble, rad_pebble) 
        pebble_mass = 4./3. * PI * pebble_radius * pebble_radius * pebble_radius * density
        particle[particleID]._add_index(bodyNum)
        particle[particleID]._add_particle_proporities(matID, groupID, pebble_radius, pebble_mass)
        particle[particleID]._add_particle_kinematics(pebble_coord, init_v + init_w.cross(pebble_coord - com_pos), init_w)

@ti.kernel
def kernel_create_level_set_body_(rigid_body: ti.template(), surface_node: ti.template(), grid: ti.template(), bounding_box: ti.template(), bounding_sphere: ti.template(), material: ti.template(), 
                                  rigidNum: int, gridNum: int, surfaceNum: int, r_bound: int, x_bound: ti.types.vector(3, float), surface_nodes: ti.types.ndarray(), node_coords: ti.types.ndarray(), distance_fields: ti.types.ndarray(),
                                  com_pos: ti.types.vector(3, float), equiv_rad: float, get_orientation: ti.template(), groupID: int, matID: int, init_v: ti.types.vector(3, float), init_w: ti.types.vector(3, float)):
    pass

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
        sphere[bodyNum]._add_index(particleNum)
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
        sphere[bodyNum]._add_index(particleNum)
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
        q = SetFromTwoVec(vec3f([0, 0, 1]), clump_orient)

        clump[bodyNum]._add_index(particleNum, particleNum + nsphere - 1)
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
        startIndex, endIndex = startIndics[nb], endIndics[nb]

        mass = density * 4./3. * PI * clump_rad * clump_rad * clump_rad
        q = SetFromTwoVec(vec3f([0, 0, 1]), clump_orient)

        clump[bodyNum]._add_index(startIndex, endIndex)
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
def kernel_position_rotate_(target: ti.types.vector(3, float), offset: ti.types.vector(3, float), body_coords: ti.template(), start_body_num: int, end_body_num: int):
    origin =vec3f([0, 0, 1]) 
    R = RodriguesRotationMatrix(origin, target)
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
    sphere_coords[insert_body_num[None]] = start_point + position
    sphere_radii[insert_body_num[None]] = radius
    insert_particle(cell_num, cell_size, position - start_point, radius, insert_particle_in_neighbor, neighbor_position, neighbor_radius, num_particle_in_cell, particle_neighbor)
    insert_body_num[None] += 1

@ti.kernel
def kernel_sphere_possion_sampling_(min_rad: float, max_rad: float, tries_default: int, expected_body_num: int, start_point: ti.types.vector(3, float), insert_body_num: ti.template(), insert_particle_in_neighbor: ti.template(), 
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
def kernel_multisphere_possion_sampling_(nspheres: int, r_equiv: float, r_bound: float, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray(), min_rad: float, max_rad: float, tries_default: int, expected_body_num: int, 
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
                pebble_coords[ipebble] = pebble_coord
                pebble_radii[ipebble] = pebble_radius
            pvol = scale_factor * scale_factor * scale_factor * volume_expect
            inserted_volume += pvol
            insert_particle_in_neighbor[None] += nspheres
            insert_body_num[None] += 1
    return inserted_volume


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
def kernel_rebuild_triangular(start: int, number: int, wall: ti.template(), active: ti.types.ndarray(), wallID: ti.types.ndarray(), materialID: ti.types.ndarray(), point1: ti.types.ndarray(), 
                              point2: ti.types.ndarray(), point3: ti.types.ndarray(), norm: ti.types.ndarray(), velocity: ti.types.ndarray()):
    for swall in range(start, start + number):
        nwall = swall - start
        wall[swall]._restart(active[nwall], wallID[nwall], materialID[nwall], vec3f(point1[nwall, 0], point1[nwall, 1], point1[nwall, 2]), vec3f(point2[nwall, 0], point2[nwall, 1], point2[nwall, 2]), 
                             vec3f(point3[nwall, 0], point3[nwall, 1], point3[nwall, 2]), vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2]), vec3f(velocity[nwall, 0], velocity[nwall, 1], velocity[nwall, 2]))


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
def kernel_add_patch(start: int, wallID: int, matID: int, vertices: ti.types.ndarray(), faces: ti.types.ndarray(), center: ti.types.vector(3, float), 
                     offset: ti.types.vector(3, float), direction: ti.types.vector(3, float), init_v: ti.types.vector(3, float), wall: ti.template()):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2]) 
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2]) 
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2]) 
        
        rot_matrix = RodriguesRotationMatrix(vec3f(0, 0, 1), direction)
        vertice1 = rot_matrix @ (vertice1 - center) + center + offset
        vertice1 = rot_matrix @ (vertice2 - center) + center + offset
        vertice1 = rot_matrix @ (vertice3 - center) + center + offset

        wall[swall].add_materialID(matID)
        wall[swall].add_wall_geometry(wallID, vertice1, vertice2, vertice3, init_v)
    

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
