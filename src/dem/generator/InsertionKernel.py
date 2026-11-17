import taichi as ti

from src.utils.constants import PI
from src.utils.GeometryFunction import intersection_ob_bs
from src.utils.Quaternion import theta_to_rotation_matrix, random_generator, set_from_euler, set_to_rotate
from src.utils.TypeDefinition import vec3f, vec2f, vec2i, vec4f, vec3u8, vec3i
from src.utils.ScalarFunction import vectorize_id, linearize3d, equal_to


@ti.func
def pebble_rotate(local_pebble_coords, target):
    R = theta_to_rotation_matrix(target)
    local_pebble_coords = R @ local_pebble_coords
    return local_pebble_coords


@ti.func
def get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble: ti.template(), rad_pebble: ti.template()):
    local_pebble_coords = vec3f(x_pebble[pebble, 0], x_pebble[pebble, 1], x_pebble[pebble, 2]) * scale_factor
    pebble_radius = rad_pebble[pebble] * scale_factor
    global_pebble_coords = pebble_rotate(local_pebble_coords, clump_orient) + com_pos
    return global_pebble_coords, pebble_radius


@ti.func
def validate_clump(
    start_point,
    com_pos,
    scale_factor,
    clump_orient,
    nspheres,
    x_pebble,
    rad_pebble,
    is_overlap,
    check_in_domain,
    insert_particle_in_neighbor,
):
    invalid = 0
    for pebble in range(nspheres):
        pcoord, pradius = get_actual_clump(pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble)
        if is_overlap(pcoord - start_point, pradius, insert_particle_in_neighbor) == 1 or not check_in_domain(
            pcoord, pradius
        ):
            invalid = 1
            break
    return invalid


@ti.kernel
def kernel_create_sphere(
    particle: ti.template(),
    sphere: ti.template(),
    material: ti.template(),
    body_num: int,
    particle_num: int,
    position: ti.types.vector(3, float),
    radius: float,
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, int),
    fix_w: ti.types.vector(3, int),
):
    density = material[mat_id]._get_density()
    mass = density * 4.0 / 3.0 * PI * radius * radius * radius
    inertia = 2.0 / 5.0 * mass * radius * radius
    inv_inertia = 1.0 / inertia
    q = random_generator()

    particle[particle_num]._add_index(-1 - body_num)
    particle[particle_num]._add_particle_proporities(mat_id, group_id, radius, mass)
    particle[particle_num]._add_particle_kinematics(position, init_v, init_w)
    sphere[body_num]._add_index(body_num, particle_num)
    sphere[body_num]._add_sphere_attribute(init_w, inv_inertia, q, fix_v, fix_w)


@ti.kernel
def kernel_create_multisphere(
    particle: ti.template(),
    clump: ti.template(),
    material: ti.template(),
    body_num: int,
    particle_num: int,
    nspheres: int,
    scale_factor: float,
    inertia: ti.types.vector(3, float),
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
    com_pos: ti.types.vector(3, float),
    equiv_rad: float,
    get_orientation: ti.template(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
):
    nsphere = nspheres
    density = material[mat_id]._get_density()
    mass = density * 4.0 / 3.0 * PI * equiv_rad * equiv_rad * equiv_rad
    inv_inertia = 1.0 / (inertia * density * scale_factor**5)
    orientation = get_orientation()
    q = set_from_euler(*orientation)

    clump[body_num]._add_index(body_num, particle_num, particle_num + nsphere - 1)
    clump[body_num]._add_clump_kinematic(com_pos, init_v, init_w, inv_inertia)
    clump[body_num]._add_clump_attribute(mass, equiv_rad, inv_inertia, q)

    for pebble in range(nsphere):
        particle_id = pebble + particle_num
        pebble_coord, pebble_radius = get_actual_clump(pebble, com_pos, scale_factor, orientation, x_pebble, rad_pebble)
        pebble_mass = 4.0 / 3.0 * PI * pebble_radius * pebble_radius * pebble_radius * density
        particle[particle_id]._add_index(body_num)
        particle[particle_id]._add_particle_proporities(mat_id, group_id, pebble_radius, pebble_mass)
        particle[particle_id]._add_particle_kinematics(
            pebble_coord, init_v + init_w.cross(pebble_coord - com_pos), init_w
        )


@ti.func
def create_bounding_sphere(rigid_num, scale, com_pos, bounding_sphere, r_bound, x_bound):
    bounding_sphere[rigid_num]._add_bounding_sphere(x_bound + com_pos, r_bound)
    bounding_sphere[rigid_num]._scale(scale, com_pos)


@ti.func
def create_bounding_box(rigid_num, scale, bounding_box, min_box, max_box, grid_num, space, gnum, extent):
    bounding_box[rigid_num]._set_bounding_box(min_box, max_box)
    bounding_box[rigid_num]._add_grid(grid_num, space, gnum, scale, extent)
    bounding_box[rigid_num]._scale(vec3f(0, 0, 0))


@ti.func
def create_deformable_grids(grid_num, grid_sum, grid, scale, distance_fields: ti.types.ndarray()):
    for i in range(grid_sum):
        grid[i + grid_num]._set_grid(distance_fields[i])
        grid[i + grid_num]._scale(scale)


@ti.func
def create_deformable_surface(
    rigid_num,
    surface_num,
    surface_sum,
    surface_node,
    scale,
    surface_nodes: ti.types.ndarray(),
    parameters: ti.types.ndarray(),
):
    for i in range(surface_sum):
        surface_node[i + surface_num]._set_master(rigid_num)
        surface_node[i + surface_num]._set_surface_node(
            vec3f(surface_nodes[i, 0], surface_nodes[i, 1], surface_nodes[i, 2])
        )
        surface_node[i + surface_num]._set_coefficient(parameters[i])
        surface_node[i + surface_num]._scale(scale, vec3f(0, 0, 0))


@ti.kernel
def kernel_create_level_set_rigid_body(
    rigid_body: ti.template(),
    bounding_box: ti.template(),
    bounding_sphere: ti.template(),
    master: ti.template(),
    material: ti.template(),
    rigid_num: int,
    grid_num: int,
    vertice_num: int,
    surface_num: int,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    surface_sum: int,
    reference_surface_area: float,
    space: float,
    gnum: ti.types.vector(3, int),
    extent: int,
    scale_factor: float,
    inertia: ti.types.vector(3, float),
    com_pos: ti.types.vector(3, float),
    equiv_rad: float,
    get_orientation: ti.template(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    is_fix: ti.types.vector(3, int),
):
    density = material[mat_id]._get_density()
    mass = 4.0 / 3.0 * PI * equiv_rad * equiv_rad * equiv_rad
    inv_inertia = 1.0 / (inertia * scale_factor**5)
    orientation = get_orientation()
    q = set_from_euler(*orientation)
    rotation_matrix = set_to_rotate(q)

    create_bounding_sphere(rigid_num, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound)
    create_bounding_box(rigid_num, scale_factor, bounding_box, min_box, max_box, grid_num, space, gnum, extent)
    bounding_box[rigid_num]._set_reference_surface_area(reference_surface_area)
    for i in range(surface_sum):
        master[i + surface_num] = rigid_num

    rigid_body[rigid_num]._add_body_attribute(com_pos, mass, equiv_rad, inv_inertia, q)
    rigid_body[rigid_num]._add_surface_index(surface_num, surface_num + surface_sum, vertice_num)
    rigid_body[rigid_num]._add_body_properties(mat_id, group_id, density)
    rigid_body[rigid_num]._add_body_kinematic(init_v, init_w, is_fix)


@ti.kernel
def kernel_create_implicit_surface_rigid_body(
    rigid_body: ti.template(),
    bounding_sphere: ti.template(),
    material: ti.template(),
    rigid_num: int,
    template_id: int,
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    scale_factor: float,
    inertia: ti.types.vector(3, float),
    com_pos: ti.types.vector(3, float),
    equiv_rad: float,
    get_orientation: ti.template(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    is_fix: ti.types.vector(3, int),
):
    density = material[mat_id]._get_density()
    mass = 4.0 / 3.0 * PI * equiv_rad * equiv_rad * equiv_rad
    inv_inertia = 1.0 / (inertia * scale_factor**5)
    orientation = get_orientation()
    q = set_from_euler(*orientation)
    rotation_matrix = set_to_rotate(q)

    create_bounding_sphere(rigid_num, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound)
    rigid_body[rigid_num]._add_body_attribute(scale_factor, com_pos, mass, equiv_rad, inv_inertia, q)
    rigid_body[rigid_num]._add_body_properties(mat_id, group_id, template_id, density)
    rigid_body[rigid_num]._add_body_kinematic(init_v, init_w, is_fix)


@ti.kernel
def kernel_create_deformable_body(
    rigid_body: ti.template(),
    surface_node: ti.template(),
    grid: ti.template(),
    bounding_box: ti.template(),
    bounding_sphere: ti.template(),
    material: ti.template(),
    rigid_num: int,
    grid_num: int,
    surface_num: int,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    surface_sum: int,
    surface_nodes: ti.types.ndarray(),
    parameters: ti.types.ndarray(),
    grid_sum: int,
    space: float,
    gnum: ti.types.vector(3, int),
    start_point: ti.types.vector(3, float),
    distance_fields: ti.types.ndarray(),
    scale_factor: float,
    inertia: ti.types.vector(3, float),
    com_pos: ti.types.vector(3, float),
    equiv_rad: float,
    get_orientation: ti.template(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    is_fix: ti.types.vector(3, int),
):
    create_bounding_sphere(rigid_num, scale_factor, com_pos, bounding_sphere, r_bound, x_bound)
    create_bounding_box(rigid_num, scale_factor, bounding_box, min_box, max_box, grid_num, space, gnum, start_point)
    create_deformable_grids(grid_num, grid_sum, grid, scale_factor, distance_fields)
    create_deformable_surface(
        rigid_num, surface_num, surface_sum, surface_node, scale_factor, surface_nodes, parameters
    )
    bounding_box[rigid_num].scale = 1.0

    density = material[mat_id]._get_density()
    mass = density * 4.0 / 3.0 * PI * equiv_rad * equiv_rad * equiv_rad
    inv_inertia = 1.0 / (inertia * density * scale_factor**5)
    orientation = get_orientation()
    q = set_from_euler(*orientation)

    rigid_body[rigid_num]._add_surface_index(surface_num, surface_num + surface_sum)
    rigid_body[rigid_num]._add_body_properties(mat_id, group_id, mass, equiv_rad, inv_inertia, q)
    rigid_body[rigid_num]._add_body_kinematic(com_pos, init_v, init_w, inv_inertia, is_fix)


@ti.kernel
def kernel_get_orient(start_body_num: int, end_body_num: int, get_orientation: ti.template(), orients: ti.template()):
    for nb in range(end_body_num - start_body_num):
        orients[start_body_num + nb] = get_orientation()


@ti.kernel
def kernel_add_sphere_packing(
    particle: ti.template(),
    sphere: ti.template(),
    material: ti.template(),
    init_body_num: int,
    init_particle_num: int,
    start_body_num: int,
    end_body_num: int,
    coords: ti.template(),
    radii: ti.template(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, int),
    fix_w: ti.types.vector(3, int),
):
    density = material[mat_id]._get_density()
    for nb in range(end_body_num - start_body_num):
        body_num = nb + init_body_num
        particle_num = nb * 1 + init_particle_num
        position, radius = coords[start_body_num + nb], radii[start_body_num + nb]
        mass = density * 4.0 / 3.0 * PI * radius * radius * radius
        inertia = 2.0 / 5.0 * mass * radius * radius
        inv_inertia = 1.0 / inertia
        q = random_generator()

        particle[particle_num]._add_index(-1 - body_num)
        particle[particle_num]._add_particle_proporities(mat_id, group_id, radius, mass)
        particle[particle_num]._add_particle_kinematics(position, init_v, init_w)
        sphere[body_num]._add_index(body_num, particle_num)
        sphere[body_num]._add_sphere_attribute(init_w, inv_inertia, q, fix_v, fix_w)


@ti.kernel
def kernel_add_sphere_files(
    particle: ti.template(),
    sphere: ti.template(),
    material: ti.template(),
    init_body_num: int,
    init_particle_num: int,
    body_num: int,
    coords: ti.types.ndarray(),
    radii: ti.types.ndarray(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, int),
    fix_w: ti.types.vector(3, int),
):
    density = material[mat_id]._get_density()
    for nb in range(body_num):
        body_index = nb + init_body_num
        particle_num = nb * 1 + init_particle_num
        position, radius = vec3f([coords[nb, 0], coords[nb, 1], coords[nb, 2]]), radii[nb]
        mass = density * 4.0 / 3.0 * PI * radius * radius * radius
        inertia = 2.0 / 5.0 * mass * radius * radius
        inv_inertia = 1.0 / inertia
        q = random_generator()

        particle[particle_num]._add_index(-1 - body_index)
        particle[particle_num]._add_particle_proporities(mat_id, group_id, radius, mass)
        particle[particle_num]._add_particle_kinematics(position, init_v, init_w)
        sphere[body_index]._add_index(body_index, particle_num)
        sphere[body_index]._add_sphere_attribute(init_w, inv_inertia, q, fix_v, fix_w)


@ti.kernel
def kernel_add_multisphere_packing(
    particle: ti.template(),
    clump: ti.template(),
    material: ti.template(),
    init_body_num: int,
    init_particle_num: int,
    start_body_num: int,
    end_body_num: int,
    start_pebble_num: int,
    template_ptr: ti.template(),
    clump_coords: ti.template(),
    clump_radii: ti.template(),
    clump_orients: ti.template(),
    pebble_coords: ti.template(),
    pebble_radii: ti.template(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
):
    density = material[mat_id]._get_density()
    for nb in range(end_body_num - start_body_num):
        nsphere = template_ptr.nspheres
        body_num = nb + init_body_num
        particle_num = nb * nsphere + init_particle_num
        clump_com, clump_rad, clump_orient = (
            clump_coords[start_body_num + nb],
            clump_radii[start_body_num + nb],
            clump_orients[start_body_num + nb],
        )
        scale_factor = clump_rad / template_ptr.r_equiv

        mass = density * 4.0 / 3.0 * PI * clump_rad * clump_rad * clump_rad
        inv_inertia = 1.0 / (scale_factor**5 * template_ptr.inertia * density)
        q = set_from_euler(*clump_orient)

        clump[body_num]._add_index(body_num, particle_num, particle_num + nsphere - 1)
        clump[body_num]._add_clump_kinematic(clump_com, init_v, init_w, inv_inertia)
        clump[body_num]._add_clump_attribute(mass, clump_rad, inv_inertia, q)

        for pebble in range(nsphere):
            particle_id = pebble + particle_num
            pebble_coord, pebble_radius = (
                pebble_coords[start_pebble_num + nb * nsphere + pebble],
                pebble_radii[start_pebble_num + nb * nsphere + pebble],
            )
            pebble_mass = 4 / 3.0 * PI * pebble_radius * pebble_radius * pebble_radius * density
            particle[particle_id]._add_index(body_num)
            particle[particle_id]._add_particle_proporities(mat_id, group_id, pebble_radius, pebble_mass)
            particle[particle_id]._add_particle_kinematics(
                pebble_coord, init_v + init_w.cross(pebble_coord - clump_com), init_w
            )


@ti.kernel
def kernel_add_multisphere_files(
    particle: ti.template(),
    clump: ti.template(),
    material: ti.template(),
    init_body_num: int,
    init_particle_num: int,
    body_num: int,
    pebble_num: int,
    clump_coords: ti.types.ndarray(),
    clump_radii: ti.types.ndarray(),
    clump_orients: ti.types.ndarray(),
    clump_inertia_vol: ti.types.ndarray(),
    start_indics: ti.types.ndarray(),
    end_indics: ti.types.ndarray(),
    pebble_coords: ti.types.ndarray(),
    pebble_radii: ti.types.ndarray(),
    multisphere_indics: ti.types.ndarray(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
):
    density = material[mat_id]._get_density()
    for nb in range(body_num):
        body_index = nb + init_body_num
        clump_com, clump_rad = vec3f([clump_coords[nb, 0], clump_coords[nb, 1], clump_coords[nb, 2]]), clump_radii[nb]
        clump_orient = vec3f([clump_orients[nb, 0], clump_orients[nb, 1], clump_orients[nb, 2]])
        clump_inv_inertia = 1.0 / (
            vec3f([clump_inertia_vol[nb, 0], clump_inertia_vol[nb, 1], clump_inertia_vol[nb, 2]]) * density
        )
        start_index, end_index = init_particle_num + int(start_indics[nb]), init_particle_num + int(end_indics[nb])

        mass = density * 4.0 / 3.0 * PI * clump_rad * clump_rad * clump_rad
        q = set_from_euler(*clump_orient)

        clump[body_index]._add_index(body_index, start_index, end_index)
        clump[body_index]._add_clump_kinematic(clump_com, init_v, init_w, clump_inv_inertia)
        clump[body_index]._add_clump_attribute(mass, clump_rad, clump_inv_inertia, q)

    for pebble in range(pebble_num):
        particle_id = pebble + init_particle_num
        pebble_coord = vec3f([pebble_coords[pebble, 0], pebble_coords[pebble, 1], pebble_coords[pebble, 2]])
        pebble_radius, multisphere_index = pebble_radii[pebble], int(multisphere_indics[pebble])
        pebble_mass = 4 / 3.0 * PI * pebble_radius * pebble_radius * pebble_radius * density
        clump_com = vec3f(
            [clump_coords[multisphere_index, 0], clump_coords[multisphere_index, 1], clump_coords[multisphere_index, 2]]
        )
        particle[particle_id]._add_index(multisphere_index)
        particle[particle_id]._add_particle_proporities(mat_id, group_id, pebble_radius, pebble_mass)
        particle[particle_id]._add_particle_kinematics(
            pebble_coord, init_v + init_w.cross(pebble_coord - clump_com), init_w
        )


@ti.kernel
def kernel_add_levelset_packing(
    rigid_body: ti.template(),
    bounding_box: ti.template(),
    bounding_sphere: ti.template(),
    master: ti.template(),
    material: ti.template(),
    rigid_num: int,
    grid_num: int,
    surface_num: int,
    vertice_num: int,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    surface_sum: int,
    reference_surface_area: float,
    space: float,
    gnum: ti.types.vector(3, int),
    extent: int,
    inertia: ti.types.vector(3, float),
    eqradius: float,
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    is_fix: ti.types.vector(3, int),
    start_body_num: int,
    end_body_num: int,
    coords: ti.template(),
    radii: ti.template(),
    orients: ti.template(),
    coordinates_are_mass_centers: ti.template(),
):
    density = material[mat_id]._get_density()
    for nb in range(end_body_num - start_body_num):
        bounding_x, bounding_r = coords[start_body_num + nb], radii[start_body_num + nb]
        scale_factor = bounding_r / r_bound
        orientation = orients[start_body_num + nb]
        q = set_from_euler(*orientation)
        rotation_matrix = set_to_rotate(q)
        com_pos = bounding_x
        if ti.static(not coordinates_are_mass_centers):
            com_pos = bounding_x - scale_factor * rotation_matrix @ x_bound
        equiv_rad = scale_factor * eqradius
        mass = 4.0 / 3.0 * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1.0 / (inertia * scale_factor**5)

        create_bounding_sphere(
            rigid_num + nb, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound
        )
        create_bounding_box(rigid_num + nb, scale_factor, bounding_box, min_box, max_box, grid_num, space, gnum, extent)
        bounding_box[rigid_num + nb]._set_reference_surface_area(reference_surface_area)
        for i in range(surface_sum):
            master[i + surface_num + nb * surface_sum] = rigid_num + nb

        rigid_body[rigid_num + nb]._add_body_attribute(com_pos, mass, equiv_rad, inv_inertia, q)
        rigid_body[rigid_num + nb]._add_surface_index(
            surface_num + nb * surface_sum, surface_num + (nb + 1) * surface_sum, vertice_num
        )
        rigid_body[rigid_num + nb]._add_body_properties(mat_id, group_id, density)
        rigid_body[rigid_num + nb]._add_body_kinematic(init_v, init_w, is_fix)


@ti.kernel
def kernel_add_implicit_surface_packing(
    rigid_body: ti.template(),
    bounding_sphere: ti.template(),
    material: ti.template(),
    rigid_num: int,
    template_id: int,
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    inertia: ti.types.vector(3, float),
    eqradius: float,
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    is_fix: ti.types.vector(3, int),
    start_body_num: int,
    end_body_num: int,
    coords: ti.template(),
    radii: ti.template(),
    orients: ti.template(),
):
    density = material[mat_id]._get_density()
    for nb in range(end_body_num - start_body_num):
        bounding_x, bounding_r = coords[start_body_num + nb], radii[start_body_num + nb]
        scale_factor = bounding_r / r_bound
        orientation = orients[start_body_num + nb]
        q = set_from_euler(*orientation)
        rotation_matrix = set_to_rotate(q)
        com_pos, equiv_rad = bounding_x - scale_factor * rotation_matrix @ x_bound, scale_factor * eqradius
        mass = 4.0 / 3.0 * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1.0 / (inertia * scale_factor**5)

        create_bounding_sphere(
            rigid_num + nb, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound
        )
        rigid_body[rigid_num + nb]._add_body_attribute(scale_factor, com_pos, mass, equiv_rad, inv_inertia, q)
        rigid_body[rigid_num + nb]._add_body_properties(mat_id, group_id, template_id, density)
        rigid_body[rigid_num + nb]._add_body_kinematic(init_v, init_w, is_fix)


@ti.kernel
def kernel_add_levelset_files(
    rigid_body: ti.template(),
    bounding_box: ti.template(),
    bounding_sphere: ti.template(),
    master: ti.template(),
    rigid_num: int,
    grid_num: int,
    surface_num: int,
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    surface_sum: int,
    reference_surface_area: float,
    inertia: ti.types.vector(3, float),
    eqradius: float,
    space: float,
    gnum: ti.types.vector(3, int),
    extent: int,
    body_num: int,
    coords: ti.types.ndarray(),
    radii: ti.types.ndarray(),
    orients: ti.types.ndarray(),
):
    for nb in range(body_num):
        bounding_x, bounding_r = vec3f([coords[nb, 0], coords[nb, 1], coords[nb, 2]]), radii[nb]
        scale_factor = bounding_r / r_bound
        orientation = vec3f([orients[nb, 0], orients[nb, 1], orients[nb, 2]])
        q = set_from_euler(*orientation)
        rotation_matrix = set_to_rotate(q)
        com_pos, equiv_rad = bounding_x - scale_factor * rotation_matrix @ x_bound, scale_factor * eqradius
        mass = 4.0 / 3.0 * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1.0 / (inertia * scale_factor**5)

        create_bounding_sphere(
            rigid_num + nb, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound
        )
        create_bounding_box(rigid_num + nb, scale_factor, bounding_box, min_box, max_box, grid_num, space, gnum, extent)
        bounding_box[rigid_num + nb]._set_reference_surface_area(reference_surface_area)
        for i in range(surface_sum):
            master[i + surface_num + nb * surface_sum] = rigid_num + nb

        rigid_body[rigid_num + nb]._add_body_attribute(com_pos, mass, equiv_rad, inv_inertia, q)


@ti.kernel
def kernel_add_rigid_body(
    rigid_body: ti.template(),
    material: ti.template(),
    rigid_num: int,
    surface_num: int,
    vertice_num: int,
    body_num: int,
    surface_sum: int,
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    is_fix: ti.types.vector(3, int),
):
    density = material[mat_id]._get_density()
    for nb in range(body_num):
        rigid_body[rigid_num + nb]._add_surface_index(
            surface_num + nb * surface_sum, surface_num + (nb + 1) * surface_sum, vertice_num
        )
        rigid_body[rigid_num + nb]._add_body_properties(mat_id, group_id, density)
        rigid_body[rigid_num + nb]._add_body_kinematic(init_v, init_w, is_fix)


@ti.kernel
def kernel_add_implicit_surface_files(
    rigid_body: ti.template(),
    bounding_sphere: ti.template(),
    material: ti.template(),
    rigid_num: int,
    template_id: int,
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    inertia: ti.types.vector(3, float),
    eqradius: float,
    body_num: int,
    coords: ti.types.ndarray(),
    radii: ti.types.ndarray(),
    orients: ti.types.ndarray(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    is_fix: ti.types.vector(3, int),
):
    density = material[mat_id]._get_density()
    for nb in range(body_num):
        bounding_x, bounding_r = vec3f([coords[nb, 0], coords[nb, 1], coords[nb, 2]]), radii[nb]
        scale_factor = bounding_r / r_bound
        orientation = vec3f([orients[nb, 0], orients[nb, 1], orients[nb, 2]])
        q = set_from_euler(*orientation)
        rotation_matrix = set_to_rotate(q)
        com_pos, equiv_rad = bounding_x - scale_factor * rotation_matrix @ x_bound, scale_factor * eqradius
        mass = 4.0 / 3.0 * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1.0 / (inertia * scale_factor**5)

        create_bounding_sphere(
            rigid_num + nb, scale_factor, com_pos, bounding_sphere, r_bound, rotation_matrix @ x_bound
        )
        rigid_body[rigid_num + nb]._add_body_attribute(scale_factor, com_pos, mass, equiv_rad, inv_inertia, q)
        rigid_body[rigid_num + nb]._add_body_properties(mat_id, group_id, template_id, density)
        rigid_body[rigid_num + nb]._add_body_kinematic(init_v, init_w, is_fix)


@ti.kernel
def kernel_add_deformable_packing(
    rigid_body: ti.template(),
    surface_node: ti.template(),
    grid: ti.template(),
    bounding_box: ti.template(),
    bounding_sphere: ti.template(),
    material: ti.template(),
    rigid_num: int,
    grid_num: int,
    surface_num: int,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    surface_sum: int,
    surface_nodes: ti.types.ndarray(),
    parameters: ti.types.ndarray(),
    grid_sum: int,
    space: float,
    gnum: ti.types.vector(3, int),
    start_point: ti.types.vector(3, float),
    distance_fields: ti.types.ndarray(),
    inertia: ti.types.vector(3, float),
    eqradius: float,
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
    start_body_num: int,
    end_body_num: int,
    coords: ti.template(),
    radii: ti.template(),
    orients: ti.template(),
    is_fix: ti.types.vector(3, int),
):
    density = material[mat_id]._get_density()
    for nb in range(end_body_num - start_body_num):
        bounding_x, bounding_r = coords[start_body_num + nb], radii[start_body_num + nb]
        scale_factor = bounding_r / r_bound
        com_pos, equiv_rad = bounding_x - scale_factor * x_bound, scale_factor * eqradius
        create_bounding_sphere(rigid_num + nb, scale_factor, com_pos, bounding_sphere, r_bound, x_bound)
        create_bounding_box(
            rigid_num + nb, scale_factor, bounding_box, min_box, max_box, grid_num, space, gnum, start_point
        )
        create_deformable_grids(grid_num + nb * grid_sum, grid_sum, grid, scale_factor, distance_fields)
        create_deformable_surface(
            rigid_num + nb,
            surface_num + nb * surface_sum,
            surface_sum,
            surface_node,
            scale_factor,
            surface_nodes,
            parameters,
        )
        bounding_box[rigid_num].scale = 1.0

        mass = density * 4.0 / 3.0 * PI * equiv_rad * equiv_rad * equiv_rad
        inv_inertia = 1.0 / (inertia * density * scale_factor**5)
        orientation = orients[start_body_num + nb]
        q = set_from_euler(*orientation)

        rigid_body[rigid_num + nb]._add_surface_index(
            surface_num + nb * surface_sum, surface_num + (nb + 1) * surface_sum
        )
        rigid_body[rigid_num + nb]._add_body_properties(mat_id, group_id, mass, equiv_rad, inv_inertia, q)
        rigid_body[rigid_num + nb]._add_body_kinematic(com_pos, init_v, init_w, inv_inertia, is_fix)


@ti.kernel
def kernel_position_rotate(
    target: ti.types.vector(3, float),
    offset: ti.types.vector(3, float),
    body_coords: ti.template(),
    start_body_num: int,
    end_body_num: int,
):
    R = theta_to_rotation_matrix(target)
    for nb in range(start_body_num, end_body_num):
        coords = body_coords[nb]
        coords -= offset
        coords = R @ coords
        coords += offset
        body_coords[nb] = coords


@ti.kernel
def kernel_update_particle_number_by_sphere(
    body_num: int, sphere: ti.template(), particle: ti.template(), check_in_region: ti.template()
) -> ti.types.vector(2, int):
    inserted_body = 0
    inserted_particle = 0
    for nb in range(body_num):
        index = sphere[nb].sphereIndex
        position = particle[index].x
        radius = particle[index].rad
        if check_in_region(position, radius):
            inserted_body += 1
            inserted_particle += 1
    return vec2i([inserted_body, inserted_particle])


@ti.kernel
def kernel_update_particle_volume_by_sphere(
    body_num: int, sphere: ti.template(), particle: ti.template(), check_in_region: ti.template()
) -> ti.types.vector(2, float):
    inserted_volume = 0.0
    inserted_particle = 0
    for nb in range(body_num):
        index = sphere[nb].sphereIndex
        position = particle[index].x
        radius = particle[index].rad
        if check_in_region(position, radius):
            inserted_volume += particle[index]._get_volume()
            inserted_particle += 1
    return vec2f([inserted_volume, float(inserted_particle)])


@ti.kernel
def kernel_update_particle_number_by_levelset(
    rigid_num: int, bounding_sphere: ti.template(), check_in_region: ti.template()
) -> ti.types.vector(2, int):
    inserted_body = 0
    inserted_particle = 0
    for index in range(rigid_num):
        position = bounding_sphere[index].x
        radius = bounding_sphere[index].rad
        if check_in_region(position, radius):
            inserted_body += 1
            inserted_particle += 1
    return vec2i([inserted_body, inserted_particle])


@ti.kernel
def kernel_update_particle_volume_by_levelset(
    rigid_num: int, bounding_sphere: ti.template(), rigid: ti.template(), check_in_region: ti.template()
) -> ti.types.vector(2, float):
    inserted_volume = 0.0
    inserted_particle = 0
    for index in range(rigid_num):
        position = bounding_sphere[index].x
        radius = bounding_sphere[index].rad
        if check_in_region(position, radius):
            inserted_volume += rigid[index]._get_volume()
            inserted_particle += 1
    return vec2f([inserted_volume, inserted_particle])


@ti.kernel
def kernel_update_pebble_number_by_clump(
    body_num: int, clump: ti.template(), particle: ti.template(), check_in_region: ti.template()
) -> ti.types.vector(2, int):
    inserted_body = 0
    inserted_particle = 0
    for nb in range(body_num):
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
def kernel_update_particle_volume_by_clump(
    body_num: int, clump: ti.template(), particle: ti.template(), check_in_region: ti.template()
) -> ti.types.vector(2, float):
    inserted_volume = 0.0
    inserted_particle = 0
    for nb in range(body_num):
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
def kernel_insert_first_sphere(
    start_point: ti.types.vector(3, float),
    position: ti.types.vector(3, float),
    radius: float,
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    sphere_coords: ti.template(),
    sphere_radii: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    neighbor_position: ti.template(),
    neighbor_radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    insert_particle: ti.template(),
):
    sphere_coords[insert_body_num[None]] = position
    sphere_radii[insert_body_num[None]] = radius
    insert_particle(
        cell_num,
        cell_size,
        position - start_point,
        radius,
        insert_particle_in_neighbor,
        neighbor_position,
        neighbor_radius,
        num_particle_in_cell,
        particle_neighbor,
    )
    insert_body_num[None] += 1


@ti.kernel
def kernel_sphere_poisson_sampling(
    radius_dist: ti.types.ndarray(),
    tries_default: int,
    expected_body_num: int,
    start_point: ti.types.vector(3, float),
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    sphere_coords: ti.template(),
    sphere_radii: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    position: ti.template(),
    radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_domain: ti.template(),
    overlap: ti.template(),
    insert_particle: ti.template(),
):
    tries = 0
    start_index = insert_body_num[None]
    while tries < insert_body_num[None] and tries < expected_body_num:
        source_x, source_rad = sphere_coords[tries], sphere_radii[tries]
        for _ in range(tries_default):
            sphere_radius = radius_dist[insert_body_num[None] - start_index]
            u, v = ti.random(), ti.random()
            theta, phi = 2 * PI * u, ti.acos(2 * v - 1)
            randvector = vec3f([ti.sin(theta) * ti.sin(phi), ti.cos(theta) * ti.sin(phi), ti.cos(phi)]).normalized()
            offset = randvector * ((1 + ti.random()) * sphere_radius + source_rad)
            sphere_coord = source_x + offset

            if (
                check_in_domain(sphere_coord, sphere_radius)
                and insert_body_num[None] < expected_body_num
                and overlap(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    num_particle_in_cell,
                    particle_neighbor,
                )
                == 0
            ):
                sphere_coords[insert_body_num[None]] = sphere_coord
                sphere_radii[insert_body_num[None]] = sphere_radius
                insert_particle(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    num_particle_in_cell,
                    particle_neighbor,
                )
                insert_body_num[None] += 1
        tries += 1


@ti.kernel
def kernel_insert_first_rigid(
    start_point: ti.types.vector(3, float),
    position: ti.types.vector(3, float),
    radius: float,
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    sphere_coords: ti.template(),
    sphere_radii: ti.template(),
    orients: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    neighbor_position: ti.template(),
    neighbor_radius: ti.template(),
    orient: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    insert_particle: ti.template(),
    get_orientation: ti.template(),
):
    ori = get_orientation()
    sphere_coords[insert_body_num[None]] = position
    sphere_radii[insert_body_num[None]] = radius
    orients[insert_body_num[None]] = ori
    insert_particle(
        cell_num,
        cell_size,
        position - start_point,
        radius,
        ori,
        insert_particle_in_neighbor,
        neighbor_position,
        neighbor_radius,
        orient,
        num_particle_in_cell,
        particle_neighbor,
    )
    insert_body_num[None] += 1


@ti.kernel
def kernel_rigid_poisson_sampling(
    radius_dist: ti.types.ndarray(),
    tries_default: int,
    expected_body_num: int,
    start_point: ti.types.vector(3, float),
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    sphere_coords: ti.template(),
    sphere_radii: ti.template(),
    orients: ti.template(),
    template_xbound: ti.types.vector(3, float),
    template_rbound: float,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    position: ti.template(),
    radius: ti.template(),
    orient: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_domain: ti.template(),
    overlap: ti.template(),
    insert_particle: ti.template(),
    get_orientation: ti.template(),
):
    tries = 0
    start_index = insert_body_num[None]
    while tries < insert_body_num[None] and tries < expected_body_num:
        source_x, source_rad = sphere_coords[tries], sphere_radii[tries]
        for _ in range(tries_default):
            sphere_radius = radius_dist[insert_body_num[None] - start_index]
            u, v = ti.random(), ti.random()
            theta, phi = 2 * PI * u, ti.acos(2 * v - 1)
            randvector = vec3f([ti.sin(theta) * ti.sin(phi), ti.cos(theta) * ti.sin(phi), ti.cos(phi)]).normalized()
            offset = randvector * ((1 + ti.random()) * sphere_radius + source_rad)
            sphere_coord = source_x + offset
            ori = get_orientation()

            if (
                check_in_domain(sphere_coord, sphere_radius)
                and insert_body_num[None] < expected_body_num
                and overlap(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    orient,
                    num_particle_in_cell,
                    particle_neighbor,
                    template_xbound,
                    template_rbound,
                    min_box,
                    max_box,
                    ori,
                )
                == 0
            ):
                sphere_coords[insert_body_num[None]] = sphere_coord
                sphere_radii[insert_body_num[None]] = sphere_radius
                orients[insert_body_num[None]] = ori
                insert_particle(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    ori,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    orient,
                    num_particle_in_cell,
                    particle_neighbor,
                )
                insert_body_num[None] += 1
        tries += 1


@ti.kernel
def kernel_sphere_generate_without_overlap(
    radius_dist: ti.types.ndarray(),
    tries_default: int,
    expected_body_num: int,
    start_point: ti.types.vector(3, float),
    region_size: ti.types.vector(3, float),
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    sphere_coords: ti.template(),
    sphere_radii: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    position: ti.template(),
    radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_domain: ti.template(),
    overlap: ti.template(),
    insert_particle: ti.template(),
):
    start_index = insert_body_num[None]
    while insert_body_num[None] < expected_body_num:
        count = 0
        for _ in range(tries_default):
            sphere_radius = radius_dist[insert_body_num[None] - start_index]
            offset = vec3f([ti.random(), ti.random(), ti.random()]) * region_size
            sphere_coord = start_point + offset

            if (
                check_in_domain(sphere_coord, sphere_radius)
                and overlap(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    num_particle_in_cell,
                    particle_neighbor,
                )
                == 0
            ):
                sphere_coords[insert_body_num[None]] = sphere_coord
                sphere_radii[insert_body_num[None]] = sphere_radius
                insert_particle(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    num_particle_in_cell,
                    particle_neighbor,
                )
                insert_body_num[None] += 1
                break
            count += 1
        if count == tries_default:
            break


@ti.kernel
def kernel_rigid_generate_without_overlap(
    radius_dist: ti.types.ndarray(),
    tries_default: int,
    expected_body_num: int,
    start_point: ti.types.vector(3, float),
    region_size: ti.types.vector(3, float),
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    sphere_coords: ti.template(),
    sphere_radii: ti.template(),
    orients: ti.template(),
    template_xbound: ti.types.vector(3, float),
    template_rbound: float,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    position: ti.template(),
    radius: ti.template(),
    orient: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_domain: ti.template(),
    overlap: ti.template(),
    insert_particle: ti.template(),
    get_orientation: ti.template(),
):
    start_index = insert_body_num[None]
    while insert_body_num[None] < expected_body_num:
        count = 0
        for _ in range(tries_default):
            sphere_radius = radius_dist[insert_body_num[None] - start_index]
            offset = vec3f([ti.random(), ti.random(), ti.random()]) * region_size
            sphere_coord = start_point + offset
            ori = get_orientation()

            if (
                check_in_domain(sphere_coord, sphere_radius)
                and overlap(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    orient,
                    num_particle_in_cell,
                    particle_neighbor,
                    template_xbound,
                    template_rbound,
                    min_box,
                    max_box,
                    ori,
                )
                == 0
            ):
                sphere_coords[insert_body_num[None]] = sphere_coord
                sphere_radii[insert_body_num[None]] = sphere_radius
                orients[insert_body_num[None]] = ori
                insert_particle(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    ori,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    orient,
                    num_particle_in_cell,
                    particle_neighbor,
                )
                insert_body_num[None] += 1
                break
            count += 1
        if count == tries_default:
            break


@ti.kernel
def kernel_sphere_generate_lattice(
    max_rad: float,
    radius_dist: ti.types.ndarray(),
    expected_body_num: int,
    available_num: int,
    position_distribution: ti.types.vector(3, int),
    start_point: ti.types.vector(3, float),
    valid: ti.template(),
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    sphere_coords: ti.template(),
    sphere_radii: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    position: ti.template(),
    radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_domain: ti.template(),
    overlap: ti.template(),
    insert_particle: ti.template(),
):
    tries = insert_body_num[None]
    start_index = insert_body_num[None]
    target_num = expected_body_num - start_index
    while tries < expected_body_num:
        sample_id = tries - start_index
        sphere_radius = radius_dist[sample_id]
        random_id = ti.min(available_num - 1, int((float(sample_id) + 0.5) * float(available_num) / float(target_num)))
        offset = vec3f([ti.random(), ti.random(), ti.random()]) * (max_rad - sphere_radius)
        sphere_coord = (
            start_point + (vec3f(vectorize_id(valid[random_id], position_distribution)) + 0.5) * 2.0 * max_rad + offset
        )
        if check_in_domain(sphere_coord, sphere_radius):
            if (
                overlap(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    num_particle_in_cell,
                    particle_neighbor,
                )
                == 0
            ):
                sphere_coords[insert_body_num[None]] = sphere_coord
                sphere_radii[insert_body_num[None]] = sphere_radius
                insert_particle(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    num_particle_in_cell,
                    particle_neighbor,
                )
                insert_body_num[None] += 1
        valid[random_id] = -1
        tries += 1


@ti.kernel
def kernel_rigid_generate_lattice(
    max_rad: float,
    radius_dist: ti.types.ndarray(),
    expected_body_num: int,
    available_num: int,
    position_distribution: ti.types.vector(3, int),
    start_point: ti.types.vector(3, float),
    valid: ti.template(),
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    sphere_coords: ti.template(),
    sphere_radii: ti.template(),
    orients: ti.template(),
    template_xbound: ti.types.vector(3, float),
    template_rbound: float,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    position: ti.template(),
    radius: ti.template(),
    orient: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_domain: ti.template(),
    overlap: ti.template(),
    insert_particle: ti.template(),
    get_orientation: ti.template(),
):
    tries = insert_body_num[None]
    start_index = insert_body_num[None]
    target_num = expected_body_num - start_index
    while tries < expected_body_num:
        sample_id = tries - start_index
        sphere_radius = radius_dist[sample_id]
        random_id = ti.min(available_num - 1, int((float(sample_id) + 0.5) * float(available_num) / float(target_num)))
        offset = vec3f([ti.random(), ti.random(), ti.random()]) * (max_rad - sphere_radius)
        sphere_coord = (
            start_point + (vec3f(vectorize_id(valid[random_id], position_distribution)) + 0.5) * 2.0 * max_rad + offset
        )
        ori = get_orientation()
        if check_in_domain(sphere_coord, sphere_radius):
            if (
                overlap(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    orient,
                    num_particle_in_cell,
                    particle_neighbor,
                    template_xbound,
                    template_rbound,
                    min_box,
                    max_box,
                    ori,
                )
                == 0
            ):
                sphere_coords[insert_body_num[None]] = sphere_coord
                sphere_radii[insert_body_num[None]] = sphere_radius
                orients[insert_body_num[None]] = ori
                insert_particle(
                    cell_num,
                    cell_size,
                    sphere_coord - start_point,
                    sphere_radius,
                    ori,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    orient,
                    num_particle_in_cell,
                    particle_neighbor,
                )
                insert_body_num[None] += 1
        valid[random_id] = -1
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
def kernel_distribute_sphere(
    radius_dist: ti.types.ndarray(),
    volume_expect: float,
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    sphere_coords: ti.template(),
    sphere_radii: ti.template(),
    start_point: ti.types.vector(3, float),
    region_size: ti.types.vector(3, float),
    check_in_domain: ti.template(),
) -> float:
    inserted_volume = 0.0
    start_index = insert_body_num[None]
    while inserted_volume < volume_expect:
        sphere_radius = radius_dist[insert_body_num[None] - start_index]
        offset = vec3f([ti.random(), ti.random(), ti.random()]) * region_size
        sphere_coord = start_point + offset

        if check_in_domain(sphere_coord, sphere_radius):
            pvol = 4.0 / 3.0 * PI * sphere_radius * sphere_radius * sphere_radius
            sphere_coords[insert_body_num[None]] = sphere_coord
            sphere_radii[insert_body_num[None]] = sphere_radius
            inserted_volume += pvol
            insert_body_num[None] += 1
            insert_particle_in_neighbor[None] += 1
    return inserted_volume


@ti.kernel
def kernel_distribute_affine_body(
    radius_dist: ti.types.ndarray(),
    volume_expect: float,
    template_volume: float,
    template_bounding_radius: float,
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    body_coords: ti.template(),
    bounding_radii: ti.template(),
    body_orients: ti.template(),
    start_point: ti.types.vector(3, float),
    region_size: ti.types.vector(3, float),
    check_in_domain: ti.template(),
    get_orientation: ti.template(),
) -> float:
    """Randomly distribute overlapping affine bodies to a target volume.

    Only the region boundary is enforced here.  Pairwise overlap is
    intentional and is removed afterwards by the frictionless DiffIPC
    projection in :class:`ParticleGenerator`.
    """
    inserted_volume = 0.0
    start_index = insert_body_num[None]
    while inserted_volume < volume_expect and insert_body_num[None] - start_index < radius_dist.shape[0]:
        local_index = insert_body_num[None] - start_index
        bounding_radius = radius_dist[local_index]
        scale = bounding_radius / ti.max(template_bounding_radius, 1.0e-30)
        offset = vec3f([ti.random(), ti.random(), ti.random()]) * region_size
        coordinate = start_point + offset
        if check_in_domain(coordinate, bounding_radius):
            body_coords[insert_body_num[None]] = coordinate
            bounding_radii[insert_body_num[None]] = bounding_radius
            body_orients[insert_body_num[None]] = get_orientation()
            inserted_volume += template_volume * scale * scale * scale
            insert_body_num[None] += 1
            insert_particle_in_neighbor[None] += 1
    return inserted_volume


@ti.kernel
def kernel_insert_first_multisphere(
    start_point: ti.types.vector(3, float),
    nspheres: int,
    r_equiv: float,
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
    com_pos: ti.types.vector(3, float),
    equiv_rad: float,
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    clump_coords: ti.template(),
    clump_radii: ti.template(),
    clump_orients: ti.template(),
    get_orientation: ti.template(),
    pebble_coords: ti.template(),
    pebble_radii: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    neighbor_position: ti.template(),
    neighbor_radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    insert_particle: ti.template(),
):
    scale_factor = equiv_rad / r_equiv
    clump_orient = get_orientation()
    iclump = insert_body_num[None]
    clump_coords[iclump] = com_pos
    clump_radii[iclump] = equiv_rad
    clump_orients[iclump] = clump_orient

    for pebble in range(nspheres):
        ipebble = insert_particle_in_neighbor[None]
        pebble_coord, pebble_radius = get_actual_clump(
            pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble
        )
        pebble_coords[ipebble] = pebble_coord
        pebble_radii[ipebble] = pebble_radius
        insert_particle(
            cell_num,
            cell_size,
            pebble_coord - start_point,
            pebble_radius,
            insert_particle_in_neighbor,
            neighbor_position,
            neighbor_radius,
            num_particle_in_cell,
            particle_neighbor,
        )
    insert_body_num[None] += 1


@ti.kernel
def kernel_multisphere_poisson_sampling(
    nspheres: int,
    r_equiv: float,
    r_bound: float,
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
    min_rad: float,
    max_rad: float,
    tries_default: int,
    expected_body_num: int,
    start_point: ti.types.vector(3, float),
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    clump_coords: ti.template(),
    clump_radii: ti.template(),
    clump_orients: ti.template(),
    get_orientation: ti.template(),
    pebble_coords: ti.template(),
    pebble_radii: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    position: ti.template(),
    radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_domain: ti.template(),
    overlap: ti.template(),
    insert_particle: ti.template(),
):
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
                if overlap(
                    cell_num,
                    cell_size,
                    pcoord - start_point,
                    pradius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    num_particle_in_cell,
                    particle_neighbor,
                ) == 1 or not check_in_domain(pcoord, pradius):
                    invalid = 1
                    break
            # invalid = validate_clump(start_point, com_pos, scale_factor, clump_orient, nspheres, x_pebble, rad_pebble, is_overlap, check_in_domain, insert_particle_in_neighbor)

            if not invalid and insert_body_num[None] < expected_body_num:
                iclump = insert_body_num[None]
                clump_coords[iclump] = com_pos
                clump_radii[iclump] = equiv_rad
                clump_orients[iclump] = clump_orient
                for pebble in range(nsphere):
                    ipebble = insert_particle_in_neighbor[None]
                    pebble_coord, pebble_radius = get_actual_clump(
                        pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble
                    )
                    pebble_coords[ipebble] = pebble_coord
                    pebble_radii[ipebble] = pebble_radius
                    insert_particle(
                        cell_num,
                        cell_size,
                        pebble_coord - start_point,
                        pebble_radius,
                        insert_particle_in_neighbor,
                        position,
                        radius,
                        num_particle_in_cell,
                        particle_neighbor,
                    )
                insert_body_num[None] += 1
        tries += 1


@ti.kernel
def kernel_multisphere_generate_without_overlap(
    nspheres: int,
    r_equiv: float,
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
    min_rad: float,
    max_rad: float,
    tries_default: int,
    expected_body_num: int,
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    clump_coords: ti.template(),
    clump_radii: ti.template(),
    clump_orients: ti.template(),
    get_orientation: ti.template(),
    pebble_coords: ti.template(),
    pebble_radii: ti.template(),
    cell_num: ti.types.vector(3, int),
    cell_size: float,
    position: ti.template(),
    radius: ti.template(),
    num_particle_in_cell: ti.template(),
    particle_neighbor: ti.template(),
    check_in_domain: ti.template(),
    start_point: ti.types.vector(3, float),
    region_size: ti.types.vector(3, float),
    overlap: ti.template(),
    insert_particle: ti.template(),
):
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
                if overlap(
                    cell_num,
                    cell_size,
                    pcoord - start_point,
                    pradius,
                    insert_particle_in_neighbor,
                    position,
                    radius,
                    num_particle_in_cell,
                    particle_neighbor,
                ) == 1 or not check_in_domain(pcoord, pradius):
                    invalid = 1
                    break
            # invalid = validate_clump(start_point, com_pos, scale_factor, clump_orient, nspheres, x_pebble, rad_pebble, is_overlap, check_in_domain, insert_particle_in_neighbor)

            if invalid == 0:
                iclump = insert_body_num[None]
                clump_coords[iclump] = com_pos
                clump_radii[iclump] = equiv_rad
                clump_orients[iclump] = clump_orient
                for pebble in range(nspheres):
                    ipebble = insert_particle_in_neighbor[None]
                    pebble_coord, pebble_radius = get_actual_clump(
                        pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble
                    )
                    pebble_coords[ipebble] = pebble_coord
                    pebble_radii[ipebble] = pebble_radius
                    insert_particle(
                        cell_num,
                        cell_size,
                        pebble_coord - start_point,
                        pebble_radius,
                        insert_particle_in_neighbor,
                        position,
                        radius,
                        num_particle_in_cell,
                        particle_neighbor,
                    )
                insert_body_num[None] += 1
                break
            count += 1
        if count == tries_default:
            break


@ti.kernel
def kernel_distribute_multisphere(
    nspheres: int,
    r_equiv: float,
    volume_expect: float,
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
    min_rad: float,
    max_rad: float,
    expected_particle_volume: float,
    insert_body_num: ti.template(),
    insert_particle_in_neighbor: ti.template(),
    clump_coords: ti.template(),
    clump_radii: ti.template(),
    clump_orients: ti.template(),
    get_orientation: ti.template(),
    pebble_coords: ti.template(),
    pebble_radii: ti.template(),
    start_point: ti.types.vector(3, float),
    region_size: ti.types.vector(3, float),
    check_in_domain: ti.template(),
) -> float:
    inserted_volume = 0.0
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
                pebble_coord, pebble_radius = get_actual_clump(
                    pebble, com_pos, scale_factor, clump_orient, x_pebble, rad_pebble
                )
                pebble_coords[ipebble + pebble] = pebble_coord
                pebble_radii[ipebble + pebble] = pebble_radius
            pvol = scale_factor * scale_factor * scale_factor * volume_expect
            inserted_volume += pvol
            insert_particle_in_neighbor[None] += nspheres
            insert_body_num[None] += 1
    return inserted_volume


@ti.kernel
def kernel_rebulid_levelset_body(
    start: int,
    number: int,
    rigid: ti.template(),
    group_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    start_node: ti.types.ndarray(),
    end_node: ti.types.ndarray(),
    local_node: ti.types.ndarray(),
    mass: ti.types.ndarray(),
    radius: ti.types.ndarray(),
    mass_center: ti.types.ndarray(),
    acceleration: ti.types.ndarray(),
    angular_moment: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    omega: ti.types.ndarray(),
    quanternion: ti.types.ndarray(),
    inverse_inertia: ti.types.ndarray(),
    is_fix: ti.types.ndarray(),
):
    for sp in range(start, start + number):
        np = sp - start
        rigid[sp]._restart(
            group_id[np],
            material_id[np],
            start_node[np],
            end_node[np],
            local_node[np],
            mass[np],
            radius[np],
            vec3f(mass_center[np, 0], mass_center[np, 1], mass_center[np, 2]),
            vec3f(acceleration[np, 0], acceleration[np, 1], acceleration[np, 2]),
            vec3f(angular_moment[np, 0], angular_moment[np, 1], angular_moment[np, 2]),
            vec3f(velocity[np, 0], velocity[np, 1], velocity[np, 2]),
            vec3f(omega[np, 0], omega[np, 1], omega[np, 2]),
            vec4f(quanternion[np, 0], quanternion[np, 1], quanternion[np, 2], quanternion[np, 3]),
            vec3f(inverse_inertia[np, 0], inverse_inertia[np, 1], inverse_inertia[np, 2]),
            vec3u8(is_fix[np, 0], is_fix[np, 1], is_fix[np, 2]),
        )


@ti.kernel
def kernel_rebulid_implicit_surface_body(
    start: int,
    number: int,
    rigid: ti.template(),
    group_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    template_id: ti.types.ndarray(),
    scale: ti.types.ndarray(),
    mass: ti.types.ndarray(),
    radius: ti.types.ndarray(),
    mass_center: ti.types.ndarray(),
    acceleration: ti.types.ndarray(),
    angular_moment: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    omega: ti.types.ndarray(),
    quanternion: ti.types.ndarray(),
    inverse_inertia: ti.types.ndarray(),
    is_fix: ti.types.ndarray(),
):
    for sp in range(start, start + number):
        np = sp - start
        rigid[sp]._restart(
            group_id[np],
            material_id[np],
            template_id[np],
            scale[np],
            mass[np],
            radius[np],
            vec3f(mass_center[np, 0], mass_center[np, 1], mass_center[np, 2]),
            vec3f(acceleration[np, 0], acceleration[np, 1], acceleration[np, 2]),
            vec3f(angular_moment[np, 0], angular_moment[np, 1], angular_moment[np, 2]),
            vec3f(velocity[np, 0], velocity[np, 1], velocity[np, 2]),
            vec3f(omega[np, 0], omega[np, 1], omega[np, 2]),
            vec4f(quanternion[np, 0], quanternion[np, 1], quanternion[np, 2], quanternion[np, 3]),
            vec3f(inverse_inertia[np, 0], inverse_inertia[np, 1], inverse_inertia[np, 2]),
            vec3u8(is_fix[np, 0], is_fix[np, 1], is_fix[np, 2]),
        )


@ti.kernel
def kernel_rebulid_levelset_grid(start: int, number: int, grid: ti.template(), distance_field: ti.types.ndarray()):
    for sp in range(start, start + number):
        np = sp - start
        grid[sp]._set_grid(distance_field[np])


@ti.kernel
def kernel_rebulid_bounding_sphere(
    start: int,
    number: int,
    particle: ti.template(),
    active: ti.types.ndarray(),
    radius: ti.types.ndarray(),
    center: ti.types.ndarray(),
):
    for sp in range(start, start + number):
        np = sp - start
        particle[sp]._restart(active[np], vec3f(center[np, 0], center[np, 1], center[np, 2]), radius[np])


@ti.kernel
def kernel_rebulid_bounding_box(
    start: int,
    number: int,
    box: ti.template(),
    min_box: ti.types.ndarray(),
    max_box: ti.types.ndarray(),
    start_grid: ti.types.ndarray(),
    grid_num: ti.types.ndarray(),
    grid_space: ti.types.ndarray(),
    scale: ti.types.ndarray(),
    extent: ti.types.ndarray(),
):
    for sp in range(start, start + number):
        np = sp - start
        box[sp]._restart(
            vec3i(grid_num[np, 0], grid_num[np, 1], grid_num[np, 2]),
            vec3f(min_box[np, 0], min_box[np, 1], min_box[np, 2]),
            vec3f(max_box[np, 0], max_box[np, 1], max_box[np, 2]),
            start_grid[np],
            grid_space[np],
            scale[np],
            extent[np],
        )


@ti.kernel
def kernel_rebulid_surface_node(
    start: int, number: int, vertice: ti.template(), vertices: ti.types.ndarray(), parameters: ti.types.ndarray()
):
    for sp in range(start, start + number):
        np = sp - start
        vertice[sp]._restart(parameters[np], vec3f(vertices[np, 0], vertices[np, 1], vertices[np, 2]))


@ti.kernel
def kernel_rebulid_particle(
    start: int,
    number: int,
    particle: ti.template(),
    active: ti.types.ndarray(),
    multisphere_index: ti.types.ndarray(),
    group_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    mass: ti.types.ndarray(),
    radius: ti.types.ndarray(),
    position: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    omega: ti.types.ndarray(),
):
    for sp in range(start, start + number):
        np = sp - start
        particle[sp]._restart(
            active[np],
            multisphere_index[np],
            group_id[np],
            material_id[np],
            mass[np],
            radius[np],
            vec3f(position[np, 0], position[np, 1], position[np, 2]),
            vec3f(velocity[np, 0], velocity[np, 1], velocity[np, 2]),
            vec3f(omega[np, 0], omega[np, 1], omega[np, 2]),
        )


@ti.kernel
def kernel_rebuild_sphere(
    start: int,
    number: int,
    sphere: ti.template(),
    grain_index: ti.types.ndarray(),
    sphere_index: ti.types.ndarray(),
    inv_I: ti.types.ndarray(),
    q: ti.types.ndarray(),
    a: ti.types.ndarray(),
    angmoment: ti.types.ndarray(),
    fix_v: ti.types.ndarray(),
    fix_w: ti.types.ndarray(),
):
    for ssphere in range(start, start + number):
        nsphere = ssphere - start
        sphere[ssphere]._restart(
            grain_index[nsphere],
            sphere_index[nsphere],
            inv_I[nsphere],
            vec4f(q[nsphere, 0], q[nsphere, 1], q[nsphere, 2], q[nsphere, 3]),
            vec3f(a[nsphere, 0], a[nsphere, 1], a[nsphere, 2]),
            vec3f(angmoment[nsphere, 0], angmoment[nsphere, 1], angmoment[nsphere, 2]),
            vec3u8(fix_v[nsphere, 0], fix_v[nsphere, 1], fix_v[nsphere, 2]),
            vec3u8(fix_w[nsphere, 0], fix_w[nsphere, 1], fix_w[nsphere, 2]),
        )


@ti.kernel
def kernel_rebuild_clump(
    start: int,
    number: int,
    clump: ti.template(),
    start_index: ti.types.ndarray(),
    end_index: ti.types.ndarray(),
    mass: ti.types.ndarray(),
    equi_r: ti.types.ndarray(),
    mass_center: ti.types.ndarray(),
    v: ti.types.ndarray(),
    w: ti.types.ndarray(),
    a: ti.types.ndarray(),
    angmoment: ti.types.ndarray(),
    q: ti.types.ndarray(),
    inv_I: ti.types.ndarray(),
):
    for sclump in range(start, start + number):
        nclump = sclump - start
        clump[sclump]._restart(
            start_index[nclump],
            end_index[nclump],
            mass[nclump],
            equi_r[nclump],
            vec3f(mass_center[nclump, 0], mass_center[nclump, 1], mass_center[nclump, 2]),
            vec3f(v[nclump, 0], v[nclump, 1], v[nclump, 2]),
            vec3f(w[nclump, 0], w[nclump, 1], w[nclump, 2]),
            vec3f(a[nclump, 0], a[nclump, 1], a[nclump, 2]),
            vec3f(angmoment[nclump, 0], angmoment[nclump, 1], angmoment[nclump, 2]),
            vec4f(q[nclump, 0], q[nclump, 1], q[nclump, 2], q[nclump, 3]),
            vec3f(inv_I[nclump, 0], inv_I[nclump, 1], inv_I[nclump, 2]),
        )


@ti.kernel
def kernel_rebuild_plane(
    start: int,
    number: int,
    wall: ti.template(),
    active: ti.types.ndarray(),
    wall_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    point: ti.types.ndarray(),
    norm: ti.types.ndarray(),
):
    for swall in range(start, start + number):
        nwall = swall - start
        wall[swall]._restart(
            active[nwall],
            wall_id[nwall],
            material_id[nwall],
            vec3f(point[nwall, 0], point[nwall, 1], point[nwall, 2]),
            vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2]),
        )


@ti.kernel
def kernel_rebuild_facet(
    start: int,
    number: int,
    wall: ti.template(),
    active: ti.types.ndarray(),
    wall_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    point1: ti.types.ndarray(),
    point2: ti.types.ndarray(),
    point3: ti.types.ndarray(),
    norm: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
):
    for swall in range(start, start + number):
        nwall = swall - start
        wall[swall]._restart(
            active[nwall],
            wall_id[nwall],
            material_id[nwall],
            vec3f(point1[nwall, 0], point1[nwall, 1], point1[nwall, 2]),
            vec3f(point2[nwall, 0], point2[nwall, 1], point2[nwall, 2]),
            vec3f(point3[nwall, 0], point3[nwall, 1], point3[nwall, 2]),
            vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2]),
            vec3f(velocity[nwall, 0], velocity[nwall, 1], velocity[nwall, 2]),
        )


@ti.kernel
def kernel_rebuild_patch(
    start: int,
    number: int,
    wall: ti.template(),
    active: ti.types.ndarray(),
    wall_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    point1: ti.types.ndarray(),
    point2: ti.types.ndarray(),
    point3: ti.types.ndarray(),
    norm: ti.types.ndarray(),
    shell_offset: ti.types.ndarray(),
):
    for swall in range(start, start + number):
        nwall = swall - start
        wall[swall]._restart(
            active[nwall],
            wall_id[nwall],
            material_id[nwall],
            vec3f(point1[nwall, 0], point1[nwall, 1], point1[nwall, 2]),
            vec3f(point2[nwall, 0], point2[nwall, 1], point2[nwall, 2]),
            vec3f(point3[nwall, 0], point3[nwall, 1], point3[nwall, 2]),
            vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2]),
            shell_offset[nwall],
        )


@ti.kernel
def kernel_rebuild_servo(
    start: int,
    number: int,
    servo: ti.template(),
    active: ti.types.ndarray(),
    start_index: ti.types.ndarray(),
    end_index: ti.types.ndarray(),
    alpha: ti.types.ndarray(),
    gain: ti.types.ndarray(),
    target_stress: ti.types.ndarray(),
    max_velocity: ti.types.ndarray(),
):
    for sservo in range(start, start + number):
        nservo = sservo - start
        servo[sservo]._restart(
            active[nservo],
            start_index[nservo],
            end_index[nservo],
            alpha[nservo],
            gain[nservo],
            target_stress[nservo],
            max_velocity[nservo],
        )


@ti.kernel
def generate_sphere_from_file(
    min_rad: float,
    max_rad: float,
    group_id: int,
    mat_id: int,
    voxelized_points_np: ti.types.ndarray(),
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
):
    for voxelize in voxelized_points_np:
        radius = min_rad + ti.random() * (max_rad - min_rad)
        new_pos = voxelized_points_np[voxelize]


@ti.kernel
def kernel_add_facet(
    start: int,
    wall_id: int,
    mat_id: int,
    vertices: ti.types.ndarray(),
    faces: ti.types.ndarray(),
    init_v: ti.types.vector(3, float),
    wall: ti.template(),
):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2])
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2])
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2])

        wall[swall].add_material_id(mat_id)
        wall[swall].add_wall_geometry_from_vertices(wall_id, vertice1, vertice2, vertice3, init_v)


@ti.kernel
def kernel_add_facet_files(
    start: int,
    wall_id: int,
    mat_id: int,
    vertices: ti.types.ndarray(),
    faces: ti.types.ndarray(),
    norm: ti.types.ndarray(),
    init_v: ti.types.vector(3, float),
    wall: ti.template(),
):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2])
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2])
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2])

        wall[swall].add_material_id(mat_id)
        wall[swall].add_wall_geometry_from_vertices(wall_id, vertice1, vertice2, vertice3, init_v)
        wall[nwall].norm = vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2])


@ti.kernel
def kernel_add_facet_files_autonorm(
    iscounterclockwise: int,
    start: int,
    wall_id: int,
    mat_id: int,
    vertices: ti.types.ndarray(),
    faces: ti.types.ndarray(),
    init_v: ti.types.vector(3, float),
    wall: ti.template(),
):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2])
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2])
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2])

        if iscounterclockwise == 0:
            vertice1, vertice3 = vertice3, vertice1

        wall[swall].add_material_id(mat_id)
        wall[swall].add_wall_geometry_from_vertices(wall_id, vertice1, vertice2, vertice3, init_v)


@ti.kernel
def kernel_add_patch(
    start: int,
    wall_id: int,
    mat_id: int,
    shell_offset: float,
    vertices: ti.types.ndarray(),
    faces: ti.types.ndarray(),
    norm: ti.types.ndarray(),
    wall: ti.template(),
):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2])
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2])
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2])
        normal = vec3f(norm[nwall, 0], norm[nwall, 1], norm[nwall, 2])

        wall[swall].add_material_id(mat_id)
        wall[swall].add_wall_geometry(wall_id, vertice1, vertice2, vertice3)
        wall[swall].add_shell_offset(shell_offset)
        wall[swall].norm = normal


@ti.kernel
def kernel_add_patch_autonorm(
    iscounterclockwise: int,
    start: int,
    wall_id: int,
    mat_id: int,
    shell_offset: float,
    vertices: ti.types.ndarray(),
    faces: ti.types.ndarray(),
    wall: ti.template(),
):
    for swall in range(start, start + faces.shape[0]):
        nwall = swall - start
        vertice1 = vec3f(vertices[faces[nwall, 0], 0], vertices[faces[nwall, 0], 1], vertices[faces[nwall, 0], 2])
        vertice2 = vec3f(vertices[faces[nwall, 1], 0], vertices[faces[nwall, 1], 1], vertices[faces[nwall, 1], 2])
        vertice3 = vec3f(vertices[faces[nwall, 2], 0], vertices[faces[nwall, 2], 1], vertices[faces[nwall, 2], 2])

        if iscounterclockwise == 0:
            vertice1, vertice3 = vertice3, vertice1

        wall[swall].add_material_id(mat_id)
        wall[swall].add_wall_geometry(wall_id, vertice1, vertice2, vertice3)
        wall[swall].add_shell_offset(shell_offset)


@ti.kernel
def kernel_set_node_coords(
    grid_sum: int,
    grid_size: float,
    gnum: ti.types.vector(3, int),
    start_point: ti.types.vector(3, float),
    node: ti.template(),
):
    for ng in ti.ndrange(grid_sum):
        ig, jg, kg = vectorize_id(ng, gnum)
        node[ng].x = start_point + vec3f(ig * grid_size, jg * grid_size, kg * grid_size)


@ti.kernel
def kernel_place_particles(
    grid_size: float,
    igrid_size: float,
    start_point: ti.types.vector(3, float),
    region_size: ti.types.vector(3, float),
    new_particle_num: int,
    npic: int,
    particle: ti.template(),
    insert_particle_num: ti.template(),
    is_in_region: ti.template(),
):
    pnum = int(region_size * npic * igrid_size)
    ti.loop_config(serialize=True)
    for np in range(new_particle_num):
        ip, jp, kp = vectorize_id(np, pnum)
        particle_pos = (vec3f([ip, jp, kp]) + 0.5) * grid_size / npic + start_point
        if is_in_region(particle_pos):
            old_particle = ti.atomic_add(insert_particle_num[None], 1)
            particle[old_particle] = particle_pos


@ti.kernel
def kernel_add_body(
    particles: ti.template(),
    init_particle_num: int,
    start_particle_num: int,
    end_particle_num: int,
    particle: ti.template(),
    psize: ti.types.vector(3, float),
    particle_volume: float,
    body_id: int,
    material_id: int,
    density: float,
    init_v: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, ti.u8),
):
    for np in range(end_particle_num - start_particle_num):
        particle_id = start_particle_num + np
        particle_num = init_particle_num + np
        particles[particle_num]._set_essential(
            body_id, material_id, density, particle_volume, psize, particle[particle_id], init_v, fix_v
        )


@ti.kernel
def kernel_read_particle_file(
    particles: ti.template(),
    input_particle_num: ti.template(),
    particle_num: int,
    particle: ti.types.ndarray(),
    psize: ti.types.ndarray(),
    particle_volume: ti.types.ndarray(),
    body_id: int,
    material_id: int,
    density: float,
    init_v: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, int),
):
    offset = input_particle_num[None]
    for np in range(particle_num):
        input_particle_num = offset + np
        particles[input_particle_num]._set_essential(
            body_id, material_id, density, particle_volume, psize, particle[np], init_v, fix_v
        )


@ti.kernel
def kernel_reload_particle(
    body_id: int,
    material_id: int,
    active: int,
    mass: ti.types.ndarray(),
    volume: ti.types.ndarray(),
    psize: ti.types.ndarray(),
    position: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    velocity_gradient: ti.types.ndarray(),
    stress: ti.types.ndarray(),
    external_force: ti.types.ndarray(),
    fix_v: ti.types.vector(3, int),
    particle: ti.template(),
    input_particle_num: ti.template(),
    material: ti.template(),
):
    particle_num = body_id.shape[0]
    for np in range(particle_num):
        particle[np]._reload_essential(
            body_id[np],
            material_id[np],
            active[np],
            mass[np],
            volume[np],
            psize[np],
            position[np],
            velocity[np],
            velocity_gradient[np],
            stress[np],
            external_force[np],
            fix_v,
        )
    input_particle_num[None] = particle_num


@ti.kernel
def kernel_add_dem_wall(
    input_wall_num: int,
    no_data: float,
    cell_size: float,
    cell_number: ti.types.vector(2, int),
    digital_elevation: ti.types.ndarray(),
    wall_id: int,
    mat_id: int,
    wall: ti.template(),
) -> int:
    wall_num = 0
    cell_sum = int(cell_number[0] * cell_number[1])
    ti.loop_config(serialize=True)
    for i in range(cell_sum):
        x_ind, y_ind = vectorize_id(i, cell_number)
        if x_ind >= cell_number[0] or y_ind >= cell_number[1]:
            continue

        ind00 = linearize3d(x_ind, y_ind, 0, cell_number + 1)
        ind10 = linearize3d(x_ind + 1, y_ind, 0, cell_number + 1)
        ind01 = linearize3d(x_ind, y_ind + 1, 0, cell_number + 1)
        ind11 = linearize3d(x_ind + 1, y_ind + 1, 0, cell_number + 1)

        height00 = digital_elevation[ind00]
        height10 = digital_elevation[ind10]
        height01 = digital_elevation[ind01]
        height11 = digital_elevation[ind11]

        xy_coord00 = vec3f(x_ind * cell_size, y_ind * cell_size, height00)
        xy_coord10 = vec3f((x_ind + 1) * cell_size, y_ind * cell_size, height10)
        xy_coord01 = vec3f(x_ind * cell_size, (y_ind + 1) * cell_size, height01)
        xy_coord11 = vec3f((x_ind + 1) * cell_size, (y_ind + 1) * cell_size, height11)

        if (
            not equal_to(height00, no_data)
            and not equal_to(height10, no_data)
            and not equal_to(height01, no_data)
            and not equal_to(height11, no_data)
        ):
            wall[input_wall_num + wall_num].add_material_id(mat_id)
            wall[input_wall_num + wall_num].add_wall_geometry(wall_id, xy_coord00, xy_coord10, xy_coord11)
            wall[input_wall_num + wall_num + 1].add_material_id(mat_id)
            wall[input_wall_num + wall_num + 1].add_wall_geometry(wall_id, xy_coord11, xy_coord01, xy_coord00)
            wall_num += 2
            continue
        if (
            equal_to(height00, no_data)
            and not equal_to(height10, no_data)
            and not equal_to(height01, no_data)
            and not equal_to(height11, no_data)
        ):
            wall[input_wall_num + wall_num].add_material_id(mat_id)
            wall[input_wall_num + wall_num].add_wall_geometry(wall_id, xy_coord10, xy_coord11, xy_coord01)
            wall_num += 1
            continue
        if (
            not equal_to(height00, no_data)
            and equal_to(height10, no_data)
            and not equal_to(height01, no_data)
            and not equal_to(height11, no_data)
        ):
            wall[input_wall_num + wall_num].add_material_id(mat_id)
            wall[input_wall_num + wall_num].add_wall_geometry(wall_id, xy_coord00, xy_coord11, xy_coord01)
            wall_num += 1
            continue
        if (
            not equal_to(height00, no_data)
            and not equal_to(height10, no_data)
            and equal_to(height01, no_data)
            and not equal_to(height11, no_data)
        ):
            wall[input_wall_num + wall_num].add_material_id(mat_id)
            wall[input_wall_num + wall_num].add_wall_geometry(wall_id, xy_coord00, xy_coord10, xy_coord11)
            wall_num += 1
            continue
        if (
            not equal_to(height00, no_data)
            and not equal_to(height10, no_data)
            and not equal_to(height01, no_data)
            and equal_to(height11, no_data)
        ):
            wall[input_wall_num + wall_num].add_material_id(mat_id)
            wall[input_wall_num + wall_num].add_wall_geometry(wall_id, xy_coord00, xy_coord10, xy_coord01)
            wall_num += 1
            continue
    return wall_num
