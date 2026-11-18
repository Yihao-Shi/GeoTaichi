import taichi as ti

from src.utils.constants import DBL_EPSILON
from src.utils.TypeDefinition import real, vec2f, vec3f, vec6f, vec2u8, vec3u8, mat2x2, mat3x3
from src.utils.VectorFunction import square_len
from src.utils.Quaternion import theta_to_rotation_matrix, theta_to_rotation_matrix2d


@ti.kernel
def kernel_calc_mass_of_center(coords: ti.types.ndarray()) -> ti.types.vector(3, float):
    position = vec3f(0, 0, 0)
    for np in range(coords.shape[0]):
        position += vec3f(coords[np, 0], coords[np, 1], coords[np, 2])
    return position / coords.shape[0]


@ti.kernel
def kernel_calc_mass_of_center_2d(coords: ti.types.ndarray()) -> ti.types.vector(2, float):
    position = vec2f(0, 0)
    for np in range(coords.shape[0]):
        position += vec2f(coords[np, 0], coords[np, 1])
    return position / coords.shape[0]


@ti.kernel
def kernel_position_rotate(
    target: ti.types.vector(3, float),
    offset: ti.types.vector(3, float),
    body_coords: ti.types.ndarray(),
    start_particle_num: int,
    end_particle_num: int,
):
    R = theta_to_rotation_matrix(target)
    for nb in range(start_particle_num, end_particle_num):
        coords = vec3f(body_coords[nb, 0], body_coords[nb, 1], body_coords[nb, 2])
        coords -= offset
        coords = R @ coords
        coords += offset
        body_coords[nb, 0] = coords[0]
        body_coords[nb, 1] = coords[1]
        body_coords[nb, 2] = coords[2]


@ti.kernel
def kernel_position_rotate_2d(
    target: float,
    offset: ti.types.vector(2, float),
    body_coords: ti.types.ndarray(),
    start_particle_num: int,
    end_particle_num: int,
):
    R = theta_to_rotation_matrix2d(target)
    for nb in range(start_particle_num, end_particle_num):
        coords = vec2f(body_coords[nb, 0], body_coords[nb, 1])
        coords -= offset
        coords = R @ coords
        coords += offset
        body_coords[nb, 0] = coords[0]
        body_coords[nb, 1] = coords[1]


@ti.kernel
def kernel_apply_stress_from_file(start: int, end: int, stress_field: ti.types.ndarray(), particle: ti.template()):
    for np in range(start, end):
        particle[np]._update_stress(
            vec6f(
                stress_field[np, 0],
                stress_field[np, 1],
                stress_field[np, 2],
                stress_field[np, 3],
                stress_field[np, 4],
                stress_field[np, 5],
            )
        )


@ti.kernel
def kernel_apply_vigot_stress(start: int, end: int, stress_field: ti.types.vector(6, float), particle: ti.template()):
    for np in range(start, end):
        particle[np]._update_stress(stress_field)


@ti.kernel
def kernel_apply_pore_pressure(start: int, end: int, pore_pressure: float, particle: ti.template()):
    for np in range(start, end):
        particle[np].pressure += pore_pressure


@ti.kernel
def kernel_activate_cell(
    start_point: ti.types.vector(3, float),
    region_size: ti.types.vector(3, float),
    nodal_coords: ti.template(),
    node_connectivity: ti.template(),
    cell_active: ti.template(),
    is_in_region: ti.template(),
):
    for nc in range(cell_active.shape[0]):
        node_id = node_connectivity[nc]
        cell_center = vec3f([0, 0, 0])
        for i in range(node_id.n):
            cell_center += nodal_coords[node_id[i]]
        cell_center /= node_id.n

        if cell_center[0] < start_point[0] or cell_center[0] > start_point[0] + region_size[0]:
            continue
        if cell_center[1] < start_point[1] or cell_center[1] > start_point[1] + region_size[1]:
            continue
        if cell_center[2] < start_point[2] or cell_center[2] > start_point[2] + region_size[2]:
            continue
        if is_in_region(cell_center, 0.0):
            cell_active[nc] = 1


@ti.kernel
def kernel_fill_particle_in_cell(
    guass_point: ti.types.ndarray(),
    cell_active: ti.template(),
    nodal_coords: ti.template(),
    node_connectivity: ti.template(),
    particle: ti.template(),
    insert_particle_num: ti.template(),
    transform_local_to_global: ti.template(),
):
    for nc in range(cell_active.shape[0]):
        if cell_active[nc] == 1:
            for nparticle in range(guass_point.shape[0]):
                natural_coords = vec3f(guass_point[nparticle, 0], guass_point[nparticle, 1], guass_point[nparticle, 2])
                particle_pos = transform_local_to_global(nc, node_connectivity, nodal_coords, natural_coords)
                old_particle = ti.atomic_add(insert_particle_num[None], 1)
                particle[old_particle] = particle_pos


@ti.func
def get_particle_offset2d(np, pnum):
    ip = np % pnum[0]
    jp = np // pnum[0]
    kp = 0
    return ip, jp, kp


@ti.func
def get_particle_offset3d(np, pnum):
    ip = (np % (pnum[0] * pnum[1])) % pnum[0]
    jp = (np % (pnum[0] * pnum[1])) // pnum[0]
    kp = np // (pnum[0] * pnum[1])
    return ip, jp, kp


@ti.kernel
def kernel_place_particles(
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    start_point: ti.types.vector(3, float),
    region_size: ti.types.vector(3, float),
    new_particle_num: int,
    npic: int,
    particle: ti.template(),
    insert_particle_num: ti.template(),
    is_in_region: ti.template(),
):
    pnum = int((1.0 + 1e-6) * region_size * npic * igrid_size)  # for numerical error
    ti.loop_config(serialize=True)
    for np in range(new_particle_num):
        ip = (np % (pnum[0] * pnum[1])) % pnum[0]
        jp = (np % (pnum[0] * pnum[1])) // pnum[0]
        kp = np // (pnum[0] * pnum[1])
        particle_pos = (vec3f([ip, jp, kp]) + 0.5) * grid_size / npic + start_point
        if is_in_region(particle_pos, 0.0):
            old_particle = ti.atomic_add(insert_particle_num[None], 1)
            particle[old_particle] = particle_pos


@ti.kernel
def kernel_place_particles_2d(
    grid_size: ti.types.vector(2, float),
    igrid_size: ti.types.vector(2, float),
    start_point: ti.types.vector(2, float),
    region_size: ti.types.vector(2, float),
    new_particle_num: int,
    npic: int,
    particle: ti.template(),
    insert_particle_num: ti.template(),
    is_in_region: ti.template(),
):
    pnum = int((1.0 + 1e-6) * region_size * npic * igrid_size)
    ti.loop_config(serialize=True)
    for np in range(new_particle_num):
        ip = np % pnum[0]
        jp = np // pnum[0]
        particle_pos = (vec2f([ip, jp]) + 0.5) * grid_size / npic + start_point
        if is_in_region(particle_pos, 0.0):
            old_particle = ti.atomic_add(insert_particle_num[None], 1)
            particle[old_particle] = particle_pos


@ti.kernel
def kernel_add_body(
    particles: ti.template(),
    init_particle_num: int,
    start_particle_num: int,
    end_particle_num: int,
    particle: ti.template(),
    particle_volume: float,
    body_id: int,
    material_id: int,
    density: ti.types.ndarray(),
    init_v: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, ti.u8),
):
    for np in range(end_particle_num - start_particle_num):
        particle_id = start_particle_num + np
        particle_num = init_particle_num + np
        particles[particle_num]._set_essential(
            particle_num, body_id, material_id, density[np], particle_volume, particle[particle_id], init_v, fix_v
        )


@ti.kernel
def kernel_add_body_2d(
    particles: ti.template(),
    init_particle_num: int,
    start_particle_num: int,
    end_particle_num: int,
    particle: ti.template(),
    particle_volume: float,
    body_id: int,
    material_id: int,
    density: ti.types.ndarray(),
    init_v: ti.types.vector(2, float),
    fix_v: ti.types.vector(2, ti.u8),
):
    for np in range(end_particle_num - start_particle_num):
        particle_id = start_particle_num + np
        particle_num = init_particle_num + np
        particles[particle_num]._set_essential(
            particle_num, body_id, material_id, density[np], particle_volume, particle[particle_id], init_v, fix_v
        )


@ti.kernel
def kernel_add_body_twophase(
    particles: ti.template(),
    init_particle_num: int,
    start_particle_num: int,
    end_particle_num: int,
    particle: ti.template(),
    particle_volume: float,
    body_id: int,
    material_id: int,
    densitys: ti.types.ndarray(),
    densityf: float,
    porosity: float,
    permeability: float,
    init_v: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, ti.u8),
):
    for np in range(end_particle_num - start_particle_num):
        particle_id = start_particle_num + np
        particle_num = init_particle_num + np
        particles[particle_num]._set_essential(
            body_id,
            material_id,
            densitys[np],
            densityf,
            porosity,
            particle_volume,
            particle[particle_id],
            init_v,
            fix_v,
            permeability,
            0.0,
        )
        particles[particle_num].particleID = particle_num


@ti.kernel
def kernel_add_body_twophase_double_point(
    particles: ti.template(),
    init_particle_num: int,
    start_particle_num: int,
    end_particle_num: int,
    particle: ti.template(),
    particle_volume: float,
    body_id: int,
    material_id: int,
    phase: int,
    densitys: ti.types.ndarray(),
    densityf: float,
    porosity: float,
    permeability: float,
    init_v: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, ti.u8),
):
    for np in range(end_particle_num - start_particle_num):
        particle_id = start_particle_num + np
        particle_num = init_particle_num + np
        particles[particle_num]._set_essential_double_point(
            body_id,
            material_id,
            phase,
            densitys[np],
            densityf,
            porosity,
            particle_volume,
            particle[particle_id],
            init_v,
            fix_v,
            permeability,
            0.0,
        )
        particles[particle_num].particleID = particle_num


@ti.kernel
def kernel_add_body_twophase_rigid(
    particles: ti.template(),
    init_particle_num: int,
    start_particle_num: int,
    end_particle_num: int,
    particle: ti.template(),
    particle_volume: float,
    body_id: int,
    material_id: int,
    density: ti.types.ndarray(),
    init_v: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, ti.u8),
):
    for np in range(end_particle_num - start_particle_num):
        particle_id = start_particle_num + np
        particle_num = init_particle_num + np
        particles[particle_num]._set_essential_rigid(
            body_id, material_id, density[np], particle_volume, particle[particle_id], init_v, fix_v, 0.0
        )
        particles[particle_num].particleID = particle_num


@ti.kernel
def kernel_add_body_twophase2d(
    particles: ti.template(),
    init_particle_num: int,
    start_particle_num: int,
    end_particle_num: int,
    particle: ti.template(),
    particle_volume: float,
    body_id: int,
    material_id: int,
    densitys: ti.types.ndarray(),
    densityf: float,
    porosity: float,
    permeability: float,
    init_v: ti.types.vector(2, float),
    fix_v: ti.types.vector(2, ti.u8),
    axis_offset: float,
):
    for np in range(end_particle_num - start_particle_num):
        particle_id = start_particle_num + np
        particle_num = init_particle_num + np
        particles[particle_num]._set_essential(
            body_id,
            material_id,
            densitys[np],
            densityf,
            porosity,
            particle_volume,
            particle[particle_id],
            init_v,
            fix_v,
            permeability,
            axis_offset,
        )
        particles[particle_num].particleID = particle_num


@ti.kernel
def kernel_add_body_twophase_double_point2d(
    particles: ti.template(),
    init_particle_num: int,
    start_particle_num: int,
    end_particle_num: int,
    particle: ti.template(),
    particle_volume: float,
    body_id: int,
    material_id: int,
    phase: int,
    densitys: ti.types.ndarray(),
    densityf: float,
    porosity: float,
    permeability: float,
    init_v: ti.types.vector(2, float),
    fix_v: ti.types.vector(2, ti.u8),
    axis_offset: float,
):
    for np in range(end_particle_num - start_particle_num):
        particle_id = start_particle_num + np
        particle_num = init_particle_num + np
        particles[particle_num]._set_essential_double_point(
            body_id,
            material_id,
            phase,
            densitys[np],
            densityf,
            porosity,
            particle_volume,
            particle[particle_id],
            init_v,
            fix_v,
            permeability,
            axis_offset,
        )
        particles[particle_num].particleID = particle_num


@ti.kernel
def kernel_add_body_2d_rigid(
    particles: ti.template(),
    init_particle_num: int,
    start_particle_num: int,
    end_particle_num: int,
    particle: ti.template(),
    particle_volume: float,
    body_id: int,
    material_id: int,
    density: ti.types.ndarray(),
    init_v: ti.types.vector(2, float),
    fix_v: ti.types.vector(2, ti.u8),
    axis_offset: float,
):
    for np in range(end_particle_num - start_particle_num):
        particle_id = start_particle_num + np
        particle_num = init_particle_num + np
        particles[particle_num]._set_essential_rigid(
            body_id, material_id, density[np], particle_volume, particle[particle_id], init_v, fix_v, axis_offset
        )
        particles[particle_num].particleID = particle_num


@ti.kernel
def kernel_read_particle_file(
    particles: ti.template(),
    input_particle_num: int,
    particle_num: int,
    particle: ti.types.ndarray(),
    particle_volume: ti.types.ndarray(),
    body_id: int,
    material_id: int,
    density: ti.types.ndarray(),
    init_v: ti.types.vector(3, float),
    fix_v: ti.types.vector(3, int),
):
    for np in range(particle_num):
        i = input_particle_num + np
        particles[i]._set_essential(
            i,
            body_id,
            material_id,
            density[np],
            particle_volume[np],
            vec3f(particle[np, 0], particle[np, 1], particle[np, 2]),
            init_v,
            fix_v,
        )


@ti.kernel
def kernel_read_particle_file_2d(
    particles: ti.template(),
    input_particle_num: int,
    particle_num: int,
    particle: ti.types.ndarray(),
    particle_volume: ti.types.ndarray(),
    body_id: int,
    material_id: int,
    density: ti.types.ndarray(),
    init_v: ti.types.vector(2, float),
    fix_v: ti.types.vector(2, int),
):
    for np in range(particle_num):
        i = input_particle_num + np
        particles[i]._set_essential(
            i,
            body_id,
            material_id,
            density[np],
            particle_volume[np],
            vec2f(particle[np, 0], particle[np, 1]),
            init_v,
            fix_v,
        )


@ti.kernel
def kernel_rebulid_particle(
    particle_number: int,
    particle: ti.template(),
    is_rigid: ti.template(),
    particle_id: ti.types.ndarray(),
    body_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    active: ti.types.ndarray(),
    mass: ti.types.ndarray(),
    position: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    volume: ti.types.ndarray(),
    stress: ti.types.ndarray(),
    velocity_gradient: ti.types.ndarray(),
    fix_v: ti.types.ndarray(),
):
    for np in range(particle_number):
        if material_id[np] == 0:
            is_rigid[body_id[np]] = 1
        restored_position = ti.Vector.zero(real, ti.static(particle.x.n))
        restored_velocity = ti.Vector.zero(real, ti.static(particle.v.n))
        restored_fix = ti.Vector.zero(ti.u8, ti.static(particle.fix_v.n))
        restored_gradient = ti.Matrix.zero(
            real, ti.static(particle.velocity_gradient.n), ti.static(particle.velocity_gradient.m)
        )
        for d in ti.static(range(particle.x.n)):
            restored_position[d] = position[np, d]
            restored_velocity[d] = velocity[np, d]
            restored_fix[d] = fix_v[np, d]
        for j, k in ti.static(ti.ndrange(particle.velocity_gradient.n, particle.velocity_gradient.m)):
            restored_gradient[j, k] = velocity_gradient[np, j, k]
        particle[np]._restart(
            particle_id[np],
            body_id[np],
            material_id[np],
            active[np],
            mass[np],
            restored_position,
            restored_velocity,
            volume[np],
            vec6f(stress[np, 0], stress[np, 1], stress[np, 2], stress[np, 3], stress[np, 4], stress[np, 5]),
            restored_gradient,
            restored_fix,
        )


@ti.kernel
def kernel_rebulid_incompressible_particle(
    particle_number: int,
    particle: ti.template(),
    is_rigid: ti.template(),
    particle_id: ti.types.ndarray(),
    body_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    active: ti.types.ndarray(),
    coupling: ti.types.ndarray(),
    mass: ti.types.ndarray(),
    position: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    volume: ti.types.ndarray(),
    pressure: ti.types.ndarray(),
    velocity_gradient: ti.types.ndarray(),
    fix_v: ti.types.ndarray(),
):
    for np in range(particle_number):
        if material_id[np] == 0:
            is_rigid[body_id[np]] = 1
        particle[np].particleID = int(particle_id[np])
        particle[np].bodyID = ti.u8(body_id[np])
        particle[np].materialID = ti.u8(material_id[np])
        particle[np].active = ti.u8(active[np])
        particle[np].coupling = ti.u8(coupling[np])
        particle[np].m = float(mass[np])
        particle[np].x = vec3f(position[np, 0], position[np, 1], position[np, 2])
        particle[np].v = vec3f(velocity[np, 0], velocity[np, 1], velocity[np, 2])
        particle[np].vol = float(volume[np])
        particle[np].pressure = float(pressure[np])
        particle[np].velocity_gradient = mat3x3(
            velocity_gradient[np, 0, 0],
            velocity_gradient[np, 0, 1],
            velocity_gradient[np, 0, 2],
            velocity_gradient[np, 1, 0],
            velocity_gradient[np, 1, 1],
            velocity_gradient[np, 1, 2],
            velocity_gradient[np, 2, 0],
            velocity_gradient[np, 2, 1],
            velocity_gradient[np, 2, 2],
        )
        particle[np].fix_v = vec3u8(fix_v[np, 0], fix_v[np, 1], fix_v[np, 2])


@ti.kernel
def kernel_rebulid_incompressible_particle_2d(
    particle_number: int,
    particle: ti.template(),
    is_rigid: ti.template(),
    particle_id: ti.types.ndarray(),
    body_id: ti.types.ndarray(),
    material_id: ti.types.ndarray(),
    active: ti.types.ndarray(),
    coupling: ti.types.ndarray(),
    mass: ti.types.ndarray(),
    position: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    volume: ti.types.ndarray(),
    pressure: ti.types.ndarray(),
    velocity_gradient: ti.types.ndarray(),
    fix_v: ti.types.ndarray(),
):
    for np in range(particle_number):
        if material_id[np] == 0:
            is_rigid[body_id[np]] = 1
        particle[np].particleID = int(particle_id[np])
        particle[np].bodyID = ti.u8(body_id[np])
        particle[np].materialID = ti.u8(material_id[np])
        particle[np].active = ti.u8(active[np])
        particle[np].coupling = ti.u8(coupling[np])
        particle[np].m = float(mass[np])
        particle[np].x = vec2f(position[np, 0], position[np, 1])
        particle[np].v = vec2f(velocity[np, 0], velocity[np, 1])
        particle[np].vol = float(volume[np])
        particle[np].pressure = float(pressure[np])
        particle[np].velocity_gradient = mat2x2(
            velocity_gradient[np, 0, 0],
            velocity_gradient[np, 0, 1],
            velocity_gradient[np, 1, 0],
            velocity_gradient[np, 1, 1],
        )
        particle[np].fix_v = vec2u8(fix_v[np, 0], fix_v[np, 1])


@ti.kernel
def kernel_rebulid_particle_coupling(
    particle_number: int, particle: ti.template(), coupling: ti.types.ndarray(), radius: ti.types.ndarray()
):
    for np in range(particle_number):
        particle[np].coupling = ti.u8(coupling[np])
        particle[np].rad = radius[np]


@ti.kernel
def kernel_rebulid_particle_neighbor_detection(
    particle_number: int,
    particle: ti.template(),
    free_surface: ti.types.ndarray(),
    mass_density: ti.types.ndarray(),
    normal: ti.types.ndarray(),
):
    for np in range(particle_number):
        particle[np].free_surface = ti.u8(free_surface[np])
        particle[np].mass_density = mass_density[np]
        particle[np].normal = vec3f(normal[np, 0], normal[np, 1], normal[np, 2])


@ti.kernel
def kernel_delete_particle_slots_in_region(
    insert_particle_num: ti.template(), particle: ti.template(), is_in_region: ti.template()
):
    for np in range(insert_particle_num[None]):
        if is_in_region(particle[np]):
            particle[np] = ti.math.nan
    ti.sync()

    remaining_particle = 0
    ti.loop_config(serialize=True)
    for np in range(insert_particle_num[None]):
        if ti.math.isnan(square_len(particle[np])):
            particle[remaining_particle] = particle[np]
            remaining_particle += 1
    insert_particle_num[None] = remaining_particle


# Soft-particle level-set insertion kernels

from src.dem.generator.InsertionKernel import (
    create_bounding_box,
    create_deformable_grids,
)
from src.utils.constants import PI
from src.utils.Quaternion import set_from_euler, set_to_rotate
from src.utils.TypeDefinition import vec3f, vec3i


@ti.kernel
def kernel_prepare_soft_grid_topology(
    soft_body: ti.template(),
    soft_start: int,
    body_count: int,
    grid_start: int,
    mpm_grid_start: int,
    grid_sum: int,
    compact_offset_i: int,
    compact_offset_j: int,
    compact_offset_k: int,
    compact_shape_i: int,
    compact_shape_j: int,
    compact_shape_k: int,
):
    compact_grid_sum = compact_shape_i * compact_shape_j * compact_shape_k
    compact_grid_offset = vec3i(compact_offset_i, compact_offset_j, compact_offset_k)
    compact_grid_shape = vec3i(compact_shape_i, compact_shape_j, compact_shape_k)
    for nb in range(body_count):
        soft_body[soft_start + nb]._add_grid_index(
            grid_start + nb * grid_sum,
            mpm_grid_start + nb * compact_grid_sum,
            grid_sum,
            compact_grid_sum,
            compact_grid_offset,
            compact_grid_shape,
        )


@ti.kernel
def upload_soft_velocity_constraints(
    start: int,
    mpm_grid_start: int,
    local_node: ti.types.ndarray(),
    constraint: ti.template(),
):
    for offset in range(local_node.shape[0]):
        constraint[start + offset] = mpm_grid_start + local_node[offset]


@ti.kernel
def kernel_initialize_level_set_soft_body(
    soft_body: ti.template(),
    rigid_body: ti.template(),
    bounding_box: ti.template(),
    bounding_sphere: ti.template(),
    material: ti.template(),
    body_id: int,
    soft_id: int,
    point_start: int,
    point_count: int,
    grid_start: int,
    mpm_grid_start: int,
    vertice_num: int,
    surface_num: int,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    surface_sum: int,
    reference_surface_area: float,
    grid_sum: int,
    space: float,
    gnum: ti.types.vector(3, int),
    extent: int,
    shape_min: ti.types.vector(3, float),
    shape_max: ti.types.vector(3, float),
    shape_radius: float,
    template_volume: float,
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
    template_point_start: int,
    template_surface_start: int,
    template_sdf_start: int,
    grid_type: int,
    template_grid_space: float,
):
    density = material[mat_id]._get_density()
    volume = template_volume * scale_factor**3
    mass = density * volume
    inv_inertia = 1.0 / (inertia * density * scale_factor**5)
    orientation = get_orientation()
    q = set_from_euler(*orientation)
    rotation_matrix = set_to_rotate(q)

    create_bounding_box(
        body_id,
        scale_factor,
        bounding_box,
        min_box,
        max_box,
        grid_start,
        space,
        gnum,
        extent,
    )
    bounding_box[body_id]._set_reference_surface_area(reference_surface_area)
    bounding_box[body_id]._set_shape_box(shape_min, shape_max)
    bounding_box[body_id]._set_shape_radius(shape_radius)
    bounding_sphere[body_id]._set_deformed_shape(
        com_pos,
        rotation_matrix,
        shape_min,
        shape_max,
        shape_radius,
        bounding_box[body_id].grid_space,
    )

    rigid_body[body_id]._add_body_attribute(
        com_pos,
        volume,
        equiv_rad,
        inv_inertia,
        q,
    )
    rigid_body[body_id]._add_surface_index(
        surface_num,
        surface_num + surface_sum,
        vertice_num,
    )
    rigid_body[body_id]._add_body_properties(mat_id, group_id, density)
    rigid_body[body_id]._add_body_kinematic(init_v, init_w, is_fix)
    rigid_body[body_id]._mark_soft_body(soft_id)

    soft_body[soft_id]._restart(
        body_id,
        point_start,
        point_start + point_count,
        group_id,
        mat_id,
    )
    soft_body[soft_id]._add_body_attribute(mass, com_pos)
    soft_body[soft_id]._add_template_support(
        template_point_start,
        template_surface_start,
        template_sdf_start,
        grid_type,
        scale_factor,
        template_grid_space,
        rotation_matrix,
    )
    soft_body[soft_id].v = init_v
    soft_body[soft_id]._add_surface_index(
        surface_num,
        surface_num + surface_sum,
        vertice_num,
    )
    soft_body[soft_id]._add_surface_point_index(point_start, point_start)


@ti.kernel
def kernel_initialize_level_set_soft_body_grids(
    soft_body: ti.template(),
    grid: ti.template(),
    soft_grid: ti.template(),
    soft_grid_owner: ti.template(),
    soft_grid_local: ti.template(),
    soft_id: int,
    grid_start: int,
    mpm_grid_start: int,
    grid_sum: int,
    distance_fields: ti.types.ndarray(),
):
    for i in range(grid_sum):
        grid[grid_start + i]._set_grid(distance_fields[i])
    for i in range(soft_body[soft_id].mpmGridNum):
        mpm_grid = mpm_grid_start + i
        soft_grid[mpm_grid]._grid_reset()
        soft_grid_owner[mpm_grid] = soft_id
        soft_grid_local[mpm_grid] = i


@ti.kernel
def kernel_initialize_level_set_soft_body_surface(
    rigid_body: ti.template(),
    master: ti.template(),
    surface_node: ti.template(),
    body_id: int,
    vertice_num: int,
    surface_num: int,
    surface_sum: int,
    scale_factor: float,
    surface_nodes: ti.types.ndarray(),
    parameters: ti.types.ndarray(),
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
):
    com_pos = rigid_body[body_id].mass_center
    rotation_matrix = set_to_rotate(rigid_body[body_id].q)
    for i in range(surface_sum):
        master[i + surface_num] = body_id
        local_node = vertice_num + i
        template_x = vec3f(
            surface_nodes[i, 0],
            surface_nodes[i, 1],
            surface_nodes[i, 2],
        )
        local_x_scaled = scale_factor * template_x
        surface_global = com_pos + rotation_matrix @ local_x_scaled
        surface_node[local_node]._set_surface_node(template_x)
        surface_node[local_node]._set_coefficient(parameters[i])
        surface_node[local_node].v = init_v + init_w.cross(surface_global - com_pos)


@ti.kernel
def kernel_initialize_level_set_soft_body_points(
    soft_body: ti.template(),
    material_point: ti.template(),
    rigid_body: ti.template(),
    bounding_box: ti.template(),
    material: ti.template(),
    grid: ti.template(),
    soft_surface_point_id: ti.template(),
    body_id: int,
    soft_id: int,
    point_start: int,
    point_count: int,
    scale_factor: float,
    material_points: ti.types.ndarray(),
    point_volumes: ti.types.ndarray(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
):
    density = material[mat_id]._get_density()
    com_pos = rigid_body[body_id].mass_center
    rotation_matrix = set_to_rotate(rigid_body[body_id].q)
    for i in range(point_count):
        local_x = vec3f(
            material_points[i, 0],
            material_points[i, 1],
            material_points[i, 2],
        )
        local_x_scaled = scale_factor * local_x
        x0 = com_pos + rotation_matrix @ local_x_scaled
        pid = point_start + i
        pvolume = point_volumes[i] * scale_factor**3
        pmass = density * pvolume
        point_padding = 0.5 * ti.pow(pvolume, 1.0 / 3.0)
        is_surface_point = (
            ti.abs(bounding_box[body_id].distance(local_x_scaled, grid))
            <= bounding_box[body_id].grid_space + point_padding
        )
        material_point[pid]._add_point(
            body_id,
            mat_id,
            group_id,
            x0,
            x0,
            init_v + init_w.cross(x0 - com_pos),
            pmass,
            pvolume,
        )
        if is_surface_point:
            slot = ti.atomic_add(soft_body[soft_id].surfacePointEnd, 1)
            soft_surface_point_id[slot] = pid
            material_point[pid]._set_surface_weight(ti.pow(pvolume, 2.0 / 3.0))


@ti.kernel
def kernel_initialize_packed_level_set_soft_body(
    soft_body: ti.template(),
    rigid_body: ti.template(),
    bounding_box: ti.template(),
    bounding_sphere: ti.template(),
    material: ti.template(),
    body_start: int,
    soft_start: int,
    point_start: int,
    point_count: int,
    grid_start: int,
    mpm_grid_start: int,
    vertice_num: int,
    surface_num: int,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    surface_sum: int,
    reference_surface_area: float,
    grid_sum: int,
    space: float,
    gnum: ti.types.vector(3, int),
    extent: int,
    base_shape_min: ti.types.vector(3, float),
    base_shape_max: ti.types.vector(3, float),
    base_shape_radius: float,
    template_volume: float,
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
    template_point_start: int,
    template_surface_start: int,
    template_sdf_start: int,
    grid_type: int,
    template_grid_space: float,
    coordinates_are_mass_centers: ti.template(),
):
    density = material[mat_id]._get_density()
    for nb in range(end_body_num - start_body_num):
        packed_id = start_body_num + nb
        body_id = body_start + nb
        soft_id = soft_start + nb
        compact_grid_sum = soft_body[soft_id].mpmGridNum
        local_point_start = point_start + nb * point_count
        local_grid_start = grid_start + nb * grid_sum
        local_mpm_grid_start = mpm_grid_start + nb * compact_grid_sum
        local_surface_start = surface_num + nb * surface_sum
        local_vertice_start = vertice_num + nb * surface_sum

        bounding_x, bounding_r = coords[packed_id], radii[packed_id]
        scale_factor = bounding_r / r_bound
        q = set_from_euler(*orients[packed_id])
        rotation_matrix = set_to_rotate(q)
        com_pos = bounding_x
        if ti.static(not coordinates_are_mass_centers):
            com_pos = bounding_x - scale_factor * rotation_matrix @ x_bound
        equiv_rad = scale_factor * eqradius
        volume = template_volume * scale_factor**3
        mass = density * volume
        inv_inertia = 1.0 / (inertia * density * scale_factor**5)
        shape_min = scale_factor * base_shape_min
        shape_max = scale_factor * base_shape_max
        shape_radius = scale_factor * base_shape_radius

        create_bounding_box(
            body_id,
            scale_factor,
            bounding_box,
            min_box,
            max_box,
            local_grid_start,
            space,
            gnum,
            extent,
        )
        bounding_box[body_id]._set_reference_surface_area(
            reference_surface_area,
        )
        bounding_box[body_id]._set_shape_box(shape_min, shape_max)
        bounding_box[body_id]._set_shape_radius(shape_radius)
        bounding_sphere[body_id]._set_deformed_shape(
            com_pos,
            rotation_matrix,
            shape_min,
            shape_max,
            shape_radius,
            bounding_box[body_id].grid_space,
        )

        rigid_body[body_id]._add_body_attribute(
            com_pos,
            volume,
            equiv_rad,
            inv_inertia,
            q,
        )
        rigid_body[body_id]._add_surface_index(
            local_surface_start,
            local_surface_start + surface_sum,
            local_vertice_start,
        )
        rigid_body[body_id]._add_body_properties(mat_id, group_id, density)
        rigid_body[body_id]._add_body_kinematic(init_v, init_w, is_fix)
        rigid_body[body_id]._mark_soft_body(soft_id)

        soft_body[soft_id]._restart(
            body_id,
            local_point_start,
            local_point_start + point_count,
            group_id,
            mat_id,
        )
        soft_body[soft_id]._add_body_attribute(mass, com_pos)
        soft_body[soft_id]._add_template_support(
            template_point_start,
            template_surface_start,
            template_sdf_start,
            grid_type,
            scale_factor,
            template_grid_space,
            rotation_matrix,
        )
        soft_body[soft_id].v = init_v
        soft_body[soft_id]._add_surface_index(
            local_surface_start,
            local_surface_start + surface_sum,
            local_vertice_start,
        )
        soft_body[soft_id]._add_surface_point_index(
            local_point_start,
            local_point_start,
        )


@ti.kernel
def kernel_initialize_packed_level_set_soft_body_grids(
    soft_body: ti.template(),
    grid: ti.template(),
    soft_grid: ti.template(),
    soft_grid_owner: ti.template(),
    soft_grid_local: ti.template(),
    soft_start: int,
    body_count: int,
    grid_start: int,
    mpm_grid_start: int,
    grid_sum: int,
    compact_grid_sum: int,
    distance_fields: ti.types.ndarray(),
):
    for flat in range(body_count * grid_sum):
        nb = flat // grid_sum
        local = flat - nb * grid_sum
        grid[grid_start + nb * grid_sum + local]._set_grid(
            distance_fields[local],
        )
    for flat in range(body_count * compact_grid_sum):
        nb = flat // compact_grid_sum
        local = flat - nb * compact_grid_sum
        mpm_grid = mpm_grid_start + nb * compact_grid_sum + local
        soft_grid[mpm_grid]._grid_reset()
        soft_grid_owner[mpm_grid] = soft_start + nb
        soft_grid_local[mpm_grid] = local


@ti.kernel
def kernel_initialize_packed_level_set_soft_body_surface(
    soft_body: ti.template(),
    rigid_body: ti.template(),
    master: ti.template(),
    surface_node: ti.template(),
    body_start: int,
    soft_start: int,
    body_count: int,
    vertice_num: int,
    surface_num: int,
    surface_sum: int,
    surface_nodes: ti.types.ndarray(),
    parameters: ti.types.ndarray(),
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
):
    for flat in range(body_count * surface_sum):
        nb = flat // surface_sum
        local = flat - nb * surface_sum
        body_id = body_start + nb
        soft_id = soft_start + nb
        local_surface_start = surface_num + nb * surface_sum
        local_vertice_start = vertice_num + nb * surface_sum
        com_pos = rigid_body[body_id].mass_center
        rotation_matrix = set_to_rotate(rigid_body[body_id].q)
        template_x = vec3f(
            surface_nodes[local, 0],
            surface_nodes[local, 1],
            surface_nodes[local, 2],
        )
        surface_global = com_pos + rotation_matrix @ (soft_body[soft_id].scale * template_x)
        master[local_surface_start + local] = body_id
        local_node = local_vertice_start + local
        surface_node[local_node]._set_surface_node(template_x)
        surface_node[local_node]._set_coefficient(parameters[local])
        surface_node[local_node].v = init_v + init_w.cross(surface_global - com_pos)


@ti.kernel
def kernel_initialize_packed_level_set_soft_body_points(
    soft_body: ti.template(),
    material_point: ti.template(),
    rigid_body: ti.template(),
    bounding_box: ti.template(),
    material: ti.template(),
    grid: ti.template(),
    soft_surface_point_id: ti.template(),
    body_start: int,
    soft_start: int,
    body_count: int,
    point_start: int,
    point_count: int,
    material_points: ti.types.ndarray(),
    point_volumes: ti.types.ndarray(),
    group_id: int,
    mat_id: int,
    init_v: ti.types.vector(3, float),
    init_w: ti.types.vector(3, float),
):
    density = material[mat_id]._get_density()
    for flat in range(body_count * point_count):
        nb = flat // point_count
        local = flat - nb * point_count
        body_id = body_start + nb
        soft_id = soft_start + nb
        scale_factor = soft_body[soft_id].scale
        com_pos = rigid_body[body_id].mass_center
        rotation_matrix = set_to_rotate(rigid_body[body_id].q)
        local_x = vec3f(
            material_points[local, 0],
            material_points[local, 1],
            material_points[local, 2],
        )
        local_x_scaled = scale_factor * local_x
        x0 = com_pos + rotation_matrix @ local_x_scaled
        pid = point_start + nb * point_count + local
        pvolume = point_volumes[local] * scale_factor**3
        pmass = density * pvolume
        point_padding = 0.5 * ti.pow(pvolume, 1.0 / 3.0)
        is_surface_point = (
            ti.abs(bounding_box[body_id].distance(local_x_scaled, grid))
            <= bounding_box[body_id].grid_space + point_padding
        )
        material_point[pid]._add_point(
            body_id,
            mat_id,
            group_id,
            x0,
            x0,
            init_v + init_w.cross(x0 - com_pos),
            pmass,
            pvolume,
        )
        if is_surface_point:
            slot = ti.atomic_add(soft_body[soft_id].surfacePointEnd, 1)
            soft_surface_point_id[slot] = pid
            material_point[pid]._set_surface_weight(ti.pow(pvolume, 2.0 / 3.0))


@ti.kernel
def kernel_create_level_set_soft_body(
    soft_body: ti.template(),
    material_point: ti.template(),
    rigid_body: ti.template(),
    bounding_box: ti.template(),
    bounding_sphere: ti.template(),
    master: ti.template(),
    material: ti.template(),
    grid: ti.template(),
    soft_grid: ti.template(),
    soft_grid_owner: ti.template(),
    soft_grid_local: ti.template(),
    surface_node: ti.template(),
    soft_surface_point_id: ti.template(),
    body_id: int,
    soft_id: int,
    point_start: int,
    point_count: int,
    grid_start: int,
    mpm_grid_start: int,
    vertice_num: int,
    surface_num: int,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    surface_sum: int,
    reference_surface_area: float,
    surface_nodes: ti.types.ndarray(),
    parameters: ti.types.ndarray(),
    grid_sum: int,
    space: float,
    gnum: ti.types.vector(3, int),
    extent: int,
    distance_fields: ti.types.ndarray(),
    material_points: ti.types.ndarray(),
    point_volumes: ti.types.ndarray(),
    shape_min: ti.types.vector(3, float),
    shape_max: ti.types.vector(3, float),
    shape_radius: float,
    template_volume: float,
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
    template_point_start: int,
    template_surface_start: int,
    template_sdf_start: int,
    grid_type: int,
    template_grid_space: float,
):
    density = material[mat_id]._get_density()
    volume = template_volume * scale_factor**3
    mass = density * volume
    inv_inertia = 1.0 / (inertia * density * scale_factor**5)
    orientation = get_orientation()
    q = set_from_euler(*orientation)
    rotation_matrix = set_to_rotate(q)
    compact_grid_sum = soft_body[soft_id].mpmGridNum

    create_bounding_box(body_id, scale_factor, bounding_box, min_box, max_box, grid_start, space, gnum, extent)
    bounding_box[body_id]._set_reference_surface_area(reference_surface_area)
    create_deformable_grids(grid_start, grid_sum, grid, 1.0, distance_fields)

    for i in range(compact_grid_sum):
        mpm_grid = mpm_grid_start + i
        soft_grid[mpm_grid]._grid_reset()
        soft_grid_owner[mpm_grid] = soft_id
        soft_grid_local[mpm_grid] = i

    bounding_box[body_id]._set_shape_box(shape_min, shape_max)
    bounding_box[body_id]._set_shape_radius(shape_radius)
    for i in range(surface_sum):
        master[i + surface_num] = body_id
        local_node = vertice_num + i
        template_x = vec3f(surface_nodes[i, 0], surface_nodes[i, 1], surface_nodes[i, 2])
        local_x_scaled = scale_factor * template_x
        surface_global = com_pos + rotation_matrix @ local_x_scaled
        surface_node[local_node]._set_surface_node(template_x)
        surface_node[local_node]._set_coefficient(parameters[i])
        surface_node[local_node].v = init_v + init_w.cross(surface_global - com_pos)
    soft_body[soft_id]._add_surface_point_index(point_start, point_start)
    for i in range(point_count):
        local_x = vec3f(material_points[i, 0], material_points[i, 1], material_points[i, 2])
        local_x_scaled = scale_factor * local_x
        x0 = com_pos + rotation_matrix @ local_x_scaled
        pid = point_start + i
        pvolume = point_volumes[i] * scale_factor * scale_factor * scale_factor
        pmass = density * pvolume
        point_padding = 0.5 * ti.pow(pvolume, 1.0 / 3.0)
        is_surface_point = (
            ti.abs(bounding_box[body_id].distance(local_x_scaled, grid))
            <= bounding_box[body_id].grid_space + point_padding
        )
        material_point[pid]._add_point(
            body_id, mat_id, group_id, x0, x0, init_v + init_w.cross(x0 - com_pos), pmass, pvolume
        )
        if is_surface_point:
            slot = ti.atomic_add(soft_body[soft_id].surfacePointEnd, 1)
            soft_surface_point_id[slot] = pid
            material_point[pid]._set_surface_weight(ti.pow(pvolume, 2.0 / 3.0))
    bounding_sphere[body_id]._set_deformed_shape(
        com_pos, rotation_matrix, shape_min, shape_max, shape_radius, bounding_box[body_id].grid_space
    )
    rigid_body[body_id]._add_body_attribute(com_pos, volume, equiv_rad, inv_inertia, q)
    rigid_body[body_id]._add_surface_index(surface_num, surface_num + surface_sum, vertice_num)
    rigid_body[body_id]._add_body_properties(mat_id, group_id, density)
    rigid_body[body_id]._add_body_kinematic(init_v, init_w, is_fix)
    rigid_body[body_id]._mark_soft_body(soft_id)

    soft_body[soft_id]._restart(body_id, point_start, point_start + point_count, group_id, mat_id)
    soft_body[soft_id]._add_body_attribute(mass, com_pos)
    soft_body[soft_id]._add_template_support(
        template_point_start,
        template_surface_start,
        template_sdf_start,
        grid_type,
        scale_factor,
        template_grid_space,
        rotation_matrix,
    )
    soft_body[soft_id].v = init_v
    soft_body[soft_id]._add_surface_index(surface_num, surface_num + surface_sum, vertice_num)


@ti.kernel
def kernel_add_levelset_soft_body_packing(
    soft_body: ti.template(),
    material_point: ti.template(),
    rigid_body: ti.template(),
    bounding_box: ti.template(),
    bounding_sphere: ti.template(),
    master: ti.template(),
    material: ti.template(),
    grid: ti.template(),
    soft_grid: ti.template(),
    soft_grid_owner: ti.template(),
    soft_grid_local: ti.template(),
    surface_node: ti.template(),
    soft_surface_point_id: ti.template(),
    body_start: int,
    soft_start: int,
    point_start: int,
    point_count: int,
    grid_start: int,
    mpm_grid_start: int,
    vertice_num: int,
    surface_num: int,
    min_box: ti.types.vector(3, float),
    max_box: ti.types.vector(3, float),
    r_bound: float,
    x_bound: ti.types.vector(3, float),
    surface_sum: int,
    reference_surface_area: float,
    surface_nodes: ti.types.ndarray(),
    parameters: ti.types.ndarray(),
    grid_sum: int,
    space: float,
    gnum: ti.types.vector(3, int),
    extent: int,
    distance_fields: ti.types.ndarray(),
    material_points: ti.types.ndarray(),
    point_volumes: ti.types.ndarray(),
    template_volume: float,
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
    template_point_start: int,
    template_surface_start: int,
    template_sdf_start: int,
    grid_type: int,
    template_grid_space: float,
    coordinates_are_mass_centers: ti.template(),
):
    density = material[mat_id]._get_density()
    for nb in range(end_body_num - start_body_num):
        packed_id = start_body_num + nb
        body_id = body_start + nb
        soft_id = soft_start + nb
        compact_grid_sum = soft_body[soft_id].mpmGridNum
        local_point_start = point_start + nb * point_count
        local_grid_start = grid_start + nb * grid_sum
        local_mpm_grid_start = mpm_grid_start + nb * compact_grid_sum
        local_surface_start = surface_num + nb * surface_sum
        local_vertice_start = vertice_num + nb * surface_sum

        bounding_x, bounding_r = coords[packed_id], radii[packed_id]
        scale_factor = bounding_r / r_bound
        orientation = orients[packed_id]
        q = set_from_euler(*orientation)
        rotation_matrix = set_to_rotate(q)
        com_pos = bounding_x
        if ti.static(not coordinates_are_mass_centers):
            com_pos = bounding_x - scale_factor * rotation_matrix @ x_bound
        equiv_rad = scale_factor * eqradius
        volume = template_volume * scale_factor**3
        mass = density * volume
        inv_inertia = 1.0 / (inertia * density * scale_factor**5)

        create_bounding_box(
            body_id, scale_factor, bounding_box, min_box, max_box, local_grid_start, space, gnum, extent
        )
        bounding_box[body_id]._set_reference_surface_area(reference_surface_area)
        create_deformable_grids(local_grid_start, grid_sum, grid, 1.0, distance_fields)

        for i in range(compact_grid_sum):
            mpm_grid = local_mpm_grid_start + i
            soft_grid[mpm_grid]._grid_reset()
            soft_grid_owner[mpm_grid] = soft_id
            soft_grid_local[mpm_grid] = i

        shape_min = vec3f(1.0e30, 1.0e30, 1.0e30)
        shape_max = vec3f(-1.0e30, -1.0e30, -1.0e30)
        shape_radius = 0.0
        for i in range(surface_sum):
            master[local_surface_start + i] = body_id
            local_node = local_vertice_start + i
            local_x = scale_factor * vec3f(surface_nodes[i, 0], surface_nodes[i, 1], surface_nodes[i, 2])
            surface_global = com_pos + rotation_matrix @ local_x
            surface_node[local_node]._set_surface_node(
                vec3f(surface_nodes[i, 0], surface_nodes[i, 1], surface_nodes[i, 2])
            )
            surface_node[local_node]._set_coefficient(parameters[i])
            surface_node[local_node].v = init_v + init_w.cross(surface_global - com_pos)
            shape_min = ti.min(shape_min, local_x)
            shape_max = ti.max(shape_max, local_x)
            shape_radius = ti.max(shape_radius, local_x.norm())

        soft_body[soft_id]._add_surface_point_index(local_point_start, local_point_start)
        for i in range(point_count):
            local_x = vec3f(material_points[i, 0], material_points[i, 1], material_points[i, 2])
            local_x_scaled = scale_factor * local_x
            x0 = com_pos + rotation_matrix @ local_x_scaled
            pid = local_point_start + i
            pvolume = point_volumes[i] * scale_factor * scale_factor * scale_factor
            pmass = density * pvolume
            point_padding = 0.5 * ti.pow(pvolume, 1.0 / 3.0)
            is_surface_point = (
                ti.abs(bounding_box[body_id].distance(local_x_scaled, grid))
                <= bounding_box[body_id].grid_space + point_padding
            )
            shape_min = ti.min(shape_min, local_x_scaled - vec3f(point_padding, point_padding, point_padding))
            shape_max = ti.max(shape_max, local_x_scaled + vec3f(point_padding, point_padding, point_padding))
            shape_radius = ti.max(
                shape_radius,
                local_x_scaled.norm() + ti.sqrt(3.0) * point_padding,
            )
            material_point[pid]._add_point(
                body_id, mat_id, group_id, x0, x0, init_v + init_w.cross(x0 - com_pos), pmass, pvolume
            )
            if is_surface_point:
                slot = ti.atomic_add(soft_body[soft_id].surfacePointEnd, 1)
                soft_surface_point_id[slot] = pid
                material_point[pid]._set_surface_weight(ti.pow(pvolume, 2.0 / 3.0))
        bounding_box[body_id]._set_shape_box(shape_min, shape_max)
        bounding_box[body_id]._set_shape_radius(shape_radius)
        bounding_sphere[body_id]._set_deformed_shape(
            com_pos, rotation_matrix, shape_min, shape_max, shape_radius, bounding_box[body_id].grid_space
        )

        rigid_body[body_id]._add_body_attribute(com_pos, volume, equiv_rad, inv_inertia, q)
        rigid_body[body_id]._add_surface_index(
            local_surface_start, local_surface_start + surface_sum, local_vertice_start
        )
        rigid_body[body_id]._add_body_properties(mat_id, group_id, density)
        rigid_body[body_id]._add_body_kinematic(init_v, init_w, is_fix)
        rigid_body[body_id]._mark_soft_body(soft_id)

        soft_body[soft_id]._restart(body_id, local_point_start, local_point_start + point_count, group_id, mat_id)
        soft_body[soft_id]._add_body_attribute(mass, com_pos)
        soft_body[soft_id]._add_template_support(
            template_point_start,
            template_surface_start,
            template_sdf_start,
            grid_type,
            scale_factor,
            template_grid_space,
            rotation_matrix,
        )
        soft_body[soft_id].v = init_v
        soft_body[soft_id]._add_surface_index(
            local_surface_start, local_surface_start + surface_sum, local_vertice_start
        )
