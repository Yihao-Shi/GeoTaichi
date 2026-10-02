"""Taichi kernels for incompressible semi-resolved DEM coupling."""

import taichi as ti

from src.mpdem.fluid_dynamics.IncompressibleCoupling import cell_center_velocity_3d
from src.utils.ShapeFunctions import Guassian
from src.utils.constants import PI, Threshold


@ti.func
def expanded_domain_gaussian(radius, relative_distance, support_size):
    # Use the entire selected neighborhood to reconstruct the undisturbed field;
    # a narrow kernel feeds the particle's own pressure disturbance back to it.
    smoothing_length = support_size * radius
    return Guassian(smoothing_length, relative_distance, 1.0)


@ti.func
def clamp_index_3d(index, lower, upper):
    clamped = index
    for d in ti.static(range(3)):
        clamped[d] = ti.min(upper[d] - 1, ti.max(lower[d], clamped[d]))
    return clamped


@ti.func
def clamp_active_cell_3d(cell, active_cnum):
    return clamp_index_3d(cell, ti.Vector([0, 0, 0]), active_cnum)


@ti.func
def sample_cell_solid_fraction_3d(cell, active_cnum, cell_solid_fraction: ti.template()):
    clamped = clamp_active_cell_3d(cell, active_cnum)
    return ti.min(1.0, ti.max(0.0, cell_solid_fraction[clamped]))


@ti.func
def face_fluid_fraction_3d(face, direction, active_cnum, cell_solid_fraction: ti.template()):
    fraction = 0.0
    count = 0
    unit = ti.Vector.unit(3, direction)
    for side in ti.static((0, 1)):
        cell = face - side * unit
        inside = True
        for d in ti.static(range(3)):
            inside = inside and cell[d] >= 0 and cell[d] < active_cnum[d]
        if inside:
            fraction += ti.max(0.05, 1.0 - sample_cell_solid_fraction_3d(cell, active_cnum, cell_solid_fraction))
            count += 1
    return fraction / ti.max(count, 1)


@ti.func
def sample_pressure_3d(
    cell,
    active_cnum,
    center_pressure,
    atmospheric_pressure,
    pressure: ti.template(),
    cell_type: ti.template(),
):
    clamped = clamp_active_cell_3d(cell, active_cnum)
    value = center_pressure
    neighbor_type = int(cell_type[clamped])
    if neighbor_type == 1:
        value = pressure[clamped]
    elif neighbor_type == 0:
        value = 2.0 * atmospheric_pressure - center_pressure
    return value


@ti.func
def pressure_gradient_3d(
    cell,
    active_cnum,
    grid_size,
    atmospheric_pressure,
    pressure: ti.template(),
    cell_type: ti.template(),
):
    clamped = clamp_active_cell_3d(cell, active_cnum)
    gradient = ti.Vector([0.0, 0.0, 0.0])
    if int(cell_type[clamped]) == 1:
        center_pressure = pressure[clamped]
        for d in ti.static(range(3)):
            offset = ti.Vector.unit(3, d)
            lower = clamp_active_cell_3d(clamped - offset, active_cnum)
            upper = clamp_active_cell_3d(clamped + offset, active_cnum)
            p_lower = sample_pressure_3d(
                lower,
                active_cnum,
                center_pressure,
                atmospheric_pressure,
                pressure,
                cell_type,
            )
            p_upper = sample_pressure_3d(
                upper,
                active_cnum,
                center_pressure,
                atmospheric_pressure,
                pressure,
                cell_type,
            )
            distance = (upper[d] - lower[d]) * grid_size[d]
            if distance > Threshold:
                gradient[d] = (p_upper - p_lower) / distance
    return gradient


@ti.func
def cell_center_3d(cell, grid_size):
    return (cell.cast(float) + 0.5) * grid_size


@ti.func
def sphere_support_bounds(position, radius, support_size, active_cnum, igrid_size):
    lower = ti.floor((position - support_size * radius) * igrid_size, int)
    upper = ti.ceil((position + support_size * radius) * igrid_size, int) + 1
    for d in ti.static(range(3)):
        lower[d] = ti.max(0, lower[d])
        upper[d] = ti.min(active_cnum[d], upper[d])
    return lower, upper


@ti.kernel
def kernel_update_sphere_cell_solid_fraction(
    sphereNum: int,
    support_size: int,
    active_cnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    particle: ti.template(),
    sphere: ti.template(),
    cell_solid_fraction: ti.template(),
    sphere_kernel_volume: ti.template(),
):
    cell_solid_fraction.fill(0.0)
    sphere_kernel_volume.fill(0.0)
    cell_volume = grid_size[0] * grid_size[1] * grid_size[2]
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        position = particle[np].x
        radius = particle[np].rad
        lower, upper = sphere_support_bounds(position, radius, support_size, active_cnum, igrid_size)
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            weight = Guassian(radius, position - cell_center_3d(I, grid_size), support_size)
            ti.atomic_add(sphere_kernel_volume[nsphere], weight * cell_volume)

    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        position = particle[np].x
        radius = particle[np].rad
        particle_volume = 4.0 / 3.0 * PI * radius * radius * radius
        kernel_volume = ti.max(sphere_kernel_volume[nsphere], Threshold)
        lower, upper = sphere_support_bounds(position, radius, support_size, active_cnum, igrid_size)
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            weight = Guassian(radius, position - cell_center_3d(I, grid_size), support_size)
            ti.atomic_add(cell_solid_fraction[I], weight * particle_volume / kernel_volume)

    for I in ti.grouped(cell_solid_fraction):
        cell_solid_fraction[I] = ti.min(1.0, ti.max(0.0, cell_solid_fraction[I]))


@ti.kernel
def kernel_compute_incompressible_sphere_drag(
    sphereNum: int,
    dependent_domain: int,
    influence_domain: int,
    active_cnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    matProps: ti.template(),
    particle: ti.template(),
    sphere: ti.template(),
    node: ti.template(),
    cell_solid_fraction: ti.template(),
    cell_drag_force: ti.template(),
    fluid_velocity: ti.template(),
    fluid_fraction: ti.template(),
    fluid_volume: ti.template(),
    drag_coefficient_model: ti.template(),
):
    cell_drag_force.fill(0.0)
    fluid_velocity.fill(0.0)
    fluid_fraction.fill(0.0)
    fluid_volume.fill(0.0)
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        position = particle[np].x
        radius = particle[np].rad
        lower, upper = sphere_support_bounds(position, radius, dependent_domain, active_cnum, igrid_size)
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            center = cell_center_3d(I, grid_size)
            weight = expanded_domain_gaussian(radius, position - center, dependent_domain)
            fluid_volume[nsphere] += weight
            fluid_velocity[nsphere] += weight * cell_center_velocity_3d(I, node)
            fluid_fraction[nsphere] += weight * (
                1.0 - sample_cell_solid_fraction_3d(I, active_cnum, cell_solid_fraction)
            )

    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        weight_sum = ti.max(fluid_volume[nsphere], Threshold)
        velocity_f = fluid_velocity[nsphere] / weight_sum
        epsilon_f = ti.min(1.0, ti.max(0.05, fluid_fraction[nsphere] / weight_sum))
        relative_velocity = velocity_f - particle[np].v
        drag_force = drag_coefficient_model.drag_law(
            epsilon_f, matProps.density, matProps.viscosity, particle[np].rad, relative_velocity
        )
        particle[np].contact_force += drag_force

        position = particle[np].x
        radius = particle[np].rad
        lower, upper = sphere_support_bounds(position, radius, influence_domain, active_cnum, igrid_size)
        mapped_weight = 0.0
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            mapped_weight += Guassian(radius, position - cell_center_3d(I, grid_size), influence_domain)
        mapped_weight = ti.max(mapped_weight, Threshold)
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            weight = Guassian(radius, position - cell_center_3d(I, grid_size), influence_domain) / mapped_weight
            ti.atomic_add(cell_drag_force[I], -weight * drag_force)


@ti.kernel
def kernel_apply_cell_drag_force_to_mac_velocity(
    cutoff: float,
    dt: ti.template(),
    active_cnum: ti.types.vector(3, int),
    node: ti.template(),
    cell_solid_fraction: ti.template(),
    cell_drag_force: ti.template(),
):
    for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
        force = cell_drag_force[I]
        for d in ti.static(range(3)):
            unit = ti.Vector.unit(3, d)
            left_face = I
            right_face = I + unit
            left_mass = node.m[d][left_face] * face_fluid_fraction_3d(left_face, d, active_cnum, cell_solid_fraction)
            right_mass = node.m[d][right_face] * face_fluid_fraction_3d(right_face, d, active_cnum, cell_solid_fraction)
            active_faces = ti.cast(left_mass > cutoff, int) + ti.cast(right_mass > cutoff, int)
            if active_faces > 0:
                impulse = force[d] * dt[None] / active_faces
                if left_mass > cutoff:
                    ti.atomic_add(node.velocity[d][left_face], impulse / left_mass)
                if right_mass > cutoff:
                    ti.atomic_add(node.velocity[d][right_face], impulse / right_mass)


@ti.kernel
def kernel_accumulate_incompressible_sphere_pressure_force(
    sphereNum: int,
    dependent_domain: int,
    active_cnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    particle: ti.template(),
    sphere: ti.template(),
    pressure: ti.template(),
    cell_type: ti.template(),
    atmospheric_pressure: float,
):
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        position = particle[np].x
        radius = particle[np].rad
        pressure_gradient = ti.Vector([0.0, 0.0, 0.0])
        weight_sum = 0.0
        lower, upper = sphere_support_bounds(position, radius, dependent_domain, active_cnum, igrid_size)
        for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            if int(cell_type[I]) == 1:
                weight = expanded_domain_gaussian(radius, position - cell_center_3d(I, grid_size), dependent_domain)
                pressure_gradient += weight * pressure_gradient_3d(
                    I,
                    active_cnum,
                    grid_size,
                    atmospheric_pressure,
                    pressure,
                    cell_type,
                )
                weight_sum += weight
        pressure_gradient /= ti.max(weight_sum, Threshold)
        particle_volume = 4.0 / 3.0 * PI * radius * radius * radius
        particle[np].contact_force += -particle_volume * pressure_gradient


@ti.kernel
def kernel_apply_integrated_added_mass(
    sphereNum: int,
    coefficient: float,
    fluid_density: float,
    gravity: ti.types.vector(3, float),
    particle: ti.template(),
    sphere: ti.template(),
):
    """Apply Vp (rho_p + C_A rho_f) a = F without changing true mass."""
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        radius = particle[np].rad
        added_mass = coefficient * fluid_density * 4.0 / 3.0 * PI * radius**3
        true_mass = particle[np].m
        mass_ratio = true_mass / (true_mass + added_mass)
        total_force = particle[np].contact_force + true_mass * gravity
        particle[np].contact_force = mass_ratio * total_force - true_mass * gravity


@ti.kernel
def kernel_apply_sphere_plane_wall_lubrication(
    sphereNum: int,
    wallNum: int,
    viscosity: float,
    activation_gap: float,
    minimum_gap: float,
    particle: ti.template(),
    sphere: ti.template(),
    wall: ti.template(),
):
    """Add the unresolved leading-order sphere--wall lubrication force."""
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        radius = particle[np].rad
        for wall_id in range(wallNum):
            if wall[wall_id].active != 0:
                normal = wall[wall_id].norm
                distance = (particle[np].x - wall[wall_id].point).dot(normal)
                gap = distance - radius
                if gap < activation_gap:
                    resolved_gap = ti.max(gap, minimum_gap)
                    normal_velocity = particle[np].v.dot(normal)
                    coefficient = 6.0 * PI * viscosity * radius * radius * (1.0 / resolved_gap - 1.0 / activation_gap)
                    particle[np].contact_force += -coefficient * normal_velocity * normal
