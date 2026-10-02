import taichi as ti

from src.levelset.FluidLevelSetKernel import free_surface_theta
from src.utils.Quaternion import SetToRotate
from src.utils.constants import Threshold


@ti.func
def min_grid_spacing_3d(grid_size):
    return ti.min(grid_size[0], ti.min(grid_size[1], grid_size[2]))


@ti.func
def clamp_cell_index_3d(index, active_cnum):
    clamped = index
    for d in ti.static(range(3)):
        clamped[d] = ti.min(active_cnum[d] - 1, ti.max(0, clamped[d]))
    return clamped


@ti.func
def cell_center_velocity_3d(cell, node: ti.template()):
    velocity = ti.Vector([0.0, 0.0, 0.0])
    for d in ti.static(range(3)):
        offset = ti.Vector.unit(3, d)
        velocity[d] = 0.5 * (node.velocity[d][cell] + node.velocity[d][cell + offset])
    return velocity


@ti.kernel
def kernel_ibm_velocity_l2_error(
    active_cnum: ti.types.vector(3, int),
    solid_fraction: ti.template(),
    solid_velocity: ti.template(),
    node: ti.template(),
) -> float:
    error2 = 0.0
    weight = 0.0
    for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
        fraction = ti.max(0.0, ti.min(1.0, solid_fraction[I]))
        if fraction > 0.5:
            difference = cell_center_velocity_3d(I, node) - solid_velocity[I]
            error2 += fraction * difference.dot(difference)
            weight += fraction
    return ti.sqrt(error2 / ti.max(weight, 1e-12))


@ti.func
def sample_cell_center_velocity_3d(cell, active_cnum, node: ti.template()):
    clamped = clamp_cell_index_3d(cell, active_cnum)
    return cell_center_velocity_3d(clamped, node)


@ti.func
def cell_velocity_laplacian_3d(cell, active_cnum, grid_size, node: ti.template()):
    center_velocity = sample_cell_center_velocity_3d(cell, active_cnum, node)
    laplacian = ti.Vector([0.0, 0.0, 0.0])
    for d in ti.static(range(3)):
        offset = ti.Vector.unit(3, d)
        left_velocity = sample_cell_center_velocity_3d(cell - offset, active_cnum, node)
        right_velocity = sample_cell_center_velocity_3d(cell + offset, active_cnum, node)
        laplacian += (left_velocity - 2.0 * center_velocity + right_velocity) / (grid_size[d] * grid_size[d])
    return laplacian


@ti.func
def cell_pressure_gradient_3d(
    cell,
    active_cnum,
    grid_size,
    atmospheric_pressure,
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
):
    gradient = ti.Vector([0.0, 0.0, 0.0])
    if int(cell_type[cell]) == 1:
        center_pressure = pressure[cell]
        for d in ti.static(range(3)):
            offset = ti.Vector.unit(3, d)
            left = clamp_cell_index_3d(cell - offset, active_cnum)
            right = clamp_cell_index_3d(cell + offset, active_cnum)
            left_pressure = center_pressure
            right_pressure = center_pressure
            left_distance = 0.0
            right_distance = 0.0
            if int(cell_type[left]) == 1:
                left_pressure = pressure[left]
                left_distance = (cell[d] - left[d]) * grid_size[d]
            elif int(cell_type[left]) == 0:
                left_pressure = atmospheric_pressure + surface_tension[left]
                left_distance = grid_size[d]
                if ti.static(use_free_surface_theta):
                    left_distance *= free_surface_theta(cell, left, fluid_sdf)
            if int(cell_type[right]) == 1:
                right_pressure = pressure[right]
                right_distance = (right[d] - cell[d]) * grid_size[d]
            elif int(cell_type[right]) == 0:
                right_pressure = atmospheric_pressure + surface_tension[right]
                right_distance = grid_size[d]
                if ti.static(use_free_surface_theta):
                    right_distance *= free_surface_theta(cell, right, fluid_sdf)
            distance = left_distance + right_distance
            if distance > Threshold:
                gradient[d] = (right_pressure - left_pressure) / distance
    return gradient


@ti.func
def partitioned_body_share(body_fraction, body_fraction_sum):
    return ti.max(0.0, body_fraction) / ti.max(body_fraction_sum, Threshold)


@ti.func
def lsdem_volume_fraction_ibm_force_density(
    solid_fraction, fluid_density, solid_density, stress_divergence, ibm_force_density
):
    mixed_density = (1.0 - solid_fraction) * fluid_density + solid_fraction * solid_density
    weighted_solid_fraction = ti.min(
        1.0, ti.max(0.0, solid_fraction * solid_density / ti.max(mixed_density, Threshold))
    )
    return weighted_solid_fraction * stress_divergence - (1.0 - weighted_solid_fraction) * ibm_force_density


@ti.func
def estimate_cell_solid_fraction_3d(
    cell, grid_size, min_dx, mass_center, rotate_matrix, body, box: ti.template(), levelset_grid: ti.template()
):
    fraction_inside = 0.0
    fraction_denominator = 0.0
    for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
        offset = ti.Vector([ox, oy, oz])
        vertex = (cell.cast(float) + offset.cast(float)) * grid_size
        local_vertex = rotate_matrix.transpose() @ (vertex - mass_center)
        vertex_phi = 1e6 * min_dx
        if box[body]._in_box(local_vertex):
            vertex_phi = box[body].distance(local_vertex, levelset_grid)
        fraction_denominator += ti.abs(vertex_phi)
        if vertex_phi < 0.0:
            fraction_inside += -vertex_phi

    solid_fraction = 0.0
    if fraction_denominator > 1e-12:
        solid_fraction = ti.min(1.0, ti.max(0.0, fraction_inside / fraction_denominator))
    return solid_fraction


@ti.kernel
def kernel_update_lsdem_cell_ibm_fields(
    rigid_num: int,
    ghost_cell: int,
    cnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    bounding_sphere: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    levelset_grid: ti.template(),
    solid_fraction: ti.template(),
    solid_fraction_sum: ti.template(),
    solid_density: ti.template(),
    solid_velocity_cell: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    # Cell-major traversal exposes one GPU task per fluid cell even when only
    # one or two rigid bodies exist. The inner body loop is intentionally
    # small; add spatial body bins if dense many-body IBM becomes a workload.
    for I in ti.grouped(solid_fraction):
        fraction_sum = 0.0
        density_sum = 0.0
        velocity_sum = ti.Vector([0.0, 0.0, 0.0])
        for body in range(rigid_num):
            if int(bounding_sphere[body].active) == 1:
                sphere_center = bounding_sphere[body]._get_position()
                sphere_radius = bounding_sphere[body]._get_radius()
                lower = ti.floor((sphere_center - sphere_radius) * igrid_size, int) - 1
                upper = ti.ceil((sphere_center + sphere_radius) * igrid_size, int) + 1
                inside_bounds = 1
                for d in ti.static(range(3)):
                    lower[d] = ti.max(-ghost_cell, lower[d])
                    upper[d] = ti.min(active_cnum[d] + ghost_cell, upper[d])
                    if I[d] < lower[d] or I[d] >= upper[d]:
                        inside_bounds = 0
                if inside_bounds == 1:
                    mass_center = rigid[body]._get_position()
                    rotate_matrix = SetToRotate(rigid[body].q)
                    position = (I.cast(float) + 0.5) * grid_size
                    cell_solid_fraction = estimate_cell_solid_fraction_3d(
                        I,
                        grid_size,
                        min_grid_spacing_3d(grid_size),
                        mass_center,
                        rotate_matrix,
                        body,
                        box,
                        levelset_grid,
                    )
                    if cell_solid_fraction > 1e-12:
                        rigid_velocity = rigid[body]._get_velocity()
                        rigid_omega = rigid[body]._get_angular_velocity()
                        body_density = rigid[body]._get_mass() / ti.max(rigid[body]._get_volume(), 1e-12)
                        solid_velocity = rigid_velocity + rigid_omega.cross(position - mass_center)
                        fraction_sum += cell_solid_fraction
                        density_sum += cell_solid_fraction * body_density
                        velocity_sum += cell_solid_fraction * solid_velocity
        if fraction_sum > 1e-12:
            solid_fraction[I] = ti.min(1.0, fraction_sum)
            solid_fraction_sum[I] = fraction_sum
            solid_density[I] = density_sum / fraction_sum
            solid_velocity_cell[I] = velocity_sum / fraction_sum
        else:
            solid_fraction[I] = 0.0
            solid_fraction_sum[I] = 0.0
            solid_density[I] = 0.0
            solid_velocity_cell[I] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def kernel_accumulate_lsdem_volume_fraction_ibm_force(
    rigid_num: int,
    ghost_cell: int,
    cnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    bounding_sphere: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    levelset_grid: ti.template(),
    node: ti.template(),
    matProps: ti.template(),
    solid_fraction: ti.template(),
    solid_fraction_sum: ti.template(),
    solid_density: ti.template(),
    ibm_force_cell: ti.template(),
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    min_dx = min_grid_spacing_3d(grid_size)
    cell_volume = grid_size[0] * grid_size[1] * grid_size[2]
    fluid_density = matProps.density
    viscosity = matProps.viscosity
    # The grid loop is the parallel dimension; rigid accumulation remains
    # atomic because many cell tasks contribute to the same body resultant.
    for I in ti.grouped(ti.ndrange(active_cnum[0], active_cnum[1], active_cnum[2])):
        total_solid_fraction = ti.min(1.0, ti.max(0.0, solid_fraction[I]))
        if total_solid_fraction > Threshold and int(cell_type[I]) == 1:
            position = (I.cast(float) + 0.5) * grid_size
            local_solid_density = solid_density[I]
            if local_solid_density <= Threshold:
                local_solid_density = fluid_density
            pressure_gradient = cell_pressure_gradient_3d(
                I,
                active_cnum,
                grid_size,
                matProps.atmospheric_pressure,
                pressure,
                surface_tension,
                cell_type,
                fluid_sdf,
                use_free_surface_theta,
            )
            viscous_stress_divergence = viscosity * cell_velocity_laplacian_3d(I, active_cnum, grid_size, node)
            stress_divergence = -pressure_gradient + viscous_stress_divergence
            force_density = lsdem_volume_fraction_ibm_force_density(
                total_solid_fraction,
                fluid_density,
                local_solid_density,
                stress_divergence,
                ibm_force_cell[I],
            )
            for body in range(rigid_num):
                if int(bounding_sphere[body].active) == 1:
                    sphere_center = bounding_sphere[body]._get_position()
                    sphere_radius = bounding_sphere[body]._get_radius() + min_dx
                    lower = ti.floor((sphere_center - sphere_radius) * igrid_size, int) - 1
                    upper = ti.ceil((sphere_center + sphere_radius) * igrid_size, int) + 1
                    inside_bounds = 1
                    for d in ti.static(range(3)):
                        lower[d] = ti.max(0, lower[d])
                        upper[d] = ti.min(active_cnum[d], upper[d])
                        if I[d] < lower[d] or I[d] >= upper[d]:
                            inside_bounds = 0
                    if inside_bounds == 1:
                        mass_center = rigid[body]._get_position()
                        rotate_matrix = SetToRotate(rigid[body].q)
                        body_solid_fraction = estimate_cell_solid_fraction_3d(
                            I,
                            grid_size,
                            min_dx,
                            mass_center,
                            rotate_matrix,
                            body,
                            box,
                            levelset_grid,
                        )
                        if body_solid_fraction > 1e-6:
                            body_share = partitioned_body_share(body_solid_fraction, solid_fraction_sum[I])
                            force = body_share * force_density * cell_volume
                            torque = (position - mass_center).cross(force)
                            for d in ti.static(range(3)):
                                ti.atomic_add(rigid[body].contact_force[d], force[d])
                                ti.atomic_add(rigid[body].contact_torque[d], torque[d])


@ti.func
def double_layer_cell_velocity_3d(
    cell, velocity_x: ti.template(), velocity_y: ti.template(), velocity_z: ti.template()
):
    return ti.Vector(
        [
            0.5 * (velocity_x[cell] + velocity_x[cell + ti.Vector([1, 0, 0])]),
            0.5 * (velocity_y[cell] + velocity_y[cell + ti.Vector([0, 1, 0])]),
            0.5 * (velocity_z[cell] + velocity_z[cell + ti.Vector([0, 0, 1])]),
        ]
    )


@ti.func
def sample_double_layer_cell_velocity_3d(
    cell, active_cnum, velocity_x: ti.template(), velocity_y: ti.template(), velocity_z: ti.template()
):
    clamped = clamp_cell_index_3d(cell, active_cnum)
    return double_layer_cell_velocity_3d(clamped, velocity_x, velocity_y, velocity_z)


@ti.func
def double_layer_velocity_laplacian_3d(
    cell,
    active_cnum,
    grid_size,
    velocity_x: ti.template(),
    velocity_y: ti.template(),
    velocity_z: ti.template(),
):
    center = sample_double_layer_cell_velocity_3d(cell, active_cnum, velocity_x, velocity_y, velocity_z)
    laplacian = ti.Vector([0.0, 0.0, 0.0])
    for d in ti.static(range(3)):
        offset = ti.Vector.unit(3, d)
        left = sample_double_layer_cell_velocity_3d(cell - offset, active_cnum, velocity_x, velocity_y, velocity_z)
        right = sample_double_layer_cell_velocity_3d(cell + offset, active_cnum, velocity_x, velocity_y, velocity_z)
        laplacian += (left - 2.0 * center + right) / (grid_size[d] * grid_size[d])
    return laplacian


@ti.func
def apply_double_layer_ibm_face_3d(
    axis,
    face,
    cutoff,
    dt,
    solid_fraction: ti.template(),
    solid_velocity_cell: ti.template(),
    fluid_mass: ti.template(),
    fluid_velocity: ti.template(),
    fluid_acceleration: ti.template(),
    reaction_cell: ti.template(),
):
    if fluid_mass[face] > cutoff:
        cell_shape = ti.Vector([solid_fraction.shape[0], solid_fraction.shape[1], solid_fraction.shape[2]])
        lower = face - ti.Vector.unit(3, axis)
        upper = face
        lower_fraction = 0.0
        upper_fraction = 0.0
        target_numerator = 0.0
        if lower[axis] >= 0:
            lower_fraction = ti.max(0.0, ti.min(1.0, solid_fraction[lower]))
            target_numerator += lower_fraction * solid_velocity_cell[lower][axis]
        if upper[axis] < cell_shape[axis]:
            upper_fraction = ti.max(0.0, ti.min(1.0, solid_fraction[upper]))
            target_numerator += upper_fraction * solid_velocity_cell[upper][axis]
        fraction_sum = lower_fraction + upper_fraction
        face_fraction = ti.min(1.0, 0.5 * fraction_sum)
        if (lower[axis] < 0) or (upper[axis] >= cell_shape[axis]):
            face_fraction = ti.min(1.0, fraction_sum)
        if face_fraction > Threshold and fraction_sum > Threshold:
            target_velocity = target_numerator / fraction_sum
            velocity_change = face_fraction * (target_velocity - fluid_velocity[face])
            fluid_velocity[face] += velocity_change
            fluid_acceleration[face] += velocity_change / dt[None]
            reaction = -fluid_mass[face] * velocity_change / dt[None]
            if lower_fraction > Threshold:
                ti.atomic_add(reaction_cell[lower][axis], reaction * lower_fraction / fraction_sum)
            if upper_fraction > Threshold:
                ti.atomic_add(reaction_cell[upper][axis], reaction * upper_fraction / fraction_sum)


@ti.kernel
def kernel_apply_double_layer_lsdem_ibm(
    cutoff: float,
    dt: ti.template(),
    solid_fraction: ti.template(),
    solid_velocity_cell: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_mass_z: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    fluid_acceleration_z: ti.template(),
    reaction_cell: ti.template(),
):
    for I in ti.grouped(fluid_velocity_x):
        apply_double_layer_ibm_face_3d(
            0,
            I,
            cutoff,
            dt,
            solid_fraction,
            solid_velocity_cell,
            fluid_mass_x,
            fluid_velocity_x,
            fluid_acceleration_x,
            reaction_cell,
        )
    for I in ti.grouped(fluid_velocity_y):
        apply_double_layer_ibm_face_3d(
            1,
            I,
            cutoff,
            dt,
            solid_fraction,
            solid_velocity_cell,
            fluid_mass_y,
            fluid_velocity_y,
            fluid_acceleration_y,
            reaction_cell,
        )
    for I in ti.grouped(fluid_velocity_z):
        apply_double_layer_ibm_face_3d(
            2,
            I,
            cutoff,
            dt,
            solid_fraction,
            solid_velocity_cell,
            fluid_mass_z,
            fluid_velocity_z,
            fluid_acceleration_z,
            reaction_cell,
        )


@ti.kernel
def kernel_accumulate_double_layer_lsdem_ibm_force(
    rigid_num: int,
    cnum: ti.types.vector(3, int),
    grid_size: ti.types.vector(3, float),
    igrid_size: ti.types.vector(3, float),
    bounding_sphere: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    levelset_grid: ti.template(),
    solid_fraction: ti.template(),
    solid_fraction_sum: ti.template(),
    solid_density: ti.template(),
    reaction_cell: ti.template(),
    pressure: ti.template(),
    zero_surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    cell_fluid_density: ti.template(),
    cell_fluid_viscosity: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
):
    min_dx = min_grid_spacing_3d(grid_size)
    cell_volume = grid_size[0] * grid_size[1] * grid_size[2]
    for I in ti.grouped(ti.ndrange(cnum[0], cnum[1], cnum[2])):
        total_fraction = ti.min(1.0, ti.max(0.0, solid_fraction[I]))
        if total_fraction > Threshold and int(cell_type[I]) == 1:
            fluid_density = ti.max(cell_fluid_density[I], Threshold)
            local_solid_density = solid_density[I]
            if local_solid_density <= Threshold:
                local_solid_density = fluid_density
            mixed_density = (1.0 - total_fraction) * fluid_density + total_fraction * local_solid_density
            weighted_fraction = ti.min(
                1.0,
                ti.max(0.0, total_fraction * local_solid_density / ti.max(mixed_density, Threshold)),
            )
            pressure_gradient = cell_pressure_gradient_3d(
                I,
                cnum,
                grid_size,
                0.0,
                pressure,
                zero_surface_tension,
                cell_type,
                fluid_sdf,
                True,
            )
            laplacian = double_layer_velocity_laplacian_3d(
                I, cnum, grid_size, fluid_velocity_x, fluid_velocity_y, fluid_velocity_z
            )
            stress_divergence = -pressure_gradient + cell_fluid_viscosity[I] * laplacian
            force = weighted_fraction * stress_divergence * cell_volume + (1.0 - weighted_fraction) * reaction_cell[I]
            position = (I.cast(float) + 0.5) * grid_size
            for body in range(rigid_num):
                if int(bounding_sphere[body].active) == 1:
                    center = bounding_sphere[body]._get_position()
                    radius = bounding_sphere[body]._get_radius() + min_dx
                    inside_bounds = (position - center).norm() <= radius + min_dx
                    if inside_bounds:
                        mass_center = rigid[body]._get_position()
                        rotate_matrix = SetToRotate(rigid[body].q)
                        body_fraction = estimate_cell_solid_fraction_3d(
                            I,
                            grid_size,
                            min_dx,
                            mass_center,
                            rotate_matrix,
                            body,
                            box,
                            levelset_grid,
                        )
                        if body_fraction > 1e-6:
                            body_force = partitioned_body_share(body_fraction, solid_fraction_sum[I]) * force
                            torque = (position - mass_center).cross(body_force)
                            for d in ti.static(range(3)):
                                ti.atomic_add(rigid[body].contact_force[d], body_force[d])
                                ti.atomic_add(rigid[body].contact_torque[d], torque[d])
