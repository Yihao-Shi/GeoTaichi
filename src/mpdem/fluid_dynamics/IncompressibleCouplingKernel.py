import taichi as ti

from src.levelset.FluidLevelSetKernel import free_surface_theta, free_surface_theta_from_phi_half_default
from src.utils.Quaternion import SetToRotate
from src.utils.constants import Threshold
from src.utils.TypeDefination import vec3f
from src.mpm.engines.EngineKernel import ibm_cell_pressure_acceleration as cell_pressure_acceleration_3d


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
def cell_viscous_stress_3d(cell, active_cnum, grid_size, viscosity, node: ti.template()):
    velocity_gradient = ti.Matrix.zero(float, 3, 3)
    for d in ti.static(range(3)):
        offset = ti.Vector.unit(3, d)
        left = clamp_cell_index_3d(cell - offset, active_cnum)
        right = clamp_cell_index_3d(cell + offset, active_cnum)
        distance = (right[d] - left[d]) * grid_size[d]
        if distance > Threshold:
            derivative = (cell_center_velocity_3d(right, node) - cell_center_velocity_3d(left, node)) / distance
            for component in ti.static(range(3)):
                velocity_gradient[component, d] = derivative[component]
    return viscosity * (velocity_gradient + velocity_gradient.transpose())


@ti.func
def cell_neighbor_pressure_gradient_3d(
    cell,
    neighbor,
    direction,
    grid_size,
    atmospheric_pressure,
    pressure: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
):
    value = 0.0
    neighbor_type = int(cell_type[neighbor])
    if int(cell_type[cell]) == 1 and neighbor[direction] != cell[direction]:
        if neighbor_type == 1:
            value = (pressure[neighbor] - pressure[cell]) / grid_size[direction]
        elif neighbor_type == 0:
            theta = 1.0
            if ti.static(use_free_surface_theta):
                theta = free_surface_theta(cell, neighbor, fluid_sdf)
            value = (atmospheric_pressure + surface_tension[neighbor] - pressure[cell]) / (grid_size[direction] * theta)
    return value


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
        for d in ti.static(range(3)):
            for side in ti.static((-1, 1)):
                neighbor = clamp_cell_index_3d(cell + side * ti.Vector.unit(3, d), active_cnum)
                gradient[d] += (
                    0.5
                    * side
                    * cell_neighbor_pressure_gradient_3d(
                        cell,
                        neighbor,
                        d,
                        grid_size,
                        atmospheric_pressure,
                        pressure,
                        surface_tension,
                        cell_type,
                        fluid_sdf,
                        use_free_surface_theta,
                    )
                )
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
    grid_size: vec3f,
    igrid_size: vec3f,
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
    grid_size: vec3f,
    igrid_size: vec3f,
    bounding_sphere: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    levelset_grid: ti.template(),
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
    dt: float,
    viscous_delta0: ti.template(),
    viscous_delta1: ti.template(),
    viscous_delta2: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    min_dx = min_grid_spacing_3d(grid_size)
    cell_volume = grid_size[0] * grid_size[1] * grid_size[2]
    fluid_density = matProps.density
    # The grid loop is the parallel dimension; rigid accumulation remains
    # atomic because many cell tasks contribute to the same body resultant.
    for I in ti.grouped(ti.ndrange(active_cnum[0], active_cnum[1], active_cnum[2])):
        total_solid_fraction = ti.min(1.0, ti.max(0.0, solid_fraction[I]))
        if total_solid_fraction > Threshold and int(cell_type[I]) == 1:
            position = (I.cast(float) + 0.5) * grid_size
            local_solid_density = solid_density[I]
            if local_solid_density <= Threshold:
                local_solid_density = fluid_density
            acceleration = cell_pressure_acceleration_3d(
                I,
                active_cnum,
                grid_size,
                fluid_density,
                matProps.atmospheric_pressure,
                solid_fraction,
                solid_density,
                pressure,
                surface_tension,
                cell_type,
                fluid_sdf,
                use_free_surface_theta,
            )
            # Use the same pressure mobility and pre-projection viscous increment as the MAC grid.
            acceleration += (
                0.5
                / dt
                * ti.Vector(
                    [
                        viscous_delta0[I] + viscous_delta0[I + ti.Vector([1, 0, 0])],
                        viscous_delta1[I] + viscous_delta1[I + ti.Vector([0, 1, 0])],
                        viscous_delta2[I] + viscous_delta2[I + ti.Vector([0, 0, 1])],
                    ]
                )
            )
            mixed_density = (1.0 - total_solid_fraction) * fluid_density + total_solid_fraction * local_solid_density
            force_density = lsdem_volume_fraction_ibm_force_density(
                total_solid_fraction,
                fluid_density,
                local_solid_density,
                mixed_density * acceleration,
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
    pressure_acceleration=0.0,
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
            velocity_change = face_fraction * (
                target_velocity - fluid_velocity[face] - dt[None] * pressure_acceleration
            )
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
def kernel_initialize_double_layer_ibm_pressure(
    particle_count: int,
    grid_size: vec3f,
    particle: ti.template(),
    pressure: ti.template(),
    weight: ti.template(),
):
    for I in ti.grouped(pressure):
        pressure[I], weight[I] = 0.0, 0.0
    for p in range(particle_count):
        if int(particle[p].active) == 1 and int(particle[p].phase) == 2:
            cell = ti.floor(particle[p].x / grid_size).cast(int)
            inside = True
            for d in ti.static(range(3)):
                inside = inside and 0 <= cell[d] < pressure.shape[d]
            if inside:
                ti.atomic_add(pressure[cell], particle[p].vol * particle[p].pressure)
                ti.atomic_add(weight[cell], particle[p].vol)
    for I in ti.grouped(pressure):
        if weight[I] > 0.0:
            pressure[I] /= weight[I]


@ti.kernel
def kernel_apply_double_layer_lsdem_ibm_pressure_predictor(
    cutoff: float,
    dt: ti.template(),
    grid_size: vec3f,
    solid_fraction: ti.template(),
    solid_velocity_cell: ti.template(),
    fluid_mass: ti.template(),
    fluid_velocity: ti.template(),
    fluid_acceleration: ti.template(),
    reaction_cell: ti.template(),
    pressure: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    fluid_density: ti.template(),
    pressure_mobility: ti.template(),
):
    for axis in ti.static(range(3)):
        for I in ti.grouped(fluid_velocity[axis]):
            acceleration = 0.0
            if 0 < I[axis] < fluid_velocity[axis].shape[axis] - 1:
                left = I - ti.Vector.unit(3, axis)
                right = I
                jump = 0.0
                theta = 1.0
                if cell_type[left] == 1 and cell_type[right] == 1:
                    jump = pressure[right] - pressure[left]
                elif cell_type[left] == 1 and cell_type[right] == 0:
                    jump = -pressure[left]
                    theta = free_surface_theta_from_phi_half_default(fluid_sdf[left], fluid_sdf[right])
                elif cell_type[left] == 0 and cell_type[right] == 1:
                    jump = pressure[right]
                    theta = free_surface_theta_from_phi_half_default(fluid_sdf[right], fluid_sdf[left])
                acceleration = -jump / (theta * grid_size[axis] * ti.max(fluid_density[axis][I], 1.0e-12))
                if ti.static(pressure_mobility != None):
                    acceleration = -jump * pressure_mobility[axis][I] / (theta * grid_size[axis])
            apply_double_layer_ibm_face_3d(
                axis,
                I,
                cutoff,
                dt,
                solid_fraction,
                solid_velocity_cell,
                fluid_mass[axis],
                fluid_velocity[axis],
                fluid_acceleration[axis],
                reaction_cell,
                acceleration,
            )


@ti.kernel
def kernel_accumulate_double_layer_lsdem_ibm_force(
    rigid_num: int,
    cnum: ti.types.vector(3, int),
    grid_size: vec3f,
    bounding_sphere: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    levelset_grid: ti.template(),
    solid_fraction: ti.template(),
    solid_fraction_sum: ti.template(),
    reaction_cell: ti.template(),
    cutoff: float,
    gravity: vec3f,
    fluid_mass: ti.template(),
    fluid_acceleration: ti.template(),
):
    min_dx = min_grid_spacing_3d(grid_size)
    for I in ti.grouped(ti.ndrange(cnum[0], cnum[1], cnum[2])):
        total_fraction = ti.min(1.0, ti.max(0.0, solid_fraction[I]))
        if total_fraction > Threshold:
            # MAC reaction already includes actual fluid mass and both IBM passes.
            # Air-cell shares still carry impulse from adjacent occupied faces.
            force = reaction_cell[I]
            # IBM also accelerates fictitious interior water. Remove its inertia
            # and weight from the physical fluid/body momentum balance.
            for d in ti.static(range(3)):
                for side in ti.static((0, 1)):
                    face = I + side * ti.Vector.unit(3, d)
                    if fluid_mass[d][face] > cutoff:
                        share = 0.5
                        if face[d] == 0 or face[d] == cnum[d]:
                            share = 1.0
                        force[d] += (
                            share * total_fraction * fluid_mass[d][face] * (fluid_acceleration[d][face] - gravity[d])
                        )
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
                        if body_fraction > 1e-12:
                            body_force = partitioned_body_share(body_fraction, solid_fraction_sum[I]) * force
                            torque = (position - mass_center).cross(body_force)
                            for d in ti.static(range(3)):
                                ti.atomic_add(rigid[body].contact_force[d], body_force[d])
                                ti.atomic_add(rigid[body].contact_torque[d], torque[d])
