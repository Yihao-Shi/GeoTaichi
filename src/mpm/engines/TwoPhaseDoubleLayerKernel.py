import taichi as ti

from src.utils.TypeDefination import vec2f, vec2i, vec3f, vec3i, mat2x2
from src.utils.ShapeFunctions import ShapeLinear, ShapeGIMP, ShapeBsplineQ, ShapeBsplineC
from src.utils.ScalarFunction import vectorize_id
from src.utils.constants import Threshold, ZEROVEC2f, ZEROVEC3f, ZEROMAT2x2, ZEROMAT3x3
from src.levelset.FluidLevelSetKernel import (
    free_surface_theta_from_phi_half_default as _pressure_free_surface_theta_from_phi,
)

PHASE_SOLID = 1
PHASE_FLUID = 2
FLUID_CELL = 1
SOLID_CELL = 2
AIR_CELL = 0
MAC_SHAPE_LINEAR = 0
MAC_SHAPE_GIMP = 1
MAC_SHAPE_QUAD_BSPLINE = 2
MAC_SHAPE_CUBIC_BSPLINE = 3


@ti.func
def _double_layer_shifting_is_interior2d(position, grid_size, cnum, cell_type):
    # Snap only roundoff at exact cell boundaries before testing interface eligibility.
    cell = ti.floor(position / grid_size + 1.0e-6).cast(int)
    active = 0 <= cell[0] < cnum[0] and 0 <= cell[1] < cnum[1]
    if active:
        active = cell_type[cell] == FLUID_CELL
        for d in ti.static(range(2)):
            for side in ti.static((-1, 1)):
                neighbor = cell + side * ti.Vector.unit(2, d)
                if 0 <= neighbor[d] < cnum[d]:
                    active = active and cell_type[neighbor] != AIR_CELL
    return active


@ti.func
def _double_layer_shifting_is_interior3d(position, grid_size, cnum, cell_type):
    cell = ti.floor(position / grid_size + 1.0e-6).cast(int)
    active = 0 <= cell[0] < cnum[0] and 0 <= cell[1] < cnum[1] and 0 <= cell[2] < cnum[2]
    if active:
        active = cell_type[cell] == FLUID_CELL
        for d in ti.static(range(3)):
            for side in ti.static((-1, 1)):
                neighbor = cell + side * ti.Vector.unit(3, d)
                if 0 <= neighbor[d] < cnum[d]:
                    active = active and cell_type[neighbor] != AIR_CELL
    return active


@ti.func
def _double_layer_shifting_volume_error(node_id, body_id, cnum, cell_type, node, reference_volume):
    dimension = ti.static(len(cell_type.shape))
    index = ti.Vector(vectorize_id(node_id, cnum + 1))
    interior, fluid = True, False
    for corner in range(2**dimension):
        cell = index + ti.Vector.zero(int, dimension)
        inside = True
        for d in ti.static(range(dimension)):
            cell[d] -= (corner >> d) & 1
            inside = inside and 0 <= cell[d] < cnum[d]
        if inside:
            interior = interior and cell_type[cell] != AIR_CELL
            fluid = fluid or cell_type[cell] == FLUID_CELL
    error = node[node_id, body_id].vol - reference_volume[node_id, body_id] * node[node_id, body_id].porosity
    # Interior deficits must attract particles; missing free-surface volume must not.
    if not (interior and fluid):
        error = ti.max(error, 0.0)
    return error


@ti.kernel
def kernel_reset_double_layer_grid(node: ti.template()):
    for ng, nb in node:
        node[ng, nb]._grid_reset()


@ti.func
def _linear_weight(r):
    return ti.max(0.0, 1.0 - ti.abs(r))


@ti.func
def _mac_bspline_boundary_type(index: int, count: int):
    btype = ti.min(2, index) - ti.min(count - 1 - index, 2)
    if btype < 0:
        btype += 3
    elif btype > 0:
        btype += 2
    return btype


@ti.func
def _mac_base_1d(xp, inv_dx, stagger, lp, shape_type: ti.template()):
    base = int(ti.floor(xp * inv_dx - stagger))
    if shape_type == MAC_SHAPE_GIMP or shape_type == MAC_SHAPE_QUAD_BSPLINE or shape_type == MAC_SHAPE_CUBIC_BSPLINE:
        base = int(ti.floor((xp - lp) * inv_dx - stagger))
    return base


@ti.func
def _mac_weight_1d(xp, xg, inv_dx, lp, btype: int, shape_type: ti.template()):
    weight = 0.0
    if ti.static(isinstance(shape_type, int)):
        if ti.static(shape_type == MAC_SHAPE_GIMP):
            weight = ShapeGIMP(xp, xg, inv_dx, lp)
        elif ti.static(shape_type == MAC_SHAPE_QUAD_BSPLINE):
            weight = ShapeBsplineQ(xp, xg, inv_dx, btype)
        elif ti.static(shape_type == MAC_SHAPE_CUBIC_BSPLINE):
            weight = ShapeBsplineC(xp, xg, inv_dx, btype)
        else:
            weight = ShapeLinear(xp, xg, inv_dx, lp)
    else:
        if shape_type == MAC_SHAPE_GIMP:
            weight = ShapeGIMP(xp, xg, inv_dx, lp)
        elif shape_type == MAC_SHAPE_QUAD_BSPLINE:
            weight = ShapeBsplineQ(xp, xg, inv_dx, btype)
        elif shape_type == MAC_SHAPE_CUBIC_BSPLINE:
            weight = ShapeBsplineC(xp, xg, inv_dx, btype)
        else:
            weight = ShapeLinear(xp, xg, inv_dx, lp)
    return weight


@ti.func
def _mac_weight2d(position, face_pos, grid_size, psize, btype, shape_type: ti.template()):
    return _mac_weight_1d(
        position[0], face_pos[0], 1.0 / grid_size[0], psize[0], btype[0], shape_type
    ) * _mac_weight_1d(position[1], face_pos[1], 1.0 / grid_size[1], psize[1], btype[1], shape_type)


@ti.func
def _mac_weight3d(position, face_pos, grid_size, psize, btype, shape_type: ti.template()):
    return (
        _mac_weight_1d(position[0], face_pos[0], 1.0 / grid_size[0], psize[0], btype[0], shape_type)
        * _mac_weight_1d(position[1], face_pos[1], 1.0 / grid_size[1], psize[1], btype[1], shape_type)
        * _mac_weight_1d(position[2], face_pos[2], 1.0 / grid_size[2], psize[2], btype[2], shape_type)
    )


@ti.func
def _pressure_free_surface_theta(fluid_cell, air_cell, fluid_sdf):
    return _pressure_free_surface_theta_from_phi(fluid_sdf[fluid_cell], fluid_sdf[air_cell])


@ti.func
def _coarse_pressure_sdf_value2d(cell, factor: int, fluid_sdf):
    base = cell * factor
    value = 0.0
    count = 0
    for ox in range(factor):
        for oy in range(factor):
            fine = base + vec2i(ox, oy)
            if fine[0] < fluid_sdf.shape[0] and fine[1] < fluid_sdf.shape[1]:
                value += fluid_sdf[fine]
                count += 1
    if count > 0:
        value /= ti.cast(count, float)
    return value


@ti.func
def _pressure_free_surface_theta_coarse2d(fluid_cell, air_cell, factor: int, fluid_sdf):
    return _pressure_free_surface_theta_from_phi(
        _coarse_pressure_sdf_value2d(fluid_cell, factor, fluid_sdf),
        _coarse_pressure_sdf_value2d(air_cell, factor, fluid_sdf),
    )


@ti.func
def _coarse_pressure_sdf_value3d(cell, factor: int, fluid_sdf):
    base = cell * factor
    value = 0.0
    count = 0
    for ox in range(factor):
        for oy in range(factor):
            for oz in range(factor):
                fine = base + vec3i(ox, oy, oz)
                if fine[0] < fluid_sdf.shape[0] and fine[1] < fluid_sdf.shape[1] and fine[2] < fluid_sdf.shape[2]:
                    value += fluid_sdf[fine]
                    count += 1
    if count > 0:
        value /= ti.cast(count, float)
    return value


@ti.func
def _pressure_free_surface_theta_coarse3d(fluid_cell, air_cell, factor: int, fluid_sdf):
    return _pressure_free_surface_theta_from_phi(
        _coarse_pressure_sdf_value3d(fluid_cell, factor, fluid_sdf),
        _coarse_pressure_sdf_value3d(air_cell, factor, fluid_sdf),
    )


@ti.func
def _mac_laplacian2d(I, grid_size, velocity):
    center = velocity[I]
    lap = 0.0
    if I[0] > 0:
        lap += (velocity[I - vec2i(1, 0)] - center) / (grid_size[0] * grid_size[0])
    if I[0] + 1 < velocity.shape[0]:
        lap += (velocity[I + vec2i(1, 0)] - center) / (grid_size[0] * grid_size[0])
    if I[1] > 0:
        lap += (velocity[I - vec2i(0, 1)] - center) / (grid_size[1] * grid_size[1])
    if I[1] + 1 < velocity.shape[1]:
        lap += (velocity[I + vec2i(0, 1)] - center) / (grid_size[1] * grid_size[1])
    return lap


@ti.func
def _mac_laplacian3d(I, grid_size, velocity):
    center = velocity[I]
    lap = 0.0
    if I[0] > 0:
        lap += (velocity[I - vec3i(1, 0, 0)] - center) / (grid_size[0] * grid_size[0])
    if I[0] + 1 < velocity.shape[0]:
        lap += (velocity[I + vec3i(1, 0, 0)] - center) / (grid_size[0] * grid_size[0])
    if I[1] > 0:
        lap += (velocity[I - vec3i(0, 1, 0)] - center) / (grid_size[1] * grid_size[1])
    if I[1] + 1 < velocity.shape[1]:
        lap += (velocity[I + vec3i(0, 1, 0)] - center) / (grid_size[1] * grid_size[1])
    if I[2] > 0:
        lap += (velocity[I - vec3i(0, 0, 1)] - center) / (grid_size[2] * grid_size[2])
    if I[2] + 1 < velocity.shape[2]:
        lap += (velocity[I + vec3i(0, 0, 1)] - center) / (grid_size[2] * grid_size[2])
    return lap


@ti.func
def _clamp_porosity(phi):
    return ti.min(1.0, ti.max(1.0e-4, phi))


@ti.func
def _mapped_porosity(solid_fraction):
    return ti.min(1.0, ti.max(1.0e-4, 1.0 - solid_fraction))


@ti.func
def _node_control_volume_scale2d(node_id, gnum):
    index = vec2i(vectorize_id(node_id, gnum))
    scale = 1.0
    for d in ti.static(range(2)):
        if index[d] == 0 or index[d] == gnum[d] - 1:
            scale *= 2.0
    return scale


@ti.func
def _node_control_volume_scale3d(node_id, gnum):
    index = vec3i(vectorize_id(node_id, gnum))
    scale = 1.0
    for d in ti.static(range(3)):
        if index[d] == 0 or index[d] == gnum[d] - 1:
            scale *= 2.0
    return scale


@ti.func
def _boundary_face_control_volume_scale(index, count):
    scale = 1.0
    if index == 0 or index == count - 1:
        scale = 2.0
    return scale


@ti.func
def _cell_porosity_from_faces2d(I, face_porosity_x, face_porosity_y):
    phi = face_porosity_x[I] + face_porosity_x[I + vec2i(1, 0)]
    phi += face_porosity_y[I] + face_porosity_y[I + vec2i(0, 1)]
    return _clamp_porosity(0.25 * phi)


@ti.func
def _face_porosity_gradient2d(I, grid_size, face_porosity_x, face_porosity_y):
    return vec2f(
        (face_porosity_x[I + vec2i(1, 0)] - face_porosity_x[I]) / grid_size[0],
        (face_porosity_y[I + vec2i(0, 1)] - face_porosity_y[I]) / grid_size[1],
    )


@ti.func
def _cell_porosity_from_faces3d(I, face_porosity_x, face_porosity_y, face_porosity_z):
    phi = face_porosity_x[I] + face_porosity_x[I + vec3i(1, 0, 0)]
    phi += face_porosity_y[I] + face_porosity_y[I + vec3i(0, 1, 0)]
    phi += face_porosity_z[I] + face_porosity_z[I + vec3i(0, 0, 1)]
    return _clamp_porosity(phi / 6.0)


@ti.func
def _cell_mobility2d(
    I, face_porosity_x, face_porosity_y, cell_solid_density, cell_fluid_density, coupled_mobility: ti.template() = None
):
    phi_f = _cell_porosity_from_faces2d(I, face_porosity_x, face_porosity_y)
    mobility = (1.0 - phi_f) / ti.max(cell_solid_density[I], 1.0e-12) + phi_f / ti.max(cell_fluid_density[I], 1.0e-12)
    if ti.static(coupled_mobility != None):
        mobility = coupled_mobility[I]
    return mobility


@ti.func
def _cell_mobility3d(
    I,
    face_porosity_x,
    face_porosity_y,
    face_porosity_z,
    cell_solid_density,
    cell_fluid_density,
    coupled_mobility: ti.template() = None,
):
    phi_f = _cell_porosity_from_faces3d(I, face_porosity_x, face_porosity_y, face_porosity_z)
    mobility = (1.0 - phi_f) / ti.max(cell_solid_density[I], 1.0e-12) + phi_f / ti.max(cell_fluid_density[I], 1.0e-12)
    if ti.static(coupled_mobility != None):
        mobility = coupled_mobility[I]
    return mobility


@ti.func
def _coarse_mobility2d(
    I,
    factor: int,
    face_porosity_x,
    face_porosity_y,
    cell_solid_density,
    cell_fluid_density,
    coupled_mobility: ti.template() = None,
):
    fine = I * factor
    fine = ti.min(fine, vec2i(cell_solid_density.shape[0] - 1, cell_solid_density.shape[1] - 1))
    return _cell_mobility2d(
        fine, face_porosity_x, face_porosity_y, cell_solid_density, cell_fluid_density, coupled_mobility
    )


@ti.func
def _coarse_mobility3d(
    I,
    factor: int,
    face_porosity_x,
    face_porosity_y,
    face_porosity_z,
    cell_solid_density,
    cell_fluid_density,
    coupled_mobility: ti.template() = None,
):
    fine = I * factor
    fine = ti.min(
        fine, vec3i(cell_solid_density.shape[0] - 1, cell_solid_density.shape[1] - 1, cell_solid_density.shape[2] - 1)
    )
    return _cell_mobility3d(
        fine,
        face_porosity_x,
        face_porosity_y,
        face_porosity_z,
        cell_solid_density,
        cell_fluid_density,
        coupled_mobility,
    )


@ti.func
def _face_porosity_gradient3d(I, grid_size, face_porosity_x, face_porosity_y, face_porosity_z):
    return vec3f(
        (face_porosity_x[I + vec3i(1, 0, 0)] - face_porosity_x[I]) / grid_size[0],
        (face_porosity_y[I + vec3i(0, 1, 0)] - face_porosity_y[I]) / grid_size[1],
        (face_porosity_z[I + vec3i(0, 0, 1)] - face_porosity_z[I]) / grid_size[2],
    )


@ti.func
def _semi_implicit_drag_acceleration_pair(solid_acc, fluid_acc, rel, dt):
    rel2 = rel.dot(rel)
    if rel2 > 1.0e-30 and dt[None] > 0.0:
        # Drag-only form of Juel et al. (2026), Appendix D: with the
        # coefficient frozen at n, backward Euler gives r[n+1]=r[n]/(1+lambda*dt).
        decay_rate = -(fluid_acc - solid_acc).dot(rel) / rel2
        if decay_rate > 0.0:
            scale = 1.0 / (1.0 + decay_rate * dt[None])
            solid_acc *= scale
            fluid_acc *= scale
    return solid_acc, fluid_acc


@ti.func
def _ergun_drag_accelerations(rel, porosity, matProps, dt):
    return _ergun_drag_accelerations_props(
        rel,
        porosity,
        matProps.solid_density,
        matProps.fluid_density,
        matProps.fluid_viscosity,
        matProps.grain_diameter,
        dt,
    )


@ti.func
def _ergun_drag_accelerations_props(rel, porosity, solid_density, fluid_density, fluid_viscosity, grain_diameter, dt):
    phi_f = _clamp_porosity(porosity)
    phi_s = ti.max(1.0e-4, 1.0 - phi_f)
    dc = ti.max(grain_diameter, 1.0e-8)
    rel_norm = ti.sqrt(rel.dot(rel))
    coeff = 150.0 * fluid_viscosity * phi_s * phi_s / (phi_f * dc * dc)
    coeff += 1.75 * fluid_density * phi_s * rel_norm / dc
    solid_acc = coeff * rel / ti.max(solid_density * phi_s, 1.0e-12)
    fluid_acc = -coeff * rel / ti.max(fluid_density * phi_f, 1.0e-12)
    return _semi_implicit_drag_acceleration_pair(solid_acc, fluid_acc, rel, dt)


@ti.func
def _drag_rates_props(
    rel,
    porosity,
    solid_density,
    fluid_density,
    fluid_viscosity,
    grain_diameter,
    permeability,
    fluid_unit_weight,
    drag_model,
):
    phi_f = _clamp_porosity(porosity)
    phi_s = ti.max(1.0e-4, 1.0 - phi_f)
    coeff = 0.0
    if phi_s <= 1.0e-4 or phi_f >= 1.0 - 1.0e-4:
        coeff = 0.0
    elif drag_model >= 1.5:
        dc = ti.max(grain_diameter, 1.0e-8)
        rel_norm = ti.sqrt(rel.dot(rel))
        reynolds = fluid_density * dc * rel_norm / ti.max(fluid_viscosity, 1.0e-12)
        f0 = 10.0 * phi_s / (phi_f * phi_f) + phi_f * phi_f * (1.0 + 1.5 * ti.sqrt(phi_s))
        fre = 0.0
        if reynolds > 1.0e-12:
            fre = 0.413 * reynolds / (24.0 * phi_f * phi_f)
            fre *= (1.0 / phi_f + 3.0 * phi_s * phi_f + 8.4 * reynolds ** (-0.343)) / (
                1.0 + 10.0 ** (3.0 * phi_s) * reynolds ** (-(1.0 + 4.0 * phi_s) / 2.0)
            )
        coeff = phi_f * phi_s * 18.0 * fluid_viscosity * (f0 + fre) / (dc * dc)
    elif drag_model >= 0.5:
        coeff = phi_f * phi_f * fluid_unit_weight / ti.max(permeability, 1.0e-12)
    else:
        dc = ti.max(grain_diameter, 1.0e-8)
        rel_norm = ti.sqrt(rel.dot(rel))
        coeff = 150.0 * fluid_viscosity * phi_s * phi_s / (phi_f * dc * dc)
        coeff += 1.75 * fluid_density * phi_s * rel_norm / dc
    return coeff / ti.max(solid_density * phi_s, 1.0e-12), coeff / ti.max(fluid_density * phi_f, 1.0e-12)


@ti.func
def _intrinsic_porosity_from_phase_masses(solid_mass, fluid_mass, solid_density, fluid_density):
    # Geometric underfill is empty space, not additional pore fluid. The drag
    # inverse must use the same phase-volume ratio as the mapped inertial masses.
    solid_volume = solid_mass / ti.max(solid_density, 1.0e-12)
    fluid_volume = fluid_mass / ti.max(fluid_density, 1.0e-12)
    return fluid_volume / ti.max(solid_volume + fluid_volume, 1.0e-30)


@ti.kernel
def kernel_map_double_layer_phase_volume_support(
    count: int,
    total_nodes: int,
    particle: ti.template(),
    lnid: ti.template(),
    shape: ti.template(),
    node_size: ti.template(),
    support: ti.template(),
):
    for I in ti.grouped(support):
        support[I] = ti.Vector.zero(float, 3)
    for p in range(count):
        if particle[p].active == 1 and particle[p].materialID > 0:
            body = int(particle[p].bodyID)
            for local in range(p * total_nodes, p * total_nodes + int(node_size[p])):
                node_id = lnid[local]
                volume = shape[local] * particle[p].vol
                if particle[p].phase == PHASE_SOLID:
                    support[node_id, body][0] += volume
                    support[node_id, body][2] += volume * (1.0 - particle[p].porosity)
                elif particle[p].phase == PHASE_FLUID:
                    support[node_id, body][1] += volume


@ti.func
def _intrinsic_pore_porosity(node_id, body_id, node: ti.template(), support: ti.template()):
    porosity = node[node_id, body_id].porosity
    if ti.static(support != None):
        volumes = support[node_id, body_id]
        # Empty basis support is outside the material, not additional pore space.
        measure = ti.max(volumes[0], volumes[1])
        porosity = _clamp_porosity(1.0 - volumes[2] / ti.max(measure, 1.0e-30))
    return porosity


@ti.func
def _implicit_phase_accelerations(rel, solid_external, fluid_external, solid_rate, fluid_rate, timestep):
    # Apply the same two-phase inverse to drag and differential forcing.
    slip = (rel + timestep * (fluid_external - solid_external)) / (1.0 + timestep * (solid_rate + fluid_rate))
    return solid_external + solid_rate * slip, fluid_external - fluid_rate * slip


@ti.func
def _coupled_pressure_mobilities(rho_s, rho_f, solid_rate, fluid_rate, timestep):
    return _implicit_phase_accelerations(
        0.0, 1.0 / ti.max(rho_s, 1.0e-12), 1.0 / ti.max(rho_f, 1.0e-12), solid_rate, fluid_rate, timestep
    )


@ti.kernel
def kernel_average_double_layer_pressure_mobility(face_mobility: ti.template(), cell_mobility: ti.template()):
    dimension = ti.static(len(face_mobility))
    for I in ti.grouped(cell_mobility):
        value = 0.0
        for axis in ti.static(range(dimension)):
            value += face_mobility[axis][I] + face_mobility[axis][I + ti.Vector.unit(dimension, axis)]
        cell_mobility[I] = value / (2.0 * dimension)


@ti.func
def _mac_relative_velocity_at_face(
    axis: ti.template(), index, fluid_velocity: ti.template(), solid_velocity: ti.template()
):
    dimension = ti.static(len(fluid_velocity))
    relative = ti.Vector.zero(float, dimension)
    for component in ti.static(range(dimension)):
        if ti.static(component == axis):
            relative[component] = fluid_velocity[component][index] - solid_velocity[component][index]
        else:
            weight = 0.0
            for i, j in ti.static(ti.ndrange(2, 2)):
                offset = ti.Vector.zero(int, dimension)
                offset[axis], offset[component] = i - 1, j
                neighbor = index + offset
                inside = True
                for d in ti.static(range(dimension)):
                    inside = inside and 0 <= neighbor[d] < fluid_velocity[component].shape[d]
                if inside:
                    relative[component] += fluid_velocity[component][neighbor] - solid_velocity[component][neighbor]
                    weight += 1.0
            if weight > 0.0:
                relative[component] /= weight
    return relative


@ti.func
def _drag_accelerations_props(
    rel,
    porosity,
    solid_density,
    fluid_density,
    fluid_viscosity,
    grain_diameter,
    permeability,
    fluid_unit_weight,
    drag_model,
    dt,
):
    solid_rate, fluid_rate = _drag_rates_props(
        rel,
        porosity,
        solid_density,
        fluid_density,
        fluid_viscosity,
        grain_diameter,
        permeability,
        fluid_unit_weight,
        drag_model,
    )
    return _implicit_phase_accelerations(rel, 0.0 * rel, 0.0 * rel, solid_rate, fluid_rate, dt[None])


@ti.func
def _sample_cell_scalar(position, grid_size, cell_type, cell_value):
    base = ti.floor(position / grid_size - 0.5).cast(int)
    value = 0.0
    weight_sum = 0.0
    for i, j in ti.static(ti.ndrange(2, 2)):
        cell = base + vec2i(i, j)
        if 0 <= cell[0] < cell_type.shape[0] and 0 <= cell[1] < cell_type.shape[1]:
            if cell_type[cell] == FLUID_CELL:
                center = (cell.cast(float) + 0.5) * grid_size
                wx = _linear_weight((position[0] - center[0]) / grid_size[0])
                wy = _linear_weight((position[1] - center[1]) / grid_size[1])
                weight = wx * wy
                value += weight * cell_value[cell]
                weight_sum += weight
    if weight_sum > Threshold:
        value /= weight_sum
    return value


@ti.func
def _sample_cell_pressure(position, grid_size, cell_type, cell_pressure):
    base = ti.floor(position / grid_size - 0.5).cast(int)
    value = 0.0
    weight_sum = 0.0
    for i, j in ti.static(ti.ndrange(2, 2)):
        cell = base + vec2i(i, j)
        if 0 <= cell[0] < cell_type.shape[0] and 0 <= cell[1] < cell_type.shape[1]:
            attr = int(cell_type[cell])
            if attr == FLUID_CELL or attr == AIR_CELL:
                center = (cell.cast(float) + 0.5) * grid_size
                wx = _linear_weight((position[0] - center[0]) / grid_size[0])
                wy = _linear_weight((position[1] - center[1]) / grid_size[1])
                weight = wx * wy
                if attr == FLUID_CELL:
                    value += weight * cell_pressure[cell]
                weight_sum += weight
    if weight_sum > Threshold:
        value /= weight_sum
    return value


@ti.func
def _ghost_air_pressure_weight2d(air_cell, cell_type, cell_pressure, fluid_sdf):
    pressure = 0.0
    weight = 0.0
    for d in ti.static(range(2)):
        for side in ti.static(range(2)):
            direction = 1 if side == 0 else -1
            fluid = air_cell + direction * ti.Vector.unit(2, d)
            if 0 <= fluid[0] < cell_type.shape[0] and 0 <= fluid[1] < cell_type.shape[1]:
                if cell_type[fluid] == FLUID_CELL:
                    theta = _pressure_free_surface_theta(fluid, air_cell, fluid_sdf)
                    pressure += -(1.0 - theta) / theta * cell_pressure[fluid]
                    weight += 1.0
    if weight > 0.0:
        pressure /= weight
    return pressure, weight


@ti.func
def _ghost_air_pressure2d(air_cell, cell_type, cell_pressure, fluid_sdf):
    pressure, _ = _ghost_air_pressure_weight2d(air_cell, cell_type, cell_pressure, fluid_sdf)
    return pressure


@ti.func
def _sample_cell_pressure_gfm(position, grid_size, cell_type, cell_pressure, fluid_sdf):
    base = ti.floor(position / grid_size - 0.5).cast(int)
    value = 0.0
    weight_sum = 0.0
    for i, j in ti.static(ti.ndrange(2, 2)):
        cell = base + vec2i(i, j)
        if 0 <= cell[0] < cell_type.shape[0] and 0 <= cell[1] < cell_type.shape[1]:
            attr = int(cell_type[cell])
            if attr == FLUID_CELL or attr == AIR_CELL:
                center = (cell.cast(float) + 0.5) * grid_size
                wx = _linear_weight((position[0] - center[0]) / grid_size[0])
                wy = _linear_weight((position[1] - center[1]) / grid_size[1])
                weight = wx * wy
                if attr == FLUID_CELL:
                    value += weight * cell_pressure[cell]
                    weight_sum += weight
                else:
                    ghost_pressure, ghost_weight = _ghost_air_pressure_weight2d(
                        cell, cell_type, cell_pressure, fluid_sdf
                    )
                    if ghost_weight > 0.0:
                        value += weight * ghost_pressure
                        weight_sum += weight
    if weight_sum > Threshold:
        value /= weight_sum
    return value


@ti.func
def _sample_cell_pressure_gfm_shape2d(
    position,
    psize,
    shape_type: ti.template(),
    influenced_node: int,
    grid_size,
    cell_type,
    cell_pressure,
    fluid_sdf,
    air_ghost_pressure: ti.template() = True,
):
    base = vec2i(
        _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], shape_type),
        _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], shape_type),
    )
    value = 0.0
    weight_sum = 0.0
    for i, j in ti.ndrange(influenced_node, influenced_node):
        cell = base + vec2i(i, j)
        if 0 <= cell[0] < cell_type.shape[0] and 0 <= cell[1] < cell_type.shape[1]:
            attr = int(cell_type[cell])
            if attr == FLUID_CELL or attr == AIR_CELL or attr == SOLID_CELL:
                cell_pos = (cell.cast(float) + 0.5) * grid_size
                # Cell centers are offset half a cell from the wall.  Nodal
                # boundary B-splines have one-sided support there and would
                # give wall-adjacent particles zero total pressure weight.
                # Use the interior basis and renormalize the valid cells below.
                weight = _mac_weight2d(position, cell_pos, grid_size, psize, vec2i(0, 0), shape_type)
                if attr == FLUID_CELL or attr == SOLID_CELL:
                    value += weight * cell_pressure[cell]
                    weight_sum += weight
                else:
                    if ti.static(air_ghost_pressure):
                        ghost_pressure, ghost_weight = _ghost_air_pressure_weight2d(
                            cell, cell_type, cell_pressure, fluid_sdf
                        )
                        if ghost_weight > 0.0:
                            value += weight * ghost_pressure
                            weight_sum += weight
                    else:
                        # Atmospheric air contributes zero physical pressure to the solid force.
                        weight_sum += weight
    if weight_sum > Threshold:
        value /= weight_sum
    return value


@ti.func
def _sample_solid_grid_velocity2d(
    position,
    grid_size,
    gnum,
    cutoff: float,
    shape_type: ti.template(),
    influenced_node: int,
    calLength: ti.template(),
    node: ti.template(),
    sample_force: ti.template() = False,
):
    velocity = ZEROVEC2f
    weight_sum = 0.0
    if shape_type == MAC_SHAPE_LINEAR or shape_type == MAC_SHAPE_GIMP:
        base = ti.floor(position / grid_size).cast(int)
        for i, j in ti.static(ti.ndrange(2, 2)):
            node_ij = base + vec2i(i, j)
            if 0 <= node_ij[0] < gnum[0] and 0 <= node_ij[1] < gnum[1]:
                node_pos = node_ij.cast(float) * grid_size
                weight = _linear_weight((position[0] - node_pos[0]) / grid_size[0])
                weight *= _linear_weight((position[1] - node_pos[1]) / grid_size[1])
                node_id = int(node_ij[0] + node_ij[1] * gnum[0])
                for body_id in range(node.shape[1]):
                    if node[node_id, body_id].ms > cutoff:
                        mass_weight = weight * node[node_id, body_id].ms
                        value = node[node_id, body_id].momentums
                        if ti.static(sample_force):
                            value = node[node_id, body_id].force / node[node_id, body_id].ms
                        velocity += mass_weight * value
                        weight_sum += mass_weight
    else:
        for body_id in range(node.shape[1]):
            psize = calLength[body_id]
            base = vec2i(
                _mac_base_1d(position[0], 1.0 / grid_size[0], 0.0, psize[0], shape_type),
                _mac_base_1d(position[1], 1.0 / grid_size[1], 0.0, psize[1], shape_type),
            )
            for i, j in ti.ndrange(influenced_node, influenced_node):
                node_ij = base + vec2i(i, j)
                if 0 <= node_ij[0] < gnum[0] and 0 <= node_ij[1] < gnum[1]:
                    node_pos = node_ij.cast(float) * grid_size
                    btype = vec2i(
                        _mac_bspline_boundary_type(node_ij[0], gnum[0]),
                        _mac_bspline_boundary_type(node_ij[1], gnum[1]),
                    )
                    weight = _mac_weight2d(position, node_pos, grid_size, psize, btype, shape_type)
                    node_id = int(node_ij[0] + node_ij[1] * gnum[0])
                    if node[node_id, body_id].ms > cutoff:
                        mass_weight = weight * node[node_id, body_id].ms
                        value = node[node_id, body_id].momentums
                        if ti.static(sample_force):
                            value = node[node_id, body_id].force / node[node_id, body_id].ms
                        velocity += mass_weight * value
                        weight_sum += mass_weight
    if weight_sum > cutoff:
        velocity /= weight_sum
    return velocity, weight_sum


@ti.func
def _sample_solid_grid_velocity_linear2d(position, grid_size, gnum, cutoff: float, node: ti.template()):
    base = ti.floor(position / grid_size).cast(int)
    velocity = ZEROVEC2f
    weight_sum = 0.0
    for i, j in ti.static(ti.ndrange(2, 2)):
        node_ij = base + vec2i(i, j)
        if 0 <= node_ij[0] < gnum[0] and 0 <= node_ij[1] < gnum[1]:
            node_pos = node_ij.cast(float) * grid_size
            weight = _linear_weight((position[0] - node_pos[0]) / grid_size[0])
            weight *= _linear_weight((position[1] - node_pos[1]) / grid_size[1])
            node_id = int(node_ij[0] + node_ij[1] * gnum[0])
            for body_id in range(node.shape[1]):
                if node[node_id, body_id].ms > cutoff:
                    mass_weight = weight * node[node_id, body_id].ms
                    velocity += mass_weight * node[node_id, body_id].momentums
                    weight_sum += mass_weight
    if weight_sum > cutoff:
        velocity /= weight_sum
    return velocity, weight_sum


@ti.func
def _sample_mac_velocity(
    position, psize, shape_type: ti.template(), influenced_node: int, grid_size, velocity_x, velocity_y
):
    vx = 0.0
    wx_sum = 0.0
    base_x = vec2i(
        _mac_base_1d(position[0], 1.0 / grid_size[0], 0.0, psize[0], shape_type),
        _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], shape_type),
    )
    for i, j in ti.ndrange(influenced_node, influenced_node):
        face = base_x + vec2i(i, j)
        if 0 <= face[0] < velocity_x.shape[0] and 0 <= face[1] < velocity_x.shape[1]:
            face_pos = vec2f(face[0] * grid_size[0], (face[1] + 0.5) * grid_size[1])
            btype = vec2i(_mac_bspline_boundary_type(face[0], velocity_x.shape[0]), 0)
            weight = _mac_weight2d(position, face_pos, grid_size, psize, btype, shape_type)
            vx += weight * velocity_x[face]
            wx_sum += weight
    if wx_sum > Threshold:
        vx /= wx_sum

    vy = 0.0
    wy_sum = 0.0
    base_y = vec2i(
        _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], shape_type),
        _mac_base_1d(position[1], 1.0 / grid_size[1], 0.0, psize[1], shape_type),
    )
    for i, j in ti.ndrange(influenced_node, influenced_node):
        face = base_y + vec2i(i, j)
        if 0 <= face[0] < velocity_y.shape[0] and 0 <= face[1] < velocity_y.shape[1]:
            face_pos = vec2f((face[0] + 0.5) * grid_size[0], face[1] * grid_size[1])
            btype = vec2i(0, _mac_bspline_boundary_type(face[1], velocity_y.shape[1]))
            weight = _mac_weight2d(position, face_pos, grid_size, psize, btype, shape_type)
            vy += weight * velocity_y[face]
            wy_sum += weight
    if wy_sum > Threshold:
        vy /= wy_sum
    return vec2f(vx, vy)


@ti.func
def _sample_mac_velocity_gradient(
    position, psize, shape_type: ti.template(), influenced_node: int, grid_size, velocity_x, velocity_y, v_pic
):
    gradv = ZEROMAT2x2
    dx_mat = ZEROMAT2x2
    dy_mat = ZEROMAT2x2
    bx = ZEROVEC2f
    by = ZEROVEC2f

    base_x = vec2i(
        _mac_base_1d(position[0], 1.0 / grid_size[0], 0.0, psize[0], shape_type),
        _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], shape_type),
    )
    for i, j in ti.ndrange(influenced_node, influenced_node):
        face = base_x + vec2i(i, j)
        if 0 <= face[0] < velocity_x.shape[0] and 0 <= face[1] < velocity_x.shape[1]:
            face_pos = vec2f(face[0] * grid_size[0], (face[1] + 0.5) * grid_size[1])
            btype = vec2i(_mac_bspline_boundary_type(face[0], velocity_x.shape[0]), 0)
            weight = _mac_weight2d(position, face_pos, grid_size, psize, btype, shape_type)
            pointer = face_pos - position
            dx_mat += weight * pointer.outer_product(pointer)
            bx += weight * (velocity_x[face] - v_pic[0]) * pointer

    base_y = vec2i(
        _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], shape_type),
        _mac_base_1d(position[1], 1.0 / grid_size[1], 0.0, psize[1], shape_type),
    )
    for i, j in ti.ndrange(influenced_node, influenced_node):
        face = base_y + vec2i(i, j)
        if 0 <= face[0] < velocity_y.shape[0] and 0 <= face[1] < velocity_y.shape[1]:
            face_pos = vec2f((face[0] + 0.5) * grid_size[0], face[1] * grid_size[1])
            btype = vec2i(0, _mac_bspline_boundary_type(face[1], velocity_y.shape[1]))
            weight = _mac_weight2d(position, face_pos, grid_size, psize, btype, shape_type)
            pointer = face_pos - position
            dy_mat += weight * pointer.outer_product(pointer)
            by += weight * (velocity_y[face] - v_pic[1]) * pointer

    dx_trace = ti.max(dx_mat[0, 0] + dx_mat[1, 1], 0.0)
    dx_det = ti.abs(dx_mat.determinant())
    if dx_trace > 1.0e-20 and dx_det > 1.0e-8 * dx_trace * dx_trace:
        row = dx_mat.inverse() @ bx
        gradv[0, 0] = row[0]
        gradv[0, 1] = row[1]
    dy_trace = ti.max(dy_mat[0, 0] + dy_mat[1, 1], 0.0)
    dy_det = ti.abs(dy_mat.determinant())
    if dy_trace > 1.0e-20 and dy_det > 1.0e-8 * dy_trace * dy_trace:
        row = dy_mat.inverse() @ by
        gradv[1, 0] = row[0]
        gradv[1, 1] = row[1]
    return gradv


@ti.kernel
def kernel_reset_double_layer_fields(
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity0_x: ti.template(),
    fluid_velocity0_y: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    solid_mass_x: ti.template(),
    solid_mass_y: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    cell_type: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_solid_mass: ti.template(),
    cell_porosity: ti.template(),
    cell_solid_velocity: ti.template(),
    cell_fluid_velocity: ti.template(),
    cell_pressure: ti.template(),
):
    for I in ti.grouped(fluid_mass_x):
        fluid_mass_x[I] = 0.0
        fluid_velocity_x[I] = 0.0
        fluid_velocity0_x[I] = 0.0
        fluid_acceleration_x[I] = 0.0
        solid_mass_x[I] = 0.0
        solid_velocity_x[I] = 0.0
        face_porosity_x[I] = 0.0
    for I in ti.grouped(fluid_mass_y):
        fluid_mass_y[I] = 0.0
        fluid_velocity_y[I] = 0.0
        fluid_velocity0_y[I] = 0.0
        fluid_acceleration_y[I] = 0.0
        solid_mass_y[I] = 0.0
        solid_velocity_y[I] = 0.0
        face_porosity_y[I] = 0.0
    for I in ti.grouped(cell_type):
        cell_type[I] = AIR_CELL
        cell_fluid_mass[I] = 0.0
        cell_solid_mass[I] = 0.0
        cell_porosity[I] = 0.0
        cell_solid_velocity[I] = ZEROVEC2f
        cell_fluid_velocity[I] = ZEROVEC2f
        cell_pressure[I] = 0.0


@ti.kernel
def kernel_reset_double_layer_material_fields(
    face_material_weight_x: ti.template(),
    face_material_weight_y: ti.template(),
    face_solid_density_x: ti.template(),
    face_solid_density_y: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    face_fluid_viscosity_x: ti.template(),
    face_fluid_viscosity_y: ti.template(),
    face_grain_diameter_x: ti.template(),
    face_grain_diameter_y: ti.template(),
    face_permeability_x: ti.template(),
    face_permeability_y: ti.template(),
    face_fluid_unit_weight_x: ti.template(),
    face_fluid_unit_weight_y: ti.template(),
    face_drag_model_x: ti.template(),
    face_drag_model_y: ti.template(),
    cell_material_weight: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    cell_fluid_viscosity: ti.template(),
    cell_grain_diameter: ti.template(),
    node_material_weight: ti.template(),
    node_solid_density: ti.template(),
    node_fluid_density: ti.template(),
    node_fluid_viscosity: ti.template(),
    node_grain_diameter: ti.template(),
    node_permeability: ti.template(),
    node_fluid_unit_weight: ti.template(),
    node_drag_model: ti.template(),
):
    for I in ti.grouped(face_material_weight_x):
        face_material_weight_x[I] = 0.0
        face_solid_density_x[I] = 0.0
        face_fluid_density_x[I] = 0.0
        face_fluid_viscosity_x[I] = 0.0
        face_grain_diameter_x[I] = 0.0
        face_permeability_x[I] = 0.0
        face_fluid_unit_weight_x[I] = 0.0
        face_drag_model_x[I] = 0.0
    for I in ti.grouped(face_material_weight_y):
        face_material_weight_y[I] = 0.0
        face_solid_density_y[I] = 0.0
        face_fluid_density_y[I] = 0.0
        face_fluid_viscosity_y[I] = 0.0
        face_grain_diameter_y[I] = 0.0
        face_permeability_y[I] = 0.0
        face_fluid_unit_weight_y[I] = 0.0
        face_drag_model_y[I] = 0.0
    for I in ti.grouped(cell_material_weight):
        cell_material_weight[I] = 0.0
        cell_solid_density[I] = 0.0
        cell_fluid_density[I] = 0.0
        cell_fluid_viscosity[I] = 0.0
        cell_grain_diameter[I] = 0.0
    for I in ti.grouped(node_material_weight):
        node_material_weight[I] = 0.0
        node_solid_density[I] = 0.0
        node_fluid_density[I] = 0.0
        node_fluid_viscosity[I] = 0.0
        node_grain_diameter[I] = 0.0
        node_permeability[I] = 0.0
        node_fluid_unit_weight[I] = 0.0
        node_drag_model[I] = 0.0


@ti.kernel
def kernel_accumulate_double_layer_material_fields2d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    grid_size: vec2f,
    mac_shape_type: ti.template(),
    mac_influenced_node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    matProps: ti.template(),
    calLength: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    face_material_weight_x: ti.template(),
    face_material_weight_y: ti.template(),
    face_solid_density_x: ti.template(),
    face_solid_density_y: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    face_fluid_viscosity_x: ti.template(),
    face_fluid_viscosity_y: ti.template(),
    face_grain_diameter_x: ti.template(),
    face_grain_diameter_y: ti.template(),
    face_permeability_x: ti.template(),
    face_permeability_y: ti.template(),
    face_fluid_unit_weight_x: ti.template(),
    face_fluid_unit_weight_y: ti.template(),
    face_drag_model_x: ti.template(),
    face_drag_model_y: ti.template(),
    cell_material_weight: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    cell_fluid_viscosity: ti.template(),
    cell_grain_diameter: ti.template(),
    node_material_weight: ti.template(),
    node_solid_density: ti.template(),
    node_fluid_density: ti.template(),
    node_fluid_viscosity: ti.template(),
    node_grain_diameter: ti.template(),
    node_permeability: ti.template(),
    node_fluid_unit_weight: ti.template(),
    node_drag_model: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            position = particle[np].x
            psize = calLength[int(particle[np].bodyID)]
            mass = ti.max(particle[np].m, particle[np].ms + particle[np].mf)
            if mass > Threshold:
                cell = ti.floor(position / grid_size).cast(int)
                if 0 <= cell[0] < cell_material_weight.shape[0] and 0 <= cell[1] < cell_material_weight.shape[1]:
                    cell_material_weight[cell] += mass
                    cell_solid_density[cell] += mass * matProps.solid_density
                    cell_fluid_density[cell] += mass * matProps.fluid_density
                    cell_fluid_viscosity[cell] += mass * matProps.fluid_viscosity
                    cell_grain_diameter[cell] += mass * matProps.grain_diameter

                base_x = vec2i(
                    _mac_base_1d(position[0], 1.0 / grid_size[0], 0.0, psize[0], mac_shape_type),
                    _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], mac_shape_type),
                )
                for ix, iy in ti.ndrange(mac_influenced_node, mac_influenced_node):
                    face = base_x + vec2i(ix, iy)
                    if (
                        0 <= face[0] < face_material_weight_x.shape[0]
                        and 0 <= face[1] < face_material_weight_x.shape[1]
                    ):
                        face_pos = vec2f(face[0] * grid_size[0], (face[1] + 0.5) * grid_size[1])
                        btype = vec2i(_mac_bspline_boundary_type(face[0], face_material_weight_x.shape[0]), 0)
                        weight = _mac_weight2d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                        weighted_mass = weight * mass
                        face_material_weight_x[face] += weighted_mass
                        face_solid_density_x[face] += weighted_mass * matProps.solid_density
                        face_fluid_density_x[face] += weighted_mass * matProps.fluid_density
                        face_fluid_viscosity_x[face] += weighted_mass * matProps.fluid_viscosity
                        face_grain_diameter_x[face] += weighted_mass * matProps.grain_diameter
                        face_permeability_x[face] += weighted_mass * matProps.permeability
                        face_fluid_unit_weight_x[face] += weighted_mass * matProps.fluid_unit_weight
                        face_drag_model_x[face] += weighted_mass * matProps.drag_model

                base_y = vec2i(
                    _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], mac_shape_type),
                    _mac_base_1d(position[1], 1.0 / grid_size[1], 0.0, psize[1], mac_shape_type),
                )
                for ix, iy in ti.ndrange(mac_influenced_node, mac_influenced_node):
                    face = base_y + vec2i(ix, iy)
                    if (
                        0 <= face[0] < face_material_weight_y.shape[0]
                        and 0 <= face[1] < face_material_weight_y.shape[1]
                    ):
                        face_pos = vec2f((face[0] + 0.5) * grid_size[0], face[1] * grid_size[1])
                        btype = vec2i(0, _mac_bspline_boundary_type(face[1], face_material_weight_y.shape[1]))
                        weight = _mac_weight2d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                        weighted_mass = weight * mass
                        face_material_weight_y[face] += weighted_mass
                        face_solid_density_y[face] += weighted_mass * matProps.solid_density
                        face_fluid_density_y[face] += weighted_mass * matProps.fluid_density
                        face_fluid_viscosity_y[face] += weighted_mass * matProps.fluid_viscosity
                        face_grain_diameter_y[face] += weighted_mass * matProps.grain_diameter
                        face_permeability_y[face] += weighted_mass * matProps.permeability
                        face_fluid_unit_weight_y[face] += weighted_mass * matProps.fluid_unit_weight
                        face_drag_model_y[face] += weighted_mass * matProps.drag_model

                bodyID = int(particle[np].bodyID)
                offset = np * total_nodes
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    weighted_mass = shapefn[ln] * mass
                    node_material_weight[nodeID, bodyID] += weighted_mass
                    node_solid_density[nodeID, bodyID] += weighted_mass * matProps.solid_density
                    node_fluid_density[nodeID, bodyID] += weighted_mass * matProps.fluid_density
                    node_fluid_viscosity[nodeID, bodyID] += weighted_mass * matProps.fluid_viscosity
                    node_grain_diameter[nodeID, bodyID] += weighted_mass * matProps.grain_diameter
                    node_permeability[nodeID, bodyID] += weighted_mass * matProps.permeability
                    node_fluid_unit_weight[nodeID, bodyID] += weighted_mass * matProps.fluid_unit_weight
                    node_drag_model[nodeID, bodyID] += weighted_mass * matProps.drag_model


@ti.kernel
def kernel_normalize_double_layer_material_fields(
    default_matProps: ti.template(),
    face_material_weight_x: ti.template(),
    face_material_weight_y: ti.template(),
    face_solid_density_x: ti.template(),
    face_solid_density_y: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    face_fluid_viscosity_x: ti.template(),
    face_fluid_viscosity_y: ti.template(),
    face_grain_diameter_x: ti.template(),
    face_grain_diameter_y: ti.template(),
    face_permeability_x: ti.template(),
    face_permeability_y: ti.template(),
    face_fluid_unit_weight_x: ti.template(),
    face_fluid_unit_weight_y: ti.template(),
    face_drag_model_x: ti.template(),
    face_drag_model_y: ti.template(),
    cell_material_weight: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    cell_fluid_viscosity: ti.template(),
    cell_grain_diameter: ti.template(),
    node_material_weight: ti.template(),
    node_solid_density: ti.template(),
    node_fluid_density: ti.template(),
    node_fluid_viscosity: ti.template(),
    node_grain_diameter: ti.template(),
    node_permeability: ti.template(),
    node_fluid_unit_weight: ti.template(),
    node_drag_model: ti.template(),
):
    for I in ti.grouped(face_material_weight_x):
        if face_material_weight_x[I] > Threshold:
            inv_weight = 1.0 / face_material_weight_x[I]
            face_solid_density_x[I] *= inv_weight
            face_fluid_density_x[I] *= inv_weight
            face_fluid_viscosity_x[I] *= inv_weight
            face_grain_diameter_x[I] *= inv_weight
            face_permeability_x[I] *= inv_weight
            face_fluid_unit_weight_x[I] *= inv_weight
            face_drag_model_x[I] *= inv_weight
        else:
            face_solid_density_x[I] = default_matProps.solid_density
            face_fluid_density_x[I] = default_matProps.fluid_density
            face_fluid_viscosity_x[I] = default_matProps.fluid_viscosity
            face_grain_diameter_x[I] = default_matProps.grain_diameter
            face_permeability_x[I] = default_matProps.permeability
            face_fluid_unit_weight_x[I] = default_matProps.fluid_unit_weight
            face_drag_model_x[I] = default_matProps.drag_model
    for I in ti.grouped(face_material_weight_y):
        if face_material_weight_y[I] > Threshold:
            inv_weight = 1.0 / face_material_weight_y[I]
            face_solid_density_y[I] *= inv_weight
            face_fluid_density_y[I] *= inv_weight
            face_fluid_viscosity_y[I] *= inv_weight
            face_grain_diameter_y[I] *= inv_weight
            face_permeability_y[I] *= inv_weight
            face_fluid_unit_weight_y[I] *= inv_weight
            face_drag_model_y[I] *= inv_weight
        else:
            face_solid_density_y[I] = default_matProps.solid_density
            face_fluid_density_y[I] = default_matProps.fluid_density
            face_fluid_viscosity_y[I] = default_matProps.fluid_viscosity
            face_grain_diameter_y[I] = default_matProps.grain_diameter
            face_permeability_y[I] = default_matProps.permeability
            face_fluid_unit_weight_y[I] = default_matProps.fluid_unit_weight
            face_drag_model_y[I] = default_matProps.drag_model
    for I in ti.grouped(cell_material_weight):
        if cell_material_weight[I] > Threshold:
            inv_weight = 1.0 / cell_material_weight[I]
            cell_solid_density[I] *= inv_weight
            cell_fluid_density[I] *= inv_weight
            cell_fluid_viscosity[I] *= inv_weight
            cell_grain_diameter[I] *= inv_weight
        else:
            cell_solid_density[I] = default_matProps.solid_density
            cell_fluid_density[I] = default_matProps.fluid_density
            cell_fluid_viscosity[I] = default_matProps.fluid_viscosity
            cell_grain_diameter[I] = default_matProps.grain_diameter
    for I in ti.grouped(node_material_weight):
        if node_material_weight[I] > Threshold:
            inv_weight = 1.0 / node_material_weight[I]
            node_solid_density[I] *= inv_weight
            node_fluid_density[I] *= inv_weight
            node_fluid_viscosity[I] *= inv_weight
            node_grain_diameter[I] *= inv_weight
            node_permeability[I] *= inv_weight
            node_fluid_unit_weight[I] *= inv_weight
            node_drag_model[I] *= inv_weight
        else:
            node_solid_density[I] = default_matProps.solid_density
            node_fluid_density[I] = default_matProps.fluid_density
            node_fluid_viscosity[I] = default_matProps.fluid_viscosity
            node_grain_diameter[I] = default_matProps.grain_diameter
            node_permeability[I] = default_matProps.permeability
            node_fluid_unit_weight[I] = default_matProps.fluid_unit_weight
            node_drag_model[I] = default_matProps.drag_model


@ti.func
def _double_layer_fluid_fraction2d(I, grid_size, cell_fluid_mass, cell_fluid_density, cell_porosity):
    capacity = cell_fluid_density[I] * _clamp_porosity(cell_porosity[I]) * grid_size[0] * grid_size[1]
    return ti.min(1.0, ti.max(0.0, cell_fluid_mass[I] / ti.max(capacity, 1.0e-12)))


@ti.kernel
def kernel_classify_double_layer_fluid_cells2d(
    particleNum: int,
    grid_size: vec2f,
    particle: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_fluid_density: ti.template(),
    cell_porosity: ti.template(),
    cell_type: ti.template(),
):
    for I in ti.grouped(cell_type):
        cell_type[I] = AIR_CELL
        if _double_layer_fluid_fraction2d(I, grid_size, cell_fluid_mass, cell_fluid_density, cell_porosity) >= 0.5:
            cell_type[I] = FLUID_CELL

    # Detached sub-cell spray stays pressure-free; attached surface support is retained.
    for np in range(particleNum):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
        ):
            cell = ti.floor(particle[np].x / grid_size).cast(int)
            if 0 <= cell[0] < cell_type.shape[0] and 0 <= cell[1] < cell_type.shape[1] and cell_type[cell] == AIR_CELL:
                resolved = False
                for d in ti.static(range(2)):
                    unit = ti.Vector.unit(2, d)
                    left = cell - unit
                    right = cell + unit
                    if 0 <= left[d]:
                        if cell_type[left] == FLUID_CELL:
                            resolved = True
                    if right[d] < cell_type.shape[d]:
                        if cell_type[right] == FLUID_CELL:
                            resolved = True
                # Saturated porous cells carry pressure even beside a resolved water cell.
                if cell_porosity[cell] < 1.0 - Threshold:
                    cell_type[cell] = 3
                elif not resolved:
                    attached = False
                    for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                        neighbor = cell + offset - vec2i(1, 1)
                        if 0 <= neighbor[0] < cell_type.shape[0] and 0 <= neighbor[1] < cell_type.shape[1]:
                            attached = attached or cell_type[neighbor] == FLUID_CELL
                    if attached:
                        cell_type[cell] = 3
    for I in ti.grouped(cell_type):
        if cell_type[I] == 3:
            cell_type[I] = FLUID_CELL

    # Fill interior deficits between resolved liquid cells, never free-surface tails.
    for i in range(cell_type.shape[0]):
        lower = -1
        for j in range(cell_type.shape[1]):
            if cell_fluid_mass[i, j] <= Threshold:
                lower = -1
            elif (
                _double_layer_fluid_fraction2d(
                    vec2i(i, j), grid_size, cell_fluid_mass, cell_fluid_density, cell_porosity
                )
                >= 0.5
            ):
                if lower >= 0:
                    for interior in range(lower + 1, j):
                        cell_type[i, interior] = FLUID_CELL
                lower = j


@ti.kernel
def kernel_extend_double_layer_cell_support_at_solid_walls(
    particle_count: int,
    grid_size: ti.template(),
    particle: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_fluid_density: ti.template(),
    cell_porosity: ti.template(),
    cell_type: ti.template(),
    mac_shape_type: ti.template(),
):
    dimension = ti.static(len(cell_type.shape))
    cell_shape_type, support = MAC_SHAPE_LINEAR, 2
    if mac_shape_type == MAC_SHAPE_QUAD_BSPLINE:
        cell_shape_type, support = MAC_SHAPE_QUAD_BSPLINE, 3
    cell_volume = 1.0
    for d in ti.static(range(dimension)):
        cell_volume *= grid_size[d]
    for p in range(particle_count):
        if int(particle[p].active) == 1 and int(particle[p].materialID) > 0:
            position = particle[p].x
            source = ti.floor(position / grid_size).cast(int)
            inside = True
            for d in ti.static(range(dimension)):
                inside = inside and 0 <= source[d] < cell_type.shape[d]
            if inside and cell_type[source] != SOLID_CELL:
                wall = ti.Vector.zero(int, dimension)
                for d in ti.static(range(dimension)):
                    unit = ti.Vector.unit(dimension, d)
                    if source[d] == 0 or cell_type[source - unit] == SOLID_CELL:
                        wall[d] = -1
                    if source[d] + 1 == cell_type.shape[d] or cell_type[source + unit] == SOLID_CELL:
                        if wall[d] == 0 or position[d] > (source[d] + 0.5) * grid_size[d]:
                            wall[d] = 1
                if wall.dot(wall) > 0:
                    # Mirror missing interpolation support, not physical particle mass.
                    for image in range(1, 2**dimension):
                        mirrored = position + ti.Vector.zero(float, dimension)
                        valid_image = True
                        for d in ti.static(range(dimension)):
                            if image & (1 << d):
                                valid_image = valid_image and wall[d] != 0
                                plane = (source[d] + (1 if wall[d] > 0 else 0)) * grid_size[d]
                                mirrored[d] = 2.0 * plane - position[d]
                        if valid_image:
                            base = ti.Vector.zero(int, dimension)
                            for d in ti.static(range(dimension)):
                                base[d] = _mac_base_1d(
                                    mirrored[d], 1.0 / grid_size[d], 0.5, 0.5 * grid_size[d], cell_shape_type
                                )
                            for offset in range(support**dimension):
                                target = base + ti.Vector.zero(int, dimension)
                                local = offset
                                for d in ti.static(range(dimension)):
                                    target[d] += local % support
                                    local //= support
                                valid_target = True
                                for d in ti.static(range(dimension)):
                                    valid_target = valid_target and 0 <= target[d] < cell_type.shape[d]
                                    if image & (1 << d):
                                        valid_target = valid_target and target[d] == source[d]
                                if valid_target:
                                    for d in ti.static(range(dimension)):
                                        if image & (1 << d):
                                            neighbor = target + wall[d] * ti.Vector.unit(dimension, d)
                                            if 0 <= neighbor[d] < cell_type.shape[d]:
                                                valid_target = valid_target and cell_type[neighbor] == SOLID_CELL
                                    if valid_target and cell_type[target] != SOLID_CELL:
                                        weight = 1.0
                                        for d in ti.static(range(dimension)):
                                            center = (target[d] + 0.5) * grid_size[d]
                                            weight *= _mac_weight_1d(
                                                mirrored[d],
                                                center,
                                                1.0 / grid_size[d],
                                                0.5 * grid_size[d],
                                                0,
                                                cell_shape_type,
                                            )
                                        if int(particle[p].phase) == PHASE_FLUID:
                                            cell_fluid_mass[target] += weight * particle[p].mf
                                        elif int(particle[p].phase) == PHASE_SOLID:
                                            cell_porosity[target] -= (
                                                weight * particle[p].vol * (1.0 - particle[p].porosity) / cell_volume
                                            )
    for I in ti.grouped(cell_type):
        cell_porosity[I] = _clamp_porosity(cell_porosity[I])
        if cell_type[I] == AIR_CELL:
            capacity = cell_fluid_density[I] * cell_porosity[I] * cell_volume
            if cell_fluid_mass[I] >= 0.5 * ti.max(capacity, 1.0e-12):
                cell_type[I] = FLUID_CELL


@ti.kernel
def kernel_extend_double_layer_face_support_at_solid_walls(
    count: int,
    grid_size: ti.template(),
    particle: ti.template(),
    kind: ti.template(),
    faces: ti.template(),
    cal_length: ti.template(),
    shape_type: ti.template(),
    support: int,
):
    dimension = ti.static(len(kind.shape))
    volume = 1.0
    for d in ti.static(range(dimension)):
        volume *= grid_size[d]
    for p in range(count):
        if particle[p].active == 1 and particle[p].materialID > 0 and particle[p].phase == PHASE_SOLID:
            position = particle[p].x
            psize = cal_length[int(particle[p].bodyID)]
            source = ti.floor(position / grid_size).cast(int)
            inside = True
            for d in ti.static(range(dimension)):
                inside = inside and 0 <= source[d] < kind.shape[d]
            if inside and kind[source] != SOLID_CELL:
                wall = ti.Vector.zero(int, dimension)
                wall_cell = source + ti.Vector.zero(int, dimension)
                for d in ti.static(range(dimension)):
                    nearest = support * grid_size[d]
                    for side in ti.static([-1, 1]):
                        for distance in range(1, support):
                            neighbor = source + side * distance * ti.Vector.unit(dimension, d)
                            if 0 <= neighbor[d] < kind.shape[d] and kind[neighbor] == SOLID_CELL:
                                boundary_cell = neighbor[d] - side
                                plane_index = boundary_cell + (1 if side > 0 else 0)
                                separation = ti.abs(position[d] - plane_index * grid_size[d])
                                if separation < nearest:
                                    nearest = separation
                                    wall[d] = side
                                    wall_cell[d] = boundary_cell
                if wall.dot(wall) > 0:
                    for image in range(1, 2**dimension):
                        mirrored = position + ti.Vector.zero(float, dimension)
                        valid_image = True
                        for d in ti.static(range(dimension)):
                            if image & (1 << d):
                                valid_image = valid_image and wall[d] != 0
                                plane = (wall_cell[d] + (1 if wall[d] > 0 else 0)) * grid_size[d]
                                mirrored[d] = 2 * plane - position[d]
                        if valid_image:
                            for axis in ti.static(range(dimension)):
                                base = ti.Vector.zero(int, dimension)
                                for d in ti.static(range(dimension)):
                                    shift = 0.0 if d == axis else 0.5
                                    base[d] = _mac_base_1d(mirrored[d], 1 / grid_size[d], shift, psize[d], shape_type)
                                for flat in range(support**dimension):
                                    face = base + ti.Vector.zero(int, dimension)
                                    local = flat
                                    valid = True
                                    for d in ti.static(range(dimension)):
                                        face[d] += local % support
                                        local //= support
                                        valid = valid and 0 <= face[d] < faces[axis].shape[d]
                                        if image & (1 << d):
                                            plane_index = wall_cell[d] + (1 if wall[d] > 0 else 0)
                                            shift = 0.0 if d == axis else 0.5
                                            if wall[d] < 0:
                                                valid = valid and face[d] + shift >= plane_index
                                            else:
                                                valid = valid and face[d] + shift <= plane_index
                                    if valid:
                                        # An image supplies only the original side of this wall.
                                        target = face + ti.Vector.zero(int, dimension)
                                        for d in ti.static(range(dimension)):
                                            target[d] = ti.min(target[d], kind.shape[d] - 1)
                                            if image & (1 << d):
                                                target[d] = wall_cell[d]
                                        for d in ti.static(range(dimension)):
                                            if image & (1 << d):
                                                neighbor = target + wall[d] * ti.Vector.unit(dimension, d)
                                                if 0 <= neighbor[d] < kind.shape[d]:
                                                    valid = valid and kind[neighbor] == SOLID_CELL
                                        if valid and kind[target] != SOLID_CELL:
                                            weight = 1.0
                                            for d in ti.static(range(dimension)):
                                                shift = 0.0 if d == axis else 0.5
                                                weight *= _mac_weight_1d(
                                                    mirrored[d],
                                                    (face[d] + shift) * grid_size[d],
                                                    1 / grid_size[d],
                                                    psize[d],
                                                    (
                                                        _mac_bspline_boundary_type(face[d], faces[axis].shape[d])
                                                        if d == axis
                                                        else 0
                                                    ),
                                                    shape_type,
                                                )
                                            faces[axis][face] -= (
                                                weight
                                                * particle[p].vol
                                                * (1 - particle[p].porosity)
                                                / volume
                                                * _boundary_face_control_volume_scale(
                                                    face[axis], faces[axis].shape[axis]
                                                )
                                            )
    for axis in ti.static(range(dimension)):
        for I in ti.grouped(faces[axis]):
            faces[axis][I] = _clamp_porosity(faces[axis][I])


@ti.kernel
def kernel_build_double_layer_fluid_sdf2d(
    grid_size: vec2f,
    cell_fluid_mass: ti.template(),
    cell_fluid_density: ti.template(),
    cell_porosity: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
):
    min_dx = ti.min(grid_size[0], grid_size[1])
    minimum_distance = 0.1 * min_dx
    for I in ti.grouped(fluid_sdf):
        fraction = _double_layer_fluid_fraction2d(I, grid_size, cell_fluid_mass, cell_fluid_density, cell_porosity)
        phi = (0.5 - fraction) * min_dx
        if cell_type[I] == FLUID_CELL:
            phi = ti.min(phi, -minimum_distance)
        else:
            phi = ti.max(phi, minimum_distance)
        fluid_sdf[I] = phi


@ti.kernel
def kernel_mark_double_layer_solid_cell_region2d(
    grid_size: vec2f,
    start_point: vec2f,
    end_point: vec2f,
    cell_type: ti.template(),
):
    for I in ti.grouped(cell_type):
        inside = True
        for d in ti.static(range(2)):
            cell_center = (ti.cast(I[d], float) + 0.5) * grid_size[d]
            eps = 1.0e-6 * grid_size[d]
            inside = inside and cell_center >= start_point[d] - eps and cell_center <= end_point[d] + eps
        if inside:
            cell_type[I] = SOLID_CELL


@ti.kernel
def kernel_mark_double_layer_solid_plane_region2d(
    grid_size: vec2f,
    start_point: vec2f,
    end_point: vec2f,
    plane_point: vec2f,
    plane_normal: vec2f,
    cell_type: ti.template(),
):
    eps = 1.0e-6 * ti.min(grid_size[0], grid_size[1])
    for I in ti.grouped(cell_type):
        cell_center = (ti.cast(I, float) + vec2f(0.5, 0.5)) * grid_size
        inside = True
        for d in ti.static(range(2)):
            inside = inside and cell_center[d] >= start_point[d] - eps and cell_center[d] <= end_point[d] + eps
        signed_distance = (cell_center - plane_point).dot(plane_normal)
        if inside and signed_distance >= -eps:
            cell_type[I] = SOLID_CELL


@ti.kernel
def kernel_constrain_double_layer_particles_to_solid_region2d(
    particle_count: int,
    domain: vec2f,
    grid_size: vec2f,
    start_point: vec2f,
    end_point: vec2f,
    particle: ti.template(),
):
    eps = 1.0e-6 * ti.max(domain[0], domain[1])
    discrete_start = ZEROVEC2f
    discrete_end = ZEROVEC2f
    for d in ti.static(range(2)):
        first_cell = ti.ceil(start_point[d] / grid_size[d] - 0.5)
        last_cell = ti.floor(end_point[d] / grid_size[d] - 0.5)
        discrete_start[d] = ti.max(0.0, first_cell * grid_size[d])
        discrete_end[d] = ti.min(domain[d], (last_cell + 1.0) * grid_size[d])
    for np in range(particle_count):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            position = particle[np].x
            inside = True
            for d in ti.static(range(2)):
                inside = inside and position[d] >= discrete_start[d] - eps and position[d] <= discrete_end[d] + eps
            if inside:
                prefer_boundary_axis = 0
                for d in ti.static(range(2)):
                    if (discrete_start[d] <= eps) != (discrete_end[d] >= domain[d] - eps):
                        prefer_boundary_axis = 1

                best_axis = -1
                best_distance = 1.0e30
                target = 0.0
                normal = 0.0
                for d in ti.static(range(2)):
                    use_axis = (
                        prefer_boundary_axis == 0 or discrete_start[d] <= eps or discrete_end[d] >= domain[d] - eps
                    ) and not (
                        # A domain-spanning tangent cannot release the particle after domain clamping.
                        discrete_start[d] <= eps
                        and discrete_end[d] >= domain[d] - eps
                    )
                    if use_axis:
                        candidate = 0.0
                        candidate_normal = 0.0
                        if discrete_start[d] <= eps:
                            candidate = discrete_end[d] + eps
                            candidate_normal = 1.0
                        elif discrete_end[d] >= domain[d] - eps:
                            candidate = discrete_start[d] - eps
                            candidate_normal = -1.0
                        else:
                            lower_distance = ti.abs(position[d] - discrete_start[d])
                            upper_distance = ti.abs(discrete_end[d] - position[d])
                            if lower_distance < upper_distance:
                                candidate = discrete_start[d] - eps
                                candidate_normal = -1.0
                            else:
                                candidate = discrete_end[d] + eps
                                candidate_normal = 1.0
                        distance = ti.abs(candidate - position[d])
                        if distance < best_distance:
                            best_distance = distance
                            best_axis = d
                            target = candidate
                            normal = candidate_normal

                for d in ti.static(range(2)):
                    if best_axis == d:
                        particle[np].x[d] = ti.min(ti.max(target, 1.0e-8), domain[d] - 1.0e-8)
                        if particle[np].v[d] * normal < 0.0:
                            particle[np].v[d] = 0.0
                        if int(particle[np].phase) == PHASE_SOLID:
                            if particle[np].vs[d] * normal < 0.0:
                                particle[np].vs[d] = 0.0
                        elif int(particle[np].phase) == PHASE_FLUID:
                            if particle[np].vf[d] * normal < 0.0:
                                particle[np].vf[d] = 0.0


@ti.kernel
def kernel_constrain_double_layer_particles_to_solid_plane_region2d(
    particle_count: int,
    domain: vec2f,
    start_point: vec2f,
    end_point: vec2f,
    plane_point: vec2f,
    plane_normal: vec2f,
    particle: ti.template(),
):
    eps = 1.0e-6 * ti.max(domain[0], domain[1])
    for np in range(particle_count):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            position = particle[np].x
            inside = True
            for d in ti.static(range(2)):
                inside = inside and position[d] >= start_point[d] - eps and position[d] <= end_point[d] + eps
            signed_distance = (position - plane_point).dot(plane_normal)
            if inside and signed_distance >= -eps:
                correction = signed_distance + eps
                new_position = position - correction * plane_normal
                for d in ti.static(range(2)):
                    particle[np].x[d] = ti.min(ti.max(new_position[d], 1.0e-8), domain[d] - 1.0e-8)

                normal_speed = particle[np].v.dot(plane_normal)
                if normal_speed > 0.0:
                    particle[np].v -= normal_speed * plane_normal
                if int(particle[np].phase) == PHASE_SOLID:
                    solid_normal_speed = particle[np].vs.dot(plane_normal)
                    if solid_normal_speed > 0.0:
                        particle[np].vs -= solid_normal_speed * plane_normal
                elif int(particle[np].phase) == PHASE_FLUID:
                    fluid_normal_speed = particle[np].vf.dot(plane_normal)
                    if fluid_normal_speed > 0.0:
                        particle[np].vf -= fluid_normal_speed * plane_normal


@ti.kernel
def kernel_enforce_double_layer_solid_plane_nodes2d(
    cutoff: float,
    grid_size: vec2f,
    gnum: ti.types.vector(2, int),
    start_point: vec2f,
    end_point: vec2f,
    plane_point: vec2f,
    plane_normal: vec2f,
    node: ti.template(),
):
    eps = 1.0e-6 * ti.min(grid_size[0], grid_size[1])
    for ng, nb in node:
        if node[ng, nb].ms > cutoff:
            position = grid_size * vec2f(vectorize_id(ng, gnum))
            inside = True
            for d in ti.static(range(2)):
                inside = inside and position[d] >= start_point[d] - eps and position[d] <= end_point[d] + eps
            if inside and (position - plane_point).dot(plane_normal) >= -eps:
                node[ng, nb].momentums -= node[ng, nb].momentums.dot(plane_normal) * plane_normal
                node[ng, nb].momentum -= node[ng, nb].momentum.dot(plane_normal) * plane_normal
                node[ng, nb].forces -= node[ng, nb].forces.dot(plane_normal) * plane_normal
                node[ng, nb].force -= node[ng, nb].force.dot(plane_normal) * plane_normal


@ti.kernel
def kernel_mark_double_layer_solid_plane_region3d(
    grid_size: vec3f,
    start_point: vec3f,
    end_point: vec3f,
    plane_point: vec3f,
    plane_normal: vec3f,
    cell_type: ti.template(),
):
    eps = 1.0e-6 * ti.min(grid_size[0], ti.min(grid_size[1], grid_size[2]))
    for I in ti.grouped(cell_type):
        cell_center = (ti.cast(I, float) + vec3f(0.5, 0.5, 0.5)) * grid_size
        inside = True
        for d in ti.static(range(3)):
            inside = inside and cell_center[d] >= start_point[d] - eps and cell_center[d] <= end_point[d] + eps
        if inside and (cell_center - plane_point).dot(plane_normal) >= -eps:
            cell_type[I] = SOLID_CELL


@ti.kernel
def kernel_constrain_double_layer_particles_to_solid_plane_region3d(
    particle_count: int,
    domain: vec3f,
    start_point: vec3f,
    end_point: vec3f,
    plane_point: vec3f,
    plane_normal: vec3f,
    particle: ti.template(),
):
    eps = 1.0e-6 * ti.max(domain[0], ti.max(domain[1], domain[2]))
    for np in range(particle_count):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            position = particle[np].x
            inside = True
            for d in ti.static(range(3)):
                inside = inside and position[d] >= start_point[d] - eps and position[d] <= end_point[d] + eps
            signed_distance = (position - plane_point).dot(plane_normal)
            if inside and signed_distance >= -eps:
                new_position = position - (signed_distance + eps) * plane_normal
                for d in ti.static(range(3)):
                    particle[np].x[d] = ti.min(ti.max(new_position[d], 1.0e-8), domain[d] - 1.0e-8)
                normal_speed = particle[np].v.dot(plane_normal)
                if normal_speed > 0.0:
                    particle[np].v -= normal_speed * plane_normal
                if int(particle[np].phase) == PHASE_SOLID:
                    solid_normal_speed = particle[np].vs.dot(plane_normal)
                    if solid_normal_speed > 0.0:
                        particle[np].vs -= solid_normal_speed * plane_normal
                elif int(particle[np].phase) == PHASE_FLUID:
                    fluid_normal_speed = particle[np].vf.dot(plane_normal)
                    if fluid_normal_speed > 0.0:
                        particle[np].vf -= fluid_normal_speed * plane_normal


@ti.kernel
def kernel_enforce_double_layer_solid_plane_nodes3d(
    cutoff: float,
    grid_size: vec3f,
    gnum: ti.types.vector(3, int),
    start_point: vec3f,
    end_point: vec3f,
    plane_point: vec3f,
    plane_normal: vec3f,
    node: ti.template(),
):
    eps = 1.0e-6 * ti.min(grid_size[0], ti.min(grid_size[1], grid_size[2]))
    for ng, nb in node:
        if node[ng, nb].ms > cutoff:
            position = grid_size * vec3f(vectorize_id(ng, gnum))
            inside = True
            for d in ti.static(range(3)):
                inside = inside and position[d] >= start_point[d] - eps and position[d] <= end_point[d] + eps
            if inside and (position - plane_point).dot(plane_normal) >= -eps:
                node[ng, nb].momentums -= node[ng, nb].momentums.dot(plane_normal) * plane_normal
                node[ng, nb].momentum -= node[ng, nb].momentum.dot(plane_normal) * plane_normal
                node[ng, nb].forces -= node[ng, nb].forces.dot(plane_normal) * plane_normal
                node[ng, nb].force -= node[ng, nb].force.dot(plane_normal) * plane_normal


@ti.kernel
def kernel_enforce_double_layer_solid_cell_faces2d(
    cell_type: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
):
    for I in ti.grouped(fluid_velocity_x):
        if 0 < I[0] < fluid_velocity_x.shape[0] - 1:
            left = I - vec2i(1, 0)
            right = I
            if (cell_type[left] == SOLID_CELL and cell_type[right] == FLUID_CELL) or (
                cell_type[left] == FLUID_CELL and cell_type[right] == SOLID_CELL
            ):
                fluid_velocity_x[I] = solid_velocity_x[I]
                fluid_acceleration_x[I] = 0.0
    for I in ti.grouped(fluid_velocity_y):
        if 0 < I[1] < fluid_velocity_y.shape[1] - 1:
            down = I - vec2i(0, 1)
            up = I
            if (cell_type[down] == SOLID_CELL and cell_type[up] == FLUID_CELL) or (
                cell_type[down] == FLUID_CELL and cell_type[up] == SOLID_CELL
            ):
                fluid_velocity_y[I] = solid_velocity_y[I]
                fluid_acceleration_y[I] = 0.0


@ti.kernel
def kernel_extrapolate_double_layer_ghost_faces(
    cell_type: ti.template(),
    velocity: ti.template(),
    acceleration: ti.template(),
    solid_velocity: ti.template(),
    fluid_mass: ti.template(),
    cutoff: float,
    depth: int,
    dt: ti.template(),
):
    dimension = ti.static(len(cell_type.shape))
    transverse_depth = depth
    if ti.static(dimension == 2):
        transverse_depth = 0
    for axis in ti.static(range(dimension)):
        for I in ti.grouped(velocity[axis]):
            left = I - ti.Vector.unit(dimension, axis)
            right = I
            left_type = SOLID_CELL
            right_type = SOLID_CELL
            if I[axis] > 0:
                left_type = cell_type[left]
            if I[axis] < cell_type.shape[axis]:
                right_type = cell_type[right]
            air_ghost = left_type == AIR_CELL and right_type == AIR_CELL
            solid_ghost = left_type != FLUID_CELL and right_type != FLUID_CELL and not air_ghost
            if (air_ghost or solid_ghost) and fluid_mass[axis][I] > cutoff:
                air_support = left_type == AIR_CELL or right_type == AIR_CELL
                best_distance = 1 << 30
                value = 0.0
                delta = 0.0
                count = 0
                normal_depth = 0
                if air_ghost:
                    normal_depth = depth
                # Only read active faces, so ghost writes cannot feed each other.
                for a, b, c in ti.ndrange(
                    (-depth, depth + 1), (-transverse_depth, transverse_depth + 1), (-normal_depth, normal_depth + 1)
                ):
                    candidate = I
                    candidate[axis] += c
                    if ti.static(dimension == 2):
                        candidate[1 - axis] += a
                    else:
                        candidate[(axis + 1) % 3] += a
                        candidate[(axis + 2) % 3] += b
                    inside = True
                    for d in ti.static(range(dimension)):
                        inside = inside and 0 <= candidate[d] < velocity[axis].shape[d]
                    distance = a * a + b * b + c * c
                    if inside and distance > 0:
                        source_left = candidate - ti.Vector.unit(dimension, axis)
                        active = False
                        if candidate[axis] > 0:
                            active = active or cell_type[source_left] == FLUID_CELL
                            air_support = air_support or cell_type[source_left] == AIR_CELL
                        if candidate[axis] < cell_type.shape[axis]:
                            active = active or cell_type[candidate] == FLUID_CELL
                            air_support = air_support or cell_type[candidate] == AIR_CELL
                        if active and distance <= best_distance:
                            if distance < best_distance:
                                best_distance = distance
                                value = 0.0
                                delta = 0.0
                                count = 0
                            value += velocity[axis][candidate]
                            delta += acceleration[axis][candidate]
                            count += 1
                normal_wall = False
                if solid_ghost:
                    # Reflect normal velocity about the wall; zero ghosts truncate APIC gradients.
                    for side in ti.static((-1, 1)):
                        for layer in range(1, depth + 1):
                            wall = I + side * layer * ti.Vector.unit(dimension, axis)
                            reflected = I + 2 * side * layer * ti.Vector.unit(dimension, axis)
                            if (
                                0 < wall[axis] < cell_type.shape[axis]
                                and 0 <= reflected[axis] < velocity[axis].shape[axis]
                            ):
                                wall_left = wall - ti.Vector.unit(dimension, axis)
                                is_wall = (cell_type[wall_left] == FLUID_CELL and cell_type[wall] == SOLID_CELL) or (
                                    cell_type[wall_left] == SOLID_CELL and cell_type[wall] == FLUID_CELL
                                )
                                normal_wall = normal_wall or is_wall
                                reflected_active = False
                                if reflected[axis] > 0:
                                    reflected_active = (
                                        cell_type[reflected - ti.Vector.unit(dimension, axis)] == FLUID_CELL
                                    )
                                if reflected[axis] < cell_type.shape[axis]:
                                    reflected_active = reflected_active or cell_type[reflected] == FLUID_CELL
                                if is_wall and reflected_active and layer * layer <= best_distance:
                                    best_distance = layer * layer
                                    value = 2.0 * velocity[axis][wall] - velocity[axis][reflected]
                                    delta = 2.0 * acceleration[axis][wall] - acceleration[axis][reflected]
                                    count = 1
                if count > 0 and not normal_wall:
                    # Extend the increment so pressure-free APIC affine motion survives extrapolation.
                    velocity[axis][I] += dt[None] * (delta / count - acceleration[axis][I])
                    acceleration[axis][I] = delta / count
                elif normal_wall or not air_support:
                    velocity[axis][I] = solid_velocity[axis][I]
                    acceleration[axis][I] = 0.0
                    if count > 0:
                        velocity[axis][I] = value / count
                        acceleration[axis][I] = delta / count


@ti.kernel
def kernel_p2g_double_layer_mass2d(
    total_nodes: int,
    particleNum: int,
    grid_size: vec2f,
    gnum: ti.types.vector(2, int),
    use_affine: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            phase = int(particle[np].phase)
            position = particle[np].x
            if phase == PHASE_SOLID:
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    shape = shapefn[ln]
                    nmass_s = shape * particle[np].ms
                    nodal_velocity_s = particle[np].vs
                    if ti.static(use_affine != 0):
                        nodal_coord = grid_size * vec2f(vectorize_id(nodeID, gnum))
                        nodal_velocity_s += particle[np].solid_velocity_gradient @ (nodal_coord - position)
                    node[nodeID, bodyID].m += nmass_s
                    node[nodeID, bodyID].ms += nmass_s
                    node[nodeID, bodyID].momentum += nmass_s * nodal_velocity_s
                    node[nodeID, bodyID].momentums += nmass_s * nodal_velocity_s
                    shape_volume = shape * particle[np].vol
                    node[nodeID, bodyID].porosity += shape_volume * (1.0 - particle[np].porosity)
            elif phase == PHASE_FLUID:
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    shape = shapefn[ln]
                    nmass_f = shape * particle[np].mf
                    nodal_velocity_f = particle[np].vf
                    if ti.static(use_affine != 0):
                        nodal_coord = grid_size * vec2f(vectorize_id(nodeID, gnum))
                        nodal_velocity_f += particle[np].fluid_velocity_gradient @ (nodal_coord - position)
                    node[nodeID, bodyID].mf += nmass_f
                    node[nodeID, bodyID].momentumf += nmass_f * nodal_velocity_f
            else:
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    shape = shapefn[ln]
                    nmass = shape * particle[np].m
                    nmass_s = shape * particle[np].ms
                    nmass_f = shape * particle[np].mf
                    nodal_velocity = particle[np].v
                    nodal_velocity_s = particle[np].vs
                    nodal_velocity_f = particle[np].vf
                    if ti.static(use_affine != 0):
                        nodal_coord = grid_size * vec2f(vectorize_id(nodeID, gnum))
                        pointer = nodal_coord - position
                        nodal_velocity += particle[np].solid_velocity_gradient @ pointer
                        nodal_velocity_s += particle[np].solid_velocity_gradient @ pointer
                        nodal_velocity_f += particle[np].fluid_velocity_gradient @ pointer
                    node[nodeID, bodyID].m += nmass
                    node[nodeID, bodyID].ms += nmass_s
                    node[nodeID, bodyID].mf += nmass_f
                    node[nodeID, bodyID].momentum += nmass * nodal_velocity
                    node[nodeID, bodyID].momentums += nmass_s * nodal_velocity_s
                    node[nodeID, bodyID].momentumf += nmass_f * nodal_velocity_f


@ti.kernel
def kernel_normalize_double_layer_nodes2d(cutoff: float, node: ti.template(), reference_volume: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            node[ng, nb].momentum /= node[ng, nb].m
        # Retain velocity support for gradients; cutoff only gates the force/mass update.
        if node[ng, nb].ms > 0.0:
            node[ng, nb].momentums /= node[ng, nb].ms
        if node[ng, nb].mf > cutoff:
            node[ng, nb].momentumf /= node[ng, nb].mf
        solid_fraction = 0.0
        if reference_volume[ng, nb] > 0.0:
            solid_fraction = node[ng, nb].porosity / reference_volume[ng, nb]
        node[ng, nb].porosity = _mapped_porosity(solid_fraction)


@ti.kernel
def kernel_update_double_layer_solid_state2d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_SOLID
        ):
            if particle[np].fix_v[0] == 0 or particle[np].fix_v[1] == 0:
                bodyID = int(particle[np].bodyID)
                offset = np * total_nodes
                gradv = ZEROMAT2x2
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    gradv += node[nodeID, bodyID].momentums.outer_product(dshapefn[ln])
                particle[np].solid_velocity_gradient = gradv
                volume_ratio = matProps.update_particle_volume_2D(np, gradv, stateVars, dt)
                porosity = matProps.update_particle_porosity(gradv, particle[np].porosity, dt)
                particle[np].vol *= volume_ratio
                particle[np].porosity = porosity
                if porosity > matProps.maximum_porosity:
                    particle[np].stress *= 0.0
                else:
                    particle[np].stress = matProps.ComputeStress2D(np, particle[np].stress, gradv, stateVars, dt)


@ti.kernel
def kernel_force_p2g_double_layer2d(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_SOLID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            internal = -particle[np].vol * particle[np].stress
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape = dshapefn[ln]
                pforce = vec2f(
                    dshape[0] * internal[0] + dshape[1] * internal[3], dshape[1] * internal[1] + dshape[0] * internal[3]
                )
                node[nodeID, bodyID].force += pforce
                node[nodeID, bodyID].forces += pforce


@ti.kernel
def kernel_mac_p2g_double_layer2d(
    particleNum: int,
    grid_size: vec2f,
    mac_shape_type: ti.template(),
    mac_influenced_node: ti.template(),
    use_affine: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    solid_mass_x: ti.template(),
    solid_mass_y: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_solid_mass: ti.template(),
    cell_porosity: ti.template(),
    cell_solid_velocity: ti.template(),
    cell_fluid_velocity: ti.template(),
    particle: ti.template(),
    calLength: ti.template(),
):
    cell_measure = grid_size[0] * grid_size[1]
    cell_shape_type, cell_support = MAC_SHAPE_LINEAR, 2
    if mac_shape_type == MAC_SHAPE_QUAD_BSPLINE:
        cell_shape_type, cell_support = MAC_SHAPE_QUAD_BSPLINE, 3
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            phase = int(particle[np].phase)
            position = particle[np].x
            psize = calLength[int(particle[np].bodyID)]
            # A linear volume map can create false air ports in a smooth Q2 liquid.
            base_cell = vec2i(
                _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], cell_shape_type),
                _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], cell_shape_type),
            )
            for i, j in ti.ndrange(cell_support, cell_support):
                cell = base_cell + vec2i(i, j)
                if 0 <= cell[0] < cell_fluid_mass.shape[0] and 0 <= cell[1] < cell_fluid_mass.shape[1]:
                    cell_pos = (cell.cast(float) + 0.5) * grid_size
                    weight = _mac_weight2d(position, cell_pos, grid_size, psize, vec2i(0, 0), cell_shape_type)
                    if weight > Threshold:
                        if phase == PHASE_SOLID:
                            mass = weight * particle[np].ms
                            cell_velocity_s = particle[np].vs
                            if ti.static(use_affine != 0):
                                cell_velocity_s += particle[np].solid_velocity_gradient @ (cell_pos - position)
                            cell_solid_mass[cell] += mass
                            cell_porosity[cell] += (
                                weight * particle[np].vol * (1.0 - particle[np].porosity) / cell_measure
                            )
                            cell_solid_velocity[cell] += mass * cell_velocity_s
                        elif phase == PHASE_FLUID:
                            mass = weight * particle[np].mf
                            cell_velocity_f = particle[np].vf
                            if ti.static(use_affine != 0):
                                cell_velocity_f += particle[np].fluid_velocity_gradient @ (cell_pos - position)
                            cell_fluid_mass[cell] += mass
                            cell_fluid_velocity[cell] += mass * cell_velocity_f

            base_x = vec2i(
                _mac_base_1d(position[0], 1.0 / grid_size[0], 0.0, psize[0], mac_shape_type),
                _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], mac_shape_type),
            )
            for i, j in ti.ndrange(mac_influenced_node, mac_influenced_node):
                face = base_x + vec2i(i, j)
                if 0 <= face[0] < fluid_mass_x.shape[0] and 0 <= face[1] < fluid_mass_x.shape[1]:
                    face_pos = vec2f(face[0] * grid_size[0], (face[1] + 0.5) * grid_size[1])
                    btype = vec2i(_mac_bspline_boundary_type(face[0], fluid_mass_x.shape[0]), 0)
                    weight = _mac_weight2d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                    if phase == PHASE_FLUID:
                        mass = weight * particle[np].mf
                        face_velocity_f = particle[np].vf
                        if ti.static(use_affine != 0):
                            face_velocity_f += particle[np].fluid_velocity_gradient @ (face_pos - position)
                        fluid_mass_x[face] += mass
                        fluid_velocity_x[face] += mass * face_velocity_f[0]
                    elif phase == PHASE_SOLID:
                        mass = weight * particle[np].ms
                        face_velocity_s = particle[np].vs
                        if ti.static(use_affine != 0):
                            face_velocity_s += particle[np].solid_velocity_gradient @ (face_pos - position)
                        solid_mass_x[face] += mass
                        solid_velocity_x[face] += mass * face_velocity_s[0]
                        face_porosity_x[face] += (
                            weight
                            * particle[np].vol
                            * (1.0 - particle[np].porosity)
                            / cell_measure
                            * _boundary_face_control_volume_scale(face[0], fluid_mass_x.shape[0])
                        )

            base_y = vec2i(
                _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], mac_shape_type),
                _mac_base_1d(position[1], 1.0 / grid_size[1], 0.0, psize[1], mac_shape_type),
            )
            for i, j in ti.ndrange(mac_influenced_node, mac_influenced_node):
                face = base_y + vec2i(i, j)
                if 0 <= face[0] < fluid_mass_y.shape[0] and 0 <= face[1] < fluid_mass_y.shape[1]:
                    face_pos = vec2f((face[0] + 0.5) * grid_size[0], face[1] * grid_size[1])
                    btype = vec2i(0, _mac_bspline_boundary_type(face[1], fluid_mass_y.shape[1]))
                    weight = _mac_weight2d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                    if phase == PHASE_FLUID:
                        mass = weight * particle[np].mf
                        face_velocity_f = particle[np].vf
                        if ti.static(use_affine != 0):
                            face_velocity_f += particle[np].fluid_velocity_gradient @ (face_pos - position)
                        fluid_mass_y[face] += mass
                        fluid_velocity_y[face] += mass * face_velocity_f[1]
                    elif phase == PHASE_SOLID:
                        mass = weight * particle[np].ms
                        face_velocity_s = particle[np].vs
                        if ti.static(use_affine != 0):
                            face_velocity_s += particle[np].solid_velocity_gradient @ (face_pos - position)
                        solid_mass_y[face] += mass
                        solid_velocity_y[face] += mass * face_velocity_s[1]
                        face_porosity_y[face] += (
                            weight
                            * particle[np].vol
                            * (1.0 - particle[np].porosity)
                            / cell_measure
                            * _boundary_face_control_volume_scale(face[1], fluid_mass_y.shape[1])
                        )


@ti.kernel
def kernel_normalize_double_layer_mac_fields2d(
    cutoff: float,
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity0_x: ti.template(),
    fluid_velocity0_y: ti.template(),
    solid_mass_x: ti.template(),
    solid_mass_y: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    cell_type: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_solid_mass: ti.template(),
    cell_porosity: ti.template(),
    cell_solid_velocity: ti.template(),
    cell_fluid_velocity: ti.template(),
):
    for I in ti.grouped(fluid_mass_x):
        if fluid_mass_x[I] > cutoff:
            fluid_velocity_x[I] /= fluid_mass_x[I]
        if solid_mass_x[I] > cutoff:
            solid_velocity_x[I] /= solid_mass_x[I]
        else:
            solid_velocity_x[I] = 0.0
        face_porosity_x[I] = _mapped_porosity(face_porosity_x[I])
        fluid_velocity0_x[I] = fluid_velocity_x[I]

    for I in ti.grouped(fluid_mass_y):
        if fluid_mass_y[I] > cutoff:
            fluid_velocity_y[I] /= fluid_mass_y[I]
        if solid_mass_y[I] > cutoff:
            solid_velocity_y[I] /= solid_mass_y[I]
        else:
            solid_velocity_y[I] = 0.0
        face_porosity_y[I] = _mapped_porosity(face_porosity_y[I])
        fluid_velocity0_y[I] = fluid_velocity_y[I]

    for I in ti.grouped(cell_type):
        if cell_fluid_mass[I] > cutoff:
            cell_type[I] = FLUID_CELL
            cell_fluid_velocity[I] /= cell_fluid_mass[I]
        if cell_solid_mass[I] > cutoff:
            cell_solid_velocity[I] /= cell_solid_mass[I]
        else:
            cell_solid_velocity[I] = ZEROVEC2f
        cell_porosity[I] = _mapped_porosity(cell_porosity[I])


@ti.kernel
def kernel_predict_double_layer2d(
    cutoff: float,
    gravity: vec3f,
    grid_size: vec2f,
    dt: ti.template(),
    damping: float,
    node: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    face_solid_density_x: ti.template(),
    face_solid_density_y: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    face_fluid_viscosity_x: ti.template(),
    face_fluid_viscosity_y: ti.template(),
    face_grain_diameter_x: ti.template(),
    face_grain_diameter_y: ti.template(),
    face_permeability_x: ti.template(),
    face_permeability_y: ti.template(),
    face_fluid_unit_weight_x: ti.template(),
    face_fluid_unit_weight_y: ti.template(),
    face_drag_model_x: ti.template(),
    face_drag_model_y: ti.template(),
    node_solid_density: ti.template(),
    node_fluid_density: ti.template(),
    node_fluid_viscosity: ti.template(),
    node_grain_diameter: ti.template(),
    node_permeability: ti.template(),
    node_fluid_unit_weight: ti.template(),
    node_drag_model: ti.template(),
    shape_type: ti.template(),
    influenced_node: ti.template(),
    calLength: ti.template(),
    solid_pressure_factor: ti.template(),
    fluid_pressure_mobility: ti.template(),
    mixture_pressure_mobility: ti.template(),
    solid_face_mass: ti.template(),
):
    gnum = vec2i(fluid_velocity_x.shape[0], fluid_velocity_y.shape[1])
    # Complete every viscous stencil before applying either phase update.
    for I in ti.grouped(fluid_velocity_x):
        if I[0] == 0 or I[0] == fluid_velocity_x.shape[0] - 1:
            fluid_velocity_x[I] = 0.0
    for I in ti.grouped(fluid_velocity_y):
        if I[1] == 0 or I[1] == fluid_velocity_y.shape[1] - 1:
            fluid_velocity_y[I] = 0.0
    for I in ti.grouped(fluid_velocity_x):
        acc = 0.0
        if fluid_mass_x[I] > cutoff:
            nu = face_fluid_viscosity_x[I] / ti.max(face_fluid_density_x[I], 1.0e-12)
            acc = nu * _mac_laplacian2d(I, grid_size, fluid_velocity_x) - damping * fluid_velocity_x[I]
        fluid_acceleration_x[I] = acc
    for I in ti.grouped(fluid_velocity_y):
        acc = 0.0
        if fluid_mass_y[I] > cutoff:
            nu = face_fluid_viscosity_y[I] / ti.max(face_fluid_density_y[I], 1.0e-12)
            acc = nu * _mac_laplacian2d(I, grid_size, fluid_velocity_y) - damping * fluid_velocity_y[I]
        fluid_acceleration_y[I] = acc
    for ng, nb in node:
        if ti.static(solid_pressure_factor != None):
            solid_pressure_factor[ng, nb] = 1.0
        if node[ng, nb].ms > cutoff:
            external_s = node[ng, nb].force / node[ng, nb].ms + vec2f(gravity[0], gravity[1])
            acc = external_s
            if node[ng, nb].mf > cutoff:
                position = vec2i(vectorize_id(ng, gnum)).cast(float) * grid_size
                external_f = _sample_mac_velocity(
                    position,
                    calLength[nb],
                    shape_type,
                    influenced_node,
                    grid_size,
                    fluid_acceleration_x,
                    fluid_acceleration_y,
                ) + vec2f(gravity[0], gravity[1])
                rel = node[ng, nb].momentumf - node[ng, nb].momentums
                a, b = _drag_rates_props(
                    rel,
                    _intrinsic_porosity_from_phase_masses(
                        node[ng, nb].ms, node[ng, nb].mf, node_solid_density[ng, nb], node_fluid_density[ng, nb]
                    ),
                    node_solid_density[ng, nb],
                    node_fluid_density[ng, nb],
                    node_fluid_viscosity[ng, nb],
                    node_grain_diameter[ng, nb],
                    node_permeability[ng, nb],
                    node_fluid_unit_weight[ng, nb],
                    node_drag_model[ng, nb],
                )
                acc, _ = _implicit_phase_accelerations(rel, external_s, external_f, a, b, dt[None])
                if ti.static(solid_pressure_factor != None):
                    mu_s, _ = _coupled_pressure_mobilities(
                        node_solid_density[ng, nb], node_fluid_density[ng, nb], a, b, dt[None]
                    )
                    solid_pressure_factor[ng, nb] = node_solid_density[ng, nb] * mu_s
            node[ng, nb].forces = acc
            node[ng, nb].momentums += dt[None] * acc
            node[ng, nb].momentum = node[ng, nb].momentums
        else:
            # G2P reads acceleration here, never the unscaled deposited force.
            node[ng, nb].forces = ZEROVEC2f
    for I in ti.grouped(fluid_velocity_x):
        acc = 0.0
        mu_s = 1.0 / ti.max(face_solid_density_x[I], 1.0e-12)
        mu_f = 1.0 / ti.max(face_fluid_density_x[I], 1.0e-12)
        if fluid_mass_x[I] > cutoff:
            position = vec2f(I[0] * grid_size[0], (I[1] + 0.5) * grid_size[1])
            rel = _mac_relative_velocity_at_face(
                0, I, (fluid_velocity_x, fluid_velocity_y), (solid_velocity_x, solid_velocity_y)
            )
            external_s, weight = _sample_solid_grid_velocity2d(
                position, grid_size, gnum, cutoff, shape_type, influenced_node, calLength, node, True
            )
            external_f = vec2f(fluid_acceleration_x[I], 0.0)
            a, b = _drag_rates_props(
                rel,
                _intrinsic_porosity_from_phase_masses(
                    solid_face_mass[0][I], fluid_mass_x[I], face_solid_density_x[I], face_fluid_density_x[I]
                ),
                face_solid_density_x[I],
                face_fluid_density_x[I],
                face_fluid_viscosity_x[I],
                face_grain_diameter_x[I],
                face_permeability_x[I],
                face_fluid_unit_weight_x[I],
                face_drag_model_x[I],
            )
            _, coupled = _implicit_phase_accelerations(rel, external_s, external_f, a, b, dt[None])
            acc = gravity[0] + coupled[0]
            mu_s, mu_f = _coupled_pressure_mobilities(face_solid_density_x[I], face_fluid_density_x[I], a, b, dt[None])
        if ti.static(fluid_pressure_mobility != None):
            fluid_pressure_mobility[0][I] = mu_f
        if ti.static(mixture_pressure_mobility != None):
            phi = _clamp_porosity(face_porosity_x[I])
            mixture_pressure_mobility[0][I] = (1.0 - phi) * mu_s + phi * mu_f
        fluid_acceleration_x[I] = acc

    for I in ti.grouped(fluid_velocity_y):
        acc = 0.0
        mu_s = 1.0 / ti.max(face_solid_density_y[I], 1.0e-12)
        mu_f = 1.0 / ti.max(face_fluid_density_y[I], 1.0e-12)
        if fluid_mass_y[I] > cutoff:
            position = vec2f((I[0] + 0.5) * grid_size[0], I[1] * grid_size[1])
            rel = _mac_relative_velocity_at_face(
                1, I, (fluid_velocity_x, fluid_velocity_y), (solid_velocity_x, solid_velocity_y)
            )
            external_s, weight = _sample_solid_grid_velocity2d(
                position, grid_size, gnum, cutoff, shape_type, influenced_node, calLength, node, True
            )
            external_f = vec2f(0.0, fluid_acceleration_y[I])
            a, b = _drag_rates_props(
                rel,
                _intrinsic_porosity_from_phase_masses(
                    solid_face_mass[1][I], fluid_mass_y[I], face_solid_density_y[I], face_fluid_density_y[I]
                ),
                face_solid_density_y[I],
                face_fluid_density_y[I],
                face_fluid_viscosity_y[I],
                face_grain_diameter_y[I],
                face_permeability_y[I],
                face_fluid_unit_weight_y[I],
                face_drag_model_y[I],
            )
            _, coupled = _implicit_phase_accelerations(rel, external_s, external_f, a, b, dt[None])
            acc = gravity[1] + coupled[1]
            mu_s, mu_f = _coupled_pressure_mobilities(face_solid_density_y[I], face_fluid_density_y[I], a, b, dt[None])
        if ti.static(fluid_pressure_mobility != None):
            fluid_pressure_mobility[1][I] = mu_f
        if ti.static(mixture_pressure_mobility != None):
            phi = _clamp_porosity(face_porosity_y[I])
            mixture_pressure_mobility[1][I] = (1.0 - phi) * mu_s + phi * mu_f
        fluid_acceleration_y[I] = acc

    # Nonlinear drag samples every velocity component at the same time level.
    for I in ti.grouped(fluid_velocity_x):
        fluid_velocity_x[I] += dt[None] * fluid_acceleration_x[I]
        if I[0] == 0 or I[0] == fluid_velocity_x.shape[0] - 1:
            fluid_velocity_x[I] = 0.0
            fluid_acceleration_x[I] = 0.0
    for I in ti.grouped(fluid_velocity_y):
        fluid_velocity_y[I] += dt[None] * fluid_acceleration_y[I]
        if I[1] == 0 or I[1] == fluid_velocity_y.shape[1] - 1:
            fluid_velocity_y[I] = 0.0
            fluid_acceleration_y[I] = 0.0


@ti.kernel
def kernel_project_solid_grid_velocity_to_mac2d(
    cutoff: float,
    grid_size: vec2f,
    gnum: ti.types.vector(2, int),
    shape_type: ti.template(),
    influenced_node: ti.template(),
    node: ti.template(),
    calLength: ti.template(),
    cell_type: ti.template(),
    solid_mass_x: ti.template(),
    solid_mass_y: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
):
    # Domain edges and SolidCell faces share the fluid no-penetration condition.
    # Moving-wall callbacks run after this projection and may overwrite it.
    for I in ti.grouped(solid_velocity_x):
        static_wall = I[0] == 0 or I[0] == solid_velocity_x.shape[0] - 1
        if 0 < I[0] < solid_velocity_x.shape[0] - 1:
            left = I - vec2i(1, 0)
            right = I
            static_wall = (cell_type[left] == SOLID_CELL and cell_type[right] == FLUID_CELL) or (
                cell_type[left] == FLUID_CELL and cell_type[right] == SOLID_CELL
            )
        if static_wall:
            solid_velocity_x[I] = 0.0
        elif solid_mass_x[I] > cutoff:
            face_pos = vec2f(I[0] * grid_size[0], (I[1] + 0.5) * grid_size[1])
            velocity, weight_sum = _sample_solid_grid_velocity2d(
                face_pos,
                grid_size,
                gnum,
                cutoff,
                shape_type,
                influenced_node,
                calLength,
                node,
            )
            if weight_sum > cutoff:
                solid_velocity_x[I] = velocity[0]

    for I in ti.grouped(solid_velocity_y):
        static_wall = I[1] == 0 or I[1] == solid_velocity_y.shape[1] - 1
        if 0 < I[1] < solid_velocity_y.shape[1] - 1:
            down = I - vec2i(0, 1)
            up = I
            static_wall = (cell_type[down] == SOLID_CELL and cell_type[up] == FLUID_CELL) or (
                cell_type[down] == FLUID_CELL and cell_type[up] == SOLID_CELL
            )
        if static_wall:
            solid_velocity_y[I] = 0.0
        elif solid_mass_y[I] > cutoff:
            face_pos = vec2f((I[0] + 0.5) * grid_size[0], I[1] * grid_size[1])
            velocity, weight_sum = _sample_solid_grid_velocity2d(
                face_pos,
                grid_size,
                gnum,
                cutoff,
                shape_type,
                influenced_node,
                calLength,
                node,
            )
            if weight_sum > cutoff:
                solid_velocity_y[I] = velocity[1]


@ti.kernel
def kernel_assemble_double_layer_pressure_rhs(
    grid_size: vec2f,
    cell_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    b: ti.template(),
):
    for I in ti.grouped(cell_type):
        b[I] = 0.0
        if cell_type[I] == FLUID_CELL:
            # A shared face flux preserves the closed-domain compatibility condition.
            divergence = 0.0
            right = I + vec2i(1, 0)
            flux_right = (
                face_porosity_x[right] * fluid_velocity_x[right]
                + (1.0 - face_porosity_x[right]) * solid_velocity_x[right]
            )
            flux_left = face_porosity_x[I] * fluid_velocity_x[I] + (1.0 - face_porosity_x[I]) * solid_velocity_x[I]
            divergence += (flux_right - flux_left) / grid_size[0]
            right = I + vec2i(0, 1)
            flux_right = (
                face_porosity_y[right] * fluid_velocity_y[right]
                + (1.0 - face_porosity_y[right]) * solid_velocity_y[right]
            )
            flux_left = face_porosity_y[I] * fluid_velocity_y[I] + (1.0 - face_porosity_y[I]) * solid_velocity_y[I]
            divergence += (flux_right - flux_left) / grid_size[1]
            b[I] = -divergence


@ti.kernel
def kernel_assemble_double_layer_pressure_A(
    grid_size: vec2f,
    dt: ti.template(),
    grid_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    fluid_sdf: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
    coupled_mobility: ti.template(),
):
    for I in ti.grouped(grid_type):
        Adiag[I] = 0.0
        Ax[I] = ZEROVEC2f
        if grid_type[I] == FLUID_CELL:
            # Keep a symmetric face-averaged mobility so PCG/MGPCG remains applicable.
            mobility = _cell_mobility2d(
                I, face_porosity_x, face_porosity_y, cell_solid_density, cell_fluid_density, coupled_mobility
            )
            scale_x = dt[None] * mobility / (grid_size[0] * grid_size[0])
            scale_y = dt[None] * mobility / (grid_size[1] * grid_size[1])
            right = I + vec2i(1, 0)
            up = I + vec2i(0, 1)
            if right[0] < grid_type.shape[0]:
                if grid_type[right] == FLUID_CELL:
                    mobility_right = _cell_mobility2d(
                        right,
                        face_porosity_x,
                        face_porosity_y,
                        cell_solid_density,
                        cell_fluid_density,
                        coupled_mobility,
                    )
                    face_scale = 0.5 * (scale_x + dt[None] * mobility_right / (grid_size[0] * grid_size[0]))
                    Ax[I][0] = -face_scale
                    Adiag[I] += face_scale
                elif grid_type[right] == AIR_CELL:
                    theta = _pressure_free_surface_theta(I, right, fluid_sdf)
                    Adiag[I] += scale_x / theta
            if up[1] < grid_type.shape[1]:
                if grid_type[up] == FLUID_CELL:
                    mobility_up = _cell_mobility2d(
                        up, face_porosity_x, face_porosity_y, cell_solid_density, cell_fluid_density, coupled_mobility
                    )
                    face_scale = 0.5 * (scale_y + dt[None] * mobility_up / (grid_size[1] * grid_size[1]))
                    Ax[I][1] = -face_scale
                    Adiag[I] += face_scale
                elif grid_type[up] == AIR_CELL:
                    theta = _pressure_free_surface_theta(I, up, fluid_sdf)
                    Adiag[I] += scale_y / theta
            left = I - vec2i(1, 0)
            down = I - vec2i(0, 1)
            if left[0] >= 0:
                if grid_type[left] == FLUID_CELL:
                    mobility_left = _cell_mobility2d(
                        left, face_porosity_x, face_porosity_y, cell_solid_density, cell_fluid_density, coupled_mobility
                    )
                    Adiag[I] += 0.5 * (scale_x + dt[None] * mobility_left / (grid_size[0] * grid_size[0]))
                elif grid_type[left] == AIR_CELL:
                    theta = _pressure_free_surface_theta(I, left, fluid_sdf)
                    Adiag[I] += scale_x / theta
            if down[1] >= 0:
                if grid_type[down] == FLUID_CELL:
                    mobility_down = _cell_mobility2d(
                        down, face_porosity_x, face_porosity_y, cell_solid_density, cell_fluid_density, coupled_mobility
                    )
                    Adiag[I] += 0.5 * (scale_y + dt[None] * mobility_down / (grid_size[1] * grid_size[1]))
                elif grid_type[down] == AIR_CELL:
                    theta = _pressure_free_surface_theta(I, down, fluid_sdf)
                    Adiag[I] += scale_y / theta
            if Adiag[I] <= 1.0e-12:
                Adiag[I] = 1.0
            else:
                Adiag[I] += 1.0e-12


@ti.kernel
def kernel_coarsen_double_layer_grid_type2d(fine_grid_type: ti.template(), coarse_grid_type: ti.template()):
    for I in ti.grouped(coarse_grid_type):
        base = I * 2
        has_fluid = 0
        has_air = 0
        for offset in ti.static(ti.grouped(ti.ndrange(2, 2))):
            fine = base + offset
            if fine[0] < fine_grid_type.shape[0] and fine[1] < fine_grid_type.shape[1]:
                attr = int(fine_grid_type[fine])
                if attr == FLUID_CELL:
                    has_fluid = 1
                elif attr == AIR_CELL:
                    has_air = 1
        if has_fluid:
            coarse_grid_type[I] = FLUID_CELL
        elif has_air:
            coarse_grid_type[I] = AIR_CELL
        else:
            coarse_grid_type[I] = SOLID_CELL


@ti.kernel
def kernel_assemble_double_layer_pressure_mg_A(
    grid_size: vec2f,
    dt: ti.template(),
    factor: int,
    grid_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    fluid_sdf: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
    coupled_mobility: ti.template(),
):
    coarse_grid_size = grid_size * factor
    for I in ti.grouped(grid_type):
        Adiag[I] = 0.0
        Ax[I] = ZEROVEC2f
        if grid_type[I] == FLUID_CELL:
            mobility = _coarse_mobility2d(
                I, factor, face_porosity_x, face_porosity_y, cell_solid_density, cell_fluid_density, coupled_mobility
            )
            scale_x = dt[None] * mobility / (coarse_grid_size[0] * coarse_grid_size[0])
            scale_y = dt[None] * mobility / (coarse_grid_size[1] * coarse_grid_size[1])
            right = I + vec2i(1, 0)
            up = I + vec2i(0, 1)
            if right[0] < grid_type.shape[0]:
                if grid_type[right] == FLUID_CELL:
                    mobility_right = _coarse_mobility2d(
                        right,
                        factor,
                        face_porosity_x,
                        face_porosity_y,
                        cell_solid_density,
                        cell_fluid_density,
                        coupled_mobility,
                    )
                    face_scale = 0.5 * (
                        scale_x + dt[None] * mobility_right / (coarse_grid_size[0] * coarse_grid_size[0])
                    )
                    Ax[I][0] = -face_scale
                    Adiag[I] += face_scale
                elif grid_type[right] == AIR_CELL:
                    theta = _pressure_free_surface_theta_coarse2d(I, right, factor, fluid_sdf)
                    Adiag[I] += scale_x / theta
            if up[1] < grid_type.shape[1]:
                if grid_type[up] == FLUID_CELL:
                    mobility_up = _coarse_mobility2d(
                        up,
                        factor,
                        face_porosity_x,
                        face_porosity_y,
                        cell_solid_density,
                        cell_fluid_density,
                        coupled_mobility,
                    )
                    face_scale = 0.5 * (scale_y + dt[None] * mobility_up / (coarse_grid_size[1] * coarse_grid_size[1]))
                    Ax[I][1] = -face_scale
                    Adiag[I] += face_scale
                elif grid_type[up] == AIR_CELL:
                    theta = _pressure_free_surface_theta_coarse2d(I, up, factor, fluid_sdf)
                    Adiag[I] += scale_y / theta
            left = I - vec2i(1, 0)
            down = I - vec2i(0, 1)
            if left[0] >= 0:
                if grid_type[left] == FLUID_CELL:
                    mobility_left = _coarse_mobility2d(
                        left,
                        factor,
                        face_porosity_x,
                        face_porosity_y,
                        cell_solid_density,
                        cell_fluid_density,
                        coupled_mobility,
                    )
                    Adiag[I] += 0.5 * (scale_x + dt[None] * mobility_left / (coarse_grid_size[0] * coarse_grid_size[0]))
                elif grid_type[left] == AIR_CELL:
                    theta = _pressure_free_surface_theta_coarse2d(I, left, factor, fluid_sdf)
                    Adiag[I] += scale_x / theta
            if down[1] >= 0:
                if grid_type[down] == FLUID_CELL:
                    mobility_down = _coarse_mobility2d(
                        down,
                        factor,
                        face_porosity_x,
                        face_porosity_y,
                        cell_solid_density,
                        cell_fluid_density,
                        coupled_mobility,
                    )
                    Adiag[I] += 0.5 * (scale_y + dt[None] * mobility_down / (coarse_grid_size[1] * coarse_grid_size[1]))
                elif grid_type[down] == AIR_CELL:
                    theta = _pressure_free_surface_theta_coarse2d(I, down, factor, fluid_sdf)
                    Adiag[I] += scale_y / theta
            if Adiag[I] <= 1.0e-12:
                Adiag[I] = 1.0
            else:
                Adiag[I] += 1.0e-12


@ti.kernel
def kernel_correct_double_layer_velocity2d(
    cutoff: float,
    grid_size: vec2f,
    dt: ti.template(),
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    pressure_mobility: ti.template(),
):
    for I in ti.grouped(fluid_velocity_x):
        if fluid_mass_x[I] > cutoff:
            if I[0] == 0 or I[0] == fluid_velocity_x.shape[0] - 1:
                fluid_velocity_x[I] = 0.0
                fluid_acceleration_x[I] = 0.0
            else:
                left = I - vec2i(1, 0)
                right = I
                gradp = 0.0
                if cell_type[left] == FLUID_CELL and cell_type[right] == FLUID_CELL:
                    gradp = (cell_pressure[right] - cell_pressure[left]) / grid_size[0]
                elif cell_type[left] == FLUID_CELL and cell_type[right] == AIR_CELL:
                    theta = _pressure_free_surface_theta(left, right, fluid_sdf)
                    gradp = (0.0 - cell_pressure[left]) / (theta * grid_size[0])
                elif cell_type[left] == AIR_CELL and cell_type[right] == FLUID_CELL:
                    theta = _pressure_free_surface_theta(right, left, fluid_sdf)
                    gradp = (cell_pressure[right] - 0.0) / (theta * grid_size[0])
                acc = -gradp / ti.max(face_fluid_density_x[I], 1.0e-12)
                if ti.static(pressure_mobility != None):
                    acc = -gradp * pressure_mobility[0][I]
                fluid_velocity_x[I] += dt[None] * acc
                fluid_acceleration_x[I] += acc

    for I in ti.grouped(fluid_velocity_y):
        if fluid_mass_y[I] > cutoff:
            if I[1] == 0 or I[1] == fluid_velocity_y.shape[1] - 1:
                fluid_velocity_y[I] = 0.0
                fluid_acceleration_y[I] = 0.0
            else:
                down = I - vec2i(0, 1)
                up = I
                gradp = 0.0
                if cell_type[down] == FLUID_CELL and cell_type[up] == FLUID_CELL:
                    gradp = (cell_pressure[up] - cell_pressure[down]) / grid_size[1]
                elif cell_type[down] == FLUID_CELL and cell_type[up] == AIR_CELL:
                    theta = _pressure_free_surface_theta(down, up, fluid_sdf)
                    gradp = (0.0 - cell_pressure[down]) / (theta * grid_size[1])
                elif cell_type[down] == AIR_CELL and cell_type[up] == FLUID_CELL:
                    theta = _pressure_free_surface_theta(up, down, fluid_sdf)
                    gradp = (cell_pressure[up] - 0.0) / (theta * grid_size[1])
                acc = -gradp / ti.max(face_fluid_density_y[I], 1.0e-12)
                if ti.static(pressure_mobility != None):
                    acc = -gradp * pressure_mobility[1][I]
                fluid_velocity_y[I] += dt[None] * acc
                fluid_acceleration_y[I] += acc


@ti.kernel
def kernel_project_double_layer_pressure_to_solid_nodes2d(
    cutoff: float,
    gnum: ti.types.vector(2, int),
    grid_size: vec2f,
    mac_shape_type: ti.template(),
    mac_influenced_node: ti.template(),
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    node: ti.template(),
    calLength: ti.template(),
):
    for ng, nb in node:
        node[ng, nb].pressure = 0.0
        if node[ng, nb].ms > cutoff:
            grid_id = vec2i(vectorize_id(ng, gnum))
            position = grid_id.cast(float) * grid_size
            psize = calLength[nb]
            node[ng, nb].pressure = _sample_cell_pressure_gfm_shape2d(
                position,
                psize,
                mac_shape_type,
                mac_influenced_node,
                grid_size,
                cell_type,
                cell_pressure,
                fluid_sdf,
                False,
            )


@ti.kernel
def kernel_sample_double_layer_solid_pressure_from_nodes2d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    cutoff: float,
    particle: ti.template(),
    material_mapping: ti.template(),
    node: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_SOLID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            pressure = 0.0
            weight = 0.0
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape = shapefn[ln]
                if node[nodeID, bodyID].ms > cutoff:
                    pressure += shape * node[nodeID, bodyID].pressure
                    weight += shape
            if weight > Threshold:
                particle[np].pressure = pressure / weight
            else:
                particle[np].pressure = 0.0


@ti.kernel
def kernel_correct_double_layer_solid_velocity_paper2d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    cutoff: float,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    pressure_factor: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_SOLID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            pressure_gradient = ZEROVEC2f
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                pressure_gradient += node[nodeID, bodyID].pressure * dshapefn[ln]
            pressure_force = -particle[np].vol * (1.0 - particle[np].porosity) * pressure_gradient
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                if node[nodeID, bodyID].ms > cutoff:
                    acc = shapefn[ln] * pressure_force / ti.max(node[nodeID, bodyID].ms, 1.0e-12)
                    if ti.static(pressure_factor != None):
                        acc *= pressure_factor[nodeID, bodyID]
                    node[nodeID, bodyID].momentums += dt[None] * acc
                    node[nodeID, bodyID].momentum = node[nodeID, bodyID].momentums
                    node[nodeID, bodyID].forces += acc


@ti.kernel
def kernel_advect_double_layer_fluid_particles2d(
    particle_count: int, domain: vec2f, dt: ti.template(), particle: ti.template()
):
    for np in range(particle_count):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
        ):
            flag_fix = ti.cast(particle[np].fix_v, float)
            flag_free = vec2f(1.0, 1.0) - flag_fix
            particle[np].x += dt[None] * (particle[np].vf * flag_free + particle[np].v * flag_fix)
            for d in ti.static(range(2)):
                particle[np].x[d] = ti.min(ti.max(particle[np].x[d], 1.0e-8), domain[d] - 1.0e-8)


@ti.kernel
def kernel_update_double_layer_fluid_volume2d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    initialize_mass: int,
    fluid_density: float,
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    phase_volume_support: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            porosity = 0.0
            weight = 0.0
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape = shapefn[ln]
                porosity += shape * _intrinsic_pore_porosity(nodeID, bodyID, node, phase_volume_support)
                weight += shape
            if weight > Threshold:
                porosity = _clamp_porosity(porosity / weight)
                if initialize_mass != 0:
                    particle[np].mf = fluid_density * porosity * particle[np].vol
                    particle[np].m = particle[np].mf
                else:
                    particle[np].vol = particle[np].mf / ti.max(fluid_density * porosity, 1.0e-12)
                particle[np].porosity = porosity
                particle[np].rad = 0.5 * ti.sqrt(particle[np].vol)


@ti.kernel
def kernel_volume_p2g_double_layer_fluid2d(
    total_nodes: int,
    particle_count: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particle_count):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                node[nodeID, bodyID].vol += shapefn[ln] * particle[np].vol * particle[np].porosity


@ti.kernel
def kernel_delta_correct_double_layer_fluid2d(
    total_nodes: int,
    particle_count: int,
    domain: vec2f,
    grid_size: vec2f,
    cnum: ti.types.vector(2, int),
    shifting_scale: float,
    cell_type: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    reference_volume: ti.template(),
    volume_error: ti.template(),
):
    min_dx = ti.min(grid_size[0], grid_size[1])
    gradient_scale = (grid_size[0] * grid_size[1]) ** 2 / min_dx
    for ng, nb in node:
        error = _double_layer_shifting_volume_error(ng, nb, cnum, cell_type, node, reference_volume)
        volume_error[ng, nb] = error

    denominator = 0.0
    for np in range(particle_count):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
            and _double_layer_shifting_is_interior2d(particle[np].x, grid_size, cnum, cell_type)
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            gradient = ZEROVEC2f
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                error = volume_error[nodeID, bodyID]
                gradient += dshapefn[ln] * error
            gradient *= 2.0 * particle[np].vol * particle[np].porosity
            particle[np].grad_E2 = gradient
            denominator += gradient.dot(gradient)

    # Reuse residual storage for J grad(E), then take the quadratic Cauchy step.
    for ng, nb in node:
        volume_error[ng, nb] = 0.0
    for np in range(particle_count):
        if (
            denominator > Threshold * gradient_scale * gradient_scale
            and int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
            and _double_layer_shifting_is_interior2d(particle[np].x, grid_size, cnum, cell_type)
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                volume_error[nodeID, bodyID] += (
                    particle[np].vol * particle[np].porosity * dshapefn[ln].dot(particle[np].grad_E2)
                )
    curvature = 0.0
    for ng, nb in node:
        curvature += volume_error[ng, nb] * volume_error[ng, nb]
    step = 0.0
    if denominator > Threshold * gradient_scale * gradient_scale and curvature > 0.0:
        step = shifting_scale * denominator / (2.0 * curvature)
    for np in range(particle_count):
        if (
            step > 0.0
            and int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
            and _double_layer_shifting_is_interior2d(particle[np].x, grid_size, cnum, cell_type)
        ):
            shift = -step * particle[np].grad_E2
            max_shift = 0.05 * min_dx
            shift_norm = shift.norm()
            if shift_norm > max_shift:
                shift *= max_shift / shift_norm
            for d in ti.static(range(2)):
                particle[np].x[d] += shift[d]
                particle[np].x[d] = ti.min(ti.max(particle[np].x[d], 1.0e-8), domain[d] - 1.0e-8)


@ti.kernel
def kernel_g2p_double_layer2d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    alpha: float,
    domain: vec2f,
    grid_size: vec2f,
    mac_shape_type: ti.template(),
    mac_influenced_node: ti.template(),
    use_affine: ti.template(),
    dt: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    calLength: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    preserve_solid_pressure: ti.template(),
    update_solid_state: ti.template(),
    delayed_fluid_advection: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            phase = int(particle[np].phase)
            position = particle[np].x
            if phase == PHASE_SOLID:
                v_pic = ZEROVEC2f
                a_pic = ZEROVEC2f
                gradv = ZEROMAT2x2
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    shape = shapefn[ln]
                    dshape = dshapefn[ln]
                    velocity = node[nodeID, bodyID].momentums
                    acceleration = node[nodeID, bodyID].forces
                    v_pic += shape * velocity
                    a_pic += shape * acceleration
                    gradv += velocity.outer_product(dshape)
                if ti.static(use_affine != 0 and mac_shape_type != MAC_SHAPE_LINEAR):
                    Dp = ZEROMAT2x2
                    Bp = ZEROMAT2x2
                    for ln in range(offset, offset + int(node_size[np])):
                        nodeID = LnID[ln]
                        shape = shapefn[ln]
                        nodal_coord = grid_size * vec2f(
                            vectorize_id(nodeID, vec2i(fluid_velocity_x.shape[0], fluid_velocity_y.shape[1]))
                        )
                        pointer = nodal_coord - position
                        velocity = node[nodeID, bodyID].momentums
                        Dp += shape * pointer.outer_product(pointer)
                        Bp += shape * (velocity - v_pic).outer_product(pointer)
                    dp_trace = ti.max(Dp[0, 0] + Dp[1, 1], 0.0)
                    dp_det = ti.abs(Dp.determinant())
                    if dp_trace > 1.0e-20 and dp_det > 1.0e-8 * dp_trace * dp_trace:
                        gradv = Bp @ Dp.inverse()
                flag_fix = ti.cast(particle[np].fix_v, float)
                flag_free = vec2f(1.0, 1.0) - flag_fix
                if particle[np].fix_v[0] != 0 and particle[np].fix_v[1] != 0:
                    # A fixed skeleton must not scatter a moving APIC field next step.
                    gradv = ZEROMAT2x2
                v_flip = particle[np].vs + dt[None] * a_pic
                new_v = (alpha * v_pic + (1.0 - alpha) * v_flip) * flag_free + particle[np].vs * flag_fix
                particle[np].vs = new_v
                particle[np].v = new_v
                particle[np].solid_velocity_gradient = gradv
                if ti.static(update_solid_state != 0):
                    volume_ratio = matProps.update_particle_volume_2D(np, gradv, stateVars, dt)
                    porosity = matProps.update_particle_porosity(gradv, particle[np].porosity, dt)
                    particle[np].vol *= volume_ratio
                    particle[np].porosity = porosity
                    if porosity > matProps.maximum_porosity:
                        particle[np].stress *= 0.0
                    else:
                        particle[np].stress = matProps.ComputeStress2D(np, particle[np].stress, gradv, stateVars, dt)
                if ti.static(preserve_solid_pressure == 0):
                    psize = calLength[bodyID]
                    particle[np].pressure = _sample_cell_pressure_gfm_shape2d(
                        position,
                        psize,
                        mac_shape_type,
                        mac_influenced_node,
                        grid_size,
                        cell_type,
                        cell_pressure,
                        fluid_sdf,
                    )
                particle[np].x += dt[None] * (v_pic * flag_free + particle[np].vs * flag_fix)
            elif phase == PHASE_FLUID:
                psize = calLength[bodyID]
                v_pic = _sample_mac_velocity(
                    position, psize, mac_shape_type, mac_influenced_node, grid_size, fluid_velocity_x, fluid_velocity_y
                )
                a_pic = _sample_mac_velocity(
                    position,
                    psize,
                    mac_shape_type,
                    mac_influenced_node,
                    grid_size,
                    fluid_acceleration_x,
                    fluid_acceleration_y,
                )
                if ti.static(use_affine != 0):
                    particle[np].fluid_velocity_gradient = _sample_mac_velocity_gradient(
                        position,
                        psize,
                        mac_shape_type,
                        mac_influenced_node,
                        grid_size,
                        fluid_velocity_x,
                        fluid_velocity_y,
                        v_pic,
                    )
                flag_fix = ti.cast(particle[np].fix_v, float)
                flag_free = vec2f(1.0, 1.0) - flag_fix
                v_flip = particle[np].vf + dt[None] * a_pic
                new_v = (alpha * v_pic + (1.0 - alpha) * v_flip) * flag_free + particle[np].vf * flag_fix
                particle[np].vf = new_v
                particle[np].v = new_v
                particle[np].pressure = _sample_cell_pressure_gfm_shape2d(
                    position,
                    psize,
                    mac_shape_type,
                    mac_influenced_node,
                    grid_size,
                    cell_type,
                    cell_pressure,
                    fluid_sdf,
                )
                if ti.static(delayed_fluid_advection == 0):
                    particle[np].x += dt[None] * (v_pic * flag_free + particle[np].vf * flag_fix)

            for d in ti.static(range(2)):
                particle[np].x[d] = ti.min(ti.max(particle[np].x[d], 1.0e-8), domain[d] - 1.0e-8)


@ti.func
def _sample_cell_scalar3d(position, grid_size, cell_type, cell_value):
    base = ti.floor(position / grid_size - 0.5).cast(int)
    value = 0.0
    weight_sum = 0.0
    for i, j, k in ti.ndrange(2, 2, 2):
        cell = base + vec3i(i, j, k)
        if (
            0 <= cell[0] < cell_type.shape[0]
            and 0 <= cell[1] < cell_type.shape[1]
            and 0 <= cell[2] < cell_type.shape[2]
        ):
            if cell_type[cell] == FLUID_CELL:
                center = (cell.cast(float) + 0.5) * grid_size
                weight = (
                    _linear_weight((position[0] - center[0]) / grid_size[0])
                    * _linear_weight((position[1] - center[1]) / grid_size[1])
                    * _linear_weight((position[2] - center[2]) / grid_size[2])
                )
                value += weight * cell_value[cell]
                weight_sum += weight
    if weight_sum > Threshold:
        value /= weight_sum
    return value


@ti.func
def _sample_cell_pressure3d(position, grid_size, cell_type, cell_pressure):
    base = ti.floor(position / grid_size - 0.5).cast(int)
    value = 0.0
    weight_sum = 0.0
    for i, j, k in ti.ndrange(2, 2, 2):
        cell = base + vec3i(i, j, k)
        if (
            0 <= cell[0] < cell_type.shape[0]
            and 0 <= cell[1] < cell_type.shape[1]
            and 0 <= cell[2] < cell_type.shape[2]
        ):
            attr = int(cell_type[cell])
            if attr == FLUID_CELL or attr == AIR_CELL:
                center = (cell.cast(float) + 0.5) * grid_size
                wx = _linear_weight((position[0] - center[0]) / grid_size[0])
                wy = _linear_weight((position[1] - center[1]) / grid_size[1])
                wz = _linear_weight((position[2] - center[2]) / grid_size[2])
                weight = wx * wy * wz
                if attr == FLUID_CELL:
                    value += weight * cell_pressure[cell]
                weight_sum += weight
    if weight_sum > Threshold:
        value /= weight_sum
    return value


@ti.func
def _ghost_air_pressure_weight3d(air_cell, cell_type, cell_pressure, fluid_sdf):
    pressure = 0.0
    weight = 0.0
    for neighbor in range(6):
        axis = neighbor // 2
        offset = vec3i(0, 0, 0)
        offset[axis] = 1 if neighbor % 2 == 0 else -1
        fluid = air_cell + offset
        valid = True
        for dd in ti.static(range(3)):
            valid = valid and 0 <= fluid[dd] < cell_type.shape[dd]
        if valid and cell_type[fluid] == FLUID_CELL:
            theta = _pressure_free_surface_theta(fluid, air_cell, fluid_sdf)
            pressure += -(1.0 - theta) / theta * cell_pressure[fluid]
            weight += 1.0
    if weight > 0.0:
        pressure /= weight
    return pressure, weight


@ti.func
def _ghost_air_pressure3d(air_cell, cell_type, cell_pressure, fluid_sdf):
    pressure, _ = _ghost_air_pressure_weight3d(air_cell, cell_type, cell_pressure, fluid_sdf)
    return pressure


@ti.func
def _sample_cell_pressure_gfm3d(position, grid_size, cell_type, cell_pressure, fluid_sdf):
    base = ti.floor(position / grid_size - 0.5).cast(int)
    value = 0.0
    weight_sum = 0.0
    for i, j, k in ti.ndrange(2, 2, 2):
        cell = base + vec3i(i, j, k)
        if (
            0 <= cell[0] < cell_type.shape[0]
            and 0 <= cell[1] < cell_type.shape[1]
            and 0 <= cell[2] < cell_type.shape[2]
        ):
            attr = int(cell_type[cell])
            if attr == FLUID_CELL or attr == AIR_CELL:
                center = (cell.cast(float) + 0.5) * grid_size
                wx = _linear_weight((position[0] - center[0]) / grid_size[0])
                wy = _linear_weight((position[1] - center[1]) / grid_size[1])
                wz = _linear_weight((position[2] - center[2]) / grid_size[2])
                weight = wx * wy * wz
                if attr == FLUID_CELL:
                    value += weight * cell_pressure[cell]
                    weight_sum += weight
                else:
                    ghost_pressure, ghost_weight = _ghost_air_pressure_weight3d(
                        cell, cell_type, cell_pressure, fluid_sdf
                    )
                    if ghost_weight > 0.0:
                        value += weight * ghost_pressure
                        weight_sum += weight
    if weight_sum > Threshold:
        value /= weight_sum
    return value


@ti.func
def _sample_cell_pressure_gfm_shape3d(
    position,
    psize,
    shape_type: ti.template(),
    influenced_node: int,
    grid_size,
    cell_type,
    cell_pressure,
    fluid_sdf,
    air_ghost_pressure: ti.template() = True,
):
    base = vec3i(
        _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], shape_type),
        _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], shape_type),
        _mac_base_1d(position[2], 1.0 / grid_size[2], 0.5, psize[2], shape_type),
    )
    value = 0.0
    weight_sum = 0.0
    for i, j, k in ti.ndrange(influenced_node, influenced_node, influenced_node):
        cell = base + vec3i(i, j, k)
        if (
            0 <= cell[0] < cell_type.shape[0]
            and 0 <= cell[1] < cell_type.shape[1]
            and 0 <= cell[2] < cell_type.shape[2]
        ):
            attr = int(cell_type[cell])
            if attr == FLUID_CELL or attr == AIR_CELL or attr == SOLID_CELL:
                cell_pos = (cell.cast(float) + 0.5) * grid_size
                weight = _mac_weight3d(position, cell_pos, grid_size, psize, vec3i(0, 0, 0), shape_type)
                if attr == FLUID_CELL or attr == SOLID_CELL:
                    value += weight * cell_pressure[cell]
                    weight_sum += weight
                else:
                    if ti.static(air_ghost_pressure):
                        ghost_pressure, ghost_weight = _ghost_air_pressure_weight3d(
                            cell, cell_type, cell_pressure, fluid_sdf
                        )
                        if ghost_weight > 0.0:
                            value += weight * ghost_pressure
                            weight_sum += weight
                    else:
                        # Atmospheric air contributes zero physical pressure to the solid force.
                        weight_sum += weight
    if weight_sum > Threshold:
        value /= weight_sum
    return value


@ti.kernel
def kernel_extend_double_layer_wall_pressure(cell_type: ti.template(), pressure: ti.template(), radius: int):
    dimension = ti.static(len(cell_type.shape))
    for I in ti.grouped(cell_type):
        if cell_type[I] == SOLID_CELL:
            nearest = 4 * dimension * radius * radius + 1
            value, count = 0.0, 0.0
            width = 2 * radius + 1
            for flat in range(width**dimension):
                offset = ti.Vector.zero(int, dimension)
                local = flat
                inside = True
                for d in ti.static(range(dimension)):
                    offset[d] = local % width - radius
                    local //= width
                    inside = inside and 0 <= I[d] + offset[d] < cell_type.shape[d]
                source = I + offset
                distance = offset.dot(offset)
                if inside and cell_type[source] == FLUID_CELL and distance <= nearest:
                    if distance < nearest:
                        nearest = distance
                        value, count = 0.0, 0.0
                    extrapolated = pressure[source]
                    for d in ti.static(range(dimension)):
                        unit = ti.Vector.unit(dimension, d)
                        lower, upper = source - unit, source + unit
                        has_lower, has_upper = False, False
                        if lower[d] >= 0:
                            has_lower = cell_type[lower] == FLUID_CELL
                        if upper[d] < cell_type.shape[d]:
                            has_upper = cell_type[upper] == FLUID_CELL
                        difference = 0.0
                        if has_lower and has_upper:
                            difference = 0.5 * (pressure[upper] - pressure[lower])
                        elif has_upper:
                            difference = pressure[upper] - pressure[source]
                        elif has_lower:
                            difference = pressure[source] - pressure[lower]
                        extrapolated -= offset[d] * difference
                    value += extrapolated
                    count += 1.0
            # Only wall slots change. Fluid pressure and the atmospheric GFM
            # values retain their physical boundary conditions.
            pressure[I] = value / count if count > 0.0 else 0.0


@ti.func
def _sample_solid_grid_velocity3d(
    position,
    grid_size,
    gnum,
    cutoff: float,
    shape_type: ti.template(),
    influenced_node: int,
    calLength: ti.template(),
    node: ti.template(),
    sample_force: ti.template() = False,
):
    velocity = ZEROVEC3f
    weight_sum = 0.0
    if shape_type == MAC_SHAPE_LINEAR or shape_type == MAC_SHAPE_GIMP:
        base = ti.floor(position / grid_size).cast(int)
        for i, j, k in ti.ndrange(2, 2, 2):
            node_ijk = base + vec3i(i, j, k)
            if 0 <= node_ijk[0] < gnum[0] and 0 <= node_ijk[1] < gnum[1] and 0 <= node_ijk[2] < gnum[2]:
                node_pos = node_ijk.cast(float) * grid_size
                weight = _linear_weight((position[0] - node_pos[0]) / grid_size[0])
                weight *= _linear_weight((position[1] - node_pos[1]) / grid_size[1])
                weight *= _linear_weight((position[2] - node_pos[2]) / grid_size[2])
                node_id = int(node_ijk[0] + node_ijk[1] * gnum[0] + node_ijk[2] * gnum[0] * gnum[1])
                for body_id in range(node.shape[1]):
                    if node[node_id, body_id].ms > cutoff:
                        mass_weight = weight * node[node_id, body_id].ms
                        value = node[node_id, body_id].momentums
                        if ti.static(sample_force):
                            value = node[node_id, body_id].force / node[node_id, body_id].ms
                        velocity += mass_weight * value
                        weight_sum += mass_weight
    else:
        for body_id in range(node.shape[1]):
            psize = calLength[body_id]
            base = vec3i(
                _mac_base_1d(position[0], 1.0 / grid_size[0], 0.0, psize[0], shape_type),
                _mac_base_1d(position[1], 1.0 / grid_size[1], 0.0, psize[1], shape_type),
                _mac_base_1d(position[2], 1.0 / grid_size[2], 0.0, psize[2], shape_type),
            )
            for i, j, k in ti.ndrange(influenced_node, influenced_node, influenced_node):
                node_ijk = base + vec3i(i, j, k)
                if 0 <= node_ijk[0] < gnum[0] and 0 <= node_ijk[1] < gnum[1] and 0 <= node_ijk[2] < gnum[2]:
                    node_pos = node_ijk.cast(float) * grid_size
                    btype = vec3i(
                        _mac_bspline_boundary_type(node_ijk[0], gnum[0]),
                        _mac_bspline_boundary_type(node_ijk[1], gnum[1]),
                        _mac_bspline_boundary_type(node_ijk[2], gnum[2]),
                    )
                    weight = _mac_weight3d(position, node_pos, grid_size, psize, btype, shape_type)
                    node_id = int(node_ijk[0] + node_ijk[1] * gnum[0] + node_ijk[2] * gnum[0] * gnum[1])
                    if node[node_id, body_id].ms > cutoff:
                        mass_weight = weight * node[node_id, body_id].ms
                        value = node[node_id, body_id].momentums
                        if ti.static(sample_force):
                            value = node[node_id, body_id].force / node[node_id, body_id].ms
                        velocity += mass_weight * value
                        weight_sum += mass_weight
    if weight_sum > cutoff:
        velocity /= weight_sum
    return velocity, weight_sum


@ti.func
def _mac_component_value3d(axis, face, velocity_x, velocity_y, velocity_z):
    value = 0.0
    if axis == 0:
        value = velocity_x[face]
    elif axis == 1:
        value = velocity_y[face]
    else:
        value = velocity_z[face]
    return value


@ti.func
def _mac_component_stencil3d(
    axis, position, psize, shape_type: ti.template(), grid_size, velocity_x, velocity_y, velocity_z
):
    shape = vec3i(velocity_x.shape[0], velocity_x.shape[1], velocity_x.shape[2])
    if axis == 1:
        shape = vec3i(velocity_y.shape[0], velocity_y.shape[1], velocity_y.shape[2])
    elif axis == 2:
        shape = vec3i(velocity_z.shape[0], velocity_z.shape[1], velocity_z.shape[2])
    offset = vec3f(0.5, 0.5, 0.5)
    offset[axis] = 0.0
    base = vec3i(0, 0, 0)
    for d in ti.static(range(3)):
        base[d] = _mac_base_1d(position[d], 1.0 / grid_size[d], offset[d], psize[d], shape_type)
    return base, shape, offset


@ti.func
def _sample_mac_velocity3d(
    position, psize, shape_type: ti.template(), influenced_node: int, grid_size, velocity_x, velocity_y, velocity_z
):
    velocity = ZEROVEC3f
    # A serial component loop avoids three copies of the boundary shape CFG.
    axis = 0
    while axis < 3:
        base, shape, offset = _mac_component_stencil3d(
            axis, position, psize, shape_type, grid_size, velocity_x, velocity_y, velocity_z
        )
        value = 0.0
        weight_sum = 0.0
        for i, j, k in ti.ndrange(influenced_node, influenced_node, influenced_node):
            face = base + vec3i(i, j, k)
            if 0 <= face[0] < shape[0] and 0 <= face[1] < shape[1] and 0 <= face[2] < shape[2]:
                face_pos = (ti.cast(face, float) + offset) * grid_size
                btype = vec3i(0, 0, 0)
                btype[axis] = _mac_bspline_boundary_type(face[axis], shape[axis])
                weight = _mac_weight3d(position, face_pos, grid_size, psize, btype, shape_type)
                value += weight * _mac_component_value3d(axis, face, velocity_x, velocity_y, velocity_z)
                weight_sum += weight
        velocity[axis] = value
        if weight_sum > Threshold:
            velocity[axis] /= weight_sum
        axis += 1
    return velocity


@ti.func
def _sample_mac_velocity_gradient3d(
    position,
    psize,
    shape_type: ti.template(),
    influenced_node: int,
    grid_size,
    velocity_x,
    velocity_y,
    velocity_z,
    v_pic,
):
    gradv = ZEROMAT3x3
    axis = 0
    while axis < 3:
        base, shape, offset = _mac_component_stencil3d(
            axis, position, psize, shape_type, grid_size, velocity_x, velocity_y, velocity_z
        )
        moment = ZEROMAT3x3
        rhs = ZEROVEC3f
        for i, j, k in ti.ndrange(influenced_node, influenced_node, influenced_node):
            face = base + vec3i(i, j, k)
            if 0 <= face[0] < shape[0] and 0 <= face[1] < shape[1] and 0 <= face[2] < shape[2]:
                face_pos = (ti.cast(face, float) + offset) * grid_size
                btype = vec3i(0, 0, 0)
                btype[axis] = _mac_bspline_boundary_type(face[axis], shape[axis])
                weight = _mac_weight3d(position, face_pos, grid_size, psize, btype, shape_type)
                pointer = face_pos - position
                moment += weight * pointer.outer_product(pointer)
                value = _mac_component_value3d(axis, face, velocity_x, velocity_y, velocity_z)
                rhs += weight * (value - v_pic[axis]) * pointer
        trace = ti.max(moment[0, 0] + moment[1, 1] + moment[2, 2], 0.0)
        determinant = ti.abs(moment.determinant())
        if trace > 1.0e-20 and determinant > 1.0e-8 * trace * trace * trace:
            row = moment.inverse() @ rhs
            for d in ti.static(range(3)):
                gradv[axis, d] = row[d]
        axis += 1
    return gradv


@ti.kernel
def kernel_reset_double_layer_fields3d(
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_mass_z: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
    fluid_velocity0_x: ti.template(),
    fluid_velocity0_y: ti.template(),
    fluid_velocity0_z: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    fluid_acceleration_z: ti.template(),
    solid_mass_x: ti.template(),
    solid_mass_y: ti.template(),
    solid_mass_z: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    solid_velocity_z: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    face_porosity_z: ti.template(),
    cell_type: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_solid_mass: ti.template(),
    cell_porosity: ti.template(),
    cell_solid_velocity: ti.template(),
    cell_fluid_velocity: ti.template(),
    cell_pressure: ti.template(),
):
    for I in ti.grouped(fluid_mass_x):
        fluid_mass_x[I] = 0.0
        fluid_velocity_x[I] = 0.0
        fluid_velocity0_x[I] = 0.0
        fluid_acceleration_x[I] = 0.0
        solid_mass_x[I] = 0.0
        solid_velocity_x[I] = 0.0
        face_porosity_x[I] = 0.0
    for I in ti.grouped(fluid_mass_y):
        fluid_mass_y[I] = 0.0
        fluid_velocity_y[I] = 0.0
        fluid_velocity0_y[I] = 0.0
        fluid_acceleration_y[I] = 0.0
        solid_mass_y[I] = 0.0
        solid_velocity_y[I] = 0.0
        face_porosity_y[I] = 0.0
    for I in ti.grouped(fluid_mass_z):
        fluid_mass_z[I] = 0.0
        fluid_velocity_z[I] = 0.0
        fluid_velocity0_z[I] = 0.0
        fluid_acceleration_z[I] = 0.0
        solid_mass_z[I] = 0.0
        solid_velocity_z[I] = 0.0
        face_porosity_z[I] = 0.0
    for I in ti.grouped(cell_type):
        cell_type[I] = AIR_CELL
        cell_fluid_mass[I] = 0.0
        cell_solid_mass[I] = 0.0
        cell_porosity[I] = 0.0
        cell_solid_velocity[I] = ZEROVEC3f
        cell_fluid_velocity[I] = ZEROVEC3f
        cell_pressure[I] = 0.0


@ti.kernel
def kernel_reset_double_layer_material_fields3d(
    face_material_weight_x: ti.template(),
    face_material_weight_y: ti.template(),
    face_material_weight_z: ti.template(),
    face_solid_density_x: ti.template(),
    face_solid_density_y: ti.template(),
    face_solid_density_z: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    face_fluid_density_z: ti.template(),
    face_fluid_viscosity_x: ti.template(),
    face_fluid_viscosity_y: ti.template(),
    face_fluid_viscosity_z: ti.template(),
    face_grain_diameter_x: ti.template(),
    face_grain_diameter_y: ti.template(),
    face_grain_diameter_z: ti.template(),
    face_permeability_x: ti.template(),
    face_permeability_y: ti.template(),
    face_permeability_z: ti.template(),
    face_fluid_unit_weight_x: ti.template(),
    face_fluid_unit_weight_y: ti.template(),
    face_fluid_unit_weight_z: ti.template(),
    face_drag_model_x: ti.template(),
    face_drag_model_y: ti.template(),
    face_drag_model_z: ti.template(),
    cell_material_weight: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    cell_fluid_viscosity: ti.template(),
    cell_grain_diameter: ti.template(),
    node_material_weight: ti.template(),
    node_solid_density: ti.template(),
    node_fluid_density: ti.template(),
    node_fluid_viscosity: ti.template(),
    node_grain_diameter: ti.template(),
    node_permeability: ti.template(),
    node_fluid_unit_weight: ti.template(),
    node_drag_model: ti.template(),
):
    for I in ti.grouped(face_material_weight_x):
        face_material_weight_x[I] = 0.0
        face_solid_density_x[I] = 0.0
        face_fluid_density_x[I] = 0.0
        face_fluid_viscosity_x[I] = 0.0
        face_grain_diameter_x[I] = 0.0
        face_permeability_x[I] = 0.0
        face_fluid_unit_weight_x[I] = 0.0
        face_drag_model_x[I] = 0.0
    for I in ti.grouped(face_material_weight_y):
        face_material_weight_y[I] = 0.0
        face_solid_density_y[I] = 0.0
        face_fluid_density_y[I] = 0.0
        face_fluid_viscosity_y[I] = 0.0
        face_grain_diameter_y[I] = 0.0
        face_permeability_y[I] = 0.0
        face_fluid_unit_weight_y[I] = 0.0
        face_drag_model_y[I] = 0.0
    for I in ti.grouped(face_material_weight_z):
        face_material_weight_z[I] = 0.0
        face_solid_density_z[I] = 0.0
        face_fluid_density_z[I] = 0.0
        face_fluid_viscosity_z[I] = 0.0
        face_grain_diameter_z[I] = 0.0
        face_permeability_z[I] = 0.0
        face_fluid_unit_weight_z[I] = 0.0
        face_drag_model_z[I] = 0.0
    for I in ti.grouped(cell_material_weight):
        cell_material_weight[I] = 0.0
        cell_solid_density[I] = 0.0
        cell_fluid_density[I] = 0.0
        cell_fluid_viscosity[I] = 0.0
        cell_grain_diameter[I] = 0.0
    for I in ti.grouped(node_material_weight):
        node_material_weight[I] = 0.0
        node_solid_density[I] = 0.0
        node_fluid_density[I] = 0.0
        node_fluid_viscosity[I] = 0.0
        node_grain_diameter[I] = 0.0
        node_permeability[I] = 0.0
        node_fluid_unit_weight[I] = 0.0
        node_drag_model[I] = 0.0


@ti.kernel
def kernel_accumulate_double_layer_material_fields3d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    grid_size: vec3f,
    mac_shape_type: ti.template(),
    mac_influenced_node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    matProps: ti.template(),
    calLength: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    face_material_weight_x: ti.template(),
    face_material_weight_y: ti.template(),
    face_material_weight_z: ti.template(),
    face_solid_density_x: ti.template(),
    face_solid_density_y: ti.template(),
    face_solid_density_z: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    face_fluid_density_z: ti.template(),
    face_fluid_viscosity_x: ti.template(),
    face_fluid_viscosity_y: ti.template(),
    face_fluid_viscosity_z: ti.template(),
    face_grain_diameter_x: ti.template(),
    face_grain_diameter_y: ti.template(),
    face_grain_diameter_z: ti.template(),
    face_permeability_x: ti.template(),
    face_permeability_y: ti.template(),
    face_permeability_z: ti.template(),
    face_fluid_unit_weight_x: ti.template(),
    face_fluid_unit_weight_y: ti.template(),
    face_fluid_unit_weight_z: ti.template(),
    face_drag_model_x: ti.template(),
    face_drag_model_y: ti.template(),
    face_drag_model_z: ti.template(),
    cell_material_weight: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    cell_fluid_viscosity: ti.template(),
    cell_grain_diameter: ti.template(),
    node_material_weight: ti.template(),
    node_solid_density: ti.template(),
    node_fluid_density: ti.template(),
    node_fluid_viscosity: ti.template(),
    node_grain_diameter: ti.template(),
    node_permeability: ti.template(),
    node_fluid_unit_weight: ti.template(),
    node_drag_model: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            position = particle[np].x
            psize = calLength[int(particle[np].bodyID)]
            mass = ti.max(particle[np].m, particle[np].ms + particle[np].mf)
            if mass > Threshold:
                cell = ti.floor(position / grid_size).cast(int)
                if (
                    0 <= cell[0] < cell_material_weight.shape[0]
                    and 0 <= cell[1] < cell_material_weight.shape[1]
                    and 0 <= cell[2] < cell_material_weight.shape[2]
                ):
                    cell_material_weight[cell] += mass
                    cell_solid_density[cell] += mass * matProps.solid_density
                    cell_fluid_density[cell] += mass * matProps.fluid_density
                    cell_fluid_viscosity[cell] += mass * matProps.fluid_viscosity
                    cell_grain_diameter[cell] += mass * matProps.grain_diameter

                base_x = vec3i(
                    _mac_base_1d(position[0], 1.0 / grid_size[0], 0.0, psize[0], mac_shape_type),
                    _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], mac_shape_type),
                    _mac_base_1d(position[2], 1.0 / grid_size[2], 0.5, psize[2], mac_shape_type),
                )
                for ix, iy, iz in ti.ndrange(mac_influenced_node, mac_influenced_node, mac_influenced_node):
                    face = base_x + vec3i(ix, iy, iz)
                    if (
                        0 <= face[0] < face_material_weight_x.shape[0]
                        and 0 <= face[1] < face_material_weight_x.shape[1]
                        and 0 <= face[2] < face_material_weight_x.shape[2]
                    ):
                        face_pos = vec3f(
                            face[0] * grid_size[0], (face[1] + 0.5) * grid_size[1], (face[2] + 0.5) * grid_size[2]
                        )
                        btype = vec3i(_mac_bspline_boundary_type(face[0], face_material_weight_x.shape[0]), 0, 0)
                        weight = _mac_weight3d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                        weighted_mass = weight * mass
                        face_material_weight_x[face] += weighted_mass
                        face_solid_density_x[face] += weighted_mass * matProps.solid_density
                        face_fluid_density_x[face] += weighted_mass * matProps.fluid_density
                        face_fluid_viscosity_x[face] += weighted_mass * matProps.fluid_viscosity
                        face_grain_diameter_x[face] += weighted_mass * matProps.grain_diameter
                        face_permeability_x[face] += weighted_mass * matProps.permeability
                        face_fluid_unit_weight_x[face] += weighted_mass * matProps.fluid_unit_weight
                        face_drag_model_x[face] += weighted_mass * matProps.drag_model

                base_y = vec3i(
                    _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], mac_shape_type),
                    _mac_base_1d(position[1], 1.0 / grid_size[1], 0.0, psize[1], mac_shape_type),
                    _mac_base_1d(position[2], 1.0 / grid_size[2], 0.5, psize[2], mac_shape_type),
                )
                for ix, iy, iz in ti.ndrange(mac_influenced_node, mac_influenced_node, mac_influenced_node):
                    face = base_y + vec3i(ix, iy, iz)
                    if (
                        0 <= face[0] < face_material_weight_y.shape[0]
                        and 0 <= face[1] < face_material_weight_y.shape[1]
                        and 0 <= face[2] < face_material_weight_y.shape[2]
                    ):
                        face_pos = vec3f(
                            (face[0] + 0.5) * grid_size[0], face[1] * grid_size[1], (face[2] + 0.5) * grid_size[2]
                        )
                        btype = vec3i(0, _mac_bspline_boundary_type(face[1], face_material_weight_y.shape[1]), 0)
                        weight = _mac_weight3d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                        weighted_mass = weight * mass
                        face_material_weight_y[face] += weighted_mass
                        face_solid_density_y[face] += weighted_mass * matProps.solid_density
                        face_fluid_density_y[face] += weighted_mass * matProps.fluid_density
                        face_fluid_viscosity_y[face] += weighted_mass * matProps.fluid_viscosity
                        face_grain_diameter_y[face] += weighted_mass * matProps.grain_diameter
                        face_permeability_y[face] += weighted_mass * matProps.permeability
                        face_fluid_unit_weight_y[face] += weighted_mass * matProps.fluid_unit_weight
                        face_drag_model_y[face] += weighted_mass * matProps.drag_model

                base_z = vec3i(
                    _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], mac_shape_type),
                    _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], mac_shape_type),
                    _mac_base_1d(position[2], 1.0 / grid_size[2], 0.0, psize[2], mac_shape_type),
                )
                for ix, iy, iz in ti.ndrange(mac_influenced_node, mac_influenced_node, mac_influenced_node):
                    face = base_z + vec3i(ix, iy, iz)
                    if (
                        0 <= face[0] < face_material_weight_z.shape[0]
                        and 0 <= face[1] < face_material_weight_z.shape[1]
                        and 0 <= face[2] < face_material_weight_z.shape[2]
                    ):
                        face_pos = vec3f(
                            (face[0] + 0.5) * grid_size[0], (face[1] + 0.5) * grid_size[1], face[2] * grid_size[2]
                        )
                        btype = vec3i(0, 0, _mac_bspline_boundary_type(face[2], face_material_weight_z.shape[2]))
                        weight = _mac_weight3d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                        weighted_mass = weight * mass
                        face_material_weight_z[face] += weighted_mass
                        face_solid_density_z[face] += weighted_mass * matProps.solid_density
                        face_fluid_density_z[face] += weighted_mass * matProps.fluid_density
                        face_fluid_viscosity_z[face] += weighted_mass * matProps.fluid_viscosity
                        face_grain_diameter_z[face] += weighted_mass * matProps.grain_diameter
                        face_permeability_z[face] += weighted_mass * matProps.permeability
                        face_fluid_unit_weight_z[face] += weighted_mass * matProps.fluid_unit_weight
                        face_drag_model_z[face] += weighted_mass * matProps.drag_model

                bodyID = int(particle[np].bodyID)
                offset = np * total_nodes
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    weighted_mass = shapefn[ln] * mass
                    node_material_weight[nodeID, bodyID] += weighted_mass
                    node_solid_density[nodeID, bodyID] += weighted_mass * matProps.solid_density
                    node_fluid_density[nodeID, bodyID] += weighted_mass * matProps.fluid_density
                    node_fluid_viscosity[nodeID, bodyID] += weighted_mass * matProps.fluid_viscosity
                    node_grain_diameter[nodeID, bodyID] += weighted_mass * matProps.grain_diameter
                    node_permeability[nodeID, bodyID] += weighted_mass * matProps.permeability
                    node_fluid_unit_weight[nodeID, bodyID] += weighted_mass * matProps.fluid_unit_weight
                    node_drag_model[nodeID, bodyID] += weighted_mass * matProps.drag_model


@ti.kernel
def kernel_normalize_double_layer_material_fields3d(
    default_matProps: ti.template(),
    face_material_weight_x: ti.template(),
    face_material_weight_y: ti.template(),
    face_material_weight_z: ti.template(),
    face_solid_density_x: ti.template(),
    face_solid_density_y: ti.template(),
    face_solid_density_z: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    face_fluid_density_z: ti.template(),
    face_fluid_viscosity_x: ti.template(),
    face_fluid_viscosity_y: ti.template(),
    face_fluid_viscosity_z: ti.template(),
    face_grain_diameter_x: ti.template(),
    face_grain_diameter_y: ti.template(),
    face_grain_diameter_z: ti.template(),
    face_permeability_x: ti.template(),
    face_permeability_y: ti.template(),
    face_permeability_z: ti.template(),
    face_fluid_unit_weight_x: ti.template(),
    face_fluid_unit_weight_y: ti.template(),
    face_fluid_unit_weight_z: ti.template(),
    face_drag_model_x: ti.template(),
    face_drag_model_y: ti.template(),
    face_drag_model_z: ti.template(),
    cell_material_weight: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    cell_fluid_viscosity: ti.template(),
    cell_grain_diameter: ti.template(),
    node_material_weight: ti.template(),
    node_solid_density: ti.template(),
    node_fluid_density: ti.template(),
    node_fluid_viscosity: ti.template(),
    node_grain_diameter: ti.template(),
    node_permeability: ti.template(),
    node_fluid_unit_weight: ti.template(),
    node_drag_model: ti.template(),
):
    for I in ti.grouped(face_material_weight_x):
        if face_material_weight_x[I] > Threshold:
            inv_weight = 1.0 / face_material_weight_x[I]
            face_solid_density_x[I] *= inv_weight
            face_fluid_density_x[I] *= inv_weight
            face_fluid_viscosity_x[I] *= inv_weight
            face_grain_diameter_x[I] *= inv_weight
            face_permeability_x[I] *= inv_weight
            face_fluid_unit_weight_x[I] *= inv_weight
            face_drag_model_x[I] *= inv_weight
        else:
            face_solid_density_x[I] = default_matProps.solid_density
            face_fluid_density_x[I] = default_matProps.fluid_density
            face_fluid_viscosity_x[I] = default_matProps.fluid_viscosity
            face_grain_diameter_x[I] = default_matProps.grain_diameter
            face_permeability_x[I] = default_matProps.permeability
            face_fluid_unit_weight_x[I] = default_matProps.fluid_unit_weight
            face_drag_model_x[I] = default_matProps.drag_model
    for I in ti.grouped(face_material_weight_y):
        if face_material_weight_y[I] > Threshold:
            inv_weight = 1.0 / face_material_weight_y[I]
            face_solid_density_y[I] *= inv_weight
            face_fluid_density_y[I] *= inv_weight
            face_fluid_viscosity_y[I] *= inv_weight
            face_grain_diameter_y[I] *= inv_weight
            face_permeability_y[I] *= inv_weight
            face_fluid_unit_weight_y[I] *= inv_weight
            face_drag_model_y[I] *= inv_weight
        else:
            face_solid_density_y[I] = default_matProps.solid_density
            face_fluid_density_y[I] = default_matProps.fluid_density
            face_fluid_viscosity_y[I] = default_matProps.fluid_viscosity
            face_grain_diameter_y[I] = default_matProps.grain_diameter
            face_permeability_y[I] = default_matProps.permeability
            face_fluid_unit_weight_y[I] = default_matProps.fluid_unit_weight
            face_drag_model_y[I] = default_matProps.drag_model
    for I in ti.grouped(face_material_weight_z):
        if face_material_weight_z[I] > Threshold:
            inv_weight = 1.0 / face_material_weight_z[I]
            face_solid_density_z[I] *= inv_weight
            face_fluid_density_z[I] *= inv_weight
            face_fluid_viscosity_z[I] *= inv_weight
            face_grain_diameter_z[I] *= inv_weight
            face_permeability_z[I] *= inv_weight
            face_fluid_unit_weight_z[I] *= inv_weight
            face_drag_model_z[I] *= inv_weight
        else:
            face_solid_density_z[I] = default_matProps.solid_density
            face_fluid_density_z[I] = default_matProps.fluid_density
            face_fluid_viscosity_z[I] = default_matProps.fluid_viscosity
            face_grain_diameter_z[I] = default_matProps.grain_diameter
            face_permeability_z[I] = default_matProps.permeability
            face_fluid_unit_weight_z[I] = default_matProps.fluid_unit_weight
            face_drag_model_z[I] = default_matProps.drag_model
    for I in ti.grouped(cell_material_weight):
        if cell_material_weight[I] > Threshold:
            inv_weight = 1.0 / cell_material_weight[I]
            cell_solid_density[I] *= inv_weight
            cell_fluid_density[I] *= inv_weight
            cell_fluid_viscosity[I] *= inv_weight
            cell_grain_diameter[I] *= inv_weight
        else:
            cell_solid_density[I] = default_matProps.solid_density
            cell_fluid_density[I] = default_matProps.fluid_density
            cell_fluid_viscosity[I] = default_matProps.fluid_viscosity
            cell_grain_diameter[I] = default_matProps.grain_diameter
    for I in ti.grouped(node_material_weight):
        if node_material_weight[I] > Threshold:
            inv_weight = 1.0 / node_material_weight[I]
            node_solid_density[I] *= inv_weight
            node_fluid_density[I] *= inv_weight
            node_fluid_viscosity[I] *= inv_weight
            node_grain_diameter[I] *= inv_weight
            node_permeability[I] *= inv_weight
            node_fluid_unit_weight[I] *= inv_weight
            node_drag_model[I] *= inv_weight
        else:
            node_solid_density[I] = default_matProps.solid_density
            node_fluid_density[I] = default_matProps.fluid_density
            node_fluid_viscosity[I] = default_matProps.fluid_viscosity
            node_grain_diameter[I] = default_matProps.grain_diameter
            node_permeability[I] = default_matProps.permeability
            node_fluid_unit_weight[I] = default_matProps.fluid_unit_weight
            node_drag_model[I] = default_matProps.drag_model


@ti.func
def _double_layer_fluid_fraction3d(I, grid_size, cell_fluid_mass, cell_fluid_density, cell_porosity):
    capacity = cell_fluid_density[I] * _clamp_porosity(cell_porosity[I]) * grid_size[0] * grid_size[1] * grid_size[2]
    return ti.min(1.0, ti.max(0.0, cell_fluid_mass[I] / ti.max(capacity, 1.0e-12)))


@ti.kernel
def kernel_classify_double_layer_fluid_cells3d(
    particleNum: int,
    grid_size: vec3f,
    particle: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_fluid_density: ti.template(),
    cell_porosity: ti.template(),
    cell_type: ti.template(),
):
    for I in ti.grouped(cell_type):
        cell_type[I] = AIR_CELL
        if _double_layer_fluid_fraction3d(I, grid_size, cell_fluid_mass, cell_fluid_density, cell_porosity) >= 0.5:
            cell_type[I] = FLUID_CELL

    for np in range(particleNum):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
        ):
            cell = ti.floor(particle[np].x / grid_size).cast(int)
            if (
                0 <= cell[0] < cell_type.shape[0]
                and 0 <= cell[1] < cell_type.shape[1]
                and 0 <= cell[2] < cell_type.shape[2]
                and cell_type[cell] == AIR_CELL
            ):
                resolved = False
                for d in ti.static(range(3)):
                    unit = ti.Vector.unit(3, d)
                    left = cell - unit
                    right = cell + unit
                    if 0 <= left[d]:
                        if cell_type[left] == FLUID_CELL:
                            resolved = True
                    if right[d] < cell_type.shape[d]:
                        if cell_type[right] == FLUID_CELL:
                            resolved = True
                # Saturated porous cells carry pressure even beside a resolved water cell.
                if cell_porosity[cell] < 1.0 - Threshold:
                    cell_type[cell] = 3
                elif not resolved:
                    attached = False
                    for ix, iy, iz in ti.ndrange(3, 3, 3):
                        neighbor = cell + vec3i(ix - 1, iy - 1, iz - 1)
                        if (
                            0 <= neighbor[0] < cell_type.shape[0]
                            and 0 <= neighbor[1] < cell_type.shape[1]
                            and 0 <= neighbor[2] < cell_type.shape[2]
                        ):
                            attached = attached or cell_type[neighbor] == FLUID_CELL
                    if attached:
                        cell_type[cell] = 3
    for I in ti.grouped(cell_type):
        if cell_type[I] == 3:
            cell_type[I] = FLUID_CELL

    for i, j in ti.ndrange(cell_type.shape[0], cell_type.shape[1]):
        lower = -1
        for k in range(cell_type.shape[2]):
            if cell_fluid_mass[i, j, k] <= Threshold:
                lower = -1
            elif (
                _double_layer_fluid_fraction3d(
                    vec3i(i, j, k), grid_size, cell_fluid_mass, cell_fluid_density, cell_porosity
                )
                >= 0.5
            ):
                if lower >= 0:
                    for interior in range(lower + 1, k):
                        cell_type[i, j, interior] = FLUID_CELL
                lower = k


@ti.kernel
def kernel_build_double_layer_fluid_sdf3d(
    grid_size: vec3f,
    cell_fluid_mass: ti.template(),
    cell_fluid_density: ti.template(),
    cell_porosity: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
):
    min_dx = ti.min(grid_size[0], ti.min(grid_size[1], grid_size[2]))
    minimum_distance = 0.1 * min_dx
    for I in ti.grouped(fluid_sdf):
        fraction = _double_layer_fluid_fraction3d(I, grid_size, cell_fluid_mass, cell_fluid_density, cell_porosity)
        phi = (0.5 - fraction) * min_dx
        if cell_type[I] == FLUID_CELL:
            phi = ti.min(phi, -minimum_distance)
        else:
            phi = ti.max(phi, minimum_distance)
        fluid_sdf[I] = phi


@ti.kernel
def kernel_mark_double_layer_solid_cell_region3d(
    grid_size: vec3f,
    start_point: vec3f,
    end_point: vec3f,
    cell_type: ti.template(),
):
    for I in ti.grouped(cell_type):
        inside = True
        for d in ti.static(range(3)):
            cell_center = (ti.cast(I[d], float) + 0.5) * grid_size[d]
            eps = 1.0e-6 * grid_size[d]
            inside = inside and cell_center >= start_point[d] - eps and cell_center <= end_point[d] + eps
        if inside:
            cell_type[I] = SOLID_CELL


@ti.kernel
def kernel_constrain_double_layer_particles_to_solid_region3d(
    particle_count: int,
    domain: vec3f,
    grid_size: vec3f,
    start_point: vec3f,
    end_point: vec3f,
    particle: ti.template(),
):
    eps = 1.0e-6 * ti.max(ti.max(domain[0], domain[1]), domain[2])
    discrete_start = ZEROVEC3f
    discrete_end = ZEROVEC3f
    for d in ti.static(range(3)):
        first_cell = ti.ceil(start_point[d] / grid_size[d] - 0.5)
        last_cell = ti.floor(end_point[d] / grid_size[d] - 0.5)
        discrete_start[d] = ti.max(0.0, first_cell * grid_size[d])
        discrete_end[d] = ti.min(domain[d], (last_cell + 1.0) * grid_size[d])
    for np in range(particle_count):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            position = particle[np].x
            inside = True
            for d in ti.static(range(3)):
                inside = inside and position[d] >= discrete_start[d] - eps and position[d] <= discrete_end[d] + eps
            if inside:
                prefer_boundary_axis = 0
                for d in ti.static(range(3)):
                    if (discrete_start[d] <= eps) != (discrete_end[d] >= domain[d] - eps):
                        prefer_boundary_axis = 1

                best_axis = -1
                best_distance = 1.0e30
                target = 0.0
                normal = 0.0
                for d in ti.static(range(3)):
                    use_axis = (
                        prefer_boundary_axis == 0 or discrete_start[d] <= eps or discrete_end[d] >= domain[d] - eps
                    ) and not (discrete_start[d] <= eps and discrete_end[d] >= domain[d] - eps)
                    if use_axis:
                        candidate = 0.0
                        candidate_normal = 0.0
                        if discrete_start[d] <= eps:
                            candidate = discrete_end[d] + eps
                            candidate_normal = 1.0
                        elif discrete_end[d] >= domain[d] - eps:
                            candidate = discrete_start[d] - eps
                            candidate_normal = -1.0
                        else:
                            lower_distance = ti.abs(position[d] - discrete_start[d])
                            upper_distance = ti.abs(discrete_end[d] - position[d])
                            if lower_distance < upper_distance:
                                candidate = discrete_start[d] - eps
                                candidate_normal = -1.0
                            else:
                                candidate = discrete_end[d] + eps
                                candidate_normal = 1.0
                        distance = ti.abs(candidate - position[d])
                        if distance < best_distance:
                            best_distance = distance
                            best_axis = d
                            target = candidate
                            normal = candidate_normal

                for d in ti.static(range(3)):
                    if best_axis == d:
                        particle[np].x[d] = ti.min(ti.max(target, 1.0e-8), domain[d] - 1.0e-8)
                        if particle[np].v[d] * normal < 0.0:
                            particle[np].v[d] = 0.0
                        if int(particle[np].phase) == PHASE_SOLID:
                            if particle[np].vs[d] * normal < 0.0:
                                particle[np].vs[d] = 0.0
                        elif int(particle[np].phase) == PHASE_FLUID:
                            if particle[np].vf[d] * normal < 0.0:
                                particle[np].vf[d] = 0.0


@ti.kernel
def kernel_enforce_double_layer_solid_cell_faces3d(
    cell_type: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    fluid_acceleration_z: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    solid_velocity_z: ti.template(),
):
    for I in ti.grouped(fluid_velocity_x):
        if 0 < I[0] < fluid_velocity_x.shape[0] - 1:
            left = I - vec3i(1, 0, 0)
            right = I
            if (cell_type[left] == SOLID_CELL and cell_type[right] == FLUID_CELL) or (
                cell_type[left] == FLUID_CELL and cell_type[right] == SOLID_CELL
            ):
                fluid_velocity_x[I] = solid_velocity_x[I]
                fluid_acceleration_x[I] = 0.0
    for I in ti.grouped(fluid_velocity_y):
        if 0 < I[1] < fluid_velocity_y.shape[1] - 1:
            down = I - vec3i(0, 1, 0)
            up = I
            if (cell_type[down] == SOLID_CELL and cell_type[up] == FLUID_CELL) or (
                cell_type[down] == FLUID_CELL and cell_type[up] == SOLID_CELL
            ):
                fluid_velocity_y[I] = solid_velocity_y[I]
                fluid_acceleration_y[I] = 0.0
    for I in ti.grouped(fluid_velocity_z):
        if 0 < I[2] < fluid_velocity_z.shape[2] - 1:
            back = I - vec3i(0, 0, 1)
            front = I
            if (cell_type[back] == SOLID_CELL and cell_type[front] == FLUID_CELL) or (
                cell_type[back] == FLUID_CELL and cell_type[front] == SOLID_CELL
            ):
                fluid_velocity_z[I] = solid_velocity_z[I]
                fluid_acceleration_z[I] = 0.0


@ti.kernel
def kernel_p2g_double_layer_mass3d(
    total_nodes: int,
    particleNum: int,
    grid_size: vec3f,
    gnum: ti.types.vector(3, int),
    use_affine: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            phase = int(particle[np].phase)
            position = particle[np].x
            if phase == PHASE_SOLID:
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    shape = shapefn[ln]
                    nmass_s = shape * particle[np].ms
                    nodal_velocity_s = particle[np].vs
                    if ti.static(use_affine != 0):
                        nodal_coord = grid_size * vec3f(vectorize_id(nodeID, gnum))
                        nodal_velocity_s += particle[np].solid_velocity_gradient @ (nodal_coord - position)
                    node[nodeID, bodyID].m += nmass_s
                    node[nodeID, bodyID].ms += nmass_s
                    node[nodeID, bodyID].momentum += nmass_s * nodal_velocity_s
                    node[nodeID, bodyID].momentums += nmass_s * nodal_velocity_s
                    shape_volume = shape * particle[np].vol
                    node[nodeID, bodyID].porosity += shape_volume * (1.0 - particle[np].porosity)
            elif phase == PHASE_FLUID:
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    shape = shapefn[ln]
                    nmass_f = shape * particle[np].mf
                    nodal_velocity_f = particle[np].vf
                    if ti.static(use_affine != 0):
                        nodal_coord = grid_size * vec3f(vectorize_id(nodeID, gnum))
                        nodal_velocity_f += particle[np].fluid_velocity_gradient @ (nodal_coord - position)
                    node[nodeID, bodyID].mf += nmass_f
                    node[nodeID, bodyID].momentumf += nmass_f * nodal_velocity_f
            else:
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    shape = shapefn[ln]
                    nmass = shape * particle[np].m
                    nmass_s = shape * particle[np].ms
                    nmass_f = shape * particle[np].mf
                    nodal_velocity = particle[np].v
                    nodal_velocity_s = particle[np].vs
                    nodal_velocity_f = particle[np].vf
                    if ti.static(use_affine != 0):
                        nodal_coord = grid_size * vec3f(vectorize_id(nodeID, gnum))
                        pointer = nodal_coord - position
                        nodal_velocity += particle[np].solid_velocity_gradient @ pointer
                        nodal_velocity_s += particle[np].solid_velocity_gradient @ pointer
                        nodal_velocity_f += particle[np].fluid_velocity_gradient @ pointer
                    node[nodeID, bodyID].m += nmass
                    node[nodeID, bodyID].ms += nmass_s
                    node[nodeID, bodyID].mf += nmass_f
                    node[nodeID, bodyID].momentum += nmass * nodal_velocity
                    node[nodeID, bodyID].momentums += nmass_s * nodal_velocity_s
                    node[nodeID, bodyID].momentumf += nmass_f * nodal_velocity_f


@ti.kernel
def kernel_normalize_double_layer_nodes3d(cutoff: float, node: ti.template(), reference_volume: ti.template()):
    for ng, nb in node:
        if node[ng, nb].m > cutoff:
            node[ng, nb].momentum /= node[ng, nb].m
        # Retain velocity support for gradients; cutoff only gates the force/mass update.
        if node[ng, nb].ms > 0.0:
            node[ng, nb].momentums /= node[ng, nb].ms
        if node[ng, nb].mf > cutoff:
            node[ng, nb].momentumf /= node[ng, nb].mf
        solid_fraction = 0.0
        if reference_volume[ng, nb] > 0.0:
            solid_fraction = node[ng, nb].porosity / reference_volume[ng, nb]
        node[ng, nb].porosity = _mapped_porosity(solid_fraction)


@ti.kernel
def kernel_update_double_layer_solid_state3d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    dt: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_SOLID
        ):
            if particle[np].fix_v[0] == 0 or particle[np].fix_v[1] == 0 or particle[np].fix_v[2] == 0:
                bodyID = int(particle[np].bodyID)
                offset = np * total_nodes
                gradv = ZEROMAT3x3
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    gradv += node[nodeID, bodyID].momentums.outer_product(dshapefn[ln])
                particle[np].solid_velocity_gradient = gradv
                volume_ratio = matProps.update_particle_volume(np, gradv, stateVars, dt)
                porosity = matProps.update_particle_porosity(gradv, particle[np].porosity, dt)
                particle[np].vol *= volume_ratio
                particle[np].porosity = porosity
                if porosity > matProps.maximum_porosity:
                    particle[np].stress *= 0.0
                else:
                    particle[np].stress = matProps.ComputeStress(np, particle[np].stress, gradv, stateVars, dt)


@ti.kernel
def kernel_force_p2g_double_layer3d(
    total_nodes: int,
    particleNum: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particleNum):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_SOLID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            internal = -particle[np].vol * particle[np].stress
            external = particle[np].external_force
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                dshape = dshapefn[ln]
                shape = shapefn[ln]
                pforce = (
                    vec3f(
                        dshape[0] * internal[0] + dshape[1] * internal[3] + dshape[2] * internal[5],
                        dshape[1] * internal[1] + dshape[0] * internal[3] + dshape[2] * internal[4],
                        dshape[2] * internal[2] + dshape[1] * internal[4] + dshape[0] * internal[5],
                    )
                    + shape * external
                )
                node[nodeID, bodyID].force += pforce
                node[nodeID, bodyID].forces += pforce


@ti.kernel
def kernel_mac_p2g_double_layer3d(
    particleNum: int,
    grid_size: vec3f,
    mac_shape_type: ti.template(),
    mac_influenced_node: ti.template(),
    use_affine: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_mass_z: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
    solid_mass_x: ti.template(),
    solid_mass_y: ti.template(),
    solid_mass_z: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    solid_velocity_z: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    face_porosity_z: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_solid_mass: ti.template(),
    cell_porosity: ti.template(),
    cell_solid_velocity: ti.template(),
    cell_fluid_velocity: ti.template(),
    particle: ti.template(),
    calLength: ti.template(),
):
    cell_measure = grid_size[0] * grid_size[1] * grid_size[2]
    cell_shape_type, cell_support = MAC_SHAPE_LINEAR, 2
    if mac_shape_type == MAC_SHAPE_QUAD_BSPLINE:
        cell_shape_type, cell_support = MAC_SHAPE_QUAD_BSPLINE, 3
    for np in range(particleNum):
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            phase = int(particle[np].phase)
            position = particle[np].x
            psize = calLength[int(particle[np].bodyID)]
            base_cell = vec3i(
                _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], cell_shape_type),
                _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], cell_shape_type),
                _mac_base_1d(position[2], 1.0 / grid_size[2], 0.5, psize[2], cell_shape_type),
            )
            for i, j, k in ti.ndrange(cell_support, cell_support, cell_support):
                cell = base_cell + vec3i(i, j, k)
                if (
                    0 <= cell[0] < cell_fluid_mass.shape[0]
                    and 0 <= cell[1] < cell_fluid_mass.shape[1]
                    and 0 <= cell[2] < cell_fluid_mass.shape[2]
                ):
                    cell_pos = (cell.cast(float) + 0.5) * grid_size
                    weight = _mac_weight3d(position, cell_pos, grid_size, psize, vec3i(0, 0, 0), cell_shape_type)
                    if weight > Threshold:
                        if phase == PHASE_SOLID:
                            mass = weight * particle[np].ms
                            cell_velocity_s = particle[np].vs
                            if ti.static(use_affine != 0):
                                cell_velocity_s += particle[np].solid_velocity_gradient @ (cell_pos - position)
                            cell_solid_mass[cell] += mass
                            cell_porosity[cell] += (
                                weight * particle[np].vol * (1.0 - particle[np].porosity) / cell_measure
                            )
                            cell_solid_velocity[cell] += mass * cell_velocity_s
                        elif phase == PHASE_FLUID:
                            mass = weight * particle[np].mf
                            cell_velocity_f = particle[np].vf
                            if ti.static(use_affine != 0):
                                cell_velocity_f += particle[np].fluid_velocity_gradient @ (cell_pos - position)
                            cell_fluid_mass[cell] += mass
                            cell_fluid_velocity[cell] += mass * cell_velocity_f

            base_x = vec3i(
                _mac_base_1d(position[0], 1.0 / grid_size[0], 0.0, psize[0], mac_shape_type),
                _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], mac_shape_type),
                _mac_base_1d(position[2], 1.0 / grid_size[2], 0.5, psize[2], mac_shape_type),
            )
            for i, j, k in ti.ndrange(mac_influenced_node, mac_influenced_node, mac_influenced_node):
                face = base_x + vec3i(i, j, k)
                if (
                    0 <= face[0] < fluid_mass_x.shape[0]
                    and 0 <= face[1] < fluid_mass_x.shape[1]
                    and 0 <= face[2] < fluid_mass_x.shape[2]
                ):
                    face_pos = vec3f(
                        face[0] * grid_size[0], (face[1] + 0.5) * grid_size[1], (face[2] + 0.5) * grid_size[2]
                    )
                    btype = vec3i(_mac_bspline_boundary_type(face[0], fluid_mass_x.shape[0]), 0, 0)
                    weight = _mac_weight3d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                    if phase == PHASE_FLUID:
                        mass = weight * particle[np].mf
                        face_velocity_f = particle[np].vf
                        if ti.static(use_affine != 0):
                            face_velocity_f += particle[np].fluid_velocity_gradient @ (face_pos - position)
                        fluid_mass_x[face] += mass
                        fluid_velocity_x[face] += mass * face_velocity_f[0]
                    elif phase == PHASE_SOLID:
                        mass = weight * particle[np].ms
                        face_velocity_s = particle[np].vs
                        if ti.static(use_affine != 0):
                            face_velocity_s += particle[np].solid_velocity_gradient @ (face_pos - position)
                        solid_mass_x[face] += mass
                        solid_velocity_x[face] += mass * face_velocity_s[0]
                        face_porosity_x[face] += (
                            weight
                            * particle[np].vol
                            * (1.0 - particle[np].porosity)
                            / cell_measure
                            * _boundary_face_control_volume_scale(face[0], fluid_mass_x.shape[0])
                        )

            base_y = vec3i(
                _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], mac_shape_type),
                _mac_base_1d(position[1], 1.0 / grid_size[1], 0.0, psize[1], mac_shape_type),
                _mac_base_1d(position[2], 1.0 / grid_size[2], 0.5, psize[2], mac_shape_type),
            )
            for i, j, k in ti.ndrange(mac_influenced_node, mac_influenced_node, mac_influenced_node):
                face = base_y + vec3i(i, j, k)
                if (
                    0 <= face[0] < fluid_mass_y.shape[0]
                    and 0 <= face[1] < fluid_mass_y.shape[1]
                    and 0 <= face[2] < fluid_mass_y.shape[2]
                ):
                    face_pos = vec3f(
                        (face[0] + 0.5) * grid_size[0], face[1] * grid_size[1], (face[2] + 0.5) * grid_size[2]
                    )
                    btype = vec3i(0, _mac_bspline_boundary_type(face[1], fluid_mass_y.shape[1]), 0)
                    weight = _mac_weight3d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                    if phase == PHASE_FLUID:
                        mass = weight * particle[np].mf
                        face_velocity_f = particle[np].vf
                        if ti.static(use_affine != 0):
                            face_velocity_f += particle[np].fluid_velocity_gradient @ (face_pos - position)
                        fluid_mass_y[face] += mass
                        fluid_velocity_y[face] += mass * face_velocity_f[1]
                    elif phase == PHASE_SOLID:
                        mass = weight * particle[np].ms
                        face_velocity_s = particle[np].vs
                        if ti.static(use_affine != 0):
                            face_velocity_s += particle[np].solid_velocity_gradient @ (face_pos - position)
                        solid_mass_y[face] += mass
                        solid_velocity_y[face] += mass * face_velocity_s[1]
                        face_porosity_y[face] += (
                            weight
                            * particle[np].vol
                            * (1.0 - particle[np].porosity)
                            / cell_measure
                            * _boundary_face_control_volume_scale(face[1], fluid_mass_y.shape[1])
                        )

            base_z = vec3i(
                _mac_base_1d(position[0], 1.0 / grid_size[0], 0.5, psize[0], mac_shape_type),
                _mac_base_1d(position[1], 1.0 / grid_size[1], 0.5, psize[1], mac_shape_type),
                _mac_base_1d(position[2], 1.0 / grid_size[2], 0.0, psize[2], mac_shape_type),
            )
            for i, j, k in ti.ndrange(mac_influenced_node, mac_influenced_node, mac_influenced_node):
                face = base_z + vec3i(i, j, k)
                if (
                    0 <= face[0] < fluid_mass_z.shape[0]
                    and 0 <= face[1] < fluid_mass_z.shape[1]
                    and 0 <= face[2] < fluid_mass_z.shape[2]
                ):
                    face_pos = vec3f(
                        (face[0] + 0.5) * grid_size[0], (face[1] + 0.5) * grid_size[1], face[2] * grid_size[2]
                    )
                    btype = vec3i(0, 0, _mac_bspline_boundary_type(face[2], fluid_mass_z.shape[2]))
                    weight = _mac_weight3d(position, face_pos, grid_size, psize, btype, mac_shape_type)
                    if phase == PHASE_FLUID:
                        mass = weight * particle[np].mf
                        face_velocity_f = particle[np].vf
                        if ti.static(use_affine != 0):
                            face_velocity_f += particle[np].fluid_velocity_gradient @ (face_pos - position)
                        fluid_mass_z[face] += mass
                        fluid_velocity_z[face] += mass * face_velocity_f[2]
                    elif phase == PHASE_SOLID:
                        mass = weight * particle[np].ms
                        face_velocity_s = particle[np].vs
                        if ti.static(use_affine != 0):
                            face_velocity_s += particle[np].solid_velocity_gradient @ (face_pos - position)
                        solid_mass_z[face] += mass
                        solid_velocity_z[face] += mass * face_velocity_s[2]
                        face_porosity_z[face] += (
                            weight
                            * particle[np].vol
                            * (1.0 - particle[np].porosity)
                            / cell_measure
                            * _boundary_face_control_volume_scale(face[2], fluid_mass_z.shape[2])
                        )


@ti.kernel
def kernel_normalize_double_layer_mac_fields3d(
    cutoff: float,
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_mass_z: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
    fluid_velocity0_x: ti.template(),
    fluid_velocity0_y: ti.template(),
    fluid_velocity0_z: ti.template(),
    solid_mass_x: ti.template(),
    solid_mass_y: ti.template(),
    solid_mass_z: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    solid_velocity_z: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    face_porosity_z: ti.template(),
    cell_type: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_solid_mass: ti.template(),
    cell_porosity: ti.template(),
    cell_solid_velocity: ti.template(),
    cell_fluid_velocity: ti.template(),
):
    for I in ti.grouped(fluid_mass_x):
        if fluid_mass_x[I] > cutoff:
            fluid_velocity_x[I] /= fluid_mass_x[I]
        if solid_mass_x[I] > cutoff:
            solid_velocity_x[I] /= solid_mass_x[I]
        else:
            solid_velocity_x[I] = 0.0
        face_porosity_x[I] = _mapped_porosity(face_porosity_x[I])
        fluid_velocity0_x[I] = fluid_velocity_x[I]
    for I in ti.grouped(fluid_mass_y):
        if fluid_mass_y[I] > cutoff:
            fluid_velocity_y[I] /= fluid_mass_y[I]
        if solid_mass_y[I] > cutoff:
            solid_velocity_y[I] /= solid_mass_y[I]
        else:
            solid_velocity_y[I] = 0.0
        face_porosity_y[I] = _mapped_porosity(face_porosity_y[I])
        fluid_velocity0_y[I] = fluid_velocity_y[I]
    for I in ti.grouped(fluid_mass_z):
        if fluid_mass_z[I] > cutoff:
            fluid_velocity_z[I] /= fluid_mass_z[I]
        if solid_mass_z[I] > cutoff:
            solid_velocity_z[I] /= solid_mass_z[I]
        else:
            solid_velocity_z[I] = 0.0
        face_porosity_z[I] = _mapped_porosity(face_porosity_z[I])
        fluid_velocity0_z[I] = fluid_velocity_z[I]

    for I in ti.grouped(cell_type):
        if cell_fluid_mass[I] > cutoff:
            cell_type[I] = FLUID_CELL
            cell_fluid_velocity[I] /= cell_fluid_mass[I]
        if cell_solid_mass[I] > cutoff:
            cell_solid_velocity[I] /= cell_solid_mass[I]
        else:
            cell_solid_velocity[I] = ZEROVEC3f
        cell_porosity[I] = _mapped_porosity(cell_porosity[I])


@ti.kernel
def kernel_predict_double_layer3d(
    cutoff: float,
    gravity: vec3f,
    grid_size: vec3f,
    dt: ti.template(),
    damping: float,
    node: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_mass_z: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    fluid_acceleration_z: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    solid_velocity_z: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    face_porosity_z: ti.template(),
    face_solid_density_x: ti.template(),
    face_solid_density_y: ti.template(),
    face_solid_density_z: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    face_fluid_density_z: ti.template(),
    face_fluid_viscosity_x: ti.template(),
    face_fluid_viscosity_y: ti.template(),
    face_fluid_viscosity_z: ti.template(),
    face_grain_diameter_x: ti.template(),
    face_grain_diameter_y: ti.template(),
    face_grain_diameter_z: ti.template(),
    face_permeability_x: ti.template(),
    face_permeability_y: ti.template(),
    face_permeability_z: ti.template(),
    face_fluid_unit_weight_x: ti.template(),
    face_fluid_unit_weight_y: ti.template(),
    face_fluid_unit_weight_z: ti.template(),
    face_drag_model_x: ti.template(),
    face_drag_model_y: ti.template(),
    face_drag_model_z: ti.template(),
    node_solid_density: ti.template(),
    node_fluid_density: ti.template(),
    node_fluid_viscosity: ti.template(),
    node_grain_diameter: ti.template(),
    node_permeability: ti.template(),
    node_fluid_unit_weight: ti.template(),
    node_drag_model: ti.template(),
    shape_type: ti.template(),
    influenced_node: ti.template(),
    calLength: ti.template(),
    solid_pressure_factor: ti.template(),
    fluid_pressure_mobility: ti.template(),
    mixture_pressure_mobility: ti.template(),
    solid_face_mass: ti.template(),
):
    gnum = vec3i(fluid_velocity_x.shape[0], fluid_velocity_y.shape[1], fluid_velocity_z.shape[2])
    # Complete every viscous stencil before applying either phase update.
    for I in ti.grouped(fluid_velocity_x):
        if I[0] == 0 or I[0] == fluid_velocity_x.shape[0] - 1:
            fluid_velocity_x[I] = 0.0
    for I in ti.grouped(fluid_velocity_y):
        if I[1] == 0 or I[1] == fluid_velocity_y.shape[1] - 1:
            fluid_velocity_y[I] = 0.0
    for I in ti.grouped(fluid_velocity_z):
        if I[2] == 0 or I[2] == fluid_velocity_z.shape[2] - 1:
            fluid_velocity_z[I] = 0.0
    for I in ti.grouped(fluid_velocity_x):
        acc = 0.0
        if fluid_mass_x[I] > cutoff:
            nu = face_fluid_viscosity_x[I] / ti.max(face_fluid_density_x[I], 1.0e-12)
            acc = nu * _mac_laplacian3d(I, grid_size, fluid_velocity_x) - damping * fluid_velocity_x[I]
        fluid_acceleration_x[I] = acc
    for I in ti.grouped(fluid_velocity_y):
        acc = 0.0
        if fluid_mass_y[I] > cutoff:
            nu = face_fluid_viscosity_y[I] / ti.max(face_fluid_density_y[I], 1.0e-12)
            acc = nu * _mac_laplacian3d(I, grid_size, fluid_velocity_y) - damping * fluid_velocity_y[I]
        fluid_acceleration_y[I] = acc
    for I in ti.grouped(fluid_velocity_z):
        acc = 0.0
        if fluid_mass_z[I] > cutoff:
            nu = face_fluid_viscosity_z[I] / ti.max(face_fluid_density_z[I], 1.0e-12)
            acc = nu * _mac_laplacian3d(I, grid_size, fluid_velocity_z) - damping * fluid_velocity_z[I]
        fluid_acceleration_z[I] = acc
    for ng, nb in node:
        if ti.static(solid_pressure_factor != None):
            solid_pressure_factor[ng, nb] = 1.0
        if node[ng, nb].ms > cutoff:
            external_s = node[ng, nb].force / node[ng, nb].ms + vec3f(gravity[0], gravity[1], gravity[2])
            acc = external_s
            if node[ng, nb].mf > cutoff:
                position = vec3i(vectorize_id(ng, gnum)).cast(float) * grid_size
                external_f = _sample_mac_velocity3d(
                    position,
                    calLength[nb],
                    shape_type,
                    influenced_node,
                    grid_size,
                    fluid_acceleration_x,
                    fluid_acceleration_y,
                    fluid_acceleration_z,
                ) + vec3f(gravity[0], gravity[1], gravity[2])
                rel = node[ng, nb].momentumf - node[ng, nb].momentums
                a, b = _drag_rates_props(
                    rel,
                    _intrinsic_porosity_from_phase_masses(
                        node[ng, nb].ms, node[ng, nb].mf, node_solid_density[ng, nb], node_fluid_density[ng, nb]
                    ),
                    node_solid_density[ng, nb],
                    node_fluid_density[ng, nb],
                    node_fluid_viscosity[ng, nb],
                    node_grain_diameter[ng, nb],
                    node_permeability[ng, nb],
                    node_fluid_unit_weight[ng, nb],
                    node_drag_model[ng, nb],
                )
                acc, _ = _implicit_phase_accelerations(rel, external_s, external_f, a, b, dt[None])
                if ti.static(solid_pressure_factor != None):
                    mu_s, _ = _coupled_pressure_mobilities(
                        node_solid_density[ng, nb], node_fluid_density[ng, nb], a, b, dt[None]
                    )
                    solid_pressure_factor[ng, nb] = node_solid_density[ng, nb] * mu_s
            node[ng, nb].forces = acc
            node[ng, nb].momentums += dt[None] * acc
            node[ng, nb].momentum = node[ng, nb].momentums
        else:
            # G2P reads acceleration here, never the unscaled deposited force.
            node[ng, nb].forces = ZEROVEC3f
    for I in ti.grouped(fluid_velocity_x):
        acc = 0.0
        mu_s = 1.0 / ti.max(face_solid_density_x[I], 1.0e-12)
        mu_f = 1.0 / ti.max(face_fluid_density_x[I], 1.0e-12)
        if fluid_mass_x[I] > cutoff:
            position = vec3f(I[0] * grid_size[0], (I[1] + 0.5) * grid_size[1], (I[2] + 0.5) * grid_size[2])
            rel = _mac_relative_velocity_at_face(
                0,
                I,
                (fluid_velocity_x, fluid_velocity_y, fluid_velocity_z),
                (solid_velocity_x, solid_velocity_y, solid_velocity_z),
            )
            external_s, weight = _sample_solid_grid_velocity3d(
                position, grid_size, gnum, cutoff, shape_type, influenced_node, calLength, node, True
            )
            external_f = vec3f(fluid_acceleration_x[I], 0.0, 0.0)
            a, b = _drag_rates_props(
                rel,
                _intrinsic_porosity_from_phase_masses(
                    solid_face_mass[0][I], fluid_mass_x[I], face_solid_density_x[I], face_fluid_density_x[I]
                ),
                face_solid_density_x[I],
                face_fluid_density_x[I],
                face_fluid_viscosity_x[I],
                face_grain_diameter_x[I],
                face_permeability_x[I],
                face_fluid_unit_weight_x[I],
                face_drag_model_x[I],
            )
            _, coupled = _implicit_phase_accelerations(rel, external_s, external_f, a, b, dt[None])
            acc = gravity[0] + coupled[0]
            mu_s, mu_f = _coupled_pressure_mobilities(face_solid_density_x[I], face_fluid_density_x[I], a, b, dt[None])
        if ti.static(fluid_pressure_mobility != None):
            fluid_pressure_mobility[0][I] = mu_f
        if ti.static(mixture_pressure_mobility != None):
            phi = _clamp_porosity(face_porosity_x[I])
            mixture_pressure_mobility[0][I] = (1.0 - phi) * mu_s + phi * mu_f
        fluid_acceleration_x[I] = acc

    for I in ti.grouped(fluid_velocity_y):
        acc = 0.0
        mu_s = 1.0 / ti.max(face_solid_density_y[I], 1.0e-12)
        mu_f = 1.0 / ti.max(face_fluid_density_y[I], 1.0e-12)
        if fluid_mass_y[I] > cutoff:
            position = vec3f((I[0] + 0.5) * grid_size[0], I[1] * grid_size[1], (I[2] + 0.5) * grid_size[2])
            rel = _mac_relative_velocity_at_face(
                1,
                I,
                (fluid_velocity_x, fluid_velocity_y, fluid_velocity_z),
                (solid_velocity_x, solid_velocity_y, solid_velocity_z),
            )
            external_s, weight = _sample_solid_grid_velocity3d(
                position, grid_size, gnum, cutoff, shape_type, influenced_node, calLength, node, True
            )
            external_f = vec3f(0.0, fluid_acceleration_y[I], 0.0)
            a, b = _drag_rates_props(
                rel,
                _intrinsic_porosity_from_phase_masses(
                    solid_face_mass[1][I], fluid_mass_y[I], face_solid_density_y[I], face_fluid_density_y[I]
                ),
                face_solid_density_y[I],
                face_fluid_density_y[I],
                face_fluid_viscosity_y[I],
                face_grain_diameter_y[I],
                face_permeability_y[I],
                face_fluid_unit_weight_y[I],
                face_drag_model_y[I],
            )
            _, coupled = _implicit_phase_accelerations(rel, external_s, external_f, a, b, dt[None])
            acc = gravity[1] + coupled[1]
            mu_s, mu_f = _coupled_pressure_mobilities(face_solid_density_y[I], face_fluid_density_y[I], a, b, dt[None])
        if ti.static(fluid_pressure_mobility != None):
            fluid_pressure_mobility[1][I] = mu_f
        if ti.static(mixture_pressure_mobility != None):
            phi = _clamp_porosity(face_porosity_y[I])
            mixture_pressure_mobility[1][I] = (1.0 - phi) * mu_s + phi * mu_f
        fluid_acceleration_y[I] = acc

    for I in ti.grouped(fluid_velocity_z):
        acc = 0.0
        mu_s = 1.0 / ti.max(face_solid_density_z[I], 1.0e-12)
        mu_f = 1.0 / ti.max(face_fluid_density_z[I], 1.0e-12)
        if fluid_mass_z[I] > cutoff:
            position = vec3f((I[0] + 0.5) * grid_size[0], (I[1] + 0.5) * grid_size[1], I[2] * grid_size[2])
            rel = _mac_relative_velocity_at_face(
                2,
                I,
                (fluid_velocity_x, fluid_velocity_y, fluid_velocity_z),
                (solid_velocity_x, solid_velocity_y, solid_velocity_z),
            )
            external_s, weight = _sample_solid_grid_velocity3d(
                position, grid_size, gnum, cutoff, shape_type, influenced_node, calLength, node, True
            )
            external_f = vec3f(0.0, 0.0, fluid_acceleration_z[I])
            a, b = _drag_rates_props(
                rel,
                _intrinsic_porosity_from_phase_masses(
                    solid_face_mass[2][I], fluid_mass_z[I], face_solid_density_z[I], face_fluid_density_z[I]
                ),
                face_solid_density_z[I],
                face_fluid_density_z[I],
                face_fluid_viscosity_z[I],
                face_grain_diameter_z[I],
                face_permeability_z[I],
                face_fluid_unit_weight_z[I],
                face_drag_model_z[I],
            )
            _, coupled = _implicit_phase_accelerations(rel, external_s, external_f, a, b, dt[None])
            acc = gravity[2] + coupled[2]
            mu_s, mu_f = _coupled_pressure_mobilities(face_solid_density_z[I], face_fluid_density_z[I], a, b, dt[None])
        if ti.static(fluid_pressure_mobility != None):
            fluid_pressure_mobility[2][I] = mu_f
        if ti.static(mixture_pressure_mobility != None):
            phi = _clamp_porosity(face_porosity_z[I])
            mixture_pressure_mobility[2][I] = (1.0 - phi) * mu_s + phi * mu_f
        fluid_acceleration_z[I] = acc

    # Nonlinear drag samples every velocity component at the same time level.
    for I in ti.grouped(fluid_velocity_x):
        fluid_velocity_x[I] += dt[None] * fluid_acceleration_x[I]
        if I[0] == 0 or I[0] == fluid_velocity_x.shape[0] - 1:
            fluid_velocity_x[I] = 0.0
            fluid_acceleration_x[I] = 0.0
    for I in ti.grouped(fluid_velocity_y):
        fluid_velocity_y[I] += dt[None] * fluid_acceleration_y[I]
        if I[1] == 0 or I[1] == fluid_velocity_y.shape[1] - 1:
            fluid_velocity_y[I] = 0.0
            fluid_acceleration_y[I] = 0.0
    for I in ti.grouped(fluid_velocity_z):
        fluid_velocity_z[I] += dt[None] * fluid_acceleration_z[I]
        if I[2] == 0 or I[2] == fluid_velocity_z.shape[2] - 1:
            fluid_velocity_z[I] = 0.0
            fluid_acceleration_z[I] = 0.0


@ti.kernel
def kernel_project_solid_grid_velocity_to_mac3d(
    cutoff: float,
    grid_size: vec3f,
    gnum: ti.types.vector(3, int),
    shape_type: ti.template(),
    influenced_node: ti.template(),
    node: ti.template(),
    calLength: ti.template(),
    cell_type: ti.template(),
    solid_mass_x: ti.template(),
    solid_mass_y: ti.template(),
    solid_mass_z: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    solid_velocity_z: ti.template(),
):
    # Domain edges and SolidCell faces share the fluid no-penetration condition.
    # Moving-wall callbacks run after this projection and may overwrite it.
    for I in ti.grouped(solid_velocity_x):
        static_wall = I[0] == 0 or I[0] == solid_velocity_x.shape[0] - 1
        if 0 < I[0] < solid_velocity_x.shape[0] - 1:
            left = I - vec3i(1, 0, 0)
            right = I
            static_wall = (cell_type[left] == SOLID_CELL and cell_type[right] == FLUID_CELL) or (
                cell_type[left] == FLUID_CELL and cell_type[right] == SOLID_CELL
            )
        if static_wall:
            solid_velocity_x[I] = 0.0
        elif solid_mass_x[I] > cutoff:
            face_pos = vec3f(I[0] * grid_size[0], (I[1] + 0.5) * grid_size[1], (I[2] + 0.5) * grid_size[2])
            velocity, weight_sum = _sample_solid_grid_velocity3d(
                face_pos,
                grid_size,
                gnum,
                cutoff,
                shape_type,
                influenced_node,
                calLength,
                node,
            )
            if weight_sum > cutoff:
                solid_velocity_x[I] = velocity[0]

    for I in ti.grouped(solid_velocity_y):
        static_wall = I[1] == 0 or I[1] == solid_velocity_y.shape[1] - 1
        if 0 < I[1] < solid_velocity_y.shape[1] - 1:
            down = I - vec3i(0, 1, 0)
            up = I
            static_wall = (cell_type[down] == SOLID_CELL and cell_type[up] == FLUID_CELL) or (
                cell_type[down] == FLUID_CELL and cell_type[up] == SOLID_CELL
            )
        if static_wall:
            solid_velocity_y[I] = 0.0
        elif solid_mass_y[I] > cutoff:
            face_pos = vec3f((I[0] + 0.5) * grid_size[0], I[1] * grid_size[1], (I[2] + 0.5) * grid_size[2])
            velocity, weight_sum = _sample_solid_grid_velocity3d(
                face_pos,
                grid_size,
                gnum,
                cutoff,
                shape_type,
                influenced_node,
                calLength,
                node,
            )
            if weight_sum > cutoff:
                solid_velocity_y[I] = velocity[1]

    for I in ti.grouped(solid_velocity_z):
        static_wall = I[2] == 0 or I[2] == solid_velocity_z.shape[2] - 1
        if 0 < I[2] < solid_velocity_z.shape[2] - 1:
            back = I - vec3i(0, 0, 1)
            front = I
            static_wall = (cell_type[back] == SOLID_CELL and cell_type[front] == FLUID_CELL) or (
                cell_type[back] == FLUID_CELL and cell_type[front] == SOLID_CELL
            )
        if static_wall:
            solid_velocity_z[I] = 0.0
        elif solid_mass_z[I] > cutoff:
            face_pos = vec3f((I[0] + 0.5) * grid_size[0], (I[1] + 0.5) * grid_size[1], I[2] * grid_size[2])
            velocity, weight_sum = _sample_solid_grid_velocity3d(
                face_pos,
                grid_size,
                gnum,
                cutoff,
                shape_type,
                influenced_node,
                calLength,
                node,
            )
            if weight_sum > cutoff:
                solid_velocity_z[I] = velocity[2]


@ti.kernel
def kernel_assemble_double_layer_pressure_rhs3d(
    grid_size: vec3f,
    cell_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    face_porosity_z: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    solid_velocity_z: ti.template(),
    b: ti.template(),
):
    for I in ti.grouped(cell_type):
        b[I] = 0.0
        if cell_type[I] == FLUID_CELL:
            # A shared face flux preserves the closed-domain compatibility condition.
            divergence = 0.0
            right = I + vec3i(1, 0, 0)
            flux_right = (
                face_porosity_x[right] * fluid_velocity_x[right]
                + (1.0 - face_porosity_x[right]) * solid_velocity_x[right]
            )
            flux_left = face_porosity_x[I] * fluid_velocity_x[I] + (1.0 - face_porosity_x[I]) * solid_velocity_x[I]
            divergence += (flux_right - flux_left) / grid_size[0]
            right = I + vec3i(0, 1, 0)
            flux_right = (
                face_porosity_y[right] * fluid_velocity_y[right]
                + (1.0 - face_porosity_y[right]) * solid_velocity_y[right]
            )
            flux_left = face_porosity_y[I] * fluid_velocity_y[I] + (1.0 - face_porosity_y[I]) * solid_velocity_y[I]
            divergence += (flux_right - flux_left) / grid_size[1]
            right = I + vec3i(0, 0, 1)
            flux_right = (
                face_porosity_z[right] * fluid_velocity_z[right]
                + (1.0 - face_porosity_z[right]) * solid_velocity_z[right]
            )
            flux_left = face_porosity_z[I] * fluid_velocity_z[I] + (1.0 - face_porosity_z[I]) * solid_velocity_z[I]
            divergence += (flux_right - flux_left) / grid_size[2]
            b[I] = -divergence


@ti.kernel
def kernel_assemble_double_layer_pressure_A3d(
    grid_size: vec3f,
    dt: ti.template(),
    grid_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    face_porosity_z: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    fluid_sdf: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
    coupled_mobility: ti.template(),
):
    for I in ti.grouped(grid_type):
        Adiag[I] = 0.0
        Ax[I] = ZEROVEC3f
        if grid_type[I] == FLUID_CELL:
            # Keep a symmetric face-averaged mobility so PCG/MGPCG remains applicable.
            mobility = _cell_mobility3d(
                I,
                face_porosity_x,
                face_porosity_y,
                face_porosity_z,
                cell_solid_density,
                cell_fluid_density,
                coupled_mobility,
            )
            for d in ti.static(range(3)):
                scale = dt[None] * mobility / (grid_size[d] * grid_size[d])
                right = I + ti.Vector.unit(3, d)
                if right[d] < grid_type.shape[d]:
                    if grid_type[right] == FLUID_CELL:
                        mobility_right = _cell_mobility3d(
                            right,
                            face_porosity_x,
                            face_porosity_y,
                            face_porosity_z,
                            cell_solid_density,
                            cell_fluid_density,
                            coupled_mobility,
                        )
                        face_scale = 0.5 * (scale + dt[None] * mobility_right / (grid_size[d] * grid_size[d]))
                        Ax[I][d] = -face_scale
                        Adiag[I] += face_scale
                    elif grid_type[right] == AIR_CELL:
                        theta = _pressure_free_surface_theta(I, right, fluid_sdf)
                        Adiag[I] += scale / theta
                left = I - ti.Vector.unit(3, d)
                if left[d] >= 0:
                    if grid_type[left] == FLUID_CELL:
                        mobility_left = _cell_mobility3d(
                            left,
                            face_porosity_x,
                            face_porosity_y,
                            face_porosity_z,
                            cell_solid_density,
                            cell_fluid_density,
                            coupled_mobility,
                        )
                        Adiag[I] += 0.5 * (scale + dt[None] * mobility_left / (grid_size[d] * grid_size[d]))
                    elif grid_type[left] == AIR_CELL:
                        theta = _pressure_free_surface_theta(I, left, fluid_sdf)
                        Adiag[I] += scale / theta
            if Adiag[I] <= 1.0e-12:
                Adiag[I] = 1.0
            else:
                Adiag[I] += 1.0e-12


@ti.kernel
def kernel_coarsen_double_layer_grid_type3d(fine_grid_type: ti.template(), coarse_grid_type: ti.template()):
    for I in ti.grouped(coarse_grid_type):
        base = I * 2
        has_fluid = 0
        has_air = 0
        for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
            fine = base + offset
            if (
                fine[0] < fine_grid_type.shape[0]
                and fine[1] < fine_grid_type.shape[1]
                and fine[2] < fine_grid_type.shape[2]
            ):
                attr = int(fine_grid_type[fine])
                if attr == FLUID_CELL:
                    has_fluid = 1
                elif attr == AIR_CELL:
                    has_air = 1
        if has_fluid:
            coarse_grid_type[I] = FLUID_CELL
        elif has_air:
            coarse_grid_type[I] = AIR_CELL
        else:
            coarse_grid_type[I] = SOLID_CELL


@ti.kernel
def kernel_assemble_double_layer_pressure_mg_A3d(
    grid_size: vec3f,
    dt: ti.template(),
    factor: int,
    grid_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    face_porosity_z: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    fluid_sdf: ti.template(),
    Adiag: ti.template(),
    Ax: ti.template(),
    coupled_mobility: ti.template(),
):
    coarse_grid_size = grid_size * factor
    for I in ti.grouped(grid_type):
        Adiag[I] = 0.0
        Ax[I] = ZEROVEC3f
        if grid_type[I] == FLUID_CELL:
            mobility = _coarse_mobility3d(
                I,
                factor,
                face_porosity_x,
                face_porosity_y,
                face_porosity_z,
                cell_solid_density,
                cell_fluid_density,
                coupled_mobility,
            )
            for d in ti.static(range(3)):
                scale = dt[None] * mobility / (coarse_grid_size[d] * coarse_grid_size[d])
                right = I + ti.Vector.unit(3, d)
                if right[d] < grid_type.shape[d]:
                    if grid_type[right] == FLUID_CELL:
                        mobility_right = _coarse_mobility3d(
                            right,
                            factor,
                            face_porosity_x,
                            face_porosity_y,
                            face_porosity_z,
                            cell_solid_density,
                            cell_fluid_density,
                            coupled_mobility,
                        )
                        face_scale = 0.5 * (
                            scale + dt[None] * mobility_right / (coarse_grid_size[d] * coarse_grid_size[d])
                        )
                        Ax[I][d] = -face_scale
                        Adiag[I] += face_scale
                    elif grid_type[right] == AIR_CELL:
                        theta = _pressure_free_surface_theta_coarse3d(I, right, factor, fluid_sdf)
                        Adiag[I] += scale / theta
                left = I - ti.Vector.unit(3, d)
                if left[d] >= 0:
                    if grid_type[left] == FLUID_CELL:
                        mobility_left = _coarse_mobility3d(
                            left,
                            factor,
                            face_porosity_x,
                            face_porosity_y,
                            face_porosity_z,
                            cell_solid_density,
                            cell_fluid_density,
                            coupled_mobility,
                        )
                        Adiag[I] += 0.5 * (
                            scale + dt[None] * mobility_left / (coarse_grid_size[d] * coarse_grid_size[d])
                        )
                    elif grid_type[left] == AIR_CELL:
                        theta = _pressure_free_surface_theta_coarse3d(I, left, factor, fluid_sdf)
                        Adiag[I] += scale / theta
            if Adiag[I] <= 1.0e-12:
                Adiag[I] = 1.0
            else:
                Adiag[I] += 1.0e-12


@ti.kernel
def kernel_correct_double_layer_velocity3d(
    cutoff: float,
    grid_size: vec3f,
    dt: ti.template(),
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_mass_z: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    fluid_acceleration_z: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    face_fluid_density_z: ti.template(),
    pressure_mobility: ti.template(),
):
    for I in ti.grouped(fluid_velocity_x):
        if fluid_mass_x[I] > cutoff:
            if I[0] == 0 or I[0] == fluid_velocity_x.shape[0] - 1:
                fluid_velocity_x[I] = 0.0
                fluid_acceleration_x[I] = 0.0
            else:
                left = I - vec3i(1, 0, 0)
                right = I
                gradp = 0.0
                if cell_type[left] == FLUID_CELL and cell_type[right] == FLUID_CELL:
                    gradp = (cell_pressure[right] - cell_pressure[left]) / grid_size[0]
                elif cell_type[left] == FLUID_CELL and cell_type[right] == AIR_CELL:
                    theta = _pressure_free_surface_theta(left, right, fluid_sdf)
                    gradp = (0.0 - cell_pressure[left]) / (theta * grid_size[0])
                elif cell_type[left] == AIR_CELL and cell_type[right] == FLUID_CELL:
                    theta = _pressure_free_surface_theta(right, left, fluid_sdf)
                    gradp = (cell_pressure[right] - 0.0) / (theta * grid_size[0])
                acc = -gradp / ti.max(face_fluid_density_x[I], 1.0e-12)
                if ti.static(pressure_mobility != None):
                    acc = -gradp * pressure_mobility[0][I]
                fluid_velocity_x[I] += dt[None] * acc
                fluid_acceleration_x[I] += acc
    for I in ti.grouped(fluid_velocity_y):
        if fluid_mass_y[I] > cutoff:
            if I[1] == 0 or I[1] == fluid_velocity_y.shape[1] - 1:
                fluid_velocity_y[I] = 0.0
                fluid_acceleration_y[I] = 0.0
            else:
                down = I - vec3i(0, 1, 0)
                up = I
                gradp = 0.0
                if cell_type[down] == FLUID_CELL and cell_type[up] == FLUID_CELL:
                    gradp = (cell_pressure[up] - cell_pressure[down]) / grid_size[1]
                elif cell_type[down] == FLUID_CELL and cell_type[up] == AIR_CELL:
                    theta = _pressure_free_surface_theta(down, up, fluid_sdf)
                    gradp = (0.0 - cell_pressure[down]) / (theta * grid_size[1])
                elif cell_type[down] == AIR_CELL and cell_type[up] == FLUID_CELL:
                    theta = _pressure_free_surface_theta(up, down, fluid_sdf)
                    gradp = (cell_pressure[up] - 0.0) / (theta * grid_size[1])
                acc = -gradp / ti.max(face_fluid_density_y[I], 1.0e-12)
                if ti.static(pressure_mobility != None):
                    acc = -gradp * pressure_mobility[1][I]
                fluid_velocity_y[I] += dt[None] * acc
                fluid_acceleration_y[I] += acc
    for I in ti.grouped(fluid_velocity_z):
        if fluid_mass_z[I] > cutoff:
            if I[2] == 0 or I[2] == fluid_velocity_z.shape[2] - 1:
                fluid_velocity_z[I] = 0.0
                fluid_acceleration_z[I] = 0.0
            else:
                back = I - vec3i(0, 0, 1)
                front = I
                gradp = 0.0
                if cell_type[back] == FLUID_CELL and cell_type[front] == FLUID_CELL:
                    gradp = (cell_pressure[front] - cell_pressure[back]) / grid_size[2]
                elif cell_type[back] == FLUID_CELL and cell_type[front] == AIR_CELL:
                    theta = _pressure_free_surface_theta(back, front, fluid_sdf)
                    gradp = (0.0 - cell_pressure[back]) / (theta * grid_size[2])
                elif cell_type[back] == AIR_CELL and cell_type[front] == FLUID_CELL:
                    theta = _pressure_free_surface_theta(front, back, fluid_sdf)
                    gradp = (cell_pressure[front] - 0.0) / (theta * grid_size[2])
                acc = -gradp / ti.max(face_fluid_density_z[I], 1.0e-12)
                if ti.static(pressure_mobility != None):
                    acc = -gradp * pressure_mobility[2][I]
                fluid_velocity_z[I] += dt[None] * acc
                fluid_acceleration_z[I] += acc


@ti.kernel
def kernel_project_double_layer_pressure_to_solid_nodes3d(
    cutoff: float,
    gnum: ti.types.vector(3, int),
    grid_size: vec3f,
    mac_shape_type: ti.template(),
    mac_influenced_node: ti.template(),
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    node: ti.template(),
    calLength: ti.template(),
):
    for ng, nb in node:
        node[ng, nb].pressure = 0.0
        if node[ng, nb].ms > cutoff:
            grid_id = vec3i(vectorize_id(ng, gnum))
            position = grid_id.cast(float) * grid_size
            psize = calLength[nb]
            node[ng, nb].pressure = _sample_cell_pressure_gfm_shape3d(
                position,
                psize,
                mac_shape_type,
                mac_influenced_node,
                grid_size,
                cell_type,
                cell_pressure,
                fluid_sdf,
                False,
            )


@ti.kernel
def kernel_sample_double_layer_solid_pressure_from_nodes3d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    cutoff: float,
    particle: ti.template(),
    material_mapping: ti.template(),
    node: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_SOLID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            pressure = 0.0
            weight = 0.0
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape = shapefn[ln]
                if node[nodeID, bodyID].ms > cutoff:
                    pressure += shape * node[nodeID, bodyID].pressure
                    weight += shape
            if weight > Threshold:
                particle[np].pressure = pressure / weight
            else:
                particle[np].pressure = 0.0


@ti.kernel
def kernel_correct_double_layer_solid_velocity_paper3d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    cutoff: float,
    dt: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    pressure_factor: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_SOLID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            pressure_gradient = ZEROVEC3f
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                pressure_gradient += node[nodeID, bodyID].pressure * dshapefn[ln]
            pressure_force = -particle[np].vol * (1.0 - particle[np].porosity) * pressure_gradient
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                if node[nodeID, bodyID].ms > cutoff:
                    acc = shapefn[ln] * pressure_force / ti.max(node[nodeID, bodyID].ms, 1.0e-12)
                    if ti.static(pressure_factor != None):
                        acc *= pressure_factor[nodeID, bodyID]
                    node[nodeID, bodyID].momentums += dt[None] * acc
                    node[nodeID, bodyID].momentum = node[nodeID, bodyID].momentums
                    node[nodeID, bodyID].forces += acc


@ti.kernel
def kernel_advect_double_layer_fluid_particles3d(
    particle_count: int, domain: vec3f, dt: ti.template(), particle: ti.template()
):
    for np in range(particle_count):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
        ):
            flag_fix = ti.cast(particle[np].fix_v, float)
            flag_free = vec3f(1.0, 1.0, 1.0) - flag_fix
            particle[np].x += dt[None] * (particle[np].vf * flag_free + particle[np].v * flag_fix)
            for d in ti.static(range(3)):
                particle[np].x[d] = ti.min(ti.max(particle[np].x[d], 1.0e-8), domain[d] - 1.0e-8)


@ti.kernel
def kernel_update_double_layer_fluid_volume3d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    initialize_mass: int,
    fluid_density: float,
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
    phase_volume_support: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            porosity = 0.0
            weight = 0.0
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                shape = shapefn[ln]
                porosity += shape * _intrinsic_pore_porosity(nodeID, bodyID, node, phase_volume_support)
                weight += shape
            if weight > Threshold:
                porosity = _clamp_porosity(porosity / weight)
                if initialize_mass != 0:
                    particle[np].mf = fluid_density * porosity * particle[np].vol
                    particle[np].m = particle[np].mf
                else:
                    particle[np].vol = particle[np].mf / ti.max(fluid_density * porosity, 1.0e-12)
                particle[np].porosity = porosity
                particle[np].rad = 0.5 * ti.pow(particle[np].vol, 1.0 / 3.0)


@ti.kernel
def kernel_volume_p2g_double_layer_fluid3d(
    total_nodes: int,
    particle_count: int,
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    node_size: ti.template(),
):
    for np in range(particle_count):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                node[nodeID, bodyID].vol += shapefn[ln] * particle[np].vol * particle[np].porosity


@ti.kernel
def kernel_delta_correct_double_layer_fluid3d(
    total_nodes: int,
    particle_count: int,
    domain: vec3f,
    grid_size: vec3f,
    cnum: ti.types.vector(3, int),
    shifting_scale: float,
    cell_type: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    reference_volume: ti.template(),
    volume_error: ti.template(),
):
    min_dx = ti.min(grid_size[0], ti.min(grid_size[1], grid_size[2]))
    gradient_scale = (grid_size[0] * grid_size[1] * grid_size[2]) ** 2 / min_dx
    for ng, nb in node:
        error = _double_layer_shifting_volume_error(ng, nb, cnum, cell_type, node, reference_volume)
        volume_error[ng, nb] = error

    denominator = 0.0
    for np in range(particle_count):
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
            and _double_layer_shifting_is_interior3d(particle[np].x, grid_size, cnum, cell_type)
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            gradient = ZEROVEC3f
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                error = volume_error[nodeID, bodyID]
                gradient += dshapefn[ln] * error
            gradient *= 2.0 * particle[np].vol * particle[np].porosity
            particle[np].grad_E2 = gradient
            denominator += gradient.dot(gradient)

    # Reuse residual storage for J grad(E), then take the quadratic Cauchy step.
    for ng, nb in node:
        volume_error[ng, nb] = 0.0
    for np in range(particle_count):
        if (
            denominator > Threshold * gradient_scale * gradient_scale
            and int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
            and _double_layer_shifting_is_interior3d(particle[np].x, grid_size, cnum, cell_type)
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for ln in range(offset, offset + int(node_size[np])):
                nodeID = LnID[ln]
                volume_error[nodeID, bodyID] += (
                    particle[np].vol * particle[np].porosity * dshapefn[ln].dot(particle[np].grad_E2)
                )
    curvature = 0.0
    for ng, nb in node:
        curvature += volume_error[ng, nb] * volume_error[ng, nb]
    step = 0.0
    if denominator > Threshold * gradient_scale * gradient_scale and curvature > 0.0:
        step = shifting_scale * denominator / (2.0 * curvature)
    for np in range(particle_count):
        if (
            step > 0.0
            and int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == PHASE_FLUID
            and _double_layer_shifting_is_interior3d(particle[np].x, grid_size, cnum, cell_type)
        ):
            shift = -step * particle[np].grad_E2
            max_shift = 0.05 * min_dx
            shift_norm = shift.norm()
            if shift_norm > max_shift:
                shift *= max_shift / shift_norm
            for d in ti.static(range(3)):
                particle[np].x[d] += shift[d]
                particle[np].x[d] = ti.min(ti.max(particle[np].x[d], 1.0e-8), domain[d] - 1.0e-8)


@ti.kernel
def _kernel_g2p_double_layer3d(
    phase_filter: ti.template(),
    total_nodes: int,
    start_index: int,
    end_index: int,
    alpha: float,
    domain: vec3f,
    grid_size: vec3f,
    mac_shape_type: ti.template(),
    mac_influenced_node: ti.template(),
    use_affine: ti.template(),
    dt: ti.template(),
    matProps: ti.template(),
    stateVars: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    calLength: ti.template(),
    LnID: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    fluid_velocity_z: ti.template(),
    fluid_acceleration_x: ti.template(),
    fluid_acceleration_y: ti.template(),
    fluid_acceleration_z: ti.template(),
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    preserve_solid_pressure: ti.template(),
    update_solid_state: ti.template(),
    delayed_fluid_advection: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if (
            int(particle[np].active) == 1
            and int(particle[np].materialID) > 0
            and int(particle[np].phase) == phase_filter
        ):
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            position = particle[np].x
            if ti.static(phase_filter == PHASE_SOLID):
                v_pic = ZEROVEC3f
                a_pic = ZEROVEC3f
                gradv = ZEROMAT3x3
                for ln in range(offset, offset + int(node_size[np])):
                    nodeID = LnID[ln]
                    shape = shapefn[ln]
                    dshape = dshapefn[ln]
                    velocity = node[nodeID, bodyID].momentums
                    acceleration = node[nodeID, bodyID].forces
                    v_pic += shape * velocity
                    a_pic += shape * acceleration
                    gradv += velocity.outer_product(dshape)
                if ti.static(use_affine != 0 and mac_shape_type != MAC_SHAPE_LINEAR):
                    Dp = ZEROMAT3x3
                    Bp = ZEROMAT3x3
                    for ln in range(offset, offset + int(node_size[np])):
                        nodeID = LnID[ln]
                        shape = shapefn[ln]
                        gnum = vec3i(fluid_velocity_x.shape[0], fluid_velocity_y.shape[1], fluid_velocity_z.shape[2])
                        nodal_coord = grid_size * vec3f(vectorize_id(nodeID, gnum))
                        pointer = nodal_coord - position
                        velocity = node[nodeID, bodyID].momentums
                        Dp += shape * pointer.outer_product(pointer)
                        Bp += shape * (velocity - v_pic).outer_product(pointer)
                    dp_trace = ti.max(Dp[0, 0] + Dp[1, 1] + Dp[2, 2], 0.0)
                    dp_det = ti.abs(Dp.determinant())
                    if dp_trace > 1.0e-20 and dp_det > 1.0e-8 * dp_trace * dp_trace * dp_trace:
                        gradv = Bp @ Dp.inverse()
                flag_fix = ti.cast(particle[np].fix_v, float)
                flag_free = vec3f(1.0, 1.0, 1.0) - flag_fix
                if particle[np].fix_v[0] != 0 and particle[np].fix_v[1] != 0 and particle[np].fix_v[2] != 0:
                    gradv = ZEROMAT3x3
                v_flip = particle[np].vs + dt[None] * a_pic
                new_v = (alpha * v_pic + (1.0 - alpha) * v_flip) * flag_free + particle[np].vs * flag_fix
                particle[np].vs = new_v
                particle[np].v = new_v
                particle[np].solid_velocity_gradient = gradv
                if ti.static(update_solid_state != 0):
                    volume_ratio = matProps.update_particle_volume(np, gradv, stateVars, dt)
                    porosity = matProps.update_particle_porosity(gradv, particle[np].porosity, dt)
                    particle[np].vol *= volume_ratio
                    particle[np].porosity = porosity
                    if porosity > matProps.maximum_porosity:
                        particle[np].stress *= 0.0
                    else:
                        particle[np].stress = matProps.ComputeStress(np, particle[np].stress, gradv, stateVars, dt)
                if ti.static(preserve_solid_pressure == 0):
                    psize = calLength[bodyID]
                    particle[np].pressure = _sample_cell_pressure_gfm_shape3d(
                        position,
                        psize,
                        mac_shape_type,
                        mac_influenced_node,
                        grid_size,
                        cell_type,
                        cell_pressure,
                        fluid_sdf,
                    )
                displacement = dt[None] * (v_pic * flag_free + particle[np].vs * flag_fix)
                particle[np].x += displacement
                if int(particle[np].coupling) == 1:
                    particle[np].verletDisp += displacement
            elif ti.static(phase_filter == PHASE_FLUID):
                psize = calLength[bodyID]
                v_pic = _sample_mac_velocity3d(
                    position,
                    psize,
                    mac_shape_type,
                    mac_influenced_node,
                    grid_size,
                    fluid_velocity_x,
                    fluid_velocity_y,
                    fluid_velocity_z,
                )
                a_pic = _sample_mac_velocity3d(
                    position,
                    psize,
                    mac_shape_type,
                    mac_influenced_node,
                    grid_size,
                    fluid_acceleration_x,
                    fluid_acceleration_y,
                    fluid_acceleration_z,
                )
                if ti.static(use_affine != 0):
                    particle[np].fluid_velocity_gradient = _sample_mac_velocity_gradient3d(
                        position,
                        psize,
                        mac_shape_type,
                        mac_influenced_node,
                        grid_size,
                        fluid_velocity_x,
                        fluid_velocity_y,
                        fluid_velocity_z,
                        v_pic,
                    )
                flag_fix = ti.cast(particle[np].fix_v, float)
                flag_free = vec3f(1.0, 1.0, 1.0) - flag_fix
                v_flip = particle[np].vf + dt[None] * a_pic
                new_v = (alpha * v_pic + (1.0 - alpha) * v_flip) * flag_free + particle[np].vf * flag_fix
                particle[np].vf = new_v
                particle[np].v = new_v
                particle[np].pressure = _sample_cell_pressure_gfm_shape3d(
                    position,
                    psize,
                    mac_shape_type,
                    mac_influenced_node,
                    grid_size,
                    cell_type,
                    cell_pressure,
                    fluid_sdf,
                )
                if ti.static(delayed_fluid_advection == 0):
                    particle[np].x += dt[None] * (v_pic * flag_free + particle[np].vf * flag_fix)

            for d in ti.static(range(3)):
                particle[np].x[d] = ti.min(ti.max(particle[np].x[d], 1.0e-8), domain[d] - 1.0e-8)


def kernel_g2p_double_layer3d(*args, **kwargs):
    # Independent phases keep compiler control-flow analysis and GPU register use bounded.
    _kernel_g2p_double_layer3d(PHASE_SOLID, *args, **kwargs)
    _kernel_g2p_double_layer3d(PHASE_FLUID, *args, **kwargs)


@ti.kernel
def kernel_remap_double_layer_shift_velocity2d(
    particle_count: int,
    capture: ti.template(),
    shape_type: ti.template(),
    influenced_node: ti.template(),
    use_affine: ti.template(),
    grid_size: vec2f,
    particle: ti.template(),
    lengths: ti.template(),
    velocity: ti.template(),
    residual: ti.template(),
):
    for p in range(particle_count):
        if int(particle[p].active) == 1 and int(particle[p].materialID) > 0 and int(particle[p].phase) == PHASE_FLUID:
            position = particle[p].x
            psize = lengths[int(particle[p].bodyID)]
            pic = _sample_mac_velocity(
                position, psize, shape_type, influenced_node, grid_size, velocity[0], velocity[1]
            )
            if ti.static(capture != 0):
                residual[p] = particle[p].vf - pic
            else:
                # Preserve the FLIP residual while evaluating the same grid field at the new coordinate.
                fixed = ti.cast(particle[p].fix_v, float)
                free = vec2f(1.0, 1.0) - fixed
                mapped = (pic + residual[p]) * free + particle[p].vf * fixed
                particle[p].vf = mapped
                particle[p].v = mapped
                if ti.static(use_affine != 0):
                    particle[p].fluid_velocity_gradient = _sample_mac_velocity_gradient(
                        position, psize, shape_type, influenced_node, grid_size, velocity[0], velocity[1], pic
                    )


@ti.kernel
def kernel_remap_double_layer_shift_velocity3d(
    particle_count: int,
    capture: ti.template(),
    shape_type: ti.template(),
    influenced_node: ti.template(),
    use_affine: ti.template(),
    grid_size: vec3f,
    particle: ti.template(),
    lengths: ti.template(),
    velocity: ti.template(),
    residual: ti.template(),
):
    for p in range(particle_count):
        if int(particle[p].active) == 1 and int(particle[p].materialID) > 0 and int(particle[p].phase) == PHASE_FLUID:
            position = particle[p].x
            psize = lengths[int(particle[p].bodyID)]
            pic = _sample_mac_velocity3d(
                position, psize, shape_type, influenced_node, grid_size, velocity[0], velocity[1], velocity[2]
            )
            if ti.static(capture != 0):
                residual[p] = particle[p].vf - pic
            else:
                # Preserve the FLIP residual while evaluating the same grid field at the new coordinate.
                fixed = ti.cast(particle[p].fix_v, float)
                free = vec3f(1.0, 1.0, 1.0) - fixed
                mapped = (pic + residual[p]) * free + particle[p].vf * fixed
                particle[p].vf = mapped
                particle[p].v = mapped
                if ti.static(use_affine != 0):
                    particle[p].fluid_velocity_gradient = _sample_mac_velocity_gradient3d(
                        position,
                        psize,
                        shape_type,
                        influenced_node,
                        grid_size,
                        velocity[0],
                        velocity[1],
                        velocity[2],
                        pic,
                    )


@ti.kernel
def kernel_capture_double_layer_solid_boundary(
    cell_type: ti.template(),
    velocity: ti.template(),
    saved: ti.template(),
    mask: ti.template(),
):
    for axis in ti.static(range(len(velocity))):
        for I in ti.grouped(velocity[axis]):
            constrained = velocity[axis][I] != saved[axis][I]
            if 0 < I[axis] < velocity[axis].shape[axis] - 1:
                left = I - ti.Vector.unit(ti.static(len(velocity)), axis)
                constrained = constrained or cell_type[left] == SOLID_CELL or cell_type[I] == SOLID_CELL
            mask[axis][I] = ti.cast(constrained, ti.i32)
            saved[axis][I] = velocity[axis][I]


@ti.kernel
def kernel_restore_double_layer_solid_boundary(
    velocity: ti.template(),
    saved: ti.template(),
    mask: ti.template(),
):
    for axis in ti.static(range(len(velocity))):
        for I in ti.grouped(velocity[axis]):
            if mask[axis][I] != 0:
                velocity[axis][I] = saved[axis][I]


@ti.kernel
def kernel_accumulate_double_layer_pressure(total: ti.template(), increment: ti.template()):
    for I in ti.grouped(total):
        total[I] += increment[I]
