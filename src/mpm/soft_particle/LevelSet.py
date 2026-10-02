import taichi as ti

from src.utils.constants import PI, Threshold, ZEROVEC3f
from src.utils.ScalarFunction import linearize3D
from src.utils.TypeDefination import vec3f, vec3i


@ti.func
def split_soft_grid_slot(grid_slot, soft_grid_owner: ti.template(), soft_grid_local: ti.template()):
    owner = soft_grid_owner[grid_slot]
    local_grid = soft_grid_local[grid_slot]
    return owner, local_grid


@ti.func
def sample_levelset_unscaled(local_pos, body_box, grid):
    scale = body_box.scale
    xmin = body_box.xmin / scale
    xmax = body_box.xmax / scale
    dx = body_box.grid_space / scale
    query = local_pos
    for d in ti.static(range(3)):
        query[d] = ti.min(ti.max(query[d], xmin[d]), xmax[d])
    base = vec3i(0, 0, 0)
    for d in ti.static(range(3)):
        base[d] = ti.min(body_box.gnum[d] - 2, ti.max(0, int((query[d] - xmin[d]) / dx)))
    origin = xmin + vec3f(base[0], base[1], base[2]) * dx
    fx = (query - origin) / dx
    value = 0.0
    for i in ti.static(range(2)):
        wx = (1.0 - fx[0]) if i == 0 else fx[0]
        for j in ti.static(range(2)):
            wy = (1.0 - fx[1]) if j == 0 else fx[1]
            for k in ti.static(range(2)):
                wz = (1.0 - fx[2]) if k == 0 else fx[2]
                node = linearize3D(base[0] + i, base[1] + j, base[2] + k, body_box.gnum) + body_box.startGrid
                value += wx * wy * wz * grid[node].distance_field
    return value


@ti.func
def sample_levelset_reference_bounds_unscaled(local_pos, body_box, grid):
    """Bounds of the stored source field on a trilinear donor cell."""
    scale = body_box.scale
    xmin = body_box.xmin / scale
    xmax = body_box.xmax / scale
    dx = body_box.grid_space / scale
    query = local_pos
    for d in ti.static(range(3)):
        query[d] = ti.min(ti.max(query[d], xmin[d]), xmax[d])
    base = vec3i(0, 0, 0)
    for d in ti.static(range(3)):
        base[d] = ti.min(
            body_box.gnum[d] - 2,
            ti.max(0, int((query[d] - xmin[d]) / dx)),
        )
    bounds = ti.Vector([1.0e30, -1.0e30])
    for i in ti.static(range(2)):
        for j in ti.static(range(2)):
            for k in ti.static(range(2)):
                node = (
                    linearize3D(
                        base[0] + i,
                        base[1] + j,
                        base[2] + k,
                        body_box.gnum,
                    )
                    + body_box.startGrid
                )
                value = grid[node].distance_field0
                bounds[0] = ti.min(bounds[0], value)
                bounds[1] = ti.max(bounds[1], value)
    return bounds


@ti.func
def soft_levelset_grid_spacing_(bodyID, box):
    return box[bodyID].grid_space / ti.max(box[bodyID].scale, Threshold)


@ti.func
def soft_local_grid_ijk_(local_grid, gnum):
    i = local_grid % gnum[0]
    j = (local_grid % (gnum[0] * gnum[1])) // gnum[0]
    k = local_grid // (gnum[0] * gnum[1])
    return vec3i(i, j, k)


@ti.func
def soft_levelset_local_id_(ijk, gnum):
    clamped = ti.min(ti.max(ijk, vec3i(0, 0, 0)), gnum - 1)
    return linearize3D(clamped[0], clamped[1], clamped[2], gnum)


@ti.func
def soft_levelset_phi_(sb, local_id, soft, grid):
    return grid[soft[sb].gridStart + local_id].distance_field


@ti.func
def soft_levelset_phi0_(sb, local_id, soft, grid):
    return grid[soft[sb].gridStart + local_id].distance_field0


@ti.func
def soft_levelset_neighbor_phi_(sb, ijk, gnum, soft, grid):
    return soft_levelset_phi_(sb, soft_levelset_local_id_(ijk, gnum), soft, grid)


@ti.func
def soft_levelset_neighbor_phi0_(sb, ijk, gnum, soft, grid):
    return soft_levelset_phi0_(sb, soft_levelset_local_id_(ijk, gnum), soft, grid)


@ti.func
def soft_levelset_centered_gradient_(sb, local_grid, soft, grid, box):
    bodyID = soft[sb].bodyID
    gnum = box[bodyID].gnum
    h = soft_levelset_grid_spacing_(bodyID, box)
    ijk = soft_local_grid_ijk_(local_grid, gnum)
    phi_c = soft_levelset_phi_(sb, local_grid, soft, grid)
    grad = ZEROVEC3f
    for d in ti.static(range(3)):
        plus = vec3i(ijk[0], ijk[1], ijk[2])
        minus = vec3i(ijk[0], ijk[1], ijk[2])
        plus[d] = ti.min(gnum[d] - 1, ijk[d] + 1)
        minus[d] = ti.max(0, ijk[d] - 1)
        phi_p = soft_levelset_neighbor_phi_(sb, plus, gnum, soft, grid)
        phi_m = soft_levelset_neighbor_phi_(sb, minus, gnum, soft, grid)
        if plus[d] != ijk[d] and minus[d] != ijk[d]:
            grad[d] = (phi_p - phi_m) / (2.0 * h)
        elif plus[d] != ijk[d]:
            grad[d] = (phi_p - phi_c) / h
        elif minus[d] != ijk[d]:
            grad[d] = (phi_c - phi_m) / h
    return grad


@ti.func
def soft_levelset_reference_gradient_(sb, local_grid, soft, grid, box):
    bodyID = soft[sb].bodyID
    gnum = box[bodyID].gnum
    h = soft_levelset_grid_spacing_(bodyID, box)
    ijk = soft_local_grid_ijk_(local_grid, gnum)
    phi_c = soft_levelset_phi0_(sb, local_grid, soft, grid)
    grad = ZEROVEC3f
    for d in ti.static(range(3)):
        plus = vec3i(ijk[0], ijk[1], ijk[2])
        minus = vec3i(ijk[0], ijk[1], ijk[2])
        plus[d] = ti.min(gnum[d] - 1, ijk[d] + 1)
        minus[d] = ti.max(0, ijk[d] - 1)
        phi_p = soft_levelset_neighbor_phi0_(sb, plus, gnum, soft, grid)
        phi_m = soft_levelset_neighbor_phi0_(sb, minus, gnum, soft, grid)
        if plus[d] != ijk[d] and minus[d] != ijk[d]:
            grad[d] = (phi_p - phi_m) / (2.0 * h)
        elif plus[d] != ijk[d]:
            grad[d] = (phi_p - phi_c) / h
        elif minus[d] != ijk[d]:
            grad[d] = (phi_c - phi_m) / h
    return grad


@ti.func
def soft_levelset_has_two_sided_variation_(sb, local_grid, soft, grid, box):
    bodyID = soft[sb].bodyID
    gnum = box[bodyID].gnum
    ijk = soft_local_grid_ijk_(local_grid, gnum)
    phi = soft_levelset_phi_(sb, local_grid, soft, grid)
    has_lower = False
    has_higher = False
    for d in ti.static(range(3)):
        plus = vec3i(ijk[0], ijk[1], ijk[2])
        minus = vec3i(ijk[0], ijk[1], ijk[2])
        plus[d] = ti.min(gnum[d] - 1, ijk[d] + 1)
        minus[d] = ti.max(0, ijk[d] - 1)
        phi_p = soft_levelset_neighbor_phi_(sb, plus, gnum, soft, grid)
        phi_m = soft_levelset_neighbor_phi_(sb, minus, gnum, soft, grid)
        has_lower = has_lower or phi_p < phi - Threshold or phi_m < phi - Threshold
        has_higher = has_higher or phi_p > phi + Threshold or phi_m > phi + Threshold
    return has_lower and has_higher


@ti.func
def soft_levelset_regularized_inside_(phi, epsilon):
    q = phi / epsilon
    value = 0.0
    if q <= -1.0:
        value = 1.0
    elif q < 1.0:
        value = 0.5 * (1.0 - q - ti.sin(PI * q) / PI)
    return value


@ti.func
def soft_levelset_regularized_delta_(phi, epsilon):
    q = phi / epsilon
    value = 0.0
    if ti.abs(q) < 1.0:
        value = 0.5 * (1.0 + ti.cos(PI * q)) / epsilon
    return value


@ti.func
def soft_levelset_tetra_inside_fraction_(values):
    phi = values
    for i in ti.static(range(4)):
        for j in ti.static(range(4)):
            if ti.static(i < j):
                if phi[i] > phi[j]:
                    swap = phi[i]
                    phi[i] = phi[j]
                    phi[j] = swap
    inside = 0
    for i in ti.static(range(4)):
        inside += ti.cast(phi[i] < 0.0, ti.i32)

    fraction = 0.0
    if inside == 4:
        fraction = 1.0
    elif inside == 1:
        a = -phi[0]
        fraction = (
            a / ti.max(a + phi[1], Threshold) * a / ti.max(a + phi[2], Threshold) * a / ti.max(a + phi[3], Threshold)
        )
    elif inside == 3:
        d = phi[3]
        outside_fraction = (
            d / ti.max(d - phi[0], Threshold) * d / ti.max(d - phi[1], Threshold) * d / ti.max(d - phi[2], Threshold)
        )
        fraction = 1.0 - outside_fraction
    elif inside == 2:
        a = -phi[0]
        b = -phi[1]
        c = phi[2]
        d = phi[3]
        e = a / ti.max(a + c, Threshold)
        f = a / ti.max(a + d, Threshold)
        g = b / ti.max(b + c, Threshold)
        h = b / ti.max(b + d, Threshold)
        fraction = e * f + f * g * (1.0 - e) + g * h * (1.0 - f)
    return ti.min(1.0, ti.max(0.0, fraction))


@ti.func
def soft_levelset_cell_inside_fraction_(corner_phi):
    fraction = 0.0
    fraction += soft_levelset_tetra_inside_fraction_(
        ti.Vector(
            [
                corner_phi[0],
                corner_phi[1],
                corner_phi[3],
                corner_phi[7],
            ]
        )
    )
    fraction += soft_levelset_tetra_inside_fraction_(
        ti.Vector(
            [
                corner_phi[0],
                corner_phi[3],
                corner_phi[2],
                corner_phi[7],
            ]
        )
    )
    fraction += soft_levelset_tetra_inside_fraction_(
        ti.Vector(
            [
                corner_phi[0],
                corner_phi[2],
                corner_phi[6],
                corner_phi[7],
            ]
        )
    )
    fraction += soft_levelset_tetra_inside_fraction_(
        ti.Vector(
            [
                corner_phi[0],
                corner_phi[6],
                corner_phi[4],
                corner_phi[7],
            ]
        )
    )
    fraction += soft_levelset_tetra_inside_fraction_(
        ti.Vector(
            [
                corner_phi[0],
                corner_phi[4],
                corner_phi[5],
                corner_phi[7],
            ]
        )
    )
    fraction += soft_levelset_tetra_inside_fraction_(
        ti.Vector(
            [
                corner_phi[0],
                corner_phi[5],
                corner_phi[1],
                corner_phi[7],
            ]
        )
    )
    return fraction / 6.0


@ti.kernel
def reset_soft_levelset_material_volume_(
    softNum: int,
    material_volume: ti.template(),
):
    for sb in range(softNum):
        material_volume[sb] = 0.0


@ti.kernel
def accumulate_soft_material_volume_(
    softNum: int,
    pointNum: int,
    soft_point: ti.template(),
    rigid: ti.template(),
    material_volume: ti.template(),
):
    for p in range(pointNum):
        if soft_point[p].active == 1:
            bodyID = soft_point[p].bodyID
            sb = rigid[bodyID].softID
            if sb >= 0 and sb < softNum:
                current_volume = soft_point[p].vol0 * soft_point[p].F.determinant()
                ti.atomic_add(material_volume[sb], current_volume)


@ti.kernel
def reset_soft_levelset_volume_integrals_(
    softNum: int,
    current_volume: ti.template(),
    interface_measure: ti.template(),
):
    for sb in range(softNum):
        current_volume[sb] = 0.0
        interface_measure[sb] = 0.0


@ti.kernel
def accumulate_soft_levelset_volume_integrals_(
    softNum: int,
    maxGridNum: int,
    epsilon_cells: float,
    soft: ti.template(),
    grid: ti.template(),
    box: ti.template(),
    current_volume: ti.template(),
    interface_measure: ti.template(),
):
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            bodyID = soft[sb].bodyID
            gnum = box[bodyID].gnum
            ijk = soft_local_grid_ijk_(local_grid, gnum)
            if ijk[0] < gnum[0] - 1 and ijk[1] < gnum[1] - 1 and ijk[2] < gnum[2] - 1:
                corner_phi = ti.Vector.zero(float, 8)
                for i in ti.static(range(2)):
                    for j in ti.static(range(2)):
                        for k in ti.static(range(2)):
                            neighbor = vec3i(ijk[0] + i, ijk[1] + j, ijk[2] + k)
                            neighbor_id = soft_levelset_local_id_(neighbor, gnum)
                            corner = i + 2 * j + 4 * k
                            corner_phi[corner] = soft_levelset_phi_(sb, neighbor_id, soft, grid)
                physical_h = box[bodyID].grid_space
                unscaled_h = soft_levelset_grid_spacing_(bodyID, box)
                derivative_probe = 0.01 * ti.max(epsilon_cells, 0.02) * unscaled_h
                cell_volume = physical_h * physical_h * physical_h
                inside_fraction = soft_levelset_cell_inside_fraction_(corner_phi)
                minus_fraction = soft_levelset_cell_inside_fraction_(corner_phi - derivative_probe)
                plus_fraction = soft_levelset_cell_inside_fraction_(corner_phi + derivative_probe)
                ti.atomic_add(
                    current_volume[sb],
                    inside_fraction * cell_volume,
                )
                ti.atomic_add(
                    interface_measure[sb],
                    (minus_fraction - plus_fraction) * cell_volume / (2.0 * derivative_probe),
                )


@ti.kernel
def initialize_soft_levelset_volume_reference_(
    softStart: int,
    softNum: int,
    reference_sdf_volume: ti.template(),
    reference_material_volume: ti.template(),
    material_volume: ti.template(),
    current_volume: ti.template(),
    target_volume: ti.template(),
    volume_shift: ti.template(),
    cumulative_shift: ti.template(),
    volume_error: ti.template(),
    max_error: ti.template(),
):
    max_error[None] = 0.0
    for sb in range(softStart, softNum):
        reference_sdf_volume[sb] = current_volume[sb]
        reference_material_volume[sb] = material_volume[sb]
        target_volume[sb] = current_volume[sb]
        volume_shift[sb] = 0.0
        cumulative_shift[sb] = 0.0
        volume_error[sb] = 0.0


@ti.kernel
def prepare_soft_levelset_target_volume_(
    softNum: int,
    reference_sdf_volume: ti.template(),
    reference_material_volume: ti.template(),
    material_volume: ti.template(),
    target_volume: ti.template(),
):
    for sb in range(softNum):
        target = reference_sdf_volume[sb]
        if reference_material_volume[sb] > Threshold:
            target *= material_volume[sb] / reference_material_volume[sb]
        target_volume[sb] = target


@ti.kernel
def compute_soft_levelset_volume_shift_(
    softNum: int,
    tolerance: float,
    max_shift_cells: float,
    soft: ti.template(),
    box: ti.template(),
    target_volume: ti.template(),
    current_volume: ti.template(),
    interface_measure: ti.template(),
    volume_shift: ti.template(),
    volume_error: ti.template(),
    max_error: ti.template(),
):
    max_error[None] = 0.0
    for sb in range(softNum):
        target = target_volume[sb]
        error = 0.0
        shift = 0.0
        if target > Threshold:
            error = (current_volume[sb] - target) / target
            if ti.abs(error) > tolerance and interface_measure[sb] > Threshold:
                bodyID = soft[sb].bodyID
                max_shift = max_shift_cells * soft_levelset_grid_spacing_(bodyID, box)
                shift = (current_volume[sb] - target) / interface_measure[sb]
                shift = ti.min(max_shift, ti.max(-max_shift, shift))
        volume_error[sb] = error
        volume_shift[sb] = shift
        ti.atomic_max(max_error[None], ti.abs(error))


@ti.kernel
def initialize_soft_levelset_volume_bracket_(
    softNum: int,
    tolerance: float,
    max_shift_cells: float,
    soft: ti.template(),
    box: ti.template(),
    target_volume: ti.template(),
    current_volume: ti.template(),
    lower_shift: ti.template(),
    upper_shift: ti.template(),
    trial_shift: ti.template(),
    volume_shift: ti.template(),
):
    for sb in range(softNum):
        target = target_volume[sb]
        lower_shift[sb] = 0.0
        upper_shift[sb] = 0.0
        trial_shift[sb] = 0.0
        volume_shift[sb] = 0.0
        if target > Threshold:
            error = (current_volume[sb] - target) / target
            if ti.abs(error) > tolerance:
                bodyID = soft[sb].bodyID
                max_shift = max_shift_cells * soft_levelset_grid_spacing_(bodyID, box)
                if error > 0.0:
                    upper_shift[sb] = max_shift
                    trial_shift[sb] = max_shift
                    volume_shift[sb] = max_shift
                else:
                    lower_shift[sb] = -max_shift
                    trial_shift[sb] = -max_shift
                    volume_shift[sb] = -max_shift


@ti.kernel
def compute_soft_levelset_safeguarded_shift_(
    softNum: int,
    tolerance: float,
    target_volume: ti.template(),
    current_volume: ti.template(),
    interface_measure: ti.template(),
    lower_shift: ti.template(),
    upper_shift: ti.template(),
    trial_shift: ti.template(),
    volume_shift: ti.template(),
    volume_error: ti.template(),
    max_error: ti.template(),
):
    max_error[None] = 0.0
    for sb in range(softNum):
        target = target_volume[sb]
        error = 0.0
        increment = 0.0
        if target > Threshold:
            residual = current_volume[sb] - target
            error = residual / target
            if ti.abs(error) > tolerance:
                current = trial_shift[sb]
                lower = lower_shift[sb]
                upper = upper_shift[sb]
                if residual > 0.0:
                    lower = ti.max(lower, current)
                else:
                    upper = ti.min(upper, current)
                candidate = 0.5 * (lower + upper)
                width = upper - lower
                if interface_measure[sb] > Threshold and width > Threshold:
                    newton = current + residual / interface_measure[sb]
                    guard = 0.05 * width
                    if newton > lower + guard and newton < upper - guard:
                        candidate = newton
                increment = candidate - current
                lower_shift[sb] = lower
                upper_shift[sb] = upper
                trial_shift[sb] = candidate
        volume_error[sb] = error
        volume_shift[sb] = increment
        ti.atomic_max(max_error[None], ti.abs(error))


@ti.kernel
def apply_soft_levelset_volume_shift_(
    softNum: int,
    maxGridNum: int,
    soft: ti.template(),
    grid: ti.template(),
    volume_shift: ti.template(),
    cumulative_shift: ti.template(),
):
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            grid[soft[sb].gridStart + local_grid].distance_field += volume_shift[sb]
            if local_grid == 0:
                cumulative_shift[sb] += volume_shift[sb]


@ti.func
def soft_levelset_godunov_norm_(phi0, dxm, dxp, dym, dyp, dzm, dzp):
    x2 = 0.0
    y2 = 0.0
    z2 = 0.0
    if phi0 >= 0.0:
        x2 = ti.max(ti.max(dxm, 0.0) * ti.max(dxm, 0.0), ti.min(dxp, 0.0) * ti.min(dxp, 0.0))
        y2 = ti.max(ti.max(dym, 0.0) * ti.max(dym, 0.0), ti.min(dyp, 0.0) * ti.min(dyp, 0.0))
        z2 = ti.max(ti.max(dzm, 0.0) * ti.max(dzm, 0.0), ti.min(dzp, 0.0) * ti.min(dzp, 0.0))
    else:
        x2 = ti.max(ti.min(dxm, 0.0) * ti.min(dxm, 0.0), ti.max(dxp, 0.0) * ti.max(dxp, 0.0))
        y2 = ti.max(ti.min(dym, 0.0) * ti.min(dym, 0.0), ti.max(dyp, 0.0) * ti.max(dyp, 0.0))
        z2 = ti.max(ti.min(dzm, 0.0) * ti.min(dzm, 0.0), ti.max(dzp, 0.0) * ti.max(dzp, 0.0))
    return ti.sqrt(ti.max(x2 + y2 + z2, 0.0))


@ti.func
def soft_levelset_minmod_(a, b):
    value = 0.0
    if a * b > 0.0:
        value = ti.math.sign(a) * ti.min(ti.abs(a), ti.abs(b))
    return value


@ti.func
def soft_levelset_eno2_derivatives_(sb, ijk, axis, gnum, h, soft, grid):
    minus1 = vec3i(ijk[0], ijk[1], ijk[2])
    minus2 = vec3i(ijk[0], ijk[1], ijk[2])
    plus1 = vec3i(ijk[0], ijk[1], ijk[2])
    plus2 = vec3i(ijk[0], ijk[1], ijk[2])
    minus1[axis] = ti.max(0, ijk[axis] - 1)
    minus2[axis] = ti.max(0, ijk[axis] - 2)
    plus1[axis] = ti.min(gnum[axis] - 1, ijk[axis] + 1)
    plus2[axis] = ti.min(gnum[axis] - 1, ijk[axis] + 2)

    phi = soft_levelset_neighbor_phi_(sb, ijk, gnum, soft, grid)
    phi_m1 = soft_levelset_neighbor_phi_(sb, minus1, gnum, soft, grid)
    phi_m2 = soft_levelset_neighbor_phi_(sb, minus2, gnum, soft, grid)
    phi_p1 = soft_levelset_neighbor_phi_(sb, plus1, gnum, soft, grid)
    phi_p2 = soft_levelset_neighbor_phi_(sb, plus2, gnum, soft, grid)
    inv_h2 = 1.0 / (h * h)
    curvature_center = (phi_p1 - 2.0 * phi + phi_m1) * inv_h2
    curvature_minus = (phi - 2.0 * phi_m1 + phi_m2) * inv_h2
    curvature_plus = (phi_p2 - 2.0 * phi_p1 + phi) * inv_h2
    derivative_minus = (phi - phi_m1) / h + 0.5 * h * soft_levelset_minmod_(curvature_center, curvature_minus)
    derivative_plus = (phi_p1 - phi) / h - 0.5 * h * soft_levelset_minmod_(curvature_center, curvature_plus)
    return derivative_minus, derivative_plus


@ti.func
def soft_levelset_has_interface_edge_(sb, ijk, gnum, soft, grid):
    phi0 = soft_levelset_neighbor_phi0_(sb, ijk, gnum, soft, grid)
    has_crossing = 0
    for axis in ti.static(range(3)):
        minus1 = vec3i(ijk[0], ijk[1], ijk[2])
        plus1 = vec3i(ijk[0], ijk[1], ijk[2])
        minus1[axis] = ti.max(0, ijk[axis] - 1)
        plus1[axis] = ti.min(gnum[axis] - 1, ijk[axis] + 1)
        phi0_minus = soft_levelset_neighbor_phi0_(sb, minus1, gnum, soft, grid)
        phi0_plus = soft_levelset_neighbor_phi0_(sb, plus1, gnum, soft, grid)
        if phi0 * phi0_minus < 0.0 or phi0 * phi0_plus < 0.0:
            has_crossing = 1
    return has_crossing


@ti.kernel
def monitor_soft_levelset_signed_distance_error_(
    softNum: int,
    maxGridNum: int,
    monitor_band_cells: float,
    soft: ti.template(),
    grid: ti.template(),
    box: ti.template(),
    grad_error: ti.template(),
):
    grad_error[None] = 0.0
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            bodyID = soft[sb].bodyID
            h = soft_levelset_grid_spacing_(bodyID, box)
            phi = soft_levelset_phi_(sb, local_grid, soft, grid)
            if ti.abs(phi) <= monitor_band_cells * h and soft_levelset_has_two_sided_variation_(
                sb, local_grid, soft, grid, box
            ):
                grad = soft_levelset_centered_gradient_(sb, local_grid, soft, grid, box)
                error = ti.abs(grad.norm() - 1.0)
                ti.atomic_max(grad_error[None], error)


@ti.kernel
def store_soft_levelset_reinit_reference_(softNum: int, maxGridNum: int, soft: ti.template(), grid: ti.template()):
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            grid[sdf_grid].distance_field0 = grid[sdf_grid].distance_field
            grid[sdf_grid].distance_field_temp = grid[sdf_grid].distance_field


@ti.kernel
def reinitialize_soft_levelset_step_(
    softNum: int,
    maxGridNum: int,
    reinit_band_cells: float,
    dtau_factor: float,
    soft: ti.template(),
    grid: ti.template(),
    box: ti.template(),
):
    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            bodyID = soft[sb].bodyID
            gnum = box[bodyID].gnum
            h = soft_levelset_grid_spacing_(bodyID, box)
            ijk = soft_local_grid_ijk_(local_grid, gnum)
            phi = soft_levelset_phi_(sb, local_grid, soft, grid)
            phi0 = soft_levelset_phi0_(sb, local_grid, soft, grid)
            phi_new = phi
            if ti.abs(phi0) <= reinit_band_cells * h:
                reference_grad_norm = soft_levelset_reference_gradient_(sb, local_grid, soft, grid, box).norm()
                if ti.abs(phi0) <= Threshold:
                    phi_new = 0.0
                else:
                    has_interface_edge = soft_levelset_has_interface_edge_(sb, ijk, gnum, soft, grid)
                    if has_interface_edge == 1:
                        phi_new = phi0
                    else:
                        dxm, dxp = soft_levelset_eno2_derivatives_(sb, ijk, 0, gnum, h, soft, grid)
                        dym, dyp = soft_levelset_eno2_derivatives_(sb, ijk, 1, gnum, h, soft, grid)
                        dzm, dzp = soft_levelset_eno2_derivatives_(sb, ijk, 2, gnum, h, soft, grid)
                        grad_norm = soft_levelset_godunov_norm_(phi0, dxm, dxp, dym, dyp, dzm, dzp)
                        sign_phi0 = phi0 / ti.sqrt(phi0 * phi0 + h * h * reference_grad_norm * reference_grad_norm)
                        phi_new = phi - dtau_factor * h * sign_phi0 * (grad_norm - 1.0)
                        if phi_new * phi0 <= 0.0:
                            phi_new = phi0
            grid[soft[sb].gridStart + local_grid].distance_field_temp = phi_new

    for sb, local_grid in ti.ndrange(softNum, maxGridNum):
        if local_grid < soft[sb].gridNum:
            sdf_grid = soft[sb].gridStart + local_grid
            grid[sdf_grid].distance_field = grid[sdf_grid].distance_field_temp
