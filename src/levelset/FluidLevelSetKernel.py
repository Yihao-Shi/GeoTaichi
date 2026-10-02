import taichi as ti

from src.utils.constants import Threshold
import src.utils.GlobalVariable as GlobalVariable


@ti.func
def min_grid_spacing(grid_size):
    dx = grid_size[0]
    for d in ti.static(range(1, GlobalVariable.DIMENSION)):
        dx = ti.min(dx, grid_size[d])
    return dx


@ti.func
def grid_cell_volume(grid_size):
    volume = 1.
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        volume *= grid_size[d]
    return volume


@ti.func
def equivalent_particle_radius(volume):
    radius = 0.
    if ti.static(GlobalVariable.DIMENSION == 2):
        radius = 0.5 * ti.sqrt(2. * ti.max(volume, 0.))
    elif ti.static(GlobalVariable.DIMENSION == 3):
        radius = 0.5 * ti.sqrt(3.) * ti.pow(ti.max(volume, 0.), 1. / 3.)
    return radius


@ti.func
def free_surface_theta_from_phi_with_default(phi_f, phi_a, default_theta: float):
    theta = default_theta
    denom = phi_f - phi_a
    if phi_f < 0. and phi_a > 0. and ti.abs(denom) > Threshold:
        theta = phi_f / denom
    return ti.min(1., ti.max(0.01, theta))


@ti.func
def free_surface_theta_from_phi(phi_f, phi_a):
    return free_surface_theta_from_phi_with_default(phi_f, phi_a, 1.)


@ti.func
def free_surface_theta_from_phi_half_default(phi_f, phi_a):
    return free_surface_theta_from_phi_with_default(phi_f, phi_a, 0.5)


@ti.func
def free_surface_theta(fluid_cell, air_cell, fluid_sdf: ti.template()):
    return free_surface_theta_from_phi(fluid_sdf[fluid_cell], fluid_sdf[air_cell])


@ti.func
def smoothed_heaviside(phi, eps):
    value = 0.
    if phi <= -eps:
        value = 1.
    elif phi < eps:
        ratio = phi / eps
        value = 0.5 - 0.75 * ratio + 0.25 * ratio * ratio * ratio
    return value


@ti.func
def clamp_cell_index(index, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int)):
    clamped = index
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        clamped[d] = ti.max(-ghost_cell, ti.min(cnum[d] - ghost_cell - 1, clamped[d]))
    return clamped


@ti.func
def sample_cell_centered_sdf(position, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), sdf: ti.template()):
    base = ti.floor(position / grid_size - 0.5).cast(int)
    local = position / grid_size - base.cast(float) - 0.5
    phi = 0.
    if ti.static(GlobalVariable.DIMENSION == 2):
        for i, j in ti.static(ti.ndrange(2, 2)):
            node = clamp_cell_index(base + ti.Vector([i, j]), ghost_cell, cnum)
            wx = 1. - local[0]
            wy = 1. - local[1]
            if ti.static(i == 1):
                wx = local[0]
            if ti.static(j == 1):
                wy = local[1]
            phi += wx * wy * sdf[node]
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            node = clamp_cell_index(base + ti.Vector([i, j, k]), ghost_cell, cnum)
            wx = 1. - local[0]
            wy = 1. - local[1]
            wz = 1. - local[2]
            if ti.static(i == 1):
                wx = local[0]
            if ti.static(j == 1):
                wy = local[1]
            if ti.static(k == 1):
                wz = local[2]
            phi += wx * wy * wz * sdf[node]
    return phi


@ti.func
def sdf_normal(index, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), sdf: ti.template()):
    grad = ti.Vector.zero(float, GlobalVariable.DIMENSION)
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        direction = ti.Vector.unit(GlobalVariable.DIMENSION, d)
        grad[d] = 0.5 * (sdf[clamp_cell_index(index + direction, ghost_cell, cnum)] -
                         sdf[clamp_cell_index(index - direction, ghost_cell, cnum)]) / grid_size[d]

    normal = ti.Vector.zero(float, GlobalVariable.DIMENSION)
    norm = grad.norm()
    if norm > Threshold:
        normal = grad / norm
    return normal


@ti.func
def surface_curvature(index, ghost_cell: int, cnum: ti.types.vector(GlobalVariable.DIMENSION, int), grid_size: ti.types.vector(GlobalVariable.DIMENSION, float), sdf: ti.template()):
    curvature = 0.
    for d in ti.static(range(GlobalVariable.DIMENSION)):
        direction = ti.Vector.unit(GlobalVariable.DIMENSION, d)
        normal_r = sdf_normal(index + direction, ghost_cell, cnum, grid_size, sdf)
        normal_l = sdf_normal(index - direction, ghost_cell, cnum, grid_size, sdf)
        curvature += 0.5 * (normal_r[d] - normal_l[d]) / grid_size[d]
    return curvature
