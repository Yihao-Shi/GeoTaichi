import taichi as ti

from src.utils.constants import Threshold
from src.utils.TypeDefination import vec2i, vec3i, vec3f, vec6f
from src.utils.ScalarFunction import sgn
from src.levelset.WENO import *


# =================================================================================== #
#                                    Flux Function                                    #
# =================================================================================== #
@ti.func
def linear_funx(vel, phi_x):
    return vel * phi_x

@ti.func
def linear_dfunx(vel, phi_x_plus, phi_x_minus):
    return ti.abs(vel)

@ti.func
def burgers_funx(vel, phi_x):
    return 0.5 * vel * phi_x * phi_x

@ti.func
def burgers_dfunx(vel, phi_x_plus, phi_x_minus):
    return max(abs(vel * phi_x_plus), abs(vel * phi_x_minus))

@ti.func
def buckley_funx(vel, phi_x):
    return 4. * vel * phi_x * phi_x / (4 * phi_x * phi_x + (1 - phi_x) * (1 - phi_x))

@ti.func
def buckley_dfunx(vel, phi_x_plus, phi_x_minus):
    @ti.func
    def dfunx(phi_x):
        return 8 * phi_x * (1 - phi_x) / ((5 * phi_x * phi_x - 2 * phi_x + 1) * (5 * phi_x * phi_x - 2 * phi_x + 1))
    return max(abs(vel * dfunx(phi_x_plus)), abs(vel * dfunx(phi_x_minus)))


# =================================================================================== #
#                                    Numerical Flux                                   #
# =================================================================================== #
@ti.func
def numerical_flux_LF(vel, u_x_plus, u_x_minus):
    # Compute numerical flux using Lax-Friedrichs approximation.
    alpha = linear_dfunx(vel, u_x_plus, u_x_minus)
    avg = 0.5 * (u_x_plus + u_x_minus)
    term1 = linear_funx(vel, avg)
    term2 = -0.5 * alpha * (u_x_plus - u_x_minus)
    return term1 + term2

# =================================================================================== #
#                              Hamilton-Jacobi Equations                              #
# =================================================================================== #
@ti.func
def rhs_velocity_term1d(vel, phi_x):
    return -vel * phi_x

@ti.func
def rhs_normal_velocity_term1d(vel_n, phi_x_plus, phi_x_minus):
    phi_x_sq_cur = 0.
    if ti.abs(vel_n) > Threshold:
        if vel_n > 0.:
            phi_x_sq_cur = max(max(phi_x_minus, 0.0) ** 2, min(phi_x_plus, 0.0) ** 2)
        else:
            phi_x_sq_cur = max(min(phi_x_minus, 0.0) ** 2, max(phi_x_plus, 0.0) ** 2)
    return -vel_n * ti.sqrt(phi_x_sq_cur)

@ti.func
def rhs_velocity_term2d(vel_x, vel_y, phi_x, phi_y):
    return -(vel_x * phi_x + vel_y * phi_y)

@ti.func
def rhs_normal_velocity_term2d(vel_n, phi_x_plus, phi_x_minus, phi_y_plus, phi_y_minus):
    phi_x_sq_cur = 0.
    if ti.abs(vel_n) > Threshold:
        if vel_n > 0.:
            phi_x_sq_cur = max(max(phi_x_minus, 0.0) ** 2, min(phi_x_plus, 0.0) ** 2) + max(max(phi_y_minus, 0.0) ** 2, min(phi_y_plus, 0.0) ** 2)
        else:
            phi_x_sq_cur = max(min(phi_x_minus, 0.0) ** 2, max(phi_x_plus, 0.0) ** 2) + max(min(phi_y_minus, 0.0) ** 2, max(phi_y_plus, 0.0) ** 2)
    return -vel_n * ti.sqrt(phi_x_sq_cur)

@ti.func
def rhs_velocity_term3d(vel_x, vel_y, vel_z, phi_x, phi_y, phi_z):
    return -(vel_x * phi_x + vel_y * phi_y + vel_z * phi_z)

@ti.func
def rhs_normal_velocity_term3d(vel_n, phi_x_plus, phi_x_minus, phi_y_plus, phi_y_minus, phi_z_plus, phi_z_minus):
    phi_x_sq_cur = 0.
    if ti.abs(vel_n) > Threshold:
        if vel_n > 0.:
            phi_x_sq_cur = max(max(phi_x_minus, 0.0) ** 2, min(phi_x_plus, 0.0) ** 2) + max(max(phi_y_minus, 0.0) ** 2, min(phi_y_plus, 0.0) ** 2) + max(max(phi_z_minus, 0.0) ** 2, min(phi_z_plus, 0.0) ** 2)
        else:
            phi_x_sq_cur = max(min(phi_x_minus, 0.0) ** 2, max(phi_x_plus, 0.0) ** 2) + max(min(phi_y_minus, 0.0) ** 2, max(phi_y_plus, 0.0) ** 2) + max(min(phi_z_minus, 0.0) ** 2, max(phi_z_plus, 0.0) ** 2)
    return -vel_n * ti.sqrt(phi_x_sq_cur)


# =================================================================================== #
#                              Upwind Finite Difference                               #
# compute: phi_t = -v * phi_x                                                         #
# =================================================================================== #
@ti.func
def compute_du_plus(dim, grid_id, igird_size, phi):
    # biased to the right
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return (phi[grid_id + unit] - phi[grid_id]) * igird_size

@ti.func
def compute_du_minus(dim, grid_id, igird_size, phi):
    # biased to the left
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return (phi[grid_id] - phi[grid_id - unit]) * igird_size

@ti.func
def compute_du_center_grad_order2(dim, grid_id, igird_size, dphi):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return 0.5 * (dphi[grid_id + unit] - dphi[grid_id - unit]) * igird_size

@ti.func
def compute_du_minus_order2(dim, grid_id, igird_size, dphi):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return (1.5 * dphi[grid_id] - 2. * dphi[grid_id - unit] + 0.5 * dphi[grid_id - 2 * unit]) * igird_size

@ti.func
def compute_du_plus_order2(dim, grid_id, igird_size, dphi):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return (-1.5 * dphi[grid_id] + 2. * dphi[grid_id + unit] - 0.5 * dphi[grid_id + 2 * unit]) * igird_size

@ti.func
def compute_du_minus_order3(dim, grid_id, igird_size, dphi):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return 1./6. * (2. * dphi[grid_id + unit] + 3. * dphi[grid_id] - 6. * dphi[grid_id - unit] + dphi[grid_id - 2 * unit]) * igird_size

@ti.func
def compute_du_plus_order3(dim, grid_id, igird_size, dphi):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return 1./6. * (-2. * dphi[grid_id - unit] - 3. * dphi[grid_id] + 6. * dphi[grid_id + unit] - dphi[grid_id + 2 * unit]) * igird_size

@ti.func
def compute_du_center_grad_order4(dim, grid_id, igird_size, dphi):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return 1./12. * (dphi[grid_id - 2 * unit] - 8. * dphi[grid_id - unit] + 8. * dphi[grid_id + unit] - dphi[grid_id + 2 * unit]) * igird_size

@ti.func
def compute_ddu_order2(dim, grid_id, igird_size, dphi):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return (dphi[grid_id + unit] - 2. * dphi[grid_id] + dphi[grid_id - unit]) * igird_size * igird_size 

@ti.func
def compute_hessian_central2d(grid_id, igird_size, dphi):
    # Voigt tensor: phi_xx, phi_yy, phi_xy
    unit10 = vec2i(1, 0)
    unit01 = vec2i(0, 1)
    unit11 = vec2i(1, 1)
    igird_size2 = igird_size * igird_size
    return vec3f([dphi[grid_id + unit10] - 2. * dphi[grid_id] + dphi[grid_id - unit10],
                  0.25 * (dphi[grid_id + unit11] - dphi[grid_id + vec2i(-1, 1)] - dphi[grid_id + vec2i(1, -1)] + dphi[grid_id - unit11]),
                  dphi[grid_id + unit01] - 2. * dphi[grid_id] + dphi[grid_id - unit01],]) * igird_size2

@ti.func
def compute_hessian_central3d(grid_id, igird_size, dphi):
    # Voigt tensor: phi_xx, phi_yy, phi_zz, phi_xy, phi_yz, phi_xz
    unit100 = vec3i(1, 0, 0)
    unit010 = vec3i(0, 1, 0)
    unit001 = vec3i(0, 0, 1)
    igird_size2 = igird_size * igird_size
    return vec6f([dphi[grid_id + unit100] - 2. * dphi[grid_id] + dphi[grid_id - unit100],
                  dphi[grid_id + unit010] - 2. * dphi[grid_id] + dphi[grid_id - unit010],
                  dphi[grid_id + unit001] - 2. * dphi[grid_id] + dphi[grid_id - unit001],
                  0.25 * (dphi[grid_id + vec3i(1, 1, 0)] - dphi[grid_id + vec3i(-1, 1, 0)] - dphi[grid_id + vec3i(1, -1, 0)] + dphi[grid_id - vec3i(-1, -1, 0)]),
                  0.25 * (dphi[grid_id + vec3i(0, 1, 1)] - dphi[grid_id + vec3i(0, -1, 1)] - dphi[grid_id + vec3i(0, 1, -1)] + dphi[grid_id - vec3i(0, -1, -1)]),
                  0.25 * (dphi[grid_id + vec3i(1, 0, 1)] - dphi[grid_id + vec3i(-1, 0, 1)] - dphi[grid_id + vec3i(1, 0, -1)] + dphi[grid_id - vec3i(-1, 0, -1)]),]) * igird_size2

@ti.func
def ComputeENO1HJFlux(dim, grid_id, inv_grid_size, dt, vel, sdf, dsdf):
    # compute phi_x_plus 
    sdf_x_plus = compute_du_plus(dim, grid_id, inv_grid_size, sdf)
    # compute phi_x_minus 
    sdf_x_minus = compute_du_minus(dim, grid_id, inv_grid_size, sdf)
    dsdf[grid_id] += -numerical_flux_LF(vel, sdf, sdf_x_plus, sdf_x_minus) * dt

@ti.func
def ComputeENO1HJUpwind(dim, grid_id, inv_grid_size, dt, vel, sdf, dsdf):
    if vel > 0.:
        dsdf[grid_id] += -vel * compute_du_minus(dim, grid_id, inv_grid_size, sdf) * dt
    else:
        dsdf[grid_id] += -vel * compute_du_plus(dim, grid_id, inv_grid_size, sdf) * dt

@ti.func
def compute_ddu_order2_num(dim, grid_id, dphi):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    return dphi[grid_id + unit] - 2. * dphi[grid_id] + dphi[grid_id - unit]

@ti.func
def ComputeWENO5HJFlux(dim, grid_id, inv_grid_size, dt, vel, sdf, dsdf):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    der1 = compute_du_plus(dim, grid_id - 2 * unit, inv_grid_size, sdf)
    der2 = compute_du_plus(dim, grid_id - 1 * unit, inv_grid_size, sdf)
    der3 = compute_du_plus(dim, grid_id + 0 * unit, inv_grid_size, sdf)
    der4 = compute_du_plus(dim, grid_id + 1 * unit, inv_grid_size, sdf)
    common = weno5HJcommon(der1, der2, der3, der4)

    # Compute second derivatives
    der1 = compute_ddu_order2_num(dim, grid_id + 2 * unit, sdf) * inv_grid_size
    der2 = compute_ddu_order2_num(dim, grid_id + 1 * unit, sdf) * inv_grid_size
    der3 = compute_ddu_order2_num(dim, grid_id + 0 * unit, sdf) * inv_grid_size
    der4 = compute_ddu_order2_num(dim, grid_id - 1 * unit, sdf) * inv_grid_size
    der5 = compute_ddu_order2_num(dim, grid_id - 2 * unit, sdf) * inv_grid_size

    # compute phi_x_plus 
    weno_plus_flux = weno5HJ(der1, der2, der3, der4)
    sdf_x_plus = common + weno_plus_flux
    # compute phi_x_minus 
    weno_minus_flux = weno5HJ(der5, der4, der3, der2)
    sdf_x_minus = common - weno_minus_flux
    # compute numerical flux
    dsdf[grid_id] += -numerical_flux_LF(vel, sdf, sdf_x_plus, sdf_x_minus) * dt


@ti.func
def ComputeWENO5HJUpwind(dim, grid_id, inv_grid_size, dt, vel, sdf, dsdf):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    der1 = compute_du_plus(dim, grid_id - 2 * unit, inv_grid_size, sdf)
    der2 = compute_du_plus(dim, grid_id - 1 * unit, inv_grid_size, sdf)
    der3 = compute_du_plus(dim, grid_id + 0 * unit, inv_grid_size, sdf)
    der4 = compute_du_plus(dim, grid_id + 1 * unit, inv_grid_size, sdf)
    common = weno5HJcommon(der1, der2, der3, der4)

    # Compute second derivatives
    der2 = compute_ddu_order2_num(dim, grid_id + 1 * unit, sdf) * inv_grid_size
    der3 = compute_ddu_order2_num(dim, grid_id + 0 * unit, sdf) * inv_grid_size
    der4 = compute_ddu_order2_num(dim, grid_id - 1 * unit, sdf) * inv_grid_size

    # compute phi_x_plus 
    if vel < 0.:
        der1 = compute_ddu_order2_num(dim, grid_id + 2 * unit, sdf) * inv_grid_size
        common += weno5HJ(der1, der2, der3, der4)
    else:
        der5 = compute_ddu_order2_num(dim, grid_id - 2 * unit, sdf) * inv_grid_size
        common += weno5HJ(der5, der4, der3, der2)
    # compute numerical flux
    dsdf[grid_id] += -vel * common * dt


# =================================================================================== #
#                                  Reinitialization                                   #
# compute: phi_t = -grad(phi) dot { sgn(psi)/|grad(psi)| grad(psi) }                  #
# =================================================================================== #
@ti.func
def reinitialization_phi_x(dim, grid_id, inv_grid_size, sdf):
    unit = ti.Vector.unit(grid_id.n, dim, int)
    der1 = compute_du_plus(dim, grid_id - 2 * unit, inv_grid_size, sdf)
    der2 = compute_du_plus(dim, grid_id - 1 * unit, inv_grid_size, sdf)
    der3 = compute_du_plus(dim, grid_id + 0 * unit, inv_grid_size, sdf)
    der4 = compute_du_plus(dim, grid_id + 1 * unit, inv_grid_size, sdf)
    common = weno5HJcommon(der1, der2, der3, der4)

    # Compute second derivatives
    der1 = compute_ddu_order2_num(dim, grid_id + 2 * unit, sdf) * inv_grid_size
    der2 = compute_ddu_order2_num(dim, grid_id + 1 * unit, sdf) * inv_grid_size
    der3 = compute_ddu_order2_num(dim, grid_id + 0 * unit, sdf) * inv_grid_size
    der4 = compute_ddu_order2_num(dim, grid_id - 1 * unit, sdf) * inv_grid_size
    der5 = compute_ddu_order2_num(dim, grid_id - 2 * unit, sdf) * inv_grid_size

    # compute phi_x_plus 
    weno_plus_flux = weno5HJ(der1, der2, der3, der4)
    sdf_x_plus = common + weno_plus_flux
    # compute phi_x_minus 
    weno_minus_flux = weno5HJ(der5, der4, der3, der2)
    sdf_x_minus = common - weno_minus_flux

    if sdf[grid_id] > 0.:
        sdf_x_plus = ti.max(-sdf_x_plus, 0.)
        sdf_x_minus = ti.max(sdf_x_minus, 0.)
    else:
        sdf_x_plus = ti.max(sdf_x_plus, 0.)
        sdf_x_minus = ti.max(-sdf_x_minus, 0.)
    return ti.max(sdf_x_plus, sdf_x_minus)

