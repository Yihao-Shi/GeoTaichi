import taichi as ti

from src.utils.constants import Threshold, DBL_EPSILON
from src.utils.TypeDefination import vec2f
from src.utils.VectorFunction import spherical_angle, normal_from_angles, coord_local2global, global2local


@ti.func
def compute_support_pairs(normal, mass_center1, mass_center2, rotate_matrix1, rotate_matrix2, params1, params2, surface1, surface2):
    local_normal1 = global2local(-normal, 1., rotate_matrix1)
    local_normal2 = global2local(normal, 1., rotate_matrix2)
    p = coord_local2global(1., rotate_matrix1, surface1.support(local_normal1, params1), mass_center1)
    q = coord_local2global(1., rotate_matrix2, surface2.support(local_normal2, params2), mass_center2)
    return p, q

@ti.func
def compute_distance_vector(spherical, mass_center1, mass_center2, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2):
    normal = normal_from_angles(*spherical)
    p, q = compute_support_pairs(normal, mass_center1, mass_center2, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2)
    d = p - q
    return d

@ti.func
def compute_jacobian_fdm(spherical, mass_center1, mass_center2, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2):
    delta = 1e-7
    d0 = compute_distance_vector(spherical, mass_center1, mass_center2, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2)
    d1 = compute_distance_vector(spherical + vec2f(delta, 0), mass_center1, scale1, scale2, mass_center2, rotate_matrix1, rotate_matrix2, surface1, surface2)
    d2 = compute_distance_vector(spherical + vec2f(0, delta), mass_center1, scale1, scale2, mass_center2, rotate_matrix1, rotate_matrix2, surface1, surface2)
    return ti.Matrix.cols([(d1 - d0) / delta, (d2 - d0) / delta])

@ti.func
def compute_jacobian(spherical, mass_center1, mass_center2, rotate_matrix1, rotate_matrix2, surface1, surface2):
    pass

@ti.func
def PCNiteration(margin1, margin2, radius1, radius2, mass_center1, mass_center2, normal, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2):
    if normal.norm() < Threshold:
        normal = (mass_center1 - mass_center2).normalized()
    spherical = spherical_angle(normal)

    is_touch = 1
    params1, params2 = surface1.physical_parameters(scale1), surface2.physical_parameters(scale2)
    p, q = compute_support_pairs(normal, mass_center1, mass_center2, rotate_matrix1, rotate_matrix2, params1, params2, surface1, surface2)
    d = p - q
    d_norm = d.norm()
    c = normal

    mu, nu = 0., 2.
    iter = 0
    stop = 0
    while d_norm > Threshold and stop == 0 and iter < 10:
        if ti.abs(d.dot(c)) / d_norm > 0.95:
            break

        jacobian = compute_jacobian_fdm(spherical, mass_center1, mass_center2, rotate_matrix1, rotate_matrix2, params1, params2, surface1, surface2)
        hess = jacobian.transpose() @ jacobian
        grad = jacobian.transpose() @ d

        if grad.norm() < Threshold:
            break

        if iter == 0:
            mu = 1e-3 * ti.abs(grad).max()

        while True:
            hess0 = hess + mu * ti.Matrix.identity(float, 2)
            den = hess0[0, 0] * hess0[1, 1] - hess0[0, 1] * hess0[0, 1]
            den = den if ti.abs(den) > Threshold else Threshold
            inv_jac_den = 1. / (hess0[0, 0] * hess0[1, 1] - hess0[0, 1] * hess0[0, 1])
            dspherical = vec2f((grad[0] * hess0[1, 1] - grad[1] * hess0[0, 1]) * inv_jac_den,
                               (grad[1] * hess0[0, 0] - grad[0] * hess0[1, 0]) * inv_jac_den)
            dspherical_norm = dspherical.norm()
            if ti.math.isnan(dspherical[0]) or ti.math.isnan(dspherical[1]):
                stop = 1
                break
            if dspherical_norm < Threshold:
                stop = 1
                break
            temp_spherical = spherical - dspherical
            if dspherical_norm >= (temp_spherical.norm() + DBL_EPSILON) / (Threshold * Threshold):
                stop = 1
                break
            c = normal_from_angles(*temp_spherical)
            p, q = compute_support_pairs(c, mass_center1, mass_center2, rotate_matrix1, rotate_matrix2, params1, params2, surface1, surface2)
            d_prime = p - q
            if d_prime.dot(c) > 0.:
                is_touch = 0
                stop = 1
                break
            d_prime_norm = d_prime.norm()
            if d_prime_norm < Threshold:
                stop = 1
                break
            lambda_ = (d_norm - d_prime_norm) / dspherical.dot(mu * dspherical + grad)
            if lambda_ > 0.:
                lambda_ = 1. - (2. * lambda_ - 1.) * (2. * lambda_ - 1.) * (2. * lambda_ - 1.)
                mu = mu * ti.max(lambda_, 1./3.)
                nu = 2.
                spherical = temp_spherical
                d = d_prime
                d_norm = d_prime_norm
                break
            mu *= nu
            nu *= 2
        
        if d.dot(c) > 0.:
            is_touch = 0
            break
        iter += 1
    return is_touch, p, q, c

