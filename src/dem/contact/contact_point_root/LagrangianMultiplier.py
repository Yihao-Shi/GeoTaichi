import taichi as ti

from src.utils.constants import Threshold
from src.utils.MatrixFunction import local2global_mat3x3
from src.utils.TypeDefination import vec3f, vec4f
from src.utils.VectorFunction import coord_global2local, coord_local2global, local2global, global2local, Squared


@ti.func
def residual(surface1, surface2, params1, params2, rotate_matrix1, rotate_matrix2, mass_center1, mass_center2, unknown):
    contact_point = vec3f(unknown[0], unknown[1], unknown[2])
    mu2 = unknown[3] * unknown[3]
    local_contact_point1 = coord_global2local(1., rotate_matrix1, contact_point, mass_center1)
    local_contact_point2 = coord_global2local(1., rotate_matrix2, contact_point, mass_center2)
    fx1, grad1 = surface1.fx(*local_contact_point1, params1), surface1.gradient(*local_contact_point1, params1)
    fx2, grad2 = surface2.fx(*local_contact_point2, params2), surface2.gradient(*local_contact_point2, params2)
    grad1 = local2global(grad1, 1., rotate_matrix1)
    grad2 = local2global(grad2, 1., rotate_matrix2)
    gradient = vec4f(grad1[0] + mu2 * grad2[0], grad1[1] + mu2 * grad2[1], grad1[2] + mu2 * grad2[2], fx1 - fx2)
    return gradient.norm()


@ti.func
def newton_iteration(mass_center1, mass_center2, contact_point, rotate_matrix1, rotate_matrix2, params1, params2, surface1, surface2):
    unknown = vec4f(contact_point[0], contact_point[1], contact_point[2], 1.)
    incre = vec4f(0., 0., 0., 0.)
    hessian = ti.Matrix.zero(float, 4, 4)

    res = residual(surface1, surface2, params1, params2, rotate_matrix1, rotate_matrix2, mass_center1, mass_center2, unknown)
    iter = 0
    mu = 0.
    while iter < 100:
        mu2 = unknown[3] * unknown[3]
        local_contact_point1 = coord_global2local(1., rotate_matrix1, contact_point, mass_center1)
        local_contact_point2 = coord_global2local(1., rotate_matrix2, contact_point, mass_center2)
        fx1, grad1, hess1 = surface1.fx(*local_contact_point1, params1), surface1.gradient(*local_contact_point1, params1), surface1.hessian(*local_contact_point1, params1)
        fx2, grad2, hess2 = surface2.fx(*local_contact_point2, params2), surface2.gradient(*local_contact_point2, params2), surface2.hessian(*local_contact_point2, params2)
        grad1 = local2global(grad1, 1., rotate_matrix1)
        grad2 = local2global(grad2, 1., rotate_matrix2)
        hess1 = local2global_mat3x3(hess1, 1., rotate_matrix1)
        hess2 = local2global_mat3x3(hess2, 1., rotate_matrix2)

        gradient = vec4f(grad1[0] + mu2 * grad2[0], grad1[1] + mu2 * grad2[1], grad1[2] + mu2 * grad2[2], fx1 - fx2)
        hessian[0, 0] = hess1[0, 0] + mu2 * hess2[0, 0]
        hessian[0, 1] = hess1[0, 1] + mu2 * hess2[0, 1]
        hessian[0, 2] = hess1[0, 2] + mu2 * hess2[0, 2]
        hessian[0, 3] = 2. * mu2 * grad2[0]
        hessian[1, 0] = hess1[1, 0] + mu2 * hess2[1, 0]
        hessian[1, 1] = hess1[1, 1] + mu2 * hess2[1, 1]
        hessian[1, 2] = hess1[1, 2] + mu2 * hess2[1, 2]
        hessian[1, 3] = 2. * mu2 * grad2[1]
        hessian[2, 0] = hess1[2, 0] + mu2 * hess2[2, 0]
        hessian[2, 1] = hess1[2, 1] + mu2 * hess2[2, 1]
        hessian[2, 2] = hess1[2, 2] + mu2 * hess2[2, 2]
        hessian[2, 3] = 2. * mu2 * grad2[2]
        hessian[3, 0] = grad1[0] - grad2[0]
        hessian[3, 1] = grad1[1] - grad2[1]
        hessian[3, 2] = grad1[2] - grad2[2]

        if iter == 0:
            mu = 1e-8 * ti.abs(hessian).max()

        AtA = hessian.transpose() @ hessian
        Atb = hessian.transpose() @ gradient
        direction = (AtA + mu * ti.Matrix.identity(float, 4)).inverse() @ Atb

        alpha = 1.0
        for _ in range(10):
            res_new = residual(surface1, surface2, params1, params2, rotate_matrix1, rotate_matrix2, mass_center1, mass_center2, unknown - alpha * direction)
            if res_new < res:
                res, res_new = res_new, res
                break
            alpha *= 0.5

        dincre = -alpha * direction
        unknown += dincre
        incre += dincre
        
        contact_point = vec3f(unknown[0], unknown[1], unknown[2])
        if dincre.norm() / incre.norm() < 1e-5 or res < 1e-6:
            break
        iter += 1
    return contact_point, iter == 100

@ti.func
def substepping(radius1, radius2, mass_center1, mass_center2, contact_point, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2):
    not_converge = 0
    for i in range(0, 11):
        temp_physical_parameter1 = surface1.evolving_physical_parameters(0.1*i, radius1, scale1)
        temp_physical_parameter2 = surface2.evolving_physical_parameters(0.1*i, radius2, scale2)
        contact_point, not_converge = newton_iteration(mass_center1, mass_center2, contact_point, rotate_matrix1, rotate_matrix2, temp_physical_parameter1, temp_physical_parameter2, surface1, surface2)
    return contact_point, not_converge

@ti.func
def LagrangianMultiplieriteration(margin1, margin2, radius1, radius2, mass_center1, mass_center2, contact_point, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2):
    if Squared(contact_point) < Threshold:
        frac1 = radius2 / (radius1 + radius2)
        frac2 = radius1 / (radius1 + radius2)
        contact_point = frac1 * mass_center1 + frac2 * mass_center2
    params1, params2 = surface1.physical_parameters(scale1), surface2.physical_parameters(scale2)
    contact_point, not_converge = newton_iteration(mass_center1, mass_center2, contact_point, rotate_matrix1, rotate_matrix2, params1, params2, surface1, surface2)
    if not_converge:
        frac1 = radius2 * (radius1 + radius2)
        frac2 = radius1 * (radius1 + radius2)
        contact_point = frac1 * mass_center1 + frac2 * mass_center2
        contact_point, not_converge = substepping(radius1, radius2, mass_center1, mass_center2, contact_point, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2)

    local_contact_point1 = coord_global2local(1., rotate_matrix1, contact_point, mass_center1)
    local_contact_point2 = coord_global2local(1., rotate_matrix2, contact_point, mass_center2)
    fx1 = surface1.fx(*local_contact_point1, params1)
    fx2 = surface2.fx(*local_contact_point2, params2)
    is_touch = fx1 < 0. and fx2 < 0.
    pa, pb = vec3f(0, 0, 0), vec3f(0, 0, 0)
    if is_touch:
        global_normal = (mass_center2 - mass_center1).normalized(Threshold)
        local_norm1, local_norm2 = global2local(global_normal, 1., rotate_matrix1), -global2local(global_normal, 1., rotate_matrix2)
        pa = coord_local2global(1., rotate_matrix1, surface1.nearest_point(local_contact_point1, local_norm1, params1), mass_center1)
        pb = coord_local2global(1., rotate_matrix2, surface2.nearest_point(local_contact_point2, local_norm2, params2), mass_center2)
    return is_touch, pa, pb, contact_point

@ti.func
def LagrangianMultiplierPWiteration(margin1, margin2, radius1, radius2, mass_center1, mass_center2, contact_point, scale1, scale2, rotate_matrix1, rotate_matrix2, surface1, surface2):
    pass
