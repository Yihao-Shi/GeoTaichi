import numpy as np
import heapq

from src.nurbs.SplinePrimitives import Spline, SplineCurve, SplineSurface
from src.nurbs.Geomtry import capsule_capsule_distance, triangle_slab_triangle_slab_distance, triangle_slab_plane_distance, visualize
from src.nurbs.OptimizerVisualize import plot_nd_slice

class Queue:
    def __init__(self):
        self.queue = []

    def append(self, patchA, patchB):
        width = max(
            patchA[0][1] - patchA[0][0],  # u-range of A
            patchA[1][1] - patchA[1][0],  # v-range of A
            patchB[0][1] - patchB[0][0],  # u-range of B
            patchB[1][1] - patchB[1][0]   # v-range of B
        )
        heapq.heappush(self.queue, (width, patchA, patchB))

    def pop(self):
        width, patchA, patchB = heapq.heappop(self.queue)
        return patchA, patchB
    
    def size(self):
        return len(self.queue)
    
def update_patch_aabb(point_cloud, tol=1e-8):
    box_min = np.min(point_cloud, axis=0)
    box_max = np.max(point_cloud, axis=0)
    center = (box_min + box_max) * 0.5
    half_size = (box_max - box_min) * 0.5
    half_size = np.maximum(half_size, tol)
    box_min = center - half_size
    box_max = center + half_size
    return {"box_min": box_min, "box_max": box_max}


def distance_aabb(bounding_box1, bounding_box2):
    lower = np.maximum(bounding_box1["box_min"], bounding_box2["box_min"])
    upper = np.minimum(bounding_box1["box_max"], bounding_box2["box_max"])
    delta = np.maximum(0.0, lower - upper)
    return np.linalg.norm(delta)

def distance_lower_bound(primitive1: Spline, primitive2: Spline, sA_intv, sB_intv):
    trimmed_ctrlpts1 = primitive1.convex_hull(*sA_intv)
    trimmed_ctrlpts2 = primitive2.convex_hull(*sB_intv)

    bounding_box1 = update_patch_aabb(trimmed_ctrlpts1)
    bounding_box2 = update_patch_aabb(trimmed_ctrlpts2)
    return distance_aabb(bounding_box1, bounding_box2)

def curve_bounding_capsule(curve1: SplineCurve, curve2: SplineCurve, sA_intv, sB_intv):
    mid_point1 = 0.5 * (sA_intv[0] + sA_intv[1])
    mid_point2 = 0.5 * (sB_intv[0] + sB_intv[1])
    curvature1 = np.linalg.norm(curve1.second_derivative(mid_point1))
    curvature2 = np.linalg.norm(curve2.second_derivative(mid_point2))
    radius1 = 0.125 * curvature1 * abs(sA_intv[1] - sA_intv[0])**2
    radius2 = 0.125 * curvature2 * abs(sB_intv[1] - sB_intv[0])**2
    start_point1 = curve1.single_point(sA_intv[0])
    end_point1 = curve1.single_point(sA_intv[1])
    start_point2 = curve2.single_point(sB_intv[0])
    end_point2 = curve2.single_point(sB_intv[1])
    return capsule_capsule_distance((start_point1, end_point1), radius1, (start_point2, end_point2), radius2)
    
def curve_curve_distance(curve1: SplineCurve, curve2: SplineCurve, residual=1e-14, max_newton_iter=50, tol=1e-8, max_bisection_iter=2000000):
    if curve1.degree == 2 and curve2.degree == 2:
        distance, pairs = curve_curve_second_order_distance_bisection(curve1, curve2, tol=tol, max_bisection_iter=max_bisection_iter)
        distance, pairs = curve_curve_distance_newton_iteration(curve1, curve2, initial_guess=pairs, residual=residual, max_newton_iter=max_newton_iter, tol=tol)
        return distance, pairs
    else:
        distance, pairs = curve_curve_distance_bisection(curve1, curve2, tol=tol, max_bisection_iter=max_bisection_iter)
        distance, pairs = curve_curve_distance_newton_iteration(curve1, curve2, initial_guess=pairs, residual=residual, max_newton_iter=max_newton_iter, tol=tol)
    return distance, pairs

def curve_curve_second_order_distance_bisection(curve1: SplineCurve, curve2: SplineCurve, tol=1e-6, max_bisection_iter=20000):
    distance, pairs = np.inf, (np.nan, np.nan)
    for elem1 in curve1.element:
        for elem2 in curve2.element:
            queue = []
            if curve_bounding_capsule(curve1, curve2, (elem1[0], elem1[1]), (elem2[0], elem2[1])) < distance:
                queue = [((elem1[0], elem1[1]), (elem2[0], elem2[1]), max(elem1[1] - elem1[0], elem2[1] - elem2[0]))]
            iter = 1
            while queue:
                sA_intv, sB_intv, width = queue.pop()
                center_sA = 0.5 * (sA_intv[0] + sA_intv[1])
                center_sB = 0.5 * (sB_intv[0] + sB_intv[1])

                if max(abs(sA_intv[1] - sA_intv[0]) > tol, abs(sB_intv[1] - sB_intv[0])) > tol:
                    lower_bound = curve_bounding_capsule(curve1, curve2, sA_intv, sB_intv)
                    if lower_bound >= distance:
                        continue

                    if abs(sA_intv[1] - sA_intv[0]) > abs(sB_intv[1] - sB_intv[0]):
                        queue.append(((sA_intv[0], center_sA), sB_intv, max(abs(center_sA - sA_intv[0]), abs(sB_intv[1] - sB_intv[0]))))
                        queue.append(((center_sA, sA_intv[1]), sB_intv, max(abs(sA_intv[1] - center_sA), abs(sB_intv[1] - sB_intv[0]))))
                    else:
                        queue.append((sA_intv, (sB_intv[0], center_sB), max(abs(sA_intv[1] - sA_intv[0]), abs(center_sB - sB_intv[0]))))
                        queue.append((sA_intv, (center_sB, sB_intv[1]), max(abs(sA_intv[1] - sA_intv[0]), abs(sB_intv[1] - center_sB))))
                    queue = sorted(queue, key=lambda x: x[2], reverse=False)
                pointA, pointB = curve1.single_point(center_sA), curve2.single_point(center_sB)
                dist = np.linalg.norm(pointA - pointB)
                if dist < distance:
                    distance = dist
                    pairs = (center_sA, center_sB)
                iter += 1

            if iter > max_bisection_iter:
                print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
                break
    return distance, pairs

def curve_curve_distance_bisection(curve1: SplineCurve, curve2: SplineCurve, tol=1e-6, max_bisection_iter=20000):
    distance, pairs = np.inf, (np.nan, np.nan)
    for elem1 in curve1.element:
        for elem2 in curve2.element:
            queue = []
            if distance_lower_bound(curve1, curve2, (elem1[0], elem1[1]), (elem2[0], elem2[1])) < distance:
                queue = [((elem1[0], elem1[1]), (elem2[0], elem2[1]), max(elem1[1] - elem1[0], elem2[1] - elem2[0]))]
    
            iter = 1
            distance, pairs = np.linalg.norm(curve1.single_point(0.5) - curve2.single_point(0.5)), (0.5, 0.5)
            while queue:
                sA_intv, sB_intv, width = queue.pop()
                center_sA = 0.5 * (sA_intv[0] + sA_intv[1])
                center_sB = 0.5 * (sB_intv[0] + sB_intv[1])

                if max(abs(sA_intv[1] - sA_intv[0]) > tol, abs(sB_intv[1] - sB_intv[0])) > tol:
                    lower_bound = distance_lower_bound(curve1, curve2, sA_intv, sB_intv)
                    if lower_bound >= distance:
                        continue

                    if abs(sA_intv[1] - sA_intv[0]) > abs(sB_intv[1] - sB_intv[0]):
                        queue.append(((sA_intv[0], center_sA), sB_intv, max(abs(center_sA - sA_intv[0]), abs(sB_intv[1] - sB_intv[0]))))
                        queue.append(((center_sA, sA_intv[1]), sB_intv, max(abs(sA_intv[1] - center_sA), abs(sB_intv[1] - sB_intv[0]))))
                    else:
                        queue.append((sA_intv, (sB_intv[0], center_sB), max(abs(sA_intv[1] - sA_intv[0]), abs(center_sB - sB_intv[0]))))
                        queue.append((sA_intv, (center_sB, sB_intv[1]), max(abs(sA_intv[1] - sA_intv[0]), abs(sB_intv[1] - center_sB))))
                    queue = sorted(queue, key=lambda x: x[2], reverse=False)
                pointA, pointB = curve1.single_point(center_sA), curve2.single_point(center_sB)
                dist = np.linalg.norm(pointA - pointB)
                if dist < distance:
                    distance = dist
                    pairs = (center_sA, center_sB)
                    tangent1 = curve1.tangent(center_sA)
                    tangent2 = curve2.tangent(center_sB)
                    if max(abs(tangent1.dot(pointA - pointB)), abs(tangent2.dot(pointA - pointB))) < tol:
                        break
                iter += 1

                if iter > max_bisection_iter:
                    print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
                    break
    return distance, pairs

def curve_curve_distance_newton_iteration(curve1: SplineCurve, curve2: SplineCurve, initial_guess, rho=10., residual=1e-10, max_outer_iter=100, max_newton_iter=50, tol=1e-8):
    Aknot, Bknot = initial_guess

    lambda_vec = np.zeros(4)
    outer_iter = 0
    outer_converged = False
    while outer_iter < max_outer_iter:
        newton_iter = 0
        Aknot0, Bknot0 = Aknot, Bknot
        newton_converged = False
        while newton_iter < max_newton_iter:
            point1, tangent1, curvature1 = curve1.single_point(Aknot), curve1.dxdknot(Aknot), curve1.d2xd2knot(Aknot)[0]
            point2, tangent2, curvature2 = curve2.single_point(Bknot), curve2.dxdknot(Bknot), curve2.d2xd2knot(Bknot)[0]
            direction = point1 - point2
            if np.linalg.norm(direction) < tol:
                newton_converged = True
                break

            gradient = np.array([np.dot(direction, tangent1), -np.dot(direction, tangent2)])
            hessian = np.array([[np.dot(tangent1, tangent1) + np.dot(direction, curvature1), -np.dot(tangent1, tangent2)],
                                [-np.dot(tangent1, tangent2), np.dot(tangent2, tangent2) - np.dot(direction, curvature2)]])
            
            # g1 = u >= 0, g2 = 1 - u >= 0, g3 = v >= 0, g4 = 1 - v >= 0
            # [[∂g1/∂u, ∂g1/∂v], [∂g2/∂u, ∂g2/∂v], [∂g3/∂u, ∂g3/∂v], [∂g4/∂u, ∂g4/∂v]]
            constraint = np.array([-Aknot, Aknot - 1., -Bknot, Bknot - 1.])
            gradient_constraint = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
            for i in range(4):
                terms = constraint[i] + lambda_vec[i] / rho
                if terms > 0.:
                    gradient += rho * terms * gradient_constraint[i]
                    hess += rho * np.outer(gradient_constraint[i], gradient_constraint[i])
                    
            if np.linalg.norm(gradient) < tol:
                newton_converged = True
                break

            if abs(Aknot) < 1e-12 or abs(Aknot - 1.) < 1e-12:
                gradient[0] = 0.
                hessian[0, :] = 0.
                hessian[:, 0] = 0.
                hessian[0, 0] = 1.
            if abs(Bknot) < 1e-12 or abs(Bknot - 1.) < 1e-12:
                gradient[1] = 0.
                hessian[1, :] = 0.
                hessian[:, 1] = 0.
                hessian[1, 1] = 1.

            eigvals = np.linalg.eigvalsh(hessian) 
            lambda_min = np.min(np.abs(eigvals))
            norm_hessian = np.linalg.norm(hessian, ord=2)
            delta = np.zeros(2)
            if lambda_min > 1e-8 * norm_hessian:
                delta = np.linalg.solve(hessian, gradient)

            alphaA = 1.
            alphaB = 1.
            if delta[0] != 0.:
                alphaA = -(1. - Aknot) / delta[0] if delta[0] < 0. else Aknot / delta[0]
            if delta[1] != 0.:
                alphaB = -(1. - Bknot) / delta[1] if delta[1] < 0. else Bknot / delta[1]
            alpha = min(alphaA, alphaB, 1.0)
            tempA = Aknot - alpha * delta[0]
            tempB = Bknot - alpha * delta[1]
            f_current = direction.dot(direction)

            for i in range(4):
                terms = max(0., constraint[i] + lambda_vec[i] / rho)
                f_current += 0.5 * rho * terms * terms
            armijo_condition = f_current + 1e-4 * alpha * (gradient[0] * delta[0] + gradient[1] * delta[1])

            iteration_count = 0
            f_update = f_current
            while f_update > armijo_condition and alpha > 1e-8 and iteration_count < 100:
                alpha *= 0.5
                tempA = Aknot - alpha * delta[0]
                tempB = Bknot - alpha * delta[1]
                iteration_count += 1; 
                
                point1 = curve1.single_point(tempA)
                point2 = curve2.single_point(tempB)
                f_update = pow(np.linalg.norm(point1 - point2), 2)
                constraint = np.array([-tempA, tempA - 1., -tempB, tempB - 1.])
                for i in range(4):
                    terms = max(0.0, constraint[i] + lambda_vec[i] / rho)
                    f_update += 0.5 * rho * terms * terms
            
            Aknot = tempA
            Bknot = tempB

            if alpha * np.linalg.norm(delta) < residual:
                newton_converged = True
                break
            newton_iter += 1

        constraint0 = np.array([-Aknot0, Aknot0 - 1., -Bknot0, Bknot0 - 1.])
        constraint = np.array([-Aknot, Aknot - 1., -Bknot, Bknot - 1.])
        max_violation = 0.0
        pre_violation = 0.0
        for i in range(4):
            lambda_vec[i] = max(0.0, lambda_vec[i] + rho * constraint[i])
            max_violation = max(max_violation, abs(max(0.0, constraint[i])))
            pre_violation = max(pre_violation, abs(max(0.0, constraint0[i])))

        if max_violation > 0.25 * pre_violation or not newton_converged:
            rho = min(1e8, rho * 10.0)

        if max_violation < tol and pow((Aknot - Aknot0), 2) + pow((Bknot - Bknot0), 2) < tol:
            outer_converged = True
            break

    assert outer_converged is True

    distance = np.linalg.norm(curve1.single_point(Aknot) - curve2.single_point(Bknot))
    pairs = (Aknot, Bknot)
    return distance, pairs

def surface_surface_distance(surface1: SplineSurface, surface2: SplineSurface, rho=10., residual=1e-14, max_outer_iter=100, max_newton_iter=50, tol=1e-8, max_bisection_iter=2000000):
    if surface1.degree_u == 2 and surface1.degree_v == 2 and surface2.degree_u == 2 and surface2.degree_v == 2:
        pairs = ((0.5, 0.5), (0.5, 0.5))
        distance, pairs = surface_surface_second_order_distance_bisection(surface1, surface2, tol=tol, max_bisection_iter=max_bisection_iter)
        distance, pairs = surface_surface_distance_newton_iteration(surface1, surface2, initial_guess=pairs, rho=rho, residual=residual, max_outer_iter=max_outer_iter, max_newton_iter=max_newton_iter, tol=tol)
        return distance, pairs
    else:
        distance, pairs = surface_surface_distance_bisection(surface1, surface2, tol=tol, max_bisection_iter=max_bisection_iter)
        distance, pairs = surface_surface_distance_newton_iteration(surface1, surface2, initial_guess=pairs, rho=rho, residual=residual, max_outer_iter=max_outer_iter, max_newton_iter=max_newton_iter, tol=tol)
        return distance, pairs
    
def max_curvature_uu(dim, surface: SplineSurface, uknot_min, uknot_max, vknot_min, vknot_max, tol=1e-8, visualize=False):
    gr = 0.5 * (5 ** 0.5 - 1)

    def fuu(u, v): return abs(surface.second_derivative(u, v)[0][int(dim)])
    initial_u = 0.5 * (uknot_min + uknot_max)
    a, b = vknot_min, vknot_max
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    while abs(b - a) > tol:
        if fuu(initial_u, c) < fuu(initial_u, d):
            a = c
            c = d
            d = a + gr * (b - a)
        else:
            b = d
            d = c
            c = b - gr * (b - a)
    max_ucurvature = fuu(initial_u, 0.5 * (a + b))
    if visualize:
        plot_nd_slice(fuu, [(uknot_min, uknot_max), (vknot_min, vknot_max)], mark_points=[(initial_u, 0.5 * (a + b))])
    return max_ucurvature

def max_curvature_vv(dim, surface: SplineSurface, uknot_min, uknot_max, vknot_min, vknot_max, tol=1e-8, visualize=False):
    gr = 0.5 * (5 ** 0.5 - 1)

    def fvv(u, v): return abs(surface.second_derivative(u, v)[1][int(dim)])
    initial_v = 0.5 * (vknot_min + vknot_max)
    a, b = uknot_min, uknot_max
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    while abs(b - a) > tol:
        if fvv(c, initial_v) < fvv(d, initial_v):
            a = c
            c = d
            d = a + gr * (b - a)
        else:
            b = d
            d = c
            c = b - gr * (b - a)
    max_vcurvature = fvv(0.5 * (a + b), initial_v)
    if visualize:
        plot_nd_slice(fvv, [(uknot_min, uknot_max), (vknot_min, vknot_max)], mark_points=[(0.5 * (a + b), initial_v)])
    return max_vcurvature

def max_curvature_uv(dim, surface: SplineSurface, uknot_min, uknot_max, vknot_min, vknot_max, tol=1e-8, visualize=False):
    def fuv(u, v): return abs(surface.second_derivative(u, v)[2][int(dim)])
    max_uvcurvature = max(fuv(uknot_min, vknot_min),
                          fuv(uknot_max, vknot_min),
                          fuv(uknot_min, vknot_max),
                          fuv(uknot_max, vknot_max))
    if visualize:
        plot_nd_slice(fuv, [(uknot_min, uknot_max), (vknot_min, vknot_max)])
    return max_uvcurvature

def max_curvature(surface: SplineSurface, uknot_min, uknot_max, vknot_min, vknot_max, tol=1e-8, visualize=False):
    M1 = max(max_curvature_uu(0, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
             max_curvature_uu(1, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
             max_curvature_uu(2, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize))
    M2 = max(max_curvature_vv(0, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
             max_curvature_vv(1, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
             max_curvature_vv(2, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize))
    M3 = max(max_curvature_uv(0, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
             max_curvature_uv(1, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
             max_curvature_uv(2, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize))
    return M1, M2, M3

def surface_bounding_slab(surface1: SplineSurface, surface2: SplineSurface, sA_intv, sB_intv):
    uA, vA = sA_intv[0], sA_intv[1]
    uB, vB = sB_intv[0], sB_intv[1]
    M1A, M2A, M3A = max_curvature(surface1, *uA, *vA)
    M1B, M2B, M3B = max_curvature(surface2, *uB, *vB)
    radius1 = 0.125 * (M1A * abs(uA[1] - uA[0])**2 + M2A * abs(vA[1] - vA[0])**2 + 2 * M3A * abs(uA[1] - uA[0]) * abs(vA[1] - vA[0]))
    radius2 = 0.125 * (M1B * abs(uB[1] - uB[0])**2 + M2B * abs(vB[1] - vB[0])**2 + 2 * M3B * abs(uB[1] - uB[0]) * abs(vB[1] - vB[0]))
    '''if flag==1:
        visualize((surface1.single_point(uA[0], vA[0]), surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1])), (surface2.single_point(uB[0], vB[0]), surface2.single_point(uB[1], vB[0]), surface2.single_point(uB[0], vB[1])))
        visualize((surface1.single_point(uA[0], vA[0]), surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1])), (surface2.single_point(uB[1], vB[0]), surface2.single_point(uB[0], vB[1]), surface2.single_point(uB[1], vB[1])))
        visualize((surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1]), surface1.single_point(uA[1], vA[1])), (surface2.single_point(uB[0], vB[0]), surface2.single_point(uB[1], vB[0]), surface2.single_point(uB[0], vB[1])))
        visualize((surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1]), surface1.single_point(uA[1], vA[1])), (surface2.single_point(uB[1], vB[0]), surface2.single_point(uB[0], vB[1]), surface2.single_point(uB[1], vB[1])))'''
    patch1 = [surface1.single_point(uA[0], vA[0]), surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1]), surface1.single_point(uA[1], vA[1])]
    patch2 = [surface2.single_point(uB[0], vB[0]), surface2.single_point(uB[1], vB[0]), surface2.single_point(uB[0], vB[1]), surface2.single_point(uB[1], vB[1])]
    dist = min(triangle_slab_triangle_slab_distance((patch1[0], patch1[1], patch1[2]), radius1, (patch2[0], patch2[1], patch2[2]), radius2),
               triangle_slab_triangle_slab_distance((patch1[0], patch1[1], patch1[2]), radius1, (patch2[1], patch2[2], patch2[3]), radius2),
               triangle_slab_triangle_slab_distance((patch1[1], patch1[2], patch1[3]), radius1, (patch2[0], patch2[1], patch2[2]), radius2),
               triangle_slab_triangle_slab_distance((patch1[1], patch1[2], patch1[3]), radius1, (patch2[1], patch2[2], patch2[3]), radius2)
               )
    return dist, min(np.sqrt(4. * radius1 * abs(dist) + radius1 * radius1), np.sqrt(4. * radius2 * abs(dist) + radius2 * radius2))

def surface_surface_second_order_distance_bisection(surface1: SplineSurface, surface2: SplineSurface, tol=1e-8, max_bisection_iter=2000):
    distance, pairs = np.inf, ((np.nan, np.nan), (np.nan, np.nan))
    for elem1_u in surface1.element_u:
        for elem1_v in surface1.element_v:
            for elem2_u in surface2.element_u:
                for elem2_v in surface2.element_v:
                    queue = Queue()
                    if surface_bounding_slab(surface1, surface2, ((elem1_u[0], elem1_u[1]), (elem1_v[0], elem1_v[1])), ((elem2_u[0], elem2_u[1]), (elem2_v[0], elem2_v[1])))[0] < distance:
                        queue.append(((elem1_u[0], elem1_u[1]), (elem1_v[0], elem1_v[1])), ((elem2_u[0], elem2_u[1]), (elem2_v[0], elem2_v[1])))
                        dist = np.linalg.norm(surface1.single_point(0.5 * (elem1_u[0] + elem1_u[1]), 0.5 * (elem1_v[0] + elem1_v[1])) - surface2.single_point(0.5 * (elem2_u[0] + elem2_u[1]), 0.5 * (elem2_v[0] + elem2_v[1])))
                        if dist < distance:
                            dist = distance
                            pairs = ((0.5 * (elem1_u[0] + elem1_u[1]), 0.5 * (elem1_v[0] + elem1_v[1])), (0.5 * (elem2_u[0] + elem2_u[1]), 0.5 * (elem2_v[0] + elem2_v[1])))

                    iter = 1
                    while queue.size() > 0.:
                        (uA, vA), (uB, vB) = queue.pop()
                        uc = 0.5 * (uA[0] + uA[1])
                        vc = 0.5 * (vA[0] + vA[1])
                        sc = 0.5 * (uB[0] + uB[1])
                        tc = 0.5 * (vB[0] + vB[1])
                        
                        if max(abs(uA[1] - uA[0]), abs(vA[1] - vA[0]), abs(uB[1] - uB[0]), abs(vB[1] - vB[0])) > tol:
                            lower_bound, error = surface_bounding_slab(surface1, surface2, ((uA[0], uA[1]), (vA[0], vA[1])), ((uB[0], uB[1]), (vB[0], vB[1])))
                            #print(uc, vc, sc, tc)
                            if lower_bound - distance > -1e-2 * distance:
                                continue
                            if error > 1e-2 * distance:
                                queue = split_bounding_box(uA, vA, uB, vB, uc, vc, sc, tc, queue)

                        knotsA = [(uc, vc), (uA[0], vA[0]), (uA[0], vA[1]), (uA[1], vA[0]), (uA[1], vA[1])]
                        knotsB = [(sc, tc), (uB[0], vB[0]), (uB[0], vB[1]), (uB[1], vB[0]), (uB[1], vB[1])]
                        values = [np.linalg.norm(surface1.single_point(ua, va) - surface2.single_point(ub, vb)) for (ua, va) in knotsA for (ub, vb) in knotsB]
                        min_idx = min(range(len(values)), key=lambda i: values[i])
                        dist = values[min_idx]
                        uv = knotsA[int(min_idx // 5)]
                        st = knotsB[int(min_idx % 5)]
                        if dist < distance:
                            distance = dist
                            pairs = (uv, st)

                        '''uv = (uc, vc)
                        st = (sc, tc)
                        dist, pair = surface_surface_distance_newton_iteration(surface1, surface2, initial_guess=(uv, st), tol=tol, uboundary=uA, vboundary=vA, sboundary=uB, tboundary=vB)
                        if dist < distance:
                            distance = dist
                            pairs = pair'''

                        iter += 1

                        if iter > max_bisection_iter:
                            print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
                            break
    return distance, pairs

def surface_surface_distance_bisection(surface1: SplineSurface, surface2: SplineSurface, tol=1e-8, max_bisection_iter=2000):
    queue = [(((surface1.knot_vector_u[surface1.degree_u], surface1.knot_vector_u[-1-surface1.degree_u]), (surface1.knot_vector_v[surface1.degree_v], surface1.knot_vector_v[-1-surface1.degree_v])), 
             ((surface2.knot_vector_u[surface2.degree_u], surface2.knot_vector_u[-1-surface2.degree_u]), (surface2.knot_vector_v[surface2.degree_v], surface2.knot_vector_v[-1-surface2.degree_v])), 1.)]

    iter = 1
    distance, pairs = np.linalg.norm(surface1.single_point(0.5, 0.5) - surface2.single_point(0.5, 0.5)), ((0.5, 0.5), (0.5, 0.5))
    while len(queue) > 0.:
        (uA, vA), (uB, vB), width = queue.pop()
        uc, vc = 0.5 * (uA[0] + uA[1]), 0.5 * (vA[0] + vA[1])
        sc, tc = 0.5 * (uB[0] + uB[1]), 0.5 * (vB[0] + vB[1])

        if max(abs(uA[1] - uA[0]), abs(vA[1] - vA[0]), abs(uB[1] - uB[0]), abs(vB[1] - vB[0])) > tol:
            lower_bound = distance_lower_bound(surface1, surface2, ((uA[0], vA[0]), (uA[1], vA[1])), ((uB[0], vB[0]), (uB[1], vB[1])))
            if lower_bound >= distance:
                continue
            
            queue1, queue2 = split_bounding_box(uA, vA, uB, vB, uc, vc, sc, tc)
            queue.append(queue1)
            queue.append(queue2)
            queue = sorted(queue, key=lambda x: x[2], reverse=False)
        pointA, pointB = surface1.single_point(uc, vc), surface2.single_point(sc, tc)
        dist = np.linalg.norm(pointA - pointB)
        if dist < distance:
            distance = dist
            pairs = ((uc, vc), (sc, tc))
            tangent1_u, tangent1_v = surface1.tangent(uc, vc)
            tangent2_u, tangent2_v = surface2.tangent(sc, tc)
            if max(abs(tangent1_u.dot(pointA - pointB)), abs(tangent1_v.dot(pointA - pointB)), abs(tangent2_u.dot(pointA - pointB)), abs(tangent2_v.dot(pointA - pointB))) < tol:
                break
        iter += 1

        if iter > max_bisection_iter:
            print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
            break
    return distance, pairs


def split_bounding_box(u1r, v1r, u2r, v2r, umid1, vmid1, umid2, vmid2, queue: Queue):
    u1min, u1max = u1r
    v1min, v1max = v1r
    u2min, u2max = u2r
    v2min, v2max = v2r

    '''queue.append(((u1min, umid1), (v1min, vmid1)), ((u2min, umid2), (v2min, vmid2)))
    queue.append(((u1min, umid1), (v1min, vmid1)), ((u2min, umid2), (vmid2, v2max)))
    queue.append(((u1min, umid1), (v1min, vmid1)), ((umid2, u2max), (v2min, vmid2)))
    queue.append(((u1min, umid1), (v1min, vmid1)), ((umid2, u2max), (vmid2, v2max)))
    queue.append(((u1min, umid1), (vmid1, v1max)), ((u2min, umid2), (v2min, vmid2)))
    queue.append(((u1min, umid1), (vmid1, v1max)), ((u2min, umid2), (vmid2, v2max)))
    queue.append(((u1min, umid1), (vmid1, v1max)), ((umid2, u2max), (v2min, vmid2)))
    queue.append(((u1min, umid1), (vmid1, v1max)), ((umid2, u2max), (vmid2, v2max)))
    queue.append(((umid1, u1max), (v1min, vmid1)), ((u2min, umid2), (v2min, vmid2)))
    queue.append(((umid1, u1max), (v1min, vmid1)), ((u2min, umid2), (vmid2, v2max)))
    queue.append(((umid1, u1max), (v1min, vmid1)), ((umid2, u2max), (v2min, vmid2)))
    queue.append(((umid1, u1max), (v1min, vmid1)), ((umid2, u2max), (vmid2, v2max)))
    queue.append(((umid1, u1max), (vmid1, v1max)), ((u2min, umid2), (v2min, vmid2)))
    queue.append(((umid1, u1max), (vmid1, v1max)), ((u2min, umid2), (vmid2, v2max)))
    queue.append(((umid1, u1max), (vmid1, v1max)), ((umid2, u2max), (v2min, vmid2)))
    queue.append(((umid1, u1max), (vmid1, v1max)), ((umid2, u2max), (vmid2, v2max)))
    return queue'''

    lengths = {'u1': u1max - u1min, 'v1': v1max - v1min, 'u2': u2max - u2min, 'v2': v2max - v2min}
    max_dir = max(lengths, key=lengths.get)
    if max_dir == 'u1':
        queue.append(((u1min, umid1), (v1min, v1max)), ((u2min, u2max), (v2min, v2max)))
        queue.append(((umid1, u1max), (v1min, v1max)), ((u2min, u2max), (v2min, v2max)))
        return queue
    elif max_dir == 'v1':
        queue.append(((u1min, u1max), (v1min, vmid1)), ((u2min, u2max), (v2min, v2max)))
        queue.append(((u1min, u1max), (vmid1, v1max)), ((u2min, u2max), (v2min, v2max)))
        return queue
    elif max_dir == 'u2':
        queue.append(((u1min, u1max), (v1min, v1max)), ((u2min, umid2), (v2min, v2max)))
        queue.append(((u1min, u1max), (v1min, v1max)), ((umid2, u2max), (v2min, v2max)))
        return queue
    elif max_dir == 'v2':
        queue.append(((u1min, u1max), (v1min, v1max)), ((u2min, u2max), (v2min, vmid2)))
        queue.append(((u1min, u1max), (v1min, v1max)), ((u2min, u2max), (vmid2, v2max)))
        return queue


def surface_surface_distance_newton_iteration(surface1: SplineSurface, surface2: SplineSurface, initial_guess, uboundary=[0., 1.], vboundary=[0., 1.], sboundary=[0., 1.], tboundary=[0., 1.], residual=1e-14, rho=10, max_outer_iter=100, max_newton_iter=500, tol=1e-8):
    Aknot, Bknot = initial_guess
    uknot, vknot = Aknot
    sknot, tknot = Bknot

    lambda_vec = np.zeros(8)
    outer_iter = 0
    outer_converged = False
    while outer_iter < max_outer_iter:
        newton_iter = 0
        uknot0, vknot0 = uknot, vknot
        sknot0, tknot0 = sknot, tknot
        newton_converged = False
        while newton_iter < max_newton_iter:
            point1, point2 = surface1.single_point(uknot, vknot), surface2.single_point(sknot, tknot)
            tangent_u, tangent_v = surface1.dxdknot(uknot, vknot)
            tangent_s, tangent_t = surface2.dxdknot(sknot, tknot)
            curvature_uu, curvature_vv, curvature_uv = surface1.d2xd2knot(uknot, vknot)
            curvature_ss, curvature_tt, curvature_st = surface2.d2xd2knot(sknot, tknot)
            direction = point1 - point2
            if np.linalg.norm(direction) < residual:
                newton_converged = 1
                break
            
            hess_uu = 2. * (np.dot(direction, curvature_uu) + np.dot(tangent_u, tangent_u))
            hess_vv = 2. * (np.dot(direction, curvature_vv) + np.dot(tangent_v, tangent_v))
            hess_ss = 2. * (-np.dot(direction, curvature_ss) + np.dot(tangent_s, tangent_s))
            hess_tt = 2. * (-np.dot(direction, curvature_tt) + np.dot(tangent_t, tangent_t))

            hess_uv = 2. * (np.dot(direction, curvature_uv) + np.dot(tangent_u, tangent_v))
            hess_us = -2. * np.dot(tangent_u, tangent_s)
            hess_ut = -2. * np.dot(tangent_u, tangent_t)
            hess_vs = -2. * np.dot(tangent_v, tangent_s)
            hess_vt = -2. * np.dot(tangent_v, tangent_t)
            hess_st = 2. * (-np.dot(direction, curvature_st) + np.dot(tangent_s, tangent_t))

            gradient = np.array([np.dot(direction, tangent_u), np.dot(direction, tangent_v), -np.dot(direction, tangent_s), -np.dot(direction, tangent_t)])
            hessian = np.array([[hess_uu, hess_uv, hess_us, hess_ut], 
                                [hess_uv, hess_vv, hess_vs, hess_vt],
                                [hess_us, hess_vs, hess_ss, hess_st],
                                [hess_ut, hess_vt, hess_st, hess_tt]])
            
            # g1 = u >= 0, g2 = 1 - u >= 0, g3 = v >= 0, g4 = 1 - v >= 0
            # [[∂g1/∂u, ∂g1/∂v, ∂g1/∂s, ∂g1/∂t], [∂g2/∂u, ∂g2/∂v, ∂g2/∂s, ∂g2/∂t], [∂g3/∂u, ∂g3/∂v, ∂g3/∂s, ∂g3/∂t], [∂g4/∂u, ∂g4/∂v, ∂g4/∂s, ∂g4/∂t]]
            constraint = np.array([uboundary[0] - uknot, uknot - uboundary[1], vboundary[0] - vknot, vknot - vboundary[1], sboundary[0] - sknot, sknot - sboundary[1], tboundary[0] - tknot, tknot - tboundary[1]])
            gradient_constraint = np.array([[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0]])
            
            for i in range(8):
                terms = constraint[i] + lambda_vec[i] / rho
                if terms > 0.:
                    gradient += rho * terms * gradient_constraint[i]
                    hessian += rho * np.outer(gradient_constraint[i], gradient_constraint[i])

            #if np.linalg.norm(gradient) < residual:
            #    newton_converged = 2
            #    break

            if abs(uknot - uboundary[0]) < 1e-12 or abs(uknot - uboundary[1]) < 1e-12:
                gradient[0] = 0.
                hessian[0, :] = 0.
                hessian[:, 0] = 0.
                hessian[0, 0] = 1.
            if abs(vknot - vboundary[0]) < 1e-12 or abs(vknot - vboundary[1]) < 1e-12:
                gradient[1] = 0.
                hessian[1, :] = 0.
                hessian[:, 1] = 0.
                hessian[1, 1] = 1.
            if abs(sknot - sboundary[0]) < 1e-12 or abs(sknot - sboundary[1]) < 1e-12:
                gradient[2] = 0.
                hessian[2, :] = 0.
                hessian[:, 2] = 0.
                hessian[2, 2] = 1.
            if abs(tknot - tboundary[0]) < 1e-12 or abs(tknot - tboundary[1]) < 1e-12:
                gradient[3] = 0.
                hessian[3, :] = 0.
                hessian[:, 3] = 0.
                hessian[3, 3] = 1.

            eigvals, eigvecs = np.linalg.eigh(hessian) 
            lambda_min = np.min(np.abs(eigvals))
            mask = np.abs(eigvals) > 1e-6
            '''T_s = eigvecs[:, mask]
            grad_r = T_s.T @ gradient
            H_r = T_s.T @ hessian @ T_s
            delta_r = np.linalg.solve(H_r, grad_r)
            delta = T_s @ delta_r'''
            norm_hessian = np.linalg.norm(hessian, ord=2)
            delta = np.zeros(4)
            print(eigvals, direction)
            if lambda_min > 1e-8 * norm_hessian:
                delta = np.linalg.solve(hessian, gradient)
            #print(eigvals);input()

            alphaA = 1.
            alphaB = 1.
            alphaC = 1.
            alphaD = 1.
            if delta[0] != 0.:
                alphaA = -(uboundary[1] - uknot) / delta[0] if delta[0] < 0. else (uknot - uboundary[0]) / delta[0]
            if delta[1] != 0.:
                alphaB = -(vboundary[1] - vknot) / delta[1] if delta[1] < 0. else (vknot - vboundary[0]) / delta[1]
            if delta[2] != 0.:
                alphaC = -(sboundary[1] - sknot) / delta[2] if delta[2] < 0. else (sknot - sboundary[0]) / delta[2]
            if delta[3] != 0.:
                alphaD = -(tboundary[1] - tknot) / delta[3] if delta[3] < 0. else (tknot - tboundary[0]) / delta[3]
            alpha = min(0.999*alphaA, 0.999*alphaB, 0.999*alphaC, 0.999*alphaD, 1.0)
            temp_u = uknot - alpha * delta[0]
            temp_v = vknot - alpha * delta[1]
            temp_s = sknot - alpha * delta[2]
            temp_t = tknot - alpha * delta[3]
            f_current = direction.dot(direction)

            for i in range(8):
                terms = max(0., constraint[i] + lambda_vec[i] / rho)
                f_current += 0.5 * rho * terms * terms
            armijo_condition = f_current + 1e-4 * alpha * (gradient[0] * delta[0] + gradient[1] * delta[1] + gradient[2] * delta[2] + gradient[3] * delta[3])

            iteration_count = 0
            f_update = f_current
            while f_update > armijo_condition and alpha > 1e-8 and iteration_count < 100:
                alpha *= 0.5
                temp_u = uknot - alpha * delta[0]
                temp_v = vknot - alpha * delta[1]
                temp_s = sknot - alpha * delta[2]
                temp_t = tknot - alpha * delta[3]
                iteration_count += 1; 
                
                point1 = surface1.single_point(temp_u, temp_v)
                point2 = surface2.single_point(temp_s, temp_t)
                f_update = pow(np.linalg.norm(point1 - point2), 2)
                constraint = np.array([uboundary[0] - temp_u, temp_u - uboundary[1], vboundary[0] - temp_v, temp_v - vboundary[1], sboundary[0] - temp_s, temp_s - sboundary[1], tboundary[0] - temp_t, temp_t - tboundary[1]])
                for i in range(4):
                    terms = max(0.0, constraint[i] + lambda_vec[i] / rho)
                    f_update += 0.5 * rho * terms * terms
            
            uknot = temp_u
            vknot = temp_v
            sknot = temp_s
            tknot = temp_t

            if alpha * np.linalg.norm(delta) < residual:
                newton_converged = 3
                break
            newton_iter += 1

        constraint0 = np.array([uboundary[0] - uknot0, uknot0 - uboundary[1], vboundary[0] - vknot0, vknot0 - vboundary[1], sboundary[0] - sknot0, sknot0 - sboundary[1], tboundary[0] - tknot0, tknot0 - tboundary[1]])
        constraint = np.array([uboundary[0] - uknot, uknot - uboundary[1], vboundary[0] - vknot, vknot - vboundary[1], sboundary[0] - sknot, sknot - sboundary[1], tboundary[0] - tknot, tknot - tboundary[1]])
        max_violation = 0.0
        pre_violation = 0.0
        for i in range(8):
            lambda_vec[i] = max(0.0, lambda_vec[i] + rho * constraint[i])
            max_violation = max(max_violation, abs(max(0.0, constraint[i])))
            pre_violation = max(pre_violation, abs(max(0.0, constraint0[i])))

        if max_violation > 0.25 * pre_violation or not newton_converged:
            rho = min(1e15, rho * 10.0)
            
        if max_violation < tol and pow((uknot - uknot0), 2) + pow((vknot - vknot0), 2) + pow((sknot - sknot0), 2) + pow((tknot - tknot0), 2) < tol:
            outer_converged = True
            break

        outer_iter += 1
    assert outer_converged is True

    distance = np.linalg.norm(surface1.single_point(uknot, vknot) - surface2.single_point(sknot, tknot))
    pairs = ((uknot, vknot), (sknot, tknot))
    return distance, pairs

def surface_ground_distance(surface: SplineSurface, z_ground, rho=10., residual=1e-14, max_outer_iter=100, max_newton_iter=500, tol=1e-8, max_bisection_iter=200000):
    if surface.degree_u == 2 and surface.degree_v == 2:
        distance, pairs = surface_ground_second_order_distance_bisection(surface, z_ground, tol=tol, max_bisection_iter=max_bisection_iter)
        distance, pairs = surface_ground_distance_newton_iteration(surface, z_ground, initial_guess=pairs, rho=rho, residual=residual, max_outer_iter=max_outer_iter, max_newton_iter=max_newton_iter, tol=tol)
        return distance, pairs
    else:
        distance, pairs = surface_ground_distance_bisection(surface, z_ground, tol=tol, max_bisection_iter=max_bisection_iter)
        distance, pairs = surface_ground_distance_newton_iteration(surface, z_ground, initial_guess=pairs, rho=rho, residual=residual, max_outer_iter=max_outer_iter, max_newton_iter=max_newton_iter, tol=tol)
        return distance, pairs

def surface_ground_bounding_slab(surface1: SplineSurface, z_ground, sA_intv, visualize=False):
    uA, vA = sA_intv[0], sA_intv[1]
    curvature1_u, curvature1_v, curvature1_uv = max_curvature(surface1, *uA, *vA)
    radiusA = 0.125 * (curvature1_u * abs(uA[1] - uA[0])**2 + curvature1_v * abs(vA[1] - vA[0])**2 + 2 * curvature1_uv * abs(uA[1] - uA[0]) * abs(vA[1] - vA[0]))
    if visualize:
        surface1.visualize(knot_u=np.linspace(uA[0], uA[1]), knot_v=np.linspace(vA[0], vA[1]), bounding_volume=[{'point1':surface1.single_point(uA[0], vA[0]), 'point2':surface1.single_point(uA[1], vA[0]), 'point3':surface1.single_point(uA[0], vA[1]), "radius": radiusA}])
    return min(triangle_slab_plane_distance((surface1.single_point(uA[0], vA[0]), surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1])), radiusA, z_ground),
               triangle_slab_plane_distance((surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1]), surface1.single_point(uA[1], vA[1])), radiusA, z_ground))

def surface_ground_second_order_distance_bisection(surface1: SplineSurface, z_ground, tol=1e-8, max_bisection_iter=64):
    distance, pairs = np.inf, (np.nan, np.nan)
    for elem1_u in surface1.element_u:
        for elem1_v in surface1.element_v:
            if surface_ground_bounding_slab(surface1, z_ground, ((elem1_u[0], elem1_u[1]), (elem1_v[0], elem1_v[1]))) < distance:
                queue = [(((elem1_u[0], elem1_u[1]), (elem1_v[0], elem1_v[1])), max(elem1_u[1] - elem1_u[0], elem1_v[1] - elem1_v[0]))]

            iter = 1
            while len(queue) > 0.:
                (uA, vA), width = queue.pop()
                uc, vc = 0.5 * (uA[0] + uA[1]), 0.5 * (vA[0] + vA[1])

                if max(abs(uA[1] - uA[0]), abs(vA[1] - vA[0])) > tol:
                    lower_bound = surface_ground_bounding_slab(surface1, z_ground, ((uA[0], uA[1]), (vA[0], vA[1])))
                    if lower_bound >= distance:
                        continue
                    
                    queue1, queue2 = split_single_bounding_box(uA, vA)
                    queue.append(queue1)
                    queue.append(queue2)
                    queue = sorted(queue, key=lambda x: x[1], reverse=False)

                dist, pair = surface_ground_distance_newton_iteration(surface1, z_ground, initial_guess=(uc, vc), uboundary=(uA[0], uA[1]), vboundary=(vA[0], vA[1]))

                if dist < distance:
                    distance = dist
                    pairs = pair
                iter += 1

                if iter > max_bisection_iter:
                    print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
                    break
    return distance, pairs

def surface_ground_distance_bisection(surface1: SplineSurface, z_ground, tol=1e-8, max_bisection_iter=64):
    queue = [(((surface1.knot_vector_u[surface1.degree_u], surface1.knot_vector_u[-1-surface1.degree_u]), (surface1.knot_vector_v[surface1.degree_v], surface1.knot_vector_v[-1-surface1.degree_v])), 1.)]

    iter = 1
    distance, pairs = surface1.single_point(0.5, 0.5)[2] - z_ground, (0.5, 0.5)
    while len(queue) > 0.:
        (uA, vA), width = queue.pop()
        uc, vc = 0.5 * (uA[0] + uA[1]), 0.5 * (vA[0] + vA[1])

        if max(abs(uA[1] - uA[0]) > tol, abs(vA[1] - vA[0])) > tol:
            lower_bound = ground_distance_lower_bound(z_ground, surface1, ((uA[0], vA[0]), (uA[1], vA[1])))
            if lower_bound >= distance:
                continue
            
            queue1, queue2 = split_single_bounding_box(uA, vA)
            queue.append(queue1)
            queue.append(queue2)
            queue = sorted(queue, key=lambda x: x[1], reverse=False)
        dist = abs(surface1.single_point(uc, vc)[2] - z_ground)
        if dist < distance:
            distance = dist
            pairs = (uc, vc)
            tangent_u, tangent_v = surface1.tangent(uc, vc)
            if max(abs(tangent_u.dot(np.array([0., 0., -1]))), abs(tangent_v.dot(np.array([0., 0., -1])))) < tol:
                break
        iter += 1

        if iter > max_bisection_iter:
            print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
            break
    return distance, pairs

def ground_distance_lower_bound(z_ground, primitive1: Spline, sA_intv):
    trimmed_ctrlpts1 = primitive1.convex_hull(*sA_intv)
    bounding_box1 = update_patch_aabb(trimmed_ctrlpts1)
    return min(abs(bounding_box1['box_min'][2] - z_ground), abs(bounding_box1['box_max'][2] - z_ground))

def split_single_bounding_box(u1r, v1r):
    u1min, u1max = u1r
    v1min, v1max = v1r

    lengths = {'u1': u1max - u1min, 'v1': v1max - v1min}
    max_dir = max(lengths, key=lengths.get)
    if max_dir == 'u1':
        umid = 0.5 * (u1min + u1max)
        return (((u1min, umid), (v1min, v1max)), max(umid - u1min, v1max - v1min)), \
               (((umid, u1max), (v1min, v1max)), max(u1max - umid, v1max - v1min))
    elif max_dir == 'v1':
        vmid = 0.5 * (v1min + v1max)
        return (((u1min, u1max), (v1min, vmid)), max(u1max - u1min, vmid - v1min)), \
               (((u1min, u1max), (vmid, v1max)), max(u1max - u1min, v1max - vmid))
    
def surface_ground_distance_newton_iteration(surface1: SplineSurface, z_ground, initial_guess, uboundary=[0., 1.], vboundary=[0., 1.], rho=10., residual=1e-14, max_outer_iter=100, max_newton_iter=50, tol=1e-8):
    lambda_vec = np.zeros(4)
    uknot, vknot = initial_guess

    outer_iter = 0
    outer_converged = False
    while outer_iter < max_outer_iter:
        newton_iter = 0
        uknot0, vknot0 = uknot, vknot
        newton_converged = 0
        while newton_iter < max_newton_iter:
            point1 = surface1.single_point(uknot, vknot)
            tangent_u, tangent_v = surface1.dxdknot(uknot, vknot)
            curvature_uu, curvature_vv, curvature_uv = surface1.d2xd2knot(uknot, vknot)
            direction = point1[2] - z_ground
            if np.linalg.norm(direction) < residual:
                newton_converged = 1
                break
            
            hess_uu = 2. * (direction * curvature_uu[2] + tangent_u[2] * tangent_u[2])
            hess_vv = 2. * (direction * curvature_vv[2] + tangent_v[2] * tangent_v[2])
            hess_uv = 2. * (direction * curvature_uv[2] + tangent_u[2] * tangent_v[2])

            gradient = np.array([direction * tangent_u[2], direction * tangent_v[2]])
            hessian = np.array([[hess_uu, hess_uv], 
                                [hess_uv, hess_vv]])
            
            # g1 = u >= 0, g2 = 1 - u >= 0, g3 = v >= 0, g4 = 1 - v >= 0
            # [[∂g1/∂u, ∂g1/∂v, ∂g1/∂s, ∂g1/∂t], [∂g2/∂u, ∂g2/∂v, ∂g2/∂s, ∂g2/∂t], [∂g3/∂u, ∂g3/∂v, ∂g3/∂s, ∂g3/∂t], [∂g4/∂u, ∂g4/∂v, ∂g4/∂s, ∂g4/∂t]]
            constraint = np.array([uboundary[0] - uknot, uknot - uboundary[1], vboundary[0] - vknot, vknot - vboundary[1]])
            gradient_constraint = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
            
            for i in range(4):
                terms = constraint[i] + lambda_vec[i] / rho
                if terms > 0.:
                    gradient += rho * terms * gradient_constraint[i]
                    hessian += rho * np.outer(gradient_constraint[i], gradient_constraint[i])

            if np.linalg.norm(gradient) < residual:
                newton_converged = 2
                break

            if abs(uknot - uboundary[0]) < 1e-12 or abs(uknot - uboundary[1]) < 1e-12:
                gradient[0] = 0.
                hessian[0, :] = 0.
                hessian[:, 0] = 0.
                hessian[0, 0] = 1.
            if abs(vknot - vboundary[0]) < 1e-12 or abs(vknot - vboundary[1]) < 1e-12:
                gradient[1] = 0.
                hessian[1, :] = 0.
                hessian[:, 1] = 0.
                hessian[1, 1] = 1.

            eigvals = np.linalg.eigvalsh(hessian) 
            lambda_min = np.min(np.abs(eigvals))
            norm_hessian = np.linalg.norm(hessian, ord=2)
            delta = np.zeros(2)
            if lambda_min > 1e-8 * norm_hessian:
                delta = np.linalg.solve(hessian, gradient)

            alphaA = 1.
            alphaB = 1.
            if delta[0] != 0.:
                alphaA = -(uboundary[1] - uknot) / delta[0] if delta[0] < 0. else (uknot - uboundary[0]) / delta[0]
            if delta[1] != 0.:
                alphaB = -(vboundary[1] - vknot) / delta[1] if delta[1] < 0. else (vknot - vboundary[0]) / delta[1]
            alpha = min(alphaA, alphaB, 1.0)
            temp_u = uknot - alpha * delta[0]
            temp_v = vknot - alpha * delta[1]

            f_current = direction * direction
            for i in range(4):
                terms = max(0., constraint[i] + lambda_vec[i] / rho)
                f_current += 0.5 * rho * terms * terms
            armijo_condition = f_current + 1e-4 * alpha * (gradient[0] * delta[0] + gradient[1] * delta[1])

            iteration_count = 0
            f_update = f_current
            while f_update > armijo_condition and alpha > 1e-8 and iteration_count < 100:
                alpha *= 0.5
                temp_u = uknot - alpha * delta[0]
                temp_v = vknot - alpha * delta[1]
                iteration_count += 1; 
                
                point1 = surface1.single_point(temp_u, temp_v)
                f_update = pow(point1[2] - z_ground, 2)
                constraint = np.array([uboundary[0] - temp_u, temp_u - uboundary[1], vboundary[0] - temp_v, temp_v - vboundary[1]])
                for i in range(4):
                    terms = max(0.0, constraint[i] + lambda_vec[i] / rho)
                    f_update += 0.5 * rho * terms * terms
            
            uknot = temp_u
            vknot = temp_v

            if alpha * np.linalg.norm(delta) < residual:
                newton_converged = 3
                break
            newton_iter += 1

        constraint0 = np.array([uboundary[0] - uknot0, uknot0 - uboundary[1], vboundary[0] - vknot0, vknot0 - vboundary[1]])
        constraint = np.array([uboundary[0] - uknot, uknot - uboundary[1], vboundary[0] - vknot, vknot - vboundary[1]])
        max_violation = 0.0
        pre_violation = 0.0
        for i in range(4):
            lambda_vec[i] = max(0.0, lambda_vec[i] + rho * constraint[i])
            max_violation = max(max_violation, abs(max(0.0, constraint[i])))
            pre_violation = max(pre_violation, abs(max(0.0, constraint0[i])))

        if max_violation > 0.25 * pre_violation or newton_converged == 0:
            rho = min(1e8, rho * 10.0)

        if max_violation < tol and pow((uknot - uknot0), 2) + pow((vknot - vknot0), 2) < tol:
            outer_converged = True
            break

    assert outer_converged is True

    distance = abs(surface1.single_point(uknot, vknot)[2] - z_ground)
    pairs = (uknot, vknot)
    return distance, pairs
