from __future__ import annotations
from src.utils.linalg import (
    normalized,
    evaluate_bounding_box,
    check_same_side,
    solve_using_cod,
    project_point_to_plane,
    point_to_edges,
    point_to_point,
    get_determinant_3x2,
    project_evd,
    quadratic_root,
    quadrilateral_to_triangle_connectivity,
    insert_midpoints,
    polygon_area,
    tsp_greedy,
    sort_points_counterclockwise,
    distance_aabb,
    pca_analysis,
)
from src.utils.errors import NonlinearSolveError, SimulationStateError
from src.nurbs.Distance import Queue
import numpy as np
from src.nurbs.Spline import Spline
from src.nurbs.Geometry import (
    capsule_capsule_distance,
    triangle_slab_triangle_slab_distance,
    triangle_slab_plane_distance,
    visualize,
)
from src.nurbs.OptimizerVisualize import plot_nd_slice
from src.nurbs.NurbsBasis import *
from src.nurbs.Operations import *
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr, cg
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib.pyplot as plt
from src.linear_solver.ScipyKrylov import checked_scipy_krylov
from src.nurbs.MIPS import MIPS
from src.nurbs.Nurbs import find_span_linear, basis_functions, one_basis_function

TOL = 2.2204460492503131e-14


def make_list(input):
    if isinstance(input, (list, tuple, np.ndarray)):
        return list(input)
    else:
        return [input]


def assemble_point_weight(point, weight):
    return np.vstack(((point.T * weight), weight)).T


def split_point_weight(wpoint):
    return (wpoint[:, 0:-1].T / wpoint[:, -1]).T, wpoint[:, -1]


def check_closed_boundary(xi, knot):
    if xi < knot[0]:
        xi += knot[-1] - knot[0]
    elif xi > knot[-1]:
        xi -= knot[-1] - knot[0]
    return xi


def check_open_boundary(xi, knot):
    if xi < knot[0]:
        xi = knot[0]
    elif xi > knot[-1]:
        xi = knot[-1]
    return xi


def normalize(knot_vector: np.ndarray):
    if not 0.0 <= knot_vector.all() <= 1.0:
        return knot_vector
    else:
        max_knot = np.max(knot_vector)
        min_knot = np.min(knot_vector)

        normalized = knot_vector - min_knot * np.ones(knot_vector.shape)
        normalized *= 1 / (max_knot - min_knot)
        return normalized


def check_knot_vector(degree, knot_vector):
    knot_vector = normalize(np.asarray(knot_vector))
    if not np.allclose(np.zeros(degree + 1), knot_vector[: degree + 1]):
        raise TypeError("Knot vector error")
    if not np.allclose(np.ones(degree + 1), knot_vector[-1 * (degree + 1)]):
        raise TypeError("Knot vector error")
    previous_knot = knot_vector[0]
    for knot in knot_vector:
        if knot < previous_knot:
            raise TypeError("Knot vector must be non-decreasing")
        previous_knot = knot
    return knot_vector


def generate_uniform(degree, num_ctrlpts):
    length_knot_vector = num_ctrlpts + degree + 1
    num_middle_knots = length_knot_vector - 2 * degree
    middle_knot_vector = np.linspace(0, 1, num_middle_knots)
    knot_vector = np.concatenate((np.zeros(degree), middle_knot_vector, np.ones(degree)))
    return knot_vector


def get_ctrlpt_dofs(arr, dim=3):
    return np.repeat(arr, dim) * dim + np.tile(np.arange(dim), len(arr))


def update_patch_aabb(point_cloud, tol=1e-8):
    box_min = np.min(point_cloud, axis=0)
    box_max = np.max(point_cloud, axis=0)
    center = (box_min + box_max) * 0.5
    half_size = (box_max - box_min) * 0.5
    half_size = np.maximum(half_size, tol)
    box_min = center - half_size
    box_max = center + half_size
    return {"box_min": box_min, "box_max": box_max}


def curve_bounding_capsule(curve1: SplineCurve, curve2: SplineCurve, sA_intv, sB_intv):
    from src.nurbs.SplineCurve import SplineCurve

    mid_point1 = 0.5 * (sA_intv[0] + sA_intv[1])
    mid_point2 = 0.5 * (sB_intv[0] + sB_intv[1])
    curvature1 = np.linalg.norm(curve1.second_derivative(mid_point1))
    curvature2 = np.linalg.norm(curve2.second_derivative(mid_point2))
    radius1 = 0.125 * curvature1 * abs(sA_intv[1] - sA_intv[0]) ** 2
    radius2 = 0.125 * curvature2 * abs(sB_intv[1] - sB_intv[0]) ** 2
    start_point1 = curve1.single_point(sA_intv[0])
    end_point1 = curve1.single_point(sA_intv[1])
    start_point2 = curve2.single_point(sB_intv[0])
    end_point2 = curve2.single_point(sB_intv[1])
    return capsule_capsule_distance((start_point1, end_point1), radius1, (start_point2, end_point2), radius2)


def curve_curve_distance(
    curve1: SplineCurve, curve2: SplineCurve, residual=1e-14, max_newton_iter=50, tol=1e-8, max_bisection_iter=2000000
):
    from src.nurbs.SplineCurve import SplineCurve

    if curve1.degree == 2 and curve2.degree == 2:
        distance, pairs = curve_curve_second_order_distance_bisection(
            curve1, curve2, tol=tol, max_bisection_iter=max_bisection_iter
        )
        distance, pairs = curve_curve_distance_newton_iteration(
            curve1, curve2, initial_guess=pairs, residual=residual, max_newton_iter=max_newton_iter, tol=tol
        )
        return distance, pairs
    else:
        distance, pairs = curve_curve_distance_bisection(curve1, curve2, tol=tol, max_bisection_iter=max_bisection_iter)
        distance, pairs = curve_curve_distance_newton_iteration(
            curve1, curve2, initial_guess=pairs, residual=residual, max_newton_iter=max_newton_iter, tol=tol
        )
    return distance, pairs


def curve_curve_second_order_distance_bisection(
    curve1: SplineCurve, curve2: SplineCurve, tol=1e-6, max_bisection_iter=20000
):
    from src.nurbs.SplineCurve import SplineCurve

    distance, pairs = np.inf, (np.nan, np.nan)
    for elem1 in curve1.element:
        for elem2 in curve2.element:
            queue = []
            if curve_bounding_capsule(curve1, curve2, (elem1[0], elem1[1]), (elem2[0], elem2[1])) < distance:
                queue = [((elem1[0], elem1[1]), (elem2[0], elem2[1]), max(elem1[1] - elem1[0], elem2[1] - elem2[0]))]
            iter = 1
            while queue:
                s_a_intv, s_b_intv, width = queue.pop()
                center_s_a = 0.5 * (s_a_intv[0] + s_a_intv[1])
                center_s_b = 0.5 * (s_b_intv[0] + s_b_intv[1])

                if max(abs(s_a_intv[1] - s_a_intv[0]) > tol, abs(s_b_intv[1] - s_b_intv[0])) > tol:
                    lower_bound = curve_bounding_capsule(curve1, curve2, s_a_intv, s_b_intv)
                    if lower_bound >= distance:
                        continue

                    if abs(s_a_intv[1] - s_a_intv[0]) > abs(s_b_intv[1] - s_b_intv[0]):
                        queue.append(
                            (
                                (s_a_intv[0], center_s_a),
                                s_b_intv,
                                max(abs(center_s_a - s_a_intv[0]), abs(s_b_intv[1] - s_b_intv[0])),
                            )
                        )
                        queue.append(
                            (
                                (center_s_a, s_a_intv[1]),
                                s_b_intv,
                                max(abs(s_a_intv[1] - center_s_a), abs(s_b_intv[1] - s_b_intv[0])),
                            )
                        )
                    else:
                        queue.append(
                            (
                                s_a_intv,
                                (s_b_intv[0], center_s_b),
                                max(abs(s_a_intv[1] - s_a_intv[0]), abs(center_s_b - s_b_intv[0])),
                            )
                        )
                        queue.append(
                            (
                                s_a_intv,
                                (center_s_b, s_b_intv[1]),
                                max(abs(s_a_intv[1] - s_a_intv[0]), abs(s_b_intv[1] - center_s_b)),
                            )
                        )
                    queue = sorted(queue, key=lambda x: x[2], reverse=False)
                point_a, point_b = curve1.single_point(center_s_a), curve2.single_point(center_s_b)
                dist = np.linalg.norm(point_a - point_b)
                if dist < distance:
                    distance = dist
                    pairs = (center_s_a, center_s_b)
                iter += 1

            if iter > max_bisection_iter:
                print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
                break
    return distance, pairs


def curve_curve_distance_bisection(curve1: SplineCurve, curve2: SplineCurve, tol=1e-6, max_bisection_iter=20000):
    from src.nurbs.SplineCurve import SplineCurve

    distance, pairs = np.inf, (np.nan, np.nan)
    for elem1 in curve1.element:
        for elem2 in curve2.element:
            queue = []
            if distance_lower_bound(curve1, curve2, (elem1[0], elem1[1]), (elem2[0], elem2[1])) < distance:
                queue = [((elem1[0], elem1[1]), (elem2[0], elem2[1]), max(elem1[1] - elem1[0], elem2[1] - elem2[0]))]

            iter = 1
            distance, pairs = np.linalg.norm(curve1.single_point(0.5) - curve2.single_point(0.5)), (0.5, 0.5)
            while queue:
                s_a_intv, s_b_intv, width = queue.pop()
                center_s_a = 0.5 * (s_a_intv[0] + s_a_intv[1])
                center_s_b = 0.5 * (s_b_intv[0] + s_b_intv[1])

                if max(abs(s_a_intv[1] - s_a_intv[0]) > tol, abs(s_b_intv[1] - s_b_intv[0])) > tol:
                    lower_bound = distance_lower_bound(curve1, curve2, s_a_intv, s_b_intv)
                    if lower_bound >= distance:
                        continue

                    if abs(s_a_intv[1] - s_a_intv[0]) > abs(s_b_intv[1] - s_b_intv[0]):
                        queue.append(
                            (
                                (s_a_intv[0], center_s_a),
                                s_b_intv,
                                max(abs(center_s_a - s_a_intv[0]), abs(s_b_intv[1] - s_b_intv[0])),
                            )
                        )
                        queue.append(
                            (
                                (center_s_a, s_a_intv[1]),
                                s_b_intv,
                                max(abs(s_a_intv[1] - center_s_a), abs(s_b_intv[1] - s_b_intv[0])),
                            )
                        )
                    else:
                        queue.append(
                            (
                                s_a_intv,
                                (s_b_intv[0], center_s_b),
                                max(abs(s_a_intv[1] - s_a_intv[0]), abs(center_s_b - s_b_intv[0])),
                            )
                        )
                        queue.append(
                            (
                                s_a_intv,
                                (center_s_b, s_b_intv[1]),
                                max(abs(s_a_intv[1] - s_a_intv[0]), abs(s_b_intv[1] - center_s_b)),
                            )
                        )
                    queue = sorted(queue, key=lambda x: x[2], reverse=False)
                point_a, point_b = curve1.single_point(center_s_a), curve2.single_point(center_s_b)
                dist = np.linalg.norm(point_a - point_b)
                if dist < distance:
                    distance = dist
                    pairs = (center_s_a, center_s_b)
                    tangent1 = curve1.tangent(center_s_a)
                    tangent2 = curve2.tangent(center_s_b)
                    if max(abs(tangent1.dot(point_a - point_b)), abs(tangent2.dot(point_a - point_b))) < tol:
                        break
                iter += 1

                if iter > max_bisection_iter:
                    print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
                    break
    return distance, pairs


def curve_curve_distance_newton_iteration(
    curve1: SplineCurve,
    curve2: SplineCurve,
    initial_guess,
    rho=10.0,
    residual=1e-10,
    max_outer_iter=100,
    max_newton_iter=50,
    tol=1e-8,
):
    from src.nurbs.SplineCurve import SplineCurve

    aknot, bknot = initial_guess

    lambda_vec = np.zeros(4)
    outer_iter = 0
    outer_converged = False
    while outer_iter < max_outer_iter:
        newton_iter = 0
        aknot0, bknot0 = aknot, bknot
        newton_converged = False
        while newton_iter < max_newton_iter:
            point1, tangent1, curvature1 = curve1.single_point(aknot), curve1.dxdknot(aknot), curve1.d2xd2knot(aknot)[0]
            point2, tangent2, curvature2 = curve2.single_point(bknot), curve2.dxdknot(bknot), curve2.d2xd2knot(bknot)[0]
            direction = point1 - point2
            if np.linalg.norm(direction) < tol:
                newton_converged = True
                break

            gradient = np.array([np.dot(direction, tangent1), -np.dot(direction, tangent2)])
            hessian = np.array(
                [
                    [np.dot(tangent1, tangent1) + np.dot(direction, curvature1), -np.dot(tangent1, tangent2)],
                    [-np.dot(tangent1, tangent2), np.dot(tangent2, tangent2) - np.dot(direction, curvature2)],
                ]
            )

            # g1 = u >= 0, g2 = 1 - u >= 0, g3 = v >= 0, g4 = 1 - v >= 0
            # [[∂g1/∂u, ∂g1/∂v], [∂g2/∂u, ∂g2/∂v], [∂g3/∂u, ∂g3/∂v], [∂g4/∂u, ∂g4/∂v]]
            constraint = np.array([-aknot, aknot - 1.0, -bknot, bknot - 1.0])
            gradient_constraint = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
            for i in range(4):
                terms = constraint[i] + lambda_vec[i] / rho
                if terms > 0.0:
                    gradient += rho * terms * gradient_constraint[i]
                    hess += rho * np.outer(gradient_constraint[i], gradient_constraint[i])

            if np.linalg.norm(gradient) < tol:
                newton_converged = True
                break

            if abs(aknot) < 1e-12 or abs(aknot - 1.0) < 1e-12:
                gradient[0] = 0.0
                hessian[0, :] = 0.0
                hessian[:, 0] = 0.0
                hessian[0, 0] = 1.0
            if abs(bknot) < 1e-12 or abs(bknot - 1.0) < 1e-12:
                gradient[1] = 0.0
                hessian[1, :] = 0.0
                hessian[:, 1] = 0.0
                hessian[1, 1] = 1.0

            eigvals = np.linalg.eigvalsh(hessian)
            lambda_min = np.min(np.abs(eigvals))
            norm_hessian = np.linalg.norm(hessian, ord=2)
            delta = np.zeros(2)
            if lambda_min > 1e-8 * norm_hessian:
                delta = np.linalg.solve(hessian, gradient)

            alpha_a = 1.0
            alpha_b = 1.0
            if delta[0] != 0.0:
                alpha_a = -(1.0 - aknot) / delta[0] if delta[0] < 0.0 else aknot / delta[0]
            if delta[1] != 0.0:
                alpha_b = -(1.0 - bknot) / delta[1] if delta[1] < 0.0 else bknot / delta[1]
            alpha = min(alpha_a, alpha_b, 1.0)
            temp_a = aknot - alpha * delta[0]
            temp_b = bknot - alpha * delta[1]
            f_current = direction.dot(direction)

            for i in range(4):
                terms = max(0.0, constraint[i] + lambda_vec[i] / rho)
                f_current += 0.5 * rho * terms * terms
            armijo_condition = f_current + 1e-4 * alpha * (gradient[0] * delta[0] + gradient[1] * delta[1])

            iteration_count = 0
            f_update = f_current
            while f_update > armijo_condition and alpha > 1e-8 and iteration_count < 100:
                alpha *= 0.5
                temp_a = aknot - alpha * delta[0]
                temp_b = bknot - alpha * delta[1]
                iteration_count += 1

                point1 = curve1.single_point(temp_a)
                point2 = curve2.single_point(temp_b)
                f_update = pow(np.linalg.norm(point1 - point2), 2)
                constraint = np.array([-temp_a, temp_a - 1.0, -temp_b, temp_b - 1.0])
                for i in range(4):
                    terms = max(0.0, constraint[i] + lambda_vec[i] / rho)
                    f_update += 0.5 * rho * terms * terms

            aknot = temp_a
            bknot = temp_b

            if alpha * np.linalg.norm(delta) < residual:
                newton_converged = True
                break
            newton_iter += 1

        constraint0 = np.array([-aknot0, aknot0 - 1.0, -bknot0, bknot0 - 1.0])
        constraint = np.array([-aknot, aknot - 1.0, -bknot, bknot - 1.0])
        max_violation = 0.0
        pre_violation = 0.0
        for i in range(4):
            lambda_vec[i] = max(0.0, lambda_vec[i] + rho * constraint[i])
            max_violation = max(max_violation, abs(max(0.0, constraint[i])))
            pre_violation = max(pre_violation, abs(max(0.0, constraint0[i])))

        if max_violation > 0.25 * pre_violation or not newton_converged:
            rho = min(1e8, rho * 10.0)

        if max_violation < tol and pow((aknot - aknot0), 2) + pow((bknot - bknot0), 2) < tol:
            outer_converged = True
            break

    assert outer_converged is True

    distance = np.linalg.norm(curve1.single_point(aknot) - curve2.single_point(bknot))
    pairs = (aknot, bknot)
    return distance, pairs


def surface_surface_distance(
    surface1: SplineSurface,
    surface2: SplineSurface,
    rho=10.0,
    residual=1e-14,
    max_outer_iter=100,
    max_newton_iter=50,
    tol=1e-8,
    max_bisection_iter=2000000,
):
    from src.nurbs.SplineSurface import SplineSurface

    if surface1.degree_u == 2 and surface1.degree_v == 2 and surface2.degree_u == 2 and surface2.degree_v == 2:
        pairs = ((0.5, 0.5), (0.5, 0.5))
        distance, pairs = surface_surface_second_order_distance_bisection(
            surface1, surface2, tol=tol, max_bisection_iter=max_bisection_iter
        )
        distance, pairs = surface_surface_distance_newton_iteration(
            surface1,
            surface2,
            initial_guess=pairs,
            rho=rho,
            residual=residual,
            max_outer_iter=max_outer_iter,
            max_newton_iter=max_newton_iter,
            tol=tol,
        )
        return distance, pairs
    else:
        distance, pairs = surface_surface_distance_bisection(
            surface1, surface2, tol=tol, max_bisection_iter=max_bisection_iter
        )
        distance, pairs = surface_surface_distance_newton_iteration(
            surface1,
            surface2,
            initial_guess=pairs,
            rho=rho,
            residual=residual,
            max_outer_iter=max_outer_iter,
            max_newton_iter=max_newton_iter,
            tol=tol,
        )
        return distance, pairs


def max_curvature_uu(
    dim, surface: SplineSurface, uknot_min, uknot_max, vknot_min, vknot_max, tol=1e-8, visualize=False
):
    from src.nurbs.SplineSurface import SplineSurface

    gr = 0.5 * (5**0.5 - 1)

    def fuu(u, v):
        return abs(surface.second_derivative(u, v)[0][int(dim)])

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


def max_curvature_vv(
    dim, surface: SplineSurface, uknot_min, uknot_max, vknot_min, vknot_max, tol=1e-8, visualize=False
):
    from src.nurbs.SplineSurface import SplineSurface

    gr = 0.5 * (5**0.5 - 1)

    def fvv(u, v):
        return abs(surface.second_derivative(u, v)[1][int(dim)])

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


def max_curvature_uv(
    dim, surface: SplineSurface, uknot_min, uknot_max, vknot_min, vknot_max, tol=1e-8, visualize=False
):
    from src.nurbs.SplineSurface import SplineSurface

    def fuv(u, v):
        return abs(surface.second_derivative(u, v)[2][int(dim)])

    max_uvcurvature = max(
        fuv(uknot_min, vknot_min), fuv(uknot_max, vknot_min), fuv(uknot_min, vknot_max), fuv(uknot_max, vknot_max)
    )
    if visualize:
        plot_nd_slice(fuv, [(uknot_min, uknot_max), (vknot_min, vknot_max)])
    return max_uvcurvature


def max_curvature(surface: SplineSurface, uknot_min, uknot_max, vknot_min, vknot_max, tol=1e-8, visualize=False):
    from src.nurbs.SplineSurface import SplineSurface

    M1 = max(
        max_curvature_uu(0, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
        max_curvature_uu(1, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
        max_curvature_uu(2, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
    )
    M2 = max(
        max_curvature_vv(0, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
        max_curvature_vv(1, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
        max_curvature_vv(2, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
    )
    M3 = max(
        max_curvature_uv(0, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
        max_curvature_uv(1, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
        max_curvature_uv(2, surface, uknot_min, uknot_max, vknot_min, vknot_max, tol=tol, visualize=visualize),
    )
    return M1, M2, M3


def surface_bounding_slab(surface1: SplineSurface, surface2: SplineSurface, sA_intv, sB_intv):
    from src.nurbs.SplineSurface import SplineSurface

    uA, vA = sA_intv[0], sA_intv[1]
    uB, vB = sB_intv[0], sB_intv[1]
    M1A, M2A, M3A = max_curvature(surface1, *uA, *vA)
    M1B, M2B, M3B = max_curvature(surface2, *uB, *vB)
    radius1 = 0.125 * (
        M1A * abs(uA[1] - uA[0]) ** 2
        + M2A * abs(vA[1] - vA[0]) ** 2
        + 2 * M3A * abs(uA[1] - uA[0]) * abs(vA[1] - vA[0])
    )
    radius2 = 0.125 * (
        M1B * abs(uB[1] - uB[0]) ** 2
        + M2B * abs(vB[1] - vB[0]) ** 2
        + 2 * M3B * abs(uB[1] - uB[0]) * abs(vB[1] - vB[0])
    )
    """if flag==1:
        visualize((surface1.single_point(uA[0], vA[0]), surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1])), (surface2.single_point(uB[0], vB[0]), surface2.single_point(uB[1], vB[0]), surface2.single_point(uB[0], vB[1])))
        visualize((surface1.single_point(uA[0], vA[0]), surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1])), (surface2.single_point(uB[1], vB[0]), surface2.single_point(uB[0], vB[1]), surface2.single_point(uB[1], vB[1])))
        visualize((surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1]), surface1.single_point(uA[1], vA[1])), (surface2.single_point(uB[0], vB[0]), surface2.single_point(uB[1], vB[0]), surface2.single_point(uB[0], vB[1])))
        visualize((surface1.single_point(uA[1], vA[0]), surface1.single_point(uA[0], vA[1]), surface1.single_point(uA[1], vA[1])), (surface2.single_point(uB[1], vB[0]), surface2.single_point(uB[0], vB[1]), surface2.single_point(uB[1], vB[1])))"""
    patch1 = [
        surface1.single_point(uA[0], vA[0]),
        surface1.single_point(uA[1], vA[0]),
        surface1.single_point(uA[0], vA[1]),
        surface1.single_point(uA[1], vA[1]),
    ]
    patch2 = [
        surface2.single_point(uB[0], vB[0]),
        surface2.single_point(uB[1], vB[0]),
        surface2.single_point(uB[0], vB[1]),
        surface2.single_point(uB[1], vB[1]),
    ]
    dist = min(
        triangle_slab_triangle_slab_distance(
            (patch1[0], patch1[1], patch1[2]), radius1, (patch2[0], patch2[1], patch2[2]), radius2
        ),
        triangle_slab_triangle_slab_distance(
            (patch1[0], patch1[1], patch1[2]), radius1, (patch2[1], patch2[2], patch2[3]), radius2
        ),
        triangle_slab_triangle_slab_distance(
            (patch1[1], patch1[2], patch1[3]), radius1, (patch2[0], patch2[1], patch2[2]), radius2
        ),
        triangle_slab_triangle_slab_distance(
            (patch1[1], patch1[2], patch1[3]), radius1, (patch2[1], patch2[2], patch2[3]), radius2
        ),
    )
    return dist, min(
        np.sqrt(4.0 * radius1 * abs(dist) + radius1 * radius1), np.sqrt(4.0 * radius2 * abs(dist) + radius2 * radius2)
    )


def surface_surface_second_order_distance_bisection(
    surface1: SplineSurface, surface2: SplineSurface, tol=1e-8, max_bisection_iter=2000
):
    from src.nurbs.SplineSurface import SplineSurface

    distance, pairs = np.inf, ((np.nan, np.nan), (np.nan, np.nan))
    for elem1_u in surface1.element_u:
        for elem1_v in surface1.element_v:
            for elem2_u in surface2.element_u:
                for elem2_v in surface2.element_v:
                    queue = Queue()
                    if (
                        surface_bounding_slab(
                            surface1,
                            surface2,
                            ((elem1_u[0], elem1_u[1]), (elem1_v[0], elem1_v[1])),
                            ((elem2_u[0], elem2_u[1]), (elem2_v[0], elem2_v[1])),
                        )[0]
                        < distance
                    ):
                        queue.append(
                            ((elem1_u[0], elem1_u[1]), (elem1_v[0], elem1_v[1])),
                            ((elem2_u[0], elem2_u[1]), (elem2_v[0], elem2_v[1])),
                        )
                        dist = np.linalg.norm(
                            surface1.single_point(0.5 * (elem1_u[0] + elem1_u[1]), 0.5 * (elem1_v[0] + elem1_v[1]))
                            - surface2.single_point(0.5 * (elem2_u[0] + elem2_u[1]), 0.5 * (elem2_v[0] + elem2_v[1]))
                        )
                        if dist < distance:
                            dist = distance
                            pairs = (
                                (0.5 * (elem1_u[0] + elem1_u[1]), 0.5 * (elem1_v[0] + elem1_v[1])),
                                (0.5 * (elem2_u[0] + elem2_u[1]), 0.5 * (elem2_v[0] + elem2_v[1])),
                            )

                    iter = 1
                    while queue.size() > 0.0:
                        (uA, vA), (uB, vB) = queue.pop()
                        uc = 0.5 * (uA[0] + uA[1])
                        vc = 0.5 * (vA[0] + vA[1])
                        sc = 0.5 * (uB[0] + uB[1])
                        tc = 0.5 * (vB[0] + vB[1])

                        if max(abs(uA[1] - uA[0]), abs(vA[1] - vA[0]), abs(uB[1] - uB[0]), abs(vB[1] - vB[0])) > tol:
                            lower_bound, error = surface_bounding_slab(
                                surface1, surface2, ((uA[0], uA[1]), (vA[0], vA[1])), ((uB[0], uB[1]), (vB[0], vB[1]))
                            )
                            # print(uc, vc, sc, tc)
                            if lower_bound - distance > -1e-2 * distance:
                                continue
                            if error > 1e-2 * distance:
                                queue = split_bounding_box(uA, vA, uB, vB, uc, vc, sc, tc, queue)

                        knots_a = [(uc, vc), (uA[0], vA[0]), (uA[0], vA[1]), (uA[1], vA[0]), (uA[1], vA[1])]
                        knots_b = [(sc, tc), (uB[0], vB[0]), (uB[0], vB[1]), (uB[1], vB[0]), (uB[1], vB[1])]
                        values = [
                            np.linalg.norm(surface1.single_point(ua, va) - surface2.single_point(ub, vb))
                            for (ua, va) in knots_a
                            for (ub, vb) in knots_b
                        ]
                        min_idx = min(range(len(values)), key=lambda i: values[i])
                        dist = values[min_idx]
                        uv = knots_a[int(min_idx // 5)]
                        st = knots_b[int(min_idx % 5)]
                        if dist < distance:
                            distance = dist
                            pairs = (uv, st)

                        """uv = (uc, vc)
                        st = (sc, tc)
                        dist, pair = surface_surface_distance_newton_iteration(surface1, surface2, initial_guess=(uv, st), tol=tol, uboundary=uA, vboundary=vA, sboundary=uB, tboundary=vB)
                        if dist < distance:
                            distance = dist
                            pairs = pair"""

                        iter += 1

                        if iter > max_bisection_iter:
                            print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
                            break
    return distance, pairs


def surface_surface_distance_bisection(
    surface1: SplineSurface, surface2: SplineSurface, tol=1e-8, max_bisection_iter=2000
):
    from src.nurbs.SplineSurface import SplineSurface

    queue = [
        (
            (
                (surface1.knot_vector_u[surface1.degree_u], surface1.knot_vector_u[-1 - surface1.degree_u]),
                (surface1.knot_vector_v[surface1.degree_v], surface1.knot_vector_v[-1 - surface1.degree_v]),
            ),
            (
                (surface2.knot_vector_u[surface2.degree_u], surface2.knot_vector_u[-1 - surface2.degree_u]),
                (surface2.knot_vector_v[surface2.degree_v], surface2.knot_vector_v[-1 - surface2.degree_v]),
            ),
            1.0,
        )
    ]

    iter = 1
    distance, pairs = np.linalg.norm(surface1.single_point(0.5, 0.5) - surface2.single_point(0.5, 0.5)), (
        (0.5, 0.5),
        (0.5, 0.5),
    )
    while len(queue) > 0.0:
        (uA, vA), (uB, vB), width = queue.pop()
        uc, vc = 0.5 * (uA[0] + uA[1]), 0.5 * (vA[0] + vA[1])
        sc, tc = 0.5 * (uB[0] + uB[1]), 0.5 * (vB[0] + vB[1])

        if max(abs(uA[1] - uA[0]), abs(vA[1] - vA[0]), abs(uB[1] - uB[0]), abs(vB[1] - vB[0])) > tol:
            lower_bound = distance_lower_bound(
                surface1, surface2, ((uA[0], vA[0]), (uA[1], vA[1])), ((uB[0], vB[0]), (uB[1], vB[1]))
            )
            if lower_bound >= distance:
                continue

            queue1, queue2 = split_bounding_box(uA, vA, uB, vB, uc, vc, sc, tc)
            queue.append(queue1)
            queue.append(queue2)
            queue = sorted(queue, key=lambda x: x[2], reverse=False)
        point_a, point_b = surface1.single_point(uc, vc), surface2.single_point(sc, tc)
        dist = np.linalg.norm(point_a - point_b)
        if dist < distance:
            distance = dist
            pairs = ((uc, vc), (sc, tc))
            tangent1_u, tangent1_v = surface1.tangent(uc, vc)
            tangent2_u, tangent2_v = surface2.tangent(sc, tc)
            if (
                max(
                    abs(tangent1_u.dot(point_a - point_b)),
                    abs(tangent1_v.dot(point_a - point_b)),
                    abs(tangent2_u.dot(point_a - point_b)),
                    abs(tangent2_v.dot(point_a - point_b)),
                )
                < tol
            ):
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

    """queue.append(((u1min, umid1), (v1min, vmid1)), ((u2min, umid2), (v2min, vmid2)))
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
    return queue"""

    lengths = {"u1": u1max - u1min, "v1": v1max - v1min, "u2": u2max - u2min, "v2": v2max - v2min}
    max_dir = max(lengths, key=lengths.get)
    if max_dir == "u1":
        queue.append(((u1min, umid1), (v1min, v1max)), ((u2min, u2max), (v2min, v2max)))
        queue.append(((umid1, u1max), (v1min, v1max)), ((u2min, u2max), (v2min, v2max)))
        return queue
    elif max_dir == "v1":
        queue.append(((u1min, u1max), (v1min, vmid1)), ((u2min, u2max), (v2min, v2max)))
        queue.append(((u1min, u1max), (vmid1, v1max)), ((u2min, u2max), (v2min, v2max)))
        return queue
    elif max_dir == "u2":
        queue.append(((u1min, u1max), (v1min, v1max)), ((u2min, umid2), (v2min, v2max)))
        queue.append(((u1min, u1max), (v1min, v1max)), ((umid2, u2max), (v2min, v2max)))
        return queue
    elif max_dir == "v2":
        queue.append(((u1min, u1max), (v1min, v1max)), ((u2min, u2max), (v2min, vmid2)))
        queue.append(((u1min, u1max), (v1min, v1max)), ((u2min, u2max), (vmid2, v2max)))
        return queue


def surface_surface_distance_newton_iteration(
    surface1: SplineSurface,
    surface2: SplineSurface,
    initial_guess,
    uboundary=[0.0, 1.0],
    vboundary=[0.0, 1.0],
    sboundary=[0.0, 1.0],
    tboundary=[0.0, 1.0],
    residual=1e-14,
    rho=10,
    max_outer_iter=100,
    max_newton_iter=500,
    tol=1e-8,
):
    from src.nurbs.SplineSurface import SplineSurface

    aknot, bknot = initial_guess
    uknot, vknot = aknot
    sknot, tknot = bknot

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

            hess_uu = 2.0 * (np.dot(direction, curvature_uu) + np.dot(tangent_u, tangent_u))
            hess_vv = 2.0 * (np.dot(direction, curvature_vv) + np.dot(tangent_v, tangent_v))
            hess_ss = 2.0 * (-np.dot(direction, curvature_ss) + np.dot(tangent_s, tangent_s))
            hess_tt = 2.0 * (-np.dot(direction, curvature_tt) + np.dot(tangent_t, tangent_t))

            hess_uv = 2.0 * (np.dot(direction, curvature_uv) + np.dot(tangent_u, tangent_v))
            hess_us = -2.0 * np.dot(tangent_u, tangent_s)
            hess_ut = -2.0 * np.dot(tangent_u, tangent_t)
            hess_vs = -2.0 * np.dot(tangent_v, tangent_s)
            hess_vt = -2.0 * np.dot(tangent_v, tangent_t)
            hess_st = 2.0 * (-np.dot(direction, curvature_st) + np.dot(tangent_s, tangent_t))

            gradient = np.array(
                [
                    np.dot(direction, tangent_u),
                    np.dot(direction, tangent_v),
                    -np.dot(direction, tangent_s),
                    -np.dot(direction, tangent_t),
                ]
            )
            hessian = np.array(
                [
                    [hess_uu, hess_uv, hess_us, hess_ut],
                    [hess_uv, hess_vv, hess_vs, hess_vt],
                    [hess_us, hess_vs, hess_ss, hess_st],
                    [hess_ut, hess_vt, hess_st, hess_tt],
                ]
            )

            # g1 = u >= 0, g2 = 1 - u >= 0, g3 = v >= 0, g4 = 1 - v >= 0
            # [[∂g1/∂u, ∂g1/∂v, ∂g1/∂s, ∂g1/∂t], [∂g2/∂u, ∂g2/∂v, ∂g2/∂s, ∂g2/∂t], [∂g3/∂u, ∂g3/∂v, ∂g3/∂s, ∂g3/∂t], [∂g4/∂u, ∂g4/∂v, ∂g4/∂s, ∂g4/∂t]]
            constraint = np.array(
                [
                    uboundary[0] - uknot,
                    uknot - uboundary[1],
                    vboundary[0] - vknot,
                    vknot - vboundary[1],
                    sboundary[0] - sknot,
                    sknot - sboundary[1],
                    tboundary[0] - tknot,
                    tknot - tboundary[1],
                ]
            )
            gradient_constraint = np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            for i in range(8):
                terms = constraint[i] + lambda_vec[i] / rho
                if terms > 0.0:
                    gradient += rho * terms * gradient_constraint[i]
                    hessian += rho * np.outer(gradient_constraint[i], gradient_constraint[i])

            # if np.linalg.norm(gradient) < residual:
            #    newton_converged = 2
            #    break

            if abs(uknot - uboundary[0]) < 1e-12 or abs(uknot - uboundary[1]) < 1e-12:
                gradient[0] = 0.0
                hessian[0, :] = 0.0
                hessian[:, 0] = 0.0
                hessian[0, 0] = 1.0
            if abs(vknot - vboundary[0]) < 1e-12 or abs(vknot - vboundary[1]) < 1e-12:
                gradient[1] = 0.0
                hessian[1, :] = 0.0
                hessian[:, 1] = 0.0
                hessian[1, 1] = 1.0
            if abs(sknot - sboundary[0]) < 1e-12 or abs(sknot - sboundary[1]) < 1e-12:
                gradient[2] = 0.0
                hessian[2, :] = 0.0
                hessian[:, 2] = 0.0
                hessian[2, 2] = 1.0
            if abs(tknot - tboundary[0]) < 1e-12 or abs(tknot - tboundary[1]) < 1e-12:
                gradient[3] = 0.0
                hessian[3, :] = 0.0
                hessian[:, 3] = 0.0
                hessian[3, 3] = 1.0

            eigvals, eigvecs = np.linalg.eigh(hessian)
            lambda_min = np.min(np.abs(eigvals))
            mask = np.abs(eigvals) > 1e-6
            """T_s = eigvecs[:, mask]
            grad_r = T_s.T @ gradient
            H_r = T_s.T @ hessian @ T_s
            delta_r = np.linalg.solve(H_r, grad_r)
            delta = T_s @ delta_r"""
            norm_hessian = np.linalg.norm(hessian, ord=2)
            delta = np.zeros(4)
            print(eigvals, direction)
            if lambda_min > 1e-8 * norm_hessian:
                delta = np.linalg.solve(hessian, gradient)
            # print(eigvals);input()

            alpha_a = 1.0
            alpha_b = 1.0
            alpha_c = 1.0
            alpha_d = 1.0
            if delta[0] != 0.0:
                alpha_a = -(uboundary[1] - uknot) / delta[0] if delta[0] < 0.0 else (uknot - uboundary[0]) / delta[0]
            if delta[1] != 0.0:
                alpha_b = -(vboundary[1] - vknot) / delta[1] if delta[1] < 0.0 else (vknot - vboundary[0]) / delta[1]
            if delta[2] != 0.0:
                alpha_c = -(sboundary[1] - sknot) / delta[2] if delta[2] < 0.0 else (sknot - sboundary[0]) / delta[2]
            if delta[3] != 0.0:
                alpha_d = -(tboundary[1] - tknot) / delta[3] if delta[3] < 0.0 else (tknot - tboundary[0]) / delta[3]
            alpha = min(0.999 * alpha_a, 0.999 * alpha_b, 0.999 * alpha_c, 0.999 * alpha_d, 1.0)
            temp_u = uknot - alpha * delta[0]
            temp_v = vknot - alpha * delta[1]
            temp_s = sknot - alpha * delta[2]
            temp_t = tknot - alpha * delta[3]
            f_current = direction.dot(direction)

            for i in range(8):
                terms = max(0.0, constraint[i] + lambda_vec[i] / rho)
                f_current += 0.5 * rho * terms * terms
            armijo_condition = f_current + 1e-4 * alpha * (
                gradient[0] * delta[0] + gradient[1] * delta[1] + gradient[2] * delta[2] + gradient[3] * delta[3]
            )

            iteration_count = 0
            f_update = f_current
            while f_update > armijo_condition and alpha > 1e-8 and iteration_count < 100:
                alpha *= 0.5
                temp_u = uknot - alpha * delta[0]
                temp_v = vknot - alpha * delta[1]
                temp_s = sknot - alpha * delta[2]
                temp_t = tknot - alpha * delta[3]
                iteration_count += 1

                point1 = surface1.single_point(temp_u, temp_v)
                point2 = surface2.single_point(temp_s, temp_t)
                f_update = pow(np.linalg.norm(point1 - point2), 2)
                constraint = np.array(
                    [
                        uboundary[0] - temp_u,
                        temp_u - uboundary[1],
                        vboundary[0] - temp_v,
                        temp_v - vboundary[1],
                        sboundary[0] - temp_s,
                        temp_s - sboundary[1],
                        tboundary[0] - temp_t,
                        temp_t - tboundary[1],
                    ]
                )
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

        constraint0 = np.array(
            [
                uboundary[0] - uknot0,
                uknot0 - uboundary[1],
                vboundary[0] - vknot0,
                vknot0 - vboundary[1],
                sboundary[0] - sknot0,
                sknot0 - sboundary[1],
                tboundary[0] - tknot0,
                tknot0 - tboundary[1],
            ]
        )
        constraint = np.array(
            [
                uboundary[0] - uknot,
                uknot - uboundary[1],
                vboundary[0] - vknot,
                vknot - vboundary[1],
                sboundary[0] - sknot,
                sknot - sboundary[1],
                tboundary[0] - tknot,
                tknot - tboundary[1],
            ]
        )
        max_violation = 0.0
        pre_violation = 0.0
        for i in range(8):
            lambda_vec[i] = max(0.0, lambda_vec[i] + rho * constraint[i])
            max_violation = max(max_violation, abs(max(0.0, constraint[i])))
            pre_violation = max(pre_violation, abs(max(0.0, constraint0[i])))

        if max_violation > 0.25 * pre_violation or not newton_converged:
            rho = min(1e15, rho * 10.0)

        if (
            max_violation < tol
            and pow((uknot - uknot0), 2)
            + pow((vknot - vknot0), 2)
            + pow((sknot - sknot0), 2)
            + pow((tknot - tknot0), 2)
            < tol
        ):
            outer_converged = True
            break

        outer_iter += 1
    assert outer_converged is True

    distance = np.linalg.norm(surface1.single_point(uknot, vknot) - surface2.single_point(sknot, tknot))
    pairs = ((uknot, vknot), (sknot, tknot))
    return distance, pairs


def surface_ground_distance(
    surface: SplineSurface,
    z_ground,
    rho=10.0,
    residual=1e-14,
    max_outer_iter=100,
    max_newton_iter=500,
    tol=1e-8,
    max_bisection_iter=200000,
):
    from src.nurbs.SplineSurface import SplineSurface

    if surface.degree_u == 2 and surface.degree_v == 2:
        distance, pairs = surface_ground_second_order_distance_bisection(
            surface, z_ground, tol=tol, max_bisection_iter=max_bisection_iter
        )
        distance, pairs = surface_ground_distance_newton_iteration(
            surface,
            z_ground,
            initial_guess=pairs,
            rho=rho,
            residual=residual,
            max_outer_iter=max_outer_iter,
            max_newton_iter=max_newton_iter,
            tol=tol,
        )
        return distance, pairs
    else:
        distance, pairs = surface_ground_distance_bisection(
            surface, z_ground, tol=tol, max_bisection_iter=max_bisection_iter
        )
        distance, pairs = surface_ground_distance_newton_iteration(
            surface,
            z_ground,
            initial_guess=pairs,
            rho=rho,
            residual=residual,
            max_outer_iter=max_outer_iter,
            max_newton_iter=max_newton_iter,
            tol=tol,
        )
        return distance, pairs


def surface_ground_bounding_slab(surface1: SplineSurface, z_ground, sA_intv, visualize=False):
    from src.nurbs.SplineSurface import SplineSurface

    uA, vA = sA_intv[0], sA_intv[1]
    curvature1_u, curvature1_v, curvature1_uv = max_curvature(surface1, *uA, *vA)
    radius_a = 0.125 * (
        curvature1_u * abs(uA[1] - uA[0]) ** 2
        + curvature1_v * abs(vA[1] - vA[0]) ** 2
        + 2 * curvature1_uv * abs(uA[1] - uA[0]) * abs(vA[1] - vA[0])
    )
    if visualize:
        surface1.visualize(
            knot_u=np.linspace(uA[0], uA[1]),
            knot_v=np.linspace(vA[0], vA[1]),
            bounding_volume=[
                {
                    "point1": surface1.single_point(uA[0], vA[0]),
                    "point2": surface1.single_point(uA[1], vA[0]),
                    "point3": surface1.single_point(uA[0], vA[1]),
                    "radius": radius_a,
                }
            ],
        )
    return min(
        triangle_slab_plane_distance(
            (
                surface1.single_point(uA[0], vA[0]),
                surface1.single_point(uA[1], vA[0]),
                surface1.single_point(uA[0], vA[1]),
            ),
            radius_a,
            z_ground,
        ),
        triangle_slab_plane_distance(
            (
                surface1.single_point(uA[1], vA[0]),
                surface1.single_point(uA[0], vA[1]),
                surface1.single_point(uA[1], vA[1]),
            ),
            radius_a,
            z_ground,
        ),
    )


def surface_ground_second_order_distance_bisection(surface1: SplineSurface, z_ground, tol=1e-8, max_bisection_iter=64):
    from src.nurbs.SplineSurface import SplineSurface

    distance, pairs = np.inf, (np.nan, np.nan)
    for elem1_u in surface1.element_u:
        for elem1_v in surface1.element_v:
            if (
                surface_ground_bounding_slab(surface1, z_ground, ((elem1_u[0], elem1_u[1]), (elem1_v[0], elem1_v[1])))
                < distance
            ):
                queue = [
                    (
                        ((elem1_u[0], elem1_u[1]), (elem1_v[0], elem1_v[1])),
                        max(elem1_u[1] - elem1_u[0], elem1_v[1] - elem1_v[0]),
                    )
                ]

            iter = 1
            while len(queue) > 0.0:
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

                dist, pair = surface_ground_distance_newton_iteration(
                    surface1, z_ground, initial_guess=(uc, vc), uboundary=(uA[0], uA[1]), vboundary=(vA[0], vA[1])
                )

                if dist < distance:
                    distance = dist
                    pairs = pair
                iter += 1

                if iter > max_bisection_iter:
                    print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
                    break
    return distance, pairs


def surface_ground_distance_bisection(surface1: SplineSurface, z_ground, tol=1e-8, max_bisection_iter=64):
    from src.nurbs.SplineSurface import SplineSurface

    queue = [
        (
            (
                (surface1.knot_vector_u[surface1.degree_u], surface1.knot_vector_u[-1 - surface1.degree_u]),
                (surface1.knot_vector_v[surface1.degree_v], surface1.knot_vector_v[-1 - surface1.degree_v]),
            ),
            1.0,
        )
    ]

    iter = 1
    distance, pairs = surface1.single_point(0.5, 0.5)[2] - z_ground, (0.5, 0.5)
    while len(queue) > 0.0:
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
            if max(abs(tangent_u.dot(np.array([0.0, 0.0, -1]))), abs(tangent_v.dot(np.array([0.0, 0.0, -1])))) < tol:
                break
        iter += 1

        if iter > max_bisection_iter:
            print(f"Max bisection iterations {max_bisection_iter} reached, stopping.")
            break
    return distance, pairs


def ground_distance_lower_bound(z_ground, primitive1: Spline, sA_intv):
    trimmed_ctrlpts1 = primitive1.convex_hull(*sA_intv)
    bounding_box1 = update_patch_aabb(trimmed_ctrlpts1)
    return min(abs(bounding_box1["box_min"][2] - z_ground), abs(bounding_box1["box_max"][2] - z_ground))


def split_single_bounding_box(u1r, v1r):
    u1min, u1max = u1r
    v1min, v1max = v1r

    lengths = {"u1": u1max - u1min, "v1": v1max - v1min}
    max_dir = max(lengths, key=lengths.get)
    if max_dir == "u1":
        umid = 0.5 * (u1min + u1max)
        return (((u1min, umid), (v1min, v1max)), max(umid - u1min, v1max - v1min)), (
            ((umid, u1max), (v1min, v1max)),
            max(u1max - umid, v1max - v1min),
        )
    elif max_dir == "v1":
        vmid = 0.5 * (v1min + v1max)
        return (((u1min, u1max), (v1min, vmid)), max(u1max - u1min, vmid - v1min)), (
            ((u1min, u1max), (vmid, v1max)),
            max(u1max - u1min, v1max - vmid),
        )


def surface_ground_distance_newton_iteration(
    surface1: SplineSurface,
    z_ground,
    initial_guess,
    uboundary=[0.0, 1.0],
    vboundary=[0.0, 1.0],
    rho=10.0,
    residual=1e-14,
    max_outer_iter=100,
    max_newton_iter=50,
    tol=1e-8,
):
    from src.nurbs.SplineSurface import SplineSurface

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

            hess_uu = 2.0 * (direction * curvature_uu[2] + tangent_u[2] * tangent_u[2])
            hess_vv = 2.0 * (direction * curvature_vv[2] + tangent_v[2] * tangent_v[2])
            hess_uv = 2.0 * (direction * curvature_uv[2] + tangent_u[2] * tangent_v[2])

            gradient = np.array([direction * tangent_u[2], direction * tangent_v[2]])
            hessian = np.array([[hess_uu, hess_uv], [hess_uv, hess_vv]])

            # g1 = u >= 0, g2 = 1 - u >= 0, g3 = v >= 0, g4 = 1 - v >= 0
            # [[∂g1/∂u, ∂g1/∂v, ∂g1/∂s, ∂g1/∂t], [∂g2/∂u, ∂g2/∂v, ∂g2/∂s, ∂g2/∂t], [∂g3/∂u, ∂g3/∂v, ∂g3/∂s, ∂g3/∂t], [∂g4/∂u, ∂g4/∂v, ∂g4/∂s, ∂g4/∂t]]
            constraint = np.array(
                [uboundary[0] - uknot, uknot - uboundary[1], vboundary[0] - vknot, vknot - vboundary[1]]
            )
            gradient_constraint = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])

            for i in range(4):
                terms = constraint[i] + lambda_vec[i] / rho
                if terms > 0.0:
                    gradient += rho * terms * gradient_constraint[i]
                    hessian += rho * np.outer(gradient_constraint[i], gradient_constraint[i])

            if np.linalg.norm(gradient) < residual:
                newton_converged = 2
                break

            if abs(uknot - uboundary[0]) < 1e-12 or abs(uknot - uboundary[1]) < 1e-12:
                gradient[0] = 0.0
                hessian[0, :] = 0.0
                hessian[:, 0] = 0.0
                hessian[0, 0] = 1.0
            if abs(vknot - vboundary[0]) < 1e-12 or abs(vknot - vboundary[1]) < 1e-12:
                gradient[1] = 0.0
                hessian[1, :] = 0.0
                hessian[:, 1] = 0.0
                hessian[1, 1] = 1.0

            eigvals = np.linalg.eigvalsh(hessian)
            lambda_min = np.min(np.abs(eigvals))
            norm_hessian = np.linalg.norm(hessian, ord=2)
            delta = np.zeros(2)
            if lambda_min > 1e-8 * norm_hessian:
                delta = np.linalg.solve(hessian, gradient)

            alpha_a = 1.0
            alpha_b = 1.0
            if delta[0] != 0.0:
                alpha_a = -(uboundary[1] - uknot) / delta[0] if delta[0] < 0.0 else (uknot - uboundary[0]) / delta[0]
            if delta[1] != 0.0:
                alpha_b = -(vboundary[1] - vknot) / delta[1] if delta[1] < 0.0 else (vknot - vboundary[0]) / delta[1]
            alpha = min(alpha_a, alpha_b, 1.0)
            temp_u = uknot - alpha * delta[0]
            temp_v = vknot - alpha * delta[1]

            f_current = direction * direction
            for i in range(4):
                terms = max(0.0, constraint[i] + lambda_vec[i] / rho)
                f_current += 0.5 * rho * terms * terms
            armijo_condition = f_current + 1e-4 * alpha * (gradient[0] * delta[0] + gradient[1] * delta[1])

            iteration_count = 0
            f_update = f_current
            while f_update > armijo_condition and alpha > 1e-8 and iteration_count < 100:
                alpha *= 0.5
                temp_u = uknot - alpha * delta[0]
                temp_v = vknot - alpha * delta[1]
                iteration_count += 1

                point1 = surface1.single_point(temp_u, temp_v)
                f_update = pow(point1[2] - z_ground, 2)
                constraint = np.array(
                    [uboundary[0] - temp_u, temp_u - uboundary[1], vboundary[0] - temp_v, temp_v - vboundary[1]]
                )
                for i in range(4):
                    terms = max(0.0, constraint[i] + lambda_vec[i] / rho)
                    f_update += 0.5 * rho * terms * terms

            uknot = temp_u
            vknot = temp_v

            if alpha * np.linalg.norm(delta) < residual:
                newton_converged = 3
                break
            newton_iter += 1

        constraint0 = np.array(
            [uboundary[0] - uknot0, uknot0 - uboundary[1], vboundary[0] - vknot0, vknot0 - vboundary[1]]
        )
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


def bspline_curve(degree, knot_vector, control_point, closed=False, parent_indices=None):
    from src.nurbs.BSplineCurve import BSplineCurve

    curve = BSplineCurve()
    curve.degree = degree
    curve.knot_vector = knot_vector
    curve.control_points = control_point
    curve.closed = closed
    curve.parent_indices = parent_indices
    return curve


def bspline_surface(degree_u, degree_v, knot_vector_u, knot_vector_v, control_point, closed=False, parent_indices=None):
    from src.nurbs.BSplineSurface import BSplineSurface

    surf = BSplineSurface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.knot_vector_u = knot_vector_u
    surf.knot_vector_v = knot_vector_v
    surf.control_points = control_point
    surf.closed = closed
    surf.parent_indices = parent_indices
    surf.activate_boundary()
    return surf


def bspline_volume(
    degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, control_point, closed=False
):
    from src.nurbs.BSplineVolume import BSplineVolume

    surf = BSplineVolume()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.degree_w = degree_w
    surf.knot_vector_u = knot_vector_u
    surf.knot_vector_v = knot_vector_v
    surf.knot_vector_w = knot_vector_w
    surf.control_points = control_point
    surf.closed = closed
    surf.activate_boundary()
    return surf


def compute_params_curve(points, centripetal=False):
    points = np.asarray(points)
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    if centripetal:
        distances = np.sqrt(distances)
    cumulative_lengths = np.insert(np.cumsum(distances), 0, 0.0)
    total_length = cumulative_lengths[-1]
    if total_length == 0:
        return np.linspace(0, 1, len(points))
    uk = cumulative_lengths / total_length
    return uk


def compute_params_surface(points, num_datapt_u, num_datapt_v, centripetal=False):
    uk_temp = np.zeros((num_datapt_v, num_datapt_u))  # (4, 5)
    for v in range(num_datapt_v):
        pts_u = points[v, :, :]  # shape = (5, 3)
        uk_temp[v, :] = compute_params_curve(pts_u, centripetal)
    uk = uk_temp.mean(axis=0)

    vl_temp = np.zeros((num_datapt_u, num_datapt_v))
    for u in range(num_datapt_u):
        pts_v = points[:, u, :]
        vl_temp[u, :] = compute_params_curve(pts_v, centripetal)
    vl = vl_temp.mean(axis=0)
    return np.array(uk), np.array(vl)


def compute_curve_knot_vector(degree, num_points, uk):
    kv_start = np.zeros(degree + 1)
    num_internal_knots = num_points - degree - 1
    if num_internal_knots > 0:
        internal_knots = np.array([np.mean(uk[i + 1 : i + degree + 1]) for i in range(num_internal_knots)])
    else:
        internal_knots = np.array([])
    kv_end = np.ones(degree + 1)
    return np.concatenate([kv_start, internal_knots, kv_end])


def compute_curve_knot_vector_with_deriv(degree, num_points, num_ders, uk):
    pass


def compute_curve_knot_vector_one(degree, num_dpts, num_cpts, params):
    kv_start = np.zeros(degree + 1)
    d = num_dpts / (num_cpts - degree)
    j_values = np.arange(1, num_cpts - degree)
    i_values = (j_values * d).astype(int)
    alpha_values = (j_values * d) - i_values
    i_values = np.clip(i_values, 1, len(params) - 1)
    internal_knots = (1.0 - alpha_values) * params[i_values - 1] + alpha_values * params[i_values]
    kv_end = np.ones(degree + 1)
    return np.concatenate([kv_start, internal_knots, kv_end])


def compute_uniform_knot_vector(degree, num_cpts):
    return np.concatenate([np.zeros(degree), np.linspace(0, 1, num_cpts - degree + 1), np.ones(degree)])


def build_coeff_matrix(degree, knotvector, params, points):
    points = np.asarray(points)
    num_points = points.shape[0]
    matrix_a = np.zeros((num_points, num_points))
    for i in range(num_points):
        span = find_span_linear(num_points, params[i], degree, knotvector)
        matrix_a[i, span - degree : span + 1] = basis_functions(span, params[i], degree, knotvector)
    return matrix_a


def build_coeff_matrix_with_deriv(degree, knotvector, params, points, derivatives):
    num_points = points.shape[0]
    num_ders = derivatives.shape[0]
    matrix_a = np.zeros((num_points, num_points))
    for i in range(num_points):
        span = find_span_linear(num_points, params[i], degree, knotvector)
        matrix_a[i, span - degree : span + 1] = basis_functions(span, params[i], degree, knotvector)
    return matrix_a


def unconstraint_curve_fitting(num_ctrlpt, degree, knot_vector, evals, points):
    ndim = points.shape[-1]
    matrix_n = np.array(
        [
            [one_basis_function(j, evals[i], degree, knot_vector) for j in range(1, num_ctrlpt - 1)]
            for i in range(1, evals.shape[0] - 1)
        ]
    )
    rk = points[1:-1] - (
        np.outer([one_basis_function(0, u, degree, knot_vector) for u in evals[1:-1]], points[0])
        + np.outer([one_basis_function(num_ctrlpt - 1, u, degree, knot_vector) for u in evals[1:-1]], points[-1])
    )

    vector_r = np.zeros((num_ctrlpt - 2, ndim))
    for j in range(1, num_ctrlpt - 1):
        basis_values = np.array([one_basis_function(j, u, degree, knot_vector) for u in evals[1:-1]])
        vector_r[j - 1] = np.dot(basis_values.T, rk)

    ctrlpts_internal = np.linalg.solve(np.dot(matrix_n.T, matrix_n), vector_r)
    return np.vstack((points[0][np.newaxis, :], ctrlpts_internal, points[-1][np.newaxis, :]))


def constraint_curve_fitting(num_ctrlpt, degree, knot_vector, evals, points, fixed):
    if np.all(fixed):
        return points

    ndim = points.shape[-1]
    fixed_indices = np.where(fixed)[0]
    free_indices = np.where(~fixed)[0]

    matrix = np.array(
        [
            [one_basis_function(j, evals[i], degree, knot_vector) for j in range(num_ctrlpt)]
            for i in range(evals.shape[0])
        ]
    )
    matrix_a = matrix[fixed_indices, :]
    matrix_n = matrix[free_indices, :]

    vector_r = np.zeros((num_ctrlpt, ndim))
    for j in range(num_ctrlpt):
        basis_values = np.array([one_basis_function(j, u, degree, knot_vector) for u in evals[free_indices]])
        vector_r[j] = np.dot(basis_values.T, points[free_indices])

    A = np.vstack(
        (
            np.hstack((np.matmul(matrix_n.transpose(), matrix_n), matrix_a.transpose())),
            np.hstack((matrix_a, np.zeros((matrix_a.shape[0], matrix_a.shape[0])))),
        )
    )
    b = np.vstack((vector_r, points[fixed_indices]))
    control_points = np.linalg.solve(A, b)
    return control_points[0:num_ctrlpt]


def compute_curve_control_points(points, resolution):
    if isinstance(resolution, (int, float)):
        num_ctrlpts = int(resolution)

    points = np.asarray(points)
    ndim = points.shape[1]
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov_matrix = centered_points.T @ centered_points
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvalues /= points.shape[0]

    bbmin, bbmax = evaluate_bounding_box(np.dot(centered_points, np.linalg.inv(eigenvectors).T))
    if ndim == 2:
        return (
            np.dot(
                np.vstack(
                    (
                        np.linspace(bbmin[0], bbmax[0] + 0.001 * (bbmax[0] - bbmin[0]) / num_ctrlpts, num_ctrlpts),
                        np.zeros(num_ctrlpts),
                    )
                ).T,
                eigenvectors.T,
            )
            + mean
        )
    elif ndim == 3:
        return (
            np.dot(
                np.vstack(
                    (
                        np.linspace(bbmin[0], bbmax[0] + 0.001 * (bbmax[0] - bbmin[0]) / num_ctrlpts, num_ctrlpts),
                        np.zeros(num_ctrlpts),
                        np.zeros(num_ctrlpts),
                    )
                ).T,
                eigenvectors.T,
            )
            + mean
        )


def compute_close_curve_control_points(points, resolution):
    if isinstance(resolution, (int, float)):
        num_ctrlpts = int(resolution)

    points = np.asarray(points)
    ndim = points.shape[1]
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    mean_radius = np.sqrt(eigenvalues[0])
    axes = np.linspace(0.0, 2.0 * np.pi * (1.0 + 0.01 / num_ctrlpts), num_ctrlpts)
    if ndim == 2:
        return np.dot(np.vstack((mean_radius * np.sin(axes), mean_radius * np.cos(axes))).T, eigenvectors.T) + mean
    elif ndim == 3:
        return (
            np.dot(
                np.vstack((mean_radius * np.sin(axes), mean_radius * np.cos(axes), np.zeros(num_ctrlpts))).T,
                eigenvectors.T,
            )
            + mean
        )


def order_points(projected_points):
    projected_points = np.asarray(projected_points)
    distance_matrix = cdist(projected_points, projected_points, metric="euclidean")
    mst_matrix = minimum_spanning_tree(distance_matrix).toarray()
    mst_graph = nx.from_numpy_array(mst_matrix)
    degrees = np.array([deg for _, deg in mst_graph.degree()])
    endpoints = np.where(degrees == 1)[0]
    farthest_pair = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    if len(endpoints) >= 2:
        farthest_pair = endpoints[:2]
    start_index = farthest_pair[0]
    ordered_indices = np.array(list(nx.dfs_preorder_nodes(mst_graph, source=start_index)))
    return np.take(projected_points, ordered_indices, axis=0)


def find_boundary_index(points, alpha=0.1):
    if points.shape[1] == 2:
        return np.unique(alpha_shape_2d(points, alpha))
    elif points.shape[1] == 3:
        return np.unique(alpha_shape_3d(points, alpha))


def alpha_shape_2d(points, alpha=0.1):
    triangulation = Delaunay(points)
    triangles = points[triangulation.simplices]

    A, B, C = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    AB, AC = B - A, C - A

    cross_product = np.cross(AB, AC)
    cross_product[cross_product == 0] = 1e-10

    AB2 = np.sum(AB**2, axis=1)
    AC2 = np.sum(AC**2, axis=1)
    R = np.sqrt(AB2 * AC2) / (2 * np.abs(cross_product))
    valid_triangles = R < (1 / alpha)
    valid_simplices = triangulation.simplices[valid_triangles]
    edges = np.vstack([valid_simplices[:, [0, 1]], valid_simplices[:, [1, 2]], valid_simplices[:, [2, 0]]])

    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    return boundary_edges


def circumradius(a, b, c, d):
    AB, AC, AD = b - a, c - a, d - a
    M = np.column_stack((AB, AC, AD))
    det_m = np.linalg.det(M)
    if abs(det_m) < 1e-10:
        return np.inf
    rhs = 0.5 * np.array([np.dot(AB, AB), np.dot(AC, AC), np.dot(AD, AD)])
    center = np.linalg.solve(M.T, rhs)
    return np.linalg.norm(center)


def alpha_shape_3d(points, alpha=0.1):
    tri = Delaunay(points)
    simplices = points[tri.simplices]
    A, B, C, D = simplices[:, 0], simplices[:, 1], simplices[:, 2], simplices[:, 3]
    radii = np.array([circumradius(A[i], B[i], C[i], D[i]) for i in range(len(A))])
    valid_tetrahedra = radii < (1 / alpha)
    valid_simplices = tri.simplices[valid_tetrahedra]
    faces = np.vstack(
        [
            valid_simplices[:, [0, 1, 2]],
            valid_simplices[:, [0, 1, 3]],
            valid_simplices[:, [0, 2, 3]],
            valid_simplices[:, [1, 2, 3]],
        ]
    )
    faces = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
    boundary_faces = unique_faces[counts == 1]
    return boundary_faces


def axis_aligned_bounding_quadrilateral(projected_points):
    bbmin, bbmax = evaluate_bounding_box(projected_points)
    return np.array([bbmin[:2], [bbmax[0], bbmin[1]], bbmax[:2], [bbmin[0], bbmax[1]]])


def optimize_bounding_quadrilateral(projected_points):
    hull = ConvexHull(projected_points)
    hull_pts = projected_points[hull.vertices]
    edges = np.diff(hull_pts, axis=0, append=hull_pts[:1])
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.unique(np.abs(angles))
    min_area = np.inf
    best_rect = None
    best_rotation = None
    for angle in angles:
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = hull_pts @ R.T
        min_x, min_y = np.min(rotated, axis=0)
        max_x, max_y = np.max(rotated, axis=0)
        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            best_rect = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
            best_rotation = R
    return best_rect @ best_rotation


def convert_quad2d_to_quad3d(quad2d, rotation_matrix):
    quad3d = np.zeros((4, 3))
    quad3d[:, :2] = quad2d
    return quad3d @ rotation_matrix.T


def subdivide_quadrilateral(quad, resolution):
    p0, p1, p2, p3 = quad
    u = np.linspace(0, 1, resolution[0] + 1)
    v = np.linspace(0, 1, resolution[1] + 1)
    u_grid, v_grid = np.meshgrid(u, v, indexing="ij")  # u_grid.shape = (res_u+1, res_v+1)

    u_grid_flat = u_grid.reshape(-1, 1)
    v_grid_flat = v_grid.reshape(-1, 1)

    pA = (1 - v_grid_flat) * p0 + v_grid_flat * p1
    pB = (1 - v_grid_flat) * p3 + v_grid_flat * p2
    vertices = (1 - u_grid_flat) * pA + u_grid_flat * pB

    idx = np.arange((resolution[0] + 1) * (resolution[1] + 1)).reshape(resolution[0] + 1, resolution[1] + 1)

    quads = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()], axis=1)

    return vertices, quads


def select_boundary_corners(boundary_points):
    """Pick four corners on an ordered planar boundary; the first pick defines A."""
    points = _prepare_planar_boundary(boundary_points)
    if plt.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
        raise RuntimeError("Corner picking requires an interactive Matplotlib backend; pass corner_points instead")
    figure, axes = plt.subplots()
    closed = np.vstack((points, points[0]))
    axes.plot(closed[:, 0], closed[:, 1], "k.-")
    axes.set_aspect("equal")
    axes.set_title("Pick four corners (first = A). Left: add; right: undo; middle: finish")
    try:
        selected = plt.ginput(4, timeout=0)
    finally:
        plt.close(figure)
    if len(selected) != 4:
        raise ValueError("Four corners are required; surface fitting was cancelled")
    selected = np.asarray(selected)
    indices = np.argmin(np.linalg.norm(points[:, None] - selected[None, :], axis=2), axis=0)
    if len(np.unique(indices)) != 4:
        raise ValueError("Select four distinct boundary corners")
    return points[indices].copy()


def _prepare_planar_boundary(boundary_points):
    points = np.asarray(boundary_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("Boundary must contain at least four finite, ordered 2D points")
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 4 or np.any(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) == 0.0):
        raise ValueError("Boundary contains repeated consecutive points")
    relative = points - points[0]
    area = np.sum(relative[:, 0] * np.roll(relative[:, 1], -1) - relative[:, 1] * np.roll(relative[:, 0], -1))
    if area == 0.0:
        raise ValueError("Boundary has zero signed area")
    for index in range(len(points) - 2):
        first, last = points[index], points[index + 1]
        starts = points[index + 2 :]
        ends = np.roll(points, -1, axis=0)[index + 2 :]
        if index == 0:
            starts, ends = starts[:-1], ends[:-1]
        direction = last - first
        others = ends - starts
        cross_a = direction[0] * (starts[:, 1] - first[1]) - direction[1] * (starts[:, 0] - first[0])
        cross_b = direction[0] * (ends[:, 1] - first[1]) - direction[1] * (ends[:, 0] - first[0])
        cross_c = others[:, 0] * (first[1] - starts[:, 1]) - others[:, 1] * (first[0] - starts[:, 0])
        cross_d = others[:, 0] * (last[1] - starts[:, 1]) - others[:, 1] * (last[0] - starts[:, 0])
        overlap = np.all(
            np.maximum(np.minimum(first, last), np.minimum(starts, ends))
            <= np.minimum(np.maximum(first, last), np.maximum(starts, ends)),
            axis=1,
        )
        if np.any((cross_a * cross_b <= 0.0) & (cross_c * cross_d <= 0.0) & overlap):
            raise ValueError("Boundary must be a single non-self-intersecting loop")
    return points if area > 0.0 else points[::-1]


def get_boundary_categories(
    surface: SplineSurface, boundary_points, corner_index=None, visualize=False, corner_points=None
):
    from src.nurbs.SplineSurface import SplineSurface
    from src.nurbs.CurvePoints import CurvePoints

    if corner_points is not None and corner_index is not None and len(corner_index):
        raise ValueError("Pass corner_points or corner_index, not both")
    if corner_points is None:
        if corner_index is not None and len(corner_index):
            if len(corner_index) != 4:
                raise ValueError("The dimension of corner_index should be 4")
            corner_points = np.asarray(boundary_points)[corner_index]
        else:
            corner_points = select_boundary_corners(boundary_points)
    boundary_categories = split_boundary_points(surface, boundary_points, corner_points, visualize=visualize)
    fittings = [CurvePoints(), CurvePoints(), CurvePoints(), CurvePoints()]
    fittings[0].setup(boundary_categories["Category 1 (A->B)"])
    fittings[1].setup(boundary_categories["Category 2 (B->C)"])
    fittings[2].setup(boundary_categories["Category 3 (C->D)"])
    fittings[3].setup(boundary_categories["Category 4 (D->A)"])
    return fittings


def fit_surface_from_boundary(boundary_points, corner_points=None, degree=(2, 2), num_ctrlpts=(4, 4)):
    """Fit four boundary curves and a planar Coons patch; pick corners only when omitted."""
    from src.nurbs.NurbsVolume import NurbsVolume
    from src.nurbs.element.NurbsSurfaceElement import NurbsSurfaceElement

    points = _prepare_planar_boundary(boundary_points)
    if corner_points is None:
        corner_points = select_boundary_corners(points)
    categories = list(split_boundary_points(None, points, corner_points).values())
    degree = np.asarray(degree)
    num_ctrlpts = np.asarray(num_ctrlpts)
    if (
        degree.shape != (2,)
        or num_ctrlpts.shape != (2,)
        or not np.issubdtype(degree.dtype, np.integer)
        or not np.issubdtype(num_ctrlpts.dtype, np.integer)
    ):
        raise ValueError("degree and num_ctrlpts must each contain two integers")
    if np.any(degree < 1) or np.any(num_ctrlpts <= degree):
        raise ValueError("Each control-point count must exceed its positive degree")
    degree = tuple(int(value) for value in degree)
    knots = [generate_uniform(int(p), int(n)) for p, n in zip(degree, num_ctrlpts)]
    curves = []
    for edge_id, edge in enumerate(categories):
        axis = edge_id % 2
        if edge_id >= 2:
            edge = edge[::-1]
        parameters = compute_params_curve(edge)
        samples = np.unique(np.r_[parameters, np.linspace(0.0, 1.0, 2 * num_ctrlpts[axis])])
        samples_points = np.column_stack([np.interp(samples, parameters, edge[:, component]) for component in range(2)])
        fixed = np.zeros(len(samples), dtype=bool)
        fixed[[0, -1]] = True
        curves.append(
            constraint_curve_fitting(num_ctrlpts[axis], degree[axis], knots[axis], samples, samples_points, fixed)
        )
    greville = [
        np.array([np.mean(knot[i + 1 : i + p + 1]) for i in range(n)]) for knot, p, n in zip(knots, degree, num_ctrlpts)
    ]
    u, v = np.meshgrid(*greville)
    u, v = u[..., None], v[..., None]
    bottom, right, top, left = curves
    corner_a, corner_b, corner_c, corner_d = bottom[0], bottom[-1], top[-1], top[0]
    bilinear = (1 - u) * (1 - v) * corner_a + u * (1 - v) * corner_b + u * v * corner_c + (1 - u) * v * corner_d
    control_points = (
        (1 - v) * bottom[None, :, :]
        + v * top[None, :, :]
        + (1 - u) * left[:, None, :]
        + u * right[:, None, :]
        - bilinear
    )
    surface = NurbsVolume.nurbs_surface(*degree, *knots, control_points.reshape(-1, 2))
    NurbsSurfaceElement(surface)
    return surface


def get_margin_curves(surface: SplineSurface, fittings):
    from src.nurbs.SplineSurface import SplineSurface

    margin_curves = [
        margin_fitting(surface, 0, fittings[0].points),
        margin_fitting(surface, 1, fittings[1].points),
        margin_fitting(surface, 2, fittings[2].points),
        margin_fitting(surface, 3, fittings[3].points),
    ]
    return margin_curves


def margin_fitting(surface: SplineSurface, index, margin_points):
    from src.nurbs.SplineSurface import SplineSurface
    from src.nurbs.CurvePoints import CurvePoints

    fitting = CurvePoints()
    fitting.setup(margin_points)
    fixed = np.zeros(margin_points.shape[0])
    fixed[0] = fixed[-1] = 1
    curve = precompute_curve(margin_points, degree=surface.boundary[index].degree, knot_vector="scale", fixed=fixed)
    curve = nurbs_curve_fitting_gauss_newton(fitting, curve, fix_dofs=[0, 1, -2, -1])
    return curve


def generate_quadrilateral_mesh_with_corner(
    surface: SplineSurface,
    fittings,
    margin_curves=None,
    density=[1, 1],
    weights=1.0,
    penalties=0.0,
    smoothness=0.0,
    visualize=False,
):
    from src.nurbs.SplineSurface import SplineSurface

    if margin_curves is None:
        margin_curves = get_margin_curves(surface, fittings)

    if isinstance(weights, float):
        weights = np.repeat(weights, len(fittings))
    if isinstance(penalties, float):
        penalties = np.repeat(penalties, len(fittings))
    if isinstance(smoothness, float):
        smoothness = np.repeat(smoothness, len(fittings))
    for fitting, weight, penalty, smooth in zip(fittings, weights, penalties, smoothness):
        fitting.weights = weight
        fitting.penalty = penalty
        fitting.smoothness = smooth
    for i in range(4):
        for _ in range(density[i % 2]):
            margin_curves[i].refine_knot(density=1)
        if fittings[i].penalty > 0.0:
            margin_curves[i] = nurbs_curve_fitting_gauss_newton(fittings[i], margin_curves[i], fix_dofs=[0, 1, -2, -1])
        elif fittings[i].smoothness > 0:
            margin_curves[i] = bspline_curve_fitting_least_squares(
                fittings[i], margin_curves[i], fix_dofs=[0, 1, -2, -1], method="sdm"
            )
        if visualize:
            margin_curves[i].visualize(datapts=fittings[i].points)
    return margin_curves


def construct_surface_from_boundary(surface: SplineSurface, margin_curves, visualize=False):
    from src.nurbs.SplineSurface import SplineSurface

    margin_ctrlpts = [
        margin_curves[0].control_points,
        margin_curves[1].control_points,
        margin_curves[2].control_points,
        margin_curves[3].control_points,
    ]
    new_ctrlpts = create_quad_mesh(
        margin_curves[0].num_ctrlpts, margin_curves[1].num_ctrlpts, margin_ctrlpts, visualize=visualize
    ).reshape(-1, 2)
    surface.update(margin_curves[0].knot_vector, margin_curves[1].knot_vector, new_ctrlpts)
    return surface


def partition_boundary(projected_control_point):
    from src.nurbs.Kmean import kmean

    edges = [
        [projected_control_point[0, 0], projected_control_point[-1, 0]],
        [projected_control_point[-1, 0], projected_control_point[-1, -1]],
        [projected_control_point[-1, -1], projected_control_point[0, -1]],
        [projected_control_point[0, -1], projected_control_point[0, 0]],
    ]
    labels, _ = kmean(projected_control_point, edges=edges, norm="L1")
    return labels


def flip_lines(points, center):
    if np.cross(points[-1] - points[0], center - points[0]) < 0:
        return np.flip(points, axis=0)
    else:
        return points


def resample_boundary3d(surface: SplineSurface, points, alpha=0.5):
    from src.nurbs.SplineSurface import SplineSurface

    points_center, eigenvectors, eigenvalues = pca_analysis(points)
    projected_points = np.dot(points - points_center, eigenvectors[:, :2])
    boundary_index = find_boundary_index(projected_points, alpha=alpha)
    boundary_points = projected_points[boundary_index]
    return resample_boundary(surface, boundary_points)


def resample_boundary2d(surface: SplineSurface, points, alpha=0.5):
    from src.nurbs.SplineSurface import SplineSurface

    boundary_index = find_boundary_index(points, alpha=alpha)
    boundary_points = points[boundary_index]
    return resample_boundary(surface, boundary_points)


def resample_boundary(surface: SplineSurface, boundary_points, alpha=0.5):
    from src.nurbs.SplineSurface import SplineSurface
    from src.nurbs.CurvePoints import CurvePoints

    labels = partition_boundary(boundary_points)
    boundaries = []
    for i in range(4):
        boundary_point = flip_lines(order_points(boundary_points[labels == i, :]), np.mean(boundary_points, axis=0))
        fixed = np.zeros(boundary_point.shape[0])
        fixed[0] = fixed[-1] = 1
        curve = precompute_curve(
            boundary_point, degree=2, control_points=surface.boundary[i].num_ctrlpts - 2, fixed=fixed
        )
        fitting = CurvePoints()
        fitting.setup(boundary_points[labels == i, :])
        curve = nurbs_curve_fitting_gauss_newton(fitting=fitting, curve=curve)
        boundaries.append(
            np.array([curve.single_point(knot) for knot in np.linspace(0.0, 1.0, surface.boundary[i].num_ctrlpts - 2)])
        )

    for i in range(4):
        boundaries[i] = np.vstack((boundaries[(i - 1) % 4][-1], boundaries[i], boundaries[(i + 1) % 4][0]))
    return boundaries


def create_quad_mesh(num_ctrlpts_u, num_ctrlpts_v, boundaries, visualize=False):
    # reference: Kuraishi, T., Yamasaki, S., Takizawa, K., Tezduyar, T. E., Xu, Z., & Kaneko, R. (2022). Space–time isogeometric analysis of car and tire aerodynamics with road contact and tire deformation and rotation. Computational Mechanics, 70(1), 49-72.
    grid_points = np.zeros((num_ctrlpts_v, num_ctrlpts_u, 2))
    grid_points[0, :] = boundaries[0]
    grid_points[:, -1] = boundaries[1]
    grid_points[-1, :] = np.flip(boundaries[2], axis=0)
    grid_points[:, 0] = np.flip(boundaries[3], axis=0)

    base_line11 = grid_points[0, -1] - grid_points[0, 0]
    base_line21 = grid_points[-1, -1] - grid_points[-1, 0]
    base_line12 = grid_points[-1, 0] - grid_points[0, 0]
    base_line22 = grid_points[-1, -1] - grid_points[0, -1]
    base_line11_norm = np.linalg.norm(base_line11)
    base_line21_norm = np.linalg.norm(base_line21)
    base_line12_norm = np.linalg.norm(base_line12)
    base_line22_norm = np.linalg.norm(base_line22)
    base_line11_den2 = 1.0 / (base_line11_norm * base_line11_norm)
    base_line21_den2 = 1.0 / (base_line21_norm * base_line21_norm)
    base_line12_den2 = 1.0 / (base_line12_norm * base_line12_norm)
    base_line22_den2 = 1.0 / (base_line22_norm * base_line22_norm)

    s11 = np.dot(grid_points[0, 1:-1] - grid_points[0, 0], base_line11) * base_line11_den2
    s21 = np.dot(grid_points[-1, 1:-1] - grid_points[-1, 0], base_line21) * base_line21_den2
    s12 = np.dot(grid_points[1:-1, 0] - grid_points[0, 0], base_line12) * base_line12_den2
    s22 = np.dot(grid_points[1:-1, -1] - grid_points[0, -1], base_line22) * base_line22_den2
    s1 = 0.5 * (s11 + s21)
    s2 = 0.5 * (s12 + s22)

    dir1 = grid_points[1:-1, -1] - grid_points[1:-1, 0]
    dir2 = grid_points[-1, 1:-1] - grid_points[0, 1:-1]
    t1 = dir1 / np.linalg.norm(dir1, axis=1)[:, np.newaxis]
    t2 = dir2 / np.linalg.norm(dir2, axis=1)[:, np.newaxis]
    for i in range(1, num_ctrlpts_v - 1):
        for j in range(1, num_ctrlpts_u - 1):
            c1 = (1.0 - s1[j - 1]) * grid_points[i, 0] + s1[j - 1] * grid_points[i, -1]
            c2 = (1.0 - s2[i - 1]) * grid_points[0, j] + s2[i - 1] * grid_points[-1, j]
            new_ctrlpts = np.linalg.solve(
                np.outer(t1[i - 1], t1[i - 1]) + np.outer(t2[j - 1], t2[j - 1]),
                np.dot(np.outer(t1[i - 1], t1[i - 1]), c1) + np.dot(np.outer(t2[j - 1], t2[j - 1]), c2),
            )
            grid_points[i, j] = new_ctrlpts

    if visualize:
        fig = plt.figure(figsize=[10, 8], dpi=96)
        ax = plt.axes(projection="3d")
        ax.plot_wireframe(grid_points[:, :, 0], grid_points[:, :, 1], np.zeros_like(grid_points[:, :, 1]), color="red")
        ax.scatter(grid_points[:, :, 0], grid_points[:, :, 1], np.zeros_like(grid_points[:, :, 1]))
        plt.show()
        plt.close()
    return grid_points


def compute_surface_control_points_2d(points, resolution, bounding_quadrilateral_method=0):
    if isinstance(resolution, (int, float)):
        num_ctrlpts = [int(resolution) - 1, int(resolution) - 1]
    elif isinstance(resolution, (np.ndarray, list, tuple)):
        num_ctrlpts = np.asarray(resolution) - 1

    get_bounding = axis_aligned_bounding_quadrilateral
    if bounding_quadrilateral_method == 1:
        get_bounding = optimize_bounding_quadrilateral

    quad2d = sort_points_counterclockwise(get_bounding(points))
    current_points, quads = subdivide_quadrilateral(quad2d, num_ctrlpts)
    return current_points


def compute_surface_control_points_3d(points, resolution, bounding_quadrilateral_method=0):
    if isinstance(resolution, (int, float)):
        num_ctrlpts = [int(resolution) - 1, int(resolution) - 1]
    elif isinstance(resolution, (np.ndarray, list, tuple)):
        num_ctrlpts = np.asarray(resolution) - 1

    get_bounding = axis_aligned_bounding_quadrilateral
    if bounding_quadrilateral_method == 1:
        get_bounding = optimize_bounding_quadrilateral

    points = np.asarray(points)
    points_center, eigenvectors, eigenvalues = pca_analysis(points)
    projected_points = np.dot(points - points_center, eigenvectors[:, :2])
    quad2d = get_bounding(projected_points)
    quad3d = convert_quad2d_to_quad3d(quad2d, eigenvectors) + points_center
    current_points, quads = subdivide_quadrilateral(quad3d, num_ctrlpts)
    return current_points


def compute_fitting_error(primitive: Spline, points, closed=None):
    residuals = primitive.distance(points, closed)
    return np.sum(np.sqrt(residuals * residuals))


def global_interpolate_bspline_curve(points, degree, derivatives=None, use_centripetal=False):
    from src.nurbs.NurbsSurface import NurbsSurface as _NurbsSurface_nurbs_curve_owner

    if not isinstance(points, (list, tuple, np.ndarray)):
        raise TypeError("Data points must be a list or a tuple")
    points = np.asarray(points)

    knot_vector, matrix_a = None, None
    uk = compute_params_curve(points, use_centripetal)
    if derivatives is None:
        knot_vector = compute_curve_knot_vector(degree, points.shape[0], uk)
        matrix_a = build_coeff_matrix(degree, knot_vector, uk, points)
    else:
        if not isinstance(derivatives, (list, tuple, np.ndarray)):
            raise TypeError("Data derivatives must be a list or a tuple")
        derivatives = np.asarray(derivatives)
        knot_vector = compute_curve_knot_vector_with_deriv(degree, points.shape[0], derivatives.shape[0], uk)
        matrix_a = build_coeff_matrix_with_deriv(degree, knot_vector, uk, points, derivatives)

    # Do global interpolation
    ctrlpts = np.linalg.solve(matrix_a, points)

    # Generate B-spline curve
    return _NurbsSurface_nurbs_curve_owner.nurbs_curve(degree, knot_vector, ctrlpts)


def global_interpolate_bspline_surface(
    num_datapt_u, num_datapt_v, points, degree_u, degree_v, derivatives=None, use_centripetal=False
):
    from src.nurbs.NurbsVolume import NurbsVolume as _NurbsVolume_nurbs_surface_owner

    if not isinstance(points, (list, tuple, np.ndarray)):
        raise TypeError("Data points must be a list or a tuple")
    ndim = points.shape[-1]
    points = np.asarray(points).reshape(num_datapt_v, num_datapt_u, -1)

    uk, vl = compute_params_surface(points, num_datapt_u, num_datapt_v, use_centripetal)
    kv_u = compute_curve_knot_vector(degree_u, num_datapt_u, uk)
    kv_v = compute_curve_knot_vector(degree_v, num_datapt_v, vl)

    n_ctrlpts_u = len(kv_u) - degree_u - 1
    n_ctrlpts_v = len(kv_v) - degree_v - 1

    ctrlpts_r = np.zeros((num_datapt_v, n_ctrlpts_u, ndim))
    for v in range(num_datapt_v):
        pts = points[v, :, :]
        A = build_coeff_matrix(degree_u, kv_u, uk, pts)
        ctrlpts_r[v, :, :] = np.linalg.solve(A, pts)

    ctrlpts = np.zeros((n_ctrlpts_v, n_ctrlpts_u, ndim))
    for u in range(n_ctrlpts_u):
        pts = ctrlpts_r[:, u, :]
        A = build_coeff_matrix(degree_v, kv_v, vl, pts)
        ctrlpts[:, u, :] = np.linalg.solve(A, pts)
    ctrlpts = ctrlpts.reshape(-1, ndim)

    # Generate B-spline surface
    return _NurbsVolume_nurbs_surface_owner.nurbs_surface(degree_u, degree_v, kv_u, kv_v, ctrlpts)


def global_approximate_bspline_curve(
    points, degree, knot_vector=None, control_point_num=None, fixed=None, use_centripetal=False
):
    from src.nurbs.NurbsSurface import NurbsSurface as _NurbsSurface_nurbs_curve_owner

    if not isinstance(points, (list, tuple, np.ndarray)):
        raise TypeError("Data points must be a list or a tuple")
    points = np.asarray(points)

    if degree <= 1:
        raise ValueError("Degree must be larger than 1")

    num_datapt = points.shape[0]
    num_ctrlpt = num_datapt - 1 if control_point_num is None else control_point_num

    if fixed is None:
        fixed = np.zeros(num_datapt, dtype=bool)
        fixed[0], fixed[-1] = True, True
    else:
        fixed = np.asarray(fixed).astype(bool)
        if fixed.shape[0] != num_datapt:
            raise ValueError(f"The dimension of fixed array should be {num_datapt}")

    if num_ctrlpt <= degree:
        raise ValueError("The number of control points must be larger than degree")

    if knot_vector is None:
        ndim = points.shape[-1]
        uk = compute_params_curve(points, use_centripetal)
        knot_vector = compute_curve_knot_vector_one(degree, num_datapt, num_ctrlpt, uk)
    else:
        # TODO: Check knot vector
        pass

    control_points = constraint_curve_fitting(num_ctrlpt, degree, knot_vector, uk, points, fixed).reshape(-1, ndim)

    # Generate B-spline curve
    return _NurbsSurface_nurbs_curve_owner.nurbs_curve(degree, knot_vector, control_points)


def global_approximate_bspline_surface(
    degree_u,
    degree_v,
    num_datapt_u,
    num_datapt_v,
    points,
    ctrlpts_size_u=None,
    ctrlpts_size_v=None,
    use_centripetal=False,
):
    from src.nurbs.NurbsVolume import NurbsVolume as _NurbsVolume_nurbs_surface_owner

    if not isinstance(points, (list, tuple, np.ndarray)):
        raise TypeError("Data points must be a list or a tuple")

    if num_datapt_u * num_datapt_v != points.shape[0]:
        raise ValueError(f"The number of data points must be equal to {points.shape[0]}")

    if degree_u <= 1 or degree_v <= 1:
        raise ValueError("Degree must be larger than 1")

    num_ctrlpt_u = num_datapt_u - 1 if ctrlpts_size_u is None else ctrlpts_size_u
    num_ctrlpt_v = num_datapt_v - 1 if ctrlpts_size_v is None else ctrlpts_size_v

    if num_ctrlpt_u >= num_datapt_u or num_ctrlpt_v >= num_datapt_v:
        raise ValueError("The number of control point should be smaller than the number of data points")

    if num_ctrlpt_u < degree_u or num_ctrlpt_v < degree_v:
        raise ValueError("The number of control points must be larger or equal to degree")

    ndim = points.shape[-1]
    points = np.asarray(points).reshape(num_datapt_v, num_datapt_u, -1)

    uk, vl = compute_params_surface(points, num_datapt_u, num_datapt_v, use_centripetal)
    knot_vector_u = compute_curve_knot_vector_one(degree_u, num_datapt_u, num_ctrlpt_u, uk)
    knot_vector_v = compute_curve_knot_vector_one(degree_v, num_datapt_v, num_ctrlpt_v, vl)

    # First fit every data row in the u direction.  There is one intermediate
    # control polygon per *data* row, each containing ``num_ctrlpt_u`` points.
    ctrlpts_u = np.zeros((num_datapt_v, num_ctrlpt_u, ndim))
    matrix_nu = np.zeros((num_datapt_u - 2, num_ctrlpt_u - 2))
    for i in range(1, num_datapt_u - 1):
        for j in range(1, num_ctrlpt_u - 1):
            matrix_nu[i - 1, j - 1] = one_basis_function(j, uk[i], degree_u, knot_vector_u)

    # Compute Nu transpose
    nt_nu = matrix_nu.T @ matrix_nu
    # Compute NTNu matrix
    lu_lu_u = np.linalg.cholesky(nt_nu)

    for v_idx in range(num_datapt_v):
        points_u = points[v_idx, :, :]
        ctrlpts_u[v_idx, 0, :] = points_u[0]
        ctrlpts_u[v_idx, -1, :] = points_u[-1]

        rk = points_u[1:-1] - (
            np.outer([one_basis_function(0, u, degree_u, knot_vector_u) for u in uk[1:-1]], points_u[0])
            + np.outer(
                [one_basis_function(num_ctrlpt_u - 1, u, degree_u, knot_vector_u) for u in uk[1:-1]], points_u[-1]
            )
        )

        ru = matrix_nu.T @ rk
        for d in range(ndim):
            y = np.linalg.solve(lu_lu_u, ru[:, d])
            x = np.linalg.solve(lu_lu_u.T, y)
            ctrlpts_u[v_idx, 1:-1, d] = x

    # Construct matrix Nv
    ctrlpts_v = np.zeros((num_ctrlpt_v, num_ctrlpt_u, ndim))
    matrix_nv = np.zeros((num_datapt_v - 2, num_ctrlpt_v - 2))
    for i in range(1, num_datapt_v - 1):
        for j in range(1, num_ctrlpt_v - 1):
            matrix_nv[i - 1, j - 1] = one_basis_function(j, vl[i], degree_v, knot_vector_v)

    # Compute Nv transpose
    nt_nv = matrix_nv.T @ matrix_nv
    lu_lu_v = np.linalg.cholesky(nt_nv)  # Cholesky decomposition

    # Fit v-direction
    for u_idx in range(num_ctrlpt_u):
        points_v = ctrlpts_u[:, u_idx, :]
        ctrlpts_v[0, u_idx, :] = points_v[0]
        ctrlpts_v[-1, u_idx, :] = points_v[-1]

        rk = points_v[1:-1] - (
            np.outer([one_basis_function(0, v, degree_v, knot_vector_v) for v in vl[1:-1]], points_v[0])
            + np.outer(
                [one_basis_function(num_ctrlpt_v - 1, v, degree_v, knot_vector_v) for v in vl[1:-1]], points_v[-1]
            )
        )

        rv = matrix_nv.T @ rk
        for d in range(ndim):
            y = np.linalg.solve(lu_lu_v, rv[:, d])
            x = np.linalg.solve(lu_lu_v.T, y)
            ctrlpts_v[1:-1, u_idx, d] = x
    ctrlpts_v = ctrlpts_v.reshape(-1, ndim)

    # Generate B-spline surface
    return _NurbsVolume_nurbs_surface_owner.nurbs_surface(degree_u, degree_v, knot_vector_u, knot_vector_v, ctrlpts_v)


def precompute_curve(
    points, degree, knot_vector="uniform", control_points=None, fixed=None, use_centripetal=False, closed=False
):
    if not isinstance(points, (list, tuple, np.ndarray)):
        raise TypeError("Data points must be a list or a tuple")
    points = np.asarray(points)

    if degree <= 1:
        raise ValueError("Degree must be larger than 1")

    num_datapt = points.shape[0]
    num_ctrlpt = degree + 1
    if isinstance(control_points, (int, float)):
        num_ctrlpt = int(control_points)
        control_points = None
    elif isinstance(control_points, (list, tuple, np.ndarray)):
        control_points = np.asarray(control_points)
        num_ctrlpt = control_points.shape[0]

    if num_ctrlpt <= degree:
        raise ValueError("The number of control points must be larger than degree")

    if not fixed is None:
        fixed = np.asarray(fixed).astype(bool)
        if fixed.shape[0] != num_datapt:
            raise ValueError(f"The dimension of fixed array should be {num_datapt}")

    if isinstance(knot_vector, str):
        if knot_vector == "scale":
            uk = compute_params_curve(points, use_centripetal)
            knot_vector = compute_curve_knot_vector_one(degree, num_datapt, num_ctrlpt, uk)
        elif knot_vector == "uniform":
            knot_vector = compute_uniform_knot_vector(degree, num_ctrlpt)
        else:
            raise ValueError("knot_vector must be scale or uniform")
    elif isinstance(knot_vector, (np.ndarray, list, tuple)):
        knot_vector = np.asarray(knot_vector)

    if control_points is None:
        if not fixed is None:
            uk = compute_params_curve(points, use_centripetal)
            control_points = constraint_curve_fitting(num_ctrlpt, degree, knot_vector, uk, points, fixed)
        else:
            if closed:
                control_points = compute_close_curve_control_points(points, num_ctrlpt)
            else:
                control_points = compute_curve_control_points(points, num_ctrlpt)
    return bspline_curve(degree, knot_vector, control_points, closed)


def curve_regularisation_assemble(curve: SplineCurve, smoothness):
    from src.nurbs.SplineCurve import SplineCurve

    ncp = curve.num_ctrlpts - 2 * (curve.degree - 1) if curve.closed else curve.num_ctrlpts
    cage_reg = curve.num_ctrlpts - 2 * (curve.degree - 1) if curve.closed else curve.num_ctrlpts - 2
    coefficient = np.zeros((curve.dimension * cage_reg, curve.dimension * ncp))
    samples = np.zeros(curve.dimension * cage_reg)
    row = 0
    for j in range(1, cage_reg + 1):
        for dim in range(curve.dimension):
            coefficient[row, curve.dimension * (j % ncp) + dim] += -2.0 * smoothness
            coefficient[row, curve.dimension * ((j - 1) % ncp) + dim] += 1.0 * smoothness
            coefficient[row, curve.dimension * ((j + 1) % ncp) + dim] += 1.0 * smoothness
            row += 1
    return coefficient, samples


def curve_pdm_assemble(points, curve: SplineCurve, parameterize_data=None):
    from src.nurbs.SplineCurve import SplineCurve

    num_datapt = points.shape[0]
    if num_datapt > 0:
        if parameterize_data is None:
            parameterize_data = curve.paramaterize(points)
        ncp = curve.num_ctrlpts - 2 * (curve.degree - 1) if curve.closed else curve.num_ctrlpts
        coefficient = np.zeros((curve.dimension * num_datapt, curve.dimension * ncp))
        samples = np.zeros(curve.dimension * num_datapt)
        row = 0
        for point, para in zip(points, parameterize_data):
            span = find_span(curve.num_ctrlpts, para, curve.degree, curve.knot_vector) - curve.degree
            coeff = curve.coefficient(para)

            for dim in range(curve.dimension):
                for deg in range(curve.degree + 1):
                    coefficient[row, curve.dimension * ((span + deg) % ncp) + dim] += coeff[deg]
                samples[row] += point[dim]
                row += 1
        return coefficient, samples
    else:
        return np.array([]), np.array([])


def curve_tdm_assemble(points, curve: SplineCurve, parameterize_data=None):
    from src.nurbs.SplineCurve import SplineCurve

    num_datapt = points.shape[0]
    if num_datapt > 0:
        if parameterize_data is None:
            parameterize_data = curve.paramaterize(points)
        ncp = curve.num_ctrlpts - 2 * (curve.degree - 1) if curve.closed else curve.num_ctrlpts
        coefficient = np.zeros((curve.dimension * num_datapt, curve.dimension * ncp))
        samples = np.zeros(curve.dimension * num_datapt)
        row = 0
        for point, para in zip(points, parameterize_data):
            span = find_span(curve.num_ctrlpts, para, curve.degree, curve.knot_vector) - curve.degree
            coeff = curve.coefficient(para)
            normal = curve.normal(para)

            for dim in range(curve.dimension):
                for deg in range(curve.degree + 1):
                    coefficient[row, curve.dimension * ((span + deg) % ncp) + dim] += coeff[deg] * normal[dim]
                samples[row] += normal[dim] * point[dim]
                row += 1
        return coefficient, samples
    else:
        return np.array([]), np.array([])


def curve_sdm_assemble(points, curve: SplineCurve, parameterize_data=None):
    from src.nurbs.SplineCurve import SplineCurve

    num_datapt = points.shape[0]
    if num_datapt > 0:
        if parameterize_data is None:
            parameterize_data = curve.paramaterize(points)
        ncp = curve.num_ctrlpts - 2 * (curve.degree - 1) if curve.closed else curve.num_ctrlpts
        coefficient = np.zeros((2 * curve.dimension * num_datapt, curve.dimension * ncp))
        samples = np.zeros(2 * curve.dimension * num_datapt)
        row = 0
        for point, para in zip(points, parameterize_data):
            span = find_span(curve.num_ctrlpts, para, curve.degree, curve.knot_vector) - curve.degree
            coeff = curve.coefficient(para)
            kappa, curvature_center = curve.curvature(para, get_center=True)
            position = curve.single_point(para)
            normal = curve.normal(para)
            tangent = curve.tangent(para)
            distance = np.linalg.norm(point - position)

            for dim in range(curve.dimension):
                for deg in range(curve.degree + 1):
                    coefficient[row, curve.dimension * ((span + deg) % ncp) + dim] += coeff[deg] * normal[dim]
                samples[row] += normal[dim] * point[dim]
                row += 1

            multiplier = distance * kappa / (1 + distance * kappa)
            if not check_same_side(curvature_center, point, position):
                for dim in range(curve.dimension):
                    for deg in range(curve.degree + 1):
                        coefficient[row, curve.dimension * ((span + deg) % ncp) + dim] += (
                            multiplier * multiplier * coeff[deg] * tangent[dim]
                        )
                    samples[row] += multiplier * multiplier * tangent[dim] * point[dim]
                    row += 1
        if row < coefficient.shape[0]:
            coefficient = coefficient[:row, :]
            samples = samples[:row]
        return coefficient, samples
    else:
        return np.array([]), np.array([])


def curve_dw_assemble(points, curve: SplineCurve):
    from src.nurbs.SplineCurve import SplineCurve


def choose_assemble_method(method):
    if method == "pdm":
        return curve_pdm_assemble
    elif method == "tdm":
        return curve_tdm_assemble
    elif method == "sdm":
        return curve_sdm_assemble
    else:
        raise ValueError("method must be pdm, tdm or sdm")


def bspline_curve_fitting_least_squares(fitting: CurvePoints, curve: SplineCurve, fix_dofs=[], method="pdm"):
    from src.nurbs.CurvePoints import CurvePoints
    from src.nurbs.SplineCurve import SplineCurve

    points = fitting.points.copy()
    while points.shape[0] < curve.num_ctrlpts:
        points = insert_midpoints(points)

    assemble = choose_assemble_method(method)
    cp_red = curve.degree - 1 if curve.closed else 0
    coeff, sample = assemble(points, curve)
    coeffr, sampler = curve_regularisation_assemble(curve, fitting.smoothness)

    A = np.concatenate((coeff, coeffr))
    b = np.concatenate((sample, sampler))
    free_dofs = np.array([i for i in range(A.shape[1]) if i not in fix_dofs])

    A_f = A[:, free_dofs]
    A_b = A[:, fix_dofs]
    b_mod = b - A_b @ curve.control_points.reshape(-1)[fix_dofs]
    result_free = solve_using_cod(A_f, b_mod)
    result = np.zeros(A.shape[1])
    result[free_dofs] = result_free
    result[fix_dofs] = curve.control_points.reshape(-1)[fix_dofs]

    ranges = np.arange(cp_red, curve.num_ctrlpts - cp_red, 1)
    diff = result.reshape(-1, curve.dimension) - curve.control_points[ranges]

    curve.control_points[ranges] += diff
    if curve.closed:
        for j in range(cp_red):
            temp = curve.control_points[curve.num_ctrlpts - 1 - cp_red + j]
            curve.control_points[j] = temp
            temp = curve.control_points[cp_red - j]
            curve.control_points[curve.num_ctrlpts - 1 - j] = temp
    return curve


def nurbs_curve_fitting_gauss_newton(
    fitting: CurvePoints, curve: SplineCurve, fix_dofs=[], tol=2e-2, iteration=100, method="pdm"
):
    from src.nurbs.CurvePoints import CurvePoints
    from src.nurbs.SplineCurve import SplineCurve

    iter = 0
    assemble = choose_assemble_method(method)
    parameterize_data = curve.paramaterize(fitting.points)
    dist_diff = fitting.weights * (
        np.array([curve.single_point(para) for para in parameterize_data]) - fitting.points
    ).reshape(-1, 1)
    reg_diff = fitting.penalty * (
        curve.control_points[2:] + curve.control_points[:-2] - 2.0 * curve.control_points[1:-1]
    ).reshape(-1, 1)
    err = np.sum(np.sqrt(dist_diff * dist_diff)) + np.sum(np.sqrt(reg_diff * reg_diff))

    err_update = 1.0
    while iter < iteration and err > tol:
        jacobian_p, _ = assemble(fitting.points, curve, parameterize_data)
        jacobian_reg, _ = curve_regularisation_assemble(curve, fitting.smoothness)

        hess = (
            fitting.weights * fitting.weights * jacobian_p.T @ jacobian_p
        )  # + fitting.penalty * fitting.penalty * jacobianReg.T @ jacobianReg
        grad = fitting.weights * jacobian_p.T @ dist_diff  # + fitting.penalty * jacobianReg.T @ reg_diff

        hess[fix_dofs, :] = 0.0
        hess[:, fix_dofs] = 0.0
        hess[fix_dofs, fix_dofs] = 1.0
        grad[fix_dofs] = 0.0
        delta = np.linalg.solve(hess, -grad).reshape(-1, curve.dimension)
        curve.control_points += delta

        alpha = 1.0
        original_ctrlpts = curve.control_points.copy()
        for _ in range(50):
            curve.control_points += alpha * delta
            parameterize_data = curve.paramaterize(fitting.points)
            dist_diff = fitting.weights * (
                np.array([curve.single_point(para) for para in parameterize_data]) - fitting.points
            ).reshape(-1, 1)
            reg_diff = fitting.penalty * (
                curve.control_points[2:] + curve.control_points[:-2] - 2.0 * curve.control_points[1:-1]
            ).reshape(-1, 1)
            err_update = np.sum(np.sqrt(dist_diff * dist_diff)) + np.sum(np.sqrt(reg_diff * reg_diff))
            if err_update < err:
                break
            else:
                curve.control_points = original_ctrlpts
                alpha *= 0.5
        iter += 1
        err, err_update = err_update, err
        if err > err_update:
            break
        iter += 1
    return curve


def precompute_surface(
    points,
    degree_u,
    degree_v,
    knot_vector_u=None,
    knot_vector_v=None,
    control_points=None,
    bounding_quadrilateral_method=0,
):
    if not isinstance(points, (list, tuple, np.ndarray)):
        raise TypeError("Data points must be a list or a tuple")

    if degree_u <= 1 or degree_v <= 1:
        raise ValueError("Degree must be larger than 1")

    num_ctrlpt_u = degree_u + 1
    num_ctrlpt_v = degree_v + 1
    if isinstance(control_points, (int, float)):
        num_ctrlpt_u = int(control_points)
        num_ctrlpt_v = int(control_points)
        control_points = None
    elif isinstance(control_points, (list, tuple, np.ndarray)):
        control_points = np.asarray(control_points)
        if control_points.shape[0] == 2:
            num_ctrlpt_u = int(control_points[0])
            num_ctrlpt_v = int(control_points[1])
            control_points = None
        elif control_points.ndim == 2:
            num_ctrlpt_v = control_points.shape[0]
            num_ctrlpt_u = control_points.shape[1]

    if num_ctrlpt_u < degree_u or num_ctrlpt_v < degree_v:
        raise ValueError("The number of control points must be larger or equal to degree")

    if knot_vector_u is None:
        knot_vector_u = compute_uniform_knot_vector(degree_u, num_ctrlpt_u)
    else:
        # TODO: Check knot vector
        pass

    if knot_vector_v is None:
        knot_vector_v = compute_uniform_knot_vector(degree_v, num_ctrlpt_v)
    else:
        # TODO: Check knot vector
        pass
    if points.shape[1] == 2:
        if control_points is None:
            control_points = compute_surface_control_points_2d(
                points,
                resolution=(num_ctrlpt_u, num_ctrlpt_v),
                bounding_quadrilateral_method=bounding_quadrilateral_method,
            )
        surf = bspline_surface(degree_u, degree_v, knot_vector_u, knot_vector_v, control_points)
        return surf
    elif points.shape[1] == 3:
        if control_points is None:
            control_points = compute_surface_control_points_3d(
                points,
                resolution=(num_ctrlpt_u, num_ctrlpt_v),
                bounding_quadrilateral_method=bounding_quadrilateral_method,
            )
        return bspline_surface(degree_u, degree_v, knot_vector_u, knot_vector_v, control_points)


def surface_pdm_assemble(weights, points, parameterize_data, surface: SplineSurface, is_boundary=None):
    from src.nurbs.SplineSurface import SplineSurface

    if points.shape[0] > 0:
        coefficient = np.zeros(
            (surface.dimension * points.shape[0], surface.dimension * surface.num_ctrlpts_u * surface.num_ctrlpts_v)
        )
        sample = np.zeros(surface.dimension * points.shape[0])
        row = 0
        for point, para in zip(points, parameterize_data):
            span_u = (
                find_span(surface.num_ctrlpts_u, para[0], surface.degree_u, surface.knot_vector_u) - surface.degree_u
            )
            span_v = (
                find_span(surface.num_ctrlpts_v, para[1], surface.degree_v, surface.knot_vector_v) - surface.degree_v
            )
            coeff = surface.coefficient(*para)

            for dim in range(surface.dimension):
                for degv in range(surface.degree_v + 1):
                    for degu in range(surface.degree_u + 1):
                        linear_index = degu + degv * (surface.degree_u + 1)
                        coefficient[
                            row, surface.dimension * (span_u + degu + (span_v + degv) * surface.num_ctrlpts_u) + dim
                        ] += (weights * coeff[linear_index])
                sample[row] += weights * point[dim]
                row += 1
        return coefficient, sample
    else:
        return np.array([]), np.array([])


def surface_tdm_assemble(weights, points, parameterize_data, surface: SplineSurface, is_boundary=False):
    from src.nurbs.SplineSurface import SplineSurface

    if points.shape[0] > 0:
        coefficient = np.zeros(
            (surface.dimension * points.shape[0], surface.dimension * surface.num_ctrlpts_u * surface.num_ctrlpts_v)
        )
        sample = np.zeros(surface.dimension * points.shape[0])
        row = 0
        for point, para in zip(points, parameterize_data):
            span_u = (
                find_span(surface.num_ctrlpts_u, para[0], surface.degree_u, surface.knot_vector_u) - surface.degree_u
            )
            span_v = (
                find_span(surface.num_ctrlpts_v, para[1], surface.degree_v, surface.knot_vector_v) - surface.degree_v
            )
            coeff = surface.coefficient(*para)
            if is_boundary:
                normal = surface.boundary_normal(*para)
            else:
                normal = surface.normal(*para)

            for dim in range(surface.dimension):
                for degv in range(surface.degree_v + 1):
                    for degu in range(surface.degree_u + 1):
                        linear_index = degu + degv * (surface.degree_u + 1)
                        coefficient[
                            row, surface.dimension * (span_u + degu + (span_v + degv) * surface.num_ctrlpts_u) + dim
                        ] += (weights * normal[dim] * coeff[linear_index])
                sample[row] += weights * normal[dim] * point[dim]
                row += 1
        return coefficient, sample
    else:
        return np.array([]), np.array([])


def surface_partial_u_assemble(points, parameterize_data, surface: SplineSurface, is_boundary=False):
    from src.nurbs.SplineSurface import SplineSurface

    if points.shape[0] > 0:
        coefficient = np.zeros((surface.dimension * points.shape[0], points.shape[0]))
        row = 0
        for para in parameterize_data:
            if is_boundary:
                dloss_du = surface.boundary_derivative(para[0], para[1])
            else:
                dloss_du = surface.derivative(para[0], para[1])

            for dim in range(surface.dimension):
                coefficient[row, int(row // surface.dimension)] += dloss_du[dim]
                row += 1
        return coefficient
    else:
        return np.array([]), np.array([])


def surface_interior_regularisation(interior_smoothness, surface: SplineSurface):
    from src.nurbs.SplineSurface import SplineSurface

    cage_reg = (surface.num_ctrlpts_u - 2) * (surface.num_ctrlpts_v - 2)
    if cage_reg > 0:
        coefficient = np.zeros(
            (surface.dimension * cage_reg, surface.dimension * surface.num_ctrlpts_u * surface.num_ctrlpts_v)
        )
        sample = np.zeros(surface.dimension * cage_reg)
        row = 0
        for i in range(1, surface.num_ctrlpts_u - 1):
            for j in range(1, surface.num_ctrlpts_v - 1):
                for dim in range(surface.dimension):
                    coefficient[row, surface.dimension * (i + 0 + (j + 0) * surface.num_ctrlpts_u) + dim] += (
                        -4.0 * interior_smoothness
                    )
                    coefficient[row, surface.dimension * (i + 0 + (j - 1) * surface.num_ctrlpts_u) + dim] += (
                        1.0 * interior_smoothness
                    )
                    coefficient[row, surface.dimension * (i + 0 + (j + 1) * surface.num_ctrlpts_u) + dim] += (
                        1.0 * interior_smoothness
                    )
                    coefficient[row, surface.dimension * (i - 1 + (j + 0) * surface.num_ctrlpts_u) + dim] += (
                        1.0 * interior_smoothness
                    )
                    coefficient[row, surface.dimension * (i + 1 + (j + 0) * surface.num_ctrlpts_u) + dim] += (
                        1.0 * interior_smoothness
                    )
                    row += 1
        return coefficient, sample
    else:
        return np.array([]), np.array([])


def surface_boundary_regularisation(boundary_smoothness, surface: SplineSurface):
    from src.nurbs.SplineSurface import SplineSurface

    cage_reg_bound = 2 * (surface.num_ctrlpts_u - 1) + 2 * (surface.num_ctrlpts_v - 1)
    if cage_reg_bound > 0:
        coefficient = np.zeros(
            (surface.dimension * cage_reg_bound, surface.dimension * surface.num_ctrlpts_u * surface.num_ctrlpts_v)
        )
        sample = np.zeros(surface.dimension * cage_reg_bound)
        row = 0
        for i in range(1, surface.num_ctrlpts_u - 1):
            for dim in range(surface.dimension):
                coefficient[
                    row, surface.dimension * (i + 0 + (surface.num_ctrlpts_v - 1) * surface.num_ctrlpts_u) + dim
                ] += (-2.0 * boundary_smoothness)
                coefficient[
                    row, surface.dimension * (i - 1 + (surface.num_ctrlpts_v - 1) * surface.num_ctrlpts_u) + dim
                ] += (1.0 * boundary_smoothness)
                coefficient[
                    row, surface.dimension * (i + 1 + (surface.num_ctrlpts_v - 1) * surface.num_ctrlpts_u) + dim
                ] += (1.0 * boundary_smoothness)
                row += 1

        for i in range(1, surface.num_ctrlpts_u - 1):
            for dim in range(surface.dimension):
                coefficient[row, surface.dimension * (i + 0) + dim] += -2.0 * boundary_smoothness
                coefficient[row, surface.dimension * (i - 1) + dim] += 1.0 * boundary_smoothness
                coefficient[row, surface.dimension * (i + 1) + dim] += 1.0 * boundary_smoothness
                row += 1

        for j in range(1, surface.num_ctrlpts_v - 1):
            for dim in range(surface.dimension):
                coefficient[
                    row, surface.dimension * (surface.num_ctrlpts_u - 1 + (j + 0) * surface.num_ctrlpts_u) + dim
                ] += (-2.0 * boundary_smoothness)
                coefficient[
                    row, surface.dimension * (surface.num_ctrlpts_u - 1 + (j - 1) * surface.num_ctrlpts_u) + dim
                ] += (1.0 * boundary_smoothness)
                coefficient[
                    row, surface.dimension * (surface.num_ctrlpts_u - 1 + (j + 1) * surface.num_ctrlpts_u) + dim
                ] += (1.0 * boundary_smoothness)
                row += 1

        for j in range(1, surface.num_ctrlpts_v - 1):
            for dim in range(surface.dimension):
                coefficient[row, surface.dimension * ((j + 0) * surface.num_ctrlpts_u) + dim] += (
                    -2.0 * boundary_smoothness
                )
                coefficient[row, surface.dimension * ((j - 1) * surface.num_ctrlpts_u) + dim] += (
                    1.0 * boundary_smoothness
                )
                coefficient[row, surface.dimension * ((j + 1) * surface.num_ctrlpts_u) + dim] += (
                    1.0 * boundary_smoothness
                )
                row += 1

        for dim in range(surface.dimension):
            coefficient[row, surface.dimension * (0 + 0 * surface.num_ctrlpts_u) + dim] += -4.0 * boundary_smoothness
            coefficient[row, surface.dimension * (1 + 0 * surface.num_ctrlpts_u) + dim] += 2.0 * boundary_smoothness
            coefficient[row, surface.dimension * (0 + 1 * surface.num_ctrlpts_u) + dim] += 2.0 * boundary_smoothness
            row += 1

        for dim in range(surface.dimension):
            coefficient[row, surface.dimension * (surface.num_ctrlpts_u - 1 + 0 * surface.num_ctrlpts_u) + dim] += (
                -4.0 * boundary_smoothness
            )
            coefficient[row, surface.dimension * (surface.num_ctrlpts_u - 2 + 0 * surface.num_ctrlpts_u) + dim] += (
                2.0 * boundary_smoothness
            )
            coefficient[row, surface.dimension * (surface.num_ctrlpts_u - 1 + 1 * surface.num_ctrlpts_u) + dim] += (
                2.0 * boundary_smoothness
            )
            row += 1

        for dim in range(surface.dimension):
            coefficient[
                row,
                surface.dimension * (surface.num_ctrlpts_u - 1 + (surface.num_ctrlpts_v - 1) * surface.num_ctrlpts_u)
                + dim,
            ] += (
                -4.0 * boundary_smoothness
            )
            coefficient[
                row,
                surface.dimension * (surface.num_ctrlpts_u - 2 + (surface.num_ctrlpts_v - 1) * surface.num_ctrlpts_u)
                + dim,
            ] += (
                2.0 * boundary_smoothness
            )
            coefficient[
                row,
                surface.dimension * (surface.num_ctrlpts_u - 1 + (surface.num_ctrlpts_v - 2) * surface.num_ctrlpts_u)
                + dim,
            ] += (
                2.0 * boundary_smoothness
            )
            row += 1

        for dim in range(surface.dimension):
            coefficient[row, surface.dimension * (0 + (surface.num_ctrlpts_v - 1) * surface.num_ctrlpts_u) + dim] += (
                -4.0 * boundary_smoothness
            )
            coefficient[row, surface.dimension * (1 + (surface.num_ctrlpts_v - 1) * surface.num_ctrlpts_u) + dim] += (
                2.0 * boundary_smoothness
            )
            coefficient[row, surface.dimension * (0 + (surface.num_ctrlpts_v - 2) * surface.num_ctrlpts_u) + dim] += (
                2.0 * boundary_smoothness
            )
            row += 1
        return coefficient, sample
    else:
        return np.array([]), np.array([])


def surface_common_boundary(slave, column_prefix, common_index, common_boundary, surfaces):
    if common_boundary.shape[0] > 0:
        coefficient = np.zeros((surfaces[slave].dimension * common_boundary.shape[0], column_prefix[-1]))
        sample = np.zeros(surfaces[slave].dimension * common_boundary.shape[0])
        row = 0
        for master, point in zip(common_index, common_boundary):
            if (
                surfaces[slave].degree_u != surfaces[slave].degree_v
                or surfaces[master].degree_u != surfaces[master].degree_v
                or surfaces[slave].degree_u != surfaces[master].degree_u
            ):
                raise SimulationStateError(
                    f"Surface {slave} and surface {master} must have the same degree along both u and v directions"
                )
            parameter1, _ = surfaces[slave].boundary_inversion(point)
            parameter2, _ = surfaces[master].boundary_inversion(point)
            tangent_u1, tangent_v1 = surfaces[slave].tangent(*parameter1[0])
            tangent_u2, tangent_v2 = surfaces[master].tangent(*parameter2[0])

            tangent = [tangent_u1, tangent_v1]
            if parameter1[0][0] == 0.0 or parameter1[0][0] == 1.0:
                tangent[0] = tangent_v1
            if parameter1[0][1] == 0.0 or parameter1[0][1] == 1.0:
                tangent[0] = tangent_u1
            if parameter2[0][0] == 0.0 or parameter2[0][0] == 1.0:
                tangent[1] = tangent_v2
            if parameter2[0][1] == 0.0 or parameter2[0][1] == 1.0:
                tangent[1] = tangent_u2

            columns = [column_prefix[slave], column_prefix[master]]
            parameters = [parameter1[0], parameter2[0]]
            surfs = [surfaces[slave], surfaces[master]]
            weights = abs(np.dot(tangent[0], tangent[1]))
            coefficient, sample = surface_parameter_constraint(
                weights, columns, parameters, surfs, coefficient, sample, row
            )
            row += 3
        return coefficient, sample
    else:
        return np.array([]), np.array([])


def surface_parameter_constraint(weights, columns, parameters, surfaces, coefficient, sample, row):
    for side in range(2):
        surf = surfaces[side]
        coeff = surf.coefficient(*parameters[side])
        span_u = find_span(surf.num_ctrlpts_u, parameters[side][0], surf.degree_u, surf.knot_vector_u) - surf.degree_u
        span_v = find_span(surf.num_ctrlpts_v, parameters[side][1], surf.degree_v, surf.knot_vector_v) - surf.degree_v
        signs = 1.0 if side == 0 else -1.0

        for degu in range(surf.degree_u + 1):
            for degv in range(surf.degree_v + 1):
                linear_index = degu + degv * (surf.degree_u + 1)
                coefficient[
                    row + 0, columns[side] + surf.dimension * (span_u + degu + (span_v + degv) * surf.num_ctrlpts_u) + 0
                ] += (weights * signs * coeff[linear_index])
                coefficient[
                    row + 1, columns[side] + surf.dimension * (span_u + degu + (span_v + degv) * surf.num_ctrlpts_u) + 1
                ] += (weights * signs * coeff[linear_index])
                coefficient[
                    row + 2, columns[side] + surf.dimension * (span_u + degu + (span_v + degv) * surf.num_ctrlpts_u) + 2
                ] += (weights * signs * coeff[linear_index])
    return coefficient, sample


def choose_surface_assemble_method(method):
    if method == "pdm":
        return surface_pdm_assemble
    elif method == "tdm":
        raise ValueError("method must be pdm")
    else:
        raise ValueError("method must be pdm")


def bspline_surface_fitting_least_squares(fitting: SurfacePoints, surface: SplineSurface, method="pdm"):
    from src.nurbs.SplineSurface import SplineSurface
    from src.nurbs.SurfacePoints import SurfacePoints

    assemble = choose_surface_assemble_method(method)
    coeff, sample = [], []
    if fitting.boundary.shape[0] > 0:
        boundary_parameterize_data, _ = surface.boundary_inversion(fitting.boundary)
        coeff_boundary, sample_boundary = assemble(
            fitting.boundary_weight, fitting.boundary, boundary_parameterize_data, surface, is_boundary=True
        )
        coeff.append(coeff_boundary)
        sample.append(sample_boundary)
    if fitting.interior.shape[0] > 0:
        interior_parameterize_data, _ = surface.inversion(fitting.interior)
        coeff_interior, sample_interior = assemble(
            fitting.interior_weight, fitting.interior, interior_parameterize_data, surface
        )
        coeff.append(coeff_interior)
        sample.append(sample_interior)
    coeff_reg, sample_reg = surface_interior_regularisation(fitting.interior_smoothness, surface)
    coeff_breg, sample_breg = surface_boundary_regularisation(fitting.boundary_smoothness, surface)
    coeff.append(coeff_reg)
    coeff.append(coeff_breg)
    sample.append(sample_reg)
    sample.append(sample_breg)
    result = solve_using_cod(np.concatenate(coeff, axis=0), np.concatenate(sample, axis=0))
    diff = result.reshape(-1, surface.dimension) - surface.control_points
    surface.control_points += diff
    return surface


def bspline_multisurface_fitting_least_squares(fittings: list, surfaces: list, method="pdm"):
    if not isinstance(fittings, list):
        fittings = list(fittings)
    if not isinstance(surfaces, list):
        surfaces = list(surfaces)

    if len(fittings) != len(surfaces):
        raise ValueError("Internal error")

    row_prefix, column_prefix = [], []
    num_interior = num_boundary = num_ctrlpts = num_common_bound = cage_reg = cage_reg_bound = 0
    for fitting, surface in zip(fittings, surfaces):
        row_prefix.append(
            surface.dimension * (num_interior + num_boundary + cage_reg + cage_reg_bound + num_common_bound)
        )
        column_prefix.append(surface.dimension * num_ctrlpts)
        num_interior += fitting.interior.shape[0]
        num_boundary += fitting.boundary.shape[0]
        num_ctrlpts += surface.num_ctrlpts_u * surface.num_ctrlpts_v
        if len(surfaces) > 1:
            num_common_bound += fitting.common_boundary.shape[0]
        else:
            fitting.common_index = np.array([])
            fitting.common_boundary = np.array([])
        cage_reg += (surface.num_ctrlpts_u - 2) * (surface.num_ctrlpts_v - 2)
        cage_reg_bound += 2 * (surface.num_ctrlpts_u - 1) + 2 * (surface.num_ctrlpts_v - 1)
    row_prefix.append(surface.dimension * (num_interior + num_boundary + cage_reg + cage_reg_bound + num_common_bound))
    column_prefix.append(surface.dimension * num_ctrlpts)

    assemble = choose_surface_assemble_method(method)
    coeffs = lil_matrix((row_prefix[-1], column_prefix[-1]))
    samples = []
    for object_id, (fitting, surface) in enumerate(zip(fittings, surfaces)):
        coeff_boundary, coeff_interior = np.array([]), np.array([])
        sample_boundary, sample_interior = np.array([]), np.array([])
        if fitting.boundary.shape[0] > 0:
            boundary_parameterize_data, _ = surface.boundary_inversion(fitting.boundary)
            coeff_boundary, sample_boundary = assemble(
                fitting.boundary, boundary_parameterize_data, surface, is_boundary=True
            )
        if fitting.interior.shape[0] > 0:
            interior_parameterize_data, _ = surface.inversion(fitting.interior)
            coeff_interior, sample_interior = assemble(fitting.interior, interior_parameterize_data, surface)
        coeff_reg, sample_reg = surface_interior_regularisation(fitting.interior_smoothness, surface)
        coeff_breg, sample_breg = surface_boundary_regularisation(fitting.boundary_smoothness, surface)
        coeff_common, sample_common = surface_common_boundary(
            object_id, column_prefix, fitting.common_index, fitting.common_boundary, surfaces
        )

        coeffs[
            row_prefix[object_id] : row_prefix[object_id + 1] - surface.dimension * fitting.common_boundary.shape[0],
            column_prefix[object_id] : column_prefix[object_id + 1],
        ] += np.concatenate((coeff_boundary, coeff_interior, coeff_reg, coeff_breg))
        if fitting.common_boundary.shape[0] > 0:
            coeffs[
                row_prefix[object_id + 1]
                - surface.dimension * fitting.common_boundary.shape[0] : row_prefix[object_id + 1],
                :,
            ] += coeff_common
        samples.append(np.concatenate((sample_boundary, sample_interior, sample_reg, sample_breg, sample_common)))
    result = lsqr(coeffs.tocsr(), np.concatenate(([sample for sample in samples])))[0]
    for surf_id, surface in enumerate(surfaces):
        surface.control_points = result[column_prefix[surf_id] : column_prefix[surf_id + 1]].reshape(
            -1, surface.dimension
        )
    return surfaces


def bspline_surface_fitting_gauss_newton2d(
    fitting: SurfacePoints, surface: SplineSurface, tol=2e-2, rho=0.5, iteration=100, method="pdm", verbose=False
):
    from src.nurbs.SplineSurface import SplineSurface
    from src.nurbs.element.NurbsSurfaceElement import NurbsSurfaceElement
    from src.nurbs.SurfacePoints import SurfacePoints

    energy = MIPS()
    surface.activate_boundary()
    surface.update_boundary()
    element = NurbsSurfaceElement(surface, energy=energy)

    assemble = choose_surface_assemble_method(method)
    boundary_parameterize_data, boundary_diff = np.array([]), np.array([])
    boundary_parameterize_data, _ = surface.boundary_inversion(fitting.boundary)
    boundary_diff = (
        np.array([surface.single_point(*para) for para in boundary_parameterize_data]) - fitting.boundary
    ).reshape(-1, 1)
    deformation_gradient = element.compute_deformation_gradient(element.vertice)
    energy = element.surface_energy(deformation_gradient)
    err = np.sum(np.sqrt(boundary_diff * boundary_diff)) + fitting.penalty * energy

    iter = 0
    err_update = 1.0
    while iter < iteration and err > tol:
        dp_boundary, _ = assemble(
            fitting.boundary_weight, fitting.boundary, boundary_parameterize_data, surface, is_boundary=True
        )
        coeff_boundary = csr_matrix(dp_boundary)
        hess = (coeff_boundary.T @ coeff_boundary).tolil()
        grad = coeff_boundary.T @ boundary_diff

        deformation_gradient = element.compute_deformation_gradient(element.vertice)
        ddeformation_dx = element.compute_ddeformation_dx()
        constraint_grad = fitting.penalty * element.surface_energy_gradient(
            deformation_gradient, ddeformation_dx
        ).reshape(-1, 1)
        constriant_hess = fitting.penalty * element.surface_energy_hessian(deformation_gradient, ddeformation_dx)
        hess = hess + constriant_hess
        grad = grad + constraint_grad
        delta = checked_scipy_krylov(cg, hess, -grad, solver_name="NURBS fitting CG")
        delta = delta.reshape(-1, surface.dimension)

        deformation_gradient = element.compute_deformation_gradient(element.vertice)
        alpha = element.compute_step_size(delta, deformation_gradient)
        if alpha < 1.0:
            alpha = alpha * rho

        init_ctrlpts = element.vertice.copy()
        try:
            while alpha > tol:
                disp = alpha * delta
                curr_ctrlpts = init_ctrlpts + disp
                try:
                    element.validate(curr_ctrlpts)
                except ValueError:
                    alpha *= 0.5
                    continue
                element.vertice = curr_ctrlpts.copy()
                surface.control_points = curr_ctrlpts.copy()
                surface.update_boundary()

                deformation_gradient = element.compute_deformation_gradient(element.vertice)
                energy = element.surface_energy(deformation_gradient)
                boundary_parameterize_data, _ = surface.boundary_inversion(fitting.boundary)
                boundary_diff = (
                    np.array([surface.single_point(*para) for para in boundary_parameterize_data]) - fitting.boundary
                ).reshape(-1, 1)

                err_update = np.sum(np.sqrt(boundary_diff * boundary_diff)) + fitting.penalty * energy
                if verbose:
                    print(f"New error is {err_update} and old error is {err}")
                if err_update <= err:
                    break
                alpha *= 0.5
            else:
                raise NonlinearSolveError("Surface fitting found no valid descent step")
        except Exception:
            element.vertice = init_ctrlpts.copy()
            surface.control_points = init_ctrlpts.copy()
            surface.update_boundary()
            raise

        err, err_update = err_update, err
        updating = np.sum(np.sqrt(delta * delta))
        iter += 1

        if verbose:
            print(f"In iteration: {iter}")
            print(f"Step size: {alpha}")
            print(f"Residual error: {err}")
            print(f"Update displacement: {updating}", "\n")

        if updating < tol:
            break

        if alpha <= tol or abs(energy) < tol:
            break

        if iter == iteration:
            raise NonlinearSolveError(f"Convergence failed, the residual error is {err}")
    validate_spline_jacobian(surface)
    return surface


def surface_fitting_least_square_jacobian(weights, boundary_diff, parameterize_data, surface: SplineSurface):
    from src.nurbs.SplineSurface import SplineSurface

    if isinstance(weights, (int, float)):
        weights = np.zeros_like(boundary_diff) + weights
    coefficient = np.zeros(surface.dimension * surface.num_ctrlpts)
    boundary_diff = boundary_diff.reshape(-1, 2)
    for weight, diff, para in zip(weights, boundary_diff, parameterize_data):
        ctrlpt_id = get_ctrlpt_dofs(surface.ctrlpt_id(*para), dim=2)
        coeff = surface.dxdctrlpts(*para)
        coefficient[ctrlpt_id] += weight * coeff @ diff
    return coefficient


def surface_fitting_least_square_hessian(weights, boundary_diff, parameterize_data, surface: SplineSurface):
    from src.nurbs.SplineSurface import SplineSurface

    if isinstance(weights, (int, float)):
        weights = np.zeros_like(boundary_diff) + weights
    coefficient = lil_matrix((surface.dimension * surface.num_ctrlpts, surface.dimension * surface.num_ctrlpts))
    for weight, para in zip(weights, parameterize_data):
        ctrlpt_id = get_ctrlpt_dofs(surface.ctrlpt_id(*para), dim=2)
        index1, index2 = np.meshgrid(ctrlpt_id, ctrlpt_id, indexing="ij")
        coeff = surface.dxdctrlpts(*para)
        coefficient[index2, index1] += weight * weight * (coeff @ coeff.T)
    return coefficient


def surface_fitting_quartic_jacobian(weights, boundary_diff, parameterize_data, surface: SplineSurface):
    from src.nurbs.SplineSurface import SplineSurface

    if isinstance(weights, (int, float)):
        weights = np.zeros_like(boundary_diff) + weights
    coefficient = np.zeros(surface.dimension * surface.num_ctrlpts)
    temp = np.dot(boundary_diff, boundary_diff)
    boundary_diff = boundary_diff.reshape(-1, 2)
    for weight, diff, para in zip(weights, boundary_diff, parameterize_data):
        ctrlpt_id = get_ctrlpt_dofs(surface.ctrlpt_id(*para), dim=2)
        coeff = surface.dxdctrlpts(*para)
        coefficient[ctrlpt_id] += weight * coeff @ diff
    return temp * coefficient


def surface_fitting_quartic_hessian(weights, boundary_diff, parameterize_data, surface: SplineSurface):
    from src.nurbs.SplineSurface import SplineSurface

    if isinstance(weights, (int, float)):
        weights = np.zeros_like(boundary_diff) + weights
    coefficient = lil_matrix((surface.dimension * surface.num_ctrlpts, surface.dimension * surface.num_ctrlpts))
    temp = np.dot(boundary_diff, boundary_diff)
    boundary_diff = boundary_diff.reshape(-1, 2)
    for weight, diff, para in zip(weights, boundary_diff, parameterize_data):
        ctrlpt_id = get_ctrlpt_dofs(surface.ctrlpt_id(*para), dim=2)
        index1, index2 = np.meshgrid(ctrlpt_id, ctrlpt_id, indexing="ij")
        coeff = surface.dxdctrlpts(*para)
        coefficient[index2, index1] += (
            weight * weight * (2.0 * np.outer(coeff @ diff, coeff @ diff) + temp * coeff @ coeff.T)
        )
    return coefficient


def bspline_surface_fitting_newton_raphson2d(
    fitting: SurfacePoints, surface: SplineSurface, tol=2e-2, rho=0.5, iteration=100, verbose=False
):
    from src.nurbs.SplineSurface import SplineSurface
    from src.nurbs.element.NurbsSurfaceElement import NurbsSurfaceElement
    from src.nurbs.SurfacePoints import SurfacePoints

    energy = MIPS()
    surface.activate_boundary()
    surface.update_boundary()
    element = NurbsSurfaceElement(surface, energy=energy)

    boundary_parameterize_data, _ = surface.boundary_inversion(fitting.boundary)
    boundary_diff = (
        np.array([surface.single_point(*para) for para in boundary_parameterize_data]) - fitting.boundary
    ).reshape(-1)
    deformation_gradient = element.compute_deformation_gradient(element.vertice)
    energy = element.surface_energy(deformation_gradient)
    err = np.sum(np.sqrt(boundary_diff * boundary_diff)) + fitting.penalty * energy

    iter = 0
    # err = 1.
    while iter < iteration and err > tol:
        hess = surface_fitting_quartic_hessian(
            fitting.boundary_weight, boundary_diff, boundary_parameterize_data, surface
        )
        grad = surface_fitting_quartic_jacobian(
            fitting.boundary_weight, boundary_diff, boundary_parameterize_data, surface
        )

        deformation_gradient = element.compute_deformation_gradient(element.vertice)
        ddeformation_dx = element.compute_ddeformation_dx()
        constraint_grad = fitting.penalty * element.surface_energy_gradient(deformation_gradient, ddeformation_dx)
        constriant_hess = fitting.penalty * element.surface_energy_hessian(deformation_gradient, ddeformation_dx)
        hess = hess + constriant_hess
        grad = grad + constraint_grad
        delta = checked_scipy_krylov(cg, hess, -grad, solver_name="NURBS fitting CG")
        delta = delta.reshape(-1, surface.dimension)

        deformation_gradient = element.compute_deformation_gradient(element.vertice)
        alpha = element.compute_step_size(delta, deformation_gradient)
        if alpha < 1.0:
            alpha = alpha * rho

        init_ctrlpts = element.vertice.copy()
        try:
            while alpha > tol:
                disp = alpha * delta
                curr_ctrlpts = init_ctrlpts + disp
                try:
                    element.validate(curr_ctrlpts)
                except ValueError:
                    alpha *= 0.5
                    continue
                element.vertice = curr_ctrlpts.copy()
                surface.control_points = curr_ctrlpts.copy()
                surface.update_boundary()

                deformation_gradient = element.compute_deformation_gradient(element.vertice)
                energy = element.surface_energy(deformation_gradient)
                boundary_parameterize_data, _ = surface.boundary_inversion(fitting.boundary)
                boundary_diff = (
                    np.array([surface.single_point(*para) for para in boundary_parameterize_data]) - fitting.boundary
                ).reshape(-1)

                err_update = np.sum(np.sqrt(boundary_diff * boundary_diff)) + fitting.penalty * energy
                if verbose:
                    print(f"New error is {err_update} and old error is {err}")
                if err_update <= err:
                    break
                alpha *= 0.5
            else:
                raise NonlinearSolveError("Surface fitting found no valid descent step")
        except Exception:
            element.vertice = init_ctrlpts.copy()
            surface.control_points = init_ctrlpts.copy()
            surface.update_boundary()
            raise

        err, err_update = err_update, err
        updating = np.sum(np.sqrt(delta * delta))
        iter += 1

        if verbose:
            print(f"In iteration: {iter}")
            print(f"Step size: {alpha}")
            print(f"Residual error: {err}")
            print(f"Residual displacement: {updating}", "\n")

        if updating < tol:
            break

        if alpha <= tol or abs(energy) < tol:
            break

        if iter == iteration:
            surface.visualize(datapts=fitting.boundary)
            raise NonlinearSolveError(f"Convergence failed, the residual error is {err}")
    validate_spline_jacobian(surface)
    return surface


def reorder_by_main_axis(surface: SplineSurface, category_1, category_2, category_3, category_4):
    from src.nurbs.SplineSurface import SplineSurface

    center1 = np.mean(category_1, axis=0)
    center2 = np.mean(category_2, axis=0)
    center3 = np.mean(category_3, axis=0)
    center4 = np.mean(category_4, axis=0)
    centers = [center1, center2, center3, center4]
    points = [category_1, category_2, category_3, category_4]
    index = []
    for j in range(4):
        direction = surface.boundary[j].control_points[-1] - surface.boundary[j].control_points[0]
        tangent = direction / np.linalg.norm(direction)
        norm = np.array([-tangent[1], tangent[0]])
        mean_position = np.mean(surface.boundary[j].control_points, axis=0)
        min_dist, line_number = 1e15, -1
        for i in range(4):
            dist = np.linalg.norm(mean_position - centers[i])
            if dist <= min_dist:
                line_number = i
                min_dist = dist
        index.append(line_number)
    return points[index[0]], points[index[1]], points[index[2]], points[index[3]]


def split_boundary_points(surface, boundary_points, corner_points, visualize=False):
    boundary_points = _prepare_planar_boundary(boundary_points)
    corner_points = np.asarray(corner_points, dtype=float)
    if corner_points.shape != (4, 2) or not np.isfinite(corner_points).all():
        raise ValueError("corner_points must contain four finite 2D coordinates")
    indices = np.argmin(np.linalg.norm(boundary_points[:, None] - corner_points[None, :], axis=2), axis=0)
    if len(np.unique(indices)) != 4:
        raise ValueError("Four distinct boundary corners are required")
    start = indices[0]
    offsets = np.sort((indices - start) % len(boundary_points))
    ordered = np.roll(boundary_points, -start, axis=0)
    ordered = np.vstack((ordered, ordered[0]))
    ends = np.r_[offsets[1:], len(boundary_points)]
    category_1, category_2, category_3, category_4 = [
        ordered[first : last + 1].copy() for first, last in zip(offsets, ends)
    ]
    categories = {
        "Category 1 (A->B)": category_1,
        "Category 2 (B->C)": category_2,
        "Category 3 (C->D)": category_3,
        "Category 4 (D->A)": category_4,
    }

    if visualize:
        plt.figure(figsize=(8, 6))
        colors = ["red", "green", "blue", "orange"]
        labels = list(categories.keys())

        boundary_array = np.concatenate((boundary_points, [boundary_points[0]]))
        plt.plot(boundary_array[:, 0], boundary_array[:, 1], "k-", linewidth=1, label="Boundary")

        for i, (label, points) in enumerate(categories.items()):
            pts = np.asarray(points)
            plt.scatter(pts[:, 0], pts[:, 1], color=colors[i], label=label, s=100)

        for i, ref in enumerate(corner_points):
            plt.scatter(ref[0], ref[1], color="black", marker="x", s=150, label=f"Reference {i+1}" if i == 0 else None)

        plt.legend()
        plt.title("Polygon boundary point classification")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
        plt.close()
    return categories


def point_parameterize(fitting: SurfacePoints, surface: SplineSurface, boundary_categories):
    from src.nurbs.SplineSurface import SplineSurface
    from src.nurbs.SurfacePoints import SurfacePoints

    boundaries_parameterize_data, boundaries_diff = np.zeros((fitting.boundary.shape[0] - 4, 2)), np.zeros(
        (fitting.boundary.shape[0] - 4, 2)
    )
    corners_parameterize_data, corners_diff = np.zeros((4, 2)), np.zeros((4, 2))
    current_index = 0
    ops = {
        0: lambda d: np.hstack((d, np.zeros_like(d))),
        1: lambda d: np.hstack((np.ones_like(d), d)),
        2: lambda d: np.hstack((d, np.ones_like(d))),
        3: lambda d: np.hstack((np.zeros_like(d), d)),
    }
    corner_ops = {
        0: np.array([[0.0, 0.0]]),
        1: np.array([[1.0, 0.0]]),
        2: np.array([[1.0, 1.0]]),
        3: np.array([[0.0, 1.0]]),
    }
    for i, points in enumerate(boundary_categories):
        boundary_parameterize_data, _ = surface.boundary[i].inversion(points[1:-1])
        boundary_parameterize_data = ops[i](boundary_parameterize_data)
        boundary_diff = np.array([surface.single_point(*para) for para in boundary_parameterize_data]) - points[1:-1]
        boundaries_parameterize_data[current_index : current_index + points.shape[0] - 2] = boundary_parameterize_data
        boundaries_diff[current_index : current_index + points.shape[0] - 2] = boundary_diff
        current_index += points.shape[0] - 2

        corner_parameterize_data = corner_ops[i]
        corners_parameterize_data[i] = corner_parameterize_data
        corners_diff[i] = np.array([surface.single_point(*para) for para in corner_parameterize_data]) - points[0]
    return (
        boundaries_parameterize_data,
        boundaries_diff.reshape(-1),
        corners_parameterize_data,
        corners_diff.reshape(-1),
    )


def bspline_surface_fitting_newton_raphson_four_points2d(
    fitting: SurfacePoints, surface: SplineSurface, boundary_categories, tol=2e-2, rho=0.5, iteration=100, verbose=False
):
    from src.nurbs.SplineSurface import SplineSurface
    from src.nurbs.element.NurbsSurfaceElement import NurbsSurfaceElement
    from src.nurbs.SurfacePoints import SurfacePoints

    energy = MIPS()
    surface.activate_boundary()
    surface.update_boundary()
    element = NurbsSurfaceElement(surface, energy=energy)

    boundaries_parameterize_data, boundaries_diff, corners_parameterize_data, corners_diff = point_parameterize(
        fitting, surface, boundary_categories
    )
    deformation_gradient = element.compute_deformation_gradient(element.vertice)
    energy = element.surface_energy(deformation_gradient)
    err = (
        fitting.boundary_weight * np.sum(np.sqrt(boundaries_diff * boundaries_diff))
        + fitting.corner_weight * np.sum(np.sqrt(corners_diff * corners_diff))
        + fitting.penalty * energy
    )
    fix_dofs = [
        0,
        1,
        surface.num_ctrlpts_u * 2 - 2,
        surface.num_ctrlpts_u * 2 - 1,
        (surface.num_ctrlpts_u * (surface.num_ctrlpts_v - 1)) * 2,
        (surface.num_ctrlpts_u * (surface.num_ctrlpts_v - 1)) * 2 + 1,
        surface.num_ctrlpts * 2 - 2,
        surface.num_ctrlpts * 2 - 1,
    ]

    iter = 0
    while iter < iteration and err > tol:
        hess = surface_fitting_least_square_hessian(
            fitting.boundary_weight, boundaries_diff, boundaries_parameterize_data, surface
        ) + surface_fitting_least_square_hessian(
            fitting.corner_weight, corners_diff, corners_parameterize_data, surface
        )
        grad = surface_fitting_least_square_jacobian(
            fitting.boundary_weight, boundaries_diff, boundaries_parameterize_data, surface
        ) + surface_fitting_least_square_jacobian(
            fitting.corner_weight, corners_diff, corners_parameterize_data, surface
        )

        deformation_gradient = element.compute_deformation_gradient(element.vertice)
        ddeformation_dx = element.compute_ddeformation_dx()
        constraint_grad = fitting.penalty * element.surface_energy_gradient(deformation_gradient, ddeformation_dx)
        constriant_hess = fitting.penalty * element.surface_energy_hessian(deformation_gradient, ddeformation_dx)
        hess = hess + constriant_hess
        grad = grad + constraint_grad
        hess[fix_dofs, :] = 0.0
        hess[:, fix_dofs] = 0.0
        hess[fix_dofs, fix_dofs] = 1.0
        grad[fix_dofs] = 0.0
        delta = checked_scipy_krylov(cg, hess, -grad, solver_name="NURBS fitting CG")
        delta = delta.reshape(-1, surface.dimension)

        deformation_gradient = element.compute_deformation_gradient(element.vertice)
        alpha = element.compute_step_size(delta, deformation_gradient)
        if alpha < 1.0:
            alpha = alpha * rho

        init_ctrlpts = element.vertice.copy()
        try:
            while alpha > tol:
                disp = alpha * delta
                curr_ctrlpts = init_ctrlpts + disp
                try:
                    element.validate(curr_ctrlpts)
                except ValueError:
                    alpha *= 0.5
                    continue
                element.vertice = curr_ctrlpts.copy()
                surface.control_points = curr_ctrlpts.copy()
                surface.update_boundary()

                deformation_gradient = element.compute_deformation_gradient(element.vertice)
                energy = element.surface_energy(deformation_gradient)
                boundaries_parameterize_data, boundaries_diff, corners_parameterize_data, corners_diff = (
                    point_parameterize(fitting, surface, boundary_categories)
                )

                err_update = (
                    fitting.boundary_weight * np.sum(np.sqrt(boundaries_diff * boundaries_diff))
                    + fitting.corner_weight * np.sum(np.sqrt(corners_diff * corners_diff))
                    + fitting.penalty * energy
                )
                if verbose:
                    print(f"New error is {err_update} and old error is {err}")
                if err_update <= err:
                    break
                alpha *= 0.5
            else:
                raise NonlinearSolveError("Surface fitting found no valid descent step")
        except Exception:
            element.vertice = init_ctrlpts.copy()
            surface.control_points = init_ctrlpts.copy()
            surface.update_boundary()
            raise

        err, err_update = err_update, err
        updating = np.sum(np.sqrt(delta * delta))
        iter += 1

        if verbose:
            print(f"In iteration: {iter}")
            print(f"Step size: {alpha}")
            print(f"Residual error: {err}")
            print(f"Residual displacement: {updating}", "\n")

        if updating < tol:
            break

        if alpha <= tol or abs(energy) < tol:
            break
    validate_spline_jacobian(surface)
    return surface


def nurbs_volume(
    degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, control_point, weight=None, closed=False
):
    from src.nurbs.NurbsVolume import NurbsVolume

    if weight is None:
        weight = np.ones(control_point.shape[0])
    volume = NurbsVolume()
    volume.degree_u = degree_u
    volume.degree_v = degree_v
    volume.degree_w = degree_w
    volume.knot_vector_u = knot_vector_u
    volume.knot_vector_v = knot_vector_v
    volume.knot_vector_w = knot_vector_w
    volume.control_points = control_point
    volume.weights = weight
    volume.closed = closed
    volume.activate_boundary()
    return volume


def validate_spline_jacobian(primitive):
    """Return the minimum sampled signed IGA Jacobian for a planar patch or solid."""
    from itertools import product
    from src.nurbs.element.NurbsSurfaceElement import NurbsSurfaceElement

    if len(primitive.degree) == 2:
        element = NurbsSurfaceElement(primitive)
        return float(np.min(element.jacobian_determinants(primitive.control_points)))
    if len(primitive.degree) != 3 or primitive.dimension != 3:
        raise ValueError("Signed Jacobians require a planar 2D patch or a 3D volume")
    points = primitive.control_points
    weights = getattr(primitive, "weights", None)
    if not np.isfinite(points).all() or (
        weights is not None and (not np.isfinite(weights).all() or np.any(weights <= 0.0))
    ):
        raise ValueError("Control points must be finite and NURBS weights finite and positive")
    minimum = np.inf
    samples = [
        np.unique(np.r_[np.linspace(0.0, 1.0, 2 * p + 1), 0.5 * (np.polynomial.legendre.leggauss(p + 1)[0] + 1.0)])
        for p in primitive.degree
    ]
    for element_id, spans in enumerate(product(primitive.element_u, primitive.element_v, primitive.element_w)):
        lower = np.asarray(spans)[:, 0]
        upper = np.asarray(spans)[:, 1]
        for parameter in product(*samples):
            knots = lower + np.asarray(parameter) * (upper - lower)
            knots = np.clip(knots, np.nextafter(lower, upper), np.nextafter(upper, lower))
            determinant = np.linalg.det(np.column_stack(primitive.derivative(*knots)))
            if not np.isfinite(determinant) or determinant <= 0.0:
                raise ValueError(
                    f"Inverted or degenerate IGA Jacobian in element {element_id} at {knots}: {determinant}"
                )
            minimum = min(minimum, determinant)
    return float(minimum)


def extrude_surface(surface, vector, degree=1):
    """Create an exact straight NURBS volume, preserving the source patch at w=0."""
    from src.nurbs.SplineSurface import SplineSurface

    if not isinstance(surface, SplineSurface) or surface.dimension not in (2, 3):
        raise ValueError("Extrusion requires a 2D or 3D spline surface")
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0.0:
        raise ValueError("Extrusion vector must contain three finite components and be nonzero")
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("Extrusion degree must be a positive integer")
    degree = int(degree)
    points = np.pad(surface.control_points, ((0, 0), (0, 3 - surface.dimension)))
    layers = np.linspace(0.0, 1.0, degree + 1)
    control_points = (points[None, :, :] + layers[:, None, None] * vector).reshape(-1, 3)
    weights = getattr(surface, "weights", None)
    if weights is None:
        weights = np.ones(surface.num_ctrlpts)
    volume = nurbs_volume(
        surface.degree_u,
        surface.degree_v,
        degree,
        surface.knot_vector_u.copy(),
        surface.knot_vector_v.copy(),
        generate_uniform(degree, degree + 1),
        control_points,
        weight=np.tile(weights, degree + 1),
    )
    validate_spline_jacobian(volume)
    return volume


def distance_lower_bound(primitive1: Spline, primitive2: Spline, sA_intv, sB_intv):
    trimmed_ctrlpts1 = primitive1.convex_hull(*sA_intv)
    trimmed_ctrlpts2 = primitive2.convex_hull(*sB_intv)

    bounding_box1 = update_patch_aabb(trimmed_ctrlpts1)
    bounding_box2 = update_patch_aabb(trimmed_ctrlpts2)
    return distance_aabb(bounding_box1, bounding_box2)
