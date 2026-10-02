import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr, cg
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib.pyplot as plt

from src.linear_solver.ScipyKrylov import checked_scipy_krylov
from src.nurbs import *
from src.nurbs.Energy import MIPS, ASAP
from src.nurbs.Kmean import kmean
from src.nurbs.SplinePrimitives import Spline, SplineCurve, SplineSurface
from src.nurbs.element import LinearQuadrilateralElement, LinearTriangleElement
from src.nurbs.Nurbs import find_span_linear, basis_functions, one_basis_function
from src.nurbs.Utilities import (
    check_same_side,
    evaluate_bounding_box,
    solve_using_cod,
    get_ctrlpt_dofs,
    insert_midpoints,
    sort_points_counterclockwise,
)
from src.nurbs.Operations import *


class CurvePoints(object):
    def __init__(self):
        self._points = None
        self._weights = 1.0
        self._smooth = 1e-6
        self._penalty = 1e-6

    def setup(self, points, weights=1.0, penalty=1e-6, smoothness=1e-6):
        self.points = points
        self.weights = weights
        self.smoothness = smoothness
        self.penalty = penalty

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = np.asarray(points)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def smoothness(self):
        return self._smoothness

    @smoothness.setter
    def smoothness(self, smoothness):
        self._smoothness = smoothness

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, penalty):
        self._penalty = penalty


class SurfacePoints(object):
    def __init__(self):
        self._interior = None
        self._boundary = None
        self._common_index = None
        self._common_boundary = None
        self._rest_shape = None
        self._boundary_weight = 1.0
        self._interior_weight = 1.0
        self._corner_weight = 10.0
        self._boundary_smooth = 1e-6
        self._interior_smooth = 1e-6
        self._regularisation_res_u = 0
        self._regularisation_res_v = 0
        self._regularisation_u = 0.0
        self._regularisation_v = 0.0
        self._penalty = 1e-6

    def setup(
        self,
        interior=np.array([]),
        boundary=np.array([]),
        common_index=np.array([]),
        common_boundary=np.array([]),
        rest_shape=np.array([]),
        interior_weight=1.0,
        boundary_weight=1.0,
        corner_weight=10.0,
        boundary_smoothness=1e-6,
        interior_smoothness=1e-6,
        regularisation_res_u=0,
        regularisation_res_v=0,
        regularisation_u=0.0,
        regularisation_v=0.0,
        penalty=1e-6,
    ):
        self.interior = interior
        self.boundary = boundary
        self.common_index = common_index
        self.common_boundary = common_boundary
        self.rest_shape = rest_shape
        self.boundary_weight = boundary_weight
        self.interior_weight = interior_weight
        self.corner_weight = corner_weight
        self.boundary_smoothness = boundary_smoothness
        self.interior_smoothness = interior_smoothness
        self.regularisation_res_u = regularisation_res_u
        self.regularisation_res_v = regularisation_res_v
        self.regularisation_u = regularisation_u
        self.regularisation_v = regularisation_v
        self.penalty = penalty

    @property
    def interior(self):
        return self._interior

    @interior.setter
    def interior(self, interior):
        self._interior = np.asarray(interior)

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        self._boundary = np.asarray(boundary)

    @property
    def common_index(self):
        return self._common_index

    @common_index.setter
    def common_index(self, common_index):
        self._common_index = np.asarray(common_index)

    @property
    def common_boundary(self):
        return self._common_boundary

    @common_boundary.setter
    def common_boundary(self, common_boundary):
        if self.common_index is None:
            raise RuntimeError("Set common_index first")
        self._common_boundary = np.asarray(common_boundary)
        if len(self._common_index) != len(self._common_boundary):
            raise ValueError(
                f"The dimension of common_index {len(self._common_index)} must be equal to common_boundary {len(self._common_boundary)}"
            )

    @property
    def rest_shape(self):
        return self._rest_shape

    @rest_shape.setter
    def rest_shape(self, rest_shape):
        self._rest_shape = rest_shape

    @property
    def boundary_weight(self):
        return self._boundary_weight

    @boundary_weight.setter
    def boundary_weight(self, boundary_weight):
        self._boundary_weight = boundary_weight

    @property
    def interior_weight(self):
        return self._interior_weight

    @interior_weight.setter
    def interior_weight(self, interior_weight):
        self._interior_weight = interior_weight

    @property
    def corner_weight(self):
        return self._corner_weight

    @corner_weight.setter
    def corner_weight(self, corner_weight):
        self._corner_weight = corner_weight

    @property
    def boundary_smoothness(self):
        return self._boundary_smoothness

    @boundary_smoothness.setter
    def boundary_smoothness(self, boundary_smoothness):
        self._boundary_smoothness = boundary_smoothness

    @property
    def interior_smoothness(self):
        return self._interior_smoothness

    @interior_smoothness.setter
    def interior_smoothness(self, interior_smoothness):
        self._interior_smoothness = interior_smoothness

    @property
    def regularisation_res_u(self):
        return self._regularisation_res_u

    @regularisation_res_u.setter
    def regularisation_res_u(self, regularisation_res_u):
        self._regularisation_res_u = regularisation_res_u

    @property
    def regularisation_res_v(self):
        return self._regularisation_res_v

    @regularisation_res_v.setter
    def regularisation_res_v(self, regularisation_res_v):
        self._regularisation_res_v = regularisation_res_v

    @property
    def regularisation_u(self):
        return self._regularisation_u

    @regularisation_u.setter
    def regularisation_u(self, regularisation_u):
        self._regularisation_u = regularisation_u

    @property
    def regularisation_v(self):
        return self._regularisation_v

    @regularisation_v.setter
    def regularisation_v(self, regularisation_v):
        self._regularisation_v = regularisation_v

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, penalty):
        self._penalty = penalty


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
        return np.unique(alpha_shape_2D(points, alpha))
    elif points.shape[1] == 3:
        return np.unique(alpha_shape_3D(points, alpha))


def alpha_shape_2D(points, alpha=0.1):
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
    detM = np.linalg.det(M)
    if abs(detM) < 1e-10:
        return np.inf
    rhs = 0.5 * np.array([np.dot(AB, AB), np.dot(AC, AC), np.dot(AD, AD)])
    center = np.linalg.solve(M.T, rhs)
    return np.linalg.norm(center)


def alpha_shape_3D(points, alpha=0.1):
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


def PCA_analysis(points):
    points = np.asarray(points)
    points_center = np.mean(points, axis=0)
    centered_points = points - points_center
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues /= points.shape[0]
    return points_center, eigenvectors, eigenvalues


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


def get_boundary_categories(surface: SplineSurface, boundary_points, corner_index=[], visualize=False):
    if len(corner_index) == 0:
        pass
    if len(corner_index) != 4:
        raise ValueError("The dimension of corner_index should be 4")
    corner_points = boundary_points[corner_index]
    boundary_categories = split_boundary_points(surface, boundary_points, corner_points, visualize=visualize)
    fittings = [CurvePoints(), CurvePoints(), CurvePoints(), CurvePoints()]
    fittings[0].setup(boundary_categories["Category 1 (A->B)"])
    fittings[1].setup(boundary_categories["Category 2 (B->C)"])
    fittings[2].setup(boundary_categories["Category 3 (C->D)"])
    fittings[3].setup(boundary_categories["Category 4 (D->A)"])
    return fittings


def get_margin_curves(surface: SplineSurface, fittings):
    margin_curves = [
        margin_fitting(surface, 0, fittings[0].points),
        margin_fitting(surface, 1, fittings[1].points),
        margin_fitting(surface, 2, fittings[2].points),
        margin_fitting(surface, 3, fittings[3].points),
    ]
    return margin_curves


def margin_fitting(surface: SplineSurface, index, margin_points):
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


def resample_boundary3D(surface: SplineSurface, points, alpha=0.5):
    points_center, eigenvectors, eigenvalues = PCA_analysis(points)
    projected_points = np.dot(points - points_center, eigenvectors[:, :2])
    boundary_index = find_boundary_index(projected_points, alpha=alpha)
    boundary_points = projected_points[boundary_index]
    return resample_boundary(surface, boundary_points)


def resample_boundary2D(surface: SplineSurface, points, alpha=0.5):
    boundary_index = find_boundary_index(points, alpha=alpha)
    boundary_points = points[boundary_index]
    return resample_boundary(surface, boundary_points)


def resample_boundary(surface: SplineSurface, boundary_points, alpha=0.5):
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


def compute_surface_control_points_2D(points, resolution, bounding_quadrilateral_method=0):
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


def compute_surface_control_points_3D(points, resolution, bounding_quadrilateral_method=0):
    if isinstance(resolution, (int, float)):
        num_ctrlpts = [int(resolution) - 1, int(resolution) - 1]
    elif isinstance(resolution, (np.ndarray, list, tuple)):
        num_ctrlpts = np.asarray(resolution) - 1

    get_bounding = axis_aligned_bounding_quadrilateral
    if bounding_quadrilateral_method == 1:
        get_bounding = optimize_bounding_quadrilateral

    points = np.asarray(points)
    points_center, eigenvectors, eigenvalues = PCA_analysis(points)
    projected_points = np.dot(points - points_center, eigenvectors[:, :2])
    quad2d = get_bounding(projected_points)
    quad3d = convert_quad2d_to_quad3d(quad2d, eigenvectors) + points_center
    current_points, quads = subdivide_quadrilateral(quad3d, num_ctrlpts)
    return current_points


def compute_fitting_error(primitive: Spline, points, closed=None):
    residuals = primitive.distance(points, closed)
    return np.sum(np.sqrt(residuals * residuals))


def global_interpolate_bspline_curve(points, degree, derivatives=None, use_centripetal=False):
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
    return nurbs_curve(degree, knot_vector, ctrlpts)


def global_interpolate_bspline_surface(
    num_datapt_u, num_datapt_v, points, degree_u, degree_v, derivatives=None, use_centripetal=False
):
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
    return nurbs_surface(degree_u, degree_v, kv_u, kv_v, ctrlpts)


def global_approximate_bspline_curve(
    points, degree, knot_vector=None, control_point_num=None, fixed=None, use_centripetal=False
):
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
    return nurbs_curve(degree, knot_vector, control_points)


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
    return nurbs_surface(degree_u, degree_v, knot_vector_u, knot_vector_v, ctrlpts_v)


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
    ncp = curve.num_ctrlpts - 2 * (curve.degree - 1) if curve.closed else curve.num_ctrlpts
    cageReg = curve.num_ctrlpts - 2 * (curve.degree - 1) if curve.closed else curve.num_ctrlpts - 2
    coefficient = np.zeros((curve.dimension * cageReg, curve.dimension * ncp))
    samples = np.zeros(curve.dimension * cageReg)
    row = 0
    for j in range(1, cageReg + 1):
        for dim in range(curve.dimension):
            coefficient[row, curve.dimension * (j % ncp) + dim] += -2.0 * smoothness
            coefficient[row, curve.dimension * ((j - 1) % ncp) + dim] += 1.0 * smoothness
            coefficient[row, curve.dimension * ((j + 1) % ncp) + dim] += 1.0 * smoothness
            row += 1
    return coefficient, samples


def curve_pdm_assemble(points, curve: SplineCurve, parameterize_data=None):
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
    pass


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
        jacobianP, _ = assemble(fitting.points, curve, parameterize_data)
        jacobianReg, _ = curve_regularisation_assemble(curve, fitting.smoothness)

        hess = (
            fitting.weights * fitting.weights * jacobianP.T @ jacobianP
        )  # + fitting.penalty * fitting.penalty * jacobianReg.T @ jacobianReg
        grad = fitting.weights * jacobianP.T @ dist_diff  # + fitting.penalty * jacobianReg.T @ reg_diff

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
            control_points = compute_surface_control_points_2D(
                points,
                resolution=(num_ctrlpt_u, num_ctrlpt_v),
                bounding_quadrilateral_method=bounding_quadrilateral_method,
            )
        surf = bspline_surface(degree_u, degree_v, knot_vector_u, knot_vector_v, control_points)
        return surf
    elif points.shape[1] == 3:
        if control_points is None:
            control_points = compute_surface_control_points_3D(
                points,
                resolution=(num_ctrlpt_u, num_ctrlpt_v),
                bounding_quadrilateral_method=bounding_quadrilateral_method,
            )
        return bspline_surface(degree_u, degree_v, knot_vector_u, knot_vector_v, control_points)


def surface_pdm_assemble(weights, points, parameterize_data, surface: SplineSurface, is_boundary=None):
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


def surface_partialU_assemble(points, parameterize_data, surface: SplineSurface, is_boundary=False):
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
    cageReg = (surface.num_ctrlpts_u - 2) * (surface.num_ctrlpts_v - 2)
    if cageReg > 0:
        coefficient = np.zeros(
            (surface.dimension * cageReg, surface.dimension * surface.num_ctrlpts_u * surface.num_ctrlpts_v)
        )
        sample = np.zeros(surface.dimension * cageReg)
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
    cageRegBound = 2 * (surface.num_ctrlpts_u - 1) + 2 * (surface.num_ctrlpts_v - 1)
    if cageRegBound > 0:
        coefficient = np.zeros(
            (surface.dimension * cageRegBound, surface.dimension * surface.num_ctrlpts_u * surface.num_ctrlpts_v)
        )
        sample = np.zeros(surface.dimension * cageRegBound)
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
                raise RuntimeError(
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
    num_interior = num_boundary = num_ctrlpts = num_commonBound = cageReg = cageRegBound = 0
    for fitting, surface in zip(fittings, surfaces):
        row_prefix.append(surface.dimension * (num_interior + num_boundary + cageReg + cageRegBound + num_commonBound))
        column_prefix.append(surface.dimension * num_ctrlpts)
        num_interior += fitting.interior.shape[0]
        num_boundary += fitting.boundary.shape[0]
        num_ctrlpts += surface.num_ctrlpts_u * surface.num_ctrlpts_v
        if len(surfaces) > 1:
            num_commonBound += fitting.common_boundary.shape[0]
        else:
            fitting.common_index = np.array([])
            fitting.common_boundary = np.array([])
        cageReg += (surface.num_ctrlpts_u - 2) * (surface.num_ctrlpts_v - 2)
        cageRegBound += 2 * (surface.num_ctrlpts_u - 1) + 2 * (surface.num_ctrlpts_v - 1)
    row_prefix.append(surface.dimension * (num_interior + num_boundary + cageReg + cageRegBound + num_commonBound))
    column_prefix.append(surface.dimension * num_ctrlpts)

    assemble = choose_surface_assemble_method(method)
    coeffs = lil_matrix((row_prefix[-1], column_prefix[-1]))
    samples = []
    for objectID, (fitting, surface) in enumerate(zip(fittings, surfaces)):
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
            objectID, column_prefix, fitting.common_index, fitting.common_boundary, surfaces
        )

        coeffs[
            row_prefix[objectID] : row_prefix[objectID + 1] - surface.dimension * fitting.common_boundary.shape[0],
            column_prefix[objectID] : column_prefix[objectID + 1],
        ] += np.concatenate((coeff_boundary, coeff_interior, coeff_reg, coeff_breg))
        if fitting.common_boundary.shape[0] > 0:
            coeffs[
                row_prefix[objectID + 1]
                - surface.dimension * fitting.common_boundary.shape[0] : row_prefix[objectID + 1],
                :,
            ] += coeff_common
        samples.append(np.concatenate((sample_boundary, sample_interior, sample_reg, sample_breg, sample_common)))
    result = lsqr(coeffs.tocsr(), np.concatenate(([sample for sample in samples])))[0]
    for surfID, surface in enumerate(surfaces):
        surface.control_points = result[column_prefix[surfID] : column_prefix[surfID + 1]].reshape(
            -1, surface.dimension
        )
    return surfaces


def bspline_surface_fitting_gauss_newton2D(
    fitting: SurfacePoints, surface: SplineSurface, tol=2e-2, rho=0.5, iteration=100, method="pdm", verbose=False
):
    energy = MIPS()
    xcoord, ycoord = np.meshgrid(
        np.linspace(1 - surface.num_ctrlpts_u, surface.num_ctrlpts_u - 1, surface.num_ctrlpts_u),
        np.linspace(1 - surface.num_ctrlpts_v, surface.num_ctrlpts_v - 1, surface.num_ctrlpts_v),
        indexing="xy",
    )
    reference_points = np.column_stack((xcoord.ravel(), ycoord.ravel()))
    element = LinearQuadrilateralElement(
        surface.control_points, reference_points, surface.connectivity, energy=energy, gauss_point_number=2
    )

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
        coeff_boundary = np.concatenate([dp_boundary])
        hess = lil_matrix(coeff_boundary.T @ coeff_boundary)
        grad = coeff_boundary.T @ boundary_diff

        deformation_gradient = element.compute_deformation_gradient(element.vertice)
        DdeformationDx = element.compute_ddeformation_dx()
        constraint_grad = fitting.penalty * element.surface_energy_gradient(
            deformation_gradient, DdeformationDx
        ).reshape(-1, 1)
        constriant_hess = fitting.penalty * element.surface_energy_hessian(deformation_gradient, DdeformationDx)
        hess = hess + constriant_hess
        grad = grad + constraint_grad
        delta = checked_scipy_krylov(cg, hess, -grad, solver_name="NURBS fitting CG")
        delta = delta.reshape(-1, surface.dimension)

        deformation_gradient = element.compute_deformation_gradient(element.vertice)
        alpha = element.compute_step_size(delta, deformation_gradient)
        if alpha < 1.0:
            alpha = alpha * rho

        init_ctrlpts = element.vertice.copy()
        while alpha > tol:
            disp = alpha * delta
            curr_ctrlpts = init_ctrlpts + disp
            element.vertice = curr_ctrlpts.copy()
            surface.control_points = curr_ctrlpts.copy()

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
            raise RuntimeError(f"Convergence failed, the residual error is {err}")
    return surface


def surface_fitting_least_square_jacobian(weights, boundary_diff, parameterize_data, surface: SplineSurface):
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


def bspline_surface_fitting_newton_raphson2D(
    fitting: SurfacePoints, surface: SplineSurface, tol=2e-2, rho=0.5, iteration=100, verbose=False
):
    energy = MIPS()
    xcoord, ycoord = np.meshgrid(
        np.linspace(1 - surface.num_ctrlpts_u, surface.num_ctrlpts_u - 1, surface.num_ctrlpts_u),
        np.linspace(1 - surface.num_ctrlpts_v, surface.num_ctrlpts_v - 1, surface.num_ctrlpts_v),
        indexing="xy",
    )
    reference_points = np.column_stack((xcoord.ravel(), ycoord.ravel()))
    element = LinearQuadrilateralElement(
        surface.control_points, reference_points, surface.connectivity, energy=energy, gauss_point_number=2
    )

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
        DdeformationDx = element.compute_ddeformation_dx()
        constraint_grad = fitting.penalty * element.surface_energy_gradient(deformation_gradient, DdeformationDx)
        constriant_hess = fitting.penalty * element.surface_energy_hessian(deformation_gradient, DdeformationDx)
        hess = hess + constriant_hess
        grad = grad + constraint_grad
        delta = checked_scipy_krylov(cg, hess, -grad, solver_name="NURBS fitting CG")
        delta = delta.reshape(-1, surface.dimension)

        deformation_gradient = element.compute_deformation_gradient(element.vertice)
        alpha = element.compute_step_size(delta, deformation_gradient)
        if alpha < 1.0:
            alpha = alpha * rho

        init_ctrlpts = element.vertice.copy()
        while alpha > tol:
            disp = alpha * delta
            curr_ctrlpts = init_ctrlpts + disp
            element.vertice = curr_ctrlpts.copy()
            surface.control_points = curr_ctrlpts.copy()

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
            raise RuntimeError(f"Convergence failed, the residual error is {err}")
    return surface


def reorder_by_main_axis(surface: SplineSurface, category_1, category_2, category_3, category_4):
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
    ref_indices = []
    boundary_array = np.array(boundary_points)

    for ref in corner_points:
        distances = np.linalg.norm(boundary_array - np.array(ref), axis=1)
        ref_indices.append(np.argmin(distances))
    ref_indices.sort()

    A_idx, B_idx, C_idx, D_idx = ref_indices
    category_1 = boundary_points[A_idx : B_idx + 1]
    category_2 = boundary_points[B_idx : C_idx + 1]
    category_3 = boundary_points[C_idx : D_idx + 1]
    category_4 = np.concatenate((boundary_points[D_idx:], boundary_points[: A_idx + 1]))
    # category_1, category_2, category_3, category_4 = reorder_by_main_axis(surface, category_1, category_2, category_3, category_4)
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


def bspline_surface_fitting_newton_raphson_four_points2D(
    fitting: SurfacePoints, surface: SplineSurface, boundary_categories, tol=2e-2, rho=0.5, iteration=100, verbose=False
):
    energy = MIPS()
    xcoord, ycoord = np.meshgrid(
        np.linspace(1 - surface.num_ctrlpts_u, surface.num_ctrlpts_u - 1, surface.num_ctrlpts_u),
        np.linspace(1 - surface.num_ctrlpts_v, surface.num_ctrlpts_v - 1, surface.num_ctrlpts_v),
        indexing="xy",
    )
    reference_points = np.column_stack((xcoord.ravel(), ycoord.ravel()))
    element = LinearQuadrilateralElement(
        surface.control_points, reference_points, surface.connectivity, energy=energy, gauss_point_number=2
    )

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
        DdeformationDx = element.compute_ddeformation_dx()
        constraint_grad = fitting.penalty * element.surface_energy_gradient(deformation_gradient, DdeformationDx)
        constriant_hess = fitting.penalty * element.surface_energy_hessian(deformation_gradient, DdeformationDx)
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
        while alpha > tol:
            disp = alpha * delta
            curr_ctrlpts = init_ctrlpts + disp
            element.vertice = curr_ctrlpts.copy()
            surface.control_points = curr_ctrlpts.copy()

            deformation_gradient = element.compute_deformation_gradient(element.vertice)
            energy = element.surface_energy(deformation_gradient)
            boundaries_parameterize_data, boundaries_diff, corners_parameterize_data, corners_diff = point_parameterize(
                fitting, surface, boundary_categories
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
    return surface
