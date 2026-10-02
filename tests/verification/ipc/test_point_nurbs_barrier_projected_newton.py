"""Verification of projected-Newton point/NURBS barrier metrics."""

import numpy as np
import taichi as ti

from src.physics_model.contact_model.ipc.NurbsContact import (
    curve_barrier_projected_metric,
    surface_barrier_projected_metric,
)


def _project_psd(matrix):
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Taichi/LLVM can leave floating-point status flags set after a kernel;
    # NumPy 2.x may report those stale flags on the next BLAS operation even
    # though every operand/result is finite.  Validate finiteness explicitly
    # and suppress only that status-flag reporting around the multiplication.
    assert np.isfinite(eigenvalues).all()
    assert np.isfinite(eigenvectors).all()
    with np.errstate(all="ignore"):
        projected = (
            eigenvectors * np.maximum(eigenvalues, 0.0)
        ) @ eigenvectors.T
    assert np.isfinite(projected).all()
    return projected


def _curve_reduced_system(
    pointer,
    tangent,
    curvature,
    shapes,
    derivatives,
    free,
    distance,
    barrier_gradient,
    barrier_hessian,
    measure,
):
    dimension = pointer.size
    reduced_dimension = dimension + 1
    jacobians = []
    for shape, derivative in zip(shapes, derivatives):
        jacobian = np.zeros((reduced_dimension, dimension))
        jacobian[:dimension] = shape * np.eye(dimension)
        if free:
            jacobian[dimension] = shape * tangent + derivative * pointer
        jacobians.append(jacobian)
    point_jacobian = np.zeros((reduced_dimension, dimension))
    point_jacobian[:dimension] = -np.eye(dimension)
    if free:
        point_jacobian[dimension] = -tangent
    jacobians.append(point_jacobian)

    alpha = 0.5 * barrier_gradient / distance
    beta = 0.25 * (
        barrier_hessian / distance**2
        - barrier_gradient / distance**3
    )
    reduced_hessian = np.zeros((reduced_dimension, reduced_dimension))
    reduced_hessian[:dimension, :dimension] = measure * (
        2.0 * alpha * np.eye(dimension)
        + 4.0 * beta * np.outer(pointer, pointer)
    )
    if free:
        coefficient = np.dot(tangent, tangent) + np.dot(pointer, curvature)
        reduced_hessian[dimension, dimension] = (
            -2.0 * alpha * measure / coefficient
        )
    return jacobians, reduced_hessian


def _surface_reduced_system(
    pointer,
    tangent_u,
    tangent_v,
    curvature_uu,
    curvature_vv,
    curvature_uv,
    shapes,
    derivative_u,
    derivative_v,
    free_u,
    free_v,
    distance,
    barrier_gradient,
    barrier_hessian,
    measure,
):
    dimension = pointer.size
    reduced_dimension = dimension + 2
    jacobians = []
    for shape, du, dv in zip(shapes, derivative_u, derivative_v):
        jacobian = np.zeros((reduced_dimension, dimension))
        jacobian[:dimension] = shape * np.eye(dimension)
        if free_u:
            jacobian[dimension] = shape * tangent_u + du * pointer
        if free_v:
            jacobian[dimension + 1] = shape * tangent_v + dv * pointer
        jacobians.append(jacobian)
    point_jacobian = np.zeros((reduced_dimension, dimension))
    point_jacobian[:dimension] = -np.eye(dimension)
    if free_u:
        point_jacobian[dimension] = -tangent_u
    if free_v:
        point_jacobian[dimension + 1] = -tangent_v
    jacobians.append(point_jacobian)

    alpha = 0.5 * barrier_gradient / distance
    beta = 0.25 * (
        barrier_hessian / distance**2
        - barrier_gradient / distance**3
    )
    reduced_hessian = np.zeros((reduced_dimension, reduced_dimension))
    reduced_hessian[:dimension, :dimension] = measure * (
        2.0 * alpha * np.eye(dimension)
        + 4.0 * beta * np.outer(pointer, pointer)
    )
    coefficient = np.asarray(
        [
            [
                np.dot(tangent_u, tangent_u)
                + np.dot(pointer, curvature_uu),
                np.dot(tangent_u, tangent_v)
                + np.dot(pointer, curvature_uv),
            ],
            [
                np.dot(tangent_v, tangent_u)
                + np.dot(pointer, curvature_uv),
                np.dot(tangent_v, tangent_v)
                + np.dot(pointer, curvature_vv),
            ],
        ]
    )
    active = np.flatnonzero([free_u, free_v])
    if active.size:
        inverse = np.linalg.inv(coefficient[np.ix_(active, active)])
        parameter_block = np.zeros((2, 2))
        parameter_block[np.ix_(active, active)] = inverse
        reduced_hessian[dimension:, dimension:] = (
            -2.0 * alpha * measure * parameter_block
        )
    return jacobians, reduced_hessian


def _assert_reduced_metric_is_full_euclidean_projection(
    jacobians, reduced_hessian, metric
):
    reduced_jacobian = np.concatenate(jacobians, axis=1)
    exact = reduced_jacobian.T @ reduced_hessian @ reduced_jacobian
    expected = _project_psd(exact)
    actual = reduced_jacobian.T @ metric @ reduced_jacobian
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2.0e-8, atol=2.0e-8)
    np.testing.assert_allclose(actual, actual.T, atol=2.0e-11)
    assert np.linalg.eigvalsh(actual).min() >= -2.0e-8

    # The particle point is projected once in contact coordinates, then
    # pulled back to its MPM nodal stencil.  This must preserve the same
    # projected contact matrix and PSD; it must not re-project CC/CM/MM blocks.
    dimension = jacobians[0].shape[1]
    control_count = len(jacobians) - 1
    mpm_weights = np.asarray([0.22, 0.31, 0.47])
    contact_dofs = (control_count + 1) * dimension
    nodal_dofs = (control_count + mpm_weights.size) * dimension
    pullback = np.zeros((contact_dofs, nodal_dofs))
    pullback[: control_count * dimension, : control_count * dimension] = (
        np.eye(control_count * dimension)
    )
    point_row = control_count * dimension
    point_column = control_count * dimension
    for node, weight in enumerate(mpm_weights):
        pullback[
            point_row : point_row + dimension,
            point_column + node * dimension : point_column + (node + 1) * dimension,
        ] = weight * np.eye(dimension)
    with np.errstate(all="ignore"):
        pulled_actual = pullback.T @ actual @ pullback
        pulled_expected = pullback.T @ expected @ pullback
    assert np.isfinite(pulled_actual).all()
    assert np.isfinite(pulled_expected).all()
    np.testing.assert_allclose(
        pulled_actual, pulled_expected, rtol=2.0e-8, atol=2.0e-8
    )
    assert np.linalg.eigvalsh(0.5 * (pulled_actual + pulled_actual.T)).min() >= -2.0e-8


@ti.data_oriented
class _CurveProjectionHarness:
    def __init__(self):
        self.pointer = ti.Vector.field(2, ti.f64, shape=())
        self.tangent = ti.Vector.field(2, ti.f64, shape=())
        self.curvature = ti.Vector.field(2, ti.f64, shape=())
        self.shapes = ti.Vector.field(3, ti.f64, shape=())
        self.derivatives = ti.Vector.field(3, ti.f64, shape=())
        self.metric = ti.Matrix.field(3, 3, ti.f64, shape=())
        self.valid = ti.field(ti.i32, shape=())

    @ti.kernel
    def run(
        self,
        free: ti.i32,
        distance: ti.f64,
        barrier_gradient: ti.f64,
        barrier_hessian: ti.f64,
        measure: ti.f64,
    ):
        metric, valid = curve_barrier_projected_metric(
            self.pointer[None],
            distance,
            self.tangent[None],
            self.curvature[None],
            self.shapes[None],
            self.derivatives[None],
            free,
            barrier_gradient,
            barrier_hessian,
            measure,
        )
        self.metric[None] = metric
        self.valid[None] = valid


@ti.data_oriented
class _SurfaceProjectionHarness:
    def __init__(self):
        self.pointer = ti.Vector.field(3, ti.f64, shape=())
        self.tangent_u = ti.Vector.field(3, ti.f64, shape=())
        self.tangent_v = ti.Vector.field(3, ti.f64, shape=())
        self.curvature_uu = ti.Vector.field(3, ti.f64, shape=())
        self.curvature_vv = ti.Vector.field(3, ti.f64, shape=())
        self.curvature_uv = ti.Vector.field(3, ti.f64, shape=())
        self.shapes = ti.Vector.field(4, ti.f64, shape=())
        self.derivative_u = ti.Vector.field(4, ti.f64, shape=())
        self.derivative_v = ti.Vector.field(4, ti.f64, shape=())
        self.metric = ti.Matrix.field(5, 5, ti.f64, shape=())
        self.valid = ti.field(ti.i32, shape=())

    @ti.kernel
    def run(
        self,
        free_u: ti.i32,
        free_v: ti.i32,
        distance: ti.f64,
        barrier_gradient: ti.f64,
        barrier_hessian: ti.f64,
        measure: ti.f64,
    ):
        metric, valid = surface_barrier_projected_metric(
            self.pointer[None],
            distance,
            self.tangent_u[None],
            self.tangent_v[None],
            self.curvature_uu[None],
            self.curvature_vv[None],
            self.curvature_uv[None],
            self.shapes[None],
            self.derivative_u[None],
            self.derivative_v[None],
            free_u,
            free_v,
            barrier_gradient,
            barrier_hessian,
            measure,
        )
        self.metric[None] = metric
        self.valid[None] = valid


def test_curve_projected_metric_matches_full_numpy_eigendecomposition(
    taichi_runtime,
):
    pointer = np.asarray([0.0, 0.16])
    tangent = np.asarray([1.1, 0.0])
    curvature = np.asarray([0.08, 0.30])
    shapes = np.asarray([0.18, 0.57, 0.25])
    derivatives = np.asarray([-0.75, 0.20, 0.55])
    distance = np.linalg.norm(pointer)
    barrier_gradient = -4.2
    barrier_hessian = 31.0
    measure = 0.73

    harness = _CurveProjectionHarness()
    harness.pointer[None] = pointer
    harness.tangent[None] = tangent
    harness.curvature[None] = curvature
    harness.shapes[None] = shapes
    harness.derivatives[None] = derivatives
    for free in (0, 1):
        harness.run(
            free, distance, barrier_gradient, barrier_hessian, measure
        )
        assert int(harness.valid[None]) == 1
        jacobians, reduced_hessian = _curve_reduced_system(
            pointer,
            tangent,
            curvature,
            shapes,
            derivatives,
            free,
            distance,
            barrier_gradient,
            barrier_hessian,
            measure,
        )
        _assert_reduced_metric_is_full_euclidean_projection(
            jacobians, reduced_hessian, harness.metric.to_numpy()
        )


def test_surface_projected_metric_matches_full_numpy_eigendecomposition(
    taichi_runtime,
):
    pointer = np.asarray([0.0, 0.0, 0.19])
    tangent_u = np.asarray([1.15, 0.12, 0.0])
    tangent_v = np.asarray([0.18, 0.92, 0.0])
    curvature_uu = np.asarray([0.06, -0.03, 0.28])
    curvature_vv = np.asarray([-0.02, 0.04, -0.17])
    curvature_uv = np.asarray([0.03, 0.02, 0.11])
    shapes = np.asarray([0.12, 0.27, 0.36, 0.25])
    derivative_u = np.asarray([-0.42, 0.18, -0.11, 0.35])
    derivative_v = np.asarray([-0.31, -0.09, 0.26, 0.14])
    distance = np.linalg.norm(pointer)
    barrier_gradient = -5.1
    barrier_hessian = 38.0
    measure = 0.61

    harness = _SurfaceProjectionHarness()
    harness.pointer[None] = pointer
    harness.tangent_u[None] = tangent_u
    harness.tangent_v[None] = tangent_v
    harness.curvature_uu[None] = curvature_uu
    harness.curvature_vv[None] = curvature_vv
    harness.curvature_uv[None] = curvature_uv
    harness.shapes[None] = shapes
    harness.derivative_u[None] = derivative_u
    harness.derivative_v[None] = derivative_v
    for free_u, free_v in ((0, 0), (1, 0), (0, 1), (1, 1)):
        harness.run(
            free_u,
            free_v,
            distance,
            barrier_gradient,
            barrier_hessian,
            measure,
        )
        assert int(harness.valid[None]) == 1
        jacobians, reduced_hessian = _surface_reduced_system(
            pointer,
            tangent_u,
            tangent_v,
            curvature_uu,
            curvature_vv,
            curvature_uv,
            shapes,
            derivative_u,
            derivative_v,
            free_u,
            free_v,
            distance,
            barrier_gradient,
            barrier_hessian,
            measure,
        )
        _assert_reduced_metric_is_full_euclidean_projection(
            jacobians, reduced_hessian, harness.metric.to_numpy()
        )
