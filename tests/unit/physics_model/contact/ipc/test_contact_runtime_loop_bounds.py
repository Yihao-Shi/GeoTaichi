"""Device equivalence checks for contact loops above the 3x3 static limit."""

import numpy as np
import pytest
import taichi as ti

from src.nurbs.core.NurbsGeometry import NurbsBasisFunction1d
from src.physics_model.contact_model.ipc.ContactAssembly import (
    scatter_flat_gradient,
    scatter_flat_weighted_gradient,
    scatter_vector_weighted_gradient,
)
from src.physics_model.contact_model.ipc.NurbsContact import (
    curve_barrier_projected_metric,
    get_distance_to_curve_fixed_dim,
    outer_product_nd,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
]


def setup_module():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)


def teardown_module():
    ti.reset()


@ti.data_oriented
class _RuntimeContactLoopHarness:
    def __init__(self):
        self.values = ti.Vector.field(12, ti.f64, shape=4)
        self.weights = ti.field(ti.f64, shape=4)
        self.flat = ti.field(ti.f64, shape=48)
        self.weighted_flat = ti.field(ti.f64, shape=48)
        self.weighted_vector = ti.Vector.field(12, ti.f64, shape=4)
        self.left = ti.Vector.field(4, ti.f64, shape=())
        self.right = ti.Vector.field(4, ti.f64, shape=())
        self.outer = ti.Matrix.field(4, 4, ti.f64, shape=())

    @ti.kernel
    def scatter(self):
        # Contacts remain the GPU-parallel outer loop.  Each 12-component
        # scatter is the runtime serial loop under test.
        for contact in self.values:
            value = self.values[contact]
            scatter_flat_gradient(self.flat, 12 * contact, value)
            scatter_flat_weighted_gradient(
                self.weighted_flat,
                12 * contact,
                self.weights[contact],
                value,
            )
            scatter_vector_weighted_gradient(
                self.weighted_vector,
                contact,
                self.weights[contact],
                value,
            )

    @ti.kernel
    def evaluate_outer_product(self):
        self.outer[None] = outer_product_nd(self.left[None], self.right[None])


@ti.data_oriented
class _CurveMetricHarness:
    def __init__(self):
        self.shape = ti.field(ti.f64, shape=5)
        self.derivative = ti.field(ti.f64, shape=5)
        self.metric = ti.Matrix.field(4, 4, ti.f64, shape=())
        self.valid = ti.field(ti.i32, shape=())

    @ti.kernel
    def evaluate(
        self,
        distance: ti.f64,
        barrier_gradient: ti.f64,
        barrier_hessian: ti.f64,
        measure: ti.f64,
    ):
        pointer = ti.Vector([0.12, -0.18, 0.27])
        tangent = ti.Vector([0.9, 0.15, -0.08])
        curvature = ti.Vector([0.05, -0.03, 0.02])
        shape = ti.Vector.zero(ti.f64, 5)
        derivative = ti.Vector.zero(ti.f64, 5)
        index = 0
        while index < 5:
            shape[index] = self.shape[index]
            derivative[index] = self.derivative[index]
            index += 1
        metric, valid = curve_barrier_projected_metric(
            pointer,
            distance,
            tangent,
            curvature,
            shape,
            derivative,
            1,
            barrier_gradient,
            barrier_hessian,
            measure,
        )
        self.metric[None] = metric
        self.valid[None] = valid


@ti.data_oriented
class _QuarticCurveDistanceHarness:
    def __init__(self):
        self.basis = NurbsBasisFunction1d(4, dimension=2)
        self.knots = ti.field(ti.f64, shape=10)
        self.control = ti.Vector.field(2, ti.f64, shape=5)
        self.weights = ti.field(ti.f64, shape=5)
        self.parameter = ti.field(ti.f64, shape=())
        self.distance = ti.field(ti.f64, shape=())
        self.residual = ti.Vector.field(2, ti.f64, shape=())
        self.knots.from_numpy(np.asarray([0.0] * 5 + [1.0] * 5, dtype=np.float64))
        self.control.from_numpy(np.asarray([[index / 4.0, 0.0] for index in range(5)]))
        self.weights.fill(1.0)

    @ti.kernel
    def query(self, point: ti.types.vector(2, ti.f64)):
        parameter, distance, residual = get_distance_to_curve_fixed_dim(
            0,
            0,
            10,
            self.knots,
            self.control,
            self.weights,
            point,
            self.basis,
        )
        self.parameter[None] = parameter
        self.distance[None] = distance
        self.residual[None] = residual


def _curve_metric_reference(
    shape,
    derivative,
    distance,
    barrier_gradient,
    barrier_hessian,
    measure,
):
    pointer = np.asarray([0.12, -0.18, 0.27])
    tangent = np.asarray([0.9, 0.15, -0.08])
    curvature = np.asarray([0.05, -0.03, 0.02])
    gram = np.zeros((4, 4))
    for value, first in zip(shape, derivative):
        jacobian = np.zeros((4, 3))
        jacobian[:3] = value * np.eye(3)
        jacobian[3] = value * tangent + first * pointer
        gram += jacobian @ jacobian.T
    point_jacobian = np.zeros((4, 3))
    point_jacobian[:3] = -np.eye(3)
    point_jacobian[3] = -tangent
    gram += point_jacobian @ point_jacobian.T

    inverse_distance = 1.0 / distance
    alpha = 0.5 * barrier_gradient * inverse_distance
    beta = 0.25 * (barrier_hessian * inverse_distance**2 - barrier_gradient * inverse_distance**3)
    hessian = np.zeros((4, 4))
    hessian[:3, :3] = (2.0 * alpha * np.eye(3) + 4.0 * beta * np.outer(pointer, pointer)) * measure
    coefficient = tangent @ tangent + curvature @ pointer
    hessian[3, 3] = -2.0 * alpha / coefficient * measure

    scale = np.sqrt(np.maximum(np.diag(gram), 0.0))
    inverse_scale = np.divide(
        1.0,
        scale,
        out=np.zeros_like(scale),
        where=scale > 1.0e-30,
    )
    normalized_gram = inverse_scale[:, None] * gram * inverse_scale[None, :]
    scaled_hessian = scale[:, None] * hessian * scale[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_gram)
    roots = np.sqrt(np.maximum(eigenvalues, 0.0))
    inverse_roots = np.divide(
        1.0,
        roots,
        out=np.zeros_like(roots),
        where=eigenvalues > 128.0 * np.finfo(np.float64).eps * 4,
    )
    square_root = (eigenvectors * roots) @ eigenvectors.T
    inverse_square_root = (eigenvectors * inverse_roots) @ eigenvectors.T
    whitened = square_root @ scaled_hessian @ square_root
    values, vectors = np.linalg.eigh(0.5 * (whitened + whitened.T))
    projected = (vectors * np.maximum(values, 0.0)) @ vectors.T
    normalized_metric = inverse_square_root @ projected @ inverse_square_root
    metric = inverse_scale[:, None] * normalized_metric * inverse_scale[None, :]
    return 0.5 * (metric + metric.T)


def test_large_contact_scatter_and_outer_product_match_dense_reference():
    harness = _RuntimeContactLoopHarness()
    values = (np.arange(48, dtype=np.float64).reshape((4, 12)) - 17.0) / 9.0
    weights = np.asarray([0.5, -1.25, 2.0, 0.75])
    left = np.asarray([0.2, -0.7, 1.1, 0.35])
    right = np.asarray([-0.3, 0.8, 0.45, -1.2])
    harness.values.from_numpy(values)
    harness.weights.from_numpy(weights)
    harness.left[None] = left
    harness.right[None] = right

    harness.scatter()
    harness.evaluate_outer_product()

    np.testing.assert_allclose(harness.flat.to_numpy().reshape((4, 12)), values, atol=0.0)
    np.testing.assert_allclose(
        harness.weighted_flat.to_numpy().reshape((4, 12)),
        weights[:, None] * values,
        atol=0.0,
    )
    np.testing.assert_allclose(
        harness.weighted_vector.to_numpy(),
        weights[:, None] * values,
        atol=0.0,
    )
    np.testing.assert_allclose(harness.outer.to_numpy(), np.outer(left, right), atol=2.0e-15)


def test_large_curve_support_metric_and_quartic_seed_match_references():
    shape = np.asarray([0.08, 0.19, 0.31, 0.27, 0.15])
    derivative = np.asarray([-0.7, -0.25, 0.1, 0.35, 0.5])
    pointer = np.asarray([0.12, -0.18, 0.27])
    distance = float(np.linalg.norm(pointer))
    barrier_gradient = -2.1
    barrier_hessian = 4.7
    measure = 0.63
    metric_harness = _CurveMetricHarness()
    metric_harness.shape.from_numpy(shape)
    metric_harness.derivative.from_numpy(derivative)
    metric_harness.evaluate(distance, barrier_gradient, barrier_hessian, measure)
    expected = _curve_metric_reference(
        shape,
        derivative,
        distance,
        barrier_gradient,
        barrier_hessian,
        measure,
    )
    assert metric_harness.valid[None] == 1
    np.testing.assert_allclose(metric_harness.metric.to_numpy(), expected, rtol=3.0e-10, atol=3.0e-10)

    distance_harness = _QuarticCurveDistanceHarness()
    distance_harness.query(np.asarray([0.37, 0.2]))
    assert np.isclose(distance_harness.parameter[None], 0.37, atol=2.0e-11)
    assert np.isclose(distance_harness.distance[None], 0.2, atol=2.0e-11)
    np.testing.assert_allclose(distance_harness.residual[None], [0.0, -0.2], atol=2.0e-11)
