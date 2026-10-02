"""Analytic derivative contracts for planar point--edge IPC geometry."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.contact_model.ipc.ContactDistance import (
    point_edge_distance_grad_hess_2d,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.geometry,
    pytest.mark.cpu,
]


@ti.data_oriented
class _PointEdgeDerivativeHarness:
    def __init__(self):
        self.positions = ti.Vector.field(2, ti.f64, shape=3)
        self.distance2 = ti.field(ti.f64, shape=())
        self.gradient = ti.Vector.field(6, ti.f64, shape=())
        self.hessian = ti.Matrix.field(6, 6, ti.f64, shape=())
        self.distance_type = ti.field(ti.i32, shape=())

    @ti.kernel
    def evaluate(self):
        distance2, gradient, hessian, distance_type = (
            point_edge_distance_grad_hess_2d(
                self.positions[0],
                self.positions[1],
                self.positions[2],
            )
        )
        self.distance2[None] = distance2
        self.gradient[None] = gradient
        self.hessian[None] = hessian
        self.distance_type[None] = distance_type


def _distance2(coordinates):
    point, endpoint0, endpoint1 = coordinates.reshape(3, 2)
    edge = endpoint1 - endpoint0
    ratio = np.dot(point - endpoint0, edge) / np.dot(edge, edge)
    closest = endpoint0 + np.clip(ratio, 0.0, 1.0) * edge
    return float(np.dot(point - closest, point - closest))


def _finite_difference_derivatives(coordinates, step=2.0e-5):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    size = coordinates.size
    gradient = np.zeros(size, dtype=np.float64)
    hessian = np.zeros((size, size), dtype=np.float64)
    center = _distance2(coordinates)
    for row in range(size):
        positive = coordinates.copy()
        negative = coordinates.copy()
        positive[row] += step
        negative[row] -= step
        gradient[row] = (
            _distance2(positive) - _distance2(negative)
        ) / (2.0 * step)
        hessian[row, row] = (
            _distance2(positive) - 2.0 * center + _distance2(negative)
        ) / (step * step)
        for column in range(row):
            pp = coordinates.copy()
            pm = coordinates.copy()
            mp = coordinates.copy()
            mm = coordinates.copy()
            pp[row] += step
            pp[column] += step
            pm[row] += step
            pm[column] -= step
            mp[row] -= step
            mp[column] += step
            mm[row] -= step
            mm[column] -= step
            value = (
                _distance2(pp)
                - _distance2(pm)
                - _distance2(mp)
                + _distance2(mm)
            ) / (4.0 * step * step)
            hessian[row, column] = value
            hessian[column, row] = value
    return gradient, hessian


def test_interior_point_edge_derivatives_are_analytic(taichi_runtime):
    coordinates = np.array(
        [0.2, 0.7, -0.5, 0.0, 1.2, 0.3], dtype=np.float64
    )
    expected_gradient, expected_hessian = _finite_difference_derivatives(
        coordinates
    )
    harness = _PointEdgeDerivativeHarness()
    harness.positions.from_numpy(coordinates.reshape(3, 2))
    harness.evaluate()

    assert int(harness.distance_type[None]) == 2
    assert float(harness.distance2[None]) == pytest.approx(
        _distance2(coordinates), abs=1.0e-13
    )
    np.testing.assert_allclose(
        harness.gradient.to_numpy(), expected_gradient, rtol=2.0e-8, atol=2.0e-8
    )
    np.testing.assert_allclose(
        harness.hessian.to_numpy(), expected_hessian, rtol=4.0e-6, atol=4.0e-6
    )
    np.testing.assert_allclose(
        harness.hessian.to_numpy(),
        harness.hessian.to_numpy().T,
        rtol=0.0,
        atol=2.0e-13,
    )


def test_endpoint_region_reduces_to_exact_point_point_terms(taichi_runtime):
    coordinates = np.array(
        [-1.0, 0.5, 0.0, 0.0, 1.0, 0.0], dtype=np.float64
    )
    harness = _PointEdgeDerivativeHarness()
    harness.positions.from_numpy(coordinates.reshape(3, 2))
    harness.evaluate()

    assert int(harness.distance_type[None]) == 0
    np.testing.assert_allclose(
        harness.gradient.to_numpy(),
        np.array([-2.0, 1.0, 2.0, -1.0, 0.0, 0.0]),
        rtol=0.0,
        atol=1.0e-14,
    )
    eigenvalues = np.linalg.eigvalsh(harness.hessian.to_numpy())
    assert eigenvalues.min() >= -1.0e-13

