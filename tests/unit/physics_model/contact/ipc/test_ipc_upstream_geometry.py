"""Closest-point and edge-edge mollifier conformance tests.

The Hessian checks differentiate the public Jacobians and cover the complete
first- and second-derivative contract.
"""

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.physics_model.contact_model.ipc.ContactGeometry import (
    edge_edge_closest_point_terms,
    point_edge_closest_point_terms,
    point_triangle_closest_point_terms,
)
from src.physics_model.contact_model.ipc.ContactMollifier import (
    edge_edge_cross_squarednorm,
    edge_edge_cross_squarednorm_gradient,
    edge_edge_cross_squarednorm_hessian,
    edge_edge_mollifier,
    edge_edge_mollifier_derivative_wrt_eps_x,
    edge_edge_mollifier_gradient,
    edge_edge_mollifier_gradient_derivative_wrt_eps_x,
    edge_edge_mollifier_gradient_jacobian_wrt_x,
    edge_edge_mollifier_gradient_wrt_x,
    edge_edge_mollifier_hessian,
    edge_edge_mollifier_scalar,
    edge_edge_mollifier_scalar_gradient,
    edge_edge_mollifier_scalar_hessian,
    edge_edge_mollifier_threshold,
    edge_edge_mollifier_threshold_gradient,
)


def setup_module():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)


def teardown_module():
    ti.reset()


def _fd_jacobian(function, x, step=1.0e-6):
    x = np.asarray(x, dtype=np.float64)
    value = np.atleast_1d(function(x))
    jacobian = np.zeros((value.size, x.size), dtype=np.float64)
    for variable in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[variable] += step
        minus[variable] -= step
        jacobian[:, variable] = (np.atleast_1d(function(plus)) - np.atleast_1d(function(minus))) / (2.0 * step)
    return jacobian


def _fd_gradient(function, x, step=1.0e-6):
    return _fd_jacobian(function, x, step=step).reshape(-1)


@ti.data_oriented
class _GeometryHarness:
    def __init__(self):
        self.pe2_coordinate = ti.field(ti.f64, shape=())
        self.pe2_jacobian = ti.field(ti.f64, shape=6)
        self.pe2_hessian = ti.field(ti.f64, shape=(6, 6))
        self.pe3_coordinate = ti.field(ti.f64, shape=())
        self.pe3_jacobian = ti.field(ti.f64, shape=9)
        self.pe3_hessian = ti.field(ti.f64, shape=(9, 9))
        self.pt_coordinates = ti.Vector.field(2, ti.f64, shape=())
        self.pt_jacobian = ti.Matrix.field(2, 12, ti.f64, shape=())
        self.pt_hessian0 = ti.Matrix.field(12, 12, ti.f64, shape=())
        self.pt_hessian1 = ti.Matrix.field(12, 12, ti.f64, shape=())
        self.ee_coordinates = ti.Vector.field(2, ti.f64, shape=())
        self.ee_jacobian = ti.Matrix.field(2, 12, ti.f64, shape=())
        self.ee_hessian0 = ti.Matrix.field(12, 12, ti.f64, shape=())
        self.ee_hessian1 = ti.Matrix.field(12, 12, ti.f64, shape=())

    @ti.kernel
    def evaluate_pe2(
        self,
        point: ti.types.vector(2, ti.f64),
        endpoint0: ti.types.vector(2, ti.f64),
        endpoint1: ti.types.vector(2, ti.f64),
    ):
        coordinate, jacobian, hessian = point_edge_closest_point_terms(point, endpoint0, endpoint1)
        self.pe2_coordinate[None] = coordinate
        i = 0
        while i < 6:
            self.pe2_jacobian[i] = jacobian[i]
            j = 0
            while j < 6:
                self.pe2_hessian[i, j] = hessian[i, j]
                j += 1
            i += 1

    @ti.kernel
    def evaluate_pe3(
        self,
        point: ti.types.vector(3, ti.f64),
        endpoint0: ti.types.vector(3, ti.f64),
        endpoint1: ti.types.vector(3, ti.f64),
    ):
        coordinate, jacobian, hessian = point_edge_closest_point_terms(point, endpoint0, endpoint1)
        self.pe3_coordinate[None] = coordinate
        i = 0
        while i < 9:
            self.pe3_jacobian[i] = jacobian[i]
            j = 0
            while j < 9:
                self.pe3_hessian[i, j] = hessian[i, j]
                j += 1
            i += 1

    @ti.kernel
    def evaluate_pt(
        self,
        point: ti.types.vector(3, ti.f64),
        vertex0: ti.types.vector(3, ti.f64),
        vertex1: ti.types.vector(3, ti.f64),
        vertex2: ti.types.vector(3, ti.f64),
    ):
        coordinates, jacobian, hessian0, hessian1 = point_triangle_closest_point_terms(point, vertex0, vertex1, vertex2)
        self.pt_coordinates[None] = coordinates
        self.pt_jacobian[None] = jacobian
        self.pt_hessian0[None] = hessian0
        self.pt_hessian1[None] = hessian1

    @ti.kernel
    def evaluate_ee(
        self,
        endpoint_a0: ti.types.vector(3, ti.f64),
        endpoint_a1: ti.types.vector(3, ti.f64),
        endpoint_b0: ti.types.vector(3, ti.f64),
        endpoint_b1: ti.types.vector(3, ti.f64),
    ):
        coordinates, jacobian, hessian0, hessian1 = edge_edge_closest_point_terms(
            endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1
        )
        self.ee_coordinates[None] = coordinates
        self.ee_jacobian[None] = jacobian
        self.ee_hessian0[None] = hessian0
        self.ee_hessian1[None] = hessian1

    def pe2(self, x):
        points = np.asarray(x, dtype=np.float64).reshape(3, 2)
        self.evaluate_pe2(*points)
        return (
            float(self.pe2_coordinate[None]),
            self.pe2_jacobian.to_numpy(),
            self.pe2_hessian.to_numpy(),
        )

    def pe3(self, x):
        points = np.asarray(x, dtype=np.float64).reshape(3, 3)
        self.evaluate_pe3(*points)
        return (
            float(self.pe3_coordinate[None]),
            self.pe3_jacobian.to_numpy(),
            self.pe3_hessian.to_numpy(),
        )

    def pt(self, x):
        points = np.asarray(x, dtype=np.float64).reshape(4, 3)
        self.evaluate_pt(*points)
        return (
            np.asarray(self.pt_coordinates[None]),
            np.asarray(self.pt_jacobian[None]),
            np.asarray(self.pt_hessian0[None]),
            np.asarray(self.pt_hessian1[None]),
        )

    def ee(self, x):
        points = np.asarray(x, dtype=np.float64).reshape(4, 3)
        self.evaluate_ee(*points)
        return (
            np.asarray(self.ee_coordinates[None]),
            np.asarray(self.ee_jacobian[None]),
            np.asarray(self.ee_hessian0[None]),
            np.asarray(self.ee_hessian1[None]),
        )


@ti.data_oriented
class _MollifierHarness:
    def __init__(self):
        self.scalar_terms = ti.Vector.field(5, ti.f64, shape=())
        self.cross_value = ti.field(ti.f64, shape=())
        self.cross_gradient = ti.Vector.field(12, ti.f64, shape=())
        self.cross_hessian = ti.Matrix.field(12, 12, ti.f64, shape=())
        self.threshold = ti.field(ti.f64, shape=())
        self.threshold_gradient = ti.Vector.field(12, ti.f64, shape=())
        self.mollifier = ti.field(ti.f64, shape=())
        self.mollifier_gradient = ti.Vector.field(12, ti.f64, shape=())
        self.mollifier_hessian = ti.Matrix.field(12, 12, ti.f64, shape=())
        self.rest_gradient = ti.Vector.field(12, ti.f64, shape=())
        self.rest_jacobian = ti.Matrix.field(12, 12, ti.f64, shape=())

    @ti.kernel
    def evaluate_scalar(self, cross_squarednorm: ti.f64, eps_x: ti.f64):
        self.scalar_terms[None] = ti.Vector(
            [
                edge_edge_mollifier_scalar(cross_squarednorm, eps_x),
                edge_edge_mollifier_scalar_gradient(cross_squarednorm, eps_x),
                edge_edge_mollifier_scalar_hessian(cross_squarednorm, eps_x),
                edge_edge_mollifier_derivative_wrt_eps_x(cross_squarednorm, eps_x),
                edge_edge_mollifier_gradient_derivative_wrt_eps_x(cross_squarednorm, eps_x),
            ]
        )

    @ti.kernel
    def evaluate(
        self,
        rest_a0: ti.types.vector(3, ti.f64),
        rest_a1: ti.types.vector(3, ti.f64),
        rest_b0: ti.types.vector(3, ti.f64),
        rest_b1: ti.types.vector(3, ti.f64),
        a0: ti.types.vector(3, ti.f64),
        a1: ti.types.vector(3, ti.f64),
        b0: ti.types.vector(3, ti.f64),
        b1: ti.types.vector(3, ti.f64),
    ):
        eps_x = edge_edge_mollifier_threshold(rest_a0, rest_a1, rest_b0, rest_b1)
        self.cross_value[None] = edge_edge_cross_squarednorm(a0, a1, b0, b1)
        self.cross_gradient[None] = edge_edge_cross_squarednorm_gradient(a0, a1, b0, b1)
        self.cross_hessian[None] = edge_edge_cross_squarednorm_hessian(a0, a1, b0, b1)
        self.threshold[None] = eps_x
        self.threshold_gradient[None] = edge_edge_mollifier_threshold_gradient(rest_a0, rest_a1, rest_b0, rest_b1)
        self.mollifier[None] = edge_edge_mollifier(a0, a1, b0, b1, eps_x)
        self.mollifier_gradient[None] = edge_edge_mollifier_gradient(a0, a1, b0, b1, eps_x)
        self.mollifier_hessian[None] = edge_edge_mollifier_hessian(a0, a1, b0, b1, eps_x)
        self.rest_gradient[None] = edge_edge_mollifier_gradient_wrt_x(
            rest_a0, rest_a1, rest_b0, rest_b1, a0, a1, b0, b1
        )
        self.rest_jacobian[None] = edge_edge_mollifier_gradient_jacobian_wrt_x(
            rest_a0, rest_a1, rest_b0, rest_b1, a0, a1, b0, b1
        )

    def scalar(self, cross_squarednorm, eps_x):
        self.evaluate_scalar(float(cross_squarednorm), float(eps_x))
        return np.asarray(self.scalar_terms[None])

    def query(self, rest, current):
        rest = np.asarray(rest, dtype=np.float64).reshape(4, 3)
        current = np.asarray(current, dtype=np.float64).reshape(4, 3)
        self.evaluate(*rest, *current)
        return {
            "cross": float(self.cross_value[None]),
            "cross_gradient": np.asarray(self.cross_gradient[None]),
            "cross_hessian": np.asarray(self.cross_hessian[None]),
            "threshold": float(self.threshold[None]),
            "threshold_gradient": np.asarray(self.threshold_gradient[None]),
            "mollifier": float(self.mollifier[None]),
            "mollifier_gradient": np.asarray(self.mollifier_gradient[None]),
            "mollifier_hessian": np.asarray(self.mollifier_hessian[None]),
            "rest_gradient": np.asarray(self.rest_gradient[None]),
            "rest_jacobian": np.asarray(self.rest_jacobian[None]),
        }


@pytest.fixture(scope="module")
def geometry_harness():
    return _GeometryHarness()


@pytest.fixture(scope="module")
def mollifier_harness():
    return _MollifierHarness()


def test_upstream_point_triangle_closest_point(geometry_harness):
    vertex0 = np.array([-1.0, 0.0, 1.0])
    vertex1 = np.array([1.0, 0.0, 1.0])
    vertex2 = np.array([0.0, 0.0, -1.0])
    expected = np.array([0.5, 0.5])
    point = vertex0 + expected[0] * (vertex1 - vertex0) + expected[1] * (vertex2 - vertex0)
    x = np.concatenate([point, vertex0, vertex1, vertex2])

    coordinates, jacobian, hessian0, hessian1 = geometry_harness.pt(x)
    assert np.allclose(coordinates, expected, atol=1.0e-12)
    reconstructed = vertex0 + coordinates[0] * (vertex1 - vertex0) + coordinates[1] * (vertex2 - vertex0)
    assert np.allclose(point, reconstructed, atol=1.0e-12)
    fd_jacobian = _fd_jacobian(lambda value: geometry_harness.pt(value)[0], x)
    assert np.allclose(jacobian, fd_jacobian, atol=2.0e-8, rtol=2.0e-8)
    fd_hessian0 = _fd_jacobian(lambda value: geometry_harness.pt(value)[1][0], x)
    fd_hessian1 = _fd_jacobian(lambda value: geometry_harness.pt(value)[1][1], x)
    assert np.allclose(hessian0, fd_hessian0, atol=2.0e-7, rtol=2.0e-7)
    assert np.allclose(hessian1, fd_hessian1, atol=2.0e-7, rtol=2.0e-7)


def test_upstream_edge_edge_closest_point(geometry_harness):
    x = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    ).reshape(-1)
    coordinates, jacobian, hessian0, hessian1 = geometry_harness.ee(x)
    assert np.allclose(coordinates, [0.5, 0.5], atol=1.0e-12)
    fd_jacobian = _fd_jacobian(lambda value: geometry_harness.ee(value)[0], x)
    assert np.allclose(jacobian, fd_jacobian, atol=2.0e-8, rtol=2.0e-8)
    fd_hessian0 = _fd_jacobian(lambda value: geometry_harness.ee(value)[1][0], x)
    fd_hessian1 = _fd_jacobian(lambda value: geometry_harness.ee(value)[1][1], x)
    assert np.allclose(hessian0, fd_hessian0, atol=2.0e-7, rtol=2.0e-7)
    assert np.allclose(hessian1, fd_hessian1, atol=2.0e-7, rtol=2.0e-7)


@pytest.mark.parametrize("dimension", [2, 3])
def test_upstream_point_edge_closest_point(geometry_harness, dimension):
    if dimension == 2:
        x = np.array([[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]).reshape(-1)
        query = geometry_harness.pe2
    else:
        x = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).reshape(-1)
        query = geometry_harness.pe3
    coordinate, jacobian, hessian = query(x)
    assert coordinate == pytest.approx(0.5, abs=1.0e-12)
    fd_jacobian = _fd_gradient(lambda value: query(value)[0], x)
    assert np.allclose(jacobian, fd_jacobian, atol=2.0e-8, rtol=2.0e-8)
    fd_hessian = _fd_jacobian(lambda value: query(value)[1], x)
    assert np.allclose(hessian, fd_hessian, atol=2.0e-7, rtol=2.0e-7)


def test_point_edge_closest_point_non_axis_aligned(geometry_harness):
    endpoint0 = np.array([-0.7, 0.4, -0.2])
    edge = np.array([1.9, -1.3, 0.8])
    endpoint1 = endpoint0 + edge
    expected = 0.37
    normal = np.array([edge[1], -edge[0], 0.0])
    normal /= np.linalg.norm(normal)
    point = endpoint0 + expected * edge + 0.6 * normal
    x = np.concatenate([point, endpoint0, endpoint1])

    coordinate, jacobian, hessian = geometry_harness.pe3(x)
    numpy_coordinate = np.dot(point - endpoint0, edge) / np.dot(edge, edge)
    assert coordinate == pytest.approx(numpy_coordinate, abs=1.0e-12)
    assert coordinate == pytest.approx(expected, abs=1.0e-12)
    fd_jacobian = _fd_gradient(lambda value: geometry_harness.pe3(value)[0], x)
    assert np.allclose(jacobian, fd_jacobian, atol=2.0e-8, rtol=2.0e-8)
    fd_hessian = _fd_jacobian(lambda value: geometry_harness.pe3(value)[1], x)
    assert np.allclose(hessian, fd_hessian, atol=2.0e-7, rtol=2.0e-7)


@pytest.mark.parametrize("signed_offset", [0.0, -1.0e-12, 1.0e-12, -1.0e-6, 1.0e-6])
def test_point_triangle_closest_point_zero_gap_stability(geometry_harness, signed_offset):
    vertex0 = np.array([-0.6, 0.2, 0.1])
    vertex1 = np.array([1.4, -0.3, 0.7])
    vertex2 = np.array([0.1, 1.6, -0.4])
    edge01 = vertex1 - vertex0
    edge02 = vertex2 - vertex0
    normal = np.cross(edge01, edge02)
    normal /= np.linalg.norm(normal)
    expected = np.array([0.23, 0.31])
    point = vertex0 + expected[0] * edge01 + expected[1] * edge02 + signed_offset * normal
    x = np.concatenate([point, vertex0, vertex1, vertex2])

    terms = geometry_harness.pt(x)
    assert np.allclose(terms[0], expected, atol=2.0e-12, rtol=0.0)
    for term in terms:
        assert np.all(np.isfinite(term))


@pytest.mark.parametrize("signed_offset", [0.0, -1.0e-12, 1.0e-12, -1.0e-6, 1.0e-6])
def test_edge_edge_closest_point_intersection_stability(geometry_harness, signed_offset):
    endpoint_a0 = np.array([-0.4, 0.2, 0.7])
    edge_a = np.array([1.7, 0.4, -0.2])
    edge_b = np.array([-0.3, 1.2, 0.5])
    expected = np.array([0.37, 0.62])
    point_on_a = endpoint_a0 + expected[0] * edge_a
    normal = np.cross(edge_a, edge_b)
    normal /= np.linalg.norm(normal)
    endpoint_b0 = point_on_a - expected[1] * edge_b + signed_offset * normal
    x = np.concatenate(
        [
            endpoint_a0,
            endpoint_a0 + edge_a,
            endpoint_b0,
            endpoint_b0 + edge_b,
        ]
    )

    terms = geometry_harness.ee(x)
    assert np.allclose(terms[0], expected, atol=2.0e-12, rtol=0.0)
    for term in terms:
        assert np.all(np.isfinite(term))


def _cross_squarednorm_numpy(flattened):
    points = np.asarray(flattened).reshape(4, 3)
    cross = np.cross(points[1] - points[0], points[3] - points[2])
    return float(cross @ cross)


@pytest.mark.parametrize(
    "current, expected",
    [
        (
            np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
            16.0,
        ),
        (
            np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 1.0e-9, 0.0], [1.0, -1.0e-9, 0.0]]),
            0.0,
        ),
    ],
)
def test_upstream_edge_edge_cross_squarednorm(mollifier_harness, current, expected):
    result = mollifier_harness.query(current, current)
    assert result["cross"] == pytest.approx(expected, abs=1.0e-9)
    flattened = current.reshape(-1)
    fd_gradient = _fd_gradient(_cross_squarednorm_numpy, flattened)
    assert np.allclose(result["cross_gradient"], fd_gradient, atol=2.0e-8, rtol=2.0e-8)
    fd_hessian = _fd_jacobian(
        lambda value: mollifier_harness.query(value, value)["cross_gradient"],
        flattened,
    )
    assert np.allclose(result["cross_hessian"], fd_hessian, atol=2.0e-7, rtol=2.0e-7)


@pytest.mark.parametrize("relative_x", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("eps_x", [1.0e-3, 1.0e-1, 1.0, 2.0])
def test_upstream_edge_edge_mollifier_scalar(mollifier_harness, relative_x, eps_x):
    cross_squarednorm = relative_x * eps_x
    value, gradient, hessian, eps_derivative, mixed_derivative = mollifier_harness.scalar(cross_squarednorm, eps_x)
    assert 0.0 <= value <= 1.0
    if cross_squarednorm > eps_x:
        assert value == pytest.approx(1.0)

    if relative_x < 1.0:
        assert value == pytest.approx(relative_x * (2.0 - relative_x))
        assert gradient == pytest.approx(2.0 * (1.0 - relative_x) / eps_x)
        assert hessian == pytest.approx(-2.0 / (eps_x * eps_x))
        assert eps_derivative == pytest.approx(2.0 * cross_squarednorm * (-eps_x + cross_squarednorm) / eps_x**3)
        assert mixed_derivative == pytest.approx(2.0 * (-eps_x + 2.0 * cross_squarednorm) / eps_x**3)
    else:
        assert gradient == pytest.approx(0.0)
        assert hessian == pytest.approx(0.0)
        assert eps_derivative == pytest.approx(0.0)
        assert mixed_derivative == pytest.approx(0.0)


def _threshold_numpy(flattened):
    points = np.asarray(flattened).reshape(4, 3)
    edge_a = points[1] - points[0]
    edge_b = points[3] - points[2]
    return 1.0e-3 * (edge_a @ edge_a) * (edge_b @ edge_b)


def test_upstream_edge_edge_mollifier_threshold(mollifier_harness):
    edges = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    result = mollifier_harness.query(edges, edges)
    assert result["threshold"] == pytest.approx(0.016)
    fd_gradient = _fd_gradient(_threshold_numpy, edges.reshape(-1))
    assert np.allclose(result["threshold_gradient"], fd_gradient, atol=2.0e-10, rtol=2.0e-8)


def test_upstream_edge_edge_mollifier_endpoint_derivatives(mollifier_harness):
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    current = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.02, 0.0], [0.5, 0.04, 0.0]])
    result = mollifier_harness.query(rest, current)
    eps_x = result["threshold"]
    flattened = current.reshape(-1)

    def value_function(value):
        return mollifier_harness.query(rest, value)["mollifier"]

    fd_gradient = _fd_gradient(value_function, flattened, step=1.0e-7)
    fd_hessian = _fd_jacobian(
        lambda value: mollifier_harness.query(rest, value)["mollifier_gradient"],
        flattened,
        step=1.0e-7,
    )
    assert 0.0 < result["mollifier"] < 1.0
    assert np.allclose(result["mollifier_gradient"], fd_gradient, atol=2.0e-7, rtol=2.0e-6)
    assert np.allclose(result["mollifier_hessian"], fd_hessian, atol=2.0e-5, rtol=2.0e-6)

    displacement = current.reshape(-1) - rest.reshape(-1)

    def shape_value(rest_flattened):
        deformed = rest_flattened + displacement
        return mollifier_harness.query(rest_flattened, deformed)["mollifier"]

    # The threshold contribution is exposed separately from the
    # ordinary current-position gradient.
    total_shape_gradient = result["mollifier_gradient"] + result["rest_gradient"]
    fd_shape_gradient = _fd_gradient(shape_value, rest.reshape(-1), step=1.0e-7)
    assert np.allclose(total_shape_gradient, fd_shape_gradient, atol=3.0e-7, rtol=3.0e-6)

    def current_gradient_with_fixed_displacement(rest_flattened):
        deformed = rest_flattened + displacement
        return mollifier_harness.query(rest_flattened, deformed)["mollifier_gradient"]

    fd_rest_jacobian = _fd_jacobian(
        current_gradient_with_fixed_displacement,
        rest.reshape(-1),
        step=1.0e-7,
    )
    # This mixed derivative stores rest-position DOFs on
    # rows and current-gradient components on columns. ``_fd_jacobian`` uses
    # the conventional output-component rows / input-DOF columns ordering.
    # This quantity omits the ordinary scalar-curvature term
    # ``m_ss (grad s)(grad s)^T``; that term belongs to the usual endpoint
    # mollifier Hessian. Remove it from the total fixed-displacement FD before
    # comparing the specialized shape-derivative block.
    scalar_hessian = mollifier_harness.scalar(result["cross"], eps_x)[2]
    fd_upstream_jacobian = fd_rest_jacobian.T - scalar_hessian * np.outer(
        result["cross_gradient"], result["cross_gradient"]
    )
    assert np.allclose(result["rest_jacobian"], fd_upstream_jacobian, atol=2.0e-5, rtol=3.0e-6)

    # Holding current endpoints fixed isolates the named
    # ``gradient_wrt_x`` threshold contribution.
    fd_threshold_only = _fd_gradient(
        lambda rest_flattened: mollifier_harness.query(rest_flattened, flattened)["mollifier"],
        rest.reshape(-1),
        step=1.0e-7,
    )
    assert np.allclose(result["rest_gradient"], fd_threshold_only, atol=2.0e-7, rtol=3.0e-6)
    assert eps_x == pytest.approx(1.0e-3)
