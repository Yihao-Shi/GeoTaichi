"""Pinned isotropic C1 lagged-friction potential conformance tests.

The tangent basis, closest point, and normal force are frozen. The suite
checks all six collision stencils and their finite-difference derivatives.
Anisotropic and Stribeck extensions are outside this test's scope.
"""

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.physics_model.contact_model.ipc.IPC import (
    ipc_friction_f0,
    ipc_friction_f1_over_speed,
    ipc_friction_hessian_term,
)


@pytest.fixture(scope="module", autouse=True)
def _initialize_taichi():
    ti.init(
        arch=ti.cpu,
        cpu_max_num_threads=2,
        offline_cache=False,
        default_fp=ti.f64,
        debug=False,
    )


@ti.kernel
def _eval_frozen_isotropic_potential(
    velocity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    tangent_map: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ndof: ti.i32,
    tangent_dimension: ti.i32,
    scale: ti.f64,
    epsv_times_h: ti.f64,
    energy: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gradient: ti.types.ndarray(dtype=ti.f64, ndim=1),
    hessian: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    tangential_velocity = ti.Vector.zero(ti.f64, 2)
    for i in range(ndof):
        tangential_velocity[0] += tangent_map[i, 0] * velocity[i]
        if tangent_dimension == 2:
            tangential_velocity[1] += tangent_map[i, 1] * velocity[i]

    speed = tangential_velocity.norm()
    f1_over_speed = ipc_friction_f1_over_speed(speed, epsv_times_h)
    radial_derivative = ipc_friction_hessian_term(speed, epsv_times_h)
    energy[0] = scale * ipc_friction_f0(speed, epsv_times_h, 1.0)

    inner = f1_over_speed * ti.Matrix.identity(ti.f64, 2)
    if speed > 0.0:
        inner += (radial_derivative / speed) * tangential_velocity.outer_product(tangential_velocity)

    for i in range(12):
        gradient[i] = 0.0
        for j in range(12):
            hessian[i, j] = 0.0

    for i in range(ndof):
        for alpha in range(tangent_dimension):
            gradient[i] += scale * f1_over_speed * tangent_map[i, alpha] * tangential_velocity[alpha]
        for j in range(ndof):
            for alpha in range(tangent_dimension):
                for beta in range(tangent_dimension):
                    hessian[i, j] += scale * tangent_map[i, alpha] * inner[alpha, beta] * tangent_map[j, beta]


def _orthonormal_tangent_basis(normal):
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal.size == 2:
        return np.array([[-normal[1]], [normal[0]]], dtype=np.float64)

    seed = np.array([1.0, 0.0, 0.0])
    if abs(normal @ seed) > 0.8:
        seed = np.array([0.0, 1.0, 0.0])
    tangent0 = seed - (seed @ normal) * normal
    tangent0 /= np.linalg.norm(tangent0)
    tangent1 = np.cross(normal, tangent0)
    return np.column_stack((tangent0, tangent1))


def _make_tangent_map(coefficients, tangent_basis):
    """Construct T = Gamma.T @ P for a frozen collision stencil."""
    tangent_basis = np.asarray(tangent_basis, dtype=np.float64)
    dimension, tangent_dimension = tangent_basis.shape
    gamma = np.hstack([coefficient * np.eye(dimension) for coefficient in coefficients])
    tangent_map = gamma.T @ tangent_basis
    assert tangent_map.shape == (
        dimension * len(coefficients),
        tangent_dimension,
    )
    return tangent_map


def _upstream_stencil_cases():
    d = 0.2

    # point-triangle: the projected point has barycentric coordinates 1/3.
    pt_basis = _orthonormal_tangent_basis([0.0, 1.0, 0.0])
    point_triangle = _make_tangent_map([1.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0], pt_basis)

    # edge-edge: both closest-point coordinates are 1/2.
    ee_basis = _orthonormal_tangent_basis([0.0, 1.0, 0.0])
    edge_edge = _make_tangent_map([0.5, 0.5, -0.5, -0.5], ee_basis)

    # point-edge 3D: the closest point is at the edge midpoint.  The normal
    # matches the generator's point (-0.5, d, 0) and z-aligned edge.
    pe3_basis = _orthonormal_tangent_basis([-0.5, d, 0.0])
    point_edge_3d = _make_tangent_map([1.0, -0.5, -0.5], pe3_basis)

    # point-point 3D: the lagged points are separated along x.
    pp3_basis = _orthonormal_tangent_basis([-1.0, 0.0, 0.0])
    point_point_3d = _make_tangent_map([1.0, -1.0], pp3_basis)

    # point-edge 2D: projection parameter is 1/4 for p=(-0.5,d).
    pe2_basis = _orthonormal_tangent_basis([0.0, d])
    point_edge_2d = _make_tangent_map([1.0, -0.75, -0.25], pe2_basis)

    # point-point 2D: the lagged points are separated along x.
    pp2_basis = _orthonormal_tangent_basis([-1.0, 0.0])
    point_point_2d = _make_tangent_map([1.0, -1.0], pp2_basis)

    return [
        pytest.param(point_triangle, id="point-triangle-3d"),
        pytest.param(edge_edge, id="edge-edge-3d"),
        pytest.param(point_edge_3d, id="point-edge-3d"),
        pytest.param(point_point_3d, id="point-point-3d"),
        pytest.param(point_edge_2d, id="point-edge-2d"),
        pytest.param(point_point_2d, id="point-point-2d"),
    ]


UPSTREAM_STENCIL_CASES = _upstream_stencil_cases()


def _host_mollifier_terms(speed, epsv_times_h):
    if speed >= epsv_times_h:
        return speed, 1.0 / speed, -1.0 / (speed * speed)
    f0 = speed * speed * (1.0 - speed / (3.0 * epsv_times_h)) / epsv_times_h + epsv_times_h / 3.0
    f1_over_speed = (2.0 - speed / epsv_times_h) / epsv_times_h
    radial_derivative = -1.0 / (epsv_times_h * epsv_times_h)
    return f0, f1_over_speed, radial_derivative


def _host_frozen_potential(velocity, tangent_map, scale, epsv_times_h):
    tangential_velocity = tangent_map.T @ velocity
    speed = float(np.linalg.norm(tangential_velocity))
    f0, f1_over_speed, radial_derivative = _host_mollifier_terms(speed, epsv_times_h)
    inner = f1_over_speed * np.eye(tangent_map.shape[1])
    if speed > 0.0:
        inner += (radial_derivative / speed) * np.outer(tangential_velocity, tangential_velocity)
    energy = scale * f0
    gradient = scale * f1_over_speed * tangent_map @ tangential_velocity
    hessian = scale * tangent_map @ inner @ tangent_map.T
    return energy, gradient, hessian


def _evaluate_taichi(velocity, tangent_map, scale, epsv_times_h):
    ndof, tangent_dimension = tangent_map.shape
    padded_velocity = np.zeros(12, dtype=np.float64)
    padded_velocity[:ndof] = velocity
    padded_tangent_map = np.zeros((12, 2), dtype=np.float64)
    padded_tangent_map[:ndof, :tangent_dimension] = tangent_map
    energy = np.zeros(1, dtype=np.float64)
    gradient = np.zeros(12, dtype=np.float64)
    hessian = np.zeros((12, 12), dtype=np.float64)
    _eval_frozen_isotropic_potential(
        padded_velocity,
        padded_tangent_map,
        ndof,
        tangent_dimension,
        scale,
        epsv_times_h,
        energy,
        gradient,
        hessian,
    )
    return energy[0], gradient[:ndof], hessian[:ndof, :ndof]


def _velocity_for_tangential_speed(tangent_map, speed):
    tangent_dimension = tangent_map.shape[1]
    if speed == 0.0:
        tangential_velocity = np.zeros(tangent_dimension)
    elif tangent_dimension == 1:
        tangential_velocity = np.array([speed])
    else:
        direction = np.array([0.8, 0.6])
        tangential_velocity = speed * direction
    gram = tangent_map.T @ tangent_map
    velocity = tangent_map @ np.linalg.solve(gram, tangential_velocity)
    np.testing.assert_allclose(
        tangent_map.T @ velocity,
        tangential_velocity,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    return velocity


@pytest.mark.parametrize("speed_ratio", [0.0, 0.37, 1.7])
@pytest.mark.parametrize("tangent_map", UPSTREAM_STENCIL_CASES)
def test_isotropic_friction_potential_gradient_and_hessian_by_fd(tangent_map, speed_ratio):
    """Check potential, gradient, and Hessian for all six stencils."""
    epsv_times_h = 0.4
    mu = 0.6
    normal_force_magnitude = 7.5
    scale = mu * normal_force_magnitude
    velocity = _velocity_for_tangential_speed(tangent_map, speed_ratio * epsv_times_h)

    actual_energy, actual_gradient, actual_hessian = _evaluate_taichi(velocity, tangent_map, scale, epsv_times_h)
    expected_energy, expected_gradient, expected_hessian = _host_frozen_potential(
        velocity, tangent_map, scale, epsv_times_h
    )
    assert actual_energy == pytest.approx(expected_energy, rel=3.0e-13)
    np.testing.assert_allclose(
        actual_gradient,
        expected_gradient,
        rtol=3.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        actual_hessian,
        expected_hessian,
        rtol=5.0e-13,
        atol=2.0e-12,
    )

    # Upstream holds the collision frame fixed and finite-differences V1.
    # Here velocity == V1 - V0, so perturbing velocity is the same operation.
    step = 2.0e-7 * epsv_times_h
    finite_difference_gradient = np.zeros_like(velocity)
    finite_difference_hessian = np.zeros_like(actual_hessian)
    for column in range(velocity.size):
        direction = np.zeros_like(velocity)
        direction[column] = step
        plus = _evaluate_taichi(velocity + direction, tangent_map, scale, epsv_times_h)
        minus = _evaluate_taichi(velocity - direction, tangent_map, scale, epsv_times_h)
        finite_difference_gradient[column] = (plus[0] - minus[0]) / (2.0 * step)
        finite_difference_hessian[:, column] = (plus[1] - minus[1]) / (2.0 * step)

    np.testing.assert_allclose(
        actual_gradient,
        finite_difference_gradient,
        rtol=3.0e-7,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        actual_hessian,
        finite_difference_hessian,
        rtol=3.0e-6,
        atol=3.0e-6,
    )
