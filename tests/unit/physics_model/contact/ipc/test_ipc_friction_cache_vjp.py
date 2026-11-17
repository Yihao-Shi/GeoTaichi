"""Differentiate lagged contact data against an independent closest-point oracle."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance_grad_hess,
    point_triangle_distance_grad_hess,
)
from src.physics_model.contact_model.ipc.Utilities import (
    ipc_barrier_force_magnitude,
    ipc_barrier_force_magnitude_gradient,
    ipc_lagged_friction_cache_vjp,
)

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]


@pytest.fixture(scope="module", autouse=True)
def runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=1)
    yield
    ti.reset()


@ti.kernel
def _vjp(
    cache: ti.types.vector(12, ti.f64),
    current: ti.types.vector(12, ti.f64),
    reference: ti.types.vector(12, ti.f64),
    adjoint: ti.types.vector(12, ti.f64),
    edge: ti.template(),
) -> ti.types.vector(12, ti.f64):
    p = ti.Matrix.zero(ti.f64, 4, 3)
    for i, j in ti.static(ti.ndrange(4, 3)):
        p[i, j] = cache[3 * i + j]
    d2 = 0.0
    g = ti.Vector.zero(ti.f64, 12)
    h = ti.Matrix.zero(ti.f64, 12, 12)
    if ti.static(edge):
        d2, g, h, _ = edge_edge_distance_grad_hess(p[0, :], p[1, :], p[2, :], p[3, :])
    else:
        d2, g, h, _ = point_triangle_distance_grad_hess(p[0, :], p[1, :], p[2, :], p[3, :])
    coefficient = 0.3 * ipc_barrier_force_magnitude(d2, 0.5, 0.0, 3.0, 0)
    derivative = 0.3 * ipc_barrier_force_magnitude_gradient(d2, g, 0.5, 0.0, 3.0, 0)
    return ipc_lagged_friction_cache_vjp(
        current, reference, adjoint, d2, g, h, coefficient, derivative, 0.1, 0.08, edge
    )


def _closest_weights(p, edge):
    candidates = []
    if edge:
        a, b = p[1] - p[0], p[3] - p[2]
        matrix = np.column_stack((a, -b))
        s, t = np.linalg.lstsq(matrix, p[2] - p[0], rcond=None)[0]
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(np.array([1 - s, s, t - 1, -t]))
        for vertex, first, second, sign in ((0, 2, 3, 1), (1, 2, 3, 1), (2, 0, 1, -1), (3, 0, 1, -1)):
            direction = p[second] - p[first]
            t = np.clip(np.dot(p[vertex] - p[first], direction) / np.dot(direction, direction), 0, 1)
            weights = np.zeros(4)
            weights[vertex], weights[first], weights[second] = sign, -sign * (1 - t), -sign * t
            candidates.append(weights)
    else:
        matrix = np.column_stack((p[2] - p[1], p[3] - p[1]))
        s, t = np.linalg.lstsq(matrix, p[0] - p[1], rcond=None)[0]
        if s >= 0 and t >= 0 and s + t <= 1:
            candidates.append(np.array([1, s + t - 1, -s, -t]))
        for first, second in ((1, 2), (2, 3), (3, 1)):
            direction = p[second] - p[first]
            t = np.clip(np.dot(p[0] - p[first], direction) / np.dot(direction, direction), 0, 1)
            weights = np.zeros(4)
            weights[0], weights[first], weights[second] = 1, t - 1, -t
            candidates.append(weights)
    return min(candidates, key=lambda weights: np.linalg.norm(weights @ p))


def _contraction(cache, current, reference, adjoint, edge):
    weights = _closest_weights(cache, edge)
    delta = weights @ cache
    distance = np.linalg.norm(delta)
    normal = delta / distance
    velocity = weights @ (current - reference) / 0.1
    velocity -= normal * np.dot(normal, velocity)
    speed = np.linalg.norm(velocity)
    profile = (2.0 - speed / 0.08) / 0.08 if speed < 0.08 else 1.0 / speed
    s, active = distance**2, 0.5**2
    gradient = -3.0 * (2.0 * (s - active) * np.log(s / active) + (s - active) ** 2 / s)
    coefficient = 0.3 * (-2.0 * distance * gradient) if s < active else 0.0
    return coefficient * profile * np.dot(weights @ adjoint, velocity)


def test_contact_cache_vjp_includes_weights_normal_and_load():
    contacts = (
        (False, [[0.3, 0.25, 0.2], [0, 0, 0], [1, 0.05, 0], [0.1, 1, 0.02]]),
        (True, [[0, 0, 0], [1, 0.1, 0], [0.4, -0.4, 0.2], [0.5, 0.7, 0.25]]),
        (False, [[0.4, -0.1, 0.2], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        (False, [[-0.1, -0.15, 0.2], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        (True, [[0, 0, 0], [1, 0, 0], [1.1, -0.4, 0.2], [1.1, 0.7, 0.2]]),
        (True, [[0, 0, 0], [1, 0, 0], [1.1, 0.1, 0.2], [1.2, 0.7, 0.2]]),
        (False, [[0.3, 0.25, 0.6], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    )
    rng = np.random.default_rng(41)
    adjoint = rng.normal(size=(4, 3))
    direction = rng.normal(size=(4, 3))
    for edge, points in contacts:
        cache = np.asarray(points, dtype=np.float64)
        for slip in (0.0, 0.002, 0.3):
            reference = cache.copy()
            current = reference + slip * direction
            analytic = np.array(_vjp(cache.ravel(), current.ravel(), reference.ravel(), adjoint.ravel(), edge))
            for step in (2e-6, 4e-7):
                numerical = np.zeros(12)
                for j in range(12):
                    perturbation = np.zeros(12)
                    perturbation[j] = step
                    plus, minus = cache + perturbation.reshape(4, 3), cache - perturbation.reshape(4, 3)
                    numerical[j] = (
                        _contraction(plus, current, reference, adjoint, edge)
                        - _contraction(minus, current, reference, adjoint, edge)
                    ) / (2 * step)
                np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-8)
