"""MAC component loops retain affine fields and their full velocity gradients."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    _sample_mac_velocity3d,
    _sample_mac_velocity_gradient3d,
)


@ti.kernel
def sample(
    shape: ti.template(),
    width: ti.template(),
    points: ti.types.ndarray(),
    vx: ti.template(),
    vy: ti.template(),
    vz: ti.template(),
    output: ti.types.ndarray(),
):
    for i in range(points.shape[0]):
        point = ti.Vector([points[i, d] for d in ti.static(range(3))])
        spacing = ti.Vector([0.1, 0.12, 0.08])
        size = ti.Vector([0.02, 0.024, 0.016])
        if ti.static(shape == 2):
            size = 0.5 * spacing
        elif ti.static(shape == 3):
            size = spacing
        velocity = _sample_mac_velocity3d(point, size, shape, width, spacing, vx, vy, vz)
        gradient = _sample_mac_velocity_gradient3d(point, size, shape, width, spacing, vx, vy, vz, velocity)
        for d in ti.static(range(3)):
            output[i, d] = velocity[d]
            for axis in ti.static(range(3)):
                output[i, 3 + 3 * axis + d] = gradient[axis, d]


@pytest.mark.parametrize("shape,width", [(0, 2), (1, 3), (2, 3), (3, 4)])
def test_affine_and_constant_mac_components(taichi_runtime, shape, width):
    spacing = np.array([0.1, 0.12, 0.08])
    matrix = np.array([[0.3, -0.2, 0.7], [0.1, 0.8, -0.3], [-0.6, 0.4, 0.2]])
    offset = np.array([0.4, -0.1, 0.2])
    fields = []
    for axis in range(3):
        grid_shape = tuple(16 + int(d == axis) for d in range(3))
        face_offset = np.full(3, 0.5)
        face_offset[axis] = 0.0
        positions = (np.moveaxis(np.indices(grid_shape), 0, -1) + face_offset) * spacing
        field = ti.field(ti.f64, shape=grid_shape)
        field.from_numpy(positions @ matrix[axis] + offset[axis])
        fields.append(field)
    points = np.random.default_rng(73).uniform(7.0, 9.0, (20, 3)) * spacing
    result = np.zeros((len(points), 12))
    sample(shape, width, points, *fields, result)
    np.testing.assert_allclose(result[:, :3], points @ matrix.T + offset, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result[:, 3:].reshape(-1, 3, 3), np.broadcast_to(matrix, (len(points), 3, 3)), rtol=1e-11, atol=1e-11
    )
    points = np.array([[0, 0, 0], [1.6, 1.92, 1.28], [0.001, 0.96, 1.279], [-2, -2, -2]])
    for field, value in zip(fields, offset):
        field.fill(value)
    result = np.zeros((len(points), 12))
    sample(shape, width, points, *fields, result)
    np.testing.assert_allclose(result[:3, :3], np.broadcast_to(offset, (3, 3)), atol=1e-12)
    np.testing.assert_allclose(result[:, 3:], 0.0, atol=1e-11)
    np.testing.assert_array_equal(result[3], 0.0)
