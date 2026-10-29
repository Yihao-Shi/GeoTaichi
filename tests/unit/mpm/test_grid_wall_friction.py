"""Coulomb impulse on stationary axis-aligned walls, after the free grid step."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.structs.GridNode import ContactNodes, ContactNodes2D, Nodes, Nodes2D

pytestmark = [pytest.mark.isolated_dimension(2), pytest.mark.cpu, pytest.mark.contact]


@pytest.mark.parametrize("node_type", [Nodes2D, ContactNodes2D])
def test_wall_impulse_sticking_sliding_and_separation(taichi_runtime, node_type):
    # [old vx, old vy, free ax, free ay, mu, normal axis, outward sign]
    samples = np.array(
        [
            [2, 0, 0, -10, 0.3, 1, -1],
            [0.1, 0, 0, -10, 0.3, 1, -1],
            [0, 0, 0, -10, 0.3, 1, -1],
            [2, 1, 0, 0, 0.3, 1, -1],
            [2, 0, 0, -10, 0, 1, -1],
            [0, -2, 10, 0, 0.3, 0, 1],
            [0, 0, 2, -10, 0.3, 1, -1],
            [0, 0, 5, -10, 0.3, 1, -1],
        ],
        dtype=np.float32 if ti.lang.impl.current_cfg().default_fp == ti.f32 else np.float64,
    )
    node = node_type.field(shape=len(samples))
    dt = ti.field(float, shape=())
    dt[None] = 0.1

    @ti.kernel
    def apply(data: ti.types.ndarray()):
        for i in range(data.shape[0]):
            node[i].momentum = ti.Vector([data[i, 0], data[i, 1]]) + dt[None] * ti.Vector([data[i, 2], data[i, 3]])
            node[i].force = ti.Vector([data[i, 2], data[i, 3]])
            node[i].friction_constraint(data[i, 4], int(data[i, 5]), int(data[i, 6]), dt)

    apply(samples)
    expected = np.array([[1.7, 0], [0, 0], [0, 0], [2, 1], [2, 0], [0, -1.7], [0, 0], [0.2, 0]])
    np.testing.assert_allclose(node.momentum.to_numpy(), expected, atol=2e-7)
    np.testing.assert_allclose(node.force.to_numpy(), (expected - samples[:, :2]) / 0.1, atol=2e-6)


def test_wall_impulse_uses_euclidean_tangent_in_3d(taichi_runtime):
    @ti.kernel
    def apply(node: ti.template(), dt: ti.template()):
        node[0].momentum = ti.Vector([3.0, -1.0, 4.0])
        node[0].force = ti.Vector([0.0, -10.0, 0.0])
        node[0].friction_constraint(0.3, 1, -1, dt)

    dt = ti.field(float, shape=())
    dt[None] = 0.1
    for node_type in [Nodes, ContactNodes]:
        node = node_type.field(shape=1)
        apply(node, dt)
        np.testing.assert_allclose(node.momentum.to_numpy()[0], [2.82, 0, 3.76], atol=3e-7)
        np.testing.assert_allclose(node.force.to_numpy()[0], [-1.8, 0, -2.4], atol=3e-6)
