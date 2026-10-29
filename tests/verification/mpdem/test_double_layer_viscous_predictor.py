"""Every viscous stencil must read the same time level on the staggered grid."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import kernel_predict_double_layer2d, kernel_predict_double_layer3d
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("nonzero_wall_input", [False, True])
def test_viscous_predictor_uses_unmodified_neighbor_velocities(taichi_runtime, dimension, nonzero_wall_input):
    size, dx, timestep, viscosity, damping = 8, 0.125, 0.001, 0.2, 0.05
    node_type = NodeTwoPhase2D if dimension == 2 else NodeTwoPhase
    node = node_type.field(shape=((size + 1) ** dimension, 1))
    shapes = [tuple(size + (d == axis) for d in range(dimension)) for axis in range(dimension)]

    def field(shape, value=0.0):
        f = ti.field(ti.f64, shape=shape)
        f.fill(value)
        return f

    mass = [field(s, 1.0) for s in shapes]
    velocity, acceleration, solid_velocity = [[field(s) for s in shapes] for _ in range(3)]
    properties = [1.0, 1.0, 1.0, viscosity, 1.0, 1.0, 1.0, 1.0]
    face_properties = [[field(s, value) for s in shapes] for value in properties]
    node_properties = [field(node.shape, value) for value in properties[1:]]
    calLength = ti.Vector.field(dimension, ti.f64, shape=1)
    calLength.fill(0.5 * dx)
    dt = field((), timestep)
    rng = np.random.default_rng(53)
    expected = []
    for axis, s in enumerate(shapes):
        old = rng.normal(0.0, 1.0, s)
        wall = [slice(None)] * dimension
        for index in (0, -1):
            wall[axis] = index
            old[tuple(wall)] = 0.0
        incoming = old.copy()
        if nonzero_wall_input:
            for index in (0, -1):
                wall[axis] = index
                incoming[tuple(wall)] = 3.0
        velocity[axis].from_numpy(incoming)
        laplace = np.zeros(s)
        for direction in range(dimension):
            left, right = [slice(None)] * dimension, [slice(None)] * dimension
            left[direction], right[direction] = slice(None, -1), slice(1, None)
            difference = (old[tuple(right)] - old[tuple(left)]) / dx**2
            laplace[tuple(left)] += difference
            laplace[tuple(right)] -= difference
        acc = viscosity * laplace - damping * old
        inactive = (size // 2,) * dimension
        mass[axis][inactive] = 0.0
        acc[inactive] = 0.0
        for index in (0, -1):
            wall[axis] = index
            acc[tuple(wall)] = 0.0
        expected.append((old + timestep * acc, acc))
    predictor = kernel_predict_double_layer2d if dimension == 2 else kernel_predict_double_layer3d
    predictor(
        1e-12,
        ti.Vector([0.0] * 3),
        ti.Vector([dx] * dimension),
        dt,
        damping,
        node,
        *mass,
        *velocity,
        *acceleration,
        *solid_velocity,
        *(f for fields in face_properties for f in fields),
        *node_properties,
        0,
        2,
        calLength,
        None,
        None,
        None,
        tuple(field(shape) for shape in shapes),
    )
    for axis in range(dimension):
        np.testing.assert_allclose(velocity[axis].to_numpy(), expected[axis][0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(acceleration[axis].to_numpy(), expected[axis][1], rtol=1e-10, atol=1e-10)
