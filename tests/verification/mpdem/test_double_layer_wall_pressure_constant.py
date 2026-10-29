"""A wall must not turn a constant pore pressure into a solid pressure gradient."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    kernel_extend_double_layer_wall_pressure,
    kernel_project_double_layer_pressure_to_solid_nodes2d,
    kernel_project_double_layer_pressure_to_solid_nodes3d,
)
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D


@pytest.mark.parametrize("shape_type", [2, 3])
@pytest.mark.parametrize("corner", [False, True])
@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("dimension", [2, 3])
def test_wall_pressure_reproduces_constant_and_linear_fields(taichi_runtime, dimension, linear, corner, shape_type):
    # Keep the full spline support inside the mesh while testing internal walls.
    dx, size, wall_index = 0.1, 12, 4
    shape = (size - 1,) * dimension
    kinds = np.ones(shape, dtype=np.int32)
    kinds[:wall_index] = 2
    if corner:
        for axis in range(1, dimension):
            wall = [slice(None)] * dimension
            wall[axis] = slice(0, wall_index)
            kinds[tuple(wall)] = 2
    air = [slice(None)] * dimension
    air[-1] = -1
    kinds[tuple(air)] = 0
    kind = ti.field(ti.i32, shape=shape)
    kind.from_numpy(kinds)
    pressure = ti.field(ti.f64, shape=shape)
    gradient = np.arange(1, dimension + 1) * 125.0 if linear else np.zeros(dimension)
    values = 1000.0 + sum(gradient[d] * (np.indices(shape)[d] + 0.5) * dx for d in range(dimension))
    values[kinds != 1] = 0.0
    pressure.from_numpy(values)
    sdf = ti.field(ti.f64, shape=shape)
    sdf.fill(-dx)
    cls = NodeTwoPhase2D if dimension == 2 else NodeTwoPhase
    nodes = cls.field(shape=(size**dimension, 1))
    nodes.ms.fill(1.0)
    lengths = ti.Vector.field(dimension, ti.f64, shape=1)
    lengths.fill(dx / 2 if shape_type == 2 else dx)
    project = (
        kernel_project_double_layer_pressure_to_solid_nodes2d
        if dimension == 2
        else kernel_project_double_layer_pressure_to_solid_nodes3d
    )
    support = 3 if shape_type == 2 else 4
    kernel_extend_double_layer_wall_pressure(kind, pressure, support - 1)
    np.testing.assert_array_equal(pressure.to_numpy()[kinds != 2], values[kinds != 2])
    project(
        1e-12,
        ti.Vector([size] * dimension),
        ti.Vector([dx] * dimension),
        shape_type,
        support,
        kind,
        pressure,
        sdf,
        nodes,
        lengths,
    )
    transverse = wall_index - 1 if corner else wall_index + 1
    offset = sum(transverse * size**d for d in range(1, dimension))
    coordinates = np.array([wall_index - 1, wall_index, wall_index + 1])
    values = np.array([float(nodes[x + offset, 0].pressure) for x in coordinates])
    expected = 1000.0 + gradient[0] * coordinates * dx + sum(gradient[1:]) * transverse * dx
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-10)
    if shape_type == 2:
        # Derivatives of the Q2 nodal basis at x=wall+.25*dx.
        actual_gradient = np.array([-0.25, -0.5, 0.75]) @ values / dx
        assert abs(actual_gradient - gradient[0]) < 1e-10, (values, actual_gradient)
