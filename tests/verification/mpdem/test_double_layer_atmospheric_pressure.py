"""Air ghost pressure is an operator extension, not pressure acting on dry soil."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    AIR_CELL,
    FLUID_CELL,
    kernel_project_double_layer_pressure_to_solid_nodes2d,
    kernel_project_double_layer_pressure_to_solid_nodes3d,
)
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize("shape_type", [2, 3])
@pytest.mark.parametrize("wet_pressure", [500.0, -500.0])
def test_solid_pressure_uses_physical_atmosphere(taichi_runtime, dimension, shape_type, wet_pressure):
    size, dx, surface = 12, 0.1, 4
    shape = (size - 1,) * dimension
    height = np.indices(shape)[-1]
    kinds = np.where(height < surface, FLUID_CELL, AIR_CELL).astype(np.int32)
    kind = ti.field(ti.i32, shape=shape)
    pressure = ti.field(ti.f64, shape=shape)
    sdf = ti.field(ti.f64, shape=shape)
    kind.from_numpy(kinds)
    pressure.from_numpy(np.where(kinds == FLUID_CELL, wet_pressure, 0.0))
    sdf.from_numpy((height + 0.5 - surface) * dx)
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
    project(
        1e-12,
        ti.Vector([size] * dimension),
        ti.Vector([dx] * dimension),
        shape_type,
        3 if shape_type == 2 else 4,
        kind,
        pressure,
        sdf,
        nodes,
        lengths,
    )
    transverse = sum(6 * size**d for d in range(dimension - 1))
    values = np.array([float(nodes[transverse + h * size ** (dimension - 1), 0].pressure) for h in (2, 4, 5, 7)])
    assert np.isfinite(values).all()
    normalized = values / wet_pressure
    assert np.all(normalized >= -1e-12) and np.all(normalized <= 1.0 + 1e-12)
    # Full wet support reproduces constants; symmetric wet/dry support averages them.
    np.testing.assert_allclose(values[:2], [wet_pressure, 0.5 * wet_pressure], rtol=0, atol=1e-10)
    assert abs(values[-1]) < 1e-10
