"""Saturated porous boundary cells must not become atmospheric pressure holes."""

import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    kernel_classify_double_layer_fluid_cells2d,
    kernel_classify_double_layer_fluid_cells3d,
)
from src.utils.TypeDefination import vec2f, vec3f


@ti.dataclass
class FluidPoint2D:
    x: vec2f
    active: ti.u8
    phase: ti.u8
    materialID: ti.u8


@ti.dataclass
class FluidPoint3D:
    x: vec3f
    active: ti.u8
    phase: ti.u8
    materialID: ti.u8


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("porous", [False, True])
def test_low_fill_cell_adjacent_to_resolved_water_keeps_pore_pressure(taichi_runtime, dimension, porous):
    shape = (6,) * dimension
    point = (FluidPoint2D if dimension == 2 else FluidPoint3D).field(shape=1)
    point[0].x = [0.025] * dimension
    point[0].active, point[0].phase, point[0].materialID = 1, 2, 1
    mass, density, porosity = [ti.field(ti.f64, shape=shape) for _ in range(3)]
    kind = ti.field(ti.i32, shape=shape)
    density.fill(1000.0)
    porosity.fill(1.0)
    cell = (2,) * dimension
    neighbor = (3,) + (2,) * (dimension - 1)
    porosity[cell] = 0.7 if porous else 1.0
    mass[cell] = 0.04 * 1000.0 * 0.01**dimension
    mass[neighbor] = 0.9 * 1000.0 * 0.01**dimension
    classify = (
        kernel_classify_double_layer_fluid_cells2d if dimension == 2 else kernel_classify_double_layer_fluid_cells3d
    )
    classify(1, ti.Vector([0.01] * dimension), point, mass, density, porosity, kind)
    assert kind[neighbor] == 1
    # Pure-water subcell tails remain air; water inside a porous skeleton carries pressure.
    assert kind[cell] == (1 if porous else 0)
