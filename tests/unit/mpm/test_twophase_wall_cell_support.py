import numpy as np
import pytest
import taichi as ti
from geotaichi import init

from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    AIR_CELL,
    FLUID_CELL,
    SOLID_CELL,
    kernel_extend_double_layer_cell_support_at_solid_walls,
)


@pytest.mark.parametrize("mac_shape_type", [0, 2])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("corner", [False, True])
def test_wall_images_restore_full_cell_capacity_without_changing_particles(dimension, corner, mac_shape_type):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    dx = 0.05
    shape = (4,) * dimension
    cell_volume = dx**dimension
    center = np.ones(dimension, dtype=int)
    types = np.full(shape, FLUID_CELL, dtype=np.int32)
    axes = list(range(dimension)) if corner else [dimension - 1]
    counts = [4 if d in axes else 8 for d in range(dimension)]
    positions = np.array([(np.array(bits) * 0.5 + 0.25) * dx for bits in np.ndindex(*counts)])
    particle = ti.Struct.field(
        {
            "x": ti.types.vector(dimension, float),
            "mf": float,
            "vol": float,
            "porosity": float,
            "phase": int,
            "active": int,
            "materialID": int,
        },
        shape=len(positions),
    )
    for axis in axes:
        index = [slice(None)] * dimension
        index[axis] = slice(2, None)
        types[tuple(index)] = SOLID_CELL
    types[tuple(center)] = AIR_CELL
    cell_type = ti.field(int, shape=shape)
    mass = ti.field(float, shape=shape)
    density = ti.field(float, shape=shape)
    porosity = ti.field(float, shape=shape)
    cell_type.from_numpy(types)
    density.fill(1000)
    # A half-cell shift truncates each wall-normal linear scatter by 1/8.
    supported = (0.84375 if mac_shape_type == 2 else 0.875) ** len(axes)
    porosity.fill(1)
    porosity[tuple(center)] = 1 - 0.6 * supported
    mass[tuple(center)] = 0.49 * 1000 * float(porosity[tuple(center)]) * cell_volume
    particle.x.from_numpy(positions)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(1)
    particle.porosity.fill(0.4)
    particle.vol.fill(cell_volume / (2**dimension))
    particle.mf.fill(0)
    before = particle.to_numpy()
    kernel_extend_double_layer_cell_support_at_solid_walls(
        len(positions),
        ti.Vector([dx] * dimension),
        particle,
        mass,
        density,
        porosity,
        cell_type,
        mac_shape_type,
    )
    assert float(porosity[tuple(center)]) == pytest.approx(0.4)
    assert cell_type[tuple(center)] == FLUID_CELL
    for key, value in before.items():
        np.testing.assert_array_equal(particle.to_numpy()[key], value)
    np.testing.assert_array_equal(cell_type.to_numpy()[types == SOLID_CELL], SOLID_CELL)


@pytest.mark.parametrize("mac_shape_type", [0, 2])
@pytest.mark.parametrize("dimension", [2, 3])
def test_wall_images_preserve_empty_gaps_and_do_not_cross_thin_wall(dimension, mac_shape_type):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (5,) * dimension
    particle = ti.Struct.field(
        {
            "x": ti.types.vector(dimension, float),
            "mf": float,
            "vol": float,
            "porosity": float,
            "phase": int,
            "active": int,
            "materialID": int,
        },
        shape=1,
    )
    types = np.full(shape, AIR_CELL, dtype=np.int32)
    wall = [slice(None)] * dimension
    wall[0] = 2
    types[tuple(wall)] = SOLID_CELL
    source = (1,) * dimension
    types[source] = FLUID_CELL
    cell_type = ti.field(int, shape=shape)
    mass = ti.field(float, shape=shape)
    density = ti.field(float, shape=shape)
    porosity = ti.field(float, shape=shape)
    cell_type.from_numpy(types)
    density.fill(1000)
    porosity.fill(1)
    particle.x.from_numpy(np.full((1, dimension), 1.25))
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(2)
    particle.mf.fill(1)
    particle.vol.fill(0.001)
    particle.porosity.fill(1)
    kernel_extend_double_layer_cell_support_at_solid_walls(
        1,
        ti.Vector([1.0] * dimension),
        particle,
        mass,
        density,
        porosity,
        cell_type,
        mac_shape_type,
    )
    # The reflected point has support on the far side, which must be discarded.
    assert np.count_nonzero(mass.to_numpy()[3:]) == 0
    np.testing.assert_array_equal(cell_type.to_numpy(), types)
