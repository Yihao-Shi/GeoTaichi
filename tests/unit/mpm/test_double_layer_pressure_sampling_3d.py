"""Runtime 3D pressure stencils preserve the original trilinear/GFM weights."""

from itertools import product

import numpy as np
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    _sample_cell_scalar3d,
    _sample_cell_pressure3d,
    _sample_cell_pressure_gfm3d,
)


@ti.kernel
def sample(
    positions: ti.types.ndarray(),
    kinds: ti.template(),
    pressure: ti.template(),
    sdf: ti.template(),
    values: ti.types.ndarray(),
):
    for i in range(positions.shape[0]):
        point = ti.Vector([positions[i, d] for d in ti.static(range(3))])
        spacing = ti.Vector([0.1, 0.1, 0.1])
        values[i, 0] = _sample_cell_scalar3d(point, spacing, kinds, pressure)
        values[i, 1] = _sample_cell_pressure3d(point, spacing, kinds, pressure)
        values[i, 2] = _sample_cell_pressure_gfm3d(point, spacing, kinds, pressure, sdf)


def test_runtime_trilinear_and_ghost_pressure_match_numpy(taichi_runtime):
    shape = (6, 6, 6)
    kinds = np.ones(shape, dtype=np.int32)
    kinds[0, :, :] = 2
    kinds[:, :, 4:] = 0
    rng = np.random.default_rng(127)
    pressure = rng.uniform(-2.0, 3.0, size=shape)
    sdf = np.broadcast_to((np.arange(6) - 3.5)[None, None, :] * 0.1, shape).copy()
    fields = [ti.field(dtype, shape=shape) for dtype in (ti.i32, ti.f64, ti.f64)]
    for field, data in zip(fields, (kinds, pressure, sdf)):
        field.from_numpy(data)
    positions = rng.uniform(0.0, 0.6, size=(64, 3))
    values = np.zeros((len(positions), 3))
    sample(positions, *fields, values)
    expected = np.zeros_like(values)
    for row, point in enumerate(positions):
        base = np.floor(point / 0.1 - 0.5).astype(int)
        totals, weights = np.zeros(3), np.zeros(3)
        for offset in product((0, 1), repeat=3):
            cell = tuple(base + offset)
            if not all(0 <= cell[d] < shape[d] for d in range(3)):
                continue
            weight = np.prod(np.maximum(0.0, 1.0 - np.abs((point - (np.asarray(cell) + 0.5) * 0.1) / 0.1)))
            if kinds[cell] == 1:
                totals += weight * pressure[cell]
                weights += weight
            elif kinds[cell] == 0:
                weights[1] += weight
                ghosts = []
                for axis, side in product(range(3), (1, -1)):
                    neighbor = list(cell)
                    neighbor[axis] += side
                    neighbor = tuple(neighbor)
                    if all(0 <= neighbor[d] < shape[d] for d in range(3)) and kinds[neighbor] == 1:
                        theta = abs(sdf[neighbor]) / (abs(sdf[neighbor]) + abs(sdf[cell]))
                        theta = max(0.01, min(1.0, theta))
                        ghosts.append(-(1.0 - theta) / theta * pressure[neighbor])
                if ghosts:
                    totals[2] += weight * np.mean(ghosts)
                    weights[2] += weight
        expected[row] = np.divide(totals, weights, out=np.zeros(3), where=weights > 1.0e-12)
    np.testing.assert_allclose(values, expected, rtol=1.0e-12, atol=1.0e-12)
