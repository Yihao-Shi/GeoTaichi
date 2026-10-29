"""A domain wall must not consume a fluid cell through coordinate roundoff."""

import math

import numpy as np
import pytest
import taichi as ti

from geotaichi import init


@pytest.mark.parametrize("dimension", [pytest.param(d, marks=pytest.mark.isolated_dimension(d)) for d in (2, 3)])
def test_upper_wall_preserves_fluid_cells_despite_float32_domain(dimension):
    init(dim=dimension, arch="cpu", default_fp="float64", cpu_max_num_threads=2, offline_cache=False, log=False)
    from src.mpm.engines.EngineKernel import mark_solid_cell_region

    spacing = float(np.float32(math.pi)) / 64
    types = ti.field(int, shape=(66,) * dimension, offset=(-1,) * dimension)
    lower = [math.pi] + [0.0] * (dimension - 1)
    upper = [math.pi + spacing] + [math.pi] * (dimension - 1)

    def mark():
        mark_solid_cell_region(
            1, ti.Vector([66] * dimension), ti.Vector([spacing] * dimension), ti.Vector(lower), ti.Vector(upper), types
        )

    mark()
    assert types[(63,) + (32,) * (dimension - 1)] == 0
    assert types[(64,) + (32,) * (dimension - 1)] == 2

    # A real overlap must still mark the intersected cell.
    lower[0] -= 0.02 * spacing
    mark()
    assert types[(63,) + (32,) * (dimension - 1)] == 2
