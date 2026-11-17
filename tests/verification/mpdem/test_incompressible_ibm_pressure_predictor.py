from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

import src.utils.GlobalVariable as GV
from src.mpm.engines.EngineKernel import kernel_apply_incompressible_ibm_mac_source
from src.mpm.structs.StaggeredGrid import StaggeredGrid
from tests.verification.mpdem.test_incompressible_affine_exchange import Fluid

pytestmark = [pytest.mark.verification, pytest.mark.mpdem, pytest.mark.cpu]


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("pressure_gauge", [0.0, 1000.0])
def test_pressure_balances_ibm_slip_without_applying_pressure_twice(taichi_runtime, dimension, pressure_gauge):
    GV.DIMENSION = dimension
    shape, offset, center = (5,) * dimension, (-1,) * dimension, (1,) * dimension
    grid = StaggeredGrid(SimpleNamespace(dimension=dimension, coupling=False), list(shape), 1)
    fraction = ti.field(float, shape=shape, offset=offset)
    density = ti.field(float, shape=shape, offset=offset)
    pressure = ti.field(float, shape=shape, offset=offset)
    cell_type = ti.field(int, shape=shape, offset=offset)
    target = ti.Vector.field(dimension, float, shape=shape, offset=offset)
    force = ti.Vector.field(dimension, float, shape=shape, offset=offset)
    dt = ti.field(float, shape=())
    dt[None] = 5e-5
    gravity = -9.81
    fraction[center], density[center] = 0.5, 1.0
    cell_type.fill(1)
    values = np.zeros(shape)
    for index in np.ndindex(shape):
        values[index] = pressure_gauge + gravity * (index[-1] - 0.5)
    pressure.from_numpy(values)
    grid.velocity[-1].fill(gravity * dt[None])

    kernel_apply_incompressible_ibm_mac_source(
        1,
        ti.Vector(list(shape)),
        dt,
        Fluid(),
        cell_type,
        fraction,
        density,
        target,
        force,
        grid,
        ti.Vector([1.0] * dimension),
        pressure,
        pressure,
        pressure,
        False,
    )

    np.testing.assert_allclose(force.to_numpy(), 0.0, atol=1e-12)
    # The ordinary projection still applies absolute pressure after this stage.
    np.testing.assert_allclose(grid.velocity[-1].to_numpy(), gravity * dt[None], atol=1e-15)
