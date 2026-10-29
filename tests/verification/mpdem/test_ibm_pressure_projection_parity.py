"""IBM's pressure predictor must match the applied variable-density projection."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

import src.utils.GlobalVariable as GV
from src.mpm.engines.EngineKernel import (
    ibm_cell_pressure_acceleration,
    kernel_correct_velocity_coupled_3d,
    mac_cell_center_velocity,
)
from src.mpm.structs.StaggeredGrid import StaggeredGrid
from tests.verification.mpdem.test_incompressible_affine_exchange import Fluid

pytestmark = [pytest.mark.verification, pytest.mark.mpdem, pytest.mark.cpu]


@ti.kernel
def compare_projection(
    h: ti.types.vector(3, float),
    dt: float,
    fraction: ti.template(),
    density: ti.template(),
    pressure: ti.template(),
    tension: ti.template(),
    kind: ti.template(),
    sdf: ti.template(),
    theta: ti.template(),
    grid: ti.template(),
    error: ti.template(),
):
    for I in ti.grouped(ti.ndrange(4, 4, 4)):
        if kind[I] == 1:
            predicted = ibm_cell_pressure_acceleration(
                I,
                ti.Vector([4, 4, 4]),
                h,
                1.0,
                0.0,
                fraction,
                density,
                pressure,
                tension,
                kind,
                sdf,
                theta,
            )
            error[I] = predicted - mac_cell_center_velocity(I, grid) / dt


@pytest.mark.parametrize("free_surface_theta", [False, True])
@pytest.mark.parametrize("solid_density", [0.8, 1.14, 2.5])
def test_ibm_predictor_matches_applied_pressure_at_interfaces(taichi_runtime, free_surface_theta, solid_density):
    GV.DIMENSION = 3
    shape = (6, 6, 6)
    grid = StaggeredGrid(SimpleNamespace(dimension=3, coupling=False), list(shape), 1)
    fields = [ti.field(float, shape=shape, offset=(-1, -1, -1)) for _ in range(6)]
    fraction, density, pressure, tension, sdf, dummy = fields
    kind = ti.field(int, shape=shape, offset=(-1, -1, -1))
    dt = ti.field(float, shape=())
    dt[None] = 0.001
    h = ti.Vector([0.8, 1.1, 1.7])
    rng = np.random.default_rng(27)
    fraction.from_numpy(rng.uniform(0.0, 1.0, shape))
    density.fill(solid_density)
    pressure.from_numpy(rng.normal(size=shape))
    tension.fill(0.3)
    sdf.fill(-0.2)
    kind.fill(2)
    for index in np.ndindex(4, 4, 4):
        kind[index] = 1
    kind[1, 1, 2], sdf[1, 1, 2], fraction[1, 1, 2] = 0, 0.7, 0.0
    kind[2, 2, 1], fraction[2, 2, 1] = 2, 0.0
    kernel_correct_velocity_coupled_3d(
        1,
        ti.Vector(list(shape)),
        h,
        dt,
        Fluid(),
        pressure,
        tension,
        kind,
        sdf,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        fraction,
        density,
        2,
        False,
        free_surface_theta,
        grid,
    )
    error = ti.Vector.field(3, float, shape=(4, 4, 4))
    compare_projection(h, dt[None], fraction, density, pressure, tension, kind, sdf, free_surface_theta, grid, error)
    np.testing.assert_allclose(error.to_numpy(), 0.0, atol=1e-12)
