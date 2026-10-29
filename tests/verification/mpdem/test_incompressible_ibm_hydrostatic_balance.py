"""IBM must preserve hydrostatic rest and neutral-body buoyancy."""

import numpy as np
import pytest

from src.mpm.engines.IncompressibleEngine import IncompressibleEngine
from src.mpm.engines.EngineKernel import kernel_correct_velocity_coupled_3d
from tests.verification.mpdem.test_incompressible_rigid_exchange import rigid_scene
from tests.verification.mpdem.test_incompressible_affine_exchange import Fluid

pytestmark = [pytest.mark.verification, pytest.mark.mpdem, pytest.mark.cpu]


@pytest.mark.parametrize("fraction", [0.25, 0.5, 1.0])
def test_ibm_preserves_hydrostatic_rest_and_buoyancy(taichi_runtime, fraction):
    coupling, body, sims, scene = rigid_scene(1.0)
    coupling.solid_fraction[2, 2, 2] = fraction
    sims.dt[None] = 5e-5
    gravity = -9.81
    cell = scene.element.cell
    pressure = np.zeros(cell.pressure.shape)
    for index in np.ndindex(pressure.shape):
        pressure[index] = gravity * (index[2] - 1 + 0.5)
    cell.pressure.from_numpy(pressure)
    scene.node.velocity[2].fill(gravity * sims.dt[None])
    engine = coupling.mengine
    engine.has_ibm_source = lambda: True

    # Apply the production predictor, then the exact hydrostatic pressure.
    # This stationary manufactured solution needs no pressure iteration.
    IncompressibleEngine.apply_ibm_source(engine, sims, scene)
    dummy = coupling.solid_fraction
    kernel_correct_velocity_coupled_3d(
        1,
        scene.element.cnum,
        scene.element.grid_size,
        sims.dt,
        Fluid(),
        cell.pressure,
        cell.surface_tension,
        cell.type,
        cell.fluid_sdf,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        coupling.solid_fraction,
        coupling.solid_density,
        2,
        False,
        False,
        scene.node,
    )
    coupling.accumulate_pressure_force(sims, scene)

    np.testing.assert_allclose([scene.node.velocity[2][2, 2, z] for z in (2, 3)], 0.0, atol=1e-15)
    np.testing.assert_allclose(coupling.ibm_force_cell[2, 2, 2].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(body.contact_force.to_numpy()[0], [0.0, 0.0, -fraction * gravity], atol=1e-12)
