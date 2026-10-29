"""Rigid IBM feedback must use the same increments as the fluid grid."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpdem.fluid_dynamics.IncompressibleCoupling import IncompressibleLSDEMCoupling
from src.mpm.engines.EngineKernel import kernel_correct_velocity_coupled_3d
from tests.verification.mpdem.test_incompressible_affine_exchange import build, Fluid
from tests.verification.mpdem.test_incompressible_coupling_kernels import AnalyticSphereSDF

pytestmark = [pytest.mark.verification, pytest.mark.mpdem, pytest.mark.cpu]


@ti.dataclass
class RigidFixture:
    x: ti.types.vector(3, ti.f64)
    q: ti.types.vector(4, ti.f64)
    contact_force: ti.types.vector(3, ti.f64)
    contact_torque: ti.types.vector(3, ti.f64)
    active: ti.u8
    radius: ti.f64

    @ti.func
    def _get_position(self):
        return self.x

    @ti.func
    def _get_radius(self):
        return self.radius


def rigid_scene(density):
    unused, cube, sims, scene = build(active=5)
    body = RigidFixture.field(shape=1)
    body[0].x, body[0].q = [2.5] * 3, [1.0, 0.0, 0.0, 0.0]
    body[0].active, body[0].radius = 1, 2.0
    box = AnalyticSphereSDF.field(shape=1)
    box[0].radius = 2.0
    dem = SimpleNamespace(
        rigidNum=np.array([1]), rigid=body, particle=body, box=box, rigid_grid=ti.field(float, shape=1)
    )
    scene.element.igrid_size = ti.Vector([1.0, 1.0, 1.0])
    coupling = IncompressibleLSDEMCoupling(sims, None, scene, dem, unused.mengine)
    coupling.attach()
    coupling.solid_fraction[2, 2, 2] = 0.5
    coupling.solid_fraction_sum[2, 2, 2] = 1.0  # Analytic fixture fully occupies this cell.
    coupling.solid_density[2, 2, 2] = density
    return coupling, body, sims, scene


@pytest.mark.parametrize("density", [0.8, 1.14, 2.5])
def test_rigid_pressure_feedback_conserves_combined_impulse(taichi_runtime, density):
    coupling, body, sims, scene = rigid_scene(density)
    cell = scene.element.cell
    cell.pressure[3, 2, 2] = 2.0
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
    fluid = np.zeros(3)
    for index in np.ndindex(5, 5, 5):
        for d in range(3):
            right = list(index)
            right[d] += 1
            fluid[d] += (
                (1.0 - coupling.solid_fraction[index])
                * 0.5
                * (scene.node.velocity[d][index] + scene.node.velocity[d][tuple(right)])
            )
    rigid = body.contact_force.to_numpy()[0] * sims.dt[None]
    np.testing.assert_allclose(fluid + rigid, 0.0, atol=1e-12)


def test_rigid_viscous_feedback_uses_applied_grid_increment(taichi_runtime):
    coupling, body, sims, scene = rigid_scene(1.14)
    coupling.mengine.coupled_viscous_delta[0].fill(0.2)
    coupling.accumulate_pressure_force(sims, scene)
    np.testing.assert_allclose(
        body.contact_force.to_numpy()[0] * sims.dt[None], [0.5 * 1.14 * 0.2, 0.0, 0.0], atol=1e-12
    )
