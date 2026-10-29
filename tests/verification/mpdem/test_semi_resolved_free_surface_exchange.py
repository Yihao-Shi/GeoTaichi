"""Water drag must exchange momentum with occupied MAC faces, including at entry."""

import numpy as np
import pytest
import taichi as ti

from tests.verification.mpdem.test_incompressible_coupling_kernels import (
    FluidProperties,
    SphereIndexFixture,
    make_mac_grid,
)
from src.mpdem.fluid_dynamics.DragForceModel import DragForce
from src.mpdem.fluid_dynamics.IncompressibleSemiResolvedKernel import (
    kernel_compute_incompressible_sphere_drag,
    kernel_apply_cell_drag_force_to_mac_velocity,
    kernel_accumulate_incompressible_sphere_pressure_force,
)

pytestmark = [pytest.mark.verification, pytest.mark.mpdem, pytest.mark.cpu]


@ti.dataclass
class SphereParticle:
    x: ti.types.vector(3, ti.f64)
    rad: ti.f64
    v: ti.types.vector(3, ti.f64)
    contact_force: ti.types.vector(3, ti.f64)


@pytest.mark.parametrize("wet_layers, face_mass", [(0, 2.0), (2, 2.0), (4, 2.0), (4, 1e-13)])
def test_drag_exchanges_only_with_present_water(taichi_runtime, wet_layers, face_mass):
    node = make_mac_grid([4, 4, 4])
    for d, field in enumerate(node.m):
        mass = np.zeros(field.shape)
        mass[:, :, : wet_layers + (d == 2 and wet_layers > 0)] = face_mass
        field.from_numpy(mass)
    particle = SphereParticle.field(shape=1)
    particle[0].x = [0.5] * 3
    particle[0].rad = 0.125
    particle[0].v = [1.0, -0.5, -2.0]
    sphere = SphereIndexFixture.field(shape=1)
    fraction = ti.field(float, shape=(4, 4, 4))
    fraction.fill(0.2)
    force = ti.Vector.field(3, float, shape=(4, 4, 4))
    velocity = ti.Vector.field(3, float, shape=1)
    velocity_weight = ti.Vector.field(3, float, shape=1)
    fluid_fraction = ti.field(float, shape=1)
    weights = ti.field(float, shape=1)
    dt = ti.field(float, shape=())
    dt[None] = 0.01
    kernel_compute_incompressible_sphere_drag(
        1,
        1e-12,
        6,
        3,
        ti.Vector([4] * 3),
        ti.Vector([0.25] * 3),
        ti.Vector([4.0] * 3),
        FluidProperties(),
        particle,
        sphere,
        node,
        fraction,
        force,
        velocity,
        velocity_weight,
        fluid_fraction,
        weights,
        DragForce({"DragForceModel": "SchillerNaumannModel"}),
    )
    kernel_apply_cell_drag_force_to_mac_velocity(
        1e-12,
        dt,
        ti.Vector([4] * 3),
        node,
        fraction,
        force,
    )
    fluid_impulse = np.array([np.sum(node.m[d].to_numpy() * 0.8 * node.velocity[d].to_numpy()) for d in range(3)])
    body_force = particle.contact_force.to_numpy()[0]
    np.testing.assert_allclose(fluid_impulse + 0.01 * body_force, 0.0, atol=1e-12)
    if wet_layers == 0 or face_mass < 1e-12:
        np.testing.assert_array_equal(body_force, np.zeros(3))
    else:
        assert np.linalg.norm(body_force) > 0.0
    if wet_layers == 4 and face_mass > 1e-12:
        slip = -np.array([1.0, -0.5, -2.0])
        reynolds = 0.8 * 2 * 1000 * 0.125 * np.linalg.norm(slip) / 1e-3
        kappa = 3.7 - 0.65 * np.exp(-0.5 * (1.5 - np.log10(reynolds)) ** 2)
        expected = 0.5 * np.pi * 0.44 * 1000 * 0.125**2 * 0.8 ** (2 - kappa) * np.linalg.norm(slip) * slip
        np.testing.assert_allclose(body_force, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("center_z, wet_fraction", [(0.375, 1.0), (0.75, 0.5), (1.0, 0.0)])
def test_hydrostatic_load_uses_wetted_kernel_volume(taichi_runtime, center_z, wet_fraction):
    # The sphere-volume kernel is symmetric about the interface at z=.75.
    # At z=1 it is dry although the wider background-pressure stencil sees water.
    shape, dx, radius, density, gravity = (12, 12, 12), 0.125, 0.0625, 1000.0, 9.8
    z = (np.indices(shape)[2] + 0.5) * dx
    pressure = ti.field(float, shape=shape)
    pressure.from_numpy(np.where(z < 0.75, density * gravity * (0.75 - z), 0.0))
    kind = ti.field(int, shape=shape)
    kind.from_numpy((z < 0.75).astype(np.int32))
    particle = SphereParticle.field(shape=1)
    particle[0].x = [0.75, 0.75, center_z]
    particle[0].rad = radius
    sphere = SphereIndexFixture.field(shape=1)
    kernel_accumulate_incompressible_sphere_pressure_force(
        1,
        6,
        3,
        ti.Vector(shape),
        ti.Vector([dx] * 3),
        ti.Vector([1 / dx] * 3),
        particle,
        sphere,
        pressure,
        kind,
        0.0,
    )
    expected = wet_fraction * density * gravity * 4 / 3 * np.pi * radius**3
    np.testing.assert_allclose(particle.contact_force.to_numpy()[0], [0, 0, expected], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("wet_layers", [2, 4])
@pytest.mark.parametrize("empty_velocity", [0.0, 20.0])
def test_uniform_water_translation_has_no_drag_at_free_surface(taichi_runtime, wet_layers, empty_velocity):
    node = make_mac_grid([4, 4, 4])
    translation = np.array([1.0, 0.5, -2.0])
    for d, mass_field in enumerate(node.m):
        mass = np.zeros(mass_field.shape)
        mass[:, :, : wet_layers + (d == 2)] = 2.0
        mass_field.from_numpy(mass)
        node.velocity[d].from_numpy(np.where(mass > 0, translation[d], empty_velocity))
    particle = SphereParticle.field(shape=1)
    particle[0].x, particle[0].rad, particle[0].v = [0.5] * 3, 0.125, translation
    sphere = SphereIndexFixture.field(shape=1)
    fraction = ti.field(float, shape=(4, 4, 4))
    fraction.fill(0.2)
    force = ti.Vector.field(3, float, shape=(4, 4, 4))
    velocity = ti.Vector.field(3, float, shape=1)
    velocity_weight = ti.Vector.field(3, float, shape=1)
    fluid_fraction = ti.field(float, shape=1)
    weights = ti.field(float, shape=1)
    kernel_compute_incompressible_sphere_drag(
        1,
        1e-12,
        6,
        3,
        ti.Vector([4] * 3),
        ti.Vector([0.25] * 3),
        ti.Vector([4.0] * 3),
        FluidProperties(),
        particle,
        sphere,
        node,
        fraction,
        force,
        velocity,
        velocity_weight,
        fluid_fraction,
        weights,
        DragForce({}),
    )
    np.testing.assert_allclose(particle.contact_force.to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(force.to_numpy(), 0.0, atol=1e-12)
