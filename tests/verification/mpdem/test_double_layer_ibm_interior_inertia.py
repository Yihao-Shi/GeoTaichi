"""Hydrodynamic load must exclude the fictitious fluid inside the rigid body."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.dem.structs.BaseStruct import BoundingSphere, RigidBody
from src.mpdem.fluid_dynamics.IncompressibleCoupling import TwoPhaseDoubleLayerLSDEMCoupling
from src.mpdem.fluid_dynamics.IncompressibleCouplingKernel import kernel_apply_double_layer_lsdem_ibm
from src.mpm.engines.TwoPhaseDoubleLayerKernel import kernel_correct_double_layer_velocity3d
from tests.verification.mpdem.test_double_layer_ibm_rigid_exchange import Plane


@pytest.mark.parametrize("acceleration", [0.0, 3.0, -9.81])
def test_prescribed_pressure_gradient_recovers_interior_surface_load(taichi_runtime, acceleration):
    shape = (4, 3, 3)
    gravity, density, face_mass, timestep = -9.81, 1000.0, 2.0, 0.01
    phi = ti.field(float, shape=shape)
    phi.fill(1.0)
    pressure = ti.field(float, shape=shape)
    pressure.from_numpy(density * (gravity - acceleration) * (np.indices(shape)[0] + 0.5))
    kind = ti.field(int, shape=shape)
    kind.fill(1)
    sdf = ti.field(float, shape=shape)
    sdf.fill(-1.0)
    target = ti.Vector.field(3, float, shape=shape)
    target.fill([acceleration * timestep, 0.0, 0.0])
    reaction = ti.Vector.field(3, float, shape=shape)
    shapes = [(5, 3, 3), (4, 4, 3), (4, 3, 4)]
    mass, velocity, acc, rho = [[ti.field(float, shape=s) for s in shapes] for _ in range(4)]
    for field in rho:
        field.fill(density)
    face = (2, 1, 1)
    mass[0][face] = face_mass
    velocity[0][face], acc[0][face] = gravity * timestep, gravity
    dt = ti.field(float, shape=())
    dt[None] = timestep
    rigid, bound, box = RigidBody.field(shape=1), BoundingSphere.field(shape=1), Plane.field(shape=1)
    rigid[0].mass_center, rigid[0].q = [2.0, 1.25, 1.25], [0.0, 0.0, 0.0, 1.0]
    bound[0].active, bound[0].x, bound[0].rad = 1, [2.0, 1.25, 1.25], 10.0
    box[0].full = 1
    engine = SimpleNamespace(
        **{f"fluid_mass_{axis}": field for axis, field in zip("xyz", mass)},
        **{f"fluid_acceleration_{axis}": field for axis, field in zip("xyz", acc)},
    )
    coupler = SimpleNamespace(
        dscene=SimpleNamespace(rigidNum=[1], particle=bound, rigid=rigid, box=box, rigid_grid=ti.field(float, shape=1)),
        solid_fraction=phi,
        solid_fraction_sum=phi,
        ibm_reaction_cell=reaction,
        mengine=engine,
    )
    scene = SimpleNamespace(
        mass_cut_off=1e-12, element=SimpleNamespace(cnum=ti.Vector(shape), grid_size=ti.Vector([1.0] * 3))
    )
    sims = SimpleNamespace(gravity=ti.Vector([gravity, 0.0, 0.0]))
    kernel_apply_double_layer_lsdem_ibm(1e-12, dt, phi, target, *mass, *velocity, *acc, reaction)
    kernel_correct_double_layer_velocity3d(
        1e-12, scene.element.grid_size, dt, kind, pressure, sdf, *mass, *velocity, *acc, *rho, None
    )
    kernel_apply_double_layer_lsdem_ibm(1e-12, dt, phi, target, *mass, *velocity, *acc, reaction)
    TwoPhaseDoubleLayerLSDEMCoupling.accumulate_pressure_force(coupler, sims, scene)

    np.testing.assert_allclose(velocity[0][face], acceleration * timestep, atol=1e-14)
    np.testing.assert_allclose(acc[0][face], acceleration, atol=1e-12)
    # Analytic surface stress: -V grad(p) = m (a-g), including buoyancy at a=0.
    expected = np.array([face_mass * (acceleration - gravity), 0.0, 0.0])
    np.testing.assert_allclose(rigid.contact_force.to_numpy()[0], expected, atol=1e-11)
    np.testing.assert_allclose(rigid.contact_torque.to_numpy()[0], np.cross([0.0, 0.25, 0.25], expected), atol=1e-11)
