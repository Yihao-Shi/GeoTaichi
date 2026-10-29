"""Hydrodynamic load must exclude the fictitious fluid inside the rigid body."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.dem.structs.BaseStruct import BoundingSphere, RigidBody
from src.mpdem.fluid_dynamics.IncompressibleCoupling import TwoPhaseDoubleLayerLSDEMCoupling
from src.mpdem.fluid_dynamics.IncompressibleCouplingKernel import (
    kernel_apply_double_layer_lsdem_ibm,
    kernel_apply_double_layer_lsdem_ibm_pressure_predictor,
)
from src.mpm.engines.TwoPhaseDoubleLayerKernel import kernel_correct_double_layer_velocity3d
from tests.verification.mpdem.test_double_layer_ibm_rigid_exchange import Plane


@pytest.mark.parametrize("interface_case", [None, (1, 0.25), (0, 0.75)])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("acceleration", [0.0, 3.0, -9.81])
def test_partial_interface_preserves_pressure_balanced_motion(taichi_runtime, acceleration, fraction, interface_case):
    shape = (4, 3, 3)
    gravity, density, face_mass, timestep = -9.81, 1000.0, 2.0, 0.01
    phi = ti.field(float, shape=shape)
    phi.fill(fraction)
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
    if interface_case is not None:
        fluid_side, theta = interface_case
        left, right = (1, 1, 1), (2, 1, 1)
        if fluid_side == 1:
            kind[right], sdf[left], sdf[right] = 0, -theta, 1 - theta
            pressure[left] = -density * (gravity - acceleration) * theta
        else:
            kind[left], sdf[left], sdf[right] = 0, 1 - theta, -theta
            pressure[right] = density * (gravity - acceleration) * theta

    mass[0][face] = face_mass
    velocity[0][face], acc[0][face] = gravity * timestep, gravity
    dt = ti.field(float, shape=())
    dt[None] = timestep
    rigid, bound, box = RigidBody.field(shape=1), BoundingSphere.field(shape=1), Plane.field(shape=1)
    rigid[0].mass_center, rigid[0].q = [2.0, 1.25, 1.25], [0.0, 0.0, 0.0, 1.0]
    bound[0].active, bound[0].x, bound[0].rad = 1, [2.0, 1.25, 1.25], 10.0
    box[0].full = 0
    box[0].fraction = fraction
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
    kernel_apply_double_layer_lsdem_ibm_pressure_predictor(
        1e-12,
        dt,
        scene.element.grid_size,
        phi,
        target,
        tuple(mass),
        tuple(velocity),
        tuple(acc),
        reaction,
        pressure,
        kind,
        sdf,
        tuple(rho),
        None,
    )
    kernel_correct_double_layer_velocity3d(
        1e-12, scene.element.grid_size, dt, kind, pressure, sdf, *mass, *velocity, *acc, *rho, None
    )
    kernel_apply_double_layer_lsdem_ibm(1e-12, dt, phi, target, *mass, *velocity, *acc, reaction)
    TwoPhaseDoubleLayerLSDEMCoupling.accumulate_pressure_force(coupler, sims, scene)

    np.testing.assert_allclose(velocity[0][face], acceleration * timestep, atol=1e-14)
    np.testing.assert_allclose(acc[0][face], acceleration, atol=1e-12)
    # Analytic surface stress: -V grad(p) = m (a-g), including buoyancy at a=0.
    expected = np.array([fraction * face_mass * (acceleration - gravity), 0.0, 0.0])
    np.testing.assert_allclose(rigid.contact_force.to_numpy()[0], expected, atol=1e-11)
    np.testing.assert_allclose(rigid.contact_torque.to_numpy()[0], np.cross([0.0, 0.25, 0.25], expected), atol=1e-11)


@ti.dataclass
class InitialFluidPressure:
    active: ti.u8
    phase: ti.u8
    x: ti.types.vector(3, ti.f64)
    vol: ti.f64
    pressure: ti.f64


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("acceleration", [0.0, 3.0, -9.81])
def test_pressure_predictor_controller_preserves_initial_and_restart_pressure(
    taichi_runtime, acceleration, fraction, restart
):
    shape = (4, 3, 3)
    gravity, density, face_mass, timestep = -9.81, 1000.0, 2.0, 0.01
    phi = ti.field(float, shape=shape)
    phi.fill(fraction)
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
    box[0].full = 0
    box[0].fraction = fraction
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
    engine.cell_type, engine.cell_fluid_sdf, engine.cell_pressure = kind, sdf, pressure
    engine._build_fluid_sdf3d = lambda *_: None
    for axis, value, density_field in zip("xyz", velocity, rho):
        setattr(engine, "fluid_velocity_" + axis, value)
        setattr(engine, "face_fluid_density_" + axis, density_field)
    engine.fluid_pressure_mobility = tuple(ti.field(float, shape=field.shape) for field in mass)
    for field in engine.fluid_pressure_mobility:
        field.fill(1.0 / density)
    initial = InitialFluidPressure.field(shape=2)
    for i in range(2):
        initial[i].active, initial[i].phase, initial[i].vol = 1, 2, 1.0
        initial[i].x = [1.5 + i, 1.5, 1.5]
        initial[i].pressure = density * (gravity - acceleration) * (1.5 + i)
    scene.particle, scene.particleNum = initial, [2]
    scene.restart_cell_pressure = pressure.to_numpy() if restart else None
    sims.dt = dt
    controller = TwoPhaseDoubleLayerLSDEMCoupling.__new__(TwoPhaseDoubleLayerLSDEMCoupling)
    controller.__dict__.update(coupler.__dict__)
    controller.solid_velocity_cell = target
    controller.external_mac_boundary = lambda *_: None
    controller.update_solid_cells = lambda *_: True
    controller.previous_pressure = ti.field(float, shape=shape)
    controller.initial_pressure_weight = ti.field(float, shape=shape)
    controller.pressure_initialized = False
    controller.apply_before_pressure(sims, scene)
    assert controller.pressure_initialized and scene.restart_cell_pressure is None
    kernel_correct_double_layer_velocity3d(
        1e-12, scene.element.grid_size, dt, kind, pressure, sdf, *mass, *velocity, *acc, *rho, None
    )
    controller.apply_after_pressure(sims, scene)
    np.testing.assert_array_equal(controller.previous_pressure.to_numpy(), pressure.to_numpy())
    controller.accumulate_pressure_force(sims, scene)

    np.testing.assert_allclose(velocity[0][face], acceleration * timestep, atol=1e-14)
    np.testing.assert_allclose(acc[0][face], acceleration, atol=1e-12)
    # Analytic surface stress: -V grad(p) = m (a-g), including buoyancy at a=0.
    expected = np.array([fraction * face_mass * (acceleration - gravity), 0.0, 0.0])
    np.testing.assert_allclose(rigid.contact_force.to_numpy()[0], expected, atol=1e-11)
    np.testing.assert_allclose(rigid.contact_torque.to_numpy()[0], np.cross([0.0, 0.25, 0.25], expected), atol=1e-11)
