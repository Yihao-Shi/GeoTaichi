from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.dem.structs.BaseStruct import BoundingSphere, RigidBody
from src.mpdem.fluid_dynamics.IncompressibleCoupling import TwoPhaseDoubleLayerLSDEMCoupling
from src.mpdem.fluid_dynamics.IncompressibleCouplingKernel import kernel_apply_double_layer_lsdem_ibm
from src.mpm.engines.TwoPhaseDoubleLayerKernel import kernel_correct_double_layer_velocity3d


@ti.dataclass
class Plane:
    full: int
    fraction: float

    @ti.func
    def _in_box(self, position):
        return True

    @ti.func
    def distance(self, position, grid):
        distance = position[1] + 0.25 - self.fraction
        if self.full:
            distance = -1.0
        return distance


@pytest.mark.parametrize("fraction", [1e-7, 0.5, 1.0])
@pytest.mark.parametrize("air_neighbor", [False, True])
@pytest.mark.parametrize("bodies", [1, 2])
def test_ibm_impulses_reach_rigid_bodies_before_and_after_pressure(taichi_runtime, fraction, air_neighbor, bodies):
    shape = (2, 1, 1)

    def scalar(value=0.0, dtype=float):
        field = ti.field(dtype, shape=shape)
        field.fill(value)
        return field

    phi, phi_sum = scalar(fraction), scalar(bodies * fraction)
    density, fluid_density = scalar(1140.0), scalar(1000.0)
    pressure, zero, sdf, kind = scalar(), scalar(), scalar(-1.0), scalar(1, int)
    pressure[0, 0, 0], pressure[1, 0, 0] = 2.0, 5.0
    if air_neighbor:
        kind[1, 0, 0], sdf[1, 0, 0] = 0, 1.0
    target = ti.Vector.field(3, float, shape=shape)
    reaction = ti.Vector.field(3, float, shape=shape)
    face_shapes = [(3, 1, 1), (2, 2, 1), (2, 1, 2)]
    mass, velocity, acceleration, face_density = [[ti.field(float, shape=s) for s in face_shapes] for _ in range(4)]
    for field in face_density:
        field.fill(1000.0)
    face = (1, 0, 0)
    mass[0][face], velocity[0][face] = 2.0, 1.0
    dt = ti.field(float, shape=())
    dt[None] = 0.1
    rigid, bound, box = RigidBody.field(shape=bodies), BoundingSphere.field(shape=bodies), Plane.field(shape=bodies)
    for body in range(bodies):
        rigid[body].mass_center, rigid[body].q = [1.0, 0.25, 0.25], [0.0, 0.0, 0.0, 1.0]
        bound[body].active, bound[body].x, bound[body].rad = 1, [1.0, 0.25, 0.25], 2.0
        box[body].full = int(fraction == 1.0)
        box[body].fraction = fraction
    dscene = SimpleNamespace(
        rigidNum=[bodies], particle=bound, rigid=rigid, box=box, rigid_grid=ti.field(float, shape=1)
    )
    engine = SimpleNamespace(
        cell_pressure=pressure,
        cell_type=kind,
        cell_fluid_sdf=sdf,
        cell_fluid_density=fluid_density,
        cell_fluid_viscosity=zero,
        fluid_velocity_x=velocity[0],
        fluid_velocity_y=velocity[1],
        fluid_velocity_z=velocity[2],
        **{f"fluid_mass_{d}": f for d, f in zip("xyz", mass)},
        **{f"fluid_acceleration_{d}": f for d, f in zip("xyz", acceleration)},
    )
    coupler = SimpleNamespace(
        dscene=dscene,
        solid_fraction=phi,
        solid_fraction_sum=phi_sum,
        solid_density=density,
        ibm_reaction_cell=reaction,
        zero_surface_tension=zero,
        mengine=engine,
    )
    scene = SimpleNamespace(
        mass_cut_off=1e-12,
        element=SimpleNamespace(cnum=ti.Vector(shape), grid_size=ti.Vector([1.0] * 3), igrid_size=ti.Vector([1.0] * 3)),
    )

    def apply_ibm():
        kernel_apply_double_layer_lsdem_ibm(1e-12, dt, phi, target, *mass, *velocity, *acceleration, reaction)

    apply_ibm()
    before_pressure = velocity[0][face]
    kernel_correct_double_layer_velocity3d(
        1e-12,
        scene.element.grid_size,
        dt,
        kind,
        pressure,
        sdf,
        *mass,
        *velocity,
        *acceleration,
        *face_density,
        None,
    )
    pressure_impulse = 2.0 * (velocity[0][face] - before_pressure)
    apply_ibm()
    TwoPhaseDoubleLayerLSDEMCoupling.accumulate_pressure_force(
        coupler,
        SimpleNamespace(gravity=ti.Vector([0.0] * 3)),
        scene,
    )

    fluid_impulse = 2.0 * (velocity[0][face] - 1.0)
    expected = np.array([pressure_impulse - fluid_impulse, 0.0, 0.0])
    np.testing.assert_allclose(reaction.to_numpy().sum(axis=(0, 1, 2)) * dt[None], expected, atol=1e-12)
    # Physical water excludes the fraction occupied by the rigid body.
    expected += np.array([fraction * fluid_impulse, 0.0, 0.0])
    np.testing.assert_allclose(
        rigid.contact_force.to_numpy() * dt[None], np.tile(expected / bodies, (bodies, 1)), atol=1e-12
    )
    expected_torque = np.cross([0.0, 0.25, 0.25], expected)
    np.testing.assert_allclose(rigid.contact_torque.to_numpy().sum(axis=0) * dt[None], expected_torque, atol=1e-12)
