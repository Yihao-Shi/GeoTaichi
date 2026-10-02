import numpy as np
import pytest


ti = pytest.importorskip("taichi")

from src.mpdem.fluid_dynamics.IncompressibleCouplingKernel import kernel_apply_double_layer_lsdem_ibm


def test_double_layer_ibm_face_exchange_is_equal_and_opposite():
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    fraction = ti.field(float, shape=(2, 1, 1))
    solid_velocity = ti.Vector.field(3, float, shape=(2, 1, 1))
    reaction = ti.Vector.field(3, float, shape=(2, 1, 1))
    mass_x = ti.field(float, shape=(3, 1, 1))
    mass_y = ti.field(float, shape=(2, 2, 1))
    mass_z = ti.field(float, shape=(2, 1, 2))
    velocity_x = ti.field(float, shape=(3, 1, 1))
    velocity_y = ti.field(float, shape=(2, 2, 1))
    velocity_z = ti.field(float, shape=(2, 1, 2))
    acceleration_x = ti.field(float, shape=(3, 1, 1))
    acceleration_y = ti.field(float, shape=(2, 2, 1))
    acceleration_z = ti.field(float, shape=(2, 1, 2))
    dt = ti.field(float, shape=())

    fraction.fill(1.0)
    mass_x[1, 0, 0] = 2.0
    velocity_x[1, 0, 0] = 1.0
    dt[None] = 0.1

    kernel_apply_double_layer_lsdem_ibm(
        1.0e-12,
        dt,
        fraction,
        solid_velocity,
        mass_x,
        mass_y,
        mass_z,
        velocity_x,
        velocity_y,
        velocity_z,
        acceleration_x,
        acceleration_y,
        acceleration_z,
        reaction,
    )

    assert velocity_x[1, 0, 0] == pytest.approx(0.0)
    assert acceleration_x[1, 0, 0] == pytest.approx(-10.0)
    body_impulse = np.sum(reaction.to_numpy(), axis=(0, 1, 2)) * dt[None]
    np.testing.assert_allclose(body_impulse, [2.0, 0.0, 0.0], atol=1.0e-12)
