import numpy as np
import pytest
import taichi as ti

from src.dem.engines.EngineKernel import (
    move_level_set_euler_,
    move_level_set_verlet_predictor_,
    move_level_set_verlet_corrector_,
)
from src.dem.structs.BaseStruct import BoundingSphere, Material, RigidBody

pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.cpu]


def _make_rotating_body():
    dt = ti.field(dtype=ti.f64, shape=())
    sphere = BoundingSphere.field(shape=1)
    rigid = RigidBody.field(shape=1)
    material = Material.field(shape=1)

    @ti.kernel
    def initialize():
        dt[None] = 0.1
        material[0].fdamp = 0.0
        material[0].tdamp = 0.0
        rigid[0].materialID = ti.u8(0)
        rigid[0].m = 1.0
        rigid[0].mass_center = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].a = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].v = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].w = ti.Vector([0.0, 0.0, 2.0])
        rigid[0].angmoment = ti.Vector([0.0, 0.0, 4.0])
        rigid[0].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        rigid[0].inv_I = ti.Vector([1.0, 1.0, 0.5])
        rigid[0].contact_force = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].contact_torque = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].is_fix = ti.Vector([1, 1, 1])
        rigid[0].is_soft = ti.u8(0)
        sphere[0].x = ti.Vector([1.0, 0.0, 0.0])
        sphere[0].verletDisp = ti.Vector([0.0, 0.0, 0.0])

    initialize()
    return dt, sphere, rigid, material


def _assert_constant_spin_state(sphere, rigid, expected_angle):
    q = np.asarray(rigid.q.to_numpy()[0], dtype=np.float64)
    omega = np.asarray(rigid.w.to_numpy()[0], dtype=np.float64)
    angular_momentum = np.asarray(rigid.angmoment.to_numpy()[0], dtype=np.float64)
    center = np.asarray(sphere.x.to_numpy()[0], dtype=np.float64)

    observed_angle = 2.0 * np.arctan2(np.linalg.norm(q[:3]), q[3])
    assert observed_angle == pytest.approx(expected_angle, abs=2.0e-12)
    np.testing.assert_allclose(omega, [0.0, 0.0, 2.0], atol=2.0e-12)
    np.testing.assert_allclose(angular_momentum, [0.0, 0.0, 4.0], atol=2.0e-12)
    np.testing.assert_allclose(
        center,
        [np.cos(expected_angle), np.sin(expected_angle), 0.0],
        atol=2.0e-12,
    )
    rotational_energy = 0.5 * float(np.dot(omega, angular_momentum))
    assert rotational_energy == pytest.approx(4.0, abs=2.0e-12)


def test_levelset_symplectic_euler_advects_existing_spin(taichi_runtime):
    dt, sphere, rigid, material = _make_rotating_body()
    move_level_set_euler_(1, dt, sphere, rigid, material, ti.Vector([0.0, 0.0, 0.0]))

    _assert_constant_spin_state(sphere, rigid, expected_angle=2.0 * np.arctan(0.1))


def test_levelset_velocity_verlet_keeps_world_momentum_consistent(
    taichi_runtime,
):
    dt, sphere, rigid, material = _make_rotating_body()
    move_level_set_verlet_predictor_(1, dt, sphere, rigid, material, ti.Vector([0.0, 0.0, 0.0]))
    move_level_set_verlet_corrector_(1, dt, rigid, material, ti.Vector([0.0, 0.0, 0.0]))

    _assert_constant_spin_state(sphere, rigid, expected_angle=0.2)
