"""Production engine staging; force law is an independent conservative spring."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.dem.engines.ExplicitEngine import ExplicitEngine
from src.dem.structs.BaseStruct import (
    BoundingSphere,
    ClumpFamily,
    Material,
    ParticleFamily,
    RigidBody,
    SphereFamily,
)
from src.mpdem.fluid_dynamics.IncompressibleSemiResolved import kernel_apply_integrated_added_mass
from src.utils.linalg import no_operation

pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.cpu]


def make_engine(family, *, engine_name="VelocityVerlet", spring=1.0):
    particle = ParticleFamily.field(shape=1)
    sphere = SphereFamily.field(shape=1)
    clump = ClumpFamily.field(shape=1)
    rigid = RigidBody.field(shape=1)
    bounding = BoundingSphere.field(shape=1)
    material = Material.field(shape=1)
    dt = ti.field(float, shape=())
    dt[None] = 0.05
    particle[0].m = 1.0
    particle[0].x = [1.0, 0.0, 0.0]
    sphere[0].fix_v = sphere[0].fix_w = [1, 1, 1]
    sphere[0].q = [0.0, 0.0, 0.0, 1.0]
    sphere[0].inv_I = 1.0
    for body in (clump, rigid):
        body[0].m = 1.0
        body[0].mass_center = [1.0, 0.0, 0.0]
        body[0].q = [0.0, 0.0, 0.0, 1.0]
        body[0].inv_I = [1.0, 1.0, 1.0]
        # A stale restart acceleration must not affect a new step's drift.
        body[0].a = [777.0, 0.0, 0.0]
    sphere[0].a = [777.0, 0.0, 0.0]
    rigid[0].is_fix = [1, 1, 1]
    bounding[0].x = [1.0, 0.0, 0.0]
    scene = SimpleNamespace(
        particleNum=np.array([1]),
        sphereNum=np.array([int(family == "sphere")]),
        clumpNum=np.array([int(family == "clump")]),
        wallNum=np.array([0]),
        particle=bounding if family == "lsdem" else particle,
        sphere=sphere,
        clump=clump,
        rigid=rigid,
        material=material,
        soft_point=None,
        is_particle_need_update_verlet_table=lambda _: 0,
    )
    sims = SimpleNamespace(
        engine=engine_name,
        scheme="LSDEM" if family == "lsdem" else "DEM",
        max_particle_num=1,
        max_sphere_num=int(family == "sphere"),
        max_clump_num=int(family == "clump"),
        max_rigid_body_num=int(family == "lsdem"),
        max_wall_num=0,
        wall_type=0,
        enable_shell=False,
        static_wall=True,
        servo_status="Off",
        monitor_type=[],
        max_servo_wall_num=0,
        dt=dt,
        gravity=ti.Vector([0.0, 0.0, 0.0]),
        timer=SimpleNamespace(begin=no_operation, end=no_operation),
    )
    load_body = rigid if family == "lsdem" else particle
    state_body = rigid if family == "lsdem" else clump if family == "clump" else particle
    position = lambda: float(state_body[0].mass_center[0] if family != "sphere" else particle[0].x[0])
    force_stages, history = [], [0.0]

    def resolve(sims, _scene, _neighbor):
        duration = float(sims.dt[None])
        force_stages.append((duration, position()))
        history[0] += duration
        load_body[0].contact_force += ti.Vector([-spring * position(), 0.0, 0.0])

    contact = SimpleNamespace(
        neighbor=object(),
        physpp=SimpleNamespace(resolve=resolve, reset=no_operation),
        physpw=SimpleNamespace(resolve=no_operation, reset=no_operation),
    )
    engine = ExplicitEngine(scene, contact)
    engine.choose_engine(sims, scene)
    engine.set_servo_mechanism(sims)
    return engine, sims, scene, load_body, state_body, position, force_stages, history


@pytest.mark.parametrize("family", ["sphere", "clump", "lsdem"])
def test_verlet_evaluates_new_configuration_and_keeps_external_impulse(taichi_runtime, family):
    engine, sims, scene, load, state, position, stages, history = make_engine(family)
    physical_dt = sims.dt
    # External loads are allowed both before and after DEM contact assembly.
    load[0].contact_force = [0.2, 0.0, 0.0]
    engine.system_resolve(sims, scene, engine.neighbor)
    assert sims.dt is physical_dt
    load[0].contact_force += ti.Vector([0.3, 0.0, 0.0])
    engine.integration(sims, scene, engine.neighbor)
    expected_x = 1.0 + 0.5 * 0.05**2 * (-1.0 + 0.5)
    expected_v = 0.5 * 0.05 * (-1.0 - expected_x + 1.0)
    assert position() == pytest.approx(expected_x, abs=1e-13)
    assert state[0].v[0] == pytest.approx(expected_v, abs=1e-13)
    np.testing.assert_allclose(stages, [(0.0, 1.0), (0.05, expected_x)], atol=1e-13, rtol=0)
    assert history[0] == pytest.approx(0.05)
    assert load[0].contact_force[0] == pytest.approx(0.5 - expected_x)
    assert sims.dt is physical_dt


@pytest.mark.parametrize("family", ["sphere", "clump", "lsdem"])
def test_verlet_harmonic_energy_is_bounded_with_changing_steps(taichi_runtime, family):
    engine, sims, scene, load, state, position, stages, history = make_engine(family)
    total = 0.0
    for step in range(400):
        duration = (0.05, 0.025, 0.0125)[step % 3]
        sims.dt[None] = duration
        engine.reset_particle_message(scene)
        engine.system_resolve(sims, scene, engine.neighbor)
        engine.integration(sims, scene, engine.neighbor)
        total += duration
        energy = 0.5 * (position() ** 2 + state[0].v[0] ** 2)
        assert abs(energy - 0.5) < 5e-4
    assert position() == pytest.approx(np.cos(total), abs=2e-3)
    assert state[0].v[0] == pytest.approx(-np.sin(total), abs=2e-3)
    assert history[0] == pytest.approx(total)


def test_verlet_probe_restores_timestep_on_contact_failure(taichi_runtime):
    engine, sims, scene, *_ = make_engine("sphere")
    physical_dt = sims.dt

    def fail(*_):
        raise RuntimeError("contact failed")

    engine.physpp.resolve = fail
    with pytest.raises(RuntimeError, match="contact failed"):
        engine.system_resolve(sims, scene, engine.neighbor)
    assert sims.dt is physical_dt
    assert not engine.verlet_force_ready


def test_symplectic_euler_keeps_single_force_evaluation(taichi_runtime):
    engine, sims, scene, load, state, position, stages, history = make_engine("sphere", engine_name="SymplecticEuler")
    engine.system_resolve(sims, scene, engine.neighbor)
    engine.integration(sims, scene, engine.neighbor)
    assert position() == pytest.approx(1.0 - 0.05**2)
    assert state[0].v[0] == pytest.approx(-0.05)
    assert stages == [(0.05, 1.0)]


@pytest.mark.parametrize("engine_name", ["SymplecticEuler", "VelocityVerlet"])
def test_integrated_added_mass_is_applied_once_per_force_stage(taichi_runtime, engine_name):
    engine, sims, scene, load, state, position, *_ = make_engine("sphere", engine_name=engine_name, spring=0.0)
    scene.particle[0].rad = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    load[0].contact_force = [1.0, 0.0, 0.0]
    engine.transform_translational_load = lambda: kernel_apply_integrated_added_mass(
        1, 2.0, 1.0, sims.gravity, scene.particle, scene.sphere
    )

    engine.system_resolve(sims, scene, engine.neighbor)
    engine.integration(sims, scene, engine.neighbor)

    acceleration = 1.0 / 3.0
    assert state[0].v[0] == pytest.approx(sims.dt[None] * acceleration, rel=1e-6)
    position_factor = 1.0 if engine_name == "SymplecticEuler" else 0.5
    assert position() == pytest.approx(1.0 + position_factor * sims.dt[None] ** 2 * acceleration, rel=1e-6)


@pytest.mark.parametrize("family", ["sphere", "lsdem"])
def test_verlet_preserves_prescribed_nonzero_velocity(taichi_runtime, family):
    engine, sims, scene, load, state, position, *_ = make_engine(family)
    if family == "sphere":
        scene.sphere[0].fix_v = [0, 1, 1]
    else:
        scene.rigid[0].is_fix = [0, 1, 1]
    state[0].v = [0.2, 0.0, 0.0]
    engine.system_resolve(sims, scene, engine.neighbor)
    engine.integration(sims, scene, engine.neighbor)
    assert state[0].v[0] == pytest.approx(0.2, abs=1e-13)
    assert position() == pytest.approx(1.01, abs=1e-13)


def test_shell_reaction_is_reset_without_output_and_not_counted_twice(taichi_runtime):
    @ti.dataclass
    class WallLoad:
        contact_force: ti.types.vector(3, float)
        contact_torque: ti.types.vector(3, float)

        @ti.func
        def _reset(self):
            self.contact_force = ti.Vector([0.0, 0.0, 0.0])
            self.contact_torque = ti.Vector([0.0, 0.0, 0.0])

    engine, sims, scene, load, state, position, *_ = make_engine("sphere")
    scene.wall = WallLoad.field(shape=1)
    scene.wallNum[0] = sims.max_wall_num = 1
    sims.enable_shell = True
    engine.choose_engine(sims, scene)

    def wall_reaction(*_):
        scene.wall[0].contact_force += ti.Vector([position(), 0.0, 0.0])

    engine.physpw.resolve = wall_reaction
    for _ in range(2):
        engine.reset_particle_message(scene)
        engine.reset_wall_message(scene)
        assert scene.wall[0].contact_force.norm() == 0.0
        scene.wall[0].contact_force = [0.25, 0.0, 0.0]
        engine.system_resolve(sims, scene, engine.neighbor)
        engine.integration(sims, scene, engine.neighbor)
        assert scene.wall[0].contact_force[0] == pytest.approx(0.25 + position(), abs=1e-13)
