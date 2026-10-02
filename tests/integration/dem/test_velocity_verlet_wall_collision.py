"""Actual DEM neighbor search, contact law and time-step loop, without fluid."""

import math
from types import SimpleNamespace

import pytest

from src.dem.mainDEM import DEM
from src.mpdem.Engine import Engine as CoupledEngine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.dem,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


def make_wall_system(tmp_path, step_size, skin=0.2):
    diameter, density, stiffness, speed = 0.015, 1120.0, 2e6, 0.128
    dem = DEM(log=False)
    dem.set_configuration(
        domain=[0.1, 0.1, 0.16], gravity=[0, 0, 0], engine="VelocityVerlet", search="LinkedCell", log=False
    )
    dem.set_solver(
        {"Timestep": step_size, "SimulationTime": 0.001, "SaveInterval": 0.001, "SavePath": str(tmp_path)}, log=False
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": 1,
            "max_sphere_number": 1,
            "max_clump_number": 0,
            "max_plane_number": 1,
            "body_coordination_number": 0,
            "wall_coordination_number": 1,
            "verlet_distance_multiplier": skin,
        },
        log=False,
    )
    dem.add_attribute(materialID=0, attribute={"Density": density, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0})
    dem.create_body(
        {
            "BodyType": "Sphere",
            "Template": [
                {
                    "GroupID": 0,
                    "MaterialID": 0,
                    "Radius": diameter / 2,
                    "BodyPoint": [0.05, 0.05, diameter / 2],
                    "InitialVelocity": [0.0, 0.0, -speed],
                    "InitialAngularVelocity": [0, 0, 0],
                    "FixVelocity": ["Free"] * 3,
                    "FixAngularVelocity": ["Free"] * 3,
                    "BodyOrientation": "uniform",
                }
            ],
        }
    )
    dem.add_wall(body={"WallType": "Plane", "MaterialID": 0, "WallCenter": [0, 0, 0], "OuterNormal": [0, 0, 1]})
    dem.choose_contact_model(None, "Linear Model")
    dem.add_property(
        0,
        0,
        {
            "NormalStiffness": stiffness,
            "TangentialStiffness": 1e6,
            "Friction": 0.0,
            "NormalViscousDamping": 0.0,
            "TangentialViscousDamping": 0.0,
        },
    )
    dem.select_save_data()
    dem.add_essentials()
    return dem


@pytest.mark.parametrize("skin", [0.0, 0.2])
def test_coupled_dem_resolves_first_impact_from_free_flight(taichi_runtime, tmp_path, skin):
    macro_dt, dem_dt, speed = 2.5e-4, 1e-5, 0.128
    dem = make_wall_system(tmp_path, dem_dt, skin)
    dem.scene.particle[0].x = [0.05, 0.05, dem.scene.particle[0].rad + 1.5 * speed * macro_dt]
    dem.sims.set_timestep(macro_dt)
    dem.enginer.pre_calculation(dem.sims, dem.scene, dem.contactor.neighbor)
    engine = object.__new__(CoupledEngine)
    engine.sims = SimpleNamespace(delta=macro_dt, dem_timestep=dem_dt)
    engine.dsims, engine.dscene = dem.sims, dem.scene
    engine.dengine, engine.dneighbor = dem.enginer, dem.contactor.neighbor
    engine.msims = engine.mscene = engine.mneighbor = None
    engine.mengine = SimpleNamespace(compute=lambda *_: None)
    engine.accumulate_pressure_force = lambda *_: None
    engine.resolve_cross_contact = engine.update_servo_wall = engine.get_wall_contact_forces = lambda: None
    engine.dem_external_force = engine.dem_external_torque = None
    for _ in range(6):
        engine.dengine.reset_wall_message(engine.dscene)
        engine.dengine.reset_particle_message(engine.dscene)
        engine.incompressible_dem_sphere_integration()
        assert dem.sims.delta == macro_dt
    assert dem.scene.particle[0].x[2] > dem.scene.particle[0].rad
    assert dem.scene.particle[0].v[2] / speed == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("step_size, error_bound", [(1e-5, 0.008), (5e-6, 0.003), (2.5e-6, 0.001)])
def test_elastic_wall_restitution_converges_to_one(taichi_runtime, tmp_path, step_size, error_bound):
    diameter, density, stiffness, speed = 0.015, 1120.0, 2e6, 0.128
    mass = density * math.pi * diameter**3 / 6.0
    dem = make_wall_system(tmp_path, step_size)
    dem.enginer.pre_calculation(dem.sims, dem.scene, dem.contactor.neighbor)
    steps = math.ceil(3 * math.pi * math.sqrt(mass / stiffness) / step_size)
    minimum_gap = 0.0
    for _ in range(steps):
        dem.solver.core(dem.scene)
        minimum_gap = min(minimum_gap, dem.scene.particle[0].x[2] - diameter / 2)
    restitution = dem.scene.particle[0].v[2] / speed
    assert minimum_gap < 0.0  # A real contact occurred; a free-flight pass is invalid.
    assert dem.scene.particle[0].x[2] > diameter / 2
    assert abs(restitution - 1.0) < error_bound


def test_wall_sticking_history_advances_once_per_physical_step(taichi_runtime, tmp_path):
    dt, velocity, overlap = 1e-5, 0.02, 1e-4
    dem = make_wall_system(tmp_path, dt)
    dem.scene.particle[0].x = [0.05, 0.05, dem.scene.particle[0].rad - overlap]
    dem.scene.particle[0].v = [velocity, 0, 0]
    dem.scene.sphere[0].fix_v = dem.scene.sphere[0].fix_w = [0, 0, 0]
    dem.contactor.physpw.surfaceProps[0].mus = 0.5
    dem.contactor.physpw.surfaceProps[0].mud = 0.5
    dem.enginer.pre_calculation(dem.sims, dem.scene, dem.contactor.neighbor)
    for step in range(1, 11):
        dem.solver.core(dem.scene)
        contact = dem.contactor.physpw.cplist[0]
        assert contact.oldTangOverlap[0] == pytest.approx(step * dt * velocity, abs=1e-13)
        assert contact.csforce[0] == pytest.approx(-1e6 * step * dt * velocity, abs=1e-9)
        assert contact.cnforce[2] == pytest.approx(2e6 * overlap, abs=1e-9)
