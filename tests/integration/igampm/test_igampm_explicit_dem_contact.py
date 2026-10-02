"""Explicit DEM-law point--NURBS IGA--MPM contact checks."""

from types import SimpleNamespace

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

import src.iga.config as iga_config
import src.igampm.config as coupling_config


pytestmark = [
    pytest.mark.integration,
    pytest.mark.coupling,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    iga_config.set_dimension(3)
    coupling_config.set_dimension(3)
    yield
    ti.reset()


@ti.kernel
def _initialize_particle(particle: ti.template()):
    particle[0]._set_essential(
        0,
        0,
        1,
        1000.0,
        1.0e-3,
        ti.Vector([0.5, 0.5, 0.169]),
        ti.Vector([1.0, 0.0, 0.0]),
        ti.Vector([0, 0, 0]),
    )


def _iga_cube(tmp_path):
    from src.iga import Cube, ExplicitIGA, Primitives

    cube = Cube()
    cube.set_parameters(start_point=[0.0, 0.0, 0.2], size=[1.0, 1.0, 0.2])
    cube.generate_knot_u(degree=2, num_ctrlpts=3)
    cube.generate_knot_v(degree=2, num_ctrlpts=3)
    cube.generate_knot_w(degree=2, num_ctrlpts=3)
    cube.generate_ctrlpts()
    cube.generate_weights()
    primitives = Primitives()
    primitives.append(cube, "cube")
    primitives.finialize()
    return ExplicitIGA(
        primitives=primitives,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0, 0.0],
        degree=[2, 2, 2],
        dt=1.0e-5,
        step=1,
        path=str(tmp_path),
    )


def _particle_scene():
    from src.mpm.structs.Particle import ParticleCoupling

    particle = ParticleCoupling.field(shape=1)
    _initialize_particle(particle)
    material = SimpleNamespace(matProps=ti.field(float, shape=2))
    return SimpleNamespace(
        particle=particle, couplingNum=np.asarray([1], dtype=np.int32), material=material
    )


@pytest.mark.parametrize("model_name", ["Linear", "HertzMindlin"])
def test_explicit_point_nurbs_dem_contact_is_balanced(tmp_path, model_name):
    from src.igampm.ContactManager import ContactManager
    from src.igampm.contact.ExplicitContact import ExplicitNurbsContact

    iga = _iga_cube(tmp_path)
    scene = _particle_scene()
    manager = ContactManager(contact_model=model_name)
    if model_name == "Linear":
        parameters = {
            "NormalStiffness": 1.0e5,
            "TangentialStiffness": 1.0e5,
            "Friction": 0.2,
            "NormalViscousDamping": 0.0,
            "TangentialViscousDamping": 0.0,
        }
    else:
        parameters = {
            "ShearModulus": 1.0e6,
            "Poisson": 0.25,
            "Friction": 0.2,
            "Restitution": 1.0,
        }
    manager.add_property(1, 0, parameters)
    contact = ExplicitNurbsContact(iga, scene, manager.phys)
    dt = ti.field(float, shape=())
    dt[None] = 1.0e-5
    iga.rhs.fill(0.0)

    active = contact.resolve(dt)
    mpm_force = scene.particle.external_force.to_numpy()[0]
    iga_force = iga.rhs.to_numpy().reshape((-1, 3)).sum(axis=0)
    rows = contact.contacts.active.to_numpy().astype(bool)
    normal_force = contact.contacts.normal_force.to_numpy()[rows].sum(axis=0)
    overlap = contact.contacts.tangential_overlap.to_numpy()[rows]

    assert active == 1
    assert normal_force[2] < 0.0
    assert mpm_force[0] < 0.0
    assert np.linalg.norm(overlap) > 0.0
    assert np.linalg.norm(mpm_force + iga_force) < 1.0e-10 * np.linalg.norm(
        mpm_force
    )


def test_explicit_dem_property_rejects_iga_rolling_dof_mismatch():
    from src.igampm.ContactManager import ContactManager

    manager = ContactManager(contact_model="Linear")
    manager.add_property(
        1,
        0,
        {
            "NormalStiffness": 1.0e5,
            "TangentialStiffness": 1.0e5,
            "Friction": 0.2,
            "RollingFriction": 0.1,
        },
    )
    with pytest.raises(ValueError, match="no rotational DOFs"):
        manager.phys.initialize(2, 1)


def test_explicit_contact_rejects_missing_material_patch_property(tmp_path):
    from src.igampm.ContactManager import ContactManager
    from src.igampm.contact.ExplicitContact import ExplicitNurbsContact

    manager = ContactManager(contact_model="Linear")
    with pytest.raises(RuntimeError, match=r"missing.*\(1, 0\)"):
        ExplicitNurbsContact(_iga_cube(tmp_path), _particle_scene(), manager.phys)


def test_public_explicit_igampm_advances_both_children(tmp_path):
    from src.iga import Cube, Primitives
    from src.iga.mainIGA import IGA
    from src.igampm import IGAMPM
    from src.mpm.mainMPM import MPM

    iga = IGA(log=False)
    mpm = MPM(log=False)
    coupling = IGAMPM(
        iga, mpm, log=False, contact_model="Linear"
    )
    mpm.set_configuration(
        domain=[2.0, 2.0, 2.0],
        gravity=[0.0, 0.0, 0.0],
        alphaPIC=0.0,
        mapping="USL",
        shape_function="Linear",
        configuration="ULMPM",
        solver_type="Explicit",
        material_type="Solid",
        visualize=False,
        log=False,
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": 8,
            "max_constraint_number": {},
        },
        log=False,
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "YoungModulus": 1.0e4,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(
        element={"ElementType": "R8N3D", "ElementSize": [0.1] * 3}
    )
    mpm.add_region(
        region={
            "Name": "point",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.45, 0.45, 0.12],
            "BoundingBoxSize": [0.1, 0.1, 0.1],
        }
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "point",
                "nParticlesPerCell": 1,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [1.0, 0.0, 0.0],
                "FixVelocity": ["Free"] * 3,
            }
        }
    )
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=False, grid=False, object=False)
    mpm.set_solver(
        {
            "Timestep": 1.0e-5,
            "SimulationTime": 1.0e-5,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
        },
        log=False,
    )

    iga.set_configuration(dimension=3, solver_type="Explicit")
    cube = Cube()
    cube.set_parameters(
        start_point=[0.0, 0.0, 0.2], size=[1.0, 1.0, 0.2]
    )
    cube.generate_knot_u(degree=2, num_ctrlpts=3)
    cube.generate_knot_v(degree=2, num_ctrlpts=3)
    cube.generate_knot_w(degree=2, num_ctrlpts=3)
    cube.generate_ctrlpts()
    cube.generate_weights()
    primitives = Primitives()
    primitives.append(cube, "cube")
    primitives.finialize()
    iga.add_primitives(primitives)
    iga.add_material(
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0, 0.0],
    )
    iga.add_element([2, 2, 2])
    iga.set_solver(dt=1.0e-5, step=1, interval=1, path=str(tmp_path))

    coupling.set_configuration(dimension=3, contact_model="Linear")
    coupling.add_property(
        1,
        0,
        {
            "NormalStiffness": 1.0e5,
            "TangentialStiffness": 1.0e5,
            "Friction": 0.2,
            "NormalViscousDamping": 0.0,
            "TangentialViscousDamping": 0.0,
        },
    )
    engine = coupling.build()
    coupling.run(steps=1, verbose=False)
    active = engine.last_contact_count

    assert active == 1
    assert engine.dt <= 1.0e-5
    assert iga.engine.dt == pytest.approx(engine.dt)
    assert engine.current_step == 1
    assert engine.current_time == pytest.approx(engine.dt)
    assert np.all(np.isfinite(iga.engine.patch.control_points.to_numpy()))
    assert np.all(np.isfinite(mpm.scene.particle.x.to_numpy()))
    assert (tmp_path / "vtks" / "NurbsVolumecube000000.vtu").is_file()
