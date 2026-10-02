import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem.generator import FEMMesh
from src.fem.mainFEM import FEM
from src.fempm.mainFEMPM import FEMPM
from src.mpm.mainMPM import MPM


pytestmark = [
    pytest.mark.integration,
    pytest.mark.fempm,
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
    yield
    ti.reset()


def _coupled_system(tmp_path, search, contact_model="Linear"):
    mpm = MPM(log=False)
    fem = FEM(log=False)
    coupling = FEMPM(fem, mpm, log=False)

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
        element={"ElementType": "R8N3D", "ElementSize": [0.1, 0.1, 0.1]}
    )
    mpm.add_region(
        region={
            "Name": "point",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.45, 0.45, 0.1],
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
                "FixVelocity": ["Free", "Free", "Free"],
            }
        }
    )
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=False, grid=False, object=False)

    fem.set_configuration(dimension=3, solver_type="Explicit")
    fem.add_mesh(
        FEMMesh(
            np.array(
                [
                    [0.0, 0.0, 0.11],
                    [1.5, 0.0, 0.11],
                    [0.0, 1.5, 0.11],
                ]
            ),
            np.array([[0, 1, 2]], dtype=np.int32),
            "TRI3",
        )
    )
    fem.add_material(
        "StVK",
        density=1000.0,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        thickness=0.05,
    )
    fem.add_boundary_condition(
        {
            "type": "Dirichlet",
            "nodes": [0, 1, 2],
            "components": "all",
            "value": 0.0,
        }
    )

    coupling.set_configuration(
        domain=[2.0, 2.0, 2.0],
        gravity=[0.0, 0.0, 0.0],
        search=search,
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": 1.0e-5,
            "SimulationTime": 1.0e-5,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
        },
        log=False,
    )
    coupling.add_surface(
        modifier={"Orientation": "Parallel", "Direction": [0, 0, 1]}
    )
    coupling.memory_allocate(
        {
            "contact_coordination_number": 8,
            "max_contact_pairs": 16,
            "max_facet_cell_pairs": 128,
        }
    )
    coupling.choose_contact_model(contact_model)
    if contact_model == "Linear":
        contact_property = {
            "NormalStiffness": 1.0e5,
            "TangentialStiffness": 1.0e5,
            "Friction": 0.2,
            "NormalViscousDamping": 0.0,
            "TangentialViscousDamping": 0.0,
        }
    else:
        contact_property = {
            "ShearModulus": 1.0e6,
            "Poisson": 0.25,
            "Friction": 0.0,
            "Restitution": 1.0,
        }
    coupling.add_property(1, 0, contact_property)
    coupling.add_essentials()
    coupling.enginer.pre_calculate()
    return coupling


@pytest.mark.parametrize("search", ["LinkedCell", "BVH"])
def test_explicit_fempm_search_force_balance_and_history(tmp_path, search):
    coupling = _coupled_system(tmp_path, search)
    assert coupling.contactor.neighbor.contact_count == 1

    coupling.enginer.reset_message()
    coupling.enginer.system_resolve()

    contacts = coupling.contactor.neighbor.contacts
    normal_force = contacts.normal_force.to_numpy()[0]
    tangential_overlap = contacts.old_tangential_overlap.to_numpy()[0].copy()
    mpm_force = coupling.mpm.scene.particle.external_force.to_numpy()[0]
    fem_force = coupling.fem.engine.state.external_force.to_numpy().sum(axis=0)
    assert normal_force[2] > 0.0
    assert np.linalg.norm(tangential_overlap) > 0.0
    assert np.linalg.norm(mpm_force + fem_force) < 1.0e-8 * np.linalg.norm(
        mpm_force
    )

    coupling.contactor.rebuild(coupling.mpm.scene)
    assert coupling.contactor.neighbor.contact_count == 1
    assert contacts.old_tangential_overlap.to_numpy()[0] == pytest.approx(
        tangential_overlap
    )

    coupling.enginer.integration()
    assert coupling.enginer.minimum_jacobian == pytest.approx(1.0)


def test_explicit_fempm_hertz_and_shared_timestep_gate(tmp_path):
    coupling = _coupled_system(tmp_path, "BVH", "HertzMindlin")
    coupling.enginer.reset_message()
    coupling.enginer.system_resolve()
    assert coupling.contactor.neighbor.contacts.normal_force.to_numpy()[0, 2] > 0.0

    coupling.sims.set_timestep(10.0)
    coupling.mpm.scene.get_critical_timestep = lambda: 4.0
    coupling.fem.engine.stable_time_step = lambda cfl: 2.0 * cfl
    coupling.contactor.critical_timestep = lambda *_: 3.0
    coupling.check_critical_timestep()
    assert coupling.sims.delta == pytest.approx(1.0)
    assert float(coupling.mpm.sims.dt[None]) == pytest.approx(1.0)
    assert coupling.fem.engine.dt == pytest.approx(1.0)
