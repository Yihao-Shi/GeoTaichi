import pytest

ti = pytest.importorskip("taichi")

from geotaichi import DEM, polyhedron
from src.fem import FEM
from src.fedem.mainFEDEM import FEDEM

pytestmark = [pytest.mark.integration, pytest.mark.fedem, pytest.mark.ipc, pytest.mark.cpu, pytest.mark.serial]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


def test_implicit_affine_ipc_accepts_cloth_surface(tmp_path):
    dem = DEM(log=False)
    dem.set_configuration(
        domain=[2.0, 2.0, 2.0],
        scheme="AffineBody",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=2.0e4,
        local_damping=0.0,
        contact_damping_stiffness=0.0,
        hessian_shift=0.0,
        friction_mode="lagged",
        friction_iterations=1,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": 1,
            "surface_node_number": 16,
            "body_coordination_number": 8,
            "wall_coordination_number": 1,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.add_attribute(materialID=0, attribute={"Density": 1000.0})
    dem.add_template(
        {"Name": "oct", "TemplateType": "AffineBody", "Object": polyhedron(file="tests/data/affine_octahedron.obj")}
    )
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": [
                {
                    "Name": "oct",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [0.35, 0.35, 0.2],
                    "ScaleFactor": 0.2,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                }
            ],
        }
    )

    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh({"Geometry": "Rectangle", "Size": (1.0, 1.0), "Divisions": (1, 1), "ElementType": "TRI3"})
    fem.add_material(
        "ClothARAP",
        stretch_stiffness=1.0e4,
        compression_stiffness=1.0e4,
        density=1000.0,
        thickness=0.05,
        bending_stiffness=0.0,
        bending_model="None",
    )
    fem.add_boundary_condition({"type": "Dirichlet", "nodes": [0, 1, 2, 3], "components": "all", "value": 0.0})

    coupling = FEDEM(dem, fem, log=False)
    coupling.set_configuration(domain=[2.0, 2.0, 2.0], search="LinkedCell", log=False)
    coupling.set_solver(
        {
            "Timestep": 1.0e-3,
            "SimulationTime": 1.0e-3,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
            "assemble_type": "HashTriplet",
            "linear_solver": "PCG",
        },
        log=False,
    )
    coupling.add_surface()
    coupling.memory_allocate({"max_contact_pairs": 64, "contact_coordination_number": 16})
    coupling.choose_contact_model("BarrierIPC", dhat=0.25, kappa=2.0e4, friction_coefficient=0.0, epsv=1.0e-3)
    coupling.add_ipc_property(AffineBody=0, FEMbody=0, property={"friction_coefficient": 0.0, "epsv": 1.0e-3})
    coupling.add_essentials()

    engine = coupling.enginer
    engine.affine.device_begin_step(engine.dt)
    engine.contact.begin_step(engine.fem.state.position, engine.fem.state.old_position, engine.dt)
    pt_count, ee_count = engine.contact.prepare(engine.fem.state.position)
    assert pt_count + ee_count > 0
    assert engine.node_count == engine.affine_controls + engine.fem_nodes
