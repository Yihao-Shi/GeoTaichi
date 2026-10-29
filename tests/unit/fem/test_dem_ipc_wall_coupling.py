"""The public ABD and FEM--ABD solvers share DEM facet servo boundaries."""

from pathlib import Path
import numpy as np
import pytest
import taichi as ti
from geotaichi import DEM, polyhedron
from src.fem import FEM
from src.fem.generator import FEMMesh
from src.fedem.mainFEDEM import FEDEM

pytestmark = [
    pytest.mark.unit,
    pytest.mark.fem,
    pytest.mark.ipc,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


def dem_case(tmp_path):
    dem = DEM(log=False)
    dem.set_configuration(domain=[2.0, 2.0, 2.0], scheme="AffineBody", search="BVH", gravity=[0.0, 0.0, 0.0], log=False)
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=2e4,
        dhat=0.1,
        barrier_stiffness=100.0,
        local_damping=0.0,
        friction_iterations=1,
        max_newton_iteration=40,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": 1,
            "surface_node_number": 16,
            "max_facet_number": 2,
            "max_servo_wall_number": 1,
            "max_point_triangle_pairs": 128,
            "max_edge_edge_pairs": 128,
            "body_coordination_number": 4,
            "wall_coordination_number": 2,
        },
        log=False,
    )
    dem.set_solver(
        {"Timestep": 0.001, "SimulationTime": 0.002, "SaveInterval": 0.001, "SavePath": str(tmp_path)}, log=False
    )
    dem.add_attribute(0, {"Density": 1000.0})
    mesh = Path(__file__).resolve().parents[2] / "data/affine_octahedron.obj"
    dem.add_template({"Name": "oct", "TemplateType": "AffineBody", "Object": polyhedron(file=str(mesh))})
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": {
                "Name": "oct",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.06, 0.7, 0.7],
                "ScaleFactor": 0.05,
                "InitialVelocity": [0.0, 0.0, 0.0],
            },
        }
    )
    dem.add_wall(
        {
            "WallType": "Facet",
            "WallShape": "Polygon",
            "WallID": 0,
            "MaterialID": 0,
            "WallVertice": {
                f"p{i}": p for i, p in enumerate([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
            },
            "OuterNormal": np.array([1.0, 0.0, 0.0]),
            "ControlType": "Stress",
            "TargetStress": 100.0,
            "LimitVelocity": 0.02,
        }
    )
    dem.add_property(0, 0, {"Dhat": 0.1, "BarrierStiffness": 100.0, "Friction": 0.0}, dType="all")
    dem.servo_switch("On")
    return dem


@pytest.mark.parametrize("coupled", [False, True])
def test_dem_wall_servo_is_used_by_abd_and_coupled_fem(taichi_runtime, tmp_path, coupled):
    dem = dem_case(tmp_path / "dem")
    if not coupled:
        dem.run()
        boundary = dem.enginer.wall_boundary
    else:
        fem = FEM(log=False)
        fem.set_configuration(dimension=3, solver_type="Implicit")
        points = np.array([[0.01, 0.25, 0.25], [0.08, 0.25, 0.25], [0.08, 0.35, 0.25], [0.08, 0.25, 0.35]])
        fem.add_mesh(FEMMesh(points, np.array([[0, 1, 2, 3]], dtype=np.int32), "TET4"))
        fem.add_material("StVK", density=1000.0, young_modulus=1e4, poisson_ratio=0.3)
        fem.add_boundary_condition({"type": "Dirichlet", "nodes": [0, 1, 2, 3], "components": "all", "value": 0.0})
        coupling = FEDEM(dem, fem, log=False)
        coupling.set_configuration(domain=[2.0, 2.0, 2.0], search="BVH", log=False)
        coupling.set_solver(
            {
                "Timestep": 0.001,
                "SimulationTime": 0.002,
                "SaveInterval": 0.001,
                "SavePath": str(tmp_path / "coupled"),
                "assemble_type": "HashTriplet",
                "linear_solver": "PCG",
            },
            log=False,
        )
        coupling.add_surface()
        coupling.memory_allocate(
            {
                "max_contact_pairs": 128,
                "max_point_triangle_pairs": 128,
                "max_edge_edge_pairs": 128,
                "contact_coordination_number": 16,
            }
        )
        coupling.choose_contact_model("IPC", dhat=0.1, kappa=100.0, friction_coefficient=0.0)
        result = coupling.run(verbose=False)
        assert result["converged"]
        boundary = coupling.enginer.wall_boundary
        assert boundary.fem_force.to_numpy()[:, 0].sum() < 0.0
    assert boundary is not None
    assert boundary.abd_force.to_numpy()[:, 0].sum() < 0.0
    assert dem.scene.servo[0].current_force > 0.0
    assert np.all(np.isfinite(boundary.vertex1.to_numpy()))
    assert np.any(boundary.vertex1.to_numpy()[:, 0] != 0.0)
