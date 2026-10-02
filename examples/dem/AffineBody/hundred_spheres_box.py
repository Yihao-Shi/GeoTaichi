import os
import sys

from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import DEM, init, polyhedron


init(arch=os.environ.get("GT_ARCH", "gpu"), log=False, debug=False, offline_cache=False)
mesh_path = Path(os.environ.get("GT_AFFINE_MESH", str(Path(ROOT) / "assets/mesh/AffineBody/lowpoly_sphere.obj")))
search_mode = os.environ.get("GT_AFFINE_SEARCH", "LinkedCell")
assemble_type = os.environ.get("GT_AFFINE_ASSEMBLE_TYPE", "COO")
body_number = int(os.environ.get("GT_AFFINE_BODY_NUMBER", "100"))
radius = float(os.environ.get("GT_AFFINE_RADIUS", "0.045"))
radius_spread = float(os.environ.get("GT_AFFINE_RADIUS_SPREAD", "0.005"))
dt = float(os.environ.get("GT_AFFINE_DT", "2.0e-4"))
simulation_time = float(os.environ.get("GT_AFFINE_SIM_TIME", "6.0e-2"))
save_interval = float(os.environ.get("GT_AFFINE_SAVE_INTERVAL", "1.5e-2"))
save_path = os.environ.get("GT_AFFINE_SAVE_PATH", "AffineHundredSpheresBox")
initial_down_velocity = float(os.environ.get("GT_AFFINE_INITIAL_DOWN_VELOCITY", "3.0"))
contact_damping_stiffness = float(os.environ.get("GT_AFFINE_CONTACT_DAMPING", "0.5"))

dem = DEM(log=False)
dem.set_configuration(
    domain=[1.4, 1.4, 1.6],
    scheme="AffineBody",
    search=search_mode,
    gravity=[0.0, 0.0, -9.8],
    visualize=False,
    log=False,
)
dem.set_affine_body_parameters(
    assemble_type=assemble_type,
    young_modulus=2.0e5,
    max_newton_iteration=2,
    linear_tolerance=float(os.environ.get("GT_AFFINE_LINEAR_TOLERANCE", "1.0e-5")),
    linear_max_iteration=int(os.environ.get("GT_AFFINE_LINEAR_MAX_ITERATION", "1000")),
    line_search_max_iteration=8,
    direct_hessian_dofs=96,
    max_step=0.01,
)
memory = {
    "max_material_number": 1,
    "max_affine_body_number": body_number,
    "surface_node_number": body_number * 128,
    "max_plane_number": 6,
    "body_coordination_number": int(os.environ.get("GT_AFFINE_BODY_COORDINATION", "32")),
    "wall_coordination_number": 6,
    "wall_per_cell": int(os.environ.get("GT_AFFINE_WALL_PER_CELL", "64")),
    "compaction_ratio": [1.0, 1.0],
}
if search_mode == "HierarchicalLinkedCell":
    sizes = [float(v) for v in os.environ.get("GT_AFFINE_HIERARCHICAL_SIZE", "0.04,0.08").split(",")]
    memory["hierarchical_level"] = len(sizes)
    memory["hierarchical_size"] = sizes
dem.memory_allocate(memory=memory, log=False)
dem.set_solver(
    {
        "Timestep": dt,
        "SimulationTime": simulation_time,
        "SaveInterval": save_interval,
        "SavePath": save_path,
    },
    log=False,
)
dem.add_attribute(materialID=0, attribute={"Density": 1200.0})
dem.add_template(template={"Name": "sphere", "Object": polyhedron(file=str(mesh_path))})

dem.add_region(
    region={
        "Name": "drop_region",
        "Type": "Rectangle",
        "BoundingBoxPoint": [0.22, 0.22, 0.16],
        "BoundingBoxSize": [0.96, 0.96, 0.78],
    }
)

dem.add_body(
    body={
        "GenerateType": "Generate",
        "RegionName": "drop_region",
        "BodyType": "RigidBody",
        "PoissonSampling": False,
        "TryNumber": int(os.environ.get("GT_AFFINE_TRY_NUMBER", "5000")),
        "Template": {
            "Name": "sphere",
            "GroupID": 0,
            "MaterialID": 0,
            "BodyNumber": body_number,
            "MinRadius": max(radius - radius_spread, 1.0e-6),
            "MaxRadius": radius + radius_spread,
            "BodyOrientation": "uniform",
            "InitialVelocity": [0.0, 0.0, -initial_down_velocity],
            "Friction": 0.3,
        },
    }
)

walls = [
    ([0.0, 0.0, 0.05], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 1.5], [0.0, 0.0, -1.0]),
    ([0.05, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ([1.35, 0.0, 0.0], [-1.0, 0.0, 0.0]),
    ([0.0, 0.05, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.35, 0.0], [0.0, -1.0, 0.0]),
]
for center, normal in walls:
    dem.add_wall(
        {
            "WallType": "Plane",
            "MaterialID": 0,
            "WallCenter": np.array(center),
            "OuterNormal": np.array(normal),
        }
    )

dem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "Dhat": 0.02,
        "BarrierStiffness": 2.0e5,
        "ContactDampingStiffness": contact_damping_stiffness,
        "Friction": 0.3,
    },
    dType="all",
)
dem.run()
