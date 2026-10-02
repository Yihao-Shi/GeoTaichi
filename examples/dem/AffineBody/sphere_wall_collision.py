import os
import sys

from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import DEM, init, polyhedron

init(
    arch=os.environ.get("GT_ARCH", "gpu"),
    default_fp=os.environ.get("GT_DEFAULT_FP", "float64"),
    log=False,
    debug=False,
    offline_cache=bool(int(os.environ.get("GT_OFFLINE_CACHE", "1"))),
)
mesh_path = Path(os.environ.get("GT_AFFINE_MESH", str(Path(ROOT) / "assets/mesh/AffineBody/lowpoly_sphere.obj")))

dem = DEM(log=False)
dem.set_configuration(
    domain=[2.0, 1.5, 1.5],
    scheme="AffineBody",
    search=os.environ.get("GT_AFFINE_SEARCH", "LinkedCell"),
    gravity=[0.0, 0.0, 0.0],
    visualize=False,
    log=False,
)
dem.set_affine_body_parameters(
    assemble_type=os.environ.get("GT_AFFINE_ASSEMBLE_TYPE", "MatrixFree"),
    young_modulus=5.0e5,
    max_newton_iteration=int(os.environ.get("GT_IPC_MAX_NEWTON", "8")),
    line_search_max_iteration=10,
    direct_hessian_dofs=128,
    max_step=0.02,
)
dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_affine_body_number": 1,
        "surface_node_number": int(os.environ.get("GT_AFFINE_SURFACE_NODE_NUMBER", "12")),
        "max_plane_number": 1,
        "body_coordination_number": 8,
        "wall_coordination_number": 4,
        "affine_contact_block_capacity": int(os.environ.get("GT_AFFINE_CONTACT_BLOCK_CAPACITY", "256")),
        "compaction_ratio": [1.0, 1.0],
    },
    log=False,
)
dem.set_solver(
    {
        "Timestep": float(os.environ.get("GT_AFFINE_DT", "2.0e-4")),
        "SimulationTime": float(os.environ.get("GT_AFFINE_SIM_TIME", "1.6e-1")),
        "SaveInterval": float(os.environ.get("GT_AFFINE_SAVE_INTERVAL", "4.0e-3")),
        "SavePath": os.environ.get("GT_AFFINE_SAVE_PATH", "AffineSphereWallCollision"),
    },
    log=False,
)
dem.add_attribute(
    materialID=0,
    attribute={
        "Density": 1200.0,
        "ForceLocalDamping": 0.0,
        "TorqueLocalDamping": 0.0,
    },
)
dem.add_template(
    template={
        "Name": "sphere",
        "TemplateType": "AffineBody",
        "Object": polyhedron(file=str(mesh_path)),
    }
)
dem.create_body(
    body={
        "BodyType": "AffineBody",
        "Template": {
            "Name": "sphere",
            "GroupID": 0,
            "MaterialID": 0,
            "BodyPoint": [0.35, 0.75, 0.75],
            "BoundingRadius": 0.18,
            "InitialVelocity": [-0.5, 0.0, 0.0],
        },
    }
)
dem.add_wall(
    {
        "WallType": "Plane",
        "MaterialID": 0,
        "WallCenter": np.array([0.15, 0.0, 0.0]),
        "OuterNormal": np.array([1.0, 0.0, 0.0]),
    }
)
dem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "Dhat": 0.04,
        "BarrierStiffness": 5.0e5,
        "ContactDampingStiffness": float(os.environ.get("GT_AFFINE_CONTACT_DAMPING", "0.0")),
        "Friction": float(os.environ.get("GT_AFFINE_FRICTION", "0.0")),
    },
    dType="all",
)
dem.run()
