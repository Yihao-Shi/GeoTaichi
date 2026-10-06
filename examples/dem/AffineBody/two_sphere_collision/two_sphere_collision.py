import os
import sys

from pathlib import Path

import numpy as np
import taichi as ti
from taichi.lang.impl import current_cfg

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
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
body_orientation = [
    float(value)
    for value in os.environ.get(
        "GT_AFFINE_BODY_ORIENTATION",
        # Align a three-fold icosahedron face-normal axis with +x.
        "-75.0,35.26438968,-135.0",
    ).split(",")
]
if len(body_orientation) != 3:
    raise ValueError("GT_AFFINE_BODY_ORIENTATION must contain three Euler angles")
search = os.environ.get("GT_AFFINE_SEARCH", "LinkedCell")
default_assemble_type = "HashTriplet" if current_cfg().arch == ti.cuda else "COO"

dem = DEM(log=False)
dem.set_configuration(
    domain=[3.0, 1.5, 1.5],
    scheme="AffineBody",
    search=search,
    gravity=[0.0, 0.0, 0.0],
    visualize=False,
    log=False,
)
dem.set_affine_body_parameters(
    assemble_type=os.environ.get("GT_AFFINE_ASSEMBLE_TYPE", default_assemble_type),
    young_modulus=5.0e5,
    max_newton_iteration=int(os.environ.get("GT_IPC_MAX_NEWTON", "8")),
    line_search_max_iteration=10,
    direct_hessian_dofs=128,
    max_step=0.02,
)
memory = {
    "max_material_number": 1,
    "max_affine_body_number": 2,
    "surface_node_number": int(os.environ.get("GT_AFFINE_SURFACE_NODE_NUMBER", "24")),
    "body_coordination_number": 8,
    "wall_coordination_number": 0,
    "affine_contact_block_capacity": int(os.environ.get("GT_AFFINE_CONTACT_BLOCK_CAPACITY", "256")),
    "compaction_ratio": [1.0, 1.0],
}
if search == "HierarchicalLinkedCell":
    memory["hierarchical_size"] = [0.08, 0.16]
dem.memory_allocate(memory=memory, log=False)
dem.set_solver(
    {
        "Timestep": float(os.environ.get("GT_AFFINE_DT", "2.0e-4")),
        "SimulationTime": float(os.environ.get("GT_AFFINE_SIM_TIME", "1.6e-1")),
        "SaveInterval": float(os.environ.get("GT_AFFINE_SAVE_INTERVAL", "4.0e-3")),
        "SavePath": os.environ.get("GT_AFFINE_SAVE_PATH", "AffineTwoSphereCollision"),
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
        "Object": polyhedron(file=str(mesh_path)).reset(False),
    }
)
dem.create_body(
    body={
        "BodyType": "AffineBody",
        "Template": [
            {
                "Name": "sphere",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [1.25, 0.75, 0.75],
                "BoundingRadius": 0.18,
                "BodyOrientation": body_orientation,
                "InitialVelocity": [0.5, 0.0, 0.0],
            },
            {
                "Name": "sphere",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [1.62, 0.75, 0.75],
                "BoundingRadius": 0.18,
                "BodyOrientation": body_orientation,
                "InitialVelocity": [-0.5, 0.0, 0.0],
            },
        ],
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
