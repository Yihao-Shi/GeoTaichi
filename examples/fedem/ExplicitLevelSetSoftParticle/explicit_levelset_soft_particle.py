"""Explicit TET4 soft particle colliding with a rigid LSDEM level set.

The FEM body does not build an SDF.  Its current boundary nodes query the
rigid LSDEM sphere's ``gapn`` and exchange force/torque through the selected
DEM contact law.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument("--search", default="BVH")
parser.add_argument("--impact-speed", type=float, default=1.0)
parser.add_argument("--dt", type=float, default=2.0e-5)
parser.add_argument("--time", type=float, default=2.0e-2)
parser.add_argument("--save-interval", type=float, default=1.0e-3)
parser.add_argument(
    "--mesh-path",
    default=str(ROOT / "assets" / "mesh" / "AffineBody" / "lowpoly_sphere.obj"),
)
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData"),
)
arguments = parser.parse_args()

# DEM's compile-time field aliases must match the f64 FEM/contact runtime.
os.environ["GEOTAICHI_REAL_DTYPE"] = arguments.default_fp

import geotaichi as gt

SPHERE_MESH = Path(arguments.mesh_path).expanduser().resolve()

gt.init(
    arch=arguments.arch,
    default_fp=arguments.default_fp,
    log=True,
    debug=False,
    offline_cache=False,
)

# The level-set body is rigid; only this side of the coupling owns an SDF.
dem = gt.DEM(log=True)
dem.set_configuration(
    domain=[1.0, 1.0, 1.2],
    scheme="LSDEM",
    engine="VelocityVerlet",
    search="LinkedCell",
    gravity=[0.0, 0.0, 0.0],
    log=True,
)
dem.memory_allocate(
    {
        "max_material_number": 1,
        "max_rigid_body_number": 1,
        "levelset_grid_number": 16000,
        "surface_node_number": 64,
        "max_sphere_number": 0,
        "max_clump_number": 0,
        "max_plane_number": 0,
        "body_coordination_number": 4,
        "wall_coordination_number": 1,
        "verlet_distance_multiplier": [0.1, 0.1],
        "compaction_ratio": [1.0, 1.0],
    },
    log=True,
)
dem.add_attribute(
    materialID=0,
    attribute={
        "Density": 2500.0,
        "ForceLocalDamping": 0.0,
        "TorqueLocalDamping": 0.0,
    },
)
dem.add_template(
    {
        "Name": "levelset_sphere",
        "Object": gt.polyhedron(file=str(SPHERE_MESH)).grids(space=0.2, extent=2),
        "WriteFile": False,
    }
)
dem.create_body(
    {
        "BodyType": "RigidBody",
        "Template": [
            {
                "Name": "levelset_sphere",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.5, 0.5, 0.30],
                "ScaleFactor": 0.11,
                "InitialVelocity": [0.0, 0.0, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "FixMotion": ["Fix", "Fix", "Fix"],
                "BodyOrientation": "constant",
            }
        ],
    }
)
# There is one rigid body and no DEM wall, so child DEM self-contact is idle.
dem.choose_contact_model(None, None)
dem.select_save_data(particle=True, surface=True)

fem = gt.FEM(log=True)
fem.set_configuration(dimension=3, solver_type="Explicit")
soft = fem.add_soft_particle(
    fem.create_mesh(
        "box",
        origin=[0.32, 0.32, 0.485],
        size=[0.36, 0.36, 0.30],
        divisions=[2, 2, 2],
        element_type="TET4",
    )
)
fem.add_material(
    "NeoHookean",
    density=1100.0,
    young_modulus=2.0e5,
    poisson_ratio=0.3,
)
initial_velocity = np.zeros_like(soft.points)
initial_velocity[:, 2] = -arguments.impact_speed

coupling = gt.FEDEM(dem=dem, fem=fem, log=True)
coupling.set_configuration(
    domain=[1.0, 1.0, 1.2],
    gravity=[0.0, 0.0, 0.0],
    search=arguments.search,
    log=True,
)
coupling.set_solver(
    {
        "Timestep": arguments.dt,
        "SimulationTime": arguments.time,
        "SaveInterval": arguments.save_interval,
        "SavePath": arguments.output_dir,
        "initial_velocity": initial_velocity,
    },
    log=True,
)
coupling.add_surface(body_ids=[0])
coupling.memory_allocate(
    {
        "contact_coordination_number": 8,
        "max_contact_pairs": 128,
        "max_levelset_cell_pairs": 4096,
        # LSDEM AABBs are rebuilt every step, so this is only the search skin.
        "verlet_distance_multiplier": 0.1,
    }
)
coupling.choose_contact_model("Linear")
coupling.add_property(
    DEMmaterial=0,
    FEMbody=0,
    property={
        "NormalStiffness": 1.0e6,
        "TangentialStiffness": 5.0e5,
        "Friction": 0.35,
        "NormalViscousDamping": 0.1,
        "TangentialViscousDamping": 0.1,
    },
)
coupling.select_save_data(contact=True)
coupling.run()
