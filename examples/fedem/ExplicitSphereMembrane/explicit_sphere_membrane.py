"""DEM spheres accumulating on a fixed-boundary explicit FEM membrane."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument("--side-count", type=int, default=1)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--radius", type=float, default=0.1)
parser.add_argument("--membrane-divisions", type=int, default=12)
parser.add_argument("--dt", type=float, default=2.0e-6)
parser.add_argument("--time", type=float, default=0.02)
parser.add_argument("--save-interval", type=float, default=2.0e-4)
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData"),
)
arguments = parser.parse_args()
if (
    arguments.side_count <= 0
    or arguments.layers <= 0
    or arguments.radius <= 0.0
    or arguments.membrane_divisions <= 0
    or arguments.dt <= 0.0
    or arguments.time <= 0.0
    or arguments.save_interval <= 0.0
):
    parser.error("counts, radius, timestep, duration, and save interval must be positive")

import geotaichi as gt

gt.init(arch=arguments.arch, default_fp=arguments.default_fp, log=True)

dem = gt.DEM(log=True)
dem.set_configuration(
    domain=[2.0, 2.0, 2.0],
    boundary=["Reflect", "Reflect", "Reflect"],
    gravity=[0.0, 0.0, -9.81],
    engine="SymplecticEuler",
    search="LinkedCell",
    log=True,
)
particle_count = arguments.side_count * arguments.side_count * arguments.layers
dem.memory_allocate(
    {
        "max_material_number": 1,
        "max_particle_number": particle_count,
        "max_sphere_number": particle_count,
        "max_clump_number": 0,
        "verlet_distance_multiplier": 0.1,
    },
    log=True,
)
dem.add_attribute(
    materialID=0,
    attribute={
        "Density": 1000.0,
        "ForceLocalDamping": 0.0,
        "TorqueLocalDamping": 0.0,
    },
)
span = min(1.1, 2.4 * arguments.radius * max(1, arguments.side_count - 1))
coordinates = (
    [0.75] if arguments.side_count == 1 else np.linspace(0.75 - 0.5 * span, 0.75 + 0.5 * span, arguments.side_count)
)
templates = []
for layer in range(arguments.layers):
    elevation = 0.10 + arguments.radius + layer * 2.2 * arguments.radius
    for x in coordinates:
        for y in coordinates:
            templates.append(
                {
                    "GroupID": 0,
                    "MaterialID": 0,
                    "InitialVelocity": [0.0, 0.0, -0.1],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "BodyPoint": [float(x), float(y), elevation],
                    "FixVelocity": ["Free", "Free", "Free"],
                    "FixAngularVelocity": ["Free", "Free", "Free"],
                    "Radius": arguments.radius,
                    "BodyOrientation": "uniform",
                }
            )
dem.create_body(
    {
        "BodyType": "Sphere",
        "Template": templates,
    }
)
dem.choose_contact_model(None, None)
dem.select_save_data(sphere=True)

fem = gt.FEM(log=True)
fem.set_configuration(dimension=3, solver_type="Explicit")
mesh = fem.add_mesh(
    {
        "Geometry": "Rectangle",
        "Size": (1.5, 1.5),
        "Divisions": (arguments.membrane_divisions, arguments.membrane_divisions),
        "ElementType": "TRI3",
    }
)
fem.add_material(
    "StVK",
    density=1000.0,
    young_modulus=1.0e6,
    poisson_ratio=0.3,
    thickness=0.02,
)
boundary_nodes = sorted(
    set(mesh.node_sets["xmin"])
    | set(mesh.node_sets["xmax"])
    | set(mesh.node_sets["ymin"])
    | set(mesh.node_sets["ymax"])
)
fem.add_boundary_condition(
    {
        "type": "Dirichlet",
        "nodes": boundary_nodes,
        "components": "all",
        "value": 0.0,
    }
)

coupling = gt.FEDEM(dem=dem, fem=fem, log=True)
coupling.set_configuration(
    domain=[2.0, 2.0, 2.0],
    gravity=[0.0, 0.0, -9.81],
    search="BVH",  # "LinkedCell" is also supported
    log=True,
)
coupling.set_solver(
    {
        "Timestep": arguments.dt,
        "SimulationTime": arguments.time,
        "SaveInterval": arguments.save_interval,
        "SavePath": arguments.output_dir,
    },
    log=True,
)
coupling.add_surface(modifier={"Orientation": "Parallel", "Direction": [0.0, 0.0, 1.0]})
coupling.memory_allocate(
    {
        "contact_coordination_number": 64,
        "max_contact_pairs": max(64, 64 * particle_count),
        "max_facet_cell_pairs": max(100000, 256 * particle_count),
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
        "Friction": 0.3,
        "NormalViscousDamping": 0.1,
        "TangentialViscousDamping": 0.1,
    },
)
coupling.select_save_data(contact=True)
coupling.run()
