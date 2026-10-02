"""A resolved MPM block impacting a deformable explicit HEX8 FEM pad."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument("--dt", type=float, default=1.0e-5)
parser.add_argument("--time", type=float, default=1.5e-2)
parser.add_argument("--save-interval", type=float, default=1.0e-3)
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData" / "explicit_point_membrane"),
)
arguments = parser.parse_args()

import geotaichi as gt

gt.init(
    dim=3,
    arch=arguments.arch,
    default_fp=arguments.default_fp,
    log=True,
)

fem = gt.FEM(log=True)
mpm = gt.MPM(log=True)
model = gt.FEMPM(fem=fem, mpm=mpm, log=True)

mpm.set_configuration(
    domain=[1.0, 1.0, 0.8],
    gravity=[0.0, 0.0, -9.81],
    mapping="USL",
    shape_function="Linear",
    configuration="ULMPM",
    solver_type="Explicit",
    material_type="Solid",
    visualize=True,
    log=True,
)
mpm.memory_allocate(
    {
        "max_material_number": 1,
        # 7 x 7 x 3 cells, with 2^3 points per cell.
        "max_particle_number": 7 * 7 * 3 * 8,
        "max_constraint_number": {},
    },
    log=True,
)
mpm.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 1,
        "Density": 1000.0,
        "YoungModulus": 2.0e5,
        "PoissonRatio": 0.3,
    },
)
mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": [0.04, 0.04, 0.04]})
mpm.add_region(
    region={
        "Name": "block",
        "Type": "Rectangle",
        "BoundingBoxPoint": [0.36, 0.36, 0.12],
        "BoundingBoxSize": [0.28, 0.28, 0.12],
    }
)
mpm.add_body(
    body={
        "Template": {
            "RegionName": "block",
            "nParticlesPerCell": 2,
            "BodyID": 0,
            "MaterialID": 1,
            "InitialVelocity": [0.25, 0.0, -0.2],
            "FixVelocity": ["Free", "Free", "Free"],
        }
    }
)
mpm.add_boundary_condition()
mpm.select_save_data(particle=True, grid=False, object=False)

fem.set_configuration(dimension=3, solver_type="Explicit")
fem_mesh = fem.add_mesh(
    geometry="box",
    origin=(0.1, 0.1, 0.0),
    size=(0.8, 0.8, 0.12),
    divisions=(8, 8, 2),
    element_type="HEX8",
)
fem.add_material(
    "NeoHookean",
    density=1000.0,
    young_modulus=2.0e5,
    poisson_ratio=0.3,
)
fem.add_boundary_condition(
    {
        "type": "Dirichlet",
        "nodes": fem_mesh.node_sets["zmin"],
        "components": "all",
        "value": 0.0,
    }
)

model.set_configuration(
    domain=[1.0, 1.0, 0.8],
    gravity=[0.0, 0.0, -9.81],
    search="BVH",
    log=True,
)
model.set_solver(
    {
        "Timestep": arguments.dt,
        "SimulationTime": arguments.time,
        "SaveInterval": arguments.save_interval,
        "SavePath": arguments.output_dir,
    },
    log=True,
)
model.add_surface()
model.memory_allocate(
    {
        "contact_coordination_number": 16,
    }
)
model.choose_contact_model("Linear")
model.add_property(
    MPMmaterial=1,
    FEMbody=0,
    property={
        "NormalStiffness": 1.0e5,
        "TangentialStiffness": 5.0e4,
        "Friction": 0.2,
        "NormalViscousDamping": 0.0,
        "TangentialViscousDamping": 0.0,
    },
)
model.run()
