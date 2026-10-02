import os
import sys
from pathlib import Path

import argparse

DEFAULT_RANDOM_FIELD = Path(__file__).resolve().parents[3] / "assets/data/MPM/RandomField.txt"
parser = argparse.ArgumentParser(description="RandomFieldSimulator")
parser.add_argument("-path", type=str, default=None, help=f"material field file (bundled: {DEFAULT_RANDOM_FIELD})")
args = parser.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2)

mpm = MPM()

mpm.set_configuration(
    domain=ti.Vector([50.0, 20.0]),
    is_2DAxisy=False,
    background_damping=0.0,
    gravity=ti.Vector([0.0, -10.0]),
    alphaPIC=0.0,
    mapping="USF",
    shape_function="GIMP",
    stabilize="B-Bar Method",
    random_field=True if args.path is not None and os.path.isfile(args.path) else False,
)

mpm.set_solver(solver={"Timestep": 3e-4, "SimulationTime": 5, "SaveInterval": 0.2, "SavePath": "DP2DPlane"})

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 5.12e5,
        "max_constraint_number": {"max_velocity_constraint": 83000, "max_friction_constraint": 83000},
    }
)

if args.path is not None and os.path.isfile(args.path):
    mpm.add_material(model="DruckerPrager", material={"MaterialID": 1, "MaterialFile": args.path})
else:
    mpm.add_material(
        model="DruckerPrager",
        material={
            "MaterialID": 1,
            "Density": 1800.0,
            "YoungModulus": 100e6,
            "PoissonRatio": 0.3,
            "Cohesion": 6700.0,
            "Friction": 20.0,
            "Dilation": 9.0,
            "Tensile": 0.0,
        },
    )

mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": ti.Vector([1.0, 1.0]), "Contact": {}})


def get_gravity(points):
    # return the distance from the current material point to the free surface: y = 1. - 0.5 * x
    import numpy as np

    return np.where(
        points[:, 0] < 20.0,
        20 - points[:, 1],
        np.where(points[:, 0] < 30.0, 40.0 - points[:, 0] - points[:, 1], 10 - points[:, 1]),
    )


mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": [50.0, 10],
        }
    ]
)

mpm.add_region(
    region=[
        {
            "Name": "region2",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 10.0],
            "BoundingBoxSize": [20, 10.0],
        }
    ]
)

mpm.add_region(
    region=[
        {
            "Name": "region3",
            "Type": "Triangle2D",
            "BoundingBoxPoint": [20.0, 10.0],
            "BoundingBoxSize": [10.0, 10.0],
        }
    ]
)

mpm.add_body(
    body={
        "WriteFile": False,
        "Template": [
            {
                "RegionName": "region1",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0, 0],
                "FixVelocity": ["Free", "Free"],
            },
            {
                "RegionName": "region2",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0, 0],
                "FixVelocity": ["Free", "Free"],
            },
            {
                "RegionName": "region3",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0, 0],
                "FixVelocity": ["Free", "Free"],
            },
        ],
    }
)

mpm.add_boundary_condition(
    boundary=[
        {"BoundaryType": "VelocityConstraint", "Velocity": [0.0, 0], "StartPoint": [0.0, 0.0], "EndPoint": [50.0, 0.0]},
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [0.0, 20.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [50.0, 0.0],
            "EndPoint": [50.0, 20.0],
        },
    ]
)

mpm.select_save_data()

mpm.run(mpm_gravity_field=True)

mpm.postprocessing(read_path="DP2DPlane")
