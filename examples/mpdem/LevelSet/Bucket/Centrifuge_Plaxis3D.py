import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Run the Plaxis-derived centrifuge bucket example.")
parser.add_argument(
    "--bucket-particles",
    default=os.path.join(ROOT, "assets", "data", "MPDEM", "Bucket", "bucket.txt"),
    help="TXT file containing the bucket particles.",
)
parser.add_argument(
    "--horizontal-load-particles",
    default=os.path.join(ROOT, "assets", "data", "MPDEM", "Bucket", "horizontal_load.txt"),
    help="TXT file containing the loaded bucket particles.",
)
parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData" / "centrifuge"))
arguments = parser.parse_args()
bucket_particles = str(Path(arguments.bucket_particles).expanduser().resolve())
horizontal_load_particles = str(Path(arguments.horizontal_load_particles).expanduser().resolve())

from geotaichi import *

init(arch="gpu", log=True, debug=False, device_memory_GB=6, kernel_profiler=False)
# init(arch='cpu', cpu_max_num_threads=1, debug=False)

mpm = MPM()

mpm.set_configuration(
    domain=ti.Vector([144.0, 46.0, 73.0]),
    background_damping=0.0,
    gravity=ti.Vector([0.0, 0.0, -9.81]),
    alphaPIC=0.00,
    mapping="USF",
    shape_function="QuadBSpline",
    # stabilize="F-Bar Method"
)

mpm.set_solver(solver={"Timestep": 1e-5, "SimulationTime": 1, "SaveInterval": 0.02, "SavePath": arguments.output_dir})

mpm.memory_allocate(
    memory={
        "max_material_number": 7,
        "max_particle_number": 480081,
        "max_constraint_number": {
            "max_velocity_constraint": 12434,
            "max_particle_traction_constraint": 20615,
            "max_friction_constraint": 10,
        },
    }
)

mpm.add_material(
    model="MohrCoulomb",
    material={
        "MaterialID": 1,
        "Density": 1173.0,
        "YoungModulus": 3.0e6,
        "poissonRatio": 0.2,  # equivalent=0.4950
        "Cohesion": 12000,
        "Friction": 15,
        "Dilation": 0.0,
    },
)  # 海侧碎石桩

mpm.add_material(
    model="MohrCoulomb",
    material={
        "MaterialID": 2,
        "Density": 1100.0,
        "YoungModulus": 2.5e6,
        "poissonRatio": 0.20,
        "Cohesion": 11000,
        "Friction": 15,
        "Dilation": 0.0,
    },
)  # 陆侧淤泥质粘土

mpm.add_material(
    model="MohrCoulomb",
    material={
        "MaterialID": 3,
        "Density": 1558.0,
        "YoungModulus": 5.0e6,
        "poissonRatio": 0.20,
        "Cohesion": 33000,
        "Friction": 17,
        "Dilation": 0.0,
    },
)  # 粉质粘土

mpm.add_material(
    model="MohrCoulomb",
    material={
        "MaterialID": 4,
        "Density": 1865.0,
        "YoungModulus": 2.0e7,
        "poissonRatio": 0.20,
        "Cohesion": 8000,
        "Friction": 35,
        "Dilation": 0.0,
    },
)  # 砂土

mpm.add_material(
    model="MohrCoulomb",
    material={
        "MaterialID": 5,
        "Density": 1865.0,
        "YoungModulus": 2.0e8,
        "poissonRatio": 0.20,
        "Cohesion": 150000,
        "Friction": 20,
        "Dilation": 0.0,
    },
)  # 回填土

mpm.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 6,
        "Density": 2650.0,
        "YoungModulus": 5.0e8,
        "poissonRatio": 0.30,
    },
)  # 桶

mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": ti.Vector([2.0, 2.0, 2.0])})

mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
            "BoundingBoxSize": ti.Vector([144.0, 46.0, 5.0]),
            "zdirection": ti.Vector([0.0, 0.0, 1.0]),
        },
        {
            "Name": "region2",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 5.0]),
            "BoundingBoxSize": ti.Vector([144.0, 46.0, 26.0]),
            "zdirection": ti.Vector([0.0, 0.0, 1.0]),
        },
        {
            "Name": "region3",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 31.0]),
            "BoundingBoxSize": ti.Vector([72.0, 46.0, 30.0]),
            "zdirection": ti.Vector([0.0, 0.0, 1.0]),
        },
        {
            "Name": "region4",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([72.0, 0.0, 31.0]),
            "BoundingBoxSize": ti.Vector([72.0, 46.0, 30.0]),
            "zdirection": ti.Vector([0.0, 0.0, 1.0]),
        },
        {
            "Name": "region5",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 60.0]),
            "BoundingBoxSize": ti.Vector([72.0, 46.0, 1.0]),
            "zdirection": ti.Vector([0.0, 0.0, 1.0]),
        },
        {
            "Name": "region6",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([72.0, 0.0, 60.0]),
            "BoundingBoxSize": ti.Vector([72.0, 46.0, 1.0]),
            "zdirection": ti.Vector([0.0, 0.0, 1.0]),
        },
    ]
)

mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "region1",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 4,
                "Traction": [],
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "FixVelocity": ["Free", "Free", "Free"],
            },
            {
                "RegionName": "region2",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 3,
                "Traction": [],
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "FixVelocity": ["Free", "Free", "Free"],
            },
            {
                "RegionName": "region3",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "ParticleStress": {
                    "GravityField": False,
                    "InternalStress": ti.Vector([-1.0e5, -1.0e5, -1.0e5, 0.0, 0.0, 0.0]),
                    "PorePressure": 0.0,
                },
                "Traction": [
                    {
                        "Pressure": ti.Vector([0, 0.0, -1.0e5]),
                        "FluidPressure": ti.Vector([0, 0.0, 0.0]),
                        "RegionName": "region5",
                    }
                ],
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "FixVelocity": ["Free", "Free", "Free"],
            },
            {
                "RegionName": "region4",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 2,
                "ParticleStress": {
                    "GravityField": False,
                    "InternalStress": ti.Vector([-1.0e5, -1.0e5, -1.0e5, 0.0, 0.0, 0.0]),
                    "PorePressure": 0.0,
                },
                "Traction": [
                    {
                        "Pressure": ti.Vector([0, 0.0, -1.0e5]),
                        "FluidPressure": ti.Vector([0, 0.0, 0.0]),
                        "RegionName": "region6",
                    }
                ],
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "FixVelocity": ["Free", "Free", "Free"],
            },
        ]
    }
)  # 回填土的模型没建立，本构参数是materialID5,表面加竖向荷载100需要扣掉桶（桶的厚度为0.016m)和桶内回填土，水平荷载只加在海侧（桶的左边）

mpm.add_body_from_file(
    body={
        "FileType": "TXT",
        # "CheckHistory": True,
        "Template": {
            "ParticleFile": bucket_particles,
            "nParticlesPerCell": 2,
            "BodyID": 0,
            "MaterialID": 6,
            "InitialVelocity": ti.Vector([0.0, 0.0, 0]),
            "FixVelocity": ["Free", "Free", "Free"],
        },
    }
)

mpm.add_body_from_file(
    body={
        "FileType": "TXT",
        "Template": {
            "ParticleFile": horizontal_load_particles,
            "nParticlesPerCell": 2,
            "BodyID": 0,
            "MaterialID": 6,
            "Traction": [
                {
                    "Pressure": ti.Vector([6.0e4, 0.0, 0.0]),
                }
            ],
            "InitialVelocity": ti.Vector([0.0, 0.0, 0]),
            "FixVelocity": ["Free", "Free", "Free"],
        },
    }
)

mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0, 0.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [144.0, 46.0, 0.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None, None],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [0.0, 46.0, 73.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None, None],
            "StartPoint": [144.0, 0.0, 0.0],
            "EndPoint": [144.0, 46.0, 73.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, 0.0, None],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [144.0, 0.0, 73.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, 0.0, None],
            "StartPoint": [0.0, 46.0, 0.0],
            "EndPoint": [144.0, 46.0, 73.0],
        },
    ]
)


def get_gravity(points):
    return 60.0 - points[:, 2]


mpm.select_save_data()

mpm.run(gravity_field=get_gravity)

mpm.postprocessing(write_background_grid=False, read_path="1_centrifuge")
