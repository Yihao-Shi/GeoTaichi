import os
import sys

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

SAVE_PATH = os.environ.get(
    "GEOTAICHI_LARGE_TANK_2D_SAVE_PATH",
    os.path.join(os.path.dirname(__file__), "OutputData", "large_tank_incompressible_2d"),
)

init(
    dim=2,
    arch=os.environ.get("GEOTAICHI_ARCH", "cpu"),
    default_fp="float64",
    debug=False,
    kernel_profiler=True,
)

mpm = MPM()

mpm.set_configuration(
    domain=[0.5, 0.3],
    background_damping=0.00,
    alphaPIC=0.5,
    mapping="USL",
    shape_function="QuadBSpline",
    gravity=[0.0, -9.8],
    material_type="Fluid",
    solver_type="Implicit",
    discretization="FDM",
    velocity_projection="Affine",
    particle_shifting=True,
    density_projection=True,
)

mpm.set_implicit_solver_parameters(linear_solver="MGPCG")

mpm.set_solver({"Timestep": 5e-4, "SimulationTime": 1, "SaveInterval": 0.01, "SavePath": SAVE_PATH})

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 80000,
        "dof_multiplier": 3,
        "max_constraint_number": {
            "max_reflection_constraint": 0,
            "max_friction_constraint": 0,
            "max_velocity_constraint": 1500,
        },
    }
)

mpm.add_material(
    model="Newtonian",
    material={
        "MaterialID": 1,
        "Density": 1000.0,
        "Modulus": 2e6,
        "Viscosity": 1e-3,
        "ElementLength": 0.01,
        "cL": 1.5,
        "cQ": 2,
    },
)

mpm.add_element(element={"ElementType": "Staggered", "ElementSize": [0.01, 0.01], "GhostCell": 1})


mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": [0.3, 0.15],
            "rotate2D": 0.0,
        }
    ]
)

mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "region1",
                # Per axis: 5**2 = 25 particles/full cell.
                "nParticlesPerCell": 5,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0, 0, 0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        ]
    }
)

mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, -1.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [0.5, 0.0],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [-1.0, 0.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [0.0, 0.3],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [1.0, 0.0],
            "StartPoint": [0.5, 0.0],
            "EndPoint": [0.5, 0.3],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, 1.0],
            "StartPoint": [0.0, 0.3],
            "EndPoint": [0.5, 0.3],
            "CellThickness": 1,
        },
    ]
)

mpm.select_save_data(grid=True)

mpm.run(gravity_field=True)

ti.profiler.print_kernel_profiler_info()
