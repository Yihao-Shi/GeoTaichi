import os
import sys

import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import MPM, init


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_LANDSLIDE_DT", "5.0e-5"))
SIMULATION_TIME = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_LANDSLIDE_TIME", "0.25"))
SAVE_INTERVAL = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_LANDSLIDE_SAVE_INTERVAL", "0.01"))
SAVE_PATH = os.environ.get(
    "GEOTAICHI_DOUBLE_POINT_LANDSLIDE_SAVE_PATH",
    os.path.join(SCRIPT_DIR, "DoublePointSubmarineLandslide"),
)
SHAPE_FUNCTION = os.environ.get("GEOTAICHI_DOUBLE_POINT_LANDSLIDE_SHAPE", "QuadBSpline")
VELOCITY_PROJECTION = os.environ.get("GEOTAICHI_DOUBLE_POINT_LANDSLIDE_VELOCITY_PROJECTION", "Affine")
ALPHA_PIC = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_LANDSLIDE_ALPHA_PIC", "1.0"))
DOMAIN = [2.0, 0.8]
ELEMENT_SIZE = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_LANDSLIDE_ELEMENT_SIZE", "0.04"))
WALL_THICKNESS = int(os.environ.get("GEOTAICHI_DOUBLE_POINT_LANDSLIDE_WALL_CELLS", "1"))
WALL_OFFSET = WALL_THICKNESS * ELEMENT_SIZE
WATER_HEIGHT = 0.50
SOIL_ORIGIN = [0.25, WALL_OFFSET]
SOIL_SIZE = [0.60, 0.20]


init(dim=2, arch="cpu", cpu_max_num_threads=4)

mpm = MPM()

mpm.set_configuration(
    domain=DOMAIN,
    background_damping=0.0,
    gravity=[0.0, -9.8],
    alphaPIC=ALPHA_PIC,
    mapping="USL",
    shape_function=SHAPE_FUNCTION,
    material_type="TwoPhaseDoubleLayer",
    solver_type="SemiImplicit",
    velocity_projection=VELOCITY_PROJECTION,
)

mpm.set_solver(
    solver={
        "Timestep": DT,
        "SimulationTime": SIMULATION_TIME,
        "SaveInterval": SAVE_INTERVAL,
        "SavePath": SAVE_PATH,
    }
)

mpm.set_semi_implicit_solver_parameters(
    {
        "assemble_type": "MatrixFree",
        "pressure_solver": "MGPCG",
        "linear_solver": "MGPCG",
        "max_iteration_number": 120,
        "residual_tolerance": 1.0e-8,
        "multilevel": 1,
        "pre_and_post_smoothing": 2,
        "bottom_smoothing": 8,
    }
)

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 20000,
        "max_constraint_number": {"max_velocity_constraint": 16000},
    }
)

mpm.add_material(
    model="MohrCoulomb",
    material={
        "MaterialID": 1,
        "SolidDensity": 2650.0,
        "FluidDensity": 1000.0,
        "Porosity": 0.42,
        "FluidBulkModulus": 2.2e8,
        "Permeability": 5.0e-4,
        "FluidViscosity": 1.0e-3,
        "GrainDiameter": 2.0e-2,
        "YoungModulus": 2.0e7,
        "PoissonRatio": 0.30,
        "Cohesion": 500.0,
        "Friction": 28.0,
        "Dilation": 0.0,
    },
)

mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [ELEMENT_SIZE, ELEMENT_SIZE]})


mpm.add_region(
    region=[
        {
            "Name": "water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [WALL_OFFSET, WALL_OFFSET],
            "BoundingBoxSize": [DOMAIN[0] - 2.0 * WALL_OFFSET, WATER_HEIGHT],
        },
        {
            "Name": "soil_block",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": SOIL_ORIGIN,
            "BoundingBoxSize": SOIL_SIZE,
        },
    ]
)

mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "water",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Fluid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            },
            {
                "RegionName": "soil_block",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Solid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            },
        ]
    }
)

mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "StartPoint": [WALL_OFFSET, WALL_OFFSET],
            "EndPoint": [DOMAIN[0] - WALL_OFFSET, WALL_OFFSET],
            "Norm": [0.0, -1.0],
            "CellThickness": WALL_THICKNESS,
        },
        {
            "BoundaryType": "SolidCell",
            "StartPoint": [WALL_OFFSET, WALL_OFFSET],
            "EndPoint": [WALL_OFFSET, DOMAIN[1]],
            "Norm": [-1.0, 0.0],
            "CellThickness": WALL_THICKNESS,
        },
        {
            "BoundaryType": "SolidCell",
            "StartPoint": [DOMAIN[0] - WALL_OFFSET, WALL_OFFSET],
            "EndPoint": [DOMAIN[0] - WALL_OFFSET, DOMAIN[1]],
            "Norm": [1.0, 0.0],
            "CellThickness": WALL_THICKNESS,
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, 0.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [DOMAIN[0], WALL_OFFSET],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [WALL_OFFSET, DOMAIN[1]],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [DOMAIN[0] - WALL_OFFSET, 0.0],
            "EndPoint": [DOMAIN[0], DOMAIN[1]],
        },
    ]
)

mpm.select_save_data(particle=True, grid=False, object=False)
mpm.run()
