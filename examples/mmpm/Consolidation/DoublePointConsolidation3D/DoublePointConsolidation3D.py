import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.mmpm.Consolidation.DoublePointConsolidation3D.DoublePointConsolidation3D_parameters import (
    BOUNDARY_CELL_THICKNESS,
    CELL_SIZE,
    COLUMN_DEPTH,
    COLUMN_HEIGHT,
    COLUMN_ORIGIN,
    COLUMN_WIDTH,
    DT,
    FLUID_DENSITY,
    PERMEABILITY,
    POISSON_RATIO,
    SAVE_INTERVAL,
    SAVE_PATH,
    SIMULATION_TIME,
    SURCHARGE,
    YOUNG_MODULUS,
)
from examples.mmpm.Consolidation.DoublePointConsolidation3D.draw.evaluate_DoublePointConsolidation3D import (
    postprocess_consolidation,
)


from geotaichi import MPM, init

ARCH = os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_ARCH", "cpu")
DOMAIN = [
    COLUMN_WIDTH + 2.0 * BOUNDARY_CELL_THICKNESS * CELL_SIZE,
    COLUMN_DEPTH + 2.0 * BOUNDARY_CELL_THICKNESS * CELL_SIZE,
    1.2,
]
POROSITY = 0.30
INITIAL_PORE_PRESSURE = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_INITIAL_PRESSURE", str(SURCHARGE)))
MG_LEVELS = int(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_MG_LEVELS", "2"))
MAX_PRESSURE_ITERATIONS = int(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_MAX_PRESSURE_ITERATIONS", "120"))
PRESSURE_RESIDUAL_TOLERANCE = float(
    os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_PRESSURE_TOLERANCE", "1.0e-8")
)
ALPHA_PIC = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_ALPHA_PIC", "1.0"))
SHAPE_FUNCTION = os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_SHAPE", "QuadBSpline")
VELOCITY_PROJECTION = os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_VELOCITY_PROJECTION", "Affine")
DELAYED_FLUID_ADVECTION = os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_DELAYED_FLUID_ADVECTION", "1") != "0"
TOP_LOAD_THICKNESS = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_TOP_LOAD_THICKNESS", str(CELL_SIZE)))
TOP_LOAD_TRACTION_SCALE = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_TOP_LOAD_SCALE", "2.0"))


def adjusted_grid_size(domain_length):
    cell_count = max(1, int(np.floor((1.0 + 1.0e-6) * domain_length / CELL_SIZE)))
    multiplier = 2 ** max(0, MG_LEVELS - 1)
    cell_count = int(multiplier * np.ceil(cell_count / multiplier))
    return domain_length / cell_count


ADJUSTED_GRID_SIZE = np.array([adjusted_grid_size(length) for length in DOMAIN], dtype=np.float64)
ESTIMATED_PARTICLE_NUMBER = int(
    2.0
    * int(np.ceil(COLUMN_WIDTH / ADJUSTED_GRID_SIZE[0]))
    * int(np.ceil(COLUMN_DEPTH / ADJUSTED_GRID_SIZE[1]))
    * int(np.ceil(COLUMN_HEIGHT / ADJUSTED_GRID_SIZE[2]))
    * 2
    * 2
    * 2
)
MAX_PARTICLE_NUMBER = int(
    os.environ.get(
        "GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_MAX_PARTICLES",
        str(max(20000, int(1.25 * ESTIMATED_PARTICLE_NUMBER))),
    )
)


def create_mpm():
    init(dim=3, arch=ARCH, cpu_max_num_threads=4)

    mpm = MPM()
    mpm.set_configuration(
        domain=DOMAIN,
        background_damping=0.0,
        gravity=[0.0, 0.0, 0.0],
        alphaPIC=ALPHA_PIC,
        mapping="USL",
        shape_function=SHAPE_FUNCTION,
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
        velocity_projection=VELOCITY_PROJECTION,
        delayed_fluid_advection=DELAYED_FLUID_ADVECTION,
        particle_traction_update_area=False,
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
            "max_iteration_number": MAX_PRESSURE_ITERATIONS,
            "residual_tolerance": PRESSURE_RESIDUAL_TOLERANCE,
            "multilevel": MG_LEVELS,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": MAX_PARTICLE_NUMBER,
            "max_constraint_number": {
                "max_velocity_constraint": 12000,
                "max_particle_traction_constraint": 4000,
            },
        }
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "SolidDensity": 2650.0,
            "FluidDensity": FLUID_DENSITY,
            "Porosity": POROSITY,
            "FluidBulkModulus": 2.2e8,
            "Permeability": PERMEABILITY,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 1.0e-2,
            "DragModel": "Darcy",
            "YoungModulus": YOUNG_MODULUS,
            "PoissonRatio": POISSON_RATIO,
        },
    )
    mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": [CELL_SIZE, CELL_SIZE, CELL_SIZE]})

    top_load_point = [
        float(COLUMN_ORIGIN[0]),
        float(COLUMN_ORIGIN[1]),
        float(COLUMN_ORIGIN[2] + COLUMN_HEIGHT - TOP_LOAD_THICKNESS),
    ]
    mpm.add_region(
        region=[
            {
                "Name": "column",
                "Type": "Rectangle",
                "BoundingBoxPoint": COLUMN_ORIGIN.tolist(),
                "BoundingBoxSize": [COLUMN_WIDTH, COLUMN_DEPTH, COLUMN_HEIGHT],
            },
            {
                "Name": "top_load",
                "Type": "Rectangle",
                "BoundingBoxPoint": top_load_point,
                "BoundingBoxSize": [COLUMN_WIDTH, COLUMN_DEPTH, TOP_LOAD_THICKNESS],
            },
        ]
    )
    mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "column",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Solid",
                    "ParticleStress": {
                        "InternalStress": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "PorePressure": INITIAL_PORE_PRESSURE,
                    },
                    "Traction": [
                        {
                            "Pressure": [0.0, 0.0, -SURCHARGE / TOP_LOAD_TRACTION_SCALE],
                            "FluidPressure": [0.0, 0.0, 0.0],
                            "RegionName": "top_load",
                        }
                    ],
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
                {
                    "RegionName": "column",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "ParticleStress": {"PorePressure": INITIAL_PORE_PRESSURE},
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
            ]
        }
    )

    x0, y0, z0 = COLUMN_ORIGIN.tolist()
    x1 = x0 + COLUMN_WIDTH
    y1 = y0 + COLUMN_DEPTH
    z1 = z0 + COLUMN_HEIGHT
    eps = 1.0e-8
    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "SolidCell",
                "StartPoint": [x0, y0, z0],
                "EndPoint": [x1, y1, z0],
                "Norm": [0.0, 0.0, -1.0],
                "CellThickness": BOUNDARY_CELL_THICKNESS,
            },
            {
                "BoundaryType": "SolidCell",
                "StartPoint": [x0, y0, z0],
                "EndPoint": [x0, y1, z1],
                "Norm": [-1.0, 0.0, 0.0],
                "CellThickness": BOUNDARY_CELL_THICKNESS,
            },
            {
                "BoundaryType": "SolidCell",
                "StartPoint": [x1, y0, z0],
                "EndPoint": [x1, y1, z1],
                "Norm": [1.0, 0.0, 0.0],
                "CellThickness": BOUNDARY_CELL_THICKNESS,
            },
            {
                "BoundaryType": "SolidCell",
                "StartPoint": [x0, y0, z0],
                "EndPoint": [x1, y0, z1],
                "Norm": [0.0, -1.0, 0.0],
                "CellThickness": BOUNDARY_CELL_THICKNESS,
            },
            {
                "BoundaryType": "SolidCell",
                "StartPoint": [x0, y1, z0],
                "EndPoint": [x1, y1, z1],
                "Norm": [0.0, 1.0, 0.0],
                "CellThickness": BOUNDARY_CELL_THICKNESS,
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, None, 0.0],
                "StartPoint": [x0, y0, 0.0],
                "EndPoint": [x1, y1, z0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [0.0, y0, z0],
                "EndPoint": [x0, y1, z1],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [x1, y0, z0],
                "EndPoint": [DOMAIN[0] - eps, y1, z1],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0, None],
                "StartPoint": [x0, 0.0, z0],
                "EndPoint": [x1, y0, z1],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0, None],
                "StartPoint": [x0, y1, z0],
                "EndPoint": [x1, DOMAIN[1] - eps, z1],
            },
        ]
    )
    mpm.select_save_data(particle=True, grid=True, object=False)
    return mpm


def build_and_run():
    mpm = create_mpm()
    mpm.run()
    postprocess_consolidation(SAVE_PATH)


if __name__ == "__main__":
    build_and_run()
