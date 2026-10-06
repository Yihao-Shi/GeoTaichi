import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.mmpm.Consolidation.DoublePointConsolidation.draw.evaluate_DoublePointConsolidation import (
    postprocess_consolidation,
)


from geotaichi import MPM, init

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(SCRIPT_DIR, "DoublePointConsolidation")
CELL_SIZE = 0.02
BOUNDARY_CELL_THICKNESS = 3
COLUMN_WIDTH = 0.2
COLUMN_HEIGHT = 1.0
COLUMN_ORIGIN = np.array([BOUNDARY_CELL_THICKNESS * CELL_SIZE, BOUNDARY_CELL_THICKNESS * CELL_SIZE], dtype=np.float64)
DOMAIN_WIDTH = COLUMN_WIDTH + 2.0 * BOUNDARY_CELL_THICKNESS * CELL_SIZE
DOMAIN_HEIGHT = 1.2
SURCHARGE = 1.0e4
PERMEABILITY = 1.0e-3
YOUNG_MODULUS = 1.0e8
POISSON_RATIO = 0.30
POROSITY = 0.30
INITIAL_PORE_PRESSURE = SURCHARGE
FLUID_DENSITY = 1000.0
GRAVITY = 9.8
DT = 1.0e-4
SIMULATION_TIME = 0.1456
SAVE_INTERVAL = 0.01456
MG_LEVELS = 3
MAX_PRESSURE_ITERATIONS = 120
PRESSURE_RESIDUAL_TOLERANCE = 1.0e-8
ALPHA_PIC = 1.0
SHAPE_FUNCTION = "QuadBSpline"
VELOCITY_PROJECTION = "Affine"
DELAYED_FLUID_ADVECTION = True
TOP_LOAD_THICKNESS = CELL_SIZE
TOP_LOAD_PARTICLE_LAYERS = 2.0
TOP_LOAD_TRACTION_SCALE = TOP_LOAD_PARTICLE_LAYERS
PROFILE_BINS = max(1, int(round(COLUMN_HEIGHT / CELL_SIZE)))
ARCH = "cpu"
CPU_THREADS = 4
RUN_POSTPROCESS = True


def parse_args():
    parser = argparse.ArgumentParser(description="Run the double-point MPM Terzaghi consolidation example.")
    parser.add_argument("--arch", choices=("cpu", "gpu", "cuda", "vulkan", "metal"), default=ARCH)
    parser.add_argument("--cpu-threads", type=int, default=CPU_THREADS)
    parser.add_argument("--output-dir", default=SAVE_PATH)
    parser.add_argument("--cell-size", type=float, default=CELL_SIZE)
    parser.add_argument("--boundary-cells", type=int, default=BOUNDARY_CELL_THICKNESS)
    parser.add_argument("--permeability", type=float, default=PERMEABILITY)
    parser.add_argument("--initial-pressure", type=float, default=INITIAL_PORE_PRESSURE)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--simulation-time", type=float, default=SIMULATION_TIME)
    parser.add_argument("--save-interval", type=float, default=SAVE_INTERVAL)
    parser.add_argument("--mg-levels", type=int, default=MG_LEVELS)
    parser.add_argument("--max-pressure-iterations", type=int, default=MAX_PRESSURE_ITERATIONS)
    parser.add_argument("--pressure-tolerance", type=float, default=PRESSURE_RESIDUAL_TOLERANCE)
    parser.add_argument("--alpha-pic", type=float, default=ALPHA_PIC)
    parser.add_argument("--shape-function", default=SHAPE_FUNCTION)
    parser.add_argument("--velocity-projection", default=VELOCITY_PROJECTION)
    parser.add_argument(
        "--delayed-fluid-advection",
        action=argparse.BooleanOptionalAction,
        default=DELAYED_FLUID_ADVECTION,
    )
    parser.add_argument("--top-load-thickness", type=float, default=None, help="Defaults to --cell-size.")
    parser.add_argument("--top-load-layers", type=float, default=TOP_LOAD_PARTICLE_LAYERS)
    parser.add_argument("--top-load-scale", type=float, default=None, help="Defaults to --top-load-layers.")
    parser.add_argument("--profile-bins", type=int, default=None, help="Defaults to column-height / cell-size.")
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure(arguments):
    global SAVE_PATH, CELL_SIZE, BOUNDARY_CELL_THICKNESS, COLUMN_ORIGIN, DOMAIN_WIDTH
    global PERMEABILITY, INITIAL_PORE_PRESSURE, DT, SIMULATION_TIME, SAVE_INTERVAL
    global MG_LEVELS, MAX_PRESSURE_ITERATIONS, PRESSURE_RESIDUAL_TOLERANCE
    global ALPHA_PIC, SHAPE_FUNCTION, VELOCITY_PROJECTION, DELAYED_FLUID_ADVECTION
    global TOP_LOAD_THICKNESS, TOP_LOAD_PARTICLE_LAYERS, TOP_LOAD_TRACTION_SCALE
    global PROFILE_BINS, ARCH, CPU_THREADS, RUN_POSTPROCESS

    SAVE_PATH = os.path.abspath(os.path.expanduser(arguments.output_dir))
    CELL_SIZE = arguments.cell_size
    BOUNDARY_CELL_THICKNESS = arguments.boundary_cells
    COLUMN_ORIGIN = np.array(
        [BOUNDARY_CELL_THICKNESS * CELL_SIZE, BOUNDARY_CELL_THICKNESS * CELL_SIZE], dtype=np.float64
    )
    DOMAIN_WIDTH = COLUMN_WIDTH + 2.0 * BOUNDARY_CELL_THICKNESS * CELL_SIZE
    PERMEABILITY = arguments.permeability
    INITIAL_PORE_PRESSURE = arguments.initial_pressure
    DT = arguments.dt
    SIMULATION_TIME = arguments.simulation_time
    SAVE_INTERVAL = arguments.save_interval
    MG_LEVELS = arguments.mg_levels
    MAX_PRESSURE_ITERATIONS = arguments.max_pressure_iterations
    PRESSURE_RESIDUAL_TOLERANCE = arguments.pressure_tolerance
    ALPHA_PIC = arguments.alpha_pic
    SHAPE_FUNCTION = arguments.shape_function
    VELOCITY_PROJECTION = arguments.velocity_projection
    DELAYED_FLUID_ADVECTION = arguments.delayed_fluid_advection
    TOP_LOAD_THICKNESS = arguments.top_load_thickness if arguments.top_load_thickness is not None else CELL_SIZE
    TOP_LOAD_PARTICLE_LAYERS = arguments.top_load_layers
    TOP_LOAD_TRACTION_SCALE = (
        arguments.top_load_scale if arguments.top_load_scale is not None else TOP_LOAD_PARTICLE_LAYERS
    )
    PROFILE_BINS = (
        arguments.profile_bins if arguments.profile_bins is not None else max(1, int(round(COLUMN_HEIGHT / CELL_SIZE)))
    )
    ARCH = arguments.arch
    CPU_THREADS = arguments.cpu_threads
    RUN_POSTPROCESS = arguments.postprocess


def consolidation_coefficient():
    constrained = YOUNG_MODULUS * (1.0 - POISSON_RATIO) / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
    return PERMEABILITY * constrained / (GRAVITY * FLUID_DENSITY)


def create_mpm():
    init(
        dim=2,
        arch=ARCH,
        cpu_max_num_threads=CPU_THREADS,
    )

    mpm = MPM()
    mpm.set_configuration(
        domain=[DOMAIN_WIDTH, DOMAIN_HEIGHT],
        background_damping=0.0,
        gravity=[0.0, 0.0],
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
            "max_particle_number": 10000,
            "max_constraint_number": {
                "max_velocity_constraint": 4000,
                "max_particle_traction_constraint": 2000,
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
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [CELL_SIZE, CELL_SIZE]})
    mpm.add_region(
        region=[
            {
                "Name": "column",
                "Type": "Rectangle2D",
                "BoundingBoxPoint": COLUMN_ORIGIN.tolist(),
                "BoundingBoxSize": [COLUMN_WIDTH, COLUMN_HEIGHT],
            },
            {
                "Name": "top_load",
                "Type": "Rectangle2D",
                "BoundingBoxPoint": [
                    float(COLUMN_ORIGIN[0]),
                    float(COLUMN_ORIGIN[1] + COLUMN_HEIGHT - TOP_LOAD_THICKNESS),
                ],
                "BoundingBoxSize": [COLUMN_WIDTH, TOP_LOAD_THICKNESS],
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
                            "Pressure": [0.0, -SURCHARGE / TOP_LOAD_TRACTION_SCALE],
                            "FluidPressure": [0.0, 0.0],
                            "RegionName": "top_load",
                        }
                    ],
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                },
                {
                    "RegionName": "column",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "ParticleStress": {"PorePressure": INITIAL_PORE_PRESSURE},
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
                "StartPoint": COLUMN_ORIGIN.tolist(),
                "EndPoint": [float(COLUMN_ORIGIN[0] + COLUMN_WIDTH), float(COLUMN_ORIGIN[1])],
                "Norm": [0.0, -1.0],
                "CellThickness": BOUNDARY_CELL_THICKNESS,
            },
            {
                "BoundaryType": "SolidCell",
                "StartPoint": COLUMN_ORIGIN.tolist(),
                "EndPoint": [float(COLUMN_ORIGIN[0]), float(COLUMN_ORIGIN[1] + COLUMN_HEIGHT)],
                "Norm": [-1.0, 0.0],
                "CellThickness": BOUNDARY_CELL_THICKNESS,
            },
            {
                "BoundaryType": "SolidCell",
                "StartPoint": [float(COLUMN_ORIGIN[0] + COLUMN_WIDTH), float(COLUMN_ORIGIN[1])],
                "EndPoint": [float(COLUMN_ORIGIN[0] + COLUMN_WIDTH), float(COLUMN_ORIGIN[1] + COLUMN_HEIGHT)],
                "Norm": [1.0, 0.0],
                "CellThickness": BOUNDARY_CELL_THICKNESS,
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0],
                "StartPoint": [float(COLUMN_ORIGIN[0]), 0.0],
                "EndPoint": [float(COLUMN_ORIGIN[0] + COLUMN_WIDTH), float(COLUMN_ORIGIN[1])],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [0.0, float(COLUMN_ORIGIN[1])],
                "EndPoint": [float(COLUMN_ORIGIN[0]), float(COLUMN_ORIGIN[1] + COLUMN_HEIGHT)],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [float(COLUMN_ORIGIN[0] + COLUMN_WIDTH), float(COLUMN_ORIGIN[1])],
                "EndPoint": [float(DOMAIN_WIDTH - 1.0e-8), float(COLUMN_ORIGIN[1] + COLUMN_HEIGHT)],
            },
        ]
    )
    mpm.select_save_data(particle=True, grid=True, object=False)
    return mpm


def build_and_run(arguments=None):
    if arguments is not None:
        configure(arguments)
    mpm = create_mpm()
    mpm.run()
    if RUN_POSTPROCESS:
        postprocess_consolidation(
            SAVE_PATH,
            simulation_time=SIMULATION_TIME,
            save_interval=SAVE_INTERVAL,
            cv=consolidation_coefficient(),
            column_height=COLUMN_HEIGHT,
            surcharge=SURCHARGE,
            dt=DT,
            column_bottom=COLUMN_ORIGIN[1],
            profile_bins=PROFILE_BINS,
        )


if __name__ == "__main__":
    build_and_run(parse_args())
