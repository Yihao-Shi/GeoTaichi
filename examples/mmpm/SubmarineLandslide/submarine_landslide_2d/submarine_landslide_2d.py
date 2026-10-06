"""Paper-scale two-dimensional submarine landslide benchmark.

The geometry, material parameters, duration, timestep, and target particle
spacing follow Section 5.5 of He et al. (2024), DOI
10.1016/j.cma.2024.117064.  GeoTaichi uses one shared background spacing for
the co-located solid and MAC fluid grids, so the paper's 0.01 m fluid spacing
is used for both.  Four solid and eight fluid points per axis recover the
paper's 0.0025 m and 0.00125 m material-point spacings, respectively.
"""

import json
import math
import os
import sys

import numpy as np
import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.mmpm.SubmarineLandslide.submarine_landslide_2d.submarine_landslide_2d_parameters import (
    MAXIMUM_POROSITY,
    REFERENCE_DOI,
)

from examples.mmpm.SubmarineLandslide.submarine_landslide_2d.submarine_landslide_2d_parameters import (
    ELEMENT_SIZE,
    SAND_LEFT_X,
    SAND_POLYGON,
    SAND_RIGHT_X,
    SAND_TOP,
    SAVE_INTERVAL,
    SAVE_PATH,
    SLOPE_NORMAL,
    SLOPE_POINT,
    SLOPE_START_X,
    TANK_LENGTH,
    WATER_DEPTH,
    WATER_POLYGON,
)
from examples.mmpm.SubmarineLandslide.submarine_landslide_2d.draw.evaluate_submarine_landslide_2d import (
    postprocess_landslide,
)


from geotaichi import MPM, init

DOMAIN_HEIGHT = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_DOMAIN_HEIGHT", "1.8"))
DOMAIN_TOP = DOMAIN_HEIGHT - 1.0e-6

PAPER_SOLID_GRID_SPACING = 0.05
PAPER_FLUID_GRID_SPACING = 0.01
PAPER_SOLID_PARTICLES_PER_CELL = 400
PAPER_FLUID_PARTICLES_PER_CELL = 64
PAPER_SOLID_PARTICLE_COUNT = 13165
PAPER_FLUID_PARTICLE_COUNT = 2413600
INITIAL_POROSITY = 0.38
SOLID_DENSITY = 2650.0
FLUID_DENSITY = 1000.0
GRAVITY = 9.8
FRICTION_ANGLE = 10.0
K0 = 1.0 - math.sin(math.radians(FRICTION_ANGLE))

SOLID_PARTICLES_PER_CELL = int(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SOLID_NPPC", "4"))
FLUID_PARTICLES_PER_CELL = int(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_FLUID_NPPC", "8"))
DT = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_DT", "5.0e-5"))
SIMULATION_TIME = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_TIME", "0.8"))
MG_LEVELS = int(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_MG_LEVELS", "2"))
MAX_PRESSURE_ITERATIONS = int(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_MAX_PRESSURE_ITERATIONS", "200"))
PRESSURE_TOLERANCE = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_PRESSURE_TOLERANCE", "1.0e-8"))
SHAPE_FUNCTION = os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SHAPE", "QuadBSpline")
VELOCITY_PROJECTION = os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_VELOCITY_PROJECTION", "Affine")
ALPHA_PIC = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_ALPHA_PIC", "1.0"))
WALL_THICKNESS = int(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_WALL_CELLS", "2"))
SAVE_GRID = os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SAVE_GRID", "0") != "0"
SKIP_POSTPROCESS = os.environ.get("GEOTAICHI_SKIP_POSTPROCESS", "0") != "0"
ARCH = os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_ARCH", "cpu")


WATER_AREA = WATER_DEPTH * SLOPE_START_X + 0.5 * WATER_DEPTH * WATER_DEPTH
SAND_AREA = 0.5 * (SAND_RIGHT_X - SAND_LEFT_X) * (SAND_TOP - (SAND_LEFT_X - SLOPE_START_X))
ESTIMATED_FLUID_PARTICLES = int(
    math.ceil(WATER_AREA * FLUID_PARTICLES_PER_CELL * FLUID_PARTICLES_PER_CELL / (ELEMENT_SIZE * ELEMENT_SIZE))
)
ESTIMATED_SOLID_PARTICLES = int(
    math.ceil(SAND_AREA * SOLID_PARTICLES_PER_CELL * SOLID_PARTICLES_PER_CELL / (ELEMENT_SIZE * ELEMENT_SIZE))
)
ESTIMATED_PARTICLES = ESTIMATED_FLUID_PARTICLES + ESTIMATED_SOLID_PARTICLES
MAX_PARTICLE_NUMBER = int(
    os.environ.get(
        "GEOTAICHI_RZADKIEWICZ_LANDSLIDE_MAX_PARTICLES",
        str(max(50000, math.ceil(1.15 * ESTIMATED_PARTICLES))),
    )
)
MAX_VELOCITY_CONSTRAINTS = int(
    os.environ.get(
        "GEOTAICHI_RZADKIEWICZ_LANDSLIDE_MAX_VELOCITY_CONSTRAINTS",
        str(max(20000, int(20.0 * (TANK_LENGTH + DOMAIN_HEIGHT) / ELEMENT_SIZE))),
    )
)


def _polygon_bounds(vertices):
    points = np.asarray(vertices, dtype=np.float64)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return lower.tolist(), (upper - lower).tolist()


@ti.kernel
def initialize_hydrostatic_landslide(
    particle_count: int,
    particle: ti.template(),
):
    for p in range(particle_count):
        if int(particle[p].active) == 1:
            water_depth = ti.max(WATER_DEPTH - particle[p].x[1], 0.0)
            particle[p].pressure = FLUID_DENSITY * GRAVITY * water_depth
            if int(particle[p].phase) == 1:
                sand_depth = ti.max(SAND_TOP - particle[p].x[1], 0.0)
                effective_vertical = (1.0 - INITIAL_POROSITY) * (SOLID_DENSITY - FLUID_DENSITY) * GRAVITY * sand_depth
                particle[p].stress = ti.Vector(
                    [
                        -K0 * effective_vertical,
                        -effective_vertical,
                        -K0 * effective_vertical,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )


def write_case_metadata(save_path=SAVE_PATH):
    os.makedirs(save_path, exist_ok=True)
    metadata = {
        "benchmark": "Rzadkiewicz submerged granular landslide",
        "reference": {
            "citation": "He et al. (2024), Section 5.5",
            "doi": REFERENCE_DOI,
            "tank_length_m": TANK_LENGTH,
            "water_depth_m": WATER_DEPTH,
            "slope_angle_deg": 45.0,
            "solid_grid_spacing_m": PAPER_SOLID_GRID_SPACING,
            "fluid_grid_spacing_m": PAPER_FLUID_GRID_SPACING,
            "solid_particles_per_cell": PAPER_SOLID_PARTICLES_PER_CELL,
            "fluid_particles_per_cell": PAPER_FLUID_PARTICLES_PER_CELL,
            "solid_particle_count": PAPER_SOLID_PARTICLE_COUNT,
            "fluid_particle_count": PAPER_FLUID_PARTICLE_COUNT,
            "timestep_s": 5.0e-5,
            "duration_s": 0.8,
        },
        "geotaichi": {
            "shared_grid_spacing_m": ELEMENT_SIZE,
            "solid_particles_per_axis": SOLID_PARTICLES_PER_CELL,
            "fluid_particles_per_axis": FLUID_PARTICLES_PER_CELL,
            "solid_particle_spacing_m": ELEMENT_SIZE / SOLID_PARTICLES_PER_CELL,
            "fluid_particle_spacing_m": ELEMENT_SIZE / FLUID_PARTICLES_PER_CELL,
            "estimated_solid_particle_count": ESTIMATED_SOLID_PARTICLES,
            "estimated_fluid_particle_count": ESTIMATED_FLUID_PARTICLES,
            "particle_capacity": MAX_PARTICLE_NUMBER,
            "timestep_s": DT,
            "duration_s": SIMULATION_TIME,
            "save_interval_s": SAVE_INTERVAL,
            "maximum_porosity": MAXIMUM_POROSITY,
            "initial_condition": "hydrostatic pore pressure and submerged K0 effective stress",
            "adaptations": [
                "The separate 0.05 m solid and 0.01 m fluid grids are represented by one shared 0.01 m grid.",
                "The paper's particle-discretized frictionless slope is represented by an analytical frictionless solid plane.",
                "Permeability, fluid bulk modulus, and the Ergun drag closure are GeoTaichi route inputs not reported in the benchmark table.",
                "The maximum porosity 0.50 follows Ceccato et al. (2016), Table 2, and activates the double-point solid-to-fluid transition.",
            ],
        },
    }
    with open(os.path.join(save_path, "benchmark_metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)


def create_mpm():
    init(dim=2, arch=ARCH, cpu_max_num_threads=4)

    mpm = MPM()
    mpm.set_configuration(
        domain=[TANK_LENGTH, DOMAIN_HEIGHT],
        background_damping=0.0,
        gravity=[0.0, -9.8],
        alphaPIC=ALPHA_PIC,
        mapping="USL",
        shape_function=SHAPE_FUNCTION,
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
        velocity_projection=VELOCITY_PROJECTION,
        delayed_fluid_advection=True,
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
            "residual_tolerance": PRESSURE_TOLERANCE,
            "multilevel": MG_LEVELS,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": MAX_PARTICLE_NUMBER,
            "max_constraint_number": {"max_velocity_constraint": MAX_VELOCITY_CONSTRAINTS},
        }
    )
    mpm.add_material(
        model="MohrCoulomb",
        material={
            "MaterialID": 1,
            "SolidDensity": SOLID_DENSITY,
            "FluidDensity": FLUID_DENSITY,
            "Porosity": INITIAL_POROSITY,
            "MaximumPorosity": MAXIMUM_POROSITY,
            "FluidBulkModulus": 2.2e8,
            "Permeability": 1.0e-3,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 6.0e-3,
            "DragModel": "Ergun",
            "YoungModulus": 5.0e6,
            "PoissonRatio": 0.30,
            "Cohesion": 0.0,
            "Friction": FRICTION_ANGLE,
            "Dilation": 0.0,
        },
    )
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [ELEMENT_SIZE, ELEMENT_SIZE]})

    water_origin, water_size = _polygon_bounds(WATER_POLYGON)
    sand_origin, sand_size = _polygon_bounds(SAND_POLYGON)
    mpm.add_region(
        region=[
            {
                "Name": "water",
                "Type": "Polygon2D",
                "BoundingBoxPoint": water_origin,
                "BoundingBoxSize": water_size,
                "Vertices": WATER_POLYGON,
            },
            {
                "Name": "sand",
                "Type": "Polygon2D",
                "BoundingBoxPoint": sand_origin,
                "BoundingBoxSize": sand_size,
                "Vertices": SAND_POLYGON,
            },
        ]
    )
    mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "water",
                    "nParticlesPerCell": FLUID_PARTICLES_PER_CELL,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                },
                {
                    "RegionName": "sand",
                    "nParticlesPerCell": SOLID_PARTICLES_PER_CELL,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Solid",
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                },
            ]
        }
    )
    initialize_hydrostatic_landslide(int(mpm.scene.particleNum[0]), mpm.scene.particle)

    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "SolidCell",
                "StartPoint": [0.0, 0.0],
                "EndPoint": [SLOPE_START_X, 0.0],
                "Norm": [0.0, -1.0],
                "CellThickness": WALL_THICKNESS,
            },
            {
                "BoundaryType": "SolidCell",
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.0, DOMAIN_TOP],
                "Norm": [-1.0, 0.0],
                "CellThickness": WALL_THICKNESS,
            },
            {
                "BoundaryType": "SolidCell",
                "StartPoint": [TANK_LENGTH, 0.0],
                "EndPoint": [TANK_LENGTH, DOMAIN_TOP],
                "Norm": [1.0, 0.0],
                "CellThickness": WALL_THICKNESS,
            },
            {
                "BoundaryType": "SolidPlaneCell",
                "StartPoint": [SLOPE_START_X, 0.0],
                "EndPoint": [TANK_LENGTH, WATER_DEPTH],
                "Point": SLOPE_POINT.tolist(),
                "Norm": SLOPE_NORMAL.tolist(),
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [SLOPE_START_X, 0.0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.0, DOMAIN_TOP],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [TANK_LENGTH, 0.0],
                "EndPoint": [TANK_LENGTH, DOMAIN_TOP],
            },
        ]
    )
    mpm.select_save_data(particle=True, grid=SAVE_GRID, object=False)
    return mpm


def build_and_run():
    print(
        "Rzadkiewicz paper-scale discretization: "
        f"dx={ELEMENT_SIZE:g} m, solid_nppc={SOLID_PARTICLES_PER_CELL}, "
        f"fluid_nppc={FLUID_PARTICLES_PER_CELL}, estimated_solid={ESTIMATED_SOLID_PARTICLES}, "
        f"estimated_fluid={ESTIMATED_FLUID_PARTICLES}, capacity={MAX_PARTICLE_NUMBER}"
    )
    write_case_metadata(SAVE_PATH)
    mpm = create_mpm()
    mpm.run()
    if not SKIP_POSTPROCESS:
        postprocess_landslide(SAVE_PATH)


if __name__ == "__main__":
    build_and_run()
