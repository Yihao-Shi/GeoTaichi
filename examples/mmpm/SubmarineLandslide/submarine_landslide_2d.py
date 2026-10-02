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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import MPM, init

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DOI = "10.1016/j.cma.2024.117064"
SAVE_PATH = os.environ.get(
    "GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SAVE_PATH",
    os.path.join(SCRIPT_DIR, "RzadkiewiczSubmarineLandslide"),
)

TANK_LENGTH = 4.0
WATER_DEPTH = 1.6
DOMAIN_HEIGHT = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_DOMAIN_HEIGHT", "1.8"))
DOMAIN_TOP = DOMAIN_HEIGHT - 1.0e-6
SLOPE_START_X = TANK_LENGTH - WATER_DEPTH
SLOPE_POINT = np.array([SLOPE_START_X, 0.0], dtype=np.float64)
SLOPE_NORMAL = np.array([1.0, -1.0], dtype=np.float64) / np.sqrt(2.0)

PAPER_SOLID_GRID_SPACING = 0.05
PAPER_FLUID_GRID_SPACING = 0.01
PAPER_SOLID_PARTICLES_PER_CELL = 400
PAPER_FLUID_PARTICLES_PER_CELL = 64
PAPER_SOLID_PARTICLE_COUNT = 13165
PAPER_FLUID_PARTICLE_COUNT = 2413600
MAXIMUM_POROSITY = 0.50
INITIAL_POROSITY = 0.38
SOLID_DENSITY = 2650.0
FLUID_DENSITY = 1000.0
GRAVITY = 9.8
FRICTION_ANGLE = 10.0
K0 = 1.0 - math.sin(math.radians(FRICTION_ANGLE))

ELEMENT_SIZE = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_ELEMENT_SIZE", "0.01"))
SOLID_PARTICLES_PER_CELL = int(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SOLID_NPPC", "4"))
FLUID_PARTICLES_PER_CELL = int(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_FLUID_NPPC", "8"))
DT = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_DT", "5.0e-5"))
SIMULATION_TIME = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_TIME", "0.8"))
SAVE_INTERVAL = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SAVE_INTERVAL", "0.04"))
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

SAND_TOP = WATER_DEPTH - 0.1
SAND_LEFT_X = 3.0
SAND_RIGHT_X = 3.9
SAND_POLYGON = [
    [SAND_LEFT_X, SAND_LEFT_X - SLOPE_START_X],
    [SAND_RIGHT_X, SAND_TOP],
    [SAND_LEFT_X, SAND_TOP],
]
WATER_POLYGON = [
    [0.0, 0.0],
    [SLOPE_START_X, 0.0],
    [TANK_LENGTH, WATER_DEPTH],
    [0.0, WATER_DEPTH],
]

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
FREE_SURFACE_BIN_COUNT = int(
    os.environ.get(
        "GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SURFACE_BINS",
        str(int(round(TANK_LENGTH / (2.0 * ELEMENT_SIZE)))),
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


def _latest_particle_files(save_path):
    particle_dir = os.path.join(save_path, "particles")
    if not os.path.isdir(particle_dir):
        return []
    return sorted(
        os.path.join(particle_dir, name)
        for name in os.listdir(particle_dir)
        if name.startswith("MPMParticle") and name.endswith(".npz")
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


def postprocess_landslide(save_path=SAVE_PATH):
    files = _latest_particle_files(save_path)
    if len(files) < 2:
        return

    diagnostics = []
    free_surface_profiles = []
    surface_edges = np.linspace(0.0, TANK_LENGTH, FREE_SURFACE_BIN_COUNT + 1)
    for save_id, file_name in enumerate(files):
        data = np.load(file_name)
        time_value = float(data["t_current"]) if "t_current" in data.files else save_id * SAVE_INTERVAL
        position = data["position"]
        solid_velocity = data["solid_velocity"]
        phase = data["phase"]
        solid = phase == 1
        fluid = phase == 2

        solid_pos = position[solid]
        solid_vel = solid_velocity[solid]
        if solid_pos.size == 0:
            continue
        solid_speed = np.linalg.norm(solid_vel, axis=1)
        centroid = solid_pos.mean(axis=0)
        front_x = float(np.max(solid_pos[:, 0]))
        max_speed = float(np.max(solid_speed))
        mean_speed = float(np.mean(solid_speed))

        surface = np.full(FREE_SURFACE_BIN_COUNT, -np.inf, dtype=np.float64)
        if np.any(fluid):
            fluid_pos = position[fluid]
            ids = np.clip(
                np.searchsorted(surface_edges, fluid_pos[:, 0], side="right") - 1,
                0,
                FREE_SURFACE_BIN_COUNT - 1,
            )
            np.maximum.at(surface, ids, fluid_pos[:, 1])
        surface[~np.isfinite(surface)] = np.nan
        free_surface_profiles.append(surface)

        diagnostics.append(
            [
                save_id,
                time_value,
                centroid[0],
                centroid[1],
                front_x,
                max_speed,
                mean_speed,
                float(np.min(solid_pos[:, 1])),
                float(np.max(solid_pos[:, 1])),
            ]
        )

    if not diagnostics:
        return

    diagnostics = np.asarray(diagnostics, dtype=np.float64)
    os.makedirs(save_path, exist_ok=True)
    np.savetxt(
        os.path.join(save_path, "landslide_diagnostics.csv"),
        diagnostics,
        delimiter=",",
        header="save_id,time_s,solid_centroid_x,solid_centroid_y,solid_front_x,solid_max_speed,solid_mean_speed,solid_min_y,solid_max_y",
        comments="",
    )
    np.savez(
        os.path.join(save_path, "landslide_diagnostics.npz"),
        diagnostics=diagnostics,
        free_surface_profiles=np.asarray(free_surface_profiles, dtype=np.float64),
        free_surface_x=0.5 * (surface_edges[:-1] + surface_edges[1:]),
        sand_polygon=np.asarray(SAND_POLYGON, dtype=np.float64),
        water_polygon=np.asarray(WATER_POLYGON, dtype=np.float64),
        slope_point=SLOPE_POINT,
        slope_normal=SLOPE_NORMAL,
    )

    final = diagnostics[-1]
    print(
        "landslide diagnostics: "
        f"time={final[1]:.4f}s, solid_front_x={final[4]:.4f}m, "
        f"solid_max_speed={final[5]:.4f}m/s, solid_mean_speed={final[6]:.4f}m/s"
    )


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
