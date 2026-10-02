import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import MPM, init


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.environ.get(
    "GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_SAVE_PATH",
    os.path.join(SCRIPT_DIR, "DoublePointConsolidation3D"),
)
ARCH = os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_ARCH", "cpu")
CELL_SIZE = 0.02
BOUNDARY_CELL_THICKNESS = int(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_BOUNDARY_CELLS", "3"))
COLUMN_WIDTH = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_WIDTH", "0.2"))
COLUMN_DEPTH = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_DEPTH", "0.2"))
COLUMN_HEIGHT = 1.0
COLUMN_ORIGIN = np.array(
    [
        BOUNDARY_CELL_THICKNESS * CELL_SIZE,
        BOUNDARY_CELL_THICKNESS * CELL_SIZE,
        BOUNDARY_CELL_THICKNESS * CELL_SIZE,
    ],
    dtype=np.float64,
)
DOMAIN = [
    COLUMN_WIDTH + 2.0 * BOUNDARY_CELL_THICKNESS * CELL_SIZE,
    COLUMN_DEPTH + 2.0 * BOUNDARY_CELL_THICKNESS * CELL_SIZE,
    1.2,
]
SURCHARGE = 1.0e4
PERMEABILITY = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_PERMEABILITY", "1.0e-3"))
YOUNG_MODULUS = 1.0e8
POISSON_RATIO = 0.30
POROSITY = 0.30
INITIAL_PORE_PRESSURE = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_INITIAL_PRESSURE", str(SURCHARGE)))
FLUID_DENSITY = 1000.0
GRAVITY = 9.8
DT = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_DT", "1.0e-4"))
SIMULATION_TIME = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_TIME", "0.01456"))
SAVE_INTERVAL = float(os.environ.get("GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_SAVE_INTERVAL", str(SIMULATION_TIME)))
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
PROFILE_BINS = int(
    os.environ.get(
        "GEOTAICHI_DOUBLE_POINT_CONSOLIDATION3D_PROFILE_BINS",
        str(max(1, int(round(COLUMN_HEIGHT / CELL_SIZE)))),
    )
)


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


def terzaghi_series(depth_from_top, time_value, height, load, cv, terms=400):
    series = np.zeros_like(depth_from_top, dtype=np.float64)
    for m in range(terms):
        n = 2 * m + 1
        series += (
            4.0
            / (n * np.pi)
            * np.sin(n * np.pi * depth_from_top / (2.0 * height))
            * np.exp(-(n * n) * np.pi * np.pi * cv * time_value / (4.0 * height * height))
        )
    return load * series


def consolidation_coefficient():
    constrained = YOUNG_MODULUS * (1.0 - POISSON_RATIO) / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
    return PERMEABILITY * constrained / (GRAVITY * FLUID_DENSITY)


def average_pressure_profile(position, pressure, phase, time_value, cv, n_bins=PROFILE_BINS):
    fluid_mask = phase == 2
    position = position[fluid_mask]
    pressure = pressure[fluid_mask]
    column_min = COLUMN_ORIGIN
    column_max = COLUMN_ORIGIN + np.array([COLUMN_WIDTH, COLUMN_DEPTH, COLUMN_HEIGHT], dtype=np.float64)
    column_bottom = COLUMN_ORIGIN[2]
    column_top = column_bottom + COLUMN_HEIGHT
    depth = column_top - position[:, 2]
    valid = (
        (position[:, 0] >= column_min[0])
        & (position[:, 0] <= column_max[0])
        & (position[:, 1] >= column_min[1])
        & (position[:, 1] <= column_max[1])
        & (depth >= 0.0)
        & (depth <= COLUMN_HEIGHT)
        & np.isfinite(pressure)
    )
    depth = depth[valid]
    pressure = pressure[valid]

    bins = np.linspace(0.0, COLUMN_HEIGHT, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.zeros(n_bins, dtype=np.float64)
    analytical = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int32)
    ids = np.clip(np.digitize(depth, bins) - 1, 0, n_bins - 1)
    particle_theory = terzaghi_series(depth, time_value, COLUMN_HEIGHT, SURCHARGE, cv)
    for pid, bid in enumerate(ids):
        values[bid] += pressure[pid]
        analytical[bid] += particle_theory[pid]
        counts[bid] += 1
    mask = counts > 0
    values[mask] /= counts[mask]
    analytical[mask] /= counts[mask]
    return centers, values, analytical, mask


def postprocess_consolidation(save_path=SAVE_PATH):
    particle_dir = os.path.join(save_path, "particles")
    file_names = sorted(
        name for name in os.listdir(particle_dir) if name.startswith("MPMParticle") and name.endswith(".npz")
    )
    expected_save_count = int(round(SIMULATION_TIME / SAVE_INTERVAL))
    file_names = file_names[: expected_save_count + 1]
    if len(file_names) < 2:
        raise RuntimeError(f"Not enough particle files in {particle_dir}")

    cv = consolidation_coefficient()
    errors = []
    profiles = []
    times = []
    for save_id, file_name in enumerate(file_names[1:], start=1):
        data = np.load(os.path.join(particle_dir, file_name))
        time_value = float(data["t_current"]) if "t_current" in data.files else save_id * SAVE_INTERVAL
        depth, numerical, analytical, mask = average_pressure_profile(
            data["position"],
            data["pressure"],
            data["phase"],
            time_value,
            cv,
        )
        if not np.any(mask):
            raise RuntimeError(f"No valid fluid pressure bins found in {file_name}")
        abs_err = np.linalg.norm(numerical[mask] - analytical[mask])
        rel_err = abs_err / max(np.linalg.norm(analytical[mask]), 1.0e-12)
        profiles.append(np.column_stack([depth, numerical, analytical]))
        times.append(time_value)
        errors.append([save_id, time_value, cv * time_value / (COLUMN_HEIGHT * COLUMN_HEIGHT), abs_err, rel_err])
        print(
            f"save={save_id:03d}, time={time_value:10.4e}, "
            f"Tv={cv * time_value / (COLUMN_HEIGHT * COLUMN_HEIGHT):8.4f}, "
            f"relative_profile_error={rel_err:10.4e}"
        )

    np.savez(
        os.path.join(save_path, "terzaghi_profiles_3d.npz"),
        profiles=np.array(profiles, dtype=object),
        times=np.array(times, dtype=np.float64),
        dt=DT,
        cv=cv,
        height=COLUMN_HEIGHT,
        surcharge=SURCHARGE,
        column_width=COLUMN_WIDTH,
        column_depth=COLUMN_DEPTH,
    )
    np.savetxt(
        os.path.join(save_path, "terzaghi_profile_errors_3d.csv"),
        np.asarray(errors, dtype=np.float64),
        delimiter=",",
        header="save_id,time_s,Tv,abs_profile_error,relative_profile_error",
        comments="",
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
