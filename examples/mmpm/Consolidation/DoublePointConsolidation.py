import argparse
import os
import sys

import numpy as np
import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import MPM, init


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    TOP_LOAD_TRACTION_SCALE = arguments.top_load_scale if arguments.top_load_scale is not None else TOP_LOAD_PARTICLE_LAYERS
    PROFILE_BINS = arguments.profile_bins if arguments.profile_bins is not None else max(1, int(round(COLUMN_HEIGHT / CELL_SIZE)))
    ARCH = arguments.arch
    CPU_THREADS = arguments.cpu_threads
    RUN_POSTPROCESS = arguments.postprocess


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


def average_pressure_profile(position, pressure, phase, height, time_value=None, cv=None, load=None, n_bins=None):
    if n_bins is None:
        n_bins = PROFILE_BINS
    fluid_mask = phase == 2
    position = position[fluid_mask]
    pressure = pressure[fluid_mask]
    column_bottom = COLUMN_ORIGIN[1]
    column_top = column_bottom + height
    depth = column_top - position[:, 1]
    valid = (depth >= 0.0) & (depth <= height) & np.isfinite(pressure)
    depth = depth[valid]
    pressure = pressure[valid]

    bins = np.linspace(0.0, height, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.zeros(n_bins, dtype=np.float64)
    analytical = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int32)
    ids = np.clip(np.digitize(depth, bins) - 1, 0, n_bins - 1)
    particle_theory = None
    if time_value is not None and cv is not None and load is not None:
        particle_theory = terzaghi_series(depth, time_value, height, load, cv)
    for pid, bid in enumerate(ids):
        values[bid] += pressure[pid]
        if particle_theory is not None:
            analytical[bid] += particle_theory[pid]
        counts[bid] += 1
    mask = counts > 0
    values[mask] /= counts[mask]
    if particle_theory is not None:
        analytical[mask] /= counts[mask]
    else:
        analytical[:] = np.nan
    return centers, values, analytical, mask


def consolidation_coefficient():
    constrained = YOUNG_MODULUS * (1.0 - POISSON_RATIO) / (
        (1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO)
    )
    return PERMEABILITY * constrained / (GRAVITY * FLUID_DENSITY)


def postprocess_consolidation(save_path=None):
    if save_path is None:
        save_path = SAVE_PATH
    particle_dir = os.path.join(save_path, "particles")
    file_names = sorted(name for name in os.listdir(particle_dir) if name.startswith("MPMParticle") and name.endswith(".npz"))
    expected_save_count = int(round(SIMULATION_TIME / SAVE_INTERVAL))
    file_names = file_names[: expected_save_count + 1]
    if len(file_names) < 2:
        raise RuntimeError(f"Not enough particle files in {particle_dir}")

    cv = consolidation_coefficient()
    profiles = []
    times = []
    errors = []
    for save_id, file_name in enumerate(file_names[1:], start=1):
        data = np.load(os.path.join(particle_dir, file_name))
        time_value = float(data["t_current"]) if "t_current" in data.files else save_id * SAVE_INTERVAL
        position = data["position"]
        pressure = data["pressure"]
        phase = data["phase"]
        fluid_mask = phase == 2
        if not np.any(fluid_mask):
            raise RuntimeError(f"No fluid particles found in {file_name}")
        finite_fluid_pressure = np.isfinite(pressure[fluid_mask])
        if not np.all(finite_fluid_pressure):
            bad = int(np.count_nonzero(~finite_fluid_pressure))
            raise RuntimeError(f"{bad} non-finite fluid particle pressures in {file_name}")
        depth, numerical, analytical, mask = average_pressure_profile(
            position, pressure, phase, COLUMN_HEIGHT, time_value, cv, SURCHARGE,
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

    profile_path = os.path.join(save_path, "terzaghi_profiles.npz")
    np.savez(
        profile_path,
        profiles=np.array(profiles, dtype=object),
        times=np.array(times, dtype=np.float64),
        dt=DT,
        cv=cv,
        height=COLUMN_HEIGHT,
        surcharge=SURCHARGE,
    )
    np.savetxt(
        os.path.join(save_path, "terzaghi_profile_errors.csv"),
        np.asarray(errors, dtype=np.float64),
        delimiter=",",
        header="save_id,time_s,Tv,abs_profile_error,relative_profile_error",
        comments="",
    )
    plot_terzaghi_profiles(profile_path, os.path.join(save_path, "terzaghi_profiles_plot.png"))


def plot_terzaghi_profiles(npz_path, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path, allow_pickle=True)
    profiles = [np.asarray(profile, dtype=np.float64) for profile in data["profiles"]]
    times = data["times"]
    cv = float(data["cv"])
    height = float(data["height"])
    surcharge = float(data["surcharge"])

    colors = ["#202020", "#e74c3c", "#2e6bd1", "#8e44ad", "#16a085", "#f39c12", "#7f8c8d"]
    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=200)
    sim_handle = None
    ana_handle = None
    for idx, (profile, time_value) in enumerate(zip(profiles, times)):
        color = colors[idx % len(colors)]
        depth = profile[:, 0] / height
        numerical = profile[:, 1] / surcharge
        analytical = profile[:, 2] / surcharge
        ana_handle = ax.plot(analytical, depth, color=color, lw=1.6, zorder=2)[0]
        sim_handle = ax.plot(
            numerical,
            depth,
            linestyle="none",
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=3,
        )[0]
        label_id = min(len(depth) - 1, max(2, len(depth) // 3))
        ax.text(
            analytical[label_id] + 0.025,
            depth[label_id],
            rf"$T_v={cv * time_value / (height * height):.2f}$",
            color=color,
            fontsize=11,
        )

    ax.set_xlabel(r"$u/u_0$")
    ax.set_ylabel(r"$z/H$")
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(
        [sim_handle, ana_handle],
        ["Double-point MPM", "Terzaghi theory"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
                "BoundingBoxPoint": [float(COLUMN_ORIGIN[0]), float(COLUMN_ORIGIN[1] + COLUMN_HEIGHT - TOP_LOAD_THICKNESS)],
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
        postprocess_consolidation(SAVE_PATH)


if __name__ == "__main__":
    build_and_run(parse_args())
