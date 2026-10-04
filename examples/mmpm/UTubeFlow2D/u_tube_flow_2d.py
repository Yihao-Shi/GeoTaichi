"""Two-dimensional U-tube seepage (Zhang et al., 2027, Section 4.2)."""

import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geotaichi import MPM, init  # noqa: E402

PREFIX = "GT_UTUBE_2D_"


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(PREFIX + name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(PREFIX + name, default))


ARCH = os.environ.get(PREFIX + "ARCH", "gpu")
OUTPUT = (
    Path(
        os.environ.get(
            PREFIX + "OUTPUT",
            Path(__file__).resolve().parent / "OutputData" / "paper_section_4_2_2d",
        )
    )
    .expanduser()
    .resolve()
)
DX = env_float("DX", 0.05)
DT = env_float("DT", 5.0e-3)
SIMULATION_TIME = env_float("TIME", 100.0)
SAVE_INTERVAL = env_float("SAVE_INTERVAL", 2.0)
PPC = env_int("PPC", 2)
HYDRAULIC_CONDUCTIVITY = env_float("HYDRAULIC_CONDUCTIVITY", 0.05)
SKIP_POSTPROCESS = os.environ.get(PREFIX + "SKIP_POSTPROCESS", "0") == "1"

WIDTH = 3.0
PHYSICAL_HEIGHT = 3.35
DOMAIN = [WIDTH, 3.4]  # one cell of headroom makes both MGPCG cell counts even
POROUS_ORIGIN = [1.0, 0.0]
POROUS_SIZE = [1.0, 1.0]
LEFT_WATER_DEPTH = 3.0
RIGHT_WATER_DEPTH = 2.0
INITIAL_HEAD_DIFFERENCE = LEFT_WATER_DEPTH - RIGHT_WATER_DEPTH
POROSITY = 0.4


def aligned_count(length: float, spacing: float) -> int:
    count = round(length / spacing)
    if not math.isclose(count * spacing, length, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"geometry length {length:g} is not aligned with spacing={spacing:g}")
    return count


DOMAIN_CELLS = [aligned_count(length, DX) for length in DOMAIN]
FLUID_AREA = LEFT_WATER_DEPTH + POROUS_SIZE[0] * POROUS_SIZE[1] + RIGHT_WATER_DEPTH
EXPECTED_FLUID_PARTICLES = math.ceil(FLUID_AREA * PPC**2 / DX**2)
EXPECTED_SOLID_PARTICLES = math.ceil(math.prod(POROUS_SIZE) * PPC**2 / DX**2)
MAX_PARTICLES = EXPECTED_FLUID_PARTICLES + EXPECTED_SOLID_PARTICLES
MAX_VELOCITY_CONSTRAINTS = 2 * (aligned_count(POROUS_SIZE[0], DX) + 1) * (aligned_count(POROUS_SIZE[1], DX) + 1)

if min(DX, DT, SIMULATION_TIME, SAVE_INTERVAL, HYDRAULIC_CONDUCTIVITY) <= 0.0 or PPC <= 0:
    raise ValueError("spacing, time, conductivity, and particles per cell must be positive")
if any(count % 2 for count in DOMAIN_CELLS):
    raise ValueError("the two-level MGPCG grid must have even cell counts")


@ti.kernel
def initialize_pressure(particle_count: int, particle: ti.template()):
    for p in range(particle_count):
        x = particle[p].x[0]
        head = LEFT_WATER_DEPTH
        if x > POROUS_ORIGIN[0]:
            head = LEFT_WATER_DEPTH - (x - POROUS_ORIGIN[0]) * INITIAL_HEAD_DIFFERENCE / POROUS_SIZE[0]
        if x >= POROUS_ORIGIN[0] + POROUS_SIZE[0]:
            head = RIGHT_WATER_DEPTH
        particle[p].pressure = 1000.0 * 9.81 * ti.max(head - particle[p].x[1], 0.0)


def postprocess(output: Path):
    files = sorted(glob.glob(str(output / "particles" / "MPMParticle*.npz")))
    if len(files) < 2:
        raise RuntimeError("U-tube recorder produced fewer than two particle states")

    rows = []
    max_solid_displacement = 0.0
    max_cavity_particles = 0
    initial_solid_position = None
    for file_name in files:
        data = np.load(file_name)
        active = data["active"] > 0
        phase = data["phase"]
        position = data["position"]
        fluid = active & (phase == 2)
        solid = active & (phase == 1)
        left = fluid & (position[:, 0] < POROUS_ORIGIN[0])
        right = fluid & (position[:, 0] > POROUS_ORIGIN[0] + POROUS_SIZE[0])
        cavity = (
            fluid
            & (position[:, 0] >= POROUS_ORIGIN[0])
            & (position[:, 0] <= POROUS_ORIGIN[0] + POROUS_SIZE[0])
            & (position[:, 1] > POROUS_ORIGIN[1] + POROUS_SIZE[1] + DX)
        )
        max_cavity_particles = max(max_cavity_particles, int(np.count_nonzero(cavity)))
        if not np.any(left) or not np.any(right):
            raise RuntimeError(f"lost a water column in {file_name}")
        left_surface = float(np.quantile(position[left, 1], 0.99))
        right_surface = float(np.quantile(position[right, 1], 0.99))
        time = float(data["t_current"])
        numerical_head = left_surface - right_surface
        analytical_head = INITIAL_HEAD_DIFFERENCE * math.exp(-2.0 * HYDRAULIC_CONDUCTIVITY * time / POROUS_SIZE[0])
        if initial_solid_position is None:
            initial_solid_position = position[solid].copy()
        max_solid_displacement = max(
            max_solid_displacement,
            float(np.max(np.linalg.norm(position[solid] - initial_solid_position, axis=1))),
        )
        speed = np.linalg.norm(data["fluid_velocity"][fluid], axis=1)
        rows.append(
            [
                time,
                left_surface,
                right_surface,
                numerical_head,
                analytical_head,
                numerical_head - analytical_head,
                float(np.quantile(speed, 0.99)),
                float(np.min(data["pressure"][fluid])),
                float(np.max(data["pressure"][fluid])),
            ]
        )

    rows = np.asarray(rows, dtype=np.float64)
    np.savetxt(
        output / "head_decay.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,left_surface_m,right_surface_m,numerical_head_difference_m,"
            "analytical_head_difference_m,error_m,fluid_speed_p99_mps,pressure_min_pa,pressure_max_pa"
        ),
        comments="",
    )
    relative_l2 = float(np.linalg.norm(rows[:, 3] - rows[:, 4]) / max(np.linalg.norm(rows[:, 4]), 1.0e-12))
    metrics = {
        "case": "Zhang et al. (2027), Section 4.2, two-dimensional U-tube flow",
        "doi": "10.1016/j.cma.2026.119401",
        "domain_m": DOMAIN,
        "physical_container_height_m": PHYSICAL_HEIGHT,
        "cell_counts": DOMAIN_CELLS,
        "element_size_m": DX,
        "timestep_s": DT,
        "simulation_time_s": SIMULATION_TIME,
        "particle_spacing_m": DX / PPC,
        "fluid_particle_count": EXPECTED_FLUID_PARTICLES,
        "solid_particle_count": EXPECTED_SOLID_PARTICLES,
        "porosity": POROSITY,
        "hydraulic_conductivity_mps": HYDRAULIC_CONDUCTIVITY,
        "initial_head_difference_m": INITIAL_HEAD_DIFFERENCE,
        "analytical_time_scale_s": POROUS_SIZE[0] / (2.0 * HYDRAULIC_CONDUCTIVITY),
        "relative_l2_head_error": relative_l2,
        "final_numerical_head_difference_m": float(rows[-1, 3]),
        "final_analytical_head_difference_m": float(rows[-1, 4]),
        "maximum_solid_displacement_m": max_solid_displacement,
        "maximum_fluid_particles_in_closed_cavity": max_cavity_particles,
        "finite": bool(np.isfinite(rows).all()),
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and max_solid_displacement <= 1.0e-10
        and max_cavity_particles == 0
        and abs(metrics["final_numerical_head_difference_m"]) <= DX
        and relative_l2 <= 0.35
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps(metrics, sort_keys=True))
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(rows[:, 0], rows[:, 4], "k-", label="analytical")
    ax.plot(rows[:, 0], rows[:, 3], "o", ms=3, label="MPM")
    ax.set(xlabel="time (s)", ylabel="head difference (m)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "head_decay.png", dpi=180)
    plt.close(fig)

    targets = (0.0, 10.0, 30.0, 100.0)
    snapshots = []
    for target in targets:
        file_name = min(files, key=lambda name: abs(float(np.load(name)["t_current"]) - target))
        snapshots.append(np.load(file_name))
    pressure_max = max(
        float(np.quantile(data["pressure"][(data["active"] > 0) & (data["phase"] == 2)], 0.99)) for data in snapshots
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.5), sharex=True, sharey=True)
    for ax, data in zip(axes.flat, snapshots):
        active = data["active"] > 0
        fluid = active & (data["phase"] == 2)
        solid = active & (data["phase"] == 1)
        position = data["position"]
        ax.scatter(position[solid, 0], position[solid, 1], s=1.0, color="0.65")
        points = ax.scatter(
            position[fluid, 0],
            position[fluid, 1],
            s=1.5,
            c=data["pressure"][fluid],
            cmap="viridis",
            vmin=0.0,
            vmax=pressure_max,
        )
        ax.set(
            title=f"t = {float(data['t_current']):g} s", aspect="equal", xlim=(0.0, WIDTH), ylim=(0.0, PHYSICAL_HEIGHT)
        )
    fig.supxlabel("x (m)")
    fig.supylabel("y (m)")
    fig.colorbar(points, ax=axes, label="pressure (Pa)", shrink=0.8)
    fig.savefig(output / "pressure_snapshots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, sort_keys=True))


if os.environ.get(PREFIX + "POSTPROCESS_ONLY", "0") == "1":
    postprocess(OUTPUT)
    raise SystemExit


print("# Zhang et al. (2027), Section 4.2: two-dimensional U-tube seepage")
print(
    f"# domain={DOMAIN}, cells={DOMAIN_CELLS}, dx={DX:g}, ppc={PPC}, "
    f"particles={MAX_PARTICLES}, steps={math.ceil(SIMULATION_TIME / DT)}"
)

init(dim=2, arch=ARCH, default_fp="float64", device_memory_GB=env_float("DEVICE_MEMORY_GB", 2.0))
mpm = MPM()
mpm.set_configuration(
    domain=DOMAIN,
    background_damping=0.01,
    gravity=[0.0, -9.81],
    alphaPIC=1.0,
    mapping="USL",
    shape_function="QuadBSpline",
    material_type="TwoPhaseDoubleLayer",
    solver_type="SemiImplicit",
    velocity_projection="Affine",
    delayed_fluid_advection=True,
    particle_shifting=True,
    visualize=True,
)
mpm.set_solver(
    {
        "Timestep": DT,
        "SimulationTime": SIMULATION_TIME,
        "SaveInterval": SAVE_INTERVAL,
        "SavePath": str(OUTPUT),
    }
)
mpm.set_semi_implicit_solver_parameters(
    {
        "assemble_type": "MatrixFree",
        "pressure_solver": "MGPCG",
        "linear_solver": "MGPCG",
        "max_iteration_number": 200,
        "residual_tolerance": 1.0e-7,
        "multilevel": 2,
        "pre_and_post_smoothing": 2,
        "bottom_smoothing": 8,
    }
)
mpm.memory_allocate(
    {
        "max_material_number": 2,
        "max_particle_number": MAX_PARTICLES,
        "max_constraint_number": {"max_velocity_constraint": MAX_VELOCITY_CONSTRAINTS},
    }
)
mpm.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 1,
        "SolidDensity": 2650.0,
        "FluidDensity": 1000.0,
        "Porosity": POROSITY,
        "FluidBulkModulus": 2.2e8,
        "Permeability": HYDRAULIC_CONDUCTIVITY,
        "FluidViscosity": 1.0e-3,
        "GrainDiameter": 3.0e-3,
        "DragModel": "Darcy",
        "YoungModulus": 4.0e7,
        "PoissonRatio": 0.25,
    },
)
mpm.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 2,
        "SolidDensity": 2650.0,
        "FluidDensity": 1000.0,
        # The constitutive input requires an open interval; 0.9999 is the
        # solver's own clear-fluid cap and is numerically equivalent to one.
        "Porosity": 0.9999,
        "FluidBulkModulus": 2.2e8,
        "Permeability": HYDRAULIC_CONDUCTIVITY,
        "FluidViscosity": 1.0e-3,
        "GrainDiameter": 3.0e-3,
        "DragModel": "Darcy",
        "YoungModulus": 4.0e7,
        "PoissonRatio": 0.25,
    },
)
mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [DX, DX]})
mpm.add_region(
    region=[
        {
            "Name": "left_water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": [1.0, LEFT_WATER_DEPTH],
        },
        {
            "Name": "porous_water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": POROUS_ORIGIN,
            "BoundingBoxSize": POROUS_SIZE,
        },
        {
            "Name": "right_water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [2.0, 0.0],
            "BoundingBoxSize": [1.0, RIGHT_WATER_DEPTH],
        },
        {
            "Name": "fixed_porous_skeleton",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": POROUS_ORIGIN,
            "BoundingBoxSize": POROUS_SIZE,
        },
    ]
)
mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": region,
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 2,
                "Phase": "Fluid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
            for region in ("left_water", "right_water")
        ]
        + [
            {
                "RegionName": "porous_water",
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Fluid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            },
            {
                "RegionName": "fixed_porous_skeleton",
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Solid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Fix", "Fix"],
            },
        ]
    }
)
initialize_pressure(int(mpm.scene.particleNum[0]), mpm.scene.particle)

walls = [
    ([0.0, 0.0], [0.0, PHYSICAL_HEIGHT], [-1.0, 0.0]),
    ([WIDTH, 0.0], [WIDTH, PHYSICAL_HEIGHT], [1.0, 0.0]),
    ([0.0, 0.0], [WIDTH, 0.0], [0.0, -1.0]),
    ([1.0, 1.0], [1.0, PHYSICAL_HEIGHT], [1.0, 0.0]),
    ([2.0, 1.0], [2.0, PHYSICAL_HEIGHT], [-1.0, 0.0]),
    ([1.0, 1.0], [2.0, 1.0], [0.0, 1.0]),
]
mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "StartPoint": start,
            "EndPoint": end,
            "Norm": normal,
            "CellThickness": 1,
        }
        for start, end, normal in walls
    ]
    + [
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0],
            "StartPoint": POROUS_ORIGIN,
            "EndPoint": [2.0, 1.0],
        }
    ]
)
mpm.select_save_data(particle=True, grid=False, object=False)
mpm.run()

if not SKIP_POSTPROCESS:
    mpm.postprocessing()
postprocess(OUTPUT)
