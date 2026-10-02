"""Juel et al. Section 5.5 case 1: dam break through a rigid porous column."""

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

PREFIX = "GT_DOUBLE_POINT_POROUS_DAM_"


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(PREFIX + name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(PREFIX + name, default))


ARCH = os.environ.get(PREFIX + "ARCH", "gpu")
OUTPUT = Path(os.environ.get(PREFIX + "OUTPUT", Path(__file__).resolve().parent / "OutputData")).expanduser().resolve()
DX = env_float("DX", 0.01)
DT = env_float("DT", 1.0e-3)
SIMULATION_TIME = env_float("TIME", 1.2)
SAVE_INTERVAL = env_float("SAVE_INTERVAL", 0.1)
ALPHA_PIC = env_float("ALPHA_PIC", 1.0)
PARTICLE_SHIFTING = os.environ.get(PREFIX + "PARTICLE_SHIFTING", "1") == "1"
FLUID_PPC = env_int("FLUID_PPC", 4)
SOLID_PPC = env_int("SOLID_PPC", 4)
POROSITY = 0.39
GRAIN_DIAMETER = 3.0e-3
# Eq. (22), the zero-relative-velocity limit of the Beetstra model.
SOLID_FRACTION = 1.0 - POROSITY
PERMEABILITY = (
    GRAIN_DIAMETER**2
    / 18.0
    / (10.0 * SOLID_FRACTION**2 / POROSITY**3 + SOLID_FRACTION * POROSITY * (1.0 + 1.5 * math.sqrt(SOLID_FRACTION)))
)
YOUNG_MODULUS = env_float("YOUNG_MODULUS", 4.0e7)
SKIP_POSTPROCESS = os.environ.get(PREFIX + "SKIP_POSTPROCESS", "0") == "1"

# The physical flume is embedded in a cell-aligned computational box.  Walls
# remain at the experimental dimensions.
EXPERIMENTAL_DOMAIN = [0.892, 0.37]
DOMAIN = [0.90, 0.38]
BASE_WATER_ORIGIN = [0.0, 0.0]
BASE_WATER_SIZE = [EXPERIMENTAL_DOMAIN[0], 0.025]
UPPER_WATER_ORIGIN = [0.0, BASE_WATER_SIZE[1]]
UPPER_WATER_SIZE = [0.28, 0.14 - BASE_WATER_SIZE[1]]
POROUS_ORIGIN = [0.30, 0.0]
POROUS_SIZE = [0.29, 0.37]


def aligned_count(length: float, spacing: float) -> int:
    count = round(length / spacing)
    if not math.isclose(count * spacing, length, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"geometry length {length:g} is not aligned with spacing={spacing:g}")
    return count


DOMAIN_CELLS = [aligned_count(length, DX) for length in DOMAIN]
POROUS_RIGHT = POROUS_ORIGIN[0] + POROUS_SIZE[0]
FLUID_PARTICLE_VOLUME = DX**2 / FLUID_PPC**2
SOLID_PARTICLE_VOLUME = DX**2 / SOLID_PPC**2
EXPECTED_FLUID_PARTICLES = math.ceil(math.prod(BASE_WATER_SIZE) / FLUID_PARTICLE_VOLUME) + math.ceil(
    math.prod(UPPER_WATER_SIZE) / FLUID_PARTICLE_VOLUME
)
EXPECTED_SOLID_PARTICLES = math.ceil(math.prod(POROUS_SIZE) / SOLID_PARTICLE_VOLUME)
MAX_PARTICLES = EXPECTED_FLUID_PARTICLES + EXPECTED_SOLID_PARTICLES
MAX_VELOCITY_CONSTRAINTS = 2 * (aligned_count(POROUS_SIZE[0], DX) + 1) * (aligned_count(POROUS_SIZE[1], DX) + 1)

if min(DX, DT, SIMULATION_TIME, SAVE_INTERVAL, PERMEABILITY, YOUNG_MODULUS) <= 0.0 or min(FLUID_PPC, SOLID_PPC) <= 0:
    raise ValueError("spacing, time, permeability, modulus, and particles per cell must be positive")
if any(count % 2 for count in DOMAIN_CELLS):
    raise ValueError("the two-level MGPCG grid must have even cell counts")


@ti.kernel
def initialize_hydrostatic_pressure(particle_count: int, particle: ti.template()):
    for p in range(particle_count):
        if int(particle[p].phase) == 2:
            water_top = BASE_WATER_SIZE[1]
            if particle[p].x[0] <= UPPER_WATER_SIZE[0]:
                water_top = UPPER_WATER_ORIGIN[1] + UPPER_WATER_SIZE[1]
            particle[p].pressure = 1000.0 * 9.81 * ti.max(water_top - particle[p].x[1], 0.0)


def postprocess(output: Path):
    files = sorted(glob.glob(str(output / "particles" / "MPMParticle*.npz")))
    if len(files) < 2:
        raise RuntimeError("double-point recorder produced fewer than two particle states")
    rows = []
    initial_solid_position = None
    initial_solid_volume = None
    upstream_ids = None
    left_wall_initial_y = None
    left_wall_ids = None
    for file_name in files:
        data = np.load(file_name)
        active = data["active"] > 0
        phase = data["phase"]
        solid = active & (phase == 1)
        fluid = active & (phase == 2)
        position = data["position"]
        particle_id = data["particleID"]
        pressure = data["pressure"]
        fluid_velocity = data["fluid_velocity"]
        solid_position = position[solid]
        solid_volume = data["volume"][solid]
        solid_porosity = data["porosity"][solid]
        solid_speed = np.linalg.norm(data["solid_velocity"][solid], axis=1)
        if initial_solid_position is None:
            initial_solid_position = solid_position.copy()
            initial_solid_volume = float(np.median(solid_volume))
            upstream_ids = particle_id[fluid & (position[:, 0] <= UPPER_WATER_SIZE[0])]
            left_wall = fluid & (position[:, 0] < DX / FLUID_PPC)
            left_wall_ids = particle_id[left_wall]
            left_wall_initial_y = position[left_wall, 1].copy()
        upstream = fluid & np.isin(particle_id, upstream_ids)
        inside = (
            fluid
            & (position[:, 0] >= POROUS_ORIGIN[0])
            & (position[:, 0] <= POROUS_RIGHT)
            & (position[:, 1] <= POROUS_ORIGIN[1] + POROUS_SIZE[1])
        )
        through = fluid & (position[:, 0] > POROUS_RIGHT) & (position[:, 1] <= POROUS_ORIGIN[1] + POROUS_SIZE[1])
        transmitted = through & np.isin(particle_id, upstream_ids)
        fluid_speed = np.linalg.norm(fluid_velocity[fluid], axis=1)
        rows.append(
            [
                float(data["t_current"]),
                float(np.max(position[upstream, 0])),
                int(np.count_nonzero(inside)),
                int(np.count_nonzero(through)),
                int(np.count_nonzero(transmitted)),
                float(np.mean(fluid_velocity[inside, 0])) if np.any(inside) else 0.0,
                float(np.min(pressure[fluid])),
                float(np.max(pressure[fluid])),
                float(np.max(np.linalg.norm(solid_position - initial_solid_position, axis=1))),
                float(np.quantile(fluid_speed, 0.99)),
                float(np.max(fluid_speed)),
                float(np.min(solid_porosity)),
                float(np.max(solid_porosity)),
                float(np.min(solid_volume) / initial_solid_volume),
                float(np.max(solid_volume) / initial_solid_volume),
                float(np.max(solid_speed)),
            ]
        )
    rows = np.asarray(rows, dtype=np.float64)
    final_id_to_y = {int(pid): float(y) for pid, y in zip(particle_id[fluid], position[fluid, 1])}
    left_wall_vertical_displacement = (
        np.asarray([final_id_to_y[int(pid)] for pid in left_wall_ids]) - left_wall_initial_y
    )
    surface_bins = np.arange(0.0, EXPERIMENTAL_DOMAIN[0] + DX, DX)
    surface = np.full(surface_bins.size - 1, np.nan)
    for i, left in enumerate(surface_bins[:-1]):
        in_bin = fluid & (position[:, 0] >= left) & (position[:, 0] < left + DX)
        if np.any(in_bin):
            surface[i] = np.quantile(position[in_bin, 1], 0.98)
    wet_surface = surface[np.isfinite(surface)]
    surface_total_variation = float(np.mean(np.abs(np.diff(wet_surface))))
    np.savetxt(
        output / "porous_dam_break_diagnostics.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,water_front_x_m,fluid_points_inside_porous,fluid_points_through_porous,"
            "upstream_fluid_points_transmitted,mean_fluid_vx_inside_mps,pressure_min_pa,pressure_max_pa,"
            "max_solid_displacement_m,fluid_speed_p99_mps,fluid_speed_max_mps"
            ",solid_porosity_min,solid_porosity_max,solid_volume_ratio_min,solid_volume_ratio_max,solid_speed_max_mps"
        ),
        comments="",
    )
    metrics = {
        "case": "Juel et al. Section 5.5 porous dam-break case 1",
        "reference": "Juel et al. (2026), CMAME 461, 119140, Section 5.5",
        "experimental_domain_m": EXPERIMENTAL_DOMAIN,
        "domain_m": DOMAIN,
        "cell_counts": DOMAIN_CELLS,
        "element_size_m": DX,
        "timestep_s": DT,
        "fluid_particles_per_cell_axis": FLUID_PPC,
        "solid_particles_per_cell_axis": SOLID_PPC,
        "fluid_material_point_spacing_m": DX / FLUID_PPC,
        "solid_material_point_spacing_m": DX / SOLID_PPC,
        "fluid_particle_count": int(np.count_nonzero(np.load(files[0])["phase"] == 2)),
        "solid_particle_count": int(np.count_nonzero(np.load(files[0])["phase"] == 1)),
        "particle_capacity": MAX_PARTICLES,
        "initial_water_column_m": [0.28, 0.14],
        "initial_shallow_water_depth_m": BASE_WATER_SIZE[1],
        "porous_column_origin_m": POROUS_ORIGIN,
        "porous_column_size_m": POROUS_SIZE,
        "porosity": POROSITY,
        "grain_diameter_m": GRAIN_DIAMETER,
        "rest_permeability_m2": PERMEABILITY,
        "drag_model": "Beetstra Eq. (17)-(21)",
        "alpha_pic": ALPHA_PIC,
        "particle_shifting": PARTICLE_SHIFTING,
        "solid_model": "rigid (both velocity components constrained over the full porous column)",
        "maximum_fluid_points_inside_porous": int(np.max(rows[:, 2])),
        "maximum_fluid_points_through_porous": int(np.max(rows[:, 3])),
        "maximum_upstream_fluid_points_transmitted": int(np.max(rows[:, 4])),
        "final_upstream_fluid_points_transmitted": int(rows[-1, 4]),
        "maximum_solid_displacement_m": float(np.max(rows[:, 8])),
        "maximum_fluid_speed_p99_mps": float(np.max(rows[:, 9])),
        "maximum_fluid_speed_mps": float(np.max(rows[:, 10])),
        "minimum_solid_porosity": float(np.min(rows[:, 11])),
        "maximum_solid_porosity": float(np.max(rows[:, 12])),
        "minimum_solid_volume_ratio": float(np.min(rows[:, 13])),
        "maximum_solid_volume_ratio": float(np.max(rows[:, 14])),
        "maximum_solid_speed_mps": float(np.max(rows[:, 15])),
        "left_wall_particle_count": int(left_wall_ids.size),
        "left_wall_vertically_mobile_count": int(np.count_nonzero(np.abs(left_wall_vertical_displacement) > 1.0e-5)),
        "final_surface_total_variation_m": surface_total_variation,
        "finite": bool(np.isfinite(rows).all()),
    }
    gravity_velocity_scale = math.sqrt(9.81 * 0.14)
    metrics["gravity_velocity_scale_mps"] = gravity_velocity_scale
    metrics["initial_water_connected"] = True
    metrics["left_wall_vertically_mobile_fraction"] = (
        metrics["left_wall_vertically_mobile_count"] / metrics["left_wall_particle_count"]
    )
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["maximum_fluid_points_inside_porous"] > 0
        and np.max(rows[:, 1]) >= POROUS_ORIGIN[0] + 0.1
        and metrics["left_wall_vertically_mobile_fraction"] >= 0.95
        and metrics["final_surface_total_variation_m"] <= 3.0e-3
        and metrics["maximum_solid_displacement_m"] <= 1.0e-10
        and metrics["maximum_fluid_speed_p99_mps"] <= 3.0 * gravity_velocity_scale
        and metrics["maximum_fluid_speed_mps"] <= 5.0 * gravity_velocity_scale
        and 0.0 < metrics["minimum_solid_porosity"]
        and metrics["maximum_solid_porosity"] < 1.0
        and metrics["minimum_solid_volume_ratio"] > 0.0
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps(metrics, sort_keys=True))
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].plot(rows[:, 0], rows[:, 1], label="water front")
    axes[0].axhspan(POROUS_ORIGIN[0], POROUS_RIGHT, color="0.85", label="porous block")
    axes[0].set(xlabel="time (s)", ylabel="x (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(rows[:, 0], rows[:, 2], label="inside")
    axes[1].plot(rows[:, 0], rows[:, 4], label="upstream points transmitted")
    axes[1].set(xlabel="time (s)", ylabel="fluid material points")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "porous_dam_break_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.scatter(position[solid, 0], position[solid, 1], s=0.15, color="0.82")
    points = ax.scatter(position[fluid, 0], position[fluid, 1], s=0.6, c=pressure[fluid], cmap="coolwarm")
    ax.set(
        xlim=(0.0, EXPERIMENTAL_DOMAIN[0]),
        ylim=(0.0, EXPERIMENTAL_DOMAIN[1]),
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title=f"Case 1 at t={rows[-1, 0]:g} s",
    )
    fig.colorbar(points, ax=ax, label="pressure (Pa)")
    fig.tight_layout()
    fig.savefig(output / "porous_dam_break_final.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, sort_keys=True))


if os.environ.get(PREFIX + "POSTPROCESS_ONLY", "0") == "1":
    postprocess(OUTPUT)
    raise SystemExit


print("# Juel et al. Section 5.5 case 1: dam break through a rigid porous column")
print(f"# domain={DOMAIN}, cells={DOMAIN_CELLS}, dx={DX:g}, " f"fluid_ppc={FLUID_PPC}, solid_ppc={SOLID_PPC}")
print(
    f"# exact capacity={MAX_PARTICLES} ({EXPECTED_FLUID_PARTICLES} fluid + "
    f"{EXPECTED_SOLID_PARTICLES} solid), Beetstra rest permeability={PERMEABILITY:g} m^2"
)

init(dim=2, arch=ARCH, default_fp="float64", device_memory_GB=env_float("DEVICE_MEMORY_GB", 4.0))
mpm = MPM()
mpm.set_configuration(
    domain=DOMAIN,
    background_damping=0.01,
    gravity=[0.0, -9.81],
    alphaPIC=ALPHA_PIC,
    mapping="USL",
    shape_function="QuadBSpline",
    material_type="TwoPhaseDoubleLayer",
    solver_type="SemiImplicit",
    velocity_projection="Affine",
    delayed_fluid_advection=True,
    particle_shifting=PARTICLE_SHIFTING,
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
        "max_material_number": 1,
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
        "Permeability": PERMEABILITY,
        "FluidViscosity": 1.0e-3,
        "GrainDiameter": GRAIN_DIAMETER,
        "DragModel": "Beetstra",
        "YoungModulus": YOUNG_MODULUS,
        "PoissonRatio": 0.25,
    },
)
mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [DX, DX]})
mpm.add_region(
    region=[
        {
            "Name": "base_water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": BASE_WATER_ORIGIN,
            "BoundingBoxSize": BASE_WATER_SIZE,
        },
        {
            "Name": "upper_water_column",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": UPPER_WATER_ORIGIN,
            "BoundingBoxSize": UPPER_WATER_SIZE,
        },
        {
            "Name": "rigid_porous_block",
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
                "RegionName": "base_water",
                "nParticlesPerCell": FLUID_PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Fluid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            },
            {
                "RegionName": "upper_water_column",
                "nParticlesPerCell": FLUID_PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Fluid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            },
            {
                "RegionName": "rigid_porous_block",
                "nParticlesPerCell": SOLID_PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Solid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Fix", "Fix"],
            },
        ]
    }
)
initialize_hydrostatic_pressure(int(mpm.scene.particleNum[0]), mpm.scene.particle)

walls = [
    ([0.0, 0.0], [0.0, EXPERIMENTAL_DOMAIN[1]], [-1.0, 0.0]),
    ([EXPERIMENTAL_DOMAIN[0], 0.0], EXPERIMENTAL_DOMAIN, [1.0, 0.0]),
    ([0.0, 0.0], [EXPERIMENTAL_DOMAIN[0], 0.0], [0.0, -1.0]),
    ([0.0, EXPERIMENTAL_DOMAIN[1]], EXPERIMENTAL_DOMAIN, [0.0, 1.0]),
]
mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "StartPoint": start,
            "EndPoint": end,
            "Norm": normal,
            "CellThickness": 2,
        }
        for start, end, normal in walls
    ]
    + [
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0],
            "StartPoint": POROUS_ORIGIN,
            "EndPoint": [POROUS_RIGHT, POROUS_ORIGIN[1] + POROUS_SIZE[1]],
        }
    ]
)
mpm.select_save_data(particle=True, grid=False, object=False)
mpm.run()

if not SKIP_POSTPROCESS:
    mpm.postprocessing()
postprocess(OUTPUT)
