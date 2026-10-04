"""Three-dimensional two-point saturated slope with a device-side piston wavemaker.

The tank is filled with fluid points and the right-hand trapezoidal bed is a
second (solid) point set.  Thus water occupies the pore volume as well as the
open water above the slope.  The left-wall velocity is updated by a Taichi
callback after every accepted step; the time integration and boundary update
therefore remain on device.
"""

import argparse
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

from geotaichi import MPM, init
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d import (
    deep_air_particle_count,
    enforce_moving_piston_particles,
    harmonic_fit,
    linear_piston_wave,
    piston_displacement,
    piston_velocity,
    sdf_mean_surface,
    sdf_surface_profile_metrics,
    sdf_surface_gauge,
)
from examples.mpm.IncompressibleFluid.validate_large_tank_incompressible_3d import enclosed_air_count
from src.utils.SolverRuntime import python_callback

TANK = (2.40, 0.48, 0.72)
DX_DEFAULT = 0.02
# Keep seven cells of initial headspace: run-up compressed the former
# three-cell band to less than half a cell and made the pressure topology fail.
WATER_DEPTH = 0.48
WALL_CELLS = 1
SLOPE_HEIGHT = WATER_DEPTH / 3.0
SLOPE_TOE = 0.65
SLOPE_RUN = 0.70
PISTON_MEAN_X = 0.08
WAVE_FREQUENCY = 0.9
WAVE_VELOCITY = 0.24
WAVE_RAMP = 1.2
SOIL_YOUNG_MODULUS = 12.0e6
SOIL_COHESION = 300.0
SOIL_FRICTION = 26.0
GAUGE_X = (0.45, 1.20, 2.10)
GAUGE_HALF_WIDTH = 0.06


def parse_args():
    parser = argparse.ArgumentParser(description="3D two-layer, two-phase incompressible MPM wavemaker tank")
    parser.add_argument("--arch", default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--dx", type=float, default=float(os.environ.get("GT_WAVEMAKER_DX", DX_DEFAULT)))
    parser.add_argument("--dt", type=float, default=float(os.environ.get("GT_WAVEMAKER_DT", "1.0e-4")))
    parser.add_argument("--time", type=float, default=float(os.environ.get("GT_WAVEMAKER_TIME", "5.0")))
    parser.add_argument(
        "--save-interval", type=float, default=float(os.environ.get("GT_WAVEMAKER_SAVE_INTERVAL", "0.05"))
    )
    parser.add_argument("--fluid-ppc", type=int, default=int(os.environ.get("GT_WAVEMAKER_FLUID_PPC", "2")))
    parser.add_argument("--solid-ppc", type=int, default=int(os.environ.get("GT_WAVEMAKER_SOLID_PPC", "2")))
    parser.add_argument("--device-memory", type=float, default=4.0)
    parser.add_argument(
        "--pressure-iterations", type=int, default=int(os.environ.get("GT_WAVEMAKER_PRESSURE_ITERATIONS", "1000"))
    )
    parser.add_argument("--alpha-pic", type=float, default=0.30)
    parser.add_argument("--wave-velocity", type=float, default=WAVE_VELOCITY)
    parser.add_argument("--frequency", type=float, default=WAVE_FREQUENCY)
    parser.add_argument("--ramp-time", type=float, default=WAVE_RAMP)
    parser.add_argument("--maximum-porosity", type=float, default=0.56)
    parser.add_argument("--soil-young-modulus", type=float, default=SOIL_YOUNG_MODULUS)
    parser.add_argument("--soil-cohesion", type=float, default=SOIL_COHESION)
    parser.add_argument("--soil-friction", type=float, default=SOIL_FRICTION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "GT_WAVEMAKER_OUTPUT",
                ROOT / "examples" / "mmpm" / "TwoPhaseWavemaker3D" / "OutputData" / "two_layer_two_phase_wavemaker_3d",
            )
        ),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    return parser.parse_args()


def validate_geometry(args):
    if (
        min(args.dx, args.dt, args.time, args.save_interval, args.frequency, args.ramp_time, args.device_memory) <= 0.0
        or min(args.fluid_ppc, args.solid_ppc, args.pressure_iterations) <= 0
    ):
        raise ValueError("dx, dt, time, save interval, device memory, and particles per cell must be positive")
    if not 0.0 <= args.alpha_pic <= 1.0 or not 0.40 <= args.maximum_porosity <= 1.0 or args.wave_velocity < 0.0:
        raise ValueError(
            "alpha PIC must lie in [0, 1], maximum porosity in [0.40, 1], and wave velocity must be nonnegative"
        )
    if args.soil_young_modulus <= 0.0 or args.soil_cohesion < 0.0 or not 0.0 <= args.soil_friction < 90.0:
        raise ValueError("soil Young's modulus must be positive, cohesion nonnegative, and friction in [0, 90)")
    cells = np.asarray(TANK, dtype=float) / args.dx
    if not np.allclose(cells, np.round(cells), rtol=0.0, atol=1.0e-12):
        raise ValueError(f"dx={args.dx:g} must divide the tank dimensions {TANK}")
    if np.any(np.round(cells).astype(int) % 2):
        raise ValueError("two-level MGPCG requires an even number of cells along every tank axis")
    if not math.isclose(WATER_DEPTH / args.dx, round(WATER_DEPTH / args.dx), abs_tol=1.0e-12):
        raise ValueError("dx must also divide WATER_DEPTH")
    if PISTON_MEAN_X - args.wave_velocity / (2.0 * math.pi * args.frequency) <= args.dx:
        raise ValueError("piston stroke leaves insufficient clearance from x=0")


def make_slope_region(wall):
    slope_end = TANK[0] - wall

    def region_volume():
        plateau = (slope_end - (SLOPE_TOE + SLOPE_RUN)) * SLOPE_HEIGHT
        return TANK[1] * (plateau + 0.5 * SLOPE_RUN * SLOPE_HEIGHT)

    def region_function(position, particle_radius=0.0):
        local_x = position[0] - SLOPE_TOE
        top = ti.min(SLOPE_HEIGHT, SLOPE_HEIGHT * local_x / SLOPE_RUN)
        return (
            SLOPE_TOE <= position[0] <= slope_end
            and wall <= position[1] <= TANK[1] - wall
            and wall + 0.0 * particle_radius <= position[2] <= top
        )

    return {
        "Name": "right_trapezoidal_slope",
        "Type": "UserDefined",
        "BoundingBoxPoint": [SLOPE_TOE, wall, wall],
        "BoundingBoxSize": [slope_end - SLOPE_TOE, TANK[1] - 2.0 * wall, SLOPE_HEIGHT - wall],
        "RegionVolume": region_volume,
        "RegionFunction": region_function,
    }


@ti.kernel
def set_moving_piston_mac_boundary(
    velocity: float,
    wall_position: float,
    tank_width: float,
    tank_height: float,
    grid_size: ti.types.vector(3, float),
    cell_type: ti.template(),
    solid_velocity_x: ti.template(),
):
    for I in ti.grouped(cell_type):
        if (I[0] + 0.5) * grid_size[0] <= wall_position:
            cell_type[I] = 2
    for I in ti.grouped(solid_velocity_x):
        x = I[0] * grid_size[0]
        y = (I[1] + 0.5) * grid_size[1]
        z = (I[2] + 0.5) * grid_size[2]
        if x <= wall_position + grid_size[0] and 0.0 < y < tank_width and 0.0 < z < tank_height:
            solid_velocity_x[I] = velocity


@ti.kernel
def initialize_hydrostatic_state(
    particle_count: int, water_depth: float, friction_angle: float, particle: ti.template()
):
    k0 = 1.0 - ti.sin(friction_angle * math.pi / 180.0)
    for p in range(particle_count):
        water_head = ti.max(water_depth - particle[p].x[2], 0.0)
        particle[p].pressure = 1000.0 * 9.81 * water_head
        if int(particle[p].phase) == 1:
            slope_surface = ti.min(
                SLOPE_HEIGHT,
                ti.max(0.0, SLOPE_HEIGHT * (particle[p].x[0] - SLOPE_TOE) / SLOPE_RUN),
            )
            soil_head = ti.max(slope_surface - particle[p].x[2], 0.0)
            effective_vertical = (1.0 - particle[p].porosity) * (2650.0 - 1000.0) * 9.81 * soil_head
            particle[p].stress = ti.Vector(
                [-k0 * effective_vertical, -k0 * effective_vertical, -effective_vertical, 0.0, 0.0, 0.0]
            )


def evaluate_metrics(rows, expected_fluid, expected_solid, args):
    period = 1.0 / args.frequency
    wavenumber, expected_amplitude, group_speed = linear_piston_wave(args.frequency, WATER_DEPTH, args.wave_velocity)
    wavelength = 2.0 * math.pi / wavenumber
    analysis_start = args.ramp_time + GAUGE_X[0] / group_speed + 0.5 * period
    analysis_end = min(
        args.time,
        args.ramp_time + (2.0 * TANK[0] - GAUGE_X[0]) / group_speed - 0.25 * period,
    )
    selected = (rows[:, 0] >= analysis_start) & (rows[:, 0] <= analysis_end)
    incident_start = args.ramp_time + 0.5 * period
    incident_end = min(args.time, args.ramp_time + (TANK[0] - PISTON_MEAN_X) / group_speed)
    incident = (rows[:, 0] >= incident_start) & (rows[:, 0] <= incident_end)
    incident_rows = rows[incident] if np.any(incident) else rows
    fit_samples = int(np.count_nonzero(selected))
    amplitude = mean_surface = fit_r2 = float("nan")
    if fit_samples >= 6:
        amplitude, mean_surface, fit_r2 = harmonic_fit(rows[selected, 0], rows[selected, 8], args.frequency)
    global_mean_surface = float("nan")
    if fit_samples >= 6:
        _, global_mean_surface, _ = harmonic_fit(rows[selected, 0], rows[selected, 15], args.frequency)
    initial_surface = float(rows[0, 8])
    amplitude_error = abs(amplitude - expected_amplitude) / max(expected_amplitude, 1.0e-12)
    duration_complete = abs(float(rows[-1, 0]) - args.time) <= 0.1 * args.dt + 1.0e-12
    particle_conservation = bool(np.all(rows[:, 1] == expected_fluid) and np.all(rows[:, 2] == expected_solid))
    metrics = {
        "case": "3D two-layer two-phase incompressible MPM wavemaker over a right trapezoidal slope",
        "reference": "Zhang et al. (2027), CMAME 463:119401, Section 6.2; scaled Lagrangian piston wave",
        "tank_m": TANK,
        "water_depth_m": WATER_DEPTH,
        "slope_height_m": SLOPE_HEIGHT,
        "slope_height_over_tank": SLOPE_HEIGHT / TANK[2],
        "slope_height_over_water": SLOPE_HEIGHT / WATER_DEPTH,
        "slope_toe_m": SLOPE_TOE,
        "slope_run_m": SLOPE_RUN,
        "plateau_length_m": TANK[0] - args.dx * WALL_CELLS - SLOPE_TOE - SLOPE_RUN,
        "wavemaker_frequency_hz": args.frequency,
        "wavemaker_velocity_amplitude_mps": args.wave_velocity,
        "wavemaker_ramp_time_s": args.ramp_time,
        "wavemaker_type": "moving rectangular piston cell boundary",
        "piston_mean_x_m": PISTON_MEAN_X,
        "piston_stroke_amplitude_m": args.wave_velocity / (2.0 * math.pi * args.frequency),
        "expected_linear_wave_amplitude_m": expected_amplitude,
        "expected_linear_wave_height_m": 2.0 * expected_amplitude,
        "linear_wavelength_m": wavelength,
        "wavelengths_in_active_tank": (TANK[0] - PISTON_MEAN_X) / wavelength,
        "resolved_wave_amplitude_cells": expected_amplitude / args.dx,
        "linear_phase_speed_mps": 2.0 * math.pi * args.frequency / wavenumber,
        "linear_group_speed_mps": group_speed,
        "expected_fluid_particles": expected_fluid,
        "expected_solid_particles": expected_solid,
        "minimum_fluid_particles": int(np.min(rows[:, 1])),
        "minimum_solid_particles": int(np.min(rows[:, 2])),
        "particle_conservation": particle_conservation,
        "snapshots": len(rows),
        "final_time_s": float(rows[-1, 0]),
        "duration_complete": duration_complete,
        "maximum_fluid_speed_mps": float(np.max(rows[:, 4])),
        "maximum_solid_speed_mps": float(np.max(rows[:, 5])),
        "maximum_solid_displacement_m": float(np.max(rows[:, 23])),
        "maximum_particles_outside_tank": int(np.max(rows[:, 11])),
        "maximum_enclosed_air_cells": int(np.max(rows[:, 12])),
        "maximum_fluid_particles_in_air_cells": int(np.max(rows[:, 13])),
        "maximum_fluid_particles_in_deep_air_cells": int(np.max(rows[:, 14])),
        "maximum_fluid_particles_in_solid_cells": int(np.max(rows[:, 16])),
        "maximum_fluid_particles_behind_piston": int(np.max(rows[:, 19])),
        "maximum_cross_tank_surface_roughness_m": float(np.max(rows[:, 20])),
        "maximum_high_frequency_surface_roughness_m": float(np.max(rows[:, 21])),
        "maximum_instantaneous_wave_height_m": float(np.max(rows[:, 22])),
        "incident_wave_window_s": [incident_start, incident_end],
        "maximum_incident_cross_tank_surface_roughness_m": float(incident_rows[:, 20].max()),
        "maximum_incident_high_frequency_surface_roughness_m": float(incident_rows[:, 21].max()),
        "minimum_solid_porosity": float(np.min(rows[:, 17])),
        "maximum_solid_porosity": float(np.max(rows[:, 18])),
        "soil_young_modulus_pa": args.soil_young_modulus,
        "soil_cohesion_pa": args.soil_cohesion,
        "soil_friction_deg": args.soil_friction,
        "fluid_surface_excursion_m": float(np.max(rows[:, 3]) - np.min(rows[:, 3])),
        "surface_gauge_x_m": GAUGE_X,
        "surface_gauge_excursion_m": {
            str(gauge_x): float(np.max(rows[:, 8 + index]) - np.min(rows[:, 8 + index]))
            for index, gauge_x in enumerate(GAUGE_X)
        },
        "left_gauge_analysis_start_s": analysis_start,
        "left_gauge_analysis_end_s": analysis_end,
        "left_gauge_analysis_samples": fit_samples,
        "left_gauge_harmonic_amplitude_m": amplitude,
        "left_gauge_harmonic_fit_r2": fit_r2,
        "left_gauge_amplitude_relative_error": amplitude_error,
        "left_gauge_mean_surface_drift_m": mean_surface - initial_surface,
        "global_mean_surface_drift_m": global_mean_surface - float(rows[0, 15]),
        "measured_wave_amplitude_cells": amplitude / args.dx,
        "finite": bool(np.isfinite(rows).all()),
        "strict_tolerances": {
            "amplitude_relative_error": 0.55,
            "harmonic_fit_r2": 0.70,
            "global_surface_drift_m": 0.50 * args.dx,
            "maximum_fluid_speed_mps": 1.25 * math.sqrt(2.0 * 9.81 * WATER_DEPTH),
            "maximum_solid_speed_mps": 1.0,
            "minimum_solid_displacement_m": 0.05 * args.dx,
            "surface_roughness_m": 0.50 * args.dx,
        },
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and duration_complete
        and particle_conservation
        and metrics["maximum_particles_outside_tank"] == 0
        and metrics["maximum_enclosed_air_cells"] == 0
        and metrics["maximum_fluid_particles_in_deep_air_cells"] == 0
        and metrics["maximum_fluid_particles_in_solid_cells"] == 0
        and metrics["maximum_fluid_particles_behind_piston"] == 0
        and 0.0 < metrics["minimum_solid_porosity"]
        and metrics["maximum_solid_porosity"] <= args.maximum_porosity + 1.0e-10
        and fit_samples >= 6
        and amplitude_error <= metrics["strict_tolerances"]["amplitude_relative_error"]
        and metrics["measured_wave_amplitude_cells"] >= 1.0
        and fit_r2 >= metrics["strict_tolerances"]["harmonic_fit_r2"]
        and abs(metrics["global_mean_surface_drift_m"]) <= metrics["strict_tolerances"]["global_surface_drift_m"]
        and metrics["wavelengths_in_active_tank"] >= 1.25
        and metrics["resolved_wave_amplitude_cells"] >= 1.75
        and metrics["maximum_incident_cross_tank_surface_roughness_m"]
        <= metrics["strict_tolerances"]["surface_roughness_m"]
        and metrics["maximum_incident_high_frequency_surface_roughness_m"]
        <= metrics["strict_tolerances"]["surface_roughness_m"]
        and metrics["maximum_instantaneous_wave_height_m"] >= 1.25 * expected_amplitude
        and metrics["surface_gauge_excursion_m"][str(GAUGE_X[1])] >= 0.5 * expected_amplitude
        and metrics["maximum_fluid_speed_mps"] < metrics["strict_tolerances"]["maximum_fluid_speed_mps"]
        and metrics["maximum_solid_speed_mps"] < metrics["strict_tolerances"]["maximum_solid_speed_mps"]
        and metrics["maximum_solid_displacement_m"] >= metrics["strict_tolerances"]["minimum_solid_displacement_m"]
    )
    return metrics


def write_metrics(output, expected_fluid, expected_solid, args):
    files = sorted(glob.glob(str(output / "particles" / "MPMParticle*.npz")))
    grid_files = {
        Path(path).stem.removeprefix("MPMGrid"): path for path in glob.glob(str(output / "grids" / "MPMGrid*.npz"))
    }
    if len(files) < 2:
        raise RuntimeError("wavemaker run produced fewer than two particle snapshots")

    rows = []
    initial_solid_position = None
    for file_name in files:
        frame = Path(file_name).stem.removeprefix("MPMParticle")
        grid_name = grid_files.get(frame)
        if grid_name is None:
            raise RuntimeError(f"missing grid snapshot for particle frame {frame}")
        with np.load(file_name) as data, np.load(grid_name) as grid:
            active = data["active"] > 0
            phase = data["phase"]
            fluid = active & (phase == 2)
            solid = active & (phase == 1)
            position = data["position"]
            fluid_velocity = data["fluid_velocity"]
            solid_velocity = data["solid_velocity"]
            pressure = data["pressure"]
            porosity = data["porosity"]
            if initial_solid_position is None:
                initial_solid_position = position[solid].copy()
            solid_displacement = float(np.linalg.norm(position[solid] - initial_solid_position, axis=1).max())
            piston_position = PISTON_MEAN_X + piston_displacement(
                float(data["t_current"]), args.frequency, args.wave_velocity, args.ramp_time
            )
            outside = active & np.any(
                (position < -1.0e-10 * args.dx) | (position > np.asarray(TANK) + 1.0e-10 * args.dx), axis=1
            )
            cell_type = np.squeeze(grid["cell_type"])
            enclosed_air = 0
            particles_in_air = 0
            particles_in_deep_air = 0
            particles_in_solid = 0
            if float(data["t_current"]) > 0.0:
                enclosed_air = enclosed_air_count(cell_type)
                fluid_cell = np.floor(position[fluid] / args.dx).astype(np.int64)
                fluid_cell = np.minimum(np.maximum(fluid_cell, 0), np.asarray(cell_type.shape) - 1)
                occupied_type = cell_type[tuple(fluid_cell.T)]
                particles_in_air = int(np.count_nonzero(occupied_type == 0))
                particles_in_deep_air = deep_air_particle_count(cell_type, fluid_cell)
                particles_in_solid = int(
                    np.count_nonzero((occupied_type == 2) & (position[fluid, 0] > piston_position + args.dx))
                )
            initial_frame = math.isclose(float(data["t_current"]), 0.0, rel_tol=0.0, abs_tol=0.1 * args.dt)
            gauge_surface = (
                [WATER_DEPTH] * len(GAUGE_X)
                if initial_frame
                else [
                    sdf_surface_gauge(grid["cell_type"], grid["cell_fluid_sdf"], args.dx, gauge_x, GAUGE_HALF_WIDTH)
                    for gauge_x in GAUGE_X
                ]
            )
            global_surface = (
                WATER_DEPTH if initial_frame else sdf_mean_surface(grid["cell_type"], grid["cell_fluid_sdf"], args.dx)
            )
            surface_profile = (
                (0.0, 0.0, 0.0)
                if initial_frame
                else sdf_surface_profile_metrics(grid["cell_type"], grid["cell_fluid_sdf"], args.dx)
            )
            rows.append(
                [
                    float(data["t_current"]),
                    int(np.count_nonzero(fluid)),
                    int(np.count_nonzero(solid)),
                    float(position[fluid, 2].max()),
                    float(np.linalg.norm(fluid_velocity[fluid], axis=1).max()),
                    float(np.linalg.norm(solid_velocity[solid], axis=1).max()),
                    float(pressure[fluid].min()),
                    float(pressure[fluid].max()),
                    *gauge_surface,
                    int(np.count_nonzero(outside)),
                    enclosed_air,
                    particles_in_air,
                    particles_in_deep_air,
                    global_surface,
                    particles_in_solid,
                    float(np.min(porosity[solid])),
                    float(np.max(porosity[solid])),
                    int(np.count_nonzero(position[fluid, 0] < piston_position - 1.0e-6 * args.dx)),
                    *surface_profile,
                    solid_displacement,
                ]
            )
    rows = np.asarray(rows, dtype=np.float64)
    np.savetxt(
        output / "wavemaker_diagnostics.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,fluid_particles,solid_particles,fluid_top_z_m,fluid_speed_max_mps,"
            "solid_speed_max_mps,fluid_pressure_min_pa,fluid_pressure_max_pa,"
            "surface_left_m,surface_mid_m,surface_right_m"
            ",particles_outside_tank,enclosed_air_cells,fluid_particles_in_air_cells,"
            "fluid_particles_in_deep_air_cells,global_mean_surface_m,"
            "fluid_particles_in_solid_cells,solid_porosity_min,solid_porosity_max,fluid_particles_behind_piston,"
            "cross_tank_surface_roughness_m,high_frequency_surface_roughness_m,instantaneous_wave_height_m,"
            "solid_displacement_max_m"
        ),
        comments="",
    )
    metrics = evaluate_metrics(rows, expected_fluid, expected_solid, args)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("wavemaker validation failed; inspect wavemaker_diagnostics.csv")


def run(args):
    validate_geometry(args)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    wall = WALL_CELLS * args.dx
    tank_cells = np.round(np.asarray(TANK) / args.dx).astype(int)
    boundary_layers = WALL_CELLS + 1
    static_velocity_constraints = boundary_layers * (
        (tank_cells[0] + 1) * (tank_cells[1] + 1)
        + 2 * (tank_cells[0] + 1) * (tank_cells[2] + 1)
        + (tank_cells[1] + 1) * (tank_cells[2] + 1)
    )
    expected_fluid = math.ceil(
        (TANK[0] - PISTON_MEAN_X - wall)
        * (TANK[1] - 2.0 * wall)
        * (WATER_DEPTH - wall)
        * args.fluid_ppc**3
        / args.dx**3
    )
    expected_solid = math.ceil(
        (TANK[0] - SLOPE_TOE - wall) * (TANK[1] - 2.0 * wall) * (SLOPE_HEIGHT - wall) * args.solid_ppc**3 / args.dx**3
    )
    max_particles = expected_fluid + expected_solid
    print(
        f"# 3D wavemaker tank: dx={args.dx:g}, water~{expected_fluid}, soil capacity~{expected_solid}, "
        f"slope height={SLOPE_HEIGHT:g} m ({SLOPE_HEIGHT / WATER_DEPTH:.2f} water depth), "
        f"plateau={TANK[0] - wall - SLOPE_TOE - SLOPE_RUN:g} m"
    )

    init(
        dim=3,
        arch=args.arch,
        default_fp="float64",
        device_memory_GB=args.device_memory,
        offline_cache=True,
        debug=False,
    )
    mpm = MPM()
    mpm.set_configuration(
        domain=list(TANK),
        background_damping=0.03,
        gravity=[0.0, 0.0, -9.81],
        alphaPIC=args.alpha_pic,
        mapping="USL",
        shape_function="QuadBSpline",
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
        velocity_projection="Affine",
        delayed_fluid_advection=True,
        particle_shifting=True,
        # DoubleLayer builds its own fluid SDF.  The generic density detector
        # assumes a single-phase velocity_gradient field.
        free_surface_detection=False,
        visualize=True,
    )
    mpm.set_semi_implicit_solver_parameters(
        {
            "assemble_type": "MatrixFree",
            "pressure_solver": "MGPCG",
            "linear_solver": "MGPCG",
            "max_iteration_number": args.pressure_iterations,
            "residual_tolerance": 1.0e-7,
            "multilevel": 2,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    mpm.set_solver(
        {"Timestep": args.dt, "SimulationTime": args.time, "SaveInterval": args.save_interval, "SavePath": str(output)}
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": max_particles,
            "max_constraint_number": {"max_velocity_constraint": static_velocity_constraints},
        }
    )
    mpm.add_material(
        model="MohrCoulomb",
        material={
            "MaterialID": 1,
            "SolidDensity": 2650.0,
            "FluidDensity": 1000.0,
            "Porosity": 0.40,
            "MaximumPorosity": args.maximum_porosity,
            "FluidBulkModulus": 2.2e8,
            "Permeability": 2.0e-4,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 3.0e-3,
            "DragModel": "Ergun",
            "YoungModulus": args.soil_young_modulus,
            "PoissonRatio": 0.30,
            "Cohesion": args.soil_cohesion,
            "Friction": args.soil_friction,
            "Dilation": 0.0,
        },
    )
    mpm.add_element({"ElementType": "R8N3D", "ElementSize": [args.dx, args.dx, args.dx]})
    mpm.add_region(
        [
            {
                "Name": "water",
                "Type": "Rectangle",
                "BoundingBoxPoint": [PISTON_MEAN_X, wall, wall],
                "BoundingBoxSize": [
                    TANK[0] - PISTON_MEAN_X - wall,
                    TANK[1] - 2.0 * wall,
                    WATER_DEPTH - wall,
                ],
            },
            make_slope_region(wall),
        ]
    )
    mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "water",
                    "nParticlesPerCell": args.fluid_ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
                {
                    "RegionName": "right_trapezoidal_slope",
                    "nParticlesPerCell": args.solid_ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Solid",
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
            ]
        }
    )
    particle_count = int(mpm.scene.particleNum[0])
    initial_phase = mpm.scene.particle.phase.to_numpy()[:particle_count]
    expected_fluid = int(np.count_nonzero(initial_phase == 2))
    expected_solid = int(np.count_nonzero(initial_phase == 1))
    initialize_hydrostatic_state(particle_count, WATER_DEPTH, args.soil_friction, mpm.scene.particle)

    walls = [
        ([wall, wall, wall], [TANK[0] - wall, wall, TANK[2]], [0.0, -1.0, 0.0]),
        ([wall, TANK[1] - wall, wall], [TANK[0] - wall, TANK[1] - wall, TANK[2]], [0.0, 1.0, 0.0]),
        ([wall, wall, wall], [TANK[0] - wall, TANK[1] - wall, wall], [0.0, 0.0, -1.0]),
        ([TANK[0] - wall, wall, wall], [TANK[0] - wall, TANK[1] - wall, TANK[2]], [1.0, 0.0, 0.0]),
    ]
    mpm.add_boundary_condition(
        [
            {
                "BoundaryType": "SolidCell",
                "StartPoint": start,
                "EndPoint": end,
                "Norm": normal,
                "CellThickness": WALL_CELLS,
            }
            for start, end, normal in walls
        ]
        + [
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, None, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [TANK[0], TANK[1], wall],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0, None],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [TANK[0], wall, TANK[2]],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0, None],
                "StartPoint": [0.0, TANK[1] - wall, 0.0],
                "EndPoint": [TANK[0], TANK[1], TANK[2]],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [TANK[0] - wall, 0.0, 0.0],
                "EndPoint": [TANK[0], TANK[1], TANK[2]],
            },
        ]
    )

    def update_wavemaker(sims, scene):
        start_time = float(sims.current_time)
        end_time = min(start_time + float(sims.delta), args.time)
        velocity = piston_velocity(0.5 * (start_time + end_time), args.frequency, args.wave_velocity, args.ramp_time)
        position = PISTON_MEAN_X + piston_displacement(end_time, args.frequency, args.wave_velocity, args.ramp_time)
        set_moving_piston_mac_boundary(
            velocity,
            position,
            TANK[1],
            TANK[2],
            scene.element.grid_size,
            mpm.enginer.cell_type,
            mpm.enginer.solid_velocity_x,
        )

    @python_callback
    def enforce_wavemaker_particles():
        start_time = float(mpm.sims.current_time)
        end_time = min(start_time + float(mpm.sims.delta), args.time)
        velocity = piston_velocity(0.5 * (start_time + end_time), args.frequency, args.wave_velocity, args.ramp_time)
        position = PISTON_MEAN_X + piston_displacement(end_time, args.frequency, args.wave_velocity, args.ramp_time)
        enforce_moving_piston_particles(int(mpm.scene.particleNum[0]), position, velocity, args.dx, mpm.scene.particle)

    mpm.select_save_data(particle=True, grid=True, object=False)
    mpm.run(mac_boundary_function=update_wavemaker, function=enforce_wavemaker_particles)
    if not args.no_post:
        mpm.postprocessing()
    write_metrics(output, expected_fluid, expected_solid, args)


if __name__ == "__main__":
    run(parse_args())
