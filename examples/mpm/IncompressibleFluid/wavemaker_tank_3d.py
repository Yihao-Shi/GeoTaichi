"""3-D incompressible MPM tank with a left-wall piston wavemaker."""

import argparse
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

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")

from geotaichi import MPM, init  # noqa: E402
from src.mpm.engines.EngineKernel import kernel_update_solid_sdf_from_box  # noqa: E402
from src.utils.SolverRuntime import python_callback  # noqa: E402
from examples.mpm.IncompressibleFluid.validate_large_tank_incompressible_3d import enclosed_air_count  # noqa: E402

TANK = np.array([2.40, 0.32, 0.64])
WATER_DEPTH = 0.36
PISTON_MEAN_X = 0.08
GAUGE_X = (0.60, 1.20, 1.80)
WATER_ORIGIN = np.array([PISTON_MEAN_X, 0.0, 0.0])
WATER_SIZE = np.array([TANK[0] - PISTON_MEAN_X, TANK[1], WATER_DEPTH])
WAVE_FREQUENCY = 0.9
WAVE_VELOCITY = 0.24
WAVE_RAMP = 1.2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--time", type=float, default=5.5)
    parser.add_argument("--save-interval", type=float, default=0.05)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--velocity-amplitude", type=float, default=WAVE_VELOCITY)
    parser.add_argument("--alpha-pic", type=float, default=0.15)
    parser.add_argument("--frequency", type=float, default=WAVE_FREQUENCY)
    parser.add_argument("--ramp-time", type=float, default=WAVE_RAMP)
    parser.add_argument("--device-memory", type=float, default=7.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "OutputData" / "wavemaker_tank_3d",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()
    if min(args.dx, args.dt, args.time, args.save_interval, args.frequency, args.ramp_time) <= 0 or args.ppc <= 0:
        parser.error("spacing, time controls, frequency, ramp time, and ppc must be positive")
    if args.velocity_amplitude < 0:
        parser.error("--velocity-amplitude must be nonnegative")
    if PISTON_MEAN_X - args.velocity_amplitude / (2.0 * math.pi * args.frequency) <= 0.5 * args.dx:
        parser.error("piston stroke leaves insufficient clearance from x=0")
    if not 0.0 <= args.alpha_pic <= 1.0:
        parser.error("--alpha-pic must be in [0, 1]")
    counts = TANK / args.dx
    if not np.allclose(counts, np.round(counts), atol=1.0e-12, rtol=0.0):
        parser.error("--dx must divide all tank dimensions")
    if not math.isclose(round(WATER_DEPTH / args.dx) * args.dx, WATER_DEPTH, abs_tol=1.0e-12):
        parser.error("--dx must divide the water depth")
    if np.any(np.round(counts).astype(int) % 2):
        parser.error("the two-level MGPCG grid needs even cell counts")
    return args


@ti.kernel
def set_moving_piston_velocity(
    velocity: float,
    wall_position: float,
    tank_width: float,
    tank_height: float,
    grid_size: ti.types.vector(3, float),
    solid_velocity_x: ti.template(),
):
    for I in ti.grouped(solid_velocity_x):
        x = I[0] * grid_size[0]
        y = (I[1] + 0.5) * grid_size[1]
        z = (I[2] + 0.5) * grid_size[2]
        if x <= wall_position + grid_size[0] and 0.0 < y < tank_width and 0.0 < z < tank_height:
            solid_velocity_x[I] = velocity


@ti.kernel
def enforce_moving_piston_particles(
    particle_num: int,
    wall_position: float,
    wall_velocity: float,
    dx: float,
    particle: ti.template(),
):
    clearance = 1.0e-4 * dx
    for p in range(particle_num):
        if int(particle[p].active) == 1 and particle[p].x[0] < wall_position + clearance:
            particle[p].x[0] = wall_position + clearance
            if particle[p].v[0] < wall_velocity:
                particle[p].v[0] = wall_velocity


def linear_piston_wave(frequency, depth, velocity_amplitude, gravity=9.81):
    omega = 2.0 * math.pi * frequency
    wavenumber = omega * omega / gravity
    for _ in range(20):
        kh = wavenumber * depth
        residual = gravity * wavenumber * math.tanh(kh) - omega * omega
        derivative = gravity * (math.tanh(kh) + kh / math.cosh(kh) ** 2)
        correction = residual / derivative
        wavenumber -= correction
        if abs(correction) <= 1.0e-14 * max(1.0, wavenumber):
            break
    kh = wavenumber * depth
    transfer = 4.0 * math.sinh(kh) ** 2 / (math.sinh(2.0 * kh) + 2.0 * kh)
    phase_speed = omega / wavenumber
    group_speed = 0.5 * phase_speed * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    return wavenumber, transfer * velocity_amplitude / omega, group_speed


def piston_velocity(time, frequency, velocity_amplitude, ramp_time):
    omega = 2.0 * math.pi * frequency
    if time >= ramp_time:
        return velocity_amplitude * math.cos(omega * time)
    phase = math.pi * time / ramp_time
    ramp = 0.5 * (1.0 - math.cos(phase))
    ramp_rate = 0.5 * math.pi / ramp_time * math.sin(phase)
    displacement_amplitude = velocity_amplitude / omega
    return displacement_amplitude * (ramp_rate * math.sin(omega * time) + ramp * omega * math.cos(omega * time))


def piston_displacement(time, frequency, velocity_amplitude, ramp_time):
    omega = 2.0 * math.pi * frequency
    ramp = 1.0 if time >= ramp_time else 0.5 * (1.0 - math.cos(math.pi * time / ramp_time))
    return ramp * velocity_amplitude / omega * math.sin(omega * time)


def harmonic_fit(time, values, frequency):
    omega_time = 2.0 * math.pi * frequency * time
    design = np.column_stack((np.sin(omega_time), np.cos(omega_time), np.ones_like(time)))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coefficients
    residual = float(np.sum((values - fitted) ** 2))
    total = float(np.sum((values - np.mean(values)) ** 2))
    return (
        float(np.hypot(coefficients[0], coefficients[1])),
        float(coefficients[2]),
        1.0 - residual / max(total, 1.0e-30),
    )


def piston_penetration(position, piston_position, dx):
    depth = max(0.0, float(piston_position - np.min(position[:, 0])))
    return int(np.count_nonzero(position[:, 0] < piston_position - 0.01 * dx)), depth


def active_grid_field(values, active_shape):
    values = np.squeeze(values)
    padding = np.asarray(values.shape) - np.asarray(active_shape)
    if np.any(padding < 0) or np.any(padding % 2):
        raise ValueError(f"grid shape {values.shape} is incompatible with active shape {tuple(active_shape)}")
    ghost = padding // 2
    return values[tuple(slice(int(width), int(width + count)) for width, count in zip(ghost, active_shape))]


def deep_air_particle_count(cell_type, particle_cell):
    occupied_type = cell_type[tuple(particle_cell.T)]
    air_cell = particle_cell[occupied_type == 0]
    if len(air_cell) == 0:
        return 0
    adjacent_fluid = np.zeros(len(air_cell), dtype=bool)
    upper = np.asarray(cell_type.shape) - 1
    for offset in np.ndindex(3, 3, 3):
        neighbor = np.minimum(np.maximum(air_cell + np.asarray(offset) - 1, 0), upper)
        adjacent_fluid |= cell_type[tuple(neighbor.T)] == 1
    return int(np.count_nonzero(~adjacent_fluid))


def sdf_surface_gauge(cell_type, fluid_sdf, dx, gauge_x, half_width=None):
    cell_type, fluid_sdf = np.squeeze(cell_type), np.squeeze(fluid_sdf)
    x_cells = np.flatnonzero(np.abs((np.arange(cell_type.shape[0]) + 0.5) * dx - gauge_x) <= (half_width or dx))
    surfaces = []
    for i in x_cells:
        for j in range(1, cell_type.shape[1] - 1):
            crossings = np.flatnonzero((cell_type[i, j, :-1] == 1) & (cell_type[i, j, 1:] == 0))
            if crossings.size:
                k = int(crossings[-1])
                phi_fluid, phi_air = float(fluid_sdf[i, j, k]), float(fluid_sdf[i, j, k + 1])
                theta = (
                    float(np.clip(phi_fluid / (phi_fluid - phi_air), 0.01, 1.0)) if phi_fluid < 0.0 < phi_air else 0.5
                )
                surfaces.append((k + 0.5 + theta) * dx)
    return float(np.median(surfaces)) if surfaces else float("nan")


def sdf_mean_surface(cell_type, fluid_sdf, dx):
    cell_type, fluid_sdf = np.squeeze(cell_type), np.squeeze(fluid_sdf)
    surfaces = []
    for i in range(1, cell_type.shape[0] - 1):
        for j in range(1, cell_type.shape[1] - 1):
            crossings = np.flatnonzero((cell_type[i, j, :-1] == 1) & (cell_type[i, j, 1:] == 0))
            if crossings.size:
                k = int(crossings[-1])
                phi_fluid, phi_air = float(fluid_sdf[i, j, k]), float(fluid_sdf[i, j, k + 1])
                theta = (
                    float(np.clip(phi_fluid / (phi_fluid - phi_air), 0.01, 1.0)) if phi_fluid < 0.0 < phi_air else 0.5
                )
                surfaces.append((k + 0.5 + theta) * dx)
    return float(np.mean(surfaces)) if surfaces else float("nan")


def sdf_surface_profile_metrics(cell_type, fluid_sdf, dx):
    cell_type, fluid_sdf = np.squeeze(cell_type), np.squeeze(fluid_sdf)
    surface = np.full(cell_type.shape[:2], np.nan)
    for i in range(1, cell_type.shape[0] - 1):
        for j in range(1, cell_type.shape[1] - 1):
            crossings = np.flatnonzero((cell_type[i, j, :-1] == 1) & (cell_type[i, j, 1:] == 0))
            if crossings.size:
                k = int(crossings[-1])
                phi_fluid, phi_air = float(fluid_sdf[i, j, k]), float(fluid_sdf[i, j, k + 1])
                theta = (
                    float(np.clip(phi_fluid / (phi_fluid - phi_air), 0.01, 1.0)) if phi_fluid < 0.0 < phi_air else 0.5
                )
                surface[i, j] = (k + 0.5 + theta) * dx
    x_margin = 2 if surface.shape[0] > 6 else 1
    y_margin = 2 if surface.shape[1] > 6 else 1
    interior = surface[x_margin:-x_margin, y_margin:-y_margin]
    if not np.isfinite(interior).any():
        return float("nan"), float("nan"), float("nan")
    interior = interior[np.any(np.isfinite(interior), axis=1)]
    centerline = np.nanmedian(interior, axis=1)
    cross_rms = float(np.sqrt(np.nanmean((interior - centerline[:, None]) ** 2)))
    centerline = centerline[np.isfinite(centerline)]
    longitudinal_rms = (
        float(np.sqrt(np.mean(np.diff(centerline, n=2) ** 2) / 6.0)) if len(centerline) >= 3 else float("nan")
    )
    return cross_rms, longitudinal_rms, float(np.ptp(centerline))


def write_metrics(output, expected_particles, requested_velocities, requested_positions, args):
    snapshots = sorted((output / "particles").glob("MPMParticle*.npz"))
    grids = {path.stem.removeprefix("MPMGrid"): path for path in (output / "grids").glob("MPMGrid[0-9]*.npz")}
    active_shape = np.round(TANK / args.dx).astype(int)
    rows = []
    outside = 0
    for path in snapshots:
        frame = path.stem.removeprefix("MPMParticle")
        if frame not in grids:
            raise RuntimeError(f"missing grid snapshot for particle frame {frame}")
        with np.load(path) as data, np.load(grids[frame]) as grid:
            active = data["active"] > 0
            position = data["position"][active]
            velocity = data["velocity"][active]
            pressure = data["pressure"][active]
            cell_type = active_grid_field(grid["cell_type"], active_shape)
            fluid_sdf = active_grid_field(grid["cell_fluid_sdf"], active_shape)
            surfaces = [sdf_surface_gauge(cell_type, fluid_sdf, args.dx, gauge_x) for gauge_x in GAUGE_X]
            outside = max(outside, int(np.count_nonzero(np.any((position < 0.0) | (position > TANK), axis=1))))
            particles_in_air = particles_in_deep_air = particles_in_solid = enclosed_air = 0
            if float(data["t_current"]) > 0.0:
                particle_cell = np.floor(position / args.dx).astype(np.int64)
                particle_cell = np.minimum(np.maximum(particle_cell, 0), active_shape - 1)
                occupied_type = cell_type[tuple(particle_cell.T)]
                particles_in_air = int(np.count_nonzero(occupied_type == 0))
                particles_in_deep_air = deep_air_particle_count(cell_type, particle_cell)
                particles_in_solid = int(np.count_nonzero(occupied_type == 2))
                enclosed_air = enclosed_air_count(cell_type)
            piston_position = PISTON_MEAN_X + piston_displacement(
                float(data["t_current"]), args.frequency, args.velocity_amplitude, args.ramp_time
            )
            piston_penetration_count, piston_penetration_depth = piston_penetration(position, piston_position, args.dx)
            surface_profile = sdf_surface_profile_metrics(cell_type, fluid_sdf, args.dx)
            rows.append(
                [
                    float(data["t_current"]),
                    int(active.sum()),
                    float(np.linalg.norm(velocity, axis=1).max()),
                    float(pressure.min()),
                    float(pressure.max()),
                    *surfaces,
                    sdf_mean_surface(cell_type, fluid_sdf, args.dx),
                    enclosed_air,
                    particles_in_air,
                    particles_in_deep_air,
                    particles_in_solid,
                    piston_penetration_count,
                    piston_penetration_depth,
                    *surface_profile,
                ]
            )
    rows = np.asarray(rows)
    surface_excursion = np.ptp(rows[:, 5:8], axis=0)
    wavenumber, expected_amplitude, group_speed = linear_piston_wave(
        args.frequency, WATER_DEPTH, args.velocity_amplitude
    )
    wavelength = 2.0 * math.pi / wavenumber
    period = 1.0 / args.frequency
    incident_start = args.ramp_time + 0.5 * period
    incident_end = min(args.time, args.ramp_time + (TANK[0] - PISTON_MEAN_X) / group_speed)
    incident = (rows[:, 0] >= incident_start) & (rows[:, 0] <= incident_end)
    incident_rows = rows[incident] if np.any(incident) else rows
    gauge_validation = []
    global_mean_surface_drift = float("nan")
    for gauge_index, (gauge, values) in enumerate(zip(GAUGE_X, rows[:, 5:8].T)):
        start = args.ramp_time + gauge / group_speed + 0.5 * period
        end = min(args.time, args.ramp_time + (2.0 * TANK[0] - gauge) / group_speed - 0.25 * period)
        selected = (rows[:, 0] >= start) & (rows[:, 0] <= end)
        if np.count_nonzero(selected) < 8 or end - start < 0.75 * period:
            continue
        amplitude, mean_surface, fit_r2 = harmonic_fit(rows[selected, 0], values[selected], args.frequency)
        if gauge_index == 0:
            _, global_mean_surface, _ = harmonic_fit(rows[selected, 0], rows[selected, 8], args.frequency)
            global_mean_surface_drift = global_mean_surface - float(rows[0, 8])
        gauge_validation.append(
            {
                "x_m": gauge,
                "analysis_window_s": [start, end],
                "harmonic_amplitude_m": amplitude,
                "amplitude_relative_error": abs(amplitude - expected_amplitude) / max(expected_amplitude, 1.0e-30),
                "mean_surface_m": mean_surface,
                "mean_surface_drift_m": mean_surface - float(rows[0, 5 + gauge_index]),
                "harmonic_fit_r2": fit_r2,
            }
        )
    metrics = {
        "case": "3-D incompressible MPM left-wall wavemaker",
        "reference": "Zhang et al. (2027), CMAME 463:119401, Section 6.2; scaled Lagrangian piston wave",
        "wavemaker_type": "moving rectangular piston cut-cell boundary",
        "tank_m": TANK.tolist(),
        "water_depth_m": WATER_DEPTH,
        "snapshots": len(rows),
        "final_time_s": float(rows[-1, 0]),
        "duration_complete": math.isclose(float(rows[-1, 0]), args.time, rel_tol=0.0, abs_tol=0.1 * args.dt),
        "expected_particles": expected_particles,
        "minimum_particles": int(rows[:, 1].min()),
        "maximum_fluid_speed_m_s": float(rows[:, 2].max()),
        "requested_wall_velocity_max_m_s": float(np.max(np.abs(requested_velocities))),
        "piston_mean_x_m": PISTON_MEAN_X,
        "piston_position_range_m": [float(np.min(requested_positions)), float(np.max(requested_positions))],
        "velocity_amplitude_m_s": args.velocity_amplitude,
        "alpha_pic": args.alpha_pic,
        "frequency_hz": args.frequency,
        "ramp_time_s": args.ramp_time,
        "surface_gauge_x_m": GAUGE_X,
        "initial_surface_m": rows[0, 5:9].tolist(),
        "surface_excursion_m": surface_excursion.tolist(),
        "linear_wavenumber_rad_m": wavenumber,
        "linear_wavelength_m": wavelength,
        "wavelengths_in_active_tank": (TANK[0] - PISTON_MEAN_X) / wavelength,
        "resolved_wave_amplitude_cells": expected_amplitude / args.dx,
        "linear_phase_speed_m_s": 2.0 * math.pi * args.frequency / wavenumber,
        "linear_wave_amplitude_m": expected_amplitude,
        "linear_group_speed_m_s": group_speed,
        "gauge_validation": gauge_validation,
        "amplitude_relative_tolerance": 0.55,
        "minimum_harmonic_fit_r2": 0.75,
        "minimum_measured_wave_amplitude_cells": (
            min(item["harmonic_amplitude_m"] for item in gauge_validation) / args.dx
            if gauge_validation
            else float("nan")
        ),
        "global_mean_surface_drift_m": global_mean_surface_drift,
        "surface_drift_tolerance_m": 0.50 * args.dx,
        "maximum_speed_tolerance_m_s": 1.25 * math.sqrt(2.0 * 9.81 * WATER_DEPTH),
        "maximum_particles_outside_tank": outside,
        "maximum_enclosed_air_cells": int(rows[:, 9].max()),
        "maximum_particles_in_air_cells": int(rows[:, 10].max()),
        "maximum_particles_in_deep_air_cells": int(rows[:, 11].max()),
        "maximum_particles_in_solid_cells": int(rows[:, 12].max()),
        "maximum_particles_behind_piston": int(rows[:, 13].max()),
        "maximum_piston_penetration_m": float(rows[:, 14].max()),
        "maximum_cross_tank_surface_roughness_m": float(rows[:, 15].max()),
        "maximum_high_frequency_surface_roughness_m": float(rows[:, 16].max()),
        "maximum_instantaneous_wave_height_m": float(rows[:, 17].max()),
        "incident_wave_window_s": [incident_start, incident_end],
        "maximum_incident_cross_tank_surface_roughness_m": float(incident_rows[:, 15].max()),
        "maximum_incident_high_frequency_surface_roughness_m": float(incident_rows[:, 16].max()),
        "surface_roughness_tolerance_m": 0.35 * args.dx,
        "piston_penetration_tolerance_m": 0.01 * args.dx,
        "finite": bool(np.isfinite(rows).all()),
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and len(rows) >= 2
        and metrics["minimum_particles"] == expected_particles
        and metrics["maximum_particles_outside_tank"] == 0
        and metrics["maximum_enclosed_air_cells"] <= 2
        and metrics["maximum_particles_in_deep_air_cells"] == 0
        and metrics["maximum_particles_in_solid_cells"] == 0
        and metrics["maximum_particles_behind_piston"] == 0
        and metrics["maximum_piston_penetration_m"] <= metrics["piston_penetration_tolerance_m"]
        and metrics["requested_wall_velocity_max_m_s"] >= 0.9 * args.velocity_amplitude
        and len(gauge_validation) >= 2
        and max(item["amplitude_relative_error"] for item in gauge_validation)
        <= metrics["amplitude_relative_tolerance"]
        and metrics["minimum_measured_wave_amplitude_cells"] >= 1.0
        and min(item["harmonic_fit_r2"] for item in gauge_validation) >= metrics["minimum_harmonic_fit_r2"]
        and abs(metrics["global_mean_surface_drift_m"]) <= metrics["surface_drift_tolerance_m"]
        and metrics["wavelengths_in_active_tank"] >= 1.25
        and metrics["resolved_wave_amplitude_cells"] >= 1.75
        and metrics["maximum_incident_cross_tank_surface_roughness_m"] <= metrics["surface_roughness_tolerance_m"]
        and metrics["maximum_incident_high_frequency_surface_roughness_m"] <= metrics["surface_roughness_tolerance_m"]
        and metrics["maximum_instantaneous_wave_height_m"] >= 1.25 * expected_amplitude
        and metrics["maximum_fluid_speed_m_s"] < metrics["maximum_speed_tolerance_m_s"]
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    np.savetxt(
        output / "wavemaker_diagnostics.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,particles,speed_max_m_s,pressure_min_pa,pressure_max_pa,"
            "surface_left_m,surface_mid_m,surface_right_m,global_mean_surface_m,"
            "enclosed_air_cells,particles_in_air_cells,particles_in_deep_air_cells,"
            "particles_in_solid_cells,particles_behind_piston,"
            "piston_penetration_m,cross_tank_surface_roughness_m,"
            "high_frequency_surface_roughness_m,instantaneous_wave_height_m"
        ),
        comments="",
    )
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("wavemaker validation failed; inspect metrics.json")


def run(args):
    output = args.output.expanduser().resolve()
    fluid_cells = np.round(WATER_SIZE / args.dx).astype(int)
    expected_particles = int(np.prod(fluid_cells) * args.ppc**3)
    print(
        f"# 3-D wavemaker tank: cells={np.round(TANK / args.dx).astype(int).tolist()}, particles={expected_particles}"
    )

    init(
        dim=3,
        arch=args.arch,
        default_fp="float64",
        device_memory_GB=args.device_memory,
        offline_cache=True,
        debug=False,
        kernel_profiler=False,
        log=False,
    )
    mpm = MPM()
    mpm.set_configuration(
        domain=TANK.tolist(),
        background_damping=0.0,
        alphaPIC=args.alpha_pic,
        mapping="USL",
        shape_function="QuadBSpline",
        gravity=[0.0, 0.0, -9.81],
        material_type="Fluid",
        velocity_projection="Affine",
        solver_type="Implicit",
        discretization="FDM",
        fluid_level_set=True,
        fluid_domain_volume_fraction=0.15,
        solid_sdf_cut_cell=True,
        fluid_wall_no_slip=False,
        particle_shifting=True,
        density_projection=True,
        density_projection_tolerance=0.008,
        density_projection_error_clamp=0.02,
        density_projection_max_shift_ratio=0.03,
        density_projection_interior_only=True,
        visualize=True,
    )
    mpm.set_implicit_solver_parameters(
        linear_solver="MGPCG",
        multilevel=2,
        pre_and_post_smoothing=2,
        bottom_smoothing=16,
        max_iteration_number=160,
        residual_tolerance=1.0e-8,
    )
    mpm.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.time,
            "SaveInterval": args.save_interval,
            "SavePath": str(output),
        }
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": expected_particles,
            "max_constraint_number": {},
        }
    )
    mpm.add_material(
        model="Newtonian",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "Modulus": 2.0e6,
            "Viscosity": 1.0e-3,
            "ElementLength": args.dx,
            "cL": 1.5,
            "cQ": 2.0,
            "atmospheric_pressure": 0.0,
        },
    )
    mpm.add_element({"ElementType": "Staggered", "ElementSize": [args.dx] * 3, "GhostCell": 1})
    mpm.add_region(
        {
            "Name": "water",
            "Type": "Rectangle",
            "BoundingBoxPoint": WATER_ORIGIN.tolist(),
            "BoundingBoxSize": WATER_SIZE.tolist(),
        }
    )
    mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "water",
                    "nParticlesPerCell": args.ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                }
            ]
        }
    )
    walls = [
        ([1.0, 0.0, 0.0], [TANK[0], 0.0, 0.0], TANK.tolist()),
        ([0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [TANK[0], 0.0, TANK[2]]),
        ([0.0, 1.0, 0.0], [0.0, TANK[1], 0.0], TANK.tolist()),
        ([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [TANK[0], TANK[1], 0.0]),
    ]
    mpm.add_boundary_condition(
        [
            {"BoundaryType": "SolidCell", "Norm": normal, "StartPoint": start, "EndPoint": end, "CellThickness": 1}
            for normal, start, end in walls
        ]
    )

    requested_velocities = []
    requested_positions = []

    def update_wavemaker(sims, scene):
        start_time = float(sims.current_time)
        end_time = min(start_time + float(sims.delta), args.time)
        velocity = piston_velocity(
            0.5 * (start_time + end_time), args.frequency, args.velocity_amplitude, args.ramp_time
        )
        position = PISTON_MEAN_X + piston_displacement(
            end_time, args.frequency, args.velocity_amplitude, args.ramp_time
        )
        requested_velocities.append(velocity)
        requested_positions.append(position)
        mpm.enginer.ensure_solid_cut_cell_fields(sims, scene)
        mpm.enginer.apply_solid_cell_boundaries(sims, scene)
        kernel_update_solid_sdf_from_box(
            scene.element.grid_size,
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([position, TANK[1], TANK[2]]),
            scene.element.cell.solid_sdf,
        )
        set_moving_piston_velocity(
            velocity,
            position,
            TANK[1],
            TANK[2],
            scene.element.grid_size,
            mpm.enginer.solid_face_velocity[0],
        )

    @python_callback
    def enforce_wavemaker_particles():
        start_time = float(mpm.sims.current_time)
        end_time = min(start_time + float(mpm.sims.delta), args.time)
        velocity = piston_velocity(
            0.5 * (start_time + end_time), args.frequency, args.velocity_amplitude, args.ramp_time
        )
        position = PISTON_MEAN_X + piston_displacement(
            end_time, args.frequency, args.velocity_amplitude, args.ramp_time
        )
        enforce_moving_piston_particles(int(mpm.scene.particleNum[0]), position, velocity, args.dx, mpm.scene.particle)

    mpm.select_save_data(particle=True, grid=True)
    mpm.run(gravity_field=True, cut_cell_function=update_wavemaker, function=enforce_wavemaker_particles)
    if not args.no_post:
        mpm.postprocessing()
    write_metrics(output, expected_particles, requested_velocities, requested_positions, args)


if __name__ == "__main__":
    run(parse_args())
