"""Evaluation and postprocessing for double_point_dam_break_porous_elastic."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import json
import math
from pathlib import Path
import numpy as np

from examples.mmpm.DamBreakPorousElastic2D.double_point_dam_break_porous_elastic_parameters import (
    ALPHA_PIC,
    BACKGROUND_DAMPING,
    BASE_WATER_SIZE,
    DOMAIN,
    DOMAIN_CELLS,
    DT,
    DX,
    EXPERIMENTAL_DOMAIN,
    FLUID_PPC,
    GRAIN_DIAMETER,
    MAX_PARTICLES,
    PARTICLE_SHIFTING,
    PARTICLE_SHIFTING_END_TIME,
    PARTICLE_SHIFTING_SETTLING_SCALE,
    PERMEABILITY,
    POROSITY,
    POROUS_ORIGIN,
    POROUS_RIGHT,
    POROUS_SIZE,
    SOLID_PPC,
    UPPER_WATER_SIZE,
    VELOCITY_PROJECTION,
)


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
    left_surface = float(np.nanmedian(surface[surface_bins[:-1] < POROUS_ORIGIN[0]]))
    right_surface = float(np.nanmedian(surface[surface_bins[:-1] >= POROUS_RIGHT]))
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
        "velocity_projection": VELOCITY_PROJECTION,
        "background_damping": BACKGROUND_DAMPING,
        "background_damping_start_time_s": PARTICLE_SHIFTING_END_TIME if PARTICLE_SHIFTING else 0.0,
        "particle_shifting": PARTICLE_SHIFTING,
        "particle_shifting_end_time_s": PARTICLE_SHIFTING_END_TIME if PARTICLE_SHIFTING else 0.0,
        "particle_shifting_settling_scale": PARTICLE_SHIFTING_SETTLING_SCALE if PARTICLE_SHIFTING else 0.0,
        "solid_model": "rigid (both velocity components constrained over the full porous column)",
        "maximum_fluid_points_inside_porous": int(np.max(rows[:, 2])),
        "maximum_fluid_points_through_porous": int(np.max(rows[:, 3])),
        "maximum_upstream_fluid_points_transmitted": int(np.max(rows[:, 4])),
        "final_upstream_fluid_points_transmitted": int(rows[-1, 4]),
        "maximum_solid_displacement_m": float(np.max(rows[:, 8])),
        "maximum_fluid_speed_p99_mps": float(np.max(rows[:, 9])),
        "maximum_fluid_speed_mps": float(np.max(rows[:, 10])),
        "final_fluid_speed_p99_mps": float(rows[-1, 9]),
        "final_fluid_speed_mps": float(rows[-1, 10]),
        "last_second_max_fluid_speed_p99_mps": float(np.max(rows[rows[:, 0] >= rows[-1, 0] - 1.0, 9])),
        "final_mean_fluid_vx_inside_porous_mps": float(rows[-1, 5]),
        "last_second_max_abs_mean_fluid_vx_inside_porous_mps": float(
            np.max(np.abs(rows[rows[:, 0] >= rows[-1, 0] - 1.0, 5]))
        ),
        "minimum_solid_porosity": float(np.min(rows[:, 11])),
        "maximum_solid_porosity": float(np.max(rows[:, 12])),
        "minimum_solid_volume_ratio": float(np.min(rows[:, 13])),
        "maximum_solid_volume_ratio": float(np.max(rows[:, 14])),
        "maximum_solid_speed_mps": float(np.max(rows[:, 15])),
        "left_wall_particle_count": int(left_wall_ids.size),
        "left_wall_vertically_mobile_count": int(np.count_nonzero(np.abs(left_wall_vertical_displacement) > 1.0e-5)),
        "final_surface_total_variation_m": surface_total_variation,
        "final_left_surface_height_m": left_surface,
        "final_right_surface_height_m": right_surface,
        "final_surface_head_difference_m": abs(left_surface - right_surface),
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
        and metrics["last_second_max_fluid_speed_p99_mps"] <= 3.0e-2
        and metrics["last_second_max_abs_mean_fluid_vx_inside_porous_mps"] <= 5.0e-3
        and metrics["final_surface_head_difference_m"] <= 1.0e-2
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
