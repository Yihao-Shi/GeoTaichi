"""Validate the settled 2-D square-SDF dam-break result."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mpm.IncompressibleFluid.validate_large_tank_incompressible_3d import enclosed_air_count


DOMAIN = np.array([0.90, 0.45])
OBSTACLE_LOWER = np.array([0.35, 0.06])
OBSTACLE_UPPER = OBSTACLE_LOWER + np.array([0.11, 0.11])


def box_penetration(position):
    center = 0.5 * (OBSTACLE_LOWER + OBSTACLE_UPPER)
    half_size = 0.5 * (OBSTACLE_UPPER - OBSTACLE_LOWER)
    offset = np.abs(position - center) - half_size
    signed_distance = np.linalg.norm(np.maximum(offset, 0.0), axis=1) + np.minimum(np.max(offset, axis=1), 0.0)
    return np.maximum(-signed_distance, 0.0)


def validate(output, expected_final_time, stability_window, particle_spacing):
    particle_files = sorted((output / "particles").glob("MPMParticle*.npz"))
    grid_files = {path.stem.removeprefix("MPMGrid"): path for path in (output / "grids").glob("MPMGrid*.npz")}
    if len(particle_files) < 3:
        raise RuntimeError(f"fewer than three particle snapshots under {output}")

    rows = []
    expected_particles = None
    finite = True
    for particle_path in particle_files:
        frame = particle_path.stem.removeprefix("MPMParticle")
        grid_path = grid_files.get(frame)
        if grid_path is None:
            raise RuntimeError(f"missing grid snapshot {frame}")
        with np.load(particle_path) as particles, np.load(grid_path) as grid:
            active = particles["active"] > 0
            position = particles["position"][active]
            velocity = particles["velocity"][active]
            pressure = particles["pressure"][active]
            mass = particles["mass"][active]
            speed2 = np.sum(velocity * velocity, axis=1)
            particle_count = int(active.sum())
            expected_particles = particle_count if expected_particles is None else expected_particles
            cell_type = np.squeeze(grid["cell_type"])
            cell_pressure = np.squeeze(grid["cell_pressure"])
            finite &= bool(
                np.isfinite(position).all()
                and np.isfinite(velocity).all()
                and np.isfinite(pressure).all()
                and np.isfinite(cell_pressure[cell_type == 1]).all()
            )
            outside = np.any((position < 0.0) | (position > DOMAIN), axis=1)
            rows.append(
                [
                    float(particles["t_current"]),
                    particle_count,
                    math.sqrt(float(np.mean(speed2))),
                    0.5 * float(np.sum(mass * speed2)),
                    float(np.sqrt(speed2).max()),
                    int(np.count_nonzero(outside)),
                    float(box_penetration(position).max()),
                    enclosed_air_count(cell_type[1:-1, 1:-1]),
                    float(position[:, 0].max()),
                ]
            )

    rows = np.asarray(rows, dtype=float)
    tail = rows[:, 0] >= rows[-1, 0] - stability_window - 1.0e-12
    peak_energy = float(np.max(rows[:, 3]))
    enclosed = rows[:, 7].astype(int)
    enclosed_frames = rows[enclosed > 0, 0]
    metrics = {
        "case": "2-D incompressible dam break around a fixed square SDF obstacle",
        "simulation_time_s": float(rows[-1, 0]),
        "snapshots": len(rows),
        "initial_particles": int(rows[0, 1]),
        "final_particles": int(rows[-1, 1]),
        "particle_conservation": bool(np.all(rows[:, 1] == expected_particles)),
        "finite": finite and bool(np.isfinite(rows).all()),
        "maximum_particles_outside_domain": int(np.max(rows[:, 5])),
        "maximum_particle_center_penetration_m": float(np.max(rows[:, 6])),
        "particle_boundary_band_tolerance_m": particle_spacing,
        "maximum_enclosed_air_cells": int(np.max(enclosed)),
        "maximum_tail_enclosed_air_cells": int(np.max(enclosed[tail])),
        "final_enclosed_air_cells": int(enclosed[-1]),
        "last_enclosed_air_time_s": float(enclosed_frames[-1]) if len(enclosed_frames) else None,
        "persistent_internal_cavity_detected": bool(np.any(enclosed[tail] > 0)),
        "final_front_x_m": float(rows[-1, 8]),
        "peak_rms_speed_mps": float(np.max(rows[:, 2])),
        "final_rms_speed_mps": float(rows[-1, 2]),
        "tail_mean_rms_speed_mps": float(np.mean(rows[tail, 2])),
        "peak_kinetic_energy_j": peak_energy,
        "final_kinetic_energy_j": float(rows[-1, 3]),
        "tail_mean_kinetic_energy_over_peak": float(np.mean(rows[tail, 3]) / peak_energy),
        "stability_window_s": stability_window,
        "strict_tolerances": {
            "maximum_tail_enclosed_air_cells": 0,
            "maximum_final_rms_speed_mps": 0.20,
            "maximum_tail_mean_rms_speed_mps": 0.20,
            "maximum_tail_mean_kinetic_energy_over_peak": 0.04,
        },
    }
    tolerance = metrics["strict_tolerances"]
    metrics["passed"] = bool(
        metrics["finite"]
        and math.isclose(metrics["simulation_time_s"], expected_final_time, abs_tol=1.0e-10)
        and metrics["particle_conservation"]
        and metrics["maximum_particles_outside_domain"] == 0
        and metrics["maximum_particle_center_penetration_m"] <= particle_spacing
        and metrics["maximum_tail_enclosed_air_cells"] <= tolerance["maximum_tail_enclosed_air_cells"]
        and metrics["final_enclosed_air_cells"] == 0
        and metrics["final_front_x_m"] >= 0.8 * DOMAIN[0]
        and metrics["final_rms_speed_mps"] <= tolerance["maximum_final_rms_speed_mps"]
        and metrics["tail_mean_rms_speed_mps"] <= tolerance["maximum_tail_mean_rms_speed_mps"]
        and metrics["tail_mean_kinetic_energy_over_peak"] <= tolerance["maximum_tail_mean_kinetic_energy_over_peak"]
    )
    np.savetxt(
        output / "validation_history.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,particles,rms_speed_mps,kinetic_energy_j,max_speed_mps,"
            "particles_outside_domain,max_obstacle_penetration_m,enclosed_air_cells,front_x_m"
        ),
        comments="",
    )
    (output / "validation_summary.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-final-time", type=float, default=4.0)
    parser.add_argument("--stability-window", type=float, default=1.0)
    parser.add_argument("--particle-spacing", type=float, default=0.01 / 3.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    metrics = validate(args.output, args.expected_final_time, args.stability_window, args.particle_spacing)
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("square-SDF dam-break validation failed; inspect validation_summary.json")


if __name__ == "__main__":
    main()
