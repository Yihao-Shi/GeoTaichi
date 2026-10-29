"""Validate saved 3-D incompressible MPM frames."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.postprocessing.mpm.fluid_metrics import (
    deep_air_particle_mask,
    nonzero_spray_pressure_count,
    saved_grid_spacing,
    enclosed_air_count,
)

DOMAIN = np.array([6.0, 1.42, 6.0])


def validate(
    output,
    expected_final_time,
    domain=DOMAIN,
    minimum_snapshots=3,
    case=None,
    metrics_name="validation_metrics.json",
    allow_enclosed_air=False,
    maximum_particles_in_air_cells=0,
    allow_detached_spray=False,
    atmospheric_pressure=0.0,
):
    if not math.isfinite(atmospheric_pressure):
        raise ValueError("atmospheric pressure must be finite")
    domain = np.asarray(domain, dtype=float)
    particle_files = sorted((output / "particles").glob("MPMParticle*.npz"))
    grid_files = {path.stem.removeprefix("MPMGrid"): path for path in (output / "grids").glob("MPMGrid*.npz")}
    if not particle_files:
        raise FileNotFoundError(f"no particle frames under {output}")

    snapshots = []
    expected_particles = None
    initial_mass = initial_volume = None
    spacing = None
    for particle_path in particle_files:
        frame = particle_path.stem.removeprefix("MPMParticle")
        grid_path = grid_files.get(frame)
        if grid_path is None:
            raise FileNotFoundError(f"missing grid frame {frame}")
        with np.load(particle_path) as particles, np.load(grid_path) as grid:
            active = particles["active"] > 0
            position = particles["position"][active]
            velocity = particles["velocity"][active]
            pressure = particles["pressure"][active]
            mass = particles["mass"][active]
            volume = particles["volume"][active]
            total_mass, total_volume = float(np.sum(mass, dtype=np.float64)), float(np.sum(volume, dtype=np.float64))
            if initial_mass is None:
                initial_mass, initial_volume = total_mass, total_volume
            active_particles = int(active.sum())
            if expected_particles is None:
                expected_particles = active_particles

            cell_type = np.squeeze(grid["cell_type"])
            cell_pressure = np.squeeze(grid["cell_pressure"])
            ghost = 1
            active_type = cell_type[ghost:-ghost, ghost:-ghost, ghost:-ghost]
            if spacing is None:
                spacing = saved_grid_spacing(grid)
                coverage = spacing * np.asarray(active_type.shape)
                tolerance = max(
                    1.0e-10 * float(np.min(spacing)),
                    np.finfo(np.float32).eps * float(np.max(np.abs(domain))),
                )
                if np.any(coverage + tolerance < domain) or np.any(coverage - domain >= spacing + tolerance):
                    raise ValueError(f"saved grid coverage {coverage} is incompatible with domain {domain}")
            position_tolerance = 1.0e-10 * float(np.min(spacing))
            inside = np.all((position >= -position_tolerance) & (position <= domain + position_tolerance), axis=1)
            cell = np.floor(np.minimum(np.maximum(position, 0.0), np.nextafter(domain, 0.0)) / spacing).astype(np.int64)
            cell = np.minimum(cell, np.asarray(active_type.shape) - 1)
            occupied_type = cell_type[tuple((cell + ghost).T)]
            spray = deep_air_particle_mask(active_type, cell)
            unsupported_spray_pressure = nonzero_spray_pressure_count(pressure, spray, atmospheric_pressure)
            finite_particles = bool(
                np.isfinite(position).all() and np.isfinite(velocity).all() and np.isfinite(pressure).all()
            )
            finite_pressure = bool(
                np.isfinite(cell_pressure[ghost:-ghost, ghost:-ghost, ghost:-ghost][active_type == 1]).all()
            )
            active_sdf = None
            if "cell_fluid_sdf" in grid:
                active_sdf = np.squeeze(grid["cell_fluid_sdf"])[ghost:-ghost, ghost:-ghost, ghost:-ghost]
            snapshot = {
                "time_s": float(particles["t_current"]),
                "active_particles": active_particles,
                "fluid_cells": int(np.count_nonzero(active_type == 1)),
                # Side and bottom walls are closed; a wall-touching pocket is
                # exterior only if it also connects to the upper headspace.
                "strict_enclosed_air_cells": enclosed_air_count(active_type, exterior_faces=[(2, -1)]),
                "sdf_enclosed_air_cells": enclosed_air_count(active_type, active_sdf, exterior_faces=[(2, -1)]),
                "particles_in_air_cells": int(np.count_nonzero(occupied_type == 0)),
                "detached_spray_particles": int(np.count_nonzero(spray)),
                "non_atmospheric_detached_spray_pressure_particles": unsupported_spray_pressure,
                "particles_in_solid_cells": int(np.count_nonzero(occupied_type == 2)),
                "particles_outside_domain": int(np.count_nonzero(~inside)),
                "finite_particle_state": finite_particles,
                "finite_fluid_pressure": finite_pressure,
                "finite_positive_particle_mass_and_volume": bool(
                    np.isfinite(mass).all() and np.isfinite(volume).all() and np.all(mass > 0) and np.all(volume > 0)
                ),
                "relative_mass_drift": abs(total_mass - initial_mass) / max(abs(initial_mass), 1e-30),
                "relative_volume_drift": abs(total_volume - initial_volume) / max(abs(initial_volume), 1e-30),
            }
            snapshot["passed"] = bool(
                active_particles == expected_particles
                and active_particles > 0
                and snapshot["fluid_cells"] > 0
                and (allow_enclosed_air or snapshot["sdf_enclosed_air_cells"] == 0)
                and (
                    unsupported_spray_pressure == 0
                    if allow_detached_spray
                    else snapshot["particles_in_air_cells"] <= maximum_particles_in_air_cells
                )
                and snapshot["particles_in_solid_cells"] == 0
                and snapshot["particles_outside_domain"] == 0
                and finite_particles
                and finite_pressure
                and snapshot["finite_positive_particle_mass_and_volume"]
                and snapshot["relative_mass_drift"] <= 1e-12
                and snapshot["relative_volume_drift"] <= 1e-12
            )
            snapshots.append(snapshot)

    metrics = {
        "case": case or "large_tank_incompressible_3d classification validation",
        "domain_m": domain.tolist(),
        "active_cell_counts": list(active_type.shape),
        "expected_particles": expected_particles,
        "final_time_s": snapshots[-1]["time_s"],
        "grid_spacing_m": spacing.tolist(),
        "grid_coverage_m": (spacing * np.asarray(active_type.shape)).tolist(),
        "enclosed_air_allowed": allow_enclosed_air,
        "ambient_boundary_faces": [[2, -1]],
        "maximum_particles_in_air_cells_allowed": None if allow_detached_spray else maximum_particles_in_air_cells,
        "particle_air_policy": (
            "attached_interface_and_pressure_free_detached_spray" if allow_detached_spray else "strict_cell_membership"
        ),
        "atmospheric_pressure_pa": atmospheric_pressure,
        "mass_and_volume_relative_tolerance": 1e-12,
        "snapshots": snapshots,
    }
    metrics["passed"] = bool(
        len(snapshots) >= minimum_snapshots
        and math.isclose(metrics["final_time_s"], expected_final_time, rel_tol=0.0, abs_tol=1.0e-10)
        and all(snapshot["passed"] for snapshot in snapshots)
    )
    (output / metrics_name).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-final-time", type=float, default=0.2)
    parser.add_argument("--domain", type=float, nargs=3, default=DOMAIN.tolist())
    parser.add_argument("--minimum-snapshots", type=int, default=3)
    parser.add_argument("--case")
    parser.add_argument("--metrics-name", default="validation_metrics.json")
    parser.add_argument("--allow-enclosed-air", action="store_true")
    parser.add_argument("--maximum-particles-in-air-cells", type=int, default=0)
    parser.add_argument("--allow-detached-spray", action="store_true")
    parser.add_argument("--atmospheric-pressure", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.maximum_particles_in_air_cells < 0:
        parser.error("--maximum-particles-in-air-cells must be nonnegative")
    metrics = validate(
        args.output,
        args.expected_final_time,
        args.domain,
        args.minimum_snapshots,
        args.case,
        args.metrics_name,
        args.allow_enclosed_air,
        args.maximum_particles_in_air_cells,
        args.allow_detached_spray,
        args.atmospheric_pressure,
    )
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError(f"incompressible output validation failed; inspect {args.metrics_name}")


if __name__ == "__main__":
    main()
