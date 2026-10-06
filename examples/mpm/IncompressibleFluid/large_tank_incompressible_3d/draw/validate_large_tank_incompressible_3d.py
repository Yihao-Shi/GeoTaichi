"""Validate saved 3-D incompressible MPM frames."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


DOMAIN = np.array([6.0, 1.42, 6.0])


def saved_grid_spacing(grid):
    dims = tuple(np.asarray(grid["dims"], dtype=np.int64))
    coords = np.asarray(grid["coords"])
    if len(dims) != coords.shape[1] or math.prod(dims) != len(coords) or min(dims) < 2:
        raise ValueError(f"invalid saved grid coordinates: dims={dims}, coords={coords.shape}")
    lattice = coords.reshape(*dims, len(dims))
    origin = lattice[(0,) * len(dims)]
    spacing = []
    for axis in range(len(dims)):
        neighbor = [0] * len(dims)
        neighbor[axis] = 1
        spacing.append(abs(float(lattice[tuple(neighbor)][axis] - origin[axis])))
    if min(spacing) <= 0.0:
        raise ValueError(f"invalid saved grid spacing: {spacing}")
    return np.asarray(spacing)


def enclosed_air_count(cell_type):
    air = cell_type == 0
    exterior = np.zeros_like(air)
    for axis in range(air.ndim):
        for index in (0, -1):
            face = [slice(None)] * air.ndim
            face[axis] = index
            exterior[tuple(face)] = air[tuple(face)]

    while True:
        connected = exterior.copy()
        for axis in range(air.ndim):
            lower = [slice(None)] * air.ndim
            upper = [slice(None)] * air.ndim
            lower[axis] = slice(1, None)
            upper[axis] = slice(None, -1)
            connected[tuple(lower)] |= exterior[tuple(upper)]
            connected[tuple(upper)] |= exterior[tuple(lower)]
        connected &= air
        if np.array_equal(connected, exterior):
            return int(np.count_nonzero(air & ~exterior))
        exterior = connected


def validate(
    output,
    expected_final_time,
    domain=DOMAIN,
    minimum_snapshots=3,
    case=None,
    metrics_name="validation_metrics.json",
    allow_enclosed_air=False,
    maximum_particles_in_air_cells=0,
):
    domain = np.asarray(domain, dtype=float)
    particle_files = sorted((output / "particles").glob("MPMParticle*.npz"))
    grid_files = {path.stem.removeprefix("MPMGrid"): path for path in (output / "grids").glob("MPMGrid*.npz")}
    if not particle_files:
        raise FileNotFoundError(f"no particle frames under {output}")

    snapshots = []
    expected_particles = None
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
            finite_particles = bool(
                np.isfinite(position).all() and np.isfinite(velocity).all() and np.isfinite(pressure).all()
            )
            finite_pressure = bool(
                np.isfinite(cell_pressure[ghost:-ghost, ghost:-ghost, ghost:-ghost][active_type == 1]).all()
            )
            snapshot = {
                "time_s": float(particles["t_current"]),
                "active_particles": active_particles,
                "fluid_cells": int(np.count_nonzero(active_type == 1)),
                "strict_enclosed_air_cells": enclosed_air_count(active_type),
                "particles_in_air_cells": int(np.count_nonzero(occupied_type == 0)),
                "particles_in_solid_cells": int(np.count_nonzero(occupied_type == 2)),
                "particles_outside_domain": int(np.count_nonzero(~inside)),
                "finite_particle_state": finite_particles,
                "finite_fluid_pressure": finite_pressure,
            }
            snapshot["passed"] = bool(
                active_particles == expected_particles
                and (allow_enclosed_air or snapshot["strict_enclosed_air_cells"] == 0)
                and snapshot["particles_in_air_cells"] <= maximum_particles_in_air_cells
                and snapshot["particles_in_solid_cells"] == 0
                and snapshot["particles_outside_domain"] == 0
                and finite_particles
                and finite_pressure
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
        "maximum_particles_in_air_cells_allowed": maximum_particles_in_air_cells,
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
    )
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError(f"incompressible output validation failed; inspect {args.metrics_name}")


if __name__ == "__main__":
    main()
