"""Evaluation and postprocessing for taylor_green_vortex_2d."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import math
import numpy as np

from examples.mpm.IncompressibleFluid.taylor_green_vortex_2d.taylor_green_vortex_2d_parameters import (
    DOMAIN,
    STRICT_TOLERANCES,
)


def analytical_velocity(position, time, viscosity, amplitude=1.0):
    decay = amplitude * math.exp(-2.0 * viscosity * time)
    x, y = position.T
    return decay * np.column_stack((np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)))


def analytical_pressure(position, time, viscosity, density=1.0, amplitude=1.0):
    decay = math.exp(-4.0 * viscosity * time)
    x, y = position.T
    return 0.25 * density * amplitude**2 * decay * (np.cos(2.0 * x) + np.cos(2.0 * y))


def write_metrics(output, diagnostics, expected_particles, args):
    rows = np.asarray(diagnostics)
    verification_index = int(np.argmin(np.abs(rows[:, 0] - min(args.verification_time, args.time))))
    exact_final_fraction = math.exp(-2.0 * args.viscosity * float(rows[-1, 0]))
    numerical_final_fraction = math.sqrt(max(float(rows[-1, 4]), 0.0) / (0.25 * args.amplitude**2))
    metrics = {
        "case": "2-D incompressible MPM Taylor-Green vortex",
        "domain_m": DOMAIN.tolist(),
        "cells": args.cells,
        "ppc": args.ppc,
        "expected_particles": expected_particles,
        "minimum_particles": int(rows[:, 1].min()),
        "dt_s": args.dt,
        "final_time_s": float(rows[-1, 0]),
        "viscosity_m2_s": args.viscosity,
        "reynolds_number": args.amplitude / args.viscosity,
        "samples": len(rows),
        "verification_time_s": float(rows[verification_index, 0]),
        "verification_velocity_relative_l2": float(rows[verification_index, 2]),
        "verification_velocity_relative_linf": float(rows[verification_index, 3]),
        "verification_kinetic_energy_relative_error": float(rows[verification_index, 6]),
        "verification_pressure_relative_l2": float(rows[verification_index, 7]),
        "final_velocity_relative_l2": float(rows[-1, 2]),
        "maximum_velocity_relative_l2": float(rows[:, 2].max()),
        "final_velocity_relative_linf": float(rows[-1, 3]),
        "final_kinetic_energy_relative_error": float(rows[-1, 6]),
        "final_pressure_relative_l2": float(rows[-1, 7]),
        "maximum_interior_air_cells": int(rows[:, 8].max()),
        "steady_fraction_tolerance": args.steady_fraction,
        "analytical_final_velocity_fraction": exact_final_fraction,
        "numerical_final_velocity_fraction": numerical_final_fraction,
        "reached_practical_steady_state": bool(
            exact_final_fraction <= args.steady_fraction * (1.0 + 1.0e-10)
            and numerical_final_fraction <= 2.0 * args.steady_fraction
        ),
        "finite": bool(np.isfinite(rows).all()),
        "duration_complete": math.isclose(float(rows[-1, 0]), args.time, rel_tol=0.0, abs_tol=0.1 * args.dt),
        "strict_tolerances": STRICT_TOLERANCES,
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and metrics["minimum_particles"] == expected_particles
        and metrics["maximum_interior_air_cells"] == 0
        and metrics["reached_practical_steady_state"]
        and metrics["verification_velocity_relative_l2"] <= STRICT_TOLERANCES["velocity_relative_l2"]
        and metrics["verification_velocity_relative_linf"] <= STRICT_TOLERANCES["velocity_relative_linf"]
        and metrics["verification_kinetic_energy_relative_error"] <= STRICT_TOLERANCES["kinetic_energy_relative_error"]
        and metrics["verification_pressure_relative_l2"] <= STRICT_TOLERANCES["pressure_relative_l2"]
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    np.savetxt(
        output / "taylor_green_comparison.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,particles,velocity_relative_l2,velocity_relative_linf,kinetic_energy,"
            "exact_kinetic_energy,kinetic_energy_relative_error,pressure_relative_l2,interior_air_cells"
        ),
        comments="",
    )
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("Taylor-Green verification failed; inspect metrics.json")


def comparison_row(mpm, args, time, dx):
    particle_count = int(mpm.scene.particleNum[0])
    active = mpm.scene.particle.active.to_numpy()[:particle_count] > 0
    position = mpm.scene.particle.x.to_numpy()[:particle_count][active]
    velocity = mpm.scene.particle.v.to_numpy()[:particle_count][active]
    exact_velocity = analytical_velocity(position, time, args.viscosity, args.amplitude)
    difference = velocity - exact_velocity
    exact_norm = np.linalg.norm(exact_velocity)
    velocity_l2 = float(np.linalg.norm(difference) / max(exact_norm, np.finfo(float).eps))
    velocity_linf = float(
        np.linalg.norm(difference, axis=1).max() / (args.amplitude * math.exp(-2.0 * args.viscosity * time))
    )
    kinetic_energy = float(0.5 * np.mean(np.sum(velocity * velocity, axis=1)))
    exact_energy = 0.25 * args.amplitude**2 * math.exp(-4.0 * args.viscosity * time)

    ghost = int(mpm.scene.element.ghost_cell)
    active_slice = (slice(ghost, ghost + args.cells),) * 2
    cell_type = np.squeeze(mpm.scene.element.cell.type.to_numpy())[active_slice]
    numerical_pressure = np.squeeze(mpm.scene.element.cell.pressure.to_numpy())[active_slice]
    coordinates = (np.arange(args.cells) + 0.5) * dx
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    exact_pressure = analytical_pressure(centers, time, args.viscosity, density=1.0, amplitude=args.amplitude).reshape(
        args.cells, args.cells
    )
    fluid = cell_type == 1
    numerical = numerical_pressure[fluid] - np.mean(numerical_pressure[fluid])
    exact = exact_pressure[fluid] - np.mean(exact_pressure[fluid])
    pressure_l2 = float(np.linalg.norm(numerical - exact) / max(np.linalg.norm(exact), np.finfo(float).eps))
    return [
        time,
        int(active.sum()),
        velocity_l2,
        velocity_linf,
        kinetic_energy,
        exact_energy,
        abs(kinetic_energy - exact_energy) / exact_energy,
        pressure_l2,
        int(np.count_nonzero(cell_type == 0)),
    ]


def decay_time_to_fraction(viscosity, fraction):
    return -math.log(fraction) / (2.0 * viscosity)
