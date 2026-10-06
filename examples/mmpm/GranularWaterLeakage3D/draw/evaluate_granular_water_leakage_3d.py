"""Evaluation and postprocessing for granular_water_leakage_3d."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import math
import numpy as np

from examples.mmpm.GranularWaterLeakage3D.granular_water_leakage_3d_parameters import (
    CRACK_LEFT,
    CRACK_RIGHT,
    DOMAIN,
    RECEIVER_ORIGIN,
    RECEIVER_SIZE,
    UPPER_ORIGIN,
)


def write_metrics(output, expected_fluid, expected_solid, args):
    files = sorted((output / "particles").glob("MPMParticle*.npz"))
    if len(files) < 2:
        raise RuntimeError("Section 5.1 run produced fewer than two particle snapshots")
    rows = []
    finite = True
    receiver_min = RECEIVER_ORIGIN
    receiver_max = RECEIVER_ORIGIN + RECEIVER_SIZE
    for file_name in files:
        with np.load(file_name) as data:
            active = data["active"] > 0
            phase = data["phase"]
            fluid = active & (phase == 2)
            solid = active & (phase == 1)
            position = data["position"]
            fluid_velocity = data["fluid_velocity"]
            solid_velocity = data["solid_velocity"]
            pressure = data["pressure"]
            porosity = data["porosity"]
            finite &= bool(
                np.isfinite(position[active]).all()
                and np.isfinite(fluid_velocity[fluid]).all()
                and np.isfinite(solid_velocity[solid]).all()
                and np.isfinite(pressure[active]).all()
            )
            outside = np.any(
                (position[active] < -1.0e-10 * args.dx) | (position[active] > np.asarray(DOMAIN) + 1.0e-10 * args.dx),
                axis=1,
            )
            fluid_position = position[fluid]
            solid_position = position[solid]
            fluid_captured = np.all(
                (fluid_position >= receiver_min - 0.5 * args.dx) & (fluid_position <= receiver_max + 0.5 * args.dx),
                axis=1,
            )
            rows.append(
                [
                    float(data["t_current"]),
                    int(np.count_nonzero(fluid)),
                    int(np.count_nonzero(solid)),
                    int(np.count_nonzero(fluid_position[:, 2] < UPPER_ORIGIN[2] - 0.5 * args.dx)),
                    int(np.count_nonzero(fluid_captured)),
                    int(np.count_nonzero(solid_position[:, 2] < UPPER_ORIGIN[2] - 0.5 * args.dx)),
                    float(np.linalg.norm(fluid_velocity[fluid], axis=1).max()),
                    float(np.linalg.norm(solid_velocity[solid], axis=1).max()),
                    float(pressure[fluid].min()),
                    float(pressure[fluid].max()),
                    int(np.count_nonzero(outside)),
                    float(porosity[solid].min()),
                    float(porosity[solid].max()),
                ]
            )
    rows = np.asarray(rows, dtype=np.float64)
    metrics = {
        "case": "Section 5.1 3D granular-water leakage without elastic plate",
        "velocity_projection": "Affine",
        "alpha_pic": getattr(args, "alpha_pic", 1.0),
        "physical_crack_width_m": CRACK_RIGHT - CRACK_LEFT,
        "represented_crack_width_m": args.dx,
        "snapshots": len(rows),
        "final_time_s": float(rows[-1, 0]),
        "duration_complete": math.isclose(float(rows[-1, 0]), args.time, abs_tol=0.1 * args.dt),
        "expected_fluid_particles": expected_fluid,
        "expected_solid_particles": expected_solid,
        "particle_conservation": bool(np.all(rows[:, 1] == expected_fluid) and np.all(rows[:, 2] == expected_solid)),
        "finite": finite,
        "maximum_leaked_fluid_particles": int(rows[:, 3].max()),
        "maximum_captured_fluid_particles": int(rows[:, 4].max()),
        "maximum_leaked_solid_particles": int(rows[:, 5].max()),
        "maximum_fluid_speed_mps": float(rows[:, 6].max()),
        "maximum_solid_speed_mps": float(rows[:, 7].max()),
        "minimum_fluid_pressure_pa": float(rows[:, 8].min()),
        "maximum_fluid_pressure_pa": float(rows[:, 9].max()),
        "maximum_particles_outside_domain": int(rows[:, 10].max()),
        "minimum_solid_porosity": float(rows[:, 11].min()),
        "maximum_solid_porosity": float(rows[:, 12].max()),
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and metrics["particle_conservation"]
        and metrics["maximum_particles_outside_domain"] == 0
        and metrics["maximum_leaked_fluid_particles"] > 0
        and metrics["maximum_captured_fluid_particles"] > 0
        and 0.01 <= metrics["maximum_fluid_speed_mps"] < 5.0
        and metrics["maximum_solid_speed_mps"] < 5.0
        and metrics["minimum_solid_porosity"] > 0.0
        and metrics["maximum_solid_porosity"] <= 0.64 + 1.0e-8
    )
    np.savetxt(
        output / "leakage_diagnostics.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,fluid_particles,solid_particles,leaked_fluid_particles,captured_fluid_particles,"
            "leaked_solid_particles,fluid_speed_max_mps,solid_speed_max_mps,fluid_pressure_min_pa,"
            "fluid_pressure_max_pa,particles_outside_domain,solid_porosity_min,solid_porosity_max"
        ),
        comments="",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("Section 5.1 leakage validation failed; inspect metrics.json")
