"""Evaluation and postprocessing for submarine_landslide_3d."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import math
import numpy as np
from examples.mmpm.SubmarineLandslide.submarine_landslide_2d.submarine_landslide_2d_parameters import (
    MAXIMUM_POROSITY,
    REFERENCE_DOI,
    TANK_LENGTH,
)

from examples.mmpm.SubmarineLandslide.submarine_landslide_3d.submarine_landslide_3d_parameters import (
    DOMAIN_HEIGHT,
)


def write_metrics(output, expected_fluid, expected_solid, args):
    files = sorted((output / "particles").glob("MPMParticle*.npz"))
    if len(files) < 2:
        raise RuntimeError("the 3D landslide produced fewer than two particle snapshots")
    rows = []
    finite = True
    for file_name in files:
        with np.load(file_name) as data:
            active = data["active"] > 0
            phase = data["phase"]
            fluid = active & (phase == 2)
            solid = active & (phase == 1)
            position = data["position"]
            fluid_velocity = data["fluid_velocity"]
            solid_velocity = data["solid_velocity"]
            porosity = data["porosity"]
            finite &= bool(
                np.isfinite(position[active]).all()
                and np.isfinite(fluid_velocity[fluid]).all()
                and np.isfinite(solid_velocity[solid]).all()
                and np.isfinite(data["pressure"][active]).all()
            )
            outside = np.any(
                (position[active] < -1.0e-10 * args.dx)
                | (position[active] > np.array([TANK_LENGTH, args.thickness, DOMAIN_HEIGHT]) + 1.0e-10 * args.dx),
                axis=1,
            )
            solid_position = position[solid]
            rows.append(
                [
                    float(data["t_current"]),
                    int(np.count_nonzero(fluid)),
                    int(np.count_nonzero(solid)),
                    *solid_position.mean(axis=0),
                    float(np.min(solid_position[:, 0])),
                    float(np.linalg.norm(solid_velocity[solid], axis=1).max()),
                    float(position[fluid, 2].max()),
                    int(np.count_nonzero(outside)),
                    float(porosity[solid].min()),
                    float(porosity[solid].max()),
                ]
            )
    rows = np.asarray(rows, dtype=np.float64)
    metrics = {
        "case": "Rzadkiewicz submerged landslide, thin 3D extrusion",
        "reference_doi": REFERENCE_DOI,
        "thickness_m": args.thickness,
        "interior_thickness_m": args.thickness - 2.0 * args.wall_cells * args.dx,
        "snapshots": len(rows),
        "final_time_s": float(rows[-1, 0]),
        "duration_complete": math.isclose(float(rows[-1, 0]), args.time, abs_tol=0.1 * args.dt),
        "expected_fluid_particles": expected_fluid,
        "expected_solid_particles": expected_solid,
        "particle_conservation": bool(np.all(rows[:, 1] == expected_fluid) and np.all(rows[:, 2] == expected_solid)),
        "finite": finite,
        "maximum_particles_outside_domain": int(rows[:, 9].max()),
        "solid_centroid_leftward_displacement_m": float(rows[0, 3] - rows[-1, 3]),
        "solid_centroid_downward_displacement_m": float(rows[0, 5] - rows[-1, 5]),
        "solid_front_leftward_displacement_m": float(rows[0, 6] - rows[:, 6].min()),
        "maximum_solid_speed_mps": float(rows[:, 7].max()),
        "free_surface_excursion_m": float(np.ptp(rows[:, 8])),
        "minimum_solid_porosity": float(rows[:, 10].min()),
        "maximum_solid_porosity": float(rows[:, 11].max()),
        "solid_stress_cutoff_porosity": MAXIMUM_POROSITY,
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and metrics["particle_conservation"]
        and metrics["maximum_particles_outside_domain"] == 0
        and metrics["solid_centroid_leftward_displacement_m"] >= 0.01
        and metrics["solid_centroid_downward_displacement_m"] >= 0.01
        and metrics["solid_front_leftward_displacement_m"] >= 0.01
        and metrics["maximum_solid_speed_mps"] >= 0.05
        and metrics["free_surface_excursion_m"] >= 0.002
        and metrics["minimum_solid_porosity"] > 0.0
        and metrics["maximum_solid_porosity"] <= 1.0 + 1.0e-8
    )
    np.savetxt(
        output / "landslide_3d_diagnostics.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,fluid_particles,solid_particles,solid_centroid_x_m,solid_centroid_y_m,"
            "solid_centroid_z_m,solid_front_x_m,solid_speed_max_mps,fluid_top_z_m,"
            "particles_outside_domain,solid_porosity_min,solid_porosity_max"
        ),
        comments="",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("3D submarine-landslide validation failed; inspect metrics.json")
