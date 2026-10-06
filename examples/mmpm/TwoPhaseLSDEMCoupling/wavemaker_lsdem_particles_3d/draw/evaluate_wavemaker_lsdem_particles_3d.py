"""Evaluation and postprocessing for wavemaker_lsdem_particles_3d."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import math
import numpy as np
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d.draw.evaluate_wavemaker_tank_3d import (
    linear_piston_wave,
    sdf_surface_gauge,
)

from examples.mmpm.TwoPhaseLSDEMCoupling.wavemaker_lsdem_particles_3d.wavemaker_lsdem_particles_3d_parameters import (
    LSDEM_BODY_COUNT,
    TANK,
    WATER_DEPTH,
)


def write_metrics(output, expected_fluid, expected_solid, args):
    mpm_files = sorted((output / "particles").glob("MPMParticle*.npz"))
    rigid_files = sorted((output / "particles").glob("LSDEMRigid*.npz"))
    grids = {path.stem.removeprefix("MPMGrid"): path for path in (output / "grids").glob("MPMGrid*.npz")}
    if len(mpm_files) < 2 or len(rigid_files) < 2:
        raise RuntimeError("coupled wavemaker produced fewer than two MPM or LSDEM snapshots")

    mpm_rows = []
    initial_solid_position = None
    finite = True
    for file_name in mpm_files:
        frame = file_name.stem.removeprefix("MPMParticle")
        if frame not in grids:
            raise RuntimeError(f"missing grid snapshot for MPM frame {frame}")
        with np.load(file_name) as data, np.load(grids[frame]) as grid:
            active = data["active"] > 0
            phase = data["phase"]
            fluid = active & (phase == 2)
            solid = active & (phase == 1)
            position = data["position"]
            if initial_solid_position is None:
                initial_solid_position = position[solid].copy()
            solid_displacement = float(np.linalg.norm(position[solid] - initial_solid_position, axis=1).max())
            time = float(data["t_current"])
            surface = (
                WATER_DEPTH
                if time == 0.0
                else sdf_surface_gauge(grid["cell_type"], grid["cell_fluid_sdf"], args.dx, 0.50, 2.0 * args.dx)
            )
            outside = np.any(
                (position[active] < -1.0e-10 * args.dx) | (position[active] > np.asarray(TANK) + 1.0e-10 * args.dx),
                axis=1,
            )
            finite &= bool(
                np.isfinite(position[active]).all()
                and np.isfinite(data["fluid_velocity"][fluid]).all()
                and np.isfinite(data["solid_velocity"][solid]).all()
                and np.isfinite(data["pressure"][active]).all()
                and np.isfinite(surface)
            )
            mpm_rows.append(
                [
                    time,
                    np.count_nonzero(fluid),
                    np.count_nonzero(solid),
                    np.count_nonzero(outside),
                    surface,
                    solid_displacement,
                ]
            )

    rigid_rows = []
    for file_name in rigid_files:
        with np.load(file_name) as data:
            centers = data["mass_center"]
            forces = data["contact_force"]
            finite &= bool(np.isfinite(centers).all() and np.isfinite(forces).all())
            rigid_rows.append([float(data["t_current"]), float(np.linalg.norm(forces, axis=1).max()), len(centers)])
    mpm_rows = np.asarray(mpm_rows, dtype=np.float64)
    rigid_rows = np.asarray(rigid_rows, dtype=np.float64)
    _, expected_amplitude, _ = linear_piston_wave(args.frequency, WATER_DEPTH, args.wave_velocity)
    with np.load(rigid_files[0]) as data:
        initial_centers = data["mass_center"].copy()
    maximum_rigid_displacement = 0.0
    for file_name in rigid_files:
        with np.load(file_name) as data:
            maximum_rigid_displacement = max(
                maximum_rigid_displacement,
                float(np.linalg.norm(data["mass_center"] - initial_centers, axis=1).max()),
            )
    metrics = {
        "case": "3D two-phase two-point semi-implicit MPM--LSDEM wavemaker",
        "coupling": {"fluid_lsdem": "IBM", "solid_lsdem": "ordinary point-level-set contact"},
        "velocity_projection": "Affine",
        "alpha_pic": getattr(args, "alpha_pic", 1.0),
        "final_time_s": float(min(mpm_rows[-1, 0], rigid_rows[-1, 0])),
        "duration_complete": bool(
            math.isclose(mpm_rows[-1, 0], args.time, abs_tol=0.1 * args.dt)
            and math.isclose(rigid_rows[-1, 0], args.time, abs_tol=0.1 * args.dt)
        ),
        "finite": finite,
        "expected_fluid_particles": expected_fluid,
        "expected_solid_particles": expected_solid,
        "particle_conservation": bool(
            np.all(mpm_rows[:, 1] == expected_fluid) and np.all(mpm_rows[:, 2] == expected_solid)
        ),
        "maximum_mpm_particles_outside_domain": int(mpm_rows[:, 3].max()),
        "maximum_mpm_solid_displacement_m": float(mpm_rows[:, 5].max()),
        "expected_linear_wave_amplitude_m": expected_amplitude,
        "surface_gauge_excursion_m": float(np.ptp(mpm_rows[:, 4])),
        "maximum_lsdem_displacement_m": maximum_rigid_displacement,
        "maximum_coupling_force_n": float(rigid_rows[:, 1].max()),
        "expected_lsdem_bodies": LSDEM_BODY_COUNT,
        "minimum_lsdem_bodies": int(rigid_rows[:, 2].min()),
        "soil_young_modulus_pa": args.soil_young_modulus,
        "soil_cohesion_pa": args.soil_cohesion,
        "soil_friction_deg": args.soil_friction,
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and metrics["particle_conservation"]
        and metrics["maximum_mpm_particles_outside_domain"] == 0
        and metrics["minimum_lsdem_bodies"] == metrics["expected_lsdem_bodies"]
        and metrics["surface_gauge_excursion_m"] >= max(0.75 * args.dx, expected_amplitude)
        and metrics["maximum_mpm_solid_displacement_m"] >= 0.05 * args.dx
        and metrics["maximum_lsdem_displacement_m"] > 1.0e-5
        and metrics["maximum_coupling_force_n"] > 0.0
    )
    np.savetxt(
        output / "coupled_wavemaker_diagnostics.csv",
        mpm_rows,
        delimiter=",",
        header="time_s,fluid_particles,solid_particles,particles_outside_domain,surface_gauge_m,solid_displacement_max_m",
        comments="",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("coupled wavemaker validation failed; inspect metrics.json")
