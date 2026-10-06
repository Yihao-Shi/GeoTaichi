"""Evaluation and postprocessing for sphere_impact_submerged_bed_3d."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import math
import numpy as np

from examples.mmpm.TwoPhaseLSDEMCoupling.sphere_impact_submerged_bed_3d.sphere_impact_submerged_bed_3d_parameters import (
    BED_HEIGHT,
    CONTAINER_RADIUS,
    CYLINDER_SIDES,
    DOMAIN,
    GLASS_POISSON_RATIO,
    GLASS_YOUNG_MODULUS,
    REFERENCE_PARTICLE_SPACING,
    SPHERE_DIAMETER,
    TEFLON_POISSON_RATIO,
    TEFLON_YOUNG_MODULUS,
    WATER_SURFACE,
)


def fluid_shell_coverage(fluid_position, center, particle_spacing):
    relative = fluid_position - center
    distance = np.linalg.norm(relative, axis=1)
    radius = 0.5 * SPHERE_DIAMETER
    shell = (distance >= radius) & (distance <= radius + 3.0 * particle_spacing)
    if not np.any(shell):
        return 0.0
    direction = relative[shell] / distance[shell, None]
    azimuth = np.floor(
        ((np.arctan2(direction[:, 1], direction[:, 0]) + 2.0 * math.pi) % (2.0 * math.pi)) * (8.0 / (2.0 * math.pi))
    ).astype(np.int64)
    polar = np.clip(np.floor((direction[:, 2] + 1.0) * 2.0).astype(np.int64), 0, 3)
    return len(np.unique(8 * polar + azimuth)) / 32.0


def write_metrics(output, expected_fluid, expected_solid, args):
    mpm_files = sorted((output / "particles").glob("MPMParticle*.npz"))
    rigid_files = sorted((output / "particles").glob("LSDEMRigid*.npz"))
    if len(mpm_files) < 2 or len(rigid_files) < 2:
        raise RuntimeError("sphere-impact run produced fewer than two MPM or LSDEM snapshots")
    mpm_rows = []
    finite = True
    angles = 2.0 * math.pi * np.arange(CYLINDER_SIDES) / CYLINDER_SIDES
    wall_normals = np.column_stack((np.cos(angles), np.sin(angles)))
    for file_name in mpm_files:
        with np.load(file_name) as data:
            active = data["active"] > 0
            phase = data["phase"]
            fluid = active & (phase == 2)
            solid = active & (phase == 1)
            position = data["position"]
            finite &= bool(
                np.isfinite(position[active]).all()
                and np.isfinite(data["fluid_velocity"][fluid]).all()
                and np.isfinite(data["solid_velocity"][solid]).all()
                and np.isfinite(data["pressure"][active]).all()
            )
            outside = np.any(
                (position[active] < -1.0e-10 * args.dx) | (position[active] > np.asarray(DOMAIN) + 1.0e-10 * args.dx),
                axis=1,
            )
            radial_position = position[active, :2] - 0.5 * np.asarray(DOMAIN[:2])
            outside_cylinder = np.any(
                radial_position @ wall_normals.T > CONTAINER_RADIUS + 1.0e-10 * args.dx,
                axis=1,
            )
            mpm_rows.append(
                [
                    float(data["t_current"]),
                    int(np.count_nonzero(fluid)),
                    int(np.count_nonzero(solid)),
                    int(np.count_nonzero(outside)),
                    int(np.count_nonzero(outside_cylinder)),
                    float(np.linalg.norm(data["fluid_velocity"][fluid], axis=1).max()),
                    float(np.linalg.norm(data["solid_velocity"][solid], axis=1).max()),
                ]
            )
    rigid_rows = []
    for file_name in rigid_files:
        with np.load(file_name) as data:
            center = data["mass_center"][0]
            velocity = data["velocity"][0]
            force = data["contact_force"][0]
            finite &= bool(np.isfinite(center).all() and np.isfinite(velocity).all() and np.isfinite(force).all())
            rigid_rows.append([float(data["t_current"]), *center, *velocity, float(np.linalg.norm(force))])
    mpm_rows = np.asarray(mpm_rows, dtype=np.float64)
    rigid_rows = np.asarray(rigid_rows, dtype=np.float64)
    submerged_coverage = []
    for mpm_file, rigid_file in zip(mpm_files, rigid_files):
        with np.load(mpm_file) as particles, np.load(rigid_file) as rigid:
            center = rigid["mass_center"][0]
            if center[2] + 0.5 * SPHERE_DIAMETER <= WATER_SURFACE + args.dx:
                active_fluid = (particles["active"] > 0) & (particles["phase"] == 2)
                submerged_coverage.append(
                    fluid_shell_coverage(particles["position"][active_fluid], center, args.dx / args.ppc)
                )
    impact_speed = math.sqrt(2.0 * 9.81 * args.drop_height)
    radius = 0.5 * SPHERE_DIAMETER
    initial_center_z = WATER_SURFACE + radius
    minimum_center_z = float(rigid_rows[:, 3].min())
    metrics = {
        "case": "Section 5.2 LSDEM sphere impact into a saturated granular bed",
        "drop_height_m": args.drop_height,
        "equivalent_impact_speed_mps": impact_speed,
        "container_diameter_m": 2.0 * CONTAINER_RADIUS,
        "cylindrical_wall_planes": CYLINDER_SIDES,
        "reference_particle_spacing_m": REFERENCE_PARTICLE_SPACING,
        "mpm_particle_spacing_m": args.dx / args.ppc,
        "teflon_young_modulus_pa": TEFLON_YOUNG_MODULUS,
        "teflon_poisson_ratio": TEFLON_POISSON_RATIO,
        "glass_young_modulus_pa": GLASS_YOUNG_MODULUS,
        "glass_poisson_ratio": GLASS_POISSON_RATIO,
        "snapshots_mpm": len(mpm_rows),
        "snapshots_lsdem": len(rigid_rows),
        "final_time_s": float(min(mpm_rows[-1, 0], rigid_rows[-1, 0])),
        "duration_complete": bool(
            math.isclose(mpm_rows[-1, 0], args.time, abs_tol=0.1 * args.dt)
            and math.isclose(rigid_rows[-1, 0], args.time, abs_tol=0.1 * args.dt)
        ),
        "expected_fluid_particles": expected_fluid,
        "expected_solid_particles": expected_solid,
        "particle_conservation": bool(
            np.all(mpm_rows[:, 1] == expected_fluid) and np.all(mpm_rows[:, 2] == expected_solid)
        ),
        "finite": finite,
        "maximum_mpm_particles_outside_domain": int(mpm_rows[:, 3].max()),
        "maximum_mpm_particles_outside_cylinder": int(mpm_rows[:, 4].max()),
        "maximum_fluid_speed_mps": float(mpm_rows[:, 5].max()),
        "maximum_solid_speed_mps": float(mpm_rows[:, 6].max()),
        "sphere_minimum_center_z_m": minimum_center_z,
        "sphere_water_entry_displacement_m": initial_center_z - minimum_center_z,
        "sphere_maximum_bed_penetration_m": max(0.0, BED_HEIGHT - (minimum_center_z - radius)),
        "sphere_final_vertical_velocity_mps": float(rigid_rows[-1, 6]),
        "maximum_coupling_force_n": float(rigid_rows[:, 7].max()),
        "minimum_submerged_fluid_shell_coverage": float(min(submerged_coverage, default=0.0)),
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and metrics["particle_conservation"]
        and metrics["maximum_mpm_particles_outside_domain"] == 0
        and metrics["maximum_mpm_particles_outside_cylinder"] == 0
        and metrics["sphere_water_entry_displacement_m"] >= 0.02
        and metrics["sphere_maximum_bed_penetration_m"] >= 0.002
        and metrics["sphere_minimum_center_z_m"] - radius >= 0.5 * args.dx
        and abs(metrics["sphere_final_vertical_velocity_mps"]) < 0.9 * impact_speed
        and metrics["maximum_coupling_force_n"] > 0.0
        and metrics["minimum_submerged_fluid_shell_coverage"] >= 0.75
    )
    np.savetxt(
        output / "sphere_impact_trajectory.csv",
        rigid_rows,
        delimiter=",",
        header="time_s,center_x_m,center_y_m,center_z_m,velocity_x_mps,velocity_y_mps,velocity_z_mps,coupling_force_n",
        comments="",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("Section 5.2 sphere-impact validation failed; inspect metrics.json")
