#!/usr/bin/env python3
"""Velocity-controlled isotropic compaction of FEM-soft/LSDEM mixtures."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.fedem.IsotropicCompaction.draw.output_audit import collect_native_output
from examples.fedem.IsotropicCompaction import isotropic_setup as base

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "OutputData/soft_050"


def _default_fem_relaxation_damping() -> float:
    particle_mass = (4.0 / 3.0) * math.pi * base.RADIUS**3 * base.SOFT_DENSITY
    characteristic_stiffness = base.YOUNG * base.RADIUS
    damping_ratio = 0.70
    return 2.0 * damping_ratio * math.sqrt(characteristic_stiffness / particle_mass)


def _set_dem_local_damping(dem, value: float) -> None:
    """Set Cundall force and torque damping for mobile LSDEM grains."""

    dem.update_material_properties(
        materialID=0,
        property_name="ForceLocalDamping",
        value=float(value),
        override=True,
    )
    dem.update_material_properties(
        materialID=0,
        property_name="TorqueLocalDamping",
        value=float(value),
        override=True,
    )


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_json(path: Path, payload) -> None:
    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.write_text(
        json.dumps(payload, indent=2, default=json_default) + os.linesep,
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _initialization_description(args) -> str:
    if args.direct_velocity_loading:
        return "direct constant-velocity loading from the generated packing"
    if args.paper_jamming_initialization:
        return (
            "contact-network preparation, fixed-volume relaxation, and "
            "low-pressure isotropic stress control until the packing "
            "fraction stabilizes"
        )
    return "constant-velocity contact-network formation followed by " "low-pressure isotropic stress equilibration"


def _initial_jamming_definition(args) -> dict:
    criteria = {
        "relative_solid_fraction_change": args.solid_fraction_tolerance,
        "maximum_kinetic_energy_ratio": args.kinetic_tolerance,
        "kinetic_energy_ratio": "K / (max(p, p0) V)",
        "required_consecutive_checks": args.required_stable_checks,
        "paper_jamming_initialization": bool(args.paper_jamming_initialization),
    }
    if args.paper_jamming_initialization:
        criteria["relative_mean_pressure_error"] = args.pressure_tolerance
        criteria["minimum_coordination_number"] = args.minimum_jamming_coordination
    else:
        criteria.update(
            {
                "maximum_wall_pressure_error": args.pressure_tolerance,
                "minimum_coordination_number": (args.minimum_jamming_coordination),
            }
        )
    return criteria


def _set_composition(soft_percent: float) -> tuple[int, int]:
    soft_count = int(math.floor(base.PARTICLE_COUNT * soft_percent / 100.0 + 0.5))
    rigid_count = base.PARTICLE_COUNT - soft_count
    base.SOFT_COUNT = soft_count
    base.RIGID_COUNT = rigid_count
    return soft_count, rigid_count


def _initial_bounding_clearance(centers, radii=None) -> dict[str, float]:
    centers = np.asarray(centers, dtype=np.float64)
    if radii is None:
        radii = np.full(len(centers), base.RADIUS)
    radii = np.asarray(radii, dtype=np.float64)
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    distance[np.diag_indices_from(distance)] = np.inf
    pair_clearance = float(np.min(distance - radii[:, None] - radii[None, :]))
    wall_clearance = float(
        np.min(
            np.column_stack(
                (
                    centers[:, 0] - base.LEFT - radii,
                    base.RIGHT - centers[:, 0] - radii,
                    centers[:, 1] - base.FRONT - radii,
                    base.BACK - centers[:, 1] - radii,
                    centers[:, 2] - base.BOTTOM - radii,
                    base.TOP - centers[:, 2] - radii,
                )
            )
        )
    )
    return {
        "minimum_bounding_pair_clearance": pair_clearance,
        "minimum_bounding_wall_clearance": wall_clearance,
    }


def _randomized_spherical_packing(args):
    """Create one deterministic, weakly polydisperse disordered packing."""

    from scipy.optimize import minimize

    count = base.PARTICLE_COUNT
    rng = np.random.default_rng(args.packing_seed)
    spread = float(args.diameter_polydispersity)
    radii = base.RADIUS * rng.uniform(1.0 - spread, 1.0 + spread, count)
    radii *= (count * base.RADIUS**3 / np.sum(radii**3)) ** (1.0 / 3.0)
    clearance = args.barrier_cutoff if args.contact_model == "barrier" else 0.0
    lower = np.column_stack(
        [np.full(count, bound) + radii + clearance for bound in (base.LEFT, base.FRONT, base.BOTTOM)]
    )
    upper = np.column_stack([np.full(count, bound) - radii - clearance for bound in (base.RIGHT, base.BACK, base.TOP)])
    pair_i, pair_j = np.triu_indices(count, 1)
    required_distance = radii[pair_i] + radii[pair_j] + clearance

    def overlap_energy_gradient(flat):
        centers = flat.reshape(count, 3)
        delta = centers[pair_i] - centers[pair_j]
        distance = np.linalg.norm(delta, axis=1)
        overlap = required_distance - distance
        active = overlap > 0.0
        if not np.any(active):
            return 0.0, np.zeros_like(flat)
        active_distance = np.maximum(distance[active], 1.0e-14)
        active_overlap = overlap[active]
        direction = delta[active] / active_distance[:, None]
        pair_gradient = -active_overlap[:, None] * direction
        gradient = np.zeros_like(centers)
        np.add.at(gradient, pair_i[active], pair_gradient)
        np.add.at(gradient, pair_j[active], -pair_gradient)
        return (
            0.5 * float(np.dot(active_overlap, active_overlap)),
            gradient.ravel(),
        )

    result = None
    centers = None
    minimum_gap = -np.inf
    bounds = list(zip(lower.ravel(), upper.ravel()))
    for _ in range(8):
        initial = rng.uniform(lower, upper)
        result = minimize(
            overlap_energy_gradient,
            initial.ravel(),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 10000,
                "ftol": 1.0e-20,
                "gtol": 1.0e-12,
                "maxls": 100,
            },
        )
        centers = result.x.reshape(count, 3)
        distances = np.linalg.norm(centers[pair_i] - centers[pair_j], axis=1)
        minimum_gap = float(np.min(distances - required_distance))
        if minimum_gap >= -1.0e-9:
            break
    if minimum_gap < -1.0e-9:
        raise RuntimeError(
            "failed to generate the randomized non-overlapping packing: " f"minimum gap {minimum_gap:.12g} m"
        )

    permutation = rng.permutation(count)
    soft_ids = np.sort(permutation[: base.SOFT_COUNT])
    rigid_ids = np.sort(permutation[base.SOFT_COUNT :])
    orientations = rng.uniform(0.0, 360.0, (count, 3))
    metadata = {
        "type": "random weakly polydisperse packing",
        "seed": int(args.packing_seed),
        "diameter_polydispersity": spread,
        "minimum_pair_clearance": minimum_gap + clearance,
        "minimum_required_clearance": clearance,
        "optimizer_iterations": int(result.nit),
    }
    return (
        centers,
        radii,
        soft_ids,
        rigid_ids,
        orientations,
    ), metadata


def _wall_positions_and_forces(dem, wall_facet_ids):
    count = int(dem.scene.wallNum[0])
    point1 = dem.scene.wall.vertice1.to_numpy()[:count]
    point2 = dem.scene.wall.vertice2.to_numpy()[:count]
    point3 = dem.scene.wall.vertice3.to_numpy()[:count]
    force = dem.scene.wall.contact_force.to_numpy()[:count]
    positions = {}
    forces = {}
    for name, facet_ids in wall_facet_ids.items():
        centers = (point1[facet_ids] + point2[facet_ids] + point3[facet_ids]) / 3.0
        axis = 0 if name in ("left", "right") else 1
        if name in ("bottom", "top"):
            axis = 2
        positions[name] = float(np.mean(centers[:, axis]))
        forces[name] = np.sum(force[facet_ids], axis=0)
    return positions, forces


def _wall_observation(dem, wall_facet_ids):
    positions, forces = _wall_positions_and_forces(dem, wall_facet_ids)
    width = positions["right"] - positions["left"]
    depth = positions["back"] - positions["front"]
    height = positions["top"] - positions["bottom"]
    areas = {
        "left": depth * height,
        "right": depth * height,
        "front": width * height,
        "back": width * height,
        "bottom": width * depth,
        "top": width * depth,
    }
    components = {
        "left": 0,
        "right": 0,
        "front": 1,
        "back": 1,
        "bottom": 2,
        "top": 2,
    }
    wall_stress = {
        name: abs(float(forces[name][components[name]])) / max(areas[name], 1.0e-30) for name in base.WALL_BODY_NAMES
    }
    sigma_x = 0.5 * (wall_stress["left"] + wall_stress["right"])
    sigma_y = 0.5 * (wall_stress["front"] + wall_stress["back"])
    sigma_z = 0.5 * (wall_stress["bottom"] + wall_stress["top"])
    return {
        "positions": positions,
        "forces": forces,
        "wall_stress": wall_stress,
        "width": width,
        "depth": depth,
        "height": height,
        "box_volume": width * depth * height,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "mean_pressure": (sigma_x + sigma_y + sigma_z) / 3.0,
    }


def _minimum_lsdem_wall_clearance(dem, wall_facet_ids) -> float:
    scene = dem.scene
    if int(scene.surfaceNum[0]) <= 0:
        return math.inf
    _, surface_points = scene.visualize_surface(dem.sims)
    surface_points = surface_points[: int(scene.surfaceNum[0])]
    wall_count = int(scene.wallNum[0])
    point1 = scene.wall.vertice1.to_numpy()[:wall_count]
    point2 = scene.wall.vertice2.to_numpy()[:wall_count]
    point3 = scene.wall.vertice3.to_numpy()[:wall_count]
    normal = scene.wall.norm.to_numpy()[:wall_count]
    minimum = math.inf
    for facet_ids in wall_facet_ids.values():
        facet = int(facet_ids[0])
        center = (point1[facet] + point2[facet] + point3[facet]) / 3.0
        minimum = min(
            minimum,
            float(np.min((surface_points - center) @ normal[facet])),
        )
    return minimum


def _make_six_wall_servo(
    ti,
    wall,
    wall_facet_ids,
    target_stress,
    velocity_limit: float,
    proportional_gain: float,
):
    """Return a device-resident stress controller for all six wall groups."""

    groups = tuple(tuple(int(value) for value in wall_facet_ids[name]) for name in base.WALL_BODY_NAMES)
    if any(len(group) != 2 for group in groups):
        raise RuntimeError(f"six-wall servo requires two facets per wall: {groups}")

    left, right, front, back, bottom, top = groups

    @ti.func
    def group_center(first: ti.i32, second: ti.i32):
        return 0.5 * (wall[first]._get_center() + wall[second]._get_center())

    @ti.func
    def group_force(first: ti.i32, second: ti.i32):
        return wall[first].contact_force + wall[second].contact_force

    @ti.func
    def normal_speed(current_stress: float):
        target = ti.max(target_stress[None], 1.0e-30)
        normalized_error = (target - current_stress) / target
        command = ti.max(
            -1.0,
            ti.min(1.0, proportional_gain * normalized_error),
        )
        return velocity_limit * command

    @ti.func
    def assign_group_velocity(first: ti.i32, second: ti.i32, speed: float):
        wall[first].v = speed * wall[first].norm
        wall[second].v = speed * wall[second].norm

    @ti.kernel
    def update():
        left_center = group_center(left[0], left[1])
        right_center = group_center(right[0], right[1])
        front_center = group_center(front[0], front[1])
        back_center = group_center(back[0], back[1])
        bottom_center = group_center(bottom[0], bottom[1])
        top_center = group_center(top[0], top[1])

        width = ti.max(right_center[0] - left_center[0], 1.0e-30)
        depth = ti.max(back_center[1] - front_center[1], 1.0e-30)
        height = ti.max(top_center[2] - bottom_center[2], 1.0e-30)
        x_area = depth * height
        y_area = width * height
        z_area = width * depth

        left_stress = ti.abs(group_force(left[0], left[1])[0]) / x_area
        right_stress = ti.abs(group_force(right[0], right[1])[0]) / x_area
        front_stress = ti.abs(group_force(front[0], front[1])[1]) / y_area
        back_stress = ti.abs(group_force(back[0], back[1])[1]) / y_area
        bottom_stress = ti.abs(group_force(bottom[0], bottom[1])[2]) / z_area
        top_stress = ti.abs(group_force(top[0], top[1])[2]) / z_area

        assign_group_velocity(left[0], left[1], normal_speed(left_stress))
        assign_group_velocity(right[0], right[1], normal_speed(right_stress))
        assign_group_velocity(front[0], front[1], normal_speed(front_stress))
        assign_group_velocity(back[0], back[1], normal_speed(back_stress))
        assign_group_velocity(bottom[0], bottom[1], normal_speed(bottom_stress))
        assign_group_velocity(top[0], top[1], normal_speed(top_stress))

    return update


def _make_isotropic_wall_velocity(ti, wall, wall_facet_ids):
    """Return a device kernel assigning one inward normal speed to six walls."""

    groups = tuple(tuple(int(value) for value in wall_facet_ids[name]) for name in base.WALL_BODY_NAMES)
    facet_ids = tuple(value for group in groups for value in group)
    if len(facet_ids) != 12:
        raise RuntimeError(f"isotropic compression requires twelve facets: {groups}")

    @ti.kernel
    def assign(speed: float):
        for facet_id in ti.static(facet_ids):
            wall[facet_id].v = speed * wall[facet_id].norm

    return assign


def _rigid_kinetic(dem, count: int):
    if count <= 0:
        return 0.0, 0.0, True
    mass = dem.scene.rigid.m.to_numpy()[:count]
    velocity = dem.scene.rigid.v.to_numpy()[:count]
    omega = dem.scene.rigid.w.to_numpy()[:count]
    angular_momentum = dem.scene.rigid.angmoment.to_numpy()[:count]
    translational = float(0.5 * np.sum(mass[:, None] * velocity * velocity))
    rotational = float(0.5 * np.sum(omega * angular_momentum))
    finite = bool(
        np.isfinite(mass).all()
        and np.isfinite(velocity).all()
        and np.isfinite(omega).all()
        and np.isfinite(angular_momentum).all()
        and math.isfinite(translational)
        and math.isfinite(rotational)
    )
    return translational, rotational, finite


def _mixed_observation(
    coupling,
    soft_ids,
    rigid_ids,
    body_ranges,
    wall_facet_ids,
    target_pressure: Optional[float],
    reduced_young: float,
    initial_wall_positions,
):
    fem_engine = coupling.enginer.fem_engine
    positions = fem_engine.state.position.to_numpy()
    velocity = fem_engine.state.velocity.to_numpy()
    mass = fem_engine.state.mass.to_numpy()
    soft_kinetic = float(0.5 * np.sum(mass[:, None] * velocity * velocity))
    rigid_translational, rigid_rotational, rigid_finite = _rigid_kinetic(coupling.dem, base.RIGID_COUNT)
    wall = _wall_observation(coupling.dem, wall_facet_ids)
    soft_volume = base._soft_volume(coupling, positions)
    if base.RIGID_COUNT > 0:
        rigid_mass = coupling.dem.scene.rigid.m.to_numpy()[: base.RIGID_COUNT]
        rigid_volume = float(np.sum(rigid_mass) / base.RIGID_DENSITY)
    else:
        rigid_volume = 0.0
    pairs = base._particle_contact_pairs(coupling, soft_ids, rigid_ids)
    soft_particles = set(int(value) for value in soft_ids)
    soft_soft_pairs = 0
    soft_rigid_pairs = 0
    rigid_rigid_pairs = 0
    for first, second in pairs:
        first_soft = first in soft_particles
        second_soft = second in soft_particles
        if first_soft and second_soft:
            soft_soft_pairs += 1
        elif first_soft or second_soft:
            soft_rigid_pairs += 1
        else:
            rigid_rigid_pairs += 1
    soft_coordination = (2.0 * soft_soft_pairs + soft_rigid_pairs) / base.SOFT_COUNT if base.SOFT_COUNT > 0 else 0.0
    rigid_coordination = (
        (2.0 * rigid_rigid_pairs + soft_rigid_pairs) / base.RIGID_COUNT if base.RIGID_COUNT > 0 else 0.0
    )
    pressure_scale = float(target_pressure) if target_pressure is not None else wall["mean_pressure"]
    wall_error = max(
        abs(value - pressure_scale) / max(abs(pressure_scale), 1.0e-30) for value in wall["wall_stress"].values()
    )
    kinetic = soft_kinetic + rigid_translational + rigid_rotational
    kinetic_reference_pressure = max(
        abs(float(wall["mean_pressure"])),
        abs(float(pressure_scale)),
        1.0e-30,
    )
    kinetic_ratio = kinetic / max(kinetic_reference_pressure * wall["box_volume"], 1.0e-30)
    minimum_jacobian = float(fem_engine._minimum_jacobian_ratio_device(fem_engine.state.position))
    solid_fraction = (soft_volume + rigid_volume) / max(wall["box_volume"], 1.0e-30)
    return {
        **wall,
        "soft_solid_volume": soft_volume,
        "rigid_solid_volume": rigid_volume,
        "solid_fraction": solid_fraction,
        "particle_contact_pair_count": len(pairs),
        "coordination_number": 2.0 * len(pairs) / base.PARTICLE_COUNT,
        "soft_soft_contact_pair_count": soft_soft_pairs,
        "soft_rigid_contact_pair_count": soft_rigid_pairs,
        "rigid_rigid_contact_pair_count": rigid_rigid_pairs,
        "soft_coordination_number": soft_coordination,
        "rigid_coordination_number": rigid_coordination,
        "soft_kinetic_energy": soft_kinetic,
        "rigid_translational_kinetic_energy": rigid_translational,
        "rigid_rotational_kinetic_energy": rigid_rotational,
        "kinetic_energy": kinetic,
        "kinetic_pressure_work_ratio": kinetic_ratio,
        "kinetic_reference_pressure": kinetic_reference_pressure,
        "relative_wall_pressure_error": wall_error,
        "minimum_jacobian": minimum_jacobian,
        "pressure_over_reduced_young": wall["mean_pressure"] / reduced_young,
        "top_wall_displacement": (initial_wall_positions["top"] - wall["positions"]["top"]),
        "axial_stress": wall["sigma_z"],
        "finite": bool(
            np.isfinite(positions).all()
            and np.isfinite(velocity).all()
            and rigid_finite
            and math.isfinite(wall["mean_pressure"])
            and math.isfinite(solid_fraction)
        ),
    }


def _path_row(
    phase: str,
    step: int,
    time_value: float,
    observation,
    initial_dimensions,
):
    wall_stress = observation["wall_stress"]
    initial_width, initial_depth, initial_height = initial_dimensions
    strain_x = 1.0 - observation["width"] / initial_width
    strain_y = 1.0 - observation["depth"] / initial_depth
    strain_z = 1.0 - observation["height"] / initial_height
    initial_volume = initial_width * initial_depth * initial_height
    return {
        "phase": phase,
        "step": step,
        "time": time_value,
        "mean_confining_pressure": observation["mean_pressure"],
        "pressure_over_reduced_young": observation["pressure_over_reduced_young"],
        "confining_stress_x": observation["sigma_x"],
        "confining_stress_y": observation["sigma_y"],
        "confining_stress_z": observation["sigma_z"],
        "pressure_anisotropy": observation["relative_wall_pressure_error"],
        "left_wall_stress": wall_stress["left"],
        "right_wall_stress": wall_stress["right"],
        "front_wall_stress": wall_stress["front"],
        "back_wall_stress": wall_stress["back"],
        "bottom_wall_stress": wall_stress["bottom"],
        "top_wall_stress": wall_stress["top"],
        "solid_fraction": observation["solid_fraction"],
        "soft_solid_volume": observation["soft_solid_volume"],
        "rigid_solid_volume": observation["rigid_solid_volume"],
        "box_volume": observation["box_volume"],
        "width": observation["width"],
        "depth": observation["depth"],
        "height": observation["height"],
        "isotropic_strain": (strain_x + strain_y + strain_z) / 3.0,
        "volumetric_strain": 1.0 - observation["box_volume"] / initial_volume,
        "particle_contact_pair_count": observation["particle_contact_pair_count"],
        "coordination_number": observation["coordination_number"],
        "soft_soft_contact_pair_count": observation["soft_soft_contact_pair_count"],
        "soft_rigid_contact_pair_count": observation["soft_rigid_contact_pair_count"],
        "rigid_rigid_contact_pair_count": observation["rigid_rigid_contact_pair_count"],
        "soft_coordination_number": observation["soft_coordination_number"],
        "rigid_coordination_number": observation["rigid_coordination_number"],
        "top_wall_displacement": observation["top_wall_displacement"],
        "axial_stress": observation["axial_stress"],
        "soft_kinetic_energy": observation["soft_kinetic_energy"],
        "rigid_translational_kinetic_energy": observation["rigid_translational_kinetic_energy"],
        "rigid_rotational_kinetic_energy": observation["rigid_rotational_kinetic_energy"],
        "kinetic_energy": observation["kinetic_energy"],
        "kinetic_pressure_work_ratio": observation["kinetic_pressure_work_ratio"],
        "kinetic_reference_pressure": observation["kinetic_reference_pressure"],
        "minimum_jacobian": observation["minimum_jacobian"],
    }


def _build_mixed(gt, args, output: Path, total_time: float):
    packing_override = None
    packing_metadata = {"type": "near-lattice monodisperse packing"}
    if args.randomized_packing:
        packing_override, packing_metadata = _randomized_spherical_packing(args)
    coupling_and_metadata = base._build(
        gt,
        output,
        args.dt,
        total_time,
        0.0,
        total_time,
        contact_model=args.contact_model,
        barrier_cutoff=args.barrier_cutoff,
        fem_verlet_distance_multiplier=(args.fem_verlet_distance_multiplier),
        dem_local_damping=args.dem_loading_local_damping,
        rigid_shape=args.rigid_shape,
        packing_override=packing_override,
    )
    coupling_and_metadata[0].initial_packing_metadata = packing_metadata
    return coupling_and_metadata


def _restart_initial_wall(checkpoint: Path, fallback):
    state_path = checkpoint.resolve().parents[2] / "state.json"
    if not state_path.is_file():
        return fallback
    state = json.loads(state_path.read_text(encoding="utf-8"))
    saved = state.get("initial_wall_positions")
    if not isinstance(saved, dict):
        return fallback
    return {name: np.asarray(value, dtype=np.float64) for name, value in saved.items()}


def _restart_csv_rows(checkpoint: Path, filename: str, maximum_step: int) -> list[dict]:
    path = checkpoint.resolve().parents[2] / filename
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if int(float(row.get("step", -1))) <= maximum_step]


def _warm_start_kinematics(coupling, checkpoint: Path) -> None:
    """Transfer an equilibrated configuration onto a rebuilt contact mesh."""

    state = coupling.enginer.fem_engine.state
    scene = coupling.dem.scene
    with np.load(checkpoint) as source:
        positions = np.asarray(source["state/fem.state/position"], dtype=np.float64)
        reference = np.asarray(source["state/fem.state/reference_position"], dtype=np.float64)
        if positions.shape != (state.node_count, 3):
            raise ValueError("warm-start FEM node count does not match")
        current_reference = state.reference_position.to_numpy()
        if not np.allclose(reference, current_reference, rtol=0.0, atol=1.0e-12):
            raise ValueError("warm-start FEM node ordering does not match")

        positions = np.ascontiguousarray(positions, dtype=state.numpy_type)
        for field in (
            state.position,
            state.old_position,
            state.trial_position,
            state.predicted_position,
        ):
            field.from_numpy(positions)
        for field in (
            state.velocity,
            state.old_velocity,
            state.acceleration,
            state.old_acceleration,
            state.reaction,
            state.external_force,
            state.residual,
            state.direction,
        ):
            field.fill(0.0)
        state.damping_dissipation.fill(0.0)

        rigid_count = int(scene.rigidNum[0])
        saved_centers = np.asarray(source["state/dem.scene/rigid/mass_center"], dtype=np.float64)
        if saved_centers.shape != (rigid_count, 3):
            raise ValueError("warm-start LSDEM body count does not match")
        initial_centers = scene.rigid.mass_center.to_numpy()
        center_shift = saved_centers - initial_centers[:rigid_count]
        initial_centers[:rigid_count] = saved_centers
        scene.rigid.mass_center.from_numpy(np.ascontiguousarray(initial_centers))

        bounding_centers = scene.particle.x.to_numpy()
        bounding_centers[:rigid_count] += center_shift
        scene.particle.x.from_numpy(np.ascontiguousarray(bounding_centers))
        scene.particle.verletDisp.fill(0.0)
        for field in (
            scene.rigid.v,
            scene.rigid.w,
            scene.rigid.a,
            scene.rigid.angmoment,
            scene.rigid.contact_force,
            scene.rigid.contact_torque,
        ):
            field.fill(0.0)
        scene.rigid.damp_energy.fill(0.0)

        wall_translation = np.asarray(source["state/dem.scene/wall/translation"], dtype=np.float64)
        current_translation = scene.wall.translation.to_numpy()
        if wall_translation.shape[0] != int(scene.wallNum[0]):
            raise ValueError("warm-start DEM wall count does not match")
        current_translation[: wall_translation.shape[0]] = wall_translation
        scene.wall.translation.from_numpy(np.ascontiguousarray(current_translation))
        scene.wall.v.fill(0.0)
        scene.wall.verletDisp.fill(0.0)
        scene.wall.contact_force.fill(0.0)


def _run_mixed(gt, ti, args, output: Path, reduced_young) -> int:
    restart_checkpoint = args.restart_checkpoint.expanduser().resolve() if args.restart_checkpoint is not None else None
    warm_start_checkpoint = (
        args.warm_start_checkpoint.expanduser().resolve() if args.warm_start_checkpoint is not None else None
    )
    if restart_checkpoint is not None:
        if not restart_checkpoint.is_file():
            raise FileNotFoundError(restart_checkpoint)
        try:
            restart_checkpoint.relative_to(output.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("restart output must differ from the checkpoint source")
    initial_pressure = args.initial_pressure_ratio * reduced_young
    compression_steps = int(
        math.ceil(
            args.maximum_isotropic_strain
            * min(base.WIDTH0, base.DEPTH0, base.HEIGHT0)
            / (2.0 * args.compression_velocity * args.dt)
        )
    )
    maximum_steps = args.max_consolidation_steps + compression_steps
    total_time = max(maximum_steps * args.dt, args.dt)
    base._clear_fresh_run_output(output)
    started = time.perf_counter()
    (
        coupling,
        packing_centers,
        soft_ids,
        rigid_ids,
        body_ranges,
        wall_facet_ids,
        body_roles,
        mesh_quality,
        nodes_per_soft_particle,
        elements_per_soft_particle,
    ) = _build_mixed(gt, args, output, total_time)
    if warm_start_checkpoint is not None:
        if not warm_start_checkpoint.is_file():
            raise FileNotFoundError(warm_start_checkpoint)
        _warm_start_kinematics(coupling, warm_start_checkpoint)
    packing_radii = coupling.initial_packing_radii
    packing_metadata = coupling.initial_packing_metadata
    coupling.contactor.fixed_facet_walls = False
    coupling.contactor.rebuild_wall_candidates(coupling.dem.scene.wall)
    initial_surface_clearances, _ = base._surface_clearances(coupling, wall_facet_ids, body_ranges)
    initial_minimum_surface_clearance = min(
        min(
            values["minimum_fem_surface_gap"],
            values["minimum_lsdem_surface_gap"],
        )
        for values in initial_surface_clearances.values()
    )
    if args.contact_model == "barrier" and initial_minimum_surface_clearance <= 0.0:
        raise RuntimeError(
            "barrier packing starts with a crossed particle--wall surface: "
            f"minimum gap {initial_minimum_surface_clearance:.12g} m"
        )
    reference_initial_wall = _wall_observation(coupling.dem, wall_facet_ids)["positions"]
    if restart_checkpoint is not None:
        coupling.read_restart(restart_checkpoint)
        coupling.sims.current_print = 0
        coupling.dem.sims.current_print = 0
    target_field = ti.field(dtype=float, shape=())
    servo = _make_six_wall_servo(
        ti,
        coupling.dem.scene.wall,
        wall_facet_ids,
        target_field,
        args.servo_velocity_limit,
        args.servo_gain,
    )
    constant_velocity = _make_isotropic_wall_velocity(ti, coupling.dem.scene.wall, wall_facet_ids)
    ti.sync()
    setup_seconds = time.perf_counter() - started
    engine = coupling.enginer
    loading_fem_damping = float(args.fem_loading_damping)
    relaxation_fem_damping = float(args.fem_relaxation_damping)
    loading_dem_damping = float(args.dem_loading_local_damping)
    relaxation_dem_damping = float(args.dem_relaxation_local_damping)
    engine.fem_engine.damping = loading_fem_damping
    _set_dem_local_damping(coupling.dem, loading_dem_damping)
    effective_dt = float(coupling.sims.delta)
    compression_steps = int(
        math.ceil(
            args.maximum_isotropic_strain
            * min(base.WIDTH0, base.DEPTH0, base.HEIGHT0)
            / (2.0 * args.compression_velocity * effective_dt)
        )
    )
    maximum_steps = args.max_consolidation_steps + compression_steps
    total_time = max(maximum_steps * effective_dt, effective_dt)
    coupling.sims.time = total_time
    coupling.dem.sims.time = total_time
    initial_wall = reference_initial_wall
    if restart_checkpoint is not None:
        initial_wall = _restart_initial_wall(restart_checkpoint, initial_wall)
    initial_dimensions = (base.WIDTH0, base.DEPTH0, base.HEIGHT0)
    inertial_number = args.compression_velocity * math.sqrt(base.SOFT_DENSITY / base.YOUNG)

    if args.preflight:
        engine.reset_message()
        engine.update_verlet_tables(check_rebuild=True)
        engine.system_resolve(check_rebuild=True)
        constant_velocity(args.compression_velocity)
        engine.integration(update_diagnostics=False, check_jacobian=True)
        ti.sync()
        velocity = coupling.dem.scene.wall.v.to_numpy()[: int(coupling.dem.scene.wallNum[0])]
        report = {
            "schema_version": 1,
            "passed": bool(
                np.isfinite(velocity).all()
                and int(coupling.dem.scene.wallNum[0]) == 12
                and coupling.contactor.fixed_facet_walls is False
                and elements_per_soft_particle >= 1000
            ),
            "loading_protocol": _initialization_description(args),
            "wall_group_count": 6,
            "wall_facet_count": int(coupling.dem.scene.wallNum[0]),
            "initial_pressure": initial_pressure,
            "initial_pressure_over_reduced_young": args.initial_pressure_ratio,
            "compression_velocity": args.compression_velocity,
            "inertial_number": inertial_number,
            "effective_dt": effective_dt,
            "fem_loading_damping_per_second": loading_fem_damping,
            "fem_relaxation_damping_per_second": (relaxation_fem_damping),
            "dem_loading_local_damping": loading_dem_damping,
            "dem_relaxation_local_damping": relaxation_dem_damping,
            "compression_step_count": compression_steps,
            **(
                {}
                if args.paper_jamming_initialization
                else {"minimum_jamming_coordination": (args.minimum_jamming_coordination)}
            ),
            "initial_jamming_definition": _initial_jamming_definition(args),
            "fem_verlet_distance_multiplier": (args.fem_verlet_distance_multiplier),
            "minimum_initial_surface_clearance": (initial_minimum_surface_clearance),
            **_initial_bounding_clearance(packing_centers, packing_radii),
            "initial_packing": packing_metadata,
            "wall_velocity": velocity.tolist(),
            "soft_particle_count": base.SOFT_COUNT,
            "rigid_particle_count": base.RIGID_COUNT,
            "fem_nodes_per_particle": nodes_per_soft_particle,
            "fem_elements_per_particle": elements_per_soft_particle,
            "mesh_quality": mesh_quality,
            "setup_seconds": setup_seconds,
        }
        _write_json(output / "preflight.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2

    current_step = int(coupling.sims.current_step)
    history_rows = (
        _restart_csv_rows(restart_checkpoint, "history.csv", current_step) if restart_checkpoint is not None else []
    )
    consolidation_rows = (
        _restart_csv_rows(restart_checkpoint, "consolidation.csv", current_step)
        if restart_checkpoint is not None
        else []
    )
    compression_rows = [row for row in history_rows if row.get("phase") == "constant_velocity_compression"]
    resume_compression = bool(restart_checkpoint and compression_rows)
    completed_compression_steps = 0
    if resume_compression:
        completed_wall_displacement = float(compression_rows[-1]["top_wall_displacement"])
        completed_compression_steps = max(
            0,
            int(round(completed_wall_displacement / (args.compression_velocity * effective_dt))),
        )
    if completed_compression_steps > compression_steps:
        raise ValueError(
            "restart compression step exceeds the requested loading path: "
            f"{completed_compression_steps} > {compression_steps}"
        )
    coupling.checkpoint_phase = "constant_velocity_compression" if resume_compression else "initial_consolidation"
    coupling.save_data()
    saved_frame_count = 1
    saved_steps = {current_step}
    finite = True
    loop_started = time.perf_counter()

    target_field[None] = initial_pressure
    stable_checks = 0
    last_phi = None
    latest = None
    latest_phi_change = math.inf
    jammed = bool(args.direct_velocity_loading or resume_compression)
    stress_control_active = bool(args.direct_velocity_loading or resume_compression)
    fixed_volume_relaxation = False
    relaxation_checks = 0
    if restart_checkpoint is not None:
        latest = _mixed_observation(
            coupling,
            soft_ids,
            rigid_ids,
            body_ranges,
            wall_facet_ids,
            initial_pressure,
            reduced_young,
            initial_wall,
        )
        finite = finite and latest["finite"]
        last_phi = latest["solid_fraction"]
        if not stress_control_active and latest["coordination_number"] >= args.minimum_jamming_coordination:
            fixed_volume_relaxation = True
            engine.fem_engine.damping = relaxation_fem_damping
            _set_dem_local_damping(coupling.dem, relaxation_dem_damping)
    elif warm_start_checkpoint is not None:
        latest = _mixed_observation(
            coupling,
            soft_ids,
            rigid_ids,
            body_ranges,
            wall_facet_ids,
            initial_pressure,
            reduced_young,
            initial_wall,
        )
        finite = finite and latest["finite"]
        last_phi = latest["solid_fraction"]
        if latest["coordination_number"] >= args.minimum_jamming_coordination:
            fixed_volume_relaxation = True
            engine.fem_engine.damping = relaxation_fem_damping
            _set_dem_local_damping(coupling.dem, relaxation_dem_damping)
    if args.direct_velocity_loading and not resume_compression:
        latest = _mixed_observation(
            coupling,
            soft_ids,
            rigid_ids,
            body_ranges,
            wall_facet_ids,
            None,
            reduced_young,
            initial_wall,
        )
        finite = finite and latest["finite"]
        history_rows.append(
            _path_row(
                "initial_state",
                current_step,
                coupling.sims.current_time,
                latest,
                initial_dimensions,
            )
        )
    for local_step in range(
        1,
        (1 if args.direct_velocity_loading or resume_compression else args.max_consolidation_steps + 1),
    ):
        check_rebuild = current_step % args.neighbor_check_stride == 0
        engine.reset_message()
        engine.update_verlet_tables(check_rebuild=check_rebuild)
        engine.system_resolve(check_rebuild=check_rebuild)
        if stress_control_active:
            servo()
        elif fixed_volume_relaxation:
            constant_velocity(0.0)
        else:
            constant_velocity(args.compression_velocity)
        engine.integration(
            update_diagnostics=False,
            check_jacobian=(current_step % args.jacobian_check_stride == 0),
        )
        coupling.sims.current_time += effective_dt
        coupling.dem.sims.current_time += effective_dt
        engine.fem_engine.time = coupling.sims.current_time
        coupling.sims.current_step += 1
        coupling.dem.sims.current_step += 1
        engine.fem_engine.step_count += 1
        current_step += 1
        if local_step % args.equilibrium_check_stride != 0:
            continue
        ti.sync()
        latest = _mixed_observation(
            coupling,
            soft_ids,
            rigid_ids,
            body_ranges,
            wall_facet_ids,
            initial_pressure,
            reduced_young,
            initial_wall,
        )
        finite = finite and latest["finite"]
        if not latest["finite"]:
            break
        latest_phi_change = (
            math.inf
            if last_phi is None
            else abs(latest["solid_fraction"] - last_phi) / max(abs(latest["solid_fraction"]), 1.0e-30)
        )
        last_phi = latest["solid_fraction"]
        initialization_stage = (
            "stress_equilibration"
            if stress_control_active
            else ("fixed_volume_relaxation" if fixed_volume_relaxation else "network_formation")
        )
        coordination_ready = bool(latest["coordination_number"] >= args.minimum_jamming_coordination)
        pressure_ready = bool(
            abs(latest["mean_pressure"] - initial_pressure) / max(initial_pressure, 1.0e-30) <= args.pressure_tolerance
            if args.paper_jamming_initialization
            else latest["relative_wall_pressure_error"] <= args.pressure_tolerance
        )
        stable_now = bool(
            stress_control_active
            and local_step >= args.minimum_consolidation_steps
            and latest["finite"]
            and latest["minimum_jacobian"] > args.minimum_jacobian
            and pressure_ready
            and latest["kinetic_pressure_work_ratio"] <= args.kinetic_tolerance
            and coordination_ready
            and latest_phi_change <= args.solid_fraction_tolerance
        )
        stable_checks = stable_checks + 1 if stable_now else 0
        if args.paper_jamming_initialization:
            if not stress_control_active and not fixed_volume_relaxation and coordination_ready:
                fixed_volume_relaxation = True
                relaxation_checks = 0
                stable_checks = 0
                engine.fem_engine.damping = relaxation_fem_damping
                _set_dem_local_damping(coupling.dem, relaxation_dem_damping)
                constant_velocity(0.0)
            elif fixed_volume_relaxation:
                relaxed_now = bool(
                    coordination_ready
                    and latest["kinetic_pressure_work_ratio"] <= args.kinetic_tolerance
                    and latest_phi_change <= args.solid_fraction_tolerance
                )
                relaxation_checks = relaxation_checks + 1 if relaxed_now else 0
                if not coordination_ready:
                    fixed_volume_relaxation = False
                    relaxation_checks = 0
                    engine.fem_engine.damping = loading_fem_damping
                    _set_dem_local_damping(coupling.dem, loading_dem_damping)
                elif relaxation_checks >= args.required_stable_checks:
                    fixed_volume_relaxation = False
                    stress_control_active = True
                    relaxation_checks = 0
                    target_field[None] = initial_pressure
            elif stress_control_active and not coordination_ready:
                stress_control_active = False
                stable_checks = 0
                engine.fem_engine.damping = loading_fem_damping
                _set_dem_local_damping(coupling.dem, loading_dem_damping)
        elif not stress_control_active and coordination_ready:
            stress_control_active = True
            stable_checks = 0
            target_field[None] = initial_pressure
        consolidation_rows.append(
            {
                "initialization_stage": initialization_stage,
                "step": current_step,
                "time": coupling.sims.current_time,
                "pressure_over_reduced_young": latest["pressure_over_reduced_young"],
                "relative_wall_pressure_error": latest["relative_wall_pressure_error"],
                "relative_mean_pressure_error": abs(latest["mean_pressure"] - initial_pressure)
                / max(initial_pressure, 1.0e-30),
                "kinetic_pressure_work_ratio": latest["kinetic_pressure_work_ratio"],
                "kinetic_reference_pressure": latest["kinetic_reference_pressure"],
                "soft_kinetic_energy": latest["soft_kinetic_energy"],
                "rigid_translational_kinetic_energy": latest["rigid_translational_kinetic_energy"],
                "rigid_rotational_kinetic_energy": latest["rigid_rotational_kinetic_energy"],
                "relative_solid_fraction_change": latest_phi_change,
                "solid_fraction": latest["solid_fraction"],
                "coordination_number": latest["coordination_number"],
                "stable_check_count": stable_checks,
                "relaxation_check_count": relaxation_checks,
                "fem_damping_per_second": engine.fem_engine.damping,
                "dem_local_damping": (
                    relaxation_dem_damping if fixed_volume_relaxation or stress_control_active else loading_dem_damping
                ),
            }
        )
        print(
            json.dumps({"phase": initialization_stage, **consolidation_rows[-1]}),
            flush=True,
        )
        if stable_checks >= args.required_stable_checks:
            jammed = True
            break

    if latest is not None and jammed and not args.direct_velocity_loading and not resume_compression:
        history_rows.append(
            _path_row(
                "jammed_state",
                current_step,
                coupling.sims.current_time,
                latest,
                initial_dimensions,
            )
        )
        coupling.save_data()
        saved_frame_count += 1
        saved_steps.add(current_step)

    stop_reason = "not_started" if args.direct_velocity_loading else "initial_consolidation_not_equilibrated"
    compression_step_count = completed_compression_steps
    if jammed:
        engine.fem_engine.damping = loading_fem_damping
        _set_dem_local_damping(coupling.dem, loading_dem_damping)
        coupling.checkpoint_phase = "constant_velocity_compression"
        constant_velocity(args.compression_velocity)
        snapshot_steps = set(
            int(value)
            for value in np.linspace(
                1,
                compression_steps,
                args.snapshot_count,
                dtype=np.int64,
            )
        )
        for local_step in range(completed_compression_steps + 1, compression_steps + 1):
            check_rebuild = current_step % args.neighbor_check_stride == 0
            engine.reset_message()
            engine.update_verlet_tables(check_rebuild=check_rebuild)
            engine.system_resolve(check_rebuild=check_rebuild)
            sample_now = local_step % args.path_sample_stride == 0 or local_step == compression_steps
            engine.integration(
                update_diagnostics=False,
                check_jacobian=(current_step % args.jacobian_check_stride == 0 or sample_now),
            )
            coupling.sims.current_time += effective_dt
            coupling.dem.sims.current_time += effective_dt
            engine.fem_engine.time = coupling.sims.current_time
            coupling.sims.current_step += 1
            coupling.dem.sims.current_step += 1
            engine.fem_engine.step_count += 1
            current_step += 1
            compression_step_count = local_step

            if local_step in snapshot_steps:
                coupling.save_data()
                saved_frame_count += 1
                saved_steps.add(current_step)
            if not sample_now:
                continue
            ti.sync()
            latest = _mixed_observation(
                coupling,
                soft_ids,
                rigid_ids,
                body_ranges,
                wall_facet_ids,
                None,
                reduced_young,
                initial_wall,
            )
            history_rows.append(
                _path_row(
                    "constant_velocity_compression",
                    current_step,
                    coupling.sims.current_time,
                    latest,
                    initial_dimensions,
                )
            )
            finite = finite and latest["finite"]
            _write_csv(output / "history.csv", history_rows)
            print(
                json.dumps(
                    {
                        "phase": "constant_velocity_compression",
                        "step": current_step,
                        "solid_fraction": latest["solid_fraction"],
                        "pressure_over_reduced_young": latest["pressure_over_reduced_young"],
                        "coordination_number": latest["coordination_number"],
                    }
                ),
                flush=True,
            )
            if not latest["finite"] or latest["minimum_jacobian"] <= args.minimum_jacobian:
                stop_reason = "invalid_fem_state"
                break
            if latest["solid_fraction"] >= args.target_solid_fraction:
                stop_reason = "target_solid_fraction"
                break
            if latest["pressure_over_reduced_young"] >= args.maximum_pressure_ratio:
                stop_reason = "maximum_pressure_ratio"
                break
        else:
            stop_reason = "maximum_isotropic_strain"
        constant_velocity(0.0)

    if current_step not in saved_steps:
        coupling.save_data()
        saved_frame_count += 1
        saved_steps.add(current_step)

    ti.sync()
    loop_seconds = time.perf_counter() - loop_started
    _write_csv(output / "history.csv", history_rows)
    _write_csv(output / "consolidation.csv", consolidation_rows)
    required_output_families = [
        "fem_vtu_count",
        "coupled_contact_npz_count",
        "checkpoint_npz_count",
    ]
    if base.RIGID_COUNT > 0:
        required_output_families.extend(
            (
                "lsdem_surface_vtu_count",
                "lsdem_body_npz_count",
                "lsdem_surface_npz_count",
            )
        )
    output_evidence = collect_native_output(
        output / "native",
        saved_frame_count,
        tuple(required_output_families),
    )
    passed = bool(
        finite
        and jammed
        and len(history_rows) >= 2
        and stop_reason != "invalid_fem_state"
        and output_evidence["complete"]
    )
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "units": "SI",
        "seed": base.SEED,
        "loading_protocol": {
            "type": "velocity-controlled isotropic compaction",
            "controlled_walls": list(base.WALL_BODY_NAMES),
            "initialization": _initialization_description(args),
            "loading": "constant inward normal velocity on all six walls",
            "particle_wall_friction": base.WALL_FRICTION,
            "initial_pressure_over_reduced_young": args.initial_pressure_ratio,
            "compression_velocity": args.compression_velocity,
            "inertial_number": inertial_number,
            "effective_dt": effective_dt,
            "compression_step_count": compression_steps,
            **_initial_bounding_clearance(packing_centers, packing_radii),
            "initial_packing": packing_metadata,
            "minimum_initial_surface_clearance": (initial_minimum_surface_clearance),
            **(
                {}
                if args.paper_jamming_initialization
                else {"minimum_jamming_coordination": (args.minimum_jamming_coordination)}
            ),
            "initial_jamming_definition": _initial_jamming_definition(args),
        },
        "parameters": {
            "particle_count": base.PARTICLE_COUNT,
            "soft_particle_count": base.SOFT_COUNT,
            "rigid_particle_count": base.RIGID_COUNT,
            "requested_soft_percent": args.soft_percent,
            "soft_particle_shape": "sphere",
            "rigid_particle_shape": args.rigid_shape,
            "young_modulus": base.YOUNG,
            "poisson_ratio": base.POISSON,
            "reduced_young_modulus": reduced_young,
            "particle_particle_friction": base.FRICTION,
            "particle_wall_friction": base.WALL_FRICTION,
            "requested_dt": args.dt,
            "effective_dt": effective_dt,
            "servo_velocity_limit": args.servo_velocity_limit,
            "servo_gain": args.servo_gain,
            "fem_loading_damping_per_second": loading_fem_damping,
            "fem_relaxation_damping_per_second": (relaxation_fem_damping),
            "dem_loading_local_damping": loading_dem_damping,
            "dem_relaxation_local_damping": relaxation_dem_damping,
            "maximum_isotropic_strain": args.maximum_isotropic_strain,
            "target_solid_fraction": args.target_solid_fraction,
            "maximum_pressure_over_reduced_young": args.maximum_pressure_ratio,
            "fem_verlet_distance_multiplier": (args.fem_verlet_distance_multiplier),
        },
        "discretization": {
            "fem_nodes_per_soft_particle": nodes_per_soft_particle,
            "fem_elements_per_soft_particle": elements_per_soft_particle,
            "fem_reference_mesh_quality": mesh_quality,
        },
        "execution": {
            "arch": args.arch,
            "precision": args.default_fp,
            "command": [sys.executable, *sys.argv],
            "restart_checkpoint": (str(restart_checkpoint) if restart_checkpoint is not None else None),
            "warm_start_checkpoint": (str(warm_start_checkpoint) if warm_start_checkpoint is not None else None),
        },
    }
    metrics = {
        "schema_version": 1,
        "passed": passed,
        "finite_state": finite,
        "contact_network_formed": bool(
            latest is not None and latest["coordination_number"] >= args.minimum_jamming_coordination
        ),
        "initial_state_jammed": bool(jammed and not args.direct_velocity_loading),
        "direct_velocity_loading": bool(args.direct_velocity_loading),
        "paper_jamming_initialization": bool(args.paper_jamming_initialization),
        "stop_reason": stop_reason,
        "path_sample_count": len(history_rows),
        "compression_step_count": compression_step_count,
        "final_consolidation": (consolidation_rows[-1] if consolidation_rows else None),
        "soft_particle_count": base.SOFT_COUNT,
        "rigid_particle_count": base.RIGID_COUNT,
        "native_output": output_evidence,
        "final": history_rows[-1] if history_rows else None,
    }
    performance = {
        "schema_version": 1,
        "host": platform.node(),
        "python": platform.python_version(),
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "setup_seconds": setup_seconds,
        "simulation_loop_seconds": loop_seconds,
        "steps": current_step,
        "saved_frame_count": saved_frame_count,
    }
    state = {
        "packing_centers": packing_centers.tolist(),
        "packing_radii": packing_radii.tolist(),
        "packing_metadata": packing_metadata,
        "soft_packing_ids": soft_ids.tolist(),
        "rigid_packing_ids": rigid_ids.tolist(),
        "body_roles": body_roles,
        "initial_wall_positions": initial_wall,
    }
    for filename, payload in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("performance.json", performance),
        ("state.json", state),
    ):
        _write_json(output / filename, payload)
    print(json.dumps(metrics, indent=2))
    return 0 if passed else 2


def _build_rigid(gt, args, output: Path, total_time: float):
    use_barrier = args.contact_model == "barrier"
    centers, _, rigid_ids, orientations = base._packing(args.barrier_cutoff if use_barrier else None)
    wall_specs = base._facet_wall_specs(0.0, total_time)
    dem = gt.DEM(log=False)
    dem.set_configuration(
        domain=[0.47, 0.47, 0.47],
        scheme="LSDEM",
        engine="SymplecticEuler",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        track_energy=True,
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 2,
            "max_rigid_body_number": base.PARTICLE_COUNT,
            "max_rigid_template_number": 1,
            "levelset_grid_number": 100000,
            "surface_node_number": 512,
            "max_sphere_number": 0,
            "max_clump_number": 0,
            "max_plane_number": 0,
            "max_facet_number": 12,
            "body_coordination_number": 128,
            "wall_coordination_number": 12,
            "verlet_distance_multiplier": [0.1, 0.1],
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    for material_id in (0, 1):
        dem.add_attribute(
            materialID=material_id,
            attribute={
                "Density": base.RIGID_DENSITY,
                "ForceLocalDamping": args.dem_loading_local_damping,
                "TorqueLocalDamping": args.dem_loading_local_damping,
            },
        )
    template_name = "isotropic_irregular_grain"
    template = {
        "Name": template_name,
        "Object": gt.polyhedron(file=str(base.IRREGULAR_SURFACE)).grids(
            space=base.IRREGULAR_LEVELSET_SPACING, extent=3
        ),
        "WriteFile": False,
    }
    if args.rigid_shape == "sphere":
        sphere_object = gt.sphere(1.0)
        sphere_object.analytical_distance(sphere_object._distance)
        template_name = "isotropic_sphere"
        template = {
            "Name": template_name,
            "Object": sphere_object.grids(space=0.1, extent=3),
            "SurfaceResolution": 1728,
            "WriteFile": False,
        }
    dem.add_template(template)
    dem.create_body_batch(
        {
            "BodyType": "RigidBody",
            "Template": {
                "Name": template_name,
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoints": centers,
                "BoundingRadii": np.full(base.PARTICLE_COUNT, base.RADIUS),
                "CoordinatesAreMassCenters": False,
                "BodyOrientationsRadians": np.deg2rad(orientations),
                "InitialVelocity": [0.0, 0.0, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "FixMotion": ["Free", "Free", "Free"],
            },
        }
    )
    walls = []
    for wall_id, (_, vertices, normal, _) in enumerate(wall_specs):
        walls.append(
            {
                "WallID": wall_id,
                "WallType": "Facet",
                "WallShape": "Polygon",
                "MaterialID": 1,
                "WallVertice": {f"vertice{index + 1}": np.asarray(vertex) for index, vertex in enumerate(vertices)},
                "OuterNormal": np.asarray(normal),
                "InitialVelocity": [0.0, 0.0, 0.0],
            }
        )
    dem.add_wall(body=walls)
    wall_ids = dem.scene.wall.wallID.to_numpy()[: int(dem.scene.wallNum[0])]
    wall_facet_ids = {
        name: np.flatnonzero(wall_ids == wall_id).tolist() for wall_id, (name, _, _, _) in enumerate(wall_specs)
    }
    dem.choose_contact_model(
        "Barrier Model" if use_barrier else "Linear Model",
        "Barrier Model" if use_barrier else "Linear Model",
    )
    particle_property = (
        {
            "Stiffness": base.NORMAL_STIFFNESS,
            "NormalCutOff": args.barrier_cutoff,
            "StiffnessRatio": 1.0,
            "Friction": base.FRICTION,
            "NormalViscousDamping": 0.20,
            "TangentialViscousDamping": 0.10,
        }
        if use_barrier
        else {
            "NormalStiffness": base.NORMAL_STIFFNESS,
            "TangentialStiffness": base.TANGENTIAL_STIFFNESS,
            "Friction": base.FRICTION,
            "NormalViscousDamping": 0.20,
            "TangentialViscousDamping": 0.10,
        }
    )
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property=particle_property,
        dType="particle-particle",
    )
    dem.add_property(
        materialID1=0,
        materialID2=1,
        property={**particle_property, "Friction": base.WALL_FRICTION},
        dType="particle-wall",
    )
    dem.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": total_time,
            "SaveInterval": total_time,
            "SavePath": str(output / "native"),
        },
        log=False,
    )
    dem.select_save_data(
        particle=True,
        surface=True,
        wall=True,
        particle_particle_contact=True,
        particle_wall_contact=True,
    )
    dem.add_essentials()
    engine = dem.enginer
    engine.pre_calculation(dem.sims, dem.scene, dem.contactor.neighbor)
    dem.check_critical_timestep()
    return dem, engine, centers, rigid_ids, wall_facet_ids


def _rigid_observation(
    dem,
    rigid_ids,
    wall_facet_ids,
    target_pressure,
    reduced_young,
    initial_wall,
):
    wall = _wall_observation(dem, wall_facet_ids)
    rigid_mass = dem.scene.rigid.m.to_numpy()[: base.PARTICLE_COUNT]
    rigid_volume = float(np.sum(rigid_mass) / base.RIGID_DENSITY)
    rigid_translational, rigid_rotational, rigid_finite = _rigid_kinetic(dem, base.PARTICLE_COUNT)
    pairs = base._rigid_particle_contact_pairs(dem, rigid_ids)
    kinetic = rigid_translational + rigid_rotational
    pressure_scale = float(target_pressure) if target_pressure is not None else wall["mean_pressure"]
    wall_error = max(
        abs(value - pressure_scale) / max(abs(pressure_scale), 1.0e-30) for value in wall["wall_stress"].values()
    )
    kinetic_reference_pressure = max(
        abs(float(wall["mean_pressure"])),
        abs(float(pressure_scale)),
        1.0e-30,
    )
    return {
        **wall,
        "soft_solid_volume": 0.0,
        "rigid_solid_volume": rigid_volume,
        "solid_fraction": rigid_volume / max(wall["box_volume"], 1.0e-30),
        "particle_contact_pair_count": len(pairs),
        "coordination_number": 2.0 * len(pairs) / base.PARTICLE_COUNT,
        "soft_kinetic_energy": 0.0,
        "rigid_translational_kinetic_energy": rigid_translational,
        "rigid_rotational_kinetic_energy": rigid_rotational,
        "kinetic_energy": kinetic,
        "kinetic_pressure_work_ratio": kinetic / max(kinetic_reference_pressure * wall["box_volume"], 1.0e-30),
        "kinetic_reference_pressure": kinetic_reference_pressure,
        "relative_wall_pressure_error": wall_error,
        "minimum_jacobian": 1.0,
        "pressure_over_reduced_young": wall["mean_pressure"] / reduced_young,
        "top_wall_displacement": initial_wall["top"] - wall["positions"]["top"],
        "axial_stress": wall["sigma_z"],
        "finite": bool(rigid_finite and math.isfinite(wall["mean_pressure"]) and math.isfinite(rigid_volume)),
    }


def _run_rigid(gt, ti, args, output: Path, reduced_young) -> int:
    initial_pressure = args.initial_pressure_ratio * reduced_young
    compression_steps = int(
        math.ceil(
            args.maximum_isotropic_strain
            * min(base.WIDTH0, base.DEPTH0, base.HEIGHT0)
            / (2.0 * args.compression_velocity * args.dt)
        )
    )
    maximum_steps = args.max_consolidation_steps + compression_steps
    total_time = max(maximum_steps * args.dt, args.dt)
    base._clear_fresh_run_output(output)
    started = time.perf_counter()
    dem, engine, centers, rigid_ids, wall_facet_ids = _build_rigid(gt, args, output, total_time)
    initial_minimum_surface_clearance = _minimum_lsdem_wall_clearance(dem, wall_facet_ids)
    if args.contact_model == "barrier" and initial_minimum_surface_clearance <= 0.0:
        raise RuntimeError(
            "barrier packing starts with a crossed particle--wall surface: "
            f"minimum gap {initial_minimum_surface_clearance:.12g} m"
        )
    target_field = ti.field(dtype=float, shape=())
    servo = _make_six_wall_servo(
        ti,
        dem.scene.wall,
        wall_facet_ids,
        target_field,
        args.servo_velocity_limit,
        args.servo_gain,
    )
    constant_velocity = _make_isotropic_wall_velocity(ti, dem.scene.wall, wall_facet_ids)
    ti.sync()
    setup_seconds = time.perf_counter() - started
    loading_dem_damping = float(args.dem_loading_local_damping)
    relaxation_dem_damping = float(args.dem_relaxation_local_damping)
    _set_dem_local_damping(dem, loading_dem_damping)
    effective_dt = float(dem.sims.delta)
    compression_steps = int(
        math.ceil(
            args.maximum_isotropic_strain
            * min(base.WIDTH0, base.DEPTH0, base.HEIGHT0)
            / (2.0 * args.compression_velocity * effective_dt)
        )
    )
    maximum_steps = args.max_consolidation_steps + compression_steps
    total_time = max(maximum_steps * effective_dt, effective_dt)
    dem.sims.time = total_time
    initial_wall = _wall_observation(dem, wall_facet_ids)["positions"]
    initial_dimensions = (base.WIDTH0, base.DEPTH0, base.HEIGHT0)
    inertial_number = args.compression_velocity * math.sqrt(base.RIGID_DENSITY / base.YOUNG)

    def resolve(check_rebuild):
        engine.reset_wall_message(dem.scene)
        engine.reset_particle_message(dem.scene)
        engine.reset_contact_energy()
        dem.contactor.neighbor.accumulate_point_relative_displacement(dem.scene)
        if check_rebuild:
            if engine.is_verlet_update(engine.limit1) == 1:
                engine.update_LSDEM_verlet_table1(dem.sims, dem.scene, dem.contactor.neighbor)
                engine.update_LSDEM_verlet_table2(dem.sims, dem.scene, dem.contactor.neighbor)
            elif engine.is_verlet_update_point(engine.limit2) == 1:
                engine.update_LSDEM_verlet_table2(dem.sims, dem.scene, dem.contactor.neighbor)
        engine.system_resolve(dem.sims, dem.scene, dem.contactor.neighbor)

    if args.preflight:
        resolve(True)
        constant_velocity(args.compression_velocity)
        engine.integration(dem.sims, dem.scene, dem.contactor.neighbor)
        ti.sync()
        velocity = dem.scene.wall.v.to_numpy()[: int(dem.scene.wallNum[0])]
        report = {
            "schema_version": 1,
            "passed": bool(np.isfinite(velocity).all() and int(dem.scene.wallNum[0]) == 12),
            "loading_protocol": _initialization_description(args),
            "wall_group_count": 6,
            "wall_facet_count": int(dem.scene.wallNum[0]),
            "initial_pressure": initial_pressure,
            "initial_pressure_over_reduced_young": args.initial_pressure_ratio,
            "compression_velocity": args.compression_velocity,
            "inertial_number": inertial_number,
            "initial_jamming_definition": _initial_jamming_definition(args),
            "effective_dt": effective_dt,
            "compression_step_count": compression_steps,
            "dem_loading_local_damping": loading_dem_damping,
            "dem_relaxation_local_damping": relaxation_dem_damping,
            "minimum_initial_surface_clearance": (initial_minimum_surface_clearance),
            **_initial_bounding_clearance(centers),
            "wall_velocity": velocity.tolist(),
            "soft_particle_count": 0,
            "rigid_particle_count": base.PARTICLE_COUNT,
            "setup_seconds": setup_seconds,
        }
        _write_json(output / "preflight.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2

    dem.save_data()
    saved_frame_count = 1
    saved_steps = {0}
    history_rows = []
    consolidation_rows = []
    current_step = 0
    finite = True
    loop_started = time.perf_counter()

    target_field[None] = initial_pressure
    stable_checks = 0
    last_phi = None
    latest = None
    latest_phi_change = math.inf
    jammed = bool(args.direct_velocity_loading)
    stress_control_active = bool(args.direct_velocity_loading)
    fixed_volume_relaxation = False
    relaxation_checks = 0
    if args.direct_velocity_loading:
        latest = _rigid_observation(
            dem,
            rigid_ids,
            wall_facet_ids,
            None,
            reduced_young,
            initial_wall,
        )
        finite = finite and latest["finite"]
        history_rows.append(
            _path_row(
                "initial_state",
                current_step,
                dem.sims.current_time,
                latest,
                initial_dimensions,
            )
        )
    for local_step in range(
        1,
        1 if args.direct_velocity_loading else args.max_consolidation_steps + 1,
    ):
        resolve(current_step % args.neighbor_check_stride == 0)
        if stress_control_active:
            servo()
        elif fixed_volume_relaxation:
            constant_velocity(0.0)
        else:
            constant_velocity(args.compression_velocity)
        engine.integration(dem.sims, dem.scene, dem.contactor.neighbor)
        dem.sims.current_time += effective_dt
        dem.sims.current_step += 1
        current_step += 1
        if local_step % args.equilibrium_check_stride != 0:
            continue
        ti.sync()
        latest = _rigid_observation(
            dem,
            rigid_ids,
            wall_facet_ids,
            initial_pressure,
            reduced_young,
            initial_wall,
        )
        finite = finite and latest["finite"]
        if not latest["finite"]:
            break
        latest_phi_change = (
            math.inf
            if last_phi is None
            else abs(latest["solid_fraction"] - last_phi) / max(abs(latest["solid_fraction"]), 1.0e-30)
        )
        last_phi = latest["solid_fraction"]
        initialization_stage = (
            "stress_equilibration"
            if stress_control_active
            else ("fixed_volume_relaxation" if fixed_volume_relaxation else "network_formation")
        )
        coordination_ready = bool(latest["coordination_number"] >= args.minimum_jamming_coordination)
        pressure_ready = bool(
            abs(latest["mean_pressure"] - initial_pressure) / max(initial_pressure, 1.0e-30) <= args.pressure_tolerance
            if args.paper_jamming_initialization
            else latest["relative_wall_pressure_error"] <= args.pressure_tolerance
        )
        stable_now = bool(
            stress_control_active
            and local_step >= args.minimum_consolidation_steps
            and latest["finite"]
            and pressure_ready
            and latest["kinetic_pressure_work_ratio"] <= args.kinetic_tolerance
            and coordination_ready
            and latest_phi_change <= args.solid_fraction_tolerance
        )
        stable_checks = stable_checks + 1 if stable_now else 0
        if args.paper_jamming_initialization:
            if not stress_control_active and not fixed_volume_relaxation and coordination_ready:
                fixed_volume_relaxation = True
                relaxation_checks = 0
                stable_checks = 0
                _set_dem_local_damping(dem, relaxation_dem_damping)
                constant_velocity(0.0)
            elif fixed_volume_relaxation:
                relaxed_now = bool(
                    coordination_ready
                    and latest["kinetic_pressure_work_ratio"] <= args.kinetic_tolerance
                    and latest_phi_change <= args.solid_fraction_tolerance
                )
                relaxation_checks = relaxation_checks + 1 if relaxed_now else 0
                if not coordination_ready:
                    fixed_volume_relaxation = False
                    relaxation_checks = 0
                    _set_dem_local_damping(dem, loading_dem_damping)
                elif relaxation_checks >= args.required_stable_checks:
                    fixed_volume_relaxation = False
                    stress_control_active = True
                    relaxation_checks = 0
                    target_field[None] = initial_pressure
            elif stress_control_active and not coordination_ready:
                stress_control_active = False
                stable_checks = 0
                _set_dem_local_damping(dem, loading_dem_damping)
        elif not stress_control_active and coordination_ready:
            stress_control_active = True
            stable_checks = 0
            target_field[None] = initial_pressure
        consolidation_rows.append(
            {
                "initialization_stage": initialization_stage,
                "step": current_step,
                "time": dem.sims.current_time,
                "pressure_over_reduced_young": latest["pressure_over_reduced_young"],
                "relative_wall_pressure_error": latest["relative_wall_pressure_error"],
                "relative_mean_pressure_error": abs(latest["mean_pressure"] - initial_pressure)
                / max(initial_pressure, 1.0e-30),
                "kinetic_pressure_work_ratio": latest["kinetic_pressure_work_ratio"],
                "kinetic_reference_pressure": latest["kinetic_reference_pressure"],
                "rigid_translational_kinetic_energy": latest["rigid_translational_kinetic_energy"],
                "rigid_rotational_kinetic_energy": latest["rigid_rotational_kinetic_energy"],
                "relative_solid_fraction_change": latest_phi_change,
                "solid_fraction": latest["solid_fraction"],
                "coordination_number": latest["coordination_number"],
                "stable_check_count": stable_checks,
                "relaxation_check_count": relaxation_checks,
                "dem_local_damping": (
                    relaxation_dem_damping if fixed_volume_relaxation or stress_control_active else loading_dem_damping
                ),
            }
        )
        print(
            json.dumps({"phase": initialization_stage, **consolidation_rows[-1]}),
            flush=True,
        )
        if stable_checks >= args.required_stable_checks:
            jammed = True
            break

    if latest is not None and jammed and not args.direct_velocity_loading:
        history_rows.append(
            _path_row(
                "jammed_state",
                current_step,
                dem.sims.current_time,
                latest,
                initial_dimensions,
            )
        )
        dem.save_data()
        saved_frame_count += 1
        saved_steps.add(current_step)

    stop_reason = "not_started" if args.direct_velocity_loading else "initial_consolidation_not_equilibrated"
    compression_step_count = 0
    if jammed:
        _set_dem_local_damping(dem, loading_dem_damping)
        constant_velocity(args.compression_velocity)
        snapshot_steps = set(
            int(value)
            for value in np.linspace(
                1,
                compression_steps,
                args.snapshot_count,
                dtype=np.int64,
            )
        )
        for local_step in range(1, compression_steps + 1):
            resolve(current_step % args.neighbor_check_stride == 0)
            engine.integration(dem.sims, dem.scene, dem.contactor.neighbor)
            dem.sims.current_time += effective_dt
            dem.sims.current_step += 1
            current_step += 1
            compression_step_count = local_step
            if local_step in snapshot_steps:
                dem.save_data()
                saved_frame_count += 1
                saved_steps.add(current_step)
            sample_now = local_step % args.path_sample_stride == 0 or local_step == compression_steps
            if not sample_now:
                continue
            ti.sync()
            latest = _rigid_observation(
                dem,
                rigid_ids,
                wall_facet_ids,
                None,
                reduced_young,
                initial_wall,
            )
            history_rows.append(
                _path_row(
                    "constant_velocity_compression",
                    current_step,
                    dem.sims.current_time,
                    latest,
                    initial_dimensions,
                )
            )
            finite = finite and latest["finite"]
            _write_csv(output / "history.csv", history_rows)
            print(
                json.dumps(
                    {
                        "phase": "constant_velocity_compression",
                        "step": current_step,
                        "solid_fraction": latest["solid_fraction"],
                        "pressure_over_reduced_young": latest["pressure_over_reduced_young"],
                        "coordination_number": latest["coordination_number"],
                    }
                ),
                flush=True,
            )
            if not latest["finite"]:
                stop_reason = "invalid_rigid_state"
                break
            if latest["solid_fraction"] >= args.target_solid_fraction:
                stop_reason = "target_solid_fraction"
                break
            if latest["pressure_over_reduced_young"] >= args.maximum_pressure_ratio:
                stop_reason = "maximum_pressure_ratio"
                break
        else:
            stop_reason = "maximum_isotropic_strain"
        constant_velocity(0.0)

    if current_step not in saved_steps:
        dem.save_data()
        saved_frame_count += 1
        saved_steps.add(current_step)

    ti.sync()
    loop_seconds = time.perf_counter() - loop_started
    _write_csv(output / "history.csv", history_rows)
    _write_csv(output / "consolidation.csv", consolidation_rows)
    passed = bool(finite and jammed and len(history_rows) >= 2 and stop_reason != "invalid_rigid_state")
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "units": "SI",
        "seed": base.SEED,
        "loading_protocol": {
            "type": "velocity-controlled isotropic compaction",
            "controlled_walls": list(base.WALL_BODY_NAMES),
            "initialization": _initialization_description(args),
            "loading": "constant inward normal velocity on all six walls",
            "initial_pressure_over_reduced_young": args.initial_pressure_ratio,
            "compression_velocity": args.compression_velocity,
            "inertial_number": inertial_number,
            "initial_jamming_definition": _initial_jamming_definition(args),
        },
        "parameters": {
            "particle_count": base.PARTICLE_COUNT,
            "soft_particle_count": 0,
            "rigid_particle_count": base.PARTICLE_COUNT,
            "requested_soft_percent": args.soft_percent,
            "soft_particle_shape": "sphere",
            "rigid_particle_shape": args.rigid_shape,
            "young_modulus_reference": base.YOUNG,
            "reduced_young_modulus_reference": reduced_young,
            "particle_particle_friction": base.FRICTION,
            "particle_wall_friction": base.WALL_FRICTION,
            "requested_dt": args.dt,
            "effective_dt": effective_dt,
            "dem_loading_local_damping": loading_dem_damping,
            "dem_relaxation_local_damping": relaxation_dem_damping,
            "maximum_isotropic_strain": args.maximum_isotropic_strain,
            "target_solid_fraction": args.target_solid_fraction,
            "maximum_pressure_over_reduced_young": args.maximum_pressure_ratio,
            "minimum_jamming_coordination": (args.minimum_jamming_coordination),
        },
        "execution": {
            "arch": args.arch,
            "precision": args.default_fp,
            "command": [sys.executable, *sys.argv],
        },
    }
    metrics = {
        "schema_version": 1,
        "passed": passed,
        "finite_state": finite,
        "contact_network_formed": bool(
            latest is not None and latest["coordination_number"] >= args.minimum_jamming_coordination
        ),
        "initial_state_jammed": bool(jammed and not args.direct_velocity_loading),
        "direct_velocity_loading": bool(args.direct_velocity_loading),
        "paper_jamming_initialization": bool(args.paper_jamming_initialization),
        "stop_reason": stop_reason,
        "path_sample_count": len(history_rows),
        "compression_step_count": compression_step_count,
        "final_consolidation": (consolidation_rows[-1] if consolidation_rows else None),
        "soft_particle_count": 0,
        "rigid_particle_count": base.PARTICLE_COUNT,
        "final": history_rows[-1] if history_rows else None,
    }
    performance = {
        "schema_version": 1,
        "host": platform.node(),
        "python": platform.python_version(),
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "setup_seconds": setup_seconds,
        "simulation_loop_seconds": loop_seconds,
        "steps": current_step,
        "saved_frame_count": saved_frame_count,
    }
    state = {
        "packing_centers": centers.tolist(),
        "soft_packing_ids": [],
        "rigid_packing_ids": rigid_ids.tolist(),
        "initial_wall_positions": initial_wall,
    }
    for filename, payload in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("performance.json", performance),
        ("state.json", state),
    ):
        _write_json(output / filename, payload)
    print(json.dumps(metrics, indent=2))
    return 0 if passed else 2


def main() -> int:
    paper_defaults = Path(sys.argv[0]).resolve() == Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--default-fp", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--soft-percent", type=float, default=50.0)
    parser.add_argument(
        "--rigid-shape",
        choices=("sphere", "irregular"),
        default="sphere",
    )
    parser.add_argument(
        "--randomized-packing",
        action=argparse.BooleanOptionalAction,
        default=paper_defaults,
        help=("use a deterministic random, weakly polydisperse, " "non-overlapping initial packing"),
    )
    parser.add_argument("--packing-seed", type=int, default=20260831)
    parser.add_argument(
        "--diameter-polydispersity",
        type=float,
        default=0.20,
        help="relative half-width of the uniform diameter distribution",
    )
    parser.add_argument("--dt", type=float, default=base.DT)
    parser.add_argument(
        "--contact-model",
        choices=("linear", "barrier"),
        default="barrier" if paper_defaults else "linear",
    )
    parser.add_argument(
        "--barrier-cutoff",
        type=float,
        default=base.BARRIER_CUTOFF,
    )
    parser.add_argument(
        "--fem-verlet-distance-multiplier",
        type=float,
        default=0.20 if paper_defaults else 0.12,
    )
    parser.add_argument("--initial-pressure-ratio", type=float, default=1.0e-4)
    parser.add_argument("--maximum-pressure-ratio", type=float, default=1.0)
    parser.add_argument("--compression-velocity", type=float, default=5.0e-3)
    parser.add_argument(
        "--maximum-isotropic-strain",
        type=float,
        default=0.50 if paper_defaults else 0.25,
    )
    parser.add_argument("--target-solid-fraction", type=float, default=0.95)
    parser.add_argument(
        "--servo-velocity-limit",
        type=float,
        default=2.0e-3 if paper_defaults else 1.0e-2,
    )
    parser.add_argument("--servo-gain", type=float, default=1.0 if paper_defaults else 0.35)
    parser.add_argument("--fem-loading-damping", type=float, default=0.04)
    parser.add_argument(
        "--fem-relaxation-damping",
        type=float,
        default=_default_fem_relaxation_damping(),
    )
    parser.add_argument("--dem-loading-local-damping", type=float, default=0.08)
    parser.add_argument("--dem-relaxation-local-damping", type=float, default=0.70)
    parser.add_argument(
        "--equilibrium-check-stride",
        type=int,
        default=2500 if paper_defaults else 250,
    )
    parser.add_argument("--path-sample-stride", type=int, default=2500)
    parser.add_argument("--snapshot-count", type=int, default=10 if paper_defaults else 8)
    parser.add_argument("--neighbor-check-stride", type=int, default=10)
    parser.add_argument("--jacobian-check-stride", type=int, default=1000)
    parser.add_argument(
        "--minimum-consolidation-steps",
        type=int,
        default=2500 if paper_defaults else 2000,
    )
    parser.add_argument(
        "--max-consolidation-steps",
        type=int,
        default=3500000 if paper_defaults else 150000,
    )
    parser.add_argument(
        "--required-stable-checks",
        type=int,
        default=3 if paper_defaults else 5,
    )
    parser.add_argument(
        "--pressure-tolerance",
        type=float,
        default=5.0e-2 if paper_defaults else 2.0e-2,
    )
    parser.add_argument("--kinetic-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--solid-fraction-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--minimum-jacobian", type=float, default=0.10)
    parser.add_argument(
        "--minimum-jamming-coordination",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--direct-velocity-loading",
        action="store_true",
        help=(
            "start six-wall constant-velocity isotropic compression directly "
            "from the generated packing instead of requiring a stress-servo "
            "equilibrium state"
        ),
    )
    parser.add_argument(
        "--paper-jamming-initialization",
        action=argparse.BooleanOptionalAction,
        default=paper_defaults,
        help=(
            "prepare the jammed reference state under a small isotropic "
            "stress and accept it when the mean stress reaches its target "
            "and the relative solid-fraction change remains below 0.01%%"
        ),
    )
    parser.add_argument(
        "--restart-checkpoint",
        type=Path,
        help=(
            "continue a mixed FEM--LSDEM consolidation from an exact coupled "
            "checkpoint after rebuilding the same model"
        ),
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        help=(
            "transfer FEM positions, rigid centers, and wall translations "
            "from a coupled checkpoint onto a rebuilt contact surface, "
            "then re-equilibrate before loading"
        ),
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    if paper_defaults and args.soft_percent == 100.0:
        if "--compression-velocity" not in sys.argv[1:]:
            args.compression_velocity = 2.0e-2
        if "--target-solid-fraction" not in sys.argv[1:]:
            args.target_solid_fraction = 0.99
        if "--maximum-pressure-ratio" not in sys.argv[1:]:
            args.maximum_pressure_ratio = 1.50
        if "--output" not in sys.argv[1:]:
            args.output = DEFAULT_OUTPUT.parent / "soft_100"

    if not 0.0 <= args.soft_percent <= 100.0:
        parser.error("--soft-percent must lie in [0, 100]")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.barrier_cutoff <= 0.0:
        parser.error("--barrier-cutoff must be positive")
    if args.fem_verlet_distance_multiplier <= 0.0:
        parser.error("--fem-verlet-distance-multiplier must be positive")
    if not 0.0 <= args.diameter_polydispersity < 0.5:
        parser.error("--diameter-polydispersity must lie in [0, 0.5)")
    if args.fem_loading_damping < 0.0:
        parser.error("--fem-loading-damping cannot be negative")
    if args.fem_relaxation_damping < 0.0:
        parser.error("--fem-relaxation-damping cannot be negative")
    for name in (
        "dem_loading_local_damping",
        "dem_relaxation_local_damping",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if args.minimum_jamming_coordination < 0.0:
        parser.error("--minimum-jamming-coordination cannot be negative")
    if not 0.0 < args.initial_pressure_ratio < args.maximum_pressure_ratio:
        parser.error("pressure-ratio bounds must be positive and increasing")
    if args.compression_velocity <= 0.0:
        parser.error("--compression-velocity must be positive")
    if not 0.0 < args.maximum_isotropic_strain <= 0.5:
        parser.error("--maximum-isotropic-strain must lie in (0, 0.5]")
    if not 0.0 < args.target_solid_fraction <= 1.0:
        parser.error("--target-solid-fraction must lie in (0, 1]")
    for name in (
        "equilibrium_check_stride",
        "path_sample_stride",
        "snapshot_count",
        "neighbor_check_stride",
        "jacobian_check_stride",
        "minimum_consolidation_steps",
        "max_consolidation_steps",
        "required_stable_checks",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.minimum_consolidation_steps > args.max_consolidation_steps:
        parser.error("minimum consolidation steps cannot exceed the maximum")
    if args.direct_velocity_loading and args.paper_jamming_initialization:
        parser.error("--direct-velocity-loading and --paper-jamming-initialization " "are mutually exclusive")
    if args.restart_checkpoint is not None and args.preflight:
        parser.error("--restart-checkpoint cannot be combined with --preflight")
    if args.restart_checkpoint is not None and args.warm_start_checkpoint is not None:
        parser.error("--restart-checkpoint and --warm-start-checkpoint are mutually exclusive")

    soft_count, _ = _set_composition(args.soft_percent)
    if args.restart_checkpoint is not None and soft_count == 0:
        parser.error("coupled checkpoints require at least one FEM particle")
    if args.warm_start_checkpoint is not None and soft_count == 0:
        parser.error("coupled warm starts require at least one FEM particle")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["GEOTAICHI_REAL_DTYPE"] = args.default_fp

    import taichi as ti
    import geotaichi as gt

    gt.init(
        arch=args.arch,
        default_fp=args.default_fp,
        log=False,
        debug=False,
        offline_cache=False,
    )
    reduced_young = base.YOUNG / (2.0 * (1.0 - base.POISSON**2))
    if soft_count == 0:
        return _run_rigid(gt, ti, args, output, reduced_young)
    return _run_mixed(gt, ti, args, output, reduced_young)


if __name__ == "__main__":
    raise SystemExit(main())
