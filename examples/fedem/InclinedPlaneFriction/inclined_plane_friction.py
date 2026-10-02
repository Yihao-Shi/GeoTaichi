#!/usr/bin/env python3
"""FEM sphere/cube friction verification on a fixed LSDEM incline."""

from __future__ import annotations

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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.fedem.InclinedPlaneFriction.mesh_reference import (
    euler_rotation,
    place_reference,
    tetrahedralize_primitive_reference,
)
from examples.fedem.InclinedPlaneFriction.output_audit import collect_native_output

PLATE_OBJ = REPO_ROOT / "assets/mesh/FEDEM/incline_plate.obj"
PROMPT = (
    "Place a deformable FEM sphere or cube on a fixed level-set DEM plane "
    "inclined at 45 degrees. Release the particle after seating and compare "
    "its motion for several friction coefficients with the sliding or "
    "rolling solution."
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "OutputData"

RADIUS = 0.04
CUBE_SIDE = 2.0 * RADIUS
DENSITY = 2500.0
YOUNG = 2.0e6
POISSON = 0.30
ANGLE_DEGREES = 45.0
GRAVITY = 9.81
NORMAL_STIFFNESS = 4.0e7
TANGENTIAL_STIFFNESS = 2.0e7
DT = 1.0e-5
SIMULATION_TIME = 0.25
OUTPUT_INTERVAL = 0.025
INITIAL_GAP = 2.0e-5
SETTLING_RAMP_TIME = 0.03
MINIMUM_SETTLING_TIME = 0.08
MAXIMUM_SETTLING_TIME = 0.60
SETTLING_REACTION_TOLERANCE = 0.05
SETTLING_NORMAL_SPEED_TOLERANCE = 5.0e-3
SETTLING_CONSECUTIVE_SAMPLES = 5
SETTLING_FEM_DAMPING = 250.0
MINIMUM_FEM_ELEMENTS = 1000
MINIMUM_TETRA_MEAN_RATIO = 0.05
CASES = ("sphere", "cube")


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _prompt() -> str:
    return PROMPT


def _frame():
    rotation = euler_rotation([0.0, ANGLE_DEGREES, 0.0])
    slope = rotation[:, 0]
    transverse = rotation[:, 1]
    normal = rotation[:, 2]
    plate_center = np.array([0.50, 0.30, 0.34], dtype=np.float64)
    return rotation, slope, transverse, normal, plate_center


def _soft_mesh(shape: str):
    reference, quality = tetrahedralize_primitive_reference(
        shape,
        minimum_element_count=MINIMUM_FEM_ELEMENTS,
        minimum_mean_ratio=MINIMUM_TETRA_MEAN_RATIO,
    )
    _, slope, _, normal, plate_center = _frame()
    bounding_radius = RADIUS if shape == "sphere" else 0.5 * math.sqrt(3.0) * CUBE_SIDE
    # The normalized cube has an orientation-dependent normal support.  Use
    # its actual oriented nodes rather than a nominal radius for placement.
    centered = place_reference(
        reference,
        [0.0, 0.0, 0.0],
        bounding_radius,
        [0.0, ANGLE_DEGREES, 0.0],
        name=f"friction_fem_{shape}",
    )
    normal_support = float(np.max(centered.points @ normal))
    center = plate_center - 0.22 * slope + (0.020 + normal_support + INITIAL_GAP) * normal
    centered.points += center[None, :]
    centered.set_rest_shape(centered.points.copy())
    return centered, quality, center


def _build_case(gt, output: Path, shape: str, dt: float, friction: float):
    rotation, _, _, _, plate_center = _frame()
    dem = gt.DEM(log=False)
    dem.set_configuration(
        domain=[1.0, 0.60, 0.80],
        scheme="LSDEM",
        engine="SymplecticEuler",
        search="LinkedCell",
        gravity=[0.0, 0.0, -GRAVITY],
        track_energy=True,
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_rigid_body_number": 1,
            "levelset_grid_number": 131072,
            "surface_node_number": 32,
            "max_sphere_number": 0,
            "max_clump_number": 0,
            "max_plane_number": 0,
            "body_coordination_number": 4,
            "wall_coordination_number": 1,
            "verlet_distance_multiplier": [0.1, 0.1],
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.add_attribute(
        materialID=0,
        attribute={
            "Density": DENSITY,
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    dem.add_template(
        {
            "Name": "fixed_lsdem_incline",
            "Object": gt.polyhedron(file=str(PLATE_OBJ)).grids(space=0.01, extent=2),
            "WriteFile": False,
        }
    )
    dem.create_body(
        {
            "BodyType": "RigidBody",
            "Template": [
                {
                    "Name": "fixed_lsdem_incline",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": plate_center.tolist(),
                    "ScaleFactor": 1.0,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "FixMotion": ["Fix", "Fix", "Fix"],
                    "BodyOrientation": [0.0, ANGLE_DEGREES, 0.0],
                }
            ],
        }
    )
    dem.choose_contact_model(None, None)
    dem.select_save_data(particle=True, surface=True)

    fem = gt.FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit", log=False)
    soft_mesh, mesh_quality, initial_center = _soft_mesh(shape)
    mesh = fem.add_soft_particle(soft_mesh)
    fem.add_material(
        "NeoHookean",
        density=DENSITY,
        young_modulus=YOUNG,
        poisson_ratio=POISSON,
    )
    initial_velocity = np.zeros_like(mesh.points)

    coupling = gt.FEDEM(dem=dem, fem=fem, log=False)
    coupling.set_configuration(
        domain=[1.0, 0.60, 0.80],
        gravity=[0.0, 0.0, -GRAVITY],
        search="BVH",
        contact_work_mode="Explicit",
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": dt,
            "SimulationTime": SIMULATION_TIME,
            "SaveInterval": OUTPUT_INTERVAL,
            "SavePath": str(output / "native"),
            "initial_velocity": initial_velocity,
            "damping": 0.0,
        },
        log=False,
    )
    coupling.select_save_data(contact=True, checkpoint=True)
    coupling.add_surface(body_ids=[0])
    coupling.memory_allocate(
        {
            "contact_coordination_number": 32,
            "max_contact_pairs": 4096,
            "max_levelset_cell_pairs": 65536,
            "verlet_distance_multiplier": 0.1,
        }
    )
    coupling.choose_contact_model("Linear")
    coupling.add_property(
        DEMmaterial=0,
        FEMbody=0,
        property={
            "NormalStiffness": NORMAL_STIFFNESS,
            "TangentialStiffness": TANGENTIAL_STIFFNESS,
            "Friction": friction,
            "NormalViscousDamping": 0.70,
            "TangentialViscousDamping": 0.10,
        },
    )
    coupling.add_essentials()
    coupling.enginer.pre_calculate()
    coupling.check_critical_timestep()
    return coupling, mesh_quality, initial_center


def _contact(coupling, friction: float):
    neighbor = coupling.contactor.neighbor
    count = int(neighbor.contact_count)
    active = neighbor.contacts.active.to_numpy()[:count].astype(bool)
    normal = neighbor.contacts.normal_force.to_numpy()[:count]
    tangential = neighbor.contacts.tangential_force.to_numpy()[:count]
    gaps = neighbor.contacts.normal_gap.to_numpy()[:count]
    normal_magnitude = np.linalg.norm(normal[active], axis=1)
    tangential_magnitude = np.linalg.norm(tangential[active], axis=1)
    ratios = (
        tangential_magnitude / np.maximum(friction * normal_magnitude, 1.0e-30)
        if friction > 0.0
        else np.zeros_like(tangential_magnitude)
    )
    fem_force = normal[active].sum(axis=0) + tangential[active].sum(axis=0)
    rigid_force = coupling.dem.scene.rigid.contact_force.to_numpy()[0]
    force_scale = max(
        float(np.linalg.norm(fem_force)),
        float(np.linalg.norm(rigid_force)),
        1.0e-30,
    )
    return {
        "candidate_count": count,
        "active_count": int(np.count_nonzero(active)),
        "normal_force": normal[active].sum(axis=0),
        "tangential_force": tangential[active].sum(axis=0),
        "maximum_tangential_force": (float(np.max(tangential_magnitude)) if tangential_magnitude.size else 0.0),
        "maximum_coulomb_ratio": float(np.max(ratios)) if ratios.size else 0.0,
        "maximum_penetration": (float(np.max(-gaps[active])) if np.any(active) else 0.0),
        "action_reaction": float(np.linalg.norm(fem_force + rigid_force) / force_scale),
        "finite": bool(np.isfinite(normal).all() and np.isfinite(tangential).all() and np.isfinite(gaps).all()),
    }


def _fem_motion(coupling):
    state = coupling.fem.engine.state
    position = state.position.to_numpy()
    velocity = state.velocity.to_numpy()
    mass = state.mass.to_numpy()
    total_mass = float(np.sum(mass))
    center = np.sum(mass[:, None] * position, axis=0) / total_mass
    center_velocity = np.sum(mass[:, None] * velocity, axis=0) / total_mass
    radius = position - center
    relative_velocity = velocity - center_velocity
    angular_momentum = np.sum(mass[:, None] * np.cross(radius, relative_velocity), axis=0)
    inertia = np.zeros((3, 3), dtype=np.float64)
    for node_mass, vector in zip(mass, radius):
        inertia += node_mass * (np.dot(vector, vector) * np.eye(3) - np.outer(vector, vector))
    omega = np.linalg.pinv(inertia, rcond=1.0e-12) @ angular_momentum
    return center, center_velocity, omega, total_mass


def _advance(engine, coupling):
    engine.reset_message()
    engine.update_verlet_tables()
    engine.system_resolve()
    engine.integration(update_diagnostics=False)
    dt = float(coupling.sims.delta)
    coupling.sims.current_time += dt
    coupling.dem.sims.current_time += dt
    engine.fem_engine.time = coupling.sims.current_time
    coupling.sims.current_step += 1
    coupling.dem.sims.current_step += 1
    engine.fem_engine.step_count += 1


def _run_case(gt, ti, output: Path, shape: str, dt: float, friction: float):
    started = time.perf_counter()
    coupling, mesh_quality, _ = _build_case(gt, output, shape, dt, friction)
    ti.sync()
    setup_seconds = time.perf_counter() - started
    engine = coupling.enginer
    effective_dt = float(coupling.sims.delta)
    _, slope, _, normal_direction, _ = _frame()
    _, _, _, total_mass = _fem_motion(coupling)

    normal_gravity = -GRAVITY * math.cos(math.radians(ANGLE_DEGREES)) * normal_direction
    target_normal_reaction = total_mass * GRAVITY * math.cos(math.radians(ANGLE_DEGREES))
    maximum_settling_steps = int(round(MAXIMUM_SETTLING_TIME / effective_dt))
    minimum_settling_steps = int(round(MINIMUM_SETTLING_TIME / effective_dt))
    settling_stride = max(1, int(round(1.0e-3 / effective_dt)))
    consecutive = 0
    settling_converged = False
    settling_reaction = 0.0
    settling_normal_speed = math.inf
    settling_steps = 0
    # Preloading is a dissipative equilibrium solve rather than part of the
    # released friction experiment. Damp FEM modes so the release state
    # cannot be an arbitrary peak-compression phase of a contact oscillation.
    engine.fem_engine.damping = SETTLING_FEM_DAMPING
    for settle_step in range(maximum_settling_steps):
        settle_time = settle_step * effective_dt
        fraction = min(max(settle_time / SETTLING_RAMP_TIME, 0.0), 1.0)
        ramp = fraction * fraction * (3.0 - 2.0 * fraction)
        gravity = ramp * normal_gravity
        coupling.dem.sims.set_gravity(gravity.tolist())
        engine.fem_engine.gravity = gravity.copy()
        _advance(engine, coupling)
        settling_steps = settle_step + 1
        if settling_steps % settling_stride == 0:
            state = _contact(coupling, friction)
            _, velocity, _, _ = _fem_motion(coupling)
            settling_reaction = float(abs(np.dot(state["normal_force"], normal_direction)))
            settling_normal_speed = float(abs(np.dot(velocity, normal_direction)))
            reaction_error = abs(settling_reaction - target_normal_reaction) / max(target_normal_reaction, 1.0e-30)
            if (
                settling_steps >= minimum_settling_steps
                and state["active_count"] > 0
                and reaction_error <= SETTLING_REACTION_TOLERANCE
                and settling_normal_speed <= SETTLING_NORMAL_SPEED_TOLERANCE
            ):
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= SETTLING_CONSECUTIVE_SAMPLES:
                settling_converged = True
                break

    engine.fem_engine.damping = 0.0
    engine.fem_engine.state.velocity.fill(0.0)
    engine.fem_engine.state.acceleration.fill(0.0)
    full_gravity = np.array([0.0, 0.0, -GRAVITY])
    coupling.dem.sims.set_gravity(full_gravity.tolist())
    engine.fem_engine.gravity = full_gravity.copy()
    coupling.sims.current_time = 0.0
    coupling.dem.sims.current_time = 0.0
    engine.fem_engine.time = 0.0
    coupling.sims.current_step = 0
    coupling.dem.sims.current_step = 0
    engine.fem_engine.step_count = 0
    initial_center, _, _, _ = _fem_motion(coupling)

    steps = int(round(SIMULATION_TIME / effective_dt))
    sample_stride = max(1, int(round(5.0e-4 / effective_dt)))
    output_stride = max(1, int(round(OUTPUT_INTERVAL / effective_dt)))
    progress_stride = max(1, steps // 10)
    records = []
    saved_steps = []
    peak_candidates = 0
    peak_active = 0
    maximum_coulomb = 0.0
    maximum_tangential_force = 0.0
    maximum_action_reaction = 0.0
    maximum_penetration = 0.0
    minimum_jacobian = 1.0
    finite = True
    loop_started = time.perf_counter()
    for step in range(steps + 1):
        time_value = min(step * effective_dt, SIMULATION_TIME)
        engine.reset_message()
        engine.update_verlet_tables()
        engine.system_resolve()
        if step % sample_stride == 0 or step == steps:
            contact = _contact(coupling, friction)
            center, velocity, omega, _ = _fem_motion(coupling)
            peak_candidates = max(peak_candidates, contact["candidate_count"])
            peak_active = max(peak_active, contact["active_count"])
            maximum_coulomb = max(maximum_coulomb, contact["maximum_coulomb_ratio"])
            maximum_tangential_force = max(
                maximum_tangential_force,
                contact["maximum_tangential_force"],
            )
            maximum_action_reaction = max(maximum_action_reaction, contact["action_reaction"])
            maximum_penetration = max(maximum_penetration, contact["maximum_penetration"])
            finite = (
                finite
                and contact["finite"]
                and bool(np.isfinite(center).all() and np.isfinite(velocity).all() and np.isfinite(omega).all())
            )
            records.append(
                {
                    "step": step,
                    "time": time_value,
                    "slope_displacement": float(np.dot(center - initial_center, slope)),
                    "normal_displacement": float(np.dot(center - initial_center, normal_direction)),
                    "normal_penetration": contact["maximum_penetration"],
                    "slope_velocity": float(np.dot(velocity, slope)),
                    "angular_velocity_y": float(omega[1]),
                    "rolling_ratio": float(abs(omega[1]) * RADIUS / max(abs(np.dot(velocity, slope)), 1.0e-30)),
                    "normal_reaction": float(abs(np.dot(contact["normal_force"], normal_direction))),
                    "tangential_reaction": float(abs(np.dot(contact["tangential_force"], slope))),
                    "coulomb_ratio": contact["maximum_coulomb_ratio"],
                    "active_contact_count": contact["active_count"],
                    "candidate_count": contact["candidate_count"],
                    "action_reaction_relative": contact["action_reaction"],
                    "minimum_jacobian": float(engine.minimum_jacobian),
                }
            )
        if (step == 0 or step % output_stride == 0 or step == steps) and (not saved_steps or saved_steps[-1] != step):
            coupling.save_data()
            saved_steps.append(step)
        if step == steps:
            break
        if step > 0 and step % progress_stride == 0:
            print(
                json.dumps({"case": shape, "mu": friction, "progress": step / steps}),
                flush=True,
            )
        engine.integration(update_diagnostics=False)
        minimum_jacobian = min(minimum_jacobian, float(engine.minimum_jacobian))
        coupling.sims.current_time += effective_dt
        coupling.dem.sims.current_time += effective_dt
        engine.fem_engine.time = coupling.sims.current_time
        coupling.sims.current_step += 1
        coupling.dem.sims.current_step += 1
        engine.fem_engine.step_count += 1
    ti.sync()
    loop_seconds = time.perf_counter() - loop_started
    output_evidence = collect_native_output(output / "native", len(saved_steps))

    fit = [row for row in records if 0.04 <= row["time"] <= 0.20 and row["active_contact_count"] > 0]
    measured_acceleration = (
        float(
            np.polyfit(
                np.asarray([row["time"] for row in fit]),
                np.asarray([row["slope_velocity"] for row in fit]),
                1,
            )[0]
        )
        if len(fit) >= 3
        else math.nan
    )
    angle = math.radians(ANGLE_DEGREES)
    sphere_stick_threshold = (2.0 / 7.0) * math.tan(angle)
    if shape == "sphere" and friction >= sphere_stick_threshold:
        regime = "stick"
        reference_acceleration = (5.0 / 7.0) * GRAVITY * math.sin(angle)
    elif shape == "sphere":
        regime = "slip"
        reference_acceleration = GRAVITY * (math.sin(angle) - friction * math.cos(angle))
    elif friction >= math.tan(angle):
        regime = "stick"
        reference_acceleration = 0.0
    else:
        regime = "slip"
        reference_acceleration = GRAVITY * (math.sin(angle) - friction * math.cos(angle))
    acceleration_error = (
        abs(measured_acceleration - reference_acceleration) / reference_acceleration
        if reference_acceleration > 1.0e-12 and math.isfinite(measured_acceleration)
        else None
    )
    summary = {
        "id": shape,
        "friction": friction,
        "inclination_degrees": ANGLE_DEGREES,
        "moving_body": f"FEM {shape}",
        "incline_body": "fixed LSDEM plate",
        "requested_dt": dt,
        "effective_dt": effective_dt,
        "steps": steps,
        "settling_steps": settling_steps,
        "settling_time": settling_steps * effective_dt,
        "settling_converged": settling_converged,
        "settling_target_normal_reaction": target_normal_reaction,
        "settling_final_normal_reaction": settling_reaction,
        "settling_final_normal_speed": settling_normal_speed,
        "settling_fem_damping_per_second": SETTLING_FEM_DAMPING,
        "settling_contact_work_mode": "Explicit",
        "released_contact_work_mode": "Explicit",
        "maximum_candidate_count": peak_candidates,
        "maximum_active_contact_count": peak_active,
        "maximum_coulomb_ratio": maximum_coulomb,
        "maximum_tangential_force": maximum_tangential_force,
        "maximum_action_reaction_relative": maximum_action_reaction,
        "maximum_normal_penetration": maximum_penetration,
        "minimum_jacobian": minimum_jacobian,
        "measured_slope_acceleration": measured_acceleration,
        "friction_regime": regime,
        "reference_acceleration": reference_acceleration,
        "acceleration_relative_error": acceleration_error,
        "final_slope_displacement": records[-1]["slope_displacement"],
        "final_rolling_ratio": records[-1]["rolling_ratio"],
        "finite_state": finite,
        "fem_mesh_quality": mesh_quality,
    }
    gates = {
        "finite_state": finite,
        "mesh_resolution": bool(mesh_quality["passed"]),
        "contact_occurred": peak_active > 0,
        "coulomb_cap": (maximum_coulomb <= 1.0 + 1.0e-8 if friction > 0.0 else maximum_tangential_force <= 1.0e-10),
        "action_reaction": maximum_action_reaction <= 1.0e-10,
        "bounded_penetration": maximum_penetration <= 0.05 * RADIUS,
        "positive_jacobian": minimum_jacobian > 0.10,
        "capacity": peak_candidates <= 4096,
        "expected_motion": records[-1]["slope_displacement"] > 0.0,
        "native_output": bool(output_evidence["complete"]),
        "settling_equilibrium": settling_converged,
    }
    if reference_acceleration > 1.0e-12:
        gates["reference_acceleration"] = bool(acceleration_error is not None and acceleration_error <= 0.15)
    performance = {
        "id": shape,
        "setup_seconds": setup_seconds,
        "simulation_loop_seconds": loop_seconds,
        "fem_nodes": int(engine.fem_engine.mesh.number_of_nodes),
        "fem_dofs": int(3 * engine.fem_engine.mesh.number_of_nodes),
        "fem_elements": int(engine.fem_engine.mesh.number_of_cells),
        "fem_mesh_quality": mesh_quality,
        "saved_frame_count": len(saved_steps),
        "saved_steps": saved_steps,
        "native_output": output_evidence,
    }
    return summary, gates, records, performance


def _preflight(gt, ti, output, selected, dt, friction):
    records = []
    for shape in selected:
        coupling, mesh_quality, _ = _build_case(gt, output / shape, shape, dt, friction)
        engine = coupling.enginer
        _advance(engine, coupling)
        ti.sync()
        position = engine.fem_engine.state.position.to_numpy()
        velocity = engine.fem_engine.state.velocity.to_numpy()
        rigid_center = coupling.dem.scene.rigid.mass_center.to_numpy()
        finite = bool(np.isfinite(position).all() and np.isfinite(velocity).all() and np.isfinite(rigid_center).all())
        record = {
            "case": shape,
            "friction": friction,
            "passed": bool(
                mesh_quality["passed"]
                and mesh_quality["element_count"] >= MINIMUM_FEM_ELEMENTS
                and finite
                and float(engine.minimum_jacobian) > 0.0
                and abs(float(coupling.sims.delta) - dt) <= 1.0e-12 * max(abs(dt), 1.0)
            ),
            "moving_body": f"FEM {shape}",
            "incline_body": "fixed LSDEM plate",
            "inclination_degrees": ANGLE_DEGREES,
            "mesh_quality": mesh_quality,
            "requested_dt": dt,
            "effective_dt": float(coupling.sims.delta),
            "one_step_finite": finite,
            "one_step_minimum_jacobian": float(engine.minimum_jacobian),
            "cross_candidate_count": int(coupling.contactor.neighbor.contact_count),
            "contact_work_mode": coupling.sims.contact_work_mode,
        }
        records.append(record)
    payload = {
        "schema_version": 1,
        "passed": all(record["passed"] for record in records),
        "cases": records,
    }
    (output / "preflight.json").write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if payload["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--default-fp", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--friction", type=float, default=0.2)
    parser.add_argument("--only", choices=CASES)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
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
    selected = [shape for shape in CASES if args.only is None or shape == args.only]
    if args.preflight:
        return _preflight(gt, ti, output, selected, args.dt, args.friction)

    summaries, case_gates, timings, history = [], {}, [], []
    for shape in selected:
        summary, gates, records, performance = _run_case(gt, ti, output, shape, args.dt, args.friction)
        summaries.append(summary)
        case_gates[shape] = gates
        timings.append(performance)
        history.extend({"shape": shape, **record} for record in records)
    metrics = {
        "schema_version": 1,
        "passed": all(all(gates.values()) for gates in case_gates.values()),
        "case_gates": case_gates,
        "summaries": summaries,
    }
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": _prompt(),
        "units": "SI",
        "execution": {
            "arch": args.arch,
            "precision": args.default_fp,
            "taichi_version": list(ti.__version__),
            "gpu": _command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
            "command": [sys.executable, *sys.argv],
        },
        "parameters": {
            "shapes": selected,
            "moving_body": "FEM particle",
            "incline_body": "fixed LSDEM plate",
            "characteristic_radius": RADIUS,
            "cube_side": CUBE_SIDE,
            "density": DENSITY,
            "young_modulus": YOUNG,
            "poisson_ratio": POISSON,
            "inclination_degrees": ANGLE_DEGREES,
            "gravity": GRAVITY,
            "friction": args.friction,
            "normal_stiffness": NORMAL_STIFFNESS,
            "tangential_stiffness": TANGENTIAL_STIFFNESS,
            "requested_dt": args.dt,
            "simulation_time": SIMULATION_TIME,
            "output_interval": OUTPUT_INTERVAL,
            "minimum_fem_elements_per_particle": MINIMUM_FEM_ELEMENTS,
            "minimum_tetra_mean_ratio": MINIMUM_TETRA_MEAN_RATIO,
            "settling_fem_damping_per_second": SETTLING_FEM_DAMPING,
            "settling_contact_work_mode": "Explicit",
            "contact_work_mode": "Explicit",
            "search": "BVH",
            "precision": args.default_fp,
        },
    }
    performance = {
        "schema_version": 1,
        "host": platform.node(),
        "python": platform.python_version(),
        "taichi": list(ti.__version__),
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "cases": timings,
    }
    for filename, payload in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("performance.json", performance),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")
    if history:
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    print(json.dumps({"passed": metrics["passed"], "summaries": summaries}, indent=2))
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
