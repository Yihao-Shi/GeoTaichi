"""Rotate and advance an inclined screw pile into a dry soil bed."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUTS = HERE / "inputs"


def soil_points(size, counts):
    spacing = size / counts
    axes = [(np.arange(n) + 0.5) * dx for n, dx in zip(counts, spacing)]
    axes[0] -= size[0] / 2
    axes[1] -= size[1] / 2
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return points, float(spacing.prod())


def command(time, tilt, start_tip, pitch, turns, settle, duration):
    progress = np.clip((time - settle) / duration, 0.0, 1.0)
    revolutions = turns * progress**2 * (3.0 - 2.0 * progress)
    advance = pitch * revolutions
    # The local right-handed helix rises with angle, so insertion spins about -local z.
    rotation = tilt * Rotation.from_rotvec([0, 0, -2.0 * np.pi * revolutions])
    return start_tip - advance * tilt.as_matrix()[:, 2], rotation, advance, revolutions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=("low", "full"), default="low")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stop-time", type=float, help="Stop a setup or contact probe early (s).")
    parser.add_argument("--contact-coordination-number", type=float)
    args = parser.parse_args()
    contract = json.loads((HERE / "model-contract.json").read_text())
    profile = contract["discretization"]["profiles"][args.resolution]
    setup, timing = contract["geometry"], contract["time"]
    dt = profile["dt_s"]
    stop = timing["duration"] if args.stop_time is None else args.stop_time
    if not dt <= stop <= timing["duration"]:
        parser.error("Require nominal dt <= stop-time <= installation end")

    mesh = trimesh.load_mesh(INPUTS / profile["mesh"], merge_tex=True, merge_norm=True)
    if not mesh.is_volume or not np.isfinite(mesh.vertices).all():
        raise ValueError("The pile must be a finite, closed, oriented solid")
    size = np.asarray(setup["soil_size_m"])
    points, volume = soil_points(size, np.asarray(profile["soil_counts"]))
    poses = json.loads((INPUTS / "inclination_cases.json").read_text())
    pose = next(p for p in poses["poses"] if p["inclination_deg"] == setup["inclination_deg"])
    tilt = Rotation.from_matrix(pose["rotation_matrix_local_to_world"])
    axis = tilt.as_matrix()[:, 2]
    mesh.vertices = tilt.apply(mesh.vertices)
    clearance = poses["initial_mesh_clearance_m"]
    tip = np.array([0, 0, size[2] + clearance - mesh.vertices[:, 2].min()])
    mesh.apply_translation(tip)
    center = mesh.center_mass
    anchors = center + np.vstack(([0, 0, 0], np.eye(3))) - 0.25
    anchor_local = tilt.inv().apply(anchors - tip)
    tip_offset = tip - anchors[0]
    head_offset = tip_offset + setup["length_m"] * axis

    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import geotaichi as gt
    import taichi as ti

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    mesh_file = output / "pile-initial.obj"
    mesh.export(mesh_file, include_normals=False, include_texture=False)
    gt.init(arch=args.arch, default_fp="float64", log=False, offline_cache=True, cpu_max_num_threads=1)
    model = gt.MPDEM(coupling=False, log=False)
    mpm, dem = model.mpm, model.dem
    material, pile = contract["materials"]
    steel = pile["parameters"]
    gravity = contract["initial_conditions"]["gravity"]
    contact = contract["contact_or_coupling"]["parameters"]
    grid_size = profile["grid_size_m"]
    domain = [16, 16, 20]
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=domain,
        gravity=gravity,
        visualize=True,
        ipc=True,
    )
    body = mpm.create_body()
    body.add_particles(
        points,
        volume=volume,
        grid_size=grid_size,
        xmin=[-8, -8, -0.5],
        xmax=[8, 8, 6],
        boundary_ids=np.arange(len(points), dtype=np.int32),
        surface_measure=volume ** (2 / 3),
    )
    mpm.add_body(body)
    floor = mpm.create_ground()
    floor.append([0, 0, 0], [0, 0, 1])
    mpm.add_ground(floor)
    mpm.add_material(model=material["model"], **material["parameters"])
    mpm.add_element({"ElementSize": grid_size, "ShapeFunction": "Linear"})
    mpm.memory_allocate({"max_material_number": 1, "max_particle_number": len(points)}, log=False)
    mpm.add_contact(
        "BarrierIPC",
        **{key: contact[key] for key in ("dhat", "dmin", "kappa", "epsv")},
        mu=contact["friction_coefficient"],
        friction_mode="lagged",
        friction_iterations=-1,
    )

    dem.set_configuration(domain=domain, scheme="AffineBody", search="BVH", gravity=gravity, visualize=True, log=False)
    model.set_configuration(
        domain=domain, coupling_scheme="MPDEM", gravity=gravity, search="BVH", visualize=True, log=False
    )
    dem.set_affine_body_parameters(
        newmark=[1.0, 0.5, 1.0],
        assemble_type="HashTriplet",
        young_modulus=steel["YoungModulus"],
        dhat=contact["dhat"],
        barrier_stiffness=contact["kappa"],
        friction_iterations=-1,
        friction_max_iterations=contact["friction_max_iterations"],
        friction_tolerance=1e-7,
        newton_tolerance=1e-7,
        max_newton_iteration=100,
        line_search_max_iteration=40,
        max_step=0.10,
        ccd_type="accd",
    )
    dem.memory_allocate(
        {
            "max_affine_body_number": 1,
            "max_rigid_template_number": 1,
            "max_material_number": 1,
            "contact_coordination_number": 1 / len(mesh.faces),
            "surface_node_number": len(mesh.vertices),
            "body_coordination_number": 4,
            "wall_coordination_number": 1,
        },
        log=False,
    )
    dem.add_attribute(0, steel)
    dem.add_template(
        {"Name": "pile", "TemplateType": "AffineBody", "Object": gt.polyhedron(file=str(mesh_file)).reset(False)}
    )
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": {"Name": "pile", "BodyPoint": center.tolist(), "MaterialID": 0, "GroupID": 0},
        }
    )
    stiffness = contract["boundary_conditions"][1]["position_stiffness_N_per_m"]
    for anchor in anchors:
        dem.add_joint(
            {
                "JointType": "Spherical",
                "BodyID1": 0,
                "BodyID2": -1,
                "WorldAnchor": anchor.tolist(),
                "PositionStiffness": stiffness,
            }
        )
    solver = dict(contract["api"]["solver"], Timestep=dt, SimulationTime=stop, SavePath=str(output))
    model.set_solver(solver, log=False)
    coordination = args.contact_coordination_number
    if coordination is None:
        coordination = profile["contact_coordination_number"]
    model.memory_allocate({"body_coordination_number": 4, "contact_coordination_number": coordination}, log=False)
    model.choose_contact_model("BarrierIPC", **contact, friction_mode="lagged", friction_iterations=-1, project_pd=True)
    model.add_ipc_property(MPMbody=0, AffineBody=0, property=contact)

    run_config = dict(
        resolution=args.resolution,
        backend=str(ti.lang.impl.current_cfg().arch),
        precision="float64",
        profile=profile,
        particles=len(points),
        pile_triangles=len(mesh.faces),
        geometry=setup,
        timing=timing,
        contact=contact,
        solver=solver,
        contact_coordination_number=coordination,
    )
    (output / "run-config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    motion = (tilt, tip, setup["pitch_m"], setup["turns"], timing["settle_s"], timing["installation_s"])
    rows = []

    def drive(system):
        target_tip, rotation, _, _ = command(system.mpm.time + system.mpm.dt, *motion)
        # Recompute world anchor targets on every attempt, including rollback retries.
        system.affine.joint_world_anchor.from_numpy(target_tip + rotation.apply(anchor_local))

    def observe(system):
        step = system.last_step_record
        time = step["time"]
        target_tip, rotation, advance, turns = command(time, *motion)
        controls = system.affine.y.to_numpy()
        gradient = (controls[1:] - controls[0]).T
        actual_tip = controls[0] + gradient @ tip_offset
        actual_head = controls[0] + gradient @ head_offset
        errors = target_tip + rotation.apply(anchor_local) - controls
        forces = stiffness * errors
        push = -axis @ forces.sum(axis=0)
        torque = -axis @ np.cross(controls - actual_head, forces).sum(axis=0)
        direction = tilt.inv().apply(gradient @ tilt.as_matrix()[:, 0])
        actual_turns = -np.arctan2(direction[1], direction[0]) / (2 * np.pi)
        actual_turns += np.round(turns - actual_turns)
        positions = system.mpm.particle.x.to_numpy()
        ground_gap = positions[:, 2].min() - contact["dmin"]
        gap = step["contact"]["minimum_distance"]
        candidates = system.last_candidate_count
        metric_error = np.linalg.norm(gradient.T @ gradient - np.eye(3))
        tracking = np.linalg.norm(errors, axis=1).max()
        retry = system.mpm.history[-1]["step_retry"]
        retries = retry["retry_count"]
        row = [
            time,
            system.mpm.dt,
            retries,
            advance,
            turns,
            (tip - actual_tip) @ axis,
            tip[2] - actual_tip[2],
            actual_turns,
            push,
            torque,
            tracking,
            metric_error,
            ground_gap,
            gap,
            system.last_contact_count,
            candidates,
            step["linear_iterations"],
        ]
        if (
            not np.isfinite(positions).all()
            or not np.isfinite(row[:13] + row[14:]).all()
            or not (ground_gap > 0 and gap > 0)
            or (not np.isfinite(gap) and candidates != 0)
            or metric_error >= 0.01
            or np.linalg.det(gradient) <= 0
        ):
            raise RuntimeError(f"Nonfinite or infeasible pile state at t={time:g} s")
        rows.append(row)
        writer.writerow(row)
        if len(rows) % 20 == 0 or retries:
            print(f"pile t={time:.6f} s advance={row[5]:.4f} m pairs={candidates} retries={retries}", flush=True)
        if retries:
            print(f"Retry at t={retry['attempts'][0]['time']:.6f} s: {retry['attempts'][0]['message']}", flush=True)
        next_dt = min(dt, system.mpm.dt * 1.2) if len(rows) % 8 == 0 and retries == 0 else system.mpm.dt
        if next_dt != model.sims.delta:
            model.modify_parameters(SimulationTime=stop, Timestep=next_dt)

    columns = (
        "time_s",
        "dt_s",
        "retries",
        "target_advance_m",
        "target_turns",
        "actual_advance_m",
        "actual_vertical_depth_m",
        "actual_turns",
        "drive_push_N",
        "drive_torque_Nm",
        "anchor_error_m",
        "affine_metric_error",
        "soil_floor_clearance_m",
        "mixed_clearance_m",
        "mixed_contacts",
        "mixed_candidates",
        "linear_iterations",
    )
    with (output / "accepted-steps.csv").open("w", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        try:
            model.run(verbose=False, preintegration_function=drive, postprocessing=[observe])
        except Exception as error:
            direct_mpm = getattr(getattr(model, "enginer", None), "mpm", None)
            failure = dict(
                exception=type(error).__name__,
                message=str(error),
                last_recorded_time_s=rows[-1][0] if rows else 0.0,
                solver_failure=getattr(direct_mpm, "last_failure", None),
            )
            (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise
    if not rows or not np.isclose(rows[-1][0], stop, rtol=0, atol=1e-12):
        raise RuntimeError("Simulation stopped before the requested end time")
    summary = dict(zip(columns, rows[-1]))
    summary.update(
        complete_installation=stop == timing["duration"],
        steps=len(rows),
        total_retries=sum(r[2] for r in rows),
        peak_mixed_contacts=max(r[14] for r in rows),
        peak_mixed_candidates=max(r[15] for r in rows),
    )
    if np.isposinf(summary["mixed_clearance_m"]):
        summary["mixed_clearance_m"] = None
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
