"""Stationary two-joint ABD excavator scooping plastic Direct-MPM soil."""

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[4]
CONTRACT = Path(__file__).with_name("model-contract.json")


def smooth(time, start, end):
    u = np.clip((time - start) / (end - start), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def targets(time, motion, geometry):
    tilt = smooth(time, motion["settle_until"], motion["tilt_until"])
    insert = smooth(time, motion["tilt_until"], motion["insert_until"])
    dig = smooth(time, motion["insert_until"], motion["dig_until"])
    # Squaring removes the angle-velocity jump at the vertical-tip arcsine branch.
    dig *= dig
    curl = smooth(time, motion["dig_until"], motion["curl_until"])
    lift = smooth(time, motion["curl_until"], motion["lift_until"])
    link = np.asarray(geometry["bucket_hinge"]) - geometry["pivot"]
    tip = np.array([geometry["bucket_x"][0], geometry["bucket_z"][0]])
    radius = np.linalg.norm(tip)
    tip_z = geometry["ground_z"] + motion["cutting_height"]
    insert_boom = np.degrees(
        np.arctan2(link[2], link[0]) - np.arcsin((tip_z + radius - geometry["pivot"][2]) / np.linalg.norm(link))
    )
    insert_pitch = np.degrees(np.arctan2(tip[0], -tip[1]))
    cutting_boom = insert_boom + (motion["dig_boom_angle"] - insert_boom) * dig

    def cutting_pitch(boom):
        angle = np.radians(boom)
        hinge_z = geometry["pivot"][2] + link[2] * np.cos(angle) - link[0] * np.sin(angle)
        return np.degrees(np.arcsin(np.clip((tip_z - hinge_z) / radius, -1.0, 1.0)) - np.arctan2(tip[1], -tip[0]))

    boom = (
        insert_boom * insert
        + (motion["dig_boom_angle"] - insert_boom) * dig
        + (motion["curl_boom_angle"] - motion["dig_boom_angle"]) * curl
        + (motion["raised_boom_angle"] - motion["curl_boom_angle"]) * lift
    )
    pitch = (
        insert_pitch * tilt
        + cutting_pitch(cutting_boom)
        - insert_pitch
        + (motion["curled_bucket_pitch"] - cutting_pitch(motion["dig_boom_angle"])) * curl
    )
    return [boom, boom - pitch]


def bucket_mesh(geometry):
    """Boundary of a box with one open-front/open-top cavity; no overlapping solids."""
    axes = [np.asarray(geometry["bucket_" + name]) for name in "xyz"]
    cells = set(itertools.product(range(2), range(3), range(2))) - {(0, 1, 1)}
    indices, vertices, faces = {}, [], []
    for cell in sorted(cells):
        for axis, sign in itertools.product(range(3), (-1, 1)):
            neighbor = list(cell)
            neighbor[axis] += sign
            if tuple(neighbor) in cells:
                continue
            quad = []
            for first, second in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = list(cell)
                point[axis] += int(sign > 0)
                point[(axis + 1) % 3] += first
                point[(axis + 2) % 3] += second
                key = tuple(point)
                if key not in indices:
                    indices[key] = len(vertices)
                    vertices.append([axes[i][point[i]] for i in range(3)])
                quad.append(indices[key])
            if sign < 0:
                quad.reverse()
            faces.extend([(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    return mesh


def soil_side_dofs(grid_shape):
    nx, ny, nz = map(int, grid_shape)
    nodes = np.arange(nx * ny * nz, dtype=np.int32).reshape(nz, ny, nx)
    x_sides = np.unique(np.concatenate((nodes[:, :, 0].ravel(), nodes[:, :, -1].ravel())))
    y_sides = np.unique(np.concatenate((nodes[:, 0, :].ravel(), nodes[:, -1, :].ravel())))
    return [list(3 * x_sides), list(3 * y_sides + 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--stop-time", type=float, default=1.6)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "OutputData/excavator_dig")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text())
    dt = contract["time"]["timestep"]
    if (
        not np.isfinite(args.stop_time)
        or not 0 < args.stop_time <= 1.6
        or not np.isclose(args.stop_time / dt, round(args.stop_time / dt))
    ):
        parser.error(f"stop-time must be a positive integer number of {dt} s steps, at most 1.6 s")
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import geotaichi as gt
    import taichi as ti

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gt.init(arch=args.arch, default_fp="float64", log=False, offline_cache=True, cpu_max_num_threads=2)
    geometry = contract["geometry"]["dimensions"]
    motion = contract["boundary_conditions"][2]
    contact = contract["contact_or_coupling"]["parameters"]
    h = contract["discretization"]["grid_spacing"]
    coupling = gt.MPDEM(coupling=False, log=False)
    mpm, dem = coupling.mpm, coupling.dem
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=[1.2, 1.0, 1.2],
        gravity=[0, 0, -9.81],
        visualize=True,
        ipc=True,
    )
    counts = np.rint((np.array(geometry["soil_max"]) - geometry["soil_min"]) / h).astype(int)
    points = np.stack(
        np.meshgrid(*[geometry["soil_min"][i] + (np.arange(counts[i]) + 0.5) * h for i in range(3)], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    body = mpm.create_body()
    body.add_particles(
        points,
        volume=h**3,
        grid_size=h,
        xmin=geometry["grid_min"],
        xmax=geometry["grid_max"],
        boundary_ids=np.arange(len(points), dtype=np.int32),
        surface_measure=h * h,
    )
    mpm.add_body(body)
    ground = mpm.create_ground()
    ground.append([0, 0, geometry["ground_z"]], [0, 0, 1])
    mpm.add_ground(ground)
    mpm.add_material(model=contract["materials"][0]["model"], **contract["materials"][0]["parameters"])
    mpm.add_element({"ElementSize": h, "ShapeFunction": "Linear"})
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    grid_shape = np.ceil((np.array(geometry["grid_max"]) - geometry["grid_min"]) / h).astype(int) + 1
    fixed = soil_side_dofs(grid_shape)
    boundary = DirichletBoundary()
    boundary.append(fixed, [0.0] * sum(map(len, fixed)))
    mpm.add_boundary_condition(dirichlet=boundary)
    mpm.memory_allocate({"max_material_number": 1, "max_particle_number": len(points)}, log=False)
    mpm.add_contact(
        "BarrierIPC",
        dhat=contact["dhat"],
        dmin=contact["dmin"],
        kappa=contact["kappa"],
        mu=contact["friction_coefficient"],
        epsv=contact["epsv"],
        friction_mode="lagged",
        friction_iterations=-1,
    )
    dem.set_configuration(
        domain=[1.2, 1.0, 1.2], scheme="AffineBody", search="BVH", gravity=[0, 0, -9.81], visualize=True, log=False
    )
    coupling.set_configuration(
        domain=[1.2, 1.0, 1.2], coupling_scheme="MPDEM", gravity=[0, 0, -9.81], search="BVH", visualize=True, log=False
    )
    dem.set_affine_body_parameters(
        newmark=[1.0, 0.5, 1.0],
        assemble_type="HashTriplet",
        young_modulus=contract["materials"][1]["parameters"]["YoungModulus"],
        dhat=contact["dhat"],
        barrier_stiffness=contact["kappa"],
        friction_iterations=-1,
        friction_max_iterations=contact["friction_max_iterations"],
        friction_tolerance=1e-7,
        newton_tolerance=1e-7,
        max_newton_iteration=100,
        line_search_max_iteration=40,
        max_step=0.02,
        ccd_type="accd",
    )
    bucket = bucket_mesh(geometry)
    pivot, hinge = np.array(geometry["pivot"]), np.array(geometry["bucket_hinge"])
    boom = trimesh.creation.box(extents=[np.linalg.norm(hinge - pivot), geometry["boom_width"], geometry["boom_width"]])
    angle = np.arctan2(pivot[2] - hinge[2], hinge[0] - pivot[0])
    boom.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
    dem.memory_allocate(
        {
            **contract["capacities"]["allocations"],
            "max_material_number": 1,
            "max_rigid_template_number": 2,
            "surface_node_number": len(boom.vertices) + len(bucket.vertices),
            "body_coordination_number": 4,
            "wall_coordination_number": 4,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.add_attribute(0, contract["materials"][1]["parameters"])
    for name, mesh, center in (("boom", boom, 0.5 * (pivot + hinge)), ("bucket", bucket, hinge + bucket.center_mass)):
        path = output / (name + ".obj")
        mesh.export(path)
        dem.add_template(
            {"Name": name, "TemplateType": "AffineBody", "Object": gt.polyhedron(file=str(path)).reset(False)}
        )
        dem.create_body(
            {
                "BodyType": "AffineBody",
                "Template": {"Name": name, "BodyPoint": center.tolist(), "MaterialID": 0, "GroupID": 0},
            }
        )
    for body_id, spec in enumerate(contract["boundary_conditions"][:2]):
        dem.add_joint(
            {
                "JointType": "Revolute",
                "BodyID1": body_id,
                "BodyID2": -1 if body_id == 0 else 0,
                "WorldAnchor": (pivot if body_id == 0 else hinge).tolist(),
                "WorldAxis": [0, 1, 0],
                "PositionStiffness": spec["position_stiffness"],
                "AxisStiffness": spec["axis_stiffness"],
                "MotorStiffness": spec["motor_stiffness"],
                "Damping": spec["damping"],
            }
        )
    dem.add_wall(
        {"WallType": "Plane", "MaterialID": 0, "WallCenter": [0, 0, geometry["ground_z"]], "OuterNormal": [0, 0, 1]}
    )
    dem.add_property(
        0,
        0,
        {"Dhat": contact["dhat"], "BarrierStiffness": contact["kappa"], "Friction": contact["friction_coefficient"]},
        dType="all",
    )
    coupling.set_solver(
        {
            **contract["api"]["solver"],
            "SimulationTime": args.stop_time,
            "SavePath": str(output),
        },
        log=False,
    )
    coupling.memory_allocate(
        {"body_coordination_number": 4, "max_point_triangle_pairs": 16384, "max_point_edge_pairs": 1}, log=False
    )
    coupling.choose_contact_model(
        "BarrierIPC", **contact, friction_mode="lagged", friction_iterations=-1, project_pd=True
    )
    for affine in (0, 1):
        coupling.add_ipc_property(MPMbody=0, AffineBody=affine, property=contact)
    rows = []

    def drive(system):
        dem.set_joint_target_angles([0, 1], targets(system.mpm.time + system.mpm.dt, motion, geometry))

    def observe(system):
        assert system.ipc.gbarrier.shape[0] >= len(points) and system.ipc.gfriction.shape[0] >= len(points)
        particles = system.mpm.particle.x.to_numpy()
        body_controls = system.affine.y.to_numpy().reshape(-1, 4, 3)
        controls = body_controls[1]
        deformation = (body_controls[:, 1:] - body_controls[:, :1]).transpose(0, 2, 1)
        minimum_determinant = float(np.linalg.det(deformation).min())
        rigidity_error = float(
            np.linalg.norm(deformation.transpose(0, 2, 1) @ deformation - np.eye(3), axis=(1, 2)).max()
        )
        joint_state = dem.get_joint_states()
        anchor_error = float(joint_state["anchor_error"].max())
        angle_error = float(abs(joint_state["angle"] - joint_state["target_angle"]).max())
        transform = (controls[1:] - controls[:1]).T
        local = np.linalg.solve(transform, (particles - controls[0]).T).T - 0.25 + bucket.center_mass
        inside = (
            (local[:, 0] > -0.14)
            & (local[:, 0] < 0.10)
            & (abs(local[:, 1]) < 0.10)
            & (local[:, 2] > -0.08)
            & (local[:, 2] < 0.10)
        )
        lifted = inside & (particles[:, 2] > geometry["soil_max"][2] + 0.05)
        carried_mass = float(lifted.sum() * h**3 * contract["materials"][0]["parameters"]["density"])
        # The mixed diagnostic already includes each pair's dmin offset.
        gap = system.mixed.diagnostics()["minimum_distance"]
        rigid_vertices = system.affine.x.to_numpy()
        rigid_gap = float(rigid_vertices[:, 2].min() - geometry["ground_z"])
        plastic_strain = system.mpm.material.equivalent_plastic_strain.to_numpy()
        if (
            not np.isfinite(particles).all()
            or not np.isfinite(rigid_vertices).all()
            or not np.isfinite(plastic_strain).all()
            or gap <= 0
            or rigid_gap <= 0
            or minimum_determinant <= 0
            or not np.isfinite(angle_error)
        ):
            raise RuntimeError("Nonfinite or infeasible excavation state")
        row = [
            coupling.sims.current_time,
            carried_mass,
            gap,
            system.last_contact_count,
            float(plastic_strain.max()),
            system.mpm.dt,
            system.mpm.history[-1]["step_retry"]["retry_count"],
            rigid_gap,
            minimum_determinant,
            rigidity_error,
            anchor_error,
            angle_error,
        ]
        rows.append(row)
        writer.writerow(row)
        if len(rows) % 20 == 0:
            print(
                f"dig t={row[0]:.3f} carried_mass={carried_mass:.4f} contacts={row[3]} plastic_strain={row[4]:.3e}",
                flush=True,
            )
        next_dt = min(dt, system.mpm.dt * 1.2) if len(rows) % 8 == 0 and row[6] == 0 else system.mpm.dt
        if next_dt != coupling.sims.delta:
            coupling.modify_parameters(SimulationTime=args.stop_time, Timestep=next_dt)

    with (output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "carried_soil_mass",
                "minimum_clearance",
                "mixed_contacts",
                "maximum_plastic_strain",
                "dt",
                "retries",
                "rigid_ground_clearance",
                "minimum_affine_determinant",
                "maximum_rigidity_error",
                "maximum_joint_anchor_error",
                "maximum_joint_angle_error_degrees",
            ]
        )
        try:
            result = coupling.run(verbose=False, preintegration_function=drive, postprocessing=[observe])
        except Exception as error:
            system = coupling.enginer
            diagnostic = dict(
                error=str(error),
                accepted_time=float(coupling.sims.current_time),
                accepted_step=int(coupling.sims.current_step),
                linear=system.last_linear_solve,
                last_step=system.last_step_record,
            )
            (output / "failure-diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
            np.savez(
                output / "failed-accepted-state.npz",
                position=system.mpm.particle.x.to_numpy(),
                velocity=system.mpm.particle.v.to_numpy(),
                acceleration=system.mpm.particle.a.to_numpy(),
                deformation_gradient=system.mpm.F0.to_numpy(),
                plastic_deformation_inverse=system.mpm.material.plastic_deformation_inverse.to_numpy(),
                equivalent_plastic_strain=system.mpm.material.equivalent_plastic_strain.to_numpy(),
                volumetric_plastic_strain=system.mpm.material.volumetric_plastic_strain.to_numpy(),
                affine_position=system.affine.y.to_numpy(),
                affine_velocity=system.affine.velocity_y.to_numpy(),
            )
            raise
    system = coupling.enginer
    # Isolate the existing mixed residual at the final assembled geometry for a force-balance diagnostic.
    system.rhs.fill(0.0)
    system.mixed.assemble(
        int(system.mixed.culling.point_triangle_count[None]),
        system.mixed.candidate_field(),
        system.mixed_source,
        system.rhs,
        False,
        True,
        int(system.mixed.friction_count),
    )
    reaction_error = float(
        np.linalg.norm(system.rhs.to_numpy().reshape(-1, 3).sum(axis=0)) / float(system.affine.scale_device[None])
    )
    initial_mass = len(points) * h**3 * contract["materials"][0]["parameters"]["density"]
    mass = float(system.mpm.particle.m.to_numpy().sum())
    complete = args.stop_time == contract["time"]["duration"]
    observed = [
        rows[-1][1],
        min(row[2] for row in rows),
        abs(mass - initial_mass) / initial_mass,
        reaction_error,
        min(row[7] for row in rows),
        max(row[9] for row in rows),
        max(row[10] for row in rows),
        max(row[11] for row in rows),
    ]
    checks = []
    for name, criterion, value in zip(
        (
            "carried soil mass",
            "contact clearance",
            "soil mass",
            "mixed action/reaction",
            "rigid ground clearance",
            "affine rigidity",
            "joint anchor accuracy",
            "joint angle accuracy",
        ),
        contract["validation"]["invariants"],
        observed,
    ):
        checks.append(
            dict(
                name=name,
                kind=criterion["kind"],
                expected=criterion["expectation"],
                observed=value,
                tolerance=criterion["tolerance"],
                status="pass" if criterion["expectation"][0] <= value <= criterion["expectation"][1] else "fail",
                evidence="trajectory.csv and summary.json",
            )
        )
    timestep_consistent = bool(np.isclose(sum(row[5] for row in rows), args.stop_time, rtol=0.0, atol=1e-12))
    passed = complete and timestep_consistent and all(check["status"] == "pass" for check in checks)
    summary = dict(
        status="validated" if passed else "partially_validated",
        passed=passed,
        complete_protocol=complete,
        backend=str(ti.lang.impl.current_cfg().arch),
        precision="float64",
        steps=int(result["step"]),
        particles=len(points),
        carried_soil_mass=observed[0],
        minimum_contact_clearance=observed[1],
        relative_mass_error=observed[2],
        reaction_force_error_N=observed[3],
        minimum_rigid_ground_clearance=observed[4],
        maximum_rigidity_error=observed[5],
        maximum_joint_anchor_error=observed[6],
        maximum_joint_angle_error_degrees=observed[7],
        minimum_affine_determinant=min(row[8] for row in rows),
        maximum_plastic_strain=max(row[4] for row in rows),
        minimum_accepted_timestep=min(row[5] for row in rows),
        retried_steps=sum(row[6] > 0 for row in rows),
    )
    evidence = dict(
        schema_version=1,
        task_status="completed",
        finite_state=True,
        capacity_overflow=False,
        solver_converged=bool(result["converged"]),
        timestep_consistent=timestep_consistent,
        command=" ".join(sys.argv),
        backend=summary["backend"],
        precision="float64",
        production_parameters=complete,
        checks=checks,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "validation-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if complete and not passed:
        raise RuntimeError("Excavation acceptance failed; inspect summary.json and trajectory.csv")


if __name__ == "__main__":
    main()
