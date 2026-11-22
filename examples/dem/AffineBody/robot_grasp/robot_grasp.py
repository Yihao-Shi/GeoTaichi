"""Pure ABD: close two hinged fingers, lift a free block, hold and release."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CONTRACT = Path(__file__).with_name("model-contract.json")


def smooth_motion(time, start, end):
    fraction = np.clip((time - start) / (end - start), 0.0, 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def joint_targets(time, motion):
    closed = smooth_motion(time, motion["settle_until"], motion["close_until"])
    closed *= 1.0 - smooth_motion(time, motion["hold_until"], motion["open_until"])
    raised = smooth_motion(time, motion["close_until"], motion["lift_until"])
    return np.r_[
        raised * motion["raised_arm_target_degrees"], closed * np.array(motion["closed_finger_targets_degrees"])
    ]


def differentiate_trajectory(dem, motion, duration, output, gradient_check, train_iterations):
    """Check full-trajectory derivatives and optimize a landing objective."""
    import taichi as ti

    steps = round(duration / dem.sims.delta)
    tape = dem.differentiable_affine(steps=steps)
    op = dem.enginer.operator
    times = (np.arange(steps) + 1) * tape.dt
    commands = np.array([joint_targets(time, motion) for time in times])
    seed = np.zeros((op.control_num, 3))
    velocity_seed = np.zeros_like(seed)
    seed[-4:] = np.array([1.0, 0.37, 0.11]) / 4
    velocity_seed[-4:] = np.array([0.05, -0.02, 0.0]) / 4
    history_length = len(dem.enginer.history)

    def forward(targets, name):
        # Diagnostic replay boundary: reset the saved device state, never host-integrate physics.
        tape._load_state(0)
        op.accepted_translation_velocity[None] = tape.translation_tape[0]
        dem.sims.current_time, dem.sims.current_step = tape.initial_time, tape.initial_step
        tape.record_count = 0
        del dem.enginer.history[history_length:]
        contacts = []
        for step, angles in enumerate(targets):
            dem.set_joint_target_angles([0, 1, 2], angles)
            tape.step()
            contacts.append(int(op.friction_contact_count[0]))
            if (step + 1) % 40 == 0:
                print(f"{name}: {step + 1}/{steps}", flush=True)
        return float(np.sum(seed * op.y.to_numpy()) + np.sum(velocity_seed * op.velocity_y.to_numpy())), contacts

    value, contacts = forward(commands, "baseline")
    terminal_y, terminal_v = op.y.to_numpy(), op.velocity_y.to_numpy()
    np.savez(
        output / "gradient-trajectory.npz",
        position=tape.position_tape.to_numpy(),
        velocity=tape.velocity_tape.to_numpy(),
        times=np.r_[0, times],
        commands=commands,
        friction_contacts=contacts,
    )
    if gradient_check:
        print("full trajectory backward", flush=True)
        gradient = tape.backward(seed, velocity_seed)["joint_target_angle_degrees_per_step"]
        np.testing.assert_array_equal(op.y.to_numpy(), terminal_y)
        np.testing.assert_array_equal(op.velocity_y.to_numpy(), terminal_v)
        np.save(output / "joint-target-gradient.npy", gradient)
    directions = {}
    arm = np.zeros_like(commands)
    arm[:, 0] = smooth_motion(times, motion["close_until"], motion["lift_until"])
    directions["arm_lift"] = arm
    for name, start, end in (
        ("finger_closing", motion["settle_until"], motion["close_until"]),
        ("finger_opening", motion["hold_until"], motion["open_until"]),
    ):
        direction = np.zeros_like(commands)
        phase = np.clip((times - start) / (end - start), 0, 1)
        direction[:, 1:] = np.sin(np.pi * phase)[:, None]
        directions[name] = direction
    result = dict(
        backend=str(ti.lang.impl.current_cfg().arch),
        precision="float64",
        steps=steps,
        dt=tape.dt,
        objective="payload center dot [1,.37,.11] + velocity dot [.05,-.02,0]",
        baseline_objective=value,
        linear_relative_tolerance=dem.sims.affine_linear_relative_tolerance,
        absolute_derivative_tolerance=1e-6,
        relative_derivative_tolerance=0.01,
        checks=[],
    )
    for name, direction in directions.items() if gradient_check else ():
        analytic = float(np.sum(gradient * direction))
        for epsilon in (0.01, 0.005):
            plus, _ = forward(commands + epsilon * direction, f"{name} +{epsilon}")
            minus, _ = forward(commands - epsilon * direction, f"{name} -{epsilon}")
            numerical = (plus - minus) / (2 * epsilon)
            error = abs(analytic - numerical)
            check = dict(
                direction=name,
                epsilon_degrees=epsilon,
                analytic=analytic,
                numerical=numerical,
                absolute_error=error,
                relative_error=error / max(abs(analytic), abs(numerical), 1e-30),
                passed=bool(error <= 1e-6 + 0.01 * max(abs(analytic), abs(numerical))),
            )
            result["checks"].append(check)
            print(json.dumps(check), flush=True)
            (output / "gradient-check.json").write_text(json.dumps(result, indent=2) + "\n")
    if gradient_check:
        result["passed"] = all(check["passed"] for check in result["checks"])
        (output / "gradient-check.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["passed"]:
            raise RuntimeError("Complete-trajectory gradient check failed; inspect gradient-check.json")
    if not train_iterations:
        return

    seed.fill(0.0)
    velocity_seed.fill(0.0)
    seed[-4:, 0] = 0.25
    amplitude = motion["raised_arm_target_degrees"]
    target_x = 1.18
    training = dict(target_payload_x=target_x, parameter="arm lift amplitude (degrees)", iterations=[])
    for iteration in range(train_iterations + 1):
        targets = commands.copy()
        targets[:, 0] = amplitude * arm[:, 0]
        payload_x, _ = forward(targets, f"training {iteration}")
        error = payload_x - target_x
        training["iterations"].append(dict(amplitude=amplitude, payload_x=payload_x, loss=0.5 * error**2))
        print(json.dumps(training["iterations"][-1]), flush=True)
        (output / "training.json").write_text(json.dumps(training, indent=2) + "\n")
        if iteration == train_iterations:
            break
        gradient = tape.backward(seed)["joint_target_angle_degrees_per_step"]
        derivative = float(np.sum(gradient * arm))
        if not np.isfinite(derivative) or abs(derivative) < 1e-8:
            raise RuntimeError("The landing objective has no resolved arm-control gradient")
        amplitude = float(np.clip(amplitude + np.clip(-error / derivative, -2.0, 2.0), -24.0, -12.0))
    positions = tape.position_tape.to_numpy().reshape(steps + 1, 4, 4, 3)
    centers = positions[:, 3].mean(axis=1)
    hold = (times >= motion["lift_until"] - 1e-10) & (times <= motion["hold_until"] + 1e-10)
    minimum_lift = float(centers[1:][hold, 2].min() - centers[0, 2])
    release_fall = float(centers[1:][hold][-1, 2] - centers[-1, 2])
    losses = [item["loss"] for item in training["iterations"]]
    training.update(
        minimum_hold_lift=minimum_lift,
        release_fall=release_fall,
        passed=bool(
            losses[-1] < 0.3 * losses[0]
            and np.all(np.diff(losses) < 0)
            and minimum_lift >= 0.12
            and release_fall >= 0.08
        ),
    )
    np.savez(output / "trained-trajectory.npz", position=positions, times=np.r_[0, times], commands=targets)
    (output / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    if not training["passed"]:
        raise RuntimeError("Trajectory optimization did not meet the predeclared loss and grasp checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--stop-time", type=float, default=1.6, help="Run a prefix of the unchanged motion protocol")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "OutputData/robot_grasp")
    parser.add_argument("--gradient-check", action="store_true", help="Check the full control trajectory VJP")
    parser.add_argument(
        "--train-iterations", type=int, default=0, help="After gradient checks, optimize the landing point"
    )
    args = parser.parse_args()
    if not np.isfinite(args.dt) or args.dt <= 0.0 or args.dt > 0.01:
        parser.error("dt must be finite in (0, 0.01] seconds")
    if not np.isfinite(args.stop_time) or not 0.0 < args.stop_time <= 1.6:
        parser.error("stop-time must be finite in (0, 1.6] seconds")
    if not np.isclose(args.stop_time / args.dt, round(args.stop_time / args.dt), rtol=0, atol=1e-9):
        parser.error("stop-time must be an integer number of time steps")
    if args.train_iterations < 0 or (args.train_iterations and args.stop_time != 1.6):
        parser.error("train-iterations must be nonnegative and training requires the complete 1.6 s protocol")

    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from geotaichi import DEM, init, polyhedron, python_callback
    import taichi as ti

    contract = json.loads(CONTRACT.read_text())
    geometry = contract["geometry"]["dimensions"]
    arm, fingers, motion = contract["boundary_conditions"][:3]
    contact = contract["contact_or_coupling"]["parameters"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    init(arch=args.arch, default_fp="float64", log=False, offline_cache=False, cpu_max_num_threads=2)
    dem = DEM(log=False)
    dem.set_configuration(**contract["api"]["configuration"], visualize=False, log=False)
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=1e7,
        dhat=contact["Dhat"],
        barrier_stiffness=contact["BarrierStiffness"],
        friction_epsv=contact["friction_epsv"],
        friction_iterations=-1,
        friction_max_iterations=50,
        friction_tolerance=1e-7,
        newton_tolerance=1e-7,
        max_newton_iteration=40,
        line_search_max_iteration=40,
        max_step=0.02,
        ccd_type="accd",
    )
    dem.memory_allocate(
        memory={
            **contract["capacities"]["allocations"],
            "max_material_number": 2,
            "surface_node_number": 32,
            "max_plane_number": 1,
            "body_coordination_number": 8,
            "wall_coordination_number": 4,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.stop_time,
            "SaveInterval": contract["time"]["save_interval"],
            "SavePath": str(output),
        },
        log=False,
    )
    for material_id, material in enumerate(contract["materials"]):
        dem.add_attribute(materialID=material_id, attribute=material["parameters"])
    for name in ("arm", "finger", "payload"):
        shape = polyhedron(file=str(ROOT / "assets/mesh/AffineBody/cube.obj")).reset(False)
        shape.mesh.apply_scale(geometry[name + "_size"])
        dem.add_template({"Name": name, "TemplateType": "AffineBody", "Object": shape})
    for body_id, (name, center) in enumerate(
        [
            ("arm", geometry["arm_center"]),
            *[("finger", point) for point in geometry["finger_centers"]],
            ("payload", geometry["payload_center"]),
        ]
    ):
        dem.create_body(
            {
                "BodyType": "AffineBody",
                "Template": {
                    "Name": name,
                    "BodyPoint": center,
                    "MaterialID": int(body_id == 3),
                    "GroupID": body_id,
                    "Friction": contact["Friction"],
                },
            }
        )
    for body_id, spec in enumerate((arm, fingers, fingers)):
        anchor = (
            arm["anchor"]
            if body_id == 0
            else [
                *geometry["finger_centers"][body_id - 1][:2],
                geometry["finger_centers"][body_id - 1][2] + 0.5 * geometry["finger_size"][2],
            ]
        )
        dem.add_joint(
            {
                "JointType": "Revolute",
                "BodyID1": body_id,
                "BodyID2": -1 if body_id == 0 else 0,
                "WorldAnchor": anchor,
                "WorldAxis": spec["axis"],
                "PositionStiffness": spec["position_stiffness"],
                "AxisStiffness": spec["axis_stiffness"],
                "MotorStiffness": spec["motor_stiffness"],
                "Damping": spec["damping"],
            }
        )
    dem.add_wall(
        {"WallType": "Plane", "MaterialID": 0, "WallCenter": [0, 0, geometry["table_z"]], "OuterNormal": [0, 0, 1]}
    )
    for first, second in ((0, 0), (0, 1), (1, 1)):
        dem.add_property(
            first, second, {key: contact[key] for key in ("Dhat", "BarrierStiffness", "Friction")}, dType="all"
        )
    assert len(dem.scene.affine_bodies) == 4 and len(dem.scene.affine_joints) == 3
    assert all(3 not in (joint["BodyID1"], joint["BodyID2"]) for joint in dem.scene.affine_joints)

    if args.gradient_check or args.train_iterations:
        dem.set_affine_body_parameters(linear_relative_tolerance=1e-10)
        differentiate_trajectory(dem, motion, args.stop_time, output, args.gradient_check, args.train_iterations)
        return

    rows = []
    complete = np.isclose(args.stop_time, motion["finish"])

    @python_callback
    def drive():
        dem.set_joint_target_angles([0, 1, 2], joint_targets(dem.sims.current_time + dem.sims.delta, motion))

    @python_callback
    def observe():
        # Read-only output boundary: physics and time integration stay in the ABD operator.
        state = dem.enginer.output_snapshot()
        controls = state["y"]
        deformation = (controls[:, 1:] - controls[:, :1]).transpose(0, 2, 1)
        center = state["center"][3]
        local = np.linalg.solve(deformation[0], center - controls[0, 0])
        rigidity = np.linalg.norm(deformation.transpose(0, 2, 1) @ deformation - np.eye(3), axis=(1, 2)).max()
        values = [
            dem.sims.current_time + dem.sims.delta,
            *center,
            *local,
            float(state["vertices"][:, 2].min() - geometry["table_z"]),
            float(np.linalg.det(deformation).min()),
            float(rigidity),
            float(dem.get_joint_states()["anchor_error"].max()),
        ]
        if not np.isfinite(values).all() or values[7] <= 0.0 or values[8] <= 0.0:
            raise RuntimeError("Grasp state is nonfinite, inverted or intersects the support plane")
        rows.append(values)
        writer.writerow(values)
        if len(rows) % 20 == 0:
            print(f"grasp t={values[0]:.3f} payload_z={center[2]:.6f} rigidity={rigidity:.3e}", flush=True)

    with (output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "payload_x",
                "payload_y",
                "payload_z",
                "local_x",
                "local_y",
                "local_z",
                "plane_gap",
                "minimum_determinant",
                "rigidity_error",
                "anchor_error",
            ]
        )
        try:
            dem.run(preintegration_function=drive, function=observe)
        except Exception:
            (output / "failure-diagnostics.json").write_text(json.dumps(dem.diagnostics_snapshot(), indent=2) + "\n")
            np.savez(output / "failed-accepted-state.npz", **dem.enginer.output_snapshot())
            raise

    values = np.asarray(rows)
    hold = values[(values[:, 0] >= motion["lift_until"] - 1e-10) & (values[:, 0] <= motion["hold_until"] + 1e-10)]
    checks = []
    observed = [None, None, None, float(values[:, 7].min())]
    if hold.size:
        observed[:2] = [
            float(hold[:, 3].min() - geometry["payload_center"][2]),
            float(np.linalg.norm(hold[:, 4:7] - hold[0, 4:7], axis=1).max()),
        ]
        if complete:
            observed[2] = float(hold[-1, 3] - values[-1, 3])
    for name, criterion, value in zip(
        ("payload lift", "hold drift", "release fall", "support-plane gap"),
        contract["validation"]["invariants"],
        observed,
    ):
        check = dict(
            name=name,
            kind=criterion["kind"],
            expected=criterion["expectation"],
            tolerance=criterion["tolerance"],
            evidence="trajectory.csv",
        )
        if value is None:
            check["status"] = "not_run"
        else:
            check["observed"] = value
            check["status"] = "pass" if check["expected"][0] <= value <= check["expected"][1] else "fail"
        checks.append(check)
    passed = complete and all(check["status"] == "pass" for check in checks) and values[:, 9].max() <= 0.01
    evidence = dict(
        schema_version=1,
        task_status="completed",
        finite_state=True,
        capacity_overflow=False,
        solver_converged=True,
        timestep_consistent=bool(len(rows) == round(args.stop_time / args.dt)),
        command=" ".join(sys.argv),
        backend=str(ti.lang.impl.current_cfg().arch),
        precision="float64",
        production_parameters=bool(complete and args.dt == contract["time"]["timestep"]),
        checks=checks,
    )
    summary = dict(
        status="validated" if passed and evidence["production_parameters"] else "partially_validated",
        passed=bool(passed),
        complete_protocol=bool(complete),
        steps=len(rows),
        timestep=args.dt,
        payload_mass=contract["materials"][1]["parameters"]["Density"] * np.prod(geometry["payload_size"]),
        minimum_hold_lift=observed[0],
        maximum_hold_drift=observed[1],
        release_fall=observed[2],
        minimum_plane_gap=observed[3],
        maximum_rigidity_error=float(values[:, 9].max()),
        maximum_anchor_error=float(values[:, 10].max()),
        backend=evidence["backend"],
    )
    (output / "validation-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if complete and not passed:
        raise RuntimeError("Grasp/lift/hold/release acceptance failed; inspect summary.json and trajectory.csv")


if __name__ == "__main__":
    main()
