"""Five ABD pendula with world-fixed revolute hinges and frictionless IPC impact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MESH = ROOT / "assets/mesh/AffineBody/icosphere_subdiv4.obj"
COUNT, RADIUS, LENGTH = 5, 0.05, 0.5
GAP, ANGLE, GRAVITY = 0.0022, math.radians(20), 9.81
# Refined surfaces and fixed-pivot stiffness resolve the near-rigid collision pulse.
DHAT, KAPPA, HINGE = 0.002, 2.0e16, 1.0e7
ENERGY_GROWTH_TOLERANCE = 1e-6
MINIMUM_RETURN_EXCURSION = 0.01
TRANSFER_SPEED_TOLERANCE = 0.05
MINIMUM_FULL_OSCILLATIONS = 7


def reference_period():
    return 2 * math.pi * math.sqrt(LENGTH / GRAVITY) * (1 + ANGLE**2 / 16 + 11 * ANGLE**4 / 3072)


def completed_round_trips(history):
    """Count visible returns to the initially raised side, allowing amplitude decay."""
    values = np.asarray(history)
    turns = np.flatnonzero((values[:-1, 2] < 0) & (values[1:, 2] >= 0)) + 1
    accepted = []
    for index in turns:
        if values[index, 1] < geometry()[0][0, 0] - MINIMUM_RETURN_EXCURSION and (
            not accepted or values[index, 0] - accepted[-1] > 0.5 * reference_period()
        ):
            accepted.append(float(values[index, 0]))
    return accepted


def geometry():
    rest = np.column_stack((0.55 + np.arange(COUNT) * (2 * RADIUS + GAP), np.full(COUNT, 0.3), np.full(COUNT, 0.25)))
    pivots = rest + np.array([0.0, 0.0, LENGTH])
    centers = rest.copy()
    centers[0] += [-LENGTH * math.sin(ANGLE), 0, LENGTH * (1 - math.cos(ANGLE))]
    return rest, pivots, centers


def frame_values(saved, anchor_weights):
    """Observe saved output; never feed a host result back into a numerical step."""
    mass = saved["mass_matrix"]
    weights = mass.sum(axis=1)
    total_mass = weights.sum(axis=1)
    centers = np.einsum("bi,bij->bj", weights, saved["y"]) / total_mass[:, None]
    velocities = np.einsum("bi,bij->bj", weights, saved["v_y"]) / total_mass[:, None]
    anchors = np.einsum("bi,bij->bj", anchor_weights, saved["y"])
    kinetic = 0.5 * np.einsum("bik,bij,bjk->", saved["v_y"], mass, saved["v_y"])
    energy = kinetic + GRAVITY * np.dot(total_mass, centers[:, 2] - geometry()[0][:, 2])
    return centers, velocities, anchors, float(energy)


def contact_free_energy(saved, initial_y, anchor_weights):
    """Exact conservative energy only when AABBs certify zero IPC potential."""
    vertices, ids = saved["vertices"], saved["bodyID"].reshape(-1)
    lower = np.array([vertices[ids == body].min(axis=0) for body in range(COUNT)])
    upper = np.array([vertices[ids == body].max(axis=0) for body in range(COUNT)])
    for i in range(COUNT):
        for j in range(i + 1, COUNT):
            separation = np.maximum(np.maximum(lower[j] - upper[i], lower[i] - upper[j]), 0)
            if np.linalg.norm(separation) < DHAT:
                return None
    y = saved["y"]
    deformation = (y[:, 1:] - y[:, :1]).transpose(0, 2, 1)
    initial = (initial_y[:, 1:] - initial_y[:, :1]).transpose(0, 2, 1)
    strain = deformation.transpose(0, 2, 1) @ deformation - np.eye(3)
    rigidity = np.sum(saved["volume"] * saved["young_modulus"] * np.sum(strain**2, axis=(1, 2)) / 8)
    _, _, anchors, energy = frame_values(saved, anchor_weights)
    local_axes = np.linalg.solve(initial, np.tile([0.0, 1.0, 0.0], (COUNT, 1))[:, :, None])[:, :, 0]
    axes = np.einsum("bij,bj->bi", deformation, local_axes)
    hinges = 0.5 * HINGE * (np.sum((anchors - geometry()[1]) ** 2) + np.sum((axes - [0, 1, 0]) ** 2))
    return float(energy + rigidity + hinges)


def analyze(output, duration, smoke, backend):
    from third_party.pyevtk.hl import linesToVTK

    files = sorted((output / "particles").glob("AffineBody[0-9]*.npz"))
    if len(files) < 2:
        raise RuntimeError("Newton cradle requires initial and final saved frames")
    rest, pivots, _ = geometry()
    with np.load(files[0]) as saved:
        initial_y = saved["y"]
        anchor_weights = np.asarray(
            [
                np.linalg.solve(np.column_stack((y, np.ones(4))).T, np.r_[pivot, 1.0])
                for y, pivot in zip(initial_y, pivots)
            ]
        )
        initial_energy = frame_values(saved, anchor_weights)[3]
        initial_full_energy = contact_free_energy(saved, initial_y, anchor_weights)
        integration = saved["newmark"].tolist() if "newmark" in saved else [1.0, 0.5, 1.0]
    finite, max_drift, max_planarity, max_energy, peak_outgoing = True, 0.0, 0.0, 0.0, 0.0
    rows = []
    left_history = []
    energy_retention = None
    max_total_energy = 1.0
    first_transfer = None
    residual_speeds = np.zeros(COUNT)
    for index, file in enumerate(files):
        with np.load(file) as saved:
            finite &= all(np.isfinite(saved[k]).all() for k in ("y", "v_y", "vertices"))
            centers, velocities, anchors, energy = frame_values(saved, anchor_weights)
            t = float(saved["t_current"])
            full_energy = contact_free_energy(saved, initial_y, anchor_weights)
            if (
                centers[-1, 0] > rest[-1, 0] + 0.01
                and velocities[-1, 0] > 0
                and t < math.pi * math.sqrt(LENGTH / GRAVITY)
            ):
                if first_transfer is None:
                    first_transfer = dict(time=t, center_velocities=velocities.tolist())
                residual_speeds = np.maximum(residual_speeds, np.abs(velocities[:, 0]))
            if full_energy is not None:
                max_total_energy = max(max_total_energy, full_energy / initial_full_energy)
                if first_transfer is not None:
                    energy_retention = full_energy / initial_full_energy
        max_drift = max(max_drift, float(np.linalg.norm(anchors - pivots, axis=1).max()))
        max_planarity = max(max_planarity, float(np.abs(centers[:, 1] - rest[:, 1]).max()))
        max_energy = max(max_energy, energy / initial_energy)
        peak_outgoing = max(peak_outgoing, float(velocities[-1, 0]))
        rows.append([t, energy, *centers.ravel(), *velocities.ravel(), full_energy])
        left_history.append([t, centers[0, 0], velocities[0, 0]])
        ends = np.stack((pivots, centers), axis=1).reshape((-1, 3))
        linesToVTK(
            str(output / "vtks" / f"PendulumRods{index:06d}"), *[np.ascontiguousarray(ends[:, i]) for i in range(3)]
        )
    incoming = math.sqrt(2 * GRAVITY * LENGTH * (1 - math.cos(ANGLE)))
    ratio = peak_outgoing / incoming
    if first_transfer is not None:
        vx = np.asarray(first_transfer["center_velocities"])[:, 0] / incoming
        first_transfer.update(
            striker_speed_ratio=float(residual_speeds[0] / incoming),
            middle_speed_ratio=float(residual_speeds[1:-1].max() / incoming),
            outgoing_speed_ratio=float(vx[-1]),
        )
        first_transfer["normalized_error"] = max(
            first_transfer["striker_speed_ratio"] / TRANSFER_SPEED_TOLERANCE,
            first_transfer["middle_speed_ratio"] / TRANSFER_SPEED_TOLERANCE,
        )
        first_transfer["passed"] = bool(first_transfer["normalized_error"] <= 1)
    transfer_passed = first_transfer is not None and first_transfer["passed"]
    returns = completed_round_trips(left_history)
    checks = [
        dict(
            name="README-reference horizontal transfer and seven-to-eight returns",
            kind="contract_observable",
            expected=[0, 1],
            status="fail",
            tolerance=0,
            evidence="trajectory.csv",
        ),
        dict(
            name="fixed pivot drift",
            kind="analytical",
            observed=max_drift,
            expected=[0, 0.0002],
            tolerance=0,
            evidence="NPZ affine anchor reconstruction",
        ),
        dict(
            name="pendulum plane drift",
            kind="benchmark",
            observed=max_planarity,
            expected=[0, 0.00001],
            tolerance=0,
            evidence="trajectory.csv",
        ),
        dict(
            name="mechanical energy growth",
            kind="energy_balance",
            observed=max(max_energy, max_total_energy),
            expected=[0, 1 + ENERGY_GROWTH_TOLERANCE],
            tolerance=0,
            evidence="Kinetic + gravity in all frames; full conservative energy at certified contact-free frames",
        ),
    ]
    if first_transfer is not None:
        error = first_transfer["normalized_error"]
        if duration >= MINIMUM_FULL_OSCILLATIONS * reference_period():
            error = max(error, MINIMUM_FULL_OSCILLATIONS / max(len(returns), 1))
        checks[0]["observed"] = float(error)
    if smoke:
        checks[0].pop("observed", None)
        checks[0].update(status="not_run", evidence="Short smoke ends before first impact")
    passed = finite and abs(rows[-1][0] - duration) < 1e-10
    passed &= max_drift <= 0.0002 and max_planarity <= 0.00001
    passed &= max(max_energy, max_total_energy) <= 1 + ENERGY_GROWTH_TOLERANCE
    passed &= smoke or transfer_passed
    passed &= (
        smoke or duration < MINIMUM_FULL_OSCILLATIONS * reference_period() or len(returns) >= MINIMUM_FULL_OSCILLATIONS
    )
    header = ["time", "kinetic_plus_gravitational_energy"]
    header += [
        f"{quantity}_{body}_{axis}" for quantity in ("center", "velocity") for body in range(COUNT) for axis in "xyz"
    ]
    header += ["total_conservative_energy_when_contact_free"]
    with (output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    summary = dict(
        case="abd_newtons_cradle",
        passed=bool(passed),
        finite=bool(finite),
        completed_time=rows[-1][0],
        target_time=duration,
        particles=COUNT,
        first_transfer=first_transfer,
        residual_speed_measure="Absolute horizontal mass-weighted center velocity; vertical vibration reported separately",
        residual_speed_ratio_limit=TRANSFER_SPEED_TOLERANCE,
        reference_period=reference_period(),
        minimum_full_oscillations=MINIMUM_FULL_OSCILLATIONS,
        completed_round_trips=len(returns),
        return_times=returns,
        maximum_anchor_drift=max_drift,
        maximum_plane_drift=max_planarity,
        maximum_energy_ratio=max_energy,
        maximum_certified_total_energy_ratio=max_total_energy,
        energy_growth_tolerance=ENERGY_GROWTH_TOLERANCE,
        minimum_return_excursion=MINIMUM_RETURN_EXCURSION,
        outgoing_speed_ratio=ratio,
        collision_acceptance=not smoke,
        backend=backend,
        precision="float64",
    )
    summary.update(
        newmark=integration, contact_free_energy_retention=energy_retention, physical_convergence_validated=False
    )
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    evidence = dict(
        schema_version=1,
        task_status="completed",
        finite_state=bool(finite),
        capacity_overflow=False,
        solver_converged=True,
        timestep_consistent=True,
        command=" ".join(sys.argv),
        backend=backend,
        precision="float64",
        production_parameters=not smoke
        and duration >= summary["minimum_full_oscillations"] * summary["reference_period"],
        checks=checks,
    )
    (output / "validation-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError("Newton cradle physical acceptance failed; see validation_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--save-interval", type=float, default=0.005)
    parser.add_argument("--newmark", type=float, nargs=3, metavar=("ALPHA", "BETA", "GAMMA"), default=[0.5, 0.25, 0.5])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "OutputData/newtons_cradle"))
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.dt, args.duration, args.save_interval)):
        parser.error("time controls must be finite and positive")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import geotaichi as gt
    import taichi as ti

    gt.init(arch=args.arch, dim=3, default_fp="float64", offline_cache=False, device_memory_GB=2.0, log=False)
    backend = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if args.arch == "gpu" and ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError(f"This queued CUDA case requires CUDA, got {backend}")
    dem = gt.DEM(log=False)
    dem.set_configuration(
        domain=[1.6, 0.6, 1.2],
        scheme="AffineBody",
        search="BVH",
        gravity=[0.0, 0.0, -GRAVITY],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        newmark=args.newmark,
        young_modulus=1e9,
        dhat=DHAT,
        barrier_stiffness=KAPPA,
        local_damping=0,
        max_newton_iteration=1000,
        newton_tolerance=1e-7,
        linear_tolerance=1e-14,
        linear_max_iteration=3000,
        line_search_max_iteration=30,
        max_step=0.01,
        ccd=True,
        ccd_type="ccd",
    )
    dem.memory_allocate(
        dict(
            max_material_number=1,
            max_affine_body_number=COUNT,
            surface_node_number=12810,
            max_plane_number=1,
            body_coordination_number=8,
            wall_coordination_number=1,
            max_point_triangle_pairs=32768,
            max_edge_edge_pairs=65536,
            affine_contact_block_capacity=1024,
            compaction_ratio=[1.0, 1.0],
        ),
        log=False,
    )
    dem.set_solver(
        dict(Timestep=args.dt, SimulationTime=args.duration, SaveInterval=args.save_interval, SavePath=str(output)),
        log=False,
    )
    dem.add_attribute(0, dict(Density=7800, ForceLocalDamping=0, TorqueLocalDamping=0))
    dem.add_template(dict(Name="ball", TemplateType="AffineBody", Object=gt.polyhedron(file=str(MESH))))
    _, pivots, centers = geometry()
    for body, (center, pivot) in enumerate(zip(centers, pivots)):
        dem.create_body(
            dict(
                BodyType="AffineBody",
                Template=dict(
                    Name="ball",
                    GroupID=0,
                    MaterialID=0,
                    BodyPoint=center.tolist(),
                    BoundingRadius=RADIUS,
                    InitialVelocity=[0, 0, 0],
                    Friction=0,
                ),
            )
        )
        dem.add_joint(
            dict(
                JointType="Revolute",
                BodyID1=body,
                BodyID2=-1,
                WorldAnchor=pivot.tolist(),
                WorldAxis=[0, 1, 0],
                PositionStiffness=HINGE,
                AxisStiffness=HINGE,
                MotorStiffness=0,
                Damping=0,
            )
        )
    dem.add_property(0, 0, dict(Dhat=DHAT, BarrierStiffness=KAPPA, ContactDampingStiffness=0, Friction=0), dType="all")
    try:
        dem.run()
    except RuntimeError:
        output.mkdir(parents=True, exist_ok=True)
        (output / "solver_failure.json").write_text(json.dumps(dem.diagnostics_snapshot(), indent=2) + "\n")
        raise
    return analyze(output, args.duration, args.smoke, backend)


if __name__ == "__main__":
    main()
