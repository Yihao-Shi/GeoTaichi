"""Direct ULMPM–ABD excavation-to-truck cycle at two resolutions."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import qmc
from shapely import contains_xy
from shapely.ops import polygonize

from excavator_dig import CONTRACT, ROOT

PHASE_ENDS = {"lift": 2.125, "swing": 3.75, "dump": 4.875, "retract": 6.0}
PROFILES = {
    "low": dict(particles=180, grid_size=0.5, dt=0.01),
    "full": dict(particles=64000, grid_size=0.125, dt=0.005),
}


def load_solid(path):
    mesh = trimesh.load(path, merge_tex=True, merge_norm=True)
    if not mesh.is_volume:
        raise ValueError(f"Expected a closed, oriented solid: {path}")
    return mesh


def load_motion(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([int(row["frame"]) - 1 for row in rows]) / 24
    if not np.array_equal(times, np.arange(145) / 24) or not np.allclose(
        times, [float(row["time_s"]) for row in rows], rtol=0, atol=5.1e-7
    ):
        raise ValueError("Expected 145 trajectory frames from 0 to 6 s at 24 Hz")
    pins = np.array([[float(row[f"pin_{axis}_m"]) for axis in "xyz"] for row in rows])
    rotations = Slerp(times, Rotation.from_quat([[float(row[f"q{axis}"]) for axis in "xyzw"] for row in rows]))
    if not np.isfinite(pins).all():
        raise ValueError("Nonfinite trajectory pin")

    def pose(time):
        time = min(time, times[-1])
        return np.array([np.interp(time, times, p) for p in pins.T]), rotations(time)

    return pose


def bucket_cavity(mesh):
    # shortcut: follows the demo’s planar side panels, use a cavity mesh for other buckets.
    side_x = np.unique(mesh.vertices[:, 0])[1:-1]
    facet = np.flatnonzero(np.isclose(mesh.facets_origin[:, 0], side_x[0]) & (mesh.facets_normal[:, 0] > 0.99))
    (outline,) = polygonize(mesh.vertices[mesh.facets_boundary[facet.item()]][:, :, 1:])
    return side_x, outline


def sample_soil(mesh, count, minimum_z):
    lower, upper = mesh.bounds.copy()
    lower[2] = max(lower[2], minimum_z)
    if lower[2] >= upper[2]:
        raise ValueError("Initial soil has no volume above the ground clearance")
    sampler = qmc.Halton(3, scramble=True, seed=17)
    batches = []
    remaining = count
    for _ in range(10 + count // 128):
        candidates = qmc.scale(sampler.random(min(1024, 4 * remaining)), lower, upper)
        inside = candidates[mesh.contains(candidates)][:remaining]
        batches.append(inside)
        remaining -= len(inside)
        if remaining == 0:
            return np.concatenate(batches)
    raise ValueError("Could not sample the requested number of soil points above ground")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path(__file__).resolve().parent / "inputs")
    parser.add_argument("--resolution", choices=PROFILES, default="low")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--particles", type=int)
    parser.add_argument("--grid-size", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--stop-time", type=float, default=6.0)
    parser.add_argument("--contact-coordination-number", type=float, default=256 / 176)
    args = parser.parse_args()
    for key, value in PROFILES[args.resolution].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not (
        args.particles > 0
        and args.grid_size > 0
        and 0 < args.dt <= args.stop_time <= 6
        and np.isfinite(args.grid_size)
        and np.isfinite(args.contact_coordination_number)
        and args.contact_coordination_number > 0
    ):
        parser.error("Require positive capacities/grid size and 0 < dt <= stop-time <= 6 s")
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import geotaichi as gt
    import taichi as ti

    baseline = json.loads(CONTRACT.read_text())
    dt = args.dt
    contact = baseline["contact_or_coupling"]["parameters"]
    soil_material, steel = [material["parameters"] for material in baseline["materials"]]
    contact_dir = args.scene_dir / "contact"
    bucket, truck, soil = [
        load_solid(contact_dir / f"{name}.obj")
        for name in ("bucket_contact", "truck_bed_contact", "soil_initial_volume")
    ]
    side_x, cavity = bucket_cavity(bucket)
    # shortcut: the demo bed is a rectangular open box, use a cavity mesh for other beds.
    bed_inner_min, bed_inner_max = np.array([np.unique(truck.vertices[:, axis])[[1, 2]] for axis in range(3)]).T
    pose = load_motion(args.scene_dir / "trajectory.csv")
    start_pin, start_rot = pose(0)
    bucket_initial = bucket.copy()
    bucket_initial.vertices = start_pin + start_rot.apply(bucket.vertices)
    points = sample_soil(soil, args.particles, contact["dmin"])
    if trimesh.proximity.signed_distance(bucket_initial, points).max() >= -contact["dmin"]:
        raise ValueError("Initial soil sampling violates bucket contact clearance")
    volume = soil.volume / len(points)
    mass_per_point = soil_material["density"] * volume
    soil_top = soil.bounds[1, 2]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gt.init(arch=args.arch, default_fp="float64", log=False, offline_cache=True, cpu_max_num_threads=1)
    model = gt.MPDEM(coupling=False, log=False)
    mpm, dem = model.mpm, model.dem
    domain = [20, 20, 10]
    gravity = [0, 0, -9.81]
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
        grid_size=args.grid_size,
        xmin=[-8, -12, -0.5],
        xmax=[12, 8, 9.5],
        boundary_ids=np.arange(len(points), dtype=np.int32),
        surface_measure=volume ** (2 / 3),
    )
    mpm.add_body(body)
    ground = mpm.create_ground()
    ground.append([0, 0, 0], [0, 0, 1])
    mpm.add_ground(ground)
    mpm.add_material(model=baseline["materials"][0]["model"], **soil_material)
    mpm.add_element({"ElementSize": args.grid_size, "ShapeFunction": "Linear"})
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
            **baseline["capacities"]["allocations"],
            "point_triangle_coordination_number": 16384 / 176,
            "edge_edge_coordination_number": 4096 / 176,
            "max_material_number": 1,
            "max_rigid_template_number": 2,
            "surface_node_number": len(bucket.vertices) + len(truck.vertices),
            "body_coordination_number": 4,
            "wall_coordination_number": 4,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.add_attribute(0, steel)
    initial_anchors = []
    for body_id, (name, mesh) in enumerate((("bucket", bucket_initial), ("truck_bed", truck))):
        path = output / f"{name}.obj"
        mesh.export(path, include_normals=False, include_texture=False)
        dem.add_template(
            {"Name": name, "TemplateType": "AffineBody", "Object": gt.polyhedron(file=str(path)).reset(False)}
        )
        center = mesh.center_mass
        dem.create_body(
            {
                "BodyType": "AffineBody",
                "Template": {
                    "Name": name,
                    "BodyPoint": center.tolist(),
                    "MaterialID": 0,
                    "GroupID": 0,
                },
            }
        )
        anchors = center + np.vstack(([0, 0, 0], np.eye(3))) - 0.25
        initial_anchors.append(anchors)
        # Four noncoplanar spherical anchors drive all affine controls with a finite penalty.
        for anchor in anchors:
            dem.add_joint(
                {
                    "JointType": "Spherical",
                    "BodyID1": body_id,
                    "BodyID2": -1,
                    "WorldAnchor": anchor.tolist(),
                    "PositionStiffness": 1e9,
                }
            )
    dem.add_wall({"WallType": "Plane", "MaterialID": 0, "WallCenter": [0, 0, 0], "OuterNormal": [0, 0, 1]})
    dem.add_property(
        0,
        0,
        {"Dhat": contact["dhat"], "BarrierStiffness": contact["kappa"], "Friction": contact["friction_coefficient"]},
        dType="all",
    )
    solver = {
        **baseline["api"]["solver"],
        "Timestep": dt,
        "SimulationTime": args.stop_time,
        "SaveInterval": 0.125,
        "SavePath": str(output),
    }
    model.set_solver(solver, log=False)
    model.memory_allocate(
        {
            "body_coordination_number": 4,
            "contact_coordination_number": args.contact_coordination_number,
        },
        log=False,
    )
    model.choose_contact_model("BarrierIPC", **contact, friction_mode="lagged", friction_iterations=-1, project_pd=True)
    for affine in (0, 1):
        model.add_ipc_property(MPMbody=0, AffineBody=affine, property=contact)

    (output / "run-config.json").write_text(
        json.dumps(
            dict(
                source=str(args.scene_dir.resolve()),
                resolution=args.resolution,
                backend=str(ti.lang.impl.current_cfg().arch),
                precision="float64",
                particles=len(points),
                grid_size_m=args.grid_size,
                soil_volume_m3=soil.volume,
                soil_material=soil_material,
                steel=steel,
                contact=contact,
                solver=solver,
                affine_newton_displacement_cap_m=0.10,
                anchor_stiffness_N_per_m=1e9,
                contact_coordination_number=args.contact_coordination_number,
                max_mixed_pairs=math.ceil((len(bucket.faces) + len(truck.faces)) * args.contact_coordination_number),
            ),
            indent=2,
        )
        + "\n"
    )

    anchor_local = start_rot.inv().apply(initial_anchors[0] - start_pin)
    vertex_local = bucket_initial.vertices - initial_anchors[0][0]
    targets = np.vstack(initial_anchors)
    rows = []
    phase_masses = {}
    lifted = np.zeros(len(points), dtype=bool)

    def drive(system):
        pin, rotation = pose(system.mpm.time + system.mpm.dt)
        targets[:4] = pin + rotation.apply(anchor_local)
        # The public joint drive is revolute-only; update just the world-anchor targets, not body state.
        system.affine.joint_world_anchor.from_numpy(targets)

    def observe(system):
        t = model.sims.current_time
        positions = system.mpm.particle.x.to_numpy()
        controls = system.affine.y.to_numpy().reshape(2, 4, 3)
        gradient = (controls[:, 1:] - controls[:, :1]).transpose(0, 2, 1)
        pin, rotation = pose(t)
        actual = controls[0, 0] + vertex_local @ gradient[0].T
        tracking_error = np.linalg.norm(actual - (pin + rotation.apply(bucket.vertices)), axis=1).max()
        truck_error = np.linalg.norm(controls[1] - initial_anchors[1], axis=1).max()
        gap = system.last_step_record["contact"]["minimum_distance"]
        rigid_gap = system.affine.x.to_numpy()[:, 2].min()
        determinant = np.linalg.det(gradient).min()
        rigidity = np.linalg.norm(gradient.transpose(0, 2, 1) @ gradient - np.eye(3), axis=(1, 2)).max()
        soil_ground_gap = positions[:, 2].min()
        plastic = system.mpm.material.equivalent_plastic_strain.to_numpy()
        if (
            not np.isfinite(positions).all()
            or not np.isfinite(plastic).all()
            or not np.isfinite([tracking_error, truck_error, rigid_gap, determinant]).all()
            or not (
                gap > 0 and rigid_gap > 0 and soil_ground_gap > contact["dmin"] and determinant > 0 and rigidity < 0.01
            )
        ):
            raise RuntimeError(f"Nonfinite or infeasible cycle state at t={t:g}")
        bucket_points = start_rot.inv().apply(
            np.linalg.solve(gradient[0], (positions - controls[0, 0]).T).T + initial_anchors[0][0] - start_pin
        )
        in_bucket = (
            (bucket_points[:, 0] > side_x[0])
            & (bucket_points[:, 0] < side_x[1])
            & contains_xy(cavity, bucket_points[:, 1], bucket_points[:, 2])
        )
        truck_points = np.linalg.solve(gradient[1], (positions - controls[1, 0]).T).T + initial_anchors[1][0]
        in_bed = ((truck_points > bed_inner_min) & (truck_points < bed_inner_max)).all(axis=1)
        bucket_mass = float(in_bucket.sum() * mass_per_point)
        bed_mass = float(in_bed.sum() * mass_per_point)
        for phase, end in PHASE_ENDS.items():
            if phase not in phase_masses and t >= end - 1e-12:
                if phase == "lift":
                    lifted[:] = in_bucket & (positions[:, 2] > soil_top)
                phase_masses[phase] = dict(
                    time_s=t,
                    bucket_kg=bucket_mass,
                    bed_kg=bed_mass,
                    lifted_cohort_kg=float(lifted.sum() * mass_per_point),
                    lifted_cohort_in_bed_kg=float((lifted & in_bed).sum() * mass_per_point),
                    lifted_cohort_outside_bucket_and_bed_kg=float(
                        (lifted & ~in_bucket & ~in_bed).sum() * mass_per_point
                    ),
                )
        retries = system.mpm.history[-1]["step_retry"]["retry_count"]
        solve = system.last_step_record
        row = [
            t,
            system.mpm.dt,
            retries,
            system.last_contact_count,
            system.last_candidate_count,
            gap,
            rigid_gap,
            tracking_error,
            truck_error,
            determinant,
            float(plastic.max()),
            rigidity,
            soil_ground_gap,
            bucket_mass,
            bed_mass,
            solve["active_mpm_dof"],
            solve["newton_iterations"],
            solve["linear_iterations"],
            solve["friction_iterations"],
        ]
        rows.append(row)
        writer.writerow(row)
        if len(rows) % 20 == 0 or retries:
            print(
                f"cycle t={t:.6f} dt={row[1]:.3g} retries={retries} mixed={row[3]} bucket={bucket_mass:.1f} kg bed={bed_mass:.1f} kg",
                flush=True,
            )
        next_dt = min(dt, system.mpm.dt * 1.2) if len(rows) % 8 == 0 and retries == 0 else system.mpm.dt
        if next_dt != model.sims.delta:
            model.modify_parameters(SimulationTime=args.stop_time, Timestep=next_dt)

    with (output / "accepted-steps.csv").open("w", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "dt",
                "retries",
                "mixed_contacts",
                "mixed_candidates",
                "mixed_gap_m",
                "rigid_ground_gap_m",
                "bucket_vertex_tracking_error_m",
                "truck_anchor_error_m",
                "minimum_affine_determinant",
                "max_plastic_strain",
                "rigidity_error",
                "soil_ground_gap_m",
                "bucket_mass_kg",
                "truck_bed_mass_kg",
                "active_mpm_dof",
                "newton_iterations",
                "linear_iterations",
                "friction_iterations",
            ]
        )
        try:
            result = model.run(verbose=False, preintegration_function=drive, postprocessing=[observe])
        except Exception as error:
            system = model.enginer
            (output / "failure.json").write_text(
                json.dumps(
                    dict(
                        error=str(error),
                        time=model.sims.current_time,
                        steps=model.sims.current_step,
                        phases=phase_masses,
                        last_step=getattr(system, "last_step_record", None),
                        linear=getattr(system, "last_linear_solve", None),
                        attempts=getattr(getattr(system, "mpm", None), "last_failure", None),
                    ),
                    indent=2,
                )
                + "\n"
            )
            raise
    steps = np.asarray(rows)
    mass = model.enginer.mpm.particle.m.to_numpy().sum()
    if len(steps) != result["step"] or not np.isclose(steps[:, 1].sum(), args.stop_time, rtol=0, atol=1e-12):
        raise RuntimeError("Accepted steps did not cover the requested duration")
    expected_mass = len(points) * mass_per_point
    if not np.isclose(mass, expected_mass, rtol=1e-10, atol=0):
        raise RuntimeError(f"Soil mass changed from {expected_mass} kg to {mass} kg")
    summary = dict(
        complete_cycle=abs(result["time"] - 6.0) < 1e-12,
        time_s=result["time"],
        backend=str(ti.lang.impl.current_cfg().arch),
        precision="float64",
        particles=len(points),
        grid_size_m=args.grid_size,
        steps=len(steps),
        bucket_triangles=len(bucket.faces),
        truck_triangles=len(truck.faces),
        mass_kg=mass,
        minimum_dt_s=float(steps[:, 1].min()),
        retried_steps=int(np.count_nonzero(steps[:, 2])),
        peak_mixed_contacts=int(steps[:, 3].max()),
        peak_mixed_candidates=int(steps[:, 4].max()),
        minimum_mixed_clearance_m=float(steps[:, 5].min()) if steps[:, 3].any() else None,
        minimum_rigid_ground_clearance_m=float(steps[:, 6].min()),
        maximum_bucket_vertex_tracking_error_m=float(steps[:, 7].max()),
        maximum_truck_anchor_error_m=float(steps[:, 8].max()),
        minimum_affine_determinant=float(steps[:, 9].min()),
        maximum_plastic_strain=float(steps[:, 10].max()),
        maximum_rigidity_error=float(steps[:, 11].max()),
        minimum_soil_ground_gap_m=float(steps[:, 12].min()),
        relative_soil_mass_error=float(abs(mass - expected_mass) / expected_mass),
        phases=phase_masses,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
