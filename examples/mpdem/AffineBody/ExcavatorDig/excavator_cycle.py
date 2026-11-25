"""Direct ULMPM–ABD excavation-to-truck cycle at two resolutions."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import qmc
from shapely import contains_xy
from shapely.ops import polygonize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONTRACT = HERE / "model-contract.json"
EXPLICIT = ROOT / "examples/mpdem/LevelSet/ExcavatorDig"
PHASE_ENDS = {"lift": 2.125, "swing": 3.75, "dump": 4.875, "retract": 6.0}
PROFILES = {
    "low": dict(particles=180, grid_size=0.5, dt=0.01),
    "full": dict(particles=150000, grid_size=0.25, dt=0.005),
}


def load_solid(path):
    mesh = trimesh.load(path, merge_tex=True, merge_norm=True)
    if not mesh.is_volume:
        raise ValueError(f"Expected a closed, oriented solid: {path}")
    return mesh


def bucket_cavity(mesh):
    # shortcut: follows the demo’s planar side panels, use a cavity mesh for other buckets.
    side_x = np.unique(mesh.vertices[:, 0])[1:-1]
    facet = np.flatnonzero(np.isclose(mesh.facets_origin[:, 0], side_x[0]) & (mesh.facets_normal[:, 0] > 0.99))
    (outline,) = polygonize(mesh.vertices[mesh.facets_boundary[facet.item()]][:, :, 1:])
    return side_x, outline


def box_points(minimum, size, counts):
    spacing = np.asarray(size) / counts
    axes = [low + (np.arange(n) + 0.5) * dx for low, n, dx in zip(minimum, counts, spacing)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return points, float(spacing.prod())


def geostatic_state(depth, material, gravity):
    """Return K0, principal Kirchhoff stresses and elastic Hencky deformation."""
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Every soil point must have an overlying free surface")
    k0 = 1.0 - np.sin(np.deg2rad(material["FrictionAngle"]))
    stress = -(material["density"] * gravity * depth[:, None]) * np.array([k0, k0, 1.0])
    young, poisson = material["young_modulus"], material["poisson_ratio"]
    log_stretch = ((1 + poisson) * stress - poisson * stress.sum(axis=1, keepdims=True)) / young
    return k0, stress, np.exp(log_stretch)[:, :, None] * np.eye(3)


def soil_boundary_dofs(lo, hi, h, container=None):
    """Fixed bottom and normal-only side supports, with an optional inner container."""
    shape = np.ceil((np.asarray(hi) - lo) / h).astype(int) + 1
    nx, ny, nz = shape
    z, y, x = np.ogrid[:nz, :ny, :nx]
    coords = [lo[0] + h * x, lo[1] + h * y, lo[2] + h * z]
    nodes = np.arange(nx * ny * nz, dtype=np.int32).reshape(nz, ny, nx)
    tol = 1e-9 * h
    floor = coords[2] <= tol
    fixed = []
    for axis, coord in enumerate(coords):
        mask = floor | (coord <= lo[axis] + tol) | (coord >= lo[axis] + h * (shape[axis] - 1) - tol)
        if container is not None and axis < 2:
            minimum, maximum, rim = container
            cross = 1 - axis
            within = (
                (coords[cross] >= minimum[cross] - h - tol)
                & (coords[cross] <= maximum[cross] + h + tol)
                & (coords[2] <= rim + tol)
            )
            sides = ((coord >= minimum[axis] - h - tol) & (coord <= minimum[axis] + tol)) | (
                (coord >= maximum[axis] - tol) & (coord <= maximum[axis] + h + tol)
            )
            mask = mask | (within & sides)
        fixed.append(3 * nodes[np.broadcast_to(mask, nodes.shape)] + axis)
    return fixed


def add_soil_boundary(mpm, lo, hi, h, container=None):
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    fixed = soil_boundary_dofs(lo, hi, h, container)
    boundary = DirichletBoundary()
    boundary.append(fixed, np.zeros(sum(map(len, fixed))))
    mpm.add_boundary_condition(dirichlet=boundary)


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


def soil_overburden(soil, points, material, gravity):
    """Return K0, principal Kirchhoff stresses and their elastic Hencky deformation."""
    depth = np.full(len(points), np.nan)
    for start in range(0, len(points), 1024):
        batch = points[start : start + 1024]
        hits, rays, _ = soil.ray.intersects_location(batch, np.tile([0, 0, 1], (len(batch), 1)), multiple_hits=False)
        depth[start + rays] = hits[:, 2] - batch[rays, 2]
    return geostatic_state(depth, material, gravity)


def paired_motion(path, settle, case):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = (np.array([int(row["frame"]) for row in rows]) - 1) / 24
    keys = ("pin_x_m", "pin_y_m", "pin_z_m", "yaw_deg", "pitch_deg")
    targets = np.array([[float(row[key]) for key in keys] for row in rows])
    if len(times) < 2 or not np.array_equal(times, np.arange(len(rows)) / 24) or not np.isfinite(targets).all():
        raise ValueError("Expected finite poses at consecutive 24 Hz frames")
    spline = CubicSpline(times, targets, bc_type="clamped")
    slowdown = 1.75
    duration = settle + math.ceil(2 * slowdown * times[-1] + 2) / 2
    phase_ends = {
        name: settle + slowdown * next(t for t, row in zip(times, rows) if row["phase"] == phase)
        for name, phase in (("lift", "Curl" if case == "trench" else "Lift"), ("swing", "Dump"), ("dump", "Retract"))
    }
    phase_ends["retract"] = duration

    def pose(time):
        target = spline(np.clip((time - settle) / slowdown, 0, times[-1]))
        return target[:3], Rotation.from_euler("ZX", target[3:], degrees=True)

    return pose, phase_ends, duration


def paired_soil(case, config, count):
    if case == "trench":
        minimum, size = (np.array(config[key]) for key in ("minimum_m", "size_m"))
        spacing = np.cbrt(size.prod() / count)
        counts = np.maximum(1, np.rint(size / spacing).astype(int))
        points, volume = box_points(minimum, size, counts)
        return points, volume, minimum[2] + size[2] - points[:, 2], (minimum, minimum + size, config["wall_height_m"])
    radius = config["radius_m"]
    height = radius * np.tan(np.deg2rad(config["slope_deg"]))
    u, v, w = qmc.Halton(3, scramble=True, seed=17).random(count).T
    scale = np.cbrt(1 - w)
    r = radius * scale * np.sqrt(u)
    angle = 2 * np.pi * v
    points = np.column_stack((r * np.cos(angle), config["center_y_m"] + r * np.sin(angle), height * (1 - scale)))
    return points, np.pi * radius**2 * height / (3 * count), height * scale * (1 - np.sqrt(u)), None


def checkpoint_fields(system):
    mpm, affine = system.mpm, system.affine
    fields = dict(
        x=mpm.particle.x,
        v=mpm.particle.v,
        a=mpm.particle.a,
        F0=mpm.F0,
        equivalent_plastic_strain=mpm.material.equivalent_plastic_strain,
        volumetric_plastic_strain=mpm.material.volumetric_plastic_strain,
        plastic_deformation_inverse=mpm.material.plastic_deformation_inverse,
        y=affine.y,
        velocity_y=affine.velocity_y,
        acceleration_y=affine.acceleration_y,
        previous_y=affine.previous_y,
        previous_velocity_y=affine.previous_velocity_y,
    )
    if mpm.velocity_proj:
        fields["gradv"] = mpm.gradv
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scene = parser.add_mutually_exclusive_group()
    scene.add_argument("--case", choices=("cone", "trench"), help="Paired explicit case (default: cone)")
    scene.add_argument("--scene-dir", type=Path, help="Legacy 6 s trajectory and contact solids")
    parser.add_argument("--resolution", choices=PROFILES, default="low")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--particles", type=int)
    parser.add_argument("--grid-size", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--stop-time", type=float)
    parser.add_argument("--contact-coordination-number", type=float, default=256 / 176)
    parser.add_argument(
        "--checkpoint-interval", type=float, default=0.05, help="Accepted simulation seconds between checkpoints"
    )
    parser.add_argument("--resume", type=Path, help="Read checkpoint.npz and continue in a new output directory")
    args = parser.parse_args()
    case = None if args.scene_dir else args.case or "cone"
    profile = PROFILES[args.resolution].copy()
    if case == "trench":
        profile["grid_size"] = 0.3125
    if not case and args.resolution == "full":
        profile.update(particles=64000, grid_size=0.125)
    settle = {"cone": 1.8, "trench": 1.1}.get(case, 0.0)
    if case:
        inputs = EXPLICIT / "explicit_cone/inputs"
        args.scene_dir = EXPLICIT / f"explicit_{case}/inputs"
        geometry_path = args.scene_dir / f"{case}.json"
        trajectory = args.scene_dir / "trajectory.csv"
        pose, phase_ends, duration = paired_motion(trajectory, settle, case)
    else:
        inputs = args.scene_dir
        trajectory = inputs / "trajectory.csv"
        pose, phase_ends, duration = load_motion(trajectory), PHASE_ENDS, 6.0
    if args.stop_time is None:
        args.stop_time = duration
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not (
        args.particles > 0
        and args.grid_size > 0
        and 0 < args.dt <= args.stop_time <= duration
        and np.isfinite(args.grid_size)
        and np.isfinite(args.contact_coordination_number)
        and args.contact_coordination_number > 0
        and math.isfinite(args.checkpoint_interval)
        and args.checkpoint_interval > 0
    ):
        parser.error(
            f"Require positive capacities/grid size/checkpoint interval and 0 < dt <= stop-time <= {duration:g} s"
        )
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import geotaichi as gt
    import taichi as ti

    contract_bytes = CONTRACT.read_bytes()
    baseline = json.loads(contract_bytes)
    dt = args.dt
    contact = baseline["contact_or_coupling"]["parameters"]
    soil_material, steel = [material["parameters"] for material in baseline["materials"]]
    contact_dir = inputs / "contact"
    files = [contact_dir / f"{name}.obj" for name in ("bucket_contact", "truck_bed_contact")]
    bucket, truck = map(load_solid, files)
    if case:
        material_path = inputs / "source-model-contract.json"
        soil_material, steel = [m["parameters"] for m in json.loads(material_path.read_text())["materials"]]
        soil_material.pop("Tensile")  # Zero cohesion places the Direct Drucker–Prager apex at zero pressure.
        files.extend((geometry_path, material_path))
    else:
        files.append(contact_dir / "soil_initial_volume.obj")
        soil = load_solid(files[-1])
    # The topology, particle order and anchor frame are rebuilt from the same case inputs.
    signature = hashlib.sha256(contract_bytes)
    for path in (trajectory, *files):
        signature.update(hashlib.sha256(path.read_bytes()).digest())
    signature.update(repr((args.particles, args.grid_size, dt, args.contact_coordination_number)).encode())
    if case:
        signature.update(repr(settle).encode())
    signature = signature.hexdigest()
    side_x, cavity = bucket_cavity(bucket)
    # shortcut: the demo bed is a rectangular open box, use a cavity mesh for other beds.
    bed_inner_min, bed_inner_max = np.array([np.unique(truck.vertices[:, axis])[[1, 2]] for axis in range(3)]).T
    start_pin, start_rot = pose(0)
    bucket_initial = bucket.copy()
    bucket_initial.vertices = start_pin + start_rot.apply(bucket.vertices)
    grid_min, grid_max = [-8, -16, -0.5], [12, 8, 9.5]
    gravity = [0, 0, -9.81]
    container = None
    if case:
        geometry = json.loads(geometry_path.read_text())
        points, volume, depth, container = paired_soil(case, geometry, args.particles)
        soil_top = float(np.max(points[:, 2] + depth))
        h = args.grid_size
        origin = container[0] if container else np.zeros(3)
        grid_min = (origin + h * np.floor((np.asarray(grid_min) - origin) / h)).tolist()
        if container:
            minimum, maximum, rim = container
            if not np.allclose((maximum[:2] - minimum[:2]) / h, np.rint((maximum[:2] - minimum[:2]) / h)):
                parser.error("Trench grid size must divide the 3.125 m width and 6.25 m length")
            container = minimum, maximum, round(rim / h) * h
        k0, initial_stress, initial_deformation = geostatic_state(depth, soil_material, -gravity[2])
    else:
        points = sample_soil(soil, args.particles, contact["dmin"])
        volume, soil_top = soil.volume / len(points), soil.bounds[1, 2]
        k0, initial_stress, initial_deformation = soil_overburden(soil, points, soil_material, -gravity[2])
    lo, hi = bucket_initial.bounds + np.array([[-contact["dmin"]], [contact["dmin"]]])
    near_bucket = points[((points >= lo) & (points <= hi)).all(axis=1)]
    if len(near_bucket) and trimesh.proximity.signed_distance(bucket_initial, near_bucket).max() >= -contact["dmin"]:
        raise ValueError("Initial soil sampling violates bucket contact clearance")
    mass_per_point = soil_material["density"] * volume
    domain = (np.array(grid_max) - grid_min).tolist()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gt.init(arch=args.arch, default_fp="float64", log=False, offline_cache=True, cpu_max_num_threads=1)
    model = gt.MPDEM(coupling=False, log=False)
    mpm, dem = model.mpm, model.dem
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
        xmin=grid_min,
        xmax=grid_max,
        boundary_ids=np.arange(len(points), dtype=np.int32),
        surface_measure=volume ** (2 / 3),
    )
    mpm.add_body(body)
    if case:
        add_soil_boundary(mpm, grid_min, grid_max, args.grid_size, container)
    else:
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
    model.set_solver({**solver, "initial_deformation_gradient": initial_deformation}, log=False)
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
                grid_bounds_m=[grid_min, grid_max],
                case=case,
                settle_time_s=settle,
                soil_volume_m3=volume * len(points),
                trench_grid_wall_top_m=container[2] if container else None,
                soil_initial_stress={"K0": k0, "minimum_vertical_kirchhoff_pa": float(initial_stress[:, 2].min())},
                soil_material=soil_material,
                steel=steel,
                contact=contact,
                solver=solver,
                affine_newton_displacement_cap_m=0.10,
                anchor_stiffness_N_per_m=1e9,
                contact_coordination_number=args.contact_coordination_number,
                max_mixed_pairs=math.ceil((len(bucket.faces) + len(truck.faces)) * args.contact_coordination_number),
                checkpoint_interval_s=args.checkpoint_interval,
                resumed_from=str(args.resume.resolve()) if args.resume else None,
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
    start_time = 0.0
    if args.resume:
        with np.load(args.resume, allow_pickle=False) as checkpoint:
            metadata = json.loads(checkpoint["metadata"].item())
            if metadata["version"] != 1 or metadata["signature"] != signature:
                raise ValueError(
                    "Checkpoint requires the same trajectory, geometry, material and discretization settings"
                )
            start_time = float(metadata["time"])
            next_dt = float(metadata["next_dt"])
            rows = checkpoint["rows"].tolist()
            if not (
                0 < start_time < args.stop_time - 1e-14
                and 0 < next_dt <= dt
                and len(rows) == metadata["step"]
                and rows
                and rows[-1][0] == start_time
            ):
                raise ValueError("Invalid checkpoint time, step history or requested stop time")
            lifted[:] = checkpoint["lifted"]
            phase_masses.update(metadata["phases"])
            model._add_direct_mpm_affine_ipc_essentials()
            system = model.enginer
            for name, field in checkpoint_fields(system).items():
                array = checkpoint[name]
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite checkpoint field {name}")
                field.from_numpy(array)
            # initial_simulation writes the first resumed VTU and would otherwise reset F0.
            system.mpm.initial_deformation_gradient = checkpoint["F0"]
            system.affine_acceleration_initialized = True
            system.mpm.time, system.mpm.step_count = start_time, len(rows)
            model.sims.CurrentTime[None] = start_time
            for sims in (model.sims, mpm.sims, dem.sims):
                sims.current_time, sims.current_step = start_time, len(rows)
            model.modify_parameters(SimulationTime=args.stop_time - start_time, Timestep=next_dt)
        print(f"Resuming accepted step {len(rows)} at t={start_time:.9g} s", flush=True)

    last_checkpoint_time = start_time

    def save_checkpoint(system, time, next_dt):
        metadata = dict(version=1, signature=signature, time=time, step=len(rows), next_dt=next_dt, phases=phase_masses)
        temporary = output / "checkpoint.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                metadata=json.dumps(metadata),
                rows=rows,
                lifted=lifted,
                **{name: field.to_numpy() for name, field in checkpoint_fields(system).items()},
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output / "checkpoint.npz")

    def drive(system):
        pin, rotation = pose(system.mpm.time + system.mpm.dt)
        targets[:4] = pin + rotation.apply(anchor_local)
        # The public joint drive is revolute-only; update just the world-anchor targets, not body state.
        system.affine.joint_world_anchor.from_numpy(targets)

    def observe(system):
        nonlocal last_checkpoint_time
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
                gap > 0
                and rigid_gap > 0
                and soil_ground_gap > (0 if case else contact["dmin"])
                and determinant > 0
                and rigidity < 0.01
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
        for phase, end in phase_ends.items():
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
        if t - last_checkpoint_time >= args.checkpoint_interval - 1e-12 or t >= args.stop_time - 1e-14:
            save_checkpoint(system, t, next_dt)
            last_checkpoint_time = t
        if next_dt != model.sims.delta:
            model.modify_parameters(SimulationTime=args.stop_time - start_time, Timestep=next_dt)

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
        writer.writerows(rows)
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
        complete_cycle=abs(result["time"] - duration) < 1e-12,
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
