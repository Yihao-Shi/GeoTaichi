"""Excavation and truck loading with explicit MPM and prescribed LSDEM bodies."""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
from shapely import contains_xy
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--source", type=Path)
parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
parser.add_argument("--stop-time", type=float, help="Duration after soil settling (s)")
parser.add_argument("--settle-damping", type=float, default=0.0, help="Grid damping applied before bucket release")
parser.add_argument("--dig-damping", type=float, default=0.0, help="Grid damping after bucket release")
parser.add_argument("--settle-max", type=float, default=15.0, help="Maximum time to wait for soil settling (s)")
parser.add_argument("--dt", type=float, default=1e-4)
parser.add_argument("--trajectory", type=Path)
parser.add_argument("--trench", type=Path, help="Use flat soil and MPM trough walls from this JSON file")
parser.add_argument("--resume-from", type=Path)
parser.add_argument("--resume-index", type=int)
parser.add_argument("--pic", type=float, default=0.0)
args = parser.parse_args()
if args.trajectory is None:
    args.trajectory = (args.trench.parent if args.trench else HERE / "inputs") / "trajectory.csv"
slowdown = 1.75
with args.trajectory.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
times = (np.array([int(row["frame"]) for row in rows]) - 1) / 24
targets = np.array(
    [[float(row[key]) for key in ("pin_x_m", "pin_y_m", "pin_z_m", "yaw_deg", "pitch_deg")] for row in rows]
)
if not np.array_equal(times, np.arange(len(rows)) / 24) or not np.isfinite(targets).all():
    parser.error("Expected finite poses at consecutive 24 Hz frames")
spline = CubicSpline(times, targets, bc_type="clamped")
full_duration = float(np.ceil(slowdown * times[-1] * 2 + 2) / 2)
cohort_start = min(t for t, row in zip(times, rows) if row["phase"] == "Lift")
cohort_end = min(t for t, row in zip(times, rows) if row["phase"] == "Dump")
if args.trench:
    cohort_start = min(t for t, row in zip(times, rows) if row["phase"] == "Curl")
if args.stop_time is None:
    args.stop_time = full_duration
if (args.resume_from is None) != (args.resume_index is None):
    parser.error("--resume-from and --resume-index must be used together")
if args.dt <= 0 or args.stop_time <= 0 or args.settle_max < 1.0:
    parser.error("Require a positive time step and cycle duration, and at least 1 s to settle")
if not 0 <= args.pic <= 1 or not all(0 <= damping < 1 for damping in (args.settle_damping, args.dig_damping)):
    parser.error("Require --pic in [0, 1] and both damping options in [0, 1)")
source = args.source.resolve() if args.source else HERE.parents[4]
sys.path.insert(0, str(source))
os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
if args.arch == "cpu":
    os.environ["GEOTAICHI_FORCE_CPU"] = "1"

import geotaichi as gt
import taichi as ti
from examples.mpdem.excavator_geometry import bucket_cavity, load_solid
from src.mpm.generator.InsertionKernel import kernel_apply_stress_from_file
from src.utils.Quaternion import set_to_rotate

count = 800_000
h = 0.125
cone = json.loads((HERE / "inputs/cone.json").read_text())
base_radius = cone["radius_m"]
height = base_radius * np.tan(np.deg2rad(cone["slope_deg"]))
inputs = HERE / "inputs"
contract = json.loads((inputs / "source-model-contract.json").read_text())
material, steel = (data["parameters"] for data in contract["materials"])
contact = contract["contact_or_coupling"]["parameters"]
bucket, truck = (load_solid(inputs / "contact" / (name + ".obj")) for name in ("bucket_contact", "truck_bed_contact"))
shape_function = "GIMP"
restart_config = dict(
    particles=count,
    grid_size_m=h,
    slowdown=slowdown,
    shape_function=shape_function,
    pic_fraction=args.pic,
    cohesion_pa=material["Cohesion"],
    tensile_cutoff_pa=material["Tensile"],
    pose_interpolation="clamped_cubic",
    floor_constraint="rough base with zero velocity on all nodes at or below z=0",
    trajectory_sha256=sha256(args.trajectory.read_bytes()).hexdigest(),
)


def pose(t):
    target = spline(min(t / slowdown, times[-1]))
    return target[:3], Rotation.from_euler("ZX", target[3:], degrees=True)


bed_min, bed_max = np.array([np.unique(truck.vertices[:, axis])[[1, 2]] for axis in range(3)]).T
lo = np.array([-8.0, -16.0, -0.5])
hi = np.array([12.0, 8.0, 9.5])
domain = hi - lo
wall_slabs = []
if args.trench:
    trench = json.loads(args.trench.read_text())
    minimum, size, counts = (np.asarray(trench[key]) for key in ("minimum_m", "size_m", "soil_counts"))
    maximum = minimum + size
    rim = trench["wall_height_m"]
    spacing = size / counts
    assert counts.prod() == count and np.allclose(spacing, h / 4)
    assert minimum[2] == 0 and maximum[2] < rim < hi[2]
    faces = np.r_[(minimum - lo) / h, (maximum - lo) / h, (rim - lo[2]) / h]
    assert np.allclose(faces, np.rint(faces))
    axes = [low + delta * (np.arange(n) + 0.5) for low, delta, n in zip(minimum, spacing, counts)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    volume = float(np.prod(spacing))
    cover = maximum[2] - points[:, 2]
    assert np.isclose(volume * count, np.prod(size)) and np.isclose(cover.min(), 0.5 * spacing[2])
    restart_config["trench"] = trench
    # Constrain exterior GIMP support nodes up to the rim and leave the interior free.
    for axis in range(2):
        for face, side in ((minimum[axis], -1), (maximum[axis], 1)):
            start, end = minimum.copy(), maximum.copy()
            start[:2] -= 2 * h
            end[:2] += 2 * h
            start[axis], end[axis] = sorted((face, face + side * 2 * h))
            end[2] = rim
            wall_slabs.append((axis, start, end))
else:
    restart_config["cone"] = dict(cone, height_m=float(height))
    # Sample the cone by cross-sectional area, then a uniform disk at each height.
    u, v, w = qmc.Halton(3, scramble=True, seed=17).random(count).T
    radial_scale = np.cbrt(1.0 - w)
    z = height * (1.0 - radial_scale)
    r = base_radius * radial_scale * np.sqrt(u)
    angle = 2.0 * np.pi * v
    points = np.column_stack((r * np.cos(angle), cone["center_y_m"] + r * np.sin(angle), z))
    volume = np.pi * base_radius**2 * height / (3.0 * count)
    spacing = np.full(3, volume ** (1 / 3))
    cover = height * (1.0 - np.hypot(points[:, 0], points[:, 1] - cone["center_y_m"]) / base_radius) - z
radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
if not np.isfinite(cover).all() or np.any(cover <= 0.0):
    raise ValueError("Every soil point must have a positive cover depth")
k0 = 1.0 - np.sin(np.deg2rad(material["FrictionAngle"]))
stress = -(material["density"] * 9.81 * cover[:, None]) * np.array([k0, k0, 1.0])
assert points.shape == stress.shape == (count, 3) and np.all((lo < points) & (points < hi))
settling = dict(rms_limit_m_s=0.01, p95_limit_m_s=0.025, hold_s=1.0, max_wait_s=args.settle_max)
if args.settle_damping:
    settling["background_damping"] = args.settle_damping
cycle_steps = round(args.stop_time / args.dt)
wait_steps = round(args.settle_max / args.dt)
for time, steps in ((args.stop_time, cycle_steps), (args.settle_max, wait_steps)):
    if not np.isclose(time, steps * args.dt, rtol=0, atol=1e-10):
        parser.error("Cycle duration and settling limit must be multiples of dt")
start_step = 0
release_step = None
quiet_samples = 0
if args.resume_from:
    args.resume_from = args.resume_from.resolve()
    config = json.loads((args.resume_from / "run-config.json").read_text())
    for key, value in dict(restart_config, dt_s=args.dt).items():
        if config.get(key) != value:
            parser.error(f"Restart requires the same {key}: expected {config.get(key)!r}, got {value!r}")
    if config.get("dig_background_damping", 0.0) != args.dig_damping:
        parser.error("Restart requires the same --dig-damping")
    name = f"MPMParticle{args.resume_index:06d}.npz"
    with np.load(args.resume_from / "particles" / name, allow_pickle=False) as data:
        start_time = float(data["t_current"])
    start_step = round(start_time / args.dt)
    if not (start_step > 0 and np.isclose(start_step * args.dt, start_time, rtol=0, atol=1e-10)):
        parser.error("The restart time must be a positive time step")
    if "settling" in config:
        if config["settling"] != settling:
            parser.error("Restart requires the same settling settings")
        with (args.resume_from / "progress.csv").open(newline="") as stream:
            history = [row for row in csv.DictReader(stream) if float(row["t"]) <= start_time + 1e-10]
        if history:
            quiet_samples = int(history[-1]["quiet_samples"])
            release_step = int(history[-1]["release_step"]) if history[-1]["release_step"] else None
    else:
        release_step = 0
end_step = wait_steps if release_step is None else release_step + cycle_steps
if start_step >= end_step:
    parser.error("Restart must precede the end of the settling or excavation stage")

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
gt.init(arch=args.arch, default_fp="float64", cpu_max_num_threads=1, log=False)
model = gt.MPDEM(log=False)
mpm, dem = model.mpm, model.dem
model.set_configuration(domain=domain.tolist(), gravity=[0, 0, -9.81], visualize=False, log=False)
mpm.set_configuration(
    solver_type="Explicit",
    configuration="ULMPM",
    mapping="USL",
    shape_function=shape_function,
    alphaPIC=args.pic,
    background_damping=args.settle_damping if release_step is None else args.dig_damping,
    log=False,
)
dem.set_configuration(scheme="LSDEM", search="LinkedCell", gravity=[0, 0, 0], visualize=False, log=False)
model.set_solver(
    dict(Timestep=args.dt, SimulationTime=end_step * args.dt, CFL=0.4, SaveInterval=5.0, SavePath=str(out)), log=False
)
nx, ny = np.rint(domain[:2] / h).astype(int) + 1
floor_guard_nodes = int(nx * ny * (round(-lo[2] / h) + 1))
wall_nodes = sum(int(np.prod(np.rint((end - start) / h).astype(int) + 1)) for _, start, end in wall_slabs)
mpm.memory_allocate(
    dict(
        max_material_number=1,
        max_particle_number=count,
        verlet_distance_multiplier=0.4,
        max_constraint_number={"max_velocity_constraint": 250000 + 3 * floor_guard_nodes + wall_nodes},
    ),
    log=False,
)
mpm.add_material(
    "DruckerPrager",
    dict(
        MaterialID=1,
        Density=material["density"],
        YoungModulus=material["young_modulus"],
        PoissonRatio=material["poisson_ratio"],
        Cohesion=material["Cohesion"],
        Tensile=material["Tensile"],
        Friction=material["FrictionAngle"],
        Dilation=material["DilationAngle"],
        dpType=material["dpType"],
    ),
)
mpm.add_element(dict(ElementType="R8N3D", ElementSize=[h] * 3))
if args.resume_from:
    mpm.read_restart(args.resume_index, str(args.resume_from), current_step=start_step)
else:
    cloud = np.column_stack((points - lo, np.full(count, volume), np.broadcast_to(0.5 * spacing, (count, 3))))
    particle_file = out / "soil.txt"
    np.savetxt(particle_file, cloud)
    mpm.add_body_from_file(dict(FileType="TXT", Template=dict(BodyID=0, MaterialID=1, ParticleFile=str(particle_file))))
    # The snapshot's text stress reader passes a filename to a dictionary helper.
    voigt = np.zeros((count, 6))
    voigt[:, :3] = stress
    kernel_apply_stress_from_file(0, count, voigt, mpm.scene.particle)
planes = [(0, lo[0], hi[2]), (0, hi[0], hi[2]), (1, lo[1], hi[2]), (1, hi[1], hi[2]), (2, hi[2], hi[2])]
for axis, coordinate, height in planes:
    start = lo.copy()
    end = hi.copy()
    end[2] = lo[2] + h * np.floor((height - lo[2]) / h + 1e-10)
    start[axis] = end[axis] = coordinate
    wall_slabs.append((axis, start, end))
for axis, start, end in wall_slabs:
    velocity = [None] * 3
    velocity[axis] = 0.0
    mpm.add_boundary_condition(
        dict(
            BoundaryType="VelocityConstraint",
            Velocity=velocity,
            StartPoint=(start - lo).tolist(),
            EndPoint=(end - lo).tolist(),
        )
    )
# Below a single constrained plane, GIMP can still transfer to unconstrained nodes.
mpm.add_boundary_condition(
    dict(
        BoundaryType="VelocityConstraint",
        Velocity=[0.0, 0.0, 0.0],
        StartPoint=[0.0, 0.0, 0.0],
        EndPoint=[*domain[:2], -lo[2]],
    )
)

dem.memory_allocate(
    dict(
        max_material_number=2,
        max_rigid_body_number=2,
        max_rigid_template_number=2,
        levelset_grid_number=1500000,
        surface_node_number=len(bucket.vertices) + len(truck.vertices),
        body_coordination_number=0,
        wall_coordination_number=0,
        verlet_distance_multiplier=[0.12, 0.1],
        point_coordination_number=[0, 0],
        compaction_ratio=[1.0, 1.0, 1.0, 1.0],
    ),
    log=False,
)
model.memory_allocate(
    dict(body_coordination_number=2, wall_coordination_number=0, compaction_ratio=[1.0, 1.0]), log=False
)
dem.add_attribute(0, dict(Density=steel["Density"], ForceLocalDamping=0.0, TorqueLocalDamping=0.0))
pin, rotation = pose(0.0)
# Use a 4 cm SDF to resolve the bucket walls. Pad enough for finite particle radii.
space = 0.04
extent = int(np.ceil(3 * radius / space)) + 4
frames = []
for name, mesh, center in [
    ("bucket", bucket, pin + rotation.apply(bucket.center_mass)),
    ("truck", truck, truck.center_mass),
]:
    _, frame = np.linalg.eigh(mesh.moment_inertia)
    frame[:, 0] *= np.linalg.det(frame)
    frames.append(frame)
    template_mesh = mesh.copy()
    template_mesh.vertices = (mesh.vertices - mesh.center_mass) @ frame
    path = out / f"{name}.obj"
    template_mesh.export(path, include_texture=False, include_normals=False)
    dem.add_template(
        dict(Name=name, Object=gt.polyhedron(file=str(path)).grids(space=space, extent=extent).reset(False))
    )
    dem.create_body(
        dict(
            GenerateType="Create",
            BodyType="RigidBody",
            Template=dict(
                Name=name,
                GroupID=0,
                MaterialID=0,
                BodyPoint=(center - lo).tolist(),
                ScaleFactor=1.0,
                BodyOrientation="constant",
            ),
        )
    )

dem.choose_contact_model(None, None)
model.choose_contact_model("Linear Model")
# k_n scales with particle length for this penalty sensitivity probe.
kn = 10 * material["young_modulus"] * volume ** (1 / 3)
model.add_property(
    0,
    1,
    dict(
        NormalStiffness=kn,
        TangentialStiffness=kn / 2,
        Friction=contact["friction_coefficient"],
        NormalViscousDamping=0.2,
        TangentialViscousDamping=0.2,
    ),
)
mpm.select_save_data()
dem.select_save_data(surface=True)
model.select_save_data(particle_particle_contact=True)
if args.resume_from:
    # Both LSDEM bodies are prescribed, so their geometry and state can be rebuilt.
    dem.sims.set_is_continue(True)
    dem.sims.current_print = args.resume_index
    dem.sims.current_time = mpm.sims.current_time
    model.read_restart(args.resume_index, str(args.resume_from), ppcontact=True)
model.add_essentials()
model.check_critical_timestep()
dt = model.sims.delta
if not np.isclose(dt, args.dt, rtol=0, atol=1e-12):
    raise RuntimeError("Prescribed motion requires the requested fixed time step")
print(f"cycle dt={dt:g} radius={radius:g} SDF={space:g} kn={kn:g} K0={k0:g} min_zz={stress[:, 2].min():g}", flush=True)

rigid = dem.scene.rigid
sphere = dem.scene.get_bounding_sphere()
initial_bounding = sphere.x.to_numpy() - rigid.mass_center.to_numpy()
vec3 = ti.types.vector(3, ti.f64)
vec4 = ti.types.vector(4, ti.f64)
contact_steps = ti.field(ti.i32, shape=())
first_contact_step = ti.field(ti.i32, shape=())
peak_force = ti.field(ti.f64, shape=())


@ti.kernel
def set_pose(body_id: int, center: vec3, q: vec4, v: vec3, w: vec3, relative_sphere: vec3, step: int):
    force = rigid[body_id].contact_force.norm()
    if step > 0 and force > 1e-12:
        if contact_steps[None] == 0:
            first_contact_step[None] = step
        contact_steps[None] += 1
        peak_force[None] = ti.max(peak_force[None], force)
    r = set_to_rotate(q)
    delta = center - rigid[body_id].mass_center
    sphere[body_id]._move(center + r @ relative_sphere - sphere[body_id].x, delta)
    rigid[body_id].mass_center = center
    rigid[body_id].q = q
    rigid[body_id].v = v
    rigid[body_id].w = w


def command(t):
    pin, rotation = pose(t)
    nxt_pin, nxt_rotation = pose(t + 0.25 * dt)
    center = pin + rotation.apply(bucket.center_mass) - lo
    next_center = nxt_pin + nxt_rotation.apply(bucket.center_mass) - lo
    return (
        center,
        (rotation * Rotation.from_matrix(frames[0])).as_quat(),
        (next_center - center) / (0.25 * dt),
        (nxt_rotation * rotation.inv()).as_rotvec() / (0.25 * dt),
    )


commands = [command(i * dt) for i in range(cycle_steps + 1)]
parked = (*commands[0][:2], np.zeros(3), np.zeros(3))


def bucket_command(step):
    if release_step is None or step <= release_step:
        return parked
    return commands[step - release_step]


set_pose(0, *bucket_command(start_step), initial_bounding[0], 0)
set_pose(
    1,
    truck.center_mass - lo,
    Rotation.from_matrix(frames[1]).as_quat(),
    np.zeros(3),
    np.zeros(3),
    initial_bounding[1],
    0,
)
if args.resume_from:
    for sims in (model.sims, mpm.sims, dem.sims):
        sims.current_step = start_step
        sims.CurrentTime[None] = start_time


# Prescribed kinematics are applied at the engine's rigid integration point.
# The mixed contact contribution to the MPM update still uses the current rigid pose and twist.
def update_rigid(*unused):
    step = model.sims.current_step
    set_pose(0, *bucket_command(step + 1), initial_bounding[0], step + 1)


model.enginer.dengine.integration = update_rigid
side_x, cavity = bucket_cavity(bucket)
snapshots = out / "snapshots"
snapshots.mkdir()
loaded = np.zeros(count, dtype=bool)
samples = max(1, round(0.1 / dt))


def in_bucket(t, x):
    pin, rotation = pose(t)
    local = rotation.inv().apply(x - pin)
    return (local[:, 0] > side_x[0]) & (local[:, 0] < side_x[1]) & contains_xy(cavity, local[:, 1], local[:, 2])


if args.resume_from and release_step is not None:
    for step in range(samples, start_step + 1, samples):
        cycle_t = (step - release_step) * dt
        if cohort_start <= cycle_t / slowdown <= cohort_end:
            loaded |= in_bucket(
                cycle_t, np.load(args.resume_from / "snapshots" / f"{step:06d}.npy", allow_pickle=False)
            )


def save_snapshot(step, x):
    final = snapshots / f"{step:06d}.npy"
    temporary = final.with_suffix(".partial.npy")
    np.save(temporary, x.astype(np.float32))
    temporary.replace(final)


save_snapshot(start_step, mpm.scene.particle.x.to_numpy()[:count] + lo)


def sample():
    global release_step, quiet_samples, end_step
    step = model.sims.current_step + 1
    if step % samples and step < end_step:
        return
    t = step * dt
    x = mpm.scene.particle.x.to_numpy()[:count] + lo
    v = mpm.scene.particle.v.to_numpy()[:count]
    particle_volume = mpm.scene.particle.vol.to_numpy()[:count]
    if (
        not np.isfinite(x).all()
        or not np.isfinite(v).all()
        or np.any((x < lo) | (x > hi))
        or not np.isfinite(particle_volume).all()
        or np.any(particle_volume <= 0)
    ):
        np.savez(out / "invalid-state.npz", x=x, v=v, volume=particle_volume, initial=points, time=t)
        raise RuntimeError(
            f"Invalid soil at {t:g}, nan x/v={(~np.isfinite(x)).sum()}/{(~np.isfinite(v)).sum()}, "
            f"min volume {np.nanmin(particle_volume):g}, "
            f"extents {np.nanmin(x, axis=0)} to {np.nanmax(x, axis=0)}, grid {lo} to {hi}"
        )
    speed = np.linalg.norm(v, axis=1)
    rms_speed = np.sqrt(np.mean(speed**2))
    p95_speed = np.quantile(speed, 0.95)
    if release_step is None:
        if rms_speed <= settling["rms_limit_m_s"] and p95_speed <= settling["p95_limit_m_s"]:
            quiet_samples += 1
        else:
            quiet_samples = 0
        # Count only complete intervals between consecutive qualifying observations.
        if (quiet_samples - 1) * samples * dt >= settling["hold_s"]:
            release_step = step
            end_step = step + cycle_steps
            model.sims.time = end_step * dt
            mpm.sims.set_background_damping(args.dig_damping)
            write_run_config()
            print(f"Soil settled at {t:.4f} s. Starting excavation.", flush=True)
    cycle_t = 0.0 if release_step is None else (step - release_step) * dt
    inside = in_bucket(cycle_t, x)
    if cohort_start <= cycle_t / slowdown <= cohort_end:
        loaded[:] |= inside
    # Allow penalty contact's sub-grid floor penetration in the bed count.
    bed = ((x >= bed_min - [0, 0, 0.025]) & (x <= bed_max)).all(axis=1)
    bucket_force, bed_force = np.linalg.norm(rigid.contact_force.to_numpy(), axis=1)
    latest[:] = (
        t,
        x[:, 2].max(),
        x[:, 2].min(),
        np.abs(v).max(),
        bucket_force,
        bed_force,
        int(inside.sum()),
        int(loaded.sum()),
        int(bed.sum()),
        int((loaded & bed).sum()),
        particle_volume.min(),
    )
    save_snapshot(step, x)
    writer.writerow([*latest, rms_speed, p95_speed, quiet_samples, release_step])
    print(
        f"t={t:.4f} s cycle={cycle_t:.4f} s rms_speed={rms_speed:.5g} p95_speed={p95_speed:.5g} "
        f"bucket={latest[6]} bed={latest[8]} delivered={latest[9]}",
        flush=True,
    )


def write_run_config():
    config = dict(
        restart_config,
        backend=str(ti.lang.impl.current_cfg().arch),
        dt_s=dt,
        duration_s=end_step * dt,
        settling=settling,
        dig_background_damping=args.dig_damping,
        release_time_s=None if release_step is None else release_step * dt,
        sdf_spacing_m=space,
        K0=k0,
        min_initial_zz_pa=float(stress[:, 2].min()),
        origin_m=lo.tolist(),
        maximum_m=hi.tolist(),
        snapshot_interval_s=samples * dt,
    )
    temporary = out / "run-config.partial.json"
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(out / "run-config.json")


write_run_config()
columns = (
    "t",
    "soil_max_z",
    "soil_min_z",
    "max_abs_velocity",
    "bucket_force_n",
    "bed_force_n",
    "bucket_points",
    "loaded_cohort",
    "bed_points",
    "delivered_cohort",
    "min_particle_volume_m3",
)
latest = []
with (out / "progress.csv").open("w", newline="", buffering=1) as stream:
    writer = csv.writer(stream)
    writer.writerow((*columns, "rms_speed_m_s", "p95_speed_m_s", "quiet_samples", "release_step"))
    model.solver.postprocess.append(sample)
    model.solver.coupling_solver(mpm.scene, dem.scene)
assert latest and np.isclose(latest[0], end_step * dt, rtol=0, atol=1e-10)
summary = dict(zip(columns, map(float, latest)))
summary.update(
    first_bucket_contact_s=first_contact_step[None] * dt if contact_steps[None] else None,
    peak_bucket_force_n=peak_force[None],
    release_time_s=None if release_step is None else release_step * dt,
    simulated_full_cycle=release_step is not None and args.stop_time == full_duration,
    verified_delivery=release_step is not None and args.stop_time == full_duration and latest[9] > 0,
)
(out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, indent=2))

if release_step is None:
    raise RuntimeError(f"Soil did not meet the settling criteria in {wait_steps * dt:g} s. Bucket remained parked.")
