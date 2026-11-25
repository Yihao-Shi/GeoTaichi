"""Prescribed screw-pile installation with explicit MPM soil and a rigid LSDEM pile."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import trimesh
from pile_archive import CHUNK, STATE_WIDTH, atomic_npz, save_frame, write_array

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUTS = HERE / "inputs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", choices=("full", "dense", "h200"), default="full")
    parser.add_argument("--stop-time", type=float)
    parser.add_argument("--mode", choices=("Normal", "Lightweight"), default="Lightweight")
    parser.add_argument("--dt", type=float, help="Override the explicit time step in seconds")
    parser.add_argument("--soil-counts", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--monitor-only", action="store_true", help="Write scalar diagnostics only")
    output_mode.add_argument("--positions-only", action="store_true", help="Write diagnostics and position snapshots")
    parser.add_argument("--snapshot-interval", type=float, help="Snapshot interval in simulation seconds")
    parser.add_argument(
        "--checkpoint", type=Path, help="Latest restart NPZ (default: OUTPUT/latest.npz in positions-only mode)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=float, default=0.25, help="Restart interval in simulation seconds"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue an existing output directory from its latest restart NPZ"
    )
    parser.add_argument(
        "--initial-overlap", type=float, default=0.0, help="Lower the pile by this distance for contact checks"
    )
    args = parser.parse_args()
    config = json.loads((INPUTS / "pile.json").read_text())
    setup = config["geometry"]
    profile = config["profiles"][args.resolution]
    installation = profile.get("seconds_per_turn", config["seconds_per_turn"]) * setup["turns"]
    settling = config["settle_s"]
    timing = dict(settle_s=settling, installation_s=installation, duration=settling + installation)
    dense = args.resolution != "full"
    size = np.asarray(setup["soil_size_m"])
    h = profile["grid_size_m"]
    cells = np.rint(size / h).astype(int)
    counts = np.array(args.soil_counts or profile["soil_counts"], dtype=np.int64)
    count = int(counts[0]) * int(counts[1]) * int(counts[2])
    if np.any(counts <= 0) or count >= 2**31:
        parser.error("Particle counts must be positive and fit in a signed 32-bit index")
    if np.any(counts > 2 * cells) or not np.allclose(cells * h, size, rtol=0.0, atol=1e-12):
        parser.error(
            "The soil box must align with the grid and have at most two initial points per cell along each axis"
        )
    if not np.isfinite(args.initial_overlap) or args.initial_overlap < 0:
        parser.error("Initial overlap must be finite and nonnegative")
    dt = profile["dt_s"] if args.dt is None else args.dt
    if not np.isfinite(dt) or dt <= 0:
        parser.error("Time step must be finite and positive")
    default_interval = 0.5 if dense else 0.25
    snapshot_interval = profile.get("snapshot_interval_s", default_interval)
    if args.snapshot_interval is not None:
        snapshot_interval = args.snapshot_interval
    if not np.isfinite(snapshot_interval) or snapshot_interval < dt:
        parser.error("Snapshot interval must be finite and at least one time step")
    snapshot_steps = round(snapshot_interval / dt)
    diagnostic_steps = round(default_interval / dt)
    stop = timing["duration"] if args.stop_time is None else args.stop_time
    steps = round(stop / dt)
    if not (1 <= steps and 0 < stop <= timing["duration"] and abs(steps * dt - stop) < 1e-10):
        parser.error("Stop time must be an integer number of steps within the installation duration")

    if args.resolution == "h200" and not (args.positions_only or args.monitor_only):
        parser.error("Use --positions-only or --monitor-only with the h200 profile")
    out = args.output_dir.resolve()
    checkpoint = args.checkpoint or (out / "latest.npz" if args.positions_only else None)
    if checkpoint and (args.mode != "Lightweight" or not (args.positions_only or args.monitor_only)):
        parser.error("Restart NPZ is supported with Lightweight and --positions-only or --monitor-only")
    if not np.isfinite(args.checkpoint_interval) or args.checkpoint_interval < dt:
        parser.error("Checkpoint interval must be finite and at least one time step")
    checkpoint_steps = round(args.checkpoint_interval / dt)
    if args.resume and (checkpoint is None or not checkpoint.is_file() or not out.is_dir()):
        parser.error("--resume requires an existing output directory and checkpoint")

    sys.path.insert(0, str(args.source.resolve()))
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    import geotaichi as gt
    import taichi as ti
    from src.utils.Quaternion import set_to_rotate

    material = config["soil"]
    gravity = config["gravity"]
    spacing = size / counts
    volume = float(spacing.prod())
    origin = np.array([-size[0] / 2 - 2 * h, -size[1] / 2 - 2 * h, -2 * h])
    domain = np.array([size[0] + 4 * h, size[1] + 4 * h, 32.0 - origin[2]])
    k0 = 1.0 - np.sin(np.deg2rad(material["FrictionAngle"]))
    gamma = material["density"] * gravity[2]
    mesh = trimesh.load_mesh(INPUTS / config["mesh"], merge_tex=True, merge_norm=True)
    if not mesh.is_volume:
        raise ValueError("The pile mesh must be a watertight oriented solid")
    poses = json.loads((INPUTS / "inclination_cases.json").read_text())
    initial = next(p for p in poses["poses"] if p["inclination_deg"] == setup["inclination_deg"])
    tilt = Rotation.from_matrix(initial["rotation_matrix_local_to_world"])
    axis = tilt.apply([0.0, 0.0, 1.0])
    travel = setup["pitch_m"] * setup["turns"]
    tip = 0.5 * travel * axis
    tip[2] = size[2] + poses["initial_mesh_clearance_m"] - tilt.apply(mesh.vertices)[:, 2].min()
    tip[2] -= args.initial_overlap
    center = mesh.center_mass
    _, principal = np.linalg.eigh(mesh.moment_inertia)
    principal[:, 0] *= np.linalg.det(principal)
    template = mesh.copy()
    template.vertices = (mesh.vertices - center) @ principal
    sdf_spacing = 0.04
    radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
    mpm_skin_ratio, dem_skin_ratio = 0.4, 0.12
    sphere = template.bounding_sphere.primitive
    bound_radius = max(sphere.radius, np.linalg.norm(template.vertices - sphere.center, axis=1).max())
    mpm_skin = mpm_skin_ratio * radius
    dem_skin = max(dem_skin_ratio * bound_radius, mpm_skin)
    padding = int(np.ceil((radius + mpm_skin + dem_skin) / sdf_spacing)) + 1
    sdf_capacity = int(np.prod(2 * (np.ceil(template.extents / (2 * sdf_spacing)).astype(int) + padding) + 1))
    signature = hashlib.sha256(
        json.dumps(
            [config, poses, args.resolution, counts.tolist(), dt, snapshot_steps, args.initial_overlap], sort_keys=True
        ).encode()
    )
    signature.update((INPUTS / config["mesh"]).read_bytes())
    restart_metadata = signature.hexdigest()
    initial_step = 0
    if args.resume:
        with np.load(checkpoint, allow_pickle=False) as archive:
            if archive["signature"].item() != restart_metadata or archive["version"].item() != 1:
                raise ValueError("Checkpoint does not match the inputs and output cadence")
            initial_step = int(archive["step"].item())
            if not 0 < initial_step < steps:
                raise ValueError("The requested stop time must follow the checkpoint")
    else:
        out.mkdir(parents=True, exist_ok=False)
        if checkpoint and checkpoint.exists():
            raise FileExistsError(checkpoint)
    snapshot_bytes = 0
    if args.positions_only:
        remaining = 1 + (steps - 1) // snapshot_steps - initial_step // snapshot_steps + int(not args.resume)
        snapshot_bytes = remaining * (6 * count + 128_000)
    # At most one LSDEM pile contact per soil point.
    checkpoint_bytes = count * (STATE_WIDTH + 5) * 8 + 256_000 if checkpoint else 0
    if checkpoint:
        checkpoint = checkpoint.resolve()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_budget = checkpoint_bytes * (1 if args.resume else 2)
    if checkpoint and out.stat().st_dev == checkpoint.parent.stat().st_dev:
        snapshot_bytes += checkpoint_budget
    elif checkpoint and checkpoint_budget > shutil.disk_usage(checkpoint.parent).free:
        raise RuntimeError(f"Restart NPZ rotation needs up to {checkpoint_budget / 2**30:.1f} GiB free")
    if snapshot_bytes > shutil.disk_usage(out).free:
        raise RuntimeError(f"Output needs up to {snapshot_bytes / 2**30:.1f} GiB free")
    mesh_path = out / "pile-local.obj"
    template.export(mesh_path, include_texture=False, include_normals=False)
    snapshots = out / "snapshots"
    if not args.monitor_only:
        snapshots.mkdir(exist_ok=args.resume)

    gt.init(arch=args.arch, default_fp="float64", cpu_max_num_threads=1, log=False)
    if args.arch == "gpu" and ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("This GPU case requires the CUDA backend")
    vec3 = ti.types.vector(3, ti.f64)
    vec4 = ti.types.vector(4, ti.f64)
    model = gt.MPDEM(log=False)
    mpm, dem = model.mpm, model.dem
    model.set_configuration(domain=domain.tolist(), gravity=gravity, visualize=False, log=False)
    mpm.set_configuration(
        solver_type="Explicit",
        configuration="ULMPM",
        mapping="USL",
        shape_function="GIMP",
        alphaPIC=0.1,
        mode=args.mode,
        log=False,
    )
    dem.set_configuration(scheme="LSDEM", search="LinkedCell", gravity=[0, 0, 0], visualize=False, log=False)
    model.set_solver(
        dict(Timestep=dt, SimulationTime=stop, CFL=0.4, SaveInterval=14.0 if dense else 2.0, SavePath=str(out)),
        log=False,
    )
    nx, ny, nz = np.rint(domain / h).astype(int) + 1
    # Three normal-constrained node layers cover each GIMP guard slab.
    constraint_capacity = int(3 * (nx * ny + 2 * nz * (nx + ny)))
    mpm.memory_allocate(
        dict(
            max_material_number=1,
            max_particle_number=count,
            verlet_distance_multiplier=mpm_skin_ratio,
            max_constraint_number={"max_velocity_constraint": constraint_capacity},
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
            Friction=material["FrictionAngle"],
            Dilation=material["DilationAngle"],
            dpType=material["dpType"],
        ),
    )
    mpm.add_element(dict(ElementType="R8N3D", ElementSize=[h] * 3))
    particle = mpm.scene.particle
    soil_min = np.array([-size[0] / 2, -size[1] / 2, 0.0]) - origin

    @ti.kernel
    def seed_soil(
        lower: vec3,
        cell: vec3,
        ny: int,
        nz: int,
        density: float,
        particle_volume: float,
        overburden: float,
        soil_depth: float,
        lateral_ratio: float,
    ):
        for p in range(count):
            index = ti.Vector([p // (ny * nz), p // nz % ny, p % nz])
            offset = (index.cast(ti.f64) + 0.5) * cell
            particle[p]._set_essential(
                p, 0, 1, density, particle_volume, lower + offset, ti.Vector.zero(float, 3), ti.Vector.zero(ti.u8, 3)
            )
            zz = overburden * (soil_depth - offset[2])
            particle[p].stress = ti.Vector([lateral_ratio * zz, lateral_ratio * zz, zz, 0.0, 0.0, 0.0])

    seed_soil(soil_min, spacing, int(counts[1]), int(counts[2]), material["density"], volume, gamma, size[2], k0)
    mpm.scene.psize = (0.5 * spacing)[None, :]
    if not (args.monitor_only or args.positions_only):
        mpm.scene.psize = np.broadcast_to(mpm.scene.psize, (count, 3)).copy()
    mpm.scene.material.state_vars_initialize(1, 0, count, particle)
    mpm.scene.particleNum[0] = count
    if args.mode == "Normal":
        mpm.scene.material.update_material_mapping(particle, count)
    for direction in range(3):
        velocity = [None] * 3
        velocity[direction] = 0.0
        for upper in ([False, True] if direction < 2 else [False]):
            start, end = np.zeros(3), domain.copy()
            if upper:
                start[direction] = domain[direction] - 2 * h
            else:
                end[direction] = 2 * h
            mpm.add_boundary_condition(
                dict(
                    BoundaryType="VelocityConstraint",
                    Velocity=velocity,
                    StartPoint=start.tolist(),
                    EndPoint=end.tolist(),
                )
            )

    dem.memory_allocate(
        dict(
            max_material_number=2,
            max_rigid_body_number=1,
            max_rigid_template_number=1,
            levelset_grid_number=sdf_capacity,
            surface_node_number=len(mesh.vertices),
            body_coordination_number=0,
            wall_coordination_number=0,
            verlet_distance_multiplier=[dem_skin_ratio, 0.1],
            point_coordination_number=[0, 0],
            compaction_ratio=[1.0, 1.0, 1.0, 1.0],
        ),
        log=False,
    )
    model.memory_allocate(
        dict(body_coordination_number=1, wall_coordination_number=0, compaction_ratio=[1.0, 1.0]), log=False
    )
    dem.add_attribute(0, dict(Density=config["steel_density"], ForceLocalDamping=0.0, TorqueLocalDamping=0.0))
    shape = gt.polyhedron(file=str(mesh_path)).grids(space=sdf_spacing, extent=padding).reset(False)
    # A BVH keeps the fine signed-distance grid from materializing mesh-wide candidate lists.
    raycast = o3d.t.geometry.RaycastingScene(nthreads=4)
    raycast.add_triangles(
        o3d.core.Tensor((shape.mesh.vertices - shape.mesh.center_mass).astype(np.float32)),
        o3d.core.Tensor(shape.mesh.faces.astype(np.uint32)),
    )

    def signed_distance(query):
        return raycast.compute_signed_distance(
            o3d.core.Tensor(query.astype(np.float32)), nthreads=4, nsamples=3
        ).numpy()

    dem.add_template(dict(Name="pile", Object=shape.analytical_distance(signed_distance)))
    dem.create_body(
        dict(
            GenerateType="Create",
            BodyType="RigidBody",
            Template=dict(
                Name="pile",
                GroupID=0,
                MaterialID=0,
                BodyPoint=(tip + tilt.apply(center) - origin).tolist(),
                ScaleFactor=1.0,
                BodyOrientation="constant",
            ),
        )
    )
    dem.choose_contact_model(None, None)
    model.choose_contact_model("Linear Model")
    kn = 10 * material["young_modulus"] * volume ** (1 / 3)
    model.add_property(
        0,
        1,
        dict(
            NormalStiffness=kn,
            TangentialStiffness=kn / 2,
            Friction=config["friction_coefficient"],
            NormalViscousDamping=0.2,
            TangentialViscousDamping=0.2,
        ),
    )
    mpm.select_save_data()
    dem.select_save_data(surface=True)
    model.select_save_data(particle_particle_contact=True)
    model.add_essentials()
    model.check_critical_timestep()
    if not np.isclose(model.sims.delta, dt, rtol=0, atol=1e-12):
        raise RuntimeError("Prescribed motion requires the requested fixed time step")

    times = dt * np.arange(steps + 1)
    phase = np.clip((times - timing["settle_s"]) / timing["installation_s"], 0.0, 1.0)
    progress = phase**2 * (3.0 - 2.0 * phase)
    progress_rate = 6.0 * phase * (1.0 - phase) / timing["installation_s"]
    spin = np.column_stack((np.zeros_like(times), np.zeros_like(times), -2 * np.pi * setup["turns"] * progress))
    rotation = tilt * Rotation.from_rotvec(spin)
    center_from_tip = rotation.apply(center)
    positions = tip - origin - progress[:, None] * travel * axis + center_from_tip
    quaternions = (rotation * Rotation.from_matrix(principal)).as_quat()
    angular_velocity = -2 * np.pi * setup["turns"] * (progress_rate[:, None] * axis)
    velocity = -travel * (progress_rate[:, None] * axis) + np.cross(angular_velocity, center_from_tip)
    rigid = dem.scene.rigid
    bounds = dem.scene.get_bounding_sphere()
    sphere_offset = bounds.x.to_numpy()[0] - rigid.mass_center.to_numpy()[0]

    @ti.kernel
    def set_pose(position: vec3, q: vec4, v: vec3, w: vec3, relative_sphere: vec3):
        sphere_center = position + set_to_rotate(q) @ relative_sphere
        bounds[0]._move(sphere_center - bounds[0].x, position - rigid[0].mass_center)
        rigid[0].mass_center = position
        rigid[0].q = q
        rigid[0].v = v
        rigid[0].w = w

    def set_step(step):
        set_pose(positions[step], quaternions[step], velocity[step], angular_velocity[step], sphere_offset)

    def update_rigid(*unused):
        set_step(model.sims.current_step + 1)

    model.enginer.dengine.integration = update_rigid
    set_step(initial_step)
    if args.monitor_only or args.positions_only:

        def update_save_time(*_):
            model.solver.last_save_time = model.sims.current_time

        model.solver.save_file = update_save_time
    (out / "run-config.json").write_text(
        json.dumps(
            dict(
                backend=str(ti.lang.impl.current_cfg().arch),
                soil_size_m=size.tolist(),
                particles=count,
                soil_counts=counts.tolist(),
                dt_s=dt,
                snapshot_interval_s=snapshot_steps * dt,
                origin_m=origin.tolist(),
                stop_time_s=stop,
                geometry=setup,
                timing=timing,
                start_tip_m=tip.tolist(),
                tilt=tilt.as_quat().tolist(),
                grid_size_m=h,
                shape_function="GIMP",
                pic_fraction=0.1,
                mode=args.mode,
                sparse_grid=False,
                initial_overlap_m=args.initial_overlap,
                material_model="DruckerPrager",
                max_initial_points_per_cell=int(np.prod((counts + cells - 1) // cells)),
                k0=k0,
                min_initial_zz_pa=float(gamma * (size[2] - 0.5 * spacing[2])),
                sdf_spacing_m=sdf_spacing,
            ),
            indent=2,
        )
        + "\n"
    )
    columns = (
        "time_s",
        "advance_m",
        "turns",
        "pile_force_n",
        "min_volume_m3",
        "min_z_m",
        "max_z_m",
        "max_speed_m_s",
        "min_zz_pa",
        "max_verlet_m",
        "elapsed_s",
    )
    last = []

    diagnostics = ti.field(float, shape=6)
    invalid = ti.field(ti.i32, shape=())

    @ti.kernel
    def measure(limit: vec3):
        for p in range(count):
            x, v, vol, stress = particle[p].x, particle[p].v, particle[p].vol, particle[p].stress
            if (
                not (vol > 0 and vol < float("inf"))
                or not all((x >= 0) & (x <= limit))
                or not all(ti.abs(v) < float("inf"))
                or not all(ti.abs(stress) < float("inf"))
                or particle[p].active == 0
            ):
                invalid[None] = 1
            ti.atomic_min(diagnostics[0], vol)
            ti.atomic_min(diagnostics[1], x[2])
            ti.atomic_max(diagnostics[2], x[2])
            ti.atomic_max(diagnostics[3], v.norm())
            ti.atomic_min(diagnostics[4], stress[2])
            ti.atomic_max(diagnostics[5], particle[p].verletDisp.norm())

    @ti.kernel
    def copy_positions(start: int, scale: vec3, positions: ti.types.ndarray(dtype=ti.u16, ndim=2)):
        for p, d in positions:
            positions[p, d] = ti.cast(ti.floor(particle[start + p].x[d] / scale[d] + 0.5), ti.u16)

    def save_positions(step):
        save_frame(snapshots / f"{step:06d}.npz", count, origin, domain / 65535, copy_positions)

    engine = model.enginer
    contacts = engine.physpp.cplist
    offsets = engine.neighbor.hist_particle_particle
    state = mpm.scene.material.stateVars

    @ti.kernel
    def transfer_state(start: int, block: ti.types.ndarray(dtype=ti.f64, ndim=2), restore: ti.template()):
        for i in range(block.shape[0]):
            p = start + i
            if ti.static(restore):
                for d in ti.static(range(3)):
                    particle[p].x[d] = block[i, d]
                    particle[p].v[d] = block[i, 3 + d]
                for d in ti.static(range(6)):
                    particle[p].stress[d] = block[i, 6 + d]
                particle[p].vol = block[i, 12]
                state[p].epdstrain = block[i, 13]
            else:
                for d in ti.static(range(3)):
                    block[i, d] = particle[p].x[d]
                    block[i, 3 + d] = particle[p].v[d]
                for d in ti.static(range(6)):
                    block[i, 6 + d] = particle[p].stress[d]
                block[i, 12] = particle[p].vol
                block[i, 13] = state[p].epdstrain

    @ti.kernel
    def read_contact(start: int, block: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        for i in range(block.shape[0]):
            c = start + i
            block[i, 0] = contacts[c].endID1
            block[i, 1] = contacts[c].normalOverlap
            for d in ti.static(range(3)):
                block[i, 2 + d] = contacts[c].oldTangOverlap[d]
            if contacts[c].normalOverlapActive == 0:
                block[i, 0] = -1.0

    @ti.kernel
    def restore_contact(block: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        for i in range(block.shape[0]):
            p = ti.cast(block[i, 0], ti.i32)
            for c in range(offsets[p], offsets[p + 1]):
                contacts[c].normalOverlap = block[i, 1]
                contacts[c].normalOverlapActive = 1
                for d in ti.static(range(3)):
                    contacts[c].oldTangOverlap[d] = block[i, 2 + d]

    def save_checkpoint(step):
        with atomic_npz(checkpoint) as archive:
            for name, value in [("version", 1), ("step", step), ("signature", restart_metadata)]:
                write_array(archive, name, np.asarray(value))
            for start in range(0, count, CHUNK):
                block = np.empty((min(CHUNK, count - start), STATE_WIDTH), dtype=np.float64)
                transfer_state(start, block, False)
                write_array(archive, f"state_{start}", block)
            for start in range(0, offsets[count], CHUNK):
                block = np.empty((min(CHUNK, offsets[count] - start), 5), dtype=np.float64)
                read_contact(start, block)
                write_array(archive, f"contact_{start}", block[block[:, 0] >= 0])

    if args.resume:
        with np.load(checkpoint, allow_pickle=False) as archive:
            for start in range(0, count, CHUNK):
                block = archive[f"state_{start}"]
                if block.dtype != np.float64 or block.shape != (min(CHUNK, count - start), STATE_WIDTH):
                    raise ValueError("Invalid checkpoint particle block")
                transfer_state(start, block, True)
            # Rebuild the broad phase without running a force evaluation twice.
            engine.mengine.pre_calculation(model.mpm.sims, mpm.scene, engine.mneighbor)
            engine.dengine.pre_calculation(model.dem.sims, dem.scene, engine.dneighbor)
            bounds.verletDisp.fill(0.0)
            engine.update_verlet_table()
            for name in archive.files:
                if name.startswith("contact_"):
                    block = archive[name]
                    if (
                        block.ndim != 2
                        or block.shape[1] != 5
                        or block.dtype != np.float64
                        or not np.isfinite(block).all()
                        or np.any(block[:, 0] < 0)
                        or np.any(block[:, 0] >= count)
                        or np.any(block[:, 0] != np.floor(block[:, 0]))
                    ):
                        raise ValueError("Invalid checkpoint contact block")
                    restore_contact(block)
        engine.pre_calculate = lambda: None
        for sims in (model.sims, mpm.sims, dem.sims):
            sims.current_step = initial_step
            sims.current_time = times[initial_step]
            sims.CurrentTime[None] = times[initial_step]
        model.solver.last_save_time = times[initial_step]
        print(f"Resumed step={initial_step} t={times[initial_step]:.6f} s", flush=True)

    if not args.monitor_only and not args.resume:
        save_positions(0)
    start_time = time.perf_counter()
    next_diagnostic = start_time

    def sample():
        nonlocal next_diagnostic
        step = model.sims.current_step + 1
        snapshot_due = not args.monitor_only and (step % snapshot_steps == 0 or step == steps)
        checkpoint_due = checkpoint and (step == 1 or step % checkpoint_steps == 0 or step == steps)
        if not (snapshot_due or checkpoint_due) and step != 1 and step % diagnostic_steps and step < steps:
            if time.perf_counter() < next_diagnostic:
                return
        next_diagnostic = time.perf_counter() + 60.0
        diagnostics.from_numpy(np.array([np.inf, np.inf, -np.inf, 0.0, np.inf, 0.0]))
        measure(domain)
        if invalid[None]:
            raise RuntimeError(f"Invalid or out-of-domain MPM state at {times[step]:g} s")
        vol_min, z_min, z_max, speed, stress, verlet = diagnostics.to_numpy()
        force = float(np.linalg.norm(rigid.contact_force.to_numpy()[0]))
        last[:] = (
            times[step],
            progress[step] * travel,
            progress[step] * setup["turns"],
            force,
            vol_min,
            z_min + origin[2],
            z_max + origin[2],
            speed,
            stress,
            verlet,
            time.perf_counter() - start_time,
        )
        writer.writerow(last)
        if snapshot_due:
            save_positions(step)
        if checkpoint_due:
            stream.flush()
            os.fsync(stream.fileno())
            save_checkpoint(step)
        print(f"pile step={step} t={last[0]:.6f} s advance={last[1]:.4f} m force={force:.3f} N", flush=True)

    progress_path = out / "progress.csv"
    with progress_path.open("r+" if args.resume else "w", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        if args.resume:
            rows = [row for row in csv.DictReader(stream) if round(float(row["time_s"]) / dt) <= initial_step]
            stream.seek(0)
            stream.truncate()
        writer.writerow(columns)
        if args.resume:
            writer.writerows([row[key] for key in columns] for row in rows)
            for path in snapshots.glob("*.npz"):
                if path.stem.isdecimal() and int(path.stem) > initial_step:
                    path.unlink()
        model.solver.postprocess.append(sample)
        model.solver.coupling_solver(mpm.scene, dem.scene)
    assert last and abs(last[0] - stop) < 1e-10
    (out / "summary.json").write_text(json.dumps(dict(zip(columns, map(float, last))), indent=2) + "\n")


if __name__ == "__main__":
    main()
