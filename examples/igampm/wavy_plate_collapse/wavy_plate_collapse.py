"""Soil-column collapse against an upper-clamped IGAMPM wavy solid plate.

SI units; x length, y width, z height; gravity is -z. This example is self-contained.
The reference wave is stress-free, with constant horizontal thickness.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parameters(args):
    for name in ("amplitude", "spacing", "dt", "time", "save_interval", "width", "height", "soil_height"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.cycles < 1 or args.samples_per_period < 4 or args.samples_per_period % 2:
        raise ValueError("use positive cycles and an even samples-per-period >= 4")
    if args.amplitude >= 0.5 or args.spacing > 0.25:
        raise ValueError("amplitude must be < 0.5 m and spacing <= 0.25 m")
    if args.height_segments < 2 or args.height_segments % 2:
        raise ValueError("height-segments must be positive and even, with a knot on the upper-half clamp")
    if args.soil_height > args.height:
        raise ValueError("soil-height must not exceed plate height")
    if args.contact_capacity < 1:
        raise ValueError("contact-capacity must be positive")
    return dict(
        method="igampm",
        dimension=3,
        cycles=args.cycles,
        width=args.width,
        y_origin=1.0,
        height_segments=args.height_segments,
        contact_capacity=args.contact_capacity,
        amplitude=args.amplitude,
        height=args.height,
        thickness=0.025,
        plate_x=2.0,
        floor=0.1,
        soil_left=1.0,
        soil_height=args.soil_height,
        samples_per_period=args.samples_per_period,
        spacing=args.spacing,
        ppc=2,
        dt=args.dt,
        time=args.time,
        save_interval=args.save_interval,
        soil_density=1800.0,
        soil_young_modulus=2e6,
        soil_poisson_ratio=0.3,
        friction_angle=30.0,
        dilation_angle=0.0,
        cohesion=100.0,
        plate_density=1200.0,
        plate_young_modulus=5e6,
        plate_poisson_ratio=0.3,
        gravity=[0.0, 0.0, -9.81],
        domain=[5.0, args.width + 2.0, args.height + 0.5],
        kappa=2e6,
        dmin=0.0,
    )


def wave(p):
    y = np.linspace(0.0, p["width"], p["cycles"] * p["samples_per_period"] + 1)
    x = p["plate_x"] + p["amplitude"] * np.cos(2 * np.pi * p["cycles"] * y / p["width"])
    return make_interp_spline(y, x, k=3, bc_type="clamped")


def plate_section(p):
    y = np.linspace(0.0, p["width"], p["cycles"] * p["samples_per_period"] + 1)
    x = np.linspace(0.0, p["thickness"], 3)
    points = np.array([[left + offset, height + p["y_origin"]] for height, left in zip(y, wave(p)(y)) for offset in x])
    cells = []
    for j in range(len(y) - 1):
        for i in range(2):
            a = 3 * j + i
            cells.extend(([a, a + 1, a + 4], [a, a + 4, a + 3]))
    cells = np.asarray(cells, dtype=np.int32)
    triangles = points[cells]
    u, v = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    area = 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    quality = 4 * np.sqrt(3) * area / (np.sum(u * u, axis=1) + np.sum(v * v, axis=1) + np.sum((u - v) ** 2, axis=1))
    if not np.all(np.isfinite(points)) or np.any(area <= 0) or quality.min() < 0.02:
        raise ValueError("wavy TRI3 mesh requires positive area and minimum mean-ratio quality >= 0.02")
    return (
        points,
        cells,
        dict(minimum_area=float(area.min()), minimum_quality=float(quality.min()), mean_quality=float(quality.mean())),
    )


def plate_mesh(p):
    section, triangles, _ = plate_section(p)
    count = len(section)
    z = np.linspace(p["floor"], p["floor"] + p["height"], p["height_segments"] + 1)
    points = np.vstack([np.column_stack((section, np.full(count, level))) for level in z])
    cells = []
    # Global vertex ordering gives matching diagonals on adjacent prism faces.
    for layer in range(p["height_segments"]):
        for a, b, c in np.sort(triangles, axis=1) + layer * count:
            cells.extend(([a, b, c, c + count], [a, b, b + count, c + count], [a, a + count, b + count, c + count]))
    cells = np.asarray(cells, dtype=np.int32)
    vertices = points[cells]
    volume = np.linalg.det(np.stack([vertices[:, i] - vertices[:, 0] for i in (1, 2, 3)], axis=-1)) / 6
    inverted = volume < 0
    cells[inverted, 0], cells[inverted, 1] = cells[inverted, 1].copy(), cells[inverted, 0].copy()
    edge_sum = sum(np.sum((vertices[:, i] - vertices[:, j]) ** 2, axis=1) for i in range(4) for j in range(i))
    quality = 12 * (3 * np.abs(volume)) ** (2 / 3) / edge_sum
    if not np.all(np.isfinite(points)) or np.any(volume == 0) or quality.min() < 0.02:
        raise ValueError("wavy TET4 mesh requires positive volume and minimum mean-ratio quality >= 0.02")
    return (
        points,
        cells,
        dict(
            minimum_volume=float(np.abs(volume).min()),
            minimum_quality=float(quality.min()),
            mean_quality=float(quality.mean()),
        ),
    )


def plate_control_points(p):
    curve = wave(p)
    y = np.array([curve.t[i + 1 : i + 4].mean() for i in range(len(curve.c))])
    z = np.linspace(p["floor"], p["floor"] + p["height"], p["height_segments"] + 1)
    points = np.array(
        [
            [left + offset, width + p["y_origin"], level]
            for level in z
            for left, width in zip(curve.c, y)
            for offset in (0.0, p["thickness"])
        ]
    )
    # Linear z basis with a knot on the clamp plane: fixing the plane and all
    # control points above it makes the entire upper half exactly stationary.
    fixed = np.flatnonzero(points[:, 2] >= p["floor"] + 0.5 * p["height"] - 1e-12)
    return points, curve.t / p["width"], fixed


def soil_particles(p):
    curve = wave(p)
    ds = p["spacing"] / p["ppc"]
    rows = math.ceil(p["width"] / ds)
    dy = p["width"] / rows
    y = (np.arange(rows) + 0.5) * dy
    mesh_y = np.linspace(0.0, p["width"], p["cycles"] * p["samples_per_period"] + 1)
    # Identical soil is strictly left of both the smooth IGA boundary and the
    # polygonal FEM boundary, including portions where the chord cuts inward.
    right = np.minimum(curve(y), np.interp(y, mesh_y, curve(mesh_y))) - 0.1 * ds
    points, volumes, measures = [], [], []
    for height, limit in zip(y, right):
        columns = math.ceil((limit - p["soil_left"]) / ds)
        dx = (limit - p["soil_left"]) / columns
        points.extend(zip(p["soil_left"] + (np.arange(columns) + 0.5) * dx, np.full(columns, height + p["y_origin"])))
        volumes.extend([dx * dy] * columns)
        measure = np.full(columns, math.sqrt(dx * dy))
        measure[-1] = dy * math.sqrt(1 + float(curve.derivative()(height)) ** 2)
        measures.extend(measure)
    layers = math.ceil(p["soil_height"] / ds)
    dz = p["soil_height"] / layers
    section = np.asarray(points)
    particles = np.vstack(
        [np.column_stack((section, np.full(len(section), p["floor"] + (z + 0.5) * dz))) for z in range(layers)]
    )
    return particles, np.tile(np.asarray(volumes) * dz, layers), np.tile(np.asarray(measures) * dz, layers)


def configure_mpm(mpm, p, output):
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    points, volumes, measures = soil_particles(p)
    dx = p["spacing"]
    # Put the floor on a grid plane for every resolution, retaining z=0.1 m.
    xmin = [0.0, 0.0, p["floor"] - 2 * dx]
    xmax = p["domain"]
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=p["domain"],
        gravity=p["gravity"],
        background_damping=0.0,
        alphaPIC=0.0,
        visualize=True,
    )
    body = mpm.create_body()
    body.add_particles(
        points,
        volume=volumes,
        init_v=[0.0, 0.0, 0.0],
        name="collapsing_soil",
        grid_size=dx,
        xmin=xmin,
        xmax=xmax,
        surface_measure=measures,
    )
    mpm.add_body(body)
    mpm.add_material(
        model="DruckerPrager",
        density=p["soil_density"],
        young_modulus=p["soil_young_modulus"],
        poisson_ratio=p["soil_poisson_ratio"],
        Cohesion=p["cohesion"],
        FrictionAngle=p["friction_angle"],
        DilationAngle=p["dilation_angle"],
        dpType="Circumscribed",
    )
    mpm.add_element({"ElementSize": dx, "ShapeFunction": "Linear"})
    nx, ny, nz = np.ceil((np.array(xmax) - xmin) / dx).astype(int) + 1
    nodes = np.arange(nx * ny * nz).reshape(nz, ny, nx)
    bottom = nodes[:3].reshape(-1)
    sides = np.concatenate((nodes[:, :, 0].reshape(-1), nodes[:, :, -1].reshape(-1)))
    ends = np.concatenate((nodes[:, 0].reshape(-1), nodes[:, -1].reshape(-1)))
    boundary = DirichletBoundary()
    boundary.append(
        [list(3 * sides), list(3 * ends + 1), list(3 * bottom + 2)],
        [0.0] * (len(sides) + len(ends) + len(bottom)),
    )
    mpm.add_boundary_condition(dirichlet=boundary)
    mpm.set_solver(
        {
            "Timestep": p["dt"],
            "SimulationTime": p["time"],
            "SaveInterval": p["save_interval"],
            "SavePath": str(output),
            "newmark": [1.0, 0.5, 1.0],
            "project_pd": True,
            "scale": 1.0,
        }
    )
    return len(points)


def configure_plate(elastic, p, output):
    elastic.set_configuration(dimension=3, solver_type="Implicit")
    from src.iga import Cube, DirichletBoundary, Primitives

    points, knots, fixed = plate_control_points(p)
    plate = Cube()
    plate.set_parameters(
        start_point=[p["plate_x"], p["y_origin"], p["floor"]], size=[p["thickness"], p["width"], p["height"]]
    )
    plate.generate_knot_u(degree=1, num_ctrlpts=2)
    plate.generate_knot_v(degree=3, num_ctrlpts=len(points) // (2 * (p["height_segments"] + 1)))
    plate.generate_knot_w(degree=1, num_ctrlpts=p["height_segments"] + 1)
    plate.knot_vector_v = knots
    plate.control_points = points
    plate.generate_weights()
    primitives = Primitives()
    primitives.append(plate, "wavy_plate")
    primitives.finialize()
    boundary = DirichletBoundary()
    boundary.append(
        [list(3 * fixed), list(3 * fixed + 1), list(3 * fixed + 2)],
        [0.0] * (3 * len(fixed)),
    )
    elastic.add_primitives(primitives)
    elastic.add_boundary_condition(dirichlet=boundary)
    elastic.add_element(degree=[1, 3, 1])
    elastic.add_material(
        density=p["plate_density"],
        young_modulus=p["plate_young_modulus"],
        poisson_ratio=p["plate_poisson_ratio"],
        gravity=p["gravity"],
    )
    elastic.set_solver(
        dt=p["dt"],
        step=math.ceil(p["time"] / p["dt"]),
        interval=max(1, round(p["save_interval"] / p["dt"])),
        path=str(output),
        newmark=[1.0, 0.5, 1.0],
        residual=1e-7,
        max_iters=100,
    )
    return points, fixed


def build(p, output):
    import geotaichi as gt

    mpm = gt.MPM(log=False)
    elastic = gt.IGA(log=False)
    count = configure_mpm(mpm, p, output)
    reference, fixed = configure_plate(elastic, p, output)
    capacity = min(6 * count, p["contact_capacity"])
    slope = 2 * np.pi * p["cycles"] * p["amplitude"] / p["width"]
    dhat = 0.25 * p["spacing"] / p["ppc"] / math.sqrt(1 + slope * slope)
    common = dict(
        assemble_type="HashTriplet",
        project_pd=True,
        contact_all_mpm_particles=True,
        enable_step_retry=True,
        step_retry_max_retries=4,
        step_retry_reduction=0.5,
        step_retry_minimum_timestep=p["dt"] / 16,
    )
    model = gt.IGAMPM(
        elastic,
        mpm,
        log=False,
        contact_model="IPC",
        activate_friction=False,
        dhat=dhat,
        dmin=p["dmin"],
        kappa=p["kappa"],
        use_physical_barrier=True,
        contact_surface_include=[(0, face) for face in range(6)],
        compact_contact_slots=True,
        barrier_nnz=capacity * 16**2,
        monolithic_max_iterations=100,
        monolithic_tolerance=1e-7,
        monolithic_force_rtol=1e-6,
        monolithic_linear_solver_tolerance=1e-8,
        monolithic_linear_solver_relative_tolerance=1e-7,
        monolithic_linear_solver_max_iters=15000,
        **common,
    )
    model.set_configuration(dimension=3, contact_model="IPC", activate_friction=False, log=False)
    engine = model.build()
    return model, engine, reference, fixed


def state_fields(engine):
    mpm = engine.mpm
    return dict(
        particle_position=mpm.particle.x,
        particle_velocity=mpm.particle.v,
        particle_acceleration=mpm.particle.a,
        particle_mass=mpm.particle.m,
        deformation_gradient=mpm.F0,
        plastic_inverse=mpm.material.plastic_deformation_inverse,
        equivalent_plastic_strain=mpm.material.equivalent_plastic_strain,
        volumetric_plastic_strain=mpm.material.volumetric_plastic_strain,
        grid_mass=mpm.grid.m,
        grid_velocity=mpm.grid.v,
        grid_acceleration=mpm.grid.a,
        elastic_position=engine.iga.patch.control_points,
        elastic_velocity=engine.iga.patch.velocitys,
        elastic_acceleration=engine.iga.patch.accelerations,
    )


def restore_checkpoint(model, engine, path, p):
    previous = json.loads((path.parent / "parameters.json").read_text())
    if any(p[name] != value for name, value in previous.items() if name not in ("time", "dt")):
        raise ValueError("checkpoint geometry, materials and output interval must match")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if not 0 <= metadata["time"] < p["time"]:
            raise ValueError("checkpoint time must precede the requested end time")
        for name, field in state_fields(engine).items():
            values = data[name]
            if values.shape != field.to_numpy().shape or not np.isfinite(values).all():
                raise ValueError(f"invalid checkpoint field: {name}")
            field.from_numpy(values)
    engine.time = float(metadata["time"])
    engine.implicit_step_index = int(metadata["step"])
    engine.last_step_record = metadata["last_step"]
    frames = round(engine.time / p["save_interval"])
    if not math.isclose(frames * p["save_interval"], engine.time, abs_tol=1e-10, rel_tol=0):
        raise ValueError("resume from a saved output frame")
    for child in (model.iga_engine, model.mpm_engine):
        child.time = engine.time
        child.step_count = engine.implicit_step_index
        child.output_count = frames + 1
    engine.iga.patch.current_print = frames + 1
    model.mpm.sims.current_print = frames + 1
    model.mpm.sims.current_step = engine.implicit_step_index
    model.mpm.sims.current_time = engine.time
    model._last_implicit_recorded_step = engine.implicit_step_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--cycles", type=int, default=15)
    parser.add_argument("--amplitude", type=float, default=0.06)
    parser.add_argument("--samples-per-period", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=0.1)
    parser.add_argument("--width", type=float, default=3.0)
    parser.add_argument("--height", type=float, default=3.0)
    parser.add_argument("--soil-height", type=float, default=2.0)
    parser.add_argument("--height-segments", type=int, default=12)
    parser.add_argument("--contact-capacity", type=int, default=32768)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--save-interval", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="continue an existing output directory from a saved state frame")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    p = parameters(args)
    _, _, quality = plate_mesh(p)
    points, volumes, _ = soil_particles(p)
    p.update(particle_count=len(points), soil_mass=float(volumes.sum() * p["soil_density"]), mesh_quality=quality)
    if args.check:
        print(json.dumps(p, indent=2))
        return
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    import geotaichi as gt

    output = args.output_dir or Path(__file__).resolve().parent / "OutputData"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "parameters.json").exists() and args.resume is None:
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    if args.resume is not None and args.resume.resolve().parent != output.resolve():
        raise ValueError("resume checkpoint must belong to the output directory")
    if args.resume is None:
        (output / "parameters.json").write_text(json.dumps(p, indent=2) + "\n")
    gt.init(dim=3, arch=args.arch, default_fp="float64", cpu_max_num_threads=8, log=True)
    model, engine, reference, fixed = build(p, output)
    mass0 = engine.mpm.particle.m.to_numpy().sum()
    engine._initialize_implicit_ipc_state()
    rows = []
    next_save = [p["save_interval"]]
    if args.resume is not None:
        restore_checkpoint(model, engine, args.resume, p)
        rows = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
        if not rows or not math.isclose(rows[-1]["time"], engine.time, abs_tol=1e-10, rel_tol=0):
            raise ValueError("checkpoint must match the last accepted history row")
        next_save[0] = (round(engine.time / p["save_interval"]) + 1) * p["save_interval"]
        (output / "parameters.json").write_text(json.dumps(p, indent=2) + "\n")
    start = time.monotonic() - (rows[-1].get("wall_seconds", 0.0) if rows else 0.0)

    def checkpoint():
        data = {name: field.to_numpy() for name, field in state_fields(engine).items()}
        data["metadata"] = np.array(json.dumps(engine.diagnostics_snapshot()))
        temporary = output / "latest_state.tmp.npz"
        np.savez(temporary, **data)
        temporary.replace(output / "latest_state.npz")
        archive = output / f"state_{engine.time:.6f}_{engine.implicit_step_index:06d}.npz"
        if not archive.exists():
            archive.hardlink_to(output / "latest_state.npz")

    def observe(current):
        if not current.last_step_record.get("converged", False):
            raise RuntimeError("an unconverged IPC step was accepted")
        elastic = current.iga.patch.control_points.to_numpy()
        soil = current.mpm.particle.x.to_numpy()
        clamp_error = float(np.max(np.abs(elastic[fixed] - reference[fixed])))
        if clamp_error > 1e-10 or not np.all(np.isfinite(soil)) or not np.all(np.isfinite(elastic)):
            raise RuntimeError("nonfinite state or motion of the fixed plate")
        contacts = current.curr_barrier_contact_num
        gap = float(current.last_step_record["minimum_distance"]) - p["dmin"]
        if contacts and (not math.isfinite(gap) or gap <= 0):
            raise RuntimeError("accepted IPC contact is not strictly separated")
        row = dict(
            time=float(current.time),
            active_contacts=int(contacts),
            minimum_gap=float(gap) if contacts else None,
            maximum_plate_displacement=float(np.linalg.norm(elastic - reference, axis=1).max()),
            clamp_error=clamp_error,
            soil_centroid_z=float(np.average(soil[:, 2], weights=current.mpm.particle.m.to_numpy())),
            material_lagged_iterations=int(current.mpm.last_material_lagged_iterations),
            material_lagged_error=float(current.mpm.last_material_lagged_error),
            step_retry=current.last_step_record["step_retry"],
            wall_seconds=time.monotonic() - start,
        )
        rows.append(row)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if current.time >= next_save[0] - 1e-12:
            checkpoint()
            model._record_implicit_frame(current)
            while next_save[0] <= current.time + 1e-12:
                next_save[0] += p["save_interval"]

    if model._last_implicit_recorded_step != engine.implicit_step_index:
        model._record_implicit_frame(engine)
    checkpoint()
    try:
        while engine.time < p["time"] - 1e-12:
            engine._set_implicit_timestep(min(p["dt"], p["time"] - engine.time, next_save[0] - engine.time))
            model.run(steps=1, verbose=False, record=False, postprocessing=[observe])
    finally:
        checkpoint()
    if model._last_implicit_recorded_step != engine.implicit_step_index:
        model._record_implicit_frame(engine)
    mass_error = abs(engine.mpm.particle.m.to_numpy().sum() / mass0 - 1)
    det_f = float(np.linalg.det(engine.mpm.F0.to_numpy()).min())
    det_p = float(np.linalg.det(engine.mpm.material.plastic_deformation_inverse.to_numpy()).min())
    if not (
        math.isfinite(mass_error)
        and mass_error <= 1e-12
        and math.isfinite(det_f)
        and det_f > 0
        and math.isfinite(det_p)
        and det_p > 0
    ):
        raise RuntimeError("mass or material orientation invariant failed")
    if not math.isclose(engine.time, p["time"], abs_tol=1e-10, rel_tol=0):
        raise RuntimeError("accepted steps did not reach the requested physical end time")
    summary = dict(
        rows[-1],
        completed_time=float(engine.time),
        steps=len(rows),
        mass_relative_error=float(mass_error),
        minimum_det_f=det_f,
        minimum_det_plastic_inverse=det_p,
        maximum_active_contacts=max(r["active_contacts"] for r in rows),
        converged=True,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
