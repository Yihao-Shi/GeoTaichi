"""Soil-column collapse against an upper-clamped FEMPM wavy solid plate.

SI units; x length, y width, z height; gravity is -z. This example is self-contained.
The reference wave is stress-free, with constant horizontal thickness.
"""

import argparse
import json
import math
import os
import sys
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
    return dict(
        method="fempm",
        dimension=3,
        cycles=args.cycles,
        width=args.width,
        y_origin=1.0,
        height_segments=args.height_segments,
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
    from src.fem import FEMMesh

    points, cells, _ = plate_mesh(p)
    elastic.add_mesh(FEMMesh(points, cells, "TET4"))
    elastic.add_material(
        "NeoHookean",
        density=p["plate_density"],
        young_modulus=p["plate_young_modulus"],
        poisson_ratio=p["plate_poisson_ratio"],
    )
    fixed = np.flatnonzero(points[:, 2] >= p["floor"] + 0.5 * p["height"] - 1e-12)
    elastic.add_boundary_condition({"type": "Dirichlet", "nodes": fixed, "components": [0, 1, 2], "value": 0.0})
    return points, fixed


def build(p, output):
    import geotaichi as gt

    mpm = gt.MPM(log=False)
    elastic = gt.FEM(log=False)
    count = configure_mpm(mpm, p, output)
    reference, fixed = configure_plate(elastic, p, output)
    capacity = max(256, 6 * count)
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
    model = gt.FEMPM(elastic, mpm, log=False)
    model.set_configuration(domain=p["domain"], gravity=p["gravity"], search="BVH", log=False)
    model.set_solver(
        dict(
            Timestep=p["dt"],
            SimulationTime=p["time"],
            SaveInterval=p["save_interval"],
            SavePath=str(output),
            linear_solver="PCG",
            max_iterations=100,
            residual_tolerance=1e-6,
            correction_velocity_tolerance=1e-7,
            linear_solver_tolerance=1e-8,
            linear_solver_relative_tolerance=1e-7,
            linear_solver_max_iters=15000,
            newmark=[1.0, 0.5, 1.0],
            scale=1.0,
            **common,
        ),
        log=False,
    )
    faces, _ = elastic.scene.mesh.boundary_facets()
    model.add_surface(facet_sets={"wavy_plate": faces})
    model.memory_allocate(
        {
            "max_particle_number": count,
            "max_contact_pairs": capacity,
            "max_point_edge_pairs": 1,
            "max_point_triangle_pairs": capacity,
        }
    )
    model.choose_contact_model(
        "IPC", dhat=dhat, dmin=p["dmin"], kappa=p["kappa"], friction_coefficient=0.0, project_pd=True
    )
    model.add_essentials()
    engine = model.enginer
    return model, engine, reference, fixed


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
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--save-interval", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path)
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
    if (output / "parameters.json").exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    (output / "parameters.json").write_text(json.dumps(p, indent=2) + "\n")
    gt.init(dim=3, arch=args.arch, default_fp="float64", cpu_max_num_threads=8, log=True)
    model, engine, reference, fixed = build(p, output)
    mass0 = engine.mpm.particle.m.to_numpy().sum()
    rows = []

    def observe(current):
        if not current.last_step_record.get("converged", False):
            raise RuntimeError("an unconverged IPC step was accepted")
        elastic = current.fem.state.position.to_numpy()
        soil = current.mpm.particle.x.to_numpy()
        clamp_error = float(np.max(np.abs(elastic[fixed] - reference[fixed])))
        if clamp_error > 1e-10 or not np.all(np.isfinite(soil)) or not np.all(np.isfinite(elastic)):
            raise RuntimeError("nonfinite state or motion of the fixed plate")
        contact = current.contact.diagnostics()
        contacts, gap = contact["active_contacts"], contact["minimum_distance"]
        if contacts and (not math.isfinite(gap) or gap <= 0):
            raise RuntimeError("accepted IPC contact is not strictly separated")
        row = dict(
            time=float(current.time),
            active_contacts=int(contacts),
            minimum_gap=float(gap) if contacts else None,
            maximum_plate_displacement=float(np.linalg.norm(elastic - reference, axis=1).max()),
            clamp_error=clamp_error,
            soil_centroid_z=float(np.average(soil[:, 2], weights=current.mpm.particle.m.to_numpy())),
        )
        rows.append(row)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    model.run(verbose=False, postprocessing=[observe])
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
