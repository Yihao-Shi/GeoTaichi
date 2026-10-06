"""Independent thick 3D FlexibleBarrier: elastic IGA and DP MPM with implicit IPC.

The original barrier.py uses a 0.1 m wide slab, explicit GIMP, linear
elasticity and nonassociated DP (dilation=0). This implicit example uses a 1 m
wide slab, linear MPM interpolation, NeoHookean elasticity and constant
nonassociated DP (friction=30 degrees, dilation=0 by default).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def parameters(args):
    for value in (args.thickness, args.spacing, args.elastic_spacing, args.dt, args.time, args.save_interval):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("dimensions, discretization and times must be finite and positive")
    if args.thickness <= 0.1 or args.ppc < 1 or args.contact_capacity < 1:
        raise ValueError("use thickness > 0.1 m and positive particle/contact capacities")
    if not math.isfinite(args.dilation_angle) or not 0 <= args.dilation_angle <= 30:
        raise ValueError("dilation-angle must be finite and between 0 and 30 degrees")
    for length, spacing in (
        (8.0, args.spacing),
        (4.0, args.spacing),
        (args.thickness, args.spacing),
        (1.0, args.elastic_spacing),
        (5.0, args.elastic_spacing),
        (args.thickness, args.elastic_spacing),
    ):
        if not math.isclose(length / spacing, round(length / spacing), abs_tol=1e-10, rel_tol=0):
            raise ValueError("spacing must divide each corresponding body dimension")
    return dict(
        method="igampm",
        dimension=3,
        thickness=args.thickness,
        soil_origin=[0.1, 0.1, 0.1],
        soil_size=[8.0, args.thickness, 4.0],
        barrier_origin=[8.1, 0.1, 0.1],
        barrier_size=[1.0, args.thickness, 5.0],
        spacing=args.spacing,
        ppc=args.ppc,
        elastic_spacing=args.elastic_spacing,
        particle_count=round(8.0 * args.thickness * 4.0 / args.spacing**3) * args.ppc**3,
        soil_mass=2500.0 * 8.0 * args.thickness * 4.0,
        density=2500.0,
        young_modulus=2e7,
        poisson_ratio=0.3,
        cohesion=0.0,
        friction_angle=30.0,
        dilation_angle=args.dilation_angle,
        original_dilation_angle=0.0,
        elastic_model="NeoHookean",
        shape_function="Linear",
        gravity=[0.0, 0.0, -9.8],
        contact_model="IPC",
        contact_friction=0.577,
        friction_mode="lagged",
        # Start at the particle-centre gap so the initial unstrained soil is
        # not impulsively repelled by an already active barrier potential.
        contact_all_mpm_particles=True,
        kappa=2e7,
        dhat=0.5 * args.spacing / args.ppc,
        dmin=0.0,
        dt=args.dt,
        target_time=args.time,
        save_interval=args.save_interval,
        linear_solver=args.linear_solver,
        contact_capacity=args.contact_capacity,
        newton_velocity_tolerance=1e-8,
        newton_force_rtol=1e-8,
        friction_velocity_tolerance=1e-7,
        differences_from_original=[
            "y thickness",
            "implicit time integration",
            "linear MPM interpolation",
            "finite-strain elastic energy",
            "nonassociated DP with constant dilation angle",
            "IPC cross-contact",
        ],
    )


def configure_mpm(mpm, p, output):
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    dx = p["spacing"]
    xmin = np.full(3, 0.1 - dx)
    xmax = np.array([12.1, 0.1 + p["thickness"] + dx, 5.6])
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=xmax.tolist(),
        gravity=p["gravity"],
        background_damping=0.0,
        alphaPIC=0.0,
        visualize=True,
    )
    body = mpm.create_body()
    body.add_cube(
        start=p["soil_origin"],
        end=np.array(p["soil_origin"]) + p["soil_size"],
        spacing=dx,
        ppc=p["ppc"],
        grid_size=dx,
        xmin=xmin,
        xmax=xmax,
        name="dp_soil",
    )
    assert body.particle_counter == p["particle_count"]
    mpm.add_body(body)
    mpm.add_material(
        model="DruckerPrager",
        density=p["density"],
        young_modulus=p["young_modulus"],
        poisson_ratio=p["poisson_ratio"],
        Cohesion=p["cohesion"],
        FrictionAngle=p["friction_angle"],
        DilationAngle=p["dilation_angle"],
        dpType="Circumscribed",
    )
    mpm.add_element({"ElementSize": dx, "ShapeFunction": p["shape_function"]})
    count = np.ceil((xmax - xmin) / dx).astype(int) + 1
    iz, iy, ix = np.indices(tuple(count[::-1]))
    nodes = np.arange(np.prod(count), dtype=np.int32).reshape(tuple(count[::-1]))
    bottom = xmin[2] + iz * dx <= 0.1 + 1e-10
    left = xmin[0] + ix * dx <= 0.1 + 1e-10
    y = xmin[1] + iy * dx
    sides = (y <= 0.1 + 1e-10) | (y >= 0.1 + p["thickness"] - 1e-10)
    entries = [
        (3 * nodes[bottom | left]).tolist(),
        (3 * nodes[bottom | sides] + 1).tolist(),
        (3 * nodes[bottom] + 2).tolist(),
    ]
    boundary = DirichletBoundary()
    boundary.append(entries, [0.0] * sum(map(len, entries)))
    mpm.add_boundary_condition(dirichlet=boundary)
    mpm.set_solver(
        {
            "Timestep": p["dt"],
            "SimulationTime": p["target_time"],
            "SaveInterval": p["save_interval"],
            "SavePath": str(output),
            "newmark": [1.0, 0.5, 1.0],
            "project_pd": True,
            "scale": 1.0,
        }
    )


def configure_elastic(elastic, p, output):
    size = p["barrier_size"]
    divisions = np.rint(np.array(size) / p["elastic_spacing"]).astype(int)
    elastic.set_configuration(dimension=3, solver_type="Implicit")
    from src.iga import Cube, DirichletBoundary, Primitives

    cube = Cube()
    cube.set_parameters(start_point=p["barrier_origin"], size=size)
    for axis, division in zip("uvw", divisions):
        getattr(cube, "generate_knot_" + axis)(degree=2, num_ctrlpts=int(division) + 2)
    cube.generate_ctrlpts()
    cube.generate_weights()
    primitives = Primitives()
    primitives.append(cube, "elastic_barrier")
    primitives.finialize()
    points = cube.control_points
    bottom = np.flatnonzero(np.isclose(points[:, 2], 0.1))
    sides = np.flatnonzero(np.isclose(points[:, 1], 0.1) | np.isclose(points[:, 1], 0.1 + p["thickness"]))
    entries = [(3 * bottom).tolist(), (3 * np.union1d(bottom, sides) + 1).tolist(), (3 * bottom + 2).tolist()]
    boundary = DirichletBoundary()
    boundary.append(entries, [0.0] * sum(map(len, entries)))
    elastic.add_primitives(primitives)
    elastic.add_boundary_condition(dirichlet=boundary)
    elastic.add_element(degree=[2, 2, 2])
    elastic.add_material(
        density=p["density"],
        young_modulus=p["young_modulus"],
        poisson_ratio=p["poisson_ratio"],
        gravity=p["gravity"],
    )
    elastic.set_solver(
        dt=p["dt"],
        newmark=[1.0, 0.5, 1.0],
        residual=5e-4,
        max_iters=100,
        interval=max(1, round(p["save_interval"] / p["dt"])),
        step=math.ceil(p["target_time"] / p["dt"]),
        path=str(output),
    )


def build(p, output):
    import geotaichi as gt

    mpm = gt.MPM(log=False)
    elastic = gt.IGA(log=False)
    configure_mpm(mpm, p, output)
    configure_elastic(elastic, p, output)
    common = dict(
        assemble_type="HashTriplet",
        project_pd=True,
        enable_step_retry=True,
        step_retry_max_retries=4,
        step_retry_reduction=0.5,
        step_retry_minimum_timestep=p["dt"] / 16.0,
        contact_all_mpm_particles=True,
    )
    blocks = p["contact_capacity"] * (9 + 8) ** 2
    model = gt.IGAMPM(
        elastic,
        mpm,
        log=False,
        contact_model="IPC",
        activate_friction=True,
        kappa=p["kappa"],
        dhat=p["dhat"],
        dmin=p["dmin"],
        mu=p["contact_friction"],
        epsv=1e-3,
        friction_mode=p["friction_mode"],
        friction_iterations=-1,
        use_physical_barrier=True,
        friction_tolerance=p["friction_velocity_tolerance"],
        compact_contact_slots=True,
        barrier_nnz=blocks,
        friction_nnz=blocks,
        # The frozen-cache inner problem must converge more
        # tightly than the 1e-7 m/s updated-friction probe.
        monolithic_max_iterations=100,
        monolithic_tolerance=p["newton_velocity_tolerance"],
        monolithic_force_rtol=p["newton_force_rtol"],
        monolithic_linear_solver_tolerance=1e-6,
        monolithic_linear_solver_relative_tolerance=1e-7,
        monolithic_linear_solver_max_iters=30000,
        **common,
    )
    model.set_configuration(dimension=3, coupling_scheme="IGAMPM", contact_model="IPC", activate_friction=True)
    engine = model.build()
    return model, engine


def fields(engine):
    mpm = engine.mpm
    data = dict(
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
    )
    data.update(
        elastic_position=engine.iga.patch.control_points,
        elastic_velocity=engine.iga.patch.velocitys,
        elastic_acceleration=engine.iga.patch.accelerations,
    )
    return {name: field.to_numpy() for name, field in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gpu")
    parser.add_argument("--thickness", type=float, default=1.0)
    parser.add_argument("--spacing", type=float, default=0.1)
    parser.add_argument("--elastic-spacing", type=float, default=0.25)
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--dilation-angle", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--time", type=float, default=3.0)
    parser.add_argument("--save-interval", type=float, default=0.1)
    parser.add_argument("--contact-capacity", type=int, default=8192)
    parser.add_argument("--linear-solver", choices=("PCG", "Scipy"), default="PCG")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="validate setup without initializing Taichi")
    args = parser.parse_args()
    p = parameters(args)
    if args.check:
        print(json.dumps(p, indent=2))
        return
    output = args.output_dir or Path(__file__).resolve().parent / "OutputData"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "parameters.json").exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    (output / "parameters.json").write_text(json.dumps(p, indent=2) + "\n")
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    import geotaichi as gt

    gt.init(dim=3, arch=args.arch, default_fp="float64", debug=False, log=True)
    model, engine = build(p, output)
    start = time.monotonic()
    next_save = [p["save_interval"]]
    peak_contacts = [0]
    minimum_gap = [math.inf]

    def checkpoint():
        data = fields(engine)
        data["metadata"] = np.array(json.dumps(engine.diagnostics_snapshot()))
        temporary = output / "latest_state.tmp.npz"
        np.savez(temporary, **data)
        temporary.replace(output / "latest_state.npz")
        step = engine.implicit_step_index
        archive = output / f"state_{engine.time:.6f}_{step:06d}.npz"
        if not archive.exists():
            archive.hardlink_to(output / "latest_state.npz")
        return data

    def observe(current):
        snapshot = current.diagnostics_snapshot()
        snapshot["last_step"] = current.last_step_record
        if not current.last_step_record.get("converged", False):
            raise RuntimeError("an unconverged step was accepted")
        gap = float(current.last_step_record["minimum_distance"]) - p["dmin"]
        contacts = current.curr_barrier_contact_num
        peak_contacts[0] = max(peak_contacts[0], int(contacts))
        minimum_gap[0] = min(minimum_gap[0], gap)
        snapshot.update(
            net_contact_gap=gap if math.isfinite(gap) else None,
            active_contacts=int(contacts),
            wall_seconds=time.monotonic() - start,
        )
        if contacts and (not math.isfinite(gap) or gap <= 0.0):
            raise RuntimeError("accepted IPC step is not strictly feasible")
        with (output / "step_diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        if current.time + 1e-12 >= next_save[0]:
            checkpoint()
            model._record_implicit_frame(current)
            while next_save[0] <= current.time + 1e-12:
                next_save[0] += p["save_interval"]
            print(json.dumps(snapshot), flush=True)

    linear_solve = None
    if args.linear_solver == "Scipy":
        from scipy.sparse.linalg import splu

        def linear_solve(matrix, rhs):
            matrix.eliminate_zeros()
            factor = splu(matrix.tocsc(), permc_spec="MMD_AT_PLUS_A")
            solution = factor.solve(rhs)
            tolerance = max(1e-6, 1e-7 * np.linalg.norm(rhs))
            for _ in range(5):
                residual = rhs - matrix @ solution
                if np.linalg.norm(residual) <= tolerance:
                    break
                solution += factor.solve(residual)
            if not np.isfinite(solution).all() or np.linalg.norm(matrix @ solution - rhs) > tolerance:
                raise RuntimeError("linear solve failed the original Ax-b tolerance")
            return solution

    try:
        engine._initialize_implicit_ipc_state()
        if model._last_implicit_recorded_step != engine.implicit_step_index:
            model._record_implicit_frame(engine)
        while engine.time < p["target_time"] - 1e-12:
            engine._set_implicit_timestep(min(p["dt"], p["target_time"] - engine.time, next_save[0] - engine.time))
            model.run(steps=1, record=False, verbose=False, linear_solve=linear_solve, postprocessing=[observe])
        if model._last_implicit_recorded_step != engine.implicit_step_index:
            model._record_implicit_frame(engine)
        data = checkpoint()
        finite = all(np.isfinite(value).all() for name, value in data.items() if name != "metadata")
        det_f = float(np.linalg.det(data["deformation_gradient"]).min())
        det_p = float(np.linalg.det(data["plastic_inverse"]).min())
        step = engine.implicit_step_index
        summary = dict(
            completed_time=float(engine.time),
            step=int(step),
            target_time=p["target_time"],
            particle_count=len(data["particle_position"]),
            finite=finite,
            minimum_det_f=det_f,
            minimum_det_plastic_inverse=det_p,
            maximum_active_contacts=peak_contacts[0],
            contact_exercised=peak_contacts[0] > 0,
            minimum_net_gap=minimum_gap[0] if math.isfinite(minimum_gap[0]) else None,
            wall_seconds=time.monotonic() - start,
        )
        summary["complete"] = bool(
            finite
            and det_f > 0.0
            and det_p > 0.0
            and (p["target_time"] < 1.0 or peak_contacts[0] > 0)
            and len(data["particle_position"]) == p["particle_count"]
            and math.isclose(engine.time, p["target_time"], abs_tol=1e-10)
        )
        (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        if not summary["complete"]:
            raise RuntimeError(f"full run validation failed: {summary}")
    except Exception as exception:
        (output / "failure.json").write_text(
            json.dumps(
                dict(time=float(engine.time), exception=type(exception).__name__, message=str(exception)), indent=2
            )
            + "\n"
        )
        checkpoint()
        raise


if __name__ == "__main__":
    main()
