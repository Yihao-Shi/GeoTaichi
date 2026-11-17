"""Twist a square Neo-Hookean column with a fixed base and a rotating top."""

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def top_displacement(reference, angle_degrees, fraction):
    # Smoothstep starts and stops the imposed rotation with zero angular speed.
    angle = np.deg2rad(angle_degrees) * fraction**2 * (3.0 - 2.0 * fraction)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return reference @ rotation.T - reference


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("explicit", "implicit"), default="explicit")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--angle", type=float, default=450.0, help="final top rotation in degrees")
    parser.add_argument("--control-points", type=int, nargs=3, default=(6, 6, 18), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--output-interval", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    implicit = args.solver == "implicit"
    if args.steps is None:
        args.steps = 300 if implicit else 15000
    if args.dt is None:
        args.dt = 0.005 if implicit else 0.0001
    if args.output_interval is None:
        args.output_interval = 10 if implicit else 500
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / f"twisting_column_3d_{args.solver}"
    if args.steps <= 0 or args.output_interval <= 0:
        parser.error("--steps and --output-interval must be positive")
    if not np.isfinite(args.dt) or args.dt <= 0.0:
        parser.error("--dt must be finite and positive")
    if not np.isfinite(args.angle):
        parser.error("--angle must be finite")
    if min(args.control_points) < 3:
        parser.error("quadratic NURBS require at least 3 control points per direction")
    return args


def record_output(engine, deformation_gradient, output_dir):
    patch = engine.patch
    engine.compute_deformation_gradient(
        patch.total_num_element[1],
        patch.prefix_num_knot[0],
        patch.prefix_num_element[0],
        patch.prefix_total_num_ctrlpts[0],
        patch.num_knot[1],
        patch.num_element[1],
        patch.num_ctrlpts[1],
        deformation_gradient,
    )
    engine.record(log=False)
    np.savez(
        output_dir / f"state_{engine.output_count - 1:06d}.npz",
        step=engine.step_count,
        time=engine.time,
        control_points=patch.control_points.to_numpy(),
        deformation_gradient=deformation_gradient.to_numpy(),
        stress=patch.stress.to_numpy(),
    )


def main(argv=None):
    args = parse_arguments(argv)
    import taichi as ti
    from geotaichi import IGA, init
    from src.iga import Cube, DirichletBoundary, Primitives
    from src.utils.FieldIO import field_to_numpy_slice

    started = time.perf_counter()
    init(dim=3, arch="cpu", default_fp="float64", log=False)
    iga = IGA(log=False)
    iga.set_configuration(dimension=3, solver_type=args.solver.capitalize(), log=False)
    column = Cube()
    column.set_parameters(start_point=[-0.3, -0.3, 0.0], size=[0.6, 0.6, 3.0])
    nx, ny, nz = args.control_points
    column.generate_knot_u(degree=2, num_ctrlpts=nx)
    column.generate_knot_v(degree=2, num_ctrlpts=ny)
    column.generate_knot_w(degree=2, num_ctrlpts=nz)
    column.generate_ctrlpts()
    column.generate_weights()
    reference = column.control_points.copy()
    bottom = np.flatnonzero(np.isclose(reference[:, 2], 0.0))
    top = np.flatnonzero(np.isclose(reference[:, 2], 3.0))
    bottom_dofs = (3 * bottom[:, None] + np.arange(3)).ravel().astype(np.int32)
    top_dofs = (3 * top[:, None] + np.arange(3)).ravel().astype(np.int32)
    boundary = DirichletBoundary()
    boundary.append([bottom_dofs.tolist()], np.zeros(bottom_dofs.size).tolist())
    boundary.append([top_dofs.tolist()], np.zeros(top_dofs.size).tolist())
    primitives = Primitives()
    primitives.append(column, "twisting_column")
    primitives.finialize()
    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=boundary)
    iga.add_element(degree=[2, 2, 2])
    iga.add_material(young_modulus=8.0e5, poisson_ratio=0.3, density=1000.0)
    iga.set_solver(
        dt=args.dt,
        step=args.steps,
        interval=1,
        damping=5.0,
        gravity=[0.0, 0.0, 0.0],
        path=str(args.output_dir),
        log=False,
    )
    if args.solver == "implicit":
        iga.set_solver(
            newmark=[1.0, 0.5, 1.0],
            residual=1.0e-6,
            max_iters=50,
            assemble_type="HashTriplet",
            linear_solver="PCG",
            project_hessian_to_psd=True,
            log=False,
        )
    engine = iga.build()
    deformation_gradient = ti.Matrix.field(3, 3, ti.f64, shape=engine.initial_deformation_gradients.shape)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").unlink(missing_ok=True)
    np.savez(args.output_dir / "reference.npz", control_points=reference, bottom=bottom, top=top)
    engine.precompute()
    record_output(engine, deformation_gradient, args.output_dir)
    ti.sync()
    setup_seconds = time.perf_counter() - started
    print(
        f"{args.solver}: {len(reference)} control points, {engine.degree_of_freedom} DOFs; compiling first step",
        flush=True,
    )
    step_seconds = []
    for step in range(1, args.steps + 1):
        target = top_displacement(reference[top], args.angle, step / args.steps)
        tick = time.perf_counter()
        prescribed = target
        if args.solver == "implicit":
            # Implicit IGA takes an increment from the last accepted position.
            current_top = field_to_numpy_slice(engine.patch.control_points, top[0], top[-1] + 1)
            prescribed = reference[top] + target - current_top
        boundary.set_constraints(top_dofs, np.ascontiguousarray(prescribed.ravel()))
        engine.substep(verbose=False, record_history=False)
        ti.sync()
        step_seconds.append(time.perf_counter() - tick)
        if step % args.output_interval == 0 or step == args.steps:
            record_output(engine, deformation_gradient, args.output_dir)
            print(f"step {step}/{args.steps}, time={engine.time:.4f} s", flush=True)
    summary = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "solver": args.solver,
        "backend": str(ti.lang.impl.current_cfg().arch),
        "precision": "float64",
        "control_points": int(len(reference)),
        "degrees_of_freedom": int(engine.degree_of_freedom),
        "elements": int((nx - 2) * (ny - 2) * (nz - 2)),
        "steps": args.steps,
        "dt": args.dt,
        "output_interval": args.output_interval,
        "physical_seconds": engine.time,
        "angle_degrees": args.angle,
        "setup_and_initial_output_seconds": setup_seconds,
        "first_step_seconds": step_seconds[0],
        "remaining_steps_seconds": sum(step_seconds[1:]),
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
