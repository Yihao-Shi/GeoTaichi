"""Fixed-step reverse differentiation for a TRI3 cloth FEM trajectory."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import taichi as ti

from src.fem import FEM


def build_problem(
    stretch_stiffness=1.0e3,
    bending_stiffness=2.5e7,
    divisions=2,
    steps=3,
    dt=1.0e-2,
    newton_velocity_tolerance=1.0e-6,
    plane_contact=False,
):
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    mesh = fem.add_mesh(geometry="rectangle", size=(1.0, 1.0), divisions=(divisions, divisions))
    if plane_contact:
        mesh.points[:, 2] += 0.05
        mesh.rest_shape[:, 2] += 0.05
    fem.add_material(
        "ClothARAP",
        stretch_stiffness=stretch_stiffness,
        compression_stiffness=stretch_stiffness,
        density=1.0,
        thickness=0.01,
        bending_stiffness=bending_stiffness,
        bending_poisson_ratio=0.3,
        bending_model="Dihedral",
    )
    fixed = mesh.node_sets["ymax"] if plane_contact else [0]
    fem.add_boundary_condition({"type": "Dirichlet", "nodes": fixed, "components": "all", "value": 0.0})
    if plane_contact:
        fem.add_contact(
            "IPC",
            self_contact=False,
            planes=[((0, 0, 0), (0, 0, 1))],
            dhat=0.02,
            kappa=2.0e4,
            friction_coefficient=0.0,
        )
    fem.set_solver(
        dt=dt,
        step=steps,
        gravity=(0.0, 0.0, -9.8),
        max_iterations=80,
        residual_tolerance=1.0e-7,
        correction_velocity_tolerance=newton_velocity_tolerance,
        linear_solver="PCG",
        linear_solver_relative_tolerance=1.0e-10,
        assemble_type="HashTriplet",
        project_pd=True,
    )
    return fem, mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--divisions", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--newton-velocity-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--bending-stiffness", type=float, default=2.5e7)
    parser.add_argument("--plane-contact", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if (
        args.divisions < 1
        or args.steps < 1
        or args.dt <= 0.0
        or args.bending_stiffness <= 0.0
        or args.newton_velocity_tolerance <= 0.0
    ):
        parser.error("divisions, steps, dt, bending stiffness, and Newton tolerance must be positive")

    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f64)
    fem, mesh = build_problem(
        bending_stiffness=args.bending_stiffness,
        divisions=args.divisions,
        steps=args.steps,
        dt=args.dt,
        newton_velocity_tolerance=args.newton_velocity_tolerance,
        plane_contact=args.plane_contact,
    )
    initial_position = mesh.points.copy()
    tape = fem.differentiable(steps=args.steps)
    started = time.perf_counter()
    for _ in range(args.steps):
        tape.step()
    forward_seconds = time.perf_counter() - started
    final_position = tape.solver.positions
    target = initial_position.copy()
    target[:, 2] -= 0.02
    difference = final_position - target
    started = time.perf_counter()
    gradient = tape.backward(difference)
    backward_seconds = time.perf_counter() - started
    summary = {
        "case": "differentiable_cloth",
        "nodes": int(mesh.number_of_nodes),
        "triangles": int(mesh.number_of_cells),
        "steps": args.steps,
        "dt": args.dt,
        "newton_velocity_tolerance": args.newton_velocity_tolerance,
        "linear_solver_relative_tolerance": 1.0e-10,
        "plane_contact": args.plane_contact,
        "loss": float(0.5 * np.sum(difference**2)),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "finite": bool(all(np.isfinite(value).all() for value in gradient.values())),
        "stretch_stiffness_vjp": float(gradient["stretch_stiffness"]),
        "bending_stiffness_vjp": float(gradient["bending_stiffness"]),
        "kappa_vjp": float(gradient["kappa"]),
        "initial_position_vjp_norm": float(np.linalg.norm(gradient["initial_position"])),
        "initial_velocity_vjp_norm": float(np.linalg.norm(gradient["initial_velocity"])),
    }
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "differentiable_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["finite"] or not summary["initial_position_vjp_norm"]:
        raise RuntimeError(summary)


if __name__ == "__main__":
    main()
