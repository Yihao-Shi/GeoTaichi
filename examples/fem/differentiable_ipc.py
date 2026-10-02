"""Elastic block sliding on an IPC plane: terminal loss and trajectory gradient.

Run with: python -m examples.fem.differentiable_ipc
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import taichi as ti

from src.fem import DifferentiableFEM, FEM


def build_problem(
    young=100.0,
    kappa=20.0,
    gravity=(0.0, 0.0, -9.8),
    velocity=(0.2, 0.0, 0.0),
    divisions=1,
    steps=3,
    dt=0.01,
    newton_velocity_tolerance=1.0e-6,
):
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    mesh = fem.add_mesh(
        geometry="box",
        size=(0.1, 0.1, 0.1),
        divisions=(divisions, divisions, divisions),
        element_type="TET4",
    )
    mesh.points[:, 2] += 0.025
    mesh.rest_shape[:, 2] += 0.025
    fem.add_material("StVK", young_modulus=young, poisson_ratio=0.3, density=1000.0)
    fem.add_contact(
        "IPC",
        self_contact=False,
        planes=[((0, 0, 0), (0, 0, 1))],
        dhat=0.05,
        kappa=kappa,
        friction_coefficient=0.0,
    )
    fem.set_solver(
        dt=dt,
        step=steps,
        gravity=gravity,
        initial_velocity=velocity,
        damping=0.1,
        max_iterations=80,
        residual_tolerance=1.0e-10,
        absolute_tolerance=1.0e-10,
        correction_velocity_tolerance=newton_velocity_tolerance,
        linear_solver="PCG",
        linear_solver_relative_tolerance=1.0e-10,
        assemble_type="HashTriplet",
        project_pd=True,
    )
    return DifferentiableFEM(fem.build())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--divisions", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--young", type=float, default=100.0)
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--newton-velocity-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if (
        args.divisions < 1
        or args.steps < 1
        or args.dt <= 0.0
        or args.young <= 0.0
        or args.kappa <= 0.0
        or args.newton_velocity_tolerance <= 0.0
    ):
        parser.error("divisions, steps, dt, stiffnesses, and Newton tolerance must be positive")

    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f64)
    simulation = build_problem(
        divisions=args.divisions,
        steps=args.steps,
        dt=args.dt,
        young=args.young,
        kappa=args.kappa,
        newton_velocity_tolerance=args.newton_velocity_tolerance,
    )
    started = time.perf_counter()
    for _ in range(args.steps):
        simulation.step()
    forward_seconds = time.perf_counter() - started
    positions = simulation.solver.positions
    target = simulation.solver.reference_positions.copy()
    target[:, 0] += 0.01
    difference = positions - target
    started = time.perf_counter()
    gradient = simulation.backward(difference)
    backward_seconds = time.perf_counter() - started
    summary = {
        "case": "differentiable_fem_fixed_plane_ipc",
        "nodes": int(simulation.node_count),
        "tetrahedra": int(simulation.solver.mesh.number_of_cells),
        "divisions": args.divisions,
        "steps": args.steps,
        "dt": args.dt,
        "newton_velocity_tolerance": args.newton_velocity_tolerance,
        "linear_solver_relative_tolerance": 1.0e-10,
        "young_modulus": args.young,
        "kappa": args.kappa,
        "loss": float(0.5 * np.sum(difference**2)),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "finite": bool(all(np.isfinite(value).all() for value in gradient.values())),
        "kappa_vjp": float(gradient["kappa"]),
        "young_modulus_vjp": float(gradient["young_modulus"]),
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
