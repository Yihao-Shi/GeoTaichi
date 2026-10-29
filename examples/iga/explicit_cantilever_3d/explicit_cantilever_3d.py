"""Elastic cantilever; use --solver implicit for the README's Newmark bar."""

import argparse
import json
import os
import platform
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from geotaichi import IGA, init
from src.iga import Cube, DirichletBoundary, Primitives
from src.utils.SolverConsole import runtime_architecture


def _default_arch():
    if platform.system() == "Darwin" or os.path.exists("/dev/nvidia0"):
        return "gpu"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("explicit", "implicit"), default="explicit")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default=_default_arch())
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--output-interval", type=int, default=100)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--damping", type=float, default=0.12, help="Explicit velocity damping only")
    parser.add_argument("--initial-velocity", type=float, default=-0.8)
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    arguments = parser.parse_args()
    if arguments.output_dir is None:
        arguments.output_dir = str(CASE_DIR / "OutputData" / f"{arguments.solver}_cantilever_3d")
    if arguments.steps <= 0 or arguments.output_interval <= 0:
        raise ValueError("--steps and --output-interval must be positive")
    if arguments.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if arguments.damping < 0.0:
        raise ValueError("--damping must be non-negative")

    init(
        dim=3,
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        debug=False,
        log=True,
    )

    iga = IGA(log=True)
    iga.set_configuration(dimension=3, solver_type=arguments.solver.capitalize())

    degree = [2, 2, 2]
    beam = Cube()
    beam.set_parameters(start_point=[0.0, -0.3, -0.3], size=[3.0, 0.6, 0.6])
    beam.generate_knot_u(degree=degree[0], num_ctrlpts=13)
    beam.generate_knot_v(degree=degree[1], num_ctrlpts=5)
    beam.generate_knot_w(degree=degree[2], num_ctrlpts=5)
    beam.generate_ctrlpts()
    beam.generate_weights()
    beam.activate_boundary()

    primitives = Primitives()
    # A transverse velocity impulse excites a clear first-mode-like bending
    # response; the fixed end removes its rigid translation component.
    primitives.append(beam, "beam", init_v=[0.0, 0.0, arguments.initial_velocity])
    primitives.finialize()

    fixed_control_points = np.where(np.isclose(beam.control_points[:, 0], 0.0))[0]
    fixed_dofs = [
        list(3 * fixed_control_points),
        list(3 * fixed_control_points + 1),
        list(3 * fixed_control_points + 2),
    ]
    dirichlet = DirichletBoundary()
    dirichlet.append(fixed_dofs, [0.0] * (3 * len(fixed_control_points)))

    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=dirichlet)
    iga.add_element(degree=degree)
    iga.add_material(young_modulus=8.0e5, poisson_ratio=0.3, density=1000.0)
    iga.set_solver(
        dt=arguments.dt,
        damping=arguments.damping,
        gravity=[0.0, 0.0, 0.0],
        interval=arguments.output_interval,
        step=arguments.steps,
        path=arguments.output_dir,
    )
    if arguments.solver == "implicit":
        iga.set_solver(
            assemble_type="HashTriplet", linear_solver="PCG", residual=1.0e-6, max_iters=50, history_interval=1
        )
    # A failed rerun must not retain an old completion record.
    (Path(arguments.output_dir) / "summary.json").unlink(missing_ok=True)
    iga.run(verbose=False)
    engine = iga.engine
    summary = dict(
        command=shlex.join([sys.executable, *sys.argv]),
        solver=arguments.solver,
        backend=str(runtime_architecture()),
        precision=arguments.default_fp,
        dt=arguments.dt,
        output_interval=arguments.output_interval,
        steps=engine.step_count,
        physical_seconds=engine.time,
        frames=engine.output_count,
    )
    if arguments.solver == "implicit":
        if len(engine.history) != engine.step_count or not all(record["converged"] for record in engine.history):
            raise RuntimeError("Incomplete implicit convergence history")
        summary.update(converged=True, maximum_correction_velocity=max(record["residual"] for record in engine.history))
        (Path(arguments.output_dir) / "convergence_history.json").write_text(
            json.dumps(engine.history, indent=2) + "\n"
        )
    (Path(arguments.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
