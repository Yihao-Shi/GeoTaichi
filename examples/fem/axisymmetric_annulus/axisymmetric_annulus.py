"""Axisymmetric FEM annulus with explicit or implicit time integration.

The stored coordinates are the meridional ``(r, z)`` section.  A small radial
body acceleration exercises the hoop stretch and revolved volume without
turning this capability example into a response-curve benchmark.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


MODE_DEFAULTS = {
    "explicit": {"steps": 2_000, "dt": 2.0e-5},
    "implicit": {"steps": 20, "dt": 1.0e-2},
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("explicit", "implicit"), default="explicit")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--radial-divisions", type=int, default=48)
    parser.add_argument("--axial-divisions", type=int, default=24)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--output-interval", type=int)
    parser.add_argument("--radial-acceleration", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default=str(CASE_DIR / "OutputData" / "axisymmetric_annulus"),
    )
    arguments = parser.parse_args()
    defaults = MODE_DEFAULTS[arguments.solver]
    if arguments.steps is None:
        arguments.steps = defaults["steps"]
    if arguments.dt is None:
        arguments.dt = defaults["dt"]
    if arguments.output_interval is None:
        arguments.output_interval = max(1, arguments.steps // 20)
    positive = {
        "radial divisions": arguments.radial_divisions,
        "axial divisions": arguments.axial_divisions,
        "steps": arguments.steps,
        "dt": arguments.dt,
        "output interval": arguments.output_interval,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(arguments.radial_acceleration):
        raise ValueError("radial acceleration must be finite")
    return arguments


def _explicit_solver_options():
    return {"damping": 0.1, "enforce_stable_time_step": True}


def _implicit_solver_options():
    return {
        "newmark": (0.25, 0.5),
        "max_iterations": 30,
        "residual_tolerance": 1.0e-7,
        "line_search": True,
        "assemble_type": "HashTriplet",
        "linear_solver": "PCG",
        "project_pd": True,
    }


SOLVER_OPTIONS = {
    "explicit": _explicit_solver_options,
    "implicit": _implicit_solver_options,
}


def main():
    arguments = parse_arguments()

    import geotaichi as gt
    from src.fem import DirichletBoundary

    gt.init(
        dim=2,
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        debug=False,
        log=True,
    )
    solver_type = arguments.solver.capitalize()
    fem = gt.FEM(log=True)
    fem.set_configuration(
        dimension=2,
        solver_type=solver_type,
        axisymmetric=True,
        axis_offset=0.0,
        backend="taichi",
    )
    fem.add_mesh(
        geometry="rectangle",
        origin=(0.4, 0.0, 0.0),
        size=(0.4, 0.4),
        divisions=(arguments.radial_divisions, arguments.axial_divisions),
        plane="xy",
    )
    fem.add_material(
        model="NeoHookean",
        young_modulus=2.0e5,
        poisson_ratio=0.3,
        density=1000.0,
    )
    fixed = DirichletBoundary().add("ymin", components="y", value=0.0)
    fem.add_boundary_condition(dirichlet=fixed)
    solver = {
        "dt": arguments.dt,
        "step": arguments.steps,
        "interval": arguments.output_interval,
        "gravity": (arguments.radial_acceleration, 0.0, 0.0),
        "path": arguments.output_dir,
        "track_energy": True,
    }
    solver.update(SOLVER_OPTIONS[arguments.solver]())
    fem.set_solver(**solver)
    result = fem.run(verbose=False)
    radial_displacement = np.asarray(result.displacement)[:, 0]
    print("maximum radial displacement:", float(np.max(radial_displacement)))


if __name__ == "__main__":
    main()
