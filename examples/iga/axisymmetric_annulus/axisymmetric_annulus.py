"""Axisymmetric IGA annulus with explicit or implicit time integration.

The NURBS patch represents an ``(r, z)`` meridian.  A small radial body
acceleration exercises the three-dimensional no-swirl material map and the
``2*pi*R`` quadrature measure; no response curve is used for acceptance.
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
    parser.add_argument("--radial-control-points", type=int, default=49)
    parser.add_argument("--axial-control-points", type=int, default=25)
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
        "radial control points": arguments.radial_control_points,
        "axial control points": arguments.axial_control_points,
        "steps": arguments.steps,
        "dt": arguments.dt,
        "output interval": arguments.output_interval,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if min(arguments.radial_control_points, arguments.axial_control_points) < 3:
        raise ValueError("quadratic IGA requires at least three control points per axis")
    if arguments.steps % arguments.output_interval != 0:
        raise ValueError("IGA steps must be divisible by the output interval")
    if not math.isfinite(arguments.radial_acceleration):
        raise ValueError("radial acceleration must be finite")
    return arguments


def _explicit_solver_options():
    return {"damping": 0.1}


def _implicit_solver_options():
    return {
        "newmark": [1.0, 0.5, 1.0],
        "residual": 1.0e-7,
        "max_iters": 30,
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
    from src.iga import DirichletBoundary, Primitives, Rectangle

    gt.init(
        dim=2,
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        debug=False,
        log=True,
    )
    solver_type = arguments.solver.capitalize()
    iga = gt.IGA(log=True)
    iga.set_configuration(
        dimension=2,
        solver_type=solver_type,
        axisymmetric=True,
        axis_offset=0.0,
    )

    annulus = Rectangle()
    annulus.set_parameters(start_point=[0.4, 0.0], size=[0.4, 0.4])
    annulus.generate_knot_u(degree=2, num_ctrlpts=arguments.radial_control_points)
    annulus.generate_knot_v(degree=2, num_ctrlpts=arguments.axial_control_points)
    annulus.generate_ctrlpts()
    annulus.generate_weights()
    primitives = Primitives()
    primitives.append(annulus, "axisymmetric_annulus")
    primitives.finialize()

    bottom = np.flatnonzero(np.isclose(annulus.control_points[:, 1], 0.0))
    fixed = DirichletBoundary()
    fixed.append([list(2 * bottom + 1)], [0.0] * bottom.size)
    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=fixed)
    iga.add_element(degree=[2, 2])
    iga.add_material(
        young_modulus=2.0e5,
        poisson_ratio=0.3,
        density=1000.0,
    )
    solver = {
        "dt": arguments.dt,
        # IGA's legacy engine stores the number of output batches in ``step``
        # and advances ``interval`` substeps per batch.  Keep this public
        # example's --steps option equal to the actual physical substep count.
        "step": arguments.steps // arguments.output_interval,
        "interval": arguments.output_interval,
        "gravity": [arguments.radial_acceleration, 0.0],
        "path": arguments.output_dir,
        "track_energy": True,
    }
    solver.update(SOLVER_OPTIONS[arguments.solver]())
    iga.set_solver(**solver)
    iga.run(verbose=False)
    displacement = iga.engine.patch.control_points.to_numpy() - iga.engine.patch.rest_control_points.to_numpy()
    print("maximum radial displacement:", float(np.max(displacement[:, 0])))


if __name__ == "__main__":
    main()
