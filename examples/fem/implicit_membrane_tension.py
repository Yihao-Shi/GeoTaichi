"""Quasi-static TRI3 membrane tension solved with Newton line search."""

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import geotaichi as gt
from src.fem import DirichletBoundary, NeumannBoundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument(
        "--output-dir",
        default=str(CASE_DIR / "OutputData" / "implicit_membrane_tension"),
    )
    arguments = parser.parse_args()

    gt.init(
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        log=True,
    )
    fem = gt.FEM(log=True)
    fem.set_configuration(dimension=3, solver_type="Implicit", backend="taichi")
    mesh = fem.add_mesh(
        geometry="rectangle",
        size=(1.0, 0.5),
        divisions=(20, 10),
    )
    fem.add_material(
        model="StVK",
        young_modulus=1.0e6,
        poisson_ratio=0.3,
        density=1000.0,
        thickness=1.0e-3,
    )

    # Clamp the left edge in-plane. A pure membrane has no initial transverse
    # stiffness, so z is fixed here for this planar tension verification.
    fixed = DirichletBoundary().add("xmin", "xy", 0.0).add("all", "z", 0.0)
    force = NeumannBoundary().add_nodal_force("xmax", value=(100.0, 0.0, 0.0), total=True)
    fem.add_boundary_condition(dirichlet=fixed, neumann=force)
    fem.set_solver(
        quasi_static=True,
        dt=1.0,
        step=1,
        max_iterations=30,
        residual_tolerance=1.0e-8,
        line_search=True,
        assemble_type="HashTriplet",
        linear_solver="PCG",
        project_pd=True,
        path=arguments.output_dir,
    )
    result = fem.run(verbose=True)
    print("mean right-edge displacement:", np.mean(result.displacement[mesh.node_sets["xmax"]], axis=0))


if __name__ == "__main__":
    main()
