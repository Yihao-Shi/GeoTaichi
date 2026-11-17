"""Quasi-static HEX8 cantilever solved by Newton with line search."""

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parents[1]
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
        default=str(CASE_DIR / "OutputData" / "implicit_volume_cantilever"),
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
        geometry="box",
        size=(2.0, 0.4, 0.4),
        divisions=(8, 2, 2),
        element_type="HEX8",
    )
    fem.add_material(
        model="NeoHookean",
        young_modulus=2.0e6,
        poisson_ratio=0.3,
        density=1000.0,
    )

    fixed = DirichletBoundary().add("xmin", components="all", value=0.0)
    traction = NeumannBoundary().add_traction(
        value=(0.0, 0.0, -2.0e4),
        selector=lambda centroid: np.isclose(centroid[:, 0], 2.0),
    )
    fem.add_boundary_condition(dirichlet=fixed, neumann=traction)
    fem.set_solver(
        quasi_static=True,
        dt=1.0,
        step=1,
        residual_tolerance=1.0e-8,
        max_iterations=40,
        line_search=True,
        assemble_type="HashTriplet",
        linear_solver="PCG",
        project_pd=True,
        path=arguments.output_dir,
    )
    result = fem.run(verbose=True)
    print("tip displacement:", np.mean(result.displacement[mesh.node_sets["xmax"]], axis=0))


if __name__ == "__main__":
    main()
