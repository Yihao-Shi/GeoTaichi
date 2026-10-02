"""TRI3 membrane falling under gravity with one clamped edge."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import geotaichi as gt
from src.fem import DirichletBoundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument(
        "--output-dir",
        default=str(CASE_DIR / "OutputData" / "explicit_membrane"),
    )
    arguments = parser.parse_args()

    gt.init(
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        log=True,
    )
    fem = gt.FEM(log=True)
    fem.set_configuration(dimension=3, solver_type="Explicit", backend="taichi")
    fem.add_mesh(
        geometry="rectangle",
        size=(1.0, 1.0),
        divisions=(24, 24),
        plane="xy",
    )
    fem.add_material(
        model="StVK",
        young_modulus=5.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        thickness=1.0e-3,
    )
    fem.add_boundary_condition(dirichlet=DirichletBoundary().add("xmin", components="all", value=0.0))
    fem.set_solver(
        dt="auto",
        cfl=0.35,
        step=500,
        interval=25,
        gravity=(0.0, 0.0, -9.8),
        damping=0.5,
        path=arguments.output_dir,
    )
    fem.run(verbose=True)


if __name__ == "__main__":
    main()
