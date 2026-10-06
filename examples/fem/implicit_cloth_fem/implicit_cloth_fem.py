"""Implicit ARAP cloth FEM with quadratic bending."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData" / "implicit_cloth_fem"),
)
arguments = parser.parse_args()

import geotaichi as gt

from src.fem import DirichletBoundary


gt.init(arch=arguments.arch, default_fp=arguments.default_fp, log=True)

fem = gt.FEM(title="Implicit cloth FEM", log=True)
fem.set_configuration(dimension=3, solver_type="Implicit")
mesh = fem.add_mesh(
    {
        "Geometry": "Rectangle",
        "Size": (1.0, 1.0),
        "Divisions": (24, 24),
        "Plane": "xy",
    }
)
fem.add_material(
    "ClothARAP",
    stretch_stiffness=5.0e4,
    compression_stiffness=8.0e4,
    density=1000.0,
    thickness=1.0e-3,
    bending_stiffness=2.0e8,
    bending_poisson_ratio=0.3,
)
top_corners = mesh.select_nodes(
    selector=lambda points: (points[:, 1] > 1.0 - 1.0e-10) & ((points[:, 0] < 1.0e-10) | (points[:, 0] > 1.0 - 1.0e-10))
)
fem.add_boundary_condition(dirichlet=DirichletBoundary().add(top_corners, "all", 0.0))
fem.set_solver(
    quasi_static=False,
    dt=2.0e-3,
    step=100,
    gravity=(0.0, 0.0, -9.81),
    damping=0.2,
    max_iterations=30,
    residual_tolerance=1.0e-7,
    line_search=True,
    project_pd=True,
    assemble_type="HashTriplet",
    linear_solver="PCG",
    interval=10,
    path=arguments.output_dir,
)
fem.run()
