"""Falling cloth FEM with IPC contact and friction."""

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
    default=str(CASE_DIR / "OutputData" / "implicit_cloth_contact"),
)
arguments = parser.parse_args()

import geotaichi as gt


gt.init(arch=arguments.arch, default_fp=arguments.default_fp, log=True)

cloth = gt.FEM("Cloth FEM IPC", log=True)
cloth.set_configuration(dimension=3, solver_type="Implicit")
mesh = cloth.add_mesh(
    geometry="rectangle",
    size=(1.0, 1.0),
    divisions=(24, 24),
    plane="xy",
)
mesh.points[:, 2] += 0.15
cloth.add_material(
    "ClothARAP",
    stretch_stiffness=5.0e4,
    compression_stiffness=8.0e4,
    density=1000.0,
    thickness=2.0e-3,
    bending_stiffness=2.0e8,
    bending_poisson_ratio=0.3,
)
cloth.add_contact(
    "IPC",
    self_contact=True,
    planes=[{"point": (0.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)}],
    dhat=2.0e-2,
    dmin=2.0e-3,
    kappa=5.0e4,
    friction_coefficient=0.3,
    epsv=1.0e-3,
    ccd_safety=0.9,
)
cloth.set_solver(
    quasi_static=False,
    dt=2.0e-3,
    step=300,
    gravity=(0.0, 0.0, -9.81),
    damping=0.02,
    max_iterations=50,
    residual_tolerance=1.0e-7,
    line_search=True,
    assemble_type="HashTriplet",
    linear_solver="PCG",
    project_pd=True,
    path=arguments.output_dir,
    output_interval=10,
)

result = cloth.run()
print(result.history[-1]["contact"])
