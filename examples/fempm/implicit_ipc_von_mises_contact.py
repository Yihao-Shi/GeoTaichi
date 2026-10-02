"""Finite-strain von Mises Direct ULMPM contacting a deformable TET4 FEM pad."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData" / "implicit_ipc_von_mises_contact"),
)
arguments = parser.parse_args()

import geotaichi as gt

gt.init(
    dim=3,
    arch=arguments.arch,
    default_fp=arguments.default_fp,
    log=True,
)

fem = gt.FEM(log=True)
fem.set_configuration(dimension=3, solver_type="Implicit")
fem_mesh = fem.add_mesh(
    geometry="box",
    origin=(0.1, 0.1, 0.0),
    size=(0.8, 0.8, 0.12),
    divisions=(8, 8, 2),
    element_type="TET4",
)
fem.add_material(
    "NeoHookean",
    density=1000.0,
    young_modulus=5.0e5,
    poisson_ratio=0.3,
)
fem.add_boundary_condition(
    {
        "type": "Dirichlet",
        "nodes": fem_mesh.node_sets["zmin"],
        "components": "all",
        "value": 0.0,
    }
)

mpm = gt.MPM(log=True)
mpm.set_configuration(
    dimension=3,
    mpm_backend="Direct",
    solver_type="Implicit",
    configuration="ULMPM",
    domain=[1.0, 1.0, 1.0],
    gravity=[0.0, 0.0, -9.81],
    visualize=True,
)
body = mpm.create_body()
body.add_cube(
    start=[0.34, 0.34, 0.16],
    end=[0.66, 0.66, 0.36],
    spacing=0.04,
    ppc=1,
    init_v=[0.5, 0.0, -1.5],
    name="von_mises_block",
    grid_size=0.04,
    xmin=[0.0, 0.0, 0.0],
    xmax=[1.0, 1.0, 1.0],
)
mpm.add_body(body)
mpm.add_material(
    model="VonMises",
    density=7800.0,
    young_modulus=2.0e5,
    poisson_ratio=0.3,
    YieldStress=1.2e3,
    HardeningModulus=5.0e3,
)
mpm.add_element({"ElementSize": 0.04, "ShapeFunction": "Linear"})

model = gt.FEMPM(fem=fem, mpm=mpm, log=True)
model.set_configuration(
    domain=[1.0, 1.0, 1.0],
    gravity=[0.0, 0.0, -9.81],
    search="BVH",
    log=True,
)
model.set_solver(
    {
        "Timestep": 2.5e-4,
        "SimulationTime": 2.0e-2,
        "SaveInterval": 2.0e-3,
        "SavePath": arguments.output_dir,
        "assemble_type": "HashTriplet",
        "linear_solver": "PCG",
        "project_pd": True,
        "max_iterations": 100,
        "residual_tolerance": 5.0e-4,
        "linear_solver_tolerance": 1.0e-8,
        "linear_solver_relative_tolerance": 1.0e-8,
        "linear_solver_max_iters": 5000,
        "scale": 0.5,
        "enable_step_retry": True,
        "step_retry_max_retries": 3,
        "step_retry_reduction": 0.5,
    },
    log=True,
)
model.add_surface(body_ids=[0])
model.memory_allocate({})
model.choose_contact_model(
    "IPC",
    dhat=0.04,
    dmin=0.004,
    kappa=2.0e5,
    friction_coefficient=0.3,
    epsv=1.0e-3,
    friction_mode="lagged",
    friction_iterations=1,
    project_pd=True,
)
model.run()
