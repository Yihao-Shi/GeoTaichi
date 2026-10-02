"""Direct MPM block contacting a deformable volume or cloth FEM pad with IPC."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument("--search", default="BVH")
parser.add_argument("--assemble-type", default="HashTriplet")
parser.add_argument("--linear-solver", default="PCG")
parser.add_argument("--linear-tolerance", type=float, default=1.0e-6)
parser.add_argument("--linear-relative-tolerance", type=float, default=0.0)
parser.add_argument("--dt", type=float, default=5.0e-4)
parser.add_argument("--time", type=float, default=1.5e-1)
parser.add_argument("--save-interval", type=float, default=2.5e-2)
parser.add_argument(
    "--material",
    choices=("NeoHookean", "DruckerPrager"),
    default="DruckerPrager",
)
parser.add_argument("--spacing", type=float, default=0.04)
parser.add_argument("--ppc", type=int, default=2, help="particles per cell in each direction")
parser.add_argument("--fem-refinement", type=int, default=1)
parser.add_argument("--fem-kind", choices=("volume", "soft", "cloth"), default="volume")
parser.add_argument("--cloth-density", type=float, default=100.0)
parser.add_argument("--cloth-stretch-stiffness", type=float, default=2.0e5)
parser.add_argument("--contact-model", choices=("BarrierIPC", "SemiIPC"), default="BarrierIPC")
parser.add_argument("--cohesion", type=float, default=250.0)
parser.add_argument("--friction-angle", type=float, default=30.0)
parser.add_argument("--gravity", type=float, default=9.81)
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData" / "implicit_ipc_drucker_prager_contact"),
)
parser.add_argument("--initial-speed", type=float, default=0.0)
parser.add_argument("--initial-vz", type=float, default=0.0)
parser.add_argument("--friction", type=float, default=0.3)
parser.add_argument("--start-z", type=float, default=0.141)
parser.add_argument("--block-width", type=float, default=0.24)
parser.add_argument("--block-depth", type=float, default=0.24)
parser.add_argument("--block-height", type=float, default=0.36)
parser.add_argument("--minimum-lateral-spread-ratio", type=float, default=0.0)
parser.add_argument("--grid-zmin", type=float, default=0.0)
parser.add_argument("--scene-manifest", help="Optional Blender SceneManifest metadata")
arguments = parser.parse_args()
if (
    not math.isfinite(arguments.spacing)
    or arguments.spacing <= 0.0
    or arguments.ppc <= 0
    or arguments.fem_refinement <= 0
    or not math.isfinite(arguments.cloth_density)
    or arguments.cloth_density <= 0.0
    or not math.isfinite(arguments.cloth_stretch_stiffness)
    or arguments.cloth_stretch_stiffness <= 0.0
    or not math.isfinite(arguments.linear_tolerance)
    or arguments.linear_tolerance < 0.0
    or not math.isfinite(arguments.linear_relative_tolerance)
    or arguments.linear_relative_tolerance < 0.0
    or arguments.linear_tolerance + arguments.linear_relative_tolerance <= 0.0
    or not math.isfinite(arguments.cohesion)
    or arguments.cohesion < 0.0
    or not 0.0 <= arguments.friction_angle < 90.0
    or not math.isfinite(arguments.gravity)
    or arguments.gravity < 0.0
    or not math.isfinite(arguments.initial_speed)
    or not math.isfinite(arguments.initial_vz)
    or any(
        not math.isfinite(length) or length <= 0.0
        for length in (arguments.block_width, arguments.block_depth, arguments.block_height)
    )
    or arguments.block_width > 0.8
    or arguments.block_depth > 0.8
    or not math.isfinite(arguments.minimum_lateral_spread_ratio)
    or arguments.minimum_lateral_spread_ratio < 0.0
    or not math.isfinite(arguments.grid_zmin)
    or arguments.grid_zmin >= arguments.start_z
    or arguments.start_z <= 0.12
    or arguments.start_z + arguments.block_height >= 1.0
):
    raise ValueError("invalid discretization or material parameter")
for length in (arguments.block_width, arguments.block_depth, arguments.block_height):
    ratio = length / arguments.spacing
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("--spacing must divide each block dimension exactly")

# The shared FEM/MPM/IPC field dtype is selected while implementation modules
# import, before geotaichi.init() can configure the Taichi runtime.
os.environ["GEOTAICHI_REAL_DTYPE"] = arguments.default_fp

import geotaichi as gt
from src.fem import FEMMesh

gt.init(dim=3, arch=arguments.arch, default_fp=arguments.default_fp, log=True)

fem = gt.FEM(log=True)
fem.set_configuration(dimension=3, solver_type="Implicit")
if arguments.fem_kind == "volume":
    fem_mesh = fem.add_mesh(
        geometry="box",
        origin=(0.1, 0.1, 0.0),
        size=(0.8, 0.8, 0.12),
        divisions=(
            8 * arguments.fem_refinement,
            8 * arguments.fem_refinement,
            2 * arguments.fem_refinement,
        ),
        element_type="TET4",
    )
    fem.add_material(
        "NeoHookean",
        density=1000.0,
        young_modulus=5.0e5,
        poisson_ratio=0.3,
    )
    fixed_nodes = fem_mesh.node_sets["zmin"]
elif arguments.fem_kind == "cloth":
    fem_mesh = fem.add_mesh(
        geometry="rectangle",
        origin=(0.1, 0.1, 0.12),
        size=(0.8, 0.8),
        divisions=(16 * arguments.fem_refinement, 16 * arguments.fem_refinement),
        plane="xy",
    )
    fem.add_material(
        "ClothARAP",
        density=arguments.cloth_density,
        stretch_stiffness=arguments.cloth_stretch_stiffness,
        compression_stiffness=arguments.cloth_stretch_stiffness,
        thickness=0.02,
        bending_stiffness=2.0e-2,
        bending_model="Quadratic",
    )
    fixed_nodes = sorted(
        set(fem_mesh.node_sets["xmin"])
        | set(fem_mesh.node_sets["xmax"])
        | set(fem_mesh.node_sets["ymin"])
        | set(fem_mesh.node_sets["ymax"])
    )
else:
    soft_meshes = [
        fem.create_mesh(
            geometry="box",
            origin=(x, y, 0.0),
            size=(0.18, 0.18, 0.12),
            divisions=(2 * arguments.fem_refinement,) * 3,
            element_type="TET4",
        )
        for x in (0.305, 0.515)
        for y in (0.305, 0.515)
    ]
    fem_mesh = fem.add_soft_particle(FEMMesh.concatenate(soft_meshes, name="four_soft_pads"))
    fem.add_material(
        "NeoHookean",
        density=1000.0,
        young_modulus=5.0e4,
        poisson_ratio=0.3,
    )
    fixed_nodes = np.unique(np.concatenate([fem_mesh.node_sets[f"body{body_id}:zmin"] for body_id in range(4)]))
fem.add_boundary_condition(
    {
        "type": "Dirichlet",
        "nodes": fixed_nodes,
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
    gravity=[0.0, 0.0, -arguments.gravity],
    visualize=True,
)
body = mpm.create_body()
block_start = [
    0.5 - 0.5 * arguments.block_width,
    0.5 - 0.5 * arguments.block_depth,
    arguments.start_z,
]
body.add_cube(
    start=block_start,
    end=[
        block_start[0] + arguments.block_width,
        block_start[1] + arguments.block_depth,
        arguments.start_z + arguments.block_height,
    ],
    spacing=arguments.spacing,
    ppc=arguments.ppc,
    init_v=[arguments.initial_speed, 0.0, arguments.initial_vz],
    name=f"{arguments.material.lower()}_block",
    grid_size=arguments.spacing,
    xmin=[0.0, 0.0, arguments.grid_zmin],
    xmax=[1.0, 1.0, 1.0],
)
mpm.add_body(body)
material = dict(
    model=arguments.material,
    density=1000.0,
    young_modulus=2.0e5,
    poisson_ratio=0.3,
)
if arguments.material == "DruckerPrager":
    material.update(
        Cohesion=arguments.cohesion,
        FrictionAngle=arguments.friction_angle,
        DilationAngle=arguments.friction_angle,
        dpType="Circumscribed",
    )
mpm.add_material(**material)
mpm.add_element({"ElementSize": arguments.spacing, "ShapeFunction": "Linear"})

model = gt.FEMPM(fem=fem, mpm=mpm, log=True)
model.set_configuration(
    domain=[1.0, 1.0, 1.0],
    gravity=[0.0, 0.0, -arguments.gravity],
    search=arguments.search,  # "LinkedCell" is equivalent contact physics.
    log=True,
)
model.set_solver(
    {
        "Timestep": arguments.dt,
        "SimulationTime": arguments.time,
        "SaveInterval": arguments.save_interval,
        "SavePath": arguments.output_dir,
        "assemble_type": arguments.assemble_type,  # e.g. HashTriplet or COO
        "linear_solver": arguments.linear_solver,  # e.g. PCG or Scipy
        "project_pd": True,
        "max_iterations": 100,
        "residual_tolerance": 5.0e-4,
        "linear_solver_tolerance": arguments.linear_tolerance,
        "linear_solver_relative_tolerance": arguments.linear_relative_tolerance,
        "linear_solver_max_iters": 5000,
        "contact_all_mpm_particles": True,
        "scale": 0.5,
        "enable_step_retry": arguments.contact_model == "BarrierIPC",
        "step_retry_max_retries": 3,
        "step_retry_reduction": 0.5,
    },
    log=True,
)
model.add_surface(body_ids=[0])
model.memory_allocate({})
model.choose_contact_model(
    arguments.contact_model,
    dhat=0.5 * arguments.spacing,
    dmin=0.5 * arguments.spacing / arguments.ppc,
    kappa=2.0e5,
    friction_coefficient=arguments.friction,
    epsv=1.0e-3,
    friction_mode="lagged",
    friction_iterations=1,
    project_pd=True,
)
peak_contacts = [0]
peak_candidates = [0]
minimum_distance = [np.inf]


def record_contact(engine):
    contact = engine.last_step_record["contact"]
    peak_contacts[0] = max(peak_contacts[0], int(contact["active_contacts"]))
    peak_candidates[0] = max(peak_candidates[0], int(contact["candidate_contacts"]))
    minimum_distance[0] = min(minimum_distance[0], float(contact["minimum_distance"]))


result = model.run(verbose=False, postprocessing=[record_contact])
fem_position = model.enginer.fem.state.position.to_numpy()
mpm_position = model.enginer.mpm.particle.x.to_numpy()
mpm_span = np.ptp(mpm_position, axis=0)
summary = {
    "case": f"ordinary_mpm_{arguments.fem_kind}_fem_ipc",
    "contact_model": arguments.contact_model,
    "assemble_type": arguments.assemble_type,
    "linear_solver": arguments.linear_solver,
    "mpm_material": arguments.material,
    "cloth_density": float(arguments.cloth_density) if arguments.fem_kind == "cloth" else None,
    "cloth_stretch_stiffness": (float(arguments.cloth_stretch_stiffness) if arguments.fem_kind == "cloth" else None),
    "minimum_lateral_spread_ratio": float(arguments.minimum_lateral_spread_ratio),
    "grid_zmin": float(arguments.grid_zmin),
    "fem_nodes": int(fem_mesh.number_of_nodes),
    "fem_elements": int(fem_mesh.number_of_cells),
    "mpm_particles": int(model.enginer.mpm.particleNum[0]),
    "steps": int(result["step"]),
    "completed_time": float(result["time"]),
    "converged": bool(result["converged"]),
    "maximum_candidate_contacts": int(peak_candidates[0]),
    "maximum_active_contacts": int(peak_contacts[0]),
    "minimum_contact_distance": float(minimum_distance[0]),
    "fem_bounds": [fem_position.min(axis=0).tolist(), fem_position.max(axis=0).tolist()],
    "mpm_bounds": [mpm_position.min(axis=0).tolist(), mpm_position.max(axis=0).tolist()],
    "mpm_span": mpm_span.tolist(),
    "lateral_spread_ratio": float(max(mpm_span[0] / arguments.block_width, mpm_span[1] / arguments.block_depth)),
    "finite": bool(np.isfinite(fem_position).all() and np.isfinite(mpm_position).all()),
}
output_path = Path(arguments.output_dir)
(output_path / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if (
    not summary["converged"]
    or not summary["finite"]
    or not summary["maximum_active_contacts"]
    or summary["lateral_spread_ratio"] < arguments.minimum_lateral_spread_ratio
    or not np.isclose(summary["completed_time"], arguments.time)
):
    raise RuntimeError(summary)
