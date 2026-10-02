"""Polyhedral AffineBody impact on clamped cloth or volume FEM through monolithic IPC."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument(
    "--mesh-path",
    default=str(REPO_ROOT / "assets" / "mesh" / "AffineBody" / "lowpoly_sphere.obj"),
)
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData"),
)
parser.add_argument("--divisions", type=int, default=32)
parser.add_argument("--fem-kind", choices=("cloth", "volume"), default="cloth")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--dt", type=float, default=1.0e-3)
parser.add_argument("--save-interval", type=float, default=2.0e-2)
parser.add_argument("--start-x", type=float, default=0.5)
parser.add_argument("--start-height", type=float, default=0.26)
parser.add_argument("--scale-factor", type=float, default=0.08)
parser.add_argument("--initial-vx", type=float, default=0.0)
parser.add_argument("--initial-vz", type=float, default=-0.25)
parser.add_argument("--friction-coefficient", type=float, default=0.0)
parser.add_argument("--objective-x-weight", type=float, default=0.0)
parser.add_argument("--correction-velocity-tolerance", type=float, default=1.0e-4)
parser.add_argument("--friction-iterations", type=int, default=1)
parser.add_argument("--friction-max-iterations", type=int, default=100)
parser.add_argument("--max-iterations", type=int, default=200)
parser.add_argument("--linear-solver-tolerance", type=float, default=1.0e-10)
parser.add_argument("--linear-solver-relative-tolerance", type=float, default=1.0e-8)
parser.add_argument("--contact-model", choices=("BarrierIPC", "SemiIPC"), default="BarrierIPC")
parser.add_argument("--assemble-type", choices=("COO", "HashTriplet"), default="HashTriplet")
parser.add_argument("--differentiable", action="store_true")
parser.add_argument("--disable-step-retry", action="store_true")
parser.add_argument("--allow-no-contact", action="store_true", help="Only for compile/smoke gates")
parser.add_argument("--verbose", action="store_true")
arguments = parser.parse_args()
if (
    arguments.divisions < 2
    or not 0.1 < arguments.start_x < 0.9
    or arguments.steps <= 0
    or arguments.dt <= 0.0
    or not np.isfinite(arguments.scale_factor)
    or arguments.scale_factor <= 0.0
    or arguments.save_interval <= 0.0
    or arguments.correction_velocity_tolerance <= 0.0
    or arguments.friction_iterations == 0
    or arguments.friction_iterations < -1
    or arguments.friction_max_iterations <= 0
    or arguments.max_iterations <= 0
    or arguments.linear_solver_tolerance < 0.0
    or arguments.linear_solver_relative_tolerance < 0.0
):
    parser.error("mesh/time/correction/friction limits must be positive and solver tolerances non-negative")
if arguments.friction_coefficient < 0.0:
    parser.error("friction coefficient must be non-negative")

import geotaichi as gt

gt.init(arch=arguments.arch, default_fp=arguments.default_fp, log=True)

mesh_path = Path(arguments.mesh_path).expanduser().resolve()
with mesh_path.open(encoding="utf-8", errors="ignore") as mesh_file:
    mesh_lines = tuple(mesh_file)
mesh_vertices = sum(line.lstrip().startswith("v ") for line in mesh_lines)
mesh_faces = sum(max(len(line.split()) - 3, 0) for line in mesh_lines if line.lstrip().startswith("f "))
if not mesh_vertices or not mesh_faces:
    raise ValueError(f"AffineBody OBJ has no vertices or faces: {mesh_path}")

dem = gt.DEM(log=True)
dem.set_configuration(
    domain=[2.0, 2.0, 2.0],
    scheme="AffineBody",
    search="BVH",
    gravity=[0.0, 0.0, -9.81],
    visualize=False,
    log=True,
)
dem.set_affine_body_parameters(
    assemble_type=arguments.assemble_type,
    young_modulus=2.0e4,
    local_damping=0.0,
    contact_damping_stiffness=0.0,
    hessian_shift=0.0,
    friction_mode="lagged",
    friction_iterations=1,
)
dem.memory_allocate(
    {
        "max_material_number": 1,
        "max_affine_body_number": 1,
        "surface_node_number": max(64, mesh_vertices),
        "max_point_triangle_pairs": 0,
        "max_edge_edge_pairs": 0,
        "body_coordination_number": 16,
        "wall_coordination_number": 1,
        "compaction_ratio": [1.0, 1.0],
    },
    log=True,
)
dem.add_attribute(materialID=0, attribute={"Density": 20.0})
dem.add_template(
    {
        "Name": "affine_impact_body",
        "TemplateType": "AffineBody",
        "Object": gt.polyhedron(file=str(mesh_path)),
    }
)
dem.create_body(
    {
        "BodyType": "AffineBody",
        "Template": [
            {
                "Name": "affine_impact_body",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [arguments.start_x, 0.5, arguments.start_height],
                "ScaleFactor": arguments.scale_factor,
                "InitialVelocity": [arguments.initial_vx, 0.0, arguments.initial_vz],
            }
        ],
    }
)

fem = gt.FEM(log=True)
fem.set_configuration(dimension=3, solver_type="Implicit")
if arguments.fem_kind == "cloth":
    surface = fem.add_mesh(
        {
            "Geometry": "Rectangle",
            "Size": (1.0, 1.0),
            "Divisions": (arguments.divisions, arguments.divisions),
            "ElementType": "TRI3",
        }
    )
    fem.add_material(
        "ClothARAP",
        density=1.0,
        stretch_stiffness=2.0e4,
        compression_stiffness=2.0e4,
        thickness=0.05,
        bending_stiffness=2.0e-2,
        bending_model="Quadratic",
    )
    boundary_nodes = sorted(
        set(surface.node_sets["xmin"])
        | set(surface.node_sets["xmax"])
        | set(surface.node_sets["ymin"])
        | set(surface.node_sets["ymax"])
    )
else:
    surface = fem.add_mesh(
        geometry="box",
        size=(1.0, 1.0, 0.08),
        divisions=(arguments.divisions, arguments.divisions, max(2, arguments.divisions // 8)),
        element_type="TET4",
    )
    fem.add_material("NeoHookean", density=1000.0, young_modulus=2.0e4, poisson_ratio=0.3)
    boundary_nodes = surface.node_sets["zmin"]
fem.add_boundary_condition(
    {
        "type": "Dirichlet",
        "nodes": boundary_nodes,
        "components": "all",
        "value": 0.0,
    }
)

coupling = gt.FEDEM(dem=dem, fem=fem, log=True)
coupling.set_configuration(domain=[2.0, 2.0, 2.0], gravity=[0.0, 0.0, -9.81], search="BVH", log=True)
coupling.set_solver(
    {
        "Timestep": arguments.dt,
        "SimulationTime": arguments.steps * arguments.dt,
        "SaveInterval": arguments.save_interval,
        "SavePath": arguments.output_dir,
        "assemble_type": arguments.assemble_type,
        "linear_solver": "PCG",
        "project_pd": True,
        "project_bending_pd": True,
        "max_iterations": arguments.max_iterations,
        "correction_velocity_tolerance": arguments.correction_velocity_tolerance,
        "linear_solver_tolerance": arguments.linear_solver_tolerance,
        "linear_solver_relative_tolerance": arguments.linear_solver_relative_tolerance,
        "linear_solver_max_iters": 10000,
        "enable_step_retry": (
            arguments.contact_model == "BarrierIPC"
            and not arguments.differentiable
            and not arguments.disable_step_retry
        ),
        "step_retry_max_retries": 3,
    },
    log=True,
)
coupling.add_surface()
coupling.memory_allocate(
    {
        "max_contact_pairs": 8192,
        "max_point_triangle_pairs": 32768,
        "max_edge_edge_pairs": 65536,
        "max_facet_cell_pairs": 65536,
        "contact_coordination_number": 64,
    }
)
coupling.choose_contact_model(
    arguments.contact_model,
    dhat=0.02,
    dmin=1.0e-3,
    kappa=2.0e4,
    friction_coefficient=arguments.friction_coefficient,
    epsv=1.0e-3,
    friction_mode="lagged",
    friction_iterations=arguments.friction_iterations if arguments.friction_coefficient > 0.0 else 1,
    friction_max_iterations=arguments.friction_max_iterations,
    friction_tolerance=1.0e-6,
    project_pd=True,
)
coupling.add_ipc_property(
    AffineBody=0,
    FEMbody=0,
    property={
        "dhat": 0.02,
        "dmin": 1.0e-3,
        "kappa": 4.0e4,
        "friction_coefficient": arguments.friction_coefficient,
        "epsv": 1.0e-3,
    },
)
peak_contacts = [0]


def record_contact(engine):
    peak_contacts[0] = max(peak_contacts[0], int(engine.last_step_record["contact"]["active_contacts"]))


if arguments.differentiable:
    trajectory = coupling.differentiable(arguments.steps)
    started = time.perf_counter()
    for _ in range(arguments.steps):
        trajectory.step()
        record_contact(coupling.enginer)
    forward_seconds = time.perf_counter() - started
    fem_seed = np.zeros((surface.number_of_nodes, 3), dtype=np.float64)
    fem_seed[:, 2] = 1.0 / surface.number_of_nodes
    fem_seed[:, 0] = arguments.objective_x_weight / surface.number_of_nodes
    affine_seed = np.zeros((coupling.enginer.affine_controls, 3), dtype=np.float64)
    affine_seed[:, 2] = 1.0 / coupling.enginer.affine_controls
    affine_seed[:, 0] = arguments.objective_x_weight / coupling.enginer.affine_controls
    started = time.perf_counter()
    gradient = trajectory.backward(fem_seed, affine_seed)
    backward_seconds = time.perf_counter() - started
    result = {
        "step": coupling.enginer.step_count,
        "time": coupling.enginer.time,
        "converged": True,
    }
else:
    started = time.perf_counter()
    result = coupling.run(verbose=arguments.verbose, postprocessing=[record_contact])
    forward_seconds = time.perf_counter() - started
    backward_seconds = 0.0
summary = {
    "case": f"affine_body_{arguments.fem_kind}_ipc",
    "fem_kind": arguments.fem_kind,
    "contact_model": arguments.contact_model,
    "assemble_type": arguments.assemble_type,
    "linear_solver": "PCG",
    "friction_coefficient": float(arguments.friction_coefficient),
    "objective_x_weight": float(arguments.objective_x_weight),
    "affine_mesh": str(mesh_path),
    "affine_mesh_vertices": int(mesh_vertices),
    "affine_mesh_faces": int(mesh_faces),
    "affine_scale_factor": float(arguments.scale_factor),
    "fem_nodes": int(surface.number_of_nodes),
    "fem_elements": int(surface.number_of_cells),
    "steps": int(result["step"]),
    "completed_time": float(result["time"]),
    "converged": bool(result["converged"]),
    "forward_seconds": float(forward_seconds),
    "backward_seconds": float(backward_seconds),
    "differentiable": bool(arguments.differentiable),
    "maximum_active_contacts": int(peak_contacts[0]),
    "finite_fem": bool(np.isfinite(coupling.enginer.fem.state.position.to_numpy()).all()),
    "finite_affine": bool(np.isfinite(coupling.enginer.affine.y.to_numpy()).all()),
    "loss": float(
        np.mean(coupling.enginer.fem.state.position.to_numpy()[:, 2])
        + np.mean(coupling.enginer.affine.y.to_numpy()[: coupling.enginer.affine_controls, 2])
        + arguments.objective_x_weight
        * (
            np.mean(coupling.enginer.fem.state.position.to_numpy()[:, 0])
            + np.mean(coupling.enginer.affine.y.to_numpy()[: coupling.enginer.affine_controls, 0])
        )
    ),
}
if arguments.fem_kind == "cloth":
    summary.update(
        cloth_nodes=summary["fem_nodes"],
        cloth_triangles=summary["fem_elements"],
        finite_cloth=summary["finite_fem"],
    )
if arguments.differentiable:
    material_vjp = gradient["stretch_stiffness"] if arguments.fem_kind == "cloth" else gradient["young_modulus"]
    summary.update(
        fem_initial_position_vjp_norm=float(np.linalg.norm(gradient["fem_initial_position"])),
        fem_initial_velocity_vjp_norm=float(np.linalg.norm(gradient["fem_initial_velocity"])),
        affine_initial_position_vjp_norm=float(np.linalg.norm(gradient["affine_initial_position"])),
        affine_initial_velocity_vjp_norm=float(np.linalg.norm(gradient["affine_initial_velocity"])),
        initial_vz_vjp=float(np.sum(gradient["affine_initial_velocity"][:, 2])),
        initial_vx_vjp=float(np.sum(gradient["affine_initial_velocity"][:, 0])),
        fem_material_vjp=float(material_vjp),
        affine_young_vjp=gradient["affine_young_modulus"].tolist(),
        gravity_vjp=gradient["gravity"].tolist(),
        mixed_friction_vjp=gradient["mixed_friction_coefficient"].tolist(),
        gradient_finite=bool(
            all(
                np.isfinite(value).all()
                for value in (
                    gradient["fem_initial_position"],
                    gradient["fem_initial_velocity"],
                    gradient["affine_initial_position"],
                    gradient["affine_initial_velocity"],
                    gradient["affine_young_modulus"],
                    gradient["gravity"],
                    gradient["mixed_friction_coefficient"],
                )
            )
            and np.isfinite(material_vjp)
        ),
    )
    if arguments.fem_kind == "cloth":
        summary.update(
            stretch_stiffness_vjp=float(gradient["stretch_stiffness"]),
            bending_stiffness_vjp=float(gradient["bending_stiffness"]),
        )
    else:
        summary["young_modulus_vjp"] = float(gradient["young_modulus"])
output_path = Path(arguments.output_dir)
output_path.mkdir(parents=True, exist_ok=True)
(output_path / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if (
    not summary["converged"]
    or not summary["finite_fem"]
    or not summary["finite_affine"]
    or (not arguments.allow_no_contact and not summary["maximum_active_contacts"])
    or (arguments.differentiable and not summary["gradient_finite"])
    or not np.isclose(summary["completed_time"], arguments.steps * arguments.dt)
):
    raise RuntimeError(summary)
