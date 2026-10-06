"""Twisting cloth with GeoTaichi implicit FEM IPC.

Both opposite boundary strips are prescribed to rotate in opposite directions
about the centreline.  The cloth remains a single TRI3 body, so IPC self
contact (PT/EE) is exercised while the time-dependent Dirichlet history stays
device resident during the solve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def grid_size(value):
    result = positive_integer(value)
    if result < 2:
        raise argparse.ArgumentTypeError("grid must contain at least two vertices per side")
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument(
    "--grid",
    type=grid_size,
    default=50,
    help="vertices per side (Newton uses a 50 by 50 cloth)",
)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--dt", type=float, default=1.0 / 60.0)
parser.add_argument(
    "--angular-velocity",
    type=float,
    default=np.pi / 5.0,
    help="per-edge speed; opposite edges accumulate 720 degrees relative rotation in 10 seconds",
)
parser.add_argument("--rotation-end-time", type=float, default=10.0)
parser.add_argument("--stretch-stiffness", type=float, default=1.0e5)
parser.add_argument("--bending-stiffness", type=float, default=3.0e4)
parser.add_argument(
    "--bending-model",
    choices=("Quadratic", "Dihedral", "None"),
    default="Dihedral",
    help="cloth bending model (default: Dihedral)",
)
parser.add_argument("--friction-coefficient", type=float, default=0.2)
parser.add_argument("--contact-model", choices=("BarrierIPC", "SemiIPC"), default="BarrierIPC")
parser.add_argument(
    "--project-pd",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="project the cloth membrane Hessian to PSD",
)
parser.add_argument("--project-bending-pd", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--contact-project-pd", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--contact",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="disable only for isolating FEM kernel compilation",
)
parser.add_argument("--output-interval", type=positive_integer, default=10)
parser.add_argument("--assemble-type", default="HashTriplet")
parser.add_argument("--linear-solver", default="PCG")
parser.add_argument("--linear-relative-tolerance", type=float, default=1.0e-5)
parser.add_argument("--newton-velocity-tolerance", type=float, default=1.0e-2)
parser.add_argument("--max-iterations", type=positive_integer, default=250)
parser.add_argument("--broad-phase", default="BVH", choices=("BVH", "LinkedCell"))
parser.add_argument("--max-point-triangle-pairs", type=positive_integer, default=None)
parser.add_argument("--max-edge-edge-pairs", type=positive_integer, default=None)
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData" / "implicit_cloth_twist"),
)
arguments = parser.parse_args()

if arguments.steps < 0:
    parser.error("--steps must be non-negative")
if not np.isfinite(arguments.dt) or arguments.dt <= 0.0:
    parser.error("--dt must be finite and positive")
if not np.isfinite(arguments.linear_relative_tolerance) or arguments.linear_relative_tolerance < 0.0:
    parser.error("--linear-relative-tolerance must be finite and non-negative")
if not np.isfinite(arguments.newton_velocity_tolerance) or arguments.newton_velocity_tolerance <= 0.0:
    parser.error("--newton-velocity-tolerance must be finite and positive")
if not np.isfinite(arguments.bending_stiffness) or arguments.bending_stiffness < 0.0:
    parser.error("--bending-stiffness must be finite and non-negative")
if not np.isfinite(arguments.stretch_stiffness) or arguments.stretch_stiffness <= 0.0:
    parser.error("--stretch-stiffness must be finite and positive")
if not np.isfinite(arguments.friction_coefficient) or arguments.friction_coefficient < 0.0:
    parser.error("--friction-coefficient must be finite and non-negative")
if not np.isfinite(arguments.angular_velocity):
    parser.error("--angular-velocity must be finite")
if not np.isfinite(arguments.rotation_end_time) or arguments.rotation_end_time < 0.0:
    parser.error("--rotation-end-time must be finite and non-negative")

import geotaichi as gt
from src.fem import DirichletBoundary


def rotating_displacement(sign, angular_velocity, end_time):
    """Return a prescribed displacement for one rotating edge."""

    def value(time, coordinates):
        theta = sign * angular_velocity * min(max(float(time), 0.0), end_time)
        cosine = np.cos(theta)
        sine = np.sin(theta)
        x = coordinates[:, 0]
        z = coordinates[:, 2]
        rotated = coordinates.copy()
        rotated[:, 0] = cosine * x + sine * z
        rotated[:, 2] = -sine * x + cosine * z
        return rotated - coordinates

    return value


gt.init(arch=arguments.arch, default_fp=arguments.default_fp, log=True)

grid = arguments.grid
width = 0.5
cloth = gt.FEM(title="Twisting cloth FEM IPC", log=True)
cloth.set_configuration(dimension=3, solver_type="Implicit")
mesh = cloth.add_mesh(
    geometry="rectangle",
    size=(width, width),
    divisions=(grid - 1, grid - 1),
    origin=(-0.5 * width, -0.5 * width, 0.0),
    plane="xy",
)

# MeshGenerator uses x-major ordering.  Newton rotates the two strips whose
# vertices span x, so select the first/last y entry from every x row.
left_side = np.arange(0, grid * grid, grid, dtype=np.int32)
right_side = left_side + grid - 1
boundary = DirichletBoundary()
boundary.add(
    left_side,
    "all",
    rotating_displacement(-1.0, arguments.angular_velocity, arguments.rotation_end_time),
)
boundary.add(
    right_side,
    "all",
    rotating_displacement(1.0, arguments.angular_velocity, arguments.rotation_end_time),
)
cloth.add_boundary_condition(dirichlet=boundary)

cloth.add_material(
    "ClothARAP",
    stretch_stiffness=arguments.stretch_stiffness,
    compression_stiffness=arguments.stretch_stiffness,
    density=0.2,
    thickness=2.0e-3,
    bending_stiffness=arguments.bending_stiffness,
    bending_poisson_ratio=0.3,
    bending_model=arguments.bending_model,
)
if arguments.contact:
    cloth.add_contact(
        arguments.contact_model,
        self_contact=True,
        broad_phase=arguments.broad_phase,
        dhat=2.0e-3,
        dmin=1.5e-3,
        kappa=1.0e3,
        friction_coefficient=arguments.friction_coefficient,
        epsv=1.0e-3,
        ccd_safety=0.9,
        project_pd=arguments.contact_project_pd,
        max_point_triangle_pairs=arguments.max_point_triangle_pairs,
        max_edge_edge_pairs=arguments.max_edge_edge_pairs,
    )
cloth.set_solver(
    quasi_static=False,
    dt=arguments.dt,
    step=arguments.steps,
    gravity=(0.0, 0.0, 0.0),
    damping=0.02,
    max_iterations=arguments.max_iterations,
    residual_tolerance=1.0e-7,
    correction_velocity_tolerance=arguments.newton_velocity_tolerance,
    line_search=True,
    project_pd=arguments.project_pd,
    project_bending_pd=arguments.project_bending_pd,
    assemble_type=arguments.assemble_type,
    linear_solver=arguments.linear_solver,
    linear_solver_tolerance=1.0e-8,
    linear_solver_relative_tolerance=arguments.linear_relative_tolerance,
    linear_solver_max_iters=4000,
    output_interval=arguments.output_interval,
    path=arguments.output_dir,
)

print(
    f"Twist cloth: nodes={mesh.number_of_nodes}, triangles={mesh.number_of_cells}, "
    f"grid={grid}x{grid}, steps={arguments.steps}, dt={arguments.dt:g}"
)
peak = {"active_contacts": 0, "pt_candidates": 0, "ee_candidates": 0}


def sample_contact(engine):
    contact = engine.last_step_record.get("contact", {})
    for key in peak:
        peak[key] = max(peak[key], int(contact.get(key, 0)))


result = cloth.run(verbose=True, postprocessing=[sample_contact])
if not np.isfinite(result.positions).all():
    raise RuntimeError("twisting cloth produced non-finite positions")
triangles = np.asarray(mesh.cells, dtype=np.int32)
edges = np.unique(
    np.sort(np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])), axis=1),
    axis=0,
)
initial_edge_length = np.linalg.norm(mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]], axis=1)
final_edge_ratio = (
    np.linalg.norm(result.positions[edges[:, 1]] - result.positions[edges[:, 0]], axis=1) / initial_edge_length
)
middle = np.abs(result.positions[:, 1]) <= max(0.1 * width, 0.5 * width / (grid - 1)) + 1.0e-12
middle_radius = np.linalg.norm(result.positions[middle][:, (0, 2)], axis=1)
summary = {
    "case": "newton_cloth_twist",
    "nodes": int(mesh.number_of_nodes),
    "triangles": int(mesh.number_of_cells),
    "steps": int(arguments.steps),
    "dt": float(arguments.dt),
    "final_time": float(result.time),
    "assemble_type": arguments.assemble_type,
    "linear_solver": arguments.linear_solver,
    "linear_relative_tolerance": float(arguments.linear_relative_tolerance),
    "newton_velocity_tolerance": float(arguments.newton_velocity_tolerance),
    "max_iterations": int(arguments.max_iterations),
    "stretch_stiffness": float(arguments.stretch_stiffness),
    "effective_membrane_stiffness": float(arguments.stretch_stiffness * 2.0e-3),
    "bending_stiffness": float(arguments.bending_stiffness),
    "effective_bending_modulus": float(arguments.bending_stiffness * 2.0e-3**3 / (24.0 * (1.0 - 0.3**2))),
    "friction_coefficient": float(arguments.friction_coefficient),
    "dhat": 2.0e-3,
    "dmin": 1.5e-3,
    "project_membrane_pd": bool(arguments.project_pd),
    "project_bending_pd": bool(arguments.project_bending_pd),
    "project_contact_pd": bool(arguments.contact_project_pd),
    "contact_model": arguments.contact_model,
    "converged": bool(result.converged),
    "finite_positions": True,
    "position_min": np.min(result.positions, axis=0).tolist(),
    "position_max": np.max(result.positions, axis=0).tolist(),
    "edge_stretch_ratio_mean": float(np.mean(final_edge_ratio)),
    "edge_stretch_ratio_p95": float(np.quantile(final_edge_ratio, 0.95)),
    "edge_stretch_ratio_maximum": float(np.max(final_edge_ratio)),
    "middle_radius_p95": float(np.quantile(middle_radius, 0.95)),
    "contact": result.history[-1].get("contact") if result.history else None,
    "maximum_active_contacts": peak["active_contacts"],
    "maximum_pt_candidates": peak["pt_candidates"],
    "maximum_ee_candidates": peak["ee_candidates"],
}
(Path(arguments.output_dir) / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
print(json.dumps(summary, indent=2, default=float))
if not summary["converged"] or not np.isclose(summary["final_time"], arguments.steps * arguments.dt):
    raise RuntimeError("twisting cloth stopped before the requested physical time")
