"""Two-layer cloth pendulum colliding with a fixed triangulated sphere.

Each cloth layer is a separate TRI3 FEM body.  The two corners on its upper
hinge edge are fixed, while all remaining cloth nodes swing under gravity.
The sphere is a third TRI3 body whose nodes are all fixed, so cloth--sphere
contact is solved by FEM IPC without introducing a moving rigid-body solver.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument("--divisions", type=positive_integer, default=24)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--dt", type=float, default=2.0e-3)
parser.add_argument("--newton-velocity-tolerance", type=float, default=1.0e-3)
parser.add_argument("--output-interval", type=positive_integer, default=10)
parser.add_argument("--assemble-type", default="HashTriplet")
parser.add_argument("--linear-solver", default="PCG")
parser.add_argument(
    "--sphere-mesh",
    default=str(REPO_ROOT / "assets" / "mesh" / "AffineBody" / "icosphere.obj"),
)
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData" / "implicit_two_layer_cloth_sphere"),
)
arguments = parser.parse_args()

if arguments.steps < 0:
    parser.error("--steps must be non-negative")
if not np.isfinite(arguments.dt) or arguments.dt <= 0.0:
    parser.error("--dt must be finite and positive")
if not np.isfinite(arguments.newton_velocity_tolerance) or arguments.newton_velocity_tolerance <= 0.0:
    parser.error("--newton-velocity-tolerance must be finite and positive")


import geotaichi as gt
from src.fem import DirichletBoundary, FEMMesh


def make_cloth_layer(fem, normal_offset, name):
    """Create one initially inclined cloth layer and its two hinge nodes."""
    width = 0.80
    length = 0.80
    anchor_x = 0.10
    anchor_y = -0.15
    anchor_z = 1.35
    inclination = np.deg2rad(15.0)

    layer = fem.create_mesh(
        "rectangle",
        size=(width, length),
        divisions=(arguments.divisions, arguments.divisions),
        plane="xy",
        origin=(anchor_x, anchor_y, 0.0),
        name=name,
    )
    material_v = layer.points[:, 1] - anchor_y
    tangent = np.array(
        (0.0, np.cos(inclination), -np.sin(inclination)),
        dtype=np.float64,
    )
    normal = np.array(
        (0.0, np.sin(inclination), np.cos(inclination)),
        dtype=np.float64,
    )
    positions = layer.points.copy()
    positions[:, 1] = anchor_y + material_v * tangent[1]
    positions[:, 2] = anchor_z + material_v * tangent[2]
    positions += float(normal_offset) * normal
    layer.points[:] = positions
    layer.set_rest_shape(positions, update_material_coordinates=True)

    tolerance = 1.0e-10
    hinge_y = anchor_y + float(normal_offset) * normal[1]
    hinge = layer.select_nodes(
        selector=lambda points: np.isclose(points[:, 1], hinge_y, atol=tolerance)
        & (
            np.isclose(points[:, 0], anchor_x, atol=tolerance)
            | np.isclose(points[:, 0], anchor_x + width, atol=tolerance)
        )
    )
    if hinge.size != 2:
        raise RuntimeError(f"{name} must have exactly two hinge corners; found {hinge.size}")
    layer.node_sets["hinge_corners"] = hinge
    return layer


def make_fixed_sphere(fem, filename):
    """Load and place the fixed triangular obstacle."""
    sphere = fem.generator.read(
        Path(filename).expanduser().resolve(),
        cell_type="TRI3",
        name="fixed_sphere",
    )
    center = np.array((0.50, 0.40, 0.72), dtype=np.float64)
    radius = 0.22
    local = sphere.points - np.mean(sphere.points, axis=0)
    source_radius = float(np.max(np.linalg.norm(local, axis=1)))
    if not np.isfinite(source_radius) or source_radius <= 0.0:
        raise ValueError("sphere mesh must have a finite positive radius")
    positions = center + radius * local / source_radius
    sphere.points[:] = positions
    sphere.set_rest_shape(positions, update_material_coordinates=True)
    sphere.node_sets["fixed"] = np.arange(sphere.number_of_nodes, dtype=np.int32)
    return sphere


gt.init(
    arch=arguments.arch,
    default_fp=arguments.default_fp,
    log=True,
)

fem = gt.FEM(title="Two-layer cloth pendulum over a fixed sphere", log=True)
fem.set_configuration(dimension=3, solver_type="Implicit")

# Offset along the cloth normal, rather than only in z, gives the two layers a
# uniform initial separation everywhere, including at the two hinge edges.
layer_gap = 1.8e-2
lower = make_cloth_layer(fem, -0.5 * layer_gap, "lower_cloth")
upper = make_cloth_layer(fem, 0.5 * layer_gap, "upper_cloth")
sphere = make_fixed_sphere(fem, arguments.sphere_mesh)

mesh = FEMMesh.concatenate(
    (lower, upper, sphere),
    name="two_layer_cloth_and_fixed_sphere",
)
fem.add_mesh(mesh)
fem.add_material(
    "ClothARAP",
    stretch_stiffness=5.0e4,
    compression_stiffness=8.0e4,
    density=1000.0,
    thickness=2.0e-3,
    bending_stiffness=2.0e8,
    bending_poisson_ratio=0.3,
)

lower_offset = 0
upper_offset = lower.number_of_nodes
sphere_offset = lower.number_of_nodes + upper.number_of_nodes
fixed_nodes = np.concatenate(
    (
        lower.node_sets["hinge_corners"] + lower_offset,
        upper.node_sets["hinge_corners"] + upper_offset,
        np.arange(sphere.number_of_nodes, dtype=np.int32) + sphere_offset,
    )
)
fem.add_boundary_condition(dirichlet=DirichletBoundary().add(fixed_nodes, "all", 0.0))

fem.add_contact(
    "IPC",
    broad_phase="BVH",
    # One shared assembler handles layer--layer, cloth--sphere and genuine
    # single-layer folds.  Using three identical pair-specific assemblers
    # would compile the same PT/EE kernels three times without changing the
    # contact law in this scene.
    self_contact=True,
    dhat=1.2e-2,
    dmin=2.0e-3,
    kappa=5.0e4,
    friction_coefficient=0.20,
    epsv=1.0e-3,
    ccd_safety=0.9,
    project_pd=True,
)

fem.set_solver(
    quasi_static=False,
    dt=arguments.dt,
    step=arguments.steps,
    gravity=(0.0, 0.0, -9.81),
    damping=0.05,
    max_iterations=100,
    residual_tolerance=1.0e-7,
    absolute_tolerance=1.0e-10,
    correction_velocity_tolerance=arguments.newton_velocity_tolerance,
    line_search=True,
    project_pd=True,
    assemble_type=arguments.assemble_type,
    linear_solver=arguments.linear_solver,
    linear_solver_tolerance=1.0e-8,
    linear_solver_max_iters=4000,
    output_interval=arguments.output_interval,
    path=arguments.output_dir,
)

print(
    "Scene bodies: lower cloth=0, upper cloth=1, fixed sphere=2; "
    f"nodes={mesh.number_of_nodes}, triangles={mesh.number_of_cells}, "
    f"fixed nodes={fixed_nodes.size}"
)
result = fem.run(verbose=True)
if result.history:
    print("Final contact diagnostics:", result.history[-1].get("contact"))
