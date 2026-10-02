"""Implicit counterpart of the LSDEM soft-particle collision example.

The same TET4 soft particle and affine sphere are coupled monolithically by
frictional IPC.  Contact culling, CCD, line search, residuals, and assembly
remain device-side; the selected sparse solver consumes a HashTriplet matrix.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

default_output = Path(__file__).resolve().parent / "OutputData" / "implicit_affine_ipc_soft_particle"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument("--assemble-type", default="HashTriplet")
parser.add_argument("--linear-solver", default="PCG")
parser.add_argument("--search", default="BVH")
parser.add_argument("--impact-speed", type=float, default=0.20)
parser.add_argument("--dt", type=float, default=1.0e-3)
parser.add_argument("--time", type=float, default=1.0e-2)
parser.add_argument("--save-interval", type=float, default=1.0e-3)
parser.add_argument("--newton-tolerance", type=float, default=1.0e-7)
parser.add_argument("--newton-absolute-tolerance", type=float, default=1.0e-10)
parser.add_argument("--newton-velocity-tolerance", type=float, default=1.0e-2)
parser.add_argument("--max-newton-iterations", type=int, default=200)
parser.add_argument("--linear-solver-absolute-tolerance", type=float, default=1.0e-8)
parser.add_argument("--linear-solver-relative-tolerance", type=float, default=1.0e-8)
parser.add_argument(
    "--mesh-path",
    default=str(ROOT / "assets" / "mesh" / "AffineBody" / "icosphere.obj"),
)
parser.add_argument("--fem-divisions", type=int, default=40)
parser.add_argument("--affine-young-modulus", type=float, default=1.0e8)
parser.add_argument("--output-dir", default=str(default_output))
parser.add_argument("--scene-manifest", help="Optional Blender SceneManifest metadata")
arguments = parser.parse_args()

if arguments.fem_divisions <= 0:
    parser.error("--fem-divisions must be positive")
if not np.isfinite(arguments.newton_tolerance) or arguments.newton_tolerance <= 0.0:
    parser.error("--newton-tolerance must be positive")
if not np.isfinite(arguments.newton_absolute_tolerance) or arguments.newton_absolute_tolerance < 0.0:
    parser.error("--newton-absolute-tolerance must be non-negative")
if not np.isfinite(arguments.newton_velocity_tolerance) or arguments.newton_velocity_tolerance <= 0.0:
    parser.error("--newton-velocity-tolerance must be positive")
if arguments.max_newton_iterations <= 0:
    parser.error("--max-newton-iterations must be positive")
if not np.isfinite(arguments.affine_young_modulus) or arguments.affine_young_modulus <= 0.0:
    parser.error("--affine-young-modulus must be positive")
if not np.isfinite(arguments.linear_solver_absolute_tolerance) or arguments.linear_solver_absolute_tolerance < 0.0:
    parser.error("--linear-solver-absolute-tolerance must be non-negative")
if not np.isfinite(arguments.linear_solver_relative_tolerance) or arguments.linear_solver_relative_tolerance < 0.0:
    parser.error("--linear-solver-relative-tolerance must be non-negative")

# Keep AffineBody fields and the f64 FEM/IPC system on one scalar type.
os.environ["GEOTAICHI_REAL_DTYPE"] = arguments.default_fp

import geotaichi as gt

gt.init(
    arch=arguments.arch,
    default_fp=arguments.default_fp,
    log=True,
    debug=False,
    offline_cache=False,
)

dem = gt.DEM(log=True)
dem.set_configuration(
    domain=[1.0, 1.0, 1.2],
    scheme="AffineBody",
    search="BVH",
    gravity=[0.0, 0.0, 0.0],
    visualize=False,
    log=True,
)
dem.set_affine_body_parameters(
    assemble_type=arguments.assemble_type,
    # ABD examples use E=1e8--1e9 for visually rigid bodies.  The
    # orthogonality potential deliberately permits a tiny affine strain; E=2e4
    # made this sphere only twice as stiff as the FEM particle and visibly soft.
    young_modulus=arguments.affine_young_modulus,
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
        "surface_node_number": 256,
        # There is one affine body, hence no Affine--Affine PT/EE contact.
        "max_point_triangle_pairs": 0,
        "max_edge_edge_pairs": 0,
        "body_coordination_number": 16,
        "wall_coordination_number": 1,
        "compaction_ratio": [1.0, 1.0],
    },
    log=True,
)
dem.add_attribute(materialID=0, attribute={"Density": 1000.0})
mesh_path = Path(arguments.mesh_path).expanduser().resolve()
dem.add_template(
    {
        "Name": "affine_sphere",
        "TemplateType": "AffineBody",
        "Object": gt.polyhedron(file=str(mesh_path)),
    }
)
dem.create_body(
    {
        "BodyType": "AffineBody",
        "Template": [
            {
                "Name": "affine_sphere",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.5, 0.5, 0.30],
                "ScaleFactor": 0.11,
                "InitialVelocity": [0.0, 0.0, 0.0],
            }
        ],
    }
)

fem = gt.FEM(log=True)
fem.set_configuration(dimension=3, solver_type="Implicit")
soft = fem.add_soft_particle(
    fem.create_mesh(
        "box",
        origin=[0.32, 0.32, 0.485],
        size=[0.36, 0.36, 0.30],
        divisions=[arguments.fem_divisions] * 3,
        element_type="TET4",
    )
)
fem.add_material(
    "NeoHookean",
    density=1100.0,
    young_modulus=1.0e4,
    poisson_ratio=0.3,
)
fem.add_boundary_condition(
    {
        "type": "Dirichlet",
        "nodes": soft.node_sets["zmax"],
        "components": "all",
        "value": 0.0,
    }
)
initial_velocity = np.zeros_like(soft.points)
initial_velocity[:, 2] = -arguments.impact_speed

coupling = gt.FEDEM(dem=dem, fem=fem, log=True)
coupling.set_configuration(
    domain=[1.0, 1.0, 1.2],
    search=arguments.search,
    log=True,
)
coupling.set_solver(
    {
        "Timestep": arguments.dt,
        "SimulationTime": arguments.time,
        "SaveInterval": arguments.save_interval,
        "SavePath": arguments.output_dir,
        "initial_velocity": initial_velocity,
        "assemble_type": arguments.assemble_type,
        "linear_solver": arguments.linear_solver,
        "residual_tolerance": arguments.newton_tolerance,
        "absolute_tolerance": arguments.newton_absolute_tolerance,
        "max_iterations": arguments.max_newton_iterations,
        # ABD Newton convergence is measured on the physical surface/FEM
        # correction divided by dt, rather than on a scale-dependent force
        # residual alone.
        "correction_velocity_tolerance": arguments.newton_velocity_tolerance,
        "linear_solver_tolerance": arguments.linear_solver_absolute_tolerance,
        "linear_solver_relative_tolerance": arguments.linear_solver_relative_tolerance,
        "linear_solver_max_iters": 2000,
        "project_pd": True,
    },
    log=True,
)
coupling.add_surface()
coupling.memory_allocate(
    {
        "max_contact_pairs": 4096,
        "max_point_triangle_pairs": 4096,
        "max_edge_edge_pairs": 4096,
        "contact_coordination_number": 64,
    }
)
coupling.choose_contact_model(
    "IPC",
    dhat=0.01,
    dmin=0.0,
    kappa=2.0e4,
    friction_coefficient=0.3,
    epsv=1.0e-3,
    friction_mode="lagged",
    friction_iterations=2,
)
coupling.add_ipc_property(
    AffineBody=0,
    FEMbody=0,
    property={
        "dhat": 0.01,
        "kappa": 4.0e4,
        "friction_coefficient": 0.4,
        "epsv": 1.0e-3,
    },
)
coupling.run()
