#!/usr/bin/env python3
"""Confined triaxial compression of a FEM-soft/ABD-stiff particle mixture."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np

from examples.fedem.FEMAffineMixedTriaxial25.fem_affine_mixed_triaxial_25_parameters import (
    CONTACT_DHAT,
    CONTACT_KAPPA,
    FEM_EE_STENCILS_PER_COMPONENT_PAIR,
    FEM_PT_STENCILS_PER_COMPONENT_PAIR,
    LOWER,
    PARTICLE_COUNT,
    PLATE_SPAN,
    SOFT_PERCENT,
    UPPER,
)
from examples.fedem.FEMAffineMixedTriaxial25.draw.evaluate_fem_affine_mixed_triaxial_25 import (
    plate_specs,
    write_results,
)

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
RADIUS = 0.04
INITIAL_HEIGHT = UPPER - LOWER
PLATE_AREA = PLATE_SPAN**2
# Affine LevelSet contact uses a raw squared-distance barrier; FEM IPC uses the
# physical barrier scale dhat / dhat^4.
DEFAULT_ABD_CONTACT_KAPPA = CONTACT_KAPPA / CONTACT_DHAT**3
# For the mixed FEM--ABD interface, the default 3x3x3 lattice has 21
# soft--rigid and 39 rigid--platen neighbours.  Its capacities therefore use
# 64 PT and 128 EE local stencils per possible mixed body pair.  FEM self
# contact has a different topology and is sized below from every unordered
# pair of FEM connected components; both estimates remain user-overridable.
DEFAULT_MIXED_PT_CAPACITY = 4096
DEFAULT_MIXED_EE_CAPACITY = 8192


def fem_contact_pair_capacities(component_count):
    pair_bound = math.comb(int(component_count), 2)
    pt_required = max(FEM_PT_STENCILS_PER_COMPONENT_PAIR * pair_bound, 1)
    ee_required = max(FEM_EE_STENCILS_PER_COMPONENT_PAIR * pair_bound, 1)
    return 1 << (pt_required - 1).bit_length(), 1 << (ee_required - 1).bit_length()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--subdivisions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=5.0e-3)
    parser.add_argument("--output-interval", type=int, default=10)
    parser.add_argument("--compression-speed", type=float, default=0.01)
    parser.add_argument("--friction", type=float, default=0.30)
    parser.add_argument("--fem-damping", type=float, default=0.04)
    parser.add_argument("--linear-solver-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--abd-contact-kappa", type=float, default=DEFAULT_ABD_CONTACT_KAPPA)
    parser.add_argument("--fem-pt-cap", type=int)
    parser.add_argument("--fem-ee-cap", type=int)
    parser.add_argument("--abd-pt-cap", type=int, default=1)
    parser.add_argument("--abd-ee-cap", type=int, default=1)
    parser.add_argument("--mixed-pt-cap", type=int, default=DEFAULT_MIXED_PT_CAPACITY)
    parser.add_argument("--mixed-ee-cap", type=int, default=DEFAULT_MIXED_EE_CAPACITY)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-dir")
    parsed = parser.parse_args()
    for name in ("steps", "dt", "output_interval", "compression_speed"):
        if not math.isfinite(float(getattr(parsed, name))) or getattr(parsed, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if parsed.output_interval > parsed.steps:
        parser.error("--output-interval cannot exceed --steps")
    if not 0.0 <= parsed.friction <= 1.0:
        parser.error("--friction must lie in [0, 1]")
    if not math.isfinite(parsed.fem_damping) or parsed.fem_damping < 0.0:
        parser.error("--fem-damping must be finite and nonnegative")
    if not math.isfinite(parsed.linear_solver_relative_tolerance) or parsed.linear_solver_relative_tolerance <= 0.0:
        parser.error("--linear-solver-relative-tolerance must be finite and positive")
    if not math.isfinite(parsed.abd_contact_kappa) or parsed.abd_contact_kappa <= 0.0:
        parser.error("--abd-contact-kappa must be finite and positive")
    for name in (
        "fem_pt_cap",
        "fem_ee_cap",
        "abd_pt_cap",
        "abd_ee_cap",
        "mixed_pt_cap",
        "mixed_ee_cap",
    ):
        if getattr(parsed, name) is not None and getattr(parsed, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if parsed.output_dir is None:
        parsed.output_dir = str(CASE_DIR / "OutputData")
    return parsed


def particle_centers():
    axis = np.asarray((0.318, 0.400, 0.482), dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def composition():
    soft_count = int(math.floor(PARTICLE_COUNT * SOFT_PERCENT / 100.0 + 0.5))
    order = np.random.default_rng(20260907).permutation(PARTICLE_COUNT)
    soft_ids = np.sort(order[:soft_count])
    rigid_ids = np.sort(order[soft_count:])
    return soft_ids, rigid_ids


def fem_sphere(center, subdivisions, name):
    import trimesh
    from src.fem.generator import FEMMesh

    surface = trimesh.creation.icosphere(subdivisions=subdivisions, radius=RADIUS)
    vertices = np.asarray(surface.vertices, dtype=np.float64) + np.asarray(center)[None, :]
    center_id = vertices.shape[0]
    points = np.vstack((vertices, np.asarray(center, dtype=np.float64)))
    cells = np.column_stack(
        (
            np.full(surface.faces.shape[0], center_id, dtype=np.int32),
            np.asarray(surface.faces, dtype=np.int32),
        )
    )
    return FEMMesh(points, cells, "TET4", name=name)


def build_fem(gt, soft_ids, centers, parsed):
    from src.fem import DirichletBoundary
    from src.fem.generator import FEMMesh

    fem = gt.FEM(log=True)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    meshes = [fem_sphere(centers[index], parsed.subdivisions, f"soft_{index:02d}") for index in soft_ids]
    ranges = []
    node_offset = sum(mesh.number_of_nodes for mesh in meshes)
    for name, origin, size in plate_specs():
        plate = fem.create_mesh(
            "box",
            origin=origin,
            size=size,
            divisions=(4, 4, 1) if size[2] == 0.020 else (1, 4, 4),
            element_type="TET4",
        )
        if size[1] == 0.020:
            plate = fem.create_mesh("box", origin=origin, size=size, divisions=(4, 1, 4), element_type="TET4")
        ranges.append((name, np.arange(node_offset, node_offset + plate.number_of_nodes, dtype=np.int32)))
        node_offset += plate.number_of_nodes
        meshes.append(plate)
    mesh = fem.add_soft_particle(FEMMesh.concatenate(meshes, name="soft_grains_and_platens"))
    fem.add_material(
        "NeoHookean",
        density=2500.0,
        young_modulus=2.0e4,
        poisson_ratio=0.30,
    )
    boundary = DirichletBoundary()
    for name, nodes in ranges:
        if name != "top":
            boundary.add(nodes, "all", 0.0)
    top_nodes = dict(ranges)["top"]
    boundary.add(top_nodes, "xy", 0.0)
    boundary.add(
        top_nodes,
        "z",
        lambda time, _coordinates: -parsed.compression_speed * time,
    )
    fem.add_boundary_condition(dirichlet=boundary)
    fem.add_contact(
        "IPC",
        broad_phase="BVH",
        dhat=CONTACT_DHAT,
        dmin=0.0,
        kappa=CONTACT_KAPPA,
        friction_coefficient=parsed.friction,
        epsv=1.0e-3,
        friction_iterations=2,
        max_point_triangle_pairs=parsed.fem_pt_cap,
        max_edge_edge_pairs=parsed.fem_ee_cap,
    )
    return fem, mesh, dict(ranges)


def affine_surface(output, subdivisions):
    import trimesh

    path = output / "mesh" / f"abd_icosphere_subdivisions_{subdivisions}.obj"
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0).export(path)
    return path


def build_dem(gt, rigid_ids, centers, parsed, surface_path):
    surface = gt.polyhedron(file=str(surface_path)).grids(space=0.1, extent=3)
    surface.generate()
    dem = gt.DEM(log=True)
    dem.set_configuration(
        domain=[0.8, 0.8, 0.8],
        scheme="AffineBody",
        search="BVH",
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        track_energy=False,
        log=True,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=1.0e8,
        local_damping=0.02,
        contact_damping_stiffness=0.0,
        hessian_shift=0.0,
        friction_mode="lagged",
        friction_iterations=1,
    )
    rigid_count = rigid_ids.size
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": int(rigid_count),
            "surface_node_number": int(rigid_count * (10 * 4**parsed.subdivisions + 2)),
            "max_point_triangle_pairs": parsed.abd_pt_cap,
            "max_edge_edge_pairs": parsed.abd_ee_cap,
            "body_coordination_number": max(int(rigid_count) - 1, 1),
            "wall_coordination_number": 1,
            "compaction_ratio": [1.0, 1.0],
        },
        log=True,
    )
    dem.add_attribute(materialID=0, attribute={"Density": 2500.0})
    dem.add_template(
        {
            "Name": "stiff_sphere",
            "TemplateType": "AffineBody",
            "Object": surface,
            "ContactRepresentation": "LevelSet",
        }
    )
    rng = np.random.default_rng(20260907)
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": [
                {
                    "Name": "stiff_sphere",
                    "GroupID": 1,
                    "MaterialID": 0,
                    "BodyPoint": centers[index].tolist(),
                    "BoundingRadius": RADIUS,
                    "BodyOrientation": rng.uniform(0.0, 360.0, 3).tolist(),
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "YoungModulus": 1.0e8,
                    "Friction": parsed.friction,
                }
                for index in rigid_ids
            ],
        }
    )
    dem.add_property(
        0,
        0,
        {
            "Dhat": CONTACT_DHAT,
            "BarrierStiffness": parsed.abd_contact_kappa,
            "ContactDampingStiffness": 0.0,
            "Friction": parsed.friction,
        },
        dType="all",
    )
    dem.select_save_data(particle=False, surface=True)
    return dem


def main():
    parsed = arguments()
    output = Path(parsed.output_dir).expanduser().resolve()
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import geotaichi as gt

    gt.init(arch=parsed.arch, default_fp="float64", log=True, debug=parsed.debug, offline_cache=True)
    centers = particle_centers()
    soft_ids, rigid_ids = composition()
    fem_component_count = int(soft_ids.size) + len(plate_specs())
    default_fem_pt_cap, default_fem_ee_cap = fem_contact_pair_capacities(fem_component_count)
    parsed.fem_pt_cap = parsed.fem_pt_cap or default_fem_pt_cap
    parsed.fem_ee_cap = parsed.fem_ee_cap or default_fem_ee_cap
    fem, _mesh, plate_nodes = build_fem(gt, soft_ids, centers, parsed)
    dem = build_dem(gt, rigid_ids, centers, parsed, affine_surface(output, parsed.subdivisions))

    coupling = gt.FEDEM(dem=dem, fem=fem, log=True)
    coupling.set_configuration(domain=[0.8, 0.8, 0.8], search="BVH", gravity=[0.0, 0.0, 0.0], log=True)
    coupling.set_solver(
        {
            "Timestep": parsed.dt,
            "SimulationTime": parsed.steps * parsed.dt,
            "SaveInterval": parsed.output_interval * parsed.dt,
            "SavePath": str(output),
            # Free grains have rigid-body modes before first contact, so the
            # inertial term is the physically appropriate regularization.
            "quasi_static": False,
            "assemble_type": "HashTriplet",
            "linear_solver": "PCG",
            "residual_tolerance": 1.0e-7,
            "absolute_tolerance": 1.0e-10,
            "correction_velocity_tolerance": 5.0e-3,
            "max_iterations": 100,
            "linear_solver_tolerance": 1.0e-8,
            "linear_solver_relative_tolerance": parsed.linear_solver_relative_tolerance,
            "project_pd": True,
            "damping": parsed.fem_damping,
            "enable_step_retry": True,
            "step_retry_max_retries": 3,
            "step_retry_reduction": 0.5,
            "step_retry_minimum_timestep": parsed.dt / 8.0,
        },
        log=True,
    )
    coupling.add_surface()
    coupling.memory_allocate(
        {
            "max_contact_pairs": 8192,
            "max_point_triangle_pairs": parsed.mixed_pt_cap,
            "max_edge_edge_pairs": parsed.mixed_ee_cap,
            "contact_coordination_number": 64,
        }
    )
    coupling.choose_contact_model(
        "IPC",
        dhat=CONTACT_DHAT,
        dmin=0.0,
        kappa=CONTACT_KAPPA,
        friction_coefficient=parsed.friction,
        epsv=1.0e-3,
        friction_mode="lagged",
        friction_iterations=2,
    )
    coupling.select_save_data(contact=False, checkpoint=False)
    coupling.add_essentials()
    initial_velocity = coupling.enginer.fem.state.velocity.to_numpy()
    initial_velocity[plate_nodes["top"], 2] = -parsed.compression_speed
    coupling.enginer.fem.state.velocity.from_numpy(np.ascontiguousarray(initial_velocity))

    curve = []

    def sample(engine):
        reaction = engine.fem.state.reaction.to_numpy()
        force = {name: np.sum(reaction[nodes], axis=0) for name, nodes in plate_nodes.items()}
        sigma_x = 0.5 * (abs(force["left"][0]) + abs(force["right"][0])) / PLATE_AREA
        sigma_y = 0.5 * (abs(force["front"][1]) + abs(force["back"][1])) / PLATE_AREA
        sigma_z = 0.5 * (abs(force["bottom"][2]) + abs(force["top"][2])) / PLATE_AREA
        lateral = 0.5 * (sigma_x + sigma_y)
        mixed = engine.contact.diagnostics()
        internal = engine.fem_contact.device_diagnostics()
        curve.append(
            {
                "step": engine.step_count,
                "time": engine.time,
                "axial_strain": parsed.compression_speed * engine.time / INITIAL_HEIGHT,
                "volumetric_strain": parsed.compression_speed * engine.time / INITIAL_HEIGHT,
                "stress_x": sigma_x,
                "stress_y": sigma_y,
                "axial_stress": sigma_z,
                "mean_stress": (sigma_x + sigma_y + sigma_z) / 3.0,
                "deviator_stress": sigma_z - lateral,
                "top_reaction": abs(force["top"][2]),
                "bottom_reaction": abs(force["bottom"][2]),
                "force_imbalance": abs(abs(force["top"][2]) - abs(force["bottom"][2]))
                / max(abs(force["top"][2]), abs(force["bottom"][2]), 1.0e-30),
                "minimum_jacobian": engine.minimum_jacobian,
                "fem_active_contacts": internal["active_contacts"],
                "abd_active_contacts": int(engine.affine.levelset_active_contacts[None]),
                "abd_minimum_gap": float(engine.affine.levelset_minimum_gap[None]),
                "mixed_active_contacts": mixed["active_contacts"],
                "mixed_minimum_distance": mixed["minimum_distance"],
            }
        )

    sample(coupling.enginer)
    result = coupling.run(postprocessing=(sample,), verbose=parsed.verbose)
    if not result["converged"]:
        raise RuntimeError("FEM--ABD triaxial compression did not converge")
    write_results(output, parsed, soft_ids, rigid_ids, curve, result)
    print(f"FEM--ABD triaxial compression finished: {output}")


if __name__ == "__main__":
    main()
