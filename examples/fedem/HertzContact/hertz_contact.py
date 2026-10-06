#!/usr/bin/env python3
"""Explicit FEM soft-sphere/rigid-LSDEM-plane Hertz verification."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.fedem.HertzContact.hertz_contact_parameters import (
    POISSON,
    PRESSURE_ANNULUS_COUNT,
    PRESSURE_TRIANGLE_SUBDIVISIONS,
    RADIUS,
    TARGET_INDENTATION,
    TARGET_LOAD,
    YOUNG,
)
from examples.fedem.HertzContact.draw.evaluate_hertz_contact import (
    _equilibrium_window,
    _hertz_contact_radius,
    _pressure_profile,
    _write_pressure_sample_archive,
)


from examples.fedem.HertzContact.draw.output_audit import collect_native_output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "OutputData/hertz_contact"
DEFAULT_SURFACE = REPO_ROOT / "assets/mesh/AffineBody/icosphere.obj"
DEFAULT_WALL = REPO_ROOT / "assets/mesh/AffineBody/cube.obj"
PROMPT = (
    "Build a three-dimensional Hertz test with a deformable FEM sphere "
    "pressed against a fixed, frictionless level-set DEM plane. Refine the "
    "tetrahedral mesh near contact and compare the pressure and "
    "force--contact-radius responses with the analytical solution."
)

DENSITY = 1200.0
INITIAL_GAP = 1.0e-4 * RADIUS
DT = 1.0e-5
RAMP_TIME = 0.15
HOLD_TIME = 0.05
MAXIMUM_HOLD_TIME = 0.45
EQUILIBRIUM_WINDOW = 0.05
OUTPUT_INTERVAL = 0.05
REACTION_ERROR_TOLERANCE = 0.02
REACTION_CV_TOLERANCE = 0.02
KINETIC_STRAIN_TOLERANCE = 0.01
CONTACT_RADIUS_DRIFT_TOLERANCE = 0.01
CONTACT_RADIUS_ERROR_TOLERANCE = 0.05
PRESSURE_DIAMETER_POINT_COUNT = 2 * PRESSURE_ANNULUS_COUNT
NORMAL_STIFFNESS = 2.0e9
BARRIER_CUTOFF = INITIAL_GAP
TANGENTIAL_STIFFNESS = 1.0e8
MINIMUM_TETRA_MEAN_RATIO = 0.05
RELAXATION_DAMPING_RATIO = 0.70
SPHERE_MASS = (4.0 / 3.0) * math.pi * RADIUS**3 * DENSITY
HERTZ_TANGENT_STIFFNESS = 2.0 * (YOUNG / (1.0 - POISSON * POISSON)) * math.sqrt(RADIUS * TARGET_INDENTATION)
HERTZ_RELAXATION_FREQUENCY = math.sqrt(HERTZ_TANGENT_STIFFNESS / SPHERE_MASS)
FEM_DAMPING = 2.0 * RELAXATION_DAMPING_RATIO * HERTZ_RELAXATION_FREQUENCY
WALL_TOP = 0.40
CENTER = np.array([0.30, 0.30, WALL_TOP + RADIUS + INITIAL_GAP])


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _source_fingerprint() -> tuple[str, list[str]]:
    """Identify the solver/diagnostic implementation used by one mesh level."""
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "src/fedem/Engine.py",
        REPO_ROOT / "src/fedem/Patch.py",
        REPO_ROOT / "src/fedem/contact/ContactKernel.py",
    )
    digest = hashlib.sha256()
    relative_paths = []
    for path in paths:
        relative = str(path.relative_to(REPO_ROOT))
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f" and len(fields) >= 4:
            polygon = [int(token.split("/")[0]) - 1 for token in fields[1:]]
            for local in range(1, len(polygon) - 1):
                faces.append([polygon[0], polygon[local], polygon[local + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _locally_refined_sphere_mesh(
    surface: Path,
    radius: float,
    center: np.ndarray,
    coarse_spacing_over_radius: float,
    contact_spacing_over_radius: float,
    surface_spacing_over_radius: float,
):
    from src.fem.generator import FEMMesh
    from scipy.spatial import Delaunay

    vertices, faces = _read_obj(surface)
    vertices -= np.mean(vertices, axis=0)
    vertices /= np.linalg.norm(vertices, axis=1).mean()
    surface_points = [radius * vertices]

    # Refine the lower spherical cap independently of the interior spacing;
    # the transition reaches the 0.25R contact-zone half-width.
    cap_points = [[0.0, 0.0, -radius]]
    boundary_spacing = surface_spacing_over_radius
    if math.isclose(boundary_spacing, 0.10, rel_tol=0.0, abs_tol=1.0e-12):
        polar_values = np.linspace(0.08, 0.36, 5)
    else:
        ring_count_polar = max(4, int(math.ceil(0.36 / boundary_spacing)))
        polar_values = np.linspace(boundary_spacing, 0.36, ring_count_polar)
    for polar in polar_values:
        ring_radius = radius * math.sin(float(polar))
        ring_count = max(
            8,
            int(math.ceil(2.0 * math.pi * ring_radius / (boundary_spacing * radius))),
        )
        for azimuth in np.linspace(0.0, 2.0 * math.pi, ring_count, endpoint=False):
            cap_points.append(
                [ring_radius * math.cos(azimuth), ring_radius * math.sin(azimuth), -radius * math.cos(float(polar))]
            )
    surface_points.append(np.asarray(cap_points, dtype=np.float64))

    coarse_spacing = coarse_spacing_over_radius * radius
    fine_spacing = contact_spacing_over_radius * radius
    coarse_axis = np.arange(-radius + coarse_spacing, radius, coarse_spacing)
    coarse = np.stack(np.meshgrid(coarse_axis, coarse_axis, coarse_axis, indexing="ij"), axis=-1).reshape(-1, 3)
    coarse = coarse[np.linalg.norm(coarse, axis=1) < 0.92 * radius]
    fine_xy = np.arange(-0.25 * radius, 0.25 * radius + 0.5 * fine_spacing, fine_spacing)
    fine_z = np.arange(-radius + fine_spacing, -0.75 * radius + 0.5 * fine_spacing, fine_spacing)
    fine = np.stack(np.meshgrid(fine_xy, fine_xy, fine_z, indexing="ij"), axis=-1).reshape(-1, 3)
    fine = fine[np.linalg.norm(fine, axis=1) < 0.995 * radius]
    local_points = np.vstack((*surface_points, coarse, fine, np.zeros((1, 3))))
    rounded = np.round(local_points / (1.0e-10 * radius)).astype(np.int64)
    _, unique = np.unique(rounded, axis=0, return_index=True)
    local_points = local_points[np.sort(unique)]
    cells = Delaunay(local_points, qhull_options="QJ Qt").simplices.astype(np.int32)
    tetra = local_points[cells]
    determinants = np.linalg.det(
        np.stack((tetra[:, 1] - tetra[:, 0], tetra[:, 2] - tetra[:, 0], tetra[:, 3] - tetra[:, 0]), axis=2)
    )
    cells = cells[np.abs(determinants) > 2.0e-13]
    used = np.unique(cells)
    inverse = np.full(local_points.shape[0], -1, dtype=np.int32)
    inverse[used] = np.arange(used.size, dtype=np.int32)
    cells = inverse[cells]
    points = local_points[used] + center[None, :]
    return FEMMesh(points, cells, "TET4", name="locally_refined_soft_sphere")


def _tetra_quality(points: np.ndarray, cells: np.ndarray) -> dict[str, float]:
    tetra = points[cells]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_square_sum = sum(np.sum((tetra[:, first] - tetra[:, second]) ** 2, axis=1) for first, second in edge_pairs)
    determinants = np.linalg.det(
        np.stack(
            (
                tetra[:, 1] - tetra[:, 0],
                tetra[:, 2] - tetra[:, 0],
                tetra[:, 3] - tetra[:, 0],
            ),
            axis=2,
        )
    )
    volumes = np.abs(determinants) / 6.0
    mean_ratio = 12.0 * np.power(3.0 * volumes, 2.0 / 3.0) / np.maximum(edge_square_sum, 1.0e-30)
    return {
        "minimum_tetra_mean_ratio": float(np.min(mean_ratio)),
        "mean_tetra_mean_ratio": float(np.mean(mean_ratio)),
        "minimum_tetra_volume": float(np.min(volumes)),
        "maximum_tetra_volume": float(np.max(volumes)),
    }


def _gmsh_locally_refined_sphere_mesh(
    radius: float,
    center: np.ndarray,
    coarse_spacing_over_radius: float,
    contact_spacing_over_radius: float,
    surface_spacing_over_radius: float,
):
    """Generate a quality-controlled sphere with a refined lower contact cap."""

    import gmsh
    from src.fem.generator import FEMMesh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("hertz_locally_refined_sphere")
        gmsh.model.occ.addSphere(*center.tolist(), radius)
        gmsh.model.occ.synchronize()

        coarse = coarse_spacing_over_radius * radius
        contact = contact_spacing_over_radius * radius
        surface = surface_spacing_over_radius * radius
        bottom = center[2] - radius

        contact_box = gmsh.model.mesh.field.add("Box")
        for name, value in (
            ("VIn", contact),
            ("VOut", coarse),
            ("XMin", center[0] - 0.25 * radius),
            ("XMax", center[0] + 0.25 * radius),
            ("YMin", center[1] - 0.25 * radius),
            ("YMax", center[1] + 0.25 * radius),
            ("ZMin", bottom - 0.02 * radius),
            ("ZMax", bottom + 0.30 * radius),
            ("Thickness", 0.10 * radius),
        ):
            gmsh.model.mesh.field.setNumber(contact_box, name, value)

        contact_cap = gmsh.model.mesh.field.add("Ball")
        for name, value in (
            ("VIn", surface),
            # This field must become inactive away from the lower cap.  Using
            # the contact-zone size as VOut and then taking Min with the box
            # field would refine the *entire* sphere to the contact size.
            ("VOut", coarse),
            ("Radius", 0.20 * radius),
            ("Thickness", 0.10 * radius),
            ("XCenter", center[0]),
            ("YCenter", center[1]),
            ("ZCenter", bottom),
        ):
            gmsh.model.mesh.field.setNumber(contact_cap, name, value)

        combined = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(combined, "FieldsList", [contact_box, contact_cap])
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)
        gmsh.option.setNumber("Mesh.MeshSizeMin", surface)
        gmsh.option.setNumber("Mesh.MeshSizeMax", coarse)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        tag_to_local = {int(tag): index for index, tag in enumerate(np.asarray(node_tags))}
        element_types, _, element_nodes = gmsh.model.mesh.getElements(dim=3)
        blocks = []
        for element_type, flattened in zip(element_types, element_nodes):
            properties = gmsh.model.mesh.getElementProperties(element_type)
            node_count = int(properties[3])
            primary_count = int(properties[5])
            if primary_count != 4:
                continue
            tags = np.asarray(flattened, dtype=np.int64).reshape(-1, node_count)[:, :4]
            blocks.append(
                np.asarray(
                    [[tag_to_local[int(tag)] for tag in row] for row in tags],
                    dtype=np.int32,
                )
            )
        if not blocks:
            raise RuntimeError("Gmsh generated no first-order tetrahedra")
        cells = np.vstack(blocks)
        tetra = points[cells]
        determinants = np.linalg.det(
            np.stack(
                (
                    tetra[:, 1] - tetra[:, 0],
                    tetra[:, 2] - tetra[:, 0],
                    tetra[:, 3] - tetra[:, 0],
                ),
                axis=2,
            )
        )
        negative = determinants < 0.0
        cells[negative, 1], cells[negative, 2] = (
            cells[negative, 2].copy(),
            cells[negative, 1].copy(),
        )
        quality = _tetra_quality(points, cells)
        return (
            FEMMesh(points, cells, "TET4", name="gmsh_refined_soft_sphere"),
            quality,
        )
    finally:
        gmsh.finalize()


def _smooth_ramp(time_value: float) -> float:
    fraction = min(max(float(time_value) / RAMP_TIME, 0.0), 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _prompt() -> str:
    return PROMPT


def _hertz_indentation(force: float) -> float:
    effective_modulus = YOUNG / (1.0 - POISSON * POISSON)
    return float((3.0 * max(force, 0.0) / (4.0 * effective_modulus * math.sqrt(RADIUS))) ** (2.0 / 3.0))


def _surface_edges(faces: np.ndarray) -> np.ndarray:
    """Return the unique edges of the fixed FEM surface topology."""
    edges = np.vstack((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def _projected_nodal_areas(
    positions: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Lump current surface-triangle area projected onto the contact plane."""
    projected = positions[faces, :2]
    edge_a = projected[:, 1] - projected[:, 0]
    edge_b = projected[:, 2] - projected[:, 0]
    triangle_area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    nodal_area = np.zeros(positions.shape[0], dtype=np.float64)
    for local_node in range(3):
        np.add.at(nodal_area, faces[:, local_node], triangle_area / 3.0)
    return nodal_area


def _plane_contact_geometry(
    positions: np.ndarray,
    surface_edges: np.ndarray,
    center_xy: np.ndarray,
    plane_z: float,
) -> dict[str, float | int]:
    """Measure the contact patch from the zero-gap contour of the FEM surface.

    The contour is obtained only from the piecewise-linear surface and the
    rigid plane.  No Hertz profile or fitted analytical parameter enters the
    measurement.  Its projected area defines an equivalent contact radius.
    """
    first = surface_edges[:, 0]
    second = surface_edges[:, 1]
    first_gap = positions[first, 2] - plane_z
    second_gap = positions[second, 2] - plane_z
    crossing = ((first_gap <= 0.0) & (second_gap >= 0.0)) | ((first_gap >= 0.0) & (second_gap <= 0.0))
    crossing &= np.abs(first_gap - second_gap) > 1.0e-15
    if not np.any(crossing):
        return {
            "contact_area": 0.0,
            "equivalent_contact_radius": 0.0,
            "mean_boundary_radius": 0.0,
            "minimum_boundary_radius": 0.0,
            "maximum_boundary_radius": 0.0,
            "boundary_radius_cv": 0.0,
            "boundary_point_count": 0,
        }

    first = first[crossing]
    second = second[crossing]
    first_gap = first_gap[crossing]
    second_gap = second_gap[crossing]
    fraction = first_gap / (first_gap - second_gap)
    points = positions[first] + fraction[:, None] * (positions[second] - positions[first])
    # A vertex exactly on the plane may be found through more than one edge.
    points = np.unique(np.round(points, decimals=14), axis=0)
    if points.shape[0] < 3:
        return {
            "contact_area": 0.0,
            "equivalent_contact_radius": 0.0,
            "mean_boundary_radius": 0.0,
            "minimum_boundary_radius": 0.0,
            "maximum_boundary_radius": 0.0,
            "boundary_radius_cv": 0.0,
            "boundary_point_count": int(points.shape[0]),
        }

    relative = points[:, :2] - center_xy[None, :]
    order = np.argsort(np.arctan2(relative[:, 1], relative[:, 0]))
    polygon = points[order, :2]
    shifted = np.roll(polygon, -1, axis=0)
    area = 0.5 * abs(float(np.sum(polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1])))
    radii = np.linalg.norm(relative, axis=1)
    mean_radius = float(np.mean(radii))
    return {
        "contact_area": area,
        "equivalent_contact_radius": math.sqrt(area / math.pi),
        "mean_boundary_radius": mean_radius,
        "minimum_boundary_radius": float(np.min(radii)),
        "maximum_boundary_radius": float(np.max(radii)),
        "boundary_radius_cv": float(np.std(radii) / max(abs(mean_radius), 1.0e-30)),
        "boundary_point_count": int(points.shape[0]),
    }


def _contact_state(coupling):
    neighbor = coupling.contactor.neighbor
    count = int(neighbor.contact_count)
    active = neighbor.contacts.active.to_numpy()[:count].astype(bool)
    node_ids = neighbor.contacts.node_id.to_numpy()[:count]
    normal = neighbor.contacts.normal_force.to_numpy()[:count]
    tangential = neighbor.contacts.tangential_force.to_numpy()[:count]
    normal_gap = neighbor.contacts.normal_gap.to_numpy()[:count]
    fem_contact = normal[active].sum(axis=0) + tangential[active].sum(axis=0)
    rigid_contact = coupling.dem.scene.rigid.contact_force.to_numpy()[0]
    residual = fem_contact + rigid_contact
    scale = max(
        float(np.linalg.norm(fem_contact)),
        float(np.linalg.norm(rigid_contact)),
        1.0e-30,
    )
    return {
        "candidate_count": count,
        "active_contact_count": int(np.sum(active)),
        "candidate_node_ids": node_ids,
        "candidate_normal_force": normal,
        "active_node_ids": node_ids[active],
        "active_normal_force": normal[active],
        "active_normal_gap": normal_gap[active],
        "fem_contact": fem_contact,
        "rigid_contact": rigid_contact,
        "action_reaction_relative": float(np.linalg.norm(residual) / scale),
        "finite": bool(
            np.isfinite(normal).all() and np.isfinite(tangential).all() and np.isfinite(rigid_contact).all()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--default-fp", choices=("float32", "float64"), default="float64")
    parser.add_argument("--surface", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--wall", type=Path, default=DEFAULT_WALL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument(
        "--mesh-generator",
        choices=("gmsh", "delaunay"),
        default="gmsh",
        help="Quality-controlled Gmsh is the formal-study default.",
    )
    parser.add_argument(
        "--coarse-spacing-over-radius",
        type=float,
        default=0.40,
        help="Far-field interior spacing divided by R.",
    )
    parser.add_argument(
        "--contact-spacing-over-radius",
        type=float,
        default=0.20,
        help="Interior spacing in the lower contact refinement box, divided by R.",
    )
    parser.add_argument(
        "--surface-spacing-over-radius",
        type=float,
        default=0.015,
        help="Lower-cap boundary-node spacing divided by R.",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=HOLD_TIME,
        help="Minimum constant-load hold before adaptive equilibrium stopping.",
    )
    parser.add_argument(
        "--maximum-hold-time",
        type=float,
        default=MAXIMUM_HOLD_TIME,
        help="Maximum constant-load hold allowed for adaptive equilibrium stopping.",
    )
    parser.add_argument(
        "--normal-stiffness",
        type=float,
        default=NORMAL_STIFFNESS,
        help="Linear normal penalty stiffness in Pa/m.",
    )
    parser.add_argument(
        "--contact-model",
        choices=("linear", "barrier"),
        default="linear",
        help="Explicit FEM--LSDEM surface law.",
    )
    parser.add_argument(
        "--barrier-cutoff",
        type=float,
        default=BARRIER_CUTOFF,
        help="Barrier activation distance and admissible penetration bound in m.",
    )
    parser.add_argument(
        "--minimum-tetra-mean-ratio",
        type=float,
        default=MINIMUM_TETRA_MEAN_RATIO,
        help="Hard lower bound for the dimensionless tetrahedral mean ratio.",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="Generate and quality-gate the FEM mesh without initializing a solver.",
    )
    args = parser.parse_args()

    if not 0.10 <= args.coarse_spacing_over_radius <= 0.60:
        parser.error("--coarse-spacing-over-radius must lie in [0.10, 0.60]")
    if not 0.02 <= args.contact_spacing_over_radius <= 0.40:
        parser.error("--contact-spacing-over-radius must lie in [0.02, 0.40]")
    if not 0.005 <= args.surface_spacing_over_radius <= 0.20:
        parser.error("--surface-spacing-over-radius must lie in [0.005, 0.20]")
    if args.hold_time <= 0.0:
        parser.error("--hold-time must be positive")
    if args.maximum_hold_time < args.hold_time:
        parser.error("--maximum-hold-time must be no smaller than --hold-time")
    if args.normal_stiffness <= 0.0:
        parser.error("--normal-stiffness must be positive")
    if args.barrier_cutoff <= 0.0:
        parser.error("--barrier-cutoff must be positive")
    if not 0.0 < args.minimum_tetra_mean_ratio <= 1.0:
        parser.error("--minimum-tetra-mean-ratio must lie in (0, 1]")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface = args.surface.expanduser().resolve()
    wall = args.wall.expanduser().resolve()
    os.environ["GEOTAICHI_REAL_DTYPE"] = args.default_fp

    if args.mesh_generator == "gmsh":
        sphere_mesh, mesh_quality = _gmsh_locally_refined_sphere_mesh(
            RADIUS,
            CENTER,
            args.coarse_spacing_over_radius,
            args.contact_spacing_over_radius,
            args.surface_spacing_over_radius,
        )
    else:
        sphere_mesh = _locally_refined_sphere_mesh(
            surface,
            RADIUS,
            CENTER,
            args.coarse_spacing_over_radius,
            args.contact_spacing_over_radius,
            args.surface_spacing_over_radius,
        )
        mesh_quality = _tetra_quality(sphere_mesh.points, sphere_mesh.cells)
    mesh_quality.update(
        {
            "generator": args.mesh_generator,
            "node_count": sphere_mesh.number_of_nodes,
            "element_count": sphere_mesh.number_of_cells,
            "minimum_required_tetra_mean_ratio": args.minimum_tetra_mean_ratio,
        }
    )
    mesh_quality["passed"] = bool(
        mesh_quality["minimum_tetra_volume"] > 0.0
        and mesh_quality["minimum_tetra_mean_ratio"] >= args.minimum_tetra_mean_ratio
    )
    if args.mesh_only:
        payload = {
            "schema_version": 1,
            "mesh_quality": mesh_quality,
            "coarse_spacing_over_radius": args.coarse_spacing_over_radius,
            "contact_spacing_over_radius": args.contact_spacing_over_radius,
            "surface_spacing_over_radius": args.surface_spacing_over_radius,
        }
        print(json.dumps(payload, indent=2), flush=True)
        (output / "mesh_quality.json").write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")
        return 0 if mesh_quality["passed"] else 2

    import taichi as ti
    import geotaichi as gt
    from src.fem.boundaries import NeumannBoundary

    gt.init(
        arch=args.arch,
        default_fp=args.default_fp,
        log=False,
        debug=False,
        offline_cache=False,
    )
    cap_selector = lambda centroids: centroids[:, 2] >= CENTER[2] + 0.60 * RADIUS
    unit_boundary = NeumannBoundary().add_pressure(-1.0, selector=cap_selector)
    unit_vertical_force = -float(unit_boundary.force(sphere_mesh).sum(axis=0)[2])
    maximum_pressure = TARGET_LOAD / unit_vertical_force
    # Store one full-load nodal field on the device.  A scalar load factor is
    # updated during the ramp, avoiding O(number_of_steps * number_of_nodes)
    # boundary-history storage for long equilibrium holds.
    boundary = NeumannBoundary().add_pressure(-maximum_pressure, selector=cap_selector)

    dem = gt.DEM(log=False)
    dem.set_configuration(
        domain=[0.60, 0.60, 0.70],
        scheme="LSDEM",
        engine="SymplecticEuler",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_rigid_body_number": 1,
            "levelset_grid_number": 16000,
            "surface_node_number": 64,
            "max_sphere_number": 0,
            "max_clump_number": 0,
            "max_plane_number": 0,
            "body_coordination_number": 4,
            "wall_coordination_number": 1,
            "verlet_distance_multiplier": [0.1, 0.1],
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.add_attribute(
        materialID=0,
        attribute={
            "Density": 2500.0,
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    dem.add_template(
        {
            "Name": "hertz_rigid_wall",
            "Object": gt.polyhedron(file=str(wall)).grids(space=0.10, extent=2),
            "WriteFile": False,
        }
    )
    dem.create_body(
        {
            "BodyType": "RigidBody",
            "Template": [
                {
                    "Name": "hertz_rigid_wall",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [0.30, 0.30, 0.20],
                    "ScaleFactor": 0.40,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "FixMotion": ["Fix", "Fix", "Fix"],
                    "BodyOrientation": "constant",
                }
            ],
        }
    )
    dem.choose_contact_model(None, None)
    dem.select_save_data(particle=True, surface=True)

    fem = gt.FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit", log=False)
    fem.add_soft_particle(sphere_mesh)
    fem.add_material(
        "NeoHookean",
        density=DENSITY,
        young_modulus=YOUNG,
        poisson_ratio=POISSON,
    )
    fem.add_boundary_condition(neumann=boundary)

    minimum_total_time = RAMP_TIME + args.hold_time
    maximum_total_time = RAMP_TIME + args.maximum_hold_time
    coupling = gt.FEDEM(dem=dem, fem=fem, log=False)
    coupling.set_configuration(
        domain=[0.60, 0.60, 0.70],
        gravity=[0.0, 0.0, 0.0],
        search="BVH",
        contact_work_mode="Explicit",
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": maximum_total_time,
            "SaveInterval": OUTPUT_INTERVAL,
            "SavePath": str(output / "native"),
            "damping": FEM_DAMPING,
        },
        log=False,
    )
    coupling.select_save_data(contact=True, checkpoint=True)
    coupling.add_surface(body_ids=[0])
    coupling.memory_allocate(
        {
            "contact_coordination_number": 24,
            "max_contact_pairs": 8192,
            "max_levelset_cell_pairs": 32768,
            "verlet_distance_multiplier": 0.1,
        }
    )
    if args.contact_model == "barrier":
        coupling.choose_contact_model("Barrier")
        coupling.add_property(
            DEMmaterial=0,
            FEMbody=0,
            property={
                "Stiffness": args.normal_stiffness,
                "NormalCutOff": args.barrier_cutoff,
                "StiffnessRatio": 1.0,
                "Friction": 0.0,
                "NormalViscousDamping": 0.10,
                "TangentialViscousDamping": 0.0,
            },
        )
    else:
        coupling.choose_contact_model("Linear")
        coupling.add_property(
            DEMmaterial=0,
            FEMbody=0,
            property={
                "NormalStiffness": args.normal_stiffness,
                "TangentialStiffness": 0.5 * args.normal_stiffness,
                "Friction": 0.0,
                "NormalViscousDamping": 0.10,
                "TangentialViscousDamping": 0.0,
            },
        )
    started = time.perf_counter()
    coupling.add_essentials()
    coupling.enginer.pre_calculate()
    coupling.check_critical_timestep()
    ti.sync()
    setup_seconds = time.perf_counter() - started

    engine = coupling.enginer
    fem_engine = engine.fem_engine
    contact_surface_faces = coupling.patch.faces.to_numpy()[: coupling.patch.face_count]
    contact_surface_edges = _surface_edges(contact_surface_faces)
    # The persistent Neumann field stores the full target load.  Start from
    # the exact unloaded state; the production loop advances this scalar with
    # the smooth ramp without rebuilding or uploading a nodal force array.
    fem_engine.state.set_boundary_force_scale(0.0)
    effective_dt = float(coupling.sims.delta)
    steps = int(round(maximum_total_time / effective_dt))
    minimum_steps = int(round(minimum_total_time / effective_dt))
    preflight = {
        "requested_dt": args.dt,
        "effective_dt": effective_dt,
        "requested_timestep_honored": math.isclose(effective_dt, args.dt, rel_tol=1.0e-12, abs_tol=1.0e-15),
        "critical_dt": float(fem_engine.critical_time_step),
        "minimum_steps": minimum_steps,
        "maximum_steps": steps,
        "minimum_total_time": minimum_total_time,
        "maximum_total_time": maximum_total_time,
        "mechanical_time_scale": RADIUS * math.sqrt(DENSITY / YOUNG),
        "hertz_relaxation_frequency_per_second": HERTZ_RELAXATION_FREQUENCY,
        "relaxation_damping_ratio": RELAXATION_DAMPING_RATIO,
        "fem_mass_proportional_damping_per_second": FEM_DAMPING,
        "fem_nodes": int(fem_engine.mesh.number_of_nodes),
        "fem_elements": int(fem_engine.mesh.number_of_cells),
        "coarse_spacing_over_radius": args.coarse_spacing_over_radius,
        "contact_spacing_over_radius": args.contact_spacing_over_radius,
        "surface_spacing_over_radius": args.surface_spacing_over_radius,
        "normal_stiffness": args.normal_stiffness,
        "contact_model": args.contact_model,
        "barrier_cutoff": args.barrier_cutoff,
        "contact_work_mode": coupling.sims.contact_work_mode,
        "mesh_quality": mesh_quality,
    }
    if args.preflight:
        first_step = {"attempted": False, "passed": False}
        if preflight["requested_timestep_honored"] and mesh_quality["passed"]:
            first_step["attempted"] = True
            try:
                engine.reset_message()
                engine.update_verlet_tables()
                engine.system_resolve()
                engine.integration(update_diagnostics=False)
                ti.sync()
                first_step.update(
                    {
                        "finite_position": bool(np.isfinite(fem_engine.state.position.to_numpy()).all()),
                        "minimum_jacobian": float(engine.minimum_jacobian),
                    }
                )
                first_step["passed"] = bool(first_step["finite_position"] and first_step["minimum_jacobian"] > 0.10)
            except Exception as exc:
                first_step["error_type"] = type(exc).__name__
                first_step["error"] = str(exc)
        preflight["first_step_integration"] = first_step
        preflight["passed"] = bool(
            preflight["requested_timestep_honored"] and mesh_quality["passed"] and first_step["passed"]
        )
        print(json.dumps({"preflight": preflight}, indent=2), flush=True)
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + os.linesep, encoding="utf-8")
        return 0 if preflight["passed"] else 2
    preflight["passed"] = bool(preflight["requested_timestep_honored"] and mesh_quality["passed"])
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + os.linesep, encoding="utf-8")
    if not preflight["requested_timestep_honored"]:
        raise RuntimeError(
            "Requested timestep was corrected by the stability check; rerun "
            "with an explicitly safe timestep instead of accepting a silent change."
        )
    if not mesh_quality["passed"]:
        raise RuntimeError(
            "FEM mesh quality gate failed: minimum tetrahedral mean ratio "
            f"{mesh_quality['minimum_tetra_mean_ratio']:.6g} is below "
            f"{args.minimum_tetra_mean_ratio:.6g}. Regenerate or improve the mesh."
        )
    sample_stride = max(1, int(round(1.0e-3 / effective_dt)))
    progress_stride = max(1, steps // 100)
    output_stride = max(1, int(round(OUTPUT_INTERVAL / effective_dt)))
    window_steps = max(1, int(round(EQUILIBRIUM_WINDOW / effective_dt)))
    first_window_step = int(round((RAMP_TIME + EQUILIBRIUM_WINDOW) / effective_dt))
    initial_center = CENTER.copy()
    records = []
    pressure_snapshots = []
    peak_candidates = 0
    peak_active = 0
    maximum_action_reaction = 0.0
    minimum_jacobian = 1.0
    minimum_contact_gap = math.inf
    finite = True
    equilibrium_windows = []
    converged = False
    saved_steps = []

    loop_started = time.perf_counter()
    for step in range(steps + 1):
        time_value = min(step * effective_dt, maximum_total_time)
        fem_engine.state.set_boundary_force_scale(_smooth_ramp(time_value))
        engine.reset_message()
        engine.update_verlet_tables()
        engine.system_resolve()
        sample_now = step % sample_stride == 0 or step == steps
        sampled_internal_force = None
        if sample_now:
            contact = _contact_state(coupling)
            peak_candidates = max(peak_candidates, contact["candidate_count"])
            peak_active = max(peak_active, contact["active_contact_count"])
            maximum_action_reaction = max(maximum_action_reaction, contact["action_reaction_relative"])
            finite = finite and contact["finite"]
            if contact["active_normal_gap"].size:
                minimum_contact_gap = min(
                    minimum_contact_gap,
                    float(np.min(contact["active_normal_gap"])),
                )
            positions = fem_engine.state.position.to_numpy()
            velocity = fem_engine.state.velocity.to_numpy()
            mass = fem_engine.state.mass.to_numpy()
            center = np.sum(mass[:, None] * positions, axis=0) / np.sum(mass)
            contact_geometry = _plane_contact_geometry(
                positions,
                contact_surface_edges,
                center[:2],
                WALL_TOP,
            )
            active_nodes = contact["active_node_ids"]
            if active_nodes.size:
                reference_node_area = coupling.patch.node_area.to_numpy()
                projected_node_area = _projected_nodal_areas(
                    positions,
                    contact_surface_faces,
                )
                candidate_nodes = contact["candidate_node_ids"]
                pressure_snapshots.append(
                    {
                        "time": time_value,
                        "center_xy": center[:2].copy(),
                        "radial_distance": np.linalg.norm(positions[active_nodes, :2] - center[None, :2], axis=1),
                        "normal_force_magnitude": np.linalg.norm(contact["active_normal_force"], axis=1),
                        "projected_node_area": projected_node_area[active_nodes],
                        "candidate_node_ids": candidate_nodes.copy(),
                        "candidate_normal_force": contact["candidate_normal_force"].copy(),
                        "candidate_position": positions[candidate_nodes].copy(),
                        "candidate_radial_distance": np.linalg.norm(
                            positions[candidate_nodes, :2] - center[None, :2],
                            axis=1,
                        ),
                        "candidate_normal_force_magnitude": np.linalg.norm(contact["candidate_normal_force"], axis=1),
                        "candidate_projected_node_area": projected_node_area[candidate_nodes],
                        "candidate_reference_node_area": reference_node_area[candidate_nodes],
                        "equivalent_contact_radius": contact_geometry["equivalent_contact_radius"],
                        "boundary_radius_cv": contact_geometry["boundary_radius_cv"],
                    }
                )
            sampled_internal_force = fem_engine._assemble_internal_device(need_stiffness=False)
            strain_energy = float(fem_engine._internal_energy_device())
            kinetic_energy = float(0.5 * np.sum(mass[:, None] * velocity * velocity))
            jacobian = float(fem_engine._minimum_jacobian_ratio_device(fem_engine.state.position))
            minimum_jacobian = min(minimum_jacobian, jacobian)
            applied_load = TARGET_LOAD * _smooth_ramp(time_value)
            # The rigid wall receives the negative of the FEM nodal contact
            # resultant. Report a positive compressive reaction magnitude.
            reaction = max(-float(contact["rigid_contact"][2]), 0.0)
            indentation = max(RADIUS - (center[2] - WALL_TOP), 0.0)
            analytical = _hertz_indentation(applied_load)
            analytical_contact_radius = _hertz_contact_radius(reaction)
            relative_error = abs(indentation - analytical) / analytical if analytical > 0.0 else 0.0
            contact_radius_relative_error = (
                abs(contact_geometry["equivalent_contact_radius"] - analytical_contact_radius)
                / analytical_contact_radius
                if analytical_contact_radius > 0.0 and contact_geometry["equivalent_contact_radius"] > 0.0
                else 0.0
            )
            records.append(
                {
                    "step": step,
                    "time": time_value,
                    "ramp": _smooth_ramp(time_value),
                    "applied_load": applied_load,
                    "reaction_force": reaction,
                    "center_z": float(center[2]),
                    "center_drop": float(initial_center[2] - center[2]),
                    "indentation": indentation,
                    "hertz_indentation": analytical,
                    "indentation_relative_error": relative_error,
                    **contact_geometry,
                    "hertz_contact_radius": analytical_contact_radius,
                    "contact_radius_relative_error": contact_radius_relative_error,
                    "kinetic_energy": kinetic_energy,
                    "strain_energy": strain_energy,
                    "active_contact_count": contact["active_contact_count"],
                    "candidate_count": contact["candidate_count"],
                    "action_reaction_relative": contact["action_reaction_relative"],
                    "minimum_jacobian": jacobian,
                    "minimum_contact_gap": (
                        float(np.min(contact["active_normal_gap"])) if contact["active_normal_gap"].size else None
                    ),
                }
            )
            finite = finite and bool(
                np.isfinite(positions).all()
                and np.isfinite(velocity).all()
                and math.isfinite(strain_energy)
                and math.isfinite(kinetic_energy)
            )
            if step >= first_window_step and step % window_steps == 0:
                window = _equilibrium_window(records, time_value, EQUILIBRIUM_WINDOW)
                previous = equilibrium_windows[-1] if equilibrium_windows else None
                window["indentation_window_drift"] = (
                    abs(window["mean_indentation"] - previous["mean_indentation"])
                    / max(
                        0.5 * (abs(window["mean_indentation"]) + abs(previous["mean_indentation"])),
                        1.0e-30,
                    )
                    if previous is not None
                    else None
                )
                window["contact_radius_window_drift"] = (
                    abs(window["mean_contact_radius"] - previous["mean_contact_radius"])
                    / max(
                        0.5 * (abs(window["mean_contact_radius"]) + abs(previous["mean_contact_radius"])),
                        1.0e-30,
                    )
                    if previous is not None
                    else None
                )
                window["passed"] = bool(
                    window["reaction_relative_error"] <= REACTION_ERROR_TOLERANCE
                    and window["reaction_coefficient_variation"] <= REACTION_CV_TOLERANCE
                    and window["mean_kinetic_strain_ratio"] <= KINETIC_STRAIN_TOLERANCE
                    and window["contact_radius_window_drift"] is not None
                    and window["contact_radius_window_drift"] <= CONTACT_RADIUS_DRIFT_TOLERANCE
                )
                equilibrium_windows.append(window)
                print(json.dumps({"equilibrium_window": window}), flush=True)
                if (
                    time_value >= minimum_total_time - 0.5 * effective_dt
                    and len(equilibrium_windows) >= 2
                    and equilibrium_windows[-1]["passed"]
                    and equilibrium_windows[-2]["passed"]
                ):
                    converged = True
        save_now = step == 0 or step % output_stride == 0 or step == steps or converged
        if save_now and (not saved_steps or saved_steps[-1] != step):
            coupling.save_data()
            saved_steps.append(step)
        if converged:
            break
        if step == steps:
            break
        if step > 0 and step % progress_stride == 0:
            print(json.dumps({"case": "hertz", "progress": step / steps}), flush=True)
        engine.integration(
            internal_force=sampled_internal_force,
            update_diagnostics=False,
        )
        coupling.sims.current_time += effective_dt
        coupling.dem.sims.current_time += effective_dt
        fem_engine.time = coupling.sims.current_time
        coupling.sims.current_step += 1
        coupling.dem.sims.current_step += 1
        fem_engine.step_count += 1
        minimum_jacobian = min(minimum_jacobian, float(engine.minimum_jacobian))

    ti.sync()
    loop_seconds = time.perf_counter() - loop_started
    output_evidence = collect_native_output(output / "native", len(saved_steps))
    actual_end_time = float(records[-1]["time"])
    equilibrium = (
        equilibrium_windows[-1]
        if equilibrium_windows
        else _equilibrium_window(records, actual_end_time, EQUILIBRIUM_WINDOW)
    )
    mean_reaction = equilibrium["mean_reaction_force"]
    mean_indentation = equilibrium["mean_indentation"]
    reaction_error = equilibrium["reaction_relative_error"]
    mean_final_error = equilibrium["indentation_relative_error"]
    reaction_cv = equilibrium["reaction_coefficient_variation"]
    pressure_profile, pressure_diameter_profile, pressure_summary = _pressure_profile(
        pressure_snapshots,
        contact_surface_faces,
        equilibrium["window_start"],
        equilibrium["window_end"],
    )
    pressure_sample_archive = _write_pressure_sample_archive(
        output,
        pressure_snapshots,
        contact_surface_faces,
        equilibrium["window_start"],
        equilibrium["window_end"],
    )
    gates = {
        "finite_state": finite,
        "requested_timestep_honored": preflight["requested_timestep_honored"],
        "contact_occurred": peak_active > 0,
        "positive_jacobian": minimum_jacobian > 0.10,
        "action_reaction": maximum_action_reaction <= 1.0e-10,
        "adaptive_equilibrium": converged,
        "load_equilibrium": reaction_error <= REACTION_ERROR_TOLERANCE,
        "quasi_static_reaction": reaction_cv <= REACTION_CV_TOLERANCE,
        "quasi_static_energy": equilibrium["mean_kinetic_strain_ratio"] <= KINETIC_STRAIN_TOLERANCE,
        "contact_radius_stationarity": equilibrium.get("contact_radius_window_drift") is not None
        and equilibrium["contact_radius_window_drift"] <= CONTACT_RADIUS_DRIFT_TOLERANCE,
        "hertz_contact_radius": equilibrium["contact_radius_relative_error"] <= CONTACT_RADIUS_ERROR_TOLERANCE,
        "pressure_profile_evidence": (
            pressure_summary["independent_annulus_count"] >= PRESSURE_ANNULUS_COUNT
            and pressure_summary["diameter_point_count"] >= PRESSURE_DIAMETER_POINT_COUNT
        ),
        "pressure_force_closure": pressure_summary["annular_force_closure_relative"] <= 1.0e-10,
        "pressure_sample_archive": pressure_sample_archive["sample_count"] == pressure_summary["snapshot_count"] > 0,
        "capacity": peak_candidates <= 8192,
        "native_output": bool(output_evidence["complete"]),
    }
    if args.contact_model == "barrier":
        gates["barrier_domain"] = minimum_contact_gap > -args.barrier_cutoff
    metrics = {
        "schema_version": 1,
        "passed": all(gates.values()),
        "gates": gates,
        "mean_final_indentation_relative_error": mean_final_error,
        "center_based_indentation_relative_error": mean_final_error,
        "mean_final_contact_radius_relative_error": equilibrium["contact_radius_relative_error"],
        "mean_final_reaction_relative_error": reaction_error,
        "equilibrium": equilibrium,
        "equilibrium_windows": equilibrium_windows,
        "actual_end_time": actual_end_time,
        "pressure_profile": pressure_summary,
        "pressure_sample_archive": pressure_sample_archive,
        "minimum_jacobian": minimum_jacobian,
        "maximum_action_reaction_relative": maximum_action_reaction,
        "maximum_candidate_count": peak_candidates,
        "maximum_active_contact_count": peak_active,
        "minimum_contact_gap": (minimum_contact_gap if math.isfinite(minimum_contact_gap) else None),
        "minimum_gap_over_barrier_cutoff": (
            minimum_contact_gap / args.barrier_cutoff
            if args.contact_model == "barrier" and math.isfinite(minimum_contact_gap)
            else None
        ),
        "final": records[-1],
        "native_output": output_evidence,
    }
    source_fingerprint, source_files = _source_fingerprint()
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": _prompt(),
        "units": "SI",
        "execution": {
            "arch": args.arch,
            "precision": args.default_fp,
            "taichi_version": list(ti.__version__),
            "gpu": (
                _command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
                if args.arch == "gpu"
                else None
            ),
            "command": [sys.executable, *sys.argv],
            "source_fingerprint": source_fingerprint,
            "source_files": source_files,
        },
        "parameters": {
            "radius": RADIUS,
            "density": DENSITY,
            "young_modulus": YOUNG,
            "poisson_ratio": POISSON,
            "effective_modulus": YOUNG / (1.0 - POISSON * POISSON),
            "initial_gap": INITIAL_GAP,
            "target_indentation": TARGET_INDENTATION,
            "target_indentation_ratio": TARGET_INDENTATION / RADIUS,
            "target_load": TARGET_LOAD,
            "maximum_pressure": maximum_pressure,
            "cap_unit_vertical_force": unit_vertical_force,
            "requested_dt": args.dt,
            "effective_dt": effective_dt,
            "ramp_time": RAMP_TIME,
            "minimum_hold_time": args.hold_time,
            "maximum_hold_time": args.maximum_hold_time,
            "actual_hold_time": actual_end_time - RAMP_TIME,
            "equilibrium_window": EQUILIBRIUM_WINDOW,
            "reaction_error_tolerance": REACTION_ERROR_TOLERANCE,
            "reaction_cv_tolerance": REACTION_CV_TOLERANCE,
            "kinetic_strain_tolerance": KINETIC_STRAIN_TOLERANCE,
            "contact_radius_drift_tolerance": CONTACT_RADIUS_DRIFT_TOLERANCE,
            "contact_radius_error_tolerance": CONTACT_RADIUS_ERROR_TOLERANCE,
            "normal_stiffness": args.normal_stiffness,
            "tangential_stiffness": 0.5 * args.normal_stiffness,
            "contact_model": args.contact_model,
            "barrier_cutoff": args.barrier_cutoff,
            "fem_mass_proportional_damping_per_second": FEM_DAMPING,
            "relaxation_damping_ratio": RELAXATION_DAMPING_RATIO,
            "hertz_tangent_stiffness": HERTZ_TANGENT_STIFFNESS,
            "hertz_relaxation_frequency_per_second": HERTZ_RELAXATION_FREQUENCY,
            "sphere_mass": SPHERE_MASS,
            "coarse_spacing_over_radius": args.coarse_spacing_over_radius,
            "fine_spacing_over_radius": args.contact_spacing_over_radius,
            "surface_spacing_over_radius": args.surface_spacing_over_radius,
            "output_interval": OUTPUT_INTERVAL,
            "checkpoint_output": True,
            "pressure_annulus_count": PRESSURE_ANNULUS_COUNT,
            "pressure_diameter_point_count": PRESSURE_DIAMETER_POINT_COUNT,
            "pressure_triangle_subdivisions": PRESSURE_TRIANGLE_SUBDIVISIONS,
            "pressure_annular_remap": "piecewise_linear_projected_surface",
            "pressure_interpolation_or_smoothing": False,
            "contact_surface_measure": "current_projected_nodal_area",
            "contact_zone_half_width_over_radius": 0.25,
            "contact_zone_depth_over_radius": 0.25,
            "search": "BVH",
            "precision": args.default_fp,
            "mesh_generator": args.mesh_generator,
            "mesh_quality": mesh_quality,
        },
    }
    performance = {
        "schema_version": 1,
        "host": platform.node(),
        "python": platform.python_version(),
        "taichi": list(ti.__version__),
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "setup_seconds": setup_seconds,
        "simulation_loop_seconds": loop_seconds,
        "steps": int(records[-1]["step"]),
        "maximum_planned_steps": steps,
        "saved_frame_count": len(saved_steps),
        "saved_steps": saved_steps,
        "fem_nodes": int(fem_engine.mesh.number_of_nodes),
        "fem_elements": int(fem_engine.mesh.number_of_cells),
    }
    for filename, payload in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("performance.json", performance),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "pressure_profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pressure_profile[0]))
        writer.writeheader()
        writer.writerows(pressure_profile)
    with (output / "pressure_profile_diameter.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "x_over_contact_radius",
                "numerical_pressure",
                "analytical_pressure",
                "quadrature_time_samples",
                "source_radius_over_contact_radius",
            ],
        )
        writer.writeheader()
        writer.writerows(pressure_diameter_profile)
    print(json.dumps({"passed": metrics["passed"], "metrics": metrics}, indent=2))
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
