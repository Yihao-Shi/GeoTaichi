#!/usr/bin/env python3
"""Case-local geometry and diagnostics for isotropic compression."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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

from examples.fedem.IsotropicCompaction.output_audit import collect_native_output
from examples.fedem.IsotropicCompaction.mesh_reference import (
    place_reference,
    tetrahedralize_primitive_reference,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "OutputData/base"
IRREGULAR_SURFACE = REPO_ROOT / "assets/mesh/LSDEM/sand.stl"
PROMPT = (
    "Create a weakly polydisperse packing of soft FEM and rigid level-set "
    "DEM spheres inside six frictionless DEM walls. Establish a low-pressure "
    "jammed state, compress all walls isotropically, and record pressure, "
    "solid fraction, coordination, and deformation."
)

SEED = 20260816
PARTICLE_COUNT = 125
SOFT_COUNT = 63
RIGID_COUNT = PARTICLE_COUNT - SOFT_COUNT
RADIUS = 0.022
SOFT_DENSITY = 1200.0
RIGID_DENSITY = 2500.0
YOUNG = 2.0e4
POISSON = 0.30
FRICTION = 0.35
WALL_FRICTION = 0.0
NORMAL_STIFFNESS = 1.0e7
TANGENTIAL_STIFFNESS = 5.0e6
FEM_WALL_NORMAL_STIFFNESS = 1.0e8
FEM_WALL_TANGENTIAL_STIFFNESS = 5.0e7
DT = 1.0e-5
COMPRESSION_TIME = 3.125
OUTPUT_INTERVAL = 0.10
AXIAL_STRAIN = 0.25
LEFT, RIGHT = 0.124, 0.346
FRONT, BACK = 0.124, 0.346
BOTTOM, TOP = 0.124, 0.346
WIDTH0, DEPTH0, HEIGHT0 = RIGHT - LEFT, BACK - FRONT, TOP - BOTTOM
WALL_BODY_NAMES = ("left", "right", "front", "back", "bottom", "top")
WALL_EXTENSION = 0.10
MAXIMUM_WALL_PENETRATION = 0.05 * RADIUS
IRREGULAR_LEVELSET_SPACING = 5.0
MINIMUM_TETRA_MEAN_RATIO = 0.05
SOFT_CONTACT_THICKNESS = 1.0e-3
SOFT_PT_CAPACITY = 1048576
SOFT_EE_CAPACITY = 4194304
SOFT_HISTORY_CAPACITY = 2097152
FEM_WALL_PAIR_CAPACITY = 262144
FEM_WALL_HISTORY_CAPACITY = 65536
BARRIER_CUTOFF = 0.01 * RADIUS


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _clear_fresh_run_output(output: Path) -> None:
    """Remove stale trajectory artifacts before a non-restart run."""

    native_root = output / "native"
    if native_root.is_dir():
        for path in native_root.rglob("*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
    for name in (
        "history.csv",
        "config.json",
        "metrics.json",
        "state.json",
        "performance.json",
    ):
        path = output / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def _box_obj(path: Path, size):
    size = np.asarray(size, dtype=np.float64)
    vertices = (
        np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        * size[None, :]
    )
    faces = (
        (1, 3, 2),
        (1, 4, 3),
        (5, 6, 7),
        (5, 7, 8),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 4, 8),
        (3, 8, 7),
        (4, 1, 5),
        (4, 5, 8),
    )
    lines = [
        *("v %.12g %.12g %.12g" % tuple(vertex) for vertex in vertices),
        *("f %d %d %d" % face for face in faces),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prompt() -> str:
    return PROMPT


def _top_displacement(
    time_value: float,
    axial_strain: float,
    compression_time: float,
) -> float:
    fraction = min(max(float(time_value) / compression_time, 0.0), 1.0)
    return -axial_strain * HEIGHT0 * fraction


def _facet_wall_specs(axial_strain: float, compression_time: float):
    """Return six extended DEM polygon walls with inward normals."""

    ex = WALL_EXTENSION
    x0, x1 = LEFT, RIGHT
    y0, y1 = FRONT, BACK
    z0, z1 = BOTTOM, TOP
    dx, dy, dz = WIDTH0, DEPTH0, HEIGHT0
    xlo, xhi = x0 - ex * dx, x1 + ex * dx
    ylo, yhi = y0 - ex * dy, y1 + ex * dy
    zlo, zhi = z0 - ex * dz, z1 + ex * dz
    fixed = [0.0, 0.0, 0.0]
    top_speed = -axial_strain * HEIGHT0 / compression_time
    return (
        ("left", ([x0, ylo, zlo], [x0, yhi, zlo], [x0, yhi, zhi], [x0, ylo, zhi]), [1.0, 0.0, 0.0], fixed),
        ("right", ([x1, ylo, zlo], [x1, yhi, zlo], [x1, yhi, zhi], [x1, ylo, zhi]), [-1.0, 0.0, 0.0], fixed),
        ("front", ([xlo, y0, zlo], [xhi, y0, zlo], [xhi, y0, zhi], [xlo, y0, zhi]), [0.0, 1.0, 0.0], fixed),
        ("back", ([xlo, y1, zlo], [xhi, y1, zlo], [xhi, y1, zhi], [xlo, y1, zhi]), [0.0, -1.0, 0.0], fixed),
        ("bottom", ([xlo, ylo, z0], [xhi, ylo, z0], [xhi, yhi, z0], [xlo, yhi, z0]), [0.0, 0.0, 1.0], fixed),
        (
            "top",
            ([xlo, ylo, z1], [xhi, ylo, z1], [xhi, yhi, z1], [xlo, yhi, z1]),
            [0.0, 0.0, -1.0],
            [0.0, 0.0, top_speed],
        ),
    )


def _packing(minimum_surface_gap: float | None = None):
    if minimum_surface_gap is None:
        margin = 1.05 * RADIUS
        jitter_amplitude = 2.5e-4
    else:
        minimum_surface_gap = float(minimum_surface_gap)
        if minimum_surface_gap < 0.0:
            raise ValueError("minimum particle surface gap cannot be negative")
        margin = RADIUS + minimum_surface_gap
        minimum_spacing = min(
            (RIGHT - LEFT - 2.0 * margin) / 4.0,
            (BACK - FRONT - 2.0 * margin) / 4.0,
            (TOP - BOTTOM - 2.0 * margin) / 4.0,
        )
        excess_gap = minimum_spacing - 2.0 * RADIUS
        if excess_gap < minimum_surface_gap:
            raise ValueError(
                "triaxial box cannot place the barrier packing without " "crossing the requested initial surface gap"
            )
        jitter_amplitude = min(
            2.5e-4,
            0.25 * (excess_gap - minimum_surface_gap),
        )
    axes = (
        np.linspace(LEFT + margin, RIGHT - margin, 5),
        np.linspace(FRONT + margin, BACK - margin, 5),
        np.linspace(BOTTOM + margin, TOP - margin, 5),
    )
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(SEED)
    centers += rng.uniform(-jitter_amplitude, jitter_amplitude, centers.shape)
    permutation = rng.permutation(PARTICLE_COUNT)
    soft_ids = np.sort(permutation[:SOFT_COUNT])
    rigid_ids = np.sort(permutation[SOFT_COUNT:])
    orientations = rng.uniform(0.0, 360.0, (PARTICLE_COUNT, 3))
    return centers, soft_ids, rigid_ids, orientations


def _build(
    gt,
    output: Path,
    dt: float,
    simulation_time: float,
    axial_strain: float,
    output_interval: float,
    contact_model: str = "linear",
    barrier_cutoff: float = BARRIER_CUTOFF,
    fem_verlet_distance_multiplier: float = 0.12,
    dem_local_damping: float = 0.08,
    rigid_shape: str = "irregular",
    packing_override=None,
):
    use_barrier = str(contact_model).lower() == "barrier"
    if packing_override is None:
        centers, soft_ids, rigid_ids, orientations = _packing(barrier_cutoff if use_barrier else None)
        packing_radii = np.full(PARTICLE_COUNT, RADIUS)
    else:
        (
            centers,
            packing_radii,
            soft_ids,
            rigid_ids,
            orientations,
        ) = packing_override
        centers = np.asarray(centers, dtype=np.float64)
        packing_radii = np.asarray(packing_radii, dtype=np.float64)
        soft_ids = np.asarray(soft_ids, dtype=np.int64)
        rigid_ids = np.asarray(rigid_ids, dtype=np.int64)
        orientations = np.asarray(orientations, dtype=np.float64)
        if centers.shape != (PARTICLE_COUNT, 3):
            raise ValueError("packing centers must have shape (N, 3)")
        if packing_radii.shape != (PARTICLE_COUNT,):
            raise ValueError("packing radii must have shape (N,)")
    wall_specs = _facet_wall_specs(axial_strain, simulation_time)

    dem = gt.DEM(log=False)
    dem.set_configuration(
        domain=[0.47, 0.47, 0.47],
        scheme="LSDEM",
        engine="SymplecticEuler",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        track_energy=True,
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 2,
            "max_rigid_body_number": RIGID_COUNT,
            "max_rigid_template_number": 1,
            "levelset_grid_number": 100000,
            "surface_node_number": 512,
            "max_sphere_number": 0,
            "max_clump_number": 0,
            "max_plane_number": 0,
            "max_facet_number": 12,
            "body_coordination_number": 128,
            "wall_coordination_number": 12,
            "verlet_distance_multiplier": [0.1, 0.1],
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    for material_id in (0, 1):
        dem.add_attribute(
            materialID=material_id,
            attribute={
                "Density": RIGID_DENSITY,
                "ForceLocalDamping": dem_local_damping,
                "TorqueLocalDamping": dem_local_damping,
            },
        )
    if RIGID_COUNT > 0:
        template_name = "triaxial_irregular_grain"
        template = {
            "Name": template_name,
            "Object": gt.polyhedron(file=str(IRREGULAR_SURFACE)).grids(space=IRREGULAR_LEVELSET_SPACING, extent=3),
            "WriteFile": False,
        }
        if rigid_shape == "sphere":
            sphere_object = gt.sphere(1.0)
            sphere_object.analytical_distance(sphere_object._distance)
            template_name = "triaxial_sphere"
            template = {
                "Name": template_name,
                "Object": sphere_object.grids(space=0.1, extent=3),
                "SurfaceResolution": 1728,
                "WriteFile": False,
            }
        dem.add_template(template)
        dem.create_body_batch(
            {
                "BodyType": "RigidBody",
                "Template": {
                    "Name": template_name,
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoints": centers[rigid_ids],
                    "BoundingRadii": packing_radii[rigid_ids],
                    "CoordinatesAreMassCenters": False,
                    "BodyOrientationsRadians": np.deg2rad(orientations[rigid_ids]),
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "FixMotion": ["Free", "Free", "Free"],
                },
            }
        )
    walls_to_create = []
    for wall_id, (name, vertices, normal, velocity) in enumerate(wall_specs):
        walls_to_create.append(
            {
                "WallID": wall_id,
                "WallType": "Facet",
                "WallShape": "Polygon",
                "MaterialID": 1,
                "WallVertice": {f"vertice{index + 1}": np.asarray(vertex) for index, vertex in enumerate(vertices)},
                "OuterNormal": np.asarray(normal),
                "InitialVelocity": velocity,
            }
        )
    dem.add_wall(body=walls_to_create)
    wall_id_values = dem.scene.wall.wallID.to_numpy()[: int(dem.scene.wallNum[0])]
    wall_facet_ids = {
        name: np.flatnonzero(wall_id_values == wall_id).tolist() for wall_id, (name, _, _, _) in enumerate(wall_specs)
    }
    if any(len(ids) != 2 for ids in wall_facet_ids.values()):
        raise RuntimeError(f"triaxial DEM wall triangulation is incomplete: {wall_facet_ids}")
    dem.choose_contact_model(
        "Barrier Model" if use_barrier else "Linear Model",
        "Barrier Model" if use_barrier else "Linear Model",
    )
    dem.select_save_data(
        particle=True,
        surface=True,
        wall=True,
        particle_particle_contact=True,
        particle_wall_contact=True,
    )
    contact_parameters = (
        {
            "Stiffness": NORMAL_STIFFNESS,
            "NormalCutOff": barrier_cutoff,
            "StiffnessRatio": 1.0,
            "Friction": FRICTION,
            "NormalViscousDamping": 0.20,
            "TangentialViscousDamping": 0.10,
        }
        if use_barrier
        else {
            "NormalStiffness": NORMAL_STIFFNESS,
            "TangentialStiffness": TANGENTIAL_STIFFNESS,
            "Friction": FRICTION,
            "NormalViscousDamping": 0.20,
            "TangentialViscousDamping": 0.10,
        }
    )
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property=contact_parameters,
        dType="particle-particle",
    )
    dem.add_property(
        materialID1=0,
        materialID2=1,
        property={**contact_parameters, "Friction": WALL_FRICTION},
        dType="particle-wall",
    )

    fem = gt.FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit", log=False)
    sphere_reference, mesh_quality = tetrahedralize_primitive_reference(
        "sphere",
        minimum_element_count=1000,
        minimum_mean_ratio=MINIMUM_TETRA_MEAN_RATIO,
    )
    if mesh_quality["connected_body_count"] != 1:
        raise RuntimeError("generated FEM sphere reference must contain exactly one body")
    if mesh_quality["minimum_tetra_mean_ratio"] < MINIMUM_TETRA_MEAN_RATIO:
        raise RuntimeError(
            "FEM sphere reference mesh failed the tetrahedron quality gate: "
            f"{mesh_quality['minimum_tetra_mean_ratio']:.6g} < "
            f"{MINIMUM_TETRA_MEAN_RATIO:.6g}"
        )
    reference_mesh_root = output / "mesh"
    reference_mesh_root.mkdir(parents=True, exist_ok=True)
    sphere_reference.write(reference_mesh_root / "fem_sphere_reference.msh")
    sphere_reference.write(reference_mesh_root / "fem_sphere_reference.vtu")
    np.savez_compressed(
        reference_mesh_root / "fem_sphere_reference.npz",
        points=sphere_reference.points,
        cells=sphere_reference.cells,
        cell_type=np.asarray("TET4"),
    )
    (reference_mesh_root / "fem_sphere_reference_quality.json").write_text(
        json.dumps(mesh_quality, indent=2) + os.linesep,
        encoding="utf-8",
    )
    body_ranges = []
    body_roles = []
    node_offset = 0
    soft_meshes = []
    for body_id, packing_id in enumerate(soft_ids):
        mesh = place_reference(
            sphere_reference,
            centers[packing_id],
            packing_radii[packing_id],
            [0.0, 0.0, 0.0],
            name=f"triaxial_soft_sphere_{body_id:04d}",
        )
        soft_meshes.append(mesh)
        body_ranges.append((node_offset, node_offset + mesh.number_of_nodes))
        body_roles.append({"role": "soft", "packing_id": int(packing_id), "shape": "sphere"})
        node_offset += mesh.number_of_nodes
    body_roles.extend(
        {
            "role": "rigid",
            "packing_id": int(packing_id),
            "shape": "irregular sand morphology",
        }
        for packing_id in rigid_ids
    )
    from src.fem.generator import FEMMesh

    fem.add_soft_particle(FEMMesh.concatenate(soft_meshes))
    fem.add_material("NeoHookean", density=SOFT_DENSITY, young_modulus=YOUNG, poisson_ratio=POISSON)
    contact_property = (
        {
            "Stiffness": NORMAL_STIFFNESS,
            "NormalCutOff": barrier_cutoff,
            "StiffnessRatio": 1.0,
            "Friction": FRICTION,
            "NormalViscousDamping": 0.20,
            "TangentialViscousDamping": 0.10,
        }
        if use_barrier
        else {
            "NormalStiffness": NORMAL_STIFFNESS,
            "TangentialStiffness": TANGENTIAL_STIFFNESS,
            "Friction": FRICTION,
            "NormalViscousDamping": 0.20,
            "TangentialViscousDamping": 0.10,
        }
    )
    fem.add_soft_particle_contact(
        "Barrier" if use_barrier else "Linear",
        search="BVH",
        verlet_distance_multiplier=fem_verlet_distance_multiplier,
        ContactThickness=(barrier_cutoff if use_barrier else SOFT_CONTACT_THICKNESS),
        max_point_triangle_pairs=SOFT_PT_CAPACITY,
        max_edge_edge_pairs=SOFT_EE_CAPACITY,
        contact_history_capacity=SOFT_HISTORY_CAPACITY,
        **contact_property,
    )
    wall_contact_property = dict(contact_property)
    wall_contact_property["Friction"] = WALL_FRICTION
    if use_barrier:
        wall_contact_property["Stiffness"] = FEM_WALL_NORMAL_STIFFNESS
    else:
        wall_contact_property["NormalStiffness"] = FEM_WALL_NORMAL_STIFFNESS
        wall_contact_property["TangentialStiffness"] = FEM_WALL_TANGENTIAL_STIFFNESS

    coupling = gt.FEDEM(dem=dem, fem=fem, log=False)
    coupling.set_configuration(
        domain=[0.47, 0.47, 0.47],
        gravity=[0.0, 0.0, 0.0],
        search="BVH",
        contact_work_mode="Explicit",
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": dt,
            "SimulationTime": simulation_time,
            "SaveInterval": output_interval,
            "SavePath": str(output / "native"),
            "damping": 0.04,
        },
        log=False,
    )
    coupling.select_save_data(contact=True, checkpoint=True)
    coupling.add_surface(body_ids=list(range(SOFT_COUNT)))
    coupling.memory_allocate(
        {
            # Keep one inert coupled-table slot for the 100%-soft endpoint;
            # ContactManager still reads the realized DEM particle count (0).
            "max_particle_number": max(RIGID_COUNT, 1),
            "contact_coordination_number": 256,
            "max_contact_pairs": 1048576 if RIGID_COUNT > 0 else 16,
            "max_levelset_cell_pairs": 4194304 if RIGID_COUNT > 0 else 16,
            # FEM--wall candidates are independent of the number of LSDEM
            # bodies.  In particular, the all-soft endpoint must not inherit
            # the inert 16-entry FEM--LSDEM capacity.
            "max_fem_wall_pairs": FEM_WALL_PAIR_CAPACITY,
            "max_fem_wall_history_pairs": FEM_WALL_HISTORY_CAPACITY,
            "verlet_distance_multiplier": fem_verlet_distance_multiplier,
        }
    )
    coupling.choose_contact_model("Barrier" if use_barrier else "Linear")
    for body_id in range(SOFT_COUNT):
        coupling.add_property(
            DEMmaterial=0,
            FEMbody=body_id,
            property=contact_property,
        )
        coupling.add_property(
            DEMmaterial=1,
            FEMbody=body_id,
            property=wall_contact_property,
        )
    coupling.add_essentials()
    coupling.enginer.pre_calculate()
    coupling.check_critical_timestep()
    coupling.initial_packing_radii = packing_radii.copy()
    coupling.realized_wall_geometry = _dem_wall_geometry(coupling, wall_facet_ids)
    return (
        coupling,
        centers,
        soft_ids,
        rigid_ids,
        body_ranges,
        wall_facet_ids,
        body_roles,
        mesh_quality,
        sphere_reference.number_of_nodes,
        sphere_reference.number_of_cells,
    )


def _soft_centers(coupling, body_ranges):
    positions = coupling.enginer.fem_engine.state.position.to_numpy()
    mass = coupling.enginer.fem_engine.state.mass.to_numpy()
    result = []
    for start, end in body_ranges[:SOFT_COUNT]:
        result.append(np.sum(mass[start:end, None] * positions[start:end], axis=0) / np.sum(mass[start:end]))
    return np.asarray(result)


def _soft_volume(coupling, positions: np.ndarray) -> float:
    cells = np.asarray(coupling.enginer.fem_engine.mesh.cells, dtype=np.int64)
    tetrahedra = positions[cells]
    determinants = np.linalg.det(
        np.stack(
            (
                tetrahedra[:, 1] - tetrahedra[:, 0],
                tetrahedra[:, 2] - tetrahedra[:, 0],
                tetrahedra[:, 3] - tetrahedra[:, 0],
            ),
            axis=2,
        )
    )
    return float(np.sum(np.abs(determinants)) / 6.0)


def _rigid_particle_contact_pairs(dem, rigid_ids):
    from src.utils.FieldIO import field_to_numpy_prefix

    pairs: set[tuple[int, int]] = set()
    rigid_ids = np.asarray(rigid_ids, dtype=np.int64)
    if rigid_ids.size <= 1:
        return pairs
    scene = dem.scene
    neighbor = dem.contactor.neighbor
    surface_count = int(scene.surfaceNum[0])
    contact_count = int(neighbor.lsparticle_lsparticle[surface_count])
    if contact_count <= 0:
        return pairs
    model = dem.contactor.physpp
    first_nodes = field_to_numpy_prefix(model.cplist.endID1, contact_count)
    second_bodies = field_to_numpy_prefix(model.cplist.endID2, contact_count)
    normal_force = field_to_numpy_prefix(model.cplist.cnforce, contact_count)
    active = np.linalg.norm(normal_force, axis=1) > 0.0
    surface_body = field_to_numpy_prefix(scene.surface, surface_count)
    for node, second in zip(first_nodes[active], second_bodies[active]):
        first = int(rigid_ids[int(surface_body[int(node)])])
        second = int(rigid_ids[int(second)])
        if first != second:
            pairs.add((min(first, second), max(first, second)))
    return pairs


def _particle_contact_pairs(coupling, soft_ids, rigid_ids):
    """Return active particle pairs after collapsing nodal contacts."""

    from src.utils.FieldIO import field_to_numpy_prefix

    pairs: set[tuple[int, int]] = set()
    soft_ids = np.asarray(soft_ids, dtype=np.int64)
    rigid_ids = np.asarray(rigid_ids, dtype=np.int64)

    def insert(first, second):
        first = int(first)
        second = int(second)
        if first != second:
            pairs.add((min(first, second), max(first, second)))

    soft_contact = coupling.enginer.fem_engine.soft_particle_contact
    node_body = np.asarray(soft_contact.mesh.node_body_ids, dtype=np.int64)
    if soft_contact.pt_count > 0:
        active = field_to_numpy_prefix(soft_contact.pt_active, soft_contact.pt_count).astype(bool)
        stencils = field_to_numpy_prefix(soft_contact.culling.point_triangle, soft_contact.pt_count)[active]
        for stencil in stencils:
            insert(
                soft_ids[node_body[int(stencil[0])]],
                soft_ids[node_body[int(stencil[1])]],
            )
    if soft_contact.ee_count > 0:
        active = field_to_numpy_prefix(soft_contact.ee_active, soft_contact.ee_count).astype(bool)
        stencils = field_to_numpy_prefix(soft_contact.culling.edge_edge, soft_contact.ee_count)[active]
        for stencil in stencils:
            insert(
                soft_ids[node_body[int(stencil[0])]],
                soft_ids[node_body[int(stencil[2])]],
            )

    cross = coupling.contactor.neighbor
    if cross is not None and cross.contact_count > 0:
        active = field_to_numpy_prefix(cross.contacts.active, cross.contact_count).astype(bool)
        nodes = field_to_numpy_prefix(cross.contacts.node_id, cross.contact_count)[active]
        rigids = field_to_numpy_prefix(cross.contacts.rigid_id, cross.contact_count)[active]
        patch_body = field_to_numpy_prefix(coupling.patch.node_body, coupling.patch.node_count)
        for node, rigid in zip(nodes, rigids):
            insert(
                soft_ids[int(patch_body[int(node)])],
                rigid_ids[int(rigid)],
            )

    pairs.update(_rigid_particle_contact_pairs(coupling.dem, rigid_ids))
    return pairs


def _wall_stresses(wall_force, width: float, depth: float, height: float):
    sigma_x = (
        0.5 * (abs(float(wall_force["left"][0])) + abs(float(wall_force["right"][0]))) / max(depth * height, 1.0e-30)
    )
    sigma_y = (
        0.5 * (abs(float(wall_force["front"][1])) + abs(float(wall_force["back"][1]))) / max(width * height, 1.0e-30)
    )
    sigma_z = (
        0.5 * (abs(float(wall_force["bottom"][2])) + abs(float(wall_force["top"][2]))) / max(width * depth, 1.0e-30)
    )
    return sigma_x, sigma_y, sigma_z, (sigma_x + sigma_y + sigma_z) / 3.0


def _run_rigid_only(gt, args, output: Path) -> int:
    """Run the exact 0%-soft endpoint with the standalone LSDEM engine."""

    import taichi as ti

    centers, _, rigid_ids, orientations = _packing()
    wall_specs = _facet_wall_specs(args.axial_strain, args.time)
    dem = gt.DEM(log=False)
    dem.set_configuration(
        domain=[0.47, 0.47, 0.47],
        scheme="LSDEM",
        engine="SymplecticEuler",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        track_energy=True,
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 2,
            "max_rigid_body_number": PARTICLE_COUNT,
            "max_rigid_template_number": 1,
            "levelset_grid_number": 100000,
            "surface_node_number": 512,
            "max_sphere_number": 0,
            "max_clump_number": 0,
            "max_plane_number": 0,
            "max_facet_number": 12,
            "body_coordination_number": 128,
            "wall_coordination_number": 12,
            "verlet_distance_multiplier": [0.1, 0.1],
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    for material_id in (0, 1):
        dem.add_attribute(
            materialID=material_id,
            attribute={
                "Density": RIGID_DENSITY,
                "ForceLocalDamping": 0.08,
                "TorqueLocalDamping": 0.08,
            },
        )
    dem.add_template(
        {
            "Name": "triaxial_irregular_grain",
            "Object": gt.polyhedron(file=str(IRREGULAR_SURFACE)).grids(space=IRREGULAR_LEVELSET_SPACING, extent=3),
            "WriteFile": False,
        }
    )
    dem.create_body_batch(
        {
            "BodyType": "RigidBody",
            "Template": {
                "Name": "triaxial_irregular_grain",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoints": centers,
                "BoundingRadii": np.full(PARTICLE_COUNT, RADIUS),
                "CoordinatesAreMassCenters": False,
                "BodyOrientationsRadians": np.deg2rad(orientations),
                "InitialVelocity": [0.0, 0.0, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "FixMotion": ["Free", "Free", "Free"],
            },
        }
    )
    walls = []
    for wall_id, (_, vertices, normal, velocity) in enumerate(wall_specs):
        walls.append(
            {
                "WallID": wall_id,
                "WallType": "Facet",
                "WallShape": "Polygon",
                "MaterialID": 1,
                "WallVertice": {f"vertice{index + 1}": np.asarray(vertex) for index, vertex in enumerate(vertices)},
                "OuterNormal": np.asarray(normal),
                "InitialVelocity": velocity,
            }
        )
    dem.add_wall(body=walls)
    wall_id_values = dem.scene.wall.wallID.to_numpy()[: int(dem.scene.wallNum[0])]
    wall_facet_ids = {
        name: np.flatnonzero(wall_id_values == wall_id).tolist() for wall_id, (name, _, _, _) in enumerate(wall_specs)
    }
    dem.choose_contact_model("Linear Model", "Linear Model")
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property={
            "NormalStiffness": NORMAL_STIFFNESS,
            "TangentialStiffness": TANGENTIAL_STIFFNESS,
            "Friction": FRICTION,
            "NormalViscousDamping": 0.20,
            "TangentialViscousDamping": 0.10,
        },
        dType="particle-particle",
    )
    dem.add_property(
        materialID1=0,
        materialID2=1,
        property={
            "NormalStiffness": NORMAL_STIFFNESS,
            "TangentialStiffness": TANGENTIAL_STIFFNESS,
            "Friction": WALL_FRICTION,
            "NormalViscousDamping": 0.20,
            "TangentialViscousDamping": 0.10,
        },
        dType="particle-wall",
    )
    dem.select_save_data(
        particle=True,
        surface=True,
        wall=True,
        particle_particle_contact=True,
        particle_wall_contact=True,
    )
    dem.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.time,
            "SaveInterval": args.output_interval,
            "SavePath": str(output / "native"),
        },
        log=False,
    )
    dem.add_essentials()
    engine = dem.enginer
    engine.pre_calculation(dem.sims, dem.scene, dem.contactor.neighbor)
    dem.check_critical_timestep()
    effective_dt = float(dem.sims.delta)
    steps = int(round(args.time / effective_dt))

    def resolve_contacts():
        engine.reset_wall_message(dem.scene)
        engine.reset_particle_message(dem.scene)
        engine.reset_contact_energy()
        engine.update_neighbor_lists(dem.sims, dem.scene, dem.contactor.neighbor)

    if args.preflight:
        resolve_contacts()
        engine.integration(dem.sims, dem.scene, dem.contactor.neighbor)
        ti.sync()
        centers_now = dem.scene.rigid.mass_center.to_numpy()[:PARTICLE_COUNT]
        preflight = {
            "schema_version": 1,
            "passed": bool(np.isfinite(centers_now).all()),
            "requested_dt": args.dt,
            "effective_dt": effective_dt,
            "steps": steps,
            "particle_count": PARTICLE_COUNT,
            "soft_particle_count": 0,
            "rigid_particle_count": PARTICLE_COUNT,
            "requested_soft_percent": 0.0,
            "realized_soft_fraction": 0.0,
            "particle_particle_friction": FRICTION,
            "particle_wall_friction": WALL_FRICTION,
        }
        (output / "preflight.json").write_text(
            json.dumps(preflight, indent=2) + os.linesep,
            encoding="utf-8",
        )
        print(json.dumps(preflight, indent=2))
        return 0 if preflight["passed"] else 2

    _clear_fresh_run_output(output)
    output_stride = max(1, int(round(args.output_interval / effective_dt)))
    sample_stride = max(1, int(round(args.sample_interval / effective_dt)))
    progress_stride = max(1, steps // 10)
    rigid_mass = dem.scene.rigid.m.to_numpy()[:PARTICLE_COUNT]
    rigid_volume = float(np.sum(rigid_mass) / RIGID_DENSITY)
    records = []
    saved_steps = []
    finite = True
    started = time.perf_counter()
    for step in range(steps + 1):
        time_value = min(step * effective_dt, args.time)
        resolve_contacts()
        if step % sample_stride == 0 or step == steps:
            facet_force = dem.scene.wall.contact_force.to_numpy()[: int(dem.scene.wallNum[0])]
            wall_force = {name: np.sum(facet_force[facet_ids], axis=0) for name, facet_ids in wall_facet_ids.items()}
            top_disp = _top_displacement(time_value, args.axial_strain, args.time)
            width = WIDTH0
            depth = DEPTH0
            height = HEIGHT0 + top_disp
            area = width * depth
            axial_stress = abs(float(wall_force["top"][2])) / max(area, 1.0e-30)
            sigma_x, sigma_y, sigma_z, mean_pressure = _wall_stresses(wall_force, width, depth, height)
            pairs = _rigid_particle_contact_pairs(dem, rigid_ids)
            reduced_young = YOUNG / (2.0 * (1.0 - POISSON**2))
            centers_now = dem.scene.rigid.mass_center.to_numpy()[:PARTICLE_COUNT]
            velocity = dem.scene.rigid.v.to_numpy()[:PARTICLE_COUNT]
            finite = finite and bool(
                np.isfinite(centers_now).all() and np.isfinite(velocity).all() and math.isfinite(mean_pressure)
            )
            records.append(
                {
                    "step": step,
                    "time": time_value,
                    "axial_strain": -top_disp / HEIGHT0,
                    "volumetric_strain": 1.0 - height / HEIGHT0,
                    "axial_stress": axial_stress,
                    "fem_axial_stress": 0.0,
                    "rigid_axial_stress": axial_stress,
                    "lateral_stress": 0.5 * (sigma_x + sigma_y),
                    "confining_stress_x": sigma_x,
                    "confining_stress_y": sigma_y,
                    "confining_stress_z": sigma_z,
                    "mean_confining_pressure": mean_pressure,
                    "pressure_over_young": mean_pressure / YOUNG,
                    "pressure_over_reduced_young": (mean_pressure / reduced_young),
                    "solid_fraction": rigid_volume / max(width * depth * height, 1.0e-30),
                    "soft_solid_volume": 0.0,
                    "rigid_solid_volume": rigid_volume,
                    "box_volume": width * depth * height,
                    "particle_contact_pair_count": len(pairs),
                    "coordination_number": 2.0 * len(pairs) / PARTICLE_COUNT,
                    "top_wall_displacement": -top_disp,
                    "soft_kinetic_energy": 0.0,
                    "rigid_kinetic_energy": float(0.5 * np.sum(rigid_mass[:, None] * velocity * velocity)),
                    "fem_strain_energy": 0.0,
                    "minimum_jacobian": 1.0,
                }
            )
        if step == 0 or step % output_stride == 0 or step == steps:
            dem.save_data()
            saved_steps.append(step)
        if step == steps:
            break
        engine.integration(dem.sims, dem.scene, dem.contactor.neighbor)
        dem.sims.current_time += effective_dt
        dem.sims.current_step += 1
        if step > 0 and step % progress_stride == 0:
            print(
                json.dumps(
                    {
                        "case": "triaxial_soft_000",
                        "progress": step / steps,
                    }
                ),
                flush=True,
            )
    ti.sync()
    loop_seconds = time.perf_counter() - started
    final_centers = dem.scene.rigid.mass_center.to_numpy()[:PARTICLE_COUNT]
    gates = {
        "finite_state": finite,
        "compressive_axial_reaction": records[-1]["axial_stress"] > 0.0,
        "target_axial_strain": records[-1]["axial_strain"] >= 0.99 * args.axial_strain,
    }
    metrics = {
        "schema_version": 1,
        "passed": all(gates.values()),
        "gates": gates,
        "particle_count": PARTICLE_COUNT,
        "soft_particle_count": 0,
        "rigid_particle_count": PARTICLE_COUNT,
        "requested_soft_percent": 0.0,
        "realized_soft_fraction": 0.0,
        "final": records[-1],
    }
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "units": "SI",
        "seed": SEED,
        "parameters": {
            "particle_count": PARTICLE_COUNT,
            "soft_count": 0,
            "rigid_count": PARTICLE_COUNT,
            "requested_soft_percent": 0.0,
            "young_modulus_reference": YOUNG,
            "reduced_young_modulus_reference": YOUNG / (2.0 * (1.0 - POISSON**2)),
            "particle_particle_friction": FRICTION,
            "particle_wall_friction": WALL_FRICTION,
            "requested_dt": args.dt,
            "effective_dt": effective_dt,
            "compression_time": args.time,
            "target_axial_strain": args.axial_strain,
            "output_interval": args.output_interval,
            "sample_interval": args.sample_interval,
        },
    }
    state = {
        "packing_centers": centers.tolist(),
        "soft_packing_ids": [],
        "rigid_packing_ids": rigid_ids.tolist(),
        "final_rigid_centers": final_centers.tolist(),
    }
    performance = {
        "schema_version": 1,
        "simulation_loop_seconds": loop_seconds,
        "steps": steps,
        "saved_frame_count": len(saved_steps),
        "saved_steps": saved_steps,
    }
    for filename, payload in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("state.json", state),
        ("performance.json", performance),
    ):
        (output / filename).write_text(
            json.dumps(payload, indent=2) + os.linesep,
            encoding="utf-8",
        )
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"passed": metrics["passed"], "metrics": metrics}, indent=2))
    return 0 if metrics["passed"] else 2


def _dem_wall_geometry(coupling, wall_facet_ids):
    scene = coupling.dem.scene
    count = int(scene.wallNum[0])
    wall_id = scene.wall.wallID.to_numpy()[:count]
    point1 = scene.wall.vertice1.to_numpy()[:count]
    point2 = scene.wall.vertice2.to_numpy()[:count]
    point3 = scene.wall.vertice3.to_numpy()[:count]
    normal = scene.wall.norm.to_numpy()[:count]
    geometry = {}
    for name, facet_ids in wall_facet_ids.items():
        points = np.concatenate((point1[facet_ids], point2[facet_ids], point3[facet_ids]), axis=0)
        geometry[name] = {
            "wall_id": int(wall_id[facet_ids[0]]),
            "facet_ids": [int(value) for value in facet_ids],
            "normal": np.mean(normal[facet_ids], axis=0).tolist(),
            "bounds": np.stack((points.min(axis=0), points.max(axis=0))).tolist(),
        }
    return geometry


def _fem_wall_reaction_by_facet(coupling):
    """Recover FEM-only reactions from persistent node--facet contacts."""

    contactor = coupling.contactor
    reactions = np.zeros((contactor.wall_count, 3), dtype=np.float64)
    active_counts = np.zeros(contactor.wall_count, dtype=np.int64)
    if contactor.wall_contacts is None or contactor.wall_candidate_count <= 0:
        return reactions, active_counts
    candidate_pairs = contactor.wall_candidate_pairs.to_numpy()[: contactor.wall_candidate_count]
    active = contactor.wall_contacts.active.to_numpy().astype(bool)
    normal_force = contactor.wall_contacts.normal_force.to_numpy()
    tangential_force = contactor.wall_contacts.tangential_force.to_numpy()
    for pair in candidate_pairs:
        pair = int(pair)
        if active[pair]:
            wall_index = pair % contactor.wall_count
            active_counts[wall_index] += 1
            reactions[wall_index] -= normal_force[pair] + tangential_force[pair]
    return reactions, active_counts


def _surface_clearances(coupling, wall_facet_ids, body_ranges):
    """Measure wall penetration and detect centroid or complete-body crossing."""

    scene = coupling.dem.scene
    wall_count = int(scene.wallNum[0])
    point1 = scene.wall.vertice1.to_numpy()[:wall_count]
    point2 = scene.wall.vertice2.to_numpy()[:wall_count]
    point3 = scene.wall.vertice3.to_numpy()[:wall_count]
    normals = scene.wall.norm.to_numpy()[:wall_count]
    fem_points = coupling.patch.nodes.to_numpy()
    fem_body = coupling.patch.node_body.to_numpy()
    fem_surface = fem_body >= 0
    fem_points = fem_points[fem_surface]
    fem_body = fem_body[fem_surface].astype(np.int64, copy=False)
    if np.unique(fem_body).size != len(body_ranges):
        raise RuntimeError("wall-crossing audit requires surface nodes for every FEM body")
    fem_centers = _soft_centers(coupling, body_ranges)
    if RIGID_COUNT > 0:
        _, rigid_points = scene.visualize_surface(coupling.dem.sims)
        rigid_points = rigid_points[: int(scene.surfaceNum[0])]
    else:
        rigid_points = np.empty((0, 3), dtype=np.float64)
    result = {}
    for name, facet_ids in wall_facet_ids.items():
        facet_id = facet_ids[0]
        center = (point1[facet_id] + point2[facet_id] + point3[facet_id]) / 3.0
        normal = normals[facet_id]
        fem_gap = (fem_points - center) @ normal
        rigid_gap = (rigid_points - center) @ normal
        centroid_gap = (fem_centers - center) @ normal
        fully_crossed = 0
        maximum_outside_fraction = 0.0
        for body_id in range(len(body_ranges)):
            body_gap = fem_gap[fem_body == body_id]
            if body_gap.size == 0:
                raise RuntimeError(f"FEM body {body_id} has no registered surface nodes")
            fully_crossed += int(float(np.max(body_gap)) < 0.0)
            maximum_outside_fraction = max(
                maximum_outside_fraction,
                float(np.mean(body_gap < 0.0)),
            )
        result[name] = {
            "minimum_fem_surface_gap": float(np.min(fem_gap)),
            "minimum_lsdem_surface_gap": (float(np.min(rigid_gap)) if rigid_gap.size else math.inf),
            "minimum_fem_body_centroid_gap": float(np.min(centroid_gap)),
            "fem_body_centroid_crossing_count": int(np.count_nonzero(centroid_gap < 0.0)),
            "fem_fully_crossed_body_count": int(fully_crossed),
            "maximum_fem_body_surface_fraction_outside": (maximum_outside_fraction),
        }
    diagnostics = {
        "maximum_fem_surface_penetration": max(
            max(0.0, -values["minimum_fem_surface_gap"]) for values in result.values()
        ),
        "maximum_lsdem_surface_penetration": max(
            max(0.0, -values["minimum_lsdem_surface_gap"]) for values in result.values()
        ),
        "fem_body_centroid_wall_crossing_pair_count": sum(
            values["fem_body_centroid_crossing_count"] for values in result.values()
        ),
        "fem_fully_crossed_wall_pair_count": sum(values["fem_fully_crossed_body_count"] for values in result.values()),
    }
    diagnostics["maximum_surface_penetration"] = max(
        diagnostics["maximum_fem_surface_penetration"],
        diagnostics["maximum_lsdem_surface_penetration"],
    )
    return result, diagnostics


def main() -> int:
    from src.utils.FieldIO import field_to_numpy_prefix

    global SOFT_COUNT, RIGID_COUNT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--default-fp", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--time", type=float, default=COMPRESSION_TIME)
    parser.add_argument("--axial-strain", type=float, default=AXIAL_STRAIN)
    parser.add_argument("--soft-percent", type=float, default=50.0)
    parser.add_argument("--output-interval", type=float, default=OUTPUT_INTERVAL)
    parser.add_argument("--sample-interval", type=float, default=OUTPUT_INTERVAL)
    parser.add_argument(
        "--restart",
        type=Path,
        help=(
            "Resume from an exact FEDEM checkpoint after rebuilding the "
            "same mesh, particles, walls, and contact model."
        ),
    )
    parser.add_argument(
        "--jacobian-check-stride",
        type=int,
        default=1000,
        help="Check the FEM minimum Jacobian every N steps and at output frames.",
    )
    parser.add_argument(
        "--neighbor-check-stride",
        type=int,
        default=10,
        help="Read device Verlet rebuild flags every N steps.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Build the complete coupled problem, report the realized discretization, and exit.",
    )
    args = parser.parse_args()
    if args.time <= 0.0:
        parser.error("--time must be positive")
    if not 0.0 <= args.soft_percent <= 100.0:
        parser.error("--soft-percent must lie in [0, 100]")
    if args.output_interval <= 0.0 or args.sample_interval <= 0.0:
        parser.error("output and sample intervals must be positive")
    if not 0.0 < args.axial_strain < 1.0:
        parser.error("--axial-strain must lie in (0, 1)")
    if args.jacobian_check_stride <= 0:
        parser.error("--jacobian-check-stride must be positive")
    if args.neighbor_check_stride <= 0:
        parser.error("--neighbor-check-stride must be positive")
    SOFT_COUNT = int(math.floor(PARTICLE_COUNT * args.soft_percent / 100.0 + 0.5))
    RIGID_COUNT = PARTICLE_COUNT - SOFT_COUNT
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cancellation = output / "CANCELLED"
    if cancellation.is_file():
        print(
            json.dumps(
                {
                    "cancelled": True,
                    "reason": cancellation.read_text(encoding="utf-8").strip(),
                    "output": str(output),
                },
                indent=2,
            )
        )
        return 75
    os.environ["GEOTAICHI_REAL_DTYPE"] = args.default_fp

    import taichi as ti
    import geotaichi as gt

    gt.init(arch=args.arch, default_fp=args.default_fp, log=False, debug=False, offline_cache=False)
    if SOFT_COUNT == 0:
        return _run_rigid_only(gt, args, output)
    started = time.perf_counter()
    (
        coupling,
        packing_centers,
        soft_ids,
        rigid_ids,
        body_ranges,
        wall_facet_ids,
        body_roles,
        mesh_quality,
        nodes_per_soft_particle,
        elements_per_soft_particle,
    ) = _build(
        gt,
        output,
        args.dt,
        args.time,
        args.axial_strain,
        args.output_interval,
    )
    restart_metadata = None
    if args.restart is not None:
        built_effective_dt = float(coupling.sims.delta)
        restart_metadata = coupling.read_restart(args.restart.expanduser().resolve())
        checkpoint_dt = float(restart_metadata["scalar_state"]["coupling.delta"])
        if not math.isclose(
            checkpoint_dt,
            built_effective_dt,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(
                "restart timestep differs from the rebuilt model: " f"{checkpoint_dt:.17g} != {built_effective_dt:.17g}"
            )
        saved_print = int(restart_metadata["scalar_state"]["coupling.current_print"])
        coupling.sims.current_print = saved_print + 1
        coupling.dem.sims.current_print = saved_print + 1
        coupling.solver.last_save_time = coupling.sims.current_time
    ti.sync()
    setup_seconds = time.perf_counter() - started
    engine = coupling.enginer
    effective_dt = float(coupling.sims.delta)
    steps = int(round(args.time / effective_dt))
    start_step = int(coupling.sims.current_step)
    if start_step > steps:
        raise ValueError(f"restart step {start_step} exceeds target step {steps}")
    coupling.sims.set_simulation_time(args.time)
    coupling.fem.engine.total_step = steps
    if args.preflight:
        step_started = time.perf_counter()
        engine.reset_message()
        engine.update_verlet_tables()
        engine.system_resolve()
        internal_force = engine.fem_engine._assemble_internal_device(need_stiffness=False)
        engine.integration(
            internal_force=internal_force,
            update_diagnostics=False,
        )
        coupling.patch.update(engine.fem_engine.position_field, update_normals=False)
        ti.sync()
        one_step_seconds = time.perf_counter() - step_started
        positions = engine.fem_engine.state.position.to_numpy()
        velocity = engine.fem_engine.state.velocity.to_numpy()
        minimum_jacobian = float(engine.fem_engine._minimum_jacobian_ratio_device(engine.fem_engine.state.position))
        cross_candidate_count = (
            0 if coupling.contactor.neighbor is None else int(coupling.contactor.neighbor.contact_count)
        )
        soft_contact_diagnostics = engine.fem_engine.soft_particle_contact.diagnostics()
        wall_contact_diagnostics = coupling.contactor.wall_contact_diagnostics()
        surface_clearance, crossing_diagnostics = _surface_clearances(coupling, wall_facet_ids, body_ranges)
        wall_velocity = coupling.dem.scene.wall.v.to_numpy()[: int(coupling.dem.scene.wallNum[0])]
        realized_wall_velocity = {name: wall_velocity[facet_ids].tolist() for name, facet_ids in wall_facet_ids.items()}
        fixed_wall_velocity_ok = all(
            np.allclose(wall_velocity[wall_facet_ids[name]], 0.0)
            for name in ("left", "right", "front", "back", "bottom")
        )
        top_wall_velocity_ok = np.allclose(
            wall_velocity[wall_facet_ids["top"]],
            [
                0.0,
                0.0,
                -args.axial_strain * HEIGHT0 / args.time,
            ],
        )
        preflight = {
            "schema_version": 1,
            "requested_dt": args.dt,
            "effective_dt": effective_dt,
            "simulation_time": args.time,
            "target_axial_strain": args.axial_strain,
            "steps": steps,
            "particle_count": PARTICLE_COUNT,
            "soft_particle_count": SOFT_COUNT,
            "rigid_particle_count": RIGID_COUNT,
            "requested_soft_percent": args.soft_percent,
            "realized_soft_fraction": SOFT_COUNT / PARTICLE_COUNT,
            "fem_nodes_per_soft_particle": nodes_per_soft_particle,
            "fem_elements_per_soft_particle": elements_per_soft_particle,
            "fem_reference_mesh_quality": mesh_quality,
            "fem_nodes": int(engine.fem_engine.mesh.number_of_nodes),
            "fem_elements": int(engine.fem_engine.mesh.number_of_cells),
            "fem_reference_connected_body_count": int(mesh_quality["connected_body_count"]),
            "assembled_fem_body_count": int(engine.fem_engine.mesh.body_ids.size),
            "cross_search": "BVH",
            "setup_seconds": setup_seconds,
            "one_step_seconds": one_step_seconds,
            "one_step_finite": bool(np.isfinite(positions).all() and np.isfinite(velocity).all()),
            "one_step_minimum_jacobian": minimum_jacobian,
            "one_step_cross_candidate_count": cross_candidate_count,
            "one_step_soft_contact_diagnostics": soft_contact_diagnostics,
            "one_step_wall_contact_diagnostics": wall_contact_diagnostics,
            "one_step_surface_clearance": surface_clearance,
            "one_step_wall_crossing_diagnostics": crossing_diagnostics,
            "one_step_maximum_surface_penetration": crossing_diagnostics["maximum_surface_penetration"],
            "realized_dem_wall_geometry": coupling.realized_wall_geometry,
            "realized_dem_wall_velocity": realized_wall_velocity,
            "contact_work_mode": coupling.sims.contact_work_mode,
        }
        preflight["passed"] = bool(
            preflight["one_step_finite"]
            and minimum_jacobian > 0.10
            and mesh_quality["minimum_tetra_mean_ratio"] >= MINIMUM_TETRA_MEAN_RATIO
            and mesh_quality["connected_body_count"] == 1
            and int(engine.fem_engine.mesh.body_ids.size) == SOFT_COUNT
            and elements_per_soft_particle >= 1000
            and int(coupling.dem.scene.wallNum[0]) == 12
            and fixed_wall_velocity_ok
            and top_wall_velocity_ok
            and crossing_diagnostics["maximum_surface_penetration"] <= MAXIMUM_WALL_PENETRATION
            and crossing_diagnostics["fem_body_centroid_wall_crossing_pair_count"] == 0
            and crossing_diagnostics["fem_fully_crossed_wall_pair_count"] == 0
        )
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + os.linesep, encoding="utf-8")
        print(json.dumps(preflight, indent=2))
        return 0 if preflight["passed"] else 2
    if args.restart is None:
        _clear_fresh_run_output(output)
    progress_stride = max(1, steps // 10)
    output_stride = max(1, int(round(args.output_interval / effective_dt)))
    # All scalar diagnostics download device data. Align them with native
    # output frames and leave the intervening explicit steps device resident.
    sample_stride = max(1, int(round(args.sample_interval / effective_dt)))
    records = []
    prior_state = {}
    prior_performance = {}
    if args.restart is not None:
        history_path = output / "history.csv"
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as stream:
                records = [row for row in csv.DictReader(stream) if int(float(row["step"])) <= start_step]
        state_path = output / "state.json"
        if state_path.is_file():
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
        performance_path = output / "performance.json"
        if performance_path.is_file():
            prior_performance = json.loads(performance_path.read_text(encoding="utf-8"))

    def _history_max(name: str, default=0.0):
        return max(
            (float(row[name]) for row in records if name in row),
            default=default,
        )

    peak_cross_candidates = int(_history_max("cross_candidate_count"))
    peak_cross_active = int(_history_max("cross_active_contact_count"))
    peak_soft_candidates = int(_history_max("soft_candidate_count"))
    peak_soft_active = int(_history_max("soft_active_contact_count"))
    peak_wall_active = int(_history_max("fem_wall_active_contact_count"))
    peak_top_fem_wall_active = int(_history_max("top_fem_wall_active_contact_count"))
    peak_fem_axial_stress = _history_max("fem_axial_stress")
    maximum_top_soft_displacement = _history_max("top_soft_mean_displacement")
    maximum_wall_penetration = _history_max("maximum_wall_penetration")
    maximum_fem_wall_penetration = _history_max("maximum_fem_wall_penetration")
    maximum_lsdem_wall_penetration = _history_max("maximum_lsdem_wall_penetration")
    maximum_fem_centroid_crossing_pairs = int(_history_max("fem_body_centroid_wall_crossing_pair_count"))
    maximum_fem_fully_crossed_pairs = int(_history_max("fem_fully_crossed_wall_pair_count"))
    latest_surface_clearance = {}
    latest_crossing_diagnostics = {}
    minimum_jacobian = min(
        (float(row["minimum_jacobian"]) for row in records),
        default=1.0,
    )
    finite = True
    saved_steps = [int(value) for value in prior_performance.get("saved_steps", [])]
    if args.restart is not None and (not saved_steps or saved_steps[-1] != start_step):
        raise ValueError("restart output prefix is incomplete or does not end at the " f"checkpoint step {start_step}")
    initial_soft_centers = np.asarray(
        prior_state.get(
            "initial_soft_centers",
            _soft_centers(coupling, body_ranges).tolist(),
        ),
        dtype=np.float64,
    )
    initial_rigid_centers = np.asarray(
        prior_state.get(
            "initial_rigid_centers",
            coupling.dem.scene.rigid.mass_center.to_numpy()[:RIGID_COUNT].tolist(),
        ),
        dtype=np.float64,
    )
    top_soft_mask = initial_soft_centers[:, 2] >= np.max(initial_soft_centers[:, 2]) - 0.5 * RADIUS

    loop_started = time.perf_counter()
    for step in range(start_step, steps + 1):
        time_value = min(step * effective_dt, args.time)
        engine.reset_message()
        check_neighbors = step % args.neighbor_check_stride == 0
        engine.update_verlet_tables(check_rebuild=check_neighbors)
        engine.system_resolve(check_rebuild=check_neighbors)
        sample_now = step % sample_stride == 0 or step == steps
        sampled_internal_force = None
        if sample_now:
            cross = coupling.contactor.neighbor
            cross_count = 0 if cross is None else int(cross.contact_count)
            cross_active = (
                np.empty(0, dtype=bool)
                if cross is None
                else field_to_numpy_prefix(cross.contacts.active, cross_count).astype(bool)
            )
            peak_cross_candidates = max(peak_cross_candidates, cross_count)
            peak_cross_active = max(peak_cross_active, int(np.sum(cross_active)))
            soft_contact = engine.fem_engine.soft_particle_contact
            soft_diag = soft_contact.diagnostics()
            soft_candidates = soft_diag["point_triangle_candidates"] + soft_diag["edge_edge_candidates"]
            soft_active = soft_diag["point_triangle_active"] + soft_diag["edge_edge_active"]
            peak_soft_candidates = max(peak_soft_candidates, soft_candidates)
            peak_soft_active = max(peak_soft_active, soft_active)
            wall_diag = coupling.contactor.wall_contact_diagnostics()
            peak_wall_active = max(peak_wall_active, wall_diag["active_count"])
            (
                latest_surface_clearance,
                latest_crossing_diagnostics,
            ) = _surface_clearances(coupling, wall_facet_ids, body_ranges)
            maximum_wall_penetration = max(
                maximum_wall_penetration,
                wall_diag["maximum_penetration"],
                latest_crossing_diagnostics["maximum_surface_penetration"],
            )
            maximum_fem_wall_penetration = max(
                maximum_fem_wall_penetration,
                latest_crossing_diagnostics["maximum_fem_surface_penetration"],
            )
            maximum_lsdem_wall_penetration = max(
                maximum_lsdem_wall_penetration,
                latest_crossing_diagnostics["maximum_lsdem_surface_penetration"],
            )
            maximum_fem_centroid_crossing_pairs = max(
                maximum_fem_centroid_crossing_pairs,
                latest_crossing_diagnostics["fem_body_centroid_wall_crossing_pair_count"],
            )
            maximum_fem_fully_crossed_pairs = max(
                maximum_fem_fully_crossed_pairs,
                latest_crossing_diagnostics["fem_fully_crossed_wall_pair_count"],
            )

            fem_engine = engine.fem_engine
            positions = fem_engine.state.position.to_numpy()
            velocity = fem_engine.state.velocity.to_numpy()
            mass = fem_engine.state.mass.to_numpy()
            sampled_internal_force = fem_engine._assemble_internal_device(need_stiffness=False)
            strain_energy = float(fem_engine._internal_energy_device())
            soft_kinetic = float(0.5 * np.sum(mass[:, None] * velocity**2))
            rigid_mass = coupling.dem.scene.rigid.m.to_numpy()[:RIGID_COUNT]
            rigid_velocity = coupling.dem.scene.rigid.v.to_numpy()[:RIGID_COUNT]
            rigid_kinetic = float(0.5 * np.sum(rigid_mass[:, None] * rigid_velocity**2))
            # Sample the wall reactions assembled by this call to
            # system_resolve().  Calling get_wall_contact_forces() again here
            # would add the LSDEM--wall contribution twice.
            facet_contact_force = coupling.dem.scene.wall.contact_force.to_numpy()
            fem_facet_reaction, fem_facet_active = _fem_wall_reaction_by_facet(coupling)
            rigid_facet_reaction = facet_contact_force - fem_facet_reaction
            wall_force = {
                name: np.sum(facet_contact_force[facet_ids], axis=0) for name, facet_ids in wall_facet_ids.items()
            }
            fem_wall_force = {
                name: np.sum(fem_facet_reaction[facet_ids], axis=0) for name, facet_ids in wall_facet_ids.items()
            }
            rigid_wall_force = {
                name: np.sum(rigid_facet_reaction[facet_ids], axis=0) for name, facet_ids in wall_facet_ids.items()
            }
            fem_wall_active = {
                name: int(np.sum(fem_facet_active[facet_ids])) for name, facet_ids in wall_facet_ids.items()
            }
            top_disp = _top_displacement(
                time_value,
                args.axial_strain,
                args.time,
            )
            width = WIDTH0
            depth = DEPTH0
            height = HEIGHT0 + top_disp
            axial_strain = -top_disp / HEIGHT0
            volumetric_strain = 1.0 - width * depth * height / (WIDTH0 * DEPTH0 * HEIGHT0)
            platen_area = max(width * depth, 1.0e-30)
            axial_stress = abs(float(wall_force["top"][2])) / platen_area
            fem_axial_stress = abs(float(fem_wall_force["top"][2])) / platen_area
            rigid_axial_stress = abs(float(rigid_wall_force["top"][2])) / platen_area
            sigma_x, sigma_y, sigma_z, mean_pressure = _wall_stresses(wall_force, width, depth, height)
            lateral_stress = 0.5 * (sigma_x + sigma_y)
            soft_volume = _soft_volume(coupling, positions)
            rigid_volume = float(np.sum(rigid_mass) / RIGID_DENSITY)
            box_volume = width * depth * height
            solid_fraction = (soft_volume + rigid_volume) / max(box_volume, 1.0e-30)
            particle_pairs = _particle_contact_pairs(coupling, soft_ids, rigid_ids)
            coordination_number = 2.0 * len(particle_pairs) / PARTICLE_COUNT
            reduced_young = YOUNG / (2.0 * (1.0 - POISSON**2))
            jacobian = float(fem_engine._minimum_jacobian_ratio_device(fem_engine.state.position))
            minimum_jacobian = min(minimum_jacobian, jacobian)
            contact_activity = cross_active.sum() + soft_active + wall_diag["active_count"]
            current_soft_centers = _soft_centers(coupling, body_ranges)
            top_soft_displacement = float(
                np.mean(initial_soft_centers[top_soft_mask, 2] - current_soft_centers[top_soft_mask, 2])
            )
            top_wall_displacement = -top_disp
            top_soft_wall_transfer_ratio = (
                top_soft_displacement / top_wall_displacement if top_wall_displacement > 0.0 else 0.0
            )
            peak_top_fem_wall_active = max(peak_top_fem_wall_active, fem_wall_active["top"])
            peak_fem_axial_stress = max(peak_fem_axial_stress, fem_axial_stress)
            maximum_top_soft_displacement = max(maximum_top_soft_displacement, top_soft_displacement)
            records.append(
                {
                    "step": step,
                    "time": time_value,
                    "axial_strain": axial_strain,
                    "volumetric_strain": volumetric_strain,
                    "axial_stress": axial_stress,
                    "fem_axial_stress": fem_axial_stress,
                    "rigid_axial_stress": rigid_axial_stress,
                    "lateral_stress": lateral_stress,
                    "confining_stress_x": sigma_x,
                    "confining_stress_y": sigma_y,
                    "confining_stress_z": sigma_z,
                    "mean_confining_pressure": mean_pressure,
                    "pressure_over_young": mean_pressure / YOUNG,
                    "pressure_over_reduced_young": (mean_pressure / reduced_young),
                    "solid_fraction": solid_fraction,
                    "soft_solid_volume": soft_volume,
                    "rigid_solid_volume": rigid_volume,
                    "box_volume": box_volume,
                    "particle_contact_pair_count": len(particle_pairs),
                    "coordination_number": coordination_number,
                    "top_wall_displacement": top_wall_displacement,
                    "top_soft_mean_displacement": top_soft_displacement,
                    "top_soft_wall_transfer_ratio": (top_soft_wall_transfer_ratio),
                    "cross_candidate_count": cross_count,
                    "cross_active_contact_count": int(np.sum(cross_active)),
                    "soft_candidate_count": soft_candidates,
                    "soft_active_contact_count": soft_active,
                    "fem_wall_active_contact_count": wall_diag["active_count"],
                    "top_fem_wall_active_contact_count": fem_wall_active["top"],
                    "maximum_wall_penetration": maximum_wall_penetration,
                    "maximum_fem_wall_penetration": (maximum_fem_wall_penetration),
                    "maximum_lsdem_wall_penetration": (maximum_lsdem_wall_penetration),
                    "fem_body_centroid_wall_crossing_pair_count": (
                        latest_crossing_diagnostics["fem_body_centroid_wall_crossing_pair_count"]
                    ),
                    "fem_fully_crossed_wall_pair_count": (
                        latest_crossing_diagnostics["fem_fully_crossed_wall_pair_count"]
                    ),
                    "contact_activity_per_particle": float(contact_activity / PARTICLE_COUNT),
                    "soft_kinetic_energy": soft_kinetic,
                    "rigid_kinetic_energy": rigid_kinetic,
                    "fem_strain_energy": strain_energy,
                    "minimum_jacobian": jacobian,
                }
            )
            finite = finite and bool(
                np.isfinite(positions).all()
                and np.isfinite(velocity).all()
                and math.isfinite(axial_stress)
                and math.isfinite(lateral_stress)
            )
        if (step == 0 or step % output_stride == 0 or step == steps) and (not saved_steps or saved_steps[-1] != step):
            coupling.save_data()
            saved_steps.append(step)
        if step == steps:
            break
        if step > 0 and step % progress_stride == 0:
            print(json.dumps({"case": "mixed_triaxial_125", "progress": step / steps}), flush=True)
        engine.integration(
            internal_force=sampled_internal_force,
            update_diagnostics=False,
            check_jacobian=(
                (step + 1) % args.jacobian_check_stride == 0 or (step + 1) % output_stride == 0 or step + 1 == steps
            ),
        )
        coupling.sims.current_time += effective_dt
        coupling.dem.sims.current_time += effective_dt
        engine.fem_engine.time = coupling.sims.current_time
        coupling.sims.current_step += 1
        coupling.dem.sims.current_step += 1
        engine.fem_engine.step_count += 1
    ti.sync()
    loop_seconds = time.perf_counter() - loop_started
    output_evidence = collect_native_output(output / "native", len(saved_steps))
    final_soft_centers = _soft_centers(coupling, body_ranges)
    final_rigid_centers = coupling.dem.scene.rigid.mass_center.to_numpy()[:RIGID_COUNT]
    final_surface_clearance, final_crossing_diagnostics = _surface_clearances(coupling, wall_facet_ids, body_ranges)
    maximum_wall_penetration = max(
        maximum_wall_penetration,
        final_crossing_diagnostics["maximum_surface_penetration"],
    )
    maximum_fem_wall_penetration = max(
        maximum_fem_wall_penetration,
        final_crossing_diagnostics["maximum_fem_surface_penetration"],
    )
    maximum_lsdem_wall_penetration = max(
        maximum_lsdem_wall_penetration,
        final_crossing_diagnostics["maximum_lsdem_surface_penetration"],
    )
    maximum_fem_centroid_crossing_pairs = max(
        maximum_fem_centroid_crossing_pairs,
        final_crossing_diagnostics["fem_body_centroid_wall_crossing_pair_count"],
    )
    maximum_fem_fully_crossed_pairs = max(
        maximum_fem_fully_crossed_pairs,
        final_crossing_diagnostics["fem_fully_crossed_wall_pair_count"],
    )
    lower = np.array([LEFT, FRONT, BOTTOM]) - RADIUS
    upper = np.array([RIGHT, BACK, TOP]) + RADIUS
    inside = lambda values: np.all((values >= lower) & (values <= upper), axis=1)
    inside_count = int(np.sum(inside(final_soft_centers)) + np.sum(inside(final_rigid_centers)))
    final = records[-1]
    gates = {
        "finite_state": finite,
        "positive_jacobian": minimum_jacobian > 0.10,
        "surface_nonpenetration": (maximum_wall_penetration <= MAXIMUM_WALL_PENETRATION),
        "no_fem_centroid_wall_crossing": (maximum_fem_centroid_crossing_pairs == 0),
        "no_fem_complete_body_wall_crossing": (maximum_fem_fully_crossed_pairs == 0),
        "compressive_axial_reaction": final["axial_stress"] > 0.0,
        "fem_top_wall_response": (
            peak_top_fem_wall_active > 0 and peak_fem_axial_stress > 0.0 and maximum_top_soft_displacement > 0.0
        ),
        "target_axial_strain": (final["axial_strain"] >= 0.99 * args.axial_strain),
        "mesh_quality": (
            mesh_quality["minimum_tetra_mean_ratio"] >= MINIMUM_TETRA_MEAN_RATIO
            and mesh_quality["connected_body_count"] == 1
            and int(engine.fem_engine.mesh.body_ids.size) == SOFT_COUNT
        ),
        "cross_capacity": peak_cross_candidates <= 1048576,
        "soft_contact_capacity": peak_soft_candidates < 10000000,
        "native_output": bool(output_evidence["complete"]),
    }
    state = {
        "body_roles": body_roles,
        "packing_centers": packing_centers.tolist(),
        "soft_packing_ids": soft_ids.tolist(),
        "rigid_packing_ids": rigid_ids.tolist(),
        "soft_shape": "sphere",
        "rigid_shape": "irregular sand morphology",
        "initial_soft_centers": initial_soft_centers.tolist(),
        "initial_rigid_centers": initial_rigid_centers.tolist(),
        "final_soft_centers": final_soft_centers.tolist(),
        "final_rigid_centers": final_rigid_centers.tolist(),
    }
    metrics = {
        "schema_version": 1,
        "passed": all(gates.values()),
        "gates": gates,
        "particle_count": PARTICLE_COUNT,
        "soft_particle_count": SOFT_COUNT,
        "rigid_particle_count": RIGID_COUNT,
        "requested_soft_percent": args.soft_percent,
        "realized_soft_fraction": SOFT_COUNT / PARTICLE_COUNT,
        "minimum_jacobian": minimum_jacobian,
        "inside_platen_count": inside_count,
        "maximum_wall_penetration": maximum_wall_penetration,
        "maximum_fem_wall_penetration": maximum_fem_wall_penetration,
        "maximum_lsdem_wall_penetration": maximum_lsdem_wall_penetration,
        "maximum_wall_penetration_gate": MAXIMUM_WALL_PENETRATION,
        "final_surface_clearance": final_surface_clearance,
        "final_wall_crossing_diagnostics": final_crossing_diagnostics,
        "maximum_fem_body_centroid_wall_crossing_pair_count": (maximum_fem_centroid_crossing_pairs),
        "maximum_fem_fully_crossed_wall_pair_count": (maximum_fem_fully_crossed_pairs),
        "fem_nodes_per_soft_particle": nodes_per_soft_particle,
        "fem_elements_per_soft_particle": elements_per_soft_particle,
        "fem_reference_mesh_quality": mesh_quality,
        "maximum_cross_candidate_count": peak_cross_candidates,
        "maximum_cross_active_count": peak_cross_active,
        "maximum_soft_candidate_count": peak_soft_candidates,
        "maximum_soft_active_count": peak_soft_active,
        "maximum_fem_wall_active_count": peak_wall_active,
        "maximum_top_fem_wall_active_count": peak_top_fem_wall_active,
        "maximum_fem_axial_stress": peak_fem_axial_stress,
        "maximum_top_soft_mean_displacement": maximum_top_soft_displacement,
        "final": final,
        "native_output": output_evidence,
    }
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": _prompt(),
        "units": "SI",
        "seed": SEED,
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
        },
        "parameters": {
            "particle_count": PARTICLE_COUNT,
            "soft_count": SOFT_COUNT,
            "rigid_count": RIGID_COUNT,
            "requested_soft_percent": args.soft_percent,
            "realized_soft_fraction": SOFT_COUNT / PARTICLE_COUNT,
            "soft_geometry": "sphere",
            "rigid_geometry": "irregular sand morphology",
            "fem_reference_mesh": "generated Gmsh unit sphere",
            "fem_reference_mesh_files": [
                "mesh/fem_sphere_reference.msh",
                "mesh/fem_sphere_reference.vtu",
                "mesh/fem_sphere_reference.npz",
                "mesh/fem_sphere_reference_quality.json",
            ],
            "fem_reference_minimum_element_count": 1000,
            "lsdem_reference_surface": str(IRREGULAR_SURFACE.relative_to(REPO_ROOT)),
            "fem_nodes_per_soft_particle": nodes_per_soft_particle,
            "fem_elements_per_soft_particle": elements_per_soft_particle,
            "fem_reference_mesh_quality": mesh_quality,
            "minimum_tetra_mean_ratio_gate": MINIMUM_TETRA_MEAN_RATIO,
            "soft_contact_thickness": SOFT_CONTACT_THICKNESS,
            "soft_point_triangle_capacity": SOFT_PT_CAPACITY,
            "soft_edge_edge_capacity": SOFT_EE_CAPACITY,
            "soft_contact_history_capacity": SOFT_HISTORY_CAPACITY,
            "radius": RADIUS,
            "soft_density": SOFT_DENSITY,
            "rigid_density": RIGID_DENSITY,
            "young_modulus": YOUNG,
            "reduced_young_modulus": YOUNG / (2.0 * (1.0 - POISSON**2)),
            "poisson_ratio": POISSON,
            "particle_particle_friction": FRICTION,
            "particle_wall_friction": WALL_FRICTION,
            "normal_stiffness": NORMAL_STIFFNESS,
            "tangential_stiffness": TANGENTIAL_STIFFNESS,
            "fem_wall_normal_stiffness": FEM_WALL_NORMAL_STIFFNESS,
            "fem_wall_tangential_stiffness": FEM_WALL_TANGENTIAL_STIFFNESS,
            "requested_dt": args.dt,
            "effective_dt": effective_dt,
            "compression_time": args.time,
            "top_wall_velocity": (-args.axial_strain * HEIGHT0 / args.time),
            "fixed_walls": ["left", "right", "front", "back", "bottom"],
            "target_axial_strain": args.axial_strain,
            "wall_type": "DEM polygon facet",
            "wall_facet_count": int(coupling.dem.scene.wallNum[0]),
            "wall_extension_fraction": WALL_EXTENSION,
            "maximum_wall_penetration_gate": MAXIMUM_WALL_PENETRATION,
            "realized_dem_wall_geometry": coupling.realized_wall_geometry,
            "irregular_levelset_spacing": IRREGULAR_LEVELSET_SPACING,
            "output_interval": args.output_interval,
            "sample_interval": args.sample_interval,
            "checkpoint_output": True,
            "search": "BVH",
            "precision": args.default_fp,
        },
    }
    performance = {
        "schema_version": 1,
        "host": platform.node(),
        "python": platform.python_version(),
        "taichi": list(ti.__version__),
        "gpu": _command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "setup_seconds": setup_seconds,
        "simulation_loop_seconds": loop_seconds,
        "restart_step": start_step if args.restart is not None else None,
        "restart_time": (
            float(restart_metadata["scalar_state"]["coupling.current_time"]) if restart_metadata is not None else None
        ),
        "steps": steps,
        "saved_frame_count": len(saved_steps),
        "saved_steps": saved_steps,
        "fem_nodes": int(engine.fem_engine.mesh.number_of_nodes),
        "fem_elements": int(engine.fem_engine.mesh.number_of_cells),
    }
    for filename, payload in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("state.json", state),
        ("performance.json", performance),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"passed": metrics["passed"], "metrics": metrics}, indent=2))
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
