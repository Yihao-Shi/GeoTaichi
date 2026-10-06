#!/usr/bin/env python3
"""Shape-dependent explicit FEM--FEM/FEM--LSDEM collision verification."""

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

from examples.fedem.ShapeCollision.draw.output_audit import collect_native_output
from examples.fedem.ShapeCollision.mesh_reference import (
    place_reference,
    tetrahedralize_primitive_reference,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "OutputData/shape_collision"
# The 12-vertex low-poly mesh is an icosahedron, not a sufficiently resolved
# sphere: point/edge bias excites artificial rotation in nominally coaxial
# sphere--sphere impact.  Use the subdivided icosphere for the verification
# case so that the geometry named in the prompt is also the geometry solved.
SPHERE_OBJ = REPO_ROOT / "assets/mesh/AffineBody/icosphere.obj"
CUBE_OBJ = REPO_ROOT / "assets/mesh/AffineBody/cube.obj"
PROMPT = (
    "Simulate sphere--sphere, sphere--cube, and cube--cube impacts using "
    "both FEM--FEM and FEM--level-set DEM contact. Use eccentric impact "
    "for the nonspherical pairs and report energy and momentum from "
    "approach through complete separation."
)

RADIUS = 0.08
SIDE = 0.14
DENSITY = 1200.0
YOUNG = 2.0e5
POISSON = 0.30
SPEED = 0.5
DT = 5.0e-6
SIMULATION_TIME = 0.16
OUTPUT_INTERVAL = 0.01
JACOBIAN_CHECK_STRIDE = 10
NORMAL_STIFFNESS = 4.0e7
TANGENTIAL_STIFFNESS = 2.0e7
FEM_FEM_NORMAL_STIFFNESS = 2.0e7
FEM_FEM_TANGENTIAL_STIFFNESS = 1.0e7
FEM_FEM_DT = 1.0e-5
FEM_FEM_PT_CAPACITY = 32768
FEM_FEM_EE_CAPACITY = 131072
FEM_FEM_HISTORY_CAPACITY = 262144
MINIMUM_FEM_ELEMENTS = 1000
MINIMUM_CUBE_ELEMENTS = 1000
MINIMUM_TETRA_MEAN_RATIO = 0.05
CASES = (
    ("fem_fem_sphere_sphere", "fem_fem", "sphere", "sphere"),
    ("fem_fem_sphere_cube", "fem_fem", "sphere", "cube"),
    ("fem_fem_cube_cube", "fem_fem", "cube", "cube"),
    ("fem_lsdem_sphere_sphere", "fem_lsdem", "sphere", "sphere"),
    ("fem_lsdem_sphere_cube", "fem_lsdem", "sphere", "cube"),
    ("fem_lsdem_cube_cube", "fem_lsdem", "cube", "cube"),
)
IMPACT_PARAMETERS = {
    "sphere_sphere": 0.0,
    "cube_cube": 0.06,
    "sphere_cube": 0.06,
}


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
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


def _soft_mesh(fem, shape: str, center: np.ndarray):
    del fem
    minimum_element_count = MINIMUM_CUBE_ELEMENTS if shape == "cube" else MINIMUM_FEM_ELEMENTS
    reference, quality = tetrahedralize_primitive_reference(
        shape,
        minimum_element_count=minimum_element_count,
        minimum_mean_ratio=MINIMUM_TETRA_MEAN_RATIO,
    )
    bounding_radius = RADIUS if shape == "sphere" else 0.5 * math.sqrt(3.0) * SIDE
    mesh = place_reference(
        reference,
        center,
        bounding_radius,
        [0.0, 0.0, 0.0],
        name=f"quality_controlled_soft_{shape}",
    )
    return mesh, quality


def _prompt() -> str:
    return PROMPT


def _build_case(
    gt,
    output: Path,
    case_id: str,
    contact_kind: str,
    soft_shape: str,
    rigid_shape: str,
    dt: float,
    fem_fem_normal_stiffness: float,
    fem_fem_tangential_stiffness: float,
    fem_lsdem_normal_stiffness: float,
    fem_lsdem_tangential_stiffness: float,
    fem_fem_verlet_multiplier: float,
    simulation_time: float,
    friction: float,
    normal_viscous_damping: float,
    tangential_viscous_damping: float,
):
    shape_pair_id = f"{soft_shape}_{rigid_shape}"
    impact_parameter = IMPACT_PARAMETERS[shape_pair_id]
    soft_center = np.array([0.405, 0.25 - 0.5 * impact_parameter, 0.25])
    rigid_center = np.array([0.595, 0.25 + 0.5 * impact_parameter, 0.25])
    rigid_scale = None

    dem = gt.DEM(log=False)
    dem.set_configuration(
        domain=[1.0, 0.50, 0.50],
        scheme="LSDEM",
        engine="SymplecticEuler",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        track_energy=True,
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_rigid_body_number": 1,
            # The three-lobed rigid template requires 26,071 SDF cells at the
            # frozen spacing; retain a power-of-two margin for all shapes.
            "levelset_grid_number": 32768,
            "surface_node_number": 256,
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
            "Density": DENSITY,
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    if contact_kind == "fem_lsdem":
        rigid_file = SPHERE_OBJ if rigid_shape == "sphere" else CUBE_OBJ
        raw_vertices, _ = _read_obj(rigid_file)
        rigid_scale = (
            RADIUS / float(np.linalg.norm(raw_vertices - raw_vertices.mean(axis=0), axis=1).mean())
            if rigid_shape == "sphere"
            else SIDE
        )
        template = f"rigid_{rigid_shape}_{case_id}"
        dem.add_template(
            {
                "Name": template,
                "Object": gt.polyhedron(file=str(rigid_file)).grids(space=0.14, extent=2),
                "WriteFile": False,
            }
        )
        dem.create_body(
            {
                "BodyType": "RigidBody",
                "Template": [
                    {
                        "Name": template,
                        "GroupID": 0,
                        "MaterialID": 0,
                        "BodyPoint": rigid_center.tolist(),
                        "ScaleFactor": rigid_scale,
                        "InitialVelocity": [-SPEED, 0.0, 0.0],
                        "InitialAngularVelocity": [0.0, 0.0, 0.0],
                        "FixMotion": ["Free", "Free", "Free"],
                        "BodyOrientation": "constant",
                    }
                ],
            }
        )
    dem.choose_contact_model(None, None)
    dem.select_save_data(particle=True, surface=True)

    fem = gt.FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit", log=False)
    first_mesh, mesh_quality = _soft_mesh(fem, soft_shape, soft_center)
    if contact_kind == "fem_fem":
        from src.fem.generator import FEMMesh

        second_mesh, second_quality = _soft_mesh(fem, rigid_shape, rigid_center)
        mesh = fem.add_soft_particle(FEMMesh.concatenate([first_mesh, second_mesh]))
        mesh_quality = {
            **mesh_quality,
            "passed": bool(mesh_quality["passed"] and second_quality["passed"]),
            "particle_count": 2,
            "element_count_per_particle": [
                int(mesh_quality["element_count"]),
                int(second_quality["element_count"]),
            ],
            "minimum_element_count_per_particle": int(
                min(
                    mesh_quality["element_count"],
                    second_quality["element_count"],
                )
            ),
            "requested_element_count_per_particle": [
                int(mesh_quality["requested_minimum_element_count"]),
                int(second_quality["requested_minimum_element_count"]),
            ],
            "total_element_count": int(mesh_quality["element_count"] + second_quality["element_count"]),
        }
    else:
        mesh = fem.add_soft_particle(first_mesh)
    fem.add_material(
        "NeoHookean",
        density=DENSITY,
        young_modulus=YOUNG,
        poisson_ratio=POISSON,
    )
    initial_velocity = np.zeros_like(mesh.points)
    if contact_kind == "fem_fem":
        initial_velocity[mesh.node_body_ids == 0, 0] = SPEED
        initial_velocity[mesh.node_body_ids == 1, 0] = -SPEED
        fem.add_soft_particle_contact(
            "Linear",
            search="BVH",
            verlet_distance_multiplier=fem_fem_verlet_multiplier,
            ContactThickness=0.0,
            max_point_triangle_pairs=FEM_FEM_PT_CAPACITY,
            max_edge_edge_pairs=FEM_FEM_EE_CAPACITY,
            contact_history_capacity=FEM_FEM_HISTORY_CAPACITY,
            NormalStiffness=fem_fem_normal_stiffness,
            TangentialStiffness=fem_fem_tangential_stiffness,
            Friction=friction,
            NormalViscousDamping=normal_viscous_damping,
            TangentialViscousDamping=tangential_viscous_damping,
        )
    else:
        initial_velocity[:, 0] = SPEED

    coupling = gt.FEDEM(dem=dem, fem=fem, log=False)
    coupling.set_configuration(
        domain=[1.0, 0.50, 0.50],
        gravity=[0.0, 0.0, 0.0],
        search="BVH",
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": dt,
            "SimulationTime": simulation_time,
            "SaveInterval": OUTPUT_INTERVAL,
            "SavePath": str(output / "native" / case_id),
            "initial_velocity": initial_velocity,
            "damping": 0.0,
        },
        log=False,
    )
    coupling.select_save_data(contact=True, checkpoint=True)
    fem_body_ids = [0, 1] if contact_kind == "fem_fem" else [0]
    coupling.add_surface(body_ids=fem_body_ids)
    coupling.memory_allocate(
        {
            "contact_coordination_number": 24,
            "max_contact_pairs": 1024,
            "max_levelset_cell_pairs": 16384,
            "verlet_distance_multiplier": 0.1,
        }
    )
    coupling.choose_contact_model("Linear")
    for fem_body_id in fem_body_ids:
        coupling.add_property(
            DEMmaterial=0,
            FEMbody=fem_body_id,
            property={
                "NormalStiffness": fem_lsdem_normal_stiffness,
                "TangentialStiffness": fem_lsdem_tangential_stiffness,
                "Friction": friction,
                "NormalViscousDamping": normal_viscous_damping,
                "TangentialViscousDamping": tangential_viscous_damping,
            },
        )
    coupling.add_essentials()
    coupling.enginer.pre_calculate()
    coupling.check_critical_timestep()
    return coupling, {
        "contact_kind": contact_kind,
        "shape_pair_id": shape_pair_id,
        "rigid_scale": rigid_scale,
        "impact_parameter": impact_parameter,
        "mesh_quality": mesh_quality,
        "normal_stiffness": (fem_fem_normal_stiffness if contact_kind == "fem_fem" else fem_lsdem_normal_stiffness),
        "tangential_stiffness": (
            fem_fem_tangential_stiffness if contact_kind == "fem_fem" else fem_lsdem_tangential_stiffness
        ),
        "friction": friction,
        "normal_viscous_damping": normal_viscous_damping,
        "tangential_viscous_damping": tangential_viscous_damping,
    }


def _completed_contact_dissipation(coupling, contact_kind):
    """Read dissipation accumulated by steps that have already integrated.

    This helper must be called before ``system_resolve``.  Contact resolution
    evaluates forces and work for the upcoming step, so reading afterwards
    advances the cumulative ledger one step ahead of the current state.
    """
    if contact_kind == "fem_fem":
        manager = coupling.enginer.fem_engine.soft_particle_contact
        energy = manager.energy_diagnostics()
    else:
        energy = coupling.contactor.energy_diagnostics()
    return {
        "friction_dissipation": float(energy["friction_dissipation"]),
        "damping_dissipation": float(energy["damping_dissipation"]),
    }


def _contact_state(coupling, contact_kind, normal_stiffness):
    if contact_kind == "fem_fem":
        manager = coupling.enginer.fem_engine.soft_particle_contact
        diagnostics = manager.diagnostics()
        pt_count = int(diagnostics["point_triangle_candidates"])
        ee_count = int(diagnostics["edge_edge_candidates"])
        pt_active = manager.pt_active.to_numpy()[:pt_count].astype(bool)
        ee_active = manager.ee_active.to_numpy()[:ee_count].astype(bool)
        pt_normal = manager.pt_normal_force.to_numpy()[:pt_count]
        ee_normal = manager.ee_normal_force.to_numpy()[:ee_count]
        pt_tangential = manager.pt_tangential_force.to_numpy()[:pt_count]
        ee_tangential = manager.ee_tangential_force.to_numpy()[:ee_count]
        node_area = manager.node_area.to_numpy()
        point_nodes = manager.culling.point_triangle.to_numpy()[:pt_count, 0]
        edge_measure = manager.culling.edge_edge_measure.to_numpy()[:ee_count]
        active_overlaps = []
        if np.any(pt_active):
            pt_measure = 0.5 * node_area[point_nodes[pt_active]]
            active_overlaps.extend(
                (
                    np.linalg.norm(pt_normal[pt_active], axis=1) / np.maximum(pt_measure * normal_stiffness, 1.0e-30)
                ).tolist()
            )
        if np.any(ee_active):
            active_overlaps.extend(
                (
                    np.linalg.norm(ee_normal[ee_active], axis=1)
                    / np.maximum(
                        edge_measure[ee_active] * normal_stiffness,
                        1.0e-30,
                    )
                ).tolist()
            )
        node_body = manager.node_body.to_numpy()
        external_force = coupling.enginer.fem_engine.state.external_force.to_numpy()
        first_force = external_force[node_body == 0].sum(axis=0)
        second_force = external_force[node_body == 1].sum(axis=0)
        scale = max(
            float(np.linalg.norm(first_force)),
            float(np.linalg.norm(second_force)),
            1.0e-30,
        )
        energy = manager.energy_diagnostics()
        finite = all(np.isfinite(values).all() for values in (pt_normal, ee_normal, pt_tangential, ee_tangential))
        return {
            "candidate_count": pt_count + ee_count,
            "active_count": int(np.sum(pt_active) + np.sum(ee_active)),
            "fem_force": first_force,
            "rigid_force": second_force,
            "action_reaction": float(np.linalg.norm(first_force + second_force) / scale),
            "contact_energy": float(energy["elastic_energy"]),
            "friction_dissipation": float(energy["friction_dissipation"]),
            "damping_dissipation": float(energy["damping_dissipation"]),
            "maximum_penetration": float(max(active_overlaps, default=0.0)),
            "finite": bool(finite),
        }

    neighbor = coupling.contactor.neighbor
    count = int(neighbor.contact_count)
    active = neighbor.contacts.active.to_numpy()[:count].astype(bool)
    node_ids = neighbor.contacts.node_id.to_numpy()[:count]
    normal = neighbor.contacts.normal_force.to_numpy()[:count]
    tangential = neighbor.contacts.tangential_force.to_numpy()[:count]
    fem_force = normal[active].sum(axis=0) + tangential[active].sum(axis=0)
    rigid_force = coupling.dem.scene.rigid.contact_force.to_numpy()[0]
    scale = max(float(np.linalg.norm(fem_force)), float(np.linalg.norm(rigid_force)), 1.0e-30)
    areas = coupling.patch.node_area.to_numpy()
    maximum_penetration = 0.0
    if np.any(active):
        magnitudes = np.linalg.norm(normal[active], axis=1)
        stiffness = normal_stiffness * areas[node_ids[active]]
        maximum_penetration = float(np.max(magnitudes / np.maximum(stiffness, 1.0e-30)))
    energy = coupling.contactor.energy_diagnostics()
    return {
        "candidate_count": count,
        "active_count": int(np.sum(active)),
        "fem_force": fem_force,
        "rigid_force": rigid_force,
        "action_reaction": float(np.linalg.norm(fem_force + rigid_force) / scale),
        "contact_energy": float(energy["elastic_energy"]),
        "friction_dissipation": float(energy["friction_dissipation"]),
        "damping_dissipation": float(energy["damping_dissipation"]),
        "maximum_penetration": maximum_penetration,
        "finite": bool(np.isfinite(normal).all() and np.isfinite(tangential).all()),
    }


def _fem_body_kinematics(positions, velocity, mass, body_ids):
    bodies = []
    for body_id in np.unique(body_ids):
        selected = body_ids == body_id
        body_mass = mass[selected]
        body_position = positions[selected]
        body_velocity = velocity[selected]
        total_mass = float(np.sum(body_mass))
        center = np.sum(body_mass[:, None] * body_position, axis=0) / total_mass
        center_velocity = np.sum(body_mass[:, None] * body_velocity, axis=0) / total_mass
        relative_position = body_position - center
        relative_velocity = body_velocity - center_velocity
        angular_momentum = np.sum(
            body_mass[:, None] * np.cross(relative_position, relative_velocity),
            axis=0,
        )
        inertia = np.zeros((3, 3), dtype=np.float64)
        for node_mass, radius_vector in zip(body_mass, relative_position):
            inertia += node_mass * (
                np.dot(radius_vector, radius_vector) * np.eye(3) - np.outer(radius_vector, radius_vector)
            )
        angular_velocity = np.linalg.pinv(inertia, rcond=1.0e-12) @ angular_momentum
        total_kinetic = float(0.5 * np.sum(body_mass[:, None] * body_velocity * body_velocity))
        translational = float(0.5 * total_mass * np.dot(center_velocity, center_velocity))
        rotational = float(0.5 * np.dot(angular_velocity, angular_momentum))
        bodies.append(
            {
                "center": center,
                "center_velocity": center_velocity,
                "angular_momentum": angular_momentum,
                "angular_velocity": angular_velocity,
                "translational": translational,
                "rotational": rotational,
                "deformation": max(total_kinetic - translational - rotational, 0.0),
            }
        )
    return bodies


def _observe(
    coupling,
    time_value: float,
    step: int,
    contact,
    internal_force=None,
    strain_energy=None,
    previous_potential=None,
    contact_kind="fem_lsdem",
    completed_dissipation=None,
):
    fem_engine = coupling.enginer.fem_engine
    positions = fem_engine.state.position.to_numpy()
    velocity = fem_engine.state.velocity.to_numpy()
    mass = fem_engine.state.mass.to_numpy()
    if internal_force is None:
        internal_force = fem_engine._assemble_internal_device(need_stiffness=False)
    fem_kinetic = float(0.5 * np.sum(mass[:, None] * velocity * velocity))
    instantaneous_strain = (
        float(fem_engine._internal_energy_device()) if strain_energy is None else float(strain_energy)
    )
    body_ids = fem_engine.mesh.node_body_ids
    fem_bodies = _fem_body_kinematics(positions, velocity, mass, body_ids)
    center = fem_bodies[0]["center"]
    center_velocity = fem_bodies[0]["center_velocity"]
    fem_linear_kinetic = sum(body["translational"] for body in fem_bodies)
    fem_rotational_kinetic = sum(body["rotational"] for body in fem_bodies)
    fem_deformation_kinetic = sum(body["deformation"] for body in fem_bodies)
    fem_angular_speed = max(float(np.linalg.norm(body["angular_velocity"])) for body in fem_bodies)
    if contact_kind == "fem_fem":
        rigid_mass = 0.0
        rigid_velocity = fem_bodies[1]["center_velocity"]
        rigid_omega = np.zeros(3, dtype=np.float64)
        rigid_angular_momentum = np.zeros(3, dtype=np.float64)
        rigid_translational = 0.0
        rigid_rotational = 0.0
        rigid_center = fem_bodies[1]["center"]
    else:
        rigid = coupling.dem.scene.rigid
        rigid_mass = float(rigid.m.to_numpy()[0])
        rigid_velocity = rigid.v.to_numpy()[0]
        rigid_omega = rigid.w.to_numpy()[0]
        rigid_angular_momentum = rigid.angmoment.to_numpy()[0]
        rigid_translational = float(0.5 * rigid_mass * np.dot(rigid_velocity, rigid_velocity))
        rigid_rotational = float(0.5 * np.dot(rigid_omega, rigid_angular_momentum))
        rigid_center = rigid.mass_center.to_numpy()[0]
    translational_kinetic = fem_linear_kinetic + rigid_translational
    rotational_kinetic = fem_rotational_kinetic + rigid_rotational
    instantaneous_contact = float(contact["contact_energy"])
    raw_time_level_total = (
        fem_kinetic + instantaneous_strain + rigid_translational + rigid_rotational + instantaneous_contact
    )
    # Symplectic Euler is the staggered central-difference recurrence: the
    # stored velocity at q_n is v_{n-1/2}.  Pair that kinetic energy with the
    # trapezoidal average of U(q_{n-1}) and U(q_n), rather than with U(q_n)
    # alone.  This changes only the diagnostic ledger, never the trajectory.
    if previous_potential is None:
        strain = instantaneous_strain
        contact_energy = instantaneous_contact
    else:
        strain = 0.5 * (float(previous_potential["strain_energy"]) + instantaneous_strain)
        contact_energy = 0.5 * (float(previous_potential["contact_energy"]) + instantaneous_contact)
    total = fem_kinetic + strain + rigid_translational + rigid_rotational + contact_energy
    completed_dissipation = completed_dissipation or {
        "friction_dissipation": 0.0,
        "damping_dissipation": 0.0,
    }
    friction_dissipation = float(completed_dissipation["friction_dissipation"])
    damping_dissipation = float(completed_dissipation["damping_dissipation"])
    accounted_total = total + friction_dissipation + damping_dissipation
    raw_accounted_total = raw_time_level_total + friction_dissipation + damping_dissipation
    jacobian = float(fem_engine._minimum_jacobian_ratio_device(fem_engine.state.position))
    total_momentum = np.sum(mass[:, None] * velocity, axis=0) + rigid_mass * rigid_velocity
    total_angular_momentum = (
        np.sum(np.cross(positions, mass[:, None] * velocity), axis=0)
        + np.cross(rigid_center, rigid_mass * rigid_velocity)
        + rigid_angular_momentum
    )
    return {
        "step": step,
        "time": time_value,
        "candidate_count": contact["candidate_count"],
        "active_contact_count": contact["active_count"],
        "contact_force": float(np.linalg.norm(contact["fem_force"])),
        "maximum_penetration": contact["maximum_penetration"],
        "action_reaction_relative": contact["action_reaction"],
        "soft_center_x": float(center[0]),
        "soft_center_y": float(center[1]),
        "rigid_center_x": float(rigid_center[0]),
        "rigid_center_y": float(rigid_center[1]),
        "soft_velocity_x": float(center_velocity[0]),
        "rigid_velocity_x": float(rigid_velocity[0]),
        "momentum_x": float(total_momentum[0]),
        "momentum_y": float(total_momentum[1]),
        "momentum_z": float(total_momentum[2]),
        "angular_momentum_x": float(total_angular_momentum[0]),
        "angular_momentum_y": float(total_angular_momentum[1]),
        "angular_momentum_z": float(total_angular_momentum[2]),
        "fem_kinetic_energy": fem_kinetic,
        "fem_translational_energy": fem_linear_kinetic,
        "fem_rotational_energy": fem_rotational_kinetic,
        "fem_deformation_kinetic_energy": fem_deformation_kinetic,
        "fem_angular_speed": fem_angular_speed,
        "fem_strain_energy": strain,
        "instantaneous_fem_strain_energy": instantaneous_strain,
        "rigid_translational_energy": rigid_translational,
        "rigid_rotational_energy": rigid_rotational,
        "rigid_angular_speed": float(np.linalg.norm(rigid_omega)),
        "translational_kinetic_energy": translational_kinetic,
        "rotational_kinetic_energy": rotational_kinetic,
        "contact_energy": contact_energy,
        "instantaneous_contact_energy": instantaneous_contact,
        "total_energy": total,
        "raw_time_level_total_energy": raw_time_level_total,
        "friction_dissipation": friction_dissipation,
        "damping_dissipation": damping_dissipation,
        "accounted_total_energy": accounted_total,
        "raw_time_level_accounted_total_energy": raw_accounted_total,
        "minimum_jacobian": jacobian,
    }


def _run_case(
    gt,
    ti,
    output,
    case_id,
    contact_kind,
    soft_shape,
    rigid_shape,
    dt,
    fem_fem_normal_stiffness,
    fem_fem_tangential_stiffness,
    fem_lsdem_normal_stiffness,
    fem_lsdem_tangential_stiffness,
    fem_fem_verlet_multiplier,
    simulation_time,
    jacobian_check_stride,
    friction,
    normal_viscous_damping,
    tangential_viscous_damping,
):
    started = time.perf_counter()
    coupling, case_metadata = _build_case(
        gt,
        output,
        case_id,
        contact_kind,
        soft_shape,
        rigid_shape,
        dt,
        fem_fem_normal_stiffness,
        fem_fem_tangential_stiffness,
        fem_lsdem_normal_stiffness,
        fem_lsdem_tangential_stiffness,
        fem_fem_verlet_multiplier,
        simulation_time,
        friction,
        normal_viscous_damping,
        tangential_viscous_damping,
    )
    contact_kind = case_metadata["contact_kind"]
    rigid_scale = case_metadata["rigid_scale"]
    impact_parameter = case_metadata["impact_parameter"]
    mesh_quality = case_metadata["mesh_quality"]
    normal_stiffness = case_metadata["normal_stiffness"]
    tangential_stiffness = case_metadata["tangential_stiffness"]
    ti.sync()
    setup_seconds = time.perf_counter() - started
    engine = coupling.enginer
    effective_dt = float(coupling.sims.delta)
    steps = int(round(simulation_time / effective_dt))
    sample_stride = max(1, int(round(2.0e-4 / effective_dt)))
    progress_stride = max(1, steps // 10)
    output_stride = max(1, int(round(OUTPUT_INTERVAL / effective_dt)))
    records = []
    peak_force = 0.0
    peak_active = 0
    peak_candidates = 0
    maximum_action_reaction = 0.0
    minimum_jacobian = 1.0
    finite = True
    contact_started = None
    contact_ended = None
    last_contact_active_time = None
    initial_energy = None
    initial_momentum = None
    initial_angular_momentum = None
    momentum_scale = None
    angular_momentum_scale = None
    maximum_energy_residual = 0.0
    maximum_raw_time_level_energy_residual = 0.0
    maximum_accounted_energy_residual = 0.0
    maximum_raw_time_level_accounted_energy_residual = 0.0
    peak_rotational_energy = 0.0
    peak_fem_deformation_kinetic = 0.0
    maximum_penetration = 0.0
    saved_steps = []
    preceding_potential = None

    loop_started = time.perf_counter()
    for step in range(steps + 1):
        time_value = min(step * effective_dt, simulation_time)
        completed_dissipation = _completed_contact_dissipation(coupling, contact_kind)
        engine.reset_message()
        engine.update_verlet_tables()
        engine.system_resolve()
        sample_now = step % sample_stride == 0 or step == steps
        prepares_sample = step < steps and ((step + 1) % sample_stride == 0 or step + 1 == steps)
        sampled_internal_force = None
        diagnostic_internal_force = None
        contact = None
        strain_energy = None
        if sample_now or prepares_sample:
            contact = _contact_state(coupling, contact_kind, normal_stiffness)
            # The explicit hot path intentionally assembles force only, so
            # ClassicalAssembler.total_energy is a sampled diagnostic rather
            # than a per-step quantity.  Refresh it at the current state
            # before reading it; otherwise the energy ledger uses the value
            # left by the preceding sample while the trajectory remains
            # current.
            diagnostic_internal_force = engine.fem_engine._assemble_internal_device(need_stiffness=False)
            strain_energy = float(engine.fem_engine._internal_energy_device())
        if sample_now:
            peak_active = max(peak_active, contact["active_count"])
            peak_candidates = max(peak_candidates, contact["candidate_count"])
            peak_force = max(peak_force, float(np.linalg.norm(contact["fem_force"])))
            maximum_penetration = max(
                maximum_penetration,
                float(contact["maximum_penetration"]),
            )
            maximum_action_reaction = max(maximum_action_reaction, contact["action_reaction"])
            finite = finite and contact["finite"]
            contact_is_active = contact["active_count"] > 0
            if contact_is_active and contact_started is None:
                contact_started = time_value
            if contact_is_active:
                last_contact_active_time = time_value
            sampled_internal_force = diagnostic_internal_force
            observation = _observe(
                coupling,
                time_value,
                step,
                contact,
                internal_force=sampled_internal_force,
                strain_energy=strain_energy,
                previous_potential=preceding_potential,
                contact_kind=contact_kind,
                completed_dissipation=completed_dissipation,
            )
            if initial_energy is None:
                initial_energy = observation["total_energy"]
                initial_momentum = np.array(
                    [observation["momentum_x"], observation["momentum_y"], observation["momentum_z"]]
                )
                initial_angular_momentum = np.array(
                    [
                        observation["angular_momentum_x"],
                        observation["angular_momentum_y"],
                        observation["angular_momentum_z"],
                    ]
                )
                fem_mass = engine.fem_engine.state.mass.to_numpy()
                rigid_mass = 0.0 if contact_kind == "fem_fem" else float(coupling.dem.scene.rigid.m.to_numpy()[0])
                momentum_scale = float(np.sum(fem_mass) * SPEED + rigid_mass * SPEED)
                angular_momentum_scale = momentum_scale * RADIUS
            observation["energy_residual"] = (observation["total_energy"] - initial_energy) / max(
                abs(initial_energy), 1.0e-30
            )
            observation["raw_time_level_energy_residual"] = (
                observation["raw_time_level_total_energy"] - initial_energy
            ) / max(abs(initial_energy), 1.0e-30)
            observation["accounted_energy_residual"] = (observation["accounted_total_energy"] - initial_energy) / max(
                abs(initial_energy), 1.0e-30
            )
            observation["raw_time_level_accounted_energy_residual"] = (
                observation["raw_time_level_accounted_total_energy"] - initial_energy
            ) / max(abs(initial_energy), 1.0e-30)
            maximum_energy_residual = max(maximum_energy_residual, abs(observation["energy_residual"]))
            maximum_raw_time_level_energy_residual = max(
                maximum_raw_time_level_energy_residual,
                abs(observation["raw_time_level_energy_residual"]),
            )
            maximum_accounted_energy_residual = max(
                maximum_accounted_energy_residual,
                abs(observation["accounted_energy_residual"]),
            )
            maximum_raw_time_level_accounted_energy_residual = max(
                maximum_raw_time_level_accounted_energy_residual,
                abs(observation["raw_time_level_accounted_energy_residual"]),
            )
            peak_rotational_energy = max(peak_rotational_energy, observation["rotational_kinetic_energy"])
            peak_fem_deformation_kinetic = max(
                peak_fem_deformation_kinetic,
                observation["fem_deformation_kinetic_energy"],
            )
            minimum_jacobian = min(minimum_jacobian, observation["minimum_jacobian"])
            records.append(observation)
        preceding_potential = (
            {
                "strain_energy": strain_energy,
                "contact_energy": contact["contact_energy"],
            }
            if prepares_sample
            else None
        )
        if (step == 0 or step % output_stride == 0 or step == steps) and (not saved_steps or saved_steps[-1] != step):
            coupling.save_data()
            saved_steps.append(step)
        if step == steps:
            break
        if step > 0 and step % progress_stride == 0:
            print(json.dumps({"case": case_id, "progress": step / steps}), flush=True)
        check_jacobian = (
            (step + 1) % jacobian_check_stride == 0
            or (step + 1) % sample_stride == 0
            or (step + 1) % output_stride == 0
            or step + 1 == steps
        )
        engine.integration(
            internal_force=sampled_internal_force,
            update_diagnostics=False,
            check_jacobian=check_jacobian,
        )
        coupling.sims.current_time += effective_dt
        coupling.dem.sims.current_time += effective_dt
        engine.fem_engine.time = coupling.sims.current_time
        coupling.sims.current_step += 1
        coupling.dem.sims.current_step += 1
        engine.fem_engine.step_count += 1
        minimum_jacobian = min(minimum_jacobian, float(engine.minimum_jacobian))
    ti.sync()
    loop_seconds = time.perf_counter() - loop_started
    output_evidence = collect_native_output(output / "native" / case_id, len(saved_steps))
    if contact_kind == "fem_fem":
        output_evidence["complete"] = all(
            int(output_evidence[name]) >= len(saved_steps)
            for name in (
                "fem_vtu_count",
                "coupled_contact_npz_count",
                "checkpoint_npz_count",
            )
        )
        output_evidence["required_for_case"] = [
            "fem_vtu_count",
            "coupled_contact_npz_count",
            "checkpoint_npz_count",
        ]
    final = records[-1]
    if last_contact_active_time is not None and int(final["active_contact_count"]) == 0:
        contact_ended = min(
            last_contact_active_time + sample_stride * effective_dt,
            simulation_time,
        )
    final_momentum = np.array([final["momentum_x"], final["momentum_y"], final["momentum_z"]])
    momentum_residual = float(np.linalg.norm(final_momentum - initial_momentum) / max(momentum_scale, 1.0e-30))
    final_angular_momentum = np.array(
        [
            final["angular_momentum_x"],
            final["angular_momentum_y"],
            final["angular_momentum_z"],
        ]
    )
    angular_momentum_residual = float(
        np.linalg.norm(final_angular_momentum - initial_angular_momentum) / max(angular_momentum_scale, 1.0e-30)
    )
    summary = {
        "id": case_id,
        "contact_kind": contact_kind,
        "first_shape": soft_shape,
        "second_shape": rigid_shape,
        # Retain the old keys for downstream readers while making the two-FEM
        # cases unambiguous through contact_kind and first/second_shape.
        "soft_shape": soft_shape,
        "rigid_shape": rigid_shape,
        "rigid_scale_factor": rigid_scale,
        "impact_parameter": impact_parameter,
        "requested_dt": dt,
        "normal_stiffness": normal_stiffness,
        "tangential_stiffness": tangential_stiffness,
        "effective_dt": effective_dt,
        "steps": steps,
        "contact_started": contact_started,
        "contact_ended": contact_ended,
        "contact_duration": (
            None if contact_started is None or contact_ended is None else contact_ended - contact_started
        ),
        "peak_contact_force": peak_force,
        "maximum_penetration": maximum_penetration,
        "maximum_penetration_fraction": maximum_penetration / RADIUS,
        "maximum_active_contact_count": peak_active,
        "maximum_candidate_count": peak_candidates,
        "maximum_action_reaction_relative": maximum_action_reaction,
        "maximum_energy_residual": maximum_energy_residual,
        "maximum_raw_time_level_energy_residual": (maximum_raw_time_level_energy_residual),
        "maximum_accounted_energy_residual": (maximum_accounted_energy_residual),
        "maximum_raw_time_level_accounted_energy_residual": (maximum_raw_time_level_accounted_energy_residual),
        "final_accounted_energy_residual": final["accounted_energy_residual"],
        "final_raw_time_level_accounted_energy_residual": final["raw_time_level_accounted_energy_residual"],
        "final_friction_dissipation": final["friction_dissipation"],
        "final_damping_dissipation": final["damping_dissipation"],
        "peak_rotational_energy": peak_rotational_energy,
        "peak_rotational_energy_fraction": peak_rotational_energy / max(abs(initial_energy), 1.0e-30),
        "peak_fem_deformation_kinetic_energy": peak_fem_deformation_kinetic,
        "maximum_rigid_angular_speed": max(record["rigid_angular_speed"] for record in records),
        "maximum_fem_angular_speed": max(record["fem_angular_speed"] for record in records),
        "momentum_residual": momentum_residual,
        "angular_momentum_residual": angular_momentum_residual,
        "minimum_jacobian": minimum_jacobian,
        "final_active_contact_count": int(final["active_contact_count"]),
        "finite_state": finite,
        "fem_mesh_quality": mesh_quality,
    }
    gates = {
        "requested_timestep": abs(effective_dt - dt) <= 1.0e-12 * max(abs(dt), 1.0),
        "finite_state": finite,
        "contact_occurred": peak_active > 0,
        "final_separation": int(final["active_contact_count"]) == 0,
        "post_contact_interval": (contact_ended is not None and simulation_time - contact_ended >= 0.01),
        "positive_jacobian": minimum_jacobian > 0.10,
        "action_reaction": maximum_action_reaction <= 1.0e-10,
        "momentum": momentum_residual <= 1.0e-6,
        # The residual is normalized by the deliberately small orbital scale
        # pR.  A 5e-5 gate remains two orders tighter than the energy audit
        # while accommodating explicit force integration in eccentric impact.
        "angular_momentum": angular_momentum_residual <= 5.0e-5,
        "raw_energy": (
            maximum_raw_time_level_energy_residual <= 0.01
            if friction == 0.0 and normal_viscous_damping == 0.0 and tangential_viscous_damping == 0.0
            else True
        ),
        "staggered_energy": (
            maximum_energy_residual <= 0.01
            if friction == 0.0 and normal_viscous_damping == 0.0 and tangential_viscous_damping == 0.0
            else True
        ),
        "accounted_energy": maximum_accounted_energy_residual <= 0.02,
        "penetration": maximum_penetration <= 0.05 * RADIUS,
        "capacity": peak_candidates
        <= (FEM_FEM_PT_CAPACITY + FEM_FEM_EE_CAPACITY if contact_kind == "fem_fem" else 1024),
        "native_output": bool(output_evidence["complete"]),
    }
    if impact_parameter > 0.0:
        gates["rotation_excited"] = peak_rotational_energy / max(abs(initial_energy), 1.0e-30) > 1.0e-4
    performance = {
        "id": case_id,
        "setup_seconds": setup_seconds,
        "simulation_loop_seconds": loop_seconds,
        "fem_nodes": int(engine.fem_engine.mesh.number_of_nodes),
        "fem_elements": int(engine.fem_engine.mesh.number_of_cells),
        "fem_dofs": int(3 * engine.fem_engine.mesh.number_of_nodes),
        "fem_mesh_quality": mesh_quality,
        "saved_frame_count": len(saved_steps),
        "saved_steps": saved_steps,
        "native_output": output_evidence,
    }
    return summary, gates, records, performance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--default-fp", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--fem-fem-dt", type=float, default=FEM_FEM_DT)
    parser.add_argument("--simulation-time", type=float, default=SIMULATION_TIME)
    parser.add_argument(
        "--jacobian-check-stride",
        type=int,
        default=JACOBIAN_CHECK_STRIDE,
        help=("Check the minimum FEM Jacobian every N steps and at every " "sample, output, and final state."),
    )
    parser.add_argument("--fem-fem-verlet-distance-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--fem-fem-normal-stiffness",
        type=float,
        default=FEM_FEM_NORMAL_STIFFNESS,
    )
    parser.add_argument(
        "--fem-fem-tangential-stiffness",
        type=float,
        default=FEM_FEM_TANGENTIAL_STIFFNESS,
    )
    parser.add_argument(
        "--fem-lsdem-normal-stiffness",
        type=float,
        default=NORMAL_STIFFNESS,
    )
    parser.add_argument(
        "--fem-lsdem-tangential-stiffness",
        type=float,
        default=TANGENTIAL_STIFFNESS,
    )
    parser.add_argument("--friction", type=float, default=0.0)
    parser.add_argument("--normal-viscous-damping", type=float, default=0.0)
    parser.add_argument("--tangential-viscous-damping", type=float, default=0.0)
    parser.add_argument("--only", choices=tuple(case[0] for case in CASES))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.dt,
            args.fem_fem_dt,
            args.fem_fem_normal_stiffness,
            args.fem_fem_tangential_stiffness,
            args.fem_lsdem_normal_stiffness,
            args.fem_lsdem_tangential_stiffness,
            args.simulation_time,
            args.fem_fem_verlet_distance_multiplier,
        )
        <= 0.0
    ):
        parser.error(
            "time steps, simulation time, contact stiffnesses, and the " "FEM--FEM Verlet multiplier must be positive"
        )
    if args.jacobian_check_stride <= 0:
        parser.error("--jacobian-check-stride must be positive")
    if (
        min(
            args.friction,
            args.normal_viscous_damping,
            args.tangential_viscous_damping,
        )
        < 0.0
    ):
        parser.error("friction and viscous damping parameters must be non-negative")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["GEOTAICHI_REAL_DTYPE"] = args.default_fp

    import taichi as ti
    import geotaichi as gt

    gt.init(
        arch=args.arch,
        default_fp=args.default_fp,
        log=False,
        debug=False,
        offline_cache=False,
    )
    selected = [case for case in CASES if args.only is None or case[0] == args.only]
    if args.preflight:
        preflights = []
        for case_id, contact_kind, soft_shape, rigid_shape in selected:
            case_dt = args.fem_fem_dt if contact_kind == "fem_fem" else args.dt
            coupling, case_metadata = _build_case(
                gt,
                output,
                case_id,
                contact_kind,
                soft_shape,
                rigid_shape,
                case_dt,
                args.fem_fem_normal_stiffness,
                args.fem_fem_tangential_stiffness,
                args.fem_lsdem_normal_stiffness,
                args.fem_lsdem_tangential_stiffness,
                args.fem_fem_verlet_distance_multiplier,
                args.simulation_time,
                args.friction,
                args.normal_viscous_damping,
                args.tangential_viscous_damping,
            )
            contact_kind = case_metadata["contact_kind"]
            mesh_quality = case_metadata["mesh_quality"]
            engine = coupling.enginer
            engine.reset_message()
            engine.update_verlet_tables()
            engine.system_resolve()
            engine.integration(update_diagnostics=False)
            ti.sync()
            positions = engine.fem_engine.state.position.to_numpy()
            velocity = engine.fem_engine.state.velocity.to_numpy()
            secondary_state = (
                positions[engine.fem_engine.mesh.node_body_ids == 1]
                if contact_kind == "fem_fem"
                else coupling.dem.scene.rigid.mass_center.to_numpy()
            )
            record = {
                "case": case_id,
                "passed": bool(
                    mesh_quality["passed"]
                    and mesh_quality.get(
                        "minimum_element_count_per_particle",
                        mesh_quality["element_count"],
                    )
                    >= MINIMUM_FEM_ELEMENTS
                    and abs(float(coupling.sims.delta) - case_dt) <= 1.0e-12 * max(abs(case_dt), 1.0)
                    and np.isfinite(positions).all()
                    and np.isfinite(velocity).all()
                    and np.isfinite(secondary_state).all()
                    and float(engine.minimum_jacobian) > 0.0
                ),
                "mesh_quality": mesh_quality,
                "requested_dt": case_dt,
                "effective_dt": float(coupling.sims.delta),
                "one_step_minimum_jacobian": float(engine.minimum_jacobian),
                "one_step_finite": bool(
                    np.isfinite(positions).all() and np.isfinite(velocity).all() and np.isfinite(secondary_state).all()
                ),
                "cross_candidate_count": int(
                    0 if coupling.contactor.neighbor is None else coupling.contactor.neighbor.contact_count
                ),
                "soft_candidate_count": (
                    sum(
                        coupling.enginer.fem_engine.soft_particle_contact.diagnostics()[name]
                        for name in (
                            "point_triangle_candidates",
                            "edge_edge_candidates",
                        )
                    )
                    if contact_kind == "fem_fem"
                    else 0
                ),
                "contact_kind": contact_kind,
                "contact_work_mode": coupling.sims.contact_work_mode,
            }
            preflights.append(record)
        payload = {
            "schema_version": 1,
            "passed": all(item["passed"] for item in preflights),
            "cases": preflights,
        }
        (output / "preflight.json").write_text(
            json.dumps(payload, indent=2) + os.linesep,
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2), flush=True)
        return 0 if payload["passed"] else 2
    summaries, case_gates, performance, history = [], {}, [], []
    for case_id, contact_kind, soft_shape, rigid_shape in selected:
        case_dt = args.fem_fem_dt if contact_kind == "fem_fem" else args.dt
        summary, gates, records, timing = _run_case(
            gt,
            ti,
            output,
            case_id,
            contact_kind,
            soft_shape,
            rigid_shape,
            case_dt,
            args.fem_fem_normal_stiffness,
            args.fem_fem_tangential_stiffness,
            args.fem_lsdem_normal_stiffness,
            args.fem_lsdem_tangential_stiffness,
            args.fem_fem_verlet_distance_multiplier,
            args.simulation_time,
            args.jacobian_check_stride,
            args.friction,
            args.normal_viscous_damping,
            args.tangential_viscous_damping,
        )
        summaries.append(summary)
        case_gates[case_id] = gates
        performance.append(timing)
        history.extend({"case": case_id, **record} for record in records)
        case_root = output / "cases" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        case_payload = {
            "schema_version": 1,
            "passed": bool(all(gates.values())),
            "summary": summary,
            "gates": gates,
            "performance": timing,
        }
        (case_root / "metrics.json").write_text(
            json.dumps(case_payload, indent=2) + os.linesep,
            encoding="utf-8",
        )
        if records:
            with (case_root / "history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)

    metrics = {
        "schema_version": 1,
        "passed": all(all(gates.values()) for gates in case_gates.values()),
        "case_gates": case_gates,
        "summaries": summaries,
    }
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
        },
        "cases": [case[0] for case in selected],
        "parameters": {
            "characteristic_radius": RADIUS,
            "cube_side": SIDE,
            "density": DENSITY,
            "young_modulus": YOUNG,
            "poisson_ratio": POISSON,
            "speed_each_body": SPEED,
            "impact_parameters": IMPACT_PARAMETERS,
            "minimum_fem_elements_per_particle": {
                "sphere": MINIMUM_FEM_ELEMENTS,
                "cube": MINIMUM_CUBE_ELEMENTS,
            },
            "minimum_tetra_mean_ratio": MINIMUM_TETRA_MEAN_RATIO,
            "requested_dt": args.dt,
            "fem_fem_requested_dt": args.fem_fem_dt,
            "simulation_time": args.simulation_time,
            "fem_fem_verlet_distance_multiplier": (args.fem_fem_verlet_distance_multiplier),
            "output_interval": OUTPUT_INTERVAL,
            "checkpoint_output": True,
            "normal_stiffness": args.fem_lsdem_normal_stiffness,
            "tangential_stiffness": args.fem_lsdem_tangential_stiffness,
            "fem_fem_normal_stiffness": args.fem_fem_normal_stiffness,
            "fem_fem_tangential_stiffness": (args.fem_fem_tangential_stiffness),
            "friction": args.friction,
            "normal_viscous_damping": args.normal_viscous_damping,
            "tangential_viscous_damping": args.tangential_viscous_damping,
            "search": "BVH",
            "precision": args.default_fp,
            "contact_work_mode": "Explicit",
            "energy_accounting": (
                "staggered half-step kinetic energy plus the trapezoidal "
                "average of adjacent strain and contact potentials"
            ),
        },
    }
    timing_payload = {
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
        "cases": performance,
    }
    for filename, payload in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("performance.json", timing_payload),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")
    if history:
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    print(json.dumps({"passed": metrics["passed"], "summaries": summaries}, indent=2))
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
