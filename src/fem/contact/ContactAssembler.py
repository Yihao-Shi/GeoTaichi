"""Taichi PT/EE contact assembly for implicit FEM."""

import numpy as np
import taichi as ti

from src.fem.engines.SparseMatrix import FEMTripletContribution
from src.fem.contact.LinkedCellBroadPhase import DynamicLinkedCellBroadPhase
from src.fem.contact.BVHBroadPhase import DynamicBVHBroadPhase
from src.fem.contact.CollisionCulling import FEMCollisionCulling
from src.fem.contact.ContactTopology import build_contact_surface
from src.physics_model.contact_model.ipc.ContactAssembly import (
    psd_project_nd,
    psd_project_gershgorin_nd,
)
from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance_grad,
    edge_edge_distance_grad_hess,
    edge_edge_distance_grad_hess_by_type,
    edge_edge_distance_type,
    point_triangle_distance_grad,
    point_triangle_distance_grad_hess,
    point_triangle_distance_grad_hess_by_type,
    point_triangle_distance_type,
)
from src.physics_model.contact_model.ipc.ContactMollifier import (
    edge_edge_mollifier,
    edge_edge_mollifier_grad,
    edge_edge_mollifier_terms,
    edge_edge_mollifier_threshold,
)
from src.physics_model.contact_model.ipc.IPC import ipc_toolkit_barrier_distance2_offset_terms
from src.physics_model.contact_model.ipc.IPC import (
    ipc_friction_f0,
    ipc_friction_f1_over_speed,
    ipc_friction_hessian_term,
    semi_ipc_find_or_insert,
    semi_ipc_terms,
    semi_ipc_update_multiplier_py,
    semi_ipc_update_multiplier,
)


def _ipc_barrier_terms_numpy(value, active_value, kappa):
    if value <= 0.0:
        return np.inf, 0.0, 0.0
    if value >= active_value:
        return 0.0, 0.0, 0.0
    difference = value - active_value
    inverse_active2 = 1.0 / (active_value * active_value)
    logarithm = np.log(value / active_value)
    energy = -kappa * difference * difference * inverse_active2 * logarithm
    gradient = -kappa * inverse_active2 * (2.0 * difference * logarithm + difference * difference / value)
    hessian = (
        -kappa
        * inverse_active2
        * (2.0 * logarithm + 4.0 * difference / value - difference * difference / (value * value))
    )
    return energy, gradient, hessian


@ti.func
def _ipc_barrier_terms(value, active_value, kappa):
    energy = 0.0
    gradient = 0.0
    hessian = 0.0
    if value <= 0.0:
        energy = ti.math.inf
    elif value < active_value:
        difference = value - active_value
        inverse_active2 = 1.0 / (active_value * active_value)
        logarithm = ti.log(value / active_value)
        energy = -kappa * difference * difference * inverse_active2 * logarithm
        gradient = -kappa * inverse_active2 * (2.0 * difference * logarithm + difference * difference / value)
        hessian = (
            -kappa
            * inverse_active2
            * (2.0 * logarithm + 4.0 * difference / value - difference * difference / (value * value))
        )
    return energy, gradient, hessian


def _require_double_taichi():
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        raise RuntimeError(
            "FEM contact requires initialized Taichi; call geotaichi.init(..., default_fp='float64') first"
        )
    if ti.lang.impl.current_cfg().default_fp != ti.f64:
        raise RuntimeError("FEM IPC/AL contact requires Taichi default_fp='float64'")


def _closest_point_triangle(point, a, b, c):
    """Host counterpart of the Ericson closest-point construction."""
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array((1.0, 0.0, 0.0))
    bp = point - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array((0.0, 1.0, 0.0))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        value = d1 / (d1 - d3)
        return a + value * ab, np.array((1.0 - value, value, 0.0))
    cp = point - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array((0.0, 0.0, 1.0))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        value = d2 / (d2 - d6)
        return a + value * ac, np.array((1.0 - value, 0.0, value))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        value = (d4 - d3) / (d4 - d3 + d5 - d6)
        return b + value * (c - b), np.array((0.0, 1.0 - value, value))
    denominator = 1.0 / (va + vb + vc)
    bary_b, bary_c = vb * denominator, vc * denominator
    barycentric = np.array((1.0 - bary_b - bary_c, bary_b, bary_c))
    return barycentric[0] * a + barycentric[1] * b + barycentric[2] * c, barycentric


def _closest_points_segments(a0, a1, b0, b1):
    da, db, offset = a1 - a0, b1 - b0, a0 - b0
    aa, ab, bb = np.dot(da, da), np.dot(da, db), np.dot(db, db)
    ao, bo = np.dot(da, offset), np.dot(db, offset)
    epsilon = 1.0e-30
    if aa <= epsilon and bb <= epsilon:
        sa = sb = 0.0
    elif aa <= epsilon:
        sa, sb = 0.0, np.clip(bo / max(bb, epsilon), 0.0, 1.0)
    elif bb <= epsilon:
        sa, sb = np.clip(-ao / max(aa, epsilon), 0.0, 1.0), 0.0
    else:
        denominator = aa * bb - ab * ab
        sa = np.clip((ab * bo - bb * ao) / denominator, 0.0, 1.0) if denominator > epsilon else 0.0
        projected_b = ab * sa + bo
        if projected_b < 0.0:
            sb, sa = 0.0, np.clip(-ao / aa, 0.0, 1.0)
        elif projected_b > bb:
            sb, sa = 1.0, np.clip((ab - ao) / aa, 0.0, 1.0)
        else:
            sb = projected_b / bb
    return a0 + sa * da, b0 + sb * db, float(sa), float(sb)


def _unit_direction(delta, fallback):
    length = float(np.linalg.norm(delta))
    if length > 1.0e-12:
        return delta / length
    fallback_length = float(np.linalg.norm(fallback))
    if fallback_length > 1.0e-12:
        return fallback / fallback_length
    return np.array((1.0, 0.0, 0.0))


@ti.data_oriented
class FEMContactAssembler:
    """Exact IPC barrier or projected non-barrier AL contact.

    PT/EE distance derivatives are shared with GeoTaichi's other IPC engines.
    The IPC branch uses lumped primitive measures, a normalized clamped-log
    barrier, EE mollification, local PSD projection and conservative CCD. The
    AL branch freezes an oriented affine gap during each Newton iteration and
    retains projected multipliers between nonlinear iterations.
    """

    def __init__(self, mesh, material, contact):
        _require_double_taichi()
        self.mesh = mesh
        self.material = material
        self.contact = contact
        selected_bodies = None if contact.body_pair is None else set(contact.body_pair)
        self.surface = build_contact_surface(mesh, selected_bodies)
        face_count = max(int(self.surface.faces.shape[0]), 1)
        self.max_point_triangle_pairs = (
            int(contact.max_point_triangle_pairs)
            if contact.max_point_triangle_pairs is not None
            else max(1, int(np.ceil(face_count * contact.point_triangle_coordination_number)))
        )
        self.max_edge_edge_pairs = (
            int(contact.max_edge_edge_pairs)
            if contact.max_edge_edge_pairs is not None
            else max(1, int(np.ceil(face_count * contact.edge_edge_coordination_number)))
        )
        diagonal = float(np.linalg.norm(np.ptp(mesh.points, axis=0)))
        characteristic = max(diagonal, 1.0)
        if contact.dmin is None:
            thickness = float(getattr(material, "thickness", 0.0)) if mesh.is_membrane else 0.0
            self.dmin = max(thickness, 0.0)
        else:
            self.dmin = float(contact.dmin)
        self.dhat = float(contact.dhat) if contact.dhat is not None else 1.0e-3 * characteristic
        self.constraint_distance = self.dmin + self.dhat
        self.kappa = float(contact.kappa)
        self.penalty = float(contact.penalty)
        self.reference_positions = np.ascontiguousarray(mesh.points, dtype=np.float64)
        self.real_type = ti.lang.impl.current_cfg().default_fp
        self.device_positions = ti.Vector.field(3, dtype=self.real_type, shape=mesh.number_of_nodes)
        self.stiffness_positions = ti.Vector.field(3, dtype=self.real_type, shape=mesh.number_of_nodes)
        self.device_end_positions = ti.Vector.field(3, dtype=self.real_type, shape=mesh.number_of_nodes)
        self.device_positions.from_numpy(self.reference_positions)
        self.device_end_positions.from_numpy(self.reference_positions)
        broad_phase_type = DynamicBVHBroadPhase if contact.broad_phase == "BVH" else DynamicLinkedCellBroadPhase
        self.broad_phase = broad_phase_type(
            self.surface.faces,
            self.surface.edges,
            self.surface.vertices,
            self.surface.node_area,
            self.surface.edge_area,
            self.reference_positions,
            max_point_triangle_pairs=self.max_point_triangle_pairs,
            max_edge_edge_pairs=self.max_edge_edge_pairs,
        )
        self.collision_culling = FEMCollisionCulling(
            self.broad_phase,
            mesh.number_of_nodes,
            mesh.node_body_ids,
            contact.body_pair,
            max_point_triangle_pairs=self.max_point_triangle_pairs,
            max_edge_edge_pairs=self.max_edge_edge_pairs,
        )
        self.pt_candidates = np.empty((0, 4), dtype=np.int32)
        self.ee_candidates = np.empty((0, 4), dtype=np.int32)
        self.pt_measures = np.empty(0, dtype=np.float64)
        self.ee_measures = np.empty(0, dtype=np.float64)
        self.pt_weights = np.empty((0, 4), dtype=np.float64)
        self.ee_weights = np.empty((0, 4), dtype=np.float64)
        self.pt_normals = np.empty((0, 3), dtype=np.float64)
        self.ee_normals = np.empty((0, 3), dtype=np.float64)
        self.pt_lambdas = np.empty(0, dtype=np.float64)
        self.ee_lambdas = np.empty(0, dtype=np.float64)
        self._al_state = {}
        self._normal_state = {}
        self._previous_violation = np.inf
        self._stalled_updates = 0
        self._last_violation = 0.0
        self._active_contacts = 0
        self._last_energy = 0.0
        self._contact_feature_mask_value = 0
        self.friction_candidates_pt = np.empty((0, 4), dtype=np.int32)
        self.friction_candidates_ee = np.empty((0, 4), dtype=np.int32)
        self.friction_weights_pt = np.empty((0, 4), dtype=np.float64)
        self.friction_weights_ee = np.empty((0, 4), dtype=np.float64)
        self.friction_normals_pt = np.empty((0, 3), dtype=np.float64)
        self.friction_normals_ee = np.empty((0, 3), dtype=np.float64)
        self.friction_lambdas_pt = np.empty(0, dtype=np.float64)
        self.friction_lambdas_ee = np.empty(0, dtype=np.float64)
        self.friction_hat_positions = self.reference_positions.copy()
        self.friction_dt = 1.0
        self.plane_friction_lambdas = np.empty((0, 0), dtype=np.float64)
        self.total_energy = ti.field(dtype=ti.f64, shape=())
        self.ccd_alpha = ti.field(dtype=ti.f64, shape=())
        self.active_count = ti.field(dtype=ti.i32, shape=())
        self.penalty_field = ti.field(dtype=ti.f64, shape=())
        self.friction_dt_field = ti.field(dtype=ti.f64, shape=())
        self.penalty_field[None] = self.penalty
        self.friction_dt_field[None] = self.friction_dt

        self.plane_count = len(self.contact.planes)
        allocated_planes = max(self.plane_count, 1)
        allocated_vertices = max(int(self.surface.vertices.size), 1)
        self.plane_origins = ti.Vector.field(3, dtype=self.real_type, shape=allocated_planes)
        self.plane_normals = ti.Vector.field(3, dtype=self.real_type, shape=allocated_planes)
        self.surface_vertices = ti.field(dtype=ti.i32, shape=allocated_vertices)
        self.surface_vertex_measure = ti.field(dtype=self.real_type, shape=allocated_vertices)
        self.plane_multiplier = ti.field(
            dtype=self.real_type,
            shape=(allocated_planes, allocated_vertices),
        )
        self.constraint_violation_field = ti.field(dtype=self.real_type, shape=())
        self.device_pt_capacity = self.max_point_triangle_pairs
        self.device_ee_capacity = self.max_edge_edge_pairs
        self.al_pt_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.device_pt_capacity)
        self.al_ee_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.device_ee_capacity)
        self.al_pt_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.device_pt_capacity)
        self.al_ee_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.device_ee_capacity)
        self.al_pt_multiplier = ti.field(dtype=self.real_type, shape=self.device_pt_capacity)
        self.al_ee_multiplier = ti.field(dtype=self.real_type, shape=self.device_ee_capacity)
        required_hash_capacity = 4 * max(
            self.max_point_triangle_pairs,
            self.max_edge_edge_pairs,
            1,
        )
        self.al_hash_capacity = 1 << (required_hash_capacity - 1).bit_length()
        self.al_pt_hash_count = ti.field(dtype=ti.i32, shape=())
        self.al_ee_hash_count = ti.field(dtype=ti.i32, shape=())
        self.al_pt_hash_state = ti.field(dtype=ti.i32, shape=self.al_hash_capacity)
        self.al_ee_hash_state = ti.field(dtype=ti.i32, shape=self.al_hash_capacity)
        self.al_pt_hash_key = ti.Vector.field(4, dtype=ti.i32, shape=self.al_hash_capacity)
        self.al_ee_hash_key = ti.Vector.field(4, dtype=ti.i32, shape=self.al_hash_capacity)
        self.al_pt_hash_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.al_hash_capacity)
        self.al_ee_hash_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.al_hash_capacity)
        self.al_pt_hash_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.al_hash_capacity)
        self.al_ee_hash_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.al_hash_capacity)
        self.al_pt_hash_multiplier = ti.field(dtype=self.real_type, shape=self.al_hash_capacity)
        self.al_ee_hash_multiplier = ti.field(dtype=self.real_type, shape=self.al_hash_capacity)
        self.friction_pt_capacity = self.max_point_triangle_pairs
        self.friction_ee_capacity = self.max_edge_edge_pairs
        self.friction_pt_count = 0
        self.friction_ee_count = 0
        self.friction_hat_position = ti.Vector.field(3, dtype=self.real_type, shape=mesh.number_of_nodes)
        self.friction_pt_candidate = ti.Vector.field(4, dtype=ti.i32, shape=self.friction_pt_capacity)
        self.friction_ee_candidate = ti.Vector.field(4, dtype=ti.i32, shape=self.friction_ee_capacity)
        self.friction_pt_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.friction_pt_capacity)
        self.friction_ee_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.friction_ee_capacity)
        self.friction_pt_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.friction_pt_capacity)
        self.friction_ee_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.friction_ee_capacity)
        self.friction_pt_normal_force = ti.field(dtype=self.real_type, shape=self.friction_pt_capacity)
        self.friction_ee_normal_force = ti.field(dtype=self.real_type, shape=self.friction_ee_capacity)
        self.plane_friction_normal_force = ti.field(
            dtype=self.real_type,
            shape=(allocated_planes, allocated_vertices),
        )
        if self.plane_count:
            self.plane_origins.from_numpy(
                np.ascontiguousarray(
                    [entry[0] for entry in self.contact.planes],
                    dtype=np.float64,
                )
            )
            self.plane_normals.from_numpy(
                np.ascontiguousarray(
                    [entry[1] for entry in self.contact.planes],
                    dtype=np.float64,
                )
            )
        vertex_buffer = np.zeros(allocated_vertices, dtype=np.int32)
        measure_buffer = np.zeros(allocated_vertices, dtype=np.float64)
        vertex_buffer[: self.surface.vertices.size] = self.surface.vertices
        measure_buffer[: self.surface.vertices.size] = self.surface.node_area[self.surface.vertices]
        self.surface_vertices.from_numpy(vertex_buffer)
        self.surface_vertex_measure.from_numpy(measure_buffer)
        self.plane_multiplier.fill(0.0)
        self.plane_friction_normal_force.fill(0.0)
        self.al_pt_hash_count[None] = 0
        self.al_ee_hash_count[None] = 0
        self.al_pt_hash_state.fill(2)
        self.al_ee_hash_state.fill(2)
        self.prepare_iteration_device(self.device_positions)

    @property
    def is_ipc(self):
        return self.contact.model == "IPC"

    @property
    def is_augmented_lagrangian(self):
        return self.contact.model == "AugmentedLagrangian"

    def _host_candidate_snapshot(self, positions, end_positions=None):
        """Diagnostic adapter around the selected device broad phase."""
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        self.device_positions.from_numpy(positions)
        if end_positions is None:
            end_field = None
        else:
            self.device_end_positions.from_numpy(np.ascontiguousarray(end_positions, dtype=np.float64))
            end_field = self.device_end_positions
        if end_field is None:
            pt_count, ee_count = self.collision_culling.rebuild_proximity(self.device_positions, self.dmin + self.dhat)
        else:
            pt_count, ee_count = self.collision_culling.rebuild_swept_candidates(
                self.device_positions,
                end_field,
                self.dmin + self.dhat,
            )
        return (
            self.collision_culling.point_triangle.to_numpy()[:pt_count].copy(),
            self.collision_culling.edge_edge.to_numpy()[:ee_count].copy(),
        )

    def set_stitch_exclusions(self, stitches):
        """Exclude garment-stitch neighborhoods from self-contact culling."""
        self.collision_culling.set_stitch_exclusions(stitches)
        self.prepare_iteration_device(self.device_positions)

    def _candidate_measure(self, candidates, kind):
        if candidates.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        if kind == "pt":
            area = self.surface.node_area[candidates[:, 0]]
        else:
            edge_lookup = {
                tuple(map(int, edge)): value for edge, value in zip(self.surface.edges, self.surface.edge_area)
            }
            first = np.asarray([edge_lookup[tuple(sorted(map(int, row[:2])))] for row in candidates])
            second = np.asarray([edge_lookup[tuple(sorted(map(int, row[2:])))] for row in candidates])
            area = first + second
        integration_weight = 0.25
        if self.is_ipc:
            return np.ascontiguousarray(integration_weight * area * self.dhat)
        return np.ascontiguousarray(integration_weight * area)

    def _key(self, prefix, stencil):
        return (prefix, *map(int, stencil))

    def _al_frames(self, positions, candidates, kind):
        count = candidates.shape[0]
        weights = np.empty((count, 4), dtype=np.float64)
        normals = np.empty((count, 3), dtype=np.float64)
        lambdas = np.empty(count, dtype=np.float64)
        for index, stencil in enumerate(candidates):
            points = positions[stencil]
            key = self._key(kind, stencil)
            if kind == "pt":
                closest, barycentric = _closest_point_triangle(points[0], points[1], points[2], points[3])
                relative = points[0] - closest
                fallback = np.cross(points[2] - points[1], points[3] - points[1])
                weights[index] = (1.0, -barycentric[0], -barycentric[1], -barycentric[2])
            else:
                closest_a, closest_b, parameter_a, parameter_b = _closest_points_segments(*points)
                relative = closest_a - closest_b
                fallback = np.cross(points[1] - points[0], points[3] - points[2])
                weights[index] = (1.0 - parameter_a, parameter_a, -(1.0 - parameter_b), -parameter_b)
            normal = _unit_direction(relative, fallback)
            previous = self._normal_state.get(key)
            if previous is not None and float(np.dot(normal, previous)) < 0.0:
                normal = -normal
            normals[index] = normal
            self._normal_state[key] = normal.copy()
            lambdas[index] = self._al_state.get(key, 0.0)
        return weights, normals, lambdas

    def _contact_frames(self, positions, candidates, kind):
        count = candidates.shape[0]
        weights = np.empty((count, 4), dtype=np.float64)
        normals = np.empty((count, 3), dtype=np.float64)
        for index, stencil in enumerate(candidates):
            points = positions[stencil]
            if kind == "pt":
                closest, barycentric = _closest_point_triangle(points[0], points[1], points[2], points[3])
                relative = points[0] - closest
                fallback = np.cross(points[2] - points[1], points[3] - points[1])
                weights[index] = (1.0, -barycentric[0], -barycentric[1], -barycentric[2])
            else:
                closest_a, closest_b, parameter_a, parameter_b = _closest_points_segments(*points)
                relative = closest_a - closest_b
                fallback = np.cross(points[1] - points[0], points[3] - points[2])
                weights[index] = (1.0 - parameter_a, parameter_a, -(1.0 - parameter_b), -parameter_b)
            normals[index] = _unit_direction(relative, fallback)
        return weights, normals

    def _frozen_normal_forces(self, positions, candidates, measures, kind):
        forces = np.zeros(candidates.shape[0], dtype=np.float64)
        if self.is_augmented_lagrangian:
            weights, normals, multipliers = self._al_frames(positions, candidates, kind)
            if candidates.shape[0]:
                relative = np.einsum("ij,ijk->ik", weights, positions[candidates])
                gaps = np.einsum("ij,ij->i", relative, normals) - self.constraint_distance
                forces[:] = measures * np.maximum(multipliers - self.penalty * gaps, 0.0)
            return forces
        active_gap2 = (2.0 * self.dmin + self.dhat) * self.dhat
        for index, stencil in enumerate(candidates):
            points = positions[stencil]
            if kind == "pt":
                closest, _ = _closest_point_triangle(points[0], points[1], points[2], points[3])
                distance = float(np.linalg.norm(points[0] - closest))
                mollifier = 1.0
            else:
                closest_a, closest_b, _, _ = _closest_points_segments(*points)
                distance = float(np.linalg.norm(closest_a - closest_b))
                edge_a = points[1] - points[0]
                edge_b = points[3] - points[2]
                cross2 = float(np.dot(np.cross(edge_a, edge_b), np.cross(edge_a, edge_b)))
                rest = self.reference_positions[stencil]
                rest_a = rest[1] - rest[0]
                rest_b = rest[3] - rest[2]
                threshold = 1.0e-3 * float(np.dot(rest_a, rest_a) * np.dot(rest_b, rest_b))
                if threshold > 0.0 and cross2 < threshold:
                    ratio = cross2 / threshold
                    mollifier = ratio * (2.0 - ratio)
                else:
                    mollifier = 1.0
            shifted_distance2 = distance * distance - self.dmin * self.dmin
            if shifted_distance2 > 0.0:
                _, derivative, _ = _ipc_barrier_terms_numpy(shifted_distance2, active_gap2, self.kappa)
                # A mollified EE force is not used as a frozen friction load.
                if kind == "pt" or mollifier >= 1.0 - 1.0e-12:
                    forces[index] = max(
                        0.0,
                        -measures[index] * derivative * 2.0 * np.sqrt(shifted_distance2),
                    )
        return forces

    def begin_step(self, positions, dt):
        """Freeze the lagged-friction state for one implicit time step."""
        if self.is_augmented_lagrangian:
            self._al_state.clear()
            self.al_pt_hash_state.fill(2)
            self.al_ee_hash_state.fill(2)
            self.al_pt_hash_multiplier.fill(0.0)
            self.al_ee_hash_multiplier.fill(0.0)
            self.al_pt_hash_count[None] = 0
            self.al_ee_hash_count[None] = 0
            self.plane_multiplier.fill(0.0)
        self.friction_hat_positions = np.ascontiguousarray(positions, dtype=np.float64).copy()
        self.friction_dt = float(dt)
        if not np.isfinite(self.friction_dt) or self.friction_dt <= 0.0:
            raise ValueError("FEM contact time step must be finite and positive")
        self.friction_dt_field[None] = self.friction_dt
        if self.contact.friction_coefficient <= 0.0:
            return
        pt, ee = self._host_candidate_snapshot(positions)
        self.friction_candidates_pt = np.ascontiguousarray(pt)
        self.friction_candidates_ee = np.ascontiguousarray(ee)
        self.friction_weights_pt, self.friction_normals_pt = self._contact_frames(positions, pt, "pt")
        self.friction_weights_ee, self.friction_normals_ee = self._contact_frames(positions, ee, "ee")
        pt_measures = self._candidate_measure(pt, "pt")
        ee_measures = self._candidate_measure(ee, "ee")
        self.friction_lambdas_pt = self._frozen_normal_forces(positions, pt, pt_measures, "pt")
        self.friction_lambdas_ee = self._frozen_normal_forces(positions, ee, ee_measures, "ee")
        if self.contact.planes:
            vertices = self.surface.vertices
            self.plane_friction_lambdas = np.zeros((len(self.contact.planes), vertices.size), dtype=np.float64)
            for plane_id, (origin, normal) in enumerate(self.contact.planes):
                gaps = (positions[vertices] - origin) @ normal - self.constraint_distance
                for vertex_id, gap in enumerate(gaps):
                    if self.is_augmented_lagrangian:
                        multiplier = self._al_state.get(
                            ("plane", plane_id, int(vertices[vertex_id])),
                            0.0,
                        )
                        self.plane_friction_lambdas[plane_id, vertex_id] = self.surface.node_area[
                            vertices[vertex_id]
                        ] * max(multiplier - self.penalty * float(gap), 0.0)
                    elif gap > 0.0:
                        _, derivative, _ = _ipc_barrier_terms_numpy(float(gap), self.dhat, self.kappa)
                        self.plane_friction_lambdas[plane_id, vertex_id] = max(
                            0.0,
                            -self.surface.node_area[vertices[vertex_id]] * self.dhat * derivative,
                        )

    def prepare_iteration(self, positions, end_positions=None):
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        radius = self.dmin + self.dhat
        if self.contact.self_contact:
            pt, ee = self._host_candidate_snapshot(positions, end_positions)
        else:
            pt = np.empty((0, 4), dtype=np.int32)
            ee = np.empty((0, 4), dtype=np.int32)
        if self.is_augmented_lagrangian and self._al_state:
            retained_pt = [key[1:] for key, value in self._al_state.items() if key[0] == "pt" and value > 0.0]
            retained_ee = [key[1:] for key, value in self._al_state.items() if key[0] == "ee" and value > 0.0]
            if retained_pt:
                pt = np.asarray(list(dict.fromkeys(map(tuple, np.vstack((pt, retained_pt))))), dtype=np.int32).reshape(
                    -1, 4
                )
            if retained_ee:
                ee = np.asarray(list(dict.fromkeys(map(tuple, np.vstack((ee, retained_ee))))), dtype=np.int32).reshape(
                    -1, 4
                )
        self.pt_candidates = np.ascontiguousarray(pt)
        self.ee_candidates = np.ascontiguousarray(ee)
        self.pt_measures = self._candidate_measure(pt, "pt")
        self.ee_measures = self._candidate_measure(ee, "ee")
        if self.is_augmented_lagrangian:
            self.pt_weights, self.pt_normals, self.pt_lambdas = self._al_frames(positions, pt, "pt")
            self.ee_weights, self.ee_normals, self.ee_lambdas = self._al_frames(positions, ee, "ee")
            self._last_violation = self.constraint_violation(positions)

    @ti.func
    def _position(self, positions: ti.template(), node: ti.i32):
        return ti.Vector([positions[node, 0], positions[node, 1], positions[node, 2]])

    @ti.kernel
    def _assemble_ipc_pt(
        self,
        positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        candidates: ti.types.ndarray(dtype=ti.i32, ndim=2),
        measures: ti.types.ndarray(dtype=ti.f64, ndim=1),
        force: ti.types.ndarray(dtype=ti.f64, ndim=2),
        hessian: ti.types.ndarray(dtype=ti.f64, ndim=5),
        need_hessian: ti.template(),
    ):
        active_gap2 = (2.0 * self.dmin + self.dhat) * self.dhat
        normalized_kappa = self.kappa / (active_gap2 * active_gap2)
        for contact_id in range(candidates.shape[0]):
            if ti.static(need_hessian):
                for local_i, local_j, row, column in ti.ndrange(4, 4, 3, 3):
                    hessian[contact_id, local_i, local_j, row, column] = 0.0
            ids = ti.Vector([candidates[contact_id, site] for site in ti.static(range(4))])
            distance2 = 0.0
            distance_gradient = ti.Vector.zero(ti.f64, 12)
            distance_hessian = ti.Matrix.zero(ti.f64, 12, 12)
            if ti.static(need_hessian):
                distance2, distance_gradient, distance_hessian, _ = point_triangle_distance_grad_hess(
                    self._position(positions, ids[0]),
                    self._position(positions, ids[1]),
                    self._position(positions, ids[2]),
                    self._position(positions, ids[3]),
                )
            else:
                distance2, distance_gradient, _ = point_triangle_distance_grad(
                    self._position(positions, ids[0]),
                    self._position(positions, ids[1]),
                    self._position(positions, ids[2]),
                    self._position(positions, ids[3]),
                )
            shifted = distance2 - self.dmin * self.dmin
            if shifted < active_gap2:
                value, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                    distance2, self.dhat, self.dmin, normalized_kappa, 0
                )
                scale = measures[contact_id]
                ti.atomic_add(self.total_energy[None], scale * value)
                ti.atomic_add(self.active_count[None], 1)
                for site, component in ti.static(ti.ndrange(4, 3)):
                    local = 3 * site + component
                    ti.atomic_add(force[ids[site], component], scale * first * distance_gradient[local])
                if ti.static(need_hessian):
                    if shifted <= 0.0:
                        continue
                    local_hessian = ti.Matrix.zero(ti.f64, 12, 12)
                    for row, column in ti.ndrange(12, 12):
                        local_hessian[row, column] = scale * (
                            first * distance_hessian[row, column]
                            + second * distance_gradient[row] * distance_gradient[column]
                        )
                    if ti.static(self.contact.project_pd):
                        local_hessian = psd_project_gershgorin_nd(local_hessian)
                    for local_i, local_j in ti.ndrange(4, 4):
                        for row, column in ti.static(ti.ndrange(3, 3)):
                            hessian[contact_id, local_i, local_j, row, column] = local_hessian[
                                3 * local_i + row, 3 * local_j + column
                            ]

    @ti.kernel
    def _assemble_ipc_ee(
        self,
        positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        reference: ti.types.ndarray(dtype=ti.f64, ndim=2),
        candidates: ti.types.ndarray(dtype=ti.i32, ndim=2),
        measures: ti.types.ndarray(dtype=ti.f64, ndim=1),
        force: ti.types.ndarray(dtype=ti.f64, ndim=2),
        hessian: ti.types.ndarray(dtype=ti.f64, ndim=5),
        need_hessian: ti.template(),
    ):
        active_gap2 = (2.0 * self.dmin + self.dhat) * self.dhat
        normalized_kappa = self.kappa / (active_gap2 * active_gap2)
        for contact_id in range(candidates.shape[0]):
            if ti.static(need_hessian):
                for local_i, local_j, row, column in ti.ndrange(4, 4, 3, 3):
                    hessian[contact_id, local_i, local_j, row, column] = 0.0
            ids = ti.Vector([candidates[contact_id, site] for site in ti.static(range(4))])
            a0, a1 = self._position(positions, ids[0]), self._position(positions, ids[1])
            b0, b1 = self._position(positions, ids[2]), self._position(positions, ids[3])
            distance2 = 0.0
            distance_gradient = ti.Vector.zero(ti.f64, 12)
            distance_hessian = ti.Matrix.zero(ti.f64, 12, 12)
            if ti.static(need_hessian):
                distance2, distance_gradient, distance_hessian, _ = edge_edge_distance_grad_hess(a0, a1, b0, b1)
            else:
                distance2, distance_gradient, _ = edge_edge_distance_grad(a0, a1, b0, b1)
            shifted = distance2 - self.dmin * self.dmin
            if shifted < active_gap2:
                value, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                    distance2, self.dhat, self.dmin, normalized_kappa, 0
                )
                ra0, ra1 = self._position(reference, ids[0]), self._position(reference, ids[1])
                rb0, rb1 = self._position(reference, ids[2]), self._position(reference, ids[3])
                threshold = edge_edge_mollifier_threshold(ra0, ra1, rb0, rb1)
                mollifier = 0.0
                mollifier_gradient = ti.Vector.zero(ti.f64, 12)
                mollifier_hessian = ti.Matrix.zero(ti.f64, 12, 12)
                if ti.static(need_hessian):
                    mollifier, mollifier_gradient, mollifier_hessian = edge_edge_mollifier_terms(
                        a0, a1, b0, b1, threshold
                    )
                else:
                    mollifier = edge_edge_mollifier(a0, a1, b0, b1, threshold)
                    mollifier_gradient = edge_edge_mollifier_grad(a0, a1, b0, b1, threshold)
                barrier_gradient = first * distance_gradient
                gradient = mollifier * barrier_gradient + value * mollifier_gradient
                scale = measures[contact_id]
                ti.atomic_add(self.total_energy[None], scale * mollifier * value)
                ti.atomic_add(self.active_count[None], 1)
                for site, component in ti.static(ti.ndrange(4, 3)):
                    ti.atomic_add(force[ids[site], component], scale * gradient[3 * site + component])
                if ti.static(need_hessian):
                    if shifted <= 0.0:
                        continue
                    local_hessian = ti.Matrix.zero(ti.f64, 12, 12)
                    for row, column in ti.ndrange(12, 12):
                        barrier_hessian = (
                            first * distance_hessian[row, column]
                            + second * distance_gradient[row] * distance_gradient[column]
                        )
                        local_hessian[row, column] = scale * (
                            mollifier * barrier_hessian
                            + value * mollifier_hessian[row, column]
                            + mollifier_gradient[row] * barrier_gradient[column]
                            + barrier_gradient[row] * mollifier_gradient[column]
                        )
                    if ti.static(self.contact.project_pd):
                        local_hessian = psd_project_gershgorin_nd(local_hessian)
                    for local_i, local_j in ti.ndrange(4, 4):
                        for row, column in ti.static(ti.ndrange(3, 3)):
                            hessian[contact_id, local_i, local_j, row, column] = local_hessian[
                                3 * local_i + row, 3 * local_j + column
                            ]

    @ti.kernel
    def _assemble_al(
        self,
        positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        candidates: ti.types.ndarray(dtype=ti.i32, ndim=2),
        measures: ti.types.ndarray(dtype=ti.f64, ndim=1),
        weights: ti.types.ndarray(dtype=ti.f64, ndim=2),
        normals: ti.types.ndarray(dtype=ti.f64, ndim=2),
        multipliers: ti.types.ndarray(dtype=ti.f64, ndim=1),
        force: ti.types.ndarray(dtype=ti.f64, ndim=2),
        hessian: ti.types.ndarray(dtype=ti.f64, ndim=5),
        need_hessian: ti.i32,
    ):
        penalty = self.penalty_field[None]
        for contact_id in range(candidates.shape[0]):
            if need_hessian:
                for local_i, local_j, row, column in ti.ndrange(4, 4, 3, 3):
                    hessian[contact_id, local_i, local_j, row, column] = 0.0
            gap = -self.constraint_distance
            for site, component in ti.static(ti.ndrange(4, 3)):
                gap += (
                    weights[contact_id, site]
                    * normals[contact_id, component]
                    * positions[candidates[contact_id, site], component]
                )
            value, derivative, curvature = semi_ipc_terms(gap, multipliers[contact_id], penalty)
            scale = measures[contact_id]
            ti.atomic_add(self.total_energy[None], scale * value)
            if curvature > 0.0:
                ti.atomic_add(self.active_count[None], 1)
                for site, component in ti.static(ti.ndrange(4, 3)):
                    gradient = weights[contact_id, site] * normals[contact_id, component]
                    ti.atomic_add(force[candidates[contact_id, site], component], scale * derivative * gradient)
                if need_hessian:
                    for first, component_i, second, component_j in ti.ndrange(4, 3, 4, 3):
                        hessian[contact_id, first, second, component_i, component_j] = (
                            scale
                            * curvature
                            * weights[contact_id, first]
                            * weights[contact_id, second]
                            * normals[contact_id, component_i]
                            * normals[contact_id, component_j]
                        )

    def _assemble_friction(
        self,
        positions,
        hat_positions,
        candidates,
        weights,
        normals,
        normal_forces,
        force,
        hessian,
        need_hessian,
    ):
        timestep = float(self.friction_dt_field[None])
        coefficient = self.contact.friction_coefficient * normal_forces
        tangent = np.eye(3)[None, :, :] - normals[:, :, None] * normals[:, None, :]
        increments = positions[candidates] - hat_positions[candidates]
        relative_increment = np.einsum("ni,nij->nj", weights, increments)
        velocity = np.einsum("nij,nj->ni", tangent, relative_increment) / timestep
        speed = np.linalg.norm(velocity, axis=1)
        epsv = self.contact.epsv
        profile = np.where(
            speed < epsv,
            (-speed + 2.0 * epsv) / (epsv * epsv),
            1.0 / np.maximum(speed, np.finfo(np.float64).tiny),
        )
        displacement = speed * timestep
        threshold = epsv * timestep
        potential = np.where(
            speed < epsv,
            displacement * displacement * (-displacement / 3.0 + threshold) / (threshold * threshold) + threshold / 3.0,
            displacement,
        )
        self.total_energy[None] = float(self.total_energy[None]) + float(np.sum(coefficient * potential))
        gradient = coefficient[:, None] * profile[:, None] * np.einsum("nij,nj->ni", tangent, velocity)
        np.add.at(
            force,
            candidates.reshape(-1),
            (weights[:, :, None] * gradient[:, None, :]).reshape(-1, 3),
        )
        if not need_hessian:
            return
        radial = np.zeros_like(speed)
        moving = speed > 0.0
        radial[moving] = (
            coefficient[moving]
            * np.where(
                speed[moving] < epsv,
                -1.0 / (epsv * epsv),
                -1.0 / (speed[moving] * speed[moving]),
            )
            / speed[moving]
        )
        inner = coefficient[:, None, None] * profile[:, None, None] * np.eye(3)
        inner += radial[:, None, None] * velocity[:, :, None] * velocity[:, None, :]
        relative_hessian = np.einsum("nij,njk,nlk->nil", tangent, inner, tangent) / timestep
        if self.contact.project_pd and self.is_ipc:
            for contact_id in range(relative_hessian.shape[0]):
                values, vectors = np.linalg.eigh(0.5 * (relative_hessian[contact_id] + relative_hessian[contact_id].T))
                relative_hessian[contact_id] = (vectors * np.maximum(values, 0.0)[None, :]) @ vectors.T
        hessian[:] = (
            weights[:, :, None, None, None] * weights[:, None, :, None, None] * relative_hessian[:, None, None, :, :]
        )

    @ti.kernel
    def _assemble_plane_friction(
        self,
        positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        hat_positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        vertices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        normals: ti.types.ndarray(dtype=ti.f64, ndim=2),
        normal_forces: ti.types.ndarray(dtype=ti.f64, ndim=2),
        force: ti.types.ndarray(dtype=ti.f64, ndim=2),
        hessian: ti.types.ndarray(dtype=ti.f64, ndim=3),
        need_hessian: ti.i32,
    ):
        identity = ti.Matrix.identity(ti.f64, 3)
        timestep = self.friction_dt_field[None]
        for plane_id, vertex_id in ti.ndrange(normals.shape[0], vertices.shape[0]):
            coefficient = self.contact.friction_coefficient * normal_forces[plane_id, vertex_id]
            if coefficient > 0.0:
                flat_id = plane_id * vertices.shape[0] + vertex_id
                node = vertices[vertex_id]
                normal = ti.Vector([normals[plane_id, component] for component in ti.static(range(3))])
                tangent = identity - normal.outer_product(normal)
                increment = self._position(positions, node) - self._position(hat_positions, node)
                velocity = tangent @ increment / timestep
                speed = velocity.norm()
                value = coefficient * ipc_friction_f0(speed, self.contact.epsv, timestep)
                profile = ipc_friction_f1_over_speed(speed, self.contact.epsv)
                gradient = coefficient * profile * (tangent @ velocity)
                ti.atomic_add(self.total_energy[None], value)
                for component in ti.static(range(3)):
                    ti.atomic_add(force[node, component], gradient[component])
                if need_hessian:
                    inner = coefficient * profile * identity
                    if speed > 0.0:
                        inner += (
                            coefficient
                            * ipc_friction_hessian_term(speed, self.contact.epsv)
                            / speed
                            * velocity.outer_product(velocity)
                        )
                    local_hessian = tangent @ inner @ tangent / timestep
                    for row, column in ti.static(ti.ndrange(3, 3)):
                        hessian[flat_id, row, column] += local_hessian[row, column]

    @ti.kernel
    def _assemble_planes(
        self,
        positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        vertices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        origins: ti.types.ndarray(dtype=ti.f64, ndim=2),
        normals: ti.types.ndarray(dtype=ti.f64, ndim=2),
        measures: ti.types.ndarray(dtype=ti.f64, ndim=1),
        multipliers: ti.types.ndarray(dtype=ti.f64, ndim=2),
        force: ti.types.ndarray(dtype=ti.f64, ndim=2),
        hessian: ti.types.ndarray(dtype=ti.f64, ndim=3),
        need_hessian: ti.i32,
        use_ipc: ti.i32,
    ):
        penalty = self.penalty_field[None]
        for plane_id, vertex_id in ti.ndrange(origins.shape[0], vertices.shape[0]):
            flat_id = plane_id * vertices.shape[0] + vertex_id
            if need_hessian:
                for row, column in ti.static(ti.ndrange(3, 3)):
                    hessian[flat_id, row, column] = 0.0
            node = vertices[vertex_id]
            distance = 0.0
            for component in ti.static(range(3)):
                distance += normals[plane_id, component] * (positions[node, component] - origins[plane_id, component])
            scale = measures[vertex_id]
            if use_ipc:
                gap = distance - self.dmin
                if gap < self.dhat:
                    value, derivative, curvature = _ipc_barrier_terms(gap, self.dhat, self.kappa)
                    ti.atomic_add(self.total_energy[None], scale * self.dhat * value)
                    ti.atomic_add(self.active_count[None], 1)
                    for component in ti.static(range(3)):
                        ti.atomic_add(
                            force[node, component], scale * self.dhat * derivative * normals[plane_id, component]
                        )
                        if need_hessian:
                            for second_component in ti.static(range(3)):
                                hessian[flat_id, component, second_component] = (
                                    scale
                                    * self.dhat
                                    * curvature
                                    * normals[plane_id, component]
                                    * normals[plane_id, second_component]
                                )
            else:
                gap = distance - self.constraint_distance
                value, derivative, curvature = semi_ipc_terms(
                    gap,
                    multipliers[plane_id, vertex_id],
                    penalty,
                )
                ti.atomic_add(self.total_energy[None], scale * value)
                if curvature > 0.0:
                    ti.atomic_add(self.active_count[None], 1)
                    for component in ti.static(range(3)):
                        ti.atomic_add(force[node, component], scale * derivative * normals[plane_id, component])
                        if need_hessian:
                            for second_component in ti.static(range(3)):
                                hessian[flat_id, component, second_component] = (
                                    scale
                                    * curvature
                                    * normals[plane_id, component]
                                    * normals[plane_id, second_component]
                                )

    @ti.kernel
    def _copy_stiffness_positions(self, positions: ti.template()):
        for node in range(self.mesh.number_of_nodes):
            self.stiffness_positions[node] = positions[node]

    @ti.func
    def _initialize_direct_stencil(
        self,
        target: ti.template(),
        assemble_hash: ti.template(),
        scalar_base,
        raw_base,
        contact_id,
        ids,
    ):
        for local_i, local_j in ti.static(ti.ndrange(4, 4)):
            if ti.static(assemble_hash):
                if local_i != local_j:
                    raw_entry = raw_base + 12 * contact_id + 3 * local_i + local_j - ti.cast(local_j > local_i, ti.i32)
                    target.initialize_raw_block_slot(raw_entry, ids[local_i], ids[local_j])
            else:
                for row, column in ti.static(ti.ndrange(3, 3)):
                    entry = scalar_base + 144 * contact_id + 36 * local_i + 9 * local_j + 3 * row + column
                    target.rows[entry] = 3 * ids[local_i] + row
                    target.cols[entry] = 3 * ids[local_j] + column
                    target.data[entry] = 0.0

    @ti.func
    def _add_direct_stencil_block(
        self,
        target: ti.template(),
        assemble_hash: ti.template(),
        scalar_base,
        raw_base,
        contact_id,
        ids,
        local_i,
        local_j,
        block,
    ):
        if ti.static(assemble_hash):
            if local_i == local_j:
                target.add_block_entry(ids[local_i], ids[local_j], block)
            else:
                raw_entry = raw_base + 12 * contact_id + 3 * local_i + local_j - ti.cast(local_j > local_i, ti.i32)
                target.atomic_add_raw_block_slot(raw_entry, block)
        else:
            for row, column in ti.static(ti.ndrange(3, 3)):
                entry = scalar_base + 144 * contact_id + 36 * local_i + 9 * local_j + 3 * row + column
                target.data[entry] += block[row, column]

    @ti.func
    def _initialize_direct_plane_block(
        self,
        target: ti.template(),
        assemble_hash: ti.template(),
        scalar_base,
        flat_id,
        node,
    ):
        if ti.static(not assemble_hash):
            for row, column in ti.static(ti.ndrange(3, 3)):
                entry = scalar_base + 9 * flat_id + 3 * row + column
                target.rows[entry] = 3 * node + row
                target.cols[entry] = 3 * node + column
                target.data[entry] = 0.0

    @ti.func
    def _add_direct_plane_block(
        self,
        target: ti.template(),
        assemble_hash: ti.template(),
        scalar_base,
        flat_id,
        node,
        block,
    ):
        if ti.static(assemble_hash):
            target.add_block_entry(node, node, block)
        else:
            for row, column in ti.static(ti.ndrange(3, 3)):
                target.data[scalar_base + 9 * flat_id + 3 * row + column] += block[row, column]

    @ti.kernel
    def _assemble_planes_device(
        self,
        positions: ti.template(),
        force: ti.template(),
    ):
        penalty = self.penalty_field[None]
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex_id]
            normal = self.plane_normals[plane_id]
            distance = normal.dot(positions[node] - self.plane_origins[plane_id])
            measure = self.surface_vertex_measure[vertex_id]
            if ti.static(self.is_ipc):
                gap = distance - self.dmin
                if gap < self.dhat:
                    value, derivative, curvature = _ipc_barrier_terms(gap, self.dhat, self.kappa)
                    ti.atomic_add(self.total_energy[None], measure * self.dhat * value)
                    ti.atomic_add(self.active_count[None], 1)
                    for component in ti.static(range(3)):
                        ti.atomic_add(
                            force[node][component],
                            measure * self.dhat * derivative * normal[component],
                        )
            else:
                gap = distance - self.constraint_distance
                multiplier = self.plane_multiplier[plane_id, vertex_id]
                value, derivative, curvature = semi_ipc_terms(gap, multiplier, penalty)
                ti.atomic_add(self.total_energy[None], measure * value)
                if curvature > 0.0:
                    ti.atomic_add(self.active_count[None], 1)
                    for component in ti.static(range(3)):
                        ti.atomic_add(
                            force[node][component],
                            measure * derivative * normal[component],
                        )

    @ti.kernel
    def _assemble_ipc_pt_device(
        self,
        positions: ti.template(),
        force: ti.template(),
        count: ti.i32,
    ):
        active_gap2 = (2.0 * self.dmin + self.dhat) * self.dhat
        normalized_kappa = self.kappa / (active_gap2 * active_gap2)
        for contact_id in range(count):
            ids = self.collision_culling.point_triangle[contact_id]
            distance2, distance_gradient, _ = point_triangle_distance_grad(
                positions[ids[0]],
                positions[ids[1]],
                positions[ids[2]],
                positions[ids[3]],
            )
            shifted = distance2 - self.dmin * self.dmin
            if shifted < active_gap2:
                value, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                    distance2,
                    self.dhat,
                    self.dmin,
                    normalized_kappa,
                    0,
                )
                scale = self.collision_culling.point_triangle_measure[contact_id] * self.dhat
                ti.atomic_add(self.total_energy[None], scale * value)
                ti.atomic_add(self.active_count[None], 1)
                for site, component in ti.static(ti.ndrange(4, 3)):
                    ti.atomic_add(
                        force[ids[site]][component],
                        scale * first * distance_gradient[3 * site + component],
                    )

    @ti.kernel
    def _assemble_ipc_ee_device(
        self,
        positions: ti.template(),
        force: ti.template(),
        count: ti.i32,
    ):
        active_gap2 = (2.0 * self.dmin + self.dhat) * self.dhat
        normalized_kappa = self.kappa / (active_gap2 * active_gap2)
        for contact_id in range(count):
            ids = self.collision_culling.edge_edge[contact_id]
            a0, a1 = positions[ids[0]], positions[ids[1]]
            b0, b1 = positions[ids[2]], positions[ids[3]]
            distance2, distance_gradient, _ = edge_edge_distance_grad(a0, a1, b0, b1)
            shifted = distance2 - self.dmin * self.dmin
            if shifted < active_gap2:
                value, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                    distance2,
                    self.dhat,
                    self.dmin,
                    normalized_kappa,
                    0,
                )
                ra0 = self.collision_culling.reference_position[ids[0]]
                ra1 = self.collision_culling.reference_position[ids[1]]
                rb0 = self.collision_culling.reference_position[ids[2]]
                rb1 = self.collision_culling.reference_position[ids[3]]
                threshold = edge_edge_mollifier_threshold(ra0, ra1, rb0, rb1)
                mollifier = edge_edge_mollifier(a0, a1, b0, b1, threshold)
                mollifier_gradient = edge_edge_mollifier_grad(a0, a1, b0, b1, threshold)
                barrier_gradient = first * distance_gradient
                gradient = mollifier * barrier_gradient + value * mollifier_gradient
                scale = self.collision_culling.edge_edge_measure[contact_id] * self.dhat
                ti.atomic_add(self.total_energy[None], scale * mollifier * value)
                ti.atomic_add(self.active_count[None], 1)
                for site, component in ti.static(ti.ndrange(4, 3)):
                    ti.atomic_add(
                        force[ids[site]][component],
                        scale * gradient[3 * site + component],
                    )

    @ti.kernel
    def _contact_feature_mask(
        self,
        pt_count: ti.i32,
        ee_count: ti.i32,
    ) -> ti.i32:
        mask = 0
        for contact_id in range(pt_count):
            ids = self.collision_culling.point_triangle[contact_id]
            mask |= 1 << point_triangle_distance_type(
                self.stiffness_positions[ids[0]],
                self.stiffness_positions[ids[1]],
                self.stiffness_positions[ids[2]],
                self.stiffness_positions[ids[3]],
            )
        for contact_id in range(ee_count):
            ids = self.collision_culling.edge_edge[contact_id]
            mask |= 1 << (
                7
                + edge_edge_distance_type(
                    self.stiffness_positions[ids[0]],
                    self.stiffness_positions[ids[1]],
                    self.stiffness_positions[ids[2]],
                    self.stiffness_positions[ids[3]],
                )
            )
        return mask

    @ti.kernel
    def _scatter_ipc_pt_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        count: ti.i32,
        scalar_base: ti.i32,
        raw_base: ti.i32,
        assemble_hash: ti.template(),
        contact_type: ti.template(),
    ):
        active_gap2 = (2.0 * self.dmin + self.dhat) * self.dhat
        normalized_kappa = self.kappa / (active_gap2 * active_gap2)
        for contact_id in range(count):
            ids = self.collision_culling.point_triangle[contact_id]
            self._initialize_direct_stencil(target, assemble_hash, scalar_base, raw_base, contact_id, ids)
            matches = (
                point_triangle_distance_type(positions[ids[0]], positions[ids[1]], positions[ids[2]], positions[ids[3]])
                == contact_type
            )
            distance2 = 0.0
            distance_gradient = ti.Vector.zero(self.real_type, 12)
            distance_hessian = ti.Matrix.zero(self.real_type, 12, 12)
            if matches:
                distance2, distance_gradient, distance_hessian = point_triangle_distance_grad_hess_by_type(
                    positions[ids[0]], positions[ids[1]], positions[ids[2]], positions[ids[3]], contact_type
                )
            shifted = distance2 - self.dmin * self.dmin
            if matches and shifted < active_gap2 and shifted > 0.0:
                _, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                    distance2, self.dhat, self.dmin, normalized_kappa, 0
                )
                scale = self.collision_culling.point_triangle_measure[contact_id] * self.dhat
                local_hessian = ti.Matrix.zero(self.real_type, 12, 12)
                for row, column in ti.ndrange(12, 12):
                    local_hessian[row, column] = scale * (
                        first * distance_hessian[row, column]
                        + second * distance_gradient[row] * distance_gradient[column]
                    )
                if ti.static(self.contact.project_pd):
                    local_hessian = psd_project_gershgorin_nd(local_hessian)
                for local_i, local_j in ti.static(ti.ndrange(4, 4)):
                    block = ti.Matrix.zero(self.real_type, 3, 3)
                    for row, column in ti.static(ti.ndrange(3, 3)):
                        block[row, column] = local_hessian[3 * local_i + row, 3 * local_j + column]
                    self._add_direct_stencil_block(
                        target,
                        assemble_hash,
                        scalar_base,
                        raw_base,
                        contact_id,
                        ids,
                        local_i,
                        local_j,
                        block,
                    )

    @ti.kernel
    def _scatter_ipc_ee_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        count: ti.i32,
        scalar_base: ti.i32,
        raw_base: ti.i32,
        assemble_hash: ti.template(),
        contact_type: ti.template(),
    ):
        active_gap2 = (2.0 * self.dmin + self.dhat) * self.dhat
        normalized_kappa = self.kappa / (active_gap2 * active_gap2)
        for contact_id in range(count):
            ids = self.collision_culling.edge_edge[contact_id]
            self._initialize_direct_stencil(target, assemble_hash, scalar_base, raw_base, contact_id, ids)
            a0, a1 = positions[ids[0]], positions[ids[1]]
            b0, b1 = positions[ids[2]], positions[ids[3]]
            matches = edge_edge_distance_type(a0, a1, b0, b1) == contact_type
            distance2 = 0.0
            distance_gradient = ti.Vector.zero(self.real_type, 12)
            distance_hessian = ti.Matrix.zero(self.real_type, 12, 12)
            if matches:
                distance2, distance_gradient, distance_hessian = edge_edge_distance_grad_hess_by_type(
                    a0, a1, b0, b1, contact_type
                )
            shifted = distance2 - self.dmin * self.dmin
            if matches and shifted < active_gap2 and shifted > 0.0:
                value, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                    distance2, self.dhat, self.dmin, normalized_kappa, 0
                )
                threshold = edge_edge_mollifier_threshold(
                    self.collision_culling.reference_position[ids[0]],
                    self.collision_culling.reference_position[ids[1]],
                    self.collision_culling.reference_position[ids[2]],
                    self.collision_culling.reference_position[ids[3]],
                )
                mollifier, mollifier_gradient, mollifier_hessian = edge_edge_mollifier_terms(a0, a1, b0, b1, threshold)
                barrier_gradient = first * distance_gradient
                scale = self.collision_culling.edge_edge_measure[contact_id] * self.dhat
                local_hessian = ti.Matrix.zero(self.real_type, 12, 12)
                for row, column in ti.ndrange(12, 12):
                    barrier_hessian = (
                        first * distance_hessian[row, column]
                        + second * distance_gradient[row] * distance_gradient[column]
                    )
                    local_hessian[row, column] = scale * (
                        mollifier * barrier_hessian
                        + value * mollifier_hessian[row, column]
                        + mollifier_gradient[row] * barrier_gradient[column]
                        + barrier_gradient[row] * mollifier_gradient[column]
                    )
                if ti.static(self.contact.project_pd):
                    local_hessian = psd_project_gershgorin_nd(local_hessian)
                for local_i, local_j in ti.static(ti.ndrange(4, 4)):
                    block = ti.Matrix.zero(self.real_type, 3, 3)
                    for row, column in ti.static(ti.ndrange(3, 3)):
                        block[row, column] = local_hessian[3 * local_i + row, 3 * local_j + column]
                    self._add_direct_stencil_block(
                        target,
                        assemble_hash,
                        scalar_base,
                        raw_base,
                        contact_id,
                        ids,
                        local_i,
                        local_j,
                        block,
                    )

    @ti.func
    def _al_frame_from_gradient(self, gradient, edge_edge):
        relative = ti.Vector.zero(self.real_type, 3)
        if edge_edge == 0:
            for component in ti.static(range(3)):
                relative[component] = gradient[component]
        else:
            for component in ti.static(range(3)):
                relative[component] = gradient[component] + gradient[3 + component]
        length = relative.norm()
        normal = ti.Vector([1.0, 0.0, 0.0])
        weights = ti.Vector.zero(self.real_type, 4)
        if edge_edge == 0:
            weights[0] = 1.0
        if length > 1.0e-15:
            normal = relative / length
            for site in ti.static(range(4)):
                site_gradient = ti.Vector([gradient[3 * site + component] for component in ti.static(range(3))])
                weights[site] = site_gradient.dot(normal) / length
        return weights, normal

    @ti.kernel
    def _prepare_al_pt_frames_device(self, positions: ti.template(), count: ti.i32, capacity: ti.i32):
        for contact_id in range(count):
            ids = self.collision_culling.point_triangle[contact_id]
            _, gradient, _ = point_triangle_distance_grad(
                positions[ids[0]],
                positions[ids[1]],
                positions[ids[2]],
                positions[ids[3]],
            )
            weights, normal = self._al_frame_from_gradient(gradient, 0)
            slot = semi_ipc_find_or_insert(
                self.al_pt_hash_state,
                self.al_pt_hash_key,
                self.al_pt_hash_multiplier,
                self.al_pt_hash_count,
                ids,
                capacity,
            )
            multiplier = 0.0
            if slot >= 0:
                previous = self.al_pt_hash_normal[slot]
                if previous.norm_sqr() > 0.0 and previous.dot(normal) < 0.0:
                    normal = -normal
                self.al_pt_hash_weight[slot] = weights
                self.al_pt_hash_normal[slot] = normal
                multiplier = self.al_pt_hash_multiplier[slot]
            self.al_pt_weight[contact_id] = weights
            self.al_pt_normal[contact_id] = normal
            self.al_pt_multiplier[contact_id] = multiplier

    @ti.kernel
    def _prepare_al_ee_frames_device(self, positions: ti.template(), count: ti.i32, capacity: ti.i32):
        for contact_id in range(count):
            ids = self.collision_culling.edge_edge[contact_id]
            _, gradient, _ = edge_edge_distance_grad(
                positions[ids[0]],
                positions[ids[1]],
                positions[ids[2]],
                positions[ids[3]],
            )
            weights, normal = self._al_frame_from_gradient(gradient, 1)
            slot = semi_ipc_find_or_insert(
                self.al_ee_hash_state,
                self.al_ee_hash_key,
                self.al_ee_hash_multiplier,
                self.al_ee_hash_count,
                ids,
                capacity,
            )
            multiplier = 0.0
            if slot >= 0:
                previous = self.al_ee_hash_normal[slot]
                if previous.norm_sqr() > 0.0 and previous.dot(normal) < 0.0:
                    normal = -normal
                self.al_ee_hash_weight[slot] = weights
                self.al_ee_hash_normal[slot] = normal
                multiplier = self.al_ee_hash_multiplier[slot]
            self.al_ee_weight[contact_id] = weights
            self.al_ee_normal[contact_id] = normal
            self.al_ee_multiplier[contact_id] = multiplier

    @ti.kernel
    def _assemble_al_device(
        self,
        positions: ti.template(),
        force: ti.template(),
        candidates: ti.template(),
        measures: ti.template(),
        weights: ti.template(),
        normals: ti.template(),
        multipliers: ti.template(),
        count: ti.i32,
    ):
        penalty = self.penalty_field[None]
        for contact_id in range(count):
            ids = candidates[contact_id]
            gap = -self.constraint_distance
            for site in ti.static(range(4)):
                gap += weights[contact_id][site] * normals[contact_id].dot(positions[ids[site]])
            multiplier = multipliers[contact_id]
            value, derivative, curvature = semi_ipc_terms(gap, multiplier, penalty)
            scale = measures[contact_id]
            ti.atomic_add(self.total_energy[None], scale * value)
            if curvature > 0.0:
                ti.atomic_add(self.active_count[None], 1)
                for site, component in ti.static(ti.ndrange(4, 3)):
                    ti.atomic_add(
                        force[ids[site]][component],
                        scale * derivative * weights[contact_id][site] * normals[contact_id][component],
                    )

    @ti.kernel
    def _scatter_al_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        candidates: ti.template(),
        measures: ti.template(),
        weights: ti.template(),
        normals: ti.template(),
        multipliers: ti.template(),
        count: ti.i32,
        scalar_base: ti.i32,
        raw_base: ti.i32,
        assemble_hash: ti.template(),
    ):
        penalty = self.penalty_field[None]
        for contact_id in range(count):
            ids = candidates[contact_id]
            self._initialize_direct_stencil(target, assemble_hash, scalar_base, raw_base, contact_id, ids)
            gap = -self.constraint_distance
            for site in ti.static(range(4)):
                gap += weights[contact_id][site] * normals[contact_id].dot(positions[ids[site]])
            _, _, curvature = semi_ipc_terms(gap, multipliers[contact_id], penalty)
            if curvature > 0.0:
                for local_i, local_j in ti.static(ti.ndrange(4, 4)):
                    block = (
                        measures[contact_id]
                        * curvature
                        * weights[contact_id][local_i]
                        * weights[contact_id][local_j]
                        * normals[contact_id].outer_product(normals[contact_id])
                    )
                    self._add_direct_stencil_block(
                        target,
                        assemble_hash,
                        scalar_base,
                        raw_base,
                        contact_id,
                        ids,
                        local_i,
                        local_j,
                        block,
                    )

    @ti.kernel
    def _update_al_hash_device(
        self,
        positions: ti.template(),
        states: ti.template(),
        keys: ti.template(),
        weights: ti.template(),
        normals: ti.template(),
        multipliers: ti.template(),
        capacity: ti.i32,
    ):
        penalty = self.penalty_field[None]
        for slot in range(capacity):
            if states[slot] == 0:
                gap = -self.constraint_distance
                for site in ti.static(range(4)):
                    gap += weights[slot][site] * normals[slot].dot(positions[keys[slot][site]])
                multipliers[slot] = semi_ipc_update_multiplier(gap, multipliers[slot], penalty)
                ti.atomic_max(
                    self.constraint_violation_field[None],
                    ti.max(-gap, 0.0),
                )

    @ti.kernel
    def _al_hash_violation_device(
        self,
        positions: ti.template(),
        states: ti.template(),
        keys: ti.template(),
        weights: ti.template(),
        normals: ti.template(),
        capacity: ti.i32,
    ):
        for slot in range(capacity):
            if states[slot] == 0:
                gap = -self.constraint_distance
                for site in ti.static(range(4)):
                    gap += weights[slot][site] * normals[slot].dot(positions[keys[slot][site]])
                ti.atomic_max(
                    self.constraint_violation_field[None],
                    ti.max(-gap, 0.0),
                )

    @ti.kernel
    def _plane_constraint_violation_device(self, positions: ti.template()):
        self.constraint_violation_field[None] = 0.0
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex_id]
            gap = (
                self.plane_normals[plane_id].dot(positions[node] - self.plane_origins[plane_id])
                - self.constraint_distance
            )
            ti.atomic_max(self.constraint_violation_field[None], ti.max(-gap, 0.0))

    @ti.kernel
    def _update_plane_multipliers_device(self, positions: ti.template()):
        penalty = self.penalty_field[None]
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex_id]
            gap = (
                self.plane_normals[plane_id].dot(positions[node] - self.plane_origins[plane_id])
                - self.constraint_distance
            )
            self.plane_multiplier[plane_id, vertex_id] = semi_ipc_update_multiplier(
                gap,
                self.plane_multiplier[plane_id, vertex_id],
                penalty,
            )

    @ti.kernel
    def _plane_ccd_device(
        self,
        positions: ti.template(),
        direction: ti.template(),
        clearance: float,
    ):
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex_id]
            gap = self.plane_normals[plane_id].dot(positions[node] - self.plane_origins[plane_id]) - clearance
            closing = self.plane_normals[plane_id].dot(direction[node])
            if gap <= 0.0 and closing < 0.0:
                ti.atomic_min(self.ccd_alpha[None], 0.0)
            elif closing < 0.0:
                ti.atomic_min(
                    self.ccd_alpha[None],
                    self.contact.ccd_safety * gap / -closing,
                )

    @ti.kernel
    def _copy_friction_hat_device(self, positions: ti.template()):
        for node in range(positions.shape[0]):
            self.friction_hat_position[node] = positions[node]

    @ti.kernel
    def _freeze_pt_friction_device(self, positions: ti.template(), count: ti.i32):
        active_gap2 = (2.0 * self.dmin + self.dhat) * self.dhat
        normalized_kappa = self.kappa / (active_gap2 * active_gap2)
        for contact_id in range(count):
            ids = self.collision_culling.point_triangle[contact_id]
            self.friction_pt_candidate[contact_id] = ids
            distance2, gradient, _ = point_triangle_distance_grad(
                positions[ids[0]],
                positions[ids[1]],
                positions[ids[2]],
                positions[ids[3]],
            )
            relative = ti.Vector([gradient[component] for component in ti.static(range(3))])
            distance = ti.sqrt(ti.max(distance2, 0.0))
            normal = ti.Vector([1.0, 0.0, 0.0])
            weights = ti.Vector([1.0, 0.0, 0.0, 0.0])
            if distance > 1.0e-15:
                normal = relative.normalized()
                denominator = 2.0 * distance
                for site in ti.static(range(4)):
                    site_gradient = ti.Vector([gradient[3 * site + component] for component in ti.static(range(3))])
                    weights[site] = site_gradient.dot(normal) / denominator
            if ti.static(self.is_augmented_lagrangian):
                weights = self.al_pt_weight[contact_id]
                normal = self.al_pt_normal[contact_id]
            self.friction_pt_weight[contact_id] = weights
            self.friction_pt_normal[contact_id] = normal
            normal_force = 0.0
            shifted = distance2 - self.dmin * self.dmin
            if ti.static(self.is_augmented_lagrangian):
                gap = -self.constraint_distance
                for site in ti.static(range(4)):
                    gap += weights[site] * normal.dot(positions[ids[site]])
                normal_force = self.collision_culling.point_triangle_measure[contact_id] * ti.max(
                    self.al_pt_multiplier[contact_id] - self.penalty_field[None] * gap,
                    0.0,
                )
            elif shifted > 0.0 and shifted < active_gap2:
                _, first, _ = ipc_toolkit_barrier_distance2_offset_terms(
                    distance2,
                    self.dhat,
                    self.dmin,
                    normalized_kappa,
                    0,
                )
                scale = self.collision_culling.point_triangle_measure[contact_id] * self.dhat
                normal_force = ti.max(0.0, -scale * first * 2.0 * ti.sqrt(shifted))
            self.friction_pt_normal_force[contact_id] = normal_force

    @ti.kernel
    def _freeze_ee_friction_device(self, positions: ti.template(), count: ti.i32):
        active_gap2 = (2.0 * self.dmin + self.dhat) * self.dhat
        normalized_kappa = self.kappa / (active_gap2 * active_gap2)
        for contact_id in range(count):
            ids = self.collision_culling.edge_edge[contact_id]
            self.friction_ee_candidate[contact_id] = ids
            a0, a1 = positions[ids[0]], positions[ids[1]]
            b0, b1 = positions[ids[2]], positions[ids[3]]
            distance2, gradient, _ = edge_edge_distance_grad(a0, a1, b0, b1)
            relative = ti.Vector.zero(self.real_type, 3)
            for component in ti.static(range(3)):
                relative[component] = gradient[component] + gradient[3 + component]
            distance = ti.sqrt(ti.max(distance2, 0.0))
            normal = ti.Vector([1.0, 0.0, 0.0])
            weights = ti.Vector.zero(self.real_type, 4)
            if distance > 1.0e-15:
                normal = relative.normalized()
                denominator = 2.0 * distance
                for site in ti.static(range(4)):
                    site_gradient = ti.Vector([gradient[3 * site + component] for component in ti.static(range(3))])
                    weights[site] = site_gradient.dot(normal) / denominator
            if ti.static(self.is_augmented_lagrangian):
                weights = self.al_ee_weight[contact_id]
                normal = self.al_ee_normal[contact_id]
            self.friction_ee_weight[contact_id] = weights
            self.friction_ee_normal[contact_id] = normal
            normal_force = 0.0
            shifted = distance2 - self.dmin * self.dmin
            if ti.static(self.is_augmented_lagrangian):
                gap = -self.constraint_distance
                for site in ti.static(range(4)):
                    gap += weights[site] * normal.dot(positions[ids[site]])
                normal_force = self.collision_culling.edge_edge_measure[contact_id] * ti.max(
                    self.al_ee_multiplier[contact_id] - self.penalty_field[None] * gap,
                    0.0,
                )
            elif shifted > 0.0 and shifted < active_gap2:
                ra0 = self.collision_culling.reference_position[ids[0]]
                ra1 = self.collision_culling.reference_position[ids[1]]
                rb0 = self.collision_culling.reference_position[ids[2]]
                rb1 = self.collision_culling.reference_position[ids[3]]
                threshold = edge_edge_mollifier_threshold(ra0, ra1, rb0, rb1)
                mollifier, mollifier_gradient, mollifier_hessian = edge_edge_mollifier_terms(a0, a1, b0, b1, threshold)
                if mollifier >= 1.0 - 1.0e-12:
                    _, first, _ = ipc_toolkit_barrier_distance2_offset_terms(
                        distance2,
                        self.dhat,
                        self.dmin,
                        normalized_kappa,
                        0,
                    )
                    scale = self.collision_culling.edge_edge_measure[contact_id] * self.dhat
                    normal_force = ti.max(0.0, -scale * first * 2.0 * ti.sqrt(shifted))
            self.friction_ee_normal_force[contact_id] = normal_force

    @ti.kernel
    def _freeze_plane_friction_device(self, positions: ti.template()):
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex_id]
            distance = self.plane_normals[plane_id].dot(positions[node] - self.plane_origins[plane_id])
            gap = distance - self.dmin
            normal_force = 0.0
            if ti.static(self.is_augmented_lagrangian):
                gap = distance - self.constraint_distance
                normal_force = self.surface_vertex_measure[vertex_id] * ti.max(
                    self.plane_multiplier[plane_id, vertex_id] - self.penalty_field[None] * gap,
                    0.0,
                )
            elif gap > 0.0 and gap < self.dhat:
                _, derivative, _ = _ipc_barrier_terms(gap, self.dhat, self.kappa)
                normal_force = ti.max(
                    0.0,
                    -self.surface_vertex_measure[vertex_id] * self.dhat * derivative,
                )
            self.plane_friction_normal_force[plane_id, vertex_id] = normal_force

    @ti.func
    def _assemble_friction_contact_device(
        self,
        positions: ti.template(),
        force: ti.template(),
        ids,
        weights,
        normal,
        normal_force,
    ):
        identity = ti.Matrix.identity(self.real_type, 3)
        timestep = self.friction_dt_field[None]
        coefficient = self.contact.friction_coefficient * normal_force
        if coefficient > 0.0:
            tangent = identity - normal.outer_product(normal)
            relative_increment = ti.Vector.zero(self.real_type, 3)
            for site in ti.static(range(4)):
                relative_increment += weights[site] * (positions[ids[site]] - self.friction_hat_position[ids[site]])
            velocity = tangent @ relative_increment / timestep
            speed = velocity.norm()
            ti.atomic_add(
                self.total_energy[None],
                coefficient * ipc_friction_f0(speed, self.contact.epsv, timestep),
            )
            profile = ipc_friction_f1_over_speed(speed, self.contact.epsv)
            relative_gradient = coefficient * profile * (tangent @ velocity)
            for site, component in ti.static(ti.ndrange(4, 3)):
                ti.atomic_add(
                    force[ids[site]][component],
                    weights[site] * relative_gradient[component],
                )

    @ti.kernel
    def _assemble_pt_friction_device(
        self,
        positions: ti.template(),
        force: ti.template(),
        count: ti.i32,
    ):
        for contact_id in range(count):
            self._assemble_friction_contact_device(
                positions,
                force,
                self.friction_pt_candidate[contact_id],
                self.friction_pt_weight[contact_id],
                self.friction_pt_normal[contact_id],
                self.friction_pt_normal_force[contact_id],
            )

    @ti.kernel
    def _assemble_ee_friction_device(
        self,
        positions: ti.template(),
        force: ti.template(),
        count: ti.i32,
    ):
        for contact_id in range(count):
            self._assemble_friction_contact_device(
                positions,
                force,
                self.friction_ee_candidate[contact_id],
                self.friction_ee_weight[contact_id],
                self.friction_ee_normal[contact_id],
                self.friction_ee_normal_force[contact_id],
            )

    @ti.func
    def _scatter_friction_contact_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        scalar_base: ti.i32,
        raw_base: ti.i32,
        contact_id,
        ids,
        weights,
        normal,
        normal_force,
        assemble_hash: ti.template(),
    ):
        identity = ti.Matrix.identity(self.real_type, 3)
        timestep = self.friction_dt_field[None]
        self._initialize_direct_stencil(target, assemble_hash, scalar_base, raw_base, contact_id, ids)
        coefficient = self.contact.friction_coefficient * normal_force
        if coefficient > 0.0:
            tangent = identity - normal.outer_product(normal)
            relative_increment = ti.Vector.zero(self.real_type, 3)
            for site in ti.static(range(4)):
                relative_increment += weights[site] * (positions[ids[site]] - self.friction_hat_position[ids[site]])
            velocity = tangent @ relative_increment / timestep
            speed = velocity.norm()
            profile = ipc_friction_f1_over_speed(speed, self.contact.epsv)
            inner = coefficient * profile * identity
            if speed > 0.0:
                inner += (
                    coefficient
                    * ipc_friction_hessian_term(speed, self.contact.epsv)
                    / speed
                    * velocity.outer_product(velocity)
                )
            relative_hessian = tangent @ inner @ tangent / timestep
            if ti.static(self.contact.project_pd):
                relative_hessian = psd_project_nd(relative_hessian)
            for local_i, local_j in ti.static(ti.ndrange(4, 4)):
                block = weights[local_i] * weights[local_j] * relative_hessian
                self._add_direct_stencil_block(
                    target,
                    assemble_hash,
                    scalar_base,
                    raw_base,
                    contact_id,
                    ids,
                    local_i,
                    local_j,
                    block,
                )

    @ti.kernel
    def _scatter_pt_friction_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        count: ti.i32,
        scalar_base: ti.i32,
        raw_base: ti.i32,
        assemble_hash: ti.template(),
    ):
        for contact_id in range(count):
            self._scatter_friction_contact_direct(
                positions,
                target,
                scalar_base,
                raw_base,
                contact_id,
                self.friction_pt_candidate[contact_id],
                self.friction_pt_weight[contact_id],
                self.friction_pt_normal[contact_id],
                self.friction_pt_normal_force[contact_id],
                assemble_hash,
            )

    @ti.kernel
    def _scatter_ee_friction_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        count: ti.i32,
        scalar_base: ti.i32,
        raw_base: ti.i32,
        assemble_hash: ti.template(),
    ):
        for contact_id in range(count):
            self._scatter_friction_contact_direct(
                positions,
                target,
                scalar_base,
                raw_base,
                contact_id,
                self.friction_ee_candidate[contact_id],
                self.friction_ee_weight[contact_id],
                self.friction_ee_normal[contact_id],
                self.friction_ee_normal_force[contact_id],
                assemble_hash,
            )

    @ti.kernel
    def _assemble_plane_friction_device(
        self,
        positions: ti.template(),
        force: ti.template(),
    ):
        identity = ti.Matrix.identity(self.real_type, 3)
        timestep = self.friction_dt_field[None]
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            coefficient = self.contact.friction_coefficient * self.plane_friction_normal_force[plane_id, vertex_id]
            if coefficient > 0.0:
                node = self.surface_vertices[vertex_id]
                normal = self.plane_normals[plane_id]
                tangent = identity - normal.outer_product(normal)
                increment = positions[node] - self.friction_hat_position[node]
                velocity = tangent @ increment / timestep
                speed = velocity.norm()
                ti.atomic_add(
                    self.total_energy[None],
                    coefficient * ipc_friction_f0(speed, self.contact.epsv, timestep),
                )
                profile = ipc_friction_f1_over_speed(speed, self.contact.epsv)
                gradient = coefficient * profile * (tangent @ velocity)
                for component in ti.static(range(3)):
                    ti.atomic_add(force[node][component], gradient[component])

    @ti.kernel
    def _scatter_planes_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        scalar_base: ti.i32,
        assemble_hash: ti.template(),
    ):
        identity = ti.Matrix.identity(self.real_type, 3)
        timestep = self.friction_dt_field[None]
        penalty = self.penalty_field[None]
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            flat_id = plane_id * self.surface.vertices.size + vertex_id
            node = self.surface_vertices[vertex_id]
            self._initialize_direct_plane_block(target, assemble_hash, scalar_base, flat_id, node)
            normal = self.plane_normals[plane_id]
            distance = normal.dot(positions[node] - self.plane_origins[plane_id])
            measure = self.surface_vertex_measure[vertex_id]
            block = ti.Matrix.zero(self.real_type, 3, 3)
            if ti.static(self.is_ipc):
                gap = distance - self.dmin
                if gap < self.dhat:
                    _, _, curvature = _ipc_barrier_terms(gap, self.dhat, self.kappa)
                    block += measure * self.dhat * curvature * normal.outer_product(normal)
            else:
                gap = distance - self.constraint_distance
                _, _, curvature = semi_ipc_terms(
                    gap,
                    self.plane_multiplier[plane_id, vertex_id],
                    penalty,
                )
                if curvature > 0.0:
                    block += measure * curvature * normal.outer_product(normal)
            coefficient = self.contact.friction_coefficient * self.plane_friction_normal_force[plane_id, vertex_id]
            if coefficient > 0.0:
                tangent = identity - normal.outer_product(normal)
                velocity = tangent @ (positions[node] - self.friction_hat_position[node]) / timestep
                speed = velocity.norm()
                profile = ipc_friction_f1_over_speed(speed, self.contact.epsv)
                inner = coefficient * profile * identity
                if speed > 0.0:
                    inner += (
                        coefficient
                        * ipc_friction_hessian_term(speed, self.contact.epsv)
                        / speed
                        * velocity.outer_product(velocity)
                    )
                block += tangent @ inner @ tangent / timestep
            self._add_direct_plane_block(target, assemble_hash, scalar_base, flat_id, node, block)

    def _require_device_contact_capacity(self, pt_count, ee_count):
        if int(pt_count) > self.device_pt_capacity:
            raise RuntimeError(
                "point-triangle contact capacity is too small: "
                f"need {int(pt_count)}, allocated {self.device_pt_capacity}; "
                "increase point_triangle_coordination_number or max_point_triangle_pairs"
            )
        if int(ee_count) > self.device_ee_capacity:
            raise RuntimeError(
                "edge-edge contact capacity is too small: "
                f"need {int(ee_count)}, allocated {self.device_ee_capacity}; "
                "increase edge_edge_coordination_number or max_edge_edge_pairs"
            )

    def _require_al_hash_capacity(self, pt_count, ee_count):
        required = 2 * max(
            int(self.al_pt_hash_count[None]) + int(pt_count),
            int(self.al_ee_hash_count[None]) + int(ee_count),
            1,
        )
        if required > self.al_hash_capacity:
            raise RuntimeError(
                "augmented-Lagrangian contact hash capacity is too small: "
                f"need {required}, allocated {self.al_hash_capacity}; increase the "
                "contact coordination numbers or explicit pair capacities"
            )

    def _require_device_friction_capacity(self, pt_count, ee_count):
        if int(pt_count) > self.friction_pt_capacity:
            raise RuntimeError(
                "point-triangle friction capacity is too small: "
                f"need {int(pt_count)}, allocated {self.friction_pt_capacity}; "
                "increase point_triangle_coordination_number or max_point_triangle_pairs"
            )
        if int(ee_count) > self.friction_ee_capacity:
            raise RuntimeError(
                "edge-edge friction capacity is too small: "
                f"need {int(ee_count)}, allocated {self.friction_ee_capacity}; "
                "increase edge_edge_coordination_number or max_edge_edge_pairs"
            )

    def prepare_iteration_device(self, positions, end_positions=None):
        if self.contact.self_contact:
            if end_positions is None:
                pt_count, ee_count = self.collision_culling.rebuild_proximity(positions, self.dmin + self.dhat)
            else:
                pt_count, ee_count = self.collision_culling.rebuild_swept_candidates(
                    positions,
                    end_positions,
                    self.dmin + self.dhat,
                )
            self._require_device_contact_capacity(pt_count, ee_count)
            if self.is_augmented_lagrangian:
                self._require_al_hash_capacity(pt_count, ee_count)
                self._prepare_al_pt_frames_device(positions, int(pt_count), self.al_hash_capacity)
                self._prepare_al_ee_frames_device(positions, int(ee_count), self.al_hash_capacity)
        else:
            self.collision_culling.clear_active()
        if self.is_augmented_lagrangian:
            self._plane_constraint_violation_device(positions)
            if self.contact.self_contact:
                self._al_hash_violation_device(
                    positions,
                    self.al_pt_hash_state,
                    self.al_pt_hash_key,
                    self.al_pt_hash_weight,
                    self.al_pt_hash_normal,
                    self.al_hash_capacity,
                )
                self._al_hash_violation_device(
                    positions,
                    self.al_ee_hash_state,
                    self.al_ee_hash_key,
                    self.al_ee_hash_weight,
                    self.al_ee_hash_normal,
                    self.al_hash_capacity,
                )

    def begin_step_device(self, positions, dt):
        if self.is_augmented_lagrangian:
            self.al_pt_hash_state.fill(2)
            self.al_ee_hash_state.fill(2)
            self.al_pt_hash_multiplier.fill(0.0)
            self.al_ee_hash_multiplier.fill(0.0)
            self.al_pt_hash_count[None] = 0
            self.al_ee_hash_count[None] = 0
            self.plane_multiplier.fill(0.0)
        self.friction_dt = float(dt)
        if not np.isfinite(self.friction_dt) or self.friction_dt <= 0.0:
            raise ValueError("FEM contact time step must be finite and positive")
        self.friction_dt_field[None] = self.friction_dt
        if self.contact.friction_coefficient > 0.0:
            self._copy_friction_hat_device(positions)
        self.refresh_friction_device(positions)

    def refresh_friction_device(self, positions):
        """Refresh lagged frames and normal forces without changing step hats."""
        self.friction_pt_count = 0
        self.friction_ee_count = 0
        if self.contact.friction_coefficient <= 0.0:
            return
        if self.contact.self_contact:
            if self.is_augmented_lagrangian:
                self.prepare_iteration_device(positions)
                pt_count = int(self.collision_culling.point_triangle_count[None])
                ee_count = int(self.collision_culling.edge_edge_count[None])
            else:
                pt_count, ee_count = self.collision_culling.rebuild_proximity(positions, self.dmin + self.dhat)
            self._require_device_friction_capacity(pt_count, ee_count)
            self.friction_pt_count = int(pt_count)
            self.friction_ee_count = int(ee_count)
            self._freeze_pt_friction_device(positions, self.friction_pt_count)
            self._freeze_ee_friction_device(positions, self.friction_ee_count)
        if self.plane_count:
            self._freeze_plane_friction_device(positions)

    @property
    def activate_friction(self):
        return self.contact.friction_coefficient > 0.0

    def assemble_device(
        self,
        positions,
        force,
        stiffness,
        *,
        need_stiffness=False,
    ):
        self.total_energy[None] = 0.0
        self.active_count[None] = 0
        pt_count = int(self.collision_culling.point_triangle_count[None])
        ee_count = int(self.collision_culling.edge_edge_count[None])
        if self.contact.self_contact and self.is_ipc:
            self._assemble_ipc_pt_device(positions, force, pt_count)
            self._assemble_ipc_ee_device(positions, force, ee_count)
        if self.contact.self_contact and self.is_augmented_lagrangian:
            self._assemble_al_device(
                positions,
                force,
                self.collision_culling.point_triangle,
                self.collision_culling.point_triangle_measure,
                self.al_pt_weight,
                self.al_pt_normal,
                self.al_pt_multiplier,
                pt_count,
            )
            self._assemble_al_device(
                positions,
                force,
                self.collision_culling.edge_edge,
                self.collision_culling.edge_edge_measure,
                self.al_ee_weight,
                self.al_ee_normal,
                self.al_ee_multiplier,
                ee_count,
            )
        if self.plane_count:
            self._assemble_planes_device(positions, force)
        if self.contact.friction_coefficient > 0.0:
            if self.friction_pt_count:
                self._assemble_pt_friction_device(
                    positions,
                    force,
                    self.friction_pt_count,
                )
            if self.friction_ee_count:
                self._assemble_ee_friction_device(
                    positions,
                    force,
                    self.friction_ee_count,
                )
            if self.plane_count:
                self._assemble_plane_friction_device(positions, force)
        if need_stiffness:
            self._copy_stiffness_positions(positions)
            self._contact_feature_mask_value = int(self._contact_feature_mask(pt_count, ee_count))
            stiffness.add_device_contribution(self)
        self._active_contacts = int(self.active_count[None])
        self._last_energy = float(self.total_energy[None])
        return force, stiffness

    def stiffness_entry_count_device(self):
        return 144 * (
            int(self.collision_culling.point_triangle_count[None])
            + int(self.collision_culling.edge_edge_count[None])
            + self.friction_pt_count
            + self.friction_ee_count
        ) + 9 * self.plane_count * int(self.surface.vertices.size)

    def stiffness_block_pair_count_device(self):
        return 12 * (
            int(self.collision_culling.point_triangle_count[None])
            + int(self.collision_culling.edge_edge_count[None])
            + self.friction_pt_count
            + self.friction_ee_count
        )

    def scatter_stiffness_to_coo(self, matrix, offset=0):
        self._scatter_stiffness_direct(matrix, int(offset), 0, False)

    def scatter_stiffness_to_hash(self, matrix):
        raw_offset = matrix.reserve_raw_block_slots(self.stiffness_block_pair_count_device())
        self._scatter_stiffness_direct(matrix, 0, raw_offset, True)

    def _scatter_stiffness_direct(self, matrix, scalar_offset, raw_offset, assemble_hash):
        pt_count = int(self.collision_culling.point_triangle_count[None])
        ee_count = int(self.collision_culling.edge_edge_count[None])
        ee_scalar = scalar_offset + 144 * pt_count
        friction_pt_scalar = ee_scalar + 144 * ee_count
        friction_ee_scalar = friction_pt_scalar + 144 * self.friction_pt_count
        plane_scalar = friction_ee_scalar + 144 * self.friction_ee_count
        ee_raw = raw_offset + 12 * pt_count
        friction_pt_raw = ee_raw + 12 * ee_count
        friction_ee_raw = friction_pt_raw + 12 * self.friction_pt_count
        if self.contact.self_contact and self.is_ipc:
            for contact_type in range(7):
                if self._contact_feature_mask_value & (1 << contact_type):
                    self._scatter_ipc_pt_direct(
                        self.stiffness_positions,
                        matrix,
                        pt_count,
                        scalar_offset,
                        raw_offset,
                        assemble_hash,
                        contact_type,
                    )
            for contact_type in range(9):
                if self._contact_feature_mask_value & (1 << (7 + contact_type)):
                    self._scatter_ipc_ee_direct(
                        self.stiffness_positions,
                        matrix,
                        ee_count,
                        ee_scalar,
                        ee_raw,
                        assemble_hash,
                        contact_type,
                    )
        elif self.contact.self_contact:
            self._scatter_al_direct(
                self.stiffness_positions,
                matrix,
                self.collision_culling.point_triangle,
                self.collision_culling.point_triangle_measure,
                self.al_pt_weight,
                self.al_pt_normal,
                self.al_pt_multiplier,
                pt_count,
                scalar_offset,
                raw_offset,
                assemble_hash,
            )
            self._scatter_al_direct(
                self.stiffness_positions,
                matrix,
                self.collision_culling.edge_edge,
                self.collision_culling.edge_edge_measure,
                self.al_ee_weight,
                self.al_ee_normal,
                self.al_ee_multiplier,
                ee_count,
                ee_scalar,
                ee_raw,
                assemble_hash,
            )
        if self.friction_pt_count:
            self._scatter_pt_friction_direct(
                self.stiffness_positions,
                matrix,
                self.friction_pt_count,
                friction_pt_scalar,
                friction_pt_raw,
                assemble_hash,
            )
        if self.friction_ee_count:
            self._scatter_ee_friction_direct(
                self.stiffness_positions,
                matrix,
                self.friction_ee_count,
                friction_ee_scalar,
                friction_ee_raw,
                assemble_hash,
            )
        if self.plane_count:
            self._scatter_planes_direct(self.stiffness_positions, matrix, plane_scalar, assemble_hash)

    def maximum_admissible_step_device(self, positions, direction):
        if not (self.is_ipc or self.is_augmented_lagrangian):
            return 1.0
        self.ccd_alpha[None] = 1.0
        clearance = 0.0 if self.is_augmented_lagrangian else self.dmin
        if self.contact.self_contact:
            self._build_swept_end(positions, direction, self.device_end_positions)
            _, _, minimum_step = self.collision_culling.compute_ccd(
                positions,
                self.device_end_positions,
                self.dmin + self.dhat,
                eta=1.0 - self.contact.ccd_safety,
                thickness=clearance,
                max_iterations=self.contact.ccd_max_iterations,
            )
            self.ccd_alpha[None] = minimum_step
        if self.plane_count:
            self._plane_ccd_device(positions, direction, clearance)
        return max(0.0, min(1.0, float(self.ccd_alpha[None])))

    @ti.kernel
    def _build_swept_end(
        self,
        positions: ti.template(),
        direction: ti.template(),
        end_positions: ti.template(),
    ):
        for node in range(positions.shape[0]):
            end_positions[node] = positions[node] + direction[node]

    def accept_update_device(self, positions, step=1.0):
        if not self.is_augmented_lagrangian:
            return
        self._plane_constraint_violation_device(positions)
        violation = float(self.constraint_violation_field[None])
        self._update_plane_multipliers_device(positions)
        if self.contact.self_contact:
            self._update_al_hash_device(
                positions,
                self.al_pt_hash_state,
                self.al_pt_hash_key,
                self.al_pt_hash_weight,
                self.al_pt_hash_normal,
                self.al_pt_hash_multiplier,
                self.al_hash_capacity,
            )
            self._update_al_hash_device(
                positions,
                self.al_ee_hash_state,
                self.al_ee_hash_key,
                self.al_ee_hash_weight,
                self.al_ee_hash_normal,
                self.al_ee_hash_multiplier,
                self.al_hash_capacity,
            )
            violation = float(self.constraint_violation_field[None])
        # SemiIPC raises mu only after sustained projected-step stagnation.
        self._stalled_updates = self._stalled_updates + 1 if step < 1.0e-4 else 0
        if (
            self._stalled_updates >= self.contact.penalty_update_interval
            and violation > self.contact.sufficient_reduction * self._previous_violation
        ):
            self.penalty = min(
                self.contact.max_penalty,
                self.penalty * self.contact.penalty_growth,
            )
            self.penalty_field[None] = self.penalty
            self._stalled_updates = 0
        self._previous_violation = violation

    def converged_device(self, positions):
        if not self.is_augmented_lagrangian:
            return True
        self._plane_constraint_violation_device(positions)
        if self.contact.self_contact:
            self._al_hash_violation_device(
                positions,
                self.al_pt_hash_state,
                self.al_pt_hash_key,
                self.al_pt_hash_weight,
                self.al_pt_hash_normal,
                self.al_hash_capacity,
            )
            self._al_hash_violation_device(
                positions,
                self.al_ee_hash_state,
                self.al_ee_hash_key,
                self.al_ee_hash_weight,
                self.al_ee_hash_normal,
                self.al_hash_capacity,
            )
        return float(self.constraint_violation_field[None]) <= self.contact.constraint_tolerance

    def prepared_converged_device(self):
        """Read convergence after ``prepare_iteration_device``.

        The preparation pass has already reduced plane and self-contact
        violations.  Newton convergence checks must not launch the same
        reduction kernels a second time at an unchanged iterate.
        """
        return (
            not self.is_augmented_lagrangian
            or float(self.constraint_violation_field[None]) <= self.contact.constraint_tolerance
        )

    def device_diagnostics(self):
        violation = 0.0
        if self.is_augmented_lagrangian:
            violation = float(self.constraint_violation_field[None])
        return {
            **self.collision_culling.diagnostics(),
            "model": self.contact.model,
            "constraint_violation": violation,
            "contact_violation": violation,
            "penalty": float(self.penalty_field[None]),
            "contact_penalty": float(self.penalty_field[None]),
            "active_contacts": int(self.active_count[None]),
        }

    def assemble_output(self, positions, mechanical, need_stiffness=False):
        energy, force, stiffness, stress = mechanical
        contact_energy, contact_force, contact_stiffness = self.assemble(positions, need_stiffness=need_stiffness)
        energy += contact_energy
        force = force + contact_force
        if need_stiffness:
            if hasattr(stiffness, "add_contribution"):
                stiffness.add_contribution(contact_stiffness)
            else:
                stiffness = stiffness + contact_stiffness.to_scipy()
        return energy, force, stiffness, stress

    def _plane_data(self):
        if not self.contact.planes:
            return None
        origins = np.ascontiguousarray([entry[0] for entry in self.contact.planes], dtype=np.float64)
        normals = np.ascontiguousarray([entry[1] for entry in self.contact.planes], dtype=np.float64)
        vertices = np.ascontiguousarray(self.surface.vertices, dtype=np.int32)
        measures = np.ascontiguousarray(self.surface.node_area[vertices], dtype=np.float64)
        multipliers = np.zeros((origins.shape[0], vertices.shape[0]), dtype=np.float64)
        if self.is_augmented_lagrangian:
            for plane_id in range(origins.shape[0]):
                for vertex_id, node in enumerate(vertices):
                    multipliers[plane_id, vertex_id] = self._al_state.get(("plane", plane_id, int(node)), 0.0)
        return vertices, origins, normals, measures, multipliers

    def assemble(self, positions, need_stiffness=False):
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        force = np.zeros_like(positions)
        self.total_energy[None] = 0.0
        self.active_count[None] = 0
        plane_data = self._plane_data()
        stencils = (
            self.pt_candidates,
            self.ee_candidates,
            self.friction_candidates_pt,
            self.friction_candidates_ee,
        )
        stiffness = None
        if need_stiffness:
            row_parts = []
            column_parts = []
            for stencil in stencils:
                row = 3 * stencil[:, :, None, None, None] + np.arange(3)[None, None, None, :, None]
                column = 3 * stencil[:, None, :, None, None] + np.arange(3)[None, None, None, None, :]
                shape = (stencil.shape[0], 4, 4, 3, 3)
                row_parts.append(np.broadcast_to(row, shape).reshape(-1))
                column_parts.append(np.broadcast_to(column, shape).reshape(-1))
            if plane_data is not None:
                vertices, origins, _, _, _ = plane_data
                plane_nodes = np.tile(vertices, origins.shape[0])
                row_parts.append(
                    (3 * plane_nodes[:, None, None] + np.arange(3)[None, :, None]).repeat(3, axis=2).reshape(-1)
                )
                column_parts.append(
                    (3 * plane_nodes[:, None, None] + np.arange(3)[None, None, :]).repeat(3, axis=1).reshape(-1)
                )
            rows = np.concatenate(row_parts).astype(np.int32, copy=False)
            columns = np.concatenate(column_parts).astype(np.int32, copy=False)
            stiffness = FEMTripletContribution(
                3 * self.mesh.number_of_nodes,
                rows,
                columns,
                np.zeros(rows.size, dtype=np.float64),
            )
            cursor = 0
            hessians = []
            for stencil in stencils:
                size = 144 * stencil.shape[0]
                hessians.append(stiffness.values[cursor : cursor + size].reshape(stencil.shape[0], 4, 4, 3, 3))
                cursor += size
            plane_hessian = stiffness.values[cursor:].reshape(-1, 3, 3)
        else:
            dummy = np.zeros((1, 4, 4, 3, 3), dtype=np.float64)
            hessians = (dummy, dummy, dummy, dummy)
            plane_hessian = np.zeros((1, 3, 3), dtype=np.float64)
        pt_hessian, ee_hessian, friction_pt_hessian, friction_ee_hessian = hessians
        if self.is_ipc:
            self._assemble_ipc_pt(
                positions, self.pt_candidates, self.pt_measures, force, pt_hessian, bool(need_stiffness)
            )
            self._assemble_ipc_ee(
                positions,
                self.reference_positions,
                self.ee_candidates,
                self.ee_measures,
                force,
                ee_hessian,
                bool(need_stiffness),
            )
        else:
            self._assemble_al(
                positions,
                self.pt_candidates,
                self.pt_measures,
                np.ascontiguousarray(self.pt_weights),
                np.ascontiguousarray(self.pt_normals),
                np.ascontiguousarray(self.pt_lambdas),
                force,
                pt_hessian,
                int(need_stiffness),
            )
            self._assemble_al(
                positions,
                self.ee_candidates,
                self.ee_measures,
                np.ascontiguousarray(self.ee_weights),
                np.ascontiguousarray(self.ee_normals),
                np.ascontiguousarray(self.ee_lambdas),
                force,
                ee_hessian,
                int(need_stiffness),
            )
        if self.contact.friction_coefficient > 0.0:
            if self.friction_candidates_pt.shape[0]:
                self._assemble_friction(
                    positions,
                    self.friction_hat_positions,
                    self.friction_candidates_pt,
                    np.ascontiguousarray(self.friction_weights_pt),
                    np.ascontiguousarray(self.friction_normals_pt),
                    np.ascontiguousarray(self.friction_lambdas_pt),
                    force,
                    friction_pt_hessian,
                    int(need_stiffness),
                )
            if self.friction_candidates_ee.shape[0]:
                self._assemble_friction(
                    positions,
                    self.friction_hat_positions,
                    self.friction_candidates_ee,
                    np.ascontiguousarray(self.friction_weights_ee),
                    np.ascontiguousarray(self.friction_normals_ee),
                    np.ascontiguousarray(self.friction_lambdas_ee),
                    force,
                    friction_ee_hessian,
                    int(need_stiffness),
                )
        if plane_data is not None:
            vertices, origins, normals, measures, multipliers = plane_data
            self._assemble_planes(
                positions,
                vertices,
                origins,
                normals,
                measures,
                multipliers,
                force,
                plane_hessian,
                int(need_stiffness),
                int(self.is_ipc),
            )
            if self.contact.friction_coefficient > 0.0:
                self._assemble_plane_friction(
                    positions,
                    self.friction_hat_positions,
                    vertices,
                    normals,
                    np.ascontiguousarray(self.plane_friction_lambdas),
                    force,
                    plane_hessian,
                    int(need_stiffness),
                )
        self._active_contacts = int(self.active_count[None])
        self._last_energy = float(self.total_energy[None])
        if need_stiffness and self.is_ipc and not np.isfinite(self._last_energy):
            raise RuntimeError(
                "the current FEM IPC state violates dmin; IPC requires an initially feasible configuration"
            )
        return self._last_energy, force, stiffness

    def maximum_admissible_step(self, positions, direction):
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        direction = np.ascontiguousarray(direction, dtype=np.float64)
        end_positions = np.ascontiguousarray(positions + direction)
        self.device_positions.from_numpy(positions)
        self.device_end_positions.from_numpy(end_positions)
        alpha = 1.0
        clearance = self.dmin if self.is_ipc else 0.0
        if self.contact.self_contact:
            _, _, alpha = self.collision_culling.compute_ccd(
                self.device_positions,
                self.device_end_positions,
                self.dmin + self.dhat,
                eta=1.0 - self.contact.ccd_safety,
                thickness=clearance,
                max_iterations=self.contact.ccd_max_iterations,
            )
        for origin, normal in self.contact.planes:
            gap = (positions[self.surface.vertices] - origin) @ normal - clearance
            closing = direction[self.surface.vertices] @ normal
            mask = (gap > 0.0) & (closing < 0.0)
            if np.any(mask):
                alpha = min(alpha, float(np.min(self.contact.ccd_safety * gap[mask] / -closing[mask])))
            if np.any((gap <= 0.0) & (closing < 0.0)):
                alpha = 0.0
        return float(np.clip(alpha, 0.0, 1.0))

    def _al_constraint_arrays(self, positions):
        values = []
        descriptors = []
        for prefix, candidates, weights, normals in (
            ("pt", self.pt_candidates, self.pt_weights, self.pt_normals),
            ("ee", self.ee_candidates, self.ee_weights, self.ee_normals),
        ):
            if candidates.shape[0]:
                relative = np.einsum("ij,ijk->ik", weights, positions[candidates])
                gaps = np.einsum("ij,ij->i", relative, normals) - self.constraint_distance
                values.extend(gaps.tolist())
                descriptors.extend(self._key(prefix, stencil) for stencil in candidates)
        for plane_id, (origin, normal) in enumerate(self.contact.planes):
            gaps = (positions[self.surface.vertices] - origin) @ normal - self.constraint_distance
            values.extend(gaps.tolist())
            descriptors.extend(("plane", plane_id, int(node)) for node in self.surface.vertices)
        return np.asarray(values, dtype=np.float64), descriptors

    def constraint_violation(self, positions):
        if not self.is_augmented_lagrangian:
            return 0.0
        gaps, _ = self._al_constraint_arrays(np.asarray(positions, dtype=np.float64))
        return 0.0 if gaps.size == 0 else float(np.max(np.maximum(-gaps, 0.0)))

    def accept_update(self, positions, step=1.0):
        if not self.is_augmented_lagrangian:
            return
        gaps, descriptors = self._al_constraint_arrays(np.asarray(positions, dtype=np.float64))
        for gap, key in zip(gaps, descriptors):
            multiplier = self._al_state.get(key, 0.0)
            updated = semi_ipc_update_multiplier_py(gap, multiplier, self.penalty)
            if updated > 0.0:
                self._al_state[key] = updated
            elif key in self._al_state:
                del self._al_state[key]
        violation = 0.0 if gaps.size == 0 else float(np.max(np.maximum(-gaps, 0.0)))
        self._stalled_updates = self._stalled_updates + 1 if step < 1.0e-4 else 0
        if (
            self._stalled_updates >= self.contact.penalty_update_interval
            and np.isfinite(self._previous_violation)
            and violation > self.contact.sufficient_reduction * self._previous_violation
        ):
            self.penalty = min(self.contact.max_penalty, self.contact.penalty_growth * self.penalty)
            self.penalty_field[None] = self.penalty
            self._stalled_updates = 0
        self._previous_violation = violation
        self._last_violation = violation

    def converged(self, positions):
        if self.is_ipc:
            return np.isfinite(self._last_energy)
        self._last_violation = self.constraint_violation(positions)
        return self._last_violation <= self.contact.constraint_tolerance

    def diagnostics(self):
        return {
            **self.collision_culling.diagnostics(),
            "model": self.contact.model,
            "active_contacts": self._active_contacts,
            "pt_candidates": int(self.pt_candidates.shape[0]),
            "ee_candidates": int(self.ee_candidates.shape[0]),
            "constraint_violation": float(self._last_violation),
            "penalty": float(self.penalty),
        }


@ti.data_oriented
class FEMMultiContactAssembler:
    """One implicit contact contribution per configured FEM body pair.

    Each child owns pair-local broad-phase buffers and constants, while all
    children scatter directly into the same nodal force and sparse matrix.
    This avoids host-side candidate partitioning and preserves the exact
    single-parameter IPC implementation for every pair.
    """

    def __init__(self, contact, assemblers):
        self.contact = contact
        self.assemblers = tuple(assemblers)
        self.is_ipc = all(entry.is_ipc for entry in self.assemblers)
        self.is_augmented_lagrangian = all(entry.is_augmented_lagrangian for entry in self.assemblers)
        self.total_energy = ti.field(dtype=ti.f64, shape=())
        self.total_energy[None] = 0.0

    @ti.kernel
    def _accumulate_energy(self, source: ti.template()):
        self.total_energy[None] += source[None]

    def set_stitch_exclusions(self, stitches):
        for assembler in self.assemblers:
            assembler.set_stitch_exclusions(stitches)

    def prepare_iteration_device(self, positions, end_positions=None):
        for assembler in self.assemblers:
            assembler.prepare_iteration_device(positions, end_positions=end_positions)

    def begin_step_device(self, positions, dt):
        for assembler in self.assemblers:
            assembler.begin_step_device(positions, dt)

    def refresh_friction_device(self, positions):
        for assembler in self.assemblers:
            assembler.refresh_friction_device(positions)

    @property
    def activate_friction(self):
        return any(assembler.activate_friction for assembler in self.assemblers)

    def assemble_device(
        self,
        positions,
        force,
        stiffness,
        *,
        need_stiffness=False,
    ):
        self.total_energy[None] = 0.0
        for assembler in self.assemblers:
            force, stiffness = assembler.assemble_device(
                positions,
                force,
                stiffness,
                need_stiffness=need_stiffness,
            )
            self._accumulate_energy(assembler.total_energy)
        return force, stiffness

    def maximum_admissible_step_device(self, positions, direction):
        step = 1.0
        for assembler in self.assemblers:
            step = min(
                step,
                assembler.maximum_admissible_step_device(positions, direction),
            )
        return step

    def accept_update_device(self, positions, step=1.0):
        for assembler in self.assemblers:
            assembler.accept_update_device(positions, step)

    def prepared_converged_device(self):
        return all(assembler.prepared_converged_device() for assembler in self.assemblers)

    def device_diagnostics(self):
        pairs = [
            {
                "body_pair": assembler.contact.body_pair,
                **assembler.device_diagnostics(),
            }
            for assembler in self.assemblers
        ]
        return {
            "model": self.contact.model,
            "active_contacts": sum(entry["active_contacts"] for entry in pairs),
            "pt_candidates": sum(entry["pt_candidates"] for entry in pairs),
            "ee_candidates": sum(entry["ee_candidates"] for entry in pairs),
            "constraint_violation": max((entry["constraint_violation"] for entry in pairs), default=0.0),
            "contact_violation": max((entry["contact_violation"] for entry in pairs), default=0.0),
            "penalty": max((entry["penalty"] for entry in pairs), default=0.0),
            "contact_penalty": max((entry["contact_penalty"] for entry in pairs), default=0.0),
            "body_pairs": pairs,
        }

    def prepare_iteration(self, positions, end_positions=None):
        for assembler in self.assemblers:
            assembler.prepare_iteration(positions, end_positions)

    def begin_step(self, positions, dt):
        for assembler in self.assemblers:
            assembler.begin_step(positions, dt)

    def assemble_output(self, positions, mechanical, need_stiffness=False):
        result = mechanical
        for assembler in self.assemblers:
            result = assembler.assemble_output(positions, result, need_stiffness=need_stiffness)
        return result

    def maximum_admissible_step(self, positions, direction):
        return min(assembler.maximum_admissible_step(positions, direction) for assembler in self.assemblers)

    def accept_update(self, positions, step=1.0):
        for assembler in self.assemblers:
            assembler.accept_update(positions, step)

    def converged(self, positions):
        return all(assembler.converged(positions) for assembler in self.assemblers)

    def diagnostics(self):
        return self.device_diagnostics()


__all__ = ["FEMContactAssembler", "FEMMultiContactAssembler"]
