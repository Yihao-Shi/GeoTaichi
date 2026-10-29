"""Host numerical contact references, available to tests only."""

import numpy as np
import taichi as ti

from src.fem.engines.SparseMatrix import FEMTripletContribution
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_gershgorin_nd
from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance_grad,
    edge_edge_distance_grad_hess,
    point_triangle_distance_grad,
    point_triangle_distance_grad_hess,
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
)


from src.fem.contact.ContactAssembler import FEMContactAssembler, _ipc_barrier_terms


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
    """Host reference geometry; production uses ContactGeometry's device functions."""
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
        cross = np.cross(da, db)
        denominator = np.dot(cross, cross)
        sa = np.clip(np.dot(np.cross(db, offset), cross) / denominator, 0.0, 1.0) if denominator > epsilon else 0.0
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
class ReferenceFEMContactAssembler(FEMContactAssembler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_candidates = np.empty((0, 4), dtype=np.int32)
        self.ee_candidates = np.empty((0, 4), dtype=np.int32)
        self.pt_measures = np.empty(0, dtype=np.float64)
        self.ee_measures = np.empty(0, dtype=np.float64)
        self.friction_candidates_pt = np.empty((0, 4), dtype=np.int32)
        self.friction_candidates_ee = np.empty((0, 4), dtype=np.int32)
        self.friction_weights_pt = np.empty((0, 4), dtype=np.float64)
        self.friction_weights_ee = np.empty((0, 4), dtype=np.float64)
        self.friction_normals_pt = np.empty((0, 3), dtype=np.float64)
        self.friction_normals_ee = np.empty((0, 3), dtype=np.float64)
        self.friction_lambdas_pt = np.empty(0, dtype=np.float64)
        self.friction_lambdas_ee = np.empty(0, dtype=np.float64)
        self.friction_hat_positions = self.reference_positions.copy()
        self.plane_friction_lambdas = np.empty((0, 0), dtype=np.float64)

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
        return np.ascontiguousarray(integration_weight * area * self.dhat)

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
                        -measures[index] * derivative * 2.0 * distance,
                    )
        return forces

    def begin_step(self, positions, dt):
        """Freeze host reference friction; solver steps use begin_step_device."""
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
                gaps = (positions[vertices] - origin) @ normal - (self.dmin + self.dhat)
                for vertex_id, gap in enumerate(gaps):
                    if gap > 0.0:
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
        self.pt_candidates = np.ascontiguousarray(pt)
        self.ee_candidates = np.ascontiguousarray(ee)
        self.pt_measures = self._candidate_measure(pt, "pt")
        self.ee_measures = self._candidate_measure(ee, "ee")

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
        if self.contact.project_pd:
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
        force: ti.types.ndarray(dtype=ti.f64, ndim=2),
        hessian: ti.types.ndarray(dtype=ti.f64, ndim=3),
        need_hessian: ti.i32,
    ):
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
            gap = distance - self.dmin
            if gap < self.dhat:
                value, derivative, curvature = _ipc_barrier_terms(gap, self.dhat, self.kappa)
                ti.atomic_add(self.total_energy[None], scale * self.dhat * value)
                ti.atomic_add(self.active_count[None], 1)
                for component in ti.static(range(3)):
                    ti.atomic_add(force[node, component], scale * self.dhat * derivative * normals[plane_id, component])
                    if need_hessian:
                        for second_component in ti.static(range(3)):
                            hessian[flat_id, component, second_component] = (
                                scale
                                * self.dhat
                                * curvature
                                * normals[plane_id, component]
                                * normals[plane_id, second_component]
                            )

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
        return vertices, origins, normals, measures

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
                vertices, origins, _, _ = plane_data
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
        self._assemble_ipc_pt(positions, self.pt_candidates, self.pt_measures, force, pt_hessian, bool(need_stiffness))
        self._assemble_ipc_ee(
            positions,
            self.reference_positions,
            self.ee_candidates,
            self.ee_measures,
            force,
            ee_hessian,
            bool(need_stiffness),
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
            vertices, origins, normals, measures = plane_data
            self._assemble_planes(
                positions,
                vertices,
                origins,
                normals,
                measures,
                force,
                plane_hessian,
                int(need_stiffness),
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
        if need_stiffness and not np.isfinite(self._last_energy):
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
        clearance = self.dmin
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

    def converged(self, positions):
        return np.isfinite(self._last_energy)

    def diagnostics(self):
        return {
            **self.collision_culling.diagnostics(),
            "model": self.contact.model,
            "active_contacts": self._active_contacts,
            "pt_candidates": int(self.pt_candidates.shape[0]),
            "ee_candidates": int(self.ee_candidates.shape[0]),
        }
