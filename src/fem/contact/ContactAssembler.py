"""Taichi PT/EE contact assembly for implicit FEM."""

import numpy as np
import taichi as ti

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
    edge_edge_distance_grad_hess_by_type,
    edge_edge_distance_type,
    point_triangle_distance_grad,
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
)


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


@ti.data_oriented
class FEMContactAssembler:
    """Exact IPC barrier contact.

    PT/EE distance derivatives are shared with GeoTaichi's other IPC engines.
    The IPC branch uses lumped primitive measures, a normalized clamped-log
    barrier, EE mollification, local PSD projection and conservative CCD.
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
        self.kappa = float(contact.kappa)
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
        self._active_contacts = 0
        self._last_energy = 0.0
        self._contact_feature_mask_value = 0
        self.friction_dt = 1.0
        self.total_energy = ti.field(dtype=ti.f64, shape=())
        self.ccd_alpha = ti.field(dtype=ti.f64, shape=())
        self.active_count = ti.field(dtype=ti.i32, shape=())
        self.friction_dt_field = ti.field(dtype=ti.f64, shape=())
        self.friction_dt_field[None] = self.friction_dt

        self.plane_count = len(self.contact.planes)
        allocated_planes = max(self.plane_count, 1)
        allocated_vertices = max(int(self.surface.vertices.size), 1)
        self.plane_origins = ti.Vector.field(3, dtype=self.real_type, shape=allocated_planes)
        self.plane_normals = ti.Vector.field(3, dtype=self.real_type, shape=allocated_planes)
        self.surface_vertices = ti.field(dtype=ti.i32, shape=allocated_vertices)
        self.surface_vertex_measure = ti.field(dtype=self.real_type, shape=allocated_vertices)
        self.device_pt_capacity = self.max_point_triangle_pairs
        self.device_ee_capacity = self.max_edge_edge_pairs
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
        self.plane_friction_normal_force.fill(0.0)
        self.prepare_iteration_device(self.device_positions)

    @property
    def is_ipc(self):
        return self.contact.model == "IPC"

    def set_stitch_exclusions(self, stitches):
        """Exclude garment-stitch neighborhoods from self-contact culling."""
        self.collision_culling.set_stitch_exclusions(stitches)
        self.prepare_iteration_device(self.device_positions)

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
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex_id]
            normal = self.plane_normals[plane_id]
            distance = normal.dot(positions[node] - self.plane_origins[plane_id])
            measure = self.surface_vertex_measure[vertex_id]
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
            self.friction_pt_weight[contact_id] = weights
            self.friction_pt_normal[contact_id] = normal
            normal_force = 0.0
            shifted = distance2 - self.dmin * self.dmin
            if shifted > 0.0 and shifted < active_gap2:
                _, first, _ = ipc_toolkit_barrier_distance2_offset_terms(
                    distance2,
                    self.dhat,
                    self.dmin,
                    normalized_kappa,
                    0,
                )
                scale = self.collision_culling.point_triangle_measure[contact_id] * self.dhat
                normal_force = ti.max(0.0, -scale * first * 2.0 * distance)
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
            self.friction_ee_weight[contact_id] = weights
            self.friction_ee_normal[contact_id] = normal
            normal_force = 0.0
            shifted = distance2 - self.dmin * self.dmin
            if shifted > 0.0 and shifted < active_gap2:
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
                    normal_force = ti.max(0.0, -scale * first * 2.0 * distance)
            self.friction_ee_normal_force[contact_id] = normal_force

    @ti.kernel
    def _freeze_plane_friction_device(self, positions: ti.template()):
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex_id]
            distance = self.plane_normals[plane_id].dot(positions[node] - self.plane_origins[plane_id])
            gap = distance - self.dmin
            normal_force = 0.0
            if gap > 0.0 and gap < self.dhat:
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
        for plane_id, vertex_id in ti.ndrange(self.plane_count, self.surface.vertices.size):
            flat_id = plane_id * self.surface.vertices.size + vertex_id
            node = self.surface_vertices[vertex_id]
            self._initialize_direct_plane_block(target, assemble_hash, scalar_base, flat_id, node)
            normal = self.plane_normals[plane_id]
            distance = normal.dot(positions[node] - self.plane_origins[plane_id])
            measure = self.surface_vertex_measure[vertex_id]
            block = ti.Matrix.zero(self.real_type, 3, 3)
            gap = distance - self.dmin
            if gap < self.dhat:
                _, _, curvature = _ipc_barrier_terms(gap, self.dhat, self.kappa)
                block += measure * self.dhat * curvature * normal.outer_product(normal)
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
        else:
            self.collision_culling.clear_active()

    def begin_step_device(self, positions, dt):
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
        if self.contact.self_contact:
            self._assemble_ipc_pt_device(positions, force, pt_count)
            self._assemble_ipc_ee_device(positions, force, ee_count)
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
        if self.contact.self_contact:
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
        self.ccd_alpha[None] = 1.0
        clearance = self.dmin
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

    def device_diagnostics(self):
        violation = 0.0
        return {
            **self.collision_culling.diagnostics(),
            "model": self.contact.model,
            "active_contacts": int(self.active_count[None]),
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
            "body_pairs": pairs,
        }


__all__ = ["FEMContactAssembler", "FEMMultiContactAssembler"]
