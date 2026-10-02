"""Mixed FEM-node/AffineBody-control IPC barrier/friction assembly."""

import numpy as np
import taichi as ti

from src.contact_detection.continuous_contact_detection.AdditiveCCD import (
    edge_edge_accd,
    point_triangle_accd,
)
from src.contact_detection.continuous_contact_detection.CCD import (
    edge_edge_ccd,
    point_triangle_ccd,
)
from src.fem.contact.BVHBroadPhase import DynamicBVHBroadPhase
from src.fem.contact.CollisionCulling import FEMCollisionCulling
from src.fem.contact.LinkedCellBroadPhase import (
    DynamicLinkedCellBroadPhase,
)
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd
from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance_type,
    g_EE,
    g_PE3D,
    g_PT,
    line_line_distance2,
    point_line_distance2,
    point_plane_distance2,
    point_point_distance2,
    point_point_grad,
    point_triangle_distance_grad,
    point_triangle_distance_grad_hess_by_type,
    point_triangle_distance_type,
    edge_edge_distance_grad,
    edge_edge_distance_grad_hess_by_type,
)
from src.physics_model.contact_model.ipc.ContactMollifier import (
    edge_edge_mollifier,
    edge_edge_mollifier_grad,
    edge_edge_mollifier_terms,
    edge_edge_mollifier_threshold,
)
from src.physics_model.contact_model.ipc.IPC import (
    ipc_friction_f0,
    ipc_friction_f1_over_speed,
    ipc_toolkit_barrier_distance2_offset_terms,
    semi_ipc_find_or_insert,
    semi_ipc_terms,
    semi_ipc_update_multiplier,
)


@ti.data_oriented
class FEMAffineIPCAssembler:
    """IPC PT/EE barrier and lagged friction pulled back to global blocks."""

    def __init__(self, fem, affine, model, simulation, fem_faces=None):
        self.fem = fem
        self.affine = affine
        self.model = model
        self.real_type = ti.lang.impl.current_cfg().default_fp
        if self.real_type != ti.f64:
            raise RuntimeError("FEM--AffineBody IPC requires Taichi float64")
        self.fem_node_count = int(fem.mesh.number_of_nodes)
        self.affine_vertex_count = int(affine.vertex_num)
        self.affine_control_count = int(affine.control_num)
        self.combined_node_count = self.fem_node_count + self.affine_vertex_count

        selected_faces = (
            np.asarray(fem.mesh.boundary_facets()[0], dtype=np.int32)
            if fem_faces is None
            else np.asarray(fem_faces, dtype=np.int32)
        )
        selected_faces = np.ascontiguousarray(selected_faces.reshape(-1, 3))
        fem_edges, fem_node_area, fem_edge_area = self._surface_measures(
            np.asarray(fem.mesh.rest_shape, dtype=np.float64),
            selected_faces,
        )
        affine_offset = self.fem_node_count
        faces = np.concatenate((selected_faces, affine.faces_np + affine_offset), axis=0).astype(np.int32, copy=False)
        edges = np.concatenate((fem_edges, affine.edges_np + affine_offset), axis=0).astype(np.int32, copy=False)
        vertices = np.concatenate(
            (
                np.unique(selected_faces).astype(np.int32),
                np.arange(
                    affine_offset,
                    affine_offset + self.affine_vertex_count,
                    dtype=np.int32,
                ),
            )
        )
        affine_reference = affine.rest_x.to_numpy()[: self.affine_vertex_count]
        reference = np.ascontiguousarray(
            np.concatenate((fem.mesh.points, affine_reference), axis=0),
            dtype=np.float64,
        )
        node_area = np.concatenate(
            (
                fem_node_area,
                affine.node_area.to_numpy()[: self.affine_vertex_count],
            )
        )
        edge_area = np.concatenate(
            (
                fem_edge_area,
                affine.edge_area.to_numpy()[: affine.edge_num],
            )
        )
        node_system = np.concatenate(
            (
                np.zeros(self.fem_node_count, dtype=np.int32),
                np.ones(self.affine_vertex_count, dtype=np.int32),
            )
        )
        broad_phase_type = (
            DynamicBVHBroadPhase
            if str(simulation.search).replace("_", "").replace("-", "").lower() == "bvh"
            else DynamicLinkedCellBroadPhase
        )
        self.max_point_triangle_pairs = int(simulation.max_point_triangle_pairs)
        self.max_edge_edge_pairs = int(simulation.max_edge_edge_pairs)
        self.broad_phase = broad_phase_type(
            faces,
            edges,
            vertices,
            node_area,
            edge_area,
            reference,
            max_point_triangle_pairs=self.max_point_triangle_pairs,
            max_edge_edge_pairs=self.max_edge_edge_pairs,
            node_system_ids=node_system,
            cross_system_only=True,
        )
        node_body = np.concatenate(
            (
                np.asarray(fem.mesh.node_body_ids, dtype=np.int32),
                np.asarray(affine.node2body_np, dtype=np.int32),
            )
        )
        self.culling = FEMCollisionCulling(
            self.broad_phase,
            self.combined_node_count,
            node_body_ids=node_body,
            node_system_ids=node_system,
            cross_system_only=True,
            max_point_triangle_pairs=self.max_point_triangle_pairs,
            max_edge_edge_pairs=self.max_edge_edge_pairs,
        )
        self.position = ti.Vector.field(3, dtype=self.real_type, shape=self.combined_node_count)
        self.end_position = ti.Vector.field(3, dtype=self.real_type, shape=self.combined_node_count)
        self.fem_node_body = ti.field(dtype=ti.i32, shape=self.fem_node_count)
        self.fem_node_body.from_numpy(np.ascontiguousarray(fem.mesh.node_body_ids, dtype=np.int32))
        support_count = np.ones(self.combined_node_count, dtype=np.int32)
        support_block = np.zeros((self.combined_node_count, 4), dtype=np.int32)
        support_weight = np.zeros((self.combined_node_count, 4), dtype=np.float64)
        support_block[: self.fem_node_count, 0] = self.affine_control_count + np.arange(
            self.fem_node_count, dtype=np.int32
        )
        support_weight[: self.fem_node_count, 0] = 1.0
        affine_basis = affine.basis.to_numpy()[: self.affine_vertex_count]
        for vertex in range(self.affine_vertex_count):
            combined = self.fem_node_count + vertex
            body = int(affine.node2body_np[vertex])
            support_count[combined] = 4
            support_block[combined] = body * 4 + np.arange(4)
            support_weight[combined] = affine_basis[vertex]
        self.support_count = ti.field(dtype=ti.i32, shape=self.combined_node_count)
        self.support_block = ti.Vector.field(4, dtype=ti.i32, shape=self.combined_node_count)
        self.support_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.combined_node_count)
        self.support_count.from_numpy(support_count)
        self.support_block.from_numpy(support_block)
        self.support_weight.from_numpy(support_weight)

        characteristic = float(np.linalg.norm(np.ptp(reference, axis=0)))
        parameters = model.parameter_arrays(
            affine.body_num,
            int(np.max(fem.mesh.node_body_ids)) + 1,
            characteristic,
        )
        self.affine_body_count, self.fem_body_count = parameters["active"].shape
        pair_shape = (self.affine_body_count, self.fem_body_count)
        self.pair_active = ti.field(dtype=ti.i32, shape=pair_shape)
        self.pair_kappa = ti.field(dtype=self.real_type, shape=pair_shape)
        self.pair_dhat = ti.field(dtype=self.real_type, shape=pair_shape)
        self.pair_dmin = ti.field(dtype=self.real_type, shape=pair_shape)
        self.pair_friction = ti.field(dtype=self.real_type, shape=pair_shape)
        self.pair_epsv = ti.field(dtype=self.real_type, shape=pair_shape)
        self.pair_penalty = ti.field(dtype=self.real_type, shape=pair_shape)
        self.pair_active.from_numpy(parameters["active"])
        self.pair_kappa.from_numpy(parameters["kappa"])
        self.pair_dhat.from_numpy(parameters["dhat"])
        self.pair_dmin.from_numpy(parameters["dmin"])
        self.pair_friction.from_numpy(parameters["friction"])
        self.pair_epsv.from_numpy(parameters["epsv"])
        self.pair_penalty.from_numpy(parameters["penalty"])
        self.search_distance = float(parameters["search_distance"])
        self.maximum_dmin = float(np.max(parameters["dmin"][parameters["active"] != 0]))
        self.ccd_eta = 1.0 - float(parameters["ccd_safety"])
        self.ccd_max_iterations = int(parameters["ccd_max_iterations"])
        self.total_energy = ti.field(dtype=self.real_type, shape=())
        self.active_count = ti.field(dtype=ti.i32, shape=())
        self.minimum_distance = ti.field(dtype=self.real_type, shape=())
        self.minimum_step = ti.field(dtype=self.real_type, shape=())
        self.minimum_step.fill(1.0)
        self.friction_energy = ti.field(dtype=self.real_type, shape=())
        self.friction_active_count = ti.field(dtype=ti.i32, shape=())
        self.friction_dt = ti.field(dtype=self.real_type, shape=())
        self.activate_friction = bool(model.activate_friction)
        self.is_semi = model.contact.model == "AugmentedLagrangian"
        self.constraint_tolerance = float(model.contact.constraint_tolerance)
        self.friction_hat = ti.Vector.field(3, dtype=self.real_type, shape=self.combined_node_count)
        self.friction_pt_count = 0
        self.friction_ee_count = 0
        self.friction_pt_capacity = self.max_point_triangle_pairs
        self.friction_ee_capacity = self.max_edge_edge_pairs
        self._allocate_friction_terms(self.friction_pt_capacity, self.friction_ee_capacity)
        self._allocate_adjoint_friction_terms()
        self.pt_capacity = self.max_point_triangle_pairs
        self.ee_capacity = self.max_edge_edge_pairs
        self._allocate_local_terms(self.pt_capacity, self.ee_capacity)
        self._allocate_semi_state()

    def _allocate_semi_state(self):
        capacity = 1 << (max(2 * max(self.pt_capacity, self.ee_capacity), 1) - 1).bit_length()
        self.semi_capacity = capacity
        self.semi_overflow = ti.field(dtype=ti.i32, shape=())
        self.constraint_violation = ti.field(dtype=self.real_type, shape=())
        self.semi_pt_count = ti.field(dtype=ti.i32, shape=())
        self.semi_ee_count = ti.field(dtype=ti.i32, shape=())
        self.semi_pt_state = ti.field(dtype=ti.i32, shape=capacity)
        self.semi_ee_state = ti.field(dtype=ti.i32, shape=capacity)
        self.semi_pt_key = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
        self.semi_ee_key = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
        self.semi_pt_weight = ti.Vector.field(4, dtype=self.real_type, shape=capacity)
        self.semi_ee_weight = ti.Vector.field(4, dtype=self.real_type, shape=capacity)
        self.semi_pt_normal = ti.Vector.field(3, dtype=self.real_type, shape=capacity)
        self.semi_ee_normal = ti.Vector.field(3, dtype=self.real_type, shape=capacity)
        self.semi_pt_multiplier = ti.field(dtype=self.real_type, shape=capacity)
        self.semi_ee_multiplier = ti.field(dtype=self.real_type, shape=capacity)
        self.semi_pt_slot = ti.field(dtype=ti.i32, shape=self.pt_capacity)
        self.semi_ee_slot = ti.field(dtype=ti.i32, shape=self.ee_capacity)
        self.semi_pt_state.fill(2)
        self.semi_ee_state.fill(2)

    def _allocate_friction_terms(self, pt_capacity, ee_capacity):
        self.friction_pt_capacity = int(pt_capacity)
        self.friction_ee_capacity = int(ee_capacity)
        self.friction_pt_candidate = ti.Vector.field(4, dtype=ti.i32, shape=self.friction_pt_capacity)
        self.friction_ee_candidate = ti.Vector.field(4, dtype=ti.i32, shape=self.friction_ee_capacity)
        self.friction_pt_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.friction_pt_capacity)
        self.friction_ee_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.friction_ee_capacity)
        self.friction_pt_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.friction_pt_capacity)
        self.friction_ee_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.friction_ee_capacity)
        self.friction_pt_coefficient = ti.field(dtype=self.real_type, shape=self.friction_pt_capacity)
        self.friction_ee_coefficient = ti.field(dtype=self.real_type, shape=self.friction_ee_capacity)
        self.friction_pt_epsv = ti.field(dtype=self.real_type, shape=self.friction_pt_capacity)
        self.friction_ee_epsv = ti.field(dtype=self.real_type, shape=self.friction_ee_capacity)
        self.friction_pt_gradient = ti.field(dtype=self.real_type, shape=(self.friction_pt_capacity, 3))
        self.friction_ee_gradient = ti.field(dtype=self.real_type, shape=(self.friction_ee_capacity, 3))
        self.friction_pt_hessian = ti.field(dtype=self.real_type, shape=(self.friction_pt_capacity, 3, 3))
        self.friction_ee_hessian = ti.field(dtype=self.real_type, shape=(self.friction_ee_capacity, 3, 3))

    def _allocate_adjoint_friction_terms(self):
        pt_capacity = self.friction_pt_capacity if self.activate_friction else 1
        ee_capacity = self.friction_ee_capacity if self.activate_friction else 1
        self.adjoint_friction_valid = ti.field(dtype=ti.i32, shape=())
        self.adjoint_friction_pt_count = ti.field(dtype=ti.i32, shape=())
        self.adjoint_friction_ee_count = ti.field(dtype=ti.i32, shape=())
        self.adjoint_friction_pt_candidate = ti.Vector.field(4, ti.i32, shape=pt_capacity)
        self.adjoint_friction_ee_candidate = ti.Vector.field(4, ti.i32, shape=ee_capacity)
        self.adjoint_friction_pt_weight = ti.Vector.field(4, self.real_type, shape=pt_capacity)
        self.adjoint_friction_ee_weight = ti.Vector.field(4, self.real_type, shape=ee_capacity)
        self.adjoint_friction_pt_normal = ti.Vector.field(3, self.real_type, shape=pt_capacity)
        self.adjoint_friction_ee_normal = ti.Vector.field(3, self.real_type, shape=ee_capacity)
        self.adjoint_friction_pt_coefficient = ti.field(self.real_type, shape=pt_capacity)
        self.adjoint_friction_ee_coefficient = ti.field(self.real_type, shape=ee_capacity)
        self.adjoint_friction_pt_epsv = ti.field(self.real_type, shape=pt_capacity)
        self.adjoint_friction_ee_epsv = ti.field(self.real_type, shape=ee_capacity)

    @ti.kernel
    def _backup_lagged_friction_for_adjoint(self, pt_count: ti.i32, ee_count: ti.i32):
        self.adjoint_friction_valid[None] = 1
        self.adjoint_friction_pt_count[None] = pt_count
        self.adjoint_friction_ee_count[None] = ee_count
        for contact in range(pt_count):
            self.adjoint_friction_pt_candidate[contact] = self.friction_pt_candidate[contact]
            self.adjoint_friction_pt_weight[contact] = self.friction_pt_weight[contact]
            self.adjoint_friction_pt_normal[contact] = self.friction_pt_normal[contact]
            self.adjoint_friction_pt_coefficient[contact] = self.friction_pt_coefficient[contact]
            self.adjoint_friction_pt_epsv[contact] = self.friction_pt_epsv[contact]
        for contact in range(ee_count):
            self.adjoint_friction_ee_candidate[contact] = self.friction_ee_candidate[contact]
            self.adjoint_friction_ee_weight[contact] = self.friction_ee_weight[contact]
            self.adjoint_friction_ee_normal[contact] = self.friction_ee_normal[contact]
            self.adjoint_friction_ee_coefficient[contact] = self.friction_ee_coefficient[contact]
            self.adjoint_friction_ee_epsv[contact] = self.friction_ee_epsv[contact]

    @ti.kernel
    def _restore_lagged_friction_for_adjoint(self):
        pt_count = self.adjoint_friction_pt_count[None]
        ee_count = self.adjoint_friction_ee_count[None]
        for contact in range(pt_count):
            self.friction_pt_candidate[contact] = self.adjoint_friction_pt_candidate[contact]
            self.friction_pt_weight[contact] = self.adjoint_friction_pt_weight[contact]
            self.friction_pt_normal[contact] = self.adjoint_friction_pt_normal[contact]
            self.friction_pt_coefficient[contact] = self.adjoint_friction_pt_coefficient[contact]
            self.friction_pt_epsv[contact] = self.adjoint_friction_pt_epsv[contact]
        for contact in range(ee_count):
            self.friction_ee_candidate[contact] = self.adjoint_friction_ee_candidate[contact]
            self.friction_ee_weight[contact] = self.adjoint_friction_ee_weight[contact]
            self.friction_ee_normal[contact] = self.adjoint_friction_ee_normal[contact]
            self.friction_ee_coefficient[contact] = self.adjoint_friction_ee_coefficient[contact]
            self.friction_ee_epsv[contact] = self.adjoint_friction_ee_epsv[contact]

    def backup_lagged_friction_for_adjoint_device(self):
        if self.activate_friction:
            self._backup_lagged_friction_for_adjoint(
                int(self.friction_pt_count),
                int(self.friction_ee_count),
            )

    def restore_lagged_friction_for_adjoint_device(self):
        if self.activate_friction and int(self.adjoint_friction_valid[None]):
            self._restore_lagged_friction_for_adjoint()
            self.friction_pt_count = int(self.adjoint_friction_pt_count[None])
            self.friction_ee_count = int(self.adjoint_friction_ee_count[None])

    def _ensure_friction_terms(self, pt_count, ee_count):
        if int(pt_count) > self.friction_pt_capacity:
            raise RuntimeError(
                "FEM--AffineBody friction PT capacity is too small: "
                f"need {pt_count}, allocated {self.friction_pt_capacity}"
            )
        if int(ee_count) > self.friction_ee_capacity:
            raise RuntimeError(
                "FEM--AffineBody friction EE capacity is too small: "
                f"need {ee_count}, allocated {self.friction_ee_capacity}"
            )

    def _allocate_local_terms(self, pt_capacity, ee_capacity):
        self.pt_capacity = int(pt_capacity)
        self.ee_capacity = int(ee_capacity)
        self.pt_active = ti.field(dtype=ti.i32, shape=self.pt_capacity)
        self.ee_active = ti.field(dtype=ti.i32, shape=self.ee_capacity)
        self.pt_gradient = ti.field(dtype=self.real_type, shape=(self.pt_capacity, 12))
        self.ee_gradient = ti.field(dtype=self.real_type, shape=(self.ee_capacity, 12))
        self.pt_distance_type = ti.field(dtype=ti.i32, shape=self.pt_capacity)
        self.ee_distance_type = ti.field(dtype=ti.i32, shape=self.ee_capacity)
        self.pt_distance2 = ti.field(dtype=self.real_type, shape=self.pt_capacity)
        self.ee_distance2 = ti.field(dtype=self.real_type, shape=self.ee_capacity)

    def _ensure_local_terms(self, pt_count, ee_count):
        if int(pt_count) > self.pt_capacity:
            raise RuntimeError(
                "FEM--AffineBody local PT capacity is too small: " f"need {pt_count}, allocated {self.pt_capacity}"
            )
        if int(ee_count) > self.ee_capacity:
            raise RuntimeError(
                "FEM--AffineBody local EE capacity is too small: " f"need {ee_count}, allocated {self.ee_capacity}"
            )

    @staticmethod
    def _surface_measures(reference, faces):
        edge_map = {}
        edges = []
        node_area = np.zeros(reference.shape[0], dtype=np.float64)
        for face in faces:
            point = reference[face]
            area = 0.5 * np.linalg.norm(np.cross(point[1] - point[0], point[2] - point[0]))
            if area <= 0.0:
                raise ValueError("FEM--AffineBody contact contains a degenerate triangle")
            node_area[face] += area / 3.0
            for first, second in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            ):
                key = tuple(sorted((int(first), int(second))))
                if key not in edge_map:
                    edge_map[key] = len(edges)
                    edges.append(key)
        edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        edge_area = np.zeros(edges.shape[0], dtype=np.float64)
        for face in faces:
            point = reference[face]
            area = 0.5 * np.linalg.norm(np.cross(point[1] - point[0], point[2] - point[0]))
            for first, second in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            ):
                edge_area[edge_map[tuple(sorted((int(first), int(second))))]] += area / 3.0
        return edges, node_area, edge_area

    @ti.kernel
    def build_positions(self, fem_position: ti.template()):
        for node in range(self.fem_node_count):
            self.position[node] = fem_position[node]
        for vertex in range(self.affine_vertex_count):
            self.position[self.fem_node_count + vertex] = self.affine.x[vertex]

    @ti.kernel
    def build_end_positions(
        self,
        fem_position: ti.template(),
        fem_direction: ti.template(),
    ):
        for node in range(self.fem_node_count):
            self.end_position[node] = fem_position[node] + fem_direction[node]
        for vertex in range(self.affine_vertex_count):
            body = self.affine.node2body[vertex]
            increment = ti.Vector.zero(self.real_type, 3)
            for local_control in ti.static(range(4)):
                increment += (
                    self.affine.basis[vertex, local_control] * self.affine.direction_y[body * 4 + local_control]
                )
            self.end_position[self.fem_node_count + vertex] = self.affine.x[vertex] + increment

    @ti.kernel
    def _copy_friction_hat(self, fem_hat_position: ti.template()):
        for node in range(self.fem_node_count):
            self.friction_hat[node] = fem_hat_position[node]
        for vertex in range(self.affine_vertex_count):
            body = self.affine.node2body[vertex]
            value = ti.Vector.zero(self.real_type, 3)
            for local_control in ti.static(range(4)):
                value += self.affine.basis[vertex, local_control] * self.affine.hat_y[body * 4 + local_control]
            self.friction_hat[self.fem_node_count + vertex] = value

    @ti.func
    def _pair_indices(self, stencil):
        affine_vertex = 0
        fem_node = 0
        if stencil[0] < ti.static(self.fem_node_count):
            fem_node = stencil[0]
            affine_vertex = stencil[1] - ti.static(self.fem_node_count)
        else:
            affine_vertex = stencil[0] - ti.static(self.fem_node_count)
            fem_node = stencil[1]
        return (
            self.affine.node2body[affine_vertex],
            self.fem_node_body[fem_node],
        )

    @ti.func
    def _edge_pair_indices(self, stencil):
        affine_vertex = 0
        fem_node = 0
        if stencil[0] < ti.static(self.fem_node_count):
            fem_node = stencil[0]
            affine_vertex = stencil[2] - ti.static(self.fem_node_count)
        else:
            affine_vertex = stencil[0] - ti.static(self.fem_node_count)
            fem_node = stencil[2]
        return (
            self.affine.node2body[affine_vertex],
            self.fem_node_body[fem_node],
        )

    @ti.func
    def _pair_is_active(self, affine_body, fem_body):
        return (
            0 <= affine_body
            and affine_body < ti.static(self.affine_body_count)
            and 0 <= fem_body
            and fem_body < ti.static(self.fem_body_count)
            and self.pair_active[affine_body, fem_body] != 0
        )

    @ti.func
    def _scatter_gradient(self, stencil, gradient, rhs):
        for site in range(4):
            node = stencil[site]
            for local_support in range(self.support_count[node]):
                block = self.support_block[node][local_support]
                weight = self.support_weight[node][local_support]
                for component in range(3):
                    ti.atomic_add(
                        rhs[3 * block + component],
                        -weight * gradient[3 * site + component],
                    )

    @ti.func
    def _scatter_hessian_block(self, stencil, first_site, second_site, hessian, matrix):
        first_node = stencil[first_site]
        second_node = stencil[second_site]
        for first_support in range(self.support_count[first_node]):
            first_block = self.support_block[first_node][first_support]
            first_weight = self.support_weight[first_node][first_support]
            for second_support in range(self.support_count[second_node]):
                second_block = self.support_block[second_node][second_support]
                second_weight = self.support_weight[second_node][second_support]
                matrix.add_block_entry(
                    first_block,
                    second_block,
                    first_weight * second_weight * hessian,
                )

    def prepare(self, fem_position):
        self.affine._reconstruct_vertices()
        self.build_positions(fem_position)
        counts = self.culling.rebuild_proximity(self.position, self.search_distance)
        self._ensure_local_terms(*counts)
        if self.is_semi:
            self.semi_overflow[None] = 0
            self._prepare_semi_pt(int(counts[0]))
            self._prepare_semi_ee(int(counts[1]))
            if int(self.semi_overflow[None]) != 0:
                raise RuntimeError("FEM--AffineBody SemiIPC multiplier hash capacity is too small")
        return counts

    @ti.func
    def _semi_frame(self, gradient, distance, edge_edge):
        relative = ti.Vector([gradient[0], gradient[1], gradient[2]])
        if edge_edge != 0:
            for component in ti.static(range(3)):
                relative[component] += gradient[3 + component]
        normal = ti.Vector([1.0, 0.0, 0.0])
        weight = ti.Vector.zero(self.real_type, 4)
        if distance > 1.0e-15 and relative.norm_sqr() > 1.0e-30:
            normal = relative.normalized()
            for site, component in ti.static(ti.ndrange(4, 3)):
                weight[site] += gradient[3 * site + component] * normal[component] / (2.0 * distance)
        return weight, normal

    @ti.func
    def _store_semi_frame(
        self,
        contact,
        key,
        weight,
        normal,
        states: ti.template(),
        keys: ti.template(),
        multipliers: ti.template(),
        count: ti.template(),
        hash_weight: ti.template(),
        hash_normal: ti.template(),
        contact_slot: ti.template(),
    ):
        slot = semi_ipc_find_or_insert(states, keys, multipliers, count, key, ti.static(self.semi_capacity))
        if slot >= 0:
            previous = hash_normal[slot]
            if previous.norm_sqr() > 0.0 and previous.dot(normal) < 0.0:
                normal = -normal
            hash_weight[slot] = weight
            hash_normal[slot] = normal
        else:
            self.semi_overflow[None] = 1
        contact_slot[contact] = slot

    @ti.kernel
    def _prepare_semi_pt(self, count: ti.i32):
        for contact in range(count):
            stencil = self.culling.point_triangle[contact]
            affine_body, fem_body = self._pair_indices(stencil)
            self.semi_pt_slot[contact] = -1
            if self._pair_is_active(affine_body, fem_body):
                distance2, gradient, _ = point_triangle_distance_grad(
                    self.position[stencil[0]],
                    self.position[stencil[1]],
                    self.position[stencil[2]],
                    self.position[stencil[3]],
                )
                weight, normal = self._semi_frame(gradient, ti.sqrt(ti.max(distance2, 0.0)), 0)
                self._store_semi_frame(
                    contact,
                    stencil,
                    weight,
                    normal,
                    self.semi_pt_state,
                    self.semi_pt_key,
                    self.semi_pt_multiplier,
                    self.semi_pt_count,
                    self.semi_pt_weight,
                    self.semi_pt_normal,
                    self.semi_pt_slot,
                )

    @ti.kernel
    def _prepare_semi_ee(self, count: ti.i32):
        for contact in range(count):
            stencil = self.culling.edge_edge[contact]
            affine_body, fem_body = self._edge_pair_indices(stencil)
            self.semi_ee_slot[contact] = -1
            if self._pair_is_active(affine_body, fem_body):
                distance2, gradient, _ = edge_edge_distance_grad(
                    self.position[stencil[0]],
                    self.position[stencil[1]],
                    self.position[stencil[2]],
                    self.position[stencil[3]],
                )
                weight, normal = self._semi_frame(gradient, ti.sqrt(ti.max(distance2, 0.0)), 1)
                self._store_semi_frame(
                    contact,
                    stencil,
                    weight,
                    normal,
                    self.semi_ee_state,
                    self.semi_ee_key,
                    self.semi_ee_multiplier,
                    self.semi_ee_count,
                    self.semi_ee_weight,
                    self.semi_ee_normal,
                    self.semi_ee_slot,
                )

    def _prepare_distance_terms(self, pt_count, ee_count):
        self._classify_contact_features(int(pt_count), int(ee_count))
        for feature in range(3):
            self._differentiate_point_triangle(int(pt_count), feature)
            self._differentiate_edge_edge(int(ee_count), feature)

    @ti.kernel
    def _reset_friction_state(self):
        self.friction_active_count[None] = 0

    @ti.kernel
    def _freeze_point_triangle_friction(self, count: ti.i32):
        for contact in range(count):
            stencil = self.culling.point_triangle[contact]
            self.friction_pt_candidate[contact] = stencil
            self.friction_pt_weight[contact] = ti.Vector([1.0, 0.0, 0.0, 0.0])
            self.friction_pt_normal[contact] = ti.Vector([1.0, 0.0, 0.0])
            self.friction_pt_coefficient[contact] = 0.0
            self.friction_pt_epsv[contact] = 1.0
            affine_body, fem_body = self._pair_indices(stencil)
            if self.pt_distance_type[contact] >= 0 and self._pair_is_active(affine_body, fem_body):
                distance2 = self.pt_distance2[contact]
                distance = ti.sqrt(ti.max(distance2, 0.0))
                relative = ti.Vector.zero(self.real_type, 3)
                for component in ti.static(range(3)):
                    relative[component] = self.pt_gradient[contact, component]
                if distance > 1.0e-15 and relative.norm() > 1.0e-15:
                    normal = relative.normalized()
                    weights = ti.Vector.zero(self.real_type, 4)
                    for site in ti.static(range(4)):
                        site_gradient = ti.Vector.zero(self.real_type, 3)
                        for component in ti.static(range(3)):
                            site_gradient[component] = self.pt_gradient[contact, 3 * site + component]
                        weights[site] = site_gradient.dot(normal) / (2.0 * distance)
                    dmin = self.pair_dmin[affine_body, fem_body]
                    dhat = self.pair_dhat[affine_body, fem_body]
                    shifted = distance2 - dmin * dmin
                    active_gap2 = (2.0 * dmin + dhat) * dhat
                    if shifted > 0.0 and shifted < active_gap2:
                        normalized_kappa = self.pair_kappa[affine_body, fem_body] / (active_gap2 * active_gap2)
                        _, first, _ = ipc_toolkit_barrier_distance2_offset_terms(
                            distance2,
                            dhat,
                            dmin,
                            normalized_kappa,
                            0,
                        )
                        scale = self.culling.point_triangle_measure[contact] * dhat
                        normal_force = ti.max(
                            0.0,
                            -scale * first * 2.0 * ti.sqrt(shifted),
                        )
                        coefficient = self.pair_friction[affine_body, fem_body] * normal_force
                        if coefficient > 0.0:
                            self.friction_pt_weight[contact] = weights
                            self.friction_pt_normal[contact] = normal
                            self.friction_pt_coefficient[contact] = coefficient
                            self.friction_pt_epsv[contact] = self.pair_epsv[affine_body, fem_body]
                            ti.atomic_add(self.friction_active_count[None], 1)

    @ti.kernel
    def _freeze_edge_edge_friction(self, count: ti.i32):
        for contact in range(count):
            stencil = self.culling.edge_edge[contact]
            self.friction_ee_candidate[contact] = stencil
            self.friction_ee_weight[contact] = ti.Vector([1.0, 0.0, 0.0, 0.0])
            self.friction_ee_normal[contact] = ti.Vector([1.0, 0.0, 0.0])
            self.friction_ee_coefficient[contact] = 0.0
            self.friction_ee_epsv[contact] = 1.0
            affine_body, fem_body = self._edge_pair_indices(stencil)
            if self.ee_distance_type[contact] >= 0 and self._pair_is_active(affine_body, fem_body):
                distance2 = self.ee_distance2[contact]
                distance = ti.sqrt(ti.max(distance2, 0.0))
                relative = ti.Vector.zero(self.real_type, 3)
                for component in ti.static(range(3)):
                    relative[component] = (
                        self.ee_gradient[contact, component] + self.ee_gradient[contact, 3 + component]
                    )
                if distance > 1.0e-15 and relative.norm() > 1.0e-15:
                    normal = relative.normalized()
                    weights = ti.Vector.zero(self.real_type, 4)
                    for site in ti.static(range(4)):
                        site_gradient = ti.Vector.zero(self.real_type, 3)
                        for component in ti.static(range(3)):
                            site_gradient[component] = self.ee_gradient[contact, 3 * site + component]
                        weights[site] = site_gradient.dot(normal) / (2.0 * distance)
                    dmin = self.pair_dmin[affine_body, fem_body]
                    dhat = self.pair_dhat[affine_body, fem_body]
                    shifted = distance2 - dmin * dmin
                    active_gap2 = (2.0 * dmin + dhat) * dhat
                    p0 = self.position[stencil[0]]
                    p1 = self.position[stencil[1]]
                    q0 = self.position[stencil[2]]
                    q1 = self.position[stencil[3]]
                    threshold = edge_edge_mollifier_threshold(
                        self.culling.reference_position[stencil[0]],
                        self.culling.reference_position[stencil[1]],
                        self.culling.reference_position[stencil[2]],
                        self.culling.reference_position[stencil[3]],
                    )
                    mollifier = edge_edge_mollifier(p0, p1, q0, q1, threshold)
                    if shifted > 0.0 and shifted < active_gap2 and mollifier >= 1.0 - 1.0e-12:
                        normalized_kappa = self.pair_kappa[affine_body, fem_body] / (active_gap2 * active_gap2)
                        _, first, _ = ipc_toolkit_barrier_distance2_offset_terms(
                            distance2,
                            dhat,
                            dmin,
                            normalized_kappa,
                            0,
                        )
                        scale = self.culling.edge_edge_measure[contact] * dhat
                        normal_force = ti.max(
                            0.0,
                            -scale * first * 2.0 * ti.sqrt(shifted),
                        )
                        coefficient = self.pair_friction[affine_body, fem_body] * normal_force
                        if coefficient > 0.0:
                            self.friction_ee_weight[contact] = weights
                            self.friction_ee_normal[contact] = normal
                            self.friction_ee_coefficient[contact] = coefficient
                            self.friction_ee_epsv[contact] = self.pair_epsv[affine_body, fem_body]
                            ti.atomic_add(self.friction_active_count[None], 1)

    @ti.kernel
    def _freeze_semi_friction(self, count: ti.i32, contact_kind: ti.template()):
        for contact in range(count):
            stencil = self.culling.point_triangle[contact]
            slot = self.semi_pt_slot[contact]
            affine_body, fem_body = self._pair_indices(stencil)
            measure = self.culling.point_triangle_measure[contact]
            if ti.static(contact_kind == 0):
                self.friction_pt_candidate[contact] = stencil
                self.friction_pt_weight[contact] = ti.Vector([1.0, 0.0, 0.0, 0.0])
                self.friction_pt_normal[contact] = ti.Vector([1.0, 0.0, 0.0])
                self.friction_pt_coefficient[contact] = 0.0
                self.friction_pt_epsv[contact] = 1.0
            else:
                stencil = self.culling.edge_edge[contact]
                slot = self.semi_ee_slot[contact]
                affine_body, fem_body = self._edge_pair_indices(stencil)
                measure = self.culling.edge_edge_measure[contact]
                self.friction_ee_candidate[contact] = stencil
                self.friction_ee_weight[contact] = ti.Vector([1.0, 0.0, 0.0, 0.0])
                self.friction_ee_normal[contact] = ti.Vector([1.0, 0.0, 0.0])
                self.friction_ee_coefficient[contact] = 0.0
                self.friction_ee_epsv[contact] = 1.0
            if slot >= 0 and self._pair_is_active(affine_body, fem_body):
                weight = self.semi_pt_weight[slot]
                normal = self.semi_pt_normal[slot]
                multiplier = self.semi_pt_multiplier[slot]
                if ti.static(contact_kind == 1):
                    weight = self.semi_ee_weight[slot]
                    normal = self.semi_ee_normal[slot]
                    multiplier = self.semi_ee_multiplier[slot]
                gap = -(self.pair_dmin[affine_body, fem_body] + self.pair_dhat[affine_body, fem_body])
                for site in ti.static(range(4)):
                    gap += weight[site] * normal.dot(self.position[stencil[site]])
                coefficient = (
                    self.pair_friction[affine_body, fem_body]
                    * measure
                    * ti.max(
                        multiplier - self.pair_penalty[affine_body, fem_body] * gap,
                        0.0,
                    )
                )
                if coefficient > 0.0:
                    if ti.static(contact_kind == 0):
                        self.friction_pt_weight[contact] = weight
                        self.friction_pt_normal[contact] = normal
                        self.friction_pt_coefficient[contact] = coefficient
                        self.friction_pt_epsv[contact] = self.pair_epsv[affine_body, fem_body]
                    else:
                        self.friction_ee_weight[contact] = weight
                        self.friction_ee_normal[contact] = normal
                        self.friction_ee_coefficient[contact] = coefficient
                        self.friction_ee_epsv[contact] = self.pair_epsv[affine_body, fem_body]
                    ti.atomic_add(self.friction_active_count[None], 1)

    def begin_step(self, fem_position, fem_hat_position, timestep):
        timestep = float(timestep)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("FEM--AffineBody friction timestep must be finite and positive")
        if self.is_semi:
            self.semi_pt_state.fill(2)
            self.semi_ee_state.fill(2)
            self.semi_pt_multiplier.fill(0.0)
            self.semi_ee_multiplier.fill(0.0)
            self.semi_pt_count[None] = 0
            self.semi_ee_count[None] = 0
        self.friction_dt[None] = timestep
        pt_count, ee_count = self.prepare(fem_position)
        self._copy_friction_hat(fem_hat_position)
        return self._freeze_friction(pt_count, ee_count)

    def refresh_friction(self, fem_position):
        """Refresh lagged frames and normal forces, preserving step hats."""
        pt_count, ee_count = self.prepare(fem_position)
        return self._freeze_friction(pt_count, ee_count)

    def _freeze_friction(self, pt_count, ee_count):
        self._reset_friction_state()
        if not self.activate_friction:
            self.friction_pt_count = 0
            self.friction_ee_count = 0
            return 0, 0
        self._ensure_friction_terms(pt_count, ee_count)
        if self.is_semi:
            self._freeze_semi_friction(int(pt_count), 0)
            self._freeze_semi_friction(int(ee_count), 1)
        else:
            self._prepare_distance_terms(pt_count, ee_count)
            self._freeze_point_triangle_friction(int(pt_count))
            self._freeze_edge_edge_friction(int(ee_count))
        self.friction_pt_count = int(pt_count)
        self.friction_ee_count = int(ee_count)
        return self.friction_pt_count, self.friction_ee_count

    @ti.kernel
    def _reset_contact_terms(self):
        self.total_energy[None] = 0.0
        self.friction_energy[None] = 0.0
        self.active_count[None] = 0
        self.minimum_distance[None] = 1.0e30

    @ti.kernel
    def _contact_feature_mask(self, pt_count: ti.i32, ee_count: ti.i32) -> ti.i32:
        mask = 0
        ti.loop_config(serialize=True)
        for contact in range(pt_count):
            stencil = self.culling.point_triangle[contact]
            affine_body, fem_body = self._pair_indices(stencil)
            if self._pair_is_active(affine_body, fem_body):
                contact_type = point_triangle_distance_type(
                    self.position[stencil[0]],
                    self.position[stencil[1]],
                    self.position[stencil[2]],
                    self.position[stencil[3]],
                )
                mask |= 1 << contact_type
        ti.loop_config(serialize=True)
        for contact in range(ee_count):
            stencil = self.culling.edge_edge[contact]
            affine_body, fem_body = self._edge_pair_indices(stencil)
            if self._pair_is_active(affine_body, fem_body):
                contact_type = edge_edge_distance_type(
                    self.position[stencil[0]],
                    self.position[stencil[1]],
                    self.position[stencil[2]],
                    self.position[stencil[3]],
                )
                mask |= 1 << (7 + contact_type)
        return mask

    @ti.func
    def _store_pt_compact(self, contact, gradient, sites, count):
        for local in range(count):
            site = sites[local]
            for component in range(3):
                self.pt_gradient[contact, 3 * site + component] = gradient[3 * local + component]

    @ti.func
    def _store_ee_compact(self, contact, gradient, sites, count):
        for local in range(count):
            site = sites[local]
            for component in range(3):
                self.ee_gradient[contact, 3 * site + component] = gradient[3 * local + component]

    @ti.kernel
    def _classify_contact_features(
        self,
        pt_count: ti.i32,
        ee_count: ti.i32,
    ):
        for contact in range(pt_count):
            self.pt_active[contact] = 0
            self.pt_distance_type[contact] = -1
            for row in range(12):
                self.pt_gradient[contact, row] = 0.0
            stencil = self.culling.point_triangle[contact]
            affine_body, fem_body = self._pair_indices(stencil)
            if self._pair_is_active(affine_body, fem_body):
                self.pt_distance_type[contact] = point_triangle_distance_type(
                    self.position[stencil[0]],
                    self.position[stencil[1]],
                    self.position[stencil[2]],
                    self.position[stencil[3]],
                )
        for contact in range(ee_count):
            self.ee_active[contact] = 0
            self.ee_distance_type[contact] = -1
            for row in range(12):
                self.ee_gradient[contact, row] = 0.0
            stencil = self.culling.edge_edge[contact]
            affine_body, fem_body = self._edge_pair_indices(stencil)
            if self._pair_is_active(affine_body, fem_body):
                self.ee_distance_type[contact] = edge_edge_distance_type(
                    self.position[stencil[0]],
                    self.position[stencil[1]],
                    self.position[stencil[2]],
                    self.position[stencil[3]],
                )

    @ti.kernel
    def _differentiate_point_triangle(self, pt_count: ti.i32, feature: ti.template()):
        for contact in range(pt_count):
            dtype = self.pt_distance_type[contact]
            stencil = self.culling.point_triangle[contact]
            if ti.static(feature == 0):
                if 0 <= dtype and dtype <= 2:
                    sites = ti.Vector([0, dtype + 1])
                    first = self.position[stencil[sites[0]]]
                    second = self.position[stencil[sites[1]]]
                    self.pt_distance2[contact] = point_point_distance2(first, second)
                    gradient = point_point_grad(first, second)
                    self._store_pt_compact(contact, gradient, sites, 2)
            elif ti.static(feature == 1):
                if 3 <= dtype and dtype <= 5:
                    first_site = 1
                    second_site = 2
                    if dtype == 4:
                        first_site, second_site = 2, 3
                    elif dtype == 5:
                        first_site, second_site = 1, 3
                    sites = ti.Vector([0, first_site, second_site])
                    point = self.position[stencil[0]]
                    first = self.position[stencil[first_site]]
                    second = self.position[stencil[second_site]]
                    self.pt_distance2[contact] = point_line_distance2(point, first, second)
                    gradient = g_PE3D(point, first, second)
                    self._store_pt_compact(contact, gradient, sites, 3)
            else:
                if dtype == 6:
                    p = self.position[stencil[0]]
                    t0 = self.position[stencil[1]]
                    t1 = self.position[stencil[2]]
                    t2 = self.position[stencil[3]]
                    self.pt_distance2[contact] = point_plane_distance2(p, t0, t1, t2)
                    gradient = g_PT(p, t0, t1, t2)
                    for row in range(12):
                        self.pt_gradient[contact, row] = gradient[row]

    @ti.kernel
    def _differentiate_edge_edge(self, ee_count: ti.i32, feature: ti.template()):
        for contact in range(ee_count):
            dtype = self.ee_distance_type[contact]
            stencil = self.culling.edge_edge[contact]
            if ti.static(feature == 0):
                if dtype == 0 or dtype == 1 or dtype == 3 or dtype == 4:
                    first_site = 0 if dtype <= 1 else 1
                    second_site = 2 if dtype == 0 or dtype == 3 else 3
                    sites = ti.Vector([first_site, second_site])
                    first = self.position[stencil[first_site]]
                    second = self.position[stencil[second_site]]
                    self.ee_distance2[contact] = point_point_distance2(first, second)
                    gradient = point_point_grad(first, second)
                    self._store_ee_compact(contact, gradient, sites, 2)
            elif ti.static(feature == 1):
                if dtype == 2 or dtype == 5 or dtype == 6 or dtype == 7:
                    point_site = 0
                    first_site, second_site = 2, 3
                    if dtype == 5:
                        point_site = 1
                    elif dtype == 6:
                        point_site = 2
                        first_site, second_site = 0, 1
                    elif dtype == 7:
                        point_site = 3
                        first_site, second_site = 0, 1
                    sites = ti.Vector([point_site, first_site, second_site])
                    point = self.position[stencil[point_site]]
                    first = self.position[stencil[first_site]]
                    second = self.position[stencil[second_site]]
                    self.ee_distance2[contact] = point_line_distance2(point, first, second)
                    gradient = g_PE3D(point, first, second)
                    self._store_ee_compact(contact, gradient, sites, 3)
            else:
                if dtype == 8:
                    p0 = self.position[stencil[0]]
                    p1 = self.position[stencil[1]]
                    q0 = self.position[stencil[2]]
                    q1 = self.position[stencil[3]]
                    self.ee_distance2[contact] = line_line_distance2(p0, p1, q0, q1)
                    gradient = g_EE(p0, p1, q0, q1)
                    for row in range(12):
                        self.ee_gradient[contact, row] = gradient[row]

    @ti.kernel
    def _compute_lagged_friction_gradient(
        self,
        count: ti.i32,
        contact_kind: ti.template(),
    ):
        timestep = self.friction_dt[None]
        for contact in range(count):
            stencil = self.friction_pt_candidate[contact]
            weights = self.friction_pt_weight[contact]
            normal = self.friction_pt_normal[contact]
            coefficient = self.friction_pt_coefficient[contact]
            epsv = self.friction_pt_epsv[contact]
            if ti.static(contact_kind == 1):
                stencil = self.friction_ee_candidate[contact]
                weights = self.friction_ee_weight[contact]
                normal = self.friction_ee_normal[contact]
                coefficient = self.friction_ee_coefficient[contact]
                epsv = self.friction_ee_epsv[contact]
            relative_gradient = ti.Vector.zero(self.real_type, 3)
            if coefficient > 0.0:
                relative_increment = ti.Vector.zero(self.real_type, 3)
                for site in range(4):
                    relative_increment += weights[site] * (
                        self.position[stencil[site]] - self.friction_hat[stencil[site]]
                    )
                velocity = (relative_increment - normal * normal.dot(relative_increment)) / timestep
                speed = velocity.norm()
                energy = coefficient * ipc_friction_f0(speed, epsv, timestep)
                ti.atomic_add(self.total_energy[None], energy)
                ti.atomic_add(self.friction_energy[None], energy)
                profile = ipc_friction_f1_over_speed(speed, epsv)
                relative_gradient = coefficient * profile * velocity
            for component in ti.static(range(3)):
                if ti.static(contact_kind == 0):
                    self.friction_pt_gradient[contact, component] = relative_gradient[component]
                else:
                    self.friction_ee_gradient[contact, component] = relative_gradient[component]

    @ti.kernel
    def _scatter_lagged_friction_gradient(
        self,
        count: ti.i32,
        contact_kind: ti.template(),
        rhs: ti.template(),
    ):
        for contact in range(count):
            stencil = self.friction_pt_candidate[contact]
            weights = self.friction_pt_weight[contact]
            if ti.static(contact_kind == 1):
                stencil = self.friction_ee_candidate[contact]
                weights = self.friction_ee_weight[contact]
            for site in range(4):
                node = stencil[site]
                for support in range(self.support_count[node]):
                    block = self.support_block[node][support]
                    scale = -self.support_weight[node][support] * weights[site]
                    for component in ti.static(range(3)):
                        value = self.friction_pt_gradient[contact, component]
                        if ti.static(contact_kind == 1):
                            value = self.friction_ee_gradient[contact, component]
                        ti.atomic_add(rhs[3 * block + component], scale * value)

    @ti.kernel
    def _compute_lagged_friction_hessian(
        self,
        count: ti.i32,
        contact_kind: ti.template(),
    ):
        timestep = self.friction_dt[None]
        for contact in range(count):
            stencil = self.friction_pt_candidate[contact]
            weights = self.friction_pt_weight[contact]
            normal = self.friction_pt_normal[contact]
            coefficient = self.friction_pt_coefficient[contact]
            epsv = self.friction_pt_epsv[contact]
            if ti.static(contact_kind == 1):
                stencil = self.friction_ee_candidate[contact]
                weights = self.friction_ee_weight[contact]
                normal = self.friction_ee_normal[contact]
                coefficient = self.friction_ee_coefficient[contact]
                epsv = self.friction_ee_epsv[contact]
            mode = 0
            profile = 0.0
            radial = 0.0
            direction = ti.Vector.zero(self.real_type, 3)
            perpendicular = ti.Vector.zero(self.real_type, 3)
            if coefficient > 0.0:
                relative_increment = ti.Vector.zero(self.real_type, 3)
                for site in range(4):
                    relative_increment += weights[site] * (
                        self.position[stencil[site]] - self.friction_hat[stencil[site]]
                    )
                velocity = (relative_increment - normal * normal.dot(relative_increment)) / timestep
                speed = velocity.norm()
                profile = ipc_friction_f1_over_speed(speed, epsv)
                if speed > 1.0e-15:
                    direction = velocity / speed
                    perpendicular = normal.cross(direction)
                    perpendicular_norm = perpendicular.norm()
                    if perpendicular_norm > 1.0e-15:
                        perpendicular /= perpendicular_norm
                        if speed < epsv:
                            radial = 2.0 * (epsv - speed) / (epsv * epsv)
                        mode = 1
                else:
                    mode = 2
            for row, column in ti.static(ti.ndrange(3, 3)):
                value = 0.0
                if mode == 1:
                    value = (
                        coefficient
                        * (
                            profile * perpendicular[row] * perpendicular[column]
                            + radial * direction[row] * direction[column]
                        )
                        / timestep
                    )
                elif mode == 2:
                    value = (
                        coefficient
                        * profile
                        * (ti.cast(row == column, self.real_type) - normal[row] * normal[column])
                        / timestep
                    )
                if ti.static(contact_kind == 0):
                    self.friction_pt_hessian[contact, row, column] = value
                else:
                    self.friction_ee_hessian[contact, row, column] = value

    @ti.kernel
    def _scatter_lagged_friction_hessian(
        self,
        count: ti.i32,
        contact_kind: ti.template(),
        matrix: ti.template(),
    ):
        for contact in range(count):
            stencil = self.friction_pt_candidate[contact]
            weights = self.friction_pt_weight[contact]
            if ti.static(contact_kind == 1):
                stencil = self.friction_ee_candidate[contact]
                weights = self.friction_ee_weight[contact]
            for first, second in ti.ndrange(4, 4):
                block = ti.Matrix.zero(self.real_type, 3, 3)
                for row, column in ti.static(ti.ndrange(3, 3)):
                    value = self.friction_pt_hessian[contact, row, column]
                    if ti.static(contact_kind == 1):
                        value = self.friction_ee_hessian[contact, row, column]
                    block[row, column] = weights[first] * weights[second] * value
                self._scatter_hessian_block(
                    stencil,
                    first,
                    second,
                    block,
                    matrix,
                )

    @ti.kernel
    def _assemble_point_triangle_barrier_direct_type(
        self,
        count: ti.i32,
        contact_type: ti.template(),
        matrix: ti.template(),
        rhs: ti.template(),
        need_matrix: ti.template(),
        project_spd: ti.template(),
    ):
        for contact in range(count):
            stencil = self.culling.point_triangle[contact]
            affine_body, fem_body = self._pair_indices(stencil)
            if self._pair_is_active(affine_body, fem_body):
                p = self.position[stencil[0]]
                t0 = self.position[stencil[1]]
                t1 = self.position[stencil[2]]
                t2 = self.position[stencil[3]]
                matches = 1
                if ti.static(need_matrix):
                    matches = ti.cast(
                        point_triangle_distance_type(p, t0, t1, t2) == contact_type,
                        ti.i32,
                    )
                if matches != 0:
                    distance2 = 0.0
                    gradient_d = ti.Vector.zero(self.real_type, 12)
                    hessian_d = ti.Matrix.zero(self.real_type, 12, 12)
                    if ti.static(need_matrix):
                        distance2, gradient_d, hessian_d = point_triangle_distance_grad_hess_by_type(
                            p, t0, t1, t2, contact_type
                        )
                    else:
                        distance2, gradient_d, _ = point_triangle_distance_grad(p, t0, t1, t2)
                    dmin = self.pair_dmin[affine_body, fem_body]
                    dhat = self.pair_dhat[affine_body, fem_body]
                    shifted = distance2 - dmin * dmin
                    active_gap2 = (2.0 * dmin + dhat) * dhat
                    ti.atomic_min(
                        self.minimum_distance[None],
                        ti.sqrt(ti.max(distance2, 0.0)) - dmin,
                    )
                    if shifted < active_gap2:
                        normalized_kappa = self.pair_kappa[affine_body, fem_body] / (active_gap2 * active_gap2)
                        energy, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                            distance2, dhat, dmin, normalized_kappa, 0
                        )
                        scale = self.culling.point_triangle_measure[contact] * dhat
                        ti.atomic_add(self.total_energy[None], scale * energy)
                        ti.atomic_add(self.active_count[None], 1)
                        self._scatter_gradient(stencil, scale * first * gradient_d, rhs)
                        if ti.static(need_matrix):
                            if shifted > 0.0:
                                local_hessian = ti.Matrix.zero(self.real_type, 12, 12)
                                for row, column in ti.ndrange(12, 12):
                                    local_hessian[row, column] = scale * (
                                        first * hessian_d[row, column] + second * gradient_d[row] * gradient_d[column]
                                    )
                                if ti.static(project_spd):
                                    local_hessian = psd_project_nd(local_hessian)
                                for local_i, local_j in ti.static(ti.ndrange(4, 4)):
                                    block = ti.Matrix.zero(self.real_type, 3, 3)
                                    for row, column in ti.static(ti.ndrange(3, 3)):
                                        block[row, column] = local_hessian[3 * local_i + row, 3 * local_j + column]
                                    self._scatter_hessian_block(stencil, local_i, local_j, block, matrix)

    @ti.kernel
    def _assemble_edge_edge_barrier_direct_type(
        self,
        count: ti.i32,
        contact_type: ti.template(),
        matrix: ti.template(),
        rhs: ti.template(),
        need_matrix: ti.template(),
        project_spd: ti.template(),
    ):
        for contact in range(count):
            stencil = self.culling.edge_edge[contact]
            affine_body, fem_body = self._edge_pair_indices(stencil)
            if self._pair_is_active(affine_body, fem_body):
                p0 = self.position[stencil[0]]
                p1 = self.position[stencil[1]]
                q0 = self.position[stencil[2]]
                q1 = self.position[stencil[3]]
                matches = 1
                if ti.static(need_matrix):
                    matches = ti.cast(
                        edge_edge_distance_type(p0, p1, q0, q1) == contact_type,
                        ti.i32,
                    )
                if matches != 0:
                    distance2 = 0.0
                    gradient_d = ti.Vector.zero(self.real_type, 12)
                    hessian_d = ti.Matrix.zero(self.real_type, 12, 12)
                    if ti.static(need_matrix):
                        distance2, gradient_d, hessian_d = edge_edge_distance_grad_hess_by_type(
                            p0, p1, q0, q1, contact_type
                        )
                    else:
                        distance2, gradient_d, _ = edge_edge_distance_grad(p0, p1, q0, q1)
                    dmin = self.pair_dmin[affine_body, fem_body]
                    dhat = self.pair_dhat[affine_body, fem_body]
                    shifted = distance2 - dmin * dmin
                    active_gap2 = (2.0 * dmin + dhat) * dhat
                    ti.atomic_min(
                        self.minimum_distance[None],
                        ti.sqrt(ti.max(distance2, 0.0)) - dmin,
                    )
                    if shifted < active_gap2:
                        normalized_kappa = self.pair_kappa[affine_body, fem_body] / (active_gap2 * active_gap2)
                        barrier, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                            distance2, dhat, dmin, normalized_kappa, 0
                        )
                        threshold = edge_edge_mollifier_threshold(
                            self.culling.reference_position[stencil[0]],
                            self.culling.reference_position[stencil[1]],
                            self.culling.reference_position[stencil[2]],
                            self.culling.reference_position[stencil[3]],
                        )
                        mollifier = 0.0
                        gradient_m = ti.Vector.zero(self.real_type, 12)
                        hessian_m = ti.Matrix.zero(self.real_type, 12, 12)
                        if ti.static(need_matrix):
                            mollifier, gradient_m, hessian_m = edge_edge_mollifier_terms(p0, p1, q0, q1, threshold)
                        else:
                            mollifier = edge_edge_mollifier(p0, p1, q0, q1, threshold)
                            gradient_m = edge_edge_mollifier_grad(p0, p1, q0, q1, threshold)
                        gradient_b = first * gradient_d
                        scale = self.culling.edge_edge_measure[contact] * dhat
                        gradient = scale * (mollifier * gradient_b + barrier * gradient_m)
                        ti.atomic_add(self.total_energy[None], scale * mollifier * barrier)
                        ti.atomic_add(self.active_count[None], 1)
                        self._scatter_gradient(stencil, gradient, rhs)
                        if ti.static(need_matrix):
                            if shifted > 0.0:
                                local_hessian = ti.Matrix.zero(self.real_type, 12, 12)
                                for row, column in ti.ndrange(12, 12):
                                    hessian_b = (
                                        first * hessian_d[row, column] + second * gradient_d[row] * gradient_d[column]
                                    )
                                    local_hessian[row, column] = scale * (
                                        mollifier * hessian_b
                                        + barrier * hessian_m[row, column]
                                        + gradient_m[row] * gradient_b[column]
                                        + gradient_b[row] * gradient_m[column]
                                    )
                                if ti.static(project_spd):
                                    local_hessian = psd_project_nd(local_hessian)
                                for local_i, local_j in ti.static(ti.ndrange(4, 4)):
                                    block = ti.Matrix.zero(self.real_type, 3, 3)
                                    for row, column in ti.static(ti.ndrange(3, 3)):
                                        block[row, column] = local_hessian[3 * local_i + row, 3 * local_j + column]
                                    self._scatter_hessian_block(stencil, local_i, local_j, block, matrix)

    @ti.kernel
    def _assemble_semi_contacts(
        self,
        count: ti.i32,
        contact_kind: ti.template(),
        matrix: ti.template(),
        rhs: ti.template(),
        need_matrix: ti.template(),
    ):
        for contact in range(count):
            stencil = self.culling.point_triangle[contact]
            slot = self.semi_pt_slot[contact]
            affine_body, fem_body = self._pair_indices(stencil)
            weight = ti.Vector.zero(self.real_type, 4)
            normal = ti.Vector([1.0, 0.0, 0.0])
            multiplier = 0.0
            measure = self.culling.point_triangle_measure[contact]
            if slot >= 0:
                weight = self.semi_pt_weight[slot]
                normal = self.semi_pt_normal[slot]
                multiplier = self.semi_pt_multiplier[slot]
            if ti.static(contact_kind == 1):
                stencil = self.culling.edge_edge[contact]
                slot = self.semi_ee_slot[contact]
                affine_body, fem_body = self._edge_pair_indices(stencil)
                measure = self.culling.edge_edge_measure[contact]
                if slot >= 0:
                    weight = self.semi_ee_weight[slot]
                    normal = self.semi_ee_normal[slot]
                    multiplier = self.semi_ee_multiplier[slot]
            if slot >= 0 and self._pair_is_active(affine_body, fem_body):
                gap = -(self.pair_dmin[affine_body, fem_body] + self.pair_dhat[affine_body, fem_body])
                for site in ti.static(range(4)):
                    gap += weight[site] * normal.dot(self.position[stencil[site]])
                value, first, second = semi_ipc_terms(gap, multiplier, self.pair_penalty[affine_body, fem_body])
                ti.atomic_add(self.total_energy[None], measure * value)
                ti.atomic_max(self.constraint_violation[None], ti.max(-gap, 0.0))
                ti.atomic_min(self.minimum_distance[None], gap)
                if second > 0.0:
                    ti.atomic_add(self.active_count[None], 1)
                    gradient = ti.Vector.zero(self.real_type, 12)
                    for site, component in ti.static(ti.ndrange(4, 3)):
                        gradient[3 * site + component] = first * weight[site] * normal[component]
                    self._scatter_gradient(stencil, measure * gradient, rhs)
                    if ti.static(need_matrix):
                        for first_site, second_site in ti.static(ti.ndrange(4, 4)):
                            block = (
                                measure
                                * second
                                * weight[first_site]
                                * weight[second_site]
                                * normal.outer_product(normal)
                            )
                            self._scatter_hessian_block(stencil, first_site, second_site, block, matrix)

    def assemble(self, pt_count, ee_count, matrix, rhs, need_matrix, project_spd=True):
        self._reset_contact_terms()
        pt_count = int(pt_count)
        ee_count = int(ee_count)
        need_matrix = bool(need_matrix)
        if self.is_semi:
            self.constraint_violation[None] = 0.0
            self._assemble_semi_contacts(pt_count, 0, matrix, rhs, need_matrix)
            self._assemble_semi_contacts(ee_count, 1, matrix, rhs, need_matrix)
        elif need_matrix:
            # Keep each CUDA module below the driver/compiler failure seen for
            # the branch-complete PT/EE Hessian kernels, and do not compile
            # feature types absent from this active set.
            feature_mask = int(self._contact_feature_mask(pt_count, ee_count)) if pt_count or ee_count else 0
            for contact_type in range(7):
                if feature_mask & (1 << contact_type):
                    self._assemble_point_triangle_barrier_direct_type(
                        pt_count, contact_type, matrix, rhs, True, bool(project_spd)
                    )
            for contact_type in range(9):
                if feature_mask & (1 << (7 + contact_type)):
                    self._assemble_edge_edge_barrier_direct_type(
                        ee_count, contact_type, matrix, rhs, True, bool(project_spd)
                    )
        else:
            self._assemble_point_triangle_barrier_direct_type(pt_count, 0, matrix, rhs, False, False)
            self._assemble_edge_edge_barrier_direct_type(ee_count, 0, matrix, rhs, False, False)
        if self.activate_friction and int(self.friction_pt_count) > 0:
            self._compute_lagged_friction_gradient(
                int(self.friction_pt_count),
                0,
            )
            self._scatter_lagged_friction_gradient(
                int(self.friction_pt_count),
                0,
                rhs,
            )
            if need_matrix:
                self._compute_lagged_friction_hessian(
                    int(self.friction_pt_count),
                    0,
                )
                self._scatter_lagged_friction_hessian(
                    int(self.friction_pt_count),
                    0,
                    matrix,
                )
        if self.activate_friction and int(self.friction_ee_count) > 0:
            self._compute_lagged_friction_gradient(
                int(self.friction_ee_count),
                1,
            )
            self._scatter_lagged_friction_gradient(
                int(self.friction_ee_count),
                1,
                rhs,
            )
            if need_matrix:
                self._compute_lagged_friction_hessian(
                    int(self.friction_ee_count),
                    1,
                )
                self._scatter_lagged_friction_hessian(
                    int(self.friction_ee_count),
                    1,
                    matrix,
                )

    @ti.kernel
    def _compute_pair_ccd(self, pt_count: ti.i32, ee_count: ti.i32):
        self.minimum_step[None] = 1.0
        for contact in range(pt_count):
            stencil = self.culling.point_triangle[contact]
            affine_body, fem_body = self._pair_indices(stencil)
            if self._pair_is_active(affine_body, fem_body):
                thickness = self.pair_dmin[affine_body, fem_body]
                if ti.static(self.is_semi):
                    thickness = 0.0
                toc = 1.0
                if thickness > 0.0:
                    toc = point_triangle_accd(
                        self.position[stencil[0]],
                        self.position[stencil[1]],
                        self.position[stencil[2]],
                        self.position[stencil[3]],
                        self.end_position[stencil[0]] - self.position[stencil[0]],
                        self.end_position[stencil[1]] - self.position[stencil[1]],
                        self.end_position[stencil[2]] - self.position[stencil[2]],
                        self.end_position[stencil[3]] - self.position[stencil[3]],
                        ti.static(self.ccd_eta),
                        thickness,
                        ti.static(self.ccd_max_iterations),
                    )
                else:
                    toc = point_triangle_ccd(
                        self.position[stencil[0]],
                        self.position[stencil[1]],
                        self.position[stencil[2]],
                        self.position[stencil[3]],
                        self.end_position[stencil[0]] - self.position[stencil[0]],
                        self.end_position[stencil[1]] - self.position[stencil[1]],
                        self.end_position[stencil[2]] - self.position[stencil[2]],
                        self.end_position[stencil[3]] - self.position[stencil[3]],
                        ti.static(self.ccd_eta),
                        ti.static(self.ccd_max_iterations),
                    )
                ti.atomic_min(self.minimum_step[None], toc)
        for contact in range(ee_count):
            stencil = self.culling.edge_edge[contact]
            affine_body, fem_body = self._edge_pair_indices(stencil)
            if self._pair_is_active(affine_body, fem_body):
                thickness = self.pair_dmin[affine_body, fem_body]
                if ti.static(self.is_semi):
                    thickness = 0.0
                toc = 1.0
                if thickness > 0.0:
                    toc = edge_edge_accd(
                        self.position[stencil[0]],
                        self.position[stencil[1]],
                        self.position[stencil[2]],
                        self.position[stencil[3]],
                        self.end_position[stencil[0]] - self.position[stencil[0]],
                        self.end_position[stencil[1]] - self.position[stencil[1]],
                        self.end_position[stencil[2]] - self.position[stencil[2]],
                        self.end_position[stencil[3]] - self.position[stencil[3]],
                        ti.static(self.ccd_eta),
                        thickness,
                        ti.static(self.ccd_max_iterations),
                    )
                else:
                    toc = edge_edge_ccd(
                        self.position[stencil[0]],
                        self.position[stencil[1]],
                        self.position[stencil[2]],
                        self.position[stencil[3]],
                        self.end_position[stencil[0]] - self.position[stencil[0]],
                        self.end_position[stencil[1]] - self.position[stencil[1]],
                        self.end_position[stencil[2]] - self.position[stencil[2]],
                        self.end_position[stencil[3]] - self.position[stencil[3]],
                        ti.static(self.ccd_eta),
                        ti.static(self.ccd_max_iterations),
                    )
                ti.atomic_min(self.minimum_step[None], toc)

    def maximum_step(self, fem_position, fem_direction):
        self.affine._reconstruct_vertices()
        self.build_positions(fem_position)
        self.build_end_positions(fem_position, fem_direction)
        pt_count, ee_count = self.culling.rebuild_swept_candidates(
            self.position, self.end_position, self.search_distance
        )
        self._compute_pair_ccd(pt_count, ee_count)
        return max(0.0, min(1.0, float(self.minimum_step[None])))

    @ti.kernel
    def _update_semi_contact_multipliers(self, count: ti.i32, contact_kind: ti.template()):
        for contact in range(count):
            stencil = self.culling.point_triangle[contact]
            slot = self.semi_pt_slot[contact]
            affine_body, fem_body = self._pair_indices(stencil)
            if ti.static(contact_kind == 1):
                stencil = self.culling.edge_edge[contact]
                slot = self.semi_ee_slot[contact]
                affine_body, fem_body = self._edge_pair_indices(stencil)
            if slot >= 0 and self._pair_is_active(affine_body, fem_body):
                weight = self.semi_pt_weight[slot]
                normal = self.semi_pt_normal[slot]
                multiplier = self.semi_pt_multiplier[slot]
                if ti.static(contact_kind == 1):
                    weight = self.semi_ee_weight[slot]
                    normal = self.semi_ee_normal[slot]
                    multiplier = self.semi_ee_multiplier[slot]
                gap = -(self.pair_dmin[affine_body, fem_body] + self.pair_dhat[affine_body, fem_body])
                for site in ti.static(range(4)):
                    gap += weight[site] * normal.dot(self.position[stencil[site]])
                updated = semi_ipc_update_multiplier(gap, multiplier, self.pair_penalty[affine_body, fem_body])
                if ti.static(contact_kind == 0):
                    self.semi_pt_multiplier[slot] = updated
                else:
                    self.semi_ee_multiplier[slot] = updated

    def accept_update(self, fem_position):
        if not self.is_semi:
            return
        pt_count, ee_count = self.prepare(fem_position)
        self.constraint_violation[None] = 0.0
        self._update_semi_contact_multipliers(int(pt_count), 0)
        self._update_semi_contact_multipliers(int(ee_count), 1)

    def contact_converged(self):
        return not self.is_semi or float(self.constraint_violation[None]) <= self.constraint_tolerance

    def contact_block_capacity(self, pt_count, ee_count):
        # A PT stencil pulls back to at most 13 global blocks (one FEM point
        # against three affine vertices), hence 13^2=169 block scatters.  An
        # EE stencil has at most ten blocks and therefore 10^2=100 scatters.
        # Barrier and frozen-friction Hessians are separate contributions.
        pt_contributions = int(pt_count)
        ee_contributions = int(ee_count)
        if self.activate_friction:
            pt_contributions += max(int(pt_count), int(self.friction_pt_count))
            ee_contributions += max(int(ee_count), int(self.friction_ee_count))
        return max(1, 169 * pt_contributions + 100 * ee_contributions)

    def diagnostics(self):
        minimum = float(self.minimum_distance[None])
        if minimum > 0.5e30:
            minimum = np.inf
        values = self.culling.diagnostics()
        values.update(
            model="SemiIPC" if self.is_semi else "BarrierIPC",
            active_contacts=int(self.active_count[None]),
            friction_contacts=int(self.friction_active_count[None]),
            minimum_distance=minimum,
            ccd_step=float(self.minimum_step[None]),
            energy=float(self.total_energy[None]),
            friction_energy=float(self.friction_energy[None]),
            constraint_violation=float(self.constraint_violation[None]) if self.is_semi else 0.0,
        )
        return values


__all__ = ["FEMAffineIPCAssembler"]
