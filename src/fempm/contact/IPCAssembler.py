"""Taichi point--triangle IPC assembly for implicit FEM--MPM coupling."""

import numpy as np
import taichi as ti

from src.contact_detection.continuous_contact_detection.AdditiveCCD import (
    point_edge_accd,
    point_triangle_accd,
)
from src.contact_detection.continuous_contact_detection.CCD import (
    point_edge_ccd,
    point_triangle_ccd,
)
from src.fem.contact.BVHBroadPhase import DynamicBVHBroadPhase
from src.fem.contact.CollisionCulling import FEMCollisionCulling
from src.fem.contact.LinkedCellBroadPhase import (
    DynamicLinkedCellBroadPhase,
)
from src.fempm.contact.PlanarCulling import FEMPMPlanarCulling
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd
from src.physics_model.contact_model.ipc.ContactDistance import (
    point_edge_distance_grad_hess_2d,
    point_triangle_distance_grad,
    point_triangle_distance_grad_hess,
)
from src.physics_model.contact_model.ipc.IPC import (
    ipc_friction_f0,
    ipc_friction_f1_over_speed,
    ipc_friction_hessian_term,
    ipc_toolkit_barrier_distance2_offset_terms,
    semi_ipc_find_or_insert,
    semi_ipc_terms,
    semi_ipc_update_multiplier,
)


@ti.data_oriented
class FEMPMIPCAssembler:
    """Device-resident IPC contact and MPM-grid pullback.

    Collision geometry is a point sampled from the current MPM interpolation
    and one deforming FEM triangle.  The 12-dimensional geometric derivatives
    are pulled back analytically to FEM nodes and compact active MPM grid
    nodes; no particle/contact arrays cross the host during assembly.
    """

    def __init__(self, fem, mpm, faces, face_body, model, simulation):
        self.fem = fem
        self.mpm = mpm
        self.model = model
        search = simulation.search
        self.max_point_triangle_pairs = int(simulation.max_point_triangle_pairs)
        self.max_point_edge_pairs = int(simulation.max_point_edge_pairs)
        self.real_type = ti.lang.impl.current_cfg().default_fp
        if self.real_type != ti.f64:
            raise RuntimeError("FEMPM IPC requires Taichi default_fp=ti.f64")
        self.fem_node_count = int(fem.mesh.number_of_nodes)
        self.dimension = int(getattr(mpm, "dimension", 3))
        if self.dimension not in (2, 3):
            raise RuntimeError("FEMPM IPC requires dimension=2 or 3")
        self.fem_site_count = 2 if self.dimension == 2 else 3
        self.stencil_size = self.fem_site_count + 1
        self.surface_count = int(mpm.total_surface_num)
        if self.surface_count <= 0:
            raise RuntimeError("FEMPM IPC requires at least one MPM surface point")
        self.combined_node_count = self.surface_count + self.fem_node_count

        faces = np.ascontiguousarray(faces, dtype=np.int32)
        face_body = np.ascontiguousarray(face_body, dtype=np.int32).reshape(-1)
        if faces.ndim != 2 or faces.shape[1] != self.fem_site_count or face_body.size != faces.shape[0]:
            raise ValueError("FEMPM IPC boundary primitives/body ids have invalid shapes")
        shifted_faces = faces + self.surface_count
        surface_ids = np.asarray(mpm.surface_id.to_numpy(), dtype=np.int32)
        particle_position = np.asarray(mpm.particle.x.to_numpy(), dtype=np.float64)
        if self.dimension == 2:
            embedded_particle_position = np.zeros((particle_position.shape[0], 3), dtype=np.float64)
            embedded_particle_position[:, :2] = particle_position
            particle_position = embedded_particle_position
        reference = np.concatenate(
            (particle_position[surface_ids], np.asarray(fem.mesh.points, dtype=np.float64)),
            axis=0,
        )
        node_system = np.concatenate(
            (
                np.zeros(self.surface_count, dtype=np.int32),
                np.ones(self.fem_node_count, dtype=np.int32),
            )
        )
        node_area = np.zeros(self.combined_node_count, dtype=np.float64)
        vertices = np.arange(self.surface_count, dtype=np.int32)
        empty_edges = np.empty((0, 2), dtype=np.int32)
        broad_phase_type = (
            DynamicBVHBroadPhase
            if str(search).replace("_", "").replace("-", "").lower() == "bvh"
            else DynamicLinkedCellBroadPhase
        )
        if self.dimension == 3:
            node_area[: self.surface_count] = 4.0 * np.asarray(mpm.surface_measure.to_numpy(), dtype=np.float64)
            self.broad_phase = broad_phase_type(
                shifted_faces,
                empty_edges,
                vertices,
                node_area,
                np.empty(0, dtype=np.float64),
                reference,
                max_point_triangle_pairs=self.max_point_triangle_pairs,
                max_edge_edge_pairs=1,
                node_system_ids=node_system,
                cross_system_only=True,
            )
            self.culling = FEMCollisionCulling(
                self.broad_phase,
                self.combined_node_count,
                node_system_ids=node_system,
                cross_system_only=True,
                max_point_triangle_pairs=self.max_point_triangle_pairs,
                max_edge_edge_pairs=1,
            )
        else:
            point_edges = np.column_stack((vertices, vertices)).astype(np.int32)
            combined_edges = np.concatenate((point_edges, shifted_faces), axis=0)
            self.broad_phase = broad_phase_type(
                np.empty((0, 3), dtype=np.int32),
                combined_edges,
                np.empty(0, dtype=np.int32),
                node_area,
                np.zeros(combined_edges.shape[0], dtype=np.float64),
                reference,
                max_point_triangle_pairs=1,
                max_edge_edge_pairs=self.max_point_edge_pairs,
                node_system_ids=node_system,
                cross_system_only=True,
            )
            self.culling = FEMPMPlanarCulling(
                self.broad_phase,
                self.surface_count,
                max_point_edge_pairs=self.max_point_edge_pairs,
            )
        self.positions = ti.Vector.field(3, dtype=self.real_type, shape=self.combined_node_count)
        self.end_positions = ti.Vector.field(3, dtype=self.real_type, shape=self.combined_node_count)
        self.friction_hat = ti.Vector.field(3, dtype=self.real_type, shape=self.combined_node_count)
        node_body = np.asarray(fem.mesh.node_body_ids, dtype=np.int32)
        self.fem_node_body = ti.field(dtype=ti.i32, shape=self.fem_node_count)
        self.fem_node_body.from_numpy(np.ascontiguousarray(node_body))

        characteristic = float(np.linalg.norm(np.ptp(reference, axis=0)))
        parameters = model.parameter_arrays(int(mpm.n_body), int(np.max(face_body)) + 1, characteristic)
        self.mpm_body_count, self.fem_body_count = parameters["active"].shape
        pair_shape = (self.mpm_body_count, self.fem_body_count)
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
        self.ccd_safety = float(parameters["ccd_safety"])
        self.ccd_max_iterations = int(parameters["ccd_max_iterations"])
        self.activate_friction = bool(model.activate_friction)
        self.is_semi = model.contact.model == "AugmentedLagrangian"
        self.constraint_tolerance = float(model.contact.constraint_tolerance)

        self.total_energy = ti.field(dtype=self.real_type, shape=())
        self.minimum_step = ti.field(dtype=self.real_type, shape=())
        self.minimum_step.fill(1.0)
        self.active_count = ti.field(dtype=ti.i32, shape=())
        self.minimum_distance = ti.field(dtype=self.real_type, shape=())
        self.friction_dt = ti.field(dtype=self.real_type, shape=())
        self.friction_count = 0
        self.friction_capacity = self.max_point_triangle_pairs if self.dimension == 3 else self.max_point_edge_pairs
        self.friction_candidate = ti.Vector.field(self.stencil_size, dtype=ti.i32, shape=self.friction_capacity)
        self.friction_weight = ti.Vector.field(
            self.stencil_size,
            dtype=self.real_type,
            shape=self.friction_capacity,
        )
        self.friction_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.friction_capacity)
        self.friction_normal_force = ti.field(dtype=self.real_type, shape=self.friction_capacity)
        self.semi_capacity = 1 << (max(2 * self.friction_capacity, 1) - 1).bit_length()
        self.semi_count = ti.field(dtype=ti.i32, shape=())
        self.semi_state = ti.field(dtype=ti.i32, shape=self.semi_capacity)
        self.semi_key = ti.Vector.field(4, dtype=ti.i32, shape=self.semi_capacity)
        self.semi_weight = ti.Vector.field(4, dtype=self.real_type, shape=self.semi_capacity)
        self.semi_normal = ti.Vector.field(3, dtype=self.real_type, shape=self.semi_capacity)
        self.semi_multiplier = ti.field(dtype=self.real_type, shape=self.semi_capacity)
        self.semi_slot = ti.field(dtype=ti.i32, shape=self.friction_capacity)
        self.semi_overflow = ti.field(dtype=ti.i32, shape=())
        self.constraint_violation = ti.field(dtype=self.real_type, shape=())
        self.semi_state.fill(2)
        self.build_positions(fem.state.position, mpm.grid_disp)
        self.copy_friction_hat()

    @ti.func
    def _pair_indices(self, stencil):
        sample = stencil[0]
        particle = self.mpm.surface_id[sample]
        mpm_body = self.mpm.particle[particle].bodyID
        fem_node = stencil[1] - ti.static(self.surface_count)
        fem_body = self.fem_node_body[fem_node]
        return sample, particle, mpm_body, fem_body

    def candidate_field(self):
        return self.culling.point_edge if self.dimension == 2 else self.culling.point_triangle

    @ti.func
    def _pair_is_active(self, mpm_body, fem_body):
        return (
            0 <= mpm_body
            and mpm_body < ti.static(self.mpm_body_count)
            and 0 <= fem_body
            and fem_body < ti.static(self.fem_body_count)
            and self.pair_active[mpm_body, fem_body] != 0
        )

    @ti.kernel
    def build_positions(self, fem_position: ti.template(), mpm_displacement: ti.template()):
        for sample in range(self.surface_count):
            particle = self.mpm.surface_id[sample]
            value = ti.Vector.zero(self.real_type, 3)
            for component in ti.static(range(self.dimension)):
                value[component] = self.mpm.particle[particle].x[component]
            for local in range(self.mpm.offset[particle]):
                grid = self.mpm.LnID[particle, local]
                block = self.mpm.node2dof[grid] - 1
                if block >= 0:
                    for component in ti.static(range(self.dimension)):
                        value[component] += (
                            self.mpm.shape[particle, local] * mpm_displacement[self.dimension * block + component]
                        )
            self.positions[sample] = value
        for node in range(self.fem_node_count):
            self.positions[self.surface_count + node] = fem_position[node]

    @ti.kernel
    def build_end_positions(
        self,
        fem_position: ti.template(),
        fem_direction: ti.template(),
        mpm_displacement: ti.template(),
        mpm_direction: ti.template(),
    ):
        for sample in range(self.surface_count):
            particle = self.mpm.surface_id[sample]
            value = ti.Vector.zero(self.real_type, 3)
            for component in ti.static(range(self.dimension)):
                value[component] = self.mpm.particle[particle].x[component]
            for local in range(self.mpm.offset[particle]):
                grid = self.mpm.LnID[particle, local]
                block = self.mpm.node2dof[grid] - 1
                if block >= 0:
                    for component in ti.static(range(self.dimension)):
                        value[component] += self.mpm.shape[particle, local] * (
                            mpm_displacement[self.dimension * block + component]
                            + mpm_direction[self.dimension * block + component]
                        )
            self.end_positions[sample] = value
        for node in range(self.fem_node_count):
            self.end_positions[self.surface_count + node] = fem_position[node] + fem_direction[node]

    def prepare(self, fem_position, mpm_displacement):
        self.build_positions(fem_position, mpm_displacement)
        count, _ = self.culling.rebuild_proximity(self.positions, self.search_distance)
        count = int(count)
        if self.is_semi:
            self.semi_overflow[None] = 0
            if self.dimension == 2:
                self._prepare_semi_2d(count, self.candidate_field())
            else:
                self._prepare_semi_3d(count, self.candidate_field())
            if int(self.semi_overflow[None]) != 0:
                raise RuntimeError("FEMPM SemiIPC multiplier hash capacity is too small")
        return count

    @ti.func
    def _semi_frame(self, gradient, distance, dimensions: ti.template()):
        normal = ti.Vector([1.0, 0.0, 0.0])
        weight = ti.Vector.zero(self.real_type, 4)
        if distance > 1.0e-15:
            for component in ti.static(range(dimensions)):
                normal[component] = gradient[component] / (2.0 * distance)
            for site, component in ti.static(ti.ndrange(self.stencil_size, dimensions)):
                weight[site] += gradient[dimensions * site + component] * normal[component] / (2.0 * distance)
        return weight, normal

    @ti.func
    def _store_semi_frame(self, contact, key, weight, normal):
        slot = semi_ipc_find_or_insert(
            self.semi_state,
            self.semi_key,
            self.semi_multiplier,
            self.semi_count,
            key,
            ti.static(self.semi_capacity),
        )
        if slot >= 0:
            previous = self.semi_normal[slot]
            if previous.norm_sqr() > 0.0 and previous.dot(normal) < 0.0:
                normal = -normal
            self.semi_weight[slot] = weight
            self.semi_normal[slot] = normal
        else:
            self.semi_overflow[None] = 1
        self.semi_slot[contact] = slot

    @ti.kernel
    def _prepare_semi_3d(self, count: ti.i32, candidates: ti.template()):
        for contact in range(count):
            stencil = candidates[contact]
            distance2, gradient, _ = point_triangle_distance_grad(
                self.positions[stencil[0]],
                self.positions[stencil[1]],
                self.positions[stencil[2]],
                self.positions[stencil[3]],
            )
            weight, normal = self._semi_frame(gradient, ti.sqrt(ti.max(distance2, 0.0)), 3)
            self._store_semi_frame(contact, stencil, weight, normal)

    @ti.kernel
    def _prepare_semi_2d(self, count: ti.i32, candidates: ti.template()):
        for contact in range(count):
            stencil = candidates[contact]
            point = ti.Vector([self.positions[stencil[0]][0], self.positions[stencil[0]][1]])
            first = ti.Vector([self.positions[stencil[1]][0], self.positions[stencil[1]][1]])
            second = ti.Vector([self.positions[stencil[2]][0], self.positions[stencil[2]][1]])
            distance2, gradient, _, _ = point_edge_distance_grad_hess_2d(point, first, second)
            weight, normal = self._semi_frame(gradient, ti.sqrt(ti.max(distance2, 0.0)), 2)
            self._store_semi_frame(
                contact,
                ti.Vector([stencil[0], stencil[1], stencil[2], -1]),
                weight,
                normal,
            )

    @ti.kernel
    def copy_friction_hat(self):
        for node in range(self.combined_node_count):
            self.friction_hat[node] = self.positions[node]

    def _ensure_friction_capacity(self, required):
        if required > self.friction_capacity:
            raise RuntimeError(
                "FEMPM friction contact capacity is too small: " f"need {required}, allocated {self.friction_capacity}"
            )
        # Fixed construction-time allocation: never create Taichi fields from
        # a Newton/search-dependent active count.

    def begin_step(self, fem_position, mpm_displacement, timestep):
        if self.is_semi:
            self.semi_state.fill(2)
            self.semi_multiplier.fill(0.0)
            self.semi_count[None] = 0
        count = self.prepare(fem_position, mpm_displacement)
        self.copy_friction_hat()
        self.friction_dt[None] = float(timestep)
        if not self.activate_friction:
            self.friction_count = 0
            return 0
        self._ensure_friction_capacity(count)
        self._freeze_friction(
            count,
            self.candidate_field(),
            self.friction_candidate,
            self.friction_weight,
            self.friction_normal,
            self.friction_normal_force,
        )
        self.friction_count = count
        return count

    def refresh_friction(self, fem_position, mpm_displacement):
        """Refresh lagged frames/normal forces while preserving the step hat."""
        count = self.prepare(fem_position, mpm_displacement)
        if not self.activate_friction:
            self.friction_count = 0
            return 0
        self._ensure_friction_capacity(count)
        self._freeze_friction(
            count,
            self.candidate_field(),
            self.friction_candidate,
            self.friction_weight,
            self.friction_normal,
            self.friction_normal_force,
        )
        self.friction_count = count
        return count

    def _freeze_friction(self, *arguments):
        if self.dimension == 2:
            return self._freeze_friction_2d(*arguments)
        return self._freeze_friction_3d(*arguments)

    @ti.kernel
    def _freeze_friction_3d(
        self,
        count: ti.i32,
        candidates: ti.template(),
        frozen_candidates: ti.template(),
        weights: ti.template(),
        normals: ti.template(),
        normal_forces: ti.template(),
    ):
        for contact_id in range(count):
            stencil = candidates[contact_id]
            frozen_candidates[contact_id] = stencil
            sample, _, mpm_body, fem_body = self._pair_indices(stencil)
            weight = ti.Vector([1.0, 0.0, 0.0, 0.0])
            normal = ti.Vector([1.0, 0.0, 0.0])
            normal_force = 0.0
            if self._pair_is_active(mpm_body, fem_body):
                distance2, gradient, _ = point_triangle_distance_grad(
                    self.positions[stencil[0]],
                    self.positions[stencil[1]],
                    self.positions[stencil[2]],
                    self.positions[stencil[3]],
                )
                distance = ti.sqrt(ti.max(distance2, 0.0))
                if distance > 1.0e-15:
                    relative = ti.Vector([gradient[c] for c in ti.static(range(3))])
                    normal = relative / (2.0 * distance)
                    for site in ti.static(range(4)):
                        site_gradient = ti.Vector([gradient[3 * site + c] for c in ti.static(range(3))])
                        weight[site] = site_gradient.dot(normal) / (2.0 * distance)
                dmin = self.pair_dmin[mpm_body, fem_body]
                dhat = self.pair_dhat[mpm_body, fem_body]
                shifted = distance2 - dmin * dmin
                active_gap2 = (2.0 * dmin + dhat) * dhat
                if ti.static(self.is_semi):
                    slot = self.semi_slot[contact_id]
                    if slot >= 0:
                        weight = self.semi_weight[slot]
                        normal = self.semi_normal[slot]
                        gap = -(dmin + dhat)
                        for site in ti.static(range(4)):
                            gap += weight[site] * normal.dot(self.positions[stencil[site]])
                        normal_force = self.mpm.surface_measure[sample] * ti.max(
                            self.semi_multiplier[slot] - self.pair_penalty[mpm_body, fem_body] * gap,
                            0.0,
                        )
                elif shifted > 0.0 and shifted < active_gap2:
                    normalized_kappa = self.pair_kappa[mpm_body, fem_body] / (active_gap2 * active_gap2)
                    _, first, _ = ipc_toolkit_barrier_distance2_offset_terms(distance2, dhat, dmin, normalized_kappa, 0)
                    scale = self.mpm.surface_measure[sample] * dhat
                    normal_force = ti.max(0.0, -scale * first * 2.0 * ti.sqrt(shifted))
            weights[contact_id] = weight
            normals[contact_id] = normal
            normal_forces[contact_id] = normal_force

    @ti.kernel
    def _freeze_friction_2d(
        self,
        count: ti.i32,
        candidates: ti.template(),
        frozen_candidates: ti.template(),
        weights: ti.template(),
        normals: ti.template(),
        normal_forces: ti.template(),
    ):
        for contact_id in range(count):
            stencil = candidates[contact_id]
            frozen_candidates[contact_id] = stencil
            sample, _, mpm_body, fem_body = self._pair_indices(stencil)
            weight = ti.Vector([1.0, 0.0, 0.0])
            normal3 = ti.Vector([1.0, 0.0, 0.0])
            normal_force = 0.0
            if self._pair_is_active(mpm_body, fem_body):
                distance2, gradient, _, _ = point_edge_distance_grad_hess_2d(
                    ti.Vector(
                        [
                            self.positions[stencil[0]][0],
                            self.positions[stencil[0]][1],
                        ]
                    ),
                    ti.Vector(
                        [
                            self.positions[stencil[1]][0],
                            self.positions[stencil[1]][1],
                        ]
                    ),
                    ti.Vector(
                        [
                            self.positions[stencil[2]][0],
                            self.positions[stencil[2]][1],
                        ]
                    ),
                )
                distance = ti.sqrt(ti.max(distance2, 0.0))
                if distance > 1.0e-15:
                    normal = ti.Vector([gradient[0], gradient[1]]) / (2.0 * distance)
                    normal3[0] = normal[0]
                    normal3[1] = normal[1]
                    for site in ti.static(range(3)):
                        site_gradient = ti.Vector([gradient[2 * site], gradient[2 * site + 1]])
                        weight[site] = site_gradient.dot(normal) / (2.0 * distance)
                dmin = self.pair_dmin[mpm_body, fem_body]
                dhat = self.pair_dhat[mpm_body, fem_body]
                shifted = distance2 - dmin * dmin
                active_gap2 = (2.0 * dmin + dhat) * dhat
                if ti.static(self.is_semi):
                    slot = self.semi_slot[contact_id]
                    if slot >= 0:
                        hash_weight = self.semi_weight[slot]
                        normal3 = self.semi_normal[slot]
                        for site in ti.static(range(3)):
                            weight[site] = hash_weight[site]
                        gap = -(dmin + dhat)
                        for site in ti.static(range(3)):
                            gap += weight[site] * normal3.dot(self.positions[stencil[site]])
                        normal_force = self.mpm.surface_measure[sample] * ti.max(
                            self.semi_multiplier[slot] - self.pair_penalty[mpm_body, fem_body] * gap,
                            0.0,
                        )
                elif shifted > 0.0 and shifted < active_gap2:
                    normalized_kappa = self.pair_kappa[mpm_body, fem_body] / (active_gap2 * active_gap2)
                    _, first, _ = ipc_toolkit_barrier_distance2_offset_terms(distance2, dhat, dmin, normalized_kappa, 0)
                    scale = self.mpm.surface_measure[sample] * dhat
                    normal_force = ti.max(0.0, -scale * first * 2.0 * ti.sqrt(shifted))
            weights[contact_id] = weight
            normals[contact_id] = normal3
            normal_forces[contact_id] = normal_force

    @ti.func
    def _scatter_gradient(self, stencil, particle, gradient, rhs):
        for site in ti.static(range(1, self.stencil_size)):
            fem_node = stencil[site] - ti.static(self.surface_count)
            for component in ti.static(range(self.dimension)):
                ti.atomic_add(
                    rhs[3 * fem_node + component],
                    -gradient[self.dimension * site + component],
                )
        point_gradient = ti.Vector([gradient[component] for component in ti.static(range(self.dimension))])
        for local in range(self.mpm.offset[particle]):
            grid = self.mpm.LnID[particle, local]
            block = self.mpm.node2dof[grid] - 1
            if block >= 0:
                coupled_block = ti.static(self.fem_node_count) + block
                shape = self.mpm.shape[particle, local]
                for component in ti.static(range(self.dimension)):
                    ti.atomic_add(
                        rhs[3 * coupled_block + component],
                        -shape * point_gradient[component],
                    )

    @ti.func
    def _scatter_hessian_block(
        self,
        stencil,
        particle,
        first_site,
        second_site,
        hessian,
        matrix,
    ):
        if first_site > 0 and second_site > 0:
            matrix.add_block_entry(
                stencil[first_site] - ti.static(self.surface_count),
                stencil[second_site] - ti.static(self.surface_count),
                hessian,
            )
        elif first_site == 0 and second_site > 0:
            fem_block = stencil[second_site] - ti.static(self.surface_count)
            for local in range(self.mpm.offset[particle]):
                grid = self.mpm.LnID[particle, local]
                block = self.mpm.node2dof[grid] - 1
                if block >= 0:
                    matrix.add_block_entry(
                        ti.static(self.fem_node_count) + block,
                        fem_block,
                        self.mpm.shape[particle, local] * hessian,
                    )
        elif first_site > 0 and second_site == 0:
            fem_block = stencil[first_site] - ti.static(self.surface_count)
            for local in range(self.mpm.offset[particle]):
                grid = self.mpm.LnID[particle, local]
                block = self.mpm.node2dof[grid] - 1
                if block >= 0:
                    matrix.add_block_entry(
                        fem_block,
                        ti.static(self.fem_node_count) + block,
                        self.mpm.shape[particle, local] * hessian,
                    )
        else:
            for local in range(self.mpm.offset[particle]):
                grid = self.mpm.LnID[particle, local]
                block = self.mpm.node2dof[grid] - 1
                if block >= 0:
                    coupled = ti.static(self.fem_node_count) + block
                    shape = self.mpm.shape[particle, local]
                    for other in range(self.mpm.offset[particle]):
                        other_grid = self.mpm.LnID[particle, other]
                        other_block = self.mpm.node2dof[other_grid] - 1
                        if other_block >= 0:
                            matrix.add_block_entry(
                                coupled,
                                ti.static(self.fem_node_count) + other_block,
                                shape * self.mpm.shape[particle, other] * hessian,
                            )

    def assemble(self, *arguments, project_pd=True):
        if self.dimension == 2:
            return self.assemble_2d(*arguments, bool(project_pd))
        return self.assemble_3d(*arguments, bool(project_pd))

    @ti.kernel
    def assemble_3d(
        self,
        count: ti.i32,
        candidates: ti.template(),
        matrix: ti.template(),
        rhs: ti.template(),
        need_matrix: ti.template(),
        include_friction: ti.template(),
        friction_count: ti.i32,
        project_pd: ti.template(),
    ):
        self.total_energy[None] = 0.0
        self.active_count[None] = 0
        self.minimum_distance[None] = 1.0e30
        self.constraint_violation[None] = 0.0
        for contact_id in range(count):
            stencil = candidates[contact_id]
            sample, particle, mpm_body, fem_body = self._pair_indices(stencil)
            if self._pair_is_active(mpm_body, fem_body):
                distance2 = 0.0
                distance_gradient = ti.Vector.zero(self.real_type, 12)
                distance_hessian = ti.Matrix.zero(self.real_type, 12, 12)
                if ti.static(need_matrix):
                    distance2, distance_gradient, distance_hessian, _ = point_triangle_distance_grad_hess(
                        self.positions[stencil[0]],
                        self.positions[stencil[1]],
                        self.positions[stencil[2]],
                        self.positions[stencil[3]],
                    )
                else:
                    distance2, distance_gradient, _ = point_triangle_distance_grad(
                        self.positions[stencil[0]],
                        self.positions[stencil[1]],
                        self.positions[stencil[2]],
                        self.positions[stencil[3]],
                    )
                dmin = self.pair_dmin[mpm_body, fem_body]
                dhat = self.pair_dhat[mpm_body, fem_body]
                distance = ti.sqrt(ti.max(distance2, 0.0))
                ti.atomic_min(self.minimum_distance[None], distance - dmin)
                shifted = distance2 - dmin * dmin
                active_gap2 = (2.0 * dmin + dhat) * dhat
                if ti.static(self.is_semi):
                    slot = self.semi_slot[contact_id]
                    if slot >= 0:
                        weight = self.semi_weight[slot]
                        normal = self.semi_normal[slot]
                        gap = -(dmin + dhat)
                        for site in ti.static(range(4)):
                            gap += weight[site] * normal.dot(self.positions[stencil[site]])
                        ti.atomic_max(self.constraint_violation[None], ti.max(-gap, 0.0))
                        value, first, second = semi_ipc_terms(
                            gap,
                            self.semi_multiplier[slot],
                            self.pair_penalty[mpm_body, fem_body],
                        )
                        scale = self.mpm.surface_measure[sample]
                        ti.atomic_add(self.total_energy[None], scale * value)
                        if second > 0.0:
                            ti.atomic_add(self.active_count[None], 1)
                            gradient = ti.Vector.zero(self.real_type, 12)
                            for site, component in ti.static(ti.ndrange(4, 3)):
                                gradient[3 * site + component] = first * weight[site] * normal[component]
                            self._scatter_gradient(stencil, particle, scale * gradient, rhs)
                            if ti.static(need_matrix):
                                for first_site, second_site in ti.ndrange(4, 4):
                                    block = (
                                        scale
                                        * second
                                        * weight[first_site]
                                        * weight[second_site]
                                        * normal.outer_product(normal)
                                    )
                                    self._scatter_hessian_block(
                                        stencil, particle, first_site, second_site, block, matrix
                                    )
                elif shifted < active_gap2:
                    normalized_kappa = self.pair_kappa[mpm_body, fem_body] / (active_gap2 * active_gap2)
                    value, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                        distance2, dhat, dmin, normalized_kappa, 0
                    )
                    scale = self.mpm.surface_measure[sample] * dhat
                    gradient = scale * first * distance_gradient
                    ti.atomic_add(self.total_energy[None], scale * value)
                    ti.atomic_add(self.active_count[None], 1)
                    self._scatter_gradient(stencil, particle, gradient, rhs)
                    if ti.static(need_matrix):
                        if shifted > 0.0:
                            local_hessian = ti.Matrix.zero(self.real_type, 12, 12)
                            for row, column in ti.ndrange(12, 12):
                                local_hessian[row, column] = scale * (
                                    first * distance_hessian[row, column]
                                    + second * distance_gradient[row] * distance_gradient[column]
                                )
                            if ti.static(project_pd):
                                local_hessian = psd_project_nd(local_hessian)
                            for first_site, second_site in ti.ndrange(4, 4):
                                block = ti.Matrix.zero(self.real_type, 3, 3)
                                for row, column in ti.static(ti.ndrange(3, 3)):
                                    block[row, column] = local_hessian[
                                        3 * first_site + row,
                                        3 * second_site + column,
                                    ]
                                self._scatter_hessian_block(
                                    stencil,
                                    particle,
                                    first_site,
                                    second_site,
                                    block,
                                    matrix,
                                )

        if ti.static(include_friction):
            identity = ti.Matrix.identity(self.real_type, 3)
            timestep = self.friction_dt[None]
            for contact_id in range(friction_count):
                stencil = self.friction_candidate[contact_id]
                _, particle, mpm_body, fem_body = self._pair_indices(stencil)
                coefficient = self.pair_friction[mpm_body, fem_body] * self.friction_normal_force[contact_id]
                if coefficient > 0.0:
                    normal = self.friction_normal[contact_id]
                    tangent = identity - normal.outer_product(normal)
                    relative_increment = ti.Vector.zero(self.real_type, 3)
                    for site in ti.static(range(4)):
                        relative_increment += self.friction_weight[contact_id][site] * (
                            self.positions[stencil[site]] - self.friction_hat[stencil[site]]
                        )
                    velocity = tangent @ relative_increment / timestep
                    speed = velocity.norm()
                    epsv = self.pair_epsv[mpm_body, fem_body]
                    ti.atomic_add(
                        self.total_energy[None],
                        coefficient * ipc_friction_f0(speed, epsv, timestep),
                    )
                    profile = ipc_friction_f1_over_speed(speed, epsv)
                    relative_gradient = coefficient * profile * (tangent @ velocity)
                    gradient = ti.Vector.zero(self.real_type, 12)
                    for site, component in ti.static(ti.ndrange(4, 3)):
                        gradient[3 * site + component] = (
                            self.friction_weight[contact_id][site] * relative_gradient[component]
                        )
                    self._scatter_gradient(stencil, particle, gradient, rhs)
                    if ti.static(need_matrix):
                        inner = coefficient * profile * identity
                        if speed > 0.0:
                            inner += (
                                coefficient
                                * ipc_friction_hessian_term(speed, epsv)
                                / speed
                                * velocity.outer_product(velocity)
                            )
                        relative_hessian = tangent @ inner @ tangent / timestep
                        for first, second in ti.ndrange(4, 4):
                            block = ti.Matrix.zero(self.real_type, 3, 3)
                            for row, column in ti.static(ti.ndrange(3, 3)):
                                block[row, column] = (
                                    self.friction_weight[contact_id][first]
                                    * self.friction_weight[contact_id][second]
                                    * relative_hessian[row, column]
                                )
                            self._scatter_hessian_block(
                                stencil,
                                particle,
                                first,
                                second,
                                block,
                                matrix,
                            )

    @ti.kernel
    def assemble_2d(
        self,
        count: ti.i32,
        candidates: ti.template(),
        matrix: ti.template(),
        rhs: ti.template(),
        need_matrix: ti.template(),
        include_friction: ti.template(),
        friction_count: ti.i32,
        project_pd: ti.template(),
    ):
        self.total_energy[None] = 0.0
        self.active_count[None] = 0
        self.minimum_distance[None] = 1.0e30
        self.constraint_violation[None] = 0.0
        for contact_id in range(count):
            stencil = candidates[contact_id]
            sample, particle, mpm_body, fem_body = self._pair_indices(stencil)
            if self._pair_is_active(mpm_body, fem_body):
                point = ti.Vector(
                    [
                        self.positions[stencil[0]][0],
                        self.positions[stencil[0]][1],
                    ]
                )
                endpoint0 = ti.Vector(
                    [
                        self.positions[stencil[1]][0],
                        self.positions[stencil[1]][1],
                    ]
                )
                endpoint1 = ti.Vector(
                    [
                        self.positions[stencil[2]][0],
                        self.positions[stencil[2]][1],
                    ]
                )
                distance2, distance_gradient, distance_hessian, _ = point_edge_distance_grad_hess_2d(
                    point, endpoint0, endpoint1
                )
                dmin = self.pair_dmin[mpm_body, fem_body]
                dhat = self.pair_dhat[mpm_body, fem_body]
                distance = ti.sqrt(ti.max(distance2, 0.0))
                ti.atomic_min(self.minimum_distance[None], distance - dmin)
                shifted = distance2 - dmin * dmin
                active_gap2 = (2.0 * dmin + dhat) * dhat
                if ti.static(self.is_semi):
                    slot = self.semi_slot[contact_id]
                    if slot >= 0:
                        hash_weight = self.semi_weight[slot]
                        normal3 = self.semi_normal[slot]
                        normal = ti.Vector([normal3[0], normal3[1]])
                        gap = -(dmin + dhat)
                        for site in ti.static(range(3)):
                            gap += hash_weight[site] * normal3.dot(self.positions[stencil[site]])
                        ti.atomic_max(self.constraint_violation[None], ti.max(-gap, 0.0))
                        value, first, second = semi_ipc_terms(
                            gap,
                            self.semi_multiplier[slot],
                            self.pair_penalty[mpm_body, fem_body],
                        )
                        scale = self.mpm.surface_measure[sample]
                        ti.atomic_add(self.total_energy[None], scale * value)
                        if second > 0.0:
                            ti.atomic_add(self.active_count[None], 1)
                            gradient = ti.Vector.zero(self.real_type, 6)
                            for site, component in ti.static(ti.ndrange(3, 2)):
                                gradient[2 * site + component] = first * hash_weight[site] * normal[component]
                            self._scatter_gradient(stencil, particle, scale * gradient, rhs)
                            if ti.static(need_matrix):
                                for first_site, second_site in ti.ndrange(3, 3):
                                    block = ti.Matrix.zero(self.real_type, 3, 3)
                                    for row, column in ti.static(ti.ndrange(2, 2)):
                                        block[row, column] = (
                                            scale
                                            * second
                                            * hash_weight[first_site]
                                            * hash_weight[second_site]
                                            * normal[row]
                                            * normal[column]
                                        )
                                    self._scatter_hessian_block(
                                        stencil, particle, first_site, second_site, block, matrix
                                    )
                elif shifted < active_gap2:
                    normalized_kappa = self.pair_kappa[mpm_body, fem_body] / (active_gap2 * active_gap2)
                    value, first, second = ipc_toolkit_barrier_distance2_offset_terms(
                        distance2, dhat, dmin, normalized_kappa, 0
                    )
                    scale = self.mpm.surface_measure[sample] * dhat
                    gradient = scale * first * distance_gradient
                    ti.atomic_add(self.total_energy[None], scale * value)
                    ti.atomic_add(self.active_count[None], 1)
                    self._scatter_gradient(stencil, particle, gradient, rhs)
                    if ti.static(need_matrix):
                        if shifted > 0.0:
                            local_hessian = ti.Matrix.zero(self.real_type, 6, 6)
                            for row, column in ti.ndrange(6, 6):
                                local_hessian[row, column] = scale * (
                                    first * distance_hessian[row, column]
                                    + second * distance_gradient[row] * distance_gradient[column]
                                )
                            if ti.static(project_pd):
                                local_hessian = psd_project_nd(local_hessian)
                            for first_site, second_site in ti.ndrange(3, 3):
                                block = ti.Matrix.zero(self.real_type, 3, 3)
                                for row, column in ti.static(ti.ndrange(2, 2)):
                                    block[row, column] = local_hessian[
                                        2 * first_site + row,
                                        2 * second_site + column,
                                    ]
                                self._scatter_hessian_block(
                                    stencil,
                                    particle,
                                    first_site,
                                    second_site,
                                    block,
                                    matrix,
                                )

        if ti.static(include_friction):
            identity = ti.Matrix.identity(self.real_type, 2)
            timestep = self.friction_dt[None]
            for contact_id in range(friction_count):
                stencil = self.friction_candidate[contact_id]
                _, particle, mpm_body, fem_body = self._pair_indices(stencil)
                coefficient = self.pair_friction[mpm_body, fem_body] * self.friction_normal_force[contact_id]
                if coefficient > 0.0:
                    normal3 = self.friction_normal[contact_id]
                    normal = ti.Vector([normal3[0], normal3[1]])
                    tangent = identity - normal.outer_product(normal)
                    relative_increment = ti.Vector.zero(self.real_type, 2)
                    for site in ti.static(range(3)):
                        delta3 = self.positions[stencil[site]] - self.friction_hat[stencil[site]]
                        relative_increment += self.friction_weight[contact_id][site] * ti.Vector([delta3[0], delta3[1]])
                    velocity = tangent @ relative_increment / timestep
                    speed = velocity.norm()
                    epsv = self.pair_epsv[mpm_body, fem_body]
                    ti.atomic_add(
                        self.total_energy[None],
                        coefficient * ipc_friction_f0(speed, epsv, timestep),
                    )
                    profile = ipc_friction_f1_over_speed(speed, epsv)
                    relative_gradient = coefficient * profile * (tangent @ velocity)
                    gradient = ti.Vector.zero(self.real_type, 6)
                    for site, component in ti.static(ti.ndrange(3, 2)):
                        gradient[2 * site + component] = (
                            self.friction_weight[contact_id][site] * relative_gradient[component]
                        )
                    self._scatter_gradient(stencil, particle, gradient, rhs)
                    if ti.static(need_matrix):
                        inner = coefficient * profile * identity
                        if speed > 0.0:
                            inner += (
                                coefficient
                                * ipc_friction_hessian_term(speed, epsv)
                                / speed
                                * velocity.outer_product(velocity)
                            )
                        relative_hessian = tangent @ inner @ tangent / timestep
                        for first, second in ti.ndrange(3, 3):
                            block = ti.Matrix.zero(self.real_type, 3, 3)
                            for row, column in ti.static(ti.ndrange(2, 2)):
                                block[row, column] = (
                                    self.friction_weight[contact_id][first]
                                    * self.friction_weight[contact_id][second]
                                    * relative_hessian[row, column]
                                )
                            self._scatter_hessian_block(
                                stencil,
                                particle,
                                first,
                                second,
                                block,
                                matrix,
                            )

    @ti.kernel
    def _compute_ccd_3d(
        self,
        raw_count: ti.i32,
        candidates: ti.template(),
    ):
        self.minimum_step[None] = 1.0
        for contact_id in range(raw_count):
            stencil = candidates[contact_id]
            _, _, mpm_body, fem_body = self._pair_indices(stencil)
            if self._pair_is_active(mpm_body, fem_body):
                eta = 1.0 - ti.static(self.ccd_safety)
                thickness = self.pair_dmin[mpm_body, fem_body]
                if ti.static(self.is_semi):
                    thickness = 0.0
                toc = 1.0
                if thickness > 0.0:
                    toc = point_triangle_accd(
                        self.positions[stencil[0]],
                        self.positions[stencil[1]],
                        self.positions[stencil[2]],
                        self.positions[stencil[3]],
                        self.end_positions[stencil[0]] - self.positions[stencil[0]],
                        self.end_positions[stencil[1]] - self.positions[stencil[1]],
                        self.end_positions[stencil[2]] - self.positions[stencil[2]],
                        self.end_positions[stencil[3]] - self.positions[stencil[3]],
                        eta,
                        thickness,
                        ti.static(self.ccd_max_iterations),
                    )
                else:
                    toc = point_triangle_ccd(
                        self.positions[stencil[0]],
                        self.positions[stencil[1]],
                        self.positions[stencil[2]],
                        self.positions[stencil[3]],
                        self.end_positions[stencil[0]] - self.positions[stencil[0]],
                        self.end_positions[stencil[1]] - self.positions[stencil[1]],
                        self.end_positions[stencil[2]] - self.positions[stencil[2]],
                        self.end_positions[stencil[3]] - self.positions[stencil[3]],
                        eta,
                        ti.static(self.ccd_max_iterations),
                    )
                ti.atomic_min(self.minimum_step[None], toc)

    @ti.kernel
    def _compute_ccd_2d(
        self,
        raw_count: ti.i32,
        candidates: ti.template(),
    ):
        self.minimum_step[None] = 1.0
        for contact_id in range(raw_count):
            stencil = candidates[contact_id]
            _, _, mpm_body, fem_body = self._pair_indices(stencil)
            if self._pair_is_active(mpm_body, fem_body):
                eta = 1.0 - ti.static(self.ccd_safety)
                thickness = self.pair_dmin[mpm_body, fem_body]
                if ti.static(self.is_semi):
                    thickness = 0.0
                point = ti.Vector(
                    [
                        self.positions[stencil[0]][0],
                        self.positions[stencil[0]][1],
                    ]
                )
                endpoint0 = ti.Vector(
                    [
                        self.positions[stencil[1]][0],
                        self.positions[stencil[1]][1],
                    ]
                )
                endpoint1 = ti.Vector(
                    [
                        self.positions[stencil[2]][0],
                        self.positions[stencil[2]][1],
                    ]
                )
                point_displacement = ti.Vector(
                    [
                        self.end_positions[stencil[0]][0] - point[0],
                        self.end_positions[stencil[0]][1] - point[1],
                    ]
                )
                endpoint0_displacement = ti.Vector(
                    [
                        self.end_positions[stencil[1]][0] - endpoint0[0],
                        self.end_positions[stencil[1]][1] - endpoint0[1],
                    ]
                )
                endpoint1_displacement = ti.Vector(
                    [
                        self.end_positions[stencil[2]][0] - endpoint1[0],
                        self.end_positions[stencil[2]][1] - endpoint1[1],
                    ]
                )
                toc = 1.0
                if thickness > 0.0:
                    toc = point_edge_accd(
                        point,
                        endpoint0,
                        endpoint1,
                        point_displacement,
                        endpoint0_displacement,
                        endpoint1_displacement,
                        eta,
                        thickness,
                        ti.static(self.ccd_max_iterations),
                    )
                else:
                    toc = point_edge_ccd(
                        point,
                        endpoint0,
                        endpoint1,
                        point_displacement,
                        endpoint0_displacement,
                        endpoint1_displacement,
                        eta,
                        ti.static(self.ccd_max_iterations),
                    )
                ti.atomic_min(self.minimum_step[None], toc)

    def maximum_step(
        self,
        fem_position,
        fem_direction,
        mpm_displacement,
        mpm_direction,
    ):
        self.build_positions(fem_position, mpm_displacement)
        self.build_end_positions(
            fem_position,
            fem_direction,
            mpm_displacement,
            mpm_direction,
        )
        raw_count, _ = self.culling.rebuild_swept_candidates(
            self.positions,
            self.end_positions,
            self.search_distance,
        )
        if self.dimension == 2:
            self._compute_ccd_2d(raw_count, self.candidate_field())
        else:
            self._compute_ccd_3d(raw_count, self.candidate_field())
        return max(0.0, min(1.0, float(self.minimum_step[None])))

    @ti.kernel
    def _update_semi_multipliers(self, count: ti.i32, candidates: ti.template()):
        self.constraint_violation[None] = 0.0
        for contact in range(count):
            stencil = candidates[contact]
            _, _, mpm_body, fem_body = self._pair_indices(stencil)
            slot = self.semi_slot[contact]
            if slot >= 0 and self._pair_is_active(mpm_body, fem_body):
                gap = -(self.pair_dmin[mpm_body, fem_body] + self.pair_dhat[mpm_body, fem_body])
                for site in ti.static(range(self.stencil_size)):
                    gap += self.semi_weight[slot][site] * self.semi_normal[slot].dot(self.positions[stencil[site]])
                self.semi_multiplier[slot] = semi_ipc_update_multiplier(
                    gap,
                    self.semi_multiplier[slot],
                    self.pair_penalty[mpm_body, fem_body],
                )
                ti.atomic_max(self.constraint_violation[None], ti.max(-gap, 0.0))

    def accept_update(self, fem_position, mpm_displacement):
        if not self.is_semi:
            return
        count = self.prepare(fem_position, mpm_displacement)
        self._update_semi_multipliers(count, self.candidate_field())

    def contact_converged(self):
        return not self.is_semi or float(self.constraint_violation[None]) <= self.constraint_tolerance

    def contact_block_capacity(self, candidate_count):
        support = int(self.mpm.shape_func.max_node_per_particle)
        contacts = int(candidate_count) + int(self.friction_count)
        return max(
            1,
            contacts * (support + self.fem_site_count) * (support + self.fem_site_count),
        )

    def configured_contact_block_capacity(self):
        candidates = self.max_point_triangle_pairs if self.dimension == 3 else self.max_point_edge_pairs
        contributions = 2 if self.activate_friction else 1
        support = int(self.mpm.shape_func.max_node_per_particle)
        stencil_blocks = support + self.fem_site_count
        return max(1, contributions * candidates * stencil_blocks**2)

    def diagnostics(self):
        minimum = float(self.minimum_distance[None])
        if minimum > 0.5e30:
            minimum = np.inf
        return {
            "model": "SemiIPC" if self.is_semi else "BarrierIPC",
            "active_contacts": int(self.active_count[None]),
            "candidate_contacts": int(
                self.culling.point_edge_count[None] if self.dimension == 2 else self.culling.point_triangle_count[None]
            ),
            "friction_contacts": int(self.friction_count),
            "minimum_distance": minimum,
            "ccd_step": float(self.minimum_step[None]),
            "energy": float(self.total_energy[None]),
            "constraint_violation": float(self.constraint_violation[None]) if self.is_semi else 0.0,
        }


__all__ = ["FEMPMIPCAssembler"]
