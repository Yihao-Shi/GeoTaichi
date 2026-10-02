"""Contact engine responsibilities."""

import math

import numpy as np
import taichi as ti

import src.igampm.config as config
from src.physics_model.contact_model.ipc.IPC import (
    semi_ipc_terms,
    semi_ipc_update_multiplier,
)
from src.physics_model.contact_model.ipc.NurbsContact import (
    curve_barrier_projected_metric,
    curve_control_reduced_jacobian,
    curve_point_reduced_jacobian,
    evaluate_distance_to_curve_fixed_dim,
    evaluate_distance_to_surface_fixed_dim,
    get_distance_to_curve_fixed_dim,
    get_distance_to_surface_fixed_dim,
    surface_barrier_projected_metric,
    surface_control_reduced_jacobian,
    surface_point_reduced_jacobian,
)


class ContactEngineMixin:
    @ti.func
    def _normal_terms(self, contact_id, distance):
        value = 0.0
        first = 0.0
        second = 0.0
        if ti.static(self.is_semi):
            value, first, second = semi_ipc_terms(
                distance - self.barrier.activation_distance_term(),
                self.semi_multiplier[contact_id],
                self.barrier.penalty[0],
            )
        else:
            value, first, second = self.barrier._terms(distance)
        return value, first, second

    @ti.func
    def _normal_gradient(self, contact_id, distance, ddistance2, measure):
        _, first, _ = self._normal_terms(contact_id, distance)
        return 0.5 * first / ti.max(distance, 1.0e-15) * ddistance2 * measure

    @ti.func
    def _normal_hessian(
        self,
        contact_id,
        distance,
        ddistance2_first,
        ddistance2_second,
        d2distance2,
        measure,
    ):
        _, first, second = self._normal_terms(contact_id, distance)
        inverse = 1.0 / ti.max(distance, 1.0e-15)
        hessian = 0.25 * second * inverse * inverse * ddistance2_first.outer_product(ddistance2_second)
        if ti.static(not self.is_semi):
            hessian += (
                0.5 * first * inverse * d2distance2
                - 0.25 * first * inverse * inverse * inverse * ddistance2_first.outer_product(ddistance2_second)
            )
        return measure * hessian

    @ti.func
    def _normal_energy(self, contact_id, distance, measure):
        value, _, _ = self._normal_terms(contact_id, distance)
        return measure * value

    @staticmethod
    def _is_paper_endpoint_scheme(integration):
        integration = np.asarray(integration, dtype=np.float64).reshape(-1)
        return integration.size >= 3 and (
            np.allclose(integration[:3], [1.0, 0.5, 1.0], rtol=0.0, atol=1.0e-14)
            or np.allclose(integration[:3], [0.5, 0.25, 0.5], rtol=0.0, atol=1.0e-14)
        )

    def _is_total_lagrangian_mpm(self):
        return self.total_lagrangian_mpm

    def _validate_active_mpm_dofs(self):
        active_dof = int(self.mpm.active_dof)
        capacity = int(self.mpm.degree_of_freedom)
        if active_dof < 0 or active_dof > capacity or active_dof % config.DIM != 0:
            raise RuntimeError(
                "invalid active MPM degree-of-freedom count: "
                f"active={active_dof}, capacity={capacity}, dim={config.DIM}"
            )
        return active_dof

    @ti.kernel
    def reset_contacts(self):
        self.contact_num[0] = 0
        for i in self.contacts:
            self.contacts[i].active = 0
            self.contacts[i].surface_id = -1
            self.contacts[i].sample_id = -1
            self.contacts[i].particle_id = -1
            self.contacts[i].distance = ti.math.inf
            self.contacts[i].knot_value = ti.Vector.zero(ti.f64, 2)

    @ti.kernel
    def load_contacts_from_buffer(self):
        self.contact_num[0] = 0
        for i in self.contacts:
            self.contacts[i].active = self.contact_active_buffer[i]
            self.contacts[i].surface_id = self.contact_surface_id_buffer[i]
            self.contacts[i].sample_id = self.contact_sample_id_buffer[i]
            self.contacts[i].particle_id = self.contact_particle_id_buffer[i]
            self.contacts[i].distance = self.contact_distance_buffer[i]
            self.contacts[i].knot_value = self.contact_knot_value_buffer[i]
            if self.contacts[i].active:
                self.contact_num[0] += 1

    @ti.kernel
    def count_contacts(self) -> ti.i32:
        total = 0
        for i in self.contacts:
            if self.contacts[i].active:
                total += 1
        return total

    @ti.kernel
    def update_particle_pos(self, grid_disp: ti.template()):
        for s in range(self.mpm.total_surface_num):
            p = self.mpm.surface_id[s]
            disp = ti.Vector.zero(ti.f64, config.DIM)
            for j in range(self.mpm.offset[p]):
                grid_id = self.mpm.LnID[p, j]
                dofs = config.DIM * (self.mpm.node2dof[grid_id] - 1)
                disp += self.mpm.shape[p, j] * ti.Vector([grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
            self.mpm.p_temp[s] = self.mpm.particle[p].x + disp

    @ti.kernel
    def find_closest_surface_contacts(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_knot_v: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        num_knot_v: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
    ):
        for s in range(self.mpm.total_surface_num):
            contact_id = s * self.contact_surface.num_surfaces + surface_id
            position = self.mpm.p_temp[s]
            uknot, vknot, distance, _ = get_distance_to_surface_fixed_dim(
                prefix_num_knot_u,
                prefix_num_knot_v,
                prefix_num_ctrlpts,
                num_knot_u,
                num_knot_v,
                surface.knot_vector_u,
                surface.knot_vector_v,
                surface.control_points_hat,
                surface.weights,
                position,
                basis,
            )
            # Keep the distance of every sample--surface pair, including
            # inactive pairs.  The conservative contact step needs the full
            # constraint set; the barrier active set remains distance < dhat.
            self.contacts[contact_id].surface_id = surface_id
            self.contacts[contact_id].sample_id = s
            self.contacts[contact_id].particle_id = self.mpm.surface_id[s]
            self.contacts[contact_id].distance = distance
            self.contacts[contact_id].knot_value = ti.Vector([uknot, vknot])
            if distance < self.barrier.activation_distance_term() or (
                ti.static(self.is_semi) and self.semi_multiplier[contact_id] > 0.0
            ):
                self.contacts[contact_id].active = 1

    @ti.kernel
    def find_closest_curve_contacts(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
    ):
        for s in range(self.mpm.total_surface_num):
            contact_id = s * self.contact_surface.num_surfaces + surface_id
            position = self.mpm.p_temp[s]
            uknot, distance, _ = get_distance_to_curve_fixed_dim(
                prefix_num_knot_u,
                prefix_num_ctrlpts,
                num_knot_u,
                surface.knot_vector_u,
                surface.control_points_hat,
                surface.weights,
                position,
                basis,
            )
            self.contacts[contact_id].surface_id = surface_id
            self.contacts[contact_id].sample_id = s
            self.contacts[contact_id].particle_id = self.mpm.surface_id[s]
            self.contacts[contact_id].distance = distance
            self.contacts[contact_id].knot_value = ti.Vector([uknot, 0.0])
            if distance < self.barrier.activation_distance_term() or (
                ti.static(self.is_semi) and self.semi_multiplier[contact_id] > 0.0
            ):
                self.contacts[contact_id].active = 1

    @ti.kernel
    def _measure_semi_constraint_violation(self, expected_pairs: ti.i32):
        self.semi_constraint_violation[None] = 0.0
        for contact_id in range(expected_pairs):
            gap = self.contacts[contact_id].distance - self.barrier.activation_distance_term()
            ti.atomic_max(self.semi_constraint_violation[None], ti.max(-gap, 0.0))

    @ti.kernel
    def _update_semi_multipliers(self, expected_pairs: ti.i32):
        self.semi_constraint_violation[None] = 0.0
        for contact_id in range(expected_pairs):
            gap = self.contacts[contact_id].distance - self.barrier.activation_distance_term()
            self.semi_multiplier[contact_id] = semi_ipc_update_multiplier(
                gap, self.semi_multiplier[contact_id], self.barrier.penalty[0]
            )
            ti.atomic_max(self.semi_constraint_violation[None], ti.max(-gap, 0.0))

    @ti.kernel
    def clear_barrier_system(self):
        self.barrier_nnz_count[0] = 0
        self.barrier_nnz_overflow[0] = 0
        for i in self.barrier_grad:
            self.barrier_grad[i] = 0.0

    @ti.kernel
    def prepare_barrier_matrix_slots(self):
        """Reserve and invalidate deterministic contact/local-pair slots."""
        required = ti.static(self.contact_pair_count * self.contact_pair_capacity)
        capacity = ti.static(self.barrier_hash_matrix.non_diag.blockI.shape[0])
        stored = ti.min(required, capacity)
        self.barrier_hash_matrix.raw_non_diag_count[0] = stored
        if required > capacity:
            self.barrier_hash_matrix.overflow[0] = 1
        for slot in range(stored):
            self.barrier_hash_matrix.non_diag.blockI[slot] = -1
            self.barrier_hash_matrix.non_diag.blockJ[slot] = -1
            for component in ti.static(range(config.DIM * config.DIM)):
                self.barrier_hash_matrix.non_diag.blockH[slot][component] = 0.0

    @ti.func
    def add_barrier_block(self, contact_id, local_i, local_j, block_i, block_j, block):
        ti.atomic_add(self.barrier_nnz_count[0], 1)
        stencil = ti.static(self.contact_stencil_capacity)
        slot = contact_id * ti.static(self.contact_pair_capacity) + local_i * stencil + local_j
        if (
            0 <= contact_id < ti.static(self.contact_pair_count)
            and 0 <= local_i < stencil
            and 0 <= local_j < stencil
            and 0 <= slot
            and slot < self.barrier_hash_matrix.non_diag.blockI.shape[0]
            and block_i >= 0
            and block_j >= 0
        ):
            if block_i == block_j:
                for row in ti.static(range(config.DIM)):
                    for column in ti.static(range(config.DIM)):
                        ti.atomic_add(
                            self.barrier_hash_matrix.diag[block_i][row * config.DIM + column],
                            block[row, column],
                        )
            else:
                self.barrier_hash_matrix.non_diag.blockI[slot] = block_i
                self.barrier_hash_matrix.non_diag.blockJ[slot] = block_j
                for row in ti.static(range(config.DIM)):
                    for column in ti.static(range(config.DIM)):
                        self.barrier_hash_matrix.non_diag.blockH[slot][row * config.DIM + column] = block[row, column]
        else:
            self.barrier_hash_matrix.overflow[0] = 1

    @ti.kernel
    def scatter_barrier_rhs(self, active_mpm_dof: ti.i32):
        for i in range(self.iga.degree_of_freedom):
            self.iga.rhs[i] += self.barrier_grad[i]
        for i in range(active_mpm_dof):
            self.mpm.rhs[i] += self.barrier_grad[self.iga.degree_of_freedom + i]

    @ti.kernel
    def assemble_barrier_gradient_for_surface(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_knot_v: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        num_knot_v: ti.i32,
        num_ctrlpts_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
    ):
        """GPU residual-only point--surface barrier assembly."""
        for c in range(self.contacts.shape[0]):
            if self.contacts[c].active and self.contacts[c].surface_id == surface_id:
                sample_id = self.contacts[c].sample_id
                particle_id = self.contacts[c].particle_id
                position = self.mpm.p_temp[sample_id]
                measure = self.mpm.surface_measure[sample_id]
                uknot = self.contacts[c].knot_value[0]
                vknot = self.contacts[c].knot_value[1]
                distance, pointer = evaluate_distance_to_surface_fixed_dim(
                    prefix_num_knot_u,
                    prefix_num_knot_v,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    num_knot_v,
                    surface.knot_vector_u,
                    surface.knot_vector_v,
                    surface.control_points_hat,
                    surface.weights,
                    position,
                    uknot,
                    vknot,
                    basis,
                )
                if ti.static(not self.is_semi) and distance >= self.barrier.activation_distance_term():
                    continue
                num_ctrlpts_v = num_knot_v - basis.basis_v.degree - 1
                span_u = basis.basis_u.FindSpan(
                    prefix_num_knot_u,
                    num_ctrlpts_u,
                    uknot,
                    surface.knot_vector_u,
                )
                span_v = basis.basis_v.FindSpan(
                    prefix_num_knot_v,
                    num_ctrlpts_v,
                    vknot,
                    surface.knot_vector_v,
                )
                nshape = basis.NurbsBasis2d(
                    prefix_num_knot_u,
                    prefix_num_knot_v,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    num_knot_v,
                    uknot,
                    vknot,
                    surface.knot_vector_u,
                    surface.knot_vector_v,
                    surface.weights,
                )
                ddistance2_dpoint = self.derivative.Ddistance2_div_Dpoint(pointer)
                for i in range(span_v - basis.basis_v.degree, span_v + 1):
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        ctrlpt_offset = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                            basis.basis_u.degree + 1
                        )
                        local_ctrlpt_id = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                        global_ctrlpt_id = surface.control_points_id[local_ctrlpt_id]
                        derivative = self.derivative.Ddistance2_div_Dctrlpt(pointer, nshape[ctrlpt_offset])
                        gradient = self._normal_gradient(c, distance, derivative, measure)
                        for d in ti.static(range(config.DIM)):
                            self.barrier_grad[config.DIM * global_ctrlpt_id + d] -= gradient[d]
                for j in range(self.mpm.offset[particle_id]):
                    base_node = self.mpm.LnID[particle_id, j]
                    mpm_offset = self.mpm.node2dof[base_node] - 1
                    if mpm_offset < 0:
                        continue
                    dpoint_dx = self.mpm.shape[particle_id, j] * ti.Matrix.identity(ti.f64, config.DIM)
                    derivative = ddistance2_dpoint @ dpoint_dx
                    gradient = self._normal_gradient(c, distance, derivative, measure)
                    for d in ti.static(range(config.DIM)):
                        self.barrier_grad[self.iga.degree_of_freedom + config.DIM * mpm_offset + d] -= gradient[d]

    @ti.kernel
    def assemble_barrier_gradient_for_curve(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
    ):
        """GPU residual-only point--curve barrier assembly."""
        for c in range(self.contacts.shape[0]):
            if self.contacts[c].active and self.contacts[c].surface_id == surface_id:
                sample_id = self.contacts[c].sample_id
                particle_id = self.contacts[c].particle_id
                position = self.mpm.p_temp[sample_id]
                measure = self.mpm.surface_measure[sample_id]
                uknot = self.contacts[c].knot_value[0]
                distance, pointer = evaluate_distance_to_curve_fixed_dim(
                    prefix_num_knot_u,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    surface.knot_vector_u,
                    surface.control_points_hat,
                    surface.weights,
                    position,
                    uknot,
                    basis,
                )
                if ti.static(not self.is_semi) and distance >= self.barrier.activation_distance_term():
                    continue
                num_ctrlpts = num_knot_u - basis.basis_u.degree - 1
                span_u = basis.basis_u.FindSpan(
                    prefix_num_knot_u,
                    num_ctrlpts,
                    uknot,
                    surface.knot_vector_u,
                )
                nshape = basis.NurbsBasis1d(
                    prefix_num_knot_u,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    uknot,
                    surface.knot_vector_u,
                    surface.weights,
                )
                ddistance2_dpoint = self.derivative.Ddistance2_div_Dpoint(pointer)
                for j in range(span_u - basis.basis_u.degree, span_u + 1):
                    ctrlpt_offset = j - span_u + basis.basis_u.degree
                    local_ctrlpt_id = prefix_num_ctrlpts + j
                    global_ctrlpt_id = surface.control_points_id[local_ctrlpt_id]
                    derivative = self.derivative.Ddistance2_div_Dctrlpt(pointer, nshape[ctrlpt_offset])
                    gradient = self._normal_gradient(c, distance, derivative, measure)
                    for d in ti.static(range(config.DIM)):
                        self.barrier_grad[config.DIM * global_ctrlpt_id + d] -= gradient[d]
                for j in range(self.mpm.offset[particle_id]):
                    base_node = self.mpm.LnID[particle_id, j]
                    mpm_offset = self.mpm.node2dof[base_node] - 1
                    if mpm_offset < 0:
                        continue
                    dpoint_dx = self.mpm.shape[particle_id, j] * ti.Matrix.identity(ti.f64, config.DIM)
                    derivative = ddistance2_dpoint @ dpoint_dx
                    gradient = self._normal_gradient(c, distance, derivative, measure)
                    for d in ti.static(range(config.DIM)):
                        self.barrier_grad[self.iga.degree_of_freedom + config.DIM * mpm_offset + d] -= gradient[d]

    @ti.kernel
    def prepare_projected_barrier_surface(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_knot_v: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        num_knot_v: ti.i32,
        num_ctrlpts_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
    ):
        """Cache the expensive local point--surface projection once per pair."""
        for c in range(self.contacts.shape[0]):
            if self.contacts[c].surface_id == surface_id:
                self.barrier_projection_active[c] = 0
                if self.contacts[c].active:
                    sample_id = self.contacts[c].sample_id
                    position = self.mpm.p_temp[sample_id]
                    measure = self.mpm.surface_measure[sample_id]
                    uknot = self.contacts[c].knot_value[0]
                    vknot = self.contacts[c].knot_value[1]
                    distance, pointer = evaluate_distance_to_surface_fixed_dim(
                        prefix_num_knot_u,
                        prefix_num_knot_v,
                        prefix_num_ctrlpts,
                        num_knot_u,
                        num_knot_v,
                        surface.knot_vector_u,
                        surface.knot_vector_v,
                        surface.control_points_hat,
                        surface.weights,
                        position,
                        uknot,
                        vknot,
                        basis,
                    )
                    if distance < self.barrier.activation_distance_term() or ti.static(self.is_semi):
                        (
                            span_u,
                            span_v,
                            nshape,
                            derivative_u,
                            derivative_v,
                            tangent_u,
                            tangent_v,
                            curvature_uu,
                            curvature_vv,
                            curvature_uv,
                        ) = basis.NurbsBasisHessian(
                            prefix_num_knot_u,
                            prefix_num_knot_v,
                            prefix_num_ctrlpts,
                            num_knot_u,
                            num_knot_v,
                            uknot,
                            vknot,
                            surface.knot_vector_u,
                            surface.knot_vector_v,
                            surface.control_points_hat,
                            surface.weights,
                        )
                        num_ctrlpts_v = num_knot_v - basis.basis_v.degree - 1
                        lower_u = surface.knot_vector_u[prefix_num_knot_u + basis.basis_u.degree]
                        upper_u = surface.knot_vector_u[prefix_num_knot_u + num_ctrlpts_u]
                        lower_v = surface.knot_vector_v[prefix_num_knot_v + basis.basis_v.degree]
                        upper_v = surface.knot_vector_v[prefix_num_knot_v + num_ctrlpts_v]
                        free_u = 1
                        free_v = 1
                        if uknot <= lower_u + 1.0e-12 or uknot >= upper_u - 1.0e-12:
                            free_u = 0
                        if vknot <= lower_v + 1.0e-12 or vknot >= upper_v - 1.0e-12:
                            free_v = 0
                        metric, projection_valid = surface_barrier_projected_metric(
                            pointer,
                            distance,
                            tangent_u,
                            tangent_v,
                            curvature_uu,
                            curvature_vv,
                            curvature_uv,
                            nshape,
                            derivative_u,
                            derivative_v,
                            free_u,
                            free_v,
                            self._normal_terms(c, distance)[1],
                            self._normal_terms(c, distance)[2],
                            measure,
                        )
                        point_jacobian = surface_point_reduced_jacobian(
                            pointer,
                            tangent_u,
                            tangent_v,
                            free_u,
                            free_v,
                        )
                        self.barrier_projection_active[c] = 1
                        self.barrier_projection_distance[c] = distance
                        self.barrier_projection_span[c] = ti.Vector([span_u, span_v])
                        self.barrier_projection_free[c] = ti.Vector([free_u, free_v])
                        self.barrier_projection_pointer[c] = pointer
                        self.barrier_projection_tangent_u[c] = tangent_u
                        self.barrier_projection_tangent_v[c] = tangent_v
                        self.barrier_projection_metric[c] = metric
                        self.barrier_projection_point_jacobian[c] = point_jacobian
                        for support in range(nshape.n):
                            self.barrier_projection_shape[c, support] = nshape[support]
                            self.barrier_projection_derivative_u[c, support] = derivative_u[support]
                            self.barrier_projection_derivative_v[c, support] = derivative_v[support]
                        if not projection_valid:
                            self.barrier_projection_status[None] = 1

    @ti.kernel
    def assemble_projected_barrier_matrix_for_surface(
        self,
        surface_id: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_ctrlpts_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
    ):
        """Scatter cached PSD point--surface blocks without recompiling geometry."""
        for c in range(self.contacts.shape[0]):
            if self.barrier_projection_active[c] and self.contacts[c].surface_id == surface_id:
                particle_id = self.contacts[c].particle_id
                sample_id = self.contacts[c].sample_id
                measure = self.mpm.surface_measure[sample_id]
                distance = self.barrier_projection_distance[c]
                pointer = self.barrier_projection_pointer[c]
                tangent_u = self.barrier_projection_tangent_u[c]
                tangent_v = self.barrier_projection_tangent_v[c]
                span_u = self.barrier_projection_span[c][0]
                span_v = self.barrier_projection_span[c][1]
                free_u = self.barrier_projection_free[c][0]
                free_v = self.barrier_projection_free[c][1]
                projected_metric = self.barrier_projection_metric[c]
                projected_point_jacobian = self.barrier_projection_point_jacobian[c]
                ddistance2_dpoint = self.derivative.Ddistance2_div_Dpoint(pointer)

                for i in range(span_v - basis.basis_v.degree, span_v + 1):
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        ctrlpt_offset_1 = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                            basis.basis_u.degree + 1
                        )
                        local_ctrlpt_id_1 = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                        global_ctrlpt_id_1 = surface.control_points_id[local_ctrlpt_id_1]
                        shape_1 = self.barrier_projection_shape[c, ctrlpt_offset_1]
                        derivative_u_1 = self.barrier_projection_derivative_u[c, ctrlpt_offset_1]
                        derivative_v_1 = self.barrier_projection_derivative_v[c, ctrlpt_offset_1]
                        ddistance2_dctrlpt_1 = self.derivative.Ddistance2_div_Dctrlpt(pointer, shape_1)
                        grad_ctrl = self._normal_gradient(c, distance, ddistance2_dctrlpt_1, measure)
                        for component in ti.static(range(config.DIM)):
                            self.barrier_grad[config.DIM * global_ctrlpt_id_1 + component] -= grad_ctrl[component]
                        projected_ctrlpt_jacobian_1 = surface_control_reduced_jacobian(
                            pointer,
                            tangent_u,
                            tangent_v,
                            shape_1,
                            derivative_u_1,
                            derivative_v_1,
                            free_u,
                            free_v,
                        )

                        for m in range(span_v - basis.basis_v.degree, span_v + 1):
                            for n in range(span_u - basis.basis_u.degree, span_u + 1):
                                ctrlpt_offset_2 = (n - span_u + basis.basis_u.degree) + (
                                    m - span_v + basis.basis_v.degree
                                ) * (basis.basis_u.degree + 1)
                                local_ctrlpt_id_2 = prefix_num_ctrlpts + n + m * num_ctrlpts_u
                                global_ctrlpt_id_2 = surface.control_points_id[local_ctrlpt_id_2]
                                projected_ctrlpt_jacobian_2 = surface_control_reduced_jacobian(
                                    pointer,
                                    tangent_u,
                                    tangent_v,
                                    self.barrier_projection_shape[c, ctrlpt_offset_2],
                                    self.barrier_projection_derivative_u[c, ctrlpt_offset_2],
                                    self.barrier_projection_derivative_v[c, ctrlpt_offset_2],
                                    free_u,
                                    free_v,
                                )
                                hessian_cc = projected_ctrlpt_jacobian_1.transpose() @ (
                                    projected_metric @ projected_ctrlpt_jacobian_2
                                )
                                self.add_barrier_block(
                                    c,
                                    ctrlpt_offset_1,
                                    ctrlpt_offset_2,
                                    global_ctrlpt_id_1,
                                    global_ctrlpt_id_2,
                                    hessian_cc,
                                )

                        for k in range(self.mpm.offset[particle_id]):
                            base_node = self.mpm.LnID[particle_id, k]
                            mpm_offset = self.mpm.node2dof[base_node] - 1
                            if mpm_offset >= 0:
                                dpoint_dx = self.mpm.shape[particle_id, k] * ti.Matrix.identity(ti.f64, config.DIM)
                                hessian_cm = projected_ctrlpt_jacobian_1.transpose() @ (
                                    projected_metric @ (projected_point_jacobian @ dpoint_dx)
                                )
                                coupled_mpm_block = self.iga.degree_of_freedom // config.DIM + mpm_offset
                                self.add_barrier_block(
                                    c,
                                    ctrlpt_offset_1,
                                    ti.static(self.contact_ctrlpts_capacity) + k,
                                    global_ctrlpt_id_1,
                                    coupled_mpm_block,
                                    hessian_cm,
                                )
                                self.add_barrier_block(
                                    c,
                                    ti.static(self.contact_ctrlpts_capacity) + k,
                                    ctrlpt_offset_1,
                                    coupled_mpm_block,
                                    global_ctrlpt_id_1,
                                    hessian_cm.transpose(),
                                )

                for j in range(self.mpm.offset[particle_id]):
                    base_jnode = self.mpm.LnID[particle_id, j]
                    mpm_offset_j = self.mpm.node2dof[base_jnode] - 1
                    if mpm_offset_j >= 0:
                        dpoint_dx1 = self.mpm.shape[particle_id, j] * ti.Matrix.identity(ti.f64, config.DIM)
                        ddistance2_dx1 = ddistance2_dpoint @ dpoint_dx1
                        grad_mpm = self._normal_gradient(c, distance, ddistance2_dx1, measure)
                        for component in ti.static(range(config.DIM)):
                            self.barrier_grad[
                                self.iga.degree_of_freedom + config.DIM * mpm_offset_j + component
                            ] -= grad_mpm[component]
                        for k in range(self.mpm.offset[particle_id]):
                            base_knode = self.mpm.LnID[particle_id, k]
                            mpm_offset_k = self.mpm.node2dof[base_knode] - 1
                            if mpm_offset_k >= 0:
                                dpoint_dx2 = self.mpm.shape[particle_id, k] * ti.Matrix.identity(ti.f64, config.DIM)
                                hessian_mm = dpoint_dx1.transpose() @ (
                                    projected_point_jacobian.transpose()
                                    @ (projected_metric @ (projected_point_jacobian @ dpoint_dx2))
                                )
                                coupled_mpm_block_j = self.iga.degree_of_freedom // config.DIM + mpm_offset_j
                                coupled_mpm_block_k = self.iga.degree_of_freedom // config.DIM + mpm_offset_k
                                self.add_barrier_block(
                                    c,
                                    ti.static(self.contact_ctrlpts_capacity) + j,
                                    ti.static(self.contact_ctrlpts_capacity) + k,
                                    coupled_mpm_block_j,
                                    coupled_mpm_block_k,
                                    hessian_mm,
                                )

    @ti.kernel
    def assemble_barrier_matrix_for_surface(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_knot_v: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        num_knot_v: ti.i32,
        num_ctrlpts_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
    ):
        for c in range(self.contacts.shape[0]):
            if self.contacts[c].active and self.contacts[c].surface_id == surface_id:
                sample_id = self.contacts[c].sample_id
                particle_id = self.contacts[c].particle_id
                position = self.mpm.p_temp[sample_id]
                measure = self.mpm.surface_measure[sample_id]
                uknot = self.contacts[c].knot_value[0]
                vknot = self.contacts[c].knot_value[1]

                distance, pointer = evaluate_distance_to_surface_fixed_dim(
                    prefix_num_knot_u,
                    prefix_num_knot_v,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    num_knot_v,
                    surface.knot_vector_u,
                    surface.knot_vector_v,
                    surface.control_points_hat,
                    surface.weights,
                    position,
                    uknot,
                    vknot,
                    basis,
                )
                if ti.static(not self.is_semi) and distance >= self.barrier.activation_distance_term():
                    continue

                (
                    span_u,
                    span_v,
                    nshape,
                    derivate_u,
                    derivate_v,
                    tangent_u,
                    tangent_v,
                    curvature_uu,
                    curvature_vv,
                    curvature_uv,
                ) = basis.NurbsBasisHessian(
                    prefix_num_knot_u,
                    prefix_num_knot_v,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    num_knot_v,
                    uknot,
                    vknot,
                    surface.knot_vector_u,
                    surface.knot_vector_v,
                    surface.control_points_hat,
                    surface.weights,
                )
                num_ctrlpts_v = num_knot_v - basis.basis_v.degree - 1
                lower_u = surface.knot_vector_u[prefix_num_knot_u + basis.basis_u.degree]
                upper_u = surface.knot_vector_u[prefix_num_knot_u + num_ctrlpts_u]
                lower_v = surface.knot_vector_v[prefix_num_knot_v + basis.basis_v.degree]
                upper_v = surface.knot_vector_v[prefix_num_knot_v + num_ctrlpts_v]
                parameter_tolerance = 1.0e-12
                free_u = 1
                free_v = 1
                if uknot <= lower_u + parameter_tolerance or uknot >= upper_u - parameter_tolerance:
                    free_u = 0
                if vknot <= lower_v + parameter_tolerance or vknot >= upper_v - parameter_tolerance:
                    free_v = 0
                ddistance2_dpoint = self.derivative.Ddistance2_div_Dpoint(pointer)
                # Keep the full IFT Hessian out of the common projected
                # specialization; Taichi otherwise compiles it before replacing it.
                d2distance_d2point = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                if ti.static(not self.project_lagged_hessians):
                    d2distance_d2point = self.derivative.D2distance2_div_D2point_SurfaceActive(
                        pointer,
                        tangent_u,
                        tangent_v,
                        curvature_uu,
                        curvature_vv,
                        curvature_uv,
                        free_u,
                        free_v,
                    )
                projected_metric = ti.Matrix.zero(ti.f64, config.DIM + 2, config.DIM + 2)
                projected_point_jacobian = ti.Matrix.zero(ti.f64, config.DIM + 2, config.DIM)
                if ti.static(self.project_lagged_hessians):
                    projected_metric, projection_valid = surface_barrier_projected_metric(
                        pointer,
                        distance,
                        tangent_u,
                        tangent_v,
                        curvature_uu,
                        curvature_vv,
                        curvature_uv,
                        nshape,
                        derivate_u,
                        derivate_v,
                        free_u,
                        free_v,
                        self._normal_terms(c, distance)[1],
                        self._normal_terms(c, distance)[2],
                        measure,
                    )
                    projected_point_jacobian = surface_point_reduced_jacobian(
                        pointer,
                        tangent_u,
                        tangent_v,
                        free_u,
                        free_v,
                    )
                    if not projection_valid:
                        self.barrier_projection_status[None] = 1

                for i in range(span_v - basis.basis_v.degree, span_v + 1):
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        local_ctrlpt_id_1 = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                        ctrlpt_offset_1 = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                            basis.basis_u.degree + 1
                        )
                        global_ctrlpt_id_1 = surface.control_points_id[local_ctrlpt_id_1]
                        ddistance2_dctrlpt_1 = self.derivative.Ddistance2_div_Dctrlpt(pointer, nshape[ctrlpt_offset_1])
                        grad_ctrl = self._normal_gradient(c, distance, ddistance2_dctrlpt_1, measure)
                        for d in ti.static(range(config.DIM)):
                            self.barrier_grad[config.DIM * global_ctrlpt_id_1 + d] -= grad_ctrl[d]

                        d2distance_dctrlpt_dpoint = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                        projected_ctrlpt_jacobian_1 = ti.Matrix.zero(ti.f64, config.DIM + 2, config.DIM)
                        if ti.static(self.project_lagged_hessians):
                            projected_ctrlpt_jacobian_1 = surface_control_reduced_jacobian(
                                pointer,
                                tangent_u,
                                tangent_v,
                                nshape[ctrlpt_offset_1],
                                derivate_u[ctrlpt_offset_1],
                                derivate_v[ctrlpt_offset_1],
                                free_u,
                                free_v,
                            )
                        else:
                            d2distance_dctrlpt_dpoint = self.derivative.D2distance2_div_DctrlptDpoint_SurfaceActive(
                                pointer,
                                tangent_u,
                                tangent_v,
                                curvature_uu,
                                curvature_vv,
                                curvature_uv,
                                nshape[ctrlpt_offset_1],
                                derivate_u[ctrlpt_offset_1],
                                derivate_v[ctrlpt_offset_1],
                                free_u,
                                free_v,
                            )

                        for m in range(span_v - basis.basis_v.degree, span_v + 1):
                            for n in range(span_u - basis.basis_u.degree, span_u + 1):
                                local_ctrlpt_id_2 = prefix_num_ctrlpts + n + m * num_ctrlpts_u
                                ctrlpt_offset_2 = (n - span_u + basis.basis_u.degree) + (
                                    m - span_v + basis.basis_v.degree
                                ) * (basis.basis_u.degree + 1)
                                global_ctrlpt_id_2 = surface.control_points_id[local_ctrlpt_id_2]
                                hessian_cc = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                                if ti.static(self.project_lagged_hessians):
                                    projected_ctrlpt_jacobian_2 = surface_control_reduced_jacobian(
                                        pointer,
                                        tangent_u,
                                        tangent_v,
                                        nshape[ctrlpt_offset_2],
                                        derivate_u[ctrlpt_offset_2],
                                        derivate_v[ctrlpt_offset_2],
                                        free_u,
                                        free_v,
                                    )
                                    hessian_cc = projected_ctrlpt_jacobian_1.transpose() @ (
                                        projected_metric @ projected_ctrlpt_jacobian_2
                                    )
                                else:
                                    ddistance2_dctrlpt_2 = self.derivative.Ddistance2_div_Dctrlpt(
                                        pointer, nshape[ctrlpt_offset_2]
                                    )
                                    d2distance_d2ctrlpt = self.derivative.D2distance2_div_D2ctrlpt_SurfaceActive(
                                        pointer,
                                        tangent_u,
                                        tangent_v,
                                        curvature_uu,
                                        curvature_vv,
                                        curvature_uv,
                                        nshape[ctrlpt_offset_1],
                                        derivate_u[ctrlpt_offset_1],
                                        derivate_v[ctrlpt_offset_1],
                                        nshape[ctrlpt_offset_2],
                                        derivate_u[ctrlpt_offset_2],
                                        derivate_v[ctrlpt_offset_2],
                                        free_u,
                                        free_v,
                                    )
                                    hessian_cc = self._normal_hessian(
                                        c,
                                        distance,
                                        ddistance2_dctrlpt_1,
                                        ddistance2_dctrlpt_2,
                                        d2distance_d2ctrlpt,
                                        measure,
                                    )
                                self.add_barrier_block(
                                    c,
                                    ctrlpt_offset_1,
                                    ctrlpt_offset_2,
                                    global_ctrlpt_id_1,
                                    global_ctrlpt_id_2,
                                    hessian_cc,
                                )

                        for k in range(self.mpm.offset[particle_id]):
                            base_node = self.mpm.LnID[particle_id, k]
                            mpm_offset = self.mpm.node2dof[base_node] - 1
                            if mpm_offset < 0:
                                continue
                            dpoint_dx = self.mpm.shape[particle_id, k] * ti.Matrix.identity(ti.f64, config.DIM)
                            ddistance2_dx = ddistance2_dpoint @ dpoint_dx
                            hessian_cm = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                            if ti.static(self.project_lagged_hessians):
                                hessian_cm = projected_ctrlpt_jacobian_1.transpose() @ (
                                    projected_metric @ (projected_point_jacobian @ dpoint_dx)
                                )
                            else:
                                d2distance_dctrlpt_dx = d2distance_dctrlpt_dpoint @ dpoint_dx
                                hessian_cm = self._normal_hessian(
                                    c,
                                    distance,
                                    ddistance2_dctrlpt_1,
                                    ddistance2_dx,
                                    d2distance_dctrlpt_dx,
                                    measure,
                                )
                            coupled_mpm_block = self.iga.degree_of_freedom // config.DIM + mpm_offset
                            self.add_barrier_block(
                                c,
                                ctrlpt_offset_1,
                                ti.static(self.contact_ctrlpts_capacity) + k,
                                global_ctrlpt_id_1,
                                coupled_mpm_block,
                                hessian_cm,
                            )
                            self.add_barrier_block(
                                c,
                                ti.static(self.contact_ctrlpts_capacity) + k,
                                ctrlpt_offset_1,
                                coupled_mpm_block,
                                global_ctrlpt_id_1,
                                hessian_cm.transpose(),
                            )

                for j in range(self.mpm.offset[particle_id]):
                    base_jnode = self.mpm.LnID[particle_id, j]
                    mpm_offset_j = self.mpm.node2dof[base_jnode] - 1
                    if mpm_offset_j < 0:
                        continue
                    dpoint_dx1 = self.mpm.shape[particle_id, j] * ti.Matrix.identity(ti.f64, config.DIM)
                    ddistance2_dx1 = ddistance2_dpoint @ dpoint_dx1
                    grad_mpm = self._normal_gradient(c, distance, ddistance2_dx1, measure)
                    for d in ti.static(range(config.DIM)):
                        self.barrier_grad[self.iga.degree_of_freedom + config.DIM * mpm_offset_j + d] -= grad_mpm[d]

                    for k in range(self.mpm.offset[particle_id]):
                        base_knode = self.mpm.LnID[particle_id, k]
                        mpm_offset_k = self.mpm.node2dof[base_knode] - 1
                        if mpm_offset_k < 0:
                            continue
                        dpoint_dx2 = self.mpm.shape[particle_id, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        ddistance2_dx2 = ddistance2_dpoint @ dpoint_dx2
                        hessian_mm = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                        if ti.static(self.project_lagged_hessians):
                            hessian_mm = dpoint_dx1.transpose() @ (
                                projected_point_jacobian.transpose()
                                @ (projected_metric @ (projected_point_jacobian @ dpoint_dx2))
                            )
                        else:
                            d2distance_dx1dx2 = dpoint_dx1 @ d2distance_d2point @ dpoint_dx2
                            hessian_mm = self._normal_hessian(
                                c,
                                distance,
                                ddistance2_dx1,
                                ddistance2_dx2,
                                d2distance_dx1dx2,
                                measure,
                            )
                        coupled_mpm_block_j = self.iga.degree_of_freedom // config.DIM + mpm_offset_j
                        coupled_mpm_block_k = self.iga.degree_of_freedom // config.DIM + mpm_offset_k
                        self.add_barrier_block(
                            c,
                            ti.static(self.contact_ctrlpts_capacity) + j,
                            ti.static(self.contact_ctrlpts_capacity) + k,
                            coupled_mpm_block_j,
                            coupled_mpm_block_k,
                            hessian_mm,
                        )

    @ti.kernel
    def assemble_barrier_matrix_for_curve(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
    ):
        for c in range(self.contacts.shape[0]):
            if self.contacts[c].active and self.contacts[c].surface_id == surface_id:
                sample_id = self.contacts[c].sample_id
                particle_id = self.contacts[c].particle_id
                position = self.mpm.p_temp[sample_id]
                measure = self.mpm.surface_measure[sample_id]
                uknot = self.contacts[c].knot_value[0]

                distance, pointer = evaluate_distance_to_curve_fixed_dim(
                    prefix_num_knot_u,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    surface.knot_vector_u,
                    surface.control_points_hat,
                    surface.weights,
                    position,
                    uknot,
                    basis,
                )
                if ti.static(not self.is_semi) and distance >= self.barrier.activation_distance_term():
                    continue

                span_u, nshape, derivative_u, tangent_u, curvature_uu = basis.NurbsBasisHessian(
                    prefix_num_knot_u,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    uknot,
                    surface.knot_vector_u,
                    surface.control_points_hat,
                    surface.weights,
                )
                num_ctrlpts = num_knot_u - basis.basis_u.degree - 1
                lower_u = surface.knot_vector_u[prefix_num_knot_u + basis.basis_u.degree]
                upper_u = surface.knot_vector_u[prefix_num_knot_u + num_ctrlpts]
                free_u = 1
                if uknot <= lower_u + 1.0e-12 or uknot >= upper_u - 1.0e-12:
                    free_u = 0
                ddistance2_dpoint = self.derivative.Ddistance2_div_Dpoint(pointer)
                d2distance_d2point = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                if ti.static(not self.project_lagged_hessians):
                    d2distance_d2point = self.derivative.D2distance2_div_D2point_CurveActive(
                        pointer, tangent_u, curvature_uu, free_u
                    )
                projected_metric = ti.Matrix.zero(ti.f64, config.DIM + 1, config.DIM + 1)
                projected_point_jacobian = ti.Matrix.zero(ti.f64, config.DIM + 1, config.DIM)
                if ti.static(self.project_lagged_hessians):
                    projected_metric, projection_valid = curve_barrier_projected_metric(
                        pointer,
                        distance,
                        tangent_u,
                        curvature_uu,
                        nshape,
                        derivative_u,
                        free_u,
                        self._normal_terms(c, distance)[1],
                        self._normal_terms(c, distance)[2],
                        measure,
                    )
                    projected_point_jacobian = curve_point_reduced_jacobian(pointer, tangent_u, free_u)
                    if not projection_valid:
                        self.barrier_projection_status[None] = 1

                for j in range(span_u - basis.basis_u.degree, span_u + 1):
                    ctrlpt_offset_1 = j - span_u + basis.basis_u.degree
                    local_ctrlpt_id_1 = prefix_num_ctrlpts + j
                    global_ctrlpt_id_1 = surface.control_points_id[local_ctrlpt_id_1]
                    ddistance2_dctrlpt_1 = self.derivative.Ddistance2_div_Dctrlpt(pointer, nshape[ctrlpt_offset_1])
                    grad_ctrl = self._normal_gradient(c, distance, ddistance2_dctrlpt_1, measure)
                    for d in ti.static(range(config.DIM)):
                        self.barrier_grad[config.DIM * global_ctrlpt_id_1 + d] -= grad_ctrl[d]

                    d2distance_dctrlpt_dpoint = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                    projected_ctrlpt_jacobian_1 = ti.Matrix.zero(ti.f64, config.DIM + 1, config.DIM)
                    if ti.static(self.project_lagged_hessians):
                        projected_ctrlpt_jacobian_1 = curve_control_reduced_jacobian(
                            pointer,
                            tangent_u,
                            nshape[ctrlpt_offset_1],
                            derivative_u[ctrlpt_offset_1],
                            free_u,
                        )
                    else:
                        d2distance_dctrlpt_dpoint = self.derivative.D2distance2_div_DctrlptDpoint_CurveActive(
                            pointer,
                            tangent_u,
                            curvature_uu,
                            nshape[ctrlpt_offset_1],
                            derivative_u[ctrlpt_offset_1],
                            free_u,
                        )

                    for n in range(span_u - basis.basis_u.degree, span_u + 1):
                        ctrlpt_offset_2 = n - span_u + basis.basis_u.degree
                        local_ctrlpt_id_2 = prefix_num_ctrlpts + n
                        global_ctrlpt_id_2 = surface.control_points_id[local_ctrlpt_id_2]
                        hessian_cc = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                        if ti.static(self.project_lagged_hessians):
                            projected_ctrlpt_jacobian_2 = curve_control_reduced_jacobian(
                                pointer,
                                tangent_u,
                                nshape[ctrlpt_offset_2],
                                derivative_u[ctrlpt_offset_2],
                                free_u,
                            )
                            hessian_cc = projected_ctrlpt_jacobian_1.transpose() @ (
                                projected_metric @ projected_ctrlpt_jacobian_2
                            )
                        else:
                            ddistance2_dctrlpt_2 = self.derivative.Ddistance2_div_Dctrlpt(
                                pointer, nshape[ctrlpt_offset_2]
                            )
                            d2distance_d2ctrlpt = self.derivative.D2distance2_div_D2ctrlpt_CurveActive(
                                pointer,
                                tangent_u,
                                curvature_uu,
                                nshape[ctrlpt_offset_1],
                                derivative_u[ctrlpt_offset_1],
                                nshape[ctrlpt_offset_2],
                                derivative_u[ctrlpt_offset_2],
                                free_u,
                            )
                            hessian_cc = self._normal_hessian(
                                c,
                                distance,
                                ddistance2_dctrlpt_1,
                                ddistance2_dctrlpt_2,
                                d2distance_d2ctrlpt,
                                measure,
                            )
                        self.add_barrier_block(
                            c,
                            ctrlpt_offset_1,
                            ctrlpt_offset_2,
                            global_ctrlpt_id_1,
                            global_ctrlpt_id_2,
                            hessian_cc,
                        )

                    for k in range(self.mpm.offset[particle_id]):
                        base_node = self.mpm.LnID[particle_id, k]
                        mpm_offset = self.mpm.node2dof[base_node] - 1
                        if mpm_offset < 0:
                            continue
                        dpoint_dx = self.mpm.shape[particle_id, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        ddistance2_dx = ddistance2_dpoint @ dpoint_dx
                        hessian_cm = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                        if ti.static(self.project_lagged_hessians):
                            hessian_cm = projected_ctrlpt_jacobian_1.transpose() @ (
                                projected_metric @ (projected_point_jacobian @ dpoint_dx)
                            )
                        else:
                            d2distance_dctrlpt_dx = d2distance_dctrlpt_dpoint @ dpoint_dx
                            hessian_cm = self._normal_hessian(
                                c,
                                distance,
                                ddistance2_dctrlpt_1,
                                ddistance2_dx,
                                d2distance_dctrlpt_dx,
                                measure,
                            )
                        coupled_mpm_block = self.iga.degree_of_freedom // config.DIM + mpm_offset
                        self.add_barrier_block(
                            c,
                            ctrlpt_offset_1,
                            ti.static(self.contact_ctrlpts_capacity) + k,
                            global_ctrlpt_id_1,
                            coupled_mpm_block,
                            hessian_cm,
                        )
                        self.add_barrier_block(
                            c,
                            ti.static(self.contact_ctrlpts_capacity) + k,
                            ctrlpt_offset_1,
                            coupled_mpm_block,
                            global_ctrlpt_id_1,
                            hessian_cm.transpose(),
                        )

                for j in range(self.mpm.offset[particle_id]):
                    base_jnode = self.mpm.LnID[particle_id, j]
                    mpm_offset_j = self.mpm.node2dof[base_jnode] - 1
                    if mpm_offset_j < 0:
                        continue
                    dpoint_dx1 = self.mpm.shape[particle_id, j] * ti.Matrix.identity(ti.f64, config.DIM)
                    ddistance2_dx1 = ddistance2_dpoint @ dpoint_dx1
                    grad_mpm = self._normal_gradient(c, distance, ddistance2_dx1, measure)
                    for d in ti.static(range(config.DIM)):
                        self.barrier_grad[self.iga.degree_of_freedom + config.DIM * mpm_offset_j + d] -= grad_mpm[d]

                    for k in range(self.mpm.offset[particle_id]):
                        base_knode = self.mpm.LnID[particle_id, k]
                        mpm_offset_k = self.mpm.node2dof[base_knode] - 1
                        if mpm_offset_k < 0:
                            continue
                        dpoint_dx2 = self.mpm.shape[particle_id, k] * ti.Matrix.identity(ti.f64, config.DIM)
                        ddistance2_dx2 = ddistance2_dpoint @ dpoint_dx2
                        hessian_mm = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                        if ti.static(self.project_lagged_hessians):
                            hessian_mm = dpoint_dx1.transpose() @ (
                                projected_point_jacobian.transpose()
                                @ (projected_metric @ (projected_point_jacobian @ dpoint_dx2))
                            )
                        else:
                            d2distance_dx1dx2 = dpoint_dx1 @ d2distance_d2point @ dpoint_dx2
                            hessian_mm = self._normal_hessian(
                                c,
                                distance,
                                ddistance2_dx1,
                                ddistance2_dx2,
                                d2distance_dx1dx2,
                                measure,
                            )
                        coupled_mpm_block_j = self.iga.degree_of_freedom // config.DIM + mpm_offset_j
                        coupled_mpm_block_k = self.iga.degree_of_freedom // config.DIM + mpm_offset_k
                        self.add_barrier_block(
                            c,
                            ti.static(self.contact_ctrlpts_capacity) + j,
                            ti.static(self.contact_ctrlpts_capacity) + k,
                            coupled_mpm_block_j,
                            coupled_mpm_block_k,
                            hessian_mm,
                        )

    def initialize_barrier(self, grid_disp=None, iga_grid_disp=None):
        """Build the complete point--NURBS constraint set on the device.

        The shared fixed-dimension closest-point routines perform safeguarded
        projected Newton solves with span and Greville multistarts directly in
        Taichi on every supported architecture.
        """
        if grid_disp is None:
            grid_disp = self.mpm.grid_disp
        if iga_grid_disp is None:
            iga_grid_disp = self.iga.grid_disp
        self.contact_surface.update_from_patch_displacement(self.iga.patch.control_points, iga_grid_disp)
        self.update_particle_pos(grid_disp)
        self.reset_contacts()
        for surface_id in range(self.contact_surface.num_surfaces):
            if config.DIM == 2:
                self.find_closest_curve_contacts(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                )
            else:
                self.find_closest_surface_contacts(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_knot_v[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    int(self.contact_surface.num_knot_v[surface_id + 1]),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                )
        self.curr_barrier_contact_num = int(self.count_contacts())
        expected_pairs = int(self.mpm.total_surface_num) * int(self.contact_surface.num_surfaces)
        # An empty contact product is feasible by construction.  Keep the
        # ``+inf`` sentinel returned by minimum_contact_distance() out of
        # the finite-distance validation below, matching initialize_barrier().
        if expected_pairs > 0:
            minimum = self.minimum_contact_distance()
            if self.is_semi:
                self._measure_semi_constraint_violation(expected_pairs)
            required = float(self.barrier.minimum_distance) + self.strict_feasibility_tolerance
            if not self.is_semi and (not math.isfinite(minimum) or minimum <= required):
                raise RuntimeError(
                    "IGA-MPM IPC requires a strictly feasible positive contact "
                    f"distance (minimum={minimum:.6e}, "
                    f"dmin={float(self.barrier.minimum_distance):.6e}, "
                    f"tolerance={self.strict_feasibility_tolerance:.6e}); "
                    "use contact-aware CCD/line search to preserve feasibility."
                )

    def semi_contact_converged(self):
        return not self.is_semi or float(self.semi_constraint_violation[None]) <= self.barrier.constraint_tolerance

    def assemble_barrier_system(self, need_matrix=True):
        need_matrix = bool(need_matrix)
        if need_matrix:
            self.barrier_hash_matrix.reset_system()
            self.prepare_barrier_matrix_slots()
            if self.project_lagged_hessians:
                self.barrier_projection_status[None] = 0
        self.clear_barrier_system()
        if self.curr_barrier_contact_num == 0:
            return
        for surface_id in range(self.contact_surface.num_surfaces):
            if config.DIM == 2:
                assembler = (
                    self.assemble_barrier_matrix_for_curve if need_matrix else self.assemble_barrier_gradient_for_curve
                )
                assembler(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                )
            else:
                prefix_num_knot_u = int(self.contact_surface.prefix_num_knot_u[surface_id])
                prefix_num_knot_v = int(self.contact_surface.prefix_num_knot_v[surface_id])
                prefix_num_ctrlpts = int(self.contact_surface.prefix_num_ctrlpts[surface_id])
                num_knot_u = int(self.contact_surface.num_knot_u[surface_id + 1])
                num_knot_v = int(self.contact_surface.num_knot_v[surface_id + 1])
                num_ctrlpts_u = int(self.contact_surface.num_ctrlpts_u[surface_id + 1])
                basis = self.contact_surface.basis[surface_id]
                if need_matrix and self.project_lagged_hessians:
                    self.prepare_projected_barrier_surface(
                        surface_id,
                        prefix_num_knot_u,
                        prefix_num_knot_v,
                        prefix_num_ctrlpts,
                        num_knot_u,
                        num_knot_v,
                        num_ctrlpts_u,
                        self.contact_surface,
                        basis,
                    )
                    self.assemble_projected_barrier_matrix_for_surface(
                        surface_id,
                        prefix_num_ctrlpts,
                        num_ctrlpts_u,
                        self.contact_surface,
                        basis,
                    )
                else:
                    assembler = (
                        self.assemble_barrier_matrix_for_surface
                        if need_matrix
                        else self.assemble_barrier_gradient_for_surface
                    )
                    assembler(
                        surface_id,
                        prefix_num_knot_u,
                        prefix_num_knot_v,
                        prefix_num_ctrlpts,
                        num_knot_u,
                        num_knot_v,
                        num_ctrlpts_u,
                        self.contact_surface,
                        basis,
                    )
        if need_matrix and self.project_lagged_hessians and int(self.barrier_projection_status[None]) != 0:
            raise RuntimeError(
                "IGA-MPM IPC barrier PSD projection failed because the "
                "active point--NURBS closest-point IFT system or its reduced "
                "Jacobian is singular; refusing a frozen-coordinate or "
                "blockwise approximation."
            )
        if need_matrix and (int(self.barrier_nnz_overflow[0]) != 0 or int(self.barrier_hash_matrix.overflow[0]) != 0):
            raise RuntimeError(
                f"IGA-MPM barrier block buffer overflow: used {int(self.barrier_nnz_count[0])}, "
                f"capacity {self.barrier_nnz_capacity}. Increase barrier_nnz."
            )

    def add_barrier_rhs_to_subsystems(self):
        self.scatter_barrier_rhs(self.mpm.active_dof)

    def barrier_matrix(self):
        self.barrier_hash_matrix.finalize_taichi_assembly()
        return self.barrier_hash_matrix.to_scipy(self.total_degree_of_freedom // config.DIM).tocsr()

    def barrier_blocks(self, active_mpm_dof=None):
        active_mpm_dof = self.mpm.active_dof if active_mpm_dof is None else int(active_mpm_dof)
        iga_dof = self.iga.degree_of_freedom
        mpm_begin = iga_dof
        mpm_end = iga_dof + active_mpm_dof
        matrix = self.barrier_matrix()
        return {
            "K_ii": matrix[:iga_dof, :iga_dof],
            "K_im": matrix[:iga_dof, mpm_begin:mpm_end],
            "K_mi": matrix[mpm_begin:mpm_end, :iga_dof],
            "K_mm": matrix[mpm_begin:mpm_end, mpm_begin:mpm_end],
        }

    def barrier_contact_forces(self, active_mpm_dof=None):
        active_mpm_dof = self.mpm.active_dof if active_mpm_dof is None else int(active_mpm_dof)
        grad = self.barrier_grad.to_numpy()
        iga_dof = self.iga.degree_of_freedom
        return {
            "iga": grad[:iga_dof].copy(),
            "mpm": grad[iga_dof : iga_dof + active_mpm_dof].copy(),
        }

    @ti.kernel
    def barrier_potential_energy(self) -> ti.f64:
        energy = 0.0
        for c in range(self.contacts.shape[0]):
            if self.contacts[c].active:
                sample_id = self.contacts[c].sample_id
                energy += self._normal_energy(
                    c,
                    self.contacts[c].distance,
                    self.mpm.surface_measure[sample_id],
                )
        return energy

    def _restore_accepted_displacements(self, iga_displacement, mpm_displacement):
        """Roll a failed coupled solve back to its entry displacement state."""
        self.iga.grid_disp.from_numpy(np.asarray(iga_displacement, dtype=np.float64))
        self.mpm.grid_disp.from_numpy(np.asarray(mpm_displacement, dtype=np.float64))
        self._synchronize_trial_state_with_accepted()
        if self.activate_fric and self.friction_mode == "lagged":
            self._initialize_friction_from_current_barrier()
        else:
            self.curr_friction_contact_num = 0

    def conservative_contact_step(
        self,
        correction,
        max_step=1.0,
        safety=None,
        verify=True,
    ):
        """Upload an external correction and run point--NURBS ACCD in Taichi."""
        self._load_external_correction(correction)
        return self.conservative_contact_step_device(max_step=max_step, safety=safety, verify=verify)

    def coupled_potential_energy(
        self,
        iga_grid_disp=None,
        mpm_grid_disp=None,
        include_friction=True,
    ):
        """Body plus IPC potential at an already initialized contact query."""
        if iga_grid_disp is None:
            iga_grid_disp = self.iga.grid_disp
        if mpm_grid_disp is None:
            mpm_grid_disp = self.mpm.grid_disp
        energy = float(self.iga.total_energy(iga_grid_disp))
        energy += float(self.mpm.total_energy(mpm_grid_disp))
        energy += float(self.barrier_potential_energy())
        if include_friction:
            energy += self.friction_potential_energy(iga_grid_disp)
        return energy

    @ti.kernel
    def _clear_mpm_body_rhs(self):
        """Clear a residual probe without destroying its Newton direction."""
        for dof in self.mpm.rhs:
            self.mpm.rhs[dof] = 0.0

    @ti.kernel
    def _finish_device_residual_constraints(self, active_dof: ti.i32):
        """Keep physical free residuals and expose fixed corrections."""
        for dof in range(active_dof):
            if self.monolithic_fixed[dof] != 0:
                self.monolithic_rhs[dof] = self.monolithic_fixed_correction[dof]

    @ti.kernel
    def _device_current_displacement_inf_norm(self, active_mpm_dof: ti.i32) -> ti.f64:
        result = 0.0
        iga_dof = ti.static(self.iga.degree_of_freedom)
        for dof in range(iga_dof):
            ti.atomic_max(result, ti.abs(self.iga.grid_disp[dof]))
        for dof in range(active_mpm_dof):
            ti.atomic_max(result, ti.abs(self.mpm.grid_disp[dof]))
        return result

    @ti.kernel
    def _clear_device_tangent_product(self, active_dof: ti.i32):
        for dof in self.monolithic_tangent_product:
            if dof < active_dof:
                self.monolithic_tangent_product[dof] = 0.0

    @ti.kernel
    def _accumulate_device_source_tangent_product(
        self,
        source: ti.template(),
        source_active_nodes: ti.i32,
        block_offset: ti.i32,
    ):
        """Accumulate an unreduced dense-block source matrix times ``p``."""
        for source_block in range(source_active_nodes):
            global_row_block = source_block + block_offset
            for row_component in ti.static(range(config.DIM)):
                value = 0.0
                for column_component in ti.static(range(config.DIM)):
                    entry = row_component * config.DIM + column_component
                    global_column = config.DIM * global_row_block + column_component
                    value += source.diag[source_block][entry] * self.monolithic_correction[global_column]
                ti.atomic_add(
                    self.monolithic_tangent_product[config.DIM * global_row_block + row_component],
                    value,
                )

        for raw_index in range(source.raw_non_diag_count[0]):
            source_row_block = source.non_diag.blockI[raw_index]
            source_column_block = source.non_diag.blockJ[raw_index]
            if 0 <= source_row_block < source_active_nodes and 0 <= source_column_block < source_active_nodes:
                global_row_block = source_row_block + block_offset
                global_column_block = source_column_block + block_offset
                for row_component in ti.static(range(config.DIM)):
                    value = 0.0
                    for column_component in ti.static(range(config.DIM)):
                        entry = row_component * config.DIM + column_component
                        global_column = config.DIM * global_column_block + column_component
                        value += source.non_diag.blockH[raw_index][entry] * self.monolithic_correction[global_column]
                    ti.atomic_add(
                        self.monolithic_tangent_product[config.DIM * global_row_block + row_component],
                        value,
                    )

    @ti.kernel
    def _device_physical_residual_squared(self, active_dof: ti.i32) -> ti.f64:
        result = 0.0
        for dof in range(active_dof):
            if self.monolithic_fixed[dof] == 0:
                value = self.monolithic_physical_rhs[dof]
                result += value * value
        return result

    def _assemble_device_physical_tangent_product(self, *, active_mpm_dof, include_friction):
        """Compute physical ``K p`` from source blocks before elimination."""
        active_mpm_dof = int(active_mpm_dof)
        iga_nodes = int(self.iga.degree_of_freedom) // config.DIM
        active_mpm_nodes = active_mpm_dof // config.DIM
        active_nodes = iga_nodes + active_mpm_nodes
        active_dof = config.DIM * active_nodes
        sources = (
            self.iga.hash_matrix,
            self.mpm.hash_matrix,
            self.barrier_hash_matrix,
            self.friction_hash_matrix if include_friction else None,
        )
        for source in sources:
            if source is not None and (source.symmetric or source.matrix_symmetric):
                raise RuntimeError("physical tangent product requires full dense source blocks")
        self._clear_device_tangent_product(active_dof)
        self._accumulate_device_source_tangent_product(self.iga.hash_matrix, iga_nodes, 0)
        self._accumulate_device_source_tangent_product(self.mpm.hash_matrix, active_mpm_nodes, iga_nodes)
        self._accumulate_device_source_tangent_product(self.barrier_hash_matrix, active_nodes, 0)
        if include_friction:
            self._accumulate_device_source_tangent_product(self.friction_hash_matrix, active_nodes, 0)
        return float(self._device_fully_implicit_merit_slope(active_dof))

    @ti.kernel
    def _device_minimum_contact_distance_kernel(self, expected_pairs: ti.i32) -> ti.f64:
        minimum = ti.math.inf
        self.contact_query_status[None] = 0
        for contact_id in range(expected_pairs):
            distance = self.contacts[contact_id].distance
            if not (distance >= 0.0 and distance < ti.math.inf):
                self.contact_query_status[None] = 2
            else:
                ti.atomic_min(minimum, distance)
        return minimum

    def minimum_contact_distance(self):
        """Minimum distance over all sample--NURBS pairs at the current query."""
        expected_pairs = int(self.mpm.total_surface_num) * int(self.contact_surface.num_surfaces)
        if expected_pairs == 0:
            return math.inf
        if expected_pairs > int(self.contacts.shape[0]):
            raise RuntimeError("IGA-MPM contact storage is smaller than the complete " "sample--surface pair set")
        minimum = float(self._device_minimum_contact_distance_kernel(expected_pairs))
        if int(self.contact_query_status[None]) != 0:
            raise RuntimeError("IGA-MPM IPC closest-point query produced a non-finite " "distance")
        return minimum

    @staticmethod
    def _probe_system_arrays(system):
        if isinstance(system, dict):
            if "correction" in system:
                return None, None, np.asarray(system["correction"], dtype=np.float64)
            return system["matrix"], np.asarray(system["rhs"], dtype=np.float64), None
        matrix, rhs = system
        return matrix, np.asarray(rhs, dtype=np.float64), None
