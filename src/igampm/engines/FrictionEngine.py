"""Friction engine responsibilities."""

import math

import numpy as np
import taichi as ti

import src.igampm.config as config
from src.igampm.engines.FullyImplicitFriction import (
    curve_geometry_jacobian,
    curve_negative_force_jacobian_block,
    curve_parameter_jacobian,
    curve_relative_velocity_jacobian,
    fully_implicit_resistance_state,
    resistance_input_jacobian,
    surface_geometry_jacobian,
    surface_negative_force_jacobian_block,
    surface_parameter_jacobian,
    surface_relative_velocity_jacobian,
)
from src.igampm.engines.TimeIntegration import newmark_endpoint_velocity_coefficients
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd
from src.physics_model.contact_model.ipc.NurbsContact import (
    get_distance_to_curve_fixed_dim,
    get_distance_to_surface_fixed_dim,
)


class FrictionEngineMixin:
    def _validate_fully_implicit_time_integration(self):
        """Require the BE/TR endpoint maps derived in the paper."""
        if not self._is_paper_endpoint_scheme(self.iga.integration):
            raise ValueError(
                "IGA fully implicit friction supports paper endpoint schemes "
                "Backward Euler [1, 0.5, 1] and trapezoidal rule "
                "[0.5, 0.25, 0.5]"
            )
        if not self._is_paper_endpoint_scheme(self.mpm.integration):
            raise ValueError(
                "MPM fully implicit friction supports paper endpoint schemes "
                "Backward Euler [1, 0.5, 1] and trapezoidal rule "
                "[0.5, 0.25, 0.5]"
            )
        iga_dt = float(self.iga.dt)
        mpm_dt = float(self.mpm.dt)
        if not np.isclose(iga_dt, mpm_dt, rtol=1.0e-12, atol=0.0):
            raise ValueError("fully implicit IGA-MPM friction requires one shared timestep")

    @ti.kernel
    def reset_friction_contacts(self):
        self.friction_contact_num[0] = 0
        for i in self.friction_contacts:
            self.friction_contacts[i].active = 0
            self.friction_contacts[i].surface_id = -1
            self.friction_contacts[i].sample_id = -1
            self.friction_contacts[i].particle_id = -1
            self.friction_contacts[i].mu_lambda = 0.0
            self.friction_contacts[i].normal = ti.Vector.zero(ti.f64, config.DIM)
            self.friction_contacts[i].knot_value = ti.Vector.zero(ti.f64, 2)

    @ti.kernel
    def count_friction_contacts(self) -> ti.i32:
        total = 0
        for i in self.friction_contacts:
            if self.friction_contacts[i].active:
                total += 1
        return total

    @ti.kernel
    def clear_friction_system(self):
        self.friction_nnz_count[0] = 0
        self.friction_nnz_overflow[0] = 0
        for i in self.friction_grad:
            self.friction_grad[i] = 0.0

    def prepare_friction_matrix_slots(self):
        if self.compact_contact_slots:
            self.mark_active_contact_slots(self.friction_contacts, self.friction_contact_slots)
            self.contact_slot_prefix.run(self.friction_contact_slots)
            self.prepare_contact_matrix_slots(self.friction_hash_matrix, self.friction_contact_slots)
        else:
            self.prepare_contact_matrix_slots(self.friction_hash_matrix, self.friction_contact_num)

    @ti.func
    def add_friction_block(self, contact_id, local_i, local_j, block_i, block_j, block):
        finite = 1
        for row in ti.static(range(config.DIM)):
            for column in ti.static(range(config.DIM)):
                value = block[row, column]
                if not (-ti.math.inf < value and value < ti.math.inf):
                    finite = 0
        if finite != 0:
            ti.atomic_add(self.friction_nnz_count[0], 1)
            stencil = ti.static(self.contact_stencil_capacity)
            matrix_contact_id = contact_id
            if ti.static(self.compact_contact_slots):
                matrix_contact_id = self.friction_contact_slots[contact_id] - 1
            slot = matrix_contact_id * ti.static(self.contact_pair_capacity) + local_i * stencil + local_j
            if (
                0 <= contact_id < ti.static(self.contact_pair_count)
                and 0 <= local_i < stencil
                and 0 <= local_j < stencil
                and 0 <= slot
                and slot < self.friction_hash_matrix.non_diag.blockI.shape[0]
                and block_i >= 0
                and block_j >= 0
            ):
                if block_i == block_j:
                    for row in ti.static(range(config.DIM)):
                        for column in ti.static(range(config.DIM)):
                            ti.atomic_add(
                                self.friction_hash_matrix.diag[block_i][row * config.DIM + column],
                                block[row, column],
                            )
                else:
                    self.friction_hash_matrix.non_diag.blockI[slot] = block_i
                    self.friction_hash_matrix.non_diag.blockJ[slot] = block_j
                    for row in ti.static(range(config.DIM)):
                        for column in ti.static(range(config.DIM)):
                            self.friction_hash_matrix.non_diag.blockH[slot][row * config.DIM + column] = block[
                                row, column
                            ]
            else:
                self.friction_hash_matrix.overflow[0] = 1
        else:
            if ti.static(self.friction_mode == "fully_implicit"):
                ti.atomic_max(self.fully_implicit_contact_status[None], 3)
            else:
                self.friction_nnz_overflow[0] = 1

    @ti.func
    def _fully_implicit_values_are_finite(self, vector, matrix):
        finite = 1
        for component in ti.static(range(config.DIM)):
            value = vector[component]
            if not (-ti.math.inf < value and value < ti.math.inf):
                finite = 0
            for column in ti.static(range(config.DIM)):
                value = matrix[component, column]
                if not (-ti.math.inf < value and value < ti.math.inf):
                    finite = 0
        return finite

    @ti.func
    def _fully_implicit_scalar_is_finite(self, value):
        return -ti.math.inf < value and value < ti.math.inf

    @ti.func
    def _fully_implicit_vector_is_finite(self, vector):
        finite = 1
        for component in ti.static(range(vector.n)):
            value = vector[component]
            if not (-ti.math.inf < value and value < ti.math.inf):
                finite = 0
        return finite

    @ti.func
    def _fully_implicit_matrix_is_finite(self, matrix):
        finite = 1
        for row in ti.static(range(matrix.n)):
            for column in ti.static(range(matrix.m)):
                value = matrix[row, column]
                if not (-ti.math.inf < value and value < ti.math.inf):
                    finite = 0
        return finite

    @ti.kernel
    def _prepare_fully_implicit_endpoint_velocity(
        self,
        active_mpm_nodes: ti.i32,
        iga_displacement_scale: ti.f64,
        iga_velocity_scale: ti.f64,
        iga_acceleration_scale: ti.f64,
        mpm_displacement_scale: ti.f64,
        mpm_velocity_scale: ti.f64,
        mpm_acceleration_scale: ti.f64,
    ):
        iga_nodes = ti.static(self.iga.degree_of_freedom // config.DIM)
        for coupled_node in self.fully_implicit_endpoint_velocity:
            velocity = ti.Vector.zero(ti.f64, config.DIM)
            if coupled_node < iga_nodes:
                for component in ti.static(range(config.DIM)):
                    velocity[component] = (
                        iga_displacement_scale * self.iga.grid_disp[config.DIM * coupled_node + component]
                        + iga_velocity_scale * self.iga.patch.velocitys[coupled_node][component]
                        + iga_acceleration_scale * self.iga.patch.accelerations[coupled_node][component]
                    )
            elif coupled_node < iga_nodes + active_mpm_nodes:
                compact_node = coupled_node - iga_nodes
                grid_node = self.mpm.dof2node[compact_node]
                for component in ti.static(range(config.DIM)):
                    velocity[component] = (
                        mpm_displacement_scale * self.mpm.grid_disp[config.DIM * compact_node + component]
                        + mpm_velocity_scale * self.mpm.grid.v[grid_node][component]
                        + mpm_acceleration_scale * self.mpm.grid.a[grid_node][component]
                    )
            self.fully_implicit_endpoint_velocity[coupled_node] = velocity

    @ti.kernel
    def scatter_friction_rhs(self, active_mpm_dof: ti.i32):
        for i in range(self.iga.degree_of_freedom):
            self.iga.rhs[i] += self.friction_grad[i]
        for i in range(active_mpm_dof):
            self.mpm.rhs[i] += self.friction_grad[self.iga.degree_of_freedom + i]

    @ti.kernel
    def initialize_surface_friction_contacts(
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
        for c in range(self.contacts.shape[0]):
            if self.contacts[c].active and self.contacts[c].surface_id == surface_id:
                sample_id = self.contacts[c].sample_id
                position = self.mpm.p_temp[sample_id]
                uknot = self.contacts[c].knot_value[0]
                vknot = self.contacts[c].knot_value[1]
                uknot, vknot, distance, pointer = get_distance_to_surface_fixed_dim(
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
                    surface,
                    surface_id,
                    ti.Vector([uknot, vknot]),
                )
                if ti.static(self.is_semi) or (
                    distance > self.barrier.dmin[0] + 1.0e-12 and distance < self.barrier.activation_distance_term()
                ):
                    mu_lambda = (
                        -self.friction.mu[0] * self._normal_terms(c, distance)[1] * self.mpm.surface_measure[sample_id]
                    )
                    if mu_lambda > 0.0:
                        self.friction_contacts[c].active = 1
                        self.friction_contacts[c].surface_id = surface_id
                        self.friction_contacts[c].sample_id = sample_id
                        self.friction_contacts[c].particle_id = self.contacts[c].particle_id
                        self.friction_contacts[c].mu_lambda = mu_lambda
                        self.friction_contacts[c].normal = pointer / distance
                        self.friction_contacts[c].knot_value = ti.Vector([uknot, vknot])

    @ti.kernel
    def initialize_curve_friction_contacts(
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
                position = self.mpm.p_temp[sample_id]
                uknot = self.contacts[c].knot_value[0]
                uknot, distance, pointer = get_distance_to_curve_fixed_dim(
                    prefix_num_knot_u,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    surface.knot_vector_u,
                    surface.control_points_hat,
                    surface.weights,
                    position,
                    basis,
                )
                if ti.static(self.is_semi) or (
                    distance > self.barrier.dmin[0] + 1.0e-12 and distance < self.barrier.activation_distance_term()
                ):
                    mu_lambda = (
                        -self.friction.mu[0] * self._normal_terms(c, distance)[1] * self.mpm.surface_measure[sample_id]
                    )
                    if mu_lambda > 0.0:
                        self.friction_contacts[c].active = 1
                        self.friction_contacts[c].surface_id = surface_id
                        self.friction_contacts[c].sample_id = sample_id
                        self.friction_contacts[c].particle_id = self.contacts[c].particle_id
                        self.friction_contacts[c].mu_lambda = mu_lambda
                        self.friction_contacts[c].normal = pointer / distance
                        self.friction_contacts[c].knot_value = ti.Vector([uknot, 0.0])

    @ti.kernel
    def assemble_friction_matrix_for_surface(
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
        for c in range(self.friction_contacts.shape[0]):
            if self.friction_contacts[c].active and self.friction_contacts[c].surface_id == surface_id:
                sample_id = self.friction_contacts[c].sample_id
                particle_id = self.friction_contacts[c].particle_id
                uknot = self.friction_contacts[c].knot_value[0]
                vknot = self.friction_contacts[c].knot_value[1]
                normal = self.friction_contacts[c].normal
                mu_lambda = self.friction_contacts[c].mu_lambda

                span_u, span_v, nshape = basis.NurbsBasisShape(
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

                iga_disp = ti.Vector.zero(ti.f64, config.DIM)
                for i in range(span_v - basis.basis_v.degree, span_v + 1):
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        ctrlpt_offset = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                            basis.basis_u.degree + 1
                        )
                        local_ctrlpt_id = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                        global_ctrlpt_id = surface.control_points_id[local_ctrlpt_id]
                        ctrl_disp = ti.Vector(
                            [
                                self.iga.grid_disp[config.DIM * global_ctrlpt_id + d]
                                for d in ti.static(range(config.DIM))
                            ]
                        )
                        iga_disp += nshape[ctrlpt_offset] * ctrl_disp

                mpm_disp = self.mpm.p_temp[sample_id] - self.mpm.particle[particle_id].x
                tangent = ti.Matrix.identity(ti.f64, config.DIM) - normal.outer_product(normal)
                rel_disp = mpm_disp - iga_disp
                vbar = tangent.transpose() @ rel_disp / self.mpm.dt
                vbarnorm = vbar.norm()
                friction_gradient = self.friction.grad_term(vbarnorm)
                friction_hessian = self.friction.hess_term(vbarnorm)
                grad_rel = mu_lambda * friction_gradient * tangent @ vbar
                inner_term = friction_gradient * ti.Matrix.identity(ti.f64, config.DIM)
                if vbarnorm != 0.0:
                    inner_term += friction_hessian / vbarnorm * vbar.outer_product(vbar)
                hess_rel = mu_lambda * tangent @ psd_project_nd(inner_term) @ tangent.transpose() / self.mpm.dt

                for i in range(span_v - basis.basis_v.degree, span_v + 1):
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        ctrlpt_offset_1 = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                            basis.basis_u.degree + 1
                        )
                        local_ctrlpt_id_1 = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                        global_ctrlpt_id_1 = surface.control_points_id[local_ctrlpt_id_1]
                        shape_i = nshape[ctrlpt_offset_1]
                        for d in ti.static(range(config.DIM)):
                            self.friction_grad[config.DIM * global_ctrlpt_id_1 + d] += shape_i * grad_rel[d]

                        for m in range(span_v - basis.basis_v.degree, span_v + 1):
                            for n in range(span_u - basis.basis_u.degree, span_u + 1):
                                ctrlpt_offset_2 = (n - span_u + basis.basis_u.degree) + (
                                    m - span_v + basis.basis_v.degree
                                ) * (basis.basis_u.degree + 1)
                                local_ctrlpt_id_2 = prefix_num_ctrlpts + n + m * num_ctrlpts_u
                                global_ctrlpt_id_2 = surface.control_points_id[local_ctrlpt_id_2]
                                shape_j = nshape[ctrlpt_offset_2]
                                block = shape_i * shape_j * hess_rel
                                self.add_friction_block(
                                    c,
                                    ctrlpt_offset_1,
                                    ctrlpt_offset_2,
                                    global_ctrlpt_id_1,
                                    global_ctrlpt_id_2,
                                    block,
                                )

                        for k in range(self.mpm.offset[particle_id]):
                            base_node = self.mpm.LnID[particle_id, k]
                            mpm_offset = self.mpm.node2dof[base_node] - 1
                            if mpm_offset < 0:
                                continue
                            shape_m = self.mpm.shape[particle_id, k]
                            block = -shape_i * shape_m * hess_rel
                            coupled_mpm_block = self.iga.degree_of_freedom // config.DIM + mpm_offset
                            self.add_friction_block(
                                c,
                                ctrlpt_offset_1,
                                ti.static(self.contact_ctrlpts_capacity) + k,
                                global_ctrlpt_id_1,
                                coupled_mpm_block,
                                block,
                            )
                            self.add_friction_block(
                                c,
                                ti.static(self.contact_ctrlpts_capacity) + k,
                                ctrlpt_offset_1,
                                coupled_mpm_block,
                                global_ctrlpt_id_1,
                                block.transpose(),
                            )

                for j in range(self.mpm.offset[particle_id]):
                    base_jnode = self.mpm.LnID[particle_id, j]
                    mpm_offset_j = self.mpm.node2dof[base_jnode] - 1
                    if mpm_offset_j < 0:
                        continue
                    shape_j = self.mpm.shape[particle_id, j]
                    for d in ti.static(range(config.DIM)):
                        self.friction_grad[self.iga.degree_of_freedom + config.DIM * mpm_offset_j + d] -= (
                            shape_j * grad_rel[d]
                        )
                    for k in range(self.mpm.offset[particle_id]):
                        base_knode = self.mpm.LnID[particle_id, k]
                        mpm_offset_k = self.mpm.node2dof[base_knode] - 1
                        if mpm_offset_k < 0:
                            continue
                        shape_k = self.mpm.shape[particle_id, k]
                        block = shape_j * shape_k * hess_rel
                        coupled_mpm_block_j = self.iga.degree_of_freedom // config.DIM + mpm_offset_j
                        coupled_mpm_block_k = self.iga.degree_of_freedom // config.DIM + mpm_offset_k
                        self.add_friction_block(
                            c,
                            ti.static(self.contact_ctrlpts_capacity) + j,
                            ti.static(self.contact_ctrlpts_capacity) + k,
                            coupled_mpm_block_j,
                            coupled_mpm_block_k,
                            block,
                        )

    @ti.kernel
    def assemble_friction_matrix_for_curve(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
    ):
        for c in range(self.friction_contacts.shape[0]):
            if self.friction_contacts[c].active and self.friction_contacts[c].surface_id == surface_id:
                sample_id = self.friction_contacts[c].sample_id
                particle_id = self.friction_contacts[c].particle_id
                uknot = self.friction_contacts[c].knot_value[0]
                normal = self.friction_contacts[c].normal
                mu_lambda = self.friction_contacts[c].mu_lambda

                span_u, nshape, derivative_u, tangent_u, curvature_uu = basis.NurbsBasisHessian(
                    prefix_num_knot_u,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    uknot,
                    surface.knot_vector_u,
                    surface.control_points_hat,
                    surface.weights,
                )

                iga_disp = ti.Vector.zero(ti.f64, config.DIM)
                for j in range(span_u - basis.basis_u.degree, span_u + 1):
                    ctrlpt_offset = j - span_u + basis.basis_u.degree
                    local_ctrlpt_id = prefix_num_ctrlpts + j
                    global_ctrlpt_id = surface.control_points_id[local_ctrlpt_id]
                    ctrl_disp = ti.Vector(
                        [self.iga.grid_disp[config.DIM * global_ctrlpt_id + d] for d in ti.static(range(config.DIM))]
                    )
                    iga_disp += nshape[ctrlpt_offset] * ctrl_disp

                mpm_disp = self.mpm.p_temp[sample_id] - self.mpm.particle[particle_id].x
                tangent = ti.Matrix.identity(ti.f64, config.DIM) - normal.outer_product(normal)
                rel_disp = mpm_disp - iga_disp
                vbar = tangent.transpose() @ rel_disp / self.mpm.dt
                vbarnorm = vbar.norm()
                friction_gradient = self.friction.grad_term(vbarnorm)
                friction_hessian = self.friction.hess_term(vbarnorm)
                grad_rel = mu_lambda * friction_gradient * tangent @ vbar
                inner_term = friction_gradient * ti.Matrix.identity(ti.f64, config.DIM)
                if vbarnorm != 0.0:
                    inner_term += friction_hessian / vbarnorm * vbar.outer_product(vbar)
                hess_rel = mu_lambda * tangent @ psd_project_nd(inner_term) @ tangent.transpose() / self.mpm.dt

                for j in range(span_u - basis.basis_u.degree, span_u + 1):
                    ctrlpt_offset_1 = j - span_u + basis.basis_u.degree
                    local_ctrlpt_id_1 = prefix_num_ctrlpts + j
                    global_ctrlpt_id_1 = surface.control_points_id[local_ctrlpt_id_1]
                    shape_i = nshape[ctrlpt_offset_1]
                    for d in ti.static(range(config.DIM)):
                        self.friction_grad[config.DIM * global_ctrlpt_id_1 + d] += shape_i * grad_rel[d]

                    for n in range(span_u - basis.basis_u.degree, span_u + 1):
                        ctrlpt_offset_2 = n - span_u + basis.basis_u.degree
                        local_ctrlpt_id_2 = prefix_num_ctrlpts + n
                        global_ctrlpt_id_2 = surface.control_points_id[local_ctrlpt_id_2]
                        shape_j = nshape[ctrlpt_offset_2]
                        block = shape_i * shape_j * hess_rel
                        self.add_friction_block(
                            c,
                            ctrlpt_offset_1,
                            ctrlpt_offset_2,
                            global_ctrlpt_id_1,
                            global_ctrlpt_id_2,
                            block,
                        )

                    for k in range(self.mpm.offset[particle_id]):
                        base_node = self.mpm.LnID[particle_id, k]
                        mpm_offset = self.mpm.node2dof[base_node] - 1
                        if mpm_offset < 0:
                            continue
                        shape_m = self.mpm.shape[particle_id, k]
                        block = -shape_i * shape_m * hess_rel
                        coupled_mpm_block = self.iga.degree_of_freedom // config.DIM + mpm_offset
                        self.add_friction_block(
                            c,
                            ctrlpt_offset_1,
                            ti.static(self.contact_ctrlpts_capacity) + k,
                            global_ctrlpt_id_1,
                            coupled_mpm_block,
                            block,
                        )
                        self.add_friction_block(
                            c,
                            ti.static(self.contact_ctrlpts_capacity) + k,
                            ctrlpt_offset_1,
                            coupled_mpm_block,
                            global_ctrlpt_id_1,
                            block.transpose(),
                        )

                for j in range(self.mpm.offset[particle_id]):
                    base_jnode = self.mpm.LnID[particle_id, j]
                    mpm_offset_j = self.mpm.node2dof[base_jnode] - 1
                    if mpm_offset_j < 0:
                        continue
                    shape_j = self.mpm.shape[particle_id, j]
                    for d in ti.static(range(config.DIM)):
                        self.friction_grad[self.iga.degree_of_freedom + config.DIM * mpm_offset_j + d] -= (
                            shape_j * grad_rel[d]
                        )
                    for k in range(self.mpm.offset[particle_id]):
                        base_knode = self.mpm.LnID[particle_id, k]
                        mpm_offset_k = self.mpm.node2dof[base_knode] - 1
                        if mpm_offset_k < 0:
                            continue
                        shape_k = self.mpm.shape[particle_id, k]
                        block = shape_j * shape_k * hess_rel
                        coupled_mpm_block_j = self.iga.degree_of_freedom // config.DIM + mpm_offset_j
                        coupled_mpm_block_k = self.iga.degree_of_freedom // config.DIM + mpm_offset_k
                        self.add_friction_block(
                            c,
                            ti.static(self.contact_ctrlpts_capacity) + j,
                            ti.static(self.contact_ctrlpts_capacity) + k,
                            coupled_mpm_block_j,
                            coupled_mpm_block_k,
                            block,
                        )

    @ti.kernel
    def assemble_fully_implicit_friction_for_curve(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        iga_endpoint_scale: ti.f64,
        mpm_endpoint_scale: ti.f64,
        surface: ti.template(),
        basis: ti.template(),
        need_matrix: ti.template(),
    ):
        """Exact curve-contact residual and complete nonsymmetric Jacobian."""
        iga_nodes = ti.static(self.iga.degree_of_freedom // config.DIM)
        for c in range(self.contacts.shape[0]):
            if self.contacts[c].active and self.contacts[c].surface_id == surface_id:
                sample_id = self.contacts[c].sample_id
                particle_id = self.contacts[c].particle_id
                uknot = self.contacts[c].knot_value[0]
                (
                    span_u,
                    nshape,
                    derivative_u,
                    tangent_u,
                    curvature_uu,
                ) = basis.NurbsBasisHessian(
                    prefix_num_knot_u,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    uknot,
                    surface.knot_vector_u,
                    surface.control_points_hat,
                    surface.weights,
                )

                surface_position = ti.Vector.zero(ti.f64, config.DIM)
                relative_velocity = ti.Vector.zero(ti.f64, config.DIM)
                surface_velocity_derivative_u = ti.Vector.zero(ti.f64, config.DIM)
                for j in range(span_u - basis.basis_u.degree, span_u + 1):
                    support = j - span_u + basis.basis_u.degree
                    local_control = prefix_num_ctrlpts + j
                    control_node = surface.control_points_id[local_control]
                    control_position = surface.control_points_hat[local_control]
                    control_velocity = self.fully_implicit_endpoint_velocity[control_node]
                    surface_position += nshape[support] * control_position
                    relative_velocity -= nshape[support] * control_velocity
                    surface_velocity_derivative_u += derivative_u[support] * control_velocity

                valid_stencil = 1
                for point_support in range(self.mpm.offset[particle_id]):
                    grid_node = self.mpm.LnID[particle_id, point_support]
                    compact_node = self.mpm.node2dof[grid_node] - 1
                    if compact_node < 0:
                        valid_stencil = 0
                    else:
                        point_weight = self.mpm.shape[particle_id, point_support]
                        relative_velocity += point_weight * (
                            self.fully_implicit_endpoint_velocity[iga_nodes + compact_node]
                        )
                if valid_stencil == 0:
                    ti.atomic_max(self.fully_implicit_contact_status[None], 4)
                    continue

                pointer = surface_position - self.mpm.p_temp[sample_id]
                distance = pointer.norm()
                if not (distance > self.barrier.dmin[0] and distance < ti.math.inf):
                    ti.atomic_max(self.fully_implicit_contact_status[None], 1)
                    continue
                normal = pointer / distance
                num_ctrlpts = num_knot_u - basis.basis_u.degree - 1
                lower_u = surface.knot_vector_u[prefix_num_knot_u + basis.basis_u.degree]
                upper_u = surface.knot_vector_u[prefix_num_knot_u + num_ctrlpts]
                free_u = 1
                if uknot <= lower_u + 1.0e-12 or uknot >= upper_u - 1.0e-12:
                    free_u = 0

                barrier_gradient = self.barrier.grad_term(distance)
                barrier_hessian = self.barrier.hess_term(distance)
                area = self.mpm.surface_measure[sample_id]
                normal_force = -area * barrier_gradient
                state_finite = (
                    self._fully_implicit_scalar_is_finite(area)
                    and self._fully_implicit_scalar_is_finite(barrier_gradient)
                    and self._fully_implicit_scalar_is_finite(normal_force)
                )
                if ti.static(need_matrix):
                    state_finite = state_finite and self._fully_implicit_scalar_is_finite(barrier_hessian)
                if state_finite == 0:
                    ti.atomic_max(self.fully_implicit_contact_status[None], 3)
                    continue
                if normal_force > 0.0:
                    (
                        projector,
                        tangential_velocity,
                        resistance,
                        velocity_jacobian,
                        factor_per_normal_force,
                    ) = fully_implicit_resistance_state(
                        relative_velocity,
                        normal,
                        normal_force,
                        self.friction.mu_dynamic[0],
                        self.friction.mu_static[0],
                        self.friction.mu_viscous[0],
                        self.friction.stribeck_velocity[0],
                        self.friction.epsv[0],
                        ti.static(self.friction.profile_id),
                    )
                    if (
                        self._fully_implicit_values_are_finite(resistance, velocity_jacobian) == 0
                        or self._fully_implicit_matrix_is_finite(projector) == 0
                        or self._fully_implicit_vector_is_finite(tangential_velocity) == 0
                        or self._fully_implicit_scalar_is_finite(factor_per_normal_force) == 0
                    ):
                        ti.atomic_max(self.fully_implicit_contact_status[None], 3)
                        continue
                    ti.atomic_add(self.friction_contact_num[0], 1)

                    # R = -B^T resistance.  Control blocks have B=-N I;
                    # point blocks have B=w I.
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        support = j - span_u + basis.basis_u.degree
                        local_control = prefix_num_ctrlpts + j
                        control_node = surface.control_points_id[local_control]
                        for component in ti.static(range(config.DIM)):
                            ti.atomic_add(
                                self.friction_grad[config.DIM * control_node + component],
                                nshape[support] * resistance[component],
                            )
                    for point_support in range(self.mpm.offset[particle_id]):
                        grid_node = self.mpm.LnID[particle_id, point_support]
                        compact_node = self.mpm.node2dof[grid_node] - 1
                        point_weight = self.mpm.shape[particle_id, point_support]
                        for component in ti.static(range(config.DIM)):
                            ti.atomic_add(
                                self.friction_grad[self.iga.degree_of_freedom + config.DIM * compact_node + component],
                                -point_weight * resistance[component],
                            )

                    if ti.static(need_matrix):
                        # Control-node block columns.
                        for input_j in range(span_u - basis.basis_u.degree, span_u + 1):
                            input_support = input_j - span_u + basis.basis_u.degree
                            input_local_control = prefix_num_ctrlpts + input_j
                            input_control_node = surface.control_points_id[input_local_control]
                            parameter_jacobian, valid_ift = curve_parameter_jacobian(
                                pointer,
                                tangent_u,
                                curvature_uu,
                                nshape[input_support],
                                derivative_u[input_support],
                                1,
                                free_u,
                            )
                            if valid_ift == 0:
                                ti.atomic_max(self.fully_implicit_contact_status[None], 2)
                            else:
                                geometry_jacobian = curve_geometry_jacobian(
                                    nshape[input_support],
                                    tangent_u,
                                    parameter_jacobian,
                                )
                                relative_velocity_jacobian = curve_relative_velocity_jacobian(
                                    -nshape[input_support],
                                    iga_endpoint_scale,
                                    surface_velocity_derivative_u,
                                    parameter_jacobian,
                                )
                                resistance_jacobian = resistance_input_jacobian(
                                    geometry_jacobian,
                                    relative_velocity_jacobian,
                                    relative_velocity,
                                    normal,
                                    distance,
                                    projector,
                                    tangential_velocity,
                                    velocity_jacobian,
                                    factor_per_normal_force,
                                    area,
                                    barrier_hessian,
                                )
                                if (
                                    self._fully_implicit_matrix_is_finite(geometry_jacobian) == 0
                                    or self._fully_implicit_matrix_is_finite(relative_velocity_jacobian) == 0
                                    or self._fully_implicit_matrix_is_finite(resistance_jacobian) == 0
                                ):
                                    ti.atomic_max(
                                        self.fully_implicit_contact_status[None],
                                        3,
                                    )
                                for output_j in range(
                                    span_u - basis.basis_u.degree,
                                    span_u + 1,
                                ):
                                    output_support = output_j - span_u + basis.basis_u.degree
                                    output_local_control = prefix_num_ctrlpts + output_j
                                    output_control_node = surface.control_points_id[output_local_control]
                                    block = curve_negative_force_jacobian_block(
                                        -nshape[output_support],
                                        1,
                                        derivative_u[output_support],
                                        parameter_jacobian,
                                        resistance,
                                        resistance_jacobian,
                                    )
                                    self.add_friction_block(
                                        c,
                                        output_support,
                                        input_support,
                                        output_control_node,
                                        input_control_node,
                                        block,
                                    )
                                for output_point in range(self.mpm.offset[particle_id]):
                                    output_grid = self.mpm.LnID[particle_id, output_point]
                                    output_compact = self.mpm.node2dof[output_grid] - 1
                                    output_weight = self.mpm.shape[particle_id, output_point]
                                    block = curve_negative_force_jacobian_block(
                                        output_weight,
                                        0,
                                        0.0,
                                        parameter_jacobian,
                                        resistance,
                                        resistance_jacobian,
                                    )
                                    self.add_friction_block(
                                        c,
                                        ti.static(self.contact_ctrlpts_capacity) + output_point,
                                        input_support,
                                        iga_nodes + output_compact,
                                        input_control_node,
                                        block,
                                    )

                        # MPM point-node block columns.
                        for input_point in range(self.mpm.offset[particle_id]):
                            input_grid = self.mpm.LnID[particle_id, input_point]
                            input_compact = self.mpm.node2dof[input_grid] - 1
                            input_weight = self.mpm.shape[particle_id, input_point]
                            parameter_jacobian, valid_ift = curve_parameter_jacobian(
                                pointer,
                                tangent_u,
                                curvature_uu,
                                -input_weight,
                                0.0,
                                0,
                                free_u,
                            )
                            if valid_ift == 0:
                                ti.atomic_max(self.fully_implicit_contact_status[None], 2)
                            else:
                                geometry_jacobian = curve_geometry_jacobian(
                                    -input_weight,
                                    tangent_u,
                                    parameter_jacobian,
                                )
                                relative_velocity_jacobian = curve_relative_velocity_jacobian(
                                    input_weight,
                                    mpm_endpoint_scale,
                                    surface_velocity_derivative_u,
                                    parameter_jacobian,
                                )
                                resistance_jacobian = resistance_input_jacobian(
                                    geometry_jacobian,
                                    relative_velocity_jacobian,
                                    relative_velocity,
                                    normal,
                                    distance,
                                    projector,
                                    tangential_velocity,
                                    velocity_jacobian,
                                    factor_per_normal_force,
                                    area,
                                    barrier_hessian,
                                )
                                if (
                                    self._fully_implicit_matrix_is_finite(geometry_jacobian) == 0
                                    or self._fully_implicit_matrix_is_finite(relative_velocity_jacobian) == 0
                                    or self._fully_implicit_matrix_is_finite(resistance_jacobian) == 0
                                ):
                                    ti.atomic_max(
                                        self.fully_implicit_contact_status[None],
                                        3,
                                    )
                                for output_j in range(
                                    span_u - basis.basis_u.degree,
                                    span_u + 1,
                                ):
                                    output_support = output_j - span_u + basis.basis_u.degree
                                    output_local_control = prefix_num_ctrlpts + output_j
                                    output_control_node = surface.control_points_id[output_local_control]
                                    block = curve_negative_force_jacobian_block(
                                        -nshape[output_support],
                                        1,
                                        derivative_u[output_support],
                                        parameter_jacobian,
                                        resistance,
                                        resistance_jacobian,
                                    )
                                    self.add_friction_block(
                                        c,
                                        output_support,
                                        ti.static(self.contact_ctrlpts_capacity) + input_point,
                                        output_control_node,
                                        iga_nodes + input_compact,
                                        block,
                                    )
                                for output_point in range(self.mpm.offset[particle_id]):
                                    output_grid = self.mpm.LnID[particle_id, output_point]
                                    output_compact = self.mpm.node2dof[output_grid] - 1
                                    output_weight = self.mpm.shape[particle_id, output_point]
                                    block = curve_negative_force_jacobian_block(
                                        output_weight,
                                        0,
                                        0.0,
                                        parameter_jacobian,
                                        resistance,
                                        resistance_jacobian,
                                    )
                                    self.add_friction_block(
                                        c,
                                        ti.static(self.contact_ctrlpts_capacity) + output_point,
                                        ti.static(self.contact_ctrlpts_capacity) + input_point,
                                        iga_nodes + output_compact,
                                        iga_nodes + input_compact,
                                        block,
                                    )

    @ti.kernel
    def assemble_fully_implicit_friction_for_surface(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_knot_v: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        num_knot_v: ti.i32,
        num_ctrlpts_u: ti.i32,
        iga_endpoint_scale: ti.f64,
        mpm_endpoint_scale: ti.f64,
        surface: ti.template(),
        basis: ti.template(),
        need_matrix: ti.template(),
    ):
        """Exact surface-contact residual and nonsymmetric IFT Jacobian."""
        iga_nodes = ti.static(self.iga.degree_of_freedom // config.DIM)
        for c in range(self.contacts.shape[0]):
            if self.contacts[c].active and self.contacts[c].surface_id == surface_id:
                sample_id = self.contacts[c].sample_id
                particle_id = self.contacts[c].particle_id
                uknot = self.contacts[c].knot_value[0]
                vknot = self.contacts[c].knot_value[1]
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

                surface_position = ti.Vector.zero(ti.f64, config.DIM)
                relative_velocity = ti.Vector.zero(ti.f64, config.DIM)
                surface_velocity_derivative_u = ti.Vector.zero(ti.f64, config.DIM)
                surface_velocity_derivative_v = ti.Vector.zero(ti.f64, config.DIM)
                for i in range(span_v - basis.basis_v.degree, span_v + 1):
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        support = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                            basis.basis_u.degree + 1
                        )
                        local_control = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                        control_node = surface.control_points_id[local_control]
                        control_position = surface.control_points_hat[local_control]
                        control_velocity = self.fully_implicit_endpoint_velocity[control_node]
                        surface_position += nshape[support] * control_position
                        relative_velocity -= nshape[support] * control_velocity
                        surface_velocity_derivative_u += derivative_u[support] * control_velocity
                        surface_velocity_derivative_v += derivative_v[support] * control_velocity

                valid_stencil = 1
                for point_support in range(self.mpm.offset[particle_id]):
                    grid_node = self.mpm.LnID[particle_id, point_support]
                    compact_node = self.mpm.node2dof[grid_node] - 1
                    if compact_node < 0:
                        valid_stencil = 0
                    else:
                        point_weight = self.mpm.shape[particle_id, point_support]
                        relative_velocity += point_weight * (
                            self.fully_implicit_endpoint_velocity[iga_nodes + compact_node]
                        )
                if valid_stencil == 0:
                    ti.atomic_max(self.fully_implicit_contact_status[None], 4)
                    continue

                pointer = surface_position - self.mpm.p_temp[sample_id]
                distance = pointer.norm()
                if not (distance > self.barrier.dmin[0] and distance < ti.math.inf):
                    ti.atomic_max(self.fully_implicit_contact_status[None], 1)
                    continue
                normal = pointer / distance
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

                barrier_gradient = self.barrier.grad_term(distance)
                barrier_hessian = self.barrier.hess_term(distance)
                area = self.mpm.surface_measure[sample_id]
                normal_force = -area * barrier_gradient
                state_finite = (
                    self._fully_implicit_scalar_is_finite(area)
                    and self._fully_implicit_scalar_is_finite(barrier_gradient)
                    and self._fully_implicit_scalar_is_finite(normal_force)
                )
                if ti.static(need_matrix):
                    state_finite = state_finite and self._fully_implicit_scalar_is_finite(barrier_hessian)
                if state_finite == 0:
                    ti.atomic_max(self.fully_implicit_contact_status[None], 3)
                    continue
                if normal_force > 0.0:
                    (
                        projector,
                        tangential_velocity,
                        resistance,
                        velocity_jacobian,
                        factor_per_normal_force,
                    ) = fully_implicit_resistance_state(
                        relative_velocity,
                        normal,
                        normal_force,
                        self.friction.mu_dynamic[0],
                        self.friction.mu_static[0],
                        self.friction.mu_viscous[0],
                        self.friction.stribeck_velocity[0],
                        self.friction.epsv[0],
                        ti.static(self.friction.profile_id),
                    )
                    if (
                        self._fully_implicit_values_are_finite(resistance, velocity_jacobian) == 0
                        or self._fully_implicit_matrix_is_finite(projector) == 0
                        or self._fully_implicit_vector_is_finite(tangential_velocity) == 0
                        or self._fully_implicit_scalar_is_finite(factor_per_normal_force) == 0
                    ):
                        ti.atomic_max(self.fully_implicit_contact_status[None], 3)
                        continue
                    ti.atomic_add(self.friction_contact_num[0], 1)

                    for i in range(span_v - basis.basis_v.degree, span_v + 1):
                        for j in range(span_u - basis.basis_u.degree, span_u + 1):
                            support = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                                basis.basis_u.degree + 1
                            )
                            local_control = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                            control_node = surface.control_points_id[local_control]
                            for component in ti.static(range(config.DIM)):
                                ti.atomic_add(
                                    self.friction_grad[config.DIM * control_node + component],
                                    nshape[support] * resistance[component],
                                )
                    for point_support in range(self.mpm.offset[particle_id]):
                        grid_node = self.mpm.LnID[particle_id, point_support]
                        compact_node = self.mpm.node2dof[grid_node] - 1
                        point_weight = self.mpm.shape[particle_id, point_support]
                        for component in ti.static(range(config.DIM)):
                            ti.atomic_add(
                                self.friction_grad[self.iga.degree_of_freedom + config.DIM * compact_node + component],
                                -point_weight * resistance[component],
                            )

                    if ti.static(need_matrix):
                        # NURBS control-point block columns.
                        for input_i in range(span_v - basis.basis_v.degree, span_v + 1):
                            for input_j in range(span_u - basis.basis_u.degree, span_u + 1):
                                input_support = (input_j - span_u + basis.basis_u.degree) + (
                                    input_i - span_v + basis.basis_v.degree
                                ) * (basis.basis_u.degree + 1)
                                input_local_control = prefix_num_ctrlpts + input_j + input_i * num_ctrlpts_u
                                input_control_node = surface.control_points_id[input_local_control]
                                parameter_jacobian, valid_ift = surface_parameter_jacobian(
                                    pointer,
                                    tangent_u,
                                    tangent_v,
                                    curvature_uu,
                                    curvature_vv,
                                    curvature_uv,
                                    nshape[input_support],
                                    derivative_u[input_support],
                                    derivative_v[input_support],
                                    1,
                                    free_u,
                                    free_v,
                                )
                                if valid_ift == 0:
                                    ti.atomic_max(
                                        self.fully_implicit_contact_status[None],
                                        2,
                                    )
                                else:
                                    geometry_jacobian = surface_geometry_jacobian(
                                        nshape[input_support],
                                        tangent_u,
                                        tangent_v,
                                        parameter_jacobian,
                                    )
                                    relative_velocity_jacobian = surface_relative_velocity_jacobian(
                                        -nshape[input_support],
                                        iga_endpoint_scale,
                                        surface_velocity_derivative_u,
                                        surface_velocity_derivative_v,
                                        parameter_jacobian,
                                    )
                                    resistance_jacobian = resistance_input_jacobian(
                                        geometry_jacobian,
                                        relative_velocity_jacobian,
                                        relative_velocity,
                                        normal,
                                        distance,
                                        projector,
                                        tangential_velocity,
                                        velocity_jacobian,
                                        factor_per_normal_force,
                                        area,
                                        barrier_hessian,
                                    )
                                    if (
                                        self._fully_implicit_matrix_is_finite(geometry_jacobian) == 0
                                        or self._fully_implicit_matrix_is_finite(relative_velocity_jacobian) == 0
                                        or self._fully_implicit_matrix_is_finite(resistance_jacobian) == 0
                                    ):
                                        ti.atomic_max(
                                            self.fully_implicit_contact_status[None],
                                            3,
                                        )
                                    for output_i in range(
                                        span_v - basis.basis_v.degree,
                                        span_v + 1,
                                    ):
                                        for output_j in range(
                                            span_u - basis.basis_u.degree,
                                            span_u + 1,
                                        ):
                                            output_support = (output_j - span_u + basis.basis_u.degree) + (
                                                output_i - span_v + basis.basis_v.degree
                                            ) * (basis.basis_u.degree + 1)
                                            output_local_control = (
                                                prefix_num_ctrlpts + output_j + output_i * num_ctrlpts_u
                                            )
                                            output_control_node = surface.control_points_id[output_local_control]
                                            block = surface_negative_force_jacobian_block(
                                                -nshape[output_support],
                                                1,
                                                derivative_u[output_support],
                                                derivative_v[output_support],
                                                parameter_jacobian,
                                                resistance,
                                                resistance_jacobian,
                                            )
                                            self.add_friction_block(
                                                c,
                                                output_support,
                                                input_support,
                                                output_control_node,
                                                input_control_node,
                                                block,
                                            )
                                    for output_point in range(self.mpm.offset[particle_id]):
                                        output_grid = self.mpm.LnID[particle_id, output_point]
                                        output_compact = self.mpm.node2dof[output_grid] - 1
                                        output_weight = self.mpm.shape[particle_id, output_point]
                                        block = surface_negative_force_jacobian_block(
                                            output_weight,
                                            0,
                                            0.0,
                                            0.0,
                                            parameter_jacobian,
                                            resistance,
                                            resistance_jacobian,
                                        )
                                        self.add_friction_block(
                                            c,
                                            ti.static(self.contact_ctrlpts_capacity) + output_point,
                                            input_support,
                                            iga_nodes + output_compact,
                                            input_control_node,
                                            block,
                                        )

                        # MPM point-node block columns.
                        for input_point in range(self.mpm.offset[particle_id]):
                            input_grid = self.mpm.LnID[particle_id, input_point]
                            input_compact = self.mpm.node2dof[input_grid] - 1
                            input_weight = self.mpm.shape[particle_id, input_point]
                            parameter_jacobian, valid_ift = surface_parameter_jacobian(
                                pointer,
                                tangent_u,
                                tangent_v,
                                curvature_uu,
                                curvature_vv,
                                curvature_uv,
                                -input_weight,
                                0.0,
                                0.0,
                                0,
                                free_u,
                                free_v,
                            )
                            if valid_ift == 0:
                                ti.atomic_max(self.fully_implicit_contact_status[None], 2)
                            else:
                                geometry_jacobian = surface_geometry_jacobian(
                                    -input_weight,
                                    tangent_u,
                                    tangent_v,
                                    parameter_jacobian,
                                )
                                relative_velocity_jacobian = surface_relative_velocity_jacobian(
                                    input_weight,
                                    mpm_endpoint_scale,
                                    surface_velocity_derivative_u,
                                    surface_velocity_derivative_v,
                                    parameter_jacobian,
                                )
                                resistance_jacobian = resistance_input_jacobian(
                                    geometry_jacobian,
                                    relative_velocity_jacobian,
                                    relative_velocity,
                                    normal,
                                    distance,
                                    projector,
                                    tangential_velocity,
                                    velocity_jacobian,
                                    factor_per_normal_force,
                                    area,
                                    barrier_hessian,
                                )
                                if (
                                    self._fully_implicit_matrix_is_finite(geometry_jacobian) == 0
                                    or self._fully_implicit_matrix_is_finite(relative_velocity_jacobian) == 0
                                    or self._fully_implicit_matrix_is_finite(resistance_jacobian) == 0
                                ):
                                    ti.atomic_max(
                                        self.fully_implicit_contact_status[None],
                                        3,
                                    )
                                for output_i in range(
                                    span_v - basis.basis_v.degree,
                                    span_v + 1,
                                ):
                                    for output_j in range(
                                        span_u - basis.basis_u.degree,
                                        span_u + 1,
                                    ):
                                        output_support = (output_j - span_u + basis.basis_u.degree) + (
                                            output_i - span_v + basis.basis_v.degree
                                        ) * (basis.basis_u.degree + 1)
                                        output_local_control = prefix_num_ctrlpts + output_j + output_i * num_ctrlpts_u
                                        output_control_node = surface.control_points_id[output_local_control]
                                        block = surface_negative_force_jacobian_block(
                                            -nshape[output_support],
                                            1,
                                            derivative_u[output_support],
                                            derivative_v[output_support],
                                            parameter_jacobian,
                                            resistance,
                                            resistance_jacobian,
                                        )
                                        self.add_friction_block(
                                            c,
                                            output_support,
                                            ti.static(self.contact_ctrlpts_capacity) + input_point,
                                            output_control_node,
                                            iga_nodes + input_compact,
                                            block,
                                        )
                                for output_point in range(self.mpm.offset[particle_id]):
                                    output_grid = self.mpm.LnID[particle_id, output_point]
                                    output_compact = self.mpm.node2dof[output_grid] - 1
                                    output_weight = self.mpm.shape[particle_id, output_point]
                                    block = surface_negative_force_jacobian_block(
                                        output_weight,
                                        0,
                                        0.0,
                                        0.0,
                                        parameter_jacobian,
                                        resistance,
                                        resistance_jacobian,
                                    )
                                    self.add_friction_block(
                                        c,
                                        ti.static(self.contact_ctrlpts_capacity) + output_point,
                                        ti.static(self.contact_ctrlpts_capacity) + input_point,
                                        iga_nodes + output_compact,
                                        iga_nodes + input_compact,
                                        block,
                                    )

    def add_friction_rhs_to_subsystems(self):
        self.scatter_friction_rhs(self.mpm.active_dof)

    def friction_matrix(self):
        self.friction_hash_matrix.finalize_taichi_assembly()
        return self.friction_hash_matrix.to_scipy(self.total_degree_of_freedom // config.DIM).tocsr()

    def _fully_implicit_device_available(self):
        """Whether the paper-exact point--NURBS law can stay in Taichi."""
        assemble_type = getattr(self, "assemble_type", "HashTriplet")
        return bool(
            self.friction_mode == "fully_implicit"
            and hasattr(self.iga, "hash_matrix")
            and hasattr(self.mpm, "hash_matrix")
            and self.monolithic_tangent_product is not None
            and (
                (
                    assemble_type == "HashTriplet"
                    and self.monolithic_hash_matrix is not None
                    and self.monolithic_hash_matrix.solver == "BiCGSTAB"
                    and not self.monolithic_hash_matrix.symmetric
                    and not self.monolithic_hash_matrix.matrix_symmetric
                )
                or (assemble_type == "COO" and self.monolithic_coo_matrix is not None)
            )
        )

    def assemble_fully_implicit_friction_system_taichi(self, need_matrix=True):
        """Assemble the exact fully implicit law without host arrays.

        The caller must first run the device barrier initializer at the same
        nonlinear state (the monolithic assembler guarantees this) so the
        stored closest coordinates are current.  The Taichi kernels then differentiate those
        coordinates through the closest-point stationarity equations, along
        with the normal, normal force, Stribeck law, and endpoint velocities.
        No symmetrization or PSD projection is applied to the resulting
        generally nonsymmetric Jacobian.
        """
        need_matrix = bool(need_matrix)
        if need_matrix:
            self.friction_hash_matrix.reset_system()
            self.prepare_friction_matrix_slots()
        self.clear_friction_system()
        self.reset_friction_contacts()
        self.fully_implicit_contact_status[None] = 0

        if not self.activate_fric or self.curr_barrier_contact_num == 0:
            self.curr_friction_contact_num = 0
            return

        active_mpm_dof = int(self.mpm.active_dof)
        if active_mpm_dof < 0 or active_mpm_dof % config.DIM != 0:
            raise RuntimeError(
                "active MPM degrees of freedom must be a non-negative " "multiple of the spatial dimension"
            )
        iga_dq, iga_vn, iga_an = newmark_endpoint_velocity_coefficients(self.iga.integration, self.iga.dt)
        mpm_dq, mpm_vn, mpm_an = newmark_endpoint_velocity_coefficients(self.mpm.integration, self.mpm.dt)
        coefficients = (
            iga_dq,
            iga_vn,
            iga_an,
            mpm_dq,
            mpm_vn,
            mpm_an,
        )
        if not np.all(np.isfinite(coefficients)):
            raise RuntimeError("fully implicit endpoint-velocity coefficients are non-finite")
        self._prepare_fully_implicit_endpoint_velocity(
            active_mpm_dof // config.DIM,
            float(iga_dq),
            float(iga_vn),
            float(iga_an),
            float(mpm_dq),
            float(mpm_vn),
            float(mpm_an),
        )

        for surface_id in range(self.contact_surface.num_surfaces):
            if config.DIM == 2:
                self.assemble_fully_implicit_friction_for_curve(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    float(iga_dq),
                    float(mpm_dq),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                    need_matrix,
                )
            else:
                self.assemble_fully_implicit_friction_for_surface(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_knot_v[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    int(self.contact_surface.num_knot_v[surface_id + 1]),
                    int(self.contact_surface.num_ctrlpts_u[surface_id + 1]),
                    float(iga_dq),
                    float(mpm_dq),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                    need_matrix,
                )

        status = int(self.fully_implicit_contact_status[None])
        if status != 0:
            messages = {
                1: "contact distance is non-finite or not strictly feasible",
                2: "closest-coordinate IFT matrix is singular",
                3: "friction resistance or exact Jacobian is non-finite",
                4: "contact particle references an inactive MPM grid node",
            }
            raise RuntimeError(
                "fully implicit point-NURBS Taichi assembly failed: "
                + messages.get(status, f"unknown device status {status}")
            )
        if need_matrix and (int(self.friction_nnz_overflow[0]) != 0 or int(self.friction_hash_matrix.overflow[0]) != 0):
            raise RuntimeError(
                "IGA-MPM fully implicit friction block buffer overflow: "
                f"used {int(self.friction_nnz_count[0])}, capacity "
                f"{self.friction_nnz_capacity}. Increase friction_nnz."
            )
        self.curr_friction_contact_num = int(self.friction_contact_num[0])

    def friction_blocks(self, active_mpm_dof=None):
        active_mpm_dof = self.mpm.active_dof if active_mpm_dof is None else int(active_mpm_dof)
        iga_dof = self.iga.degree_of_freedom
        mpm_begin = iga_dof
        mpm_end = iga_dof + active_mpm_dof
        matrix = self.friction_matrix()
        return {
            "K_ii": matrix[:iga_dof, :iga_dof],
            "K_im": matrix[:iga_dof, mpm_begin:mpm_end],
            "K_mi": matrix[mpm_begin:mpm_end, :iga_dof],
            "K_mm": matrix[mpm_begin:mpm_end, mpm_begin:mpm_end],
        }

    def friction_contact_forces(self, active_mpm_dof=None):
        active_mpm_dof = self.mpm.active_dof if active_mpm_dof is None else int(active_mpm_dof)
        grad = self.friction_grad.to_numpy()
        iga_dof = self.iga.degree_of_freedom
        return {
            "iga": grad[:iga_dof].copy(),
            "mpm": grad[iga_dof : iga_dof + active_mpm_dof].copy(),
        }

    @ti.kernel
    def friction_potential_energy_for_curve(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
        iga_grid_disp: ti.template(),
    ) -> ti.f64:
        energy = 0.0
        for c in range(self.friction_contacts.shape[0]):
            if self.friction_contacts[c].active and self.friction_contacts[c].surface_id == surface_id:
                sample_id = self.friction_contacts[c].sample_id
                particle_id = self.friction_contacts[c].particle_id
                uknot = self.friction_contacts[c].knot_value[0]
                normal = self.friction_contacts[c].normal
                (
                    span_u,
                    nshape,
                    derivative_u,
                    tangent_u,
                    curvature_uu,
                ) = basis.NurbsBasisHessian(
                    prefix_num_knot_u,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    uknot,
                    surface.knot_vector_u,
                    surface.control_points_hat,
                    surface.weights,
                )
                iga_disp = ti.Vector.zero(ti.f64, config.DIM)
                for j in range(span_u - basis.basis_u.degree, span_u + 1):
                    offset = j - span_u + basis.basis_u.degree
                    local_id = prefix_num_ctrlpts + j
                    global_id = surface.control_points_id[local_id]
                    ctrl_disp = ti.Vector(
                        [iga_grid_disp[config.DIM * global_id + d] for d in ti.static(range(config.DIM))]
                    )
                    iga_disp += nshape[offset] * ctrl_disp
                mpm_disp = self.mpm.p_temp[sample_id] - self.mpm.particle[particle_id].x
                energy += self.friction.energy(
                    mpm_disp - iga_disp,
                    normal,
                    self.friction_contacts[c].mu_lambda,
                    self.mpm.dt,
                )
        return energy

    @ti.kernel
    def friction_potential_energy_for_surface(
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
        iga_grid_disp: ti.template(),
    ) -> ti.f64:
        energy = 0.0
        for c in range(self.friction_contacts.shape[0]):
            if self.friction_contacts[c].active and self.friction_contacts[c].surface_id == surface_id:
                sample_id = self.friction_contacts[c].sample_id
                particle_id = self.friction_contacts[c].particle_id
                knot = self.friction_contacts[c].knot_value
                normal = self.friction_contacts[c].normal
                span_u, span_v, nshape = basis.NurbsBasisShape(
                    prefix_num_knot_u,
                    prefix_num_knot_v,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    num_knot_v,
                    knot[0],
                    knot[1],
                    surface.knot_vector_u,
                    surface.knot_vector_v,
                    surface.weights,
                )
                iga_disp = ti.Vector.zero(ti.f64, config.DIM)
                for i in range(span_v - basis.basis_v.degree, span_v + 1):
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        offset = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                            basis.basis_u.degree + 1
                        )
                        local_id = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                        global_id = surface.control_points_id[local_id]
                        ctrl_disp = ti.Vector(
                            [iga_grid_disp[config.DIM * global_id + d] for d in ti.static(range(config.DIM))]
                        )
                        iga_disp += nshape[offset] * ctrl_disp
                mpm_disp = self.mpm.p_temp[sample_id] - self.mpm.particle[particle_id].x
                energy += self.friction.energy(
                    mpm_disp - iga_disp,
                    normal,
                    self.friction_contacts[c].mu_lambda,
                    self.mpm.dt,
                )
        return energy

    def friction_potential_energy(self, iga_grid_disp=None):
        """Return the conservative potential for the frozen lagged cache."""
        if not self.activate_fric or self.curr_friction_contact_num == 0:
            return 0.0
        if iga_grid_disp is None:
            iga_grid_disp = self.iga.grid_disp
        energy = 0.0
        for surface_id in range(self.contact_surface.num_surfaces):
            if config.DIM == 2:
                energy += self.friction_potential_energy_for_curve(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                    iga_grid_disp,
                )
            else:
                energy += self.friction_potential_energy_for_surface(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_knot_v[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    int(self.contact_surface.num_knot_v[surface_id + 1]),
                    int(self.contact_surface.num_ctrlpts_u[surface_id + 1]),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                    iga_grid_disp,
                )
        return float(energy)

    @ti.kernel
    def _build_device_fully_implicit_velocity_predictor(
        self,
        active_mpm_nodes: ti.i32,
        iga_dq: ti.f64,
        iga_vn: ti.f64,
        iga_an: ti.f64,
        mpm_dq: ti.f64,
        mpm_vn: ti.f64,
        mpm_an: ti.f64,
    ):
        iga_nodes = ti.static(self.iga.degree_of_freedom // config.DIM)
        active_nodes = iga_nodes + active_mpm_nodes
        for coupled_node in range(active_nodes):
            for component in ti.static(range(config.DIM)):
                dof = config.DIM * coupled_node + component
                target = 0.0
                if coupled_node < iga_nodes:
                    target = (
                        (1.0 - iga_vn) * self.iga.patch.velocitys[coupled_node][component]
                        - iga_an * self.iga.patch.accelerations[coupled_node][component]
                    ) / iga_dq
                else:
                    compact_node = coupled_node - iga_nodes
                    grid_node = self.mpm.dof2node[compact_node]
                    target = (
                        (1.0 - mpm_vn) * self.mpm.grid.v[grid_node][component]
                        - mpm_an * self.mpm.grid.a[grid_node][component]
                    ) / mpm_dq
                if self.monolithic_fixed[dof] != 0:
                    target = self.monolithic_fixed_correction[dof]
                self.monolithic_correction[dof] = target

    @ti.kernel
    def _device_fully_implicit_merit_slope(self, active_dof: ti.i32) -> ti.f64:
        result = 0.0
        for dof in range(active_dof):
            if self.monolithic_fixed[dof] == 0:
                result -= self.monolithic_physical_rhs[dof] * self.monolithic_tangent_product[dof]
        return result

    def _initialize_fully_implicit_velocity_guess_device(self):
        """Apply the paper's endpoint-velocity predictor on the device."""
        if not self.fully_implicit_velocity_predictor:
            return
        active_mpm_dof = int(self.mpm.active_dof)
        if float(self._device_current_displacement_inf_norm(active_mpm_dof)) > 0.0:
            return
        iga_dq, iga_vn, iga_an = newmark_endpoint_velocity_coefficients(self.iga.integration, self.iga.dt)
        mpm_dq, mpm_vn, mpm_an = newmark_endpoint_velocity_coefficients(self.mpm.integration, self.mpm.dt)
        coefficients = (
            iga_dq,
            iga_vn,
            iga_an,
            mpm_dq,
            mpm_vn,
            mpm_an,
        )
        if not all(math.isfinite(value) for value in coefficients) or min(iga_dq, mpm_dq) <= 0.0:
            raise RuntimeError("fully implicit velocity predictor has invalid Newmark " "coefficients")
        self._prepare_device_monolithic_vectors(active_mpm_dof, False)
        active_mpm_nodes = active_mpm_dof // config.DIM
        if self.iga.dirichlet.num > 0:
            self._load_device_iga_dirichlet()
        if self.mpm.dirichlet.num > 0:
            self._load_device_mpm_dirichlet(active_mpm_nodes)
        self._build_device_fully_implicit_velocity_predictor(
            active_mpm_nodes,
            float(iga_dq),
            float(iga_vn),
            float(iga_an),
            float(mpm_dq),
            float(mpm_vn),
            float(mpm_an),
        )
        self._split_device_monolithic_correction(active_mpm_dof)
        alpha = self._material_feasible_step_device()
        alpha = self.conservative_contact_step_device(max_step=alpha)
        if alpha < self.contact_ccd_min_step:
            raise RuntimeError("fully implicit velocity predictor has no feasible " "IPC/material step")
        self._set_device_trial_displacements(alpha)
        self._accept_device_trial_displacements()
        self._synchronize_device_trial_state_with_accepted()

    def fully_implicit_residual_armijo_device(
        self,
        current_residual,
        merit_slope,
        *,
        initial_step=1.0,
        include_friction=True,
        verbose=False,
    ):
        """Device-resident residual-merit Armijo for a nonconservative system."""
        current_residual = float(current_residual)
        merit_slope = float(merit_slope)
        if not math.isfinite(current_residual) or current_residual < 0.0:
            raise ValueError("current residual norm must be finite and non-negative")
        if not math.isfinite(merit_slope) or merit_slope >= 0.0:
            raise RuntimeError("Newton correction is not a descent direction for 0.5 * ||R||^2")

        current_merit = 0.5 * current_residual * current_residual
        # ACCD supplies the first conservative upper bound.  Trial-state
        # validation belongs to the Armijo loop below, where a failed closest
        # query or residual assembly can safely reduce alpha instead of
        # aborting the whole implicit step before backtracking starts.
        alpha = self._conservative_contact_step_device_impl(
            float(initial_step),
            None,
            False,
            prepare_contacts=True,
        )
        # ``grid_disp_temp`` is the immutable base during the line search;
        # trial states are evaluated in the current fields because the body
        # and exact endpoint-velocity kernels consume those fields directly.
        self._sync_device_trial_displacements()
        accepted = False
        accepted_system = None
        trial_residual = math.inf
        trial_failure = None
        backtracks = 0
        try:
            for backtracks in range(self.armijo_max_backtracks):
                if alpha < self.contact_ccd_min_step:
                    break
                self._set_device_current_from_trial_base(alpha)
                try:
                    candidate = self.assemble_monolithic_newton_system(
                        include_friction=bool(include_friction),
                        need_matrix=False,
                    )
                    residual_squared = float(self._device_physical_residual_squared(candidate["active_dof"]))
                    trial_residual = math.sqrt(residual_squared)
                except (FloatingPointError, RuntimeError, ValueError) as exception:
                    candidate = None
                    trial_residual = math.inf
                    trial_failure = f"{type(exception).__name__}: {exception}"
                trial_merit = 0.5 * trial_residual * trial_residual
                bound = current_merit + self.armijo_c1 * alpha * merit_slope
                if math.isfinite(trial_merit) and trial_merit <= bound:
                    accepted = True
                    accepted_system = candidate
                    break
                alpha *= self.fully_implicit_armijo_reduction
                if verbose:
                    failure = "" if trial_failure is None else f", failure={trial_failure}"
                    print(
                        "IGA-MPM Taichi residual-merit Armijo "
                        f"{backtracks + 1}: alpha={alpha:.6e}, "
                        f"residual={trial_residual:.6e}{failure}"
                    )
        finally:
            if accepted:
                self._sync_device_trial_displacements()
            else:
                self._restore_device_current_from_trial_base()
            self.initialize_barrier(self.mpm.grid_disp, self.iga.grid_disp)

        self.last_armijo_step = float(alpha if accepted else 0.0)
        self.last_armijo_backtracks = int(backtracks)
        return {
            "accepted": accepted,
            "step": self.last_armijo_step,
            "backtracks": self.last_armijo_backtracks,
            "residual": float(trial_residual),
            "failure": trial_failure,
            "merit_slope": merit_slope,
            "system": accepted_system,
            "minimum_distance": self.minimum_contact_distance(),
            "backend": "taichi_device_residual_armijo",
        }

    def _solve_fully_implicit_newton_device(
        self,
        *,
        include_friction,
        max_iterations,
        absolute_tolerance,
        dirichlet_tolerance,
        verbose,
        linear_solve=None,
    ):
        """Exact nonsymmetric FI Newton solve with device-resident vectors."""
        self._save_device_monolithic_entry_displacements()
        self.last_monolithic_iterations = 0
        self.last_monolithic_residual = math.inf
        self.last_monolithic_dirichlet_residual = math.inf
        self.last_monolithic_converged = False
        initial_residual = None
        last_system = None
        last_armijo = None
        try:
            self._initialize_fully_implicit_velocity_guess_device()
            for iteration in range(max_iterations):
                last_system = self.assemble_monolithic_newton_system(
                    include_friction=bool(include_friction),
                    need_matrix=True,
                )
                residual_squared = float(self._device_physical_residual_squared(last_system["active_dof"]))
                residual = math.sqrt(residual_squared)
                dirichlet_residual = float(self._device_dirichlet_residual(last_system["active_dof"]))
                if not (math.isfinite(residual) and math.isfinite(dirichlet_residual)):
                    raise RuntimeError("fully implicit IGA-MPM Taichi residual is non-finite")
                if initial_residual is None:
                    initial_residual = residual
                target = max(
                    absolute_tolerance,
                    self.fully_implicit_force_rtol * initial_residual,
                )
                self.last_monolithic_residual = float(residual)
                self.last_monolithic_dirichlet_residual = dirichlet_residual
                if residual <= target and dirichlet_residual <= dirichlet_tolerance:
                    self.last_monolithic_converged = True
                    break

                solve_result = self._solve_monolithic_linear_system(last_system, linear_solve=linear_solve)
                if not solve_result["converged"]:
                    raise RuntimeError(
                        "IGA-MPM Taichi fully implicit BiCGSTAB did not "
                        "converge: residual="
                        f"{solve_result['residual']:.6e}, iterations="
                        f"{solve_result['iterations']}"
                    )
                self._split_device_monolithic_correction(last_system["active_mpm_dof"])
                merit_slope = self._assemble_device_physical_tangent_product(
                    active_mpm_dof=last_system["active_mpm_dof"],
                    include_friction=bool(include_friction),
                )
                if not math.isfinite(merit_slope) or merit_slope >= 0.0:
                    raise RuntimeError(
                        "fully implicit IGA-MPM Taichi Newton correction is "
                        "not a descent direction for 0.5 * ||R||^2"
                    )
                initial_step = self._material_feasible_step_device()
                last_armijo = self.fully_implicit_residual_armijo_device(
                    residual,
                    merit_slope,
                    initial_step=initial_step,
                    include_friction=bool(include_friction),
                    verbose=verbose,
                )
                self.last_monolithic_iterations = iteration + 1
                last_system["linear_solve"] = solve_result
                if not last_armijo["accepted"]:
                    break
                if verbose:
                    print(
                        "IGA-MPM Taichi fully implicit Newton "
                        f"{iteration + 1}: residual={residual:.6e}, "
                        f"alpha={last_armijo['step']:.6e}"
                    )

            if not self.last_monolithic_converged and last_armijo is not None and last_armijo["accepted"]:
                self.last_monolithic_residual = float(last_armijo["residual"])
                candidate = last_armijo["system"]
                self.last_monolithic_dirichlet_residual = float(
                    self._device_dirichlet_residual(candidate["active_dof"])
                )
                target = max(
                    absolute_tolerance,
                    self.fully_implicit_force_rtol * initial_residual,
                )
                self.last_monolithic_converged = (
                    self.last_monolithic_residual <= target
                    and self.last_monolithic_dirichlet_residual <= dirichlet_tolerance
                )

            if not self.last_monolithic_converged:
                reason = (
                    "residual_line_search"
                    if last_armijo is not None and not last_armijo["accepted"]
                    else "maximum_newton_iterations"
                )
                raise RuntimeError(
                    "IGA-MPM fully implicit Taichi friction Newton solve did "
                    "not converge; the trial displacement was rolled back "
                    f"(reason={reason}, residual="
                    f"{self.last_monolithic_residual:.6e}, "
                    "dirichlet_residual="
                    f"{self.last_monolithic_dirichlet_residual:.6e})"
                )
        except BaseException:
            self._restore_device_monolithic_entry_displacements()
            self.initialize_barrier(self.mpm.grid_disp, self.iga.grid_disp)
            raise

        return {
            "iterations": self.last_monolithic_iterations,
            "residual": self.last_monolithic_residual,
            "dirichlet_residual": self.last_monolithic_dirichlet_residual,
            "initial_residual": float(math.inf if initial_residual is None else initial_residual),
            "converged": self.last_monolithic_converged,
            "system": last_system,
            "line_search": last_armijo,
            "friction_mode": "fully_implicit",
            "backend": "taichi_device_exact_fi_bicgstab",
        }

    def solve_fully_implicit_friction_newton(
        self,
        include_friction=True,
        max_iterations=None,
        tolerance=None,
        linear_solve=None,
        verbose=False,
    ):
        """Solve the paper's nonconservative residual with its full Jacobian."""
        if self.friction_mode != "fully_implicit":
            raise RuntimeError("fully implicit Newton requires friction_mode='fully_implicit'")
        max_iterations = self.monolithic_max_iterations if max_iterations is None else int(max_iterations)
        if max_iterations <= 0:
            raise ValueError("fully implicit Newton max_iterations must be positive")
        absolute_tolerance = self.fully_implicit_force_atol if tolerance is None else float(tolerance)
        if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
            raise ValueError("fully implicit Newton tolerance must be finite and non-negative")
        dirichlet_tolerance = float(getattr(self, "fully_implicit_dirichlet_atol", 1.0e-12))
        if not np.isfinite(dirichlet_tolerance) or dirichlet_tolerance < 0.0:
            raise ValueError("fully implicit Dirichlet tolerance must be finite and " "non-negative")
        if self._device_monolithic_available(bool(include_friction)):
            return self._solve_fully_implicit_newton_device(
                include_friction=bool(include_friction),
                max_iterations=max_iterations,
                absolute_tolerance=absolute_tolerance,
                dirichlet_tolerance=dirichlet_tolerance,
                verbose=verbose,
                linear_solve=linear_solve,
            )
        raise RuntimeError(
            "fully implicit IGA-MPM requires device COO or HashTriplet "
            "assembly; selecting a SciPy/custom linear solve changes only "
            "the finalized linear system boundary and does not enable a "
            "NumPy residual, contact, or line-search backend"
        )

    def initialize_friction(self, grid_disp=None):
        if not self.activate_fric:
            self.curr_friction_contact_num = 0
            return
        self.initialize_barrier(grid_disp)
        if self.friction_mode == "fully_implicit":
            # No cache is frozen. Assembly consumes the current barrier query
            # and differentiates its closest coordinates and normal force.
            self.reset_friction_contacts()
            self.curr_friction_contact_num = self.curr_barrier_contact_num
        else:
            self._initialize_friction_from_current_barrier()

    def _initialize_friction_from_current_barrier(self):
        """Freeze normal, closest coordinates, and normal force once."""
        self.reset_friction_contacts()
        if self.curr_barrier_contact_num == 0:
            self.curr_friction_contact_num = 0
            return
        for surface_id in range(self.contact_surface.num_surfaces):
            if config.DIM == 2:
                self.initialize_curve_friction_contacts(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                )
            else:
                self.initialize_surface_friction_contacts(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_knot_v[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    int(self.contact_surface.num_knot_v[surface_id + 1]),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                )
        self.curr_friction_contact_num = int(self.count_friction_contacts())

    def refresh_lagged_friction_cache(self, grid_disp=None):
        """Refresh the active set and lagged data after a complete inner solve."""
        self.initialize_barrier(grid_disp)
        if self.activate_fric:
            self._initialize_friction_from_current_barrier()
        else:
            self.curr_friction_contact_num = 0

    def friction_outer_iteration_limit(self):
        if self.friction_iterations == -1:
            return self.friction_max_iterations
        return self.friction_iterations

    def _host_lagged_probe_velocity(self, correction):
        """Return the mixed IGA/MPM correction norm in velocity units."""
        correction = np.asarray(correction, dtype=np.float64).reshape(-1)
        iga_dof = min(
            max(int(getattr(self.iga, "degree_of_freedom", 0)), 0),
            correction.size,
        )
        iga_dt = float(getattr(self.iga, "dt", 1.0))
        mpm_dt = float(getattr(self.mpm, "dt", 1.0))
        if not np.isfinite(iga_dt) or iga_dt <= 0.0 or not np.isfinite(mpm_dt) or mpm_dt <= 0.0:
            raise RuntimeError("IGA-MPM lagged friction convergence requires finite positive " "subsystem timesteps")
        result = 0.0
        if iga_dof > 0:
            result = max(
                result,
                float(np.linalg.norm(correction[:iga_dof], np.inf)) / iga_dt,
            )
        if iga_dof < correction.size:
            result = max(
                result,
                float(np.linalg.norm(correction[iga_dof:], np.inf)) / mpm_dt,
            )
        return result

    def solve_lagged_friction_fixed_point(
        self,
        inner_solve=None,
        assemble_updated_system=None,
        grid_disp=None,
        probe_solve=None,
        tolerance=None,
        include_friction=None,
        linear_solve=None,
        energy_function=None,
        newton_max_iterations=None,
        newton_tolerance=None,
        verbose=False,
    ):
        """Run the lagged-friction outer fixed-point sequence.

        ``inner_solve(engine, outer_iteration)`` must fully solve the current
        conservative problem while the friction cache remains frozen and
        update ``iga.grid_disp``/``mpm.grid_disp``.  After it returns, this
        method refreshes the active set, closest coordinates, normals, and
        normal-force weights.  ``assemble_updated_system(engine)`` then
        returns either ``{"matrix", "rhs"}``, ``(matrix, rhs)``, or an
        already-computed ``{"correction"}``.  The resulting correction is a
        convergence probe only and is never applied to either subsystem.  Its
        infinity norm is divided by the corresponding subsystem timestep, so
        ``tolerance`` has velocity units just like the inner Newton tolerance.

        If both callbacks are omitted, the built-in monolithic Newton path
        assembles the existing IGA and direct implicit-MPM body backends,
        contact blocks, coupled essential constraints, conservative contact
        step, and contact-aware Armijo search.  Supplying callbacks remains
        supported for custom subsystem formulations.
        """
        use_builtin = inner_solve is None and assemble_updated_system is None
        device_builtin_probe = False
        if (inner_solve is None) != (assemble_updated_system is None):
            raise TypeError(
                "inner_solve and assemble_updated_system must either both be " "provided or both be omitted"
            )
        if use_builtin:
            if include_friction is None:
                include_friction = self.activate_fric
            device_builtin_probe = self._device_monolithic_available(include_friction)

            def inner_solve(current, outer_iteration):
                return current.solve_monolithic_newton(
                    outer_iteration=outer_iteration,
                    include_friction=include_friction,
                    max_iterations=newton_max_iterations,
                    tolerance=newton_tolerance,
                    linear_solve=linear_solve,
                    energy_function=energy_function,
                    verbose=verbose,
                )

            def assemble_updated_system(current):
                if device_builtin_probe:
                    return current.assemble_monolithic_newton_system(include_friction=include_friction)
                return current.assemble_monolithic_newton_system(include_friction=include_friction)

        if not callable(inner_solve) or not callable(assemble_updated_system):
            raise TypeError(
                "IGA-MPM lagged fixed-point solve requires callable " "inner_solve and assemble_updated_system"
            )
        if self.friction_mode != "lagged":
            raise RuntimeError("IGA-MPM fixed-point driver requires lagged friction")

        tolerance = self.friction_tolerance if tolerance is None else float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("friction fixed-point tolerance must be finite and non-negative")
        if grid_disp is None:
            grid_disp = self.mpm.grid_disp

        if use_builtin and not device_builtin_probe:
            raise RuntimeError(
                "built-in lagged IGA-MPM friction requires device COO or "
                "HashTriplet assembly and the device monolithic solver; no "
                "NumPy/SciPy fallback is performed"
            )
        if device_builtin_probe:
            self._save_device_monolithic_entry_displacements()
            iga_entry_displacement = None
            mpm_entry_displacement = None
        else:
            iga_entry_displacement = self.iga.grid_disp.to_numpy().copy()
            mpm_entry_displacement = self.mpm.grid_disp.to_numpy().copy()
        self.last_friction_iterations = 0
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        last_inner_result = None
        try:
            self.refresh_lagged_friction_cache(grid_disp)

            for outer_iteration in range(self.friction_outer_iteration_limit()):
                # The cache is not touched anywhere inside this callback.
                last_inner_result = inner_solve(self, outer_iteration)
                self.last_friction_iterations = outer_iteration + 1
                if use_builtin and (
                    not isinstance(last_inner_result, dict) or not bool(last_inner_result.get("converged", False))
                ):
                    residual = (
                        last_inner_result.get("residual", np.inf) if isinstance(last_inner_result, dict) else np.inf
                    )
                    raise RuntimeError(
                        "IGA-MPM frozen-friction Newton solve did not converge "
                        f"(outer iteration {outer_iteration + 1}, "
                        f"residual={float(residual):.6e})"
                    )

                self.refresh_lagged_friction_cache(grid_disp)
                updated_system = assemble_updated_system(self)
                if device_builtin_probe:
                    probe_result = self._solve_monolithic_linear_system(
                        updated_system,
                        linear_solve=(probe_solve if probe_solve is not None else linear_solve),
                    )
                    if not probe_result["converged"]:
                        raise RuntimeError(
                            "IGA-MPM Taichi updated-friction PCG probe did "
                            "not converge: "
                            f"residual={probe_result['residual']:.6e}, "
                            f"iterations={probe_result['iterations']}"
                        )
                    if "active_mpm_dof" in updated_system:
                        self.last_friction_residual = float(
                            self._device_monolithic_correction_residual(
                                int(updated_system["active_mpm_dof"]),
                                float(self.iga.dt),
                                float(self.mpm.dt),
                            )
                        )
                    else:
                        # Lightweight orchestration tests/custom matrix
                        # adapters may expose only the aggregate norm.
                        dt = min(
                            float(getattr(self.iga, "dt", 1.0)),
                            float(getattr(self.mpm, "dt", 1.0)),
                        )
                        self.last_friction_residual = float(probe_result["solution_inf_norm"]) / dt
                else:
                    matrix, rhs, correction = self._probe_system_arrays(updated_system)
                    if correction is None:
                        solve = probe_solve
                        if solve is None:
                            solve = linear_solve
                        if solve is None:
                            raise RuntimeError(
                                "a custom host convergence probe must supply " "probe_solve or linear_solve explicitly"
                            )
                        correction = np.asarray(solve(matrix, rhs), dtype=np.float64)
                    correction = np.asarray(correction, dtype=np.float64).reshape(-1)
                    if not np.all(np.isfinite(correction)):
                        raise RuntimeError(
                            "IGA-MPM updated-friction convergence probe " "returned a non-finite correction"
                        )
                    self._load_external_correction(correction)
                    self.last_friction_residual = self._host_lagged_probe_velocity(correction)
                if verbose:
                    print(
                        "IGA-MPM friction fixed-point "
                        f"{self.last_friction_iterations}: "
                        f"probe={self.last_friction_residual:.6e}"
                    )
                if self.last_friction_residual <= tolerance:
                    self.last_friction_converged = True
                    break

            if self.friction_iterations == -1 and not self.last_friction_converged:
                raise RuntimeError(
                    "IGA-MPM friction fixed-point iteration exhausted its "
                    f"safety cap ({self.friction_max_iterations}) without "
                    f"convergence (residual={self.last_friction_residual:.6e})"
                )
        except BaseException:
            if device_builtin_probe:
                self._restore_device_monolithic_entry_displacements()
            else:
                self._restore_accepted_displacements(iga_entry_displacement, mpm_entry_displacement)
            raise

        return {
            "inner_result": last_inner_result,
            "iterations": self.last_friction_iterations,
            "residual": self.last_friction_residual,
            "converged": self.last_friction_converged,
            "approximate": bool(self.friction_iterations > 0 and not self.last_friction_converged),
        }

    def assemble_friction_system(self, need_matrix=True):
        need_matrix = bool(need_matrix)
        if self.friction_mode == "fully_implicit":
            if self._fully_implicit_device_available():
                self.assemble_fully_implicit_friction_system_taichi(need_matrix=need_matrix)
            else:
                raise RuntimeError(
                    "fully implicit IGA-MPM friction requires device COO or "
                    "HashTriplet assembly and the nonsymmetric Jacobian path; "
                    "the runtime does not fall back to NumPy contact assembly"
                )
            return
        if need_matrix:
            self.friction_hash_matrix.reset_system()
            self.prepare_friction_matrix_slots()
        self.clear_friction_system()
        if not need_matrix:
            raise RuntimeError("residual-only friction assembly is supported only for " "fully implicit friction")
        if not self.activate_fric or self.curr_friction_contact_num == 0:
            return
        for surface_id in range(self.contact_surface.num_surfaces):
            if config.DIM == 2:
                self.assemble_friction_matrix_for_curve(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                )
            else:
                self.assemble_friction_matrix_for_surface(
                    surface_id,
                    int(self.contact_surface.prefix_num_knot_u[surface_id]),
                    int(self.contact_surface.prefix_num_knot_v[surface_id]),
                    int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                    int(self.contact_surface.num_knot_u[surface_id + 1]),
                    int(self.contact_surface.num_knot_v[surface_id + 1]),
                    int(self.contact_surface.num_ctrlpts_u[surface_id + 1]),
                    self.contact_surface,
                    self.contact_surface.basis[surface_id],
                )
        if int(self.friction_nnz_overflow[0]) != 0 or int(self.friction_hash_matrix.overflow[0]) != 0:
            raise RuntimeError(
                f"IGA-MPM friction block buffer overflow: used {int(self.friction_nnz_count[0])}, "
                f"capacity {self.friction_nnz_capacity}. Increase friction_nnz."
            )
