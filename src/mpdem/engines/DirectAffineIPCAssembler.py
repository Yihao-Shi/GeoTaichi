"""Direct-MPM point contact pulled back to AffineBody controls."""

from types import SimpleNamespace

import taichi as ti

from src.contact_detection.Capacity import contact_pair_capacity
from src.fempm.contact.IPCAssembler import FEMPMIPCAssembler
from src.physics_model.contact_model.ipc.ContactDistance import point_triangle_distance_grad_hess
from src.physics_model.contact_model.ipc.Utilities import (
    ipc_barrier_distance2_offset_terms,
    ipc_friction_f1_over_speed,
    ipc_friction_hessian_term,
    ipc_barrier_force_magnitude_gradient,
    ipc_lagged_friction_cache_vjp,
)


@ti.data_oriented
class DirectAffineIPCAssembler(FEMPMIPCAssembler):
    """Reuse FEM--MPM IPC geometry with ABD's four-control basis."""

    def __init__(self, affine, mpm, model, simulation):
        if int(getattr(mpm, "dimension", 3)) != 3:
            raise ValueError("Direct MPM--ABD IPC currently requires 3D")
        self.affine = affine
        self.affine_control_count = int(affine.control_num)
        mesh = SimpleNamespace(
            number_of_nodes=int(affine.vertex_num),
            points=affine.rest_x_np,
            node_body_ids=affine.node2body_np,
        )
        search = str(getattr(simulation, "contact_search", "BVH"))
        if search.replace("_", "").replace("-", "").lower() not in {
            "bvh",
            "linkedcell",
        }:
            search = str(getattr(simulation, "search", "BVH"))
        contact_simulation = SimpleNamespace(
            search=search,
            max_point_triangle_pairs=contact_pair_capacity(
                len(affine.faces_np), simulation.contact_coordination_number
            ),
            max_point_edge_pairs=1,
        )
        super().__init__(
            SimpleNamespace(
                mesh=mesh,
                state=SimpleNamespace(position=affine.x),
            ),
            mpm,
            affine.faces_np,
            affine.face2body_np,
            model,
            contact_simulation,
        )
        self.adjoint_friction_count = ti.field(ti.i32, shape=())
        self.adjoint_friction_valid = ti.field(ti.i32, shape=())
        self.adjoint_friction_candidate = ti.Vector.field(
            self.stencil_size,
            ti.i32,
            shape=self.friction_capacity,
        )
        self.adjoint_friction_weight = ti.Vector.field(
            self.stencil_size,
            self.real_type,
            shape=self.friction_capacity,
        )
        self.adjoint_friction_normal = ti.Vector.field(3, self.real_type, shape=self.friction_capacity)
        self.adjoint_friction_normal_force = ti.field(self.real_type, shape=self.friction_capacity)

    @ti.kernel
    def _backup_lagged_friction_for_adjoint(self, count: ti.i32):
        self.adjoint_friction_count[None] = count
        self.adjoint_friction_valid[None] = 1
        for contact in range(count):
            self.adjoint_friction_candidate[contact] = self.friction_candidate[contact]
            self.adjoint_friction_weight[contact] = self.friction_weight[contact]
            self.adjoint_friction_normal[contact] = self.friction_normal[contact]
            self.adjoint_friction_normal_force[contact] = self.friction_normal_force[contact]

    @ti.kernel
    def _restore_lagged_friction_for_adjoint(self):
        count = self.adjoint_friction_count[None]
        for contact in range(count):
            self.friction_candidate[contact] = self.adjoint_friction_candidate[contact]
            self.friction_weight[contact] = self.adjoint_friction_weight[contact]
            self.friction_normal[contact] = self.adjoint_friction_normal[contact]
            self.friction_normal_force[contact] = self.adjoint_friction_normal_force[contact]

    def backup_lagged_friction_for_adjoint_device(self):
        if self.activate_friction:
            self._backup_lagged_friction_for_adjoint(int(self.friction_count))

    def restore_lagged_friction_for_adjoint_device(self):
        if self.activate_friction and int(self.adjoint_friction_valid[None]):
            self._restore_lagged_friction_for_adjoint()
            self.friction_count = int(self.adjoint_friction_count[None])

    def begin_step(self, fem_position, mpm_displacement, timestep):
        self.adjoint_friction_valid[None] = 0
        return super().begin_step(fem_position, mpm_displacement, timestep)

    @ti.kernel
    def copy_friction_hat(self):
        for node in range(self.combined_node_count):
            self.friction_hat[node] = self.positions[node]
            if ti.static(self.affine.use_newmark):
                if self.affine.initial_force_evaluation[None] != 0:
                    if node < self.surface_count:
                        particle = self.mpm.surface_id[node]
                        velocity = ti.Vector.zero(self.real_type, 3)
                        for local in range(self.mpm.offset[particle]):
                            grid = self.mpm.LnID[particle, local]
                            velocity += self.mpm.shape[particle, local] * self.mpm.grid[grid].v
                        self.friction_hat[node] -= self.affine.dt_device[None] * velocity
                    else:
                        self.friction_hat[node] = self.affine.friction_reference_vertex(node - self.surface_count)

    @ti.kernel
    def build_end_positions(
        self,
        affine_position: ti.template(),
        affine_direction: ti.template(),
        mpm_displacement: ti.template(),
        mpm_direction: ti.template(),
    ):
        for sample in range(self.surface_count):
            particle = self.mpm.surface_id[sample]
            value = ti.Vector.zero(self.real_type, 3)
            for component in ti.static(range(3)):
                value[component] = self.mpm.particle[particle].x[component]
            for local in range(self.mpm.offset[particle]):
                grid = self.mpm.LnID[particle, local]
                block = self.mpm.node2dof[grid] - 1
                if block >= 0:
                    for component in ti.static(range(3)):
                        value[component] += self.mpm.shape[particle, local] * (
                            mpm_displacement[3 * block + component] + mpm_direction[3 * block + component]
                        )
            self.end_positions[sample] = value
        for vertex in range(self.fem_node_count):
            body = self.affine.node2body[vertex]
            direction = ti.Vector.zero(self.real_type, 3)
            for control in ti.static(range(4)):
                direction += self.affine.basis[vertex, control] * affine_direction[4 * body + control]
            self.end_positions[self.surface_count + vertex] = affine_position[vertex] + direction

    @ti.func
    def _scatter_gradient(self, stencil, particle, gradient, rhs):
        scale = self.affine.scale_device[None]
        for site in ti.static(range(1, 4)):
            vertex = stencil[site] - ti.static(self.surface_count)
            body = self.affine.node2body[vertex]
            for local_control in ti.static(range(4)):
                weight = self.affine.basis[vertex, local_control]
                block = 4 * body + local_control
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        rhs[3 * block + component],
                        -scale * weight * gradient[3 * site + component],
                    )
        point_gradient = ti.Vector([gradient[component] for component in ti.static(range(3))])
        for local in range(self.mpm.offset[particle]):
            grid = self.mpm.LnID[particle, local]
            block = self.mpm.node2dof[grid] - 1
            if block >= 0:
                weight = self.mpm.shape[particle, local]
                coupled = ti.static(self.affine_control_count) + block
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        rhs[3 * coupled + component],
                        -scale * weight * point_gradient[component],
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
        scale = self.affine.scale_device[None]
        if first_site > 0 and second_site > 0:
            first_vertex = stencil[first_site] - ti.static(self.surface_count)
            second_vertex = stencil[second_site] - ti.static(self.surface_count)
            first_body = self.affine.node2body[first_vertex]
            second_body = self.affine.node2body[second_vertex]
            for first_control, second_control in ti.static(ti.ndrange(4, 4)):
                matrix.add_block_entry(
                    4 * first_body + first_control,
                    4 * second_body + second_control,
                    scale
                    * self.affine.basis[first_vertex, first_control]
                    * self.affine.basis[second_vertex, second_control]
                    * hessian,
                )
        elif first_site == 0 and second_site > 0:
            vertex = stencil[second_site] - ti.static(self.surface_count)
            body = self.affine.node2body[vertex]
            for local in range(self.mpm.offset[particle]):
                grid = self.mpm.LnID[particle, local]
                block = self.mpm.node2dof[grid] - 1
                if block >= 0:
                    for control in ti.static(range(4)):
                        matrix.add_block_entry(
                            ti.static(self.affine_control_count) + block,
                            4 * body + control,
                            scale * self.mpm.shape[particle, local] * self.affine.basis[vertex, control] * hessian,
                        )
        elif first_site > 0 and second_site == 0:
            vertex = stencil[first_site] - ti.static(self.surface_count)
            body = self.affine.node2body[vertex]
            for local in range(self.mpm.offset[particle]):
                grid = self.mpm.LnID[particle, local]
                block = self.mpm.node2dof[grid] - 1
                if block >= 0:
                    for control in ti.static(range(4)):
                        matrix.add_block_entry(
                            4 * body + control,
                            ti.static(self.affine_control_count) + block,
                            scale * self.mpm.shape[particle, local] * self.affine.basis[vertex, control] * hessian,
                        )
        else:
            for first_local in range(self.mpm.offset[particle]):
                for second_local in range(self.mpm.offset[particle]):
                    first_grid = self.mpm.LnID[particle, first_local]
                    second_grid = self.mpm.LnID[particle, second_local]
                    first_block = self.mpm.node2dof[first_grid] - 1
                    second_block = self.mpm.node2dof[second_grid] - 1
                    if first_block >= 0 and second_block >= 0:
                        matrix.add_block_entry(
                            ti.static(self.affine_control_count) + first_block,
                            ti.static(self.affine_control_count) + second_block,
                            scale
                            * self.mpm.shape[particle, first_local]
                            * self.mpm.shape[particle, second_local]
                            * hessian,
                        )

    def _raw_blocks_per_contact(self):
        support = int(self.mpm.shape_func.max_node_per_particle)
        # The three triangle vertices all map to the same four controls.
        # All 3x3 site combinations on each equal control bypass raw storage.
        return (support + 3 * 4) ** 2 - support - 4 * 3**2

    @ti.func
    def _contact_adjoint(self, stencil, particle):
        adjoint = ti.Vector.zero(self.real_type, 12)
        for local in range(self.mpm.offset[particle]):
            node = self.mpm.LnID[particle, local]
            block = self.mpm.node2dof[node] - 1
            if block >= 0:
                for component in ti.static(range(3)):
                    adjoint[component] += (
                        self.mpm.shape[particle, local]
                        * self.mpm.incre_resolution[3 * block + component]
                        / self.affine.scale_device[None]
                    )
        for site in ti.static(range(1, 4)):
            vertex = stencil[site] - ti.static(self.surface_count)
            body = self.affine.node2body[vertex]
            for control, component in ti.static(ti.ndrange(4, 3)):
                adjoint[3 * site + component] += (
                    self.affine.basis[vertex, control] * self.affine.linear_x[12 * body + 3 * control + component]
                )
        return adjoint

    @ti.func
    def _scatter_particle_position_vjp(self, ipc, particle, point_gradient, response, include_position):
        displacement_gradient = ti.Matrix.zero(self.real_type, 3, 3)
        adjoint_gradient = ti.Matrix.zero(self.real_type, 3, 3)
        for local in range(self.mpm.offset[particle]):
            node = self.mpm.LnID[particle, local]
            block = self.mpm.node2dof[node] - 1
            if block >= 0:
                displacement = ti.Vector.zero(self.real_type, 3)
                adjoint = ti.Vector.zero(self.real_type, 3)
                for component in ti.static(range(3)):
                    displacement[component] = self.mpm.grid_disp[3 * block + component]
                    adjoint[component] = (
                        self.mpm.incre_resolution[3 * block + component] / self.affine.scale_device[None]
                    )
                gradient = self.mpm.dshape[particle, local]
                displacement_gradient += displacement.outer_product(gradient)
                adjoint_gradient += adjoint.outer_product(gradient)
        value = -self.affine.scale_device[None] * (
            adjoint_gradient.transpose() @ point_gradient + displacement_gradient.transpose() @ response
        )
        if include_position:
            value -= self.affine.scale_device[None] * response
        for component in ti.static(range(3)):
            ti.atomic_add(ipc.particle_position_vjp[particle][component], value[component])

    @ti.kernel
    def pullback_barrier_input_state(self, ipc: ti.template(), count: ti.i32, candidates: ti.template()):
        for contact in range(count):
            stencil = candidates[contact]
            sample, particle, mpm_body, affine_body = self._pair_indices(stencil)
            if self._pair_is_active(mpm_body, affine_body):
                distance2, gradient, hessian, _ = point_triangle_distance_grad_hess(
                    self.positions[stencil[0]],
                    self.positions[stencil[1]],
                    self.positions[stencil[2]],
                    self.positions[stencil[3]],
                )
                dmin = self.pair_dmin[mpm_body, affine_body]
                dhat = self.pair_dhat[mpm_body, affine_body]
                active_gap2 = (2.0 * dmin + dhat) * dhat
                if distance2 - dmin * dmin < active_gap2:
                    kappa = self.pair_kappa[mpm_body, affine_body] / active_gap2**2
                    _, first, second = ipc_barrier_distance2_offset_terms(distance2, dhat, dmin, kappa, 0)
                    measure = self.mpm.surface_measure[sample] * dhat
                    local_adjoint = self._contact_adjoint(stencil, particle)
                    response = measure * (
                        first * (hessian.transpose() @ local_adjoint) + second * gradient * gradient.dot(local_adjoint)
                    )
                    self._scatter_particle_position_vjp(
                        ipc,
                        particle,
                        measure * first * ti.Vector([gradient[c] for c in ti.static(range(3))]),
                        ti.Vector([response[c] for c in ti.static(range(3))]),
                        True,
                    )

    @ti.kernel
    def pullback_friction_input_state(self, ipc: ti.template(), count: ti.i32, coefficient_vjp: ti.template()):
        # Direct reference/transfer partial; cache geometry is pulled back separately.
        for contact in range(count):
            stencil = self.friction_candidate[contact]
            _, particle, mpm_body, affine_body = self._pair_indices(stencil)
            normal = self.friction_normal[contact]
            tangent = ti.Matrix.identity(self.real_type, 3) - normal.outer_product(normal)
            relative = ti.Vector.zero(self.real_type, 3)
            relative_adjoint = ti.Vector.zero(self.real_type, 3)
            local_adjoint = self._contact_adjoint(stencil, particle)
            for site in ti.static(range(4)):
                weight = self.friction_weight[contact][site]
                relative += weight * (self.positions[stencil[site]] - self.friction_hat[stencil[site]])
                for component in ti.static(range(3)):
                    relative_adjoint[component] += weight * local_adjoint[3 * site + component]
            dt = self.friction_dt[None]
            velocity = tangent @ relative / dt
            speed = velocity.norm()
            epsv = self.pair_epsv[mpm_body, affine_body]
            normal_force = self.friction_normal_force[contact]
            coefficient = self.pair_friction[mpm_body, affine_body] * normal_force
            profile = ipc_friction_f1_over_speed(speed, epsv)
            force = coefficient * profile * (tangent @ velocity)
            inner = coefficient * profile * ti.Matrix.identity(self.real_type, 3)
            if speed > 0.0:
                inner += coefficient * ipc_friction_hessian_term(speed, epsv) / speed * velocity.outer_product(velocity)
            response = tangent @ inner @ tangent @ relative_adjoint / dt
            self._scatter_particle_position_vjp(
                ipc,
                particle,
                self.friction_weight[contact][0] * force,
                self.friction_weight[contact][0] * response,
                False,
            )
            for site in ti.static(range(1, 4)):
                vertex = stencil[site] - ti.static(self.surface_count)
                body = self.affine.node2body[vertex]
                for control, component in ti.static(ti.ndrange(4, 3)):
                    ti.atomic_add(
                        self.affine.state_y_vjp[4 * body + control][component],
                        self.affine.scale_device[None]
                        * self.friction_weight[contact][site]
                        * self.affine.basis[vertex, control]
                        * response[component],
                    )
            ti.atomic_add(
                coefficient_vjp[mpm_body, affine_body],
                -self.affine.scale_device[None] * normal_force * profile * relative_adjoint.dot(tangent @ velocity),
            )

    @ti.kernel
    def pullback_friction_cache(
        self,
        ipc: ti.template(),
        count: ti.i32,
        cache_positions: ti.template(),
        displacement_tape: ti.template(),
        record: ti.i32,
        rhs: ti.template(),
    ):
        for contact in range(count):
            stencil = self.friction_candidate[contact]
            sample, particle, mpm_body, affine_body = self._pair_indices(stencil)
            distance2, gradient, hessian, _ = point_triangle_distance_grad_hess(
                cache_positions[stencil[0]],
                cache_positions[stencil[1]],
                cache_positions[stencil[2]],
                cache_positions[stencil[3]],
            )
            dhat = self.pair_dhat[mpm_body, affine_body]
            dmin = self.pair_dmin[mpm_body, affine_body]
            active_gap2 = (2.0 * dmin + dhat) * dhat
            coefficient_gradient = (
                self.pair_friction[mpm_body, affine_body]
                * self.mpm.surface_measure[sample]
                * dhat
                * ipc_barrier_force_magnitude_gradient(
                    distance2,
                    gradient,
                    dhat,
                    dmin,
                    self.pair_kappa[mpm_body, affine_body] / active_gap2**2,
                    0,
                )
            )
            positions = ti.Vector.zero(self.real_type, 12)
            references = ti.Vector.zero(self.real_type, 12)
            for site, component in ti.static(ti.ndrange(4, 3)):
                positions[3 * site + component] = self.positions[stencil[site]][component]
                references[3 * site + component] = self.friction_hat[stencil[site]][component]
            value = -self.affine.scale_device[None] * ipc_lagged_friction_cache_vjp(
                positions,
                references,
                self._contact_adjoint(stencil, particle),
                distance2,
                gradient,
                hessian,
                self.pair_friction[mpm_body, affine_body] * self.friction_normal_force[contact],
                coefficient_gradient,
                self.friction_dt[None],
                self.pair_epsv[mpm_body, affine_body],
                False,
            )
            ipc.scatter_friction_cache_point_vjp(
                particle,
                ti.Vector([value[c] for c in ti.static(range(3))]),
                displacement_tape,
                record,
                rhs,
                3 * ti.static(self.affine_control_count),
            )
            for site in ti.static(range(1, 4)):
                vertex = stencil[site] - ti.static(self.surface_count)
                body = self.affine.node2body[vertex]
                for control, component in ti.static(ti.ndrange(4, 3)):
                    ti.atomic_add(
                        rhs[12 * body + 3 * control + component],
                        self.affine.basis[vertex, control] * value[3 * site + component],
                    )


__all__ = ["DirectAffineIPCAssembler"]
