"""Direct-MPM point contact pulled back to AffineBody controls."""

from types import SimpleNamespace

import taichi as ti

from src.fempm.contact.IPCAssembler import FEMPMIPCAssembler


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
            max_point_triangle_pairs=int(simulation.max_point_triangle_pairs),
            max_point_edge_pairs=max(int(getattr(simulation, "max_point_edge_pairs", 1)), 1),
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

    def contact_block_capacity(self, candidate_count):
        support = int(self.mpm.shape_func.max_node_per_particle) + 12
        contacts = int(candidate_count) + int(self.friction_count)
        return max(1, contacts * support * support)

    def configured_contact_block_capacity(self):
        support = int(self.mpm.shape_func.max_node_per_particle) + 12
        contributions = 2 if self.activate_friction else 1
        return max(
            1,
            contributions * int(self.max_point_triangle_pairs) * support * support,
        )


__all__ = ["DirectAffineIPCAssembler"]
