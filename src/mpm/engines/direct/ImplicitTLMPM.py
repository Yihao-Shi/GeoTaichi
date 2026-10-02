import taichi as ti

import src.mpm.config as config
from src.contact_detection.continuous_contact_detection import (
    deformation_gradient_ccd,
)
from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM
from src.mpm.engines.direct.MPMSolver import MPMConvergenceError
from src.mpm.utils import clear_field, copy_field, vectorize_id
from src.utils.StepRetry import is_recoverable_nonlinear_failure
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd


@ti.data_oriented
class ImplicitTLMPM(ImplicitMPM):
    def __init__(self, bodies, dirichlet=None, neumann=None, **kwargs):
        self.configuration = "TL"
        super().__init__(bodies, dirichlet, neumann, **kwargs)
        if self.is_axisymmetric:
            raise ValueError("Direct axisymmetric implicit MPM currently requires " "configuration='ULMPM'")
        if self.is_finite_strain_plastic:
            raise ValueError(
                "Direct finite-strain plasticity currently requires "
                "configuration='ULMPM'. TLMPM must track F_p explicitly "
                "instead of replacing it with the updated elastic state."
            )
        if self.is_plane_strain:
            raise ValueError("Direct 3D-embedded plane strain currently requires " "configuration='ULMPM'")

    @ti.kernel
    def grid_reset(self):
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                self.grid[i].v = ti.Vector.zero(ti.f64, config.DIM)
                self.grid[i].a = ti.Vector.zero(ti.f64, config.DIM)

    @ti.kernel
    def mass_p2g(self):
        for i in range(self.particleNum[0]):
            p_mass = self.particle[i].m
            for j in range(self.offset[i]):
                nodeID = self.LnID[i, j]
                shape_fn = self.shape[i, j]
                self.grid[nodeID].m += shape_fn * p_mass

    @ti.kernel
    def vel_acc_p2g(self):
        for i in range(self.particleNum[0]):
            bodyID = self.particle[i].bodyID
            goffset = self.body[bodyID].goffset
            grid_num = self.body[bodyID].grid_num
            grid_size = self.body[bodyID].grid_size
            xmin = self.body[bodyID].xmin

            p_pos = self.particle[i].x
            p_mass = self.particle[i].m
            p_vel = self.particle[i].v
            p_acc = self.particle[i].a
            for j in range(self.offset[i]):
                nodeID = self.LnID[i, j]
                shape_fn = self.shape[i, j]
                v_p2g = p_vel
                if ti.static(self.velocity_proj):
                    node_pos = xmin + grid_size * ti.Vector(vectorize_id(nodeID - goffset, grid_num))
                    v_p2g += self.gradv[i] @ (node_pos - p_pos)
                self.grid[nodeID].v += shape_fn * p_mass * v_p2g
                self.grid[nodeID].a += shape_fn * p_mass * p_acc

    @ti.kernel
    def assemble_material_force(self, active_dof: int, grid_disp: ti.template()):
        for i in range(self.particleNum[0]):
            gradu = self.get_displacement_incre(i, grid_disp)
            deformation_gradient = gradu + self.F0[i]
            pvol = self.particle[i].vol0
            dPsi_dF = self.material.dPsi_div_dF(deformation_gradient) * pvol

            for j in range(self.offset[i]):
                grid_id = self.LnID[i, j]
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                dF_dx = self.dshape[i, j]
                dPsi_dx = ti.Vector.zero(ti.f64, config.DIM)
                if ti.static(config.DIM == 2):
                    dPsi_dx[0] = dPsi_dF[0] * dF_dx[0] + dPsi_dF[2] * dF_dx[1]
                    dPsi_dx[1] = dPsi_dF[1] * dF_dx[0] + dPsi_dF[3] * dF_dx[1]
                else:
                    dPsi_dx[0] = dPsi_dF[0] * dF_dx[0] + dPsi_dF[3] * dF_dx[1] + dPsi_dF[6] * dF_dx[2]
                    dPsi_dx[1] = dPsi_dF[1] * dF_dx[0] + dPsi_dF[4] * dF_dx[1] + dPsi_dF[7] * dF_dx[2]
                    dPsi_dx[2] = dPsi_dF[2] * dF_dx[0] + dPsi_dF[5] * dF_dx[1] + dPsi_dF[8] * dF_dx[2]
                for d in ti.static(range(config.DIM)):
                    self.rhs[dofs + d] -= dPsi_dx[d]

    def assemble_stiffness_matrix(
        self,
        active_dof: int,
        grid_disp,
        project_spd=False,
        exact_plastic_tangent=False,
    ):
        self.assemble_stiffness_matrix_hash(
            active_dof,
            grid_disp,
            project_spd=project_spd,
            exact_plastic_tangent=exact_plastic_tangent,
        )

    def assemble_stiffness_matrix_hash(
        self,
        active_dof: int,
        grid_disp,
        project_spd=False,
        exact_plastic_tangent=False,
    ):
        """Assemble the exact or projected-Newton material tangent.

        Finite-strain plasticity always uses the symmetric PSD tangent of the
        projected-Newton method.  Other models retain the exact
        tangent unless the caller explicitly requests projection.
        """
        self._assemble_stiffness_matrix_hash(
            active_dof,
            grid_disp,
            bool(project_spd) or (self.is_finite_strain_plastic and not bool(exact_plastic_tangent)),
        )

    @ti.kernel
    def _assemble_stiffness_matrix_hash(
        self,
        active_dof: int,
        grid_disp: ti.template(),
        project_spd: ti.template(),
    ):
        self.hash_matrix.raw_non_diag_count[0] = self.particleNum[0] * self.stiffness_stencil_stride
        for i in range(self.particleNum[0]):
            self.invalidate_particle_stiffness_slots(i)
            pvol = self.particle[i].vol0
            gradu = self.get_displacement_incre(i, grid_disp)
            deformation_gradient = gradu + self.F0[i]
            d2Psi_d2F = self.material.d2Psi_div_d2F(deformation_gradient) * pvol
            if ti.static(project_spd):
                d2Psi_d2F = psd_project_nd(d2Psi_d2F)
            for j in range(self.offset[i]):
                base_jnode = self.LnID[i, j]
                dF_dx1 = self.dshape[i, j]
                base_joffset = self.node2dof[base_jnode] - 1
                for k in range(self.offset[i]):
                    base_knode = self.LnID[i, k]
                    dF_dx2 = self.dshape[i, k]
                    base_koffset = self.node2dof[base_knode] - 1
                    d2Psi_dx2 = self.local_stiffness(dF_dx1, dF_dx2, d2Psi_d2F)
                    raw_index = i * self.stiffness_stencil_stride + j * self.stiffness_stencil_width + k
                    self.add_fixed_stiffness_block_entry(raw_index, base_joffset, base_koffset, d2Psi_dx2)

    @ti.kernel
    def advent_particles(self, coeffPIC: ti.f64):
        dt = self.TIdt[None]
        for i in range(self.particleNum[0]):
            bodyID = self.particle[i].bodyID
            goffset = self.body[bodyID].goffset
            grid_num = self.body[bodyID].grid_num
            grid_size = self.body[bodyID].grid_size
            xmin = self.body[bodyID].xmin

            p_pos = self.particle[i].x
            acc = ti.Vector.zero(ti.f64, config.DIM)
            vel = ti.Vector.zero(ti.f64, config.DIM)
            disp = ti.Vector.zero(ti.f64, config.DIM)
            Bp = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            Dp = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            gradu = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            v0 = self.particle[i].v
            for j in range(self.offset[i]):
                grid_id = self.LnID[i, j]
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                shape_fn = self.shape[i, j]
                grid_v = self.grid[grid_id].v
                grid_a = self.grid[grid_id].a
                acc += shape_fn * grid_a
                vel += shape_fn * grid_v
                grid_displacement = ti.Vector([self.grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
                disp += shape_fn * grid_displacement
                gradu += grid_displacement.outer_product(self.dshape[i, j])
                if ti.static(self.velocity_proj):
                    dpos = xmin + grid_size * ti.Vector(vectorize_id(grid_id - goffset, grid_num)) - p_pos
                    Bp += shape_fn * grid_v.outer_product(dpos)
                    Dp += shape_fn * dpos.outer_product(dpos)
            v1 = (1.0 - coeffPIC) * (v0 + acc * dt) + coeffPIC * vel
            self.particle[i].v = v1
            self.particle[i].a = acc
            self.particle[i].x += disp
            self.F0[i] += gradu
            if ti.static(self.velocity_proj):
                self.gradv[i] = Bp @ Dp.inverse()

    @ti.kernel
    def material_ccd(self, slackness: ti.f64) -> ti.f64:
        alpha = 1.0
        for i in range(self.particleNum[0]):
            deformation_gradient_incre = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            curr_deformation_gradient_incre = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            for j in range(self.offset[i]):
                grid_id = self.LnID[i, j]
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                curr_deformation_gradient_incre += ti.Vector(
                    [self.grid_disp[dofs + d] for d in ti.static(range(config.DIM))]
                ).outer_product(self.dshape[i, j])
                deformation_gradient_incre += ti.Vector(
                    [self.incre_resolution[dofs + d] for d in ti.static(range(config.DIM))]
                ).outer_product(self.dshape[i, j])
            deformation_gradient = self.F0[i] + curr_deformation_gradient_incre
            solution = deformation_gradient_ccd(
                deformation_gradient,
                deformation_gradient_incre,
                slackness,
            )
            ti.atomic_min(alpha, solution)
        return alpha

    @ti.kernel
    def get_material_energy(self, grid_disp: ti.template()):
        for i in range(self.particleNum[0]):
            pvol = self.particle[i].vol0
            gradu = self.get_displacement_incre(i, grid_disp)
            deformation_gradient = self.F0[i] + gradu
            self.energy[None] += self.material.Psi(deformation_gradient) * pvol

    def get_degree_of_freedom(self, **kwargs):
        self.compute_shapefn()
        self.mass_p2g()
        return config.DIM * int(self.count_mass_active_nodes())

    @ti.kernel
    def count_mass_active_nodes(self) -> ti.i32:
        active_nodes = 0
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                active_nodes += 1
        return active_nodes

    def initial_simulation(self):
        self.mass_vec.fill(0)
        self.find_active_node()
        self.prefix_sum_executor.run(self.node2dof)
        self.active_dof = self.set_active_dof()
        if config.DYNAMIC:
            self.compute_mass_list(self.integration)
        super().initial_simulation()

    def substep(self, verbose=True):
        self.grid_reset()
        self.vel_acc_p2g()
        self.assemble_traction_step()
        self.compute_nodal_vel_acc()

        self.grid_disp.fill(0)
        iter_num = 0
        residual = 1.0
        while iter_num < self.max_iters:
            clear_field(self.active_dof, self.rhs)
            clear_field(self.active_dof, self.incre_resolution)
            self.hash_matrix.reset_system()

            self.assemble_inertia_force(self.active_dof, self.damping, self.gravity, self.integration, self.grid_disp)
            self.assemble_material_force(self.active_dof, self.grid_disp)
            self.assemble_stiffness_matrix_hash(
                self.active_dof,
                self.grid_disp,
                project_spd=self.project_pd,
            )
            if config.DYNAMIC:
                self.assemble_mass_matrix_hash()

            self.assemble_neumann_step()
            self.apply_dirichlet_step(self.active_dof)

            try:
                solve_result = self.solve_hash_system(self.active_dof)
            except RuntimeError as exception:
                if not is_recoverable_nonlinear_failure(exception):
                    raise
                raise MPMConvergenceError("Direct implicit TLMPM linear solve failed to converge") from exception
            residual = solve_result["solution_inf_norm"] / self.dt
            if residual < self.tol:
                self.update_grid_disp(self.active_dof, 1.0)
                copy_field(self.active_dof, self.grid_disp, self.grid_disp_temp)
                break

            success = self.line_search(self.active_dof, verbose)
            if not success:
                break
            iter_num += 1
        if residual >= self.tol:
            raise MPMConvergenceError(
                f"Direct implicit TLMPM did not converge: " f"residual={residual:.6e} after {iter_num} iterations"
            )
        self.update_nodal_acc(self.integration)
        self.advent_particles(self.coeffPIC)
        if verbose:
            print(f"Iteration {iter_num}, residual: {residual}")
        return {"iterations": int(iter_num), "residual": float(residual)}
