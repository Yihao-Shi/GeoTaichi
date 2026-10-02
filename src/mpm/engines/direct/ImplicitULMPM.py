import taichi as ti
import numpy as np
import math

import src.mpm.config as config
from src.contact_detection.continuous_contact_detection import (
    deformation_gradient_ccd,
)
from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM
from src.mpm.engines.direct.MPMSolver import MPMConvergenceError
from src.mpm.utils import copy_field, vectorize_id
from src.utils.StepRetry import is_recoverable_nonlinear_failure
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd


@ti.data_oriented
class ImplicitULMPM(ImplicitMPM):
    def __init__(self, bodies, dirichlet=None, neumann=None, **kwargs):
        self.configuration = "UL"
        super().__init__(bodies, dirichlet, neumann, **kwargs)
        self.has_lagged_material = bool(getattr(self.material, "requires_lagged_incremental_potential", False))
        self.material_lagged_tolerance = float(getattr(self.material, "lagged_tolerance", 0.0))
        self.material_lagged_max_iterations = int(getattr(self.material, "lagged_max_iterations", 1))
        self.last_material_lagged_error = 0.0
        self.last_material_lagged_iterations = 0

    @ti.kernel
    def _begin_lagged_material_state(self):
        for particle_id in range(self.particleNum[0]):
            self.material.begin_lagged_incremental_potential(particle_id)

    def begin_lagged_material_state(self):
        if self.has_lagged_material:
            self._begin_lagged_material_state()
        self.last_material_lagged_error = 0.0
        self.last_material_lagged_iterations = 0

    @ti.kernel
    def _refresh_lagged_material_state(self, grid_disp: ti.template()) -> ti.f64:
        error = 0.0
        for particle_id in range(self.particleNum[0]):
            deformation_gradient = ti.Matrix.identity(ti.f64, self.material_dimension)
            if ti.static(self.is_axisymmetric):
                deformation_gradient = (
                    self.get_axisymmetric_incremental_map(particle_id, grid_disp) @ self.F0[particle_id]
                )
            elif ti.static(self.is_plane_strain):
                deformation_gradient = (
                    self.get_plane_strain_incremental_map(particle_id, grid_disp) @ self.F0[particle_id]
                )
            else:
                gradu = self.get_displacement_incre(particle_id, grid_disp)
                deformation_gradient = (ti.Matrix.identity(ti.f64, config.DIM) + gradu) @ self.F0[particle_id]
            ti.atomic_max(
                error,
                self.material.refresh_lagged_incremental_potential(particle_id, deformation_gradient),
            )
        return error

    def refresh_lagged_material_state(self, grid_disp):
        if not self.has_lagged_material:
            return 0.0
        self.last_material_lagged_error = float(self._refresh_lagged_material_state(grid_disp))
        self.last_material_lagged_iterations += 1
        return self.last_material_lagged_error

    @ti.kernel
    def grid_reset(self):
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                self.grid[i].m = 0.0
                self.grid[i].v = ti.Vector.zero(ti.f64, config.DIM)
                self.grid[i].a = ti.Vector.zero(ti.f64, config.DIM)

    @ti.kernel
    def mass_vel_acc_p2g(self):
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
                self.grid[nodeID].m += shape_fn * p_mass
                self.grid[nodeID].v += shape_fn * p_mass * v_p2g
                self.grid[nodeID].a += shape_fn * p_mass * p_acc

    @ti.kernel
    def assemble_material_force(self, active_dof: int, grid_disp: ti.template()):
        for i in range(self.particleNum[0]):
            deformation_gradient = ti.Matrix.identity(ti.f64, self.material_dimension)
            if ti.static(self.is_axisymmetric):
                deformation_gradient = self.get_axisymmetric_incremental_map(i, grid_disp) @ self.F0[i]
            elif ti.static(self.is_plane_strain):
                deformation_gradient = self.get_plane_strain_incremental_map(i, grid_disp) @ self.F0[i]
            else:
                gradu = self.get_displacement_incre(i, grid_disp)
                deformation_gradient = (ti.Matrix.identity(ti.f64, config.DIM) + gradu) @ self.F0[i]
            pvol = self.particle[i].vol0
            dPsi_dF = ti.Vector.zero(ti.f64, self.material_dimension * self.material_dimension)
            if ti.static(self.is_finite_strain_plastic):
                dPsi_dF = self.material.total_dPsi_div_dF_at(i, deformation_gradient) * pvol
            else:
                dPsi_dF = self.material.dPsi_div_dF(deformation_gradient) * pvol
            for j in range(self.offset[i]):
                grid_id = self.LnID[i, j]
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                dPsi_dx = ti.Vector.zero(ti.f64, config.DIM)
                if ti.static(self.is_axisymmetric):
                    for component in ti.static(range(2)):
                        derivative = self.axisymmetric_dF_du(i, j, component)
                        for column, row in ti.static(ti.ndrange(3, 3)):
                            dPsi_dx[component] += derivative[row, column] * dPsi_dF[row + 3 * column]
                elif ti.static(self.is_plane_strain):
                    for component in ti.static(range(2)):
                        derivative = self.plane_strain_dF_du(i, j, component)
                        for column, row in ti.static(ti.ndrange(3, 3)):
                            dPsi_dx[component] += derivative[row, column] * dPsi_dF[row + 3 * column]
                elif ti.static(config.DIM == 2):
                    dF_dx = self.F0[i].transpose() @ self.dshape[i, j]
                    dPsi_dx[0] = dPsi_dF[0] * dF_dx[0] + dPsi_dF[2] * dF_dx[1]
                    dPsi_dx[1] = dPsi_dF[1] * dF_dx[0] + dPsi_dF[3] * dF_dx[1]
                else:
                    dF_dx = self.F0[i].transpose() @ self.dshape[i, j]
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
            deformation_gradient = ti.Matrix.identity(ti.f64, self.material_dimension)
            if ti.static(self.is_axisymmetric):
                deformation_gradient = self.get_axisymmetric_incremental_map(i, grid_disp) @ self.F0[i]
            elif ti.static(self.is_plane_strain):
                deformation_gradient = self.get_plane_strain_incremental_map(i, grid_disp) @ self.F0[i]
            else:
                gradu = self.get_displacement_incre(i, grid_disp)
                deformation_gradient = (ti.Matrix.identity(ti.f64, config.DIM) + gradu) @ self.F0[i]
            d2Psi_d2F = ti.Matrix.zero(
                ti.f64,
                self.material_dimension * self.material_dimension,
                self.material_dimension * self.material_dimension,
            )
            if ti.static(self.is_finite_strain_plastic):
                d2Psi_d2F = self.material.total_d2Psi_div_d2F_at(i, deformation_gradient) * pvol
            else:
                d2Psi_d2F = self.material.d2Psi_div_d2F(deformation_gradient) * pvol
            if ti.static(project_spd):
                d2Psi_d2F = psd_project_nd(d2Psi_d2F)
            for j in range(self.offset[i]):
                base_jnode = self.LnID[i, j]
                base_joffset = self.node2dof[base_jnode] - 1
                for k in range(self.offset[i]):
                    base_knode = self.LnID[i, k]
                    base_koffset = self.node2dof[base_knode] - 1
                    d2Psi_dx2 = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
                    if ti.static(self.is_axisymmetric):
                        d2Psi_dx2 = self.axisymmetric_local_stiffness(i, j, k, d2Psi_d2F)
                    elif ti.static(self.is_plane_strain):
                        d2Psi_dx2 = self.plane_strain_local_stiffness(i, j, k, d2Psi_d2F)
                    else:
                        dF_dx1 = self.F0[i].transpose() @ self.dshape[i, j]
                        dF_dx2 = self.F0[i].transpose() @ self.dshape[i, k]
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
            radial_displacement = 0.0
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
                if ti.static(self.is_axisymmetric):
                    radial_displacement += shape_fn * grid_displacement[0]
                if ti.static(self.velocity_proj):
                    dpos = xmin + grid_size * ti.Vector(vectorize_id(grid_id - goffset, grid_num)) - p_pos
                    Bp += shape_fn * grid_v.outer_product(dpos)
                    Dp += shape_fn * dpos.outer_product(dpos)
            v1 = (1.0 - coeffPIC) * (v0 + acc * dt) + coeffPIC * vel
            self.particle[i].v = v1
            self.particle[i].a = acc
            self.particle[i].x += disp
            trial_deformation_gradient = ti.Matrix.identity(ti.f64, self.material_dimension)
            if ti.static(self.is_axisymmetric):
                incremental = ti.Matrix.identity(ti.f64, 3)
                for row, column in ti.static(ti.ndrange(2, 2)):
                    incremental[row, column] += gradu[row, column]
                radius = p_pos[0] - ti.static(self.axis_offset)
                incremental[2, 2] += radial_displacement / ti.max(radius, 1.0e-30)
                trial_deformation_gradient = incremental @ self.F0[i]
            elif ti.static(self.is_plane_strain):
                incremental = ti.Matrix.identity(ti.f64, 3)
                for row, column in ti.static(ti.ndrange(2, 2)):
                    incremental[row, column] += gradu[row, column]
                trial_deformation_gradient = incremental @ self.F0[i]
            else:
                trial_deformation_gradient = (ti.Matrix.identity(ti.f64, config.DIM) + gradu) @ self.F0[i]
            if ti.static(self.is_finite_strain_plastic):
                self.F0[i] = self.material.commit_total_state(i, trial_deformation_gradient)
            else:
                self.F0[i] = trial_deformation_gradient
            if ti.static(self.velocity_proj):
                self.gradv[i] = Bp @ Dp.inverse()

    @ti.kernel
    def material_ccd(self, slackness: ti.f64) -> ti.f64:
        alpha = 1.0
        for i in range(self.particleNum[0]):
            map_dimension = ti.static(self.material_dimension)
            current_gradu = ti.Matrix.zero(ti.f64, map_dimension, map_dimension)
            increment_gradu = ti.Matrix.zero(ti.f64, map_dimension, map_dimension)
            current_radial = 0.0
            increment_radial = 0.0
            for j in range(self.offset[i]):
                grid_id = self.LnID[i, j]
                dofs = config.DIM * (self.node2dof[grid_id] - 1)
                current_displacement = ti.Vector([self.grid_disp[dofs + d] for d in ti.static(range(config.DIM))])
                increment_displacement = ti.Vector(
                    [self.incre_resolution[dofs + d] for d in ti.static(range(config.DIM))]
                )
                for row, column in ti.static(ti.ndrange(config.DIM, config.DIM)):
                    current_gradu[row, column] += current_displacement[row] * self.dshape[i, j][column]
                    increment_gradu[row, column] += increment_displacement[row] * self.dshape[i, j][column]
                if ti.static(self.is_axisymmetric):
                    current_radial += self.shape[i, j] * current_displacement[0]
                    increment_radial += self.shape[i, j] * increment_displacement[0]

            if ti.static(self.is_axisymmetric):
                radius = self.particle[i].x[0] - ti.static(self.axis_offset)
                current_gradu[2, 2] = current_radial / ti.max(radius, 1.0e-30)
                increment_gradu[2, 2] = increment_radial / ti.max(radius, 1.0e-30)

            # UL kinematics at a line-search trial are
            # F(alpha) = (I + grad(u_k) + alpha grad(p)) F_n.  det(F_n)
            # is a positive constant, so its first inversion time is exactly
            # the first root for the two spatial maps below.  As in IPC/TL,
            # stop with (1-slackness) of the current determinant remaining.
            current_map = ti.Matrix.identity(ti.f64, map_dimension) + current_gradu
            solution = deformation_gradient_ccd(current_map, increment_gradu, slackness)
            ti.atomic_min(alpha, solution)
        return alpha

    @ti.kernel
    def get_material_energy(self, grid_disp: ti.template()):
        for i in range(self.particleNum[0]):
            pvol = self.particle[i].vol0
            deformation_gradient = ti.Matrix.identity(ti.f64, self.material_dimension)
            if ti.static(self.is_axisymmetric):
                deformation_gradient = self.get_axisymmetric_incremental_map(i, grid_disp) @ self.F0[i]
            elif ti.static(self.is_plane_strain):
                deformation_gradient = self.get_plane_strain_incremental_map(i, grid_disp) @ self.F0[i]
            else:
                gradu = self.get_displacement_incre(i, grid_disp)
                deformation_gradient = (ti.Matrix.identity(ti.f64, config.DIM) + gradu) @ self.F0[i]
            if ti.static(self.is_finite_strain_plastic):
                self.energy[None] += self.material.total_strain_energy_density_at(i, deformation_gradient) * pvol
            else:
                self.energy[None] += self.material.Psi(deformation_gradient) * pvol

    def get_degree_of_freedom(self, **kwargs):
        return int(kwargs.get("scale", 0.2) * config.DIM * self.total_background_grid_num)

    def _solve_lagged_material_inner(self, verbose):
        iter_num = 0
        residual = 1.0
        while iter_num < self.max_iters:
            self.matrix_reset()
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
                raise MPMConvergenceError("Direct implicit ULMPM linear solve failed to converge") from exception
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
                f"Direct implicit ULMPM did not converge: " f"residual={residual:.6e} after {iter_num} iterations"
            )
        return {"iterations": int(iter_num), "residual": float(residual)}

    def substep(self, verbose=True):
        self.mass_vec.fill(0)
        self.grid_reset()
        self.compute_shapefn()
        self.mass_vel_acc_p2g()
        self.assemble_traction_step()
        self.find_active_node()
        self.prefix_sum_executor.run(self.node2dof)
        self.active_dof = self.set_active_dof()
        self.compute_nodal_vel_acc()
        if config.DYNAMIC:
            self.compute_mass_list(self.integration)

        self.grid_disp.fill(0)
        self.begin_lagged_material_state()
        result = None
        material_converged = not self.has_lagged_material
        outer_limit = self.material_lagged_max_iterations if self.has_lagged_material else 1
        for _ in range(outer_limit):
            result = self._solve_lagged_material_inner(verbose)
            if not self.has_lagged_material:
                material_converged = True
                break
            if self.refresh_lagged_material_state(self.grid_disp) <= self.material_lagged_tolerance:
                material_converged = True
                break
        if not material_converged:
            raise MPMConvergenceError(
                "Direct implicit ULMPM lagged MCC hardening did not converge: "
                f"error={self.last_material_lagged_error:.6e} after "
                f"{self.material_lagged_max_iterations} iterations"
            )
        self.update_nodal_acc(self.integration)
        self.advent_particles(self.coeffPIC)
        if verbose:
            print(
                f"Iteration {result['iterations']}, residual: {result['residual']}, "
                f"material_lagged_iterations: {self.last_material_lagged_iterations}"
            )
        result["material_lagged_iterations"] = self.last_material_lagged_iterations
        result["material_lagged_error"] = self.last_material_lagged_error
        return result
