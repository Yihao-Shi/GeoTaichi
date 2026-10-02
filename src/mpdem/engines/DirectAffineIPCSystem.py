"""Monolithic Direct-MPM/AffineBody IPC linearization."""

import math

import numpy as np
import taichi as ti

from src.dem.engines.AffineBodyOperator import MATRIX_HASH_TRIPLET
from src.linear_solver.BuildTriplet import BuildTriplet


@ti.data_oriented
class DirectAffineIPCSystem:
    """Merge existing ABD, Direct-MPM, and mixed IPC device sources once."""

    def __init__(self, affine, ipc, mixed_contact):
        self.affine = affine
        self.ipc = ipc
        self.mpm = ipc.mpm
        self.mixed = mixed_contact
        self.last_contact_count = 0
        self.last_candidate_count = 0
        self.last_step_record = None
        if int(getattr(self.mpm, "dimension", 3)) != 3:
            raise ValueError("Direct MPM--ABD IPC currently requires 3D")
        if not (bool(affine.is_semi) == bool(ipc.is_semi) == bool(mixed_contact.is_semi)):
            raise ValueError("Direct MPM--ABD must use one common BarrierIPC/SemiIPC mode")
        if bool(getattr(affine, "fully_implicit", False)) or str(getattr(ipc, "friction_mode", "lagged")) != "lagged":
            raise ValueError(
                "Direct MPM--ABD supports lagged friction only; fully " "implicit friction is not implemented"
            )

        self.affine_controls = int(affine.control_num)
        self.mpm_node_capacity = int(self.mpm.degree_of_freedom // 3)
        self.node_capacity = self.affine_controls + self.mpm_node_capacity
        self.dof_capacity = 3 * self.node_capacity
        self.rhs = ti.field(ti.f64, shape=self.dof_capacity)
        self.physical_rhs = ti.field(ti.f64, shape=self.dof_capacity)
        self.correction = ti.field(ti.f64, shape=self.dof_capacity)

        affine_raw = max(int(affine.max_hash_triplets), 1)
        self.affine_source = BuildTriplet(
            dim=3,
            max_pairs_num=affine_raw,
            max_nonzeros=1,
            max_active_nodes=max(self.affine_controls, 1),
            symmetric=False,
            solver="BiCGSTAB",
            matrix_symmetric=False,
            device_reduction=True,
            raw_only=True,
        )
        affine.bind_hash_triplet(self.affine_source, full_symmetric_input=True)

        mixed_raw = int(mixed_contact.configured_contact_block_capacity())
        self.mixed_source = BuildTriplet(
            dim=3,
            max_pairs_num=max(mixed_raw, 1),
            max_nonzeros=1,
            max_active_nodes=max(self.node_capacity, 1),
            symmetric=False,
            solver="BiCGSTAB",
            matrix_symmetric=False,
            device_reduction=True,
            raw_only=True,
        )
        self.mpm_sources = [self.mpm.hash_matrix]
        if ipc.activate_barrier:
            self.mpm_sources.append(ipc.barrier_hash_matrix)
        if ipc.activate_fric:
            self.mpm_sources.append(ipc.friction_hash_matrix)
        total_raw = affine_raw + mixed_raw + sum(int(source.non_diag.max_pairs_num) for source in self.mpm_sources)
        maximum_upper = self.node_capacity * max(self.node_capacity - 1, 0) // 2
        reduced = min(max(total_raw, 1), max(maximum_upper, 1))
        self.matrix = BuildTriplet(
            dim=3,
            max_pairs_num=max(total_raw, 1),
            max_nonzeros=reduced,
            max_active_nodes=max(self.node_capacity, 1),
            symmetric=False,
            solver="PCG",
            matrix_symmetric=True,
            full_symmetric_input=True,
            device_reduction=True,
        )

    def begin_step_device(self, timestep):
        """Initialize one coupled step and freeze all three friction caches."""
        timestep = float(timestep)
        self.affine.set_timestep(timestep)
        if self.affine.is_semi:
            self.affine.reset_semi_state()
            self.ipc.semi_state.fill(2)
            self.ipc.semi_multiplier.fill(0.0)
            self.ipc.semi_count[None] = 0
            self.ipc.semi_overflow[None] = 0
        self.affine.device_backup_step_start()
        self.affine.device_begin_step(timestep)
        self.mpm.grid_disp.fill(0.0)
        self.affine.initialize_contact_damping_device()
        self.ipc.begin_friction_step(self.mpm.grid_disp)
        self.mixed.begin_step(self.affine.x, self.mpm.grid_disp, timestep)

    def refresh_lagged_friction_device(self):
        self.affine.initialize_contact_damping_device()
        self.ipc.refresh_friction_cache(self.mpm.grid_disp)
        self.mixed.refresh_friction(self.affine.x, self.mpm.grid_disp)

    def backup_lagged_friction_for_adjoint_device(self):
        self.affine.device_backup_lagged_friction_for_adjoint()
        if self.ipc.activate_fric:
            self.ipc._backup_lagged_friction_for_adjoint()
        self.mixed.backup_lagged_friction_for_adjoint_device()

    def restore_lagged_friction_for_adjoint_device(self):
        self.affine.device_restore_lagged_friction_for_adjoint()
        if self.ipc.activate_fric:
            self.ipc._restore_lagged_friction_for_adjoint()
            self.ipc.curr_friction_contact_num = int(
                self.ipc.adjoint_gfriction_num[None] + self.ipc.adjoint_pfriction_num[None]
            )
        self.mixed.restore_lagged_friction_for_adjoint_device()

    @ti.kernel
    def _load_rhs(self, active_mpm_dof: ti.i32):
        scale = self.affine.scale_device[None]
        active_dof = ti.static(3 * self.affine_controls) + active_mpm_dof
        for dof in range(self.dof_capacity):
            value = 0.0
            if dof < ti.static(3 * self.affine_controls):
                control = dof // 3
                component = dof % 3
                value = -self.affine.grad[control][component]
            elif dof < active_dof:
                value = scale * self.mpm.rhs[dof - ti.static(3 * self.affine_controls)]
            self.rhs[dof] = value

    @ti.func
    def _mpm_fixed(self, block, component):
        fixed = False
        if block >= ti.static(self.affine_controls):
            local = block - ti.static(self.affine_controls)
            grid = self.mpm.dof2node[local]
            fixed = self.mpm.dirichlet.node[3 * grid + component] != 0
        return fixed

    @ti.func
    def _mpm_fixed_correction(self, block, component):
        local = block - ti.static(self.affine_controls)
        grid = self.mpm.dof2node[local]
        return self.mpm.dirichlet.value[3 * grid + component] - self.mpm.grid_disp[3 * local + component]

    @ti.kernel
    def _copy_physical_rhs(self, active_dof: ti.i32):
        for dof in range(self.dof_capacity):
            self.physical_rhs[dof] = self.rhs[dof] if dof < active_dof else 0.0

    @ti.kernel
    def _scatter_correction(self, active_mpm_dof: ti.i32):
        for control in range(self.affine_controls):
            for component in ti.static(range(3)):
                self.affine.direction_y[control][component] = self.correction[3 * control + component]
        for dof in range(self.mpm.degree_of_freedom):
            self.mpm.incre_resolution[dof] = (
                self.correction[3 * ti.static(self.affine_controls) + dof] if dof < active_mpm_dof else 0.0
            )

    @ti.kernel
    def _scatter_adjoint(self, active_mpm_dof: ti.i32):
        """Scatter the monolithic transpose solve to both device operators."""
        scale = self.affine.scale_device[None]
        for dof in range(3 * self.affine_controls):
            self.affine.linear_x[dof] = self.correction[dof]
        for dof in range(self.mpm.degree_of_freedom):
            self.mpm.incre_resolution[dof] = (
                scale * self.correction[3 * self.affine_controls + dof] if dof < active_mpm_dof else 0.0
            )

    @ti.kernel
    def _copy_adjoint_rhs(self, source: ti.template(), active_dof: ti.i32):
        for dof in range(self.dof_capacity):
            self.rhs[dof] = source[dof] if dof < active_dof else 0.0

    def solve_adjoint_device(
        self,
        loss_gradient,
        active_mpm_dof=None,
        *,
        exact_plastic_tangent=False,
    ):
        """Solve one coupled transpose system without host sparse conversion."""
        self.restore_lagged_friction_for_adjoint_device()
        assembly = self.assemble_linearization_device(
            project_spd=False,
            exact_plastic_tangent=bool(exact_plastic_tangent),
            solver_shift=False,
        )
        assembled_mpm_dof = int(assembly["active_mpm_dof"])
        active_mpm_dof = assembled_mpm_dof if active_mpm_dof is None else int(active_mpm_dof)
        if active_mpm_dof != assembled_mpm_dof:
            raise ValueError("active_mpm_dof does not match the exact coupled adjoint assembly")
        active_nodes = self.affine_controls + active_mpm_dof // 3
        active_dof = 3 * active_nodes
        if isinstance(loss_gradient, ti.ScalarField):
            if len(loss_gradient.shape) != 1 or int(loss_gradient.shape[0]) < active_dof:
                raise ValueError("loss_gradient field must cover coupled active DOFs")
            self._copy_adjoint_rhs(loss_gradient, active_dof)
        else:
            values = np.asarray(loss_gradient, dtype=np.float64).reshape(-1)
            if values.size != active_dof or not np.isfinite(values).all():
                raise ValueError("loss_gradient must be finite and match coupled active DOFs")
            padded = np.zeros(self.dof_capacity, dtype=np.float64)
            padded[:active_dof] = values
            self.rhs.from_numpy(padded)
        self.correction.fill(0.0)
        forward_solver = self.matrix.solver
        self.matrix.solver = "PCG" if self.matrix.matrix_symmetric else "BiCGSTAB"
        try:
            result = self.matrix.solve_flat_system(
                self.rhs,
                self.correction,
                active_nodes=active_nodes,
                tol=self.mpm.linear_solver_tolerance,
                maxiter=self.mpm.linear_solver_max_iters,
                return_solution=False,
                transpose=not self.matrix.matrix_symmetric,
                fallback_to_bicgstab=self.matrix.matrix_symmetric,
            )
        finally:
            self.matrix.solver = forward_solver
        if not result["converged"]:
            raise RuntimeError(
                "Direct MPM--ABD coupled adjoint did not converge: " f"residual={result['residual']:.6e}"
            )
        self._scatter_adjoint(active_mpm_dof)
        return self.correction

    def differentiate_plastic_equilibrium_parameters(self, loss_gradient, active_mpm_dof=None):
        """One-step DP/VM material VJP for ordinary MPM--polyhedral ABD IPC."""
        if type(self.mpm.material).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError("coupled material VJP supports DP and von Mises only")
        adjoint = self.solve_adjoint_device(
            loss_gradient,
            active_mpm_dof,
            exact_plastic_tangent=True,
        )
        self.ipc.pullback_plastic_equilibrium_from_current_adjoint_device()
        self.affine._differentiate_gravity_parameter()
        self.affine._differentiate_young_parameter()
        self.affine._differentiate_joint_target_parameters()
        self.affine._differentiate_joint_damping_parameters()
        self.affine._differentiate_friction_scale_parameter()
        values = np.asarray(self.ipc.material_parameter_vjp[None], dtype=np.float64)
        if type(self.mpm.material).__name__ == "FiniteStrainDruckerPragerModel":
            material_names = {
                "cohesion": float(values[2]),
                "friction_angle_degrees": float(values[3]),
            }
        else:
            material_names = {
                "yield_stress": float(values[2]),
                "hardening_modulus": float(values[3]),
            }
        return {
            "adjoint": adjoint,
            "material_parameters": values.copy(),
            "young_modulus": float(values[0]),
            "poisson_ratio": float(values[1]),
            **material_names,
            "gravity": np.asarray(self.ipc.gravity_vjp[None], dtype=np.float64),
            "affine_gravity": np.asarray(self.affine.gravity_vjp[None], dtype=np.float64),
            "affine_young_modulus": self.affine.young_vjp.to_numpy()[: self.affine.body_num].copy(),
            "joint_target_angle_radians": self.affine.joint_target_vjp.to_numpy()[: self.affine.joint_num].copy(),
            "joint_damping": self.affine.joint_damping_vjp.to_numpy()[: self.affine.joint_num].copy(),
            "friction_scale": float(self.affine.friction_scale_vjp[None]),
            "friction_coefficient": float(self.ipc.friction_parameter_vjp[None][0]),
        }

    @ti.kernel
    def _scale_correction(self, active_dof: ti.i32, scale: ti.f64):
        for dof in range(active_dof):
            self.correction[dof] *= scale

    @ti.kernel
    def _accept_mpm_trial(self, active_mpm_dof: ti.i32):
        for dof in range(self.mpm.degree_of_freedom):
            if dof < active_mpm_dof:
                self.mpm.grid_disp[dof] = self.mpm.grid_disp_temp[dof]

    @ti.kernel
    def _active_inf_norm(self, active_dof: ti.i32, values: ti.template()) -> ti.f64:
        result = 0.0
        for dof in range(active_dof):
            ti.atomic_max(result, ti.abs(values[dof]))
        return result

    @ti.kernel
    def _gradient_direction_dot(self, active_dof: ti.i32) -> ti.f64:
        result = 0.0
        for dof in range(active_dof):
            result -= self.physical_rhs[dof] * self.correction[dof]
        return result

    @ti.kernel
    def _eliminate_mpm_dirichlet(self, active_nodes: ti.i32):
        for block in range(active_nodes):
            diagonal = self.matrix.diag[block]
            for row, column in ti.static(ti.ndrange(3, 3)):
                row_fixed = self._mpm_fixed(block, row)
                column_fixed = self._mpm_fixed(block, column)
                value = diagonal[3 * row + column]
                if column_fixed:
                    self.rhs[3 * block + row] -= value * (self._mpm_fixed_correction(block, column))
                if row_fixed or column_fixed:
                    diagonal[3 * row + column] = 0.0
            for component in ti.static(range(3)):
                if self._mpm_fixed(block, component):
                    diagonal[3 * component + component] = 1.0
            self.matrix.diag[block] = diagonal

        for entry in range(self.matrix.raw_non_diag_count[0]):
            first = self.matrix.non_diag.blockI[entry]
            second = self.matrix.non_diag.blockJ[entry]
            if 0 <= first < active_nodes and 0 <= second < active_nodes:
                block = self.matrix.non_diag.blockH[entry]
                for row, column in ti.static(ti.ndrange(3, 3)):
                    row_fixed = self._mpm_fixed(first, row)
                    column_fixed = self._mpm_fixed(second, column)
                    value = block[3 * row + column]
                    if column_fixed:
                        ti.atomic_add(
                            self.rhs[3 * first + row],
                            -value * self._mpm_fixed_correction(second, column),
                        )
                    if row_fixed:
                        ti.atomic_add(
                            self.rhs[3 * second + column],
                            -value * self._mpm_fixed_correction(first, row),
                        )
                    if row_fixed or column_fixed:
                        block[3 * row + column] = 0.0
                self.matrix.non_diag.blockH[entry] = block

        for block in range(ti.static(self.affine_controls), active_nodes):
            for component in ti.static(range(3)):
                if self._mpm_fixed(block, component):
                    self.rhs[3 * block + component] = self._mpm_fixed_correction(block, component)

    def assemble_linearization_device(
        self,
        *,
        need_matrix=True,
        project_spd=True,
        exact_plastic_tangent=False,
        include_mixed_friction=True,
        mpm_displacement=None,
        solver_shift=True,
    ):
        """Assemble one coupled residual and optional projected tangent."""
        if mpm_displacement is None:
            mpm_displacement = self.mpm.grid_disp
        if need_matrix:
            self.affine_source.reset_system()
            self.affine.bind_hash_triplet(self.affine_source, full_symmetric_input=True)
        self.affine.assemble_device(
            need_matrix=bool(need_matrix),
            matrix_mode=MATRIX_HASH_TRIPLET,
            project_spd=bool(project_spd),
            solver_shift=bool(solver_shift),
        )
        active_mpm_dof = self.ipc.assemble_current_sources(
            mpm_displacement,
            need_matrix=bool(need_matrix),
            project_spd=bool(project_spd),
            exact_plastic_tangent=bool(exact_plastic_tangent),
        )
        contact_count = self.mixed.prepare(self.affine.x, mpm_displacement)
        self._load_rhs(int(active_mpm_dof))
        if need_matrix:
            self.mixed_source.reset_system()
        self.mixed.assemble(
            int(contact_count),
            self.mixed.candidate_field(),
            self.mixed_source,
            self.rhs,
            bool(need_matrix),
            bool(include_mixed_friction and self.mixed.activate_friction),
            int(self.mixed.friction_count),
            project_pd=bool(project_spd),
        )
        active_mpm_nodes = int(active_mpm_dof) // 3
        active_nodes = self.affine_controls + active_mpm_nodes
        active_dof = 3 * active_nodes
        self._copy_physical_rhs(active_dof)
        if not need_matrix:
            return None

        self.matrix.reset_system()
        self.matrix.append_raw_from(
            self.affine_source,
            active_nodes=self.affine_controls,
        )
        scale = float(self.affine.scale_device[None])
        for source in self.mpm_sources:
            self.matrix.append_raw_from(
                source,
                active_nodes=active_mpm_nodes,
                block_offset=self.affine_controls,
                scale=scale,
            )
        self.matrix.append_raw_from(
            self.mixed_source,
            active_nodes=active_nodes,
        )
        self.matrix.canonicalize_full_symmetric_input()
        if self.mpm.dirichlet.num > 0:
            self._eliminate_mpm_dirichlet(active_nodes)
        self.matrix.finalize_taichi_assembly()
        return {
            "matrix": self.matrix,
            "rhs": self.rhs,
            "active_mpm_dof": int(active_mpm_dof),
            "active_nodes": active_nodes,
            "active_dof": active_dof,
            "contact_count": int(contact_count),
        }

    def solve_direction_device(self, active_mpm_dof=None):
        """Solve the assembled coupled system and scatter both directions."""
        active_mpm_dof = int(self.mpm.active_dof) if active_mpm_dof is None else int(active_mpm_dof)
        active_nodes = self.affine_controls + active_mpm_dof // 3
        self.correction.fill(0.0)
        result = self.matrix.solve_flat_system(
            self.rhs,
            self.correction,
            active_nodes=active_nodes,
            tol=self.mpm.linear_solver_tolerance,
            maxiter=self.mpm.linear_solver_max_iters,
            return_solution=False,
        )
        if not result["converged"]:
            raise RuntimeError(
                "Direct MPM--ABD coupled PCG did not converge: "
                f"residual={result['residual']:.6e}, "
                f"iterations={result['iterations']}"
            )
        self._scatter_correction(active_mpm_dof)
        return result

    def direction_inf_norm(self, active_mpm_dof=None):
        active_mpm_dof = int(self.mpm.active_dof) if active_mpm_dof is None else int(active_mpm_dof)
        return max(
            float(self.affine.device_surface_direction_inf_norm()),
            float(
                self._active_inf_norm(
                    active_mpm_dof,
                    self.mpm.incre_resolution,
                )
            ),
        )

    def gradient_direction_dot(self, active_mpm_dof=None):
        active_mpm_dof = int(self.mpm.active_dof) if active_mpm_dof is None else int(active_mpm_dof)
        return float(self._gradient_direction_dot(3 * self.affine_controls + active_mpm_dof))

    def scale_direction_device(self, scale, active_mpm_dof=None):
        active_mpm_dof = int(self.mpm.active_dof) if active_mpm_dof is None else int(active_mpm_dof)
        self._scale_correction(
            3 * self.affine_controls + active_mpm_dof,
            float(scale),
        )
        self._scatter_correction(active_mpm_dof)

    def maximum_step_device(
        self,
        *,
        active_mpm_dof=None,
        ccd_type="ccd",
        ccd_eta=0.2,
        accd_tolerance=1.0e-7,
        ccd_max_iterations=10000,
    ):
        """Return the common CCD bound for ABD, MPM, and cross contact."""
        active_mpm_dof = int(self.mpm.active_dof) if active_mpm_dof is None else int(active_mpm_dof)
        affine_step = self.affine.init_step_size_device(
            ccd_type=ccd_type,
            eta=float(ccd_eta),
            accd_tolerance=float(accd_tolerance),
            max_iteration=int(ccd_max_iterations),
        )
        mpm_step = self.ipc.ccd(active_mpm_dof)
        mixed_step = self.mixed.maximum_step(
            self.affine.x,
            self.affine.direction_y,
            self.mpm.grid_disp,
            self.mpm.incre_resolution,
        )
        return min(float(affine_step), float(mpm_step), float(mixed_step))

    def begin_line_search_device(self):
        self.affine.device_backup_line_search_base()

    def set_line_search_trial_device(self, scale, active_mpm_dof=None):
        active_mpm_dof = int(self.mpm.active_dof) if active_mpm_dof is None else int(active_mpm_dof)
        self.affine.device_set_line_search_trial(float(scale))
        self.mpm.update_grid_disp(active_mpm_dof, float(scale))
        return self.mpm.grid_disp_temp

    def restore_line_search_base_device(self):
        self.affine.device_restore_line_search_base()

    def accept_line_search_trial_device(self, active_mpm_dof=None):
        active_mpm_dof = int(self.mpm.active_dof) if active_mpm_dof is None else int(active_mpm_dof)
        self._accept_mpm_trial(active_mpm_dof)
        if self.affine.is_semi:
            self.affine.accept_semi_update_device()
            self.ipc.accept_update()
            self.mixed.accept_update(self.affine.x, self.mpm.grid_disp)

    def advance_semi_multipliers_device(self):
        if not self.affine.is_semi:
            return
        self.affine.accept_semi_update_device()
        self.ipc.accept_update()
        self.mixed.accept_update(self.affine.x, self.mpm.grid_disp)

    def contact_converged(self):
        return self.affine.semi_contact_converged() and self.ipc.contact_converged() and self.mixed.contact_converged()

    def total_energy_device(self, mpm_displacement=None, include_mixed_friction=True):
        """Evaluate the coupled incremental potential at the current ABD state."""
        if mpm_displacement is None:
            mpm_displacement = self.mpm.grid_disp
        affine_energy = self.affine.assemble_device(
            need_matrix=False,
            matrix_mode=MATRIX_HASH_TRIPLET,
            project_spd=True,
        )
        mpm_energy = self.ipc.total_energy(mpm_displacement)
        contact_count = self.mixed.prepare(self.affine.x, mpm_displacement)
        self.rhs.fill(0.0)
        self.mixed.assemble(
            int(contact_count),
            self.mixed.candidate_field(),
            self.mixed_source,
            self.rhs,
            False,
            bool(include_mixed_friction and self.mixed.activate_friction),
            int(self.mixed.friction_count),
        )
        scale = float(self.affine.scale_device[None])
        return float(affine_energy) + scale * (float(mpm_energy) + float(self.mixed.total_energy[None]))

    def _line_search_device(self, sims, energy, active_mpm_dof):
        slope = self.gradient_direction_dot(active_mpm_dof)
        if not math.isfinite(slope):
            raise RuntimeError("Direct MPM--ABD lagged line-search slope is non-finite")
        if slope > 0.0:
            self.scale_direction_device(-1.0, active_mpm_dof)
            slope = -slope
        if slope >= 0.0:
            raise RuntimeError("Direct MPM--ABD lagged line search requires a descent direction")
        alpha = (
            self.maximum_step_device(
                active_mpm_dof=active_mpm_dof,
                ccd_type=sims.affine_ccd_type,
                ccd_eta=sims.affine_ccd_eta,
                accd_tolerance=sims.affine_accd_tolerance,
                ccd_max_iterations=sims.affine_ccd_max_iteration,
            )
            if sims.affine_ccd
            else 1.0
        )
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise RuntimeError("Direct MPM--ABD lagged CCD produced no feasible step")
        self.begin_line_search_device()
        for _ in range(int(sims.affine_line_search_max_iteration)):
            trial = self.set_line_search_trial_device(alpha, active_mpm_dof)
            trial_energy = self.total_energy_device(trial)
            if math.isfinite(trial_energy) and trial_energy <= energy:
                self.assemble_linearization_device(mpm_displacement=trial)
                self.accept_line_search_trial_device(active_mpm_dof)
                if self.affine.is_semi:
                    trial_energy = self.total_energy_device()
                    self.assemble_linearization_device()
                return trial_energy, alpha
            alpha *= 0.5
        self.restore_line_search_base_device()
        self.assemble_linearization_device()
        raise RuntimeError("Direct MPM--ABD lagged monotone line search failed")

    def _solve_lagged_equilibrium_device(self, sims, record_adjoint=False, begin_step=True):
        """Solve one pre-commit monolithic MPM--ABD equilibrium."""
        timestep = float(self.mpm.dt)
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("Direct MPM--ABD requires a positive time step")
        raw_outer = float(sims.affine_friction_iterations)
        requested_outer = int(raw_outer)
        if not math.isfinite(raw_outer) or raw_outer != requested_outer:
            raise ValueError("Direct MPM--ABD friction_iterations must be an integer")
        max_outer = int(sims.affine_friction_max_iterations) if requested_outer <= 0 else requested_outer
        if max_outer <= 0 or int(sims.affine_max_newton_iteration) <= 0:
            raise ValueError("Direct MPM--ABD iteration limits must be positive")
        tolerance = float(sims.affine_newton_tolerance)
        friction_tolerance = float(sims.affine_friction_tolerance)
        if (
            not math.isfinite(tolerance)
            or tolerance <= 0.0
            or not math.isfinite(friction_tolerance)
            or friction_tolerance < 0.0
        ):
            raise ValueError("Direct MPM--ABD tolerances must be finite and non-negative")

        if begin_step:
            self.begin_step_device(timestep)
        total_newton = 0
        outer_iterations = 0
        friction_residual = math.inf
        friction_converged = False
        try:
            energy = self.total_energy_device()
            if not math.isfinite(energy):
                raise RuntimeError("Direct MPM--ABD initial IPC energy is non-finite")
            assembly = self.assemble_linearization_device()
            active_mpm_dof = int(assembly["active_mpm_dof"])
            for _ in range(max_outer):
                if record_adjoint:
                    self.backup_lagged_friction_for_adjoint_device()
                outer_iterations += 1
                inner_converged = False
                for iteration in range(int(sims.affine_max_newton_iteration) + 1):
                    self.solve_direction_device(active_mpm_dof)
                    correction = self.direction_inf_norm(active_mpm_dof)
                    if not math.isfinite(correction):
                        raise RuntimeError("Direct MPM--ABD Newton correction is non-finite")
                    correction_velocity = correction / timestep
                    if correction == 0.0 or (iteration > 0 and correction_velocity < tolerance):
                        if self.affine.is_semi and not self.contact_converged():
                            self.advance_semi_multipliers_device()
                            energy = self.total_energy_device()
                            self.assemble_linearization_device()
                            continue
                        inner_converged = True
                        break
                    if iteration == int(sims.affine_max_newton_iteration):
                        break
                    max_step = float(sims.affine_max_step)
                    if max_step > 0.0 and correction > max_step:
                        self.scale_direction_device(max_step / correction, active_mpm_dof)
                    energy, _ = self._line_search_device(sims, energy, active_mpm_dof)
                    total_newton += 1
                if not inner_converged:
                    raise RuntimeError("Direct MPM--ABD inner Newton solve did not converge")

                self.refresh_lagged_friction_device()
                energy = self.total_energy_device()
                self.assemble_linearization_device()
                self.solve_direction_device(active_mpm_dof)
                friction_residual = self.direction_inf_norm(active_mpm_dof) / timestep
                if friction_residual < friction_tolerance:
                    friction_converged = True
                    break
            if requested_outer <= 0 and not friction_converged:
                raise RuntimeError(
                    "Direct MPM--ABD lagged friction fixed point did not "
                    f"converge (residual={friction_residual:.6e})"
                )
        except BaseException:
            self.affine.device_restore_step_start()
            self.mpm.grid_disp.fill(0.0)
            raise
        return {
            "newton_iterations": total_newton,
            "friction_iterations": outer_iterations,
            "friction_residual": friction_residual,
            "friction_converged": friction_converged,
            "energy": energy,
            "active_mpm_dof": active_mpm_dof,
        }

    def solve_lagged_equilibrium_device(self, sims, record_adjoint=False):
        if getattr(self.mpm, "has_lagged_material", False) is not True:
            return self._solve_lagged_equilibrium_device(sims, record_adjoint)

        self.mpm.begin_lagged_material_state()
        total_newton = 0
        result = None
        for material_iteration in range(self.mpm.material_lagged_max_iterations):
            result = self._solve_lagged_equilibrium_device(
                sims,
                record_adjoint,
                begin_step=material_iteration == 0,
            )
            total_newton += result["newton_iterations"]
            if self.mpm.refresh_lagged_material_state(self.mpm.grid_disp) <= self.mpm.material_lagged_tolerance:
                result["newton_iterations"] = total_newton
                result["material_lagged_iterations"] = self.mpm.last_material_lagged_iterations
                result["material_lagged_error"] = self.mpm.last_material_lagged_error
                return result
        self.affine.device_restore_step_start()
        self.mpm.grid_disp.fill(0.0)
        raise RuntimeError(
            "Direct MPM--ABD lagged MCC hardening did not converge: "
            f"error={self.mpm.last_material_lagged_error:.6e} after "
            f"{self.mpm.material_lagged_max_iterations} iterations"
        )

    def commit_step_device(self):
        """Commit the converged coupled state after any pre-commit adjoint."""
        timestep = float(self.mpm.dt)
        self.affine.device_accept_step(timestep)
        self.mpm.update_nodal_acc(self.mpm.integration)
        self.mpm.advent_particles(self.mpm.coeffPIC)
        self.ipc.ground.move(timestep)


__all__ = ["DirectAffineIPCSystem"]
