"""Legacy host soft--ABD reference iterations for isolated tests only."""

from tests.helpers.affine_contact_probe import surface_direction_inf_norm
from src.contact_detection.continuous_contact_detection import ccd_mode_parameters

import math
import taichi as ti
import numpy as np
from src.utils.SolverRuntime import accepts_monotone_energy
from src.mpdem.engines.SoftAffineIPCOperator import SoftAffineIPCOperator
from src.mpdem.engines.SoftAffineIPCEngine import SoftAffineIPCEngine


@ti.data_oriented
class ReferenceSoftAffineIPCEngine(SoftAffineIPCEngine):
    def _step_lagged(self, sims, requested_outer):
        max_outer = sims.affine_friction_max_iterations if requested_outer < 0 else requested_outer
        if max_outer <= 0:
            raise RuntimeError("SoftAffineIPC friction_max_iterations must be a positive integer")
        tolerance = float(sims.affine_friction_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise RuntimeError("SoftAffineIPC friction_tolerance must be finite and non-negative")
        self.last_newton_iterations = 0
        self.last_friction_iterations = 0
        self.last_friction_residual = math.inf
        self.last_friction_converged = False
        self.last_friction_terminated_by_cap = False
        self.last_inner_converged = False
        self.last_inner_failure_reason = ""

        self.operator.begin_step()
        y_flat = self.operator.affine_state.pack()
        self.operator.refresh_lagged_friction(y_flat)
        energy, grad = self.operator.assemble(y_flat, need_matrix=True)
        for outer_iteration in range(max_outer):
            y_flat, energy, grad, inner_iterations = self._solve_lagged_inner(sims, y_flat, energy, grad)
            self.last_newton_iterations += inner_iterations
            self.last_friction_iterations = outer_iteration + 1

            # Official lagged IPC semantics: refresh only after a complete
            # frozen-friction solve, then solve the updated system once.  The
            # resulting correction is a convergence probe and is not applied.
            self.operator.refresh_lagged_friction(y_flat)
            energy, grad = self.operator.assemble(y_flat, need_matrix=True)
            affine_direction, _, _ = self.operator.solve_direction(sims, grad)
            # The stopping residual is a velocity: the infinity
            # norm of the *unapplied, unclamped* Newton correction divided by
            # the time step.  Keeping the outer tolerance separately
            # configurable; its default is looser than the Newton tolerance.
            self.last_friction_residual = float(
                self.operator.physical_direction_inf_norm(affine_direction) / self._lagged_time_step()
            )
            if self.last_friction_residual < tolerance:
                self.last_friction_converged = True
                break
        self.last_friction_terminated_by_cap = not self.last_friction_converged
        if requested_outer < 0 and not self.last_friction_converged:
            raise RuntimeError(
                "SoftAffineIPC lagged IPC friction fixed-point iteration did "
                f"not converge within safety cap {max_outer} "
                f"(residual={self.last_friction_residual:.6e})"
            )
        self.operator.accept_step(y_flat)

    def _step_fully_implicit(self, sims):
        """Solve the coupled nonconservative residual with exact Newton."""
        self.last_newton_iterations = 0
        self.last_friction_iterations = 0
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        self.last_friction_terminated_by_cap = False
        self.last_inner_converged = False
        self.last_inner_failure_reason = ""

        self.operator.begin_step()
        initial_y = self.operator.affine_state.pack().copy()
        y_flat = initial_y.copy()
        try:
            y_flat = self.operator.initialize_fully_implicit_velocity_predictor(sims, y_flat)
            initial_energy, residual = self.operator.assemble(y_flat, need_matrix=True)
            if not np.isfinite(initial_energy):
                self.last_inner_failure_reason = "non_finite_energy"
                raise RuntimeError("SoftAffineIPC fully implicit initial state has " "non-finite IPC energy")
            if not np.all(np.isfinite(residual)):
                self.last_inner_failure_reason = "non_finite_residual"
                raise RuntimeError("SoftAffineIPC fully implicit residual is non-finite")
            y_flat, residual, iterations = self._solve_fully_implicit_newton(sims, y_flat, residual)
            self.last_newton_iterations = iterations
            self.last_friction_iterations = 1
            self.last_friction_residual = float(np.linalg.norm(residual, ord=np.inf))
            self.last_friction_converged = self.last_inner_converged
            self.operator.accept_step(y_flat)
        except Exception:
            # No particle or affine state is committed before accept_step.
            # Restore trial fields as well, so a caller may safely retry with
            # a smaller time step after a Newton/line-search failure.
            self.operator.soft_disp.fill(0.0)
            self.operator.soft_disp_base.fill(0.0)
            self.operator.affine.y.from_numpy(
                np.ascontiguousarray(
                    initial_y.reshape((self.operator.affine.control_num, 3)),
                    dtype=np.float64,
                )
            )
            raise

    def _solve_fully_implicit_newton(self, sims, y_flat, residual):
        iterations = 0
        self.last_inner_converged = False
        self.last_inner_failure_reason = "maximum_newton_iterations"
        initial_residual = float(np.linalg.norm(residual, ord=np.inf))
        tolerance = self._fully_implicit_force_target(sims, initial_residual, self.operator.dt)
        for iteration in range(int(sims.affine_max_newton_iteration)):
            if not np.all(np.isfinite(residual)):
                self.last_inner_failure_reason = "non_finite_residual"
                raise RuntimeError("SoftAffineIPC fully implicit residual is non-finite")
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if residual_norm <= tolerance:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
            affine_dir, _, direction = self.operator.solve_direction(sims, residual)
            direction = self._clamp_direction(direction, sims.affine_max_step)
            if not np.all(np.isfinite(direction)):
                self.last_inner_failure_reason = "non_finite_direction"
                raise RuntimeError("SoftAffineIPC fully implicit Newton correction is non-finite")
            if np.linalg.norm(direction, ord=np.inf) == 0.0:
                self.last_inner_failure_reason = "newton_stagnation"
                raise RuntimeError(
                    "SoftAffineIPC fully implicit Newton solve stagnated "
                    f"with residual={residual_norm:.6e}, "
                    f"tolerance={tolerance:.6e}"
                )
            jacobian_direction = self.operator.apply_jacobian(direction)
            merit_slope = float(np.dot(residual, jacobian_direction))
            if not np.isfinite(merit_slope) or merit_slope >= 0.0:
                self.last_inner_failure_reason = "non_descent_merit_direction"
                raise RuntimeError(
                    "SoftAffineIPC fully implicit Newton direction is not " "a descent direction for 0.5 * ||R||^2"
                )
            affine_dir = direction[: self.operator.affine_dof]
            soft_dir = np.zeros(self.operator.max_soft_dof, dtype=np.float64)
            soft_count = max(self.operator.total_dof - self.operator.affine_dof, 0)
            if soft_count > 0:
                soft_dir[:soft_count] = direction[self.operator.affine_dof : self.operator.total_dof]
            self.operator.soft_direction.from_numpy(soft_dir)
            alpha, new_residual = self._fully_implicit_line_search(sims, y_flat, residual, affine_dir, merit_slope)
            y_flat = y_flat + alpha * affine_dir
            residual = new_residual
            iterations = iteration + 1
            if np.linalg.norm(residual, ord=np.inf) <= tolerance:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
        if not self.last_inner_converged:
            raise RuntimeError(
                "SoftAffineIPC fully implicit Newton solve did not converge "
                f"(reason={self.last_inner_failure_reason}, "
                f"iterations={iterations}, "
                f"residual={np.linalg.norm(residual, ord=np.inf):.6e}, "
                f"tolerance={tolerance:.6e})"
            )
        return y_flat, residual, iterations

    def _fully_implicit_line_search(self, sims, y_flat, residual, affine_direction, merit_slope):
        """Armijo globalization of ``0.5 * ||R||^2``, capped by IPC CCD."""
        alpha = (
            self.operator.init_step_size(
                y_flat,
                affine_direction,
                ccd_type=sims.affine_ccd_type,
                eta=sims.affine_ccd_eta,
                accd_tolerance=sims.affine_accd_tolerance,
                max_iteration=sims.affine_ccd_max_iteration,
            )
            if sims.affine_ccd
            else 1.0
        )
        if not np.isfinite(alpha) or alpha <= 0.0:
            self.last_inner_failure_reason = "ccd_step_size"
            raise RuntimeError("SoftAffineIPC fully implicit CCD produced no feasible step")
        armijo = float(sims.affine_fully_implicit_armijo)
        contraction = float(sims.affine_fully_implicit_line_search_contraction)
        if not 0.0 < armijo < 1.0 or not 0.0 < contraction < 1.0:
            raise ValueError("SoftAffineIPC fully implicit Armijo and contraction must lie in (0, 1)")
        base_merit = 0.5 * float(np.dot(residual, residual))
        if not np.isfinite(merit_slope) or merit_slope >= 0.0:
            self.last_inner_failure_reason = "non_descent_merit_direction"
            raise RuntimeError(
                "SoftAffineIPC fully implicit Armijo requires a negative " "R^T (J p) directional derivative"
            )
        self.operator.store_soft_base()
        for _ in range(int(sims.affine_line_search_max_iteration)):
            trial = y_flat + alpha * affine_direction
            self.operator.set_soft_trial(alpha)
            trial_energy, trial_residual = self.operator.assemble(trial, need_matrix=False)
            trial_merit = 0.5 * float(np.dot(trial_residual, trial_residual))
            if (
                np.isfinite(trial_energy)
                and np.isfinite(trial_merit)
                and trial_merit <= base_merit + armijo * alpha * merit_slope
            ):
                accepted_energy, trial_residual = self.operator.assemble(trial, need_matrix=True)
                if not np.isfinite(accepted_energy):
                    self.last_inner_failure_reason = "non_finite_energy"
                    raise RuntimeError("SoftAffineIPC fully implicit accepted state has " "non-finite IPC energy")
                return alpha, trial_residual
            alpha *= contraction
        self.operator.restore_soft_base()
        self.operator.assemble(y_flat, need_matrix=True)
        self.last_inner_failure_reason = "residual_armijo_line_search"
        raise RuntimeError("SoftAffineIPC fully implicit residual Armijo line search failed")

    def _solve_lagged_inner(self, sims, y_flat, energy, grad):
        """Solve one conservative coupled problem with friction data frozen."""
        iterations = 0
        self.last_inner_converged = False
        self.last_inner_failure_reason = "maximum_newton_iterations"
        tolerance = float(sims.affine_newton_tolerance)
        dt = self._lagged_time_step()
        correction_velocity = np.inf
        self.last_inner_correction_history = []
        for iteration in range(int(sims.affine_max_newton_iteration)):
            if not np.isfinite(energy):
                self.last_inner_failure_reason = "non_finite_energy"
                raise RuntimeError("SoftAffineIPC lagged initial/current state has " "non-finite IPC energy")
            if not np.all(np.isfinite(grad)):
                self.last_inner_failure_reason = "non_finite_gradient"
                raise RuntimeError("SoftAffineIPC lagged residual is non-finite")
            affine_dir, _, direction = self.operator.solve_direction(sims, grad)
            direction = np.asarray(direction, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(direction)):
                self.last_inner_failure_reason = "non_finite_direction"
                raise RuntimeError("SoftAffineIPC lagged Newton correction is non-finite")
            # The host reference uses the same physical norm as the device loop.
            correction_velocity = self.operator.physical_direction_inf_norm(affine_dir) / dt
            self.last_inner_correction_history.append(correction_velocity)
            if correction_velocity == 0.0 or (iteration > 0 and correction_velocity < tolerance):
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
            direction = self._clamp_direction(direction, sims.affine_max_step)
            affine_dir = direction[: self.operator.affine_dof]
            soft_dir = np.zeros(self.operator.max_soft_dof, dtype=np.float64)
            soft_count = max(self.operator.total_dof - self.operator.affine_dof, 0)
            if soft_count > 0:
                soft_dir[:soft_count] = direction[self.operator.affine_dof : self.operator.total_dof]
            self.operator.soft_direction.from_numpy(soft_dir)
            step_size, new_energy, new_grad, used_affine_dir, used_direction = self._line_search(
                sims, y_flat, energy, grad, affine_dir
            )
            y_flat = y_flat + step_size * used_affine_dir
            energy, grad = new_energy, new_grad
            iterations = iteration + 1
        if not self.last_inner_converged:
            raise RuntimeError(
                "SoftAffineIPC inner Newton solve did not converge "
                f"(reason={self.last_inner_failure_reason}, "
                f"iterations={iterations}, "
                f"correction_velocity={correction_velocity:.6e} m/s, "
                f"recent_corrections="
                f"{self.last_inner_correction_history[-5:]})"
            )
        return y_flat, energy, grad, iterations

    @staticmethod
    def _clamp_direction(direction, max_step):
        norm = np.linalg.norm(direction, ord=np.inf)
        if max_step > 0.0 and norm > max_step:
            return direction * (max_step / norm)
        return direction

    def _line_search(self, sims, y_flat, energy, grad, affine_direction):
        full_direction = np.zeros(self.operator.total_dof, dtype=np.float64)
        full_direction[: self.operator.affine_dof] = affine_direction
        soft_count = self.operator.total_dof - self.operator.affine_dof
        if soft_count > 0:
            full_direction[self.operator.affine_dof : self.operator.total_dof] = (
                self.operator.soft_direction.to_numpy()[:soft_count]
            )
        slope = float(np.dot(grad, full_direction))
        if not np.isfinite(slope):
            self.last_inner_failure_reason = "non_descent_direction"
            raise RuntimeError("SoftAffineIPC lagged line-search direction has a non-finite " "energy derivative")
        if slope > 0.0:
            affine_direction = -affine_direction
            self._negate_soft_direction()
            full_direction = -full_direction
            slope = -slope
        if slope >= 0.0:
            self.last_inner_failure_reason = "non_descent_direction"
            raise RuntimeError("SoftAffineIPC lagged line search requires a strict descent " "direction")
        alpha = (
            self.operator.init_step_size(
                y_flat,
                affine_direction,
                ccd_type=sims.affine_ccd_type,
                eta=sims.affine_ccd_eta,
                accd_tolerance=sims.affine_accd_tolerance,
                max_iteration=sims.affine_ccd_max_iteration,
            )
            if sims.affine_ccd
            else 1.0
        )
        if not np.isfinite(alpha) or alpha <= 0.0:
            self.last_inner_failure_reason = "ccd_step_size"
            raise RuntimeError("SoftAffineIPC lagged CCD produced no feasible step")
        self.operator.store_soft_base()
        for _ in range(int(sims.affine_line_search_max_iteration)):
            trial = y_flat + alpha * affine_direction
            self.operator.set_soft_trial(alpha)
            trial_energy, trial_grad = self.operator.assemble(trial, need_matrix=False)
            # Standard lagged IPC backtracks the conservative incremental
            # potential monotonically. Fully implicit mode keeps the paper's
            # residual-merit Armijo globalization.
            if accepts_monotone_energy(energy, trial_energy, alpha, slope):
                self.operator.assemble(trial, need_matrix=True)
                return alpha, trial_energy, trial_grad, affine_direction, full_direction
            alpha *= 0.5
        self.operator.restore_soft_base()
        self.operator.assemble(y_flat, need_matrix=True)
        self.last_inner_failure_reason = "monotone_line_search"
        raise RuntimeError(
            "SoftAffineIPC lagged monotone line search failed to find a " "feasible non-increasing-potential step"
        )

    def _negate_soft_direction(self):
        values = -self.operator.soft_direction.to_numpy()
        self.operator.soft_direction.from_numpy(values)


import time
import math
import numpy as np
import taichi as ti
from src.utils.MatrixFunction import contraction
from src.mpdem.engines.SoftAffineIPCOperator import SoftAffineIPCOperator


@ti.data_oriented
class ReferenceSoftAffineIPCOperator(SoftAffineIPCOperator):
    def physical_direction_inf_norm(self, affine_direction):
        return max(
            surface_direction_inf_norm(self.affine, affine_direction),
            float(self._soft_physical_correction_measure()),
        )

    def _reject_cuda_host_vector_path(self, operation):
        if self.cuda_hot_loop:
            raise RuntimeError(
                f"SoftAffineIPC {operation} is a CPU reference path and is "
                "disabled in the runtime backend because it transfers "
                "complete nonlinear vectors through NumPy"
            )

    def begin_step(self):
        self._reject_cuda_host_vector_path("begin_step")
        self.affine_state.begin_step(self.dt)
        # Keep the affine cache initialized for direct operator users.  The
        # coupled driver refreshes all three caches together immediately after
        # the soft grid has been prepared.
        y_flat = self.affine_state.pack()
        self.affine.initialize_contact_damping(y_flat, self.affine_state.hat_y)
        self._prepare_soft_grid()
        self.prefix_sum.run(self.soft_node2dof)
        self.soft_active_nodes = int(self._fill_soft_dof())
        self.total_dof = self.affine_dof + 3 * self.soft_active_nodes
        self._initialize_soft_step()

    def initialize_fully_implicit_velocity_predictor(self, sims, y_flat):
        """Initialize Newton with the paper's ``v_{n+1}^0 = v_n`` state."""
        current = np.asarray(y_flat, dtype=np.float64).copy()
        affine_direction = self.affine_state.tilde_y.reshape(-1) - current
        self._build_fully_implicit_velocity_predictor()
        alpha = (
            self.init_step_size(
                current,
                affine_direction,
                ccd_type=sims.affine_ccd_type,
                eta=sims.affine_ccd_eta,
                accd_tolerance=sims.affine_accd_tolerance,
                max_iteration=sims.affine_ccd_max_iteration,
            )
            if sims.affine_ccd
            else 1.0
        )
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise RuntimeError("SoftAffineIPC fully implicit predictor produced no feasible " "positive CCD step")
        alpha = min(float(alpha), 1.0)
        self.soft_disp_base.fill(0.0)
        self._set_soft_trial(alpha)
        return current + alpha * affine_direction

    def refresh_lagged_friction(self, affine_y_flat):
        """Freeze active set, normal force and tangent basis for one inner solve."""
        self._reject_cuda_host_vector_path("refresh_lagged_friction")
        self.affine.initialize_contact_damping(affine_y_flat, self.affine_state.hat_y)
        self._build_mixed_pairs(0)
        self._update_mixed_mesh_candidates(False)
        self._reset_lagged_friction_cache()
        self._initialize_soft_lagged_friction()
        if self.affine.levelset_contact:
            self._freeze_mixed_levelset_friction_geometry()
        else:
            self._initialize_mixed_lagged_friction()
        soft_used = int(self.soft_friction_count[None])
        mixed_used = int(self.mixed_friction_count[None])
        if int(self.soft_friction_overflow[None]) != 0:
            raise RuntimeError(
                "SoftAffineIPC soft-soft lagged-friction buffer overflow: "
                f"used {soft_used}, capacity {self.soft_friction_capacity}"
            )
        if int(self.mixed_friction_overflow[None]) != 0:
            raise RuntimeError(
                "SoftAffineIPC mixed lagged-friction buffer overflow: "
                f"used {mixed_used}, capacity {self.mixed_friction_capacity}"
            )

    def _assemble_affine_self(self, affine_y_flat, need_matrix):
        affine = self.affine
        affine.y.from_numpy(np.ascontiguousarray(affine_y_flat.reshape((affine.control_num, 3)), dtype=np.float64))
        affine.tilde_y.from_numpy(
            np.ascontiguousarray(self.affine_state.tilde_y.reshape((affine.control_num, 3)), dtype=np.float64)
        )
        affine.hat_y.from_numpy(
            np.ascontiguousarray(self.affine_state.hat_y.reshape((affine.control_num, 3)), dtype=np.float64)
        )
        self._assemble_affine_self_fields(bool(need_matrix))
        grad = affine.grad.to_numpy()[: affine.control_num].reshape(-1)
        return float(affine.energy[None]), grad

    def assemble(self, affine_y_flat, need_matrix=True):
        self._reject_cuda_host_vector_path("assemble")
        tick = time.time()
        if need_matrix:
            # Line-search probes only need the energy/residual. Every Taichi
            # assembly kernel below uses a compile-time ``need_matrix`` guard,
            # so keep the last accepted Jacobian intact instead of clearing a
            # potentially large GPU HashTriplet for each rejected trial.
            self.hash_triplet.reset_system()
            tick = self._profile_stage("reset_system", tick)
        energy, affine_grad = self._assemble_affine_self(affine_y_flat, bool(need_matrix))
        tick = self._profile_stage("affine_self_assemble", tick)
        grad = np.zeros(self.max_dof, dtype=np.float64)
        grad[: self.affine_dof] = affine_grad
        self.global_grad.from_numpy(grad)
        self.energy[None] = float(energy)
        tick = self._profile_stage("load_gradient", tick)
        self._assemble_soft_energy_gradient(
            bool(need_matrix),
            bool(not self.fully_implicit),
        )
        tick = self._profile_stage("soft_energy", tick)
        self._build_mixed_pairs(0)
        tick = self._profile_stage("build_pairs", tick)
        self._update_mixed_mesh_candidates(False)
        tick = self._profile_stage("mixed_bvh", tick)
        self._assemble_soft_soft_contact(bool(need_matrix))
        tick = self._profile_stage("soft_soft_contact", tick)
        self._assemble_mixed_contact(bool(need_matrix))
        tick = self._profile_stage("soft_affine_contact", tick)
        matrix_shift = (
            float(self.sims.affine_fully_implicit_jacobian_shift)
            if self.fully_implicit
            else float(self.sims.affine_hessian_shift)
        )
        if need_matrix and matrix_shift > 0.0:
            self._add_hessian_shift(matrix_shift)
            tick = self._profile_stage("hessian_shift", tick)
        if need_matrix:
            # Residual-only line-search probes do not emit triplets.  Every
            # matrix assembly, however, must fail before a truncated Jacobian
            # can reach the linear solver.
            self._raise_hash_triplet_overflow("coupled matrix assembly")
        grad = self.global_grad.to_numpy()[: self.total_dof].copy()
        self._profile_stage("read_gradient", tick)
        return float(self.energy[None]), grad

    def solve_direction(self, sims, grad):
        self._reject_cuda_host_vector_path("solve_direction")
        tick = time.time()
        active_blocks = self.total_dof // 3
        rhs = -np.asarray(grad, dtype=np.float64).reshape((active_blocks, 3))
        self.hash_triplet.finalize_taichi_assembly()
        tick = self._profile_stage("finalize_triplet", tick)
        result = self.hash_triplet.solve(
            rhs=rhs,
            active_nodes=active_blocks,
            tol=self._linear_tolerance(sims, np.linalg.norm(grad, ord=np.inf) if self.fully_implicit else 0.0),
            maxiter=sims.affine_linear_max_iteration,
            return_solution=True,
        )
        tick = self._profile_stage(
            "bicgstab_solve" if self.fully_implicit else "pcg_solve",
            tick,
        )
        if not result["converged"]:
            raise RuntimeError(
                "SoftAffineIPC HashTriplet "
                f"{self.hash_triplet.solver} did not converge: "
                f"residual={result['residual']:.6e}, "
                f"iterations={result['iterations']}"
            )
        direction = result["x"].reshape(-1)
        affine_dir = direction[: self.affine_dof]
        soft_dir = np.zeros(self.max_soft_dof, dtype=np.float64)
        soft_count = max(self.total_dof - self.affine_dof, 0)
        if soft_count > 0:
            soft_dir[:soft_count] = direction[self.affine_dof : self.total_dof]
        self.soft_direction.from_numpy(soft_dir)
        self._profile_stage("store_direction", tick)
        return affine_dir, soft_dir[:soft_count], direction

    def apply_jacobian(self, direction):
        """Apply the assembled Jacobian to a scalar coupled direction."""
        self._reject_cuda_host_vector_path("apply_jacobian")
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        if direction.size != self.total_dof:
            raise ValueError(
                "SoftAffineIPC Jacobian direction size does not match the " "active coupled degrees of freedom"
            )
        active_blocks = self.total_dof // 3
        self.hash_triplet.x.from_numpy(self.hash_triplet._pad_vector_array(direction.reshape((active_blocks, 3))))
        nnz = int(self.hash_triplet.non_diag.element_pair_num[0])
        self.hash_triplet.matvec(
            active_blocks,
            nnz,
            self.hash_triplet.x,
            self.hash_triplet.Ax,
        )
        product = self.hash_triplet.Ax.to_numpy()[:active_blocks].reshape(-1).copy()
        if self.fully_implicit:
            # The optional diagonal shift regularizes the Newton solve but is
            # not part of the physical residual derivative used by the merit
            # function.  Remove its action after the matrix-vector product.
            shift = float(self.sims.affine_fully_implicit_jacobian_shift)
            if shift > 0.0:
                product -= shift * direction
        return product

    def accept_step(self, affine_y_flat):
        self.affine_state.accept_step(self.affine_state.unpack(affine_y_flat), self.dt)
        self._accept_soft_step()

    def init_step_size(
        self, y_flat, affine_direction, ccd_type="ccd", eta=0.2, accd_tolerance=1.0e-7, max_iteration=10000
    ):
        self._reject_cuda_host_vector_path("init_step_size")
        affine_alpha = self.affine.init_step_size(
            y_flat,
            affine_direction,
            ccd_type=ccd_type,
            eta=eta,
            accd_tolerance=accd_tolerance,
            max_iteration=max_iteration,
        )
        mixed_alpha = self._mixed_ccd_step_size(
            y_flat,
            affine_direction,
            ccd_type=ccd_type,
            eta=eta,
            accd_tolerance=accd_tolerance,
            max_iteration=max_iteration,
        )
        soft_alpha = self._soft_soft_ccd_step_size(
            ccd_type=ccd_type,
            eta=eta,
            accd_tolerance=accd_tolerance,
            max_iteration=max_iteration,
        )
        return max(0.0, min(1.0, affine_alpha, mixed_alpha, soft_alpha))

    def _mixed_ccd_step_size(
        self, y_flat, affine_direction, ccd_type="ccd", eta=0.2, accd_tolerance=1.0e-7, max_iteration=10000
    ):
        mode, eta, thickness = ccd_mode_parameters(ccd_type, eta, accd_tolerance)
        if mode in ("none", "off"):
            return 1.0
        y_np = np.ascontiguousarray(y_flat.reshape((self.affine.control_num, 3)), dtype=np.float64)
        direction_np = np.ascontiguousarray(affine_direction.reshape((self.affine.control_num, 3)), dtype=np.float64)
        self.affine.y.from_numpy(y_np)
        self.affine.direction_y.from_numpy(direction_np)
        self.affine._reconstruct_vertices()
        self.affine._reconstruct_vertex_directions()
        # CCD candidates must cover the full segment from the current
        # iterate to the proposed iterate.  A current-only body AABB misses
        # fast contacts whose endpoints both start outside ``dhat``.
        self._build_mixed_pairs(1)
        self._update_mixed_mesh_candidates(True)
        self.ccd_alpha[None] = 1.0
        if self.affine.levelset_contact:
            self._compute_mixed_levelset_ccd_alpha(float(eta), min(int(max_iteration), 64))
        elif self.mixed_bvh is not None:
            self._compute_mixed_ccd_alpha(float(eta), thickness, int(max_iteration), mode == "accd")
        return float(self.ccd_alpha[None])
