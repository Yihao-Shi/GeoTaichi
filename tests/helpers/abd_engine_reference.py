"""Host ABD Newton/line-search oracles for isolated tests only."""

import numpy as np

from src.utils.SolverRuntime import accepts_monotone_energy

from src.dem.engines.AffineBodyOperator import (
    MATRIX_COO,
    MATRIX_HASH_TRIPLET,
    _normalize_affine_assemble_type,
    _normalize_affine_friction_mode,
)


from src.dem.engines.AffineBodyEngine import AffineBodyEngine


class ReferenceAffineBodyEngine(AffineBodyEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_surface_direction_norm = lambda direction: float(np.linalg.norm(direction, ord=np.inf))

    def _surface_direction_inf_norm(self, direction):
        return float(self.evaluate_surface_direction_norm(direction))

    def _step_once(self, sims, scene):
        self.last_line_search_accepted = False
        self.last_line_search_alpha = np.nan
        self.last_line_search_backtracks = 0
        self.step_line_search_calls = 0
        self.step_line_search_min_alpha = np.nan
        self.step_line_search_max_backtracks = 0
        self.step_line_search_converged = True
        dt = float(sims.dt[None])
        mode = _normalize_affine_friction_mode(sims.affine_friction_mode)
        operator_fully_implicit = self.operator.fully_implicit
        if (mode == "fully_implicit") != operator_fully_implicit:
            raise RuntimeError(
                "AffineBody friction_mode cannot be changed after engine "
                "initialization; configure it before add_essentials()"
            )
        device_path = self.device_nonlinear_path
        if device_path:
            self.operator.device_backup_step_start()
            self.operator.device_begin_step(dt)
            y_flat = None
        else:
            self.state.begin_step(dt)
            y_flat = self.state.pack()
        if self.operator.contact_damping_stiffness > 0.0:
            self._ensure_linear_solver(sims)
        if mode == "fully_implicit":
            return self._step_fully_implicit(sims, y_flat, dt)
        raw_outer = sims.affine_friction_iterations
        requested_outer = int(raw_outer)
        if not np.isfinite(float(raw_outer)) or float(raw_outer) != requested_outer:
            raise RuntimeError("AffineBody friction_iterations must be an integer")
        requested_outer = -1 if requested_outer <= 0 else requested_outer
        max_outer = sims.affine_friction_max_iterations if requested_outer < 0 else requested_outer
        if max_outer <= 0:
            raise RuntimeError("AffineBody friction_max_iterations must be a positive integer")
        tolerance = float(sims.affine_friction_tolerance)
        self.last_newton_iterations = 0
        self.last_friction_iterations = 0
        self.last_friction_outer_iteration = 0
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        self.last_friction_terminated_by_cap = False
        self.last_inner_converged = False
        self.last_inner_failure_reason = ""
        self.last_inner_residual = np.inf

        if device_path:
            self.last_device_nonlinear_path = True
            try:
                return self._step_lagged_cuda(
                    sims,
                    dt,
                    requested_outer=requested_outer,
                    max_outer=max_outer,
                    tolerance=tolerance,
                )
            except BaseException as exception:
                try:
                    self.operator.device_restore_step_start()
                except BaseException as rollback_error:
                    # Keep the solver failure visible; a CUDA rollback error
                    # must not mask the contact/assembly exception that caused
                    # the rejected step.
                    raise exception from rollback_error
                raise
        self.last_device_nonlinear_path = False

        self.operator.initialize_contact_damping(y_flat, self.state.hat_y)
        energy, grad = self._assemble_system(sims, y_flat, need_matrix=True)
        for outer_iteration in range(max_outer):
            self.last_friction_outer_iteration = outer_iteration + 1
            y_flat, energy, grad, inner_iterations = self._solve_lagged_inner(sims, y_flat, energy, grad)
            self.last_newton_iterations += inner_iterations
            self.last_friction_iterations = outer_iteration + 1

            # Match reference IPC's fricIterAmt loop: after every complete
            # frozen-friction solve (including fricIterAmt=1), rebuild the
            # lagged contact data and solve the updated linear system once.
            # This probe is used only for fixed-point convergence; it is not
            # applied to the state.
            self.operator.initialize_contact_damping(y_flat, self.state.hat_y)
            energy, grad = self._assemble_system(sims, y_flat, need_matrix=True)
            updated_direction = self._solve_direction(sims, grad)
            self.last_friction_residual = float(self._surface_direction_inf_norm(updated_direction) / dt)
            if self.last_friction_residual < tolerance:
                self.last_friction_converged = True
                break

        self.last_friction_terminated_by_cap = not self.last_friction_converged
        if requested_outer < 0 and not self.last_friction_converged:
            raise RuntimeError(
                "Affine lagged IPC friction fixed-point iteration did not "
                f"converge within safety cap {max_outer} "
                f"(residual={self.last_friction_residual:.6e})"
            )

        self.state.accept_step(self.state.unpack(y_flat), dt)

    def _step_fully_implicit(self, sims, previous_y, dt):
        """Run one transactional fully implicit step.

        A rejected Newton/line-search trial updates the operator's Taichi
        configuration even though ``AffineBodyState`` is not accepted.  Put
        the operator back at the last accepted configuration so callers can
        safely reduce ``dt`` and retry.  Recovery is best-effort and must
        never hide the original solver exception.
        """
        try:
            return self._step_fully_implicit_impl(sims, previous_y, dt)
        except Exception as exception:
            try:
                if self.last_device_nonlinear_path:
                    self.operator.device_restore_step_start()
                    self._assemble_system_device(sims, need_matrix=False)
                else:
                    self._assemble_system(sims, np.asarray(previous_y).copy(), need_matrix=False)
            except Exception as rollback_error:
                raise exception from rollback_error
            raise

    def _step_fully_implicit_impl(self, sims, previous_y, dt):
        """Solve the paper's nonconservative residual with exact Newton."""
        self.last_newton_iterations = 0
        self.last_friction_iterations = 1
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        self.last_friction_terminated_by_cap = False
        self.last_inner_converged = False
        self.last_inner_failure_reason = ""
        self.last_inner_residual = np.inf
        self.last_fully_implicit_merit_slope = np.nan

        if self.device_nonlinear_path:
            self.last_device_nonlinear_path = True
            return self._step_fully_implicit_cuda(sims, previous_y, dt)
        self.last_device_nonlinear_path = False

        self.operator.initialize_contact_damping(previous_y, self.state.hat_y)

        # Paper initial guess v_{n+1}^{(0)} = v_n.  In displacement form this
        # is q^{(0)} = q_n + h v_n = tilde_q.  IPC CCD caps the predictor so
        # the very first residual evaluation remains strictly feasible.
        predictor = self.state.pack(self.state.tilde_y)
        predictor_direction = predictor - previous_y
        if np.linalg.norm(predictor_direction, ord=np.inf) > 0.0:
            alpha = self._ccd_step_size(sims, previous_y, predictor_direction)
            if not np.isfinite(alpha) or alpha <= 0.0:
                raise RuntimeError("Affine fully implicit predictor CCD returned an " "invalid/non-positive step")
            y_flat = previous_y + alpha * predictor_direction
        else:
            y_flat = previous_y.copy()

        initial_energy, residual = self._assemble_system(sims, y_flat, need_matrix=True)
        if not np.isfinite(initial_energy):
            self.last_inner_failure_reason = "non_finite_energy"
            raise RuntimeError("Affine fully implicit initial state has non-finite IPC " "energy")
        if not np.all(np.isfinite(residual)):
            self.last_inner_failure_reason = "non_finite_residual"
            raise RuntimeError("Affine fully implicit residual is non-finite")
        iterations = 0
        initial_residual = float(np.linalg.norm(residual, ord=np.inf))
        tolerance = self._fully_implicit_force_target(sims, initial_residual, dt)
        self.last_inner_failure_reason = "maximum_newton_iterations"
        for iteration in range(int(sims.affine_max_newton_iteration)):
            if not np.all(np.isfinite(residual)):
                self.last_inner_failure_reason = "non_finite_residual"
                raise RuntimeError("Affine fully implicit residual is non-finite")
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if residual_norm <= tolerance:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
            direction = self._solve_direction(sims, residual)
            direction = self._clamp_direction(direction, sims.affine_max_step)
            if not np.all(np.isfinite(direction)):
                self.last_inner_failure_reason = "non_finite_direction"
                raise RuntimeError("Affine fully implicit Newton correction is non-finite")
            if np.linalg.norm(direction, ord=np.inf) <= np.finfo(float).eps:
                self.last_inner_failure_reason = "newton_stagnation"
                raise RuntimeError("Affine fully implicit Newton stagnated before the " "residual tolerance was met")
            jacobian_direction = self._fully_implicit_jacobian_direction(sims, direction)
            alpha, new_residual = self._fully_implicit_line_search(
                sims, y_flat, residual, direction, jacobian_direction
            )
            y_flat = y_flat + alpha * direction
            residual = new_residual
            iterations = iteration + 1
            if np.linalg.norm(residual, ord=np.inf) <= tolerance:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break

        self.last_newton_iterations = iterations
        self.last_friction_residual = float(np.linalg.norm(residual, ord=np.inf))
        self.last_friction_converged = self.last_inner_converged
        if not self.last_inner_converged:
            raise RuntimeError(
                "Affine fully implicit Newton solve did not converge "
                f"(reason={self.last_inner_failure_reason}, "
                f"iterations={iterations}, "
                f"residual={self.last_friction_residual:.6e}, "
                f"tolerance={tolerance:.6e})"
            )
        self.state.accept_step(self.state.unpack(y_flat), dt)

    def _fully_implicit_line_search(
        self,
        sims,
        y_flat,
        residual,
        direction,
        jacobian_direction,
    ):
        """Residual-merit Armijo globalization preceded by IPC CCD.

        The merit is ``phi(q) = 0.5 * ||R(q)||^2`` and its exact directional
        derivative is ``R^T (J p)``.  Computing that product from the
        assembled Jacobian is essential after max-step clamping, an optional
        Jacobian shift, or an inexact HashTriplet solve; in those cases one
        cannot assume ``J p == -R``.
        """
        alpha = self._ccd_step_size(sims, y_flat, direction)
        if not np.isfinite(alpha) or alpha <= 0.0:
            self._record_line_search(False)
            self.last_inner_failure_reason = "ccd_step_size"
            raise RuntimeError("Affine fully implicit CCD produced no feasible step")
        armijo = float(sims.affine_fully_implicit_armijo)
        contraction = float(sims.affine_fully_implicit_line_search_contraction)
        if not 0.0 < armijo < 1.0 or not 0.0 < contraction < 1.0:
            raise ValueError("Affine fully implicit Armijo and contraction must lie " "in (0, 1)")
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        jacobian_direction = np.asarray(jacobian_direction, dtype=np.float64).reshape(-1)
        base_merit = 0.5 * float(np.dot(residual, residual))
        merit_slope = float(np.dot(residual, jacobian_direction))
        self.last_fully_implicit_merit_slope = merit_slope
        if not np.isfinite(base_merit) or not np.isfinite(merit_slope):
            self._record_line_search(False)
            self.last_inner_failure_reason = "non_finite_residual_merit"
            raise RuntimeError("Affine fully implicit residual merit or directional " "derivative is non-finite")
        if merit_slope >= 0.0:
            self._record_line_search(False)
            self.last_inner_failure_reason = "non_descent_residual_merit_direction"
            raise RuntimeError(
                "Affine fully implicit Newton correction is not a descent " "direction for 0.5 * ||R||^2"
            )
        for backtracks in range(int(sims.affine_line_search_max_iteration)):
            trial = y_flat + alpha * direction
            trial_energy, trial_residual = self._assemble_system(sims, trial, need_matrix=False)
            trial_residual = np.asarray(trial_residual, dtype=np.float64).reshape(-1)
            trial_merit = 0.5 * float(np.dot(trial_residual, trial_residual))
            if (
                np.isfinite(trial_energy)
                and np.isfinite(trial_merit)
                and trial_merit <= base_merit + armijo * alpha * merit_slope
            ):
                accepted_energy, trial_residual = self._assemble_system(sims, trial, need_matrix=True)
                if not np.isfinite(accepted_energy):
                    self._record_line_search(False, backtracks=backtracks)
                    self.last_inner_failure_reason = "non_finite_energy"
                    raise RuntimeError("Affine fully implicit accepted state has " "non-finite IPC energy")
                self._record_line_search(True, alpha, backtracks)
                return alpha, trial_residual
            alpha *= contraction
        self._assemble_system(sims, y_flat, need_matrix=True)
        self._record_line_search(False, backtracks=int(sims.affine_line_search_max_iteration))
        self.last_inner_failure_reason = "residual_armijo_line_search"
        raise RuntimeError("Affine fully implicit residual Armijo line search failed")

    def _solve_lagged_inner(self, sims, y_flat, energy, grad):
        """Solve one conservative IPC problem with frozen friction data.

        Reference IPC's nonlinear tolerance has velocity units: convergence is
        ``||delta_x||_inf / dt < tol`` for the raw Newton correction.  An
        absolute gradient test is not equivalent because barrier stiffness and
        quadrature weights arbitrarily scale that gradient.
        """
        direction = np.zeros_like(y_flat)
        iterations = 0
        self.last_inner_converged = False
        self.last_inner_failure_reason = "maximum_newton_iterations"
        self.last_inner_residual = np.inf
        self.last_inner_control_residual = np.inf
        self.last_inner_correction_history = []
        dt = float(self.operator.dt) if self.operator is not None else 1.0
        maximum_updates = int(sims.affine_max_newton_iteration)
        for iteration in range(maximum_updates + 1):
            if not np.isfinite(energy):
                self.last_inner_failure_reason = "non_finite_energy"
                raise RuntimeError(
                    "Affine lagged IPC energy is non-finite; the current " "configuration is not strictly feasible"
                )
            if not np.all(np.isfinite(grad)):
                self.last_inner_failure_reason = "non_finite_gradient"
                raise RuntimeError("Affine lagged Newton residual is non-finite")
            direction = self._solve_direction(sims, grad)
            if not np.all(np.isfinite(direction)):
                self.last_inner_failure_reason = "non_finite_direction"
                raise RuntimeError("Affine lagged Newton correction is non-finite")
            self.last_inner_residual = float(self._surface_direction_inf_norm(direction) / dt)
            self.last_inner_control_residual = float(np.linalg.norm(direction, ord=np.inf) / dt)
            self.last_inner_correction_history.append(self.last_inner_residual)
            # Reference IPC evaluates convergence from the previous Newton
            # correction and therefore never discards a nonzero first update
            # (``k && gradVanish``).  Applying that first update is required
            # to leave the mollified stick branch when ``tol > epsv``.
            if self.last_inner_residual == 0.0 or (
                iteration > 0 and self.last_inner_residual < sims.affine_newton_tolerance
            ):
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
            # ``maximum_updates`` counts accepted Newton updates. Probe the
            # state produced by the last allowed update once more, but never
            # apply an unbudgeted extra correction.
            if iteration == maximum_updates:
                break
            direction = self._clamp_direction(direction, sims.affine_max_step)
            # Keep the direction used by CCD/line search identical to the one
            # committed below.  Reversing it only inside ``_line_search``
            # would validate one trial and then apply the opposite update.
            direction_slope = float(np.dot(grad, direction))
            if np.isfinite(direction_slope) and direction_slope > 0.0:
                direction = -direction
            step_size, new_energy, new_grad = self._line_search(sims, y_flat, energy, grad, direction)
            y_flat = y_flat + step_size * direction
            energy, grad = new_energy, new_grad
            iterations = iteration + 1
        if not self.last_inner_converged:
            raise RuntimeError(
                "Affine IPC inner Newton solve did not converge "
                f"(reason={self.last_inner_failure_reason}, "
                f"time_step={sims.current_step}, "
                f"outer_iteration={self.last_friction_outer_iteration}, "
                f"iterations={iterations}, "
                f"correction_velocity={self.last_inner_residual:.6e}, "
                f"recent_corrections="
                f"{self.last_inner_correction_history[-5:]}, "
                f"linear_method={self.last_linear_method}, "
                f"linear_residual={self.last_linear_residual:.6e}, "
                f"original_linear_residual="
                f"{self.last_linear_original_residual:.6e}, "
                f"lambda_min={self.last_linear_min_eigenvalue:.6e}, "
                f"relative_negative="
                f"{self.last_linear_relative_negative_curvature:.6e}, "
                f"spectral_resolution="
                f"{self.last_linear_spectral_resolution:.6e}, "
                f"unresolved_modes="
                f"{self.last_linear_unresolved_modes}, "
                f"projection_correction="
                f"{self.last_linear_projection_correction:.6e})"
            )
        return y_flat, energy, grad, iterations

    def _assemble_system(self, sims, y_flat, need_matrix=True):
        y = self.state.unpack(y_flat)
        matrix_mode = self._matrix_mode(sims) if need_matrix else MATRIX_COO
        direct_threshold = int(sims.affine_direct_hessian_dofs)
        separate_inertia_matrix = bool(
            need_matrix
            and matrix_mode == MATRIX_COO
            and not self.operator.fully_implicit
            and direct_threshold > 0
            and self.operator.dof <= direct_threshold
        )
        if need_matrix:
            self._ensure_linear_solver(sims)
            if matrix_mode == MATRIX_HASH_TRIPLET:
                self.hash_triplet.reset_system()
        energy, grad = self.operator.assemble(
            y,
            self.state.tilde_y,
            self.state.hat_y,
            need_matrix=need_matrix,
            matrix_mode=matrix_mode,
            include_inertia_matrix=not separate_inertia_matrix,
        )
        self.last_candidate_pairs = self.operator.last_candidate_pairs
        return energy, grad

    def _solve_direction(self, sims, grad):
        backend_label = _normalize_affine_assemble_type(sims.affine_assemble_type)
        backend = backend_label.upper()
        self.last_linear_backend = backend_label
        self.last_linear_method = None
        self.last_linear_min_eigenvalue = np.nan
        self.last_linear_relative_negative_curvature = np.nan
        self.last_linear_projection_correction = 0.0
        self.last_linear_spectral_resolution = np.nan
        self.last_linear_unresolved_modes = 0
        self.last_linear_original_residual = np.inf
        matrix_mode = self._matrix_mode(sims)
        rhs = -grad
        if self.operator.fully_implicit:
            self._ensure_linear_solver(sims)
            if backend == "HASHTRIPLET":
                return self._solve_hash_triplet(sims, rhs)
            self.last_linear_method = "SparseDirect"
            from scipy.sparse.linalg import spsolve

            matrix = self.coo_matrix._to_scipy().tocsc()
            direction = np.asarray(spsolve(matrix, rhs), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(direction)):
                raise RuntimeError(
                    "Affine fully implicit nonsymmetric direct solve " "returned a non-finite correction"
                )
            self.last_linear_converged = True
            self.last_linear_initial_residual = float(np.linalg.norm(rhs))
            self.last_linear_residual = float(np.linalg.norm(matrix @ direction - rhs))
            self.last_linear_original_residual = self.last_linear_residual
            self.last_linear_iterations = 1
            self.last_linear_failure_reason = ""
            return direction
        base_shift = max(float(sims.affine_hessian_shift), 1.0e-12)
        self._ensure_linear_solver(sims)
        last_error = None
        for attempt in range(6):
            try:
                if backend == "HASHTRIPLET":
                    return self._solve_hash_triplet(sims, rhs)
                direct_threshold = int(sims.affine_direct_hessian_dofs)
                if direct_threshold > 0 and self.operator.dof <= direct_threshold:
                    return self._solve_coo_direct(sims, rhs)
                return self._solve_coo_pcg(sims, rhs, base_shift)
            except AssertionError as exc:
                last_error = exc
            except RuntimeError as exc:
                if "did not converge" not in str(exc):
                    raise
                last_error = exc
            if attempt == 5:
                break
            self.operator._add_hessian_shift(matrix_mode, base_shift * (10.0**attempt))
            if matrix_mode == MATRIX_COO:
                self.operator.finalize_coo_assembly()
        raise RuntimeError(
            f"Affine {backend} linear solve did not converge after " "projected-Hessian regularization."
        ) from last_error

    def _fully_implicit_jacobian_direction(self, sims, direction):
        """Return the exact residual-Jacobian product ``J @ direction``.

        The matrix used by the linear solve may include the optional explicit
        diagonal regularization.  Remove that contribution so the Armijo
        slope remains the derivative of the actual residual merit.
        """
        backend = _normalize_affine_assemble_type(sims.affine_assemble_type)
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        if backend == "HashTriplet":
            # Keep the HashTriplet path on the Taichi device. Building a
            # SciPy matrix here repeated block expansion and a device-to-host
            # transfer solely for one Jp product in the residual merit slope.
            self.hash_triplet.x.from_numpy(
                self.hash_triplet._pad_vector_array(direction.reshape((self.operator.control_num, 3)))
            )
            nnz = int(self.hash_triplet.non_diag.element_pair_num[0])
            self.hash_triplet.matvec(
                self.operator.control_num,
                nnz,
                self.hash_triplet.x,
                self.hash_triplet.Ax,
            )
            product = self.hash_triplet.Ax.to_numpy()[: self.operator.control_num].reshape(-1).copy()
        else:
            matrix = self.coo_matrix._to_scipy()
            product = np.asarray(matrix @ direction, dtype=np.float64).reshape(-1)
        shift = float(self.operator._active_jacobian_shift())
        if shift != 0.0:
            product -= shift * direction
        if not np.all(np.isfinite(product)):
            self.last_inner_failure_reason = "non_finite_jacobian_direction"
            raise RuntimeError("Affine fully implicit residual Jacobian-vector product " "is non-finite")
        return product

    def _solve_coo_pcg(self, sims, rhs, min_preconditioner):
        """CPU/Metal oracle for the same SPD lagged system used on CUDA."""
        self.last_linear_method = "PCG"
        self.operator.prepare_linear_fields(rhs, min_preconditioner)
        succeeded = self.coo_matrix.solve(
            self.operator.linear_rhs,
            self.operator.linear_x,
            self.operator.linear_M,
            tol=sims.affine_linear_tolerance,
            maxiter=sims.affine_linear_max_iteration,
        )
        solver = self.coo_matrix.linear_solver
        self.last_linear_converged = bool(succeeded)
        self.last_linear_initial_residual = float(solver.last_initial_residual)
        self.last_linear_residual = float(solver.last_residual)
        self.last_linear_original_residual = self.last_linear_residual
        self.last_linear_iterations = int(solver.last_iterations)
        self.last_linear_failure_reason = str(solver.last_breakdown_reason)
        if not succeeded:
            raise RuntimeError(
                "Affine COO PCG did not converge: "
                f"initial_residual={self.last_linear_initial_residual:.6e}, "
                f"residual={self.last_linear_residual:.6e}, "
                f"iterations={self.last_linear_iterations}, "
                f"reason={self.last_linear_failure_reason or 'unknown'}"
            )
        return self.operator.coo_solution_numpy()

    def _solve_coo_direct(self, sims, rhs):
        """Solve a configured small lagged projected-Newton system.

        ``direct_hessian_dofs`` has always been part of the public AffineBody
        configuration, but historically it was ignored and every CPU/Metal
        system went through scalar-Jacobi PCG. Near an IPC barrier the Hessian
        can be so stiff that atomic COO matvec summation reaches its residual
        floor before an absolute Krylov tolerance, even for a 12-DOF body.
        A dense symmetric eigensolve is both faster and more accurate for the
        explicitly configured small-system range.

        Every non-inertial lagged block is projected to PSD before pullback.
        The COO assembler therefore keeps that block separate from the exact
        control-space mass matrix for this path. We whiten by the mass
        Cholesky factor, project only round-off-level negative eigenvalues of
        the non-inertial block, and add the identity in spectral coordinates.
        This is algebraically the same ``M + K_psd`` Newton system, but it
        cannot lose O(1) inertia modes when a barrier reaches O(1e17).
        Material negative curvature is still rejected. CUDA production
        systems remain on the device-resident HashTriplet PCG path.
        """

        self.last_linear_method = "DenseMassWhitenedProjectedEigen"
        noninertial_matrix = np.asarray(self.coo_matrix._to_scipy().toarray(), dtype=np.float64)
        mass_matrix = getattr(self.operator, "control_mass_matrix_np", None)
        if mass_matrix is None:
            mass_matrix = np.zeros((self.operator.dof, self.operator.dof), dtype=np.float64)
            for body_id, body in enumerate(self.operator.state.bodies):
                for component in range(3):
                    indices = body_id * 12 + 3 * np.arange(4, dtype=np.int64) + component
                    mass_matrix[np.ix_(indices, indices)] = body["mass_matrix"]
        mass_matrix = np.asarray(mass_matrix, dtype=np.float64)
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if (
            noninertial_matrix.shape != (self.operator.dof, self.operator.dof)
            or mass_matrix.shape != (self.operator.dof, self.operator.dof)
            or rhs.size != self.operator.dof
            or not np.all(np.isfinite(noninertial_matrix))
            or not np.all(np.isfinite(mass_matrix))
            or not np.all(np.isfinite(rhs))
        ):
            raise RuntimeError("Affine COO direct solve did not converge: non-finite or " "inconsistent linear system")

        matrix_norm = float(np.linalg.norm(noninertial_matrix, ord=np.inf))
        skew_norm = float(
            np.linalg.norm(
                noninertial_matrix - noninertial_matrix.T,
                ord=np.inf,
            )
        )
        symmetry_scale = max(matrix_norm, np.finfo(np.float64).tiny)
        symmetry_error = skew_norm / symmetry_scale
        symmetry_tolerance = max(
            100.0 * np.finfo(np.float64).eps * max(self.operator.dof, 1),
            1.0e-12,
        )
        if symmetry_error > symmetry_tolerance:
            raise RuntimeError(
                "Affine lagged projected Hessian is not symmetric " f"(relative_skew={symmetry_error:.6e})"
            )

        symmetric_noninertial = 0.5 * (noninertial_matrix + noninertial_matrix.T)
        try:
            mass_cholesky = np.linalg.cholesky(0.5 * (mass_matrix + mass_matrix.T))
            left_whitened = np.linalg.solve(mass_cholesky, symmetric_noninertial)
            whitened_noninertial = np.linalg.solve(mass_cholesky, left_whitened.T).T
            whitened_noninertial = 0.5 * (whitened_noninertial + whitened_noninertial.T)
            eigenvalues, eigenvectors = np.linalg.eigh(whitened_noninertial)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Affine COO direct solve did not converge: mass "
                "factorization or projected-Hessian eigendecomposition "
                "failed"
            ) from exc

        minimum_eigenvalue = float(eigenvalues.min())
        spectral_scale = max(
            float(np.max(np.abs(eigenvalues))),
            np.finfo(np.float64).tiny,
        )
        relative_negative_curvature = max(-minimum_eigenvalue, 0.0) / spectral_scale
        # A formed f64 matrix cannot resolve eigenvalues below the backward
        # error of symmetric eigendecomposition/assembly. Treat both signs in
        # this band as unresolved. Clamping only negative noise while keeping
        # an equally plausible positive O(ulp) mode creates a false stiffness
        # and can make the Newton correction spuriously vanish.
        spectral_resolution = 8.0 * max(self.operator.dof, 1) * np.finfo(np.float64).eps * spectral_scale
        self.last_linear_spectral_resolution = spectral_resolution
        self.last_linear_min_eigenvalue = minimum_eigenvalue
        self.last_linear_relative_negative_curvature = relative_negative_curvature
        if minimum_eigenvalue < -spectral_resolution:
            raise RuntimeError(
                "Affine lagged projected Hessian contains material negative "
                "curvature "
                f"(lambda_min={minimum_eigenvalue:.6e}, "
                f"spectral_resolution={spectral_resolution:.6e}, "
                f"relative_negative={relative_negative_curvature:.6e})"
            )

        unresolved = np.abs(eigenvalues) <= spectral_resolution
        corrected_eigenvalues = np.where(unresolved, 0.0, eigenvalues)
        self.last_linear_unresolved_modes = int(np.count_nonzero(unresolved))
        self.last_linear_projection_correction = float(np.max(np.abs(corrected_eigenvalues - eigenvalues)))
        whitened_rhs = np.linalg.solve(mass_cholesky, rhs)
        projected_rhs = eigenvectors.T @ whitened_rhs
        whitened_direction = eigenvectors @ (projected_rhs / (1.0 + corrected_eigenvalues))
        direction = np.linalg.solve(mass_cholesky.T, whitened_direction)

        # Verify the mass-whitened system actually used for the projected
        # Newton direction. The original residual is evaluated as two
        # separate products so the diagnostic does not first erase M in a
        # dense ``M + K`` addition.
        projected_residual_whitened = (
            eigenvectors @ ((1.0 + corrected_eigenvalues) * (eigenvectors.T @ whitened_direction)) - whitened_rhs
        )
        projected_residual_vector = mass_cholesky @ projected_residual_whitened
        original_residual_vector = mass_matrix @ direction + symmetric_noninertial @ direction - rhs
        residual = float(np.linalg.norm(projected_residual_vector))
        self.last_linear_original_residual = float(np.linalg.norm(original_residual_vector))
        denominator = (1.0 + float(np.max(corrected_eigenvalues))) * float(
            np.linalg.norm(whitened_direction, ord=np.inf)
        ) + float(np.linalg.norm(whitened_rhs, ord=np.inf))
        backward_error = (
            float(np.linalg.norm(projected_residual_whitened, ord=np.inf)) / denominator if denominator > 0.0 else 0.0
        )
        backward_tolerance = max(
            float(sims.affine_linear_tolerance),
            100.0 * np.finfo(np.float64).eps * max(self.operator.dof, 1),
        )
        self.last_linear_initial_residual = float(np.linalg.norm(rhs))
        self.last_linear_residual = residual
        self.last_linear_iterations = 1
        self.last_linear_converged = bool(
            np.all(np.isfinite(direction)) and np.isfinite(backward_error) and backward_error <= backward_tolerance
        )
        self.last_linear_failure_reason = "" if self.last_linear_converged else "backward_error"
        if not self.last_linear_converged:
            raise RuntimeError(
                "Affine COO direct solve did not converge: "
                f"residual={residual:.6e}, "
                f"backward_error={backward_error:.6e}, "
                f"tolerance={backward_tolerance:.6e}, "
                f"lambda_min={minimum_eigenvalue:.6e}, "
                f"projection_correction="
                f"{self.last_linear_projection_correction:.6e}"
            )
        return direction

    def _solve_hash_triplet(self, sims, rhs):
        self.last_linear_method = "BiCGSTAB" if self.operator.fully_implicit else "PCG"
        active_nodes = self.operator.control_num
        self.maximum_hash_triplets = max(self.maximum_hash_triplets, int(self.hash_triplet.raw_non_diag_count[0]))
        self.hash_triplet.finalize_taichi_assembly()
        result = self.hash_triplet.solve(
            rhs=rhs.reshape((active_nodes, 3)),
            active_nodes=active_nodes,
            tol=self._linear_tolerance(sims, float(np.linalg.norm(rhs, ord=np.inf))),
            maxiter=sims.affine_linear_max_iteration,
            return_solution=True,
        )
        self.last_linear_converged = bool(result["converged"])
        self.last_linear_initial_residual = float(np.linalg.norm(rhs))
        self.last_linear_residual = float(result["residual"])
        self.last_linear_original_residual = self.last_linear_residual
        self.last_linear_iterations = int(result["iterations"])
        self.last_linear_failure_reason = "" if result["converged"] else "maximum_iterations_or_breakdown"
        if not result["converged"]:
            method = "BiCGSTAB" if self.operator.fully_implicit else "PCG"
            raise RuntimeError(
                f"Affine HashTriplet {method} did not converge: "
                f"residual={result['residual']:.6e}, "
                f"iterations={result['iterations']}"
            )
        return result["x"].reshape(-1)

    @staticmethod
    def _clamp_direction(direction, max_step):
        norm = np.linalg.norm(direction, ord=np.inf)
        if max_step > 0.0 and norm > max_step:
            return direction * (max_step / norm)
        return direction

    def _line_search(self, sims, y_flat, energy, grad, direction):
        slope = float(np.dot(grad, direction))
        if not np.isfinite(slope) or slope >= 0.0:
            self._record_line_search(False)
            self.last_inner_failure_reason = "non_descent_direction"
            raise RuntimeError("Affine IPC line search requires a descent direction")
        alpha = self._ccd_step_size(sims, y_flat, direction)
        if not np.isfinite(alpha) or alpha <= 0.0:
            self._record_line_search(False)
            self.last_inner_failure_reason = "ccd_step_size"
            raise RuntimeError("Affine lagged IPC CCD produced no strictly feasible step")
        self.last_line_search_initial_alpha = float(alpha)
        self.last_line_search_slope = slope
        self.last_line_search_base_energy = float(energy)
        self.last_line_search_trials = []
        for backtracks in range(int(sims.affine_line_search_max_iteration)):
            trial = y_flat + alpha * direction
            trial_energy, trial_grad = self._assemble_system(sims, trial, need_matrix=False)
            self.last_line_search_trials.append((float(alpha), float(trial_energy)))
            # Use conservative-potential backtracking ``E(x+αp) <= E(x)``.
            # Residual-merit Armijo is reserved for fully implicit friction.
            if np.isfinite(trial_energy) and accepts_monotone_energy(energy, trial_energy, alpha, slope):
                self._assemble_system(sims, trial, need_matrix=True)
                self._record_line_search(True, alpha, backtracks)
                return alpha, trial_energy, trial_grad
            alpha *= 0.5
        # IPC relies on the line search to preserve strict feasibility and
        # monotone potential decrease. Never accept alpha=0 or an
        # resolvably energy-increasing fallback.
        self._assemble_system(sims, y_flat, need_matrix=True)
        self._record_line_search(False, backtracks=len(self.last_line_search_trials))
        self.last_inner_failure_reason = "monotone_line_search"
        raise RuntimeError(
            "Affine lagged IPC monotone line search failed "
            f"(base_energy={energy:.17e}, slope={slope:.17e}, "
            f"initial_alpha={self.last_line_search_initial_alpha:.17e}, "
            f"trials={self.last_line_search_trials[-8:]}, "
            f"recent_corrections={self.last_inner_correction_history[-5:]}, "
            f"linear_method={self.last_linear_method}, "
            f"linear_residual={self.last_linear_residual:.6e}, "
            f"original_linear_residual="
            f"{self.last_linear_original_residual:.6e}, "
            f"lambda_min={self.last_linear_min_eigenvalue:.6e}, "
            f"relative_negative="
            f"{self.last_linear_relative_negative_curvature:.6e}, "
            f"projection_correction="
            f"{self.last_linear_projection_correction:.6e})"
        )

    def _ccd_step_size(self, sims, y_flat, direction):
        self.last_ccd_type = str(sims.affine_ccd_type)
        if not sims.affine_ccd:
            self.last_ccd_step = 1.0
            return 1.0
        alpha = self.operator.init_step_size(
            y_flat,
            direction,
            ccd_type=sims.affine_ccd_type,
            eta=sims.affine_ccd_eta,
            accd_tolerance=sims.affine_accd_tolerance,
            max_iteration=sims.affine_ccd_max_iteration,
        )
        self.last_ccd_step = alpha
        return alpha
