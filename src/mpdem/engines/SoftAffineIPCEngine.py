"""Nonlinear solution policy for soft-particle/affine-body IPC."""

import math
import os

import numpy as np

from src.dem.engines.AffineBodyOperator import _normalize_affine_assemble_type
from src.mpm.MaterialManager import SoftParticleMaterialManager
from src.utils.SolverRuntime import accepts_monotone_energy
from src.utils.StepRetry import (
    StepRetryPolicy,
    is_recoverable_nonlinear_failure,
    nonlinear_failure_kind,
)

from .SoftAffineIPCOperator import (
    SoftAffineIPCOperator,
    _validate_soft_affine_lagged_friction_configuration,
    write_soft_affine_surface_vtu,
)


class SoftAffineIPCEngine(object):
    def __init__(self):
        self.operator = None
        self.soft_material = None
        self.last_newton_iterations = 0
        self.last_friction_iterations = 0
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        self.last_friction_terminated_by_cap = False
        self.last_inner_converged = False
        self.last_inner_failure_reason = ""
        self.last_linear_backend = "HashTriplet"
        self.last_candidate_pairs = 0
        self.step_retry = StepRetryPolicy()
        self.last_failure = None
        self.history = []
        self.advance = None
        self.requested_friction_iterations = 1
        self.last_projection_step = 1.0
        self.pending_elastic_adjoint_seed = None
        self.pending_adjoint_mode = None
        self.pending_plastic_state_vjp = None
        self.last_elastic_differentiation = None

    @staticmethod
    def _fully_implicit_force_target(sims, initial_residual, dt):
        """Keep nonconservative force convergence separate from PN velocity."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise RuntimeError("SoftAffineIPC fully implicit convergence requires a finite positive time step")
        absolute = float(sims.affine_fully_implicit_force_atol)
        relative = float(sims.affine_fully_implicit_force_rtol)
        if not math.isfinite(absolute) or absolute < 0.0 or not math.isfinite(relative) or relative < 0.0:
            raise RuntimeError("SoftAffineIPC fully implicit force tolerances must be " "finite and non-negative")
        alpha, beta, _ = getattr(sims, "affine_newmark", (1.0, 0.5, 1.0))
        return 2.0 * alpha * beta * float(dt) ** 2 * absolute + relative * float(initial_residual)

    def choose_engine(self, sims, scene):
        return

    def set_servo_mechanism(self, sims, callback=None):
        return

    def initialize(self, sims, scene):
        mode, requested_outer = _validate_soft_affine_lagged_friction_configuration(sims)
        if int(scene.softNum[0]) <= 0:
            raise RuntimeError("SoftAffineIPC requires at least one LSMPM soft body.")
        if len(scene.affine_bodies) == 0:
            raise RuntimeError("SoftAffineIPC requires at least one affine rigid body.")
        if _normalize_affine_assemble_type(sims.affine_assemble_type) != "HashTriplet":
            sims.set_affine_body_parameters(assemble_type="HashTriplet")
        if self.soft_material is None:
            self.soft_material = SoftParticleMaterialManager()
            self.soft_material.setup(scene, sims)
        if self.operator is None:
            self.operator = SoftAffineIPCOperator(scene, sims, self.soft_material)
        elif self.operator.friction_mode != mode:
            raise RuntimeError(
                "SoftAffineIPC friction_mode cannot be changed after engine "
                "initialization; configure it before add_essentials()"
            )
        self.requested_friction_iterations = requested_outer
        self.advance = self._step_fully_implicit_device if mode == "fully_implicit" else self._step_lagged_device
        sims.freeze_affine_body_parameters()
        sims.freeze_lsmpm_soft_rigid_contact()
        self.step_retry = StepRetryPolicy(
            enabled=bool(sims.enable_step_retry),
            maximum_retries=sims.step_retry_max_retries,
            reduction=sims.step_retry_reduction,
            minimum_timestep=sims.step_retry_minimum_timestep,
        )

    def step(self, sims, scene):
        original_timestep = float(sims.dt[None])
        attempt_timestep = original_timestep
        attempts = []
        for attempt in range(self.step_retry.maximum_retries + 1):
            if self.step_retry.enabled:
                sims.set_timestep(attempt_timestep)
            getattr(self.operator, "set_timestep", lambda _dt: None)(attempt_timestep)
            try:
                result = self.advance(sims, self.requested_friction_iterations)
            except RuntimeError as exception:
                if not is_recoverable_nonlinear_failure(exception):
                    raise
                self.operator.rollback_step_device()
                failure = {
                    "kind": nonlinear_failure_kind(exception),
                    "exception": type(exception).__name__,
                    "message": str(exception),
                    "attempt": int(attempt),
                    "timestep": float(attempt_timestep),
                    "time": float(sims.current_time),
                    "step": int(sims.current_step),
                }
                attempts.append(failure)
                next_timestep = self.step_retry.next_timestep(attempt_timestep, attempt)
                if next_timestep is None:
                    self.last_failure = {
                        **failure,
                        "original_timestep": original_timestep,
                        "attempts": attempts,
                    }
                    if self.step_retry.enabled:
                        sims.set_timestep(original_timestep)
                        getattr(self.operator, "set_timestep", lambda _dt: None)(original_timestep)
                    raise
                attempt_timestep = next_timestep
                continue

            retry_record = {
                "enabled": bool(self.step_retry.enabled),
                "original_timestep": original_timestep,
                "accepted_timestep": float(attempt_timestep),
                "retry_count": int(attempt),
                "attempts": attempts,
            }
            self.history.append(
                {
                    "step": int(sims.current_step + 1),
                    "time": float(sims.current_time + sims.delta),
                    "step_retry": retry_record,
                    "newton_iterations": int(self.last_newton_iterations),
                    "friction_residual": float(self.last_friction_residual),
                }
            )
            self.last_failure = None
            return result

        raise AssertionError("unreachable Soft-Affine IPC retry state")

    def _differentiate_step(self, sims, scene, loss_gradient, mode, plastic_state_vjp=None):
        if self.operator is None:
            raise RuntimeError("SoftAffineIPCEngine must be initialized before differentiation")
        if self.pending_elastic_adjoint_seed is not None:
            raise RuntimeError("A Soft-Affine differentiation step is already active")
        self.pending_elastic_adjoint_seed = loss_gradient
        self.pending_adjoint_mode = mode
        self.pending_plastic_state_vjp = plastic_state_vjp
        self.last_elastic_differentiation = None
        try:
            self.step(sims, scene)
            if self.last_elastic_differentiation is None:
                raise RuntimeError("Soft-Affine step completed without its pre-commit adjoint")
            return self.last_elastic_differentiation
        finally:
            self.pending_elastic_adjoint_seed = None
            self.pending_adjoint_mode = None
            self.pending_plastic_state_vjp = None

    def differentiate_elastic_step(self, sims, scene, loss_gradient):
        """Advance once and differentiate at the converged pre-commit state."""
        return self._differentiate_step(sims, scene, loss_gradient, "elastic")

    def differentiate_plastic_equilibrium_step(self, sims, scene, loss_gradient):
        raise ValueError(
            "Soft-particle/LSMPM materials are hyperelastic-only; "
            "plastic differentiation is available only on ordinary Direct MPM."
        )

    def differentiate_plastic_step(self, sims, scene, loss_gradient, state_vjp):
        raise ValueError(
            "Soft-particle/LSMPM materials are hyperelastic-only; "
            "plastic differentiation is available only on ordinary Direct MPM."
        )

    def diagnostics_snapshot(self):
        return {
            "schema_version": 1,
            "subsystem": "mpdem_soft_affine_ipc",
            "contact": {
                "model": (
                    getattr(self.operator.affine, "contact_model", "BarrierIPC")
                    if self.operator is not None
                    else "BarrierIPC"
                ),
                "candidate_pairs": int(self.last_candidate_pairs),
                "constraint_violation": (0.0),
                "friction_iterations": int(self.last_friction_iterations),
                "friction_residual": float(self.last_friction_residual),
                "friction_converged": bool(self.last_friction_converged),
            },
            "linear_solver": {"backend": self.last_linear_backend},
            "last_failure": self.last_failure,
            "last_step": self.history[-1] if self.history else None,
        }

    def _reset_nonlinear_diagnostics(self):
        self.last_newton_iterations = 0
        self.last_friction_iterations = 0
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        self.last_friction_terminated_by_cap = False
        self.last_inner_converged = False
        self.last_inner_failure_reason = ""

    def _step_lagged_device(self, sims, requested_outer):
        """Official lagged fixed point with device-resident Newton vectors."""
        self.operator._assert_cuda_device_residency()
        self.last_linear_backend = "TaichiHashTripletPCG"
        max_outer = sims.affine_friction_max_iterations if requested_outer < 0 else requested_outer
        if max_outer <= 0:
            raise RuntimeError("SoftAffineIPC friction_max_iterations must be a positive integer")
        tolerance = float(sims.affine_friction_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise RuntimeError("SoftAffineIPC friction_tolerance must be finite and non-negative")
        self._reset_nonlinear_diagnostics()
        self.operator.begin_step_device()
        self.operator.refresh_lagged_friction_device()
        energy = self.operator.assemble_device(need_matrix=True)
        for outer_iteration in range(max_outer):
            self.operator.backup_lagged_friction_for_adjoint_device()
            energy, inner_iterations = self._solve_lagged_inner_device(sims, energy)
            self.last_newton_iterations += inner_iterations
            self.last_friction_iterations = outer_iteration + 1

            # Match lagged IPC: refresh after the complete frozen solve and
            # measure the updated-system correction without applying it.
            self.operator.refresh_lagged_friction_device()
            energy = self.operator.assemble_device(need_matrix=True)
            probe = self.operator.solve_direction_device(sims, clamp_direction=False)
            self.last_friction_residual = float(probe["unclamped_correction_velocity"])
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
        self._accept_lagged_solution_device()

    def _accept_lagged_solution_device(self):
        try:
            self._differentiate_before_commit()
            self.operator.accept_step_device()
        except BaseException:
            self.operator.rollback_step_device()
            raise

    def _differentiate_before_commit(self):
        if self.pending_elastic_adjoint_seed is not None:
            if self.pending_adjoint_mode == "trajectory":
                self.operator.pullback_coupled_step_device()
                self.last_elastic_differentiation = True
            elif self.pending_adjoint_mode == "plastic_state":
                self.last_elastic_differentiation = self.operator.differentiate_plastic_step_parameters(
                    self.pending_elastic_adjoint_seed,
                    self.pending_plastic_state_vjp,
                )
            elif self.pending_adjoint_mode == "plastic":
                self.last_elastic_differentiation = self.operator.differentiate_plastic_equilibrium_parameters(
                    self.pending_elastic_adjoint_seed
                )
            else:
                self.last_elastic_differentiation = self.operator.differentiate_elastic_parameters(
                    self.pending_elastic_adjoint_seed
                )

    def _step_fully_implicit_device(self, sims, _requested_outer=None):
        """Exact nonsymmetric fully implicit Newton solve on Taichi CUDA."""
        self.operator._assert_cuda_device_residency()
        self.last_linear_backend = "TaichiHashTripletBiCGSTAB"
        self._reset_nonlinear_diagnostics()
        self.operator.begin_step_device()
        try:
            self.operator.initialize_fully_implicit_velocity_predictor_device(sims)
            initial_energy = self.operator.assemble_device(need_matrix=True)
            if not math.isfinite(initial_energy):
                self.last_inner_failure_reason = "non_finite_energy"
                raise RuntimeError("SoftAffineIPC fully implicit initial state has " "non-finite IPC energy")
            iterations = self._solve_fully_implicit_newton_device(sims)
            self.last_newton_iterations = iterations
            self.last_friction_iterations = 1
            self.last_friction_residual = self.operator.residual_inf_norm_device()
            self.last_friction_converged = self.last_inner_converged
            self._differentiate_before_commit()
            self.operator.accept_step_device()
        except Exception:
            self.operator.rollback_step_device()
            raise

    def _solve_fully_implicit_newton_device(self, sims):
        iterations = 0
        self.last_inner_converged = False
        self.last_inner_failure_reason = "maximum_newton_iterations"
        residual_norm = self.operator.residual_inf_norm_device()
        tolerance = self._fully_implicit_force_target(sims, residual_norm, self.operator.dt)
        for iteration in range(int(sims.affine_max_newton_iteration)):
            if residual_norm <= tolerance:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
            result = self.operator.solve_direction_device(sims)
            direction_norm = float(result["solution_inf_norm"])
            if not math.isfinite(direction_norm):
                self.last_inner_failure_reason = "non_finite_direction"
                raise RuntimeError("SoftAffineIPC fully implicit Newton correction is non-finite")
            if direction_norm == 0.0:
                self.last_inner_failure_reason = "newton_stagnation"
                raise RuntimeError(
                    "SoftAffineIPC fully implicit Newton solve stagnated "
                    f"with residual={residual_norm:.6e}, "
                    f"tolerance={tolerance:.6e}"
                )
            merit_slope = self.operator.residual_merit_directional_derivative_device()
            if not math.isfinite(merit_slope) or merit_slope >= 0.0:
                self.last_inner_failure_reason = "non_descent_merit_direction"
                raise RuntimeError(
                    "SoftAffineIPC fully implicit Newton direction is not " "a descent direction for 0.5 * ||R||^2"
                )
            self._fully_implicit_line_search_device(sims, merit_slope)
            iterations = iteration + 1
            residual_norm = self.operator.residual_inf_norm_device()
            if residual_norm <= tolerance:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
        if not self.last_inner_converged:
            raise RuntimeError(
                "SoftAffineIPC fully implicit Newton solve did not converge "
                f"(reason={self.last_inner_failure_reason}, "
                f"iterations={iterations}, residual={residual_norm:.6e}, "
                f"tolerance={tolerance:.6e})"
            )
        return iterations

    def _fully_implicit_line_search_device(self, sims, merit_slope):
        alpha = (
            self.operator.init_step_size_device(
                ccd_type=sims.affine_ccd_type,
                eta=sims.affine_ccd_eta,
                accd_tolerance=sims.affine_accd_tolerance,
                max_iteration=sims.affine_ccd_max_iteration,
            )
            if sims.affine_ccd
            else 1.0
        )
        if not math.isfinite(alpha) or alpha <= 0.0:
            self.last_inner_failure_reason = "ccd_step_size"
            raise RuntimeError("SoftAffineIPC fully implicit CCD produced no feasible step")
        armijo = float(sims.affine_fully_implicit_armijo)
        contraction = float(sims.affine_fully_implicit_line_search_contraction)
        if not 0.0 < armijo < 1.0 or not 0.0 < contraction < 1.0:
            raise ValueError("SoftAffineIPC fully implicit Armijo and contraction must lie in (0, 1)")
        base_merit = self.operator.residual_merit_device()
        if not math.isfinite(base_merit):
            self.last_inner_failure_reason = "non_finite_residual_merit"
            raise RuntimeError("SoftAffineIPC fully implicit base residual merit is " "non-finite")
        if not math.isfinite(merit_slope) or merit_slope >= 0.0:
            self.last_inner_failure_reason = "non_descent_merit_direction"
            raise RuntimeError(
                "SoftAffineIPC fully implicit Armijo requires a negative " "R^T (J p) directional derivative"
            )
        self.operator.store_coupled_base_device()
        for _ in range(int(sims.affine_line_search_max_iteration)):
            self.operator.set_coupled_trial_device(alpha)
            trial_energy = self.operator.assemble_device(need_matrix=False)
            trial_merit = self.operator.residual_merit_device()
            if (
                math.isfinite(trial_energy)
                and math.isfinite(trial_merit)
                and trial_merit <= base_merit + armijo * alpha * merit_slope
            ):
                accepted_energy = self.operator.assemble_device(need_matrix=True)
                if not math.isfinite(accepted_energy):
                    self.last_inner_failure_reason = "non_finite_energy"
                    raise RuntimeError("SoftAffineIPC fully implicit accepted state has " "non-finite IPC energy")
                return float(alpha)
            alpha *= contraction
        self.operator.restore_coupled_base_device()
        self.operator.assemble_device(need_matrix=True)
        self.last_inner_failure_reason = "residual_armijo_line_search"
        raise RuntimeError("SoftAffineIPC fully implicit residual Armijo line search failed")

    def _solve_lagged_inner_device(self, sims, energy):
        iterations = 0
        self.last_inner_converged = False
        self.last_inner_failure_reason = "maximum_newton_iterations"
        tolerance = float(sims.affine_newton_tolerance)
        dt = self._lagged_time_step()
        correction_velocity = math.inf
        self.last_inner_correction_history = []
        for iteration in range(int(sims.affine_max_newton_iteration)):
            if not math.isfinite(energy):
                self.last_inner_failure_reason = "non_finite_energy"
                raise RuntimeError("SoftAffineIPC lagged initial/current state has " "non-finite IPC energy")
            # Test represented motion/strain before max-step truncation.
            result = self.operator.solve_direction_device(sims)
            correction_velocity = float(result["unclamped_correction_velocity"])
            self.last_inner_correction_history.append(correction_velocity)
            # The ``k && gradVanish`` convergence guard evaluates
            # the correction from a completed Newton update.  A nonzero first
            # correction must be applied even when it is below ``tol`` so a
            # mollified sticking state can enter the sliding branch.
            if correction_velocity == 0.0 or (iteration > 0 and correction_velocity < tolerance):
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
            energy = self._line_search_device(sims, energy)
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
        return energy, iterations

    def _line_search_device(self, sims, energy):
        slope = self.operator.energy_directional_derivative_device()
        if not math.isfinite(slope):
            self.last_inner_failure_reason = "non_descent_direction"
            raise RuntimeError("SoftAffineIPC lagged line-search direction has a non-finite " "energy derivative")
        if slope > 0.0:
            self.operator.negate_direction_device()
            slope = -slope
        if slope >= 0.0:
            self.last_inner_failure_reason = "non_descent_direction"
            raise RuntimeError("SoftAffineIPC lagged line search requires a strict descent " "direction")
        alpha = (
            self.operator.init_step_size_device(
                ccd_type=sims.affine_ccd_type,
                eta=sims.affine_ccd_eta,
                accd_tolerance=sims.affine_accd_tolerance,
                max_iteration=sims.affine_ccd_max_iteration,
            )
            if sims.affine_ccd
            else 1.0
        )
        if not math.isfinite(alpha) or alpha <= 0.0:
            self.last_inner_failure_reason = "ccd_step_size"
            raise RuntimeError("SoftAffineIPC lagged CCD produced no feasible step")
        self.operator.store_coupled_base_device()
        for _ in range(int(sims.affine_line_search_max_iteration)):
            self.operator.set_coupled_trial_device(alpha)
            trial_energy = self.operator.assemble_device(need_matrix=False)
            if accepts_monotone_energy(energy, trial_energy, alpha, slope):
                self.operator.assemble_device(need_matrix=True)
                self.last_projection_step = float(alpha)
                return float(trial_energy)
            alpha *= 0.5
        self.operator.restore_coupled_base_device()
        self.operator.assemble_device(need_matrix=True)
        self.last_inner_failure_reason = "monotone_line_search"
        raise RuntimeError(
            "SoftAffineIPC lagged monotone line search failed to find a " "feasible non-increasing-potential step"
        )

    def _lagged_time_step(self):
        """Return the positive step used to convert IPC corrections to m/s."""
        dt = float(self.operator.dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise RuntimeError("SoftAffineIPC lagged stopping criterion requires a finite " "positive time step")
        return dt

    def output_snapshot(self):
        if self.operator is None:
            raise RuntimeError("SoftAffineIPCEngine has not been initialized.")
        return self.operator.output_snapshot()

    def save_affine(self, sims):
        if sims.path is None or self.operator is None:
            return
        snapshot = self.output_snapshot()
        os.makedirs(os.path.join(sims.path, "particles"), exist_ok=True)
        os.makedirs(os.path.join(sims.path, "vtks"), exist_ok=True)
        np.savez(
            os.path.join(sims.path, "particles", f"AffineBody{sims.current_print:06d}.npz"),
            t_current=sims.current_time,
            **snapshot,
        )
        write_soft_affine_surface_vtu(
            os.path.join(sims.path, "vtks", f"GraphicAffineBody{sims.current_print:06d}"),
            snapshot,
        )


__all__ = ["SoftAffineIPCEngine"]
