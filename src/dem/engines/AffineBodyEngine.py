"""Nonlinear affine-body solver orchestration."""

import os

import numpy as np
import taichi as ti
from taichi.lang.impl import current_cfg

from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.physics_model.contact_model.ipc.LevelSetAffine import (
    NonpenetrationResult,
)
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverConsole import print_save_file_info, print_simulation_start
from src.utils.StepRetry import (
    StepRetryPolicy,
    is_recoverable_nonlinear_failure,
    nonlinear_failure_kind,
)
from third_party.pyevtk.hl import unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle

from .AffineBodyOperator import (
    MATRIX_CONTACT_DAMPING,
    MATRIX_COO,
    MATRIX_HASH_TRIPLET,
    TaichiAffineBodyOperator,
    _normalize_affine_assemble_type,
    _normalize_affine_friction_mode,
)
from .AffineBodyState import AffineBodyState
from .AffineDiffIPC import (
    TaichiAffineDiffIPCProjector,
    TaichiAffineMeshDiffIPCProjector,
)


class AffineBodyEngine(object):
    def __init__(self, scene=None, contactor=None):
        self.scene = scene
        self.contactor = contactor
        self.sims = None
        self.state = None
        self.operator = None
        self.last_newton_iterations = 0
        self.last_friction_iterations = 0
        self.last_friction_outer_iteration = 0
        self.last_friction_residual = np.inf
        self.last_friction_converged = False
        self.last_friction_terminated_by_cap = False
        self.last_inner_converged = False
        self.last_inner_failure_reason = ""
        self.last_inner_residual = np.inf
        self.last_inner_control_residual = np.inf
        self.last_inner_correction_history = []
        self.last_fully_implicit_merit_slope = np.nan
        self.last_linear_backend = None
        self.last_linear_converged = False
        self.last_linear_initial_residual = np.inf
        self.last_linear_residual = np.inf
        self.last_linear_original_residual = np.inf
        self.last_linear_iterations = 0
        self.last_linear_failure_reason = ""
        self.last_linear_method = None
        self.last_linear_min_eigenvalue = np.nan
        self.last_linear_relative_negative_curvature = np.nan
        self.last_linear_projection_correction = 0.0
        self.last_linear_spectral_resolution = np.nan
        self.last_linear_unresolved_modes = 0
        self.last_line_search_initial_alpha = np.nan
        self.last_line_search_slope = np.nan
        self.last_line_search_base_energy = np.nan
        self.last_line_search_trials = []
        self.last_line_search_accepted = False
        self.last_line_search_alpha = np.nan
        self.last_line_search_backtracks = 0
        self.step_line_search_calls = 0
        self.step_line_search_min_alpha = np.nan
        self.step_line_search_max_backtracks = 0
        self.step_line_search_converged = True
        self.last_search_mode = None
        self.last_ccd_type = None
        self.last_ccd_step = 1.0
        self.last_candidate_pairs = 0
        self.last_device_nonlinear_path = False
        self.coo_matrix = None
        self.hash_triplet = None
        self.adjoint_triplet = None
        self.last_adjoint_result = None
        self.nonpenetration_solver = None
        self.last_nonpenetration_result = None
        self.levelset_adjoint_seed = None
        self.levelset_direct_adjoint = None
        self.step_retry = StepRetryPolicy()
        self.last_failure = None
        self.history = []
        self.device_nonlinear_path = False
        self.enforce_free_translation = False
        self.evaluate_surface_direction_norm = lambda direction: float(np.linalg.norm(direction, ord=np.inf))
        self.evaluate_device_surface_direction_norm = self._reference_device_surface_direction_norm

    @staticmethod
    def _fully_implicit_force_target(sims, initial_residual):
        """Return a force-unit absolute/relative Newton tolerance."""
        absolute = float(sims.affine_fully_implicit_force_atol)
        relative = float(sims.affine_fully_implicit_force_rtol)
        if not np.isfinite(absolute) or absolute < 0.0 or not np.isfinite(relative) or relative < 0.0:
            raise RuntimeError("Affine fully implicit force tolerances must be finite and " "non-negative")
        return absolute + relative * float(initial_residual)

    def choose_engine(self, sims, scene):
        return

    def initialize(self, sims, scene):
        self.sims = sims
        if len(scene.affine_bodies) == 0:
            raise RuntimeError("AffineBody simulation has no affine bodies.")
        if self.state is None:
            self.state = AffineBodyState.from_scene(scene, sims)
            self.operator = TaichiAffineBodyOperator(self.state, sims, scene)
            self.device_nonlinear_path = self._select_device_nonlinear_path(sims)
            self.evaluate_surface_direction_norm = self.operator.surface_direction_inf_norm
            self.evaluate_device_surface_direction_norm = self._operator_device_surface_direction_norm
            self.enforce_free_translation = self.operator.wall_num == 0 and all(
                float(body.get("force_damping", 0.0)) == 0.0 for body in self.state.bodies
            )
            if self.enforce_free_translation:
                self._accept_device_state = self._accept_device_state_free_translation
            if self.operator.levelset_contact and sims.affine_levelset_auto_initialize:
                self.resolve_levelset_initial_overlaps(sims)
        sims.freeze_affine_body_parameters()
        self.last_search_mode = sims.search
        self.step_retry = StepRetryPolicy(
            enabled=bool(sims.enable_step_retry),
            maximum_retries=sims.step_retry_max_retries,
            reduction=sims.step_retry_reduction,
            minimum_timestep=sims.step_retry_minimum_timestep,
        )
        if getattr(self.operator, "is_semi", False) and self.step_retry.enabled:
            raise ValueError(
                "AffineBody step retry does not support SemiIPC because its "
                "multipliers advance inside Newton iterations"
            )

    def resolve_levelset_initial_overlaps(self, sims=None, **overrides):
        """Push overlapping level-set affine bodies to a strict IPC state."""
        if sims is None:
            raise ValueError("sims is required to resolve level-set initial overlaps")
        if self.state is None or self.operator is None or not self.operator.levelset_contact:
            raise RuntimeError("Initial level-set nonpenetration requires an initialized LevelSet AffineBody engine")
        parameters = {
            "dhat": float(self.operator.max_dhat),
            "kappa": float(np.max(self.operator.pp_kappa_np)),
            "anchor_stiffness": float(
                getattr(
                    sims,
                    "affine_levelset_initial_anchor_stiffness",
                    1.0,
                )
            ),
            "continuation_ratio": float(
                getattr(
                    sims,
                    "affine_levelset_initial_continuation_ratio",
                    0.1,
                )
            ),
            "gap_tolerance": float(
                getattr(
                    sims,
                    "affine_levelset_initial_gap_tolerance",
                    1.0e-8,
                )
            ),
            "maximum_stages": int(
                getattr(
                    sims,
                    "affine_levelset_initial_maximum_stages",
                    8,
                )
            ),
            "stiffness_growth": float(
                getattr(
                    sims,
                    "affine_levelset_initial_stiffness_growth",
                    10.0,
                )
            ),
        }
        if "feasibility_tolerance" in overrides:
            overrides["gap_tolerance"] = overrides.pop("feasibility_tolerance")
        bounds = overrides.pop("bounds", None)
        parameters.update(overrides)
        parameters["maximum_iterations"] = int(
            parameters.get(
                "maximum_iterations",
                getattr(
                    sims,
                    "affine_levelset_initial_maximum_iterations",
                    300,
                ),
            )
        )
        self.nonpenetration_solver = TaichiAffineDiffIPCProjector(
            self.operator,
            self.state.y,
            bounds,
        )
        device_result = self.nonpenetration_solver.solve(**parameters)
        controls = self.operator.y.to_numpy()[: self.operator.control_num].reshape(self.state.y.shape)
        translations = self.nonpenetration_solver.translation.to_numpy()[: self.operator.body_num]
        result = NonpenetrationResult(
            controls=controls,
            translations=translations,
            initial_minimum_gap=float(device_result["initial_minimum_gap"]),
            minimum_gap=float(device_result["minimum_gap"]),
            iterations=int(device_result["iterations"]),
            continuation_stages=int(device_result["continuation_stages"]),
            success=bool(device_result["success"]),
            message=(
                "strictly feasible affine IPC state constructed"
                if device_result["success"]
                else "continuation stages exhausted before the requested positive-gap tolerance"
            ),
            objective=float(device_result["energy"]),
        )
        self.last_nonpenetration_result = result
        if not result.success:
            raise RuntimeError(
                "LevelSet AffineBody initial nonpenetration failed: "
                f"{result.message}; minimum_gap={result.minimum_gap:.6e}"
            )
        self.state.y = result.controls.copy()
        self.state.y_n1 = result.controls.copy()
        self.state.hat_y = result.controls.copy()
        self.state.tilde_y = result.controls.copy()
        return result

    def differentiate_levelset_initialization(self, loss_translation_gradient, direct_anchor_gradient=None):
        """Apply the stored adjoint of the initial nonpenetration solve."""
        if self.nonpenetration_solver is None:
            raise RuntimeError("No level-set initial nonpenetration solve is available")
        if isinstance(loss_translation_gradient, ti.MatrixField):
            return self.nonpenetration_solver.solve_adjoint(
                loss_translation_gradient,
                direct_anchor_gradient=direct_anchor_gradient,
            )
        shape = (self.operator.body_num, 3)
        loss = np.asarray(loss_translation_gradient, dtype=np.float64).reshape(shape)
        direct = np.zeros(shape, dtype=np.float64)
        if direct_anchor_gradient is not None:
            direct = np.asarray(direct_anchor_gradient, dtype=np.float64).reshape(shape)
        if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(direct)):
            raise ValueError("level-set initialization adjoint seeds must be finite")
        if self.levelset_adjoint_seed is None:
            self.levelset_adjoint_seed = ti.Vector.field(3, float, shape=self.operator.body_num)
            self.levelset_direct_adjoint = ti.Vector.field(3, float, shape=self.operator.body_num)
        self.levelset_adjoint_seed.from_numpy(np.ascontiguousarray(loss))
        self.levelset_direct_adjoint.from_numpy(np.ascontiguousarray(direct))
        result = self.nonpenetration_solver.solve_adjoint(
            self.levelset_adjoint_seed,
            direct_anchor_gradient=self.levelset_direct_adjoint,
        )
        return result.to_numpy().reshape(-1)

    def _validate_differentiable_configuration(self):
        if self.state is None or self.operator is None:
            raise RuntimeError("AffineBodyEngine must complete a forward step before solve_adjoint")
        if self.operator.is_semi:
            raise ValueError("AffineBody adjoint currently requires BarrierIPC")
        if self.operator.levelset_contact:
            raise ValueError(
                "AffineBody adjoint currently supports triangle-mesh contact; level-set contact is not differentiable yet"
            )
        if self.operator.fully_implicit:
            raise ValueError(
                "AffineBody differentiable simulation supports lagged friction; "
                "fully_implicit friction is not differentiable yet"
            )
        if self.operator.contact_damping_stiffness > 0.0:
            raise ValueError("AffineBody adjoint currently excludes contact damping history")

    def _solve_adjoint_device(self):
        """Solve with ``linear_rhs`` already resident; leave y pre-commit."""
        self.operator.device_restore_lagged_friction_for_adjoint()
        self.operator.device_enter_equilibrium_adjoint()
        if self.adjoint_triplet is None:
            # Reuse the forward HashTriplet when it already exists.  Taichi
            # scatters capture the field object when the kernel specializes;
            # swapping in a second triplet after a HashTriplet forward solve
            # leaves that compiled kernel writing the old matrix.
            self._ensure_linear_solver(self.sims)
            self.adjoint_triplet = self.hash_triplet
            if self.adjoint_triplet is None:
                self.adjoint_triplet = BuildTriplet(
                    dim=3,
                    max_pairs_num=max(self.operator.max_hash_triplets, 1),
                    max_nonzeros=max(
                        1,
                        min(
                            self.operator.max_hash_triplets,
                            self.operator.control_num * max(self.operator.control_num - 1, 0) // 2,
                        ),
                    ),
                    max_active_nodes=max(self.operator.control_num, 1),
                    symmetric=False,
                    solver="PCG",
                    matrix_symmetric=True,
                    device_reduction=True,
                )
        self.operator.bind_hash_triplet(self.adjoint_triplet)
        self.adjoint_triplet.reset_system()
        self.operator.assemble_device(
            need_matrix=True,
            matrix_mode=MATRIX_HASH_TRIPLET,
            # PSD projection and diagonal shifts define only the forward
            # Newton search direction.  The discrete adjoint differentiates
            # the converged physical residual and therefore needs its exact
            # unregularized Jacobian.
            project_spd=False,
            solver_shift=False,
        )
        self.adjoint_triplet.finalize_taichi_assembly()
        forward_residual = float(self.operator.device_gradient_inf_norm())
        forward_solver = self.adjoint_triplet.solver
        self.adjoint_triplet.solver = "PCG"
        try:
            result = self.adjoint_triplet.solve_flat_system(
                self.operator.linear_rhs,
                self.operator.linear_x,
                active_nodes=self.operator.control_num,
                tol=self.sims.affine_linear_tolerance,
                maxiter=self.sims.affine_linear_max_iteration,
                return_solution=False,
                transpose=not self.adjoint_triplet.matrix_symmetric,
                fallback_to_bicgstab=True,
            )
        finally:
            self.adjoint_triplet.solver = forward_solver
        if not result["converged"]:
            raise RuntimeError(
                "AffineBody adjoint device solve did not converge: "
                f"residual={result['residual']:.6e}, iterations={result['iterations']}"
            )
        self.last_adjoint_result = {
            "linear_residual": float(result["residual"]),
            "forward_residual": forward_residual,
            "iterations": int(result["iterations"]),
            "converged": True,
        }

    def solve_adjoint(self, loss_gradient):
        """Solve the exact one-step elastic ABD/BarrierIPC adjoint."""
        self._validate_differentiable_configuration()

        seed = np.asarray(loss_gradient, dtype=np.float64)
        if seed.size != self.operator.dof or not np.all(np.isfinite(seed)):
            raise ValueError(f"loss_gradient must contain {self.operator.dof} finite values")
        self.operator.linear_rhs.from_numpy(np.ascontiguousarray(seed.reshape(-1)))
        try:
            self._solve_adjoint_device()
            adjoint = self.operator.linear_x.to_numpy()[: self.operator.dof].reshape(self.state.y.shape)
            self.last_adjoint_result["adjoint"] = adjoint
            return adjoint
        finally:
            self.operator.device_leave_equilibrium_adjoint()

    def differentiate_step_parameters(self, loss_gradient):
        """Return gravity and motor-target VJPs for the last converged step."""
        self._validate_differentiable_configuration()
        seed = np.asarray(loss_gradient, dtype=np.float64)
        if seed.size != self.operator.dof or not np.all(np.isfinite(seed)):
            raise ValueError(f"loss_gradient must contain {self.operator.dof} finite values")
        self.operator.linear_rhs.from_numpy(np.ascontiguousarray(seed.reshape(-1)))
        try:
            self._solve_adjoint_device()
            gravity, young, target_radians = self.operator.differentiate_step_parameters()
            joint_damping = self.operator.joint_damping_vjp.to_numpy()[: self.operator.joint_num].copy()
            friction_scale = float(self.operator.friction_scale_vjp[None])
            adjoint = self.operator.linear_x.to_numpy()[: self.operator.dof].reshape(self.state.y.shape)
            self.last_adjoint_result["adjoint"] = adjoint
        finally:
            self.operator.device_leave_equilibrium_adjoint()
        return {
            "gravity": gravity,
            "affine_young_modulus": young,
            "joint_target_angle_radians": target_radians,
            "joint_target_angle_degrees": target_radians * (np.pi / 180.0),
            "joint_damping": joint_damping,
            "friction_scale": friction_scale,
            "adjoint": adjoint,
        }

    def pullback_step_device(self):
        """Pull accepted ``state_*_vjp`` fields through the replayed step."""
        self._validate_differentiable_configuration()
        operator = self.operator
        operator.device_prepare_step_state_adjoint(int(self.enforce_free_translation))
        try:
            self._solve_adjoint_device()
            operator._differentiate_gravity_parameter()
            operator._differentiate_young_parameter()
            operator._differentiate_joint_target_parameters()
            operator._differentiate_joint_damping_parameters()
            operator._differentiate_friction_scale_parameter()
            operator.device_add_translation_gravity_vjp()
            operator.device_propagate_step_state_adjoint()
            operator.device_propagate_lagged_friction_state_adjoint()
        finally:
            operator.device_leave_equilibrium_adjoint()

    def run(self, sims, scene):
        self.initialize(sims, scene)
        self._prepare_output(sims)
        self._save(sims, scene)
        last_save_time = float(sims.current_time)
        target_time = float(sims.current_time + sims.time)
        print_simulation_start("DEM")
        while sims.current_time < target_time - 1.0e-14:
            self.step(sims, scene)
            sims.current_time += sims.delta
            sims.current_step += 1
            runtime_checkpoint()
            if sims.current_time - last_save_time >= sims.save_interval - 0.1 * sims.delta:
                self._save(sims, scene)
                last_save_time = float(sims.current_time)
        if abs(sims.current_time - last_save_time) > 0.9 * sims.save_interval:
            self._save(sims, scene)
        print("#", " End Affine Body Simulation ".center(67, "="), "#", "\n")

    def _failure_diagnostics(self, sims, exception, attempt, timestep):
        return {
            "kind": nonlinear_failure_kind(exception),
            "exception": type(exception).__name__,
            "message": str(exception),
            "attempt": int(attempt),
            "timestep": float(timestep),
            "time": float(sims.current_time),
            "step": int(sims.current_step),
            "contact": {
                "model": (
                    getattr(self.operator, "contact_model", "BarrierIPC") if self.operator is not None else "BarrierIPC"
                ),
                "ccd_step": float(self.last_ccd_step),
                "candidate_pairs": int(self.last_candidate_pairs),
                "constraint_violation": (
                    float(self.operator.semi_constraint_violation[None])
                    if self.operator is not None and getattr(self.operator, "is_semi", False)
                    else 0.0
                ),
            },
        }

    def diagnostics_snapshot(self):
        return {
            "schema_version": 1,
            "subsystem": "dem_affine_body",
            "contact": {
                "model": (
                    getattr(self.operator, "contact_model", "BarrierIPC") if self.operator is not None else "BarrierIPC"
                ),
                "ccd_step": float(self.last_ccd_step),
                "candidate_pairs": int(self.last_candidate_pairs),
                "constraint_violation": (
                    float(self.operator.semi_constraint_violation[None])
                    if self.operator is not None and getattr(self.operator, "is_semi", False)
                    else 0.0
                ),
                "friction_iterations": int(self.last_friction_iterations),
                "friction_residual": float(self.last_friction_residual),
                "friction_converged": bool(self.last_friction_converged),
            },
            "linear_solver": {
                "method": self.last_linear_method,
                "iterations": int(self.last_linear_iterations),
                "initial_residual": float(self.last_linear_initial_residual),
                "residual": float(self.last_linear_residual),
                "converged": bool(self.last_linear_converged),
            },
            "line_search": {
                "converged": bool(self.step_line_search_converged),
                "calls": int(self.step_line_search_calls),
                "minimum_alpha": float(self.step_line_search_min_alpha),
                "maximum_backtracks": int(self.step_line_search_max_backtracks),
                "last_alpha": float(self.last_line_search_alpha),
                "last_backtracks": int(self.last_line_search_backtracks),
            },
            "last_failure": self.last_failure,
            "last_step": self.history[-1] if self.history else None,
        }

    def translate_wall(self, wall_id, offset):
        """Translate a standalone AffineBody IPC wall between accepted steps."""
        if self.operator is None:
            raise RuntimeError("AffineBody engine must be initialized before moving a wall")
        wall_id = int(wall_id)
        if not 0 <= wall_id < self.operator.wall_num:
            raise IndexError(f"Affine wall id {wall_id} is outside [0, {self.operator.wall_num})")
        offset = np.asarray(offset, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(offset)):
            raise ValueError("Affine wall translation must be finite")
        self.operator.device_translate_wall(wall_id, *offset.tolist())

    def step(self, sims, scene):
        original_timestep = float(sims.dt[None])
        attempt_timestep = original_timestep
        attempts = []
        for attempt in range(self.step_retry.maximum_retries + 1):
            if self.step_retry.enabled:
                sims.set_timestep(attempt_timestep)
            getattr(self.operator, "set_timestep", lambda _dt: None)(attempt_timestep)
            try:
                result = self._step_once(sims, scene)
            except RuntimeError as exception:
                if not is_recoverable_nonlinear_failure(exception):
                    raise
                failure = self._failure_diagnostics(sims, exception, attempt, attempt_timestep)
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
                    "ccd_step": float(self.last_ccd_step),
                    "line_search_calls": int(self.step_line_search_calls),
                    "line_search_minimum_alpha": float(self.step_line_search_min_alpha),
                    "line_search_maximum_backtracks": int(self.step_line_search_max_backtracks),
                }
            )
            self.last_failure = None
            return result

        raise AssertionError("unreachable AffineBody retry state")

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
        if getattr(self.operator, "is_semi", False):
            self.operator.reset_semi_state()
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

        if self.last_device_nonlinear_path:
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
        tolerance = self._fully_implicit_force_target(sims, initial_residual)
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

    @staticmethod
    def _select_device_nonlinear_path(sims):
        """Validate and select the production device-resident implementation."""
        try:
            current_cfg().arch
        except (AttributeError, RuntimeError):
            raise RuntimeError(
                "AffineBody runtime requires an initialized Taichi " "architecture; the NumPy oracle is not a backend"
            )
        return _normalize_affine_assemble_type(sims.affine_assemble_type) in (
            "MatrixFree",
            "COO",
            "HashTriplet",
        )

    def _assemble_system_device(self, sims, need_matrix=True):
        """Assemble energy/residual/Jacobian without downloading residual."""
        self._ensure_linear_solver(sims)
        backend = _normalize_affine_assemble_type(sims.affine_assemble_type)
        matrix_mode = MATRIX_HASH_TRIPLET if backend == "HashTriplet" else MATRIX_COO
        if need_matrix:
            if matrix_mode == MATRIX_HASH_TRIPLET:
                self.hash_triplet.reset_system()
            else:
                self.coo_matrix.reset()
        energy = self.operator.assemble_device(
            need_matrix=bool(need_matrix),
            matrix_mode=matrix_mode,
        )
        self.last_candidate_pairs = self.operator.last_candidate_pairs
        return energy

    def _record_line_search(self, accepted, alpha=np.nan, backtracks=0):
        self.last_line_search_accepted = bool(accepted)
        self.last_line_search_alpha = float(alpha)
        self.last_line_search_backtracks = int(backtracks)
        if not accepted:
            self.step_line_search_converged = False
            return
        self.step_line_search_calls += 1
        if not np.isfinite(self.step_line_search_min_alpha):
            self.step_line_search_min_alpha = float(alpha)
        else:
            self.step_line_search_min_alpha = min(
                self.step_line_search_min_alpha,
                float(alpha),
            )
        self.step_line_search_max_backtracks = max(
            self.step_line_search_max_backtracks,
            int(backtracks),
        )

    def _solve_hash_direction_device(self, sims):
        """Solve J p = -R with both vectors resident in Taichi fields."""
        self._ensure_linear_solver(sims)
        active_nodes = self.operator.control_num
        backend = _normalize_affine_assemble_type(sims.affine_assemble_type)
        self.last_linear_method = "BiCGSTAB" if self.operator.fully_implicit else "PCG"
        self.last_linear_converged = False
        self.last_linear_initial_residual = np.inf
        self.last_linear_residual = np.inf
        self.last_linear_original_residual = np.inf
        self.last_linear_iterations = 0
        self.last_linear_failure_reason = ""
        self.last_linear_min_eigenvalue = np.nan
        self.last_linear_relative_negative_curvature = np.nan
        self.last_linear_projection_correction = 0.0
        self.last_linear_spectral_resolution = np.nan
        self.last_linear_unresolved_modes = 0
        if backend != "HashTriplet":
            self.last_linear_backend = backend
            self.operator.device_load_negative_gradient_scalar(self.operator.linear_rhs)
            attempts = 1 if self.operator.fully_implicit else 6
            base_shift = max(float(sims.affine_hessian_shift), 1.0e-12)
            for attempt in range(attempts):
                self.operator.linear_x.fill(0.0)
                self.operator._build_coo_preconditioner(base_shift)
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
                if succeeded:
                    self.operator.device_copy_scalar_solution_to_direction(self.operator.linear_x)
                    return {
                        "converged": True,
                        "iterations": int(solver.last_iterations),
                        "residual": float(solver.last_residual),
                    }
                if self.operator.fully_implicit:
                    break
                if attempt + 1 < attempts:
                    self.operator._add_hessian_shift(
                        MATRIX_COO,
                        base_shift * (10.0**attempt),
                    )
                    self.operator.finalize_coo_assembly()
            method = "BiCGSTAB" if self.operator.fully_implicit else "PCG"
            raise RuntimeError(
                f"Affine {backend} device {method} did not converge: "
                f"residual={solver.last_residual:.6e}, "
                f"iterations={solver.last_iterations}"
            )

        self.last_linear_backend = "HashTriplet"
        self.hash_triplet.finalize_taichi_assembly()
        self.operator.device_load_negative_gradient(self.hash_triplet.rhs)

        attempts = 1 if self.operator.fully_implicit else 6
        base_shift = max(float(sims.affine_hessian_shift), 1.0e-12)
        last_result = None
        last_error = None
        for attempt in range(attempts):
            try:
                result = self.hash_triplet.solve(
                    active_nodes=active_nodes,
                    tol=sims.affine_linear_tolerance,
                    maxiter=sims.affine_linear_max_iteration,
                    return_solution=False,
                )
                last_result = result
                self.last_linear_converged = bool(result["converged"])
                self.last_linear_initial_residual = float(result["initial_residual"])
                self.last_linear_residual = float(result["residual"])
                self.last_linear_original_residual = self.last_linear_residual
                self.last_linear_iterations = int(result["iterations"])
                self.last_linear_failure_reason = "" if result["converged"] else "maximum_iterations_or_breakdown"
                if result["converged"]:
                    self.operator.device_copy_hash_solution_to_direction(self.hash_triplet.x)
                    return result
            except AssertionError as exc:
                last_error = exc
            if self.operator.fully_implicit:
                break
            if attempt + 1 < attempts:
                self.operator._add_hessian_shift(
                    MATRIX_HASH_TRIPLET,
                    base_shift * (10.0**attempt),
                )

        method = "BiCGSTAB" if self.operator.fully_implicit else "PCG"
        if last_result is not None:
            message = (
                f"Affine HashTriplet {method} did not converge: "
                f"residual={last_result['residual']:.6e}, "
                f"iterations={last_result['iterations']}"
            )
        else:
            message = f"Affine HashTriplet {method} did not converge"
        raise RuntimeError(message) from last_error

    def _clamp_device_direction(self, max_step):
        norm = float(self.operator.device_direction_inf_norm())
        if max_step > 0.0 and norm > max_step:
            self.operator.device_scale_direction(float(max_step) / norm)
            return float(max_step)
        return norm

    def _surface_direction_inf_norm(self, direction):
        return float(self.evaluate_surface_direction_norm(direction))

    def _device_surface_direction_inf_norm(self, control_norm=None):
        return float(self.evaluate_device_surface_direction_norm(control_norm))

    def _reference_device_surface_direction_norm(self, control_norm=None):
        if control_norm is not None:
            return float(control_norm)
        return float(self.operator.device_direction_inf_norm())

    def _operator_device_surface_direction_norm(self, _control_norm=None):
        return float(self.operator.device_surface_direction_inf_norm())

    def _ccd_step_size_device(self, sims):
        self.last_ccd_type = str(sims.affine_ccd_type)
        if not sims.affine_ccd:
            self.last_ccd_step = 1.0
            return 1.0
        alpha = self.operator.init_step_size_device(
            ccd_type=sims.affine_ccd_type,
            eta=sims.affine_ccd_eta,
            accd_tolerance=sims.affine_accd_tolerance,
            max_iteration=sims.affine_ccd_max_iteration,
        )
        self.last_ccd_step = alpha
        return alpha

    def _accept_device_state(self, dt):
        self.operator.device_accept_step(dt)

    def _accept_device_state_free_translation(self, dt):
        self.operator.device_accept_step(dt)
        self.operator.device_enforce_free_translation_momentum(dt)

    def _step_lagged_cuda(self, sims, dt, requested_outer, max_outer, tolerance):
        """Official lagged IPC fixed-point/Newton flow on Taichi CUDA."""
        self.operator.initialize_contact_damping_device()
        energy = self._assemble_system_device(sims, need_matrix=True)

        for outer_iteration in range(max_outer):
            self.last_friction_outer_iteration = outer_iteration + 1
            self.operator.device_backup_lagged_friction_for_adjoint()
            energy, inner_iterations = self._solve_lagged_inner_cuda(sims, energy)
            self.last_newton_iterations += inner_iterations
            self.last_friction_iterations = outer_iteration + 1

            # Rebuild the frozen contact/friction data only after a complete
            # inner solve, then probe the updated fixed point
            # without applying that correction.
            self.operator.initialize_contact_damping_device()
            energy = self._assemble_system_device(sims, need_matrix=True)
            self._solve_hash_direction_device(sims)
            self.last_friction_residual = float(self._device_surface_direction_inf_norm() / dt)
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
        self._accept_device_state(dt)

    def _solve_lagged_inner_cuda(self, sims, energy):
        """Newton solve with iterate, gradient, and correction on CUDA."""
        iterations = 0
        tolerance = float(sims.affine_newton_tolerance)
        self.last_inner_converged = False
        self.last_inner_failure_reason = "maximum_newton_iterations"
        self.last_inner_residual = np.inf
        self.last_inner_control_residual = np.inf
        self.last_inner_correction_history = []
        semi_progress = 0.0
        maximum_updates = int(sims.affine_max_newton_iteration)
        for iteration in range(maximum_updates + 1):
            if getattr(self.operator, "is_semi", False) and iteration > 1 and semi_progress > 0.999:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
            if not np.isfinite(energy):
                self.last_inner_failure_reason = "non_finite_energy"
                raise RuntimeError(
                    "Affine lagged IPC energy is non-finite; the current " "configuration is not strictly feasible"
                )
            if int(self.operator.device_gradient_has_nonfinite()) != 0:
                self.last_inner_failure_reason = "non_finite_gradient"
                raise RuntimeError("Affine lagged Newton residual is non-finite")
            self._solve_hash_direction_device(sims)
            if int(self.operator.device_direction_has_nonfinite()) != 0:
                self.last_inner_failure_reason = "non_finite_direction"
                raise RuntimeError("Affine lagged Newton correction is non-finite")
            raw_direction_norm = float(self.operator.device_direction_inf_norm())
            surface_direction_norm = self._device_surface_direction_inf_norm(raw_direction_norm)
            self.last_inner_control_residual = raw_direction_norm / self.operator.dt
            self.last_inner_residual = surface_direction_norm / self.operator.dt
            self.last_inner_correction_history.append(self.last_inner_residual)
            # Reference IPC defines ``tol`` in m/s and stops on the full
            # (unclamped, unapplied) Newton correction from the *previous*
            # Newton update.  Its ``k && gradVanish`` guard forces a nonzero
            # first correction to be applied.  That detail is essential near
            # the mollified stick/slip transition: the first correction can
            # be smaller than the nonlinear tolerance while still crossing
            # ``epsv`` and exposing the sliding branch on the next solve.
            # An exactly zero initial correction is already stationary and
            # avoids sending a zero direction into CCD/line search.
            if self.last_inner_residual == 0.0 or (iteration > 0 and self.last_inner_residual < tolerance):
                if getattr(self.operator, "is_semi", False) and not self.operator.semi_contact_converged():
                    self.operator.accept_semi_update_device()
                    energy = self._assemble_system_device(sims, need_matrix=True)
                    continue
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
            if iteration == maximum_updates:
                break
            self._clamp_device_direction(sims.affine_max_step)
            slope = float(self.operator.device_gradient_direction_dot())
            if np.isfinite(slope) and slope > 0.0:
                self.operator.device_scale_direction(-1.0)
            energy = self._line_search_cuda(sims, energy)
            if getattr(self.operator, "is_semi", False):
                alpha = float(self.last_line_search_alpha)
                semi_progress += (1.0 - semi_progress) * alpha
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
        return energy, iterations

    def _line_search_cuda(self, sims, energy):
        slope = float(self.operator.device_gradient_direction_dot())
        if not np.isfinite(slope) or slope >= 0.0:
            self._record_line_search(False)
            self.last_inner_failure_reason = "non_descent_direction"
            raise RuntimeError("Affine IPC line search requires a descent direction")
        alpha = self._ccd_step_size_device(sims)
        if not np.isfinite(alpha) or alpha <= 0.0:
            self._record_line_search(False)
            self.last_inner_failure_reason = "ccd_step_size"
            raise RuntimeError("Affine lagged IPC CCD produced no strictly feasible step")
        self.last_line_search_initial_alpha = float(alpha)
        self.last_line_search_slope = slope
        self.last_line_search_base_energy = float(energy)
        self.last_line_search_trials = []
        self.operator.device_backup_line_search_base()
        for backtracks in range(int(sims.affine_line_search_max_iteration)):
            self.operator.device_set_line_search_trial(alpha)
            trial_energy = self._assemble_system_device(sims, need_matrix=False)
            self.last_line_search_trials.append((float(alpha), float(trial_energy)))
            if (
                np.isfinite(trial_energy)
                and int(self.operator.device_gradient_has_nonfinite()) == 0
                and trial_energy <= energy
            ):
                self._assemble_system_device(sims, need_matrix=True)
                if getattr(self.operator, "is_semi", False):
                    self.operator.accept_semi_update_device()
                    trial_energy = self._assemble_system_device(sims, need_matrix=True)
                self._record_line_search(True, alpha, backtracks)
                return trial_energy
            alpha *= 0.5
        self.operator.device_restore_line_search_base()
        self._assemble_system_device(sims, need_matrix=True)
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

    def _step_fully_implicit_cuda(self, sims, previous_y, dt):
        """Exact nonsymmetric fully implicit friction Newton solve on CUDA."""
        self.operator.device_backup_step_start()
        self.operator.initialize_contact_damping_device()

        # Paper predictor q^(0) = q_n + h v_n, clipped by IPC CCD.
        self.operator.device_set_predictor_direction()
        predictor_norm = float(self.operator.device_direction_inf_norm())
        if predictor_norm > 0.0:
            alpha = self._ccd_step_size_device(sims)
            if not np.isfinite(alpha) or alpha <= 0.0:
                raise RuntimeError("Affine fully implicit predictor CCD returned an " "invalid/non-positive step")
            self.operator.device_backup_line_search_base()
            self.operator.device_set_line_search_trial(alpha)

        initial_energy = self._assemble_system_device(sims, need_matrix=True)
        if not np.isfinite(initial_energy):
            self.last_inner_failure_reason = "non_finite_energy"
            raise RuntimeError("Affine fully implicit initial state has non-finite IPC " "energy")
        iterations = 0
        initial_residual = float(self.operator.device_gradient_inf_norm())
        if not np.isfinite(initial_residual):
            self.last_inner_failure_reason = "non_finite_residual"
            raise RuntimeError("Affine fully implicit initial residual is non-finite")
        tolerance = self._fully_implicit_force_target(sims, initial_residual)
        self.last_inner_failure_reason = "maximum_newton_iterations"
        for iteration in range(int(sims.affine_max_newton_iteration)):
            if int(self.operator.device_gradient_has_nonfinite()) != 0:
                self.last_inner_failure_reason = "non_finite_residual"
                raise RuntimeError("Affine fully implicit residual is non-finite")
            residual_norm = float(self.operator.device_gradient_inf_norm())
            if residual_norm <= tolerance:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break

            self._solve_hash_direction_device(sims)
            direction_norm = self._clamp_device_direction(sims.affine_max_step)
            if int(self.operator.device_direction_has_nonfinite()) != 0:
                self.last_inner_failure_reason = "non_finite_direction"
                raise RuntimeError("Affine fully implicit Newton correction is non-finite")
            if direction_norm <= np.finfo(float).eps:
                self.last_inner_failure_reason = "newton_stagnation"
                raise RuntimeError("Affine fully implicit Newton stagnated before the " "residual tolerance was met")

            self._fully_implicit_line_search_cuda(sims)
            iterations = iteration + 1
            if (
                int(self.operator.device_gradient_has_nonfinite()) == 0
                and float(self.operator.device_gradient_inf_norm()) <= tolerance
            ):
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break

        self.last_newton_iterations = iterations
        self.last_friction_residual = float(self.operator.device_gradient_inf_norm())
        self.last_friction_converged = self.last_inner_converged
        if not self.last_inner_converged:
            raise RuntimeError(
                "Affine fully implicit Newton solve did not converge "
                f"(reason={self.last_inner_failure_reason}, "
                f"iterations={iterations}, "
                f"residual={self.last_friction_residual:.6e}, "
                f"tolerance={tolerance:.6e})"
            )
        self._accept_device_state(dt)

    def _fully_implicit_line_search_cuda(self, sims):
        """Exact residual-merit Armijo search with device-resident vectors."""
        # The solve output may have been max-step clamped.  Put the exact
        # correction used by CCD/Armijo back in the Krylov vector before Jp.
        self.operator.device_copy_direction_to_hash_vector(self.hash_triplet.x)
        nnz = int(self.hash_triplet.non_diag.element_pair_num[0])
        self.hash_triplet.matvec(
            self.operator.control_num,
            nnz,
            self.hash_triplet.x,
            self.hash_triplet.Ax,
        )
        base_merit = 0.5 * float(self.operator.device_gradient_squared_norm())
        merit_slope = float(
            self.operator.device_residual_jacobian_direction_dot(
                self.hash_triplet.Ax,
                float(self.operator._active_jacobian_shift()),
            )
        )
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

        alpha = self._ccd_step_size_device(sims)
        if not np.isfinite(alpha) or alpha <= 0.0:
            self._record_line_search(False)
            self.last_inner_failure_reason = "ccd_step_size"
            raise RuntimeError("Affine fully implicit CCD produced no feasible step")
        armijo = float(sims.affine_fully_implicit_armijo)
        contraction = float(sims.affine_fully_implicit_line_search_contraction)
        if not 0.0 < armijo < 1.0 or not 0.0 < contraction < 1.0:
            raise ValueError("Affine fully implicit Armijo and contraction must lie " "in (0, 1)")

        self.operator.device_backup_line_search_base()
        for backtracks in range(int(sims.affine_line_search_max_iteration)):
            self.operator.device_set_line_search_trial(alpha)
            trial_energy = self._assemble_system_device(sims, need_matrix=False)
            trial_merit = np.inf
            if np.isfinite(trial_energy) and int(self.operator.device_gradient_has_nonfinite()) == 0:
                trial_merit = 0.5 * float(self.operator.device_gradient_squared_norm())
            if np.isfinite(trial_merit) and trial_merit <= base_merit + armijo * alpha * merit_slope:
                accepted_energy = self._assemble_system_device(sims, need_matrix=True)
                if not np.isfinite(accepted_energy):
                    self._record_line_search(False, backtracks=backtracks)
                    self.last_inner_failure_reason = "non_finite_energy"
                    raise RuntimeError("Affine fully implicit accepted state has " "non-finite IPC energy")
                self._record_line_search(True, alpha, backtracks)
                return alpha
            alpha *= contraction

        self.operator.device_restore_line_search_base()
        self._assemble_system_device(sims, need_matrix=True)
        self._record_line_search(False, backtracks=int(sims.affine_line_search_max_iteration))
        self.last_inner_failure_reason = "residual_armijo_line_search"
        raise RuntimeError("Affine fully implicit residual Armijo line search failed")

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
        semi_progress = 0.0
        dt = float(self.operator.dt) if self.operator is not None else 1.0
        maximum_updates = int(sims.affine_max_newton_iteration)
        for iteration in range(maximum_updates + 1):
            if getattr(self.operator, "is_semi", False) and iteration > 1 and semi_progress > 0.999:
                self.last_inner_converged = True
                self.last_inner_failure_reason = ""
                break
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
                if getattr(self.operator, "is_semi", False) and not self.operator.semi_contact_converged():
                    self.operator.accept_semi_update_device()
                    energy, grad = self._assemble_system(sims, y_flat, need_matrix=True)
                    continue
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
            if getattr(self.operator, "is_semi", False):
                semi_progress += (1.0 - semi_progress) * step_size
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

    def _matrix_mode(self, sims):
        backend = _normalize_affine_assemble_type(sims.affine_assemble_type)
        if backend == "HashTriplet":
            return MATRIX_HASH_TRIPLET
        return MATRIX_COO

    def _ensure_linear_solver(self, sims):
        backend = _normalize_affine_assemble_type(sims.affine_assemble_type)
        if backend == "HashTriplet":
            if self.hash_triplet is None:
                self.hash_triplet = BuildTriplet(
                    dim=3,
                    max_pairs_num=max(self.operator.max_hash_triplets, 1),
                    max_nonzeros=max(
                        1,
                        min(
                            self.operator.max_hash_triplets,
                            self.operator.control_num
                            * max(self.operator.control_num - 1, 0)
                            // (1 if self.operator.fully_implicit else 2),
                        ),
                    ),
                    max_active_nodes=max(self.operator.control_num, 1),
                    symmetric=False,
                    # Official lagged IPC projects each complete local
                    # non-inertial Hessian before block scatter, so inertia
                    # is SPD and uses PCG.  FI keeps the exact nonsymmetric
                    # residual Jacobian and therefore uses BiCGSTAB.
                    solver=("BiCGSTAB" if self.operator.fully_implicit else "PCG"),
                    matrix_symmetric=not self.operator.fully_implicit,
                    device_reduction=True,
                )
            self.operator.bind_hash_triplet(self.hash_triplet)
        else:
            if self.coo_matrix is None:
                self.coo_matrix = CoordinateSparseMatrix(
                    max(self.operator.max_coo_entries, 1),
                    max(self.operator.dof, 1),
                    is_sparse=False,
                    preconditioned=True,
                    symmetry=not self.operator.fully_implicit,
                    linear_solver=True,
                )
            self.operator.bind_coo_matrix(self.coo_matrix)

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
        mass_matrix = np.asarray(
            self.operator.control_mass_matrix_np,
            dtype=np.float64,
        )
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
        self.hash_triplet.finalize_taichi_assembly()
        result = self.hash_triplet.solve(
            rhs=rhs.reshape((active_nodes, 3)),
            active_nodes=active_nodes,
            tol=sims.affine_linear_tolerance,
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
            if np.isfinite(trial_energy) and trial_energy <= energy:
                self._assemble_system(sims, trial, need_matrix=True)
                if getattr(self.operator, "is_semi", False):
                    self.operator.accept_semi_update_device()
                    trial_energy, trial_grad = self._assemble_system(sims, trial, need_matrix=True)
                self._record_line_search(True, alpha, backtracks)
                return alpha, trial_energy, trial_grad
            alpha *= 0.5
        # IPC relies on the line search to preserve strict feasibility and
        # monotone potential decrease. Never accept alpha=0 or an
        # energy-increasing fallback.
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

    def _prepare_output(self, sims):
        if sims.path is None:
            return
        os.makedirs(os.path.join(sims.path, "particles"), exist_ok=True)
        os.makedirs(os.path.join(sims.path, "vtks"), exist_ok=True)

    def _save(self, sims, scene):
        if sims.path is None:
            return
        if self.last_device_nonlinear_path:
            self.operator.sync_output_state()
        vertices, faces, body_ids, group_ids = self.state.surface_mesh()
        face_body_ids, face_group_ids, face_material_ids = self._surface_cell_data()
        path = os.path.join(sims.path, "particles", f"AffineBody{sims.current_print:06d}.npz")
        np.savez(
            path,
            t_current=sims.current_time,
            vertices=vertices,
            faces=faces,
            bodyID=body_ids,
            groupID=group_ids,
            faceBodyID=face_body_ids,
            faceGroupID=face_group_ids,
            faceMaterialID=face_material_ids,
            y=self.state.y,
            v_y=self.state.v_y,
            materialID=np.asarray([body["materialID"] for body in self.state.bodies], dtype=np.int32),
            forceLocalDamping=np.asarray(
                [body.get("force_damping", 0.0) for body in self.state.bodies], dtype=np.float64
            ),
            torqueLocalDamping=np.asarray(
                [body.get("torque_damping", 0.0) for body in self.state.bodies], dtype=np.float64
            ),
        )
        self._save_vtk(sims, vertices, faces, body_ids, group_ids, face_body_ids, face_group_ids, face_material_ids)
        print_save_file_info(
            "DEM",
            sims.current_step,
            sims.current_print,
            sims.current_time,
            sims.path,
        )
        sims.current_print += 1

    def _surface_cell_data(self):
        face_body_ids = []
        face_group_ids = []
        face_material_ids = []
        for body_id, body in enumerate(self.state.bodies):
            face_num = int(np.asarray(body["faces"]).shape[0])
            face_body_ids.extend([body_id] * face_num)
            face_group_ids.extend([body["groupID"]] * face_num)
            face_material_ids.extend([body["materialID"]] * face_num)
        return (
            np.asarray(face_body_ids, dtype=np.int32),
            np.asarray(face_group_ids, dtype=np.int32),
            np.asarray(face_material_ids, dtype=np.int32),
        )

    def output_snapshot(self):
        if self.state is None:
            raise RuntimeError("AffineBodyEngine has not been initialized.")
        if self.last_device_nonlinear_path:
            # The accepted affine state stays device-resident throughout the
            # nonlinear loop.  Synchronize it once at the recorder/user
            # observation boundary; per-step host downloads would put output
            # work back into the physical hot path.
            self.operator.sync_output_state()
        vertices, faces, body_ids, group_ids = self.state.surface_mesh()
        face_body_ids, face_group_ids, face_material_ids = self._surface_cell_data()
        body_vertices = self.state.world_vertices()
        body_num = self.state.body_num
        centers = np.zeros((body_num, 3), dtype=np.float64)
        bbox_min = np.zeros((body_num, 3), dtype=np.float64)
        bbox_max = np.zeros((body_num, 3), dtype=np.float64)
        radius = np.zeros(body_num, dtype=np.float64)
        for body_id, bverts in enumerate(body_vertices):
            centers[body_id] = np.mean(bverts, axis=0)
            bbox_min[body_id] = np.min(bverts, axis=0)
            bbox_max[body_id] = np.max(bverts, axis=0)
            radius[body_id] = np.max(np.linalg.norm(bverts - centers[body_id], axis=1))
        body_group_ids = np.asarray([body["groupID"] for body in self.state.bodies], dtype=np.int32)
        body_material_ids = np.asarray([body["materialID"] for body in self.state.bodies], dtype=np.int32)
        volume = np.asarray([body["volume"] for body in self.state.bodies], dtype=np.float64)
        mass_matrix = np.asarray([body["mass_matrix"] for body in self.state.bodies], dtype=np.float64)
        young = np.asarray([body["young"] for body in self.state.bodies], dtype=np.float64)
        friction = np.asarray([body["mu"] for body in self.state.bodies], dtype=np.float64)
        force_damping = np.asarray([body.get("force_damping", 0.0) for body in self.state.bodies], dtype=np.float64)
        torque_damping = np.asarray([body.get("torque_damping", 0.0) for body in self.state.bodies], dtype=np.float64)
        return {
            "body_num": body_num,
            "vertices": vertices,
            "faces": faces,
            "bodyID": body_ids,
            "groupID": group_ids,
            "faceBodyID": face_body_ids,
            "faceGroupID": face_group_ids,
            "faceMaterialID": face_material_ids,
            "bodyGroupID": body_group_ids,
            "materialID": body_material_ids,
            "volume": volume,
            "mass_matrix": mass_matrix,
            "young_modulus": young,
            "friction": friction,
            "forceLocalDamping": force_damping,
            "torqueLocalDamping": torque_damping,
            "y": self.state.y.copy(),
            "v_y": self.state.v_y.copy(),
            "center": centers,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "radius": radius,
        }

    def contact_snapshot(self):
        if self.operator is None:
            return {
                "vertex": np.zeros(0, dtype=np.int32),
                "face": np.zeros(0, dtype=np.int32),
                "edge0": np.zeros(0, dtype=np.int32),
                "edge1": np.zeros(0, dtype=np.int32),
                "search": "",
                "vf_count": 0,
                "ee_count": 0,
            }
        self.operator.refresh_neighbor_candidates(self.state.y, self.operator.dhat)
        neighbor = self.operator.neighbor
        vf_count = int(neighbor.candidate_count[None])
        ee_count = int(neighbor.edge_candidate_count[None])
        return {
            "vertex": np.ascontiguousarray(neighbor.candidate_vertex.to_numpy()[:vf_count]),
            "face": np.ascontiguousarray(neighbor.candidate_face.to_numpy()[:vf_count]),
            "edge0": np.ascontiguousarray(neighbor.candidate_edge0.to_numpy()[:ee_count]),
            "edge1": np.ascontiguousarray(neighbor.candidate_edge1.to_numpy()[:ee_count]),
            "search": str(neighbor.last_mode),
            "vf_count": vf_count,
            "ee_count": ee_count,
        }

    def _save_vtk(self, sims, vertices, faces, body_ids, group_ids, face_body_ids, face_group_ids, face_material_ids):
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            return
        vtk_path = os.path.join(sims.path, "vtks", f"GraphicAffineBody{sims.current_print:06d}")
        nface = int(faces.shape[0])
        unstructuredGridToVTK(
            vtk_path,
            np.ascontiguousarray(vertices[:, 0]),
            np.ascontiguousarray(vertices[:, 1]),
            np.ascontiguousarray(vertices[:, 2]),
            connectivity=np.ascontiguousarray(faces.reshape(-1).astype(np.int32)),
            offsets=np.ascontiguousarray(np.arange(3, 3 * nface + 1, 3, dtype=np.int32)),
            cell_types=np.ascontiguousarray(np.repeat(VtkTriangle.tid, nface).astype(np.uint8)),
            pointData={
                "bodyID": np.ascontiguousarray(body_ids.astype(np.int32)),
                "groupID": np.ascontiguousarray(group_ids.astype(np.int32)),
            },
            cellData={
                "bodyID": np.ascontiguousarray(face_body_ids),
                "groupID": np.ascontiguousarray(face_group_ids),
                "materialID": np.ascontiguousarray(face_material_ids),
            },
        )


__all__ = ["AffineBodyEngine"]
