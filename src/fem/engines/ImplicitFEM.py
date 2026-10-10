"""Device-resident sparse Newton FEM with Newmark dynamics and line search."""

from __future__ import annotations

import math
import time

import taichi as ti

from src.fem.engines.FEMSolver import FEMSolver
from src.fem.engines.LineSearch import ArmijoLineSearch, LineSearchError
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.StepRetry import StepRetryPolicy, nonlinear_failure_kind


class NewtonConvergenceError(RuntimeError):
    pass


class ImplicitFEM(FEMSolver):
    def __init__(self, mesh, material, dirichlet=None, neumann=None, **kwargs):
        super().__init__(mesh, material, dirichlet, neumann, **kwargs)
        self.dt = float(kwargs.get("dt", kwargs.get("time_step", 1.0)))
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("implicit FEM time step/load increment must be finite and positive")
        requested_steps = kwargs.get("step", kwargs.get("steps", None))
        simulation_time = kwargs.get("simulation_time", kwargs.get("time", None))
        self._requested_simulation_time = None if simulation_time is None else float(simulation_time)
        if requested_steps is None:
            requested_steps = 1 if simulation_time is None else int(math.ceil(float(simulation_time) / self.dt))
        self.total_step = int(requested_steps)
        if self.total_step < 0:
            raise ValueError("number of implicit FEM steps cannot be negative")
        self.quasi_static = bool(kwargs.get("quasi_static", kwargs.get("static", False)))
        newmark = tuple(kwargs.get("newmark", (0.25, 0.5)))
        if len(newmark) == 2:
            self.beta, self.gamma = map(float, newmark)
        elif len(newmark) >= 3:
            self.beta, self.gamma = float(newmark[1]), float(newmark[2])
        else:
            raise ValueError("newmark must contain [beta, gamma] or [alpha, beta, gamma]")
        if self.beta <= 0.0 or self.gamma <= 0.0:
            raise ValueError("Newmark beta and gamma must be positive")
        self.max_iterations = int(kwargs.get("max_iterations", kwargs.get("max_iters", 40)))
        self.residual_tolerance = float(kwargs.get("residual_tolerance", kwargs.get("residual", 1.0e-8)))
        self.absolute_tolerance = float(kwargs.get("absolute_tolerance", 1.0e-10))
        self.correction_velocity_tolerance = float(
            kwargs.get(
                "correction_velocity_tolerance",
                kwargs.get("newton_velocity_tolerance", 1.0e-7),
            )
        )
        self.raise_on_nonconvergence = bool(kwargs.get("raise_on_nonconvergence", True))
        self.minimum_jacobian = float(kwargs.get("minimum_jacobian", 1.0e-8))
        self.use_line_search = bool(kwargs.get("line_search", True))
        self.line_search = ArmijoLineSearch(
            reduction=kwargs.get("line_search_reduction", 0.5),
            sufficient_decrease=kwargs.get("line_search_sufficient_decrease", 1.0e-4),
            max_backtracks=kwargs.get("line_search_max_backtracks", 24),
            minimum_step=kwargs.get("line_search_minimum_step", 1.0e-10),
        )
        self.contact_assembler = None
        self.step_retry = StepRetryPolicy(
            enabled=kwargs.get("enable_step_retry", False),
            maximum_retries=kwargs.get("step_retry_max_retries", 2),
            reduction=kwargs.get("step_retry_reduction", 0.5),
            minimum_timestep=kwargs.get("step_retry_minimum_timestep", 0.0),
        )
        self.last_failure = None
        self.last_step_record = None
        self.last_friction_iterations = 0
        self.last_friction_residual = math.inf
        self.last_friction_converged = False
        if self.max_iterations < 0:
            raise ValueError("implicit FEM max_iterations must be non-negative")
        if not math.isfinite(self.residual_tolerance) or self.residual_tolerance < 0.0:
            raise ValueError("implicit FEM residual_tolerance must be finite and non-negative")
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0.0:
            raise ValueError("implicit FEM absolute_tolerance must be finite and non-negative")
        if not math.isfinite(self.correction_velocity_tolerance) or self.correction_velocity_tolerance <= 0.0:
            raise ValueError("implicit FEM correction_velocity_tolerance must be finite " "and positive")
        contact = kwargs.get("contact")
        if contact is not None:
            if not self.use_line_search:
                raise ValueError(
                    "implicit FEM contact requires line_search=True for " "CCD/swept contact globalization"
                )
            from src.fem.contact.ContactAssembler import (
                FEMContactAssembler,
                FEMMultiContactAssembler,
            )

            contacts = contact.contacts_for_mesh(self.mesh)
            assemblers = [FEMContactAssembler(self.mesh, self.material, entry) for entry in contacts]
            self.contact_assembler = (
                assemblers[0] if len(assemblers) == 1 else FEMMultiContactAssembler(contact, assemblers)
            )
            if self.step_retry.enabled and self.contact_assembler.is_augmented_lagrangian:
                raise ValueError(
                    "FEM step retry does not support augmented-Lagrangian "
                    "contact because its multipliers advance inside Newton "
                    "iterations"
                )

    def _validate_pcg_projection(self):
        if getattr(self, "linear_solver", None) != "PCG":
            return
        if not getattr(self, "project_pd", False):
            raise ValueError("FEM PCG requires project_pd=True")
        if (
            self.contact_assembler is not None
            and self.contact_assembler.is_ipc
            and not all(
                entry.contact.project_pd
                for entry in getattr(
                    self.contact_assembler,
                    "assemblers",
                    (self.contact_assembler,),
                )
            )
        ):
            raise ValueError("FEM PCG with IPC contact requires contact project_pd=True")

    # Host-output compatibility path. Production Newton iterations call the
    # device methods supplied by ClassicalFEM/ClothFEM mixins instead.
    def _combine_contact_assembly(self, mechanical, positions, need_stiffness):
        if self.contact_assembler is None:
            return mechanical
        return self.contact_assembler.assemble_output(positions, mechanical, need_stiffness=need_stiffness)

    def _prepare_contact_iteration_device(self, positions, end_positions=None):
        if self.contact_assembler is not None:
            self.contact_assembler.prepare_iteration_device(positions, end_positions=end_positions)

    def _prepare_contact_iteration(self, positions):
        """Host diagnostic adapter retained for explicit user inspections."""
        if self.contact_assembler is not None:
            self.contact_assembler.prepare_iteration(positions)

    def _maximum_admissible_step_device(self):
        if self.contact_assembler is None:
            return 1.0
        return self.contact_assembler.maximum_admissible_step_device(self.state.position, self.state.direction)

    def _after_nonlinear_update_device(self, step):
        if self.contact_assembler is not None:
            self.contact_assembler.accept_update_device(self.state.position, step)

    def _contact_converged_device(self):
        return self.contact_assembler is None or self.contact_assembler.prepared_converged_device()

    def _dynamic_diagonal_factor(self):
        if self.quasi_static:
            return 0.0
        return 1.0 / (self.beta * self.dt**2) + (self.damping * self.gamma / (self.beta * self.dt))

    def _current_potential(self, positions):
        self.state.reduce_dynamic_potential(
            positions,
            self.damping,
            self.dt,
            self.beta,
            self.gamma,
            int(self.quasi_static),
        )
        return self._internal_energy_device() + float(self.state.dynamic_potential[None])

    def _trial_potential(self):
        minimum = self._minimum_jacobian_ratio_device(self.state.trial_position)
        if minimum <= self.minimum_jacobian:
            return math.inf
        freeze_al_self_contact = (
            self.contact_assembler is not None
            and self.contact_assembler.is_augmented_lagrangian
            and self.contact_assembler.contact.self_contact
        )
        if not freeze_al_self_contact:
            self._prepare_contact_iteration_device(self.state.trial_position)
        self._assemble_internal_device_at(self.state.trial_position, need_stiffness=False)
        value = self._current_potential(self.state.trial_position)
        return value if math.isfinite(value) else math.inf

    def _line_search_device(self):
        self.state.reduce_residual_and_slope()
        slope = float(self.state.directional_derivative[None])
        if not math.isfinite(slope) or slope >= 0.0:
            raise LineSearchError("Newton direction is not a finite descent direction")
        initial = self._current_potential(self.state.position)
        step = min(1.0, float(self._maximum_admissible_step_device()))
        last_value = math.inf
        last_trial_step = step
        for backtrack in range(self.line_search.max_backtracks + 1):
            if step < self.line_search.minimum_step:
                break
            last_trial_step = step
            self.state.set_trial_position(step)
            value = self._trial_potential()
            last_value = value
            # Parallel energy reductions vary at roundoff scale. Relax Armijo
            # only when its predicted decrease is equally unresolvable.
            roundoff = (1.0e-12 if self.state.real_type == ti.f64 else 1.0e-6) * max(
                1.0,
                abs(initial),
                abs(value),
            )
            if self.line_search.accepts(initial, value, step, slope, roundoff):
                self.state.accept_trial_position()
                return step, backtrack, value
            step *= self.line_search.reduction
        raise LineSearchError(
            "Armijo line search failed after "
            f"{self.line_search.max_backtracks + 1} trials; "
            f"initial={initial:.16e}, last={last_value:.16e}, "
            f"slope={slope:.16e}, last step={last_trial_step:.3e}"
        )

    def _solve_direction_device(self, stiffness):
        factor = self._dynamic_diagonal_factor()
        if factor:
            stiffness.set_mass_diagonal(self.state.mass, factor)
        else:
            stiffness.clear_additional_diagonal()
        try:
            stiffness.solve_device(
                self.state.residual,
                self.state.direction,
                self.state.constrained,
            )
        except RuntimeError as exc:
            raise NewtonConvergenceError(
                "implicit FEM tangent solve failed; check constraints, " "project_pd, and the selected linear solver"
            ) from exc
        self.state.reduce_residual_and_slope()
        slope = float(self.state.directional_derivative[None])
        if not math.isfinite(slope) or slope >= 0.0:
            raise NewtonConvergenceError("implicit FEM tangent has no finite descent direction")

    def _solve_frozen_friction_newton_device(self, next_time):
        iteration_history = []
        converged = False
        initial_norm = getattr(self, "_friction_force_reference", None)
        semi_progress = 0.0
        for iteration in range(self.max_iterations + 1):
            if (
                self.contact_assembler is not None
                and self.contact_assembler.is_augmented_lagrangian
                and iteration > 1
                and semi_progress > 0.999
            ):
                converged = True
                break
            self._prepare_contact_iteration_device(self.state.position)
            internal_force = self._assemble_internal_device(need_stiffness=True)
            stiffness = self._current_stiffness
            self.state.assemble_implicit_residual(
                internal_force,
                self.damping,
                self.dt,
                self.beta,
                self.gamma,
                int(self.quasi_static),
            )
            residual_norm = self.state.residual_norm()
            if initial_norm is None:
                initial_norm = max(residual_norm, self.state.external_norm(), 1.0)
                self._friction_force_reference = initial_norm
            iteration_record = {
                "iteration": iteration,
                "residual_norm": residual_norm,
                "relative_residual": residual_norm / initial_norm,
                "residual_tolerance": (self.absolute_tolerance + self.residual_tolerance * initial_norm),
                "convergence_reason": None,
                "line_search_step": 0.0,
                "line_search_backtracks": 0,
            }
            if self.contact_assembler is not None:
                iteration_record.update(self.contact_assembler.device_diagnostics())
            iteration_history.append(iteration_record)
            if (
                residual_norm <= self.absolute_tolerance + self.residual_tolerance * initial_norm
                and self._contact_converged_device()
            ):
                iteration_record["convergence_reason"] = "force_residual"
                converged = True
                break
            if iteration == self.max_iterations:
                break
            self._solve_direction_device(stiffness)
            correction_norm = self.state.direction_inf_norm()
            correction_velocity = correction_norm / self.dt
            iteration_record["correction_inf_norm"] = correction_norm
            iteration_record["correction_velocity"] = correction_velocity
            iteration_record["correction_velocity_tolerance"] = self.correction_velocity_tolerance
            if correction_velocity <= self.correction_velocity_tolerance and self._contact_converged_device():
                iteration_record["convergence_reason"] = "physical_correction_velocity"
                converged = True
                break
            if self.use_line_search:
                try:
                    step, backtracks, _ = self._line_search_device()
                except LineSearchError as exc:
                    raise NewtonConvergenceError(
                        "implicit FEM line search failed at time " f"{next_time:.6g}, iteration {iteration}"
                    ) from exc
            else:
                self.state.set_trial_position(1.0)
                if self._minimum_jacobian_ratio_device(self.state.trial_position) <= self.minimum_jacobian:
                    raise NewtonConvergenceError(
                        "full Newton step inverted an element; enable " "line_search or reduce the load step"
                    )
                self.state.accept_trial_position()
                step, backtracks = 1.0, 0
            iteration_record["line_search_step"] = step
            iteration_record["line_search_backtracks"] = backtracks
            self.state.apply_boundary(self.dt, 0)
            self._after_nonlinear_update_device(step)
            if self.contact_assembler is not None and self.contact_assembler.is_augmented_lagrangian:
                semi_progress += (1.0 - semi_progress) * step

        if not converged and self.raise_on_nonconvergence:
            final_iteration = iteration_history[-1]
            correction = final_iteration.get("correction_velocity", math.inf)
            tail = ", ".join(
                f"{entry['iteration']}:r={entry['residual_norm']:.3e},"
                f"cv={entry.get('correction_velocity', math.inf):.3e},"
                f"a={entry['line_search_step']:.3e},bt={entry['line_search_backtracks']},"
                f"c={entry.get('active_contacts', 0)}"
                for entry in iteration_history[-5:]
            )
            raise NewtonConvergenceError(
                f"implicit FEM did not converge at time {next_time:.6g}: "
                f"residual={final_iteration['residual_norm']:.3e}, "
                f"correction_velocity={correction:.3e} "
                f"after {self.max_iterations} iterations; tail=[{tail}]"
            )
        return converged, iteration_history

    def _updated_friction_residual_device(self):
        self._prepare_contact_iteration_device(self.state.position)
        internal_force = self._assemble_internal_device(need_stiffness=True)
        self.state.assemble_implicit_residual(
            internal_force,
            self.damping,
            self.dt,
            self.beta,
            self.gamma,
            int(self.quasi_static),
        )
        if self.state.residual_norm() == 0.0:
            return 0.0
        self._solve_direction_device(self._current_stiffness)
        return float(self.state.direction_inf_norm()) / self.dt

    def solve_lagged_friction_fixed_point(self, next_time):
        friction_active = (
            self.contact_assembler is not None
            and self.contact_assembler.is_ipc
            and self.contact_assembler.activate_friction
        )
        settings = self.contact_assembler.contact if friction_active else None
        requested_outer = settings.friction_iterations if friction_active else 1
        outer_limit = settings.friction_max_iterations if requested_outer == -1 else requested_outer
        outer_history = []
        converged = True
        self._friction_force_reference = None
        self.last_friction_iterations = 0
        self.last_friction_residual = 0.0 if not friction_active else math.inf
        self.last_friction_converged = not friction_active
        for outer in range(outer_limit):
            outer_converged, iteration_history = self._solve_frozen_friction_newton_device(next_time)
            outer_history.append(iteration_history)
            converged = converged and outer_converged
            if not outer_converged:
                break
            if friction_active:
                self.last_friction_iterations = outer + 1
                self.contact_assembler.refresh_friction_device(self.state.position)
                self.last_friction_residual = self._updated_friction_residual_device()
                if not math.isfinite(self.last_friction_residual):
                    raise NewtonConvergenceError("FEM lagged IPC friction residual is non-finite")
                if self.last_friction_residual <= settings.friction_tolerance:
                    self.last_friction_converged = True
                    break

        if requested_outer == -1 and not self.last_friction_converged:
            raise NewtonConvergenceError(
                "FEM lagged IPC friction fixed point did not converge within "
                f"{outer_limit} iterations (residual={self.last_friction_residual:.6e}, "
                f"tolerance={settings.friction_tolerance:.6e})"
            )
        return converged, outer_history

    def _substep_once(self, record_history=True):
        self.state.save_step_state()
        try:
            if self.contact_assembler is not None:
                self.contact_assembler.begin_step_device(self.state.old_position, self.dt)
            next_time = self.time + self.dt
            self.set_boundary_data_step(next_time, self.step_count + 1)
            self.state.apply_boundary(self.dt, 0)
            self.update_external_force_step(next_time, self.step_count + 1)
            self.state.build_newmark_prediction(self.dt, self.beta)

            converged, outer_history = self.solve_lagged_friction_fixed_point(next_time)
            friction_active = (
                self.contact_assembler is not None
                and self.contact_assembler.is_ipc
                and self.contact_assembler.activate_friction
            )

            self._prepare_contact_iteration_device(self.state.position)
            final_internal = self._assemble_internal_device(need_stiffness=False)
            self.state.finalize_newmark(
                self.dt,
                self.beta,
                self.gamma,
                int(self.quasi_static),
            )
            self.state.assemble_equilibrium(
                final_internal,
                0.0 if self.quasi_static else self.damping,
                int(not self.quasi_static),
            )
            minimum_jacobian = self._minimum_jacobian_ratio_device(self.state.position)
            self.time = next_time
            self.step_count += 1
            record = {
                "step": self.step_count,
                "time": self.time,
                "converged": converged,
                "iterations": outer_history[-1],
                "friction_iterations": outer_history if friction_active else [],
                "friction_residual": self.last_friction_residual,
                "friction_converged": self.last_friction_converged,
                "minimum_jacobian": minimum_jacobian,
            }
            if self.contact_assembler is not None:
                record["contact"] = self.contact_assembler.device_diagnostics()
            self.last_step_record = record
            if record_history:
                self.step_schedule.append_history(self.history, record)
            return converged
        except BaseException:
            self.state.restore_step_state()
            raise

    def _set_timestep(self, timestep):
        self.dt = float(timestep)

    def _failure_diagnostics(self, exception, attempt, timestep):
        return {
            "kind": nonlinear_failure_kind(exception),
            "exception": type(exception).__name__,
            "message": str(exception),
            "attempt": int(attempt),
            "timestep": float(timestep),
            "time": float(self.time),
            "step": int(self.step_count),
        }

    def diagnostics_snapshot(self):
        contact = None
        if self.contact_assembler is not None:
            try:
                contact = self.contact_assembler.device_diagnostics()
            except Exception as diagnostic_error:
                contact = {"unavailable": str(diagnostic_error)}
        return {
            "schema_version": 1,
            "subsystem": "fem_implicit",
            "time": float(self.time),
            "step": int(self.step_count),
            "timestep": float(self.dt),
            "contact": contact,
            "friction": {
                "iterations": int(self.last_friction_iterations),
                "residual": float(self.last_friction_residual),
                "converged": bool(self.last_friction_converged),
            },
            "last_failure": self.last_failure,
            "last_step": self.last_step_record,
        }

    def substep(self, record_history=True):
        original_timestep = self.dt
        attempt_timestep = original_timestep
        attempts = []
        for attempt in range(self.step_retry.maximum_retries + 1):
            self._set_timestep(attempt_timestep)
            try:
                result = self._substep_once(record_history=record_history)
            except NewtonConvergenceError as exception:
                failure = self._failure_diagnostics(exception, attempt, attempt_timestep)
                attempts.append(failure)
                next_timestep = self.step_retry.next_timestep(attempt_timestep, attempt)
                if next_timestep is None:
                    self.last_failure = {
                        **failure,
                        "original_timestep": float(original_timestep),
                        "attempts": attempts,
                    }
                    self._set_timestep(original_timestep)
                    raise
                attempt_timestep = next_timestep
                continue

            if self.last_step_record is not None:
                self.last_step_record["step_retry"] = {
                    "enabled": bool(self.step_retry.enabled),
                    "original_timestep": float(original_timestep),
                    "accepted_timestep": float(attempt_timestep),
                    "retry_count": int(attempt),
                    "attempts": attempts,
                }
            self.last_failure = None
            return result

        raise AssertionError("unreachable FEM retry state")

    def run(self, steps=None, verbose=True, postprocessing=()):
        postprocessing = normalize_callbacks(postprocessing)
        target_time = self._requested_simulation_time if steps is None else None
        target_step = (
            self.step_count + (self.total_step if steps is None else int(steps)) if target_time is None else None
        )
        overall_convergence = True
        if self.path is not None and self.step_count == 0:
            with self.timer.section("Output"):
                self.record()
            self.timer.profile0()
        while self.time < target_time - 1.0e-14 if target_time is not None else self.step_count < target_step:
            next_step = self.step_count + 1
            output_due = self.path is not None and next_step % self.output_interval == 0
            final_step = (
                self.time + self.dt >= target_time - 1.0e-14 if target_time is not None else next_step >= target_step
            )
            record_history = self.step_schedule.history_due(next_step, output=output_due, final=final_step)
            compiling = self.compile_seconds is None
            if compiling:
                print("Compiling first ... ...")
                compile_start = time.perf_counter()
            with self.timer.section("FEM implicit step"):
                converged = self.substep(record_history=record_history)
            if compiling:
                ti.sync()
                self.compile_seconds = time.perf_counter() - compile_start
                print(f"Compiling time = {self.compile_seconds} \n")
                self.timer.profile1()
            overall_convergence = overall_convergence and converged
            if self.path is not None and (self.step_count % self.output_interval == 0 or final_step):
                with self.timer.section("Output"):
                    self.record()
                self.timer.profile0()
            with self.timer.section("Postprocess"):
                for callback in postprocessing:
                    callback(self)
            if verbose:
                record = self.last_step_record
                final_iteration = record["iterations"][-1]
                print(
                    f"FEM implicit step {self.step_count}: "
                    f"t={self.time:.6g}, "
                    f"Newton={len(record['iterations']) - 1}, "
                    f"residual={final_iteration['residual_norm']:.4e}, "
                    f"min(J)={record['minimum_jacobian']:.4e}"
                )
            runtime_checkpoint()
        return self.result(overall_convergence)


__all__ = ["ImplicitFEM", "NewtonConvergenceError"]
