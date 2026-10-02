"""Lumped-mass explicit total-Lagrangian FEM integrator."""

from __future__ import annotations

import math
import time

import taichi as ti

from src.fem.engines.FEMSolver import FEMSolver
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.linalg import no_operation


class ExplicitFEM(FEMSolver):
    def __init__(self, mesh, material, dirichlet=None, neumann=None, **kwargs):
        super().__init__(mesh, material, dirichlet, neumann, **kwargs)
        requested_dt = kwargs.get("dt", kwargs.get("time_step", None))
        self.automatic_time_step = requested_dt is None or str(requested_dt).lower() == "auto"
        self.cfl = float(kwargs.get("cfl", kwargs.get("CFL", 0.5)))
        self.critical_time_step = self.stable_time_step(self.cfl)
        if self.automatic_time_step:
            if not math.isfinite(self.critical_time_step):
                raise ValueError(
                    "automatic explicit FEM time-step estimation is unavailable "
                    "for this state-dependent material; specify a finite positive dt"
                )
            self.dt = self.critical_time_step
        else:
            self.dt = float(requested_dt)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("explicit FEM time step must be finite and positive")
        if self.dt > self.critical_time_step and kwargs.get("enforce_stable_time_step", False):
            raise ValueError(f"explicit dt={self.dt:.3e} exceeds estimated critical dt={self.critical_time_step:.3e}")
        simulation_time = kwargs.get("simulation_time", kwargs.get("time", None))
        requested_steps = kwargs.get("step", kwargs.get("steps", None))
        self._requested_steps_was_none = requested_steps is None
        self._requested_simulation_time = simulation_time
        if requested_steps is None:
            requested_steps = 1 if simulation_time is None else int(math.ceil(float(simulation_time) / self.dt))
        self.total_step = int(requested_steps)
        if self.total_step < 0:
            raise ValueError("number of explicit FEM steps cannot be negative")
        self.soft_particle_contact = None
        self.minimum_jacobian = 1.0
        self.advance_constitutive_state = no_operation
        self.assemble_explicit_internal_force = None
        self.resolve_soft_particle_contact_step = no_operation
        self.sample_energy_step = no_operation
        self.latest_energy = {}

    def _initialize_soft_particle_contact(self, kwargs):
        contact = kwargs.get("soft_particle_contact")
        if contact is None:
            return
        from src.fem.soft_particle.ContactManager import (
            FEMSoftParticleContactManager,
        )

        self.soft_particle_contact = FEMSoftParticleContactManager(
            self.mesh,
            self.state,
            contact,
            track_energy=self.track_energy,
        )
        contact_step = self.soft_particle_contact.critical_timestep()
        self.critical_time_step = min(self.critical_time_step, contact_step)
        if self.automatic_time_step:
            self.dt = self.critical_time_step
            if self._requested_steps_was_none and self._requested_simulation_time is not None:
                self.total_step = int(math.ceil(float(self._requested_simulation_time) / self.dt))
        if self.dt > self.critical_time_step and kwargs.get("enforce_stable_time_step", False):
            raise ValueError(
                f"explicit dt={self.dt:.3e} exceeds FEM/soft-contact critical " f"dt={self.critical_time_step:.3e}"
            )

    def bind_runtime_functions(self):
        """Freeze the concrete explicit operations after construction."""
        self.advance_constitutive_state = self._advance_constitutive_state
        self.assemble_explicit_internal_force = self._assemble_internal_force_device
        self.resolve_soft_particle_contact_step = (
            self._resolve_soft_particle_contact if self.soft_particle_contact is not None else no_operation
        )
        self.sample_energy_step = self._sample_energy if self.track_energy else no_operation

    def _sample_energy(self):
        self.state.reduce_kinetic_energy()
        energy = {
            "kinetic_energy": float(self.state.kinetic_energy[None]),
            "internal_energy": float(self._internal_energy_device()),
            "structural_damping_dissipation": float(self.state.damping_dissipation[None]),
        }
        if self.soft_particle_contact is not None:
            contact = self.soft_particle_contact.energy_diagnostics()
            energy.update(
                {
                    "contact_elastic_energy": contact["elastic_energy"],
                    "contact_friction_dissipation": contact["friction_dissipation"],
                    "contact_damping_dissipation": contact["damping_dissipation"],
                }
            )
        self.latest_energy = energy

    def _resolve_soft_particle_contact(
        self,
        *,
        advance_history=True,
        check_rebuild=True,
    ):
        self.soft_particle_contact.resolve(
            self.dt,
            advance_history=advance_history,
            check_rebuild=check_rebuild,
        )

    def substep(
        self,
        *,
        update_diagnostics=True,
        check_jacobian=True,
        record_history=True,
    ):
        self.prepare_explicit_step(self.time, self.step_count)
        self.resolve_soft_particle_contact_step(advance_history=True)
        self.advance_constitutive_state()
        internal_force = self.assemble_explicit_internal_force()
        self.state.explicit_update(
            internal_force,
            self.damping,
            self.dt,
            self.track_energy,
        )
        next_time = self.time + self.dt
        self.set_boundary_data_step(next_time, self.step_count + 1)
        self.apply_boundary_step(self.dt, 1)
        if check_jacobian:
            self.minimum_jacobian = self._minimum_jacobian_ratio_device(self.state.position)
            if self.minimum_jacobian <= 1.0e-10:
                raise RuntimeError(
                    f"explicit FEM produced an inverted/collapsed element at time {next_time:.6g}; "
                    "reduce dt or refine the mesh"
                )
        if update_diagnostics:
            self.update_external_force_step(next_time, self.step_count + 1)
            self.resolve_soft_particle_contact_step(advance_history=False)
            updated_internal = self.assemble_explicit_internal_force()
            self.state.assemble_equilibrium(updated_internal, self.damping, 0)
            self.sample_energy_step()
        self.time = next_time
        self.step_count += 1
        if record_history:
            record = {
                "step": self.step_count,
                "time": self.time,
                "minimum_jacobian": self.minimum_jacobian,
            }
            if update_diagnostics:
                record.update(self.latest_energy)
            self.step_schedule.append_history(self.history, record)

    def run(self, steps=None, verbose=True, postprocessing=()):
        steps = self.total_step if steps is None else int(steps)
        if self.path is not None and self.step_count == 0:
            with self.timer.section("Output"):
                self.record()
            self.timer.profile0()
        callbacks = normalize_callbacks(postprocessing)
        for local_step in range(steps):
            next_step = self.step_count + 1
            output_due = next_step % self.output_interval == 0
            final_step = local_step + 1 == steps
            update_diagnostics = self.step_schedule.diagnostics_due(
                next_step,
                output=output_due,
                callback_requires_diagnostics=bool(callbacks),
                final=final_step,
            )
            check_jacobian = self.step_schedule.jacobian_due(next_step, output=output_due, final=final_step)
            record_history = self.step_schedule.history_due(next_step, output=output_due, final=final_step)
            compiling = self.compile_seconds is None
            if compiling:
                print("Compiling first ... ...")
                compile_start = time.perf_counter()
            with self.timer.section("FEM explicit step"):
                self.substep(
                    update_diagnostics=update_diagnostics,
                    check_jacobian=check_jacobian,
                    record_history=record_history,
                )
            if compiling:
                ti.sync()
                self.compile_seconds = time.perf_counter() - compile_start
                print(f"Compiling time = {self.compile_seconds} \n")
                self.timer.profile1()
            if self.path is not None and self.step_count % self.output_interval == 0:
                with self.timer.section("Output"):
                    self.record()
                self.timer.profile0()
            with self.timer.section("Postprocess"):
                for callback in callbacks:
                    callback(self)
            if verbose and self.step_count % self.output_interval == 0:
                print(f"FEM explicit step {self.step_count}: t={self.time:.6g}, " f"min(J)={self.minimum_jacobian:.4e}")
            runtime_checkpoint()
        return self.result()


__all__ = ["ExplicitFEM"]
