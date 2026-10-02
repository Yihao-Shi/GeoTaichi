"""Explicit two-way IGA--MPM stepping with DEM contact constitutive laws."""

from __future__ import annotations

import math
import time

import taichi as ti

from src.igampm.contact.ExplicitContact import ExplicitNurbsContact
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import StepSchedule, normalize_callbacks
from src.utils.SolverConsole import print_save_file_info
from src.utils.TimeTicker import Timer
from src.utils.linalg import no_operation


class ExplicitEngine:
    def __init__(self, iga_wrapper, mpm_wrapper, contactor, **kwargs):
        self.iga_wrapper = iga_wrapper
        self.mpm_wrapper = mpm_wrapper
        self.iga = iga_wrapper.engine
        self.mpm = mpm_wrapper.enginer
        self.contactor = contactor
        self.kwargs = dict(kwargs)
        self.contact = None
        self.initialized = False
        self.current_step = int(mpm_wrapper.sims.current_step)
        self.current_time = float(mpm_wrapper.sims.current_time)
        self.last_contact_count = 0
        self.last_recorded_step = None
        self.last_step_record = None
        self.timer = Timer()
        self.compile_seconds = None
        self.apply_iga_dirichlet = self.iga.apply_dirichlet_step
        self.output_interval = max(1, int(self.iga.output_interval))
        self.step_schedule = StepSchedule.from_options(kwargs, output_interval=self.output_interval)
        self.history = self.iga.history
        self.track_iga_energy = bool(self.iga.track_energy)
        self.track_contact_energy = bool(self.mpm_wrapper.sims.energy_tracking)
        requested_tracking = kwargs.get("track_energy")
        if requested_tracking is not None:
            requested_tracking = bool(requested_tracking)
            if requested_tracking != self.track_iga_energy:
                raise ValueError(
                    "explicit IGA-MPM track_energy must match the IGA solver "
                    "setting because IGA kernels are specialized at build time"
                )
            if requested_tracking != self.track_contact_energy:
                raise ValueError(
                    "explicit IGA-MPM track_energy must match the MPM solver "
                    "setting because DEM contact kernels are specialized at "
                    "build time"
                )
        self.track_energy = self.track_iga_energy or self.track_contact_energy
        self.sample_iga_energy = self.iga.sample_energy_step
        self.add_iga_energy_record = self.iga.add_energy_record
        self.add_contact_energy_record = self._add_contact_energy_record if self.track_contact_energy else no_operation
        self.record_mpm_frame = no_operation
        if self.mpm_wrapper.recorder is not None:
            self.record_mpm_frame = self._record_mpm_frame

    @property
    def dt(self):
        return float(self.mpm_wrapper.sims.dt[None])

    def _save_path(self):
        iga_path = getattr(self.iga, "path", None)
        mpm_path = getattr(self.mpm_wrapper.sims, "path", None)
        if iga_path == mpm_path or mpm_path is None:
            return iga_path
        if iga_path is None:
            return mpm_path
        return f"IGA={iga_path}, MPM={mpm_path}"

    def initialize(self):
        if self.initialized:
            return
        # Compile the specialized point--NURBS kernels before the much larger
        # IGA quadrature/MPM transfer programs.  The warm-up disables contact
        # at runtime, so it allocates no history and applies no force.
        self.contact = ExplicitNurbsContact(
            self.iga,
            self.mpm_wrapper.scene,
            self.contactor.phys,
            **self.kwargs,
        )
        self.contact.compile_kernels(self.mpm_wrapper.sims.dt)
        self.iga.precompute()
        self.apply_iga_dirichlet()
        self.mpm.pre_calculation(
            self.mpm_wrapper.sims,
            self.mpm_wrapper.scene,
            self.mpm_wrapper.neighbor,
        )
        self.check_critical_timestep()
        self.initialized = True

    def _prepare_mpm_step(self):
        scene = self.mpm_wrapper.scene
        sims = self.mpm_wrapper.sims
        self.mpm.reset_grid_message(scene)
        self.mpm.reset_particle_message(scene)
        requires_rebuild = self.mpm.is_verlet_update(scene) == 1
        if requires_rebuild:
            self.mpm.execute_board_serach(sims, scene, self.mpm_wrapper.neighbor)
        # Lagrangian coupling moves mass/momentum P2G to this hook.  For
        # engines where it is a no-op, the unconditional call is harmless.
        self.mpm.system_resolve(sims, scene)

    def step(self, record_history=True):
        self.initialize()
        self.iga.assemble_force()
        self._prepare_mpm_step()
        self.last_contact_count = self.contact.resolve(self.mpm_wrapper.sims.dt)
        self.mpm.compute(
            self.mpm_wrapper.sims,
            self.mpm_wrapper.scene,
            self.mpm_wrapper.neighbor,
        )
        self.iga.advance()
        self.current_step += 1
        self.current_time += self.dt
        self.iga.step_count = self.current_step
        self.iga.time = self.current_time
        self.mpm_wrapper.sims.current_step += 1
        self.mpm_wrapper.sims.current_time += self.dt
        record = {
            "step": int(self.current_step),
            "time": float(self.current_time),
            "active_contacts": int(self.last_contact_count),
        }
        self.last_step_record = record
        if record_history:
            self.sample_iga_energy()
            self.add_iga_energy_record(record)
            self.add_contact_energy_record(record)
            self.step_schedule.append_history(self.history, record)
        return self.last_contact_count

    def _add_contact_energy_record(self, record):
        record.update(self.contact.read_contact_energy())

    def critical_timestep(self):
        scene = self.mpm_wrapper.scene
        return self.contactor.phys.critical_timestep(scene.find_particle_min_mass(), scene.find_particle_max_radius())

    def check_critical_timestep(self):
        """Apply the native MPM CFL factor to material/contact estimates."""
        scene = self.mpm_wrapper.scene
        sims = self.mpm_wrapper.sims
        stable = float(sims.CFL) * min(
            float(scene.get_critical_timestep()),
            float(self.critical_timestep()),
        )
        if stable < self.dt:
            print("The IGA-MPM time step is corrected as:", stable, "\n")
            sims.set_timestep(stable)
            self.iga.dt = stable
        return self.dt

    def record_frame(self):
        self.iga.step_count = self.current_step
        self.iga.time = self.current_time
        print_save_file_info(
            "IGAMPM",
            self.current_step,
            self.iga.output_count,
            self.current_time,
            self._save_path(),
        )
        self.iga.record(log=False)
        self.record_mpm_frame()
        self.mpm_wrapper.sims.current_print += 1
        self.last_recorded_step = self.current_step

    def _record_mpm_frame(self):
        self.mpm_wrapper.recorder.output(self.mpm_wrapper.sims, self.mpm_wrapper.scene)

    def run(self, steps=None, verbose=True, record=True, postprocessing=()):
        if self.compile_seconds is None:
            with self.timer.section("IGAMPM initialization"):
                self.initialize()
        else:
            self.initialize()
        if steps is None:
            remaining = max(
                0.0,
                float(self.mpm_wrapper.sims.time) - float(self.mpm_wrapper.sims.current_time),
            )
            steps = int(math.ceil(remaining / self.dt - 1.0e-12))
        steps = int(steps)
        if steps < 0:
            raise ValueError("explicit IGA-MPM steps must be non-negative")
        callbacks = normalize_callbacks(postprocessing)
        if record and self.current_step == 0 and self.last_recorded_step is None:
            with self.timer.section("Output"):
                self.record_frame()
            self.timer.profile0()
        for local_step in range(steps):
            next_step = self.current_step + 1
            output_due = record and next_step % self.output_interval == 0
            final_step = local_step + 1 == steps
            record_history = self.step_schedule.history_due(next_step, output=output_due, final=final_step)
            compiling = self.compile_seconds is None
            if compiling:
                print("Compiling first ... ...")
                compile_start = time.perf_counter()
            with self.timer.section("IGAMPM explicit step"):
                self.step(record_history=record_history)
            if compiling:
                ti.sync()
                self.compile_seconds = time.perf_counter() - compile_start
                print(f"Compiling time = {self.compile_seconds} \n")
                self.timer.profile1()
            with self.timer.section("Postprocess"):
                for callback in callbacks:
                    callback()
            if record and self.current_step % self.output_interval == 0:
                with self.timer.section("Output"):
                    self.record_frame()
                self.timer.profile0()
            runtime_checkpoint()
        ti.sync()
        if record and self.last_recorded_step != self.current_step:
            with self.timer.section("Output"):
                self.record_frame()
            self.timer.profile0()
        if verbose:
            print(
                "IGA-MPM explicit steps =",
                steps,
                "time =",
                self.current_time,
                "active contacts =",
                self.last_contact_count,
            )
        return self

    def diagnostics_snapshot(self):
        return {
            "schema_version": 1,
            "subsystem": "igampm_explicit",
            "time": float(self.current_time),
            "step": int(self.current_step),
            "timestep": float(self.dt),
            "contact": {"active_count": int(self.last_contact_count)},
            "last_step": self.last_step_record,
        }


__all__ = ["ExplicitEngine"]
