"""Time loop for explicit FEM--MPM coupling."""

from __future__ import annotations

import time

import taichi as ti

from src.utils.SolverConsole import print_save_file_info, print_simulation_start
from src.utils.SolverRuntime import normalize_callbacks


class Solver:
    def __init__(self, simulation, fem, mpm, engine, recorder):
        self.simulation = simulation
        self.fem = fem
        self.mpm = mpm
        self.engine = engine
        self.recorder = recorder
        self.postprocess = []
        self.last_save_time = 0.0
        self.compile_seconds = None

    def set_callback_function(self, functions):
        self.postprocess.extend(normalize_callbacks(functions, ti.kernel))

    def save_file(self):
        print_save_file_info(
            "FEMPM",
            self.simulation.current_step,
            self.simulation.current_print,
            self.simulation.current_time,
            self.simulation.path,
        )
        with self.simulation.timer.section("Output"):
            self.recorder.output()
        self.simulation.timer.profile0()
        self.simulation.current_print += 1
        self.mpm.sims.current_print += 1
        self.last_save_time = self.simulation.current_time

    def CouplingSolver(self, precalculated=False):
        print_simulation_start("FEMPM")
        if not precalculated:
            self.engine.pre_calculate()
        if self.simulation.current_time <= 1.0e-14:
            self.save_file()
        ti.sync()
        start = time.perf_counter()
        callbacks_require_diagnostics = bool(self.postprocess)
        schedule = self.simulation.step_schedule
        while self.simulation.current_time < self.simulation.time - 0.5 * self.simulation.delta:
            compiling = self.compile_seconds is None
            if compiling:
                print("Compiling first ... ...")
                compile_start = time.perf_counter()
            next_step = self.simulation.current_step + 1
            next_time = self.simulation.current_time + self.simulation.delta
            output_due = next_time - self.last_save_time >= self.simulation.save_interval - 0.1 * self.simulation.delta
            final_step = next_time >= self.simulation.time - 0.5 * self.simulation.delta
            update_diagnostics = schedule.diagnostics_due(
                next_step,
                output=output_due,
                callback_requires_diagnostics=callbacks_require_diagnostics,
                final=final_step,
            )
            check_jacobian = schedule.jacobian_due(next_step, output=output_due, final=final_step)
            record_history = schedule.history_due(next_step, output=output_due, final=final_step)
            self.engine.step(
                update_diagnostics=update_diagnostics,
                check_jacobian=check_jacobian,
            )
            with self.simulation.timer.section("Postprocess"):
                for callback in self.postprocess:
                    callback()
            if compiling:
                ti.sync()
                self.compile_seconds = time.perf_counter() - compile_start
                print(f"Compiling time = {self.compile_seconds} \n")
                self.simulation.timer.profile1()
            self.simulation.current_time += self.simulation.delta
            self.mpm.sims.current_time += self.simulation.delta
            self.fem.engine.time = self.simulation.current_time
            self.simulation.current_step += 1
            self.mpm.sims.current_step += 1
            self.fem.engine.step_count += 1
            if record_history:
                history_record = {
                    "step": self.fem.engine.step_count,
                    "time": self.fem.engine.time,
                    "minimum_jacobian": self.engine.minimum_jacobian,
                }
                if update_diagnostics and self.fem.engine.track_energy:
                    history_record["kinetic_energy"] = float(self.fem.engine.state.kinetic_energy[None])
                schedule.append_history(self.fem.engine.history, history_record)
            if output_due:
                self.save_file()
        ti.sync()
        if self.simulation.current_time - self.last_save_time > 0.9 * self.simulation.save_interval:
            self.save_file()
        print(f"FEMPM simulation-loop time = {time.perf_counter() - start:.6g} s")


__all__ = ["Solver"]
