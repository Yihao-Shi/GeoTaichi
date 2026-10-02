"""Driver loop and output scheduling for soft-affine IPC."""

import time

import taichi as ti

from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverConsole import print_save_file_info, print_simulation_start
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.constants import Threshold


class SoftAffineIPCSolver(object):
    def __init__(self, sims, generator, contact, engine, recorder):
        self.sims = sims
        self.generator = generator
        self.contact = contact
        self.engine = engine
        self.recorder = recorder
        self.last_save_time = 0.0
        self.postprocess = []
        self.compile_seconds = None

    def set_callback_function(self, functions):
        self.postprocess.extend(normalize_callbacks(functions, ti.kernel))

    def set_preintegration_callback_function(self, functions):
        if normalize_callbacks(functions):
            raise ValueError(
                "LSMPM soft-affine IPC does not expose a state between " "contact assembly and monolithic integration"
            )

    def clear_preintegration_callback_functions(self):
        return

    def set_particle_calm(self, scene, calm_interval):
        return

    def save_file(self, scene):
        print_save_file_info(
            "MPDEM",
            self.sims.current_step,
            self.sims.current_print,
            self.sims.current_time,
            self.sims.path,
        )
        with self.sims.timer.section("Output"):
            self.recorder.output(self.sims, scene)
            self.engine.save_affine(self.sims)
        self.sims.timer.profile0()
        self.sims.current_print += 1
        self.last_save_time = 1.0 * self.sims.current_time
        print("\n")

    def Solver(self, scene):
        print_simulation_start("MPDEM")
        ti.sync()
        self.engine.initialize(self.sims, scene)
        if self.sims.current_time < Threshold:
            self.save_file(scene)
        start_time = time.time()
        target_time = float(self.sims.current_time + self.sims.time)
        while self.sims.current_time < target_time - 1.0e-14:
            compiling = self.compile_seconds is None
            if compiling:
                print("Compiling first ... ...")
                compile_start = time.perf_counter()
            self.sims.timer.begin("Soft-Affine IPC step")
            self.engine.step(self.sims, scene)
            self.sims.timer.end("Soft-Affine IPC step")
            self.sims.timer.begin("Postprocess")
            for functions in self.postprocess:
                functions()
            self.sims.timer.end("Postprocess")
            if compiling:
                ti.sync()
                self.compile_seconds = time.perf_counter() - compile_start
                print(f"Compiling time = {self.compile_seconds} \n")
                self.sims.timer.profile1()
            self.sims.current_time += self.sims.delta
            self.sims.current_step += 1
            runtime_checkpoint()
            if self.sims.current_time - self.last_save_time >= self.sims.save_interval - 0.1 * self.sims.delta:
                self.save_file(scene)
        end_time = time.time()
        if abs(self.sims.current_time - self.last_save_time) > 0.9 * self.sims.save_interval:
            self.save_file(scene)
        print("Physical time = ", end_time - start_time)
        print("#", " End LSMPM Soft-Affine IPC Simulation ".center(67, "="), "#", "\n")


__all__ = ["SoftAffineIPCSolver"]
