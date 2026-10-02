import time

import taichi as ti

from src.mpm.engines.Engine import Engine
from src.mpm.GenerateManager import GenerateManager
from src.mpm.Recorder import WriteFile
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverConsole import print_save_file_info, print_simulation_start
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.TimeTicker import advance_time, has_remaining_time, time_tolerance
from src.utils.constants import Threshold


class Solver:
    sims: Simulation
    generator: GenerateManager
    engine: Engine
    recorder: WriteFile

    def __init__(self, sims, generator, engine, recorder):
        self.sims = sims
        self.generator = generator
        self.engine = engine
        self.recorder = recorder

        self.last_save_time = 0.0
        self.last_print_time = 0.0
        self.postprocess = []

    def set_callback_function(self, functions):
        self.postprocess.extend(normalize_callbacks(functions, ti.kernel))

    def save_file(self, scene):
        print_save_file_info(
            "MPM",
            self.sims.current_step,
            self.sims.current_print,
            self.sims.current_time,
            self.sims.path,
        )
        with self.sims.timer.section("Output"):
            self.recorder.output(self.sims, scene)
        self.sims.timer.profile0()
        self.sims.current_print += 1
        self.last_save_time = 1.0 * self.sims.current_time
        print("\n")

    def compile(self, scene, neighbor):
        print("Compiling first ... ...")
        start_time = time.time()
        step_dt = self.sims.delta
        self.core(scene, neighbor)
        # JIT warm-up executes a complete physical step, so it must be counted.
        advance_time(self.sims, step_dt)
        self.sims.current_step += 1
        end_time = time.time()
        print(f"Compiling time = {end_time - start_time} \n")
        self.sims.timer.profile1()

    def Solver(self, scene: myScene, neighbor):
        print_simulation_start("MPM")
        ti.sync()

        self.engine.pre_calculation(self.sims, scene, neighbor)
        if self.sims.current_time < Threshold:
            self.save_file(scene)
            self.last_save_time = 1.0 * self.sims.current_time

        if has_remaining_time(self.sims.current_time, self.sims.time, self.sims.delta):
            self.sims.set_timestep(min(self.sims.delta, self.sims.time - self.sims.current_time))
            compile_step = self.sims.delta
            self.compile(scene, neighbor)
            if self.sims.current_time - self.last_save_time >= self.sims.save_interval - 0.1 * compile_step:
                self.save_file(scene)
        start_time = time.time()
        while has_remaining_time(self.sims.current_time, self.sims.time, self.sims.delta):
            self.sims.set_timestep(min(self.sims.delta, self.sims.time - self.sims.current_time))
            step_dt = self.sims.delta
            self.core(scene, neighbor)

            new_body = self.generator.regenerate(scene)
            advance_time(self.sims, step_dt)
            self.sims.current_step += 1
            if self.sims.current_time - self.last_save_time >= self.sims.save_interval - 0.1 * step_dt or new_body:
                self.save_file(scene)

            runtime_checkpoint()

        if abs(self.sims.time - self.sims.current_time) <= time_tolerance(self.sims.delta):
            self.sims.current_time = self.sims.time

        end_time = time.time()

        if has_remaining_time(self.last_save_time, self.sims.current_time, self.sims.delta):
            self.save_file(scene)

        print("Physical time = ", end_time - start_time)
        print("#", " End Simulation ".center(67, "="), "#", "\n")

    def Visualize(self, scene: myScene, neighbor):
        from src.visualization.solver_adapters import run_mpm_gui

        return run_mpm_gui(self, scene, neighbor)

    def core(self, scene: myScene, neighbor):
        self.sims.timer.begin("Grid reset")
        self.engine.reset_grid_messages(scene)
        self.sims.timer.end("Grid reset")
        self.engine.bulid_neighbor_list(self.sims, scene, neighbor)
        self.engine.compute(self.sims, scene, neighbor)
        self.sims.timer.begin("Adaptive time step")
        self.engine.adaptive_timestep(self.sims, scene)
        self.sims.timer.end("Adaptive time step")
        self.sims.timer.begin("Postprocess")
        for functions in self.postprocess:
            functions()
        self.sims.timer.end("Postprocess")
