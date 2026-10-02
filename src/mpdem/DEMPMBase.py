import time

import taichi as ti

from src.mpdem.GenerateManager import GenerateManager
from src.dem.SceneManager import myScene as DEMScene
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.Recorder import WriteFile as DEMWriteFile
from src.mpdem.Engine import Engine
from src.mpdem.Recorder import WriteFile
from src.mpdem.Simulation import Simulation
from src.mpm.SceneManager import myScene as MPMScene
from src.mpm.Simulation import Simulation as MPMSimulation
from src.mpm.Recorder import WriteFile as MPMWriteFile
from src.utils.constants import Threshold
from src.utils.ObjectIO import DictIO
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverConsole import print_save_file_info, print_simulation_start
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.TimeTicker import advance_time, has_remaining_time, time_tolerance


class Solver:
    sims: Simulation
    dsims: DEMSimulation
    msims: MPMSimulation
    drecorder: DEMWriteFile
    mrecorder: MPMWriteFile
    generator: GenerateManager
    engine: Engine
    recorder: WriteFile

    def __init__(self, sims, msims, dsims, mrecorder, drecorder, generator, engine, recorder):
        self.sims = sims
        self.dsims = dsims
        self.msims = msims
        self.mrecorder = mrecorder
        self.drecorder = drecorder
        self.engine = engine
        self.generator = generator
        self.recorder = recorder
        self.postprocess = []

        self.last_save_time = 0.0
        self.solve = None
        self.compile_seconds = 0.0
        self.physical_seconds = 0.0
        self.in_loop_output_seconds = 0.0
        self.solver_compute_seconds = 0.0

    def set_callback_function(self, functions):
        self.postprocess.extend(normalize_callbacks(functions, ti.kernel))

    def set_particle_calm(self, scene, calm_interval):
        if calm_interval:
            self.calm_interval = calm_interval
            self.postprocess.append(lambda: self.engine.dengine.calm(self.sims.current_step, self.calm_interval, scene))

    def save_file(self, mscene: MPMScene, dscene: DEMScene):
        print_save_file_info(
            self.sims.coupling_scheme,
            self.sims.current_step,
            self.sims.current_print,
            self.sims.current_time,
            self.sims.path,
        )
        with self.sims.timer.section("Output"):
            self.recorder.output(self.sims, self.msims, mscene, self.dsims, dscene)

        self.dsims.timer.profile0()
        self.msims.timer.profile0()
        self.sims.timer.profile0()

        self.sims.current_print += 1
        self.msims.current_print += 1
        self.dsims.current_print += 1
        self.last_save_time = 1.0 * self.sims.current_time
        print("\n")

    def compile(self):
        print("Compiling first ... ...")
        start_time = time.perf_counter()
        step_dt = self.sims.delta
        self.core()
        # JIT warm-up executes a complete coupled step on all three clocks.
        advance_time(self.sims, step_dt)
        advance_time(self.msims, step_dt)
        advance_time(self.dsims, step_dt)
        self.sims.current_step += 1
        ti.sync()
        end_time = time.perf_counter()
        self.compile_seconds = end_time - start_time
        print(f"Compiling time = {self.compile_seconds} \n")
        self.sims.timer.profile1()

    def CouplingSolver(self, mscene: MPMScene, dscene: DEMScene):
        print_simulation_start(self.sims.coupling_scheme)
        ti.sync()

        self.engine.pre_calculate()
        if self.sims.current_time < Threshold:
            self.save_file(mscene, dscene)
            self.last_save_time = 1.0 * self.sims.current_time

        if has_remaining_time(self.sims.current_time, self.sims.time, self.sims.delta):
            step_dt = min(self.sims.delta, self.sims.time - self.sims.current_time)
            self.sims.set_timestep(step_dt)
            self.msims.set_timestep(step_dt)
            self.dsims.set_timestep(step_dt)
            self.compile()
            if self.sims.current_time - self.last_save_time >= self.sims.save_interval - 0.1 * step_dt:
                self.save_file(mscene, dscene)
        self.in_loop_output_seconds = 0.0
        self.solver_compute_seconds = 0.0
        ti.sync()
        start_time = time.perf_counter()
        while has_remaining_time(self.sims.current_time, self.sims.time, self.sims.delta):
            step_dt = min(self.sims.delta, self.sims.time - self.sims.current_time)
            self.sims.set_timestep(step_dt)
            self.msims.set_timestep(step_dt)
            self.dsims.set_timestep(step_dt)
            self.core()

            new_body = self.generator.regenerate(self.sims, self.dsims, self.msims, mscene, dscene)
            advance_time(self.sims, step_dt)
            advance_time(self.msims, step_dt)
            advance_time(self.dsims, step_dt)
            self.sims.current_step += 1
            if self.sims.current_time - self.last_save_time >= self.sims.save_interval - 0.1 * step_dt or new_body:
                ti.sync()
                output_start = time.perf_counter()
                self.save_file(mscene, dscene)
                ti.sync()
                self.in_loop_output_seconds += time.perf_counter() - output_start
                if new_body:
                    self.dsims.set_max_bounding_sphere_radius(dscene.find_bounding_sphere_max_radius(self.dsims))

            runtime_checkpoint()
        if abs(self.sims.time - self.sims.current_time) <= time_tolerance(self.sims.delta):
            self.sims.current_time = self.sims.time
            self.msims.current_time = self.sims.time
            self.dsims.current_time = self.sims.time
        ti.sync()
        end_time = time.perf_counter()
        self.physical_seconds = end_time - start_time
        self.solver_compute_seconds = max(self.physical_seconds - self.in_loop_output_seconds, 0.0)

        if has_remaining_time(self.last_save_time, self.sims.current_time, self.sims.delta):
            self.save_file(mscene, dscene)
            self.engine.reset_message()

        print("Simulation-loop time = ", self.physical_seconds)
        print("In-loop output time = ", self.in_loop_output_seconds)
        print("Solver compute time = ", self.solver_compute_seconds)
        print("#", " End Simulation ".center(67, "="), "#", "\n")

    def core(self):
        self.engine.reset_message()
        self.engine.compute()
        self.engine.adaptive_timestep()
        for functions in self.postprocess:
            functions()
