import time

import taichi as ti

from src.dem.ContactManager import ContactManager
from src.dem.engines.ExplicitEngine import ExplicitEngine
from src.dem.GenerateManager import GenerateManager
from src.dem.Recorder import WriteFile
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverConsole import print_save_file_info, print_simulation_start
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.TimeTicker import advance_time, has_remaining_time, time_tolerance
from src.utils.constants import Threshold
from src.utils.linalg import no_operation


class Solver(object):
    sims: Simulation
    generator: GenerateManager
    contact: ContactManager
    engine: ExplicitEngine
    recorder: WriteFile

    def __init__(self, sims, generator, contact, engine, recorder):
        self.sims = sims
        self.generator = generator
        self.contact = contact
        self.engine = engine
        self.recorder = recorder

        self.last_save_time = 0.0
        self.last_print_time = 0.0
        self.calm_interval = 0
        self.last_calm = 0
        self.preintegration = []
        self.postprocess = []
        self.run_preintegration_callbacks = no_operation
        self.compile_seconds = 0.0
        self.physical_seconds = 0.0
        self.in_loop_output_seconds = 0.0
        self.solver_compute_seconds = 0.0

    def set_callback_function(self, functions):
        self.postprocess.extend(normalize_callbacks(functions, ti.kernel))

    def set_preintegration_callback_function(self, functions):
        self.preintegration.extend(normalize_callbacks(functions, ti.kernel))
        if self.preintegration:
            self.run_preintegration_callbacks = self._run_preintegration_callbacks

    def clear_preintegration_callback_functions(self):
        self.preintegration.clear()
        self.run_preintegration_callbacks = no_operation

    def _run_preintegration_callbacks(self):
        self.sims.timer.begin("Pre-integration diagnostics")
        for functions in self.preintegration:
            functions()
        self.sims.timer.end("Pre-integration diagnostics")

    def set_particle_calm(self, scene, calm_interval):
        if calm_interval:
            self.calm_interval = calm_interval
            self.postprocess.append(lambda: self.engine.calm(self.sims.current_step, self.calm_interval, scene))

    def save_file(self, scene):
        print_save_file_info(
            "DEM",
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

    def compile(self, scene):
        print("Compiling first ... ...")
        start_time = time.perf_counter()
        step_dt = self.sims.delta
        self.core(scene)
        # JIT warm-up executes a complete physical step, so it must be counted.
        advance_time(self.sims, step_dt)
        self.sims.current_step += 1
        ti.sync()
        end_time = time.perf_counter()
        self.compile_seconds = end_time - start_time
        print(f"Compiling time = {self.compile_seconds} \n")
        self.sims.timer.profile1()

    def Solver(self, scene: myScene):
        print_simulation_start("DEM")
        ti.sync()

        self.engine.pre_calculation(self.sims, scene, self.contact.neighbor)
        if self.sims.current_time < Threshold:
            self.save_file(scene)
            self.last_save_time = 1.0 * self.sims.current_time

        if has_remaining_time(self.sims.current_time, self.sims.time, self.sims.delta):
            self.sims.set_timestep(min(self.sims.delta, self.sims.time - self.sims.current_time))
            compile_step = self.sims.delta
            self.compile(scene)
            if self.sims.current_time - self.last_save_time >= self.sims.save_interval - 0.1 * compile_step:
                self.save_file(scene)
        self.in_loop_output_seconds = 0.0
        self.solver_compute_seconds = 0.0
        ti.sync()
        start_time = time.perf_counter()
        while has_remaining_time(self.sims.current_time, self.sims.time, self.sims.delta):
            self.sims.set_timestep(min(self.sims.delta, self.sims.time - self.sims.current_time))
            step_dt = self.sims.delta
            self.core(scene)

            new_body = self.generator.regenerate(scene)
            advance_time(self.sims, step_dt)
            self.sims.current_step += 1
            if self.sims.current_time - self.last_save_time >= self.sims.save_interval - 0.1 * step_dt or new_body:
                ti.sync()
                output_start = time.perf_counter()
                self.save_file(scene)
                ti.sync()
                self.in_loop_output_seconds += time.perf_counter() - output_start
                if new_body:
                    self.engine.update_verlet_table(self.sims, scene, self.contact.neighbor)
                    self.sims.set_max_bounding_sphere_radius(scene.find_bounding_sphere_max_radius(self.sims))

            runtime_checkpoint()
        if abs(self.sims.time - self.sims.current_time) <= time_tolerance(self.sims.delta):
            self.sims.current_time = self.sims.time
        ti.sync()
        end_time = time.perf_counter()
        self.physical_seconds = end_time - start_time
        self.solver_compute_seconds = max(self.physical_seconds - self.in_loop_output_seconds, 0.0)

        if has_remaining_time(self.last_save_time, self.sims.current_time, self.sims.delta):
            self.save_file(scene)

        print("Simulation-loop time = ", self.physical_seconds)
        print("In-loop output time = ", self.in_loop_output_seconds)
        print("Solver compute time = ", self.solver_compute_seconds)
        print("#", " End Simulation ".center(67, "="), "#", "\n")

    def Visualize(self, scene: myScene):
        from src.visualization.solver_adapters import run_dem_gui

        return run_dem_gui(self, scene)

    def core(self, scene: myScene):
        self.sims.timer.begin("Reset")
        self.engine.reset_wall_message(scene)
        self.engine.reset_particle_message(scene)
        self.engine.reset_contact_energy()
        self.sims.timer.end("Reset")
        self.engine.update_neighbor_lists(self.sims, scene, self.contact.neighbor)
        self.run_preintegration_callbacks()
        self.engine.integration(self.sims, scene, self.contact.neighbor)
        self.sims.timer.begin("Adaptive time step")
        self.engine.adaptive_timestep(self.sims, scene)
        self.sims.timer.end("Adaptive time step")
        self.sims.timer.begin("Postprocess")
        for functions in self.postprocess:
            functions()
        self.sims.timer.end("Postprocess")


class AffineBodySolver(Solver):
    def Solver(self, scene: myScene):
        print_simulation_start("DEM")
        ti.sync()

        self.engine.initialize(self.sims, scene)
        if self.sims.current_time < Threshold:
            self.save_file(scene)
            self.last_save_time = 1.0 * self.sims.current_time

        start_time = time.time()
        target_time = float(self.sims.current_time + self.sims.time)
        if self.sims.current_time < target_time - 1.0e-14:
            self.compile(scene)
        while self.sims.current_time < target_time - 1.0e-14:
            self.core(scene)
            self.sims.current_time += self.sims.delta
            self.sims.current_step += 1
            runtime_checkpoint()

            if self.sims.current_time - self.last_save_time >= self.sims.save_interval - 0.1 * self.sims.delta:
                self.save_file(scene)
        end_time = time.time()
        self.physical_seconds = end_time - start_time

        if has_remaining_time(self.last_save_time, self.sims.current_time, self.sims.delta):
            self.save_file(scene)

        print("Physical time = ", self.physical_seconds)
        print("#", " End Affine Body Simulation ".center(67, "="), "#", "\n")

    def core(self, scene: myScene):
        self.sims.timer.begin("Affine body step")
        self.engine.step(self.sims, scene)
        self.sims.timer.end("Affine body step")
        self.sims.timer.begin("Postprocess")
        for functions in self.postprocess:
            functions()
        self.sims.timer.end("Postprocess")
