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

        self.last_save_time = 0.
        self.solve = None

    def set_callback_function(self, functions):
        if not functions is None:
            if isinstance(functions, list):
                for f in functions:
                    self.postprocess.append(ti.kernel(f))
            elif isinstance(functions, dict):
                for f in functions.values():
                    self.postprocess.append(ti.kernel(f))
            elif isinstance(functions, type(lambda: None)):
                self.postprocess.append(ti.kernel(functions))
    
    def set_particle_calm(self, scene, calm_interval):
        if calm_interval:
            self.calm_interval = calm_interval
            self.postprocess.append(lambda: self.engine.dengine.calm(self.sims.current_step, self.calm_interval, scene))

    def save_file(self, mscene:MPMScene, dscene: DEMScene):
        print('# Step =', self.sims.current_step, '   ', 'Save Number =', self.sims.current_print, '   ', 'Simulation time =', self.sims.current_time, '\n')
        self.recorder.output(self.sims, self.msims, mscene, self.dsims, dscene)

    def compile(self):
        print("Compiling first ... ...")
        start_time = time.time()
        self.core()
        end_time = time.time()
        print(f'Compiling time = {end_time - start_time} \n')

    def CouplingSolver(self, mscene: MPMScene, dscene: DEMScene):
        print("#", " Start Simulation ".center(67,"="), "#")
        
        self.engine.pre_calculate()
        if self.sims.current_time < Threshold:
            self.save_file(mscene, dscene)
            self.sims.current_print += 1
            self.msims.current_print += 1
            self.dsims.current_print += 1
            self.last_save_time = -0.8 * self.sims.delta
            
        self.compile()
        start_time = time.time()
        while self.sims.current_time <= self.sims.time:
            self.core()

            new_body = self.generator.regenerate(self.sims, mscene, dscene)
            if self.sims.current_time - self.last_save_time + 0.1 * self.sims.delta> self.sims.save_interval or new_body:
                self.save_file(mscene, dscene)
                self.last_save_time = 1. * self.sims.current_time
                self.sims.current_print += 1
                self.msims.current_print += 1
                self.dsims.current_print += 1
                if new_body:
                    self.engine.enforce_update_verlet_table()
                    self.dsims.set_max_bounding_sphere_radius(dscene.find_bounding_sphere_max_radius(self.dsims))

            self.sims.current_time += self.sims.delta
            self.msims.current_time += self.sims.delta
            self.dsims.current_time += self.sims.delta
            self.sims.current_step += 1
        end_time = time.time()

        if abs(self.sims.current_time - self.last_save_time) > 0.99 * self.sims.save_interval:
            self.save_file(mscene, dscene)
            self.last_save_time = 1. * self.sims.current_time
            self.sims.current_print += 1
            self.msims.current_print += 1
            self.dsims.current_print += 1
            self.engine.reset_message()

        print('Physical time = ', end_time - start_time)
        print("#", " End Simulation ".center(67,"="), "#", '\n')

    def core(self):
        self.engine.reset_message()
        self.engine.compute()
        for functions in self.postprocess:
            functions()




