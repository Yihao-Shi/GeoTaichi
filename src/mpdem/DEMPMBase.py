import time

import taichi as ti

from src.mpdem.MixtureGenerate import MixtureGenerator
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


class Solver:
    sims: Simulation
    dsims: DEMSimulation
    msims: MPMSimulation
    drecorder: DEMWriteFile
    mrecorder: MPMWriteFile
    generator: MixtureGenerator
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

        self.last_save_time = 0.
        self.solve = None

    def save_file(self, mscene:MPMScene, dscene: DEMScene):
        print('# Step =', self.sims.current_step, '   ', 'Save Number =', self.sims.current_print, '   ', 'Simulation time =', self.sims.current_time, '\n')
        self.recorder.output(self.sims, self.msims, mscene, self.dsims, dscene)

    def CouplingSolver(self, mscene: MPMScene, dscene: DEMScene):
        print("#", " Start Simulation ".center(67,"="), "#")

        self.engine.pre_calculate()
        if self.sims.current_time < Threshold:
            self.save_file(mscene, dscene)
            self.sims.current_print += 1
            self.msims.current_print += 1
            self.dsims.current_print += 1
            self.last_save_time = -0.8 * self.sims.delta
            
        start_time = time.time()
        while self.sims.current_time <= self.sims.time:
            self.engine.reset_message()
            self.engine.compute()

            new_body = self.generator.regenerate(self.sims, mscene, dscene)
            if self.sims.current_time - self.last_save_time + 0.1 * self.sims.delta> self.sims.save_interval or new_body:
                self.save_file(mscene, dscene)
                self.last_save_time = 1. * self.sims.current_time
                self.sims.current_print += 1
                self.msims.current_print += 1
                self.dsims.current_print += 1
                if new_body:
                    self.engine.enforce_update_verlet_table()
                    self.dsims.max_particle_radius = dscene.find_particle_max_radius()

            self.sims.current_time += self.sims.delta
            self.msims.current_time += self.sims.delta
            self.dsims.current_time += self.sims.delta
            self.sims.current_step += 1
        end_time = time.time()

        if abs(self.sims.current_time - self.last_save_time) > self.sims.save_interval:
            self.save_file(mscene, dscene)
            self.last_save_time = 1. * self.sims.current_time
            self.sims.current_print += 1
            self.msims.current_print += 1
            self.dsims.current_print += 1
            self.engine.reset_message()

        print('Physical time = ', end_time - start_time)
        print("#", " End Simulation ".center(67,"="), "#", '\n')



