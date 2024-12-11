import taichi as ti
from functools import reduce

from src.mpm.BaseKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.constants import Threshold
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.TypeDefination import vec3i, vec3f


class SpatialHashGrid(object):
    sims: Simulation

    def __init__(self, sims):
        self.sims = sims
        self.cnum = None
        self.cellSum = 0
        self.grid_size = None
        self.igrid_size = None

        self.particle_pse = None
        self.cell_pse = None
        self.place_particles = None
        self.current = None
        self.count = None
        self.ParticleID = None
        
    def resize_neighbor(self, scene):
        del self.cell_pse, self.igrid_size, self.cnum, self.cellSum
        del self.current, self.count
        self.neighbor_initialze(scene)

    def neighbor_initialze(self, scene: myScene):
        self.place_particles = self.no_operation
        if self.sims.neighbor_detection:
            rad_max, rad_min = scene.find_bounding_sphere_radius()
            self.sims.set_verlet_distance(rad_min)
            self.grid_size = 2 * rad_max + self.sims.verlet_distance
            if self.grid_size < 1e-3 * Threshold:
                raise RuntimeError("Particle radius is equal to zero!")
            self.igrid_size = 1. / self.grid_size

            if self.sims.pbc:
                pass

            self.cnum = ti.Vector([max(1, int(domain * self.igrid_size) + 1) for domain in self.sims.domain])
            self.cellSum = reduce((lambda x, y: int(x * y)), list(self.cnum))
            self.cell_pse = PrefixSumExecutor(self.cellSum)
            self.set_hash_table()
            self.place_particles = self.place_particle_to_cell

    def set_hash_table(self):
        self.current = ti.field(int, shape=self.cellSum)
        self.count = ti.field(int, shape=self.cell_pse.get_length())
        self.ParticleID = ti.field(int, shape=self.sims.max_particle_num)

    def pre_neighbor(self, scene: myScene):
        self.place_particles(scene)

    def no_operation(self, scene):
        pass

    def place_particle_to_cell_v2(self, scene: myScene):
        calculate_particles_position_v2_(int(scene.particleNum[0]), self.igrid_size, scene.particle, self.count, self.cnum)
        self.cell_pse.run(self.count)
        insert_particle_to_cell_v2_(self.igrid_size, int(scene.particleNum[0]), scene.particle, self.count, self.current, self.ParticleID, self.cnum)

    def place_particle_to_cell(self, scene: myScene):
        calculate_particles_position_(int(scene.particleNum[0]), self.igrid_size, scene.particle, self.count, self.cnum)
        self.cell_pse.run(self.count)
        insert_particle_to_cell_(self.igrid_size, int(scene.particleNum[0]), scene.particle, self.count, self.current, self.ParticleID, self.cnum)
