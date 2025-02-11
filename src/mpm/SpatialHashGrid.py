import taichi as ti
from functools import reduce

from src.mpm.BaseKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.constants import Threshold
from src.utils.sorting.BinSort import BinSort
from src.utils.linalg import no_operation


class SpatialHashGrid(object):
    sims: Simulation

    def __init__(self, sims):
        self.sims = sims
        self.cnum = None
        self.cellSum = 0
        self.grid_size = None
        self.igrid_size = None
        
    def resize_neighbor(self, scene):
        del self.igrid_size, self.cnum, self.cellSum
        self.neighbor_initialze(scene)

    def neighbor_initialze(self, scene: myScene, grid_size=None):
        self.place_particles = no_operation
        if self.sims.neighbor_detection: # or self.sims.solver_type == "G2P2G" or self.sims.mode == "Lightweight":
            if grid_size is None:
                rad_max, rad_min = scene.find_bounding_sphere_radius()
                self.sims.set_verlet_distance(rad_min)
                self.grid_size = 2 * rad_max + self.sims.verlet_distance
            else:
                self.grid_size = grid_size
                
            if self.grid_size < Threshold:
                raise RuntimeError("Particle radius is equal to zero!")
            self.igrid_size = 1. / self.grid_size

            if self.sims.pbc:
                pass

            self.cnum = ti.Vector([max(1, int(domain * self.igrid_size) + 1) for domain in self.sims.domain])
            self.cellSum = reduce((lambda x, y: int(max(1, x) * max(1, y))), list(self.cnum))
            self.sorted = BinSort(self.grid_size, self.cnum, self.sims.max_particle_num)
            self.place_particles = self.place_particle

    def pre_neighbor(self, scene: myScene):
        self.place_particle(scene)

    def place_particle(self, scene: myScene):
        self.sorted.run(int(scene.particleNum[0]), scene.particle.x)
