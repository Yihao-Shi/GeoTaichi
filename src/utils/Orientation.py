import taichi as ti
import numpy as np

from src.utils.TypeDefination import vec3f


@ti.data_oriented
class set_orientation:
    def __init__(self, orientation) -> None:
        direction = [0., 0., 1.]
        if orientation is None or orientation == "constant":
            self.particle_orientation = "constant"
        elif orientation == "uniform":
            self.particle_orientation = "uniform"
        elif isinstance(orientation, (list, tuple, np.ndarray)):
            self.particle_orientation = "constant"
            direction = list(orientation)
        
        self.get_orientation = None
        self.set_orientation()
        if not orientation == "uniform":
            self.fix_orient = ti.Vector.field(3, float, shape=())
            self.record_orientation(vec3f(direction))

    def set_orientation(self):
        if self.particle_orientation == 'constant':
            self.get_orientation = self.get_fixed_orientation
        elif self.particle_orientation == 'uniform':
            self.get_orientation = self.get_uniform_orientation
        else:
            raise ValueError("Orientation distribution error!")

    @ti.func
    def get_fixed_orientation(self):
        return self.fix_orient[None]

    @ti.func
    def get_uniform_orientation(self):
        return vec3f([360.*ti.random(float), 360.*ti.random(float), 360.*ti.random(float)])

    @ti.kernel
    def record_orientation(self, orient: ti.types.vector(3, float)):
        self.fix_orient[None] = orient