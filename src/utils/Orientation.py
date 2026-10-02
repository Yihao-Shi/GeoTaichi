import taichi as ti
import numpy as np

from src.utils.constants import PI
from src.utils.TypeDefination import vec3f


@ti.data_oriented
class set_orientation:
    def __init__(self, orientation) -> None:
        euler_angle = [0., 0., 0.]
        if orientation is None:
            self.particle_orientation = "constant"
        elif isinstance(orientation, str):
            self.particle_orientation = orientation
        elif isinstance(orientation, (list, tuple, np.ndarray)):
            self.particle_orientation = "constant"
            euler_angle = list(orientation)
        
        self.get_orientation = None
        self.set_orientation()
        if not "uniform" in self.particle_orientation:
            self.fix_orient = ti.Vector.field(3, float, shape=())
            self.record_orientation(vec3f(euler_angle))

    def set_orientation(self):
        if self.particle_orientation == 'constant':
            self.get_orientation = self.get_fixed_orientation
        elif self.particle_orientation == 'uniform':
            self.get_orientation = self.get_uniform_orientation
        elif self.particle_orientation == 'uniformXY':
            self.get_orientation = self.get_uniform_orientationXY
        elif self.particle_orientation == 'uniformYZ':
            self.get_orientation = self.get_uniform_orientationYZ
        elif self.particle_orientation == 'uniformXZ':
            self.get_orientation = self.get_uniform_orientationXZ
        else:
            raise ValueError("Orientation distribution error!")

    @ti.func
    def get_fixed_orientation(self):
        return self.fix_orient[None]

    @ti.func
    def get_uniform_orientation(self):
        return vec3f([2*PI*ti.random(float), 2*PI*ti.random(float), 2*PI*ti.random(float)])

    @ti.func
    def get_uniform_orientationXY(self):
        return vec3f([2*PI*ti.random(float), 2*PI*ti.random(float), 0.])

    @ti.func
    def get_uniform_orientationYZ(self):
        return vec3f([0., 2*PI*ti.random(float), 2*PI*ti.random(float)])

    @ti.func
    def get_uniform_orientationXZ(self):
        return vec3f([2*PI*ti.random(float), 0., 2*PI*ti.random(float)])

    @ti.kernel
    def record_orientation(self, orient: ti.types.vector(3, float)):
        self.fix_orient[None] = 2*PI*orient/360.