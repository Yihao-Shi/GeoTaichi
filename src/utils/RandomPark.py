import taichi as ti
import math

from src.utils.ObjectIO import DictIO

class RandomPark(object):
    def __init__(self) -> None:
        self.paramters = ti.field(float, shape=3)

    def set_seed(self, random):
        random_type = DictIO(random, "RandomType", "Constant")
        if random_type == 'constant':
            self.get_orientation = self.get_fixed_orientation
        elif random_type == 'uniform':
            self.get_orientation = self.get_uniform_orientation
        elif random_type == 'gaussian': 
            self.get_orientation = self.get_uniform_orientation
        else:
            raise ValueError("Orientation distribution error!")

