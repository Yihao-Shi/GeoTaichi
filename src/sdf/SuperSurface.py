from src.sdf.Surface import Surface
import numpy as np


class SuperSurface(Surface):
    def __init__(self, xrad1, yrad1, zrad1, xrad2=None, yrad2=None, zrad2=None) -> None:
        super().__init__()
        self.xrad1 = xrad1
        self.yrad1 = yrad1
        self.zrad1 = zrad1
        self.xrad2 = xrad2 or xrad1
        self.yrad2 = yrad2 or yrad1
        self.zrad2 = zrad2 or zrad1

    def _estimate_bounding_box(self):
        self._lower_bound = -np.array([self.xrad1, self.yrad1, self.zrad1])
        self._higher_bound = np.array([self.xrad2, self.yrad2, self.zrad2])
        self._is_bound = True
