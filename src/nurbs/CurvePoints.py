import numpy as np
from src.nurbs.Operations import *


class CurvePoints(object):
    def __init__(self):
        self._points = None
        self._weights = 1.0
        self._smooth = 1e-6
        self._penalty = 1e-6

    def setup(self, points, weights=1.0, penalty=1e-6, smoothness=1e-6):
        self.points = points
        self.weights = weights
        self.smoothness = smoothness
        self.penalty = penalty

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = np.asarray(points)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def smoothness(self):
        return self._smoothness

    @smoothness.setter
    def smoothness(self, smoothness):
        self._smoothness = smoothness

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, penalty):
        self._penalty = penalty
