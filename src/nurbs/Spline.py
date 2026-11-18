from copy import deepcopy


class Spline(object):
    def __init__(self, **kwargs):
        self._dimension = kwargs.get("dimension", 3)
        self._closed = kwargs.get("closed", False)
        self._control_points = None
        self.topology_update = False

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, dimension):
        self._check_dimension(dimension)
        self._dimension = dimension

    @property
    def closed(self):
        return self._closed

    @closed.setter
    def closed(self, closed=False):
        self._closed = closed

    @property
    def control_points(self):
        return self._control_points

    def copy(self):
        return deepcopy(self)

    def validate_knot(self, knot):
        return 0.0 <= knot <= 1.0

    def _check_dimension(self, dim):
        raise NotImplementedError

    def ctrlpt_id(self, *args):
        raise NotImplementedError

    def coefficient(self, *args):
        raise NotImplementedError

    def coefficient_derivate(self, *args):
        raise NotImplementedError

    def coefficient_second_derivate(self, *args):
        raise NotImplementedError

    def single_point(self, *args):
        raise NotImplementedError

    def derivative(self, *args):
        raise NotImplementedError

    def second_derivative(self, *args):
        raise NotImplementedError

    def normal(self, *args):
        raise NotImplementedError

    def tangent(self, *args):
        raise NotImplementedError

    def curvature(self, *args):
        raise NotImplementedError

    def distance(self, *args):
        raise NotImplementedError

    def paramaterize(self, *args):
        raise NotImplementedError

    def inversion(self, *args):
        raise NotImplementedError

    def projection(self, *args):
        raise NotImplementedError

    def refine_knot(self, *args):
        raise NotImplementedError

    def interpolation(self, *args):
        raise NotImplementedError

    def insert_knot(self, *args):
        raise NotImplementedError

    def dxdknot(self, *args):
        raise NotImplementedError

    def d2xd2knot(self, *args):
        raise NotImplementedError

    def dxdctrlpts(self, *args):
        raise NotImplementedError

    def get_connectivity(self, *args):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError

    def convex_hull(self, *args):
        raise NotImplementedError

    def write(self, *args):
        raise NotImplementedError

    def read(self, *args):
        raise NotImplementedError
