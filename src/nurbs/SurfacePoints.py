from src.utils.errors import SimulationStateError
import numpy as np
from src.nurbs.Operations import *


class SurfacePoints(object):
    def __init__(self):
        self._interior = None
        self._boundary = None
        self._common_index = None
        self._common_boundary = None
        self._rest_shape = None
        self._boundary_weight = 1.0
        self._interior_weight = 1.0
        self._corner_weight = 10.0
        self._boundary_smooth = 1e-6
        self._interior_smooth = 1e-6
        self._regularisation_res_u = 0
        self._regularisation_res_v = 0
        self._regularisation_u = 0.0
        self._regularisation_v = 0.0
        self._penalty = 1e-6

    def setup(
        self,
        interior=np.array([]),
        boundary=np.array([]),
        common_index=np.array([]),
        common_boundary=np.array([]),
        rest_shape=np.array([]),
        interior_weight=1.0,
        boundary_weight=1.0,
        corner_weight=10.0,
        boundary_smoothness=1e-6,
        interior_smoothness=1e-6,
        regularisation_res_u=0,
        regularisation_res_v=0,
        regularisation_u=0.0,
        regularisation_v=0.0,
        penalty=1e-6,
    ):
        self.interior = interior
        self.boundary = boundary
        self.common_index = common_index
        self.common_boundary = common_boundary
        self.rest_shape = rest_shape
        self.boundary_weight = boundary_weight
        self.interior_weight = interior_weight
        self.corner_weight = corner_weight
        self.boundary_smoothness = boundary_smoothness
        self.interior_smoothness = interior_smoothness
        self.regularisation_res_u = regularisation_res_u
        self.regularisation_res_v = regularisation_res_v
        self.regularisation_u = regularisation_u
        self.regularisation_v = regularisation_v
        self.penalty = penalty

    @property
    def interior(self):
        return self._interior

    @interior.setter
    def interior(self, interior):
        self._interior = np.asarray(interior)

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        self._boundary = np.asarray(boundary)

    @property
    def common_index(self):
        return self._common_index

    @common_index.setter
    def common_index(self, common_index):
        self._common_index = np.asarray(common_index)

    @property
    def common_boundary(self):
        return self._common_boundary

    @common_boundary.setter
    def common_boundary(self, common_boundary):
        if self.common_index is None:
            raise SimulationStateError("Set common_index first")
        self._common_boundary = np.asarray(common_boundary)
        if len(self._common_index) != len(self._common_boundary):
            raise ValueError(
                f"The dimension of common_index {len(self._common_index)} must be equal to common_boundary {len(self._common_boundary)}"
            )

    @property
    def rest_shape(self):
        return self._rest_shape

    @rest_shape.setter
    def rest_shape(self, rest_shape):
        self._rest_shape = rest_shape

    @property
    def boundary_weight(self):
        return self._boundary_weight

    @boundary_weight.setter
    def boundary_weight(self, boundary_weight):
        self._boundary_weight = boundary_weight

    @property
    def interior_weight(self):
        return self._interior_weight

    @interior_weight.setter
    def interior_weight(self, interior_weight):
        self._interior_weight = interior_weight

    @property
    def corner_weight(self):
        return self._corner_weight

    @corner_weight.setter
    def corner_weight(self, corner_weight):
        self._corner_weight = corner_weight

    @property
    def boundary_smoothness(self):
        return self._boundary_smoothness

    @boundary_smoothness.setter
    def boundary_smoothness(self, boundary_smoothness):
        self._boundary_smoothness = boundary_smoothness

    @property
    def interior_smoothness(self):
        return self._interior_smoothness

    @interior_smoothness.setter
    def interior_smoothness(self, interior_smoothness):
        self._interior_smoothness = interior_smoothness

    @property
    def regularisation_res_u(self):
        return self._regularisation_res_u

    @regularisation_res_u.setter
    def regularisation_res_u(self, regularisation_res_u):
        self._regularisation_res_u = regularisation_res_u

    @property
    def regularisation_res_v(self):
        return self._regularisation_res_v

    @regularisation_res_v.setter
    def regularisation_res_v(self, regularisation_res_v):
        self._regularisation_res_v = regularisation_res_v

    @property
    def regularisation_u(self):
        return self._regularisation_u

    @regularisation_u.setter
    def regularisation_u(self, regularisation_u):
        self._regularisation_u = regularisation_u

    @property
    def regularisation_v(self):
        return self._regularisation_v

    @regularisation_v.setter
    def regularisation_v(self, regularisation_v):
        self._regularisation_v = regularisation_v

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, penalty):
        self._penalty = penalty
