from src.utils.linalg import normalized
from src.utils.errors import SimulationStateError
from src.nurbs.Spline import Spline

import numpy as np
from src.nurbs.Nurbs import find_span
from src.nurbs.NurbsPlot import NurbsPlot
from src.nurbs.Utilities import check_knot_vector, make_list


class SplineCurve(Spline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._degree = None
        self._knot_vector = None
        self._multiplicity = None
        self._element = None
        self._control_points = None
        self._num_ctrlpts = 0
        self._num_element = 0
        self._num_knot = 0
        self._length = None
        self._parent_indices = None
        self.degree = kwargs.get("degree", None)
        self.knot_vector = kwargs.get("knot_vector", None)
        self.control_points = kwargs.get("control_point", None)

    @property
    def length(self):
        if self._length is None:
            raise SimulationStateError("Run estimate_length first!")
        return self._length

    @property
    def parent_indices(self):
        return self._parent_indices

    @parent_indices.setter
    def parent_indices(self, parent_indices):
        self._parent_indices = parent_indices

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree):
        if degree is None:
            self._degree = None
        else:
            if not isinstance(degree, (int, float)):
                raise TypeError("degree must be in int type")
            if degree < 1:
                raise ValueError("Degree must be greater than zero")
            self._degree = degree

    @property
    def knot_vector(self):
        return self._knot_vector

    @knot_vector.setter
    def knot_vector(self, kv):
        if kv is None:
            self._knot_vector = None
        else:
            kv = np.asarray(kv)
            if self._degree is None:
                raise ValueError("Curve degree must be set befor setting knot vector")

            self._knot_vector = self._check_knot_vector(kv)
            self._multiplicity = np.unique(self._knot_vector)
            self._element = np.column_stack((self._multiplicity[:-1], self._multiplicity[1:]))
            self._num_ctrlpts = self._knot_vector.shape[0] - self._degree - 1
            self._num_element = self._element.shape[0]
            self._num_knot = self._knot_vector.shape[0]
        self.topology_update = True

    @property
    def num_ctrlpts(self):
        return self._num_ctrlpts

    @property
    def num_knot(self):
        return self._num_knot

    @property
    def num_element(self):
        return self._num_element

    @property
    def multiplicity(self):
        return self._multiplicity

    @property
    def element(self):
        return self._element

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, ctrlpt_array):
        if ctrlpt_array is None:
            self._control_points = None
        else:
            ctrlpt_array = np.asarray(ctrlpt_array)
            if ctrlpt_array.ndim != 2:
                raise TypeError("Array Control points must have a dimension of 2.")

            if self._degree is None:
                raise ValueError("Curve degree must be set before setting control points")

            if self._knot_vector is None:
                raise ValueError("Curve knot vectors must be set before setting control points")

            if ctrlpt_array.shape[-1] != self.dimension:
                self.dimension = ctrlpt_array.shape[-1]

            if ctrlpt_array.shape[0] != self._num_ctrlpts:
                raise TypeError(f"Array Control points must have a shape of {[self._num_ctrlpts, self.dimension]}")

            self._control_points = ctrlpt_array

    def get_connectivity(self):
        pass

    def normal(self, knot):
        if self.dimension == 2:
            tangent = self.tangent(knot)
            return normalized(np.array([-tangent[1], tangent[0]]))
        elif self.dimension == 3:
            Su, Suu = self.second_derivative(knot, first_derivative=True)
            return normalized(Suu - np.dot(Suu, Su) * Su)

    def tangent(self, knot):
        return self.derivative(knot, normalize=True)

    def curvature(self, knot, get_center=False):
        Su, Suu = self.second_derivative(knot, first_derivative=True)
        den = pow(np.dot(Su, Su), 1.5)
        kappa = np.linalg.norm(np.cross(Su, Suu)) / den if den != 0 else np.inf

        if not get_center:
            return abs(kappa)
        else:
            p = self.single_point(knot)
            normal = self.normal(knot)
            return abs(kappa), p + kappa * normal

    def interpolation(self, knot):
        if isinstance(knot, float):
            point = self.single_point(knot)
            return point
        elif isinstance(knot, (list, tuple, np.ndarray)):
            knot_array = np.asarray(knot)
            if knot_array.ndim != 1.0:
                raise ValueError("Parameter array must be 1D")
            values = [self.single_point(parameter) for parameter in knot_array]
            return np.array(values)

    def degree_operation(self):
        pass

    def ctrlpt_id(self, knot):
        span = find_span(self.num_ctrlpts, knot, self.degree, self.knot_vector)
        return np.arange(span - self.degree, span + 1)

    def dxdknot(self, knot):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get partial derivative of [u, v, w]
        """
        return self.derivative(knot, normalize=False)

    def d2xd2knot(self, knot):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get the second partial derivative of [u, v, w]
        """
        return self.second_derivative(knot, normalize=False)

    def dxdctrlpts(self, knot):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get partial derivative of Pᵢ, return a numpy array [[C(u)_xᵢ=Nᵢ, 0, 0], [0, C(u)_yᵢ=Nᵢ, 0], [0, 0, C(u)_zᵢ=Nᵢ], ...]
        """
        coeff = self.coefficient(knot)
        identity = np.eye(self.dimension, dtype=coeff.dtype)
        result = np.vstack(
            [np.column_stack((coeff[i] * identity[i], identity[i] * coeff[i])) for i in range(self.dimension)]
        )
        return result

    def estimate_length(self, resolution=100):
        self._length = np.zeros(self.num_element)
        for index_u, knot_u in enumerate(self.element):
            knots = np.linspace(*knot_u, resolution)
            evalpts = self.interpolation(knots)
            p00 = evalpts[:-1, :-1]
            p10 = evalpts[1:, :-1]
            v4 = p10 - p00
            self._length[index_u] = np.linalg.norm(v4, axis=0)

    def convex_hull(self, start_knot, end_knot):
        if start_knot > end_knot:
            raise ValueError(f"Start knot {start_knot} must be less than or equal to end knot {end_knot}")
        start_span = find_span(self.num_ctrlpts, start_knot, self.degree, self.knot_vector)
        end_span = find_span(self.num_ctrlpts, end_knot, self.degree, self.knot_vector)
        start_support_knot_gap = start_knot - self.knot_vector[start_span]
        end_support_knot_gap = self.knot_vector[end_span + 1] - end_knot
        start_support_knot_gap = start_support_knot_gap if abs(start_support_knot_gap) > 1e-14 else 1e-5
        end_support_knot_gap = end_support_knot_gap if abs(end_support_knot_gap) > 1e-14 else 1e-5
        start_support_knot = max(0.0, start_knot - 0.01 * start_support_knot_gap)
        end_support_knot = min(1.0, end_knot + 0.01 * end_support_knot_gap)

        inserted_knot = []
        if not start_support_knot in self.knot_vector:
            inserted_knot.append(start_support_knot)
        if not start_knot in self.knot_vector:
            inserted_knot.append(start_knot)
        if not end_knot in self.knot_vector:
            inserted_knot.append(end_knot)
        if not end_support_knot in self.knot_vector:
            inserted_knot.append(end_support_knot)
        curve_copy = self.copy()
        curve_copy.insert_knot(inserted_knot)

        begin_element_id = np.where(curve_copy.element[:, 0] == start_knot)[0][0]
        end_element_id = np.where(curve_copy.element[:, 1] == end_knot)[0][0]
        return curve_copy.control_points[begin_element_id : end_element_id + curve_copy.degree + 1, :]

    def visualize(self, resolution=100, **kwargs):
        knots = np.linspace(0.0, 1.0, resolution)
        evalpts = self.interpolation(knots)
        plot = NurbsPlot(dimension=self.dimension)
        plot.append(evalpts=evalpts, control_points=self.control_points)
        if "other_curves" in kwargs:
            curves = make_list(kwargs["other_curves"])
            for curve in curves:
                evalpts = curve.interpolation(knots)
                plot.append(evalpts=evalpts, control_points=curve.control_points)
        plot.initialize(**kwargs)
        plot.plot_curve()

    def _check_knot_vector(self, knot_vector):
        return check_knot_vector(self._degree, knot_vector)

    def check_knot_vector(self, knot):
        knot_u = float(knot)
        if not self.validate_knot(knot_u):
            raise ValueError(f"Knot paramter {knot_u} must be in the interval [0, 1]")

    def _check_dimension(self, dimension):
        if dimension != 2 and dimension != 3:
            raise ValueError("Dimension must be 2 or 3")
