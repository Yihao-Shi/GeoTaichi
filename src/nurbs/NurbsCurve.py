from src.utils.linalg import normalized
import numpy as np
from src.nurbs.SplineCurve import SplineCurve
from src.nurbs.NurbsBasis import *
from src.nurbs.Operations import *
from src.nurbs.Utilities import *


class NurbsCurve(SplineCurve):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = kwargs.get("weight", None)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weight_array):
        if weight_array is None:
            self._weights = None
        else:
            weight_array = np.asarray(weight_array)
            if weight_array.ndim != 1:
                raise TypeError("Array Weights must be 1D.")

            if self._degree is None:
                raise ValueError("Curve degree must be set before setting weights")

            if self._knot_vector is None:
                raise ValueError("Curve knot vectors must be set before setting weights")

            if self._control_points is None:
                raise ValueError("Curve control points must be set before setting weights")

            if weight_array.shape[0] != self._num_ctrlpts:
                raise TypeError(f"Array weights must have a shape of {[self._num_ctrlpts]}")

            self._weights = weight_array

    def coefficient(self, knot):
        knot = float(knot)
        self.check_knot_vector(knot)
        return nurbs_basis1d(knot, self.degree, self.knot_vector, self.weights)

    def coefficient_derivate(self, knot):
        knot = float(knot)
        self.check_knot_vector(knot)
        return nurbs_basis_ders1d(knot, self.degree, self.knot_vector, self.weights)

    def coefficient_second_derivate(self, knot):
        knot = float(knot)
        self.check_knot_vector(knot)
        return nurbs_basis2nd_ders1d(knot, self.degree, self.knot_vector, self.weights)

    def single_point(self, knot):
        knot = float(knot)
        self.check_knot_vector(knot)
        return nurbs_basis_interpolations1d(knot, self.degree, self.knot_vector, self.control_points, self.weights)

    def derivative(self, knot, normalize=False):
        knot = float(knot)
        self.check_knot_vector(knot)
        _, derivatives = nurbs_basis_interpolations_ders1d(
            knot, self.degree, self.knot_vector, self.control_points, self.weights
        )
        if normalize:
            return normalized(derivatives)
        else:
            return derivatives

    def second_derivative(self, knot, first_derivative=False, normalize=False):
        self.check_knot_vector(knot)
        _, derivatives, second_derivatives = nurbs_basis_interpolations2nd_ders1d(
            knot, self.degree, self.knot_vector, self.control_points, self.weights
        )
        if normalize:
            get_value = tuple(normalized(second_derivatives))
            if first_derivative:
                get_value = tuple(normalized(derivatives)) + get_value
            return get_value
        else:
            get_value = tuple(second_derivatives)
            if first_derivative:
                get_value = tuple(derivatives) + get_value
            return get_value

    def distance(self, points, closed=None):
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != self.dimension:
            raise ValueError(f"Point must be in R$^{self.dimension}$")
        if closed is None:
            closed = self.closed
        return np.array(
            [
                curve_inversion(
                    point,
                    self.degree,
                    self.knot_vector,
                    self.control_points,
                    self.weights,
                    get_knot=False,
                    closed=closed,
                )
                for point in points
            ]
        )

    def paramaterize(self, points, closed=None):
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != self.dimension:
            raise ValueError(f"Point must be in R$^{self.dimension}$")
        if closed is None:
            closed = self.closed
        return np.array(
            [
                curve_inversion(
                    point,
                    self.degree,
                    self.knot_vector,
                    self.control_points,
                    self.weights,
                    get_distance=False,
                    closed=closed,
                )
                for point in points
            ]
        )

    def projection(self, points, closed=None):
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != self.dimension:
            raise ValueError(f"Point must be in R$^{self.dimension}$")
        if closed is None:
            closed = self.closed
        knots = np.array(
            [
                curve_inversion(
                    point,
                    self.degree,
                    self.knot_vector,
                    self.control_points,
                    self.weights,
                    get_distance=False,
                    closed=closed,
                )
                for point in points
            ]
        )
        return np.array([self.single_point(knot) for knot in knots])

    def inversion(self, points, closed=None):
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != self.dimension:
            raise ValueError(f"Point must be in R$^{self.dimension}$")
        if closed is None:
            closed = self.closed
        new_matrix = np.array(
            [
                curve_inversion(point, self.degree, self.knot_vector, self.control_points, self.weights, closed=closed)
                for point in points
            ]
        )
        return new_matrix[:, :1], new_matrix[:, 1:]

    def insert_knot(self, knot_list=list()):
        knot_list = np.asarray(make_list(knot_list))
        wpoints = assemble_point_weight(self.control_points, self.weights)
        new_knot_vector, new_wpoints = knot_insertion(self.degree, self.knot_vector, wpoints, knot_list)
        new_control_points, new_weights = split_point_weight(new_wpoints)
        self.update(new_knot_vector, new_control_points, new_weights)

    def refine_knot(self, density=1, knot_list=list()):
        knot_list = np.asarray(make_list(knot_list))
        wpoints = assemble_point_weight(self.control_points, self.weights)
        new_knot_vector, new_wpoints = knot_refinement(self.degree, self.knot_vector, wpoints, density, knot_list)
        new_control_points, new_weights = split_point_weight(new_wpoints)
        self.update(new_knot_vector, new_control_points, new_weights)

    def update(self, knot_vector, control_points, weights):
        self.knot_vector = knot_vector
        self.control_points = control_points
        self.weights = weights
        self.get_connectivity()

    def dxdweight(self, knot):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get partial derivative of wᵢ
        """
        return nurbs_dx_dweight1d(knot, self.degree, self.knot_vector, self.weights)

    @classmethod
    def from_dict(cls, data: dict):
        needed_keys = {"degree", "knot_vector", "control_points", "weights"}
        missing_keys = needed_keys ^ data.keys()
        if missing_keys:
            raise ValueError(f"Missing the following information needed to instaniate curve {missing_keys}")
        degree = data.get("degree", 2)
        if not isinstance(degree, (int, float)):
            raise TypeError("degree must be in int type")
        knot_vector = np.array(data("knot_vector"))
        control_points = np.array(data("control_points"))
        weights = np.array(data("weights"))

        dimension = control_points.shape[-1]
        cls(degree=degree, dimension=dimension)
        cls.knot_vector = knot_vector
        cls.control_points = control_points
        cls.weights = weights
        return cls

    def to_dict(self):
        return {
            "degree": self.degree,
            "knot_vector": self.knot_vector.to_list(),
            "control_points": self.control_points.to_list(),
            "weights": self.weights.to_list(),
        }

    def write(self, path="output.txt"):
        with open(path, "w") as file:
            file.write("degree " + str(self.degree) + "\n")
            file.write("\n")
            file.write("knot_vector " + " ".join(str(x) for x in self.knot_vector) + "\n")
            file.write("\n")
            file.write("control_points " + str(self.num_ctrlpts) + "\n")
            for ctrlpts, weights in zip(self.control_points, self.weights):
                file.write("c " + " ".join(str(x) for x in list(ctrlpts).append(weights)) + "\n")

    def read(self, path="patch.txt"):
        num_ctrlpts = None
        ctrlpts = []
        weights = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line_info = line.strip()
                if len(line_info) > 0:
                    if line_info.startswith("degree"):
                        parts = line_info.split()
                        self.degree = int(parts[1])
                    if line_info.startswith("knot_vector "):
                        parts = line_info.split()
                        self.knot_vector = np.array([float(parts[knot_id]) for knot_id in range(1, len(parts))])
                    if line_info.startswith("control_points "):
                        parts = line_info.split()
                        num_ctrlpts = [int(parts[1])]
                        assert self.num_ctrlpts == num_ctrlpts[0]
                    if line_info.startswith("c "):
                        parts = line_info.split()
                        if num_ctrlpts is not None:
                            ctrlpts.append([float(parts[knot_id]) for knot_id in range(1, len(parts))])
                            weights.append(float(parts[-1]))
        self.control_points = np.asarray(ctrlpts)
        self.weights = np.asarray(weights)
