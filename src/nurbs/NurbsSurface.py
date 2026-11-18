from src.utils.linalg import normalized
import numpy as np
from src.nurbs.SplineSurface import SplineSurface
from src.nurbs.NurbsBasis import *
from src.nurbs.Operations import *
from src.nurbs.Utilities import *


class NurbsSurface(SplineSurface):

    @staticmethod
    def nurbs_curve(degree, knot_vector, control_point, weight=None, closed=False, parent_indices=None):
        from src.nurbs.NurbsCurve import NurbsCurve

        if weight is None:
            weight = np.ones(control_point.shape[0])
        curve = NurbsCurve()
        curve.degree = degree
        curve.knot_vector = knot_vector
        curve.control_points = control_point
        curve.weights = weight
        curve.closed = closed
        curve.parent_indices = parent_indices
        return curve

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
                raise TypeError("Array Weights must have a dimension of 1.")

            if self._degree_u is None:
                raise ValueError("Surface degree u must be set before setting weights")

            if self._degree_v is None:
                raise ValueError("Surface degree v must be set before setting weights")

            if self._knot_vector_u is None:
                raise ValueError("Curve knot vector u must be set before setting weights")

            if self._knot_vector_v is None:
                raise ValueError("Curve knot vector v must be set before setting weights")

            if self._control_points is None:
                raise ValueError("Curve control points must be set before setting weights")

            if weight_array.shape[0] != self._num_ctrlpts_u * self._num_ctrlpts_v:
                raise TypeError(f"Array weights must have a shape of {[self._num_ctrlpts_u * self._num_ctrlpts_v]}")

            self._weights = weight_array

    def activate_boundary(self):
        if len(self._boundary) == 0:
            ctrlpts = self.control_points.reshape(self.num_ctrlpts_v, self.num_ctrlpts_u, -1)
            weights = self.weights.reshape(self.num_ctrlpts_v, self.num_ctrlpts_u)
            indices = np.arange(self.control_points.shape[0], dtype=np.int32).reshape(
                self.num_ctrlpts_v, self.num_ctrlpts_u
            )
            self._boundary.append(
                NurbsSurface.nurbs_curve(
                    degree=self.degree_u,
                    knot_vector=self.knot_vector_u,
                    control_point=ctrlpts[0, :],
                    weight=weights[0, :],
                    parent_indices=indices[0, :],
                )
            )
            self._boundary.append(
                NurbsSurface.nurbs_curve(
                    degree=self.degree_v,
                    knot_vector=self.knot_vector_v,
                    control_point=ctrlpts[:, -1],
                    weight=weights[:, -1],
                    parent_indices=indices[:, -1],
                )
            )
            self._boundary.append(
                NurbsSurface.nurbs_curve(
                    degree=self.degree_u,
                    knot_vector=self.knot_vector_u,
                    control_point=ctrlpts[-1, :],
                    weight=weights[-1, :],
                    parent_indices=indices[-1, :],
                )
            )
            self._boundary.append(
                NurbsSurface.nurbs_curve(
                    degree=self.degree_v,
                    knot_vector=self.knot_vector_v,
                    control_point=ctrlpts[:, 0],
                    weight=weights[:, 0],
                    parent_indices=indices[:, 0],
                )
            )

    def update_boundary(self):
        if len(self._boundary) > 0:
            ctrlpts = self.control_points.reshape(self.num_ctrlpts_v, self.num_ctrlpts_u, -1)
            weights = self.weights.reshape(self.num_ctrlpts_v, self.num_ctrlpts_u)
            self._boundary[0].knot_vector = self.knot_vector_u
            self._boundary[0].control_points = ctrlpts[0, :]
            self._boundary[0].weights = weights[0, :]
            self._boundary[1].knot_vector = self.knot_vector_v
            self._boundary[1].control_points = ctrlpts[:, -1]
            self._boundary[1].weights = weights[:, -1]
            self._boundary[2].knot_vector = self.knot_vector_u
            self._boundary[2].control_points = ctrlpts[-1, :]
            self._boundary[2].weights = weights[-1, :]
            self._boundary[3].knot_vector = self.knot_vector_v
            self._boundary[3].control_points = ctrlpts[:, 0]
            self._boundary[3].weights = weights[:, 0]

    def coefficient(self, knot_u, knot_v):
        knot_u, knot_v = float(knot_u), float(knot_v)
        self.check_knot_vector(knot_u, knot_v)
        return nurbs_basis2d(
            knot_u, knot_v, self.degree_u, self.degree_v, self.knot_vector_u, self.knot_vector_v, self.weights
        )

    def coefficient_derivate(self, knot_u, knot_v):
        knot_u, knot_v = float(knot_u), float(knot_v)
        self.check_knot_vector(knot_u, knot_v)
        return nurbs_basis_ders2d(
            knot_u, knot_v, self.degree_u, self.degree_v, self.knot_vector_u, self.knot_vector_v, self.weights
        )

    def coefficient_second_derivate(self, knot_u, knot_v):
        knot_u, knot_v = float(knot_u), float(knot_v)
        self.check_knot_vector(knot_u, knot_v)
        return nurbs_basis2nd_ders2d(
            knot_u, knot_v, self.degree_u, self.degree_v, self.knot_vector_u, self.knot_vector_v, self.weights
        )

    def single_point(self, knot_u, knot_v):
        knot_u, knot_v = float(knot_u), float(knot_v)
        self.check_knot_vector(knot_u, knot_v)
        return nurbs_basis_interpolations2d(
            knot_u,
            knot_v,
            self.degree_u,
            self.degree_v,
            self.knot_vector_u,
            self.knot_vector_v,
            self.control_points,
            self.weights,
        )

    def derivative(self, knot_u, knot_v, normalize=False):
        knot_u, knot_v = float(knot_u), float(knot_v)
        self.check_knot_vector(knot_u, knot_v)
        _, derivatives = nurbs_basis_interpolations_ders2d(
            knot_u,
            knot_v,
            self.degree_u,
            self.degree_v,
            self.knot_vector_u,
            self.knot_vector_v,
            self.control_points,
            self.weights,
        )
        if normalize:
            return normalized(derivatives[0]), normalized(derivatives[1])
        else:
            return derivatives[0], derivatives[1]

    def second_derivative(self, knot_u, knot_v, first_derivative=False, normalize=False):
        self.check_knot_vector(knot_u, knot_v)
        _, derivatives, sceond_derivatives = nurbs_basis_interpolations2nd_ders2d(
            knot_u,
            knot_v,
            self.degree_u,
            self.degree_v,
            self.knot_vector_u,
            self.knot_vector_v,
            self.control_points,
            self.weights,
        )
        if normalize:
            get_value = tuple(
                (
                    normalized(sceond_derivatives[0]),
                    normalized(sceond_derivatives[1]),
                    normalized(sceond_derivatives[2]),
                )
            )
            if first_derivative:
                get_value = tuple((normalized(derivatives[0]), normalized(derivatives[1]))) + get_value
            return get_value
        else:
            get_value = tuple((sceond_derivatives[0], sceond_derivatives[1], sceond_derivatives[2]))
            if first_derivative:
                get_value = tuple((derivatives[0], derivatives[1])) + get_value
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
                surface_inversion(
                    point,
                    self.degree_u,
                    self.degree_v,
                    self.knot_vector_u,
                    self.knot_vector_v,
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
                surface_inversion(
                    point,
                    self.degree_u,
                    self.degree_v,
                    self.knot_vector_u,
                    self.knot_vector_v,
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
                surface_inversion(
                    point,
                    self.degree_u,
                    self.degree_v,
                    self.knot_vector_u,
                    self.knot_vector_v,
                    self.control_points,
                    self.weights,
                    get_distance=False,
                    closed=closed,
                )
                for point in points
            ]
        )
        return np.array([self.single_point(*knot) for knot in knots])

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
                surface_inversion(
                    point,
                    self.degree_u,
                    self.degree_v,
                    self.knot_vector_u,
                    self.knot_vector_v,
                    self.control_points,
                    self.weights,
                    closed=closed,
                )
                for point in points
            ]
        )
        return new_matrix[:, 0:2], new_matrix[:, 2]

    def insert_knot(self, knot_list_u=list(), knot_list_v=list()):
        knot_list_u = np.asarray(make_list(knot_list_u))
        knot_list_v = np.asarray(make_list(knot_list_v))

        wpoints = assemble_point_weight(self.control_points, self.weights).reshape(self.num_ctrlpts_v, -1)
        new_knot_v, new_wpoint = knot_insertion(self.degree_v, self.knot_vector_v, wpoints, addition_knot=knot_list_v)
        self.knot_vector_v = new_knot_v
        point = np.swapaxes(new_wpoint.reshape(self.num_ctrlpts_v, self.num_ctrlpts_u, -1), 1, 0).reshape(
            self.num_ctrlpts_u, -1
        )

        new_knot_u, new_wpoint = knot_insertion(self.degree_u, self.knot_vector_u, point, addition_knot=knot_list_u)
        self.knot_vector_u = new_knot_u
        point = np.swapaxes(new_wpoint.reshape(self.num_ctrlpts_u, self.num_ctrlpts_v, -1), 1, 0).reshape(
            self.num_ctrlpts_v, -1
        )

        new_ctrlpts, new_weights = split_point_weight(new_wpoint.reshape(self.num_ctrlpts_u * self.num_ctrlpts_v, -1))
        self.update(new_knot_u, new_knot_v, new_ctrlpts, new_weights)

    def refine_knot(self, density=[1, 1], knot_list_u=list(), knot_list_v=list()):
        if len(list(density)) != 2:
            raise ValueError("The dimension of density must be 2")
        knot_list_u = np.asarray(make_list(knot_list_u))
        knot_list_v = np.asarray(make_list(knot_list_v))

        wpoints = assemble_point_weight(self.control_points, self.weights).reshape(self.num_ctrlpts_v, -1)
        new_knot_v, new_wpoint = knot_refinement(
            self.degree_v, self.knot_vector_v, wpoints, density=density[1], addition_knot=knot_list_v
        )
        self.knot_vector_v = new_knot_v
        point = np.swapaxes(new_wpoint.reshape(self.num_ctrlpts_v, self.num_ctrlpts_u, -1), 1, 0).reshape(
            self.num_ctrlpts_u, -1
        )

        new_knot_u, new_wpoint = knot_refinement(
            self.degree_u, self.knot_vector_u, point, density=density[0], addition_knot=knot_list_u
        )
        self.knot_vector_u = new_knot_u
        point = np.swapaxes(new_wpoint.reshape(self.num_ctrlpts_u, self.num_ctrlpts_v, -1), 1, 0).reshape(
            self.num_ctrlpts_v, -1
        )

        new_ctrlpts, new_weights = split_point_weight(new_wpoint.reshape(self.num_ctrlpts_u * self.num_ctrlpts_v, -1))
        self.update(new_knot_u, new_knot_v, new_ctrlpts, new_weights)

    def update(self, knot_u, knot_v, control_points, weights):
        self.knot_vector_u = knot_u
        self.knot_vector_v = knot_v
        self.control_points = control_points
        self.weights = weights
        self.get_connectivity()
        self.update_boundary()

    def dxdweight(self, knot_u, knot_v):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get partial derivative of wᵢ
        """
        return nurbs_dx_dweight2d(
            knot_u, knot_v, self.degree_u, self.degree_v, self.knot_vector_u, self.knot_vector_v, self.weights
        )

    @classmethod
    def from_dict(cls, data: dict):
        needed_keys = {"degree", "knot_vector_u", "knot_vector_v", "control_points", "weights"}
        missing_keys = needed_keys ^ data.keys()
        if missing_keys:
            raise ValueError(f"Missing the following information needed to instaniate curve {missing_keys}")
        degree = data.get("degree", 2)
        if not isinstance(degree, (int, float)):
            raise TypeError("degree must be in int type")
        knot_vector_u = np.array(data("knot_vector_u"))
        knot_vector_v = np.array(data("knot_vector_v"))
        control_points = np.array(data("control_points"))
        weights = np.array(data("weights"))

        dimension = control_points.shape[-1]
        cls(degree=degree, dimension=dimension)
        cls.knot_vector_u = knot_vector_u
        cls.knot_vector_v = knot_vector_v
        cls.control_points = control_points
        cls.weights = weights
        return cls

    def to_dict(self):
        return {
            "degree": self.degree,
            "knot_vector_u": self.knot_vector_u.to_list(),
            "knot_vector_v": self.knot_vector_v.to_list(),
            "control_points": self.control_points.to_list(),
            "weights": self.weights.to_list(),
        }

    def write(self, path="output.txt"):
        with open(path, "w") as file:
            file.write("degree_u " + str(self.degree_u) + "\n")
            file.write("degree_v " + str(self.degree_v) + "\n")
            file.write("\n")
            file.write("knot_vector_u " + " ".join(str(x) for x in self.knot_vector_u) + "\n")
            file.write("\n")
            file.write("knot_vector_v " + " ".join(str(x) for x in self.knot_vector_v) + "\n")
            file.write("\n")
            file.write("control_points " + str(self.num_ctrlpts_u) + " " + str(self.num_ctrlpts_v) + "\n")
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
                    if line_info.startswith("degree_u"):
                        parts = line_info.split()
                        self.degree_u = int(parts[1])
                    if line_info.startswith("degree_v"):
                        parts = line_info.split()
                        self.degree_v = int(parts[1])
                    if line_info.startswith("knot_vector_u "):
                        parts = line_info.split()
                        self.knot_vector_u = np.array([float(parts[knot_id]) for knot_id in range(1, len(parts))])
                    if line_info.startswith("knot_vector_v "):
                        parts = line_info.split()
                        self.knot_vector_v = np.array([float(parts[knot_id]) for knot_id in range(1, len(parts))])
                    if line_info.startswith("control_points "):
                        parts = line_info.split()
                        num_ctrlpts = [int(parts[1]), int(parts[2])]
                        assert self.num_ctrlpts_u == num_ctrlpts[0] and self.num_ctrlpts_v == num_ctrlpts[1]
                    if line_info.startswith("c "):
                        parts = line_info.split()
                        if num_ctrlpts is not None:
                            ctrlpts.append([float(parts[knot_id]) for knot_id in range(1, len(parts))])
                            weights.append(float(parts[-1]))
        self.control_points = np.asarray(ctrlpts)
        self.weights = weights
        self.activate_boundary()
