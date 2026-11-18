from src.nurbs.Spline import Spline

import numpy as np
from itertools import product
from src.nurbs.Nurbs import find_span
from src.nurbs.NurbsPlot import NurbsPlot
from src.nurbs.Utilities import check_knot_vector


class SplineVolume(Spline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._degree = None
        self._knot_vector_u = None
        self._knot_vector_v = None
        self._knot_vector_w = None
        self._multiplicity_u = None
        self._multiplicity_v = None
        self._multiplicity_w = None
        self._element_u = None
        self._element_v = None
        self._element_w = None
        self._control_points = None
        self._num_ctrlpts_u = 0
        self._num_ctrlpts_v = 0
        self._num_ctrlpts_w = 0
        self._num_element_u = 0
        self._num_element_v = 0
        self._num_element_w = 0
        self._num_knot_u = 0
        self._num_knot_v = 0
        self._num_knot_w = 0
        self._boundary = []
        self.degree = kwargs.get("degree", None)
        self.knot_vector_u = kwargs.get("knot_vector_u", None)
        self.knot_vector_v = kwargs.get("knot_vector_v", None)
        self.knot_vector_w = kwargs.get("knot_vector_w", None)
        self.control_points = kwargs.get("control_point", None)

    @property
    def boundary(self):
        return self._boundary

    @property
    def degree(self):
        return self._degree_u, self._degree_v, self._degree_w

    @degree.setter
    def degree(self, degree):
        if degree is None:
            self._degree_u = None
            self._degree_v = None
            self._degree_w = None
        else:
            if isinstance(degree, (int, float)):
                degree = [degree, degree, degree]
            elif isinstance(degree, (tuple, list, np.ndarray)):
                if list(len(degree)) != 3:
                    raise TypeError("degree should has a length of 3.")
            self._degree_u = degree[0]
            self._degree_v = degree[1]
            self._degree_w = degree[2]

    @property
    def degree_u(self):
        return self._degree_u

    @degree_u.setter
    def degree_u(self, degree):
        if degree < 1:
            raise ValueError("Degree u must be greater than zero")
        self._degree_u = degree

    @property
    def degree_v(self):
        return self._degree_v

    @degree_v.setter
    def degree_v(self, degree):
        if degree < 1:
            raise ValueError("Degree v must be greater than zero")
        self._degree_v = degree

    @property
    def degree_w(self):
        return self._degree_w

    @degree_w.setter
    def degree_w(self, degree):
        if degree < 1:
            raise ValueError("Degree w must be greater than zero")
        self._degree_w = degree

    @property
    def num_ctrlpts_u(self):
        return self._num_ctrlpts_u

    @property
    def num_ctrlpts_v(self):
        return self._num_ctrlpts_v

    @property
    def num_ctrlpts_w(self):
        return self._num_ctrlpts_w

    @property
    def num_ctrlpts(self):
        return self._num_ctrlpts_u * self._num_ctrlpts_v * self._num_ctrlpts_w

    @property
    def num_ctrlpts_list(self):
        return np.array([self._num_ctrlpts_u, self._num_ctrlpts_v, self._num_ctrlpts_w])

    @property
    def num_element_u(self):
        return self._num_element_u

    @property
    def num_element_v(self):
        return self._num_element_v

    @property
    def num_element_w(self):
        return self._num_element_w

    @property
    def num_element(self):
        return self._num_element_u * self._num_element_v * self._num_element_w

    @property
    def num_element_list(self):
        return np.array([self._num_element_u, self._num_element_v, self._num_element_w])

    @property
    def num_knot_u(self):
        return self._num_knot_u

    @property
    def num_knot_v(self):
        return self._num_knot_v

    @property
    def num_knot_w(self):
        return self._num_knot_w

    @property
    def num_knot(self):
        return self._num_knot_u * self._num_knot_v * self._num_knot_w

    @property
    def num_knot_list(self):
        return np.array([self._num_knot_u, self._num_knot_v, self._num_knot_w])

    @property
    def element_u(self):
        return self._element_u

    @property
    def element_v(self):
        return self._element_v

    @property
    def element_w(self):
        return self._element_w

    @property
    def multiplicity_u(self):
        return self._multiplicity_u

    @property
    def multiplicity_v(self):
        return self._multiplicity_v

    @property
    def multiplicity_w(self):
        return self._multiplicity_w

    @property
    def knot_vector_u(self):
        return self._knot_vector_u

    @knot_vector_u.setter
    def knot_vector_u(self, kv):
        if kv is None:
            self._knot_vector_u = None
        else:
            if self._degree_u is None:
                raise ValueError("Surface degree in u direction must be set before setting knot vector")

            self._knot_vector_u = self._check_knot_vector(kv, direction="u")
            self._multiplicity_u = np.unique(self._knot_vector_u)
            self._element_u = np.column_stack((self._multiplicity_u[:-1], self._multiplicity_u[1:]))
            self._num_ctrlpts_u = self._knot_vector_u.shape[0] - self._degree_u - 1
            self._num_element_u = self._element_u.shape[0]
            self._num_knot_u = self._knot_vector_u.shape[0]
        self.topology_update = True

    @property
    def knot_vector_v(self):
        return self._knot_vector_v

    @knot_vector_v.setter
    def knot_vector_v(self, kv):
        if kv is None:
            self._knot_vector_v = None
        else:
            if self._degree_v is None:
                raise ValueError("Surface degree in v direction must be set before setting knot vector")

            self._knot_vector_v = self._check_knot_vector(kv, direction="v")
            self._multiplicity_v = np.unique(self._knot_vector_v)
            self._element_v = np.column_stack((self._multiplicity_v[:-1], self._multiplicity_v[1:]))
            self._num_ctrlpts_v = self._knot_vector_v.shape[0] - self._degree_v - 1
            self._num_element_v = self._element_v.shape[0]
            self._num_knot_v = self._knot_vector_v.shape[0]
        self.topology_update = True

    @property
    def knot_vector_w(self):
        return self._knot_vector_w

    @knot_vector_w.setter
    def knot_vector_w(self, kv):
        if kv is None:
            self._knot_vector_w = None
        else:
            if self._degree_w is None:
                raise ValueError("Surface degree in w direction must be set before setting knot vector")

            self._knot_vector_w = self._check_knot_vector(kv, direction="w")
            self._multiplicity_w = np.unique(self._knot_vector_w)
            self._element_w = np.column_stack((self._multiplicity_w[:-1], self._multiplicity_w[1:]))
            self._num_ctrlpts_w = self._knot_vector_w.shape[0] - self._degree_w - 1
            self._num_element_w = self._element_w.shape[0]
            self._num_knot_w = self._knot_vector_w.shape[0]
        self.topology_update = True

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

            if self._degree_u is None:
                raise ValueError("Surface degree u must be set before setting control points")

            if self._degree_v is None:
                raise ValueError("Surface degree v must be set before setting control points")

            if self._degree_w is None:
                raise ValueError("Surface degree w must be set before setting control points")

            if self._knot_vector_u is None:
                raise ValueError("Curve knot vector u must be set before setting control points")

            if self._knot_vector_v is None:
                raise ValueError("Curve knot vector v must be set before setting control points")

            if self._knot_vector_w is None:
                raise ValueError("Curve knot vector w must be set before setting control points")

            if ctrlpt_array.shape[-1] != self.dimension:
                self.dimension = ctrlpt_array.shape[-1]

            if ctrlpt_array.shape[0] != self._num_ctrlpts_u * self._num_ctrlpts_v * self._num_ctrlpts_w:
                raise TypeError(
                    f"Array Control points must have a shape of {[self._num_ctrlpts_u * self._num_ctrlpts_v * self._num_ctrlpts_w, self.dimension]}"
                )

            self._control_points = ctrlpt_array

    def ctrlpt_id(self, knot_u, knot_v, knot_w):
        span_u = find_span(self.num_ctrlpts_u, knot_u, self.degree_u, self.knot_vector_u)
        span_v = find_span(self.num_ctrlpts_v, knot_v, self.degree_v, self.knot_vector_v)
        span_w = find_span(self.num_ctrlpts_w, knot_w, self.degree_w, self.knot_vector_w)
        control_point_id = np.array(
            list(
                product(
                    np.arange(span_w - self.degree_w, span_w + 1),
                    np.arange(span_v - self.degree_v, span_v + 1),
                    np.arange(span_u - self.degree_u, span_u + 1),
                )
            )
        )
        return (
            control_point_id[:, 2]
            + control_point_id[:, 1] * self.num_ctrlpts_u
            + control_point_id[:, 0] * self.num_ctrlpts_u * self.num_ctrlpts_v
        )

    def interpolation(self, knot_u, knot_v, knot_w):
        if isinstance(knot_u, float) and isinstance(knot_v, float) and isinstance(knot_w, float):
            point = self.single_point(knot_u, knot_v, knot_w)
            return point
        elif (
            isinstance(knot_u, (list, tuple, np.ndarray))
            and isinstance(knot_v, (list, tuple, np.ndarray))
            and isinstance(knot_w, (list, tuple, np.ndarray))
        ):
            knot_array_u = np.asarray(knot_u)
            knot_array_v = np.asarray(knot_v)
            knot_array_w = np.asarray(knot_w)
            if knot_array_u.ndim != 1.0:
                raise ValueError("Parameter array must be 1D")
            if knot_array_v.ndim != 1.0:
                raise ValueError("Parameter array must be 1D")
            if knot_array_w.ndim != 1.0:
                raise ValueError("Parameter array must be 1D")
            return np.array(
                [
                    self.single_point(parameter1, parameter2, parameter3)
                    for parameter1 in knot_array_u
                    for parameter2 in knot_array_v
                    for parameter3 in knot_array_w
                ]
            )

    def dxdknot(self, knot_u, knot_v, knot_w):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get partial derivative of [u, v, w]
        """
        return self.derivative(knot_u, knot_v, knot_w, normalize=False)

    def dxdctrlpts(self, knot_u, knot_v, knot_w):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get partial derivative of Pᵢ, return a numpy array [[C(u)_xᵢ=Nᵢ, 0, 0], [0, C(u)_yᵢ=Nᵢ, 0], [0, 0, C(u)_zᵢ=Nᵢ], ...]
        """
        coeff = self.coefficient(knot_u, knot_v)
        identity = np.eye(self.dimension, dtype=coeff.dtype)
        result = np.vstack([nshape * identity for nshape in coeff])
        return result

    def visualize(self, knot_u=None, knot_v=None, knot_w=None, resolution=100, **kwargs):
        knots_u = np.linspace(0.0, 1.0, resolution) if knot_u is None else np.asarray(knot_u)
        knots_v = np.linspace(0.0, 1.0, resolution) if knot_v is None else np.asarray(knot_v)
        knots_w = np.linspace(0.0, 1.0, resolution) if knot_w is None else np.asarray(knot_w)
        evalpts = self.interpolation(knots_u, knots_v, knots_w)
        plot = NurbsPlot(dimension=self.dimension)
        plot.append(
            evalpts=evalpts.reshape(knots_w.shape[0], knots_v.shape[0], knots_u.shape[0], -1),
            control_points=self.control_points.reshape(self.num_ctrlpts_w, self.num_ctrlpts_v, self.num_ctrlpts_u, -1),
        )
        plot.initialize(**kwargs)
        plot.plot_volume()

    def check_knot_vector(self, knot_u, knot_v, knot_w):
        knot_u = float(knot_u)
        knot_v = float(knot_v)
        knot_w = float(knot_w)
        if not self.validate_knot(knot_u):
            raise ValueError(f"Knot paramter {knot_u} must be in the interval [0, 1]")
        if not self.validate_knot(knot_v):
            raise ValueError(f"Knot paramter {knot_v} must be in the interval [0, 1]")
        if not self.validate_knot(knot_w):
            raise ValueError(f"Knot paramter {knot_w} must be in the interval [0, 1]")

    def _check_knot_vector(self, knot_vector, direction="u"):
        if direction == "u":
            return check_knot_vector(self._degree_u, knot_vector)
        elif direction == "v":
            return check_knot_vector(self._degree_v, knot_vector)
        else:
            return check_knot_vector(self._degree_w, knot_vector)

    def gather_boundary_ctrlpts(self):
        indices = np.arange(self.control_points.shape[0], dtype=np.int32).reshape(
            self.num_ctrlpts_w, self.num_ctrlpts_v, self.num_ctrlpts_u
        )
        boundary_ctrlpts_ids = []
        face_sizes = []
        current_prefix = 0
        faces = [
            indices[0, :, :],  # u=0 面 (uv-plane at w=0)
            indices[-1, :, :],  # u=max 面 (uv-plane at w=max)
            indices[:, 0, :],  # v=0 面 (uw-plane at v=0)
            indices[:, -1, :],  # v=max 面 (uw-plane at v=max)
            indices[:, :, 0],  # w=0 面 (vw-plane at u=0)
            indices[:, :, -1],  # w=max 面 (vw-plane at u=max)
        ]
        for face in faces:
            face_flat = face.reshape(-1)
            _, unique_idx = np.unique(face_flat, return_index=True)
            unique_face = face_flat[np.sort(unique_idx)]

            boundary_ctrlpts_ids.extend(unique_face.tolist())
            face_sizes.append(len(unique_face))
            current_prefix += len(unique_face)
        boundary_ctrlpts_ids = np.array(boundary_ctrlpts_ids, dtype=np.int32)
        knot_vector_list = [
            (self.knot_vector_u, self.knot_vector_v),
            (self.knot_vector_u, self.knot_vector_v),
            (self.knot_vector_u, self.knot_vector_w),
            (self.knot_vector_u, self.knot_vector_w),
            (self.knot_vector_v, self.knot_vector_w),
            (self.knot_vector_v, self.knot_vector_w),
        ]
        degree_list = [
            (self.degree_u, self.degree_v),
            (self.degree_u, self.degree_v),
            (self.degree_u, self.degree_w),
            (self.degree_u, self.degree_w),
            (self.degree_v, self.degree_w),
            (self.degree_v, self.degree_w),
        ]
        return face_sizes, boundary_ctrlpts_ids, knot_vector_list, degree_list

    def activate_boundary(self):
        raise NotImplementedError

    def update_boundary(self):
        raise NotImplementedError
