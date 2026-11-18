from src.utils.linalg import normalized
from src.utils.errors import SimulationStateError
from src.nurbs.Spline import Spline

import numpy as np
from itertools import product
from src.nurbs.Nurbs import find_span
from src.nurbs.NurbsPlot import NurbsPlot
from src.nurbs.Utilities import check_knot_vector, make_list


class SplineSurface(Spline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._degree = None
        self._knot_vector_u = None
        self._knot_vector_v = None
        self._multiplicity_u = None
        self._multiplicity_v = None
        self._element_u = None
        self._element_v = None
        self._control_points = None
        self._num_ctrlpts_u = 0
        self._num_ctrlpts_v = 0
        self._num_element_u = 0
        self._num_element_v = 0
        self._num_knot_u = 0
        self._num_knot_v = 0
        self._connectivity = None
        self._parent_indices = None
        self._area = None
        self._boundary = []
        self.degree = kwargs.get("degree", None)
        self.knot_vector_u = kwargs.get("knot_vector_u", None)
        self.knot_vector_v = kwargs.get("knot_vector_v", None)
        self.control_points = kwargs.get("control_point", None)

    @property
    def area(self):
        if self._area is None:
            raise SimulationStateError("Run estimate_area first!")
        return self._area

    @property
    def parent_indices(self):
        return self._parent_indices

    @parent_indices.setter
    def parent_indices(self, parent_indices):
        self._parent_indices = parent_indices

    @property
    def boundary(self):
        return self._boundary

    @property
    def degree(self):
        return self._degree_u, self._degree_v

    @property
    def connectivity(self):
        if self.topology_update:
            self._connectivity = self.get_connectivity()
            self.topology_update = False
        return self._connectivity

    @degree.setter
    def degree(self, degree):
        if degree is None:
            self._degree_u = None
            self._degree_v = None
        else:
            if isinstance(degree, (int, float)):
                degree = [degree, degree]
            elif isinstance(degree, (tuple, list, np.ndarray)):
                if len(list(degree)) != 2:
                    raise TypeError("degree should has a length of 2.")
            self._degree_u = degree[0]
            self._degree_v = degree[1]

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
    def num_ctrlpts_u(self):
        return self._num_ctrlpts_u

    @property
    def num_ctrlpts_v(self):
        return self._num_ctrlpts_v

    @property
    def num_ctrlpts(self):
        return self._num_ctrlpts_u * self._num_ctrlpts_v

    @property
    def num_ctrlpts_list(self):
        return np.array([self._num_ctrlpts_u, self._num_ctrlpts_v])

    @property
    def num_knot_u(self):
        return self._num_knot_u

    @property
    def num_knot_v(self):
        return self._num_knot_v

    @property
    def num_knot(self):
        return self._num_knot_u * self._num_knot_v

    @property
    def num_knot_list(self):
        return np.array([self._num_knot_u, self._num_knot_v])

    @property
    def num_element_u(self):
        return self._num_element_u

    @property
    def num_element_v(self):
        return self._num_element_v

    @property
    def num_element(self):
        return self._num_element_u * self._num_element_v

    @property
    def num_element_list(self):
        return np.array([self._num_element_u, self._num_element_v])

    @property
    def element_u(self):
        return self._element_u

    @property
    def element_v(self):
        return self._element_v

    @property
    def multiplicity_u(self):
        return self._multiplicity_u

    @property
    def multiplicity_v(self):
        return self._multiplicity_v

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

            if self._knot_vector_u is None:
                raise ValueError("Curve knot vector u must be set before setting control points")

            if self._knot_vector_v is None:
                raise ValueError("Curve knot vector v must be set before setting control points")

            if ctrlpt_array.shape[-1] != self.dimension:
                self.dimension = ctrlpt_array.shape[-1]

            if ctrlpt_array.shape[0] != self._num_ctrlpts_u * self._num_ctrlpts_v:
                raise TypeError(
                    f"Array Control points must have a shape of {[self._num_ctrlpts_u * self._num_ctrlpts_v, self.dimension]}"
                )

            self._control_points = ctrlpt_array.copy()

    def get_connectivity(self):
        col_idx, row_idx = np.meshgrid(
            np.arange(self.num_ctrlpts_v - 1), np.arange(self.num_ctrlpts_u - 1), indexing="ij"
        )
        base_idx = row_idx + col_idx * self.num_ctrlpts_u
        return np.stack(
            [base_idx, base_idx + 1, base_idx + self.num_ctrlpts_u, base_idx + self.num_ctrlpts_u + 1], axis=-1
        ).reshape(-1, 4)

    def boundary_derivative(self, knot_u, knot_v):
        if knot_v == 0 and 0 <= knot_u < 1:
            return self.boundary[0].derivative(knot_u)
        elif knot_u == 1 and 0 <= knot_v < 1:
            return self.boundary[1].derivative(knot_v)
        elif knot_v == 1 and 0 < knot_u <= 1:
            return self.boundary[2].derivative(knot_u)
        elif knot_u == 0 and 0 < knot_v <= 1:
            return self.boundary[3].derivative(knot_v)
        else:
            raise ValueError(f"Knot value: {knot_u, knot_v} are not on the boundary")

    def ctrlpt_id(self, knot_u, knot_v):
        span_u = find_span(self.num_ctrlpts_u, knot_u, self.degree_u, self.knot_vector_u)
        span_v = find_span(self.num_ctrlpts_v, knot_v, self.degree_v, self.knot_vector_v)
        control_point_id = np.array(
            list(product(np.arange(span_v - self.degree_v, span_v + 1), np.arange(span_u - self.degree_u, span_u + 1)))
        )
        return control_point_id[:, 1] + control_point_id[:, 0] * self.num_ctrlpts_u

    def normal(self, knot_u, knot_v):
        tangent_u, tangent_v = self.tangent(knot_u, knot_v)
        return normalized(np.cross(tangent_u, tangent_v))

    def boundary_normal(self, knot_u, knot_v):
        if np.abs(knot_v) < 1e-12 and 0 <= knot_u < 1:
            return self.boundary[0].normal(knot_u)
        elif np.abs(knot_u - 1) < 1e-12 and 0 <= knot_v < 1:
            return self.boundary[1].normal(knot_v)
        elif np.abs(knot_v - 1) < 1e-12 and 0 < knot_u <= 1:
            return self.boundary[2].normal(knot_u)
        elif np.abs(knot_u) < 1e-12 and 0 < knot_v <= 1:
            return self.boundary[3].normal(knot_v)
        else:
            raise ValueError(f"Knot value: {knot_u, knot_v} are not on the boundary")

    def tangent(self, knot_u, knot_v):
        return self.derivative(knot_u, knot_v, normalize=True)

    def boundary_tangent(self, knot_u, knot_v):
        if knot_v == 0 and 0 <= knot_u < 1:
            return self.boundary[0].tangent(knot_u)
        elif knot_u == 1 and 0 <= knot_v < 1:
            return self.boundary[1].tangent(knot_v)
        elif knot_v == 1 and 0 < knot_u <= 1:
            return self.boundary[2].tangent(knot_u)
        elif knot_u == 0 and 0 < knot_v <= 1:
            return self.boundary[3].tangent(knot_v)
        else:
            raise ValueError(f"Knot value: {knot_u, knot_v} are not on the boundary")

    def boundary_inversion(self, points):
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != self.dimension:
            raise ValueError(f"Point must be in R$^{self.dimension}$")
        num_datapts = points.shape[0]
        north = np.insert(np.hstack(self.boundary[0].inversion(points)), 1, np.zeros(num_datapts), axis=1)
        east = np.insert(np.hstack(self.boundary[1].inversion(points)), 0, np.ones(num_datapts), axis=1)
        south = np.insert(np.hstack(self.boundary[2].inversion(points)), 1, np.ones(num_datapts), axis=1)
        west = np.insert(np.hstack(self.boundary[3].inversion(points)), 0, np.zeros(num_datapts), axis=1)
        all_matrices = np.stack((north, south, west, east), axis=0)
        min_indices = np.argmin(all_matrices[:, :, -1], axis=0)
        new_matrix = np.array([all_matrices[min_indices[row], row, :3] for row in range(all_matrices.shape[1])])
        return new_matrix[:, 0:2], new_matrix[:, 2]

    def curvature(self, knot_u, knot_v, get_center=False):
        Su, Sv, Suu, Svv, Suv = self.second_derivative(knot_u, knot_v, first_derivative=True)
        normal = self.normal(knot_u, knot_v)

        l = np.dot(Suu, normal)
        m = np.dot(Suv, normal)
        n = np.dot(Svv, normal)
        II = np.array([[l, m], [m, n]])

        e = np.dot(Su, Su)
        f = np.dot(Su, Sv)
        g = np.dot(Sv, Sv)
        I = np.array([[e, f], [f, g]])

        eigenvalues = np.linalg.eigvals(np.linalg.inv(I).dot(II))
        k1, k2 = eigenvalues

        R1 = 1 / k1 if k1 != 0 else np.inf
        R2 = 1 / k2 if k2 != 0 else np.inf

        if not get_center:
            return abs(k1), abs(k2)
        else:
            p = self.single_point(knot_u, knot_v)
            return abs(k1), abs(k2), p + R1 * normal, p + R2 * normal

    def interpolation(self, knot_u, knot_v):
        if isinstance(knot_u, float) and isinstance(knot_v, float):
            point = self.single_point(knot_u, knot_v)
            return point
        elif isinstance(knot_u, (list, tuple, np.ndarray)) and isinstance(knot_v, (list, tuple, np.ndarray)):
            knot_array_u = np.asarray(knot_u)
            knot_array_v = np.asarray(knot_v)
            if knot_array_u.ndim != 1.0:
                raise ValueError("Parameter array must be 1D")
            if knot_array_v.ndim != 1.0:
                raise ValueError("Parameter array must be 1D")
            return np.array(
                [
                    self.single_point(parameter1, parameter2)
                    for parameter1 in knot_array_u
                    for parameter2 in knot_array_v
                ]
            )

    def dxdknot(self, knot_u, knot_v):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get the partial derivative of [u, v, w]
        """
        return self.derivative(knot_u, knot_v, normalize=False)

    def d2xd2knot(self, knot_u, knot_v):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get the second partial derivative of [u, v, w]
        """
        return self.second_derivative(knot_u, knot_v, normalize=False)

    def dxdctrlpts(self, knot_u, knot_v):
        """
        for a given point in nurbs space, C(u)=∑NᵢPᵢ, with Pᵢ = [xᵢ, yᵢ, zᵢ] for three dimension and [xᵢ, yᵢ] for two dimension
        Get the partial derivative of Pᵢ, return a numpy array [[C(u)_xᵢ=Nᵢ, 0, 0], [0, C(u)_yᵢ=Nᵢ, 0], [0, 0, C(u)_zᵢ=Nᵢ], ...]
        """
        coeff = self.coefficient(knot_u, knot_v)
        identity = np.eye(self.dimension, dtype=coeff.dtype)
        result = np.vstack([nshape * identity for nshape in coeff])
        return result

    def convex_hull(self, start_knot, end_knot):
        if start_knot[0] > end_knot[0] or start_knot[1] > end_knot[1]:
            raise ValueError(f"Start knot {start_knot} must be less than or equal to end knot {end_knot}")

        start_span = [
            find_span(self.num_ctrlpts_u, start_knot[0], self.degree_u, self.knot_vector_u),
            find_span(self.num_ctrlpts_v, start_knot[1], self.degree_v, self.knot_vector_v),
        ]
        end_span = [
            find_span(self.num_ctrlpts_u, end_knot[0], self.degree_u, self.knot_vector_u),
            find_span(self.num_ctrlpts_v, end_knot[1], self.degree_v, self.knot_vector_v),
        ]
        start_support_knot_gap = [
            start_knot[0] - self.knot_vector_u[start_span[0]],
            start_knot[1] - self.knot_vector_v[start_span[1]],
        ]
        end_support_knot_gap = [
            self.knot_vector_u[end_span[0] + 1] - end_knot[0],
            self.knot_vector_v[end_span[1] + 1] - end_knot[1],
        ]
        start_support_knot_gap = [
            start_support_knot_gap[0] if abs(start_support_knot_gap[0]) > 1e-14 else 1e-5,
            start_support_knot_gap[1] if abs(start_support_knot_gap[1]) > 1e-14 else 1e-5,
        ]
        end_support_knot_gap = [
            end_support_knot_gap[0] if abs(end_support_knot_gap[1]) > 1e-14 else 1e-5,
            end_support_knot_gap[1] if abs(end_support_knot_gap[1]) > 1e-14 else 1e-5,
        ]
        start_support_knot = [
            max(0.0, start_knot[0] - 0.01 * start_support_knot_gap[0]),
            max(0.0, start_knot[1] - 0.01 * start_support_knot_gap[1]),
        ]
        end_support_knot = [
            min(1.0, end_knot[0] + 0.01 * end_support_knot_gap[0]),
            min(1.0, end_knot[1] + 0.01 * end_support_knot_gap[1]),
        ]

        inserted_knot_u = []
        inserted_knot_v = []
        if not start_support_knot[0] in self.knot_vector_u:
            inserted_knot_u.append(start_support_knot[0])
        if not start_support_knot[1] in self.knot_vector_v:
            inserted_knot_v.append(start_support_knot[1])
        if not start_knot[0] in self.knot_vector_u:
            inserted_knot_u.append(start_knot[0])
        if not start_knot[1] in self.knot_vector_v:
            inserted_knot_v.append(start_knot[1])
        if not end_knot[0] in self.knot_vector_u:
            inserted_knot_u.append(end_knot[0])
        if not end_knot[1] in self.knot_vector_v:
            inserted_knot_v.append(end_knot[1])
        if not end_support_knot[0] in self.knot_vector_u:
            inserted_knot_u.append(end_support_knot[0])
        if not end_support_knot[1] in self.knot_vector_v:
            inserted_knot_v.append(end_support_knot[1])
        surface_copy = self.copy()
        surface_copy.insert_knot(inserted_knot_u, inserted_knot_v)

        begin_element_uid = np.where(surface_copy.element_u[:, 0] == start_knot[0])[0][0]
        begin_element_vid = np.where(surface_copy.element_v[:, 0] == start_knot[1])[0][0]
        end_element_uid = np.where(surface_copy.element_u[:, 1] == end_knot[0])[0][0]
        end_element_vid = np.where(surface_copy.element_v[:, 1] == end_knot[1])[0][0]
        ctrlpts = surface_copy.control_points.reshape(surface_copy.num_ctrlpts_u, surface_copy.num_ctrlpts_v, -1)
        return ctrlpts[
            begin_element_uid : end_element_uid + surface_copy.degree_u + 1,
            begin_element_vid : end_element_vid + surface_copy.degree_v + 1,
            :,
        ].reshape(-1, surface_copy.dimension)

    def estimate_area(self, resolution=100):
        self._area = np.zeros(self.num_element_u * self.num_element_v)
        for index_v, knot_v in enumerate(self.element_v):
            for index_u, knot_u in enumerate(self.element_u):
                knots_u = np.linspace(*knot_u, resolution)
                knots_v = np.linspace(*knot_v, resolution)
                evalpts = self.interpolation(knots_u, knots_v).reshape(resolution, resolution, -1)
                p00 = evalpts[:-1, :-1]
                p01 = evalpts[:-1, 1:]
                p10 = evalpts[1:, :-1]
                p11 = evalpts[1:, 1:]

                # First triangle area
                v1 = p01 - p00
                v2 = p11 - p00
                area1 = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=-1)

                # Second triangle area
                v3 = p10 - p00
                area2 = 0.5 * np.linalg.norm(np.cross(v3, v1), axis=-1)
                self._area[index_u + index_v * self.num_element_u] = np.sum(area1 + area2)

    def visualize(self, knot_u=None, knot_v=None, resolution=100, **kwargs):
        knots_u = np.linspace(0.0, 1.0, resolution) if knot_u is None else np.asarray(knot_u)
        knots_v = np.linspace(0.0, 1.0, resolution) if knot_v is None else np.asarray(knot_v)
        evalpts = self.interpolation(knots_u, knots_v)
        plot = NurbsPlot(dimension=self.dimension)
        plot.append(
            evalpts=evalpts.reshape(knots_v.shape[0], knots_u.shape[0], -1),
            control_points=self.control_points.reshape(self.num_ctrlpts_v, self.num_ctrlpts_u, -1),
        )
        if "other_surfaces" in kwargs:
            surfaces = make_list(kwargs["other_surfaces"])
            for surf in surfaces:
                evalpts = surf.interpolation(knots_u, knots_v)
                plot.append(
                    evalpts=evalpts.reshape(knots_v.shape[0], knots_u.shape[0], -1),
                    control_points=surf.control_points.reshape(surf.num_ctrlpts_v, surf.num_ctrlpts_u, -1),
                )
        plot.initialize(**kwargs)
        plot.plot_surface()

    def check_knot_vector(self, knot_u, knot_v):
        knot_u = float(knot_u)
        knot_v = float(knot_v)
        if not self.validate_knot(knot_u):
            raise ValueError(f"Knot paramter {knot_u} must be in the interval [0, 1]")
        if not self.validate_knot(knot_v):
            raise ValueError(f"Knot paramter {knot_v} must be in the interval [0, 1]")

    def _check_knot_vector(self, knot_vector, direction="u"):
        if direction == "u":
            return check_knot_vector(self._degree_u, knot_vector)
        else:
            return check_knot_vector(self._degree_v, knot_vector)

    def _check_dimension(self, dimension):
        if dimension != 2 and dimension != 3:
            raise ValueError("Dimension must be 2 or 3")

    def gather_boundary_ctrlpts(self):
        indices = np.arange(self.control_points.shape[0], dtype=np.int32).reshape(
            self.num_ctrlpts_v, self.num_ctrlpts_u
        )
        boundary_ctrlpts_ids = []
        patch_sizes = []
        edges = [indices[0, :], indices[:, -1], indices[-1, :], indices[:, 0]]
        current_prefix = 0
        for edge in edges:
            _, unique_idx = np.unique(edge, return_index=True)
            unique_edge = edge[np.sort(unique_idx)]
            boundary_ctrlpts_ids.extend(unique_edge.tolist())
            patch_sizes.append(len(unique_edge))
            current_prefix += len(unique_edge)
        boundary_ctrlpts_ids = np.array(boundary_ctrlpts_ids, dtype=np.int32)
        knot_vector_list = [self.knot_vector_u, self.knot_vector_v, self.knot_vector_u, self.knot_vector_v]
        degree_list = [self.degree_u, self.degree_v, self.degree_u, self.degree_v]
        return patch_sizes, boundary_ctrlpts_ids, knot_vector_list, degree_list

    def activate_boundary(self):
        raise NotImplementedError

    def update_boundary(self):
        raise NotImplementedError

    def update(self, knot_u, knot_v, control_points):
        self.knot_vector_u = knot_u
        self.knot_vector_v = knot_v
        self.control_points = control_points
        self.get_connectivity()
        self.update_boundary()
