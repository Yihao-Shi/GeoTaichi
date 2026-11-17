from src.utils.linalg import normalized
from src.utils.errors import ConfigurationError
from src.nurbs.Utilities import bspline_curve

import numpy as np
from src.nurbs.SplineVolume import SplineVolume
from src.nurbs.NurbsBasis import *
from src.nurbs.Operations import *
from src.nurbs.Utilities import *


class BSplineVolume(SplineVolume):

    def activate_boundary(self):
        if len(self._boundary) == 0:
            ctrlpts = self.control_points.reshape(self.num_ctrlpts_w, self.num_ctrlpts_v, self.num_ctrlpts_u, -1)
            indices = np.arange(self.control_points.shape[0], dtype=np.int32).reshape(
                self.num_ctrlpts_w, self.num_ctrlpts_v, self.num_ctrlpts_u
            )
            ctrlpt_uv_0 = ctrlpts[0, ...].reshape(-1, self.dimension)
            index_uv_0 = indices[0, ...].reshape(-1)
            ctrlpt_uv_1 = ctrlpts[-1, ...].reshape(-1, self.dimension)
            index_uv_1 = indices[-1, ...].reshape(-1)
            ctrlpt_vw_0 = ctrlpts[..., 0, :].reshape(-1, self.dimension)
            index_vw_0 = indices[..., 0].reshape(-1)
            ctrlpt_vw_1 = ctrlpts[..., -1, :].reshape(-1, self.dimension)
            index_vw_1 = indices[..., -1].reshape(-1)
            ctrlpt_uw_0 = ctrlpts[:, 0, ...].reshape(-1, self.dimension)
            index_uw_0 = indices[:, 0, ...].reshape(-1)
            ctrlpt_uw_1 = ctrlpts[:, -1, ...].reshape(-1, self.dimension)
            index_uw_1 = indices[:, -1, ...].reshape(-1)
            self._boundary.append(
                bspline_curve(
                    degree_u=self.degree_u,
                    degree_v=self.degree_v,
                    knot_vector_u=self.knot_vector_u,
                    knot_vector_v=self.knot_vector_v,
                    control_point=ctrlpt_uv_0,
                    parent_indices=index_uv_0,
                )
            )
            self._boundary.append(
                bspline_curve(
                    degree_u=self.degree_u,
                    degree_v=self.degree_v,
                    knot_vector_u=self.knot_vector_u,
                    knot_vector_v=self.knot_vector_v,
                    control_point=ctrlpt_uv_1,
                    parent_indices=index_uv_1,
                )
            )
            self._boundary.append(
                bspline_curve(
                    degree_u=self.degree_v,
                    degree_v=self.degree_w,
                    knot_vector_u=self.knot_vector_v,
                    knot_vector_v=self.knot_vector_w,
                    control_point=ctrlpt_vw_0,
                    parent_indices=index_vw_0,
                )
            )
            self._boundary.append(
                bspline_curve(
                    degree_u=self.degree_v,
                    degree_v=self.degree_w,
                    knot_vector_u=self.knot_vector_v,
                    knot_vector_v=self.knot_vector_w,
                    control_point=ctrlpt_vw_1,
                    parent_indices=index_vw_1,
                )
            )
            self._boundary.append(
                bspline_curve(
                    degree_u=self.degree_u,
                    degree_v=self.degree_w,
                    knot_vector_u=self.knot_vector_u,
                    knot_vector_v=self.knot_vector_w,
                    control_point=ctrlpt_uw_0,
                    parent_indices=index_uw_0,
                )
            )
            self._boundary.append(
                bspline_curve(
                    degree_u=self.degree_u,
                    degree_v=self.degree_w,
                    knot_vector_u=self.knot_vector_u,
                    knot_vector_v=self.knot_vector_w,
                    control_point=ctrlpt_uw_1,
                    parent_indices=index_uw_1,
                )
            )

    def update_boundary(self):
        if len(self._boundary) > 0:
            ctrlpts = self.control_points.reshape(self.num_ctrlpts_w, self.num_ctrlpts_v, self.num_ctrlpts_u, -1)
            self._boundary[0].knot_vector_u = self.knot_vector_u
            self._boundary[0].knot_vector_v = self.knot_vector_v
            self._boundary[0].control_points = ctrlpts[0, ...].reshape(-1, self.dimension)
            self._boundary[1].knot_vector_u = self.knot_vector_u
            self._boundary[1].knot_vector_v = self.knot_vector_v
            self._boundary[1].control_points = ctrlpts[-1, ...].reshape(-1, self.dimension)
            self._boundary[2].knot_vector_u = self.knot_vector_v
            self._boundary[2].knot_vector_v = self.knot_vector_w
            self._boundary[2].control_points = ctrlpts[..., 0, :].reshape(-1, self.dimension)
            self._boundary[3].knot_vector_u = self.knot_vector_v
            self._boundary[3].knot_vector_v = self.knot_vector_w
            self._boundary[3].control_points = ctrlpts[..., -1, :].reshape(-1, self.dimension)
            self._boundary[4].knot_vector_u = self.knot_vector_u
            self._boundary[4].knot_vector_v = self.knot_vector_w
            self._boundary[4].control_points = ctrlpts[:, 0, ...].reshape(-1, self.dimension)
            self._boundary[5].knot_vector_u = self.knot_vector_u
            self._boundary[5].knot_vector_v = self.knot_vector_w
            self._boundary[5].control_points = ctrlpts[:, -1, ...].reshape(-1, self.dimension)

    def coefficient(self, knot_u, knot_v, knot_w):
        knot_u, knot_v, knot_w = float(knot_u), float(knot_v), float(knot_w)
        self.check_knot_vector(knot_u, knot_v, knot_w)
        return bspline_basis3d(
            knot_u,
            knot_v,
            knot_w,
            self.degree_u,
            self.degree_v,
            self.degree_w,
            self.knot_vector_u,
            self.knot_vector_v,
            self.knot_vector_w,
        )

    def coefficient_derivate(self, knot_u, knot_v, knot_w):
        knot_u, knot_v, knot_w = float(knot_u), float(knot_v), float(knot_w)
        self.check_knot_vector(knot_u, knot_v, knot_w)
        return bspline_basis_ders3d(
            knot_u,
            knot_v,
            knot_w,
            self.degree_u,
            self.degree_v,
            self.degree_w,
            self.knot_vector_u,
            self.knot_vector_v,
            self.knot_vector_w,
        )

    def coefficient_second_derivate(self, knot_u, knot_v, knot_w):
        knot_u, knot_v, knot_w = float(knot_u), float(knot_v), float(knot_w)
        self.check_knot_vector(knot_u, knot_v, knot_w)
        return bspline_basis2nd_ders3d(
            knot_u,
            knot_v,
            knot_w,
            self.degree_u,
            self.degree_v,
            self.degree_w,
            self.knot_vector_u,
            self.knot_vector_v,
            self.knot_vector_w,
        )

    def single_point(self, knot_u, knot_v, knot_w):
        knot_u, knot_v, knot_w = float(knot_u), float(knot_v), float(knot_w)
        self.check_knot_vector(knot_u, knot_v, knot_w)
        return bspline_basis_interpolations3d(
            knot_u,
            knot_v,
            knot_w,
            self.degree_u,
            self.degree_v,
            self.degree_w,
            self.knot_vector_u,
            self.knot_vector_v,
            self.knot_vector_w,
            self.control_points,
        )

    def derivative(self, knot_u, knot_v, knot_w, normalize=False):
        knot = float(knot)
        self.check_knot_vector(knot_u, knot_v, knot_w)
        _, derivatives = bspline_basis_interpolations_ders3d(
            knot_u,
            knot_v,
            knot_w,
            self.degree_u,
            self.degree_v,
            self.degree_w,
            self.knot_vector_u,
            self.knot_vector_v,
            self.knot_vector_w,
            self.control_points,
        )
        if normalize:
            return normalized(derivatives[0]), normalized(derivatives[1]), normalized(derivatives[2])
        else:
            return derivatives[0], derivatives[1], derivatives[2]

    def second_derivative(self, knot_u, knot_v, knot_w, first_derivative=False, normalize=False):
        self.check_knot_vector(knot_u, knot_v, knot_w)
        _, derivatives, sceond_derivatives = bspline_basis_interpolations2nd_ders3d(
            knot_u,
            knot_v,
            knot_w,
            self.degree_u,
            self.degree_v,
            self.degree_w,
            self.knot_vector_u,
            self.knot_vector_v,
            self.knot_vector_w,
            self.control_points,
        )
        if normalize:
            get_value = tuple(
                (
                    normalized(sceond_derivatives[0]),
                    normalized(sceond_derivatives[1]),
                    normalized(sceond_derivatives[2]),
                    normalized(sceond_derivatives[3]),
                    normalized(sceond_derivatives[4]),
                    normalized(sceond_derivatives[5]),
                )
            )
            if first_derivative:
                get_value = (
                    tuple((normalized(derivatives[0]), normalized(derivatives[1]), normalized(derivatives[3])))
                    + get_value
                )
            return get_value
        else:
            get_value = tuple(
                (
                    sceond_derivatives[0],
                    sceond_derivatives[1],
                    sceond_derivatives[2],
                    sceond_derivatives[3],
                    sceond_derivatives[4],
                    sceond_derivatives[5],
                )
            )
            if first_derivative:
                get_value = tuple((derivatives[0], derivatives[1], derivatives[2])) + get_value
            return get_value

    def insert_knot(self, knot_list_u=list(), knot_list_v=list(), knot_list_w=list()):
        knot_list_u = np.asarray(make_list(knot_list_u))
        knot_list_v = np.asarray(make_list(knot_list_v))
        knot_list_w = np.asarray(make_list(knot_list_w))

        wpoints = self.control_points.reshape(self.num_ctrlpts_w, -1)
        new_knot, new_wpoint = knot_insertion(self.degree_w, self.knot_vector_w, wpoints, addition_knot=knot_list_w)
        self.knot_vector_w = new_knot
        point = np.swapaxes(
            new_wpoint.reshape(self.num_ctrlpts_w, self.num_ctrlpts_v, self.num_ctrlpts_u, -1), 1, 0
        ).reshape(self.num_ctrlpts_w, -1)

        new_knot, new_wpoint = knot_insertion(self.degree_v, self.knot_vector_v, point, addition_knot=knot_list_v)
        self.knot_vector_v = new_knot
        point = np.swapaxes(
            new_wpoint.reshape(self.num_ctrlpts_v, self.num_ctrlpts_w, self.num_ctrlpts_u, -1), 2, 0
        ).reshape(self.num_ctrlpts_v, -1)

        new_knot, new_wpoint = knot_insertion(self.degree_u, self.knot_vector_u, point, addition_knot=knot_list_u)
        self.knot_vector_u = new_knot
        point = np.swapaxes(
            new_wpoint.reshape(self.num_ctrlpts_u, self.num_ctrlpts_v, self.num_ctrlpts_w, -1), 2, 0
        ).reshape(self.num_ctrlpts_u, -1)

        self.control_points = new_wpoint.reshape(self.num_ctrlpts_u * self.num_ctrlpts_v * self.num_ctrlpts_w, -1)
        self.update_boundary()

    def refine_knot(self, density=[1, 1, 1], knot_list_u=list(), knot_list_v=list(), knot_list_w=list()):
        if len(list(density)) != 3:
            raise ConfigurationError("The dimension of density must be 3")
        knot_list_u = np.asarray(make_list(knot_list_u))
        knot_list_v = np.asarray(make_list(knot_list_v))
        knot_list_w = np.asarray(make_list(knot_list_w))

        wpoints = self.control_points.reshape(self.num_ctrlpts_w, -1)
        new_knot, new_wpoint = knot_refinement(
            self.degree_w, self.knot_vector_w, wpoints, density=density[2], addition_knot=knot_list_w
        )
        self.knot_vector_w = new_knot
        point = np.swapaxes(
            new_wpoint.reshape(self.num_ctrlpts_w, self.num_ctrlpts_v, self.num_ctrlpts_u, -1), 1, 0
        ).reshape(self.num_ctrlpts_w, -1)

        new_knot, new_wpoint = knot_refinement(
            self.degree_v, self.knot_vector_v, point, density=density[1], addition_knot=knot_list_v
        )
        self.knot_vector_v = new_knot
        point = np.swapaxes(
            new_wpoint.reshape(self.num_ctrlpts_v, self.num_ctrlpts_w, self.num_ctrlpts_u, -1), 2, 0
        ).reshape(self.num_ctrlpts_v, -1)

        new_knot, new_wpoint = knot_refinement(
            self.degree_u, self.knot_vector_u, point, density=density[0], addition_knot=knot_list_u
        )
        self.knot_vector_u = new_knot
        point = np.swapaxes(
            new_wpoint.reshape(self.num_ctrlpts_u, self.num_ctrlpts_v, self.num_ctrlpts_w, -1), 2, 0
        ).reshape(self.num_ctrlpts_u, -1)

        self.control_points = new_wpoint.reshape(self.num_ctrlpts_u * self.num_ctrlpts_v * self.num_ctrlpts_w, -1)
        self.update_boundary()

    def update(self, knot_u, knot_v, knot_w, control_points):
        self.knot_vector_u = knot_u
        self.knot_vector_v = knot_v
        self.knot_vector_w = knot_w
        self.control_points = control_points
        self.get_connectivity()
        self.update_boundary()

    @classmethod
    def from_dict(cls, data: dict):
        needed_keys = {"degree", "knot_vector_u", "knot_vector_v", "knot_vector_w", "control_points"}
        missing_keys = needed_keys ^ data.keys()
        if missing_keys:
            raise ValueError(f"Missing the following information needed to instaniate curve {missing_keys}")
        degree = data.get("degree", [2, 2, 2])
        if not isinstance(degree, (int, float)):
            raise TypeError("degree must be in int type")
        knot_vector_u = np.array(data("knot_vector_u"))
        knot_vector_v = np.array(data("knot_vector_v"))
        knot_vector_w = np.array(data("knot_vector_w"))
        control_points = np.array(data("control_points"))

        dimension = control_points.shape[-1]
        cls(degree=degree, dimension=dimension)
        cls.knot_vector_u = knot_vector_u
        cls.knot_vector_v = knot_vector_v
        cls.knot_vector_w = knot_vector_w
        cls.control_points = control_points
        return cls

    def to_dict(self):
        return {
            "degree": self.degree,
            "knot_vector_u": self.knot_vector_u.to_list(),
            "knot_vector_v": self.knot_vector_v.to_list(),
            "knot_vector_w": self.knot_vector_w.to_list(),
            "control_points": self.control_points.to_list(),
        }

    def write(self, path="output.txt"):
        with open(path, "w") as file:
            file.write("degree_u " + str(self.degree_u) + "\n")
            file.write("degree_v " + str(self.degree_v) + "\n")
            file.write("degree_w " + str(self.degree_w) + "\n")
            file.write("\n")
            file.write("knot_vector_u " + " ".join(str(x) for x in self.knot_vector_u) + "\n")
            file.write("\n")
            file.write("knot_vector_v " + " ".join(str(x) for x in self.knot_vector_v) + "\n")
            file.write("\n")
            file.write("knot_vector_w " + " ".join(str(x) for x in self.knot_vector_w) + "\n")
            file.write("\n")
            file.write(
                "control_points "
                + str(self.num_ctrlpts_u)
                + " "
                + str(self.num_ctrlpts_v)
                + " "
                + str(self.num_ctrlpts_w)
                + "\n"
            )
            for ctrlpts in self.control_points:
                file.write("c " + " ".join(str(x) for x in ctrlpts) + "\n")

    def read(self, path="patch.txt"):
        num_ctrlpts = None
        ctrlpts = []
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
                    if line_info.startswith("degree_w"):
                        parts = line_info.split()
                        self.degree_w = int(parts[1])
                    if line_info.startswith("knot_vector_u "):
                        parts = line_info.split()
                        self.knot_vector_u = np.array([float(parts[knot_id]) for knot_id in range(1, len(parts))])
                    if line_info.startswith("knot_vector_v "):
                        parts = line_info.split()
                        self.knot_vector_v = np.array([float(parts[knot_id]) for knot_id in range(1, len(parts))])
                    if line_info.startswith("knot_vector_w "):
                        parts = line_info.split()
                        self.knot_vector_w = np.array([float(parts[knot_id]) for knot_id in range(1, len(parts))])
                    if line_info.startswith("control_points "):
                        parts = line_info.split()
                        num_ctrlpts = [int(parts[1]), int(parts[2]), int(parts[3])]
                        assert (
                            self.num_ctrlpts_u == num_ctrlpts[0]
                            and self.num_ctrlpts_v == num_ctrlpts[1]
                            and self.num_ctrlpts_w == num_ctrlpts[2]
                        )
                    if line_info.startswith("c "):
                        parts = line_info.split()
                        if num_ctrlpts is not None:
                            ctrlpts.append([float(parts[knot_id]) for knot_id in range(1, len(parts))])
        self.control_points = np.asarray(ctrlpts)
        self.activate_boundary()
