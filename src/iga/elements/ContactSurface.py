"""Contact-surface extraction from an initialized IGA patch."""

from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

import src.iga.config as config
from src.nurbs.core.NurbsGeometry import NurbsBasisFunction1d, NurbsBasisFunction2d

if TYPE_CHECKING:
    from src.iga.elements.Patch import Patch


@ti.data_oriented
class ContactSurface:
    def __init__(self, patch: "Patch"):
        self.num_surfaces = 2 * config.DIM * patch.num_patch
        self.num_surface_node = 0
        self.num_knots = [0, 0]

        self.num_knot = np.zeros((self.num_surfaces + 1, config.DIM - 1), dtype=np.int32)
        self.total_num_ctrlpts = np.zeros(self.num_surfaces + 1, dtype=np.int32)
        self.prefix_num_knot = np.zeros((self.num_surfaces + 1, config.DIM - 1), dtype=np.int32)
        self.prefix_total_num_ctrlpts = np.zeros(self.num_surfaces + 1, dtype=np.int32)

        self.basis = []
        self.initialize(patch)

    def initialize(self, patch: "Patch"):
        patch_sizes, boundary_ctrlpts_ids, knot_vector_lists, degree_lists = [], [], [], []
        for index, (name, meta) in enumerate(patch.primitive.body.items()):
            primitive = meta["primitive"]
            patch_size, boundary_ctrlpts_id, knot_vector_list, degree_list = primitive.gather_boundary_ctrlpts()
            patch_sizes.append(patch_size)
            boundary_ctrlpts_ids.append(boundary_ctrlpts_id)
            knot_vector_lists.append(knot_vector_list)
            degree_lists.append(degree_list)
            for knot_vec in knot_vector_list:
                self.num_knots[0] += len(knot_vec[0])
                if config.DIM == 3:
                    self.num_knots[1] += len(knot_vec[1])
            self.num_surface_node += len(boundary_ctrlpts_id)

        self.patch2surface = ti.Vector.field(config.DIM, ti.f64, shape=self.num_knots[0])
        self.elements = ti.Vector.field(config.DIM, ti.f64, shape=self.num_knots[0])
        self.surface_node_id = ti.field(ti.i32, shape=self.num_surface_node)
        self.ctrlpts_temp = ti.Vector.field(config.DIM, ti.f64, shape=self.num_surface_node)
        self.ctrlptsv_temp = ti.Vector.field(config.DIM, ti.f64, shape=self.num_surface_node)

        begin_knot = [0, 0]
        all_surface_ids = []
        for index, (name, meta) in enumerate(patch.primitive.body.items()):
            primitive = meta["primitive"]
            prefix_total_num_ctrlpts = patch.prefix_total_num_ctrlpts[index]
            patch_size, boundary_ctrlpts_id, knot_vector_list, degree_list = (
                patch_sizes[index],
                boundary_ctrlpts_ids[index],
                knot_vector_lists[index],
                degree_lists[index],
            )

            for surfIndex in range(2 * config.DIM):
                self.total_num_ctrlpts[2 * config.DIM * index + surfIndex + 1] = len(patch_size[surfIndex])
                self.num_knot[2 * config.DIM * index + surfIndex + 1, 0] = len(knot_vector_list[surfIndex][0])
                self.fill_knot_vector(begin_knot[0], np.asarray(knot_vector_list[surfIndex][0]), self.knot_vector_u)
                begin_knot[0] += len(knot_vector_list[surfIndex][0])
                if config.DIM == 3:
                    self.num_knot[2 * config.DIM * index + surfIndex + 1, 1] = len(knot_vector_list[surfIndex][1])
                    self.fill_knot_vector(begin_knot[1], np.asarray(knot_vector_list[surfIndex][1]), self.knot_vector_v)
                    begin_knot[1] += len(knot_vector_list[surfIndex][1])
                    self.basis.append(NurbsBasisFunction2d(degree_list[surfIndex][0], degree_list[surfIndex][1]))
                else:
                    self.basis.append(NurbsBasisFunction1d(degree_list[surfIndex][0]))

            boundary_ctrlpts_id = np.asarray(boundary_ctrlpts_id) + prefix_total_num_ctrlpts
            all_surface_ids.append(boundary_ctrlpts_id)
        all_surface_ids = np.concatenate(all_surface_ids, axis=0)
        self.surface_node_id.from_numpy(all_surface_ids)

        for d in range(config.DIM - 1):
            self.prefix_num_knot[:, d] = np.cumsum(self.num_knot[:, d])
        self.prefix_total_num_ctrlpts = np.cumsum(self.total_num_ctrlpts)

    @ti.kernel
    def fill_knot_vector(
        self, begin_index: ti.i32, knot_vector_src: ti.types.ndarray(), knot_vector_dst: ti.template()
    ):
        for i in range(knot_vector_src.shape[0]):
            knot_vector_dst[begin_index + i] = knot_vector_src[i]


__all__ = ["ContactSurface"]
