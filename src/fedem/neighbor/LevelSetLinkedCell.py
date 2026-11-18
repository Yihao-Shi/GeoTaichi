from src.utils.ScalarFunction import linearize3d
from src.utils.errors import CapacityError
from src.fedem.neighbor.LevelSet import LevelSetCandidateBase

import math
import numpy as np
import taichi as ti
from src.fedem.neighbor.LinkedCell import FEDEMLinkedCell
from src.utils.PrefixSum import PrefixSumExecutor


@ti.data_oriented
class LevelSetLinkedCell(LevelSetCandidateBase):
    """Per-rebuild rigid-AABB cell count, prefix sum, fill, and node query."""

    def __init__(self, simulation, patch, rigid_count, maximum_radius):
        super().__init__(simulation, patch, rigid_count)
        self.domain = tuple(float(value) for value in simulation.domain)
        if len(self.domain) != 3 or min(self.domain) <= 0.0:
            raise ValueError("FEM--LSDEM linked cell requires a positive 3D domain")
        self.grid_size = 2.0 * float(maximum_radius)
        if not math.isfinite(self.grid_size) or self.grid_size <= 0.0:
            raise ValueError("FEM--LSDEM linked-cell size must be positive")
        self.inverse_grid_size = 1.0 / self.grid_size
        self.cell_number = tuple(max(int(math.floor(length * self.inverse_grid_size)) + 1, 1) for length in self.domain)
        self.cell_count = int(np.prod(self.cell_number))
        self.cell_scan = PrefixSumExecutor(self.cell_count + 1)
        self.cell_offsets = ti.field(dtype=ti.i32, shape=self.cell_scan.get_length())
        self.cell_cursor = ti.field(dtype=ti.i32, shape=self.cell_count)
        self.membership_capacity = int(
            getattr(
                simulation,
                "max_levelset_cell_pairs",
                max(8 * self.max_contact_pairs, self.rigid_count, 1),
            )
        )
        self.cell_rigid = ti.field(dtype=ti.i32, shape=self.membership_capacity)

    @ti.func
    def _cell_span(self, rigid_id, domain, cell_number):
        lower = ti.max(self.aabbs.aabbs[rigid_id].min, 0.0)
        upper = ti.min(self.aabbs.aabbs[rigid_id].max, domain)
        begin = ti.cast(ti.floor(lower * self.inverse_grid_size), ti.i32)
        end = ti.cast(ti.floor(upper * self.inverse_grid_size), ti.i32)
        begin = ti.max(begin, 0)
        end = ti.min(end, cell_number - 1)
        valid = True
        for component in ti.static(range(3)):
            valid = valid and lower[component] <= upper[component]
        return begin, end, valid

    @ti.kernel
    def _count_memberships(
        self,
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        self.cell_offsets.fill(0)
        for rigid_id in range(self.rigid_count):
            begin, end, valid = self._cell_span(rigid_id, domain, cell_number)
            if valid:
                for i, j, k in ti.ndrange(
                    (begin[0], end[0] + 1),
                    (begin[1], end[1] + 1),
                    (begin[2], end[2] + 1),
                ):
                    cell = linearize3d(i, j, k, cell_number)
                    ti.atomic_add(self.cell_offsets[cell + 1], 1)

    @ti.kernel
    def _fill_memberships(
        self,
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        self.cell_cursor.fill(0)
        for rigid_id in range(self.rigid_count):
            begin, end, valid = self._cell_span(rigid_id, domain, cell_number)
            if valid:
                for i, j, k in ti.ndrange(
                    (begin[0], end[0] + 1),
                    (begin[1], end[1] + 1),
                    (begin[2], end[2] + 1),
                ):
                    cell = linearize3d(i, j, k, cell_number)
                    offset = ti.atomic_add(self.cell_cursor[cell], 1)
                    self.cell_rigid[self.cell_offsets[cell] + offset] = rigid_id

    @ti.kernel
    def _count_candidates(
        self,
        fem_position: ti.template(),
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        self.node_prefix[0] = 0
        for local_vertex in range(self.vertex_count):
            node = self.patch.surface_vertices[local_vertex]
            point = fem_position[node]
            inside = True
            for component in ti.static(range(3)):
                inside = inside and 0.0 <= point[component] <= domain[component]
            count = 0
            if inside:
                index = ti.cast(ti.floor(point * self.inverse_grid_size), ti.i32)
                index = ti.max(0, ti.min(index, cell_number - 1))
                cell = linearize3d(index[0], index[1], index[2], cell_number)
                for cursor in range(self.cell_offsets[cell], self.cell_offsets[cell + 1]):
                    rigid_id = self.cell_rigid[cursor]
                    if self.aabbs.aabbs[rigid_id].inside(point, 0.0):
                        count += 1
            self.node_prefix[local_vertex + 1] = count

    @ti.kernel
    def _fill_candidates(
        self,
        fem_position: ti.template(),
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        for local_vertex in range(self.vertex_count):
            node = self.patch.surface_vertices[local_vertex]
            point = fem_position[node]
            inside = True
            for component in ti.static(range(3)):
                inside = inside and 0.0 <= point[component] <= domain[component]
            if inside:
                index = ti.cast(ti.floor(point * self.inverse_grid_size), ti.i32)
                index = ti.max(0, ti.min(index, cell_number - 1))
                cell = linearize3d(index[0], index[1], index[2], cell_number)
                output = self.node_prefix[local_vertex]
                for cursor in range(self.cell_offsets[cell], self.cell_offsets[cell + 1]):
                    rigid_id = self.cell_rigid[cursor]
                    if self.aabbs.aabbs[rigid_id].inside(point, 0.0):
                        self.candidate_node[output] = node
                        self.candidate_rigid[output] = rigid_id
                        output += 1

    def rebuild(self, fem_position, rigid, box, skin=0.0):
        self.aabbs.set_levelset_body_aabbs(self.rigid_count, 0, skin, rigid, box)
        self._count_memberships(self.domain, self.cell_number)
        self.cell_scan.run(self.cell_offsets)
        memberships = int(self.cell_offsets[self.cell_count])
        if memberships > self.membership_capacity:
            raise CapacityError(
                "FEM--LSDEM max_levelset_cell_pairs is too small: "
                f"need {memberships}, allocated {self.membership_capacity}"
            )
        self._fill_memberships(self.domain, self.cell_number)
        self._count_candidates(fem_position, self.domain, self.cell_number)
        count = self._candidate_count()
        self._fill_candidates(fem_position, self.domain, self.cell_number)
        return count
