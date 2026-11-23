from src.dem.neighbor.Utilities import _as_domain
from src.dem.neighbor.Utilities import _coordination_value
from src.dem.neighbor.AffineBodyNeighborBase import AffineBodyNeighborBase

import numpy as np
import taichi as ti
from src.dem.neighbor.HierarchicalLinkedCell import HierarchicalLinkedCell
from src.dem.neighbor.NeighborKernel import (
    board_search_affine_edge_edge_linked_cell_hierarchical,
    board_search_affine_vertex_face_linked_cell_hierarchical,
    initialize_affine_edge_levels,
    initialize_affine_face_levels,
    initialize_grid_information,
    insert_affine_edges_to_cell_hierarchical,
    insert_affine_faces_to_cell_hierarchical,
)
from src.dem.structs.BaseStruct import HierarchicalCell


class AffineBodyHierarchicalLinkedCell(HierarchicalLinkedCell, AffineBodyNeighborBase):
    def __init__(self, sims, vertex_num, face_num, edge_num, dhat):
        AffineBodyNeighborBase.__init__(self, sims, vertex_num, face_num, edge_num, dhat)
        self.domain = _as_domain(sims.domain)
        self.levels = max(int(getattr(sims, "hierarchical_level", 1)), 1)
        sizes = list(getattr(sims, "hierarchical_size", []))
        if len(sizes) == 0:
            sizes = [max(self.dhat, min(self.domain) / 64.0)]
        sizes = sorted(float(v) for v in sizes)
        while len(sizes) < self.levels:
            sizes.append(sizes[-1] * 2.0)
        self.hierarchical_size = np.asarray(sizes[: self.levels], dtype=np.float64)
        self.face_level = ti.field(dtype=ti.i32, shape=max(self.face_num, 1))
        self.edge_level = ti.field(dtype=ti.i32, shape=max(self.edge_num, 1))
        self.grid = HierarchicalCell.field(shape=self.levels)
        self._initialize_grid_storage()

    def _wall_per_cell_values(self):
        base = getattr(self.sims, "wall_per_cell", 4)
        if isinstance(base, (list, tuple, np.ndarray)):
            values = [int(max(v, 1)) for v in base]
        else:
            values = [int(max(base, 1)) for _ in range(self.levels)]
        coord = _coordination_value(getattr(self.sims, "body_coordination_number", 16), 16)
        values = [max(v, 32, 2 * coord) for v in values]
        while len(values) < self.levels:
            values.append(values[-1])
        return values[: self.levels]

    def _initialize_grid_storage(self):
        gsize = []
        cnum = []
        csum = []
        factor = []
        for level in range(self.levels):
            grid_size = max(2.0 * (self.hierarchical_size[level] + self.dhat), 1.0e-5)
            igrid = 1.0 / grid_size
            cells = [max(int(self.domain[i] * igrid), 1) for i in range(3)]
            gsize.append(grid_size)
            cnum.append(cells)
            csum.append(int(cells[0] * cells[1] * cells[2]))
            factor.append(0.5 + self.hierarchical_size[level] / grid_size)
        self.gsize_np = np.asarray(gsize, dtype=np.float64)
        self.cnum_np = np.asarray(cnum, dtype=np.int32)
        self.csum_np = np.asarray(csum, dtype=np.int32)
        self.factor_np = np.asarray(factor, dtype=np.float64)
        self.cell_sum = int(sum(csum))
        wall_per_cell = self._wall_per_cell_values()
        self.wall_per_cell_np = np.asarray(wall_per_cell, dtype=np.int32)
        self.wall_in_cell = int(
            initialize_grid_information(
                self.levels,
                self.gsize_np,
                self.cnum_np,
                self.csum_np,
                self.factor_np,
                self.wall_per_cell_np,
                self.grid,
            )
        )
        self.cell_count = ti.field(dtype=ti.i32, shape=max(self.cell_sum, 1))
        self.cell_face = ti.field(dtype=ti.i32, shape=max(self.wall_in_cell, 1))
        self.edge_cell_count = ti.field(dtype=ti.i32, shape=max(self.cell_sum, 1))
        self.cell_edge = ti.field(dtype=ti.i32, shape=max(self.wall_in_cell, 1))
        self.cell_overflow = ti.field(dtype=ti.i32, shape=())
        self.edge_cell_overflow = ti.field(dtype=ti.i32, shape=())

    def _grow_cells(self):
        self.wall_per_cell_np *= 2
        self.wall_in_cell = int(
            initialize_grid_information(
                self.levels,
                self.gsize_np,
                self.cnum_np,
                self.csum_np,
                self.factor_np,
                self.wall_per_cell_np,
                self.grid,
            )
        )
        self.cell_face = ti.field(dtype=ti.i32, shape=max(self.wall_in_cell, 1))
        self.cell_edge = ti.field(dtype=ti.i32, shape=max(self.wall_in_cell, 1))

    def update(self, x, dx, faces, edges, node2body, face2body, edge2body, margin, swept=False):
        self._prepare_bounds(x, dx, faces, edges, margin, swept)
        initialize_affine_face_levels(
            self.face_num, self.levels, self.hierarchical_size, self.face_radius, self.face_level
        )
        initialize_affine_edge_levels(
            self.edge_num, self.levels, self.hierarchical_size, self.edge_radius, self.edge_level
        )
        while True:
            insert_affine_faces_to_cell_hierarchical(
                self.face_num,
                self.face_min,
                self.face_max,
                self.face_level,
                self.cell_count,
                self.cell_face,
                self.cell_overflow,
                self.grid,
            )
            insert_affine_edges_to_cell_hierarchical(
                self.edge_num,
                self.edge_min,
                self.edge_max,
                self.edge_level,
                self.edge_cell_count,
                self.cell_edge,
                self.edge_cell_overflow,
                self.grid,
            )
            board_search_affine_vertex_face_linked_cell_hierarchical(
                self.vertex_num,
                self.levels,
                self.candidate_capacity,
                self.cell_count,
                self.cell_face,
                self.vertex_min,
                self.vertex_max,
                self.face_min,
                self.face_max,
                node2body,
                face2body,
                self.candidate_vertex,
                self.candidate_face,
                self.candidate_count,
                self.candidate_overflow,
                self.grid,
            )
            board_search_affine_edge_edge_linked_cell_hierarchical(
                self.edge_num,
                self.levels,
                self.edge_candidate_capacity,
                self.edge_cell_count,
                self.cell_edge,
                self.edge_min,
                self.edge_max,
                edge2body,
                self.candidate_edge0,
                self.candidate_edge1,
                self.edge_candidate_count,
                self.edge_candidate_overflow,
                self.grid,
            )
            count = self._finish_update()
            self.last_cell_overflow = max(int(self.cell_overflow[None]), int(self.edge_cell_overflow[None]))
            if self.last_cell_overflow:
                self._grow_cells()
                continue
            if self.last_candidate_overflow:
                self._require_candidate_capacity(self.last_vertex_face_candidate_pairs)
            if self.last_edge_candidate_overflow:
                self._require_edge_candidate_capacity(self.last_edge_edge_candidate_pairs)
            return count
